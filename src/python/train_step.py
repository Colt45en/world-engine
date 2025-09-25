"""
World Engine Training Step - Complete Training Infrastructure

Advanced training utilities for the World Engine neural architecture.
Includes multi-task learning, gradient accumulation, mixed precision training,
and comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_step(model, batch, optimizer, w_rec=1.0, w_roles=1.0, w_contrastive=0.0,
               use_amp=False, grad_scaler=None, max_grad_norm=1.0):
    """
    Advanced training step with multiple loss components and mixed precision support.

    Args:
        model: WorldEngine model
        batch: Training batch dictionary
        optimizer: PyTorch optimizer
        w_rec: Weight for reconstruction loss
        w_roles: Weight for role classification loss
        w_contrastive: Weight for contrastive loss
        use_amp: Whether to use automatic mixed precision
        grad_scaler: GradScaler for AMP
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Dictionary with loss components and training metrics
    """
    model.train()

    if use_amp and grad_scaler is None:
        grad_scaler = GradScaler()

    # Forward pass with optional mixed precision
    if use_amp:
        with autocast():
            outputs = model(
                batch["tok_ids"],
                batch["pos_ids"],
                batch["feat_rows"],
                batch["lengths"],
                batch.get("edge_index"),
                batch.get("edge_type"),
                return_attention=True
            )

            # Compute losses
            loss_components = _compute_losses(model, outputs, batch, w_rec, w_roles, w_contrastive)
            total_loss = loss_components["total_loss"]
    else:
        outputs = model(
            batch["tok_ids"],
            batch["pos_ids"],
            batch["feat_rows"],
            batch["lengths"],
            batch.get("edge_index"),
            batch.get("edge_type"),
            return_attention=True
        )

        # Compute losses
        loss_components = _compute_losses(model, outputs, batch, w_rec, w_roles, w_contrastive)
        total_loss = loss_components["total_loss"]

    # Backward pass
    optimizer.zero_grad()

    if use_amp:
        grad_scaler.scale(total_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    # Compute training metrics
    metrics = _compute_training_metrics(outputs, batch, loss_components)

    return {
        **loss_components,
        **metrics,
        "learning_rate": optimizer.param_groups[0]['lr']
    }


def _compute_losses(model, outputs, batch, w_rec, w_roles, w_contrastive):
    """Compute all loss components."""
    losses = {}
    total_loss = 0.0

    # Reconstruction loss
    if w_rec > 0:
        loss_rec = model.loss_reconstruction(outputs["feat_hat"], batch["feat_rows"], outputs["mask"])
        losses["reconstruction_loss"] = loss_rec.item()
        total_loss += w_rec * loss_rec

    # Role classification loss
    if w_roles > 0 and "role_labels" in batch:
        loss_roles = model.loss_roles(outputs["role_logits"], batch["role_labels"], outputs["mask"])
        losses["role_loss"] = loss_roles.item()
        total_loss += w_roles * loss_roles

    # Contrastive loss for semantic similarity
    if w_contrastive > 0 and "positive_pairs" in batch and "negative_pairs" in batch:
        loss_contrastive = model.loss_contrastive(
            outputs["z"],
            batch["positive_pairs"],
            batch["negative_pairs"]
        )
        losses["contrastive_loss"] = loss_contrastive.item()
        total_loss += w_contrastive * loss_contrastive

    losses["total_loss"] = total_loss
    return losses


def _compute_training_metrics(outputs, batch, loss_components):
    """Compute training metrics for monitoring."""
    metrics = {}

    # Role classification accuracy
    if "role_labels" in batch:
        role_preds = outputs["role_logits"].argmax(dim=-1)
        mask = outputs["mask"]
        correct = (role_preds == batch["role_labels"]) & mask
        total_tokens = mask.sum().item()

        if total_tokens > 0:
            metrics["role_accuracy"] = (correct.sum().item() / total_tokens) * 100
        else:
            metrics["role_accuracy"] = 0.0

    # Feature reconstruction quality
    if "feat_hat" in outputs:
        # Compute correlation between predicted and target features
        feat_hat = outputs["feat_hat"]
        mask = outputs["mask"]

        # Sentence-level targets
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        feat_target = (batch["feat_rows"] * mask.unsqueeze(-1)).sum(dim=1) / denom

        # Pearson correlation
        correlation = torch.corrcoef(torch.stack([
            feat_hat.flatten(),
            feat_target.flatten()
        ]))[0, 1]

        if not torch.isnan(correlation):
            metrics["feature_correlation"] = correlation.item()

    # Latent space statistics
    if "z" in outputs:
        z = outputs["z"]
        metrics["latent_mean"] = z.mean().item()
        metrics["latent_std"] = z.std().item()
        metrics["latent_norm"] = z.norm(dim=-1).mean().item()

    # Attention entropy (measure of attention focus)
    if "attention_weights" in outputs:
        attn_weights = outputs["attention_weights"]  # [B, H, N, N]
        # Compute entropy for each head and average
        attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)
        metrics["attention_entropy"] = attn_entropy.mean().item()

    return metrics


class AdvancedTrainer:
    """Advanced trainer with learning rate scheduling, early stopping, and model checkpointing."""

    def __init__(self, model, optimizer, scheduler=None, device='cuda',
                 patience=10, min_delta=1e-4, checkpoint_dir='./checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.global_step = 0

        # History tracking
        self.train_history = []
        self.val_history = []

        # Mixed precision
        self.use_amp = device == 'cuda'
        self.grad_scaler = GradScaler() if self.use_amp else None

        logger.info(f"Initialized AdvancedTrainer with device: {device}, AMP: {self.use_amp}")

    def train_epoch(self, train_loader, val_loader=None,
                   w_rec=1.0, w_roles=1.0, w_contrastive=0.0,
                   accumulation_steps=1, log_interval=100):
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'role_loss': 0.0,
            'role_accuracy': 0.0,
            'samples_processed': 0
        }

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                    for k, v in batch.items()}

            # Training step
            step_metrics = train_step(
                self.model, batch, self.optimizer,
                w_rec=w_rec, w_roles=w_roles, w_contrastive=w_contrastive,
                use_amp=self.use_amp, grad_scaler=self.grad_scaler
            )

            # Accumulate metrics
            batch_size = batch['tok_ids'].size(0)
            epoch_metrics['samples_processed'] += batch_size

            for key in ['total_loss', 'reconstruction_loss', 'role_loss']:
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key] * batch_size

            if 'role_accuracy' in step_metrics:
                epoch_metrics['role_accuracy'] += step_metrics['role_accuracy'] * batch_size

            self.global_step += 1

            # Logging
            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {step_metrics.get('total_loss', 0):.4f}, "
                    f"LR: {step_metrics.get('learning_rate', 0):.6f}, "
                    f"Time: {elapsed:.2f}s"
                )

        # Average metrics
        num_samples = epoch_metrics['samples_processed']
        for key in epoch_metrics:
            if key != 'samples_processed':
                epoch_metrics[key] /= num_samples

        # Validation
        if val_loader is not None:
            val_metrics = self.validate(val_loader, w_rec, w_roles, w_contrastive)
            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        # Learning rate scheduling
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = epoch_metrics.get('val_total_loss', epoch_metrics['total_loss'])
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        # Early stopping check
        val_loss = epoch_metrics.get('val_total_loss', epoch_metrics['total_loss'])
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(is_best=True)
        else:
            self.patience_counter += 1

        # Record history
        self.train_history.append({
            'epoch': self.epoch,
            'metrics': {k: v for k, v in epoch_metrics.items() if not k.startswith('val_')}
        })

        if 'val_total_loss' in epoch_metrics:
            self.val_history.append({
                'epoch': self.epoch,
                'metrics': {k[4:]: v for k, v in epoch_metrics.items() if k.startswith('val_')}
            })

        self.epoch += 1

        return epoch_metrics

    def validate(self, val_loader, w_rec=1.0, w_roles=1.0, w_contrastive=0.0):
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'role_loss': 0.0,
            'role_accuracy': 0.0,
            'samples_processed': 0
        }

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                        for k, v in batch.items()}

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            batch["tok_ids"], batch["pos_ids"], batch["feat_rows"],
                            batch["lengths"], batch.get("edge_index"), batch.get("edge_type")
                        )
                        loss_components = _compute_losses(
                            self.model, outputs, batch, w_rec, w_roles, w_contrastive
                        )
                else:
                    outputs = self.model(
                        batch["tok_ids"], batch["pos_ids"], batch["feat_rows"],
                        batch["lengths"], batch.get("edge_index"), batch.get("edge_type")
                    )
                    loss_components = _compute_losses(
                        self.model, outputs, batch, w_rec, w_roles, w_contrastive
                    )

                # Compute metrics
                step_metrics = _compute_training_metrics(outputs, batch, loss_components)
                step_metrics.update(loss_components)

                # Accumulate
                batch_size = batch['tok_ids'].size(0)
                val_metrics['samples_processed'] += batch_size

                for key in ['total_loss', 'reconstruction_loss', 'role_loss']:
                    if key in step_metrics:
                        val_metrics[key] += step_metrics[key] * batch_size

                if 'role_accuracy' in step_metrics:
                    val_metrics['role_accuracy'] += step_metrics['role_accuracy'] * batch_size

        # Average metrics
        num_samples = val_metrics['samples_processed']
        for key in val_metrics:
            if key != 'samples_processed':
                val_metrics[key] /= num_samples

        return val_metrics

    def train(self, train_loader, val_loader=None, num_epochs=100,
             w_rec=1.0, w_roles=1.0, w_contrastive=0.0):
        """Full training loop with early stopping."""
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(
                train_loader, val_loader,
                w_rec=w_rec, w_roles=w_roles, w_contrastive=w_contrastive
            )

            # Log epoch results
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Train Loss: {epoch_metrics['total_loss']:.4f}")
            if 'val_total_loss' in epoch_metrics:
                logger.info(f"  Val Loss: {epoch_metrics['val_total_loss']:.4f}")
            if 'role_accuracy' in epoch_metrics:
                logger.info(f"  Role Accuracy: {epoch_metrics['role_accuracy']:.2f}%")

            # Early stopping check
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info("Training completed!")
        return self.train_history, self.val_history

    def save_checkpoint(self, filepath=None, is_best=False):
        """Save model checkpoint."""
        if filepath is None:
            filepath = f"{self.checkpoint_dir}/checkpoint_epoch_{self.epoch}.pth"

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        torch.save(checkpoint, filepath)

        if is_best:
            best_filepath = f"{self.checkpoint_dir}/best_model.pth"
            torch.save(checkpoint, best_filepath)
            logger.info(f"Best model saved to {best_filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        logger.info(f"Checkpoint loaded from {filepath}")


def create_trainer_config(model_config):
    """Create default trainer configuration."""
    return {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'scheduler_type': 'cosine',
        'warmup_epochs': 5,
        'max_epochs': 100,
        'patience': 10,
        'grad_clip_norm': 1.0,
        'loss_weights': {
            'reconstruction': 1.0,
            'roles': 1.0,
            'contrastive': 0.1
        }
    }


def validate_batch(batch):
    """Validate batch format and contents."""
    required_keys = ['tok_ids', 'pos_ids', 'feat_rows', 'lengths']
    for key in required_keys:
        if key not in batch:
            raise ValueError(f"Missing required key in batch: {key}")

    # Check tensor shapes
    B, N = batch['tok_ids'].shape
    assert batch['pos_ids'].shape == (B, N), "pos_ids shape mismatch"
    assert batch['feat_rows'].shape[:2] == (B, N), "feat_rows shape mismatch"
    assert batch['lengths'].shape == (B,), "lengths shape mismatch"

    # Check value ranges
    assert (batch['lengths'] <= N).all(), "Invalid sequence lengths"
    assert (batch['tok_ids'] >= 0).all(), "Invalid token IDs"
    assert (batch['pos_ids'] >= 0).all(), "Invalid POS IDs"

    return True

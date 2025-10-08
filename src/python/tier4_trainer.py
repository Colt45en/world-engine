"""
Tier-4 Trainer
==============

Standalone training harness exposing ``Tier4Trainer`` and ``Tier4Config``.
See ``train_step.py`` for backward compatibility.
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F  # noqa: F401  # kept for compatibility with potential custom losses
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - tensorboard optional
    SummaryWriter = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or torch.distributed.get_rank() == 0


def dist_avg(x: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across ranks (safe when not distributed)."""
    if not is_dist():
        return x
    xt = x.clone()
    torch.distributed.all_reduce(xt, op=torch.distributed.ReduceOp.SUM)
    xt /= torch.distributed.get_world_size()
    return xt


def move_batch_to_device(batch: Dict, device: torch.device):
    return {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Tier4Config:
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 0
    grad_clip_norm: float = 1.0
    accumulation_steps: int = 1  # TRUE grad accumulation (fixed)

    # Mixed precision
    amp: bool = True

    # Extras
    compile_model: bool = False  # torch.compile if PyTorch 2.x
    use_ema: bool = True
    ema_decay: float = 0.999

    use_swa: bool = False
    swa_start_epoch: int = 10
    swa_lr: float = 5e-5
    swa_anneal_epochs: int = 5

    log_dir: str = "./runs/tier4"
    ckpt_dir: str = "./checkpoints"
    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    # Loss weights
    w_rec: float = 1.0
    w_roles: float = 1.0
    w_contrastive: float = 0.0


# ────────────────────────────────────────────────────────────────────────────────
# EMA helper (simple, fast)
# ────────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ────────────────────────────────────────────────────────────────────────────────
# Losses & metrics (compatible with your WorldEngine API)
# ────────────────────────────────────────────────────────────────────────────────
def _compute_losses(model, outputs, batch, w_rec, w_roles, w_contrastive):
    """Returns (total_loss, dict_of_float_metrics)."""
    losses = {}
    total_loss = torch.zeros([], device=outputs["z"].device)

    # Reconstruction
    if w_rec > 0:
        l = model.loss_reconstruction(outputs["feat_hat"], batch["feat_rows"], outputs["mask"])
        losses["reconstruction_loss"] = l.detach()
        total_loss = total_loss + w_rec * l

    # Roles
    if w_roles > 0 and "role_labels" in batch:
        l = model.loss_roles(outputs["role_logits"], batch["role_labels"], outputs["mask"])
        losses["role_loss"] = l.detach()
        total_loss = total_loss + w_roles * l

    # Contrastive
    if w_contrastive > 0 and "positive_pairs" in batch and "negative_pairs" in batch:
        l = model.loss_contrastive(outputs["z"], batch["positive_pairs"], batch["negative_pairs"])
        losses["contrastive_loss"] = l.detach()
        total_loss = total_loss + w_contrastive * l

    losses["total_loss"] = total_loss.detach()
    return total_loss, {k: float(v.item()) for k, v in losses.items()}


def _compute_metrics(outputs, batch):
    m = {}
    mask = outputs.get("mask")
    # Role accuracy
    if "role_labels" in batch and "role_logits" in outputs and mask is not None:
        preds = outputs["role_logits"].argmax(dim=-1)
        correct = ((preds == batch["role_labels"]) & mask).sum()
        total = mask.sum().clamp_min(1)
        m["role_accuracy"] = float((correct / total).item() * 100.0)

    # Feature correlation (Pearson)
    if "feat_hat" in outputs and mask is not None:
        feat_hat = outputs["feat_hat"]
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        feat_tgt = (batch["feat_rows"] * mask.unsqueeze(-1)).sum(dim=1) / denom
        x = feat_hat.flatten()
        y = feat_tgt.flatten()
        x = x - x.mean()
        y = y - y.mean()
        num = (x * y).sum()
        den = (x.norm() * y.norm()).clamp_min(1e-9)
        m["feature_correlation"] = float((num / den).item())

    # z stats
    if "z" in outputs:
        z = outputs["z"]
        m["latent_mean"] = float(z.mean().item())
        m["latent_std"] = float(z.std().item())
        m["latent_norm"] = float(z.norm(dim=-1).mean().item())

    # Attention entropy
    aw = outputs.get("attention_weights")
    if aw is not None:
        p = aw.clamp_min(1e-9)
        ent = (-p * p.log()).sum(dim=-1).mean()
        m["attention_entropy"] = float(ent.item())

    return m


# ────────────────────────────────────────────────────────────────────────────────
# Tier-4 Trainer
# ────────────────────────────────────────────────────────────────────────────────
class Tier4Trainer:
    def __init__(self, model, optimizer, scheduler=None, device="cuda", cfg: Tier4Config = Tier4Config()):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optional compile (PyTorch 2+)
        if cfg.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            logger.info("Model compiled with torch.compile")

        # DDP wrap (if initialized outside)
        if is_dist():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=None if self.device.type != "cuda" else [torch.cuda.current_device()],
                find_unused_parameters=False,
            )
            logger.info("Model wrapped in DistributedDataParallel")

        # AMP
        self.use_amp = (self.device.type == "cuda") and cfg.amp
        self.scaler = GradScaler(enabled=self.use_amp)

        # EMA
        self.ema = EMA(self.unwrap(self.model), decay=cfg.ema_decay) if cfg.use_ema else None

        # SWA
        self.swa_model = AveragedModel(self.unwrap(self.model)) if cfg.use_swa else None
        self.swa_scheduler = None

        # TB writer
        self.writer = SummaryWriter(cfg.log_dir) if (rank0() and SummaryWriter is not None) else None
        if self.writer is None and rank0() and SummaryWriter is None:
            logger.warning("TensorBoard not available; install tensorboard to enable logging.")

        # Bookkeeping
        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")
        self.no_improve = 0
        self.global_step = 0
        self.epoch = 0

        if rank0():
            logger.info(f"Tier4Trainer init: {json.dumps(asdict(cfg), indent=2)}")

    @staticmethod
    def unwrap(m):
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    # ── Tier-4 Operators (ST/UP/PR/CV/RB/RS) ───────────────────────────────────
    def operator(self, op: str):
        op = op.upper()
        if op == "ST":  # Stabilize: reduce LR & increase EMA decay slightly
            for g in self.optimizer.param_groups:
                g["lr"] *= 0.5
            if self.ema:
                self.ema.decay = min(0.9999, self.ema.decay * 1.02)
            if rank0():
                logger.info("OP[ST] stabilization applied (lr halved, ema decay nudged).")

        elif op == "UP":  # Update/progress: one scheduler step
            if self.scheduler is not None and not isinstance(self.scheduler, SWALR):
                self.scheduler.step()
                if rank0():
                    logger.info("OP[UP] scheduler step executed.")

        elif op == "PR":  # Progress report
            if rank0():
                logger.info(f"OP[PR] step={self.global_step}, epoch={self.epoch}")

        elif op == "CV":  # Converge: swap to SWA weights (if enabled)
            if self.cfg.use_swa and self.swa_model:
                torch.optim.swa_utils.update_bn(self.train_loader_for_bn, self.swa_model, device=self.device) if hasattr(self, "train_loader_for_bn") else None
                if rank0():
                    logger.info("OP[CV] SWA BN updated (if loader provided).")

        elif op == "RB":  # Rollback to best checkpoint
            best = Path(self.cfg.ckpt_dir) / "best.pth"
            if best.exists():
                self.load_checkpoint(best)
                if rank0():
                    logger.info("OP[RB] rolled back to best.pth")
            else:
                if rank0():
                    logger.warning("OP[RB] best.pth not found.")

        elif op == "RS":  # Reset scheduler state (if supported)
            if self.scheduler and hasattr(self.scheduler, "last_epoch"):
                self.scheduler.last_epoch = -1
                if rank0():
                    logger.info("OP[RS] scheduler last_epoch reset.")
        else:
            if rank0():
                logger.warning(f"Unknown operator: {op}")

    # ── Core loops ─────────────────────────────────────────────────────────────
    def _forward(self, batch, return_attention=True):
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        return model(
            batch["tok_ids"],
            batch["pos_ids"],
            batch["feat_rows"],
            batch["lengths"],
            batch.get("edge_index"),
            batch.get("edge_type"),
            return_attention=return_attention,
        )

    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        self.train_loader_for_bn = train_loader  # for CV op (SWA BN update), optional

        start = time.time()
        meters = {"loss": 0.0, "role_acc": 0.0, "count": 0}
        steps_in_epoch = 0

        self.optimizer.zero_grad(set_to_none=True)

        for bidx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, self.device)
            micro = self.cfg.accumulation_steps
            use_no_sync = is_dist() and ((bidx + 1) % micro != 0)

            if use_no_sync and hasattr(self.model, "no_sync"):
                cm = self.model.no_sync()
            else:
                class Dummy:
                    def __enter__(self):
                        pass

                    def __exit__(self, *a):
                        pass

                cm = Dummy()

            with cm:
                with autocast(enabled=self.use_amp):
                    outputs = self._forward(batch, return_attention=True)
                    total_loss, loss_dict = _compute_losses(
                        self.unwrap(self.model),
                        outputs,
                        batch,
                        self.cfg.w_rec,
                        self.cfg.w_roles,
                        self.cfg.w_contrastive,
                    )
                    # Average over accumulation steps to keep effective LR stable
                    loss = total_loss / micro

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Step on accumulation boundary
            if (bidx + 1) % micro == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.unwrap(self.model).parameters(), self.cfg.grad_clip_norm)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # EMA update
                if self.ema:
                    self.ema.update(self.unwrap(self.model))

                # SWA update (per step or per epoch; we do per-step for smoother averaging)
                if self.cfg.use_swa and self.swa_model:
                    self.swa_model.update_parameters(self.unwrap(self.model))

                # Scheduler (step-wise schedulers)
                if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if not isinstance(self.scheduler, SWALR):
                        self.scheduler.step()

            # Metrics (batch)
            with torch.no_grad():
                met = _compute_metrics(outputs, batch)
                meters["loss"] += float(loss_dict["total_loss"]) * batch["tok_ids"].size(0)
                if "role_accuracy" in met:
                    meters["role_acc"] += met["role_accuracy"] * batch["tok_ids"].size(0)
                meters["count"] += batch["tok_ids"].size(0)

                # TB logging (step)
                if rank0() and self.writer:
                    self.writer.add_scalar("train/total_loss", float(loss_dict["total_loss"]), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                    for k, v in met.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)

            self.global_step += 1
            steps_in_epoch += 1

        # SWA LR scheduling (epoch-wise)
        if self.cfg.use_swa and self.swa_model:
            if self.epoch == self.cfg.swa_start_epoch and self.scheduler:
                # swap to SWA scheduler
                self.swa_scheduler = SWALR(
                    self.optimizer,
                    swa_lr=self.cfg.swa_lr,
                    anneal_epochs=self.cfg.swa_anneal_epochs,
                )
                self.scheduler = self.swa_scheduler
            if isinstance(self.scheduler, SWALR):
                self.scheduler.step()

        # Reduce across ranks
        if is_dist():
            for k in ["loss", "role_acc", "count"]:
                t = torch.tensor([meters[k]], device=self.device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                meters[k] = float(t.item())

        epoch_loss = meters["loss"] / max(1, meters["count"])
        epoch_acc = meters["role_acc"] / max(1, meters["count"])

        if rank0():
            took = time.time() - start
            logger.info(f"[Epoch {self.epoch}] train loss={epoch_loss:.4f} acc={epoch_acc:.2f}% ({took:.1f}s)")
            if self.writer:
                self.writer.add_scalar("epoch/train_loss", epoch_loss, self.epoch)
                self.writer.add_scalar("epoch/train_role_acc", epoch_acc, self.epoch)

        # Return for external logs
        return {"train_loss": epoch_loss, "train_role_acc": epoch_acc}

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        meters = {"loss": 0.0, "role_acc": 0.0, "count": 0}

        for batch in val_loader:
            batch = move_batch_to_device(batch, self.device)
            with autocast(enabled=self.use_amp):
                outputs = self._forward(batch, return_attention=False)
                total_loss, loss_dict = _compute_losses(
                    self.unwrap(self.model),
                    outputs,
                    batch,
                    self.cfg.w_rec,
                    self.cfg.w_roles,
                    self.cfg.w_contrastive,
                )
            met = _compute_metrics(outputs, batch)

            meters["loss"] += float(loss_dict["total_loss"]) * batch["tok_ids"].size(0)
            if "role_accuracy" in met:
                meters["role_acc"] += met["role_accuracy"] * batch["tok_ids"].size(0)
            meters["count"] += batch["tok_ids"].size(0)

        # Reduce across ranks
        if is_dist():
            for k in ["loss", "role_acc", "count"]:
                t = torch.tensor([meters[k]], device=self.device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                meters[k] = float(t.item())

        val_loss = meters["loss"] / max(1, meters["count"])
        val_acc = meters["role_acc"] / max(1, meters["count"])

        if rank0() and self.writer:
            self.writer.add_scalar("epoch/val_loss", val_loss, self.epoch)
            self.writer.add_scalar("epoch/val_role_acc", val_acc, self.epoch)

        if rank0():
            logger.info(f"[Epoch {self.epoch}]   val loss={val_loss:.4f} acc={val_acc:.2f}%")

        return {"val_loss": val_loss, "val_role_acc": val_acc}

    def fit(self, train_loader, val_loader=None):
        best = float("inf")
        for _ in range(self.cfg.max_epochs):
            train_stats = self.train_epoch(train_loader, val_loader)
            val_stats = self.validate(val_loader) if val_loader is not None else {"val_loss": train_stats["train_loss"]}
            val_loss = val_stats["val_loss"]

            # Plateau scheduler
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            # Early stopping + checkpointing (rank 0 only)
            if rank0():
                self.save_checkpoint(Path(self.cfg.ckpt_dir) / f"epoch_{self.epoch}.pth")
                if val_loss + self.cfg.min_delta < best:
                    best = val_loss
                    self.no_improve = 0
                    self.save_checkpoint(Path(self.cfg.ckpt_dir) / "best.pth", is_best=True)
                else:
                    self.no_improve += 1

                if self.no_improve >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

            self.epoch += 1

        # Final SWA: swap weights and (optionally) update BN using train loader
        if self.cfg.use_swa and self.swa_model:
            if rank0():
                logger.info("Applying final SWA weights.")
            self.unwrap(self.model).load_state_dict(self.swa_model.module.state_dict())

    # ── Checkpoints ────────────────────────────────────────────────────────────
    def save_checkpoint(self, path: Path, is_best=False):
        if not rank0():
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        to_save = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.unwrap(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "cfg": asdict(self.cfg),
            "best_val": self.best_val,
        }
        if self.ema:
            to_save["ema"] = self.ema.shadow
        if self.cfg.use_swa and self.swa_model:
            to_save["swa"] = self.swa_model.module.state_dict()

        torch.save(to_save, str(path))
        if is_best:
            logger.info(f"Saved BEST checkpoint -> {path}")
        else:
            logger.info(f"Saved checkpoint -> {path}")

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(str(path), map_location=self.device)
        self.unwrap(self.model).load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"] and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt and self.ema:
            # Restore EMA shadow
            self.ema.shadow = ckpt["ema"]
        if "swa" in ckpt and self.cfg.use_swa and self.swa_model:
            self.swa_model.module.load_state_dict(ckpt["swa"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        if rank0():
            logger.info(f"Loaded checkpoint from {path}")


__all__ = [
    "Tier4Config",
    "Tier4Trainer",
    "is_dist",
    "rank0",
    "dist_avg",
    "move_batch_to_device",
]

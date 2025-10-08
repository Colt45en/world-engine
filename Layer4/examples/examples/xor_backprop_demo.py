"""
XOR Backpropagation Demo - World Engine Integration
==================================================

This module demonstrates the fundamental backpropagation principles that power
the World Engine V3.1 architecture using the classic XOR problem as a foundation.

This implementation shows how the same principles scale from simple XOR learning
to the sophisticated multi-modal architectures in World Engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import time


class XORNetwork(nn.Module):
    """
    Classic XOR learning network that demonstrates fundamental backprop principles.

    This mirrors the architectural patterns used in World Engine but at a smaller scale.
    """

    def __init__(self, hidden_size: int = 8, use_advanced_features: bool = True):
        super().__init__()

        # Core architecture (similar to World Engine components)
        self.hidden = nn.Linear(2, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        # Advanced features inspired by World Engine V3.1
        if use_advanced_features:
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.1)
        else:
            self.layer_norm = nn.Identity()
            self.dropout = nn.Identity()

        self.use_advanced_features = use_advanced_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional advanced features."""
        # Hidden representation learning (core of backprop)
        hidden_output = self.hidden(x)

        # Advanced processing (World Engine style)
        if self.use_advanced_features:
            hidden_output = self.layer_norm(hidden_output)

        hidden_output = torch.tanh(hidden_output)  # Non-linearity for representation
        hidden_output = self.dropout(hidden_output)

        # Output prediction
        output = torch.sigmoid(self.output(hidden_output))

        return output

    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned hidden representations (like World Engine feature extraction)."""
        with torch.no_grad():
            hidden_output = self.hidden(x)
            if self.use_advanced_features:
                hidden_output = self.layer_norm(hidden_output)
            hidden_output = torch.tanh(hidden_output)

        return hidden_output


class WorldEngineStyleTrainer:
    """
    Training system inspired by World Engine V3.1 training infrastructure.

    Demonstrates how the same backprop principles scale to complex architectures.
    """

    def __init__(self, model: XORNetwork, learning_rate: float = 0.01):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # World Engine uses AdamW
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.loss_fn = nn.BCELoss()

        # Training history (like World Engine monitoring)
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'representations': []
        }

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare XOR training data."""
        # XOR truth table
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        return X, y

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Single training epoch with World Engine style monitoring."""
        self.model.train()

        # Forward pass
        predictions = self.model(X)
        loss = self.loss_fn(predictions, y)

        # Backpropagation (the core mechanism)
        self.optimizer.zero_grad()
        loss.backward()  # This is where the magic happens!
        self.optimizer.step()
        self.scheduler.step()

        # Calculate accuracy
        with torch.no_grad():
            predicted_classes = (predictions > 0.5).float()
            accuracy = (predicted_classes == y).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'lr': self.scheduler.get_last_lr()[0]
        }

    def train(self, epochs: int = 1000, verbose: bool = True) -> Dict[str, List]:
        """Train the network and monitor representation learning."""
        X, y = self.prepare_data()

        for epoch in range(epochs):
            metrics = self.train_epoch(X, y)

            # Store history
            self.training_history['losses'].append(metrics['loss'])
            self.training_history['accuracies'].append(metrics['accuracy'])

            # Store representations every 100 epochs
            if epoch % 100 == 0:
                representations = self.model.get_hidden_representations(X)
                self.training_history['representations'].append(representations.clone())

                if verbose:
                    print(f"Epoch {epoch:4d}: Loss {metrics['loss']:.6f}, "
                          f"Accuracy {metrics['accuracy']:.3f}, LR {metrics['lr']:.6f}")

        return self.training_history

    def analyze_learned_representations(self) -> None:
        """Analyze what the network learned (similar to World Engine analysis tools)."""
        X, y = self.prepare_data()

        with torch.no_grad():
            # Final predictions
            predictions = self.model(X)
            representations = self.model.get_hidden_representations(X)

            print("\n=== XOR Learning Analysis ===")
            print("Input -> Target | Prediction | Hidden Representations")
            print("-" * 60)

            for i in range(len(X)):
                input_vals = X[i].numpy()
                target = y[i].item()
                pred = predictions[i].item()
                hidden = representations[i].numpy()

                print(f"({input_vals[0]:.0f},{input_vals[1]:.0f}) -> {target:.0f}     | "
                      f"{pred:.3f}      | {hidden}")

            print(f"\nFinal Accuracy: {((predictions > 0.5).float() == y).float().mean():.3f}")

    def demonstrate_representation_evolution(self) -> None:
        """Show how representations evolve during training (World Engine style analysis)."""
        if not self.training_history['representations']:
            print("No representation history available. Train with monitoring enabled.")
            return

        print("\n=== Representation Evolution ===")
        X, _ = self.prepare_data()

        for i, representations in enumerate(self.training_history['representations']):
            epoch = i * 100
            print(f"\nEpoch {epoch}:")
            print("Input | Hidden Representations")
            print("-" * 40)

            for j in range(len(X)):
                input_vals = X[j].numpy()
                hidden = representations[j].numpy()
                print(f"({input_vals[0]:.0f},{input_vals[1]:.0f}) | {hidden}")


def demonstrate_scaling_to_world_engine():
    """
    Show how XOR principles scale to World Engine complexity.

    This demonstrates the conceptual bridge between simple backprop
    and the sophisticated architectures in World Engine V3.1.
    """
    print("=" * 60)
    print("SCALING FROM XOR TO WORLD ENGINE V3.1")
    print("=" * 60)

    print("\n1. Basic XOR Network (Classic Backprop)")
    basic_model = XORNetwork(hidden_size=4, use_advanced_features=False)
    basic_trainer = WorldEngineStyleTrainer(basic_model)
    basic_trainer.train(epochs=500, verbose=False)
    basic_trainer.analyze_learned_representations()

    print("\n" + "="*60)
    print("\n2. Advanced XOR Network (World Engine Style)")
    advanced_model = XORNetwork(hidden_size=16, use_advanced_features=True)
    advanced_trainer = WorldEngineStyleTrainer(advanced_model, learning_rate=0.001)
    advanced_trainer.train(epochs=500, verbose=False)
    advanced_trainer.analyze_learned_representations()

    print("\n" + "="*60)
    print("\n3. Scaling Principles to World Engine V3.1:")
    print("   • XOR: 2 inputs → 4-16 hidden units → 1 output")
    print("   • World Engine: ~thousands of features → multi-scale processing → complex outputs")
    print("   • Same backprop principles, different scale!")
    print("   • Representation learning: XOR learns AND/OR, World Engine learns semantic concepts")
    print("   • Both discover hidden structure that makes complex problems solvable")


if __name__ == "__main__":
    print("World Engine V3.1 - Backpropagation Fundamentals Demo")
    print("This demonstrates the core principles that power the advanced neural architecture.")
    print()

    # Run the scaling demonstration
    demonstrate_scaling_to_world_engine()

    print("\n" + "="*60)
    print("\nKey Insights:")
    print("• Backprop enables representation learning at any scale")
    print("• Hidden layers discover useful internal representations")
    print("• Same principles work from XOR to multi-modal World Engine")
    print("• Complex behaviors emerge from simple learning rules")
    print("• World Engine V3.1 applies these principles to linguistic/semantic domains")

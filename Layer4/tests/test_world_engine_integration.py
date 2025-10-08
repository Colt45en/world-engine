#!/usr/bin/env python3

"""
Simple test script to verify World Engine core functionality.
"""

import torch
import torch.nn as nn

# Simple MultiScaleProcessor for testing
class SimpleMultiScaleProcessor(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Simple multi-scale convolutions
        self.conv1 = nn.Conv1d(d_model, d_model//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model//4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(d_model, d_model//4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(d_model, d_model//4, kernel_size=9, padding=4)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_scale_outputs=False):
        B, N, D = x.shape

        # Transpose for conv1d: [B, D, N]
        x_t = x.transpose(1, 2)

        # Apply multiple scales
        out1 = self.conv1(x_t).transpose(1, 2)  # [B, N, D//4]
        out2 = self.conv2(x_t).transpose(1, 2)  # [B, N, D//4]
        out3 = self.conv3(x_t).transpose(1, 2)  # [B, N, D//4]
        out4 = self.conv4(x_t).transpose(1, 2)  # [B, N, D//4]

        # Concatenate
        output = torch.cat([out1, out2, out3, out4], dim=-1)  # [B, N, D]
        output = self.norm(output)
        output = self.dropout(output)

        if mask is not None:
            output = output * mask.unsqueeze(-1)

        if return_scale_outputs:
            return output, [out1, out2, out3, out4]
        return output

def test_world_engine():
    """Test basic World Engine functionality."""
    try:
        # Import with potential fallback
        try:
            from world_engine import WorldEngine, create_world_engine
        except ImportError as e:
            print(f"Import error: {e}")
            print("World Engine may have import issues, attempting basic test...")
            return False

        # Test configuration
        config = {
            'vocab_size': 500,
            'd_model': 64,
            'k_feats': 20,
            'n_pos': 10,
            'n_rels': 5,
            'n_layers': 2,
            'n_heads': 4,
            'dropout': 0.1,
            'use_transformer': True,
            'use_gnn': False,  # Disable GNN for simple test
            'use_crf': False,  # Disable CRF for simple test
            'num_role_labels': 3
        }

        print("Creating World Engine...")
        model = create_world_engine(config)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Create test data
        batch_size = 2
        seq_len = 8
        device = 'cpu'

        tok_ids = torch.randint(1, config['vocab_size'], (batch_size, seq_len))
        pos_ids = torch.randint(1, config['n_pos'], (batch_size, seq_len))
        feat_rows = torch.randn(batch_size, seq_len, config['k_feats'])
        lengths = torch.tensor([seq_len, seq_len-2])

        print("Testing forward pass...")
        with torch.no_grad():
            output = model(tok_ids, pos_ids, feat_rows, lengths)
            print(f"✅ Forward pass successful")
            print(f"   z shape: {output['z'].shape}")
            print(f"   feat_hat shape: {output['feat_hat'].shape}")
            print(f"   role_logits shape: {output['role_logits'].shape}")

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running World Engine Integration Test")
    print("=" * 50)
    success = test_world_engine()
    print("=" * 50)
    if success:
        print("🎉 World Engine is working correctly!")
    else:
        print("❌ World Engine has issues that need fixing")

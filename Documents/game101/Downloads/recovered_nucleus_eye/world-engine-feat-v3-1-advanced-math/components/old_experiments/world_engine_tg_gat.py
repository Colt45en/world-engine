import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchcrf import CRF
    HAS_CRF = True
except Exception:
    HAS_CRF = False

from rel_gatv2_encoder_tg import RelGATv2EncoderTG

def sinusoidal_positions(n_pos: int, d: int, device=None):
    pe = torch.zeros(n_pos, d, device=device)
    position = torch.arange(0, n_pos, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class WorldEngineTG_GAT(nn.Module):
    """
    WE-v1 (GATv2 variant):
      - Same inputs/outputs/losses as your TG/RGCN model
      - Typed dependency attention via edge attributes
    Inputs
      tok_ids:   Long[B,N]
      pos_ids:   Long[B,N]
      feat_rows: Float[B,N,K]
      lengths:   Long[B]
      edge_index: Long[2,E]
      edge_type:  Long[E]
    """
    def __init__(self, vocab_size, d_model, k_feats, n_pos, n_rels,
                 d_tok=None, d_pos=None, d_feat=None, p_drop=0.1,
                 n_transformer_layers=2, n_heads=4,
                 gat_layers=2, gat_heads=4, rel_dim=64,
                 use_crf=False, num_role_labels=3, pe_max_len=512):
        super().__init__()
        d_tok  = d_tok  or d_model // 2
        d_posE = d_pos  or d_model // 4
        d_feat = d_feat or (d_model - d_tok - d_posE)
        assert d_tok + d_posE + d_feat == d_model

        self.emb_tok = nn.Embedding(vocab_size, d_tok, padding_idx=0)
        self.emb_pos = nn.Embedding(n_pos,     d_posE, padding_idx=0)
        self.W_feat  = nn.Linear(k_feats, d_feat, bias=False)
        self.norm_in = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(p_drop)

        # Positional encodings
        self.register_buffer("pe", sinusoidal_positions(pe_max_len, d_model), persistent=False)

        # Sequence encoder (Transformer)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=p_drop, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc_seq = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

        # GATv2 relational graph encoder
        self.gnn = RelGATv2EncoderTG(d_model, n_rels, num_layers=gat_layers, heads=gat_heads,
                                     dropout=p_drop, rel_dim=rel_dim)

        # Latent roots head + reconstruction
        self.enc_lat = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 32)
        )
        self.dec_feat = nn.Linear(32, k_feats)

        # Role tagger
        self.role_head = nn.Linear(d_model, num_role_labels)
        self.use_crf = use_crf and HAS_CRF and (num_role_labels > 1)
        if self.use_crf:
            self.crf = CRF(num_role_labels, batch_first=True)

    def extend_pe(self, N_needed):
        if N_needed <= self.pe.size(1): return
        with torch.no_grad():
            self.pe = sinusoidal_positions(N_needed, self.pe.size(-1), device=self.pe.device)

    def forward(self, tok_ids, pos_ids, feat_rows, lengths, edge_index=None, edge_type=None):
        B, N = tok_ids.shape
        device = tok_ids.device
        mask = torch.arange(N, device=device)[None, :] < lengths[:, None]  # [B,N]

        x = torch.cat([
            self.emb_tok(tok_ids),
            self.emb_pos(pos_ids),
            self.W_feat(feat_rows)
        ], dim=-1)                         # [B,N,d]
        x = self.norm_in(x)
        self.extend_pe(N)
        x = x + self.pe[:, :N, :]
        x = self.drop(x)

        # Sequence encoder with padding mask
        h = self.enc_seq(x, src_key_padding_mask=~mask)      # [B,N,d]

        # Graph attention over flattened batch graph
        h_flat = h.reshape(B*N, -1)
        valid_flat = mask.reshape(B*N)
        if edge_index is not None and edge_type is not None and edge_index.numel() > 0:
            h_flat = self.gnn(h_flat, edge_index, edge_type, valid_mask_flat=valid_flat)
        h = h_flat.view(B, N, -1)

        # Sentence-level roots (masked mean)
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        h_sent = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        z = self.enc_lat(h_sent)            # [B,32]
        feat_hat = self.dec_feat(z)         # [B,K]

        # Roles
        role_logits = self.role_head(h)     # [B,N,C]
        return {"z": z, "feat_hat": feat_hat, "role_logits": role_logits, "mask": mask}

    # ---- Losses (unchanged) ----
    def loss_reconstruction(self, feat_hat, feat_rows, mask=None, reduction="mean"):
        if mask is None:
            target = feat_rows.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            target = (feat_rows * mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.binary_cross_entropy_with_logits(feat_hat, target, reduction=reduction)

    def loss_roles(self, role_logits, role_labels, mask):
        if self.use_crf:
            ll = self.crf(role_logits, role_labels, mask=mask, reduction='mean')
            return -ll
        C = role_logits.size(-1)
        loss = F.cross_entropy(role_logits.view(-1, C), role_labels.view(-1), reduction='none')
        loss = loss.view_as(role_labels) * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1)
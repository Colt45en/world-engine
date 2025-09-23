import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchcrf import CRF  # pip install pytorch-crf for CRF support
    HAS_CRF = True
except Exception:
    HAS_CRF = False

def sinusoidal_positions(n_pos: int, d: int, device=None):
    """[1, n_pos, d] sinusoidal PE, standard transformer-style."""
    pe = torch.zeros(n_pos, d, device=device)
    position = torch.arange(0, n_pos, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, n_pos, d]

class Residual(nn.Module):
    def __init__(self, mod, d_model, p=0.1):
        super().__init__()
        self.mod = mod
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)
    def forward(self, x, *args, **kwargs):
        return self.norm(x + self.drop(self.mod(x, *args, **kwargs)))

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out)
        )
    def forward(self, x): return self.net(x)

class WorldEngine(nn.Module):
    """
    WE-v1: sentence-level 'roots' z plus token-level roles.
    Inputs
      tok_ids:  Long[B,N]       (token ids; 0 = pad)
      pos_ids:  Long[B,N]       (pos ids; 0 = pad)
      feat_rows:Float[B,N,K]    (interpretable features per token; can be zeros for pads)
      lengths:  Long[B]         (true lengths for masking)
      edge_index: Long[2, E]    (global COO over batch with sentence offsets)  (optional)
      edge_type:  Long[E]       (relation ids aligned with edge_index)        (optional)
    """
    def __init__(self, vocab_size, d_model, k_feats, n_pos, n_rels,
                 d_tok=None, d_pos=None, d_feat=None, p_drop=0.1,
                 use_transformer=True, n_layers=2, n_heads=4,
                 use_gnn=False, use_crf=False, num_role_labels=3):
        super().__init__()
        d_tok  = d_tok  or d_model // 2
        d_posE = d_pos  or d_model // 4
        d_feat = d_feat or d_model - d_tok - d_posE
        assert d_tok + d_posE + d_feat == d_model, "dims must sum to d_model"

        self.emb_tok = nn.Embedding(vocab_size, d_tok, padding_idx=0)
        self.emb_pos = nn.Embedding(n_pos,     d_posE, padding_idx=0)
        self.W_feat  = nn.Linear(k_feats, d_feat, bias=False)

        # positional encoding buffer (max length set at init; can be extended)
        self.register_buffer("pe", sinusoidal_positions(512, d_model), persistent=False)

        enc_layers = []
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                dropout=p_drop, batch_first=True, activation="gelu", norm_first=True
            )
            self.enc_seq = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.enc_seq = nn.Identity()

        # optional GNN hook: keep interface even if disabled
        self.use_gnn = use_gnn
        if use_gnn:
            # simple message-passing via attention over edges (no external deps)
            self.edge_rel_emb = nn.Embedding(n_rels, d_model)
            self.msg_proj = nn.Linear(2*d_model, d_model)
            self.msg_norm = nn.LayerNorm(d_model)

        # latent head (sentence-level roots)
        self.enc_lat = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 32)  # z
        )

        # reconstruct interpretable features (either sentence-avg or token-avg target)
        self.dec_feat = nn.Linear(32, k_feats)

        # token role tagging head
        self.role_head = nn.Linear(d_model, num_role_labels)
        self.use_crf = use_crf and HAS_CRF and (num_role_labels > 1)
        if self.use_crf:
            self.crf = CRF(num_role_labels, batch_first=True)

        self.dropout = nn.Dropout(p_drop)
        self.norm_in = nn.LayerNorm(d_model)

    def extend_pe(self, n_pos_needed):
        if n_pos_needed <= self.pe.size(1): return
        with torch.no_grad():
            self.pe = sinusoidal_positions(n_pos_needed, self.pe.size(-1), device=self.pe.device)

    def forward(self, tok_ids, pos_ids, feat_rows, lengths,
                edge_index=None, edge_type=None):
        B, N = tok_ids.shape
        mask = torch.arange(N, device=tok_ids.device)[None, :] < lengths[:, None]  # [B,N] True for valid

        # embeddings
        x = torch.cat([
            self.emb_tok(tok_ids),
            self.emb_pos(pos_ids),
            self.W_feat(feat_rows)
        ], dim=-1)  # [B,N,d_model]
        x = self.norm_in(x)
        self.extend_pe(N)
        x = x + self.pe[:, :N, :]

        # sequence encoder
        h = self.enc_seq(x, src_key_padding_mask=~mask) if isinstance(self.enc_seq, nn.TransformerEncoder) else self.enc_seq(x)

        # optional simple GNN message passing over typed edges
        if self.use_gnn and edge_index is not None and edge_type is not None and edge_index.numel() > 0:
            # edge_index: [2,E] in global (batched) space; assume caller already offset indices
            src, dst = edge_index  # [E]
            rel = edge_type
            rel_e = self.edge_rel_emb(rel)                 # [E,d]
            m = torch.cat([h.view(-1, h.size(-1))[src], rel_e], dim=-1)  # [E, 2d]
            m = torch.tanh(self.msg_proj(m))               # [E,d]
            # aggregate messages to dst (simple scatter-add)
            H = h.view(-1, h.size(-1))
            agg = torch.zeros_like(H)
            agg.index_add_(0, dst, m)
            h = self.msg_norm(H + self.dropout(agg)).view(B, N, -1)

        # sentence-level roots via masked mean
        h_masked = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        h_sent = h_masked.sum(dim=1) / denom
        z = self.enc_lat(h_sent)  # [B,32]

        # reconstruct average interpretable features as an auxiliary target
        feat_hat = self.dec_feat(z)  # [B,K]

        # token roles
        role_logits = self.role_head(h)  # [B,N,C]
        return {"z": z, "feat_hat": feat_hat, "role_logits": role_logits, "mask": mask}

    # ---- losses you can call in your training loop ----
    def loss_reconstruction(self, feat_hat, feat_rows, mask=None, reduction="mean"):
        # compare sentence-mean(features) with feat_hat
        if mask is None:
            sent_target = feat_rows.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            sent_target = (feat_rows * mask.unsqueeze(-1)).sum(dim=1) / denom
        # If features are 0/1 probabilities, BCEWithLogits makes sense:
        return F.binary_cross_entropy_with_logits(feat_hat, sent_target, reduction=reduction)

    def loss_roles(self, role_logits, role_labels, mask):
        # role_labels: Long[B,N], mask: Bool[B,N]
        if self.use_crf:
            ll = self.crf(role_logits, role_labels, mask=mask, reduction='mean')
            return -ll
        else:
            C = role_logits.size(-1)
            loss = F.cross_entropy(role_logits.view(-1, C), role_labels.view(-1), reduction='none')
            loss = loss.view(role_labels.shape)
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp_min(1)

# ---------------- MINIMAL TRAINING STEP SKETCH ----------------
def train_step(model, batch, optimizer, w_rec=1.0, w_roles=1.0):
    model.train()
    out = model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"],
                batch.get("edge_index"), batch.get("edge_type"))
    loss = 0.0
    if w_rec:
        loss_rec = model.loss_reconstruction(out["feat_hat"], batch["feat_rows"], out["mask"])
        loss = loss + w_rec * loss_rec
    if w_roles and "role_labels" in batch:
        loss_roles = model.loss_roles(out["role_logits"], batch["role_labels"], out["mask"])
        loss = loss + w_roles * loss_roles
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}
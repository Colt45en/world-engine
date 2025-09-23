import torch
from types import SimpleNamespace

# ---- knobs ----
B,N,K,C = 3, 7, 12, 3
V, P, R = 1000, 20, 8
D = 256
device = "cpu"

# ---- fake batch ----
g = torch.Generator().manual_seed(42)
tok_ids   = torch.randint(1, V, (B,N), generator=g, device=device)
pos_ids   = torch.randint(1, P, (B,N), generator=g, device=device)
feat_rows = torch.rand(B,N,K, generator=g, device=device)
lengths   = torch.tensor([7,5,6], device=device)

# three per-sentence edge lists in local indices
e0 = (torch.tensor([[1,4,5],[0,2,3]], device=device), torch.tensor([0,1,2], device=device))
e1 = (torch.tensor([[1,3],[0,2]],     device=device), torch.tensor([1,1],   device=device))
e2 = (torch.tensor([[2,4,5],[1,3,0]], device=device), torch.tensor([2,0,2], device=device))

def build_batched_edges(batch_edges, lengths):
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths[:-1]), dim=0)), dtype=torch.long)
    all_src, all_dst, all_type = [], [], []
    for b, (edge_index_b, edge_type_b) in enumerate(batch_edges):
        off = int(offsets[b].item())
        all_src.append(edge_index_b[0] + off)
        all_dst.append(edge_index_b[1] + off)
        all_type.append(edge_type_b)
    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    edge_type  = torch.cat(all_type)
    return edge_index, edge_type

edge_index, edge_type = build_batched_edges([e0,e1,e2], lengths.tolist())

from physics.world_engine_tg import WorldEngineTG          # RGCN flavor
from world_engine_tg_gat import WorldEngineTG_GAT  # GATv2 flavor

def run_model(M):
    model = M(vocab_size=V, d_model=D, k_feats=K, n_pos=P, n_rels=R).to(device)
    out = model(tok_ids, pos_ids, feat_rows, lengths, edge_index, edge_type)
    # sanity
    assert out["z"].shape == (B, 32)
    assert out["feat_hat"].shape == (B, K)
    assert out["role_logits"].shape == (B, N, 3)
    assert out["mask"].dtype == torch.bool
    # losses
    loss = (model.loss_reconstruction(out["feat_hat"], feat_rows, out["mask"])
            + model.loss_roles(out["role_logits"], torch.zeros(B,N,dtype=torch.long,device=device), out["mask"]))
    loss.backward()
    # no NaNs
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    print(M.__name__, "OK:", float(loss.item()))

if __name__ == "__main__":
    run_model(WorldEngineTG)
    run_model(WorldEngineTG_GAT)
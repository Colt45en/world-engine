import torch
from world_engine_tg_gat import WorldEngineTG_GAT

def fake_batch(B=2,N=6,K=8,V=200,P=10,R=5, device="cpu"):
    g = torch.Generator().manual_seed(0)
    tok = torch.randint(1,V,(B,N),generator=g,device=device)
    pos = torch.randint(1,P,(B,N),generator=g,device=device)
    feat= torch.rand(B,N,K,generator=g,device=device)
    lengths = torch.tensor([N, N-2], device=device)
    edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
    edge_type  = torch.zeros((0,),   dtype=torch.long, device=device)
    return tok,pos,feat,lengths,edge_index,edge_type

def test_masking_no_edges():
    tok,pos,feat,lengths,ei,et = fake_batch()
    m = WorldEngineTG_GAT(vocab_size=200, d_model=128, k_feats=8, n_pos=10, n_rels=5)
    out = m(tok,pos,feat,lengths,ei,et)
    # last two positions of sentence 2 are pads -> mask False there
    assert out["mask"][1,-1] == False and out["mask"][1,-2] == False
    # forward/backward without edges
    loss = (m.loss_reconstruction(out["feat_hat"], feat, out["mask"])
            + m.loss_roles(out["role_logits"], torch.zeros_like(tok), out["mask"]))
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

def test_shapes():
    tok,pos,feat,lengths,ei,et = fake_batch()
    m = WorldEngineTG_GAT(vocab_size=200, d_model=128, k_feats=8, n_pos=10, n_rels=5)
    out = m(tok,pos,feat,lengths,ei,et)
    assert out["z"].shape[-1] == 32
    assert out["feat_hat"].shape[-1] == 8
    assert out["role_logits"].shape[-1] == 3
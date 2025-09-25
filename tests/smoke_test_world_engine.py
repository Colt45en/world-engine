"""import torch

Smoke Test for World Enginefrom types import SimpleNamespace



Simple validation that the world engine modules can be imported and instantiated.# ---- knobs ----

"""B,N,K,C = 3, 7, 12, 3

V, P, R = 1000, 20, 8

import sysD = 256

import osdevice = "cpu"

import torch

from types import SimpleNamespace# ---- fake batch ----

g = torch.Generator().manual_seed(42)

# Add src directory to pathtok_ids   = torch.randint(1, V, (B,N), generator=g, device=device)

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))pos_ids   = torch.randint(1, P, (B,N), generator=g, device=device)

feat_rows = torch.rand(B,N,K, generator=g, device=device)

# Test parameterslengths   = torch.tensor([7,5,6], device=device)

device = "cpu"

# three per-sentence edge lists in local indices

def test_world_engine_imports():e0 = (torch.tensor([[1,4,5],[0,2,3]], device=device), torch.tensor([0,1,2], device=device))

    """Test that we can import available world engine modules."""e1 = (torch.tensor([[1,3],[0,2]],     device=device), torch.tensor([1,1],   device=device))

    e2 = (torch.tensor([[2,4,5],[1,3,0]], device=device), torch.tensor([2,0,2], device=device))

    models_tested = []

    def build_batched_edges(batch_edges, lengths):

    # Test world_engine.py    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths[:-1]), dim=0)), dtype=torch.long)

    try:    all_src, all_dst, all_type = [], [], []

        from world_engine import WorldEngine    for b, (edge_index_b, edge_type_b) in enumerate(batch_edges):

        print("✓ Successfully imported WorldEngine from world_engine.py")        off = int(offsets[b].item())

        models_tested.append(("WorldEngine", WorldEngine))        all_src.append(edge_index_b[0] + off)

    except ImportError as e:        all_dst.append(edge_index_b[1] + off)

        print(f"✗ Could not import WorldEngine: {e}")        all_type.append(edge_type_b)

        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)

    # Test world_engine_robust.py      edge_type  = torch.cat(all_type)

    try:    return edge_index, edge_type

        from world_engine_robust import WorldEngine as WorldEngineRobust

        print("✓ Successfully imported WorldEngineRobust from world_engine_robust.py")edge_index, edge_type = build_batched_edges([e0,e1,e2], lengths.tolist())

        models_tested.append(("WorldEngineRobust", WorldEngineRobust))

    except ImportError as e:import sys

        print(f"✗ Could not import WorldEngineRobust: {e}")import os

    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

    return models_tested

from world_engine_tg import WorldEngineTG          # RGCN flavor

def test_model_instantiation(name, model_class):from world_engine_tg_gat import WorldEngineTG_GAT  # GATv2 flavor

    """Test basic model instantiation."""

    def run_model(M):

    try:    model = M(vocab_size=V, d_model=D, k_feats=K, n_pos=P, n_rels=R).to(device)

        print(f"\nTesting {name} instantiation...")    out = model(tok_ids, pos_ids, feat_rows, lengths, edge_index, edge_type)

            # sanity

        # Try basic instantiation first    assert out["z"].shape == (B, 32)

        model = model_class()    assert out["feat_hat"].shape == (B, K)

        print(f"✓ {name} instantiated successfully")    assert out["role_logits"].shape == (B, N, 3)

            assert out["mask"].dtype == torch.bool

        # Try to get some basic info about the model    # losses

        if hasattr(model, 'parameters'):    loss = (model.loss_reconstruction(out["feat_hat"], feat_rows, out["mask"])

            param_count = sum(p.numel() for p in model.parameters())            + model.loss_roles(out["role_logits"], torch.zeros(B,N,dtype=torch.long,device=device), out["mask"]))

            print(f"  - Model has {param_count:,} parameters")    loss.backward()

            # no NaNs

        return True    for p in model.parameters():

                if p.grad is not None:

    except Exception as e:            assert torch.isfinite(p.grad).all()

        print(f"✗ {name} instantiation failed: {e}")    print(M.__name__, "OK:", float(loss.item()))

        return False

if __name__ == "__main__":

def main():    print("Running smoke tests for available models...")

    print("=== World Engine Smoke Test ===\n")    for model_class in available_models:

            run_model(model_class)
    print("Testing imports...")
    available_models = test_world_engine_imports()

    if not available_models:
        print("\n❌ No world engine models could be imported!")
        return False

    print(f"\nFound {len(available_models)} available models")

    # Test instantiation
    success_count = 0
    for name, model_class in available_models:
        if test_model_instantiation(name, model_class):
            success_count += 1

    print(f"\n=== Results ===")
    print(f"✓ {success_count}/{len(available_models)} models passed basic tests")

    if success_count == len(available_models):
        print("🎉 All available models passed smoke tests!")
        return True
    else:
        print("⚠️  Some models failed basic tests")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

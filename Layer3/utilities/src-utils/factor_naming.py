import torch

def name_factors(model, feature_vocab, topk=5):
    """Auto-label each latent z-dimension by top-k feature weights."""
    W = model.dec_feat.weight.detach().T  # [K, z_dim]
    names = []
    for k in range(W.shape[1]):
        top = torch.topk(W[:,k], k=topk).indices.tolist()
        names.append([feature_vocab[i] for i in top])
    return names

# Usage: print(name_factors(model, feature_vocab))
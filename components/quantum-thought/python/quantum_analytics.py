import torch
import matplotlib.pyplot as plt

try:
    import umap  # from umap-learn
except Exception:
    umap = None

def _find_first_linear(module: torch.nn.Module):
    """Find the first nn.Linear inside a Sequential-like head."""
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return m
    return None

@torch.no_grad()
def name_factors(model, feature_vocab, topk=5):
    """
    Auto-label each z-dimension with top-k features it most influences.
    Strategy:
      1) Prefer the first Linear inside model.dec_feat (closest to z).
      2) If not present, compute a local Jacobian d(feat)/dz at z=0 and rank.
    """
    first_linear = None
    if hasattr(model, "dec_feat"):
        first_linear = _find_first_linear(model.dec_feat)

    if isinstance(first_linear, torch.nn.Linear) and first_linear.in_features <= 512:
        W = first_linear.weight.detach()
        last_linear = None
        for m in model.dec_feat.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        if last_linear is not None and last_linear.in_features == first_linear.out_features:
            W_eff = last_linear.weight.detach() @ first_linear.weight.detach()
        else:
            W_eff = getattr(getattr(model, "dec_feat", first_linear), "weight", None)
            if W_eff is None or W_eff.dim() != 2:
                W_eff = torch.zeros(len(feature_vocab), first_linear.in_features)
        Wkz = W_eff
        z_dim = Wkz.shape[1]
        names = []
        for k in range(z_dim):
            top = torch.topk(Wkz[:, k].abs(), k=topk).indices.tolist()
            names.append([feature_vocab[i] for i in top])
        return names

    try:
        params = list(model.parameters())
        if not params:
            raise RuntimeError("Model has no parameters; cannot compute Jacobian")
        exemplar_param = params[0]
        device = exemplar_param.device
        dtype = exemplar_param.dtype
        zdim = 64
        z0 = torch.zeros(1, zdim, device=device, dtype=dtype).requires_grad_(True)
        if hasattr(model, "dec_feat"):
            feat = model.dec_feat(z0)
        else:
            raise RuntimeError("Model has no dec_feat head")
        K = feat.shape[-1]
        J = []
        for k in range(K):
            model.zero_grad(set_to_none=True)
            grad = torch.autograd.grad(feat[0, k], z0, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(z0)
            J.append(grad[0].abs())
        J = torch.stack(J, dim=0)
        z_dim = J.shape[1]
        names = []
        for k in range(z_dim):
            top = torch.topk(J[:, k], k=topk).indices.tolist()
            names.append([feature_vocab[i] for i in top])
        return names
    except Exception:
        return [[] for _ in range(64)]

def plot_z_umap(z, labels=None):
    """
    UMAP projection of z (numpy or torch). Requires umap-learn.
    """
    if umap is None:
        raise ImportError("umap-learn is not installed. pip install umap-learn")
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42)
    z2d = reducer.fit_transform(z)
    plt.scatter(
        z2d[:, 0],
        z2d[:, 1],
        c=labels if labels is not None else None,
        s=10,
        cmap="Spectral",
    )
    plt.title("z-space UMAP")
    plt.tight_layout()
    plt.show()

def plot_pipeline_3d(pipeline):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for z in pipeline.get_zones():
        if hasattr(z, "position"):
            x, y, zz = z.position
            sx, sy, sz = z.scale
            name = z.name
        else:
            x, y, zz = z["position"]
            sx, sy, sz = z["scale"]
            name = z["name"]
        ax.scatter([x], [y], [zz], s=80, label=f"{name}")
    a = pipeline.get_agent()
    if hasattr(a, "position"):
        ax.scatter([a.position[0]], [a.position[1]], [a.position[2]], s=120, marker="^", label="agent", depthshade=True)
    else:
        ax.scatter([a["position"][0]], [a["position"][1]], [a["position"][2]], s=120, marker="^", label="agent", depthshade=True)
    ax.legend()
    ax.set_title("Quantum Thought Field")
    plt.show()

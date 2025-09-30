# Quantum Thought (combined)

## Build & install (in a virtualenv)

```bash
pip install -U pip wheel
pip install pybind11
pip install .
```

## Usage

```python
from qtp import QuantumThoughtPipeline
from quantum_analytics import name_factors, plot_z_umap, plot_pipeline_3d

q = QuantumThoughtPipeline()
print(q.get_agent().id, len(q.get_zones()))
plot_pipeline_3d(q)

# With your WorldEngine model:
# vocab = [...]  # length K (feature names)
# labels = name_factors(model, vocab, topk=5)
# z = model.generate_embeddings(dataloader)  # numpy or torch (N, zdim)
# plot_z_umap(z)
```

## Notes

- QuantumThoughtPipeline uses `std::array<float, 3>` for safe C++↔Python interop.
- `build_field()` prints a confirmation message once zones and agent are in place.
- `name_factors` adapts between linear and shallow-MLP decoder heads, falling back to a Jacobian estimate.
- `plot_z_umap` requires `umap-learn` (install with `pip install umap-learn`).
- `plot_pipeline_3d` offers a quick 3D visualization of the C++ field and agent.

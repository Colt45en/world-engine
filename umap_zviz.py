import torch
import umap
import matplotlib.pyplot as plt

def plot_z_umap(z, labels=None):
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42)
    z2d = reducer.fit_transform(z)
    plt.scatter(z2d[:,0], z2d[:,1], c=labels if labels is not None else 'b', cmap='Spectral', s=10)
    plt.title("z-space UMAP")
    plt.show()

# Usage:
# z = all_z_vectors_as_numpy
# plot_z_umap(z, labels=pos_ids or role_ids or None)
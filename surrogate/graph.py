"""Dalton-style augmented graph (PI-GNN, Dalton et al. CMAME 2023).

Builds a hierarchical augmented graph on top of a tetrahedral mesh:

  layer 0 : the real mesh nodes (mesh vertices)
  layers 1..L-1 : virtual nodes, one cluster center per `root_leaf_factor`
                  leaves (recursive integer division of the node count),
                  positioned at k-means cluster centers of the previous layer.

Edges (all made bidirectional + deduplicated):
  - the real mesh edges (layer 0),
  - inter-layer: each leaf node -> its cluster root,
  - intra-layer: kNN connections within each virtual layer,
  - the last virtual layer is fully connected.

Virtual nodes are *hidden/latent*: they carry zero features and zero
displacement, and only aggregate / re-distribute information during message
passing (this is what gives the graph its multi-scale inductive bias).

All Dalton geometries share one node numbering, so the graph is built once on a
reference shape (e.g. the mean training shape) and reused; ``aug_coords`` places
the virtual nodes of another geometry at the centroids of the same clusters.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def calc_n_nodes_per_layer(n_real: int, root_leaf_factor: int = 4,
                           min_root_nodes: int = 6):
    """Recursive integer division of the node count (Dalton Algorithm 1)."""
    counts = [n_real]
    while True:
        c = counts[-1] // root_leaf_factor
        if c < min_root_nodes:
            break
        counts.append(c)
    return counts


def build_augmented_graph(coords, mesh_edges, root_leaf_factor=4,
                          min_root_nodes=6, n_nearest_nbrs=3, rng_seed=0):
    """Build the augmented graph for a fixed mesh.

    Args:
        coords:     (n_real, 3) reference coordinates of the real mesh nodes.
        mesh_edges: (E, 2) int array of directed mesh edges (either direction
                    is fine; the result is made bidirectional anyway).

    Returns a dict with:
        aug_coords    (n_total, 3) reference coords of ALL nodes (virtual
                       nodes = k-means cluster centers).
        topology      (n_edges, 2) bidirectional directed edges.
        layer_labels  (n_total,)   0 = real, 1,2,... = virtual layers.
        real_indices  (n_total,)   bool mask of the real nodes.
        n_real, n_total, counts, n_edges.
    """
    n_real = coords.shape[0]
    counts = calc_n_nodes_per_layer(n_real, root_leaf_factor, min_root_nodes)
    n_total = int(np.sum(counts))
    labels = np.concatenate([np.full(c, i) for i, c in enumerate(counts)]).astype(np.int64)

    aug_coords = np.zeros((n_total, 3), dtype=np.float64)
    aug_coords[labels == 0] = coords

    # start from the (bidirectional, deduplicated) real mesh edges
    topology = np.unique(np.asarray(mesh_edges, dtype=np.int64), axis=0)
    parents = []

    for layer in range(1, len(counts)):
        leaf_idx = np.flatnonzero(labels == layer - 1)
        leaf_coords = aug_coords[leaf_idx]
        n_roots = counts[layer]

        # virtual nodes = k-means cluster centers of the previous layer
        km = KMeans(n_clusters=n_roots, random_state=rng_seed, n_init=10).fit(leaf_coords)
        roots = km.cluster_centers_
        aug_coords[labels == layer] = roots

        shift = int(np.sum(labels < layer))
        root_idx = (km.labels_ + shift).astype(np.int64)
        parents.append((leaf_idx, root_idx))

        # inter-layer: leaf -> root
        topology = np.vstack([topology, np.column_stack([leaf_idx, root_idx])])

        # intra-layer connections among the virtual nodes
        if layer < len(counts) - 1:
            nbrs = NearestNeighbors(n_neighbors=n_nearest_nbrs + 1,
                                    algorithm='ball_tree').fit(roots)
            _, idx = nbrs.kneighbors(roots)
            senders = np.repeat(np.arange(n_roots), n_nearest_nbrs) + shift
            receivers = idx[:, 1:].flatten() + shift
            topology = np.vstack([topology, np.column_stack([senders, receivers])])
        else:
            n = n_roots
            idxs = np.array([[i, j] for i in range(n) for j in range(n) if i != j])
            topology = np.vstack([topology, idxs + shift])

    # bidirectional + deduplicate
    topology = np.unique(np.vstack([topology, topology[:, ::-1]]), axis=0)

    return dict(aug_coords=aug_coords, topology=topology, layer_labels=labels,
                real_indices=(labels == 0), n_real=n_real, n_total=n_total,
                counts=counts, n_edges=topology.shape[0], parents=parents)


def aug_coords(coords, aug):
    """Reference coordinates of all nodes for another geometry with the same node
    numbering: virtual nodes = centroids of their (fixed) clusters, layer by layer."""
    X = np.array(aug["aug_coords"], dtype=np.float64)
    X[:aug["n_real"]] = coords
    for leaf_idx, root_idx in aug["parents"]:
        s = np.zeros_like(X)
        c = np.zeros(len(X))
        np.add.at(s, root_idx, X[leaf_idx])
        np.add.at(c, root_idx, 1.0)
        m = c > 0
        X[m] = s[m] / c[m, None]
    return X

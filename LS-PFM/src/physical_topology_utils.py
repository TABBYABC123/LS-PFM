"""Utilities for physical-topology latent pooling.

The important design choice in v7 is that the number of topology latent slots
is derived from the physical adjacency graph by default.  For the current data:

- cluster1 has 5 physical nodes and suggests 2 coarse topology regions.
- cluster2 has 14 physical nodes and suggests 4 coarse topology regions.

Training fixes the model topology size from the source graph, then target
graphs are projected into that same topology size during evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

EPS = 1e-8


def resolve_read_path(path: str, root: Path) -> str:
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    return str((root / p).resolve())


def resolve_write_path(path: str, root: Path) -> str:
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if p.parts and p.parts[0] == "..":
        return str((Path.cwd() / p).resolve())
    return str((root / p).resolve())


def parse_topo_nodes(value: str, adj: np.ndarray, max_auto: int = 8) -> int:
    value = str(value or "auto").lower().strip()
    if value == "auto":
        return estimate_topology_nodes(adj, max_auto=max_auto)
    k = int(value)
    if k <= 0:
        raise ValueError("--topo_nodes must be auto or a positive integer.")
    return k


def clean_similarity(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adj must be square, got {a.shape}")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.maximum(a, 0.0)
    if a.max() > 0:
        a = a / a.max()
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 1.0)
    return a


def estimate_topology_nodes(adj: np.ndarray, max_auto: int = 8) -> int:
    """Estimate a coarse topology size from the normalized-Laplacian eigengap."""
    a = clean_similarity(adj)
    n = a.shape[0]
    if n <= 2:
        return n
    sim = a.copy()
    np.fill_diagonal(sim, 0.0)
    deg = sim.sum(axis=1)
    if float(deg.sum()) <= EPS:
        return min(n, 2)
    inv = np.diag(1.0 / np.sqrt(np.maximum(deg, EPS)))
    lap = np.eye(n) - inv @ sim @ inv
    eig = np.sort(np.linalg.eigvalsh(lap))
    # Candidate k is in [2, max_auto], capped by graph size and a sqrt rule so
    # tiny graphs are not over-clustered.
    cap = min(n, int(max_auto), max(2, int(math.ceil(math.sqrt(n) * 1.5))))
    if cap <= 2:
        return 2
    gaps = np.diff(eig[: cap + 1])
    if len(gaps) <= 1:
        return 2
    return int(np.argmax(gaps[1:]) + 2)


def _kmeans_np(x: np.ndarray, k: int, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n == 0:
        raise ValueError("Cannot run kmeans on empty input.")
    k_eff = min(k, n)
    if k_eff == 1:
        return np.zeros(n, dtype=np.int64), x[:1].copy()
    idx = np.linspace(0, n - 1, num=k_eff).round().astype(np.int64)
    centers = x[idx].copy()
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, int(iters))):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_assign = dist.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k_eff):
            mask = new_assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(axis=0)
        if np.array_equal(new_assign, assign) and np.linalg.norm(new_centers - centers) < 1e-8:
            assign = new_assign
            centers = new_centers
            break
        assign = new_assign
        centers = new_centers
    return assign, centers


def physical_topology_from_adj(adj: np.ndarray, topo_nodes: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return row-normalized assignment [K,D] and topology adjacency [K,K]."""
    a = clean_similarity(adj)
    n = a.shape[0]
    k = int(topo_nodes)
    if k <= 0:
        raise ValueError("topo_nodes must be positive.")

    sim = a.copy()
    np.fill_diagonal(sim, 0.0)
    deg = sim.sum(axis=1)
    if float(deg.sum()) <= EPS:
        embed = np.linspace(0.0, 1.0, n).reshape(n, 1)
    else:
        inv = np.diag(1.0 / np.sqrt(np.maximum(deg, EPS)))
        lap = np.eye(n) - inv @ sim @ inv
        vals, vecs = np.linalg.eigh(lap)
        order = np.argsort(vals)
        dims = max(1, min(k, n) - 1)
        embed = vecs[:, order[1 : 1 + dims]]
        if embed.ndim == 1:
            embed = embed[:, None]
        embed = embed / np.maximum(np.linalg.norm(embed, axis=1, keepdims=True), EPS)

    assign_idx, _ = _kmeans_np(embed, min(k, n), iters=50)
    assign = np.zeros((k, n), dtype=np.float32)
    for node, cluster in enumerate(assign_idx):
        assign[int(cluster), node] = 1.0

    # If k > n, fill extra slots with high-degree representative nodes.  This
    # keeps shape stable while making over-large k visibly redundant.
    if k > n:
        reps = np.argsort(-deg if float(deg.sum()) > EPS else np.arange(n))[: k - n]
        for extra, node in enumerate(reps, start=n):
            if extra < k:
                assign[extra, int(node)] = 1.0

    # Repair empty clusters for k <= n by assigning high-degree nodes.
    empty = np.where(assign.sum(axis=1) <= 0)[0]
    if len(empty):
        reps = np.argsort(-deg if float(deg.sum()) > EPS else np.arange(n))
        for row, node in zip(empty, reps):
            assign[row, int(node)] = 1.0

    cluster_node_weights = assign.sum(axis=1).astype(float).tolist()
    assign = assign / np.maximum(assign.sum(axis=1, keepdims=True), EPS)
    topo_adj = assign @ a @ assign.T
    topo_adj = 0.5 * (topo_adj + topo_adj.T)
    np.fill_diagonal(topo_adj, np.maximum(np.diag(topo_adj), 1.0))
    topo_adj = topo_adj / np.maximum(topo_adj.sum(axis=1, keepdims=True), EPS)
    payload = {
        "physical_nodes": int(n),
        "topo_nodes": int(k),
        "cluster_node_weights": cluster_node_weights,
        "assignment": assign.astype(float).tolist(),
    }
    return assign.astype(np.float32), topo_adj.astype(np.float32), payload


def save_topology_info(path: str, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

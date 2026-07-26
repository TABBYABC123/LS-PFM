"""Residual-GWNM model.

Main additions over the original GW-NM:
1. reconstruction head, producing a direct anomaly signal;
2. physical-graph + adaptive-graph residual reconstruction;
3. dynamic hidden sequence for discriminator scoring;
4. latent Gaussian measure output for W2 normal-measure scoring.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def row_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    adj = torch.nan_to_num(adj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = adj.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return adj / denom


class DenseGraphConv(nn.Module):
    """Dense graph convolution: A @ W(x) + W_root(x).

    This avoids torch_geometric dependency and works for small/medium base-station graphs.
    """

    def __init__(self, in_dim: int, out_dim: int, root_weight: bool = True, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.root_lin = nn.Linear(in_dim, out_dim, bias=False) if root_weight else None
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B,D,C], adj: [D,D]
        adj = row_normalize_adj(adj).to(x.device, dtype=x.dtype)
        out = torch.matmul(adj, self.lin(x))
        if self.root_lin is not None:
            out = out + self.root_lin(x)
        if self.bias is not None:
            out = out + self.bias
        return out


class DenseGraphGRUCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reset_gate = DenseGraphConv(in_dim + hidden_dim, hidden_dim)
        self.update_gate = DenseGraphConv(in_dim + hidden_dim, hidden_dim)
        self.candidate_gate = DenseGraphConv(in_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        gate_in = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.reset_gate(gate_in, adj))
        z = torch.sigmoid(self.update_gate(gate_in, adj))
        cand_in = torch.cat([x, r * h], dim=-1)
        n = torch.tanh(self.candidate_gate(cand_in, adj))
        return z * h + (1.0 - z) * n


class DenseGraphGRU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            DenseGraphGRUCell(in_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D,C]
        b, t, d, _ = x.shape
        states = [x.new_zeros(b, d, self.hidden_dim) for _ in range(self.num_layers)]
        outs = []
        for step in range(t):
            inp = x[:, step]
            next_states = []
            for layer, cell in enumerate(self.cells):
                h = cell(inp, states[layer], adj)
                inp = self.dropout(h) if layer + 1 < self.num_layers else h
                next_states.append(h)
            states = next_states
            outs.append(states[-1])
        return torch.stack(outs, dim=1)  # [B,T,D,H]


class AdaptiveGraphBuilder(nn.Module):
    """Variable-node adaptive graph from station metadata.

    Different from fixed nodevec1/nodevec2, this works when D differs by cluster.
    """

    def __init__(self, meta_dim: int, rank: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(meta_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, rank))
        self.k = nn.Sequential(nn.Linear(meta_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, rank))

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        # meta: [D,M]
        q = self.q(meta)
        k = self.k(meta)
        logits = torch.relu(q @ k.T) / math.sqrt(max(q.shape[-1], 1))
        return torch.softmax(logits, dim=-1)


class ResidualMeasureEncoder(nn.Module):
    def __init__(
        self,
        window_len: int,
        meta_dim: int,
        time_dim: int = 6,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        graph_layers: int = 1,
        dropout: float = 0.05,
        adaptive_rank: int = 16,
        use_temporal_embedding: bool = True,
        use_residual_branch: bool = True,
        use_adaptive_graph: bool = True,
    ):
        super().__init__()
        self.window_len = int(window_len)
        self.meta_dim = int(meta_dim)
        self.time_dim = int(time_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.use_temporal_embedding = bool(use_temporal_embedding)
        self.use_residual_branch = bool(use_residual_branch)
        self.use_adaptive_graph = bool(use_adaptive_graph)

        self.value_proj1 = nn.Linear(1, hidden_dim)
        self.value_proj2 = nn.Linear(1, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.physical_gru = DenseGraphGRU(hidden_dim, hidden_dim, num_layers=graph_layers, dropout=dropout)
        self.residual_gru = DenseGraphGRU(hidden_dim, hidden_dim, num_layers=graph_layers, dropout=dropout)
        self.adaptive_graph = AdaptiveGraphBuilder(meta_dim, rank=adaptive_rank, hidden_dim=hidden_dim)

        self.rec1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.rec2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.token_to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(latent_dim, max(hidden_dim // 2, 8)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )
        self.context_norm = nn.LayerNorm(latent_dim)
        self.mu_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))
        self.log_sigma_head = nn.Sequential(nn.Linear(latent_dim, max(hidden_dim // 2, 8)), nn.SiLU(), nn.Linear(max(hidden_dim // 2, 8), 1))

    def _make_input(self, x_bt_d_1: torch.Tensor, time_feat: torch.Tensor, meta: torch.Tensor, branch: int) -> torch.Tensor:
        # x_bt_d_1: [B,T,D,1], time_feat: [B,T,F], meta: [D,M]
        b, t, d, _ = x_bt_d_1.shape
        v = self.value_proj1(x_bt_d_1) if branch == 1 else self.value_proj2(x_bt_d_1)
        m = self.meta_proj(meta).view(1, 1, d, self.hidden_dim)
        if self.use_temporal_embedding:
            te = self.time_proj(time_feat).view(b, t, 1, self.hidden_dim)
        else:
            te = torch.zeros(b, t, 1, self.hidden_dim, device=x_bt_d_1.device, dtype=x_bt_d_1.dtype)
        return v + m + te

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        adj: torch.Tensor,
        time_feat: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # x: [B,D,T]
        if x.ndim != 3:
            raise ValueError(f"x must be [B,D,T], got {tuple(x.shape)}")
        b, d, t = x.shape
        if t != self.window_len:
            raise ValueError(f"Expected window length {self.window_len}, got {t}.")
        if meta.ndim != 2 or meta.shape[0] != d:
            raise ValueError(f"meta shape {tuple(meta.shape)} incompatible with x {tuple(x.shape)}")
        if time_feat is None:
            time_feat = x.new_zeros(b, t, self.time_dim)
        if time_feat.ndim != 3 or time_feat.shape[:2] != (b, t):
            raise ValueError(f"time_feat must be [B,T,F], got {tuple(time_feat.shape)}")

        meta = meta.to(x.device, dtype=x.dtype)
        adj = adj.to(x.device, dtype=x.dtype)
        time_feat = time_feat.to(x.device, dtype=x.dtype)

        x_bt = x.transpose(1, 2).unsqueeze(-1)  # [B,T,D,1]
        inp1 = self._make_input(x_bt, time_feat, meta, branch=1)
        h1 = self.physical_gru(inp1, adj)
        rec1 = self.rec1(h1).squeeze(-1)  # [B,T,D]

        if self.use_residual_branch:
            residual = x_bt.squeeze(-1) - rec1
            inp2 = self._make_input(residual.unsqueeze(-1), time_feat, meta, branch=2)
            if self.use_adaptive_graph:
                a_adp = self.adaptive_graph(meta)
            else:
                a_adp = adj
            h2 = self.residual_gru(inp2, a_adp)
            rec2 = self.rec2(h2).squeeze(-1)
        else:
            h2 = torch.zeros_like(h1)
            rec2 = torch.zeros_like(rec1)
            a_adp = adj

        rec_bt = rec1 + rec2
        x_rec = rec_bt.transpose(1, 2).contiguous()  # [B,D,T]

        h_dyn = self.fusion(torch.cat([h1, h2], dim=-1))  # [B,T,D,H]
        node_context = h_dyn.mean(dim=1)                 # [B,D,H]
        token_z = self.token_to_latent(node_context)      # [B,D,L]

        attn_logits = self.attn(token_z).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        context = torch.sum(token_z * attn.unsqueeze(-1), dim=1)
        context = self.context_norm(context)
        mu = self.mu_head(context)
        sigma = F.softplus(self.log_sigma_head(context)) + 1e-4

        if return_aux:
            return {
                "mu": mu,
                "sigma": sigma,
                "token_z": token_z,
                "x_rec": x_rec,
                "h_dyn": h_dyn,
                "adaptive_adj": a_adp,
            }
        return mu, sigma, token_z


def pairwise_token_distance(token_z: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(token_z, token_z, p=2)
    b, d, _ = dist.shape
    if d > 1:
        mask = ~torch.eye(d, dtype=torch.bool, device=dist.device)
        mean = dist[:, mask].mean(dim=1).view(b, 1, 1).clamp_min(EPS)
        dist = dist / mean
    return dist


def structure_loss(token_z: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    dist_z = pairwise_token_distance(token_z)
    d = distance.to(token_z.device, dtype=token_z.dtype)
    n = d.shape[0]
    if n <= 1:
        return torch.zeros((), device=token_z.device, dtype=token_z.dtype)
    mask = ~torch.eye(n, dtype=torch.bool, device=d.device)
    target = d / d[mask].mean().clamp_min(EPS)
    return ((dist_z[:, mask] - target[mask].unsqueeze(0)) ** 2).mean()


@torch.no_grad()
def structure_score(token_z: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    dist_z = pairwise_token_distance(token_z)
    d = distance.to(token_z.device, dtype=token_z.dtype)
    n = d.shape[0]
    if n <= 1:
        return torch.zeros(token_z.shape[0], device=token_z.device, dtype=token_z.dtype)
    mask = ~torch.eye(n, dtype=torch.bool, device=d.device)
    target = d / d[mask].mean().clamp_min(EPS)
    return ((dist_z[:, mask] - target[mask].unsqueeze(0)) ** 2).mean(dim=1)


def gaussian_w2_isotropic(mu: torch.Tensor, sigma: torch.Tensor, ref_mu: torch.Tensor, ref_sigma: torch.Tensor) -> torch.Tensor:
    ref_mu = ref_mu.to(mu.device, dtype=mu.dtype)
    ref_sigma = ref_sigma.to(mu.device, dtype=mu.dtype)
    mean_term = ((mu - ref_mu) ** 2).sum(dim=-1)
    sig = sigma.view(-1)
    rs = ref_sigma.view(-1)[0] if ref_sigma.numel() > 1 else ref_sigma.reshape(())
    cov_term = mu.shape[-1] * (torch.sqrt(sig.clamp_min(EPS)) - torch.sqrt(rs.clamp_min(EPS))) ** 2
    return mean_term + cov_term


@torch.no_grad()
def encode_windows(
    model: ResidualMeasureEncoder,
    windows: torch.Tensor,
    time_feat: torch.Tensor,
    meta: torch.Tensor,
    adj: torch.Tensor,
    batch_size: int = 128,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    model.eval()
    outs: Dict[str, list] = {"mu": [], "sigma": [], "token_z": [], "x_rec": [], "h_dyn": []}
    n = windows.shape[0]
    meta = meta.to(device)
    adj = adj.to(device)
    for i in range(0, n, batch_size):
        xb = windows[i:i + batch_size].to(device)
        tb = time_feat[i:i + batch_size].to(device)
        out = model(xb, meta=meta, adj=adj, time_feat=tb, return_aux=True)
        for k in outs:
            outs[k].append(out[k].detach().cpu())
    return {k: torch.cat(v, dim=0) for k, v in outs.items()}


@torch.no_grad()
def estimate_reference(mu: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return mu.mean(dim=0), sigma.mean(dim=0)


def reconstruction_score(x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - x_rec) ** 2, dim=(1, 2)) + EPS)

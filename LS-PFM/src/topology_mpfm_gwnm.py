"""Topology-latent MPFM-Residual-GWNM.

This module keeps the original ResidualMeasureEncoder, but changes the latent
space used by the prototype/flow head:

old: token_z [B, D, L] -> pooled h [B, L] -> prototypes [P, L]
new: token_z [B, D, L] -> fixed h_topo [B, K, L] -> prototypes [P, K, L]

K is a fixed topology size, default 30.  Variable-size station graphs are
softly projected onto these K topology anchors with station metadata and the
physical adjacency, so the anomaly head sees a structured latent graph instead
of one collapsed vector.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gwnm_model import ResidualMeasureEncoder

EPS = 1e-8


def row_normalize(a: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return a / a.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _offdiag_mask(n: int, device) -> torch.Tensor:
    return ~torch.eye(n, dtype=torch.bool, device=device)


class TopologyGraphBlock(nn.Module):
    """Small dense graph residual block for fixed topology nodes."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.05):
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        self.self_proj = nn.Linear(dim, hidden_dim)
        self.neigh_proj = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B,K,C], adj: [K,K]
        adj = row_normalize(adj).to(x.device, dtype=x.dtype)
        neigh = torch.einsum("ij,bjc->bic", adj, x)
        h = F.silu(self.self_proj(x) + self.neigh_proj(neigh))
        return self.norm(x + self.dropout(self.out(h)))


class TopologyProjector(nn.Module):
    """Project variable station tokens to a fixed topology latent graph."""

    def __init__(
        self,
        meta_dim: int,
        latent_dim: int,
        topo_nodes: int = 30,
        topo_rank: int = 32,
        topo_layers: int = 2,
        dropout: float = 0.05,
        order_bias: float = 2.0,
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.latent_dim = int(latent_dim)
        self.topo_nodes = int(topo_nodes)
        self.order_bias = float(order_bias)

        self.anchor_meta = nn.Parameter(torch.randn(topo_nodes, meta_dim) * 0.05)
        self.anchor_q = nn.Sequential(
            nn.Linear(meta_dim, topo_rank),
            nn.SiLU(),
            nn.Linear(topo_rank, topo_rank),
        )
        self.node_k = nn.Sequential(
            nn.Linear(meta_dim, topo_rank),
            nn.SiLU(),
            nn.Linear(topo_rank, topo_rank),
        )
        self.pos_emb = nn.Parameter(torch.randn(topo_nodes, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TopologyGraphBlock(latent_dim, hidden_dim=max(latent_dim * 2, 32), dropout=dropout) for _ in range(topo_layers)]
        )
        self.out_norm = nn.LayerNorm(latent_dim)
        self.register_buffer("anchor_pos", torch.linspace(0.0, 1.0, topo_nodes).view(topo_nodes, 1))

    def anchor_assignment(self, meta: torch.Tensor) -> torch.Tensor:
        # meta: [D,M].  meta[:,0] is the deterministic station index feature
        # produced by build_station_meta(); the bias gives the fixed K anchors a
        # stable coarse ordering before learning refines the mapping.
        q = self.anchor_q(self.anchor_meta.to(meta.device, dtype=meta.dtype))
        k = self.node_k(meta)
        logits = q @ k.T / math.sqrt(max(q.shape[-1], 1))
        if meta.shape[1] > 0 and self.order_bias > 0:
            node_pos = meta[:, 0].view(1, -1).to(logits.dtype)
            bias = -torch.abs(self.anchor_pos.to(meta.device, logits.dtype) - node_pos)
            logits = logits + self.order_bias * bias
        return torch.softmax(logits, dim=-1)  # [K,D]

    def projected_adjacency(self, assign: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # assign: [K,D], adj: [D,D]
        adj = row_normalize(adj).to(assign.device, dtype=assign.dtype)
        topo_adj = assign @ adj @ assign.T
        topo_adj = 0.5 * (topo_adj + topo_adj.T)
        eye = torch.eye(topo_adj.shape[0], device=topo_adj.device, dtype=topo_adj.dtype)
        topo_adj = topo_adj + eye
        return row_normalize(topo_adj)

    def forward(self, token_z: torch.Tensor, meta: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # token_z: [B,D,L]
        if token_z.ndim != 3:
            raise ValueError(f"token_z must be [B,D,L], got {tuple(token_z.shape)}")
        meta = meta.to(token_z.device, dtype=token_z.dtype)
        adj = adj.to(token_z.device, dtype=token_z.dtype)
        assign = self.anchor_assignment(meta)  # [K,D]
        h = torch.einsum("kd,bdl->bkl", assign, token_z)
        h = h + self.pos_emb.to(h.device, h.dtype).unsqueeze(0)
        topo_adj = self.projected_adjacency(assign, adj)
        for block in self.blocks:
            h = block(h, topo_adj)
        return self.out_norm(h), topo_adj, assign


class TopologyVelocityField(nn.Module):
    """Flow-matching velocity field on [B,K,L] topology latents."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.05):
        super().__init__()
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.ctx_proj = nn.Linear(latent_dim, hidden_dim)
        self.t_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([TopologyGraphBlock(hidden_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor, topo_adj: torch.Tensor) -> torch.Tensor:
        # x_t: [B,K,L], context: [K,L], t: [B,1] or [B,1,1]
        if t.ndim == 3:
            t_in = t[:, 0, :]
        elif t.ndim == 1:
            t_in = t.unsqueeze(-1)
        else:
            t_in = t
        ctx = self.ctx_proj(context.to(x_t.device, x_t.dtype)).unsqueeze(0)
        h = self.x_proj(x_t) + ctx + self.t_proj(t_in.to(x_t.dtype)).unsqueeze(1)
        for block in self.blocks:
            h = block(h, topo_adj)
        return self.out(self.out_norm(h))


class TopologyPrototypeFlow(nn.Module):
    """Shared/private prototypes that keep topology: prototypes are [P,K,L]."""

    def __init__(
        self,
        latent_dim: int,
        topo_nodes: int,
        num_domains: int,
        k_shared: int = 16,
        k_private: int = 8,
        tau: float = 0.5,
        flow_hidden: int = 128,
        flow_depth: int = 3,
        dropout: float = 0.05,
        init_scale: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.topo_nodes = int(topo_nodes)
        self.num_domains = int(num_domains)
        self.k_shared = int(k_shared)
        self.k_private = int(k_private)
        self.tau = float(tau)
        self.shared = nn.Parameter(torch.randn(k_shared, topo_nodes, latent_dim) * init_scale)
        self.private = nn.Parameter(torch.randn(num_domains, k_private, topo_nodes, latent_dim) * init_scale)
        self.flow = TopologyVelocityField(latent_dim, hidden_dim=flow_hidden, depth=flow_depth, dropout=dropout)

    def prototypes(self, domain_idx: int, private_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        if private_override is None:
            priv = self.private[int(domain_idx)]
        else:
            priv = private_override.to(self.shared.device, dtype=self.shared.dtype)
        if priv.ndim != 3:
            raise ValueError(f"private prototypes must be [Kp,N,L], got {tuple(priv.shape)}")
        return torch.cat([self.shared, priv], dim=0)

    def domain_context(self, domain_idx: int = 0, private_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.prototypes(domain_idx, private_override=private_override).mean(dim=0)  # [N,L]

    def assign(
        self,
        h: torch.Tensor,
        domain_idx: int = 0,
        private_override: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        # h: [B,N,L], p: [P,N,L]
        p = self.prototypes(domain_idx, private_override=private_override)
        dist2_node = (h.unsqueeze(1) - p.unsqueeze(0)).pow(2).mean(dim=-1)  # [B,P,N]
        dist2 = dist2_node.mean(dim=-1)  # [B,P]
        temp = self.tau if tau is None else float(tau)
        alpha = torch.softmax(-dist2 / max(temp, EPS), dim=-1)
        h_hat = torch.einsum("bp,pnl->bnl", alpha, p)
        min_dist = torch.sqrt(dist2.min(dim=-1).values + EPS)
        proto_rec = torch.sqrt((h - h_hat).pow(2).mean(dim=(1, 2)) + EPS)
        node_proto_rec = torch.sqrt((h - h_hat).pow(2).mean(dim=-1) + EPS)
        nearest = dist2.argmin(dim=-1)
        return {
            "prototypes": p,
            "alpha": alpha,
            "h_hat": h_hat,
            "min_dist": min_dist,
            "proto_rec": proto_rec,
            "node_proto_rec": node_proto_rec,
            "nearest": nearest,
        }

    def flow_loss(self, target: torch.Tensor, context: torch.Tensor, topo_adj: torch.Tensor) -> torch.Tensor:
        b = target.shape[0]
        eps = torch.randn_like(target)
        t = torch.rand(b, 1, 1, device=target.device, dtype=target.dtype)
        x_t = (1.0 - t) * eps + t * target
        v_target = target - eps
        v_pred = self.flow(x_t, t, context, topo_adj)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def flow_error(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        topo_adj: torch.Tensor,
        t_star: float = 0.5,
        mc_samples: int = 4,
        deterministic: bool = False,
    ) -> torch.Tensor:
        errs = []
        for _ in range(max(1, int(mc_samples))):
            eps = torch.zeros_like(z) if deterministic else torch.randn_like(z)
            t = torch.full((z.shape[0], 1, 1), float(t_star), device=z.device, dtype=z.dtype)
            x_t = (1.0 - t) * eps + t * z
            v_target = z - eps
            v_pred = self.flow(x_t, t, context, topo_adj)
            errs.append((v_pred - v_target).pow(2).mean(dim=(1, 2)))
        return torch.stack(errs, dim=0).mean(dim=0)

    def orthogonal_loss(self) -> torch.Tensor:
        def _ortho(p: torch.Tensor) -> torch.Tensor:
            if p.shape[0] <= 1:
                return p.new_tensor(0.0)
            flat = F.normalize(p.flatten(1), dim=-1)
            gram = flat @ flat.T
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            return (gram - eye).pow(2).mean()

        loss = _ortho(self.shared)
        for d in range(self.private.shape[0]):
            loss = loss + _ortho(self.private[d])
        return loss / max(1, self.private.shape[0] + 1)

    def prototype_smoothness_loss(self, topo_adj: torch.Tensor) -> torch.Tensor:
        p = torch.cat([self.shared, self.private.flatten(0, 1)], dim=0)
        return topology_smoothness_loss(p, topo_adj)

    def info_nce_loss(self, h: torch.Tensor, domain_idx: int, private_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = self.prototypes(domain_idx, private_override=private_override)
        h_n = F.normalize(h.flatten(1), dim=-1)
        p_n = F.normalize(p.flatten(1), dim=-1)
        logits = h_n @ p_n.T / max(self.tau, EPS)
        target = logits.argmax(dim=-1).detach()
        return F.cross_entropy(logits, target)

    def hard_drift_loss(
        self,
        domain_idx: int,
        prev_shared: Optional[torch.Tensor],
        prev_private_domain: Optional[torch.Tensor],
        topo_adj: torch.Tensor,
        beta: float = 0.5,
        topk: int = 8,
    ) -> torch.Tensor:
        if prev_shared is None or prev_private_domain is None:
            return self.shared.new_tensor(0.0)
        curr = self.prototypes(domain_idx)
        prev = torch.cat([prev_shared.to(curr.device, curr.dtype), prev_private_domain.to(curr.device, curr.dtype)], dim=0)
        if curr.shape != prev.shape:
            return self.shared.new_tensor(0.0)
        drift = curr.detach() - prev.detach()
        cand = torch.cat([curr.detach() - float(beta) * drift, curr.detach() + float(beta) * drift], dim=0)
        ctx = self.domain_context(domain_idx).detach()
        b = cand.shape[0]
        eps = torch.randn_like(cand)
        t = torch.rand(b, 1, 1, device=cand.device, dtype=cand.dtype)
        x_t = (1.0 - t) * eps + t * cand
        v_target = cand - eps
        v_pred = self.flow(x_t, t, ctx, topo_adj)
        err = (v_pred - v_target).pow(2).mean(dim=(1, 2))
        k = min(max(1, int(topk)), err.numel())
        return err[torch.topk(err.detach(), k=k, largest=True).indices].mean()


class TopologyLatentMPFMResidualGWNM(nn.Module):
    """Residual-GWNM encoder plus topology-preserving MPFM head."""

    def __init__(
        self,
        window_len: int,
        meta_dim: int,
        time_dim: int,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        graph_layers: int = 1,
        dropout: float = 0.05,
        adaptive_rank: int = 16,
        use_temporal_embedding: bool = True,
        use_residual_branch: bool = True,
        use_adaptive_graph: bool = True,
        num_domains: int = 1,
        k_shared: int = 16,
        k_private: int = 8,
        tau: float = 0.5,
        flow_hidden: int = 128,
        flow_depth: int = 3,
        topo_nodes: int = 30,
        topo_rank: int = 32,
        topo_layers: int = 2,
        topo_order_bias: float = 2.0,
    ):
        super().__init__()
        self.encoder = ResidualMeasureEncoder(
            window_len=window_len,
            meta_dim=meta_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            adaptive_rank=adaptive_rank,
            use_temporal_embedding=use_temporal_embedding,
            use_residual_branch=use_residual_branch,
            use_adaptive_graph=use_adaptive_graph,
        )
        self.topology = TopologyProjector(
            meta_dim=meta_dim,
            latent_dim=latent_dim,
            topo_nodes=topo_nodes,
            topo_rank=topo_rank,
            topo_layers=topo_layers,
            dropout=dropout,
            order_bias=topo_order_bias,
        )
        self.topo_norm = nn.LayerNorm(latent_dim)
        self.mpfm = TopologyPrototypeFlow(
            latent_dim=latent_dim,
            topo_nodes=topo_nodes,
            num_domains=num_domains,
            k_shared=k_shared,
            k_private=k_private,
            tau=tau,
            flow_hidden=flow_hidden,
            flow_depth=flow_depth,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor, meta: torch.Tensor, adj: torch.Tensor, time_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(x, meta=meta, adj=adj, time_feat=time_feat, return_aux=True)
        h_topo, topo_adj, topo_assign = self.topology(out["token_z"], meta=meta, adj=adj)
        out["h_topo"] = self.topo_norm(h_topo)
        out["topo_adj"] = topo_adj
        out["topo_assign"] = topo_assign
        # Kept only for logging/backward-compatible summaries.  The anomaly head
        # below uses h_topo [B,K,L], not this pooled vector.
        out["h"] = out["h_topo"].mean(dim=1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        adj: torch.Tensor,
        time_feat: torch.Tensor,
        domain_idx: int = 0,
        private_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.encode(x, meta=meta, adj=adj, time_feat=time_feat)
        pinfo = self.mpfm.assign(out["h_topo"], domain_idx=domain_idx, private_override=private_override)
        out.update({"proto_" + k: v for k, v in pinfo.items() if k != "prototypes"})
        out["prototypes"] = pinfo["prototypes"]
        return out


def topology_smoothness_loss(h: torch.Tensor, topo_adj: torch.Tensor) -> torch.Tensor:
    """Graph smoothness for [B,K,L] or [P,K,L] tensors."""
    if h.ndim != 3:
        raise ValueError(f"h must be [B,K,L] or [P,K,L], got {tuple(h.shape)}")
    adj = topo_adj.to(h.device, dtype=h.dtype)
    n = h.shape[1]
    mask = _offdiag_mask(n, h.device)
    weights = adj.masked_fill(~mask, 0.0)
    if float(weights.sum().detach().cpu()) <= EPS:
        return h.new_tensor(0.0)
    diff2 = (h.unsqueeze(2) - h.unsqueeze(1)).pow(2).mean(dim=-1)
    return (diff2 * weights.unsqueeze(0)).sum() / (weights.sum() * h.shape[0]).clamp_min(EPS)


def topology_spread_loss(h: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Discourage the fixed topology nodes from collapsing to one latent point."""
    if h.shape[1] <= 1:
        return h.new_tensor(0.0)
    dist = torch.cdist(h, h, p=2)
    mask = _offdiag_mask(h.shape[1], h.device)
    return F.relu(float(margin) - dist[:, mask]).pow(2).mean()


def topology_roughness_score(h: torch.Tensor, topo_adj: torch.Tensor) -> torch.Tensor:
    """Per-window roughness score on the fixed topology graph."""
    adj = topo_adj.to(h.device, dtype=h.dtype)
    n = h.shape[1]
    mask = _offdiag_mask(n, h.device)
    weights = adj.masked_fill(~mask, 0.0)
    if float(weights.sum().detach().cpu()) <= EPS:
        return h.new_zeros(h.shape[0])
    diff2 = (h.unsqueeze(2) - h.unsqueeze(1)).pow(2).mean(dim=-1)
    return (diff2 * weights.unsqueeze(0)).sum(dim=(1, 2)) / weights.sum().clamp_min(EPS)


def prototype_usage_loss(alpha: torch.Tensor) -> torch.Tensor:
    """KL(mean assignment || uniform), minimized when all prototypes are used."""
    usage = alpha.mean(dim=0).clamp_min(EPS)
    uniform = usage.new_full(usage.shape, 1.0 / usage.numel())
    return (usage * (usage / uniform).log()).sum()


def _torch_kmeans_flat(x: torch.Tensor, k: int, iters: int = 25) -> torch.Tensor:
    x = x.detach().float()
    n, dim = x.shape
    if n == 0:
        raise ValueError("Cannot initialize prototypes from an empty support set.")
    k = int(k)
    if n >= k:
        idx = torch.linspace(0, n - 1, steps=k, device=x.device).long()
        centers = x[idx].clone()
    else:
        repeat = math.ceil(k / n)
        centers = x.repeat(repeat, 1)[:k].clone()
        centers = centers + 0.01 * torch.randn_like(centers)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers, p=2)
        assign = dist.argmin(dim=1)
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(dim=0)
        if torch.norm(new_centers - centers) < 1e-5:
            centers = new_centers
            break
        centers = new_centers
    return centers


@torch.no_grad()
def init_private_from_support(
    model: TopologyLatentMPFMResidualGWNM,
    support_windows: torch.Tensor,
    support_time: torch.Tensor,
    meta: torch.Tensor,
    adj: torch.Tensor,
    batch_size: int,
    device: str,
    k_private: Optional[int] = None,
    kmeans_iters: int = 25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    hs, mus, sigmas = [], [], []
    meta = meta.to(device)
    adj = adj.to(device)
    for i in range(0, support_windows.shape[0], batch_size):
        xb = support_windows[i:i + batch_size].to(device)
        tb = support_time[i:i + batch_size].to(device)
        out = model.encode(xb, meta=meta, adj=adj, time_feat=tb)
        hs.append(out["h_topo"].detach())
        mus.append(out["mu"].detach())
        sigmas.append(out["sigma"].detach())
    h = torch.cat(hs, dim=0)
    k = int(k_private or model.mpfm.k_private)
    flat = h.flatten(1)
    private = _torch_kmeans_flat(flat, k=k, iters=kmeans_iters).view(k, h.shape[1], h.shape[2])
    stats = {"h_topo": h, "h": h.mean(dim=1), "mu": torch.cat(mus, dim=0), "sigma": torch.cat(sigmas, dim=0)}
    return private.detach(), stats


def adapt_private_on_support(
    model: TopologyLatentMPFMResidualGWNM,
    private_init: torch.Tensor,
    support_h_topo: torch.Tensor,
    topo_adj: torch.Tensor,
    steps: int = 100,
    lr: float = 1e-2,
    lambda_flow: float = 0.05,
    lambda_ortho: float = 0.01,
    lambda_smooth: float = 0.02,
    lambda_anchor: float = 0.2,
) -> torch.Tensor:
    """Adapt only target private topology prototypes from normal support."""
    if steps <= 0:
        return private_init.detach()
    private0 = private_init.detach().clone()
    private = nn.Parameter(private0.clone())
    opt = torch.optim.Adam([private], lr=float(lr))
    support_h_topo = support_h_topo.detach()
    topo_adj = topo_adj.to(support_h_topo.device, support_h_topo.dtype)
    for _ in range(int(steps)):
        pinfo = model.mpfm.assign(support_h_topo, domain_idx=0, private_override=private)
        ctx = torch.cat([model.mpfm.shared, private], dim=0).mean(dim=0)
        loss_proto = (support_h_topo - pinfo["h_hat"]).pow(2).mean()
        loss_flow = model.mpfm.flow_loss(pinfo["h_hat"], ctx, topo_adj)
        flat = F.normalize(private.flatten(1), dim=-1)
        gram = flat @ flat.T
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        loss_ortho = (gram - eye).pow(2).mean()
        loss_smooth = topology_smoothness_loss(private, topo_adj)
        loss_anchor = (private - private0).pow(2).mean()
        loss = (
            loss_proto
            + float(lambda_flow) * loss_flow
            + float(lambda_ortho) * loss_ortho
            + float(lambda_smooth) * loss_smooth
            + float(lambda_anchor) * loss_anchor
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([private], 5.0)
        opt.step()
    return private.detach()

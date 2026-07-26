"""Physical-topology latent MPFM-GWNM.

The model preserves a structured latent tensor [B,K,L], where K is derived from
the physical adjacency graph instead of being blindly fixed to 30.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gwnm_model import ResidualMeasureEncoder
from .topology_mpfm_gwnm import (
    TopologyGraphBlock,
    TopologyPrototypeFlow,
    prototype_usage_loss,
    topology_roughness_score,
    topology_smoothness_loss,
    topology_spread_loss,
)

EPS = 1e-8


class PhysicalTopologyProjector(nn.Module):
    """Project variable physical node tokens to K physical topology regions."""

    def __init__(self, latent_dim: int, topo_nodes: int, topo_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.topo_nodes = int(topo_nodes)
        self.pos_emb = nn.Parameter(torch.randn(topo_nodes, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TopologyGraphBlock(latent_dim, hidden_dim=max(32, latent_dim * 2), dropout=dropout) for _ in range(topo_layers)]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, token_z: torch.Tensor, topo_assign: torch.Tensor, topo_adj: torch.Tensor) -> torch.Tensor:
        # token_z: [B,D,L], topo_assign: [K,D], topo_adj: [K,K]
        if token_z.ndim != 3:
            raise ValueError(f"token_z must be [B,D,L], got {tuple(token_z.shape)}")
        topo_assign = topo_assign.to(token_z.device, dtype=token_z.dtype)
        topo_adj = topo_adj.to(token_z.device, dtype=token_z.dtype)
        if topo_assign.shape[0] != self.topo_nodes or topo_assign.shape[1] != token_z.shape[1]:
            raise ValueError(
                f"topo_assign shape {tuple(topo_assign.shape)} incompatible with token_z {tuple(token_z.shape)} and K={self.topo_nodes}"
            )
        h = torch.einsum("kd,bdl->bkl", topo_assign, token_z)
        h = h + self.pos_emb.to(h.device, h.dtype).unsqueeze(0)
        for block in self.blocks:
            h = block(h, topo_adj)
        return self.norm(h)


class PhysicalTopologyMPFMResidualGWNM(nn.Module):
    """Residual-GWNM encoder plus physical-topology prototype/flow head."""

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
        topo_nodes: int = 4,
        topo_layers: int = 1,
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
        self.topology = PhysicalTopologyProjector(latent_dim=latent_dim, topo_nodes=topo_nodes, topo_layers=topo_layers, dropout=dropout)
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

    def encode(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        adj: torch.Tensor,
        time_feat: torch.Tensor,
        topo_assign: torch.Tensor,
        topo_adj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(x, meta=meta, adj=adj, time_feat=time_feat, return_aux=True)
        out["h_topo"] = self.topology(out["token_z"], topo_assign=topo_assign, topo_adj=topo_adj)
        out["topo_adj"] = topo_adj.to(out["h_topo"].device, out["h_topo"].dtype)
        out["topo_assign"] = topo_assign.to(out["h_topo"].device, out["h_topo"].dtype)
        out["h"] = out["h_topo"].mean(dim=1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        adj: torch.Tensor,
        time_feat: torch.Tensor,
        topo_assign: torch.Tensor,
        topo_adj: torch.Tensor,
        domain_idx: int = 0,
        private_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.encode(x, meta=meta, adj=adj, time_feat=time_feat, topo_assign=topo_assign, topo_adj=topo_adj)
        pinfo = self.mpfm.assign(out["h_topo"], domain_idx=domain_idx, private_override=private_override)
        out.update({"proto_" + k: v for k, v in pinfo.items() if k != "prototypes"})
        out["prototypes"] = pinfo["prototypes"]
        return out


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
        centers = x.repeat(repeat, 1)[:k].clone() + 0.01 * torch.randn(k, dim, device=x.device)
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
    model: PhysicalTopologyMPFMResidualGWNM,
    support_windows: torch.Tensor,
    support_time: torch.Tensor,
    meta: torch.Tensor,
    adj: torch.Tensor,
    topo_assign: torch.Tensor,
    topo_adj: torch.Tensor,
    batch_size: int,
    device: str,
    k_private: Optional[int] = None,
    kmeans_iters: int = 25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    hs, mus, sigmas = [], [], []
    meta = meta.to(device)
    adj = adj.to(device)
    topo_assign = topo_assign.to(device)
    topo_adj = topo_adj.to(device)
    for i in range(0, support_windows.shape[0], batch_size):
        xb = support_windows[i : i + batch_size].to(device)
        tb = support_time[i : i + batch_size].to(device)
        out = model.encode(xb, meta=meta, adj=adj, time_feat=tb, topo_assign=topo_assign, topo_adj=topo_adj)
        hs.append(out["h_topo"].detach())
        mus.append(out["mu"].detach())
        sigmas.append(out["sigma"].detach())
    h = torch.cat(hs, dim=0)
    k = int(k_private or model.mpfm.k_private)
    private = _torch_kmeans_flat(h.flatten(1), k=k, iters=kmeans_iters).view(k, h.shape[1], h.shape[2])
    return private.detach(), {"h_topo": h, "h": h.mean(dim=1), "mu": torch.cat(mus, dim=0), "sigma": torch.cat(sigmas, dim=0)}


def adapt_private_on_support(
    model: PhysicalTopologyMPFMResidualGWNM,
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
        loss = loss_proto + float(lambda_flow) * loss_flow + float(lambda_ortho) * loss_ortho + float(lambda_smooth) * loss_smooth + float(lambda_anchor) * loss_anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([private], 5.0)
        opt.step()
    return private.detach()

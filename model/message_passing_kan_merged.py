"""
Single-file module merged from:
- basic.py
- global_message_passing.py
- local_message_passing.py

Edits:
- All MLPs inside message passing blocks are replaced by KAN (efficient_kan implementation).
- Residual blocks in message passing are implemented with KAN as well.

Notes:
- This file still expects `utils.py` to provide `bessel_basis` and `real_sph_harm`,
  exactly as in your original basic.py.
"""

from __future__ import annotations

import math
from math import pi as PI
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG / Scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter

# Sympy is used to build closed-form spherical basis functions (same as your original basic.py)
import sympy as sym

# Keep the same dependency as your original basic.py:
from utils import bessel_basis, real_sph_harm


# =========================================================
# efficient_kan.py (inlined)
# =========================================================

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """B-spline bases: (B, in_features, grid_size + spline_order)."""
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least squares to get spline coefficients."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, B, coeff)
        B = y.transpose(0, 1)                  # (in_features, B, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, coeff, out)
        result = solution.permute(2, 0, 1)            # (out, in, coeff)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """Optional: update grid based on data distribution."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)          # (in, B, coeff)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)  # (in, coeff, out)
        unreduced = torch.bmm(splines, orig_coeff).permute(1, 0, 2)  # (B, in, out)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced))

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_act = l1_fake.sum()
        p = l1_fake / reg_act
        reg_ent = -torch.sum(p * p.log())
        return regularize_activation * reg_act + regularize_entropy * reg_ent


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)


# =========================================================
# basic.py (merged)
# =========================================================

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def MLP(channels: List[int]) -> nn.Sequential:
    """Original MLP (kept for compatibility)."""
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), SiLU())
        for i in range(1, len(channels))
    ])


class Res(nn.Module):
    """Original Res (kept for compatibility)."""
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = MLP([dim, dim, dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.mlp(x)
        return x_out + x


class ResKAN(nn.Module):
    """KAN-based residual block used in message passing after the rewrite."""
    def __init__(self, dim: int):
        super().__init__()
        self.kan = KAN([dim, dim, dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x) + x


class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        env_val = 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x

        zero = torch.zeros_like(x)
        return torch.where(x < 1, env_val, zero)


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float, envelope_exponent: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = torch.nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical: int, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}

        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda t: torch.zeros_like(t) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist: torch.Tensor, angle: torch.Tensor, idx_kj: torch.Tensor) -> torch.Tensor:
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


# =========================================================
# Message Passing (rewritten to use KAN)
# =========================================================

class Global_MessagePassing(MessagePassing):
    """
    Global message passing with KAN replacing all MLPs in the original file.
    """
    def __init__(self, config):
        super().__init__(aggr="add")
        self.dim = config.dim

        # --- KAN replacements of original MLPs ---
        self.mlp_x1 = KAN([self.dim, self.dim])
        self.mlp_x2 = KAN([self.dim, self.dim])

        self.res1 = ResKAN(self.dim)
        self.res2 = ResKAN(self.dim)
        self.res3 = ResKAN(self.dim)

        self.mlp_m = KAN([self.dim * 3, self.dim])
        self.W_edge_attr = nn.Linear(self.dim, self.dim, bias=False)

        # output head (original already uses KAN)
        self.mlp_out = KAN([self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor):
        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        x = x + self.propagate(edge_index, x=x, num_nodes=x.size(0), edge_attr=edge_attr)
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)
        return x, out, att_score

    def message(self, x_i, x_j, edge_attr, edge_index=None, num_nodes=None):
        m = torch.cat((x_i, x_j, edge_attr), dim=-1)
        m = self.mlp_m(m)
        return m * self.W_edge_attr(edge_attr)

    def update(self, aggr_out):
        return aggr_out


class Local_MessagePassing(torch.nn.Module):
    """
    Local message passing with KAN replacing all MLPs in the original file.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim

        # --- KAN replacements of original MLPs ---
        self.mlp_x1 = KAN([self.dim, self.dim])
        self.mlp_m_ji = KAN([3 * self.dim, self.dim])
        self.mlp_m_kj = KAN([3 * self.dim, self.dim])
        self.mlp_sbf = KAN([self.dim, self.dim, self.dim])

        self.lin_rbf = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = ResKAN(self.dim)
        self.res2 = ResKAN(self.dim)
        self.res3 = ResKAN(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)
        self.mlp_x2 = KAN([self.dim, self.dim])

        self.mlp_out = KAN([self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        sbf2: torch.Tensor,
        sbf1: torch.Tensor,
        idx_kj: torch.Tensor,
        idx_ji: torch.Tensor,
        idx_jj_pair: torch.Tensor,
        idx_ji_pair: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        j, i = edge_index
        idx = torch.cat((idx_kj, idx_jj_pair), dim=0)
        idx_scatter = torch.cat((idx_ji, idx_ji_pair), dim=0)
        sbf = torch.cat((sbf2, sbf1), dim=0)

        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        m_in = torch.cat([x[i], x[j], rbf], dim=-1)
        m_ji = self.mlp_m_ji(m_in)

        m_neighbor = self.mlp_m_kj(m_in) * self.lin_rbf(rbf)
        m_other = m_neighbor[idx] * self.mlp_sbf(sbf)
        m_other = scatter(m_other, idx_scatter, dim=0, dim_size=m_in.size(0), reduce='add')

        m = m_ji + m_other
        m = self.lin_rbf_out(rbf) * m

        x = x + scatter(m, i, dim=0, dim_size=x.size(0), reduce='add')
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)
        return x, out, att_score


class Local_MessagePassing_s(torch.nn.Module):
    """
    's' variant from your original local_message_passing.py, rewritten to use KAN as well.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim

        self.mlp_x1 = KAN([self.dim, self.dim])
        self.mlp_m_ji = KAN([3 * self.dim, self.dim])
        self.mlp_m_jj = KAN([3 * self.dim, self.dim])
        self.mlp_sbf = KAN([self.dim, self.dim, self.dim])
        self.lin_rbf = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = ResKAN(self.dim)
        self.res2 = ResKAN(self.dim)
        self.res3 = ResKAN(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)
        self.mlp_x2 = KAN([self.dim, self.dim])

        self.mlp_out = KAN([self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        sbf: torch.Tensor,
        idx_jj_pair: torch.Tensor,
        idx_ji_pair: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        j, i = edge_index

        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        m_in = torch.cat([x[i], x[j], rbf], dim=-1)
        m_ji = self.mlp_m_ji(m_in)

        m_neighbor = self.mlp_m_jj(m_in) * self.lin_rbf(rbf)
        m_other = m_neighbor[idx_jj_pair] * self.mlp_sbf(sbf)
        m_other = scatter(m_other, idx_ji_pair, dim=0, dim_size=m_in.size(0), reduce='add')

        m = m_ji + m_other
        m = self.lin_rbf_out(rbf) * m

        x = x + scatter(m, i, dim=0, dim_size=x.size(0), reduce='add')
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)
        return x, out, att_score

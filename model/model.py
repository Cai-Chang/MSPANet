import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops, to_dense_batch


from message_passing_kan_merged import (
    KAN,
    Global_MessagePassing,
    Local_MessagePassing,
    BesselBasisLayer,
    SphericalBasisLayer,
)


@dataclass
class Config:
    """Minimal config for PDBbind-only PAMNet."""
    dim: int = 128
    n_layer: int = 3
    cutoff_l: float = 2.0
    cutoff_g: float = 6.0


class MSPANet(nn.Module):
    """
    PDBbind-only PAMNet variant:
    - Keeps ONLY the PDBbind branch (removes QM9 / RNA-Puzzles logic)
    - Uses merged message passing (KAN-in-message-passing)
    - Edge embeddings for rbf/sbf also use KAN
    - Transformer encoder is applied per-graph (prevents cross-graph attention leakage)
    """

    def __init__(
        self,
        config: Config,
        num_spherical: int = 7,
        num_radial: int = 6,
        envelope_exponent: int = 5,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        max_num_neighbors: int = 1000,
    ):
        super().__init__()

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = float(config.cutoff_l)
        self.cutoff_g = float(config.cutoff_g)
        self.max_num_neighbors = int(max_num_neighbors)

        # PDBbind node features:
        # data.x is expected to be [N, 3 + 18] = [N, 21]
        # where first 3 dims are xyz coords, remaining 18 dims are atom descriptors
        self.init_linear = nn.Linear(18, self.dim, bias=False)

        # Transformer encoder (applied per-graph)
        # Use batch_first=True if available (PyTorch >= 1.9); otherwise fall back.
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.dim, nhead=transformer_heads, batch_first=True
            )
            self._tf_batch_first = True
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.dim, nhead=transformer_heads
            )
            self._tf_batch_first = False

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Basis layers
        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        # Edge embedding networks (KAN)
        sbf_dim = num_spherical * num_radial
        self.mlp_rbf_g = KAN([16, 32, self.dim])
        self.mlp_rbf_l = KAN([16, 32, self.dim])
        self.mlp_sbf1 = KAN([sbf_dim, 32, self.dim])
        self.mlp_sbf2 = KAN([sbf_dim, 32, self.dim])

        # Message passing stacks (KAN-inside)
        self.global_layer = nn.ModuleList([Global_MessagePassing(config) for _ in range(self.n_layer)])
        self.local_layer = nn.ModuleList([Local_MessagePassing(config) for _ in range(self.n_layer)])

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _get_edge_info(edge_index: torch.Tensor, pos: torch.Tensor):
        """Remove self-loops and compute pairwise distances for edge_index (j->i)."""
        edge_index, _ = remove_self_loops(edge_index)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist

    @staticmethod
    def _indices(edge_index: torch.Tensor, num_nodes: int):
        """
        Build triplet/pair indices used by SphericalBasisLayer (same as your original code).
        """
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))

        # --- two-hop triplets (i-j-k) ---
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        # --- one-hop pairs (i-j1-j2) around the same central edge list ---
        adj_t_col = adj_t[col]
        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)

        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j1 == j2
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def _transformer_per_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer encoder per-graph using padding/mask to prevent
        attention across different complexes in the same mini-batch.
        """
        x_dense, mask = to_dense_batch(x, batch)  # [B, S, D], mask: [B, S]
        key_padding_mask = ~mask  # True indicates padding positions

        if self._tf_batch_first:
            x_dense = self.transformer_encoder(x_dense, src_key_padding_mask=key_padding_mask)
        else:
            # old API expects [S, B, D]
            x_t = x_dense.transpose(0, 1)  # [S, B, D]
            x_t = self.transformer_encoder(x_t, src_key_padding_mask=key_padding_mask)
            x_dense = x_t.transpose(0, 1)  # [B, S, D]

        return x_dense[mask]  # back to [N, D]

    def forward(self, data):
        """
        Expects a PyG Data object for PDBbind:
          - data.x: [N, 21] (xyz + 18-d atom descriptors)
          - data.batch: [N]
        Returns:
          - out: [B] graph-level prediction
        """
        x_raw = data.x
        batch = data.batch

        # Ensure 2D feature tensor
        if x_raw.dim() == 1:
            x_raw = x_raw.unsqueeze(-1)

        # Node embedding + coordinates
        pos = x_raw[:, :3].contiguous()
        x = self.init_linear(x_raw[:, 3:])

        # Indices for energy-difference style pooling (kept from your original PDBbind branch)
        pos_index = torch.ones_like(pos[:, 0])
        neg_index = torch.ones_like(pos[:, 0]) * (-1.0)
        all_index = torch.where(pos[:, 0] > 40.0, neg_index, pos_index)

        # Global edges: radius graph with cutoff_g
        row, col = radius(
            pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=self.max_num_neighbors
        )
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, dist_g = self._get_edge_info(edge_index_g, pos)

        # Local edges: subset of global edges within cutoff_l
        mask_l = dist_g <= (torch.ones_like(dist_g, device=dist_g.device) * self.cutoff_l)
        edge_index_l = edge_index_g[:, mask_l]
        edge_index_l, dist_l = self._get_edge_info(edge_index_l, pos)

        # Transformer pre-encoding (per graph)
        x = self._transformer_per_graph(x, batch)

        # Build indices for angles / sbf
        (
            idx_i, idx_j, idx_k, idx_kj, idx_ji,
            idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair
        ) = self._indices(edge_index_l, num_nodes=x.size(0))

        # Two-hop angles (i-j-k)
        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a2 = (pos_ji * pos_kj).sum(dim=-1)
        b2 = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle2 = torch.atan2(b2, a2)

        # One-hop angles (i-j1-j2)
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a1 = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b1 = torch.cross(pos_ji_pair, pos_jj_pair).norm(dim=-1)
        angle1 = torch.atan2(b1, a1)

        # Basis embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf1 = self.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = self.sbf(dist_l, angle2, idx_kj)

        # Edge attribute projections (KAN)
        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = self.mlp_sbf1(sbf1)
        edge_attr_sbf2 = self.mlp_sbf2(sbf2)

        # Message passing blocks
        out_global, out_local = [], []
        att_score_global, att_score_local = [], []

        for layer in range(self.n_layer):
            # Residual connection around global block
            res_g = x
            x, out_g, att_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            x = x + res_g
            out_global.append(out_g)
            att_score_global.append(att_g)

            # Residual connection around local block
            res_l = x
            x, out_l, att_l = self.local_layer[layer](
                x,
                edge_attr_rbf_l,
                edge_attr_sbf2,
                edge_attr_sbf1,
                idx_kj,
                idx_ji,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            x = x + res_l
            out_local.append(out_l)
            att_score_local.append(att_l)

        # Fusion
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), -1)
        att_score = F.leaky_relu(att_score, 0.2)
        att_weight = self.softmax(att_score)

        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), -1)
        out = (out * att_weight).sum(dim=-1)      # weighted sum over features
        out = out.sum(dim=0).unsqueeze(-1)        # sum over layers -> [N, 1]

        # PDBbind pooling (kept)
        out = out * all_index.unsqueeze(-1)
        out = global_add_pool(out, batch)

        return out.view(-1)

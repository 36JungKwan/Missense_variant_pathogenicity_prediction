"""Factorized low-rank fusion for DNA, protein, and biological features."""
from __future__ import annotations

import torch
from torch import nn


class LowRankMultimodalFusion(nn.Module):
    """Three-way low-rank multiplicative fusion with a unimodal skip path.

    Each modality is projected independently. Modality-specific factors are
    reshaped to ``[batch, rank, fusion_dim]``, multiplied elementwise across
    modalities, then summed across rank. A learned linear skip from all three
    projected modalities preserves information that need not be represented
    as a three-way interaction.
    """

    def __init__(
        self,
        dna_dim: int,
        protein_dim: int,
        biological_dim: int,
        hidden_dim: int = 256,
        fusion_dim: int = 256,
        rank: int = 8,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = (dna_dim, protein_dim, biological_dim, hidden_dim, fusion_dim, rank, head_dim)
        if any(int(dim) <= 0 for dim in dims):
            raise ValueError("All low-rank model dimensions and rank must be positive")

        self.hidden_dim = int(hidden_dim)
        self.fusion_dim = int(fusion_dim)
        self.rank = int(rank)

        def projection(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.dna_projection = projection(dna_dim)
        self.protein_projection = projection(protein_dim)
        self.biological_projection = projection(biological_dim)

        # The appended constant feature gives each factor a learned bias term.
        factor_input_dim = self.hidden_dim + 1
        factor_output_dim = self.rank * self.fusion_dim
        self.dna_factor = nn.Linear(factor_input_dim, factor_output_dim)
        self.protein_factor = nn.Linear(factor_input_dim, factor_output_dim)
        self.biological_factor = nn.Linear(factor_input_dim, factor_output_dim)

        self.interaction_norm = nn.LayerNorm(self.fusion_dim)
        self.interaction_dropout = nn.Dropout(dropout)
        self.skip_projection = nn.Linear(3 * self.hidden_dim, self.fusion_dim)

        self.head_main = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, head_dim),
            nn.LayerNorm(head_dim),
        )
        self.head_skip = nn.Linear(self.fusion_dim, head_dim)
        self.head_activation = nn.GELU()
        self.head_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(head_dim, 1)

    def _factor(self, value: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        bias = value.new_ones((value.shape[0], 1))
        value = torch.cat((value, bias), dim=-1)
        return layer(value).view(value.shape[0], self.rank, self.fusion_dim)

    def forward(
        self,
        v_dna: torch.Tensor,
        v_protein: torch.Tensor,
        bio_features: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_dna = self.dna_projection(v_dna)
        h_protein = self.protein_projection(v_protein)
        biological_input = (
            torch.cat((bio_features, geom_features), dim=-1)
            if geom_features is not None
            else bio_features
        )
        h_bio = self.biological_projection(biological_input)

        dna_factor = self._factor(h_dna, self.dna_factor)
        protein_factor = self._factor(h_protein, self.protein_factor)
        biological_factor = self._factor(h_bio, self.biological_factor)
        interaction = (dna_factor * protein_factor * biological_factor).sum(dim=1)

        unimodal_skip = self.skip_projection(torch.cat((h_dna, h_protein, h_bio), dim=-1))
        fused = self.interaction_dropout(self.head_activation(
            self.interaction_norm(interaction) + unimodal_skip
        ))
        hidden = self.head_activation(self.head_main(fused) + self.head_skip(fused))
        return self.output(self.head_dropout(hidden))

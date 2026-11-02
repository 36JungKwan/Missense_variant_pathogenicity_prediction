"""Biology-conditioned FiLM fusion model for the first Module 6 experiment."""
from __future__ import annotations

import torch
from torch import nn


class BiologyConditionedFiLM(nn.Module):
    """Three-branch FiLM model for DNA, protein, and bio/geometry features.

    Biological features generate feature-wise affine parameters for residual
    transformations of the DNA and protein projections. The biological vector
    also remains an explicit input to the classifier head.
    """

    def __init__(
        self,
        dna_dim: int,
        protein_dim: int,
        biological_dim: int,
        hidden_dim: int = 512,
        bio_hidden_dim: int = 512,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if min(dna_dim, protein_dim, biological_dim, hidden_dim, bio_hidden_dim, head_dim) <= 0:
            raise ValueError("All FiLM input and hidden dimensions must be positive")

        self.hidden_dim = int(hidden_dim)
        self.bio_hidden_dim = int(bio_hidden_dim)

        def projection(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.dna_projection = projection(dna_dim, hidden_dim)
        self.protein_projection = projection(protein_dim, hidden_dim)
        self.biological_projection = projection(biological_dim, bio_hidden_dim)

        hyper_dim = max(hidden_dim, bio_hidden_dim)
        self.film_hypernetwork = nn.Sequential(
            nn.Linear(bio_hidden_dim, hyper_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hyper_dim, 4 * hidden_dim),
        )
        # Identity initialization: both residual FiLM blocks initially add 0.
        nn.init.zeros_(self.film_hypernetwork[-1].weight)
        nn.init.zeros_(self.film_hypernetwork[-1].bias)

        self.dna_film_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.protein_film_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        fused_dim = 2 * hidden_dim + bio_hidden_dim
        self.head_main = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, head_dim),
            nn.LayerNorm(head_dim),
        )
        self.head_skip = nn.Linear(fused_dim, head_dim)
        self.head_activation = nn.GELU()
        self.head_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(head_dim, 1)

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
            torch.cat([bio_features, geom_features], dim=-1)
            if geom_features is not None
            else bio_features
        )
        h_bio = self.biological_projection(biological_input)

        gamma_dna, beta_dna, gamma_protein, beta_protein = self.film_hypernetwork(h_bio).chunk(4, dim=-1)
        dna_delta = torch.nn.functional.gelu(
            gamma_dna * self.dna_film_transform(h_dna) + beta_dna
        )
        protein_delta = torch.nn.functional.gelu(
            gamma_protein * self.protein_film_transform(h_protein) + beta_protein
        )
        h_dna_film = h_dna + dna_delta
        h_protein_film = h_protein + protein_delta

        fused = torch.cat([h_dna_film, h_protein_film, h_bio], dim=-1)
        hidden = self.head_activation(self.head_main(fused) + self.head_skip(fused))
        return self.output(self.head_dropout(hidden))


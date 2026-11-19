"""Independent bio-conditioned feature gates for DNA and protein branches."""
from __future__ import annotations

import torch
from torch import nn


class IndependentModalityGating(nn.Module):
    """CoGate-Net: separate gates generated from shared DNA/protein context."""

    def __init__(
        self,
        dna_dim: int,
        protein_dim: int,
        biological_dim: int,
        hidden_dim: int = 512,
        bio_hidden_dim: int = 256,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = (
            dna_dim, protein_dim, biological_dim, hidden_dim,
            bio_hidden_dim, head_dim,
        )
        if any(int(dim) <= 0 for dim in dims):
            raise ValueError("All independent-gating input and hidden dimensions must be positive")

        def projection(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.dna_projection = projection(dna_dim, hidden_dim)
        self.protein_projection = projection(protein_dim, hidden_dim)
        self.biological_input_norm = nn.LayerNorm(biological_dim)
        self.biological_projection = projection(biological_dim, bio_hidden_dim)
        self.dna_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.protein_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.sequence_norm = nn.LayerNorm(2 * hidden_dim)

        classifier_input_dim = 2 * hidden_dim + bio_hidden_dim
        self.classifier_main = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, head_dim),
            nn.LayerNorm(head_dim),
        )
        self.classifier_skip = nn.Linear(classifier_input_dim, head_dim)
        self.classifier_activation = nn.GELU()
        self.classifier_dropout = nn.Dropout(dropout)
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
            torch.cat((bio_features, geom_features), dim=-1)
            if geom_features is not None
            else bio_features
        )
        h_bio = self.biological_projection(self.biological_input_norm(biological_input))
        shared_context = torch.cat((h_dna, h_protein), dim=-1)
        dna_gate = torch.sigmoid(self.dna_gate(shared_context))
        protein_gate = torch.sigmoid(self.protein_gate(shared_context))
        h_dna_gated = h_dna * (1.0 + dna_gate)
        h_protein_gated = h_protein * (1.0 + protein_gate)
        h_sequence = self.sequence_norm(torch.cat((h_dna_gated, h_protein_gated), dim=-1))
        fused = torch.cat((h_sequence, h_bio), dim=-1)
        hidden = self.classifier_activation(
            self.classifier_main(fused) + self.classifier_skip(fused)
        )
        return self.output(self.classifier_dropout(hidden))

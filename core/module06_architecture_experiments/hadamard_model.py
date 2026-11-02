"""Pairwise Hadamard DNA/protein fusion with modality-preserving skip paths."""
from __future__ import annotations

import torch
from torch import nn


class HadamardEmbeddingFusion(nn.Module):
    """Fuse projected DNA and protein deltas with elementwise interactions.

    The elementwise product provides pairwise cross-modal features. Direct
    projected DNA/protein paths remain in both the sequence fusion block and
    the classifier, while bio/geometry features enter through a separate path.
    """

    def __init__(
        self,
        dna_dim: int,
        protein_dim: int,
        biological_dim: int,
        hidden_dim: int = 512,
        fusion_dim: int = 512,
        bio_hidden_dim: int = 256,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = (dna_dim, protein_dim, biological_dim, hidden_dim, fusion_dim, bio_hidden_dim, head_dim)
        if any(int(dim) <= 0 for dim in dims):
            raise ValueError("All Hadamard model input and hidden dimensions must be positive")

        self.hidden_dim = int(hidden_dim)
        self.fusion_dim = int(fusion_dim)

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

        self.sequence_interaction = nn.Sequential(
            nn.Linear(3 * hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sequence_skip = nn.Linear(2 * hidden_dim, fusion_dim)
        self.sequence_activation = nn.GELU()
        self.sequence_norm = nn.LayerNorm(fusion_dim)

        classifier_input_dim = fusion_dim + bio_hidden_dim
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
        h_bio_input = (
            torch.cat((bio_features, geom_features), dim=-1)
            if geom_features is not None
            else bio_features
        )
        h_bio = self.biological_projection(h_bio_input)

        pairwise = h_dna * h_protein
        h_sequence = self.sequence_interaction(
            torch.cat((h_dna, h_protein, pairwise), dim=-1)
        )
        h_sequence = self.sequence_norm(
            h_sequence + self.sequence_skip(torch.cat((h_dna, h_protein), dim=-1))
        )
        h_sequence = self.sequence_activation(h_sequence)

        fused = torch.cat((h_sequence, h_bio), dim=-1)
        hidden = self.classifier_activation(
            self.classifier_main(fused) + self.classifier_skip(fused)
        )
        return self.output(self.classifier_dropout(hidden))

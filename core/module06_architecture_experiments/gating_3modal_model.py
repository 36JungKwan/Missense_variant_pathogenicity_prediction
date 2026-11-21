"""Three-modality feature gating over paired ref/alt sequence representations."""
from __future__ import annotations

import torch
from torch import nn


class GatingThreeModalities(nn.Module):
    """Independent DNA/protein/bio gates generated from a shared tri-modal context."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config):
        super().__init__()
        dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        feature_dropout = float(config["feature_dropout"])

        def projection(input_dim, p=dropout):
            return nn.Sequential(
                nn.Linear(int(input_dim), dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(p)
            )

        self.dna_projection = projection(dna_dim)
        self.protein_projection = projection(protein_dim)
        self.bio_projection = projection(int(bio_dim) + int(geom_dim), feature_dropout)
        context_dim = 3 * dim
        self.gates = nn.ModuleList([nn.Linear(context_dim, dim) for _ in range(3)])
        self.output_norm = nn.LayerNorm(context_dim)
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    def encode_features(self, v_dna, v_protein, bio_features, geom_features=None,
                        token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                        token_protein=None, token_protein_mask=None,
                        token_protein_variant_index=None):
        biological = (
            torch.cat((bio_features, geom_features), dim=-1)
            if geom_features is not None else bio_features
        )
        modalities = (
            self.dna_projection(v_dna),
            self.protein_projection(v_protein),
            self.bio_projection(biological),
        )
        context = torch.cat(modalities, dim=-1)
        gated = [value * (1.0 + torch.sigmoid(gate(context)))
                 for value, gate in zip(modalities, self.gates)]
        return self.output_norm(torch.cat(gated, dim=-1))

    def forward(self, v_dna, v_protein, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        features = self.encode_features(
            v_dna, v_protein, bio_features, geom_features,
            token_dna, token_dna_mask, token_dna_variant_index,
            token_protein, token_protein_mask, token_protein_variant_index,
        )
        return self.classifier(features).squeeze(-1)

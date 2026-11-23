"""Three-expert router with a joint-logit correction branch."""
from __future__ import annotations

import torch
from torch import nn


class ResidualLogitExpert(nn.Module):
    """A small residual MLP expert producing one binary logit per sample."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(value)
        hidden = self.activation(hidden + self.residual(hidden))
        return self.output(hidden)


class RouterExpertCorrection(nn.Module):
    """Router mixture of DNA/protein/bio experts plus a joint correction logit.

    The router emits three per-sample weights over independent modality experts.
    A separate correction MLP sees all three embeddings and adjusts the base
    mixture. It is a correction path, not a fourth expert.
    """

    def __init__(
        self,
        dna_dim: int,
        protein_dim: int,
        biological_dim: int,
        embedding_dim: int = 256,
        expert_hidden_dim: int = 256,
        router_hidden_dim: int = 128,
        correction_hidden_dim: int = 256,
        router_temperature: float = 1.0,
        beta_init: float = 0.1,
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = (
            dna_dim, protein_dim, biological_dim, embedding_dim,
            expert_hidden_dim, router_hidden_dim, correction_hidden_dim,
        )
        if any(int(dim) <= 0 for dim in dims):
            raise ValueError("All router-expert input and hidden dimensions must be positive")
        if float(router_temperature) <= 0:
            raise ValueError("router_temperature must be positive")

        self.router_temperature = float(router_temperature)
        self.dna_projection = self._projection(dna_dim, embedding_dim, dropout)
        self.protein_projection = self._projection(protein_dim, embedding_dim, dropout)
        self.biological_projection = self._projection(biological_dim, embedding_dim, dropout)

        self.dna_expert = ResidualLogitExpert(embedding_dim, expert_hidden_dim, dropout)
        self.protein_expert = ResidualLogitExpert(embedding_dim, expert_hidden_dim, dropout)
        self.biological_expert = ResidualLogitExpert(embedding_dim, expert_hidden_dim, dropout)

        joint_dim = 3 * embedding_dim
        self.router = nn.Sequential(
            nn.Linear(joint_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 3),
        )
        # Begin with equal expert weights rather than a random routing bias.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.correction = nn.Sequential(
            nn.Linear(joint_dim, correction_hidden_dim),
            nn.LayerNorm(correction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(correction_hidden_dim, correction_hidden_dim),
            nn.LayerNorm(correction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(correction_hidden_dim, 1),
        )
        # Keep nonzero correction output initialization so beta initialized at
        # zero can receive a gradient and then enable learning in this branch.
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    @staticmethod
    def _projection(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        v_dna: torch.Tensor,
        v_protein: torch.Tensor,
        bio_features: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        biological_input = (
            torch.cat((bio_features, geom_features), dim=-1)
            if geom_features is not None
            else bio_features
        )
        h_dna = self.dna_projection(v_dna)
        h_protein = self.protein_projection(v_protein)
        h_bio = self.biological_projection(biological_input)
        joint = torch.cat((h_dna, h_protein, h_bio), dim=-1)

        expert_logits = torch.cat((
            self.dna_expert(h_dna),
            self.protein_expert(h_protein),
            self.biological_expert(h_bio),
        ), dim=-1)
        router_weights = torch.softmax(
            self.router(joint) / self.router_temperature,
            dim=-1,
        )
        base_logit = (router_weights * expert_logits).sum(dim=-1, keepdim=True)
        delta_joint_logit = self.correction(joint)
        final_logit = base_logit + self.beta * delta_joint_logit

        if return_components:
            return {
                "logit": final_logit,
                "base_logit": base_logit,
                "expert_logits": expert_logits,
                "router_weights": router_weights,
                "delta_joint_logit": delta_joint_logit,
                "beta": self.beta,
            }
        return final_logit
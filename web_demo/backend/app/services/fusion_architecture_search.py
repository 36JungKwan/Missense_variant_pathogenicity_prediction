"""Shared neural fusion architecture used by DL search and seed stages."""
import torch
import torch.nn as nn

class FusionArchitectureVariant(nn.Module):
    """Structured fusion architecture candidates with controlled training hyperparameters."""
    def __init__(self, dna_dim, prot_dim, bio_dim, geom_dim, active_modalities,
                 variant, fusion_dim, dropout, head_scale=1.0, fusion_hparams=None,
                 dna_embedding_dim=None, prot_embedding_dim=None, bio_embedding_dim=128):
        super().__init__()
        self.variant = variant
        self.fusion_hparams = fusion_hparams or {}
        self.active_mods = [str(x).lower() for x in active_modalities]
        self.has_bio = bool({"bio", "bio_core", "spliceai"} & set(self.active_mods))
        self.has_geom = "geom" in self.active_mods
        projection_norm = nn.Identity if variant == "concat_no_projection_norm" else nn.LayerNorm
        dna_proj_dim = fusion_dim * 2 if variant == "concat_unequal_proj" else fusion_dim
        dna_embedding_dim = int(dna_embedding_dim or dna_proj_dim)
        prot_embedding_dim = int(prot_embedding_dim or fusion_dim)
        bio_embedding_dim = int(bio_embedding_dim or 128)
        self.dna_proj = nn.Sequential(
            nn.Linear(dna_dim, dna_embedding_dim), projection_norm(dna_embedding_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, prot_embedding_dim), projection_norm(prot_embedding_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # Each modality first gets its own configurable bottleneck width; these
        # adapters align the representations for fusion families that require a
        # shared hidden size (attention, transformer, gating).
        self.dna_fusion_align = (
            nn.Identity() if dna_embedding_dim == dna_proj_dim else nn.Linear(dna_embedding_dim, dna_proj_dim)
        )
        self.prot_fusion_align = (
            nn.Identity() if prot_embedding_dim == fusion_dim else nn.Linear(prot_embedding_dim, fusion_dim)
        )
        bio_geom_dim = (bio_dim if self.has_bio else 0) + (geom_dim if self.has_geom else 0)
        self.has_ml_features = bio_geom_dim > 0
        self.separate_feature_tokens = variant == "transformer_separate_feature_tokens_2layer"
        self.bio_as_token = variant in {
            "transformer_bio_token", "transformer_bio_token_3layer", "transformer_separate_feature_tokens_2layer",
            "concat_bio_token",
        }
        if self.has_ml_features and self.bio_as_token and self.separate_feature_tokens:
            self.bio_token_proj = nn.Sequential(
                nn.Linear(bio_dim, bio_embedding_dim), nn.LayerNorm(bio_embedding_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(bio_embedding_dim, fusion_dim),
            ) if self.has_bio else None
            self.geom_token_proj = nn.Sequential(
                nn.Linear(geom_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout),
            ) if self.has_geom else None
        elif self.has_ml_features and self.bio_as_token:
            self.bio_token_proj = nn.Sequential(
                nn.Linear(bio_geom_dim, bio_embedding_dim), nn.LayerNorm(bio_embedding_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(bio_embedding_dim, fusion_dim),
            )
        elif self.has_ml_features:
            self.bio_geom_upsample = nn.Sequential(
                nn.Linear(bio_geom_dim, bio_embedding_dim), nn.LayerNorm(bio_embedding_dim),
                nn.GELU(), nn.Dropout(dropout),
            )
            self.bio_output_dim = bio_embedding_dim

        if variant.startswith("concat_"):
            extra = int("product" in variant) + int("absdiff" in variant) + int("sum" in variant) + int("bilinear" in variant)
            seq_dim = (fusion_dim * 3 if variant == "concat_unequal_proj" else fusion_dim * (2 + extra))
            if variant == "concat_bio_token":
                seq_dim += fusion_dim
            if "bilinear" in variant:
                bilinear_rank = int(self.fusion_hparams.get("concat_bilinear_rank", min(64, max(16, fusion_dim // 4))))
                self.bilinear_dna = nn.Linear(fusion_dim, bilinear_rank, bias=False)
                self.bilinear_prot = nn.Linear(fusion_dim, bilinear_rank, bias=False)
                self.bilinear_out = nn.Linear(bilinear_rank, fusion_dim, bias=False)
        elif variant == "cross_gated":
            self.gate_dna = nn.Linear(fusion_dim, fusion_dim)
            self.gate_prot = nn.Linear(fusion_dim, fusion_dim)
            self.cross_dna_ln = nn.LayerNorm(fusion_dim)
            self.cross_prot_ln = nn.LayerNorm(fusion_dim)
            seq_dim = fusion_dim * 2
        elif variant == "cross_film":
            self.film_from_prot = nn.Linear(fusion_dim, fusion_dim * 2)
            self.film_from_dna = nn.Linear(fusion_dim, fusion_dim * 2)
            seq_dim = fusion_dim * 2
        elif variant.startswith("cross_") and ("token_attention" in variant or variant.endswith("token_oneway") or variant.endswith("token_attnpool") or variant.endswith("token_maxpool")):
            token_spec = variant.split("_")[1]
            self.token_count = int(token_spec[:-5]) if token_spec.endswith("token") and token_spec[:-5].isdigit() else 8
            self.one_way_attention = variant.endswith("oneway")
            self.cross_pool = "attention" if variant.endswith("attnpool") else ("max" if variant.endswith("maxpool") else "mean")
            if fusion_dim % self.token_count:
                raise ValueError(f"fusion_dim must be divisible by {self.token_count} tokens")
            self.token_dim = fusion_dim // self.token_count
            self.cross_pre_norm = "_prenorm" in variant
            self.cross_input_norm_dna = nn.LayerNorm(self.token_dim) if self.cross_pre_norm else nn.Identity()
            self.cross_input_norm_prot = nn.LayerNorm(self.token_dim) if self.cross_pre_norm else nn.Identity()
            requested_heads = int(self.fusion_hparams.get("cross_attention_heads", min(8, self.token_dim)))
            self.attn_heads = min(requested_heads, self.token_dim)
            while self.token_dim % self.attn_heads:
                self.attn_heads -= 1
            attention_dropout = float(self.fusion_hparams.get("cross_attention_dropout", dropout))
            self.attn_prot_to_dna = nn.MultiheadAttention(
                self.token_dim, num_heads=self.attn_heads, batch_first=True, dropout=attention_dropout
            )
            self.attn_dna_to_prot = (
                self.attn_prot_to_dna if "shared_attention" in variant else
                nn.MultiheadAttention(self.token_dim, num_heads=self.attn_heads, batch_first=True, dropout=attention_dropout)
            )
            self.token_dna_ln = nn.LayerNorm(self.token_dim)
            self.cross_two_blocks = "_2block" in variant
            if self.cross_two_blocks:
                self.attn_prot_to_dna_2 = nn.MultiheadAttention(self.token_dim, num_heads=self.attn_heads, batch_first=True, dropout=attention_dropout)
                self.attn_dna_to_prot_2 = nn.MultiheadAttention(self.token_dim, num_heads=self.attn_heads, batch_first=True, dropout=attention_dropout)
            self.token_prot_ln = nn.LayerNorm(self.token_dim)
            self.token_pool_score = nn.Linear(self.token_dim, 1) if self.cross_pool == "attention" else None
            token_pool_dim = self.token_dim * (1 if self.one_way_attention else 2)
            if "ffn4" in variant:
                self.token_out_proj = nn.Sequential(
                    nn.Linear(token_pool_dim, fusion_dim * 4), nn.LayerNorm(fusion_dim * 4),
                    nn.GELU(), nn.Dropout(dropout), nn.Linear(fusion_dim * 4, fusion_dim * 2),
                )
            else:
                self.token_out_proj = nn.Sequential(
                    nn.Linear(token_pool_dim, fusion_dim * 2),
                    nn.LayerNorm(fusion_dim * 2), nn.GELU(), nn.Dropout(dropout),
                )
            seq_dim = fusion_dim * 2
        elif variant.startswith("transformer_"):
            n_tokens = 2 + (int(self.has_bio) + int(self.has_geom) if self.separate_feature_tokens else int(self.bio_as_token))
            self.modality_embed = None if "no_modality_embed" in variant else nn.Parameter(
                torch.randn(1, n_tokens, fusion_dim) * 0.02
            )
            default_heads = 4 if "_4heads" in variant else (16 if "_16heads" in variant else 8)
            n_heads = int(self.fusion_hparams.get("transformer_num_heads", default_heads))
            while fusion_dim % n_heads:
                n_heads -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim, nhead=n_heads, batch_first=True, dropout=dropout,
                norm_first="_preln" in variant,
                dim_feedforward=fusion_dim * int(self.fusion_hparams.get("transformer_ffn_expansion", 4)),
            )
            if "1layer" in variant:
                default_depth = 1
            elif "3layer" in variant:
                default_depth = 3
            elif "4layer" in variant:
                default_depth = 4
            else:
                default_depth = 2
            depth = int(self.fusion_hparams.get("transformer_depth", default_depth))
            self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
            self.transformer_pool = "attention" if "attnpool" in variant else ("cls" if "clspool" in variant else "mean")
            self.transformer_pool_query = nn.Parameter(torch.randn(fusion_dim) * 0.02) if self.transformer_pool == "attention" else None
            self.transformer_cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02) if self.transformer_pool == "cls" else None
            seq_dim = fusion_dim * n_tokens if self.transformer_pool == "mean" else fusion_dim * 2
        elif variant in {"gating_with_skip", "gating_scalar_skip", "gating_dual_scalar", "gating_residual", "gating_product", "gating_bio_conditioned", "gating_highway", "gating_tanh"} or variant.startswith("gating_softmax"):
            gate_dim = 2 if variant == "gating_dual_scalar" or variant.startswith("gating_softmax") else (1 if variant == "gating_scalar_skip" else fusion_dim)
            gate_input_dim = fusion_dim * 2 + (bio_geom_dim if variant == "gating_bio_conditioned" else 0)
            self.gate = nn.Linear(gate_input_dim, gate_dim)
            seq_dim = fusion_dim * 3
        elif variant == "gating_interaction":
            self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
            seq_dim = fusion_dim * 3
        else:
            raise ValueError(f"Unknown architecture variant: {variant}")

        self.post_fusion_ln = nn.LayerNorm(seq_dim)
        global_dim = seq_dim + (0 if self.bio_as_token else (self.bio_output_dim if self.has_ml_features else 0))
        if variant == "concat_head1":
            head_dims = [128]
        elif variant == "concat_head3":
            head_dims = [512, 512, 512]
        else:
            head_dims = [256, 128]
        head_dims = [max(32, int(dim * head_scale)) for dim in head_dims]
        head_layers, previous_dim = [], global_dim
        for head_dim in head_dims:
            head_layers.extend([
                nn.Linear(previous_dim, head_dim), nn.LayerNorm(head_dim),
                nn.GELU(), nn.Dropout(dropout),
            ])
            previous_dim = head_dim
        self.classifier_hidden = nn.Sequential(*head_layers)
        self.shortcut = nn.Linear(global_dim, previous_dim)
        self.output_head = nn.Linear(previous_dim, 1)

    def _pool_tokens(self, tokens):
        if self.cross_pool == "max":
            return tokens.max(dim=1).values
        if self.cross_pool == "attention":
            weights = torch.softmax(self.token_pool_score(tokens).squeeze(-1), dim=1)
            return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return tokens.mean(dim=1)

    def _bio_geom_input(self, bio_features, geom_features):
        values = []
        if self.has_bio:
            values.append(bio_features)
        if self.has_geom:
            values.append(geom_features)
        return torch.cat(values, dim=-1)

    def forward(self, v_dna=None, v_prot=None, bio_features=None, geom_features=None, return_features=False):
        h_dna = self.dna_fusion_align(self.dna_proj(v_dna))
        h_prot = self.prot_fusion_align(self.prot_proj(v_prot))
        variant = self.variant
        if variant.startswith("concat_"):
            parts = [h_dna, h_prot]
            if "product" in variant:
                parts.append(h_dna * h_prot)
            if "absdiff" in variant:
                parts.append(torch.abs(h_dna - h_prot))
            if "sum" in variant:
                parts.append(h_dna + h_prot)
            if "bilinear" in variant:
                parts.append(self.bilinear_out(self.bilinear_dna(h_dna) * self.bilinear_prot(h_prot)))
            if variant == "concat_bio_token":
                parts.append(self.bio_token_proj(self._bio_geom_input(bio_features, geom_features)))
            seq = torch.cat(parts, dim=-1)
        elif variant == "cross_gated":
            gate_dna = torch.sigmoid(self.gate_dna(h_prot))
            gate_prot = torch.sigmoid(self.gate_prot(h_dna))
            seq = torch.cat([
                self.cross_dna_ln(h_dna + gate_dna * h_prot),
                self.cross_prot_ln(h_prot + gate_prot * h_dna),
            ], dim=-1)
        elif variant == "cross_film":
            gamma_dna, beta_dna = self.film_from_prot(h_prot).chunk(2, dim=-1)
            gamma_prot, beta_prot = self.film_from_dna(h_dna).chunk(2, dim=-1)
            conditioned_dna = h_dna * (1.0 + torch.tanh(gamma_dna)) + beta_dna
            conditioned_prot = h_prot * (1.0 + torch.tanh(gamma_prot)) + beta_prot
            seq = torch.cat([conditioned_dna, conditioned_prot], dim=-1)
        elif variant.startswith("cross_") and ("token_attention" in variant or variant.endswith("token_oneway") or variant.endswith("token_attnpool") or variant.endswith("token_maxpool")):
            dna_tokens = h_dna.reshape(h_dna.shape[0], self.token_count, self.token_dim)
            prot_tokens = h_prot.reshape(h_prot.shape[0], self.token_count, self.token_dim)
            dna_tokens_for_attention = self.cross_input_norm_dna(dna_tokens)
            prot_tokens_for_attention = self.cross_input_norm_prot(prot_tokens)
            prot_to_dna, _ = self.attn_prot_to_dna(
                query=prot_tokens_for_attention, key=dna_tokens_for_attention,
                value=dna_tokens_for_attention, need_weights=False
            )
            prot_conditioned = self.token_prot_ln(prot_tokens + prot_to_dna)
            if self.one_way_attention:
                pooled_parts = [self._pool_tokens(prot_conditioned)]
            else:
                dna_to_prot, _ = self.attn_dna_to_prot(
                    query=dna_tokens_for_attention, key=prot_tokens_for_attention,
                    value=prot_tokens_for_attention, need_weights=False
                )
                dna_conditioned = self.token_dna_ln(dna_tokens + dna_to_prot)
                if self.cross_two_blocks:
                    prot_to_dna_2, _ = self.attn_prot_to_dna_2(
                        query=prot_conditioned, key=dna_conditioned, value=dna_conditioned, need_weights=False
                    )
                    dna_to_prot_2, _ = self.attn_dna_to_prot_2(
                        query=dna_conditioned, key=prot_conditioned, value=prot_conditioned, need_weights=False
                    )
                    prot_conditioned = self.token_prot_ln(prot_conditioned + prot_to_dna_2)
                    dna_conditioned = self.token_dna_ln(dna_conditioned + dna_to_prot_2)
                pooled_parts = [self._pool_tokens(prot_conditioned), self._pool_tokens(dna_conditioned)]
            pooled = torch.cat(pooled_parts, dim=-1)
            seq = self.token_out_proj(pooled)
        elif variant.startswith("transformer_"):
            tokens = [h_dna, h_prot]
            if self.bio_as_token:
                if self.separate_feature_tokens:
                    if self.has_bio:
                        tokens.append(self.bio_token_proj(bio_features))
                    if self.has_geom:
                        tokens.append(self.geom_token_proj(geom_features))
                else:
                    tokens.append(self.bio_token_proj(self._bio_geom_input(bio_features, geom_features)))
            token_tensor = torch.stack(tokens, dim=1)
            if self.modality_embed is not None:
                token_tensor = token_tensor + self.modality_embed[:, :len(tokens), :]
            if self.transformer_pool == "cls":
                cls_token = self.transformer_cls_token.expand(token_tensor.shape[0], -1, -1)
                token_tensor = torch.cat([cls_token, token_tensor], dim=1)
            transformed = self.transformer(token_tensor)
            if self.transformer_pool == "mean":
                seq = transformed.reshape(transformed.shape[0], -1)
            elif self.transformer_pool == "cls":
                seq = torch.cat([transformed[:, 0], transformed[:, 1:].mean(dim=1)], dim=-1)
            else:
                weights = torch.softmax(torch.einsum("btd,d->bt", transformed, self.transformer_pool_query), dim=1)
                pooled = torch.sum(transformed * weights.unsqueeze(-1), dim=1)
                seq = torch.cat([pooled, transformed.mean(dim=1)], dim=-1)
        elif variant in {"gating_with_skip", "gating_scalar_skip", "gating_dual_scalar", "gating_residual", "gating_product", "gating_bio_conditioned", "gating_highway", "gating_tanh"} or variant.startswith("gating_softmax"):
            gate_inputs = [h_dna, h_prot]
            if variant == "gating_bio_conditioned":
                gate_inputs.append(self._bio_geom_input(bio_features, geom_features))
            gate = self.gate(torch.cat(gate_inputs, dim=-1))
            if variant == "gating_tanh":
                gate = torch.tanh(gate)
                mixed = gate * h_dna + (1.0 - gate) * h_prot
            elif variant.startswith("gating_softmax"):
                temperature = float(self.fusion_hparams.get("gating_temperature", 0.5 if variant.endswith("temp05") else (2.0 if variant.endswith("temp2") else 1.0)))
                gate_dna, gate_prot = torch.softmax(gate / temperature, dim=-1).chunk(2, dim=-1)
                mixed = gate_dna * h_dna + gate_prot * h_prot
            elif variant == "gating_dual_scalar":
                gate_dna, gate_prot = torch.sigmoid(gate).chunk(2, dim=-1)
                mixed = gate_dna * h_dna + gate_prot * h_prot
            else:
                gate = torch.sigmoid(gate)
                mixed = gate * h_dna + (1.0 - gate) * h_prot
            if variant == "gating_residual":
                seq = torch.cat([mixed, h_dna + h_prot, torch.abs(h_dna - h_prot)], dim=-1)
            elif variant == "gating_product":
                seq = torch.cat([mixed, h_dna, h_dna * h_prot], dim=-1)
            elif variant == "gating_highway":
                seq = torch.cat([mixed + h_dna, h_dna, h_prot], dim=-1)
            else:
                seq = torch.cat([mixed, h_dna, h_prot], dim=-1)
        else:  # gating_interaction
            gate = torch.sigmoid(self.gate(torch.cat([h_dna, h_prot], dim=-1)))
            mixed = gate * h_dna + (1.0 - gate) * h_prot
            seq = torch.cat([mixed, h_dna * h_prot, torch.abs(h_dna - h_prot)], dim=-1)

        features = [self.post_fusion_ln(seq)]
        if self.has_ml_features and not self.bio_as_token:
            features.append(self.bio_geom_upsample(self._bio_geom_input(bio_features, geom_features)))
        global_features = torch.cat(features, dim=-1)
        if return_features:
            return global_features
        return self.output_head(
            self.classifier_hidden(global_features) + self.shortcut(global_features)
        )

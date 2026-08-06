import torch
import torch.nn as nn

class MultiStrategyFusionModel(nn.Module):
    """
    Module 5: Lớp Hòa trộn Đa chiến lược & Mạng Phân loại.
    Bản vá V2: Tích hợp Modality Embeddings và Post-Fusion LayerNorm.
    """
    def __init__(self, 
                 dna_in_dim: int, 
                 prot_in_dim: int, 
                 fusion_dim: int = 512,
                 bio_geom_dim: int = 19, # Chốt cứng 19 features (11 Bio + 8 Geom)
                 bio_upsample_dim: int = 128,
                 fusion_strategy: str = 'concat',
                 dropout: float = 0.2):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy.lower()
        self.fusion_dim = fusion_dim
        
        # =====================================================================
        # 1. BƯỚC ĐỒNG BỘ SỐ CHIỀU (Linear Projections)
        # =====================================================================
        self.dna_proj = nn.Sequential(
            nn.Linear(dna_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # =====================================================================
        # 2. KHỐI BẢO VỆ ĐẶC TRƯNG SINH HỌC & HÌNH HỌC (Upsampling)
        # =====================================================================
        self.bio_geom_upsample = nn.Sequential(
            nn.Linear(bio_geom_dim, bio_upsample_dim),
            nn.LayerNorm(bio_upsample_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # =====================================================================
        # 3. KHÔNG GIAN HÒA TRỘN (FUSION ABLATION SPACE)
        # =====================================================================
        if self.fusion_strategy == 'concat':
            lm_out_dim = fusion_dim * 2
            
        elif self.fusion_strategy == 'cross_attention':
            self.attn_p2d = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=dropout)
            self.attn_d2p = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=dropout)
            lm_out_dim = fusion_dim * 2
            
        elif self.fusion_strategy == 'transformer':
            # [BẢN VÁ] Modality Embedding để phân biệt DNA (0) và Protein (1)
            self.modality_embed = nn.Parameter(torch.randn(1, 2, fusion_dim) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=8, batch_first=True, dropout=dropout)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            lm_out_dim = fusion_dim * 2
            
        elif self.fusion_strategy == 'gating':
            self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
            lm_out_dim = fusion_dim
            
        else:
            raise ValueError(f"[LỖI] Không hỗ trợ chiến lược: {self.fusion_strategy}")
            
        # [BẢN VÁ] Ổn định phương sai hậu Hòa trộn
        self.post_fusion_ln = nn.LayerNorm(lm_out_dim)
        
        # =====================================================================
        # 4. BỘ PHÂN LOẠI (CLASSIFIER - RESIDUAL MLP)
        # =====================================================================
        self.global_dim = lm_out_dim + bio_upsample_dim
        
        self.classifier_hidden = nn.Sequential(
            nn.Linear(self.global_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.shortcut = nn.Linear(self.global_dim, 128)
        self.output_head = nn.Linear(128, 1)

    def forward(self, v_dna, v_prot, bio_features, geom_features, return_features=False):
        batch_size = v_dna.size(0)
        
        # 1. Ép chiều (B, 512)
        h_dna = self.dna_proj(v_dna)
        h_prot = self.prot_proj(v_prot)
        
        # 2. Xử lý Bio/Geom -> (B, 128)
        f_bio_geom = torch.cat([bio_features, geom_features], dim=-1)
        h_bio = self.bio_geom_upsample(f_bio_geom)
        
        # 3. Hòa trộn (Fusion)
        if self.fusion_strategy == 'concat':
            f_lm = torch.cat([h_dna, h_prot], dim=-1)
            
        elif self.fusion_strategy == 'cross_attention':
            q_dna, q_prot = h_dna.unsqueeze(1), h_prot.unsqueeze(1)
            
            attn_p2d, _ = self.attn_p2d(query=q_prot, key=q_dna, value=q_dna)
            attn_d2p, _ = self.attn_d2p(query=q_dna, key=q_prot, value=q_prot)
            
            f_lm = torch.cat([attn_p2d.squeeze(1), attn_d2p.squeeze(1)], dim=-1)
            
        elif self.fusion_strategy == 'transformer':
            # [BẢN VÁ] Cộng thêm Modality Embedding trước khi vào Transformer
            seq_tokens = torch.stack([h_dna, h_prot], dim=1) 
            seq_tokens = seq_tokens + self.modality_embed
            
            trans_out = self.transformer(seq_tokens)
            f_lm = trans_out.reshape(batch_size, -1)
            
        elif self.fusion_strategy == 'gating':
            combined = torch.cat([h_dna, h_prot], dim=-1)
            alpha = torch.sigmoid(self.gate(combined))
            f_lm = alpha * h_dna + (1 - alpha) * h_prot

        # [BẢN VÁ] Ổn định phương sai
        f_lm = self.post_fusion_ln(f_lm)

        # 4. Gộp Toàn cục
        f_global = torch.cat([f_lm, h_bio], dim=-1)
        
        # Mode XGBoost Hybrid
        if return_features:
            return f_global
            
        # 5. Phân loại End-to-End
        hidden = self.classifier_hidden(f_global)
        res = self.shortcut(f_global)
        
        out_features = hidden + res
        logits = self.output_head(out_features)
        
        return logits
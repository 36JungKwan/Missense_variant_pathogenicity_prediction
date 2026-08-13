import torch
import torch.nn as nn

class MultiStrategyFusionModel(nn.Module):
    """
    Module 5: Lớp Hòa trộn Đa chiến lược & Mạng Phân loại.
    Bản vá V4 (Ablation-Ready & Bulletproof): 
    - Hỗ trợ Bật/Tắt linh hoạt từng phương thức (DNA, Prot, Bio, Geom).
    - Tự động thoái hóa khối Fusion (Graceful Degradation).
    - Lá chắn bảo vệ dữ liệu đầu vào (Input Validation Shield).
    """
    def __init__(self, 
                 dna_in_dim: int, 
                 prot_in_dim: int, 
                 active_modalities: list = None,
                 fusion_dim: int = 512,
                 bio_upsample_dim: int = 128,
                 fusion_strategy: str = 'concat',
                 dropout: float = 0.2):
        super().__init__()
        
        # Mặc định bật toàn bộ nếu không truyền list
        if active_modalities is None:
            active_modalities = ['dna', 'prot', 'bio', 'geom']
        self.active_mods = [m.lower() for m in active_modalities]
        
        self.fusion_strategy = fusion_strategy.lower()
        self.fusion_dim = fusion_dim
        
        # Xác định sự hiện diện của các nhánh
        self.has_dna = 'dna' in self.active_mods
        self.has_prot = 'prot' in self.active_mods
        self.has_bio = 'bio' in self.active_mods
        self.has_geom = 'geom' in self.active_mods
        
        self.num_seq_mods = int(self.has_dna) + int(self.has_prot)
        
        # =====================================================================
        # 1. BƯỚC ĐỒNG BỘ SỐ CHIỀU (Khởi tạo động)
        # =====================================================================
        if self.has_dna:
            self.dna_proj = nn.Sequential(
                nn.Linear(dna_in_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        if self.has_prot:
            self.prot_proj = nn.Sequential(
                nn.Linear(prot_in_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        # =====================================================================
        # 2. KHỐI BẢO VỆ ĐẶC TRƯNG SINH HỌC & HÌNH HỌC (Khởi tạo động)
        # =====================================================================
        bio_geom_dim = 0
        if self.has_bio: bio_geom_dim += 11
        if self.has_geom: bio_geom_dim += 8
        self.has_ml_features = (bio_geom_dim > 0)
        
        actual_bio_upsample_dim = 0
        if self.has_ml_features:
            self.bio_geom_upsample = nn.Sequential(
                nn.Linear(bio_geom_dim, bio_upsample_dim),
                nn.LayerNorm(bio_upsample_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            actual_bio_upsample_dim = bio_upsample_dim
            
        # =====================================================================
        # 3. KHÔNG GIAN HÒA TRỘN (Xử lý thoái hóa Toán học)
        # =====================================================================
        lm_out_dim = 0
        
        if self.num_seq_mods == 2:
            # A. CÓ ĐỦ 2 CHUỖI -> Chạy Fusion bình thường
            if self.fusion_strategy == 'concat':
                lm_out_dim = fusion_dim * 2
                
            elif self.fusion_strategy == 'cross_attention':
                self.attn_p2d = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=dropout)
                self.attn_d2p = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=dropout)
                lm_out_dim = fusion_dim * 2
                
            elif self.fusion_strategy == 'transformer':
                self.modality_embed = nn.Parameter(torch.randn(1, 2, fusion_dim) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=8, batch_first=True, dropout=dropout)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
                lm_out_dim = fusion_dim * 2
                
            elif self.fusion_strategy == 'gating':
                self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
                lm_out_dim = fusion_dim
                
            else:
                raise ValueError(f"[LỖI] Không hỗ trợ chiến lược: {self.fusion_strategy}")
                
        elif self.num_seq_mods == 1:
            # B. CHỈ CÓ 1 CHUỖI -> Thoái hóa thành Pass-through (Bỏ qua khối Fusion)
            lm_out_dim = fusion_dim
            
        # Khởi tạo LayerNorm hậu Hòa trộn (Nếu có ít nhất 1 chuỗi)
        if lm_out_dim > 0:
            self.post_fusion_ln = nn.LayerNorm(lm_out_dim)
            
        # =====================================================================
        # 4. BỘ PHÂN LOẠI (CLASSIFIER - Khởi tạo theo số chiều thực tế)
        # =====================================================================
        self.global_dim = lm_out_dim + actual_bio_upsample_dim
        
        if self.global_dim == 0:
            raise ValueError("[LỖI CHÍ MẠNG] Không có nhánh dữ liệu nào được kích hoạt!")
            
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

    def forward(self, v_dna=None, v_prot=None, bio_features=None, geom_features=None, return_features=False):
        # =====================================================================
        # LÁ CHẮN BẢO VỆ DỮ LIỆU ĐẦU VÀO (INPUT VALIDATION SHIELD)
        # =====================================================================
        if self.has_dna and v_dna is None: 
            raise ValueError("[LỖI] Cấu hình yêu cầu phương thức DNA nhưng v_dna bị None!")
        if self.has_prot and v_prot is None: 
            raise ValueError("[LỖI] Cấu hình yêu cầu phương thức Protein nhưng v_prot bị None!")
        if self.has_bio and bio_features is None: 
            raise ValueError("[LỖI] Cấu hình yêu cầu phương thức Bio nhưng bio_features bị None!")
        if self.has_geom and geom_features is None: 
            raise ValueError("[LỖI] Cấu hình yêu cầu phương thức Geom nhưng geom_features bị None!")

        # 1. Ép chiều
        h_dna = self.dna_proj(v_dna) if self.has_dna else None
        h_prot = self.prot_proj(v_prot) if self.has_prot else None
        
        # 2. Xử lý Bio/Geom
        h_bio = None
        if self.has_ml_features:
            f_bio_geom_list = []
            if self.has_bio: f_bio_geom_list.append(bio_features)
            if self.has_geom: f_bio_geom_list.append(geom_features)
            
            f_bio_geom = torch.cat(f_bio_geom_list, dim=-1)
            h_bio = self.bio_geom_upsample(f_bio_geom)
            
        # 3. Hòa trộn (Fusion) với logic thoái hóa
        f_lm = None
        if self.num_seq_mods == 2:
            if self.fusion_strategy == 'concat':
                f_lm = torch.cat([h_dna, h_prot], dim=-1)
                
            elif self.fusion_strategy == 'cross_attention':
                q_dna, q_prot = h_dna.unsqueeze(1), h_prot.unsqueeze(1)
                attn_p2d, _ = self.attn_p2d(query=q_prot, key=q_dna, value=q_dna)
                attn_d2p, _ = self.attn_d2p(query=q_dna, key=q_prot, value=q_prot)
                f_lm = torch.cat([attn_p2d.squeeze(1), attn_d2p.squeeze(1)], dim=-1)
                
            elif self.fusion_strategy == 'transformer':
                seq_tokens = torch.stack([h_dna, h_prot], dim=1) 
                seq_tokens = seq_tokens + self.modality_embed
                trans_out = self.transformer(seq_tokens)
                # [BẢN VÁ] Reshape an toàn không phụ thuộc biến ngoại vi
                f_lm = trans_out.reshape(trans_out.size(0), -1) 
                
            elif self.fusion_strategy == 'gating':
                combined = torch.cat([h_dna, h_prot], dim=-1)
                alpha = torch.sigmoid(self.gate(combined))
                f_lm = alpha * h_dna + (1 - alpha) * h_prot
                
        elif self.num_seq_mods == 1:
            # Thoái hóa: Trực tiếp lấy nhánh duy nhất đang tồn tại
            f_lm = h_dna if h_dna is not None else h_prot
            
        if f_lm is not None:
            f_lm = self.post_fusion_ln(f_lm)

        # 4. Gộp Toàn cục
        global_list = []
        if f_lm is not None: global_list.append(f_lm)
        if h_bio is not None: global_list.append(h_bio)
        
        f_global = torch.cat(global_list, dim=-1)
        
        if return_features:
            return f_global
            
        # 5. Phân loại End-to-End
        hidden = self.classifier_hidden(f_global)
        res = self.shortcut(f_global)
        
        out_features = hidden + res
        logits = self.output_head(out_features)
        
        return logits
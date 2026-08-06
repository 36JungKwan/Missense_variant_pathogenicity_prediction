import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class VariantFusionDataset(Dataset):
    """
    Module 5 Dataset: Điểm hội tụ Đa phương thức (Multi-modal Fusion).
    Đã tối ưu hóa O(1) Pre-computation để đạt tốc độ nạp batch tối đa cho GPU.
    """
    def __init__(self, 
                 bio_parquet_path: str,
                 dna_geom_path: str, 
                 prot_geom_path: str, 
                 dna_pt_path: str, 
                 prot_pt_path: str, 
                 is_train: bool = True):
        
        self.is_train = is_train
        
        # =====================================================================
        # 1. TẢI VÀ ĐỒNG BỘ DỮ LIỆU BẢNG (TABULAR DATA)
        # =====================================================================
        print("[*] Đang nạp và hợp nhất (Merge) dữ liệu bảng...")
        df_bio = pd.read_parquet(bio_parquet_path)
        
        df_dna_geom = pd.read_parquet(dna_geom_path)
        dna_rename_dict = {col: f"dna_{col}" for col in ["LLR", "LVD_L2", "LVD_Cosine", "LID"]}
        df_dna_geom = df_dna_geom.rename(columns=dna_rename_dict)
        
        df_prot_geom = pd.read_parquet(prot_geom_path)
        prot_rename_dict = {col: f"prot_{col}" for col in ["LLR", "LVD_L2", "LVD_Cosine", "LID"]}
        df_prot_geom = df_prot_geom.rename(columns=prot_rename_dict)
        
        # Inner Join: Chỉ giữ lại các Variant có mặt ở CẢ 3 bảng
        self.df = df_bio.merge(df_dna_geom, on="Variant_ID").merge(df_prot_geom, on="Variant_ID")
        
        # =====================================================================
        # 2. TẢI TENSORS & KIỂM TRA TOÀN VẸN
        # =====================================================================
        print("[*] Đang nạp Embeddings và xác thực tính toàn vẹn...")
        self.dna_data = torch.load(dna_pt_path, weights_only=False)
        self.prot_data = torch.load(prot_pt_path, weights_only=False)
        
        dna_idx_map = {vid: idx for idx, vid in enumerate(self.dna_data["metadata"])}
        prot_idx_map = {vid: idx for idx, vid in enumerate(self.prot_data["metadata"])}
        
        # [BẢN VÁ LOGIC] Kiểm tra kỹ cả DNA và Protein
        variant_ids = self.df["Variant_ID"].tolist()
        
        missing_in_dna = set(variant_ids) - set(dna_idx_map.keys())
        if missing_in_dna:
            raise ValueError(f"[LỖI] Thiếu {len(missing_in_dna)} Variant_ID trong DNA Embeddings!")
            
        missing_in_prot = set(variant_ids) - set(prot_idx_map.keys())
        if missing_in_prot:
            raise ValueError(f"[LỖI] Thiếu {len(missing_in_prot)} Variant_ID trong Protein Embeddings!")
            
        # =====================================================================
        # 3. PRE-COMPUTATION (TỐI ƯU HÓA HIỆU NĂNG CHO BATCH LOADER)
        # =====================================================================
        # Thay vì dùng Hash Map tra cứu string từng dòng trong lúc train,
        # Ta tính sẵn mảng số nguyên chỉ mục (integer indices) để truy xuất O(1)
        self.dna_indices = [dna_idx_map[vid] for vid in variant_ids]
        self.prot_indices = [prot_idx_map[vid] for vid in variant_ids]

        # =====================================================================
        # 4. GOM NHÓM ĐẶC TRƯNG VÀ CHUYỂN THÀNH TENSOR TĨNH
        # =====================================================================
        self.bio_cols = [
            "AF", "gnomADe_AF", "phyloP100way_vertebrate", "phyloP470way_mammalian", 
            "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons470way_mammalian", 
            "phastCons17way_primate", "GERP++_RS", "GERP++_NR", "GERP_92_mammals"
        ]
        self.geom_cols = list(dna_rename_dict.values()) + list(prot_rename_dict.values())
        
        # Đẩy sẵn các ma trận Bảng lên RAM dưới định dạng Tensor
        self.bio_tensor = torch.tensor(self.df[self.bio_cols].values, dtype=torch.float32)
        self.geom_tensor = torch.tensor(self.df[self.geom_cols].values, dtype=torch.float32)
        
        # Xử lý Label (chỉ có lúc Train/Val)
        if self.is_train and "Pathogenicity_Label" in self.df.columns:
            self.labels = torch.tensor(self.df["Pathogenicity_Label"].values, dtype=torch.float32)
        else:
            self.labels = None
            
        print(f"[+] Dataset sẵn sàng: {len(self.df)} mẫu.\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # [BẢN VÁ HIỆU NĂNG] Truy xuất chỉ mục O(1) chớp nhoáng
        dna_idx = self.dna_indices[idx]
        prot_idx = self.prot_indices[idx]
        
        # 1. Tính toán Delta V (Sự dịch chuyển không gian)
        dna_ref = self.dna_data["E_ref"][dna_idx]
        dna_alt = self.dna_data["E_alt"][dna_idx]
        v_dna = dna_alt - dna_ref
        
        prot_ref = self.prot_data["E_ref"][prot_idx]
        prot_alt = self.prot_data["E_alt"][prot_idx]
        v_prot = prot_alt - prot_ref
        
        # 2. Gom 11 Sinh học + 8 Hình học
        bio_features = self.bio_tensor[idx]
        geom_features = self.geom_tensor[idx]
        
        # 3. Đóng gói cho PyTorch DataLoader
        sample = {
            "v_dna": v_dna.to(torch.float32),
            "v_prot": v_prot.to(torch.float32),
            "bio_features": bio_features,
            "geom_features": geom_features
        }
        
        if self.labels is not None:
            sample["label"] = self.labels[idx].unsqueeze(0) 
            
        return sample
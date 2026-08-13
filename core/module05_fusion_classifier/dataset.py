import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import gc

class VariantFusionDataset(Dataset):
    """
    Module 5 Dataset: Điểm hội tụ Đa phương thức (Multi-modal Fusion).
    Bản vá V4 (Ultimate Performance):
    - Pre-computation Vector (E_alt - E_ref) ngay tại __init__.
    - Giải phóng 100% RAM dư thừa từ file .pt gốc.
    - Bảo vệ NaN (NaN-shield) cho dữ liệu Tabular.
    """
    def __init__(self, 
                 bio_parquet_path: str,
                 dna_geom_path: str = None, 
                 prot_geom_path: str = None, 
                 dna_pt_path: str = None, 
                 prot_pt_path: str = None, 
                 active_modalities: list = None,
                 is_train: bool = True):
        
        self.is_train = is_train
        
        if active_modalities is None:
            active_modalities = ['dna', 'prot', 'bio', 'geom']
        self.active_mods = [m.lower() for m in active_modalities]
        
        self.has_dna = 'dna' in self.active_mods
        self.has_prot = 'prot' in self.active_mods
        self.has_bio = 'bio' in self.active_mods
        self.has_geom = 'geom' in self.active_mods

        # =====================================================================
        # 1. TẢI VÀ ĐỒNG BỘ DỮ LIỆU BẢNG (TABULAR DATA BACKBONE)
        # =====================================================================
        print(f"[*] Đang nạp dữ liệu bảng (Backbone). Active Mods: {self.active_mods}")
        self.df = pd.read_parquet(bio_parquet_path)
        
        self.geom_cols = []
        if self.has_geom:
            if dna_geom_path:
                df_dna_geom = pd.read_parquet(dna_geom_path)
                dna_rename_dict = {col: f"dna_{col}" for col in ["LLR", "LVD_L2", "LVD_Cosine", "LID"]}
                df_dna_geom = df_dna_geom.rename(columns=dna_rename_dict)
                self.df = self.df.merge(df_dna_geom, on="Variant_ID")
                self.geom_cols.extend(dna_rename_dict.values())
                
            if prot_geom_path:
                df_prot_geom = pd.read_parquet(prot_geom_path)
                prot_rename_dict = {col: f"prot_{col}" for col in ["LLR", "LVD_L2", "LVD_Cosine", "LID"]}
                df_prot_geom = df_prot_geom.rename(columns=prot_rename_dict)
                self.df = self.df.merge(df_prot_geom, on="Variant_ID")
                self.geom_cols.extend(prot_rename_dict.values())
        
        variant_ids = self.df["Variant_ID"].tolist()
        num_samples = len(self.df)

        # =====================================================================
        # 2. TẢI, TÍNH TOÁN TRƯỚC VÀ DỌN RÁC (PRE-COMPUTATION & GC)
        # =====================================================================
        if self.has_dna:
            print("  -> Đang nạp và tiền xử lý DNA Embeddings...")
            dna_data = torch.load(dna_pt_path, weights_only=False)
            dna_idx_map = {vid: idx for idx, vid in enumerate(dna_data["metadata"])}
            
            missing_in_dna = set(variant_ids) - set(dna_idx_map.keys())
            if missing_in_dna:
                raise ValueError(f"[LỖI] Thiếu {len(missing_in_dna)} Variant_ID trong DNA Embeddings!")
                
            # [BẢN VÁ TỐI ƯU RAM] Cắt trích chính xác và tính V_dna sẵn
            sample_dim = dna_data["E_ref"][0].shape[0]
            self.v_dna_tensor = torch.zeros((num_samples, sample_dim), dtype=torch.float32)
            
            for i, vid in enumerate(variant_ids):
                idx = dna_idx_map[vid]
                self.v_dna_tensor[i] = (dna_data["E_alt"][idx] - dna_data["E_ref"][idx]).to(torch.float32)
                
            # Xóa sổ toàn bộ file .pt khổng lồ khỏi RAM
            del dna_data, dna_idx_map
            gc.collect()

        if self.has_prot:
            print("  -> Đang nạp và tiền xử lý Protein Embeddings...")
            prot_data = torch.load(prot_pt_path, weights_only=False)
            prot_idx_map = {vid: idx for idx, vid in enumerate(prot_data["metadata"])}
            
            missing_in_prot = set(variant_ids) - set(prot_idx_map.keys())
            if missing_in_prot:
                raise ValueError(f"[LỖI] Thiếu {len(missing_in_prot)} Variant_ID trong Protein Embeddings!")
                
            sample_dim = prot_data["E_ref"][0].shape[0]
            self.v_prot_tensor = torch.zeros((num_samples, sample_dim), dtype=torch.float32)
            
            for i, vid in enumerate(variant_ids):
                idx = prot_idx_map[vid]
                self.v_prot_tensor[i] = (prot_data["E_alt"][idx] - prot_data["E_ref"][idx]).to(torch.float32)
                
            del prot_data, prot_idx_map
            gc.collect()

        # =====================================================================
        # 3. GOM NHÓM ĐẶC TRƯNG BẢNG & LỌC NaN
        # =====================================================================
        if self.has_bio:
            self.bio_cols = [
                "AF", "gnomADe_AF", "phyloP100way_vertebrate", "phyloP470way_mammalian", 
                "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons470way_mammalian", 
                "phastCons17way_primate", "GERP++_RS", "GERP++_NR", "GERP_92_mammals"
            ]
            # [BẢN VÁ LỖI] Lọc NaN bằng np.nan_to_num
            bio_values = np.nan_to_num(self.df[self.bio_cols].values, nan=0.0)
            self.bio_tensor = torch.tensor(bio_values, dtype=torch.float32)

        if self.has_geom:
            geom_values = np.nan_to_num(self.df[self.geom_cols].values, nan=0.0)
            self.geom_tensor = torch.tensor(geom_values, dtype=torch.float32)
            
        if self.is_train and "Pathogenicity_Label" in self.df.columns:
            self.labels = torch.tensor(self.df["Pathogenicity_Label"].values, dtype=torch.float32)
        else:
            self.labels = None
            
        print(f"[+] Dataset sẵn sàng: {num_samples} mẫu. Đã tối ưu hóa RAM & CPU 100%.\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {"variant_id": self.df["Variant_ID"].values[idx]}
        
        # Hàm __getitem__ giờ đây đạt tốc độ tra cứu O(1) thuần túy
        if self.has_dna:
            sample["v_dna"] = self.v_dna_tensor[idx]
            
        if self.has_prot:
            sample["v_prot"] = self.v_prot_tensor[idx]
            
        if self.has_bio:
            sample["bio_features"] = self.bio_tensor[idx]
            
        if self.has_geom:
            sample["geom_features"] = self.geom_tensor[idx]
            
        if self.labels is not None:
            sample["label"] = self.labels[idx].unsqueeze(0) 

        return sample
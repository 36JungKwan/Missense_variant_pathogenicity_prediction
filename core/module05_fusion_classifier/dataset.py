import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import gc


BIO_CORE_COLUMNS = [
    "AF", "gnomADe_AF", "phyloP100way_vertebrate", "phyloP470way_mammalian",
    "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons470way_mammalian",
    "phastCons17way_primate", "GERP++_RS", "GERP++_NR", "GERP_92_mammals",
]

SPLICEAI_COLUMNS = [
    "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG",
    "SpliceAI_pred_DS_DL", "SpliceAI_pred_DP_AG", "SpliceAI_pred_DP_AL",
    "SpliceAI_pred_DP_DG", "SpliceAI_pred_DP_DL", "SpliceAI_pred_DS_max",
]

GEOM_SOURCE_COLUMNS = ["LLR", "LVD_L2", "LVD_Cosine", "LID"]
GEOM_FEATURE_COLUMNS = [
    *(f"dna_{col}" for col in GEOM_SOURCE_COLUMNS),
    *(f"prot_{col}" for col in GEOM_SOURCE_COLUMNS),
]
GEOM_V2_SOURCE_COLUMNS = [*GEOM_SOURCE_COLUMNS, "LVD_Relative"]
GEOM_V2_FEATURE_COLUMNS = [
    *(f"dna_{col}" for col in GEOM_V2_SOURCE_COLUMNS),
    *(f"prot_{col}" for col in GEOM_V2_SOURCE_COLUMNS),
]
ALL_GEOM_FEATURE_COLUMNS = [
    *GEOM_FEATURE_COLUMNS,
    *[name for name in GEOM_V2_FEATURE_COLUMNS if name not in GEOM_FEATURE_COLUMNS],
]


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
                 geom_feature_names: list = None,
                 is_train: bool = True):
        
        self.is_train = is_train
        
        if active_modalities is None:
            active_modalities = ['dna', 'prot', 'bio', 'geom']
        self.active_mods = [m.lower() for m in active_modalities]
        
        self.has_dna = 'dna' in self.active_mods
        self.has_prot = 'prot' in self.active_mods
        # ``bio`` is the legacy 11-feature group. SpliceAI is opt-in through
        # the explicit ``spliceai`` token so old checkpoints remain compatible.
        self.has_bio_core = 'bio' in self.active_mods or 'bio_core' in self.active_mods
        self.has_spliceai = 'spliceai' in self.active_mods
        self.has_bio = self.has_bio_core or self.has_spliceai
        self.has_geom = 'geom' in self.active_mods

        self.geom_feature_names = (
            list(geom_feature_names)
            if geom_feature_names is not None
            else list(GEOM_FEATURE_COLUMNS)
        )
        unknown_geom = [name for name in self.geom_feature_names if name not in ALL_GEOM_FEATURE_COLUMNS]
        if unknown_geom:
            raise ValueError(f"Geom feature khong hop le: {unknown_geom}")
        if self.has_geom and len(self.geom_feature_names) == 0:
            raise ValueError("Cau hinh geom phai co it nhat mot geom feature")

        # =====================================================================
        # 1. TẢI VÀ ĐỒNG BỘ DỮ LIỆU BẢNG (TABULAR DATA BACKBONE)
        # =====================================================================
        print(f"[*] Đang nạp dữ liệu bảng (Backbone). Active Mods: {self.active_mods}")
        self.df = pd.read_parquet(bio_parquet_path)
        
        self.geom_cols = []
        if self.has_geom:
            selected_dna = [
                name.removeprefix("dna_")
                for name in self.geom_feature_names
                if name.startswith("dna_")
            ]
            selected_prot = [
                name.removeprefix("prot_")
                for name in self.geom_feature_names
                if name.startswith("prot_")
            ]

            if dna_geom_path and selected_dna:
                df_dna_geom = pd.read_parquet(dna_geom_path)
                dna_rename_dict = {col: f"dna_{col}" for col in selected_dna}
                missing_dna_geom = [col for col in selected_dna if col not in df_dna_geom.columns]
                if missing_dna_geom:
                    raise KeyError(f"Thieu DNA geom columns: {missing_dna_geom}")
                df_dna_geom = df_dna_geom.rename(columns=dna_rename_dict)
                self.df = self.df.merge(df_dna_geom, on="Variant_ID")
                self.geom_cols.extend(dna_rename_dict.values())
                
            if prot_geom_path and selected_prot:
                df_prot_geom = pd.read_parquet(prot_geom_path)
                prot_rename_dict = {col: f"prot_{col}" for col in selected_prot}
                missing_prot_geom = [col for col in selected_prot if col not in df_prot_geom.columns]
                if missing_prot_geom:
                    raise KeyError(f"Thieu protein geom columns: {missing_prot_geom}")
                df_prot_geom = df_prot_geom.rename(columns=prot_rename_dict)
                self.df = self.df.merge(df_prot_geom, on="Variant_ID")
                self.geom_cols.extend(prot_rename_dict.values())

            missing_selected_geom = [col for col in self.geom_feature_names if col not in self.geom_cols]
            if missing_selected_geom:
                raise KeyError(
                    "Khong nap duoc cac geom feature duoc chon: "
                    f"{missing_selected_geom}"
                )
        
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
            self.bio_cols = []
            if self.has_bio_core:
                self.bio_cols.extend(BIO_CORE_COLUMNS)
            if self.has_spliceai:
                self.bio_cols.extend(SPLICEAI_COLUMNS)

            missing_bio_cols = [col for col in self.bio_cols if col not in self.df.columns]
            if missing_bio_cols:
                raise KeyError(
                    "Thieu cot bio/spliceai trong normalized parquet: "
                    f"{missing_bio_cols}"
                )
            # [BẢN VÁ LỖI] Lọc NaN bằng np.nan_to_num
            bio_values = np.nan_to_num(self.df[self.bio_cols].values, nan=0.0)
            self.bio_tensor = torch.tensor(bio_values, dtype=torch.float32)

        if self.has_geom:
            geom_values = np.nan_to_num(self.df[self.geom_cols].values, nan=0.0)
            self.geom_tensor = torch.tensor(geom_values, dtype=torch.float32)

        label_candidates = ["Pathogenicity_Label", "Label", "label", "target", "Target"]
        label_col = next((c for c in label_candidates if c in self.df.columns), None)
        if label_col is not None:
            label_values = pd.to_numeric(self.df[label_col], errors="coerce").fillna(0.0).values
            self.labels = torch.tensor(label_values, dtype=torch.float32)
        else:
            self.labels = None
            if self.is_train:
                raise KeyError(
                    "Khong tim thay cot nhan. Can mot trong cac cot: "
                    f"{label_candidates}"
                )
            
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

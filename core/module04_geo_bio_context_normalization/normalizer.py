import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer

class AdvancedFeatureNormalizer:
    """
    Module 4: Xử lý hậu kỳ (Điền khuyết & Chuẩn hóa) các đặc trưng sinh học từ VEP.
    Phân tách logic nghiêm ngặt giữa dải xác suất [0,1] và dải điểm vô cực.
    """
    def __init__(self):
        self.geom_llr_cols = ["LLR"]
        self.geom_lvd_cols = ["LVD_L2", "LVD_Cosine"]
        self.geom_lid_cols = ["LID"]

        self.bio_freq_cols = ["AF", "gnomADe_AF"]
        self.bio_phast_cols = [
            "phastCons100way_vertebrate", "phastCons470way_mammalian", "phastCons17way_primate"
        ]
        self.bio_phylo_gerp_cols = [
            "phyloP100way_vertebrate", "phyloP470way_mammalian", "phyloP17way_primate",
            "GERP++_RS", "GERP++_NR", "GERP_92_mammals"
        ]
        self.bio_mean_impute_cols = self.bio_phast_cols + self.bio_phylo_gerp_cols

    def _coerce_numeric(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Ép toàn bộ cột sinh học về kiểu số thực (float), biến các ký tự rác thành NaN."""
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"[LỖI] Dữ liệu đầu vào đang thiếu cột đặc trưng: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # =========================================================================
    # NHÁNH 1: XỬ LÝ DỮ LIỆU SINH HỌC (CHẠY 1 LẦN DUY NHẤT)
    # =========================================================================
    def fit_transform_bio(self, df: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
        """Luồng Huấn luyện: Học phân phối từ tập Train và lưu file .pkl"""
        print(f"[*] Đang học phân phối và chuẩn hóa tập Train ({len(df)} variants)...")
        df_out = df.copy()
        os.makedirs(artifacts_dir, exist_ok=True)
        
        df_out = self._coerce_numeric(df_out, self.bio_freq_cols + self.bio_mean_impute_cols)

        # 1. Tần số (AF): Điền 0 -> Log10(x + eps) -> StandardScaler
        df_out[self.bio_freq_cols] = df_out[self.bio_freq_cols].fillna(0.0)
        df_out[self.bio_freq_cols] = np.log10(df_out[self.bio_freq_cols] + self.epsilon)
        freq_scaler = StandardScaler()
        df_out[self.bio_freq_cols] = freq_scaler.fit_transform(df_out[self.bio_freq_cols])

        # 2. Điền khuyết Mean chung cho toàn bộ Phast, Phylo, GERP
        mean_imputer = SimpleImputer(strategy='mean')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_out[self.bio_mean_impute_cols] = mean_imputer.fit_transform(df_out[self.bio_mean_impute_cols])

        # 3. PhastCons: StandardScaler
        phast_scaler = StandardScaler()
        df_out[self.bio_phast_cols] = phast_scaler.fit_transform(df_out[self.bio_phast_cols])

        # 4. PhyloP & GERP: QuantileTransformer (Gaussian)
        phylo_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        df_out[self.bio_phylo_gerp_cols] = phylo_scaler.fit_transform(df_out[self.bio_phylo_gerp_cols])
        
        # Lưu Artifacts
        joblib.dump(mean_imputer, f"{artifacts_dir}/bio_mean_imputer.pkl")
        joblib.dump(freq_scaler, f"{artifacts_dir}/bio_freq_scaler.pkl")
        joblib.dump(phast_scaler, f"{artifacts_dir}/bio_phast_scaler.pkl")
        joblib.dump(phylo_scaler, f"{artifacts_dir}/bio_phylo_scaler.pkl")
        
        return df_out

    def transform_bio(self, df: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
        """Luồng Đánh giá: Nạp .pkl tĩnh để chuẩn hóa tập Val/Test."""
        print(f"[*] Đang nạp Artifacts để chuẩn hóa tập Val/Test ({len(df)} variants)...")
        df_out = df.copy()
        df_out = self._coerce_numeric(df_out, self.bio_freq_cols + self.bio_mean_impute_cols)
        
        # Tải Artifacts
        mean_imputer = joblib.load(f"{artifacts_dir}/bio_mean_imputer.pkl")
        freq_scaler = joblib.load(f"{artifacts_dir}/bio_freq_scaler.pkl")
        phast_scaler = joblib.load(f"{artifacts_dir}/bio_phast_scaler.pkl")
        phylo_scaler = joblib.load(f"{artifacts_dir}/bio_phylo_scaler.pkl")
        
        df_out[self.bio_freq_cols] = df_out[self.bio_freq_cols].fillna(0.0)
        df_out[self.bio_freq_cols] = freq_scaler.transform(np.log10(df_out[self.bio_freq_cols] + self.epsilon))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_out[self.bio_mean_impute_cols] = mean_imputer.transform(df_out[self.bio_mean_impute_cols])
            
        df_out[self.bio_phast_cols] = phast_scaler.transform(df_out[self.bio_phast_cols])
        df_out[self.bio_phylo_gerp_cols] = phylo_scaler.transform(df_out[self.bio_phylo_gerp_cols])

        return df_out

    # =========================================================================
    # NHÁNH 2: XỬ LÝ DỮ LIỆU HÌNH HỌC (CHẠY THEO TỪNG CẤU HÌNH)
    # =========================================================================
    def fit_transform_geom(self, df: pd.DataFrame, artifacts_dir: str, config_name: str) -> pd.DataFrame:
        df_out = df.copy()
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # [BẢN VÁ] Chốt chặn an toàn: Ép kiểu và xử lý NaN tiềm ẩn từ PyTorch/FAISS
        all_geom_cols = self.geom_llr_cols + self.geom_lvd_cols + self.geom_lid_cols
        df_out = self._coerce_numeric(df_out, all_geom_cols)
        df_out[all_geom_cols] = df_out[all_geom_cols].fillna(0.0)
        
        # 1. LLR: StandardScaler
        llr_scaler = StandardScaler()
        df_out[self.geom_llr_cols] = llr_scaler.fit_transform(df_out[self.geom_llr_cols])
        
        # 2. LVD (L2, Cosine): Log10(x + eps) -> StandardScaler
        df_out[self.geom_lvd_cols] = np.log10(df_out[self.geom_lvd_cols] + self.epsilon)
        lvd_scaler = StandardScaler()
        df_out[self.geom_lvd_cols] = lvd_scaler.fit_transform(df_out[self.geom_lvd_cols])
        
        # 3. LID: QuantileTransformer (Gaussian)
        lid_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        df_out[self.geom_lid_cols] = lid_scaler.fit_transform(df_out[self.geom_lid_cols])
        
        # Lưu Artifacts
        joblib.dump(llr_scaler, f"{artifacts_dir}/{config_name}_llr_scaler.pkl")
        joblib.dump(lvd_scaler, f"{artifacts_dir}/{config_name}_lvd_scaler.pkl")
        joblib.dump(lid_scaler, f"{artifacts_dir}/{config_name}_lid_scaler.pkl")
        
        return df_out

    def transform_geom(self, df: pd.DataFrame, artifacts_dir: str, config_name: str) -> pd.DataFrame:
        df_out = df.copy()
        
        # [BẢN VÁ] Chốt chặn an toàn cho tập Val/Test
        all_geom_cols = self.geom_llr_cols + self.geom_lvd_cols + self.geom_lid_cols
        df_out = self._coerce_numeric(df_out, all_geom_cols)
        df_out[all_geom_cols] = df_out[all_geom_cols].fillna(0.0)
        
        llr_scaler = joblib.load(f"{artifacts_dir}/{config_name}_llr_scaler.pkl")
        lvd_scaler = joblib.load(f"{artifacts_dir}/{config_name}_lvd_scaler.pkl")
        lid_scaler = joblib.load(f"{artifacts_dir}/{config_name}_lid_scaler.pkl")
        
        df_out[self.geom_llr_cols] = llr_scaler.transform(df_out[self.geom_llr_cols])
        df_out[self.geom_lvd_cols] = lvd_scaler.transform(np.log10(df_out[self.geom_lvd_cols] + self.epsilon))
        df_out[self.geom_lid_cols] = lid_scaler.transform(df_out[self.geom_lid_cols])
        
        return df_out
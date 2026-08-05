import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import warnings

class BioFeaturePreprocessor:
    """
    Module 4: Xử lý hậu kỳ (Điền khuyết & Chuẩn hóa) các đặc trưng sinh học từ VEP.
    Phân tách logic nghiêm ngặt giữa dải xác suất [0,1] và dải điểm vô cực.
    """
    def __init__(self):
        # Nhóm 1: Tần số (Dải tự nhiên [0, 1]) -> Chỉ điền 0, KHÔNG chuẩn hóa
        self.freq_cols = [
            "AF", "gnomADe_AF"
        ]
        
        # Nhóm 2: Điểm bảo tồn (Chứa Outliers) -> Điền Mean + Áp dụng RobustScaler
        self.conservation_cols = [
            "phyloP100way_vertebrate", "phyloP470way_mammalian", "phyloP17way_primate",
            "phastCons100way_vertebrate", "phastCons470way_mammalian", "phastCons17way_primate",
            "GERP++_RS", "GERP++_NR", "GERP_92_mammals"
        ]
        
        self.all_bio_cols = self.freq_cols + self.conservation_cols

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ép toàn bộ cột sinh học về kiểu số thực (float), biến các ký tự rác thành NaN."""
        for col in self.all_bio_cols:
            if col not in df.columns:
                raise ValueError(f"[LỖI] Dữ liệu đầu vào đang thiếu cột đặc trưng: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def fit_transform_and_save(self, df: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
        """Luồng Huấn luyện: Học phân phối từ tập Train và lưu file .pkl"""
        print(f"[*] Đang học phân phối và chuẩn hóa tập Train ({len(df)} variants)...")
        df_out = df.copy()
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # 1. Ép kiểu dữ liệu an toàn
        df_out = self._coerce_numeric(df_out)

        # 2. Nhóm Tần số: Chỉ điền khuyết 0.0, bảo toàn dải [0, 1]
        df_out[self.freq_cols] = df_out[self.freq_cols].fillna(0.0)

        # 3. Nhóm Bảo tồn: Điền khuyết Mean
        mean_imputer = SimpleImputer(strategy='mean')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_out[self.conservation_cols] = mean_imputer.fit_transform(df_out[self.conservation_cols])

        # 4. Nhóm Bảo tồn: Chuẩn hóa RobustScaler (Chống outliers dải PhyloP/GERP)
        scaler = RobustScaler()
        df_out[self.conservation_cols] = scaler.fit_transform(df_out[self.conservation_cols])

        # 5. Lưu Artifacts tĩnh
        mean_imputer_path = os.path.join(artifacts_dir, "bio_imputer_mean.pkl")
        scaler_path = os.path.join(artifacts_dir, "bio_scaler_robust.pkl")
        
        joblib.dump(mean_imputer, mean_imputer_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"[+] Đã lưu Imputer tại: {mean_imputer_path}")
        print(f"[+] Đã lưu Scaler tại: {scaler_path}")
        
        return df_out

    def load_and_transform(self, df: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
        """Luồng Đánh giá: Nạp .pkl tĩnh để chuẩn hóa tập Val/Test."""
        print(f"[*] Đang nạp Artifacts để chuẩn hóa tập Val/Test ({len(df)} variants)...")
        df_out = df.copy()
        
        mean_imputer_path = os.path.join(artifacts_dir, "bio_imputer_mean.pkl")
        scaler_path = os.path.join(artifacts_dir, "bio_scaler_robust.pkl")
        
        if not os.path.exists(mean_imputer_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("[LỖI] Không tìm thấy file .pkl. Vui lòng chạy tập Train trước.")
            
        mean_imputer = joblib.load(mean_imputer_path)
        scaler = joblib.load(scaler_path)

        # 1. Ép kiểu dữ liệu an toàn
        df_out = self._coerce_numeric(df_out)

        # 2. Nhóm Tần số: Chỉ điền khuyết 0.0
        df_out[self.freq_cols] = df_out[self.freq_cols].fillna(0.0)

        # 3. Nhóm Bảo tồn: Transform bằng Mean của tập Train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_out[self.conservation_cols] = mean_imputer.transform(df_out[self.conservation_cols])

        # 4. Nhóm Bảo tồn: Transform bằng Median/IQR của tập Train
        df_out[self.conservation_cols] = scaler.transform(df_out[self.conservation_cols])

        print("[+] Chuẩn hóa thành công dựa trên quy tắc của tập Train!")
        return df_out
import os
import numpy as np
import xgboost as xgb
import joblib
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

class XGBoostFusionManager:
    """
    Module 5: Trình quản lý thuật toán XGBoost.
    Bản vá V2: Hỗ trợ chuyển đổi tự động Tensor -> Numpy an toàn, 
    Dynamic PCA và tương thích XGBoost v2.0+.
    """
    def __init__(self, pca_components: int = 256, random_state: int = 42):
        self.pca_components = pca_components
        self.random_state = random_state
        
        self.pca_dna = None
        self.pca_prot = None
        
        # [BẢN VÁ 3] Dời early_stopping_rounds lên đây theo chuẩn API mới
        self.base_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'early_stopping_rounds': 20, 
            'tree_method': 'hist',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        self.model = None

    # [BẢN VÁ 1] Helper an toàn chuyển đổi mọi thứ thành Numpy
    def _to_numpy(self, data) -> np.ndarray:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.array(data)

    def _calc_scale_pos_weight(self, y_train: np.ndarray) -> float:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        if pos_count == 0:
            return 1.0
        return float(neg_count / pos_count)

    def _evaluate_metrics(self, y_true, y_probs, y_preds):
        return {
            "ROC-AUC": roc_auc_score(y_true, y_probs),
            "PR-AUC": average_precision_score(y_true, y_probs),
            "MCC": matthews_corrcoef(y_true, y_preds)
        }

    # =========================================================================
    # HƯỚNG 3 (HYBRID): NHẬN F_GLOBAL TỪ PYTORCH
    # =========================================================================
    def train_hybrid(self, X_train, y_train, X_val=None, y_val=None):
        print("[*] Đang khởi tạo XGBoost (Hybrid Mode)...")
        
        # Đảm bảo dữ liệu 100% là Numpy
        X_train_np = self._to_numpy(X_train)
        y_train_np = self._to_numpy(y_train)
        
        spw = self._calc_scale_pos_weight(y_train_np)
        
        # Nếu không có Val, phải xóa early_stopping_rounds ra khỏi dict
        params = self.base_params.copy()
        if X_val is None or y_val is None:
            params.pop('early_stopping_rounds', None)
            
        self.model = xgb.XGBClassifier(**params, scale_pos_weight=spw)
        
        if X_val is not None and y_val is not None:
            X_val_np = self._to_numpy(X_val)
            y_val_np = self._to_numpy(y_val)
            
            self.model.fit(
                X_train_np, y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                verbose=False
            )
            print(f"[+] Huấn luyện xong! Dừng ở cây thứ {self.model.best_iteration}.")
        else:
            self.model.fit(X_train_np, y_train_np)
            print("[+] Huấn luyện xong (Không dùng validation).")
            
    def predict_hybrid(self, X_test, y_test=None):
        X_test_np = self._to_numpy(X_test)
        
        probs = self.model.predict_proba(X_test_np)[:, 1]
        preds = self.model.predict(X_test_np)
        
        metrics = None
        if y_test is not None:
            y_test_np = self._to_numpy(y_test)
            metrics = self._evaluate_metrics(y_test_np, probs, preds)
            
        return probs, preds, metrics

    # =========================================================================
    # HƯỚNG 2 (PURE ML): NHẬN VECTOR GỐC VÀ DÙNG PCA
    # =========================================================================
    def fit_transform_pca(self, v_dna, v_prot):
        v_dna_np = self._to_numpy(v_dna)
        v_prot_np = self._to_numpy(v_prot)
        
        # [BẢN VÁ 2] Dynamic PCA Components chống crash khi data nhỏ
        n_samples = v_dna_np.shape[0]
        actual_comp = min(self.pca_components, n_samples)
        
        print(f"[*] Đang nén PCA từ không gian gốc xuống {actual_comp} chiều...")
        self.pca_dna = PCA(n_components=actual_comp, random_state=self.random_state)
        self.pca_prot = PCA(n_components=actual_comp, random_state=self.random_state)
        
        v_dna_pca = self.pca_dna.fit_transform(v_dna_np)
        v_prot_pca = self.pca_prot.fit_transform(v_prot_np)
        return v_dna_pca, v_prot_pca

    def transform_pca(self, v_dna, v_prot):
        v_dna_np = self._to_numpy(v_dna)
        v_prot_np = self._to_numpy(v_prot)
        
        v_dna_pca = self.pca_dna.transform(v_dna_np)
        v_prot_pca = self.pca_prot.transform(v_prot_np)
        return v_dna_pca, v_prot_pca

    def _build_pure_features(self, v_dna_pca, v_prot_pca, bio_geom):
        bg_np = self._to_numpy(bio_geom)
        return np.concatenate([v_dna_pca, v_prot_pca, bg_np], axis=1)

    def train_pure(self, v_dna_train, v_prot_train, bg_train, y_train, 
                   v_dna_val=None, v_prot_val=None, bg_val=None, y_val=None):
        v_dna_pca_tr, v_prot_pca_tr = self.fit_transform_pca(v_dna_train, v_prot_train)
        X_train = self._build_pure_features(v_dna_pca_tr, v_prot_pca_tr, bg_train)
        
        X_val = None
        if v_dna_val is not None:
            v_dna_pca_vl, v_prot_pca_vl = self.transform_pca(v_dna_val, v_prot_val)
            X_val = self._build_pure_features(v_dna_pca_vl, v_prot_pca_vl, bg_val)
            
        self.train_hybrid(X_train, y_train, X_val, y_val)
        
    def predict_pure(self, v_dna_test, v_prot_test, bg_test, y_test=None):
        v_dna_pca_ts, v_prot_pca_ts = self.transform_pca(v_dna_test, v_prot_test)
        X_test = self._build_pure_features(v_dna_pca_ts, v_prot_pca_ts, bg_test)
        
        return self.predict_hybrid(X_test, y_test)

    def save_model(self, save_dir: str, prefix: str = "baseline"):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(f"{save_dir}/{prefix}_xgboost.json")
        if self.pca_dna is not None:
            joblib.dump(self.pca_dna, f"{save_dir}/{prefix}_pca_dna.pkl")
            joblib.dump(self.pca_prot, f"{save_dir}/{prefix}_pca_prot.pkl")
            
    def load_model(self, load_dir: str, prefix: str = "baseline"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(f"{load_dir}/{prefix}_xgboost.json")
        pca_dna_path = f"{load_dir}/{prefix}_pca_dna.pkl"
        if os.path.exists(pca_dna_path):
            self.pca_dna = joblib.load(pca_dna_path)
            self.pca_prot = joblib.load(f"{load_dir}/{prefix}_pca_prot.pkl")
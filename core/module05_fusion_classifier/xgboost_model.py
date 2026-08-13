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
    Bản vá V4 (Bulletproof): 
    - Kháng lỗi bất đối xứng PCA giữa Train/Test.
    - Ép chuẩn Shape 2D tự động.
    - Tối ưu API XGBoost 2.0+ (Không còn spam warning).
    """
    def __init__(self, pca_components: int = 256, random_state: int = 42):
        self.pca_components = pca_components
        self.random_state = random_state
        
        self.pca_dna = None
        self.pca_prot = None
        
        # [BẢN VÁ 2] Thêm eval_metric để tương thích hoàn hảo API mới
        self.base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',  # Ổn định Early Stopping và tắt cảnh báo
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

    def _to_numpy(self, data) -> np.ndarray:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.array(data)

    def _calc_scale_pos_weight(self, y_train: np.ndarray) -> float:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        # [BẢN VÁ] Bảo vệ tuyệt đối chia cho 0
        if pos_count == 0 or neg_count == 0:
            return 1.0
        return float(neg_count / pos_count)

    def _evaluate_metrics(self, y_true, y_probs, y_preds):
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_probs)
            auprc = average_precision_score(y_true, y_probs)
        else:
            auroc, auprc = 0.0, 0.0
            
        return {
            "ROC-AUC": round(auroc, 4),
            "PR-AUC": round(auprc, 4),
            "MCC": round(matthews_corrcoef(y_true, y_preds), 4)
        }

    # =========================================================================
    # HƯỚNG 3 (HYBRID): NHẬN F_GLOBAL TỪ PYTORCH
    # =========================================================================
    def train_hybrid(self, X_train, y_train, X_val=None, y_val=None):
        print("[*] Đang khởi tạo XGBoost...")
        
        X_train_np = self._to_numpy(X_train)
        y_train_np = self._to_numpy(y_train)
        spw = self._calc_scale_pos_weight(y_train_np)
        
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
            # Khắc phục lỗi in ấn nếu early_stopping không được trigger
            best_iter = getattr(self.model, 'best_iteration', 'N/A')
            print(f"[+] Huấn luyện xong! Dừng ở cây thứ {best_iter}.")
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
    # HƯỚNG 2 (PURE ML): NÉN ĐỘNG VÀ HÒA TRỘN TỰ ĐỘNG
    # =========================================================================
    def fit_transform_pca(self, v_dna=None, v_prot=None):
        v_dna_pca, v_prot_pca = None, None
        
        if v_dna is not None:
            v_dna_np = self._to_numpy(v_dna)
            actual_comp = min(self.pca_components, v_dna_np.shape[0], v_dna_np.shape[1])
            print(f"[*] PCA (DNA): Nén từ {v_dna_np.shape[1]}D xuống {actual_comp}D...")
            self.pca_dna = PCA(n_components=actual_comp, random_state=self.random_state)
            v_dna_pca = self.pca_dna.fit_transform(v_dna_np)
            
        if v_prot is not None:
            v_prot_np = self._to_numpy(v_prot)
            actual_comp = min(self.pca_components, v_prot_np.shape[0], v_prot_np.shape[1])
            print(f"[*] PCA (Protein): Nén từ {v_prot_np.shape[1]}D xuống {actual_comp}D...")
            self.pca_prot = PCA(n_components=actual_comp, random_state=self.random_state)
            v_prot_pca = self.pca_prot.fit_transform(v_prot_np)
            
        return v_dna_pca, v_prot_pca

    def transform_pca(self, v_dna=None, v_prot=None):
        v_dna_pca, v_prot_pca = None, None
        
        # [BẢN VÁ 1] Chặn Lỗi Bất Đối Xứng (Symmetric Guard)
        if v_dna is not None:
            if self.pca_dna is None:
                raise ValueError("[LỖI KHOA HỌC] Dữ liệu Test có DNA nhưng Mô hình chưa học PCA cho DNA!")
            v_dna_pca = self.pca_dna.transform(self._to_numpy(v_dna))
            
        if v_prot is not None:
            if self.pca_prot is None:
                raise ValueError("[LỖI KHOA HỌC] Dữ liệu Test có Protein nhưng Mô hình chưa học PCA cho Protein!")
            v_prot_pca = self.pca_prot.transform(self._to_numpy(v_prot))
            
        return v_dna_pca, v_prot_pca

    def _build_pure_features(self, v_dna_pca=None, v_prot_pca=None, bio_geom=None):
        features = []
        if v_dna_pca is not None: features.append(v_dna_pca)
        if v_prot_pca is not None: features.append(v_prot_pca)
        
        if bio_geom is not None:
            bg_np = self._to_numpy(bio_geom)
            
            # [BẢN VÁ 3] Ép hình khối (Shape Enforcer)
            if len(bg_np.shape) == 1:
                bg_np = bg_np.reshape(-1, 1)
                
            if bg_np.shape[-1] > 0: 
                features.append(bg_np)
                
        if len(features) == 0:
            raise ValueError("[LỖI CHÍ MẠNG] Không có đặc trưng nào được nạp vào cho XGBoost!")
            
        return np.concatenate(features, axis=1)

    def train_pure(self, v_dna_train=None, v_prot_train=None, bg_train=None, y_train=None, 
                   v_dna_val=None, v_prot_val=None, bg_val=None, y_val=None):
        
        v_dna_pca_tr, v_prot_pca_tr = self.fit_transform_pca(v_dna_train, v_prot_train)
        X_train = self._build_pure_features(v_dna_pca_tr, v_prot_pca_tr, bg_train)
        
        X_val = None
        if y_val is not None:
            v_dna_pca_vl, v_prot_pca_vl = self.transform_pca(v_dna_val, v_prot_val)
            X_val = self._build_pure_features(v_dna_pca_vl, v_prot_pca_vl, bg_val)
            
        self.train_hybrid(X_train, y_train, X_val, y_val)
        
    def predict_pure(self, v_dna_test=None, v_prot_test=None, bg_test=None, y_test=None):
        v_dna_pca_ts, v_prot_pca_ts = self.transform_pca(v_dna_test, v_prot_test)
        X_test = self._build_pure_features(v_dna_pca_ts, v_prot_pca_ts, bg_test)
        
        return self.predict_hybrid(X_test, y_test)

    def save_model(self, save_dir: str, prefix: str = "baseline"):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(f"{save_dir}/{prefix}_xgboost.json")
        
        if self.pca_dna is not None:
            joblib.dump(self.pca_dna, f"{save_dir}/{prefix}_pca_dna.pkl")
        if self.pca_prot is not None:
            joblib.dump(self.pca_prot, f"{save_dir}/{prefix}_pca_prot.pkl")
            
    def load_model(self, load_dir: str, prefix: str = "baseline"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(f"{load_dir}/{prefix}_xgboost.json")
        
        pca_dna_path = f"{load_dir}/{prefix}_pca_dna.pkl"
        if os.path.exists(pca_dna_path):
            self.pca_dna = joblib.load(pca_dna_path)
            
        pca_prot_path = f"{load_dir}/{prefix}_pca_prot.pkl"
        if os.path.exists(pca_prot_path):
            self.pca_prot = joblib.load(pca_prot_path)
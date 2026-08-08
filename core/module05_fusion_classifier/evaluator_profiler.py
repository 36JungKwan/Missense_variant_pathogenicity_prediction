import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
)
from torch.utils.tensorboard import SummaryWriter
try:
    from thop import profile
except ImportError:
    print("[CẢNH BÁO] Thư viện 'thop' chưa được cài đặt. Profiling GFLOPs sẽ trả về 0.")

class FusionEvaluatorProfiler:
    """
    Trình đánh giá và đo lường End-to-End cho Module 5.
    V2: Vá lỗi Peak VRAM ảo, chống crash khi dữ liệu thiếu class, an toàn IO.
    """
    def __init__(self, tensorboard_dir: str, fm_profile_json: str, device: torch.device):
        self.device = device
        self.fm_profile_json = fm_profile_json
        
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        self.fusion_metrics = {
            "gflops_per_sample": 0.0,
            "param_memory_mb": 0.0,
            "num_parameters": 0,
            "inference_time_ms": 0.0,
            "peak_memory_mb": 0.0
        }
        
        self._inference_start = 0.0
        self._total_inference_time = 0.0

    # =========================================================================
    # 1. TÍNH TOÁN 8 CHỈ SỐ PHÂN LOẠI
    # =========================================================================
    def compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray, y_preds: np.ndarray) -> dict:
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_true, y_preds)
        prec = precision_score(y_true, y_preds, zero_division=0)
        rec = recall_score(y_true, y_preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_preds, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_preds)
        
        # [BẢN VÁ 2] Chống sập hệ thống khi tập dữ liệu chỉ có 1 class (thường gặp ở tập mẫu nhỏ)
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_probs)
            auprc = average_precision_score(y_true, y_probs)
        else:
            auroc = 0.0
            auprc = 0.0
            print("[CẢNH BÁO] Tập dữ liệu đánh giá chỉ chứa 1 Class (Toàn 0 hoặc toàn 1). AUROC/AUPRC = 0.0")
        
        return {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1_Score": round(f1, 4),
            "MCC": round(mcc, 4),
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4)
        }

    # =========================================================================
    # 2. ĐO LƯỜNG PHẦN CỨNG LỚP FUSION
    # =========================================================================
    def profile_pytorch_fusion(self, model: torch.nn.Module, dummy_inputs: tuple):
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        param_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        self.fusion_metrics["num_parameters"] = total_params
        self.fusion_metrics["param_memory_mb"] = round(param_mem_mb, 2)
        
        try:
            # Lưu ý: dummy_inputs phải là tuple chứa đủ (v_dna, v_prot, bio, geom)
            macs, _ = profile(model, inputs=dummy_inputs, verbose=False)
            gflops = (macs * 2) / (10 ** 9)
            self.fusion_metrics["gflops_per_sample"] = round(gflops, 6)
        except Exception as e:
            print(f"[CẢNH BÁO] thop không thể đo GFLOPs cho Fusion Model: {e}")

    # [BẢN VÁ 1] Làm sạch bộ đếm Peak VRAM trước khi vào Inference
    def reset_memory_stats(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def tic_inference(self):
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self._inference_start = time.time()
        
    def toc_inference(self):
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self._total_inference_time += (time.time() - self._inference_start)
        
    def finalize_fusion_inference_profiling(self, num_samples: int):
        if self.device.type == 'cuda':
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram = 0.0
            
        latency_ms = (self._total_inference_time / num_samples) * 1000
        self.fusion_metrics["inference_time_ms"] = round(latency_ms, 4)
        self.fusion_metrics["peak_memory_mb"] = round(peak_vram, 2)
        self._total_inference_time = 0.0 

    # =========================================================================
    # 3. TỔNG HỢP END-TO-END (Cộng gộp với Foundation Models)
    # =========================================================================
    def get_e2e_profiling(self, dna_model_name: str, prot_model_name: str) -> dict:
        fm_data = {}
        if os.path.exists(self.fm_profile_json):
            # [BẢN VÁ 3] An toàn chống lỗi khi file JSON bị rỗng/lỗi định dạng
            try:
                with open(self.fm_profile_json, 'r') as f:
                    fm_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[CẢNH BÁO] File {self.fm_profile_json} bị lỗi định dạng. Bỏ qua thông số FM.")
        else:
            print(f"[CẢNH BÁO] Không tìm thấy {self.fm_profile_json}. Báo cáo E2E sẽ thiếu số liệu FM.")
            
        dna_stats = fm_data.get(dna_model_name, {})
        prot_stats = fm_data.get(prot_model_name, {})
        
        def safe_get(d, key): return d.get(key, 0.0)

        e2e_gflops = safe_get(dna_stats, "gflops_per_sample") + safe_get(prot_stats, "gflops_per_sample") + self.fusion_metrics["gflops_per_sample"]
        e2e_latency = safe_get(dna_stats, "inference_time_ms") + safe_get(prot_stats, "inference_time_ms") + self.fusion_metrics["inference_time_ms"]
        e2e_params = safe_get(dna_stats, "num_parameters") + safe_get(prot_stats, "num_parameters") + self.fusion_metrics["num_parameters"]
        e2e_param_mem = safe_get(dna_stats, "param_memory_mb") + safe_get(prot_stats, "param_memory_mb") + self.fusion_metrics["param_memory_mb"]
        
        e2e_peak_mem = max([
            safe_get(dna_stats, "peak_memory_mb"), 
            safe_get(prot_stats, "peak_memory_mb"), 
            self.fusion_metrics["peak_memory_mb"]
        ])
        
        return {
            "E2E_GFLOPs/Sample": round(e2e_gflops, 4),
            "E2E_Latency_ms/Sample": round(e2e_latency, 2),
            "E2E_Total_Params": int(e2e_params),
            "E2E_Param_Mem_MB": round(e2e_param_mem, 2),
            "Max_Peak_VRAM_MB": round(e2e_peak_mem, 2)
        }

    # =========================================================================
    # 4. TENSORBOARD LOGGING
    # =========================================================================
    def log_epoch_scalars(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict):
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}', value, epoch)

    def log_hparams(self, hparam_dict: dict, final_metrics: dict):
        self.writer.add_hparams(
            hparam_dict=hparam_dict,
            metric_dict=final_metrics,
            run_name="."
        )
        self.writer.flush()
        
    def close(self):
        self.writer.close()
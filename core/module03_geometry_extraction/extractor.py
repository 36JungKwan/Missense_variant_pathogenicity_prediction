import os
import time
import json
import copy
import torch
import torch.nn.functional as F
import numpy as np
import faiss


class GeometryExtractionProfiler:
    """Profiler cho Module 3: do thoi gian index/LVD/LID va peak memory."""

    def __init__(self, config_name: str, device: torch.device):
        self.config_name = config_name
        self.device = device
        self.metrics = {}
        self._timers = {}
        self._acc = {
            "index_build_time_s": 0.0,
            "index_load_time_s": 0.0,
            "lvd_time_s": 0.0,
            "lid_time_s": 0.0,
            "geometry_total_time_s": 0.0,
        }

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def tic(self, key: str):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._timers[key] = time.time()

    def toc(self, key: str):
        start = self._timers.get(key)
        if start is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - start
        self._acc[key] = self._acc.get(key, 0.0) + dt
        self._timers[key] = None

    def finalize(self, num_samples: int):
        peak_memory_mb = 0.0
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        self.metrics.update({
            "num_samples": int(num_samples),
            "index_build_time_s": round(self._acc["index_build_time_s"], 4),
            "index_load_time_s": round(self._acc["index_load_time_s"], 4),
            "lvd_time_s": round(self._acc["lvd_time_s"], 4),
            "lid_time_s": round(self._acc["lid_time_s"], 4),
            "geometry_total_time_s": round(self._acc["geometry_total_time_s"], 4),
            "lvd_time_ms_per_sample": round((self._acc["lvd_time_s"] * 1000.0) / max(num_samples, 1), 4),
            "lid_time_ms_per_sample": round((self._acc["lid_time_s"] * 1000.0) / max(num_samples, 1), 4),
            "geometry_total_ms_per_sample": round((self._acc["geometry_total_time_s"] * 1000.0) / max(num_samples, 1), 4),
            "peak_memory_mb": round(peak_memory_mb, 2),
        })

    def reset_accumulators(self):
        for key in self._acc:
            self._acc[key] = 0.0

    def export_to_json(self, filepath: str, split_name: str, source_path: str | None = None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    all_metrics = json.load(f)
                except json.JSONDecodeError:
                    all_metrics = {}
        else:
            all_metrics = {}

        config_entry = all_metrics.get(self.config_name, {})
        if not isinstance(config_entry, dict):
            config_entry = {"__legacy__": config_entry}

        record = copy.deepcopy(self.metrics)
        if source_path is not None:
            record["source_path"] = source_path
        config_entry[split_name] = record
        all_metrics[self.config_name] = config_entry

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4)

class LatentGeometryCalculator:
    """
    Lăng kính Hình học Tiềm ẩn: Định lượng rào cản vật lý và tiến hóa thông qua 
    không gian nhúng (embeddings) của các Foundation Models[cite: 6].
    """
    def __init__(self, k_neighbors: int = 32, epsilon: float = 1e-8):
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon

        # FAISS tren mot so moi truong Windows chi co ban CPU (khong co StandardGpuResources).
        self.use_faiss_gpu = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
        self.gpu_res = faiss.StandardGpuResources() if self.use_faiss_gpu else None
        if self.use_faiss_gpu:
            print("[FAISS] GPU mode enabled")
        else:
            print("[FAISS] GPU APIs khong kha dung, fallback sang CPU mode")
        self.index = None

    def compute_lvd(self, e_ref: torch.Tensor, e_alt: torch.Tensor):
        """
        Tính toán Latent Variant Displacement (LVD) kép: L2 và Cosine[cite: 6].
        Thực thi hoàn toàn trên GPU bằng PyTorch để tối đa tốc độ.
        """
        if not e_ref.is_cuda: e_ref = e_ref.cuda()
        if not e_alt.is_cuda: e_alt = e_alt.cuda()
        
        e_ref = e_ref.to(torch.float32)
        e_alt = e_alt.to(torch.float32)

        # 1. LVD L2 (Khoảng cách Euclidean tuyệt đối)[cite: 6]
        lvd_l2 = torch.norm(e_alt - e_ref, p=2, dim=1, keepdim=True)
        
        # 2. LVD Cosine (Sự thay đổi ngữ nghĩa/chiều hướng)[cite: 6]
        cos_sim = F.cosine_similarity(e_ref, e_alt, dim=1).unsqueeze(1)
        lvd_cosine = 1.0 - cos_sim
        
        return lvd_l2.cpu().to(torch.float16), lvd_cosine.cpu().to(torch.float16)

    def build_and_save_global_index(self, delta_train: np.ndarray, index_path: str, profiler: GeometryExtractionProfiler | None = None):
        """
        Huấn luyện FAISS Index từ tập Delta (E_alt - E_ref) của tập Train[cite: 6].
        Sử dụng GPU để add tốc độ cao, sau đó lưu xuống đĩa cứng.
        """
        if profiler is not None:
            profiler.tic("index_build_time_s")

        print(f"[*] Đang khởi tạo FAISS Index không gian {delta_train.shape[1]} chiều...")
        d = delta_train.shape[1]

        cpu_index = faiss.IndexFlatL2(d)

        print(f"[*] Đang nạp {delta_train.shape[0]} vector vào {'GPU' if self.use_faiss_gpu else 'CPU'} Index...")
        delta_train_f32 = np.ascontiguousarray(delta_train, dtype=np.float32)
        if self.use_faiss_gpu:
            work_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
        else:
            work_index = cpu_index

        work_index.add(delta_train_f32)
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        cpu_index_to_save = faiss.index_gpu_to_cpu(work_index) if self.use_faiss_gpu else work_index
        faiss.write_index(cpu_index_to_save, index_path)
        print(f"[+] Đã huấn luyện và lưu Global FAISS Index tại: {index_path}\n")

        if profiler is not None:
            profiler.toc("index_build_time_s")

    def load_global_index(self, index_path: str, profiler: GeometryExtractionProfiler | None = None):
        """Nạp Index tĩnh từ đĩa cứng thẳng lên GPU."""
        if profiler is not None:
            profiler.tic("index_load_time_s")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file FAISS Index: {index_path}")
        
        cpu_index = faiss.read_index(index_path)
        if self.use_faiss_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
            print(f"[+] Đã nạp thành công Index vào VRAM từ: {index_path}")
        else:
            self.index = cpu_index
            print(f"[+] Đã nạp thành công CPU Index từ: {index_path}")

        if profiler is not None:
            profiler.toc("index_load_time_s")

    def compute_lid(self, delta_queries: torch.Tensor):
        """
        Tính toán Local Intrinsic Dimensionality (LID) bằng FAISS k-NN[cite: 6].
        """
        if self.index is None:
            raise ValueError("Chưa nạp FAISS Index. Vui lòng gọi load_global_index() trước.")
        
        delta_np = np.ascontiguousarray(
            delta_queries.detach().cpu().numpy().astype(np.float32)
        )
        
        D, I = self.index.search(delta_np, self.k_neighbors)
        
        # r là khoảng cách thực tế (căn bậc hai của L2 squared distance)
        r = np.sqrt(np.maximum(D, 0))
        r_k = r[:, -1:] # Khoảng cách đến hàng xóm thứ k
        
        # Bảo vệ phép chia cho 0
        safe_r_k = r_k + self.epsilon
        ratio = r / safe_r_k
        
        # Kẹp tỷ lệ trong đoạn [epsilon, 1.0] để bảo vệ logarit chính xác tuyệt đối tại r_i = r_k
        safe_ratio = np.clip(ratio, self.epsilon, 1.0)
        log_ratio = np.log(safe_ratio)
        
        sum_log = np.sum(log_ratio, axis=1, keepdims=True)
        sum_log = np.minimum(sum_log, -self.epsilon)
        
        lid = - (self.k_neighbors / sum_log)
        
        return torch.tensor(lid, dtype=torch.float16)

    def extract_geometry_features(self, e_ref: torch.Tensor, e_alt: torch.Tensor, profiler: GeometryExtractionProfiler | None = None):
        """
        Hàm Wrapper thực thi toàn bộ pipeline toán học của Module 3[cite: 6].
        Trả về Dictionary chứa các vector vô hướng.
        """
        if profiler is not None:
            profiler.tic("geometry_total_time_s")
            profiler.tic("lvd_time_s")
        lvd_l2, lvd_cosine = self.compute_lvd(e_ref, e_alt)
        if profiler is not None:
            profiler.toc("lvd_time_s")

        delta = e_alt - e_ref
        if profiler is not None:
            profiler.tic("lid_time_s")
        lid = self.compute_lid(delta)
        if profiler is not None:
            profiler.toc("lid_time_s")
            profiler.toc("geometry_total_time_s")
        
        return {
            "LVD_L2": lvd_l2,
            "LVD_Cosine": lvd_cosine,
            "LID": lid
        }
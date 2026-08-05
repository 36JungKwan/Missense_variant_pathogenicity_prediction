import os
import torch
import torch.nn.functional as F
import numpy as np
import faiss

class LatentGeometryCalculator:
    """
    Lăng kính Hình học Tiềm ẩn: Định lượng rào cản vật lý và tiến hóa thông qua 
    không gian nhúng (embeddings) của các Foundation Models[cite: 6].
    """
    def __init__(self, k_neighbors: int = 32, epsilon: float = 1e-8):
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon
        
        # Khởi tạo tài nguyên GPU cho FAISS
        self.gpu_res = faiss.StandardGpuResources()
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

    def build_and_save_global_index(self, delta_train: np.ndarray, index_path: str):
        """
        Huấn luyện FAISS Index từ tập Delta (E_alt - E_ref) của tập Train[cite: 6].
        Sử dụng GPU để add tốc độ cao, sau đó lưu xuống đĩa cứng.
        """
        print(f"[*] Đang khởi tạo FAISS Index không gian {delta_train.shape[1]} chiều...")
        d = delta_train.shape[1]
        
        cpu_index = faiss.IndexFlatL2(d)
        gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
        
        print(f"[*] Đang nạp {delta_train.shape[0]} vector vào GPU Index...")
        delta_train_f32 = np.ascontiguousarray(delta_train, dtype=np.float32)
        gpu_index.add(delta_train_f32)
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        cpu_index_to_save = faiss.index_gpu_to_cpu(gpu_index)
        faiss.write_index(cpu_index_to_save, index_path)
        print(f"[+] Đã huấn luyện và lưu Global FAISS Index tại: {index_path}\n")

    def load_global_index(self, index_path: str):
        """Nạp Index tĩnh từ đĩa cứng thẳng lên GPU."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file FAISS Index: {index_path}")
        
        cpu_index = faiss.read_index(index_path)
        self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
        print(f"[+] Đã nạp thành công Index vào VRAM từ: {index_path}")

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

    def extract_geometry_features(self, e_ref: torch.Tensor, e_alt: torch.Tensor):
        """
        Hàm Wrapper thực thi toàn bộ pipeline toán học của Module 3[cite: 6].
        Trả về Dictionary chứa các vector vô hướng.
        """
        lvd_l2, lvd_cosine = self.compute_lvd(e_ref, e_alt)
        delta = e_alt - e_ref
        lid = self.compute_lid(delta)
        
        return {
            "LVD_L2": lvd_l2,
            "LVD_Cosine": lvd_cosine,
            "LID": lid
        }
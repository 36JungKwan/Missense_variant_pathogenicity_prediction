import time
import json
import os
import torch
try:
    from thop import profile
except ImportError:
    print("[CẢNH BÁO] Chưa cài đặt 'thop'. Chạy lệnh: pip install thop")

class FoundationModelProfiler:
    """
    Công cụ Profiling chuyên dụng cho Foundation Models (Chuẩn báo cáo Khoa học).
    V2: Tách biệt I/O overhead, đo thời gian Forward Pass chuẩn xác 100%.
    """
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.metrics = {}
        
        self._total_forward_time = 0.0
        self._forward_start = 0.0
        self._gflops_measured = False # [BẢN VÁ 2] Cờ chặn đo lặp lại
        
    def measure_static_memory(self, model: torch.nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        param_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        self.metrics["num_parameters"] = total_params
        self.metrics["param_memory_mb"] = round(param_mem_mb, 2)
        print(f"  [Profiler] Tải trọng: {total_params:,} params | {param_mem_mb:.2f} MB")
        
    def measure_gflops(self, model: torch.nn.Module, dummy_input: tuple):
        if self._gflops_measured:
            return
            
        try:
            model.eval()
            macs, _ = profile(model, inputs=dummy_input, verbose=False)
            gflops = (macs * 2) / (10 ** 9)
            self.metrics["gflops_per_sample"] = round(gflops, 4)
            print(f"  [Profiler] Toán học: {gflops:.4f} GFLOPs/sample")
            self._gflops_measured = True
        except Exception as e:
            print(f"  [CẢNH BÁO] thop không thể đo GFLOPs: {e}")
            self.metrics["gflops_per_sample"] = 0.0

    def reset_peak_memory(self):
        """[BẢN VÁ 3] Dọn sạch VRAM trước khi đo Peak."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    # [BẢN VÁ 1] Đồng hồ bấm giờ siêu vi mô (Micro-Stopwatch)
    def tic(self):
        """Bấm giờ TRƯỚC KHI chui vào Model Forward."""
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self._forward_start = time.time()
        
    def toc(self):
        """Chốt giờ SAU KHI Model Forward xong."""
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self._total_forward_time += (time.time() - self._forward_start)
        
    def calculate_final_metrics(self, num_samples: int):
        """Tổng hợp Peak VRAM và Average Latency (loại bỏ 100% độ trễ Ổ cứng)."""
        if self.device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram_mb = 0.0
            
        latency_ms = (self._total_forward_time / num_samples) * 1000
        
        self.metrics["inference_time_ms"] = round(latency_ms, 2)
        self.metrics["peak_memory_mb"] = round(peak_vram_mb, 2)
        print(f"  [Profiler] Tốc độ thuần Model: {latency_ms:.2f} ms/sample | Peak VRAM: {peak_vram_mb:.2f} MB")
        
        # Reset thời gian cho vòng lặp tập dataset kế tiếp
        self._total_forward_time = 0.0 
        
    def export_to_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    all_metrics = json.load(f)
                except json.JSONDecodeError:
                    all_metrics = {}
        else:
            all_metrics = {}
            
        all_metrics[self.model_name] = self.metrics
        with open(filepath, 'w') as f:
            json.dump(all_metrics, f, indent=4)
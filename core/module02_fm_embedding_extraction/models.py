import torch
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
import numpy as np

class BioModelManager:
    """
    Trình quản lý nạp mô hình Sinh học đa phương thức (DNA/Protein) với Mixed Precision.
    """
    def __init__(self, model_id: str, model_type: str, device: str = "cuda"):
        """
        Args:
            model_id: Đường dẫn HuggingFace (VD: "facebook/esm2_t33_650M_UR50D")
            model_type: "mlm" (Masked LM) hoặc "causal" (Autoregressive LM)
            device: Thiết bị chạy inference
        """
        self.model_id = model_id
        self.model_type = model_type.lower()
        self.device = device
        
        self.tokenizer = None
        self.model = None
        
        self._load_engine()

    def _load_engine(self):
        print(f"[*] Đang nạp Tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Sửa lỗi một số tokenizer không có pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
            
        print(f"[*] Đang nạp Mô hình (Trọng số float16): {self.model_id}")
        if self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        elif self.model_type == "mlm":
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"model_type không hợp lệ: {self.model_type}")
            
        self.model.eval()
        self.model.to(self.device)
        print("[+] Nạp mô hình thành công!\n")

    def get_variant_token_index(self, tokenized_outputs, seq_type: str):
        """
        Tìm chỉ số (index) của token chứa vị trí đột biến.
        seq_type = "dna" (đột biến ở index 300) hoặc "protein" (đột biến ở index 50)
        """
        target_char_pos = 300 if seq_type == "dna" else 50
        batch_size = tokenized_outputs["input_ids"].shape[0]
        variant_indices = []

        # Nếu tokenizer hỗ trợ offset_mapping (như NT)
        if "offset_mapping" in tokenized_outputs:
            for i in range(batch_size):
                offsets = tokenized_outputs["offset_mapping"][i].cpu().numpy()
                idx_found = -1
                for idx, (start, end) in enumerate(offsets):
                    if start <= target_char_pos < end:
                        idx_found = idx
                        break
                variant_indices.append(idx_found)
        else:
            # Fallback thủ công cho các mô hình Protein (như ESM) hoặc k-mer ko hỗ trợ offset
            # ESM tokenizer thường là char-level: token_index = char_index + num_special_tokens_at_start
            for i in range(batch_size):
                input_ids_list = tokenized_outputs["input_ids"][i].tolist()
                # Đếm số lượng special tokens (như <cls>) ở đầu chuỗi
                special_start = 0
                while special_start < len(input_ids_list) and input_ids_list[special_start] in self.tokenizer.all_special_ids:
                    special_start += 1
                variant_indices.append(target_char_pos + special_start)
                
        return torch.tensor(variant_indices, device=self.device)

    def prepare_masked_inputs(self, input_ids, variant_indices):
        """
        Tạo một bản sao input_ids và ghi đè token [MASK] vào vị trí đột biến.
        Phục vụ riêng cho lần Forward Pass tính LLR của Masked LM.
        """
        masked_input_ids = input_ids.clone()
        mask_id = self.tokenizer.mask_token_id
        
        if mask_id is None:
            raise ValueError(f"Tokenizer của {self.model_id} không có mask_token_id!")
            
        for i in range(len(variant_indices)):
            idx = variant_indices[i]
            masked_input_ids[i, idx] = mask_id
            
        return masked_input_ids
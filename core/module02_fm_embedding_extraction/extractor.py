import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. DATASET BỘ ĐỌC PARQUET
# ==========================================
class BioSequenceDataset(Dataset):
    """
    Dataset tối ưu để nạp dữ liệu từ file Parquet thành các batch cho PyTorch.
    """
    def __init__(self, parquet_path: str, seq_type: str):
        """
        Args:
            parquet_path: Đường dẫn file Parquet.
            seq_type: "dna" (để lấy DNA_Ref/Alt) hoặc "protein" (để lấy Protein_Ref/Alt)
        """
        self.df = pd.read_parquet(parquet_path)
        self.seq_type = seq_type
        
        # Ánh xạ cột dựa trên loại chuỗi
        self.ref_col = "ref_seq" if seq_type == "dna" else "prot_ref_seq"
        self.alt_col = "alt_seq" if seq_type == "dna" else "prot_alt_seq"
        
        # Đảm bảo dữ liệu tồn tại
        if self.ref_col not in self.df.columns or self.alt_col not in self.df.columns:
            raise ValueError(f"File Parquet thiếu cột {self.ref_col} hoặc {self.alt_col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "name": row["Name"],
            "ref_seq": row[self.ref_col],
            "alt_seq": row[self.alt_col]
        }

# ==========================================
# 2. FEATURE EXTRACTOR (CORE PIPELINE)
# ==========================================
class FeatureExtractor:
    def __init__(self, model_manager):
        self.manager = model_manager
        self.device = model_manager.device
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model

    def _tokenize_batch(self, sequences):
        """Tokenize có hỗ trợ offset_mapping cho các mô hình dùng FastTokenizer"""
        # Kiểm tra xem tokenizer có phải là Fast không để bật offset_mapping
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.tokenizer.is_fast:
            kwargs["return_offsets_mapping"] = True
            
        return self.tokenizer(sequences, **kwargs).to(self.device)

    def _compute_llr(self, logits, variant_indices, ref_ids, alt_ids):
        """Tính Log-Likelihood Ratio theo đúng công thức phân phối xác suất."""
        batch_size = logits.size(0)
        llrs = []
        
        for i in range(batch_size):
            var_idx = variant_indices[i]
            
            # Xử lý bài toán Off-by-one của Causal LM (EVO2)
            if self.manager.model_type == "causal":
                var_idx = max(0, var_idx - 1)
                
            logit_target = logits[i, var_idx, :]
            log_probs = F.log_softmax(logit_target, dim=-1)
            
            llr = log_probs[alt_ids[i]] - log_probs[ref_ids[i]]
            llrs.append(llr.item())
            
        return torch.tensor(llrs, dtype=torch.float16)

    def _extract_poolings(self, hidden_states, attention_mask, variant_indices):
        """Trích xuất 3 loại vector nhúng: Center, Mean, CLS"""
        batch_size = hidden_states.size(0)
        
        # 1. CLS Pooling (Token đầu tiên)
        cls_pooling = hidden_states[:, 0, :]
        
        # 2. Center Pooling (Token tại vị trí đột biến)
        center_pooling = torch.stack([hidden_states[i, variant_indices[i], :] for i in range(batch_size)])
        
        # 3. Mean Pooling (Bỏ qua Padding tokens)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooling = sum_embeddings / sum_mask
        
        return cls_pooling, center_pooling, mean_pooling

    def _process_batch_with_oom_protection(self, batch_data, seq_type):
        """Xử lý forward pass an toàn, đệ quy chia đôi batch nếu tràn RAM (OOM)"""
        names = batch_data["name"]
        ref_seqs = batch_data["ref_seq"]
        alt_seqs = batch_data["alt_seq"]
        batch_size = len(names)
        
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                # 1. Chuẩn bị token (Tokenize một lần duy nhất để tái sử dụng)
                inputs_ref = self._tokenize_batch(ref_seqs)
                inputs_alt = self._tokenize_batch(alt_seqs)
                
                var_indices = self.manager.get_variant_token_index(inputs_ref, seq_type)
                
                # Trích xuất ID của allele gốc và đột biến
                ref_ids = [inputs_ref["input_ids"][i, var_indices[i]].item() for i in range(batch_size)]
                alt_ids = [inputs_alt["input_ids"][i, var_indices[i]].item() for i in range(batch_size)]

                # 2. Tính LLR (Log-Likelihood Ratio)
                if self.manager.model_type == "mlm":
                    # Masked LM: Chạy chuỗi bị MASK (Tái sử dụng input_ids từ inputs_ref)
                    masked_input_ids = self.manager.prepare_masked_inputs(inputs_ref["input_ids"], var_indices)
                    outputs_llr = self.model(input_ids=masked_input_ids, attention_mask=inputs_ref["attention_mask"])
                else:
                    # Causal LM: Chạy thẳng chuỗi Ref
                    outputs_llr = self.model(input_ids=inputs_ref["input_ids"], attention_mask=inputs_ref["attention_mask"])
                
                llr_scores = self._compute_llr(outputs_llr.logits, var_indices, ref_ids, alt_ids)
                
                # Giải phóng RAM lập tức cho biến logits khổng lồ
                del outputs_llr
                
                # 3. Tính Embeddings (Chạy chuỗi nguyên bản Ref)
                outputs_emb = self.model(
                    input_ids=inputs_ref["input_ids"], 
                    attention_mask=inputs_ref["attention_mask"], 
                    output_hidden_states=True
                )
                
                # --- BẢN VÁ: Trích xuất hidden_states an toàn cho mọi kiến trúc (bao gồm ESMC) ---
                if hasattr(outputs_emb, "hidden_states") and outputs_emb.hidden_states is not None:
                    hidden_states = outputs_emb.hidden_states[-1] # Lấy layer cuối
                elif isinstance(outputs_emb, dict) and "hidden_states" in outputs_emb:
                    hidden_states = outputs_emb["hidden_states"][-1]
                else:
                    # Fallback cho dạng tuple/list (thường hidden states layer cuối nằm ở index 0 của tuple)
                    hidden_states = outputs_emb[0]
                # ---------------------------------------------------------------------------------
                
                cls_emb, center_emb, mean_emb = self._extract_poolings(hidden_states, inputs_ref["attention_mask"], var_indices)
                
                # Chuyển đổi về CPU float16 NumPy để lưu trữ
                return {
                    "names": names,
                    "llr": llr_scores.cpu().numpy(),
                    "cls": cls_emb.cpu().numpy().astype(np.float16),
                    "center": center_emb.cpu().numpy().astype(np.float16),
                    "mean": mean_emb.cpu().numpy().astype(np.float16)
                }
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                if batch_size == 1:
                    raise RuntimeError("OOM với Batch Size = 1. Cần giảm kích thước chuỗi (context length) hoặc đổi GPU.")
                
                print(f"\n[!] Cảnh báo OOM. Đang chia đôi batch từ {batch_size} -> {batch_size // 2}...")
                mid = batch_size // 2
                
                # Đệ quy chia lô
                batch_part_1 = {k: v[:mid] for k, v in batch_data.items()}
                batch_part_2 = {k: v[mid:] for k, v in batch_data.items()}
                
                res_1 = self._process_batch_with_oom_protection(batch_part_1, seq_type)
                res_2 = self._process_batch_with_oom_protection(batch_part_2, seq_type)
                
                # Gộp kết quả
                return {k: np.concatenate([res_1[k], res_2[k]]) if k != "names" else res_1[k] + res_2[k] for k in res_1.keys()}
            else:
                raise e

    def run_extraction(self, parquet_path: str, seq_type: str, batch_size: int, output_prefix: str):
        """Chạy toàn bộ file Parquet và lưu 3 file .pt cho 3 chiến lược pooling"""
        dataset = BioSequenceDataset(parquet_path, seq_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_results = {"names": [], "llr": [], "cls": [], "center": [], "mean": []}
        
        print(f"[*] Đang trích xuất: {parquet_path}")
        for batch in tqdm(dataloader, desc="Inference"):
            res = self._process_batch_with_oom_protection(batch, seq_type)
            for k in all_results.keys():
                if k == "names":
                    all_results[k].extend(res[k])
                else:
                    all_results[k].append(res[k])
                    
        print("[*] Đang lưu các ma trận đặc trưng...")
        
        # Nối tất cả numpy arrays và đóng gói thành Tensor
        final_llr = torch.tensor(np.concatenate(all_results["llr"]))
        
        poolings = ["cls", "center", "mean"]
        for p in poolings:
            final_emb = torch.tensor(np.concatenate(all_results[p]))
            output_dict = {
                "metadata": all_results["names"],
                "llr": final_llr,
                "embeddings": final_emb
            }
            save_path = f"{output_prefix}_{p}.pt"
            torch.save(output_dict, save_path)
            print(f"  -> Đã lưu: {save_path}")
            
        print("[+] Hoàn tất trích xuất!\n")
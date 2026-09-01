import torch
import types
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Optional, Sequence


def _install_transformers_compat_shim_if_needed():
    """Backfill API cu ma mot so remote code van import tu transformers.pytorch_utils."""
    try:
        import transformers.pytorch_utils as pu
        if hasattr(pu, "find_pruneable_heads_and_indices"):
            return False

        # Thu nhat: lay implementation tu transformers neu con ton tai.
        try:
            from transformers.modeling_utils import find_pruneable_heads_and_indices
            pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
            return True
        except Exception:
            pass

        # Thu hai: fallback local implementation de tuong thich remote code cu.
        def _compat_find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = _compat_find_pruneable_heads_and_indices
        return True
    except Exception:
        return False


def _install_esm_legacy_config_shim_if_needed():
    """Bo sung cac thuoc tinh config thieu cho remote ESM code doi cu."""
    installed = False

    legacy_defaults = {
        "is_decoder": False,
        "add_cross_attention": False,
        "chunk_size_feed_forward": 0,
        "position_embedding_type": "absolute",
        "use_cache": False,
    }

    # Phu hop cho nhieu version transformers: EsmConfig co the khong export o top-level.
    try:
        from transformers.models.esm.configuration_esm import EsmConfig
        for attr, default_val in legacy_defaults.items():
            if not hasattr(EsmConfig, attr):
                setattr(EsmConfig, attr, default_val)
                installed = True
    except Exception:
        pass

    # Fallback bo sung tren lop config goc de tranh loi voi cac bien the config custom.
    try:
        from transformers.configuration_utils import PretrainedConfig
        for attr, default_val in legacy_defaults.items():
            if not hasattr(PretrainedConfig, attr):
                setattr(PretrainedConfig, attr, default_val)
                installed = True
    except Exception:
        pass

    return installed


def _install_pretrainedmodel_legacy_shim_if_needed():
    """Bo sung thuoc tinh ma remote model class doi cu co the thieu tren transformers moi."""
    installed = False
    try:
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
            installed = True

        if not hasattr(PreTrainedModel, "get_head_mask"):
            def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers

                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                else:
                    raise ValueError("head_mask should be 1D or 2D")

                head_mask = head_mask.to(dtype=self.dtype)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
                return head_mask

            PreTrainedModel.get_head_mask = _compat_get_head_mask
            installed = True
    except Exception:
        pass
    return installed


def _patch_remote_esm_instance_if_needed(model):
    """Patch method thieu tren instance remote EsmModel (khong ke thua day du mixins moi)."""
    try:
        esm_core = getattr(model, "esm", None)
        if esm_core is None:
            return False

        if hasattr(esm_core, "get_head_mask"):
            return False

        def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers

            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError("head_mask should be 1D or 2D")

            model_dtype = next(self.parameters()).dtype
            head_mask = head_mask.to(dtype=model_dtype)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        # Patch o cap class de tat ca instance EsmModel sau do deu co method nay.
        esm_cls = esm_core.__class__
        if not hasattr(esm_cls, "get_head_mask"):
            setattr(esm_cls, "get_head_mask", _compat_get_head_mask)

        # Gan them vao instance hien tai de dam bao hieu luc ngay lap tuc.
        esm_core.get_head_mask = types.MethodType(_compat_get_head_mask, esm_core)
        print("[!] Da patch get_head_mask cho remote EsmModel (class + instance).")
        return True
    except Exception:
        return False

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
        self._announced_strategies = set()
        self.last_token_strategy = None
        
        self._load_engine()

    def _load_engine(self):
        print(f"[*] Đang nạp Tokenizer: {self.model_id}")
        model_id_lower = self.model_id.lower()

        # Mac dinh cho phep remote code vi nhieu model DNA can custom implementation.
        tokenizer_trust_remote = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=tokenizer_trust_remote,
                use_fast=True,
            )
        except Exception as e:
            print(f"[!] Fast tokenizer không khả dụng cho {self.model_id}: {e}")
            print("[!] Thử fallback sang slow tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=tokenizer_trust_remote,
                    use_fast=False,
                )
            except Exception:
                # Fallback cuoi: doi trust_remote_code de mo rong tinh tuong thich.
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=not tokenizer_trust_remote,
                    use_fast=False,
                )
        
        # Sửa lỗi một số tokenizer không có pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        load_dtype = torch.float16
        # NTv3 custom architecture co the pha tron nhieu nhanh dtype noi bo.
        # Neu ep half ngay luc load de gay loi mismatch Float/Half trong Conv1d.
        if "ntv3" in model_id_lower:
            load_dtype = torch.float32
            print("[!] Phat hien NTv3: su dung float32 de tranh xung dot dtype noi bo.")
            
        print(f"[*] Đang nạp Mô hình (Trọng số {str(load_dtype).replace('torch.', '')}): {self.model_id}")
        model_trust_remote = True

        def _load_model(trust_remote_flag: bool):
            if self.model_type == "causal":
                return AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=load_dtype,
                    trust_remote_code=trust_remote_flag
                )
            elif self.model_type == "mlm":
                return AutoModelForMaskedLM.from_pretrained(
                    self.model_id,
                    torch_dtype=load_dtype,
                    trust_remote_code=trust_remote_flag
                )
            else:
                raise ValueError(f"model_type không hợp lệ: {self.model_type}")

        def _load_model_with_compat():
            applied_shims = set()
            for _ in range(6):
                try:
                    return _load_model(model_trust_remote)
                except Exception as e:
                    err = str(e)

                    if model_trust_remote and "find_pruneable_heads_and_indices" in err:
                        if "transformers_compat" in applied_shims:
                            raise
                        print("[!] Phat hien loi tuong thich transformers API cu, dang thu cai shim... ")
                        installed = _install_transformers_compat_shim_if_needed()
                        if not installed:
                            raise RuntimeError(
                                "Khong cai duoc transformers compat shim cho remote code. "
                                "Model nay yeu cau trust_remote_code=True va API cuong thich hop."
                            ) from e
                        applied_shims.add("transformers_compat")
                        continue

                    if model_trust_remote and "'EsmConfig' object has no attribute" in err:
                        if "esm_legacy_config" in applied_shims:
                            raise
                        print("[!] Phat hien EsmConfig thieu thuoc tinh legacy, dang thu cai shim...")
                        installed = _install_esm_legacy_config_shim_if_needed()
                        if not installed:
                            raise RuntimeError(
                                "Khong the bo sung ESM legacy config shim cho transformers hien tai."
                            ) from e
                        applied_shims.add("esm_legacy_config")
                        continue

                    if model_trust_remote and "all_tied_weights_keys" in err:
                        if "pretrainedmodel_legacy" in applied_shims:
                            raise
                        print("[!] Phat hien thieu all_tied_weights_keys, dang thu cai shim PreTrainedModel...")
                        installed = _install_pretrainedmodel_legacy_shim_if_needed()
                        if not installed:
                            raise RuntimeError(
                                "Khong the bo sung PreTrainedModel legacy shim cho transformers hien tai."
                            ) from e
                        applied_shims.add("pretrainedmodel_legacy")
                        continue

                    raise

            raise RuntimeError("Load model that bai sau nhieu lan thu shim tuong thich.")

        if self.model_type in ("causal", "mlm"):
            self.model = _load_model_with_compat()
        else:
            raise ValueError(f"model_type không hợp lệ: {self.model_type}")

        _patch_remote_esm_instance_if_needed(self.model)
            
        self.model.eval()
        self.model.to(self.device)
        print("[+] Nạp mô hình thành công!\n")

    def _infer_variant_indices_from_token_diffs(self, input_ids_ref, input_ids_alt, attention_mask=None):
        """Fallback cho DNA khi tokenizer không có offset: suy ra vùng đột biến từ token thay đổi giữa ref/alt."""
        batch_size = input_ids_ref.shape[0]
        variant_indices = []
        special_ids = set(self.tokenizer.all_special_ids or [])

        for i in range(batch_size):
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
            else:
                valid_len = int(input_ids_ref.shape[1])

            ref_tokens = input_ids_ref[i, :valid_len].tolist()
            alt_tokens = input_ids_alt[i, :valid_len].tolist()

            diff_positions = [
                j for j, (r_tok, a_tok) in enumerate(zip(ref_tokens, alt_tokens))
                if r_tok != a_tok and r_tok not in special_ids and a_tok not in special_ids
            ]

            if not diff_positions:
                raise ValueError(
                    "Không tìm thấy token thay đổi giữa ref/alt cho DNA khi tokenizer không có offset_mapping."
                )

            # Với tokenizer k-mer/subword, mutation thường làm thay đổi một cụm token lân cận.
            # Chọn phần tử giữa của cụm để đại diện vị trí đột biến trong không gian token.
            variant_indices.append(diff_positions[len(diff_positions) // 2])

        return variant_indices

    def _announce_strategy_once(self, seq_type: str, strategy_name: str):
        key = (seq_type, strategy_name)
        if key in self._announced_strategies:
            return
        self._announced_strategies.add(key)
        print(
            f"[Token Mapping] model={self.model_id} | seq_type={seq_type} | strategy={strategy_name}"
        )

    def probe_token_mapping_strategy(self, seq_type: str):
        """Probe nhẹ để báo trước chiến lược map token của model hiện tại."""
        target_char_pos = 300 if seq_type == "dna" else 50
        base_char = "A" if seq_type == "dna" else "M"
        alt_char = "T" if seq_type == "dna" else "V"
        seq_len = target_char_pos + 5

        ref = base_char * seq_len
        alt = ref[:target_char_pos] + alt_char + ref[target_char_pos + 1:]

        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.tokenizer.is_fast:
            kwargs["return_offsets_mapping"] = True

        tokenized_ref = self.tokenizer([ref], **kwargs)
        tokenized_alt = self.tokenizer([alt], **kwargs)

        self.get_variant_token_index(
            tokenized_ref,
            seq_type=seq_type,
            target_char_positions=[target_char_pos],
            tokenized_alt=tokenized_alt,
        )
        return self.last_token_strategy

    def get_variant_token_index(
        self,
        tokenized_outputs,
        seq_type: str,
        target_char_positions: Optional[Sequence[int]] = None,
        tokenized_alt=None,
    ):
        """
        Tìm chỉ số (index) của token chứa vị trí đột biến.
        seq_type = "dna" (đột biến ở index 300) hoặc "protein" (đột biến ở index 50)
        """
        default_target_char_pos = 300 if seq_type == "dna" else 50
        batch_size = tokenized_outputs["input_ids"].shape[0]
        if target_char_positions is None:
            target_char_positions = [default_target_char_pos] * batch_size
        else:
            target_char_positions = [int(p) for p in target_char_positions]
            if len(target_char_positions) != batch_size:
                raise ValueError(
                    f"Số lượng vị trí đột biến ({len(target_char_positions)}) không khớp batch size ({batch_size})."
                )

        variant_indices = []

        model_id_lower = self.model_id.lower()

        # Ưu tiên offset mapping nếu tokenizer hỗ trợ (đáng tin cậy nhất).
        if "offset_mapping" in tokenized_outputs:
            strategy_name = "offset_mapping"
            for i in range(batch_size):
                target_char_pos = target_char_positions[i]
                offsets = tokenized_outputs["offset_mapping"][i].cpu().numpy()
                idx_found = -1
                for idx, (start, end) in enumerate(offsets):
                    if start <= target_char_pos < end:
                        idx_found = idx
                        break
                if idx_found < 0:
                    raise ValueError(
                        f"Không ánh xạ được vị trí đột biến char_pos={target_char_pos} sang token index ở sample {i}. "
                        "Có thể chuỗi đã bị truncation hoặc vị trí đột biến không hợp lệ."
                    )
                variant_indices.append(idx_found)
        else:
            if seq_type == "dna":
                if tokenized_alt is None:
                    raise ValueError(
                        f"Tokenizer của {self.model_id} không hỗ trợ offset_mapping cho DNA và thiếu tokenized_alt. "
                        "Không thể ánh xạ mutation token một cách an toàn."
                    )

                # DNA model không offset: fallback bằng token-diff ref/alt.
                strategy_name = "dna_token_diff_fallback"
                variant_indices = self._infer_variant_indices_from_token_diffs(
                    tokenized_outputs["input_ids"],
                    tokenized_alt["input_ids"],
                    attention_mask=tokenized_outputs.get("attention_mask"),
                )
            else:
                strategy_name = "protein_char_fallback"
                # Fallback thủ công cho mô hình Protein (đa phần char-level như ESM).
                # token_index = char_index + số special tokens ở đầu chuỗi.
                for i in range(batch_size):
                    target_char_pos = target_char_positions[i]
                    input_ids_list = tokenized_outputs["input_ids"][i].tolist()
                    # Đếm số lượng special tokens (như <cls>) ở đầu chuỗi
                    special_start = 0
                    while special_start < len(input_ids_list) and input_ids_list[special_start] in self.tokenizer.all_special_ids:
                        special_start += 1
                    variant_indices.append(target_char_pos + special_start)

        # Validate cứng để không bao giờ dùng nhầm index âm hoặc vượt vùng token thật.
        attention_mask = tokenized_outputs.get("attention_mask")
        for i, idx in enumerate(variant_indices):
            if idx < 0:
                raise ValueError(f"variant token index âm tại sample {i}: {idx}")
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
            else:
                valid_len = int(tokenized_outputs["input_ids"].shape[1])
            if idx >= valid_len:
                raise ValueError(
                    f"variant token index={idx} vượt token length hữu hiệu={valid_len} tại sample {i}. "
                    "Có thể do truncation hoặc vị trí đột biến nằm ngoài chuỗi."
                )

            self.last_token_strategy = strategy_name
            self._announce_strategy_once(seq_type, strategy_name)
                
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
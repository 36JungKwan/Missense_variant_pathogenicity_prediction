"""Tải trọng số Foundation Models về HF_HOME (backend/.env). Chạy lại được: tự resume phần dang dở.

    python scripts/download_models.py                      # DNA_MODEL_ID + PROT_MODEL_ID trong .env
    python scripts/download_models.py <repo_id> [...]      # tải các repo chỉ định
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402  (đặt HF_HOME trước khi import huggingface_hub)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # HTTP thường: resume ổn định theo Range
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import list_repo_files, snapshot_download  # noqa: E402

# Chỉ lấy tokenizer/config + MỘT file trọng số PyTorch (bỏ tf_model.h5 và bản trùng lặp).


def ensure_safetensors(snapshot_dir: str) -> None:
    """Repo chỉ có pytorch_model.bin (vd NT-v1): transformers>=4.50 không nạp .bin khi torch<2.6
    (CVE-2025-32434), nên chuyển sang model.safetensors ngay trong cache."""
    target = os.path.join(snapshot_dir, "model.safetensors")
    source = os.path.join(snapshot_dir, "pytorch_model.bin")
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return
    if not os.path.isfile(source):
        return
    import torch
    from safetensors.torch import save_file

    state = torch.load(source, map_location="cpu", weights_only=True)
    state = {k: v.contiguous() for k, v in state.items()}
    if os.path.lexists(target):
        os.remove(target)
    save_file(state, target, metadata={"format": "pt"})
    # HF cache ghi nhớ "repo không có model.safetensors" -> xoá dấu đó để transformers thấy file mới.
    snapshot = os.path.normpath(snapshot_dir)
    marker = os.path.join(
        os.path.dirname(os.path.dirname(snapshot)), ".no_exist", os.path.basename(snapshot), "model.safetensors"
    )
    if os.path.isfile(marker):
        os.remove(marker)
    print(f"[converted] {source} -> model.safetensors", flush=True)


def download(repo_id: str, retries: int = 20) -> None:
    for attempt in range(1, retries + 1):
        try:
            names = list_repo_files(repo_id)
            weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
            path = snapshot_download(repo_id, allow_patterns=["*.json", "*.txt", "*.py", weight])
            ensure_safetensors(path)
            print(f"[OK] {repo_id} -> {path}", flush=True)
            return
        except Exception as exc:  # mạng chập chờn -> thử lại, phần đã tải được giữ nguyên
            print(f"[retry {attempt}/{retries}] {repo_id}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(10)
    raise SystemExit(f"[FAILED] {repo_id}")


if __name__ == "__main__":
    for repo in sys.argv[1:] or [settings.DNA_MODEL_ID, settings.PROT_MODEL_ID]:
        download(repo)
    print("[DONE]", flush=True)

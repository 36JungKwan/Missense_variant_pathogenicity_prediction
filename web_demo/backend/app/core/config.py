import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

# Khoá registry: 3 bit (use_dna, use_prot, use_biogeom) hoặc danh sách modality "dna,prot,bio".
DEFAULT_CHECKPOINT_REGISTRY: Dict[str, str] = {
    "111": "artifacts/best_model_B_with_geometry.pth",
    "110": "artifacts/best_model_110.pth",
    "101": "artifacts/best_model_101.pth",
    "011": "artifacts/best_model_011.pth",
    "100": "artifacts/best_model_100.pth",
    "010": "artifacts/best_model_010.pth",
    "001": "artifacts/best_model_001.pth",
    # Checkpoint dự phòng (không nằm trong 7 tổ hợp 3 nút, dùng khi UI tắt riêng Geometry).
    "dna,prot,bio": "artifacts/best_model_A_no_geometry.pth",
}

# Ngưỡng tối ưu MCC lấy từ router_expert_validation_metrics.csv của nhóm.
DEFAULT_DECISION_THRESHOLDS: Dict[str, float] = {
    "111": 0.637582,
    "dna,prot,bio": 0.707937,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str
    API_V1_STR: str
    DATABASE_URL: str
    BACKEND_CORS_ORIGINS: List[str]
    LOG_LEVEL: str = "INFO"
    # Nạp sẵn FM, FASTA, FAISS, checkpoint 111 và scaler trong lifespan để request đầu không bị cold-start.
    PRELOAD_ON_STARTUP: bool = True

    # --- Ensembl VEP (Docker, offline) ---
    VEP_CONTAINER_NAME: str = "rbnet_vep"
    VEP_TMP_DIR: str = "F:/BioData/vep_data/tmp_io"
    VEP_TMP_CONTAINER_DIR: str = "/tmp_io"
    VEP_CACHE_VERSION: int = 116
    VEP_CONTAINER_CACHE_DIR: str = "/vep_cache"
    VEP_CONTAINER_PLUGIN_DIR: str = "/vep_cache/Plugins"
    VEP_CONTAINER_FASTA: str = "/vep_resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    DBNSFP_CONTAINER_PATH: str = "/vep_resources/dbNSFP5.3.1a_grch38.gz"
    VEP_TIMEOUT_SECONDS: int = 900
    # Batch VCF: bật `--fork` khi số biến thể vượt ngưỡng (1 biến thể thì fork chỉ làm chậm hơn).
    VEP_FORK_THRESHOLD: int = 50
    VEP_FORK_WORKERS: int = 4

    # --- Tài nguyên tham chiếu trên host ---
    HG38_FASTA_PATH: str = "F:/VEP_RESOURCES/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    HG38_PEP_FASTA_PATH: str = "F:/VEP_RESOURCES/Homo_sapiens.GRCh38.pep.all.fa"
    SPLICEAI_VCF_PATH: str = (
        "F:/VEP_RESOURCES/spliceai_scores.raw.snv.ensembl_mane_v1.4.grch38.vcf.gz"
    )

    # --- Foundation models ---
    # Đối chiếu thống kê embedding với scaler nt_v1_500m_dna_1001_center: khớp bản human-ref.
    DNA_MODEL_ID: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    PROT_MODEL_ID: str = "facebook/esm2_t33_650M_UR50D"
    DNA_CONFIG_ID: str = "nt_v1_500m_dna_1001_center"
    PROT_CONFIG_ID: str = "esm2_650m_prot_501_center"
    DNA_WINDOW: int = 1001
    PROT_WINDOW: int = 501
    DNA_PAD_CHAR: str = "N"
    PROT_PAD_CHAR: str = "X"
    INFERENCE_DEVICE: str = "auto"  # auto | cuda | cpu
    # Độ chính xác FM trên GPU. fp32: embedding khớp bản CPU đã kiểm chứng (cosine >= 0.9999998, cùng xác suất),
    # ESM ~330 ms/biến thể. fp16: nhanh ~3x, VRAM 2.7 GB thay vì 5.5 GB, nhưng ESM cosine ~0.99988 (xác suất lệch ~2e-4).
    FM_CUDA_DTYPE: str = "fp32"
    # Số biến thể gộp vào một forward pass khi chạy Batch VCF (mỗi biến thể = 3 chuỗi: masked_ref, ref, alt).
    FM_BATCH_VARIANTS: int = 8
    HF_HOME: Optional[str] = None  # nơi lưu trọng số HuggingFace (nên đặt trên ổ còn trống)

    # --- Artifact của nhóm ---
    MODEL_ARTIFACTS_DIR: str = "artifacts"  # chứa *_geom_v2_scalers.pkl
    BIO_SCALER_DIR: str = "artifacts"  # chứa 6 file bio_*.pkl của normalizer.transform_bio
    FAISS_INDEX_DIR: str = "artifacts/faiss_indexes"  # {config_id}.index (context train3) để tính LID
    # k=64 khớp *_geom_v2_scalers.pkl (median + khoảng 5-95% của log1p(LID) trên index train3); k=32 lệch phân phối.
    GEOM_K_NEIGHBORS: int = 64
    MODEL_CHECKPOINT_REGISTRY: Dict[str, str] = DEFAULT_CHECKPOINT_REGISTRY
    MODEL_DECISION_THRESHOLDS: Dict[str, float] = DEFAULT_DECISION_THRESHOLDS
    ROUTER_TEMPERATURE: float = 0.3074146877278755  # hparam không nằm trong state_dict
    SHAP_PERMUTATIONS: int = 128
    SHAP_EVAL_CHUNK: int = 2048  # số dòng mask mỗi lần forward khi ước lượng Shapley
    SPLICEAI_WINDOW_SIZE: int = 1024  # bp mỗi truy vấn tabix (chỉ giải nén các block BGZF quanh vị trí cần tra)

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("MODEL_CHECKPOINT_REGISTRY", mode="after")
    @classmethod
    def merge_registry_defaults(cls, v: Dict[str, str]) -> Dict[str, str]:
        return {**DEFAULT_CHECKPOINT_REGISTRY, **v}

    @field_validator("MODEL_DECISION_THRESHOLDS", mode="after")
    @classmethod
    def merge_threshold_defaults(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {**DEFAULT_DECISION_THRESHOLDS, **v}

    def resolve_path(self, raw: str) -> Path:
        """Đường dẫn tương đối được tính từ thư mục backend/."""
        path = Path(raw)
        return path if path.is_absolute() else BACKEND_ROOT / path


settings = Settings()

# Phải đặt trước khi transformers/huggingface_hub được import lần đầu.
if settings.HF_HOME:
    import os

    os.environ.setdefault("HF_HOME", settings.HF_HOME)

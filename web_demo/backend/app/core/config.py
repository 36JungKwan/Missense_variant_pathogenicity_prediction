import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

# Khoá registry: 3 bit (use_dna, use_prot, use_biogeom).
DEFAULT_CHECKPOINT_REGISTRY: Dict[str, str] = {
    "111": "artifacts/trimvpp/checkpoints/trimvpp_111.pth",
    "110": "artifacts/trimvpp/checkpoints/trimvpp_110.pth",
    "101": "artifacts/trimvpp/checkpoints/trimvpp_101.pth",
    "011": "artifacts/trimvpp/checkpoints/trimvpp_011.pth",
    "100": "artifacts/trimvpp/checkpoints/trimvpp_100.pth",
    "010": "artifacts/trimvpp/checkpoints/trimvpp_010.pth",
    "001": "artifacts/trimvpp/checkpoints/trimvpp_001.pth",
}

# Ngưỡng tối ưu MCC trên tập validation (Decision_Threshold trong *_validation_metrics.csv của từng model).
DEFAULT_DECISION_THRESHOLDS: Dict[str, float] = {
    "111": 0.4791317582130432,
    "110": 0.6340423822402954,
    "101": 0.7515003681182861,
    "011": 0.5660408735275269,
    "100": 0.4819318056106567,
    "010": 0.7382758259773254,
    "001": 0.6309370994567871,
}

# MCC trên tập Test nội bộ (Train3Val_Test, 6.095 biến thể) trong *_test_metrics.csv; chỉ để hiển thị.
DEFAULT_TEST_MCC: Dict[str, float] = {
    "111": 0.8424,
    "110": 0.7259,
    "101": 0.7676,
    "011": 0.8390,
    "100": 0.2569,
    "010": 0.5873,
    "001": 0.7615,
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
    PROT_MODEL_ID: str = "facebook/esm1b_t33_650M_UR50S"
    DNA_CONFIG_ID: str = "nt_v1_500m_dna_1001_center"
    PROT_CONFIG_ID: str = "esm1b_650m_prot_501_center"
    DNA_WINDOW: int = 1001
    PROT_WINDOW: int = 501
    DNA_PAD_CHAR: str = "N"
    PROT_PAD_CHAR: str = "X"
    INFERENCE_DEVICE: str = "auto"  # auto | cuda | cpu
    # Độ chính xác FM trên GPU. fp16: giống pipeline huấn luyện của nhóm (load_dtype=float16, không autocast).
    FM_CUDA_DTYPE: str = "fp16"
    # Số biến thể gộp vào một forward pass khi chạy Batch VCF (mỗi biến thể = 2 chuỗi: ref, alt).
    FM_BATCH_VARIANTS: int = 8
    # Batch Screening chia file thành các khối: VEP + SpliceAI + FM + dự đoán theo từng khối, kết quả gửi ngay;
    # VEP của khối kế tiếp chạy song song với phần GPU của khối hiện tại.
    BATCH_CHUNK_VARIANTS: int = 64
    HF_HOME: Optional[str] = None  # nơi lưu trọng số HuggingFace (nên đặt trên ổ còn trống)

    # --- Artifact của nhóm ---
    MODEL_ARTIFACTS_DIR: str = "artifacts/trimvpp/geom_scalers"  # chứa *_geom_v2_scalers.pkl
    BIO_SCALER_DIR: str = "artifacts"  # chứa 6 file bio_*.pkl của normalizer.transform_bio
    FAISS_INDEX_DIR: str = "artifacts/trimvpp/faiss_indexes"  # {config_id}.index (context train3) để tính LID
    # k=64 khớp *_geom_v2_scalers.pkl (median + khoảng 5-95% của log1p(LID) trên index train3); k=32 lệch phân phối.
    GEOM_K_NEIGHBORS: int = 64
    MODEL_CHECKPOINT_REGISTRY: Dict[str, str] = DEFAULT_CHECKPOINT_REGISTRY
    MODEL_DECISION_THRESHOLDS: Dict[str, float] = DEFAULT_DECISION_THRESHOLDS
    MODEL_TEST_MCC: Dict[str, float] = DEFAULT_TEST_MCC
    # Integrated Gradients (notebook 19): baseline train3 256 mẫu seed 2026, 64 bước, chạy lại 128 bước nếu
    # |residual| / |logit - logit_baseline| > 5%; IG_INTERNAL_BATCH = số điểm nội suy mỗi lần forward/backward.
    IG_BASELINE_PATH: str = "artifacts/trimvpp/ig_baseline_train3_seed2026.pt"
    IG_STEPS: int = 64
    IG_RETRY_STEPS: int = 128
    IG_RETRY_THRESHOLD: float = 0.05
    IG_INTERNAL_BATCH: int = 8
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

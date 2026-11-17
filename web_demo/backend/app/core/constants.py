import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from app.core.errors import InvalidVariantError, NoModalitySelectedError

PredictionLabel = Literal["Pathogenic", "Benign"]
# "Uncertain": ClinVar không có nhãn rõ ràng (VUS / conflicting / chưa có bản ghi).
ClassificationLabel = Literal["Pathogenic", "Benign", "Uncertain"]

PATHOGENIC_THRESHOLD: float = 0.5

MODALITY_ORDER: Tuple[str, ...] = ("dna", "prot", "bio", "geom")

# Thứ tự cột đúng theo pipeline huấn luyện (BIO_CORE_COLUMNS + SPLICEAI_COLUMNS, rồi geometry DNA -> Protein).
BIO_CORE_COLUMNS: List[str] = [
    "AF", "gnomADe_AF",
    "phyloP100way_vertebrate", "phyloP470way_mammalian", "phyloP17way_primate",
    "phastCons100way_vertebrate", "phastCons470way_mammalian", "phastCons17way_primate",
    "GERP++_RS", "GERP++_NR", "GERP_92_mammals",
]
SPLICEAI_DS_COLUMNS: List[str] = [
    "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL",
]
SPLICEAI_COLUMNS: List[str] = SPLICEAI_DS_COLUMNS + [
    "SpliceAI_pred_DP_AG", "SpliceAI_pred_DP_AL", "SpliceAI_pred_DP_DG", "SpliceAI_pred_DP_DL",
    "SpliceAI_pred_DS_max",
]
GEOM_BRANCH_COLUMNS: List[str] = ["LVD_L2", "LVD_Cosine", "LID", "LVD_Relative"]
GEOM_COLUMNS: List[str] = [f"{p}_{c}" for p in ("dna", "prot") for c in GEOM_BRANCH_COLUMNS]

# Số chiều nhánh Bio trong checkpoint -> bộ cột tương ứng.
BIO_COLUMN_SETS = {
    11: BIO_CORE_COLUMNS,
    15: BIO_CORE_COLUMNS + SPLICEAI_DS_COLUMNS,
    20: BIO_CORE_COLUMNS + SPLICEAI_COLUMNS,
}


def compute_ablation_mask(
    dna_enabled: bool,
    protein_enabled: bool,
    bio_enabled: bool,
    geo_enabled: bool,
) -> int:
    """4 toggle -> bitmask 0..15 (bit 3 DNA, bit 2 Protein, bit 1 Bio, bit 0 Geometry)."""
    return (
        (int(dna_enabled) << 3)
        | (int(protein_enabled) << 2)
        | (int(bio_enabled) << 1)
        | int(geo_enabled)
    )


def mask_to_modalities(mask: int) -> Tuple[str, ...]:
    mods = tuple(m for bit, m in zip((8, 4, 2, 1), MODALITY_ORDER) if mask & bit)
    if not mods:
        raise NoModalitySelectedError(
            "At least one modality (DNA, Protein, Bio+Geom) must be enabled."
        )
    return mods


def normalize_clinvar_label(raw_label: Optional[str]) -> ClassificationLabel:
    """Quy nhãn ClinVar (kể cả chuỗi clnsig thô của dbNSFP/VEP) về Pathogenic / Benign / Uncertain."""
    cleaned = (raw_label or "").strip().lower()
    if not cleaned or "conflicting" in cleaned:
        return "Uncertain"
    has_patho = "pathogenic" in cleaned
    has_benign = "benign" in cleaned
    if has_patho and not has_benign:
        return "Pathogenic"
    if has_benign and not has_patho:
        return "Benign"
    return "Uncertain"


@dataclass(frozen=True)
class VariantKey:
    chrom: str  # không có tiền tố "chr", ty thể = "MT" (theo Ensembl)
    pos: int
    ref: str
    alt: str

    @property
    def variant_id(self) -> str:
        chrom = "M" if self.chrom == "MT" else self.chrom
        return f"chr{chrom}:{self.pos}:{self.ref}>{self.alt}"


VARIANT_ID_PATTERN = re.compile(
    r"^(?:chr)?([0-9]{1,2}|X|Y|M|MT)[:_\-\s](\d+)[:_\-\s]([ACGTN]+)(?:>|/|[:_\-\s])([ACGTN]+)$",
    re.IGNORECASE,
)


def parse_variant_id(raw_id: str) -> VariantKey:
    match = VARIANT_ID_PATTERN.match((raw_id or "").strip())
    if not match:
        raise InvalidVariantError(
            f"Cannot parse variant '{raw_id}'. Expected hg38 coordinates like chr7:140753336:A>T."
        )
    chrom = match.group(1).upper()
    if chrom == "M":
        chrom = "MT"
    return VariantKey(chrom=chrom, pos=int(match.group(2)), ref=match.group(3).upper(), alt=match.group(4).upper())

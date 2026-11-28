"""Cắt cửa sổ DNA 1001 bp (hg38) và Protein 501 aa (Ensembl pep) có tâm tại vị trí đột biến."""
from __future__ import annotations

import threading
from typing import Optional, Tuple

from pyfaidx import Fasta

from app.core.config import settings
from app.core.constants import VariantKey
from app.core.errors import FeatureArtifactMissingError, InvalidVariantError

_lock = threading.Lock()
_genome: Optional[Fasta] = None
_proteome: Optional[Fasta] = None


def _open(path: str, **kwargs) -> Fasta:
    try:
        return Fasta(path, **kwargs)
    except Exception as exc:
        raise FeatureArtifactMissingError(f"Cannot open FASTA '{path}': {exc}") from exc


def _get_genome() -> Fasta:
    global _genome
    with _lock:
        if _genome is None:
            _genome = _open(settings.HG38_FASTA_PATH, as_raw=True, sequence_always_upper=True)
        return _genome


def _get_proteome() -> Fasta:
    global _proteome
    with _lock:
        if _proteome is None:
            # Header dạng "ENSP00000493543.1 pep ..." -> khoá không kèm version để khớp cột ENSP của VEP.
            # Lần đầu pyfaidx sẽ tạo file .fai cạnh FASTA (vài giây).
            _proteome = _open(
                settings.HG38_PEP_FASTA_PATH,
                as_raw=True,
                key_function=lambda name: name.split(".")[0],
                duplicate_action="first",
            )
        return _proteome


def preload() -> None:
    """Mở sẵn hg38 + proteome (đọc .fai) để request đầu không mất ~2,7 s."""
    _get_genome()
    _get_proteome()


GRCH38_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def validate_coordinates(variant: VariantKey) -> None:
    """Kiểm tra toạ độ TRƯỚC khi gọi VEP (chỉ đọc FASTA, vài micro giây): REF khác ALT, nhiễm sắc thể có trong
    GRCh38, vị trí nằm trong nhiễm sắc thể, REF khớp hg38. Toạ độ sai được báo đúng lý do thay vì để VEP trả về
    một consequence gây hiểu nhầm (intergenic, not_annotated...)."""
    vid = variant.variant_id
    if variant.ref == variant.alt:
        raise InvalidVariantError(f"{vid}: REF and ALT alleles are identical, so this is not a variant.")
    genome = _get_genome()
    if variant.chrom not in GRCH38_CHROMOSOMES or variant.chrom not in genome:
        raise InvalidVariantError(
            f"{vid}: chromosome '{variant.chrom}' does not exist in GRCh38 (expected 1-22, X, Y or MT)."
        )
    contig = genome[variant.chrom]
    length = len(contig)
    end = variant.pos - 1 + len(variant.ref)
    if variant.pos < 1 or end > length:
        raise InvalidVariantError(
            f"{vid}: position {variant.pos:,} is outside chromosome {variant.chrom} (length {length:,} bp)."
        )
    base = str(contig[variant.pos - 1:end]).upper()
    if base != variant.ref:
        raise InvalidVariantError(
            f"{vid}: REF allele '{variant.ref}' does not match hg38 base '{base}' at chr{variant.chrom}:{variant.pos}. "
            "Check the genome build (GRCh38 required).",
            genome_base=base,
        )


def _centered_window(sequence_length: int, center0: int, window: int) -> Tuple[int, int, int, int]:
    """Trả (start, end, pad_left, pad_right) cho cửa sổ [center-half, center+half]."""
    half = window // 2
    start, end = center0 - half, center0 + half + 1
    pad_left, pad_right = max(0, -start), max(0, end - sequence_length)
    return max(0, start), min(sequence_length, end), pad_left, pad_right


def dna_windows(variant: VariantKey) -> Tuple[str, str, int]:
    """Trả (ref_dna, alt_dna, center_index). Chỉ hỗ trợ SNV; kiểm tra REF khớp hg38."""
    if len(variant.ref) != 1 or len(variant.alt) != 1 or variant.ref == variant.alt:
        raise InvalidVariantError(
            f"{variant.variant_id}: only single-nucleotide substitutions are supported."
        )
    genome = _get_genome()
    if variant.chrom not in genome:
        raise InvalidVariantError(f"{variant.variant_id}: contig '{variant.chrom}' not found in the hg38 FASTA.")
    contig = genome[variant.chrom]
    contig_length = len(contig)
    center0 = variant.pos - 1
    if not 0 <= center0 < contig_length:
        raise InvalidVariantError(f"{variant.variant_id}: position is outside chromosome {variant.chrom}.")

    start, end, pad_left, pad_right = _centered_window(contig_length, center0, settings.DNA_WINDOW)
    pad = settings.DNA_PAD_CHAR
    ref_seq = pad * pad_left + str(contig[start:end]).upper() + pad * pad_right
    center = settings.DNA_WINDOW // 2
    if ref_seq[center] != variant.ref:
        raise InvalidVariantError(
            f"{variant.variant_id}: REF allele '{variant.ref}' does not match hg38 base "
            f"'{ref_seq[center]}' at chr{variant.chrom}:{variant.pos}. Check the genome build (GRCh38 required).",
            genome_base=ref_seq[center],
        )
    alt_seq = ref_seq[:center] + variant.alt + ref_seq[center + 1:]
    return ref_seq, alt_seq, center


def protein_windows(ensp: str, protein_position: int, ref_aa: str, alt_aa: str) -> Tuple[str, str, int]:
    """Trả (ref_prot, alt_prot, center_index) trên transcript MANE Select/Canonical mà VEP đã chọn."""
    proteome = _get_proteome()
    key = ensp.split(".")[0]
    if key not in proteome:
        raise InvalidVariantError(f"Protein {ensp} not found in the Ensembl peptide FASTA.")
    protein = str(proteome[key][:]).upper()
    center0 = protein_position - 1
    if not 0 <= center0 < len(protein):
        raise InvalidVariantError(f"Protein position {protein_position} is outside {ensp} (length {len(protein)}).")
    if protein[center0] != ref_aa:
        raise InvalidVariantError(
            f"{ensp}: reference residue '{protein[center0]}' at position {protein_position} does not match "
            f"VEP amino acid '{ref_aa}' (peptide FASTA and VEP cache releases differ?)."
        )
    start, end, pad_left, pad_right = _centered_window(len(protein), center0, settings.PROT_WINDOW)
    pad = settings.PROT_PAD_CHAR
    ref_seq = pad * pad_left + protein[start:end] + pad * pad_right
    center = settings.PROT_WINDOW // 2
    alt_seq = ref_seq[:center] + alt_aa + ref_seq[center + 1:]
    return ref_seq, alt_seq, center

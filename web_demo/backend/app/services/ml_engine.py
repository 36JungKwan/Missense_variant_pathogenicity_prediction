"""Suy luận Pure DL + giải thích Shapley trên checkpoint đã nạp từ registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.config import settings
from app.core.constants import PredictionLabel
from app.core.log import timed_stage
from app.services.checkpoint_registry import LoadedModel
from app.services.feature_pipeline import VariantFeatures, normalize_tabular

_GEOM_META = {
    "LVD_L2": ("LVD (L2 Distance)", "Euclidean shift between ref and alt embeddings"),
    "LVD_Cosine": ("LVD (Cosine)", "Cosine divergence between ref and alt embeddings"),
    "LID": ("Local Intrinsic Dim", "Local manifold dimensionality around the variant delta vector"),
    "LVD_Relative": ("Relative LVD", "Embedding displacement relative to the reference norm"),
}
_BRANCH_LABEL = {"dna": "DNA", "prot": "Protein"}
_POPULATION_AF_COLUMNS = ("AF", "gnomADe_AF")
_BIO_META: Dict[str, Tuple[str, str]] = {
    "AF": ("1000 Genomes AF", "Allele frequency in 1000 Genomes (log-scaled)"),
    "gnomADe_AF": ("gnomAD Exomes AF", "Allele frequency in gnomAD exomes (log-scaled)"),
    "GERP++_RS": ("GERP++ RS", "Rejected substitutions evolutionary constraint score"),
    "GERP++_NR": ("GERP++ NR", "Neutral substitution rate at the site"),
    "GERP_92_mammals": ("GERP 92 mammals", "GERP constraint across 92 mammals"),
    "SpliceAI_pred_DS_AG": ("SpliceAI Acceptor Gain", "Predicted splice acceptor gain delta score"),
    "SpliceAI_pred_DS_AL": ("SpliceAI Acceptor Loss", "Predicted splice acceptor loss delta score"),
    "SpliceAI_pred_DS_DG": ("SpliceAI Donor Gain", "Predicted splice donor gain delta score"),
    "SpliceAI_pred_DS_DL": ("SpliceAI Donor Loss", "Predicted splice donor loss delta score"),
    "SpliceAI_pred_DS_max": ("SpliceAI Max Delta", "Largest of the four SpliceAI delta scores"),
}


def _feature_meta(column: str) -> Tuple[str, str]:
    for prefix, label in _BRANCH_LABEL.items():
        if column.startswith(f"{prefix}_") and column[len(prefix) + 1:] in _GEOM_META:
            name, desc = _GEOM_META[column[len(prefix) + 1:]]
            return f"{label} {name}", f"{desc} ({label} foundation model space)"
    if column in _BIO_META:
        return _BIO_META[column]
    if column.startswith("SpliceAI_pred_DP_"):
        return f"SpliceAI Position {column[-2:]}", "Predicted splice-site position offset"
    if column.startswith(("phyloP", "phastCons")):
        return column.replace("way_", "-way ").replace("phyloP", "PhyloP ").replace("phastCons", "PhastCons "), \
            "Evolutionary conservation score from dbNSFP"
    return column.replace("_", " "), f"Tabular feature {column}"


@dataclass
class PredictionOutput:
    probability: float
    label: PredictionLabel
    shap_items: List[Dict[str, Any]]


def _shapley_values(value_fn, n_players: int, n_permutations: int, seed: int = 0) -> np.ndarray:
    """Ước lượng Shapley bằng lấy mẫu hoán vị đối xứng (antithetic); thoả tính hiệu quả:
    tổng đóng góp = f(đủ đặc trưng) - f(baseline). `value_fn` nhận ma trận mask (rows, n_players)."""
    rng = np.random.default_rng(seed)
    orders: List[np.ndarray] = []
    for _ in range(max(1, n_permutations // 2)):
        order = rng.permutation(n_players)
        orders.extend([order, order[::-1]])

    masks = np.zeros((len(orders), n_players + 1, n_players), dtype=np.float32)
    for row, order in enumerate(orders):
        for step, player in enumerate(order):
            masks[row, step + 1:, player] = 1.0
    values = value_fn(masks.reshape(-1, n_players)).reshape(len(orders), n_players + 1)

    phi = np.zeros(n_players, dtype=np.float64)
    for row, order in enumerate(orders):
        phi[order] += values[row, 1:] - values[row, :-1]
    return phi / len(orders)


@torch.inference_mode()
def predict(loaded: LoadedModel, features: VariantFeatures) -> PredictionOutput:
    device = loaded.device
    mods = loaded.modalities
    needs_seq = loaded.architecture == "RouterExpertCorrection"

    dna = torch.tensor(features.dna_alt - features.dna_ref, device=device) if ("dna" in mods or needs_seq) else None
    prot = torch.tensor(features.prot_alt - features.prot_ref, device=device) if ("prot" in mods or needs_seq) else None
    columns = loaded.tabular_columns
    tabular_np = normalize_tabular(features.raw, loaded.bio_columns, loaded.geom_columns) if columns else None
    tabular = torch.tensor(tabular_np, device=device) if columns else None

    def forward(dna_b, prot_b, tab_b) -> torch.Tensor:
        return loaded.forward_logit(dna_b, prot_b, tab_b).float()

    with timed_stage("shap"):
        logit = forward(
            dna[None] if dna is not None else None,
            prot[None] if prot is not None else None,
            tabular[None] if tabular is not None else None,
        )
        probability = float(torch.sigmoid(logit)[0])

    # Người chơi Shapley: khối embedding DNA, khối embedding Protein, và từng đặc trưng bảng.
    # Baseline = 0: delta embedding bằng 0 (không đổi) và giá trị trung tâm của tập train sau chuẩn hoá.
    players: List[Tuple[str, Optional[str]]] = []
    if dna is not None:
        players.append(("dna_block", None))
    if prot is not None:
        players.append(("prot_block", None))
    players.extend(("tabular", column) for column in columns)
    n_seq = len(players) - len(columns)

    def value_fn(masks: np.ndarray) -> np.ndarray:
        outputs = []
        chunk = settings.SHAP_EVAL_CHUNK
        for start in range(0, len(masks), chunk):
            m = torch.tensor(masks[start:start + chunk], device=device)
            col = 0
            dna_b = prot_b = tab_b = None
            if dna is not None:
                dna_b = m[:, col:col + 1] * dna[None]
                col += 1
            if prot is not None:
                prot_b = m[:, col:col + 1] * prot[None]
                col += 1
            if tabular is not None:
                tab_b = m[:, n_seq:] * tabular[None]
            outputs.append(forward(dna_b, prot_b, tab_b).cpu().numpy())
        return np.concatenate(outputs)

    with timed_stage("shap"):
        phi = _shapley_values(value_fn, len(players), settings.SHAP_PERMUTATIONS)

    items: List[Dict[str, Any]] = []
    for (kind, column), contribution in zip(players, phi):
        if kind == "dna_block":
            feature, name = "NT_v1_DNA_Embedding", "DNA Foundation Model (NT-v1 500M)"
            description = "Joint contribution of the 1280-d DNA embedding delta (E_alt - E_ref)"
        elif kind == "prot_block":
            feature, name = "ESM2_Protein_Embedding", "Protein Foundation Model (ESM-2 650M)"
            description = "Joint contribution of the 1280-d protein embedding delta (E_alt - E_ref)"
        else:
            feature = column
            name, description = _feature_meta(column)
            raw_value = features.raw.get(column)
            if raw_value is not None:
                shown = f"{raw_value:.4g}"
            elif column in _POPULATION_AF_COLUMNS:
                # normalizer điền 0 cho tần số thiếu (giống lúc train): biến thể vắng mặt trong quần thể.
                shown = "absent in population (AF = 0)"
            else:
                shown = "n/a (imputed)"
            description += f" | raw={shown}"
        items.append({
            "feature": feature,
            "name": name,
            "contribution_value": round(float(contribution), 4),
            "description": description,
        })
    items.sort(key=lambda item: abs(item["contribution_value"]), reverse=True)

    label: PredictionLabel = "Pathogenic" if probability >= loaded.threshold else "Benign"
    return PredictionOutput(probability=round(probability, 4), label=label, shap_items=items)

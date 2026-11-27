"""Suy luận TriMVPP; Integrated Gradients và attention map chỉ tính khi mở trang XAI."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.config import settings
from app.core.constants import PredictionLabel
from app.core.errors import FeatureArtifactMissingError
from app.core.log import timed_stage
from app.ml_core import xai
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


_baseline_lock = threading.Lock()
_baselines: Dict[str, Dict[str, torch.Tensor]] = {}


def ig_baseline(device: torch.device) -> Dict[str, torch.Tensor]:
    """Baseline IG của nhóm (train3, 256 mẫu, seed 2026), nạp một lần cho mỗi thiết bị."""
    with _baseline_lock:
        if str(device) not in _baselines:
            path = settings.resolve_path(settings.IG_BASELINE_PATH)
            if not path.is_file():
                raise FeatureArtifactMissingError(
                    f"Integrated Gradients baseline '{path}' is missing. Copy ig_baseline_train3_seed2026.pt "
                    "(notebook 19, module06) to that path.",
                    status="ig_baseline_missing", expected_path=str(path),
                )
            _baselines[str(device)] = {k: v.to(device) for k, v in xai.load_baseline(path).items()}
        return _baselines[str(device)]


def _model_inputs(loaded: LoadedModel, features: VariantFeatures) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Đầu vào batch = 1 của TriMVPP: (giá trị được giải thích, mask cố định)."""
    device = loaded.device
    values: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}
    for name, modality, window in (("dna", "dna", features.dna), ("protein", "prot", features.prot)):
        if modality not in loaded.modalities:
            continue
        values[f"token_{name}"] = torch.from_numpy(np.array(window.hidden, dtype=np.float32)).unsqueeze(0).to(device)
        masks[f"mask_{name}"] = torch.from_numpy(np.array(window.mask, dtype=bool)).unsqueeze(0).to(device)
    if loaded.tabular_columns:
        tabular = torch.from_numpy(normalize_tabular(features.raw, loaded.bio_columns, loaded.geom_columns))
        tabular = tabular.to(device=device, dtype=torch.float32).unsqueeze(0)
        n_bio = len(loaded.bio_columns)
        values["bio"], values["geom"] = tabular[:, :n_bio], tabular[:, n_bio:]
    return values, masks


def _raw_display(features: VariantFeatures, column: str) -> Tuple[Optional[float], Optional[str]]:
    raw_value = features.raw.get(column)
    if raw_value is not None:
        return float(raw_value), None
    if column in _POPULATION_AF_COLUMNS:
        # normalizer điền 0 cho tần số thiếu (giống lúc train): biến thể vắng mặt trong quần thể.
        return None, "absent in population (AF = 0)"
    return None, "n/a (imputed)"


@torch.inference_mode()
def predict(loaded: LoadedModel, features: VariantFeatures) -> PredictionOutput:
    """Xác suất Pathogenic (FP32) so với ngưỡng tau của checkpoint. IG được tính riêng bằng explain()."""
    values, masks = _model_inputs(loaded, features)
    logit = float(loaded.forward_logit(**values, **masks)[0])
    probability = float(torch.sigmoid(torch.tensor(logit)))
    label: PredictionLabel = "Pathogenic" if probability >= loaded.threshold else "Benign"
    return PredictionOutput(probability=round(probability, 4), label=label)


def explain(loaded: LoadedModel, features: VariantFeatures) -> Dict[str, Any]:
    """IG full_prediction (logit trước sigmoid, baseline train3), gọi khi mở trang XAI."""
    values, masks = _model_inputs(loaded, features)
    with timed_stage("ig"):
        ig = xai.explain_ig(
            loaded.model, values, masks, ig_baseline(loaded.device),
            bio_columns=loaded.bio_columns, geom_columns=loaded.geom_columns,
            n_steps=settings.IG_STEPS, retry_steps=settings.IG_RETRY_STEPS,
            retry_threshold=settings.IG_RETRY_THRESHOLD, internal_batch_size=settings.IG_INTERNAL_BATCH,
        )
    for item in ig["features"]:
        item["name"], item["description"] = _feature_meta(item["feature"])
        item["raw_value"], item["raw_note"] = _raw_display(features, item["feature"])
    ig["features"].sort(key=lambda item: abs(item["signed"]), reverse=True)
    return ig


def attention(loaded: LoadedModel, features: VariantFeatures) -> Optional[Dict[str, Any]]:
    """Attention của 3 lớp x 8 head (tính khi cần, không lưu). None với tổ hợp một nhánh."""
    if len(loaded.model.cross_layers) == 0:
        return None
    values, masks = _model_inputs(loaded, features)
    with timed_stage("attn"):
        maps = xai.attention_maps(loaded.model, values, masks,
                                  bio_columns=loaded.bio_columns, geom_columns=loaded.geom_columns)
    for layer in maps["layers"]:
        for entry in layer["maps"]:
            entry["weights"] = np.round(entry["weights"].astype(np.float64), 5).tolist()
    return maps

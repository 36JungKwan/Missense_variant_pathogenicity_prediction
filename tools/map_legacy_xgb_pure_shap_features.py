"""Map legacy xgb_pure_XXXX SHAP feature names to interpretable names.

This script is for old SHAP outputs generated before Pure XGBoost feature
names were made explicit in the pipeline. It reads legacy files such as:

  shap_global_importance.csv
  shap_local_top_features.csv

and writes mapped files next to them:

  shap_global_importance_mapped.csv
  shap_local_top_features_mapped_long.csv

Usage:
  python tools/map_legacy_xgb_pure_shap_features.py --root experiments
  python tools/map_legacy_xgb_pure_shap_features.py --path path/to/explainability
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BIO_COLS = [
    "AF",
    "gnomADe_AF",
    "phyloP100way_vertebrate",
    "phyloP470way_mammalian",
    "phyloP17way_primate",
    "phastCons100way_vertebrate",
    "phastCons470way_mammalian",
    "phastCons17way_primate",
    "GERP++_RS",
    "GERP++_NR",
    "GERP_92_mammals",
]

GEOM_BASE_COLS = ["LLR", "LVD_L2", "LVD_Cosine", "LID"]
POOLING_TOKENS = {"cls", "center", "mean"}
MODALITY_TOKENS = {"bio", "dna", "geom", "prot"}
LEGACY_FEATURE_RE = re.compile(r"^xgb_pure_(\d+)$")
LEGACY_LOCAL_COL_RE = re.compile(r"^shap_(xgb_pure_\d+)$")


@dataclass(frozen=True)
class LegacyXgbPureConfig:
    exp_name: str
    pooling: str
    ablation: str
    active_modalities: tuple[str, ...]
    dna_model: str
    prot_model: str


def _parse_exp_name(exp_name: str) -> LegacyXgbPureConfig:
    suffix = "_Pure_XGBoost_Concat"
    if not exp_name.endswith(suffix):
        raise ValueError(f"Khong phai Pure_XGBoost_Concat exp dir: {exp_name}")

    stem = exp_name[: -len(suffix)]
    tokens = stem.split("_")
    pooling_idx = next((i for i, token in enumerate(tokens) if token in POOLING_TOKENS), None)
    if pooling_idx is None:
        raise ValueError(f"Khong parse duoc pooling tu exp dir: {exp_name}")

    pooling = tokens[pooling_idx]
    pos = pooling_idx + 1
    modalities = []
    while pos < len(tokens) and tokens[pos] in MODALITY_TOKENS:
        modalities.append(tokens[pos])
        pos += 1
    if not modalities:
        raise ValueError(f"Khong parse duoc ablation tu exp dir: {exp_name}")

    dna_model, prot_model = _parse_model_names(tokens[pos:])
    return LegacyXgbPureConfig(
        exp_name=exp_name,
        pooling=pooling,
        ablation="_".join(modalities),
        active_modalities=tuple(modalities),
        dna_model=dna_model,
        prot_model=prot_model,
    )


def _parse_model_names(tokens: list[str]) -> tuple[str, str]:
    if not tokens:
        return "None", "None"

    if tokens[0] == "None":
        prot = "_".join(tokens[1:]) if len(tokens) > 1 else "None"
        return "None", prot or "None"

    if tokens[-1] == "None":
        dna = "_".join(tokens[:-1]) if len(tokens) > 1 else "None"
        return dna or "None", "None"

    prot_start = next((i for i, token in enumerate(tokens) if token.startswith("esm")), None)
    if prot_start is None:
        return "_".join(tokens), "None"

    dna = "_".join(tokens[:prot_start]) if prot_start > 0 else "None"
    prot = "_".join(tokens[prot_start:])
    return dna or "None", prot or "None"


def _exp_dir_from_shap_path(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "tests":
            return parent.parent
    if path.name == "explainability":
        return path.parents[1]
    if path.parent.name == "explainability":
        return path.parents[3]
    raise ValueError(f"Khong tim thay exp dir tu path: {path}")


def _tabular_feature_names(config: LegacyXgbPureConfig) -> list[str]:
    active = set(config.active_modalities)
    names = []

    if "bio" in active:
        names.extend(BIO_COLS)

    if "geom" in active:
        if config.dna_model != "None":
            names.extend([f"dna_{col}" for col in GEOM_BASE_COLS])
        if config.prot_model != "None":
            names.extend([f"prot_{col}" for col in GEOM_BASE_COLS])

    return names


def build_legacy_xgb_pure_feature_map(
    n_features: int,
    config: LegacyXgbPureConfig,
    pca_components: int = 256,
    dna_pca_dim: int | None = None,
    prot_pca_dim: int | None = None,
) -> pd.DataFrame:
    active = set(config.active_modalities)
    tabular_names = _tabular_feature_names(config)
    seq_total = n_features - len(tabular_names)
    if seq_total < 0:
        raise ValueError(
            f"n_features={n_features} nho hon so cot tabular suy ra={len(tabular_names)}"
        )

    has_dna = "dna" in active
    has_prot = "prot" in active

    if has_dna and has_prot:
        if dna_pca_dim is None and prot_pca_dim is None:
            dna_pca_dim = min(pca_components, seq_total)
            prot_pca_dim = seq_total - dna_pca_dim
        elif dna_pca_dim is None:
            dna_pca_dim = seq_total - int(prot_pca_dim)
        elif prot_pca_dim is None:
            prot_pca_dim = seq_total - int(dna_pca_dim)
    elif has_dna:
        dna_pca_dim = seq_total
        prot_pca_dim = 0
    elif has_prot:
        dna_pca_dim = 0
        prot_pca_dim = seq_total
    else:
        dna_pca_dim = 0
        prot_pca_dim = 0

    dna_pca_dim = int(dna_pca_dim or 0)
    prot_pca_dim = int(prot_pca_dim or 0)
    if dna_pca_dim < 0 or prot_pca_dim < 0 or dna_pca_dim + prot_pca_dim != seq_total:
        raise ValueError(
            "Khong infer duoc PCA dims hop le: "
            f"dna={dna_pca_dim}, prot={prot_pca_dim}, seq_total={seq_total}"
        )

    mapped_names = []
    groups = []
    notes = []

    for i in range(dna_pca_dim):
        mapped_names.append(f"dna_pca_{i:03d}")
        groups.append("dna_pca")
        notes.append("")

    for i in range(prot_pca_dim):
        mapped_names.append(f"prot_pca_{i:03d}")
        groups.append("prot_pca")
        notes.append("")

    for name in tabular_names:
        mapped_names.append(name)
        groups.append("bio" if name in BIO_COLS else "geom")
        notes.append("")

    return pd.DataFrame(
        {
            "legacy_feature": [f"xgb_pure_{i:04d}" for i in range(n_features)],
            "feature_index": list(range(n_features)),
            "mapped_feature": mapped_names,
            "feature_group": groups,
            "mapping_note": notes,
        }
    )


def _feature_index(feature_name: str) -> int | None:
    match = LEGACY_FEATURE_RE.match(str(feature_name))
    return int(match.group(1)) if match else None


def map_global_file(
    path: Path,
    out_path: Path | None = None,
    pca_components: int = 256,
    dna_pca_dim: int | None = None,
    prot_pca_dim: int | None = None,
    overwrite: bool = False,
) -> Path | None:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        print(f"[skip] Thieu cot feature: {path}")
        return None

    exp_dir = _exp_dir_from_shap_path(path)
    config = _parse_exp_name(exp_dir.name)
    indices = df["feature"].map(_feature_index)
    if indices.isna().all():
        print(f"[skip] Khong co feature legacy xgb_pure_XXXX: {path}")
        return None

    n_features = int(indices.max()) + 1
    fmap = build_legacy_xgb_pure_feature_map(
        n_features=n_features,
        config=config,
        pca_components=pca_components,
        dna_pca_dim=dna_pca_dim,
        prot_pca_dim=prot_pca_dim,
    )
    out = df.merge(fmap, left_on="feature", right_on="legacy_feature", how="left")
    out.insert(0, "exp_name", config.exp_name)
    out.insert(1, "ablation", config.ablation)
    out.insert(2, "pooling", config.pooling)
    out.insert(3, "dna_model", config.dna_model)
    out.insert(4, "prot_model", config.prot_model)

    if out_path is None:
        out_path = path.with_name("shap_global_importance_mapped.csv")
    if out_path.exists() and not overwrite:
        print(f"[skip] Da ton tai: {out_path}")
        return out_path

    out.to_csv(out_path, index=False)
    print(f"[ok] {path} -> {out_path}")
    return out_path


def map_local_wide_file(
    path: Path,
    out_path: Path | None = None,
    pca_components: int = 256,
    dna_pca_dim: int | None = None,
    prot_pca_dim: int | None = None,
    overwrite: bool = False,
) -> Path | None:
    df = pd.read_csv(path)
    shap_cols = [col for col in df.columns if LEGACY_LOCAL_COL_RE.match(col)]
    if not shap_cols:
        print(f"[skip] Khong co cot shap_xgb_pure_XXXX: {path}")
        return None

    exp_dir = _exp_dir_from_shap_path(path)
    config = _parse_exp_name(exp_dir.name)
    max_idx = max(_feature_index(LEGACY_LOCAL_COL_RE.match(col).group(1)) for col in shap_cols)
    fmap = build_legacy_xgb_pure_feature_map(
        n_features=max_idx + 1,
        config=config,
        pca_components=pca_components,
        dna_pca_dim=dna_pca_dim,
        prot_pca_dim=prot_pca_dim,
    ).set_index("legacy_feature")

    id_cols = [col for col in df.columns if col not in shap_cols]
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=shap_cols,
        var_name="legacy_shap_column",
        value_name="shap_value",
    )
    long_df["legacy_feature"] = long_df["legacy_shap_column"].str.replace(
        r"^shap_",
        "",
        regex=True,
    )
    long_df = long_df.join(fmap, on="legacy_feature")
    long_df["abs_shap"] = long_df["shap_value"].abs()
    if "Variant_ID" in long_df.columns:
        long_df["rank_within_variant"] = (
            long_df.groupby("Variant_ID")["abs_shap"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

    long_df.insert(0, "exp_name", config.exp_name)
    long_df.insert(1, "ablation", config.ablation)
    long_df.insert(2, "pooling", config.pooling)
    long_df.insert(3, "dna_model", config.dna_model)
    long_df.insert(4, "prot_model", config.prot_model)

    if out_path is None:
        out_path = path.with_name("shap_local_top_features_mapped_long.csv")
    if out_path.exists() and not overwrite:
        print(f"[skip] Da ton tai: {out_path}")
        return out_path

    long_df.to_csv(out_path, index=False)
    print(f"[ok] {path} -> {out_path}")
    return out_path


def _iter_target_files(path: Path) -> tuple[list[Path], list[Path]]:
    if path.is_file():
        if path.name == "shap_global_importance.csv":
            return [path], []
        if path.name == "shap_local_top_features.csv":
            return [], [path]
        raise ValueError(f"File khong duoc ho tro: {path}")

    global_files = sorted(
        p for p in path.glob("**/shap_global_importance.csv")
        if "Pure_XGBoost_Concat" in str(p)
    )
    local_files = sorted(
        p for p in path.glob("**/shap_local_top_features.csv")
        if "Pure_XGBoost_Concat" in str(p)
    )
    return global_files, local_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map legacy xgb_pure_XXXX SHAP feature names to readable names.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Root folder to scan, e.g. experiments")
    parser.add_argument("--path", type=Path, default=None, help="Specific file or folder to map")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--dna-pca-dim", type=int, default=None)
    parser.add_argument("--prot-pca-dim", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target = args.path or args.root or Path("experiments")
    global_files, local_files = _iter_target_files(target)

    print(f"[info] global files: {len(global_files)}")
    print(f"[info] local wide files: {len(local_files)}")

    for path in global_files:
        map_global_file(
            path,
            pca_components=args.pca_components,
            dna_pca_dim=args.dna_pca_dim,
            prot_pca_dim=args.prot_pca_dim,
            overwrite=args.overwrite,
        )

    for path in local_files:
        map_local_wide_file(
            path,
            pca_components=args.pca_components,
            dna_pca_dim=args.dna_pca_dim,
            prot_pca_dim=args.prot_pca_dim,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

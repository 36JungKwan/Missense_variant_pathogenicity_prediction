"""Publication-oriented plots and cross-method diagnostics for notebook 19."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_GUIDE = pd.DataFrame([
    {
        "Method": "Integrated Gradients (IG)",
        "What_it_shows": "Signed input attribution relative to a declared baseline; token and feature level.",
        "Positive_or_high": "Positive signed attribution raises the pathogenic logit along the IG path; negative lowers it.",
        "Caveat": "Depends on the baseline and integration path; check the completeness residual.",
    },
    {
        "Method": "Occlusion",
        "What_it_shows": "Change in pathogenic logit when one input feature/token or group is replaced by its reference value.",
        "Positive_or_high": "Actual-minus-occluded > 0 means the observed input raised the logit versus that replacement.",
        "Caveat": "A perturbation can create an unusual input; effects may be nonlinear and overlapping.",
    },
    {
        "Method": "Attention map",
        "What_it_shows": "How much a query token weights context keys inside a cross-attention layer.",
        "Positive_or_high": "A larger weight means more attention is assigned to that context key.",
        "Caveat": "Attention is not a signed contribution and does not by itself explain the final prediction.",
    },
    {
        "Method": "Exact SHAP (4 groups)",
        "What_it_shows": "Exact Shapley allocation among DNA, Protein, Bio/SpliceAI and Geometry groups.",
        "Positive_or_high": "Positive raises and negative lowers pathogenic logit relative to its coalition baseline.",
        "Caveat": "Exact only across four grouped players; baseline is the specified mean-token/median-feature input.",
    },
    {
        "Method": "Gradient / Deep SHAP",
        "What_it_shows": "Approximate signed attribution for token positions and individual tabular features.",
        "Positive_or_high": "Positive raises and negative lowers pathogenic logit relative to the train background.",
        "Caveat": "Approximate, background-dependent; inspect additivity residual and seed variability.",
    },
])


def _subset(frame, dataset, seed=None, label=None):
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "Dataset" in out:
        out = out[out["Dataset"].astype(str).eq(str(dataset))]
    if seed is not None and "Seed" in out:
        out = out[pd.to_numeric(out["Seed"], errors="coerce").eq(int(seed))]
    if label is not None and "True_Label" in out:
        out = out[pd.to_numeric(out["True_Label"], errors="coerce").eq(int(label))]
    return out


def _weighted_mean(values, weights):
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[good], weights=weights[good])) if good.any() else np.nan


def _spearman(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return np.nan
    xr, yr = x[valid].rank(method="average"), y[valid].rank(method="average")
    return float(xr.corr(yr))


def _restrict_to_common_samples(frames):
    """Use only sample identities present in every supplied method frame."""
    identity = ["Dataset", "Seed", "Variant_ID", "True_Label"]
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable or any(not set(identity).issubset(frame.columns) for frame in usable):
        return frames
    common = None
    for frame in usable:
        keys = set(map(tuple, frame[identity].astype(str).drop_duplicates().to_numpy()))
        common = keys if common is None else common.intersection(keys)
    result = []
    for frame in frames:
        if frame is None or frame.empty:
            result.append(frame)
            continue
        keys = frame[identity].astype(str).agg("|".join, axis=1)
        common_joined = {"|".join(key) for key in common or set()}
        result.append(frame.loc[keys.isin(common_joined)].copy())
    return result


def _make_signed_heatmap(frame, *, method, dataset, seed, label, mode, output_dir):
    if frame.empty:
        return
    work = _subset(frame, dataset, seed, label)
    if method.startswith("IG"):
        if "Attribution_Mode" in work:
            work = work[work["Attribution_Mode"].eq(mode)]
        value_col = "Signed_Attribution"
    else:
        value_col = "Signed_Attribution"
    if work.empty or value_col not in work:
        return
    modalities = ["DNA", "Protein"]
    alleles = ["Ref", "Alt"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    image = None
    panel_count = 0
    for ax, (modality, allele) in zip(axes.flat, [(m, a) for m in modalities for a in alleles]):
        part = work[work["Modality"].eq(modality) & work["Allele"].eq(allele)]
        if part.empty:
            ax.set_visible(False)
            continue
        part = part.copy()
        part["Sample_Label"] = part["Variant_ID"].astype(str) + part["True_Label"].map(
            {0: " [benign]", 1: " [pathogenic]"}
        ).fillna("")
        pivot = part.pivot_table(
            index="Sample_Label", columns="Relative_Token_Position", values=value_col,
            aggfunc="mean",
        ).sort_index(axis=1)
        if pivot.empty:
            ax.set_visible(False)
            continue
        # Keep the figure legible if a future run explains a substantially larger sample.
        pivot = pivot.iloc[:60]
        matrix = pivot.to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.0
        max_abs = max(max_abs, 1e-8)
        image = ax.imshow(
            matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r",
            vmin=-max_abs, vmax=max_abs,
        )
        ax.set_title(f"{modality} {allele}")
        ax.set_xlabel("Relative position (variant at 0)")
        ax.set_ylabel("Variant sample")
        xticks = np.linspace(0, len(pivot.columns) - 1, min(9, len(pivot.columns)), dtype=int)
        ax.set_xticks(xticks, [str(pivot.columns[i]) for i in xticks])
        yticks = np.linspace(0, len(pivot.index) - 1, min(10, len(pivot.index)), dtype=int)
        ax.set_yticks(yticks, [str(pivot.index[i])[:22] for i in yticks], fontsize=7)
        panel_count += 1
    if image is None or not panel_count:
        plt.close(fig)
        return
    fig.colorbar(image, ax=list(axes.flat), shrink=0.82, label="Signed attribution to pathogenic logit")
    label_text = "both labels" if label is None else ("benign" if int(label) == 0 else "pathogenic")
    fig.suptitle(
        f"{method} signed token attribution | {dataset} | seed {seed} | {label_text}\n"
        + (f"IG mode: {mode}. Red/blue direction is interpreted by the colorbar." if method.startswith("IG")
           else "Each row is one sample; positive raises and negative lowers the pathogenic logit."),
        fontsize=12,
    )
    mode_suffix = f"_{mode}" if method.startswith("IG") else ""
    filename = f"{method.lower()}_signed_token_heatmap{mode_suffix}_{dataset}_seed{seed}_{label_text.replace(' ', '_')}.png"
    fig.savefig(output_dir / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _draw_beeswarm(frame, *, explainer, dataset, seed, label, top_features, output_dir):
    if label is None and not frame.empty and "True_Label" in frame:
        for true_label in (0, 1):
            _draw_beeswarm(frame, explainer=explainer, dataset=dataset, seed=seed,
                           label=true_label, top_features=top_features, output_dir=output_dir)
        return
    if frame.empty or "Input_Value" not in frame:
        print(f"{explainer} SHAP beeswarm skipped: no feature values yet; rerun the SHAP cell after its output-shape fix.")
        return
    part = _subset(frame, dataset, seed, label)
    if part.empty:
        print(f"{explainer} SHAP beeswarm skipped: no completed rows for this selector.")
        return
    key_columns = [c for c in ("Dataset", "Seed", "Variant_ID", "True_Label", "Feature") if c in part]
    part = part.drop_duplicates(key_columns)
    value_wide = part.pivot_table(index=key_columns[:-1], columns="Feature", values="Signed_Attribution", aggfunc="mean")
    input_wide = part.pivot_table(index=key_columns[:-1], columns="Feature", values="Input_Value", aggfunc="mean")
    if value_wide.empty:
        return
    importance = value_wide.abs().mean().sort_values(ascending=False)
    # The requested global view is the complete 28-feature set; retain a larger
    # user-selected number if the upstream feature table has grown.
    names = importance.head(max(28, int(top_features))).index.tolist()
    values = value_wide[names].to_numpy(float)
    data = input_wide.reindex(columns=names).to_numpy(float)
    try:
        import shap
        explanation = shap.Explanation(values=values, data=data, feature_names=names)
        shap.plots.beeswarm(explanation, max_display=len(names), show=False)
        fig = plt.gcf()
        label_text = "both" if label is None else ("benign" if int(label) == 0 else "pathogenic")
        fig.suptitle(
            f"{explainer} feature beeswarm | {dataset} | seed {seed} | {label_text}\n"
            "X: signed effect on pathogenic logit; color: normalized input value; samples include seed-specific model runs.",
            y=1.02,
        )
        fig.savefig(output_dir / f"{explainer.lower()}_feature_beeswarm_{dataset}_seed{seed}_{label_text}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Could not render {explainer} beeswarm: {type(exc).__name__}: {exc}")


def _waterfall(shap_module, *, values, names, base, output, title, path, max_display=18):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values, names = values[finite], [name for name, keep in zip(names, finite) if keep]
    if not len(values) or not np.isfinite(base) or not np.isfinite(output):
        return
    explanation = shap_module.Explanation(
        values=values, base_values=float(base), feature_names=[str(n) for n in names]
    )
    shap_module.plots.waterfall(explanation, max_display=min(int(max_display), len(values)), show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _representative_local_case(group):
    """Pick the sample nearest the median predicted probability within its true class."""
    candidates = group.drop_duplicates("Variant_ID").copy()
    if "Model_Logit" not in candidates:
        return candidates.sort_values("Variant_ID").iloc[0]
    logits = pd.to_numeric(candidates["Model_Logit"], errors="coerce")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    if not probabilities.notna().any():
        return candidates.sort_values("Variant_ID").iloc[0]
    median_probability = float(probabilities.median())
    candidates["_distance_to_median_probability"] = (probabilities - median_probability).abs()
    return candidates.sort_values(["_distance_to_median_probability", "Variant_ID"]).iloc[0]


def _render_waterfalls(*, exact, feature_frames, position_frames, completeness_frames,
                       dataset, seed, label, output_dir):
    try:
        import shap
    except Exception as exc:
        print(f"SHAP waterfall plots unavailable: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    cases = []
    # Exact grouped modality waterfall.
    if not exact.empty:
        work = _subset(exact, dataset, seed, label)
        for true_label, group in work.groupby("True_Label", sort=True):
            if group.empty:
                continue
            sample = _representative_local_case(group)
            rows = group[group["Variant_ID"].eq(sample["Variant_ID"])].drop_duplicates("Modality")
            exact_logit = float(rows["Model_Logit"].iloc[0])
            exact_probability = float(1.0 / (1.0 + np.exp(-np.clip(exact_logit, -80, 80))))
            exact_threshold = float(rows["Decision_Threshold"].iloc[0]) if "Decision_Threshold" in rows else np.nan
            exact_predicted = int(exact_probability >= exact_threshold) if np.isfinite(exact_threshold) else -1
            _waterfall(
                shap, values=rows["SHAP_Value_Logit"].to_numpy(float),
                names=rows["Modality"].astype(str).tolist(),
                base=float(rows["Base_Logit"].iloc[0]), output=float(rows["Model_Logit"].iloc[0]),
                title=f"Exact SHAP (4 modality groups) | {dataset} | seed {seed} | {sample['Variant_ID']} | true={int(true_label)} | predicted={exact_predicted}, p={exact_probability:.3f}, logit={exact_logit:.3f}",
                path=output_dir / f"waterfall_exact_{dataset}_seed{seed}_label{int(true_label)}.png",
            )
            cases.append({"Explainer": "Exact", "Dataset": dataset, "Seed": seed,
                          "Variant_ID": sample["Variant_ID"], "True_Label": int(true_label),
                          "Case_Selection": "nearest median probability within true class",
                          "Base_Logit": float(rows["Base_Logit"].iloc[0]),
                          "Model_Logit": float(rows["Model_Logit"].iloc[0]),
                          "Pathogenic_Probability": exact_probability,
                          "Decision_Threshold": exact_threshold,
                          "Predicted_Label": exact_predicted if exact_predicted >= 0 else np.nan})
    for name, feature_frame, position_frame, complete_frame in zip(
        ("Gradient", "Deep"), feature_frames, position_frames, completeness_frames
    ):
        if feature_frame.empty or position_frame.empty or complete_frame.empty:
            continue
        features = _subset(feature_frame, dataset, seed, label)
        positions = _subset(position_frame, dataset, seed, label)
        complete = _subset(complete_frame, dataset, seed, label)
        if features.empty or positions.empty or complete.empty:
            continue
        for true_label, group in complete.groupby("True_Label", sort=True):
            sample = _representative_local_case(group)
            keys = ["Dataset", "Seed", "Variant_ID"]
            feature_rows = features
            position_rows = positions
            for key in keys:
                if key in sample:
                    feature_rows = feature_rows[feature_rows[key].astype(str).eq(str(sample[key]))]
                    position_rows = position_rows[position_rows[key].astype(str).eq(str(sample[key]))]
            feature_rows = feature_rows.drop_duplicates("Feature")
            position_rows = position_rows.drop_duplicates(["Modality", "Allele", "Relative_Token_Position"])
            names = feature_rows["Feature"].astype(str).tolist()
            vals = pd.to_numeric(feature_rows["Signed_Attribution"], errors="coerce").tolist()
            if "Input_Value" in feature_rows:
                names = [f"{n} = {v:.3g}" for n, v in zip(names, feature_rows["Input_Value"])]
            for _, row in position_rows.iterrows():
                names.append(f"{row['Modality']} {row['Allele']} {int(row['Relative_Token_Position']):+d}")
                vals.append(float(row["Signed_Attribution"]))
            completeness_row = group[group["Variant_ID"].astype(str).eq(str(sample["Variant_ID"]))].iloc[0]
            local_logit = float(completeness_row["Model_Logit"])
            local_probability = float(1.0 / (1.0 + np.exp(-np.clip(local_logit, -80, 80))))
            local_threshold = float(completeness_row["Decision_Threshold"]) if "Decision_Threshold" in completeness_row else np.nan
            local_predicted = int(local_probability >= local_threshold) if np.isfinite(local_threshold) else -1
            _waterfall(
                shap, values=vals, names=names,
                base=float(completeness_row["Background_Mean_Logit"]),
                output=float(completeness_row["Model_Logit"]),
                title=f"{name} SHAP | {dataset} | seed {seed} | {sample['Variant_ID']} | true={int(true_label)} | predicted={local_predicted}, p={local_probability:.3f}, logit={local_logit:.3f}",
                path=output_dir / f"waterfall_{name.lower()}_{dataset}_seed{seed}_label{int(true_label)}.png",
            )
            cases.append({"Explainer": name, "Dataset": dataset, "Seed": seed,
                          "Variant_ID": sample["Variant_ID"], "True_Label": int(true_label),
                          "Case_Selection": "nearest median probability within true class",
                          "Base_Logit": float(completeness_row["Background_Mean_Logit"]),
                          "Model_Logit": float(completeness_row["Model_Logit"]),
                          "Pathogenic_Probability": local_probability,
                          "Decision_Threshold": local_threshold,
                          "Predicted_Label": local_predicted if local_predicted >= 0 else np.nan,
                          "Relative_Additivity_Residual": completeness_row.get("Relative_Additivity_Residual")})
    return pd.DataFrame(cases)


def _feature_method_comparison(ig, occ, gradient, deep, *, dataset, mode):
    keys = ["Dataset", "Seed", "Variant_ID", "True_Label", "Feature"]
    frames = []
    if not ig.empty:
        part = ig[ig.get("Attribution_Mode", pd.Series(index=ig.index, dtype=object)).eq(mode)].copy()
        frames.append(part[keys + ["Signed_Attribution", "Absolute_Attribution", "Sampling_Weight"]]
                      .rename(columns={"Signed_Attribution": "IG_Signed", "Absolute_Attribution": "IG_Absolute"}))
    for explainer, frame in (("Gradient", gradient), ("Deep", deep)):
        if frame.empty:
            continue
        frames.append(frame[keys + ["Signed_Attribution", "Absolute_Attribution"]]
                      .rename(columns={"Signed_Attribution": f"{explainer}_Signed", "Absolute_Attribution": f"{explainer}_Absolute"}))
    occ_part = pd.DataFrame()
    if not occ.empty:
        occ_part = occ[~occ["Feature_or_Position"].astype(str).str.startswith(("DNA_", "PROTEIN_"))].copy()
        occ_part = occ_part.rename(columns={"Feature_or_Position": "Feature", "Logit_Delta_Actual_Minus_Occluded": "Occlusion_Signed"})
        occ_part = occ_part[keys + ["Occlusion_Signed"]].drop_duplicates(keys)
        frames.append(occ_part)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    # Compare methods only on shared cases (the SHAP sample set is intentionally
    # smaller than the full IG/occlusion cohort).
    frames = _restrict_to_common_samples(frames)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="outer", validate="one_to_one")
    merged = merged[merged["Dataset"].astype(str).eq(str(dataset))]
    pairs = [("IG_Signed", "Gradient_Signed", "IG vs Gradient SHAP"),
             ("IG_Signed", "Deep_Signed", "IG vs Deep SHAP"),
             ("IG_Signed", "Occlusion_Signed", "IG vs feature occlusion"),
             ("Gradient_Signed", "Deep_Signed", "Gradient vs Deep SHAP")]
    rows = []
    for left, right, name in pairs:
        if left not in merged or right not in merged:
            continue
        valid = merged[left].notna() & merged[right].notna()
        if not valid.any():
            continue
        sample_correlations = []
        sample_keys = ["Dataset", "Seed", "Variant_ID", "True_Label"]
        for _, sample_group in merged.loc[valid].groupby(sample_keys, dropna=False):
            if len(sample_group) >= 3:
                sample_correlations.append(_spearman(sample_group[left], sample_group[right]))
        sample_correlations = [value for value in sample_correlations if np.isfinite(value)]
        rows.append({"Comparison": name, "N_matched_sample_feature_rows": int(valid.sum()),
                     "N_matched_samples": int(merged.loc[valid, sample_keys].drop_duplicates().shape[0]),
                     "Spearman_signed": _spearman(merged.loc[valid, left], merged.loc[valid, right]),
                     "Spearman_absolute": _spearman(merged.loc[valid, left].abs(), merged.loc[valid, right].abs()),
                     "Median_within_sample_Spearman_signed": float(np.median(sample_correlations)) if sample_correlations else np.nan,
                     "Sign_agreement": float((np.sign(merged.loc[valid, left]) == np.sign(merged.loc[valid, right])).mean())})
    return merged, pd.DataFrame(rows)


def _token_method_comparison(ig, occ, gradient, deep, *, dataset, mode):
    keys = ["Dataset", "Seed", "Variant_ID", "True_Label", "Modality", "Relative_Token_Position"]
    frames = []
    if not ig.empty:
        part = ig[(ig["Attribution_Mode"].eq(mode)) & ig["Allele"].eq("Alt")].copy()
        frames.append(part[keys + ["Signed_Attribution"]].rename(columns={"Signed_Attribution": "IG_Signed"}))
    for explainer, frame in (("Gradient", gradient), ("Deep", deep)):
        if not frame.empty:
            part = frame[frame["Allele"].eq("Alt")]
            frames.append(part[keys + ["Signed_Attribution"]].rename(columns={"Signed_Attribution": f"{explainer}_Signed"}))
    if not occ.empty:
        part = occ[occ["Feature_or_Position"].astype(str).str.startswith(("DNA_", "PROTEIN_"))].copy()
        parsed = part["Feature_or_Position"].astype(str).str.extract(r"^(DNA|PROTEIN)_alt_to_ref_([+-]?\d+)$")
        part["Modality"] = parsed[0].map({"DNA": "DNA", "PROTEIN": "Protein"})
        part["Relative_Token_Position"] = pd.to_numeric(parsed[1], errors="coerce")
        part = part.dropna(subset=["Modality", "Relative_Token_Position"])
        part["Relative_Token_Position"] = part["Relative_Token_Position"].astype(int)
        frames.append(part.rename(columns={"Logit_Delta_Actual_Minus_Occluded": "Occlusion_Signed"})
                      [keys + ["Occlusion_Signed"]].drop_duplicates(keys))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    # Restrict the comparison to the shared SHAP/IG/occlusion sample subset. This
    # keeps the diagnostic interpretable and avoids materializing all test tokens.
    frames = _restrict_to_common_samples(frames)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="outer", validate="one_to_one")
    merged = merged[merged["Dataset"].astype(str).eq(str(dataset))]
    pairs = [("IG_Signed", "Gradient_Signed", "IG vs Gradient SHAP"),
             ("IG_Signed", "Deep_Signed", "IG vs Deep SHAP"),
             ("IG_Signed", "Occlusion_Signed", "IG vs token occlusion"),
             ("Gradient_Signed", "Deep_Signed", "Gradient vs Deep SHAP")]
    rows = []
    for left, right, name in pairs:
        if left not in merged or right not in merged:
            continue
        valid = merged[left].notna() & merged[right].notna()
        if valid.any():
            sample_correlations = []
            sample_keys = ["Dataset", "Seed", "Variant_ID", "True_Label"]
            for _, sample_group in merged.loc[valid].groupby(sample_keys, dropna=False):
                if len(sample_group) >= 3:
                    sample_correlations.append(_spearman(sample_group[left], sample_group[right]))
            sample_correlations = [value for value in sample_correlations if np.isfinite(value)]
            rows.append({"Comparison": name, "N_matched_token_rows": int(valid.sum()),
                         "N_matched_samples": int(merged.loc[valid, sample_keys].drop_duplicates().shape[0]),
                         "Spearman_signed": _spearman(merged.loc[valid, left], merged.loc[valid, right]),
                         "Median_within_sample_Spearman_signed": float(np.median(sample_correlations)) if sample_correlations else np.nan,
                         "Sign_agreement": float((np.sign(merged.loc[valid, left]) == np.sign(merged.loc[valid, right])).mean())})
    return merged, pd.DataFrame(rows)


def _modality_method_comparison(ig, exact, gradient, deep, *, dataset, mode):
    keys = ["Dataset", "Seed", "Variant_ID", "True_Label", "Modality"]
    frames = []
    if not ig.empty:
        part = ig[ig["Attribution_Mode"].eq(mode)]
        frames.append(part[keys + ["Signed_Attribution_Sum", "Absolute_Attribution_Sum"]]
                      .rename(columns={"Signed_Attribution_Sum": "IG_Signed", "Absolute_Attribution_Sum": "IG_Absolute"}))
    if not exact.empty:
        frames.append(exact[keys + ["SHAP_Value_Logit"]].assign(
            Absolute=lambda df: df["SHAP_Value_Logit"].abs()
        ).rename(columns={"SHAP_Value_Logit": "Exact_Signed", "Absolute": "Exact_Absolute"}))
    for method, frame in (("Gradient", gradient), ("Deep", deep)):
        if not frame.empty:
            frames.append(frame[keys + ["Signed_Attribution_Sum", "Absolute_Attribution_Sum"]]
                          .rename(columns={"Signed_Attribution_Sum": f"{method}_Signed",
                                           "Absolute_Attribution_Sum": f"{method}_Absolute"}))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    frames = _restrict_to_common_samples(frames)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="outer", validate="one_to_one")
    merged = merged[merged["Dataset"].astype(str).eq(str(dataset))]
    pairs = [("IG_Signed", "Exact_Signed", "IG vs Exact SHAP"),
             ("IG_Signed", "Gradient_Signed", "IG vs Gradient SHAP"),
             ("IG_Signed", "Deep_Signed", "IG vs Deep SHAP"),
             ("Gradient_Signed", "Deep_Signed", "Gradient vs Deep SHAP")]
    rows = []
    for left, right, name in pairs:
        if left not in merged or right not in merged:
            continue
        valid = merged[left].notna() & merged[right].notna()
        if not valid.any():
            continue
        rows.append({"Comparison": name, "N_matched_sample_modality_rows": int(valid.sum()),
                     "Spearman_signed": _spearman(merged.loc[valid, left], merged.loc[valid, right]),
                     "Spearman_absolute": _spearman(merged.loc[valid, left].abs(), merged.loc[valid, right].abs()),
                     "Sign_agreement": float((np.sign(merged.loc[valid, left]) == np.sign(merged.loc[valid, right])).mean())})
    return merged, pd.DataFrame(rows)


def _plot_modality_scatter(frame, summary, *, dataset, output_path):
    if frame.empty or summary.empty:
        return
    pairs = [("IG_Signed", "Exact_Signed", "IG vs Exact SHAP"),
             ("IG_Signed", "Gradient_Signed", "IG vs Gradient SHAP"),
             ("IG_Signed", "Deep_Signed", "IG vs Deep SHAP")]
    pairs = [(x, y, title) for x, y, title in pairs if x in frame and y in frame and summary["Comparison"].eq(title).any()]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5), squeeze=False)
    for ax, (x, y, title) in zip(axes.flat, pairs):
        valid = frame[x].notna() & frame[y].notna()
        ax.scatter(frame.loc[valid, x], frame.loc[valid, y], s=16, alpha=0.55)
        if valid.any():
            low = min(frame.loc[valid, x].min(), frame.loc[valid, y].min())
            high = max(frame.loc[valid, x].max(), frame.loc[valid, y].max())
            ax.plot([low, high], [low, high], "k--", linewidth=1)
        rho = summary.loc[summary["Comparison"].eq(title), "Spearman_signed"].iloc[0]
        ax.set_title(f"{title}\nSpearman={rho:.3f}")
        ax.set_xlabel("IG signed attribution sum")
        ax.set_ylabel("SHAP signed attribution sum")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Modality-level signed effects | {dataset}\nDirection/rank diagnostic; each method has a different baseline.")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_method_scatter(frame, summary, *, value_columns, title_prefix, output_path):
    if frame.empty or summary.empty:
        return
    pairs = [("IG_Signed", "Gradient_Signed", "IG vs Gradient SHAP"),
             ("IG_Signed", "Deep_Signed", "IG vs Deep SHAP")]
    occlusion_name = summary.loc[
        summary["Comparison"].astype(str).str.contains("IG vs .*occlusion", case=False), "Comparison"
    ]
    if "Occlusion_Signed" in frame and not occlusion_name.empty:
        pairs.append(("IG_Signed", "Occlusion_Signed", str(occlusion_name.iloc[0])))
    pairs = [(x, y, name) for x, y, name in pairs if x in frame and y in frame and summary["Comparison"].eq(name).any()]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
    for ax, (xcol, ycol, name) in zip(axes.flat, pairs):
        valid = frame[xcol].notna() & frame[ycol].notna()
        if not valid.any():
            ax.set_visible(False)
            continue
        ax.scatter(frame.loc[valid, xcol], frame.loc[valid, ycol], s=12, alpha=0.45)
        low = min(frame.loc[valid, xcol].min(), frame.loc[valid, ycol].min())
        high = max(frame.loc[valid, xcol].max(), frame.loc[valid, ycol].max())
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        rho = summary.loc[summary["Comparison"].eq(name), "Spearman_signed"].iloc[0]
        ax.set_title(f"{name}\nSpearman={rho:.3f}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(alpha=0.2)
    fig.suptitle(title_prefix + " — same-sample signed effects; scales/baselines differ")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _seed_stability(ig, gradient, deep, *, dataset, mode, label, output_dir):
    method_frames = []
    if not ig.empty:
        part = ig[ig["Attribution_Mode"].eq(mode)].copy()
        part["Method"] = "IG"
        method_frames.append(part)
    for method, frame in (("Gradient SHAP", gradient), ("Deep SHAP", deep)):
        if not frame.empty:
            part = frame.copy()
            part["Method"] = method
            method_frames.append(part)
    if not method_frames:
        return pd.DataFrame(), pd.DataFrame()
    all_rows = pd.concat(method_frames, ignore_index=True)
    all_rows = all_rows[all_rows["Dataset"].astype(str).eq(str(dataset))]
    if label is not None:
        all_rows = all_rows[pd.to_numeric(all_rows["True_Label"], errors="coerce").eq(int(label))]
    if all_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    seed_rows = []
    for (method, seed, true_label, feature), group in all_rows.groupby(
        ["Method", "Seed", "True_Label", "Feature"], dropna=False
    ):
        weights = group.get("Sampling_Weight", pd.Series(1.0, index=group.index))
        mean_abs = _weighted_mean(group["Absolute_Attribution"], weights)
        seed_rows.append({"Method": method, "Seed": int(seed), "True_Label": int(true_label),
                          "Feature": feature, "Weighted_Mean_Absolute_Attribution": mean_abs,
                          "N": int(group["Variant_ID"].nunique())})
    per_seed = pd.DataFrame(seed_rows)
    across = per_seed.groupby(["Method", "True_Label", "Feature"], dropna=False)[
        "Weighted_Mean_Absolute_Attribution"
    ].agg(Mean="mean", SD="std", Seed_Count="count").reset_index()
    per_seed.to_csv(output_dir / f"feature_attribution_by_seed_{dataset}.csv", index=False)
    across.to_csv(output_dir / f"feature_attribution_seed_stability_{dataset}.csv", index=False)
    labels = [0, 1] if label is None else [int(label)]
    methods = ["IG", "Gradient SHAP", "Deep SHAP"]
    fig, axes = plt.subplots(len(labels), len(methods), figsize=(19, 6.5 * len(labels)), squeeze=False)
    for row_index, true_label in enumerate(labels):
        for col_index, method in enumerate(methods):
            ax = axes[row_index, col_index]
            part = across[(across["True_Label"].eq(true_label)) & across["Method"].eq(method)]
            if part.empty:
                ax.text(0.5, 0.5, "No completed rows", ha="center", va="center")
                ax.set_axis_off()
                continue
            chosen = part.nlargest(12, "Mean").sort_values("Mean")
            sd = chosen["SD"].fillna(0).to_numpy(float)
            ax.errorbar(chosen["Mean"], chosen["Feature"], xerr=sd, fmt="o", capsize=3)
            ax.set_title(f"{method} | {'Benign' if true_label == 0 else 'Pathogenic'} | mean ± SD across seeds")
            ax.set_xlabel("Mean absolute attribution (logit units)")
            ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Feature attribution stability across seeds | {dataset}\nSHAP uses the selected explanation subset; compare variability within a method.")
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_attribution_seed_stability_{dataset}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return per_seed, across


def _attention_context_heatmap(attention, *, dataset, seed, output_dir):
    part = _subset(attention, dataset, seed, None)
    if part.empty:
        return
    needed = {"Query_Modality", "Context_Modality", "Mean_Attention_Over_Heads"}
    if not needed.issubset(part.columns):
        return
    per_query_mass = part.groupby(
        ["Variant_ID", "True_Label", "Layer", "Query_Modality", "Context_Modality", "Query_Token"],
        dropna=False,
    )["Mean_Attention_Over_Heads"].sum().rename("Context_Mass").reset_index()
    summary = per_query_mass.groupby(
        ["True_Label", "Query_Modality", "Context_Modality"], dropna=False
    )["Context_Mass"].mean().reset_index()
    labels = sorted(summary["True_Label"].dropna().unique())
    fig, axes = plt.subplots(1, max(1, len(labels)), figsize=(6 * max(1, len(labels)), 5), squeeze=False)
    image = None
    for ax, true_label in zip(axes.flat, labels):
        table = summary[summary["True_Label"].eq(true_label)].pivot(
            index="Query_Modality", columns="Context_Modality", values="Context_Mass"
        )
        image = ax.imshow(table.to_numpy(float), cmap="magma", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set_title("Benign" if int(true_label) == 0 else "Pathogenic")
        ax.set_xlabel("Context modality")
        ax.set_ylabel("Query modality")
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat), label="Mean attention weight")
    fig.suptitle(f"Mean attention mass by query → context modality | {dataset} | seed {seed}\nMass sums key weights within each context per query before averaging. It describes information flow, not signed effect on prediction.")
    fig.tight_layout()
    fig.savefig(output_dir / f"attention_modality_routing_{dataset}_seed{seed}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_explainability_comparison(
    *, output_root, dataset, seed, label, mode, top_features,
    ig_modalities, ig_features, ig_positions, occlusion_individual, attention_summary,
    shap_exact_groups, shap_gradient_modalities, shap_gradient_features,
    shap_gradient_positions, shap_gradient_completeness,
    shap_deep_modalities, shap_deep_features, shap_deep_positions,
    shap_deep_completeness,
):
    """Save detailed visual explanations plus matched, same-sample diagnostics."""
    output_dir = Path(output_root) / "interpretability_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    METHOD_GUIDE.to_csv(output_dir / "how_to_read_explainers.csv", index=False)

    # Per-sample signed token maps preserve direction, unlike absolute-importance curves.
    for ig_mode in ("full_prediction", "alt_vs_ref"):
        _make_signed_heatmap(ig_positions, method="IG", dataset=dataset, seed=seed,
                             label=label, mode=ig_mode, output_dir=output_dir)
    _make_signed_heatmap(shap_gradient_positions, method="Gradient_SHAP", dataset=dataset,
                         seed=seed, label=label, mode=mode, output_dir=output_dir)
    _make_signed_heatmap(shap_deep_positions, method="Deep_SHAP", dataset=dataset,
                         seed=seed, label=label, mode=mode, output_dir=output_dir)

    # SHAP beeswarms show both effect direction/magnitude and feature value.
    _draw_beeswarm(shap_gradient_features, explainer="Gradient", dataset=dataset,
                   seed=seed, label=label, top_features=top_features, output_dir=output_dir)
    _draw_beeswarm(shap_deep_features, explainer="Deep", dataset=dataset,
                   seed=seed, label=label, top_features=top_features, output_dir=output_dir)

    waterfall_cases = _render_waterfalls(
        exact=shap_exact_groups,
        feature_frames=(shap_gradient_features, shap_deep_features),
        position_frames=(shap_gradient_positions, shap_deep_positions),
        completeness_frames=(shap_gradient_completeness, shap_deep_completeness),
        dataset=dataset, seed=seed, label=label, output_dir=output_dir,
    )
    waterfall_cases.to_csv(output_dir / f"local_waterfall_cases_{dataset}_seed{seed}.csv", index=False)

    # Same-sample feature and token-level agreement; values are not assumed to be identical.
    matched_features, feature_agreement = _feature_method_comparison(
        ig_features, occlusion_individual, shap_gradient_features, shap_deep_features,
        dataset=dataset, mode=mode,
    )
    matched_tokens, token_agreement = _token_method_comparison(
        ig_positions, occlusion_individual, shap_gradient_positions,
        shap_deep_positions, dataset=dataset, mode=mode,
    )
    matched_features.to_csv(output_dir / f"matched_feature_method_rows_{dataset}.csv", index=False)
    feature_agreement.to_csv(output_dir / f"matched_feature_method_agreement_{dataset}.csv", index=False)
    matched_tokens.to_csv(output_dir / f"matched_token_method_rows_{dataset}.csv", index=False)
    token_agreement.to_csv(output_dir / f"matched_token_method_agreement_{dataset}.csv", index=False)
    matched_modalities, modality_agreement = _modality_method_comparison(
        ig_modalities, shap_exact_groups, shap_gradient_modalities,
        shap_deep_modalities, dataset=dataset, mode=mode,
    )
    matched_modalities.to_csv(output_dir / f"matched_modality_method_rows_{dataset}.csv", index=False)
    modality_agreement.to_csv(output_dir / f"matched_modality_method_agreement_{dataset}.csv", index=False)
    _plot_method_scatter(
        matched_features, feature_agreement, value_columns=None,
        title_prefix=f"Feature-level method agreement | {dataset}",
        output_path=output_dir / f"matched_feature_method_scatter_{dataset}.png",
    )
    _plot_method_scatter(
        matched_tokens, token_agreement, value_columns=None,
        title_prefix=f"Token-level method agreement | {dataset}",
        output_path=output_dir / f"matched_token_method_scatter_{dataset}.png",
    )
    _plot_modality_scatter(
        matched_modalities, modality_agreement, dataset=dataset,
        output_path=output_dir / f"matched_modality_method_scatter_{dataset}.png",
    )

    _, seed_stability = _seed_stability(
        ig_features, shap_gradient_features, shap_deep_features,
        dataset=dataset, mode=mode, label=label, output_dir=output_dir,
    )
    _attention_context_heatmap(attention_summary, dataset=dataset, seed=seed, output_dir=output_dir)

    print("Interpretability comparison saved to:", output_dir)
    print("Signed effects: positive raises the pathogenic logit, negative lowers it. Attention maps show routing weights, not effect direction.")
    print("Method-agreement tables are diagnostics on matched samples; different baselines/perturbations mean their magnitudes are not interchangeable.")
    return {
        "output_dir": output_dir,
        "feature_method_agreement": feature_agreement,
        "token_method_agreement": token_agreement,
        "modality_method_agreement": modality_agreement,
        "seed_stability": seed_stability,
        "waterfall_cases": waterfall_cases,
    }

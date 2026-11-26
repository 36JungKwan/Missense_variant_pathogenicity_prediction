"""Official figures of the TriMVPP capstone report (Chapter 6, Results and Discussion).

This is the script used to draw every figure of Chapter 6 of the report. It is kept in
the repository so that the figures can be regenerated from the experiment outputs.

Usage:  python tools/official_visualization.py --report REPORT_DIR [--interpret DIR] [--data DIR] [--out DIR]

REPORT_DIR is the LaTeX report folder (it contains Chapter6/chapter6.tex and
Appendix2/appendix2.tex). Figures are written to --out (default:
visualization/official/chapter6_figures) and the per-stratum accuracy tables to
<out>/../figure_data.
Needs:  matplotlib, numpy, pandas

Result figures read their numbers directly from the tables in Appendix2/appendix2.tex,
so a figure can never disagree with its table. Interpretability figures read the CSV
files in DIR (default: Missense_new/experiments/cross_attention_explainability_v2)
and the annotated parquet files in --data to rebuild figure_data/accuracy_*.csv.
Each figure is written as PDF (used by LaTeX) and PNG (preview, 300 dpi) in Figs/.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "visualization" / "official" / "chapter6_figures"
DATA_OUT = REPO / "visualization" / "official" / "figure_data"
TEXS = []  # set from --report in __main__
WIDTH = 5.6  # inches, text width of the thesis

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED, GRID = "#222222", "#555555", "#dddddd"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f7fd", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "legend.frameon": False, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- helpers
def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", name)


def unwrap(cell):
    """Remove LaTeX commands such as \\textbf{...} while keeping their content."""
    old = None
    while cell != old:
        old = cell
        cell = re.sub(r"\\(?:textbf|underline|texttt|textit|emph|mathbf)\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace(r"$-$", "-").replace(r"$\pm$", "+-").replace(r"\&", "&")
    return re.sub(r"\s+", " ", cell.replace(r"\_", "_").replace("~", " ")).strip()


def table(label):
    """Return the cells of all data rows of the table with the given label."""
    tex = "\n".join(p.read_text(encoding="utf8") for p in TEXS)
    tex = re.sub(r"\\multirow\{\d+\}\{\*\}\{(.*?)\}(?=\s*&)", r"\1", tex)
    body = tex[tex.index(r"\label{%s}" % label):]
    body = body[:body.index(r"\end{tabular}")]
    body = body[body.index(r"\midrule"):]
    rows = []
    for line in body.splitlines():
        if "&" in re.sub(r"\\&", "", line) and not line.strip().startswith("%"):
            line = re.sub(r"\\\\\s*$", "", line.strip())
            rows.append([unwrap(c) for c in re.split(r"(?<!\\)&", line)])
    return rows


def num(cell):
    """'0.8227 +- 0.0016' -> (0.8227, 0.0016); '-0.0461' -> (-0.0461, nan)."""
    parts = [float(p.replace(",", "")) for p in cell.split("+-")]
    return parts[0], (parts[1] if len(parts) > 1 else np.nan)


# ---------------------------------------------------------------- heatmaps
def heatmap(mean, sd, rows, cols, xlabel, ylabel, name, height):
    fig, ax = plt.subplots(figsize=(WIDTH, height), constrained_layout=True)
    im = ax.pcolormesh(np.arange(len(cols) + 1) - .5, np.arange(len(rows) + 1) - .5, mean,
                       cmap=SEQ, edgecolors="white", linewidth=2)
    ax.invert_yaxis()
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    best = np.unravel_index(np.argmax(mean), mean.shape)
    dark = mean.min() + 0.62 * (mean.max() - mean.min())
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mean[i, j]:.4f}\n\u00b1{sd[i, j]:.4f}", ha="center", va="center",
                    fontsize=8, linespacing=1.25, fontweight="bold" if (i, j) == best else "normal",
                    color="white" if mean[i, j] > dark else INK)
    ax.add_patch(plt.Rectangle((best[1] - .48, best[0] - .48), .96, .96, fill=False, ec=INK, lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Validation MCC")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, name)


def fig_backbones():
    rows = table("tab:res_step1_val_backbones")
    dna = list(dict.fromkeys(r[0] for r in rows))
    prot = list(dict.fromkeys(r[1] for r in rows))
    m, s = np.zeros((len(dna), len(prot))), np.zeros((len(dna), len(prot)))
    for r in rows:
        m[dna.index(r[0]), prot.index(r[1])], s[dna.index(r[0]), prot.index(r[1])] = num(r[2])
    heatmap(m, s, dna, prot, "Protein foundation model", "DNA foundation model",
            "fig_backbone_pairs", 2.3)


def fig_lengths():
    rows = table("tab:res_step2_val_lengths")
    dna = list(dict.fromkeys(r[0] for r in rows))
    prot = list(dict.fromkeys(r[1] for r in rows))
    m, s = np.zeros((len(dna), len(prot))), np.zeros((len(dna), len(prot)))
    for r in rows:
        m[dna.index(r[0]), prot.index(r[1])], s[dna.index(r[0]), prot.index(r[1])] = num(r[2])
    heatmap(m, s, dna, prot, "Protein window length (aa)", "DNA window length (bp)",
            "fig_context_lengths", 3.2)


# ---------------------------------------------------------------- ablation bars
def delta_bars(rows, names, name, height, labels=True, fs=8):
    val = np.array([num(r[2])[0] for r in rows])
    test = np.array([num(r[4])[0] for r in rows])
    sd = np.array([num(r[4])[1] for r in rows])
    y, h = np.arange(len(rows))[::-1], 0.36
    fig, ax = plt.subplots(figsize=(WIDTH, height), constrained_layout=True)
    ax.barh(y + h / 2 + .02, val, h, color=BLUE, label="Validation")
    ax.barh(y - h / 2 - .02, test, h, color=ORANGE, label="Held-out Test", xerr=sd,
            error_kw=dict(ecolor=MUTED, elinewidth=0.7, capsize=1.6))
    lo = min(val.min(), np.nanmin(test - sd))
    if labels:
        off = abs(lo) * 0.012
        for yi, v, t, e in zip(y, val, test, sd):
            ax.text(v - off, yi + h / 2 + .02, f"{v:.4f}".replace("-", "\u2212"), ha="right", va="center", fontsize=7.2)
            ax.text(t - e - off, yi - h / 2 - .02, f"{t:.4f}".replace("-", "\u2212"), ha="right", va="center", fontsize=7.2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, names, fontsize=fs)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    hi = max(0.0, np.nanmax(test + sd))
    ax.set_xlim(lo * (1.16 if labels else 1.04), hi + abs(lo) * 0.02)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$\Delta$MCC relative to the full model")
    fig.legend(loc="outside upper right", ncol=2, handlelength=1.1, columnspacing=1.4)
    save(fig, name)


def short(cell):
    return re.sub(r"\s*\(.*\)$", "", cell)


def fig_modality():
    rows = [r for r in table("tab:res_modality_ablation") if not r[0].startswith("Full")]
    delta_bars(rows, [short(r[0]) for r in rows], "fig_ablation_modality", 4.3)


def fig_groups():
    rows = [r for r in table("tab:res_feature_group_ablation") if not r[0].startswith("Full")]
    delta_bars(rows, [short(r[0]) for r in rows], "fig_ablation_groups", 3.1)


FEAT = {
    "gnomADe_AF": "gnomAD exome AF", "AF": "1000 Genomes AF",
    "phyloP100way_vertebrate": "phyloP (100 vertebrates)", "phyloP470way_mammalian": "phyloP (470 mammals)",
    "phyloP17way_primate": "phyloP (17 primates)", "phastCons100way_vertebrate": "phastCons (100 vertebrates)",
    "phastCons470way_mammalian": "phastCons (470 mammals)", "phastCons17way_primate": "phastCons (17 primates)",
    "GERP++_RS": "GERP++ RS", "GERP++_NR": "GERP++ NR", "GERP_92_mammals": "GERP (92 mammals)",
    "dna_LVD_L2": "DNA LVD (L2)", "dna_LVD_Cosine": "DNA LVD (cosine)", "dna_LVD_Relative": "DNA LVD (relative)",
    "prot_LVD_L2": "Protein LVD (L2)", "prot_LVD_Cosine": "Protein LVD (cosine)",
    "prot_LVD_Relative": "Protein LVD (relative)", "dna_LID": "DNA LID", "prot_LID": "Protein LID",
}
FEAT.update({f"SpliceAI_pred_{a}_{b}": f"SpliceAI {a} {b}" for a in ("DS", "DP") for b in ("AG", "AL", "DG", "DL", "max")})


def feat(n):
    return FEAT.get(n, n)


def fig_lofo():
    rows = [r for r in table("tab:res_feature28_ablation") if not r[0].startswith("Full")]
    delta_bars(rows, [r[0] for r in rows],
               "fig_ablation_lofo", 6.6, labels=False, fs=7.8)


# ---------------------------------------------------------------- figures that replace result tables
GREEN, YELLOW = "#1baf7a", "#eda100"
CLF = {"MLP": (BLUE, "o"), "XGBoost": (ORANGE, "s"), "CatBoost": (GREEN, "^"), "LightGBM": (YELLOW, "D")}
BENCH3 = ["Held-out Test", "ClinVar HQ", "ProteinGym"]


def _hgrid(ax):
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def fig_pooling():
    rows = table("tab:res_step3_val_pooling")
    names = [r[0].replace("$", "") for r in rows]
    m = np.array([num(r[1]) for r in rows])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4), constrained_layout=True)
    col = [GRAY, GRAY, BLUE, BLUE, BLUE, BLUE]
    for yi, (v, s), c in zip(y, m, col):
        ax.errorbar(v, yi, xerr=s, fmt="o", ms=5, color=c, elinewidth=.9, capsize=2)
        ax.text(v + s + 0.0012, yi, f"{v:.4f}", va="center", fontsize=7.8)
    ax.set_yticks(y, names)
    ax.set_xlim(0.76, 0.832)
    ax.set_xlabel("Validation MCC")
    _hgrid(ax)
    save(fig, "fig_pooling")


def fig_internal_benchmark():
    fu = table("tab:res_fusion_classifier_full")
    pm = table("tab:res_pure_ml_baselines")
    strat, cur = [], None
    rec = {}  # (strategy, classifier, bench, metric) -> (mean, sd)
    for r in fu:
        cur = r[0] or cur
        if cur not in strat:
            strat.append(cur)
        for b, bn in enumerate(BENCH3):
            rec[(cur, r[1], bn, "MCC")] = num(r[2 + 3 * b])
            rec[(cur, r[1], bn, "AUPRC")] = num(r[3 + 3 * b])
            rec[(cur, r[1], bn, "AUROC")] = num(r[4 + 3 * b])
    for i, r in enumerate(pm):
        bn = BENCH3[i // 3]
        rec[("Pure ML", r[0], bn, "MCC")] = num(r[2])
        rec[("Pure ML", r[0], bn, "AUPRC")] = num(r[5])
        rec[("Pure ML", r[0], bn, "AUROC")] = num(r[6])
    strat.append("Pure ML")
    y0 = np.arange(len(strat))[::-1]
    mets = ("MCC", "AUROC", "AUPRC")
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 7.9), sharey=True, constrained_layout=True)
    for i, met in enumerate(mets):
        for j, bn in enumerate(BENCH3):
            ax = axes[i, j]
            for k, (c, (colr, mk)) in enumerate(CLF.items()):
                for s, yy in zip(strat, y0):
                    v = rec.get((s, c, bn, met))
                    if v is None:
                        continue
                    ax.errorbar(v[0], yy + (1.5 - k) * 0.17, xerr=v[1], fmt=mk, ms=3.6, color=colr, mec="white",
                                mew=0.4, elinewidth=0.7, capsize=0, label=c if (s == strat[0]) else None)
            ax.axhspan(y0[strat.index("Cross-Attention")] - .45, y0[strat.index("Cross-Attention")] + .45,
                       color="#eef3fb", zorder=0)
            ax.axhline(0.5, color=GRID, lw=.8)
            _hgrid(ax)
            ax.tick_params(axis="x", labelsize=7.5)
            if i == 0:
                ax.set_title(bn, fontsize=9)
            ax.set_xlabel(met, fontsize=8.5)
    for i in range(len(mets)):
        axes[i, 0].set_yticks(y0, strat)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=4, handlelength=1.0, title="Classifier", title_fontsize=8.5,
               alignment="left")
    save(fig, "fig_internal_benchmark")


def _sota():
    rows = table("tab:res_ours_vs_sota_full")
    names = [r[0] for r in rows]
    vals = np.array([[num(c)[0] for c in r[1:16]] for r in rows])
    sds = np.array([[num(c)[1] for c in r[1:16]] for r in rows])
    return names, vals, sds  # columns: per benchmark MCC, Acc, F1, AUPRC, AUROC


def fig_sota():
    names, v, s = _sota()
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 5.0), sharey=True, constrained_layout=True)
    for j, (ax, bn) in enumerate(zip(axes, BENCH3)):
        mcc, prc, auc = v[:, 5 * j], v[:, 5 * j + 3], v[:, 5 * j + 4]
        lo, hi = np.minimum.reduce([mcc, prc, auc]), np.maximum.reduce([mcc, prc, auc])
        ax.hlines(y, lo, hi, color=GRID, lw=1.2, zorder=1)
        ax.errorbar(mcc, y, xerr=np.nan_to_num(s[:, 5 * j]), fmt="o", ms=4, color=BLUE, mec="white", mew=.4,
                    elinewidth=.7, label="MCC", zorder=3)
        ax.plot(auc, y, "D", ms=3.6, mfc="white", mec=ORANGE, mew=1.0, label="AUROC", zorder=3)
        ax.plot(prc, y, "^", ms=3.9, mfc="white", mec=GREEN, mew=1.0, label="AUPRC", zorder=3)
        ax.plot(mcc[0], y[0], "o", ms=5.5, color=INK, zorder=4)
        ax.plot(auc[0], y[0], "D", ms=4.4, mfc="white", mec=INK, mew=1.2, zorder=4)
        ax.plot(prc[0], y[0], "^", ms=4.7, mfc="white", mec=INK, mew=1.2, zorder=4)
        ax.set_title(bn, fontsize=9)
        ax.set_xlim(0.3, 1.0)
        _hgrid(ax)
    axes[0].set_yticks(y, names, fontsize=7.8)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig.supxlabel("Score", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=3, handlelength=1.0)
    save(fig, "fig_sota")


def fig_sota_threshold():
    names, v, _ = _sota()
    fig = plt.figure(figsize=(WIDTH, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax, bx, cx = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])
    lab = {"TriMVPP", "REVEL", "gMVP", "popEVE", "CADD", "DANN", "M-CAP", "MPC", "PrimateAI"}
    off = {"TriMVPP": (-6, 9, "right"), "REVEL": (-5, 3, "right"),
           "gMVP": (4, 0, "left"), "popEVE": (4, -2, "left"), "CADD": (-4, 2, "right"), "DANN": (4, 0, "left"),
           "M-CAP": (4, 0, "left"), "MPC": (4, -3, "left"), "PrimateAI": (4, 2, "left")}
    off_prc = dict(off, gMVP=(-2, -10, "right"), REVEL=(-5, 4, "right"), **{"M-CAP": (4, -2, "left")},
                   popEVE=(4, 2, "left"), PrimateAI=(-4, 2, "right"), MPC=(4, -2, "left"))
    for a, (xi, yi, xl, yl), o in ((ax, (4, 0, "AUROC", "MCC"), off), (bx, (3, 0, "AUPRC", "MCC"), off_prc),
                                   (cx, (1, 2, "Accuracy", "F1"), off)):
        x, yv = v[:, xi], v[:, yi]
        a.scatter(x[1:], yv[1:], s=16, color=BLUE, edgecolor="white", linewidth=.5, zorder=3)
        a.scatter(x[:1], yv[:1], s=26, color=INK, zorder=4)
        for n, xx, yy in zip(names, x, yv):
            if n in lab:
                dx, dy, ha = o[n]
                a.annotate(n, (xx, yy), xytext=(dx, dy), textcoords="offset points", fontsize=7, ha=ha, va="center")
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.grid(color=GRID, lw=.6)
        a.set_axisbelow(True)
    allbenign = 4437 / 6095
    cx.axvline(allbenign, color=MUTED, lw=.8, ls=(0, (3, 2)))
    cx.text(allbenign + 0.004, 0.36, "All benign", fontsize=7, color=MUTED, rotation=90, va="bottom")
    ax.set_title("(a) AUROC and MCC", loc="left", fontsize=9)
    bx.set_title("(b) AUPRC and MCC", loc="left", fontsize=9)
    cx.set_title("(c) Accuracy and F1", loc="left", fontsize=9)
    save(fig, "fig_sota_threshold")


def fig_af_distribution():
    rows = table("tab:res_af_distribution")
    rows = [r for r in rows if not r[0].startswith("Total")]
    lab = ["Not observed", "$\\leq 10^{-5}$", "$10^{-5}$ to $10^{-4}$", "$> 10^{-4}$"]
    cnt = np.array([[float(c.replace(",", "")) for c in r[1:7]] for r in rows])  # strata x (bench*class)
    cols = ["#1c5cab", "#5b93db", "#a9c8f0", "#e3edf9"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.9), sharey=True, constrained_layout=True)
    for j, (ax, bn) in enumerate(zip(axes, BENCH3)):
        for k, cl in enumerate(("Benign", "Pathogenic")):
            c = cnt[:, 2 * j + k]
            fr = c / c.sum()
            left = 0
            for st in range(4):
                ax.barh(1 - k, fr[st], 0.62, left=left, color=cols[st], edgecolor="white", linewidth=1,
                        label=lab[st] if (j == 0 and k == 0) else None)
                if fr[st] > 0.09:
                    ax.text(left + fr[st] / 2, 1 - k, f"{int(c[st]):,}", ha="center", va="center", fontsize=6.8,
                            color="white" if st < 2 else INK)
                left += fr[st]
        ax.set_xlim(0, 1)
        ax.set_title(bn, fontsize=9)
        ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([1, 0], ["Benign", "Pathogenic"])
    fig.supxlabel("Fraction of variants", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=4, handlelength=1.0, title="gnomAD exome AF",
               title_fontsize=8, alignment="left", fontsize=7.8)
    save(fig, "fig_af_distribution")


def fig_af_intervention():
    rows = table("tab:res_af_intervention")
    cur, rec = None, {}
    for r in rows:
        cur = r[0] or cur
        rec[(cur, r[1])] = [num(c) if c != "-" else (np.nan, np.nan) for c in r[2:6]]
    sc = [("Observed", "Observed AF", BLUE), ("AF = median", "AF = training median", ORANGE), ("AF = 0", "AF = 0", GRAY)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), constrained_layout=True)
    axes = axes.ravel()
    x, w = np.arange(3), 0.26
    for ax, (mi, ml) in zip(axes, ((0, "(a) MCC"), (1, "(b) AUROC"), (2, "(c) AUPRC"), (3, "(d) Flip rate"))):
        for k, (key, lab_, col) in enumerate(sc):
            m = [rec[(bn, key)][mi][0] for bn in BENCH3]
            s = [rec[(bn, key)][mi][1] for bn in BENCH3]
            ax.bar(x + (k - 1) * w, m, w * 0.94, yerr=s, color=col, label=lab_,
                   error_kw=dict(ecolor=MUTED, elinewidth=.6, capsize=1.2))
        ax.set_xticks(x, ["Held-out\nTest", "ClinVar\nHQ", "Protein-\nGym"], fontsize=7.8)
        ax.set_title(ml, loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0.5, 0.92)
    axes[1].set_ylim(0.85, 1.0)
    axes[2].set_ylim(0.8, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=3, handlelength=1.0)
    save(fig, "fig_af_intervention")


# ---------------------------------------------------------------- interpretability (v2 run)
BENCH = {"Train3Val_Test": ("Held-out Test", BLUE), "Train3Val_ClinVarHQ": ("ClinVar HQ", ORANGE),
         "Train3Val_ProteinGym": ("ProteinGym", GRAY)}
STRATA = ["0", "(0,1e-5]", "(1e-5,1e-4]", ">1e-4"]
STRATA_LAB = ["Not\nobserved", "$\\leq 10^{-5}$", "$10^{-5}$ to\n$10^{-4}$", "$> 10^{-4}$"]
CLASS = {0: "Benign", 1: "Pathogenic"}


def wmean(g, col):
    return np.average(g[col], weights=g["Sampling_Weight"].astype(float))


def fig_ig_features(d, top=10):
    """Sampling-weighted mean |IG| of each tabular feature, mean and SD across seeds."""
    t = pd.read_csv(d / "ig_tabular_features.csv",
                    usecols=["Seed", "Dataset", "Attribution_Mode", "Feature", "Absolute_Attribution", "Sampling_Weight"])
    t = t[t.Attribution_Mode == "full_prediction"]
    per_seed = t.groupby(["Dataset", "Seed", "Feature"]).apply(wmean, "Absolute_Attribution", include_groups=False)
    g = per_seed.groupby(["Dataset", "Feature"]).agg(["mean", "std"])
    order = g.loc["Train3Val_Test"]["mean"].sort_values(ascending=False).index[:top]
    y, h = np.arange(top)[::-1], 0.26
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3), constrained_layout=True)
    for k, (ds, (lab, col)) in enumerate(BENCH.items()):
        m, s = g.loc[ds].loc[order, "mean"], g.loc[ds].loc[order, "std"]
        ax.barh(y + (1 - k) * (h + .02), m, h, xerr=s, color=col, label=lab,
                error_kw=dict(ecolor=MUTED, elinewidth=.7, capsize=1.5))
    ax.set_yticks(y, [feat(f) for f in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean absolute integrated gradients attribution")
    fig.legend(loc="outside upper right", ncol=3, handlelength=1.1, columnspacing=1.4)
    save(fig, "fig_ig_features")
    return g


def fig_ig_tokens(d):
    s = pd.read_csv(d / "ig_token_position_summary_weighted.csv")
    s = s[(s.Dataset == "Train3Val_Test") & (s.Allele == "Alt")]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.6), sharex=True, constrained_layout=True)
    for ax, (mode, tag) in zip(axes, [("full_prediction", "(a) Empirical baseline"), ("alt_vs_ref", "(b) REF baseline")]):
        q = s[s.Attribution_Mode == mode].groupby(["Modality", "Relative_Token_Position"]).Weighted_Mean_Absolute_Attribution.mean()
        ax.axvline(0, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=0)
        for m, c, mk in (("Protein", ORANGE, "o"), ("DNA", BLUE, "s")):
            v = q[m]
            ax.plot(v.index, v.values, color=c, lw=1.5, marker=mk, ms=3.2, label=m)
            if m == "Protein":
                ax.annotate(f"{v[0]:.2f}", (0, v[0]), xytext=(7, -3), textcoords="offset points", fontsize=8)
        ax.set_title(tag, loc="left", fontsize=9)
        ax.set_ylabel("Mean absolute\nattribution")
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=-0.03 * q.max())
    axes[1].set_xlabel("Token position relative to the variant token")
    axes[1].set_xticks(range(-16, 17, 4))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=2, handlelength=1.6)
    save(fig, "fig_ig_tokens")


def _strata_axes(ax):
    ax.set_xticks(range(4), STRATA_LAB, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)


def fig_ig_af(d):
    t = pd.read_csv(d / "ig_tabular_features.csv",
                    usecols=["Seed", "Dataset", "True_Label", "AF_Stratum", "Attribution_Mode", "Feature", "Signed_Attribution"])
    t = t[(t.Attribution_Mode == "full_prediction") & (t.Feature == "gnomADe_AF")]
    g = t.groupby(["Dataset", "True_Label", "AF_Stratum", "Seed"]).Signed_Attribution.mean()
    g = g.groupby(["Dataset", "True_Label", "AF_Stratum"]).agg(["mean", "std"])
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharey=True, constrained_layout=True)
    for ax, (ds, (lab, _)) in zip(axes, BENCH.items()):
        ax.axhline(0, color=INK, lw=.8)
        for lb, c, mk, dx in ((0, BLUE, "o", -.1), (1, ORANGE, "^", .1)):
            v = g.loc[(ds, lb)].reindex(STRATA)
            ax.errorbar(np.arange(4) + dx, v["mean"], yerr=v["std"], fmt=mk + "-", ms=5, lw=1.2, color=c,
                        capsize=2, label=CLASS[lb])
        ax.set_title(lab, fontsize=9)
        _strata_axes(ax)
    axes[0].set_ylabel("Signed attribution")
    fig.supxlabel("gnomAD exome AF", fontsize=9.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=2, handlelength=1.6)
    save(fig, "fig_ig_gnomad_af")


OCC = [("AF", "AF"), ("Conservation", "Conservation"), ("Protein_variant_center_alt_to_ref", "Protein, variant token"),
       ("Protein_neighbor_tokens_alt_to_ref", "Protein, other tokens"), ("Geometry", "Geometry"),
       ("SpliceAI", "SpliceAI"), ("DNA_window_alt_to_ref", "DNA window")]


def fig_occlusion(d):
    o = pd.read_csv(d / "occlusion_groups.csv",
                    usecols=["Seed", "Dataset", "True_Label", "Feature_or_Position", "Logit_Delta_Actual_Minus_Occluded", "Sampling_Weight"])
    per = o.groupby(["Dataset", "True_Label", "Feature_or_Position", "Seed"]).apply(
        wmean, "Logit_Delta_Actual_Minus_Occluded", include_groups=False)
    g = per.groupby(["Dataset", "True_Label", "Feature_or_Position"]).agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), sharey=True, constrained_layout=True)
    y, h = np.arange(len(OCC))[::-1], 0.26
    for ax, lb in zip(axes, (0, 1)):
        for k, (ds, (lab, col)) in enumerate(BENCH.items()):
            v = g.loc[(ds, lb)].reindex([k_ for k_, _ in OCC])
            ax.barh(y + (1 - k) * (h + .02), v["mean"], h, xerr=v["std"], color=col, label=lab,
                    error_kw=dict(ecolor=MUTED, elinewidth=.7, capsize=1.5))
        ax.axvline(0, color=INK, lw=.8)
        ax.set_title(f"({'ab'[lb]}) {CLASS[lb]} variants", loc="left", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [n for _, n in OCC])
    axes[0].spines["left"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    fig.supxlabel("Change in logit when the group is replaced (observed minus occluded)", fontsize=9)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside upper right", ncol=3, handlelength=1.1)
    save(fig, "fig_occlusion_groups")
    return g


def accuracy_tables(d, data_dir):
    """Per-variant predictions of the five seeds joined with raw annotation."""
    files = {"Train3Val_Test": "clinvar_dbsnp_train_val_test/test/test_full_seq_after_vep_final.parquet",
             "Train3Val_ClinVarHQ": "benchmark_test/after_vep/clinvarhq_full_seq_after_vep_final.parquet",
             "Train3Val_ProteinGym": "benchmark_test/after_vep/proteingym_full_seq_after_vep_final.parquet"}
    ann = pd.concat([pd.read_parquet(data_dir / f, columns=["Variant_ID", "gnomADe_AF", "phyloP100way_vertebrate"]).assign(Dataset=k)
                     for k, f in files.items()])
    p = pd.read_csv(d / "predictions_all.csv", usecols=["Seed", "Dataset", "Variant_ID", "True_Label", "Predicted_Label"])
    p = p.merge(ann, on=["Dataset", "Variant_ID"], how="left", validate="m:1")
    af = pd.to_numeric(p.gnomADe_AF, errors="coerce").fillna(0)
    p["AF_Stratum"] = pd.cut(af, [-1, 0, 1e-5, 1e-4, 1], labels=STRATA).astype(str)
    p["PhyloP"] = pd.cut(pd.to_numeric(p.phyloP100way_vertebrate, errors="coerce"), [-1e9, 2, 5, 1e9],
                         labels=["< 2", "2 to 5", "> 5"]).astype(str)
    p["Correct"] = (p.True_Label == p.Predicted_Label).astype(float)
    out = DATA_OUT
    out.mkdir(exist_ok=True)
    for name, keys in (("accuracy_by_af", ["Dataset", "True_Label", "AF_Stratum"]),
                       ("accuracy_by_af_phylop", ["Dataset", "True_Label", "AF_Stratum", "PhyloP"])):
        a = p.groupby(keys + ["Seed"]).Correct.mean().groupby(keys).agg(["mean", "std"])
        a["N"] = p[p.Seed == p.Seed.iloc[0]].groupby(keys).size()
        a.reset_index().to_csv(out / f"{name}.csv", index=False)


def fig_error(d, data_dir=None):
    if data_dir is not None:
        accuracy_tables(d, data_dir)
    a = pd.read_csv(DATA_OUT / "accuracy_by_af.csv")
    b = pd.read_csv(DATA_OUT / "accuracy_by_af_phylop.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), constrained_layout=True)
    for ax, lb in zip(axes[:2], (0, 1)):
        for k, (ds, (lab, col)) in enumerate(BENCH.items()):
            v = a[(a.Dataset == ds) & (a.True_Label == lb)].set_index("AF_Stratum").reindex(STRATA)
            ax.errorbar(np.arange(4) + (k - 1) * .08, v["mean"], yerr=v["std"], fmt="o-", ms=4, lw=1.2,
                        color=col, capsize=2, label=lab)
        ax.set_ylim(0, 1.04)
        ax.set_title(f"({'ab'[lb]}) {CLASS[lb]}: {'specificity' if lb == 0 else 'sensitivity'}", loc="left", fontsize=9)
        _strata_axes(ax)
    axes[0].set_ylabel("Fraction correctly classified")
    axes[1].set_xlabel("gnomAD exome AF", fontsize=9)
    ax = axes[2]
    for k, (ds, (lab, col)) in enumerate(BENCH.items()):
        v = b[(b.Dataset == ds) & (b.True_Label == 0) & (b.AF_Stratum == "0")].set_index("PhyloP").reindex(["< 2", "2 to 5", "> 5"])
        if v["mean"].notna().any():
            ax.errorbar(np.arange(3) + (k - 1) * .08, v["mean"], yerr=v["std"], fmt="o-", ms=4, lw=1.2, color=col, capsize=2)
    ax.set_xticks(range(3), ["< 2", "2 to 5", "> 5"], fontsize=8)
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title("(c) Benign, not observed", loc="left", fontsize=9)
    ax.set_xlabel("phyloP (100 vertebrates)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=3, handlelength=1.4)
    save(fig, "fig_error_af")


def fig_attention(d):
    a = pd.read_csv(d / "attention_summary.csv",
                    usecols=["Seed", "Dataset", "Variant_ID", "Layer", "Query_Token", "Context_Modality", "Context_Token", "Mean_Attention_Over_Heads"])
    a = a.rename(columns={"Mean_Attention_Over_Heads": "w"})
    sh = a.groupby(["Query_Token", "Layer", "Context_Modality", "Dataset", "Variant_ID", "Seed"]).w.sum()
    sh = sh.groupby(["Query_Token", "Layer", "Context_Modality"]).mean().unstack()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(WIDTH, 4.9), constrained_layout=True, height_ratios=[1.1, 1])
    cols = {"dna": "#eda100", "protein": "#1baf7a", "bio_geometry": "#2a78d6"}
    names = {"Protein:+0": "Query: protein variant token", "gnomADe_AF": "Query: gnomAD exome AF token"}
    xt, xl, x = [], [], 0.0
    for q in ("Protein:+0", "gnomADe_AF"):
        for lay in (1, 2, 3):
            b = 0
            for c in ("bio_geometry", "protein", "dna"):
                v = sh.loc[(q, lay)].get(c, np.nan)
                if np.isnan(v):
                    continue
                ax.bar(x, v, 0.74, bottom=b, color=cols[c], edgecolor="white", linewidth=1)
                ax.text(x, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=INK if c == "dna" else "white")
                b += v
            xt.append(x)
            xl.append(f"Layer {lay}")
            x += 0.85
        ax.text(x - 0.85 * 2, -0.13, names[q], ha="center", va="top", fontsize=9)
        x += 0.6
    ax.set_xticks(xt, xl, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of attention")
    ax.set_title("(a)", loc="left", fontsize=9, pad=8)
    h = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in ("bio_geometry", "protein", "dna")]
    fig.legend(h, ["Attended stream: tabular", "protein", "DNA"], loc="outside upper right", ncol=3, handlelength=1.1)
    p = a[(a.Query_Token == "gnomADe_AF") & (a.Context_Modality == "protein")].copy()
    p["pos"] = p.Context_Token.str.split(":").str[1].astype(int)
    s = p.groupby(["pos", "Seed"]).w.mean().groupby("pos").agg(["mean", "std"])
    bx.bar(s.index, s["mean"], 0.72, yerr=s["std"], color="#1baf7a", error_kw=dict(ecolor=MUTED, elinewidth=.6, capsize=1))
    bx.set_xlabel("Protein token position relative to the variant token", labelpad=3)
    bx.set_ylabel("Mean attention")
    bx.set_xticks(range(-16, 17, 4))
    bx.grid(axis="y", color=GRID, lw=.6)
    bx.set_axisbelow(True)
    bx.set_title("(b) Attention of the gnomAD exome AF token to protein positions", loc="left", fontsize=9)
    bx.text(0.7, s.loc[0, "mean"], f"{s.loc[0, 'mean']:.2f}", fontsize=8, va="top")
    save(fig, "fig_attention")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True,
                    help="LaTeX report folder containing Chapter6/chapter6.tex and Appendix2/appendix2.tex")
    ap.add_argument("--interpret", type=Path, default=REPO / "experiments" / "cross_attention_explainability_v2")
    ap.add_argument("--data", type=Path, default=REPO.parent / "data",
                    help="folder with the annotated parquet files; used to rebuild figure_data/accuracy_*.csv")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder for the figures")
    a = ap.parse_args()
    TEXS[:] = [a.report / "Chapter6" / "chapter6.tex", a.report / "Appendix2" / "appendix2.tex"]
    OUT = a.out
    fig_backbones()
    fig_lengths()
    fig_modality()
    fig_groups()
    fig_lofo()
    fig_pooling()
    fig_internal_benchmark()
    fig_sota()
    fig_sota_threshold()
    fig_af_distribution()
    fig_af_intervention()
    fig_ig_features(a.interpret)
    fig_ig_tokens(a.interpret)
    fig_ig_af(a.interpret)
    fig_occlusion(a.interpret)
    fig_error(a.interpret, a.data if a.data.is_dir() else None)
    fig_attention(a.interpret)

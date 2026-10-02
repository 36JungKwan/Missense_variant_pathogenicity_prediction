"""
Missense Variant Pathogenicity Prediction - Comprehensive Ablation Visualization Tool
====================================================================================
This script reads the results from four ablation experiment batches and generates
publication-ready visualizations focused on the Matthews Correlation Coefficient (MCC):
1. Modality/Feature Group Ablation (batch_run_20260904_1800)
2. Geometric Features Ablation (batch_run_20260929_1740)
3. Local Intrinsic Dimensionality (LID) Ablation & Architectures (batch_run_20260930_lid_ablation)
4. Sequence Length Grid Ablation (sequence_ablation_20261001_1206)

Outputs include:
- Global overview panels
- Dedicated per-dataset figures for each of the 4 benchmarks:
  * ClinVarHQ
  * ProteinGym
  * Test
  * UniProt

Output directory: experiments/visualizations_ablation/
Format: Both PNG (300 DPI) and PDF (Vector format).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Set overall publication style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
sns.set_theme(style="whitegrid", font="sans-serif")

# Color Palettes
BENCHMARK_COLORS = {
    'ClinVarHQ': '#2b5c8f',
    'ProteinGym': '#d95f02',
    'Test': '#7570b3',
    'UniProt': '#1b9e77',
    'Average': '#e7298a'
}

DATASET_NAME_MAP = {
    'Train3Val_ClinVarHQ': 'ClinVarHQ',
    'Train3Val_ProteinGym': 'ProteinGym',
    'Train3Val_Test': 'Test',
    'Train3Val_UniProt': 'UniProt',
    'clinvarhq': 'ClinVarHQ',
    'proteingym': 'ProteinGym',
    'test': 'Test',
    'uniprot': 'UniProt'
}


# ==============================================================================
# 1. MODALITY / FEATURE GROUP ABLATION (batch_run_20260904_1800)
# ==============================================================================
def run_modality_ablation(csv_path, out_dir):
    print("\n[1/4] Processing Modality/Feature Group Ablation...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter specified config: mean pooling, nt_v2_500m, esm1b_650m, Pure_XGBoost_Concat
    mask = (
        (df['Pooling'] == 'mean') &
        (df['DNA_Model'] == 'nt_v2_500m') &
        (df['Prot_Model'] == 'esm1b_650m') &
        (df['Network'] == 'Pure_XGBoost_Concat') &
        (df['Dataset'].isin(DATASET_NAME_MAP.keys()))
    )
    df_filtered = df[mask].copy()
    df_filtered['Clean_Dataset'] = df_filtered['Dataset'].map(DATASET_NAME_MAP)

    datasets = ['ClinVarHQ', 'ProteinGym', 'Test', 'UniProt']

    # Process Best Geom Combo vs other ablations for each dataset
    processed_rows = []
    for ds in datasets:
        sub = df_filtered[df_filtered['Clean_Dataset'] == ds]
        
        # Best geom combo for bio_core_dna_prot_spliceai_geom
        geom_combos = sub[sub['Ablation'] == 'bio_core_dna_prot_spliceai_geom']
        if not geom_combos.empty:
            best_geom = geom_combos.sort_values(by='MCC', ascending=False).iloc[0].to_dict()
            best_geom['Ablation_Label'] = 'Full Model*\n(+Best Geom)'
            best_geom['Ablation_Key'] = 'full_best_geom'
            processed_rows.append(best_geom)
        
        # Other modality combinations
        others = sub[sub['Ablation'] != 'bio_core_dna_prot_spliceai_geom']
        for _, row in others.iterrows():
            r = row.to_dict()
            r['Ablation_Key'] = row['Ablation']
            r['Ablation_Label'] = row['Ablation']
            processed_rows.append(r)

    df_proc = pd.DataFrame(processed_rows)

    # Standard clean labels for presentation (with newline wrapping)
    label_map = {
        'geom': 'Geom only',
        'dna_geom': 'Geom + DNA',
        'dna_prot': 'DNA + Prot',
        'geom_prot': 'Geom + Prot',
        'dna_geom_prot': 'DNA + Prot\n+ Geom',
        'dna_prot_spliceai': 'DNA + Prot\n+ SpliceAI',
        'bio_geom': 'Bio + Geom',
        'bio_dna_geom': 'Bio + DNA\n+ Geom',
        'bio_geom_prot': 'Bio + Geom\n+ Prot',
        'bio_dna_prot': 'Bio + DNA\n+ Prot',
        'bio_dna_geom_prot': 'Bio + DNA\n+ Prot + Geom',
        'bio_core_dna_prot_spliceai': 'Full w/o Geom\n(BioCore+DNA+Prot+SpliceAI)',
        'full_best_geom': 'Full Model*\n(+Best Geom)'
    }
    df_proc['Clean_Ablation'] = df_proc['Ablation_Key'].map(lambda x: label_map.get(x, x))

    prog_keys = ['geom', 'dna_geom', 'dna_geom_prot', 'bio_dna_geom_prot', 'full_best_geom']
    prog_labels = ['Geom only', '+ DNA', '+ Prot', '+ Bio', '+ SpliceAI & BioCore\n(Full Model*)']

    loo_pairs = [
        ('w/o Geom', 'bio_core_dna_prot_spliceai'),
        ('w/o BioCore/SpliceAI', 'bio_dna_geom_prot'),
        ('w/o Bio', 'dna_geom_prot'),
        ('w/o DNA', 'bio_geom_prot'),
        ('w/o Prot', 'bio_dna_geom')
    ]

    pivot_mcc = df_proc.pivot(index='Ablation_Key', columns='Clean_Dataset', values='MCC')
    pivot_mcc['Average'] = pivot_mcc.mean(axis=1)

    # -------------------------------------------------------------
    # 1. Global Overview 3-Panel Figure
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.52, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    order = [
        'geom', 'dna_geom', 'geom_prot', 'dna_prot', 'dna_geom_prot',
        'dna_prot_spliceai', 'bio_geom', 'bio_dna_geom', 'bio_geom_prot',
        'bio_dna_prot', 'bio_dna_geom_prot', 'bio_core_dna_prot_spliceai', 'full_best_geom'
    ]
    df_order = df_proc[df_proc['Ablation_Key'].isin(order)].copy()
    avg_map = pivot_mcc.loc[order, 'Average'].to_dict()
    df_order['Avg_MCC'] = df_order['Ablation_Key'].map(avg_map)
    df_order = df_order.sort_values(by='Avg_MCC', ascending=True)
    sorted_keys = df_order['Ablation_Key'].unique()
    sorted_labels = [label_map.get(k, k) for k in sorted_keys]

    x_indices = np.arange(len(sorted_keys))
    bar_width = 0.18

    for i, ds in enumerate(datasets):
        ds_vals = [pivot_mcc.loc[k, ds] if k in pivot_mcc.index else 0 for k in sorted_keys]
        ax1.bar(x_indices + (i - 1.5) * bar_width, ds_vals, width=bar_width,
                label=ds, color=BENCHMARK_COLORS[ds], alpha=0.9, edgecolor='black', linewidth=0.5)

    ax1.plot(x_indices, [pivot_mcc.loc[k, 'Average'] for k in sorted_keys],
             color='#d62728', marker='o', linewidth=2.5, markersize=7, label='Average (All Datasets)', zorder=5)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(sorted_labels, rotation=30, ha='right', fontsize=9.5, fontweight='medium')
    ax1.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Comprehensive Modality Ablation (mean pooling, nt_v2_500m, esm1b_650m, Pure_XGBoost_Concat)',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylim(0.35, 0.95)
    ax1.legend(loc='upper left', ncol=5, frameon=True, fontsize=10)

    # Cumulative Addition
    ax2 = fig.add_subplot(gs[1, 0])
    x_steps = np.arange(len(prog_keys))
    for ds in datasets:
        vals = [pivot_mcc.loc[k, ds] for k in prog_keys]
        ax2.plot(x_steps, vals, marker='s', linewidth=2, label=ds, color=BENCHMARK_COLORS[ds], alpha=0.85)
    
    avg_vals = [pivot_mcc.loc[k, 'Average'] for k in prog_keys]
    ax2.plot(x_steps, avg_vals, marker='D', linewidth=3, color='#e7298a', label='Average', zorder=6)
    
    for idx, (x, y) in enumerate(zip(x_steps, avg_vals)):
        if idx == 0:
            ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=9, fontweight='bold', color='#e7298a')
        else:
            delta = y - avg_vals[idx - 1]
            sign = "+" if delta >= 0 else ""
            ax2.annotate(f"{y:.4f}\n({sign}{delta:.4f})", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8.5, fontweight='bold', color='#e7298a')

    ax2.set_xticks(x_steps)
    ax2.set_xticklabels(prog_labels, fontsize=10, fontweight='medium')
    ax2.set_ylabel('MCC', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Cumulative Modality Addition (Progression)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0.40, 0.95)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)

    # Leave-One-Out
    ax3 = fig.add_subplot(gs[1, 1])
    loo_labels = [p[0] for p in loo_pairs]
    x_loo = np.arange(len(loo_pairs))

    for i, ds in enumerate(datasets):
        full_val = pivot_mcc.loc['full_best_geom', ds]
        drops = [full_val - pivot_mcc.loc[p[1], ds] for p in loo_pairs]
        ax3.bar(x_loo + (i - 1.5) * bar_width, drops, width=bar_width,
                label=ds, color=BENCHMARK_COLORS[ds], alpha=0.9, edgecolor='black', linewidth=0.5)

    full_avg = pivot_mcc.loc['full_best_geom', 'Average']
    avg_drops = [full_avg - pivot_mcc.loc[p[1], 'Average'] for p in loo_pairs]
    ax3.plot(x_loo, avg_drops, color='#d62728', marker='^', linewidth=2.5, markersize=8, label='Average Drop', zorder=5)

    for x, d in zip(x_loo, avg_drops):
        ax3.annotate(f"-{d:.4f}", (x, d), textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=8.5, fontweight='bold', color='#d62728')

    ax3.set_xticks(x_loo)
    ax3.set_xticklabels(loo_labels, fontsize=10, fontweight='medium')
    ax3.set_ylabel(r'MCC Drop when Omitted ($\Delta$ Drop)', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Leave-One-Out Impact (Drop Relative to Full Model)', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)

    plt.savefig(os.path.join(out_dir, '1_modality_ablation_panel.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, '1_modality_ablation_panel.pdf'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. Individual Per-Dataset Modality Ablation Figures (4 Figures)
    # -------------------------------------------------------------
    for ds in datasets:
        fig_ds = plt.figure(figsize=(16, 12))
        gs_ds = fig_ds.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.48, wspace=0.25)
        ds_color = BENCHMARK_COLORS[ds]

        # Top: Bar chart of all modalities sorted for this dataset
        ax_top = fig_ds.add_subplot(gs_ds[0, :])
        ds_series = pivot_mcc[ds].loc[order].sort_values(ascending=True)
        ds_keys = ds_series.index
        ds_labels = [label_map.get(k, k) for k in ds_keys]
        x_idx = np.arange(len(ds_keys))

        bars = ax_top.bar(x_idx, ds_series.values, color=ds_color, alpha=0.88, edgecolor='black', linewidth=0.6, width=0.65)
        ax_top.set_xticks(x_idx)
        ax_top.set_xticklabels(ds_labels, rotation=28, ha='right', fontsize=9.5, fontweight='medium')
        ax_top.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
        ax_top.set_title(f"(A) Modality Ablation - {ds} Benchmark", fontsize=13, fontweight='bold', pad=10)
        ax_top.set_ylim(min(ds_series.values) * 0.9, max(ds_series.values) * 1.05)

        for bar in bars:
            h = bar.get_height()
            ax_top.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                            textcoords="offset points", xytext=(0, 4), ha='center',
                            fontsize=8.5, fontweight='bold')

        # Bottom Left: Cumulative Stepwise Addition on this dataset
        ax_bl = fig_ds.add_subplot(gs_ds[1, 0])
        ds_prog_vals = [pivot_mcc.loc[k, ds] for k in prog_keys]
        ax_bl.plot(x_steps, ds_prog_vals, marker='o', markersize=8, linewidth=2.8, color=ds_color, zorder=4)

        for idx, (x, y) in enumerate(zip(x_steps, ds_prog_vals)):
            if idx == 0:
                ax_bl.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=9.5, fontweight='bold', color=ds_color)
            else:
                delta = y - ds_prog_vals[idx - 1]
                sign = "+" if delta >= 0 else ""
                ax_bl.annotate(f"{y:.4f}\n({sign}{delta:.4f})", (x, y), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=ds_color)

        ax_bl.set_xticks(x_steps)
        ax_bl.set_xticklabels(prog_labels, fontsize=10, fontweight='medium')
        ax_bl.set_ylabel('MCC', fontsize=11, fontweight='bold')
        ax_bl.set_title(f"(B) Cumulative Addition Progression ({ds})", fontsize=12, fontweight='bold', pad=10)
        ax_bl.set_ylim(min(ds_prog_vals) * 0.92, max(ds_prog_vals) * 1.08)

        # Bottom Right: Leave-One-Out Drop on this dataset
        ax_br = fig_ds.add_subplot(gs_ds[1, 1])
        full_ds_val = pivot_mcc.loc['full_best_geom', ds]
        ds_drops = [full_ds_val - pivot_mcc.loc[p[1], ds] for p in loo_pairs]
        bars_drop = ax_br.bar(x_loo, ds_drops, width=0.55, color=ds_color, alpha=0.88, edgecolor='black', linewidth=0.6)

        for bar, d in zip(bars_drop, ds_drops):
            sign = "-" if d >= 0 else "+"
            val_txt = f"{sign}{abs(d):.4f}"
            y_pos = bar.get_height()
            offset = 4 if y_pos >= 0 else -12
            ax_br.annotate(val_txt, (bar.get_x() + bar.get_width() / 2, y_pos),
                           textcoords="offset points", xytext=(0, offset), ha='center',
                           fontsize=9, fontweight='bold', color='#d62728' if d > 0 else '#2ca02c')

        ax_br.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax_br.set_xticks(x_loo)
        ax_br.set_xticklabels(loo_labels, fontsize=10, fontweight='medium')
        ax_br.set_ylabel(r'MCC Drop when Omitted ($\Delta$ Drop)', fontsize=11, fontweight='bold')
        ax_br.set_title(f"(C) Leave-One-Out Drop ({ds})", fontsize=12, fontweight='bold', pad=10)

        out_png = os.path.join(out_dir, f'1_modality_ablation_{ds}.png')
        out_pdf = os.path.join(out_dir, f'1_modality_ablation_{ds}.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_png}")


# ==============================================================================
# 2. GEOMETRIC FEATURES ABLATION (batch_run_20260929_1740)
# ==============================================================================
def run_geom_features_ablation(csv_path, out_dir):
    print("\n[2/4] Processing Geometric Features Ablation (1023 Combinations)...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    ours = df[(df['Source'] == 'OURS') & (df['Dataset'].isin(DATASET_NAME_MAP.keys()))].copy()
    ours['Clean_Dataset'] = ours['Dataset'].map(DATASET_NAME_MAP)

    all_geom_features = [
        'dna_LID', 'dna_LLR', 'dna_LVD_Cosine', 'dna_LVD_L2', 'dna_LVD_Relative',
        'prot_LID', 'prot_LLR', 'prot_LVD_Cosine', 'prot_LVD_L2', 'prot_LVD_Relative'
    ]
    datasets = ['ClinVarHQ', 'ProteinGym', 'Test', 'UniProt']
    full_set = set(all_geom_features)

    ours['feat_set'] = ours['Geom_Features'].apply(lambda x: set(x.split('+')) if pd.notna(x) else set())
    ours['feat_count'] = ours['feat_set'].apply(len)

    # -------------------------------------------------------------
    # 1. Global Overview 3-Panel Figure
    # -------------------------------------------------------------
    marginal_results = []
    for feat in all_geom_features:
        row = {'Feature': feat, 'Modality': 'DNA' if feat.startswith('dna') else 'Protein'}
        for ds in datasets:
            sub = ours[ours['Clean_Dataset'] == ds]
            present_mcc = sub[sub['feat_set'].apply(lambda s: feat in s)]['MCC'].mean()
            absent_mcc = sub[sub['feat_set'].apply(lambda s: feat not in s)]['MCC'].mean()
            row[ds] = present_mcc - absent_mcc
        row['Average_Delta'] = np.mean([row[ds] for ds in datasets])
        marginal_results.append(row)
    df_marginal = pd.DataFrame(marginal_results).sort_values(by='Average_Delta', ascending=True)

    single_loo_results = []
    for feat in all_geom_features:
        row = {'Feature': feat, 'Modality': 'DNA' if feat.startswith('dna') else 'Protein'}
        loo_target = full_set - {feat}
        for ds in datasets:
            sub = ours[ours['Clean_Dataset'] == ds]
            single_r = sub[sub['feat_set'] == {feat}]
            row[f'{ds}_single'] = single_r['MCC'].values[0] if not single_r.empty else np.nan
            full_r = sub[sub['feat_set'] == full_set]
            row[f'{ds}_full'] = full_r['MCC'].values[0] if not full_r.empty else np.nan
        row['Avg_Single'] = np.mean([row[f'{ds}_single'] for ds in datasets])
        row['Avg_Full'] = np.mean([row[f'{ds}_full'] for ds in datasets])
        single_loo_results.append(row)
    df_single_sorted = pd.DataFrame(single_loo_results).sort_values(by='Avg_Single', ascending=True)

    top_k_counts = {feat: {'Top20': 0, 'Top50': 0} for feat in all_geom_features}
    for ds in datasets:
        sub = ours[ours['Clean_Dataset'] == ds].sort_values(by='MCC', ascending=False)
        top20 = sub.iloc[:20]
        top50 = sub.iloc[:50]
        for feat in all_geom_features:
            top_k_counts[feat]['Top20'] += top20['feat_set'].apply(lambda s: feat in s).sum() / (20 * len(datasets)) * 100
            top_k_counts[feat]['Top50'] += top50['feat_set'].apply(lambda s: feat in s).sum() / (50 * len(datasets)) * 100
    df_top_k = pd.DataFrame([
        {'Feature': feat, 'Top20_Pct': top_k_counts[feat]['Top20'], 'Top50_Pct': top_k_counts[feat]['Top50'],
         'Modality': 'DNA' if feat.startswith('dna') else 'Protein'}
        for feat in all_geom_features
    ]).sort_values(by='Top20_Pct', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
    plt.subplots_adjust(wspace=0.34)

    # Panel A: Marginal
    ax1 = axes[0]
    y_pos = np.arange(len(df_marginal))
    colors1 = ['#2b5c8f' if m == 'DNA' else '#e7298a' for m in df_marginal['Modality']]
    bars1 = ax1.barh(y_pos, df_marginal['Average_Delta'], color=colors1, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1.2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_marginal['Feature'], fontsize=10.5, fontweight='medium')
    ax1.set_xlabel(r'Marginal $\Delta$ MCC (Included vs Excluded)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Marginal Contribution of Geom Features\n(Across all 1,023 configurations)', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.0060, 0.0210)
    for bar in bars1:
        w = bar.get_width()
        sign = "+" if w >= 0 else ""
        offset = 0.0004 if w >= 0 else -0.0004
        ha = 'left' if w >= 0 else 'right'
        ax1.annotate(f"{sign}{w:.4f}", (w + offset, bar.get_y() + bar.get_height() / 2),
                     va='center', ha=ha, fontsize=8.5, fontweight='bold')

    custom_lines = [
        patches.Patch(facecolor='#2b5c8f', label='DNA Geom Feature'),
        patches.Patch(facecolor='#e7298a', label='Protein Geom Feature')
    ]
    ax1.legend(handles=custom_lines, loc='lower right', frameon=True, fontsize=9.5)

    # Panel B: Standalone
    ax2 = axes[1]
    y_pos2 = np.arange(len(df_single_sorted))
    colors2 = ['#2b5c8f' if m == 'DNA' else '#e7298a' for m in df_single_sorted['Modality']]
    bars2 = ax2.barh(y_pos2, df_single_sorted['Avg_Single'], color=colors2, alpha=0.85, edgecolor='black', linewidth=0.5)
    full_avg_val = df_single_sorted['Avg_Full'].iloc[0]
    ax2.axvline(full_avg_val, color='#d62728', linestyle='--', linewidth=1.8, label=f'Full 10-Geom ({full_avg_val:.4f})')
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(df_single_sorted['Feature'], fontsize=10.5, fontweight='medium')
    ax2.set_xlabel('Standalone MCC (Single Feature Alone)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Standalone Feature Performance\n(Single Feature Alone vs Full 10-Feature Ensemble)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True, fontsize=9.5)
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.4f}", (w - 0.015, bar.get_y() + bar.get_height() / 2),
                     va='center', ha='right', fontsize=8.5, fontweight='bold', color='white')

    # Panel C: Top-K
    ax3 = axes[2]
    y_pos3 = np.arange(len(df_top_k))
    bar_height = 0.38
    b_top20 = ax3.barh(y_pos3 + bar_height / 2, df_top_k['Top20_Pct'], height=bar_height,
                      color='#2ca02c', label='Top 20 Models', edgecolor='black', linewidth=0.5, alpha=0.85)
    b_top50 = ax3.barh(y_pos3 - bar_height / 2, df_top_k['Top50_Pct'], height=bar_height,
                      color='#1f77b4', label='Top 50 Models', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax3.set_yticks(y_pos3)
    ax3.set_yticklabels(df_top_k['Feature'], fontsize=10.5, fontweight='medium')
    ax3.set_xlabel('Frequency in Top Configurations (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Geometric Feature Selection Frequency\n(Prevalence among Top-Performing Models)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 105)
    ax3.legend(loc='lower right', frameon=True, fontsize=9.5)
    for bar in b_top20:
        w = bar.get_width()
        ax3.annotate(f"{w:.1f}%", (w + 1.2, bar.get_y() + bar.get_height() / 2),
                     va='center', ha='left', fontsize=8, fontweight='bold', color='#2ca02c')

    plt.savefig(os.path.join(out_dir, '2_geom_features_ablation_panel.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, '2_geom_features_ablation_panel.pdf'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. Individual Per-Dataset Geometric Features Figures (4 Figures)
    # -------------------------------------------------------------
    for ds in datasets:
        sub_ds = ours[ours['Clean_Dataset'] == ds].copy()

        # Marginal for this dataset
        ds_marginal = []
        for feat in all_geom_features:
            p_mcc = sub_ds[sub_ds['feat_set'].apply(lambda s: feat in s)]['MCC'].mean()
            a_mcc = sub_ds[sub_ds['feat_set'].apply(lambda s: feat not in s)]['MCC'].mean()
            ds_marginal.append({
                'Feature': feat,
                'Delta': p_mcc - a_mcc,
                'Modality': 'DNA' if feat.startswith('dna') else 'Protein'
            })
        df_ds_marg = pd.DataFrame(ds_marginal).sort_values(by='Delta', ascending=True)

        # Standalone for this dataset
        ds_standalone = []
        full_row = sub_ds[sub_ds['feat_set'] == full_set]
        full_ds_mcc = full_row['MCC'].values[0] if not full_row.empty else np.nan
        for feat in all_geom_features:
            single_r = sub_ds[sub_ds['feat_set'] == {feat}]
            single_v = single_r['MCC'].values[0] if not single_r.empty else np.nan
            ds_standalone.append({
                'Feature': feat,
                'Single_MCC': single_v,
                'Modality': 'DNA' if feat.startswith('dna') else 'Protein'
            })
        df_ds_stand = pd.DataFrame(ds_standalone).sort_values(by='Single_MCC', ascending=True)

        # Top K for this dataset
        sub_sorted = sub_ds.sort_values(by='MCC', ascending=False)
        top20_ds = sub_sorted.iloc[:20]
        top50_ds = sub_sorted.iloc[:50]
        ds_top = []
        for feat in all_geom_features:
            t20_pct = top20_ds['feat_set'].apply(lambda s: feat in s).sum() / 20 * 100
            t50_pct = top50_ds['feat_set'].apply(lambda s: feat in s).sum() / 50 * 100
            ds_top.append({
                'Feature': feat,
                'Top20': t20_pct,
                'Top50': t50_pct,
                'Modality': 'DNA' if feat.startswith('dna') else 'Protein'
            })
        df_ds_top = pd.DataFrame(ds_top).sort_values(by='Top20', ascending=True)

        # 3-Panel Figure for this dataset
        fig_d, ax_arr = plt.subplots(1, 3, figsize=(20, 7.5))
        plt.subplots_adjust(wspace=0.34)

        # Panel A
        y1 = np.arange(len(df_ds_marg))
        c1 = ['#2b5c8f' if m == 'DNA' else '#e7298a' for m in df_ds_marg['Modality']]
        b1 = ax_arr[0].barh(y1, df_ds_marg['Delta'], color=c1, alpha=0.88, edgecolor='black', linewidth=0.5)
        ax_arr[0].axvline(0, color='gray', linestyle='--', linewidth=1.2)
        ax_arr[0].set_yticks(y1)
        ax_arr[0].set_yticklabels(df_ds_marg['Feature'], fontsize=10.5, fontweight='medium')
        ax_arr[0].set_xlabel(r'Marginal $\Delta$ MCC (Included vs Excluded)', fontsize=11, fontweight='bold')
        ax_arr[0].set_title(f'(A) Marginal Contribution of Geom Features\n({ds} Benchmark)', fontsize=12, fontweight='bold')
        
        # Set dynamic xlim
        min_d = df_ds_marg['Delta'].min()
        max_d = df_ds_marg['Delta'].max()
        ax_arr[0].set_xlim(min(min_d * 1.35, -0.005), max(max_d * 1.3, 0.015))
        for bar in b1:
            w = bar.get_width()
            sign = "+" if w >= 0 else ""
            offset = 0.0004 if w >= 0 else -0.0004
            ha = 'left' if w >= 0 else 'right'
            ax_arr[0].annotate(f"{sign}{w:.4f}", (w + offset, bar.get_y() + bar.get_height() / 2),
                               va='center', ha=ha, fontsize=8.5, fontweight='bold')
        ax_arr[0].legend(handles=custom_lines, loc='lower right', frameon=True, fontsize=9.5)

        # Panel B
        y2 = np.arange(len(df_ds_stand))
        c2 = ['#2b5c8f' if m == 'DNA' else '#e7298a' for m in df_ds_stand['Modality']]
        b2 = ax_arr[1].barh(y2, df_ds_stand['Single_MCC'], color=c2, alpha=0.88, edgecolor='black', linewidth=0.5)
        ax_arr[1].axvline(full_ds_mcc, color='#d62728', linestyle='--', linewidth=1.8, label=f'Full 10-Geom ({full_ds_mcc:.4f})')
        ax_arr[1].set_yticks(y2)
        ax_arr[1].set_yticklabels(df_ds_stand['Feature'], fontsize=10.5, fontweight='medium')
        ax_arr[1].set_xlabel('Standalone MCC (Single Feature Alone)', fontsize=11, fontweight='bold')
        ax_arr[1].set_title(f'(B) Standalone Feature Performance\n({ds} Benchmark)', fontsize=12, fontweight='bold')
        ax_arr[1].legend(loc='lower right', frameon=True, fontsize=9.5)
        for bar in b2:
            w = bar.get_width()
            ax_arr[1].annotate(f"{w:.4f}", (w - 0.015, bar.get_y() + bar.get_height() / 2),
                               va='center', ha='right', fontsize=8.5, fontweight='bold', color='white')

        # Panel C
        y3 = np.arange(len(df_ds_top))
        bh = 0.38
        b_t20 = ax_arr[2].barh(y3 + bh / 2, df_ds_top['Top20'], height=bh,
                               color='#2ca02c', label='Top 20 Models', edgecolor='black', linewidth=0.5, alpha=0.85)
        b_t50 = ax_arr[2].barh(y3 - bh / 2, df_ds_top['Top50'], height=bh,
                               color='#1f77b4', label='Top 50 Models', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax_arr[2].set_yticks(y3)
        ax_arr[2].set_yticklabels(df_ds_top['Feature'], fontsize=10.5, fontweight='medium')
        ax_arr[2].set_xlabel('Frequency in Top Configurations (%)', fontsize=11, fontweight='bold')
        ax_arr[2].set_title(f'(C) Feature Selection Frequency\n({ds} Benchmark)', fontsize=12, fontweight='bold')
        ax_arr[2].set_xlim(0, 105)
        ax_arr[2].legend(loc='lower right', frameon=True, fontsize=9.5)
        for bar in b_t20:
            w = bar.get_width()
            ax_arr[2].annotate(f"{w:.1f}%", (w + 1.2, bar.get_y() + bar.get_height() / 2),
                               va='center', ha='left', fontsize=8, fontweight='bold', color='#2ca02c')

        out_png = os.path.join(out_dir, f'2_geom_ablation_{ds}.png')
        out_pdf = os.path.join(out_dir, f'2_geom_ablation_{ds}.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_png}")


# ==============================================================================
# 3. LID ABLATION & ARCHITECTURES (batch_run_20260930_lid_ablation)
# ==============================================================================
def run_lid_ablation(csv_path, out_dir):
    print("\n[3/4] Processing LID Ablation & Architecture Comparison...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    df['Clean_Dataset'] = df['Dataset'].map(DATASET_NAME_MAP)
    datasets = ['ClinVarHQ', 'ProteinGym', 'Test', 'UniProt']

    k_values = [8, 16, 32, 64]
    lid_types = ['dna', 'prot', 'both']
    lid_labels = {'dna': 'DNA LID', 'prot': 'Protein LID', 'both': 'Both (DNA + Protein) LID'}
    lid_colors = {'dna': '#2b5c8f', 'prot': '#e7298a', 'both': '#2ca02c'}
    lid_markers = {'dna': 'o', 'prot': 's', 'both': '^'}

    arch_rename = {
        'Pure_XGBoost_Concat': 'Pure XGBoost (Concat)',
        'Hybrid_Transformer_XGBoost': 'Hybrid Transformer-XGBoost',
        'Hybrid_Gating_XGBoost': 'Hybrid Gating-XGBoost',
        'Hybrid_Concat_XGBoost': 'Hybrid Concat-XGBoost',
        'Hybrid_CrossAttn_XGBoost': 'Hybrid CrossAttn-XGBoost',
        'PyTorch_CrossAttn': 'PyTorch Cross-Attention',
        'PyTorch_Transformer': 'PyTorch Transformer',
        'PyTorch_Concat': 'PyTorch Concat MLP',
        'PyTorch_Gating': 'PyTorch Gating Net'
    }

    df_xgb = df[df['Network'] == 'Pure_XGBoost_Concat'].copy()
    df_arch = df[df['Ablation'].str.contains('architecture')].copy()
    df_arch['Clean_Network'] = df_arch['Network'].map(lambda x: arch_rename.get(x, x))

    # -------------------------------------------------------------
    # 1. Global Overview Plots (2x2 Trend + Grouped Bar Architecture)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df_xgb[df_xgb['Clean_Dataset'] == ds]
        ctrl_row = sub[sub['Ablation'] == 'bio_core_dna_prot_spliceai_geom_lid_control']
        ctrl_mcc = ctrl_row['MCC'].values[0] if not ctrl_row.empty else np.nan
        ax.axhline(ctrl_mcc, color='#d62728', linestyle='--', linewidth=1.8,
                   label=f'Control: No LID ({ctrl_mcc:.4f})', zorder=2)

        for l_type in lid_types:
            mcc_curve = []
            for k in k_values:
                target_ablation = f'bio_core_dna_prot_spliceai_geom_lid_{l_type}_k{k}'
                r = sub[sub['Ablation'] == target_ablation]
                val = r['MCC'].values[0] if not r.empty else np.nan
                mcc_curve.append(val)

            ax.plot(k_values, mcc_curve, marker=lid_markers[l_type], color=lid_colors[l_type],
                    linewidth=2.2, markersize=8, label=lid_labels[l_type], zorder=4)

            if not np.isnan(mcc_curve).all():
                max_idx = int(np.nanargmax(mcc_curve))
                max_val = mcc_curve[max_idx]
                ax.annotate(f"{max_val:.4f}", (k_values[max_idx], max_val),
                            textcoords="offset points", xytext=(0, 7), ha='center',
                            fontsize=8.5, fontweight='bold', color=lid_colors[l_type])

        ax.set_title(f"{ds} Benchmark", fontsize=12.5, fontweight='bold', pad=8)
        ax.set_xlabel('Neighborhood Size $k$', fontsize=11, fontweight='bold')
        ax.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=11, fontweight='bold')
        ax.set_xticks(k_values)
        ax.set_xticklabels([f"k={k}" for k in k_values], fontsize=10)
        ax.legend(loc='lower right', frameon=True, fontsize=9.5)

    fig.suptitle('(A) Local Intrinsic Dimensionality (LID) Ablation across Neighborhood Sizes $k$',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_lid_k_trend_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, '3_lid_k_trend_curves.pdf'), bbox_inches='tight')
    plt.close()

    # Architecture Overview
    pivot_arch = df_arch.pivot(index='Clean_Network', columns='Clean_Dataset', values='MCC')
    pivot_arch['Average'] = pivot_arch.mean(axis=1)
    pivot_arch = pivot_arch.sort_values(by='Average', ascending=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    y_pos = np.arange(len(pivot_arch))
    bar_height = 0.18
    for idx, ds in enumerate(datasets):
        vals = pivot_arch[ds]
        ax.barh(y_pos + (idx - 1.5) * bar_height, vals, height=bar_height,
                label=ds, color=BENCHMARK_COLORS[ds], alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.plot(pivot_arch['Average'], y_pos, color='#d62728', marker='D', linewidth=2.5,
            markersize=8, label='Average (All Datasets)', zorder=6)
    for y, avg_v in zip(y_pos, pivot_arch['Average']):
        ax.annotate(f"{avg_v:.4f}", (avg_v + 0.010, y), va='center',
                    fontsize=9, fontweight='bold', color='#d62728')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot_arch.index, fontsize=10.5, fontweight='medium')
    ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
    ax.set_title('(B) Neural vs Hybrid vs Pure Architectures with LID Feature (DNA k=32)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0.35, 0.96)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.savefig(os.path.join(out_dir, '3_lid_architecture_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, '3_lid_architecture_comparison.pdf'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. Individual Per-Dataset LID Figures (4 Figures)
    # -------------------------------------------------------------
    for ds in datasets:
        fig_d, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7.5))
        plt.subplots_adjust(wspace=0.28)
        ds_col = BENCHMARK_COLORS[ds]

        # Left: LID Trend Curves on this dataset
        sub_xgb = df_xgb[df_xgb['Clean_Dataset'] == ds]
        ctrl_r = sub_xgb[sub_xgb['Ablation'] == 'bio_core_dna_prot_spliceai_geom_lid_control']
        c_mcc = ctrl_r['MCC'].values[0] if not ctrl_r.empty else np.nan
        ax_l.axhline(c_mcc, color='#d62728', linestyle='--', linewidth=2.0,
                     label=f'Control: No LID ({c_mcc:.4f})', zorder=2)

        for l_type in lid_types:
            curve = []
            for k in k_values:
                t_ab = f'bio_core_dna_prot_spliceai_geom_lid_{l_type}_k{k}'
                row_k = sub_xgb[sub_xgb['Ablation'] == t_ab]
                curve.append(row_k['MCC'].values[0] if not row_k.empty else np.nan)

            ax_l.plot(k_values, curve, marker=lid_markers[l_type], color=lid_colors[l_type],
                      linewidth=2.5, markersize=8.5, label=lid_labels[l_type], zorder=4)

            if not np.isnan(curve).all():
                m_idx = int(np.nanargmax(curve))
                m_val = curve[m_idx]
                ax_l.annotate(f"{m_val:.4f}", (k_values[m_idx], m_val),
                              textcoords="offset points", xytext=(0, 8), ha='center',
                              fontsize=9, fontweight='bold', color=lid_colors[l_type])

        ax_l.set_title(f"(A) LID Neighborhood Size k Ablation ({ds})", fontsize=12.5, fontweight='bold', pad=10)
        ax_l.set_xlabel('Neighborhood Size $k$', fontsize=11, fontweight='bold')
        ax_l.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=11, fontweight='bold')
        ax_l.set_xticks(k_values)
        ax_l.set_xticklabels([f"k={k}" for k in k_values], fontsize=10.5)
        ax_l.legend(loc='lower right', frameon=True, fontsize=10)

        # Right: Architecture Comparison on this dataset
        sub_arch = df_arch[df_arch['Clean_Dataset'] == ds].sort_values(by='MCC', ascending=True)
        y_pos_arch = np.arange(len(sub_arch))
        bars_arch = ax_r.barh(y_pos_arch, sub_arch['MCC'], color=ds_col, alpha=0.88, edgecolor='black', linewidth=0.6, height=0.62)

        ax_r.set_yticks(y_pos_arch)
        ax_r.set_yticklabels(sub_arch['Clean_Network'], fontsize=10.5, fontweight='medium')
        ax_r.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=11, fontweight='bold')
        ax_r.set_title(f"(B) Neural vs Hybrid vs Pure Architectures ({ds} - DNA k=32)", fontsize=12.5, fontweight='bold', pad=10)
        ax_r.set_xlim(min(sub_arch['MCC']) * 0.9, max(sub_arch['MCC']) * 1.08)

        for bar in bars_arch:
            w = bar.get_width()
            ax_r.annotate(f"{w:.4f}", (w + 0.008, bar.get_y() + bar.get_height() / 2),
                          va='center', fontsize=9, fontweight='bold', color=ds_col)

        out_png = os.path.join(out_dir, f'3_lid_ablation_{ds}.png')
        out_pdf = os.path.join(out_dir, f'3_lid_ablation_{ds}.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_png}")


# ==============================================================================
# 4. SEQUENCE LENGTH GRID ABLATION (sequence_ablation_20261001_1206)
# ==============================================================================
def run_sequence_length_ablation(csv_path, out_dir):
    print("\n[4/4] Processing Sequence Length Grid Ablation (5x5 Heatmaps)...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    df['dna_len'] = df['DNA_Model'].apply(lambda x: int(x.split('_dna_')[-1]))
    df['prot_len'] = df['Prot_Model'].apply(lambda x: int(x.split('_prot_')[-1]))

    def clean_ds(name):
        lower = name.lower()
        if 'clinvar' in lower:
            return 'ClinVarHQ'
        elif 'protein' in lower:
            return 'ProteinGym'
        elif 'uniprot' in lower:
            return 'UniProt'
        elif 'test' in lower:
            return 'Test'
        return name

    df['Clean_Dataset'] = df['Dataset'].apply(clean_ds)
    benchmarks = ['ClinVarHQ', 'ProteinGym', 'Test', 'UniProt']
    dna_order = [301, 601, 1001, 2001, 10001]
    prot_order = [31, 101, 201, 501, 1001]

    # -------------------------------------------------------------
    # 1. Global Overview 2x2 Subplots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    axes = axes.flatten()

    for idx, bmark in enumerate(benchmarks):
        ax = axes[idx]
        sub = df[df['Clean_Dataset'] == bmark]
        grid = sub.groupby(['dna_len', 'prot_len'])['MCC'].mean().unstack(level='prot_len')
        grid = grid.reindex(index=dna_order, columns=prot_order)

        sns.heatmap(grid, ax=ax, annot=True, fmt=".4f", cmap="YlGnBu",
                    cbar_kws={'label': 'MCC'}, annot_kws={"size": 10, "weight": "bold"},
                    linewidths=1.0, linecolor='white')

        max_val = grid.values.max()
        max_idx = np.unravel_index(np.argmax(grid.values), grid.values.shape)
        rect = patches.Rectangle((max_idx[1], max_idx[0]), 1, 1, fill=False,
                                 edgecolor='#d62728', lw=3.0, zorder=5)
        ax.add_patch(rect)

        best_dna = grid.index[max_idx[0]]
        best_prot = grid.columns[max_idx[1]]
        ax.set_title(f"{bmark} (Optimal: DNA={best_dna} bp, Prot={best_prot} aa | MCC={max_val:.4f})",
                     fontsize=11.5, fontweight='bold', pad=10)
        ax.set_xlabel('Protein Sequence Context Length (aa)', fontsize=10.5, fontweight='bold')
        ax.set_ylabel('DNA Sequence Context Length (bp)', fontsize=10.5, fontweight='bold')

    fig.suptitle('(A) Sequence Context Length Ablation across 4 Clinical Benchmarks (Averaged over 3-fold CV)',
                 fontsize=13.5, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, '4_seq_length_heatmaps_subplots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, '4_seq_length_heatmaps_subplots.pdf'), bbox_inches='tight')
    plt.close()

    # Grand Average Heatmap
    grand_grid = df.groupby(['dna_len', 'prot_len'])['MCC'].mean().unstack(level='prot_len')
    grand_grid = grand_grid.reindex(index=dna_order, columns=prot_order)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(grand_grid, ax=ax, annot=True, fmt=".4f", cmap="YlGnBu",
                cbar_kws={'label': 'Grand Average MCC'}, annot_kws={"size": 12, "weight": "bold"},
                linewidths=1.5, linecolor='white')

    max_val_grand = grand_grid.values.max()
    max_idx_grand = np.unravel_index(np.argmax(grand_grid.values), grand_grid.values.shape)
    rect = patches.Rectangle((max_idx_grand[1], max_idx_grand[0]), 1, 1, fill=False,
                             edgecolor='#d62728', lw=3.5, zorder=5)
    ax.add_patch(rect)

    best_dna_g = grand_grid.index[max_idx_grand[0]]
    best_prot_g = grand_grid.columns[max_idx_grand[1]]
    baseline_val = grand_grid.loc[301, 31]
    rel_gain = ((max_val_grand - baseline_val) / baseline_val) * 100

    ax.set_title(f"(B) Grand Average Sequence Context Length Grid\n"
                 f"Optimal: DNA = {best_dna_g} bp, Protein = {best_prot_g} aa (MCC = {max_val_grand:.4f}, +{rel_gain:.2f}% vs shortest)",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Protein Sequence Context Length (aa)', fontsize=11, fontweight='bold')
    ax.set_ylabel('DNA Sequence Context Length (bp)', fontsize=11, fontweight='bold')

    plt.savefig(os.path.join(out_dir, '4_seq_length_heatmap_grand_average.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, '4_seq_length_heatmap_grand_average.pdf'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. Individual Per-Dataset Heatmap Figures (4 Figures)
    # -------------------------------------------------------------
    for bmark in benchmarks:
        sub_b = df[df['Clean_Dataset'] == bmark]
        grid_b = sub_b.groupby(['dna_len', 'prot_len'])['MCC'].mean().unstack(level='prot_len')
        grid_b = grid_b.reindex(index=dna_order, columns=prot_order)

        fig_b, ax_b = plt.subplots(figsize=(9.5, 8))
        sns.heatmap(grid_b, ax=ax_b, annot=True, fmt=".4f", cmap="YlGnBu",
                    cbar_kws={'label': 'Matthews Correlation Coefficient (MCC)'},
                    annot_kws={"size": 12.5, "weight": "bold"},
                    linewidths=1.5, linecolor='white')

        max_v = grid_b.values.max()
        m_idx = np.unravel_index(np.argmax(grid_b.values), grid_b.values.shape)
        rect_b = patches.Rectangle((m_idx[1], m_idx[0]), 1, 1, fill=False,
                                   edgecolor='#d62728', lw=3.8, zorder=5)
        ax_b.add_patch(rect_b)

        b_dna = grid_b.index[m_idx[0]]
        b_prot = grid_b.columns[m_idx[1]]
        base_v = grid_b.loc[301, 31]
        gain_pct = ((max_v - base_v) / base_v) * 100

        ax_b.set_title(f"Sequence Context Length Grid - {bmark}\n"
                       f"Optimal: DNA = {b_dna} bp, Protein = {b_prot} aa (MCC = {max_v:.4f}, +{gain_pct:.2f}% vs shortest)",
                       fontsize=12.5, fontweight='bold', pad=14)
        ax_b.set_xlabel('Protein Sequence Context Length (aa)', fontsize=11.5, fontweight='bold')
        ax_b.set_ylabel('DNA Sequence Context Length (bp)', fontsize=11.5, fontweight='bold')

        out_png = os.path.join(out_dir, f'4_seq_length_heatmap_{bmark}.png')
        out_pdf = os.path.join(out_dir, f'4_seq_length_heatmap_{bmark}.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_png}")


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Ablation Visualizations.")
    parser.add_argument('--all', action='store_true', default=True, help="Run all 4 ablations.")
    parser.add_argument('--out_dir', type=str, default='experiments/visualizations_ablation',
                        help="Output directory for generated plots.")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Target Visualization Directory: {os.path.abspath(out_dir)}")

    # 1. Modality Ablation
    f1 = 'experiments/batch_run_20260904_1800/global_compare_ours_vs_sota.csv'
    run_modality_ablation(f1, out_dir)

    # 2. Geometric Features Ablation
    f2 = 'experiments/batch_run_20260929_1740/global_compare_ours_vs_sota.csv'
    run_geom_features_ablation(f2, out_dir)

    # 3. LID Ablation
    f3 = 'experiments/batch_run_20260930_lid_ablation/global_leaderboard_metrics.csv'
    run_lid_ablation(f3, out_dir)

    # 4. Sequence Length Ablation
    f4 = 'experiments/sequence_ablation_20261001_1206/global_leaderboard_metrics.csv'
    run_sequence_length_ablation(f4, out_dir)

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"Results located at: {os.path.abspath(out_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()

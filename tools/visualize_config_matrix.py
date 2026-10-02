"""
tools/visualize_config_matrix.py
Generates the comprehensive Combinatorial Configuration Matrix & Taxonomy visualization
for the entire Multimodal Missense Variant Pathogenicity Prediction system.

Components included:
1. 3 Token Pooling strategies: Mean, Center, CLS
2. 5 DNA sequence context lengths: 301, 1001, 2001, 5001, 10001 bp
3. 5 Protein sequence context lengths: 31, 101, 251, 501, 1001 aa
4. 3 DNA Foundation Models: nt_v1_500m, nt_v2_500m, nt_v3_650m
5. 4 Protein Foundation Models: esm1b_650m, esm1v_650m, esm2_650m, esmc_600m
6. Bio features (11 features = 2 AF + 9 CS)
7. DNA Embed & Prot Embed (Continuous latent representations)
8. SpliceAI scores (9 functional splicing disruption scores)
9. Geometry features (10 manifold & information-theoretic scores)
10. 9 Multimodal Classification Architectures (PyTorch, Pure XGBoost, Hybrid)
All labels, text, annotations, and descriptions are strictly in English.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_config_matrix_visualization():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "visualizations_ablation"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Widescreen publication-grade canvas (26 x 17 inches)
    fig = plt.figure(figsize=(26, 17), dpi=300, facecolor="#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)  # Top-to-bottom coordinates
    ax.axis('off')
    
    # =========================================================================
    # HELPER: DRAW ROUNDED CARD WITH SHADOW AND HEADER
    # =========================================================================
    def draw_card(x, y, w, h, bg_color, border_color, title, title_color, badge_text="", badge_bg="#3B82F6"):
        # Drop shadow
        shadow = FancyBboxPatch((x + 0.3, y + 0.3), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc="#CBD5E1", ec="none", alpha=0.4, zorder=1)
        ax.add_patch(shadow)
        
        # Main card
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                              fc=bg_color, ec=border_color, lw=2.0, zorder=2)
        ax.add_patch(card)
        
        # Header banner
        header_h = 3.2
        header = FancyBboxPatch((x, y), w, header_h, 
                                boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=border_color, ec="none", zorder=3)
        ax.add_patch(header)
        rect_cover = patches.Rectangle((x, y + header_h - 1.2), w, 1.2, fc=border_color, ec="none", zorder=3)
        ax.add_patch(rect_cover)
        
        # Title text
        ax.text(x + 1.2, y + 1.6, title, fontsize=11.2, fontweight='bold', color=title_color, 
                va='center', zorder=4, fontfamily='sans-serif')
        
        # Badge
        if badge_text:
            badge_w = len(badge_text) * 0.65 + 1.5
            badge_box = FancyBboxPatch((x + w - badge_w - 1.2, y + 0.8), badge_w, 1.6,
                                       boxstyle="round,pad=0.1,rounding_size=0.4",
                                       fc=badge_bg, ec="none", zorder=4)
            ax.add_patch(badge_box)
            ax.text(x + w - 1.2 - badge_w / 2, y + 1.6, badge_text, fontsize=9.0, fontweight='bold',
                    color="#FFFFFF", ha='center', va='center', zorder=5)

    # =========================================================================
    # MAIN HEADER BANNER
    # =========================================================================
    top_banner = FancyBboxPatch((2, 1.5), 96, 6.2, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc="#0F172A", ec="#1E293B", lw=2, zorder=2)
    ax.add_patch(top_banner)
    
    ax.text(4, 3.7, "MULTIMODAL MISSENSE VARIANT PATHOGENICITY PREDICTION", 
            fontsize=19.5, fontweight='black', color="#38BDF8", va='center')
    ax.text(4, 5.8, "Comprehensive Combinatorial Configuration Matrix, Feature Taxonomy & Architecture Space", 
            fontsize=11.5, fontweight='medium', color="#E2E8F0", va='center')
    
    # Metadata badges on header right
    badges = [
        ("Combinatorial Space: >10^5 Paths", "#0284C7"),
        ("10 Geom Ablations: 1,023 Subsets", "#7C3AED"),
        ("Strict Metric: MCC", "#059669")
    ]
    cur_x = 96
    for b_text, b_col in reversed(badges):
        bw = len(b_text) * 0.52 + 1.8
        cur_x -= bw + 1.0
        bb = FancyBboxPatch((cur_x, 3.2), bw, 2.8, boxstyle="round,pad=0.15,rounding_size=0.4",
                            fc=b_col, ec="none", zorder=3)
        ax.add_patch(bb)
        ax.text(cur_x + bw/2, 4.6, b_text, fontsize=9.2, fontweight='bold', color="#FFFFFF",
                ha='center', va='center', zorder=4)

    # =========================================================================
    # MODULE 1: INPUT SEQUENCE CONTEXT WINDOWS (DNA & PROTEIN)
    # =========================================================================
    draw_card(2, 9.0, 29, 23.5, "#F0FDF4", "#15803D", 
              "1. SEQUENCE CONTEXT SPACE", "#FFFFFF", "5 x 5 = 25 Contexts", "#16A34A")
    
    # DNA Lengths Sub-box
    dna_box = FancyBboxPatch((3.2, 13.0), 26.6, 9.0, boxstyle="round,pad=0.15,rounding_size=0.5",
                             fc="#DCFCE7", ec="#86EFAC", lw=1.2, zorder=3)
    ax.add_patch(dna_box)
    ax.text(4.2, 14.5, "[DNA] Context Lengths (Centered on SNV):", fontsize=10.2, fontweight='bold', color="#14532D")
    
    dna_lengths = [
        ("301 bp", "Baseline local flanking genomic sequence"),
        ("1001 bp", "Proximal promoter & exon-intron boundaries"),
        ("2001 bp", "[*] Optimal genomic context window"),
        ("5001 bp", "Extended cis-regulatory / enhancer regions"),
        ("10001 bp", "Long-range chromatin interaction domain")
    ]
    for idx, (length, desc) in enumerate(dna_lengths):
        y_pos = 16.3 + idx * 1.3
        tag_bg = "#15803D" if "2001" in length else "#BBF7D0"
        tag_fg = "#FFFFFF" if "2001" in length else "#14532D"
        ax.text(4.5, y_pos, f"  {length:8s}", fontsize=9.0, fontweight='bold', color=tag_fg, 
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(10.8, y_pos, f": {desc}", fontsize=8.6, color="#166534")

    # Protein Lengths Sub-box
    prot_box = FancyBboxPatch((3.2, 22.8), 26.6, 9.0, boxstyle="round,pad=0.15,rounding_size=0.5",
                              fc="#DCFCE7", ec="#86EFAC", lw=1.2, zorder=3)
    ax.add_patch(prot_box)
    ax.text(4.2, 24.3, "[Protein] Context Lengths (Centered on Missense):", fontsize=10.2, fontweight='bold', color="#14532D")
    
    prot_lengths = [
        ("31 aa", "Immediate local peptide microenvironment"),
        ("101 aa", "Secondary structure & conserved motif window"),
        ("251 aa", "Domain-scale functional folding context"),
        ("501 aa", "Multi-domain functional interaction context"),
        ("1001 aa", "[*] Grand optimal full-length protein context")
    ]
    for idx, (length, desc) in enumerate(prot_lengths):
        y_pos = 26.1 + idx * 1.3
        tag_bg = "#15803D" if "1001" in length else "#BBF7D0"
        tag_fg = "#FFFFFF" if "1001" in length else "#14532D"
        ax.text(4.5, y_pos, f"  {length:8s}", fontsize=9.0, fontweight='bold', color=tag_fg,
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(10.8, y_pos, f": {desc}", fontsize=8.6, color="#166534")

    # =========================================================================
    # MODULE 2: FOUNDATION MODELS SPACE (3 DNA + 4 PROTEIN)
    # =========================================================================
    draw_card(32.5, 9.0, 33.5, 23.5, "#F0F9FF", "#0369A1", 
              "2. FOUNDATION MODELS", "#FFFFFF", "3 x 4 = 12 Model Pairs", "#0284C7")
    
    # 3 DNA Foundation Models
    d_model_box = FancyBboxPatch((33.7, 13.0), 31.1, 8.8, boxstyle="round,pad=0.15,rounding_size=0.5",
                                 fc="#E0F2FE", ec="#7DD3FC", lw=1.2, zorder=3)
    ax.add_patch(d_model_box)
    ax.text(34.7, 14.5, "[DNA Models] 3 Genomic Foundation Models:", fontsize=10.2, fontweight='bold', color="#0369A1")
    
    dna_models = [
        ("nt_v1_500m", "InstaDeepAI/nucleotide-transformer-500m-human-ref", "MLM, Human reference genome"),
        ("nt_v2_500m", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "[*] Multi-species SOTA DNA model"),
        ("nt_v3_650m", "InstaDeepAI/NTv3_650M_pre", "MLM, Next-Gen Transformer Architecture")
    ]
    for idx, (m_id, hf_path, notes) in enumerate(dna_models):
        y_pos = 16.5 + idx * 1.65
        is_opt = "nt_v2" in m_id
        tag_bg = "#0284C7" if is_opt else "#BAE6FD"
        tag_fg = "#FFFFFF" if is_opt else "#075985"
        ax.text(35.0, y_pos, f" 1.{idx+1} {m_id:11s}", fontsize=9.0, fontweight='bold', color=tag_fg,
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(45.0, y_pos, f"| {notes}", fontsize=8.4, color="#0C4A6E")

    # 4 Protein Foundation Models
    p_model_box = FancyBboxPatch((33.7, 22.5), 31.1, 9.3, boxstyle="round,pad=0.15,rounding_size=0.5",
                                 fc="#E0F2FE", ec="#7DD3FC", lw=1.2, zorder=3)
    ax.add_patch(p_model_box)
    ax.text(34.7, 23.9, "[Protein Models] 4 Protein Language Models (pLM):", fontsize=10.2, fontweight='bold', color="#0369A1")
    
    prot_models = [
        ("esm1b_650m", "facebook/esm1b_t33_650M_UR50S", "[*] Classic BERT pLM (UniRef50 Benchmark)"),
        ("esm1v_650m", "facebook/esm1v_t33_650M_UR90S_1", "Zero-shot variant effect 5-model ensemble"),
        ("esm2_650m", "facebook/esm2_t33_650M_UR50D", "Modern RoPE SOTA Evolutionary Scale Model"),
        ("esmc_600m", "EvolutionaryScale/esmc-600m", "Next-Gen ESM Cambrian Model (Autoregressive)")
    ]
    for idx, (m_id, hf_path, notes) in enumerate(prot_models):
        y_pos = 25.6 + idx * 1.45
        is_opt = "esm1b" in m_id
        tag_bg = "#0284C7" if is_opt else "#BAE6FD"
        tag_fg = "#FFFFFF" if is_opt else "#075985"
        ax.text(35.0, y_pos, f" 2.{idx+1} {m_id:11s}", fontsize=9.0, fontweight='bold', color=tag_fg,
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(45.0, y_pos, f"| {notes}", fontsize=8.4, color="#0C4A6E")

    # =========================================================================
    # MODULE 3: POOLING STRATEGIES & REPRESENTATION EMBEDDINGS
    # =========================================================================
    draw_card(67.5, 9.0, 30.5, 23.5, "#FAF5FF", "#7E22CE", 
              "3. POOLING & EMBEDDINGS", "#FFFFFF", "3 Pooling x 2 Embeds", "#9333EA")
    
    # 3 Pooling Strategies Box
    pool_box = FancyBboxPatch((68.7, 13.0), 28.1, 8.8, boxstyle="round,pad=0.15,rounding_size=0.5",
                              fc="#F3E8FF", ec="#D8B4FE", lw=1.2, zorder=3)
    ax.add_patch(pool_box)
    ax.text(69.7, 14.5, "[Pooling] 3 Token Aggregation Strategies:", fontsize=10.2, fontweight='bold', color="#6B21A8")
    
    poolings = [
        ("Mean Pooling", "Non-pad token average; captures broad global sequence context"),
        ("Center Pooling", "Direct token extraction at the exact missense mutation site"),
        ("CLS Pooling", "Sequence-level representation from leading [CLS]/Header token")
    ]
    for idx, (p_name, desc) in enumerate(poolings):
        y_pos = 16.5 + idx * 1.65
        is_mean = "Mean" in p_name
        tag_bg = "#9333EA" if is_mean else "#E9D5FF"
        tag_fg = "#FFFFFF" if is_mean else "#581C87"
        ax.text(70.0, y_pos, f" * {p_name:14s}", fontsize=9.0, fontweight='bold', color=tag_fg,
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(82.0, y_pos, f"-> {desc}", fontsize=8.2, color="#4C1D95")

    # Embeddings Box (DNA Embed + Prot Embed)
    embed_box = FancyBboxPatch((68.7, 22.5), 28.1, 9.3, boxstyle="round,pad=0.15,rounding_size=0.5",
                               fc="#F3E8FF", ec="#D8B4FE", lw=1.2, zorder=3)
    ax.add_patch(embed_box)
    ax.text(69.7, 23.9, "[Embeddings] Continuous Latent Shift Representations:", fontsize=10.2, fontweight='bold', color="#6B21A8")
    
    embed_items = [
        ("DNA Embed", "E_alt_dna - E_ref_dna (dim: 1024 / 1280)", "Nucleotide latent space displacement"),
        ("Prot Embed", "E_alt_prot - E_ref_prot (dim: 1280)", "Amino acid evolutionary space displacement"),
        ("Delta Vector", "Delta_E = E_alt - E_ref", "Preserves mutation directionality (Latent Shift)"),
    ]
    for idx, (e_name, dim_str, e_desc) in enumerate(embed_items):
        y_pos = 25.6 + idx * 1.45
        ax.text(70.0, y_pos, f" * {e_name:12s}", fontsize=8.8, fontweight='bold', color="#581C87",
                bbox=dict(boxstyle='round,pad=0.15', fc="#E9D5FF", ec='none'))
        ax.text(79.5, y_pos, f": {e_desc}", fontsize=8.2, color="#4C1D95")

    # =========================================================================
    # MODULE 4: BIOLOGICAL FEATURES (2 AF + 9 CS)
    # =========================================================================
    draw_card(2, 34.0, 30, 31.0, "#FFFBEB", "#B45309", 
              "4. BIOLOGICAL FEATURES", "#FFFFFF", "2 AF + 9 CS (11 Total)", "#D97706")
    
    # 2 Allele Frequency Features
    af_box = FancyBboxPatch((3.2, 38.0), 27.6, 6.0, boxstyle="round,pad=0.15,rounding_size=0.5",
                            fc="#FEF3C7", ec="#FCD34D", lw=1.2, zorder=3)
    ax.add_patch(af_box)
    ax.text(4.2, 39.5, "[AF] 2 Allele Frequency Features (Population Genetics):", fontsize=10.2, fontweight='bold', color="#92400E")
    
    af_features = [
        ("AF", "1000 Genomes Project global alternate allele frequency"),
        ("gnomADe_AF", "gnomAD Exome sequencing global population allele frequency")
    ]
    for idx, (f_name, f_desc) in enumerate(af_features):
        y_pos = 41.2 + idx * 1.45
        ax.text(4.5, y_pos, f" 1.{idx+1} {f_name:12s}", fontsize=9.0, fontweight='bold', color="#78350F",
                bbox=dict(boxstyle='round,pad=0.15', fc="#FDE68A", ec='none'))
        ax.text(13.2, y_pos, f": {f_desc}", fontsize=8.2, color="#78350F")

    # 9 Conservation Scores (CS) Features
    cs_box = FancyBboxPatch((3.2, 45.0), 27.6, 19.3, boxstyle="round,pad=0.15,rounding_size=0.5",
                            fc="#FEF3C7", ec="#FCD34D", lw=1.2, zorder=3)
    ax.add_patch(cs_box)
    ax.text(4.2, 46.5, "[CS] 9 Evolutionary Conservation Scores:", fontsize=10.2, fontweight='bold', color="#92400E")
    
    cs_features = [
        ("phyloP100way_vertebrate", "PhyloP conservation score across 100 vertebrate genomes"),
        ("phyloP470way_mammalian", "PhyloP conservation score across 470 mammalian genomes"),
        ("phyloP17way_primate", "PhyloP conservation score across 17 primate genomes"),
        ("phastCons100way_vertebrate", "PhastCons conserved element probability across 100 vertebrates"),
        ("phastCons470way_mammalian", "PhastCons conserved element probability across 470 mammals"),
        ("phastCons17way_primate", "PhastCons conserved element probability across 17 primates"),
        ("GERP++_RS", "GERP++ Rejected Substitutions (evolutionary constraint score)"),
        ("GERP++_NR", "GERP++ Neutral Rate (expected neutral substitution rate)"),
        ("GERP_92_mammals", "Standardized GERP score aligned across 92 mammalian species")
    ]
    for idx, (f_name, f_desc) in enumerate(cs_features):
        y_pos = 48.3 + idx * 1.65
        ax.text(4.5, y_pos, f" 2.{idx+1} {f_name:24s}", fontsize=8.5, fontweight='bold', color="#78350F",
                bbox=dict(boxstyle='round,pad=0.12', fc="#FDE68A", ec='none'))
        ax.text(18.0, y_pos, f": {f_desc}", fontsize=7.8, color="#78350F")

    # =========================================================================
    # MODULE 5: SPLICEAI SCORES (9 FUNCTIONAL DISRUPTION SCORES)
    # =========================================================================
    draw_card(33.5, 34.0, 31.5, 31.0, "#FFF1F2", "#BE123C", 
              "5. SPLICEAI SCORES", "#FFFFFF", "9 Disruption Scores", "#E11D48")
    
    # 4 Delta Scores Box
    ds_box = FancyBboxPatch((34.7, 38.0), 29.1, 9.2, boxstyle="round,pad=0.15,rounding_size=0.5",
                            fc="#FFE4E6", ec="#FDA4AF", lw=1.2, zorder=3)
    ax.add_patch(ds_box)
    ax.text(35.7, 39.5, "[DS] 4 Delta Probability Scores:", fontsize=10.2, fontweight='bold', color="#9F1239")
    
    ds_scores = [
        ("SpliceAI_pred_DS_AG", "Delta Score - Acceptor Gain probability (novel acceptor site)"),
        ("SpliceAI_pred_DS_AL", "Delta Score - Acceptor Loss probability (loss of native site)"),
        ("SpliceAI_pred_DS_DG", "Delta Score - Donor Gain probability (novel donor splice site)"),
        ("SpliceAI_pred_DS_DL", "Delta Score - Donor Loss probability (loss of native donor site)")
    ]
    for idx, (s_name, s_desc) in enumerate(ds_scores):
        y_pos = 41.3 + idx * 1.5
        ax.text(36.0, y_pos, f" 1.{idx+1} {s_name:20s}", fontsize=8.6, fontweight='bold', color="#881337",
                bbox=dict(boxstyle='round,pad=0.12', fc="#FECDD3", ec='none'))
        ax.text(49.2, y_pos, f": {s_desc}", fontsize=7.8, color="#881337")

    # 4 Delta Position Scores Box + 1 Max Score Box
    dp_box = FancyBboxPatch((34.7, 48.0), 29.1, 16.3, boxstyle="round,pad=0.15,rounding_size=0.5",
                            fc="#FFE4E6", ec="#FDA4AF", lw=1.2, zorder=3)
    ax.add_patch(dp_box)
    ax.text(35.7, 49.5, "[DP & Max] 4 Delta Positions & 1 Summary Score:", fontsize=10.2, fontweight='bold', color="#9F1239")
    
    dp_scores = [
        ("SpliceAI_pred_DP_AG", "Delta Position - Acceptor Gain distance (bp to novel site)"),
        ("SpliceAI_pred_DP_AL", "Delta Position - Acceptor Loss distance (bp to lost native site)"),
        ("SpliceAI_pred_DP_DG", "Delta Position - Donor Gain distance (bp to novel donor site)"),
        ("SpliceAI_pred_DP_DL", "Delta Position - Donor Loss distance (bp to lost donor site)"),
        ("SpliceAI_pred_DS_max", "[*] Max Delta Score = max(DS_AG, AL, DG, DL) [Splicing impact]")
    ]
    for idx, (s_name, s_desc) in enumerate(dp_scores):
        y_pos = 51.5 + idx * 2.3
        is_max = "DS_max" in s_name
        tag_bg = "#E11D48" if is_max else "#FECDD3"
        tag_fg = "#FFFFFF" if is_max else "#881337"
        ax.text(36.0, y_pos, f" 2.{idx+1} {s_name:20s}", fontsize=8.6, fontweight='bold', color=tag_fg,
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(49.2, y_pos, f": {s_desc}", fontsize=7.6, color="#881337")

    # =========================================================================
    # MODULE 6: GEOMETRIC FEATURES (10 FEATURES = 5 DNA + 5 PROTEIN)
    # =========================================================================
    draw_card(66.5, 34.0, 31.5, 31.0, "#FDF2F8", "#9D174D", 
              "6. GEOMETRIC FEATURES", "#FFFFFF", "10 Features (1,023 Sets)", "#DB2777")
    
    # 5 DNA Geometry Features
    dna_g_box = FancyBboxPatch((67.7, 38.0), 29.1, 11.8, boxstyle="round,pad=0.15,rounding_size=0.5",
                               fc="#FCE7F3", ec="#F472B6", lw=1.2, zorder=3)
    ax.add_patch(dna_g_box)
    ax.text(68.7, 39.5, "[DNA Geom] 5 DNA Manifold Descriptors:", fontsize=10.2, fontweight='bold', color="#831843")
    
    dna_geoms = [
        ("dna_LLR", "Log-Likelihood Ratio on DNA language model output distribution"),
        ("dna_LVD_L2", "Euclidean L2 distance between Ref and Alt latent vectors"),
        ("dna_LVD_Cosine", "Cosine angular deviation between Ref and Alt latent vectors"),
        ("dna_LID", "Local Intrinsic Dimensionality (local manifold complexity k-NN)"),
        ("dna_LVD_Relative", "Relative latent displacement normalized by baseline norm (V2)")
    ]
    for idx, (g_name, g_desc) in enumerate(dna_geoms):
        y_pos = 41.3 + idx * 1.55
        ax.text(69.0, y_pos, f" 1.{idx+1} {g_name:18s}", fontsize=8.6, fontweight='bold', color="#700733",
                bbox=dict(boxstyle='round,pad=0.12', fc="#FBCFE8", ec='none'))
        ax.text(80.5, y_pos, f": {g_desc}", fontsize=7.8, color="#700733")

    # 5 Protein Geometry Features
    prot_g_box = FancyBboxPatch((67.7, 50.5), 29.1, 13.8, boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc="#FCE7F3", ec="#F472B6", lw=1.2, zorder=3)
    ax.add_patch(prot_g_box)
    ax.text(68.7, 52.0, "[Protein Geom] 5 Protein Manifold Descriptors:", fontsize=10.2, fontweight='bold', color="#831843")
    
    prot_geoms = [
        ("prot_LLR", "[*] Top single predictor (+0.0171 MCC; >75% frequency in Top-20)"),
        ("prot_LVD_L2", "Euclidean L2 distance in protein evolutionary embedding space"),
        ("prot_LVD_Cosine", "Cosine angular displacement in pLM latent vector space"),
        ("prot_LID", "Local Intrinsic Dimensionality k-NN on protein manifold"),
        ("prot_LVD_Relative", "Relative latent displacement ratio of variant site (V2)")
    ]
    for idx, (g_name, g_desc) in enumerate(prot_geoms):
        y_pos = 53.8 + idx * 1.75
        is_best = "prot_LLR" in g_name
        tag_bg = "#DB2777" if is_best else "#FBCFE8"
        tag_fg = "#FFFFFF" if is_best else "#700733"
        ax.text(69.0, y_pos, f" 2.{idx+1} {g_name:18s}", fontsize=8.6, fontweight='bold', color=tag_fg,
                bbox=dict(boxstyle='round,pad=0.15', fc=tag_bg, ec='none'))
        ax.text(80.5, y_pos, f": {g_desc}", fontsize=7.8, color="#700733")

    # =========================================================================
    # MODULE 7: 9 MULTIMODAL FUSION & CLASSIFICATION ARCHITECTURES
    # =========================================================================
    draw_card(2, 66.5, 96, 32.0, "#F8FAFC", "#334155", 
              "7. MULTIMODAL FUSION & CLASSIFICATION ARCHITECTURE SPACE (9 ARCHITECTURES)", 
              "#FFFFFF", "3 Framework Families", "#475569")
    
    # Column 1: 4 PyTorch End-to-End Architectures
    pyt_box = FancyBboxPatch((3.5, 70.5), 30.0, 27.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                             fc="#F1F5F9", ec="#94A3B8", lw=1.5, zorder=3)
    ax.add_patch(pyt_box)
    ax.text(4.5, 72.2, "Family 1: PyTorch End-to-End (4 Models)", fontsize=11.2, fontweight='bold', color="#0F172A")
    ax.text(4.5, 73.8, "End-to-end backpropagation training through specialized neural fusion modules", fontsize=8.5, color="#475569")
    
    pyt_archs = [
        ("1. PyTorch_Concat", "Linear Concatenation + Deep MLP Head",
         "Direct concatenation of normalized multimodal feature vectors through deep feed-forward layers with BatchNorm & Dropout."),
        ("2. PyTorch_CrossAttn", "Bidirectional Cross-Modal Attention",
         "Dual cross-attention mechanism allowing DNA and Protein modalities to query mutual contextual representations."),
        ("3. PyTorch_Transformer", "Multi-Head Self-Attention Fusion",
         "Treats each modality as an independent context token, processed through Transformer Encoders for higher-order interactions."),
        ("4. PyTorch_Gating", "Gated Multimodal Unit (Dynamic GMU)",
         "Adaptive gating mechanism learning instance-dependent dynamic routing weights (w_dna, w_prot) across modalities.")
    ]
    for idx, (a_name, a_sub, a_desc) in enumerate(pyt_archs):
        y_pos = 75.8 + idx * 5.7
        ax.text(5.0, y_pos, a_name, fontsize=9.6, fontweight='bold', color="#1E293B",
                bbox=dict(boxstyle='round,pad=0.18', fc="#E2E8F0", ec='#CBD5E1'))
        ax.text(5.0, y_pos + 1.4, a_sub, fontsize=8.5, fontweight='semibold', color="#2563EB")
        ax.text(5.0, y_pos + 2.7, a_desc, fontsize=7.8, color="#334155")

    # Column 2: 1 Pure XGBoost Architecture (TOP 1 SOTA)
    xgb_box = FancyBboxPatch((35.0, 70.5), 28.5, 27.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                             fc="#FEFCE8", ec="#CA8A04", lw=2.0, zorder=3)
    ax.add_patch(xgb_box)
    ax.text(36.0, 72.2, "Family 2: Pure Tree-Based (Top 1 SOTA)", fontsize=11.2, fontweight='bold', color="#854D0E")
    ax.text(36.0, 73.8, "Gradient Boosted Decision Trees directly on unified feature representations", fontsize=8.5, color="#713F12")
    
    # Big Highlight Badge for Pure_XGBoost_Concat
    xgb_badge = FancyBboxPatch((36.5, 75.5), 25.5, 8.8, boxstyle="round,pad=0.3,rounding_size=0.6",
                               fc="#FEF08A", ec="#EAB308", lw=1.5, zorder=4)
    ax.add_patch(xgb_badge)
    ax.text(37.5, 77.2, "[*] 5. Pure_XGBoost_Concat", fontsize=11.2, fontweight='black', color="#713F12", zorder=5)
    ax.text(37.5, 79.0, "PCA (k=32..64) + Tabular Concatenation + XGBoost", fontsize=8.8, fontweight='bold', color="#A16207", zorder=5)
    ax.text(37.5, 80.8, "- Rank #1 Absolute Winner across all 4 benchmark datasets", fontsize=8.6, fontweight='bold', color="#15803D", zorder=5)
    ax.text(37.5, 82.5, "- Peak Mean Performance: MCC = 0.6798 ~ 0.6995", fontsize=8.6, fontweight='bold', color="#15803D", zorder=5)
    
    ax.text(36.5, 86.0, "Core Technical Advantages:", fontsize=9.2, fontweight='bold', color="#854D0E")
    xgb_reasons = [
        "1. Natural non-linear thresholding across Bio features (AF, GERP) and high-dimensional Foundation embeddings.",
        "2. Complete immunity to overfitting via PCA dimensionality reduction on 1280-dim latent representations.",
        "3. Optimal decision boundaries and superior sensitivity under severe pathogenic class imbalance."
    ]
    for idx, reason in enumerate(xgb_reasons):
        ax.text(36.5, 87.8 + idx * 2.8, reason, fontsize=7.8, color="#713F12")

    # Column 3: 4 Hybrid Architectures
    hyb_box = FancyBboxPatch((65.0, 70.5), 31.5, 27.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                             fc="#F5F3FF", ec="#8B5CF6", lw=1.5, zorder=3)
    ax.add_patch(hyb_box)
    ax.text(66.0, 72.2, "Family 3: Hybrid Deep-Tree (4 Models)", fontsize=11.2, fontweight='bold', color="#4C1D95")
    ax.text(66.0, 73.8, "Extract deep latent representation (f_global) from PyTorch to train an XGBoost Head", fontsize=8.5, color="#5B21B6")
    
    hyb_archs = [
        ("6. Hybrid_Concat_XGBoost", "PyTorch Concat Embedding -> XGBoost",
         "Trains PyTorch Concat network to extract high-level f_global vector, followed by XGBoost classification."),
        ("7. Hybrid_CrossAttn_XGBoost", "Cross-Attention Latent Code -> XGBoost",
         "Distills cross-modal interaction representations from Cross-Attention into boosted tree ensembles."),
        ("8. Hybrid_Transformer_XGBoost", "Transformer Context Vector -> XGBoost",
         "Extracts global context vectors from Self-Attention Encoders for gradient-boosted decision boundary learning."),
        ("9. Hybrid_Gating_XGBoost", "Gated Weighted Representation -> XGBoost",
         "Passes dynamically gated multimodal representations from GMU into an XGBoost classifier head.")
    ]
    for idx, (a_name, a_sub, a_desc) in enumerate(hyb_archs):
        y_pos = 75.8 + idx * 5.7
        ax.text(66.5, y_pos, a_name, fontsize=9.6, fontweight='bold', color="#3B0764",
                bbox=dict(boxstyle='round,pad=0.18', fc="#EDE9FE", ec='#DDD6FE'))
        ax.text(66.5, y_pos + 1.4, a_sub, fontsize=8.5, fontweight='semibold', color="#7C3AED")
        ax.text(66.5, y_pos + 2.7, a_desc, fontsize=7.8, color="#4C1D95")

    # =========================================================================
    # SAVE OUTPUTS: HIGH-RES PNG (300 DPI) AND VECTOR PDF
    # =========================================================================
    png_path = os.path.join(output_dir, "0_combinatorial_configuration_matrix.png")
    pdf_path = os.path.join(output_dir, "0_combinatorial_configuration_matrix.pdf")
    
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Successfully saved English Configuration Matrix Visualizations:")
    print(f"     - PNG (300 DPI): {png_path}")
    print(f"     - Vector PDF:   {pdf_path}")

if __name__ == "__main__":
    create_config_matrix_visualization()

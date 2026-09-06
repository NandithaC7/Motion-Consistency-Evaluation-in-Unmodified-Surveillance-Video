r"""
generate_model_analysis_report.py
-----------------------------------
Generates a downloadable MS Word report (.docx) and Markdown report (.md) detailing:
  1. What the model did till now (Architecture, 6 stages, Guard Layers, PyTorch LSTM-AE).
  2. Where the model is failing (Detailed failure points, spatial background dilution, recall limitations).
  3. How it has done it (Mathematical formulas, 4x4 spatial grid pooling, Z-score normalization, temporal velocity loss).
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "Data_set" / "processed_output" / "stage23"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCX_PATH = OUT_DIR / "Model_Performance_and_Failure_Analysis.docx"
MD_PATH   = OUT_DIR / "Model_Performance_and_Failure_Analysis.md"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    for run in h.runs:
        run.font.name = "Arial"
        if level == 1:
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy
        elif level == 2:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2B, 0x54, 0x7E)
        elif level == 3:
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return h


def build_docx_report():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ── Title Block ───────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_t = title_p.add_run("MOTION CONSISTENCY EVALUATION MODEL")
    run_t.font.name = "Arial"
    run_t.font.size = Pt(22)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(18)
    run_sub = sub_p.add_run("Comprehensive Technical Report: Achievements, Failure Analysis & Methodology")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Metadata callout box
    meta_table = doc.add_table(rows=1, cols=1)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = meta_table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    mp = cell.paragraphs[0]
    mp.paragraph_format.space_after = Pt(0)
    r_meta = mp.add_run("Project: Motion Consistency Evaluation in Unmodified Surveillance Video\n"
                        "Author: AI Research & Engineering Team  |  Date: September 2026\n"
                        "Architecture: 6-Stage Dual-Stream PyTorch LSTM Autoencoder with Guard Layers")
    r_meta.font.name = "Arial"
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(0x33, 0x4E, 0x68)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ── Executive Summary ──────────────────────────────────────────────────────
    add_styled_heading(doc, "Executive Summary", level=1)
    p_exec = doc.add_paragraph()
    p_exec.paragraph_format.space_after = Pt(10)
    p_exec.add_run(
        "This report provides a comprehensive technical audit of the 6-Stage Dual-Stream Motion Consistency Evaluation system developed for unmodified surveillance video feeds. "
        "The system combines lightweight statistical Guard Layers (G1 Pixel Variance and G2 SSIM Streak Analysis) with deep learning Dual-Stream PyTorch LSTM Autoencoders (Stream A Frame Differencing and Stream B Optical Flow). "
        "Through architectural upgrades—specifically 4x4 Spatial Grid Pooling, Stream Z-Score Normalization, and Temporal Velocity Difference Loss—the model achieved a "
    )
    r_bold = p_exec.add_run("90.91% Precision (up from 58.33%) and a 3.44x increase in overall F1-Score (0.5607 vs 0.1628), alongside 100.0% Recall on static Frozen video feeds.")
    r_bold.bold = True

    # ── Section 1: What the Model Has Done Till Now ─────────────────────────────
    add_styled_heading(doc, "1. What the Model Has Done Till Now (Accomplishments & Pipeline)", level=1)

    p_s1 = doc.add_paragraph()
    p_s1.paragraph_format.space_after = Pt(8)
    p_s1.add_run("The project fully implements the 6-stage end-to-end motion consistency architecture:")

    bullets_s1 = [
        ("Stage 1: Video Ingestion & Guard Screening", 
         "Ingests raw surveillance video (.mp4/.avi/.mov), extracts frames at 64x64 resolution, normalizes pixels to [0,1], and tracks temporal metadata. Executes Guard Layer G1 (Pixel Variance < 0.0005) and Guard Layer G2 (SSIM streak >= 8 consecutive identical frame pairs) to immediately intercept static/frozen video feeds with 100.0% recall without wasting GPU compute."),
        ("Stage 2: Feature Extraction & 4x4 Spatial Grid Pooling", 
         "Extracts 16-frame sliding windows (stride=1). Computes Stream A (Frame Difference) and Stream B (Farneback Optical Flow magnitude). Applies 4x4 Spatial Grid Pooling to partition each 64x64 frame into 16 spatial motion regions, preserving spatial trajectory information."),
        ("Stage 3: Dual PyTorch LSTM Autoencoders", 
         "Trains two independent 2-layer PyTorch LSTM Autoencoders (Hidden Dim = 64) on healthy video clips. Stream A models spatial frame differencing; Stream B models optical flow magnitude dynamics."),
        ("Stage 4: Z-Score Normalized Stream Divergence Scoring", 
         "Calculates reconstruction errors e_A and e_B, normalizes them via Z-scores (z_A, z_B) so both streams contribute equally (50/50), and evaluates the Divergence Anomaly Score: Score = z_A + z_B + lambda * |z_A - z_B| with hyperparameter grid search (optimal lambda = 0.1)."),
        ("Stage 5: Dual-Window Rolling Adaptive Thresholding", 
         "Calculates rolling dynamic thresholds comparing short-window baseline (Ns=20, mu_s + 2*sigma_s) and long-window baseline (Nl=100, mu_l + 2*sigma_l) to adapt to natural camera scene activity without static threshold hardcoding."),
        ("Stage 6: Multi-Class Classification & Feed Reliability Score (R)", 
         "Classifies video clips into HEALTHY, FROZEN, or IRREGULAR. Computes the Feed Reliability Index R in [0, 1] as an executive camera integrity metric: R = 1 - (Unhealthy Clips / Total Clips)."),
    ]

    for title, desc in bullets_s1:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(4)
        rt = bp.add_run(f"{title}: ")
        rt.bold = True
        rt.font.name = "Arial"
        rd = bp.add_run(desc)
        rd.font.name = "Arial"

    # ── Section 2: Where the Model Is Failing (Failure Analysis) ────────────────
    add_styled_heading(doc, "2. Where the Model Is Failing (Detailed Failure Analysis)", level=1)

    p_fail_intro = doc.add_paragraph()
    p_fail_intro.paragraph_format.space_after = Pt(8)
    p_fail_intro.add_run(
        "While the system achieves outstanding performance on Frozen feeds (100.0% recall) and high precision (90.91%), "
        "thorough stress testing on synthetic temporal anomalies (SHUFFLE and DROP faults) revealed three key technical limitations:"
    )

    failures = [
        ("Failure Point 1: Spatial Background Dilution in Surveillance Feeds",
         "In real-world traffic surveillance footage (e.g. AIC21-Track4), static background pixels (asphalt, sky, buildings) occupy over 90-95% of the frame, while moving vehicles occupy only 5-10%. "
         "When 4x4 spatial grid pooling averages 16x16 pixel blocks, subtle motion of small or distant vehicles is diluted by the surrounding static background pixels. "
         "As a result, dropping 8 frames of a small distant vehicle produces a relatively small MSE change, causing the model to miss subtle distant frame drops."),
        
        ("Failure Point 2: Temporal Anomaly Recall Gap (40.54% Recall vs 90.91% Precision)",
         "The upgraded model achieved a high Precision of 90.91% (meaning when it flags an anomaly, it is almost certainly genuine), but its Recall on SHUFFLE and DROP faults is 40.54%. "
         "This means 59.46% of subtle temporal frame reorderings (where vehicles move slowly or predictably) do not create a reconstruction error large enough to exceed the anomaly threshold."),

        ("Failure Point 3: 1D LSTM Spatial-Temporal Memory Limitation (Need for ConvLSTM)",
         "Standard PyTorch LSTM layers process spatial feature vectors as 1D sequence tensors. "
         "They lack 2D convolutional receptive fields (e.g. 3x3 spatial convolutions over time) that explicitly model physical object velocity vectors across 2D pixel space. "
         "When a vehicle moves across patches, a standard LSTM treats spatial patches as independent features rather than a continuous 2D motion field."),

        ("Failure Point 4: Static Percentile Threshold vs Live Stream Drift",
         "When evaluating dataset-wide percentiles on small sample datasets (184 clips), natural variance between different normal video scenes can be as wide as synthetic temporal fault variations. "
         "In live deployments, a static dataset-wide percentile struggles compared to a stream-specific rolling baseline.")
    ]

    for title, desc in failures:
        add_styled_heading(doc, title, level=2)
        fp = doc.add_paragraph()
        fp.paragraph_format.space_after = Pt(6)
        r_desc = fp.add_run(desc)
        r_desc.font.name = "Arial"

    # ── Section 3: How the Model Has Done It (Methodology & Implementation) ────
    add_styled_heading(doc, "3. How the Model Has Done It (Technical Methodology & Formulas)", level=1)

    p_how = doc.add_paragraph()
    p_how.paragraph_format.space_after = Pt(8)
    p_how.add_run("The improvements were achieved through three core engineering enhancements:")

    # Table of Performance Progression
    table = doc.add_table(rows=6, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Metric / Component", "Initial Baseline Model", "Upgraded Model", "Engineering Impact"]
    
    # Header row styling
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    data_rows = [
        ("Guard Layers (FROZEN)", "100.0% Recall", "100.0% Recall", "Instant screening of static feeds (G1/G2)"),
        ("Spatial Representation", "1D Scalar Mean", "4x4 Spatial Grid Pooling (16 Regions)", "Preserves spatial motion locality per frame"),
        ("Loss Function", "Standard MSE Loss", "MSE + Temporal Velocity Loss (L_temp)", "Penalizes first-order velocity deviations"),
        ("Stream Divergence", "Raw Sum (B dominated)", "Z-Score Stream Normalization", "Balanced 50/50 stream contribution"),
        ("Precision / F1-Score", "Prec=58.3%, F1=0.1628", "Prec=90.9%, F1=0.5607", "3.44x boost in overall anomaly F1-score"),
    ]

    for row_idx, row_data in enumerate(data_rows, start=1):
        row_cells = table.rows[row_idx].cells
        bg_color = "F9FBFD" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = cell_value
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            for run in p.runs:
                run.font.name = "Arial"
                run.font.size = Pt(9.0)
                if col_idx == 2:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Formulas Detail
    add_styled_heading(doc, "Mathematical Formulation & Loss Functions", level=2)

    formulas = [
        ("1. Temporal Velocity Loss", 
         "L_total = MSE(x, hat{x}) + beta * MSE( x_{t} - x_{t-1}, hat{x}_{t} - hat{x}_{t-1} )\n"
         "Penalizes first-order motion derivatives to catch sudden frame drops or speed changes."),
        
        ("2. Stream Z-Score Normalization",
         "z_A = (e_A - mu_A) / sigma_A,   z_B = (e_B - mu_B) / sigma_B\n"
         "Equalizes scale variance between Frame Difference (Stream A) and Optical Flow (Stream B)."),

        ("3. Divergence Anomaly Score Formula",
         "Score_i = z_A,i + z_B,i + lambda * | z_A,i - z_B,i |\n"
         "Optimal lambda = 0.1 selected via grid search to weight stream disagreement."),

        ("4. Feed Reliability Index (R)",
         "R = 1 - (1 / N) * sum( Class_i != HEALTHY ),   R in [0, 1]\n"
         "Provides an executive single-number camera integrity rating.")
    ]

    for title, formula_text in formulas:
        p_fm = doc.add_paragraph()
        p_fm.paragraph_format.space_after = Pt(4)
        rt = p_fm.add_run(f"{title}:\n")
        rt.bold = True
        rt.font.name = "Arial"
        rf = p_fm.add_run(formula_text)
        rf.font.name = "Courier New"
        rf.font.size = Pt(9.5)

    # ── Section 4: Roadmap to 95%+ Performance ─────────────────────────────────
    add_styled_heading(doc, "4. Technical Roadmap to Achieve 95%+ Performance", level=1)

    p_road = doc.add_paragraph()
    p_road.paragraph_format.space_after = Pt(8)
    p_road.add_run("To eliminate remaining false negatives and achieve 95%+ Recall & F1-Score:")

    roadmap_steps = [
        ("Upgrade to 2D ConvLSTM-AE", 
         "Replace flat 1D LSTMs with 2D Convolutional LSTMs (ConvLSTM) operating on (B, T, C, H, W) tensors to preserve spatial convolution kernels across time."),
        ("Motion Bounding Box Masking / Patch Max Pooling", 
         "Apply spatial max pooling or background subtraction masks to zero out static background pixels before grid pooling, ensuring vehicle motion signals are not diluted by stationary pixels."),
        ("Temporal Derivative Contrastive Loss", 
         "Incorporate triplet or contrastive loss between temporally continuous clips and temporally shuffled clips to widen the reconstruction error margin.")
    ]

    for title, desc in roadmap_steps:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(4)
        rt = bp.add_run(f"{title}: ")
        rt.bold = True
        rt.font.name = "Arial"
        rd = bp.add_run(desc)
        rd.font.name = "Arial"

    # Save docx
    doc.save(str(DOCX_PATH))
    print(f"[Report] Saved DOCX report to: {DOCX_PATH}")


def build_md_report():
    md_content = """# MOTION CONSISTENCY EVALUATION MODEL
## Comprehensive Technical Report: Achievements, Failure Analysis & Methodology

**Project**: Motion Consistency Evaluation in Unmodified Surveillance Video  
**Author**: AI Research & Engineering Team  |  **Date**: September 2026  
**Architecture**: 6-Stage Dual-Stream PyTorch LSTM Autoencoder with Guard Layers  

---

### Executive Summary

This report provides a comprehensive technical audit of the 6-Stage Dual-Stream Motion Consistency Evaluation system developed for unmodified surveillance video feeds. The system combines lightweight statistical Guard Layers (G1 Pixel Variance and G2 SSIM Streak Analysis) with deep learning Dual-Stream PyTorch LSTM Autoencoders (Stream A Frame Differencing and Stream B Optical Flow). 

Through architectural upgrades—specifically **4x4 Spatial Grid Pooling**, **Stream Z-Score Normalization**, and **Temporal Velocity Difference Loss**—the model achieved a **90.91% Precision (up from 58.33%)** and a **3.44x increase in overall F1-Score (0.5607 vs 0.1628)**, alongside **100.0% Recall on static Frozen video feeds**.

---

### 1. What the Model Has Done Till Now (Accomplishments & Pipeline)

The project fully implements the 6-stage end-to-end motion consistency architecture:

- **Stage 1: Video Ingestion & Guard Screening**:
  Ingests raw surveillance video (`.mp4`/`.avi`/`.mov`), extracts frames at $64 \\times 64$ resolution, normalizes pixels to $[0,1]$, and tracks temporal metadata. Executes Guard Layer G1 ($\text{Variance} < 0.0005$) and Guard Layer G2 ($\text{SSIM streak} \\ge 8$ consecutive identical frame pairs) to immediately intercept static/frozen video feeds with **100.0% recall** without wasting GPU compute.

- **Stage 2: Feature Extraction & 4x4 Spatial Grid Pooling**:
  Extracts 16-frame sliding windows (stride=1). Computes Stream A (Frame Difference) and Stream B (Farneback Optical Flow magnitude). Applies $4 \\times 4$ Spatial Grid Pooling to partition each $64 \\times 64$ frame into 16 spatial motion regions, preserving spatial trajectory information.

- **Stage 3: Dual PyTorch LSTM Autoencoders**:
  Trains two independent 2-layer PyTorch LSTM Autoencoders (Hidden Dim = 64) on healthy video clips. Stream A models spatial frame differencing; Stream B models optical flow magnitude dynamics.

- **Stage 4: Z-Score Normalized Stream Divergence Scoring**:
  Calculates reconstruction errors $e_A$ and $e_B$, normalizes them via Z-scores ($z_A, z_B$) so both streams contribute equally (50/50), and evaluates the Divergence Anomaly Score:
  $$\\text{Score}_i = z_{A,i} + z_{B,i} + \\lambda \\cdot |z_{A,i} - z_{B,i}|$$
  with hyperparameter grid search (optimal $\\lambda = 0.1$).

- **Stage 5: Dual-Window Rolling Adaptive Thresholding**:
  Calculates rolling dynamic thresholds comparing short-window baseline ($N_s=20, \\mu_s + 2\\sigma_s$) and long-window baseline ($N_l=100, \\mu_l + 2\\sigma_l$) to adapt to natural camera scene activity without static threshold hardcoding.

- **Stage 6: Multi-Class Classification & Feed Reliability Score ($R$)**:
  Classifies video clips into `HEALTHY`, `FROZEN`, or `IRREGULAR`. Computes the Feed Reliability Index $R \\in [0, 1]$ as an executive camera integrity metric:
  $$R = 1 - \\frac{N_{\\text{unhealthy}}}{N_{\\text{total}}}$$

---

### 2. Where the Model Is Failing (Detailed Failure Analysis)

While the system achieves outstanding performance on Frozen feeds (100.0% recall) and high precision (90.91%), thorough stress testing on synthetic temporal anomalies (SHUFFLE and DROP faults) revealed three key technical limitations:

#### Failure Point 1: Spatial Background Dilution in Surveillance Feeds
In real-world traffic surveillance footage (e.g. AIC21-Track4), static background pixels (asphalt, sky, buildings) occupy over 90-95% of the frame, while moving vehicles occupy only 5-10%. When $4 \\times 4$ spatial grid pooling averages $16 \\times 16$ pixel blocks, subtle motion of small or distant vehicles is diluted by the surrounding static background pixels. As a result, dropping 8 frames of a small distant vehicle produces a relatively small MSE change, causing the model to miss subtle distant frame drops.

#### Failure Point 2: Temporal Anomaly Recall Gap (40.54% Recall vs 90.91% Precision)
The upgraded model achieved a high Precision of 90.91% (meaning when it flags an anomaly, it is almost certainly genuine), but its Recall on SHUFFLE and DROP faults is 40.54%. This means 59.46% of subtle temporal frame reorderings (where vehicles move slowly or predictably) do not create a reconstruction error large enough to exceed the anomaly threshold.

#### Failure Point 3: 1D LSTM Spatial-Temporal Memory Limitation (Need for ConvLSTM)
Standard PyTorch LSTM layers process spatial feature vectors as 1D sequence tensors. They lack 2D convolutional receptive fields (e.g. $3 \\times 3$ spatial convolutions over time) that explicitly model physical object velocity vectors across 2D pixel space. When a vehicle moves across patches, a standard LSTM treats spatial patches as independent features rather than a continuous 2D motion field.

#### Failure Point 4: Static Percentile Threshold vs Live Stream Drift
When evaluating dataset-wide percentiles on small sample datasets (184 clips), natural variance between different normal video scenes can be as wide as synthetic temporal fault variations. In live deployments, a static dataset-wide percentile struggles compared to a stream-specific rolling baseline.

---

### 3. How the Model Has Done It (Technical Methodology & Performance Table)

| Metric / Component | Initial Baseline Model | Upgraded Model | Engineering Impact |
| :--- | :--- | :--- | :--- |
| **Guard Layers (FROZEN)** | 100.0% Recall | **100.0% Recall** | Instant screening of static feeds (G1/G2) |
| **Spatial Representation** | 1D Scalar Mean | **4x4 Spatial Grid Pooling (16 Regions)** | Preserves spatial motion locality per frame |
| **Loss Function** | Standard MSE Loss | **MSE + Temporal Velocity Loss ($\\mathcal{L}_{\\text{temp}}$)** | Penalizes first-order velocity deviations |
| **Stream Divergence** | Raw Sum (B dominated) | **Z-Score Stream Normalization** | Balanced 50/50 stream contribution |
| **Precision / F1-Score** | Prec=58.3%, F1=0.1628 | **Prec=90.91%, F1=0.5607** | **3.44x boost in overall anomaly F1-score** |

#### Mathematical Formulation & Loss Functions

1. **Temporal Velocity Loss**:
   $$\\mathcal{L}_{\\text{total}} = \\text{MSE}(x, \\hat{x}) + \\beta \\cdot \\text{MSE}\\left( x_{t} - x_{t-1}, \\hat{x}_{t} - \\hat{x}_{t-1} \\right)$$
   Penalizes first-order motion derivatives to catch sudden frame drops or speed changes.

2. **Stream Z-Score Normalization**:
   $$z_A = \\frac{e_A - \\mu_A}{\\sigma_A}, \\quad z_B = \\frac{e_B - \mu_B}{\\sigma_B}$$
   Equalizes scale variance between Frame Difference (Stream A) and Optical Flow (Stream B).

3. **Divergence Anomaly Score Formula**:
   $$\\text{Score}_i = z_{A,i} + z_{B,i} + \\lambda \\cdot |z_{A,i} - z_{B,i}|$$
   Optimal $\\lambda = 0.1$ selected via grid search to weight stream disagreement.

4. **Feed Reliability Index ($R$)**:
   $$R = 1 - \\frac{1}{N} \\sum_{i=1}^N \\mathbf{1}(\\text{Class}_i \\neq \\text{HEALTHY}), \\quad R \\in [0, 1]$$
   Provides an executive single-number camera integrity rating.

---

### 4. Technical Roadmap to Achieve 95%+ Performance

1. **Upgrade to 2D ConvLSTM-AE**: Replace flat 1D LSTMs with 2D Convolutional LSTMs (ConvLSTM) operating on $(B, T, C, H, W)$ tensors to preserve spatial convolution kernels across time.
2. **Motion Bounding Box Masking / Patch Max Pooling**: Apply spatial max pooling or background subtraction masks to zero out static background pixels before grid pooling, ensuring vehicle motion signals are not diluted by stationary pixels.
3. **Temporal Derivative Contrastive Loss**: Incorporate triplet or contrastive loss between temporally continuous clips and temporally shuffled clips to widen the reconstruction error margin.
"""

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Report] Saved Markdown report to: {MD_PATH}")


if __name__ == "__main__":
    build_docx_report()
    build_md_report()

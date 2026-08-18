#!/usr/bin/env python3
"""
Build the full multi-prompt benchmark PDF report with quality scores.
Adds: quality score matrix, quality-speed metric, updated summary.
"""
import json
import os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

RESULTS = os.path.expanduser("~/projects/jetson-llm-benchmark/multiprompt_results.json")
QUALITY = os.path.expanduser("~/projects/jetson-llm-benchmark/quality_scores.json")
OUTPUT = os.path.expanduser("~/projects/jetson-llm-benchmark/Multi_Prompt_Benchmark_Report.pdf")

with open(RESULTS) as f:
    DATA = json.load(f)
with open(QUALITY) as f:
    QUAL = json.load(f)

MODELS = [
    ("gemma3:1b",          "Gemma 3 1B",        "1.0B",  "Q4_K_M","0.76 GiB"),
    ("codegemma:2b",       "CodeGemma 2B",      "2.51B", "Q4_0",  "1.44 GiB"),
    ("granite3-dense:2b",  "Granite 3.0 2B",    "2.63B", "Q4_K_M","1.49 GiB"),
    ("granite3.2:2b",      "Granite 3.2 2B",    "2.53B", "Q4_K_M","1.44 GiB"),
    ("gemma4 E2B",         "Gemma 4 E2B",       "4.63B", "Q4_0",  "2.63 GiB"),
    ("gemma2:2b",          "Gemma 2 2B",        "2.61B", "Q4_0",  "1.51 GiB"),
    ("lfm2.5:2.6b",        "LFM 2.5 2.6B",      "2.70B", "Q4_K_M","1.55 GiB"),
    ("stablelm-zephyr",    "StableLM Zephyr",   "2.70B", "Q4_K_M","1.55 GiB"),
    ("starcoder2:3b",      "StarCoder2 3B",     "3.03B", "Q4_K_M","1.60 GiB"),
    ("qwen2.5:3b",         "Qwen 2.5 3B",       "3.09B", "Q4_K_M","1.79 GiB"),
    ("qwen2.5-coder:3b",   "Qwen2.5-Coder 3B",  "3.09B", "Q4_K_M","1.82 GiB"),
    ("llama3.2:3b",        "Llama 3.2 3B",      "3.21B", "Q4_K_M","1.87 GiB"),
    ("hermes3:3b",         "Hermes 3 3B",       "3.21B", "Q4_K_M","1.87 GiB"),
    ("granite4:3b",        "Granite 4 3B",      "3.40B", "Q4_K_M","1.95 GiB"),
    ("granite4.1:3b",      "Granite 4.1 3B",    "3.40B", "Q4_K_M","2.00 GiB"),
    ("orca-mini:3b",       "Orca-Mini 3B",      "3.02B", "Q4_K_M","1.90 GiB"),
    ("phi3:3.8b",          "Phi-3 3.8B",        "3.82B", "Q4_0",  "2.03 GiB"),
    ("smallthinker:3b",    "SmallThinker 3B",   "3.40B", "Q8_0",  "3.36 GiB"),
]

PROMPT_IDS = ["code", "iambic", "prose", "creative", "math"]
PROMPT_LABELS = {"code": "Code", "iambic": "Iambic", "prose": "Prose", "creative": "Creative", "math": "Math"}

HEADER_BG = colors.HexColor("#1a237e")
HEADER_FG = colors.white
BEST_BG = colors.HexColor("#e8f5e9")
GOOD_BG = colors.HexColor("#fff8e1")
BAD_BG = colors.HexColor("#ffebee")
ALT_ROW = colors.HexColor("#f5f5f5")

styles = getSampleStyleSheet()

title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=16, spaceAfter=6, textColor=HEADER_BG)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=10)
section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6, textColor=HEADER_BG)
cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=8.5, alignment=TA_CENTER)
cell_left = ParagraphStyle("CellLeft", parent=cell_style, alignment=TA_LEFT)
header_cell = ParagraphStyle("HeaderCell", parent=styles["Normal"], fontSize=7, leading=8.5, alignment=TA_CENTER, textColor=colors.white, fontName="Helvetica-Bold")
header_left = ParagraphStyle("HeaderLeft", parent=header_cell, alignment=TA_LEFT)
note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=4)
small_note = ParagraphStyle("SmallNote", parent=styles["Normal"], fontSize=6.5, leading=8, textColor=colors.grey)


def P(text, style=cell_style):
    return Paragraph(str(text), style)

def get_data(model_key, prompt_id, field):
    return DATA["models"].get(model_key, {}).get(prompt_id, {}).get(field)

def get_quality(model_key, prompt_id):
    return QUAL["scores"].get(model_key, {}).get(prompt_id, {})

def get_metric(model_key, metric):
    return QUAL["metrics"].get(model_key, {}).get(metric, 0)


# ---- Table builders ----

def build_gen_speed_table():
    header = [P("Model", header_left), P("Params", header_cell), P("Quant", header_cell), P("Size", header_cell)]
    for pid in PROMPT_IDS:
        header.append(P(PROMPT_LABELS[pid], header_cell))
    header.append(P("Avg", header_cell))

    rows = [header]
    all_avgs = []
    for key, name, params, quant, size in MODELS:
        row = [P(name, cell_left), P(params), P(quant), P(size)]
        speeds = []
        for pid in PROMPT_IDS:
            v = get_data(key, pid, "gen_tps")
            if v:
                row.append(P(f"{v:.1f}"))
                speeds.append(v)
            else:
                row.append(P("-"))
        avg = sum(speeds) / len(speeds) if speeds else 0
        all_avgs.append(avg)
        row.append(P(f"{avg:.1f}"))
        rows.append(row)

    best_avg = max(all_avgs)
    col_widths = [32*mm, 14*mm, 14*mm, 16*mm] + [18*mm]*5 + [14*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
    for i, avg in enumerate(all_avgs, 1):
        if avg == best_avg:
            style_cmds.append(("BACKGROUND", (-1, i), (-1, i), BEST_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_prompt_eval_table():
    header = [P("Model", header_left)]
    for pid in PROMPT_IDS:
        header.append(P(PROMPT_LABELS[pid], header_cell))
    header.append(P("Avg", header_cell))

    rows = [header]
    all_avgs = []
    for key, name, params, quant, size in MODELS:
        row = [P(name, cell_left)]
        speeds = []
        for pid in PROMPT_IDS:
            v = get_data(key, pid, "prompt_tps")
            if v:
                row.append(P(f"{v:.0f}"))
                speeds.append(v)
            else:
                row.append(P("-"))
        avg = sum(speeds) / len(speeds) if speeds else 0
        all_avgs.append(avg)
        row.append(P(f"{avg:.0f}"))
        rows.append(row)

    best_avg = max(all_avgs)
    col_widths = [36*mm] + [20*mm]*5 + [16*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
    for i, avg in enumerate(all_avgs, 1):
        if avg == best_avg:
            style_cmds.append(("BACKGROUND", (-1, i), (-1, i), BEST_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_token_count_table():
    header = [P("Model", header_left)]
    for pid in PROMPT_IDS:
        header.append(P(PROMPT_LABELS[pid], header_cell))
    header.append(P("Avg", header_cell))

    rows = [header]
    all_avgs = []
    for key, name, params, quant, size in MODELS:
        row = [P(name, cell_left)]
        toks = []
        for pid in PROMPT_IDS:
            v = get_data(key, pid, "tokens_generated")
            if v:
                row.append(P(str(v)))
                toks.append(v)
            else:
                row.append(P("-"))
        avg = sum(toks) / len(toks) if toks else 0
        all_avgs.append(avg)
        row.append(P(f"{avg:.0f}"))
        rows.append(row)

    best_avg = max(all_avgs)
    col_widths = [36*mm] + [20*mm]*5 + [16*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
    for i, avg in enumerate(all_avgs, 1):
        if avg == best_avg:
            style_cmds.append(("BACKGROUND", (-1, i), (-1, i), BEST_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_quality_score_table():
    """Quality score matrix (1-10) for each model x prompt type."""
    header = [P("Model", header_left)]
    for pid in PROMPT_IDS:
        header.append(P(PROMPT_LABELS[pid], header_cell))
    header.append(P("Avg", header_cell))

    rows = [header]
    all_avgs = []
    for key, name, params, quant, size in MODELS:
        row = [P(name, cell_left)]
        scores = []
        for pid in PROMPT_IDS:
            q = get_quality(key, pid)
            score = q.get("score", 0)
            row.append(P(str(score)))
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0
        all_avgs.append(avg)
        row.append(P(f"{avg:.1f}"))
        rows.append(row)

    best_avg = max(all_avgs)
    col_widths = [36*mm] + [20*mm]*5 + [16*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
    # Color-code cells: 8-10 green, 5-7 yellow, 1-4 red
    for row_idx in range(1, len(rows)):
        for col_idx in range(1, 6):
            val = int(rows[row_idx][col_idx].text) if hasattr(rows[row_idx][col_idx], 'text') else 0
            if val >= 8:
                style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), BEST_BG))
            elif val <= 4:
                style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), BAD_BG))
            elif val >= 5 and val <= 7:
                style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), GOOD_BG))
    # Highlight best average
    for i, avg in enumerate(all_avgs, 1):
        if avg == best_avg:
            style_cmds.append(("BACKGROUND", (-1, i), (-1, i), BEST_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_quality_speed_table():
    """Quality-Speed metric: combines quality scores with generation speed."""
    header = [
        P("Model", header_left),
        P("Avg Quality\n(1-10)", header_cell),
        P("Avg Gen\n(tok/s)", header_cell),
        P("Quality-Speed\n(Q x tps / 10)", header_cell),
        P("Quality-Efficiency\n(Q x tok / wall_s)", header_cell),
        P("Rank", header_cell),
    ]

    # Sort by quality-speed descending
    model_metrics = []
    for key, name, params, quant, size in MODELS:
        m = QUAL["metrics"].get(key, {})
        qs = m.get("avg_quality_speed", 0)
        model_metrics.append((key, name, m, qs))
    model_metrics.sort(key=lambda x: x[3], reverse=True)

    rows = [header]
    for rank, (key, name, m, qs) in enumerate(model_metrics, 1):
        rows.append([
            P(name, cell_left),
            P(f"{m.get('avg_quality', 0):.1f}"),
            P(f"{m.get('avg_gen_tps', 0):.1f}"),
            P(f"{qs:.2f}"),
            P(f"{m.get('avg_quality_efficiency', 0):.1f}"),
            P(f"#{rank}"),
        ])

    col_widths = [36*mm, 28*mm, 24*mm, 32*mm, 36*mm, 14*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
    # Highlight top 3
    for i in range(1, min(4, len(rows))):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BEST_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_quality_notes_table():
    """Per-model quality notes for each prompt type."""
    header = [P("Prompt Type", header_left), P("Model", header_cell), P("Score", header_cell), P("Quality Notes", header_left)]

    rows = [header]
    # Show best and worst for each category
    for pid in PROMPT_IDS:
        label = PROMPT_LABELS[pid]
        # Collect all scores for this prompt
        model_scores = []
        for key, name, params, quant, size in MODELS:
            q = get_quality(key, pid)
            score = q.get("score", 0)
            notes = q.get("notes", "")
            model_scores.append((key, name, score, notes))
        # Sort by score descending
        model_scores.sort(key=lambda x: x[2], reverse=True)

        # Best
        best = model_scores[0]
        rows.append([P(label, cell_left), P(best[1], cell_left), P(f"{best[2]}/10"), P(best[3], cell_left)])
        # Worst
        worst = model_scores[-1]
        rows.append([P("", cell_left), P(worst[1], cell_left), P(f"{worst[2]}/10"), P(worst[3], cell_left)])

    col_widths = [22*mm, 30*mm, 14*mm, 100*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    # Color best rows green, worst rows red
    for i in range(1, len(rows)):
        if i % 2 == 1:  # Best rows
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), BEST_BG))
        else:  # Worst rows
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), BAD_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_summary_table():
    """Updated summary with quality data."""
    rec_data = [
        ["Use Case", "Best Model", "Quality", "Gen tok/s", "Why"],
        ["Best overall quality", "Gemma 2 2B", "8.2/10", "25.2", "Highest avg quality, best poetry+math, strong prose+creative"],
        ["Code generation", "CodeGemma 2B", "8/10 (code)", "30.7", "Fastest + correct code, but fails on non-code prompts"],
        ["Best 2B generalist", "Granite 3.2 2B", "7.0/10", "27.3", "Consistent quality, fast, good across all categories"],
        ["All-rounder (3B)", "Qwen 2.5 3B", "7.6/10", "21.5", "Consistent 8/10 across prose, creative, math, code"],
        ["Creative/prose", "Gemma 2 2B", "8/10", "25.2", "Best poetry meter, vivid creative writing, strong clinical prose"],
        ["Math/proofs", "Gemma 2 2B", "9/10", "25.2", "Cleanest proof structure, correct contradiction logic"],
        ["Reasoning/thinking", "Gemma 4 E2B", "5.8/10", "26.7", "Thinking model with reasoning tokens (truncated by 300-token limit)"],
        ["Fastest overall", "CodeGemma 2B", "2.6/10 avg", "30.7", "Fastest gen + prompt eval, but code-only quality"],
        ["Math reasoning (fixed)", "SmallThinker 3B", "8/10 (math)", "19.2", "Full thinking chain then clean formal proof (--jinja + 2000 tok)"],
        ["Best Quality-Speed", "Gemma 2 2B", "QS=20.66", "25.2", "Best balance of quality and generation speed"],
        ["Best value 2B", "StableLM Zephyr", "7.0/10", "27.5", "High quality at 2.7B params, good QS=19.25"],
        ["Avoid", "Orca-Mini 3B", "2.8/10", "7.5", "Refused creative task, wrong math, poor formatting"],
        ["Avoid (base model)", "StarCoder2 3B", "1.8/10", "28.6", "Base code model, not instruction-tuned, echoes prompts"],
    ]

    rows = []
    for r in rec_data:
        if r == rec_data[0]:
            rows.append([P(c, header_cell if i > 0 else header_left) for i, c in enumerate(r)])
        else:
            rows.append([P(c, cell_left if i == 0 else cell_style) for i, c in enumerate(r)])

    col_widths = [34*mm, 28*mm, 20*mm, 18*mm, 66*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_output_samples():
    """Output preview samples for quality comparison."""
    samples = [
        ("Iambic Pentameter - Gemma 2 2B (best: 9/10)", "iambic", "gemma2:2b"),
        ("Iambic Pentameter - CodeGemma 2B (failed: 1/10)", "iambic", "codegemma:2b"),
        ("Math Proof - Gemma 2 2B (best: 9/10)", "math", "gemma2:2b"),
        ("Math Proof - SmallThinker 3B (thinking model, fixed: 8/10)", "math", "smallthinker:3b"),
        ("Code - CodeGemma 2B (specialist: 8/10)", "code", "codegemma:2b"),
        ("Creative - Gemma 2 2B (best: 8/10)", "creative", "gemma2:2b"),
    ]

    elements = []
    for title, pid, model_key in samples:
        elements.append(Paragraph(title, section_style))
        preview = get_data(model_key, pid, "output_preview") or "N/A"
        preview = preview.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|channel|>", "")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        pstyle = ParagraphStyle("Preview", parent=styles["Normal"], fontSize=7, leading=9,
                                fontName="Courier", backColor=colors.HexColor("#f8f8f8"),
                                borderPadding=4, spaceAfter=8)
        elements.append(Paragraph(preview.replace("\n", "<br/>").replace("<", "&lt;").replace(">", "&gt;"), pstyle))

    return elements


# ---- Build the document ----
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=landscape(A4),
    leftMargin=12*mm, rightMargin=12*mm,
    topMargin=12*mm, bottomMargin=12*mm,
)

elements = []

# Page 1: Gen Speed + Prompt Eval Speed
elements.append(Paragraph("Multi-Prompt Benchmark Report", title_style))
elements.append(Paragraph(
    "18 models x 5 prompt styles (code, iambic pentameter, clinical prose, creative writing, math proof) "
    "| NVIDIA Jetson Nano 8GB | llama.cpp 0b1bad1 | GUI off | -ngl 99 -fa on --jinja --temp 0.3 | 2000 token limit",
    subtitle_style
))

elements.append(Paragraph("1. Generation Speed (tok/s)", section_style))
elements.append(Paragraph(
    "Tokens generated per second during inference. Consistent across prompt types because "
    "generation is memory-bandwidth bound (reading weights), not compute bound.",
    small_note
))
elements.append(build_gen_speed_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "CodeGemma 2B leads at 30.7 tok/s avg. The 3B class (Qwen, Hermes, Llama, Phi) "
    "clusters at 20-22 tok/s. Gemma 4 E2B and Granite 2B bridge the gap at 25-28 tok/s.",
    note_style
))

elements.append(Paragraph("2. Prompt Eval Speed (tok/s) - Context Ingestion", section_style))
elements.append(Paragraph(
    "How fast the model processes the input prompt. Varies by prompt type due to token distribution "
    "and attention patterns. Higher = faster time-to-first-token.",
    small_note
))
elements.append(build_prompt_eval_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "CodeGemma leads at 482 tok/s avg. Gemma 4 E2B is slowest at 128 tok/s (thinking model overhead). "
    "Creative/prose prompts generally process faster than code/math across all models.",
    note_style
))

# Page 2: Token counts + Quality Score Matrix
elements.append(PageBreak())
elements.append(Paragraph("3. Tokens Generated - Output Completeness", section_style))
elements.append(Paragraph(
    "Word count of generated output. Lower counts may indicate truncated or incomplete responses. "
    "Higher counts suggest more thorough answers (but not necessarily better quality). "
    "Re-benchmarked with 2000 token limit and --jinja flag for thinking models.",
    small_note
))
elements.append(build_token_count_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "SmallThinker now produces full output (845 tokens on code) after --jinja fix. "
    "Granite models produce clean output after banner-stripping fix. "
    "Thinking models (Gemma4, LFM, SmallThinker) spend tokens on internal reasoning.",
    note_style
))

# Page 3: Quality Score Matrix + Quality-Speed
elements.append(PageBreak())
elements.append(Paragraph("4. Quality Score Matrix (1-10)", section_style))
elements.append(Paragraph(
    "Each model's output scored 1-10 by category. Green = 8-10 (excellent), Yellow = 5-7 (adequate), Red = 1-4 (poor/failure). "
    "Criteria: Code=correctness+hints+docstring; Iambic=meter+form+imagery; Prose=accuracy+structure+terminology; "
    "Creative=sensory+atmosphere+originality; Math=proof+logic+notation.",
    small_note
))
elements.append(build_quality_score_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Gemma 2 2B is the overall quality leader at 8.2/10 avg, winning or tying for best in 4 of 5 categories. "
    "CodeGemma scores 8/10 on code but 1/10 on everything else (chat template loop on non-code prompts). "
    "Qwen 2.5 3B is the most consistent 3B model (7.6/10 avg, no score below 6).",
    note_style
))

elements.append(Paragraph("5. Quality-Speed Metric", section_style))
elements.append(Paragraph(
    "Combines quality scores with generation speed. Quality-Speed = (avg_quality x avg_gen_tps) / 10. "
    "Quality-Efficiency = (quality x tokens) / wall_time. Higher is better. "
    "Rewards models that produce high-quality output quickly.",
    small_note
))
elements.append(build_quality_speed_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Gemma 2 2B wins Quality-Speed at 20.66 - excellent quality at 25.2 tok/s. "
    "Granite 3.2 2B is second at 19.01 - best value 2B model. "
    "CodeGemma's high speed can't compensate for poor non-code quality (QS=7.97).",
    note_style
))

# Page 4: Quality notes + Summary
elements.append(PageBreak())
elements.append(Paragraph("6. Best vs Worst by Category", section_style))
elements.append(Paragraph(
    "Top and bottom performing model for each prompt type, with quality notes explaining the score.",
    small_note
))
elements.append(build_quality_notes_table())
elements.append(Spacer(1, 6*mm))

elements.append(Paragraph("7. Summary Recommendations", section_style))
elements.append(build_summary_table())

# Page 5: Output samples
elements.append(PageBreak())
elements.append(Paragraph("8. Output Samples", section_style))
elements.append(Paragraph(
    "Representative output previews for quality comparison. Truncated to ~500 chars. "
    "Shows best and worst examples per category.",
    small_note
))
elements.extend(build_output_samples())

# Footer
elements.append(Spacer(1, 8*mm))
elements.append(Paragraph(
    "Generated by multiprompt_bench.py + rebench_fixed.py + quality_scoring.py | "
    "Hardware: Jetson Nano 8GB (tegra234, sm_87, 32 Ampere TCs) | "
    "llama.cpp build 0b1bad1 | Q4_0/Q4_K_M/Q8_0 quantization | Flash attention ON | "
    "All layers on GPU (-ngl 99) | --jinja flag for chat templates | 2000 token limit",
    small_note
))

doc.build(elements)
print(f"PDF written to {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT)} bytes")
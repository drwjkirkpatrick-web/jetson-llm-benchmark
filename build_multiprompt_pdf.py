#!/usr/bin/env python3
"""
Build a landscape PDF report for multi-prompt benchmark results.
8 models x 5 prompt styles with gen tok/s, prompt eval tok/s, token counts,
quality observations, and a summary recommendation table.
"""

import json
import os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

RESULTS = os.path.expanduser("~/projects/jetson-llm-benchmark/multiprompt_results.json")
OUTPUT = os.path.expanduser("~/projects/jetson-llm-benchmark/Multi_Prompt_Benchmark_Report.pdf")

with open(RESULTS) as f:
    DATA = json.load(f)

MODELS = [
    ("codegemma:2b",       "CodeGemma 2B",      "2.51B", "Q4_0",  "1.44 GiB"),
    ("granite3-dense:2b",  "Granite 3.0 2B",    "2.63B", "Q4_K_M","1.49 GiB"),
    ("gemma4 E2B",         "Gemma 4 E2B",       "4.63B", "Q4_0",  "2.63 GiB"),
    ("gemma2:2b",          "Gemma 2 2B",        "2.61B", "Q4_0",  "1.51 GiB"),
    ("lfm2.5:2.6b",        "LFM 2.5 2.6B",      "2.70B", "Q4_K_M","1.55 GiB"),
    ("qwen2.5:3b",         "Qwen 2.5 3B",       "3.09B", "Q4_K_M","1.79 GiB"),
    ("hermes3:3b",         "Hermes 3 3B",       "3.21B", "Q4_K_M","1.87 GiB"),
    ("llama3.2:3b",        "Llama 3.2 3B",      "3.21B", "Q4_K_M","1.87 GiB"),
    ("phi3:3.8b",          "Phi-3 3.8B",        "3.82B", "Q4_0",  "2.03 GiB"),
    ("smallthinker:3b",    "SmallThinker 3B",   "3.40B", "Q8_0",  "3.36 GiB"),
]

PROMPT_IDS = ["code", "iambic", "prose", "creative", "math"]
PROMPT_LABELS = {"code": "Code", "iambic": "Iambic", "prose": "Prose", "creative": "Creative", "math": "Math"}

# Color scheme
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


def build_gen_speed_table():
    """Generation speed table."""
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

    # Find best for highlighting
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
    # Alternate row colors
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
    # Highlight best average
    for i, avg in enumerate(all_avgs, 1):
        if avg == best_avg:
            style_cmds.append(("BACKGROUND", (-1, i), (-1, i), BEST_BG))

    t.setStyle(TableStyle(style_cmds))
    return t


def build_prompt_eval_table():
    """Prompt eval speed table."""
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
    """Tokens generated table — output completeness."""
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


def build_quality_table():
    """Quality observations by prompt type."""
    quality_data = [
        ["Prompt Type", "Best Model", "Runner-Up", "Key Observation"],
        ["Code Generation", "codegemma:2b", "granite3-dense:2b", "CodeGemma fastest + correct; Granite good general code"],
        ["Iambic Pentameter", "gemma2:2b", "granite3-dense:2b", "Gemma2 best meter (10-syllable); Granite good form"],
        ["Clinical Prose", "qwen2.5:3b", "gemma2:2b", "Qwen most detailed; Gemma2 best structured; all adequate"],
        ["Creative Writing", "qwen2.5:3b", "lfm2.5:2.6b", "Qwen most vivid; LFM good narrative flow"],
        ["Math Proof", "qwen2.5:3b", "gemma2:2b", "Qwen cleanest proof; Gemma2 proper theorem format"],
    ]

    rows = []
    for r in quality_data:
        if r == quality_data[0]:
            rows.append([P(c, header_cell if i > 0 else header_left) for i, c in enumerate(r)])
        else:
            rows.append([P(c, cell_left if i == 0 else cell_style) for i, c in enumerate(r)])

    col_widths = [30*mm, 32*mm, 32*mm, 70*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 1), (-1, 1), BEST_BG),
        ("BACKGROUND", (0, 2), (-1, 2), ALT_ROW),
        ("BACKGROUND", (0, 3), (-1, 3), BEST_BG),
        ("BACKGROUND", (0, 4), (-1, 4), ALT_ROW),
        ("BACKGROUND", (0, 5), (-1, 5), BEST_BG),
    ]))
    return t


def build_summary_table():
    """Summary recommendation table."""
    rec_data = [
        ["Use Case", "Best Model", "Gen tok/s", "Why"],
        ["Fastest overall", "codegemma:2b", "30.7", "Fastest gen + prompt eval, but code-only quality"],
        ["Best 2B generalist", "granite3-dense:2b", "27.3", "Consistent quality, fast, good code+prose"],
        ["All-rounder (3B)", "qwen2.5:3b", "21.5", "Consistent quality across all 5 prompt styles"],
        ["Code generation", "codegemma:2b", "30.7", "Correct output, code specialist architecture"],
        ["Creative/prose", "gemma2:2b", "25.2", "Best poetry meter, strong clinical prose"],
        ["Math/proofs", "qwen2.5:3b", "21.5", "Cleanest proof structure, correct reasoning"],
        ["Reasoning/thinking", "gemma4 E2B", "26.7", "Thinking model with reasoning tokens"],
        ["Most complete output", "lfm2.5:2.6b", "25.3", "Highest avg token count across prompts"],
    ]

    rows = []
    for r in rec_data:
        if r == rec_data[0]:
            rows.append([P(c, header_cell if i > 0 else header_left) for i, c in enumerate(r)])
        else:
            rows.append([P(c, cell_left if i == 0 else cell_style) for i, c in enumerate(r)])

    col_widths = [38*mm, 32*mm, 20*mm, 74*mm]
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
        ("Iambic Pentameter — gemma2:2b (best)", "iambic", "gemma2:2b"),
        ("Iambic Pentameter — codegemma:2b (failed - chat loop)", "iambic", "codegemma:2b"),
        ("Math Proof — qwen2.5:3b (best)", "math", "qwen2.5:3b"),
        ("Math Proof — gemma2:2b (runner-up)", "math", "gemma2:2b"),
        ("Code — codegemma:2b (specialist)", "code", "codegemma:2b"),
        ("Creative — qwen2.5:3b (best)", "creative", "qwen2.5:3b"),
    ]

    elements = []
    for title, pid, model_key in samples:
        elements.append(Paragraph(title, section_style))
        preview = get_data(model_key, pid, "output_preview") or "N/A"
        # Clean up preview
        preview = preview.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|channel|>", "")
        # Truncate to ~400 chars
        if len(preview) > 400:
            preview = preview[:400] + "..."
        pstyle = ParagraphStyle("Preview", parent=styles["Normal"], fontSize=7, leading=9,
                                fontName="Courier", backColor=colors.HexColor("#f8f8f8"),
                                borderPadding=4, spaceAfter=8)
        elements.append(Paragraph(preview.replace("\n", "<br/>"), pstyle))

    return elements


# Build the document
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=landscape(A4),
    leftMargin=12*mm, rightMargin=12*mm,
    topMargin=12*mm, bottomMargin=12*mm,
)

elements = []

# Title
elements.append(Paragraph("Multi-Prompt Benchmark Report", title_style))
elements.append(Paragraph(
    "10 models x 5 prompt styles (code, iambic pentameter, clinical prose, creative writing, math proof) "
    "| NVIDIA Jetson Orin Nano 8GB | llama.cpp 0b1bad1 | GUI off | -ngl 99 -fa on --temp 0.3",
    subtitle_style
))

# Page 1: Generation Speed
elements.append(Paragraph("1. Generation Speed (tok/s)", section_style))
elements.append(Paragraph(
    "Tokens generated per second during inference. Consistent across prompt types because "
    "generation is memory-bandwidth bound (reading weights), not compute bound.",
    small_note
))
elements.append(build_gen_speed_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Highlight: CodeGemma 2B leads at 30.7 tok/s avg. The 3B class (Qwen, Hermes, Llama, Phi) "
    "clusters tightly at 20-22 tok/s. Gemma 4 E2B (MatFormer) and LFM 2.5 bridge the gap at 25-27 tok/s.",
    note_style
))

# Prompt Eval Speed
elements.append(Paragraph("2. Prompt Eval Speed (tok/s) — Context Ingestion", section_style))
elements.append(Paragraph(
    "How fast the model processes the input prompt. Varies by prompt type due to token distribution "
    "and attention patterns. Higher = faster time-to-first-token.",
    small_note
))
elements.append(build_prompt_eval_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Highlight: CodeGemma leads at 482 tok/s avg. Gemma 4 E2B is slowest at 128 tok/s due to "
    "thinking model overhead (processes internal reasoning context). Creative/prose prompts "
    "generally process faster than code/math across all models.",
    note_style
))

# Page 2: Token counts + quality
elements.append(PageBreak())

elements.append(Paragraph("3. Tokens Generated — Output Completeness", section_style))
elements.append(Paragraph(
    "Word count of generated output. Lower counts may indicate truncated or incomplete responses. "
    "Higher counts suggest more thorough answers (but not necessarily better quality).",
    small_note
))
elements.append(build_token_count_table())
elements.append(Spacer(1, 4*mm))
elements.append(Paragraph(
    "Highlight: LFM 2.5 and Qwen 2.5 produce the most output (219 avg). CodeGemma produces "
    "the least (158 avg) and failed on iambic pentameter (40 tokens of chat template tokens). "
    "Gemma 4 E2B and LFM 2.5 are thinking models that spend tokens on internal reasoning.",
    note_style
))

# Quality observations
elements.append(Paragraph("4. Quality Observations by Prompt Type", section_style))
elements.append(build_quality_table())

# Page 3: Summary + output samples
elements.append(PageBreak())

elements.append(Paragraph("5. Summary Recommendations", section_style))
elements.append(build_summary_table())
elements.append(Spacer(1, 6*mm))

elements.append(Paragraph("6. Output Samples", section_style))
elements.append(Paragraph(
    "Representative output previews for quality comparison. Truncated to ~400 chars.",
    small_note
))
elements.extend(build_output_samples())

# Footer
elements.append(Spacer(1, 8*mm))
elements.append(Paragraph(
    "Generated by multiprompt_bench.py | Hardware: Jetson Orin Nano 8GB (tegra234, sm_87, 32 Ampere TCs) "
    "| llama.cpp build 0b1bad1 | Q4_0/Q4_K_M quantization | Flash attention ON | All layers on GPU (-ngl 99)",
    small_note
))

doc.build(elements)
print(f"PDF written to {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT)} bytes")
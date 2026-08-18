#!/usr/bin/env python3
"""
Build the Coding Benchmark PDF report: 16 models x 5 programming languages.
Landscape A4, Paragraph-wrapped tables, color-coded quality scores.
"""
import json, os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

RESULTS = os.path.expanduser('~/projects/jetson-llm-benchmark/coding_benchmark_results.json')
SCORES = os.path.expanduser('~/projects/jetson-llm-benchmark/coding_quality_scores.json')
PDF_OUT = os.path.expanduser('~/projects/jetson-llm-benchmark/Coding_Benchmark_Report.pdf')

with open(RESULTS) as f:
    bench = json.load(f)
with open(SCORES) as f:
    quality = json.load(f)

MODELS = [
    ("codegemma:2b",       "CodeGemma 2B",      "2.51B", "Q4_0",  "1.44 GiB"),
    ("granite3-dense:2b",  "Granite 3.0 2B",    "2.63B", "Q4_K_M","1.49 GiB"),
    ("granite3.2:2b",      "Granite 3.2 2B",    "2.63B", "Q4_K_M","1.44 GiB"),
    ("gemma4 E2B",         "Gemma 4 E2B",       "5.1B/2B","Q4_0", "2.63 GiB"),
    ("gemma2:2b",          "Gemma 2 2B",        "2.51B", "Q4_0",  "1.51 GiB"),
    ("lfm2.5:2.6b",        "LFM 2.5 2.6B",      "2.77B", "Q4_K_M","1.55 GiB"),
    ("qwen2.5:3b",         "Qwen 2.5 3B",       "3.09B", "Q4_K_M","1.79 GiB"),
    ("hermes3:3b",         "Hermes 3 3B",       "3.82B", "Q4_K_M","1.87 GiB"),
    ("llama3.2:3b",        "Llama 3.2 3B",      "3.21B", "Q4_K_M","1.87 GiB"),
    ("granite4:3b",        "Granite 4 3B",      "3.66B", "Q4_K_M","2.1 GiB"),
    ("granite4.1:3b",      "Granite 4.1 3B",    "3.66B", "Q4_K_M","2.0 GiB"),
    ("phi3:3.8b",          "Phi-3 3.8B",        "3.82B", "Q4_K_M","2.03 GiB"),
    ("stablelm-zephyr",    "StableLM Zephyr",   "1.64B", "Q4_K_M","1.5 GiB"),
    ("smallthinker:3b",    "SmallThinker 3B",   "3.08B", "Q4_K_M","3.36 GiB"),
    ("orca-mini:3b",       "Orca-Mini 3B",      "2.75B", "Q4_K_M","1.9 GiB"),
    ("starcoder2:3b",      "StarCoder2 3B",     "3.03B", "Q4_K_M","1.6 GiB"),
]

PROMPTS = ['html', 'python', 'c', 'basic', 'julia']
PROMPT_LABELS = {
    'html': 'HTML/CSS\nWeb Page',
    'python': 'Python\nData Processing',
    'c': 'C System\nProgramming',
    'basic': 'TRS-80 BASIC\nRetro Game',
    'julia': 'Julia\nNumerical Computing',
}

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, spaceAfter=4, alignment=TA_CENTER)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.grey)
h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceBefore=10, spaceAfter=6)
cell_style = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=7, leading=9)
cell_center = ParagraphStyle('CellCenter', parent=cell_style, alignment=TA_CENTER)
cell_bold = ParagraphStyle('CellBold', parent=cell_style, fontName='Helvetica-Bold')
cell_bold_center = ParagraphStyle('CellBoldCenter', parent=cell_bold, alignment=TA_CENTER)
note_style = ParagraphStyle('Note', parent=styles['Normal'], fontSize=7, leading=9, textColor=colors.darkgrey)

def score_color(score):
    if score >= 8:
        return colors.HexColor('#2d7d2d')  # green
    elif score >= 5:
        return colors.HexColor('#b8a020')  # yellow/gold
    elif score >= 1:
        return colors.HexColor('#b84d4d')  # red
    else:
        return colors.HexColor('#666666')  # grey

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUT,
        pagesize=landscape(A4),
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=12*mm, bottomMargin=12*mm,
    )
    story = []

    # Title
    story.append(Paragraph("Jetson Edge LLM Coding Benchmark", title_style))
    story.append(Paragraph(
        "16 models x 5 programming languages (HTML/CSS, Python, C, TRS-80 BASIC, Julia) "
        "&mdash; NVIDIA Jetson Orin Nano 8GB, llama.cpp, Flash Attention ON, Auto MMQ, --jinja, 2000 tokens",
        subtitle_style))
    story.append(Spacer(1, 8))

    # --- Page 1: Speed Table ---
    story.append(Paragraph("Generation Speed (tokens/sec)", h2_style))

    speed_header = ["Model", "Params", "Quant", "Size"] + [PROMPT_LABELS[p] for p in PROMPTS] + ["Avg"]
    speed_data = [speed_header]

    for model_key, display, params, quant, size in MODELS:
        m = bench['models'].get(model_key, {})
        speeds = []
        row = [Paragraph(display, cell_bold), Paragraph(params, cell_style), Paragraph(quant, cell_style), Paragraph(size, cell_style)]
        for pid in PROMPTS:
            gen = m.get(pid, {}).get('gen_tps', 0)
            speeds.append(gen)
            row.append(Paragraph(f"{gen:.1f}" if gen > 0 else "&mdash;", cell_center))
        avg = sum(speeds) / len(speeds) if speeds else 0
        row.append(Paragraph(f"<b>{avg:.1f}</b>", cell_bold_center))
        speed_data.append(row)

    col_widths = [75, 38, 42, 50] + [68]*5 + [50]
    t = Table(speed_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>Note: Gemma 4 E2B failed to load (wrong tensor count: 601 vs 2012 expected). "
        "StarCoder2 and CodeGemma are base code models, not instruction-tuned &mdash; they echo prompts "
        "and generate completions rather than following chat instructions. "
        "Orca-Mini refused some tasks. All other models used --jinja + 2000 token limit.</i>",
        note_style))

    story.append(PageBreak())

    # --- Page 2: Token Output Table ---
    story.append(Paragraph("Tokens Generated per Prompt", h2_style))

    tok_header = ["Model"] + [PROMPT_LABELS[p] for p in PROMPTS] + ["Avg"]
    tok_data = [tok_header]

    for model_key, display, _, _, _ in MODELS:
        m = bench['models'].get(model_key, {})
        toks = []
        row = [Paragraph(display, cell_bold)]
        for pid in PROMPTS:
            tok = m.get(pid, {}).get('tokens_generated', 0)
            toks.append(tok)
            display_val = str(tok) if tok > 0 else "&mdash;"
            row.append(Paragraph(display_val, cell_center))
        avg = sum(toks) / len(toks) if toks else 0
        row.append(Paragraph(f"<b>{avg:.0f}</b>", cell_bold_center))
        tok_data.append(row)

    col_widths_tok = [90] + [80]*5 + [60]
    t2 = Table(tok_data, colWidths=col_widths_tok, repeatRows=1)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>Higher = more complete output. SmallThinker and LFM 2.5 use ~1000+ tokens for internal reasoning "
        "before producing answers. CodeGemma and StarCoder2 generate 2000+ tokens but much is prompt echoing "
        "or chat loop repetition. 2000 token limit applied to all models.</i>",
        note_style))

    story.append(PageBreak())

    # --- Page 3: Quality Score Matrix ---
    story.append(Paragraph("Code Quality Scores (1-10)", h2_style))

    q_header = ["Model"] + [PROMPT_LABELS[p] for p in PROMPTS] + ["Avg", "QS"]
    q_data = [q_header]

    q_scores = quality['scores']
    q_summary = quality['summary']

    # Sort by avg quality descending
    sorted_models = sorted(MODELS, key=lambda m: q_summary.get(m[0], {}).get('avg_quality', 0), reverse=True)

    for model_key, display, _, _, _ in sorted_models:
        cats = q_scores.get(model_key, {})
        scores = []
        row = [Paragraph(display, cell_bold)]
        for pid in PROMPTS:
            s = cats.get(pid, {}).get('score', 0)
            scores.append(s)
            row.append(Paragraph(f"<b>{s}</b>", cell_bold_center))
        avg = q_summary.get(model_key, {}).get('avg_quality', 0)
        qs = q_summary.get(model_key, {}).get('quality_speed', 0)
        row.append(Paragraph(f"<b>{avg:.1f}</b>", cell_bold_center))
        row.append(Paragraph(f"<b>{qs:.1f}</b>", cell_bold_center))
        q_data.append(row)

    col_widths_q = [90] + [68]*5 + [45, 45]
    t3 = Table(q_data, colWidths=col_widths_q, repeatRows=1)

    # Build color-coded cell backgrounds for scores
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]

    # Color-code score cells (columns 1-5, rows 1+)
    for row_idx, (model_key, display, _, _, _) in enumerate(sorted_models, start=1):
        cats = q_scores.get(model_key, {})
        for col_idx, pid in enumerate(PROMPTS, start=1):
            s = cats.get(pid, {}).get('score', 0)
            bg = score_color(s)
            # Use lighter background
            if s >= 8:
                bg = colors.HexColor('#c6efc6')  # light green
            elif s >= 5:
                bg = colors.HexColor('#ffeb9c')  # light yellow
            elif s >= 1:
                bg = colors.HexColor('#ffc7c7')  # light red
            else:
                bg = colors.HexColor('#dddddd')  # light grey
            style_cmds.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), bg))

    t3.setStyle(TableStyle(style_cmds))
    story.append(t3)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>QS = Quality-Speed metric = (avg_quality &times; avg_gen_tps) / 10. "
        "Green &ge;8, Yellow 5-7, Red 1-4, Grey = 0 (failed). "
        "Sorted by average quality (descending).</i>",
        note_style))

    story.append(PageBreak())

    # --- Page 4: Summary & Recommendations ---
    story.append(Paragraph("Coding Benchmark Summary & Recommendations", h2_style))

    rec_data = [
        ["Use Case", "Best Model", "Quality", "Gen tok/s", "Why"],
        ["Best overall code quality", "Gemma 2 2B + Qwen 2.5 3B", "8.6/10", "25.0 / 21.4",
         "Tie on quality. Gemma 2 faster. Both produce complete, correct code across all 5 languages."],
        ["Best value (quality + speed)", "Granite 3.2 2B", "8.4/10", "27.0",
         "Highest QS (22.66). Fastest among top-quality models. Excellent Julia and HTML."],
        ["Best 2B for coding", "Granite 3.2 2B", "8.4/10", "27.0",
         "Beats Gemma 2 on speed, nearly ties on quality. Best all-round 2B code model."],
        ["Best 3B for coding", "Qwen 2.5 3B", "8.6/10", "21.4",
         "Highest quality among 3B models. Complete code with type hints, error handling, docstrings."],
        ["Fastest code generation", "CodeGemma 2B", "4.4/10", "30.7",
         "30.7 tok/s but poor quality. Chat template issues cause prompt echoing. Good for autocomplete only."],
        ["Best for C / systems code", "Gemma 2 2B", "9/10", "25.0",
         "Perfect C output: linked list, pthread mutex, all 5 functions, memory management, main demo."],
        ["Best for HTML/CSS", "Gemma 2 2B + Qwen 2.5 3B", "9/10", "25.0 / 21.4",
         "Both produce complete responsive pages with flexbox, color scheme, all sections."],
        ["Best for Python", "Gemma 2 2B + Qwen 2.5 3B", "9/10", "25.0 / 21.4",
         "Both: class with type hints, docstrings, csv+json, error handling, example usage."],
        ["Best for TRS-80 BASIC", "Qwen 2.5 3B", "8/10", "21.4",
         "Only model to properly use line numbers, GOTO, string vars, and complete game logic."],
        ["Best for Julia", "Granite 3.2 2B", "9/10", "27.0",
         "Type annotations (Function, Number, Int), docstring with tuple return type. Best Julia syntax."],
        ["Best thinking model", "SmallThinker 3B", "7.0/10", "19.2",
         "Reasons through approach before coding. Good quality but thinking overhead limits output."],
        ["Worst overall", "Orca-Mini 3B", "2.6/10", "12.5",
         "Refused HTML, failed BASIC (user loop), no type hints, slowest. Avoid for coding."],
        ["Not instruction-tuned", "StarCoder2 3B", "2.0/10", "28.7",
         "Base code model. Echoes prompts, generates different tasks. Only useful for code completion."],
        ["Failed to load", "Gemma 4 E2B", "0/10", "0.0",
         "Wrong tensor count (601 vs 2012). Needs clean text-only GGUF. Excluded from coding results."],
    ]

    rec_widths = [100, 95, 55, 65, 330]
    rec_rows = []
    for i, row in enumerate(rec_data):
        if i == 0:
            rec_rows.append([Paragraph(c, ParagraphStyle('hdr', parent=cell_bold, textColor=colors.white, alignment=TA_CENTER)) for c in row])
        else:
            rec_rows.append([Paragraph(row[0], cell_bold), Paragraph(row[1], cell_bold), Paragraph(row[2], cell_center), Paragraph(row[3], cell_center), Paragraph(row[4], cell_style)])

    t4 = Table(rec_rows, colWidths=rec_widths, repeatRows=1)
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t4)

    story.append(Spacer(1, 10))

    # Key findings
    story.append(Paragraph("Key Findings", h2_style))
    findings = [
        "&bull; <b>Gemma 2 2B and Qwen 2.5 3B tie for best code quality</b> (8.6/10). Gemma 2 is faster (25.0 vs 21.4 tok/s).",
        "&bull; <b>Granite 3.2 2B has the best Quality-Speed score</b> (22.66) &mdash; fast (27.0 t/s) and high quality (8.4/10).",
        "&bull; <b>Model size does NOT predict code quality</b> &mdash; 2B models (Gemma 2, Granite 3.2) outperform most 3B+ models.",
        "&bull; <b>Instruction tuning matters more than code pretraining</b> &mdash; StarCoder2 (code specialist) scores 2.0/10 because it is a base model, not instruction-tuned.",
        "&bull; <b>CodeGemma underperforms on coding</b> despite its name &mdash; chat template issues cause prompt echoing and chat loops (score 4.4/10).",
        "&bull; <b>TRS-80 BASIC is the great differentiator</b> &mdash; only Qwen 2.5 3B (8/10) and Granite 3.2 2B (8/10) produce proper line-numbered BASIC with GOTO.",
        "&bull; <b>Julia separates the pack</b> &mdash; Granite 3.2 2B (9/10) and Gemma 2 2B (9/10) are the only models to use proper Julia type annotations.",
        "&bull; <b>Thinking models (SmallThinker, LFM 2.5) produce good code</b> (7.0/10) but reasoning tokens consume 50%+ of the output budget.",
        "&bull; <b>Orca-Mini 3B is unsuitable for coding</b> &mdash; refuses tasks, generates chat loops, produces bare functions without classes.",
        "&bull; <b>Hermes 3 3B truncates HTML</b> &mdash; starts output mid-page (closing tags), suggesting context/positioning issues.",
    ]
    for f in findings:
        story.append(Paragraph(f, cell_style))
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    # --- Page 5: Output Samples (best vs worst per language) ---
    story.append(Paragraph("Output Samples: Best vs Worst per Language", h2_style))

    samples = [
        ('html', 'Granite 3.2 2B (9/10)', 'Orca-Mini 3B (1/10)'),
        ('python', 'Gemma 2 2B (9/10)', 'Orca-Mini 3B (3/10)'),
        ('c', 'Gemma 2 2B (9/10)', 'StarCoder2 3B (2/10)'),
        ('basic', 'Qwen 2.5 3B (8/10)', 'CodeGemma 2B (1/10)'),
        ('julia', 'Granite 3.2 2B (9/10)', 'StarCoder2 3B (2/10)'),
    ]

    for pid, best_label, worst_label in samples:
        # Find the actual models
        best_model = best_label.split(' (')[0].replace('Granite 3.2 2B', 'granite3.2:2b').replace('Gemma 2 2B', 'gemma2:2b').replace('Qwen 2.5 3B', 'qwen2.5:3b')
        worst_model = worst_label.split(' (')[0].replace('Orca-Mini 3B', 'orca-mini:3b').replace('StarCoder2 3B', 'starcoder2:3b').replace('CodeGemma 2B', 'codegemma:2b')

        best_out = bench['models'].get(best_model, {}).get(pid, {}).get('output_preview', 'N/A')
        worst_out = bench['models'].get(worst_model, {}).get(pid, {}).get('output_preview', 'N/A')

        lang_name = PROMPT_LABELS[pid].replace('\n', ' ')
        story.append(Paragraph(f"<b>{lang_name}</b>", ParagraphStyle('lang', parent=h2_style, fontSize=10, spaceBefore=8, spaceAfter=4)))

        sample_data = [
            [Paragraph(f"<b>BEST:</b> {best_label}", cell_bold), Paragraph(f"<b>WORST:</b> {worst_label}", cell_bold)],
            [Paragraph(best_out[:400].replace('<', '&lt;').replace('>', '&gt;'), cell_style),
             Paragraph(worst_out[:400].replace('<', '&lt;').replace('>', '&gt;'), cell_style)],
        ]
        t5 = Table(sample_data, colWidths=[330, 330])
        t5.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#c6efc6')),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#ffc7c7')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t5)
        story.append(Spacer(1, 4))

    # Build
    doc.build(story)
    size = os.path.getsize(PDF_OUT)
    print(f"PDF written to {PDF_OUT}")
    print(f"Size: {size} bytes")

    # Verify
    from pypdf import PdfReader
    r = PdfReader(PDF_OUT)
    print(f"Pages: {len(r.pages)}")

if __name__ == '__main__':
    build_pdf()
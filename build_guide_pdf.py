#!/usr/bin/env python3
"""Build the Jetson LLM Model Guide PDF with proper table wrapping.

Uses landscape A4, Paragraph-wrapped cells, and explicit column widths
so tables don't overflow the page.
"""
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_LEFT

PAGE = landscape(A4)  # 842 x 595 pts
MARGIN = 36  # 0.5 inch
USABLE_W = PAGE[0] - 2 * MARGIN  # ~770 pts

styles = getSampleStyleSheet()
cell_style = ParagraphStyle('Cell', parent=styles['BodyText'],
    fontSize=7, leading=9, alignment=TA_LEFT, spaceBefore=0, spaceAfter=0)
header_cell_style = ParagraphStyle('HdrCell', parent=cell_style,
    fontSize=7, leading=9, fontName='Helvetica-Bold')
title_style = ParagraphStyle('Title2', parent=styles['Heading1'],
    fontSize=16, leading=20)
h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
    fontSize=12, leading=15)

def P(text, style=cell_style):
    """Wrap text as a Paragraph for table cells."""
    return Paragraph(str(text), style)

def make_table(header, rows, col_widths):
    """Build a table with Paragraph-wrapped cells and fixed column widths."""
    hdr = [P(h, header_cell_style) for h in header]
    data = [hdr]
    for row in rows:
        data.append([P(c) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d0d0d0')),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return t

def draw_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(PAGE[0] / 2.0, 0.3 * inch, f"Page {doc.page}")
    canvas.restoreState()

# ---- Content ----
story = []

# Title
story.append(Paragraph("Jetson Edge LLM Model Selection Guide", title_style))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "2B & 3B parameter LLM guide for NVIDIA Jetson 8GB. Covers 24 models: strengths, "
    "best workflows, benchmark speeds, and quick-pick recommendations. Compiled August 18, 2026.",
    styles['BodyText']))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<i>Benchmark method: llama.cpp with -ngl 99 (GPU offload), Q4_K_M quantization, "
    "200-token generation, flash attention. Models with unsupported GGUF formats benchmarked "
    "via Ollama API. Prompt: &quot;Explain the difference between hypothyroidism and "
    "hyperthyroidism in 3 sentences.&quot;</i>",
    styles['BodyText']))
story.append(Spacer(1, 12))

# Part 1: Benchmark Results
story.append(Paragraph("Part 1: Verified Benchmark Results", h2_style))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Models tested and verified on this Jetson 8GB device. Speeds are generation tok/s unless noted.",
    styles['BodyText']))
story.append(Spacer(1, 6))

bench_cols = [85, 55, 60, 70, 70, 430]  # ~770 total
story.append(make_table(
    ["Model", "Params", "Gen tok/s", "Prompt tok/s", "Method", "Notes"],
    [
        ["gemma4:e2b", "2B", "27.1", "41.5", "llama.cpp", "Fastest generator"],
        ["gemma2:2b", "2.6B", "25.1", "117.1", "llama.cpp", "Strong generalist"],
        ["lfm2.5:2.6b", "2.7B", "25.1", "115.0", "llama.cpp", "Best for agentic/tool use"],
        ["qwen2.5:3b", "3.1B", "21.3", "258.4", "llama.cpp", "Best math at 3B"],
        ["phi3:3.8b", "3.8B", "21.2", "186.9", "llama.cpp", "Strongest reasoning/param"],
        ["hermes3:3b", "3B", "20.7", "180.3", "llama.cpp", "Best function calling at 3B"],
        ["llama3.2:3b", "3.2B", "20.6", "293.7", "llama.cpp", "Best ecosystem support"],
        ["qwen3.5:2b", "2B", "8.7", "120.9", "Ollama API", "New RoPE unsupported by llama.cpp"],
    ],
    bench_cols
))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Remaining models are still downloading and will be benchmarked as they arrive.",
    styles['BodyText']))

story.append(PageBreak())

# Part 2: Model Guide
story.append(Paragraph("Part 2: Model Guide - All 24 Models", h2_style))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Organized by model family. Each entry includes developer, architecture, strengths, "
    "best workflows, and cautions.", styles['BodyText']))
story.append(Spacer(1, 8))

# 6-column table widths for model guide tables
# Model, Dev, Params, Key Strengths, Best Workflow, Cautions
mg_cols = [90, 60, 60, 220, 200, 140]  # ~770

# Google Gemma Family
story.append(Paragraph("Google Gemma Family", styles['Heading3']))
story.append(Spacer(1, 4))
story.append(make_table(
    ["Model", "Dev", "Params", "Key Strengths", "Best Workflow", "Cautions"],
    [
        ["Gemma 2 2B", "Google", "2.6B",
         "Knowledge distillation training; punches above weight; strong writing quality",
         "Local chat assistant, summarization, general text generation on edge",
         "8K context only; text-only; struggles with complex reasoning"],
        ["Gemma 3 1B", "Google", "1B",
         "Extremely compact (~529MB int4); 32K context; QAT baked in; multilingual",
         "On-device/mobile text processing, offline assistants on extreme low-memory hardware",
         "1B params limit capability; no vision; best for simple text only"],
        ["Gemma 3n E2B", "Google", "5B raw / 2B eff.",
         "Multimodal: text+image+audio+video; MatFormer elastic inference; 140+ languages; 32K context",
         "Edge multimodal AI: medical image analysis, audio transcription, video frame processing",
         "~5B on-disk footprint; 10-15% accuracy loss vs E4B; complex deployment"],
        ["CodeGemma 2B", "Google", "2.6B",
         "Specialized FIM code completion; trained on 500B+ code tokens; IDE integration",
         "IDE autocomplete / fill-in-the-middle; NOT a chat model",
         "Not instruction-tuned; outputs nonsense as chat; older Gemma 1 arch; use 7B-IT for code chat"],
    ],
    mg_cols
))
story.append(Spacer(1, 12))

# Alibaba Qwen Family
story.append(Paragraph("Alibaba Qwen Family & Microsoft Phi", styles['Heading3']))
story.append(Spacer(1, 4))
story.append(make_table(
    ["Model", "Dev", "Params", "Key Strengths", "Best Workflow", "Cautions"],
    [
        ["Qwen 2.5 3B Instruct", "Alibaba", "3.09B",
         "Exceptional math (65.9 MATH, 86.7 GSM8K); 29+ languages; 32K context (128K extended); Apache 2.0",
         "General-purpose instruction following, multilingual Q&A, math problem solving, structured data extraction",
         "World knowledge limited at 3B; instruction strictness below Phi-3.5"],
        ["Qwen 2.5 Coder 3B", "Alibaba", "3.09B",
         "Code-specialized on 5.5T code tokens; 92+ programming languages; same efficient 3B footprint",
         "Code generation, completion, review/debugging, SQL generation, developer assistant",
         "Narrower general NLP ability; not for open-ended knowledge Q&A or creative writing"],
        ["Qwen 3.5 2B", "Alibaba", "2B",
         "Hybrid DeltaNet+Attention; dual thinking/non-thinking modes; multimodal (vision); 262K context; 201 languages",
         "Minimal VRAM multimodal tasks, long-context documents, multilingual applications, thinking-mode reasoning",
         "Thinking mode must be explicitly enabled; very new (Feb 2026); fewer community benchmarks; llama.cpp unsupported"],
        ["Phi-3 Mini 3.8B", "Microsoft", "3.8B",
         "Rivals Mixtral 8x7B on reasoning (69% MMLU); edge-optimized CPU-first design; 128K context variant",
         "On-device reasoning, math/logic problems, structured extraction, RAG with search augmentation",
         "Default 4K context; limited world knowledge (authors recommend search augmentation); English-focused"],
    ],
    mg_cols
))
story.append(Spacer(1, 12))

# Meta Llama & Fine-tunes
story.append(Paragraph("Meta Llama & Fine-tunes", styles['Heading3']))
story.append(Spacer(1, 4))
story.append(make_table(
    ["Model", "Dev", "Params", "Key Strengths", "Best Workflow", "Cautions"],
    [
        ["Llama 3.2 3B Instruct", "Meta", "3.21B",
         "8 multilingual languages; 128K context; edge-optimized with native quantization; best ecosystem support",
         "Reliable baseline assistant, multilingual edge inference, RAG with long context",
         "No native tool calling; weak coding at 3B; knowledge cutoff Dec 2023; Llama license restrictions"],
        ["Hermes 3 3B", "Nous Research", "3.21B",
         "Full fine-tune of Llama 3.2 3B; advanced function calling + JSON structured output; strong roleplay; user-steerable",
         "Agentic function-calling workflows, structured JSON output, roleplay/character assistants, multi-turn agents",
         "Coding still weak at 3B; permissive alignment may need evaluation for clinical use; inherits Llama knowledge limits"],
        ["Ministral 3B", "Mistral AI", "~3B",
         "Purpose-built edge tier; 128K-256K context; strong multilingual (European + Asian); efficient dense architecture",
         "On-device multilingual assistants, long-context RAG, privacy-preserving local inference",
         "Mistral Research License (non-commercial) - requires commercial license for production/clinical use"],
        ["Cogito 3B", "Deep Cogito", "3B",
         "Hybrid reasoning (direct + reflective modes); IDA training outperforms RLHF; strong native tool calling; 30+ languages",
         "Clinical decision support, multi-step STEM, on-demand reasoning, agentic tool-calling",
         "Preview checkpoint; recommends Q8 quantization (higher VRAM); newer ecosystem with fewer integrations"],
    ],
    mg_cols
))
story.append(Spacer(1, 12))

# IBM Granite Family
story.append(Paragraph("IBM Granite Family", styles['Heading3']))
story.append(Spacer(1, 4))
story.append(make_table(
    ["Model", "Dev", "Params", "Key Strengths", "Best Workflow", "Cautions"],
    [
        ["Granite 3.0 2B Dense", "IBM", "2B",
         "Outperforms Llama-3.2-1B on MMLU/math/code; enterprise safety guardrails; Apache 2.0; 12T training tokens",
         "Lightweight text generation, summarization, RAG retrieval, baseline text-only LLM",
         "No reasoning/CoT; no vision; superseded by 3.2 and 4.x; limited math depth"],
        ["Granite 3.2 2B", "IBM", "2B",
         "Toggleable chain-of-thought (thinking on/off); Vision variant matches 5x larger models on DocVQA/ChartQA; 128K context",
         "Enterprise document understanding + optional reasoning; fast Q&A + occasional step-by-step",
         "Reasoning mode experimental; 2B limits CoT depth; vision variant adds memory overhead"],
        ["Granite 4.0 3B (H-Micro)", "IBM", "3B",
         "Hybrid Mamba-2/Transformer (9:1 ratio); 70% less memory, 2x faster; linear long-context scaling (128K in ~4GB); Apache 2.0",
         "Memory-constrained edge deployment, long-context RAG without chunking, multi-session concurrent inference",
         "Mamba hybrid not fully supported in llama.cpp/PEFT; no CoT mode; fewer GGUF variants available"],
        ["Granite 4.1 3B", "IBM", "3B",
         "Best raw capability in Granite family: tool calling, coding, math, instruction following; standard dense transformer; 512K context",
         "Production agentic workflows, function calling, API orchestration, coding assistance, domain fine-tuning",
         "No Mamba efficiency gains; no toggleable thinking; newest model (mid-2026), less community validation"],
    ],
    mg_cols
))
story.append(Spacer(1, 12))

# Stability AI & BigCode
story.append(Paragraph("Stability AI & BigCode", styles['Heading3']))
story.append(Spacer(1, 4))
story.append(make_table(
    ["Model", "Dev", "Params", "Key Strengths", "Best Workflow", "Cautions"],
    [
        ["StableLM Zephyr 3B", "Stability AI", "3B",
         "60% smaller than 7B yet competitive (MT-Bench 6.64); DPO-tuned for instruction following; edge-optimized",
         "Edge chat assistants, instruction-following Q&A, summarization where latency matters",
         "English-only; weak math (GSM8K 42.3%); no code generation; modest MMLU (46.3)"],
        ["Stable Code 3B", "Stability AI", "2.7B",
         "Matches Code Llama 7B despite 60% smaller; 18 programming languages; FIM support; edge throughput measured",
         "IDE code completion, fill-in-the-middle, lightweight coding on edge hardware",
         "Base model not instruction-tuned (use Instruct variant for chat); surpassed by StarCoder2-3B; code-only"],
        ["StarCoder2 3B", "BigCode", "3B",
         "Outperforms StarCoderBase-15B (5x smaller); 600+ languages; 16K context; FIM support; fully open weights + training data",
         "Transparent/auditable code completion for regulated environments, multi-language code generation",
         "Base model not instruction-tuned; weak math (GSM8K 27.7); OpenRAIL license has use-case restrictions"],
        ["Orca Mini 3B", "Community", "3B",
         "Trained on GPT-4 explanation traces; better reasoning than typical 3B; runs in ~2GB at Q4; easy Ollama deployment",
         "Entry-level edge chatbot, lightweight Q&A and explanation tasks",
         "Older OpenLLaMA architecture; known whitespace generation issues; hallucinates more than newer 3B models"],
    ],
    mg_cols
))
story.append(Spacer(1, 12))

# LG AI, Liquid AI & Community
story.append(Paragraph("LG AI, Liquid AI & Community", styles['Heading3']))
story.append(Spacer(1, 4))
story.append(make_table(
    ["Model", "Dev", "Params", "Key Strengths", "Best Workflow", "Cautions"],
    [
        ["EXAONE 3.5 2.4B", "LG AI Research", "2.4B",
         "Highest instruction-following at its size; 32K long-context; bilingual Korean-English; 6.5T training tokens",
         "Bilingual Korean-English workflows, long-context RAG, clinical/business translation, precise instruction compliance",
         "Research-only license (commercial requires contacting LG); weaker in non-Korean/English languages"],
        ["LFM 2.5 2.6B", "Liquid AI", "2.69B",
         "Hybrid architecture (22 conv + 8 attention); agentic RL trained (BFCL 56.88); 128K context in <2.5GB; ~30 tok/s on phones",
         "On-device AI agents with tool calling, long-context RAG on edge, data extraction pipelines, multi-step agentic workflows",
         "Not for agentic coding or knowledge-heavy tasks; hybrid architecture less mature in tooling; limited knowledge at 2.6B"],
        ["LFM 2.5 230M", "Liquid AI", "230M",
         "213 tok/s on phone, 42 tok/s on Pi 5; only 293-375MB at Q4; tool use + data extraction at tiny scale; fine-tunable",
         "Routing/skill-selection layer for robotics, data extraction pipelines, ultra-low-latency classification on CPU",
         "Not for reasoning, math, code, or creative writing; MMLU-Pro 20.25; a fine-tuning base, not a general assistant"],
        ["SmallThinker 3B", "PowerInfer", "3B",
         "O1-style chain-of-thought reasoning in 3B; step-by-step traces before answering; STEM reasoning above weight class; edge-optimized",
         "Edge reasoning tasks, math/logic decomposition, STEM education, clinical decision support with transparent reasoning",
         "English-only; preview model (may be unstable); reasoning traces add latency; inherits Qwen2.5-3B knowledge limits"],
    ],
    mg_cols
))

story.append(PageBreak())

# Part 3: Quick-Pick Guide
story.append(Paragraph("Part 3: Quick-Pick Guide - Right Model for the Job", h2_style))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Recommendations based on benchmark speeds, model strengths, and Jetson 8GB constraints. "
    "'Verified' = tested on this device.", styles['BodyText']))
story.append(Spacer(1, 6))

qp_cols = [160, 140, 320, 150]  # ~770
story.append(make_table(
    ["Task", "Recommended Model", "Why", "Verified"],
    [
        ["Fastest generation", "Gemma 4 E2B (gemma4:e2b)", "27.1 tok/s - fastest on Jetson; 2B effective params", "Yes - 27.1 tok/s"],
        ["Best generalist chat", "Qwen 2.5 3B Instruct", "Strongest math + multilingual + good instruction following at 3B", "Yes - 21.3 tok/s"],
        ["Best reasoning per param", "Phi-3 Mini 3.8B", "69% MMLU, rivals Mixtral 8x7B; edge-optimized", "Yes - 21.2 tok/s"],
        ["Best function calling / agentic", "Hermes 3 3B", "Full fine-tune with dedicated tool-use training; JSON structured output", "Yes - 20.7 tok/s"],
        ["Best code completion", "StarCoder2 3B", "600+ languages, 16K context, FIM, outperforms 15B models", "Downloading"],
        ["Best code chat/assistant", "Qwen 2.5 Coder 3B", "Instruction-tuned code specialist, 92+ languages, same 3B footprint", "Downloading"],
        ["Best multimodal (vision/audio)", "Gemma 3n E2B", "Text+image+audio+video in ~2GB VRAM via MatFormer", "Downloading"],
        ["Best long context (128K+)", "Llama 3.2 3B or Ministral 3B", "128K-256K context; Llama has best ecosystem, Ministral has 256K", "Llama: Yes 20.6 tok/s"],
        ["Best memory efficiency", "Granite 4.0 3B H-Micro", "Mamba hybrid: 70% less memory, 2x faster, linear context scaling", "Downloading"],
        ["Best reasoning + tool calling", "Cogito 3B", "Dual-mode (direct/reflective); IDA training; native tool calling", "Downloading"],
        ["Best edge agent (tool use)", "LFM 2.5 2.6B", "Agentic RL trained; 128K in <2.5GB; hybrid architecture", "Yes - 25.1 tok/s"],
        ["Best clinical document understanding", "Granite 3.2 2B (Vision)", "DocVQA 0.89, ChartQA 0.87; toggleable reasoning; physician-friendly", "Downloading"],
        ["Best bilingual (Korean/English)", "EXAONE 3.5 2.4B", "Top instruction following + long context; Korean-English bilingual", "Downloading"],
        ["Smallest viable assistant", "Gemma 3 1B", "~529MB int4; 32K context; runs on phones and embedded", "Downloading"],
        ["Best STEM reasoning on edge", "SmallThinker 3B", "O1-style thinking traces; step-by-step decomposition at 3B", "Downloading"],
        ["Best ultra-light routing/classification", "LFM 2.5 230M", "213 tok/s on phone; 293MB; tool use at tiny scale", "Yes (prior tests)"],
        ["Best enterprise baseline", "Granite 4.1 3B", "Tool calling + coding + math; Apache 2.0; ISO certified; fine-tune ready", "Downloading"],
        ["Best toggleable reasoning", "Granite 3.2 2B", "Switch thinking on/off per query; no latency cost when off", "Downloading"],
    ],
    qp_cols
))

story.append(PageBreak())

# Part 4: License & Clinical Notes
story.append(Paragraph("Part 4: License & Clinical Notes", h2_style))
story.append(Spacer(1, 6))

lic_cols = [130, 300, 340]  # ~770
story.append(make_table(
    ["License Type", "Models", "Clinical/Commercial Use"],
    [
        ["Apache 2.0", "Qwen 2.5/3.5, Granite 3.0/3.2/4.0/4.1, StarCoder2, SmallThinker, LFM 2.5",
         "Fully open - commercial and clinical use permitted"],
        ["Llama 3 License", "Llama 3.2 3B, Hermes 3 3B, Cogito 3B",
         "Permitted with restrictions (>700M MAU requires Meta agreement)"],
        ["MIT", "Phi-3 Mini",
         "Fully open - commercial and clinical use permitted"],
        ["Mistral Research License", "Ministral 3B",
         "Non-commercial research only; commercial license required for production/clinical"],
        ["OpenRAIL", "StarCoder2 3B (alt view)",
         "Use-case restrictions apply; check terms for clinical deployment"],
        ["Research-only", "EXAONE 3.5 2.4B",
         "Contact LG AI Research for commercial/clinical licensing"],
        ["Gemma Terms of Use", "Gemma 2/3/3n, CodeGemma",
         "Permitted with Google's acceptance of terms; redistribution allowed"],
    ],
    lic_cols
))
story.append(Spacer(1, 12))
story.append(Paragraph(
    "<b>Clinical note:</b> Hermes provides drafts and support only. Walker must approve all "
    "diagnosis, prescriptions, orders, and disposition. Models marked 'research-only' or "
    "'non-commercial' require explicit licensing before clinical deployment. Preview models "
    "(Cogito 3B, SmallThinker 3B) may have instability - validate outputs carefully.",
    styles['BodyText']))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Jetson 8GB constraint:</b> ~4.9GB available RAM for model weights. All Q4_K_M quantized "
    "2B-3B models fit comfortably. Q8 quantization (Cogito 3B recommendation) uses ~3.2GB "
    "weights + context buffer - still fits but with less headroom.",
    styles['BodyText']))

# ---- Build ----
out_path = "/home/walker/projects/jetson-llm-benchmark/Jetson_Edge_LLM_Model_Guide.pdf"
doc = SimpleDocTemplate(
    out_path,
    pagesize=PAGE,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Jetson Edge LLM Model Selection Guide",
    author="Walker Kirkpatrick",
)
doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
print(f"Built: {out_path}")
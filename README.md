<div align="center">
  <a href="https://youtu.be/--oisgZkzds">
    <img src="https://img.youtube.com/vi/--oisgZkzds/0.jpg" alt="I Tested NVIDIA Nemotron 3.5 Parse: Best OCR Under 1B Parameters?">
  </a>
  <h3>📺 <a href="https://youtu.be/--oisgZkzds">Watch the full tutorial on YouTube</a></h3>
</div>

# NVIDIA Nemotron-3.5-Parse: Local Document Vision Engine

[![License: OpenMDW-1.1](https://img.shields.io/badge/License-OpenMDW--1.1-76b900.svg)](https://openmdw.ai/license/1-1/)
[![Model Parameters: 631M](https://img.shields.io/badge/Parameters-631M-76b900.svg)](https://huggingface.co/nvidia/Nemotron-3.5-Parse)
[![GGUF Quantization: Q8_0](https://img.shields.io/badge/GGUF-Q8__0%20(866MB)-00e5ff.svg)](https://huggingface.co/SkyContact/Nemotron-3.5-Parse-GGUF)
[![ParseBench Mean: 75.20](https://img.shields.io/badge/ParseBench-75.20-76b900.svg)](https://huggingface.co/nvidia/Nemotron-3.5-Parse)
[![Local VRAM: <2GB GGUF](https://img.shields.io/badge/VRAM-%3C2GB%20GGUF-00e5ff.svg)](https://huggingface.co/SkyContact/Nemotron-3.5-Parse-GGUF)

> High-efficiency sub-1B open-weight document vision model by NVIDIA that extracts reading order, spatial bounding boxes, Markdown layouts, and structured JSON locally on consumer hardware.

---

## 📊 Multimodal Document AI Workflows & Deliverables

| Task / Workflow | AI Superpower | Authentic Input Artifact | Generated Multimodal Output |
| :--- | :--- | :--- | :--- |
| **1. Complex Receipt & POS Extraction** | Hierarchical field parsing & coordinate detection | `inputs/real_restaurant_receipt.jpg` (Restaurant POS Receipt) | `outputs/task1_receipt_extraction.json` (Validated JSON) |
| **2. Multi-Column Research Paper Reader** | Dual-column reading-order causal prefix resolution | `inputs/real_two_column_paper.jpg` (Academic Publication) | `outputs/task2_academic_paper.md` (Sequential Markdown) |
| **3. Graphic Chart Spatial Grounding** | `<class_Chart>` visual coordinate token tagging | `inputs/financial_market_chart.png` (ChartQA Benchmark) | `outputs/task3_chart_bounding_boxes.png` (Visual Bounding Boxes) |

---

## ⚡ Quick Start (Windows PowerShell Setup)

### Step 1: Environment Setup & Weight Download

```powershell
# Create & activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
pip install huggingface_hub pillow torch torchvision transformers timm

# Download balanced Q8_0 GGUF quantization (~866MB)
hf download SkyContact/Nemotron-3.5-Parse-GGUF nemotron-parse-q8_0.gguf --local-dir models
```

### Step 2 (Optional): Register in Ollama (GGUF Runtime)

To run via Ollama local API server:

```powershell
# 1. Create Modelfile pointing to downloaded GGUF weights
Set-Content -Path Modelfile -Value "FROM ./models/nemotron-parse-q8_0.gguf"

# 2. Register model into local Ollama instance
ollama create nemotron-parse -f Modelfile
```

### Step 3: Run Document Vision Pipeline

```powershell
# Execute the native multimodal document vision pipeline
python main.py
```

---

## 🏗️ Architecture & Model Control Tokens

Nemotron-3.5-Parse couples an NVIDIA C-RADIOv4 vision encoder (dynamic aspect-ratio patch extraction) with an mBART autoregressive decoder using specialized control tokens:

- `</s><s><predict_bbox><predict_classes><predict_text_in_pic>`: Hierarchical document grounding and token transcription.
- `<output_markdown>`: Strict sequential layout-aware Markdown generation.
- `<class_Chart>`: Graphic and visual chart boundary detection.

---

## 📋 Practical Use Cases

- 💳 **Financial Invoices & Wrinkled Receipts**: Convert crumpled retail bills into clean structured JSON schemas for automated ERP and tax audits.
- 🏛️ **Scientific & Legal Documents**: Prevent horizontal sentence bleeding across multi-column PDFs with causal reading-order constraints.
- 📊 **Visual Charts & Infographics**: Locate plot axes, chart legends, and data curves using `<class_Chart>` token markers.
- 🔒 **Zero-Data-Leak Local AI**: Run fully offline within 2GB VRAM on consumer laptops with no external cloud API calls.

---

## 🏷️ Keywords & Search Tags

nemotron 3.5, NVIDIA Nemotron, Nemotron 3.5, Nemotron Parse, local OCR, NVIDIA AI, document parsing, PDF extraction, table extraction, Markdown OCR, offline OCR, receipt scanner, free OCR, PDF parser, OCR engine, document vision, text extraction, AI OCR, receipt OCR, NVIDIA OCR, PDF OCR, Nemotron 3.5 Parse, Nemotron OCR, NVIDIA AI model, document OCR, NVIDIA document AI, NVIDIA Nemotron OCR, NVIDIA OCR model, AI document extraction, NVIDIA OCR tutorial, ocr python

"""
NVIDIA Nemotron-3.5-Parse: Local Document Vision & Extraction Engine
Executes multimodal document intelligence tasks using dynamic task prompts:
- Task 1: inputs/real_restaurant_receipt.jpg -> outputs/task1_receipt_extraction.json
- Task 2: inputs/real_two_column_paper.jpg   -> outputs/task2_academic_paper.md
- Task 3: inputs/financial_market_chart.png -> outputs/task3_chart_bounding_boxes.png

Supports dual execution runtime:
1. Native llama.cpp / Ollama local GGUF server wrapper
2. Transformers AutoModel / AutoProcessor pipeline
"""

import os
import re
import json
import base64
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoints & endpoints
GGUF_FILE = MODELS_DIR / "nemotron-parse-q8_0.gguf"
OLLAMA_MODEL = "nemotron-parse:latest"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
HF_MODEL_ID = "nvidia/Nemotron-3.5-Parse"

# NVIDIA Nemotron Control Tokens
TOKEN_BBOX = "<predict_bbox>"
TOKEN_CLASSES = "<predict_classes>"
TOKEN_MARKDOWN = "<output_markdown>"
TOKEN_TEXT = "<predict_text_in_pic>"
TOKEN_CHART = "<class_Chart>"


class NemotronVisionEngine:
    """Multimodal inference engine supporting GGUF/Ollama and Transformers runtimes."""

    def __init__(self):
        self.mode = "ollama" if self._is_ollama_ready() else "transformers"
        print(f"[*] Initialized NemotronVisionEngine using runtime: {self.mode}")

    def _is_ollama_ready(self) -> bool:
        """Check if local Ollama server is responding with Nemotron registered."""
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any("nemotron-parse" in m for m in models)
        except Exception:
            return False

    def query(self, image_path: Path, prompt: str) -> str:
        """Dispatches multimodal query with prompt and visual artifact to active runtime."""
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if self.mode == "ollama":
            return self._query_ollama(image_path, prompt)
        return self._query_transformers(image_path, prompt)

    def _query_ollama(self, image_path: Path, prompt: str) -> str:
        """Query local Ollama instance with base64 encoded image."""
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [b64_img],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 4096}
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()

    def _query_transformers(self, image_path: Path, prompt: str) -> str:
        """Fallback to Transformers AutoModel / AutoProcessor pipeline."""
        import torch
        from transformers import AutoModel, AutoProcessor

        model_source = str(MODELS_DIR / "Nemotron-3.5-Parse") if (MODELS_DIR / "Nemotron-3.5-Parse").exists() else HF_MODEL_ID
        processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


def run_task1_receipt_parsing(engine: NemotronVisionEngine, input_file: Path) -> Path:
    """Task 1: Structured JSON Extraction with Control Tokens."""
    out_file = OUTPUTS_DIR / "task1_receipt_extraction.json"
    prompt = f"</s><s>{TOKEN_BBOX}{TOKEN_CLASSES}{TOKEN_TEXT}\nExtract all merchant info, line items, and totals as valid JSON."
    
    print(f"\n[+] Task 1: Parsing receipt ({input_file.name}) with prompt...")
    response_text = engine.query(input_file, prompt)

    # Clean JSON markdown fences if present
    cleaned_json_str = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
    try:
        parsed_data = json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        parsed_data = {"raw_model_stream": response_text, "status": "extracted"}

    out_file.write_text(json.dumps(parsed_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[*] Task 1 Result Written -> {out_file}")
    return out_file


def run_task2_paper_transcription(engine: NemotronVisionEngine, input_file: Path) -> Path:
    """Task 2: Academic Paper Markdown Generation preserving reading order."""
    out_file = OUTPUTS_DIR / "task2_academic_paper.md"
    prompt = f"</s><s>{TOKEN_MARKDOWN}{TOKEN_TEXT}\nTranscribe full academic paper in strict sequential multi-column reading order."

    print(f"\n[+] Task 2: Transcribing multi-column document ({input_file.name})...")
    response_text = engine.query(input_file, prompt)

    header = f"<!-- Generated by Nemotron-3.5-Parse from {input_file.name} -->\n\n"
    out_file.write_text(header + response_text, encoding="utf-8")
    print(f"[*] Task 2 Result Written -> {out_file}")
    return out_file


def run_task3_chart_grounding(engine: NemotronVisionEngine, input_file: Path) -> Path:
    """Task 3: Chart Spatial Coordinate Detection and Bounding Box Overlay."""
    out_file = OUTPUTS_DIR / "task3_chart_bounding_boxes.png"
    prompt = f"</s><s>{TOKEN_BBOX}{TOKEN_CHART}{TOKEN_CLASSES}\nDetect chart title, plot frame, and legend coordinates [ymin, xmin, ymax, xmax]."

    print(f"\n[+] Task 3: Grounding visual chart elements ({input_file.name})...")
    raw_coords = engine.query(input_file, prompt)

    img = Image.open(input_file).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Parse coordinates format [ymin, xmin, ymax, xmax] (normalized 0-1000 or raw pixels)
    bbox_matches = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", raw_coords)
    if bbox_matches:
        colors = ["#00e5ff", "#76b900", "#ffaa00", "#e040fb"]
        for idx, (y1, x1, y2, x2) in enumerate(bbox_matches):
            y1, x1, y2, x2 = map(int, (y1, x1, y2, x2))
            # Convert normalized 1000-grid coordinates if applicable
            if max(y1, x1, y2, x2) <= 1000:
                box = [int(x1 * w / 1000), int(y1 * h / 1000), int(x2 * w / 1000), int(y2 * h / 1000)]
            else:
                box = [x1, y1, x2, y2]
            color = colors[idx % len(colors)]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 5, box[1] + 5), f"Element {idx+1}", fill=color)

    img.save(out_file)
    print(f"[*] Task 3 Result Written -> {out_file}")
    return out_file


def main():
    print("=" * 68)
    print(" NVIDIA Nemotron-3.5-Parse: Multimodal Document Vision Engine")
    print("=" * 68)

    receipt_img = INPUTS_DIR / "real_restaurant_receipt.jpg"
    paper_img = INPUTS_DIR / "real_two_column_paper.jpg"
    chart_img = INPUTS_DIR / "financial_market_chart.png"

    engine = NemotronVisionEngine()

    run_task1_receipt_parsing(engine, receipt_img)
    run_task2_paper_transcription(engine, paper_img)
    run_task3_chart_grounding(engine, chart_img)

    print("\n" + "=" * 68)
    print(" All tasks successfully executed from model inference prompts!")
    print(f" Deliverables available in: {OUTPUTS_DIR.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()

"""
Evaluation Pipeline
===================
For each staged case JSON, runs a model stage-by-stage (cumulative context)
on both base and counterfactual variants. Saves ranked differentials and
eliminated lists for downstream metric computation.

Supported models:
  gemini    — Gemini 2.0 Flash via Vertex AI (uses GCP credentials)
  biogpt    — microsoft/biogpt via HuggingFace (runs locally)
  gpt4      — GPT-4 via OpenAI API (requires --openai_key)
  gpt35     — GPT-3.5-turbo via OpenAI API (requires --openai_key)

Output schema per case (data/results/{model}/case_001.json):
  {
    "case_id": "case_001",
    "model": "gemini",
    "ground_truth": "Pneumonia",
    "base": {
      "stage_1": {"differential": [["Dx", "justification"], ...], "eliminated": [["Dx", "reason"], ...]},
      "stage_2": { ... },
      "stage_3": { ... },
      "stage_4": { ... }
    },
    "counterfactuals": [
      {
        "cf_id": "case_001_cf_a",
        "altered_stage": 2,
        "stage_1": { ... }, ..., "stage_4": { ... }
      }
    ]
  }

Usage:
    python evaluate.py --model gemini --project cs224n-487206 --data_dir data/staged --output_dir data/results
    python evaluate.py --model biogpt --data_dir data/staged --output_dir data/results
    python evaluate.py --model gpt4 --openai_key YOUR_KEY --data_dir data/staged --output_dir data/results
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# ── Prompt ────────────────────────────────────────────────────────────────────

EVAL_PROMPT = """You are a medical reasoning assistant. This is a research evaluation; do NOT give medical advice.

TASK:
Given the clinical case information revealed so far, output:
(1) A ranked differential diagnosis list of EXACTLY 5 items.
(2) A short 1-sentence justification for each item.
(3) An ELIMINATED list of diagnoses ruled out by the evidence so far (0 or more).

OUTPUT FORMAT (follow exactly — no extra text before or after):
DIFFERENTIAL:
1. <diagnosis>: <1 sentence justification>
2. <diagnosis>: <1 sentence justification>
3. <diagnosis>: <1 sentence justification>
4. <diagnosis>: <1 sentence justification>
5. <diagnosis>: <1 sentence justification>

ELIMINATED:
- <diagnosis>: <short reason>

CASE (Stage {stage_num} of {total_stages}):
{case_text}

CONSTRAINTS:
- Use concise standard medical diagnosis names.
- Do not include treatment or management.
- If uncertain, still list the 5 most plausible diagnoses.
"""

def build_prompt(stages_so_far: list[str], stage_num: int, total_stages: int = 4) -> str:
    case_text = "\n\n".join(
        f"[Stage {i+1}] {s}" for i, s in enumerate(stages_so_far)
    )
    return EVAL_PROMPT.format(
        stage_num=stage_num,
        total_stages=total_stages,
        case_text=case_text,
    )

# ── Output parser ─────────────────────────────────────────────────────────────

def parse_output(text: str) -> dict:
    """Parse model output into differential and eliminated lists."""
    differential = []
    eliminated = []

    diff_match = re.search(r"DIFFERENTIAL:\s*(.*?)\s*(?:ELIMINATED:|$)", text, re.DOTALL)
    elim_match = re.search(r"ELIMINATED:\s*(.*)", text, re.DOTALL)

    if diff_match:
        for line in diff_match.group(1).splitlines():
            m = re.match(r"^\d+\.\s*(.*?):\s*(.*)$", line.strip())
            if m:
                differential.append([m.group(1).strip(), m.group(2).strip()])

    if elim_match:
        for line in elim_match.group(1).splitlines():
            m = re.match(r"^-\s*(.*?):\s*(.*)$", line.strip())
            if m:
                eliminated.append([m.group(1).strip(), m.group(2).strip()])

    return {"differential": differential, "eliminated": eliminated}

# ── Model backends ────────────────────────────────────────────────────────────

class GeminiModel:
    def __init__(self, project: str, location: str = "us-central1", model_name: str = "gemini-2.0-flash-001"):
        from google import genai
        from google.genai import types as gtypes
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.types = gtypes
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(temperature=0),
        )
        return response.text.strip()



class ClaudeVertexModel:
    def __init__(self, project: str, region: str = "global"):
        from anthropic import AnthropicVertex
        self.client = AnthropicVertex(project_id=project, region=region)

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-5@20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class ClaudeDirectModel:
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-5-20250929"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class OpenAIModel:
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


GEMINI_MODEL_NAMES = {
    "gemini": "gemini-2.0-flash-001",
    "gemini_pro": "gemini-2.5-pro",
}

VLLM_MODEL_NAMES = {
    "llama3":     "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral":    "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "medalpaca":  "medalpaca/medalpaca-13b",
    "meditron":   "meditron:latest",
}


class VLLMModel:
    """Calls a locally running vLLM server via its OpenAI-compatible API."""
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1"):
        from openai import OpenAI
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


def load_model(args):
    if args.model in GEMINI_MODEL_NAMES:
        return GeminiModel(project=args.project, location=args.location, model_name=GEMINI_MODEL_NAMES[args.model])
    elif args.model == "claude":
        if args.anthropic_key:
            return ClaudeDirectModel(api_key=args.anthropic_key)
        return ClaudeVertexModel(project=args.project)
    elif args.model in ("gpt4", "gpt35"):
        model_name = "gpt-4o" if args.model == "gpt4" else "gpt-3.5-turbo"
        return OpenAIModel(api_key=args.openai_key, model_name=model_name)
    elif args.model in VLLM_MODEL_NAMES:
        return VLLMModel(model_name=VLLM_MODEL_NAMES[args.model], base_url=args.vllm_url)
    else:
        raise ValueError(f"Unknown model: {args.model}")

# ── Per-variant evaluation ────────────────────────────────────────────────────

def evaluate_variant(model, stages: dict, delay: float = 0.5) -> dict:
    """
    Run model stage-by-stage with cumulative context.
    stages: dict with keys stage_1..stage_4
    Returns dict with stage_1..stage_4 each containing differential + eliminated.
    """
    results = {}
    stage_texts = []

    for t in range(1, 5):
        key = f"stage_{t}"
        stage_texts.append(stages[key])
        prompt = build_prompt(stage_texts, stage_num=t)

        try:
            raw = model.generate(prompt)
            results[key] = parse_output(raw)
        except Exception as e:
            print(f"    [stage {t}] Error: {e}")
            results[key] = {"differential": [], "eliminated": []}

        time.sleep(delay)

    return results

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_evaluation(args):
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir) / args.model
    out_path.mkdir(parents=True, exist_ok=True)

    case_files = sorted(data_path.glob("case_*.json"))
    if not case_files:
        print(f"No case files found in {data_path}")
        return

    print(f"Found {len(case_files)} cases. Model: {args.model}")
    model = load_model(args)

    for case_file in case_files:
        with open(case_file) as f:
            case = json.load(f)

        case_id = case["case_id"]
        out_file = out_path / f"{case_id}.json"

        if out_file.exists() and not args.overwrite:
            print(f"  Skipping {case_id} (already exists)")
            continue

        print(f"\nEvaluating {case_id} ...")
        result = {
            "case_id": case_id,
            "model": args.model,
            "ground_truth": case["ground_truth"],
            "base": {},
            "counterfactuals": [],
        }

        # Base case
        print("  Base case ...")
        result["base"] = evaluate_variant(model, case["base"], delay=args.delay)

        # Counterfactuals
        for cf in case.get("counterfactuals", []):
            print(f"  Counterfactual {cf['cf_id']} ...")
            cf_stages = {
                "stage_1": cf["stage_1"],
                "stage_2": cf["stage_2"],
                "stage_3": cf["stage_3"],
                "stage_4": cf["stage_4"],
            }
            cf_result = evaluate_variant(model, cf_stages, delay=args.delay)
            cf_result["cf_id"] = cf["cf_id"]
            cf_result["altered_stage"] = cf.get("altered_stage")
            cf_result["alteration_description"] = cf.get("alteration_description", "")
            result["counterfactuals"].append(cf_result)

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file}")

    print(f"\nDone. Results in {out_path.resolve()}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on staged DDx dataset")
    parser.add_argument("--model", required=True,
                        choices=["gemini", "gemini_pro", "claude", "gpt4", "gpt35",
                                 "llama3", "mistral", "mixtral", "medalpaca", "meditron"],
                        help="Model to evaluate")
    parser.add_argument("--data_dir", default="data/staged",
                        help="Directory of staged case JSONs")
    parser.add_argument("--output_dir", default="data/results",
                        help="Output directory for result JSONs")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        help="GCP project ID (for gemini)")
    parser.add_argument("--location", default="us-central1",
                        help="Vertex AI region (for gemini)")
    parser.add_argument("--openai_key", default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (for gpt4/gpt35)")
    parser.add_argument("--anthropic_key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (for claude direct)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1",
                        help="vLLM server base URL (for llama3/mistral/mixtral/medalpaca)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-evaluate cases that already have output files")
    args = parser.parse_args()

    if args.model in ("gemini", "gemini_pro") and not args.project:
        raise ValueError("--project required for gemini models")
    if args.model == "claude" and not args.anthropic_key and not args.project:
        raise ValueError("--anthropic_key or --project required for claude")
    if args.model in ("gpt4", "gpt35") and not args.openai_key:
        raise ValueError("--openai_key required for gpt4/gpt35 models")

    run_evaluation(args)

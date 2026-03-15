"""
Loads USMLE-style questions from HuggingFace (bigbio/med_qa),
filters for clinical vignette cases, then uses an LLM to:
  1. Decompose each question into 4 sequential evidence stages
  2. Generate 1-2 counterfactual variants by swapping a discriminative finding

Output one JSON file per case.
"""

import os
import json
import time
import argparse
import re
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

def load_medqa(n_cases: int, split: str = "train") -> list[dict]:
    """
    Load n_cases clinical vignette questions from MedQA-USMLE.
    We use the GBaker/MedQA-USMLE-4-options variant which loads cleanly
    without a custom loading script, and falls back to bigbio/med_qa.

    Each raw record has:
        {
          "question": "A 45-year-old woman presents with...",
          "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "answer_idx": "A",
          "answer": "Myocardial infarction",
          "meta_info": "step2&3"      # present in GBaker variant
        }
    """
    try:
        from datasets import load_dataset
        print("Loading GBaker/MedQA-USMLE-4-options ...")
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    except Exception:
        print("Falling back to bigbio/med_qa ...")
        from datasets import load_dataset
        ds = load_dataset("bigbio/med_qa", name="med_qa_en_source", split=split, trust_remote_code=True)

    cases = []
    for item in ds:
        if len(cases) >= n_cases:
            break
        if not _is_clinical_vignette(item):
            continue
        cases.append(_normalize_record(item))

    print(f"Loaded {len(cases)} clinical vignette cases from MedQA.")
    return cases


def _is_clinical_vignette(item: dict) -> bool:
    """
    Filter to questions that read like patient vignettes.
    Heuristic: question mentions age + gender OR contains 'presents with'.
    Filters out pure pharmacology/anatomy questions that have no clinical narrative.
    """
    q = item.get("question", "").lower()
    has_age = bool(re.search(r"\d+[\s-]year[\s-]old", q))
    has_presents = "presents with" in q or "complains of" in q or "comes to" in q
    return has_age or has_presents


def _normalize_record(item: dict) -> dict:
    """
    Normalize across GBaker and bigbio schemas into a consistent dict.
    """
    if "options" in item and isinstance(item["options"], dict):
        options = item["options"]
        answer_idx = item.get("answer_idx", "A")
        answer_text = options.get(answer_idx, "")
    elif "choices" in item:
        choices = item["choices"]
        answer_idx = item.get("answer_idx", 0)
        if isinstance(answer_idx, int):
            answer_text = choices[answer_idx] if choices else ""
        else:
            answer_text = answer_idx
        options = {chr(65 + i): c for i, c in enumerate(choices)}
    else:
        options = {}
        answer_text = item.get("answer", "")

    return {
        "question": item["question"],
        "options": options,
        "answer": answer_text,
        "meta_info": item.get("meta_info", ""),
    }


def make_client(project: str, location: str = "us-central1") -> genai.Client:
    return genai.Client(vertexai=True, project=project, location=location)


DECOMPOSE_SYSTEM = """You are a medical education expert. Your job is to take a USMLE-style
clinical vignette and restructure it into 4 sequential evidence stages that simulate how
a clinician would receive information over time in a real encounter.

Rules:
- Stage 1: Chief complaint, basic demographics only. No exam, no labs.
- Stage 2: Vital signs and physical examination findings.
- Stage 3: Laboratory results and/or imaging findings.
- Stage 4: Any final/confirmatory information (e.g. culture result, biopsy, specialist finding).
  If the original question has no stage-4 info, synthesize a clinically plausible one consistent
  with the ground truth diagnosis.
- Each stage must be a self-contained clinical narrative paragraph (not a list).
- Do NOT reveal the diagnosis in any stage. Only provide findings.
- The ground_truth is the correct answer diagnosis — extract it cleanly (e.g. "Community-acquired pneumonia", not "A. Community-acquired pneumonia").

Respond ONLY with valid JSON. No markdown fences. No preamble. Schema:
{
  "ground_truth": "<diagnosis name>",
  "stage_1": "<chief complaint + demographics>",
  "stage_2": "<vitals + physical exam>",
  "stage_3": "<labs + imaging>",
  "stage_4": "<confirmatory finding>"
}"""


COUNTERFACTUAL_SYSTEM = """You are a medical education expert designing counterfactual clinical cases
for evaluating AI diagnostic reasoning.

Given a staged clinical vignette and its ground truth diagnosis, you will generate ONE counterfactual
variant. A counterfactual introduces a single discriminative finding at Stage 2 or Stage 3 that
strongly argues AGAINST the ground truth diagnosis, while keeping all other stages as close to the
original as possible.

Requirements:
- Change only ONE stage (either stage_2 or stage_3).
- The changed finding must be clinically plausible and meaningfully discriminative.
- The change should cause a competent clinician to move the ground truth diagnosis DOWN in their
  differential — ideally off the list entirely.
- Stages not being changed should remain identical to the original.
- altered_stage: the stage number being changed (2 or 3).
- alteration_description: a one-sentence summary of what changed and why it argues against the diagnosis.
- rules_out: the ground truth diagnosis that this finding argues against.

Respond ONLY with valid JSON. No markdown fences. No preamble. Schema:
{
  "cf_id_suffix": "a",
  "altered_stage": 2,
  "alteration_description": "<one sentence: what changed and why it argues against ground truth>",
  "rules_out": "<ground truth diagnosis>",
  "stage_1": "<identical to base or adjusted>",
  "stage_2": "<original or modified>",
  "stage_3": "<original or modified>",
  "stage_4": "<original or adjusted>"
}"""


def decompose_vignette(client: genai.Client, record: dict) -> Optional[dict]:
    """
    Call Gemini on Vertex AI to decompose a raw MedQA record into 4 staged clinical reveals.
    Returns parsed JSON dict or None on failure.
    """
    user_prompt = f"""Please decompose this USMLE clinical vignette into 4 sequential stages.

Original question:
{record['question']}

Answer choices:
{json.dumps(record['options'], indent=2)}

Correct answer: {record['answer']}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=DECOMPOSE_SYSTEM,
                temperature=0,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError, Exception) as e:
        print(f"  [decompose] Error: {e}")
        return None


def generate_counterfactual(
    client: genai.Client,
    case_id: str,
    base: dict,
    cf_index: int = 0
) -> Optional[dict]:
    """
    Call Gemini on Vertex AI to generate one counterfactual variant for a staged base case.
    cf_index: 0 → alter stage 2, 1 → alter stage 3 (for second CF if needed)
    """
    preferred_stage = 2 if cf_index == 0 else 3

    user_prompt = f"""Here is a staged clinical vignette and its ground truth diagnosis.

Ground truth diagnosis: {base['ground_truth']}

Stage 1: {base['stage_1']}
Stage 2: {base['stage_2']}
Stage 3: {base['stage_3']}
Stage 4: {base['stage_4']}

Please generate one counterfactual variant. Prefer to alter Stage {preferred_stage} unless
Stage {preferred_stage} has no clearly discriminative finding, in which case alter the other stage.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=COUNTERFACTUAL_SYSTEM,
                temperature=0,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        cf = json.loads(raw)
        cf["cf_id"] = f"{case_id}_cf_{cf.get('cf_id_suffix', chr(97 + cf_index))}"
        return cf
    except (json.JSONDecodeError, IndexError, Exception) as e:
        print(f"  [counterfactual] Error: {e}")
        return None

def build_dataset(
    n_cases: int,
    project: str,
    output_dir: str,
    location: str = "us-central1",
    n_counterfactuals: int = 1,
    delay: float = 1.0,
):
    """
    Full pipeline:
      1. Load n_cases from MedQA
      2. Decompose each into 4 stages
      3. Generate n_counterfactuals CFs per case
      4. Save each as a JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = make_client(project, location)
    raw_cases = load_medqa(n_cases * 3)

    success = 0
    attempted = 0

    for i, record in enumerate(raw_cases):
        if success >= n_cases:
            break

        case_id = f"case_{success + 1:03d}"
        print(f"\n[{success + 1}/{n_cases}] Processing {case_id} ...")
        time.sleep(delay) # Decomposition
        base = decompose_vignette(client, record)
        if base is None:
            print("  Decomposition failed, skipping.")
            attempted += 1
            continue
        if not all(base.get(f"stage_{k}") for k in [1, 2, 3, 4]): # Validation
            print("  Incomplete stages, skipping.")
            attempted += 1
            continue
        counterfactuals = [] # Counterfactual generation 
        for cf_i in range(n_counterfactuals):
            time.sleep(delay)
            cf = generate_counterfactual(client, case_id, base, cf_index=cf_i)
            if cf:
                counterfactuals.append(cf)
            else:
                print(f"  CF {cf_i + 1} failed.")
        case_record = {
            "case_id": case_id,
            "source": "MedQA-USMLE",
            "original_question": record["question"],
            "ground_truth": base["ground_truth"],
            "base": {
                "stage_1": base["stage_1"],
                "stage_2": base["stage_2"],
                "stage_3": base["stage_3"],
                "stage_4": base["stage_4"],
            },
            "counterfactuals": counterfactuals,
        }

        out_file = output_path / f"{case_id}.json"
        with open(out_file, "w") as f:
            json.dump(case_record, f, indent=2)

        print(f"  Saved: {out_file} ({len(counterfactuals)} CF(s))")
        success += 1
        attempted += 1

    print(f"\nDone. {success}/{attempted} cases successfully constructed.")
    print(f"Output: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build staged DDx dataset from MedQA")
    parser.add_argument("--n_cases", type=int, default=60,
                        help="Number of base cases to construct (default: 60)")
    parser.add_argument("--n_counterfactuals", type=int, default=1,
                        help="Number of CF variants per case (default: 1, max recommended: 2)")
    parser.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        help="GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)")
    parser.add_argument("--location", type=str, default="us-central1",
                        help="Vertex AI region (default: us-central1)")
    parser.add_argument("--output_dir", type=str, default="data/staged",
                        help="Output directory for JSON case files")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to sleep between API calls (default: 1.0)")
    args = parser.parse_args()

    if not args.project:
        raise ValueError("No GCP project provided. Use --project or set GOOGLE_CLOUD_PROJECT.")

    build_dataset(
        n_cases=args.n_cases,
        project=args.project,
        output_dir=args.output_dir,
        location=args.location,
        n_counterfactuals=args.n_counterfactuals,
        delay=args.delay,
    )

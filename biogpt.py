import re
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM

# ----------------------------
# 1. Load BioGPT locally
# ----------------------------
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
model.eval()

# ----------------------------
# 2. Prompt template
# ----------------------------
PROMPT_TEMPLATE = """You are a medical reasoning assistant. This is a research evaluation; do NOT give medical advice.

TASK:
Given the case information so far, output:
(1) a ranked differential diagnosis list of EXACTLY K=5 items,
(2) a short justification for each item (1 sentence each),
(3) and an ELIMINATED list (0 or more diagnoses) that are ruled out by the evidence so far.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
DIFFERENTIAL:
1. <diagnosis>: <1 sentence justification>
2. <diagnosis>: <1 sentence justification>
3. <diagnosis>: <1 sentence justification>
4. <diagnosis>: <1 sentence justification>
5. <diagnosis>: <1 sentence justification>

ELIMINATED:
- <diagnosis>: <short reason>
- <diagnosis>: <short reason>

CASE (STAGE {stage_num}/{total_stages}):
{case_text}

IMPORTANT CONSTRAINTS:
- Use concise, standard medical diagnosis names.
- Do not include treatment or management.
- If uncertain, still choose the 5 most plausible diagnoses.
- Do not add any text before or after the required format.
"""

# ----------------------------
# 3. Example case
# ----------------------------
base_case = """45-year-old with 3 days of fever, cough, and shortness of breath.
No chest pain. No known chronic lung disease.
Oxygen saturation 92% on room air."""

counterfactual_case = """45-year-old with 3 days of fever, cough, and shortness of breath.
No chest pain. No known chronic lung disease.
Oxygen saturation 92% on room air.
Chest X-ray shows no focal consolidation.
Procalcitonin is low.
Symptoms improved significantly with bronchodilator."""

# ----------------------------
# 4. Build prompt
# ----------------------------
def make_prompt(case_text: str, stage_num: int = 1, total_stages: int = 1) -> str:
    return PROMPT_TEMPLATE.format(
        stage_num=stage_num,
        total_stages=total_stages,
        case_text=case_text,
    )

# ----------------------------
# 5. Run model
# ----------------------------
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=5,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # BioGPT often echoes the prompt; strip it if present
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()

# ----------------------------
# 6. Parse output
# ----------------------------
def parse_output(text: str):
    differential = []
    eliminated = []

    diff_match = re.search(r"DIFFERENTIAL:\s*(.*?)\s*ELIMINATED:", text, re.DOTALL)
    elim_match = re.search(r"ELIMINATED:\s*(.*)", text, re.DOTALL)

    if diff_match:
        diff_block = diff_match.group(1)
        for line in diff_block.splitlines():
            line = line.strip()
            m = re.match(r"^\d+\.\s*(.*?):\s*(.*)$", line)
            if m:
                diagnosis, justification = m.groups()
                differential.append((diagnosis.strip(), justification.strip()))

    if elim_match:
        elim_block = elim_match.group(1)
        for line in elim_block.splitlines():
            line = line.strip()
            m = re.match(r"^-\s*(.*?):\s*(.*)$", line)
            if m:
                diagnosis, reason = m.groups()
                eliminated.append((diagnosis.strip(), reason.strip()))

    return differential, eliminated

# ----------------------------
# 7. Rank -> distribution
# ----------------------------
def rank_to_distribution(differential):
    """
    differential: list of (diagnosis, justification)
    returns dict diagnosis -> probability
    """
    if not differential:
        return {}

    scores = {}
    for i, (diag, _) in enumerate(differential, start=1):
        scores[diag] = 1.0 / i

    Z = sum(scores.values())
    return {diag: score / Z for diag, score in scores.items()}

# ----------------------------
# 8. KL divergence
# ----------------------------
def kl_divergence(p, q, epsilon=1e-8):
    """
    p, q are dicts diagnosis -> probability
    """
    diagnoses = set(p.keys()) | set(q.keys())
    kl = 0.0
    for d in diagnoses:
        pd = p.get(d, epsilon)
        qd = q.get(d, epsilon)
        kl += pd * torch.log(torch.tensor(pd / qd)).item()
    return kl

# ----------------------------
# 9. Run base + counterfactual
# ----------------------------
base_prompt = make_prompt(base_case)
cf_prompt = make_prompt(counterfactual_case)

base_output = generate_response(base_prompt)
cf_output = generate_response(cf_prompt)

print("=== BASE OUTPUT ===")
print(base_output)
print("\n=== COUNTERFACTUAL OUTPUT ===")
print(cf_output)

base_diff, base_elim = parse_output(base_output)
cf_diff, cf_elim = parse_output(cf_output)

print("\n=== PARSED BASE DIFFERENTIAL ===")
print(base_diff)
print("\n=== PARSED COUNTERFACTUAL DIFFERENTIAL ===")
print(cf_diff)

base_dist = rank_to_distribution(base_diff)
cf_dist = rank_to_distribution(cf_diff)

print("\n=== BASE DISTRIBUTION ===")
print(base_dist)
print("\n=== COUNTERFACTUAL DISTRIBUTION ===")
print(cf_dist)

if base_dist and cf_dist:
    print("\n=== KL DIVERGENCE ===")
    print(kl_divergence(base_dist, cf_dist))
#!/usr/bin/env python3
"""
SenseMath v2 Inference + Analysis Pipeline — OpenAI API version.

Runs Use-level inference on the v2 item-family benchmark using the OpenAI API
with async concurrent requests for speed.

Usage:
  # Inference (API key required)
  python run_sensemath_v2_api.py --step inference --model gpt-5-mini --scale 2

  # With higher concurrency
  python run_sensemath_v2_api.py --step inference --model gpt-5-mini --scale 2 --concurrency 40

  # Analyze (no API key needed)
  python run_sensemath_v2_api.py --step analyze --scale 2
"""

import os
import re
import json
import asyncio
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).parent
V2_DATA_DIR = SCRIPT_DIR.parent / "benchmark"
V2_RESULTS_DIR = SCRIPT_DIR.parent / "results"

COT_TEMPLATE = (
    "Please solve the following multiple-choice problem. "
    "Show your step-by-step reasoning, then provide your final answer as "
    "a single capital letter (A, B, C, or D) inside \\boxed{{}}.\n\n"
    "{problem_block}\n\n"
    "Put your final answer in \\boxed{{}}."
)

NC_TEMPLATE = (
    "Please solve the following multiple-choice problem WITHOUT performing precise calculations. "
    "Rely on your mathematical intuition, number sense, and estimation. "
    "Show brief reasoning, then provide your final answer as "
    "a single capital letter (A, B, C, or D) inside \\boxed{{}}.\n\n"
    "{problem_block}\n\n"
    "Put your final answer in \\boxed{{}}."
)

STRICT_TEMPLATE = (
    "Please solve the following multiple-choice problem using strict sequential computation. "
    "You MUST perform every arithmetic operation digit by digit in order. "
    "Do NOT use shortcuts, estimation, rounding, algebraic identities, "
    "or any mental math tricks. Do NOT skip any intermediate steps. "
    "Show each computation fully, then provide your final answer as "
    "a single capital letter (A, B, C, or D) inside \\boxed{{}}.\n\n"
    "{problem_block}\n\n"
    "Put your final answer in \\boxed{{}}."
)


# ── Helpers (same as run_sensemath_v2.py) ────────────────────────────────────

def safe_model_name(model: str) -> str:
    return re.sub(r"[/: ]", "_", model)


def load_json(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_mc_block(question: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D"]
    lines = [f"Question: {question}", "Options:"]
    for i, opt in enumerate(options[:4]):
        lines.append(f"{labels[i]}. {opt}")
    return "\n".join(lines)


def find_correct_letter(options: List[str], answer: str) -> str:
    labels = ["A", "B", "C", "D"]
    for i, opt in enumerate(options[:4]):
        if str(opt).strip().lower() == str(answer).strip().lower():
            return labels[i]
    return "A"


def extract_mc_answer(response: str) -> str:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # 1. Try \boxed{X}
    m = re.search(r"\\boxed\{([A-D])\}", response)
    if m:
        return m.group(1).upper()
    # 2. Try "final answer" / "answer" (skip template like (A/B/C/D))
    m = re.search(r"(?:final answer|answer)\s*(?:\([A-D/]+\))?\s*[:\s]\s*([A-D])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3. Last standalone letter on its own line
    m = re.search(r"(?:^|\n)\s*([A-D])\s*$", response)
    if m:
        return m.group(1).upper()
    # 4. Last occurrence of standalone A-D
    matches = re.findall(r"\b([A-D])\b", response)
    if matches:
        return matches[-1].upper()
    return ""


# ── Async API inference ──────────────────────────────────────────────────────

async def call_openai(client, model: str, prompt: str, max_tokens: int, semaphore: asyncio.Semaphore):
    """Make a single OpenAI chat completion call with concurrency control."""
    async with semaphore:
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            # Some models (e.g. gpt-5-mini) don't support temperature=0
            if "5-mini" not in model and "o1" not in model and "o3" not in model and "o4" not in model:
                kwargs["temperature"] = 0
            response = await client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            token_count = response.usage.completion_tokens if response.usage else 0
            return text.strip(), token_count
        except Exception as e:
            print(f"  API error: {e}")
            return "", 0


async def run_inference_async(args):
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    scale = args.scale
    model = args.model
    model_safe = safe_model_name(model)

    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_path = V2_DATA_DIR / f"sensemath_v2_d{scale}.json"
    if not ds_path.exists():
        print(f"Dataset not found: {ds_path}")
        return
    families = load_json(ds_path)
    print(f"Loaded {len(families)} families from {ds_path}")

    # Load existing results for resumption
    out_path = V2_RESULTS_DIR / f"use_{model_safe}_d{scale}.json"
    existing = load_json(out_path) if out_path.exists() else []
    done = {(r["family_id"], r["variant"], r["condition"]) for r in existing}

    # Build prompts: each family x 3 variants x 2 conditions (CoT, NC) = 6 prompts
    tasks = []  # list of (prompt_text, meta_dict)

    for fam in families:
        fid = fam["family_id"]
        cat = fam["category"]

        for var_name in ("strong_shortcut", "weak_shortcut", "control"):
            task = fam[var_name]
            pm = task["pure_math"]
            options = pm.get("options", [])
            question = pm["question"]
            answer = pm["answer"]
            correct_letter = find_correct_letter(options, answer)

            for cond, template in [("CoT", COT_TEMPLATE), ("NC", NC_TEMPLATE), ("Strict", STRICT_TEMPLATE)]:
                if (fid, var_name, cond) in done:
                    continue

                block = build_mc_block(question, options)
                prompt = template.format(problem_block=block)
                meta = {
                    "family_id": fid,
                    "category": cat,
                    "variant": var_name,
                    "condition": cond,
                    "model": model,
                    "problem": question,
                    "ground_truth": answer,
                    "ground_truth_letter": correct_letter,
                    "shortcut_strength": task.get("shortcut_strength", 0),
                }
                tasks.append((prompt, meta))

    if not tasks:
        print("All conditions already done.")
        return

    print(f"To infer: {len(tasks)} prompts ({len(tasks)//6} families x 6)")

    # Create async client and semaphore for concurrency control
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Launch all requests concurrently (bounded by semaphore)
    async_tasks = [
        call_openai(client, model, prompt, args.max_tokens, semaphore)
        for prompt, _ in tasks
    ]

    print(f"Running {len(async_tasks)} API calls with concurrency={args.concurrency}...")
    results = await asyncio.gather(*async_tasks)

    # Build records
    new_records = []
    for (prompt, meta), (response_text, token_count) in zip(tasks, results):
        rec = dict(meta)
        rec["raw_response"] = response_text
        rec["response_token_count"] = token_count
        new_records.append(rec)

    all_records = existing + new_records
    save_json(all_records, out_path)
    print(f"Saved {len(all_records)} records -> {out_path}")

    # Summary
    cond_counts = Counter((r["variant"], r["condition"]) for r in all_records)
    print(f"\nRecords by variant x condition:")
    for k, v in sorted(cond_counts.items()):
        print(f"  {k[0]:>20} x {k[1]:<4}: {v}")

    await client.close()


def step_inference(args):
    asyncio.run(run_inference_async(args))


# ── Analysis (identical to run_sensemath_v2.py) ─────────────────────────────

def step_analyze(args):
    scale = args.scale

    # Load all model results for this scale
    all_records = []
    for f in sorted(V2_RESULTS_DIR.glob(f"use_*_d{scale}.json")):
        records = load_json(f)
        all_records.extend(records)

    if not all_records:
        print(f"No results found for d={scale}")
        return

    # Score correctness
    for r in all_records:
        predicted = extract_mc_answer(r["raw_response"])
        r["predicted"] = predicted
        r["is_correct"] = (predicted == r.get("ground_truth_letter", ""))

    models = sorted(set(r["model"] for r in all_records))
    categories = sorted(set(r["category"] for r in all_records))

    print(f"\n{'='*90}")
    print(f"  SenseMath v2 Analysis — d={scale}")
    print(f"  {len(all_records)} records, {len(models)} models, {len(categories)} categories")
    print(f"{'='*90}")

    results = {}

    for model in models:
        model_recs = [r for r in all_records if r["model"] == model]
        model_short = model.split("/")[-1][:30]

        # Per variant x condition accuracy
        print(f"\n  Model: {model_short}")
        print(f"  {'Variant':<20} {'Cond':<5} {'N':>5} {'Acc%':>7} {'AvgTok':>7}")
        print(f"  {'-'*50}")

        for var in ("strong_shortcut", "weak_shortcut", "control"):
            for cond in ("CoT", "NC"):
                recs = [r for r in model_recs if r["variant"] == var and r["condition"] == cond]
                if not recs:
                    continue
                n = len(recs)
                acc = sum(1 for r in recs if r["is_correct"]) / n
                avg_tok = sum(r.get("response_token_count", 0) for r in recs) / n
                print(f"  {var:<20} {cond:<5} {n:>5} {acc:>6.1%} {avg_tok:>7.0f}")

                key = f"{safe_model_name(model)}|{var}|{cond}"
                results[key] = {
                    "model": model, "variant": var, "condition": cond,
                    "n": n, "accuracy": round(acc, 4),
                    "avg_tokens": round(avg_tok, 1),
                }

        # Interaction effect: (NS advantage on strong_shortcut) - (NS advantage on control)
        # Per family
        by_family = defaultdict(dict)
        for r in model_recs:
            by_family[(r["family_id"], r["variant"], r["condition"])] = r

        family_ids = sorted(set(r["family_id"] for r in model_recs))
        Y = []
        for fid in family_ids:
            ns_strong = by_family.get((fid, "strong_shortcut", "NC"))
            cot_strong = by_family.get((fid, "strong_shortcut", "CoT"))
            ns_ctrl = by_family.get((fid, "control", "NC"))
            cot_ctrl = by_family.get((fid, "control", "CoT"))

            if not all([ns_strong, cot_strong, ns_ctrl, cot_ctrl]):
                continue

            y = (int(ns_strong["is_correct"]) - int(cot_strong["is_correct"])) - \
                (int(ns_ctrl["is_correct"]) - int(cot_ctrl["is_correct"]))
            Y.append(y)

        if Y:
            Y = np.array(Y, dtype=float)
            mean_Y = float(np.mean(Y))

            # Permutation test (sign-flip)
            rng = np.random.RandomState(42)
            n_perm = 10000
            perm_means = np.zeros(n_perm)
            for p in range(n_perm):
                signs = rng.choice([-1, 1], size=len(Y))
                perm_means[p] = np.mean(Y * signs)
            p_value = float(np.mean(np.abs(perm_means) >= np.abs(mean_Y)))

            # Bootstrap CI
            boot_means = np.zeros(n_perm)
            for b in range(n_perm):
                idx = rng.randint(0, len(Y), size=len(Y))
                boot_means[b] = np.mean(Y[idx])
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            wins = int(np.sum(Y > 0))
            losses = int(np.sum(Y < 0))
            ties = int(np.sum(Y == 0))

            sig = "***" if p_value < 0.0042 else ""  # Bonferroni
            print(f"\n  Interaction (NS vs CoT, strong vs control):")
            print(f"    mean(Y) = {mean_Y:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_value:.6f} {sig}")
            print(f"    W/L/T = {wins}/{losses}/{ties}  (n={len(Y)} families)")

            results[f"{safe_model_name(model)}|interaction"] = {
                "model": model, "n_families": len(Y),
                "mean_Y": round(mean_Y, 4),
                "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.0042,
                "wins": wins, "losses": losses, "ties": ties,
            }

        # Per-category breakdown
        print(f"\n  Per-category accuracy (NC, strong vs control):")
        for cat in categories:
            cat_strong = [r for r in model_recs if r["category"] == cat and r["variant"] == "strong_shortcut" and r["condition"] == "NC"]
            cat_ctrl = [r for r in model_recs if r["category"] == cat and r["variant"] == "control" and r["condition"] == "NC"]
            if cat_strong and cat_ctrl:
                acc_s = sum(1 for r in cat_strong if r["is_correct"]) / len(cat_strong)
                acc_c = sum(1 for r in cat_ctrl if r["is_correct"]) / len(cat_ctrl)
                delta = acc_s - acc_c
                print(f"    {cat:<25} strong={acc_s:.1%}  ctrl={acc_c:.1%}  delta={delta:+.1%}")

    # Save
    out_path = V2_RESULTS_DIR / f"analysis_d{scale}.json"
    save_json(results, out_path)
    print(f"\nSaved analysis -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SenseMath v2 inference via OpenAI API")
    parser.add_argument("--step", required=True,
                        choices=["inference", "analyze"])
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests")
    args = parser.parse_args()

    if args.step == "inference":
        step_inference(args)
    elif args.step == "analyze":
        step_analyze(args)


if __name__ == "__main__":
    main()

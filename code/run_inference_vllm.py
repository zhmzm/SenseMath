#!/usr/bin/env python3
"""
SenseMath v2 Inference + Analysis Pipeline.

Runs Use-level inference on the v2 item-family benchmark (6 categories, 3 variants).
Also supports Judge (J1/J2/J3) and Generate (G1/G2) tasks.

Usage:
  # Inference (GPU required)
  python run_sensemath_v2.py --step inference --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4 --scale 2

  # Analyze (no GPU)
  python run_sensemath_v2.py --step analyze --scale 2

  # Judge tasks (GPU or API)
  python run_sensemath_v2.py --step judge-inference --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4

  # Generate tasks (GPU or API)
  python run_sensemath_v2.py --step generate-inference --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
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
    "Please solve the following multiple-choice problem using only easy calculations. "
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

FR_COT_TEMPLATE = (
    "Solve the following math problem. Show your step-by-step reasoning, "
    "then state your final answer clearly.\n\n"
    "Problem: {question}\n\nSolution:"
)

FR_NC_TEMPLATE = (
    "Solve the following math problem WITHOUT performing precise calculations. "
    "Use estimation, number sense, and shortcuts. "
    "Show brief reasoning, then state your final answer.\n\n"
    "Problem: {question}\n\nSolution:"
)


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
    # 1. Try \boxed{X} (most reliable)
    m = re.search(r"\\boxed\{([A-D])\}", response)
    if m:
        return m.group(1).upper()
    # 2. Try "final answer" / "answer" followed by letter (skip template like (A/B/C/D))
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


# ── Use-level inference ─────────────────────────────────────────────────────

def step_inference(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

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

    # Load existing results
    out_path = V2_RESULTS_DIR / f"use_{model_safe}_d{scale}.json"
    existing = load_json(out_path) if out_path.exists() else []
    done = {(r["family_id"], r["variant"], r["condition"]) for r in existing}

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME"),
    )
    chat = hasattr(tok, "chat_template") and tok.chat_template

    # Detect Qwen3 thinking models — disable thinking for fair comparison
    is_thinking_model = "enable_thinking" in (tok.chat_template or "")

    def apply_template(prompt):
        if chat:
            messages = [{"role": "user", "content": prompt}]
            try:
                kwargs = dict(tokenize=False, add_generation_prompt=True)
                if is_thinking_model:
                    kwargs["enable_thinking"] = False
                return tok.apply_chat_template(messages, **kwargs)
            except:
                pass
        return prompt

    # Build prompts: each family × 3 variants × 3 conditions (CoT, NC, Strict) = 9 prompts
    prompts = []
    meta = []

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
                raw = template.format(problem_block=block)
                final = apply_template(raw)
                prompts.append(final)
                meta.append({
                    "family_id": fid,
                    "category": cat,
                    "variant": var_name,
                    "condition": cond,
                    "model": model,
                    "problem": question,
                    "ground_truth": answer,
                    "ground_truth_letter": correct_letter,
                    "shortcut_strength": task.get("shortcut_strength", 0),
                })

    if not prompts:
        print("All conditions already done.")
        return

    print(f"To infer: {len(prompts)} prompts ({len(prompts)//9} families × 9)")

    # Load vLLM
    llm_kwargs = dict(
        model=model, dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        max_model_len=2048,
    )
    if os.environ.get("HF_HOME"):
        llm_kwargs["download_dir"] = os.environ["HF_HOME"]

    print("Loading vLLM model...")
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=42)

    print(f"Running batch inference...")
    outputs = llm.generate(prompts, sp)

    new_records = []
    for out, m in zip(outputs, meta):
        rec = dict(m)
        rec["raw_response"] = out.outputs[0].text.strip()
        rec["response_token_count"] = len(out.outputs[0].token_ids)
        new_records.append(rec)

    all_records = existing + new_records
    save_json(all_records, out_path)
    print(f"Saved {len(all_records)} records → {out_path}")

    # Summary
    cond_counts = Counter((r["variant"], r["condition"]) for r in all_records)
    print(f"\nRecords by variant × condition:")
    for k, v in sorted(cond_counts.items()):
        print(f"  {k[0]:>20} × {k[1]:<4}: {v}")


# ── Analysis ─────────────────────────────────────────────────────────────────

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

        # Per variant × condition accuracy
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
                print(f"    {cat:<25} strong={acc_s:.1%}  ctrl={acc_c:.1%}  Δ={delta:+.1%}")

    # Save
    out_path = V2_RESULTS_DIR / f"analysis_d{scale}.json"
    save_json(results, out_path)
    print(f"\nSaved analysis → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["inference", "analyze", "judge-inference", "generate-inference"])
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4, 8, 16])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    if args.step == "inference":
        step_inference(args)
    elif args.step == "analyze":
        step_analyze(args)
    elif args.step in ("judge-inference", "generate-inference"):
        print(f"TODO: {args.step} not yet implemented")


if __name__ == "__main__":
    main()

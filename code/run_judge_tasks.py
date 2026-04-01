#!/usr/bin/env python3
"""
SenseMath v2 Judge Task Inference + Analysis (J1 / J2 / J3).

Supports both vLLM local models and OpenAI API.

Usage:
  # vLLM inference
  python run_judge_tasks.py --step inference --task j1 --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4

  # OpenAI API inference
  python run_judge_tasks.py --step inference --task j1 --api --api-model gpt-5-mini --concurrency 20

  # Run all judge tasks at once
  python run_judge_tasks.py --step inference --task all --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4

  # Analyze results
  python run_judge_tasks.py --step analyze
"""

import os
import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
V2_DATA_DIR = SCRIPT_DIR.parent / "benchmark"
V2_RESULTS_DIR = SCRIPT_DIR.parent / "results"

JUDGE_FILES = {
    "j1": V2_DATA_DIR / "judge_j1.json",
    "j2": V2_DATA_DIR / "judge_j2.json",
    "j3": V2_DATA_DIR / "judge_j3.json",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_model_name(model: str) -> str:
    return re.sub(r"[/: ]", "_", model)


def load_json(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_yes_no(response: str) -> str:
    """Extract YES/NO from a J1 response."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Look for explicit YES/NO (case-insensitive)
    m = re.search(r"\b(YES|NO)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def extract_ab(response: str) -> str:
    """Extract A/B from a J3 response."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Look for "answer ... A/B" pattern first
    m = re.search(r"(?:answer|choice)[:\s]*\(?([AB])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Single A or B at end
    m = re.search(r"\b([AB])\b\s*\.?\s*$", text)
    if m:
        return m.group(1).upper()
    # First A or B mention
    m = re.search(r"\b([AB])\b", text)
    if m:
        return m.group(1).upper()
    return ""


def result_path(task: str, model_name: str) -> Path:
    return V2_RESULTS_DIR / f"judge_{task}_{safe_model_name(model_name)}.json"


def build_record(item: Dict, task: str, model: str, raw_response: str, token_count: int) -> Dict:
    """Build a result record from a judge item and its response."""
    rec = {
        "task_id": item["task_id"],
        "task_type": item["task_type"],
        "category": item["category"],
        "model": model,
        "prompt": item["prompt"],
        "raw_response": raw_response,
        "response_token_count": token_count,
    }

    # Variant (J1 has it, J2/J3 may not)
    if "variant" in item:
        rec["variant"] = item["variant"]

    # Ground truth field differs by task
    if task == "j1":
        rec["ground_truth"] = item["ground_truth"]
        rec["predicted"] = extract_yes_no(raw_response)
    elif task == "j2":
        rec["correct_answer"] = item["correct_answer"]
        rec["error_type"] = item.get("error_type", "")
        rec["error_description"] = item.get("error_description", "")
        rec["predicted"] = raw_response.strip()  # full response, manual scoring
    elif task == "j3":
        rec["correct_answer"] = item["correct_answer"]
        rec["predicted"] = extract_ab(raw_response)

    return rec


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_vllm_inference(args, task: str):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model = args.model
    model_safe = safe_model_name(model)
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_path = JUDGE_FILES[task]
    if not ds_path.exists():
        print(f"Dataset not found: {ds_path}")
        return
    items = load_json(ds_path)
    print(f"Loaded {len(items)} items from {ds_path}")

    # Load existing results for resumption
    out_path = result_path(task, model)
    existing = load_json(out_path) if out_path.exists() else []
    done_ids = {r["task_id"] for r in existing}

    # Filter to pending items
    pending = [it for it in items if it["task_id"] not in done_ids]
    if not pending:
        print(f"All {task.upper()} items already done for {model}.")
        return
    print(f"Pending: {len(pending)} / {len(items)}")

    # Load tokenizer for chat template
    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME"),
    )
    chat = hasattr(tok, "chat_template") and tok.chat_template
    is_thinking_model = "enable_thinking" in (tok.chat_template or "")

    def apply_template(prompt: str) -> str:
        if chat:
            messages = [{"role": "user", "content": prompt}]
            try:
                kwargs = dict(tokenize=False, add_generation_prompt=True)
                if is_thinking_model:
                    kwargs["enable_thinking"] = False
                return tok.apply_chat_template(messages, **kwargs)
            except Exception:
                pass
        return prompt

    # Build prompts
    prompts = [apply_template(it["prompt"]) for it in pending]

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

    print(f"Running batch inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sp)

    new_records = []
    for out, item in zip(outputs, pending):
        raw = out.outputs[0].text.strip()
        tok_count = len(out.outputs[0].token_ids)
        rec = build_record(item, task, model, raw, tok_count)
        new_records.append(rec)

    all_records = existing + new_records
    save_json(all_records, out_path)
    print(f"Saved {len(all_records)} records -> {out_path}")


# ── API inference ─────────────────────────────────────────────────────────────

async def call_openai(client, model: str, prompt: str, max_tokens: int,
                      semaphore: asyncio.Semaphore):
    """Make a single OpenAI chat completion call with concurrency control."""
    async with semaphore:
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            # Some models don't support temperature=0
            if "5-mini" not in model and "o1" not in model and "o3" not in model and "o4" not in model:
                kwargs["temperature"] = 0
            response = await client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            token_count = response.usage.completion_tokens if response.usage else 0
            return text.strip(), token_count
        except Exception as e:
            print(f"  API error: {e}")
            return "", 0


async def run_api_inference_async(args, task: str):
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    model = args.api_model
    model_safe = safe_model_name(model)
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_path = JUDGE_FILES[task]
    if not ds_path.exists():
        print(f"Dataset not found: {ds_path}")
        return
    items = load_json(ds_path)
    print(f"Loaded {len(items)} items from {ds_path}")

    # Load existing results for resumption
    out_path = result_path(task, model)
    existing = load_json(out_path) if out_path.exists() else []
    done_ids = {r["task_id"] for r in existing}

    pending = [it for it in items if it["task_id"] not in done_ids]
    if not pending:
        print(f"All {task.upper()} items already done for {model}.")
        return
    print(f"Pending: {len(pending)} / {len(items)}")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async_tasks = [
        call_openai(client, model, it["prompt"], args.max_tokens, semaphore)
        for it in pending
    ]

    print(f"Running {len(async_tasks)} API calls with concurrency={args.concurrency}...")
    results = await asyncio.gather(*async_tasks)

    new_records = []
    for item, (response_text, token_count) in zip(pending, results):
        rec = build_record(item, task, model, response_text, token_count)
        new_records.append(rec)

    all_records = existing + new_records
    save_json(all_records, out_path)
    print(f"Saved {len(all_records)} records -> {out_path}")

    await client.close()


def run_api_inference(args, task: str):
    asyncio.run(run_api_inference_async(args, task))


# ── Inference dispatcher ──────────────────────────────────────────────────────

def step_inference(args):
    tasks = ["j1", "j2", "j3"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  Judge task: {task.upper()}")
        print(f"{'='*60}")

        if args.api:
            run_api_inference(args, task)
        else:
            run_vllm_inference(args, task)


# ── Analysis ──────────────────────────────────────────────────────────────────

def step_analyze(args):
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all judge result files
    j1_files = sorted(V2_RESULTS_DIR.glob("judge_j1_*.json"))
    j2_files = sorted(V2_RESULTS_DIR.glob("judge_j2_*.json"))
    j3_files = sorted(V2_RESULTS_DIR.glob("judge_j3_*.json"))

    if not j1_files and not j2_files and not j3_files:
        print("No judge result files found.")
        return

    print(f"\n{'='*80}")
    print(f"  SenseMath v2 Judge Task Analysis")
    print(f"{'='*80}")

    summary = {}

    # ── J1 Analysis (YES/NO accuracy) ────────────────────────────────────
    if j1_files:
        print(f"\n--- J1: Strategy Appropriateness (YES/NO) ---")
        print(f"  {'Model':<35} {'N':>5} {'Acc%':>7} | "
              f"{'strong':>7} {'weak':>7} {'control':>7}")
        print(f"  {'-'*80}")

        for fpath in j1_files:
            records = load_json(fpath)
            if not records:
                continue
            model = records[0]["model"]
            model_short = model.split("/")[-1][:33]

            total = len(records)
            correct = sum(1 for r in records if r.get("predicted") == r.get("ground_truth"))
            acc = correct / total if total else 0

            # Per-variant accuracy
            variant_acc = {}
            for var in ("strong_shortcut", "weak_shortcut", "control"):
                var_recs = [r for r in records if r.get("variant") == var]
                if var_recs:
                    va = sum(1 for r in var_recs if r.get("predicted") == r.get("ground_truth")) / len(var_recs)
                    variant_acc[var] = va
                else:
                    variant_acc[var] = float("nan")

            print(f"  {model_short:<35} {total:>5} {acc:>6.1%} | "
                  f"{variant_acc.get('strong_shortcut', 0):>6.1%} "
                  f"{variant_acc.get('weak_shortcut', 0):>6.1%} "
                  f"{variant_acc.get('control', 0):>6.1%}")

            # Per-category breakdown
            categories = sorted(set(r["category"] for r in records))
            cat_detail = {}
            for cat in categories:
                cat_recs = [r for r in records if r["category"] == cat]
                cat_acc = sum(1 for r in cat_recs if r.get("predicted") == r.get("ground_truth")) / len(cat_recs)
                cat_detail[cat] = round(cat_acc, 4)

            summary[f"J1|{safe_model_name(model)}"] = {
                "model": model, "task": "J1", "n": total,
                "accuracy": round(acc, 4),
                "variant_accuracy": {k: round(v, 4) for k, v in variant_acc.items()},
                "category_accuracy": cat_detail,
            }

        # Detailed per-category table
        print(f"\n  J1 per-category breakdown:")
        for fpath in j1_files:
            records = load_json(fpath)
            if not records:
                continue
            model_short = records[0]["model"].split("/")[-1][:25]
            categories = sorted(set(r["category"] for r in records))
            print(f"    {model_short}:")
            for cat in categories:
                cat_recs = [r for r in records if r["category"] == cat]
                cat_acc = sum(1 for r in cat_recs if r.get("predicted") == r.get("ground_truth")) / len(cat_recs)
                print(f"      {cat:<30} {len(cat_recs):>4} items  {cat_acc:>6.1%}")

    # ── J2 Analysis (free-text, summary only) ────────────────────────────
    if j2_files:
        print(f"\n--- J2: Error Detection (free-text, manual scoring) ---")
        print(f"  {'Model':<35} {'N':>5} {'AvgLen':>8}")
        print(f"  {'-'*50}")

        for fpath in j2_files:
            records = load_json(fpath)
            if not records:
                continue
            model = records[0]["model"]
            model_short = model.split("/")[-1][:33]
            n = len(records)
            avg_len = sum(len(r.get("raw_response", "")) for r in records) / n if n else 0

            print(f"  {model_short:<35} {n:>5} {avg_len:>7.0f}")

            summary[f"J2|{safe_model_name(model)}"] = {
                "model": model, "task": "J2", "n": n,
                "avg_response_length": round(avg_len, 1),
                "note": "Manual scoring required",
            }

    # ── J3 Analysis (A/B accuracy) ───────────────────────────────────────
    if j3_files:
        print(f"\n--- J3: Pairwise Efficiency (A/B) ---")
        print(f"  {'Model':<35} {'N':>5} {'Acc%':>7}")
        print(f"  {'-'*50}")

        for fpath in j3_files:
            records = load_json(fpath)
            if not records:
                continue
            model = records[0]["model"]
            model_short = model.split("/")[-1][:33]

            total = len(records)
            correct = sum(1 for r in records if r.get("predicted") == r.get("correct_answer"))
            acc = correct / total if total else 0

            print(f"  {model_short:<35} {total:>5} {acc:>6.1%}")

            # Per-category breakdown
            categories = sorted(set(r["category"] for r in records))
            cat_detail = {}
            for cat in categories:
                cat_recs = [r for r in records if r["category"] == cat]
                cat_acc = sum(1 for r in cat_recs if r.get("predicted") == r.get("correct_answer")) / len(cat_recs)
                cat_detail[cat] = round(cat_acc, 4)

            summary[f"J3|{safe_model_name(model)}"] = {
                "model": model, "task": "J3", "n": total,
                "accuracy": round(acc, 4),
                "category_accuracy": cat_detail,
            }

        # Detailed per-category table
        print(f"\n  J3 per-category breakdown:")
        for fpath in j3_files:
            records = load_json(fpath)
            if not records:
                continue
            model_short = records[0]["model"].split("/")[-1][:25]
            categories = sorted(set(r["category"] for r in records))
            print(f"    {model_short}:")
            for cat in categories:
                cat_recs = [r for r in records if r["category"] == cat]
                cat_acc = sum(1 for r in cat_recs if r.get("predicted") == r.get("correct_answer")) / len(cat_recs)
                print(f"      {cat:<30} {len(cat_recs):>4} items  {cat_acc:>6.1%}")

    # Save summary
    out_path = V2_RESULTS_DIR / "judge_analysis.json"
    save_json(summary, out_path)
    print(f"\nSaved analysis -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SenseMath v2 Judge Task Inference + Analysis (J1/J2/J3)")

    parser.add_argument("--step", required=True,
                        choices=["inference", "analyze"])
    parser.add_argument("--task", default="all",
                        choices=["j1", "j2", "j3", "all"],
                        help="Which judge task to run (default: all)")

    # vLLM options
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="HuggingFace model for vLLM")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu-mem", type=float, default=0.90,
                        help="GPU memory utilization for vLLM")

    # API options
    parser.add_argument("--api", action="store_true",
                        help="Use OpenAI API instead of vLLM")
    parser.add_argument("--api-model", default="gpt-5-mini",
                        help="OpenAI model name (used with --api)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests")

    # Shared
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for generation")

    args = parser.parse_args()

    if args.step == "inference":
        step_inference(args)
    elif args.step == "analyze":
        step_analyze(args)


if __name__ == "__main__":
    main()

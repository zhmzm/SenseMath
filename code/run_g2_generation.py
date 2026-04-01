#!/usr/bin/env python3
"""
SenseMath v2 G2: Problem Generation Task.

Prepares 60 prompts (12 per category x 5 categories) and runs them through
either a local vLLM model or the OpenAI API.  Each prompt shows the model
one example pair from d4 and asks it to generate a NEW strong/control pair.

Usage:
  # vLLM inference
  python run_g2_generation.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4

  # OpenAI API inference
  python run_g2_generation.py --api --api-model gpt-5-mini --concurrency 20
"""

import os
import re
import json
import asyncio
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

SCRIPT_DIR = Path(__file__).parent
V2_DATA_DIR = SCRIPT_DIR.parent / "benchmark"
V2_RESULTS_DIR = SCRIPT_DIR.parent / "results"

CATEGORIES = [
    "magnitude_estimation",
    "structural_shortcuts",
    "relative_distance",
    "cancellation_identity",
    "compatible_numbers",
    "equation_reasoning",
    "option_elimination",
    "landmark_comparison",
]

CATEGORY_DESCRIPTIONS = {
    "magnitude_estimation": "Estimate A × B by rounding both factors to nearby powers of 10",
    "structural_shortcuts": "Compute A × B where A is very close to a power of 10, using the distributive law",
    "relative_distance": "Compare two fractions by checking which side of 1/2 each falls on",
    "cancellation_identity": "Evaluate A + B - C where B and C nearly cancel each other",
    "compatible_numbers": "Estimate A × B by rounding to product-friendly pairs like 25×4 or 50×2",
    "equation_reasoning": "Fill in the blank in an equation where algebraic shortcuts (commutativity, cancellation of common terms) reduce the computation to trivial steps",
    "option_elimination": "Compute A × B where the correct answer can be identified from the options by checking a structural feature (trailing digit, parity, or number of digits) that only one option satisfies",
    "landmark_comparison": "Determine whether X% of Y is greater than Z, where X% is very close to a landmark percentage (10%, 25%, 50%) that makes the comparison trivial",
}

PROMPT_TEMPLATE = """\
You are given a category of number-sense problems and one example.
Your task is to generate a NEW problem pair: one "strong_shortcut" version
where a specific mental math shortcut works, and one "control" version
where the same shortcut does NOT work.

Category: {category_name}
Description: {category_description}

Example:
  Strong shortcut: {example_strong_question}
    → The shortcut works because: {example_strong_explanation}
    → Answer: {example_strong_answer}
  Control: {example_control_question}
    → The shortcut does NOT work because: {example_control_explanation}
    → Answer: {example_control_answer}

Now generate a NEW problem pair. The problems must be DIFFERENT from the
example (different numbers). Output in this exact JSON format:

{{
  "strong_shortcut": {{
    "question": "...",
    "math_expression": "...",
    "answer": <number or string>,
    "why_shortcut_works": "..."
  }},
  "control": {{
    "question": "...",
    "math_expression": "...",
    "answer": <number or string>,
    "why_shortcut_fails": "..."
  }}
}}"""

PROMPTS_PER_CATEGORY = 12


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


def parse_generation_json(raw: str) -> Optional[Dict]:
    """Try to parse the model's generation output as JSON.

    Strategy: json.loads first, then regex fallback.
    """
    # Strip think tags if present
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try to extract JSON block (possibly wrapped in ```json ... ```)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text_candidate = json_match.group(1)
    else:
        # Find the outermost { ... }
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text_candidate = brace_match.group(0)
        else:
            text_candidate = text

    # Attempt 1: direct json.loads
    try:
        parsed = json.loads(text_candidate)
        if "strong_shortcut" in parsed and "control" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: regex fallback — extract key fields
    try:
        strong_block = re.search(
            r'"strong_shortcut"\s*:\s*\{(.*?)\}',
            text, re.DOTALL
        )
        control_block = re.search(
            r'"control"\s*:\s*\{(.*?)\}',
            text, re.DOTALL
        )
        if strong_block and control_block:
            def extract_fields(block_text, variant="strong"):
                result = {}
                for key in ["question", "math_expression", "answer",
                            "why_shortcut_works", "why_shortcut_fails"]:
                    m = re.search(
                        rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*?"|\d+(?:\.\d+)?)',
                        block_text
                    )
                    if m:
                        val = m.group(1)
                        if val.startswith('"'):
                            result[key] = json.loads(val)
                        else:
                            try:
                                result[key] = int(val)
                            except ValueError:
                                result[key] = float(val)
                return result

            strong = extract_fields(strong_block.group(1), "strong")
            control = extract_fields(control_block.group(1), "control")
            if strong.get("math_expression") and control.get("math_expression"):
                return {"strong_shortcut": strong, "control": control}
    except Exception:
        pass

    return None


# ── Prompt preparation ────────────────────────────────────────────────────────

def prepare_prompts(seed: int = 42) -> List[Dict]:
    """Build 60 prompts, 12 per category, each with a different example."""
    ds_path = V2_DATA_DIR / "sensemath_v2_d4.json"
    families = load_json(ds_path)

    rng = random.Random(seed)

    prompts = []
    prompt_id = 0

    for cat in CATEGORIES:
        cat_families = [f for f in families if f["category"] == cat]
        # Sample 12 distinct examples
        sampled = rng.sample(cat_families, min(PROMPTS_PER_CATEGORY, len(cat_families)))

        for fam in sampled:
            strong = fam["strong_shortcut"]
            control = fam["control"]

            # Build the question text from math_expression for non-relative_distance
            if cat == "relative_distance":
                strong_q = strong["math_expression"]
                control_q = control["math_expression"]
            else:
                strong_q = strong["math_expression"]
                control_q = control["math_expression"]

            prompt_text = PROMPT_TEMPLATE.format(
                category_name=cat,
                category_description=CATEGORY_DESCRIPTIONS[cat],
                example_strong_question=strong_q,
                example_strong_explanation=strong["ns_shortcut"],
                example_strong_answer=strong["pure_math"]["answer"],
                example_control_question=control_q,
                example_control_explanation=control["ns_shortcut"],
                example_control_answer=control["pure_math"]["answer"],
            )

            prompts.append({
                "prompt_id": f"g2_{cat}_{prompt_id:03d}",
                "category": cat,
                "example_family_id": fam["family_id"],
                "example_strong_expr": strong["math_expression"],
                "example_control_expr": control["math_expression"],
                "prompt": prompt_text,
            })
            prompt_id += 1

    return prompts


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_vllm_inference(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model = args.model
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prompts = prepare_prompts()
    print(f"Prepared {len(prompts)} G2 prompts")

    # Load existing results for resumption
    out_path = V2_RESULTS_DIR / f"g2_{safe_model_name(model)}.json"
    existing = load_json(out_path) if out_path.exists() else []
    done_ids = {r["prompt_id"] for r in existing}

    pending = [p for p in prompts if p["prompt_id"] not in done_ids]
    if not pending:
        print(f"All G2 prompts already done for {model}.")
        return
    print(f"Pending: {len(pending)} / {len(prompts)}")

    # Load tokenizer for chat template
    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME"),
    )
    chat = hasattr(tok, "chat_template") and tok.chat_template
    is_thinking_model = "enable_thinking" in (tok.chat_template or "")

    def apply_template(prompt_text: str) -> str:
        if chat:
            messages = [{"role": "user", "content": prompt_text}]
            try:
                kwargs = dict(tokenize=False, add_generation_prompt=True)
                if is_thinking_model:
                    kwargs["enable_thinking"] = False
                return tok.apply_chat_template(messages, **kwargs)
            except Exception:
                pass
        return prompt_text

    # Build templated prompts
    templated = [apply_template(p["prompt"]) for p in pending]

    # Load vLLM
    llm_kwargs = dict(
        model=model, dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        max_model_len=4096,
    )
    if os.environ.get("HF_HOME"):
        llm_kwargs["download_dir"] = os.environ["HF_HOME"]

    print("Loading vLLM model...")
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=42)

    print(f"Running batch inference on {len(templated)} prompts...")
    outputs = llm.generate(templated, sp)

    new_records = []
    for out, item in zip(outputs, pending):
        raw = out.outputs[0].text.strip()
        tok_count = len(out.outputs[0].token_ids)
        parsed = parse_generation_json(raw)

        rec = {
            "prompt_id": item["prompt_id"],
            "category": item["category"],
            "example_family_id": item["example_family_id"],
            "example_strong_expr": item["example_strong_expr"],
            "example_control_expr": item["example_control_expr"],
            "model": model,
            "raw_response": raw,
            "response_token_count": tok_count,
            "parsed_json": parsed,
            "format_success": parsed is not None,
        }
        new_records.append(rec)

    all_records = existing + new_records
    save_json(all_records, out_path)
    print(f"Saved {len(all_records)} records -> {out_path}")

    # Quick summary
    fmt_ok = sum(1 for r in all_records if r["format_success"])
    print(f"Format success: {fmt_ok}/{len(all_records)} ({fmt_ok/len(all_records):.0%})")


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


async def run_api_inference_async(args):
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    model = args.api_model
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prompts = prepare_prompts()
    print(f"Prepared {len(prompts)} G2 prompts")

    # Load existing results for resumption
    out_path = V2_RESULTS_DIR / f"g2_{safe_model_name(model)}.json"
    existing = load_json(out_path) if out_path.exists() else []
    done_ids = {r["prompt_id"] for r in existing}

    pending = [p for p in prompts if p["prompt_id"] not in done_ids]
    if not pending:
        print(f"All G2 prompts already done for {model}.")
        return
    print(f"Pending: {len(pending)} / {len(prompts)}")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async_tasks = [
        call_openai(client, model, p["prompt"], args.max_tokens, semaphore)
        for p in pending
    ]

    print(f"Running {len(async_tasks)} API calls with concurrency={args.concurrency}...")
    results = await asyncio.gather(*async_tasks)

    new_records = []
    for item, (response_text, token_count) in zip(pending, results):
        parsed = parse_generation_json(response_text)

        rec = {
            "prompt_id": item["prompt_id"],
            "category": item["category"],
            "example_family_id": item["example_family_id"],
            "example_strong_expr": item["example_strong_expr"],
            "example_control_expr": item["example_control_expr"],
            "model": model,
            "raw_response": response_text,
            "response_token_count": token_count,
            "parsed_json": parsed,
            "format_success": parsed is not None,
        }
        new_records.append(rec)

    all_records = existing + new_records
    save_json(all_records, out_path)
    print(f"Saved {len(all_records)} records -> {out_path}")

    # Quick summary
    fmt_ok = sum(1 for r in all_records if r["format_success"])
    print(f"Format success: {fmt_ok}/{len(all_records)} ({fmt_ok/len(all_records):.0%})")

    await client.close()


def run_api_inference(args):
    asyncio.run(run_api_inference_async(args))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SenseMath v2 G2: Problem Generation Task")

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
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens for generation")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  G2: Problem Generation")
    print(f"{'='*60}")

    if args.api:
        run_api_inference(args)
    else:
        run_vllm_inference(args)


if __name__ == "__main__":
    main()

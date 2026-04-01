#!/usr/bin/env python3
"""
SenseMath v2 G2: Deterministic Code-Based Verification.

Reads g2 result files and runs 6 checks per generation:
  1. Strong answer correct
  2. Control answer correct
  3. Strong shortcut exists (category-specific rule)
  4. Control blocks shortcut (category rule returns False)
  5. Family matching (answers within 10x)
  6. Novelty + digit scale

Usage:
  # Verify all g2 result files
  python verify_g2.py

  # Analyze verified results
  python verify_g2.py --analyze
"""

import os
import re
import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from fractions import Fraction

SCRIPT_DIR = Path(__file__).parent
V2_RESULTS_DIR = SCRIPT_DIR.parent / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Safe eval ─────────────────────────────────────────────────────────────────

SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Constant,  # covers Num/Str in Python 3.8+
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
)

# Also allow ast.Num for Python < 3.12
if hasattr(ast, "Num"):
    SAFE_NODES = SAFE_NODES + (ast.Num,)


def safe_eval(expr: str) -> Optional[float]:
    """Safely evaluate an arithmetic expression using AST whitelist."""
    expr = expr.strip()
    if not expr:
        return None
    # Normalize common symbols
    expr = expr.replace('×', '*').replace('÷', '/').replace('\u00d7', '*')
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, SAFE_NODES):
                return None
        return float(eval(compile(tree, '<g2>', 'eval')))
    except Exception:
        return None


# ── Expression parsers ────────────────────────────────────────────────────────

def parse_mult(expression: str) -> Optional[Tuple[float, float]]:
    """Parse 'A * B' or 'A × B' and return (A, B)."""
    expr = expression.strip()
    # Try splitting on * or ×
    for sep in ['*', '\u00d7', 'x', 'X']:
        if sep in expr:
            parts = expr.split(sep, 1)
            if len(parts) == 2:
                try:
                    a = float(parts[0].strip())
                    b = float(parts[1].strip())
                    return (a, b)
                except ValueError:
                    continue
    return None


def parse_abc(expression: str) -> Optional[Tuple[float, float, float]]:
    """Parse 'A + B - C' and return (A, B, C)."""
    expr = expression.strip()
    # Match A + B - C pattern
    m = re.match(r'(-?\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', expr)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return None


def parse_fractions(expression: str) -> Optional[Tuple[float, float]]:
    """Parse 'Which is larger: a/b or c/d?' and return (a/b, c/d) as floats."""
    # Match patterns like "a/b or c/d" or "a/b vs c/d"
    m = re.findall(r'(\d+)\s*/\s*(\d+)', expression)
    if len(m) >= 2:
        try:
            f1 = float(m[0][0]) / float(m[0][1])
            f2 = float(m[1][0]) / float(m[1][1])
            return (f1, f2)
        except (ValueError, ZeroDivisionError):
            pass
    return None


# ── Check 1 & 2: Answer correctness ──────────────────────────────────────────

def check_answer_correct(expression: str, stated_answer, category: str) -> bool:
    """Evaluate expression and compare to stated answer."""
    if not expression or stated_answer is None:
        return False

    try:
        stated = float(stated_answer)
    except (ValueError, TypeError):
        # For relative_distance, answer might be a fraction string like "3/7"
        if category == "relative_distance":
            return check_fraction_answer(expression, stated_answer)
        # For landmark_comparison, answer is "Yes"/"No"
        if category == "landmark_comparison":
            import re as _re
            # Try multiple formats: "X% of Y vs Z", "X% × Y vs Z", "Is X% of Y > Z"
            m = _re.search(r'(\d+)%\s*(?:of|×|\*)\s*(\d+)\s*(?:vs|>|greater\s*than)\s*(\d+)', expression)
            if m:
                pct, base, thresh = int(m.group(1)), int(m.group(2)), int(m.group(3))
                is_greater = base * pct / 100 > thresh
                expected = "Yes" if is_greater else "No"
                return str(stated_answer).strip().lower() == expected.lower()
            return False
        # For equation_reasoning, answer is a number but expression is an equation
        if category == "equation_reasoning":
            import re as _re
            # Parse A+B+___=C+D form, compute ___
            expr = expression.replace('×', '*').replace(' ', '')
            if '=' not in expr or '___' not in expr:
                return False
            lhs, rhs = expr.split('=', 1)
            try:
                lhs_eval = lhs.replace('___', f'({stated_answer})')
                rhs_eval = rhs.replace('___', f'({stated_answer})')
                return abs(eval(lhs_eval) - eval(rhs_eval)) < 0.01
            except:
                return False
        return False

    computed = safe_eval(expression)
    if computed is None:
        return False

    # Allow small relative or absolute tolerance
    if stated == 0:
        return abs(computed) < 0.01
    return abs(computed - stated) / max(1, abs(stated)) < 0.001 or abs(computed - stated) < 0.01


def check_fraction_answer(expression: str, stated_answer) -> bool:
    """For relative_distance: check which fraction is larger."""
    fracs = parse_fractions(expression)
    if fracs is None:
        return False
    f1, f2 = fracs

    # The stated answer should indicate which fraction is larger
    stated_str = str(stated_answer).strip()

    # Try to parse stated answer as a fraction
    frac_matches = re.findall(r'(\d+)\s*/\s*(\d+)', stated_str)
    if frac_matches:
        sa_num = float(frac_matches[0][0])
        sa_den = float(frac_matches[0][1])
        stated_val = sa_num / sa_den

        # Find which parsed fraction matches the stated answer
        all_fracs = re.findall(r'(\d+)\s*/\s*(\d+)', expression)
        if len(all_fracs) >= 2:
            fv1 = float(all_fracs[0][0]) / float(all_fracs[0][1])
            fv2 = float(all_fracs[1][0]) / float(all_fracs[1][1])

            # The stated answer should be the larger fraction
            actual_larger = fv1 if fv1 > fv2 else fv2

            # Check if stated answer matches the actual larger one
            if abs(stated_val - actual_larger) < 0.001:
                return True

    return False


# ── Check 3 & 4: Shortcut existence ──────────────────────────────────────────

def check_shortcut_exists(expression: str, category: str) -> bool:
    """Return True if the expression has a valid category-specific shortcut."""
    if not expression:
        return False

    if category == "magnitude_estimation":
        parsed = parse_mult(expression)
        if parsed is None:
            return False
        a, b = parsed
        # Both factors within 5% of a power of 10
        def _near_power_of_10(x):
            if x <= 0:
                return False
            # Check the nearest power of 10 (could be above or below)
            import math
            d = round(math.log10(x))
            if d < 1:
                d = 1
            target = 10 ** d
            return abs(x - target) / target < 0.05
        return _near_power_of_10(a) and _near_power_of_10(b)

    elif category == "structural_shortcuts":
        parsed = parse_mult(expression)
        if parsed is None:
            return False
        a, b = parsed
        # One factor within ±5 of a power of 10
        def _near_power_pm5(x):
            if x <= 0:
                return False
            import math
            d = round(math.log10(x))
            if d < 1:
                return False
            return abs(x - 10**d) <= 5
        return _near_power_pm5(a) or _near_power_pm5(b)

    elif category == "relative_distance":
        fracs = parse_fractions(expression)
        if fracs is None:
            return False
        f1, f2 = fracs
        # Straddle 1/2 and both within 0.3 of 0.5
        straddle = (f1 < 0.5 and f2 > 0.5) or (f1 > 0.5 and f2 < 0.5)
        near_half = abs(f1 - 0.5) < 0.3 and abs(f2 - 0.5) < 0.3
        return straddle and near_half

    elif category == "cancellation_identity":
        parsed = parse_abc(expression)
        if parsed is None:
            return False
        a, b, c = parsed
        return abs(b - c) <= 15

    elif category == "compatible_numbers":
        parsed = parse_mult(expression)
        if parsed is None:
            return False
        a, b = parsed
        PAIRS = [
            (25, 4), (50, 2), (125, 8), (250, 4), (500, 2),
            (2500, 4), (5000, 2),
        ]
        # Check both orderings
        all_pairs = PAIRS + [(tb, ta) for ta, tb in PAIRS]
        for ta, tb in all_pairs:
            if (abs(a - ta) / max(ta, 1) < 0.05 and
                    abs(b - tb) / max(tb, 1) < 0.10):
                return True
        return False

    elif category == "equation_reasoning":
        # Check if equation has common terms that cancel
        # Expression should be like "A+B+___=B+C+D" or "A×B×C=___×C×A"
        expr = expression.upper().replace('×', '*').replace(' ', '')
        if '=' not in expr:
            return False
        lhs, rhs = expr.split('=', 1)
        # Check for multiplication commutativity (terms appear on both sides)
        if '*' in lhs and '*' in rhs:
            lhs_terms = set(lhs.replace('___', '').split('*'))
            rhs_terms = set(rhs.replace('___', '').split('*'))
            common = lhs_terms & rhs_terms - {'', '___'}
            return len(common) >= 1
        # Check for addition with common terms
        if '+' in lhs or '+' in rhs:
            import re
            lhs_nums = set(re.findall(r'\d+', lhs))
            rhs_nums = set(re.findall(r'\d+', rhs))
            common = lhs_nums & rhs_nums
            return len(common) >= 1
        return False

    elif category == "option_elimination":
        # Strong: the correct product should have a unique structural feature
        # among options (trailing digit, parity, or digit count)
        # We just check that the expression is a valid multiplication
        parsed = parse_mult(expression)
        return parsed is not None

    elif category == "landmark_comparison":
        # Check if percentage is within 3pp of a landmark (10, 25, 50, 75, 100)
        import re
        m = re.search(r'(\d+)%', expression)
        if not m:
            return False
        pct = int(m.group(1))
        landmarks = [10, 25, 50, 75, 100]
        return any(abs(pct - lm) <= 3 for lm in landmarks)

    return False


# ── Check 5: Family matching ─────────────────────────────────────────────────

def check_family_matching(strong_answer, control_answer) -> bool:
    """Answers should be within 10x of each other."""
    try:
        sa = abs(float(strong_answer))
        ca = abs(float(control_answer))
    except (ValueError, TypeError):
        return False

    if sa == 0 or ca == 0:
        return False

    ratio = max(sa, ca) / max(1, min(sa, ca))
    return ratio < 10


# ── Check 6: Novelty + digit scale ───────────────────────────────────────────

def check_novelty(generated_expr: str, example_expr: str) -> bool:
    """Generated numbers must differ from example numbers."""
    if not generated_expr or not example_expr:
        return False
    gen_nums = set(re.findall(r'\d+', generated_expr))
    ex_nums = set(re.findall(r'\d+', example_expr))
    new_nums = gen_nums - ex_nums
    return len(new_nums) >= 1 and generated_expr.strip() != example_expr.strip()


def check_digit_scale(expression: str, category: str, target_digits: int = 4) -> bool:
    """Main operands should have >= target_digits digits."""
    if not expression:
        return False
    numbers = re.findall(r'\d+', expression)
    if category in ("magnitude_estimation", "structural_shortcuts", "cancellation_identity"):
        return any(len(n) >= target_digits for n in numbers)
    elif category == "compatible_numbers":
        return len(numbers) >= 2
    elif category == "relative_distance":
        return any(len(n) >= target_digits for n in numbers)
    return True


def check_novelty_and_scale(gen_strong_expr: str, gen_control_expr: str,
                             ex_strong_expr: str, ex_control_expr: str,
                             category: str) -> bool:
    """Combined novelty + digit scale check."""
    novel_strong = check_novelty(gen_strong_expr, ex_strong_expr)
    novel_control = check_novelty(gen_control_expr, ex_control_expr)
    scale_strong = check_digit_scale(gen_strong_expr, category)
    scale_control = check_digit_scale(gen_control_expr, category)
    return (novel_strong and novel_control and scale_strong and scale_control)


# ── Main verification ─────────────────────────────────────────────────────────

def verify_record(record: Dict) -> Dict:
    """Run all 6 checks on a single generation record."""
    checks = {
        "strong_answer_correct": False,
        "control_answer_correct": False,
        "strong_shortcut_exists": False,
        "control_blocks_shortcut": False,
        "family_matching": False,
        "novelty_digit_scale": False,
    }

    parsed = record.get("parsed_json")
    if not parsed or not record.get("format_success"):
        return {
            "prompt_id": record["prompt_id"],
            "category": record["category"],
            "model": record["model"],
            "format_success": False,
            "format_score": 0,
            "semantic_score": 0,
            "checks": checks,
        }

    category = record["category"]
    strong = parsed.get("strong_shortcut", {})
    control = parsed.get("control", {})

    strong_expr = strong.get("math_expression", "")
    control_expr = control.get("math_expression", "")
    strong_answer = strong.get("answer")
    control_answer = control.get("answer")

    # Check 1: Strong answer correct
    checks["strong_answer_correct"] = check_answer_correct(
        strong_expr, strong_answer, category
    )

    # Check 2: Control answer correct
    checks["control_answer_correct"] = check_answer_correct(
        control_expr, control_answer, category
    )

    # Check 3: Strong shortcut exists
    checks["strong_shortcut_exists"] = check_shortcut_exists(
        strong_expr, category
    )

    # Check 4: Control blocks shortcut
    checks["control_blocks_shortcut"] = not check_shortcut_exists(
        control_expr, category
    )

    # Check 5: Family matching
    checks["family_matching"] = check_family_matching(
        strong_answer, control_answer
    )

    # Check 6: Novelty + digit scale
    checks["novelty_digit_scale"] = check_novelty_and_scale(
        strong_expr, control_expr,
        record.get("example_strong_expr", ""),
        record.get("example_control_expr", ""),
        category,
    )

    semantic_score = sum(1 for v in checks.values() if v)

    return {
        "prompt_id": record["prompt_id"],
        "category": record["category"],
        "model": record["model"],
        "example_family_id": record.get("example_family_id", ""),
        "format_success": True,
        "format_score": 1,
        "semantic_score": semantic_score,
        "checks": checks,
        "strong_expr": strong_expr,
        "control_expr": control_expr,
        "strong_answer": strong_answer,
        "control_answer": control_answer,
    }


def run_verification():
    """Verify all g2 result files."""
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    g2_files = sorted(V2_RESULTS_DIR.glob("g2_*.json"))
    # Exclude already-verified files
    g2_files = [f for f in g2_files if "verified" not in f.name]

    if not g2_files:
        print("No g2 result files found.")
        return

    all_verified = []

    for fpath in g2_files:
        print(f"\nVerifying {fpath.name}...")
        records = load_json(fpath)
        print(f"  {len(records)} records")

        for rec in records:
            result = verify_record(rec)
            all_verified.append(result)

        # Quick summary for this file
        fmt_ok = sum(1 for r in all_verified if r["model"] == records[0].get("model") and r["format_success"])
        total_model = sum(1 for r in all_verified if r["model"] == records[0].get("model"))
        if total_model > 0:
            sem_scores = [r["semantic_score"] for r in all_verified
                         if r["model"] == records[0].get("model") and r["format_success"]]
            avg_sem = sum(sem_scores) / len(sem_scores) if sem_scores else 0
            perfect = sum(1 for s in sem_scores if s == 6)
            print(f"  Format success: {fmt_ok}/{total_model} ({fmt_ok/total_model:.0%})")
            print(f"  Avg semantic score: {avg_sem:.2f}/6")
            print(f"  Perfect (6/6): {perfect}/{len(sem_scores)}")

    out_path = V2_RESULTS_DIR / "g2_verified.json"
    save_json(all_verified, out_path)
    print(f"\nSaved {len(all_verified)} verified records -> {out_path}")


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_analysis():
    """Print per-model pass rates, per-category breakdown, format success."""
    verified_path = V2_RESULTS_DIR / "g2_verified.json"
    if not verified_path.exists():
        print(f"No verified data found at {verified_path}. Run verification first.")
        return

    data = load_json(verified_path)
    if not data:
        print("No verified records found.")
        return

    models = sorted(set(r["model"] for r in data))
    categories = sorted(set(r["category"] for r in data))

    print(f"\n{'='*90}")
    print(f"  G2 Verification Analysis")
    print(f"  {len(data)} records, {len(models)} models, {len(categories)} categories")
    print(f"{'='*90}")

    # ── Per-model summary ─────────────────────────────────────────────────
    print(f"\n--- Per-model summary ---")
    print(f"  {'Model':<35} {'N':>4} {'Fmt%':>6} {'AvgSem':>7} {'Pass6/6':>8} {'Pass%':>6}")
    print(f"  {'-'*70}")

    for model in models:
        recs = [r for r in data if r["model"] == model]
        n = len(recs)
        fmt_ok = sum(1 for r in recs if r["format_success"])
        fmt_pct = fmt_ok / n if n else 0

        sem_recs = [r for r in recs if r["format_success"]]
        if sem_recs:
            avg_sem = sum(r["semantic_score"] for r in sem_recs) / len(sem_recs)
            perfect = sum(1 for r in sem_recs if r["semantic_score"] == 6)
            pass_pct = perfect / len(sem_recs) if sem_recs else 0
        else:
            avg_sem = 0
            perfect = 0
            pass_pct = 0

        model_short = model.split("/")[-1][:33]
        print(f"  {model_short:<35} {n:>4} {fmt_pct:>5.0%} {avg_sem:>7.2f} "
              f"{perfect:>4}/{len(sem_recs):<3} {pass_pct:>5.0%}")

    # ── Per-check pass rates ──────────────────────────────────────────────
    check_names = [
        "strong_answer_correct", "control_answer_correct",
        "strong_shortcut_exists", "control_blocks_shortcut",
        "family_matching", "novelty_digit_scale",
    ]

    print(f"\n--- Per-check pass rates (format-success only) ---")
    header = f"  {'Model':<35}"
    for cn in check_names:
        short = cn[:12]
        header += f" {short:>12}"
    print(header)
    print(f"  {'-'*110}")

    for model in models:
        sem_recs = [r for r in data if r["model"] == model and r["format_success"]]
        if not sem_recs:
            continue
        model_short = model.split("/")[-1][:33]
        line = f"  {model_short:<35}"
        for cn in check_names:
            passed = sum(1 for r in sem_recs if r["checks"].get(cn, False))
            pct = passed / len(sem_recs)
            line += f" {pct:>11.0%}"
        print(line)

    # ── Per-category breakdown ────────────────────────────────────────────
    print(f"\n--- Per-category breakdown (format-success, avg semantic score) ---")
    header = f"  {'Model':<25}"
    for cat in categories:
        cat_short = cat[:15]
        header += f" {cat_short:>15}"
    print(header)
    print(f"  {'-'*105}")

    for model in models:
        model_short = model.split("/")[-1][:23]
        line = f"  {model_short:<25}"
        for cat in categories:
            cat_recs = [r for r in data
                       if r["model"] == model and r["category"] == cat and r["format_success"]]
            if cat_recs:
                avg = sum(r["semantic_score"] for r in cat_recs) / len(cat_recs)
                perfect = sum(1 for r in cat_recs if r["semantic_score"] == 6)
                line += f" {avg:.1f} ({perfect}/{len(cat_recs)})"
            else:
                line += f" {'N/A':>15}"
        print(line)

    # ── Format success per category ───────────────────────────────────────
    print(f"\n--- Format success rates per category ---")
    header = f"  {'Model':<25}"
    for cat in categories:
        cat_short = cat[:15]
        header += f" {cat_short:>15}"
    print(header)
    print(f"  {'-'*105}")

    for model in models:
        model_short = model.split("/")[-1][:23]
        line = f"  {model_short:<25}"
        for cat in categories:
            cat_recs = [r for r in data if r["model"] == model and r["category"] == cat]
            if cat_recs:
                fmt_ok = sum(1 for r in cat_recs if r["format_success"])
                line += f" {fmt_ok}/{len(cat_recs):>11}"
            else:
                line += f" {'N/A':>15}"
        print(line)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SenseMath v2 G2: Deterministic Verification")
    parser.add_argument("--analyze", action="store_true",
                        help="Print analysis of verified results (skip verification)")
    args = parser.parse_args()

    if args.analyze:
        run_analysis()
    else:
        run_verification()
        # Also print analysis after verification
        run_analysis()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SenseMath v2 Dataset Generator
==============================
Generates item-family benchmark with 6 categories × 3 variants (strong/weak/control)
plus Judge and Generate task items.

New in v2:
  - 2 new categories: compatible_numbers, landmark_comparison
  - 3 variants per family: strong_shortcut, weak_shortcut, control
  - Family matching: same answer magnitude, balanced option positions
  - Typed distractors from plausible errors
  - Free-response format for 30% of families
  - Shortcut strength metadata (continuous, category-specific)
  - Judge tasks (J1, J2, J3) and Generate tasks (G1, G2)

Usage:
  # Phase 1: small test
  python generate_dataset_v2.py --phase test --digits 2

  # Phase 2: full benchmark
  python generate_dataset_v2.py --phase full --digits 2 4 8

  # Generate Judge/Generate tasks
  python generate_dataset_v2.py --phase judge --digits 2 4
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

CATEGORIES = [
    "magnitude_estimation",
    "structural_shortcuts",
    "relative_distance",
    "cancellation_identity",
    "compatible_numbers",
    "landmark_comparison",
    "equation_reasoning",
    "option_elimination",
]

_counter = 0


def _next_id(category: str) -> str:
    global _counter
    _counter += 1
    return f"{category}_{_counter:06d}"


# ── Number difficulty helpers ──────────────────────────────────────────────

def _is_hard_number(x: int) -> bool:
    """Return True if x is hard to round/approximate.
    A number is 'hard' when its last two digits fall in [25, 75]
    AND it is not divisible by 10 AND not near any x000 boundary."""
    x = abs(x)
    if x < 100:
        return True  # 2-digit numbers are always "easy" but we can't avoid them at d=2
    if x % 10 == 0:
        return False  # trailing zero → easy to factor out 10
    last2 = x % 100
    if last2 < 25 or last2 > 75:
        return False  # easy to round to nearest 100
    last3 = x % 1000
    if last3 < 25 or last3 > 975:
        return False  # easy to round to nearest 1000
    return True


# ── Distractor generation ──────────────────────────────────────────────────

def _typed_numeric_distractors(
    correct: float,
    category: str,
    shortcut_hint: Optional[Dict] = None,
    n: int = 3,
    as_int: bool = True,
    preserve_last_digit: bool = True,
) -> Tuple[List[str], str, List[str]]:
    """Generate typed distractors from plausible error patterns.
    When preserve_last_digit=True, all distractors share the same trailing
    digit as the correct answer (prevents option-level shortcut leakage).
    Returns (all_options, correct_answer_str, distractor_types)."""
    correct_val = int(round(correct)) if as_int else round(correct, 4)
    last_digit = abs(correct_val) % 10 if as_int else None
    distractors = []
    types = []

    def _add(val, dtype):
        val = int(round(val)) if as_int else round(val, 4)
        if val != correct_val and val not in distractors:
            # If preserving last digit, adjust to match
            if preserve_last_digit and as_int and last_digit is not None:
                diff = (val % 10 - last_digit) % 10
                if diff != 0:
                    val = val - diff  # snap to same last digit
                    if val == correct_val:
                        val += 10  # shift by 10 to keep last digit
            if val != correct_val and val not in distractors:
                distractors.append(val)
                types.append(dtype)

    # Type 1: Nearby with same last digit (multiples of 10 offset)
    for mult in [10, 20, -10, -20, 30, -30]:
        if len(distractors) >= n:
            break
        _add(correct_val + mult, "nearby_same_digit")

    # Type 2: Larger offset (still same last digit)
    for mult in [100, -100, 50, -50, 200, -200]:
        if len(distractors) >= n:
            break
        _add(correct_val + mult, "moderate_offset")

    # Type 3: Magnitude error (×10 or ÷10, preserving last digit)
    if abs(correct_val) >= 100 and len(distractors) < n:
        mag_err = correct_val * 10
        _add(mag_err, "magnitude_error")

    # Fill remaining (max 100 attempts)
    fill_attempts = 0
    while len(distractors) < n and fill_attempts < 100:
        fill_attempts += 1
        offset = random.choice([-1, 1]) * random.randint(1, max(3, abs(correct_val) // 50)) * 10
        _add(correct_val + offset, "random_perturbation")

    distractors = distractors[:n]
    types = types[:n]

    # Combine and shuffle
    all_opts = [correct_val] + distractors
    combined = list(zip(all_opts, ["correct"] + types))
    random.shuffle(combined)
    all_opts, all_types = zip(*combined)

    if as_int:
        opt_strs = [str(int(o)) for o in all_opts]
        ans = str(int(correct_val))
    else:
        opt_strs = [f"{float(o):.4g}" for o in all_opts]
        ans = f"{float(correct_val):.4g}"

    return list(opt_strs), ans, list(all_types)


def _make_yn_options(is_true: bool) -> Tuple[List[str], str]:
    opts = ["Yes", "No", "Cannot be determined", "Need more information"]
    return opts, ("Yes" if is_true else "No")


# ── Task/pair builders ──────────────────────────────────────────────────────

def _task(
    expr: str,
    question: str,
    options: List[str],
    answer: str,
    correct_value: Any,
    explanation: str,
    ns_shortcut: str,
    task_type: str,
    shortcut_strength: float = 0.0,
    distractor_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    t = {
        "task_type": task_type,
        "math_expression": expr,
        "ns_shortcut": ns_shortcut,
        "shortcut_strength": shortcut_strength,
        "pure_math": {
            "question": question,
            "options": options,
            "answer": answer,
            "correct_value": correct_value,
            "explanation": explanation,
        },
    }
    if distractor_types:
        t["distractor_types"] = distractor_types
    return t


def _family(
    category: str,
    strong: Dict,
    weak: Dict,
    control: Dict,
) -> Dict[str, Any]:
    """Build a 3-variant item family."""
    return {
        "family_id": _next_id(category),
        "category": category,
        "strong_shortcut": strong,
        "weak_shortcut": weak,
        "control": control,
    }


def _rand_d_digit(d: int) -> int:
    lo = 10 ** (d - 1)
    hi = 10**d - 1
    return random.randint(lo, hi)


# ── Category generators (each returns a family with 3 variants) ─────────

def _gen_magnitude(digits: int) -> Dict[str, Any]:
    """Magnitude estimation: estimate A × B by rounding to nearby powers."""
    scale = 10 ** (digits - 1)
    round_target = 10**digits

    # Strong shortcut: both numbers very close to round target (±1-2 units)
    da = random.choice([-2, -1, 1, 2]) * max(1, scale // 10)
    db = random.choice([-2, -1, 1, 2]) * max(1, scale // 10)
    a_s = max(10 ** (digits - 1), round_target + da)
    b_s = max(10 ** (digits - 1), round_target + db)
    val_s = a_s * b_s
    strength_s = 1.0 - (abs(a_s - round_target) + abs(b_s - round_target)) / (2 * round_target)
    opts_s, ans_s, dtypes_s = _typed_numeric_distractors(val_s, "magnitude_estimation")
    strong = _task(
        f"{a_s} * {b_s}", f"Which of the following is the best estimate for {a_s} * {b_s}?",
        opts_s, ans_s, val_s,
        f"Round to {round_target}: {a_s}≈{round_target}, {b_s}≈{round_target}.",
        f"Round both to {round_target} and multiply.",
        "strong_shortcut", strength_s, dtypes_s,
    )

    # Weak shortcut: one number close to round target, other near a carry boundary
    # (e.g., 4500 — ambiguous whether to round to 4000 or 5000)
    a_w = round_target + random.choice([-2, -1, 1, 2]) * max(1, scale // 10)
    # b_w near a x500 boundary (halfway between two round thousands)
    half_scale = 10 ** (digits - 1) * 5  # e.g., 5000 for d=4
    b_w = half_scale + random.choice([-1, 1]) * random.randint(0, max(1, scale // 5))
    a_w = max(10 ** (digits - 1), a_w)
    b_w = max(10 ** (digits - 1), b_w)
    val_w = a_w * b_w
    strength_w = 0.5 + 0.3 * (1.0 - abs(a_w - round_target) / round_target)
    opts_w, ans_w, dtypes_w = _typed_numeric_distractors(val_w, "magnitude_estimation")
    weak = _task(
        f"{a_w} * {b_w}", f"Which of the following is the best estimate for {a_w} * {b_w}?",
        opts_w, ans_w, val_w,
        f"One number rounds cleanly, the other less so.",
        f"Round {a_w}≈{round_target}, then adjust.",
        "weak_shortcut", strength_w, dtypes_w,
    )

    # Control: both numbers far from ANY round target (no rounding shortcut)
    # Generate numbers in the middle of the digit range, far from powers of 10
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    mid_lo = lo + (hi - lo) // 3  # middle third of range
    mid_hi = lo + 2 * (hi - lo) // 3
    for _try in range(100):
        a_c = random.randint(mid_lo, mid_hi)
        b_c = random.randint(mid_lo, mid_hi)
        # Ensure not accidentally close to a power of 10
        too_close = False
        for target in [10**d for d in range(digits - 1, digits + 2)]:
            if abs(a_c - target) / target < 0.15 or abs(b_c - target) / target < 0.15:
                too_close = True
                break
        if too_close:
            continue
        if _is_hard_number(a_c) and _is_hard_number(b_c):
            break
    val_c = a_c * b_c
    opts_c, ans_c, dtypes_c = _typed_numeric_distractors(val_c, "magnitude_estimation")
    control = _task(
        f"{a_c} * {b_c}", f"Which of the following is the best estimate for {a_c} * {b_c}?",
        opts_c, ans_c, val_c,
        "Both numbers far from any round target; no clean rounding shortcut.",
        "No clean shortcut; compute precisely.",
        "control", 0.0, dtypes_c,
    )

    return _family("magnitude_estimation", strong, weak, control)


def _gen_structural(digits: int) -> Dict[str, Any]:
    """Structural shortcuts: multiply by near-power-of-10 using distributive law."""
    base = 10**digits

    # Strong: very close to power of 10 (±1 or ±2)
    delta_s = random.choice([-2, -1, 1, 2])
    a_s = base + delta_s
    b_s = _rand_d_digit(digits)
    val_s = a_s * b_s
    strength_s = 1.0 - abs(delta_s) / base
    sign_s = "+" if delta_s > 0 else "-"
    opts_s, ans_s, dtypes_s = _typed_numeric_distractors(val_s, "structural_shortcuts")
    strong = _task(
        f"{a_s} * {b_s}", f"What is {a_s} * {b_s}?",
        opts_s, ans_s, val_s,
        f"Distributive: ({base}{sign_s}{abs(delta_s)})*{b_s} = {base}*{b_s} {sign_s} {abs(delta_s)}*{b_s}.",
        f"Use ({base}{sign_s}{abs(delta_s)})*m = {base}*m {sign_s} {abs(delta_s)}*m.",
        "strong_shortcut", strength_s, dtypes_s,
    )

    # Weak: correction requires 2-digit multiplication (±15 to ±50)
    delta_w = random.choice([-1, 1]) * random.randint(15, 50)
    a_w = base + delta_w
    b_w = _rand_d_digit(digits)
    val_w = a_w * b_w
    strength_w = max(0.3, 1.0 - abs(delta_w) / (base * 0.05))
    sign_w = "+" if delta_w > 0 else "-"
    opts_w, ans_w, dtypes_w = _typed_numeric_distractors(val_w, "structural_shortcuts")
    weak = _task(
        f"{a_w} * {b_w}", f"What is {a_w} * {b_w}?",
        opts_w, ans_w, val_w,
        f"Distributive still works but correction term is larger.",
        f"Use ({base}{sign_w}{abs(delta_w)})*m, correction is {abs(delta_w)}*{b_w}.",
        "weak_shortcut", strength_w, dtypes_w,
    )

    # Control: NEITHER factor near any power of 10 (no distributive shortcut)
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    mid_lo = lo + (hi - lo) // 3
    mid_hi = lo + 2 * (hi - lo) // 3
    for _try in range(100):
        a_c = random.randint(mid_lo, mid_hi)
        b_c = random.randint(mid_lo, mid_hi)
        too_close = False
        for target in [10**d for d in range(digits - 1, digits + 2)]:
            if abs(a_c - target) / target < 0.15 or abs(b_c - target) / target < 0.15:
                too_close = True
                break
        if too_close:
            continue
        if _is_hard_number(a_c) and _is_hard_number(b_c):
            break
    val_c = a_c * b_c
    opts_c, ans_c, dtypes_c = _typed_numeric_distractors(val_c, "structural_shortcuts")
    control = _task(
        f"{a_c} * {b_c}", f"What is {a_c} * {b_c}?",
        opts_c, ans_c, val_c,
        "Neither factor near a power of 10; no clean decomposition.",
        "No clean shortcut; compute precisely.",
        "control", 0.0, dtypes_c,
    )

    return _family("structural_shortcuts", strong, weak, control)


def _fraction_straddle(den_digits, bm_num, bm_den):
    """Generate two fractions straddling a benchmark value.
    Ensures each fraction is at least 2% away from the benchmark for d>=4."""
    lo = 10 ** (den_digits - 1)
    hi = 10**den_digits - 1
    bm_val = bm_num / bm_den
    min_gap = max(2, lo // 50)  # at least 2% of denominator away from benchmark
    for _try in range(100):
        d1 = random.randint(lo, hi)
        d2 = random.randint(lo, hi)
        # n1 < bm_val * d1 (below benchmark)
        n1_max = max(1, math.floor(d1 * bm_val) - min_gap)
        n1_lo = max(1, n1_max - max(2, d1 // 50))
        n1 = random.randint(n1_lo, n1_max) if n1_lo <= n1_max else max(1, n1_max)
        # n2 > bm_val * d2 (above benchmark)
        n2_min = min(d2 - 1, math.ceil(d2 * bm_val) + min_gap)
        n2_hi = min(d2 - 1, n2_min + max(2, d2 // 50))
        n2 = random.randint(n2_min, n2_hi) if n2_min <= n2_hi else min(d2 - 1, n2_min)
        # Verify straddle with meaningful gap
        v1, v2 = n1 / d1, n2 / d2
        if v1 < bm_val < v2 and abs(v1 - bm_val) > 0.01 and abs(v2 - bm_val) > 0.01:
            return (n1, d1), (n2, d2)
    # Fallback: deterministic safe straddle
    d1 = random.randint(lo, hi)
    d2 = random.randint(lo, hi)
    n1 = max(1, round(d1 * (bm_val - 0.03)))
    n2 = min(d2 - 1, round(d2 * (bm_val + 0.03)))
    return (n1, d1), (n2, d2)


def _gen_relative(digits: int) -> Dict[str, Any]:
    """Relative distance: compare fractions using benchmark comparison."""
    def make_label_options(correct, f1, f2, digits):
        """Generate options: the two fractions + 2 plausible random fractions.
        Distractors must have decimal values within ±0.1 of the correct answer
        to prevent magnitude-based elimination."""
        lo = 10 ** (digits - 1)
        hi = 10**digits - 1
        opts = [correct]
        other = f1 if correct == f2 else f2
        if other not in opts:
            opts.append(other)
        # Target decimal value of the correct answer
        cn, cd = correct.split('/')
        target_val = int(cn) / int(cd)
        # Generate distractors with similar decimal value (±0.05 to ±0.1)
        attempts = 0
        while len(opts) < 4 and attempts < 100:
            attempts += 1
            nd = random.randint(lo, hi)
            # numerator that gives decimal value near target ±0.05-0.1
            offset = random.uniform(-0.1, 0.1)
            nn = max(1, min(nd - 1, round(nd * (target_val + offset))))
            cand = f"{nn}/{nd}"
            cand_val = nn / nd
            if cand not in opts and abs(cand_val - target_val) <= 0.12:
                opts.append(cand)
        opts = opts[:4]
        random.shuffle(opts)
        return opts, correct

    # Strong: fractions clearly straddle 1/2
    (n1, d1), (n2, d2) = _fraction_straddle(digits, 1, 2)
    f1, f2 = f"{n1}/{d1}", f"{n2}/{d2}"
    v1, v2 = n1/d1, n2/d2
    larger_s = f2 if v2 > v1 else f1
    strength_s = abs(v1 - 0.5) + abs(v2 - 0.5)
    opts_s, ans_s = make_label_options(larger_s, f1, f2, digits)
    strong = _task(
        f"Which is larger: {f1} or {f2}?", f"Which is larger: {f1} or {f2}?",
        opts_s, ans_s, max(v1, v2),
        "Both straddle 1/2; compare to benchmark.",
        "One is below 1/2, the other above; no computation needed.",
        "strong_shortcut", min(1.0, strength_s),
    )

    # Weak: fractions straddle 1/2 but both very close (0.49 vs 0.52)
    # The benchmark shortcut still works but the margin is tight
    lo = 10 ** (digits - 1)
    hi = 10**digits - 1
    d3 = random.randint(lo, hi)
    n3 = max(1, round(d3 * random.uniform(0.485, 0.498)))  # just below 1/2
    d4 = random.randint(lo, hi)
    n4 = max(1, round(d4 * random.uniform(0.502, 0.515)))  # just above 1/2
    f3, f4 = f"{n3}/{d3}", f"{n4}/{d4}"
    v3, v4 = n3/d3, n4/d4
    # Ensure they actually straddle
    if v3 >= 0.5:
        n3 = max(1, int(d3 * 0.49))
        v3 = n3/d3
    if v4 <= 0.5:
        n4 = min(d4-1, int(d4 * 0.51) + 1)
        v4 = n4/d4
    f3, f4 = f"{n3}/{d3}", f"{n4}/{d4}"
    larger_w = f4 if v4 > v3 else f3
    strength_w = abs(v3 - 0.5) + abs(v4 - 0.5)
    opts_w, ans_w = make_label_options(larger_w, f3, f4, digits)
    weak = _task(
        f"Which is larger: {f3} or {f4}?", f"Which is larger: {f3} or {f4}?",
        opts_w, ans_w, max(v3, v4),
        "One is very close to 1/2, shortcut harder to apply.",
        "Compare to 1/2, but margin is tight.",
        "weak_shortcut", min(1.0, strength_w),
    )

    # Control: both on same side of 1/2
    d5 = random.randint(lo, hi)
    d6 = random.randint(lo, hi)
    n5 = max(1, math.ceil(d5 * 0.55))
    n6 = max(1, math.ceil(d6 * 0.58))
    n5, n6 = min(n5, d5-1), min(n6, d6-1)
    if abs(n5/d5 - n6/d6) < 1e-4:
        n6 = min(d6-1, n6+1)
    f5, f6 = f"{n5}/{d5}", f"{n6}/{d6}"
    v5, v6 = n5/d5, n6/d6
    larger_c = f6 if v6 > v5 else f5
    opts_c, ans_c = make_label_options(larger_c, f5, f6, digits)
    control = _task(
        f"Which is larger: {f5} or {f6}?", f"Which is larger: {f5} or {f6}?",
        opts_c, ans_c, max(v5, v6),
        "Both above 1/2; benchmark comparison won't help.",
        "Both on same side; must cross-multiply.",
        "control", 0.0,
    )

    return _family("relative_distance", strong, weak, control)


def _gen_cancellation(digits: int) -> Dict[str, Any]:
    """Cancellation/identity: A + B - C where B ≈ C."""
    lower = 10 ** (digits - 1)
    upper = 10**digits - 1

    a = _rand_d_digit(digits)
    b = random.randint(lower + 5, upper - 5)

    # Strong: B and C differ by 1-2
    offset_s = random.choice([-2, -1, 1, 2])
    c_s = b - offset_s
    val_s = a + b - c_s
    strength_s = 1.0 - abs(offset_s) / b
    opts_s, ans_s, dtypes_s = _typed_numeric_distractors(val_s, "cancellation_identity")
    strong = _task(
        f"{a} + {b} - {c_s}", f"Evaluate: {a} + {b} - {c_s}",
        opts_s, ans_s, val_s,
        f"B-C = {offset_s}; result ≈ A + offset.",
        f"Near-cancellation: {b} - {c_s} = {offset_s}, so answer ≈ {a} + {offset_s}.",
        "strong_shortcut", strength_s, dtypes_s,
    )

    # Weak: B and C differ by 2-5% of B (meaningful but manageable offset)
    pct = random.uniform(0.02, 0.05)
    offset_w = random.choice([-1, 1]) * max(10, int(b * pct))
    c_w = b - offset_w
    c_w = max(lower, min(upper, c_w))
    val_w = a + b - c_w
    strength_w = 1.0 - abs(offset_w) / b
    opts_w, ans_w, dtypes_w = _typed_numeric_distractors(val_w, "cancellation_identity")
    weak = _task(
        f"{a} + {b} - {c_w}", f"Evaluate: {a} + {b} - {c_w}",
        opts_w, ans_w, val_w,
        f"B-C = {b - c_w}; still a moderate shortcut.",
        f"Near-cancellation: {b} - {c_w} = {b - c_w}, offset is manageable.",
        "weak_shortcut", max(0, strength_w), dtypes_w,
    )

    # Control: B and C differ by a lot
    min_gap = max(40, int(abs(b) * 0.3))
    delta = random.randint(min_gap, min_gap + max(50, b // 3))
    c_c = b + random.choice([-1, 1]) * delta
    if c_c < lower:
        c_c = b + delta
    val_c = a + b - c_c
    opts_c, ans_c, dtypes_c = _typed_numeric_distractors(val_c, "cancellation_identity")
    control = _task(
        f"{a} + {b} - {c_c}", f"Evaluate: {a} + {b} - {c_c}",
        opts_c, ans_c, val_c,
        "Large gap between B and C; no cancellation shortcut.",
        "No clean shortcut; compute precisely.",
        "control", 0.0, dtypes_c,
    )

    return _family("cancellation_identity", strong, weak, control)


def _gen_compatible(digits: int) -> Dict[str, Any]:
    """Compatible numbers: round to product-friendly pairs (25×4, 50×2, 125×8)."""
    # Compatible pairs at different scales
    if digits <= 2:
        targets = [(25, 4), (50, 2), (25, 8)]
    elif digits <= 4:
        targets = [(2500, 4000), (5000, 2000), (1250, 8000)]
    else:
        targets = [(25000, 4000), (50000, 2000), (12500, 8000)]

    t_a, t_b = random.choice(targets)

    # Strong: very close to compatible pair
    da = random.choice([-1, 1]) * random.randint(0, max(1, t_a // 50))
    db = random.choice([-1, 1]) * random.randint(0, max(1, t_b // 10))
    a_s = t_a + da
    b_s = t_b + db
    val_s = a_s * b_s
    strength_s = 1.0 - (abs(da)/t_a + abs(db)/t_b) / 2
    opts_s, ans_s, dtypes_s = _typed_numeric_distractors(val_s, "compatible_numbers")
    strong = _task(
        f"{a_s} * {b_s}", f"Estimate {a_s} × {b_s}.",
        opts_s, ans_s, val_s,
        f"Round to compatible pair {t_a}×{t_b} = {t_a*t_b}.",
        f"Round to {t_a}×{t_b} for easy mental computation.",
        "strong_shortcut", strength_s, dtypes_s,
    )

    # Weak: one number close to target, other moderately off
    da_w = random.choice([-1, 1]) * random.randint(0, max(1, t_a // 50))
    db_w = random.choice([-1, 1]) * random.randint(max(1, t_b // 5), max(2, t_b // 3))
    a_w = t_a + da_w
    b_w = t_b + db_w
    val_w = a_w * b_w
    strength_w = 1.0 - (abs(da_w)/t_a + abs(db_w)/t_b) / 2
    opts_w, ans_w, dtypes_w = _typed_numeric_distractors(val_w, "compatible_numbers")
    weak = _task(
        f"{a_w} * {b_w}", f"Estimate {a_w} × {b_w}.",
        opts_w, ans_w, val_w,
        f"One number rounds to {t_a}, other is farther from {t_b}.",
        f"Compatible pair shortcut works but less precise.",
        "weak_shortcut", max(0, strength_w), dtypes_w,
    )

    # Control: two random d-digit "hard" numbers far from any compatible pair
    lo_d = 10 ** (digits - 1)
    hi_d = 10**digits - 1
    _compat_anchors = [25,50,125,250,500,1000,1250,2000,2500,4000,5000,8000,25000,50000]
    for _try in range(100):
        a_c = random.randint(lo_d, hi_d)
        b_c = random.randint(lo_d, hi_d)
        if not (_is_hard_number(a_c) and _is_hard_number(b_c)):
            continue
        near_anchor = False
        for anch in _compat_anchors:
            if abs(a_c - anch) <= max(2, anch // 20) or abs(b_c - anch) <= max(2, anch // 20):
                near_anchor = True
                break
        if not near_anchor:
            break
    val_c = a_c * b_c
    opts_c, ans_c, dtypes_c = _typed_numeric_distractors(val_c, "compatible_numbers")
    control = _task(
        f"{a_c} * {b_c}", f"Estimate {a_c} × {b_c}.",
        opts_c, ans_c, val_c,
        "Not near any compatible pair; full multiplication needed.",
        "No clean shortcut; compute precisely.",
        "control", 0.0, dtypes_c,
    )

    return _family("compatible_numbers", strong, weak, control)


def _gen_landmark(digits: int) -> Dict[str, Any]:
    """Landmark comparison: anchor percentage/fraction calculation to known landmark."""
    base = _rand_d_digit(digits) * 10  # base number (e.g., 800)
    threshold = random.randint(max(10, base // 5), max(50, base // 2))

    # Strong: percentage is very close to a landmark (50%, 25%, 10%)
    landmark_pct = random.choice([50, 25, 10])
    actual_pct = landmark_pct + random.choice([-1, 1]) * random.randint(0, 2)
    actual_val = base * actual_pct / 100
    is_greater = actual_val > threshold
    strength_s = 1.0 - abs(actual_pct - landmark_pct) / 100
    opts_s, ans_s = _make_yn_options(is_greater)
    strong = _task(
        f"{actual_pct}% of {base} vs {threshold}",
        f"Is {actual_pct}% of {base} greater than {threshold}?",
        opts_s, ans_s, actual_val,
        f"{actual_pct}% ≈ {landmark_pct}%. {landmark_pct}% of {base} = {base * landmark_pct // 100}. Compare to {threshold}.",
        f"Use {landmark_pct}% as landmark: {landmark_pct}% of {base} = {base * landmark_pct // 100}.",
        "strong_shortcut", strength_s,
    )

    # Weak: percentage near landmark but farther off
    actual_pct_w = landmark_pct + random.choice([-1, 1]) * random.randint(3, 7)
    actual_pct_w = max(1, min(99, actual_pct_w))
    actual_val_w = base * actual_pct_w / 100
    is_greater_w = actual_val_w > threshold
    strength_w = 1.0 - abs(actual_pct_w - landmark_pct) / 100
    opts_w, ans_w = _make_yn_options(is_greater_w)
    weak = _task(
        f"{actual_pct_w}% of {base} vs {threshold}",
        f"Is {actual_pct_w}% of {base} greater than {threshold}?",
        opts_w, ans_w, actual_val_w,
        f"{actual_pct_w}% is near {landmark_pct}% but requires adjustment.",
        f"Use {landmark_pct}% as approximate anchor, then adjust.",
        "weak_shortcut", max(0, strength_w),
    )

    # Control: percentage far from any landmark
    ctrl_pct = random.choice([37, 43, 63, 83, 87])
    actual_val_c = base * ctrl_pct / 100
    is_greater_c = actual_val_c > threshold
    opts_c, ans_c = _make_yn_options(is_greater_c)
    control = _task(
        f"{ctrl_pct}% of {base} vs {threshold}",
        f"Is {ctrl_pct}% of {base} greater than {threshold}?",
        opts_c, ans_c, actual_val_c,
        f"{ctrl_pct}% is not near any standard landmark; compute directly.",
        "No clean shortcut; compute precisely.",
        "control", 0.0,
    )

    return _family("landmark_comparison", strong, weak, control)


def _gen_equation(digits: int) -> Dict[str, Any]:
    """Equation reasoning: use algebraic properties instead of computing."""

    # ── Strong shortcut: equation solved by recognising algebraic property ──
    subtype = random.choice(["add_commute", "mul_commute", "distributive", "balance"])

    if subtype == "add_commute":
        # A + B + ___ = B + C + A  →  answer is C
        A = _rand_d_digit(digits)
        B = _rand_d_digit(digits)
        C = _rand_d_digit(digits)
        expr_s = f"{A} + {B} + ___ = {B} + {C} + {A}"
        question_s = f"What value fills the blank?  {expr_s}"
        correct_s = C
        explanation_s = (f"A and B appear on both sides and cancel; blank = C = {C}.")
        ns_hint_s = "Recognize commutativity/distributive law — matching terms cancel."
    elif subtype == "mul_commute":
        # A × B × C = ___ × C × A  →  answer is B
        A = _rand_d_digit(digits)
        B = _rand_d_digit(digits)
        C = _rand_d_digit(digits)
        expr_s = f"{A} × {B} × {C} = ___ × {C} × {A}"
        question_s = f"What value fills the blank?  {expr_s}"
        correct_s = B
        explanation_s = (f"A and C appear on both sides; blank = B = {B}.")
        ns_hint_s = "Recognize commutativity/distributive law — matching terms cancel."
    elif subtype == "distributive":
        # A × (B + C) = A × ___ + A × C  →  answer is B
        A = _rand_d_digit(digits)
        B = _rand_d_digit(digits)
        C = _rand_d_digit(digits)
        expr_s = f"{A} × ({B} + {C}) = {A} × ___ + {A} × {C}"
        question_s = f"What value fills the blank?  {expr_s}"
        correct_s = B
        explanation_s = (f"Distributive law: A×(B+C) = A×B + A×C; blank = B = {B}.")
        ns_hint_s = "Recognize commutativity/distributive law — matching terms cancel."
    else:  # balance
        # A + B = ___ + C  →  answer = A + B - C
        A = _rand_d_digit(digits)
        B = _rand_d_digit(digits)
        C = _rand_d_digit(digits)
        correct_s = A + B - C
        expr_s = f"{A} + {B} = ___ + {C}"
        question_s = f"What value fills the blank?  {expr_s}"
        explanation_s = (f"Blank = A + B − C = {A} + {B} − {C} = {correct_s}.")
        ns_hint_s = "Recognize commutativity/distributive law — matching terms cancel."

    opts_s, ans_s, dtypes_s = _typed_numeric_distractors(correct_s, "equation_reasoning")
    strong = _task(
        expr_s, question_s,
        opts_s, ans_s, correct_s,
        explanation_s, ns_hint_s,
        "strong_shortcut", 1.0, dtypes_s,
    )

    # ── Weak shortcut: partial match, some terms cancel but not all ──────
    A2 = _rand_d_digit(digits)
    B2 = _rand_d_digit(digits)
    C2 = _rand_d_digit(digits)
    D2 = _rand_d_digit(digits)
    # A + B + ___ = B + C + D  →  blank = C + D − A  (B cancels, but A→D don't)
    correct_w = B2 + C2 + D2 - A2 - B2  # simplifies to C2 + D2 - A2
    correct_w = C2 + D2 - A2
    expr_w = f"{A2} + {B2} + ___ = {B2} + {C2} + {D2}"
    question_w = f"What value fills the blank?  {expr_w}"
    explanation_w = (f"B cancels; blank = C + D − A = {C2} + {D2} − {A2} = {correct_w}.")
    opts_w, ans_w, dtypes_w = _typed_numeric_distractors(correct_w, "equation_reasoning")
    weak = _task(
        expr_w, question_w,
        opts_w, ans_w, correct_w,
        explanation_w,
        "Partial cancellation: B cancels but remaining terms must be computed.",
        "weak_shortcut", 0.5, dtypes_w,
    )

    # ── Control: no matching terms, must compute ─────────────────────────
    A3 = _rand_d_digit(digits)
    B3 = _rand_d_digit(digits)
    D3 = _rand_d_digit(digits)
    E3 = _rand_d_digit(digits)
    correct_c = D3 + E3 - A3 - B3
    expr_c = f"{A3} + {B3} + ___ = {D3} + {E3}"
    question_c = f"What value fills the blank?  {expr_c}"
    explanation_c = (f"No matching terms; blank = {D3} + {E3} − {A3} − {B3} = {correct_c}.")
    opts_c, ans_c, dtypes_c = _typed_numeric_distractors(correct_c, "equation_reasoning")
    control = _task(
        expr_c, question_c,
        opts_c, ans_c, correct_c,
        explanation_c,
        "No algebraic shortcut; compute precisely.",
        "control", 0.0, dtypes_c,
    )

    return _family("equation_reasoning", strong, weak, control)


def _gen_option_elimination(digits: int) -> Dict[str, Any]:
    """Option elimination: use number sense on OPTIONS to eliminate wrong answers."""

    # ── Strong shortcut: correct answer has a unique detectable feature ───
    subtype = random.choice(["trailing_zero", "magnitude", "parity", "div_by_3"])

    if subtype == "trailing_zero":
        # One factor is multiple of 5, the other is even → product ends in 0
        lo, hi = 10 ** (digits - 1), 10**digits - 1
        a_s = random.randrange(lo // 5 * 5, hi, 5)  # multiple of 5
        if a_s < lo:
            a_s = lo + (5 - lo % 5) % 5
        b_s = random.randrange(lo if lo % 2 == 0 else lo + 1, hi, 2)  # even
        correct_s = a_s * b_s
        # Distractors that do NOT end in 0
        dists = []
        for offset in [1, -1, 3, -3, 7, -7, 11, -11]:
            c = correct_s + offset
            if c > 0 and c % 10 != 0 and c not in dists:
                dists.append(c)
            if len(dists) >= 3:
                break
        while len(dists) < 3:
            c = correct_s + random.choice([-1, 1]) * random.randint(1, 9)
            if c > 0 and c % 10 != 0 and c not in dists:
                dists.append(c)
        feature_desc = "ends in 0 (5 × even)"
        ns_hint_s = "Check trailing digit / parity / magnitude — only one option matches."

    elif subtype == "magnitude":
        # Only one option has the right number of digits
        a_s = _rand_d_digit(digits)
        b_s = _rand_d_digit(digits)
        correct_s = a_s * b_s
        n_digits_correct = len(str(abs(correct_s)))
        dists = []
        # Make distractors with WRONG number of digits
        for _ in range(20):
            factor = random.choice([10, 100]) if n_digits_correct > 2 else 10
            sign = random.choice([-1, 1])
            c = correct_s + sign * correct_s // factor * random.randint(8, 12)
            if c > 0 and len(str(c)) != n_digits_correct and c not in dists and c != correct_s:
                dists.append(c)
            if len(dists) >= 3:
                break
        # Fill with different-magnitude values if needed
        while len(dists) < 3:
            c = correct_s * random.choice([10, 100]) + random.randint(1, 99)
            if c != correct_s and c not in dists:
                dists.append(c)
        feature_desc = f"has {n_digits_correct} digits"
        ns_hint_s = "Check trailing digit / parity / magnitude — only one option matches."

    elif subtype == "parity":
        # odd × odd = odd; distractors are even
        lo, hi = 10 ** (digits - 1), 10**digits - 1
        a_s = random.randrange(lo if lo % 2 == 1 else lo + 1, hi, 2)  # odd
        b_s = random.randrange(lo if lo % 2 == 1 else lo + 1, hi, 2)  # odd
        correct_s = a_s * b_s  # odd
        dists = []
        for offset in [1, -1, 3, -3, 5, -5, 7, -7]:
            c = correct_s + offset
            if c > 0 and c % 2 == 0 and c not in dists:
                dists.append(c)
            if len(dists) >= 3:
                break
        while len(dists) < 3:
            c = correct_s + random.choice([-1, 1]) * random.randint(1, 20) * 2
            if c > 0 and c % 2 == 0 and c not in dists:
                dists.append(c)
        feature_desc = "odd (odd × odd)"
        ns_hint_s = "Check trailing digit / parity / magnitude — only one option matches."

    else:  # div_by_3
        # Both factors divisible by 3 → product divisible by 9
        lo, hi = 10 ** (digits - 1), 10**digits - 1
        a_s = random.randint(lo // 3, hi // 3) * 3
        if a_s < lo:
            a_s = lo + (3 - lo % 3) % 3
        b_s = random.randint(lo // 3, hi // 3) * 3
        if b_s < lo:
            b_s = lo + (3 - lo % 3) % 3
        correct_s = a_s * b_s  # divisible by 9
        dists = []
        for offset in [1, -1, 2, -2, 4, -4, 5, -5, 7, -7]:
            c = correct_s + offset
            if c > 0 and c % 9 != 0 and c not in dists:
                dists.append(c)
            if len(dists) >= 3:
                break
        while len(dists) < 3:
            c = correct_s + random.randint(1, 8)
            if c > 0 and c % 9 != 0 and c not in dists:
                dists.append(c)
        feature_desc = "divisible by 9 (both factors divisible by 3)"
        ns_hint_s = "Check trailing digit / parity / magnitude — only one option matches."

    # Build strong options (correct + 3 distractors, shuffled)
    all_opts_s = [correct_s] + dists[:3]
    combined_s = list(zip(all_opts_s, ["correct", "distractor", "distractor", "distractor"]))
    random.shuffle(combined_s)
    all_opts_s, all_types_s = zip(*combined_s)
    opt_strs_s = [str(int(o)) for o in all_opts_s]
    ans_str_s = str(int(correct_s))

    strong = _task(
        f"{a_s} × {b_s}", f"What is {a_s} × {b_s}?",
        list(opt_strs_s), ans_str_s, correct_s,
        f"Product {feature_desc}; only one option matches.",
        ns_hint_s,
        "strong_shortcut", 1.0, list(all_types_s),
    )

    # ── Control: all options share the same structural features ───────────
    # Both factors odd → product odd, all distractors also odd
    # Both must be "hard" numbers (not near round values, not easy to decompose)
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    for _try in range(100):
        a_c = random.randrange(lo if lo % 2 == 1 else lo + 1, hi, 2)
        b_c = random.randrange(lo if lo % 2 == 1 else lo + 1, hi, 2)
        if _is_hard_number(a_c) and _is_hard_number(b_c):
            break
    correct_c = a_c * b_c
    n_dig_c = len(str(abs(correct_c)))
    # Distractors must share the last 50% of digits with the correct answer
    # to prevent option-level elimination shortcuts
    half_digits = max(1, n_dig_c // 2)
    suffix_mod = 10 ** half_digits
    correct_suffix = correct_c % suffix_mod
    prefix_unit = suffix_mod  # offset must be a multiple of this to preserve suffix
    dists_c = []
    for attempts in range(200):
        offset = random.choice([-1, 1]) * random.randint(1, max(2, correct_c // (prefix_unit * 5))) * prefix_unit
        c = correct_c + offset
        if (c > 0 and c != correct_c and c not in dists_c
                and c % suffix_mod == correct_suffix  # same last 50% digits
                and len(str(abs(c))) == n_dig_c):  # same magnitude
            dists_c.append(c)
        if len(dists_c) >= 3:
            break
    # Fill remaining with small prefix offsets
    fill_step = 1
    while len(dists_c) < 3:
        c = correct_c + fill_step * prefix_unit
        if c > 0 and c != correct_c and c not in dists_c and len(str(abs(c))) == n_dig_c:
            dists_c.append(c)
        fill_step += 1
        if fill_step > 20:
            c = correct_c - fill_step * prefix_unit
            if c > 0 and c != correct_c and c not in dists_c:
                dists_c.append(c)

    all_opts_c = [correct_c] + dists_c[:3]
    combined_c = list(zip(all_opts_c, ["correct", "distractor", "distractor", "distractor"]))
    random.shuffle(combined_c)
    all_opts_c, all_types_c = zip(*combined_c)
    opt_strs_c = [str(int(o)) for o in all_opts_c]
    ans_str_c = str(int(correct_c))

    control = _task(
        f"{a_c} × {b_c}", f"What is {a_c} × {b_c}?",
        list(opt_strs_c), ans_str_c, correct_c,
        "All options have same structural features; must compute precisely.",
        "All options have same structural features; must compute precisely.",
        "control", 0.0, list(all_types_c),
    )

    # ── Weak shortcut: feature present but shared by 2 options ───────────
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    a_w = random.randrange(lo // 5 * 5, hi, 5)
    if a_w < lo:
        a_w = lo + (5 - lo % 5) % 5
    b_w = random.randrange(lo if lo % 2 == 0 else lo + 1, hi, 2)
    correct_w = a_w * b_w  # ends in 0
    dists_w = []
    # One distractor also ends in 0 (partial help)
    c_also_0 = correct_w + random.choice([-10, 10, -20, 20])
    if c_also_0 > 0:
        dists_w.append(c_also_0)
    # Two distractors that don't end in 0
    for offset in [1, -1, 3, -3, 7, -7]:
        c = correct_w + offset
        if c > 0 and c % 10 != 0 and c not in dists_w:
            dists_w.append(c)
        if len(dists_w) >= 3:
            break
    while len(dists_w) < 3:
        c = correct_w + random.choice([-1, 1]) * random.randint(1, 9)
        if c > 0 and c not in dists_w and c != correct_w:
            dists_w.append(c)

    all_opts_w = [correct_w] + dists_w[:3]
    combined_w = list(zip(all_opts_w, ["correct", "distractor", "distractor", "distractor"]))
    random.shuffle(combined_w)
    all_opts_w, all_types_w = zip(*combined_w)
    opt_strs_w = [str(int(o)) for o in all_opts_w]
    ans_str_w = str(int(correct_w))

    weak = _task(
        f"{a_w} × {b_w}", f"What is {a_w} × {b_w}?",
        list(opt_strs_w), ans_str_w, correct_w,
        "Feature narrows to 2 options; eliminates some but not all distractors.",
        "Trailing digit narrows options but does not uniquely identify the answer.",
        "weak_shortcut", 0.5, list(all_types_w),
    )

    return _family("option_elimination", strong, weak, control)


GENERATORS = {
    "magnitude_estimation": _gen_magnitude,
    "structural_shortcuts": _gen_structural,
    "relative_distance": _gen_relative,
    "cancellation_identity": _gen_cancellation,
    "compatible_numbers": _gen_compatible,
    "landmark_comparison": _gen_landmark,
    "equation_reasoning": _gen_equation,
    "option_elimination": _gen_option_elimination,
}


# ── Dataset generation ──────────────────────────────────────────────────────

def generate_families(n_per_category: int, digits: int) -> List[Dict]:
    data = []
    for cat in CATEGORIES:
        fn = GENERATORS[cat]
        print(f"  Generating {n_per_category} families for '{cat}' at d={digits}...", flush=True)
        generated = 0
        attempts = 0
        limit = n_per_category * 15
        while generated < n_per_category and attempts < limit:
            attempts += 1
            try:
                fam = fn(digits)
                # Verify all variants have different expressions
                exprs = set()
                for var in ("strong_shortcut", "weak_shortcut", "control"):
                    exprs.add(fam[var]["math_expression"])
                if len(exprs) < 3:
                    continue
                # Verify answers are in options (for MC categories)
                valid = True
                for var in ("strong_shortcut", "weak_shortcut", "control"):
                    pm = fam[var]["pure_math"]
                    if pm["answer"] not in pm["options"]:
                        valid = False
                        break
                if not valid:
                    continue
                data.append(fam)
                generated += 1
            except Exception:
                continue

        if generated < n_per_category:
            print(f"    [WARN] only generated {generated}/{n_per_category}")

    return data


def generate_free_response(families: List[Dict], fraction: float = 0.3) -> List[Dict]:
    """Convert a fraction of families to free-response format (no options)."""
    n = int(len(families) * fraction)
    selected = random.sample(families, min(n, len(families)))

    fr_items = []
    for fam in selected:
        fr_fam = {
            "family_id": fam["family_id"] + "_FR",
            "category": fam["category"],
            "format": "free_response",
        }
        for var in ("strong_shortcut", "weak_shortcut", "control"):
            task = fam[var]
            pm = task["pure_math"]
            fr_fam[var] = {
                "task_type": task["task_type"],
                "math_expression": task["math_expression"],
                "ns_shortcut": task["ns_shortcut"],
                "shortcut_strength": task["shortcut_strength"],
                "pure_math": {
                    "question": pm["question"].replace("Which of the following is the best estimate for", "What is the best estimate for"),
                    "answer": pm["answer"],
                    "correct_value": pm["correct_value"],
                    "explanation": pm["explanation"],
                },
            }
        fr_items.append(fr_fam)

    return fr_items


# ── Judge task generators ────────────────────────────────────────────────────

def generate_j1_items(families: List[Dict], n: int = 300) -> List[Dict]:
    """J1: Strategy Appropriateness — is a shortcut effective for this problem?"""
    items = []
    per_cat = n // len(CATEGORIES)
    for cat in CATEGORIES:
        cat_fams = [f for f in families if f["category"] == cat]
        random.shuffle(cat_fams)
        count = 0
        for fam in cat_fams:
            if count >= per_cat:
                break
            for var in ("strong_shortcut", "weak_shortcut", "control"):
                task = fam[var]
                ground_truth = "YES" if var != "control" else "NO"
                opts = task['pure_math']['options']
                opt_str = f"(A) {opts[0]}  (B) {opts[1]}  (C) {opts[2]}  (D) {opts[3]}" if opts and len(opts) == 4 else ""
                problem_text = task['pure_math']['question']
                if opt_str:
                    problem_text = f"{problem_text}\n{opt_str}"
                items.append({
                    "task_id": f"J1_{fam['family_id']}_{var}",
                    "task_type": "J1_strategy_appropriateness",
                    "category": cat,
                    "variant": var,
                    "problem": problem_text,
                    "ground_truth": ground_truth,
                    "shortcut_strength": task["shortcut_strength"],
                    "prompt": f"Consider this math problem. Could you solve it significantly faster using mental math or a clever observation, compared to computing it step by step? Answer YES or NO, then explain in one sentence why.\n\nProblem: {problem_text}",
                })
            count += 1
    random.shuffle(items)
    return items[:n]


def generate_j2_items(families: List[Dict], n: int = 120) -> List[Dict]:
    """J2: Error Detection — find the planted error in an NS solution."""
    items = []
    per_cat = n // len(CATEGORIES)
    error_types = ["wrong_rounding", "wrong_operand", "arithmetic_slip", "transcription_error"]

    for cat in CATEGORIES:
        cat_fams = [f for f in families if f["category"] == cat]
        random.shuffle(cat_fams)
        for i, fam in enumerate(cat_fams[:per_cat]):
            task = fam["strong_shortcut"]
            pm = task["pure_math"]
            error_type = error_types[i % len(error_types)]

            # Generate a plausible wrong NS solution
            correct_val = pm["correct_value"]
            if error_type == "wrong_rounding":
                wrong_val = correct_val + random.choice([-1, 1]) * max(1, abs(correct_val) // 20)
                error_desc = "Rounded in the wrong direction"
            elif error_type == "wrong_operand":
                wrong_val = correct_val * random.choice([2, 0.5])
                error_desc = "Applied shortcut to wrong operand"
            elif error_type == "arithmetic_slip":
                wrong_val = correct_val + random.choice([-1, 1]) * random.randint(1, 5)
                error_desc = "Arithmetic slip in the shortcut step"
            else:
                wrong_val = correct_val
                wrong_val_str = str(int(wrong_val))
                if len(wrong_val_str) > 1:
                    idx = random.randint(0, len(wrong_val_str) - 1)
                    new_digit = str((int(wrong_val_str[idx]) + random.randint(1, 3)) % 10)
                    wrong_val = int(wrong_val_str[:idx] + new_digit + wrong_val_str[idx+1:])
                error_desc = "Transcription error in final answer"

            fake_solution = f"Using {task['ns_shortcut']}\nApproximation gives: {int(wrong_val)}"

            items.append({
                "task_id": f"J2_{fam['family_id']}",
                "task_type": "J2_error_detection",
                "category": cat,
                "problem": pm["question"],
                "correct_answer": str(pm["answer"]),
                "ns_solution_with_error": fake_solution,
                "error_type": error_type,
                "error_description": error_desc,
                "prompt": f"This solution attempts to use a number sense shortcut. Find the error and explain what went wrong.\n\nProblem: {pm['question']}\n\nSolution: {fake_solution}",
            })

    random.shuffle(items)
    return items[:n]


def generate_j3_items(families: List[Dict], n: int = 100) -> List[Dict]:
    """J3: Pairwise Efficiency — which solution uses fewer reasoning steps?"""
    items = []
    per_cat = n // len(CATEGORIES)

    # Step-count inventory (preregistered)
    step_counts = {
        "magnitude_estimation": {"ns": 2, "algo": 4},  # NS: round + multiply; Algo: digit-by-digit multiply + round
        "structural_shortcuts": {"ns": 3, "algo": 4},   # NS: decompose + two multiplies + add; Algo: full multiply
        "relative_distance": {"ns": 2, "algo": 3},      # NS: compare to benchmark; Algo: cross-multiply + compare
        "cancellation_identity": {"ns": 2, "algo": 3},   # NS: compute offset + add; Algo: two additions
        "compatible_numbers": {"ns": 2, "algo": 4},      # NS: round + multiply; Algo: digit-by-digit
        "landmark_comparison": {"ns": 2, "algo": 3},     # NS: landmark + compare; Algo: percentage calc + compare
        "equation_reasoning": {"ns": 1, "algo": 3},      # NS: recognize property; Algo: compute both sides
        "option_elimination": {"ns": 2, "algo": 4},      # NS: check feature + select; Algo: full multiply
    }

    for cat in CATEGORIES:
        cat_fams = [f for f in families if f["category"] == cat]
        random.shuffle(cat_fams)
        sc = step_counts[cat]
        for fam in cat_fams[:per_cat]:
            task = fam["strong_shortcut"]
            pm = task["pure_math"]

            ns_sol = f"NS approach ({sc['ns']} steps): {task['ns_shortcut']} → {pm['answer']}"
            algo_sol = f"Algorithmic approach ({sc['algo']} steps): Compute step by step → {pm['answer']}"

            # Randomize order
            if random.random() < 0.5:
                sol_a, sol_b = ns_sol, algo_sol
                correct = "A"
            else:
                sol_a, sol_b = algo_sol, ns_sol
                correct = "B"

            items.append({
                "task_id": f"J3_{fam['family_id']}",
                "task_type": "J3_pairwise_efficiency",
                "category": cat,
                "problem": pm["question"],
                "solution_a": sol_a,
                "solution_b": sol_b,
                "correct_answer": correct,
                "ns_steps": sc["ns"],
                "algo_steps": sc["algo"],
                "prompt": f"Both solutions below are correct. Which one uses fewer reasoning steps? Answer A or B.\n\nProblem: {pm['question']}\n\nSolution A: {sol_a}\n\nSolution B: {sol_b}",
            })

    random.shuffle(items)
    return items[:n]


# ── Generate-level task generators ───────────────────────────────────────────

def generate_g1_items(n_per_category: int, digits: int) -> List[Dict]:
    """G1: NS Solution Generation — solve using shortcut on held-out problems."""
    items = []
    for cat in CATEGORIES:
        fn = GENERATORS[cat]
        for i in range(n_per_category):
            try:
                fam = fn(digits)
                task = fam["strong_shortcut"]
                pm = task["pure_math"]
                items.append({
                    "task_id": f"G1_{cat}_{i:04d}",
                    "task_type": "G1_ns_solution_generation",
                    "category": cat,
                    "problem": pm["question"],
                    "correct_answer": pm["answer"],
                    "correct_value": pm["correct_value"],
                    "shortcut_hint": task["ns_shortcut"],
                    "prompt": f"Solve this using a number sense shortcut. State which shortcut strategy you used.\n\nProblem: {pm['question']}",
                })
            except Exception:
                continue
    random.shuffle(items)
    return items


def generate_g2_items() -> List[Dict]:
    """G2: NS Problem Generation — create shortcut/control pairs."""
    items = []
    for cat in CATEGORIES:
        for digits in [2, 4]:
            for i in range(5):
                items.append({
                    "task_id": f"G2_{cat}_d{digits}_{i}",
                    "task_type": "G2_ns_problem_generation",
                    "category": cat,
                    "digit_scale": digits,
                    "prompt": f"Create a math problem that can be solved using a '{cat}' number sense shortcut with {digits}-digit numbers. Also create a matched control problem where the shortcut does NOT work. Provide the answer for both.\n\nFormat:\nShortcut problem: [problem]\nShortcut answer: [answer]\nControl problem: [problem]\nControl answer: [answer]",
                })
    random.shuffle(items)
    return items


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SenseMath v2 Generator")
    parser.add_argument("--phase", choices=["test", "full", "judge", "generate"],
                        default="test")
    parser.add_argument("--digits", type=int, nargs="+", default=[2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-category", type=int, default=None,
                        help="Families per category (default: 10 for test, 400 for full)")
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(__file__).parent.parent / "benchmark"
    root.mkdir(exist_ok=True)

    if args.phase in ("test", "full"):
        n = args.n_per_category or (10 if args.phase == "test" else 400)

        for d in args.digits:
            print(f"\n{'='*60}")
            print(f"  SenseMath v2 — d={d}, n={n}/category, {len(CATEGORIES)} categories")
            print(f"{'='*60}")

            families = generate_families(n, d)

            # Save MC families
            mc_path = root / f"sensemath_v2_d{d}.json"
            with mc_path.open("w") as f:
                json.dump(families, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(families)} families → {mc_path}")

            # Save free-response audit slice
            fr = generate_free_response(families, fraction=0.3)
            fr_path = root / f"sensemath_v2_d{d}_fr.json"
            with fr_path.open("w") as f:
                json.dump(fr, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(fr)} free-response families → {fr_path}")

            # Stats
            cats = Counter(f["category"] for f in families)
            print(f"  Categories: {dict(cats)}")
            for cat in CATEGORIES:
                cat_fams = [f for f in families if f["category"] == cat]
                if cat_fams:
                    strengths = [f["strong_shortcut"]["shortcut_strength"] for f in cat_fams]
                    print(f"    {cat}: {len(cat_fams)} families, "
                          f"strong strength mean={sum(strengths)/len(strengths):.3f}")

    elif args.phase == "judge":
        # Load full families to generate judge tasks from
        all_families = []
        for d in args.digits:
            fpath = root / f"sensemath_v2_d{d}.json"
            if fpath.exists():
                with fpath.open() as f:
                    all_families.extend(json.load(f))
        if not all_families:
            print("No families found. Run --phase full first.")
            return

        j1 = generate_j1_items(all_families, n=300)
        j2 = generate_j2_items(all_families, n=120)
        j3 = generate_j3_items(all_families, n=100)

        for name, items in [("j1", j1), ("j2", j2), ("j3", j3)]:
            path = root / f"judge_{name}.json"
            with path.open("w") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(items)} {name.upper()} items → {path}")

    elif args.phase == "generate":
        g1 = generate_g1_items(n_per_category=34, digits=args.digits[0])
        g2 = generate_g2_items()

        for name, items in [("g1", g1), ("g2", g2)]:
            path = root / f"generate_{name}.json"
            with path.open("w") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(items)} {name.upper()} items → {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

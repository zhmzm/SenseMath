from pathlib import Path
#!/usr/bin/env python3
"""Build TABLE_2_interaction.tex for SenseMath paper (v2 — with weak-shortcut NS rates)."""

import json
import re
from collections import defaultdict

RESULTS_DIR = str(Path(__file__).parent.parent.parent / "results")
OUT_PATH = str(Path(__file__).parent / "TABLE_2_interaction.tex")

SCALES = [2, 4, 8, 16]
VARIANTS = ["strong_shortcut", "weak_shortcut", "control"]
VAR_SHORT = {"strong_shortcut": "S", "weak_shortcut": "W", "control": "C"}


# ─────────────────────────────────────────────
# Answer extraction for GPU models
# ─────────────────────────────────────────────
def extract_mc(resp):
    """Extract multiple-choice answer letter from raw model response."""
    # Strip <think> blocks (for reasoning models)
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    # Prefer \boxed{X}
    m = re.search(r'\\boxed\{([A-D])\}', resp)
    if m:
        return m.group(1).upper()
    # "final answer / answer : X"
    m = re.search(r'(?:final answer|answer)\s*(?:\([A-D/]+\))?\s*[:\s]\s*([A-D])',
                  resp, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Last standalone letter in last 300 chars
    matches = re.findall(r'\b([A-D])\b', resp[-300:])
    if matches:
        return matches[-1].upper()
    # Fallback: last standalone letter anywhere
    matches = re.findall(r'\b([A-D])\b', resp)
    return matches[-1].upper() if matches else ''


# ─────────────────────────────────────────────
# Load GPT accuracy data
# ─────────────────────────────────────────────
def load_gpt_acc(path, nc_cond_name="EasyNC"):
    """Return dict[scale][cond][var] = accuracy (0–1)."""
    with open(path) as f:
        records = json.load(f)

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in records:
        var = r["var"]
        cond = r["cond"]
        if var not in VARIANTS:
            continue
        if cond not in ("CoT", nc_cond_name):
            continue
        cond_key = "NC" if cond == nc_cond_name else "CoT"
        sc = r["scale"]
        ok = int(r["ok"])
        counts[sc][cond_key][var][0] += ok
        counts[sc][cond_key][var][1] += 1

    acc = {}
    for sc in SCALES:
        acc[sc] = {}
        for cond in ("CoT", "NC"):
            acc[sc][cond] = {}
            for var in VARIANTS:
                c = counts[sc][cond][var]
                acc[sc][cond][var] = c[0] / c[1] if c[1] > 0 else None
    return acc


# ─────────────────────────────────────────────
# Load GPU accuracy data
# ─────────────────────────────────────────────
def load_gpu_acc(model_file_prefix):
    """Return dict[scale][cond][var] = accuracy (0–1).
    Conditions in files: CoT / NC.
    """
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for sc in SCALES:
        path = f"{RESULTS_DIR}/{model_file_prefix}_d{sc}.json"
        with open(path) as f:
            records = json.load(f)
        for r in records:
            var = r["variant"]
            cond = r["condition"]
            if var not in VARIANTS:
                continue
            if cond not in ("CoT", "NC"):
                continue
            pred = extract_mc(r["raw_response"])
            ok = int(pred == r["ground_truth_letter"])
            counts[sc][cond][var][0] += ok
            counts[sc][cond][var][1] += 1

    acc = {}
    for sc in SCALES:
        acc[sc] = {}
        for cond in ("CoT", "NC"):
            acc[sc][cond] = {}
            for var in VARIANTS:
                c = counts[sc][cond][var]
                acc[sc][cond][var] = c[0] / c[1] if c[1] > 0 else None
    return acc


# ─────────────────────────────────────────────
# Load NS rate data
# ─────────────────────────────────────────────
def load_ns(ns_data, model_key, nc_cond_label):
    """Return dict[scale][cond][var] = ns_rate (0–100) or None.
    Now includes weak_shortcut (no longer returns '--' for W).
    """
    model_data = ns_data[model_key]
    ns = {}
    for sc in SCALES:
        ns[sc] = {}
        for cond_key, cond_label in [("CoT", "CoT"), ("NC", nc_cond_label)]:
            ns[sc][cond_key] = {}
            for var in VARIANTS:
                key = f"{sc}|{cond_label}|{var}"
                if key in model_data:
                    d = model_data[key]
                    rate = d["shortcut"] / d["n"] * 100 if d["n"] > 0 else None
                    ns[sc][cond_key][var] = rate
                else:
                    ns[sc][cond_key][var] = None
    return ns


# ─────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────
def fmt_int(val):
    """Format value as integer string, or '--' if None."""
    if val is None:
        return "--"
    return str(int(round(val)))


def triple_acc(acc, sc, cond):
    """S & W & C accuracy cells (percent, integer)."""
    vals = []
    for v in VARIANTS:
        x = acc[sc][cond][v]
        vals.append(fmt_int(x * 100) if x is not None else "--")
    return " & ".join(vals)


def triple_ns(ns, sc, cond):
    """S & W & C NS-rate cells (already in percent, integer)."""
    vals = []
    for v in VARIANTS:
        x = ns[sc][cond][v]
        vals.append(fmt_int(x) if x is not None else "--")
    return " & ".join(vals)


# ─────────────────────────────────────────────
# Build rows for one model (no \multirow)
# ─────────────────────────────────────────────
def model_rows(model_name, acc, ns):
    """Return list of LaTeX row strings for one model (4 rows, no \\multirow)."""
    rows_data = {
        r"$\text{Acc}_{\text{CoT}}$":  ("CoT", triple_acc, acc),
        r"$\text{Acc}_{\text{NS}}$":   ("NC",  triple_acc, acc),
        r"$\text{NS\%}_{\text{CoT}}$": ("CoT", triple_ns,  ns),
        r"$\text{NS\%}_{\text{NS}}$":  ("NC",  triple_ns,  ns),
    }

    row_order = [
        r"$\text{Acc}_{\text{CoT}}$",
        r"$\text{Acc}_{\text{NS}}$",
        r"$\text{NS\%}_{\text{CoT}}$",
        r"$\text{NS\%}_{\text{NS}}$",
    ]

    lines = []
    for i, metric_label in enumerate(row_order):
        cond, fn, data = rows_data[metric_label]
        cells = " & ".join(fn(data, sc, cond) for sc in SCALES)
        if i == 0:
            model_cell = model_name
        else:
            model_cell = ""
        lines.append(f"    {model_cell} & {metric_label} & {cells} \\\\")
    return lines


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Load NS rates
    with open(f"{RESULTS_DIR}/ns_rate_all_models.json") as f:
        ns_data = json.load(f)

    # ── GPT models ──────────────────────────────
    gpt4omini_acc = load_gpt_acc(f"{RESULTS_DIR}/gpt4omini_easyNC_all.json", "EasyNC")
    gpt41mini_acc = load_gpt_acc(f"{RESULTS_DIR}/gpt41mini_easyNC_all.json", "EasyNC")
    gpt4omini_ns  = load_ns(ns_data, "gpt4omini", "EasyNC")
    gpt41mini_ns  = load_ns(ns_data, "gpt41mini", "EasyNC")

    # ── GPU models ──────────────────────────────
    qwen8b_acc    = load_gpu_acc("use_Qwen_Qwen3-8B")
    qwen30b_acc   = load_gpu_acc("use_Qwen_Qwen3-30B-A3B-Instruct-2507")
    llama8b_acc   = load_gpu_acc("use_meta-llama_Llama-3.1-8B-Instruct")

    qwen8b_ns     = load_ns(ns_data, "qwen8b",  "NC")
    qwen30b_ns    = load_ns(ns_data, "qwen30b", "NC")
    llama8b_ns    = load_ns(ns_data, "llama8b", "NC")

    # ── Print diagnostics ────────────────────────
    print("=== Accuracy values ===")
    for label, acc in [
        ("gpt4omini", gpt4omini_acc),
        ("gpt41mini", gpt41mini_acc),
        ("qwen8b",    qwen8b_acc),
        ("qwen30b",   qwen30b_acc),
        ("llama8b",   llama8b_acc),
    ]:
        print(f"\n{label}:")
        for sc in SCALES:
            for cond in ("CoT", "NC"):
                row = " | ".join(
                    f"{VAR_SHORT[v]}={fmt_int(acc[sc][cond][v]*100 if acc[sc][cond][v] is not None else None)}"
                    for v in VARIANTS
                )
                print(f"  d={sc} {cond}: {row}")

    print("\n=== NS rates ===")
    for label, ns in [
        ("gpt4omini", gpt4omini_ns),
        ("gpt41mini", gpt41mini_ns),
        ("qwen8b",    qwen8b_ns),
        ("qwen30b",   qwen30b_ns),
        ("llama8b",   llama8b_ns),
    ]:
        print(f"\n{label}:")
        for sc in SCALES:
            for cond in ("CoT", "NC"):
                row = " | ".join(
                    f"{VAR_SHORT[v]}={fmt_int(ns[sc][cond][v])}"
                    for v in VARIANTS
                )
                print(f"  d={sc} {cond}: {row}")

    # ─────────────────────────────────────────────
    # Build LaTeX
    # ─────────────────────────────────────────────

    # Column spec: Model | Metric | d=2 (S W C) | d=4 (S W C) | d=8 (S W C) | d=16 (S W C)
    # Vertical bars between digit-scale groups, as required
    col_spec = r"{ll|rrr|rrr|rrr|rrr}"

    # Scale header: \multicolumn{3}{c|}{...} for each group
    # Last group has no trailing bar
    scale_header_parts = []
    for i, sc in enumerate(SCALES):
        if i < len(SCALES) - 1:
            scale_header_parts.append(
                f"\\multicolumn{{3}}{{c|}}{{$d={sc}$ (\\%)}}"
            )
        else:
            scale_header_parts.append(
                f"\\multicolumn{{3}}{{c}}{{$d={sc}$ (\\%)}}"
            )
    scale_header = " & ".join(scale_header_parts)

    # Sub-header row: Model & Metric & S & W & C & S & W & C & ...
    swc = "$S$ & $W$ & $C$"
    swc_header = " & ".join([swc] * 4)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{2.5pt}")
    lines.append(
        r"  \caption{Per-condition results across digit scales for five models. "
        r"Acc = accuracy; NS\% = shortcut-strategy rate classified by GPT-4.1-mini judge. "
        r"$S$ = strong-shortcut variant, $W$ = weak-shortcut variant, $C$ = control variant. "
        r"All numbers are percentages (rounded to integers).}"
    )
    lines.append(r"  \label{tab:main_results}")
    lines.append(f"  \\begin{{tabular}}{col_spec}")
    lines.append(r"    \toprule")

    # Header row 1: Model & Metric & d=2 (%) ... d=16 (%)
    lines.append(f"    \\textbf{{Model}} & \\textbf{{Metric}} & {scale_header} \\\\")

    # Header row 2: blank & blank & S & W & C repeated
    lines.append(f"     &  & {swc_header} \\\\")
    lines.append(r"    \midrule")

    model_configs = [
        ("GPT-4o-mini",        gpt4omini_acc, gpt4omini_ns),
        ("GPT-4.1-mini",       gpt41mini_acc, gpt41mini_ns),
        ("Qwen3-8B",           qwen8b_acc,    qwen8b_ns),
        ("Qwen3-30B-A3B",      qwen30b_acc,   qwen30b_ns),
        ("Llama-3.1-8B",       llama8b_acc,   llama8b_ns),
    ]

    for i, (name, acc, ns) in enumerate(model_configs):
        rows = model_rows(name, acc, ns)
        lines.extend(rows)
        if i < len(model_configs) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    with open(OUT_PATH, "w") as f:
        f.write(latex + "\n")
    print(f"\nWrote {OUT_PATH}")
    print("\n=== LaTeX output ===")
    print(latex)


if __name__ == "__main__":
    main()

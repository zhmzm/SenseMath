#!/usr/bin/env python3
"""Generate all paper figures for SenseMath v2."""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paper_plot_style import *

DATA_DIR = Path(__file__).parent.parent.parent / 'dataset_v2' / 'sensemath_v2_results'
DS_DIR = Path(__file__).parent.parent.parent / 'dataset_v2' / 'sensemath_v2'

def extract_mc(resp):
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    # 1. Try \boxed{X}
    m = re.search(r'\\boxed\{([A-D])\}', resp)
    if m: return m.group(1).upper()
    # 2. Try "answer" pattern, skip (A/B/C/D) template
    m = re.search(r'(?:final answer|answer)\s*(?:\([A-D/]+\))?\s*[:\s]\s*([A-D])', resp, re.IGNORECASE)
    if m: return m.group(1).upper()
    # 3. Last standalone letter on its own line
    m = re.search(r'(?:^|\n)\s*([A-D])\s*$', resp)
    if m: return m.group(1).upper()
    # 4. Last occurrence
    matches = re.findall(r'\b([A-D])\b', resp)
    if matches: return matches[-1].upper()
    return ''

def load_results(scale, exclude_strict=True, exclude_haiku=True):
    """Load all model results for a scale, compute accuracy."""
    all_data = {}
    for f in sorted(DATA_DIR.glob(f'use_*_d{scale}.json')):
        if exclude_strict and 'strict' in f.name: continue
        if exclude_haiku and 'haiku' in f.name: continue
        records = json.load(open(f))
        if not records: continue
        model = short_name(records[0].get('model', ''))

        groups = {}
        for r in records:
            key = (r['variant'], r['condition'])
            if key not in groups: groups[key] = []
            pred = extract_mc(r.get('raw_response', ''))
            correct = r.get('ground_truth_letter', '')
            groups[key].append(1 if pred == correct else 0)

        all_data[model] = groups
    return all_data


def compute_interaction(groups):
    """Compute interaction from groups dict. Returns (cot_s, nc_s, cot_c, nc_c, interact)."""
    def acc(var, cond):
        k = (var, cond)
        if k not in groups or not groups[k]: return 0
        return sum(groups[k]) / len(groups[k])

    cot_name = 'StrictCoT' if ('strong_shortcut', 'StrictCoT') in groups else 'CoT'
    cot_s = acc('strong_shortcut', cot_name)
    nc_s = acc('strong_shortcut', 'NC')
    cot_c = acc('control', cot_name)
    nc_c = acc('control', 'NC')
    interact = (nc_s - cot_s) - (nc_c - cot_c)
    return cot_s, nc_s, cot_c, nc_c, interact


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Interaction Effect Heatmap (models × scales)
# ══════════════════════════════════════════════════════════════════════════════

def gen_fig2_interaction_heatmap():
    models_order = MODEL_ORDER
    scales = [2, 4, 8]

    matrix = np.full((len(models_order), len(scales)), np.nan)
    pvals = np.full((len(models_order), len(scales)), 1.0)

    for si, scale in enumerate(scales):
        data = load_results(scale)
        for mi, model in enumerate(models_order):
            if model not in data: continue
            groups = data[model]
            _, _, _, _, interact = compute_interaction(groups)
            matrix[mi, si] = interact

            # Quick permutation test
            by_fam_keys = set()
            fam_data = {}
            cot_name = 'StrictCoT' if ('strong_shortcut', 'StrictCoT') in groups else 'CoT'

            # Build per-family Y
            n_items = len(groups.get(('strong_shortcut', cot_name), []))
            Y = []
            gs = groups.get(('strong_shortcut', 'NC'), [])
            gc = groups.get(('strong_shortcut', cot_name), [])
            gns_c = groups.get(('control', 'NC'), [])
            gcc = groups.get(('control', cot_name), [])
            for i in range(min(len(gs), len(gc), len(gns_c), len(gcc))):
                Y.append((gs[i] - gc[i]) - (gns_c[i] - gcc[i]))

            if Y:
                Y = np.array(Y, dtype=float)
                mean_Y = np.mean(Y)
                rng = np.random.RandomState(42)
                perms = np.array([np.mean(Y * rng.choice([-1,1], size=len(Y))) for _ in range(10000)])
                pvals[mi, si] = np.mean(np.abs(perms) >= np.abs(mean_Y))

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.15, vmax=0.2, aspect='auto')

    for mi in range(len(models_order)):
        for si in range(len(scales)):
            val = matrix[mi, si]
            pv = pvals[mi, si]
            if np.isnan(val): continue
            sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
            color = 'white' if abs(val) > 0.1 else 'black'
            ax.text(si, mi, f'{val:+.3f}{sig}', ha='center', va='center',
                    fontsize=FONT_SIZE - 1, color=color, fontweight='bold' if sig else 'normal')

    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'd={s}' for s in scales])
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels(models_order)
    ax.set_xlabel('Digit Scale')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Interaction Effect $Y$', fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    save_fig(fig, 'fig2_interaction_heatmap')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Grouped Bar — CoT vs NC accuracy at d=4
# ══════════════════════════════════════════════════════════════════════════════

def gen_fig3_accuracy_bars():
    data = load_results(4)
    models = [m for m in MODEL_ORDER if m in data]

    cot_s_vals, nc_s_vals = [], []
    cot_c_vals, nc_c_vals = [], []

    for model in models:
        cot_s, nc_s, cot_c, nc_c, _ = compute_interaction(data[model])
        cot_s_vals.append(cot_s * 100)
        nc_s_vals.append(nc_s * 100)
        cot_c_vals.append(cot_c * 100)
        nc_c_vals.append(nc_c * 100)

    x = np.arange(len(models))
    w = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    ax.bar(x - 1.5*w, cot_s_vals, w, label='CoT / Strong', color=COLORS[0], alpha=0.8)
    ax.bar(x - 0.5*w, nc_s_vals, w, label='NC / Strong', color=COLORS[0], alpha=0.4, edgecolor=COLORS[0], linewidth=1)
    ax.bar(x + 0.5*w, cot_c_vals, w, label='CoT / Control', color=COLORS[2], alpha=0.8)
    ax.bar(x + 1.5*w, nc_c_vals, w, label='NC / Control', color=COLORS[2], alpha=0.4, edgecolor=COLORS[2], linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2, loc='upper right', fontsize=FONT_SIZE - 2)

    fig.tight_layout()
    save_fig(fig, 'fig3_accuracy_bars_d4')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Scatter — MATH-500 vs SenseMath interaction (d=4)
# ══════════════════════════════════════════════════════════════════════════════

def gen_fig4_math500_scatter():
    # MATH-500 accuracies
    math500 = {}
    for f in sorted(DATA_DIR.glob('math500_*.json')):
        if 'analysis' in f.name: continue
        records = json.load(open(f))
        model_raw = f.stem.replace('math500_', '')
        # Map to short name
        mapping = {
            'Qwen_Qwen3-30B-A3B-Instruct-2507': 'Qwen3-30B',
            'Qwen_Qwen3-8B': 'Qwen3-8B',
            'meta-llama_Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
            'gpt-5-mini': 'GPT-5-mini',
            'gpt-4o-mini': 'GPT-4o-mini',
        }
        name = mapping.get(model_raw, model_raw)
        n = len(records)
        correct = sum(1 for r in records if r.get('is_correct'))
        math500[name] = correct / n * 100 if n else 0

    # SenseMath interaction at d=4
    data = load_results(4)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    for model in MODEL_ORDER:
        if model not in data or model not in math500: continue
        _, _, _, _, interact = compute_interaction(data[model])
        mx = math500[model]
        color = MODEL_COLORS.get(model, 'gray')
        ax.scatter(mx, interact, color=color, s=80, zorder=5)
        # Label offset
        offset_x, offset_y = 1.5, 0.005
        if model == 'GPT-4o-mini': offset_y = -0.015
        if model == 'Llama-3.1-8B': offset_y = -0.015
        ax.annotate(model, (mx, interact), xytext=(mx + offset_x, interact + offset_y),
                    fontsize=FONT_SIZE - 2)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('MATH-500 Accuracy (%)')
    ax.set_ylabel('SenseMath Interaction $Y$ (d=4)')

    fig.tight_layout()
    save_fig(fig, 'fig4_math500_scatter')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Per-category interaction at d=4 for Qwen3-30B
# ══════════════════════════════════════════════════════════════════════════════

def gen_fig5_per_category():
    scale = 4
    target_model = 'Qwen/Qwen3-30B-A3B-Instruct-2507'

    # Load raw records
    for f in DATA_DIR.glob(f'use_*_d{scale}.json'):
        if 'strict' in f.name or 'haiku' in f.name: continue
        records = json.load(open(f))
        if not records: continue
        if records[0].get('model') == target_model:
            break
    else:
        print("Qwen3-30B d=4 not found")
        return

    categories = sorted(set(r['category'] for r in records))
    cat_short = {
        'cancellation_identity': 'Cancel.',
        'compatible_numbers': 'Compat.',
        'landmark_comparison': 'Landmark',
        'magnitude_estimation': 'Magnit.',
        'relative_distance': 'Rel. Dist.',
        'structural_shortcuts': 'Struct.',
    }

    nc_s_acc, nc_c_acc = [], []

    for cat in categories:
        s_recs = [r for r in records if r['category'] == cat and r['variant'] == 'strong_shortcut' and r['condition'] == 'NC']
        c_recs = [r for r in records if r['category'] == cat and r['variant'] == 'control' and r['condition'] == 'NC']

        s_correct = sum(1 for r in s_recs if extract_mc(r.get('raw_response', '')) == r.get('ground_truth_letter', ''))
        c_correct = sum(1 for r in c_recs if extract_mc(r.get('raw_response', '')) == r.get('ground_truth_letter', ''))

        nc_s_acc.append(s_correct / len(s_recs) * 100 if s_recs else 0)
        nc_c_acc.append(c_correct / len(c_recs) * 100 if c_recs else 0)

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    ax.bar(x - w/2, nc_s_acc, w, label='NC / Strong', color=COLORS[0], alpha=0.8)
    ax.bar(x + w/2, nc_c_acc, w, label='NC / Control', color=COLORS[2], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([cat_short.get(c, c) for c in categories], rotation=20, ha='right')
    ax.set_ylabel('NC Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2)

    # Add delta labels
    for i in range(len(categories)):
        delta = nc_s_acc[i] - nc_c_acc[i]
        y_pos = max(nc_s_acc[i], nc_c_acc[i]) + 2
        ax.text(i, y_pos, f'{delta:+.0f}%', ha='center', va='bottom', fontsize=FONT_SIZE - 2,
                color='green' if delta > 0 else 'red', fontweight='bold')

    fig.tight_layout()
    save_fig(fig, 'fig5_per_category_d4')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: Main interaction results
# ══════════════════════════════════════════════════════════════════════════════

def gen_table2_main_results():
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Interaction effect $Y$ across models and digit scales. $Y = (\text{NC}_\text{strong} - \text{CoT}_\text{strong}) - (\text{NC}_\text{control} - \text{CoT}_\text{control})$. Positive $Y$ means NS prompt selectively helps on shortcut items. $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$ (permutation test, 10{,}000 iterations).}')
    lines.append(r'\label{tab:interaction}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l ccc ccc ccc}')
    lines.append(r'\toprule')
    lines.append(r'& \multicolumn{3}{c}{$d=2$} & \multicolumn{3}{c}{$d=4$} & \multicolumn{3}{c}{$d=8$} \\')
    lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
    lines.append(r'Model & CoT$_s$ & NC$_s$ & $Y$ & CoT$_s$ & NC$_s$ & $Y$ & CoT$_s$ & NC$_s$ & $Y$ \\')
    lines.append(r'\midrule')

    for model in MODEL_ORDER:
        row = [model]
        for scale in [2, 4, 8]:
            data = load_results(scale)
            if model not in data:
                row.extend(['--', '--', '--'])
                continue
            cot_s, nc_s, cot_c, nc_c, interact = compute_interaction(data[model])

            # Permutation test
            groups = data[model]
            cot_name = 'StrictCoT' if ('strong_shortcut', 'StrictCoT') in groups else 'CoT'
            gs = groups.get(('strong_shortcut', 'NC'), [])
            gc = groups.get(('strong_shortcut', cot_name), [])
            gns_c = groups.get(('control', 'NC'), [])
            gcc = groups.get(('control', cot_name), [])
            Y_arr = []
            for i in range(min(len(gs), len(gc), len(gns_c), len(gcc))):
                Y_arr.append((gs[i] - gc[i]) - (gns_c[i] - gcc[i]))
            pv = 1.0
            if Y_arr:
                Y_arr = np.array(Y_arr, dtype=float)
                rng = np.random.RandomState(42)
                perms = np.array([np.mean(Y_arr * rng.choice([-1,1], size=len(Y_arr))) for _ in range(10000)])
                pv = np.mean(np.abs(perms) >= np.abs(np.mean(Y_arr)))

            sig = '^{***}' if pv < 0.001 else ('^{**}' if pv < 0.01 else ('^{*}' if pv < 0.05 else ''))
            row.append(f'{cot_s:.1%}'.replace('%', r'\%'))
            row.append(f'{nc_s:.1%}'.replace('%', r'\%'))
            y_str = f'${interact:+.3f}{sig}$'
            row.append(y_str)

        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(f'{FIG_DIR}/TABLE_2_interaction.tex', 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {FIG_DIR}/TABLE_2_interaction.tex')


# ══════════════════════════════════════════════════════════════════════════════
# Table 4: Rank comparison (Use vs Judge vs MATH-500)
# ══════════════════════════════════════════════════════════════════════════════

def gen_table4_rank_comparison():
    # Use interaction at d=4
    data_d4 = load_results(4)
    use_ranks = {}
    for model in MODEL_ORDER:
        if model not in data_d4: continue
        _, _, _, _, interact = compute_interaction(data_d4[model])
        use_ranks[model] = interact

    # Judge J1 accuracy
    judge_ranks = {}
    for f in DATA_DIR.glob('judge_j1_*.json'):
        records = json.load(open(f))
        if not records: continue
        model_raw = records[0].get('model', '')
        name = short_name(model_raw)
        if name not in MODEL_ORDER: continue
        correct = sum(1 for r in records if r.get('predicted', '').upper() == r.get('ground_truth', '').upper())
        judge_ranks[name] = correct / len(records) * 100

    # MATH-500
    math_ranks = {}
    mapping = {
        'Qwen_Qwen3-30B-A3B-Instruct-2507': 'Qwen3-30B',
        'Qwen_Qwen3-8B': 'Qwen3-8B',
        'meta-llama_Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
        'gpt-5-mini': 'GPT-5-mini',
        'gpt-4o-mini': 'GPT-4o-mini',
    }
    for f in DATA_DIR.glob('math500_*.json'):
        if 'analysis' in f.name: continue
        records = json.load(open(f))
        name = mapping.get(f.stem.replace('math500_', ''), '')
        if not name or name not in MODEL_ORDER: continue
        correct = sum(1 for r in records if r.get('is_correct'))
        math_ranks[name] = correct / len(records) * 100

    # Sort and assign ranks
    use_sorted = sorted(use_ranks.items(), key=lambda x: -x[1])
    judge_sorted = sorted(judge_ranks.items(), key=lambda x: -x[1])
    math_sorted = sorted(math_ranks.items(), key=lambda x: -x[1])

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Model rankings across SenseMath Use (interaction at $d\!=\!4$), Judge (J1 accuracy), and MATH-500. Rank swaps between tasks demonstrate that SenseMath measures distinct competence dimensions.}')
    lines.append(r'\label{tab:ranks}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{c lc lc lc}')
    lines.append(r'\toprule')
    lines.append(r'Rank & \multicolumn{2}{c}{Use ($Y$, $d\!=\!4$)} & \multicolumn{2}{c}{Judge (J1 Acc)} & \multicolumn{2}{c}{MATH-500 Acc} \\')
    lines.append(r'\midrule')

    for i in range(min(5, len(use_sorted))):
        parts = [str(i+1)]
        for sorted_list in [use_sorted, judge_sorted, math_sorted]:
            if i < len(sorted_list):
                name, val = sorted_list[i]
                if sorted_list is use_sorted:
                    parts.extend([name, f'${val:+.3f}$'])
                else:
                    parts.extend([name, f'{val:.1f}\\%'])
            else:
                parts.extend(['--', '--'])
        lines.append(' & '.join(parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(f'{FIG_DIR}/TABLE_4_ranks.tex', 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {FIG_DIR}/TABLE_4_ranks.tex')


# ══════════════════════════════════════════════════════════════════════════════
# Figure: Radar plot — per-category interaction Y at d=4
# ══════════════════════════════════════════════════════════════════════════════

def gen_fig_radar_d4():
    """Radar plot of per-category interaction Y at d=4 for 5 models."""
    categories = [
        'magnitude_estimation', 'structural_shortcuts', 'relative_distance',
        'cancellation_identity', 'compatible_numbers', 'landmark_comparison',
        'equation_reasoning', 'option_elimination',
    ]
    cat_labels = [
        'Magnitude\nEst.', 'Structural\nDecomp.', 'Fraction\nComp.',
        'Near\nCancel.', 'Compatible\nNum.', 'Landmark\nComp.',
        'Equation\nReason.', 'Option\nElim.',
    ]

    scale = 4

    def extract_mc_local(resp):
        import re as _re
        resp = _re.sub(r'<think>.*?</think>', '', resp, flags=_re.DOTALL).strip()
        # 1. Try \boxed{X}
        m = _re.search(r'\\boxed\{([A-D])\}', resp)
        if m: return m.group(1).upper()
        # 2. Try "answer" pattern, skip (A/B/C/D) template
        m = _re.search(r'(?:final answer|answer)\s*(?:\([A-D/]+\))?\s*[:\s]\s*([A-D])', resp, _re.IGNORECASE)
        if m: return m.group(1).upper()
        # 3. Last standalone letter on its own line
        m = _re.search(r'(?:^|\n)\s*([A-D])\s*$', resp)
        if m: return m.group(1).upper()
        # 4. Last occurrence
        matches = _re.findall(r'\b([A-D])\b', resp)
        if matches: return matches[-1].upper()
        return ''

    def acc_std(lst):
        """Accuracy for standard records (ground_truth_letter field)."""
        if not lst: return 0
        return sum(1 for r in lst if extract_mc_local(r.get('raw_response', '')) == r.get('ground_truth_letter', '')) / len(lst)

    def acc_gpt(lst):
        """Accuracy for GPT special records (ok field)."""
        if not lst: return 0
        return sum(1 for r in lst if r.get('ok', False)) / len(lst)

    # Load GPT special files
    gpt4omini_records = json.load(open(DATA_DIR / 'gpt4omini_easyNC_all.json'))
    gpt41mini_records = json.load(open(DATA_DIR / 'gpt41mini_easyNC_all.json'))

    # Compute per-category Y for each model
    model_values = {}

    # GPT-4o-mini: filter by scale, use 'EasyNC' as NC and 'CoT' as CoT
    for model_name, gpt_recs in [('GPT-4o-mini', gpt4omini_records), ('GPT-4.1-mini', gpt41mini_records)]:
        recs = [r for r in gpt_recs if r['scale'] == scale]
        vals = []
        for cat in categories:
            cot_s = [r for r in recs if r['cat'] == cat and r['var'] == 'strong_shortcut' and r['cond'] == 'CoT']
            nc_s  = [r for r in recs if r['cat'] == cat and r['var'] == 'strong_shortcut' and r['cond'] == 'EasyNC']
            cot_c = [r for r in recs if r['cat'] == cat and r['var'] == 'control'          and r['cond'] == 'CoT']
            nc_c  = [r for r in recs if r['cat'] == cat and r['var'] == 'control'          and r['cond'] == 'EasyNC']
            interact = (acc_gpt(nc_s) - acc_gpt(cot_s)) - (acc_gpt(nc_c) - acc_gpt(cot_c))
            vals.append(interact)
        model_values[model_name] = vals

    # Qwen/Llama: standard use_*_d{scale}.json files
    std_models = [
        ('Qwen3-30B',    DATA_DIR / f'use_Qwen_Qwen3-30B-A3B-Instruct-2507_d{scale}.json'),
        ('Qwen3-8B',     DATA_DIR / f'use_Qwen_Qwen3-8B_d{scale}.json'),
        ('Llama-3.1-8B', DATA_DIR / f'use_meta-llama_Llama-3.1-8B-Instruct_d{scale}.json'),
    ]
    for model_name, fpath in std_models:
        records = json.load(open(fpath))
        vals = []
        for cat in categories:
            cot_s = [r for r in records if r['category'] == cat and r['variant'] == 'strong_shortcut' and r['condition'] == 'CoT']
            nc_s  = [r for r in records if r['category'] == cat and r['variant'] == 'strong_shortcut' and r['condition'] == 'NC']
            cot_c = [r for r in records if r['category'] == cat and r['variant'] == 'control'          and r['condition'] == 'CoT']
            nc_c  = [r for r in records if r['category'] == cat and r['variant'] == 'control'          and r['condition'] == 'NC']
            interact = (acc_std(nc_s) - acc_std(cot_s)) - (acc_std(nc_c) - acc_std(cot_c))
            vals.append(interact)
        model_values[model_name] = vals

    # Use MODEL_ORDER to set plot order
    ordered_model_names = [m for m in MODEL_ORDER if m in model_values]

    # Print Y values table for verification
    col_w = 15
    header = f"{'Category':<28}" + "".join(f"{m:<{col_w}}" for m in ordered_model_names)
    print(f"\n--- Radar Y values (per-category interaction, d={scale}) ---")
    print(header)
    print("-" * len(header))
    for i, cat in enumerate(categories):
        row = f"{cat:<28}" + "".join(f"{model_values[m][i]:< {col_w}.4f}" for m in ordered_model_names)
        print(row)
    print("-" * len(header))

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    # Y-axis range and ticks
    y_min, y_max = -0.5, 0.8
    ax.set_ylim(y_min, y_max)
    ytick_vals = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels(
        [f'{v:+.1f}' if v != 0 else '0' for v in ytick_vals],
        fontsize=FONT_SIZE - 2,
    )

    # Highlight the 0.0 baseline circle with a red dashed line
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_circle, np.zeros_like(theta_circle),
            linestyle='--', linewidth=1.8, color='red', alpha=0.8, zorder=4)

    # Move radial labels outward to avoid overlap
    ax.set_rlabel_position(22.5)

    # Plot each model in MODEL_ORDER
    for model_name in ordered_model_names:
        color = MODEL_COLORS.get(model_name, 'gray')
        vals = model_values[model_name] + [model_values[model_name][0]]
        ax.plot(angles, vals, '-o', color=color, linewidth=1.5, markersize=4, label=model_name)
        ax.fill(angles, vals, color=color, alpha=0.08)

    # Category labels pushed outward with increased pad
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=8)
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle)
        if 90 < angle_deg < 270:
            label.set_ha('right')
        else:
            label.set_ha('left')
        label.set_va('center')

    # Increase the pad between labels and the plot boundary
    ax.tick_params(axis='x', pad=12)

    ax.legend(loc='upper left', bbox_to_anchor=(1.35, 1.15),
              frameon=True, framealpha=0.9, fontsize=FONT_SIZE - 2, labelspacing=0.6)

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    save_fig(fig, 'fig_radar_d4')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure: Radar plot — per-category interaction Y for all scales (2x2 grid)
# ══════════════════════════════════════════════════════════════════════════════

def gen_fig_radar_all_scales():
    """2x2 radar plots of per-category interaction Y for d=2,4,8,16."""
    categories = [
        'magnitude_estimation', 'structural_shortcuts', 'relative_distance',
        'cancellation_identity', 'compatible_numbers', 'landmark_comparison',
        'equation_reasoning', 'option_elimination',
    ]
    cat_labels = [
        'Magnitude\nEst.', 'Structural\nDecomp.', 'Fraction\nComp.',
        'Near\nCancel.', 'Compatible\nNum.', 'Landmark\nComp.',
        'Equation\nReason.', 'Option\nElim.',
    ]

    scales = [2, 4, 8, 16]

    # Standard Qwen/Llama model file keys
    std_models_keys = [
        ('Qwen3-30B',    'Qwen_Qwen3-30B-A3B-Instruct-2507'),
        ('Qwen3-8B',     'Qwen_Qwen3-8B'),
        ('Llama-3.1-8B', 'meta-llama_Llama-3.1-8B-Instruct'),
    ]
    # GPT special models: (display_name, json_filename)
    gpt_models_files = [
        ('GPT-4o-mini',  'gpt4omini_easyNC_all.json'),
        ('GPT-4.1-mini', 'gpt41mini_easyNC_all.json'),
    ]

    def extract_mc_local(resp):
        import re as _re
        resp = _re.sub(r'<think>.*?</think>', '', resp, flags=_re.DOTALL).strip()
        m = _re.search(r'\\boxed\{([A-D])\}', resp)
        if m: return m.group(1).upper()
        m = _re.search(r'(?:final answer|answer)\s*(?:\([A-D/]+\))?\s*[:\s]\s*([A-D])', resp, _re.IGNORECASE)
        if m: return m.group(1).upper()
        m = _re.search(r'(?:^|\n)\s*([A-D])\s*$', resp)
        if m: return m.group(1).upper()
        matches = _re.findall(r'\b([A-D])\b', resp)
        if matches: return matches[-1].upper()
        return ''

    def acc_std(lst):
        """Accuracy for standard records (ground_truth_letter field)."""
        if not lst: return 0
        return sum(1 for r in lst if extract_mc_local(r.get('raw_response', '')) == r.get('ground_truth_letter', '')) / len(lst)

    def acc_gpt(lst):
        """Accuracy for GPT special records (ok field)."""
        if not lst: return 0
        return sum(1 for r in lst if r.get('ok', False)) / len(lst)

    # Load GPT special files once
    gpt_all_records = {
        name: json.load(open(DATA_DIR / fname))
        for name, fname in gpt_models_files
    }

    # All model names in MODEL_ORDER
    all_model_names = [m for m in MODEL_ORDER
                       if m in [n for n, _ in gpt_models_files] or m in [n for n, _ in std_models_keys]]

    # Compute per-category Y for all models and scales
    # all_values[scale][model_name] = list of Y per category
    all_values = {}
    for scale in scales:
        all_values[scale] = {}

        # GPT special models
        for model_name, _ in gpt_models_files:
            gpt_recs = gpt_all_records[model_name]
            recs = [r for r in gpt_recs if r['scale'] == scale]
            vals = []
            for cat in categories:
                cot_s = [r for r in recs if r['cat'] == cat and r['var'] == 'strong_shortcut' and r['cond'] == 'CoT']
                nc_s  = [r for r in recs if r['cat'] == cat and r['var'] == 'strong_shortcut' and r['cond'] == 'EasyNC']
                cot_c = [r for r in recs if r['cat'] == cat and r['var'] == 'control'          and r['cond'] == 'CoT']
                nc_c  = [r for r in recs if r['cat'] == cat and r['var'] == 'control'          and r['cond'] == 'EasyNC']
                interact = (acc_gpt(nc_s) - acc_gpt(cot_s)) - (acc_gpt(nc_c) - acc_gpt(cot_c))
                vals.append(interact)
            all_values[scale][model_name] = vals

        # Standard Qwen/Llama models
        for model_name, file_key in std_models_keys:
            fpath = DATA_DIR / f'use_{file_key}_d{scale}.json'
            records = json.load(open(fpath))
            vals = []
            for cat in categories:
                cot_s = [r for r in records if r['category'] == cat and r['variant'] == 'strong_shortcut' and r['condition'] == 'CoT']
                nc_s  = [r for r in records if r['category'] == cat and r['variant'] == 'strong_shortcut' and r['condition'] == 'NC']
                cot_c = [r for r in records if r['category'] == cat and r['variant'] == 'control'          and r['condition'] == 'CoT']
                nc_c  = [r for r in records if r['category'] == cat and r['variant'] == 'control'          and r['condition'] == 'NC']
                interact = (acc_std(nc_s) - acc_std(cot_s)) - (acc_std(nc_c) - acc_std(cot_c))
                vals.append(interact)
            all_values[scale][model_name] = vals

    # Print Y values table for all scales
    print("\n--- Radar Y values (per-category interaction, all scales) ---")
    for scale in scales:
        col_w = 15
        header = f"{'Category':<28}" + "".join(f"{m:<{col_w}}" for m in all_model_names)
        print(f"\nd={scale}:")
        print(header)
        print("-" * len(header))
        for i, cat in enumerate(categories):
            row = f"{cat:<28}" + "".join(f"{all_values[scale][m][i]:< {col_w}.4f}" for m in all_model_names)
            print(row)
        print("-" * len(header))

    # Compute global y range across all scales and models
    all_flat = [v for scale in scales for model_name in all_model_names for v in all_values[scale][model_name]]
    global_min = min(all_flat)
    global_max = max(all_flat)
    # Round outward with some padding
    y_pad = 0.05
    y_min = np.floor((global_min - y_pad) * 10) / 10
    y_max = np.ceil((global_max + y_pad) * 10) / 10
    print(f"Global Y range: [{global_min:.4f}, {global_max:.4f}] -> plot [{y_min}, {y_max}]")

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()

    model_colors = [MODEL_COLORS.get(m, 'gray') for m in all_model_names]

    # Build y ticks within range
    tick_step = 0.2
    ytick_start = np.ceil(y_min / tick_step) * tick_step
    ytick_vals = []
    v = ytick_start
    while v <= y_max + 1e-9:
        ytick_vals.append(round(v, 2))
        v += tick_step
    if 0.0 not in ytick_vals:
        ytick_vals.append(0.0)
        ytick_vals.sort()

    for idx, (scale, ax) in enumerate(zip(scales, axes_flat)):
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(ytick_vals)
        ax.set_yticklabels(
            [f'{v:+.1f}' if v != 0 else '0' for v in ytick_vals],
            fontsize=FONT_SIZE - 3,
        )

        # Highlight 0.0 baseline in red
        theta_circle = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_circle, np.zeros_like(theta_circle),
                linestyle='--', linewidth=1.8, color='red', alpha=0.8, zorder=4)

        ax.set_rlabel_position(22.5)

        for model_name, color in zip(all_model_names, model_colors):
            vals = all_values[scale][model_name] + [all_values[scale][model_name][0]]
            label = model_name if idx == 0 else None
            ax.plot(angles, vals, '-o', color=color, linewidth=1.5, markersize=4, label=label)
            ax.fill(angles, vals, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cat_labels, fontsize=7)
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            angle_deg = np.degrees(angle)
            if 90 < angle_deg < 270:
                label.set_ha('right')
            else:
                label.set_ha('left')
            label.set_va('center')

        ax.tick_params(axis='x', pad=12)
        ax.set_title(f'd = {scale}', fontsize=FONT_SIZE + 1, fontweight='bold', pad=18)

    # Add shared legend between the two columns at the top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=len(all_model_names), frameon=True, framealpha=0.9,
               fontsize=FONT_SIZE - 2, columnspacing=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, 'fig_radar_all_scales')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating figures...")
    gen_fig2_interaction_heatmap()
    gen_fig3_accuracy_bars()
    gen_fig4_math500_scatter()
    gen_fig5_per_category()
    gen_table2_main_results()
    gen_table4_rank_comparison()
    gen_fig_radar_d4()
    gen_fig_radar_all_scales()
    print("\nAll figures generated!")

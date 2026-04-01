"""Publication-quality grouped bar charts + normalized radar.
Following figures4papers style."""
import json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Publication style ──
PALETTE = {"cot": "#0F4D92", "ns": "#B64342"}
MODEL_COLORS = {
    'GPT-4o-mini': '#1f77b4', 'GPT-4.1-mini': '#9467bd',
    'Qwen3-30B': '#2ca02c', 'Qwen3-8B': '#ff7f0e', 'Llama-3.1-8B': '#d62728',
}

def apply_style(font_size=16):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': font_size,
        'axes.linewidth': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 14,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    })

DATA_DIR = Path(__file__).parent.parent / 'results'
CATEGORIES = [
    'magnitude_estimation', 'structural_shortcuts', 'relative_distance',
    'cancellation_identity', 'compatible_numbers', 'landmark_comparison',
    'equation_reasoning', 'option_elimination',
]
CAT_SHORT = ['ME', 'SS', 'RD', 'CI', 'CN', 'LC', 'ER', 'OE']
CAT_RADAR = ['Magnitude\nEst.', 'Structural', 'Relative\nDist.', 'Cancel.',
             'Compatible', 'Landmark', 'Equation', 'Option\nElim.']
MODEL_ORDER = ['GPT-4o-mini', 'GPT-4.1-mini', 'Qwen3-30B', 'Qwen3-8B', 'Llama-3.1-8B']

def extract_mc(resp):
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    m = re.search(r'\\boxed\{([A-D])\}', resp)
    if m: return m.group(1).upper()
    m = re.search(r'(?:final answer|answer)\s*(?:\([A-D/]+\))?\s*[:\s]\s*([A-D])', resp, re.IGNORECASE)
    if m: return m.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', resp)
    return matches[-1].upper() if matches else ''

def load_accuracy(scale):
    results = {}
    for name, fname in [('GPT-4o-mini', 'gpt4omini_easyNC_all.json'), ('GPT-4.1-mini', 'gpt41mini_easyNC_all.json')]:
        data = json.load(open(DATA_DIR / fname))
        recs = [r for r in data if r['scale'] == scale]
        model_data = {}
        for cat in CATEGORIES:
            cot = [r for r in recs if r['cat']==cat and r['var']=='strong_shortcut' and r['cond']=='CoT']
            ns = [r for r in recs if r['cat']==cat and r['var']=='strong_shortcut' and r['cond']=='EasyNC']
            model_data[cat] = {
                'CoT': sum(1 for r in cot if r.get('ok'))/len(cot)*100 if cot else 0,
                'NC': sum(1 for r in ns if r.get('ok'))/len(ns)*100 if ns else 0,
            }
        results[name] = model_data
    for name, key in [('Qwen3-30B','Qwen_Qwen3-30B-A3B-Instruct-2507'),('Qwen3-8B','Qwen_Qwen3-8B'),('Llama-3.1-8B','meta-llama_Llama-3.1-8B-Instruct')]:
        data = json.load(open(DATA_DIR / f'use_{key}_d{scale}.json'))
        model_data = {}
        for cat in CATEGORIES:
            cot = [r for r in data if r['category']==cat and r['variant']=='strong_shortcut' and r['condition']=='CoT']
            ns = [r for r in data if r['category']==cat and r['variant']=='strong_shortcut' and r['condition']=='NC']
            model_data[cat] = {
                'CoT': sum(1 for r in cot if extract_mc(r.get('raw_response',''))==r.get('ground_truth_letter',''))/len(cot)*100 if cot else 0,
                'NC': sum(1 for r in ns if extract_mc(r.get('raw_response',''))==r.get('ground_truth_letter',''))/len(ns)*100 if ns else 0,
            }
        results[name] = model_data
    return results

def gen_bar_single_scale(scale):
    """One figure per scale: 5 models, CoT vs NS grouped bars, adaptive y-axis."""
    apply_style()
    results = load_accuracy(scale)
    n_cats = len(CATEGORIES)
    n_models = len(MODEL_ORDER)

    # Compute global y range across all models
    all_vals = []
    for model in MODEL_ORDER:
        for cat in CATEGORIES:
            all_vals.extend([results[model][cat]['CoT'], results[model][cat]['NC']])
    global_min = min(all_vals)
    global_max = max(all_vals)
    y_min = max(0, int(global_min / 5) * 5 - 5)  # round down to nearest 5
    y_max = min(105, int(global_max / 5) * 5 + 10)

    fig, axes = plt.subplots(1, n_models, figsize=(22, 5), sharey=True)
    x = np.arange(n_cats)
    bar_width = 0.35

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        cot_vals = [results[model][cat]['CoT'] for cat in CATEGORIES]
        ns_vals = [results[model][cat]['NC'] for cat in CATEGORIES]

        ax.bar(x - bar_width/2, cot_vals, bar_width,
               label='CoT' if idx == 0 else None,
               color=PALETTE['cot'], edgecolor='black', linewidth=1.2, alpha=0.9)
        ax.bar(x + bar_width/2, ns_vals, bar_width,
               label='NS' if idx == 0 else None,
               color=PALETTE['ns'], edgecolor='black', linewidth=1.2, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(CAT_SHORT, fontsize=12, fontweight='bold')
        ax.set_title(model, fontsize=15, fontweight='bold', pad=10)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', length=0)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=14, frameon=False, columnspacing=2)
    fig.tight_layout(pad=2)
    fig.subplots_adjust(top=0.88)

    fname = f'fig_bar_d{scale}'
    fig.savefig(f'{fname}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f'Saved {fname}.pdf')

def gen_normalized_radar_d4():
    """Radar plot using normalized metric: (NS-CoT)/(1-CoT) at d=4.
    This measures what fraction of the remaining improvement space NS captures."""
    apply_style(font_size=12)
    results = load_accuracy(4)

    N = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))

    # Compute normalized values per model
    for model in MODEL_ORDER:
        vals = []
        for cat in CATEGORIES:
            cot = results[model][cat]['CoT'] / 100.0
            ns = results[model][cat]['NC'] / 100.0
            if cot >= 0.999:  # ceiling, no room
                norm = 0.0
            else:
                norm = (ns - cot) / (1.0 - cot)
            vals.append(norm)
        vals_plot = vals + [vals[0]]
        color = MODEL_COLORS[model]
        ax.plot(angles, vals_plot, '-o', color=color, linewidth=2, markersize=5, label=model)
        ax.fill(angles, vals_plot, color=color, alpha=0.06)

    # Zero reference line
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_circle, np.zeros_like(theta_circle),
            linestyle='--', linewidth=1.8, color='red', alpha=0.7, zorder=4)

    # Axis setup
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CAT_RADAR, fontsize=10)
    ax.tick_params(axis='x', pad=15)

    # Y range
    ax.set_ylim(-1.5, 1.5)
    yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{v:+.1f}' if v != 0 else '0' for v in yticks], fontsize=9)
    ax.set_rlabel_position(22.5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1),
              frameon=True, framealpha=0.9, fontsize=11, labelspacing=0.6)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig('fig_radar_normalized_d4.pdf', dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('Saved fig_radar_normalized_d4.pdf')

    # Print values for verification
    print('\nNormalized (NS-CoT)/(1-CoT) at d=4:')
    print(f'{"":>15s}', '  '.join(f'{c:>5s}' for c in CAT_SHORT))
    for model in MODEL_ORDER:
        row = []
        for cat in CATEGORIES:
            cot = results[model][cat]['CoT'] / 100.0
            ns = results[model][cat]['NC'] / 100.0
            norm = (ns - cot) / (1.0 - cot) if cot < 0.999 else 0.0
            row.append(f'{norm:+.2f}')
        print(f'{model:>15s}', '  '.join(row))

if __name__ == '__main__':
    for scale in [2, 4, 8, 16]:
        gen_bar_single_scale(scale)
    gen_normalized_radar_d4()
    print('\nAll figures generated!')

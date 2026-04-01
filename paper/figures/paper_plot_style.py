"""Shared plotting style for SenseMath v2 paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FONT_SIZE = 10
DPI = 300
FIG_DIR = '.'

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

COLORS = plt.cm.tab10.colors
# Model display names
MODEL_NAMES = {
    'Qwen/Qwen3-30B-A3B-Instruct-2507': 'Qwen3-30B',
    'Qwen3-30B-A3B-Instruct-2507': 'Qwen3-30B',
    'Qwen/Qwen3-8B': 'Qwen3-8B',
    'Qwen3-8B': 'Qwen3-8B',
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'gpt-5-mini': 'GPT-5-mini',
    'gpt-4o-mini': 'GPT-4o-mini',
    'gpt-4.1-mini': 'GPT-4.1-mini',
    'claude-haiku-4-5-20251001': 'Haiku-4.5',
    'gpt-5-mini-strict': 'GPT-5-mini (strict)',
}

MODEL_ORDER = ['GPT-4o-mini', 'GPT-4.1-mini', 'Qwen3-30B', 'Qwen3-8B', 'Llama-3.1-8B']
MODEL_COLORS = {
    'Qwen3-30B': COLORS[0],
    'Qwen3-8B': COLORS[1],
    'Llama-3.1-8B': COLORS[2],
    'GPT-5-mini': COLORS[3],
    'GPT-4o-mini': COLORS[4],
    'GPT-4.1-mini': '#9467bd',
    'Haiku-4.5': COLORS[5],
}

def short_name(model):
    return MODEL_NAMES.get(model, model.split('/')[-1][:20])

def save_fig(fig, name, fmt='pdf'):
    fig.savefig(f'{FIG_DIR}/{name}.{fmt}')
    print(f'Saved: {FIG_DIR}/{name}.{fmt}')

#!/usr/bin/env python3
"""Generate Figure 1: Readability vs Controllability gap across models."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 9

SCRIPT_DIR = "/Users/arth_rogerthat/Downloads/Papers/readibility-isnt-controlobility/ICML-2026"

# Load probe data
models = {
    "Llama-3.1-8B": {
        "probe_file": f"{SCRIPT_DIR}/readability_results/readability_llama3.1-8b.json",
        "ctrl_file": f"{SCRIPT_DIR}/experiment_results/controllability_llama3.1-8b.json",
        "num_layers": 32, "ctrl_layer": 19,
    },
    "Mistral-7B": {
        "probe_file": f"{SCRIPT_DIR}/readability_results/readability_mistral-7b.json",
        "ctrl_file": f"{SCRIPT_DIR}/experiment_results/controllability_mistral-7b.json",
        "num_layers": 32, "ctrl_layer": 16,
    },
    "Qwen3-8B": {
        "probe_file": f"{SCRIPT_DIR}/readability_results/readability_qwen3-8b.json",
        "ctrl_file": f"{SCRIPT_DIR}/experiment_results/controllability_qwen3-8b.json",
        "num_layers": 36, "ctrl_layer": 22,
    },
    "Gemma-3-27B": {
        "probe_file": f"{SCRIPT_DIR}/experiment_results/readability_gemma3-27b.json",
        "ctrl_file": None,  # No per-layer controllability sweep for Gemma in new format
        "num_layers": 62, "ctrl_layer": 45,
    },
}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
axes = axes.flatten()

colors = {
    "sep": "#2166AC",
    "probe": "#4393C3",
    "ctrl": "#D6604D",
    "gap": "#FDDBC7",
}

for idx, (model_name, info) in enumerate(models.items()):
    ax = axes[idx]
    L = info["num_layers"]

    # Load probe/separation data
    with open(info["probe_file"]) as f:
        probe_data = json.load(f)

    # Extract layer-wise probe accuracy
    probe_layers = []
    probe_accs = []
    for layer_str, v in sorted(probe_data["probe_by_layer"].items(), key=lambda x: int(x[0])):
        probe_layers.append(int(layer_str))
        probe_accs.append(v["accuracy"])

    # Extract layer-wise separation (normalized to 0-1 for plotting)
    sep_layers = []
    sep_vals = []
    for layer_str, v in sorted(probe_data["separation_by_layer"].items(), key=lambda x: int(x[0])):
        sep_layers.append(int(layer_str))
        sep_vals.append(v["l2_separation"])

    # Normalize separation to 0-1 range for dual-axis plotting
    sep_max = max(sep_vals) if sep_vals else 1
    sep_norm = [s / sep_max for s in sep_vals]

    # Convert to fraction of L
    probe_frac = [l / L for l in probe_layers]
    sep_frac = [l / L for l in sep_layers]

    # Load controllability data if available
    ctrl_fracs = []
    ctrl_asrs = []
    if info["ctrl_file"]:
        with open(info["ctrl_file"]) as f:
            ctrl_data = json.load(f)
        # Get best ASR per layer (max over strengths)
        layer_best = {}
        for config in ctrl_data["configs"]:
            layer = config["layer"]
            asr = config["asr"]
            if layer not in layer_best or asr > layer_best[layer]:
                layer_best[layer] = asr
        for layer in sorted(layer_best.keys()):
            ctrl_fracs.append(layer / L)
            ctrl_asrs.append(layer_best[layer] / 100.0)  # normalize to 0-1

    # Plot
    ax.plot(sep_frac, sep_norm, color=colors["sep"], linewidth=1.5, label="L2 Sep. (norm.)", zorder=3)
    ax.plot(probe_frac, probe_accs, color=colors["probe"], linewidth=1.5, linestyle="--", label="Probe Acc.", zorder=3)

    if ctrl_fracs:
        ax.plot(ctrl_fracs, ctrl_asrs, color=colors["ctrl"], linewidth=2.0, marker="o", markersize=4, label="Dual ASR", zorder=4)

    # Mark controllability peak
    ctrl_frac = info["ctrl_layer"] / L
    ax.axvline(ctrl_frac, color=colors["ctrl"], linewidth=0.8, linestyle=":", alpha=0.7)

    # Mark separation peak
    sep_peak_layer = probe_data["peak_separation_layer"]
    sep_peak_frac = sep_peak_layer / L
    ax.axvline(sep_peak_frac, color=colors["sep"], linewidth=0.8, linestyle=":", alpha=0.7)

    # Shade the gap region
    gap_left = min(ctrl_frac, sep_peak_frac)
    gap_right = max(ctrl_frac, sep_peak_frac)
    ax.axvspan(gap_left, gap_right, alpha=0.15, color=colors["gap"], zorder=1)

    # Labels
    gap_pct = abs(sep_peak_layer - info["ctrl_layer"]) / L * 100
    ax.set_title(f"{model_name} (gap: {gap_pct:.0f}%)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer (fraction of $L$)", fontsize=8)
    ax.set_ylabel("Normalized score", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)

    if idx == 0:
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

plt.tight_layout(pad=0.8)
out_path = f"{SCRIPT_DIR}/figures/fig1_gap.pdf"
import os
os.makedirs(f"{SCRIPT_DIR}/figures", exist_ok=True)
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print(f"Saved: {out_path}")
print(f"Saved: {out_path.replace('.pdf', '.png')}")

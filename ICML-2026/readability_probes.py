#!/usr/bin/env python3
"""
Readability Probes: Layer-wise refusal direction separation + logistic regression probes
=========================================================================================

For each model (Llama-3.1-8B, Qwen3-8B, Mistral-7B):
  1. Load model onto a single specified GPU
  2. Forward pass 100 harmful + 100 harmless prompts (no generation, just encoding)
  3. Extract last-token hidden states at EVERY layer
  4. Compute refusal direction separation (cosine distance of mean difference) per layer
  5. Train logistic regression probe per layer (5-fold CV, Nadeau-Bengio corrected CIs)
  6. Save results as JSON

Optimized for RTX Pro 6000 Blackwell (97GB VRAM):
  - 8B models fit entirely in bfloat16 (~16GB), plenty of headroom
  - Batch size 32 to minimize forward passes
  - TF32 enabled for faster matmul on Blackwell
  - No generation, forward pass only — no API keys needed

Usage:
  CUDA_VISIBLE_DEVICES=3 python readability_probes.py --model llama
  CUDA_VISIBLE_DEVICES=3 python readability_probes.py --model qwen
  CUDA_VISIBLE_DEVICES=3 python readability_probes.py --model mistral
  CUDA_VISIBLE_DEVICES=3 python readability_probes.py --model all
"""

import torch
import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

RANDOM_SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROMPT_PAIRS_PATH = os.path.join(PROJECT_ROOT, "data", "refusal_direction_pairs.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "readability_results")

MODEL_CONFIGS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "short_name": "llama3.1-8b",
        "num_layers": 32,
        "hidden_dim": 4096,
        "batch_size": 32,
    },
    "qwen": {
        "name": "Qwen/Qwen3-8B",
        "short_name": "qwen3-8b",
        "num_layers": 36,
        "hidden_dim": 4096,
        "batch_size": 32,
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "short_name": "mistral-7b",
        "num_layers": 32,
        "hidden_dim": 4096,
        "batch_size": 32,
    },
}


def get_model_layers(model):
    """Get transformer layer modules from any HF causal LM."""
    if hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
            return inner.language_model.layers
        if hasattr(inner, 'layers'):
            return inner.layers
    raise RuntimeError(f"Cannot find layers. Structure: {[n for n, _ in model.named_children()]}")


def extract_hidden_from_output(output):
    """Extract hidden state tensor from layer output (handles tuple or tensor)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    return None


def load_prompt_pairs():
    with open(PROMPT_PAIRS_PATH, "r") as f:
        data = json.load(f)
    return [(p["harmful"], p["harmless"]) for p in data["pairs"]]


def extract_all_layer_hiddens(model, tokenizer, layers, prompts, model_key, batch_size):
    """
    Forward pass only (no generation). Extract last-token hidden states at every layer.
    Returns: {layer_idx: np.ndarray of shape [n_prompts, hidden_dim]}
    """
    num_layers = len(layers)
    is_qwen = model_key == "qwen"
    device = next(model.parameters()).device

    all_hiddens = {i: [] for i in range(num_layers)}

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]

        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if is_qwen:
            template_kwargs["enable_thinking"] = False

        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], **template_kwargs
            )
            for p in batch
        ]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(device)

        # Get attention mask to find actual last token per sequence
        attention_mask = inputs["attention_mask"]
        # Last real token index per sequence: sum of mask - 1
        last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]

        captured = {}
        handles = []

        for layer_idx in range(num_layers):
            captured[layer_idx] = [None]

            def make_hook(idx):
                def hook(module, inp, output):
                    hidden = extract_hidden_from_output(output)
                    if hidden is not None:
                        # Extract last REAL token (not padding) per sequence
                        # Keep all index tensors on same device as hidden
                        dev = hidden.device
                        bi = torch.arange(hidden.size(0), device=dev)
                        li = last_token_indices.to(dev)
                        last_hidden = hidden[bi, li, :]
                        captured[idx][0] = last_hidden.detach().float().cpu()
                return hook

            handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.inference_mode():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()

        for layer_idx in range(num_layers):
            if captured[layer_idx][0] is not None:
                all_hiddens[layer_idx].append(captured[layer_idx][0])

        done = min(batch_start + batch_size, len(prompts))
        print(f"    [{done}/{len(prompts)}]", flush=True)

    result = {}
    for layer_idx in range(num_layers):
        if all_hiddens[layer_idx]:
            result[layer_idx] = torch.cat(all_hiddens[layer_idx], dim=0).numpy()

    return result


def compute_separation(harmful_hiddens, harmless_hiddens, num_layers):
    """
    Compute refusal direction separation at each layer.
    Separation = L2 norm of (mean_harmful - mean_harmless), normalized by hidden dim.
    Also computes cosine separation.
    """
    results = {}
    for layer_idx in range(num_layers):
        if layer_idx not in harmful_hiddens or layer_idx not in harmless_hiddens:
            continue

        h_harm = harmful_hiddens[layer_idx]
        h_safe = harmless_hiddens[layer_idx]

        mean_harm = h_harm.mean(axis=0)
        mean_safe = h_safe.mean(axis=0)
        diff = mean_harm - mean_safe

        l2_sep = float(np.linalg.norm(diff))
        cosine_sep = float(
            np.dot(mean_harm, mean_safe) / (np.linalg.norm(mean_harm) * np.linalg.norm(mean_safe) + 1e-10)
        )

        results[layer_idx] = {
            "l2_separation": l2_sep,
            "cosine_similarity": cosine_sep,
            "cosine_separation": 1.0 - cosine_sep,
        }

    return results


def train_probes(harmful_hiddens, harmless_hiddens, num_layers):
    """
    Train logistic regression probe at each layer with 5-fold CV.
    CI via Nadeau-Bengio corrected resampled t-test.
    """
    results = {}

    for layer_idx in range(num_layers):
        if layer_idx not in harmful_hiddens or layer_idx not in harmless_hiddens:
            continue

        X_harm = harmful_hiddens[layer_idx]
        X_safe = harmless_hiddens[layer_idx]

        n = min(len(X_harm), len(X_safe))
        X = np.concatenate([X_harm[:n], X_safe[:n]], axis=0)
        y = np.array([1] * n + [0] * n)

        k = 5
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()

        n_test = len(X) // k
        n_train = len(X) - n_test
        corrected_var = (1.0 / k + n_test / n_train) * np.var(scores, ddof=1)
        corrected_se = np.sqrt(corrected_var)
        t_crit = stats.t.ppf(0.975, df=k - 1)
        ci_low = max(0.0, mean_acc - t_crit * corrected_se)
        ci_high = min(1.0, mean_acc + t_crit * corrected_se)

        results[layer_idx] = {
            "accuracy": float(mean_acc),
            "std": float(scores.std()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "cv_scores": [float(s) for s in scores],
            "n_samples": 2 * n,
        }

        print(f"    L{layer_idx:2d}: {mean_acc:.3f} [{ci_low:.3f}, {ci_high:.3f}]", flush=True)

    return results


def run_model(model_key):
    config = MODEL_CONFIGS[model_key]

    print("=" * 70)
    print(f"READABILITY PROBES: {config['name']}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    prompt_pairs = load_prompt_pairs()
    harmful_prompts = [h for h, _ in prompt_pairs]
    harmless_prompts = [s for _, s in prompt_pairs]
    print(f"  {len(prompt_pairs)} prompt pairs loaded")

    # Load model — single GPU, bfloat16
    print(f"\n[Loading {config['name']}...]")
    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    layers = get_model_layers(model)
    num_layers = len(layers)
    print(f"  {num_layers} layers, hidden_dim={config['hidden_dim']}")

    # Enable TF32 for RTX Pro 6000
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Extract hidden states
    print(f"\n[Extracting harmful hidden states (100 prompts, {num_layers} layers)...]")
    harmful_hiddens = extract_all_layer_hiddens(
        model, tokenizer, layers, harmful_prompts, model_key, config["batch_size"]
    )

    print(f"\n[Extracting harmless hidden states (100 prompts, {num_layers} layers)...]")
    harmless_hiddens = extract_all_layer_hiddens(
        model, tokenizer, layers, harmless_prompts, model_key, config["batch_size"]
    )

    # Free GPU memory — don't need the model anymore
    del model
    torch.cuda.empty_cache()
    print(f"\n  Model unloaded, GPU memory freed")

    # Compute refusal direction separation
    print(f"\n[Computing refusal direction separation at {num_layers} layers...]")
    separation_results = compute_separation(harmful_hiddens, harmless_hiddens, num_layers)
    for layer_idx in sorted(separation_results.keys()):
        s = separation_results[layer_idx]
        print(f"    L{layer_idx:2d}: L2={s['l2_separation']:.1f}  cos_sep={s['cosine_separation']:.4f}")

    # Train probes
    print(f"\n[Training logistic regression probes at {num_layers} layers...]")
    probe_results = train_probes(harmful_hiddens, harmless_hiddens, num_layers)

    # Find peaks
    peak_sep_layer = max(separation_results, key=lambda k: separation_results[k]["l2_separation"])
    peak_sep_val = separation_results[peak_sep_layer]["l2_separation"]

    peak_probe_layer = max(probe_results, key=lambda k: probe_results[k]["accuracy"])
    peak_probe_acc = probe_results[peak_probe_layer]["accuracy"]

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY: {config['short_name']}")
    print(f"{'='*70}")
    print(f"  Total layers: {num_layers}")
    print(f"  Peak separation: L{peak_sep_layer} ({peak_sep_layer/num_layers:.2f}L) = {peak_sep_val:.1f}")
    print(f"  Peak probe acc:  L{peak_probe_layer} ({peak_probe_layer/num_layers:.2f}L) = {peak_probe_acc:.3f}")

    # Save
    output = {
        "model_key": model_key,
        "model_name": config["name"],
        "short_name": config["short_name"],
        "num_layers": num_layers,
        "hidden_dim": config["hidden_dim"],
        "n_harmful_prompts": len(harmful_prompts),
        "n_harmless_prompts": len(harmless_prompts),
        "peak_separation_layer": peak_sep_layer,
        "peak_separation_value": peak_sep_val,
        "peak_separation_fraction": peak_sep_layer / num_layers,
        "peak_probe_layer": peak_probe_layer,
        "peak_probe_accuracy": peak_probe_acc,
        "peak_probe_fraction": peak_probe_layer / num_layers,
        "separation_by_layer": {str(k): v for k, v in separation_results.items()},
        "probe_by_layer": {str(k): v for k, v in probe_results.items()},
        "timestamp": datetime.now().isoformat(),
        "seed": RANDOM_SEED,
    }

    out_file = os.path.join(RESULTS_DIR, f"readability_{config['short_name']}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")
    print(f"  Done at {datetime.now().isoformat()}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Readability probes (separation + logistic regression)")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        help="Model to evaluate, or 'all' to run sequentially")
    args = parser.parse_args()

    if args.model == "all":
        results = {}
        for model_key in MODEL_CONFIGS:
            results[model_key] = run_model(model_key)
            print("\n\n")

        # Print cross-model summary
        print("=" * 70)
        print("CROSS-MODEL READABILITY SUMMARY")
        print("=" * 70)
        for model_key, r in results.items():
            print(f"  {r['short_name']:15s}  sep_peak=L{r['peak_separation_layer']} ({r['peak_separation_fraction']:.2f}L)"
                  f"  probe_peak=L{r['peak_probe_layer']} ({r['peak_probe_fraction']:.2f}L)")
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()

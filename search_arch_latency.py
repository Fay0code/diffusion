#!/usr/bin/env python3
"""Search FFN/layer configurations with vLLM's dummy weight loader.

The controller creates a config-only model directory for every candidate and
starts a fresh worker process.  A fresh process is intentional: it releases
device memory and graph/compiler state between architectures.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


# ======================== Search space (edit here) ========================
# These defaults deliberately live in Python rather than in a shell script.
# FFN dimensions are aligned to 64, which is also friendly to accelerator
# kernels.  With 13 layer choices and 65 FFN choices this produces 845 raw
# combinations; --only-larger (enabled by default) removes combinations whose
# estimated parameter count is not larger than the base model.
SEARCH_LAYERS = list(range(24, 37))
SEARCH_FFN_DIMS = list(range(4096, 8192 + 1, 64))


def int_list(value: str) -> list[int]:
    try:
        values = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def config_value(config: dict, *names: str) -> int:
    for name in names:
        if name in config:
            return int(config[name])
    raise KeyError(f"none of {names!r} is present in config.json")


def estimate_dense_parameters(config: dict) -> int:
    """Estimate dense decoder parameters (sufficient for ordering candidates)."""
    hidden = config_value(config, "hidden_size", "n_embd")
    layers = config_value(config, "num_hidden_layers", "n_layer")
    ffn = config_value(config, "intermediate_size", "ffn_dim", "n_inner")
    heads = config_value(config, "num_attention_heads", "n_head")
    kv_heads = int(config.get("num_key_value_heads", heads))
    vocab = config_value(config, "vocab_size")
    head_dim = int(config.get("head_dim", hidden // heads))

    # Q + K + V + O.  Biases/norms are included below but are insignificant.
    attention = hidden * hidden + 2 * hidden * kv_heads * head_dim + hidden * hidden
    architectures = " ".join(config.get("architectures", [] )).lower()
    gated = any(x in architectures for x in ("qwen", "llama", "mistral", "gemma"))
    mlp = (3 if gated else 2) * hidden * ffn
    per_layer = attention + mlp + 2 * hidden
    embeddings = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings", False) else embeddings
    return embeddings + lm_head + layers * per_layer + hidden


def sync_device() -> None:
    import torch
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def run_worker(args: argparse.Namespace) -> int:
    # Imports stay in the worker so the controller never initializes a device.
    import torch
    from torch.autograd.profiler import record_function
    from vllm import LLM
    from vllm.sampling_params import BeamSearchParams

    model_dir = Path(args.worker_model_dir)
    llm = LLM(
        model=str(model_dir),
        tokenizer=args.tokenizer,
        load_format="dummy",
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, args.batch_size * args.prompt_tokens),
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=True,
        max_logprobs=max(args.max_logprobs, args.beam_width),
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": (
                [(j + 1) * 128 for j in range(32)]
                + [1, 2, 4, 8, 16, 32, 64]
            ),
        },
        seed=args.seed,
    )
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    token_id = min(max(1, int(config.get("bos_token_id", 1) or 1)), vocab_size - 1)
    inputs = [{"prompt_token_ids": [token_id] * args.prompt_tokens}
              for _ in range(args.batch_size)]
    beam_params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.output_tokens,
        padding=args.beam_padding,
    )

    # Keep this outside the timed region, exactly as generate_items_beam_search().
    llm.init_beam_search_parallel(inputs, params=beam_params)

    samples_ms = []
    enable_profile = args.torch_profile or "VLLM_TORCH_PROFILER_DIR" in os.environ
    for index in range(args.repeat):
        # Match run_inference.py: profile only the penultimate iteration.
        profile_this_iteration = enable_profile and index == args.repeat - 2
        if profile_this_iteration:
            llm.start_profile()
        start = time.time()
        with record_function("#### beam_search_parallel ###"):
            llm.beam_search_parallel(inputs, params=beam_params)
            sync_device()
        samples_ms.append((time.time() - start) * 1000.0)
        if profile_this_iteration:
            llm.stop_profile()

    result = {
        "status": "ok",
        "samples_ms": samples_ms,
        "profiled_ms": samples_ms[-2] if len(samples_ms) >= 2 else samples_ms[-1],
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
    }
    Path(args.worker_result).write_text(json.dumps(result), encoding="utf-8")
    return 0


def write_candidate_config(base: dict, directory: Path, layers: int, ffn: int) -> dict:
    config = dict(base)
    layer_key = "num_hidden_layers" if "num_hidden_layers" in config else "n_layer"
    if "intermediate_size" in config:
        ffn_key = "intermediate_size"
    elif "ffn_dim" in config:
        ffn_key = "ffn_dim"
    else:
        ffn_key = "n_inner"
    config[layer_key] = layers
    config[ffn_key] = ffn
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def run_controller(args: argparse.Namespace) -> int:
    model = Path(args.model).resolve()
    config_path = model / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"config.json not found: {config_path}")
    base = json.loads(config_path.read_text(encoding="utf-8"))
    base_params = estimate_dense_parameters(base)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "latency_results.csv"
    fields = ["is_baseline", "larger_and_faster", "layers", "ffn_dim",
              "estimated_parameters", "parameter_billions",
              "profiled_ms", "mean_ms", "median_ms", "min_ms", "samples_ms",
              "status", "error"]

    candidates = []
    for layers, ffn in itertools.product(args.layers, args.ffn_dims):
        candidate = dict(base)
        candidate["num_hidden_layers" if "num_hidden_layers" in candidate else "n_layer"] = layers
        candidate["intermediate_size" if "intermediate_size" in candidate else
                  ("ffn_dim" if "ffn_dim" in candidate else "n_inner")] = ffn
        params = estimate_dense_parameters(candidate)
        if not args.only_larger or params > base_params:
            candidates.append((params, layers, ffn))
    candidates.sort()
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]
    base_layers = config_value(base, "num_hidden_layers", "n_layer")
    base_ffn = config_value(base, "intermediate_size", "ffn_dim", "n_inner")
    # Always measure the unmodified architecture first with exactly the same
    # dummy input and beam-search path. This is the latency baseline used by
    # the larger-and-faster filter.
    jobs = [(base_params, base_layers, base_ffn, True)]
    jobs.extend((params, layers, ffn, False)
                for params, layers, ffn in candidates
                if (layers, ffn) != (base_layers, base_ffn))
    print(f"base estimated parameters: {base_params:,}; candidates: {len(jobs) - 1}")

    rows = []
    baseline_median_ms = None
    for index, (params, layers, ffn, is_baseline) in enumerate(jobs, 1):
        label = "baseline" if is_baseline else f"layers_{layers}_ffn_{ffn}"
        print(f"[{index}/{len(jobs)}] {label}, params~{params:,}", flush=True)
        case_dir = output / "configs" / label
        write_candidate_config(base, case_dir, layers, ffn)
        result_path = case_dir / "result.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--worker-model-dir", str(case_dir), "--worker-result", str(result_path)]
        for name in ("tensor_parallel_size", "dtype", "max_model_len", "batch_size",
                     "prompt_tokens", "output_tokens", "repeat", "beam_width",
                     "max_logprobs", "gpu_memory_utilization", "seed", "tokenizer"):
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        for flag in ("enforce_eager", "trust_remote_code", "torch_profile", "beam_padding"):
            if getattr(args, flag):
                cmd.append("--" + flag.replace("_", "-"))
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo_root / "vllm") + os.pathsep + env.get("PYTHONPATH", "")
        if args.torch_profile:
            profile_dir = output / "profiles" / f"layers_{layers}_ffn_{ffn}"
            profile_dir.mkdir(parents=True, exist_ok=True)
            env.setdefault("VLLM_TORCH_PROFILER_DIR", str(profile_dir))
            env.setdefault("VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY", "1")
            env.setdefault("VLLM_TORCH_PROFILER_WITH_STACK", "0")
        proc = subprocess.run(cmd, env=env, text=True)
        result = json.loads(result_path.read_text()) if proc.returncode == 0 and result_path.exists() else {
            "status": "failed", "error": f"worker exit code {proc.returncode}"}
        if is_baseline and result.get("status") == "ok":
            baseline_median_ms = float(result["median_ms"])
        larger_and_faster = (
            not is_baseline
            and baseline_median_ms is not None
            and params > base_params
            and result.get("status") == "ok"
            and float(result["median_ms"]) < baseline_median_ms
        )
        row = {"is_baseline": is_baseline,
               "larger_and_faster": larger_and_faster,
               "layers": layers, "ffn_dim": ffn, "estimated_parameters": params,
               "parameter_billions": f"{params / 1e9:.6f}",
               "profiled_ms": result.get("profiled_ms", ""),
               "mean_ms": result.get("mean_ms", ""),
               "median_ms": result.get("median_ms", ""), "min_ms": result.get("min_ms", ""),
               "samples_ms": json.dumps(result.get("samples_ms", [])),
               "status": result.get("status", "failed"), "error": result.get("error", "")}
        rows.append(row)
        if larger_and_faster:
            write_candidate_config(
                base,
                output / "larger_and_faster_configs" / label,
                layers,
                ffn,
            )
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        winners = [item for item in rows if item["larger_and_faster"]]
        with (output / "larger_and_faster.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(winners)
    print(f"results: {csv_path}")
    print(f"larger and faster: {output / 'larger_and_faster.csv'}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search layer/FFN dimensions using dummy vLLM weights")
    p.add_argument("--model", help="base Hugging Face model directory (config.json is required)")
    p.add_argument("--layers", type=int_list, default=SEARCH_LAYERS,
                   help="comma-separated layer counts; default comes from SEARCH_LAYERS")
    p.add_argument("--ffn-dims", type=int_list, default=SEARCH_FFN_DIMS,
                   help="comma-separated FFN sizes; default uses step 64")
    p.add_argument("--output", default="arch_latency_search")
    p.add_argument("--only-larger", action=argparse.BooleanOptionalAction, default=True,
                   help="only profile candidates larger than the base config (default: true)")
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--output-tokens", type=int, default=3,
                   help="stage-2 beam-search token count (run_inference default: 3)")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--beam-width", type=int, default=32)
    p.add_argument("--beam-padding", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-logprobs", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-profile", action="store_true",
                   help="also emit a torch profiler trace via VLLM_TORCH_PROFILER_DIR")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-model-dir", help=argparse.SUPPRESS)
    p.add_argument("--worker-result", help=argparse.SUPPRESS)
    p.add_argument("--tokenizer", help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.worker:
        return run_worker(args)
    if not args.model:
        raise SystemExit("--model is required")
    args.tokenizer = str(Path(args.model).resolve())
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())

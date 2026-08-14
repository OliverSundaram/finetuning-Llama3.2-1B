"""
Inference speed benchmark for the Llama-3.2-1B MathCodeInstruct models.

What this measures
------------------
Decode throughput (tokens/sec) under identical conditions for every model.

Three things make the numbers comparable, which the naive version of this
script did not do:

1. Every model generates EXACTLY the same number of tokens. `min_new_tokens`
   is set equal to `max_new_tokens`, so a model that would otherwise emit EOS
   early (the base model does this constantly on a raw, non-chat-template
   prompt) is forced to keep going. Without this you are dividing a fixed
   per-call overhead by wildly different token counts and calling the result
   "speed".

2. Prefill is timed separately and subtracted. Prefill is one parallel forward
   pass over the prompt; decode is N sequential forward passes. Mixing them
   into one average hides both.

3. Models are interleaved across rounds rather than run back-to-back, so GPU
   boost/throttle drift is spread evenly instead of penalising whichever model
   ran last.

Expected result
---------------
LoRA weights merged with `merged_16bit` change no tensor shapes, no parameter
count, and no dtype. All four models should therefore land within a few percent
of each other. A large spread means the harness is broken, not the model.

Usage (PowerShell, from the project root)
-----------------------------------------
python benchmark_speed.py `
    --model base=unsloth/Llama-3.2-1B `
    --model 5k=Llama-3.2-1B-MathCodeInstruct-5k/outputs/llama-3.2-1b-5k `
    --model 10k=Llama-3.2-1B-MathCodeInstruct-10k/outputs/llama-3.2-1b-10k `
    --model 20k=Llama-3.2-1B-MathCodeInstruct-20k/outputs/llama-3.2-1b-20k
"""

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in "
    "April and May? Solve step by step."
)

DEFAULT_TOKENS = 256
DEFAULT_ROUNDS = 3
WARMUP_RUNS = 1
PREFILL_REPS = 3

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "outputs" / "charts" / "speed_results.json"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def gpu_state() -> str:
    """Best-effort temperature / clock readout, for spotting thermal drift."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        temp, clock = (x.strip() for x in out.split(","))
        return f"{temp}C / {clock}MHz"
    except Exception:
        return "n/a"


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # transformers >=5 renamed torch_dtype -> dtype; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
    model.eval()
    return tokenizer, model


def unload_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gen_kwargs(tokenizer, n_tokens: int) -> dict:
    """
    min_new_tokens == max_new_tokens is the important part. The
    MinNewTokensLengthLogitsProcessor masks EOS to -inf until the minimum is
    reached, so no model can stop early no matter what its generation_config
    or chat template says.
    """
    return dict(
        max_new_tokens=n_tokens,
        min_new_tokens=n_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )


def timed_generate(model, inputs, kwargs) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - start, output


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure(model, tokenizer, n_tokens: int) -> dict:
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    full_kwargs = gen_kwargs(tokenizer, n_tokens)
    prefill_kwargs = gen_kwargs(tokenizer, 1)

    # Warm up: triggers kernel autotuning and sizes the caching allocator.
    for _ in range(WARMUP_RUNS):
        timed_generate(model, inputs, full_kwargs)

    # Prefill: one prompt forward pass + fixed per-call overhead.
    # Take the minimum, not the mean -- we want the floor, since any excess is
    # noise from the OS scheduler rather than real work.
    prefill_times = [timed_generate(model, inputs, prefill_kwargs)[0]
                     for _ in range(PREFILL_REPS)]
    prefill_s = min(prefill_times)

    # Full generation.
    full_s, output = timed_generate(model, inputs, full_kwargs)
    produced = output.shape[1] - prompt_len

    if produced != n_tokens:
        raise RuntimeError(
            f"expected exactly {n_tokens} new tokens, got {produced}. "
            "min_new_tokens did not take effect -- check the transformers "
            "version before trusting any of these numbers."
        )

    decode_s = full_s - prefill_s
    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": produced,
        "prefill_s": round(prefill_s, 4),
        "total_s": round(full_s, 4),
        "decode_tok_s": round((n_tokens - 1) / decode_s, 2),
        "end_to_end_tok_s": round(n_tokens / full_s, 2),
        "gpu": gpu_state(),
    }


def run_sweep(models, n_tokens, rounds, sequential) -> dict:
    runs = {name: [] for name, _ in models}

    if sequential:
        # Faster (one load per model) but every model sees a different point on
        # the GPU's thermal curve. Fine for a sanity check, not for publishing.
        for name, path in models:
            print(f"\n=== {name} ({path}) ===")
            tokenizer, model = load_model(path)
            for r in range(rounds):
                res = measure(model, tokenizer, n_tokens)
                print(f"  round {r + 1}: {res['decode_tok_s']:>7.2f} tok/s decode  "
                      f"| prefill {res['prefill_s'] * 1000:.0f} ms  | {res['gpu']}")
                runs[name].append(res)
            unload_model(model)
    else:
        for r in range(rounds):
            print(f"\n=== round {r + 1} / {rounds} ===")
            for name, path in models:
                tokenizer, model = load_model(path)
                res = measure(model, tokenizer, n_tokens)
                print(f"  {name:<6} {res['decode_tok_s']:>7.2f} tok/s decode  "
                      f"| prefill {res['prefill_s'] * 1000:.0f} ms  | {res['gpu']}")
                runs[name].append(res)
                unload_model(model)

    return runs


def summarise(name, path, run_list) -> dict:
    decode = [r["decode_tok_s"] for r in run_list]
    e2e = [r["end_to_end_tok_s"] for r in run_list]
    prefill = [r["prefill_s"] for r in run_list]
    return {
        "model_path": path,
        # Median, not mean: one throttled round shouldn't drag the headline
        # number down.
        "decode_tok_s": round(statistics.median(decode), 2),
        "decode_tok_s_min": round(min(decode), 2),
        "decode_tok_s_max": round(max(decode), 2),
        "end_to_end_tok_s": round(statistics.median(e2e), 2),
        "prefill_ms": round(statistics.median(prefill) * 1000, 1),
        "prompt_tokens": run_list[0]["prompt_tokens"],
        "generated_tokens": run_list[0]["generated_tokens"],
        "rounds": run_list,
    }


# --------------------------------------------------------------------------

def parse_model_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--model needs the form name=path, got {value!r}"
        )
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"bad --model value: {value!r}")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare decode throughput across models on identical work."
    )
    parser.add_argument(
        "--model", action="append", required=True, type=parse_model_arg,
        metavar="NAME=PATH",
        help="repeat once per model, e.g. --model 5k=outputs/llama-3.2-1b-5k",
    )
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"tokens to generate per run (default {DEFAULT_TOKENS})")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"timed rounds per model (default {DEFAULT_ROUNDS})")
    parser.add_argument("--sequential", action="store_true",
                        help="finish each model before starting the next "
                             "(faster, but exposed to thermal drift)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="where to write the JSON results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA device found.")

    models = args.model
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Torch  : {torch.__version__}")
    print(f"Models : {', '.join(n for n, _ in models)}")
    print(f"Config : {args.tokens} tokens x {args.rounds} rounds, "
          f"{'sequential' if args.sequential else 'interleaved'}")

    runs = run_sweep(models, args.tokens, args.rounds, args.sequential)

    results = {
        "config": {
            "prompt": PROMPT,
            "tokens_per_run": args.tokens,
            "rounds": args.rounds,
            "interleaved": not args.sequential,
            "dtype": "bfloat16",
            "greedy": True,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "models": {
            name: summarise(name, path, runs[name]) for name, path in models
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "-" * 58)
    print(f"{'model':<10}{'decode tok/s':>14}{'spread':>14}{'prefill ms':>14}")
    print("-" * 58)
    for name, _ in models:
        s = results["models"][name]
        spread = f"{s['decode_tok_s_min']:.1f}-{s['decode_tok_s_max']:.1f}"
        print(f"{name:<10}{s['decode_tok_s']:>14.2f}{spread:>14}{s['prefill_ms']:>14.1f}")
    print("-" * 58)

    best = max(results["models"].values(), key=lambda s: s["decode_tok_s"])
    worst = min(results["models"].values(), key=lambda s: s["decode_tok_s"])
    gap = (best["decode_tok_s"] - worst["decode_tok_s"]) / worst["decode_tok_s"] * 100
    print(f"spread across models: {gap:.1f}%")
    if gap > 10:
        print("  -> larger than expected. These models are architecturally "
              "identical, so suspect the harness (background GPU load, "
              "thermal throttling, a model that didn't load in bf16) before "
              "reporting this as a real difference.")

    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
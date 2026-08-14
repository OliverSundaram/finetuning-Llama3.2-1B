import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in "
    "April and May? Solve step by step."
)

NUM_NEW_TOKENS = 256
NUM_TIMED_RUNS = 5
NUM_WARMUP_RUNS = 2
OUT_PATH = Path("outputs/charts/speed_results.json")


def benchmark(model_path: str) -> dict:
    print(f"Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    gen_kwargs = dict(
        max_new_tokens=NUM_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(f"Warming up ({NUM_WARMUP_RUNS} runs)...")
    for _ in range(NUM_WARMUP_RUNS):
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()

    print(f"Timing ({NUM_TIMED_RUNS} runs)...")
    times = []
    generated_tokens = 0
    for i in range(NUM_TIMED_RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        print(f"  run {i + 1}: {elapsed:.2f}s ({generated_tokens / elapsed:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    tokens_per_sec = generated_tokens / avg_time

    result = {
        "model_path": model_path,
        "generated_tokens_per_run": generated_tokens,
        "avg_seconds_per_run": round(avg_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "num_timed_runs": NUM_TIMED_RUNS,
    }

    print(f"\nResult: {tokens_per_sec:.2f} tokens/sec (avg over {NUM_TIMED_RUNS} runs)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or HF repo id of the model to benchmark")
    parser.add_argument("--name", required=True, help="Short label for this run, e.g. base / 5k / 10k / 20k")
    args = parser.parse_args()

    result = benchmark(args.model)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    all_results[args.name] = result
    OUT_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")
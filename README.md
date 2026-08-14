# Llama-3.2-1B-MathCodeInstruct

Fine-tuning Llama-3.2-1B on the [MathCodeInstruct](https://huggingface.co/datasets/MathLLMs/MathCodeInstruct)
dataset at three data scales (5k / 10k / 20k examples), benchmarked across math, general knowledge,
commonsense reasoning, and inference speed to see what scaling fine-tuning data actually buys you —
and what it costs.

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Unsloth](https://img.shields.io/badge/fine--tuned%20with-Unsloth-orange.svg)

**Models on Hugging Face:**
[5k](https://huggingface.co/OliverSundaram/Llama-3.2-1B-MathCodeInstruct-5k) ·
[10k](https://huggingface.co/OliverSundaram/Llama-3.2-1B-MathCodeInstruct-10k) ·
[20k](https://huggingface.co/OliverSundaram/Llama-3.2-1B-MathCodeInstruct-20k)

---

## Table of contents

- [TL;DR](#tldr)
- [Motivation](#motivation)
- [Repo structure](#repo-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Training](#training)
- [Challenges & debugging](#challenges--debugging)
- [Evaluation](#evaluation)
- [Results](#results)
- [Key findings](#key-findings)
- [Models](#models)
- [Reproducing this project](#reproducing-this-project)
- [Environment](#environment)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Acknowledgments & citations](#acknowledgments--citations)
- [License](#license)

---

## TL;DR

> *GSM8K accuracy had a diminutive, but noticeable increase as fine-tuning data expanded,
> but Arc-Challenge and HellaSwag accuracy dropped simultaneously with data size,
> suggesting a capability trade off — develop math skills at the expense of deteriorated general reasoning*

## Motivation

I wanted to answer a question that's easy to state but that most single-model fine-tuning write-ups
don't actually test: **if you fine-tune a small model on a narrow domain, what happens to everything
else it can do?** Most math/code fine-tune posts report the target benchmark going up and stop there.
This project trains the *same* base model on the *same* dataset at three different sizes, then runs the
*same* battery of general-purpose benchmarks against all three — plus the untouched base model — so any
change is attributable to the fine-tuning itself, not to a different setup.

This was done as a self-directed project for my ML portfolio, run entirely on a single consumer GPU
(RTX 4060, 8GB VRAM) rather than rented cloud compute — part of the point was seeing what's actually
achievable on hardware a student would realistically have.

## Repo structure

```
Llama-3.2-1B-MathCodeInstruct/
├── data/
│   ├── mathcodeinstruct-train-*.json  # per-subset train splits (5k/10k/20k, nested)
│   ├── mathcodeinstruct-eval.json     # ONE shared held-out eval set, used by all three
│   └── prepare_data.ipynb             # filtering, chat-template formatting, splitting
├── Llama-3.2-1B-MathCodeInstruct-5k/
│   ├── 01_training.ipynb
│   └── outputs/llama-3.2-1b-5k/       # merged 16-bit model
├── Llama-3.2-1B-MathCodeInstruct-10k/  # same layout, 10k subset
├── Llama-3.2-1B-MathCodeInstruct-20k/  # same layout, 20k subset
├── visualize_results.py               # builds MMLU-by-subject chart + comparison table per subset
├── measure_speed.py                   # decode-throughput benchmark across all 4 models
├── pushing_to_hf.py                   # pushes each merged model + card to the Hub
├── .gitignore
└── README.md
```

## Dataset

[MathLLMs/MathCodeInstruct](https://huggingface.co/datasets/MathLLMs/MathCodeInstruct) pairs math word
problems with GPT-4-Code-Interpreter-style solutions that interleave natural-language reasoning, Python
code, and code execution results (the "LCE" format — Language, Code, Execution).

**Preprocessing** (`data/prepare_data.ipynb`):
1. Loaded the full dataset (79,067 rows) and formatted each example with the `unsloth/Llama-3.2-1B`
   tokenizer's Llama 3.1 chat template (`system` prompt: *"Below is a math problem. Please solve it step
   by step."*), rendering each typed content block (text / code / execution) into a formatted string.
2. Filtered to examples that fit in 2048 tokens, leaving **78,201** usable rows, then shuffled with a
   fixed seed (123) for reproducibility.
3. Held out the **last 1,000 rows as a single shared evaluation set**, used to track validation loss
   during training for all three subsets alike.
4. From the remaining 77,201-row training pool, took the first **5,000 / 10,000 / 20,000** rows for each
   subset — meaning the subsets are **nested, not disjoint**: every example in the 5k run also appears in
   the 10k run, and every example in the 10k run also appears in the 20k run.

| Subset | Train rows | Eval rows (shared) |
|---|---|---|
| 5k | first 5,000 of the training pool | last 1,000 of the full filtered dataset |
| 10k | first 10,000 of the training pool | same shared eval set |
| 20k | first 20,000 of the training pool | same shared eval set |

The nesting is intentional, and it's actually the methodologically cleaner choice for this kind of
scaling study: because the 20k subset fully contains the 10k subset, which fully contains the 5k subset,
any difference between the three runs is attributable purely to *how much* data was used — not to
*which* examples happened to land in each slice.

## Methodology

| | |
|---|---|
| Base model | [`unsloth/Llama-3.2-1B`](https://huggingface.co/unsloth/Llama-3.2-1B), loaded in bf16 (not 4-bit) |
| Fine-tuning method | LoRA (r=16, α=16, dropout=0) on all 7 projections (q, k, v, o, gate, up, down), merged to full 16-bit weights after training |
| Training framework | Unsloth + TRL `SFTTrainer` |
| Epochs | 1 |
| Batch size | per-device **1**, gradient accumulation **16** → effective batch size **16** |
| Learning rate | 2e-4, cosine schedule, warmup ratio 0.03 |
| Optimizer | `adamw_8bit`, weight decay 0.01 |
| Seed | 3407 |
| Loss masking | `train_on_responses_only` — loss computed only on assistant tokens, not the prompt |
| Max sequence length | 2048 |

## Training

| Subset | Steps | Training time | Final train loss | Final val loss |
|---|---|---------------|---|---|
| 5k | 313 | 22:45         | 0.4357 | 0.4262 |
| 10k | 625 | 43:58         | 0.4091 | 0.3962 |
| 20k | 1,250 | 1:27:56       | 0.3674 | 0.3683 |

Both train and validation loss decreased consistently as dataset size increased — a reasonable sign the
model is genuinely learning more from more data, rather than just training for longer on the same
signal. Since all three subsets are evaluated against the same shared eval set, these validation-loss
numbers are directly comparable across subsets, not just within one.

<details>
<summary><b>Full per-checkpoint loss logs</b></summary>

**5k**

| Step | Training Loss | Validation Loss |
|---|---|---|
| 100 | 0.454637 | 0.464289 |
| 200 | 0.434012 | 0.434042 |
| 300 | 0.401614 | 0.426247 |
| 313 | 0.435691 | 0.426212 |

**10k**

| Step | Training Loss | Validation Loss |
|---|---|---|
| 100 | 0.450246 | 0.464120 |
| 200 | 0.437593 | 0.430946 |
| 300 | 0.414189 | 0.417405 |
| 400 | 0.400237 | 0.404597 |
| 500 | 0.394241 | 0.398486 |
| 600 | 0.392017 | 0.396188 |
| 625 | 0.409071 | 0.396211 |

**20k**

| Step | Training Loss | Validation Loss |
|---|---|---|
| 100 | 0.471793 | 0.471913 |
| 200 | 0.413839 | 0.435277 |
| 300 | 0.405774 | 0.418804 |
| 400 | 0.435141 | 0.404988 |
| 500 | 0.408781 | 0.396609 |
| 600 | 0.374038 | 0.388064 |
| 700 | 0.379729 | 0.382723 |
| 800 | 0.387904 | 0.376873 |
| 900 | 0.389400 | 0.373317 |
| 1000 | 0.358660 | 0.370228 |
| 1100 | 0.368594 | 0.369003 |
| 1200 | 0.361067 | 0.368311 |
| 1250 | 0.367407 | 0.368333 |

</details>

## Challenges & debugging

Real problems hit during training, on the theory that the debugging is often more informative than a
clean success would be

**10k run: silent slowdown, then an OOM crash — and the actual fix.** The first 10k attempt (4-bit) ran
fine for the first ~137 steps at ~7 s/step, then progressively slowed to ~65 s/step by step 174 — the
ETA jumped from ~58 minutes to ~2.5 hours mid-run with no error thrown. Switched to bf16 (full precision,
no 4-bit quantization) to rule out a quantization-related slowdown, which fixed the speed (~2.9 s/step)
but then crashed with a CUDA out-of-memory error at step 31/625 — trying to allocate 992 MiB with only
593 MiB free on the 8 GB card. The eventual fix that made the successful 625-step run possible:
`per_device_train_batch_size` dropped to **1** (with gradient accumulation raised to 16 to keep the same
effective batch size), combined with setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before loading the model — this tells PyTorch's CUDA
allocator to grow memory segments instead of hunting for one large contiguous free block, which is what
was causing the allocator to fail even with a few hundred MB technically free but fragmented.

**Interrupted overnight eval.** The 5k model's overnight `lm-eval` benchmark run got cut off when the PC
shut down partway through (it had been running long enough to trigger an idle/update shutdown). After
restarting the machine, the process appeared to pick back up and completed normally. Since `lm-eval`
only writes its `results.json` after *every* task fully completes — there's no partial/incomplete
results file — its existence was itself confirmation the run had gone the distance; cross-checked that
the 5k results file listed the same task set as the 10k/20k runs to be sure nothing silently got skipped.
Adjusted Windows power settings afterward to prevent a repeat.

## Evaluation

All benchmarks run with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
against all **four** models — the three fine-tunes plus the untouched base — so every result below is a
direct before/after comparison, not just an absolute score in isolation.

| Task | Shots | Primary metric | What it measures |
|---|---|---|---|
| GSM8K | 5-shot | `exact_match` (flexible-extract) | Grade-school math word problems — the fine-tuning target |
| MMLU | 5-shot | `acc` | Broad general knowledge across 57 subjects |
| ARC-Challenge | 25-shot | `acc_norm` | Grade-school science reasoning |
| HellaSwag | 10-shot | `acc_norm` | Commonsense sentence completion |
| WinoGrande | 5-shot | `acc` | Commonsense pronoun resolution |

Shot counts match each task's standard, publicly-reported default (the same ones used by the Open LLM
Leaderboard), so these numbers are directly comparable to other published results, not just to each
other. `acc_norm` is used instead of raw `acc` for ARC-Challenge and HellaSwag because it normalizes
each answer choice's likelihood by its token length, correcting for models being biased toward longer
answer options regardless of correctness.

## Results

### GSM8K / ARC-Challenge / HellaSwag / WinoGrande

| Benchmark | Llama-3.2-1B (base) | MathCodeInstruct-5k | Change |
|---|---|---|---|
| GSM8K | 5.8% | 7.4% | 🟢 +1.5% |
| ARC-Challenge | 36.9% | 36.8% | ⚪ -0.1% |
| HellaSwag | 64.2% | 63.8% | 🔴 -0.3% |
| WinoGrande | 60.8% | 62.4% | 🟢 +1.7% |

| Benchmark | Llama-3.2-1B (base) | MathCodeInstruct-10k | Change |
|---|---|---|---|
| GSM8K | 5.8% | 8.7% | 🟢 +2.9% |
| ARC-Challenge | 36.9% | 36.1% | 🔴 -0.8% |
| HellaSwag | 64.2% | 63.8% | 🔴 -0.4% |
| WinoGrande | 60.8% | 62.0% | 🟢 +1.3% |

| Benchmark | Llama-3.2-1B (base) | MathCodeInstruct-20k | Change |
|---|---|---|---|
| GSM8K | 5.8% | 8.9% | 🟢 +3.1% |
| ARC-Challenge | 36.9% | 35.8% | 🔴 -1.1% |
| HellaSwag | 64.2% | 63.6% | 🔴 -0.6% |
| WinoGrande | 60.8% | 61.4% | 🟢 +0.6% |

### MMLU by subject

Unlike a simple 4-category rollup, `visualize_results.py` charts **all 57 individual MMLU subjects**
(plus an overall score) side by side, base vs. fine-tune — a much more granular view of exactly where a
fine-tune helps or hurts general knowledge, rather than just an average that could hide subject-level
swings in either direction.

<details>
<summary><b>MMLU-5k</b></summary>

![MMLU comparison](Llama-3.2-1B-MathCOdeInstruct-5k/outputs/llama-3.2-1b-5k/assets/mmlu_5k.png)
</details>

<details>
<summary><b>MMLU-10k</b></summary>

![MMLU comparison](Llama-3.2-1B-MathCOdeInstruct-10k/outputs/llama-3.2-1b-10k/assets/mmlu_10k.png)
</details>

<details>
<summary><b>MMLU-5k</b></summary>

![MMLU comparison](Llama-3.2-1B-MathCOdeInstruct-20k/outputs/llama-3.2-1b-20k/assets/mmlu_20k.png)
</details>


### Inference speed


| Model                | Tokens/s |
|----------------------|----------|
| Base                 | 40.59    |
| MathCodeInstruct-5k  | 40.67    |
| MathCodeInstruct-10k | 40.53    |
| MathCodeInstruct-20k | 40.63    |

## Key findings

 - *GSM8K accuracy had a moderate increase to data size*
 - *Arc-Challenge and HellaSwag accuracy had a negligible decrease to data size*
 - *WinoGrande accuracy displayed a momentary noticeable increase at the 5k data size, but later normalized back down*
 - *MMLU accuracy displayed a mixture of changes. Unsurprisingly, math subjects — like Abstract Algebra and Elementary Mathematics — had an increased accuracy as data size grew. But most notably, many subjects decreased in accuracy, such as College Chemistry and Computer Science*
 - *The speed (Tokens/second) remained unchanged as datasize grew*

## Models

| Model                             | Hugging Face                                                                    | GSM8K  | MMLU  |
|-----------------------------------|---------------------------------------------------------------------------------|--------|-------|
| Llama-3.2-1B (base)               | [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B)             | *5.8%* | *38%* |
| Llama-3.2-1B-MathCodeInstruct-5k  | [link](https://huggingface.co/OliverSundaram/Llama-3.2-1B-MathCodeInstruct-5k)  | *7.4%* | *40%* |
| Llama-3.2-1B-MathCodeInstruct-10k | [link](https://huggingface.co/OliverSundaram/Llama-3.2-1B-MathCodeInstruct-10k) | *8.7%* | *39%* |
| Llama-3.2-1B-MathCodeInstruct-20k | [link](https://huggingface.co/OliverSundaram/Llama-3.2-1B-MathCodeInstruct-20k) | *8.9%* | *39%* |

## Reproducing this project

```bash
git clone https://github.com/OliverSundaram/Llama-3.2-1B-MathCodeInstruct.git
cd Llama-3.2-1B-MathCodeInstruct
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**1. Prepare the data** — run `data/prepare_data.ipynb` top to bottom. Produces the filtered, formatted
`mathcodeinstruct-train-{5k,10k,20k}.json` files and the single shared `mathcodeinstruct-eval.json`.

**2. Train each subset** — run `01_training.ipynb` inside each of the three subset folders (change the
`SUBSET` variable at the top of the notebook to `"5k"`, `"10k"`, or `"20k"`). Each merges its own LoRA
adapter to `outputs/llama-3.2-1b-{size}/` when done. Note the notebook sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before loading the model — worth keeping if you're
also on an 8GB card.

**3. Run the benchmark sweep:**
```powershell
$models = @{
  "base" = "unsloth/Llama-3.2-1B"
  "5k"   = "Llama-3.2-1B-MathCodeInstruct-5k\outputs\llama-3.2-1b-5k"
  "10k"  = "Llama-3.2-1B-MathCodeInstruct-10k\outputs\llama-3.2-1b-10k"
  "20k"  = "Llama-3.2-1B-MathCodeInstruct-20k\outputs\llama-3.2-1b-20k"
}
foreach ($name in $models.Keys) {
  lm_eval --model hf `
    --model_args pretrained="$($models[$name])",dtype=bfloat16 `
    --tasks gsm8k,arc_challenge,hellaswag,winogrande,mmlu `
    --batch_size auto `
    --output_path "outputs\lm_eval\$name"
}
```

**4. Generate charts and tables:**
```powershell
python visualize_results.py --tuned 5k
python visualize_results.py --tuned 10k
python visualize_results.py --tuned 20k
```

**5. Measure speed** (all four models in one run — `--model` is repeatable and required per model):
```powershell
python measure_speed.py `
  --model base=unsloth/Llama-3.2-1B `
  --model 5k=Llama-3.2-1B-MathCodeInstruct-5k/outputs/llama-3.2-1b-5k `
  --model 10k=Llama-3.2-1B-MathCodeInstruct-10k/outputs/llama-3.2-1b-10k `
  --model 20k=Llama-3.2-1B-MathCodeInstruct-20k/outputs/llama-3.2-1b-20k
```

**6. Push to Hugging Face** — `pushing_to_hf.py` uploads each merged model folder (weights + your filled-in
`README.md` + `assets/`) to its Hub repo. Note this assumes the three repos already exist; the first time,
create them with `huggingface_hub.create_repo("OliverSundaram/Llama-3.2-1B-MathCodeInstruct-{size}")`
before running the push script.

## Environment

Trained and evaluated entirely on a single consumer GPU — no cloud rental.

| | |
|---|---|
| GPU | NVIDIA RTX 4060, 8 GB VRAM |
| OS | Windows |
| Unsloth | 2026.8.15 |
| Transformers | 5.5.0 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.7.1 |
| Xformers | 0.0.35 (no FlashAttention-2) |

## Limitations

- Each model trained for a single epoch on its subset — not intended as a general-purpose assistant.
- The 5k/10k/20k subsets are nested rather than independent samples (see [Dataset](#dataset)) — a
  deliberate choice for isolating the effect of data volume, but it does mean the three runs aren't
  fully independent experiments.
- All benchmark numbers reflect a 1B-parameter model and should be read relative to the base model's own
  scores, not against much larger models.
- No RLHF or additional safety alignment applied beyond what base Llama-3.2-1B already has.

## Acknowledgments & citations

This project builds on the following open-source work:

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** (EleutherAI) — used
  for all benchmark evaluation.
- **[Unsloth](https://github.com/unslothai/unsloth)** — used for efficient LoRA fine-tuning and model
  merging.
- **[MathCodeInstruct](https://huggingface.co/datasets/MathLLMs/MathCodeInstruct)** dataset, from the
  MathCoder paper (Wang et al., ICLR 2024, [arXiv:2310.03731](https://arxiv.org/abs/2310.03731)).
- **[Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B)** (Meta) — base model, via the
  [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) mirror used for training.
- **[Hugging Face `transformers`](https://github.com/huggingface/transformers)** and
  **[TRL](https://github.com/huggingface/trl)** — the `SFTTrainer` and model-loading infrastructure
  Unsloth builds on.

```bibtex
@misc{wang2023mathcoder,
  title={MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning},
  author={Ke Wang and Houxing Ren and Aojun Zhou and Zimu Lu and Sichun Luo and Weikang Shi and Renrui Zhang and Linqi Song and Mingjie Zhan and Hongsheng Li},
  year={2023},
  eprint={2310.03731},
  archivePrefix={arXiv}
}
```

## License

Code in this repository is licensed under [MIT](LICENSE). The fine-tuned model weights are derivatives
of Llama 3.2 and are distributed under [Meta's Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) — see each model's card on Hugging Face.
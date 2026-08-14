import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path("outputs/lm_eval")
OUT_DIR = Path("outputs/charts")

TASK_METRIC = {
    "gsm8k": "exact_match,flexible-extract",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
}
TASK_DISPLAY_NAMES = {
    "gsm8k": "GSM8K",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
}

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

MMLU_GROUPS = ["mmlu_" + s for s in MMLU_SUBJECTS] + ["mmlu"]
MMLU_LABELS = [s.replace("_", " ").title() for s in MMLU_SUBJECTS] + ["Overall"]

COLOR_BASE = "#94A3B8"
COLOR_TUNE = "#2563EB"


def find_results_json(folder: Path) -> Path:
    matches = sorted(folder.glob("**/results_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No results_*.json found under {folder} — did the eval run finish?"
        )
    return matches[-1]


def load_results(name: str) -> dict:
    path = find_results_json(BASE_DIR / name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def get_value(results: dict, task: str, metric: str) -> float:
    try:
        return results[task][metric]
    except KeyError as e:
        raise KeyError(
            f"Couldn't find {metric!r} under {task!r}. "
            f"Available keys for this task: {list(results.get(task, {}).keys())}"
        ) from e


def build_mmlu_chart(tuned_name: str, tuned_label: str, out_path: Path) -> None:
    base = load_results("base")
    tuned = load_results(tuned_name)

    base_vals = [get_value(base, g, "acc,none") for g in MMLU_GROUPS]
    tuned_vals = [get_value(tuned, g, "acc,none") for g in MMLU_GROUPS]

    y = list(range(len(MMLU_LABELS)))
    height = 0.35

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(MMLU_LABELS) + 1.5))

    ax.barh([i + height / 2 for i in y], base_vals, height=height,
            color=COLOR_BASE, label="Llama-3.2-1B (base)")
    ax.barh([i - height / 2 for i in y], tuned_vals, height=height,
            color=COLOR_TUNE, label=tuned_label)

    ax.set_yticks(y)
    ax.set_yticklabels(MMLU_LABELS, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy (5-shot)")
    ax.set_title("MMLU Accuracy by Subject", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False)

    # Give the "Overall" row a visual break from the 57 subject rows above it.
    ax.axhline(len(MMLU_LABELS) - 1.5, color="#CBD5E1", linewidth=0.8)

    # Strip the chart-junk: no box around the plot, faint gridlines only.
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(left=False)

    for i, (b, t) in enumerate(zip(base_vals, tuned_vals)):
        ax.text(b + 0.015, i + height / 2, f"{b:.0%}", va="center", fontsize=6.5, color="#475569")
        ax.text(t + 0.015, i - height / 2, f"{t:.0%}", va="center", fontsize=6.5,
                color=COLOR_TUNE, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart -> {out_path}")


def build_comparison_table(tuned_name: str, tuned_label: str) -> str:
    base = load_results("base")
    tuned = load_results(tuned_name)

    lines = [
        f"| Benchmark | Llama-3.2-1B (base) | {tuned_label} | Change |",
        "|---|---|---|---|",
    ]
    for task, metric in TASK_METRIC.items():
        b = get_value(base, task, metric)
        t = get_value(tuned, task, metric)
        delta = t - b
        arrow = "🟢" if delta > 0.001 else ("🔴" if delta < -0.001 else "⚪")
        lines.append(
            f"| {TASK_DISPLAY_NAMES[task]} | {b:.1%} | {t:.1%} | {arrow} {delta:+.1%} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", required=True, choices=["5k", "10k", "20k"],
                         help="Which fine-tuned subset to compare against base")
    args = parser.parse_args()

    tuned_label = f"MathCodeInstruct-{args.tuned}"
    chart_path = OUT_DIR / f"mmlu_{args.tuned}.png"

    build_mmlu_chart(args.tuned, tuned_label, chart_path)

    table_md = build_comparison_table(args.tuned, tuned_label)
    table_path = OUT_DIR / f"table_{args.tuned}.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"Saved table -> {table_path}\n")
    print(table_md)
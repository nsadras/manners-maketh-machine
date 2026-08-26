import argparse
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

TONE_ORDER = ["sycophantic", "very_polite", "polite", "neutral",
              "rude", "very_rude", "threatening"]
 
TONE_LABELS = {
    "sycophantic": "Sycophantic",
    "very_polite": "Very polite",
    "polite": "Polite",
    "neutral": "Neutral",
    "rude": "Rude",
    "very_rude": "Very rude",
    "threatening": "Threatening",
}

COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=Path, help='experiment results csv file path')
    parser.add_argument('--out', type=Path, default=None, help='output image file path (default: <results_file stem>_plot.png)')
    args = parser.parse_args()

    if not args.results_file.exists():
        raise FileNotFoundError(f"No such file: {args.results_file}")

    out_path = args.out or args.results_file.with_name(f"{args.results_file.stem}_plot.png")

    df = pd.read_csv(args.results_file)

    df["tone"] = pd.Categorical(df["tone"], categories=TONE_ORDER, ordered=True)

    # average over variants, compute sem across questions
    per_question = (
        df.groupby(["model", "tone", "question_id"], observed=True)["is_correct"]
        .mean()
        .reset_index()
    )
    acc = (
        per_question.groupby(["model", "tone"], observed=True)["is_correct"]
        .agg(mean="mean", sem=lambda x: x.std() / len(x) ** 0.5)
        .reset_index()
    )
     
    models = df["model"].unique()
 
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
 
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = list(range(len(TONE_ORDER)))
    xticklabels = [TONE_LABELS.get(t, t) for t in TONE_ORDER]
 
    for i, model in enumerate(models):
        color = COLORS[i % len(COLORS)]
        sub = acc[acc["model"] == model].set_index("tone").reindex(TONE_ORDER)
        ax.errorbar(x, sub["mean"] * 100, yerr=sub["sem"] * 100 if sub["sem"].notna().any() else None,
                    marker="o", linewidth=2.5, markersize=7, label=model, color=color, capsize=4)
 
    ax.set_title("Accuracy by prompt tone\n(averaged across questions per tone)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
 
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False)
    fig.suptitle("Does tone change what an LLM tells you?", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()

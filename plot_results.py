import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

SUBJECT_GROUPS = {
    "STEM": {
        "high_school_physics",
        "college_computer_science",
        "elementary_mathematics",
    },
    "Non-STEM": {
        "philosophy",
        "world_religions",
        "high_school_us_history",
    },
}

SUBJECT_GROUP_COLORS = {
    "STEM": "#2563eb",
    "Non-STEM": "#dc2626",
}


def summarize_accuracy(df, group_columns):
    """Average variants first, then compute uncertainty across questions."""
    per_question = (
        df.groupby([*group_columns, "question_id"], observed=True)["is_correct"]
        .mean()
        .reset_index()
    )
    return (
        per_question.groupby(group_columns, observed=True)["is_correct"]
        .agg(mean="mean", sem=lambda x: x.std() / len(x) ** 0.5)
        .reset_index()
    )


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def plot_model_subject_groups(df, model, out_path, x, xticklabels):
    model_df = df[df["model"] == model].copy()
    model_df["subject_group"] = model_df["subject"].map(
        {
            subject: group
            for group, subjects in SUBJECT_GROUPS.items()
            for subject in subjects
        }
    )

    unknown_subjects = sorted(model_df.loc[model_df["subject_group"].isna(), "subject"].unique())
    if unknown_subjects:
        raise ValueError(
            "Subjects missing from SUBJECT_GROUPS: " + ", ".join(unknown_subjects)
        )

    acc = summarize_accuracy(model_df, ["subject_group", "tone"])
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for subject_group in SUBJECT_GROUPS:
        sub = acc[acc["subject_group"] == subject_group].set_index("tone").reindex(TONE_ORDER)
        ax.errorbar(
            x,
            sub["mean"] * 100,
            yerr=sub["sem"] * 100 if sub["sem"].notna().any() else None,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=subject_group,
            color=SUBJECT_GROUP_COLORS[subject_group],
            capsize=4,
        )

    ax.set_title(f"Accuracy by prompt tone\n{model}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.05), frameon=True)
    fig.suptitle("STEM vs. non-STEM performance", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


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
    acc = summarize_accuracy(df, ["model", "tone"])
     
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
 
    ax.set_title("Accuracy by prompt tone",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
 
    ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.05), ncol=1, frameon=True)
    fig.suptitle("Does tone change what an LLM tells you?", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    for model in models:
        model_out_path = out_path.with_name(
            f"{out_path.stem}_{safe_filename(model)}_stem_vs_non_stem{out_path.suffix}"
        )
        plot_model_subject_groups(df, model, model_out_path, x, xticklabels)

if __name__ == '__main__':
    main()

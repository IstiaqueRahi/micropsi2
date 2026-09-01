"""Create aggregate comparison charts for the nine-run Survivor ensemble."""

import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ABUNDANCE_ORDER = ("sparse", "medium", "plentiful")
DISTANCE_ORDER = ("close", "medium", "far")
METRICS = ("emo_pleasure", "emo_valence")
COLORS = {"emo_pleasure": "#2a9d65", "emo_valence": "#e39a24"}
LABELS = {"emo_pleasure": "Pleasure", "emo_valence": "Valence"}


def metric(config, name):
    return float(config["emo_modulators"][name]["mean"])


def descriptive_aggregate(configs, axis_name, axis_order, metric_name):
    """Return cell mean, population SD, and raw values across the other factor."""
    means, spreads, raw = [], [], []
    for level in axis_order:
        values = np.array([metric(config, metric_name) for config in configs if config[axis_name] == level])
        means.append(float(values.mean()))
        spreads.append(float(values.std(ddof=0)))
        raw.append(values)
    return np.array(means), np.array(spreads), raw


def draw_trend_panel(ax, configs, axis_name, axis_order, title):
    x = np.arange(len(axis_order))
    offsets = {"emo_pleasure": -0.055, "emo_valence": 0.055}
    for metric_name in METRICS:
        means, spreads, raw = descriptive_aggregate(configs, axis_name, axis_order, metric_name)
        ax.errorbar(
            x + offsets[metric_name],
            means,
            yerr=spreads,
            color=COLORS[metric_name],
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=5,
            label=f"{LABELS[metric_name]} mean ± descriptive SD",
            zorder=3,
        )
        for level_index, values in enumerate(raw):
            jitter = np.linspace(-0.025, 0.025, len(values))
            ax.scatter(
                np.full(len(values), x[level_index] + offsets[metric_name]) + jitter,
                values,
                s=20,
                color=COLORS[metric_name],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.55,
                zorder=4,
            )
            ax.annotate(
                f"{means[level_index]:.3f}",
                (x[level_index] + offsets[metric_name], means[level_index]),
                xytext=(0, 8 if metric_name == "emo_pleasure" else -14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=COLORS[metric_name],
            )
    ax.set_xticks(x, [level.title() for level in axis_order])
    ax.set_ylabel("Mean logged modulator value")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=7.5, loc="best")


def main():
    summary_path = os.path.join(OUTPUT_DIR, "ensemble_summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        document = json.load(handle)
    configs = document["configs"]
    by_cell = {(c["abundance"], c["starting_distance"]): c for c in configs}
    if len(by_cell) != 9:
        raise ValueError(f"Expected nine unique factorial cells, found {len(by_cell)}")

    all_cell_values = [metric(config, name) for config in configs for name in METRICS]
    lower = min(-0.05, min(all_cell_values) - 0.08)
    upper = max(0.05, max(all_cell_values) + 0.08)

    fig = plt.figure(figsize=(15, 13), dpi=150, constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=(2.8, 1.2), hspace=0.27)
    cells = outer[0].subgridspec(3, 3, wspace=0.18, hspace=0.30)

    for row, abundance in enumerate(ABUNDANCE_ORDER):
        for column, distance in enumerate(DISTANCE_ORDER):
            config = by_cell[(abundance, distance)]
            ax = fig.add_subplot(cells[row, column])
            values = [metric(config, name) for name in METRICS]
            bars = ax.bar(
                [0, 1], values, width=0.62, color=[COLORS[name] for name in METRICS], alpha=0.9
            )
            ax.axhline(0, color="#555555", linewidth=0.7)
            ax.set_xticks([0, 1], [LABELS[name] for name in METRICS], fontsize=8)
            ax.set_ylim(lower, upper)
            ax.grid(axis="y", alpha=0.18)
            for bar, value in zip(bars, values):
                vertical_offset = 3 if value >= 0 else -11
                ax.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, vertical_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7.5,
                )
            state = "survived" if config["survived"] else f"died @ {config['died_at_step']}"
            ax.text(
                0.5,
                0.03,
                f"{state}  ·  visited {config['total_distinct_resources_visited']}\n"
                f"nearest initially {config['nearest_safe_resource_distance']:.0f} units",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
            if row == 0:
                ax.set_title(f"{distance.title()} start", fontsize=10.5, weight="bold")
            if column == 0:
                ax.set_ylabel(f"{abundance.title()}\nmean modulator value", fontsize=9)
            elif column > 0:
                ax.tick_params(labelleft=False)
            ax.text(
                0.98,
                0.96,
                f"C{config['config_id']} · seed {config['seed']}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#555555",
            )

    trends = outer[1].subgridspec(1, 2, wspace=0.24)
    abundance_ax = fig.add_subplot(trends[0, 0])
    draw_trend_panel(
        abundance_ax,
        configs,
        "abundance",
        ABUNDANCE_ORDER,
        "Resource-abundance effect (averaged across starts)",
    )
    distance_ax = fig.add_subplot(trends[0, 1])
    draw_trend_panel(
        distance_ax,
        configs,
        "starting_distance",
        DISTANCE_ORDER,
        "Starting-distance effect (averaged across abundances)",
    )

    fig.suptitle(
        "Survivor ensemble: measured pleasure and valence in a 3 × 3 factorial design",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Each factorial cell is one deterministic 800-step run with its own seed. "
        "Error bars are descriptive SD across the three cells of the other factor—not replicate-run "
        "uncertainty or inferential confidence intervals. Faint points are the three underlying cell values.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )

    output_path = os.path.join(OUTPUT_DIR, "ensemble_overview.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for axis_name, axis_order, label in (
        ("abundance", ABUNDANCE_ORDER, "abundance"),
        ("starting_distance", DISTANCE_ORDER, "starting distance"),
    ):
        print(label.title())
        for metric_name in METRICS:
            means, spreads, _ = descriptive_aggregate(configs, axis_name, axis_order, metric_name)
            values = ", ".join(
                f"{level}={mean:.4f} (descriptive SD {spread:.4f})"
                for level, mean, spread in zip(axis_order, means, spreads)
            )
            print(f"  {LABELS[metric_name]}: {values}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

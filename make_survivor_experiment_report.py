"""Build a visual multipage PDF report for the replicated Survivor experiment."""

import json
import math
import os
import textwrap


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np


ABUNDANCE = ("sparse", "medium", "plentiful")
DISTANCE = ("close", "medium", "far")
COLORS = {
    "navy": "#183153",
    "blue": "#2878b5",
    "green": "#2a9d65",
    "orange": "#e49b2f",
    "red": "#d64b3c",
    "purple": "#8b52a1",
    "teal": "#1b8a82",
    "ink": "#20252b",
    "muted": "#69737d",
    "pale": "#eef3f7",
}
RESOURCE_MARKERS = {
    "Waterhole": ("#2582d8", "o"),
    "Champignon": ("#2a9d65", "^"),
    "Wirselkraut": ("#1b8a82", "s"),
    "Juniper": ("#8b52a1", "D"),
    "FlyAgaric": ("#c83d35", "X"),
}


def add_footer(fig, page, text="Replicated Survivor factorial experiment"):
    fig.text(0.025, 0.018, text, fontsize=7.5, color=COLORS["muted"])
    fig.text(0.975, 0.018, f"Page {page}", ha="right", fontsize=7.5, color=COLORS["muted"])


def save_page(pdf, fig, page):
    add_footer(fig, page)
    # Keep every PDF page on the same fixed landscape canvas. Tight bounding
    # boxes crop each page around its artists and therefore produce unequal
    # page dimensions when charts, legends, or text have different extents.
    fig.set_size_inches(11.69, 8.27, forward=True)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def card(ax, title, value, subtitle, color):
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.08), 0.96, 0.84,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            facecolor=color, edgecolor="none", alpha=0.96,
        )
    )
    ax.text(0.08, 0.73, title.upper(), color="white", fontsize=8, weight="bold")
    ax.text(0.08, 0.40, value, color="white", fontsize=24, weight="bold", va="center")
    ax.text(0.08, 0.16, subtitle, color="white", fontsize=7.5, va="bottom")


def cell_lookup(document):
    return {
        (cell["abundance"], cell["starting_distance"]): cell
        for cell in document["analysis"]["cell_statistics"]
    }


def cell_matrix(cells, metric):
    return np.array(
        [[cells[(abundance, distance)][f"{metric}_mean"] for distance in DISTANCE] for abundance in ABUNDANCE],
        dtype=float,
    )


def heatmap(ax, matrix, title, fmt=".3f", cmap="viridis", vmin=None, vmax=None):
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3), [item.title() for item in DISTANCE], fontsize=8)
    ax.set_yticks(range(3), [item.title() for item in ABUNDANCE], fontsize=8)
    ax.set_xlabel("Starting distance", fontsize=8)
    ax.set_ylabel("Resource abundance", fontsize=8)
    ax.set_title(title, fontsize=10, weight="bold", color=COLORS["ink"])
    span = (np.nanmax(matrix) - np.nanmin(matrix)) or 1
    for row in range(3):
        for column in range(3):
            value = matrix[row, column]
            normalized = (value - np.nanmin(matrix)) / span
            text_color = "white" if normalized < 0.25 or normalized > 0.72 else COLORS["ink"]
            ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=9, weight="bold", color=text_color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def factor_panel(ax, main_effects, metric, title, color, levels):
    means = [main_effects[level][metric]["mean"] for level in levels]
    errors = [
        main_effects[level][metric]["ci95_high"] - main_effects[level][metric]["mean"]
        for level in levels
    ]
    x = np.arange(len(levels))
    ax.errorbar(x, means, yerr=errors, marker="o", capsize=5, lw=2.2, color=color)
    for index, value in enumerate(means):
        ax.annotate(f"{value:.4f}", (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(x, [level.title() for level in levels])
    ax.set_title(title, fontsize=9.5, weight="bold")
    ax.grid(axis="y", alpha=0.2)


def wrapped(fig, x, y, text, width=96, fontsize=10, color=None, weight=None, linespacing=1.35):
    fig.text(
        x, y, textwrap.fill(text, width), fontsize=fontsize,
        color=color or COLORS["ink"], weight=weight, va="top", linespacing=linespacing,
    )


def make_report():
    result_path = os.path.join(OUTPUT_DIR, "survivor_experiment_results.json")
    with open(result_path, encoding="utf-8") as handle:
        document = json.load(handle)
    runs = document["runs"]
    events = document["events"]
    traces = {
        (trace["abundance"], trace["starting_distance"]): trace
        for trace in document["representative_traces"]
    }
    cells = cell_lookup(document)
    design = document["design"]
    analysis = document["analysis"]
    pdf_path = os.path.join(OUTPUT_DIR, "survivor_experiment_report.pdf")
    background = mpimg.imread(
        os.path.join(
            SCRIPT_DIR, "micropsi_core", "world", "island", "resources", "groundmaps", "psi_1.png"
        )
    )

    with PdfPages(pdf_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Replicated Survivor Factorial Experiment"
        metadata["Author"] = "micropsi2 Variation D"
        metadata["Subject"] = "180-run Monte Carlo factorial experiment with event-aligned analysis"

        # Page 1 — executive dashboard
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(0.05, 0.94, "SURVIVOR FACTORIAL EXPERIMENT", fontsize=11, weight="bold", color=COLORS["green"])
        fig.text(0.05, 0.885, "What 180 replicated runs actually show", fontsize=27, weight="bold", color=COLORS["navy"])
        fig.text(0.05, 0.845, "Resource abundance × starting distance · corrected live valence input · event-level analysis", fontsize=11, color=COLORS["muted"])
        grid = fig.add_gridspec(1, 4, left=0.05, right=0.95, bottom=0.61, top=0.79, wspace=0.05)
        card(fig.add_subplot(grid[0, 0]), "Independent runs", "180", "20 matched seeds in each of 9 cells", COLORS["navy"])
        card(fig.add_subplot(grid[0, 1]), "Simulated steps", "144k", "800 body–decision cycles per run", COLORS["blue"])
        card(fig.add_subplot(grid[0, 2]), "Survival", "100%", "All 180 agents completed 800 steps", COLORS["green"])
        card(fig.add_subplot(grid[0, 3]), "Action events", "21,060", "Food, water, healing, and poison", COLORS["orange"])
        fig.text(0.06, 0.555, "THE CENTRAL RESULT", fontsize=10, weight="bold", color=COLORS["red"])
        wrapped(
            fig, 0.06, 0.515,
            "The 20 seeds generated zero run-level variance inside every factorial cell. Random escape behavior was not engaged strongly enough to alter measured outcomes. The replication therefore exposes determinism, not sampling uncertainty: confidence intervals collapse and classical F/p tests are undefined.",
            width=105, fontsize=12, weight="bold", color=COLORS["ink"],
        )
        observations = [
            "Correcting base_sum_of_urges makes mean valence responsive: cell means range from 0.329 to 0.400 instead of remaining pinned near −0.5.",
            "More resources do not yield a monotonic pleasure increase: medium abundance is lowest; plentiful-far is highest.",
            "Close/medium starts encounter the FlyAgaric in 120 runs; far starts avoid it in all 60 runs.",
            "Poisoning produces the dominant emotional event: mean pleasure −0.585, activation 1.488, valence −1.310.",
        ]
        for index, observation in enumerate(observations):
            y = 0.36 - index * 0.075
            fig.text(0.065, y, f"{index + 1}", fontsize=13, weight="bold", color="white", ha="center", va="center",
                     bbox={"boxstyle": "circle,pad=0.35", "facecolor": COLORS["green"], "edgecolor": "none"})
            wrapped(fig, 0.095, y + 0.018, observation, width=105, fontsize=9.5)
        save_page(pdf, fig, 1)

        # Page 2 — design and island layouts
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Experimental design and controlled environments", fontsize=19, weight="bold", color=COLORS["navy"], y=0.965)
        fig.text(0.5, 0.925, "Nested layouts: 6 / 10 / 14 safe objects; one fixed FlyAgaric in every cell", ha="center", fontsize=9.5, color=COLORS["muted"])
        grid = fig.add_gridspec(1, 3, left=0.04, right=0.98, bottom=0.24, top=0.88, wspace=0.08)
        for column, abundance in enumerate(ABUNDANCE):
            ax = fig.add_subplot(grid[0, column])
            trace = traces[(abundance, "close")]
            ax.imshow(background, extent=[0, 2048, 2048, 0])
            for resource in trace["resources"]:
                color, marker = RESOURCE_MARKERS[resource["type"]]
                ax.scatter(*zip(resource["position"]), s=55, color=color, marker=marker, edgecolor="black", linewidth=0.5, zorder=4)
            for distance, start_color in zip(DISTANCE, (COLORS["green"], COLORS["orange"], COLORS["red"])):
                start = traces[(abundance, distance)]["start_position"]
                ax.scatter([start[0]], [start[1]], s=95, facecolor="white", edgecolor=start_color, linewidth=2.2, zorder=5)
                ax.annotate(distance[0].upper(), start, ha="center", va="center", fontsize=7, weight="bold", color=start_color, zorder=6)
            ax.set_xlim(150, 1850); ax.set_ylim(1650, 200); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{abundance.title()} · {len(trace['resources']) - 1} safe + 1 trap", fontsize=10.5, weight="bold")
        legend = [Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="black", label=name, markersize=8)
                  for name, (color, marker) in RESOURCE_MARKERS.items()]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.17), ncol=5, frameon=False, fontsize=8)
        fig.text(0.05, 0.105, "DESIGN", fontsize=9, weight="bold", color=COLORS["green"])
        fig.text(0.05, 0.07, "3 abundance levels × 3 distance levels × 20 matched seed blocks × 800 steps = 144,000 simulated cycles", fontsize=10.5, weight="bold", color=COLORS["ink"])
        save_page(pdf, fig, 2)

        # Page 3 — emotional and bodily heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Factorial outcome dashboard — means across 20 runs per cell", fontsize=18, weight="bold", color=COLORS["navy"], y=0.975)
        heatmap(axes[0, 0], cell_matrix(cells, "mean_pleasure"), "Mean pleasure", ".4f", "RdYlGn")
        heatmap(axes[0, 1], cell_matrix(cells, "mean_valence"), "Mean corrected valence", ".3f", "RdYlGn")
        heatmap(axes[0, 2], cell_matrix(cells, "mean_activation"), "Mean activation", ".3f", "YlOrRd")
        heatmap(axes[1, 0], cell_matrix(cells, "mean_energy"), "Mean energy state", ".3f", "Blues", 0.9, 1.0)
        heatmap(axes[1, 1], cell_matrix(cells, "mean_water"), "Mean water state", ".3f", "Blues", 0.9, 1.0)
        heatmap(axes[1, 2], cell_matrix(cells, "mean_integrity"), "Mean integrity state", ".3f", "Purples", 0.95, 1.0)
        fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.08, wspace=0.35, hspace=0.38)
        save_page(pdf, fig, 3)

        # Page 4 — behavior and exposure
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Behavior, access, and hazard exposure", fontsize=18, weight="bold", color=COLORS["navy"], y=0.975)
        heatmap(axes[0, 0], cell_matrix(cells, "resources_visited"), "Distinct resource objects visited", ".1f", "GnBu")
        heatmap(axes[0, 1], cell_matrix(cells, "first_resource_step"), "Step of first resource arrival", ".0f", "YlOrBr")
        heatmap(axes[0, 2], cell_matrix(cells, "motive_switches"), "Motive switches per run", ".1f", "PuBu")
        heatmap(axes[1, 0], cell_matrix(cells, "action_successes"), "Successful resource actions", ".1f", "Greens")
        heatmap(axes[1, 1], cell_matrix(cells, "trap_encounter") * 100, "FlyAgaric encounter rate (%)", ".0f", "Reds", 0, 100)
        heatmap(axes[1, 2], cell_matrix(cells, "min_integrity"), "Minimum integrity", ".2f", "RdYlGn", 0, 1)
        fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.08, wspace=0.35, hspace=0.38)
        save_page(pdf, fig, 4)

        # Page 5 — main effects
        fig, axes = plt.subplots(2, 4, figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Main effects — descriptive means and 95% intervals", fontsize=18, weight="bold", color=COLORS["navy"], y=0.975)
        abundance_effects = analysis["abundance_main_effects"]
        distance_effects = analysis["distance_main_effects"]
        panels = (
            ("mean_pleasure", "Pleasure", COLORS["green"]),
            ("mean_valence", "Corrected valence", COLORS["orange"]),
            ("mean_activation", "Activation", COLORS["red"]),
            ("resources_visited", "Resources visited", COLORS["blue"]),
        )
        for column, (metric, title, color) in enumerate(panels):
            factor_panel(axes[0, column], abundance_effects, metric, f"{title} vs abundance", color, ABUNDANCE)
            factor_panel(axes[1, column], distance_effects, metric, f"{title} vs distance", color, DISTANCE)
        fig.text(0.5, 0.035, "Intervals have zero width because all 20 seed replicates within each cell produced identical outcomes.", ha="center", fontsize=9, color=COLORS["red"], weight="bold")
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.10, wspace=0.32, hspace=0.38)
        save_page(pdf, fig, 5)

        # Page 6 — trap contrast and recovery
        trap_runs = [run for run in runs if run["trap_encounter"]]
        safe_runs = [run for run in runs if not run["trap_encounter"]]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("The FlyAgaric dominates individual emotional experience", fontsize=18, weight="bold", color=COLORS["navy"], y=0.975)
        metrics = ("mean_pleasure", "mean_valence", "mean_activation", "mean_integrity")
        labels = ("Pleasure", "Valence", "Activation", "Integrity")
        x = np.arange(4)
        trap_values = [average([run[metric] for run in trap_runs]) for metric in metrics]
        safe_values = [average([run[metric] for run in safe_runs]) for metric in metrics]
        width = 0.35
        axes[0, 0].bar(x - width / 2, trap_values, width, label="Trap encountered (n=120)", color=COLORS["red"])
        axes[0, 0].bar(x + width / 2, safe_values, width, label="Trap avoided (n=60)", color=COLORS["green"])
        axes[0, 0].set_xticks(x, labels); axes[0, 0].set_title("Run-level means", weight="bold"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(axis="y", alpha=0.2)
        trap_event = analysis["event_statistics"]["trap:FlyAgaric"]
        resource_events = [value for key, value in analysis["event_statistics"].items() if key.startswith("successful")]
        axes[0, 1].bar(
            ["Pleasure", "Activation", "Valence"],
            [trap_event["mean_pleasure_at_event"], trap_event["mean_activation_at_event"], trap_event["mean_valence_at_event"]],
            color=[COLORS["red"], COLORS["orange"], COLORS["purple"]],
        )
        axes[0, 1].axhline(0, color="#555", lw=0.7); axes[0, 1].set_title("At the poisoning step (120 events)", weight="bold"); axes[0, 1].grid(axis="y", alpha=0.2)
        recovery = {}
        for abundance in ABUNDANCE:
            values = [run["trap_recovery_steps"] for run in trap_runs if run["abundance"] == abundance and run["trap_recovery_steps"] is not None]
            recovery[abundance] = average(values)
        axes[1, 0].bar([a.title() for a in ABUNDANCE], [recovery[a] for a in ABUNDANCE], color=["#8bc5a5", "#4fa77c", "#1e7c55"])
        axes[1, 0].set_ylabel("Steps from poison to integrity ≥ 0.8"); axes[1, 0].set_title("Mean recovery time after poisoning", weight="bold"); axes[1, 0].grid(axis="y", alpha=0.2)
        encounter = cell_matrix(cells, "trap_encounter") * 100
        heatmap(axes[1, 1], encounter, "Trap exposure is determined by start geometry", ".0f", "Reds", 0, 100)
        fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08, wspace=0.27, hspace=0.35)
        save_page(pdf, fig, 6)

        # Page 7 — event-level analysis
        event_stats = analysis["event_statistics"]
        keys = ["successful_resource_action:Champignon", "successful_resource_action:Waterhole", "successful_resource_action:Wirselkraut", "trap:FlyAgaric"]
        short = ["Food", "Water", "Healing", "Poison"]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Event-aligned emotional analysis — the 800-step mean is not the experience", fontsize=17, weight="bold", color=COLORS["navy"], y=0.975)
        for ax, metric, title, color in (
            (axes[0, 0], "mean_pleasure_at_event", "Pleasure at action", COLORS["green"]),
            (axes[0, 1], "mean_activation_at_event", "Activation at action", COLORS["orange"]),
            (axes[1, 0], "mean_valence_at_event", "Valence at action", COLORS["purple"]),
        ):
            values = [event_stats[key][metric] for key in keys]
            bars = ax.bar(short, values, color=[COLORS["green"], COLORS["blue"], COLORS["teal"], COLORS["red"]])
            ax.axhline(0, color="#555", lw=0.7); ax.set_title(title, weight="bold"); ax.grid(axis="y", alpha=0.2)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width()/2, value), xytext=(0, 4 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=8)
        pre = [event_stats[key]["mean_pre_pleasure_5"] for key in keys]
        at = [event_stats[key]["mean_pleasure_at_event"] for key in keys]
        post = [event_stats[key]["mean_post_pleasure_5"] for key in keys]
        for index, label in enumerate(short):
            axes[1, 1].plot([-1, 0, 1], [pre[index], at[index], post[index]], marker="o", lw=2, label=label,
                            color=[COLORS["green"], COLORS["blue"], COLORS["teal"], COLORS["red"]][index])
        axes[1, 1].set_xticks([-1, 0, 1], ["5-step pre mean", "Action", "5-step post mean"])
        axes[1, 1].set_title("Pleasure is a transient event signal", weight="bold"); axes[1, 1].axhline(0, color="#555", lw=0.7); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.2)
        fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08, wspace=0.27, hspace=0.35)
        save_page(pdf, fig, 7)

        # Page 8 — representative trajectories
        fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Representative trajectories — matched seed 1001", fontsize=18, weight="bold", color=COLORS["navy"], y=0.975)
        for row, abundance in enumerate(ABUNDANCE):
            for column, distance in enumerate(DISTANCE):
                ax = axes[row, column]
                trace = traces[(abundance, distance)]
                ax.imshow(background, extent=[0, 2048, 2048, 0])
                for resource in trace["resources"]:
                    color, marker = RESOURCE_MARKERS[resource["type"]]
                    ax.scatter(*zip(resource["position"]), s=25, color=color, marker=marker, edgecolor="black", linewidth=0.3, zorder=4)
                xs = [point["x"] for point in trace["steps"]]
                ys = [point["y"] for point in trace["steps"]]
                ax.plot(xs, ys, color="white", lw=2, alpha=0.8, zorder=5)
                ax.plot(xs, ys, color=COLORS["red"], lw=0.9, alpha=0.9, zorder=6)
                ax.scatter([xs[0]], [ys[0]], s=28, facecolor="white", edgecolor="black", zorder=7)
                ax.set_xlim(150, 1850); ax.set_ylim(1650, 200); ax.set_xticks([]); ax.set_yticks([])
                if row == 0: ax.set_title(distance.title(), fontsize=10, weight="bold")
                if column == 0: ax.set_ylabel(abundance.title(), fontsize=10, weight="bold")
                run = next(item for item in runs if item["config_id"] == trace["config_id"] and item["seed"] == trace["seed"])
                ax.text(0.02, 0.02, f"P {run['mean_pleasure']:.4f} · V {run['mean_valence']:.3f}\nvisits {run['resources_visited']} · trap {run['trap_encounter']}", transform=ax.transAxes, fontsize=6.3,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2})
        fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.06, wspace=0.08, hspace=0.13)
        save_page(pdf, fig, 8)

        # Page 9 — two individual histories
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), facecolor="white", sharex="col")
        fig.suptitle("Individual view — similar survival, radically different experience", fontsize=18, weight="bold", color=COLORS["navy"], y=0.975)
        for column, (cell, label) in enumerate(((('sparse', 'close'), "Sparse / close: poisoned"), (('plentiful', 'far'), "Plentiful / far: trap avoided"))):
            trace = traces[cell]; rows = trace["steps"]; steps = [row["step"] for row in rows]
            axes[0, column].plot(steps, [row["energy"] for row in rows], label="Energy", color=COLORS["red"], lw=1.1)
            axes[0, column].plot(steps, [row["water"] for row in rows], label="Water", color=COLORS["blue"], lw=1.1)
            axes[0, column].plot(steps, [row["integrity"] for row in rows], label="Integrity", color=COLORS["purple"], lw=1.1)
            axes[0, column].set_ylim(0, 1.05); axes[0, column].set_title(label, weight="bold"); axes[0, column].grid(alpha=0.2); axes[0, column].legend(fontsize=8)
            axes[1, column].plot(steps, [row["emo_pleasure"] for row in rows], label="Pleasure", color=COLORS["green"], lw=1)
            axes[1, column].plot(steps, [row["emo_valence"] for row in rows], label="Valence", color=COLORS["orange"], lw=1)
            axes[1, column].plot(steps, [row["emo_activation"] for row in rows], label="Activation", color=COLORS["red"], lw=1)
            axes[1, column].axhline(0, color="#555", lw=0.6); axes[1, column].grid(alpha=0.2); axes[1, column].legend(fontsize=8); axes[1, column].set_xlabel("Simulation step")
            trap_step = next((row["step"] for row in rows if row["unexpected"]), None)
            if trap_step is not None:
                for ax in axes[:, column]: ax.axvline(trap_step, color=COLORS["red"], ls="--", lw=1)
                axes[0, column].annotate("FlyAgaric", (trap_step, 0.12), xytext=(trap_step + 35, 0.32), arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, color=COLORS["red"], fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.08, wspace=0.20, hspace=0.26)
        save_page(pdf, fig, 9)

        # Page 10 — inference audit
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Inference audit — replication revealed a deterministic experiment", fontsize=18, weight="bold", color=COLORS["navy"], y=0.965)
        grid = fig.add_gridspec(1, 2, left=0.06, right=0.96, bottom=0.17, top=0.86, wspace=0.18)
        ax = fig.add_subplot(grid[0, 0])
        for index, cell in enumerate(analysis["cell_statistics"]):
            values = [run["mean_valence"] for run in runs if run["abundance"] == cell["abundance"] and run["starting_distance"] == cell["starting_distance"]]
            ax.scatter(np.full(len(values), index), values, s=28, alpha=0.35, color=COLORS["blue"])
        ax.set_xticks(range(9), [f"{a[0].upper()}{d[0].upper()}" for a in ABUNDANCE for d in DISTANCE], rotation=0)
        ax.set_ylabel("Mean corrected valence"); ax.set_title("20 seed points overlap exactly in every cell", weight="bold"); ax.grid(axis="y", alpha=0.2)
        right = fig.add_subplot(grid[0, 1]); right.axis("off")
        right.text(0, 0.98, "WHY F AND p ARE UNDEFINED", fontsize=12, weight="bold", color=COLORS["red"], va="top")
        explanation = (
            textwrap.fill(
                "A factorial F-test divides between-condition variation by residual "
                "(within-cell) variation. Here the residual variation is exactly zero "
                "for the recorded outcomes. Division by zero is not evidence of infinite "
                "certainty; it means the stochastic replication did not generate a "
                "sampling distribution.",
                55,
            )
            + "\n\nWhat remains valid:\n"
            + "\n".join(
                "• " + textwrap.fill(item, 50, subsequent_indent="  ")
                for item in (
                    "exact descriptive differences among these nine controlled trajectories",
                    "causal attribution to explicitly changed layouts within this deterministic model",
                    "event-level contrasts such as poison versus successful feeding",
                )
            )
            + "\n\nWhat is not valid:\n"
            + "\n".join(
                "• " + textwrap.fill(item, 50, subsequent_indent="  ")
                for item in (
                    "population p-values or confidence claims",
                    "claims that effects generalize to perturbed layouts or policies",
                    "treating 800 time steps as 800 independent observations",
                )
            )
        )
        right.text(0, 0.90, explanation, fontsize=10, va="top", linespacing=1.5, color=COLORS["ink"])
        fig.text(0.5, 0.085, "Next replication must randomize an input that actually changes trajectories: start jitter, resource-position jitter, metabolic rate, or action reliability.", ha="center", fontsize=10, color=COLORS["navy"], weight="bold")
        save_page(pdf, fig, 10)

        # Page 11 — complete cell table
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.suptitle("Complete factorial-cell observations", fontsize=18, weight="bold", color=COLORS["navy"], y=0.965)
        ax = fig.add_axes([0.03, 0.10, 0.94, 0.78]); ax.axis("off")
        columns = ["Cell", "P", "V", "Act.", "Energy", "Water", "Integrity", "Visits", "1st res.", "Switches", "Trap %", "Survive %"]
        rows_table = []
        for abundance in ABUNDANCE:
            for distance in DISTANCE:
                cell = cells[(abundance, distance)]
                rows_table.append([
                    f"{abundance.title()} / {distance.title()}",
                    f"{cell['mean_pleasure_mean']:.4f}", f"{cell['mean_valence_mean']:.3f}", f"{cell['mean_activation_mean']:.3f}",
                    f"{cell['mean_energy_mean']:.3f}", f"{cell['mean_water_mean']:.3f}", f"{cell['mean_integrity_mean']:.3f}",
                    f"{cell['resources_visited_mean']:.1f}", f"{cell['first_resource_step_mean']:.0f}", f"{cell['motive_switches_mean']:.1f}",
                    f"{cell['trap_encounter_mean'] * 100:.0f}", "100",
                ])
        table = ax.table(cellText=rows_table, colLabels=columns, cellLoc="center", loc="center", colWidths=[0.17] + [0.075] * 11)
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 2.05)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(COLORS["navy"]); cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor(COLORS["pale"])
            cell.set_edgecolor("white")
        fig.text(0.05, 0.07, "P pleasure · V corrected valence · Act. activation · 1st res. first resource-arrival step. Every value is the exact mean of 20 identical seed outcomes in that cell.", fontsize=8, color=COLORS["muted"])
        save_page(pdf, fig, 11)

        # Page 12 — conclusions
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(0.06, 0.92, "CONCLUSIONS", fontsize=11, weight="bold", color=COLORS["green"])
        fig.text(0.06, 0.855, "What this stronger experiment changed", fontsize=25, weight="bold", color=COLORS["navy"])
        conclusions = [
            ("Valence became meaningful", "Supplying the live total urge moved cell mean valence into 0.329–0.400 and produced −1.310 at poisoning. The original near-constant −0.5 was a measurement-input artifact."),
            ("The simple happiness claim still fails", "Pleasure is non-monotonic across abundance. Plentiful-far performs best, but medium abundance is worse than sparse under these placements."),
            ("Geometry dominates hazard exposure", "All close/medium runs hit the fixed poison trap; every far run avoided it. Resource count alone is an incomplete environmental descriptor."),
            ("Events matter more than long means", "Successful food/water actions average about +0.069 pleasure; healing +0.136; poisoning −0.585. An 800-step average near zero hides these experiences."),
            ("Seeds were not true perturbations", "Twenty seeds per cell did not change outcomes. The next experiment should randomize resource coordinates, starting jitter, metabolism, or action reliability while retaining seed blocks."),
        ]
        for index, (title, body) in enumerate(conclusions):
            y = 0.75 - index * 0.135
            fig.text(0.07, y, f"{index + 1}", fontsize=13, weight="bold", color="white", ha="center", va="center",
                     bbox={"boxstyle": "circle,pad=0.38", "facecolor": [COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["blue"], COLORS["purple"]][index], "edgecolor": "none"})
            fig.text(0.105, y + 0.025, title, fontsize=11, weight="bold", color=COLORS["ink"])
            wrapped(fig, 0.105, y - 0.005, body, width=100, fontsize=9)
        fig.text(0.06, 0.085, "DATA PRODUCTS", fontsize=9, weight="bold", color=COLORS["green"])
        fig.text(0.06, 0.052, "survivor_experiment_results.json · survivor_experiment_runs.csv · survivor_experiment_events.csv · 30-second H.264 representative simulation", fontsize=8.5, color=COLORS["muted"])
        save_page(pdf, fig, 12)

    print(f"Saved {pdf_path} (12 visual pages)")
    return pdf_path


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    make_report()

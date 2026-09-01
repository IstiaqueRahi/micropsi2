"""Run a replicated 3 x 3 Survivor Monte Carlo factorial experiment.

Design
------
* Factors: safe-resource abundance (sparse/medium/plentiful) and starting
  distance (close/medium/far).
* Replication: 20 matched random-seed blocks per cell, 180 runs total.
* Run length: 800 steps.
* Policy: unchanged baseline motive hysteresis, target locking, blacklist,
  stuck escape, and terrain-aware movement from run_survivor.py.
* Measurement correction: ``base_sum_of_urges`` is updated before every
  Doernerian modulator step so valence receives the live total urge input.

The independent observation is a complete run, not an individual time step.
Full traces are retained for seed 1001 in all nine cells for visualisation;
all other runs retain run summaries and action-aligned event observations.
"""

import csv
import json
import math
import os
from statistics import mean, stdev

import numpy as np
from scipy.stats import f as f_distribution

import run_survivor_ensemble as ensemble


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
SEEDS = tuple(range(1001, 1021))
REPRESENTATIVE_SEED = SEEDS[0]
N_STEPS = 800

ANOVA_METRICS = (
    "mean_pleasure",
    "mean_valence",
    "mean_activation",
    "mean_energy",
    "mean_water",
    "mean_integrity",
    "min_energy",
    "min_water",
    "min_integrity",
    "resources_visited",
    "motive_switches",
    "action_successes",
    "first_resource_step",
    "trap_encounter",
)


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def extract_action_events(result, config, seed):
    rows = result["steps"]
    events = []
    for index, row in enumerate(rows):
        if not row.get("action_success"):
            continue
        before = rows[max(0, index - 5):index]
        after = rows[index + 1:min(len(rows), index + 6)]
        previous = rows[index - 1] if index else row
        event_type = "trap" if row["unexpected"] else "successful_resource_action"
        events.append(
            {
                "config_id": config["id"],
                "abundance": config["abundance"],
                "starting_distance": config["distance"],
                "seed": seed,
                "step": row["step"],
                "event_type": event_type,
                "resource_type": row.get("action_resource_type"),
                "ruling_motive": row["ruling_motive"],
                "pleasure_at_event": row["emo_pleasure"],
                "activation_at_event": row["emo_activation"],
                "valence_at_event": row["emo_valence"],
                "pre_pleasure_5": average(item["emo_pleasure"] for item in before),
                "post_pleasure_5": average(item["emo_pleasure"] for item in after),
                "delta_energy": row["energy"] - previous["energy"],
                "delta_water": row["water"] - previous["water"],
                "delta_integrity": row["integrity"] - previous["integrity"],
                "urge_change": row["urge_change"],
            }
        )
    return events


def summarize_experimental_run(result, config, seed, seed_block):
    rows = result["steps"]
    trap_index = next((i for i, row in enumerate(rows) if row["unexpected"]), None)
    recovery_steps = None
    if trap_index is not None:
        recovery_index = next(
            (i for i in range(trap_index + 1, len(rows)) if rows[i]["integrity"] >= 0.8),
            None,
        )
        recovery_steps = recovery_index - trap_index if recovery_index is not None else None

    path_length = sum(
        math.hypot(rows[i]["x"] - rows[i - 1]["x"], rows[i]["y"] - rows[i - 1]["y"])
        for i in range(1, len(rows))
    )
    first_resource_step = next(
        (row["step"] for row in rows if row.get("visited_resource_uid") is not None),
        len(rows),
    )
    motive_switches = sum(
        rows[index]["ruling_motive"] != rows[index - 1]["ruling_motive"]
        for index in range(1, len(rows))
    )

    run = {
        "run_id": f"C{config['id']}_S{seed}",
        "config_id": config["id"],
        "abundance": config["abundance"],
        "starting_distance": config["distance"],
        "seed": seed,
        "seed_block": seed_block,
        "safe_resource_count": len(ensemble.RESOURCE_LAYOUTS[config["abundance"]]) - 1,
        "start_x": config["start"][0],
        "start_y": config["start"][1],
        "nearest_safe_resource_distance": result["nearest_safe_resource_distance"],
        "steps_completed": result["steps_completed"],
        "survived": int(result["survived"]),
        "died_at_step": result["died_at_step"],
        "resources_visited": len(result["visited_resource_uids"]),
        "motive_switches": motive_switches,
        "action_successes": sum(row.get("action_success", 0) for row in rows),
        "unexpected_events": sum(row["unexpected"] for row in rows),
        "trap_encounter": int(trap_index is not None),
        "trap_step": rows[trap_index]["step"] if trap_index is not None else None,
        "trap_recovery_steps": recovery_steps,
        "first_resource_step": first_resource_step,
        "path_length": path_length,
        "mean_energy": average(row["energy"] for row in rows),
        "mean_water": average(row["water"] for row in rows),
        "mean_integrity": average(row["integrity"] for row in rows),
        "min_energy": min(row["energy"] for row in rows),
        "min_water": min(row["water"] for row in rows),
        "min_integrity": min(row["integrity"] for row in rows),
        "final_energy": rows[-1]["energy"],
        "final_water": rows[-1]["water"],
        "final_integrity": rows[-1]["integrity"],
        "mean_pleasure": average(row["emo_pleasure"] for row in rows),
        "mean_valence": average(row["emo_valence"] for row in rows),
        "mean_activation": average(row["emo_activation"] for row in rows),
        "mean_competence": average(row["emo_competence"] for row in rows),
        "min_pleasure": min(row["emo_pleasure"] for row in rows),
        "max_pleasure": max(row["emo_pleasure"] for row in rows),
        "max_activation": max(row["emo_activation"] for row in rows),
        "positive_pleasure_steps": sum(row["emo_pleasure"] > 0.02 for row in rows),
        "negative_pleasure_steps": sum(row["emo_pleasure"] < -0.02 for row in rows),
        "energy_motive_steps": sum(row["ruling_motive"] == "energy" for row in rows),
        "water_motive_steps": sum(row["ruling_motive"] == "water" for row in rows),
        "integrity_motive_steps": sum(row["ruling_motive"] == "integrity" for row in rows),
    }
    return run


def grouped_statistics(runs, factor, levels, metrics):
    output = {}
    for level in levels:
        group = [run for run in runs if run[factor] == level]
        output[level] = {}
        for metric in metrics:
            values = [float(run[metric]) for run in group if run[metric] is not None]
            sample_sd = stdev(values) if len(values) > 1 else 0.0
            standard_error = sample_sd / math.sqrt(len(values)) if values else None
            output[level][metric] = {
                "n": len(values),
                "mean": mean(values) if values else None,
                "sd": sample_sd,
                "se": standard_error,
                "ci95_low": mean(values) - 1.96 * standard_error if values else None,
                "ci95_high": mean(values) + 1.96 * standard_error if values else None,
            }
    return output


def cell_statistics(runs, metrics):
    cells = []
    for abundance in ensemble.RESOURCE_LAYOUTS:
        for distance in ("close", "medium", "far"):
            group = [
                run for run in runs
                if run["abundance"] == abundance and run["starting_distance"] == distance
            ]
            record = {
                "abundance": abundance,
                "starting_distance": distance,
                "n": len(group),
            }
            for metric in metrics:
                values = [float(run[metric]) for run in group if run[metric] is not None]
                record[f"{metric}_mean"] = mean(values) if values else None
                record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
                record[f"{metric}_ci95"] = (
                    1.96 * record[f"{metric}_sd"] / math.sqrt(len(values)) if values else None
                )
            cells.append(record)
    return cells


def blocked_factorial_anova(runs, metric):
    """Balanced OLS ANOVA: abundance * distance + matched seed block."""
    abundance_levels = ("sparse", "medium", "plentiful")
    distance_levels = ("close", "medium", "far")
    y = np.array([float(run[metric]) for run in runs])

    intercept = np.ones((len(runs), 1))
    abundance = np.array(
        [[int(run["abundance"] == level) for level in abundance_levels[1:]] for run in runs],
        dtype=float,
    )
    distance = np.array(
        [[int(run["starting_distance"] == level) for level in distance_levels[1:]] for run in runs],
        dtype=float,
    )
    interaction = np.column_stack(
        [abundance[:, a] * distance[:, d] for a in range(2) for d in range(2)]
    )
    seed_blocks = np.array(
        [[int(run["seed_block"] == block) for block in range(2, len(SEEDS) + 1)] for run in runs],
        dtype=float,
    )
    parts = {
        "abundance": abundance,
        "starting_distance": distance,
        "interaction": interaction,
        "seed_block": seed_blocks,
    }
    full = np.column_stack([intercept] + list(parts.values()))
    coefficients, _, rank_full, _ = np.linalg.lstsq(full, y, rcond=None)
    residual = y - full @ coefficients
    rss_full = float(residual @ residual)
    df_residual = len(y) - rank_full

    results = {}
    for term in ("abundance", "starting_distance", "interaction", "seed_block"):
        reduced = np.column_stack([intercept] + [value for name, value in parts.items() if name != term])
        reduced_coef, _, rank_reduced, _ = np.linalg.lstsq(reduced, y, rcond=None)
        reduced_residual = y - reduced @ reduced_coef
        rss_reduced = float(reduced_residual @ reduced_residual)
        df_term = rank_full - rank_reduced
        ss_term = max(0.0, rss_reduced - rss_full)
        if rss_full <= 1e-24 or df_residual <= 0 or df_term <= 0:
            f_value = None
            p_value = None
        else:
            f_value = (ss_term / df_term) / (rss_full / df_residual)
            p_value = float(f_distribution.sf(f_value, df_term, df_residual))
        results[term] = {
            "df": int(df_term),
            "df_residual": int(df_residual),
            "sum_squares": ss_term,
            "f": f_value,
            "p": p_value,
            "partial_eta_squared": ss_term / (ss_term + rss_full) if ss_term + rss_full else 0.0,
        }
    return results


def event_statistics(events):
    output = {}
    keys = sorted({(event["event_type"], event["resource_type"]) for event in events})
    for event_type, resource_type in keys:
        group = [
            event for event in events
            if event["event_type"] == event_type and event["resource_type"] == resource_type
        ]
        label = f"{event_type}:{resource_type}"
        output[label] = {
            "n": len(group),
            "mean_pleasure_at_event": average(event["pleasure_at_event"] for event in group),
            "mean_activation_at_event": average(event["activation_at_event"] for event in group),
            "mean_valence_at_event": average(event["valence_at_event"] for event in group),
            "mean_pre_pleasure_5": average(
                event["pre_pleasure_5"] for event in group if event["pre_pleasure_5"] is not None
            ),
            "mean_post_pleasure_5": average(
                event["post_pleasure_5"] for event in group if event["post_pleasure_5"] is not None
            ),
        }
    return output


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensemble.prepare_runtime()
    runs = []
    events = []
    representative_traces = []
    total_runs = len(ensemble.CONFIGS) * len(SEEDS)
    run_number = 0

    for config in ensemble.CONFIGS:
        for seed_block, seed in enumerate(SEEDS, start=1):
            run_number += 1
            print(
                f"Run {run_number:03d}/{total_runs}: C{config['id']} "
                f"{config['abundance']}/{config['distance']} seed {seed}"
            )
            result = ensemble.run_one_simulation(
                ensemble.RESOURCE_LAYOUTS[config["abundance"]],
                config["start"],
                N_STEPS,
                seed,
                config["id"],
                update_base_sum_of_urges=True,
            )
            run = summarize_experimental_run(result, config, seed, seed_block)
            runs.append(run)
            events.extend(extract_action_events(result, config, seed))
            if seed == REPRESENTATIVE_SEED:
                representative_traces.append(
                    {
                        "config_id": config["id"],
                        "abundance": config["abundance"],
                        "starting_distance": config["distance"],
                        "seed": seed,
                        "start_position": list(config["start"]),
                        "resources": result["resources"],
                        "survived": result["survived"],
                        "steps": result["steps"],
                    }
                )

    abundance_order = ["sparse", "medium", "plentiful"]
    distance_order = ["close", "medium", "far"]
    analysis = {
        "cell_statistics": cell_statistics(runs, ANOVA_METRICS),
        "abundance_main_effects": grouped_statistics(
            runs, "abundance", abundance_order, ANOVA_METRICS
        ),
        "distance_main_effects": grouped_statistics(
            runs, "starting_distance", distance_order, ANOVA_METRICS
        ),
        "blocked_factorial_anova": {
            metric: blocked_factorial_anova(runs, metric) for metric in ANOVA_METRICS
        },
        "event_statistics": event_statistics(events),
    }
    document = {
        "experiment": "Replicated Survivor resource-abundance x starting-distance factorial experiment",
        "design": {
            "factorial_cells": 9,
            "replicates_per_cell": len(SEEDS),
            "total_runs": total_runs,
            "steps_per_run": N_STEPS,
            "total_simulated_steps": total_runs * N_STEPS,
            "seed_blocks": list(SEEDS),
            "representative_seed": REPRESENTATIVE_SEED,
            "base_sum_of_urges_updated": True,
            "independent_observation": "one complete 800-step run",
        },
        "runs": runs,
        "events": events,
        "representative_traces": representative_traces,
        "analysis": analysis,
    }

    json_path = os.path.join(OUTPUT_DIR, "survivor_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
    csv_path = os.path.join(OUTPUT_DIR, "survivor_experiment_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    event_path = os.path.join(OUTPUT_DIR, "survivor_experiment_events.csv")
    with open(event_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {event_path}")
    print(
        f"Survival: {sum(run['survived'] for run in runs)}/{len(runs)}; "
        f"action events: {len(events)}"
    )


if __name__ == "__main__":
    main()

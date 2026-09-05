"""Focused 18-cell design, pilot sizing, and paired primary analysis."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Iterable

from scipy.stats import t as student_t

from .config import CONTROLLER_CONDITIONS, ExperimentConfig
from .engine import run_simulation
from .provenance import collect_provenance
from .storage import read_json, save_result, write_json


@dataclass(frozen=True)
class Condition:
    id: str
    family: str
    controller: str
    combat_enabled: bool
    agent_count: int
    regeneration_multiplier: float


def focused_conditions() -> tuple[Condition, ...]:
    primary = tuple(
        Condition(
            id=f"primary-{controller}-combat-{'on' if combat else 'off'}",
            family="primary_and_ablations",
            controller=controller,
            combat_enabled=combat,
            agent_count=12,
            regeneration_multiplier=1.0,
        )
        for controller in CONTROLLER_CONDITIONS
        for combat in (True, False)
    )
    resource = tuple(
        Condition(
            id=f"resource-{controller}-regen-{multiplier:g}",
            family="resource_sensitivity",
            controller=controller,
            combat_enabled=True,
            agent_count=12,
            regeneration_multiplier=multiplier,
        )
        for controller in ("dynamic", "fixed")
        for multiplier in (0.25, 4.0)
    )
    population = tuple(
        Condition(
            id=f"population-{controller}-agents-{count}",
            family="population_sensitivity",
            controller=controller,
            combat_enabled=True,
            agent_count=count,
            regeneration_multiplier=1.0,
        )
        for controller in ("dynamic", "fixed")
        for count in (6, 24)
    )
    conditions = primary + resource + population
    if len(conditions) != 18 or len({condition.id for condition in conditions}) != 18:
        raise AssertionError("The focused design must contain exactly 18 unique cells")
    return conditions


def condition_config(base: ExperimentConfig, condition: Condition, seed: int, trace_level: str = "summary") -> ExperimentConfig:
    result = replace(
        base,
        world_seed=seed,
        agent_count=condition.agent_count,
        controller_condition=condition.controller,
        combat_enabled=condition.combat_enabled,
        regeneration_multiplier=condition.regeneration_multiplier,
        trace_level=trace_level,
    )
    result.validate()
    return result


def _run_one(arguments: tuple[ExperimentConfig, Condition, int, str]) -> dict[str, Any]:
    config, condition, seed, output_path = arguments
    started = time.perf_counter()
    result = run_simulation(config)
    elapsed = time.perf_counter() - started
    if result.summary["ticks_completed"] != config.playback.ticks:
        raise RuntimeError(f"Incomplete run {condition.id} seed {seed}")
    result.summary["elapsed_compute_seconds"] = elapsed
    save_result(result, output_path)
    return {
        "condition_id": condition.id,
        "family": condition.family,
        "controller": condition.controller,
        "combat_enabled": condition.combat_enabled,
        "agent_count": condition.agent_count,
        "regeneration_multiplier": condition.regeneration_multiplier,
        "seed": seed,
        "configuration_digest": result.configuration_digest,
        "restricted_mean_survival_time": result.summary["restricted_mean_survival_time"],
        "survivors": result.summary["survivors"],
        "survivor_fraction": result.summary["survivor_fraction"],
        "secondary_metrics": result.summary["secondary_metrics"],
        "elapsed_compute_seconds": elapsed,
        "artifact": output_path,
    }


def run_matrix(
    base: ExperimentConfig,
    seeds: Iterable[int],
    output_dir: str | Path,
    *,
    phase: str,
    workers: int = 1,
    retry_once: bool = True,
    condition_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    all_conditions = focused_conditions()
    if condition_ids is None:
        conditions = all_conditions
    else:
        requested = set(condition_ids)
        conditions = tuple(condition for condition in all_conditions if condition.id in requested)
        missing = requested - {condition.id for condition in conditions}
        if missing:
            raise ValueError(f"Unknown focused-design condition IDs: {sorted(missing)}")
        if not conditions:
            raise ValueError("At least one condition must be selected")
    seeds = tuple(int(seed) for seed in seeds)
    root = Path(output_dir)
    run_dir = root / phase / "runs"
    arguments = []
    for condition in conditions:
        for seed in seeds:
            output_path = run_dir / condition.id / f"seed-{seed}.json.gz"
            arguments.append((condition_config(base, condition, seed), condition, seed, str(output_path)))
    rows: list[dict[str, Any]] = []
    failures: list[tuple[tuple[ExperimentConfig, Condition, int, str], str]] = []
    if workers <= 1:
        for argument in arguments:
            try:
                rows.append(_run_one(argument))
            except Exception as error:
                failures.append((argument, repr(error)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, argument): argument for argument in arguments}
            for future in as_completed(futures):
                argument = futures[future]
                try:
                    rows.append(future.result())
                except Exception as error:
                    failures.append((argument, repr(error)))
    if failures and retry_once:
        retry_failures = []
        for argument, _first_error in failures:
            try:
                rows.append(_run_one(argument))
            except Exception as error:
                retry_failures.append((argument, repr(error)))
        failures = retry_failures
    if failures:
        failure_document = [
            {"condition_id": argument[1].id, "seed": argument[2], "error": error}
            for argument, error in failures
        ]
        write_json(root / phase / "failures.json", failure_document, compressed=False)
        raise RuntimeError(
            f"{len(failures)} simulations failed twice; the batch is invalid and no outcomes were imputed"
        )
    rows.sort(key=lambda row: (row["condition_id"], row["seed"]))
    document = {
        "phase": phase,
        "design": [asdict(condition) for condition in conditions],
        "seed_blocks": list(seeds),
        "conditions": len(conditions),
        "runs": len(rows),
        "base_configuration": base.to_dict(),
        "rows": rows,
    }
    write_json(root / phase / "matrix-summary.json", document, compressed=False)
    return document


def primary_differences(matrix: dict[str, Any]) -> list[dict[str, float | int]]:
    dynamic_id = "primary-dynamic-combat-on"
    fixed_id = "primary-fixed-combat-on"
    by_key = {(row["condition_id"], row["seed"]): row for row in matrix["rows"]}
    differences = []
    for seed in matrix["seed_blocks"]:
        dynamic = by_key.get((dynamic_id, seed))
        fixed = by_key.get((fixed_id, seed))
        if dynamic is None or fixed is None:
            raise ValueError(f"Missing primary pair for seed {seed}")
        differences.append({
            "seed": seed,
            "dynamic": dynamic["restricted_mean_survival_time"],
            "fixed": fixed["restricted_mean_survival_time"],
            "difference": dynamic["restricted_mean_survival_time"] - fixed["restricted_mean_survival_time"],
        })
    return differences


def pilot_sample_size(matrix: dict[str, Any], base: ExperimentConfig) -> dict[str, Any]:
    recorded_config = json.dumps(matrix.get("base_configuration"), sort_keys=True, separators=(",", ":"))
    current_config = json.dumps(base.to_dict(), sort_keys=True, separators=(",", ":"))
    if recorded_config != current_config:
        raise ValueError(
            "Pilot configuration differs from the configuration being frozen; rerun the final primary pilot pair"
        )
    if tuple(matrix.get("seed_blocks", ())) != base.preregistration.pilot_seeds:
        raise ValueError("Sample-size planning requires exactly the preregistered pilot seed registry")
    differences = primary_differences(matrix)
    if len(differences) != len(base.preregistration.pilot_seeds):
        raise ValueError("Sample-size planning requires the complete final ten-seed primary pilot pair")
    values = [float(item["difference"]) for item in differences]
    pilot_sd = stdev(values)
    target = base.preregistration.target_ci_half_width_seconds
    n = base.preregistration.minimum_evaluation_blocks
    while True:
        critical = float(student_t.ppf(0.975, n - 1))
        planned_half_width = critical * pilot_sd / math.sqrt(n)
        if planned_half_width <= target:
            break
        n += 1
    pilot_digest = hashlib.sha256(
        json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provenance = collect_provenance(base)
    return {
        "freeze_status": "ready_for_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "primary-contrast paired-t precision planning",
        "pilot_seeds": list(base.preregistration.pilot_seeds),
        "pilot_differences": differences,
        "pilot_difference_sd": pilot_sd,
        "target_half_width_seconds": target,
        "minimum_n": base.preregistration.minimum_evaluation_blocks,
        "frozen_evaluation_n": n,
        "planned_half_width_seconds": planned_half_width,
        "evaluation_seeds": list(range(base.preregistration.evaluation_seed_start, base.preregistration.evaluation_seed_start + n)),
        "evaluation_runs": 18 * n,
        "focused_design": [asdict(condition) for condition in focused_conditions()],
        "base_configuration": base.to_dict(),
        "base_configuration_digest": base.digest(),
        "pilot_summary_sha256": pilot_digest,
        "provenance": provenance,
        "primary_analysis": {
            "outcome": "mean per-run restricted survival time across all initially present agents",
            "contrast": "dynamic minus fixed",
            "environment": "12 agents, baseline regeneration, hunting enabled",
            "interval": "95% paired-t over complete seed-block differences",
            "horizon_seconds": 3600,
        },
        "warning": "The pilot variance estimate is uncertain; evaluation n must not change after outcomes are inspected.",
    }


def validate_freeze_manifest(manifest: dict[str, Any], base: ExperimentConfig) -> None:
    if manifest.get("freeze_status") != "ready_for_evaluation":
        raise ValueError("Freeze manifest is not marked ready for evaluation")
    if manifest.get("base_configuration_digest") != base.digest():
        raise ValueError("Current final configuration does not match the frozen configuration")
    expected_seeds = list(
        range(
            base.preregistration.evaluation_seed_start,
            base.preregistration.evaluation_seed_start + int(manifest["frozen_evaluation_n"]),
        )
    )
    if manifest.get("evaluation_seeds") != expected_seeds:
        raise ValueError("Evaluation seed registry is inconsistent with the preregistration")
    current = collect_provenance(base)
    frozen = manifest.get("provenance", {})
    for key in ("operator_sha256", "implementation_tree_sha256", "adapter_version", "classifier_version"):
        if frozen.get(key) != current.get(key):
            raise ValueError(f"Frozen provenance mismatch for {key}")


def primary_analysis(matrix: dict[str, Any], confidence: float = 0.95) -> dict[str, Any]:
    differences = primary_differences(matrix)
    values = [float(item["difference"]) for item in differences]
    n = len(values)
    if n < 2:
        raise ValueError("At least two complete pairs are required")
    estimate = mean(values)
    sd = stdev(values)
    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / 2.0, n - 1))
    half_width = critical * sd / math.sqrt(n)
    return {
        "estimand": "dynamic minus fixed mean per-run restricted survival time",
        "horizon_seconds": 3600,
        "analysis_unit": "complete paired seed block",
        "n_pairs": n,
        "mean_difference_seconds": estimate,
        "sample_sd_seconds": sd,
        "confidence_level": confidence,
        "interval_method": "paired-t",
        "ci_lower_seconds": estimate - half_width,
        "ci_upper_seconds": estimate + half_width,
        "achieved_half_width_seconds": half_width,
        "seed_level_differences": differences,
    }


def _secondary_contrast_registry() -> tuple[tuple[str, str, str], ...]:
    contrasts: list[tuple[str, str, str]] = [
        (
            "dynamic_minus_fixed_combat_off",
            "primary-dynamic-combat-off",
            "primary-fixed-combat-off",
        ),
        ("dynamic_minus_fixed_regen_low", "resource-dynamic-regen-0.25", "resource-fixed-regen-0.25"),
        ("dynamic_minus_fixed_regen_high", "resource-dynamic-regen-4", "resource-fixed-regen-4"),
        ("dynamic_minus_fixed_agents_6", "population-dynamic-agents-6", "population-fixed-agents-6"),
        ("dynamic_minus_fixed_agents_24", "population-dynamic-agents-24", "population-fixed-agents-24"),
    ]
    for combat in ("on", "off"):
        for ablation in ("ablate_resolution", "ablate_selection", "ablate_securing"):
            contrasts.append((
                f"dynamic_minus_{ablation}_combat_{combat}",
                f"primary-dynamic-combat-{combat}",
                f"primary-{ablation}-combat-{combat}",
            ))
    for controller in CONTROLLER_CONDITIONS:
        contrasts.append((
            f"combat_on_minus_off_{controller}",
            f"primary-{controller}-combat-on",
            f"primary-{controller}-combat-off",
        ))
    return tuple(contrasts)


def _endpoint(row: dict[str, Any], name: str) -> float:
    if name in row:
        return float(row[name])
    value = row["secondary_metrics"][name]
    if value is None:
        raise ValueError(f"Endpoint {name} is undefined for condition {row['condition_id']} seed {row['seed']}")
    return float(value)


def _paired_endpoint_analysis(
    matrix: dict[str, Any],
    left_id: str,
    right_id: str,
    endpoint: str,
    confidence: float,
) -> dict[str, Any]:
    by_key = {(row["condition_id"], row["seed"]): row for row in matrix["rows"]}
    records = []
    for seed in matrix["seed_blocks"]:
        left = by_key.get((left_id, seed))
        right = by_key.get((right_id, seed))
        if left is None or right is None:
            raise ValueError(f"Missing pair {left_id} versus {right_id} for seed {seed}")
        left_value = _endpoint(left, endpoint)
        right_value = _endpoint(right, endpoint)
        records.append({
            "seed": seed,
            "left": left_value,
            "right": right_value,
            "difference": left_value - right_value,
        })
    values = [record["difference"] for record in records]
    n = len(values)
    if n < 2:
        raise ValueError("At least two complete seed pairs are required")
    estimate = mean(values)
    sd = stdev(values)
    critical = float(student_t.ppf(1.0 - (1.0 - confidence) / 2.0, n - 1))
    half_width = critical * sd / math.sqrt(n)
    return {
        "endpoint": endpoint,
        "left_condition": left_id,
        "right_condition": right_id,
        "estimand": "mean paired left-minus-right run-level difference",
        "n_pairs": n,
        "mean_difference": estimate,
        "sample_sd": sd,
        "confidence_level": confidence,
        "interval_method": "paired-t",
        "ci_lower": estimate - half_width,
        "ci_upper": estimate + half_width,
        "achieved_half_width": half_width,
        "seed_level_differences": records,
    }


def study_analysis(matrix: dict[str, Any], confidence: float = 0.95) -> dict[str, Any]:
    """Primary preregistered estimate plus balanced secondary paired summaries."""
    endpoints = (
        "restricted_mean_survival_time",
        "survivor_fraction",
        "resource_quantity_acquired",
        "failed_journeys",
        "inspections",
        "securing_scans",
        "motive_switches",
        "strategy_switches",
        "attack_requests",
        "resolved_duels",
        "unclassified_fraction",
        "prediction_expected",
        "prediction_unexpected",
    )
    secondary = {}
    for name, left_id, right_id in _secondary_contrast_registry():
        secondary[name] = {
            endpoint: _paired_endpoint_analysis(matrix, left_id, right_id, endpoint, confidence)
            for endpoint in endpoints
        }
    return {
        "analysis_version": "focused-study-analysis-v1",
        "analysis_unit": "complete paired seed block; agents and ticks are not independent replicates",
        "primary": primary_analysis(matrix, confidence),
        "secondary": secondary,
        "interpretation_warning": (
            "The primary pilot precision target applies only to the primary contrast. "
            "Secondary intervals report achieved precision and do not isolate unmodeled interactions."
        ),
    }

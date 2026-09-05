"""Command-line entry points for validation, simulation, and frozen batches."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

from .config import ExperimentConfig
from .engine import run_simulation
from .experiment import pilot_sample_size, run_matrix, study_analysis, validate_freeze_manifest
from .storage import load_config, read_json, save_config, save_result, write_json


def _base_config(path: str | None) -> ExperimentConfig:
    return load_config(path) if path else ExperimentConfig()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="psi-survival")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("write-config")
    config_parser.add_argument("path")

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config")
    simulate.add_argument("--seed", type=int)
    simulate.add_argument("--agents", type=int)
    simulate.add_argument("--controller", choices=("dynamic", "fixed", "ablate_resolution", "ablate_selection", "ablate_securing"))
    simulate.add_argument("--combat", choices=("on", "off"))
    simulate.add_argument("--regen", type=float)
    simulate.add_argument("--ticks", type=int)
    simulate.add_argument("--trace", choices=("full", "summary"), default="full")
    simulate.add_argument("--output", required=True)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--config")
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--workers", type=int, default=1)
    pilot.add_argument(
        "--primary-only",
        action="store_true",
        help="rerun only the final dynamic/fixed combat-on primary pilot pair",
    )

    size = subparsers.add_parser("size-evaluation")
    size.add_argument("--config")
    size.add_argument("--pilot-summary", required=True)
    size.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config")
    evaluate.add_argument("--freeze-manifest", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--workers", type=int, default=1)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--matrix-summary", required=True)
    analyze.add_argument("--output", required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--result", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--ema", type=float, help="display-only activation EMA lambda")

    args = parser.parse_args(argv)
    if args.command == "write-config":
        save_config(ExperimentConfig(), args.path)
        return 0
    if args.command == "simulate":
        config = _base_config(args.config)
        changes = {"trace_level": args.trace}
        if args.seed is not None:
            changes["world_seed"] = args.seed
        if args.agents is not None:
            changes["agent_count"] = args.agents
        if args.controller is not None:
            changes["controller_condition"] = args.controller
        if args.combat is not None:
            changes["combat_enabled"] = args.combat == "on"
        if args.regen is not None:
            changes["regeneration_multiplier"] = args.regen
        if args.ticks is not None:
            changes["playback"] = replace(config.playback, ticks=args.ticks)
        config = replace(config, **changes)
        config.validate()
        started = time.perf_counter()
        result = run_simulation(config)
        result.summary["elapsed_compute_seconds"] = time.perf_counter() - started
        save_result(result, args.output)
        return 0
    if args.command == "pilot":
        config = _base_config(args.config)
        run_matrix(
            config,
            config.preregistration.pilot_seeds,
            args.output_dir,
            phase="pilot",
            workers=args.workers,
            condition_ids=(
                ("primary-dynamic-combat-on", "primary-fixed-combat-on")
                if args.primary_only
                else None
            ),
        )
        return 0
    if args.command == "size-evaluation":
        config = _base_config(args.config)
        manifest = pilot_sample_size(read_json(args.pilot_summary), config)
        write_json(args.output, manifest, compressed=False)
        return 0
    if args.command == "evaluate":
        config = _base_config(args.config)
        manifest = read_json(args.freeze_manifest)
        validate_freeze_manifest(manifest, config)
        run_matrix(
            config,
            manifest["evaluation_seeds"],
            args.output_dir,
            phase="evaluation",
            workers=args.workers,
        )
        return 0
    if args.command == "analyze":
        analysis = study_analysis(read_json(args.matrix_summary))
        write_json(args.output, analysis, compressed=False)
        return 0
    if args.command == "render":
        from .render import render_all
        render_all(args.result, args.output_dir, display_ema_lambda=args.ema)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

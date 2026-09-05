"""Versioned configuration for the PSI survival experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Any


CONTROLLER_CONDITIONS = ("dynamic", "fixed", "ablate_resolution", "ablate_selection", "ablate_securing")


@dataclass(frozen=True)
class SourceParameters:
    upstream_repository: str = "https://github.com/joschabach/micropsi2"
    operator_path: str = "micropsi_core/nodenet/stepoperators.py"
    operator_last_commit: str = "8b9397f965fec5b8e28c8f0007d3da0336440972"
    operator_sha256: str = "adee2f202381665d752b7a913d860464eea48775ea36a1f666da9b9d71cdf0e9"
    paper_filename: str = "1-s2.0-S1389041711000647-main.pdf"
    paper_sha256: str = "4f4e56bd4369eb9fdc4c0057f98a1357e96f07a398d928ff181d22a8fcd2227e"


@dataclass(frozen=True)
class TerrainParameters:
    width: int = 64
    height: int = 64
    road_speed: float = 1.0
    rocky_speed: float = 0.65
    puddle_speed: float = 0.35
    road_energy_cost: float = 0.001
    rocky_energy_cost: float = 0.002
    puddle_energy_cost: float = 0.004
    road_lines: tuple[int, ...] = (8, 24, 40, 56)
    puddle_patch_count: int = 8
    puddle_patch_size: int = 5
    cluster_radius: int = 6
    placement_attempts: int = 50_000


@dataclass(frozen=True)
class PhysiologyParameters:
    initial_energy: float = 0.90
    initial_water: float = 0.95
    initial_integrity: float = 0.95
    energy_decay: float = 0.0010
    water_decay: float = 0.0012
    integrity_decay: float = 0.0002
    energy_threshold: float = 0.0
    water_threshold: float = 0.0
    integrity_threshold: float = 0.0


@dataclass(frozen=True)
class ResourceParameters:
    capacity: float = 1.0
    regeneration_rate: float = 1.0 / 120.0
    max_request: float = 0.25
    food_count: int = 8
    well_count: int = 8
    healing_count: int = 4
    harmful_count: int = 4
    clustered_sites: int = 18
    cluster_count: int = 3
    landmark_probability: float = 0.5
    # Fixed marginal conditional prior among unidentified initial sites:
    # expected unidentified Food/Well/Healing/Harmful counts = 4/4/2/4.
    unidentified_prior: tuple[tuple[str, float], ...] = (
        ("food", 2.0 / 7.0),
        ("well", 2.0 / 7.0),
        ("healing", 1.0 / 7.0),
        ("mixed_harmful", 2.0 / 7.0),
    )


@dataclass(frozen=True)
class PerceptionParameters:
    min_radius: float = 3.0
    resolution_radius_span: float = 7.0
    securing_scan_radius: float = 10.0
    agent_memory_seconds: int = 30
    agent_memory_tau: float = 30.0
    resource_confidence_tau: float = 120.0
    threat_memory_seconds: int = 10
    adjacency_distance: float = 1.0


@dataclass(frozen=True)
class PlannerParameters:
    horizon_seconds: int = 60
    urgency_horizon: float = 300.0
    wait_durations: tuple[int, ...] = (5, 15, 30)
    death_penalty: float = 2.0
    information_weight: float = 0.05
    uncertainty_weight: float = 0.05
    motive_margin_floor: float = 0.02
    motive_margin_scale: float = 0.18
    target_margin_floor: float = 0.02
    target_margin_scale: float = 0.08
    planning_budget_floor: int = 2
    planning_budget_scale: int = 6


@dataclass(frozen=True)
class CombatParameters:
    strength_energy_weight: float = 0.4
    strength_water_weight: float = 0.2
    strength_integrity_weight: float = 0.4
    sigmoid_scale: float = 0.15
    engagement_distance: float = 1.0


@dataclass(frozen=True)
class EmotionParameters:
    activation_min: float = 0.0
    activation_max: float = 1.0
    securing_raw_min: float = -0.5
    securing_raw_max: float = 2.0
    pleasure_raw_min: float = -2.0
    pleasure_raw_max: float = 2.0
    display_score_threshold: float = 0.50
    display_gap_threshold: float = 0.05
    fuzzy_sharpness: float = 150.0
    display_ema_lambda: float = 0.1
    adapter_version: str = "hybrid-adapter-v1"
    classifier_version: str = "cai-table2-literal-v1"


@dataclass(frozen=True)
class PlaybackParameters:
    ticks: int = 3_600
    dt: float = 1.0
    fps: int = 20
    heatmap_bins: int = 120
    width_px: int = 1280
    height_px: int = 720

    @property
    def seconds(self) -> float:
        return self.ticks / self.fps


@dataclass(frozen=True)
class Preregistration:
    pilot_seeds: tuple[int, ...] = tuple(range(1001, 1011))
    evaluation_seed_start: int = 2001
    minimum_evaluation_blocks: int = 30
    demonstration_seed: int = 900_001
    target_ci_half_width_seconds: float = 180.0
    confidence_level: float = 0.95
    interval_method: str = "paired_t"


@dataclass(frozen=True)
class ExperimentConfig:
    protocol_version: str = "2.0"
    world_seed: int = 900_001
    agent_count: int = 12
    controller_condition: str = "dynamic"
    combat_enabled: bool = True
    regeneration_multiplier: float = 1.0
    trace_level: str = "full"
    sources: SourceParameters = field(default_factory=SourceParameters)
    terrain: TerrainParameters = field(default_factory=TerrainParameters)
    physiology: PhysiologyParameters = field(default_factory=PhysiologyParameters)
    resources: ResourceParameters = field(default_factory=ResourceParameters)
    perception: PerceptionParameters = field(default_factory=PerceptionParameters)
    planner: PlannerParameters = field(default_factory=PlannerParameters)
    combat: CombatParameters = field(default_factory=CombatParameters)
    emotion: EmotionParameters = field(default_factory=EmotionParameters)
    playback: PlaybackParameters = field(default_factory=PlaybackParameters)
    preregistration: Preregistration = field(default_factory=Preregistration)

    def validate(self) -> None:
        if self.protocol_version != "2.0":
            raise ValueError("This implementation accepts protocol version 2.0 only")
        if self.controller_condition not in CONTROLLER_CONDITIONS:
            raise ValueError(f"Unknown controller condition: {self.controller_condition}")
        if self.agent_count < 1:
            raise ValueError("agent_count must be positive")
        if self.regeneration_multiplier < 0:
            raise ValueError("regeneration_multiplier cannot be negative")
        if self.trace_level not in {"full", "summary"}:
            raise ValueError("trace_level must be 'full' or 'summary'")
        if self.playback.dt <= 0 or self.playback.ticks <= 0 or self.playback.fps <= 0:
            raise ValueError("dt, ticks, and fps must be positive")
        if self.playback.ticks != 3_600 or self.playback.fps != 20:
            # Alternate settings are allowed for tests and pilots, but cannot be
            # mislabelled as the preregistered final playback.
            if self.trace_level == "full" and self.world_seed == self.preregistration.demonstration_seed:
                raise ValueError("The demonstration must be 3,600 ticks at 20 fps")
        inventory = (
            self.resources.food_count
            + self.resources.well_count
            + self.resources.healing_count
            + self.resources.harmful_count
        )
        if inventory != 24:
            raise ValueError("The baseline resource inventory must contain 24 sites")
        if self.resources.clustered_sites % self.resources.cluster_count:
            raise ValueError("clustered_sites must divide evenly across clusters")
        prior_total = sum(value for _, value in self.resources.unidentified_prior)
        if abs(prior_total - 1.0) > 1e-12:
            raise ValueError("The unidentified resource prior must sum to one")
        if set(self.preregistration.pilot_seeds) & {self.preregistration.demonstration_seed}:
            raise ValueError("The demonstration seed must be excluded from pilot seeds")
        if self.preregistration.evaluation_seed_start <= max(self.preregistration.pilot_seeds):
            raise ValueError("Evaluation seeds must be disjoint from pilot seeds")

    def with_condition(self, **changes: Any) -> "ExperimentConfig":
        result = replace(self, **changes)
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        def tuple_fields(values: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
            output = dict(values)
            for name in names:
                if name in output:
                    output[name] = tuple(tuple(item) if isinstance(item, list) else item for item in output[name])
            return output

        data = dict(payload)
        data["sources"] = SourceParameters(**data.get("sources", {}))
        data["terrain"] = TerrainParameters(**tuple_fields(data.get("terrain", {}), ("road_lines",)))
        data["physiology"] = PhysiologyParameters(**data.get("physiology", {}))
        data["resources"] = ResourceParameters(
            **tuple_fields(data.get("resources", {}), ("unidentified_prior",))
        )
        data["perception"] = PerceptionParameters(**data.get("perception", {}))
        data["planner"] = PlannerParameters(**tuple_fields(data.get("planner", {}), ("wait_durations",)))
        data["combat"] = CombatParameters(**data.get("combat", {}))
        data["emotion"] = EmotionParameters(**data.get("emotion", {}))
        data["playback"] = PlaybackParameters(**data.get("playback", {}))
        data["preregistration"] = Preregistration(
            **tuple_fields(data.get("preregistration", {}), ("pilot_seeds",))
        )
        result = cls(**data)
        result.validate()
        return result

"""Deterministic initial-world generation and serialization."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import ExperimentConfig
from .rng import bernoulli, stream
from .state import AgentState, Need, RESOURCE_EFFECTS, ResourceState, ResourceType, WorldState
from .terrain import TerrainMap


@dataclass
class InitialWorld:
    terrain: TerrainMap
    world: WorldState

    def artifact(self, config: ExperimentConfig) -> dict:
        return {
            "protocol_version": config.protocol_version,
            "configuration_digest": config.digest(),
            "world_seed": config.world_seed,
            "terrain": self.terrain.to_rows(),
            "resources": [
                {
                    "id": resource.id,
                    "position": list(resource.position),
                    "type": resource.type.value,
                    "effect_per_unit": list(resource.effect_per_unit),
                    "capacity": resource.capacity,
                    "stock": resource.stock,
                    "regeneration_rate": resource.regeneration_rate,
                    "public_type": resource.public_type,
                    "is_corpse": resource.is_corpse,
                }
                for resource in sorted(self.world.resources.values(), key=lambda item: item.id)
            ],
            "agents": [
                {
                    "id": agent.id,
                    "position": list(agent.position),
                    "energy": agent.energy,
                    "water": agent.water,
                    "integrity": agent.integrity,
                    "ruling_need": agent.ruling_need.value,
                }
                for agent in sorted(self.world.agents.values(), key=lambda item: item.id)
            ],
        }


def _resource_positions(config: ExperimentConfig) -> list[tuple[int, int]]:
    terrain = config.terrain
    resources = config.resources
    random = stream(config.world_seed, "initial-world", "resource-sites")
    per_cluster = resources.clustered_sites // resources.cluster_count
    used: set[tuple[int, int]] = set()
    positions: list[tuple[int, int]] = []
    offsets = [
        (dx, dy)
        for dx in range(-terrain.cluster_radius, terrain.cluster_radius + 1)
        for dy in range(-terrain.cluster_radius, terrain.cluster_radius + 1)
        if dx * dx + dy * dy <= terrain.cluster_radius * terrain.cluster_radius
    ]
    for cluster_index in range(resources.cluster_count):
        center = (
            random.randint(terrain.cluster_radius, terrain.width - terrain.cluster_radius - 1),
            random.randint(terrain.cluster_radius, terrain.height - terrain.cluster_radius - 1),
        )
        candidates = list(offsets)
        random.shuffle(candidates)
        accepted = 0
        for dx, dy in candidates:
            candidate = (center[0] + dx, center[1] + dy)
            if candidate in used:
                continue
            used.add(candidate)
            positions.append(candidate)
            accepted += 1
            if accepted == per_cluster:
                break
        if accepted != per_cluster:
            raise ValueError(f"Could not place six unique resources in cluster {cluster_index}")
    attempts = 0
    while len(positions) < 24 and attempts < terrain.placement_attempts:
        attempts += 1
        candidate = (random.randrange(terrain.width), random.randrange(terrain.height))
        if candidate in used:
            continue
        used.add(candidate)
        positions.append(candidate)
    if len(positions) != 24:
        raise ValueError("Could not place all 24 resource sites within the configured attempt bound")
    return positions


def _resources(config: ExperimentConfig) -> dict[str, ResourceState]:
    positions = _resource_positions(config)
    inventory = (
        [ResourceType.FOOD] * config.resources.food_count
        + [ResourceType.WELL] * config.resources.well_count
        + [ResourceType.HEALING] * config.resources.healing_count
        + [ResourceType.MIXED_HARMFUL] * config.resources.harmful_count
    )
    random = stream(config.world_seed, "initial-world", "resource-types")
    random.shuffle(inventory)
    output: dict[str, ResourceState] = {}
    for index, (position, resource_type) in enumerate(zip(positions, inventory), start=1):
        public_type = resource_type is not ResourceType.MIXED_HARMFUL and bernoulli(
            config.resources.landmark_probability,
            config.world_seed,
            "initial-world",
            "landmark",
            index,
        )
        resource_id = f"resource-{index:02d}"
        output[resource_id] = ResourceState(
            id=resource_id,
            position=(float(position[0]), float(position[1])),
            type=resource_type,
            effect_per_unit=RESOURCE_EFFECTS[resource_type],
            capacity=config.resources.capacity,
            stock=config.resources.capacity,
            regeneration_rate=config.resources.regeneration_rate * config.regeneration_multiplier,
            public_type=public_type,
        )
    return output


def _initial_ruling_need(config: ExperimentConfig) -> Need:
    physiology = config.physiology
    body = {
        Need.ENERGY: (physiology.initial_energy, physiology.energy_decay),
        Need.WATER: (physiology.initial_water, physiology.water_decay),
        Need.INTEGRITY: (physiology.initial_integrity, physiology.integrity_decay),
    }
    order = (Need.WATER, Need.ENERGY, Need.INTEGRITY)

    def strength(need: Need) -> float:
        value, depletion = body[need]
        urge = 1.0 - value
        tau = value / depletion if depletion > 0 else math.inf
        urgency = min(1.0, max(0.0, 1.0 - tau / config.planner.urgency_horizon))
        return 0.5 * urge + 0.5 * urgency

    return max(order, key=strength)


def _agents(config: ExperimentConfig, resources: dict[str, ResourceState]) -> dict[str, AgentState]:
    random = stream(config.world_seed, "initial-world", "agent-placement", config.agent_count)
    occupied: list[tuple[float, float]] = []
    resource_positions = [resource.position for resource in resources.values()]
    output: dict[str, AgentState] = {}
    attempts = 0
    while len(output) < config.agent_count and attempts < config.terrain.placement_attempts:
        attempts += 1
        candidate = (float(random.randrange(config.terrain.width)), float(random.randrange(config.terrain.height)))
        if any(math.dist(candidate, position) < 2.0 for position in resource_positions):
            continue
        if any(math.dist(candidate, position) < 2.0 for position in occupied):
            continue
        agent_id = f"agent-{len(output) + 1:02d}"
        output[agent_id] = AgentState(
            id=agent_id,
            position=candidate,
            energy=config.physiology.initial_energy,
            water=config.physiology.initial_water,
            integrity=config.physiology.initial_integrity,
            ruling_need=_initial_ruling_need(config),
        )
        occupied.append(candidate)
    if len(output) != config.agent_count:
        raise ValueError(
            f"Could not place {config.agent_count} agents at the required separation after "
            f"{config.terrain.placement_attempts} attempts"
        )
    return output


def generate_initial_world(config: ExperimentConfig) -> InitialWorld:
    config.validate()
    terrain = TerrainMap.generate(config.terrain, config.world_seed)
    resources = _resources(config)
    agents = _agents(config, resources)
    return InitialWorld(terrain=terrain, world=WorldState(tick=0, time=0.0, agents=agents, resources=resources))


"""Immutable observations and perception application."""

from __future__ import annotations

from dataclasses import dataclass

from .beliefs import age_agent_beliefs, observe_agent, observe_resource_presence, remember_attack
from .config import ExperimentConfig
from .rng import bernoulli
from .state import AgentState, Position, ResourceState, WorldEvent, WorldState, distance


@dataclass(frozen=True)
class ResourcePresence:
    id: str
    position: Position
    is_corpse: bool


@dataclass(frozen=True)
class AgentPresence:
    id: str
    position: Position
    alive: bool


@dataclass(frozen=True)
class PerceptionResult:
    normal_radius: float
    securing_scan: bool
    resources: tuple[ResourcePresence, ...]
    agents: tuple[AgentPresence, ...]
    normal_agent_ids: tuple[str, ...]
    attack_events: tuple[tuple[str, Position], ...]


def perceive(agent: AgentState, snapshot: WorldState, prior_events: list[WorldEvent], config: ExperimentConfig) -> PerceptionResult:
    controls = agent.previous_controls
    normal_radius = config.perception.min_radius + config.perception.resolution_radius_span * controls.resolution
    securing = bernoulli(
        controls.securing_rate,
        config.world_seed,
        "perception-scan",
        snapshot.tick,
        agent.id,
    )
    effective_radius = max(normal_radius, config.perception.securing_scan_radius if securing else 0.0)
    resources = tuple(
        ResourcePresence(resource.id, resource.position, resource.is_corpse)
        for resource in sorted(snapshot.resources.values(), key=lambda item: item.id)
        if distance(agent.position, resource.position) <= effective_radius
    )
    agents = tuple(
        AgentPresence(other.id, other.position, other.alive)
        for other in sorted(snapshot.agents.values(), key=lambda item: item.id)
        if other.id != agent.id and distance(agent.position, other.position) <= effective_radius
    )
    normal_agent_ids = tuple(
        other.id
        for other in sorted(snapshot.agents.values(), key=lambda item: item.id)
        if other.id != agent.id
        and other.alive
        and distance(agent.position, other.position) <= normal_radius
    )
    attacks = tuple(
        (
            str(event.agent_id),
            tuple(event.details.get("attacker_position", event.position or agent.position)),
        )
        for event in prior_events
        if event.event_type == "attack"
        and event.target_id == agent.id
        and event.agent_id is not None
    )
    return PerceptionResult(normal_radius, securing, resources, agents, normal_agent_ids, attacks)


def apply_perception(
    agent: AgentState,
    observation: PerceptionResult,
    now: float,
    config: ExperimentConfig,
) -> None:
    age_agent_beliefs(agent, now, config)
    for presence in observation.resources:
        observe_resource_presence(
            agent,
            presence.id,
            presence.position,
            presence.is_corpse,
            now,
            config,
        )
    for presence in observation.agents:
        observe_agent(agent, presence.id, presence.position, presence.alive, now)
    for attacker_id, attacker_position in observation.attack_events:
        remember_attack(agent, attacker_id, attacker_position, now)

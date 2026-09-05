"""Global ten-phase engine for the preregistered hybrid experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .beliefs import (
    initialize_resource_beliefs,
    observe_resource_exact,
    refresh_resource_estimates,
)
from .combat import AttackRequest, match_attacks, resolve_duel, win_probability
from .config import ExperimentConfig
from .emotions import DriveState, compute_drives, execute_operator
from .initialization import InitialWorld, generate_initial_world
from .observations import PerceptionResult, apply_perception, perceive
from .planner import PlanningDecision, choose_plan
from .predictions import (
    abandon_active_episode,
    complete_episode,
    consume_cognitive_outcomes,
    create_prediction,
    resolve_episode_predictions,
    start_episode,
)
from .provenance import collect_provenance
from .rng import choice
from .state import (
    AgentState,
    Commitment,
    Need,
    ResourceBelief,
    ResourceState,
    ResourceType,
    Strategy,
    WorldEvent,
    WorldState,
    distance,
)
from .terrain import Route, TraversalResult, traverse_route


@dataclass
class RunAccumulator:
    strategy_ticks: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    motive_ticks: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    scans: Counter = field(default_factory=Counter)
    inspections: Counter = field(default_factory=Counter)
    resource_quantity: Counter = field(default_factory=Counter)
    motive_switches: Counter = field(default_factory=Counter)
    strategy_switches: Counter = field(default_factory=Counter)
    failed_journeys: Counter = field(default_factory=Counter)
    clipping: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    emotion_ticks: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))


@dataclass
class SimulationResult:
    config: dict[str, Any]
    configuration_digest: str
    provenance: dict[str, Any]
    initial_state: dict[str, Any]
    agents: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    analysis_telemetry: dict[str, dict[str, Any]]
    traces: dict[str, list[dict[str, Any]]]
    world_frames: list[dict[str, Any]]
    summary: dict[str, Any]


def _event(
    world: WorldState,
    tick: int,
    time: float,
    phase: str,
    event_type: str,
    *,
    agent_id: str | None = None,
    target_id: str | None = None,
    resource_id: str | None = None,
    position: tuple[float, float] | None = None,
    details: dict[str, Any] | None = None,
) -> WorldEvent:
    item = WorldEvent(tick, time, phase, event_type, position, agent_id, target_id, resource_id, details or {})
    world.events.append(item)
    world.events_since_snapshot.append(item)
    return item


def _prediction(agent: AgentState, episode_id: str | None, category: str):
    if episode_id is None:
        return None
    return agent.pending_predictions.get(f"{episode_id}:prediction:{category}")


def _create_corpse(agent: AgentState, world: WorldState, tick: int, time: float) -> ResourceState:
    corpse_id = f"corpse-{agent.id}-{tick:04d}"
    effect = (0.8 * max(0.0, agent.energy), 0.8 * max(0.0, agent.water), 0.0)
    stock = 0.0 if effect[0] == 0.0 and effect[1] == 0.0 else 1.0
    corpse = ResourceState(
        id=corpse_id,
        position=agent.position,
        type=ResourceType.CORPSE,
        effect_per_unit=effect,
        capacity=1.0,
        stock=stock,
        regeneration_rate=0.0,
        is_corpse=True,
        public_type=False,
        spawned_at=time,
        source_agent_id=agent.id,
    )
    world.resources[corpse_id] = corpse
    return corpse


def _kill_agent(agent: AgentState, world: WorldState, tick: int, time: float, cause: str) -> ResourceState | None:
    if not agent.alive:
        return None
    agent.alive = False
    agent.death_time = time
    agent.death_cause = cause
    abandon_active_episode(agent, tick, involuntary_death=True)
    for prediction in agent.pending_predictions.values():
        if prediction.status == "pending" and prediction.category == "route_arrival":
            prediction.status = "resolved"
            prediction.actual = False
            prediction.resolved_tick = tick
    corpse = _create_corpse(agent, world, tick, time)
    _event(
        world,
        tick,
        time,
        "death",
        "death",
        agent_id=agent.id,
        resource_id=corpse.id,
        position=agent.position,
        details={"cause": cause, "body": list(agent.body), "corpse_stock": corpse.stock},
    )
    return corpse


def _install_commitment(agent: AgentState, decision: PlanningDecision, tick: int, config: ExperimentConfig) -> None:
    plan = decision.plan
    previous_commitment = agent.commitment
    if decision.commitment_changed:
        abandon_active_episode(agent, tick)
        episode = start_episode(
            agent,
            plan.template.strategy,
            plan.template.objective,
            plan.template.target_id,
            tick,
        )
        episode_id = episode.id
    else:
        episode_id = agent.commitment.episode_id if agent.commitment else None
        if episode_id is None:
            episode_id = start_episode(
                agent,
                plan.template.strategy,
                plan.template.objective,
                plan.template.target_id,
                tick,
            ).id
    agent.commitment = Commitment(
        strategy=plan.template.strategy,
        target_id=plan.template.target_id,
        target_position=plan.template.target_position,
        value=plan.value,
        need=plan.template.need,
        route_kind=plan.route_kind,
        wait_seconds=plan.template.wait_seconds,
        wait_until_time=(
            previous_commitment.wait_until_time
            if previous_commitment is not None and not decision.commitment_changed
            else None
        ),
        episode_id=episode_id,
        created_tick=previous_commitment.created_tick if previous_commitment and not decision.commitment_changed else tick,
    )
    agent.strategy = plan.template.strategy
    agent.target_id = plan.template.target_id
    agent.plan_trace = decision.considered

    create_prediction(agent, episode_id, "route_arrival", True, "arrive at planned destination while alive", tick)
    belief = plan.template.resource_belief
    if belief is not None and plan.template.strategy in {Strategy.FORAGE, Strategy.INSPECT, Strategy.WAIT}:
        predicted_stock = belief.estimated_stock >= config.resources.max_request
        create_prediction(agent, episode_id, "stock_sufficient", predicted_stock, "inspect or consume target stock", tick)
        if not belief.identified:
            predicted_type = choice(
                belief.most_likely_types,
                config.world_seed,
                "policy-tie",
                tick,
                agent.id,
                belief.resource_id,
                "resource-type",
            )
            create_prediction(agent, episode_id, "resource_type", predicted_type.value, "reveal target type", tick)
    if plan.template.strategy is Strategy.HUNT and plan.template.agent_belief is not None:
        predicted_win = win_probability(agent.body, plan.template.agent_belief.estimated_body, config) >= 0.5
        create_prediction(agent, episode_id, "hunt_win", predicted_win, "duel resolves", tick)


def _serialize_event(event: WorldEvent) -> dict[str, Any]:
    return {
        "tick": event.tick,
        "time": event.time,
        "phase": event.phase,
        "event_type": event.event_type,
        "position": list(event.position) if event.position is not None else None,
        "agent_id": event.agent_id,
        "target_id": event.target_id,
        "resource_id": event.resource_id,
        "details": event.details,
    }


class SimulationEngine:
    def __init__(self, config: ExperimentConfig, initial: InitialWorld | None = None):
        config.validate()
        self.config = config
        self.initial = initial or generate_initial_world(config)
        self.terrain = self.initial.terrain
        self.world = self.initial.world
        self.terrain.register_static_goals(
            resource.position for resource in self.world.resources.values() if not resource.is_corpse
        )
        self.initial_artifact = self.initial.artifact(config)
        self.provenance = collect_provenance(config)
        self.accumulator = RunAccumulator()
        self.traces: dict[str, list[dict[str, Any]]] = {agent_id: [] for agent_id in self.world.agents}
        self.analysis_telemetry: dict[str, dict[str, Any]] = {
            agent_id: {
                "tick": [],
                "time": [],
                "alive": [],
                "energy": [],
                "water": [],
                "integrity": [],
                "position_x": [],
                "position_y": [],
                "ruling_need": [],
                "strategy": [],
                "target_id": [],
                "certainty": [],
                "securing_scan": [],
                "selected_plan_value": [],
                "selected_death_probability": [],
                "literal_emotion": [],
                "dominant_emotion": [],
                "raw_modulators": {},
                "control_modulators": {},
                "classifier_inputs": {},
                "emotion_scores": {},
            }
            for agent_id in self.world.agents
        }
        self.world_frames: list[dict[str, Any]] = []
        for agent in self.world.agents.values():
            initialize_resource_beliefs(agent, self.world.resources, config)

    def _perception_phase(self, prior_events: list[WorldEvent]) -> dict[str, PerceptionResult]:
        living = sorted(self.world.living_agents, key=lambda item: item.id)
        observations = {agent.id: perceive(agent, self.world, prior_events, self.config) for agent in living}
        for agent in living:
            observation = observations[agent.id]
            apply_perception(agent, observation, self.world.time, self.config)
            refresh_resource_estimates(agent, self.world.time, self.config)
            self.accumulator.scans[agent.id] += int(observation.securing_scan)
            if (
                agent.commitment is not None
                and agent.commitment.strategy is Strategy.FLEE
                and agent.commitment.target_id not in observation.normal_agent_ids
            ):
                complete_episode(agent, agent.commitment.episode_id, self.world.tick, True)
                agent.commitment = None
        return observations

    def _cognitive_phase(self) -> tuple[dict[str, DriveState], dict[str, dict[str, Any]]]:
        drives: dict[str, DriveState] = {}
        cognitive_log: dict[str, dict[str, Any]] = {}
        for agent in sorted(self.world.living_agents, key=lambda item: item.id):
            expected, unexpected, prediction_records, episode_records = consume_cognitive_outcomes(agent)
            drive = compute_drives(agent, self.config)
            result = execute_operator(agent, drive, expected, unexpected, self.config)
            drives[agent.id] = drive
            for field, clipped in result.clipping.items():
                self.accumulator.clipping[agent.id][field] += int(clipped)
            cognitive_log[agent.id] = {
                "expected_events": expected,
                "unexpected_events": unexpected,
                "resolved_predictions": prediction_records,
                "completed_episodes": episode_records,
                "clipping": result.clipping,
                "score_maximum": result.score_maximum,
                "score_gap": result.score_gap,
            }
        return drives, cognitive_log

    def _planning_phase(self, drives: dict[str, DriveState], tick: int) -> dict[str, PlanningDecision]:
        decisions: dict[str, PlanningDecision] = {}
        for agent in sorted(self.world.living_agents, key=lambda item: item.id):
            decision = choose_plan(agent, drives[agent.id], self.terrain, self.config, tick, self.world.time)
            if decision.motive_switched or decision.infeasibility_override:
                self.accumulator.motive_switches[agent.id] += 1
                _event(
                    self.world,
                    tick,
                    self.world.time,
                    "planning",
                    "motive_switch",
                    agent_id=agent.id,
                    details={
                        "from": decision.ruling_need_before.value,
                        "to": decision.ruling_need_after.value,
                        "infeasibility_override": decision.infeasibility_override,
                    },
                )
            if decision.commitment_changed:
                self.accumulator.strategy_switches[agent.id] += 1
                _event(
                    self.world,
                    tick,
                    self.world.time,
                    "planning",
                    "strategy_switch",
                    agent_id=agent.id,
                    target_id=decision.plan.template.target_id,
                    details={
                        "strategy": decision.plan.template.strategy.value,
                        "reason": decision.change_reason,
                        "value": decision.plan.value,
                    },
                )
            if decision.all_plans_predicted_fatal:
                _event(
                    self.world,
                    tick,
                    self.world.time,
                    "planning",
                    "all_plans_predicted_fatal",
                    agent_id=agent.id,
                    target_id=decision.plan.template.target_id,
                    details={"selected_value": decision.plan.value},
                )
            if decision.planning_budget_override:
                _event(
                    self.world,
                    tick,
                    self.world.time,
                    "planning",
                    "planning_budget_override",
                    agent_id=agent.id,
                    details={
                        "nominal_budget": decision.planning_budget,
                        "mandatory_candidates": decision.mandatory_candidates,
                    },
                )
            _install_commitment(agent, decision, tick, self.config)
            decisions[agent.id] = decision
        return decisions

    def _movement_phase(self, decisions: dict[str, PlanningDecision], tick: int) -> dict[str, TraversalResult]:
        results: dict[str, TraversalResult] = {}
        dt = self.config.playback.dt
        for agent_id, decision in sorted(decisions.items()):
            agent = self.world.agents[agent_id]
            result = traverse_route(decision.plan.route, agent.body, dt, self.config.physiology)
            agent.position = result.position
            agent.set_body(result.body)
            results[agent_id] = result
            route_prediction = _prediction(agent, agent.commitment.episode_id if agent.commitment else None, "route_arrival")
            if result.reached_route_end and route_prediction is not None and route_prediction.status == "pending":
                route_prediction.status = "resolved"
                route_prediction.actual = True
                route_prediction.resolved_tick = tick
            if result.died:
                self.accumulator.failed_journeys[agent.id] += 1
                death_time = self.world.time + (result.death_offset if result.death_offset is not None else dt)
                _kill_agent(agent, self.world, tick, death_time, "physical_depletion")
            elif result.moving_time > 0:
                _event(
                    self.world,
                    tick,
                    self.world.time + dt,
                    "movement",
                    "movement",
                    agent_id=agent.id,
                    target_id=agent.target_id,
                    position=agent.position,
                    details={
                        "moving_time": result.moving_time,
                        "movement_energy": result.movement_energy,
                        "route_kind": decision.plan.route_kind,
                    },
                )
        return results

    def _combat_phase(self, decisions: dict[str, PlanningDecision], tick: int) -> set[str]:
        requests: list[AttackRequest] = []
        dt = self.config.playback.dt
        for agent_id, decision in sorted(decisions.items()):
            agent = self.world.agents[agent_id]
            if not agent.alive or decision.plan.template.strategy is not Strategy.HUNT:
                continue
            target_id = decision.plan.template.target_id
            target = self.world.agents.get(target_id or "")
            intended = decision.plan.route.travel_time <= dt + 1e-9
            if not intended:
                continue
            _event(
                self.world,
                tick,
                self.world.time + dt,
                "combat",
                "attack",
                agent_id=agent.id,
                target_id=target_id,
                position=agent.position,
                details={"attacker_position": list(agent.position)},
            )
            if target is None or not target.alive or distance(agent.position, target.position) > self.config.combat.engagement_distance:
                self.accumulator.failed_journeys[agent.id] += 1
                resolve_episode_predictions(
                    agent,
                    agent.commitment.episode_id if agent.commitment else None,
                    {"route_arrival": False},
                    tick,
                )
                complete_episode(agent, agent.commitment.episode_id if agent.commitment else None, tick, False)
                agent.commitment = None
                _event(
                    self.world,
                    tick,
                    self.world.time + dt,
                    "combat",
                    "attack_out_of_range",
                    agent_id=agent.id,
                    target_id=target_id,
                    position=agent.position,
                )
                continue
            requests.append(AttackRequest(agent.id, target.id))
        pairs, deferred = match_attacks(requests, tick, self.config)
        for request in deferred:
            _event(
                self.world,
                tick,
                self.world.time + dt,
                "combat",
                "attack_deferred",
                agent_id=request.attacker_id,
                target_id=request.defender_id,
            )
        matched: set[str] = set()
        for pair in pairs:
            if not self.world.agents[pair.first_id].alive or not self.world.agents[pair.second_id].alive:
                continue
            result = resolve_duel(pair, self.world.agents, tick, self.config)
            matched.update((pair.first_id, pair.second_id))
            winner = self.world.agents[result.winner_id]
            loser = self.world.agents[result.loser_id]
            winner.kills += 1
            for participant, opponent, won in ((winner, loser, True), (loser, winner, False)):
                deliberate_request = (
                    participant.id in pair.requesters
                    and participant.strategy is Strategy.HUNT
                    and participant.commitment is not None
                    and participant.commitment.target_id == opponent.id
                )
                if deliberate_request:
                    resolve_episode_predictions(
                        participant,
                        participant.commitment.episode_id if participant.commitment else None,
                        {"hunt_win": won, "route_arrival": True},
                        tick,
                    )
                    complete_episode(participant, participant.commitment.episode_id if participant.commitment else None, tick, won)
                    participant.commitment = None
            _event(
                self.world,
                tick,
                self.world.time + dt,
                "combat",
                "duel",
                agent_id=winner.id,
                target_id=loser.id,
                position=winner.position,
                details={
                    "winner": winner.id,
                    "loser": loser.id,
                    "first_win_probability": result.first_win_probability,
                    "draw": result.draw,
                    "requesters": list(pair.requesters),
                },
            )
            _kill_agent(loser, self.world, tick, self.world.time + dt, "combat")
        return matched

    def _inspection_and_requests(
        self,
        decisions: dict[str, PlanningDecision],
        matched: set[str],
        tick: int,
    ) -> dict[str, list[tuple[AgentState, float, float]]]:
        requests: dict[str, list[tuple[AgentState, float, float]]] = defaultdict(list)
        dt = self.config.playback.dt
        for agent_id, decision in sorted(decisions.items()):
            agent = self.world.agents[agent_id]
            if not agent.alive or agent_id in matched or agent.commitment is None:
                continue
            strategy = agent.commitment.strategy
            target_id = agent.commitment.target_id
            resource = self.world.resources.get(target_id or "")
            if strategy is Strategy.EXPLORE:
                if distance(agent.position, agent.commitment.target_position) <= self.config.perception.adjacency_distance:
                    complete_episode(agent, agent.commitment.episode_id, tick, True)
                    agent.commitment = None
                continue
            if strategy is Strategy.FLEE:
                continue
            if strategy is Strategy.HUNT or resource is None:
                continue
            if distance(agent.position, resource.position) > self.config.perception.adjacency_distance:
                continue
            pre_stock = resource.stock
            if strategy is Strategy.WAIT and agent.commitment.wait_seconds > 0:
                interaction_time = self.world.time + dt
                if agent.commitment.wait_until_time is None:
                    agent.commitment.wait_until_time = interaction_time + agent.commitment.wait_seconds
                    continue
                if interaction_time + 1e-9 < agent.commitment.wait_until_time:
                    continue
            if strategy is Strategy.INSPECT:
                belief_before = agent.resource_beliefs[resource.id]
                was_identified = belief_before.identified
                # The exact observation precedes the tick-end regeneration.
                # Timestamping it at the interval start lets the known one-tick
                # regeneration enter the next decision's predictor.
                observe_resource_exact(agent, resource, self.world.time)
                resolve_episode_predictions(
                    agent,
                    agent.commitment.episode_id,
                    {
                        "stock_sufficient": pre_stock >= self.config.resources.max_request,
                        "resource_type": resource.type.value,
                        "route_arrival": True,
                    },
                    tick,
                )
                complete_episode(agent, agent.commitment.episode_id, tick, not was_identified)
                self.accumulator.inspections[agent.id] += 1
                _event(
                    self.world,
                    tick,
                    self.world.time + dt,
                    "interaction",
                    "inspection",
                    agent_id=agent.id,
                    resource_id=resource.id,
                    position=agent.position,
                    details={"type": resource.type.value, "stock": pre_stock},
                )
                agent.commitment = None
                continue
            if strategy in {Strategy.FORAGE, Strategy.WAIT}:
                observe_resource_exact(agent, resource, self.world.time)
                resolve_episode_predictions(
                    agent,
                    agent.commitment.episode_id,
                    {"stock_sufficient": pre_stock >= self.config.resources.max_request, "route_arrival": True},
                    tick,
                )
                requests[resource.id].append((agent, self.config.resources.max_request, pre_stock))
        return requests

    def _consumption_phase(self, requests: dict[str, list[tuple[AgentState, float, float]]], tick: int) -> None:
        dt = self.config.playback.dt
        pending_deaths: list[AgentState] = []
        for resource_id, items in sorted(requests.items()):
            resource = self.world.resources[resource_id]
            total_requested = sum(request for _agent, request, _stock in items)
            scale = min(1.0, resource.stock / total_requested) if total_requested > 0 else 0.0
            allocations = [(agent, request * scale, observed_stock) for agent, request, observed_stock in items]
            resource.stock -= sum(quantity for _agent, quantity, _stock in allocations)
            resource.clamp_stock()
            for agent, quantity, observed_stock in allocations:
                before = agent.body
                uncapped = tuple(value + quantity * effect for value, effect in zip(before, resource.effect_per_unit))
                after = tuple(max(0.0, min(1.0, value)) for value in uncapped)
                excess = tuple(max(0.0, uncapped[index] - after[index]) for index in range(3))
                agent.set_body(after)  # type: ignore[arg-type]
                belief = agent.resource_beliefs[resource.id]
                # Being adjacent during the resolved interaction legitimately
                # reveals the post-allocation stock, including visible
                # contention. Remote consumption remains hidden.
                belief.last_observed_stock = resource.stock
                belief.last_observed_time = self.world.time
                belief.estimated_stock = belief.last_observed_stock
                belief.confidence = 1.0
                need_index = {Need.ENERGY: 0, Need.WATER: 1, Need.INTEGRITY: 2}[agent.commitment.need]
                helped = quantity > 0 and after[need_index] > before[need_index]
                died = any(value <= 0.0 for value in after)
                complete_episode(agent, agent.commitment.episode_id, tick, helped and not died)
                self.accumulator.resource_quantity[agent.id] += quantity
                _event(
                    self.world,
                    tick,
                    self.world.time + dt,
                    "interaction",
                    "consumption" if quantity > 0 else "failed_consumption",
                    agent_id=agent.id,
                    resource_id=resource.id,
                    position=agent.position,
                    details={
                        "requested": self.config.resources.max_request,
                        "allocated": quantity,
                        "observed_stock": observed_stock,
                        "body_before": list(before),
                        "body_after": list(after),
                        "excess_discarded": list(excess),
                        "resource_type": resource.type.value,
                    },
                )
                agent.commitment = None
                if died:
                    pending_deaths.append(agent)
        for agent in pending_deaths:
            _kill_agent(agent, self.world, tick, self.world.time + dt, "harmful_consumption")

    def _regeneration_phase(self, tick: int) -> None:
        amount_scale = self.config.playback.dt
        for resource in self.world.resources.values():
            if resource.is_corpse or resource.stock >= resource.capacity:
                continue
            resource.stock = min(resource.capacity, resource.stock + resource.regeneration_rate * amount_scale)

    def _log_tick(
        self,
        tick: int,
        observations: dict[str, PerceptionResult],
        cognitive: dict[str, dict[str, Any]],
        drives: dict[str, DriveState],
        decisions: dict[str, PlanningDecision],
    ) -> None:
        for agent_id, agent in sorted(self.world.agents.items()):
            if agent_id not in drives:
                continue
            decision = decisions[agent_id]
            observation = observations[agent_id]
            compact = self.analysis_telemetry[agent_id]
            compact["tick"].append(tick)
            compact["time"].append(self.world.time + self.config.playback.dt)
            compact["alive"].append(agent.alive)
            compact["energy"].append(agent.energy)
            compact["water"].append(agent.water)
            compact["integrity"].append(agent.integrity)
            compact["position_x"].append(agent.position[0])
            compact["position_y"].append(agent.position[1])
            compact["ruling_need"].append(agent.ruling_need.value)
            compact["strategy"].append(agent.strategy.value if agent.strategy else None)
            compact["target_id"].append(agent.target_id)
            compact["certainty"].append(agent.certainty)
            compact["securing_scan"].append(observation.securing_scan)
            compact["selected_plan_value"].append(decision.plan.value)
            compact["selected_death_probability"].append(decision.plan.death_probability)
            compact["literal_emotion"].append(agent.literal_emotion)
            compact["dominant_emotion"].append(agent.dominant_emotion)
            for group_name, values in (
                ("raw_modulators", agent.raw_modulators),
                ("control_modulators", asdict(agent.controls)),
                ("classifier_inputs", agent.classifier_inputs),
                ("emotion_scores", agent.emotion_scores),
            ):
                columns = compact[group_name]
                for name, value in values.items():
                    columns.setdefault(name, []).append(value)
            if self.config.trace_level != "full":
                continue
            row = {
                "tick": tick,
                "time": self.world.time + self.config.playback.dt,
                "alive": agent.alive,
                "position": list(agent.position),
                "energy": agent.energy,
                "water": agent.water,
                "integrity": agent.integrity,
                "urges": {need.value: value for need, value in drives[agent_id].urges.items()},
                "urgency": {need.value: value for need, value in drives[agent_id].urgency.items()},
                "motive_strength": {need.value: value for need, value in drives[agent_id].strength.items()},
                "ruling_need": agent.ruling_need.value,
                "strategy": agent.strategy.value if agent.strategy else None,
                "target_id": agent.target_id,
                "kills": agent.kills,
                "raw_modulators": dict(agent.raw_modulators),
                "control_modulators": asdict(agent.controls),
                "classifier_inputs": dict(agent.classifier_inputs),
                "emotion_scores": dict(agent.emotion_scores),
                "literal_emotion": agent.literal_emotion,
                "dominant_emotion": agent.dominant_emotion,
                "certainty": agent.certainty,
                "perception": {
                    "normal_radius": observation.normal_radius,
                    "securing_scan": observation.securing_scan,
                    "resource_ids": [item.id for item in observation.resources],
                    "agent_ids": [item.id for item in observation.agents],
                },
                "cognitive_events": cognitive[agent_id],
                "decision": {
                    "value": decision.plan.value,
                    "route_kind": decision.plan.route_kind,
                    "route": [list(point) for point in decision.plan.route.waypoints],
                    "route_time": decision.plan.route.travel_time,
                    "death_probability": decision.plan.death_probability,
                    "reason": decision.change_reason,
                    "infeasibility_override": decision.infeasibility_override,
                    "all_plans_predicted_fatal": decision.all_plans_predicted_fatal,
                    "planning_budget": decision.planning_budget,
                    "mandatory_candidates": decision.mandatory_candidates,
                    "planning_budget_override": decision.planning_budget_override,
                    "considered": decision.considered,
                },
            }
            self.traces[agent_id].append(row)

    def _log_world_frame(self, tick: int) -> None:
        if self.config.trace_level != "full":
            return
        self.world_frames.append({
            "tick": tick,
            "time": self.world.time + self.config.playback.dt,
            "agents": {
                agent.id: {
                    "position": list(agent.position),
                    "alive": agent.alive,
                    "body": list(agent.body),
                    "ruling_need": agent.ruling_need.value,
                    "strategy": agent.strategy.value if agent.strategy else None,
                    "target_id": agent.target_id,
                    "emotion": agent.dominant_emotion,
                    "kills": agent.kills,
                }
                for agent in sorted(self.world.agents.values(), key=lambda item: item.id)
            },
            "resources": {
                resource.id: {
                    "position": list(resource.position),
                    "type": resource.type.value,
                    "stock": resource.stock,
                    "capacity": resource.capacity,
                    "is_corpse": resource.is_corpse,
                }
                for resource in sorted(self.world.resources.values(), key=lambda item: item.id)
            },
        })

    def step(self) -> None:
        tick = self.world.tick + 1
        prior_events = list(self.world.events_since_snapshot)
        self.world.events_since_snapshot = []
        if self.world.living_agents:
            observations = self._perception_phase(prior_events)
            drives, cognitive = self._cognitive_phase()
            decisions = self._planning_phase(drives, tick)
            for agent_id, decision in decisions.items():
                self.accumulator.strategy_ticks[agent_id][decision.plan.template.strategy.value] += 1
                self.accumulator.motive_ticks[agent_id][self.world.agents[agent_id].ruling_need.value] += 1
                self.accumulator.emotion_ticks[agent_id][self.world.agents[agent_id].dominant_emotion] += 1
            self._movement_phase(decisions, tick)
            matched = self._combat_phase(decisions, tick) if self.config.combat_enabled else set()
            requests = self._inspection_and_requests(decisions, matched, tick)
            self._consumption_phase(requests, tick)
            self._regeneration_phase(tick)
            self._log_tick(tick, observations, cognitive, drives, decisions)
            for agent in self.world.agents.values():
                if agent.id in decisions:
                    agent.previous_controls = agent.controls
        else:
            self._regeneration_phase(tick)
        self._log_world_frame(tick)
        self.world.tick = tick
        self.world.time += self.config.playback.dt
        if not self.world.living_agents and self.world.extinct_at is None:
            self.world.extinct_at = max(
                agent.death_time or 0.0 for agent in self.world.agents.values()
            )

    def run(self) -> SimulationResult:
        while self.world.tick < self.config.playback.ticks:
            self.step()
        return self.result()

    def result(self) -> SimulationResult:
        horizon = self.config.playback.ticks * self.config.playback.dt
        agent_summaries = {
            agent.id: {
                "alive": agent.alive,
                "death_time": agent.death_time,
                "death_cause": agent.death_cause,
                "restricted_survival_time": min(agent.death_time if agent.death_time is not None else horizon, horizon),
                "final_body": list(agent.body),
                "kills": agent.kills,
                "strategy_ticks": dict(self.accumulator.strategy_ticks[agent.id]),
                "motive_ticks": dict(self.accumulator.motive_ticks[agent.id]),
                "emotion_ticks": dict(self.accumulator.emotion_ticks[agent.id]),
                "scans": self.accumulator.scans[agent.id],
                "inspections": self.accumulator.inspections[agent.id],
                "resource_quantity": self.accumulator.resource_quantity[agent.id],
                "motive_switches": self.accumulator.motive_switches[agent.id],
                "strategy_switches": self.accumulator.strategy_switches[agent.id],
                "failed_journeys": self.accumulator.failed_journeys[agent.id],
                "clipping": dict(self.accumulator.clipping[agent.id]),
                "competence_by_strategy": {
                    strategy.value: value for strategy, value in agent.competence_by_strategy.items()
                },
                "observed_event_ledger": list(agent.observed_event_ledger),
                "predictions": [asdict(item) for item in agent.pending_predictions.values()],
                "episodes": [asdict(item) for item in agent.episodes.values()],
            }
            for agent in sorted(self.world.agents.values(), key=lambda item: item.id)
        }
        y = sum(item["restricted_survival_time"] for item in agent_summaries.values()) / len(agent_summaries)
        event_counts = Counter(event.event_type for event in self.world.events)
        strategy_totals: Counter = Counter()
        emotion_totals: Counter = Counter()
        clipping_totals: Counter = Counter()
        death_causes: Counter = Counter()
        prediction_expected = 0
        prediction_unexpected = 0
        for item in agent_summaries.values():
            strategy_totals.update(item["strategy_ticks"])
            emotion_totals.update(item["emotion_ticks"])
            clipping_totals.update(item["clipping"])
            if item["death_cause"] is not None:
                death_causes[item["death_cause"]] += 1
            for record in item["observed_event_ledger"]:
                if record["kind"] == "prediction":
                    prediction_expected += int(record["agreement"])
                    prediction_unexpected += int(not record["agreement"])
        classified_ticks = sum(emotion_totals.values())
        summary = {
            "restricted_mean_survival_time": y,
            "survivors": sum(item["alive"] for item in agent_summaries.values()),
            "survivor_fraction": sum(item["alive"] for item in agent_summaries.values()) / len(agent_summaries),
            "extinct_at": self.world.extinct_at,
            "event_counts": dict(event_counts),
            "secondary_metrics": {
                "resource_quantity_acquired": sum(item["resource_quantity"] for item in agent_summaries.values()),
                "failed_journeys": sum(item["failed_journeys"] for item in agent_summaries.values()),
                "inspections": sum(item["inspections"] for item in agent_summaries.values()),
                "securing_scans": sum(item["scans"] for item in agent_summaries.values()),
                "motive_switches": sum(item["motive_switches"] for item in agent_summaries.values()),
                "strategy_switches": sum(item["strategy_switches"] for item in agent_summaries.values()),
                "attack_requests": event_counts["attack"],
                "resolved_duels": event_counts["duel"],
                "deaths_by_cause": dict(death_causes),
                "strategy_ticks": dict(strategy_totals),
                "emotion_ticks": dict(emotion_totals),
                "unclassified_fraction": (
                    emotion_totals["Unclassified"] / classified_ticks if classified_ticks else 0.0
                ),
                "clipping_counts": dict(clipping_totals),
                "prediction_expected": prediction_expected,
                "prediction_unexpected": prediction_unexpected,
                "prediction_error_rate": (
                    prediction_unexpected / (prediction_expected + prediction_unexpected)
                    if prediction_expected + prediction_unexpected
                    else None
                ),
            },
            "ticks_completed": self.world.tick,
            "simulated_seconds": self.world.time,
        }
        return SimulationResult(
            config=self.config.to_dict(),
            configuration_digest=self.config.digest(),
            provenance=self.provenance,
            initial_state=self.initial_artifact,
            agents=agent_summaries,
            events=[_serialize_event(event) for event in self.world.events],
            analysis_telemetry=self.analysis_telemetry,
            traces=self.traces if self.config.trace_level == "full" else {},
            world_frames=self.world_frames if self.config.trace_level == "full" else [],
            summary=summary,
        )


def run_simulation(config: ExperimentConfig) -> SimulationResult:
    return SimulationEngine(config).run()

"""Belief-only candidate planning and mutually exclusive outcome evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .beliefs import active_threats, expected_effect, mean_consulted_confidence, type_entropy
from .combat import win_probability
from .config import ExperimentConfig
from .emotions import DriveState
from .rng import choice
from .state import (
    AgentBelief,
    AgentState,
    BodyVector,
    Commitment,
    Need,
    Position,
    RESOURCE_EFFECTS,
    ResourceBelief,
    ResourceType,
    Strategy,
    body_deficit,
    distance,
)
from .terrain import EPSILON, Route, TerrainMap, traverse_route


NEED_INDEX = {Need.ENERGY: 0, Need.WATER: 1, Need.INTEGRITY: 2}
NEED_ORDER = (Need.WATER, Need.ENERGY, Need.INTEGRITY)


@dataclass
class CandidateTemplate:
    strategy: Strategy
    need: Need
    target_id: str | None
    target_position: Position
    objective: str
    preliminary_rank: float
    wait_seconds: int = 0
    wait_in_progress: bool = False
    resource_belief: ResourceBelief | None = None
    agent_belief: AgentBelief | None = None
    mandatory_reason: str | None = None


@dataclass
class OutcomeLeaf:
    probability: float
    body: BodyVector
    died: bool
    death_time: float | None
    label: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatedPlan:
    template: CandidateTemplate
    route: Route
    route_kind: str
    action_time: float | None
    outcomes: list[OutcomeLeaf]
    expected_deficit: float
    death_probability: float
    information_gain: float
    mean_uncertainty: float
    certainty: float
    value: float
    physiologically_feasible: bool

    @property
    def identity(self) -> tuple:
        return (
            self.template.strategy.value,
            self.template.target_id,
            self.template.need.value,
            self.template.wait_seconds,
            self.route_kind,
        )


@dataclass
class PlanningDecision:
    plan: EvaluatedPlan
    ruling_need_before: Need
    ruling_need_after: Need
    motive_switched: bool
    infeasibility_override: bool
    commitment_changed: bool
    change_reason: str
    all_plans_predicted_fatal: bool
    planning_budget: int
    mandatory_candidates: int
    planning_budget_override: bool
    considered: list[dict[str, Any]]


def _action_time(route: Route, wait_seconds: int = 0, wait_in_progress: bool = False) -> float:
    # Movement and one interaction share a tick, but every interaction resolves
    # after that tick's passive depletion. Exact-integer arrivals act that tick.
    arrival_tick = max(1, int(math.ceil(route.travel_time - EPSILON)))
    if wait_in_progress and route.travel_time <= EPSILON:
        return float(max(1, wait_seconds))
    return float(arrival_tick + wait_seconds)


def _apply_effect(body: BodyVector, effect: BodyVector, quantity: float, competence: float) -> tuple[BodyVector, bool]:
    values = []
    for value, per_unit in zip(body, effect):
        adjusted = per_unit * competence if per_unit > 0 else per_unit
        values.append(min(1.0, value + quantity * adjusted))
    died = any(value <= 0.0 for value in values)
    result = tuple(max(0.0, value) for value in values)  # type: ignore[assignment]
    return result, died


def _advance(route: Route, body: BodyVector, duration: float, config: ExperimentConfig):
    return traverse_route(route, body, max(0.0, duration), config.physiology)


def _idle(position: Position, body: BodyVector, duration: float, config: ExperimentConfig):
    return traverse_route(Route(position, position, "time", ()), body, max(0.0, duration), config.physiology)


def _position_on_route(route: Route, at_time: float) -> Position:
    if not route.segments:
        return route.start
    remaining = max(0.0, at_time)
    for segment in route.segments:
        if remaining <= segment.travel_time + EPSILON:
            fraction = min(1.0, remaining / segment.travel_time)
            return (
                segment.start[0] + (segment.end[0] - segment.start[0]) * fraction,
                segment.start[1] + (segment.end[1] - segment.start[1]) * fraction,
            )
        remaining -= segment.travel_time
    return route.segments[-1].end


def _estimate_threat_intercept(
    agent: AgentState,
    route: Route,
    threats: list[AgentBelief],
    terrain: TerrainMap,
    config: ExperimentConfig,
) -> tuple[float, AgentBelief] | None:
    earliest: tuple[float, AgentBelief] | None = None
    for threat in threats:
        for second in range(1, config.planner.horizon_seconds + 1):
            predicted_position = _position_on_route(route, float(second))
            pursuit = terrain.route(
                threat.last_position,
                predicted_position,
                "time",
                stop_distance=config.combat.engagement_distance,
            )
            if pursuit.travel_time <= second + EPSILON:
                candidate = (float(second), threat)
                if earliest is None or candidate[0] < earliest[0] or (
                    candidate[0] == earliest[0] and candidate[1].agent_id < earliest[1].agent_id
                ):
                    earliest = candidate
                break
    return earliest


def _availability_probability(
    belief: ResourceBelief,
    requested: float,
    config: ExperimentConfig,
    elapsed_seconds: float,
) -> float:
    estimated = min(config.resources.capacity, belief.estimated_stock + (
        0.0
        if belief.is_corpse
        else config.resources.regeneration_rate * config.regeneration_multiplier * max(0.0, elapsed_seconds)
    ))
    if belief.last_observed_stock is not None and belief.confidence >= 1.0 - 1e-12 and estimated >= requested:
        return 1.0
    factor = estimated / config.resources.capacity
    return min(1.0, max(0.0, factor * (0.5 + 0.5 * belief.confidence)))


def _resource_scenarios(plan: CandidateTemplate, route: Route, config: ExperimentConfig) -> list[dict[str, Any]]:
    belief = plan.resource_belief
    if belief is None or plan.strategy in {Strategy.EXPLORE, Strategy.FLEE, Strategy.HUNT}:
        return [{"probability": 1.0, "type": None, "available": False, "effect": (0.0, 0.0, 0.0)}]
    type_distribution = belief.type_distribution
    primary_time = _action_time(route, plan.wait_seconds, plan.wait_in_progress)
    availability_time = max(0.0, primary_time - 1.0)
    if plan.strategy is Strategy.INSPECT:
        availability_time += 1.0
    availability = _availability_probability(
        belief,
        config.resources.max_request,
        config,
        availability_time,
    )
    scenarios = []
    for resource_type, type_probability in type_distribution.items():
        effect = belief.observed_effect if len(type_distribution) == 1 and belief.observed_effect is not None else RESOURCE_EFFECTS.get(
            resource_type,
            (0.4, 0.4, 0.0) if resource_type is ResourceType.CORPSE else (0.0, 0.0, 0.0),
        )
        useful = effect[NEED_INDEX[plan.need]] > 0
        consumes = plan.strategy in {Strategy.FORAGE, Strategy.WAIT} or (plan.strategy is Strategy.INSPECT and useful)
        if not consumes:
            scenarios.append({
                "probability": type_probability,
                "type": resource_type,
                "available": False,
                "effect": effect,
            })
            continue
        scenarios.append({
            "probability": type_probability * availability,
            "type": resource_type,
            "available": True,
            "effect": effect,
        })
        if availability < 1.0:
            scenarios.append({
                "probability": type_probability * (1.0 - availability),
                "type": resource_type,
                "available": False,
                "effect": effect,
            })
    return [scenario for scenario in scenarios if scenario["probability"] > 0]


def _complete_alive_leaf(
    probability: float,
    body: BodyVector,
    position: Position,
    elapsed: float,
    label: str,
    details: dict[str, Any],
    config: ExperimentConfig,
) -> OutcomeLeaf:
    remaining = config.planner.horizon_seconds - elapsed
    if remaining <= 0:
        return OutcomeLeaf(probability, body, False, None, label, details)
    result = _idle(position, body, remaining, config)
    return OutcomeLeaf(
        probability,
        result.body,
        result.died,
        elapsed + result.death_offset if result.died and result.death_offset is not None else None,
        label + (":physiology-death" if result.died else ":alive"),
        details,
    )


def _evaluate_scenario(
    agent: AgentState,
    template: CandidateTemplate,
    route: Route,
    scenario: dict[str, Any],
    threats: list[AgentBelief],
    terrain: TerrainMap,
    config: ExperimentConfig,
    advance_from_start,
) -> list[OutcomeLeaf]:
    horizon = float(config.planner.horizon_seconds)
    primary_time = _action_time(route, template.wait_seconds, template.wait_in_progress)
    consume_time: float | None = None
    if template.strategy in {Strategy.FORAGE, Strategy.WAIT} and scenario["available"]:
        consume_time = primary_time
    elif template.strategy is Strategy.INSPECT and scenario["available"]:
        consume_time = primary_time + 1.0

    planned_duel: tuple[float, AgentBelief] | None = None
    if template.strategy is Strategy.HUNT and template.agent_belief is not None:
        planned_duel = (primary_time, template.agent_belief)
    threat_duel = _estimate_threat_intercept(agent, route, threats, terrain, config) if threats else None
    duel_candidates = [item for item in (planned_duel, threat_duel) if item is not None and item[0] <= horizon]
    duel = min(duel_candidates, key=lambda item: (item[0], item[1].agent_id)) if duel_candidates else None
    deliberate_duel_occurs = bool(
        planned_duel is not None
        and duel is not None
        and duel[1].agent_id == planned_duel[1].agent_id
        and duel[0] <= planned_duel[0]
    )

    probability = float(scenario["probability"])
    details = {
        "resource_type": scenario["type"].value if scenario["type"] else None,
        "resource_available": scenario["available"],
        "consume_time": consume_time,
        "duel_time": duel[0] if duel else None,
        "duel_agent": duel[1].agent_id if duel else None,
        "deliberate_duel": deliberate_duel_occurs,
    }

    first_event_time = min(value for value in (consume_time, duel[0] if duel else None, horizon) if value is not None)
    base = advance_from_start(first_event_time)
    if base.died:
        return [OutcomeLeaf(probability, base.body, True, base.death_offset, "pre-action-physiology-death", details)]

    effect_before_duel = consume_time is not None and (duel is None or consume_time < duel[0] - EPSILON)
    body_at_duel = base.body
    position_at_duel = base.position
    elapsed_at_duel = first_event_time
    if effect_before_duel:
        body_at_duel, effect_death = _apply_effect(
            base.body,
            scenario["effect"],
            config.resources.max_request,
            agent.competence_by_strategy[template.strategy],
        )
        if effect_death:
            return [OutcomeLeaf(probability, body_at_duel, True, consume_time, "resource-effect-death", details)]
        if duel is not None:
            idle_to_duel = _idle(base.position, body_at_duel, duel[0] - consume_time, config)
            if idle_to_duel.died:
                return [OutcomeLeaf(probability, idle_to_duel.body, True, consume_time + (idle_to_duel.death_offset or 0), "pre-duel-physiology-death", details)]
            body_at_duel = idle_to_duel.body
            position_at_duel = idle_to_duel.position
            elapsed_at_duel = duel[0]

    if duel is not None:
        if not effect_before_duel:
            duel_state = advance_from_start(duel[0])
            if duel_state.died:
                return [OutcomeLeaf(probability, duel_state.body, True, duel_state.death_offset, "pre-duel-physiology-death", details)]
            body_at_duel = duel_state.body
            position_at_duel = duel_state.position
            elapsed_at_duel = duel[0]
        win_p = win_probability(body_at_duel, duel[1].estimated_body, config)
        loser = OutcomeLeaf(probability * (1.0 - win_p), body_at_duel, True, duel[0], "duel-loss", {**details, "estimated_win_probability": win_p})
        winner_body = body_at_duel
        winner_position = position_at_duel
        winner_elapsed = elapsed_at_duel

        if template.strategy is Strategy.HUNT and deliberate_duel_occurs:
            corpse_time = duel[0] + 1.0
            if corpse_time <= horizon:
                wait = _idle(winner_position, winner_body, 1.0, config)
                if wait.died:
                    winner = OutcomeLeaf(probability * win_p, wait.body, True, duel[0] + (wait.death_offset or 0), "post-duel-physiology-death", details)
                    return [loser, winner]
                corpse_effect = (0.8 * duel[1].estimated_body[0], 0.8 * duel[1].estimated_body[1], 0.0)
                winner_body, corpse_death = _apply_effect(
                    wait.body,
                    corpse_effect,
                    config.resources.max_request,
                    agent.competence_by_strategy[Strategy.HUNT],
                )
                winner_position = wait.position
                winner_elapsed = corpse_time
                if corpse_death:
                    winner = OutcomeLeaf(probability * win_p, winner_body, True, corpse_time, "corpse-effect-death", details)
                    return [loser, winner]
        elif consume_time is not None and not effect_before_duel:
            # A threat intercepted before the planned noncombat interaction. A
            # same-tick duel cancels that interaction; the next opportunity is
            # one tick after combat. An earlier duel does not add extra delay.
            resumed_consume_time = max(consume_time, duel[0] + 1.0)
            if template.strategy is Strategy.INSPECT and abs(duel[0] - primary_time) <= EPSILON:
                # Inspection itself was cancelled, so both inspection and its
                # contingent next-tick consumption move one tick later.
                resumed_consume_time = max(resumed_consume_time, consume_time + 1.0)
            details["resumed_consume_time"] = resumed_consume_time
            if resumed_consume_time > horizon:
                winner = _complete_alive_leaf(
                    probability * win_p,
                    winner_body,
                    winner_position,
                    winner_elapsed,
                    "duel-win-no-time-for-resource",
                    {**details, "estimated_win_probability": win_p},
                    config,
                )
                return [loser, winner]
            action_state = advance_from_start(resumed_consume_time)
            if action_state.died:
                winner = OutcomeLeaf(probability * win_p, action_state.body, True, action_state.death_offset, "post-duel-route-death", details)
                return [loser, winner]
            winner_body, effect_death = _apply_effect(
                action_state.body,
                scenario["effect"],
                config.resources.max_request,
                agent.competence_by_strategy[template.strategy],
            )
            winner_position = action_state.position
            winner_elapsed = resumed_consume_time
            if effect_death:
                winner = OutcomeLeaf(probability * win_p, winner_body, True, resumed_consume_time, "resource-effect-death", details)
                return [loser, winner]
        winner = _complete_alive_leaf(
            probability * win_p,
            winner_body,
            winner_position,
            winner_elapsed,
            "duel-win",
            {**details, "estimated_win_probability": win_p},
            config,
        )
        return [loser, winner]

    if consume_time is not None and consume_time <= horizon:
        action_state = advance_from_start(consume_time)
        if action_state.died:
            return [OutcomeLeaf(probability, action_state.body, True, action_state.death_offset, "pre-consumption-death", details)]
        result_body, effect_death = _apply_effect(
            action_state.body,
            scenario["effect"],
            config.resources.max_request,
            agent.competence_by_strategy[template.strategy],
        )
        if effect_death:
            return [OutcomeLeaf(probability, result_body, True, consume_time, "resource-effect-death", details)]
        return [_complete_alive_leaf(probability, result_body, action_state.position, consume_time, "resource", details, config)]

    final = advance_from_start(horizon)
    return [OutcomeLeaf(probability, final.body, final.died, final.death_offset, "no-primary-benefit", details)]


def _information_and_uncertainty(template: CandidateTemplate) -> tuple[float, float]:
    if template.resource_belief is not None:
        belief = template.resource_belief
        type_uncertainty = type_entropy(belief)
        stock_uncertainty = 1.0 - belief.confidence
        uncertainty = 0.5 * (type_uncertainty + stock_uncertainty)
        observes_exact = template.strategy in {Strategy.FORAGE, Strategy.INSPECT, Strategy.WAIT}
        return (uncertainty if observes_exact else 0.0), uncertainty
    if template.agent_belief is not None:
        return 0.0, 1.0 - template.agent_belief.confidence
    return 0.0, 0.0


def evaluate_template(
    agent: AgentState,
    template: CandidateTemplate,
    route: Route,
    route_kind: str,
    terrain: TerrainMap,
    config: ExperimentConfig,
) -> EvaluatedPlan:
    threats = active_threats(agent, config_time(agent), config)
    scenarios = _resource_scenarios(template, route, config)
    advance_cache: dict[float, Any] = {}
    def advance_from_start(duration: float):
        if duration not in advance_cache:
            advance_cache[duration] = _advance(route, agent.body, duration, config)
        return advance_cache[duration]
    outcomes = [
        outcome
        for scenario in scenarios
        for outcome in _evaluate_scenario(
            agent,
            template,
            route,
            scenario,
            threats,
            terrain,
            config,
            advance_from_start,
        )
    ]
    total_probability = sum(outcome.probability for outcome in outcomes)
    if abs(total_probability - 1.0) > 1e-8:
        raise AssertionError(f"Outcome probabilities sum to {total_probability} for {template}")
    expected_deficit = sum(
        outcome.probability * (1.0 if outcome.died else body_deficit(outcome.body))
        for outcome in outcomes
    )
    death_probability = sum(outcome.probability for outcome in outcomes if outcome.died)
    information, uncertainty = _information_and_uncertainty(template)
    certainty = mean_consulted_confidence(
        [template.resource_belief] if template.resource_belief is not None else [],
        [template.agent_belief] if template.agent_belief is not None else [],
    )
    value = (
        body_deficit(agent.body)
        - expected_deficit
        - config.planner.death_penalty * death_probability
        + config.planner.information_weight * information
        - config.planner.uncertainty_weight * uncertainty
    )
    action_time = None if template.strategy in {Strategy.EXPLORE, Strategy.FLEE} else _action_time(
        route, template.wait_seconds, template.wait_in_progress
    )
    feasibility_horizon = action_time if action_time is not None else config.planner.horizon_seconds
    feasible = not all(
        outcome.died
        and (outcome.death_time if outcome.death_time is not None else math.inf) <= feasibility_horizon
        for outcome in outcomes
    )
    return EvaluatedPlan(
        template,
        route,
        route_kind,
        action_time,
        outcomes,
        expected_deficit,
        death_probability,
        information,
        uncertainty,
        certainty,
        value,
        feasible,
    )


def config_time(agent: AgentState) -> float:
    # The engine installs the snapshot time immediately before planning. Keeping
    # it outside beliefs prevents controller access to authoritative world state.
    return float(getattr(agent, "_decision_time", 0.0))


def _preliminary_resource_templates(agent: AgentState, terrain: TerrainMap, config: ExperimentConfig) -> list[CandidateTemplate]:
    templates: list[CandidateTemplate] = []
    for belief in sorted(agent.resource_beliefs.values(), key=lambda item: item.resource_id):
        travel_time = terrain.estimate_route_cost(
            agent.position,
            belief.position,
            "time",
            config.perception.adjacency_distance,
        )
        time = max(1.0, travel_time)
        effect = expected_effect(belief)
        availability = _availability_probability(
            belief,
            config.resources.max_request,
            config,
            max(0.0, math.ceil(travel_time - EPSILON) - 1.0),
        )
        if belief.identified:
            for need in Need:
                benefit = max(0.0, effect[NEED_INDEX[need]]) * availability
                if benefit > 0:
                    templates.append(CandidateTemplate(
                        Strategy.FORAGE,
                        need,
                        belief.resource_id,
                        belief.position,
                        f"consume for {need.value}",
                        benefit / time,
                        resource_belief=belief,
                    ))
            if (
                not belief.is_corpse
                and belief.last_observed_stock is not None
                and belief.estimated_stock < config.resources.max_request
            ):
                for wait in config.planner.wait_durations:
                    for need in Need:
                        if effect[NEED_INDEX[need]] > 0:
                            templates.append(CandidateTemplate(
                                Strategy.WAIT,
                                need,
                                belief.resource_id,
                                belief.position,
                                f"wait {wait}s then consume for {need.value}",
                                max(0.0, effect[NEED_INDEX[need]]) / (time + wait),
                                wait_seconds=wait,
                                resource_belief=belief,
                            ))
            if belief.last_observed_time is None or belief.confidence < 0.25:
                templates.append(CandidateTemplate(
                    Strategy.EXPLORE,
                    agent.ruling_need,
                    belief.resource_id,
                    belief.position,
                    "refresh stale resource presence",
                    (1.0 - belief.confidence) / time,
                    resource_belief=belief,
                ))
        else:
            entropy = type_entropy(belief)
            for need in Need:
                templates.append(CandidateTemplate(
                    Strategy.INSPECT,
                    need,
                    belief.resource_id,
                    belief.position,
                    "identify resource and consume next tick if useful",
                    (entropy + max(0.0, effect[NEED_INDEX[need]])) / time,
                    resource_belief=belief,
                ))
    return templates


def _hunt_templates(agent: AgentState, terrain: TerrainMap, config: ExperimentConfig) -> list[CandidateTemplate]:
    if not config.combat_enabled:
        return []
    templates = []
    for belief in sorted(agent.agent_beliefs.values(), key=lambda item: item.agent_id):
        if not belief.alive:
            continue
        travel_time = terrain.estimate_route_cost(
            agent.position,
            belief.last_position,
            "time",
            config.combat.engagement_distance,
        )
        time = max(1.0, travel_time)
        probability = win_probability(agent.body, belief.estimated_body, config)
        corpse_effect = (0.8 * belief.estimated_body[0], 0.8 * belief.estimated_body[1], 0.0)
        benefit = max(0.0, corpse_effect[NEED_INDEX[agent.ruling_need]]) * probability
        templates.append(CandidateTemplate(
            Strategy.HUNT,
            agent.ruling_need,
            belief.agent_id,
            belief.last_position,
            "pursue, attack, and consume corpse next tick after a win",
            benefit / time,
            agent_belief=belief,
        ))
    return templates


def _flee_template(agent: AgentState, terrain: TerrainMap, config: ExperimentConfig) -> CandidateTemplate | None:
    threats = active_threats(agent, config_time(agent), config)
    if not threats:
        return None
    anchors = terrain._anchors(agent.position)
    candidates: set[tuple[int, int]] = set()
    for anchor in anchors:
        candidates.update(terrain.neighbors(anchor))
    if not candidates:
        return None
    target = max(
        sorted(candidates),
        key=lambda node: min(distance((float(node[0]), float(node[1])), threat.last_position) for threat in threats),
    )
    position = (float(target[0]), float(target[1]))
    separation = min(distance(position, threat.last_position) for threat in threats)
    return CandidateTemplate(
        Strategy.FLEE,
        agent.ruling_need,
        threats[0].agent_id,
        position,
        "increase separation from an observed attacker",
        separation,
        agent_belief=threats[0],
        mandatory_reason="observed threat",
    )


def _template_identity(template: CandidateTemplate) -> tuple:
    return template.strategy, template.target_id, template.need, template.wait_seconds


def _current_template(agent: AgentState, now: float) -> CandidateTemplate | None:
    commitment = agent.commitment
    if commitment is None:
        return None
    resource = agent.resource_beliefs.get(commitment.target_id or "")
    other = agent.agent_beliefs.get(commitment.target_id or "")
    if commitment.target_id and resource is None and other is None and commitment.strategy is not Strategy.FLEE:
        return None
    wait_in_progress = commitment.strategy is Strategy.WAIT and commitment.wait_until_time is not None
    remaining_wait = commitment.wait_seconds
    if wait_in_progress:
        remaining_wait = max(0, int(math.ceil((commitment.wait_until_time or now) - now - EPSILON)))
    return CandidateTemplate(
        commitment.strategy,
        commitment.need,
        commitment.target_id,
        resource.position if resource else (other.last_position if other else commitment.target_position),
        "retain current commitment",
        math.inf,
        wait_seconds=remaining_wait,
        wait_in_progress=wait_in_progress,
        resource_belief=resource,
        agent_belief=other,
        mandatory_reason="current commitment",
    )


def _route_variants(
    agent: AgentState,
    template: CandidateTemplate,
    terrain: TerrainMap,
    config: ExperimentConfig,
    cache: dict[tuple[Position, str, float], Route] | None = None,
) -> list[tuple[str, Route]]:
    stop = 0.0 if template.strategy is Strategy.FLEE else config.perception.adjacency_distance
    route_cache = cache if cache is not None else {}

    def get_route(metric: str) -> Route:
        key = (template.target_position, metric, stop)
        if key not in route_cache:
            route_cache[key] = terrain.route(agent.position, template.target_position, metric, stop)
        return route_cache[key]

    fastest = get_route("time")
    energy = get_route("energy")
    variants = [("fastest", fastest)]
    if energy.waypoints != fastest.waypoints:
        variants.append(("least_energy", energy))
    return variants


def _select_templates(
    agent: AgentState,
    templates: list[CandidateTemplate],
    config: ExperimentConfig,
    now: float,
) -> list[CandidateTemplate]:
    selected: dict[tuple, CandidateTemplate] = {}
    current = _current_template(agent, now)
    if current is not None:
        selected[_template_identity(current)] = current
    for need in Need:
        candidates = [template for template in templates if template.need is need]
        if candidates:
            best = max(candidates, key=lambda item: (item.preliminary_rank, str(_template_identity(item))))
            best.mandatory_reason = best.mandatory_reason or f"best candidate for {need.value}"
            selected[_template_identity(best)] = best
    flee = next((template for template in templates if template.strategy is Strategy.FLEE), None)
    if flee is not None:
        selected[_template_identity(flee)] = flee
    budget = config.planner.planning_budget_floor + math.floor(config.planner.planning_budget_scale * agent.controls.resolution)
    for template in sorted(templates, key=lambda item: (-item.preliminary_rank, str(_template_identity(item)))):
        if len(selected) >= budget:
            break
        selected.setdefault(_template_identity(template), template)
    return list(selected.values())


def choose_plan(agent: AgentState, drives: DriveState, terrain: TerrainMap, config: ExperimentConfig, tick: int, now: float) -> PlanningDecision:
    setattr(agent, "_decision_time", now)
    ruling_before = agent.ruling_need
    strongest = max(NEED_ORDER, key=lambda need: drives.strength[need])
    margin = config.planner.motive_margin_floor + config.planner.motive_margin_scale * agent.controls.selection_threshold
    motive_switched = False
    if drives.strength[strongest] > drives.strength[agent.ruling_need] + margin:
        agent.ruling_need = strongest
        motive_switched = strongest is not ruling_before

    templates = _preliminary_resource_templates(agent, terrain, config)
    templates.extend(_hunt_templates(agent, terrain, config))
    flee = _flee_template(agent, terrain, config)
    if flee is not None:
        templates.append(flee)
    selected_templates = _select_templates(agent, templates, config, now)
    planning_budget = config.planner.planning_budget_floor + math.floor(
        config.planner.planning_budget_scale * agent.controls.resolution
    )
    mandatory_candidates = sum(template.mandatory_reason is not None for template in selected_templates)
    consulted_resources = {
        template.resource_belief.resource_id: template.resource_belief
        for template in selected_templates
        if template.resource_belief is not None
    }
    consulted_agents = {
        template.agent_belief.agent_id: template.agent_belief
        for template in selected_templates
        if template.agent_belief is not None
    }
    decision_certainty = mean_consulted_confidence(
        list(consulted_resources.values()),
        list(consulted_agents.values()),
    )
    route_cache: dict[tuple[Position, str, float], Route] = {}
    evaluated = [
        evaluate_template(agent, template, route, route_kind, terrain, config)
        for template in selected_templates
        for route_kind, route in _route_variants(agent, template, terrain, config, route_cache)
    ]
    if not evaluated:
        raise RuntimeError(f"No candidate plans generated for {agent.id}")

    infeasibility_override = False
    ruling_plans = [plan for plan in evaluated if plan.template.need is agent.ruling_need and plan.physiologically_feasible]
    if not ruling_plans:
        feasible_needs = [
            need for need in NEED_ORDER
            if any(plan.template.need is need and plan.physiologically_feasible for plan in evaluated)
        ]
        if feasible_needs:
            replacement = max(feasible_needs, key=lambda need: drives.strength[need])
            infeasibility_override = replacement is not agent.ruling_need
            agent.ruling_need = replacement

    ranked = sorted(
        evaluated,
        key=lambda plan: (-plan.value, plan.template.strategy.value, plan.template.target_id or "", plan.route_kind),
    )
    best = ranked[0]
    current = next((plan for plan in evaluated if plan.template.mandatory_reason == "current commitment"), None)
    immediate_reason = None
    if current is None and agent.commitment is not None:
        immediate_reason = "invalid target"
    if flee is not None and (current is None or current.template.strategy is not Strategy.FLEE):
        immediate_reason = "new observed attack"
    if current is not None and not current.physiologically_feasible:
        immediate_reason = "newly fatal route"

    target_margin = config.planner.target_margin_floor + config.planner.target_margin_scale * agent.controls.selection_threshold
    if current is not None and immediate_reason is None and best.identity != current.identity and best.value <= current.value + target_margin:
        chosen = current
        changed = False
        reason = "commitment retained by target-switch margin"
    else:
        chosen = best
        changed = (
            False
            if chosen.template.mandatory_reason == "current commitment"
            else agent.commitment is None or (
                agent.commitment.strategy.value,
                agent.commitment.target_id,
                agent.commitment.need.value,
                agent.commitment.wait_seconds,
                agent.commitment.route_kind,
            ) != chosen.identity
        )
        reason = immediate_reason or ("highest plan value" if changed else "commitment remains highest")
    agent.certainty = decision_certainty

    trace = [
        {
            "strategy": plan.template.strategy.value,
            "need": plan.template.need.value,
            "target_id": plan.template.target_id,
            "route_kind": plan.route_kind,
            "route_time": plan.route.travel_time,
            "route_energy": plan.route.movement_energy,
            "action_time": plan.action_time,
            "value": plan.value,
            "death_probability": plan.death_probability,
            "expected_deficit": plan.expected_deficit,
            "information_gain": plan.information_gain,
            "uncertainty": plan.mean_uncertainty,
            "certainty": plan.certainty,
            "feasible": plan.physiologically_feasible,
            "mandatory_reason": plan.template.mandatory_reason,
            "outcome_probability_sum": sum(outcome.probability for outcome in plan.outcomes),
        }
        for plan in ranked
    ]
    return PlanningDecision(
        chosen,
        ruling_before,
        agent.ruling_need,
        motive_switched,
        infeasibility_override,
        changed,
        reason,
        all(plan.death_probability >= 1.0 - 1e-12 for plan in evaluated),
        planning_budget,
        mandatory_candidates,
        mandatory_candidates > planning_budget,
        trace,
    )

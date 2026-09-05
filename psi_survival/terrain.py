"""Seeded terrain, mixed-terrain routes, and continuous traversal."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import bisect
import heapq
import math
from typing import Iterable

from .config import PhysiologyParameters, TerrainParameters
from .rng import stream
from .state import BodyVector, Position, Terrain


EPSILON = 1e-9


@dataclass(frozen=True)
class AtomicSegment:
    start: Position
    end: Position
    terrain: Terrain
    length: float
    travel_time: float
    movement_energy: float


@dataclass(frozen=True)
class Route:
    start: Position
    goal: Position
    metric: str
    segments: tuple[AtomicSegment, ...]

    @cached_property
    def travel_time(self) -> float:
        return sum(segment.travel_time for segment in self.segments)

    @cached_property
    def movement_energy(self) -> float:
        return sum(segment.movement_energy for segment in self.segments)

    @cached_property
    def length(self) -> float:
        return sum(segment.length for segment in self.segments)

    @cached_property
    def waypoints(self) -> tuple[Position, ...]:
        if not self.segments:
            return (self.start,)
        points = [self.segments[0].start]
        for segment in self.segments:
            if not points or segment.end != points[-1]:
                points.append(segment.end)
        return tuple(points)

    @cached_property
    def cumulative_times(self) -> tuple[float, ...]:
        total = 0.0
        values = []
        for segment in self.segments:
            total += segment.travel_time
            values.append(total)
        return tuple(values)

    @cached_property
    def cumulative_movement_energy(self) -> tuple[float, ...]:
        total = 0.0
        values = []
        for segment in self.segments:
            total += segment.movement_energy
            values.append(total)
        return tuple(values)


@dataclass(frozen=True)
class TraversalResult:
    position: Position
    body: BodyVector
    elapsed: float
    moving_time: float
    movement_energy: float
    reached_route_end: bool
    died: bool
    death_offset: float | None


class TerrainMap:
    def __init__(self, cells: tuple[tuple[Terrain, ...], ...], parameters: TerrainParameters):
        self.cells = cells
        self.parameters = parameters
        self.height = len(cells)
        self.width = len(cells[0]) if cells else 0
        self._field_cache: dict[tuple[tuple[int, int], str], tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]]] = {}
        self._pair_cache: dict[tuple[tuple[int, int], tuple[int, int], str], tuple[float, tuple[tuple[int, int], ...]]] = {}
        self._node_segments_cache: dict[
            tuple[tuple[int, int], tuple[int, int], str, float],
            tuple[AtomicSegment, ...],
        ] = {}
        self._static_goals: set[tuple[int, int]] = set()

    def register_static_goals(self, positions: Iterable[Position]) -> None:
        """Mark immutable destinations that benefit from reverse path fields."""
        for position in positions:
            self._static_goals.update(self._anchors(position))

    @classmethod
    def generate(cls, parameters: TerrainParameters, seed: int) -> "TerrainMap":
        cells = [[Terrain.ROCKY for _ in range(parameters.width)] for _ in range(parameters.height)]
        for line in parameters.road_lines:
            if 0 <= line < parameters.height:
                for x in range(parameters.width):
                    cells[line][x] = Terrain.ROAD
            if 0 <= line < parameters.width:
                for y in range(parameters.height):
                    cells[y][line] = Terrain.ROAD
        random = stream(seed, "map", "puddle-patches")
        maximum_x = parameters.width - parameters.puddle_patch_size
        maximum_y = parameters.height - parameters.puddle_patch_size
        for _ in range(parameters.puddle_patch_count):
            origin_x = random.randint(0, maximum_x)
            origin_y = random.randint(0, maximum_y)
            for y in range(origin_y, origin_y + parameters.puddle_patch_size):
                for x in range(origin_x, origin_x + parameters.puddle_patch_size):
                    if cells[y][x] is not Terrain.ROAD:
                        cells[y][x] = Terrain.PUDDLE
        return cls(tuple(tuple(row) for row in cells), parameters)

    def to_rows(self) -> list[list[str]]:
        return [[cell.value for cell in row] for row in self.cells]

    def in_bounds(self, node: tuple[int, int]) -> bool:
        return 0 <= node[0] < self.width and 0 <= node[1] < self.height

    def terrain_at_node(self, node: tuple[int, int]) -> Terrain:
        if not self.in_bounds(node):
            raise ValueError(f"Position outside terrain: {node}")
        return self.cells[node[1]][node[0]]

    def speed(self, terrain: Terrain) -> float:
        return {
            Terrain.ROAD: self.parameters.road_speed,
            Terrain.ROCKY: self.parameters.rocky_speed,
            Terrain.PUDDLE: self.parameters.puddle_speed,
        }[terrain]

    def energy_cost(self, terrain: Terrain) -> float:
        return {
            Terrain.ROAD: self.parameters.road_energy_cost,
            Terrain.ROCKY: self.parameters.rocky_energy_cost,
            Terrain.PUDDLE: self.parameters.puddle_energy_cost,
        }[terrain]

    def neighbors(self, node: tuple[int, int]) -> Iterable[tuple[int, int]]:
        x, y = node
        for candidate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self.in_bounds(candidate):
                yield candidate

    def edge_weight(self, first: tuple[int, int], second: tuple[int, int], metric: str) -> float:
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
            raise ValueError("Terrain graph edges must connect four-neighbor cells")
        first_terrain = self.terrain_at_node(first)
        second_terrain = self.terrain_at_node(second)
        if metric == "time":
            return 0.5 / self.speed(first_terrain) + 0.5 / self.speed(second_terrain)
        if metric == "energy":
            return 0.5 * self.energy_cost(first_terrain) + 0.5 * self.energy_cost(second_terrain)
        raise ValueError(f"Unknown route metric: {metric}")

    def _field(self, goal: tuple[int, int], metric: str):
        cache_key = (goal, metric)
        if cache_key in self._field_cache:
            return self._field_cache[cache_key]
        distances = {goal: 0.0}
        next_node: dict[tuple[int, int], tuple[int, int]] = {}
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, goal)]
        while queue:
            current_distance, current = heapq.heappop(queue)
            if current_distance > distances[current] + EPSILON:
                continue
            for neighbor in self.neighbors(current):
                candidate = current_distance + self.edge_weight(current, neighbor, metric)
                if candidate + EPSILON < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    next_node[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
                elif abs(candidate - distances.get(neighbor, math.inf)) <= EPSILON:
                    # Fixed node ordering makes exact path ties independent of dict order.
                    if current < next_node.get(neighbor, (self.width, self.height)):
                        next_node[neighbor] = current
        self._field_cache[cache_key] = distances, next_node
        return distances, next_node

    def _node_route(self, start: tuple[int, int], goal: tuple[int, int], metric: str) -> tuple[float, tuple[tuple[int, int], ...]]:
        key = (start, goal, metric)
        if key in self._pair_cache:
            return self._pair_cache[key]
        if goal in self._static_goals:
            distances, next_node = self._field(goal, metric)
            if start not in distances:
                return math.inf, ()
            route = [start]
            while route[-1] != goal:
                route.append(next_node[route[-1]])
            result = distances[start], tuple(route)
        else:
            result = self._point_route(start, goal, metric)
        self._pair_cache[key] = result
        return result

    def _point_route(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        metric: str,
    ) -> tuple[float, tuple[tuple[int, int], ...]]:
        """A* for moving destinations, avoiding a full reverse field per cell."""
        if start == goal:
            return 0.0, (start,)
        minimum_edge = (
            1.0 / self.parameters.road_speed
            if metric == "time"
            else self.parameters.road_energy_cost
        )
        distance_from_start = {start: 0.0}
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        queue: list[tuple[float, float, tuple[int, int]]] = [
            (minimum_edge * (abs(start[0] - goal[0]) + abs(start[1] - goal[1])), 0.0, start)
        ]
        while queue:
            _estimate, distance_so_far, current = heapq.heappop(queue)
            if distance_so_far > distance_from_start[current] + EPSILON:
                continue
            if current == goal:
                break
            for neighbor in self.neighbors(current):
                candidate = distance_so_far + self.edge_weight(current, neighbor, metric)
                existing = distance_from_start.get(neighbor, math.inf)
                if candidate + EPSILON < existing or (
                    abs(candidate - existing) <= EPSILON
                    and current < previous.get(neighbor, (self.width, self.height))
                ):
                    distance_from_start[neighbor] = candidate
                    previous[neighbor] = current
                    heuristic = minimum_edge * (
                        abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    )
                    heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
        if goal not in distance_from_start:
            return math.inf, ()
        route = [goal]
        while route[-1] != start:
            route.append(previous[route[-1]])
        route.reverse()
        return distance_from_start[goal], tuple(route)

    def _anchors(self, position: Position) -> tuple[tuple[int, int], ...]:
        x, y = position
        rounded_x, rounded_y = round(x), round(y)
        x_integer = abs(x - rounded_x) <= EPSILON
        y_integer = abs(y - rounded_y) <= EPSILON
        if x_integer and y_integer:
            node = (int(rounded_x), int(rounded_y))
            if not self.in_bounds(node):
                raise ValueError(f"Position outside terrain: {position}")
            return (node,)
        if not x_integer and y_integer:
            candidates = ((math.floor(x), int(rounded_y)), (math.ceil(x), int(rounded_y)))
        elif x_integer and not y_integer:
            candidates = ((int(rounded_x), math.floor(y)), (int(rounded_x), math.ceil(y)))
        else:
            raise ValueError(f"Position is not on a grid edge: {position}")
        return tuple(candidate for candidate in candidates if self.in_bounds(candidate))

    def _partial_edge_segments(self, start: Position, end: Position) -> list[AtomicSegment]:
        if math.dist(start, end) <= EPSILON:
            return []
        anchors = set(self._anchors(start)) | set(self._anchors(end))
        if len(anchors) > 2:
            raise ValueError(f"Partial segment does not lie on one grid edge: {start} -> {end}")
        if abs(start[1] - end[1]) <= EPSILON:
            fixed = int(round(start[1]))
            low = (int(math.floor(min(start[0], end[0]))), fixed)
            high = (low[0] + 1, fixed)
            coordinate = 0
        elif abs(start[0] - end[0]) <= EPSILON:
            fixed = int(round(start[0]))
            low = (fixed, int(math.floor(min(start[1], end[1]))))
            high = (fixed, low[1] + 1)
            coordinate = 1
        else:
            raise ValueError("Routes may only move along grid edges")
        if not self.in_bounds(low) or not self.in_bounds(high):
            # A zero-length edge at the outer boundary uses its sole cell.
            node = next(iter(anchors))
            terrain = self.terrain_at_node(node)
            length = math.dist(start, end)
            return [self._atomic(start, end, terrain, length)]
        midpoint_value = (low[coordinate] + high[coordinate]) / 2.0
        start_value, end_value = start[coordinate], end[coordinate]
        split_required = (start_value - midpoint_value) * (end_value - midpoint_value) < -EPSILON
        points = [start]
        if split_required:
            midpoint = (midpoint_value, start[1]) if coordinate == 0 else (start[0], midpoint_value)
            points.append(midpoint)
        points.append(end)
        output: list[AtomicSegment] = []
        for first, second in zip(points, points[1:]):
            center = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
            terrain = self.terrain_at_node(low if center[coordinate] < midpoint_value else high)
            output.append(self._atomic(first, second, terrain, math.dist(first, second)))
        return output

    def _atomic(self, start: Position, end: Position, terrain: Terrain, length: float) -> AtomicSegment:
        return AtomicSegment(
            start=start,
            end=end,
            terrain=terrain,
            length=length,
            travel_time=length / self.speed(terrain),
            movement_energy=length * self.energy_cost(terrain),
        )

    def _segments_from_points(self, points: list[Position]) -> list[AtomicSegment]:
        result: list[AtomicSegment] = []
        for first, second in zip(points, points[1:]):
            result.extend(self._partial_edge_segments(first, second))
        return result

    def _node_path_segments(
        self,
        nodes: tuple[tuple[int, int], ...],
        metric: str,
        stop_distance: float,
    ) -> tuple[AtomicSegment, ...]:
        if not nodes:
            return ()
        key = (nodes[0], nodes[-1], metric, stop_distance)
        cached = self._node_segments_cache.get(key)
        if cached is not None:
            return cached
        result: list[AtomicSegment] = []
        for first, second in zip(nodes, nodes[1:]):
            first_position = (float(first[0]), float(first[1]))
            second_position = (float(second[0]), float(second[1]))
            midpoint = (
                (first_position[0] + second_position[0]) / 2.0,
                (first_position[1] + second_position[1]) / 2.0,
            )
            result.append(self._atomic(first_position, midpoint, self.terrain_at_node(first), 0.5))
            result.append(self._atomic(midpoint, second_position, self.terrain_at_node(second), 0.5))
        if stop_distance > 0:
            result = _trim_route_end(result, stop_distance, self)
        cached = tuple(result)
        self._node_segments_cache[key] = cached
        return cached

    def estimate_route_cost(self, start: Position, goal: Position, metric: str, stop_distance: float = 0.0) -> float:
        """Return route cost without constructing physical segment objects.

        Resource goals are grid centers.  For a one-unit interaction radius,
        reaching the preceding route node is exact and avoids rebuilding every
        half-edge solely for preliminary candidate ranking.
        """
        if math.dist(start, goal) <= stop_distance + EPSILON:
            return 0.0
        goal_anchors = self._anchors(goal)
        best = math.inf
        for start_anchor in self._anchors(start):
            connector = self._segments_from_points([start, (float(start_anchor[0]), float(start_anchor[1]))])
            connector_cost = sum(s.travel_time if metric == "time" else s.movement_energy for s in connector)
            for goal_anchor in goal_anchors:
                node_cost, nodes = self._node_route(start_anchor, goal_anchor, metric)
                if not nodes:
                    continue
                goal_connector = self._segments_from_points([(float(goal_anchor[0]), float(goal_anchor[1])), goal])
                goal_cost = sum(s.travel_time if metric == "time" else s.movement_energy for s in goal_connector)
                total = connector_cost + node_cost + goal_cost
                if stop_distance >= 1.0 - EPSILON and len(goal_anchors) == 1 and len(nodes) > 1:
                    total -= self.edge_weight(nodes[-2], nodes[-1], metric)
                best = min(best, max(0.0, total))
        if not math.isfinite(best):
            raise RuntimeError(f"No path from {start} to {goal}")
        return best

    def route(self, start: Position, goal: Position, metric: str, stop_distance: float = 0.0) -> Route:
        if math.dist(start, goal) <= stop_distance + EPSILON:
            return Route(start, goal, metric, ())
        best: tuple[float, list[Position]] | None = None
        start_anchors = self._anchors(start)
        goal_anchors = self._anchors(goal)
        for start_anchor in start_anchors:
            start_connector = self._segments_from_points([start, (float(start_anchor[0]), float(start_anchor[1]))])
            start_cost = sum(s.travel_time if metric == "time" else s.movement_energy for s in start_connector)
            for goal_anchor in goal_anchors:
                node_cost, nodes = self._node_route(start_anchor, goal_anchor, metric)
                if not nodes:
                    continue
                goal_connector = self._segments_from_points([(float(goal_anchor[0]), float(goal_anchor[1])), goal])
                goal_cost = sum(s.travel_time if metric == "time" else s.movement_energy for s in goal_connector)
                total = start_cost + node_cost + goal_cost
                points = [start]
                points.extend((float(x), float(y)) for x, y in nodes)
                points.append(goal)
                points = _deduplicate_points(points)
                candidate = (total, points)
                if best is None or candidate[0] < best[0] - EPSILON or (
                    abs(candidate[0] - best[0]) <= EPSILON and tuple(candidate[1]) < tuple(best[1])
                ):
                    best = candidate
        if best is None:
            raise RuntimeError(f"No path from {start} to {goal}")
        points = best[1]
        start_anchors = self._anchors(start)
        goal_anchors = self._anchors(goal)
        def point_node(point: Position) -> tuple[int, int] | None:
            rounded = (int(round(point[0])), int(round(point[1])))
            if abs(point[0] - rounded[0]) <= EPSILON and abs(point[1] - rounded[1]) <= EPSILON:
                return rounded
            return None
        first_node_index = next(
            (index for index, point in enumerate(points) if point_node(point) in start_anchors),
            0,
        )
        last_node_index = max(
            index for index, point in enumerate(points)
            if point_node(point) in goal_anchors
        )
        node_points = points[first_node_index:last_node_index + 1]
        nodes = tuple((int(round(point[0])), int(round(point[1]))) for point in node_points)
        start_connector = self._segments_from_points(points[:first_node_index + 1])
        goal_connector = self._segments_from_points(points[last_node_index:])
        trim_nodes = stop_distance if math.dist(goal, (float(nodes[-1][0]), float(nodes[-1][1]))) <= EPSILON else 0.0
        segments = start_connector + list(self._node_path_segments(nodes, metric, trim_nodes)) + goal_connector
        if stop_distance > 0 and trim_nodes == 0.0:
            segments = _trim_route_end(segments, stop_distance, self)
        return Route(start, goal, metric, tuple(segments))


def _deduplicate_points(points: list[Position]) -> list[Position]:
    result: list[Position] = []
    for point in points:
        if not result or math.dist(point, result[-1]) > EPSILON:
            result.append(point)
    return result


def _trim_route_end(segments: list[AtomicSegment], distance_to_trim: float, terrain_map: TerrainMap) -> list[AtomicSegment]:
    remaining = distance_to_trim
    result = list(segments)
    while result and remaining > EPSILON:
        segment = result[-1]
        if segment.length <= remaining + EPSILON:
            remaining -= segment.length
            result.pop()
            continue
        fraction = (segment.length - remaining) / segment.length
        end = (
            segment.start[0] + (segment.end[0] - segment.start[0]) * fraction,
            segment.start[1] + (segment.end[1] - segment.start[1]) * fraction,
        )
        result[-1] = terrain_map._atomic(segment.start, end, segment.terrain, segment.length - remaining)
        remaining = 0.0
    return result


def traverse_route(
    route: Route,
    body: BodyVector,
    dt: float,
    physiology: PhysiologyParameters,
) -> TraversalResult:
    """Integrate a route exactly using cumulative segment profiles.

    Planning evaluates the same route several times for different outcome
    branches.  The cumulative representation keeps the physical convention
    exact while making each query logarithmic in route length.
    """
    duration = max(0.0, dt)
    thresholds = (
        physiology.energy_threshold,
        physiology.water_threshold,
        physiology.integrity_threshold,
    )

    def energy_drain_at(at_time: float) -> float:
        moving = min(max(0.0, at_time), route.travel_time)
        return physiology.energy_decay * at_time + movement_energy_at(moving)

    def movement_energy_at(moving_time: float) -> float:
        if not route.segments or moving_time <= 0.0:
            return 0.0
        if moving_time >= route.travel_time - EPSILON:
            return route.movement_energy
        index = bisect.bisect_right(route.cumulative_times, moving_time)
        previous_time = 0.0 if index == 0 else route.cumulative_times[index - 1]
        previous_energy = 0.0 if index == 0 else route.cumulative_movement_energy[index - 1]
        segment = route.segments[index]
        return previous_energy + (moving_time - previous_time) * (
            segment.movement_energy / segment.travel_time
        )

    def time_for_energy_budget(budget: float) -> float:
        if budget <= 0.0:
            return 0.0
        cumulative_drain = tuple(
            physiology.energy_decay * at_time + movement_energy
            for at_time, movement_energy in zip(route.cumulative_times, route.cumulative_movement_energy)
        )
        if cumulative_drain and budget <= cumulative_drain[-1] + EPSILON:
            index = bisect.bisect_left(cumulative_drain, budget)
            previous_time = 0.0 if index == 0 else route.cumulative_times[index - 1]
            previous_drain = 0.0 if index == 0 else cumulative_drain[index - 1]
            segment = route.segments[index]
            rate = physiology.energy_decay + segment.movement_energy / segment.travel_time
            return previous_time + max(0.0, budget - previous_drain) / rate
        remaining = budget - (cumulative_drain[-1] if cumulative_drain else 0.0)
        if physiology.energy_decay <= 0.0:
            return math.inf
        return route.travel_time + max(0.0, remaining) / physiology.energy_decay

    possible_crossings = [time_for_energy_budget(body[0] - thresholds[0])]
    for value, threshold, rate in (
        (body[1], thresholds[1], physiology.water_decay),
        (body[2], thresholds[2], physiology.integrity_decay),
    ):
        possible_crossings.append((value - threshold) / rate if rate > 0.0 else math.inf)
    crossing = max(0.0, min(possible_crossings))
    died = crossing <= duration + EPSILON
    elapsed = min(duration, crossing) if died else duration
    moving_time = min(elapsed, route.travel_time)
    movement_energy = movement_energy_at(moving_time)

    if not route.segments or moving_time >= route.travel_time - EPSILON:
        position = route.start if not route.segments else route.segments[-1].end
    else:
        index = bisect.bisect_right(route.cumulative_times, moving_time)
        previous_time = 0.0 if index == 0 else route.cumulative_times[index - 1]
        segment = route.segments[index]
        fraction = min(1.0, max(0.0, (moving_time - previous_time) / segment.travel_time))
        position = (
            segment.start[0] + (segment.end[0] - segment.start[0]) * fraction,
            segment.start[1] + (segment.end[1] - segment.start[1]) * fraction,
        )

    values: BodyVector = (
        max(0.0, body[0] - energy_drain_at(elapsed)),
        max(0.0, body[1] - physiology.water_decay * elapsed),
        max(0.0, body[2] - physiology.integrity_decay * elapsed),
    )
    reached_end = not died and moving_time >= route.travel_time - EPSILON
    return TraversalResult(
        position,
        values,
        elapsed,
        moving_time,
        movement_energy,
        reached_end,
        died,
        crossing if died else None,
    )

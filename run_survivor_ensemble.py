"""Run a reproducible 3 x 3 ensemble of the baseline Survivor policy.

The agent policy below is intentionally copied from ``run_survivor.py``:
motive hysteresis, target locking, blacklist cooldowns, stuck escapes, and
terrain-aware cardinal movement all retain the baseline constants and logic.
Only resource abundance, start position, run seed, and run length differ.

Factorial design (all coordinates are island-world x/y coordinates)
====================================================================

The layouts are nested, so each abundance level contains every safe object
from the preceding level.  Every layout also has exactly one FlyAgaric trap
at (620, 420).

Safe resource layouts:

    sparse (6):
      Waterhole   (300, 500), (1200, 1000)
      Champignon  (500, 350), (1150, 550)
      Wirselkraut (750, 650)
      Juniper     (350, 900)

    medium (10; sparse plus):
      Waterhole   (1250, 950)
      Champignon  (850, 1050)
      Wirselkraut (1050, 750)
      Juniper     (650, 1100)

    plentiful (14; medium plus):
      Waterhole   (500, 950)
      Champignon  (1300, 1000)
      Wirselkraut (450, 700)
      Juniper     (950, 400)

The nine explicit configurations are:

    ID  abundance   distance  start         nearest safe resource  seed
     1  sparse      close     (700, 700)       70.71               101
     2  sparse      medium    (650, 900)       269.26              102
     3  sparse      far       (1700, 900)      509.90              103
     4  medium      close     (700, 700)       70.71               104
     5  medium      medium    (750, 850)       200.00              105
     6  medium      far       (1700, 900)      452.77              106
     7  plentiful  close     (700, 700)       70.71               107
     8  plentiful  medium    (800, 850)       206.16              108
     9  plentiful  far       (1700, 900)      412.31              109

Resource "visited" means that the agent came within the baseline action
radius (strictly less than 45 world units) of a safe resource or the trap.
All 800 steps are logged; no temporal downsampling is used.
"""

from collections import Counter
import json
import math
import os
import random
import shutil
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(SCRIPT_DIR, "output", ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import micropsi_core.runtime as runtime
from micropsi_core.world.island.island import ground_types


WORLD_UID = "d4b3f5740adc11e5b9fe20c9d087b4b7"
N_STEPS = 800
SWITCH_MARGIN = 0.12
VISIT_RADIUS = 45.0
FOOD_TYPES = {"Champignon", "Wirselkraut", "Juniper", "FlyAgaric"}
WATER_TYPES = {"Waterhole"}
RESOURCE_TYPES = FOOD_TYPES | WATER_TYPES

SPARSE_RESOURCES = (
    ("Waterhole", (300, 500)),
    ("Waterhole", (1200, 1000)),
    ("Champignon", (500, 350)),
    ("Champignon", (1150, 550)),
    ("Wirselkraut", (750, 650)),
    ("Juniper", (350, 900)),
)
MEDIUM_ADDITIONS = (
    ("Waterhole", (1250, 950)),
    ("Champignon", (850, 1050)),
    ("Wirselkraut", (1050, 750)),
    ("Juniper", (650, 1100)),
)
PLENTIFUL_ADDITIONS = (
    ("Waterhole", (500, 950)),
    ("Champignon", (1300, 1000)),
    ("Wirselkraut", (450, 700)),
    ("Juniper", (950, 400)),
)
TRAP_SPEC = ("FlyAgaric", (620, 420))

RESOURCE_LAYOUTS = {
    "sparse": SPARSE_RESOURCES + (TRAP_SPEC,),
    "medium": SPARSE_RESOURCES + MEDIUM_ADDITIONS + (TRAP_SPEC,),
    "plentiful": SPARSE_RESOURCES + MEDIUM_ADDITIONS + PLENTIFUL_ADDITIONS + (TRAP_SPEC,),
}

CONFIGS = (
    {"id": 1, "abundance": "sparse", "distance": "close", "start": (700, 700), "seed": 101},
    {"id": 2, "abundance": "sparse", "distance": "medium", "start": (650, 900), "seed": 102},
    {"id": 3, "abundance": "sparse", "distance": "far", "start": (1700, 900), "seed": 103},
    {"id": 4, "abundance": "medium", "distance": "close", "start": (700, 700), "seed": 104},
    {"id": 5, "abundance": "medium", "distance": "medium", "start": (750, 850), "seed": 105},
    {"id": 6, "abundance": "medium", "distance": "far", "start": (1700, 900), "seed": 106},
    {"id": 7, "abundance": "plentiful", "distance": "close", "start": (700, 700), "seed": 107},
    {"id": 8, "abundance": "plentiful", "distance": "medium", "start": (800, 850), "seed": 108},
    {"id": 9, "abundance": "plentiful", "distance": "far", "start": (1700, 900), "seed": 109},
)


def prepare_runtime():
    """Initialize MicroPsi from an isolated copy of the bundled demo data."""
    data_dir = os.path.join(SCRIPT_DIR, "demo_run_data")
    if not os.path.isdir(data_dir):
        os.makedirs(os.path.join(data_dir, "nodenets"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "worlds"), exist_ok=True)
        for filename in os.listdir(os.path.join(SCRIPT_DIR, "demo_data", "nodenets")):
            shutil.copy(
                os.path.join(SCRIPT_DIR, "demo_data", "nodenets", filename),
                os.path.join(data_dir, "nodenets", filename),
            )
        for filename in os.listdir(os.path.join(SCRIPT_DIR, "demo_data", "worlds")):
            shutil.copy(
                os.path.join(SCRIPT_DIR, "demo_data", "worlds", filename),
                os.path.join(data_dir, "worlds", filename),
            )
    runtime.initialize(persistency_path=data_dir, resource_path=data_dir)
    return runtime.worlds[WORLD_UID]


def squared_distance(first, second):
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def clean_controlled_resources(world):
    """Remove resource-class objects while retaining the demo Lightsource."""
    for uid, obj in list(world.objects.items()):
        if getattr(obj, "structured_object_type", None) in RESOURCE_TYPES:
            runtime.delete_worldobject(WORLD_UID, uid)


def validate_layout(world, resource_layout, start_position):
    """Fail early when a documented coordinate is outside walkable terrain."""
    points = [("start", start_position)] + list(resource_layout)
    for label, position in points:
        ground = ground_types[world.get_ground_at(*position)]
        if not ground["agent_allowed"]:
            raise ValueError(f"{label} coordinate {position} is on impassable {ground['type']} terrain")


def add_resources(resource_layout, config_id):
    """Populate one controlled layout and return serializable object metadata."""
    resources = []
    for index, (objtype, position) in enumerate(resource_layout, start=1):
        uid = f"ensemble_c{config_id}_resource_{index:02d}"
        ok, returned_uid = runtime.add_worldobject(
            WORLD_UID, objtype, position, name=f"{objtype} C{config_id}", uid=uid
        )
        if not ok:
            raise RuntimeError(f"Could not add {objtype} at {position}: {returned_uid}")
        resources.append({"uid": returned_uid, "type": objtype, "position": list(position)})
    return resources


def summarize_run(config, result):
    steps = result["steps"]
    emo_keys = sorted({key for row in steps for key in row if key.startswith("emo_")})

    def mean(key):
        return sum(float(row[key]) for row in steps) / len(steps)

    return {
        "config_id": config["id"],
        "abundance": config["abundance"],
        "starting_distance": config["distance"],
        "start_position": list(config["start"]),
        "seed": config["seed"],
        "safe_resource_count": sum(r["type"] != "FlyAgaric" for r in result["resources"]),
        "trap_count": sum(r["type"] == "FlyAgaric" for r in result["resources"]),
        "nearest_safe_resource_distance": result["nearest_safe_resource_distance"],
        "steps_requested": result["steps_requested"],
        "steps_completed": result["steps_completed"],
        "survived": result["survived"],
        "died_at_step": result["died_at_step"],
        "emo_modulators": {
            key: {"mean": mean(key), "final": float(steps[-1][key])} for key in emo_keys
        },
        "body_demands": {
            key: {"mean": mean(key), "final": float(steps[-1][key])}
            for key in ("energy", "water", "integrity")
        },
        "ruling_motive_counts": dict(Counter(row["ruling_motive"] for row in steps)),
        "total_distinct_resources_visited": len(result["visited_resource_uids"]),
        "visited_resource_uids": result["visited_resource_uids"],
        "visited_resource_types": result["visited_resource_types"],
    }


def run_one_simulation(
    resource_layout,
    start_position,
    n_steps,
    seed,
    config_id,
    update_base_sum_of_urges=False,
):
    """Build and run one baseline-policy simulation, returning its full log."""
    world = runtime.worlds[WORLD_UID]
    agent_uid = f"survivor_ensemble_agent_{config_id}"
    clean_controlled_resources(world)
    if agent_uid in runtime.nodenets:
        runtime.delete_nodenet(agent_uid)

    validate_layout(world, resource_layout, start_position)
    random.seed(seed)
    resources = add_resources(resource_layout, config_id)
    resource_by_uid = {resource["uid"]: resource for resource in resources}
    safe_positions = [r["position"] for r in resources if r["type"] != "FlyAgaric"]
    nearest_safe_distance = math.sqrt(min(squared_distance(start_position, p) for p in safe_positions))

    ok, nodenet_uid = runtime.new_nodenet(
        f"SurvivorEnsemble{config_id}",
        engine="dict_engine",
        worldadapter="Survivor",
        world_uid=WORLD_UID,
        uid=agent_uid,
    )
    if not ok:
        clean_controlled_resources(world)
        raise RuntimeError(f"Could not create ensemble nodenet {config_id}: {nodenet_uid}")

    nn = runtime.nodenets[nodenet_uid]
    wa = nn.worldadapter_instance
    wa.name = "Survivor"
    wa.position = start_position
    world.data["agents"][nodenet_uid]["position"] = wa.position

    def is_walkable(x, y):
        idx = world.get_ground_at(x, y)
        return ground_types[idx]["agent_allowed"]

    def best_move_toward(pos, tx, ty):
        """Baseline four-way greedy, terrain-aware navigation."""
        candidates = {
            "loco_east": (pos[0] + 50, pos[1]),
            "loco_west": (pos[0] - 50, pos[1]),
            "loco_north": (pos[0], pos[1] + 50),
            "loco_south": (pos[0], pos[1] - 50),
        }
        cur_d = (pos[0] - tx) ** 2 + (pos[1] - ty) ** 2
        best_key, best_d = None, cur_d
        for key, (next_x, next_y) in candidates.items():
            if not is_walkable(next_x, next_y):
                continue
            distance = (next_x - tx) ** 2 + (next_y - ty) ** 2
            if distance < best_d:
                best_key, best_d = key, distance
        return best_key

    def nearest_of(types, position):
        best, best_distance = None, float("inf")
        for obj in world.objects.values():
            if getattr(obj, "structured_object_type", None) in types:
                distance = squared_distance(obj.position, position)
                if distance < best_distance:
                    best, best_distance = obj, distance
        return best, math.sqrt(best_distance) if best else None

    competence_estimate = {"energy": 0.5, "water": 0.5, "integrity": 0.5}
    stuck_counter = 0
    escape_dir = None
    escape_ttl = 0
    current_ruling = "energy"
    current_target_uid = None
    blacklist = {}
    log = []
    visited_resource_uids = set()

    try:
        for step in range(n_steps):
            blacklist = {key: value - 1 for key, value in blacklist.items() if value - 1 > 0}
            energy = wa.datasources["body-energy"]
            water = wa.datasources["body-water"]
            integrity = wa.datasources["body-integrity"]

            urges = {
                "energy": max(0.0, 1.0 - energy),
                "water": max(0.0, 1.0 - water),
                "integrity": max(0.0, 1.0 - integrity),
            }
            sum_urges = sum(urges.values())

            best_rival = max(urges, key=urges.get)
            if urges[best_rival] > urges[current_ruling] + SWITCH_MARGIN:
                current_ruling = best_rival
                current_target_uid = None
            ruling = current_ruling
            n_active_motives = sum(1 for value in urges.values() if value > 0.08)

            type_set = WATER_TYPES if ruling == "water" else (
                FOOD_TYPES if ruling == "energy" else {"Wirselkraut"}
            )
            target = world.objects.get(current_target_uid) if current_target_uid else None
            if target is None or target.uid in blacklist:
                candidates = {
                    uid: obj
                    for uid, obj in world.objects.items()
                    if getattr(obj, "structured_object_type", None) in type_set and uid not in blacklist
                }
                target, _ = nearest_of(set(type_set), wa.position) if not candidates else (
                    min(candidates.values(), key=lambda obj: squared_distance(obj.position, wa.position)),
                    None,
                )
                current_target_uid = target.uid if target else None
            distance_to_target = None
            if target is not None:
                distance_to_target = math.sqrt(squared_distance(target.position, wa.position))

            for key in ("loco_north", "loco_south", "loco_east", "loco_west"):
                wa.datatargets[key] = 0

            if escape_ttl > 0:
                wa.datatargets[escape_dir] = 1
                escape_ttl -= 1
            elif target is not None:
                move_key = best_move_toward(wa.position, target.position[0], target.position[1])
                if move_key:
                    wa.datatargets[move_key] = 1
                if distance_to_target is not None and distance_to_target < VISIT_RADIUS:
                    if ruling == "water":
                        wa.datatargets["action_drink"] = 1
                    else:
                        wa.datatargets["action_eat"] = 1

            movement_keys = ("loco_north", "loco_south", "loco_east", "loco_west")
            moved_intentionally = any(wa.datatargets[key] for key in movement_keys)
            prev_integrity = integrity
            pos_before = wa.position
            wa.update_data_sources_and_targets()

            if moved_intentionally and wa.position == pos_before and escape_ttl == 0:
                stuck_counter += 1
                if stuck_counter >= 4:
                    escape_dir = random.choice(list(movement_keys))
                    escape_ttl = 8
                    stuck_counter = 0
                elif random.random() < 0.15:
                    current_target_uid = None
            else:
                stuck_counter = 0

            new_energy = wa.datasources["body-energy"]
            new_water = wa.datasources["body-water"]
            new_integrity = wa.datasources["body-integrity"]
            unexpected = 1 if (new_integrity - prev_integrity) < -0.05 else 0
            fed = wa.datatarget_feedback.get("action_eat", 0) or wa.datatarget_feedback.get("action_drink", 0)
            expected = 1 if fed and not unexpected else 0
            action_resource_uid = target.uid if fed and target is not None else None
            action_resource_type = (
                getattr(target, "structured_object_type", None) if action_resource_uid else None
            )
            if fed:
                helped = {
                    "energy": new_energy - energy,
                    "water": new_water - water,
                    "integrity": new_integrity - integrity,
                }[ruling] > 0.01
                if helped:
                    current_target_uid = None
                elif current_target_uid:
                    blacklist[current_target_uid] = 120
                    current_target_uid = None

            if fed:
                gain = (new_energy - energy) + (new_water - water) + (new_integrity - integrity)
                success = 1.0 if gain > 0.01 else 0.0
                competence_estimate[ruling] = 0.8 * competence_estimate[ruling] + 0.2 * success

            new_urges = {
                "energy": max(0.0, 1.0 - new_energy),
                "water": max(0.0, 1.0 - new_water),
                "integrity": max(0.0, 1.0 - new_integrity),
            }
            new_sum_urges = sum(new_urges.values())
            urge_change = new_sum_urges - sum_urges

            nn.set_modulator("base_sum_importance_of_intentions", new_sum_urges)
            nn.set_modulator("base_sum_urgency_of_intentions", new_sum_urges)
            nn.set_modulator("base_competence_for_intention", competence_estimate[ruling])
            nn.set_modulator("base_importance_of_intention", new_urges[ruling])
            nn.set_modulator("base_urgency_of_intention", new_urges[ruling])
            nn.set_modulator("base_number_of_active_motives", max(1, n_active_motives))
            nn.set_modulator("base_number_of_expected_events", expected)
            nn.set_modulator("base_number_of_unexpected_events", unexpected)
            nn.set_modulator("base_urge_change", urge_change)
            if update_base_sum_of_urges:
                nn.set_modulator("base_sum_of_urges", new_sum_urges)
            nn.step()

            mods = {
                key: value
                for key, value in nn.construct_modulators_dict().items()
                if key.startswith("emo_")
            }
            nearby = [
                uid
                for uid, resource in resource_by_uid.items()
                if squared_distance(wa.position, resource["position"]) < VISIT_RADIUS**2
            ]
            visited_resource_uids.update(nearby)

            log.append(
                {
                    "step": step,
                    "x": wa.position[0],
                    "y": wa.position[1],
                    "energy": new_energy,
                    "water": new_water,
                    "integrity": new_integrity,
                    "ruling_motive": ruling,
                    "unexpected": unexpected,
                    "target_dist": distance_to_target,
                    "target_uid": current_target_uid,
                    "visited_resource_uid": nearby[0] if nearby else None,
                    "action_success": int(bool(fed)),
                    "action_resource_uid": action_resource_uid,
                    "action_resource_type": action_resource_type,
                    "expected": expected,
                    "urge_change": urge_change,
                    "sum_urges": new_sum_urges,
                    **mods,
                }
            )
            if wa.is_dead:
                break
    finally:
        if nodenet_uid in runtime.nodenets:
            runtime.delete_nodenet(nodenet_uid)
        clean_controlled_resources(world)

    if not log:
        raise RuntimeError(f"Configuration {config_id} produced no log rows")
    survived = len(log) == n_steps and not wa.is_dead
    visited_sorted = sorted(visited_resource_uids)
    return {
        "config_id": config_id,
        "seed": seed,
        "start_position": list(start_position),
        "nearest_safe_resource_distance": nearest_safe_distance,
        "steps_requested": n_steps,
        "steps_completed": len(log),
        "base_sum_of_urges_updated": update_base_sum_of_urges,
        "survived": survived,
        "died_at_step": None if survived else log[-1]["step"],
        "resources": resources,
        "visited_resource_uids": visited_sorted,
        "visited_resource_types": [resource_by_uid[uid]["type"] for uid in visited_sorted],
        "steps": log,
    }


def main():
    world = prepare_runtime()
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    summaries = []

    for config in CONFIGS:
        print(
            f"Running config {config['id']}/9: {config['abundance']} resources, "
            f"{config['distance']} start, seed {config['seed']}"
        )
        result = run_one_simulation(
            RESOURCE_LAYOUTS[config["abundance"]],
            config["start"],
            N_STEPS,
            config["seed"],
            config["id"],
        )
        result["abundance"] = config["abundance"]
        result["starting_distance"] = config["distance"]
        log_path = os.path.join(output_dir, f"ensemble_config_{config['id']}.json")
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
        summary = summarize_run(config, result)
        summaries.append(summary)
        outcome = "survived" if result["survived"] else f"died at step {result['died_at_step']}"
        print(
            f"  {outcome}; {result['steps_completed']} rows; "
            f"{summary['total_distinct_resources_visited']} distinct resources visited"
        )

    summary_document = {
        "experiment": "Variation D: resource abundance x starting distance",
        "steps_per_config": N_STEPS,
        "logging_interval_steps": 1,
        "switch_margin": SWITCH_MARGIN,
        "visit_definition": "agent position strictly less than 45 world units from a resource",
        "abundance_order": ["sparse", "medium", "plentiful"],
        "distance_order": ["close", "medium", "far"],
        "configs": summaries,
    }
    summary_path = os.path.join(output_dir, "ensemble_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_document, handle, indent=2)
    print(f"Ensemble complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

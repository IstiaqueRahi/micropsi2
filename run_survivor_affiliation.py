import json
import math
import os
import random
import shutil
import sys
from collections import Counter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import micropsi_core.runtime as runtime
from micropsi_core.world.island.island import ground_types


# Seed the persistency directory exactly as the baseline driver does.  This
# variation owns only its two nodenet UIDs and leaves every other nodenet alone.
DATA_DIR = os.path.join(SCRIPT_DIR, "demo_run_data")
if not os.path.isdir(DATA_DIR):
    os.makedirs(os.path.join(DATA_DIR, "nodenets"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "worlds"), exist_ok=True)
    for filename in os.listdir(os.path.join(SCRIPT_DIR, "demo_data", "nodenets")):
        shutil.copy(
            os.path.join(SCRIPT_DIR, "demo_data", "nodenets", filename),
            os.path.join(DATA_DIR, "nodenets", filename),
        )
    for filename in os.listdir(os.path.join(SCRIPT_DIR, "demo_data", "worlds")):
        shutil.copy(
            os.path.join(SCRIPT_DIR, "demo_data", "worlds", filename),
            os.path.join(DATA_DIR, "worlds", filename),
        )


WORLD_UID = "d4b3f5740adc11e5b9fe20c9d087b4b7"
AGENT_SPECS = {
    "agent_a": {
        "uid": "survivor_agent_a",
        "name": "Survivor Agent A",
        "position": (400, 500),
        "energy": 0.75,
        "water": 1.0,
        "seed": 17,
    },
    "agent_b": {
        "uid": "survivor_agent_b",
        "name": "Survivor Agent B",
        "position": (1200, 600),
        "energy": 1.0,
        "water": 0.72,
        "seed": 29,
    },
}

# A separation of 180 map units (about 3.6 cardinal moves) counts as socially
# close.  The urge then rises linearly and saturates at 2800 units, near the
# 2048-by-2048 island's corner-to-corner distance.  These values make
# affiliation important during genuine separation without crowding out urgent
# food, water, or safety needs.  A 1.9-power response keeps ordinary mid-range
# separation mild while preserving 0 and 1 at the documented thresholds.
AFFILIATION_CLOSE_DISTANCE = 180.0
AFFILIATION_FAR_DISTANCE = 2800.0
SWITCH_MARGIN = 0.12
HOME_AFFINITY_WEIGHT = 2.0
ACTION_DISTANCE = 60.0
N_STEPS = 1400

FOOD_TYPES = {"Champignon", "Wirselkraut", "Juniper", "FlyAgaric"}
WATER_TYPES = {"Waterhole"}
MOVEMENT_TARGETS = ("loco_north", "loco_south", "loco_east", "loco_west")


for spec in AGENT_SPECS.values():
    stale_nodenet = os.path.join(DATA_DIR, "nodenets", spec["uid"] + ".json")
    if os.path.exists(stale_nodenet):
        os.remove(stale_nodenet)

runtime.initialize(persistency_path=DATA_DIR, resource_path=DATA_DIR)
world = runtime.worlds[WORLD_UID]


# Populate the same baseline island with the same resources.
random.seed(7)
RESOURCE_SPECS = [
    ("Waterhole", (300, 500)),
    ("Waterhole", (900, 1400)),
    ("Champignon", (500, 350)),
    ("Champignon", (1150, 550)),
    ("Wirselkraut", (750, 650)),
    ("Juniper", (350, 900)),
    ("FlyAgaric", (620, 420)),
    ("PalmTree", (1000, 300)),
    ("Stone", (450, 1100)),
    ("Boulder", (1300, 700)),
]
for object_type, position in RESOURCE_SPECS:
    runtime.add_worldobject(WORLD_UID, object_type, position, name=object_type)


def is_walkable(x, y):
    ground_index = world.get_ground_at(x, y)
    return ground_types[ground_index]["agent_allowed"]


def best_move_toward(position, target_x, target_y):
    """Return the walkable cardinal move that most reduces target distance."""
    candidates = {
        "loco_east": (position[0] + 50, position[1]),
        "loco_west": (position[0] - 50, position[1]),
        "loco_north": (position[0], position[1] + 50),
        "loco_south": (position[0], position[1] - 50),
    }
    current_distance_squared = (
        (position[0] - target_x) ** 2 + (position[1] - target_y) ** 2
    )
    best_key = None
    best_distance_squared = current_distance_squared
    for key, (next_x, next_y) in candidates.items():
        if not is_walkable(next_x, next_y):
            continue
        candidate_distance_squared = (
            (next_x - target_x) ** 2 + (next_y - target_y) ** 2
        )
        if candidate_distance_squared < best_distance_squared:
            best_key = key
            best_distance_squared = candidate_distance_squared
    return best_key


def walkable_moves(position):
    candidates = {
        "loco_east": (position[0] + 50, position[1]),
        "loco_west": (position[0] - 50, position[1]),
        "loco_north": (position[0], position[1] + 50),
        "loco_south": (position[0], position[1] - 50),
    }
    return [key for key, next_position in candidates.items() if is_walkable(*next_position)]


def euclidean_distance(position_a, position_b):
    return math.hypot(
        position_a[0] - position_b[0], position_a[1] - position_b[1]
    )


def affiliation_urge(distance):
    span = AFFILIATION_FAR_DISTANCE - AFFILIATION_CLOSE_DISTANCE
    normalized = max(
        0.0, min(1.0, (distance - AFFILIATION_CLOSE_DISTANCE) / span)
    )
    return normalized**1.9


class AgentController:
    """Per-agent motive, navigation, learning, and modulator state."""

    def __init__(self, key, nodenet, worldadapter, rng_seed, home_position):
        self.key = key
        self.nn = nodenet
        self.wa = worldadapter
        self.rng = random.Random(rng_seed)
        self.home_position = home_position
        self.competence_estimate = {
            "energy": 0.5,
            "water": 0.5,
            "integrity": 0.5,
            "affiliation": 0.5,
        }
        self.stuck_counter = 0
        self.escape_direction = None
        self.escape_ttl = 0
        self.current_ruling = "energy"
        self.current_target_uid = None
        self.blacklist = {}
        self.prepared = None

    def nearest_available(self, object_types):
        candidates = [
            obj
            for uid, obj in world.objects.items()
            if getattr(obj, "structured_object_type", None) in object_types
            and uid not in self.blacklist
        ]
        if not candidates:
            return None
        def target_cost(obj):
            travel_cost = (obj.position[0] - self.wa.position[0]) ** 2 + (
                obj.position[1] - self.wa.position[1]
            ) ** 2
            # A stable home-region affinity keeps two independent agents from
            # deterministically choosing the same nearest resource forever
            # after meeting.  Current travel distance still matters, and all
            # targets remain ordinary baseline world objects.
            home_cost = (obj.position[0] - self.home_position[0]) ** 2 + (
                obj.position[1] - self.home_position[1]
            ) ** 2
            return travel_cost + HOME_AFFINITY_WEIGHT * home_cost

        return min(candidates, key=target_cost)

    def begin_step(self, other_position, distance_to_other):
        self.blacklist = {
            uid: remaining - 1
            for uid, remaining in self.blacklist.items()
            if remaining - 1 > 0
        }

        energy = self.wa.datasources["body-energy"]
        water = self.wa.datasources["body-water"]
        integrity = self.wa.datasources["body-integrity"]
        urges = {
            "energy": max(0.0, 1.0 - energy),
            "water": max(0.0, 1.0 - water),
            "integrity": max(0.0, 1.0 - integrity),
            # Computed independently for this agent, even though Euclidean
            # distance makes the two affiliation magnitudes symmetric.
            "affiliation": affiliation_urge(distance_to_other),
        }
        sum_urges = sum(urges.values())

        best_rival = max(urges, key=urges.get)
        switched = False
        if urges[best_rival] > urges[self.current_ruling] + SWITCH_MARGIN:
            self.current_ruling = best_rival
            self.current_target_uid = None
            switched = True
        ruling = self.current_ruling
        active_motives = sum(value > 0.08 for value in urges.values())

        target = None
        target_distance = None
        navigation_position = None
        if ruling == "affiliation":
            # The other agent is a live target: never store or lock this point.
            navigation_position = other_position
            target_distance = distance_to_other
            self.current_target_uid = None
        else:
            object_types = (
                WATER_TYPES
                if ruling == "water"
                else (FOOD_TYPES if ruling == "energy" else {"Wirselkraut"})
            )
            target = (
                world.objects.get(self.current_target_uid)
                if self.current_target_uid
                else None
            )
            if target is None or target.uid in self.blacklist:
                target = self.nearest_available(object_types)
                self.current_target_uid = target.uid if target else None
            if target is not None:
                navigation_position = target.position
                target_distance = euclidean_distance(self.wa.position, target.position)

        for movement_target in MOVEMENT_TARGETS:
            self.wa.datatargets[movement_target] = 0
        self.wa.datatargets["action_eat"] = 0
        self.wa.datatargets["action_drink"] = 0

        if self.escape_ttl > 0:
            self.wa.datatargets[self.escape_direction] = 1
            self.escape_ttl -= 1
        elif navigation_position is not None:
            affiliation_satisfied = (
                ruling == "affiliation"
                and distance_to_other <= AFFILIATION_CLOSE_DISTANCE
            )
            if not affiliation_satisfied:
                move_key = best_move_toward(
                    self.wa.position, navigation_position[0], navigation_position[1]
                )
                if move_key is None:
                    # Greedy cardinal descent can stop at a shoreline or ridge
                    # even though a route exists.  Start a short walkable
                    # detour, then retry the still-locked target.
                    detours = walkable_moves(self.wa.position)
                    if detours:
                        move_key = self.rng.choice(detours)
                        self.escape_direction = move_key
                        self.escape_ttl = 7
                if move_key:
                    self.wa.datatargets[move_key] = 1
            if (
                ruling != "affiliation"
                and target_distance is not None
                and target_distance <= ACTION_DISTANCE
            ):
                action = "action_drink" if ruling == "water" else "action_eat"
                self.wa.datatargets[action] = 1

        moved_intentionally = any(
            self.wa.datatargets[target] for target in MOVEMENT_TARGETS
        )
        self.prepared = {
            "energy": energy,
            "water": water,
            "integrity": integrity,
            "urges": urges,
            "sum_urges": sum_urges,
            "ruling": ruling,
            "switched": switched,
            "active_motives": active_motives,
            "target_distance": target_distance,
            "distance_to_other": distance_to_other,
            "position_before": self.wa.position,
            "moved_intentionally": moved_intentionally,
            "action_requested": "action_drink"
            if self.wa.datatargets["action_drink"]
            else ("action_eat" if self.wa.datatargets["action_eat"] else None),
        }

    def advance_body(self):
        self.wa.update_data_sources_and_targets()

    def finish_step(self, new_distance_to_other):
        state = self.prepared
        ruling = state["ruling"]

        if (
            state["moved_intentionally"]
            and self.wa.position == state["position_before"]
            and self.escape_ttl == 0
        ):
            self.stuck_counter += 1
            if self.stuck_counter >= 4:
                self.escape_direction = self.rng.choice(MOVEMENT_TARGETS)
                self.escape_ttl = 8
                self.stuck_counter = 0
            elif self.rng.random() < 0.15 and ruling != "affiliation":
                self.current_target_uid = None
        else:
            self.stuck_counter = 0

        new_energy = self.wa.datasources["body-energy"]
        new_water = self.wa.datasources["body-water"]
        new_integrity = self.wa.datasources["body-integrity"]
        unexpected = int(new_integrity - state["integrity"] < -0.05)
        fed = bool(
            self.wa.datatarget_feedback.get("action_eat", 0)
            or self.wa.datatarget_feedback.get("action_drink", 0)
        )

        affiliation_satisfied = (
            ruling == "affiliation"
            and new_distance_to_other <= AFFILIATION_CLOSE_DISTANCE
        )
        expected = int((fed and not unexpected) or affiliation_satisfied)

        if fed:
            if unexpected and self.current_target_uid:
                # A severe adverse event is stronger evidence than a short
                # need-specific failure: remember that object for this run.
                self.blacklist[self.current_target_uid] = N_STEPS
            helped = {
                "energy": new_energy - state["energy"],
                "water": new_water - state["water"],
                "integrity": new_integrity - state["integrity"],
            }[ruling] > 0.01
            if helped:
                self.current_target_uid = None
            elif self.current_target_uid:
                self.blacklist[self.current_target_uid] = 120
                self.current_target_uid = None

            gain = (
                (new_energy - state["energy"])
                + (new_water - state["water"])
                + (new_integrity - state["integrity"])
            )
            success = 1.0 if gain > 0.01 else 0.0
            self.competence_estimate[ruling] = (
                0.8 * self.competence_estimate[ruling] + 0.2 * success
            )
        elif ruling == "affiliation":
            social_progress = (
                affiliation_satisfied
                or new_distance_to_other < state["distance_to_other"] - 1e-9
            )
            self.competence_estimate["affiliation"] = (
                0.95 * self.competence_estimate["affiliation"]
                + 0.05 * float(social_progress)
            )

        new_urges = {
            "energy": max(0.0, 1.0 - new_energy),
            "water": max(0.0, 1.0 - new_water),
            "integrity": max(0.0, 1.0 - new_integrity),
            "affiliation": affiliation_urge(new_distance_to_other),
        }
        new_sum_urges = sum(new_urges.values())
        urge_change = new_sum_urges - state["sum_urges"]

        # Each controller feeds and steps only its own nodenet.
        self.nn.set_modulator("base_sum_importance_of_intentions", new_sum_urges)
        self.nn.set_modulator("base_sum_urgency_of_intentions", new_sum_urges)
        self.nn.set_modulator(
            "base_competence_for_intention", self.competence_estimate[ruling]
        )
        self.nn.set_modulator("base_importance_of_intention", new_urges[ruling])
        self.nn.set_modulator("base_urgency_of_intention", new_urges[ruling])
        self.nn.set_modulator(
            "base_number_of_active_motives", max(1, state["active_motives"])
        )
        self.nn.set_modulator("base_number_of_expected_events", expected)
        self.nn.set_modulator("base_number_of_unexpected_events", unexpected)
        self.nn.set_modulator("base_urge_change", urge_change)
        self.nn.step()

        modulators = {
            key: value
            for key, value in self.nn.construct_modulators_dict().items()
            if key.startswith("emo_")
        }
        return {
            "x": self.wa.position[0],
            "y": self.wa.position[1],
            "energy": new_energy,
            "water": new_water,
            "integrity": new_integrity,
            "affiliation_urge": new_urges["affiliation"],
            "distance_to_other": new_distance_to_other,
            "ruling_motive": ruling,
            "motive_switched": state["switched"],
            "unexpected": unexpected,
            "action_requested": state["action_requested"],
            "action_succeeded": fed,
            "target_dist": state["target_distance"],
            "target_uid": self.current_target_uid,
            **modulators,
        }


controllers = {}
for key, spec in AGENT_SPECS.items():
    ok, nodenet_uid = runtime.new_nodenet(
        spec["name"],
        engine="dict_engine",
        worldadapter="Survivor",
        world_uid=WORLD_UID,
        uid=spec["uid"],
    )
    if not ok:
        raise RuntimeError("Could not create %s: %s" % (key, nodenet_uid))
    nodenet = runtime.nodenets[nodenet_uid]
    worldadapter = nodenet.worldadapter_instance
    worldadapter.name = spec["name"]
    worldadapter.position = spec["position"]
    # Different initial deficits prevent the two otherwise-identical
    # controllers from becoming permanently phase-locked after their first
    # meeting.  Both values remain comfortably inside the normal body range.
    worldadapter.energy = spec["energy"]
    worldadapter.water = spec["water"]
    worldadapter.datasources["body-energy"] = worldadapter.energy
    worldadapter.datasources["body-water"] = worldadapter.water
    # The position property already writes through to world.data; this explicit
    # assignment mirrors the baseline and makes the registration check obvious.
    world.data["agents"][nodenet_uid]["position"] = worldadapter.position
    controllers[key] = AgentController(
        key,
        nodenet,
        worldadapter,
        rng_seed=spec["seed"],
        home_position=spec["position"],
    )

registered_uids = set(world.data["agents"])
expected_uids = {spec["uid"] for spec in AGENT_SPECS.values()}
if not expected_uids.issubset(registered_uids):
    raise RuntimeError(
        "Both Survivor agents were not registered: %s" % sorted(registered_uids)
    )
print("Registered both Survivor agents:", sorted(expected_uids))
print("World agent records:", sorted(registered_uids))
for key, controller in controllers.items():
    print(key, "start:", controller.wa.position)


log = []
for step in range(N_STEPS):
    # Snapshot both live positions before either body advances, so motive inputs
    # are symmetric and neither agent gets an update-order advantage.
    position_a = controllers["agent_a"].wa.position
    position_b = controllers["agent_b"].wa.position
    distance_before = euclidean_distance(position_a, position_b)

    controllers["agent_a"].begin_step(position_b, distance_before)
    controllers["agent_b"].begin_step(position_a, distance_before)

    # Advance the two real Survivor bodies separately.
    controllers["agent_a"].advance_body()
    controllers["agent_b"].advance_body()

    distance_after = euclidean_distance(
        controllers["agent_a"].wa.position, controllers["agent_b"].wa.position
    )
    step_entry = {"step": step, "distance": distance_after}
    for key, controller in controllers.items():
        step_entry[key] = controller.finish_step(distance_after)
    log.append(step_entry)

    dead_agents = [
        key for key, controller in controllers.items() if controller.wa.is_dead
    ]
    if dead_agents:
        print("Agent(s) died at step", step, ":", ", ".join(dead_agents))
        break


output_dir = os.path.join(SCRIPT_DIR, "output")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "survivor_affiliation_log.json"), "w") as handle:
    json.dump(log, handle)

print("Simulation complete.", len(log), "steps logged.")
for key in controllers:
    final = log[-1][key]
    distribution = Counter(entry[key]["ruling_motive"] for entry in log)
    switches = sum(entry[key]["motive_switched"] for entry in log)
    print(
        key,
        "final demands:",
        round(final["energy"], 3),
        round(final["water"], 3),
        round(final["integrity"], 3),
    )
    print(key, "motive distribution:", distribution, "switches:", switches)
print(
    "Distance range:",
    round(min(entry["distance"] for entry in log), 1),
    "to",
    round(max(entry["distance"] for entry in log), 1),
)

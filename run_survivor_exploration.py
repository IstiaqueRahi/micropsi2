"""Variation B — a synthetic certainty / exploration demand.

Sibling of run_survivor.py (the baseline). Same scripted PSI decision policy and
the same three body demands, plus a *fourth*, purely-synthetic demand computed
entirely in this script: **certainty**. Dörner treats certainty ("I know what is
around me") as one of the five core drives; the repo's Survivor world adapter
never exposes it. Here the driver keeps an 8x8 map-familiarity grid, lets each
cell go "dark" the longer it has been since the agent was there, and turns the
FRACTION of the map currently dark into an urge that competes for ruling-motive
status through the *same* SWITCH_MARGIN hysteresis the body motives already use
(see the STALENESS_CEIL / CERTAINTY_CAP notes for why fraction-dark and not the
average staleness the task first suggests). The result: the agent periodically
leaves a working food/water loop to go re-familiarise itself with regions it has
not seen in a while.

Standalone: does NOT modify island.py, run_survivor.py, or any other variation's
files. Writes output/survivor_exploration_log.json.
"""
import sys, json, math, random, os, shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import micropsi_core.runtime as runtime

# seed the persistency dir from the repo's own demo data (island world + demo nodenet)
# on first run, so this script works straight out of a fresh clone.
DATA_DIR = os.path.join(SCRIPT_DIR, 'demo_run_data')
if not os.path.isdir(DATA_DIR):
    os.makedirs(os.path.join(DATA_DIR, 'nodenets'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'worlds'), exist_ok=True)
    for f in os.listdir(os.path.join(SCRIPT_DIR, 'demo_data', 'nodenets')):
        shutil.copy(os.path.join(SCRIPT_DIR, 'demo_data', 'nodenets', f), os.path.join(DATA_DIR, 'nodenets', f))
    for f in os.listdir(os.path.join(SCRIPT_DIR, 'demo_data', 'worlds')):
        shutil.copy(os.path.join(SCRIPT_DIR, 'demo_data', 'worlds', f), os.path.join(DATA_DIR, 'worlds', f))

# clean up any agent nodenet left over from a previous run of this script (idempotent re-runs)
stale_nn = os.path.join(DATA_DIR, 'nodenets', 'survivor_demo_agent.json')
if os.path.exists(stale_nn):
    os.remove(stale_nn)

runtime.initialize(persistency_path=DATA_DIR, resource_path=DATA_DIR)

WORLD_UID = 'd4b3f5740adc11e5b9fe20c9d087b4b7'
world = runtime.worlds[WORLD_UID]

# ---- Populate the island with real resources (it starts with only a Lightsource) ----
random.seed(7)
resource_specs = [
    ('Waterhole', (300, 500)),
    ('Waterhole', (900, 1400)),
    ('Champignon', (500, 350)),
    ('Champignon', (1150, 550)),
    ('Wirselkraut', (750, 650)),
    ('Juniper', (350, 900)),
    ('FlyAgaric', (620, 420)),   # trap: looks like food, damages integrity
    ('PalmTree', (1000, 300)),
    ('Stone', (450, 1100)),
    ('Boulder', (1300, 700)),
]
for objtype, pos in resource_specs:
    runtime.add_worldobject(WORLD_UID, objtype, pos, name=objtype)

print('World now has', len(world.objects), 'objects')
for uid, obj in world.objects.items():
    print('  ', uid, getattr(obj, 'structured_object_type', None), obj.position)

# ---- Create a fresh Survivor nodenet ----
ok, nodenet_uid = runtime.new_nodenet('SurvivorAgent', engine='dict_engine',
                                       worldadapter='Survivor', world_uid=WORLD_UID,
                                       uid='survivor_demo_agent')
print('new_nodenet ok:', ok, nodenet_uid)
nn = runtime.nodenets[nodenet_uid]
wa = nn.worldadapter_instance
wa.name = 'Survivor'
print('worldadapter instance:', type(wa).__name__)
print('start position:', wa.position)
print('start body state:', wa.energy, wa.water, wa.integrity)

# place agent at a deliberate starting point, away from everything, so it has to travel
wa.position = (650, 900)
world.data['agents'][nodenet_uid]['position'] = wa.position
print('repositioned start:', wa.position)

FOOD_TYPES = {'Champignon', 'Wirselkraut', 'Juniper', 'FlyAgaric'}
WATER_TYPES = {'Waterhole'}

from micropsi_core.world.island.island import ground_types

def is_walkable(x, y):
    idx = world.get_ground_at(x, y)
    return ground_types[idx]['agent_allowed']

def best_move_toward(pos, tx, ty):
    """Evaluate the 4 cardinal moves, keep only walkable ones, pick whichever
    most reduces distance to (tx,ty). Returns a loco_* key or None."""
    candidates = {
        'loco_east':  (pos[0]+50, pos[1]),
        'loco_west':  (pos[0]-50, pos[1]),
        'loco_north': (pos[0], pos[1]+50),
        'loco_south': (pos[0], pos[1]-50),
    }
    cur_d = (pos[0]-tx)**2 + (pos[1]-ty)**2
    best_key, best_d = None, cur_d
    for key, (nx, ny) in candidates.items():
        if not is_walkable(nx, ny):
            continue
        d = (nx-tx)**2 + (ny-ty)**2
        if d < best_d:
            best_key, best_d = key, d
    return best_key

# --- true walkable-space reachability from the agent's start, by flood fill ---
# Geometric nearness is not reachability: the 2nd Waterhole at (900,1400) has
# walkable shoreline within 45u but it's across open water with no path to it.
# The baseline never notices (it never roams that far); once this variation sends
# the agent exploring, an unreachable target-lock is lethal (it beelines,
# best_move_toward returns None, and the stuck-detector -- which needs an
# *attempted* move -- never trips). So flood-fill the walkable cells reachable
# from the start and use that for both resource targeting and frontier cells.
FLOOD_STEP = 32
_start_pt = (int(650 // FLOOD_STEP) * FLOOD_STEP, int(900 // FLOOD_STEP) * FLOOD_STEP)
_flood = set()
_frontier_bfs = [_start_pt]
while _frontier_bfs:
    px, py = _frontier_bfs.pop()
    if (px, py) in _flood or not (0 <= px < 2048 and 0 <= py < 2048) or not is_walkable(px, py):
        continue
    _flood.add((px, py))
    _frontier_bfs.extend([(px + FLOOD_STEP, py), (px - FLOOD_STEP, py),
                          (px, py + FLOOD_STEP), (px, py - FLOOD_STEP)])
print('flood-reachable walkable points (%du grid): %d' % (FLOOD_STEP, len(_flood)))

def _reachable_near(x, y, radius=48):
    return any((px - x) ** 2 + (py - y) ** 2 <= radius * radius for px, py in _flood)

APPROACHABLE = {uid for uid, o in world.objects.items()
                if getattr(o, 'structured_object_type', None) and _reachable_near(*o.position)}
_unreach = [(world.objects[u].data.get('type'), world.objects[u].position)
            for u in world.objects
            if u not in APPROACHABLE and getattr(world.objects[u], 'structured_object_type', None)]
if _unreach:
    print('unreachable resources excluded from targeting:', _unreach)

def nearest_of(types, position):
    best, bestd = None, float('inf')
    for uid, obj in world.objects.items():
        if uid not in APPROACHABLE:
            continue
        if getattr(obj, 'structured_object_type', None) in types:
            d = (obj.position[0]-position[0])**2 + (obj.position[1]-position[1])**2
            if d < bestd:
                best, bestd = obj, d
    return best, math.sqrt(bestd) if best else None

# ======================================================================
#  MAP-FAMILIARITY GRID  (the certainty / exploration demand lives here)
# ======================================================================
GRID = 8                       # 8x8 grid over the 2048x2048 world -> 256-unit cells
CELL = 2048 // GRID
# A cell counts as "gone dark" once it has been STALENESS_CEIL steps since the
# agent was last there. The task suggests 300-500; that band made "fraction dark"
# sit near 1.0 whenever the agent was in its resource loop, so certainty won
# ~40-90% of steps and the agent explored itself to death (the "dominating at
# 50%+" case the task says to retune). 750 stretches the ramp so a cell only
# goes dark well after the agent's last sweep of it -> certainty stays quiet
# while the agent keeps the map fresh and only spikes if it neglects exploring
# for a long stretch. Landed here by sweeping {650,700,750,800} x cap.
STALENESS_CEIL = 750
# The certainty urge is the FRACTION of reachable cells currently gone dark
# (rather than a normalised average staleness). Rationale, after trying the
# average: with only ~35 reachable cells and the agent's resource loop touching
# ~5 of them, average staleness sits near the ceiling essentially always, so
# certainty won ~90% of steps and the agent explored itself to death -- exactly
# the "dominating at 50%+" case the task says to fix in the urge formula. The
# fraction-dark version instead stays low while the agent sweeps the map often
# enough and only spikes if it neglects exploration for > STALENESS_CEIL steps.
# CERTAINTY_CAP keeps even that spike below the body-demand danger zone so a
# hungry agent's urge always reclaims ruling with buffer to spare. With
# STALENESS_CEIL=750, cap 0.26 gives certainty ~23% of steps and ~55 motive
# switches (matching the baseline); 0.30 pushed it to ~28-32%.
CERTAINTY_CAP = 0.26

def cell_of(x, y):
    return (min(GRID - 1, max(0, int(x) // CELL)), min(GRID - 1, max(0, int(y) // CELL)))

# A grid cell counts for the certainty demand only if the agent can actually get
# into it (>= 3 flood-reachable points). CELL_AIM is the centroid of that cell's
# reachable points -- a spot the agent can genuinely stand on (many cells' geometric
# centres sit in water even though the cell has usable shoreline).
_cell_pts = {}
for _px, _py in _flood:
    _cell_pts.setdefault((_px // CELL, _py // CELL), []).append((_px, _py))
REACHABLE_CELLS = []
CELL_AIM = {}
for _c, _pts in _cell_pts.items():
    if len(_pts) >= 3:
        REACHABLE_CELLS.append(_c)
        CELL_AIM[_c] = (sum(p[0] for p in _pts) / len(_pts), sum(p[1] for p in _pts) / len(_pts))
REACHABLE_CELLS.sort()
print('flood-reachable grid cells: %d / %d' % (len(REACHABLE_CELLS), GRID * GRID))

# Flood-connectivity is necessary but not sufficient: the southern peninsula is
# flood-connected yet the greedy pather can't route between it and the mainland
# resources (it heads straight into the water gap and wedges). So keep, as
# frontier targets, only cells from which the agent could still greedily walk to
# a food AND a water source -- i.e. cells it can explore without stranding itself.
_STEP = {'loco_east': (50, 0), 'loco_west': (-50, 0), 'loco_north': (0, 50), 'loco_south': (0, -50)}

def _greedy_reaches(sx, sy, tx, ty, max_steps=220):
    x, y = sx, sy
    for _ in range(max_steps):
        if math.hypot(tx - x, ty - y) < 45:
            return True
        mk = best_move_toward((x, y), tx, ty)
        if mk is None:
            return False
        dx, dy = _STEP[mk]
        x, y = x + dx, y + dy
    return False

def _serviceable(ax, ay):
    # the agent always targets the *nearest* food / water, so those specific ones
    # must be greedily reachable from here -- not merely "some food is reachable
    # by a long detour" (true of the peninsula via the far west, yet the agent
    # heads straight north into the water gap and wedges).
    nf, _ = nearest_of(FOOD_TYPES, (ax, ay))
    nw, _ = nearest_of(WATER_TYPES, (ax, ay))
    return bool(nf and nw
                and _greedy_reaches(ax, ay, *nf.position)
                and _greedy_reaches(ax, ay, *nw.position))

FRONTIER_CELLS = [c for c in REACHABLE_CELLS if _serviceable(*CELL_AIM[c])]
print('serviceable frontier cells (nearest food+water still greedily reachable): %d' % len(FRONTIER_CELLS))

# last_visited_step per cell, initialised "maximally stale" (large negative).
last_visited = {(cx, cy): -10 ** 9 for cx in range(GRID) for cy in range(GRID)}

def certainty_urge_at(step, extra_fresh_cell=None):
    """Fraction of the (serviceable) map that has gone dark -- staleness past
    STALENESS_CEIL -- clamped to CERTAINTY_CAP.  Rises as regions the agent hasn't
    seen in a while pile up; falls sharply as an exploration sweep re-lights them.
    `extra_fresh_cell` folds in the cell the agent just stepped into so an arrival
    shows up at once."""
    dark = 0
    for c in FRONTIER_CELLS:
        lv = step if c == extra_fresh_cell else last_visited[c]
        if step - lv > STALENESS_CEIL:
            dark += 1
    return min(CERTAINTY_CAP, dark / len(FRONTIER_CELLS))

def pick_frontier_cell(pos, step):
    """The frontier target when `certainty` rules: the stalest reachable cell,
    ties broken by nearest (so when the whole map is uniformly dark the agent
    spreads outward from where it is instead of darting to a random far corner).
    Skips the cell we're in and cells the greedy pather has already proven it
    can't reach."""
    here = cell_of(*pos)
    ax, ay = pos
    best, best_key = None, None
    for c in FRONTIER_CELLS:
        if c == here or c in unreachable_frontier:
            continue
        s = step - last_visited[c]
        if s <= 0:
            continue
        key = (-min(s, STALENESS_CEIL), (CELL_AIM[c][0] - ax) ** 2 + (CELL_AIM[c][1] - ay) ** 2)
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best

competence_estimate = {'energy': 0.5, 'water': 0.5, 'integrity': 0.5, 'certainty': 0.5}
last_position = None
stuck_counter = 0
escape_dir = None
escape_ttl = 0
current_ruling = 'energy'          # persistent motive commitment (models "selection threshold")
current_target_uid = None          # persistent object-target commitment (body motives)
current_explore_cell = None        # persistent frontier-cell commitment (certainty motive)
explore_stuck = 0                  # steps we've failed to progress toward the current frontier cell
unreachable_frontier = set()       # frontier cells the greedy pather couldn't get to -> don't retry
blacklist = {}                     # uid -> steps remaining before it's considered again (useless-for-need cooldown)
SWITCH_MARGIN = 0.12               # a rival urge must exceed the current motive by this much to take over
explore_arrivals = 0              # frontier cells actually reached
N = 1400
log = []

for step in range(N):
    blacklist = {k: v-1 for k, v in blacklist.items() if v-1 > 0}
    energy = wa.datasources['body-energy']
    water = wa.datasources['body-water']
    integrity = wa.datasources['body-integrity']

    # register the cell the agent is currently in as "just seen"
    last_visited[cell_of(wa.position[0], wa.position[1])] = step

    urges = {
        'energy': max(0.0, 1.0 - energy),
        'water': max(0.0, 1.0 - water),
        'integrity': max(0.0, 1.0 - integrity),
        'certainty': certainty_urge_at(step),   # synthetic 4th demand
    }
    sum_urges = sum(urges.values())

    # motive selection with hysteresis (PSI's "selection threshold": don't abandon
    # the current ruling motive for a rival unless the rival is clearly more pressing).
    # certainty competes here on exactly the same footing as energy/water/integrity.
    best_rival = max(urges, key=urges.get)
    if urges[best_rival] > urges[current_ruling] + SWITCH_MARGIN:
        current_ruling = best_rival
        current_target_uid = None      # motive changed -> drop both kinds of target
        current_explore_cell = None
    ruling = current_ruling
    n_active_motives = sum(1 for v in urges.values() if v > 0.08)

    # --- pick (and stick with) a target for the current ruling motive ---
    target = None
    dist = None
    if ruling == 'certainty':
        # frontier point, not a WorldObject. Pick the stalest reachable cell; if
        # the greedy pather can't make any progress toward it from here, retire
        # that cell (for the rest of the run) and try the next -- otherwise the
        # agent wedges against a terrain pinch for dozens of steps, as
        # best_move_toward returning None never trips the stuck detector below.
        current_target_uid = None
        tx = ty = None
        for _ in range(len(FRONTIER_CELLS)):
            if current_explore_cell is None:
                current_explore_cell = pick_frontier_cell(wa.position, step)
            if current_explore_cell is None:
                break
            cx, cy = CELL_AIM[current_explore_cell]
            if best_move_toward(wa.position, cx, cy) is None and cell_of(*wa.position) != current_explore_cell:
                unreachable_frontier.add(current_explore_cell)
                current_explore_cell = None
                continue
            tx, ty = cx, cy
            dist = math.hypot(tx - wa.position[0], ty - wa.position[1])
            break
    else:
        current_explore_cell = None
        type_set = WATER_TYPES if ruling == 'water' else (FOOD_TYPES if ruling == 'energy' else {'Wirselkraut'})
        target = world.objects.get(current_target_uid) if current_target_uid else None
        if target is None or target.uid in blacklist:
            candidates = {uid: o for uid, o in world.objects.items()
                          if getattr(o, 'structured_object_type', None) in type_set
                          and uid in APPROACHABLE and uid not in blacklist}
            target, _ = nearest_of(set(type_set), wa.position) if not candidates else (
                min(candidates.values(), key=lambda o: (o.position[0]-wa.position[0])**2 + (o.position[1]-wa.position[1])**2), None)
            current_target_uid = target.uid if target else None
        if target is not None:
            tx, ty = target.position
            dist = math.sqrt((tx-wa.position[0])**2 + (ty-wa.position[1])**2)
        else:
            tx = ty = None

    # --- navigation ---
    # also clear the action datatargets: manage_body_parameters leaves them set
    # while action_cooloff is ticking down, and a stale action_eat must not fire
    # during a certainty (explore) step, where there's nothing to consume.
    for k in ['loco_north', 'loco_south', 'loco_east', 'loco_west', 'action_eat', 'action_drink']:
        wa.datatargets[k] = 0

    if escape_ttl > 0:
        wa.datatargets[escape_dir] = 1
        escape_ttl -= 1
    elif ruling == 'certainty':
        if tx is not None:
            move_key = best_move_toward(wa.position, tx, ty)
            if move_key:
                wa.datatargets[move_key] = 1
        # no action_eat / action_drink -- arriving in the cell is the whole point
    elif target is not None:
        move_key = best_move_toward(wa.position, tx, ty)
        if move_key:
            wa.datatargets[move_key] = 1
        if dist is not None and dist < 45:
            if ruling == 'water':
                wa.datatargets['action_drink'] = 1
            else:
                wa.datatargets['action_eat'] = 1

    moved_intentionally = any(wa.datatargets[k] for k in ['loco_north', 'loco_south', 'loco_east', 'loco_west'])

    prev_integrity = integrity
    pos_before = wa.position

    # --- advance world physics for this agent ---
    wa.update_data_sources_and_targets()

    # certainty "arrival": reaching the target cell is itself what makes the
    # unknown known (drives its staleness to 0 next loop-top and lowers the urge)
    reached_frontier = (ruling == 'certainty' and current_explore_cell is not None
                        and cell_of(wa.position[0], wa.position[1]) == current_explore_cell)
    if reached_frontier:
        explore_arrivals += 1
        current_explore_cell = None
        explore_stuck = 0
    elif ruling == 'certainty' and current_explore_cell is not None:
        # made no headway toward the frontier cell -> retire it and pick another
        explore_stuck = explore_stuck + 1 if wa.position == pos_before else 0
        if explore_stuck >= 3:
            unreachable_frontier.add(current_explore_cell)
            current_explore_cell = None
            explore_stuck = 0
    else:
        explore_stuck = 0

    # stuck detection -> only counts if we actually tried to move and failed
    # (don't punish correctly holding position at a target during action cooldown)
    escaped_this_step = False
    if moved_intentionally and wa.position == pos_before and escape_ttl == 0:
        stuck_counter += 1
        if stuck_counter >= 4:
            escape_dir = random.choice(['loco_north', 'loco_south', 'loco_east', 'loco_west'])
            escape_ttl = 8
            stuck_counter = 0
            escaped_this_step = True
        elif random.random() < 0.15:
            current_target_uid = None       # give this target up occasionally, try a different one
            current_explore_cell = None
    else:
        stuck_counter = 0

    new_energy = wa.datasources['body-energy']
    new_water = wa.datasources['body-water']
    new_integrity = wa.datasources['body-integrity']

    unexpected = 1 if (new_integrity - prev_integrity) < -0.05 else 0   # got hurt = surprise
    fed = wa.datatarget_feedback.get('action_eat', 0) or wa.datatarget_feedback.get('action_drink', 0)
    expected = 1 if fed and not unexpected else 0
    if fed and ruling in ('energy', 'water', 'integrity'):
        helped = {'energy': new_energy - energy, 'water': new_water - water,
                  'integrity': new_integrity - integrity}[ruling] > 0.01
        if helped:
            current_target_uid = None       # satisfied -> release, re-evaluate next step
        elif current_target_uid:
            blacklist[current_target_uid] = 120   # this object doesn't serve the current need; avoid it a while
            current_target_uid = None

    # rolling competence estimate for the ruling need.
    if fed and ruling in ('energy', 'water', 'integrity'):
        gain = (new_energy - energy) + (new_water - water) + (new_integrity - integrity)
        success = 1.0 if gain > 0.01 else 0.0
        competence_estimate[ruling] = 0.8 * competence_estimate[ruling] + 0.2 * success
    elif ruling == 'certainty' and (reached_frontier or escaped_this_step):
        # no eat/drink success signal for a certainty step, so the proxy is:
        # did the agent actually reduce a frontier cell's staleness to ~0
        # (reached it) -> success; did it get wedged and have to bail -> failure.
        success = 1.0 if reached_frontier else 0.0
        competence_estimate['certainty'] = 0.8 * competence_estimate['certainty'] + 0.2 * success

    # recompute the four urges post-move (fold in the cell we just entered so an
    # arrival shows up immediately), then feed the REAL Doernerian equations.
    post_cell = cell_of(wa.position[0], wa.position[1])
    new_certainty = certainty_urge_at(step, extra_fresh_cell=post_cell)
    new_urges = {
        'energy': max(0.0, 1.0 - new_energy),
        'water': max(0.0, 1.0 - new_water),
        'integrity': max(0.0, 1.0 - new_integrity),
        'certainty': new_certainty,
    }
    new_sum_urges = sum(new_urges.values())
    urge_change = new_sum_urges - sum_urges  # positive = getting worse

    # --- feed the REAL Doernerian modulator equations (now four demands' worth) ---
    nn.set_modulator('base_sum_importance_of_intentions', new_sum_urges)
    nn.set_modulator('base_sum_urgency_of_intentions', new_sum_urges)
    nn.set_modulator('base_competence_for_intention', competence_estimate[ruling])
    nn.set_modulator('base_importance_of_intention', new_urges[ruling])
    nn.set_modulator('base_urgency_of_intention', new_urges[ruling])
    nn.set_modulator('base_number_of_active_motives', max(1, n_active_motives))
    nn.set_modulator('base_number_of_expected_events', expected)
    nn.set_modulator('base_number_of_unexpected_events', unexpected)
    nn.set_modulator('base_urge_change', urge_change)

    nn.step()  # runs DictPropagate, DictCalculate, DoernerianEmotionalModulators

    mods = {k: v for k, v in nn.construct_modulators_dict().items() if k.startswith('emo_')}

    log.append({
        'step': step, 'x': wa.position[0], 'y': wa.position[1],
        'energy': new_energy, 'water': new_water, 'integrity': new_integrity,
        'certainty_urge': new_certainty,                       # derived, not a body datasource
        'ruling_motive': ruling, 'unexpected': unexpected, 'target_dist': dist,
        'target_uid': current_target_uid,
        'explore_target': list(current_explore_cell) if current_explore_cell else None,
        'explore_target_xy': [round(tx, 1), round(ty, 1)] if (ruling == 'certainty' and tx is not None) else None,
        **mods
    })

    if wa.is_dead:
        print('Agent died at step', step)
        break

os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_exploration_log.json'), 'w') as f:
    json.dump(log, f)

# ---- coverage reconstructed from the trajectory alone (8x8 grid cells) ----
seen = set(cell_of(d['x'], d['y']) for d in log)

print('Simulation complete.', len(log), 'steps logged.')
print('Final demands:', log[-1]['energy'], log[-1]['water'], log[-1]['integrity'])
from collections import Counter
mc = Counter(d['ruling_motive'] for d in log)
print('Motive distribution:', mc)
print('  certainty share: %.1f%%  (baseline: 0%%)' % (100.0 * mc['certainty'] / len(log)))
motives = [d['ruling_motive'] for d in log]
switches = sum(1 for i in range(1, len(motives)) if motives[i] != motives[i-1])
print('Motive switches:', switches, ' (baseline ~55)')
print('Frontier cells reached:', explore_arrivals)
print('Grid cells visited: %d / 64 total,  %d / %d flood-reachable,  %d / %d serviceable  (baseline visits ~5)'
      % (len(seen), len(seen & set(REACHABLE_CELLS)), len(REACHABLE_CELLS),
         len(seen & set(FRONTIER_CELLS)), len(FRONTIER_CELLS)))
print('Unexpected events:', sum(d['unexpected'] for d in log))

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

def nearest_of(types, position):
    best, bestd = None, float('inf')
    for uid, obj in world.objects.items():
        if getattr(obj, 'structured_object_type', None) in types:
            d = (obj.position[0]-position[0])**2 + (obj.position[1]-position[1])**2
            if d < bestd:
                best, bestd = obj, d
    return best, math.sqrt(bestd) if best else None

competence_estimate = {'energy': 0.5, 'water': 0.5, 'integrity': 0.5}
last_position = None
stuck_counter = 0
escape_dir = None
escape_ttl = 0
current_ruling = 'energy'          # persistent motive commitment (models "selection threshold")
current_target_uid = None          # persistent target commitment, cleared on motive switch or arrival
blacklist = {}                     # uid -> steps remaining before it's considered again (useless-for-need cooldown)
SWITCH_MARGIN = 0.12               # a rival urge must exceed the current motive by this much to take over
N = 1400
log = []

for step in range(N):
    blacklist = {k: v-1 for k, v in blacklist.items() if v-1 > 0}
    energy = wa.datasources['body-energy']
    water = wa.datasources['body-water']
    integrity = wa.datasources['body-integrity']

    urges = {
        'energy': max(0.0, 1.0 - energy),
        'water': max(0.0, 1.0 - water),
        'integrity': max(0.0, 1.0 - integrity),
    }
    sum_urges = sum(urges.values())

    # motive selection with hysteresis (PSI's "selection threshold": don't abandon
    # the current ruling motive for a rival unless the rival is clearly more pressing)
    best_rival = max(urges, key=urges.get)
    if urges[best_rival] > urges[current_ruling] + SWITCH_MARGIN:
        current_ruling = best_rival
        current_target_uid = None   # motive changed -> re-pick a target
    ruling = current_ruling
    n_active_motives = sum(1 for v in urges.values() if v > 0.08)

    # --- pick (and stick with) a target object for the current ruling motive ---
    type_set = WATER_TYPES if ruling == 'water' else (FOOD_TYPES if ruling == 'energy' else {'Wirselkraut'})
    target = world.objects.get(current_target_uid) if current_target_uid else None
    if target is None or target.uid in blacklist:
        candidates = {uid: o for uid, o in world.objects.items()
                      if getattr(o, 'structured_object_type', None) in type_set and uid not in blacklist}
        target, _ = nearest_of(set(type_set), wa.position) if not candidates else (
            min(candidates.values(), key=lambda o: (o.position[0]-wa.position[0])**2 + (o.position[1]-wa.position[1])**2), None)
        current_target_uid = target.uid if target else None
    dist = None
    if target is not None:
        dist = math.sqrt((target.position[0]-wa.position[0])**2 + (target.position[1]-wa.position[1])**2)

    # --- navigation: move toward the (locked) target ---
    for k in ['loco_north', 'loco_south', 'loco_east', 'loco_west']:
        wa.datatargets[k] = 0

    if escape_ttl > 0:
        wa.datatargets[escape_dir] = 1
        escape_ttl -= 1
    elif target is not None:
        move_key = best_move_toward(wa.position, target.position[0], target.position[1])
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

    # stuck detection -> only counts if we actually tried to move and failed
    # (don't punish correctly holding position at a target during action cooldown)
    if moved_intentionally and wa.position == pos_before and escape_ttl == 0:
        stuck_counter += 1
        if stuck_counter >= 4:
            escape_dir = random.choice(['loco_north', 'loco_south', 'loco_east', 'loco_west'])
            escape_ttl = 8
            stuck_counter = 0
        elif random.random() < 0.15:
            current_target_uid = None  # give this target up occasionally, try a different one
    else:
        stuck_counter = 0

    new_energy = wa.datasources['body-energy']
    new_water = wa.datasources['body-water']
    new_integrity = wa.datasources['body-integrity']

    unexpected = 1 if (new_integrity - prev_integrity) < -0.05 else 0   # got hurt = surprise
    fed = wa.datatarget_feedback.get('action_eat', 0) or wa.datatarget_feedback.get('action_drink', 0)
    expected = 1 if fed and not unexpected else 0
    if fed:
        helped = {'energy': new_energy - energy, 'water': new_water - water,
                  'integrity': new_integrity - integrity}[ruling] > 0.01
        if helped:
            current_target_uid = None       # satisfied -> release, re-evaluate next step
        elif current_target_uid:
            blacklist[current_target_uid] = 120   # this object doesn't serve the current need; avoid it a while
            current_target_uid = None

    # rolling competence estimate for the ruling need
    if fed:
        gain = (new_energy - energy) + (new_water - water) + (new_integrity - integrity)
        success = 1.0 if gain > 0.01 else 0.0
        competence_estimate[ruling] = 0.8 * competence_estimate[ruling] + 0.2 * success

    new_urges = {
        'energy': max(0.0, 1.0 - new_energy),
        'water': max(0.0, 1.0 - new_water),
        'integrity': max(0.0, 1.0 - new_integrity),
    }
    new_sum_urges = sum(new_urges.values())
    urge_change = new_sum_urges - sum_urges  # positive = getting worse

    # --- feed the REAL Doernerian modulator equations ---
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
        'ruling_motive': ruling, 'unexpected': unexpected, 'target_dist': dist,
        'target_uid': current_target_uid,
        **mods
    })

    if wa.is_dead:
        print('Agent died at step', step)
        break

os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_log.json'), 'w') as f:
    json.dump(log, f)

print('Simulation complete.', len(log), 'steps logged.')
print('Final demands:', log[-1]['energy'], log[-1]['water'], log[-1]['integrity'])
from collections import Counter
print('Motive distribution:', Counter(d['ruling_motive'] for d in log))
print('Unexpected events:', sum(d['unexpected'] for d in log))

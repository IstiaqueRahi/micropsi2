"""Variation A — resource scarcity / depletion.

Sibling of run_survivor.py (the baseline). Same scripted PSI decision policy,
but the island's Waterhole / Champignon / Wirselkraut / Juniper resources now
carry a finite, slowly-regenerating supply, so the agent is periodically forced
to abandon a drained source and travel to another one.

Standalone: this variation does NOT modify island.py. The depletable-supply
behaviour is grafted onto the individual world-object instances at runtime by
make_depletable() below (attaches supply bookkeeping + wraps the resource's
yielding action). Nothing here depends on any other variation's files.

Writes output/survivor_depletion_log.json.
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

# --- runtime "depletable resource" behaviour, grafted onto instances (no island.py edits) ---
CONSUME_PER_USE = 1.0
# which action actually draws on each resource's supply (the one with a non-zero
# body delta -- e.g. Champignon.action_drink returns (True,0,0,0) and is left alone)
EFFECTIVE_ACTION = {'Waterhole': 'action_drink', 'Champignon': 'action_eat',
                    'Wirselkraut': 'action_eat', 'Juniper': 'action_eat'}

def make_depletable(obj, max_supply=7.0, regen_rate=0.022):
    """Give one world-object instance a finite, regenerating supply reservoir.

    supply is measured in "servings"; the effective eat/drink draws
    CONSUME_PER_USE and fails cleanly -- (False, 0, 0, 0), the same "can't do"
    shape Survivor.manage_body_parameters already handles -- once less than a
    serving is left (an exact "supply <= 0" is never observed: the agent camps
    on a resource and acts every few steps, and regen lifts supply a hair above
    zero between uses). max_supply / regen_rate are sized so an actively-used
    resource runs dry after ~140 steps of feeding and refills in ~320 idle
    steps: slow enough to force travel, fast enough that nothing is permanent.
    regenerate() is called once per simulation step from the driver loop below
    (this script never goes through World.step()).
    """
    obj.supply = obj.max_supply = max_supply
    obj.regen_rate = regen_rate
    obj.consume_per_use = CONSUME_PER_USE
    action_name = EFFECTIVE_ACTION[obj.structured_object_type]
    base_action = getattr(obj, action_name)

    def wrapped_action():
        base = base_action()
        if not base[0]:                                  # base itself says "can't"
            return base
        if obj.supply < obj.consume_per_use:             # not enough left for a serving -> depleted
            return False, 0, 0, 0
        obj.supply = max(0.0, obj.supply - obj.consume_per_use)
        return base

    setattr(obj, action_name, wrapped_action)
    obj.regenerate = lambda: setattr(obj, 'supply', min(obj.max_supply, obj.supply + obj.regen_rate))

# ---- Populate the island with real resources (it starts with only a Lightsource) ----
random.seed(7)
resource_specs = [
    ('Waterhole', (300, 500)),
    ('Waterhole', (1350, 650)),   # 2nd water source: reachable, ~1060 units east of the first,
                                  # pairs with Champignon @ (1150,550) as a self-sufficient 2nd hub
                                  # (the baseline's (900,1400) sat in impassable water -- unreachable
                                  # the moment depletion forced the agent off the first waterhole)
    ('Champignon', (500, 350)),
    ('Champignon', (1150, 550)),
    ('Wirselkraut', (750, 650)),
    ('Juniper', (350, 900)),
    ('FlyAgaric', (620, 420)),   # trap: looks like food, damages integrity -- left infinite on purpose
    ('PalmTree', (1000, 300)),
    ('Stone', (450, 1100)),
    ('Boulder', (1300, 700)),
]
# track the depletable resources under stable labels (type + per-type index,
# deterministic creation order) so the log and the visualisers can refer to them
# across runs even though the world assigns fresh uids every time.
DEPLETABLE_TYPES = {'Waterhole', 'Champignon', 'Wirselkraut', 'Juniper'}
tracked_resources = []          # [(label, obj)] in creation order
uid_to_label = {}
_type_counts = {}
for objtype, pos in resource_specs:
    ok, obj_uid = runtime.add_worldobject(WORLD_UID, objtype, pos, name=objtype)
    if objtype in DEPLETABLE_TYPES:
        obj = world.objects[obj_uid]
        make_depletable(obj)
        idx = _type_counts.get(objtype, 0)
        _type_counts[objtype] = idx + 1
        label = '%s%d' % (objtype, idx)
        tracked_resources.append((label, obj))
        uid_to_label[obj_uid] = label

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

def has_supply(obj):
    """False only for a depletable resource that is completely empty. Note this
    is deliberately more lenient than make_depletable's own wrapped action (which
    needs a whole serving): the agent stays committed while a trickle remains,
    walks up, and the failed attempt registers as a genuine depletion surprise --
    after which it goes on the blacklist so the agent actually travels elsewhere."""
    return getattr(obj, 'supply', float('inf')) > 0

def nearest_of(types, position, exclude=()):
    """Nearest object of one of `types`, skipping excluded uids and empty resources."""
    best, bestd = None, float('inf')
    for uid, obj in world.objects.items():
        if uid in exclude or not has_supply(obj):
            continue
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
visited_resources = set()          # labels of depletable resources the agent has actually fed from
depletion_events = 0               # eat/drink attempts that failed because the resource was empty
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
    # break the lock if the target went stale: blacklisted as wrong-for-need, or
    # drained dry since we committed to it (another agent, or just a visit that
    # emptied it, could do that while we're still en route).
    if target is not None and (target.uid in blacklist or not has_supply(target)):
        target, current_target_uid = None, None
    if target is None:
        target, _ = nearest_of(set(type_set), wa.position, exclude=set(blacklist))
        if target is None:   # everything of this type is blacklisted -> ignore the blacklist, but still skip empties
            target, _ = nearest_of(set(type_set), wa.position)
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

    # remember what we're about to attempt, so that after the step we can tell a
    # "resource was empty" failure apart from a cooldown / wrong-object no-op
    attempted_action = ('action_drink' if wa.datatargets['action_drink']
                        else ('action_eat' if wa.datatargets['action_eat'] else None))
    cooloff_ready = wa.action_cooloff <= 0

    prev_integrity = integrity
    pos_before = wa.position

    # --- advance world physics for this agent ---
    wa.update_data_sources_and_targets()

    # slow regeneration of every depletable resource (done here rather than in
    # World.step(), which this script never calls)
    for _label, _obj in tracked_resources:
        _obj.regenerate()

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

    fed = wa.datatarget_feedback.get('action_eat', 0) or wa.datatarget_feedback.get('action_drink', 0)

    # --- expectation violations, fed to Doerner's base_number_of_unexpected_events ---
    integrity_surprise = (new_integrity - prev_integrity) < -0.05        # FlyAgaric trap: "food" that hurts
    # we walked up to a resource we'd committed to and the eat/drink came back
    # empty-handed because it had run dry (not cooldown, not wrong object type):
    # exactly the appraisal-relevant surprise base_number_of_unexpected_events exists for
    depletion_surprise = bool(
        attempted_action and cooloff_ready and not fed
        and target is not None and dist is not None and dist < 45
        and EFFECTIVE_ACTION.get(getattr(target, 'structured_object_type', None)) == attempted_action
        and getattr(target, 'supply', float('inf')) < getattr(target, 'consume_per_use', 0))
    unexpected = 1 if (integrity_surprise or depletion_surprise) else 0
    surprise_kind = 'integrity' if integrity_surprise else ('depletion' if depletion_surprise else None)
    expected = 1 if fed and not unexpected else 0

    if depletion_surprise:
        depletion_events += 1
        if current_target_uid:
            blacklist[current_target_uid] = 160   # ran dry under us -> give it time to regrow before reconsidering
        current_target_uid = None

    if fed and target is not None and target.uid in uid_to_label:
        visited_resources.add(uid_to_label[target.uid])

    if fed:
        helped = {'energy': new_energy - energy, 'water': new_water - water,
                  'integrity': new_integrity - integrity}[ruling] > 0.01
        if integrity_surprise and current_target_uid:
            blacklist[current_target_uid] = 150   # that "food" wrecked our integrity (FlyAgaric) -> steer away a while
            current_target_uid = None
        elif helped:
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
        'ruling_motive': ruling, 'unexpected': unexpected, 'surprise_kind': surprise_kind,
        'target_dist': dist, 'target_uid': current_target_uid,
        'supply': {label: round(obj.supply, 3) for label, obj in tracked_resources},
        **mods
    })

    if wa.is_dead:
        print('Agent died at step', step)
        break

os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_depletion_log.json'), 'w') as f:
    json.dump(log, f)

print('Simulation complete.', len(log), 'steps logged.')
print('Final demands:', log[-1]['energy'], log[-1]['water'], log[-1]['integrity'])
from collections import Counter
print('Motive distribution:', Counter(d['ruling_motive'] for d in log))
print('Unexpected events:', sum(d['unexpected'] for d in log),
      '(' + ', '.join('%s=%d' % (k, v) for k, v in
                      Counter(d['surprise_kind'] for d in log if d['surprise_kind']).items()) + ')')
print('Distinct resources fed from:', len(visited_resources), sorted(visited_resources))
print('Depletion-driven failed attempts:', depletion_events)
print('Final supply levels:', {label: round(obj.supply, 2) for label, obj in tracked_resources})

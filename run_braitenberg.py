import sys, json, os, shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import micropsi_core.runtime as runtime

DATA_DIR = os.path.join(SCRIPT_DIR, 'demo_run_data')
if not os.path.isdir(DATA_DIR):
    os.makedirs(os.path.join(DATA_DIR, 'nodenets'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'worlds'), exist_ok=True)
    for f in os.listdir(os.path.join(SCRIPT_DIR, 'demo_data', 'nodenets')):
        shutil.copy(os.path.join(SCRIPT_DIR, 'demo_data', 'nodenets', f), os.path.join(DATA_DIR, 'nodenets', f))
    for f in os.listdir(os.path.join(SCRIPT_DIR, 'demo_data', 'worlds')):
        shutil.copy(os.path.join(SCRIPT_DIR, 'demo_data', 'worlds', f), os.path.join(DATA_DIR, 'worlds', f))

runtime.initialize(persistency_path=DATA_DIR, resource_path=DATA_DIR)

WORLD_UID = 'd4b3f5740adc11e5b9fe20c9d087b4b7'
NODENET_UID = 'b6f40e6417ee11e4bbe920c9d087b4b7'   # the repo's own demo Braitenberg agent
runtime.load_nodenet(NODENET_UID)

nn = runtime.nodenets[NODENET_UID]
w = runtime.worlds[WORLD_UID]

N = 400
log = []
for i in range(N):
    w.step()
    runtime.step_nodenets_in_world(WORLD_UID, nodenet_uid=NODENET_UID, steps=1)
    pos = w.data['agents'][NODENET_UID]['position']
    mods = {k: v for k, v in nn.construct_modulators_dict().items() if k.startswith('emo_')}
    log.append({'step': i, 'x': pos[0], 'y': pos[1], **mods})

os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, 'output', 'sim_log.json'), 'w') as f:
    json.dump(log, f)

print('done. steps logged:', len(log))
print('final position:', log[-1]['x'], log[-1]['y'])

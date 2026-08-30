# Running the PSI island simulations locally (VS Code / any machine)

This is Joscha Bach's own `micropsi2` reference implementation of Dörner's PSI
theory (github.com/joschabach/micropsi2), patched to run on a modern Python
(3.10+), plus two driver scripts that build and run actual agents on the
Dörner-Island world and log their real `DoernerianEmotionalModulators` state.

## 1. Clone the repo

```bash
git clone https://github.com/joschabach/micropsi2.git
cd micropsi2
```

## 2. Apply the patch

The original repo is from ~2015 and has a few Python 3.4-era incompatibilities,
plus two genuine bugs in `island.py` (agent could never move north; drinking
silently called the wrong method). The patch fixes all of that:

```bash
git apply /path/to/micropsi2_compat_and_bugfixes.patch
```

(If `git apply` complains about whitespace, try `git apply --whitespace=fix
/path/to/micropsi2_compat_and_bugfixes.patch`.)

## 3. Install dependencies

The repo's own `requirements.txt` pins very old versions (Theano 0.7,
CherryPy 3.6, pycrypto) that won't build on a modern Python. You don't need
any of those for this — the demo agents use the pure-Python `dict_engine`,
not the Theano engine, and we're running headless (no CherryPy web server
needed). Install the modern equivalents instead:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r /path/to/requirements-modern.txt
```

## 4. Copy the driver scripts into the repo root

Copy these five files into the `micropsi2/` folder you just cloned (same
level as `micropsi_core/`, `micropsi_server/`, etc.):

- `run_braitenberg.py` — simple demo: the repo's own light-seeking vehicle
- `make_braitenberg_overview.py` / `make_braitenberg_gif.py` — visualize it
- `run_survivor.py` — the richer agent: real energy/water/integrity demands,
  competing motives, a poison-mushroom trap, real navigation
- `make_survivor_overview.py` / `make_survivor_gif.py` — visualize it

All six scripts are self-locating (they figure out paths relative to their
own location), so as long as they sit directly in the repo root, no path
editing is needed.

## 5. Run it

```bash
# quick sanity check first (takes a few seconds)
python run_braitenberg.py
python make_braitenberg_overview.py
python make_braitenberg_gif.py

# the actual interesting one (takes ~30s-1min)
python run_survivor.py
python make_survivor_overview.py
python make_survivor_gif.py
```

Everything gets written to `micropsi2/output/`:
- `sim_log.json` / `survivor_log.json` — raw step-by-step logs (position,
  demands, all `emo_*` modulator values) — open these directly if you want
  to plot something different yourself
- `*_overview.png` — static full-run charts
- `*.gif` — the animations

## What `run_survivor.py` actually does

The real world physics and the real emotion equations
(`DoernerianEmotionalModulators` in `micropsi_core/nodenet/stepoperators.py`
— worth reading directly, it's a direct citation of Bach's *Principles of
Synthetic Intelligence*, p.185) are untouched, genuine micropsi2 code. What
the script adds is a from-scratch Python decision layer standing in for what
would otherwise be a hand-built spreading-activation node network:

- reads the agent's `body-energy` / `body-water` / `body-integrity`
  datasources every step
- picks a single "ruling motive" using **hysteresis** (won't abandon the
  current motive for a rival unless the rival's urge clearly exceeds it —
  this is a direct implementation of Dörner's "selection threshold" concept)
- locks onto one target object and commits to it (won't flip-flop between
  two similarly-distant resources)
- does terrain-aware greedy movement (checks `ground_types[...]['agent_allowed']`
  before committing to a direction, so it doesn't get wedged against a
  blocked corner tile)
- feeds the real modulator equations with real numbers each step:
  `base_importance_of_intention`, `base_urgency_of_intention`,
  `base_competence_for_intention`, `base_number_of_active_motives`,
  `base_urge_change`, `base_number_of_expected/unexpected_events`

You'll want to tune `resource_specs` (object placement) and `SWITCH_MARGIN`
(the hysteresis threshold) in `run_survivor.py` if you want to see different
dynamics — e.g. a lower `SWITCH_MARGIN` reintroduces motive flip-flopping,
which is a good way to see viscerally why Dörner thought selection threshold
was necessary in the first place.

## Bugs found in the original repo (fixed by the patch)

1. **`island.py`, `Survivor.update_data_sources_and_targets`**: the y-component
   of the movement vector used `-(50 * -loco_south)`, which equals
   `+50 * loco_south` — identical in sign to the `loco_north` term. The agent
   could physically never move "north."
2. **`island.py`, `Survivor.manage_body_parameters`**: regardless of which
   action fired, the code always called `nearest_worldobject.action_eat()`
   instead of `getattr(nearest_worldobject, datatarget)()`. `action_drink`
   silently never worked — every attempted drink secretly called `action_eat`
   on the target instead.
3. Three Python 3.4→3.12 compatibility issues: `collections.MutableSet`
   moved to `collections.abc`, and two uses of the removed
   `inspect.getargspec` (replaced with `inspect.getfullargspec`, which has
   the same relevant `.args`/`.defaults` attributes).

# Hybrid PSI survival and predation experiment (protocol 2.0)

This package implements the preregistered multi-agent experiment in
`psi_survival/`. It is a hybrid model: raw emotional modulation comes from the
pinned MicroPsi2 `DoernerianEmotionalModulators` operator, while the physiology,
belief model, planner, combat, scheduler, and normalized adapters are explicit
project extensions. The named emotions are fuzzy classifications of model
coordinates, not claims about subjective experience.

## Reproducibility boundary

The baseline configuration pins:

- operator file commit `8b9397f965fec5b8e28c8f0007d3da0336440972`;
- operator SHA-256 `adee2f202381665d752b7a913d860464eea48775ea36a1f666da9b9d71cdf0e9`;
- supplied Cai et al. PDF SHA-256
  `4f4e56bd4369eb9fdc4c0057f98a1357e96f07a398d928ff181d22a8fcd2227e`;
- adapter `hybrid-adapter-v1` and classifier `cai-table2-literal-v1`.

Every run records the complete configuration, initial world, keyed seed,
repository revision, actual source hashes, per-agent outcomes, and compute
duration. A source mismatch is fatal. The operator's persistent state is
initialized once; only its declared per-cycle inputs are refreshed.

## Commands

Run these from the repository root with the modern environment activated:

```bash
python -m psi_survival.cli write-config output/psi-v2/baseline.json
python -m unittest discover -s tests_psi_survival -v

# Predeclared demonstration, including the full saved trace
python -m psi_survival.cli simulate \
  --config output/psi-v2/baseline.json \
  --seed 900001 --trace full \
  --output output/psi-v2/demonstration.json.gz
python -m psi_survival.cli render \
  --result output/psi-v2/demonstration.json.gz \
  --output-dir output/psi-v2/demonstration

# One complete 18-cell pilot pass: 10 seed blocks, 180 runs
python -m psi_survival.cli pilot \
  --config output/psi-v2/baseline.json \
  --output-dir output/psi-v2/study --workers 8

# After any calibration change, rerun the final primary pair (20 runs) in a
# fresh output directory before sizing; do not reuse obsolete pilot variance.
python -m psi_survival.cli pilot \
  --config output/psi-v2/final-calibrated.json \
  --output-dir output/psi-v2/final-primary-pilot --workers 8 --primary-only

# Calculate n from the final-configuration primary pilot pairs and freeze it
python -m psi_survival.cli size-evaluation \
  --config output/psi-v2/baseline.json \
  --pilot-summary output/psi-v2/study/pilot/matrix-summary.json \
  --output output/psi-v2/freeze-manifest.json

# This rejects any implementation, operator, adapter, classifier, config, or
# seed-registry mismatch against the freeze manifest.
python -m psi_survival.cli evaluate \
  --config output/psi-v2/baseline.json \
  --freeze-manifest output/psi-v2/freeze-manifest.json \
  --output-dir output/psi-v2/study --workers 8
python -m psi_survival.cli analyze \
  --matrix-summary output/psi-v2/study/evaluation/matrix-summary.json \
  --output output/psi-v2/primary-analysis.json
```

`--workers` should be calibrated to the machine. Rendering reads a saved run
and cannot change simulation state. `render --ema VALUE` changes only the
display overlay.

## Focused design

The study has 18 cells:

| Family | Cells |
| --- | ---: |
| Dynamic, fixed, and three single-connection ablations × combat on/off; 12 agents, regeneration 1 | 10 |
| Dynamic/fixed × regeneration 0.25/4; 12 agents, combat on | 4 |
| Dynamic/fixed × 6/24 agents; regeneration 1, combat on | 4 |

Pilot seeds are 1001–1010. Evaluation seeds are the first `n` consecutive
integers beginning at 2001. Demonstration seed 900001 is disjoint from both.
The minimum evaluation is `18 × 30 = 540` runs. A complete pilot pass is 180
runs, but calibration changes invalidate earlier pilot variance estimates.

The primary contrast is dynamic minus fixed modulation at 12 agents, baseline
regeneration, and combat enabled. For seed block `s`, the run outcome is

```text
Y_s = mean_i(min(T_is, 3600))
```

over every agent initially present. A horizon survivor contributes exactly
3,600 seconds alive in the observation window. Crashed or incomplete runs are
invalid and are never treated as survivors.

After rerunning the primary pilot pair under the final calibrated
configuration, let `s_D,pilot` be the sample standard deviation of the ten
paired differences. The implementation selects the smallest integer `n >= 30`
such that

```text
t(0.975, n-1) * s_D,pilot / sqrt(n) <= 180 seconds.
```

That `n` is frozen and used in all 18 cells. The primary report gives the 95%
paired-t interval and retains all seed-level differences. The planning target
does not guarantee achieved precision and does not provide equal precision for
secondary contrasts. The analysis artifact also reports paired intervals for
the declared dynamic/fixed sensitivity comparisons, dynamic/ablation
comparisons, and combat-on/off comparisons across the recorded secondary
endpoints; these are explicitly labeled secondary.

## Pilot and evaluation gates

Pilot calibration may alter resource capacity/regeneration, passive
depletion, movement costs, resource spacing, and execution budgets. It must
not tune toward a desired psychological result. Changing operator equations,
principal utility weights, or normalization mappings creates a separately
versioned model.

Before frozen evaluation:

1. Run all automated tests and controlled fixtures.
2. Establish physical survival, genuine resource constraint, a decision change
   after inspection, behavioral consequences for `R`, `S`, and `Q`, and a
   constructed case where hunting wins the shared utility comparison.
3. Calibrate using only pilot seeds, then rerun the final primary pilot pair.
4. Freeze the validated implementation, final configuration, source hashes,
   analysis definitions, pilot-derived `n`, and seed registry.
5. Do not add evaluation seeds after inspecting their outcomes.

No attack in ordinary evaluation is an admissible finding. It is an
implementation failure only if the constructed hunting fixture cannot select
hunting when the declared common utility makes it best.

## Engine and policy boundary

Each tick uses one shared snapshot, perception with one-tick-lagged controls,
one upstream operator execution, belief-only planning, simultaneous movement,
depletion deaths, matched probabilistic duels, proportional noncombat
interactions, regeneration, and logging. Policies can use stock and type that
inspection or consumption legitimately revealed. They cannot query hidden
authoritative fields.

Planner uncertainty is represented by mutually exclusive outcome leaves.
Resource type, availability, physiology, one deliberate or threat-driven duel,
and subsequent short-sequence actions share the same leaves for both expected
deficit and death probability. Each leaf has one terminal death indicator and
the leaf probabilities sum to one. Hunting includes possible corpse consumption
on the next interaction opportunity; inspection similarly includes later
consumption when the revealed type is useful.

The classifier implements the literal Cai et al. Table 2 profiles. Its direct
paper reproduction test bypasses the hybrid adapter and recovers the rounded
Table 4 scores to absolute tolerance 0.001. Displayed labels use the declared
score and runner-up-gap thresholds; raw scores and literal argmax are retained.

The source-discrepancy register remains explicit: Table 2's low Angry
selection threshold is preserved; Cai Eq. 10's direction relative to its prose
and Eq. 12's securing-threshold direction are not silently corrected; and the
MicroPsi2 resolution equation is used instead of being blended with Cai's
generative equation. Any paper-only reproduction or corrected profile must use
a new model/classifier version.

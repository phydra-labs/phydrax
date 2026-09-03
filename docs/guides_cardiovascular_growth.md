# Cardiovascular growth and sarcomere energetics

The cardiovascular mechanics package separates three time scales and two kinds of
state transition:

1. fast electromechanics produces tensor-valued samples over complete cardiac
   cycles;
2. complete cycles are reduced to fixed-shape homeostatic sufficient statistics;
3. slow growth is proposed, certified, and atomically committed;
4. an anatomy or unloaded-reference replacement is a separate, discrete epoch
   transaction; and
5. mean-field cross-bridge kinetics advances on its own fast time step with an
   explicit chemical and mechanical power ledger.

A failed certificate never partially changes committed state.

## Units and signs

The mechanics kernels use the platform scale directly.

| Quantity | Kernel unit | Exact SI factor |
| --- | --- | ---: |
| length | `mm` | `1e-3 m` |
| time | `ms` | `1e-3 s` |
| mass | `mg` | `1e-6 kg` |
| voltage | `mV` | `1e-3 V` |
| pressure/stress | `kPa` | `1e3 Pa` |
| volume | `mm3` | `1e-9 m3` |
| energy | `mg*mm2/ms2` | `1e-6 J` |
| power density | `kPa/ms` | `1e6 W/m3` |
| chemical amount density | `pmol/mm3` | `1e-3 mol/m3` |

Positive sarcomere shortening velocity means shortening and positive external
mechanical power output. Negative velocity means externally driven lengthening.
Growth stimulus tensors and homeostatic targets share the caller-declared stress
unit, normally `kPa`. Growth gains consequently have units of logarithmic strain
per normalized stimulus per `ms`.

## Multiplicative finite growth

A committed `LogGrowthTensorState` stores a symmetric tensor H_g at every material
point. The growth and elastic factors are

```text
F_g = exp(H_g)
F_e = F F_g^-1
F   = F_e F_g .
```

The native Hermitian spectral exponential evaluates both `exp(H_g)` and
`exp(-H_g)`. Thus every finite log tensor gives a symmetric positive-definite
`F_g`, rather than relying on determinant clipping after an unconstrained update.
`GrowthKinematics` reports all three Jacobians, the reconstruction error,
positivity, and finiteness.

### Geometry is declared, not inferred

`GrowthPlan.reference_directions` has shape `(material_points, channels,
dimension)`. Each direction is normalized during planning and identified by a
stable `direction_id`. The cycle reducer projects a symmetric stimulus tensor S
using nᵀ S n. The package never labels a direction radial, circumferential,
longitudinal, or transmural and never assumes a sphere or ellipsoid. Anatomy owns
those meanings and supplies the vectors in its reference configuration.

A minimal setup is:

```python
import numpy as np

from phydrax.applications.cardiovascular.mechanics import (
    GrowthPlan,
    GrowthReferenceEpoch,
    initialize_growth_cycle_accumulator,
    initialize_growth_state,
    prepare_growth,
)

point_ids = ("lv-q0", "lv-q1")
direction_ids = ("fiber", "sheet")
directions = np.asarray(
    [
        [[0.91, 0.35, 0.22], [-0.18, 0.77, 0.61]],
        [[0.80, -0.52, 0.30], [0.12, 0.62, -0.78]],
    ]
)
plan = GrowthPlan(
    point_ids,
    direction_ids,
    directions,
    homeostatic_targets=np.full((2, 2), 12.0),
    stimulus_scales=np.full((2, 2), 4.0),
    growth_gains=np.full((2, 2), 1.0e-8),
    minimum_cycles=3,
    deadband=0.02,
    maximum_log_increment=0.01,
)
epoch = GrowthReferenceEpoch(
    "patient-anatomy-17",
    "unloaded-reference-4",
    "ventricular-tets-12",
    point_ids,
)
prepared = prepare_growth(plan, epoch)
state = initialize_growth_state(prepared)
cycles = initialize_growth_cycle_accumulator(prepared)
```

## Cycle aggregation and homeostasis

`aggregate_growth_cycle` accepts strictly increasing sample times and symmetric
stimulus tensors shaped `(samples, material_points, dimension, dimension)`. It
uses a time-weighted trapezoidal mean, so irregular fast-step spacing does not
bias the slow stimulus. `accumulate_growth_cycle` accepts each `cycle_index`
exactly once and in order. The accumulator retains only integrated directional
stimulus, total duration, cycle count, and the last cycle index; its arrays remain
fixed shape.

For observed cycle mean s, target s_h, scale s_0, and declared deadband d,
the effective deadband is the larger of d and a dtype-scaled projection-roundoff
bound. The channel drive is

```text
e = (s - s_h) / s_0
r = gain * sign(e) * max(abs(e) - effective_deadband, 0).
```

The tensor rate is the sum of `r * outer(direction, direction)` over declared
channels. At homeostasis it is exactly zero, including when unit-vector
projection differs from its mathematical target only by floating-point
roundoff. A proposal is inadmissible until
`minimum_cycles` complete cycles have been accumulated.

```python
from phydrax.applications.cardiovascular.mechanics import (
    accumulate_growth_cycle,
    aggregate_growth_cycle,
    commit_growth_step,
    evaluate_growth_proposal,
    propose_growth_step,
    refine_growth_proposal,
)

cycle = aggregate_growth_cycle(
    prepared,
    0,
    sample_times_ms,
    symmetric_stimulus_tensors,
)
cycles = accumulate_growth_cycle(prepared, cycles, cycle)
proposal = propose_growth_step(prepared, state, cycles, 86_400_000.0)
evidence = evaluate_growth_proposal(prepared, state, proposal)
while not bool(evidence.passed) and bool(evidence.refinement_available):
    proposal = refine_growth_proposal(prepared, state, cycles, proposal)
    evidence = evaluate_growth_proposal(prepared, state, proposal)
result = commit_growth_step(prepared, state, proposal, evidence)
state = result.state
```

Refinement halves the effective slow step while retaining the original requested
horizon and source-state identity. Evidence checks the Frobenius norm of the log
increment, the allowed log-spectrum bound, symmetry, finite reconstruction, and
positive growth Jacobian. Exhausting the declared refinement budget is a hard
failure. The caller advances the remaining horizon only after committing each
accepted substep and obtaining the cycle policy required by its protocol.

## Anatomy and reference epoch transactions

`PreparedGrowth` is valid only for the exact tuple of anatomy, unloaded reference,
topology, and stable material-point IDs in its `GrowthReferenceEpoch`. A changed
reference is not a differentiable deformation of the old prepared object.
Construct a complete target `GrowthPlan` and `PreparedGrowth`, then provide a
fixed transfer matrix:

```python
from phydrax.applications.cardiovascular.mechanics import (
    commit_growth_epoch_transfer,
    evaluate_growth_epoch_transfer,
    propose_growth_epoch_transfer,
)

candidate = propose_growth_epoch_transfer(
    old_prepared,
    old_state,
    rebuilt_target_prepared,
    target_from_source_weights,
)
evidence = evaluate_growth_epoch_transfer(old_prepared, old_state, candidate)
transition = commit_growth_epoch_transfer(
    old_prepared,
    old_state,
    old_cycle_accumulator,
    candidate,
    evidence,
)
```

Each target row must be a convex partition of unity. Mapping the symmetric log
tensor, rather than averaging `F_g`, preserves a well-defined positive growth
factor. The commit atomically selects the target prepared object, transferred
state, and an empty target-epoch cycle accumulator. Otherwise it returns all
source objects unchanged.

A successful epoch change explicitly requires all of the following:

- transfer the committed growth state;
- rebuild the mechanics reference operators;
- discard and rebuild cycle aggregation; and
- rebuild observation operators tied to support or reference geometry.

`GrowthEpochRebuildRequirements.differentiation` is
`"discrete-stop-gradient"`, and `ordinary_gradient_supported` is false.
`discrete_growth_log_transfer` applies `stop_gradient` to the complete mapped
tensor. Sensitivities may be taken inside a fixed prepared epoch, but an ordinary
AD gradient must never cross an anatomy/reference replacement. An outer discrete
optimization or explicitly derived transfer sensitivity is required for that
problem.

## Mean-field sarcomere cycle

`SarcomereState.crossbridge_fractions` ends in four populations:

1. detached, primed myosin with bound ADP and Pi;
2. weak-bound myosin with ADP and Pi;
3. strong-bound myosin with ADP; and
4. rigor-bound myosin.

The closed reaction cycle is rigor plus ATP to detached-primed, attachment to
weak-bound, power stroke and Pi release to strong-bound, and ADP release to
rigor. Exponential source outflows keep each simultaneous reaction extent below
its source population. ATP binding is additionally limited by free ATP.

`SarcomereCouplingInputs` makes every external modulation visible:

- calcium concentration from electrophysiology, in `mM`;
- sarcomere length and shortening velocity from mechanics, in `mm` and `mm/ms`;
- oxygen tension from perfusion, in `kPa`; and
- local oxidative capacity, in `pmol/mm3/ms`.

A Hill response supplies calcium activation. A Gaussian overlap law and an
explicit force-velocity law supply mechanical modulation. Oxygen limitation is
`pO2 / (pO2 + K_O2)`. It modulates oxidative ATP regeneration and interpolates
cross-bridge kinetics between the declared oxygen floor and fully oxygenated
rates. Oxidative regeneration consumes one free ADP and Pi for each ATP produced,
with `oxygen_per_atp` reported as oxygen demand.

```python
from phydrax.applications.cardiovascular.mechanics import (
    MeanFieldSarcomerePlan,
    SarcomereCouplingInputs,
    initialize_sarcomere_state,
    step_mean_field_sarcomere,
)

plan = MeanFieldSarcomerePlan(
    attachment_rate_per_ms=0.08,
    powerstroke_rate_per_ms=0.05,
    adp_release_rate_per_ms=0.03,
    atp_binding_rate_per_ms=0.04,
    calcium_half_saturation_mM=5.0e-4,
    calcium_cooperativity=2.0,
    atp_half_saturation=50.0,
    oxidative_adp_half_saturation=10.0,
    oxidative_pi_half_saturation=10.0,
    oxygen_half_saturation_kpa=2.0,
    oxygen_kinetic_floor=0.15,
    oxygen_per_atp=0.2,
    myosin_site_density=20.0,
    atp_free_energy=0.02,
    resting_length_mm=0.002,
    overlap_width_mm=0.0004,
    shortening_velocity_scale_mm_per_ms=0.0001,
    maximum_active_stress_kpa=50.0,
)
state = initialize_sarcomere_state(
    plan,
    (quadrature_point_count,),
    atp_pmol_per_mm3=100.0,
    adp_pmol_per_mm3=20.0,
    phosphate_pmol_per_mm3=20.0,
)
inputs = SarcomereCouplingInputs(
    calcium_mM,
    sarcomere_length_mm,
    shortening_velocity_mm_per_ms,
    oxygen_tension_kpa,
    oxidative_capacity_pmol_per_mm3_ms,
)
step = step_mean_field_sarcomere(plan, state, inputs, 0.02)
state = step.state
```

The returned state is the candidate only when all evidence passes; otherwise it
is the unchanged source.

## Species and power certificates

Free-pool balance includes nucleotide bound to cross-bridges. With site density
rho, the two conserved amount densities are

```text
adenylate = ATP + ADP + rho * (detached + weak + strong)
phosphoryl = ATP + Pi + rho * (detached + weak).
```

They remain invariant under cross-bridge reactions and oxidative regeneration.
`SarcomereStepEvidence` reports before/after values and both residuals, along with
population normalization and the minimum free-species concentration.

The power ledger tracks high-energy chemical storage in free ATP and the primed
and weak-bound states. It reports metabolic input from ATP regeneration,
chemical storage rate, power-stroke release, signed external mechanical power,
and heat, all in `kPa/ms`. Two independently visible closures are checked:

```text
metabolic input - chemical storage rate - power-stroke release = 0
metabolic input - chemical storage rate - mechanical output - heat = 0.
```

Negative heat beyond tolerance is thermodynamically inadmissible and rejects the
step rather than being clipped or hidden. This makes an inconsistent choice of
rates, force scale, state, velocity, or step size observable to the coupling
controller.

## Fidelity boundary

`MeanFieldSarcomereFidelity` and `StochasticMolecularSarcomereFidelity` are
distinct types, not string modes. A `MeanFieldSarcomerePlan` rejects the
stochastic type. Molecular-count realizations require their own random state,
ensemble evidence, and solver route; mean fractions and deterministic chemical
pools are never silently reinterpreted as individual molecules.

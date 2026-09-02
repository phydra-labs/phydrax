# Systems biology

`phydrax.applications.systems_biology` provides a fixed-shape intermediate representation for compartmental biological process networks, exact stochastic simulation through the PhydraX jump-process solvers, declared deterministic and chemical-Langevin approximations, a prepared telegraph gene-expression model, host evidence bindings, and atomic multiprocess whole-cell assembly.

## Compartmental process plans

A plan separates host-side biological meaning from the arrays used by compiled execution:

```python
import jax.numpy as jnp
from phydrax.applications import systems_biology as sb

cell = sb.CompartmentSpec("cell", 1.0, unit="cell")
plan = sb.StoichiometricNetworkPlan(
    "conversion",
    (cell,),
    (sb.SpeciesSpec("a", "cell"), sb.SpeciesSpec("b", "cell")),
    (
        sb.StoichiometricProcessSpec(
            "forward",
            {"a": -1, "b": 1},
            sb.MassActionPropensity(0.1, {"a": 1}),
        ),
    ),
    stoichiometry_capacity=2,
)
network = plan.prepare()
state = network.initial_state(jnp.asarray([100.0, 0.0]))
evaluation = network.evaluate(state)
```

Stoichiometry is supplied sparsely with nonzero integer coefficients. Preparation resolves names once and produces dense stoichiometry, reservoir-masked dynamic stoichiometry, and fixed-capacity sparse index/value/mask arrays. Unknown species, duplicate identities, nonintegral coefficients, cross-compartment higher-order mass action, and insufficient sparse capacity are rejected on the host.

Four propensity laws are available:

- `MassActionPropensity` uses falling-factorial combinatorics for exact counts and the corresponding power law for deterministic/CLE execution. Compartment measure scaling and factorial normalization are identical in both paths.
- `HillPropensity` supports activating and repressing responses, a positive Hill coefficient and half-saturation concentration, and a nonnegative basal rate.
- `MichaelisMentenPropensity` is an explicit saturating substrate law.
- `PromoterTransitionPropensity` is a first-order discrete transition gated by the source promoter count.

Every evaluation reports state validity, parameter validity, finite/nonnegative rates, a status code, source and sink rates caused by fixed reservoirs, and residuals in the prepared conservation basis. Invalid rates are returned as `NaN` rather than silently clipped. A reaction that lacks the molecules it consumes has zero exact propensity, so an SSA jump cannot make a valid count state negative.

## Exact SSA and approximations

`network.exact_jump_process()` implements the native `AbstractJumpProcess` contract. Compose it with `PoissonClockRealization` and `solve_direct_ssa` or `solve_next_reaction`; the realization's `process_id` must equal the adapter's identity. Reusing a realization therefore reproduces the same path, and capacity extension preserves existing random prefixes.

```python
import jax
from phydrax.solver import solve_direct_ssa
from phydrax.stochastic import PoissonClockRealization

process = network.exact_jump_process()
noise = PoissonClockRealization(
    jax.random.key(0),
    process.num_channels,
    support=(0.0, 10.0),
    max_events_per_channel=128,
    process_id=process.process_id,
)
solution = solve_direct_ssa(
    process,
    noise,
    state,
    t0=0.0,
    t1=10.0,
    save_times=jnp.linspace(0.0, 10.0, 101),
    args=network.default_runtime(),
)
```

`deterministic_step` and `cle_step` return candidates; call `result.commit(state)` to retain the input on numerical failure. Their `ApproximationEvidence` deliberately distinguishes numerical validity from scientific regime validity:

- deterministic validity declares a configurable minimum copy number across input and candidate endpoints for nonreservoir species whose quantity is `count`; non-count deterministic fields are not compared to copy thresholds;
- CLE is restricted to count-valued networks and additionally declares a configurable minimum expected event count over the step;
- neither approximation hides a negative or nonfinite candidate;
- deterministic differentiation is declared only away from Hill/Michaelis--Menten driver boundaries and reservoir-ledger kinks;
- CLE differentiation additionally requires positive duration and channel intensities for its fixed-noise reparameterized candidate;
- exact SSA sample paths are not declared differentiable because discrete event choices change with parameters.

A false `regime_valid` value does not rewrite the mathematical candidate. It tells the caller that exact SSA, a smaller step, or a better approximation is required.

## Thermochemical specialization

This package does not redefine chemical species thermodynamics, Arrhenius rates, reversibility, charge balance, or element balance. Those semantics remain in `phydrax.equations.ChemicalMechanismIR` and `PreparedChemicalMechanism`, with exact-count chemical jumps in `phydrax.stochastic.ChemicalJumpProcess`.

For a biological process that is also a thermochemical reaction:

1. give each `SpeciesSpec` a `thermochemical_name`;
2. give each mass-action `StoichiometricProcessSpec` its `thermochemical_reaction` name;
3. call `network.bind_thermochemical(prepared_mechanism)`.

The binding fails unless every species resolves and every declared reaction has exactly matching reactant coefficients, product coefficients, and forward mass-action orders. The biological reactant side is reconstructed from its explicit kinetic orders, including catalysts restored on the product side. `ThermochemicalInteropEvidence` retains the exact resolved process, reaction, and species index arrays plus a `mechanism_content_id` that covers every static and dynamic field of the prepared mechanism and rate plans. Thermochemical kinetics must still be evaluated by the chemical mechanism; the systems-biology propensity is not a substitute for those semantics.

## Telegraph transcription and count measurement

`TelegraphGeneExpressionPlan` prepares four species (`promoter_off`, `promoter_on`, `nascent`, and `mature`) and five reactions: activation, deactivation, transcription, splicing, and mature-RNA degradation.

```python
model = sb.TelegraphGeneExpressionPlan(
    1.5,  # promoter activation
    2.5,  # promoter deactivation
    8.0,  # transcription
    3.0,  # splicing
    1.0,  # mature degradation
).prepare()
initial = model.initial_state(promoter_on=False, nascent=0, mature=0)
moments = model.stationary_moments()
```

`stationary_moments` gives exact means, variances, and promoter/nascent/mature covariances for this linear Markov network. `TelegraphFitTarget` stores the five fitting observables and their uncertainty scales. `model.fitting_objective(log_rates, target)` is differentiable and works directly with `jax.grad`. Stationary moments determine rate ratios but not an overall time scale, so `identifiability_evidence` normally reports rank four for the five rate parameters. Time-resolved evidence is needed to identify the missing scale.

`CountMeasurementPlan` describes binomial molecular capture followed by independent Poisson false-positive counts. Preparation fixes the maximum observable count, making the exact convolution JIT-compatible. `log_likelihood(observed, latent_count)` reports invalid integer counts and capacity overflow explicitly. `observed_moments` propagates latent mean and variance through the same measurement model.

## Biological evidence bindings

Biological values are not silently copied into plans. A `BiologicalFact` has a namespace, unit, and exact `BiologicalReference`; a `BiologicalCondition` names the context; and `PlanFieldAssertion` names one field from a target's closed evidence-field set. `bind_biological_evidence` requires both value and declared target unit to match and ties the result to the exact prepared identity. Network time units are explicit; general propensity units are derived from changed-species, driver-species, compartment-measure, order, and time units, while telegraph rates declare `s^-1`.

Conflicting facts with the same namespaced key, conflicting conditions, two different facts targeting one field under one condition, unknown fields, and conflicting references are errors. Reserved path/key delimiters are rejected at construction so field paths and namespaced keys are injective. A value, boolean type, or unit mismatch produces `valid=False`, preserving auditable negative evidence. Repeating an identical binding produces the same `binding_id`.

## Atomic multiprocess whole-cell assembly

Whole-cell assembly shares typed scalar fields without merging the internal identities of its component networks:

1. declare `ExchangeFieldSpec` values with quantity, unit, and global reservoir semantics;
2. map every species of each prepared network with `WholeCellProcessBinding`;
3. assign each process a static substep count, minimum copy number, and regime-enforcement policy using `MultirateScheduleEntry`;
4. reserve explicit `field_capacity` and `process_capacity` in `WholeCellAssemblyPlan`;
5. prepare once and use `step` followed by `commit`.

Every process advances from the same macro-step snapshot. Its internal substeps may differ, but all process deltas are combined into one candidate. Regime evidence checks both each isolated substep path and the final coupled shared-field candidate. `require_regime_valid=True` rejects a low-copy candidate; setting it to `False` is an explicit policy waiver that retains `regime_valid=False` in the result. A commit succeeds only if the base lineage, epoch, amounts, and both ledgers still match and every required process, finite-value check, nonnegativity check, and conservation check succeeds. Otherwise amounts, source ledger, sink ledger, and epoch all remain unchanged.

Reservoir values stay fixed. Every process mapping a field must agree with its global reservoir declaration, preventing one process from treating a shared dynamic field as a chemostat. Consumption from a reservoir increments the source ledger; production into a reservoir increments the sink ledger. Boundary contributions are split per reaction before summation, so simultaneous opposing flows remain visible rather than cancelling. Conservation is checked on `dynamic_delta + sink_delta - source_delta`, which reconstructs the full closed-system change. `checkpoint(state)` computes a host identity over the assembly, lineage, epoch, values, and both cumulative ledgers, so even ledger-only differences cannot collide.

Run `python benchmarks/systems_biology.py` to measure lowering, compilation, steady execution, logical memory, compiler cost/memory evidence, and conservation for a compiled gene-expression/translation whole-cell step.

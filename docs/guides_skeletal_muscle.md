# Skeletal-muscle modeling

`phydrax.applications.skeletal_muscle` begins with one deliberately bounded model:
the deterministic mean-rate sustained-isometric motor-unit population of Potvin and
Fuglevand (2017). It is not a generic muscle model, a musculotendon actuator, a cellular
excitation-contraction model, or a source of physical force in newtons.

## Scientific identity and scope

The implementation is independently derived from:

- Potvin and Fuglevand, *A Motor Unit-Based Model of Muscle Fatigue*, PLOS
  Computational Biology 13(6), 2017, DOI
  [`10.1371/journal.pcbi.1005581`](https://doi.org/10.1371/journal.pcbi.1005581);
- the paper-linked author MATLAB reference at commit
  `15462f85106ed9ebde3d78ab6fe665c88bf8b32e`.

The model represents 120 heterogeneous motor units by default. All units receive one
common excitatory drive. Ordered recruitment, rate coding, firing-rate adaptation,
force-frequency fusion, peripheral force-capacity loss, and fatigue-induced twitch
slowing are explicit. The shipped parameterization is generally representative rather
than calibrated to a particular human muscle.

Supported claim: deterministic mean-rate force and capacity during sustained isometric
contractions. The model does not include recovery, individual spikes, force noise,
length or velocity effects, tendon, pennation, calcium, crossbridges, geometry, sensory
feedback, or physical force calibration.

## State and outputs

The only independent state per motor unit is:

- time since first recruitment, in seconds;
- current twitch-force capacity, relative to the rested twitch force of motor unit one.

Twitch contraction time is derived from current capacity and cannot drift as an
independent state. The interval output includes unadapted and adapted firing rates,
contraction time, normalized firing rate and force, per-unit relative force, total
relative force, capacity fractions, recruitment/saturation masks, and distances to the
nearest hard branch boundaries.
The aggregate capacity fraction uses the paper's rested maximum-excitation fusion
weighting; it is not the unweighted mean of per-unit capacity fractions.

Force is evaluated from the interval source state. Fatigue then constructs the target
candidate state. A failed candidate retains its diagnostic output and evidence but
commits the complete source state.

## Units

The runtime uses seconds and hertz. Relative twitch and population forces are
nondimensional model units. The original 30--90 ms contraction-time parameters are
converted to 0.03--0.09 s during parameter construction. No result is labeled as
newtons or percent MVC without an explicit external normalization.

Canonical quantity definitions are available through
`skeletal_muscle_quantity(...)`. These are application quantities, not a general unit
registry.

## Basic execution

```python
from phydrax.applications import skeletal_muscle

plan = skeletal_muscle.motor_units.PotvinFuglevand2017Plan()
runtime = plan.prepare()
state = runtime.initialize()
candidate = runtime.candidate(state, 40.0, 0.1)
assert bool(candidate.evidence.successful)
state = candidate.commit()
assert candidate.output.total_force > 0.0
assert state.current_twitch_force.shape == (120,)
```

`candidate.output` describes the source-time force for this interval; `state` is the
accepted target state. `runtime.advance(...)` combines candidate construction and
commit while retaining the same output and evidence.

## Dynamics and batching

`plan.as_discrete_system()` exposes a one-population array-state `DiscreteSystem`.
Pass a `PotvinFuglevand2017Parameters` instance as the dynamics `args`; parameters are
numeric JAX leaves rather than static closure values. Homogeneous independent
populations use `jax.vmap`. Populations with different motor-unit counts require
separate prepared runtimes.

## Differentiation

The implementation is JIT-, scan-, vmap-, JVP-, and VJP-compatible within a fixed hard
branch regime. Recruitment thresholds, firing-rate saturation, and capacity clipping
are nonsmooth. Derivatives at or across those boundaries are not supported.

`minimum_recruitment_margin` and `minimum_saturation_margin` expose distance to the
nearest branch. A future smooth relaxation, if scientifically justified, must use a
different fidelity identity and be replayed against this hard model; it will not
silently replace the published transition.

## Parameters and fitting

Topology, model identity, mechanism selections, and the maximum qualified step are
static plan fields. Physiological coefficients and per-unit arrays remain trainable JAX
leaves in `PotvinFuglevand2017Parameters`. Runtime admissibility checks remain active
after parameter transformation and cause whole-state rollback when trained values leave
the model domain.

The default plan enables both intrinsic firing-rate adaptation and peripheral fatigue.
Disabling either mechanism is a static, identity-bearing ablation. It does not add
recovery.

## Failure semantics

Candidates are unsuccessful for:

- nonfinite state, input, parameters, or output;
- negative or over-maximum excitation;
- nonpositive or over-qualified timestep;
- negative duration or capacity outside the rested range;
- unordered or physically inadmissible parameter arrays.

Shape/topology mismatches are host errors. Numerical failure never commits a partially
advanced population.

## Qualification and benchmark

Run the independent scalar and published-protocol qualification:

```text
python tools/skeletal_muscle_motor_unit_qualification.py
```

Run the performance and storage benchmark:

```text
python benchmarks/skeletal_muscle_motor_units.py
```

The qualification checks exact default population endpoints, independently evaluated
single-step equations, rested maximum-excitation force, the reported 50% and 80% MVC
endurance protocols to within one 0.1 s source sample, and timestep refinement. The
checked-in benchmark is environment-bound evidence, not a portable performance
guarantee.

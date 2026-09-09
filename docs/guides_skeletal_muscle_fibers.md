# Skeletal-muscle fiber bundles

`SkeletalFiberBundlePlan` advances a fixed homogeneous bundle of one-dimensional
fibers using the complete Shorten 2007 fast-twitch reaction model and a no-flux
finite-difference monodomain diffusion term. Each fiber has a fixed node count;
fibers with different node counts belong to separate execution worksets rather than
one global padded state.

The state shape is `(fiber, node, 56)`. Only the surface membrane potential `vS`
diffuses. The remaining cellular states evolve locally through the pinned Shorten
reaction equations. Fiber lengths use mm, time uses ms, and diffusivity uses
`mm2/ms`.

`PrescribedFiberStimulusSchedule` owns a fixed pulse capacity and an explicit
`(pulse, fiber, node)` support mask. Pulses are left-closed and right-open. Candidate
windows may end on a pulse boundary but may not cross an unrepresented interior
boundary; such a candidate retains its evidence and rolls the whole bundle back.
Motor-unit firing rates are not converted to current.

The original prepared runtime is the retained **dense small-case oracle**. It
solves the coupled reaction-diffusion system with Phydrax's Diffrax backend and
Kvaerno5, including a full-bundle Jacobian. It exposes membrane/t-tubule potential, cytosolic
calcium, the Shorten force-bearing `A_2` crossbridge concentration, and the applied
stimulus field. `A_2` is a biochemical concentration, not physical force or stress.

The spatial discretization uses mirrored no-flux endpoint values. Constant membrane
potential therefore has exactly zero diffusion contribution. Qualification checks
support selectivity, neighbor propagation, event alignment, finiteness, and complete
rollback. It does not claim endplate physiology, EMG, three-dimensional mechanics,
or MPI-scale performance.

## Sparse motor-unit territories

`MotorUnitTerritoryPlan` stores one motor-unit index and one endplate node for each
fiber. The representation scales as motor units plus fibers plus event slots; it does
not allocate a dense motor-unit-by-fiber-by-node tensor. Preparation requires every
fiber to be assigned and every declared motor unit to own at least one fiber.

`bind_events(...)` routes a fixed `(motor_unit, event_slot)` event block to a
`MotorUnitEndplateStimulus`. Event times may come from a qualified stochastic motor
unit model, but pulse amplitude, duration, and `stimulus_source_id` are explicit
inputs. This adapter performs no firing-rate-to-current conversion and makes no
universal neuromuscular-junction claim. Inactive event slots remain masked and do not
allocate or contribute current.

Run `examples/skeletal_muscle_motor_territories.py` to inspect the sparse routing
contract.

Run:

```text
python examples/skeletal_muscle_fibers.py
python tools/skeletal_muscle_fiber_qualification.py
python benchmarks/skeletal_muscle_fibers.py
```

## Structured source-bound response

`StructuredFiberResponsePlan` is a separate numerical identity, not a replacement
for the original source or an automatic runtime fallback. It applies a Strang
reaction–diffusion–reaction split with independently vmapped local Kvaerno5 solves
and the native `solve_tridiagonal_lines` substrate for no-flux Crank–Nicolson
diffusion. The local implicit Jacobian has only the cellular state count; no
full-bundle reaction Jacobian is formed. Diffusion storage is linear in fiber nodes.
The split requires its own time-refinement evidence against the unchanged dense
small-case oracle.

The admitted binding is `Shorten2007FiberReaction(ShortenFastTwitchModel())`.
`AbstractFiberReaction` defines the reusable local-cell contract:
`initialize(batch_shape)`, and local-vector
`rhs(time_ms, values, current_uA_per_cm2, stretch, stretch_rate_per_ms)` and
`admissible(...)`. Source identity and state/voltage layout are explicit; parameters
remain dynamic. The existing Shorten binding deliberately ignores kinematics:
**moving cable geometry does not add non-isometric crossbridge biology**.
No Heidlauf–Röhrle 2014 binding or cell–continuum reproduction is admitted here.

### Moving one-dimensional metric

Reference and current coordinates have shape `(fiber, node, 3)` in mm in the
caller-declared world frame. `geometry_source_id` identifies the reference geometry
and frame asset. Each candidate supplies every endpoint of a fixed substep geometry
path: `(substeps + 1, fiber, node, 3)`. The first endpoint must equal accepted
geometry exactly; no broadcasting or implicit rebinding is permitted.
`linear_geometry_path(state, end_positions_mm)` constructs the declared affine
path; supplying the accepted endpoint explicitly constructs a stationary path.

Before each diffusion solve, midpoint current positions determine segment
arclengths, lumped linear-element dual lengths, current/reference segment and node
Jacobians, and segment conductance. Spatial diffusivity has explicit shape
`(fiber, segment)` and units `mm2/ms`; conductance is diffusivity divided by current
segment length. On a uniform stationary line, this recovers the original mirrored
no-flux endpoint stencil. The voltage equation uses a material derivative with no
invented dilution or cross-sectional-area law. Rotations/translations preserve the
metric. A path that collapses a segment at an interior time is rejected even when
both endpoint geometries are nondegenerate.

Local reaction stretch is the ratio of current/reference dual lengths. Within
each fixed substep it follows affine endpoint interpolation, and its signed
velocity is the endpoint stretch difference divided by that substep's duration in
ms. Any future non-isometric binding must explicitly supply source-compatible
reference sarcomere length and convert these inputs to source length/velocity.
A qualified continuum-to-fiber transfer and three-dimensional deformation
admissibility remain separate prerequisites; these cable metrics do not establish
virtual-power consistency or a mechanical force owner.

```python
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.cellular import ShortenFastTwitchModel
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    Shorten2007FiberReaction,
    StructuredFiberResponsePlan,
)

positions = jnp.zeros((1, 5, 3)).at[0, :, 0].set(jnp.linspace(0.0, 10.0, 5))
stimulus = PrescribedFiberStimulusSchedule(
    jnp.asarray([0.0]), jnp.asarray([0.05]), jnp.asarray([150.0]),
    jnp.zeros((1, 1, 5), dtype=bool).at[0, 0, 0].set(True),
)
runtime = StructuredFiberResponsePlan(
    ("fiber-a",), positions, stimulus, jnp.asarray([0.0, 0.5, 1.0]),
    geometry_source_id="caller-manufactured-10mm-world-x",
).prepare(
    Shorten2007FiberReaction(ShortenFastTwitchModel()),
    jnp.full((1, 4), 0.1),
)
state = runtime.initialize()
path = runtime.linear_geometry_path(state, state.node_positions_mm)
candidate = runtime.candidate(state, 0.05, path)
state = runtime.commit(state, candidate)
```

### Events, transactions and numerical limits

Every pulse edge inside a candidate window must coincide exactly with one of its
fixed substep times. Interior unaligned events are rejected, not crossed using an
averaged stimulus. Current is held at the interval's interior value during each
reaction solve so a right-endpoint pulse transition cannot contaminate an implicit
stage. Split the requested window or prepare an explicitly aligned grid when an
event does not fit; no adaptive numerical-fidelity fallback occurs.

State includes time, cell values, node positions, accepted-step counter and
prepared identity. Candidate evidence includes local reaction status/step counts,
diffusion residuals, no-flux weighted-voltage balance, minimum pivots, geometry
support and event alignment. Any failure rolls back all state fields. Commit
compares the entire source snapshot against the supplied current state: a stale
candidate preserves that current state rather than restoring its own old source.
Foreign prepared candidates are rejected.

Plans bind geometry, source/frame identity, schedules, substep grid and all
numerical policies. Prepared identities use the shared semantic/numeric identity
substrate and include the complete reaction parameter content and segment
diffusivity. Reprepare after changing physical parameters between accepted runs;
parameters remain differentiable array leaves, not static trainable exclusions.
Direct-adjoint local solves support branch-local JVP/VJP; pulse edges, failed
solves, stale commits and degenerate geometry are not smooth physical branches.

Crank–Nicolson is A-stable, **not unconditionally monotone or positivity preserving**.
For each node, a sufficient monotonicity bound is
`dt <= 2 * dual_length / (left_conductance + right_conductance)`, evaluated in the
current midpoint geometry; zero conductance imposes no bound. Large steps may
produce alternating diffusion modes without violating the linear residual.
Action-potential accuracy and splitting error impose additional refinement
requirements; successful solver status is not physiological qualification.

Run the independent numerical and measured-workload drivers:

```text
python tools/skeletal_muscle_structured_fiber_qualification.py --smoke
python tools/skeletal_muscle_structured_fiber_qualification.py
python benchmarks/skeletal_muscle_structured_fibers.py --smoke
python benchmarks/skeletal_muscle_structured_fibers.py --fibers 8 --nodes 65
```

The qualification driver records dense-oracle split refinement, not 2014 source
reproduction. The benchmark separates lowering, compilation, steady execution and
individual reaction/diffusion block timings, with backend compiler memory
estimates. Compiler estimates are not measured peak device memory, separate block
launches are not an additive profiler trace, and a local-device run establishes
neither distributed execution nor an unmeasured production capacity.

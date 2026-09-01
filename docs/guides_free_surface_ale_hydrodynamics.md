# One-phase free-surface ALE hydrodynamics

Phydrax provides a fixed-topology, single-phase, fully nonhydrostatic graph-surface
hydrodynamics product. It is separate from rigid-lid Boussinesq flow, hydrostatic
primitive equations, two-phase VOF, potential flow, and dispersive shallow water.

## Authoritative state

`FreeSurfaceALEState` stores:

- horizontal cell free-surface elevation;
- extensive face momentum work-conjugate to the mapped velocity Hodge;
- extensive cell scalar content.

Velocity and scalar concentration are geometry-dependent views. Pressure head is
method history rather than physical prognostic state.

## Graph ALE map

`GraphSurfaceALEPlan` maps a fixed reference column by

`z = b + sigma (eta - b)`.

The map is single-valued and cannot represent overturning. A stage passes eta and
eta-rate through an explicit local-time payload so the native ALE JVP obtains the
correct mesh velocity, cell-volume rate, mesh flux, and GCL residual.

The initial product requires:

- positive liquid height;
- fixed logical topology;
- bounded vertical reference axis;
- graph slope below the configured limit;
- positive mapped cell/face/dual measures;
- passing ALE geometry evidence.

## Surface kinematics

Surface eta is coupled to the physical top-face volume flux. The kinematic solve uses
the Jacobian of mapped column volume with respect to eta:

`J_eta eta_rate = top physical volume flux`.

This avoids assuming that a cell eta maps trivially to vertex displacement. The result
reports target/reproduced volume rate, residual, and solver evidence.

## Nonorthogonal velocity Hodge

The mapped Hodge combines positive diagonal face-dual kinetic energy with reconstructed
cell-vector kinetic energy. Momentum is the gradient of this quadratic form with
respect to face-normal velocity. Its inverse is a checked matrix-free CG solve.

This supplies positive extensive momentum while retaining cross-metric coupling on a
sloping graph mesh.

## Pressure projection

`MappedFreeSurfaceProjectionPlan` performs a mixed-boundary projection:

- fitted top pressure head is Dirichlet;
- closed bottom/lateral normal velocity is fixed;
- periodic sides remain periodic;
- no mean pressure gauge is applied when the top Dirichlet boundary is present.

The pressure gradient covector is the exact transpose of mapped divergence under the
cell-volume pairing. The same mapped Hodge inverse appears in tentative momentum and
the pressure action.

The product uses modified pressure head

`Pi = p / rho + g z - p_atm / rho`.

Without capillarity, top pressure head is `g eta` plus the declared atmospheric
anomaly convention.

## Coupled stage and time stepping

`OnePhaseFreeSurfaceALEMethod` uses explicit midpoint stepping. Every half/full stage
iterates:

1. eta and eta-rate;
2. mapped geometry and grid velocity;
3. mapped inviscid momentum tendency;
4. end-stage extensive momentum;
5. free-surface pressure loading;
6. mixed pressure projection;
7. corrected top physical volume flux;
8. kinematic eta-rate solve.

A stage is accepted only when geometry, pressure projection, kinematic solve, and
nonlinear coupling all pass.

## ALE scalar transport

Scalar content is transported with the exact relative mesh flux used by ALE momentum:

`q_rel = A dot (u - w_grid)`.

Upwind concentration is multiplied by this integrated face flux. Uniform concentration
is preserved under compatible mesh motion and GCL closure.

## Ledger

`FreeSurfaceALELedger` records accepted:

- liquid volume change;
- scalar content changes;
- kinetic-energy change;
- gravitational free-surface energy change;
- pressure work;
- GCL residual;
- divergence residual;
- kinematic residual;
- dynamic boundary residual;
- nonlinear-stage residual;
- total energy residual.

Rejected transitions preserve state, pressure history, eta-rate history, and ledger.

## Restart and output

`write_free_surface_checkpoint` stores model/method IDs, eta, momentum, scalar content,
eta-rate, pressure warm start, ledger, time, and accepted step. Rigid-lid, hydrostatic,
VOF, and potential-flow checkpoints are rejected.

`free_surface_diagnostic_view` and `write_free_surface_output` expose mapped vertices,
cell volumes, eta, velocity, scalars, pressure head, energies, volume, and ledger
without making derived fields authoritative.

## Minimal construction

```python
import jax.numpy as jnp
import phydrax as phx

reference_grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(3, periodic=True),
        phx.discretization.UniformCellAxisSpec(3, periodic=True),
        phx.discretization.UniformCellAxisSpec(3, periodic=False),
    ),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray(((0.0, 0.0, -1.0), (3.0, 3.0, 0.0))))
reference = phx.discretization.FiniteVolumePlan(
    reference_grid, component_names=("hydrodynamics",)
).prepare()
surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
    reference, jnp.full((3, 3), -1.0)
)
hydrodynamics = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
    surface
).prepare()
state = hydrodynamics.initial_state(jnp.zeros((3, 3)))
continuation = (
    phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(
        state
    )
)
```

## Deliberate limits

- inviscid only;
- fixed graph topology;
- no wave generation/absorption;
- no surface tension;
- no rigid or hydroelastic bodies;
- no remeshing;
- no wetting/drying;
- no two-phase flow;
- no breaking or contact;
- no distributed execution.

Those are separate evidence-gated extensions rather than flags on the initial product.

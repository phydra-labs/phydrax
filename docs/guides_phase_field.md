# Binary phase-field production

`phydrax.applications.phase_field` provides prepared, single-device finite-element
routes for closed binary Allen–Cahn and Cahn–Hilliard evolution. Both routes use one
canonical thermodynamic model, compile their finite-element stationarity problem
once, and promote a candidate only after nonlinear, energy, and conservation gates
pass.

## Qualified envelope

The production profile is deliberately narrow:

- symmetric quartic `DoubleWellFreeEnergy`;
- positive constant bulk scale, isotropic gradient coefficient, and scalar mobility;
- first-order convex splitting;
- scalar conforming P1 fields on one homogeneous triangle block;
- fixed topology and geometry;
- natural closed boundaries;
- deterministic float64 single-device execution.

Preparation rejects nonzero wetting strength, an unsupported free energy, an
underresolved interface, multiple cell blocks, non-P1 fields, non-triangular cells,
or non-float64 finite-element precision. Periodic constraints, imposed boundary
work, sources, noise, anisotropy, AMR, and distributed execution are not inferred.

## Thermodynamic model

For effective quartic coefficient $A$ and gradient coefficient $\kappa$, the
physical free energy is

$$
F[\phi] = \int_\Omega \left[
\frac{A}{4}(\phi^2-1)^2 + \frac{\kappa}{2}|\nabla\phi|^2
\right] \, d\Omega.
$$

`BinaryPhaseFieldModel` binds `BinaryThermodynamicParameters` to the existing
`BinaryPhaseThermodynamicClosure`. The same model evaluates physical energy,
provides the quartic convex split, and reports characteristic interface width and
planar surface tension. The LBM and kinetic thermodynamic closures therefore remain
separate numerical realizations of the same constitutive data.

## Convex-split time steps

Allen–Cahn uses the weak equation generated from

$$
J_{AC} = \int_\Omega \left[
\frac{(\phi-\phi_n)^2}{2\Delta t}
+ M A\left(\frac{\phi^4}{4}-\phi_n\phi\right)
+ \frac{M\kappa}{2}|\nabla\phi|^2
\right] \, d\Omega.
$$

Cahn–Hilliard uses the mixed saddle functional

$$
J_{CH} = \int_\Omega \left[
(c-c_n)\mu + \frac{\Delta t M}{2}|\nabla\mu|^2
- A\left(\frac{c^4}{4}-c_nc\right)
- \frac{\kappa}{2}|\nabla c|^2
\right] \, d\Omega.
$$

Both are declared as `variational.Functional` values and differentiated by the
finite-element executor. Previous-state quadrature values and the attempted step
size are dynamic arguments; changing either does not rebuild the form.

## Preparation and direct stepping

```python
import jax.numpy as jnp
import phydrax as phx

model = phx.applications.phase_field.BinaryPhaseFieldModel(
    phx.equations.BinaryThermodynamicParameters(1.0, 0.05)
)
plan = phx.applications.phase_field.CahnHilliardFEMPlan(
    model,
    mobility=1.0,
)
method = plan.prepare(discretization, "c", "mu")
state = method.initialize(c0, chemical_potential=mu0)
result = method.step_detailed(
    jnp.asarray(0),
    jnp.asarray(0.0),
    state,
    jnp.asarray(5.0e-4),
)
state = result.accepted_state
```

If `chemical_potential` is omitted, initialization uses a zero continuation seed.
It is only a nonlinear initial guess; the first mixed root establishes the chemical
potential satisfying the discrete equations.

`PreparedAllenCahnFEM` and `PreparedCahnHilliardFEM` expose `mass`, `energy`,
`initialize`, and `step_detailed`. They also implement `AbstractFixedStepMethod`, so
the generic retry and production runtimes consume them directly.

## Acceptance evidence

`PhaseFieldStepEvidence` records the candidate field range, nonlinear work and
residual, physical energy before and after, dissipation, energy-balance defect,
mass values, thresholds, and every acceptance decision.

For Allen–Cahn,

$$
D_{AC} = \int_\Omega
\frac{(\phi_{n+1}-\phi_n)^2}{M\Delta t} \, d\Omega.
$$

For Cahn–Hilliard,

$$
D_{CH} = \Delta t M \int_\Omega |\nabla\mu_{n+1}|^2 \, d\Omega.
$$

The energy gate checks

$$
F_{n+1} - F_n + D \leq \tau_E.
$$

Cahn–Hilliard additionally compares every candidate mass with the mass stored at
initialization, rather than only with the preceding step. A rejected nonlinear root,
nonfinite candidate, positive energy-ledger excess, or excessive mass defect leaves
the complete prior accepted state unchanged. No clipping or mass correction is
applied.

## Interface resolution

The model reports the characteristic width

$$
\lambda = \sqrt{2\kappa/A}.
$$

`PhaseFieldResolutionEvidence` defines the transition span as $4\lambda$ and
reports that span divided by the maximum cell diameter. The default production
profile requires at least four cells across the transition span. This is an explicit
qualification threshold, not a universal phase-field convergence criterion.

## Production execution and restart

```python
from pathlib import Path

case = method.production_case("spinodal-reference", state)
run_plan = method.production_run_plan(
    step_size=5.0e-4,
    end_time=0.1,
    maximum_steps=200,
    checkpoint_interval=10,
    segment_steps=10,
    retry_policy=phx.solver.RobustRetryPolicy(maximum_retries=1),
)
store = phx.solver.DurableCheckpointStore(
    Path("phase-field-checkpoints"),
    case.manifest,
    phx.solver.CheckpointGenerationPolicy(3),
)
runtime = phx.solver.PreparedProductionRun(case.manifest, run_plan, store)
run = runtime.run(runtime.initial_state(case.initial_state))
```

The case identity binds the initial accepted state, model and mobility, compiled
method, finite-element topology and geometry layout, precision policy, and dtype.
A checkpoint from a different physical or numerical case is rejected.

The nonlinear solve and fixed-route step are differentiable on the selected branch.
Acceptance, retry, checkpoint publication, and restart selection are discrete
operations and are not advertised as one smooth map.

## Qualification and performance evidence

Run the scientific qualification with float64 enabled:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 python \
  tools/phase_field_production_qualification.py \
  --output benchmarks/phase_field_production_qualification.json
```

It checks planar-interface surface tension under spatial refinement, first-order
Allen–Cahn and Cahn–Hilliard temporal refinement, energy/dissipation ledgers,
Cahn–Hilliard mass conservation, and sparse/matrix-free agreement.

Run the compiled performance benchmark with:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 python \
  benchmarks/phase_field_production.py \
  --output benchmarks/phase_field_production.json
```

The benchmark records preparation, lowering, compilation, synchronized steady-step
timing, compiler work and memory estimates, nonlinear work, and physical diagnostics.
It does not impose a cross-hardware wall-time threshold.

## Nonclaims

This profile does not claim periodic finite elements, multiple cell blocks, adaptive
meshes, anisotropic or tensor mobility, contact-angle wetting, stochastic forcing,
anti-trapping currents, grand-potential or CALPHAD closure, nucleation, thermal
solidification, active multiphase storage, MPI execution, or GPU-specific kernels.
Diffuse fracture remains under `phydrax.applications.fracture` because its history,
irreversibility, and degradation contracts differ from phase evolution.

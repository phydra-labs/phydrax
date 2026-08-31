# Time integrators

Phydrax separates the mathematical equation form, temporal method, controller,
nonlinear/linear stage solve, and output schedule. `TimeGrid` contains requested output
nodes. Native adaptive solvers separately report a `RealizedTemporalMesh` of accepted
internal steps.

## Equation forms

| Form | Public problem | Backend |
| --- | --- | --- |
| Explicit ODE | `DifferentialProblem` | Diffrax, Rosenbrock-W, or Gauss IRK |
| Additive ODE | `SplitDifferentialProblem` | Diffrax ARK/IMEX |
| Implicit residual | `DifferentialAlgebraicProblem` | Native BDF or endpoint theta |
| Second order | `SecondOrderDifferentialProblem` | Native generalized-alpha |
| Slow/fast partition | `PartitionedDifferentialProblem` | Native fixed-ratio partitioned RK |
| SDE | `DifferentialProblem` plus `WienerTerm` | Diffrax |
| Manifold ODE/SDE | `DifferentialProblem` plus state geometry | Phydrax Diffrax solvers |
| Neural field manifold | `NeuralGalerkinProblem` lowered to a parameter `DifferentialProblem` | Diffrax with fixed field metric |

Phydrax never inverts a mass matrix or discards an additive split implicitly.

Neural Galerkin accepts the ordinary deterministic Diffrax catalog after lowering
selected model leaves to one parameter vector. Its integration realization and
evaluation key remain fixed for the complete solve so embedded error estimates see
one deterministic vector field. Saved-node audits, not hidden RK stages, retain the
tangent linear status and projection defect.
Unsupported problem/method/controller combinations fail before numerical execution.

## Method catalog

| Method | Order | Adaptive | Principal properties |
| --- | ---: | --- | --- |
| Diffrax ERK | 1--8 | Method-dependent | Nonstiff explicit ODE |
| Diffrax Kvaerno | 3--5 | Yes | A/L-stable ESDIRK |
| KenCarp | 3--5 | Yes | Additive ERK/ESDIRK IMEX |
| `SSPRK33` | 3 | No | SSP coefficient 1 |
| `SSPRK54` | 4 | No | Five-stage low-storage SSP |
| `BDFMethod(1..5)` | 1--5 | Yes | Residual ODE/regular index-one DAE |
| `ThetaMethod(theta, endpoint=True)` | 1 or 2 | No | Backward Euler and Crank--Nicolson |
| `RosenbrockWMethod` | 3 | Yes | Matrix-free RA34PW2, embedded order 2 |
| `GeneralizedAlphaMethod` | 2 | No | Controlled high-frequency damping |
| `MultiratePartitionedRK` | 2 or 3 | No | Fixed-ratio synchronized subcycling |

| `GaussLegendreIRK(1..3)` | 2, 4, or 6 | No | A-stable, symplectic collocation |
| Geometric Euler/RKMK/CF | 1--4 | No | Retraction-based manifold integration |
| Störmer--Verlet | 2 | No | Separable canonical Hamiltonians |
| Exponential Euler/Milstein | 1 | No | Matrix-free semilinear PDE/SPDE |

`multirate_amr_subcycling_plan(method)` binds the same refinement ratio and temporal
method identity into conservative `FDAMRSubcyclingPlan` reflux execution.

`TemporalMethodCapabilities` reports equation forms, endpoint and dense-output order,
stability properties, stochastic requirements, stage abscissae, history depth, and
whether each claim is verified.

## Additive IMEX

```text
problem = phx.solver.SplitDifferentialProblem(
    explicit_drift,
    implicit_drift,
    initial_state,
    t0=0.0,
    t1=1.0,
    problem_id="reaction-diffusion-split",
)
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.linspace(0.0, 1.0, 101),
    solver=dfx.KenCarp4(),
)
```

A certified `CompiledDiscreteDynamics.semilinear_drift` can be lowered with
`split_differential_problem`. Its spatial `DiscretizationBundle` is preserved.

## Residual BDF and theta

```text
policy = phx.solver.DAESolvePolicy(
    method=phx.solver.BDFMethod(5),
    adaptive=phx.solver.DAEAdaptivePolicy(
        relative_tolerance=1e-5,
        absolute_tolerance=1e-8,
        maximum_accepted_steps=2048,
        maximum_attempts=4096,
    ),
)
solution = phx.solver.solve_dae(problem, output_grid, policy=policy)
```

`BDFMethod(k)` selects the maximum permitted order. Adaptive startup and rejected
steps lower the realized order when history or adjacent-step ratios are insufficient.
Continuation retains six state/rate slots and five accepted step sizes.

The stiffly accurate endpoint theta form supports residual DAEs:

```python
import phydrax as phx
crank_nicolson = phx.solver.ThetaMethod(0.5, endpoint=True)
```

The non-endpoint midpoint form is not used for residual DAEs. The equivalent
one-stage implicit midpoint method is `GaussLegendreIRK(1)` for explicit-form ODEs.

## Matrix-free stiff methods

`solve_rosenbrock` applies fixed-grid RA34PW2 to a deterministic Euclidean
`DifferentialProblem`. `solve_rosenbrock_adaptive` realizes a bounded accepted-step
schedule and replays it with the controller choices stopped from differentiation.
Jacobian products are JVPs and every stage uses a native `AbstractLinearOperator`; the
state Jacobian is not materialized.
`solve_implicit_runge_kutta` solves all Gauss stages as one nonlinear system. One,
two, and three stages have orders two, four, and six. `dense=True` retains the
collocation polynomial for arbitrary query times.

## Fixed-step rollout retention

`FixedStepRolloutPlan` separates full-state retention from full, step, or block
reverse-mode replay:

- `retention="final"` returns only the terminal state;
- `retention="checkpoints"` returns the initial state, every requested stride, and
  the terminal state exactly once;
- `retention="trajectory"` returns every endpoint and matches the legacy
  `solve_fixed_step` state layout.

Built-in scalar histories remain one value per physical step. An optional diagnostic
callback observes the fail-closed accepted endpoint and must return scalar array
leaves. It does not force population/state trajectory retention.

`replay=FixedStepReplayPolicy("step")` rematerializes deterministic per-step work.
`FixedStepReplayPolicy("block", block_size=...)` retains block-boundary carries and
recomputes each block. Replay does not change primal values or output retention.

::: phydrax.solver.FixedStepRolloutPlan

---

::: phydrax.solver.FixedStepRolloutResult

::: phydrax.solver.FixedStepReplayPolicy

## Differentiation

- Diffrax paths use the selected Diffrax adjoint.
- Fixed native implicit methods differentiate every converged root implicitly.
- Adaptive native BDF first realizes a mesh, then replays the accepted times, orders,
  and steps with those discrete choices stopped from differentiation.
- Adaptive Rosenbrock-W uses the same realize-then-replay frozen-grid derivative.
- Method coefficients, partitions, capacities, and requested grids are nontrainable.
- Failed adaptive primals have no valid derivative.

## Stochastic compatibility

Phydrax validates solver interpretation, declared noise structure, and required
Levy-area depth. In particular, additive-only SRK methods reject commutative/general
noise and Milstein rejects undeclared general noise. Additive noise may use either an
Ito- or Stratonovich-marked solver because both interpretations coincide.

## Provenance

`DifferentialSolution.temporal_evidence` records the method capabilities, complete
solver/controller/adjoint/event configuration identity, backend, equation form,
adaptivity, dense-output choice, and step capacity. Runtime parameter values remain
ordinary JAX leaves rather than being folded into static identities.

## Particle conversion and reactive scheduling

::: phydrax.solver.ParticleConversionBackend

---

::: phydrax.solver.ParticleConversionSolverPlan

---

::: phydrax.solver.advance_particle_conversion

---

::: phydrax.solver.HybridEventPlan

---

::: phydrax.solver.localize_hybrid_event

---

::: phydrax.solver.ParticleConversionSensitivityPolicy

---

::: phydrax.solver.ReactiveParticleCouplingSchedulePlan

---

::: phydrax.solver.advance_reactive_cfd_dem_window

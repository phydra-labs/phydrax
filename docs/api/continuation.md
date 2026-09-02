# Continuation and bifurcation

`phydrax.continuation` owns generic parameterized residual curves, reusable bordered
corrections, stability evidence, event localization, branch switching, and local
fold/Hopf/pitchfork workflows. `phydrax.dynamics` may supply equilibrium,
periodic-orbit, or Floquet operators, but does not maintain a second continuation
runtime.

## Residual curves and physical parameter paths

`ContinuationCurveProblem` defines `F(state, coordinate, args) = 0` on one declared
real scalar coordinate. `ParameterContinuationProblem` is the direct scalar-parameter
case. `ParameterPathContinuationProblem` maps that scalar coordinate to a validated
physical-parameter PyTree. Each accepted `BranchPoint` stores both the scalar
coordinate/tangent and physical parameters/tangent parameters; generic artifacts do
not relabel the scalar coordinate as a physical parameter.

```python
import jax.numpy as jnp
import phydrax as phx

problem = phx.continuation.ParameterContinuationProblem(
    lambda state, coordinate, _: {
        "x": state["x"] ** 2 + coordinate - 1.0,
    },
    problem_id="quadratic-fold",
)
result = phx.continuation.continue_branch(
    problem,
    {"x": jnp.asarray(1.0)},
    jnp.asarray(0.0),
    num_steps=12,
    method=phx.continuation.PseudoArclengthContinuation(
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-8,
            relative_residual=0.0,
        ),
        initial_step=0.18,
        maximum_step=0.24,
    ),
)
```

Natural continuation uses the coordinate as the step variable and exposes its
turning-point limitation. Pseudo-arclength continuation predicts in the augmented
state-coordinate space and corrects with an arclength equation. Step growth,
contraction, retries, tangent orientation, and termination all remain explicit.
`terminal_coordinate` requests an exact corrected section: natural continuation
shortens its coordinate step, while pseudo-arclength corrects the first branch segment
that crosses the requested coordinate. Exhausting the branch before that section and
failing its fixed-coordinate corrector are distinct terminal statuses.

Each continuation method composes one complete
`phydrax.nonlinear.AbstractNonlinearMethod` with `NonlinearTermination`.
`NewtonKrylov` and `NewtonTrustRegion` retain prepared Jacobian, linear-plan, and
preconditioner state across compatible branch steps. Other complete nonlinear methods
remain valid correctors but run without cross-step prepared-root reuse. Finite
`AbstractNonlinearUpdate` values are not correctors by themselves; wrap them in a
complete method such as `NonlinearRichardson`.

## Public states, real execution coordinates, and geometry

`ContinuationRepresentationPolicy` binds optional `AbstractRealCoordinateMap` values
for public state and residual storage. The branch remains public-facing while all
corrector, tangent, and arclength work runs in canonical real execution coordinates.
This supports native-complex arrays, finite-real-algebra layouts, and constrained
Hermitian spectral coordinates without introducing a second packing convention.
Constrained representatives must satisfy their declared defect tolerance; continuation
never projects them silently.

`ContinuationGeometry` binds public and execution spaces, their pairings, coordinate
maps, and one positive `coordinate_scale`. State and residual PyTrees may differ when
their real coordinate dimensions and dtypes agree. Execution-space overrides may
supply a mesh or mass pairing while retaining a coordinate map's representation.
Every accepted branch stores this geometry once, and localization, stability,
nullspace evidence, normal forms, and branch switching consume the same artifact.

Natural continuation supports constant or implicit-tangent predictors.
Pseudo-arclength supports metric-normalized secants or a full bordered tangent solve.
`minimum_tangent_alignment` rejects excessive branch curvature before stability
analysis, monitors, or prepared-state commitment.

::: phydrax.continuation.ContinuationRepresentationPolicy

---

::: phydrax.continuation.ContinuationGeometry

---

::: phydrax.continuation.ContinuationCurveProblem

---

::: phydrax.continuation.ParameterContinuationProblem

---

::: phydrax.continuation.ParameterPathContinuationProblem

---

::: phydrax.continuation.NaturalParameterContinuation

---

::: phydrax.continuation.PseudoArclengthContinuation

---

::: phydrax.continuation.BranchPoint

---

::: phydrax.continuation.ContinuationBranch

---

::: phydrax.continuation.ContinuationResult

---

::: phydrax.continuation.ContinuationStatus

---

::: phydrax.continuation.ContinuationDiagnostics

---

::: phydrax.continuation.ContinuationProvenance

## Plan, prepare, refresh, and run

`plan_continuation` validates the immutable method, complete nonlinear corrector,
termination policy, stability analyzer, monitors, step-count, optional terminal
coordinate, and identifiers. `prepare_continuation` binds a numerical seed and owns
reusable prepared state when the selected corrector supports it.
`refresh_continuation` accepts a same-structure seed, preserves plan and prepared
identity, and increments the numeric version. `run_continuation` executes that exact
prepared artifact. `continue_branch` is the one-call convenience wrapper.

The core pseudo-arclength corrector solves the full augmented Jacobian system. The
separate bordered subsystem exposes a principal-Schur strategy for workflows that
explicitly know the principal operator is invertible. A principal solve or Schur
failure is not treated as proof that the full augmented system is singular, and
neither path substitutes a least-squares step.

Tangent and adjoint linear policies come from
`phydrax.nonlinear.ImplicitRootDerivativePolicy`. With Newton correctors they default
to the corrector linear policy; derivative-based continuation with another nonlinear
method must supply the required tangent policy explicitly.

::: phydrax.continuation.ContinuationPlan

---

::: phydrax.continuation.PreparedContinuation

---

::: phydrax.continuation.plan_continuation

---

::: phydrax.continuation.prepare_continuation

---

::: phydrax.continuation.refresh_continuation

---

::: phydrax.continuation.run_continuation

---

::: phydrax.continuation.continue_branch

---

::: phydrax.continuation.BorderedLinearSystem

---

::: phydrax.continuation.BorderedSolvePlan

---

::: phydrax.continuation.PreparedBorderedSolve

---

::: phydrax.continuation.BorderedSolveResult

---

::: phydrax.continuation.BorderedSolveStatus

---

::: phydrax.continuation.plan_bordered_solve

---

::: phydrax.continuation.prepare_bordered_solve

---

::: phydrax.continuation.refresh_bordered_solve

---

::: phydrax.continuation.solve_bordered

## Stability and event evidence

`DenseSchurStabilityAnalyzer` is an explicit small-system path.
`SelfAdjointKrylovStabilityAnalyzer` uses the self-adjoint spectral contract.
`GeneralKrylovStabilityAnalyzer` constructs a matrix-free Jacobian operator and uses
native restarted Arnoldi through the public general-eigen API. All analyzers act on
the bound real execution operator, so nonholomorphic complex public residuals retain
their full real-linear spectrum. Automatic stability requires an execution-space
endomorphism; incompatible state and residual spaces return `CAPABILITY_REJECTED`
rather than an arbitrary coordinate-rebased spectrum.

Crossing monitors produce `ContinuationEvent` and `EventBracket` records. A bracket
is a candidate interval, not a bifurcation certificate. `localize_event` repeatedly
interpolates a state/coordinate seed, solves the full augmented correction through a
complete nonlinear method, evaluates the caller-declared indicator, and updates the
bracket. It returns explicit success, invalid-bracket, corrector-failure, nonfinite,
and maximum-step states.

::: phydrax.continuation.DenseSchurStabilityAnalyzer

---

::: phydrax.continuation.SelfAdjointKrylovStabilityAnalyzer

---

::: phydrax.continuation.GeneralKrylovStabilityAnalyzer

---

::: phydrax.continuation.StabilityEvidence

---

::: phydrax.continuation.StabilityAnalysisStatus

---

::: phydrax.continuation.BifurcationIndicators

---

::: phydrax.continuation.ContinuationEvent

---

::: phydrax.continuation.EventBracket

---

::: phydrax.continuation.EventLocalizationPolicy

---

::: phydrax.continuation.EventLocalizationResult

---

::: phydrax.continuation.EventLocalizationStatus

---

::: phydrax.continuation.localize_event

## Monitors and validated branch switching

An `AbstractBranchMonitor` observes accepted points and appends user-defined evidence.
A branch-switch hook receives a complete accepted branch and its events, then returns
validated `BranchSeed` values. `propose_branch_seeds` records parent point, branch and
hook IDs. `switch_branches_from_nullspace` additionally consumes independently
validated nullspace evidence. No workflow treats an indicator crossing as proof of a
new branch.

::: phydrax.continuation.AbstractBranchMonitor

---

::: phydrax.continuation.CallableBranchMonitor

---

::: phydrax.continuation.AbstractBranchSwitchHook

---

::: phydrax.continuation.CallableBranchSwitchHook

---

::: phydrax.continuation.BranchSeed

---

::: phydrax.continuation.propose_branch_seeds

---

::: phydrax.continuation.switch_branches_from_nullspace

## Extended fold, Hopf, and pitchfork workflows

Fold and Hopf solvers operate on explicit augmented residual blocks. Their results
separate numerical convergence from a problem-specific certificate. Callable
nullspace and Hopf analyzers let domain code supply independently justified evidence;
the core runtime does not guess symmetry, transversality, or normal-form assumptions.
Right nullvectors belong to the execution state space, left nullvectors belong to the
execution residual space, and certificate pairings use the bound Riesz geometry.

`fold_normal_form`, `hopf_first_lyapunov`, and `pitchfork_normal_form` consume
`ContinuationGeometry`, explicit multilinear actions, and a caller-declared range
solver. Diagnostics count every multilinear and linear action.
`CallableNormalFormLinearSolver` must report residual and status evidence; returning
an array alone is insufficient. Hopf normal forms require an execution-space
endomorphism; mapped public states that would require a higher-complex perturbation
representation are rejected explicitly.

::: phydrax.continuation.FoldProblem

---

::: phydrax.continuation.FoldMethod

---

::: phydrax.continuation.FoldState

---

::: phydrax.continuation.FoldResult

---

::: phydrax.continuation.FoldResidualBlocks

---

::: phydrax.continuation.HopfProblem

---

::: phydrax.continuation.HopfMethod

---

::: phydrax.continuation.HopfState

---

::: phydrax.continuation.HopfResult

---

::: phydrax.continuation.HopfResidualBlocks

---

::: phydrax.continuation.BifurcationCertificate

---

::: phydrax.continuation.BifurcationTolerances

---

::: phydrax.continuation.certify_fold

---

::: phydrax.continuation.certify_hopf

---

::: phydrax.continuation.certify_pitchfork

---

::: phydrax.continuation.certify_branch_point

---

::: phydrax.continuation.CallableNullspaceAnalyzer

---

::: phydrax.continuation.CallableHopfAnalyzer

---

::: phydrax.continuation.evaluate_nullspace

---

::: phydrax.continuation.NormalFormPolicy

---

::: phydrax.continuation.CallableNormalFormLinearSolver

---

::: phydrax.continuation.FoldNormalFormResult

---

::: phydrax.continuation.HopfNormalFormResult

---

::: phydrax.continuation.PitchforkNormalFormResult

---

::: phydrax.continuation.fold_normal_form

---

::: phydrax.continuation.hopf_first_lyapunov

---

::: phydrax.continuation.pitchfork_normal_form

## Homotopy and deflation

`linear_homotopy` and `parameter_homotopy` reuse the continuation lifecycle and add an
endpoint certificate evaluated at the declared target. `RootDeflation` modifies a
root residual by distances to known roots under an explicit metric.
`solve_deflated` returns both the transformed solve evidence and the original physical
residual; success requires the original residual to pass.

::: phydrax.continuation.HomotopyProblem

---

::: phydrax.continuation.HomotopyEndpointCertificate

---

::: phydrax.continuation.HomotopyEndpointStatus

---

::: phydrax.continuation.linear_homotopy

---

::: phydrax.continuation.parameter_homotopy

---

::: phydrax.continuation.RootDeflation

---

::: phydrax.continuation.DeflationPolicy

---

::: phydrax.continuation.VectorSpaceDeflationMetric

---

::: phydrax.continuation.CallableDeflationMetric

---

::: phydrax.continuation.DeflatedRootResult

---

::: phydrax.continuation.DeflatedRootStatus

---

::: phydrax.continuation.solve_deflated

## Declared generalized pencils and Hopf loci

`ContinuationStabilityPencil` supplies a square standard or generalized pencil on
one declared stability space. A rectangular residual is analyzable only when the
caller supplies both a lift and projection; absent that closure it has no intrinsic
stability spectrum. The DAE adapter uses the documented `A = -F_y`,
`B = F_ydot` convention and preserves finite/infinite generalized-mode evidence.

`GeneralizedPencilStabilityAnalyzer` routes the pencil through PhydraX's native
general eigensolve. `HopfContinuationAdapter` represents the conjugate mode in real
blocks, adds normalization and reference-phase equations, and returns an ordinary
`ContinuationCurveProblem` for the existing plan/prepare/run lifecycle.
`HopfPointEvidence` is accepted only for positive frequency, small real-block
residual, spectral isolation, transversality, and a full-rank augmented corrector.
This is local evidence for the finite represented branch, not branch completeness,
center-manifold validity, or a normal-form coefficient.

::: phydrax.continuation.ContinuationStabilityPencil

---

::: phydrax.continuation.GeneralizedPencilStabilityAnalyzer

---

::: phydrax.continuation.HopfContinuationAdapter

---

::: phydrax.continuation.hopf_point_evidence

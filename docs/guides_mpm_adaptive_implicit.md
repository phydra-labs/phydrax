# Adaptive and implicit MPM

## Adaptive explicit execution

`AdaptiveMPMRolloutPlan` performs a bounded fixed-capacity attempt scan. Only this
controller may retry. Every attempt records requested/stable/suggested step, limiter,
retry number, status, rejection flags, route digest, schedule code, and topology
generation.

Accepted endpoints form `RealizedTemporalMesh`. Controller decisions use stopped
values. Use `ScheduledMPMRolloutPlan.from_realized` for replay and differentiation;
the derivative is conditional on the accepted temporal and branch journal.

## Backward-Euler implicit MPM

`PreparedImplicitMPMDynamics` uses nodal velocity as the global unknown. Routes and
active/free components are frozen for one nonlinear solve. The residual is:

```text
R(v) = M (v - v_before) - dt [f_internal(F(v)) + f_external]
F(v) = [I + dt sum_i(v_i outer grad N_i)] F_n.
```

Zero-mass nodes are identity constrained and prescribed velocity components are
eliminated through identity residual rows. This keeps the root square and prevents a
singular inactive-node nullspace.

The solver uses `NonlinearSystemProblem`, matrix-free `NewtonKrylov`, Phydrax linear
policies, and `implicit_root_result`. Material response is recomputed from committed
history at every residual evaluation. History is committed only after a converged,
admissible global root.

`AbstractImplicitMPMConstitutivePlan` supplies the algorithmic `dP/dF` evidence. The
same plane-stress Schur complement and finite-strain J2 tangent used explicitly are
used for implicit compatibility checks.

Initial support is one dense logical nodal field, fixed routes, no sharp contact, and
no compact sparse operator. uGIMP, multifield, smooth/VI contact, and compact sparse
implicit operators require separate qualified adapters.

Implicit derivatives use the root implicit-function theorem and qualified tangent and
transpose linear solves. Newton iteration replay is not the derivative contract.

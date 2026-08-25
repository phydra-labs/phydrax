# Boundary layer potentials

Boundary layer fields are finite weighted sums of PDE fundamental solutions. Phydrax
keeps two guarantees separate:

1. **PDE membership:** every finite Laplace or Helmholtz kernel sum satisfies the
   homogeneous PDE exactly at targets outside its singular support.
2. **Operator approximation:** panel density, quadrature, principal-value trace, jump,
   boundary-condition, and close-evaluation errors describe how accurately the finite
   sum approximates an intended continuous layer potential.

Quadrature never weakens the first claim. It selects a different exact homogeneous
solution.

## Two-dimensional Laplace substrate

`BoundaryPanelization2D` lowers an oriented `BoundaryAtlas` to fixed
Gauss–Legendre source nodes, normals, Jacobians, physical weights, panel IDs, and a
content-addressed singular-support identity.

```python
geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
panelization = phx.operators.BoundaryPanelization2D(
    geometry.boundary_atlas,
    panels_per_chart=8,
    quadrature_order=8,
    geometry=geometry,
)
```

`LaplaceLayerPotential2D` supports single- and outward-source-normal double-layer sums.
Its `TrialSpaceCertificate` remains algebraic and has validity region
`off-singular-support`. The certificate contains no numerical clearance threshold.

```python
evaluation = phx.operators.evaluate_layer_potential(
    potential,
    targets,
    phx.operators.LayerEvaluationPlan2D(accuracy_clearance=0.05),
    target_side="interior",
)
values = evaluation.values
target_report = evaluation.target_report
evaluation_report = evaluation.evaluation_report
```

`LayerPotentialTargetReport` checks the complete continuous boundary through the
panelization's exact, sign-reliable signed-distance and region queries. The report
separates:

- target-side membership and boundary intersection, which determine whether the
  pointwise PDE claim applies;
- policy-defined clearance, which affects supported evaluation accuracy only.

Construct the panelization with its compiled geometry to enable these reports.
Panelizations created from a bare atlas still support kernel evaluation, but refuse
target admissibility certification because quadrature nodes cannot certify continuous
boundary exclusion.

A target may be arbitrarily close to the boundary and remain in the PDE nullspace when
the geometry query resolves it strictly off the boundary. A failed accuracy-clearance
policy never changes the PDE certificate or its ID.

Pass that same report to `audit_trial_space(..., admissibility=target_report)`.
For an `off-singular-support` certificate, the audit checks the report's continuous
support identity, target fingerprint, target count, and PDE-domain membership before
constructing any differential residual. Boundary targets and mismatched reports are
rejected. `evaluation_accuracy_supported` remains a separate audit result and does not
alter PDE membership.

`LayerDiscretizationReport` records panelization, quadrature, density space, and
trace policy independently. `LayerEvaluationReport` records the explicit evaluator,
target binding, finite status, and evaluator-specific error evidence. The direct B0
evaluator deliberately reports `unestimated-direct`; it does not claim close-target
accuracy merely because the target-clearance policy passes.

## Interior Dirichlet solve

`solve_interior_laplace_dirichlet_2d` uses an outward-normal double-layer
representation and the fixed interior jump convention. The principal-value matrix uses
local removable-diagonal limits and is routed through `phydrax.linalg`.

```python
boundary_values = jnp.ones((panelization.node_count,))
result = phx.solver.solve_interior_laplace_dirichlet_2d(
    panelization,
    boundary_values,
)
assert bool(result.valid)
```

The result retains linear-solve diagnostics and separate layer discretization
evidence. The returned off-surface potential carries the algebraic Laplace certificate.

## Sign convention

- normals point outward from the bounded interior;
- the fundamental solution is `-log(|x-y|)/(2π)`;
- the source-normal derivative defines the double layer;
- the interior double-layer trace uses `K - I/2`.

Other references may use opposite signs or interchange interior/exterior `+` and `-`
labels. Phydrax uses semantic side names and regression-tests this convention on the
unit circle.

## Adaptive near and self evaluation

`LayerEvaluationPlan2D("adaptive", ...)` classifies target-to-panel regimes and
uses the shared `AdaptiveQuadraturePlan` engine for every panel. The global report
aggregates all panel errors and refuses accuracy support unless the accumulated
bound satisfies the requested tolerance. A boundary source-node single layer uses
logarithmic product regularization; its status and error remain in the evaluator report.

## Corners and grading

`BoundaryCornerTopology2D` declares chart endpoints and opening angles.
`BoundaryPanelPartition2D` supports uniform, Kress, and dyadic endpoint grading.
The partition is content-addressed and is stored by `BoundaryPanelization2D`; geometry
derivatives may vary while the discrete topology remains fixed.

## Helmholtz combined fields

`HelmholtzLayerKernel2D` uses the outgoing Hankel fundamental solution. Exterior
Dirichlet solves use the Brakhage--Werner field
`D - i eta S`. The solver requires an explicit `AdaptiveQuadraturePlan` for logarithmic
self-block product integration and returns a `BoundaryOperatorAssemblyReport`; failed
corrected blocks cannot enter the CFIE matrix.

## Local QBX expansions

`LayerEvaluationPlan2D("qbx", qbx_order=..., qbx_radius_factor=...)` evaluates the
analytic finite layer field from target-associated local Taylor expansions. Truncation
error is reported separately. Boundary targets are evaluated by averaging the two
one-sided local expansions; PDE membership remains off-support only.

## Three-dimensional surfaces

`SurfacePanelization3D` lowers triangular boundary charts through a reference-triangle
rule. `LaplaceLayerPotential3D` and `evaluate_laplace_layer_3d` require compiled,
continuous geometry evidence and reject unresolved or on-surface direct targets.

## Reference near/far backend

`AbstractLayerBackend` separates backend execution from layer representation.
`DirectNearFarReferenceBackend2D` is an exact direct decomposition used for parity
and work-accounting tests. It is not an acceleration backend and makes no FMM claim.
An eventual FMM backend must add its own approximation and adjoint evidence without
replacing singular or near-panel corrections.

## Current support boundary

- 2D Laplace direct, adaptive near/self, corner grading, and coefficient-quadrature QBX;
- 2D outgoing Helmholtz kernels and explicit Brakhage--Werner CFIE assembly;
- direct 3D Laplace triangular surface panels with target-centered Duffy self rules;
- 3D coefficient-quadrature QBX with continuous signed-distance clearance;
- explicit direct near/far reference accounting;
- genuine 2D Laplace FMM M2M/M2L/L2L translations;
- global 2D QBX/FMM coupling with panel coefficient near corrections.

Still separate:

- topology-changing geometry derivatives.

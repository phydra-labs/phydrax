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
potential = phx.operators.LaplaceLayerPotential2D(
    panelization,
    kind="double",
    density=density,
)
values, target_report = potential.evaluate_with_report(
    targets,
    target_side="interior",
    accuracy_clearance=0.05,
)
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

`BoundaryLayerApproximationReport` records panelization, quadrature, density space, and
trace policy independently.

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

The result retains linear-solve diagnostics and separate boundary-layer approximation
evidence. The returned off-surface potential carries the algebraic Laplace certificate.

## Sign convention

- normals point outward from the bounded interior;
- the fundamental solution is `-log(|x-y|)/(2π)`;
- the source-normal derivative defines the double layer;
- the interior double-layer trace uses `K - I/2`.

Other references may use opposite signs or interchange interior/exterior `+` and `-`
labels. Phydrax uses semantic side names and regression-tests this convention on the
unit circle.

## Current support boundary

- curves in two dimensions;
- scalar Laplace kernel;
- fixed Gauss–Legendre panelization;
- smooth off-surface evaluation;
- local-diagonal Nyström principal-value trace;
- dense interior Dirichlet solve.

Not yet claimed:

- Kress logarithmic product integration;
- corner-graded meshes;
- close-evaluation error estimates or QBX;
- three-dimensional singular quadrature;
- Helmholtz combined fields;
- FMM acceleration;
- topology-changing geometry derivatives.

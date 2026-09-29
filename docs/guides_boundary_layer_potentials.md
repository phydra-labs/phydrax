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

```text
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

```text
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

```text
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
uses the shared `AdaptiveQuadraturePlan` engine for every panel. Breakpoints are
routed only to panels whose reference intervals contain them, and `throw=True`
remains authoritative through panel correction and QBX coefficient quadrature.
The global report aggregates all panel errors and refuses accuracy support unless
the accumulated bound satisfies the requested tolerance. A boundary source-node
single layer uses one shared barycentric density reconstruction and logarithmic
product regularization; its status and error remain in the evaluator report.

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
analytic finite layer field from target-associated local Taylor expansions.
Centers use the geometry's target boundary normal rather than a nearest quadrature
normal. Coefficient-quadrature errors propagate, and the omitted analytic tail is
bounded from the retained term and certified expansion clearance. Boundary targets
are evaluated by averaging the two one-sided local expansions; a tangent expansion
disk without positive convergence margin cannot claim finite accuracy.

## Three-dimensional surfaces

`SurfacePanelization3D` maps the reference rule through each declared affine
triangular trim cell, retains that cell and its prepared inverse map, and rejects
non-triangular or holed reference cells rather than integrating the wrong domain.
QBX and target-centered Duffy evaluation share its panel-polynomial density
reconstruction. `LaplaceLayerPotential3D` and `evaluate_laplace_layer_3d` require
compiled, continuous geometry evidence and reject unresolved or on-surface direct
targets.

## Three-dimensional DP0 Galerkin capacitance

`prepare_laplace_single_layer_dp0_3d` binds one fixed `MeshRegion` to the existing
triangle-DP0 finite-element space and `SurfacePanelization3D`. The initial contract
accepts closed, outward-oriented components with strictly separated component bounding
boxes. Nested, intersecting, inward-oriented, open, curved, or topology-changing surfaces
are rejected.

The prepared weak operator maps DP0 density coefficients to DP0 test covectors. Its
Gram map is the diagonal face-area matrix `M`; the solve-facing strong operator is
`M⁻¹ V`. The single-layer operator contains no trace jump. Singular and near interactions
are removed from the streamed regular complement and evaluated by class-specific
coincident, shared-edge, shared-vertex, or bounded near rules. Production actions cannot
materialize. A dense oracle exists only when explicitly requested under a
`MaterializationPolicy`.

```text
region = phx.geometry.MeshRegion(vertices, faces)
galerkin = phx.operators.prepare_laplace_single_layer_dp0_3d(region)
epoch = phx.operators.BoundaryMeshEpoch(galerkin._binding.mesh)

left = phx.discretization.EntitySelection(galerkin.surface_entities, left_face_mask)
right = phx.discretization.EntitySelection(galerkin.surface_entities, right_face_mask)

prepared = phx.solver.LaplaceCapacitancePlan3D(
    epoch,
    galerkin,
    {"left": left, "right": right},
).prepare()
result = prepared.solve(permittivity=epsilon)
```

Conductor selections are canonicalized by name. They must be nonempty, disjoint, cover
every face, and assign each connected surface component wholly to one conductor. Column
`j` solves the unit-voltage problem for conductor `j`. The solved layer density `mu`
satisfies `V mu = g`; physical charge density is `epsilon * mu`, and entry `(i, j)` of
`result.capacitance` is its face-area integral over conductor `i`.

`result.potentials[j]` is an ordinary `LaplaceLayerPotential3D` constructed from `mu`,
not from the permittivity-scaled charge. It therefore works unchanged with
`evaluate_laplace_layer_3d` and `evaluate_qbx_3d`. Changing permittivity scales charge
and capacitance but not the unit-excitation potential.

The default Jacobi-preconditioned FGMRES route is a bounded baseline for a first-kind
equation, not a mesh-independent preconditioner. Per-column linalg diagnostics and the
capacitance reciprocity defect remain observable. `valid` certifies finite assembly and
successful solves; no continuum BEM discretization-error estimator is claimed. Solve
differentiation, accelerated far fields, mixed boundary conditions, and fused
multi-right-hand-side execution remain outside this contract.

## Two-dimensional Laplace Galerkin operator

`BoundaryPanelization2D` and `LaplaceLayerPotential2D` carry quadrature-node densities,
and `double_layer_principal_value_matrix` is a Nyström/collocation matrix. Neither is a
Galerkin boundary operator. `prepare_scalar_laplace_galerkin_2d` is the native
Galerkin product for the same kernel `G = -log|x-y|/(2π)` and the same outward-normal
convention.

**Geometry.** `ClosedPolygonalCurve2D(vertices, source_id=...)` is one closed simple
polygon with exact straight panels. Vertices keep their declared order; panel `k` joins
vertex `k` to `k + 1`. Either traversal is accepted and every normal points from the
bounded interior to the unbounded exterior. Zero-length panels, zero area, fold-back
corners, and touching or crossing non-adjacent panels are refused. Curved charts are
not approximated.

**Trace spaces.** `ScalarBoundarySpaces2D` publishes the continuous P1 Dirichlet trace
(vertex coefficients) and the DP0 conormal trace (panel coefficients) as
`BoundaryTraceSpace2D` records. Both vector spaces carry physical arc-length pairings:
DP0 uses the diagonal of panel lengths, P1 uses its tridiagonal Gram map with a
prepared native PCG inverse. `mixed_mass` is the DP0 x P1 duality pairing
`∫ q φ ds`; `integral_weights` are the exact covectors `∫ basis ds`.

**Operators.** `single_layer` is the weak `V` (DP0 x DP0) and `double_layer` the weak
`K` (DP0 test x P1 trial); both map into the DP0 dual. For a bounded exterior harmonic
field, `u = c + Dφ - Sq` with `∫ q ds = 0`, where `φ` and `q` are the exterior
Dirichlet and outward conormal traces and `c` is the far-field constant. Its exterior
trace, using `γ0⁺D = K + I/2`, gives

```text
(M/2 - K) φ + V q - m c = 0
```

tested with DP0, where `M` is the mixed mass and `m` the panel lengths. The interior
relation `(M/2 + K) φ - V q = 0` is a different equation. `exterior_relation` publishes
the rectangular block map from `(dirichlet_trace, conormal, far_field_constant)` to
`(exterior_boundary_equation, total_conormal)`.

**Quadrature.** Every panel pair is classified. Coincident pairs use closed forms (the
straight-panel double layer vanishes, which is its principal value). Shared-endpoint
pairs use the Duffy map about the shared vertex: the `ρ log ρ` part of `V` and the `ρ`
cancellation of `K` are integrated exactly and the smooth angular remainder uses
adaptive Gauss--Kronrod (21) quadrature. Near pairs integrate the exact straight-panel
inner integrals with adaptive Kronrod quadrature over the test panel. Regular pairs use
a fixed tensor Gauss--Legendre rule at run time; preparation certifies every regular pair
against the exact-inner Kronrod reference and promotes any pair above the tolerance to
the near class. `ScalarLaplaceGalerkinReport2D` reports per-class counts, normalized
maximum errors, evaluations, adaptive depth, promoted pairs, and preparation, resident,
and per-action bytes. Its `support` field is the exact candidate
`BoundarySupportEnvelope`, with unsupported claims stated explicitly.

**Execution.** Actions are blocked-direct: regular pairs are summed block by block
(`lax.fori_loop`) with sparse exception corrections, and no panel-pair tensor is
retained. Transposes and pairing adjoints are available. Dense matrices exist only
through `phx.linalg.materialize` under an explicit `MaterializationPolicy`. No FMM
Galerkin route is provided.

```text
curve = phx.operators.ClosedPolygonalCurve2D(vertices, source_id="outer-boundary")
galerkin = phx.operators.prepare_scalar_laplace_galerkin_2d(curve)
projection = phx.operators.prepare_boundary_trace_projection_2d(galerkin.spaces, order=8)
dirichlet = projection.project_dirichlet(trace_samples)   # declared L2 projection
prepared = phx.operators.prepare_exterior_laplace_dirichlet_2d(
    galerkin, far_field="decaying", far_field_tolerance=1.0e-3
)
result = phx.operators.solve_exterior_laplace_dirichlet_2d(prepared, dirichlet.coefficients)
field = galerkin.evaluate_field(
    targets,
    side="exterior",
    dirichlet=dirichlet.coefficients,
    conormal=result.conormal,
    far_field_constant=result.far_field_constant,
)
```

**Bordered exterior solve.** `prepare_exterior_laplace_dirichlet_2d` assembles the
square system with unknowns `(conormal, far_field_constant)` and equations
(exterior boundary equation, `m^T q = 0`). Each equation passes through the inverse
Riesz map of its row space, so the operator is an endomorphism usable by native
Krylov methods; a dense direct policy materializes only within its own budget. The
system is nonsingular because `V` is positive definite on zero-mean densities. The
far field is declared: `bounded` accepts any solved constant; `decaying` additionally
requires `|c|` below `far_field_tolerance`, and a nonzero constant is reported as an
unsatisfied far field rather than silently accepted. The result keeps the native linear
status and recertifies the original weak equation and the zero-total-conormal row.
Only `rhs-only` (the default) or `none` differentiation is admitted.

**Derivatives.** Geometry, pair classes, and singular corrections are prepared once on
the host and never traced. `ScalarLaplaceGalerkin2D.derivative_capability` admits the
densities `dirichlet` and `conormal` and the `far_field_constant` as direct inputs:
`V`, `K`, `exterior_relation`, and `evaluate_field` are linear in them, so a JVP is the
action itself and a VJP is its transpose. Field target positions, geometry, kernel,
and quadrature derivatives are refused; a transformation that requests one raises
`derivative-unsupported`. The exterior solve admits the Dirichlet data under
`rhs-only` as an implicit solution-map derivative of `(q, c)`. Its result keeps
`accepted` separate from `derivative_valid`; a result that is not accepted (failed
solve, failed recertification, or a violated decaying far field) returns NaN
tangents under the status failure mode. Finite-difference or directional checks of
a `decaying` solve must perturb inside that regime, for example along the trace of a
field that itself decays. Under `none` any Dirichlet-data derivative raises.

**Trace projection.** `prepare_boundary_trace_projection_2d` fixes Gauss--Legendre
sample points on each panel. `project_dirichlet` is the L2 projection onto continuous
P1 and `project_conormal` the L2 projection onto DP0; loads are exact for per-panel
polynomial traces up to `exact_polynomial_degree`. Every result carries its measured
defect `||f - Πf||`; a projected higher-order trace is never labeled exact.
`dirichlet_projection` is the same projection as a linear operator whose Hilbert adjoint
is P1 evaluation at the samples.

**Field evaluation.** `evaluate_field` uses exact straight-panel integrals of the
discrete densities, reports the winding number `-D[1]` and boundary distance of every
target, and refuses to accept on-boundary targets or targets on the wrong side.

Not supported: open curves, several boundary components, curved or moving geometry,
Helmholtz kernels, the hypersingular operator, FMM Galerkin actions, geometry
derivatives, and any continuum discretization-error certificate.

## Boundary trace-space capabilities

Boundary-integral owners publish their boundary coefficient spaces to coupling
consumers as `phydrax.discretization.BoundaryTraceSpaceCapability` records. A record
names the trace `quantity` (`"dirichlet"`, `"neumann"`, `"surface-current"`,
`"surface-current-dual"`), its `representation` (`"continuous-p1"`, `"dp0"`, `"rwg"`,
`"buffa-christiansen"`), the implied Sobolev `conformity`, the `orientation`, the
coefficient-carrying boundary entities, and the geometry `revision_id`, a fingerprint
of the vertex coordinates and cells, so moved or rewound geometry is a new revision.
`coefficient_space` is the owner's native coordinate space, the one its boundary
operators act on; `gram_space` pairs the same coordinates through the physical
arc-length or area Gram map, whose inverse is a prepared Jacobi-PCG Riesz solve, and
`mass` maps them into that dual. Neither record carries a volume support or an
integration domain.

Scalar Cauchy data are published together as a `CauchyTraceCapability`: one owner, one
revision, a Dirichlet and a Neumann part, and the sparse `duality` map
`(B φ)_i = ∫ φ q_i ds`, so `pair(q, φ)` is `∫ q φ ds`. `interior` names the declared
interior; Dirichlet traces are unoriented and the Neumann trace differentiates along the
normal pointing out of it.

```python
cauchy = galerkin.spaces.cauchy_trace_capability()          # 2-D P1/DP0
cauchy3 = calderon.spaces.cauchy_trace_capability()         # 3-D P1/DP0
currents = rwg_space.trace_capability()                      # RWG, H(div_Γ)
energy = cauchy.dirichlet.gram_space.inner(phi, phi)         # ∫ φ² ds
```

| Owner | Dirichlet | Neumann | Orientation of the Neumann trace |
| --- | --- | --- | --- |
| `ScalarBoundarySpaces2D.cauchy_trace_capability()` | vertex P1, tridiagonal arc-length Gram | panel DP0, panel lengths | out of the bounded interior for either declared traversal |
| `ScalarBoundarySpaces3D.cauchy_trace_capability(gram_tolerance=...)` (from `prepare_scalar_calderon_3d`) | vertex P1, `A(1 + δ_ij)/12` area Gram | face DP0, face areas | out of the bounded interior; `MeshRegion` orients each closed component outward for either declared winding |

`RWGSurfaceCurrentSpace3D.trace_capability()` publishes the tangential current as a
distinct `"surface-current"` / `"rwg"` record in `H^(-1/2)(div_Γ)`, paired by the exact
RWG area Gram map; `BuffaChristiansenDualSpace3D.trace_capability()` publishes its
Buffa--Christiansen dual (`"surface-current-dual"`) with the Gram map of its barycentric
RWG representation. Currents follow the oriented surface complex: reversing every
triangle winding negates each basis function and changes the revision. A current record
is never accepted as a scalar Cauchy part, and a representation declared for another
quantity is refused. The 3-D scalar P1 coordinate space now carries the kernel's
coefficient dtype and its own space identity, distinct from the DP0 space.

## Reference near/far backend

`AbstractLayerBackend` separates backend execution from layer representation.
`DirectNearFarReferenceBackend2D` remains the exact parity reference. Treecode and
FMM backends bind the source panelization, require an opening angle strictly inside
`(0, 1)`, and report geometric upper bounds for omitted multipoles rather than the
last retained term. Global QBX/FMM adds coefficient and local-expansion bounds.
Evaluation fingerprints include targets, densities, policy controls, and source
revision identities; singular and near-panel corrections remain direct.

## Current support boundary

- 2D Laplace direct, adaptive near/self, corner grading, and coefficient-quadrature QBX;
- 2D outgoing Helmholtz kernels and explicit Brakhage--Werner CFIE assembly;
- direct 3D Laplace triangular surface panels with target-centered Duffy self rules;
- 3D coefficient-quadrature QBX with continuous signed-distance clearance;
- 3D Laplace DP0 Galerkin single-layer assembly and conductor capacitance solves;
- 2D Laplace P1/DP0 Galerkin `V`/`K`, bordered exterior Dirichlet-to-Neumann solves,
  and declared L2 trace projections on closed straight-panel polygons;
- explicit direct near/far reference accounting;
- genuine 2D Laplace FMM M2M/M2L/L2L translations;
- global 2D QBX/FMM coupling with panel coefficient near corrections.

Still separate:

- topology-changing geometry derivatives.

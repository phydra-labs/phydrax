# Meshfree solvers

Meshfree preparation belongs to `phydrax.discretization.meshfree`. Sparse relations,
linear algebra, exterior calculus, time integration, and coupling remain owned by
the corresponding native substrates. These capabilities are unreleased candidates;
a successful numerical campaign is not scientific release authorization.

## Local approximation

`MeshfreeNeighborhoodPlan` prepares certified Morton nearest-neighbor routes. Its
candidate capacity and target chunk size bound the working set. Duplicate sources,
uncertified neighbors, and capacity overflow are refused rather than replaced with
an all-pairs fallback. Sources and targets may differ. `MeshfreeEdgeRelationPlan`
prepares canonical unordered radius pairs with explicit pair-capacity evidence.
Both accept a canonical support declaration: an explicit `MortonAddressPlan`
(physical cell, per-axis periodicity, depth), stable point identities
(`source_ids`/`point_ids`, which order ties), and a `MeshfreePrecisionPolicy`
(`precision`, float64 roles by default). Coordinates are stored in its geometry
role, and neighbor selection, distances and gap witnesses are decided in its
certification role, which is never narrower than geometry. Without an address a
padded non-periodic box is inferred; a periodic cell is never inferred. Periodic
offsets use the minimum image, and a neighborhood reaching half a periodic length
is refused as ambiguous. Morton addressing uses 64-bit codes, so construction is
refused unless `jax_enable_x64` is on; float32 coordinates are declared with
`MeshfreePrecisionPolicy(geometry_dtype="float32")`.

Periodicity is owned by the domain: a `phydrax.domain.PeriodicIdentification`
glues the lower (source) face of one Cartesian coordinate to its upper (target)
face. The meshfree address is derived from those identifications, never declared
beside them:

```python
box = phx.domain.HyperRectangle(np.zeros(2), np.ones(2))
seams = tuple(phx.domain.PeriodicIdentification(box, "x", component=c) for c in range(2))
address = phx.discretization.spatial.MortonAddressPlan.from_periodic_identifications(
    seams, maximum_depth=10
)
```

Bounds come from the factor faces and exactly the identified coordinates are
periodic. `coordinates=((label, component), ...)` binds point axes to domain
coordinates when the domain has several labels; coordinates identified twice,
identifications on different domains, and identified coordinates that are not
point axes are refused. The ordered identification revisions and coordinate
bindings enter the address `plan_id`, so equal numerical boxes over different
seams are distinct supports. Minimum images, distributed images, edge charts,
and evolution rebase/wrap remain arithmetic on this derived address. A raw
`MortonAddressPlan(lower, upper, depth, periodic_axes=...)` is the domain-less
numerical box used by low-level spatial consumers.

A periodic address is half-open, `[lower, upper)`: it holds one representative
per seam orbit. A cloud carrying both the lower-face node and its upper-face copy
(the closed fundamental box) is refused on a periodic address because the two
coincide after wrapping. Closed-box clouds that keep both faces realize the seam
with periodic boundary rows on a non-periodic address instead.

Without `maximum_candidates` the candidate capacity is derived from the neighbor
count and dimension for quasi-uniform clouds: the coarse 3^d Morton stencil
around a k-th distance `r` spans a cube of side `6 r`, so with local density
varying by at most a factor two it holds at most `2 * 6^d / V_d * (k + 1)`
sources (`V_d` the unit-ball volume; about 23, 103 and 6 times `k + 1` in 2-D,
3-D and 1-D). The derivation is part of the plan identity, and a row exceeding
it is refused as `CANDIDATE_OVERFLOW`, never regrown; graded or clustered clouds
declare `maximum_candidates`.
The stencil fit runs in the precision's fit role and stores weights in its
coefficient role; `LocalStencilReport.precision` records the effective
`(role, dtype)` pairs. A fit at unit roundoff `u` refuses rows whose condition
exceeds `1e-3 / u` as `ILL_CONDITIONED` (float32: about 8.4e3), and
reduced-precision fits refuse moment residuals above their rounding bound as
`MOMENT_FAILURE`; float32 PHS-RBF-FD saddles (condition 1e4-1e6 on stratified
clouds) are therefore refused rather than silently inaccurate and are not a
declared float32 support tuple. Float64 fits keep `condition_limit` and the
1e-9 moment limit. `MeshfreeOperator`
spaces default to the compute (source) and output (target) roles and accumulate
in the accumulation role.
`amplification_limit` bounds the absolute row sum of weights, which scales like
`gap^-order` near closely spaced points; interpolatory PHS stencils exhibit this
directly, so fine or clustered clouds declare a limit matching their spacing.

`LocalStencilPolicy` selects `gmls` or `phs-rbf-fd`, polynomial degree,
conditioning, amplification, and chunk capacity. Method parameters are
method-specific: GMLS owns `weight_kernel` (default `inverse-square`), PHS-RBF-FD
owns its odd `phs_power` (default 3), and supplying the other method's parameter is
refused. `MeshfreeFunctional` declares a linear combination of derivative
multi-indices, including mixed derivatives and row-dependent normal derivatives.
`prepare_local_stencils` returns weights and per-row `LocalStencilEvidence`;
`LocalStencilReport` summarizes the admitted fit. Rank deficiency and a full-rank
but ill-conditioned row have different statuses. The declared condition limit is
enforced, not merely fingerprinted.

Preparation assembles every requested functional once and runs one stable compiled
fit (`fit_chart_stencils`) over padded fixed-size row chunks mapped on device, with
native batched SVD/minimum-norm or local saddle solves; row evidence is reduced on
device and the host synchronizes once, at admission. Results do not depend on the
chunk capacity beyond floating-point batching. Runtime application uses
`MeshfreeOperator` and native sparse routes. Coordinate transpose and Hilbert
adjoint differ when source and target measures differ. Polynomial reproduction does
not establish conservation, PDE stability, or a maximum principle.

`refresh_local_stencils(stencils, sources, targets)` refits the frozen relation at
moved coordinates and returns a `LocalStencilRefresh` with
`LocalStencilRefreshStatus`: the support is certified only while the anchored
displacement is strictly below every row's trust margin (`SUPPORT_EXCEEDED`
otherwise), nonfinite coordinates are `INVALID_COORDINATES`, and refused rows
under `acceptance="refuse"` are `ROW_REFUSED`. The refresh is traceable and
differentiable in the coordinates on the fixed support; a refused refresh
publishes NaN weights with NaN derivatives. Support discovery belongs to a new
preparation (epoch boundary). `refresh_chart_stencils` is the chart-offset form,
taking the chart owner's displacement bound. See
[Sensitivities and nonsmooth events](#sensitivities-and-nonsmooth-events) for the
smooth fixed-radius support.

`MortonRadiusShellWitnessPlan` certifies a radius relation against motion: for a
half width `2 delta` it enumerates every incidence within `radius + 2 delta` on a
fixed candidate buffer (periodic images included) and certifies when none lies in
the closed shell `|d - radius| <= 2 delta`. Ties, candidate overflow, or a missing
owner make the witness uncertified, never an empty shell. Exterior-calculus
preparation uses it (half width `radius/2`) for its topology trust margin, charging
the query work to the spatial witness rather than the metric symbolic budget.

## Strong-form point clouds

`PointCloudPlan(..., stencil=LocalStencilPolicy(...), neighbors=...)` binds points,
positive quadrature, boundary data, local approximation, and optional stable
`point_ids` and `address`. Prepared derivatives retain arbitrary trailing payload
axes. Mixed partials use an explicit derivative multi-index.
`divergence(..., dual=True)` is refused: a nodal component gradient is not an
exterior one-cochain, and no dual calculus is inferred from its shape.
`PreparedPointCloudDiscretization.refresh(points)` returns a `PointCloudRefresh`
candidate on the fixed support: the anchored plan geometry, stable identities,
measures, boundary data, and relation are retained while coordinates and stencil
weights are refitted; consumers inspect `accepted`. A point-cloud reconstruction
anchors the owner's current coordinates; within its declared
`coordinate_envelope` it is differentiated in moved coordinates, and beyond it is
prepared again after a refresh.

Point-cloud observations use `prepare_point_cloud_field_reconstruction` with a
bounded BVH neighborhood and partial coverage. The default `polynomial`
reconstruction retains rank/condition evidence. Explicit `reconstruction="shepard"`
selects a positive, normalized, degree-zero reconstruction. It is a different
accuracy contract, not a fallback after a failed polynomial fit. Complete query
coverage refuses invalid rows; masked coverage retains their statuses. Point
clouds publish no facet trace.

## Elliptic problems

`PointDiffusionOperator` exposes two distinct forms:

- `collocated`: the continuum product-rule approximation of `div(K grad(u))`,
  including mixed second derivatives for tensor `K`.
- `dissipative`: the quadrature-adjoint action `-M^-1 sum_ij D_i^T M K_ij D_j`. Its
  weighted energy is nonpositive by construction. This does **not** establish
  continuum consistency for arbitrary quadrature or an arbitrary cloud.

The diffusivity is declared `kind="scalar"` (`()` or `(points,)`) or
`kind="tensor"` (`(d, d)` or `(points, d, d)`). Finiteness, symmetry, and positive
definiteness are verified with native dense property evidence
(`PointDiffusivityEvidence`: eigenvalue range, condition, symmetry defect), which
reaches every solve result. Invalid fields are refused; nothing is symmetrized or
shifted.

Boundaries are declared per entity. `PointBoundaryCondition(kind, rows, values,
label=..., normals=..., measure=..., robin_coefficient=..., side=..., partners=...,
seam=..., coordinates=..., seam_tolerance=...)` names one boundary entity with a
closed kind (`dirichlet`, `neumann`, `robin`, `periodic`), its equation rows, data,
physical outward normals, and physical boundary measure.
`PointBoundaryPlan(conditions, row_count=..., components=...)`
requires every `(row, component)` to have a single owner, so corners are owned
explicitly, and the owned rows must equal the cloud's declared boundary rows.
Normals are never inferred from coordinates.

A periodic entity is the closed-box point realization of one canonical seam:
`seam` is a `PeriodicIdentification`, or a `phydrax.conditions.Periodic`
relation on its pairing that owns the seam target `g` (then `values` is
omitted). `rows` are the target (upper-face) rows and `partners` their source
(lower-face) images, so value rows enforce the canonical `u(upper) - u(lower) =
g` and partner rows enforce conormal-flux balance. The normals are the
identification's target-face normal and are not declared. Preparation certifies
the discrete pairing against the row coordinates: rows lie on the upper face,
partners on the lower face at equal transverse coordinates (the identification's
face map), within `seam_tolerance` times the larger of the period and the cloud
extent; rows and partners are unique and disjoint, and the shared `measure`
weighs both sides. Unmatched, transverse-mismatched, or reversed pairings are
refused, as are seam rows on a periodic address, which already identifies the
seam. Only identity transport is realized: antiperiodic, Bloch, event-linear,
and jet relations are refused.

`PointCloudPoissonPlan(cloud, boundary, ...)` is the canonical scalar elliptic plan.
`.prepare(diffusivity)` returns `PreparedPointCloudPoisson`; `.solve(source,
boundary_values={label: values})` returns the values, native linear diagnostics,
physical residual, boundary residual, diffusivity evidence, and per-component
compatibility evidence. Numeric coefficient refresh retains symbolic assembly.

For `form="dissipative"`, weak Neumann/Robin rows retain the volume equation:
the load is `M f + B g`, and Robin data add `B alpha` to the stiffness.
`B` comes from declared boundary measures, not unit weights. Dirichlet values
are lifted/eliminated. This single-sided weak route does not accept the
collocated side/interface or periodic-seam row-replacement declarations.

Algebraic convergence of `D^T M K D`, polynomial reproduction, and even the full
SBP coefficient identities do not establish global continuum stability or
accuracy for an arbitrary constrained meshfree derivative. Supplying such a
`PointSBPDerivatives` with `sbp=` preserves its algebraic values and identity
diagnostics, but does not authorize a continuum-success claim.
`PointCloudPoissonResult.continuum_consistent` and scientific `successful`
require a stable realization on the dissipative route. The supplied native
tensor SBP family described below owns that realization; observed analytic
errors and refinement rates remain separate accuracy evidence.
The solve is selected by one native `LinearSolvePolicy` (method, tolerance,
preconditioning, failure, resources). Without a supplied policy, square
collocation below 2048 points uses GMRES with ILU; at 2048 points and above the
plan prepares `plan.hierarchy_plan()` itself and uses GMRES with the native
meshfree multigrid V-cycle (symmetric Gauss–Seidel smoothing, complete sparse
coarse factorization, `reuse-transfers` refresh, Galerkin products bounded by
the plan's `assembly_policy`). The default `assembly_policy` scales with the
declared capacity: with `R` equation rows and stencil width `w`, each product
may hold `R w²` contributions and entries (the former fixed limits are the
floor), and the scaled limits are part of `plan_id`. `plan.hierarchy_plan()` returns a
`MeshfreeHierarchyPlan` that declares `plan.eliminated_rows` (Dirichlet and
gauge identity rows) as `eliminated`, so their identity equations never enter a
Galerkin coarse operator, and the remaining non-bulk rows of square collocation
as `boundary` (kept for the policy's `boundary_retention_levels`). A supplied
prepared hierarchy must eliminate every Dirichlet and gauge row; it selects the
same cycle unless a policy is supplied. `solve` runs through one stable compiled entry, so repeated
solves of a prepared plan do not re-trace the Krylov loops. `plan_id` binds the
boundary and interface declarations, gauges, form, diffusivity kind, linear and
assembly policies, stability policy and assessment, hierarchy, and precision.
Krylov stops are tightened so that
the Krylov certificate implies the original-equation acceptance: lifted
Dirichlet systems stop at the original tolerance, gauged systems at that
tolerance divided by `sqrt(n)` (the replaced gauge row's residual is the
negative sum of the others for a constant left kernel).

**Collocation stability.** Polynomial reproduction does not make a square
collocated Laplacian stable. On irregular (non-quasi-uniform) clouds, GMLS
least-squares rows at nearly coincident points are nearly identical, and their
difference modes are eigenvalues with nonpositive real part (measured on a
random unit-disk cloud with 1024 points: GMLS degree 2/3/4 have 70/98/87 such
eigenvalues; cubic PHS-RBF-FD with 20–30 neighbors has none, smallest real part
≈ `k λ1`). Square collocation therefore declares
`stability: PointStabilityPolicy`, default `"require-assessment"`: preparation and
refresh restrict the native solve operator to the rows that are not eliminated
identity rows (Dirichlet and gauge rows contribute the exact eigenvalue 1) and run
the native general eigensolve (`phydrax.linalg.eigen.general_eigensolve`; the
default `stability_assessment` is a `GeneralEigenSolvePolicy` with
`RestartedArnoldi(restart="krylov-schur")`, `ShiftInvertTransform(0.0)` and
`vectors="right"`). Its shift-invert transform is chosen deterministically from
the declared capacity and recorded in `plan_id`. The fill of a fill-reducing
factor is estimated as `R w` in 1-D, `R w ceil(log2 R)` in 2-D, and
`2 w R^(4/3)` in 3-D (`R` rows, stencil width `w`). While that fits the fixed
sparse-factorization limits, or when the plan has no preconditioner, the
transform is one sparse LU, factored once and refreshed, assessing the 16
eigenvalues closest to zero. Above it the transform is device-bound GMRES on
the assessed block, preconditioned by the plan's own prepared solve
preconditioner restricted to the assessed rows: with the eliminated identity
rows last, the solve operator is `[[B, C], [0, I]]`, so the restriction of its
inverse is exactly `B⁻¹` and no second factorization is built. Each
transformed action then costs about one preconditioned solve, so this route
assesses the 8 closest eigenvalues. Real operators keep a real Krylov basis
(real Krylov–Schur restarts), so each action is one real solve. Each converged estimate `ρ` carries a rigorous
backward error `ε`: `ρ` is an exact eigenvalue of an operator within `ε` of the
solve operator. Preparation raises `PointCollocationStabilityRefusal` (outcome
`"nonpositive-real-part"`) when a converged estimate has `Re ρ <= 0`, and
(outcome `"stability-unassessed"`) when the assessment does not converge within
its declared `max_steps` or its factorization resources are refused.
`"admitted"` means that no converged estimate near zero has `Re <= 0`; it is an
estimate over those modes, not a certificate of the whole spectrum. The same
admission is the plan-independent `assess_square_collocation(physical, solved,
space=, eliminated=, bulk=, policy=, assessment=, assembly_policy=, problem_id=,
prior=, transform_preconditioner=)`, shared by scalar and coupled square collocation, with
`collocation_stability_assessment(rows, width, dimension, preconditioned=)` as
its capacity-selected default assessment. Coefficient refresh is declared by
`stability_refresh: PointStabilityRefresh` (in `plan_id`). The default
`"reassess"` reruns the eigensolve, warm-started from the previous converged
eigenvectors (`refresh_general_eigensolve(..., warm_start=)`), and verifies every
pair against the refreshed operator. `"reuse-within-perturbation"` first
bounds `δ ≥ ‖A' - A‖₂` (`sqrt(‖ΔA‖₁ ‖ΔA‖∞)` against the last assessed operator):
a pair with backward error `ε` keeps backward error `≤ ε + δ`, so it is carried
only while `ε + δ` still meets the assessment's convergence tolerance. A
carried nonpositive pair keeps the refusal. An admission is carried only for
a certified self-adjoint operator (Bauer–Fike), and otherwise the operator is
reassessed. `PointCollocationStability.reused` and `.perturbation_bound` record
which happened. Because collocation operators have `‖A‖` far above the
eigenvalues near zero, certified reuse in practice covers only negligible
changes, so the warm start is what makes refresh cheaper.
`stability="diagnostic"` records the same `PreparedPointCloudPoisson.stability`
(`PointCollocationStability`) evidence and proceeds. The fail-closed default is
deliberate: an unstable collocation produces wrong solutions (GMLS direct-solve
errors 0.04–0.2 against 2.5e-5 for cubic PHS on the measured disk), and the
assessment costs one sparse factorization plus a few dozen solves. Measured on
random unit disks (Dirichlet ring): GMLS degree 2 / 16 neighbors, 1024 points,
is refused with 8 of its 16 nearest eigenvalues at `Re <= 0`, matching the dense
spectrum; cubic PHS / 30 neighbors is admitted with minimum real part 11.583;
cubic PHS / 20 neighbors at 4096 points is admitted (minimum real part 11.560)
although a bulk row has a nonpositive diagonal. The center dominance
`a_ii / sum_j |a_ij|` of bulk rows stays in the evidence as a diagnostic only.
The recommended
square-collocation policy is `LocalStencilPolicy(approximation="phs-rbf-fd",
polynomial_degree=3)` with about three times the polynomial basis size as
neighbors (30 in two dimensions); twice the basis size left one spurious
boundary eigenvalue for Neumann and mixed rows in the measured disk. Uniformly
random clouds also lose stencil conditioning as their separation shrinks (PHS
rows are refused as ill-conditioned at 16384 random disk points); use
quasi-uniform clouds, the dissipative form, or the oversampled least-squares
route otherwise.

**Boundary ghost layers (PDE+BC route).** Square collocation that replaces the
PDE at a Neumann or Robin point by its flux condition alone leaves one-sided
boundary stencils whose difference modes are spurious eigenvalues (measured on
a 1067-point quasi-uniform disk, cubic PHS / 20 neighbors: four modes with
`Re < 0`, minimum -0.29), so the default stability assessment refuses it. The
explicitly selected remedy keeps two equations per flux boundary point:
`PointGhostLayerPlan(boundary, offset=1.0, minimum_separation=0.5)` reads the
Neumann and Robin rows and their outward normals from a `PointBoundaryPlan`
and, prepared on the cloud, places one ghost point `x_g = x_b + δ_b n_b`
outside the domain per such row (`δ_b = offset · h_b`, `h_b` the distance from
`x_b` to its nearest cloud point) with one ghost unknown.
`PointCloudPoissonPlan(cloud, boundary, ghosts=layer)` (route
`"ghost-collocation"`) then collocates the PDE at every non-Dirichlet cloud
point with ghost-extended stencils and places the boundary condition at `x_b`,
divided by `δ_b` so it carries PDE units, on the ghost's row. The unknowns are
the cloud values followed by the ghosts; `physical_rhs` holds the source on
cloud rows (so the source is needed at flux boundary points too) and `g_b / δ_b`
on ghost rows; `result.values` holds the cloud values and `result.ghost_values`
the ghosts. Variable diffusivity is cloud data: its gradient uses the cloud's own
stencils. The extension law is declared: `u_g` is the value at `x_g` of the
continuation of `u` satisfying both the PDE and the condition at `x_b`, fixed by
the coupled equations rather than by extrapolation. `result.ghost_extension_defect`
reports `max_g |u_g - (E u)(x_g)|` against the one-sided reconstruction `E` from
cloud points only; it decreases with refinement for smooth solutions (2-D disk
Neumann: 3.8e-6, 3.1e-7, 7.7e-8). `PreparedPointGhostLayer.evidence` reports
offsets, a certified lower bound on each ghost's separation from every other
point (refused below `minimum_separation`, which also refuses inward normals
because their ghosts land among the cloud's samples), and the ghost-block
diagonal dominance of the normal derivative (evidence only; measured 0.2–0.8,
so admission comes from the plan's spectral assessment). Rows with a positive
outward weight on their own ghost are required. Periodic rows, side support,
the dissipative form, and the oversampled route are refused with the ghost
layer. The default meshfree hierarchy coarsens ghost rows like bulk rows and
eliminates only Dirichlet and gauge rows: retaining the ghosts kept 3-D coarse
levels at the ghost count (4735 unknowns: 837 s setup) and gave no fewer
iterations than coarsening them (16 s). Dividing the boundary rows by `δ_b`
matters for multigrid: unscaled rows needed 36/58 iterations at 1k/4k points,
scaled rows 12/13.

Measured with the default policies (`u = exp(sum(x)/d)`, `K = 2 + 0.1 sum(x)`,
quasi-uniform jittered disk/ball clouds, cubic PHS with 20/40 neighbors,
Neumann problems with `compatibility="project"`, mixed = Dirichlet on `x0 < 0`,
Neumann on `x0 >= 0`; GMRES with ILU below 2048 unknowns and the meshfree
V-cycle above; the 16k-point runs declared a larger `stability_assessment`, and
the 3-D 16k runs the same V-cycle with 4 GiB solve resources and
`stability="diagnostic"`; every completed assessment admitted its operator):

| Case | Cloud points (ghosts) | Iterations | Max error | Observed order |
| --- | --- | --- | --- | --- |
| 2-D Neumann | 1067 (106) / 4091 (209) / 16190 (417) | 39 ILU / 15 / 16 | 2.5e-5 / 7.3e-6 / 1.8e-6 | 1.8, 2.0 |
| 2-D mixed | 1067 (53) / 4091 (105) / 16190 (209) | 27 ILU / 10 / 11 | 1.5e-4 / 4.1e-5 / 1.0e-5 | 1.9, 2.0 |
| 3-D Neumann | 1013 (410) / 3696 (1039) / 14715 (2718) | 30 ILU / 13 / 14 | 7.5e-5 / 3.0e-5 / 1.2e-5 | 1.9, 2.0 |
| 3-D mixed | 1013 (205) / 3696 (520) / 14715 (1359) | 19 ILU / 10 / 11 | 3.3e-4 / 1.5e-4 / 5.9e-5 | 1.7, 2.0 |

Independent dense spectra at about 1000 points (numpy `eig` of the assembled
cloud-plus-ghost operator) have exactly one eigenvalue at zero for pure Neumann
(the gauge mode) and none with `Re <= 0` otherwise; eliminating the ghosts
through the boundary rows leaves the heat-equation operator, whose lowest
nonzero eigenvalue matches the continuum Neumann eigenvalue `K λ1` (unit disk
`1.8412²`, unit ball `2.0816²` within 2%). At 16384 points in 3-D the default
assessment's sparse LU exceeds its declared symbolic-work budget
(`"stability-unassessed"`); such clouds declare a larger `stability_assessment`
or `stability="diagnostic"`, and their solve needs a declared `LinearSolvePolicy`
with resources beyond the default 512 MiB workspace.

**Batched Schwarz patches.** `PreparedMeshfreeSchwarz.block_term()` returns one
additive term whose local space stacks every patch into equal blocks (largest
patch rounded up to a multiple of 32, absent coordinates declared padding of
`BlockJacobiPreconditionerBuilder(..., padding=...)`). It equals the additive
composition of `terms()` with exact local solves while factoring all patches in
one batched local-block factorization instead of one compiled sparse
factorization per patch. Multiplicative sweeps keep the ordered `terms()`.

Dirichlet data are lifted before solving the column-eliminated equation. Every
connected component of the unknown graph without a Dirichlet or positive-Robin row
floats and receives one explicit interior gauge (`plan.gauges`, or user `gauges=`
with exactly one row per floating component). Compatibility is checked per
component against all original equations, including the equation replaced by its
gauge; a quadrature sum alone is not assumed to be the algebraic left-nullspace
condition. `compatibility="refuse"` rejects incompatible data.
`compatibility="project"` computes one constant bulk source correction per floating
component with a native response solve and reports the correction, the per-component
residuals, and the response solve diagnostics.

Discontinuous materials use side-labeled support. `PointSideSupportPlan(cloud,
membership, sides=...)` declares which points belong to each material side;
`prepare()` fits one stencil family per side from that side's actual points only
(no ghost points) and reports `PointSideAdmissionEvidence`, including the rows
evaluated one-sidedly. With `sides=`, the diffusivity is one field per side, and
each shared row must be owned by a `PointInterfaceCondition` (single-valued
unknown, flux transmission `k_- grad_- u.n - k_+ grad_+ u.n = flux_jump` with the
normal oriented from `minus` into `plus`; sides may come from a geometry
`SurfaceInterface`) or by a boundary condition naming its evaluation `side`.

Physical boundary samples come from geometry authority:
`sample_boundary_atlas(atlas, FacetTraceRule(...), shape)` maps a facet reference rule
through oriented `BoundaryAtlas` charts (points, oriented unit normals, measure =
reference weight times chart Jacobian) and refuses irregular or trimmed-out samples;
`sample_cubature_atlas` does the same for a `CubatureAtlas` that supplies normals and
refuses one that does not.

The oversampled route takes independent targets:
`PointCollocationPlan(targets, row_measure, boundary_mask=..., boundary_weight=...)`
prepared on the cloud gives value and derivative stencils from cloud sources to
target rows; `PointCloudPoissonPlan(..., collocation=...)` then minimizes
`sum_t w_t r_t^2` through a native weighted `LeastSquaresProblem` (default
`GeneralizedLSMR`). Bulk rows use their volume measure; boundary rows use their
condition's physical measure times the declared `boundary_weight`. This is a
different discrete objective from square collocation. Nonpositive measures, a weight
range beyond `maximum_weight_ratio`, fewer targets than unknowns, floating
components, and side support are refused; least-squares status and condition evidence
come from the native solver.

`point_sbp_report` checks the complete sparse coefficient identity, including the
boundary flux term. It has an explicit preparation budget. A finite polynomial
probe is not labeled a full SBP certificate. `prepare_point_sbp_derivatives(cloud,
reproduction_degree=q)` is the explicitly selected constrained alternative: per
axis it solves `min ||D - D_0||^2` over the symmetrized stencil pattern subject to
polynomial reproduction and every coefficient of `M D + D^T M = B n`, as a native
sparse conic program. A diagonal-norm identity requires the volume cubature to
integrate degree `2q - 1` exactly; otherwise the conic owner reports primal
infeasibility with its ray certificate, and the result is not relabeled as SBP.

An admitted constrained preparation is consumed directly, not merely reported
beside a different GMLS derivative; its solve remains algebraic evidence:

```python
prepared_sbp = prepare_point_sbp_derivatives(
    cloud, reproduction_degree=q
)
plan = PointCloudPoissonPlan(
    cloud, boundary, form="dissipative", sbp=prepared_sbp
)
result = plan.prepare(diffusivity).solve(source)
```

The cloud must have the declared positive volume/boundary cubature and support
needed for feasibility. Binding checks source revision, the physical measures
and normals, coefficient identity, constant reproduction, and admission.
Changing a same-shaped cloud does not make the prepared certificate reusable.
Arbitrary local GMLS admission on a jittered cloud is not an SBP certificate.

`prepare_tensor_point_sbp(cloud, derivatives)` bridges actual prepared native
`SBPDerivativePlan` operators to `PointSBPDerivatives`. Declare one bounded
point-primary `TensorGridPlan` and one prepared derivative per grid axis in
grid-axis order. The bridge checks actual shared grid ownership, coordinate
columns, explicit `cloud.plan.point_ids` as the C-order grid row map, positive
`SBPGridNorm` volume weights, and signed physical tensor-face cubature.
It does not infer tensor geometry from a cloud's shape or display name.
The retained native families and binding authorize `stable_realization`, a host
preparation diagnostic rather than a jitted scalar flag. Generic constrained
SBP results report it as false even when their SBP identities pass. It is not
an error or refinement-order oracle. `PointCloudPoissonResult.algebraically_successful`
preserves raw solve/residual acceptance separately from scientific `successful`.

For example, the bounded second-order route starts from native owners:

```python
grid = TensorGridPlan(
    (UniformAxisSpec(n),), axis_names=("x0",)
).prepare(jnp.asarray([[-1.0], [1.0]], dtype=jnp.float64))
derivatives = (SBPDerivativePlan(grid, "x0", interior_order=2).prepare(),)
norm = SBPGridNorm(derivatives)
boundary_mask = jnp.isclose(jnp.abs(grid.points[:, 0]), 1.0)
cloud = PointCloudPlan(
    grid.points,
    norm.weights.reshape(-1),
    point_ids=np.arange(n, dtype=np.int64),
    boundary_mask=boundary_mask,
    boundary_normals=jnp.where(boundary_mask[:, None], grid.points, 0.0),
    boundary_quadrature_weights=boundary_mask.astype(jnp.float64),
    stencil=LocalStencilPolicy(polynomial_degree=2),
).prepare()
prepared_sbp = prepare_tensor_point_sbp(cloud, derivatives)
```



## Coupled block systems

`PointBlockSystemPlan(cloud, boundary, components=(...))` assembles
`-sum_b div(C_ab grad u_b) = f_a` with a fourth-order coefficient `C[a, b, i, j]`
(constant or per point) through native `BlockSpace`/`BlockLinearOperator` sparse
assembly. Each component owns its boundary rows; the conormal row of component `a`
is `sum_b n_i C_abij d_j u_b` (elastic traction for
`isotropic_elasticity_coefficients`). `C` must be finite, major-symmetric, and
positive semidefinite in its flattened `(a i), (b j)` form (`PointCouplingEvidence`);
strong ellipticity is decided by the solve residual. Components that float on any
connected component are refused rather than gauged implicitly.

Preparation runs the native square-collocation spectral assessment on the solved
operator (`stability`, default `"require-assessment"`; `"diagnostic"` records
`PreparedPointBlockSystem.stability` without refusing). Square collocation that
replaces the PDE by a traction row leaves one-sided stencils with spurious
modes: a jittered 9×9 square with one traction face has 4 of its 16 nearest
eigenvalues with `Re < 0`, so it is refused. `ghosts=PointGhostLayerPlan(boundary).prepare(cloud)`
selects the ghost route: every row that any component owns with a Neumann or
Robin condition gets one shared ghost point and one ghost unknown per
component. The solved unknowns are cloud values then ghost values per component
(`equation_space`, `solve_space`); every non-Dirichlet cloud point (traction
points included) carries its PDE with ghost-extended stencils, and ghost row `g`
carries component `a`'s own condition at its boundary point divided by the
ghost offset `δ_b` (its PDE at `x_b` where `a` is Dirichlet, as on rollers).
Each component's condition uses its own declared normal. Where the flux-owning
components declare different normals (a corner between two traction faces), the
shared ghost lies on their outward bisector, and normals that cancel are refused.
`PointGhostLayerEvidence.direction` and `component_normals` record both.
The same 9×9 patch is then admitted and solves the affine field to `1e-8`.
`equation_rhs` gives the solved right-hand side; `PointBlockSystemResult`
publishes `ghost_values` and `ghost_extension_defect`. `apply`, `physical_rhs`,
and `physical_operator` keep the square physical rows on cloud values (traction
rows report `sigma n - t`). The ghost route serves linear collocated block
equations; `solve_nonlinear` and periodic rows are refused.

Without an explicit `linear_policy`, the solve is GMRES (restart up to 200,
Krylov basis budget declared from that capacity) right-preconditioned by a
numerically refreshed ILU of the assembled block operator.
`auxiliary=MeshfreeNearNullspace(...)` selects the auxiliary-space route:
`meshfree_auxiliary_stiffness` assembles the SPD linear-simplex stiffness of the
same coefficient on a Delaunay triangulation of the same points (canonical
`DelaunayTriangulation` with `provider="qhull"`, screened by
`SimplexQualitySubcomplex(minimum_quality=0.1)`: 3-D slivers, whose P1 energy
grossly overestimates the continuum energy, are excluded unless needed for
coverage or facet connectivity, and `MeshfreeAuxiliaryStiffness.quality` reports
the excluded count and measure fraction; native per-block P1 cell assembly), and
`meshfree_auxiliary_builder` composes, as a native
`MultiplicativeSubspaceCorrectionBuilder`, a damped-Jacobi meshfree
hierarchy correction of that stiffness on lumped-mass-scaled interior rows with an exact sparse
elimination of the flux/traction rows (on the ghost route: the points include
the ghosts, and the ghost condition rows are the eliminated trace rows; ghost
coordinates are never eliminated as Dirichlet). The auxiliary stiffness carries the
coefficients, so it is prepared with them and rebuilt on coefficient refresh.
The route needs Dirichlet rows in every component, no periodic rows, and 2-D or
3-D points. `multigrid=MeshfreeNearNullspace(...)` instead builds block
meshfree multigrid of the collocated operator itself on the component-major
(`"block"`) coordinates. One `MeshfreeHierarchyPlan` coarsens the cloud with
every Dirichlet coordinate eliminated (a decoupled identity row: empty
prolongation row, never coarse, never in a Galerkin coarse operator) and uses
the declared transfer candidates. No stationary smoother contracts collocated
block operators, so its iteration count grows with resolution.
`MeshfreeNearNullspace("rigid-body")` serves elasticity and is refused unless
the component count equals the spatial dimension. `coarsening`
(`MeshfreeCoarseningPolicy`) requires a `multigrid` declaration. These
configure only the default policy; combining them with an explicit native
`linear_policy` is refused, because that policy owns its own preconditioning.
Linear status, residual, and iterations are published on
`PointBlockSystemResult`. Mixed owners compose the verified blocks directly.
`physical_operator(coefficients)` returns the physical `BlockLinearOperator`
with its `PointCouplingEvidence`. `flat_operator(assembled)` re-expresses an
assembled block operator on the component-major `solve_space`. It changes the
coordinate space only, never the entries.

`solve_nonlinear(reaction, source)` adds a pointwise reaction on bulk rows and uses
native `NewtonKrylov` with matrix-free prepared Jacobian actions. Cross blocks are
genuine couplings: applying the scalar operator channel by channel is a different
equation.

```sh
JAX_ENABLE_X64=1 python examples/point_cloud_poisson.py --size 64 --dimension 2 --boundary-kind mixed
```

The example differentiates an independent manufactured variable-coefficient
solution to obtain its source; it does not manufacture a source by applying the
operator being tested.
The dissipative selection explicitly uses `UniformAxisSpec` point-primary
native tensor axes, `SBPDerivativePlan(interior_order=2)`, and `SBPGridNorm`.
Samples come from the owner's `grid.points`; volume weights and tangential
boundary cubature come from its norm. The report distinguishes that derivative
realization from auxiliary local cloud stencils and identifies grid, norm,
families, axes, and interior order. This is a structured native tensor SBP
example, not an arbitrary-cloud meshfree demonstration. It retains measured
solution/boundary errors, original-load residuals, source corrections, and
native solver status. The former constrained degree-one GMLS Robin refinement
was unstable despite passing identities and tiny algebraic residuals;
historical artifacts remain failed evidence under their original support IDs.
Q2's replacement method is `native-tensor-sbp-interior-order-2-dissipative`,
with geometry authority `native-point-primary-tensor-grid-sbp-norm`; this new
support identity does not recategorize or rehabilitate the old artifacts.


## Meshfree solid mechanics {#meshfree-solid-mechanics}

Three mechanics owners compose the block system on one prepared cloud. Boundary
rows belong to individual components, so rollers, clamps, and traction faces
can be mixed freely. Corners need explicit ownership.

- `MeshfreeElasticityPlan(cloud, boundary, tensor)` solves small-strain
  `-div(C : eps(u)) = f` for one verified homogeneous `LinearElasticityTensor`
  of the cloud dimension. Neumann rows are tractions `sigma n = t`, Robin rows
  add `k u`, and Dirichlet rows prescribe displacement. `preconditioner`
  selects `"auxiliary"` (default in 2-D and 3-D), `"multigrid"` (default in
  1-D), or `"ilu"`; `linear_policy` and `coarsening` are passed through.
  Measured GMRES iterations (jittered unit square, PHS degree 3, `lambda = 1`,
  `mu = 0.5`) are tabulated in the multilevel section below; outside a route's
  envelope the solve is refused through the block status, never accepted.
  `traction_route` (`MeshfreeTractionRoute`) selects `"ghost"` (default: when
  traction or Robin rows exist the block system receives
  `PointGhostLayerPlan(boundary).prepare(cloud)`) or `"square"` (traction rows
  replace the PDE; the default spectral assessment refuses its spurious modes,
  `stability="diagnostic"` records them).
  `PreparedMeshfreeElasticity` exposes `strain`, `stress`, `traction`, the
  physical `residual` rows, and `strain_energy`.
  `displacement_gradient(cloud, u)` returns `d_j u_a` from the cloud's
  admitted first-derivative rows.
- `MeshfreeHyperelasticPlan(cloud, boundary, law, load_steps=...,
  termination=..., preconditioner=...)` solves finite-strain `-Div P(F) = f`
  with `F = I + grad u` and a `NeoHookeanLaw`. Two-dimensional clouds are
  plane strain through `diag(F, 1)`. Neumann rows are reference tractions
  `P N = t`, and periodic rows are refused. Body force, traction, and
  displacement data are scaled by the load factors `k / load_steps`. Each
  increment runs native `NewtonKrylov` with a `J > 0` trial-validity guard and
  GMRES solves to a constant `1e-6` forcing, right-preconditioned by the block
  owner's prepared preconditioner of the reference tangent, frozen along the
  load path. At `F = I` that tangent is the Hooke tensor of the same Lamé
  parameters, so `preconditioner` takes the `MeshfreeElasticityPlan` routes
  (`"auxiliary"` by default, `"multigrid"`, `"ilu"`); Newton then converges
  quadratically in two or three steps per increment. `traction_route` is the
  elasticity choice. On the default ghost route the equation unknowns
  (`unknown_count` per component) are cloud displacements then ghost
  displacements. Cloud rows carry `-A(F) : D^2 u - f` with `A = dP/dF`, the
  non-conservative form of `-Div P` for a homogeneous law, using ghost-extended
  stencils. Ghost rows carry `(P N + k u - t) / δ_b`. The linearization at
  `u = 0` is the elasticity ghost system, so the frozen preconditioner matches.
  `extend(u)` fills ghosts from the one-sided extension law; the result publishes
  `ghost_displacement` and `ghost_extension_defect`. `residual` and `linearize`
  (a `PreparedLinearization` with JVP and VJP) take the `(unknown_count, d)`
  equation unknowns, and `problem()` (the native `NonlinearSystemProblem`) is
  public.
- `MeshfreeGeneralizedStokesPlan(cloud, boundary, shear_modulus=mu,
  compressibility=kappa, stabilization=s)` is the generalized Stokes /
  Herrmann mixed form `-div(2 mu eps(u)) + grad p = f`, `div u + kappa p = 0`.
  With `kappa = 0` it is Stokes flow; with `kappa = 1 / lambda` it is
  near-incompressible elasticity, and the displacement is not volumetrically
  locked as `lambda` grows. Equal-order collocation is not inf-sup stable.
  The pressure rows therefore carry two consistent residual terms with
  `tau = s h^2 / mu` at bulk and Dirichlet nodes and `0` at traction nodes,
  whose pressure the `-p n` rows already carry: the weak-form term
  `V^-1 G^T V tau (-div(2 mu eps(u)) + grad p - f)` and the compact collocated
  term `-tau ((1 + 2 mu kappa) lap_h p - div f)`. Both vanish on the exact
  solution, and a nonpositive `stabilization` is refused. The compact term
  damps high-frequency pressure, which the wide weak-form operator barely
  sees. The weak-form term is volume-self-adjoint positive semidefinite and
  controls boundary and corner rows, where `lap_h` has positive diagonals and
  the compact term alone made the operator singular near `kappa ~ 1e-2`.
  Without traction rows the solve pins one bulk pressure row
  and borders a uniform continuity source, which stays well posed for every
  `kappa >= 0`. For `kappa > 0` the mean pressure is closed by the discrete
  volume balance `kappa sum V p = -oint u.n` over the declared boundary
  quadrature (a clamped `kappa > 0` plan without
  `boundary_quadrature_weights` is refused at construction), and what the
  collocated rows leave is the reported `compatibility_residual`. For
  `kappa = 0` the bordered source is that residual and the pressure is
  reported with zero mean. The saddle solve is native GMRES with an upper
  block-factorization preconditioner: ILU on the viscous block and ILU on the
  Schur model `(1/mu + kappa) I` plus the stabilization's pressure block.

Each result carries physical evidence:

- `MeshfreeElasticityResult` publishes strain, stress, boundary traction,
  strain energy `1/2 sum w sigma:eps`, external work `sum w f.u + oint t.u`,
  `clapeyron_defect = |2U - W| / |W|`, the body and boundary force resultants,
  `force_balance_defect`, and the block result. Boundary integrals use the
  cloud's `boundary_quadrature_weights` and are NaN without them.
- `MeshfreeHyperelasticResult` publishes the committed and candidate
  displacement, `F`, `P`, `J`, the stored energy, the external work along the
  load path (trapezoid in load), `energy_work_defect`, the residual norm, the
  accepted load factor, per-increment Newton status and iterations, and the
  final `NonlinearResult`.
- `MeshfreeMixedResult` publishes field, pressure, strain, stress, traction,
  divergence, `volumetric_defect = ||div u + kappa p||`, the momentum and
  pressure residual norms, the gauge `compatibility_residual`, and the native
  linear result. Its `pressure_oscillation` is `O(h^2)` for a smooth pressure
  and `O(1)` for checkerboard modes.

`MechanicsStatus` reports `NONFINITE`, `INADMISSIBLE_DEFORMATION`,
`SOLVE_REFUSED`, and `RESIDUAL_TOO_LARGE` in that precedence. A non-accepted
elasticity result carries NaN `displacement` and the attempted
`candidate_displacement`. A refused hyperelastic increment rolls back: the
committed `displacement` stays at `accepted_load_factor`,
`candidate_displacement` is the last Newton attempt, and no later increment is
committed. Construction refuses a tensor of the wrong dimension or one that
fails verification, floating bodies (a component without anchoring rows),
periodic finite-strain or mixed rows, and nonpositive moduli.

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_elasticity
```

The example reports manufactured small-strain refinement with a traction face,
zero strain under rigid motion, and closed-form plate tension: displacement,
`U = t eps_xx / 2`, `W = 2U`, and force balance. It also reports the
homogeneous neo-Hookean stretch with stored energy and load-path work, the
linear limit under a small load, a Herrmann solid as `kappa -> 0`, and a
Newton refusal with rollback.

## Stabilization

`HyperviscosityPlan(coefficient, order=...)` prepares
`-coefficient * (-L)^order` using repeated sparse dissipative Laplacian actions.
No matrix power is materialized. The power-iteration spectral quantities in
`HyperviscosityEvidence` (raw Rayleigh quotient, `dissipative` flag,
`explicit_step(interval)`) are estimates, not bounds or unconditional time-step
stability certificates. The term enters a semidiscrete residual only where it
is declared (`MeshfreeEvolutionPlan(hyperviscosity=...)`, implicit part); no
failed run is silently stabilized or repaired.

## Multilevel preparation

`MeshfreeHierarchyPlan` selects each coarse level as a maximal independent set
of the symmetric k-nearest-neighbor conflict graph. Priorities are hashed from
stable IDs, so the levels do not depend on input ordering, and the selection
runs in bounded device rounds (`maximum_coarsening_rounds`). `features` nodes
are always retained and `boundary` nodes only on the finest
`boundary_retention_levels` transitions (default 0; `None` retains them on every
level) with exact nodal inclusion; the remaining fine nodes use
polyharmonic-spline interpolation reproducing every polynomial of
`reproduction_degree`. `eliminated` marks fine coordinates whose equations are
decoupled identity rows (lifted Dirichlet values, gauges): their prolongation
rows are empty, so those equations never enter a Galerkin coarse operator.
Retaining such rows kept a 2-D elasticity hierarchy at 289→131→85→70→65→64
points with nonpositive coarse diagonals; eliminating them gives 289→93→31→10.
Coarsening stops with an explicit `MeshfreeHierarchyEvidence.stopping_reason`
(`"no-progress-retention"`, `"insufficient-reduction"` above
`maximum_coarse_fraction`, `"coarsening-round-limit"`, `"unisolvence-minimum"`,
`"transfer-row-refusal"` when a nearest coarse support does not admit the
declared polynomial space, `"empty-coarse-coordinate"`, `"minimum-coarse-points"`,
`"maximum-levels"`); not every boundary-heavy cloud coarsens and refused
supports are never widened or regularized. Evidence reports per-degree
reproduction defects, near-nullspace defects, coarsening rounds, route storage
and grid complexity. Restriction is the coordinate transpose required by the
stiffness-coordinate Galerkin equation, not an implicitly substituted mass
adjoint.

`MeshfreeComponentSpace(components, layout="nodal" | "block")` prepares coupled
vector systems: every component shares the scalar point transfer (`P ⊗ I` or
`I ⊗ P`), so an affine-reproducing transfer reproduces all rigid-body modes.
`MeshfreeNearNullspace("constant" | "rigid-body" | "affine", vectors=...)`
declares the candidates whose transfer defects are reported, including supplied
fields such as indicators of disconnected components.

`meshfree_multigrid_builder` supplies these transfers to the native
`GalerkinHierarchyBuilder` with the selected V/W/F/full `cycle`. Numeric
refresh reuses transfers and symbolic sparse products; a new hierarchy is a
declared transfer-dependency change and rebuilds the coarse operators. Forward
sparse Gauss–Seidel in stable-ID order is the default fine smoother (and the
scalar coarse smoother); coupled component hierarchies default to ILU(0) on the
Galerkin coarse levels, where collocated elasticity measured coarse
Gauss–Seidel radius 23 against ILU(0) 0.2 (34 versus 59 GMRES iterations at
N=289). Point-block Jacobi, symmetric or ordered Gauss–Seidel, ILU/ILUT and
Chebyshev builders may be supplied and enforce their own property requirements.
No stationary smoother is claimed to contract collocated operators (forward
Gauss–Seidel radius 3.4 on that fine level). Nonsymmetric collocation
can be strongly nonnormal: no universal cycle contraction or superiority over
ILU is claimed; Galerkin operator complexity is reported by the native setup
diagnostics. A singular coarse problem requires an explicitly declared native
nullspace whose kernels every transfer reproduces exactly (constants,
rigid-body modes, component indicators) and a bounded projected pseudoinverse
coarse solve; kernels that are not reproduced are refused.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_multilevel_poisson.py --size 64
```

## Migration

| Removed API | Canonical replacement |
| --- | --- |
| `MeshfreeStencilPlan`, `prepare_meshfree_operator` | `MeshfreeNeighborhoodPlan`, `prepare_local_stencils`, `MeshfreeOperator` |
| `MeshfreeMethod`, `MeshfreeOperatorKind` | `MeshfreeApproximation`, `MeshfreeFunctional` |
| `PreparedMeshfreeOperator`, `MeshfreeReproductionEvidence` | `PreparedLocalStencils`, `LocalStencilEvidence`, `LocalStencilReport` |
| `PointStencilReport` | `LocalStencilReport` |
| `PointCloudPlan(degree=..., neighbor_count=..., condition_limit=...)` | `PointCloudPlan(stencil=LocalStencilPolicy(polynomial_degree=..., condition_limit=...), neighbors=...)` |
| `DissipativePointDiffusion` | `PointDiffusionOperator(form="dissipative")` |
| `solve_point_cloud_poisson` | `PointCloudPoissonPlan(...).prepare(...).solve(...)` |
| `PointConormalInterface`, `DistributedPointPartition` | Removed unused metadata; no distributed or collocated material-interface execution claim |

There are no deprecated aliases or compatibility wrappers. See the
[meshfree API](api/discretization/meshfree.md) for exact signatures.

## Conservative graph metrics

`MeshfreeExteriorCalculusPlan` prepares positive node measures, canonical radius
edges, equation masks, and the declared polynomial moment constraints. Supplied
quadrature is preferred. Density-derived volumes require an explicit total domain
measure; Dirichlet rows are excluded from equations, never assigned zero mass.

`MeshfreeMetricPolicy` chooses signed or nonnegative coefficients, exact or
explicitly relaxed moments, and `moment_degree`: every moment
`1 <= |alpha| <= degree` of the Laplacian functional, in a deterministic
degree-graded multi-index order with row scaling `h^|alpha|`. Higher moments may
be infeasible for any edge metric; they are refused with evidence or relaxed only
on request, never truncated.

The exact signed metric minimizes `0.5 w^T Phi^-1 w` subject to **every**
original moment row. With `S = sqrt(Phi)` it is the native minimum-norm solve
of `B z = D^-1 b`, `B = D^-1 A S`, `w = S z` (`MinimumNormProblem`). It is
audited by the true constraint residual and a minimum-norm stationarity witness.
`MeshfreeMetricPolicy.solver` selects the route; both have the same objective
and solution.

- `"multilevel-craig"` (default) runs preconditioned Craig on `B B^T` and
  returns `z = B^T y`.
- `"lsmr"` runs `GeneralizedLSMR` on row-equilibrated `B`.

The default exists because `B B^T` is a sixth-order operator. Its smallest
eigenvalues belong to the line-integral jets `(g, sym grad g)` of smooth vector
fields `g`; at moment degree two their edge residual is a fourth-order Taylor
remainder. The condition number therefore grows like `h^-6`, and LSMR
iterations grow like `h^-2.5`. On a 0.15h-jittered unit square they were 429,
2373 and 16636 at sides 9, 17 and 33. Row or column scaling cannot change that
order.

The Craig preconditioner is a frozen Galerkin V-cycle (`GalerkinHierarchyBuilder`)
with symmetric Gauss-Seidel smoothing. Its transfers are jets of
tensor-product B-spline vector fields of degree `moment_degree + 1`, on
dyadically nested grids over the moment-node box, with periodic splines along
periodic axes. On the same clouds Craig needed 55, 66, 73 and 78 iterations at
sides 9, 17, 33 and 65, and agreed with the dense minimum-norm oracle to `3e-10`.

The preconditioner only changes the iteration count. It is prepared once
(`PreparedMeshfreeMetric.preconditioner_levels` records its level sizes), and
refreshed coefficients and priors reuse it. The hierarchy needs moment
displacements that are coordinate displacements: default or minimum-image charts
(`MeshfreeMetricGeometry`). Intrinsic surface charts declare `solver="lsmr"`.
`multilevel_resources` and `multilevel_assembly` bound the host-assembled
`B B^T` and its Galerkin products; anything larger is refused at preparation.
Craig reaches `B` only through `B B^T`, so its stationarity floor is
`sqrt(eps)`-class. A consistent system whose preconditioned `B` has condition
number beyond about `1e7` cannot be told apart from an incompatible one, and
Craig's witness leaks about `1e-9` relative. Declare `solver="lsmr"` when an
eps-grade witness is required.

An incompatible right-hand side is `INCOMPATIBLE_MOMENTS` with a validated
left-null witness `incompatibility_witness` (`A^T y = 0`, `<y, b> != 0`).
Nonconvergence is a provider failure, never an infeasibility claim. The relaxed
signed objective `0.5 |z|^2 + C/2 |D^-1 (A S z - b)|^2`, including the original
row scaling, is the stacked least-squares problem `[sqrt(C) D^-1 A S; I]` and
always uses `"lsmr"`. No Schur factor, row selection or explicit inverse is
formed.

Rank evidence is a native SVD rank certificate under `MeshfreeMetricPolicy.rank`
(a `linalg.svd.SVDSolvePolicy`, dense by default); beyond its budgets `rank_certificate.route` is
`"unavailable"`, `rank` is `-1`, and exact signed derivatives are refused (NaN)
while the forward metric may still be accepted. Exact signed derivatives need a
matching fixed-rank certificate of maximal rank. `row_rank_profile=True` records
the native sparse pivot profile as a diagnostic only; it is never a rank
certificate. Exact nonnegative metrics pose only the profile's independent rows
as conic equalities (an interior-point presolve) and still audit every original
row afterwards.

`MeshfreeMetricResult` retains provider status and the native solve or conic
execution, original moment residuals by degree, amplification, sign counts,
compatibility, and derivative availability. Relaxed metrics retain their slack;
conservation does not erase the consistency lost. Nonnegative infeasibility is a
failure, not a reason to clip weights. `PreparedMeshfreeMetric.bind_active_set`
eagerly binds the native fixed-active-set projection-KKT sensitivity
(constraint roles, strict-complementarity margin, original-coordinate KKT
residuals, Weyl-verified KKT nonsingularity) to a nonnegative result;
`strict_active_jvp` then publishes weight tangents for prior, RHS and moment
coefficient perturbations, NaN when the active set is ambiguous, changed or the
KKT system is not certified.

Strictly positive admitted weights bind native `CochainDiscretization`, Hilbert
spaces, and `CochainComplexIR`. Signed weights remain constitutive stiffness
coefficients over canonical incidence; they are not negative Hodge weights.
Zero or signed metrics cannot be converted to a positive Hilbert complex.
Reduced coercivity is assessed under `MeshfreeCoercivityPolicy`: a no-shift
sparse Cholesky with a fill-reducing ordering (approximate minimum degree by
default, nested dissection on request) charged to the policy's factor and
symbolic-work limits, or `"unassessed"`, which reports SPD and inertia as
unavailable. Unknown inertia is never inferred from finite coefficients.

`PreparedMeshfreeExteriorCalculus.refresh` raises when coordinates leave the
certified fixed-radius topology; `try_refresh` is the status-returning
transaction (`MeshfreeExteriorRefreshResult`, `MeshfreeExteriorRefreshStatus`):
geometry is admitted before any metric solve, the raw candidate metric is kept
as evidence, and a refused or unaccepted candidate leaves the owner unchanged.
`displacement(points)` and `within_topology_trust(points)` expose the trust test.

Bounded clouds declare `boundary_area_vectors` on `MeshfreeExteriorCalculusPlan`:
per node `s_i = Σ (facet area × outward normal)` over the boundary facets of
its control volume, zero inside (from `PointBoundaryPlan` data:
`measure[:, None] * normal`). Nodes with `s_i ≠ 0` keep their tangential
(boundary–boundary) edges and carry closure rows selected by
`MeshfreeMetricPolicy.boundary_closure`. `"second-moment"` (default) poses
`Σ_e w_e d_e d_eᵀ = 2 V_i I`; the boundary normal measure is the deficit
`exterior.boundary_normal_measure = −Σ_e w_e d_e`, with `Σ S = 0` and
`Σ x Sᵀ = Σ V · I` exactly, so edge transport and divergence are exact for
affine fields. `"area-vector"` also poses `Σ_e w_e d_e = −s_i`. Both families
together are overdetermined (discrete Gauss identities such as
`Σ x_l x_m s_k = Σ V (x_l δ_km + x_m δ_kl)` must hold): lattices satisfy them,
irregular clouds generally do not, and such moments are refused with a witness
rather than relaxed. Area vectors are fixed by preparation; refresh keeps them.

Diffusion uses explicit harmonic, arithmetic, or supplied edge coefficients.
Natural boundary loads require declared boundary quadrature.
`MeshfreeAdvection.rate` publishes the instantaneous low-order content rate of a
prescribed integrated volume flux (`MeshfreeAdvectionRate`: content/value rate,
edge flux, outgoing flux, forward-Euler `stable_step`); it integrates nothing.
`edge_upwind_content` is the forward-Euler admission certificate of that rate,
kept for one-step remaps such as the surface shift. Both reuse the prepared
native sparse incidence for donor values, transpose accumulation, and outgoing
CFL. Fluxes are integrated volume fluxes: the metric is not multiplied a second
time. Positivity requires an admitted nonnegative metric, nonnegative
states/sources, and the outgoing CFL bound, not merely conservation. The public
`value` and `content` of the certificate are accepted only for `ACCEPTED`
status and otherwise hold the input; the raw unclipped update stays in
`candidate_value`/`candidate_content` with its ledger and CFL evidence.
Derivatives through a refused public state are invalid (NaN) rather than a
successful gradient of the held input.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_conservative_diffusion.py --size 25
```

## Bulk transport and semidiscrete evolution

`ConservativeTransport(exterior, scheme=..., reconstruction=cloud)` owns the
conservative edge transport on one prepared exterior graph. A nodal velocity
becomes the integrated edge volume flux `w_e (u_i+u_j)/2 · e_ij` (exact `V div
u` for affine velocities at equation nodes); boundary (non-equation) nodes are
closed by the graph's own first-moment deficit, so a uniform translation is
divergence free at every node and a uniform state is preserved exactly. An
already integrated `TransportVolumeFlux` (for example a projected solenoidal
flux) is used as given. `rate(values, advection, inflow=..., source=...)`
returns the content rate with low-order, antidiffusive and final edge fluxes,
boundary exchange, local bounds, limiter coefficients and a conservation
audit. Schemes:

- `"upwind"`: donor-cell flux.
- `"reconstructed"`: the Hermite-corrected midpoint flux
  `w_e [(q_i+q_j)/2 · e − ¼ e·(∇q_j − ∇q_i) e]` of `q = c u` (and of `u` for
  the volume flux) with GMLS gradients from a point cloud on exactly the graph
  nodes. Its cubic Taylor part vanishes, so with degree-two metric moments the
  nodal truncation is `O(h²)` regardless of the metric's third moments or
  sign; the plain midpoint average leaves `¼ ∇²q : Σ w e e e`, an `O(h)` error
  that stalls nonuniform flows on irregular clouds. A supplied
  `TransportVolumeFlux` has no nodal velocity and uses upwind-biased MUSCL
  face values instead. No positivity certificate.
- `"limited"`: the reconstructed antidiffusive flux paired-limited
  (Zalesak/Kuzmin) against neighbor bounds with nodal capacity
  `limiter_capacity * outflow`. Limiter coefficients are frozen in
  derivatives; `limiter_switched` reports the nonsmooth points.

`cfl(advection, dt)` is the forward-Euler certificate of the scheme
(`dt (outflow + capacity) / V <= 1`); an SSP method inherits it only with its
own SSP coefficient, and no other temporal method inherits it. A graph with
boundary nodes requires a declared inflow state and an exterior prepared with
`boundary_area_vectors` (outward facet area times normal per boundary node,
e.g. boundary measure times outward normal). Those give prescribed nodes
second-moment rows and keep their tangential edges, so the deficit
`S_i = sum_j w_ij (x_i - x_j)` closes the boundary flux consistently; an
exterior whose prescribed nodes carry no moment rows (an `O(1)` boundary rate
error) is refused. Weak (upwind) inflow nodes still lag the inflow state by
`O(h)`; passing `inflow_rate` (the inflow's time derivative) makes inflow nodes
strong prescribed rows whose supplied content is ledgered in
`boundary_content_rate`. `MeshfreeEvolutionPlan(inflow_treatment="strong")`
differentiates the declared inflow law in time for these rows; the default
`"weak"` suits characteristic (tangential) boundaries, where roundoff-signed
flux must not prescribe rows. `refresh(points)` re-solves
the metric on the frozen topology inside the radius-topology witness and
returns the unchanged owner with `TOPOLOGY_TRUST_EXCEEDED` beyond it;
`minimum_image_edge_charts` prepares periodic endpoint charts by minimum-image
arithmetic on the edge relation's derived address; chart orientation is the
graph edge's target minus source, not the seam's source/target roles.

`MeshfreeEvolutionPlan(spatial, velocity=..., diffusion=MeshfreeDiffusionLaw(...),
reaction=MeshfreeReactionLaw(...), hyperviscosity=..., inflow=..., boundary=...,
motion=MeshfreeMotion(...), positivity=...)` assembles the semidiscrete
advection–diffusion–reaction residual on a GMLS point cloud (collocation; state
is the concentration) or on a `ConservativeTransport` (state is the content
`V c`, conservative diffusion through the graph stiffness). Diffusivities are
scalar, nodal, or SPD tensor fields times an optional constitutive factor
`g(c)`; a non-positive factor makes the rate nonfinite rather than clipped.
The prepared evolution owns no time loop. It publishes `rate`,
`explicit_rate`/`implicit_rate`, `differential_problem` (SSP-RK, Rosenbrock,
adaptive Diffrax), `split_problem`, `dae_problem` (BDF residual
`y' - f(t, y)`), `ssprk_method` (`MeshfreeEvolutionSSPMethod`: accepted step
gated by `admission`, publishing `MeshfreeEvolutionStepEvidence` — the
candidate's admission and, on the graph route, the attempt's transport CFL — for
accepted and refused attempts alike), and
`imex_method` (`ConservationIMEXFixedStepMethod` over any additive IMEX tableau,
with native linear or Newton–Krylov stage solves and per-stage evidence).
`admission(state)` reports `MeshfreeEvolutionStatus` without repairing the
candidate; refused steps keep the raw candidate and hold the state.
`spectral_estimate` and `transport_cfl` publish the estimated implicit radius
and the graph CFL certificate.

Measures are named. Under `MeshfreeMotion("ale", mesh_velocity=...)` the state
is `[content, volumes, points]` with `dV/dt = V div_h w` and transport by the
material-minus-mesh velocity, so translation and affine dilation preserve a
uniform state and the volume law exactly. `MeshfreeMotion("material",
material=MaterialParticleMeasure(particles, population, density))` uses the
persistent particle mass as the conserved measure (identities must equal the
cloud's stable ids; nodes are never relabeled), with the ALE volume as the
density carrier and an optional Verlet cache gating fixed support. Every
rate evaluation refreshes the fixed GMLS support at the stage coordinates; a
refused refresh is fail closed and the step is refused with
`SUPPORT_EXCEEDED`. Motion stays in raw unwrapped coordinates; `rebase(state)`
re-anchors identical nodes as a new support epoch and only then wraps periodic
coordinates into the address cell. Population changes belong to the meshfree
epoch/transfer owners.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_bulk_advection_diffusion.py --size 12
```

## Higher exterior degrees with geometry authority {#higher-exterior-degrees}

The graph route above is honest only through degree one. Every degree k ≤ n is
realized over a supplied oriented simplicial complex with geometry authority:

1. `ComplexGeometryAuthority(mesh, identity, *, orientation=None)` takes a
   `CellMesh` (vertex embedding plus provenance) and a declared
   `ComplexDomainIdentity(domain_id, role, *, measure, betti)`. Roles are
   `"domain"` (n = D ≤ 3), `"closed_surface"` and `"open_surface"`
   (codimension one, n ≤ 2). Admission refuses non-simplicial, impure,
   degenerate, non-manifold, incoherently oriented or inverted complexes, and a
   complex whose measure or exact rational Betti numbers differ from the
   declared identity: a triangulation that fills a hole is refused, not
   relabelled. `orientation` reorients positive-degree cells through the native
   cochain owner.
2. `MeshfreeCellComplexPlan(authority, *, chart, policy)` prepares, per degree,
   facet-adjacency patches around every top simplex and a GMLS fit of full
   P_m Λ^k polynomial forms (in the simplex's orthonormal intrinsic frame) to
   the exact moments ∫_σ of the patch k-cells. Patches grow until the native
   weighted SVD reports full rank below `maximum_condition`; exhausted patches
   and exceeded `maximum_patch_cells`, `maximum_patch_entities` or
   `maximum_workset_entries` capacities are refused with
   `MeshfreeComplexAdmissionError`.
3. Each Hodge is a `SparseHodge`: the consistency term Σ_T ∫_T ⟨R c, R c⟩ plus a
   moment-residual stabilization scaled by |T|/|σ|². Polynomial cochains have
   zero residual, so the Hodge is exact for P_m forms on flat cells. Native
   sparse Cholesky must admit every degree; failure is refused, never repaired.

`PreparedMeshfreeCellComplex.cochain` is the canonical `CochainDiscretization`
(with boundary-facet closure masks for relative conditions) and `.bridge` the
exact `DeRhamBridge`. Consume them through the native interfaces:
`integrate_form`, `validate_de_rham_commutation`, `trace_map`,
`validate_harmonic_cohomology`, `HodgeLaplacePlan` and `UnstructuredMaxwellPlan`.
`sample(degree, values)` maps ambient k-form samples at the prepared top-simplex
nodes to GMLS k-cell moments; `reconstruct(form)` evaluates the local
reconstruction at those nodes. `evidence` records rank-deficient patches,
conditioning, polynomial-reproduction residuals and native Hodge admission with
`fidelity="geometry-authorized"`.

`radius_clique_complex(points, radius, *, policy=RadiusCliquePolicy(...))` is a
separate abstract research route: a bounded native radius relation, clique
enumeration that refuses before `maximum_simplices` or `maximum_work`,
ascending-vertex orientation, exact d∘d = 0 through `ExactChainComplex`, and
exact rational Betti numbers. Its record carries `fidelity="abstract-research"`;
it has no measure, Hodge or interpolation and is refused as geometry authority.
Matching Betti numbers never establish continuum fidelity.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_higher_forms.py
```

## Intrinsic surfaces

`SurfacePointCloudPlan` binds a declared smooth curve or sheet, positive
measures, bounded support, and tangent-chart GMLS/PHS approximation. The
embedding is declared, never inferred from array shapes: curves in R2/R3 and
sheets in R3. Three source kinds stay distinct (`evidence.source`):

- `ImplicitSurfaceGeometry` (`"implicit"`) uses the native geometric or
  regular-level-set source; its intrinsic dimension is ambient minus declared
  codimension (a space curve is a codimension-two level set). Normal projection
  belongs to `RegularLevelSetManifold.project_normal`.
- `ChartSurfaceGeometry` (`"chart"`) is authoritative: every sample carries its
  preimage in one or more `metrix.EmbeddedChart`s, and tangent bases, metric and
  second fundamental form come from the chart jets with native frame rank and
  conditioning evidence.
- `SampledSurfaceGeometry` (`"sample-estimate"`) fits degree-`fit_degree` (2–4)
  tangent height graphs with `oversampling` and reports independent fit,
  normal and curvature error estimates. Space curves are oriented by tangents.

`SurfaceGeometryEvaluation` publishes gauge frames plus gauge-invariant
`projectors` and the ambient `second_fundamental_form`; `gauge_margin` exposes
discrete frame switches. `mean_curvature_vector` is defined for every
codimension; `normals`, `curvature_tensor` and `mean_curvature` only for
codimension one. Sampled normals, projector comparisons, sheet-separation and
error estimates do not prove global reach, a closest point, or topology. An
implicit tube declaration is an explicit source-domain premise.

Closed sources refuse a boundary; open sources require a declared
`SurfaceBoundary` (boundary nodes, measures and outward weighted conormals,
e.g. `ChartSurfaceGeometry.box_boundary`). Boundary rows admit one-sided
physical supports; corners keep the conormals of both faces. The paired
`surface_divergence` then satisfies the discrete Green identity with the
boundary flux term exactly; `conormal_derivative` gives outward fluxes.
`SurfaceEllipticSystem` assembles collocated `-k Lap u + r u = f` with
Dirichlet or conormal-flux rows over one or more patches; a sharp crease is
two smooth patches joined by a `SurfacePatchInterface` (continuity plus flux
balance), never an averaged normal.

`surface_gradient`, `laplace_beltrami` and `surface_hessian` (covariant
Hessian) are strong intrinsic operators. Authoritative charts additionally
publish `chart_derivatives`, intrinsic derivatives in a declared chart's own
coordinates. Geometry accuracy can limit PDE accuracy even when local
polynomial reproduction passes. Quadrature kinds: `"chart-cubature"`
integrates a declared `BoundaryAtlas`/`CubatureAtlas` (e.g. `chart_box_atlas`)
and transfers the exact chart/Jacobian rule to nodes by bounded GMLS values,
reporting `reference_area` and a moment `transfer_residual`; compiled
`"tangent-voronoi"` is a positive local estimate that refuses open patches;
area-normalized density requires an explicit area and does not independently
recover it.

`SurfaceTangentCalculus` adds tangential vector and typed tensor operators:
`covariant_gradient`, `covariant_derivative(TensorType)`, `exterior_derivative`,
`tensor_divergence`, distinct `"bochner"` and `"hodge"` vector Laplacians
(their difference is the Gauss-equation `ricci` action), `chart_components`
through metrix index raising, a tangency-constrained saddle system and a
closed-surface Stokes/Brinkman block system.

Native surface reconstruction has partial coverage over an explicit region
admission envelope. That envelope is not a volumetric extension of the surface
field. Off-surface queries are refused; sampled-only sources admit sample sites,
not an invented exact continuous surface. Ambient field derivatives are not
published by this value-only reconstruction.

Fixed-support refresh retains an anchored reference geometry. Measure updates
use the current/reference fitted area-Jacobian ratio, so unchanged geometry
cannot accumulate area drift. Host prepared identities distinguish actual source
programs and numeric revisions, not diagnostic NaNs, coincident samples, or
generic display names.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_surface_laplace_beltrami.py --size 256
JAX_ENABLE_X64=1 python examples/meshfree_open_surface_diffusion.py
JAX_ENABLE_X64=1 python examples/meshfree_surface_vector_pde.py
```

The combined sphere/torus workflow requires at least 256 points. Its torus chart
uses eight neighbors and an explicit candidate budget; its sphere uses a
different declared support. Smaller requests are refused rather than silently
resized or admitted by loosening geometric thresholds. On sampled torus clouds
of 1024–8192 points, degree-4 height charts and stencils reduce the
Laplace–Beltrami error 5–9x relative to degree 2 (measured rates against mean
support radius 4.8/3.9/3.2 versus 1.5/2.5/3.2); this is a measured campaign,
not an asymptotic guarantee.

### Surface Stokes in strain form

`phydrax.solver.MeshfreeSurfaceStokesPlan(calculus, viscosity=nu, reaction=r)`
solves closed-surface Stokes/Brinkman flow with the physical viscous stress
`2 nu E(U)`, `E(U) = 1/2 P (nabla U + nabla U^T) P`, composed from
`SurfaceTangentCalculus.covariant_gradient` and `tensor_divergence`. Pressure
coupling, the divergence/tangency constraints, and the measure-mean pressure
gauge are the calculus's own Stokes blocks; the block operator is prepared once
with the declared native `LinearSolvePolicy`. On a sheet with Gauss curvature
`K`, `2 Div E(U) = Delta_B U + K U + grad div U`, so rigid motions (Killing
fields) carry no strain or dissipation. For divergence-free flow on a
constant-curvature surface, the vector-Laplacian form of `stokes_system` matches
the strain form only after the reaction shift `r - nu K`.
`MeshfreeSurfaceStokesResult` reports velocity, pressure, normal
multipliers, `max |N^T U|`, `max |div U|`, the gauge residual (discrete
divergence-theorem defect), the dissipation `2 nu int E:E`, and the native
linear result and status. Open surfaces are refused until velocity boundary
rows exist. With zero reaction, Killing fields lie in the velocity kernel and
no rigid-motion gauge is imposed; inspect the native status. The block system is
nonsymmetric and, without preconditioning, poorly conditioned; the generic
planner's GMRES(20) stalls on a 400-node sphere. `linear_policy=None` therefore
selects `GMRES(restart=min(400, unknowns))`, relative tolerance `1e-10`, at most
4000 steps, and status-reported failure (about 530 iterations on that sphere).
An explicit `LinearSolvePolicy`, including preconditioning, replaces it.

## Motion, shifting, and topology epochs

`MovingSurfaceState` stores extensive content, measures, geometry, the exact
`TopologyEpoch`, and a bounded rolling window of live histories. Concentration is
recovered as content divided by measure. `MovingSurfacePlan` takes a geometry
provider, a typed motion law, an intensive reaction, and a named
`AdditiveIMEXScheme` (`"forward-backward-euler"`, `"ars-222"`, `"ars-443"`) or an
`AdditiveIMEXTableau` whose explicit and implicit parts share stage times. It
drives native `ConservationIMEXMethod` over one packed state (content, a
measure witness, the reaction ledger, and integrated coordinates), so geometry,
measures, motion, reaction and diffusion are evaluated at every stage's own time
and coordinates. The implicit content solve is prepared once as a native
`LinearSolveTemplate` and bound to each stage's coefficients.

Typed motion laws are `PrescribedVelocityMotion`, `ChartMotion`
(authoritative positions, never time-integrated), `LevelSetMotion` (exact normal
speed `-phi_t/|grad phi|`), `MeanCurvatureMotion` (Laplace–Beltrami of the
embedding coordinates) and `BulkDrivenMotion` (a native bulk field
reconstruction queried at stage points). Each splits the velocity into the shared
normal speed and the tangential mesh velocity and returns the measure-rate source
`w (H V_n + div_G u_tau)`. Non-chart `"normal-only"` laws discard tangential
components; `"material"` laws move the mesh with the material. An authoritative
`ChartMotion` requires material motion: discarding its tangential chart velocity
would disagree with its prescribed node trajectory and omit relative transport.
`LevelSetMotion.gradient_floor` participates in its law and checkpoint-owning
plan identity because it changes both velocity and admission.
`SurfaceGeometryProvider(surface, diffusion, mode=...)` is the stage geometry of
one prepared support epoch, a callable PyTree whose surface and diffusion arrays
stay dynamic leaves. `"similarity"` refreshes modulo translation and isotropic
scale with the exact similarity weights, so the support trust bounds only the
non-similar deformation. The geometry, reaction and law callables of a plan are
PyTree leaves, not static metadata; `plan_id` and each `law_id` are the explicit
identities of these opaque callables.

The growing-sphere example binds cochain data, node measures, and reference
coordinates with `eqx.Partial`; a plain function closure would hide those arrays
from PyTree traversal. Its callback layout and operator/law/plan identities
intentionally changed. Rebuild prepared plans and checkpoints; previous example
IDs are not aliases. One module-level compiled step receives the plan, state,
and numerical step size as dynamic arguments and returns the complete native
step result, including refusal and conservation evidence.

An epoch transition publishes its target runtime only together with an accepted
target state. A refused remap retains the source runtime and source state, so
the held pair remains usable.

A surface whose `LocalStencilPolicy` declares a `SmoothSupportEnvelope` builds
its support epoch from the envelope's candidate relation: compact weights vanish
smoothly at the fixed physical radius, and fixed-support refresh admits every
anchored displacement strictly below the declared envelope. The compact weight
uses the same ambient distance that selected the candidates (a tangent-chart
distance never exceeds it and could weight a source outside the envelope); the
polynomial basis stays in chart coordinates. Without an envelope the trust is
the nearest-neighbor selection gap, which on irregular samples can be orders of
magnitude smaller than the spacing.

The measure witness integrates the source identity with the tableau's explicit
weights. `MovingGCLPolicy` admits a step when the relative measure-rate defect
`max|w(X_end) - W_end| / (dt max w)` is below its tolerance, or reports it only
(`"diagnostic"`); the endpoint measure-rate residual is always diagnostic.
`MovingMeasureLaw` selects the measures of record: `"geometric"` (the refreshed
measures; the witness is the check above) or `"conservative"` (the witness
itself; concentration is content over a measure advanced by the same discrete
divergence as the relative content flux, so the GCL defect vanishes by
construction and `geometric_drift` reports the accumulated relative difference
from the refreshed measures).
`MovingSurfaceEvidence` keeps trust, tube, geometry, motion, relative transport,
positivity, diffusion conservation, content-ledger conservation, GCL, solver and
history admissions as independent flags beside the first refusing
`MovingSurfaceStatus`. A rejected candidate leaves the accepted state intact.

Live histories occupy ring slots `lifetime % capacity`; `live_slots()` and
`live_lifetimes()` list them oldest first. With `archive="rolling"` the oldest live
history is evicted once the window is full, so runs are unbounded. With
`archive="acknowledged"` eviction requires `acknowledge_archive` to have advanced
the archive cursor past the evicted lifetime; otherwise the step is refused with
`HISTORY_EXHAUSTED`.

Tangential shifting is a mesh velocity, not a physical motion law.
`SurfaceMeshShift(SurfaceShiftPolicy(...), exterior)` adds the policy's
tangential repulsion to the mesh velocity inside the plan's native stage rule:
the coordinates, the measures and the content all advance together, and
material crosses the moving mesh with the relative velocity through the native
`MeshfreeAdvection` upwind rate on the surface exterior graph (frozen metric
weights, stage coordinates). The shift's measure source is that graph's volume
rate, so a shift requires `measure_law="conservative"` and a constant
concentration stays exactly constant. Authoritative chart motion refuses a
shift. Evidence reports `transport_cfl` (the step times the largest stage
outgoing relative-flux rate), the minimum candidate concentration and, with
`require_positivity=True`, a `POSITIVITY_REFUSED` sign check; positivity is
certified only for a nonnegative metric under the forward-Euler outflow bound.
`SurfaceResamplingPolicy` prepares bounded deterministic sample-repair
proposals: at most one insertion, removal or relaxation per iteration, with
witnessed spacing and quality triggers, projection, and a capacity proposal.
`SurfaceResamplingResult.source_indices` records the input sample of every
proposed sample (`-1` for an inserted probe). Nonconvergence and capacity
refusal remain visible. Resampling never manufactures a physical topology event.

`PointTransferPlan` prepares one declared concentration transfer on fixed sparse
routes, for bulk and surface supports alike. `PointTransferRequest` declares the
constraints: `"conservative-signed"` (`w_newᵀ T = w_oldᵀ`),
`"conservative-positive"` (also `T >= 0`), or `"joint"` (conservation, `T 1 = 1`,
monomial moments up to `moment_degree`, optionally nonnegative), and the
objective (`"coefficient-change"` or `"measure-weighted-change"`). The executed
coefficients are the minimum change from the base coefficients. For signed
requests every target's constant and moment rows are eliminated exactly with
the native batched pseudoinverse (`z = C⁺c`, projector `I - C⁺C` per target),
so the native rectangular `MinimumNormProblem` (`linear_policy`) only carries
one conservation row per source; nonnegative requests use the native sparse
conic program (`conic_policy`). An already feasible base is kept unchanged.
`PointTransferEvidence` audits every equation, sign and coverage on the host:
conservation, constant and moment residuals, the correction and objective, the
Lebesgue constant `max_r Σ_e |t_e|`, and `PointTransferStatus`. A degree-`p`
exact transfer errs by at most `(1 + Λ) |f|_{p+1} ρ^{p+1} / (p+1)!`, so its order
needs a bounded Λ; a declared `lebesgue_bound` refuses larger amplification with
`AMPLIFICATION_EXCEEDED`. Uncovered sources/targets are refused with their
indices. If total measures differ, conservation and constant reproduction are
jointly impossible (summing both equation sets proves it); `MEASURE_OBSTRUCTION`
carries that explicit witness. A certified witness (left-null vector of the
local or reduced rows, or a Farkas ray, audited against the declared rows) gives
`INFEASIBLE`, distinct from `PROVIDER_UNRESOLVED`. Equal total area still does
not imply constant reproduction. `SurfaceTransferPlan` adds only the surface
preparation: moments use each target's tangent coordinates from its
authoritative normal, and
`normal_departure` reports how far the routes leave those tangent planes. The
coordinate dual is the transpose; the Hilbert adjoint includes both measures.
Values are differentiable through a frozen transfer:
`TopologyEpochTransition.pullback` is the VJP of `apply`, `adjoint` its Hilbert
adjoint, and `value_derivative_available` is set while
`differentiation_available` (epoch selection) stays false.

`stage_meshfree_epoch` and `commit_meshfree_epoch` own one staged
`phx.lifecycle` composition transaction per epoch change. Every entry
reachable from the epoch entry is either a derived artifact (geometry,
measures, routes, stencils, metric, hierarchy, coupling queries, compiled
derivative plans), which must be reprepared on the target epoch, or state
(current fields, every live history, predictors), which must be remapped by its
own frozen route; independent entries such as controller state are retained.
Missing routes or rebuilds are refused before staging, and a failed route in
any history returns the source composition object unchanged.
`MeshfreeEpochChange` has exactly one cause: `"sample-repair"` (a converged
resampling proposal), `"surface-event"` (`MeshfreeEventLineage` of a committed
multiregion pass, with event kinds, CCD/volume evidence and region parents), or
`"adaptive-refinement"` (an admitted `MeshfreeAdaptationProposal`, see
[Adaptivity](#adaptivity)).
`surface_event_epoch` adapts a committed pass: the support is the active sheet
slots (`MeshfreeSheetSupport`) and the transfer is the concentration form of the
authority's own conservative sheet transfer. `remap_live_histories` transfers
all histories of one change through their own routes or reports every failure.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_surface_topology_events.py
```

Epoch transitions remap concentration and then reconstruct extensive content.
Every live history has its own source/target measures and transfer, keeps its
ring slot and lifetime index, and the archive cursor is preserved. Failed history
transfer refuses the entire transaction; epoch geometry derivatives are
unavailable. `MeshfreeCapacityPolicy` records storage buckets, while native
positive spaces contain compact active rows only, never zero-mass dummy nodes.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_moving_surface_reaction_diffusion.py --size 48 --method ars-222
```

The example uses a positive native graph diffusion and explicitly does not claim
high-order Laplace–Beltrami accuracy for that graph. Its geometry is the native
surface refresh along an authoritative growing-sphere chart. It exercises
growth, reaction, shifting, two topology changes with next-epoch target
measures, complete live-history transfer, continued integration, and rollback.

## Sensitivities and nonsmooth events {#sensitivities-and-nonsmooth-events}

Meshfree derivatives are published only for maps that are actually smooth at the
evaluated state, with the owner's status and rank/conditioning evidence. A
refused map is NaN together with its tangents and cotangents; there is no masked
zero gradient.

**Fixed-support coordinate sensitivities.**
`PreparedPointCloudDiscretization.coordinate_sensitivity(points)` returns a
`PointCloudCoordinateSensitivity`: a `PreparedLinearization` of the map from point
coordinates to every prepared derivative stencil (`mixed_weights` order), its
`LocalStencilRefreshStatus`, `accepted`, and per-row `LocalStencilEvidence` (rank,
condition, minimum singular value). `point_cloud_reconstruction_sensitivity`
publishes the same contract for reconstructed values at fixed queries, with
per-query `FieldQueryEvidence`; its candidates are frozen at preparation within
`radius + coordinate_envelope`, so derivatives exist strictly inside the declared
envelope (the default zero envelope admits the anchor and has no coordinate
derivative).

```python
sensitivity = cloud.coordinate_sensitivity(moved_points)
assert bool(sensitivity.accepted)
tangents = sensitivity.linearization.jvp(direction)
gradient = sensitivity.linearization.vjp(cotangents)
```

**Nearest-neighbor supports are not smooth.** A k-nearest selection is
differentiable only strictly inside its selection gap (`trust_margin`, a quarter
of the gap). At a tie the admitted anchor has a NaN derivative and any motion is
`SUPPORT_EXCEEDED` with NaN weights; neither PHS-RBF-FD nor nearest-neighbor GMLS
claims smoothness through a neighbor change.

**Smooth fixed-radius GMLS.** A `SmoothSupportEnvelope(radius, displacement)`
declares a fixed physical support radius and an anchored displacement envelope.
`MeshfreeNeighborhoodPlan(..., neighbors=capacity, envelope=support)` holds every
source within `radius + 2 * displacement` of each target (the union of physical
supports over every admitted trajectory) in a fixed candidate capacity; a row
with more candidates is refused at preparation, never truncated.
`LocalStencilPolicy(support=support, weight_kernel=..., coordinate_order=p)` scales
offsets by the fixed radius and uses a compact kernel vanishing at the cutoff
through coordinate order `p`: `"wendland-c2"` (through order three),
`"wendland-c4"` (default, through five), or `"compact-polynomial"`
`(1 - r^2)^(p + 1)`. A neighbor entering or leaving the physical radius is then a
zero-weight event of a fixed relation: weights are `C^p` in the coordinates with
matching two-sided derivatives at the cutoff, and per-row rank evidence of the
positively weighted design is retained at every state (rank loss is
`ROW_REFUSED`). Motion at or beyond the envelope is `SUPPORT_EXCEEDED` (NaN);
rebuilding the envelope or changing the point population is an epoch event.
Envelope neighborhoods and smooth policies are only accepted together.
`PointCloudPlan(..., stencil=LocalStencilPolicy(support=...), neighbors=capacity)`
uses the same route.

**Frozen-remap value derivatives.** Across an epoch change the integer topology
decision is fixed, but values cross the frozen conservative routes linearly.
`remap_live_histories` values are differentiable in every live history (each
history through its own route and measures); `MeshfreeHistoryRemap.pullback` and
`.adjoint` publish each history's coordinate dual and Hilbert adjoint for adjoint
sweeps outside one JAX trace. A failed remap is NaN and refuses both reverse
maps. Epoch selection, split/merge events, and geometry changes have no
derivative (`require_differentiable_selection` raises).

**Unavailable derivatives.** Weak complementarity, genuine active-set changes,
and topology selection have no default smooth gradient. Steady nonlinear coupled
implicit sensitivities remain disabled until a linearization acceptance audit
of the coupled root exists; transient coupled derivatives are NaN for an
unaccepted solve.

## Adaptivity {#adaptivity}

Adaptive h/p/support refinement is a decision layer over the existing
owners: indicators rank points, a marking selects stable IDs, a bounded
proposal describes the candidate cloud, and one `adaptive-refinement` epoch
publishes it atomically or not at all. Every indicator is an *indicator*: no
reliability or efficiency constant is claimed, and a local fit residual is not
a PDE error bound. Report errors against independent references.

- `probe_residual_indicator(cloud, u, residual)` evaluates a declared strong
  form at off-node probes (midpoints of each point's nearest edges) from one
  fitted source-to-probe stencil family (value, gradient, Hessian in
  `MeshfreeProbeJet`). Collocation zeroes the nodal residual, so only off-node
  probes carry information; `admit=` drops probes outside nonconvex domains.
- `flux_jump_indicator(cloud, u, diffusivity=K)` compares the two one-sided
  quadratic reconstructions' normal fluxes at edge midpoints, each with its own
  diffusivity (one-sided at material interfaces).
- `degree_difference_indicator(lower, higher, u, measures, ids)` applies the
  same declared operator fitted at degrees `p` and `p + 1` (bulk Laplacians,
  surface Laplace–Beltrami) and reports `sqrt(m_i) |(L_{p+1} - L_p) u|`.
- `support_quality(cloud)` exposes stencil condition, amplification and
  support spacing ratio; its `indicator` ranks by amplification.

`mark_points(indicator, MeshfreeMarkingPolicy("dorfler" | "maximum",
fraction=..., maximum_marked=...))` reuses the finite-element Dörfler and
maximum strategies with stable-ID tie breaking; selections beyond the cap keep
the largest indicators and report `capped` and the achieved bulk share.
Dörfler marking on an indicator spanning many orders of magnitude (an
exponential peak) selects only a handful of points; the small refined patches
then place their 2:1 transition bands where the solution is still steep, and on
the sphere this measurably made the error worse than uniform refinement. Use
`"maximum"` marking to refine such a region as a whole (the example's
`sphere_marking`).

`propose_adaptation(support, indicator, marking, MeshfreeAdaptationPolicy(...))`
returns a deterministic `MeshfreeAdaptationProposal`. Refinement is graded:
the marking, its `closure_neighbors` nearest neighbors, and then every point
whose expected spacing exceeds `grading` (> 2, default 2.05) times that of a
refined neighbor are refined. The residual indicator is local while elliptic
pollution is not: on the `eps = 0.04` boundary layer the max-norm error sat
in the unrefined band beside the layer, at a point the indicator ranked 124th
of 665, and with grading 2.5 three levels stalled (0.0290 → 0.0300 at 383 → 665
points). The default 2.05 widens the graded band and two levels reached 0.0183
at 520 points. Scale-normalized stencil amplification stayed at the uniform
level, so the band was under-resolved, not destabilized. Refined points bisect
their nearest edges; children too close to an existing point or a
higher-priority child are dropped through a
stable-key maximal independent set; low-indicator unrefined interior points
are removed as a maximal independent set (never a new child's parent), lowest
first up to the cap.
Boundary-edge children need `boundary_projection` (otherwise they are counted
as unresolved and dropped); surface children need `manifold_projection`.
Retained points keep their IDs; children receive fresh IDs above every source
ID in canonical parent order. A proposal above `maximum_inserted` or
`maximum_points` is `CAPACITY_REFUSED` whole, never truncated, and cannot
become an epoch. `target_degree` / `target_neighbors` request cloud-wide
degree or support changes: per-row degree needs a separately fitted basis per
degree class in the point-cloud owner and is not offered. Target measures
follow `meshfree_fill_measures`, a declared normalized fill-volume quadrature
whose total equals the declared domain measure.

`prepare_adaptation_transfer(cloud, proposal)` corrects the cloud's own value
stencils into the joint conservative and constant-preserving transfer of the
native owner (both clouds must use the same declared total measure).
`moment_degree=q` adds moments, which with conservation also requires both
quadratures to integrate every monomial of degree `<= q` identically; two fill
quadratures generally do not, and the owner then certifies `INFEASIBLE` with a
left-null witness. `adaptation_acceptance(proposal, transfer, target_cloud.report,
solve_successful=..., stability=..., indicator_before=..., indicator_after=...)`
lists every failed criterion; its `accepted` Boolean is the explicit boundary
passed to `commit_meshfree_epoch`. For square collocation, prepare the target
Poisson plan with `stability="diagnostic"` and pass `prepared.stability`: a
successful solve of an operator with a spurious nonpositive-real eigenvalue
can be badly wrong, and such a candidate is refused as `operator-unstable`. A
rejected boundary returns the source composition object unchanged.

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_adaptive_learning --quick
```

The example refines a nonpolynomial bulk boundary layer and a peaked zonal
field on the sphere, compares each with uniform refinement at equal point count
and reports observed solve time, commits accepted epochs, and shows a rejected
coarsening transaction and a capacity refusal.

## Bulk–surface exchange {#bulk-surface-exchange}

`MeshfreeComponent` publishes native residual, reconstruction, and capacity
contracts to `phydrax.solver.coupling`; it does not advertise a facet trace.
`SurfaceExchangeLaw` uses complete constant-reproducing value queries of any
reconstruction-capable bulk (meshfree, finite-element, ...) and their exact
coordinate transpose to pair bulk loss with surface gain on a native meshfree
surface `MeshfreeComponent`. The law records the surface owner revision, point
enumeration, and capacity diagonal it was built on and refuses a stale
same-shaped surface at preparation.
`LangmuirAdsorptionFlux` reuses the native adsorption kinetics.

`SurfaceDeposition` selects the amount partition. `"signed"` admits any
constant-reproducing route: signed polynomial weights conserve through the law
but claim no positivity. `"positive"` requires a nonnegative bounded gather
route (for example a canonical Shepard reconstruction). Its certificate also
gates that no bulk node gains amount while the surface adsorbs. The evidence
records the mode and the minimum query weight. `MeshfreeBulkSurfaceMethod` uses
the same positive-partition admission. Source program and numeric revision
admission rejects stale same-shaped reconstruction.

Moving exchange sites are relocated at host coupling-window boundaries. Within
a window they are frozen; lag and displacement are evidence, and within-window
geometry differentiation is not claimed. `refresh_at_window` relocates a
fixed-topology surface endpoint. `relocate_at_epoch` handles a resampled or
topology-changed surface whose point count may change. It stages one meshfree
epoch in which the re-prepared exchange and the target surface are derived
artifacts, and every surface state history crosses through its own
conservative route. A failed route or an unaccepted boundary publishes nothing,
and the returned `SurfaceEpochRelocation` then keeps the source law and state.
The coupled example reports any metric slack separately from its exact amount
balance.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_bulk_surface_exchange.py --size 128 --dimension 3
```

## Hybrid overlap coupling and calibration {#hybrid-overlap}

A point cloud near the true boundary and a cell-centered finite-volume grid in
the interior can discretize one original elliptic problem on overlapping
subdomains. Declare the cloud's artificial rows as zero-data Dirichlet rows of
its `PointCloudPoissonPlan` and publish its physical sparse rows through a raw
`MeshfreeComponent` with the physical right-hand side as `load`. Declare the
grid's artificial faces as zero-data Dirichlet faces of its
`ConservativeDiffusionPlan`. Then `OverlapDirichletLaw` with
`PreparedOverlapTransfers` closes each artificial boundary with the other
owner's native value stencils, evaluated at sites taken from the owners' own
geometry. The coupled solution is the composite discretization. The law's
gated certificate is the artificial-node relation, and the owners' overlap
mismatch is reported as discretization evidence.

`prepare_overlap_schwarz` supplies one exact-block sparse factorization term per
subdomain to the native additive (parallel) or multiplicative (alternating)
subspace-correction builders. GMRES with these terms is no private Schwarz
loop. The example runs a refinement study on three levels. It reports the
analytic errors of both owners with observed rates, the overlap mismatch, the
law and component certificates, and a monolithic point-cloud reference. Each
level also records Krylov iterations without preconditioning (with its native
status) and with additive and multiplicative Schwarz.

Calibration uses `phydrax.uq` only. Complete cases are independent
manufactured problems drawn i.i.d. from a declared parameter family; that is
the exchangeability assumption. The predictor is the coarse hybrid solve, and
the target is the analytic solution on a fixed probe set. Disjoint train,
calibration, and test case IDs (`ProcessValidationSplit`) fit the per-probe
score scale, the finite-sample radius (`ProcessConformalCalibrator`), and the
measured simultaneous coverage (`process_conformal_diagnostics`), in that
order. An out-of-family boundary-layer regime and a narrower-overlap geometry
report their measured coverage and width separately, with no distribution-free
claim.

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_hybrid_calibrated
```

## Incompressible meshfree flow {#incompressible-meshfree-flow}

`phydrax.solver.MeshfreeIncompressibleFlowPlan(exterior, cloud, viscosity=nu,
domain_betti_number=b1, ...)` advances transient incompressible
(Navier–)Stokes flow. The state is a nodal velocity on the compact nodes of a
prepared exterior graph.

The exterior must be closed: it must carry an admitted positive metric and no
boundary rows, as in periodic boxes and closed manifolds. Bounded steady flow
belongs to `MeshfreeGeneralizedStokesPlan`. `cloud` is a prepared point cloud
on exactly the exterior's nodes. It supplies the transport reconstruction and
the nodal GMLS divergence diagnostic. Construction refuses open graphs, a
negative viscosity, a nonpositive reference density, and a Betti number
larger than the graph cycle space.

Each step (`advance`, or `step` inside native rollouts) is one native additive
IMEX step (default `"ars-222"`) of the projected system, composed from
existing owners:

1. Explicit conservative momentum transport by `ConservativeTransport`, plus
   density transport for `density_model="variable"`; `transport_scheme` is
   `"limited"`, `"reconstructed"`, or `"upwind"`. An optional
   `body_force(time, points)` acceleration and the state's pressure gradient
   (incremental pressure correction) enter the explicit part.
2. Implicit viscous diffusion with the exterior's conservative graph
   Laplacian, which conserves momentum and dissipates energy.
3. At the end of every implicit stage, projection of the velocity's de Rham
   edge cochain `u_e = (u_i + u_j)/2 . (x_j - x_i)` by
   `CompatibleIncompressibleProjection` or the variable-density owner, on the
   exterior's native cochain complex. The graph divergence of the projected
   cochain vanishes to solver tolerance, and its volume flux advects the next
   explicit stage, so every stage rate uses its own projected state and flux.
4. Reconstruction of nodal velocity by `MeshfreeVelocityReconstruction`. It
   uses moment-consistent weighted least squares
   `v_i = M_i^{-1} sum_j w_ij u_ij (x_j - x_i)`, the Hodge adjoint of the de
   Rham map, and publishes its conditioning.

The tableau's stages after the first, and its last stage, must be implicit;
a tableau that is not stiffly accurate projects its combined result once more.
The nodal projection `v - R(I - P) I v` is approximate: it leaves an `O(h^2)`
share of the gradient it removes. Removing only the pressure increment keeps
that share `O(h^2 dt^2)` per step, so the velocity keeps the tableau's
temporal order. A flux lagged over the step, a single projection after the
step, or a projection of the whole pressure gradient each step is first order
in time. The state's `pressure` is the step pressure whose nodal gradient
balances the momentum update (a midpoint quantity, first order in time for
ARS(2,2,2)); `initialize` sets it to zero. On the periodic Taylor–Green
lattice, `benchmarks/meshfree_incompressible_refinement.py` measures the
temporal (fixed `n`, halved `dt`) and spatial (fixed small `dt`) orders
separately.

A radius graph has `E - V + C` independent cycles, while the domain has only
`b_1` harmonic fields. Edge content that the nodal reconstruction cannot
represent is an artificial cycle mode. `classify_edge_velocity` returns
`MeshfreeCycleEvidence`: the cycle-space and artificial dimensions,
`nonphysical_fraction`, and `physical` against `cycle_tolerance`. A projected
field above the tolerance is refused; it is never accepted as flow.

`MeshfreeFlowStatus` reports `INVALID_STEP`, `NONFINITE`,
`NONPOSITIVE_DENSITY`, `CFL_REFUSED` (against `cfl_limit`),
`TRANSPORT_REFUSED`, `IMPLICIT_REFUSED`, `PROJECTION_REFUSED`,
`RECONSTRUCTION_REFUSED`, and `SPURIOUS_CYCLES` in that precedence. A refused
step returns the unchanged state and still publishes the candidate.
`initialize` projects an initial velocity and, if refused, publishes the
unprojected input. `MeshfreeFlowEvidence` records the CFL number, the implicit
and pressure iterations and residuals, graph and nodal divergence before and
after, the reconstruction condition, the cycle evidence, and the kinetic
energy, mass, and momentum ledgers.

`PreparedMeshfreeIncompressibleFlow` is a native `AbstractFixedStepMethod`.
`solve_fixed_step(FixedStepProblem(flow, state, t0=..., t1=...,
step_size=..., state_geometry=EuclideanStateGeometry()),
evidence_retention="steps")` runs a traced rollout and retains every step's
`MeshfreeFlowEvidence`. After a refusal the rollout holds the committed state,
and `FixedStepEvidence.refused` and `refused_step` identify the refused step.
A new rollout from the held state continues the run.

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_incompressible_flow
```

The example advances a perturbed periodic Taylor–Green vortex. It reports
graph and nodal divergence, kinetic energy against the exact decay
`e^{-4 nu t}`, momentum, and cycle content. It classifies random
graph-solenoidal noise as non-physical, and it refuses a step that violates
the CFL bound, rolls it back, and continues. It also solves a manufactured
bounded Stokes problem with the stabilized mixed owner.

## Lagrangian particle flow {#lagrangian-particle-flow}

`MeshfreeLagrangianFlowPlan` declares a material (Lagrangian) incompressible
flow on moving GMLS points. Its measure is an explicit `EvolutionMeasure`.
`"quadrature-volume"` points carry an authoritative quadrature volume `V`,
evolved by the discrete geometric conservation law, and mass `m = rho V`.
`"material-mass"` points carry the persistent masses and identities of a
`MaterialParticleMeasure`; their volume is `V = m / rho_SPH`, from the native
SPH summation density on the measure's Verlet cache. Masses never change, so
each accepted step conserves total mass exactly.

`PreparedMeshfreeLagrangianFlow.step` runs these phases in order:

1. Refresh the anchored GMLS support at the accepted positions. If the refresh
   refuses, re-prepare the cloud with the same stable ids (`rebase`) and report
   `reprepared`.
2. Predict `u* = u + dt (g + nu lap_h u)`.
3. Solve the weak projection as the native minimum-norm least-squares problem
   `min_p ||G p - (rho/dt) u*||_M` with LSMR. Its normal equations are
   `G^T M G p = (rho/dt) G^T M u*`. There are no boundary rows: a fully
   periodic address gives a periodic projection, and a bounded cloud gets the
   natural weak free-slip boundary, where `u.n = 0` holds only weakly.
4. Correct `u = u* - (dt/rho) G p` and move the points, `x += dt u`.

The projected velocity has zero weak divergence `-M^{-1} G^T M u` to solver
tolerance. Because the projection is `M`-orthogonal, kinetic energy with masses
`rho M` cannot increase. The strong GMLS divergence is published as a
consistency measure, not as a projection invariant.

The kernel of `G` is not only the constants. On symmetric lattices the
antisymmetric GMLS weights also annihilate odd-even modes, and nearby clouds
have near-null modes. A gauged square solve of `G^T M G` is then singular or
ill-conditioned beyond its gauge; GMRES with ILU on it stagnated or broke down
on periodic lattices beyond 12 x 12. The least-squares form needs no gauge and
is consistent by construction. Its Krylov iterates stay in the row space of
`G`, so the minimum-norm pressure carries no kernel content and `G p` is exact
to solver tolerance whatever the kernel. `linear_policy` must be an
unpreconditioned status-mode `LSMR` policy, since preconditioning would change
the minimized norm. The solve template is prepared once per support epoch, and
each step binds the refreshed stencil weights to it.

The projection's gradient/transpose pairing, right-hand side, and divergence
evidence bind the incoming state's authoritative `volumes`, not the original
support cloud's quadrature weights. The support anchor remains unchanged until
an actual support rebase. `weak_divergence(velocity)` without a state is an
anchor-quadrature diagnostic, not the moving step's current-volume evidence.
Transactional incompressible plans require status-returning pressure policies;
a raising failure mode is refused before it can bypass projection rollback.

The collocated Neumann route is not offered. Its wall rows replace the pressure
equation, so wall-row divergence is uncontrolled and grows under material
motion.

On a regular lattice, choose `neighbors` to complete distance shells (21 in 2-D
for degree 3). A truncated shell breaks the stencil symmetry, and the weak
divergence then sees spurious sources.

Acceptance is transactional through `lifecycle.commit_candidate`. A step is
refused for an invalid step size, a nonfinite candidate, a failed Verlet
update, a status-mode pressure solve failure, or a Courant violation. A refused step
keeps the accepted state and still publishes the candidate, a
`MeshfreeLagrangianStatus`, and `MeshfreeLagrangianEvidence`. The evidence
includes:

- weak and strong divergence before and after the projection;
- the mass, momentum, center-of-mass, energy, and volume ledgers;
- body, viscous, and pressure impulses (the pressure impulse is reported, never
  claimed zero);
- the refresh status;
- pressure solve status, iterations, normal residual, and LSMR condition
  estimate.

`result.fixed_step` is the native `FixedStepResult` record of the attempt. Cloud
refresh and re-preparation are host work in every step, so the flow is driven by a
host loop, not by a traced `FixedStepRolloutPlan`.

`MeshfreeMeasureTransferPlan` transfers material content between two declared
point measures, for example particles onto a fixed GMLS cloud and back. It
composes cross-target value stencils with `PointTransferPlan`; the default is
`conservative-positive`, and `joint` moments are available on request. Mass and
momentum are transferred as extensive content, so the audited conservation
equations conserve them to round-off. The result reports mass, momentum, and
center-of-mass defects and the density mismatch. If the target routes do not
cover every source, the status is `UNCOVERED_SOURCE` and `apply` refuses.

`MeshfreeSPHReconstruction` exchanges reconstructions only. It evaluates the
native SPH summation density and continuity divergence on a Verlet relation and
compares them with the GMLS divergence and the measure density `m/V`
(`compare`). Its `convert` method changes the authoritative measure at fixed
points (`V = m / rho_SPH`) and reports the mass defect and density mismatch.
IISPH and DFSPH remain separate SPH owners. Their public `initialize_state`
accepts these particle positions and velocities directly, so there is no
second pressure loop or adapter. When `MeshfreeLagrangianFlowPlan` pairs SPH
with a periodic GMLS address, the Verlet relation's box must be
`ParticleBox.from_address(address)`: the same bounds and periodic mask, so both
reconstructions use one periodic cell. A different box, or a periodic box
without a periodic address, is refused.

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_lagrangian_flow
```

## Constitutive learning

`EdgeFrameFeatures` declares scalar, vector, and symmetric-tensor features,
scientific source identity, and edge-reorientation parity.
`MonotoneEdgeConductance` uses an input-convex potential with an odd symmetrized
slope. Context is immutable external data; arbitrary unknown-state averages
would not establish global operator monotonicity. A negative-offset positive
transform cannot claim the native input-convex construction certificate.

`LipschitzEdgeFlux` derives bounds for admitted canonical model/activation
families. Unsupported low-rank/model families are refused. Contraction evidence
uses the declared background-energy norm; uncertified contraction remains
uncertified, not a claim inferred from a spectral estimate.
`EdgeFeatureCoverage` combines quantile and covariance-support evidence using
native property verification/factorization, with no inverse or jitter repair.

Coupled multi-component laws act on a packed O(3) edge state `D` (the
endpoint difference of a scalar/pseudoscalar/vector/pseudovector/rank-two
tensor field declared by an `O3Representation`) in a three-dimensional frame
and return a flux in the same representation. Both laws are exactly odd under
edge reversal, `F(-D; -t, even, -odd) = -F(D; t, even, odd)`, and covariant
under proper and improper frame changes. `MonotoneCoupledEdgeFlux` is the
gradient of an invariant convex potential: `O3EdgeInvariants` maps the state
and the polar edge frame `(1, t, t t^T - I/3)` through O(3) tensor products to
linear invariants `l` (entered as `(l, -l)`) and convex squared channel norms
`q`, and an input-convex network declared with
`input_monotonicity="nondecreasing"` composes them. An unconstrained input-convex
network of nonlinear invariants is not convex in the state and is refused. The
flux subtracts its zero-jump value, so `energy` is a dissipation potential at
least `b |D|^2 / 2`, and the `InvariantConvexEdgeCertificate` states whole-state
strong monotonicity `b`; no global Lipschitz bound is claimed.
`LipschitzCoupledEdgeFlux` adds an orientation-odd `O3EdgeNetwork` perturbation
whose single whole-output Lipschitz bound is derived from the current
tensor-product weights, never concatenated from per-component estimates.
`derivative` returns per-edge Jacobian blocks and `linearize` a native
`PreparedLinearization` for matrix-free Jacobian and transpose actions.

`EdgeFeatureCoverage` is train-fitted empirical support, not calibration.
Predictive calibration uses the UQ owner directly; meshfree has no conformal
class. Declare disjoint complete physical cases with
`phydrax.uq.ProcessValidationSplit(train_ids, calibration_ids, test_ids)`, solve
each complete calibration case once, stack predicted and reference fields on a
case axis, and fit `ProcessConformalCalibrator.calibrate_observable` (scalar
observables use `SplitConformal`, fields `FunctionalConformal`) or
`calibrate_trajectory`. Measure held-out coverage and width on the test cases
with `interval_coverage`/`interval_width`, and report shifted-geometry or
changed-regime cases separately: finite-sample coverage assumes exchangeable
cases and is not a guarantee under drift.

`prepare_meshfree_conservation_solve` binds the native nonlinear and sparse
derivative owners. Array parameters are separated from immutable law metadata;
changed static traits require re-preparation. MODEL-authorized parameter bindings
and `SolverObjective` carry implicit gradients through the converged solve.
Primal, adjoint, coverage, conservation, and coercivity evidence reach the result.
The background stiffness assessment uses the exterior's `MeshfreeCoercivityPolicy`;
when it is unassessed a root may still converge, but no coercivity or contraction
certificate is claimed. A converged root and a contraction certificate remain
distinct facts.

`RuntimeInput("meshfree", "metric_weights")` (role `"coefficient"`) binds an
externally corrected edge metric in compact exterior edge order, replacing the
prepared metric in the residual, background solve, adjoint and coercivity
assessment. It is admitted only finite and strictly positive (otherwise the
forward is refused, never clipped); the prepared coercivity factor is refreshed
numerically at the runtime metric (`background_assessment`), never re-analysed,
and its parameter derivative is the implicit-root derivative.

`MeshfreeCoupledConservationProblem` and
`prepare_meshfree_coupled_conservation_solve` solve the block equations of an
`AbstractCoupledEdgeConstitutiveLaw` with one packed O(3) component vector per
node, through native `NewtonKrylov` on a fixed block sparse Jacobian, implicit
root derivatives, and an explicit block adjoint. Ledgers are per component
(`MeshfreeComponentConservationLedger`); coverage concerns the shared immutable
edge features. Coercivity needs whole-state strong monotonicity on a certified
background; Lipschitz coupled laws also report the Picard bound `L/b`.
Failed forwards have unusable public states and are rejected by training.
Their public-state and adjoint sensitivities use native portable-status guards:
both JVP and VJP are undefined NaN, never a masked plausible finite zero.
`require_success()` remains an explicit forward refusal boundary.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_learned_edge_flux.py --size 12 --dimension 2 --steps 80
```

Learning transfers a local law; it does not replace the global solve or promise
surrogate-speed inference. Data-efficiency and geometry-transfer claims require
the corresponding measured campaign.

## Learned metric and flux corrections {#learned-corrections}

A learned model may propose metric edge weights `w_c = w_base + delta`, but the
published metric is never the raw proposal. `MeshfreeMetricCorrectionPlan`
projects the candidate onto the full moment set `{w : A w = b}` of the prepared
exterior (every original row, including dependent rows) in the declared norm
`|w - w_c|_{Phi^-1}` (`Phi` is the metric prior unless `norm_weights` is given).
The signed route is the native rectangular `MinimumNormProblem` with
`GeneralizedLSMR` on the equilibrated design `E A S`, `S = sqrt(Phi)`; the map is
affine in the candidate and `project` is differentiable through the native
`rhs-only` derivative, which is exactly the weighted projector.

`MeshfreeMetricCorrectionPolicy(route, margin=..., sign=...)` declares the
route (`"signed-minimum-norm"` or `"nonnegative-conic"`), a required positive
weight margin, and the response to a positivity conflict: `"refuse"` publishes
no weights (`POSITIVITY_CONFLICT`); `"constrained"` solves the explicitly
selected native conic correction `min |w - w_c|^2_{Phi^-1}` with `A w = b` and
`w >= margin`. Weights are never clipped. A margin that no moment-exact metric
can meet is refused as `INFEASIBLE` with an independently audited Farkas
witness; incompatible moment rows are `MOMENT_INCOMPATIBLE` with a left-null
witness; nonconvergence stays `PROVIDER_UNRESOLVED`.

`MeshfreeMetricCorrectionEvidence` keeps the audits separate: every original
moment residual, the sign margin `min(w) - margin`, coercivity from the native
no-shift sparse Cholesky of the Dirichlet-reduced weighted graph Laplacian
(`MeshfreeStiffnessEvidence`; plans refuse an unassessed exterior), the
weighted correction norm, provider status and witness. `weights` exists only
for `ADMITTED`. `plan.tangent(correction, v)` returns the candidate tangent of
an admitted correction: the native projector JVP on the signed route, the
native fixed-active-set conic JVP on the conic route, NaN with
`available=False` where that derivative is not regular.

Undirected positive metric coefficients are not oriented fluxes. A learned
edge flux is a `MeshfreeEdgeFluxCandidate` evaluated in both endpoint orders;
`MeshfreeEdgeFluxCorrectionPlan` refuses a candidate whose parity defect
`forward + reverse` is nonzero (`PARITY_CONFLICT`, never antisymmetrized) and
projects an antisymmetric candidate onto the declared nodal balance
`G F = q` on equation nodes.

A correction model is trained end to end through the projection and the
implicit conservation solve: bind the corrected weights to
`RuntimeInput("meshfree", "metric_weights")` (role `"coefficient"`, one value per
compact edge) of a `MeshfreeConservationProblem`, and in the
`SolverObjective` measure pass `plan.project(candidate).weights` as that runtime
metric. The case is accepted only when `projection.admissible` (native success
and sign margin) and the conservation root are both accepted; with
`accepted_results="reject-attempt"`, `train_components` rolls back any other
attempt instead of stepping on it, and nonpositive runtime weights are refused by
the conservation primal itself. Derivatives combine the native `rhs-only`
projection derivative with the implicit root derivative of the solve.
`examples/meshfree_learned_metric_correction.py` (also run by `run_learned` in
`examples/meshfree_adaptive_learning.py`) checks the gradient against central
differences, trains a log-linear edge-feature model with `optax.lbfgs`, and
rejects a failed attempt.

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_learned_metric_correction --steps 6
```

## Production runs and restart {#production-runs-and-restart}

Long meshfree runs use the solver's production runtime; this layer adds no
second checkpoint format. Every fixed-step meshfree participant binds directly:
`PreparedMeshfreeEvolution.ssprk_method`/`imex_method`, the coupled
`MeshfreeBulkSurfaceMethod`, a `CallableFixedStepMethod` over a
`DistributedMeshfreeOperator`, and a moving surface through
`MovingSurfaceFixedStepMethod(plan)`. `ProductionRunPlan` runs bounded compiled
segments with exact output schedules, streaming moments and triggers, and keeps
`evidence_retention="terminal"` method evidence in `ProductionRunState.evidence`:
the evidence of the last accepted step, the most recent refused attempt (a retry
refusal inside a later accepted step or the refusal that stopped the run), their
step and attempt cursors, and the count of refused attempts. Storage does not grow
with the run; per-step history belongs to published outputs.

`meshfree_runtime_inventory(participant, source=..., program=plan.plan_id,
method=..., controller=..., precision=..., rng=...)` names every identity a
restore must match: build source, program, method, controller, precision and RNG
addressing, plus the native cloud, anchored geometry, support relation, stable
point IDs, measure realization, support trust, stencil, metric, boundary rows,
query/interface sources, owner partition, and capacity bucket.
`ProductionCaseManifest.from_inventory` derives the case identities from it, and
both checkpoint stores compare the archived inventory before any value is read; a
difference raises `StaleRuntimeCheckpointError` with the changed roles. Truncated
or foreign payloads are refused by the archive and store bindings. Restored Strict
values are validated once before use.

```python
plan = solver.ProductionRunPlan(
    evolution.imex_method("ars-222"), retry, step_size=dt, end_time=t1,
    maximum_steps=n, checkpoint_interval=10, segment_steps=5,
    output_schedule=schedule,
)
inventory = meshfree.meshfree_runtime_inventory(
    evolution, source=build_id, program=plan.plan_id,
    method=plan.method, controller=retry.policy_id, precision="float64",
)
manifest = solver.ProductionCaseManifest.from_inventory(
    inventory, problem_id="case", dtype="float64"
)
runtime = solver.PreparedProductionRun(manifest, plan, store, publisher=publisher)
state = runtime.resume(template) if restarting else runtime.initial_state(y0)
result = runtime.run(state, memory_sampling_interval=0.05)
```

Restart relations stay distinct. A same-topology restart uses the identity
relation. A refused support epoch (for example `SUPPORT_EXCEEDED` after ALE
motion exhausts trust) ends a run with a `step-rejected` failure and checkpoints
the held state; `support_epoch_relation(source_inventory, target_inventory,
evolution, evolution.rebase(held), held, source_template=held)` restores that
checkpoint into the rebased epoch through the rebase's own state map. An ownership
change uses `ownership_migration_relation(...)`, which commits
`DistributedPointLayout.migrate` once, records its packet/epoch evidence in a
`RuntimeMigrationReceipt`, and replays the same transfer on the archived
owner-blocked state. Each receipt admits only the roles its transport can change:
ownership moves partition, program and capacity; an epoch never moves ownership,
build, method or precision. Migration restarts require an `ArtifactCheckpointStore`.

An epoch relation binds the complete archived source-anchor content, not only
its inventory. A relation built at a later, uncheckpointed anchor refuses an
earlier checkpoint before destination lineage publication. Migration restorers
retain transport/template arrays as dynamic callable leaves, validate unpacked
source and reconstructed destination values, and intentionally change the
relation's PyTree representation; old container fingerprints are not reused.

Ownership restart transports every retained output through source decoding,
admitted owner transport, and destination binding, including already-delivered
snapshots. Output IDs, cursors, delivery flags, and logical payloads remain
unchanged. A first identity binding may commit `restart-lineage` without an
output acknowledgement; a repeated same-identity restart leaves its manifest
unchanged. Checkpoint staging admits state, outbox, and lineage array bytes before
encoding, then applies the separate encoded-payload limits.

A fresh process can restart inside a post-rebase epoch without knowing it. When
an `ArtifactCheckpointStore` restores through a migration relation, the
`restart-lineage` snapshot it writes carries a `CheckpointMigrationRecord`: the
`RuntimeMigrationReceipt` and the transported source state (for an epoch, the
held anchor). That record lives in the same atomic repository commit as the
first checkpoint of the new epoch, and every later snapshot of the artifact
carries it forward. No crash leaves a receipt without its epoch, and there is
no second checkpoint runtime. `store.migration_lineage()` reads the lineage
from the committed artifact before any runtime is bound.
`resume_support_epochs(evolution, store, inventory)` starts from the epoch-0
evolution and `inventory(evolution)`, the run's per-epoch inventory builder.
For each record, oldest first, it restores the anchor, rebases, and requires
the epoch's inventory to match the receipt source and the successor's
inventory to match the receipt target; a mismatch raises
`StaleRuntimeCheckpointError` with the changed roles. The re-derived
`support_epoch_relation` must reproduce the committed receipt bitwise.
`resume_support_epochs` returns the current `SupportEpochChain` (`evolution`,
`inventory`, `epoch`), whose checkpoint resumes through the identity relation.
A held refusal is visible as `store.terminal_record` after `resume`.
When the committed terminal is the `step-rejected` refusal, the process was
killed before the rebase committed. The driver then rebases rather than
repeating the refused step.

A moving surface with `archive="acknowledged"` binds
`moving_archive_policy(plan)`: each scheduled checkpoint commits the state whose
archive cursor covers its newest lifetime, so the live-history ring is the hard
window. A plan whose checkpoint interval exceeds `history_capacity` is refused
when it is built. A requested memory sample (`PhaseMemorySampler`) reaches
`ProductionRunResult.memory` and the terminal record as evidence; it is never
used as a ceiling.

`examples/meshfree_production_restart.py` is a killable CLI run with a deliberately
slow archive behind a byte-bounded publisher:

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_production_restart --root /tmp/run
# kill it at any point, then
JAX_ENABLE_X64=1 python -m examples.meshfree_production_restart --root /tmp/run --resume
```

The resumed run reports the same state, moment, and evidence digests as an
uninterrupted run. The RNG state is carried and its addressing is part of the
identity; current meshfree participants draw no random numbers inside a step.

`examples/meshfree_production_epochs.py` drives an ALE drift through repeated
support exhaustion. Each refusal is rebased at the held state and continues in
the next epoch. `--resume` re-derives the current epoch with
`resume_support_epochs` and ends with the same digests as an uninterrupted run,
even when killed inside a rebased epoch:

```sh
JAX_ENABLE_X64=1 python -m examples.meshfree_production_epochs --root /tmp/epochs
JAX_ENABLE_X64=1 python -m examples.meshfree_production_epochs --root /tmp/epochs --resume
```

## Qualification and performance evidence

`python -m tools.meshfree_qualification` selects Q1–Q17 and their explicitly
declared expected-refusal campaigns. Capacity, dimension, seeds, precision,
geometry authority, temporal method, and resource limits identify each workload.
Use `--help` to select a campaign with its own declared envelope; a capacity
refusal does not pass a workload that requires a successful solve.

The `benchmarks.meshfree_*` drivers and `benchmarks.meshfree_closure` separate
preparation, lowering, compilation, first/warm actions, solves, refresh,
sensitivities, epoch transactions, communication, and restart where those phases
exist. Wall time, CPU time, compilation counts, and sampled host/device memory
are distinct evidence. Compiler temporary/output/code estimates, allocator
reservation, sampled RSS, and visible retained arrays are not interchangeable
physical peak-memory bounds. Unavailable measurements stay unavailable.
Before/after comparisons require an actual compatible baseline.

Effective neighbor/chunk sweeps are part of workload identity. Source identity
includes transitive first-party benchmark helpers and independent example
oracles. Forced-device qualification binds the executing child runtime's
backend, topology, and precision rather than the orchestrator's environment.
Failed repeats retain completed phase/compiler/memory observations, reservations,
and their full originating failures. Post-execution resource refusals remain
distinct from pre-admission refusal; neither becomes a measured pass.
Adaptive uniform-reference extensions respect the declared `max_points` limit.
Previously retained campaign records remain bound to their recorded source
identities. Audit changes do not renew or relabel those historical measurements;
updated release claims require replay at the new scientific source identity.

The phase recorder passes callable numerical leaves as compiler arguments.
JVP/VJP preparation reuses the traced primal rather than tracing the entire
prepared workflow again. The moving benchmark evaluates concentration on the
returned measures and retains each temporal method's native success, conservation,
and GCL evidence. Its 48/64-point records include forward/backward Euler,
ARS-222, and ARS-443; the 25/49-point exterior records retain independent host
edge-ledger action and conservation checks.

Numerical implementation, qualification, resource admission, scientific validation,
and production release are separate. The workflows in this guide include bulk
transient and incompressible problems, particle interoperability, physical
surface events, open/patchwise-sharp geometry, intrinsic vector/tensor problems,
higher forms, and material/mixed-method interfaces. Each has its own scientific
identity, admission conditions, and numerical evidence; the presence of an API
does not qualify every cloud, boundary, provider, precision, or capacity.

`DistributedMeshfreeOperator` parity on forced CPU devices does not establish
GPU or multi-host performance. Hardware-unavailable rows remain blocked.
Geometry-authorized higher forms and abstract clique research remain distinct.
Frozen remaps admit value derivatives; fixed-support geometry and fixed-active
KKT sensitivities require their declared regularity. A general topology or
active-set change has no default smooth derivative.

`meshfree_support` and `meshfree_candidate_profiles` declare exact support tuples
for each implemented capability. Campaigns retain criterion/start/observation/
evidence identities and coverage, including failed, unexecuted, and blocked
rows. Independent-reference, reviewer, and release-authority requirements
cannot be supplied by an automated campaign; meshfree profiles remain
unreleased candidates.
Use separate workload selections when campaigns have different capacity envelopes:

```sh
python -m tools.meshfree_qualification --campaigns Q1 Q1-refusal --dimension 2 --neighbors 8 --output benchmarks/meshfree_qualification_strong_form_2d.json
python -m tools.meshfree_qualification --campaigns Q2 --elliptic-forms collocated dissipative --output benchmarks/meshfree_qualification_elliptic.json
python -m tools.meshfree_qualification --campaigns Q4 Q4-refusal --sizes 25 49 --output benchmarks/meshfree_qualification_exterior.json
python -m tools.meshfree_qualification --campaigns Q3 --sizes 64 128 --dimension 2 --output benchmarks/meshfree_qualification_multilevel.json
python -m tools.meshfree_qualification --campaigns Q5 --sizes 256 512 --dimension 3 --seeds 4 --output benchmarks/meshfree_qualification_surface.json
python -m tools.meshfree_qualification --campaigns Q6 --output benchmarks/meshfree_qualification_moving_surface.json
python -m tools.meshfree_qualification --campaigns Q7 --output benchmarks/meshfree_qualification_bulk_surface.json
python -m tools.meshfree_qualification --campaigns Q8 --sizes 12 24 --dimension 2 --neighbors 12 --output benchmarks/meshfree_qualification_learned.json
```

The learned-flux campaign has separate accepted 256/512-point targets. Signed
exact metrics use the declared transformed minimum-norm route and its native
preconditioner rather than the former mandatory Schur symbolic factorization.
Constraint, stationarity, resource, rank, and derivative evidence remain
separate. Existing benchmark records must be interpreted at their recorded
source/provider/precision identity; a result preceding a numerical cutover does
not qualify the changed implementation. No dense fallback or automatic budget
increase is implied.

### Local replay boundaries

The current native tensor-SBP elliptic replay, moving-surface replay, joint
signed quadratic transfer replay, and 2-D adaptive/restart replay have retained
accepted results. These are specific numerical candidates, not release approval.
Historical failures remain in their original records.

The block-ghost fluid–structure replay passes physical-channel equation and
interface-work gates. Its manufactured balance has a near-cancellation at
resolution 8; the coarse 8/11/16/21 campaign fails the fitted energy-order gate.
The unchanged problem's 24-to-32 refinement measures energy order 1.88.
`Q11-fluid-structure-energy` separately selects resolutions 16/21/24/32
(578/968/1250/2178 total cloud points), under the same declared resource limits
and unchanged order threshold. It does not relabel the coarse campaign.
The retained asymptotic replay passes all four rows and measures field order
2.20 and energy order 1.82 against measured fill distance; both exceed the
unchanged 1.5 threshold.

The 3-D adaptive replay improves field accuracy at equal point count, but its
measured equal-work gain is 0.276 and fails the required gain of at least one.
That replay does not qualify adaptive computational efficiency. Its restart
and learned-law scenarios pass independently.

The 1024-point strict-active derivative certificate exceeds its declared
256 MiB envelope; that row remains resource-blocked. Forced-CPU distributed
parity does not fill the separate GPU/multi-host hardware gates. Independent
physical references, review, and release authority remain required.

### Elliptic campaign policy and numerical limits

The public Poisson example uses an explicit uniform PHS-RBF-FD degree-three
policy for Dirichlet, Neumann, and Robin cases. Native default support is twice
the complete polynomial basis size: 20 neighbors in two dimensions and 40 in
three. Q2's `--elliptic-stencil`, `--elliptic-degree`, and
`--elliptic-neighbors` controls are distinct from Q1/Q3's generic bulk controls;
records and pre-execution reservations use the actual selected policy.

Refinement uses the independent nonpolynomial field
`u=exp(sum(x)/d)`, `k=2+0.1 sum(x)`,
`f=-u*(k/d+0.1)`. The quadratic example remains an exactness smoke, not an
approximation-rate oracle. Neumann comparisons disclose the native constant
source projection and use the continuum solution shifted at the actual gauge.
Original incompatibility is not relabeled as compatible.

Lifted Dirichlet equations are solved to the unchanged original physical
residual tolerance; a large lifted RHS cannot weaken that criterion.
Successful local stencil admission does not prove global collocation stability.
A larger Neumann case can exhaust native conditioning/work budgets even when an
independent host sparse reference confirms discretization convergence. Such
references are diagnostics, never automatic runtime fallbacks.

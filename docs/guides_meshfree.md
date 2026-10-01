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

`LocalStencilPolicy` selects `gmls` or `phs-rbf-fd`, polynomial degree, weighting,
conditioning, amplification, and chunk capacity. `MeshfreeFunctional` declares a
linear combination of derivative multi-indices, including mixed derivatives and
row-dependent normal derivatives. `prepare_local_stencils` returns weights and
per-row `LocalStencilEvidence`; `LocalStencilReport` summarizes the admitted fit.
Rank deficiency and a full-rank but ill-conditioned row have different statuses.
The declared condition limit is enforced, not merely fingerprinted.

Preparation uses native batched SVD/minimum-norm or local saddle solves. Runtime
application uses `MeshfreeOperator` and native sparse routes. Coordinate transpose
and Hilbert adjoint differ when source and target measures differ. Polynomial
reproduction does not establish conservation, PDE stability, or a maximum
principle.

## Strong-form point clouds

`PointCloudPlan(..., stencil=LocalStencilPolicy(...), neighbors=...)` binds points,
positive quadrature, boundary data, and local approximation. Prepared derivatives
retain arbitrary trailing payload axes. Mixed partials use an explicit derivative
multi-index. `divergence(..., dual=True)` is refused: a nodal component gradient is
not an exterior one-cochain, and no dual calculus is inferred from its shape.

Point-cloud observations use `prepare_point_cloud_field_reconstruction` with a
bounded BVH neighborhood and partial coverage. The default `polynomial`
reconstruction retains rank/condition evidence. Explicit `reconstruction="shepard"`
selects a positive, normalized, degree-zero reconstruction. It is a different
accuracy contract, not a fallback after a failed polynomial fit. Complete query
coverage refuses invalid rows; masked coverage retains their statuses. Point
clouds publish no facet trace.

## Elliptic problems

`PointDiffusionOperator` exposes two distinct forms:

- `collocated`: the continuum product-rule approximation of `div(k grad(u))`.
- `dissipative`: the quadrature-adjoint action `-M^-1 sum(D^T M k D)`. Its weighted
  energy is nonpositive by construction. This does **not** establish continuum
  consistency for arbitrary quadrature or an arbitrary cloud.

`PointCloudPoissonPlan(cloud, PointBoundaryPlan(...), ...)` prepares sparse
assembly and a reusable native solve. `.prepare(diffusivity)` returns
`PreparedPointCloudPoisson`; `.solve(source, boundary_values=...)` returns the
values, native linear diagnostics, physical residual, boundary residual, and
compatibility evidence. Numeric coefficient refresh retains symbolic assembly
and the selected hierarchy.

Dirichlet data are lifted before solving the homogeneous constrained equation.
Neumann problems use an explicit interior gauge. Compatibility is checked against
all original equations, including the equation replaced by the gauge; a
quadrature sum alone is not assumed to be the algebraic left-nullspace condition.
`compatibility="refuse"` rejects incompatible data. `compatibility="project"`
computes a constant interior source correction with a native response solve and
reports both the correction and its solve diagnostics.

`point_sbp_report` checks the complete sparse coefficient identity, including the
boundary flux term. It has an explicit preparation budget. A finite polynomial
probe is not labeled a full SBP certificate.

```sh
JAX_ENABLE_X64=1 python examples/point_cloud_poisson.py --size 64 --dimension 2
```

The example differentiates an independent manufactured variable-coefficient
solution to obtain its source; it does not manufacture a source by applying the
operator being tested.

## Stabilization

`HyperviscosityPlan(coefficient, order=...)` prepares
`-coefficient * (-L)^order` using repeated sparse dissipative Laplacian actions.
No matrix power is materialized. The power-iteration spectral quantities in
`HyperviscosityEvidence` are estimates, not bounds or unconditional time-step
stability certificates. The coefficient is explicit; no failed advection run is
silently repaired.

## Multilevel preparation

`MeshfreeHierarchyPlan` builds stable-ID subset levels and polynomially augmented
PHS interpolation. Each level reports constant/linear reproduction defects and
route storage. Coarsening stops when it cannot genuinely reduce the cloud,
including boundary-only cases. Restriction is the coordinate transpose required
by the stiffness-coordinate Galerkin equation, not an implicitly substituted
mass adjoint.

`meshfree_multigrid_builder` supplies these transfers to the native
`GalerkinHierarchyBuilder`. Numeric refresh reuses transfers. Sparse forward
Gauss–Seidel is the default smoother. Nonsymmetric collocation can be strongly
nonnormal: no universal V-cycle contraction or superiority over ILU is claimed.
A singular coarse problem requires an explicitly declared native nullspace and a
bounded projected pseudoinverse preconditioner; undeclared kernels are refused.

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
edges, equation masks, and degree-one moment constraints. Supplied quadrature is
preferred. Density-derived volumes require an explicit total domain measure;
Dirichlet rows are excluded from equations, never assigned zero mass.

`MeshfreeMetricPolicy` chooses signed or nonnegative coefficients and exact or
explicitly relaxed moments. Preparation selects a tolerance-defined sparse row
profile through native `prepare_sparse_row_rank`, solves the selected sparse
Schur/conic problem, and audits **every** original moment equation. The pivot
profile is not labeled an exact algebraic-rank or singular-value certificate.
Fill, input nonzeros, and elimination work have separate refusal budgets.

`MeshfreeMetricResult` retains provider status, original moment residuals,
amplification, sign counts, compatibility, and derivative availability. Relaxed
metrics retain their slack; conservation does not erase the consistency lost.
Nonnegative infeasibility is a failure, not a reason to clip weights.

Strictly positive admitted weights bind native `CochainDiscretization`, Hilbert
spaces, and `CochainComplexIR`. Signed weights remain constitutive stiffness
coefficients over canonical incidence; they are not negative Hodge weights.
Zero or signed metrics cannot be converted to a positive Hilbert complex.
Reduced coercivity comes from native factorization evidence. Unknown inertia is
reported as unavailable rather than inferred from finite coefficients.

Diffusion uses explicit harmonic, arithmetic, or supplied edge coefficients.
Natural boundary loads require declared boundary quadrature.
`MeshfreeAdvection` and `edge_upwind_content` reuse the prepared native sparse
incidence for donor values, transpose accumulation, and outgoing CFL. Fluxes
are integrated volume fluxes: the metric is not multiplied a second time.
Positivity requires an admitted nonnegative metric, nonnegative states/sources,
and the outgoing CFL bound, not merely conservation.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_conservative_diffusion.py --size 25
```

## Intrinsic surfaces

`SurfacePointCloudPlan` binds a declared smooth closed source, positive measures,
bounded support, and tangent-chart GMLS/PHS approximation.
`ImplicitSurfaceGeometry` uses the native geometric or regular-level-set source;
normal projection belongs to `RegularLevelSetManifold.project_normal`, not a
second meshfree Newton implementation. `SampledSurfaceGeometry` fits oriented
local height charts and reports conditioning, fit, and support failures.

Sampled normals, projector comparisons, and sheet-separation diagnostics are
estimates. They do not prove global reach, a closest point, or topology safety.
An implicit tube declaration is an explicit source-domain premise, not a radius
manufactured from reciprocal curvature. Sharp or open sampled surfaces and
unsupported neighborhoods are refused.

`surface_gradient` and `laplace_beltrami` are strong intrinsic approximation
operators. The paired surface divergence supplies a weighted Green identity;
it is distinguished from pointwise ambient tangential divergence. Geometry
accuracy can limit PDE accuracy even when local polynomial reproduction passes.
Tangent-Voronoi quadrature is positive but approximate. Area-normalized density
requires an explicit area and does not independently recover that area.

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
```

The combined sphere/torus workflow requires at least 256 points. Its torus chart
uses eight neighbors and an explicit candidate budget; its sphere uses a
different declared support. Smaller requests are refused rather than silently
resized or admitted by loosening geometric thresholds.

## Motion, shifting, and topology epochs

`MovingSurfaceState` stores extensive content, measures, geometry, exact
`TopologyEpoch`, and complete histories. Concentration is recovered as content
divided by measure. `MovingSurfacePlan` composes native `ConservationIMEXMethod`
with implicit diffusion, explicit reactions, and transactional commit/rollback.
Its statuses retain geometry, trust, tube, implicit-solve, conservation, and
history-capacity failures. A rejected candidate leaves the accepted state intact.

Tangential shifting uses material-minus-mesh relative velocity. The shared
native-incidence upwind kernel conserves content; positivity remains conditional
on metric and CFL evidence. `SurfaceResamplingPolicy` performs bounded host
insertion/removal/projection/relaxation with witnessed spacing and quality
triggers. Nonconvergence and capacity refusal remain visible.

`SurfaceTransferPlan` prepares sparse high-order correction or nonnegative
column allocation. It checks weighted column conservation, actual row-constant
reproduction, and actual coefficient signs separately. Equal total old/new area
does not prove constant preservation. Uncovered old columns are refused.
The coordinate dual is the transpose; the Hilbert adjoint includes both measures.

Epoch transitions remap concentration and then reconstruct extensive content.
Every history has its own source/target measures and transfer. Failed history
transfer refuses the entire transaction; epoch geometry derivatives are
unavailable. `MeshfreeCapacityPolicy` records storage buckets, while native
positive spaces contain compact active rows only, never zero-mass dummy nodes.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_moving_surface_reaction_diffusion.py --size 48
```

The example uses a positive native graph diffusion and explicitly does not claim
high-order Laplace–Beltrami accuracy for that graph. It exercises growth,
reaction, shifting, two topology changes, complete-history transfer, continued
integration, and rollback.

## Bulk–surface exchange {#bulk-surface-exchange}

`MeshfreeComponent` publishes native residual, reconstruction, and capacity
contracts to `phydrax.solver.coupling`; it does not advertise a facet trace.
`SurfaceExchangeLaw` uses complete constant-reproducing bulk queries and their
exact coordinate transpose to pair bulk loss with surface gain.
`LangmuirAdsorptionFlux` reuses the native adsorption kinetics.

Signed polynomial query weights can conserve through the law, but cannot claim
a positive native amount partition. `MeshfreeBulkSurfaceMethod` therefore
requires an explicitly positive deposition route, supplied by a canonical
Shepard reconstruction. Source program and numeric revision admission rejects
stale same-shaped reconstruction.

Moving exchange sites are relocated at host coupling-window boundaries. Within
a window they are frozen; lag and displacement are evidence, and within-window
geometry differentiation is not claimed. The coupled example reports any metric
slack separately from its exact amount balance.

```sh
JAX_ENABLE_X64=1 python examples/meshfree_bulk_surface_exchange.py --size 128 --dimension 3
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

`prepare_meshfree_conservation_solve` binds the native nonlinear and sparse
derivative owners. Array parameters are separated from immutable law metadata;
changed static traits require re-preparation. MODEL-authorized parameter bindings
and `SolverObjective` carry implicit gradients through the converged solve.
Primal, adjoint, coverage, conservation, and coercivity evidence reach the result.
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

## Qualification and performance evidence

`python -m tools.meshfree_qualification` selects Q1–Q8, with capacities, dimension,
seeds, precision, support, and memory limits explicit. Combined campaign defaults
are ambient dimension three and capacities 256/512. Surface campaigns refuse
dimension two rather than relabeling the requested geometry.

The four `benchmarks.meshfree_*` drivers separate preparation, lowering,
compilation, first/warm actions, solves, refresh, and epochs where those phases
actually exist. Compiler temporary/output/code estimates and visible retained
arrays are not physical peak memory. Unmeasured phases and closure-captured
memory are reported unavailable, not fabricated. Before/after comparisons require
an actual compatible baseline.

Numerical gates, scaling evidence, resource refusal, and scientific release are
different. No full-envelope scaling or GPU performance claim follows from a
small CPU smoke. Distributed execution, bulk SPH/incompressible flow, surface
topology change, sharp/open surfaces, vector/tensor surface PDEs, higher exterior
degrees, collocated material-interface conditions, and derivatives across epochs
or changing conic active sets are not claimed.

Use separate workload selections when campaigns have different capacity envelopes:

```sh
python -m tools.meshfree_qualification --campaigns Q1 Q2 Q4 --sizes 64 128 --dimension 2 --output benchmarks/meshfree_qualification_bulk.json
python -m tools.meshfree_qualification --campaigns Q3 --sizes 64 128 --dimension 2 --output benchmarks/meshfree_qualification_multilevel.json
python -m tools.meshfree_qualification --campaigns Q5 Q6 Q7 --sizes 256 512 --dimension 3 --seeds 4 --output benchmarks/meshfree_qualification_surface.json
python -m tools.meshfree_qualification --campaigns Q8 --sizes 12 24 --dimension 2 --neighbors 12 --output benchmarks/meshfree_qualification_learned.json
```

The recorded 12/24-point learning campaign passes its primal, adjoint, gradient,
coverage-refusal, and conservation gates. It does not qualify the combined default
256/512-point envelope: the signed metric's native sparse symbolic factorization
refuses work beyond its declared two-million-operation budget. No dense fallback
or automatic budget increase is used.

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

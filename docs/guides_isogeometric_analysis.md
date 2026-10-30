# Isogeometric analysis

## Monolithic contact bindings

A compiled IGA weak form supplies mechanics residual/JVP and candidate history;
an existing `PreparedIGASplinePatchProxy` supplies contact samples through
`prepare_iga_contact_participant(proxy, source_space)`. The latter is a geometry
adapter, not a private IGA solver. Bind it to the appropriate mechanics block in
`MonolithicContactPlan`; a fixed geometry uses an explicit fixed state.
The shared pullback retains cross blocks and geometry curvature.

The adapter binds a conservative full-path enclosure provider from the existing
proxy plan and source space. Geometry declaring `NONLINEAR_TRAJECTORY` is admitted
only with that provider, participant-domain limits, and a complete fixed search
envelope. CCD subdivides whole-interval enclosures, and the root bounds motion
within the envelope; missing providers refuse instead of using endpoint distances.
A proxy's static error/coverage certificate alone does not certify arbitrary
motion. The stronger rounding-certified AABB CCD route refuses these nonlinear
enclosures, and nonlinear friction uses residual rather than potential merit.
Contact integration does not enlarge the IGA release surface below. See
[Monolithic implicit contact](guides_monolithic_contact.md).


This page documents the legacy S1 release surface. The implementation also
contains broader 1D/2D/3D, periodic, multipatch, trimmed, hierarchical, T-spline,
and compatible-space candidate routes. Their exact admitted tuples and nonclaims
are recorded in the [IGA support matrix](guides_isogeometric_analysis_support.md)
and the generated [capability inventory](api/capabilities.md); implementation
availability does not expand the S1 release surface.

Phydrax S1 isogeometric analysis is a narrow, fail-closed finite-element path.
It covers one regular, untrimmed, full-dimensional 2D NURBS patch and one
exactly isoparametric scalar H1 field. The topology is a pair of fixed,
nonperiodic, clamped B-spline grids with one common degree. Numerical execution
uses an explicit isotropic Gauss rule and the existing finite-element weak-form
compiler with a matrix-free, sum-factorized local kernel.

## A single-patch plan

`BSplineGrid` is the canonical fixed grid also used by spline interpolation; the
IGA namespace directly re-exports that class rather than defining a second knot
convention. Knots are static topology. Control points and strictly positive
weights are runtime values, and `NURBSGeometryState` retains the authored
strictly-positive bank without imposing a projective gauge normalization. A
common positive scaling leaves the rational map unchanged; its weight-space null
direction remains visible to design derivatives.

```python
import jax.numpy as jnp
import phydrax as phx

p = 2
grid = phx.discretization.iga.BSplineGrid.open_uniform(
    p, 4, interval=(0.0, 1.0)
)
s = grid.greville_abscissae
xx, yy = jnp.meshgrid(s, s, indexing="ij")
geometry = phx.discretization.iga.NURBSGeometryState(
    jnp.stack((xx, yy), axis=-1),
    jnp.ones((grid.coefficient_count, grid.coefficient_count)),
)
plan = phx.discretization.iga.IsogeometricPlan.isoparametric(
    (grid, grid),
    geometry,
    field_name="u",
    axis_names=("xi", "eta"),
    quadrature_policy=phx.discretization.iga.IsogeometricQuadraturePolicy(p + 1),
    qualification_policy=phx.discretization.iga.IsogeometricH1QualificationPolicy(),
)
discretization = plan.prepare(numeric_version="geometry-0")
constraint = discretization.homogeneous_trace_constraint("u")
```

The `numeric_version` labels numeric geometry state; it does not change the
frozen knot topology or coefficient layout. For design calculations,
`prepare_runtime(new_geometry, numeric_version=...)` refreshes control points and
weights within that layout. Pass the returned runtime through
`FiniteElementExecutionContext`. Changing knots, degree, control shape, or field
layout requires a new plan and compilation.

## Weak forms and boundaries

The prepared object satisfies the finite-element compiler contract. Build an
existing `FiniteElementForm`, then call
`compile_finite_element_problem(..., constraint=constraint,
execution_policy=FiniteElementExecutionPolicy(realization="matrix_free",
local_kernel="sum_factorized"))`. The end-to-end
`examples/iga_single_patch_poisson.py` program solves a manufactured
homogeneous-Dirichlet diffusion problem.

S1 provides the homogeneous trace constraint above. With no boundary load, the
usual omitted weak boundary term is natural zero Neumann data. Nonzero strong
Dirichlet data, Nitsche variants, periodic identification, and patch-interface
conditions are outside this release surface.

## What geometry evidence means

Preparation samples the realized map at every declared quadrature site and
records scale-normalized minimum rational-denominator, Jacobian-rank, and
orientation margins in `IsogeometricGeometryEvidence`. The qualification policy
checks those margins. Runtime geometry refresh computes new evidence; evidence
from an older runtime is not transferable.

This is **sampled map evidence**, not a proof of global injectivity or absence of
folds between samples. The qualification producer also constructs an analytically
exact rational quarter-annulus and reports its sampled margins. “Exact” there
means that the chosen homogeneous spline polynomials represent the stated map;
it does not promote the sampled regularity checks to a continuous certificate.

An IGA patch here is a full-dimensional map from a 2D parameter square to a 2D
physical region. It is not a BRep surface embedded in 3D. The BRep geometry
subsystem owns surface topology, trimming, CAD faces, and boundary atlases; none
of those capabilities are implied by `NURBSGeometryState`.

## Qualification and timing artifacts

Run `python tools/iga_h1_qualification.py` to generate
`benchmarks/iga_h1_qualification.json`. The producer uses exactly four refinement
levels for both an affine square and an exact rational quarter-annulus. Error
norms are evaluated by a tool-independent Gauss implementation with two more
points per axis than the solve rule. The frozen policy requires

- H1 rate at least p − 0.25 and L2 rate at least p + 0.75;
- normalized free residual no larger than
  `max(10 * solver_rtol, 1024 * eps)`;
- matrix-free/private assembled parity no larger than
  `max(0.01 * solver_rtol, 1024 * eps)`;
- tangent/adjoint duality no larger than
  `max(0.01 * solver_rtol, 4096 * eps)`;
- Taylor slopes in [1.8, 2.2] on at least three non-roundoff intervals; and
- solve-rule versus q+2 defect no larger than 0.1 times the larger of the
  independently evaluated discretization error and normalized solve bound.

The rational case additionally checks exact physical-linear reproduction against
the larger residual/duality bound. Its nonrepresentable manufactured solution is
what supplies the four-level rates. The policy ID includes all thresholds,
including explicit geometry-margin overrides; default denominator, orientation,
and reciprocal-condition margins resolve to `sqrt(eps)` for the realized dtype.
The tool prints deterministic, sorted JSON, writes atomically, and exits nonzero
when any policy gate fails.

The private qualification parity path realizes coordinate entries from the same
public matrix-free action and applies them to a fixed probe. It checks execution
and scatter consistency; it is not an independent numerical oracle or a public
IGA sparse backend. `python tools/iga_h1_benchmarks.py` writes
`benchmarks/iga_h1.json` for degrees 2, 3, and 4 over fixed span counts using the
shared synchronized timing helpers. It is record-only: solver status and
residual are recorded, but elapsed time has no pass threshold.

## Prepared field queries and side traces

`prepare_isogeometric_field_reconstruction(discretization, "u", ...)` returns a
`PreparedFieldReconstruction` that evaluates a field at physical points. Each
point is mapped back to the parameter box by a fixed number of clamped Newton
steps on the runtime NURBS map, seeded from samples of every
integration-overlay cell; this inverse map runs in JAX, so a prepared query
locates once and `apply`/`transpose` reuse the route for any coefficients.
Points whose closest parameter lies on the patch box with a nonzero residual are
`OUTSIDE_SUPPORT`; unconverged interior iterations are `LOCATION_FAILED`.
Values and first physical derivatives are exact: physical-linear fields are
reproduced exactly on the rational quarter annulus, and tensor polynomials of
the field degree on affine patches. On a knot line where the requested order
exceeds the field or geometry continuity, a trace side is required
(`side="owner", cell_ids=...` names the C-ordered overlay cell). Coefficients
use the public tensor layout `control_shape + component_shape`.

The support region of an axis-aligned affine patch is derived. A curved patch
needs an explicit `support_geometry`: its boundary must contain the patch
boundary, its interior must contain the patch interior, and a declared interior
measure must equal the patch measure.

```python
region = (
    (phx.geometry.Ball((0.0, 0.0), 2.0) - phx.geometry.Ball((0.0, 0.0), 1.0))
    & phx.geometry.Orthotope((1.0, 1.0), (2.0, 2.0))
).compile()
reconstruction = phx.discretization.iga.prepare_isogeometric_field_reconstruction(
    annulus, "u", support_geometry=region
)
slopes = reconstruction.prepare_query(points, derivative=(1, 0))
```

`discretization.prepare_side_trace(field, domain, rule=FacetTraceRule(...))`
publishes the exact trace on patch-boundary faces (`exterior_facet`) or on the
interior knot faces between overlay cells
(`integration_domain("interior_facet")`). Sites are the owner cell's facet rule
points, shared by the `"owner"` and `"neighbor"` sides; weights are the rule
weights times the physical surface Jacobian; normals point out of the traced
cell. `"normal"` and `"tangential"` traces contract vector fields with the
outward normal. The trace acts on the compiled field's public control layout
`(*control_shape, *component_shape)`, the same coefficients the finite-element
compiler solves for, so its dual pullbacks, residual-reaction fluxes, and coupling
laws act on that field directly. `support_rows` are the C-order flattened control
indices (`trace.row_shape` is the control shape; `trace.flatten_rows` and
`local_field_binding(name).flatten` apply the same exact map). Trace degrees are
reported only for polynomial (uniform-weight) fields, and for normal/tangential
traces only on affine patches. Local variational regions remain cell and
exterior-facet only.

## Explicit S1 boundary

S1 does not support trimming, holes described by trim curves, multipatch coupling,
multipatch interfaces (interior knot faces of one patch are traced, not coupled),
extraordinary points, hierarchical or locally refined splines,
trainable knots, anisotropic degree, periodic knots, 3D volume maps, embedded
shell/surface mechanics, vector or mixed fields, H(div)/H(curl) mappings, contact,
or public sparse assembly. It does not infer quadrature order. A positive weight
vector alone does not certify a valid map; preparation must also pass sampled
rank and orientation checks.

## Rectilinear ASTS2D T-spline surfaces

The T-spline candidate path shares one source-parametric carrier between native CAD
and scalar surface H1 IGA. Its admitted profile is a rectilinear bicubic
analysis-suitable T-mesh with simple interior knot intervals, topology-derived local
knot windows, open boundaries, and no extraordinary vertices. Arbitrary T-meshes,
repeated interior knots, shell mechanics, and vendor-specific T-spline variants are
not implied.

`TMesh2D` owns exact parameter topology, persistent anchor IDs, and T-junction
extensions. `TSplineSurfacePatch` owns the raw control-point/positive-weight leaves
and the exact source bank; the same patch can be supplied directly to
`TSplineGeometryState` and `TSplineIsogeometricPlan`. Exact rational coefficients
such as `1/3` and nonzero small coefficients remain in the source/prolongation data;
the fixed-route JAX realization is binary64 and carries its coefficient and
homogeneous-realization error bounds. Geometry prolongation applies the nested
operator to homogeneous `(wX, w)` controls. A rational field with independent weights
uses `q' = P(aq) / P(a)`, not polynomial coefficient transfer.

An anchor's topological parameter is not a control-point sample. For polynomial
reproduction, derive control values from each basis's Greville parameter rather
than treating `anchor.parameter` as an interpolation node.

```python
from fractions import Fraction

import numpy as np
import phydrax as phx
from phydrax.geometry.brep import TSplineSurfacePatch

iga = phx.discretization.iga
breaks = tuple(Fraction(index, 4) for index in range(5))
mesh = iga.TMesh2D.tensor_product(breaks, breaks)
greville = tuple(anchor.local_knots.greville_parameter for anchor in mesh.anchors)
exact_controls = tuple((u, v, u * v) for u, v in greville)
surface = TSplineSurfacePatch(
    mesh,
    np.asarray([[float(value) for value in row] for row in exact_controls]),
    np.ones((mesh.coefficient_count,)),
    exact_control_points=exact_controls,
    exact_weights=tuple(Fraction(1) for _ in mesh.anchors),
)
geometry = iga.TSplineGeometryState(surface)
trim = iga.UVTrimmedSurface(
    iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
)
plan = iga.TSplineIsogeometricPlan(
    geometry,
    iga.TSplineH1FieldSpec(
        "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
    ),
    trim=trim,
    quadrature_policy=iga.IsogeometricQuadraturePolicy(6),
)
refinement = plan.refine((mesh.cells[5].cell_id,))
prepared = refinement.target_plan.prepare(numeric_version="local-refinement-0")
```

`TMesh2D.certify()` proves the selected ASTS basis premises, exact Bernstein
partition/nonnegativity identities, and declared cross-cell trace continuity.
`TSplineGeometryCertificate` is a separate, numeric-bank-bound proof of positive
rational denominator, physical regularity, and global embedding. Thus a valid
parameter partition does not make a folded or collapsed physical map admissible.
CAD `is_c1_along` queries reuse the retained immutable basis certificate; they do
not claim physical-map regularity. Fixed-topology derivatives hold incidence, knots,
active trim/query routes, and the certified map branch fixed. Refinement is discrete
and creates new topology and source-to-target lineage.
The Scott local-refinement closure inserts only selected extension-graph skeleton
segments and verifies complete source-basis inclusion on shared local polynomial
pieces. It adds only coordinates from the finite source-and-marked skeleton and
resolves extension-graph conflicts. Its exact extraction partition is virtual:
active local knot lines subdivide control T-cells into Bernstein pieces without
changing the control-cell topology or removing T-junctions.

`source_authority_id` identifies the exact root mesh, control points, and raw
weights; `source_definition_id` also identifies the current target and its
refinement/transfer chain. They are not interchangeable. Explicit
`realize_parameter_bank(..., source_authority_id=...)` calls bind differentiable
numeric leaves to that root authority, including after refinement. A foreign
root or a concrete reauthored bank refuses; tracing does not bypass source
validation or detach control-point and weight derivatives.

`SurfaceSourceCharts` and descendant restrictions keep three identities distinct:
the mesh root cell ID, canonical source-root atlas ID, and T-spline source-piece
key `(patch, control_cell_id, extraction_piece_id)`. The control-cell ID names
the owning T-mesh cell; the extraction-piece ID selects a local Bernstein piece.
Patch resolution uses source entity key plus occurrence path, never row order.
`SurfaceChartWitness` also retains an explicit per-row `SurfaceRootMeshCellBinding`
(`root_geometry_id`, `root_topology_id`, `root_cell_global_id`) when the chart has
an actual root `CellMesh`; analytic source charts without a root mesh retain `None`.
Restriction proofs require and validate the binding against the root owner rather
than inferring root-cell identity from atlas IDs, row order, or piece keys.
The source-root atlas binds exact spline control coordinates and positive weights
to the rounded root mesh; exact `Fraction`/`RadicalElement` source leaves are not
re-authored from their binary64 execution shadows. `SplineCellGeometryElement`
retains exact weights alongside the numeric tabulation bank. B-spline source
control IDs are required; missing identities are refused rather than inferred from
coefficient positions. Geometry prolongation acts on homogeneous coordinates and
weights together, and rational fields use their own weight bank. CAD and IGA consume
the same raw authored weights and retained source-to-target chain.

The direct curved-surface manufactured Poisson solve is runnable as
`python -m examples.iga_tspline_surface_poisson`. The end-to-end CAD-to-mesh-to-IGA
example in `examples/native_tspline_cad_analysis.py` plans and executes accepted
native parametric-surface meshing through `NativeMeshingProvider`, independently
renews the source-chart witness/fidelity proof, then solves a T-spline Poisson
problem:
`python -m examples.native_tspline_cad_analysis`.
The closure qualification adds a positive T-junction/locality case with exact-source
and numeric map certificates: `python tools/iga_closure_qualification.py`. Actual
closure and preparation timings are record-only:
`python tools/iga_tspline_benchmarks.py --smoke` or
`python tools/iga_tspline_benchmarks.py`.

This path is outside the separately allow-listed `iga.tensor` R1 release profile;
availability and these candidate qualification tools do not publish a release tuple.

## Public compatible spline complexes

`SplineDeRhamComplex(grids, periodic=..., geometry=..., geometry_id=...,
twist="untwisted", quadrature_degree=..., hodge_policy=...)` is the public
exterior realization, separate from the scalar `IsogeometricPlan` workflow.
It supplies exact sparse d and paired degree spaces through
`hilbert_complex(boundary="absolute" | "relative")`. Incidence routes are prepared
directly in canonical row-major order, including periodic cancellation. Identity
tensor geometry uses native Kronecker Gram blocks; mapped geometry uses
local-support quadrature gather/transpose actions and prepared local metric
solves, without a global dense basis table or Gram matrix. At fixed degree and
dimension, incidence and mapped pairing storage grow linearly with the DOFs.

`interpolant`, physical-component `reconstruction`, commuting `transfer` and
outward `trace(axis, side)` retain canonical form degree/twist/component ordering.
Trace and transfer are `ComplexMap` objects with numerical commutation evidence,
not private Piola or signed-trace helper carriers. The assembled public complex
validates exactness and relative closure before numerical execution.
API availability alone does not expand the released support matrix.

Mapped-complex `refresh_geometry(geometry)` reuses prepared basis/quadrature/moment
data and static binding ids while refreshing metric/face pairings on device.
For a dynamic identity chart, prepare it as a callable with explicit geometry_id;
the unmapped identity shortcut deliberately has no geometry refresh contract.
Geometry refresh always recomputes numerical Jacobians and metrics; it does not
cache arbitrary geometry callables. Realization, differential, and Gram IDs stay
unchanged. Prepared PyTree leaves now hold local-support routes and metric-weighted
basis coefficients rather than global dense basis/Gram arrays; recreate persistence templates
for this canonical representation.


See [IGA APIs](api/discretization/isogeometric.md#public-spline-de-rham-complex)
and [exterior complexes](api/exterior/complexes.md).

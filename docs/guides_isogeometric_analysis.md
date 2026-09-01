# Isogeometric analysis

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
weights are runtime values, and `NURBSGeometryState` fixes the otherwise
nonunique common weight scale by normalizing weights to mean one.

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

## Explicit S1 boundary

S1 does not support trimming, holes described by trim curves, multipatch coupling,
interfaces, extraordinary points, hierarchical or locally refined splines,
trainable knots, anisotropic degree, periodic knots, 3D volume maps, embedded
shell/surface mechanics, vector or mixed fields, H(div)/H(curl) mappings, contact,
or public sparse assembly. It does not infer quadrature order. A positive weight
vector alone does not certify a valid map; preparation must also pass sampled
rank and orientation checks.

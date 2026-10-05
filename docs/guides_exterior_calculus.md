# Exterior calculus: smooth fields, complexes and chains

The exterior substrate separates scientific form semantics from value carriers and
numerical realizations. A chart callable and a labeled `DomainFunction` retain
their existing owners; both share `FormType`, coefficient ordering and algebra.
A discrete realization supplies metric pairing and exact d through a
`HilbertComplex`. Topology remains combinatorial and materials remain constitutive.

## Smooth calculus: an executable example

Run this block in the installed worktree environment. The scalar component axis is
explicit, and d² vanishes independently of metric or orientation.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("exterior-plane", ("x", "y"))
scalar = phx.metrix.DifferentialForm(
    lambda x: jnp.asarray([x[0] ** 2 + x[1] ** 2]),
    chart=chart,
    degree=0,
    twist="untwisted",
)
alpha = phx.metrix.DifferentialForm(
    lambda x: jnp.asarray([-x[1], x[0]]),
    chart=chart,
    degree=1,
    twist="untwisted",
)
x = jnp.asarray([0.2, 0.3])
d_scalar = phx.metrix.exterior_derivative(scalar)
d_alpha = phx.metrix.exterior_derivative(alpha)
d2_scalar = phx.metrix.exterior_derivative(d_scalar)
assert jnp.allclose(d_scalar(x), jnp.asarray([0.4, 0.6]))
assert jnp.allclose(d_alpha(x), jnp.asarray([2.0]))
assert jnp.allclose(d2_scalar(x), jnp.asarray([0.0]))
print("smooth exterior calculus: analytic d and d² agree")
```

Coefficient layout is `(*batch, choose(ambient_dimension, degree), *fiber_shape)`.
Multi-indices are increasing lexicographic combinations. Matrix fibers use
`wedge(..., product="matrix")`; scalar graded commutativity is not claimed for
noncommuting matrix coefficients.

## Twist and physical proxies

```python
import jax.numpy as jnp
from phydrax.exterior import FormType, FormValueSpec, vector_to_form, form_to_vector

flux_spec = FormValueSpec(FormType(3, 2, twist="twisted"), proxy="flux")
v = jnp.asarray([1.0, 2.0, 3.0])
coefficients = vector_to_form(v, flux_spec)
assert jnp.allclose(coefficients, jnp.asarray([3.0, -2.0, 1.0]))
assert jnp.allclose(form_to_vector(coefficients, flux_spec), v)
```

Circulation is degree1 and covariant. Flux is degree n−1 and contravariant;
in 2-D it packs as `(−v_y, v_x)`. Density is top degree. Ambiguous degrees require
an explicit proxy; a flux vector and a circulation vector of the same shape are
not interchangeable. Piola uses signed determinant for untwisted and absolute
for twisted forms. Embedded pullbacks require the appropriate coorientation.

Mapping application packs value-only broadcast axes as multiple right-hand
sides, so one geometric matrix is factored once for all basis values and fibers.
Larger square mappings differentiate by reusing their native LU factors; they
retain native solve status without reserving an unrelated Krylov basis.

Smooth star is orientation-free and flips twist. In dimension n, degree k and
metric negative index q:

```text
⋆⋆ = (−1)^(k(n−k)+q)
δ = (−1)^(n(k+1)+q+1) ⋆ d ⋆
Δ = dδ + δd
```

Only `to_untwisted(form, orientation)` / `to_twisted(form, orientation)` choose an
orientation. δ at degree0 raises instead of returning an ambiguous zero-form.
`DomainDifferentialForm` follows the same rules with explicitly shaped
`DomainFunction` coefficients and keeps derivative mode/backend dependencies.

## Metric cochains and relative boundary

A cell realization binds canonical topology to one metric-only Hodge per degree.
The Hilbert complex owns vector spaces; `DiscreteForm` additionally binds values
to `realization_id` and `form_type`. Equal vector lengths never replace those ids.

The following self-contained interval complex exercises a sparse restricted
pairing and the physical sign of the codifferential. Interior vertex1 remains
active under the relative boundary; its stiffness action is exactly two.

```python
import jax.numpy as jnp
from phydrax.discretization import (
    CochainDiscretization, DiagonalHodge, SparseHodge, interval_cell_complex,
)

topology = interval_cell_complex(jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32), 3)
vertex_hodge = SparseHodge(
    jnp.arange(3, dtype=jnp.int32), jnp.arange(3, dtype=jnp.int32),
    jnp.ones(3, dtype=jnp.float64), 3,
)
realization = CochainDiscretization(
    topology, (vertex_hodge, DiagonalHodge(jnp.ones(2, dtype=jnp.float64))),
    boundary_masks=(
        jnp.asarray([True, False, True]),
        jnp.asarray([False, False]),
    ),
    numeric_revision="guide-interval-metric",
)
values = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
gradient = realization.exterior_derivative(0, values, boundary="relative")
laplacian = realization.codifferential(1, gradient, boundary="relative")
assert jnp.allclose(laplacian, jnp.asarray([0.0, 2.0, 0.0]))
assert realization.hilbert_complex(boundary="relative").space(0).size == 1
print("relative sparse pairing: active stiffness and boundary restriction agree")
```

Relative operators restrict incidence and the metric Gram to active coordinates,
invert that restricted Gram and zero-extend the output. They do not mask an
unrestricted inverse. Harmonic counts and spectra exclude inactive DOFs.
Physical divergence is −δ; complex adjoints conjugate.

`DiagonalHodge` and `SparseHodge(..., policy=LinearSolvePolicy(...))` are metric
owners, not material containers. Complex loss belongs to separate constitutive
coordinate forms. Device evidence and solve status must reach the application.
Use host `admit()` at admission, not inside a compiled iteration. Explicit stable
`numeric_revision` binding metadata allows numeric leaf refresh without rebuilding
static identities per step. Native PCG/Jacobi is the default; host direct factors
are eager-only and never run inside scan.

Sparse metric admission uses native reverse Cuthill–McKee ordering for its
Cholesky evidence. Factor nonzeros, bytes, and symbolic work remain bounded
by the declared native resource limits; ordering does not relax those guards.

## Choosing a realization

| Realization | Degree values / pairing |
|---|---|
| Cell cochains | Canonical oriented cell integrals, diagonal or sparse metric Gram |
| Finite elements | Canonical form DOFs, sparse metric Gram; top degree is density |
| Splines | Compatible spline component spaces and IGA-owned mapped pairing |
| Fourier | Compact admissible modes, Hermitian Parseval; explicit Nyquist policy |
| Sphere | Scalar and normalized poloidal/toroidal modes, Betti (1, 0, 1) |
| Meshfree complex | Cell integrals on an authorized simplicial complex; GMLS-reconstructed sparse metric Gram |

Structured bridge pack/unpack/proxy behavior stays canonical. GraphIR is a
lowering for diagonal Hodges, not a second calculus. MAC exposes its existing
face/cell Hilbert slice; SBP remains a separate collocated stack.

Cell cochain and finite-element `exterior_derivative`, `codifferential`, and
`hodge_laplacian` methods accept and return full degree coordinates, including
for relative boundaries.
Use `active_indices(degree, boundary=...)` to enter the compact coordinates of
`hilbert_complex(boundary=...)`; its differential and Riesz maps operate only
on those compact vectors. Finite-element relative codifferentiation inverts
the principal metric Gram before zero-extending, rather than masking a full
metric inverse.

### Meshfree higher degrees

`PreparedMeshfreeCellComplex` (see the
[meshfree guide](guides_meshfree.md#higher-exterior-degrees)) publishes all
degrees of an authoritative 2-D, 3-D or codimension-one surface complex as a
native `CochainDiscretization` with natively admitted `SparseHodge` metrics and
an exact `DeRhamBridge`. It is not a second exterior algebra: d, δ, relative
restriction, `trace_map`, harmonic cohomology, `HodgeLaplacePlan` and
`UnstructuredMaxwellPlan` act on its `cochain` unchanged. Commutation
R(dα) = d(Rα) holds to quadrature tolerance by Stokes on the authoritative
cells; the GMLS `sample` route commutes exactly on local polynomials and to
consistency order otherwise. Abstract radius-clique complexes are research
records with exact topology only and are refused as geometry authority.

## Bridges, trajectories and conservation

`DeRhamBridge` combines a realization, chart and explicit cell parameterizations.
`integrate_form` yields a `DiscreteForm`; `validate_de_rham_commutation` measures
R(dα)−d(Rα). Quadrature is declared, not silently changed by refinement.

Chain kernels prepare reusable gather/deposit routes. `PreparedChainQuery.gather`
and `.deposit` are conjugate adjoints. Structured spline Whitney routes and affine
simplicial Whitney routes share this contract. The simplicial locator walks every
facet interval exactly, reports exit/overflow and uses deterministic tie handling;
endpoint-only location is not trajectory integration. Zero segments are admitted.
Curved exactness is not inferred from an affine walker.

`WhitneyProductPlan` retains prepared quadrature geometry and reconstruction
queries. For compiled repeated products, pass the plan itself to
`eqx.filter_jit` as a PyTree argument; its numeric caches, masks and query status
then remain dynamic, and geometry derivatives propagate through the prepared
blades and reconstruction coefficients. Capturing the plan in a `jax.jit`
closure deliberately fixes a numeric snapshot instead. Matrix-fiber wedges
preserve ordered matrix multiplication and use a single flattened batch
contraction, without an outer-product temporary or changing compiler passes.
Failed or overflowing reconstruction queries are still refused inside compiled
products.

PIC gathers physical E and constrained raw B (`magnetic_flux`), not H. Positive
current content follows +∫W; the existing end-to-end continuity sign is retained.
Materials with μ≠1 do not change the Lorentz-force B field. Far-field and PSATD
surface geometries retain their different owning stencils.

Boundary trace maps carry outward-normal-first orientation and commutation
checks. Maxwell matching FEM–BEM coupling includes trace, dual conormal and
boundary operators; a user-supplied periodic block solve alone is not automatic
coupling. Exact cup diagonals remain topology-owned; coefficient systems carry
transport/orientation/Bloch structure rather than embedding it in scalar metadata.

## Forms-aware equations and solver recipes

`PDEField.form` declares `FormValueSpec`. Typed vector operations are views of d:
grad maps scalar→circulation, 3-D curl circulation→flux and div flux→density.
There is no implicit proxy/component crossing. Exterior PDE compilation consumes
an explicit realization and boundary, retaining form and realization identity.

For smooth lowering, `compile_pde_expression` / `compile_pde_problem` receive
charted `DomainDifferentialForm` fields and explicit `PDEFormGeometry`; trace
regions bind declared `PDEFormTrace` maps/coorientation. Spatial partials of a
multi-axis coordinate require an explicit axis. Native domain derivative contracts
and mode/backend selection are retained.

`compile_exterior_pde(problem, realization, fields=..., boundary=...)` returns
`CompiledExteriorPDE` with stable compilation/realization identities and dynamic
numeric fields. `residuals(fields=None)` and `condition_residuals(fields=None)`
return named arrays. `ExteriorPDERealization(complex, products=..., traces=...)`
declares Whitney product geometry and region `ComplexMap` traces explicitly;
shape or region spelling never invents geometry. Serialization carries mandatory
`form` (null or nested FormValueSpec payload), and wedge alone carries its
`scalar`/`matrix` product. Old missing-form payloads are explicitly refused.

::: phydrax.equations.PDEFormGeometry

::: phydrax.equations.PDEFormTrace

::: phydrax.equations.ExteriorPDERealization

::: phydrax.equations.CompiledExteriorPDE

::: phydrax.equations.compile_exterior_pde


`HodgeLaplacePlan(realization, k, boundary=..., formulation="mixed" | "primal",
harmonic=..., preconditioner=...)` requires complete harmonic information, including
a declared zero-dimensional harmonic space. Mixed solves retain the harmonic
multiplier; primal solves refuse incompatible loads. Cavity modes use separate
lossless material coordinate Grams and a resource-bounded eager native full
`DenseEigh` admission, then select positive/kernel-complement modes. The result
retains actual dense provenance, status, per-mode convergence and conservative
full-spectrum orthogonality evidence; no redundant iterative solve or scalable
LOBPCG claim is attached to this recipe.

## Evidence and performance

Correctness evidence is consumer-visible: d², adjoint duality, trace commutation,
closed-form Whitney masses, harmonic dimension/period residuals, conservation,
analytic spectra and fault adequacy. Old/new parity belongs to throwaway migration
smokes, not a permanent duplicate substrate.

`python -m tools.exterior_calculus_qualification` runs the selected native contract
scenarios. `python -m benchmarks.exterior_calculus --sizes 8 16 32 --repeats 5`
records lowering, compile, first and warm execution, cold/prepared reuse,
controlling capacities, compiler temporary/output/code and retained bytes.
Qualification is not release authorization, and benchmark structure is not an
empirical timing claim until the integrated run succeeds.

See [the API](api/exterior/index.md), [complexes](api/exterior/complexes.md),
[bridges](api/exterior/bridges.md) and the
[complete execution plan](plans/exterior-calculus.md).

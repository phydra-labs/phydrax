# Clifford fields and monogenic trial spaces

Phydrax represents a real Clifford algebra through immutable metric, blade-layout, and
prepared-product contracts. Numeric multivectors remain ordinary JAX arrays whose last
axis is the declared blade axis. There is no mutable, operator-overloaded multivector
runtime and no generic hypercomplex capability flag.

## Algebra convention

`CliffordAlgebraSpec` stores the diagonal basis-vector squares directly:

```text
eᵢ² = sᵢ,  sᵢ ∈ {−1, 0, +1}.
```

This avoids ambiguity between competing `Cl(p,q)` conventions. The canonical blade
order is grade-major and lexicographic within each grade, matching the increasing
multi-index convention used by `DifferentialForm`.

```python
import jax.numpy as jnp
import phydrax as phx

cl = phx.metrix.clifford
algebra = cl.CliffordAlgebraSpec((1, 1, 1))
layout = cl.CliffordBladeLayout.full(algebra)
product = cl.prepare_product(algebra, layout, layout)

e0 = cl.basis_blade(layout, 1)
e1 = cl.basis_blade(layout, 2)
assert jnp.array_equal(product(e0, e1), -product(e1, e0))
```

A product plan contains exact source routes, integer coefficients, output support,
resource evidence, and an execution backend. Sparse replay gathers only structurally
nonzero terms. A small dense contraction is selected only when admitted by the declared
resource budget.

## Geometric, exterior, and contraction products

The geometric product is the primary associative operation. Exterior and contraction
products are explicit grade-filtered plans. For general multivectors, the geometric
product is not merely one wedge term plus one contraction term.

Product preparation never silently discards a nonzero blade. Supply an output layout
only when it contains the exact product closure; apply `project_grades` separately when
projection is intended.

## Involutions

The algebra package provides grade involution, reversion, Clifford conjugation, scalar
extraction, and explicit layout embedding/extraction. Generic inversion, Pin/Spin
parameterization, and degenerate-metric duals are intentionally absent.

## Differential-form bridge

`CliffordMetricBridge` connects one constant, nondegenerate orthogonal frame to a
`CoordinateChart`. It raises a homogeneous covariant form into Clifford coefficients and
lowers one Clifford grade back into a `DifferentialForm`. Both paths use the same
exterior-basis index and sign implementation as the existing form calculus.

```python
chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
bridge = cl.CliffordMetricBridge(cl.CliffordAlgebraSpec((1, 1)), chart)
alpha = phx.metrix.DifferentialForm(
    lambda q: jnp.asarray([q[0], 2.0 * q[1]]),
    chart=chart,
    degree=1,
)
alpha_field = bridge.embed(alpha)
recovered = bridge.extract(alpha_field, 1)
```

The bridge rejects radical directions and coordinate-dependent metrics. A varying metric
requires a separate orthonormal-frame/vielbein contract rather than changing a prepared
algebra at every point.

## Isometries and equivariance evidence

Three contracts are distinct:

- `FiniteMetricIsometryGroup` owns a genuinely finite, composition-closed subgroup.
- `MetricIsometryAction` owns one validated matrix satisfying `Rᵀ G R = G`.
- `MetricIsometryAuditSet` is a collection of independent actions with no closure claim.

A nontrivial Lorentz boost has infinite order and belongs to a standalone action/audit
set, never a finite group table. `CliffordOutermorphismPlan` extends one validated vector
action grade by grade. `audit_clifford_action` checks that it preserves the geometric
product.

Initial neural equivariance is stable for Euclidean `Cl(n,0)`. Mixed-signature actions
support linear and polynomial audits, while indefinite norm-based nonlinearities remain
unsupported.

## Clifford neural fields

`CliffordGradeRepresentation` stores one channel multiplicity per complete grade.
`CliffordGradeLinear` mixes channel multiplicities with scalar weights and never mixes
inequivalent grades. `CliffordGeometricProductLayer` uses learned scalar coefficients on
explicit grade projections of true geometric products. Only scalar channels receive a
bias under the full orthogonal group.

```python
representation = phx.nn.operator.representations.CliffordGradeRepresentation(
    algebra,
    (2, 2, 2, 2),
)
layer = phx.nn.operator.layers.CliffordGradeLinear(
    representation,
    representation,
)
```

`clifford_gated_activation` applies an ordinary activation to scalar channels and an
invariant scalar gate to every nonzero grade. It requires a positive-definite algebra;
it never takes a square root of a negative or null pseudo-norm.

Algebra representation, sampled equivariance, and PDE exactness remain independent
evidence channels.

## Dirac operator and monogenic fields

For a constant nondegenerate diagonal metric, Phydrax defines the left Dirac operator

```text
D f = Σᵢ eⁱ ∂ᵢ f.
```

Its square is the flat signed Laplacian:

```text
D² f = Σᵢ sᵢ ∂ᵢ² f.
```

`MonogenicPolynomialBasis` constructs an exact rational nullspace of the polynomial
Dirac map. `LinearMonogenicField` places real trainable coefficients over that fixed
multivector-valued basis and supplies optimized analytic partial derivatives.

```python
algebra = cl.CliffordAlgebraSpec((1, 1))
basis = phx.equations.MonogenicPolynomialBasis(algebra, 2)
model = phx.equations.LinearMonogenicField(basis)

domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
field = domain.Model("x")(model)
```

The bound field carries a `TrialSpaceCertificate` with equation family `"dirac"`.
`audit_trial_space` evaluates the declared Dirac residual on an explicit batch. Generic
field algebra and hard enforcement drop or reject the certificate because monogenic
fields are not generally closed under products or arbitrary nonlinear transformations.

## Spatial metric versus entropy geometry

The spatial Clifford metric and a conservation system's entropy Hessian are different
objects:

```text
Gspace     fixed physical-frame metric; blades, products, Dirac, equivariance
Hstate(u)  state-dependent entropy Hessian; entropy variables and relative entropy
```

`ConvexEntropyPair.hessian_geometry()` must not be passed to `CliffordAlgebraSpec`.
Entropy pair IDs belong to problem and benchmark provenance; they do not alter the
Clifford algebra identity.

## Experimental differential-context candidate

The benchmark tooling includes a non-promoted candidate that combines an explicit
periodic Fourier Laplacian with Clifford grade lifts, geometric-product interactions,
and a residual update. Its smoke scenarios cover velocity/vorticity, entropy-aware
Euler state packing, and electric-vector/magnetic-bivector Maxwell fields.

The smoke only proves executable data and model contracts. It is not a promotion-eligible
training result and makes no superiority claim.

## Unsupported scope

The current release does not claim:

- arbitrary symmetric or coordinate-dependent Clifford metrics;
- generic multivector inverses;
- conformal or projective geometric-algebra convenience models;
- discrete cochain geometric products;
- nonlinear pseudo-orthogonal normalization;
- Cauchy--Clifford boundary projectors;
- Pin/Spin or spinor representations;
- octonionic or generic hypercomplex layers;
- CAN/CliffordNet compatibility.

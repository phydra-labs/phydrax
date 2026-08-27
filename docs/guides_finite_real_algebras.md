# Finite real algebras and coordinate execution

Phydrax represents finite-dimensional algebras as exact multiplication data over real
coordinates. Numeric values remain ordinary JAX arrays; there is no operator-overloaded
hypercomplex runtime.

## Algebra versus storage

`ComplexAlgebraSpec`, `QuaternionAlgebraSpec`, `OctonionAlgebraSpec`,
`CayleyDicksonAlgebraSpec`, and `MulticomplexAlgebraSpec` declare mathematical basis,
product, conjugation, convention, property evidence, and resource limits.

`AlgebraCoordinatePlan` separately declares how algebra values are stored. Native
complex arrays can map to a leading real/imaginary axis. Quaternion, octonion, and
multicomplex values use an explicit real algebra axis.

```python
import jax.numpy as jnp
import phydrax as phx

quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
coordinates = phx.linalg.AlgebraCoordinatePlan(
    quaternion,
    public_storage="real_coordinates",
    public_dtype=jnp.float64,
).prepare((32,))
```

The public value shape is `(32, 4)` and the canonical backend shape is `(4, 32)`.
Both retain the same quaternion algebra identity.

## Exact products and laws

Products are prepared from sparse rational structure terms. A budgeted dense kernel is
optional; both routes have one product identity and preserve syntax-tree bracketing.

```python
product = quaternion.prepare_product(backend="sparse")
i, j = jnp.eye(4)[1:3]
k = product(i, j)
```

Property records are three-valued: `proven`, `disproven`, or `unknown`. Exact basis
audits establish finite polynomial identities such as associativity, alternativity,
flexibility, and conjugation anti-automorphism. Division and norm claims require family
construction evidence or an explicit witness; finite samples do not mint proofs.

## Derived nonassociative operations

Exact specifications and prepared products expose the same commutator, symmetrized
Jordan product, and associator. The associator retains both bracket trees:

```python
octonion = phx.metrix.algebra.OctonionAlgebraSpec()
product = octonion.prepare_product(backend="sparse")
associator = product.associator(left, middle, right)
```

`product.associator(left, middle, right)` evaluates
`product(product(left, middle), right) - product(left, product(middle, right))`.
No compiler path flattens that expression. `commutator` and `jordan_product` are
defined for every algebra; calculating them does not assert commutativity or the
Jordan identity.

## Family boundaries

- complex values are commutative associative division-algebra values;
- quaternions are associative but noncommutative, so left and right multiplication differ;
- octonions are alternative but nonassociative, so product bracketing is part of the program;
- sedenions and later Cayley-Dickson levels do not inherit octonion alternativity or division;
- multicomplex generators commute and rank-two or higher algebras contain zero divisors.

Equal coordinate count does not imply equal algebra. Bicomplex and quaternion values
both use four real coordinates but have different product and property identities.

## Real-coordinate maps

`AbstractRealCoordinateMap` is the canonical public-to-real execution boundary.
Full algebra realification and constrained Hermitian spectral coordinates are distinct:

- algebra realification is a full bijection;
- `HermitianSpectralCoordinates` is a minimal chart over the conjugate-symmetric Fourier subspace.

The maps expose source/coordinate spaces, projection behavior, defect, norm relation,
and stable evidence.

## Operators

`lift_real_operator_to_algebra` applies one real base-space operator independently to
every algebra coordinate. `complexify_real_operator` gives the corresponding native
complex action without duplicating real/imaginary logic. Generic quaternion or
octonion matrices require explicit real-, left-, right-, or bimodule semantics; no
ambiguous algebra-valued matrix multiplication is inferred.

`algebra_regular_action_operator` binds multiplication by one fixed algebra value as
a Phydrax-native real-linear operator. The required `side="left"` or `side="right"`
is part of the operator identity. Composing two left actions remains ordinary operator
composition and is not collapsed to multiplication by one element.

Derivation constraints encode the exact Leibniz equations from the rational structure
table. `plan_algebra_derivations` and `prepare_algebra_derivations` then compute a
resource-bounded numerical nullspace with explicit rank-gap and residual evidence.
The canonical quaternion and octonion derivation spaces have dimensions three and
fourteen, respectively. This is numerical subspace evidence, not an exact nullspace
proof.

## Differential solvers

`DiffraxAlgebraStatePolicy` binds prepared algebra coordinates to ordinary Diffrax
execution. Public callbacks receive the declared algebra layout; the backend integrates
the canonical real coordinates. Real-coordinate quaternion and octonion states also
work directly with delay, jump, and rough solvers because those backends already
operate on real arrays.

Continuation uses the same maps through
`phydrax.continuation.ContinuationRepresentationPolicy`. Native-complex and explicit
algebra-array states remain public branch values, while branch correction and tangent
systems execute over canonical real coordinates. Constrained coordinate maps are
accepted only when state and residual defects satisfy the declared tolerance; no
projection is inserted by the continuation runtime.

Nontrivial geometry remains family-specific. Unit complex and unit quaternion
geometries are provided over real coordinates. Unit octonions are not assigned a Lie
group geometry: their multiplication is nonassociative and their unit sphere is a
Moufang loop.

## Octonions and local G2 geometry

`OctonionG2Bridge` derives the seven-dimensional cross product and canonical
associative three-form directly from the configured octonion multiplication table.
It therefore cannot drift to a second Fano-plane sign convention. `LocalG2Structure`
couples a degree-three form to an explicit seven-dimensional Riemannian metric.
`validate_local_g2_structure` reports metric compatibility, volume normalization,
closure, coclosure, torsion freedom, and Ricci-flatness separately.

Prepared octonion derivations can be checked with `validate_g2_derivations`; the
fourteen-dimensional derivation space infinitesimally preserves the G2 three-form.
This does not make unit octonions a Lie group. Unit-norm octonion states use the
ordinary `SphereManifold(8)` geometry, while multiplication-aware Moufang-loop
integrators remain outside this contract.

## Clifford relationship

`CliffordFiniteAlgebraProvider` exposes a full Clifford blade layout through the
finite-algebra provider protocol while retaining the specialized metric, grade,
exterior, contraction, involution, resource, and outermorphism contracts owned by
`phydrax.metrix.clifford`.

## Resource and backend contracts

Coordinate count, product pairs, sparse terms, exact audit work, plan bytes, and dense
kernel bytes are preflighted before expensive construction. Algebra products can lower
to the shared `LoweredOperatorProgram` for explicit JAX/NumPy backend parity without
making lowered dictionary execution the default callable path.

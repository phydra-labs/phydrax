# Sparse derivatives

`phydrax.sparse` separates sparse derivative compilation from repeated numerical
execution:

1. a `SparsePattern` fixes canonical matrix coordinates,
2. a `SparseColoring` fixes seed colors and extraction indices,
3. a `SparseDerivativePlan` evaluates compressed JAX derivatives at changing
   points and arguments,
4. `plan.operator(...)` returns a `SparseCoordinateOperator` accepted by the
   shared `phydrax.linalg` runtime.

The runtime does not materialize a dense Jacobian or Hessian. It evaluates one
compressed JVP, VJP, or HVP per active color and extracts coefficients directly
in the pattern's canonical route order. Calling `as_dense()` remains an explicit
interoperability operation.

## Declared structural pattern

Use a declared pattern when domain structure already determines the possible
nonzeros. With `compiler="native"`, compilation colors the declared routes and
evaluation remains native JAX.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

source = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
target = phx.linalg.ArraySpace((3,), dtype=jnp.float64)


def residual(values, scale):
    differences = values[1:] - values[:-1]
    return scale * differences**2


pattern = phx.sparse.SparsePattern.from_coo(
    jnp.array([0, 0, 1, 1, 2, 2]),
    jnp.array([0, 1, 1, 2, 2, 3]),
    (3, 4),
    origin="structural",
)
point = jnp.array([0.0, 1.0, 3.0, 6.0])
plan = phx.sparse.compile_sparse_jacobian(
    residual,
    point,
    source=source,
    target=target,
    sample_args=jnp.array(2.0),
    structure=pattern,
    compiler="native",
    chunk_size=2,
)


@jax.jit
def derivative_action(values, scale, direction):
    return plan.operator(values, scale).mv(direction)


image = derivative_action(point, jnp.array(3.0), jnp.ones_like(point))
```

`point` may change freely while preserving the source space. Runtime argument
leaves may change values while preserving the sample PyTree structure, leaf
shapes, and dtypes. Values captured invisibly by the Python closure are fixed at
compilation; expose changing values through `point` or `args`.

## Automatic global detection

With `compiler="auto"` and no supplied structure, Phydrax analyzes the JAXPR
computation graph, propagates global element dependencies, constructs a
`SparsePattern(origin="structural")`, and colors it natively. Unsupported JAX
primitives fail explicitly rather than silently assuming either dense or sparse
structure.

```python
space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)


def energy(values, _):
    differences = values[1:] - values[:-1]
    return jnp.sum(differences**2) + jnp.sum(values**2)


hessian_plan = phx.sparse.compile_sparse_hessian(
    energy,
    point,
    space=space,
    compiler="auto",  # native global JAXPR tracing because structure is omitted
    properties=phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
            "positive_semidefinite": "asserted",
        },
    ),
)
hessian = hessian_plan.operator(point)
result = phx.linalg.solve(
    phx.linalg.LinearSystem(hessian),
    jnp.ones((4,)),
)
```

Structural tracing and graph coloring happen once during plan construction.
Repeated coefficient evaluation and operator application remain compiled native
JAX execution.

## Structure and compiler resolution

`structure` accepts one of:

- `None`: trace and color a global structural pattern automatically,
- `EdgeRelation`: canonicalize and color the declared routes,
- `SparsePattern`: color the supplied canonical pattern,
- `SparseColoring`: reuse the complete precompiled artifact without recoloring.

Compiler resolution is deterministic:

| `structure` | `compiler="auto"` | `compiler="native"` |
| --- | --- | --- |
| `None` | native structural tracing and coloring | rejected |
| relation or pattern | native greedy coloring | native greedy coloring |
| coloring | exact reuse | exact reuse |

Native automatic Jacobian coloring evaluates row and column candidates and uses
the smaller color count, breaking ties toward forward mode. Native Hessian
coloring uses deterministic collision-free column coloring over the symmetric
structural pattern.

## Modes and chunking

Jacobian plans support:

- `fwd`: compressed JVPs seeded in source coordinates,
- `rev`: compressed VJPs seeded in target coordinates.

Hessian plans support `fwd_over_rev`, `rev_over_fwd`, and `rev_over_rev` HVPs.
All return coefficients in exactly the same canonical pattern order.

`chunk_size` limits the number of simultaneous color seeds. A chunk creates at
most `chunk_size × seed_dimension` seed values; the retained compressed result
has size `num_colors × opposite_dimension`. Chunking does not change coefficient
order, values, differentiation, or JIT semantics.

## Portable structural artifacts

`SparsePattern` canonicalizes coordinates by matrix row and then column, removes
duplicates, validates bounds, and computes a deterministic SHA-256 identity.
For symmetric patterns, every transpose entry must be explicit.

`SparseColoring` stores the pattern, color vector, route-wise extraction indices,
mode, compiler provenance, and its own deterministic identity. Both artifacts
support canonical JSON-compatible `to_dict()` and `from_dict()` round trips.
Unknown fields, invalid extraction indices, and fingerprint mismatches are
rejected. Executable functions and derivative plans are intentionally not serialized.

## Verification boundary

A declared pattern is a contract. Phydrax does not infer missing entries from a
sample value or silently replace the pattern with a dense one. Validate a plan
at important points with matrix-free probes:

```python
verification = phx.sparse.verify_sparse_derivative(
    plan,
    point,
    args=jnp.array(2.0),
    key=jax.random.key(0),
    num_probes=4,
)
assert verification.passed
```

Verification compares sparse operator actions with direct JVPs or HVPs without
materializing a dense derivative. Its scope is the supplied point and arguments;
it does not prove that a user-declared pattern is globally valid. Structurally
traced patterns retain separate provenance from caller-declared routes.

## Mathematical restrictions and explicit extensions

`SparseDerivativePrecisionPolicy` separates source seed, target cotangent,
coefficient, accumulation, and output dtypes. Source and target spaces may use
different homogeneous inexact dtypes; action boundaries cast explicitly instead
of requiring one coefficient dtype to validate both directions.

Complex Jacobians require `complex_semantics="holomorphic"` with native complex
forward mode, or `complex_semantics="real-frechet"` with explicit source and
target `AbstractRealCoordinateMap` values. Holomorphy and a Wirtinger convention
are never inferred from samples. `PreparedRealCoordinateTree` composes maps for
homogeneous PyTrees without field-name inference.

`SparseHessianContract("bilinear")` returns the derivative of a covector as a
map to `DualSpace`. `"riesz"` raises it only under a declared constant diagonal
pairing and fixed pattern. `"cotangent"` forms the real scalarization
`Re <cotangent, F(x)>` for vector residuals. There is intentionally no claim of
one universal complex/vector-valued Hessian matrix or hidden dense Riesz map.
Positive definiteness, semidefiniteness, and rank remain explicit evidence.

## API reference

::: phydrax.sparse.SparsePattern

---

::: phydrax.sparse.SparseColoring

---

::: phydrax.sparse.SparseDerivativePlan

---

::: phydrax.sparse.SparseDerivativeVerification

::: phydrax.sparse.SparseDerivativePrecisionPolicy

---

::: phydrax.sparse.SparseHessianContract

---

::: phydrax.sparse.compile_sparse_jacobian

---

::: phydrax.sparse.compile_sparse_hessian

---

::: phydrax.sparse.verify_sparse_derivative

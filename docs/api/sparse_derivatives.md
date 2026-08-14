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
nonzeros. With `compiler="native"`, neither compilation nor evaluation calls
ASDEX.

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

ASDEX is a base dependency and the automatic compiler for an omitted pattern.
It analyzes the sample computation graph, detects a global structural pattern,
and supplies optimized coloring. Phydrax immediately normalizes that output to
`SparsePattern` and `SparseColoring`; the returned plan retains no ASDEX object
and repeated evaluation is native JAX.

```python
space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)


def energy(values, _):
    differences = values[1:] - values[:-1]
    return jnp.sum(differences**2) + jnp.sum(values**2)


hessian_plan = phx.sparse.compile_sparse_hessian(
    energy,
    point,
    space=space,
    compiler="auto",  # ASDEX detection/coloring because structure is omitted
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

ASDEX is imported lazily by sparse compilation, not by `import phydrax` or by
`phydrax.linalg`. Its detection and graph-coloring cost is therefore paid once,
not in the training or solve loop.

## Structure and compiler resolution

`structure` accepts one of:

- `None`: detect and color automatically with ASDEX,
- `EdgeRelation`: canonicalize and color the declared routes,
- `SparsePattern`: color the supplied canonical pattern,
- `SparseColoring`: reuse the complete precompiled artifact without recoloring.

Compiler resolution is deterministic:

| `structure` | `compiler="auto"` | `compiler="native"` | `compiler="asdex"` |
| --- | --- | --- | --- |
| `None` | ASDEX detection and coloring | rejected | ASDEX detection and coloring |
| relation or pattern | native greedy coloring | native greedy coloring | ASDEX optimized coloring |
| coloring | exact reuse | exact reuse | rejected |

Native automatic Jacobian coloring evaluates row and column candidates and uses
the smaller color count, breaking ties toward forward mode. Native Hessian
coloring uses ordinary collision-free column coloring. ASDEX may use optimized
symmetric star coloring for Hessians.

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
support versioned JSON-compatible `to_dict()` and `from_dict()` round trips.
Unknown fields, unsupported schema versions, invalid extraction indices, and
fingerprint mismatches are rejected. Executable functions and derivative plans
are intentionally not serialized.

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
it does not prove that a user-declared pattern is globally valid. ASDEX-origin
patterns retain separate provenance that their structure came from global graph
analysis.

## Mathematical restrictions

Sparse Jacobians currently require real floating-point source and target
coordinates with one shared dtype. Complex or mixed-precision differentiation
needs a more explicit linear-action contract and is rejected.

Sparse Hessians additionally require a scalar real-valued function, a square
symmetric structural pattern, and a Euclidean pairing. A coordinate Hessian is
not silently treated as a primal-space endomorphism under a non-Euclidean
pairing. Positive definiteness, semidefiniteness, and rank are never inferred by
sampling; supply certified `OperatorProperties` when a solver may rely on them.

## API reference

::: phydrax.sparse.SparsePattern

---

::: phydrax.sparse.SparseColoring

---

::: phydrax.sparse.SparseDerivativePlan

---

::: phydrax.sparse.SparseDerivativeVerification

---

::: phydrax.sparse.compile_sparse_jacobian

---

::: phydrax.sparse.compile_sparse_hessian

---

::: phydrax.sparse.verify_sparse_derivative

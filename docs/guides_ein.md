# Einstein operations

`phydrax.ein` provides one stable import boundary for tensor contractions and a
small named-axis language for unary JAX transformations.

## Contractions

`ein.contract` is the exact `opt_einsum.contract` function. Existing compact
equations and optimization controls work unchanged:

```python
import jax.numpy as jnp
import phydrax as phx

values = jnp.arange(6.0).reshape(2, 3)
weight = jnp.arange(12.0).reshape(3, 4)
result = phx.ein.contract("...i,ij->...j", values, weight, backend="jax")
```

For semantic labels that do not fit the compact single-character notation, use
the unambiguous interleaved form:

```python
result = phx.ein.contract(
    values,
    ("batch", "feature"),
    weight,
    ("feature", "output"),
    ("batch", "output"),
    backend="jax",
)
```

Backend inference, explicit contraction paths, `PathOptimizer` instances,
`memory_limit`, `use_blas`, and backend-specific keyword behavior belong to
opt-einsum. The convenience boundary does not add a hidden plan cache or change
contraction order.

Use `phydrax.tensor_network` when a contraction requires admitted resource
policies, immutable schedules, execution evidence, reverse replay, or
caller-owned plan caching.

## Unary pattern grammar

The unary operations use multi-character logical axis names:

```text
input expression -> output expression
```

| Construct | Meaning |
|---|---|
| `batch channel` | Two physical and logical axes |
| `(height patch)` | One physical axis factored into two logical axes |
| `...` | Zero or more physical axes resolved from input rank |
| `(...)` | Collapse all resolved ellipsis axes into one output axis |
| `1` | Anonymous singleton physical factor |
| empty side | Rank-zero input or output |

Names are case-sensitive ASCII identifiers beginning with a letter. Parentheses
may be nested only one level, and `1` is the only anonymous numeric literal.
Input grouping and output regrouping use C-order semantics: the rightmost
logical factor varies fastest.

Pattern strings and explicit axis sizes are Python-static. Explicit sizes must
be positive integers. A grouped input physical axis may have at most one
unknown logical factor; Phydrax infers it only when the physical size is exactly
divisible by every known factor.

## Rearrangement

`rearrange` preserves the named-axis set while changing physical grouping and
order:

```python
patches = phx.ein.rearrange(
    image,
    "batch channel (height patch_h) (width patch_w) "
    "-> batch height width (patch_h patch_w channel)",
    patch_h=2,
    patch_w=2,
)
```

Singleton insertion and removal are also rearrangements:

```python
expanded = phx.ein.rearrange(values, "batch channel -> 1 channel batch 1")
```

A direct `reshape`, `transpose`, or `swapaxes` remains preferable when it is
already clearer than a named pattern.

## Reduction

`reduce` removes named or ellipsis axes. The initial reduction vocabulary is
`sum`, `mean`, `prod`, `min`, `max`, `all`, and `any`:

```python
channel_mean = phx.ein.reduce(
    values,
    "batch time channel -> batch channel",
    "mean",
)
scalar = phx.ein.reduce(values, "batch time channel ->", "sum")
```

Reduction dtype promotion, empty-axis identities, NaN behavior, and no-identity
errors follow the corresponding JAX primitive. A reduction must remove at least
one actual axis after ellipsis expansion.

## Repetition

`repeat` adds explicitly sized named axes using JAX broadcast semantics:

```python
replicated = phx.ein.repeat(
    values,
    "batch channel -> batch replica channel",
    replica=4,
)
```

The operation emits `lax.broadcast_in_dim`, not an eager tile. Device
materialization remains a compiler decision. A repetition must add at least one
named axis; use `rearrange` for singleton-only shape changes.

## Ellipsis and scalars

Input ellipsis must be standalone. Output ellipsis may be moved, preserved, or
collapsed inside one group:

```python
channels_first = phx.ein.rearrange(values, "... channel -> channel ...")
flattened_batch = phx.ein.rearrange(values, "... channel -> channel (...)")
```

A zero-width ellipsis expands to no axes. When it appears inside an output
group, its product is `1`. Empty expressions represent scalars:

```python
identity = phx.ein.rearrange(scalar, "->")
vector = phx.ein.repeat(scalar, "-> replica", replica=3)
```

## JAX transformations

Pattern parsing and shape specialization happen in Python during tracing. Only
array values enter compiled execution. Close over patterns and sizes or mark
them static when using `jax.jit`.

The native transforms compose with `jax.jit`, `jax.grad`, forward- and
reverse-mode differentiation, and `jax.vmap`; they do not install custom
transformation rules. Dynamic rank and traced factor sizes are rejected.

## Coordax fields

Ein pattern names describe positional payload axes for one array operation.
They do not match or mutate persistent `coordax.Field` dimension names.
Apply an ein operation inside `coordax.cmap` when named sampling axes should be
vectorized, or explicitly untag a named dimension before contracting it. Equal
strings in the two systems never imply automatic contraction or retagging.

## Current non-goals

The native unary language intentionally excludes:

- automatic operation inference,
- named contraction strings,
- nested groups and general symbolic algebra,
- callable reductions,
- packing and concatenation syntax,
- runtime backend registration,
- sharding or mesh annotations,
- generic contraction plan objects.

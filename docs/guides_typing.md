# Typing and structural contracts

PhydraX types its public surface for static checkers and uses one small
annotation vocabulary, `phydrax.typing`, to declare the structural facts of
arrays and static metadata. The same annotation is read three ways:

- a **static checker** sees an ordinary base type (`jax.Array`,
  `numpy.typing.NDArray[numpy.float64]`, `int`, `str`, `tuple[str, ...]`);
- the **runtime contract language** checks object kind, dtype, rank, extents,
  named-dimension agreement, sizes, identifiers, and closed selectors;
- the **documentation** shows the declared dtype and shape.

The layers of the architecture own different facts:

| Layer | Owns |
|---|---|
| `phydrax.typing` | static forms, explicit host/device conversion, structural field contracts, selector parsing |
| Scientific owners | axis identity (`phydrax.axes`), units (`phydrax.units`), vector spaces, model ports, dtype names, exact PyTree schemas |
| Status and evidence | numerical validity, support, rank, conditioning, convergence, provenance |

Structural contracts never replace a scientific owner: finiteness,
positivity, definiteness, rank, and conditioning stay with the owner that can
report status and evidence for them.

## Dimensions are size variables

A `Dim` subclass is a nominal size variable. Its identity is the class object,
so a typo is a missing name rather than a new, unrelated dimension.

```python
import phydrax.typing as pt


class ComponentDim(pt.Dim, minimum=1):
    """Number of chemical components."""


class ElementDim(pt.Dim):
    """Number of chemical elements."""


class BatchDims(pt.VariadicDim):
    """Leading batch axes."""
```

A dimension only states that extents agree wherever it appears. It is not a
scientific axis: source/target, row/column, or covariant/contravariant identity
belongs to `phydrax.axes.AxisKey` and `AxisRef`. By convention ordinary
dimensions end in `Dim` and variadic groups in `Dims`.

## Tensor forms

JAX forms have the static type `jax.Array`; host forms have a NumPy array type
that carries the dtype for exact forms.

| JAX | Host | Accepted dtypes |
|---|---|---|
| `Bool` | `HostBool` | boolean |
| `Int32`, `Int64`, `UInt32` | `HostInt32`, `HostInt64` | that exact dtype |
| `Float32`, `Float64` | `HostFloat32`, `HostFloat64` | that exact dtype |
| `Complex64`, `Complex128` | `HostComplex128` | that exact dtype |
| `Integer`, `Float`, `Complex`, `Inexact` | `HostInteger`, `HostFloat`, `HostInexact` | the dtype category |
| `Shaped` | `HostShaped` | any dtype |

Shape arguments are:

- `Literal[3]`: a fixed extent;
- a `Dim` subclass: a named extent;
- `AnyDim`: one extent of any size;
- `Broadcast[D]`: an extent of one, or the extent bound to `D`;
- a `VariadicDim` subclass: zero or more extents, at most one group per form;
- `Scalar`: rank zero, as the only argument;
- `AnyShape`: any rank, as the only argument.

```python
from typing import Literal

values: pt.Float64[ComponentDim]
composition: pt.Int32[ElementDim, ComponentDim]
points: pt.Float64[BatchDims, Literal[3]]
scale: pt.Float64[pt.Scalar]
host_table: pt.HostFloat64[pt.AnyDim, ComponentDim]
```

## Metadata forms

- `Size[D]` is an exact `int` (never `bool` or a NumPy integer) that binds `D`.
- `Identifier` is a non-empty string without surrounding whitespace.
- `Identifiers[D]` is a tuple of unique identifiers whose length binds `D`.
- `PRNGKey` is a typed scalar key from `jax.random.key`; legacy `uint32[2]`
  keys are not keys.
- `Literal[...]` aliases and `Enum` subclasses are closed selectors.
- `X | None`, unions of contract forms, and fixed tuples of contract forms
  compose. A union may not mix contract forms with ordinary annotations.

The grammar is closed. Any other annotation is an ordinary, static-only
annotation; Phydrax vocabulary in any other placement (for example
`list[Float64[D]]`) is refused with `TypeError`.

## Scopes

One `Scope` holds the dimension bindings of one validation operation, together
with the field that established each binding. Failed union alternatives roll
their bindings back; separate scopes are independent.

```python
import jax.numpy as jnp

scope = pt.Scope()
names = pt.parse(("h2", "o2"), pt.Identifiers[ComponentDim], "names", scope=scope)
masses = pt.parse(jnp.asarray((2.016, 31.998)), pt.Float64[ComponentDim], "masses", scope=scope)
assert scope.size(ComponentDim) == 2
```

## Parsing and conversion

- `parse(value, form, name)` validates a value and returns it. For a `Literal`
  selector it returns the declared literal when the value merely compares equal
  (for example a `numpy.str_`); exact-type matches take precedence, and arrays
  and containers are refused.
- `as_array(value, form, name)` converts once to a JAX array. JAX inputs stay on
  device; host inputs go through `numpy.asarray` once. Exact forms cast under
  the requested NumPy casting policy (default `"same_kind"`); category forms keep
  the input dtype and never promote across categories. Strings, bytes, mappings,
  object arrays, and ragged sequences are refused.
- `as_host_array(value, form, name)` converts once to a NumPy array, with an
  explicit device-to-host transfer for concrete JAX arrays; traced values are
  refused.

`ConvertibleToArray` (and `Like[F]`, which documents the target form) is the
static type of conversion inputs: JAX array-likes, objects exporting
`__array__`, and nested numeric lists and tuples up to depth four. Deeper host
data must already be a NumPy or JAX array. `Like[F]` describes inputs only and
is refused on stored fields. Convert exactly once at the boundary; internal
kernels accept canonical arrays.

## Field contracts

`validate(instance)` checks every contract field of a dataclass-based module in
declaration order. Stored fields are checked read-only, so selector fields must
hold the exact declared literal type. Field validation never converts, mutates,
or adds JAX operations.

```python
import equinox as eqx

from phydrax import StrictModule


class Catalog(StrictModule):
    names: pt.Identifiers[ComponentDim] = eqx.field(static=True)
    masses: pt.Float64[ComponentDim]
    composition: pt.Int32[ElementDim, ComponentDim]
    count: pt.Size[ComponentDim] = eqx.field(static=True)
```

Annotations of contract fields must resolve at runtime: names imported only under
`TYPE_CHECKING` are refused with the class and field in the error.

## Transformations

Checks read only shape and dtype metadata, so they run on tracers at trace time
and add no operations to the compiled program. Under `vmap` a check sees the per-example
shape. Symbolic (shape-polymorphic) extents are refused. Transformations such as
`jax.tree_util.tree_map` and `equinox.tree_at` rebuild modules without running
constructors; consumers call `validate` explicitly where a transformed value
must satisfy its declared contract.

## Exceptions

- Wrong object kind, backend, or dtype: `TypeError`.
- Wrong rank, extent, dimension binding, minimum, identifier value, or selector
  value: `ValueError`.

## Static checking

The repository is checked with the pinned ty through `tools/check_typing.py`:

```bash
python tools/check_typing.py check
```

The gate requires zero ty diagnostics and complete annotations: every function,
method, and nested helper annotates all parameters and its return type. Only
`ty: ignore[rule]` suppressions are honored, and only for proven checker or
third-party stub defects.

Constructors are typed by each concrete class. A concrete module that inherits a
custom constructor from an abstract owner declares a checker-only alias,
matching Equinox's runtime constructor choice:

```python
from typing import TYPE_CHECKING


class Doubling(AbstractScale):
    if TYPE_CHECKING:
        __init__ = AbstractScale.__init__
```

Field order is never changed to satisfy a checker: Equinox flattens fields in
declaration order, so reordering would change PyTree leaves and fingerprints.

## Outlook: JAX-level types

Structural contracts are backend-neutral records of canonical dtype names and
nominal dimensions. Opaque JAX-level value types, whose semantic identity,
tangent, batching, and sharding rules would be visible inside jaxprs, are a
separate future track; package code does not depend on experimental JAX type
extensions.

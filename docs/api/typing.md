# Typing

`phydrax.typing` provides the annotation vocabulary and explicit structural
boundaries: nominal dimensions, JAX and host tensor forms, metadata forms,
binding scopes, parsing, one-time conversion, and checked signatures.

See the [typing guide](../guides_typing.md) for the grammar, scopes,
conversion, signature, and exception rules.

## Dimensions and shape tokens

::: phydrax.typing.Dim
    options:
      show_root_heading: true
      show_source: false
      members: false

::: phydrax.typing.VariadicDim
    options:
      show_root_heading: true
      show_source: false
      members: false

::: phydrax.typing.AnyDim
    options:
      show_root_heading: true
      show_source: false
      members: false

::: phydrax.typing.AnyShape
    options:
      show_root_heading: true
      show_source: false
      members: false

::: phydrax.typing.Scalar
    options:
      show_root_heading: true
      show_source: false
      members: false

::: phydrax.typing.Broadcast
    options:
      show_root_heading: true
      show_source: false
      members: false

## Scopes and boundaries

::: phydrax.typing.Scope
    options:
      show_root_heading: true
      show_source: false
      members:
        - size
        - shape

::: phydrax.typing.parse
    options:
      show_root_heading: true
      show_source: false

::: phydrax.typing.as_array
    options:
      show_root_heading: true
      show_source: false

::: phydrax.typing.as_host_array
    options:
      show_root_heading: true
      show_source: false

::: phydrax.typing.validate
    options:
      show_root_heading: true
      show_source: false

::: phydrax.typing.checked
    options:
      show_root_heading: true
      show_source: false

## Conversion inputs

::: phydrax.typing.SupportsArray
    options:
      show_root_heading: true
      show_source: false
      members: false

## Forms

The tensor forms `Bool`, `Int32`, `Int64`, `UInt32`, `Float32`, `Float64`,
`Complex64`, `Complex128`, `Integer`, `Float`, `Complex`, `Inexact`, and
`Shaped` have the static type `jax.Array`. The host forms `HostBool`,
`HostInt32`, `HostInt64`, `HostFloat32`, `HostFloat64`, `HostComplex128`,
`HostInteger`, `HostFloat`, `HostInexact`, and `HostShaped` have NumPy array
static types. `Size[D]` is `int`, `Identifier` is `str`, `Identifiers[D]` is
`tuple[str, ...]`, `PRNGKey` is `jax.Array`, and `ConvertibleToArray` and
`Like[F]` describe conversion inputs.

Exact forms retain their declared NumPy dtype in immutable contract plans.
For JAX forms, checking and conversion resolve that declaration to JAX's effective
canonical width under the current x64 setting. Thus `Float64` means float32 with
x64 disabled and float64 with x64 enabled; `Int64` and `Complex128` follow the same
canonicalization policy. Host exact forms retain the declared NumPy width
regardless of x64; float32 never satisfies `HostFloat64`.

Cached parse, class, and signature plans do not freeze effective widths. Scoped
numerical settings apply when checking or converting, including during JIT
tracing. Validation preserves array identity and refuses incompatible historical
arrays rather than converting them; explicit conversion with `casting="safe"`
refuses narrowing to the effective target. Scientific precision requirements
remain with their owning substrates.

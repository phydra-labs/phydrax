# Typing

`phydrax.typing` provides the annotation vocabulary and explicit structural
boundaries: nominal dimensions, JAX and host tensor forms, metadata forms,
binding scopes, parsing, and one-time conversion.

See the [typing guide](../guides_typing.md) for the grammar, scopes,
conversion, and exception rules.

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

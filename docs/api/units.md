# Physical units

`phydrax.units` provides exact static dimensions, immutable multiplicative unit
definitions, a canonical catalog, and explicit value conversion. Unit metadata
is designed for plans, adapters, provenance, and artifacts rather than as a
wrapper around numerical array leaves.

See the [physical dimensions and units guide](../guides_units.md) for execution,
compatibility, and persistence rules.

## Core types

::: phydrax.units.DimensionSignature
    options:
      show_root_heading: true
      show_source: false

::: phydrax.units.UnitDefinition
    options:
      show_root_heading: true
      show_source: false

## Conversion

::: phydrax.units.conversion_factor
    options:
      show_root_heading: true
      show_source: false

::: phydrax.units.convert_value
    options:
      show_root_heading: true
      show_source: false

::: phydrax.units.derived_unit
    options:
      show_root_heading: true
      show_source: false

## Unit expressions

`parse_unit` resolves a declared unit string such as `"kg/(m^2*s)"` or
`"(m/s)^2"` into an exact `UnitDefinition` over the atomic catalog symbols,
`1`, integer power suffixes (`m3`), `*`, `/`, whitespace products, parentheses,
and integer or rational `^` powers. Unknown symbols and malformed expressions
raise `ValueError`.

::: phydrax.units.parse_unit
    options:
      show_root_heading: true
      show_source: false

## Canonical catalog

The namespace exports canonical definitions for the SI/coherent base and the
physical units used by current PhydraX domains. Text aliases are resolved only
by explicit domain adapters or the closed `parse_unit` grammar; there is no
global runtime registry.

Nuclear and magnetic applications additionally export exact `BARN`,
`BECQUEREL`, `KILOELECTRONVOLT`, `MEGAELECTRONVOLT`, `WEBER`, `TESLA`, and
`HENRY` definitions. Astrophysical images and visibilities use the canonical
`JANSKY` definition ($10^{-26}\,\mathrm{W\,m^{-2}\,Hz^{-1}}$).

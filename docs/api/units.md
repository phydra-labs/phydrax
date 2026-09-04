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

## Canonical catalog

The namespace exports canonical definitions for the SI/coherent base and the
physical units used by current PhydraX domains. Text aliases are resolved only
by explicit domain adapters; there is no global runtime registry or expression
parser.

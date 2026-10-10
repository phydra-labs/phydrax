# Nuclear

`HeavyMetalReference`, `ExposureState`, `advance_exposure`, and
`FissionGasYieldTable` provide source-bound reference fuel exposure and gas
production. `HeliumProductionPlan` binds explicit isotope-resolved ternary
yields and native decay transitions; `HeliumProductionRates` retains their
separate contributions in atoms/m³/s. These APIs do not supply an evaluated
helium library, fuel transport laws, isotope depletion, or scientific fuel
release. See [nuclear data](../guides_nuclear_data.md) and
[fuel source boundaries](../nuclear_fuel_sources.md).

::: phydrax.nuclear
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Interchange

::: phydrax.nuclear.interchange
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Internal dosimetry

::: phydrax.nuclear.dosimetry
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Native transport and depletion candidates

`SlabSNTransportPlan` implements bounded one-dimensional multigroup discrete
ordinates with prescribed/vacuum inflow, source iteration, and a finite
criticality power iteration. `BatemanDepletionPlan` applies the native matrix
exponential to a caller-supplied finite transition matrix; `decay_heat_w`
contracts inventories, decay constants, and recoverable energies.

These are exact unreleased candidate profiles, not evaluated-data processing,
continuous-energy Monte Carlo, multidimensional reactor transport, burnup-chain
authority, thermal-hydraulic feedback, shielding certification, or nuclear-safety
claims.

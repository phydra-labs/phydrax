# Nuclear

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

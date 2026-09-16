# Algebraic systems

`phydrax.algebraic` represents sparse polynomial systems once and keeps structural
analysis, numerical candidates, exact symbolic claims, and physical acceptance
separate.

## Core contracts

::: phydrax.algebraic
    options:
      members_order: source
      show_root_heading: true
      show_source: false

## Claim boundaries

- A small residual is numerical root evidence, not a completeness certificate.
- `all_expected_paths_accounted` describes the selected provider start system; it
  does not certify that every reported endpoint is distinct or real.
- Near-real coordinates are candidates. Certified realness needs a proof-producing
  provider.
- Witness points represent a sliced component and do not by themselves prove
  irreducibility.
- Exact symbolic results remain provider-attributed unless Phydrax independently
  verifies the exact identity.

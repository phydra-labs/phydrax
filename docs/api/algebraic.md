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

## Krawczyk workspace admission

`krawczyk_certificate` requires a dense center-Jacobian preconditioner for its
interval enclosure. Its keyword-only `maximum_dense_workspace_bytes` defaults to
64 MiB. Before evaluating the Jacobian, it admits the conservative float64 estimate
`8 * (10*n*n + 12*n)` bytes, where `n` is the variable count, and refuses an
over-budget request. The estimate covers dense factors, solve scratch, and interval
matrices, not provider-private or polynomial-evaluation storage.

`KrawczykCertificate.dense_workspace_bytes` and
`KrawczykCertificate.maximum_dense_workspace_bytes` report the estimate and ceiling.
Changing the admitted ceiling does not change the scientific `certificate_id`.

# Astrophysics API

The general-relativistic observation surface provides observer screens, bounded null
and timelike rays, ordered terminal events, parallel-transport/Jacobi evidence,
fast-light and slow-light plasma sampling, MNY96 Stokes-$I$ synchrotron with validated
$K_2$ support and reference-unqualified active polarization/Faraday approximations,
exact ray-result/metric `PolarizedRayPath`, chart/path/snapshot-bound fast-light
segment-midpoint sampling, invariant scalar/polarized transfer, Jy-aware physical
Stokes images, direct visibilities and closure products, neutral FITS/UVFITS payloads,
and fixed-branch inference. Read the
[GR imaging guide](../guides_black_hole_imaging.md) for exact conventions and
derivative boundaries.

`gr_chart_identity` addresses chart name and ordered coordinates.
`gr_metric_identity` additionally addresses metric convention and callable content:
same-chart metrics with different physical parameters retain different metric IDs.
Screens, rays, and polarized paths compare both IDs before composition. Opaque metric
callables require paired semantic and numeric content IDs rather than a display label.

::: phydrax.applications.astrophysics
    options:
      members: true
      show_root_heading: true

## Gravitational-wave inference

Canonical detector spectra, waveform response, normalized inference,
marginalization, and qualified likelihood compression have a dedicated
[API reference](applications/gravitational_waves.md).

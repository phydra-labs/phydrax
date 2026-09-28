# Thin-film optics API

See the [thin-film interference and color guide](../guides_thin_film_optics.md)
for the model, conventions, error bounds and nonclaims.

## Interference

::: phydrax.optics.wave.ThinFilmInterferencePlan

---

::: phydrax.optics.wave.ThinFilmInterferenceResult

---

::: phydrax.optics.wave.ThinFilmInterferenceEvidence

---

::: phydrax.optics.wave.ThinFilmInterferenceStatus

## Colorimetry and appearance

The spectral colorimetry and appearance contracts are members of
`phydrax.rendering` and are documented once, on the [Rendering API](rendering.md)
page:

- plans and results: `SpectralColorimetryPlan`, `SpectralColorimetryResult`,
  `SpectralColorimetryEvidence`, `SpectralColorimetryStatus`;
- observers: `AnalyticColorMatchingFunctions`,
  `TabulatedColorMatchingFunctions`, `ColorMatchingFitError`,
  `COLOR_MATCHING_TABLE_MODEL`;
- illuminants: `SpectralIlluminant`, `read_spectral_illuminant`;
- transforms: `spectral_to_xyz`, `xyz_to_linear_srgb`, `encode_srgb`;
- appearance: `ThinFilmAppearancePlan`, `ThinFilmAppearanceResult`,
  `ThinFilmAppearanceEvidence`, `ThinFilmAppearanceStatus`;
- support-aware surface fields: `thin_film_surface_colors`,
  `ThinFilmSurfaceColorResult`, `ThinFilmSurfaceColorStatus`.

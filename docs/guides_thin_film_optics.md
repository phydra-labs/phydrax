# Thin-film interference and color

Phydrax renders the interference colors of thin liquid films (soap films,
foam sheets, coatings on a substrate) in three explicit stages:

1. `phydrax.optics.wave.ThinFilmInterferencePlan` computes complex amplitudes,
   reflectance and transmittance of one homogeneous film between two
   semi-infinite media;
2. `phydrax.rendering.SpectralColorimetryPlan` integrates spectral reflectance
   factors against a CIE 1931 observer and an illuminant and encodes sRGB;
3. `phydrax.rendering.ThinFilmAppearancePlan` maps film thickness and
   orientation samples to colors through the two stages above.

Each stage returns per-sample status bits and evidence. Rendering is
downstream of film physics: it consumes thickness and orientation arrays from
any solver and never alters or certifies the film state.

## Scope and nonclaims

- One lossless film (real index) between a lossless ambient medium and a
  passive substrate (complex index allowed). Absorbing films and general
  multilayer stacks are not claimed; a stable absorbing multilayer route needs a
  scattering-matrix formulation and is deferred until a consumer requires it.
- Coherent, specular, plane-wave optics. Surface roughness, scattering, finite
  source coherence (the washing-out of very thick films), and polarization-resolved
  rendering are not modeled; the appearance stage reports unpolarized reflectance.
- No chromatic adaptation. Colors are relative to the declared illuminant, and a
  non-D65 illuminant white is reported, not corrected (`white_neutral_error`).
- No CIE data table is bundled. The CIE color-matching and illuminant datasets
  are licensed CC BY-SA 4.0; they enter only as caller-supplied host resources
  whose SHA-256 is verified.

## Interference model

Wavelengths are strictly increasing vacuum wavelengths in meters, thickness is
the physical film thickness in meters and the incidence is `cos(theta_0)` in the
ambient medium. Each refractive index is a scalar or one value per wavelength,
so a dispersive law enters as samples, for example from
`phydrax.optics.materials.evaluate_refractive_index` at `omega = 2 pi c / lambda`
after checking its `accepted` flags. The film and substrate indices are
inferable parameter leaves.

The multiple-beam series is summed exactly (Born and Wolf, section 7.6):

```text
r    = (r01 + r12 exp(2 i beta)) / (1 + r01 r12 exp(2 i beta))
t    = t01 t12 exp(i beta)       / (1 + r01 r12 exp(2 i beta))
beta = 2 pi d n1 cos(theta_1) / lambda
```

The single-interface amplitudes `r_ij`, `t_ij` are the Fresnel amplitudes of
`phydrax.optics.geometric.evaluate_refractive_interface` (same kernel, same
`(s, p)` ordering and p basis, `exp(-i omega t)` convention). Normal wavenumber
components `n_j cos(theta_j) = sqrt(n_j^2 - n_0^2 sin^2(theta_0))` use the passive
branch `Im >= 0`, so frustrated total internal reflection in the film and
evanescent or absorbing substrates need no special cases.

`T` is the normal Poynting flux entering the substrate over the incident flux:
`Re(n2 cos theta_2) |t_s|^2 / (n0 cos theta_0)` for s and
`Re(conj(n2) cos theta_2) |t_p|^2 / (n0 cos theta_0)` for p. Because ambient and
film are lossless, `R + T = 1` for every passive substrate; `energy_residual`
reports `|R + T - 1|` and the plan flags samples above `energy_tolerance`
(default `256 eps` of the evaluation dtype).

For a free-standing film (`n0 = n2`), reflection vanishes when
`2 n1 d cos(theta_1) = m lambda` and peaks at
`4 R1 / (1 + R1)^2` when `2 n1 d cos(theta_1) = (m + 1/2) lambda`. The `d -> 0`
limit is the black film: the two reflections cancel through the half-wave phase
difference between the two faces.

### Spectral fringe sampling

Spectral colors are quadratures over the wavelength grid, so the reflectance
fringes must be resolved. The evidence reports the round-trip phase
`phi = 4 pi d Re(n1 cos theta_1) / lambda` and, for every adjacent wavelength
interval, `2 pi / |Delta phi|` samples per fringe. For a nondispersive film this
equals `lambda_a lambda_b / (2 n1 d cos(theta_1) Delta lambda)`, i.e. the fringe
period `Delta lambda_fringe ~ lambda^2 / (2 n d cos theta)` divided by the grid
spacing. `required_samples_per_fringe` (default 4) turns an undersampled
spectrum into `SPECTRAL_UNDERSAMPLED`; `None` records the evidence without
enforcing it, for discrete laser lines. On a 5 nm grid from 380 nm the default
admits films up to about 2 micrometers at normal incidence.

### Status and evidence

| `ThinFilmInterferenceStatus` bit | Meaning |
|---|---|
| `INVALID_THICKNESS` | negative or non-finite thickness |
| `INVALID_INCIDENCE` | `cos(theta_0)` not in `(0, 1]` |
| `SPECTRAL_UNDERSAMPLED` | fewer samples per fringe than required |
| `ENERGY_RESIDUAL_EXCEEDED` | `max |R + T - 1|` above tolerance |
| `NONFINITE` | non-finite amplitudes or flux |

Every rejected sample carries NaN amplitude, flux, and spectrum outputs,
including finite calculations rejected for spectral undersampling or excess
energy residual. The nested interference evidence retains finite quality
diagnostics such as fringe sampling and maximum energy residual, together with
the owning rejection status. It also reports `film_evanescent` and
`substrate_evanescent` (no propagating wave in the film or substrate) per
wavelength. Evaluation is differentiable with respect to thickness, incidence
and the film and substrate indices; derivatives are only meaningful where the
status stays zero.

## Colorimetry

`SpectralColorimetryPlan(wavelengths, illuminant)` computes, for a reflectance
or transmittance factor `rho`,

```text
XYZ = k sum_j w_j S_j rho_j cmf_j,   k = 1 / sum_j w_j S_j ybar_j
```

with trapezoid weights `w`, illuminant power `S` linearly interpolated from its
table and color-matching functions `cmf = (xbar, ybar, zbar)`. A perfect
reflector has `Y = 1`. The plan wavelengths must lie inside both the observer
support and the illuminant table; nothing is extrapolated.
`color_matching_coverage` reports the fraction of each color-matching integral
inside the plan interval.

Linear sRGB is `exposure * M XYZ`, where `M` is derived at full precision from
the IEC 61966-2-1 primaries and the D65 white point `(0.3127, 0.3290)`, so that
D65 white maps exactly to `(1, 1, 1)`. `encode_srgb` applies the IEC transfer
function (`12.92 c` below `0.0031308`, `1.055 c^(1/2.4) - 0.055` above; 0 and 1
are fixed points; the standard's rounded constants leave a branch mismatch of
about 3e-8). `gamut_mapping="clip"` clips linear sRGB to `[0, 1]` before
encoding; `"extended"` keeps out-of-range values and encodes them with the
sign-symmetric extension. Interference colors are often outside the sRGB gamut,
so `out_of_gamut` and `gamut_excess` are always reported.

### Observers

`AnalyticColorMatchingFunctions` (the default) is the Wyman, Sloan and Shirley
(2013) multi-lobe piecewise-Gaussian fit of the CIE 1931 2-degree observer
(their Eq. 4 and Table 1). Its published error against the 1 nm CIE curves is
carried as `fit_error`: maximum squared errors `(2.0e-4, 6.4e-5, 4.9e-4)` and
mean squared errors `(3.1e-5, 7.1e-6, 1.6e-5)` for `(x, y, z)`, below the
within-subject variance of the color-matching experiments.

`TabulatedColorMatchingFunctions(artifact, manifest, policy=...)` decodes a
headerless `wavelength_nm,x,y,z` CSV (the CIE `CIE_xyz_1931_2deg.csv` layout)
from an admitted host artifact whose manifest `model` is
`COLOR_MATCHING_TABLE_MODEL`. The bytes are re-read and their size and SHA-256
re-verified against the trusted manifest on construction.

### Illuminants

`SpectralIlluminant(wavelengths, relative_power, illuminant_id=...)` accepts
caller arrays (a measured lamp, a Planckian radiator, CIE illuminant E as a
constant). `read_spectral_illuminant(artifact, manifest, policy=...,
illuminant_id=...)` decodes a headerless `wavelength_nm,power` CSV (the CIE
`CIE_std_illum_D65.csv` layout) from an admitted artifact and records its
SHA-256 in `source_sha256`.

CIE standard illuminant D65 is published by the CIE as dataset
DOI 10.25039/CIE.DS.hjfjmt59 (CC BY-SA 4.0; SHA-256
`e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb`). The CIE 1931
observer table is DOI 10.25039/CIE.DS.xvudnb9b (CC BY-SA 4.0; SHA-256
`fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1`). To use them,
download the files yourself and admit them with a pinned manifest:

```python
from phydrax.artifacts import (
    ArtifactManifest,
    ExternalArtifactPolicy,
    admit_external_artifact,
)
from phydrax.rendering import read_spectral_illuminant

policy = ExternalArtifactPolicy(
    "/data/cie",
    maximum_bytes=1 << 20,
    allowed_license_ids=["CC-BY-SA-4.0"],
    allowed_suffixes=[".csv"],
)
manifest = ArtifactManifest(
    artifact_id="CIE_std_illum_D65.csv",
    producer="CIE",
    version="10.25039/CIE.DS.hjfjmt59",
    sha256="e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb",
    byte_size=6820,
    source_uri="https://files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv",
    license_id="CC-BY-SA-4.0",
    model="cie-standard-illuminant-d65",
    coverage="300-830 nm, 1 nm",
)
d65 = read_spectral_illuminant(
    admit_external_artifact("CIE_std_illum_D65.csv", manifest, policy=policy),
    manifest,
    policy=policy,
    illuminant_id="cie-standard-illuminant-d65",
)
```

### Documented approximation error

`tools/thin_film_optics_qualification.py --fetch` downloads both CIE tables into
a local cache only when they match the pins above, admits them through the same
artifact policy and records the following (CPU, float64):

| Check | Result | Bound |
|---|---|---|
| Analytic observer vs CIE 1931 table, max squared error `(x, y, z)` | `(1.96e-4, 6.40e-5, 4.91e-4)` | published Table 2, 5% |
| Analytic observer vs CIE 1931 table, mean squared error | `(3.13e-5, 7.12e-6, 1.59e-5)` | published Table 2, 5% |
| Unit reflector under D65, analytic observer, 360-830 nm at 1 nm | `max |RGB - 1| = 2.27e-3` | `3e-3` |
| Unit reflector under D65, analytic observer, 380-780 nm at 5 nm | `2.22e-3` | `3e-3` |
| Unit reflector under D65, CIE table, 360-830 nm at 1 nm | `2.43e-4` | `1e-3` |
| Unit reflector under D65, CIE table, 380-780 nm at 5 nm | `3.54e-4` | `1e-3` |
| Equal-energy white chromaticity deviation from `1/3`, analytic / table | `3.9e-4` / `4.5e-5` | `1e-3` / `1e-4` |
| Airy vs independent characteristic-matrix `R`, `T` (glass, silver-like, silicon-like substrates) | `<= 1.5e-14` | `1e-12` |

So a unit reflector under CIE D65 maps to neutral sRGB white within `3e-3` per
linear channel with the analytic default. Each plan reports its own
`white_neutral_error` for the illuminant and observer actually used.

## Appearance

`ThinFilmAppearancePlan(interference, colorimetry, two_sided=...)` requires both
plans to share the same wavelengths. `evaluate(thickness, normals,
view_directions, illumination_directions=None)` accepts plain arrays that
broadcast to a sample shape `B` (vertex or face fields of any film or foam
solver); directions are unit-normalized, normals point into the ambient medium.

- Without `illumination_directions`, the film reflects a uniform spectral
  environment: the observed path arrives from the mirror direction
  `l = 2 (n . v) n - v`, so the incidence is `cos(theta) = n . v`. The returned
  `illumination_directions` are those mirror directions.
- With directional light, the reflected path is evaluated at the half vector
  `h = normalize(l + v)`, `cos(theta) = h . v`, as in microfacet iridescence
  models (Belcour and Barla, 2017). `facet_alignment = n . h` is one exactly in the
  mirror configuration, the only one in which a smooth film reflects the source
  towards the viewer; shading lobes remain the caller's.
- `two_sided=True` flips normals towards the viewer and requires identical real
  ambient and substrate indices (a free-standing film).

| `ThinFilmAppearanceStatus` bit | Meaning |
|---|---|
| `INVALID_GEOMETRY` | zero or non-finite vectors, or light exactly opposite the viewer |
| `BACK_FACING` | viewer or light on the substrate side of a one-sided film |
| `INTERFERENCE_REJECTED` | interference status nonzero for a usable sample |
| `COLORIMETRY_REJECTED` | colorimetry status nonzero for an accepted spectrum |

Any nonzero interference status masks the spectrum before the colorimetry
quadrature. XYZ, unmapped and display-linear sRGB, and encoded sRGB are
therefore NaN, `colorimetry_status` reports the rejected non-finite spectrum,
and `interference_status` preserves the originating sampling, energy, input, or
non-finite cause. Valid samples in the same batch remain unchanged.

Per-vertex encoded colors can be passed to `SurfaceImagePlan` as a
three-component vertex field to form an exact primary-ray image of a film mesh.
Large images are evaluated in chunks: an evaluation holds `B x W x 2` complex
amplitudes.

`thin_film_surface_colors(appearance, thickness_m, normals, view_directions,
optical_support, illumination_directions)` is the support-aware plain-array
composition for physical surfaces. It returns per-vertex or per-face
`linear_srgb` and `encoded_srgb` fields for `SurfaceImagePlan`, together with
the declared support, the detailed `ThinFilmAppearanceStatus`, and a
`ThinFilmSurfaceColorStatus`. Unsupported samples are masked before optical
evaluation; unsupported and rejected samples remain NaN rather than acquiring
an inferred thickness or plausible display color. Plateau-border and junction
geometry therefore needs its own explicit pixel mask or rendering primitive.

## Example

`examples/advanced_thin_film_iridescence.py` renders an air-soap-air film
(`n = 1.33`) over thickness 0-1500 nm and viewing angle 0-60 degrees under a
6504 K Planckian environment, writes an sRGB PNG to the system temporary
directory and reports acceptance, fringe sampling, energy residual,
out-of-gamut fraction and the illuminant white error (a Planckian 6504 K white
is not D65 and is reported as about `4e-2` from neutral).

`examples/advanced_foam_iridescence.py` composes the same owner with a small
draining spherical B film and E's per-sheet Plateau-border state. Its
double-bubble panel evaluates each manifold sheet with that sheet's slot
thickness and oriented normals, marks a rupture-proposal sheet and Plateau
borders explicitly, writes a PNG, and checks content hashes before and after
rendering. Those hashes demonstrate non-mutation only; rendering does not
certify film thickness or rupture evidence.

## References

- M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University
  Press, 1999, sections 1.6 and 7.6.
- H. A. Macleod, *Thin-Film Optical Filters*, 4th ed., CRC Press, 2010.
- C. Wyman, P.-P. Sloan and P. Shirley, "Simple Analytic Approximations to the
  CIE XYZ Color Matching Functions", *Journal of Computer Graphics Techniques*
  2(2), 2013.
- CIE 015:2018, *Colorimetry*, 4th ed., doi:10.25039/TR.015.2018.
- IEC 61966-2-1:1999, *Default RGB colour space - sRGB*.
- L. Belcour and P. Barla, "A Practical Extension to Microfacet Theory for the
  Modeling of Varying Iridescence", *ACM Transactions on Graphics* 36(4), 2017.

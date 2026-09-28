# Scientific rendering

`phydrax.rendering` maps physical geometry and fields to predicted measurements. It is not a plotting package, interactive viewer, or asset scene graph. Display color exists only as explicit spectral colorimetry with a declared observer, illuminant, and sRGB encoding (see [Spectral colorimetry and thin-film appearance](#spectral-colorimetry-and-thin-film-appearance)).

## Point image formation

The generic Gaussian rasterizer, photometric response, camera-stack image formation, and their support/flux evidence live under `phydrax.rendering`. Velocimetry consumes these operators but no longer owns them.

`GaussianRasterExecutionPlan` selects the existing reference scan or a
tile-major realization. The tiled path emits bounded particle-to-pixel routes,
maps pixels into padded tile-major storage, and reduces them through
`RelationExecutionPlan`. Deterministic and compensated modes preserve
particle-major route order within each pixel. The dense image remains the
scientific output; tiling changes placement and execution only.

`maximum_tile_routes` is an explicit runtime admission. Exceeding it produces
`route_overflow`, a zero unusable image candidate, and an unsuccessful result.
Per-particle support-radius overflow, border clipping, finite-input status, and
deposited flux remain separate evidence channels. Route indices are
stopped-gradient; Gaussian values and coordinates retain fixed-support
derivatives.

## Exact surface images

`SurfaceImagePlan` binds:

- an audited triangular `SurfaceRealization`;
- an `ImagePlaneSupport`;
- a camera;
- field quantity, component layout, and sampling semantics;
- fixed BVH and traversal capacities.

Preparation fixes triangle topology and acceleration routes. Execution accepts current fixed-topology vertices and vertex- or face-associated field values. The result carries the predicted quantity field plus depth, hit mask, primitive and physical entity IDs, barycentric coordinates, world points, normals, facing, uniqueness margins, and `RenderEvidence`.

The triangle BVH is conservatively refitted to current coordinates. Arbitrary fixed-topology motion can reduce traversal efficiency but cannot make successful bounds incomplete.

## Differentiability

Hard visibility is differentiable only while the nearest primitive and route remain unchanged. `uniqueness_margin` and `route_stable` expose that condition. No straight-through visibility gradient or implicit soft rasterizer is used.

A successful render means finite geometry and values, exact visibility for the represented triangles, complete required capacity, and no traversal overflow. A missed ray is a valid exact result, not incomplete coverage.

## Quantity discipline

Rendering outputs `PreparedQuantityField`, not an anonymous image. The caller declares whether the rendered field is temperature, displacement, radiance, detector signal, or another quantity. Colormaps and tone mapping remain presentation concerns unless they are part of a calibrated detector model.

## Spectral colorimetry and thin-film appearance

`SpectralColorimetryPlan` integrates spectral reflectance factors against CIE
1931 color-matching functions and a `SpectralIlluminant`, normalizes a perfect
reflector to `Y = 1`, and maps to linear and IEC 61966-2-1 encoded sRGB with
explicit gamut evidence. `ThinFilmAppearancePlan` composes it with
`phydrax.optics.wave.ThinFilmInterferencePlan` to color film samples from
thickness, normals, and view or illumination directions. A spectrum with any
nonzero interference status is masked before colorimetry, so rejected
undersampled or energy-inconsistent calculations cannot acquire plausible XYZ
or sRGB values; the interference evidence retains the originating status and
finite quality diagnostics.

`thin_film_surface_colors` adds an explicit physics-owned support mask and
returns linear/encoded sRGB surface fields plus status; unsupported and
rejected samples stay NaN. The fields can enter `SurfaceImagePlan`, while
Plateau borders and junctions remain separate supported geometry rather than
receiving invented thickness. Appearance is downstream of physics: it never
alters or certifies film state. The [thin-film optics
guide](guides_thin_film_optics.md) documents observers, illuminant resources,
and error bounds.

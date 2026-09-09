# Scientific rendering

`phydrax.rendering` maps physical geometry and fields to predicted measurements. It is not a plotting package, interactive viewer, asset scene graph, or display-colour system.

## Point image formation

The generic Gaussian rasterizer, photometric response, camera-stack image formation, and their support/flux evidence live under `phydrax.rendering`. Velocimetry consumes these operators but no longer owns them.

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

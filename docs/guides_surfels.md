# Surfels

A Phydrax surfel is a zero-dimensional ownership entity carrying codimension-one
geometry, a finite tangent footprint, and physical surface measure. Surfels do
not carry mesh incidence and do not imply a watertight surface.

Four quantities remain distinct:

- physical surface quadrature weight;
- tangent footprint area or length;
- confidence used by reconstruction;
- application fields such as traction, temperature, radiance, or labels.

The point topology owns stable identity. `SurfelGeometryState` owns current
position, normal, tangent footprint, and physical weight. Morton hierarchies
index the centers and a separate primitive-bounds view contains the complete
footprints.

## Stable surfel support

```python
surfel_plan = phx.discretization.SurfelSetPlan(
    surfel_ids,
    reference_positions,
    reference_surface_weights,
    active_mask=active,
    source_entity_ids=source_ids,
)
surfels = surfel_plan.prepare()
```

`PreparedSurfelDiscretization` uses `PointTopology`. Its reference
`DiscreteMeasure` is surface measure, while its position, normal, and tangent
axis spaces are point-value spaces. Activation changes are topology events.

## Geometry convention

```python
geometry = phx.discretization.SurfelGeometryPlan(surfels).materialize(
    positions,
    normals,
    tangent_axes,
    physical_surface_weights=current_weights,
    certificate=certificate,
)
```

For ambient dimension `d`, `tangent_axes` has shape
`(capacity, d, d - 1)`. Its columns are scaled physical semiaxis vectors. If A
is the tangent-axis matrix, the footprint is the image of the unit tangent disk
or interval under A.

In three dimensions its ellipse area is `π √det(AᵀA)`. In two dimensions its
segment length is `2 √(AᵀA)`. Neither value replaces the physical quadrature
weight.

Materialization reports:

- normal norm defect;
- normal/tangent orthogonality defect;
- tangent Gram eigenvalue and condition;
- footprint measure;
- orientation consistency;
- physical-weight validity;
- finite and successful status.

Invalid active geometry is sanitized for finite compiled execution but remains
unsuccessful. Inactive nonfinite padding is ignored.

## Geometry certificates

`SurfelGeometryCertificate` states:

- source geometry and provenance;
- position and normal accuracy;
- global, component, local, or absent orientation;
- certified, sampled, or unknown coverage;
- quadrature, reconstruction, or acquisition footprint meaning;
- optional position, normal-angle, and curvature bounds;
- one-sided or two-sided interpretation.

A numerically successful surfel set is not automatically a certified surface.
Local tangent patches do not establish global sign, closure, or watertightness.

## Footprint evaluation

`SurfelFootprintPlan` evaluates paired points and surfel slots. It returns:

- signed normal distance;
- tangent coordinates;
- projection onto the tangent plane;
- normalized tangent radius;
- compact C2 reconstruction weight;
- inside, active, finite, solve, and successful masks.

Tangent coordinates use the native small-linear-solve substrate. The compact
kernel is exactly zero outside the footprint.

## Boundary-atlas surfels

```python
plan = phx.geometry.BoundaryAtlasSurfelPlan(
    marker_quadrature,
    footprint_area_ratio=1.25,
)
prepared = plan.prepare(atlas, reference_time)
materialized = prepared.materialize(atlas, time, velocity=boundary_velocity)
```

The atlas supplies positions, oriented normals, tangent frames, surface
Jacobians, source entities, and physical quadrature weights. The initial atlas
adapter uses isotropic footprints whose area or length equals the explicit
footprint ratio times physical quadrature weight.

`BoundaryAtlasSurfelMaterialization.marker_kinematics` maps the same stable IDs
and materialized velocity into a compatible `LagrangianMarkerDiscretization`.
The immersed spreading radius remains independent of the surfel footprint.

## Simplicial surfels

```python
prepared = phx.geometry.SimplicialSurfelPlan(
    triangle_surface,
    footprint_area_ratio=1.0,
).prepare()
```

The current simplicial view creates one centroid quadrature surfel per oriented
triangle. Position, face normal, source ID, and quadrature area are inherited
from the mesh. The elliptical footprint is an approximate filtering patch; it
is not a replacement for the triangle or its incidence.

## Extended primitive bounds

Center ownership alone is insufficient because a surfel can cross its Morton
cell. Build a hierarchy and refit conservative footprint AABBs:

```python
hierarchy = phx.discretization.MortonPointHierarchyPlan(
    address_plan,
    surfels.capacity,
).build(
    geometry.position,
    active_mask=geometry.active_mask,
    stable_ids=surfels.surfel_ids,
)

bounds = phx.discretization.MortonPrimitiveBoundsPlan(
    hierarchy,
    geometry.ambient_dimension,
).refit(
    geometry.position - geometry.footprint_half_width,
    geometry.position + geometry.footprint_half_width,
)
```

Every surfel is stored once in its center leaf. Primitive bounds are reduced
from items to leaves and then from children to parents. A geometry-only update
can refit bounds without changing Morton ownership.

## Ray intersections

```python
query = phx.discretization.SurfelRayQueryPlan(
    bounds,
    geometry,
    maximum_hits_per_ray=8,
)
hits = query.query(ray_origins, ray_directions)
```

Moderate fixed capacities use a batched branchless exact primitive pass.
Larger supports use packed node-AABB traversal before testing surfel planes
and bounded tangent ellipses. Results are ordered by ray distance and stable
surfel ID. Sidedness is read from the geometry certificate.

Hit and traversal capacities are explicit. Overflow returns the nearest stored
hits for diagnostics but marks the ray unsuccessful; it never presents a
truncated hit set as complete.

Route selection and depth ordering are discrete. Hit distance, position, and
local tangent coordinates are recomputed from selected routes so they remain
differentiable while the route is fixed.

## Sparse voxel projection

```python
projection_plan = phx.discretization.SurfelVoxelProjectionPlan(
    sparse_voxel_grid,
    geometry,
    maximum_voxels_per_surfel=max_candidates,
    route_capacity=max_routes,
    normal_distance_support=band_width,
    route_padding=motion_envelope,
)
prepared_projection = projection_plan.prepare(geometry)
field = prepared_projection.project(
    geometry,
    confidence=confidence,
    attributes=surfel_attributes,
)
```

Preparation enumerates active voxel centers inside each padded surfel AABB and
compacts canonical routes. Projection evaluates the actual tangent footprint
and compact normal-distance kernel on those routes.

The result includes:

- local implicit value;
- reconstruction denominator;
- deposited weighted surface measure;
- reconstructed normal and coherence;
- contributor count;
- minimum and maximum signed distances;
- optional reconstructed attributes;
- supported and conflicting masks;
- stale-route and capacity evidence.

The implicit value is a local point-to-plane reconstruction, not an exact SDF.
Unoriented surfels cannot produce signed support. Opposing normals mark a voxel
conflicting rather than cancelling into false certainty. Geometry leaving its
prepared padding envelope marks routes stale.

The initial route builder requires nonperiodic voxel axes. Periodic image
routing is not exposed as an inactive option.

## Surface quadrature

```python
quadrature = phx.discretization.SurfelQuadraturePlan(surfels)
result = quadrature.evaluate(geometry, values)
```

The result provides the physical-weight integral, average, total measure, and
finite/successful evidence. It does not normalize by footprint overlap.

## Differentiation

Smooth over a fixed realized topology and route set:

- positions;
- normals;
- tangent axes;
- physical surface weights;
- footprint coordinates and weights;
- ray hit geometry for selected hits;
- voxel projection values;
- confidence and attributes;
- surface quadrature.

Discrete or nonsmooth:

- Morton ownership and sorting;
- hierarchy rebuild;
- AABB candidate selection;
- ray hit membership and ordering;
- voxel route compaction;
- footprint support membership;
- normal-conflict classification;
- activation and future resampling.

## Choosing the authoritative representation

Use surfels for oriented quadrature, local surface reconstruction, sparse
surface observations, and bounded point-rendering primitives. Use triangles or
BEM topology when incidence, shared-edge continuity, or exact piecewise-planar
ownership matters. Use implicit geometry when global sign and distance
certificates matter. Use particles when the primitive represents material or
volume rather than a codimension-one patch.

Camera EWA rendering, automatic raw-point normal estimation, adaptive
split/merge, multiresolution representatives, exact closest-ellipse distance,
and distributed surfel routing are separate capabilities and are not claimed by
this release.

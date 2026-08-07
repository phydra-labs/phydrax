# Three-dimensional geometry domains

Three-dimensional analytic primitives, simplicial meshes, OCCT B-Reps,
fixed-topology differentiable B-Reps, and reconstructed solids all lower to
`CompiledGeometry`. `phydrax.domain.GeometryDomain` is the thin labeled adapter
used by fields, components, integration, constraints, and sampling.

```python
import phydrax as phx

left = phx.geometry.Sphere((-0.4, 0.0, 0.0), 1.0, feature_id="left")
right = phx.geometry.Box(
    center=(0.4, 0.0, 0.0),
    size=(1.2, 1.2, 1.2),
    feature_id="right",
)
source = (left | right).rotated((0.0, 0.0, 1.0), 0.2)
domain = phx.domain.GeometryDomain(source.compile())
```

`GeometryDomain` exposes the compiled region field, capabilities, field
certificate, volume, boundary measure, boundary atlas, normals, and bounded
sampling without depending on its source representation.

## Mesh and CAD input

`mesh_region_from_source(...)` validates and canonicalizes build123d, meshio,
PyVista, trimesh, or file-backed triangular meshes. Surface meshes must be
finite, nondegenerate, consistently oriented, watertight, and enclose nonzero
volume.

STEP, IGES, and BREP input lowers to `BRepSource`, preserving face patches,
trims, topology identities, and import provenance:

```python
import build123d as bd

model = phx.geometry.model_from_occt_shape(
    bd.Box(1.0, 2.0, 3.0).wrapped,
    linear_deflection=0.1,
)
solid = phx.domain.GeometryDomain(phx.geometry.BRepSource(model).compile())
print(solid.geometry.field_certificate)
print(solid.boundary_atlas.source_entity_ids)
```

Point clouds, DEMs, and LiDAR scenes use explicit reconstruction functions.
Each returns a `ReconstructedGeometrySource`; its report records algorithms,
parameters, filtering, topology checks, approximation counts, warnings, and the
input digest.

## Domain adapter

::: phydrax.domain.GeometryDomain

## Analytic sources

::: phydrax.geometry.Sphere

---

::: phydrax.geometry.Ellipsoid

---

::: phydrax.geometry.Box

---

::: phydrax.geometry.Cube

---

::: phydrax.geometry.Cylinder

---

::: phydrax.geometry.Cone

---

::: phydrax.geometry.Torus

---

::: phydrax.geometry.Wedge

## Simplicial, CAD, and reconstruction sources

::: phydrax.geometry.MeshRegion

---

::: phydrax.geometry.mesh_region_from_source

---

::: phydrax.geometry.BRepSource

---

::: phydrax.geometry.FixedTopologyBRepSource

---

::: phydrax.geometry.reconstruct_surface_region

---

::: phydrax.geometry.reconstruct_dem_region

---

::: phydrax.geometry.reconstruct_lidar_region

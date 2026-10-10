# Three-dimensional geometry domains

Three-dimensional analytic primitives, simplicial meshes, native B-Reps,
fixed-topology differentiable B-Reps, and reconstructed solids all lower to
`CompiledGeometry`. Live OCCT shapes are a separate optional interchange
boundary, not the native B-Rep representation or a query fallback.
`phydrax.domain.GeometryDomain` is the thin labeled adapter used by fields,
components, integration, constraints, and sampling.

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

`mesh_region_from_source(...)` validates and canonicalizes native triangle
arrays, `TriangleMesh`, Meshio data, or Meshio-supported paths. Surface meshes
must be finite, nondegenerate, consistently oriented, watertight, and enclose
nonzero volume.

Native STEP and IGES readers, plus the native codec for supported externally
defined OCCT BRep text profiles, return `CadImportResult.model` while preserving
their exact represented face/edge carriers, trims, topology identities, units,
coverage, and import provenance. The BRep text format name does not imply that
an OCCT runtime was loaded. Pass the model to `BRepSource`; query tessellations
do not replace exact CAD authority. Native construction uses the same model
contract:

```python
model = phx.geometry.brep_box(
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    coordinate_contract=phx.SpatialCoordinateContract.si(),
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

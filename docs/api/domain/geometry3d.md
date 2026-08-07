# Geometry (3D)

Three-dimensional domain constructors are adapters over the common
`phydrax.geometry` substrate. Analytic primitives, simplicial meshes, direct OCCT
B-Reps, fixed-topology differentiable B-Reps, and reconstructed solids all lower to
`CompiledGeometry` before entering labeled domain, integration, and constraint APIs.

A geometry domain exposes one certified negative-inside field through `adf`, region
membership, physical volume and surface measure, outward normals, bounded sampling,
and a representation-independent `boundary_atlas`. `field_certificate` records the
zero-set, sign, distance, regularity, and parameter-differentiability guarantees.

## Direct source composition

Construct transformations and CSG in `phx.geometry`, then adapt the compiled result:

```python
import phydrax as phx

left = phx.geometry.Sphere((-0.4, 0.0, 0.0), 1.0, feature_id="left")
right = phx.geometry.Box(
    center=(0.4, 0.0, 0.0),
    size=(1.2, 1.2, 1.2),
    feature_id="right",
)

source = (left | right).rotated((0.0, 0.0, 1.0), 0.2)
geometry = phx.domain.GeometryDomain(source.compile())
```

Sharp CSG uses `|`, `&`, and `-` on `GeometrySource` objects. Domain adapters
intentionally remain thin and do not duplicate source construction operations.

## Mesh and CAD input

`Geometry3DFromCAD` accepts canonical mesh inputs and OCCT-backed files. Surface
meshes must be finite, nondegenerate, consistently oriented, watertight, and have
nonzero signed volume. Mesh input lowers to `MeshRegion`; STEP, IGES, and BREP input
lowers to `BRepSource`, preserving face patches, trims, topology identities, and the
CAD import report.

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

Point clouds, DEMs, and LiDAR scenes use explicit reconstruction pipelines and expose
an immutable `reconstruction_report` on the returned domain. The report records the
algorithm, parameters, filtering, topology checks, approximation counts, warnings,
and input digest.

::: phydrax.domain.Geometry3DFromCAD
    options:
        members:
            - sample_interior
            - sample_boundary
            - estimate_boundary_subset_measure

---

::: phydrax.domain.Geometry3DFromPointCloud

---

::: phydrax.domain.Geometry3DFromDEM

---

::: phydrax.domain.Geometry3DFromLidarScene

## Primitives

::: phydrax.domain.Sphere

---

::: phydrax.domain.Ellipsoid

---

::: phydrax.domain.Cuboid

---

::: phydrax.domain.Cube

---

::: phydrax.domain.Cylinder

---

::: phydrax.domain.Cone

---

::: phydrax.domain.Torus

---

::: phydrax.domain.Wedge

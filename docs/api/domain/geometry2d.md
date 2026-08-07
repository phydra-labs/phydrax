# Two-dimensional geometry domains

Two-dimensional construction lives in `phydrax.geometry`. Sources are
representation-aware host objects; `compile()` lowers them to a common JAX-safe
`CompiledGeometry`; `phydrax.domain.GeometryDomain` adds a coordinate label and
the domain/component/sampling algebra.

```python
import phydrax as phx

source = phx.geometry.Circle(
    center=(0.0, 0.0),
    radius=1.0,
    feature_id="disk",
)
domain = phx.domain.GeometryDomain(source.compile(), label="x")
```

Compose CSG and transforms before compilation:

```python
left = phx.geometry.Circle((-0.35, 0.0), 0.75, feature_id="left")
right = phx.geometry.Square((0.35, 0.0), 1.0, feature_id="right")
source = (left | right).translated((0.0, 0.25))
domain = phx.domain.GeometryDomain(source.compile())
```

Sharp `|`, `&`, and `-` operations preserve set membership and record their
nonsmooth seam regularity in the compiled field certificate. Domain adapters do
not duplicate source construction operations.

## Planar mesh and reconstruction input

`planar_region_from_source(...)` canonicalizes a triangular mesh as a
`PlanarMeshRegion`. `reconstruct_planar_region(...)` is a separate approximate
pipeline returning a `ReconstructedGeometrySource` with an immutable
`ReconstructionReport`.

```python
import meshio

mesh = meshio.Mesh(
    points=[
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ],
    cells=[("triangle", [[0, 1, 2], [0, 2, 3]])],
)
points = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [0.0, 0.0],
]
mesh_source = phx.geometry.planar_region_from_source(mesh)
mesh_domain = phx.domain.GeometryDomain(mesh_source.compile())

reconstructed = phx.geometry.reconstruct_planar_region(points)
point_domain = phx.domain.GeometryDomain(reconstructed.compile())
print(point_domain.reconstruction_report)
```

## Domain adapter

::: phydrax.domain.GeometryDomain

## Analytic sources

::: phydrax.geometry.Circle

---

::: phydrax.geometry.Ellipse

---

::: phydrax.geometry.Rectangle

---

::: phydrax.geometry.Square

---

::: phydrax.geometry.Polygon

---

::: phydrax.geometry.Triangle

## Simplicial and reconstruction sources

::: phydrax.geometry.PlanarMeshRegion

---

::: phydrax.geometry.planar_region_from_source

---

::: phydrax.geometry.reconstruct_planar_region

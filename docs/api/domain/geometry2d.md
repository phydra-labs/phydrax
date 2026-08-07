# Geometry (2D)

Two-dimensional domain constructors are adapters over the common
`phydrax.geometry` substrate. `phx.domain.Circle(...)`, `Square(...)`, and the
other convenience constructors compile an analytic source and return a
`GeometryDomain`; they do not maintain a second geometry implementation.

A geometry domain exposes one certified negative-inside field through `adf`, exact
or approximate membership through the compiled kernel, a representation-independent
`boundary_atlas`, physical measure, boundary measure, normals, and bounded sampling.
`field_certificate` states whether the field is an exact distance, an approximate
distance, or a general level set.

## Direct source composition

Construct and compose geometry sources before adapting them to a labeled domain:

```python
import phydrax as phx

left = phx.geometry.Circle((-0.35, 0.0), 0.75, feature_id="left")
right = phx.geometry.Square((0.35, 0.0), 1.0, feature_id="right")

source = left | right                 # union
# source = left & right               # intersection
# source = left - right               # difference
source = source.translated((0.0, 0.25))

geometry = phx.domain.GeometryDomain(source.compile())
```

Sharp CSG uses `|`, `&`, and `-` on `GeometrySource` objects. The resulting field is
a nonsmooth level set at operation seams, and its `FieldCertificate` reports that
loss of regularity. Domain adapters intentionally do not duplicate the CSG API.

## Planar mesh input

`Geometry2DFromCAD` canonicalizes a finite planar triangle mesh into
`PlanarMeshRegion`. Triangles may contain holes or multiple connected components;
the oriented boundary loops determine membership and boundary charts. Point-cloud
reconstruction is a separate, reported approximation pipeline.

::: phydrax.domain.Geometry2DFromCAD
    options:
        members:
            - sample_interior
            - sample_boundary
            - estimate_boundary_subset_measure

---

::: phydrax.domain.Geometry2DFromPointCloud

## Primitives

::: phydrax.domain.Circle

---

::: phydrax.domain.Ellipse

---

::: phydrax.domain.Rectangle

---

::: phydrax.domain.Square

---

::: phydrax.domain.Polygon

---

::: phydrax.domain.Triangle

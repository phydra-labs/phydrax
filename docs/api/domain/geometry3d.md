# Geometry (3D)

## Mesh-based geometries

### Boolean / CSG operations

`Geometry3DFromCAD` supports boolean operations via operator overloading:

- `A + B`: union ($\Omega = \Omega_A \cup \Omega_B$)
- `A - B`: difference ($\Omega = \Omega_A \setminus \Omega_B$)
- `A & B`: intersection ($\Omega = \Omega_A \cap \Omega_B$)

!!! example
    ```python
    import phydrax as phx

    # In real workflows you can load meshes from disk via Geometry3DFromCAD("path.stl").
    # For a runnable example without external files, use primitives (they produce CAD-backed geometries).
    A = phx.domain.Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
    B = phx.domain.Cube(center=(0.25, 0.0, 0.0), side=1.2)

    #     U = A + B
    #     D = A - B
    #     I = A & B
    ```

::: phydrax.domain.Geometry3DFromCAD
    options:
        members:
            - __init__
            - __add__
            - __sub__
            - __and__
            - sample_interior
            - sample_boundary
            - translate
            - scale

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

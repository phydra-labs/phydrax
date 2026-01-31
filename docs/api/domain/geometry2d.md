# Geometry (2D)

## Mesh-based geometries

### Boolean / CSG operations

`Geometry2DFromCAD` supports boolean operations via operator overloading:

- `A + B`: union ($\Omega = \Omega_A \cup \Omega_B$)
- `A - B`: difference ($\Omega = \Omega_A \setminus \Omega_B$)
- `A & B`: intersection ($\Omega = \Omega_A \cap \Omega_B$)

!!! example
    ```python
    import phydrax as phx

    # In real workflows you can load meshes from disk via Geometry2DFromCAD("path.stl").
    # For a runnable example without external files, use primitives (they produce CAD-backed geometries).
    A = phx.domain.Circle(center=(0.0, 0.0), radius=1.0)
    B = phx.domain.Square(center=(0.25, 0.0), side=1.0)

    #     U = A + B
    #     D = A - B
    #     I = A & B
    ```

::: phydrax.domain.Geometry2DFromCAD
    options:
        members:
            - __init__
            - __add__
            - __sub__
            - __and__
            - sample_interior
            - sample_boundary

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

# Integral operators

!!! note
    For a more detailed mathematical guide (measures, sampling, quadrature), see
    [Guides → Integrals and measures](../../guides_integrals.md).

!!! note
    When integrating over a `CoordSeparableBatch`, Phydrax uses per-axis quadrature weights
    from `batch.axis_discretization_by_axis` when available (e.g. Gauss–Legendre weights
    from `LegendreAxisSpec`). Otherwise it falls back to uniform per-axis weights based on
    the factor's axis-aligned bounding box.

::: phydrax.operators.build_quadrature

---

::: phydrax.operators.build_ball_quadrature

---

::: phydrax.operators.integral._batch_ops.integral

---

::: phydrax.operators.mean

---

::: phydrax.operators.integrate_interior

---

::: phydrax.operators.integrate_boundary

---

::: phydrax.operators.spatial_integral

---

::: phydrax.operators.local_integral

---

::: phydrax.operators.local_integral_ball

---

::: phydrax.operators.nonlocal_integral

---

::: phydrax.operators.time_convolution

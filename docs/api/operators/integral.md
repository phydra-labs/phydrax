# Integral field operators

!!! note
    Global measure-aware quadrature and sampling live under
    [`phydrax.integration`](../integration.md). See
    [Integrals and measures](../../guides_integrals.md) for the target/plan/estimate
    workflow.

The operators below construct or reduce domain fields inside larger operator
expressions. They are retained for local, nonlocal, spatial, and convolutional field
transforms. New global integrals should use `phydrax.integration.integrate`.

`time_convolution` is a deterministic field operator. It maps a declared fixed
`IntervalRule` onto each causal interval, is exactly zero at the time-domain
start, and records its rule in `DomainFunction.metadata`. Randomized inner
integrals must retain independent realizations and use an estimator-aware
randomized term; they are not exposed as an averaged field that can be squared
silently.

::: phydrax.operators.integral

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

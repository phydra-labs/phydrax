# Continuous constraints

These helpers construct sampled constraints over domain components. For details on reduction,
measures, and filtering, see [Guides → Constraints and objectives](../../guides_constraints.md).

For residual-style continuous constraints, `weight` can be either a scalar global
multiplier or a pointwise `DomainFunction`.

## Interior / initial sampling constraints

::: phydrax.constraints.ContinuousPointwiseInteriorConstraint

---

::: phydrax.constraints.ContinuousInitialConstraint

---

::: phydrax.constraints.ContinuousInitialFunctionConstraint

## Integral constraints

::: phydrax.constraints.ContinuousIntegralInteriorConstraint

---

::: phydrax.constraints.ContinuousIntegralBoundaryConstraint

---

::: phydrax.constraints.ContinuousIntegralInitialConstraint

## ODE constraints

::: phydrax.constraints.ContinuousODEConstraint

---

::: phydrax.constraints.InitialODEConstraint

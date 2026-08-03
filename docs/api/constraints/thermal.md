# Thermal constraints

## Boundary conditions

Heat-flux arguments use the physical outward convention
\(q_n=-k\,\nabla T\cdot n\), where \(n\) is the outward unit normal. Thus
`flux` and discrete `values` are prescribed \(q_n\), and convection enforces
\(q_n=h(T-T_\infty)\). Continuous and discrete helpers use the same sign.

::: phydrax.constraints.ContinuousHeatFluxBoundaryConstraint

---

::: phydrax.constraints.ContinuousConvectionBoundaryConstraint

---

::: phydrax.constraints.DiscreteHeatFluxBoundaryConstraint

---

::: phydrax.constraints.DiscreteConvectionBoundaryConstraint

---

::: phydrax.constraints.DiscreteRobinBoundaryConstraint

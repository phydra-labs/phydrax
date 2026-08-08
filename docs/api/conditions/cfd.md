# CFD conditions

CFD semantics are grouped under `phydrax.conditions.cfd`. Realize each condition
with the generic `phydrax.terms.ResidualPenalty` and an explicit integration
source.

::: phydrax.conditions.cfd.NoPenetration

---

::: phydrax.conditions.cfd.SlipWall

---

::: phydrax.conditions.cfd.SymmetryVelocity

---

::: phydrax.conditions.cfd.ZeroNormalGradientVelocity

## Conservation moments

Integral flow and pressure requirements are grouped under
`phydrax.conditions.conservation` and use `phydrax.terms.MomentPenalty` with
`phydrax.integration.over(condition.on)`.

::: phydrax.conditions.conservation.FlowRate

---

::: phydrax.conditions.conservation.KineticEnergyFlux

---

::: phydrax.conditions.conservation.PressureIntegral

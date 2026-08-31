# Material point method

## Spatial method and runtime state

::: phydrax.discretization.mpm
    options:
      show_root_heading: false
      members_order: source

## Problem declaration and compilation

::: phydrax.equations.AbstractMPMConstitutivePlan

---

::: phydrax.equations.MPMConstitutiveResponse

---

::: phydrax.equations.MaterialPointArguments

---

::: phydrax.equations.MaterialPointProblemIR

---

::: phydrax.equations.CompiledMaterialPointProblem

---

::: phydrax.equations.compile_material_point_problem

## Fixed-temporal rollout

::: phydrax.solver.MPMReplayPolicy

---

::: phydrax.solver.ScheduledMPMRolloutPlan

---

::: phydrax.solver.MPMRolloutResult

---

::: phydrax.solver.MPMGradientReport

## Baseline solid material

::: phydrax.applications.solid_mechanics.NeoHookeanParameters

---

::: phydrax.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan

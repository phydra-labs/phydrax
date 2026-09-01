# Free-surface FLIP

## Spatial method and runtime state

::: phydrax.discretization.flip
    options:
      show_root_heading: false
      members_order: source

## Problem declaration and compilation

::: phydrax.equations.FLIPProblemIR

---

::: phydrax.equations.CompiledFLIPProblem

---

::: phydrax.equations.compile_flip_problem

## Atmospheric MAC projection

::: phydrax.solver.MACFreeSurfaceProjectionPlan

---

::: phydrax.solver.MACFreeSurfaceProjectionResult

## Fixed-step adapter

::: phydrax.solver.FLIPFixedStepMethod

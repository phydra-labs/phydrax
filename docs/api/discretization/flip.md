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

`FLIPProblemIR.viscous_law` is `GeneralizedNewtonianLaw | MuILaw | None`.
`mu_i_parameters` is required only for `MuILaw`; it remains dynamic.
`CompiledFLIPProblem.initialize_state(initial_pressure=...)` requires a finite
atmospheric-gauge cell field for μ(I), with zero air pressure and actual free
surface support. Subsequent stress stages consume previous accepted pressure.

Viscous FLIP includes the accepted pressure gradient in its implicit momentum
predictor, then projects a **pressure increment** and adds it to accepted
physical pressure. This first-order backward-Euler incremental scheme preserves
hydrostatic pressure support independently of viscosity. The inviscid path keeps
the original absolute-pressure projection. See the [scheme and primary references](../../guides_flip.md#implicit-viscous-stress).

`FLIPMethodPlan` bounds Picard and inner Krylov work separately. Nonconvergence
or undefined closure rejects the complete particle/pressure transaction.
`FLIPStepResult.viscous` is `None` on the unchanged inviscid path, otherwise a
`FLIPViscousEvidence` with closure, normalized true momentum residual, nonlinear
iteration/work, native Krylov status, and energy/dissipation evidence.
`compile_sparse_flip_problem` explicitly refuses viscous laws.

::: phydrax.discretization.flip.FLIPViscousEvidence

## Atmospheric MAC projection

::: phydrax.solver.MACFreeSurfaceProjectionPlan

`project(..., incremental=True, pressure=p_n)` solves a correction with zero
atmospheric/homogeneous pressure boundary data, returning total pressure in
`pressure` and the solved correction in `pressure_increment`. Physical velocity
and sharp-wall data are unchanged. Its caller must already apply the old pressure
force in the predictor; the default `incremental=False` remains the absolute solve.

---

::: phydrax.solver.MACFreeSurfaceProjectionResult

## Fixed-step adapter

::: phydrax.solver.FLIPFixedStepMethod

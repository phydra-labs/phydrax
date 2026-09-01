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

::: phydrax.equations.MPMConstitutiveCapabilities

---

::: phydrax.equations.MPMLinearizedConstitutiveResponse

---

::: phydrax.equations.LocalConstitutiveRootPlan

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

## Adaptive, implicit, and fracture solvers

::: phydrax.solver.MPMAdaptivePolicy

---

::: phydrax.solver.AdaptiveMPMRolloutPlan

---

::: phydrax.solver.ImplicitMPMMethodPlan

---

::: phydrax.solver.PreparedImplicitMPMDynamics

---

::: phydrax.solver.MPMPhaseFieldFracturePlan

---

::: phydrax.solver.PreparedMPMPhaseFieldDynamics

## Baseline solid material

::: phydrax.applications.solid_mechanics.NeoHookeanParameters

---

::: phydrax.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.PlaneStressMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.FiniteStrainJ2Parameters

---

::: phydrax.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.MPMPhaseFieldParameters

---

::: phydrax.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan

## Commercial claims, execution, and derivatives

::: phydrax.discretization.mpm.MPMIntendedUse

---

::: phydrax.discretization.mpm.MPMClaimTuple

---

::: phydrax.discretization.mpm.MPMSupportMatrix

---

::: phydrax.discretization.mpm.MPMExecutionPlan

---

::: phydrax.discretization.mpm.MPMGradientResult

---

::: phydrax.discretization.mpm.KWayMPMContactPlan

---

::: phydrax.discretization.mpm.MPMDistributedPlan

---

::: phydrax.discretization.mpm.MPMParticleLifecyclePlan

---

::: phydrax.discretization.mpm.MPMAMRPlan

## Commercial runtime

::: phydrax.solver.MPMCheckpointPlan

---

::: phydrax.solver.MPMOutputPlan

---

::: phydrax.solver.MPMRunSupervisor

---

::: phydrax.solver.MPMImplicitTopologyPlan

---

::: phydrax.solver.MPMCompactImplicitOperator

## Commercial materials and coupled fields

::: phydrax.applications.solid_mechanics.GeneralPlaneStressMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.PreparedMPMCoupledFieldOperator

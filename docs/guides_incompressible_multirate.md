# Incompressible and multirate workflows

## Pressure correction

`phydrax.applications.incompressible_flow` wires supplied advection, velocity
Helmholtz, divergence, pressure Poisson, and gradient operators into an accepted
pressure-correction step. Both pressure and pressure-increment forms are
explicit policies.

A nonincremental step performs:

```text
advection RHS
→ velocity Helmholtz solve
→ divergence
→ pressure Poisson solve
→ velocity gradient correction
→ assessment
```

An incremental step includes the accepted pressure gradient in the velocity
RHS, solves for a pressure increment, updates pressure, and corrects velocity
with the increment gradient.

The application package does not define another linear or nonlinear solver.
`IncompressibleFlowOperators` receives prepared callable actions and
`incompressible_flow_schedule` connects them to the accepted-step runtime.

The schedule was informed by libParanumal's
[incompressible split step](https://github.com/paranumal/libparanumal/blob/main/solvers/ins/src/insStep.cpp)
and [advection subcycling](https://github.com/paranumal/libparanumal/blob/main/solvers/ins/src/insSubcycle.cpp).

## OIFS history combination

`oifs_history_combination` forms the normalized partial history sum from oldest
to newest contributions. A supplied subcycle action may evolve the combined
state while evaluating time-dependent boundary data consistently. Time
integration remains an explicit application policy.

## Multirate DG traces

`DGMultirateTracePlan` declares power-of-two side-rate levels.
`DGTraceHistory` stores accepted trace values and predicts traces at intermediate
times with polynomial interpolation. Rejected updates leave history unchanged.

`conservative_multirate_flux` evaluates one shared numerical flux and returns
exactly opposite plus/minus contributions together with a conservation defect.
This follows the interface-history concept in libParanumal's
[multirate AB3 implementation](https://github.com/paranumal/libparanumal/blob/main/libs/timeStepper/timeStepperMRAB3.cpp)
without adding a duplicate MRAB integrator.

## Relation to LES

This callback-oriented pressure-correction workflow does not infer or insert an SGS
model. Implemented periodic spectral, MAC, channel, ocean, distributed, and
unstructured LES routes use their typed compiler/adapters and evidence described in
[Large-eddy simulation](guides_large_eddy_simulation.md). A closure rate supplied as
an arbitrary callback does not acquire a `ResolvedLESFilter`,
`LESParameterProvenance`, prepared-action identity, energy ledger, or qualification.

## Derivative scope

Pressure-correction and subcycle branching describe the executed algorithm.
Initial-guess histories, rate categories, history eviction, and acceptance
branches are algorithmic decisions and are not smooth model parameters.
Differentiable workflows must choose between executed-schedule differentiation
and an independently compiled monolithic implicit formulation.

## Current scope

The workflow is fixed-mesh and single-device. Distributed trace exchange,
partitioning, and mesh adaptation are not introduced.

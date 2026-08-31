# Reactive CFD–DEM

`ReactiveCFDDEMCouplingPlan` composes prepared spherical DEM dynamics, prepared particle-conversion dynamics, and conservative particle–grid exchange. It is a single-process fixed-capacity contract. Fluid discretization and its update remain user supplied.

## Exchange channels

`ParticleContinuumExchangePlan` gathers cell temperature and species concentration with a prepared `ConservativeParticleGridTransfer`. It computes particle heat and species sources, then deposits their exact opposites as extensive cell sources using the same normalized weights.

`ParticleContactExchangePlan` uses resolved contact overlap and effective contact radius to exchange heat between internal particle surface states. Pair contributions are reciprocal. The contact exchange result reports its energy residual and explicit restriction separately from mechanical contact dissipation.

`ReciprocalPairRadiationPlan` adds reciprocal particle–particle radiation and optional wall radiation. View factors, emissivity, and areas are explicit inputs. Radiation is never inferred from contact state.

## Macro-window schedule

`advance_reactive_cfd_dem_window` executes one atomic macro window:

1. sample fluid fields at particle support;
2. evaluate continuum exchange;
3. advance a half conversion step;
4. execute the configured DEM substeps;
5. advance the second conversion half-step;
6. deposit extensive fluid momentum, energy, and species sources;
7. evaluate morphology and optional radiation;
8. accept every subsystem together or roll all of them back.

`ReactiveParticleCouplingSchedulePlan` selects `STRANG_FROZEN_FLUID` or `ITERATED_STAGGERED` coupling through `ReactiveCouplingMode`. Iterated coupling repeats the coupled map a fixed number of times and requires the declared residual tolerance. It is not a monolithic nonlinear solve.

```text
schedule = phx.solver.ReactiveParticleCouplingSchedulePlan(
    phx.solver.ParticleConversionSolverPlan(
        phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE
    ),
    dem_substeps=4,
    mode=phx.solver.ReactiveCouplingMode.STRANG_FROZEN_FLUID,
)
```

## Fluid callback contract

The sampling callback returns `ReactiveFluidFields`: velocity, density, viscosity, vorticity, temperature, and species concentration at transfer cells. The fluid-update callback receives the current opaque fluid state plus extensive momentum, energy, and species sources and the macro-step size.

The coupling layer checks returned trees for finite values. It does not assume a finite-volume, finite-element, or spectral fluid state layout.

## Balance and rollback

`ReactiveCFDDEMEvaluation` exposes momentum, energy, and species residuals; conversion, DEM, continuum, contact, morphology, and radiation success flags; coupling residual; and rejection reason bits. An invalid fluid callback, failed particle conversion, invalid contact step, exhausted capacity, balance defect, or unconverged iterated schedule preserves the previous `ReactiveCFDDEMState` exactly.

Accepted-window ledgers distinguish boundary heat, contact heat, radiation, continuum species exchange, reaction energy, and phase-change energy. Mechanical dissipation does not become heat unless a model supplies that transfer explicitly.

## Replay, inverse problems, and UQ

`checkpointed_reactive_cfd_dem_rollout` records route digests and rematerializes accepted blocks for reverse differentiation. `reactive_cfd_dem_vjp` requires deterministic replay. `evaluate_reactive_parameter_ensemble` retains invalid-member rates rather than dropping failed worlds.

## Scope

Supported: nondistributed unresolved coupling, radial intraparticle heat/species transport, reactions, evaporation, morphology, contact heat, and radiation. Unsupported: distributed ownership, dynamically growing arrays, turbulence modulation closures, added mass, and monolithic fluid–particle Newton solves.

Run `examples/reactive_cfd_dem.py` and `tools/reactive_cfd_dem_qualification.py` for the complete callback and balance contracts.

# Wave dark matter: finite differences and AMR

The global periodic Fourier solver remains the uniform-grid spectral authority. The AMR
family is a separate pure-complex-wave profile; it does not silently exchange state
with particles or a Hamilton--Jacobi/Madelung fluid.

## External-potential wave actions

`PreparedPeriodicWaveDarkMatter` exposes `density`, `kinetic_drift`, and
`potential_kick`. These actions retain the established convention `phi = a Phi` and
exact endpoint scale-factor factors. They let mixed cosmology consume one externally
owned potential without invoking a second wave Poisson solve.

## Periodic finite-difference authority

`PeriodicWaveFiniteDifferencePlan` binds a complex cell-centered field, a
volume-paired self-adjoint negative Laplacian, periodic cosmology, and a fixed increasing
scale schedule. `PreparedPeriodicWaveFiniteDifference` advances the kinetic subflow with
a global Cayley action through `phydrax.linalg`, applies explicit potential phases, and
returns solve, norm, energy, phase, finite and rollback evidence.

`WaveContactSelfInteractionPlan` is a separately identified Gross--Pitaevskii contact
profile. Its nonlinear phase, dealiasing, energy and collapse-resolution gates do not
alter the zero-coupling profile identity.

## Complex AMR state

`ComplexCompositeAMRCellLayout` lifts the existing real composite AMR geometry to native
complex fields without changing coordinate topology. Complex FillPatch uses the existing
prepared FD hierarchy and reports source/coverage evidence.

`WaveAMRDiscretizationPlan` consumes one `PreparedFDAMRHierarchy` and binds the composite
leaf-cell volume inner product and operator. `WaveAMRPhysicsPlan` binds boson/cosmology,
time, solve and acceptance policy. `PreparedWaveAMR` performs one hierarchy-wide
volume-paired Cayley solve and one existing composite Poisson solve. Patchwise independent
Crank--Nicolson solves are not an admitted alternative.

The initial admitted profile uses synchronized levels and no temporal subcycling.

## Topology adaptation

`WaveAMRAdaptivityPlan` evaluates de Broglie, phase, density, quantum-potential and
vortex indicators with explicit hysteresis and fixed topology capacity. A proposal is
compiled and transferred only at an accepted synchronization boundary.

Complex transfer reports:

- represented-probability defect;
- current defect;
- phase defect;
- node/vortex intersection;
- winding before/after;
- topology and transfer identities.

Smooth nodeless regions may use density/phase transfer. Node or vortex regions use
real/imaginary transfer and fail if probability/current/winding gates are not met. No
derivative passes through regridding, phase unwrapping, winding, node ownership or
capacity decisions.

## Distributed and boundary profiles

`PreparedDistributedWaveAMR` binds the existing distributed block hierarchy to packed
owner-computes state, exact ppermute halo/FillPatch routes, collective projected Poisson
and Cayley solves, global norm/current/self-adjoint/energy evidence, and all-rank atomic
rollback. Accepted-boundary topology successors run phase-aware owner-local transfer,
stable-ID repartition, exact resource admission, and changed-sharding checkpoint
restore. There is no host-gather fallback.

Periodic and isolated boundaries are separate descriptors. `IsolatedPotentialGauge`
requires an explicit finite-domain/multipole convention. `AbsorbingWaveBoundaryPolicy`
reports removed probability and rejects excessive absorption. Isolated physics is never
relabeled as periodic mean-zero gravity.

## Required evidence

- weighted self-adjoint residual;
- Cayley linear residual and norm defect;
- composite Poisson and zero-mode/gauge residual;
- interface probability current and reflection;
- de Broglie and phase resolution;
- topology transfer probability/current/winding defects;
- solver, capacity and resource status;
- complete rollback on any failed stage.

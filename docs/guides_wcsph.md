# Weakly compressible SPH

Phydrax weakly compressible SPH is a first-order particle method with explicit
choices for density state, physical viscosity, neighborhood execution, and SSP
integration. It remains separate from the conservative position--momentum
Hamiltonian SPH path.

## State layouts

`SummationDensityPlan` packs

```text
[position(d), velocity(d)]
```

and recomputes density from the current particle geometry. `ContinuityDensityPlan`
packs

```text
[position(d), velocity(d), density]
```

and evolves density. When continuity density is selected and no initial density
is supplied, the compiler performs one explicit kernel-summation initialization.
It never silently resets density later.

## Pair equations

For pair displacement `rᵢⱼ = qᵢ - qⱼ` and kernel gradient `∇ᵢWᵢⱼ`, the
pair-once continuity rate uses

```text
sᵢⱼ = (vᵢ - vⱼ) · ∇ᵢWᵢⱼ
ρ̇ᵢ += mⱼ sᵢⱼ
ρ̇ⱼ += mᵢ sᵢⱼ.
```

Pressure uses the same symmetric conservative exchange as barotropic SPH:

```text
Gᵢⱼ = mᵢmⱼ (pᵢ/ρᵢ² + pⱼ/ρⱼ²) ∇ᵢWᵢⱼ
Fᵢ += -Gᵢⱼ
Fⱼ +=  Gᵢⱼ.
```

For inviscid continuity-density SPH, the pressure kinetic power and barotropic
internal-energy rate cancel semi-discretely. This balance is a permanent test and
a runtime diagnostic.

## Morris physical viscosity

`MorrisViscosityPlan` accepts constant kinematic viscosity and a positive kernel
regularization. It constructs one equal/opposite force per unordered pair. For a
monotone kernel, pair kinetic power is non-positive. Diagnostics report viscous
power, non-negative dissipation, positive-power defect, net force, and torque.
The formulation claims linear-momentum conservation, not exact angular-momentum
conservation.

## Compilation

```text
method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
    phx.discretization.WendlandC2SPHKernel(2),
    smoothing_length=h,
    density=phx.discretization.ContinuityDensityPlan(),
    viscosity=phx.discretization.MorrisViscosityPlan(0.01),
)
compiled = phx.equations.compile_weakly_compressible_sph_problem(
    phx.equations.WeaklyCompressibleFluidProblemIR(
        "fluid",
        phx.equations.TaitBarotropicMaterial(1.0, 10.0),
    ),
    particles,
    method,
    neighborhood=cell_neighborhood,
)
problem = compiled.as_differential_problem(
    initial_position,
    initial_velocity,
    t0=0.0,
    t1=1.0,
)
solution = phx.solver.solve_diffrax(
    problem,
    save_times=save_times,
    solver=phx.solver.SSPRK33(),
    dt0=dt,
)
```

`SSPRK54` is also supported. Every SSP stage rebuilds its fixed-capacity
neighborhood and refuses cell overflow, pair overflow, or a nonperiodic-domain
violation.

## External acceleration

`WeaklyCompressibleFluidProblemIR` may own a nonconservative acceleration callable
with signature

```text
acceleration(time, position, velocity, density, args).
```

It requires a stable ID. Diagnostics report total external force and power; no
conservation claim hides external work.

## Step restrictions

The method reports acoustic, force, and viscous restrictions separately. The
selected bound is their minimum. The spatial method does not silently adapt the
timestep.

## Differentiation

The derivative contract is the fixed discrete trajectory. Cell IDs, sorting,
packed routes, kernel-support decisions, and periodic images are branchwise.
Positions, velocity, density, material evaluation, pair geometry, pressure,
viscosity, and selected-route reductions remain differentiable. Fixed timesteps
are the inverse-problem default.

## Runtime emission

WCSPH uses the shared `SPHParticleSourcePlan` and `SPHRuntimeState`. Continuity
density sources initialize the declared positive source density; summation
density sources validate it and let the existing density operator recompute the
accepted value. Source mass remainder plus population mass is exact, while
momentum and barotropic reset energy have explicit cumulative ledgers. Activity
changes occur only at the accepted event boundary and invalidate cached
neighborhood/splat routes before the next epoch.

## Current limits

The method has one barotropic material, fixed smoothing length, periodic or
unbounded interactions, and optional Morris physical viscosity. It does not yet
include wall particles, free-surface correction, delta-SPH, artificial
viscosity, density renormalization, transport velocity, adaptive h, multiphase
flow, IISPH, or DFSPH.

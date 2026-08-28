# Smoothed particle hydrodynamics

The first Phydrax SPH method is fixed-h conservative barotropic SPH with
summation density. It is a complete Hamiltonian spatial discretization rather
than a graphics-oriented time-step routine.

## Discrete model

For particle masses mᵢ, current positions qᵢ, and smoothing length h, density is

```text
ρᵢ(q) = Σⱼ mⱼ W(‖qᵢ − qⱼ‖, h).
```

A barotropic material provides pressure p(ρ), sound speed c(ρ), and specific
internal energy e(ρ) satisfying

```text
de/dρ = p(ρ) / ρ².
```

The discrete potential and kinetic energies are

```text
V(q) = Σᵢ mᵢ e(ρᵢ(q)),
T(p) = Σᵢ ‖pᵢ‖² / (2mᵢ),
```

where canonical momentum is pᵢ = mᵢvᵢ. `PreparedBarotropicSPHDynamics` evaluates
the analytic symmetric pressure gradient once per unordered pair. Its permanent
qualification compares that gradient with automatic differentiation of V.

## Kernels

`WendlandC2SPHKernel` and `CubicSplineSPHKernel` use q = r/h and support radius
2h. Both expose value, radial derivative, spatial gradient, smoothing-length
derivative, normalization dimension, support factor, and cutoff regularity.
Coincident-particle gradients are finite and exactly zero.

## Compilation and integration

```python
import jax.numpy as jnp
import phydrax as phx

count = 16
spacing = 1.0 / count
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(count),
    jnp.full((count,), spacing),
    ambient_dimension=1,
).prepare()

problem = phx.equations.BarotropicFluidProblemIR(
    "periodic-fluid",
    phx.equations.TaitBarotropicMaterial(1.0, 1.0),
)
method = phx.discretization.BarotropicSPHMethodPlan(
    phx.discretization.WendlandC2SPHKernel(1),
    1.25 * spacing,
)
compiled = phx.equations.compile_barotropic_sph_problem(
    problem,
    particles,
    method,
    neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
        search_radius=2.5 * spacing,
        maximum_particles_per_cell=8,
        maximum_pairs=4 * count,
        box=phx.discretization.ParticleBox([0.0], [1.0]),
    ),
)

position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
velocity = jnp.zeros_like(position)
ivp = compiled.as_differential_problem(
    position,
    velocity,
    t0=0.0,
    t1=0.01,
)
solution = phx.solver.solve_diffrax(
    ivp,
    save_times=jnp.asarray([0.0, 0.01]),
    solver=phx.solver.StormerVerlet(1),
    dt0=1.0e-4,
)
```

The compiler packs `[position, momentum]` on the trailing state axis, binds the
particle, neighborhood, and method records into one `DiscretizationBundle`, and
sets the canonical Euclidean state geometry required by `StormerVerlet`.

Use `DenseParticleNeighborhoodPlan` as the O(N²) correctness authority.
`CellListParticleNeighborhoodPlan` uses fixed particle-per-cell and edge
capacities. Its runtime `ParticleNeighborhoodState` reports actual pair count,
maximum cell occupancy, cell overflow, pair overflow, and nonperiodic-domain
violations. SPH refuses to evaluate an unsuccessful relation. The search radius
must cover the full kernel support radius.

`compiled.dynamics.graph_view(position)` exposes the exact kernel-supported
interaction graph as directed or undirected `GraphIR`, preserving logical
particle IDs and padding masks.

## Evidence and diagnostics

Prepared dynamics expose:

- density and pair counts;
- internal, kinetic, external-potential, and total energy;
- linear and angular momentum;
- net internal force and torque;
- material admissibility;
- acoustic and force step restrictions;
- JVP and VJP linearization;
- preparation, resource, precision, and bundle identities.

An optional external potential must be scalar, conservative, and carry a stable
ID. Arbitrary accelerations are intentionally excluded from this Hamiltonian
route.

## Differentiability

Gradients are derivatives of the implemented fixed-step discrete trajectory.
Kernel support, periodic images, cell IDs, sorting, and packed edge choices are
almost-everywhere branches; geometry and kernel weights remain differentiable
on the selected routes. Use a fixed step schedule for inverse problems and avoid
configurations exactly on a cell boundary, kernel cutoff, or half-box tie.
`RecursiveCheckpointAdjoint` remains available through the standard temporal
solver interface.

## Current limits

This method has one barotropic material, one particle set, fixed h, summation
density, no viscosity, and no walls or free-surface correction. Neighborhoods
are rebuilt for every evaluation; cached Verlet lists and fused cell traversal
are not yet implemented. First-order summation/continuity density and Morris
physical viscosity are provided by the distinct
[weakly compressible SPH](guides_wcsph.md) contract. Neither method yet claims
delta-SPH, adaptive h, compressible-energy SPH, IISPH/DFSPH, rigid coupling, wall
particles, or particle emission.

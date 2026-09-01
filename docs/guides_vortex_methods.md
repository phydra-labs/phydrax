# Vortex methods

Phydrax represents vorticity carriers as fixed-capacity, identity-bearing sources whose
positions and strengths are dynamic arrays. The method is circulation based: particle
mass from a `ParticleDiscretization` is support metadata and is never interpreted as
vortex strength.

```text
ParticleDiscretization       stable IDs, active mask, capacity
VortexParticleProperties     core radius and area/volume
VortexParticleStateLayout    position plus integrated vorticity
velocity plan                free-space direct or periodic VIC
optional diffusion plan      particle-strength exchange
VortexParticleMethodPlan     one fixed-topology differential program
```

The smallest public calculation is a prepared field evaluation.
`examples/vortex_particle_pair.py` exercises the Gaussian direct plan.
`examples/vortex_periodic_vic.py` and `examples/vortex_pse_diffusion.py` exercise
the distinct periodic and conservative-diffusion routes.

## State, dimensions, and units

Phydrax does not attach a unit package to arrays. Every array in one problem must use a
single consistent unit system. If length is \(L\) and time is \(T\), the public state
has these meanings:

| Quantity | Shape | Dimension |
| --- | --- | --- |
| position | `(capacity, dimension)` | \(L\) |
| 2-D strength | `(capacity,)` | circulation, \(L^2/T\) |
| 3-D strength | `(capacity, 3)` | integrated vorticity, \(L^3/T\) |
| core radius | scalar or one value per source | \(L\) |
| 2-D particle area | one value per source | \(L^2\) |
| 3-D particle volume | one value per source | \(L^3\) |
| velocity | `(..., dimension)` | \(L/T\) |
| velocity gradient and vorticity | `(..., dimension, dimension)` or dimension-specific curl shape | \(1/T\) |
| kinematic viscosity | scalar | \(L^2/T\) |

A 3-D particle strength is the volume integral of vorticity over that carrier. It is
not pointwise vorticity. Likewise, a 2-D particle strength is circulation, not a
probability weight or material mass. PSE uses the separately supplied particle volume
to turn a pointwise exchange approximation into conservative integrated-strength
rates.

Inactive capacity slots remain finite and inert. They are part of the compiled array
shape but contribute neither fields nor diagnostics. Activation, deactivation,
shedding, remeshing, and tree rebuilding are explicit state transitions outside a
smooth ODE right-hand side.

## Field requests and evidence

`VortexFieldRequest` chooses velocity, velocity gradient, and reconstructed vorticity
at preparation/evaluation boundaries. Expensive unrequested fields are not silently
computed. A `VortexVelocityEvaluation` retains the optional arrays, a success flag,
backend and evaluation identities, and backend-specific diagnostics.

Success is necessary but is not an accuracy theorem. Inspect the evidence appropriate
to the backend:

| Backend | Evidence to inspect | What it does not prove |
| --- | --- | --- |
| Gaussian direct | finite state, source/target mapping, interaction and chunk budgets | continuum convergence at one core size |
| periodic VIC | transfer balance/partition, spectral divergence, imaginary leakage, zero-mode compatibility | equality with a free-space image sum |
| accelerated direct | accepted fixed tree, direct-near accounting, estimated far error, direct parity when requested | gradients through a tree rebuild |
| PSE diffusion | circulation defect, pair/search capacity, cutoff and kernel-moment evidence, stable-step restriction | accuracy after an unresolved particle layout |

Prepared objects and results carry stable identities. Reusing evidence with a different
source layout, grid, target set, request, or policy is therefore not supported.

## Free space is not periodic space

`GaussianDirectVortexPlan2D` and `GaussianErfDirectVortexPlan3D` evaluate regularized
free-space Biot--Savart sums. They do not apply minimum-image displacement. Passing
points from a periodic box to a direct plan still computes the free-space problem.

`PeriodicVortexInCellPlan` is the periodic route. It deposits integrated vorticity on
a matched periodic tensor grid, performs Fourier inversion with the zero mode treated
explicitly, and gathers requested fields. A periodic velocity cannot represent an
incompatible mean-vorticity mode. In 2-D this requires zero total circulation within
tolerance; in 3-D it requires each component of total integrated vorticity to be
compatible. Rejection is evidence, not an implicit subtraction of the mean.

The periodic method is a particle--mesh discretization. Its transfer kernel, grid,
spectral truncation, and gather are part of the numerical method. It must not be
described as an exact Gaussian-direct calculation or as a minimum-image shortcut.

## Core and filter semantics

The core radius regularizes Biot--Savart evaluation at source coincidence:

- the 2-D Gaussian blob uses the Gaussian-smoothed point-vortex kernel;
- the 3-D Gaussian--erf blob uses a radially regularized vector Biot--Savart kernel;
- distinct coincident sources remain finite and are not discarded;
- self exclusion is identity based through explicit target-to-source indices, never a
  distance test.

A core is not a particle mass, a collision radius, a neighbor-search box, or a proof of
spatial resolution. The direct core and the VIC assignment filter are different
objects. PSE has another normalized Gaussian exchange kernel and a finite cutoff.
Changing any of these changes the discrete method.

`GaussianParticleStrengthExchangePlan` constructs symmetric volume-weighted pair
fluxes. Integrated strength is conserved by paired scatter even for unequal positive
volumes. Periodic minimum-image geometry is used only for this compact diffusion
operator when its neighborhood explicitly declares periodic support.

## Two-dimensional dynamics

For scalar strengths \(\Gamma_i\), the inviscid fixed-particle-strength equations are

\[
  \dot{x}_i = u(x_i), \qquad \dot{\Gamma}_i = 0.
\]

A compiled problem may add a provenance-bearing background velocity and a prepared
diffusion rate. `VortexParticleFlowProblem`, `VortexParticleMethodPlan`, and
`compile_vortex_particle_flow` bind those choices before creating the ordinary
Phydrax `DifferentialProblem` adapter. The backend restriction, diffusion restriction,
finite-state status, total circulation, and impulse diagnostics remain observable.

## Three-dimensional stretching

In three dimensions the classic vortex-particle state evolves as

\[
  \dot{x}_i = u(x_i), \qquad
  \dot{\boldsymbol{\Gamma}}_i = (\nabla u(x_i))
  \boldsymbol{\Gamma}_i + \dot{\boldsymbol{\Gamma}}_{i,\nu}.
\]

The stretching product uses the requested analytic velocity gradient. Diffusion, when
present, is composed as a separate rate.
`examples/vortex_3d_stretching.py` exposes total vector strength, impulse, and
stretching evidence rather than treating a finite output as sufficient validation.

The classic equation does not silently apply strength relaxation, core spreading, or a
reformulated VPM correction. Reformulated strength/core evolution and relaxation are
opt-in advanced operators with their own compatibility and conservation diagnostics.

## Fixed-topology differentiation

JAX differentiation applies to the fixed discrete numerical program:

- source positions, strengths, core radii, kernel arithmetic, grid transfer weights
  within a fixed route, Fourier operations, and smooth background callbacks are
  differentiable;
- source activation, compact-support membership, cell routing, self-index mappings,
  wake shedding, remeshing, neighbor rebuilding, and accelerator tree construction are
  discrete decisions;
- derivatives through an accepted fixed accelerator topology do not include the
  derivative of rebuilding that topology;
- no straight-through estimator is supplied for those decisions.

For inverse problems, prepare topology and capacity before differentiation, keep the
same active-set and route decisions across the differentiated evaluation, and reject a
sample when its evidence reports that those assumptions failed.

## Complete advanced workflows and remaining mathematical boundaries

Field execution includes direct, periodic Ewald, free-space FFT, corrected P3M,
hierarchical FMM, and explicit sharding. Actuator sources and passive probes
remain distinct. Native rigid/flexible FSI replaces the former standalone Euler
wrapper. Random vortices advance complete ensembles with boundary and variance
policies. Learned vorticity uses native optimization, incompressible
reconstruction, assimilation, and constrained closure evidence.

Fixed-program gradients, event pullbacks, and transversal saltation maps are
available. Event selection, arbitrary reconnection choices, and capacity-policy
branching are not represented as ordinary smooth derivatives. A universal
turbulence closure, exact under-resolved loads, and a purely solenoidal
compressible-flow model remain intentionally unclaimed.

## Primary method references

- Chorin, "Numerical study of slightly viscous flow," *Journal of Fluid Mechanics*
  57 (1973), [doi:10.1017/S0022112073000874](https://doi.org/10.1017/S0022112073000874).
- Leonard, "Vortex methods for flow simulation," *Journal of Computational Physics*
  37 (1980), [doi:10.1016/0021-9991(80)90040-6](https://doi.org/10.1016/0021-9991(80)90040-6).
- Beale and Majda, "Vortex methods. I: Convergence in three dimensions,"
  *Mathematics of Computation* 39 (1982),
  [doi:10.1090/S0025-5718-1982-0658212-7](https://doi.org/10.1090/S0025-5718-1982-0658212-7).
- Degond and Mas-Gallic, "The weighted particle method for convection-diffusion
  equations. Part 1: The case of an isotropic viscosity," *Mathematics of Computation*
  53 (1989), [doi:10.1090/S0025-5718-1989-0969496-2](https://doi.org/10.1090/S0025-5718-1989-0969496-2).
- Winckelmans and Leonard, "Contributions to vortex particle methods for the
  computation of three-dimensional incompressible unsteady flows," *Journal of
  Computational Physics* 109 (1993),
  [doi:10.1006/jcph.1993.1216](https://doi.org/10.1006/jcph.1993.1216).
- Barnes and Hut, "A hierarchical O(N log N) force-calculation algorithm,"
  *Nature* 324 (1986), [doi:10.1038/324446a0](https://doi.org/10.1038/324446a0).

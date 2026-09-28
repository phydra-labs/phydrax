# Advanced particle-grid physics

This guide covers the fixed-capacity extensions built over PhydraX PIC and FLIP. Every runtime
keeps structural particle support fixed, records discrete events explicitly, and commits complete
candidate states atomically.

## Runtime particle populations

`ParticlePopulationPlan` separates prepared slot eligibility from runtime activity, mass, and
incarnation. Reused slots receive a new incarnation, so contact, collision, ionization, and
reseed history cannot alias a previous occupant. Allocation and deactivation use fixed request
arrays and fail closed on capacity or incarnation overflow.

`PreparedParticleGridSplat.build(..., active_mask=...)` intersects this runtime activity with the
structural particle mask. Dynamic PIC and FLIP methods pass runtime mass/charge as explicit
payloads; static `ParticleDiscretization` measures remain the immutable preparation reference.

## Charge state, collisions, and ionization

`PICChargeModelPlan` stores one signed base specific charge and bounded integer charge numbers.
Runtime macrocharge is derived from population mass and charge number. Charge transitions are
accepted only when their compensating product charge closes the total charge ledger.

`CoulombCollisionPlan` applies deterministic random pairing and an isotropic binary rotation that
preserves pair momentum and kinetic energy. `BackgroundMCCPlan` preserves relative speed against a
prescribed background and reports background momentum/energy sources. Both reject collision
probabilities above their prepared bound.

`ElectronImpactIonizationPlan` changes an ion charge and activates a collocated product electron in
one fixed-capacity transaction, accounting for threshold energy and momentum. `FieldIonizationPlan`
uses a bounded field-dependent rate and reports the ionization-energy field source. Population
capacity failure rejects the complete event batch.

## Reduced-dimensional and open PIC

`CompatibleMaxwell2DPlan` implements explicit TE/TM 2D3V Yee blocks; `CompatibleMaxwell1DPlan`
implements the longitudinal field plus two transverse wave pairs. `ReducedPICTransferPlan` performs
periodic CIC transfer and projects midpoint current onto the exact discrete continuity constraint.
`ReducedElectromagneticPICPlan` composes these fields with relativistic Boris stepping.

`PICOpenBoundaryPlan` clips trajectories against axis-aligned faces, supports absorbing or
reflecting particle policies, and records boundary mass, charge, kinetic energy, hit location, and
surface accumulation. Electromagnetic PIC now accepts passive instantaneous Maxwell CPML state;
CPML dissipation remains owned and reported by the Maxwell runtime.

`PICMovingWindowPlan` shifts full cochain orientations, compatible auxiliary/observer arrays, local
particle positions, global window origin, and trailing outflow in one integer-cell accepted-step
transaction.

## Unstructured and semi-implicit PIC

`PreparedSimplicialCellLocator` computes deterministic affine barycentric ownership for triangle or
tetrahedron `CellMesh` blocks. `UnstructuredElectrostaticPICPlan` deposits P1 nodal charge content,
solves the native stiffness system, gathers cellwise electric field, and advances particles.
`ElectrostaticConductorCoupling` solves fixed-size equipotential/charge constraints through one
native KKT system.

`UnstructuredWhitneyCurrentPlan` deposits Whitney-0 endpoint charge and integrated Whitney-1 path
current over bounded in-cell trajectory segments. `UnstructuredElectromagneticPICPlan` couples it
to existing compatible tetrahedral Maxwell evolution and rejects any path whose subdivision does
not resolve cell ownership.

`PICParticleResponsePlan` supplies a matrix-free gather/rotation/scatter response.
`SemiImplicitPICPlan` solves the periodic nonrelativistic theta response through bounded GMRES and
reports linear, energy, Gauss, and magnetic defects. Theta one-half is the energy-qualified value.
`PICGaussCorrectionPlan` is an explicit bounded charge-fit operation; its result does not claim
trajectory-current continuity.

## Advanced FLIP interface physics

`FLIPReseedingPlan` performs deterministic fixed-pool split/merge operations while preserving mass
and momentum and reporting the kinetic-energy defect. `ParticleLevelSetPlan` reconstructs one
fixed-band particle sphere-union level set, cell/face fractions, ghost fractions, normals, and
curvature.

`MACGhostFluidProjectionPlan` adds sharp interface pressure jumps to the existing compatible MAC
projection. `MACGhostFluidCapillaryPlan` supplies the jump `sigma times curvature` and surface
energy without adding a second continuum-surface-force body force.

`MACDiffuseSDFGeometryPlan` samples a smooth signed-distance ramp and wall velocity for
diffuse viscosity/visualization models. It is explicitly unqualified and cannot enter
sharp pressure or conservative transfer. `MACExactSDFMeasurePlan` instead produces
bounded absolute fluid volumes and open face measures from an exact-SDF enclosure;
accepted `QualifiedSharpGeometry` may bind matched FLIP transfer, sharp projection,
and `FLIPSolidBoundaryPlan` collision under one source identity. Collision records
impulse and moving-wall work.

`MACFreeSurfaceViscousMeasurePlan` combines liquid and solid measures into a face density
(density times open liquid face fraction) and a cell viscosity (viscosity times open liquid
cell fraction). `solver.MACVariationalViscosityPlan` is the implicit variable-density,
variable-viscosity MAC stage built once from `PreparedMACMomentumOperators`. Each call
solves `(rho_f + dt A_mu) u = rho_f u_star - dt b_mu`, where `A_mu` is the staggered
deviatoric-strain action of `PreparedMACVariationalViscosityAction` with no-slip or
free-slip wall handling and `b_mu` is its prescribed-wall offset. It then enforces the stage
boundary. `A_mu` is self-adjoint only in the dual-measure pairing, so the stage assembles
the symmetric positive-definite form `M (rho_f + dt A_mu)`. It refreshes one prepared
native PCG solve with the runtime density, viscosity and step. The solve stops and is
accepted on one true residual threshold, `tolerance * (||b|| + ||H u_0||)`. Free faces
with zero density and no viscous coupling have no momentum equation. They keep their
input value and are counted as `decoupled_face_count`. The result reports:

- the dissipation rate `integral 2 mu S_d:S_d`;
- the wall power;
- the kinetic energy before and after the stage;
- the excess of the energy change over `dt` times the wall power;
- the native linear status, true residual and iteration count.

On failure, the velocity is returned unchanged.

`MultiphaseFLIPPlan` takes a prepared FLIP transfer, per-phase densities and
viscosities, an optional pairwise drag matrix, and an optional maximum phase count
(all positional). P2G runs independently per phase, and results
expose per-phase deposited mass plus per-phase face mass, momentum, and velocity. The
optional drag matrix must be finite, nonnegative, symmetric, and zero on its diagonal;
pairwise drag is applied implicitly per face, so each pair exchanges equal and opposite
impulses, relative phase velocity decays without overshoot, global face momentum is
conserved, and the reported pair work is nonpositive. Invalid phase IDs on active
particles and invalid pair matrices fail closed.

## Nonperiodic PIC, curved location, and ALE epochs

Reduced PIC no longer modulo-wraps nonperiodic axes. Its current result separates
volume continuity, boundary flux, and global charge defect, and rejects a trajectory
whose bounded path capacity is exhausted. Reduced Maxwell uses the existing
PEC/PMC/impedance boundary plans on nonperiodic prepared tensor axes. A supplied
`MaxwellCPMLPlan` is prepared into fixed boundary-packed directional terms; its
memory is part of the reduced Maxwell state, advances only on an accepted step, and
therefore follows ordinary checkpoint rollback. `reset_pml` clears only that memory
while preserving electric, magnetic, and charge fields.

`PreparedSimplicialCellLocator` consumes a canonical
`PreparedFiniteElementCellMap` plus runtime geometry coordinates. Bounded
multi-seed damped Newton reports reference coordinates, geometry residual,
iterations, Jacobian condition, candidate exhaustion, and inverse-map exhaustion.
Cell ownership and ties are discrete stopped decisions. Quadratic triangle and
tetrahedron coordinate maps use the same path as affine maps.

`ALEFLIPPlan` consumes the canonical mesh splat and FE cell-map epochs. Fixed
topology steps report physical and relative particle velocities, conservative
mass/momentum deposition, and a geometric-conservation defect. Remeshing occurs
only at an accepted boundary through `prepare_particle_grid_splat_transition`;
missing conservative target transfer, particle coverage, or transferred
pressure/history for changed topology retains the old epoch. A prepared ALE object
records the accepted epoch number, so execution with a stale prepared epoch also
rolls back atomically.

## Differentiability and limits

Smooth derivatives are local to fixed ownership, phase IDs, trajectory segments,
active support, and topology. Allocation, boundary hits, remesh selection,
connectivity changes, and solver acceptance are stopped events. Phase, particle,
candidate, Newton, splat, and remap capacities remain static inside a trace.

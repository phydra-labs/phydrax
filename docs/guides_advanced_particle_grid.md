# Advanced particle-grid physics

This guide covers the fixed-capacity extensions built over PhydraX PIC and FLIP. Every runtime
keeps structural particle support fixed, records discrete events explicitly, and commits complete
candidate states atomically.

## Runtime particle populations

`ParticlePopulationPlan` separates prepared slot eligibility from runtime activity, mass, and
incarnation. Reused slots receive a new incarnation, so contact, collision, ionization, and
reseed history cannot alias a previous occupant. Allocation and deactivation use fixed request
arrays and fail closed on capacity or incarnation overflow.

Every particle also carries a persistent 64-bit global identity stored as two `uint32` words,
`ParticlePopulationState.id_hi` and `id_lo`, and the identity of the particle it was created from,
`parent_hi` and `parent_lo` (`has_parent` is false for primary particles). Identity is independent
of the storage slot: it survives deactivation, slot reuse, and slot permutation. The population
counter `next_id_hi`/`next_id_lo` numbers particles in creation order: `initialize` numbers
active slots in slot order; `allocate` numbers accepted requests in ascending event-ID order
(`ParticleAllocationRequest(..., parents=(hi, lo))` records lineage); `update_particle_population`
numbers newly active slots in slot order; FLIP split children descend from their cell's receiver;
and ionization electrons descend from the ionized ion. Other creating events call
`assign_particle_identities` with the event's request ranks. The all-ones identity is reserved
for never-occupied slots and absent parents, and allocation that would reach it fails with
`ParticlePopulationStatus.IDENTITY_EXHAUSTED`. Per-particle random streams fold both words,
`derive_key(root, address, step, id_hi, id_lo, event)`, so draws follow the particle, not its
slot.

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
`ReducedMaxwellPICFieldSolver` exposes these fields to `ElectromagneticPICPlan` through the PIC
field-solver protocol. Reduced 2-D fields with a nonperiodic axis are refused at preparation
because their Gauss charge does not follow the deposited current.

`PICOpenBoundaryPlan` clips trajectories against axis-aligned faces, supports absorbing or
reflecting particle policies, and records boundary mass, charge, kinetic energy, hit location, and
surface accumulation. It is the `boundaries` input of `ElectromagneticPICPlan`: reflected paths
deposit two segments joined at the wall and absorbed charge is kept on the grid as wall charge.
Reduced 1-D electromagnetic PIC accepts passive instantaneous Maxwell CPML state; CPML dissipation
remains owned and reported by the Maxwell runtime. The full 3-D cochain solver requires every axis
periodic and therefore cannot carry CPML.

`PICMovingWindowPlan` shifts the field through the solver's `PICWindowShift` capability (full
cochain orientations and compatible auxiliary/observer arrays, or reduced E/B/charge), local
particle positions, global window origin, trailing outflow, and optional `PICWindowInjection`
particles in one integer-cell accepted-step transaction, and reports the particle↔field charge
defect after the shift.

## Unstructured and semi-implicit PIC

`PreparedSimplicialCellLocator` computes deterministic affine barycentric ownership for triangle or
tetrahedron `CellMesh` blocks. `UnstructuredElectrostaticPICPlan` deposits P1 nodal charge content,
solves the native stiffness system, gathers cellwise electric field, and advances particles.
`ElectrostaticConductorCoupling` solves fixed-size equipotential/charge constraints through one
native KKT system.

`UnstructuredWhitneyCurrentPlan` deposits Whitney-0 endpoint charge and integrated Whitney-1 path
current over bounded in-cell trajectory segments. `UnstructuredMaxwellPICFieldSolver` couples it
to compatible tetrahedral Maxwell evolution through the inverse Hodge stars and rejects any path
whose subdivision does not resolve cell ownership.

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

`MACFreeSurfaceViscousMeasurePlan` combines liquid and solid measures. The matrix-free
`MACVariationalViscosityPlan` differentiates the symmetric strain dissipation form to obtain a
coupled, self-adjoint positive viscous action and solves it with native block linalg.

`MultiphaseFLIPPlan` accepts a finite declared phase count. P2G remains independent
per phase; results expose per-phase face mass, momentum, and velocity. Symmetric
zero-diagonal drag produces equal/opposite impulses, reports pair work, conserves
global momentum, and fails closed on invalid phase IDs or pair matrices.

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

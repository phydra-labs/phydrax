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

`MACFreeSurfaceViscousMeasurePlan` combines liquid and solid measures. The matrix-free
`MACVariationalViscosityPlan` differentiates the symmetric strain dissipation form to obtain a
coupled, self-adjoint positive viscous action and solves it with native block linalg.

`MultiphaseFLIPPlan` is intentionally two-phase and one-velocity: it deposits per-phase volume,
mass, and momentum, reconstructs mixture density/viscosity and face inverse density, and composes
`MACVariableDensityProjectionPlan` through `MACMultiphaseProjectionPlan`.

## Differentiability and limits

Smooth derivatives are local to a fixed discrete program. Allocation/deactivation, charge changes,
random pairing, ionization events, boundary hits, window shifts, simplex ownership, current path
subdivision, reseeding, liquid masks, interface topology, solid collision, and solver acceptance are
stopped branch decisions. Each result reports its status rather than applying straight-through
estimators.

The current advanced methods retain explicit initial limits: periodic reduced Maxwell, affine
simplex location, bounded path subdivision, two-phase one-velocity FLIP, fixed interface bands,
and no in-JIT capacity growth or dynamic mesh topology.

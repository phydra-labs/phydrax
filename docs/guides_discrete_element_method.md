# Discrete element method

Phydrax provides an experimental, fixed-capacity, JAX-native compliant discrete element method. `CompliantDEMMethodPlan` and `PreparedCompliantDEMDynamics` use one compiler, contact history, force reduction, and rigid-body stepper for spheres, sphere clumps, smooth convex superquadrics, and closed convex polyhedra. Shape-specific geometry never falls back to bounding-sphere contact.

## Supported contract

- isotropic disks in 2-D and spheres in 3-D, rigid sphere clumps, smooth convex superquadrics, and convex polyhedra;
- dynamic mass, uniform shape scale, and COM inertia updates from accepted morphology;
- linear spring–dashpot, Hertz, or Thornton plastic normal response;
- optional DMT, prescribed linear bridges, fitted Bagheri bridges, and near-contact lubrication;
- optional Cundall–Strack or Mindlin tangential history;
- optional constant or elastic rolling–torsional resistance;
- optional elastic half-space multicontact correction;
- fixed periodic boxes or dense-authority deforming periodic cells;
- static, prescribed, or force/torque-servo barriers;
- triangle-wall traction, work, heat, and Finnie-wear observables;
- dense, cell-list, sparse multilevel-radius, or cached Verlet neighborhoods;
- explicit kick–drift–contact–kick integration;
- branchwise differentiation through one realized contact route;
- balance-audited particle-to-continuum density, momentum, and contact-stress fields.

All shapes use `RigidBodyKinematics` and `PreparedRigidBodySet`. Spheres default to identity orientation, which cannot affect their isotropic geometry. Clumps, superquadrics, and convex bodies require explicit `orientation=` at initialization; angular velocity remains the fourth positional argument. Three-dimensional orientation is a normalized scalar-first quaternion. `CompliantDEMBodyProperties` stores population, inverse masses, shape scale, body-frame COM inertia, and inverse COM inertia (scalar per 2-D body, tensor per 3-D body). Its static `shape_id` binds properties to the prepared shape and rejects mismatched morphology updates or mixed-shape batches.

Sphere-capacity growth rebases the scale coordinate while preserving physical
radius. For frozen source and target prepared radii, the tangent chart relation
is `R_old ds_old = R_new ds_new`; `pullback_particle_epoch_transition` therefore
multiplies target scale covectors by `R_old / R_new`. Rotation, tensor-inertia,
and load covectors retain their physical frames and original owner lanes, and
the returned record restores the source prepared-shape identity.

## Problem construction

```text
particles = phx.discretization.ParticleSetPlan(
    particle_ids,
    masses,
    ambient_dimension=2,
).prepare()

spheres = phx.discretization.RigidSphereSetPlan(
    radii,
    material_ids,
)

materials = phx.equations.DEMMaterialTable(
    young_modulus,
    poisson_ratio,
    restitution_pair_table,
    friction_pair_table,
)

contact = phx.discretization.DEMContactModelPlan(
    phx.discretization.LinearSpringDashpotNormalPlan(normal_stiffness),
    tangential=phx.discretization.CundallStrackTangentialPlan(
        tangential_stiffness
    ),
)

method = phx.discretization.CompliantDEMMethodPlan(contact)
problem_ir = phx.equations.DiscreteElementProblemIR(
    "granular-problem",
    materials,
    gravity=(0.0, -9.81),
    barriers=(barrier,),
)
compiled = phx.equations.compile_discrete_element_problem(
    problem_ir,
    particles,
    spheres,
    method,
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(maximum_pairs),
)
```

Initialize cached forces before constructing the fixed-step problem:

```text
state = compiled.initialize_state(0.0, position, velocity)
problem = phx.solver.FixedStepProblem(
    phx.solver.DEMFixedStepMethod(compiled.dynamics),
    state,
    t0=0.0,
    t1=1.0,
    step_size=1.0e-4,
    state_geometry=compiled.dynamics.state_geometry,
    discretization_bundle=compiled.discretization_bundle,
)
solution = phx.solver.solve_fixed_step(problem)
```

`compiled.dynamics.state_geometry` composes the native rigid-body pose chart with additive continuous contact/history state and frozen discrete routes. It does not add quaternion components as Euclidean state: 3-D points store four quaternion components, while local, tangent, and cotangent orientation roles carry three angular components per body. `project_tangent` maps ambient quaternion vectors to those roles. Retraction/inverse retraction and their JVP/VJP actions preserve this distinction; 2-D orientation uses the principal-angle chart. Stable contact keys and discrete neighborhood decisions stay fixed within a chart.

## Pair identity and contact memory

Neighborhood slots are not physical contact identity. A cell-list rebuild may move the same physical pair to another slot. `ParticlePairKeySpace` maps canonical stable particle IDs to collision-free triangular ordinals. `match_particle_pair_keys` then remaps the complete `DEMContactHistory` tree by sorting old keys and searching new keys.

Normal, cohesion, tangential, and rotational channels own separate typed histories. Continued contacts preserve their constitutive state after objective frame transport. New contacts start from zero history; separated contacts are removed. Pair sorting and matching are discrete stopped-gradient operations, while gathered history values remain differentiable.

## Contact geometry

For two radii `rᵢ` and `rⱼ` at center distance `d`:

```text
gap = d − rᵢ − rⱼ
overlap = max(−gap, 0)
```

The normal points from the right endpoint to the left endpoint. Both lever arms terminate at one common soft-contact point, preserving the action–reaction angular-momentum identity. An overlapping pair with coincident centers is undefined and rejects the step rather than receiving an arbitrary normal.

`RigidContactGeometry` records owner indices, stable feature keys, a common contact point, owner and contact arms, raw `normal_gradient`, unit normal, point weight, curvature validity, and certificate margins. S8 point contacts use `normal_gradient=normal`, `surface_area_weight=1`, and `measure_kind="point_force"`; stiffness keeps force-per-displacement units. `DEMEvaluation.geometry` is a fixed-capacity `RigidContactGeometryBatch` with candidate/active counts, overflow, and success evidence. Clump component forces and common-point moments reduce to owner bodies; same-owner component pairs are excluded.

`maximum_contact_routes` bounds shape-generated routes independently of broad-phase pair capacity. Overflow refuses the complete candidate, never truncates contacts. The bound covers every valid broad-phase candidate, including positive-gap routes that may carry cohesion; it is not sized only from overlapping contacts. Hertz additionally requires valid physical curvature; sharp convex features do not acquire an inferred effective radius.

Clump mass, inertia, and volume use additive component measures, not geometric union measures: overlapping pebble material is counted additively. Convex templates must be closed, consistently outward-oriented triangle shells; preparation refuses open, inconsistent, or inward shells rather than silently flipping faces. Their exact shell volume/COM/inertia are prepared once, and the shape set centers local vertices at COM.

## Linear spring–dashpot contact

The normal force magnitude is

```text
Fₙ = max(kₙ δ − γₙ vₙ, 0)
```

where damping is derived from the explicit pair restitution coefficient and effective mass. Clipping prevents an attractive viscous normal force during rapid separation. The elastic energy is `½kₙδ²`.

Cundall–Strack tangential contact transports the previous tangential displacement into the new contact frame, integrates tangential relative velocity, forms a spring–dashpot trial force, and enforces

```text
‖Fₜ‖ ≤ μFₙ
```

When sliding occurs, the stored displacement is back-solved from the capped force. Clipping force without correcting history is not the implemented model.

## Hertz–Mindlin contact

`HertzNormalContactPlan` uses effective radius and effective Young modulus with restitution-derived Tsuji-style damping. `MindlinTangentialContactPlan` uses effective shear modulus and overlap-dependent tangent stiffness. Mindlin tangential contact may only be combined with Hertz normal contact.

Hertz contact against an implicit barrier requires
`GeometryCapability.CONTACT_CURVATURE`. Smooth analytic circles and spheres
provide certified curvature; unsupported or nonsmooth features reject.

## Exact signed-distance barriers

`ImplicitDEMBarrier` requires exact signed distance, reliable sign, and boundary normals. `DEMBarrierSide.INTERIOR` confines particles to the negative-inside geometry; `EXTERIOR` keeps particles outside it. Barrier contact occupies fixed barrier–particle history slots and reports equal-and-opposite reaction force and torque.

Barriers may use static, prescribed rigid, or servo-controlled kinematics.
Wall contact-point velocity enters relative contact kinematics and prescribed
wall work is separated from contact balance loss.

## Timestep and overlap evidence

`compiled.step_restriction()` reports contact-period and applicable Rayleigh estimates. It does not silently choose or adapt the timestep. `CompliantDEMMethodPlan.maximum_overlap_fraction` is the hard runtime admissibility guard; exceeding it rejects the complete structured step atomically.

## Differentiation

`DEMSensitivityPolicy(mode=...)` takes a `phx.BranchDifferentiationPolicy` and
accepts three members. `BRANCHWISE` differentiates only the executed discrete
route and returns a local-validity certificate covering gap, no-tension,
friction, frame, overlap, and neighborhood-cache margins. `SMOOTH_SURROGATE`
uses separately fingerprinted soft normal activation and smooth Coulomb
projection, with an explicit forward-bias certificate. `EVENT_AWARE` localizes
one transverse guard and returns its saltation matrix; grazing or simultaneous
events fail qualification. Any other member raises `ValueError`. Sensitivity
results carry the policy's canonical `derivative_contract`. Pair construction,
stable-ID remapping, material IDs, and topology events remain stopped-gradient.

## Failure semantics

The following reject the candidate step without partially accepting state:

- neighborhood occupancy or pair-capacity overflow;
- out-of-domain cell-list position;
- invalid material parameters or IDs;
- overlapping coincident centers;
- invalid shape queries, unsupported curvature, or contact-route overflow;
- nonfinite loads or state;
- ill-defined tangential-frame transport;
- capillary-fit domain or conserved-liquid balance violation;
- deforming-cell conditioning, unique-image, or strain-increment failure;
- excessive relative overlap.

Detailed failure inputs can be replayed through `PreparedCompliantDEMDynamics.step_detailed`.

## Accepted-step energy ledger

`DEMResolvedLoad` retains particle-contact, per-barrier, gravity, external, and
total endpoint loads. `DEMStepEnergyLedger` evaluates source impulse work with
the average endpoint velocity and records contact stored-energy change,
prescribed-wall work, deforming-cell work, signed contact balance loss, and
closure residual. Rejected candidates expose their candidate ledger but
preserve the accepted `DEMEnergyLedgerState` exactly.

External-load callbacks return one COM force and world-frame torque per body.
`PreparedRigidBodySet.load` converts payloads to the prepared dtype and requires
the exact body-count and physical-vector dimensions; broadcasting a single
body's force over the whole population is rejected.

Contact balance loss is a discrete accounting quantity, not automatically pure
thermodynamic dissipation. Normal viscous, tangential constitutive, rolling, and
plastic estimates remain separately labeled.

## Cached and specialized neighborhoods

`VerletParticleNeighborhoodPlan` reuses a fixed candidate relation while every
active particle remains within `skin / 2` of its reference position. Reuse
preserves contact slots directly; rebuild remaps by stable pair key exactly
once. Certified failure, capacity overflow, or a bad key rejects the complete
step.

Reference, dense-fused, cell-fused, and Verlet-fused pair reductions share the
same contact and accumulation semantics. `HierarchicalRadiusParticleNeighborhoodPlan`
implements the multilevel contact-search construction of [Ogarko and Luding](https://doi.org/10.1016/j.cpc.2011.12.019) using immutable interaction-radius envelopes, sparse occupied-cell keys, and
bottom-up cross-level search. Its constructor takes per-particle envelope
radii, level edges, cell occupancy, pair capacity, and `ParticleBox`; the DEM
compiler rejects any envelope smaller than the body-plus-contact-law reach.

## Deforming periodic rheology

`DEMPeriodicCellControlPlan` combines disjoint prescribed strain-rate and
stress-controlled tensor masks. Stress feedback uses the current kinetic and
pair virial stress. `PeriodicCell(maximum_condition_number=...)` preallocates
the minimum-image stencil for the complete admissible deformation envelope.
The candidate cell, affinely mapped particle positions, velocities, and cell
work commit atomically. Conditioning, unique-image, determinant, and maximum
strain-increment failures roll back the complete DEM step.

The current deforming-cell authority is fully periodic, gravity-free, has no
fixed particles or implicit barriers, and requires
`DenseParticleNeighborhoodPlan`. Static `PeriodicCell` neighborhoods retain
their existing behavior.

## Additional contact physics

- `ConstantRollingResistancePlan` adds a stateless Coulomb-like rolling couple.
- `ElasticRollingTorsionalResistancePlan` stores objective rolling and torsional elastic displacement, applies damping and friction caps, and reports energy and yield margins.
- `DMTContactCohesionPlan` adds finite-range DMT attraction beside an explicit normal law.
- `LinearCapillaryBridgePlan` owns bridge birth, rupture, volume source/release, and liquid-balance residuals.
- `NearContactLubricationPlan` adds regularized finite-gap viscous resistance.
- `CompositeDEMCohesionPlan` composes ordered cohesion contributions without erasing component history or diagnostics.
- `BagheriCapillaryBridgePlan` implements the fitted finite-volume model of [Bagheri et al.](https://doi.org/10.1016/j.softx.2024.102048): force, analytic branch potential, tangent stiffness, exposed liquid area, radius-derived rupture distance, and fit margins. Its paper-backed domain is `1e-6 <= V/R^3 <= 1e-1` and `0 <= contact_angle <= 50 degrees`; separations above `0.9` of rupture are explicitly reported as fit extrapolation. Unequal spheres use the published characteristic-radius extension and remain experimental; implicit-barrier contacts reject at compilation.
- `ConservedLiquidBridgeProcessPlan` draws simultaneous bridge births proportionally from endpoint films, returns rupture volume to both films, and removes `evaporation_flux * exposed_area * dt`. Film plus bridge plus cumulative evaporated volume is checked after every transaction. It requires exactly one `BagheriCapillaryBridgePlan(..., conserve_liquid=True)` and currently supports particle-particle bridges without barriers.
- `ThorntonLinearPlasticNormalPlan` stores maximum overlap, plastic overlap, and irreversible loss under bilinear loading/unloading.
- `ParticleContactExchangePlan` provides conservative contact heat exchange as a separate coupling channel.

## Rigid shapes, walls, and bonds

`RigidBodySetPlan` supports SO(2) angles or scalar-first SO(3) quaternions,
body-frame inertia, and world-frame angular velocity. Rigid sphere clumps store
only owner pose dynamically and use immutable component templates and stable
owner/component contact keys.

`RigidContactGeometry` is the common contact-point/normal/lever-arm contract. Static triangle walls use deterministic face/edge/vertex ownership. Convex polyhedra use a complete face/edge separating-axis oracle. Certified implicit contact currently supports sphere-to-implicit queries with an explicit distance-error and Lipschitz certificate. `SuperquadricContactPlan` uses fixed-iteration support-map geometry with witness, curvature, and convergence residuals; it rejects rather than substituting a bounding-sphere force.

`FixedBondGraphPlan` is independent of transient contact search. Elastic bonds
retain stable IDs and axial/shear/bending/twisting energy. Mixed-mode damage is
monotone and accumulates fracture energy. Fixed-pool topology transitions
activate preallocated fragment owners atomically and reject on conservation or
capacity failure.

## Particle-to-continuum observables

`ParticleCoarseGrainingPlan` implements the balance-law construction of [Weinhart et al.](https://doi.org/10.1063/1.4812153). It composes an existing `ParticleGridSplatPlan` with
fixed Gauss--Legendre interaction-segment quadrature. It deposits mass, volume,
momentum, raw momentum flux, external force, constituent-resolved primary
fields, and the line-integrated pair virial. `ParticleContinuumFields` keeps
primary content and density separate from derived mean velocity, kinetic
stress, contact stress, and bulk stress. Every particle and segment deposition
retains the splat balance evidence; route selection remains branchwise while
weights and payloads remain differentiable.

## Coupling

Canonical particle/continuum exchange uses `PreparedMeshParticleGridSplat`
with one normalized compact-kernel route for gather and extensive deposition.
Unresolved CFD–DEM coupling returns paired particle and fluid momentum sources
and enforces closure validity. Accepted multirate windows commit fluid and DEM
candidates atomically.

Resolved immersed-boundary coupling uses the same marker interpolation/spreading weights and checks discrete work adjointness. `ReactiveCFDDEMCouplingPlan` composes conservative continuum heat/species exchange, contact heat, radiation, particle conversion, morphology, and DEM under one atomic macro-window acceptance decision.

## Wet barriers and periodic rheology

`DEMBarrierCapillaryPlan` binds one barrier ID to a planar or certified
isotropic-curvature sphere/surface model. Conserved births withdraw their
declared endpoint split from particle films and the barrier reservoir using
simultaneous per-particle/per-barrier scaling. Rupture returns the same split;
evaporation, pair bridges, wall bridges, films, and reservoirs share one
inventory residual and commit with mechanical history, reaction, and wall work.
Birth, rupture, feature selection, and allocation are replay events rather than
ordinary differentiable branches.

`PeriodicNeighborhoodEnvelope` declares singular-value, lattice-height, and
deformation bounds for dynamic-cell routes. `DEMBulkStressPlan` explicitly
selects contact, kinetic, barrier-virial, and body-force moment terms and reports
the origin, frame, and symmetry defect. No undeclared gravity frame, lab-fixed
wall, or out-of-envelope deformation is silently accepted.

## Current maturity and limitations

The original spherical forward path is evidence-backed but each newly enabled law, shape, coupling, sensitivity, and backend retains its own support-matrix status. Distributed execution remains explicitly unsupported. Convex polyhedron contact uses linear penalty semantics until physical effective curvature is supplied and qualified; implicit contact does not infer Hertz curvature from sampled fields. Superquadric triangle-wall witnesses feed the shared stepper. Radius-dependent CFD, reactive morphology, insertion, and periodic specializations remain explicitly sphere-only rather than treating a nonspherical bound as a physical radius.

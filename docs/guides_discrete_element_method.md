# Discrete element method

Phydrax provides an experimental, fixed-capacity, JAX-native soft-sphere discrete element method for two- and three-dimensional rigid spheres. The method composes the existing particle support, dense or cell-list neighborhoods, stable physical pair identities, deterministic reductions, native signed-distance geometry, and fail-closed fixed-step solver.

## Supported contract

- isotropic disks in 2-D and spheres in 3-D;
- scalar mass moment of inertia and angular velocity;
- fully mobile or fully fixed bodies;
- linear spring–dashpot or Hertz normal response;
- optional Cundall–Strack or Mindlin tangential history;
- periodic particle boxes;
- static exact-signed-distance barriers for linear contact;
- dense-reference and fixed-capacity cell-list candidate relations;
- explicit kick–drift–contact–kick integration;
- branchwise differentiation through one realized contact route.

Sphere orientation is intentionally absent: it cannot affect isotropic sphere contact. Clumps and nonspherical bodies will introduce orientation with their own state geometry.

## Problem construction

```python
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

method = phx.discretization.SoftSphereDEMMethodPlan(contact)
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

```python
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

## Pair identity and contact memory

Neighborhood slots are not physical contact identity. A cell-list rebuild may move the same physical pair to another slot. `ParticlePairKeySpace` maps canonical stable particle IDs to collision-free triangular ordinals. `match_particle_pair_keys` then remaps edge-local state by sorting old keys and searching new keys.

A continued contact preserves its tangential displacement and previous normal. A new contact starts from zero history. A separated contact is removed. Pair sorting and matching are discrete stopped-gradient operations; gathered history values remain differentiable.

## Contact geometry

For two radii `rᵢ` and `rⱼ` at center distance `d`:

```text
gap = d − rᵢ − rⱼ
overlap = max(−gap, 0)
```

The normal points from the right endpoint to the left endpoint. Both lever arms terminate at one common soft-contact point, preserving the action–reaction angular-momentum identity. An overlapping pair with coincident centers is undefined and rejects the step rather than receiving an arbitrary normal.

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

`compiled.step_restriction()` reports linear contact-period and Rayleigh estimates. It does not silently choose or adapt the timestep. `SoftSphereDEMMethodPlan.maximum_overlap_fraction` is the hard runtime admissibility guard; exceeding it rejects the complete structured step atomically.

## Differentiation

Three explicit modes are available. `sharp_branchwise` differentiates only the
executed discrete route and returns a local-validity certificate covering gap,
no-tension, friction, frame, overlap, and neighborhood-cache margins.
`smooth_surrogate` uses separately fingerprinted soft normal activation and
smooth Coulomb projection, with an explicit forward-bias certificate.
`hybrid_event_aware` localizes one transverse guard and returns its saltation
matrix; grazing or simultaneous events fail qualification. Pair construction,
stable-ID remapping, material IDs, and topology events remain stopped-gradient.

## Failure semantics

The following reject the candidate step without partially accepting state:

- neighborhood occupancy or pair-capacity overflow;
- out-of-domain cell-list position;
- invalid material parameters or IDs;
- overlapping coincident centers;
- nonfinite loads or state;
- ill-defined tangential-frame transport;
- excessive relative overlap.

Detailed failure inputs can be replayed through `PreparedSoftSphereDEMDynamics.step_detailed`.

## Accepted-step energy ledger

`DEMResolvedLoad` retains particle-contact, per-barrier, gravity, external, and
total endpoint loads. `DEMStepEnergyLedger` evaluates source impulse work with
the average endpoint velocity and records contact stored-energy change,
prescribed-wall work, signed contact balance loss, and closure residual.
Rejected candidates expose their candidate ledger but preserve the accepted
`DEMEnergyLedgerState` exactly.

Contact balance loss is a discrete accounting quantity, not automatically pure
thermodynamic dissipation. Normal viscous, tangential constitutive, rolling, and
plastic estimates remain separately labelled.

## Cached and specialized neighborhoods

`VerletParticleNeighborhoodPlan` reuses a fixed candidate relation while every
active particle remains within `skin / 2` of its reference position. Reuse
preserves contact slots directly; rebuild remaps by stable pair key exactly
once. Certified failure, capacity overflow, or a bad key rejects the complete
step.

Reference, dense-fused, cell-fused, and Verlet-fused pair reductions share the
same contact and accumulation semantics. `HierarchicalRadiusParticleNeighborhoodPlan`
adds immutable radius classes and pair-specific diameter-plus-skin filtering.

## Additional contact physics

- `ConstantRollingResistancePlan` adds a stateless Coulomb-like rolling couple.
- `DMTAdhesiveNormalPlan` supplies a finite-range attractive potential and
  declares its neighbor-search reach.
- `ThorntonLinearPlasticNormalPlan` stores maximum overlap, plastic overlap, and
  irreversible loss under bilinear loading/unloading.
- `LumpedContactThermalPlan` conservatively exchanges pair heat and reports an
  explicit thermal timestep restriction.

## Rigid shapes, walls, and bonds

`RigidBodySetPlan` supports SO(2) angles or scalar-first SO(3) quaternions,
body-frame inertia, and world-frame angular velocity. Rigid sphere clumps store
only owner pose dynamically and use immutable component templates and stable
owner/component contact keys.

`RigidContactGeometry` is the common contact-point/normal/lever-arm contract.
Static triangle walls use deterministic face/edge/vertex ownership. Convex
polyhedra use a complete face/edge separating-axis oracle. Certified implicit
contact currently supports sphere-to-implicit queries with an explicit
distance-error and Lipschitz certificate.

`FixedBondGraphPlan` is independent of transient contact search. Elastic bonds
retain stable IDs and axial/shear/bending/twisting energy. Mixed-mode damage is
monotone and accumulates fracture energy. Fixed-pool topology transitions
activate preallocated fragment owners atomically and reject on conservation or
capacity failure.

## Coupling

`ConservativeParticleGridTransferPlan` uses one normalized weight relation for
gather and extensive-content deposition. Unresolved CFD–DEM coupling returns
paired particle and fluid momentum sources and enforces closure validity.
Accepted multirate windows commit fluid and DEM candidates atomically.

Resolved immersed-boundary coupling uses the same marker interpolation/spreading
weights and checks discrete work adjointness. Thermal CFD–DEM exchanges equal
and opposite particle/fluid heat sources.

## Current maturity and limitations

The original spherical forward path is evidence-backed but each newly enabled
law, shape, coupling, sensitivity, and backend retains its own support-matrix
status. Distributed execution remains explicitly unsupported. Convex contact
uses linear penalty semantics until a separately qualified effective-curvature
law exists; implicit contact does not infer Hertz curvature from sampled fields.

# Extended constrained and deformable mechanics

Phydrax separates geometric admissibility, physical laws, numerical algorithms, and accepted runtime state. The APIs in this guide extend hard rigid joints without introducing a universal `Constraint` object.

## Joint geometry and coordinates

`RigidJointGraphPlan` supports fixed and ball joints in two and three dimensions, spatial hinges, and dimension-native prismatic and distance joints. A prepared graph owns a canonical `RigidJointRowLayout`; rows are grouped by stable joint ID and have a deterministic kind/local-row identity.

- Planar rigid state is `(x in R2, angle, v in R2, angular velocity)` with principal-angle retraction.
- Spatial rigid state uses scalar-first Hamilton quaternions and world-frame angular velocity.
- A planar ball joint is the native revolute anchor-coincidence relation. `HingeJointSetPlan` remains spatial-only.
- Prismatic joints leave one axial displacement free. Distance joints use a scaled squared-length residual and reject zero rest length.

`prepare_rigid_joint_coordinates` evaluates fixed errors, ball relative rotation, hinge angle/rate, prismatic position/rate, distance/rate, and chart evidence. `RigidJointCoordinateState` supplies multi-turn hinge unwrapping without modifying immutable graph geometry.

## Compliance, damping, and actuation

Joint laws are physical models prepared against stable graph joint IDs:

- `CompliantRigidJointLawPlan` supplies conservative energy and equal/opposite body loads.
- `DissipativeRigidJointLawPlan` supplies a PSD dissipation law and nonnegative loss rate.
- `RigidJointEffortMotorPlan` applies generalized effort to a free coordinate.
- `RigidJointPDServoPlan` provides bounded position/rate feedback with source-power and saturation evidence.

A law evaluation reports coordinates, rates, `RigidBodyLoad`, stored energy, dissipation rate, actuator source power, chart margins, compatibility, and finite/valid evidence. Hard-row compatibility is checked explicitly; a spring or motor is not silently applied to a coordinate annihilated by hard projection.

## Joint limits and unilateral rows

`FixedCapacityUnilateralPlan` and `PreparedUnilateralRows` implement nonnegative fixed-capacity variational-inequality rows with candidate/accepted warm starts and complementarity certificates.

`JointLimitPlan` adds lower and upper hinge stops. It distinguishes free, active, and releasing states; reports gap and velocity complementarity; and atomically rolls back coordinate and impulse state on failure. Limits remain separate from bilateral multipliers.

## Hard contact, restitution, and friction

`HardContactRoutePlan` binds fixed-capacity rigid contact routes to body slots or the world. `PreparedHardContact` consumes `RigidContactGeometry`, rigid kinematics, timestep, and `HardContactState` to produce a candidate impulse response and complete feasibility evidence.

The contact response includes:

- nonnegative normal impulses and nonpenetration/stabilization evidence;
- Newton velocity restitution applied only to newly closing impacts;
- resting-contact classification to prevent repeated rebound;
- exact planar friction-wedge or spatial second-order-cone Coulomb projection;
- stick/slip, cone, dissipation, energy, feature, and route margins;
- candidate/accepted normal and tangent warm starts.

Penalty DEM remains a different physical model with elastic/plastic/tangential history. Hard-contact impulses never reuse `DEMContactHistory`.

## Breakage and dynamic topology

`BreakableRigidJointLawPlan` owns monotone damage, fracture dissipation, hysteresis, and deterministic break proposals. `RigidTopologyPlan` owns fixed body/joint capacities and a typed event journal. `PreparedRigidTopology` executes deterministic ID-ordered activation/deactivation transactions.

`InactiveRigidJointDualGauge` maps runtime joint activity through the prepared row layout so inactive physical rows have an explicit zero-dual gauge rather than a singular zero row. A successful transition commits masks, multiplier resets, damage, event IDs, replay digest, and contact-cache epoch together. Capacity, rank, identity, or certificate failure rolls every leaf back.

Topology changes inside a compiled epoch are limited to predeclared slots. Capacity, endpoint, or layout changes require a new prepared epoch at an accepted host boundary. Sensitivities are branchwise only when the event journal and active masks remain unchanged.

## Transient volumetric FEM

`FiniteElementDynamicsPlan` wraps an existing compiled second-order finite-element system in an implicit Newmark solve. It provides:

- prepared nonlinear roots and implicit derivatives;
- displacement, velocity, acceleration, and committed material state;
- admissibility and inversion hooks;
- candidate/accepted material transactions;
- kinetic, stored, external-work, and balance evidence.

`prepare_finite_element_dynamics_step` refreshes numeric arguments without changing the symbolic plan, and `solve_finite_element_dynamics_step` commits or rolls back the complete state.

## Rigid--deformable attachments and mixed constraints

`PreparedFiniteElementPointInterpolation` owns fixed host-cell interpolation and its exact transpose scatter. `RigidDeformableAttachmentPlan` builds translational rigid--FE attachment residuals, KKT operators, rank evidence, and action/reaction/moment certificates.

`MixedVolumetricConstraintPlan` supplies a displacement--pressure saddle payload with explicit pressure gauge and rank evidence. It does not reuse fluid pressure projection or hide bulk penalties as hard incompressibility.

## Cosserat rods

`RodPlan` describes fixed two- or three-dimensional centerline topology, material frames, mass/inertia, and PSD stretch/shear/bend/twist stiffness. `PreparedRod` evaluates objective discrete Cosserat strains, energy-gradient force/moment loads, kinetic energy, chart evidence, and endpoint wrench transfer.

`PreparedRodDynamics` provides bounded symplectic candidate/accepted evolution with an optional fixed-work inextensibility projection. Rods are not force-density networks or distance-only particle chains.

## Shells and cloth

`TriangularShellPlan` prepares surface rest metrics, areas, interior-edge bending stencils, thickness, density, and membrane/bending parameters. `PreparedTriangularShell` evaluates objective membrane and hinge-bending energies and gradient-derived forces.

`PreparedShellDynamics` provides damped fixed-node candidate/accepted dynamics with area/orientation/degeneracy certificates. Its fixed-capacity self-contact output is a `RigidContactGeometry`-compatible payload; shell constitutive response remains distinct from contact.

## Deformable and MPM coupling

`DeformableContactPlan` provides fixed-capacity node/plane, node, segment, and triangle routes with signed gaps, witnesses, interpolation/scatter weights, stable keys, and exact transpose action.

`RigidMPMCouplingPlan` keeps weld, penalty, and unilateral impulse coupling modes explicit. It preserves MPM route, stability, material-state, and candidate/accepted ownership while reporting equal/opposite rigid and particle/grid action plus route/cache certificates.

## Failure and differentiation

Every subsystem reports finite/valid evidence and preserves the prior accepted state on failure. No failed hard solve falls back to penalty physics, and no topology overflow silently drops events.

Smooth physical laws support ordinary differentiation away from chart and saturation boundaries. Hard limits, contact, restitution, and Coulomb friction expose branch margins for fixed active/impact/stick states. Breakage and topology events invalidate derivatives when the event route changes.

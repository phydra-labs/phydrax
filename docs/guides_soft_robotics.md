# Soft robotics

Phydrax does not define one universal “soft robot” backend. It composes exact
geometry, mechanics, actuation, contact, observation, inference, and transaction
contracts. A claim applies only to the prepared profile and capability tuple that
produced its evidence. A finite result is not by itself evidence of validity;
inspect each result's `successful`, `status`, `valid`, and residual fields.

## Ownership map

| Concern | Public owner | Principal contracts |
| --- | --- | --- |
| Point and differential geometry | `phydrax.metrix` | `AbstractStateGeometry`, `LocalRetraction`, `StateChartEvidence`, `StateTransportEvidence` |
| Primal and true-dual vector spaces | `phydrax.linalg` | `ArraySpace`, `BlockSpace`, `DualSpace`, `dual_transpose` |
| Atomic plant lifecycle and replay | `phydrax.dynamics` | `AbstractDiscretePlant`, `PlantRuntimeState`, `PlantStepContext`, `PlantResetResult`, `PlantStepResult`, `PlantCheckpoint`, `PlantReplayResult` |
| Complete vector codecs | `phydrax.dynamics` | `PlantStateVectorCodec`, `ControlVectorCodec`, `EncodedPlantState`, `EncodedPlantVector`, `EncodedControl`, `PlantPowerEvidence` |
| Native and reduced rods | `phydrax.applications.solid_mechanics` | `RodPlan`, `RodStrainBasisPlan`, `ReducedRodPlan`, `RodReconstructionPlan`, `PreparedReducedRodDynamics` |
| Tendons and other rod actuators | `phydrax.applications.solid_mechanics` | `TendonRoutePlan`, `FrictionlessElasticTendonPlan`, `TendonDrivenRodPlant`, pressure, intrinsic-strain, variable-stiffness, and magnetic plans |
| Capsule contact | `phydrax.applications.contact` | `RodCapsuleGeometryPlan`, `RodContactSearchPlan`, `RodContactCCDPlan`, `RodContactManifoldState`, `CompositeContactResponse` |
| Atomic reduced-rod contact | `phydrax.applications.solid_mechanics` | `PreparedReducedRodContactPlant`, `FRICTIONLESS_ROD_CONTACT_CAPABILITY`, `ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY` |
| Tasks, observations, control, inference | `phydrax.applications.robotics` | continuum IK, trajectory, observation, calibration, co-design, and MPC contracts |
| Floating and rigid–soft composition | solid mechanics and robotics | `FloatingReducedRodPlant`, plant ports, attachment plans, `HybridRigidSoftPlant` |
| Continuum profiles | `phydrax.applications.robotics` | `FEMSoftPlant`, `MPMSoftPlant`, `MJXAdapter` |

Plans and prepared objects are immutable PyTrees. Content identities bind the
basis, topology, material worksets, layouts, provider features, and numerical
revision. Do not transplant runtime values between preparations merely because
their array shapes match.

## Point storage, B0, tangents, and true duals

`AbstractStateGeometry` separates a state point from four differential roles:

1. the local perturbation space at a chosen base point, denoted B0;
2. the physical tangent space at the retracted point;
3. the local covector space, the algebraic dual of B0;
4. the physical cotangent space, the algebraic dual of the physical tangent.

Point storage may have a fifth, different shape. For example, a quaternion pose
uses seven point coordinates but six physical tangent coordinates. The
contracts therefore name every map:

- `retract(x, delta)` maps B0 coordinates to a point;
- `inverse_retract(x, y)` maps a nearby point back to B0;
- `retraction_jvp(x, delta, velocity)` maps a local velocity to a physical
  tangent at the retracted point;
- `retraction_inverse_jvp(x, y, tangent)` maps the physical tangent back to B0;
- `retraction_vjp(x, delta, cotangent)` is the algebraic transpose into the
  local dual;
- `transport_tangent` and `transport_cotangent_pullback` move paired quantities
  between base points.

`StateChartEvidence` measures inverse and differential round trips plus VJP
pairing residuals. `StateTransportEvidence` separately measures identity,
round-trip, duality, and optional isometry residuals. Capability booleans such
as `supports_exact_inverse` and `supports_transport` are requirements to
negotiate, not promises to infer from a class name.

`DualSpace(primal)` is a true declared coordinate dual. Its `pair(covector,
vector)` applies the primal's declared pairing. `dual_transpose(operator)` maps
the target dual to the source dual. Neither operation silently identifies a
covector with a vector through a Euclidean Riesz map. Rod forces, moments,
tendon tensions, attachment wrenches, contact impulses, and reduced efforts use
these pairings for measured virtual-power evidence.

## Complete plant lifecycle, codecs, and replay

Every `AbstractDiscretePlant` owns exact state, control, and parameter schemas;
semantic and numeric identities; an execution signature; a finite-input policy;
and a reset fallback. Domain implementations provide `propose_reset` and
`propose_step`. Callers use the final public transactions:

```python
reset = plant.reset(key, parameters)
source = reset.accepted_state
context = PlantStepContext(source.time, source.time + dt, source.step_index)
step = plant.step(context, source, command, parameters)
```

`PlantRuntimeState` contains the complete domain payload, time, step index, PRNG
key, and all owning identities. `PlantResetResult` and `PlantStepResult` retain
both candidate and accepted states. Acceptance is casewise and atomic: failed
cases retain the entire source state, including clock and key, rather than a
partially updated payload. `attempted`, `successful`, plant `status`,
`backend_status`, and domain evidence remain distinct.

`checkpoint` records the exact complete-state digest; `verify_checkpoint` and
`restore` fail closed on identity or byte mismatch. `replay(checkpoint,
contexts, commands, parameters, expected_digests=...)` feeds accepted states
forward, records every result and first failure, and independently reports the
first digest mismatch. Replay proves equality for that preparation and input
sequence; it is not a cross-platform bitwise promise.

`PlantStateVectorCodec` is a complete point-storage bridge to a declared
four-space `StateLayout`. It encodes every mutable state leaf and explicitly
names immutable mode paths. Its local/physical tangent and covector methods use
the geometry JVP/VJP rather than padding or truncating coordinates.
`PlantPowerEvidence` measures the resulting pairing identity. `ControlVectorCodec`
performs the exact control-schema bridge. Every encoded value carries codec,
schema, semantics, numeric-revision, and execution-signature identities; stale
or foreign vectors are rejected.

## Native 3-D rod mechanics

A spatial `RodPlan` declares ordered segment endpoints, rest node positions,
material frames, nodal masses, segment inertias, stretch/shear stiffness, and
bend/twist stiffness. `prepare_rod` validates the chain and creates native
configuration, velocity, and true effort spaces. `RodState` stores nodal
positions and velocities plus scalar-first unit-quaternion segment orientations
and material angular velocities. Spatial velocity blocks are not assumed to
share point-storage shape.

`evaluate_rod` reports stretch/shear and bend/twist strain, internal loads,
energy, chart and finite evidence. `RodDynamicsPlan` and
`prepare_rod_dynamics` provide the native discrete rod step. The reduced stack
below is a different, explicit parameterization; it never relabels a reduced
result as a native-discretization result.

### Basis, reduction, and reconstruction

The canonical six material-strain components are axial/shear `nu_x`, `nu_y`,
`nu_z` and bend/twist `kappa_x`, `kappa_y`, `kappa_z`.
`RodStrainBasisPlan` supports:

- `piecewise_constant`: PCS, one coordinate per selected component and material
  interval;
- `shifted_legendre`: GVS, one global shifted-Legendre family per selected
  component;
- `explicit`: caller-declared piecewise polynomial coefficients.

`prepare_rod_strain_basis` reports rank, condition, dtype, quadrature, and
coverage evidence. `ReducedRodPlan` binds that basis, reference coefficients,
base policy, optional fixed-base pose, and tolerances to the native rod.
`prepare_reduced_rod` produces `PreparedReducedRod`; its authoritative lift maps
are `lift_configuration`, `lift_velocity_operator`, and
`lift_effort_pullback_operator`. The first reconstructs a native configuration;
the latter two are an exact JVP and its true-dual pullback. The convenience
functions `lift_reduced_rod_state`, `lift_reduced_rod_velocity`, and
`pullback_reduced_rod_loads` use the same maps; these three public maps are the
complete lift surface.

A `RodReconstructionPlan` binds physical `RodFrameQueryPlan` arc lengths and
`method="auto"`, `"pcs"`, or `"gvs"`:

- PCS is available only for a piecewise-constant basis and composes exact
  constant-strain panel exponentials;
- GVS integrates the varying strain field with declared refinement and
  quadrature tolerance;
- `auto` selects PCS for a piecewise-constant basis and GVS otherwise.

`evaluate_rod_reconstruction` reports material poses, body/world twists,
strains, domain coverage, chart margin, quadrature error, and discrepancy from
the native discrete rod. The native-discretization discrepancy is evidence, not
an equality assertion. `compare_reduced_rod_discretizations` records coarse,
medium, and optional fine errors and observed order without declaring an
asymptotic regime when the evidence does not support one.

### Materials, loads, dynamics, and integrators

`RodMaterialWorkset` assigns physical measures and reference strains to exact
material sites. `LinearElasticRodMaterialPlan` is memoryless;
`KelvinVoigtRodMaterialPlan` adds rate-dependent viscosity. Prepared material
trials retain source/candidate/accepted history, work, dissipation, and
constitutive residual evidence.

`ReducedRodLoadBundle` and `RodLoadLedger` keep force/moment frame, power
channel, source identity, and quadrature explicit. A `ReducedRodDirectLoad` is
already a reduced covector and still names its power source.

`prepare_reduced_rod_dynamics` composes the reduction, prepared materials,
gravity, and either `ReducedRodDenseCholeskyPlan` or
`ReducedRodMatrixFreeCGPlan`. Mass, inverse mass, bias, energy, force, forward
dynamics, inverse dynamics, and dense-reference actions all return status and
residual-bearing results. The matrix-free route is not a dense-result alias.

`ReducedRodSemiImplicitVelocityEuler` is fixed-work and explicit.
`ReducedRodImplicitMidpoint` uses a declared nonlinear method and termination
policy. `integrate_reduced_rod_step` returns candidate/accepted integration
states and a `ReducedRodEnergyWorkLedger`; failed material, mass, nonlinear,
chart, finite, or ledger checks retain the complete source state.
`PreparedReducedRodPlant` wraps either policy in the shared atomic plant
lifecycle and exposes an identity- and state-revision-bound mass response.

## Tendons and advanced rod actuation

`RodMaterialStation(segment_id, xi, offset)` places a tendon eyelet in a material
frame. `TendonRoutePlan` requires at least two ordered stations.
`PreparedTendonRoute` evaluates world eyelet points, velocities, span lengths,
and rates. Its native/reduced length-rate operators and effort-pullback
operators certify the sign convention

`tendon power on rod = - tension × tendon length rate`.

`FrictionlessElasticTendonPlan` declares stiffness, free-length, payout-rate,
physical-length, and maximum-tension bounds. `TendonActuatorState` owns free
length; `TendonPayoutCommand` owns payout rate. `evaluate_tendon_actuation`
retains unilateral tension, stored energy, rod power, and power residuals.

`TendonDrivenRodPlant` is the exact contact-free tuple

`(fixed-base reduced 3-D rod, semi-implicit velocity Euler, one-or-more
frictionless elastic tendons, payout-rate commands, optional reduced external
effort)`.

Construct it with `prepare_tendon_driven_rod_plant(base_plant, tendons,
initial_free_lengths)`. `command(payout_rates, external_effort=...)` creates the
complete command. Mechanics and every `TendonActuatorState` commit together
only when command bounds, mechanics evidence, source/candidate tendon evidence,
finiteness, and the `TendonDrivenRodActuationLedger` pass. This profile excludes
contact, capstan friction, pressure, intrinsic-strain, variable-stiffness, and
magnetic actuation; those have separate contracts and are not silently routed
through the tendon plant.

`CapstanTendonFrictionPlan` is a separate semismooth variational-inequality
profile with span tensions, stress-free lengths, slip, dissipation, and power
evidence. It is not the frictionless tendon plant.

The advanced evaluators have exact qualified/excluded tuples:

| Profile | Qualified tuple | Explicit exclusions |
| --- | --- | --- |
| Reduced tube chamber | spatial reduced rod; fixed material stations; prescribed cross-sectional areas; dead volume; closed caps | deformable cross-sections, interacting chambers, valves, compressors, leakage, mass-flow networks, thermal networks, vacuum, volumetric bodies |
| Regulated tube pressure | bounded target gauge pressure with bounded rise/fall rates | valves, compressors, leakage, thermal networks |
| Sealed tube pressure | fixed charge scale and polytropic exponent | valves, compressors, leakage, thermal networks |
| Intrinsic strain | calibrated finite mode shapes with bounded rate-limited activation | hysteresis, shape-memory phase change, dielectric field solve, thermal networks, swelling transport |
| Variable stiffness | calibrated bounded stiffness interpolation with rate-limited activation | jamming, hysteresis, phase change, rate-dependent modulus, damping modulation, topology change |
| Affine magnetic | affine uniform field and gradient per current, bounded currents and positions, fixed material dipoles | nonlinear ferromagnetics, hysteresis, mutual fields, Maxwell solves, coupled RL circuits |

Passing an excluded feature flag is rejected during plan construction. These
are actuator evaluations, not claims of a combined multiphysics plant.

## Continuum tasks, observations, control, and inference

### IK and trajectory tasks

`ContinuumPositionTask`, `ContinuumOrientationTask`, `ContinuumPoseTask`, and
`ContinuumShapeTask` bind targets to one prepared physical reconstruction;
`ContinuumPostureTask` binds a reduced-coordinate target. Orientation uses a
scalar-first quaternion and the principal SO(3) logarithm chart.
`ContinuumInverseKinematicsPlan` requires a nonempty, duplicate-free task set
from one reconstruction and returns source, candidate, accepted state plus task,
feasibility, chart, finite, and optimizer evidence.

`ContinuumDifferentialIKPlan` linearizes the same fixed queries into a native
convex QP with explicit velocity bounds. Its accepted velocity/state remain
separate from the candidate. `SmoothReducedRodTrajectoryPlan` is limited to a
fixed-base, contact-free reduced plant; it lowers a fixed-horizon task/control
problem to SQP and performs an authoritative accepted plant replay through the
state and control codecs. It does not optimize through contact-mode changes.

### Observations and sensors

`SoftObservationPlan` composes any of these fixed-shape queries:

- `SoftReducedStateQueryPlan`: dimensionless coefficients and/or physical-time
  coefficient rates;
- `SoftFrameQueryPlan`: material poses and selected `body`, `world_origin`, or
  `frame_world` twists at physical arc lengths;
- `SoftStrainQueryPlan`: total strain and optional reduced increment;
- `SoftTendonQueryPlan`: length, rate, unilateral tension, and stored energy;
- `SoftEnergyLoadQueryPlan`: native-authority energy/load views and optional
  accepted-step ledger.

`prepare_soft_observation_plan` binds the exact ABI to one prepared fixed-base
reduced plant. `SoftObservationLayout` records every component's name, unit,
frame, query origin, and flat slice. `SoftSensorPlan` adds declared bias,
noise, sample period, and latency. `SoftSensorState` is caller-owned and
contains bias/sample-hold state; observation does not mutate it. Randomness is
derived from the accepted plant key. `SoftRobotObservation` preserves ideal,
bias, noise, age, freshness, finite, and query-valid evidence rather than
returning an unqualified vector.

### Calibration, co-design, and MPC

`PositiveParameterMap`, `BoundedParameterMap`, and `SPDParameterMap` map latent
coordinates to physically constrained values. `ReducedRodParameterization`
requires an exact named map set.

A `ReducedRodCalibrationProblem` requires at least one training and one held-out
`CalibrationExperiment`; optional validation becomes mandatory only when
`CalibrationAcceptance.require_validation` is true. Only training residuals
enter the solve and identifiability SVD. Acceptance separately requires the
declared training, validation, held-out RMSE/maximum-absolute bounds, route
validity, full-rank/condition evidence, finite candidate realization, and any
admissibility predicate. Failure retains the source realization.

`SoftRobotCoDesignProblem` is fixed-mode all-at-once state/morphology/actuator/
controller design. `FixedModeDerivativeEvidence` must match the exact
morphology, actuator, control, mode, and primal-result identities. At least one
disjoint `CoDesignHeldOutScenario` can reject but never train the design. The
result atomically selects the candidate or source realization.

`build_soft_plant_mpc` binds sampling MPC to one declared soft-plant
realization/mode. Sampling, sorting, and model resampling are not advertised as
differentiable. `SoftPlantMPCResult` adds the selected accepted replay and its
fixed-mode derivative-domain evidence to the finite-work sampling result. No
sampling or SQP result certifies global optimality or robust stability.

## Contact capsules, history, CCD, and atomic contact

`RodCapsuleGeometryPlan` gives each segment a circular radius and stable
participant/body/material/patch/feature IDs. Its optional solver clearance and
proxy error remain separate from the physical radius. A prepared reduced-rod
participant maps witnesses to exact kinematics and true-dual effort pullbacks,
with `RodCapsuleDualityEvidence` for virtual power.

`RodContactSearchPlan` owns a fixed witness capacity, activation distance,
plane capacity, and exact `dense` or `bvh` route. Search returns a full
`RodContactWitnessBatch` and `RodContactSearchEvidence`; overflow or incomplete
traversal is failure, never truncation. Canonical route keys distinguish
capsule–capsule and capsule–plane features and filter adjacent self-contact.

`RodContactManifoldState` is complete persistent contact history: occupied and
active masks, route keys, witnesses, normals, tangent bases, impulses,
stick/slip, age, retention, and material revision. Its transition remaps by
canonical route key; it does not match contacts by array position.

`RodContactCCDPlan` uses conservative advancement over start/end centerlines.
`RodContactCCDEvidence` distinguishes a full-step-safe interval, an impact, and
a certified safe prefix. A safe prefix alone is not a successful full requested
step. `CompositeContactResponse` composes participant blocks through the true
Delassus action and returns candidate and accepted impulses with cone,
complementarity, equation, dissipation, duality, and finite evidence.

`PreparedReducedRodContactPlant` is exactly one of these spatial 3-D tuples:

- `fixed-base-circular-capsule-plane-self-frictionless`;
- `fixed-base-circular-capsule-plane-self-isotropic-coulomb`.

The plant requires a spatial reduction, prepared capsule participant/search,
`RodContactCCDPlan`, reduced integrator, and fixed capacities. It covers either
the entire requested interval or rejects it. Reduced mechanics, constitutive
history, contact history, clock, and key commit atomically. Search overflow,
CCD safe-prefix-only, cone failure, penetration, non-dissipative friction,
conservation, finite, or ledger failure retains the entire source state. The
profile excludes nonspherical cross-sections, adhesion, rolling/spinning laws,
plastic impact, deformable contact patches, topology change, and gradients
through contact-mode changes.

See `examples/soft_robot_contact.py` for this exact profile. It supersedes the
older statement that rod contact was only an operator utility; the rigid
articulation contact utility remains fixed-route and non-atomic.

## Floating and hybrid profiles

`FloatingReducedRodPlan` composes a free SE(3) action with a native-discrete
spatial reduced rod. Point storage has a seven-coordinate base pose plus reduced
coefficients; the tangent has a six-coordinate base twist plus coefficient
velocities. The declared `body` or `spatial` twist convention controls the
maps. `PreparedFloatingReducedRod` provides contact-free mass, inverse mass,
bias, gravity, energy, momentum, forward/inverse dynamics, and true-dual load
results. `FloatingReducedRodPlant` is an atomic fixed-topology plant with a
complete generalized-effort command. It excludes contact and topology change.

Rigid–soft composition uses explicit ports rather than inspecting child
internals: `PreparedReducedRodPlantPort`, `FloatingReducedRodPlantPort`, and
`TendonDrivenRodPlantPort`. `RigidFrameAttachmentPlan` and
`SoftEndpointAttachmentPlan` define fixed child frames;
`RigidSoftAttachmentPlan` pairs them. `route_attachment_wrench` shifts one
attachment-frame wrench into equal-and-opposite child covectors and records
power evidence.

`HybridRigidSoftPlant` is the exact tuple

`(two identity-bound AbstractDiscretePlant children, fixed attachment topology,
source-explicit synchronized step, optional fixed duration)`.

The child commands and attachment wrench tuple are complete and fixed-shape.
Both child payloads, shared clock/key, and coupling evidence commit atomically.
This is one source-explicit partitioned step, not an implicit constraint solve,
contact solve, variable attachment topology, or asynchronous/subcycled
coupling claim.

## FEM, MPM, and MJX profiles

### Fixed-mesh FEM

`FEMSoftPlant` is the exact profile

`(native transient FEM, fixed mesh, implicit-newmark step, one selected
linear-elastic/hyperelastic/viscoelastic constitutive capability, declared
pressure/fiber/body-force routes, declared region-displacement/region-force
sensors, exact state/control codecs, atomic replay)`.

The manifest uses these exact capability IDs when the corresponding route is
present:

- `phydrax.soft-fem.constitutive.linear-elasticity.v1`,
  `phydrax.soft-fem.constitutive.hyperelasticity.v1`, or
  `phydrax.soft-fem.constitutive.viscoelasticity.v1`;
- `phydrax.soft-fem.actuation.region-pressure.v1`;
- `phydrax.soft-fem.actuation.region-fiber.v1`;
- `phydrax.soft-fem.actuation.region-body-force.v1`;
- `phydrax.soft-fem.sensor.region-displacement.v1`;
- `phydrax.soft-fem.sensor.region-force.v1`;
- `phydrax.soft-fem.codec.complete-state-exact.v1`;
- `phydrax.soft-fem.codec.control-exact.v1`;
- `phydrax.soft-fem.transaction.atomic-replay.v1`.

Its robotics profile supports only `step` and `sensors` on the prepared device
and dtype; step names solver `implicit-newmark`. Differentiability is
`conditional` only when the FEM plan has a derivative policy, otherwise `none`.
`forward-kinematics`, `smooth-dynamics`, `contact`, `model-batching`, `jit`,
`vmap`, `jvp`, and `vjp` expose no callable. The manifest explicitly rejects
`phydrax.soft-fem.topology.remesh.v1`,
`phydrax.soft-fem.topology.fracture.v1`, and
`phydrax.soft-fem.contact.v1`.

### Fixed-topology MPM

`MPMSoftPlant` is the exact profile

`(native explicit MPM, fixed particle ownership, topology generation zero,
complete particle/material-history/last-grid state, optional fixed or commanded
body acceleration, particle-region and oriented grid-surface observations,
casewise atomic rollback)`.

Every manifest contains `fixed-topology`, `particle-region-observation`, and
`grid-surface-observation`; it additionally contains exactly one applicable
`body-force-command` or `fixed-body-force` feature when prepared that way.
`MPMSoftResolutionEvidence` binds particle capacity, grid shape, field count,
and preparation identity to optional `MPMSoftResolutionRequirement` values.
The robotics profile supports only `step` with solver `explicit-mpm` and
`sensors`, both conditionally differentiable on the exact prepared device and
dtype. `forward-kinematics`, `smooth-dynamics`, `contact`, `model-batching`,
`jit`, `vmap`, `jvp`, and `vjp` expose no callable. The plant rejects `contact`,
`amr`, and `topology-change`; it also rejects a requested body-force feature not
bound by the prepared acceleration route.

### MJX-JAX plant

Importing robotics does not import MuJoCo. `mjx_availability()` requires both
`mujoco` and `mujoco-mjx` with exactly matching 3.12.x base releases.
`prepare_mjx_adapter` accepts an already compiled public `mujoco.MjModel` and
builds a closed `MJXPreparedModelManifest`. There is no provider substitute.

The static `MJX_JAX_PROFILE` supports only:

- `step`: devices `cpu`, `gpu`, `tpu`; dtypes `float32`, `float64`; solvers
  `cg`, `newton`; conditional differentiability; and exactly these contact
  features: `box-box`, `box-mesh`, `capsule-box`, `capsule-capsule`,
  `capsule-cylinder`, `capsule-ellipsoid`, `capsule-mesh`,
  `cylinder-cylinder`, `ellipsoid-cylinder`, `ellipsoid-ellipsoid`,
  `hfield-box`, `hfield-capsule`, `hfield-mesh`, `hfield-sphere`, `mesh-mesh`,
  `plane-box`, `plane-capsule`, `plane-cylinder`, `plane-ellipsoid`,
  `plane-mesh`, `plane-sphere`, `sphere-box`, `sphere-capsule`,
  `sphere-cylinder`, `sphere-ellipsoid`, `sphere-mesh`, and `sphere-sphere`;
- `sensors`: devices `cpu`, `gpu`, `tpu`; dtypes `float32`, `float64`;
  conditional differentiability; no solver or contact-feature qualifier.

It marks `forward-kinematics`, `smooth-dynamics`, `contact` as a standalone
operation, `model-batching`, `jit`, `vmap`, `jvp`, and `vjp` unsupported because
the adapter exposes no callable for those operations. `MJX_WARP_PROFILE` marks
every operation unsupported: no MJX-Warp adapter callable is implemented, and
JVP/VJP additionally have no automatic differentiation.

Preparation accepts only manifest-recognized integrators (`euler`, `rk4`,
`implicitfast`), solvers (`cg`, `newton`), cones (`pyramidal`, `elliptic`),
Jacobian modes (`dense`, `sparse`, `auto`), joint/geometry/actuator/equality/
sensor/tendon-wrap enums, and enabled bits. It rejects flexible bodies, plugin
state, unknown enabled bits, unsupported collision pairs, `implicitfast` with
fluid drag, mesh/hfield margin or gap, elliptic `condim=1`, and unsupported
contact-sensor matching/reduction combinations before device transfer.

`MJXAdapter` is itself an `AbstractDiscretePlant`. Its complete payload is
`MJXState(opaque, epoch, sensor_epoch)` inside `PlantRuntimeState`; the state
schema is the canonical complete `mjx.Data` PyTree and is not a separate public
MJX schema object. Use the shared plant lifecycle:

```python
reset = adapter.reset(key, adapter.parameters)
command = adapter.control(reset.accepted_state)
step = adapter.step(context, reset.accepted_state, command, adapter.parameters)
stale = step.accepted_state
refresh = adapter.refresh(stale, MJXObservationRequest())
observation = adapter.observe(refresh.accepted_state, MJXObservationRequest())
```

A successful step increments the state epoch and leaves the sensor epoch stale.
`observe` rejects a stale sensor-bearing request. `refresh` runs `mjx.forward`
and updates sensor epoch only for complete finite accepted cases. The shared
`PlantStepResult` is authoritative for every step.

## Qualification boundary

The native rod/contact examples and benchmark produce runtime evidence for the
exact cases they execute. They do not establish a general production-readiness,
safety, real-time, hardware, durability, or regulatory claim. FEM, MPM, hybrid,
floating, advanced-actuator, inference, and MJX support remains limited to the
qualified tuples above until a caller records runtime evidence for its own
prepared problem, device, dtype, capacities, tolerances, and provider releases.

See the [soft robotics API](api/applications/soft_robotics.md),
`examples/soft_robot_tendon.py`, `examples/soft_robot_contact.py`, and
`tools/robotics_soft_rod_benchmarks.py`.

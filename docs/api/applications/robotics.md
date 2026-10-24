# Robotics applications

## Robot adaptation, kinematics, environments, and backends

::: phydrax.applications.robotics
    options:
      members: true
      show_root_heading: true
      members_order: source

Soft-robot-specific rods, actuators, contact, continuum tasks, observations,
inference, floating/hybrid plants, FEM, MPM, and the MJX plant lifecycle are
grouped in the [soft robotics API](soft_robotics.md).

The robotics members above include the native model, URDF and compiled-MuJoCo
adapters, and the complete-state native plant. Their supported profiles are
summarized below; the [native robotics guide](../../guides_robotics.md) and the
[native robotics qualification guide](../../guides_native_robotics_qualification.md)
describe usage and evidence.

### Native robot model

`NativeRobotModel` is the one executable model authority. It composes the
existing particle, rigid-body, joint-graph and reduced-articulation plans and
their zero-configuration reference with named `RobotFramePlan` frames and
optional `ScalarJointMotorPlan`, `ScalarJointDampingPlan`,
`ReducedJointStopPlan` and `RobotCollisionGeometryPlan` components; absent
components are `None`, never inferred from source bounds.
`lower_native_robot_tree` builds one from COM-centered bodies and
`NativeTreeJoint` edges (`NativeTreeJointKind = Literal["fixed", "hinge",
"prismatic"]`). The supported profile is three-dimensional and fixed-base with a
fixed root of strictly positive mass; ball and distance joints, floating bases,
loops, tendons and stateful actuators are refused. `model.prepare()` returns
`PreparedNativeRobotModel`, binding every owner and the lifted-coordinate
winding owner to the same reference. `RobotNameIDMap` assigns deterministic
signed 64-bit sorted-name IDs; `NativeRobotModelManifest` records counts,
executed capabilities and component identities of `model.model_id`.

`RobotCollisionGeometryPlan` declares body-attached positive-radius spheres and
capsules (endpoint order defines the capsule axis) and static planes
`n·x = offset` with free space on `n·x > offset`; at least one body primitive is
required. Every material pair that can meet needs an explicit
`RobotContactMaterialTable` entry with `0 <= dynamic <= static`; there is no
mixing rule. `RobotContactLaw` is exactly `"inelastic-coulomb-hard-contact"`:
the native position-level law, not a provider's regularized contact.

### Source adaptation

`RobotAdaptation[SourceEvidence]` is the single adaptation result: `model`
(`NativeRobotModel`), `report` (`AdapterReport`) and source-specific `evidence`
(`AbstractRobotSourceEvidence`). The former flat fields and forwarding
properties are removed:

| Former access | Current access |
| --- | --- |
| `particles`, `particle_plan` | `model.particles` |
| `bodies`, `body_plan` | `model.bodies` |
| `joints`, `joint_plan` | `model.joints` |
| `articulation`, `articulation_plan` | `model.articulation` |
| `reference`, `reference_kinematics` | `model.reference` |
| `dimensions`, `dimensional_contract` | `model.dimensions` |
| `link_ids`, `link_name_to_id`, `link_id_to_name` | `model.body_names` (`name_to_id`, `id_to_name`) |
| `joint_ids`, `joint_name_to_id`, `joint_id_to_name` | `model.joint_names` |
| `target_id` | `model.model_id` |
| `negotiation` | `report.negotiation` |

`parse_urdf_text`/`parse_urdf_file` return `RobotAdaptation[URDFFormatEvidence]`.
URDF supplies no contact law, actuator or physical stop: callers declare
`URDFCollisionPolicy` (materials, link materials, `URDFStaticPlane` planes,
exclusions), explicit `URDFJointMotor` mappings and
`joint_stops="native-unilateral-stops"` (`URDFJointStopPolicy`; the default
`"evidence-only"` records bounds without stops). Only URDF `<sphere>` collisions
are lowered (`URDFCollisionEvidence`); boxes, cylinders and meshes are
unwaivable losses when collision is requested, and no URDF capsule is invented.

`adapt_mujoco_model(model, contact=..., joint_stops=...)` adapts one already
compiled public `mujoco.MjModel` on the host (`mujoco_host_availability()`
probes the optional provider) and retains no provider object. Its closed
profile is:

- exactly one jointless physical base welded to the world, positive body
  masses, at most one hinge or slide joint per body (jointless bodies become
  fixed edges);
- compiled COM offsets, inertial quaternions and principal inertias without an
  extra parallel-axis shift; native coordinate `qpos - qpos0`, velocity `qvel`;
- stateless joint motors only: joint transmission, no activation dynamics,
  fixed gain, zero bias, scalar gear, optional `ctrlrange` clamp and symmetric
  `forcerange` (native gear `gain · gear`, effort limit `|gear| · F`);
- executed linear DOF damping; body-origin and site frames;
- joint position/velocity, clock and world-referenced frame
  position/quaternion/linear/angular-velocity sensors without noise, cutoff,
  delay, interval or history;
- with an explicit contact policy, colliding spheres, capsules and world planes.

Free/ball/multi-joint bodies, flex, tendons, equalities, mocap, plugins,
activation state, history buffers, gravity compensation, fluid options,
stiffness, armature, actuator damping/delay, servo bias, nonjoint transmissions,
other geoms and sensors, nonzero limit margins and dry friction loss when
executed are refused with `MujocoModelAdaptationError`. The evidence retains
timestep, integrator, gravity and option flags; the adapter does not map MuJoCo
integrators, `solref`/`solimp`, `condim`, margins/gaps or friction mixing.

`MujocoModelEvidence` separates two kinds of result. Smooth-mechanics mapping —
bodies, inertias, joint frames and `qpos0`, motors, damping, frames, sensors —
is the same model content (`MujocoBodyMap`, `MujocoJointMap`,
`MujocoActuatorMap`, `MujocoSensorMap`, `coordinate_map`). Contact and limits are
not mapped: `MujocoNativeContactTransformation` (explicit
`RobotContactMaterialTable`, rationale) replaces MuJoCo's regularized soft
contact by the native hard-contact law, `MujocoContactExclusion` removes every
contact, and `MujocoJointStopTransformation` replaces regularized limits by
native hard stops on `[range_low - qpos0, range_high - qpos0]`. Each is recorded
as an interpretation-changing transformed or dropped loss with the caller's
rationale as its waiver. `MujocoModelEvidence.contact_mode` is one of
`"no-compiled-contact"`, `"excluded"` or
`"native-inelastic-coulomb-hard-contact"`, `joint_limit_mode` one of `"none"`,
`"disabled-in-source"` or `"native-hard-stops"`, and each `MujocoGeomMap.role`
is `"primitive"`, `"plane"`, `"excluded-contact"` or `"non-colliding"`. A model
with compiled contact pairs or executed limits but no policy is refused. Pair
admission follows MuJoCo's explicit pairs, weld/parent filtering, excludes and
`contype`/`conaffinity`, and must be all-or-none between two native bodies; the
host pair inventory is refused before scanning when `G(G-1)/2` exceeds
`maximum_geom_pair_checks`, and the native model keeps only sparse excluded body
pairs.

### Complete-state native plant

`prepare_articulated_plant(model, NativeArticulatedPlantPlan(step_policy,
gravity=..., actuation_mode=..., case_ndim=..., ...))` returns
`PreparedArticulatedPlant`, an `AbstractDiscretePlant` over the native contact
step. Its `ArticulatedPlantState` payload is the complete mechanical history:
principal configuration and integer winding, velocity, contact manifold, stop
impulses and branches, last command, transition and sensor epochs, sensor
samples (`ArticulatedPlantSensors`), and the geometry/material revisions. The
runtime envelope owns clock, step index and typed PRNG key. Reset admits the
initial payload with the step's static admissibility. A step commits a case's
whole payload, clock and key only when the native step covered the full
requested interval and the revisions match the plan (otherwise
`ArticulatedPlantStatus.REVISION_MISMATCH`); failed cases keep the complete
prior state, and `ArticulatedPlantStepEvidence` keeps every staged result.
`observe` projects accepted states without advancing them
(`ArticulatedPlantObservation.fresh` is `sensor_epoch == epoch`), `refresh`
resamples sensors transactionally, and `control_view` gives lifted coordinates
and velocity. Commands are `ArticulatedPlantCommand` in the declared actuation
units; autonomous plants take no command. Gravity is the shared dynamic
parameter `ArticulatedPlantParameters`; bind it with `bind_parameters()`.
Cases are explicit leading axes (`case_ndim`) over one pure single-case
transition. Checkpoint, restore and replay are the shared plant lifecycle.

## Reduced articulation

::: phydrax.discretization.ReducedArticulationPlan

---

::: phydrax.discretization.PreparedReducedArticulation

---

::: phydrax.discretization.ReducedArticulationState

---

::: phydrax.discretization.ArticulationKinematics

---

::: phydrax.discretization.ArticulationDualityEvidence

## Reduced rigid dynamics

::: phydrax.discretization.ReducedDynamicsStatus

---

::: phydrax.discretization.ReducedEnergyResult

---

::: phydrax.discretization.ReducedMassMatrixResult

---

::: phydrax.discretization.ReducedBiasTermsResult

---

::: phydrax.discretization.ReducedInverseDynamicsResult

---

::: phydrax.discretization.ReducedForwardDynamicsResult

---

::: phydrax.discretization.ReducedSemiImplicitVelocityEulerStepPolicy

---

::: phydrax.discretization.ReducedSemiImplicitVelocityEulerStepDiagnostics

---

::: phydrax.discretization.ReducedSemiImplicitVelocityEulerStepResult

---

::: phydrax.discretization.reduced_energy

---

::: phydrax.discretization.reduced_mass_matrix

---

::: phydrax.discretization.reduced_bias_terms

---

::: phydrax.discretization.reduced_inverse_dynamics

---

::: phydrax.discretization.reduced_forward_dynamics

---

::: phydrax.discretization.reduced_semi_implicit_velocity_euler_step

## Prepared articulated inertia

`prepare_articulated_mass(articulation, q, policy=...)` prepares the ABA inertia
factors of `M(q)` once: transforms, motion subspaces, articulated inertias,
projected columns and pivots as dynamic arrays bound to the exact configuration
and mass properties (`ArticulatedMassBinding`). Moving pivots must be finite,
positive and above the relative threshold; there is no jitter or sign repair,
and a positive pivot is not an eigenvalue bound. `inverse_mass_operator` maps
the generalized effort dual space to generalized velocity and `mass_operator`
maps back through an independent recursion; neither materializes a dense
matrix. `apply_inverse` accepts one vector or an explicit RHS axis (bounded by
`ArticulatedMassPolicy.maximum_rhs`), certifies each residual and returns zero
`value` with an inspectable `candidate` on failure; `is_current` and the
`STALE_BINDING` status refuse a different configuration or inertia.
`reduced_forward_dynamics(..., mass_policy=...)` returns the factor it used in
`ReducedForwardDynamicsResult.mass`; without an explicit policy its pivot
acceptance keeps the former finite-positive rule.

::: phydrax.discretization.prepare_articulated_mass

---

::: phydrax.discretization.PreparedArticulatedMass

---

::: phydrax.discretization.ArticulatedMassPolicy

---

::: phydrax.discretization.ArticulatedMassBinding

---

::: phydrax.discretization.ArticulatedInertiaFactor

---

::: phydrax.discretization.ArticulatedInverseMassOperator

---

::: phydrax.discretization.ArticulatedMassOperator

---

::: phydrax.discretization.ArticulatedMassEvidence

---

::: phydrax.discretization.ArticulatedMassActionResult

---

::: phydrax.discretization.ArticulatedMassProductResult

---

::: phydrax.discretization.ArticulatedMassAdjointEvidence

---

::: phydrax.discretization.ArticulatedMassStatus

## COM-centered rigid inertial realization

::: phydrax.discretization.RigidBodyMassProperties

---

::: phydrax.discretization.RigidInertialCoordinates

---

::: phydrax.discretization.RigidInertialParameters

---

::: phydrax.discretization.RigidInertialEvaluation

---

::: phydrax.discretization.RigidInertialParameterization

---

::: phydrax.discretization.RigidBodyReferenceFrameRebase

---

::: phydrax.discretization.RigidInertialRealization

---

::: phydrax.discretization.realize_rigid_body_plans

## Fixed-route articulated contact

::: phydrax.applications.contact.ContactConeProgram

---

::: phydrax.applications.contact.ContactConeSolverPlan

---

::: phydrax.applications.contact.ContactConeNumericRevision

---

::: phydrax.applications.contact.ContactConeEvidence

---

::: phydrax.applications.contact.ContactConeResult

---

::: phydrax.applications.contact.make_articulated_contact_participant

---

::: phydrax.applications.contact.build_contact_velocity_operator

---

::: phydrax.applications.contact.build_delassus_operator

---

::: phydrax.applications.contact.build_mixed_contact_velocity_operator

---

::: phydrax.applications.contact.contact_duality_evidence

---

::: phydrax.applications.contact.prepare_articulated_contact

---

::: phydrax.applications.contact.PreparedArticulatedContact

---

::: phydrax.applications.contact.solve_articulated_contact

---

::: phydrax.applications.contact.apply_articulated_contact_impulse

---

::: phydrax.applications.contact.ArticulatedContactPreparationEvidence

---

::: phydrax.applications.contact.ArticulatedContactDualityEvidence

---

::: phydrax.applications.contact.ArticulatedContactEvidence

---

::: phydrax.applications.contact.ArticulatedContactResult

## Result-preserving discrete evolution

::: phydrax.dynamics.DiscreteTransitionResult

---

::: phydrax.dynamics.DiscreteTransitionEvidence

---

::: phydrax.dynamics.DiscreteEvolution

---

::: phydrax.dynamics.EvolutionStep

---

::: phydrax.dynamics.EvolutionTrajectory

---

::: phydrax.dynamics.evolve

## Status-aware rollout and sampling MPC

::: phydrax.control.DiscreteControlDynamics

---

::: phydrax.control.ControlTrajectory

---

::: phydrax.control.SamplingMPCStatus

---

::: phydrax.control.SamplingMPCPlan

---

::: phydrax.control.SamplingMPCState

---

::: phydrax.control.SamplingMPCEvidence

---

::: phydrax.control.SamplingMPCResult

---

::: phydrax.control.plan_sampling_mpc

---

::: phydrax.control.initialize_sampling_mpc

---

::: phydrax.control.shift_sampling_mpc_state

---

::: phydrax.control.solve_sampling_mpc

`SamplingMPCRealizations(parameters, realization_ids, weights=..., support_mask=...,
posterior_id=..., campaign_id=..., policy="fixed" | "resample")` declares a
fixed-capacity model axis of physical parameter realizations: every finite
parameter leaf carries the common leading capacity axis, IDs are unique, and
the weights/support define the risk aggregation (`SamplingMPCRealizationPolicy`
selects fixed or resampled realizations). Explicit realizations require a
scalar-case problem and exclude separately supplied `model_weights` or
`model_support`; a `SamplingMPCRealizationBinding` maps each realization into
the problem's `args` and needs a nonempty `realization_binding_id` when custom.
Only `problem.args` changes per realization; the complete initial plant state is
never rebuilt from its continuous view. `SamplingMPCSelectedReplay` keeps the
selected rollout's states, controls, candidate and accepted transition
payloads, per-model objectives and feasibility, the realization indices,
weights and IDs, and `plant_history` for complete-state plants. With explicit
realizations and `PlantControlDynamics`, planning refuses a plan whose
`model_count × retained_storage_bytes` exceeds the dynamics'
`maximum_retained_bytes`, and the replay's `plant_history` carries an explicit
realization axis (`realization_count`) with time-leading retained histories.
Realizations are not plant cases: `state_at` refuses until
`select_realization(index)` removes the model axis, leaving the plant's own case
axes unchanged.

::: phydrax.control.SamplingMPCRealizations

---

::: phydrax.control.SamplingMPCSelectedReplay

## Complete-state plant control

`PlantControlDynamics(plant, state_codec, control_codec, parameters=...,
retention="endpoint" | "full", maximum_retained_bytes=...)` rolls out an
explicitly controlled `AbstractDiscretePlant` with the complete
`PlantRuntimeState` as scan carry. The state codec is only the controller's
continuous view; the next plant step receives the complete accepted state,
including discrete modes and histories, and failed candidates never feed
forward. Autonomous plants are refused. `ControlProblem` stores that complete
initial state; its public `initial_state` is the continuous view.
`ControlTrajectory.plant_history` is a `PlantControlHistory` with the initial,
final and final-candidate complete states; `"full"` retention additionally
keeps time-leading accepted and candidate `PlantRuntimeHistory` stacks whose
`state_at(index)` restores an ordinary runtime state for checkpointing; a
history with a `realization_count` requires `select_realization(index)` first.
`maximum_retained_bytes` bounds the per-rollout retained-storage estimate of
either mode and is checked before rollout. Sampling MPC accepts
`PlantControlDynamics`, evaluates every candidate from the exact complete
initial state, and keeps the selected replay's `plant_history`. Mode-changing
forward rollout is supported; no derivative through contact, stop or winding
transitions is claimed, and fixed-branch derivatives belong to the native
step's `step_jvp`/`step_vjp`. `parameters` is a required keyword
`PlantParameters`; it is never inferred from the plant. For the native
articulated plant pass `parameters=plant.bind_parameters()`.

::: phydrax.control.PlantControlDynamics

---

::: phydrax.control.PlantControlHistory

---

::: phydrax.control.PlantRuntimeHistory

## Manifold Radau transcription

::: phydrax.control.ManifoldCollocationStages

---

::: phydrax.control.ManifoldCollocationEvidence

---

::: phydrax.control.ManifoldRadauCollocationDefects

---

::: phydrax.control.manifold_radau_stages

---

::: phydrax.control.manifold_radau_collocation_defects

## Shared complete plant lifecycle

::: phydrax.dynamics.AbstractDiscretePlant

---

::: phydrax.dynamics.PlantRuntimeState

---

::: phydrax.dynamics.PlantStepContext

---

::: phydrax.dynamics.PlantResetResult

---

::: phydrax.dynamics.PlantStepResult

---

::: phydrax.dynamics.PlantCheckpoint

---

::: phydrax.dynamics.PlantReplayResult

---

::: phydrax.dynamics.PlantStateVectorCodec

---

::: phydrax.dynamics.ControlVectorCodec

## Reduced rod maps

::: phydrax.applications.solid_mechanics.ReducedRodPlan

---

::: phydrax.applications.solid_mechanics.PreparedReducedRod

---

::: phydrax.applications.solid_mechanics.lift_configuration

---

::: phydrax.applications.solid_mechanics.lift_velocity_operator

---

::: phydrax.applications.solid_mechanics.lift_effort_pullback_operator

---

::: phydrax.applications.solid_mechanics.evaluate_reduced_rod

## Bounded resource contracts

::: phydrax.interchange.ResourceLimits

---

::: phydrax.interchange.ResourceManifest

---

::: phydrax.interchange.BoundedResource

---

::: phydrax.interchange.ResourceReadError

---

::: phydrax.interchange.bounded_resource_from_bytes

---

::: phydrax.interchange.read_bounded_resource

---

::: phydrax.interchange.account_bounded_resource

## Interchange negotiation and composition

::: phydrax.interchange.AdapterStatus

---

::: phydrax.interchange.AdapterLoss

---

::: phydrax.interchange.AdapterRequirement

---

::: phydrax.interchange.AdapterCapability

---

::: phydrax.interchange.AdapterWaiver

---

::: phydrax.interchange.AdapterNegotiationResult

---

::: phydrax.interchange.AdapterReport

---

::: phydrax.interchange.negotiate_adapter

---

::: phydrax.interchange.compose_adapter_reports

---

::: phydrax.interchange.require_lossless

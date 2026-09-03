# Native robotics

Phydrax robotics is a composition of existing numerical owners rather than a
second simulation stack. `phydrax.applications.robotics` owns robot adaptation,
frame inverse kinematics, immutable task environments, backend capability
contracts, and the optional MJX boundary. Reduced articulation and rigid
inertial realization live in `phydrax.discretization`; rollout and trajectory
optimization live in `phydrax.control`; articulated impact lives in
`phydrax.applications.contact`; reduced flexible rods live in
`phydrax.applications.solid_mechanics`; and conversion accounting lives in
`phydrax.interchange`.

The first native articulation scope is deliberately narrow: three-dimensional,
fixed-base trees containing fixed, hinge, and prismatic joints. It does not
provide floating bases or ball joints. Keeping that boundary explicit lets the
native path expose stable layouts, operator duality, fail-closed status, and
pure-JAX runtime actions without pretending to cover a more general mechanism.

## Contract map

| Concern | Public owner | Primary contracts |
| --- | --- | --- |
| URDF adaptation | `phydrax.applications.robotics` | `parse_urdf_text`, `parse_urdf_file`, `RobotAdaptation`, `URDFFormatEvidence` |
| Reduced tree kinematics and dynamics | `phydrax.discretization` | `ReducedArticulationPlan`, `PreparedReducedArticulation`, `reduced_inverse_dynamics`, `reduced_forward_dynamics` |
| Inertial design coordinates | `phydrax.discretization` | `RigidInertialParameterization`, `RigidInertialEvaluation`, `realize_rigid_body_plans` |
| Frame IK | `phydrax.applications.robotics` | `FramePositionTask`, `FrameOrientationTask`, `FramePoseTask`, `FrameInverseKinematicsPlan` |
| Articulated impact | `phydrax.applications.contact` | `prepare_articulated_contact`, `solve_articulated_contact` |
| Status-aware rollout and MPC | `phydrax.control` | `DiscreteControlDynamics`, `ControlTrajectory`, `plan_sampling_mpc` |
| Manifold transcription | `phydrax.control` | `manifold_radau_stages`, `manifold_radau_collocation_defects` |
| Task environments | `phydrax.applications.robotics` | `AbstractRobotTask`, `AbstractRobotEnvironmentWrapper`, `PreparedRobotEnvironment` |
| Reduced flexible members | `phydrax.applications.solid_mechanics` | `ReducedRodPlan`, `PreparedReducedRod`, `evaluate_reduced_rod` |
| Optional provider execution | `phydrax.applications.robotics` | `RoboticsBackendProfile`, `RoboticsOperationRequirement`, `MJXAdapter` |

Plans and prepared objects are immutable PyTrees. Stable IDs bind topology,
layouts, reference frames, and provider projections. Runtime arrays remain
separate from host preparation and negotiation. A `successful` or `valid`
field is evidence to inspect, not a value to infer from finite-looking output.

## Construct a native fixed-base articulation

A successful URDF adaptation returns plans; it does not silently prepare or
execute them. Preparation follows the ownership chain:

```python
from phydrax.applications.robotics import parse_urdf_text

adaptation = parse_urdf_text(urdf_text)
particles = adaptation.particles.prepare()
bodies = adaptation.bodies.prepare(particles)
graph = adaptation.joints.prepare(bodies, adaptation.reference)
articulation = adaptation.articulation.prepare(graph, adaptation.reference)
```

`adaptation.reference` is the exact zero-configuration COM-centred
`RigidBodyKinematics` used to prepare both the joint graph and articulation.
Substituting a different reference is rejected. `articulation.prepared_id`,
`state_layout`, and `input_layout` bind the resulting runtime contract. A tree
with no moving joint has no nonempty reduced state or input layout.

The adapter also returns deterministic signed 64-bit name maps. Use
`adaptation.link_ids.id_for_name(name)` and
`adaptation.joint_ids.id_for_name(name)` instead of assuming source order.
Prepared `joint_ids`, `joint_configuration_slices`, and
`joint_velocity_slices` identify the topological runtime order.

## URDF security and semantic loss

`parse_urdf_text` parses bounded UTF-8 text in memory and never resolves an
external resource. `parse_urdf_file` additionally requires `allowed_root`; the
normalized file must remain beneath that root and must fit `max_bytes`. Both
paths are host-only. Network access, DTD/entity/notation/CDATA declarations,
non-XML processing instructions, xacro/include expansion, and plugin execution
are disabled. No visual, collision, transmission, mesh, or plugin asset is
opened.

The accepted source is an unnamespaced URDF 1.0 connected tree with one root,
positive link masses, physically admissible inertias, and fixed, revolute,
continuous, or prismatic joints. Revolute and continuous joints lower to native
hinges. Joint limits and damping are retained as `URDFJointEvidence`; they are
not silently installed as dynamics forces or constraints. Link-frame to
COM-frame offsets and rotated inertia tensors are retained in
`URDFLinkEvidence`.

Every unsupported or dropped source field becomes an `AdapterLoss`. Visual
geometry is a declared non-interpretation-changing loss. Collision geometry,
transmissions, joint friction, and other dynamics-changing data are
interpretation-changing losses and make adaptation fail closed unless the
caller explicitly accepts the exact loss. A caller may supply an
`AdapterWaiver` by loss identity or name exact entries through
`waived_loss_paths`. Unknown paths, duplicate waivers, missing required
semantics, malformed sources, and unwaived interpretation changes are errors.
Inspect all of:

```python
adaptation.report.status
adaptation.report.losses
adaptation.negotiation.unwaived_losses
adaptation.evidence.loss_paths
```

`require_lossless(adaptation.report)` is stricter: it rejects every declared
loss, including a waived or non-interpretation-changing one. Multi-stage
conversion reports may be joined with `compose_adapter_reports`, which checks
source/target identity, format-profile continuity, stage ordering, and the
cumulative negotiation instead of concatenating prose.

## State, frame, and unit conventions

The native adapter declares SI units. URDF lengths are metres, masses are
kilograms, angles are radians, and inertia is kg·m² about the link COM. The
unique root is fixed to the world.

For a prepared articulation:

- `nq` is the number of configuration scalars and `nv` is the number of
  generalized velocities. In the supported fixed-base scalar-joint scope,
  `nq == nv`.
- Fixed edges own no reduced coordinate. Each hinge or prismatic edge owns one.
- Hinge configurations use principal-angle differences and integration;
  prismatic configurations are ordinary linear displacements.
- `ReducedArticulationState(configuration, velocity)` is the structured state.
  `pack_state` returns the rank-one array `[q, v]`, and `unpack_state` reverses
  it. `state_layout.geometry` supplies the same retraction convention to
  manifold-aware solvers.
- Rigid-body positions are COM positions. Orientations are scalar-first unit
  quaternions `(w, x, y, z)`. Homogeneous transforms map body-local coordinates
  into the world; `frame_transform` composes a body-to-frame local transform on
  the right.
- Spatial velocity rows are ordered as three linear components followed by
  three angular components. `RigidBodyLoad` pairs world-frame force and torque
  arrays with body-capacity shape `(N, 3)`.

`reference_configuration()` is zero by construction; it does not mean every
source link frame is at the world origin. The reference body poses include the
URDF joint origins and COM offsets.

## Forward kinematics, Jacobians, and duality

`forward_kinematics(q, v)` returns body poses, body velocities, homogeneous
transforms, and finite/success evidence. Omitting `v` means zero generalized
velocity. `body_transform` and `frame_transform` provide selected transforms
without defining a second frame registry.

The Jacobians are linear operators rather than mandatory dense arrays:

```python
body_J = articulation.body_jacobian_operator(q)
body_twists = body_J.mv(v)

frame_J = articulation.frame_jacobian_operator(q, body_id, local_position)
frame_twist = frame_J.mv(v)
generalized_load = frame_J.transpose_mv(frame_wrench)
```

`body_jacobian_operator` maps generalized velocity to all body spatial
velocities; `frame_jacobian_operator` maps to one body-attached frame.
Transpose actions are the covector pullbacks. For body loads,
`body_load_pullback(q, load, v)` returns both the generalized load and
`ArticulationDualityEvidence`, checking the measured equality between body
power and generalized power. This operator identity is the contract; forming a
dense Jacobian is optional.

## Inertial realization and reduced dynamics

`RigidInertialParameterization` binds unconstrained host coordinates to one
prepared three-dimensional rigid-body set. Softplus mass coordinates and
lower-triangular covariance coordinates produce positive masses and physically
realizable COM inertias. `evaluate` reports finite masks, SPD and principal
moment checks, pseudo-inertia checks, condition numbers, reconstruction
residuals, and distance from the source body data.

Changing inertia is an identity boundary, not an in-place mutation.
`realize_rigid_body_plans` returns fresh `ParticleSetPlan` and
`RigidBodySetPlan` values plus `RigidInertialEvaluation`; the caller must
prepare those plans and then reprepare dependent joints and articulation.
`evaluation.requires_repreparation` is always explicit.

Native reduced dynamics uses the convention
`M(q) a + c(q, v) + g(q) = τ + Jᵀw`:

- `reduced_mass_matrix` materializes the symmetric generalized mass operator
  and reports symmetry and positive-definiteness evidence.
- `reduced_bias_terms` separates velocity bias, gravity, and their sum.
- `reduced_inverse_dynamics` uses a topological recursive Newton–Euler action
  and reports its decomposition and external-load power residuals.
- `reduced_forward_dynamics` uses articulated-body dynamics and certifies the
  candidate acceleration by inverse reconstruction.
- `reduced_energy` reports kinetic, potential, and total energy for an explicit
  world gravity vector.
- `reduced_symplectic_step` proposes bounded symplectic Euler, checks dynamics,
  inverse/forward reconstruction, step size, finiteness, and an energy-work
  defect, then atomically accepts or returns the source state.

Result objects distinguish candidate values from accepted values. Failed
inverse dynamics returns zero accepted `generalized_effort`; failed forward
dynamics returns zero accepted `acceleration`; a failed step retains the source
`accepted_state`. Read `status`, `successful`, and the residual evidence before
using a candidate.

## Local frame inverse kinematics

A frame task fixes one immutable `(body_id, frame_id, local_transform)` and a
target:

- `FramePositionTask` contributes three positional residuals.
- `FrameOrientationTask` contributes three principal SO(3) logarithm residuals.
- `FramePoseTask` contributes both, with separate position and orientation
  weights.

Each task owns residual bounds and target-validity evidence.
`FrameInverseKinematicsPlan` combines one or more tasks and an optional posture
residual. `residual(q)` is compatible with JIT and AD. `solve` requires an
explicit least-squares method and termination policy; bounded methods require
`joint_bounds`, while unbounded methods require their absence. The result
contains optimizer, feasibility, chart, finiteness, task-residual, and final
kinematics evidence.

`implicit_solution` is narrower: it requires an unbounded method that advertises
implicit differentiation and returns sensitivity only for a regular local
root. Orientation residuals use the principal SO(3) chart and reject targets at
its singular margin. The IK layer is local least squares, not global IK, and it
does not certify discovery of every solution.

## Articulated contact

Articulated impact reuses fixed-route contact kinematics and the native cone
solver. `make_articulated_contact_participant` adapts an existing collision
surface, configuration space, forward kinematics, velocity action, and force
pullback. `build_contact_velocity_operator` produces the contact velocity
operator `G`; the supplied inverse-mass operator must be an endomorphism of the
same constrained generalized tangent space. `build_delassus_operator` composes
`W = G M⁻¹ G*` without routing through unconstrained body response.

`prepare_articulated_contact` checks that recorded route velocities match the
operator action, materializes the Delassus operator under an explicit policy,
checks symmetry and nonnegative diagonal evidence, and builds the fixed cone
program. `solve_articulated_contact` solves and applies an impulse through
`G*` and `M⁻¹`. It reports cone defects, post-contact normal feasibility,
contact/generalized power duality, and finiteness. If the certificate fails,
the applied impulse and velocity update are zero and the free velocity is
retained; `fail_closed` records that invariant.

This is an articulated Delassus impact path for declared fixed routes. It does
not provide general gradients through contact-mode changes, nor does it turn an
uncertified cone result into a valid transition.

## Status-aware control rollout

A robotics transition should return `DiscreteTransitionResult` with a candidate
state, accepted state, scalar `successful`, and scalar integer `status`.
`DiscreteControlDynamics.rollout` batches that contract over the declared case
axes and runs it with `jax.lax.scan`. It feeds accepted states forward, never
silently substitutes a failed candidate, preserves the first backend failure
status, marks the trajectory invalid from a failed transition onward, and does
not send nonfinite user controls into the plant.

`ControlTrajectory.status` is the control-level result; `backend_status` retains
the plant status. Both are required when diagnosing a rollout. This same
contract lets reduced symplectic mechanics, immutable environments, candidate
search, and trajectory optimization share failure semantics.

## Fixed-work sampling MPC

`plan_sampling_mpc` accepts a `ControlProblem` with `DiscreteControlDynamics`
and a knot-based public control parameterization. The plan declares candidate,
iteration, and elite counts; predictive or CEM updates; clip or reject bounds;
minimum proposal deviation; expectation, worst-case, or explicit risk
aggregation; model weights; and hold or zero warm-start tails.

`initialize_sampling_mpc` creates the Gaussian proposal without consuming
randomness. The caller owns the PRNG key passed to `solve_sampling_mpc`.
`shift_sampling_mpc_state` advances the proposal by exactly one knot. Evidence
retains every candidate control, per-model objective, rollout-valid and
feasibility mask, elite decision, proposal history, completed iteration count,
candidate evaluation count, and model rollout count.

Every candidate is shared across the `ControlProblem.case_shape`; model and
candidate axes remain distinct. The result is only the best valid sample
observed in the declared finite work. Sampling MPC is not global optimization
and provides no certified robustness guarantee.

## Manifold Radau defects

The articulation's packed state layout carries its hinge-aware geometry.
`manifold_radau_stages` retracts Radau stage combinations and the endpoint from
one anchor. `manifold_radau_collocation_defects` makes both conventions
mandatory:

```python
configuration_convention="retraction"
tangent_convention="shared-local"
```

Node states are manifold points. Stage rates are shared-local tangent values;
explicit dynamics are projected and converted with `to_local`, while implicit
dynamics receive rates converted with `from_local`. Endpoint differences use
`inverse_retract` from the same left-node anchor. Evidence separately reports
finiteness, membership, chart round-trip validity, equation validity, and their
conjunction. This avoids treating wrapped hinge coordinates as globally
Euclidean, but does not make a failed chart or defect a feasible trajectory.

## Immutable task environments

`PreparedRobotEnvironment` composes one rank-one `DiscreteSystem`, one
`AbstractRobotTask`, and an ordered tuple of
`AbstractRobotEnvironmentWrapper` values. The task owns its fixed-structure
state, observation, reward-component names, termination, and descriptor.
Wrappers own administrative state, action repetition, horizons, truncation, and
optional auto-reset. They do not mutate the plant or hide task termination.

`reset` initializes the plant, task, wrappers, caller-derived PRNG stream, and
clock. `step` performs the statically declared repeat count. Candidate plant,
task, wrapper, reward, and PRNG state commit together only if every attempted
mechanics substep succeeds. Otherwise the source state is returned, reward is
zero, termination/truncation is suppressed, and rollback evidence records the
failure. With auto-reset, `final_observation` remains the terminal observation
while `observation` belongs to the reset state.

Fixed work and immutable state improve compilation and auditability; they are
not a hard real-time scheduling guarantee.

## Reduced flexible rods

`ReducedRodPlan` defines a finite planar strain basis over an existing prepared
Cosserat rod. Stretch/shear basis shape is
`(segments, 2, coordinates)` and bend/twist basis shape is
`(segments - 1, 1, coordinates)`. The combined basis must have full column
rank. Optional fixed base position and orientation choose the reconstruction
anchor.

`prepare_reduced_rod` currently accepts only extensible planar `PreparedRod`
values. `ReducedRodState` packs coefficients and coefficient velocities.
`lift_reduced_rod_state` and `lift_reduced_rod_velocity` reconstruct native
states and rates through the exact lift JVP;
`pullback_reduced_rod_loads` applies the transpose action.
`evaluate_reduced_rod` reports native mechanics, generalized internal load,
energy, strain reconstruction, quadrature, fixed-base, and virtual-power
evidence. This is a reduced flexible-member model, not a ball-joint or
floating-base extension of rigid articulation.

## Optional backend capabilities and observation freshness

Importing the robotics package does not require MuJoCo or MJX.
`mjx_availability()` probes the optional provider through the shared backend
boundary. `prepare_mjx_adapter` performs the lazy import and requires an already
compiled public `mujoco.MjModel`; unavailability raises the shared explicit
backend error rather than selecting a fallback.

Before execution, negotiate exact `RoboticsOperationRequirement` values against
a `RoboticsBackendProfile`. Requirements may constrain operation, device,
dtype, minimum differentiability, solver, and contact feature. Profiles list
one capability per operation plus exclusions. `profile.negotiate` returns all
rejections; `profile.require` raises on the first unmet requirement. It never
weakens a request.

`MJX_JAX_PROFILE` declares conditional—not guaranteed—differentiability and
lists solver/contact exclusions. `MJX_WARP_PROFILE` is a capability declaration;
the public prepared adapter in this release is the MJX-JAX adapter. MJX support
is not universally differentiable.

`MJXAdapter` owns a complete opaque `mjx.Data` state and rejects states from a
different prepared adapter. Stable `RoboticsProjectionMap` values name complete
`qpos`, `qvel`, control, and observation ranges. Observation projections carry
one of three freshness labels:

- `state-current`: projected directly from the supplied complete state;
- `pre-step`: captured after installing control and before `mjx.step`;
- `post-step-refreshed`: captured only after an explicit post-step
  `mjx.forward` refresh.

Request `observations="both"` when both times are required. A nonfinite MJX
candidate causes complete-state rollback; the result reports `NONFINITE` and
never commits a partially finite foreign state.

## Limits to keep visible

The delivered native tree is 3-D and fixed-base with fixed, hinge, and
prismatic joints. Floating bases and ball joints are outside this reduced path.
Frame IK is local, not global. Articulated contact is fixed-route and does not
claim general contact-mode gradients. Sampling MPC reports finite sampled work,
not certified robustness or global optimality. Immutable fixed-work execution
is not hard real time. The MJX path is optional, capability-gated, and only
conditionally differentiable for the operations its profile declares.

See the [robotics API reference](api/applications/robotics.md), the
`examples/robotics_articulation.py` workflow, and the standalone
`tools/robotics_articulation_benchmarks.py` benchmark. The
benchmark reports native timings and evidence only; it makes no crossover or
speedup claim because it measures no reference implementation.

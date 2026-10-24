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
provide floating bases or ball joints. Fixed-route articulated contact is an
operator utility over routes supplied by the caller; it is not collision
discovery or an atomic robot/contact simulation step. Keeping those boundaries
explicit lets the native path expose stable layouts, operator duality,
fail-closed status, and pure-JAX runtime actions without making a broader
simulation claim.

## Contract map

| Concern | Public owner | Primary contracts |
| --- | --- | --- |
| Native robot model and URDF adaptation | `phydrax.applications.robotics` | `NativeRobotModel`, `PreparedNativeRobotModel`, `lower_native_robot_tree`, `RobotCollisionGeometryPlan`, `RobotContactMaterialTable`, `RobotFramePlan`, `parse_urdf_text`, `parse_urdf_file`, `RobotAdaptation`, `URDFFormatEvidence`, `URDFJointMotor`, `URDFCollisionPolicy` |
| Stateless joint laws, stops, and winding | `phydrax.discretization` | `ScalarJointMotorPlan`, `ScalarJointDampingPlan`, `ReducedJointStopPlan`, `PreparedLiftedJointCoordinates`, `LiftedJointCoordinateState` |
| Bounded local resources | `phydrax.interchange` | `ResourceLimits`, `ResourceManifest`, `BoundedResource`, `read_bounded_resource` |
| Reduced tree kinematics and dynamics | `phydrax.discretization` | `ReducedArticulationPlan`, `PreparedReducedArticulation`, `reduced_inverse_dynamics`, `reduced_forward_dynamics`, `reduced_semi_implicit_velocity_euler_step` |
| Inertial design coordinates | `phydrax.discretization` | `RigidBodyMassProperties`, `RigidInertialParameterization`, `RigidInertialRealization`, `RigidBodyReferenceFrameRebase` |
| Frame IK | `phydrax.applications.robotics` | `FramePositionTask`, `FrameOrientationTask`, `FramePoseTask`, `FrameInverseKinematicsPlan` |
| Fixed-route articulated impact | `phydrax.applications.contact` | `prepare_articulated_contact`, `ContactConeNumericRevision`, `solve_articulated_contact` |
| Result-preserving discrete evolution | `phydrax.dynamics` | `DiscreteTransitionResult`, `DiscreteTransitionEvidence`, `DiscreteEvolution` |
| Status-aware rollout and MPC | `phydrax.control` | `DiscreteControlDynamics`, `ControlTrajectory`, `plan_sampling_mpc` |
| Manifold transcription | `phydrax.control` | `manifold_radau_stages`, `manifold_radau_collocation_defects` |
| Task environments | `phydrax.applications.robotics` | `AbstractRobotTask`, `AbstractRobotEnvironmentWrapper`, `PreparedRobotEnvironment` |
| Reduced and continuum soft robotics | `phydrax.applications.solid_mechanics`, `phydrax.applications.contact`, `phydrax.applications.robotics` | spatial rods, PCS/GVS reconstruction, tendon/contact/floating/hybrid/FEM/MPM profiles |
| Optional provider execution | `phydrax.applications.robotics` | `RoboticsBackendProfile`, `RoboticsProjectionProvenance`, `MJXPreparedModelManifest`, `MJXAdapter` |

Plans and prepared objects are immutable PyTrees. Stable IDs bind topology,
layouts, reference frames, and provider projections. Runtime arrays remain
separate from host preparation and negotiation. A `successful` or `valid`
field is evidence to inspect, not a value to infer from finite-looking output.

## The native robot model

`NativeRobotModel` is the single executable model authority, whether it is
built directly with `lower_native_robot_tree` or produced by an adapter. It
composes the existing native plans — particle masses, COM-centered rigid
bodies, the rigid-joint graph, the reduced articulation and its
zero-configuration reference — with named `RobotFramePlan` frames and the
optional stateless `ScalarJointMotorPlan`, executed `ScalarJointDampingPlan`,
unilateral `ReducedJointStopPlan`, and `RobotCollisionGeometryPlan`
declarations. Absent components are `None`; nothing is inferred from source
bounds. `model.manifest` records counts, the executed capabilities, and the
component identities that form `model.model_id`.

```python
from phydrax.applications.robotics import parse_urdf_text

adaptation = parse_urdf_text(urdf_text, root_policy="fixed_world")
model = adaptation.model
prepared = model.prepare()
articulation = prepared.articulation
```

`prepared` holds every prepared owner (`particles`, `bodies`, `reference`,
`graph`, `articulation`, `coordinates`, `motors`, `damping`, `stops`,
`collision`). `model.reference` is the exact zero-configuration COM-centered
`RigidBodyKinematics` used to prepare both the joint graph and articulation.
`articulation.prepared_id`, `state_layout`, and `input_layout` bind the
resulting runtime contract. A tree with no moving joint has no nonempty reduced
state or input layout.

Bodies, joints, frames, actuators, and collision primitives are addressed by
deterministic signed 64-bit sorted-name maps: use
`model.body_names.id_for_name(name)` and `model.joint_names.id_for_name(name)`
instead of assuming source order. Prepared `joint_ids`,
`joint_configuration_slices`, and `joint_velocity_slices` identify the
topological runtime order. The articulation root must be a fixed body with its
own positive mass properties; a massless world anchor is refused rather than
given an invented mass.

### Stateless actuation, damping, and stops

A `ScalarJointMotorPlan` maps a separate command space (rows in
`model.actuator_names` order) to generalized effort in a fixed order: clip the
command to its bounds, multiply by the signed gear, then clip to
`±effort_limit`. An effort limit of exactly zero is an explicitly disabled
motor. Evaluations report command and effort margins, saturation flags, and the
exact source power `tau · v`. `ScalarJointDampingPlan` executes `-c v` and
reports the nonnegative dissipation rate `c v²`.

`ReducedJointStopPlan` bounds lifted coordinates (radians with full winding,
or meters). Its prepared rows are normal-only constraint rows, `(joint, side)`
typed, whose Jacobian `G` and exact transpose `G*` feed the same generalized
contact response as geometric contacts; stops never clip `q` or `v`. Gaps are
affine in lifted coordinates, so `path_minimum_gaps` is exact on the linear
lifted path. Activation, saturation, and stop kinks are discrete events:
derivatives are admitted only away from them.

### Winding and lifted coordinates

The reduced state keeps principal hinge angles for forward kinematics.
`PreparedLiftedJointCoordinates` carries the integer winding history that lifts
it. `advance(q0, state, v, h)` returns the articulation integrator's principal
endpoint and adds the turns removed by its wrap of the actual increment
`q0 + h v`, so a full turn with identical endpoints still advances the winding.
The counter range is explicit (`maximum_turns`); overflow is refused and the
source history is retained. `lift_resolution` reports the coordinate spacing at
the largest admitted lifted magnitude.

### Collision declarations

`RobotCollisionGeometryPlan` declares body-attached spheres and capsules (two
body-frame endpoints whose order defines the capsule axis, positive radii) and
static planes `n · x = offset` with free space on `n · x > offset`. Every
primitive/primitive and primitive/plane material pair needs an explicit
`RobotContactMaterialTable` entry with `0 <= dynamic <= static` Coulomb
friction; no mixing rule is implied. The declared law is the native
position-level inelastic hard-contact law, not an equivalent of any provider's
regularized contact.

## URDF security, root policy, and semantic loss

Both URDF entry points require an explicit `root_policy`. Select
`root_policy="fixed_world"` to attach the unique URDF root to the world.
`root_policy="reject_unpinned"` rejects the source because URDF does not encode
a world attachment for that root; there is no implicit default or inferred
floating-base interpretation.

`parse_urdf_text` bounds the exact UTF-8 bytes supplied in memory and never
resolves an external resource. `parse_urdf_file` additionally requires
`allowed_root`. It delegates to the generic `phydrax.interchange` bounded
resource contract: the trusted root is opened as a no-follow directory, then
each descendant component is opened descriptor-relative with no-follow
semantics. Traversal and network locations are rejected, the final object must
be a regular file, and directory/file identity is checked across the bounded
read. Symlinks are not followed.
`max_bytes`, `max_depth`, `max_nodes`, `max_attributes`, and `max_losses` are
finite limits on the resource and its decode.

Both paths are host-only. DTD/entity/notation/CDATA declarations, non-XML
processing instructions, xacro/include expansion, and plugin execution are
disabled. No visual, collision, transmission, mesh, or plugin asset is opened.
`adaptation.evidence.source_bytes` retains the exact accepted bytes.
`adaptation.evidence.resource_manifest` records their complete bounded-source
identity: memory/file kind, source path, trusted-root and file identities where
applicable, descriptor-relative components, byte size and SHA-256, configured
limits, observed depth/node/attribute/loss counts, and a manifest identity.

The accepted source is an unnamespaced URDF 1.0 connected tree with one root,
positive link masses, physically admissible inertias, and fixed, revolute,
continuous, or prismatic joints. Revolute and continuous joints lower to native
hinges. Each link frame is retained as a named model frame offset from the COM
body frame; rotated inertia tensors are retained in `URDFLinkEvidence`.

Lowering installs only explicitly supported runtime semantics:

- URDF `<dynamics damping>` is executed as native viscous joint damping.
- Position limits are evidence by default. `joint_stops="native-unilateral-stops"`
  installs them as native unilateral stops on lifted coordinates.
- Effort and velocity limits are not actuators or stops. Motors exist only for
  explicit `URDFJointMotor` mappings; the joint's URDF effort limit is the
  motor's effort clamp, and a zero effort limit is a disabled motor. URDF
  transmissions are never interpreted; once motors are requested, a
  transmission is an unwaivable loss.
- Collision geometry is lowered only under an explicit `URDFCollisionPolicy`
  (contact materials, per-link material, static planes, excluded link pairs).
  Only `<sphere>` is lowered; there is no URDF capsule tag and cylinders,
  boxes, and meshes are not approximated. Once collision is requested, every
  other shape is an unwaivable loss. Native capsules are available through
  direct construction.

Every unsupported or dropped source field becomes an `AdapterLoss` with exact
`affected_capability_ids`. Visual geometry is a declared
non-interpretation-changing loss. Unlowered collision geometry, transmissions,
joint friction, and other dynamics-changing data are interpretation-changing
losses.
An interpretation-changing loss that affects a required capability is
non-waivable: supplying its loss ID or path does not make negotiation valid.
Only an otherwise eligible loss can be accepted by an exact `AdapterWaiver` or
`waived_loss_paths` entry. Unknown or duplicate entries, stale or unused
waivers, missing required semantics, malformed sources, and every unwaived or
unwaivable interpretation change fail closed. Inspect all of:

```python
adaptation.report.status
adaptation.report.losses
adaptation.report.negotiation.unwaived_losses
adaptation.evidence.loss_paths
adaptation.evidence.resource_manifest
```

`require_lossless(adaptation.report)` is stricter: it rejects every declared
loss, including a waived or non-interpretation-changing one. Multi-stage
conversion reports may be joined with `compose_adapter_reports`, which checks
source/target identity, format-profile continuity, stage ordering, and the
cumulative negotiation instead of concatenating prose.

## State, frame, and unit conventions

The native adapter declares SI units. URDF lengths are meters, masses are
kilograms, angles are radians, and inertia is kg·m² about the link COM. A
successful adaptation has explicitly selected the `fixed_world` root policy.

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

Both Jacobian operators take `articulation.generalized_velocity_space` as their
source; its dual, `articulation.generalized_effort_space`, carries generalized
loads and impulses. `articulation.hinge_dof_indices` lists the configuration
indices of hinge coordinates. Body-ID lookup for `body_transform`,
`frame_transform`, and `frame_jacobian_operator` uses exact Python-integer
execution metadata, so it also works when the articulation is closed over by a
compiled function. The scientific body-ID payload (`articulation.body_ids`) and
`prepared_id` are the same; only the static execution representation of the
ID order changed, so PyTree structures of articulations prepared before and
after this change are not interchangeable.

## COM-centered inertial realization and reduced dynamics

`RigidBodyMassProperties` is the shared prepared contract for maximal and
reduced rigid mechanics. Body position and linear velocity are evaluated at the
center of mass, `first_moments` is identically zero, and `inertia_com` is the
rotational block of spatial inertia. The convenience `inertia_body` property
also means body-coordinate inertia *about the COM*; it is not inertia about an
arbitrary body-frame origin.

`RigidInertialParameterization` binds unconstrained host coordinates to one
prepared three-dimensional rigid-body set. Softplus mass coordinates and
lower-triangular covariance coordinates produce positive masses and physically
realizable COM inertias. `evaluate` reports finite masks, SPD and principal
moment checks, pseudo-inertia checks, condition numbers, reconstruction
residuals, and distance from the source body data.

Changing mass, COM, or inertia is an identity and reference-frame boundary.
`realize_rigid_body_plans` returns one `RigidInertialRealization` containing
fresh particle/body plans, the evaluation, and an identity-bound
`RigidBodyReferenceFrameRebase`; dependent prepared objects cannot be reused.
After preparing the realized plans, a nonzero COM offset requires both
`reference_frame_rebase.rebase_kinematics(...)` for poses and twists and
`reference_frame_rebase.rebase_local_points(...)` for every old-body-local
joint, frame, or contact attachment. Dependent joints and articulation must then
be prepared against that rebased reference. Ignoring either transfer changes
the represented mechanism. `evaluation.requires_repreparation` and the rebase
identity make this cutover explicit.

Native reduced dynamics uses the convention
`M(q) a + c(q, v) + g(q) = τ + Jᵀw`:

- `reduced_mass_matrix` materializes the symmetric generalized mass operator
  and reports symmetry and positive-definiteness evidence.
- `reduced_bias_terms` separates velocity bias, gravity, and their sum.
- `reduced_inverse_dynamics` uses a topological recursive Newton–Euler action
  and reports its decomposition and external-load power residuals.
- `reduced_forward_dynamics` uses articulated-body dynamics and certifies the
  candidate acceleration by inverse reconstruction. Its result carries the
  configuration-bound factor it used as `mass`. Without `mass_policy` it and
  the Euler step keep their original pivot acceptance (finite and above the
  smallest normal number); an explicit `ArticulatedMassPolicy` requires the
  relative pivot certificate described below.
- `reduced_energy` reports kinetic, potential, and total energy for an explicit
  world gravity vector.
- `reduced_semi_implicit_velocity_euler_step` performs bounded semi-implicit
  velocity Euler on the fixed-base reduced articulation: it advances velocity,
  integrates configuration with that candidate velocity, checks dynamics,
  inverse/forward reconstruction, step size, finiteness, and the energy-work
  defect, then atomically accepts the candidate or retains the source state.

`prepare_articulated_mass(articulation, q)` separates the configuration- and
inertia-only part of the articulated-body algorithm from velocity, bias, force,
and right-hand-side work. The returned `PreparedArticulatedMass` stores the
twist transforms, motion subspaces, articulated inertias, projected columns,
scalar pivots, and reciprocal pivots as dynamic leaves; it never forms a dense
mass matrix or inverse:

```python
mass = prepare_articulated_mass(articulation, q)
response = mass.apply_inverse(articulation, q, impulse)            # vector
columns = mass.apply_inverse(articulation, q, impulses, rhs_axis=1)  # (nv, K)
inverse_mass = mass.inverse_mass_operator  # effort dual -> velocity
mass_action = mass.mass_operator           # velocity -> effort dual
```

`inverse_mass_operator` maps `generalized_effort_space` into
`generalized_velocity_space`; `mass_operator` is the independent composite
`JᵀGJ` recursion. `apply_inverse` certifies every column by the relative
energy-norm residual `sqrt(rᵀM⁻¹r / fᵀM⁻¹f)` with `r = M x - f` and refuses
more columns than `ArticulatedMassPolicy.maximum_rhs`. A moving pivot is
accepted only when it is finite and exceeds the policy's relative tolerance
times `|S|ᵀ|Iᴬ||S|`; pivots are never jittered or sign-repaired, and
`mass.evidence` reports the pivots, their scales, and the minima.

`mass.binding` retains the exact configuration, masses, COM inertias, and
reference-frame leaves that produced the factor, while `mass.mass_id` is only
the static topology/schema/policy identity. `mass.is_current(articulation, q)`
and every result-bearing action compare the supplied state with that binding,
so a factor is rejected with `STALE_BINDING` after any configuration, frame, or
material change. A binding holding any NaN or infinite bound value is never
current, even against itself; the infinite `minimum_pivot` of a tree without
moving joints is evidence, not bound data. The operators themselves do not gate on `mass.successful`;
combine it with the consumer's status. Preparation and actions are
differentiable in `q`, COM placement, masses, and inertias on the admitted
positive-pivot domain.

On admitted Mosaic GPU targets, inertia factorization, factor substitution and
the reverse body-force reduction each use one coarse device kernel, with the
tree recurrence inside the kernel. Equal float32 or float64 numerical inputs,
1–1024 edges, 1–4096 bodies and 1–256 RHS columns are supported subject to
bounded indexing and layout admission. Shared factors remain shared across
mapped RHS columns. Other layouts, mixed dtypes, empty trees and unsupported
targets retain the canonical JAX route; derivatives use that reference
computation. Native execution does not weaken pivot or residual certificates.

Reduced pose, body-velocity, velocity-pullback and directional-motion traversals
also have coarse native kernels. Their canonical JAX references preserve the
original unrolled quaternion arithmetic, small-angle branches, normalization
floors and carry dtypes. Motion directions retain JAX's distinction between
structurally absent terms and numerical zeros; this matters for nonfinite
inputs. General derivatives and the transpose of the pullback use the canonical
reference, rather than substituting a primal action with different zero-product
behavior. Kernel architecture is selected from the compilation target, even
when the caller's default device is CPU.

Forward dynamics shares the zero-rate pose/motion evidence with mass
preparation and reuses the prepared factor's geometry and world inertias in
inverse reconstruction. Mass, bias, gravity and direct-effort columns remain
independent. Nonfinite velocity does not contaminate configuration-bound mass
preparation or velocity-independent reconstruction columns; nonfinite pose
geometry is still refused. Reconstruction, power, duality and failure evidence
remain part of the result, not an optional performance mode.

Compiled CUDA forward queries reserve supported computed result storage before
their private work and publish into those allocations afterward. Retaining an
earlier result therefore does not turn its successor's result buffers into
kinematic scratch. Caller buffers are never donated or mutated. Forwarded
inputs and constants remain shared, repeated computed leaves are published
once, and unused result projections are pruned. Reservation identity includes
the computation, argument/capture bindings, output slots and nested batching
interpretation, so distinct queries cannot accidentally share a reservation.

This allocation guarantee covers the query's directly returned, supported
array leaves, not caller computations before or after the query or Equinox's
completion marker. Leaves with unrepresentable dependencies, unsupported
targets or oversized output storage retain ordinary allocation and the same
numerical meaning. No global VMM setting or device-wide synchronization is
required. Compiler identities change with these native operations; rebuild
compiled execution artifacts. Prepared articulation fields and scientific
topology identities are unchanged.

The step policy, diagnostics, and result are respectively
`ReducedSemiImplicitVelocityEulerStepPolicy`,
`ReducedSemiImplicitVelocityEulerStepDiagnostics`, and
`ReducedSemiImplicitVelocityEulerStepResult`. This integrator is not a
collision/contact discovery or coupled robot-contact step.

Compile repeated queries as functions that receive the prepared articulation
as an argument:

```python
def query(articulation, q, v, tau, gravity):
    return reduced_forward_dynamics(articulation, q, v, tau, gravity)

compiled = jax.jit(query)  # eqx.filter_jit is equivalent here
result = compiled(articulation, q, v, tau, gravity)
```

Every dynamic leaf of a prepared articulation is a JAX array, and its execution
metadata is static. Plain `jax.jit` and `eqx.filter_jit` therefore both accept
it. Keep Python tolerances and policies inside the compiled function rather
than passing them as traced arguments. Do not close over the articulation:
closed-over arrays become compiled constants, so mass, inertia or frame updates
made with `eqx.tree_at` would never reach the executable. The plant and
contact-step entry points keep `eqx.filter_jit`, which raises runtime
validation failures (`eqx.error_if`) synchronously as `EquinoxRuntimeError`.

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

## Fixed-route articulated contact

Articulated impact reuses caller-supplied `ContactKinematicsEpoch` routes and
the native cone solver. `make_articulated_contact_participant` adapts an
existing collision surface, configuration space, forward kinematics, velocity
action, and force pullback. `build_contact_velocity_operator` produces the
contact velocity operator `G`; the supplied inverse-mass operator must be an
endomorphism of the same constrained generalized tangent space.
`build_delassus_operator` composes `W = G M⁻¹ G*` without routing through
unconstrained body response.

`prepare_articulated_contact` is certificate-bearing preparation, not matrix
assembly. The inverse mass is either a `PreparedArticulatedMass` (the matrix-free
route: its verified positive pivots supply the semidefinite hypothesis and its
numeric binding is bound into the cone revision) or an explicit true-dual
inverse-mass operator. `certify_inverse_mass_operator` verifies the operator's
represented bilinear matrix with `phydrax.linalg.verify_dense_properties`
(Hermitian symmetry and positive semidefiniteness required, explicit
tolerances) and keeps its `DensePropertyEvidence`. The matrix is the stored
matrix of a `DenseLinearOperator` or, for any other operator, a materialization
only under the caller's explicit bounded `inverse_mass_materialization` policy.
Without that policy a matrix-free operator is refused, and a policy too small
for it raises `LinearCapabilityError`; the certificate records which route was
used, and a zero-dimensional map is trivially accepted without dense evidence.
The original matrix stays the response; nothing is repaired. Dual
maps never carry endomorphism property flags. Preparation requires
complete participant routes and a mechanical material law for every valid
route, checks recorded route velocities against `G`, and builds the cone
program over the composed operator `W = G M⁻¹ G*` with an explicit
`ContactCertificatePolicy`. It never materializes the Delassus operator: two
deterministic probes of `W` record finite-precision adjoint consistency and
nonnegativity evidence only.

The cone result retains both paths. `ContactConeResult.candidate_impulse` and
its candidate velocities diagnose the solver proposal;
`ContactConeResult.impulse` is the accepted impulse and is exactly zero when
convergence or any dimensional physical certificate fails (material-law
completeness, numeric inputs, impulse sign, Signorini feasibility,
complementarity work, Coulomb cone, sticking/sliding law, padded components or
tangential work). `ContactConeNumericRevision` snapshots the response
operator's array leaves, the mass binding, the law data, layout, certificate
tolerances and solver controls used to obtain that result.

`apply_articulated_contact_impulse` re-certifies that numeric revision against
the prepared program before applying anything. It also checks the contact
equation, the agreement of `G v_post` with the physical post-impulse velocity,
law feasibility, contact/generalized power duality, and finiteness. Evidence
keeps the biased law normal velocity (`minimum_law_normal_velocity`) distinct
from the physical normal velocity (`minimum_physical_normal_velocity`). Only the
accepted impulse passes through `G*` and `M⁻¹`; otherwise the articulated result
exposes zero applied/generalized impulse and velocity update, retains the free
velocity, and records `fail_closed`. `build_mixed_contact_velocity_operator`
stacks contact rows and normal-only joint-stop rows into one `G`, so contacts
and stops share one coupled response solve.

This is an operator-level articulated impact utility for already declared,
fixed routes. It neither discovers collisions nor commits mechanics and contact
as one atomic robot state transition, and it does not claim general gradients
through contact-mode changes.

## Result-preserving discrete evolution and control rollout

Robotics plants should return `DiscreteTransitionResult`: candidate state,
accepted state, scalar `successful`, and scalar integer `status`.
`DiscreteSystem.evaluate_result` preserves all four values, whereas
`DiscreteSystem.evaluate` intentionally returns only the accepted state.
`DiscreteEvolution.advance` and `evolve` preserve per-step candidates, accepted
states, success, status, and deterministic first-failure summaries in
`DiscreteTransitionEvidence`. Dynamic transition parameters passed as `args`
flow unchanged through the input policy and plant transition.

`DiscreteControlDynamics.rollout` applies that same result contract over the
declared case axes with `jax.lax.scan` and forwards its `args` to every plant
transition. It feeds accepted states forward, never substitutes a failed
candidate, does not send nonfinite user controls into the plant, retains
candidate/accepted/status evidence for every attempted transition, preserves
the first backend failure status, and marks the trajectory invalid from a
failed transition onward.

`ControlProblem.rollout` supplies `problem.args` through this path, so
higher-level control consumers use the same dynamic plant parameters and
result-bearing trajectory rather than reconstructing a state-only rollout.

`ControlTrajectory.status` is the control-level result;
`ControlTrajectory.backend_status` retains the plant status, and
`ControlTrajectory.transition_evidence` retains the plant's per-step result.
All three are required when diagnosing a rollout. The same contract lets
semi-implicit reduced mechanics, immutable environments, candidate search, and
trajectory optimization share failure semantics without discarding the
candidate evidence.

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

`PreparedRobotEnvironment` composes one complete-state `AbstractDiscretePlant`
with its bound `PlantParameters`, one `AbstractRobotTask`, and an ordered tuple
of `AbstractRobotEnvironmentWrapper` values. The plant may be the native
`PreparedArticulatedPlant` or a legacy rank-one `DiscreteSystem` adapted by
`prepare_array_robot_environment`. The environment carries the complete
`PlantRuntimeState`: for the native plant that is configuration and winding,
velocity, contact manifold, stop reactions and branches, last command,
transition/sensor epochs, sensor samples, geometry/material revisions, clock,
step index, and key. Nothing is reconstructed from `q`/`v`. The task owns its
fixed-structure state, observation, reward-component names, termination, and
descriptor. Wrappers own administrative state, action repetition, horizons,
truncation, and optional auto-reset. They do not mutate the plant or hide task
termination.

Actions are complete plant commands. An optional `control_codec`
(`ControlVectorCodec`) instead admits command vectors, either shared or one per
case. The codec must be unbatched, describe the plant's complete command
contract, and carry the plant's semantic, numeric, and executable identities.
Its `codec_id` enters the environment provenance. Autonomous plants
(`control_schema is None`) take `action=None`. The task receives the same
validated command as the plant. Non-finite commands reach the task as zeros and
are rejected by the plant transaction before mechanics.

Environment construction derives semantic `provenance_id` from the plant
semantic, numeric, state-schema, and executable identities, the parameter
schema, the optional control codec, task and wrapper configuration, array
fingerprints, repetition and horizon policy, and PRNG representation. The plant
transition identity, step constraints, layouts, and initializer identity enter
through the plant identities. Callables are identified by content:
StrictModule callables by their fields and plain module-level functions by
their code, static defaults, and the scalar constants they read from module
globals. Lambdas, closures, methods, partials, functions reading module-level
arrays, and other opaque callables are refused unless their owner declares both identities;
`prepare_array_robot_environment` accepts `transition_semantic_id`/
`transition_numeric_id` and `initializer_semantic_id`/`initializer_numeric_id`
for this purpose. Semantic identities enter the plant semantic provenance and
numeric identities its numeric revision. Runtime states carry both that
provenance and `environment_id`; `step` rejects a state from a different
semantic environment even if its arrays have compatible shapes.

`reset` initializes the plant, task, wrappers, caller-derived PRNG stream, and
clock. The plant receives typed JAX PRNG keys, which its runtime state retains.
The environment's own episode key remains raw `uint32` key data.
`step` advances each substep through the plant's own transaction over
`[t, t + step_size]` (one addition in the clock dtype), using the environment's
bound parameters. It performs the statically declared repeat count. Task, wrapper,
parameter, and initializer arrays remain dynamic leaves under compilation.
Runtime loops perform no host synchronization. The task receives both source
and **accepted** mechanics state, and wrappers likewise receive the accepted
plant state; neither consumes a rejected mechanics candidate. Candidate plant
state remains separately visible in the environment result. A source case that
is already invalid proposes from the plant's reset fallback and never commits.

Plant, task, wrapper, reward, clock, and PRNG state commit together, per case,
only if every attempted mechanics substep succeeds and every output is finite.
Otherwise that case's `accepted_state` is its complete source state, including
discrete contact/stop history, sensor epochs, and keys. Its reward is zero, its
termination/truncation is suppressed, and rollback evidence retains attempted
steps, mechanics success/status, and source/candidate indices. With auto-reset,
`final_observation` remains the accepted terminal observation while
`observation` belongs to the reset state. `reset_successful`/`reset_status`
report the plant reset transaction of the reset cases. A failed plant reset
leaves the case at the plant's reset fallback.

Fixed work and immutable state improve compilation and auditability; they are
not a hard real-time scheduling guarantee.

## Soft robotics composition

Spatial rods now use the shared four-space geometry and true-dual operator
contracts. Native discrete rods, PCS/GVS bases and physical reconstruction,
materials, reduced dynamics and integrators, tendons and advanced actuator
evaluators, continuum tasks and observations, calibration/co-design/control,
floating rods, rigid–soft composition, fixed-mesh FEM, fixed-topology MPM, and
MJX each retain separate ownership and capability evidence.

The atomic reduced-rod contact plant adds collision discovery for its exact
fixed-base spatial circular-capsule plane/self-contact profile, with
fixed-capacity search, persistent route-keyed history, conservative-advancement
CCD, frictionless or isotropic Coulomb response, and full-interval-or-rollback
semantics. This does not broaden the rigid-articulation contact utility above:
rigid articulation contact still requires caller-supplied fixed routes.

See [Soft robotics](guides_soft_robotics.md) for the full geometry, plant,
codec/replay, rod, actuator, contact, observation, inference, floating, hybrid,
FEM, MPM, and qualified/excluded capability contracts.

## Optional MJX plant lifecycle and provenance

Importing the robotics package does not require MuJoCo or MJX.
`mjx_availability()` probes both `mujoco` and `mujoco-mjx`; their base releases
must match exactly in the qualified 3.12.x minor. `prepare_mjx_adapter` performs
the lazy import and requires an already compiled public `mujoco.MjModel`.
Missing, mismatched, unsupported, or unqualified providers fail explicitly.

Preparation builds `MJXPreparedModelManifest`, the closed feature set actually
present in the model. Unsupported enums, feature bits, collision combinations,
flexible bodies, plugin state, and the other declared exclusions fail before
transfer. `MJXAdapter` is an `AbstractDiscretePlant`; its public `state_schema`
describes the canonical complete `mjx.Data` PyTree, and `MJXState` binds that
opaque payload to state and sensor epochs.

The shared plant lifecycle is:

```python
reset = adapter.reset(key, adapter.parameters)
command = adapter.control(reset.accepted_state)
stepped = adapter.step(context, reset.accepted_state, command, adapter.parameters)
refreshed = adapter.refresh(stepped.accepted_state, request)
observation = adapter.observe(refreshed.accepted_state, request)
```

Inspect the shared `PlantStepResult.successful/status` and
`MJXRefreshResult.successful/status`. A successful step advances the state epoch
while leaving derived sensors stale. A sensor-bearing observation is valid only
after successful refresh. Step and refresh retain failed complete cases.

`MJX_JAX_PROFILE` supports only `step` and `sensors` on its exact device,
dtype, solver, and contact-feature tuples, with conditional differentiability.
The other robotics operations have no adapter callable. `MJX_WARP_PROFILE`
marks every operation unsupported because no MJX-Warp adapter callable is
implemented.

## Limits to keep visible

The native reduced rigid tree remains 3-D fixed-base fixed/hinge/prismatic;
floating bases and ball joints are outside that articulation profile. Frame and
continuum IK are local, not global. Sampling MPC and SQP do not certify global
optimality or robust stability. Fixed work is not hard real time. Soft robotics
claims apply only to the exact capability tuples and exclusions in the soft
robotics guide and to runtime evidence for the caller's prepared device, dtype,
capacities, tolerances, and provider releases.

See the [robotics API reference](api/applications/robotics.md), the
[soft robotics API reference](api/applications/soft_robotics.md),
`examples/robotics_articulation.py`, `examples/soft_robot_tendon.py`,
`examples/soft_robot_contact.py`, `tools/robotics_articulation_benchmarks.py`,
and `tools/robotics_soft_rod_benchmarks.py`. Each benchmark reports native
timings and evidence only; neither makes a crossover or speedup claim without a
measured reference implementation.

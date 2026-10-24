# Native robotics qualification

This guide states what the native fixed-base articulated robotics path claims,
how those claims are exercised, and which comparisons with MuJoCo or MJX are
scientifically valid. A qualification statement holds only for its code,
provider, device, dtype and tolerance identity. A missing optional provider is
not a product failure, and an unsupported comparison is not a pass.

The native runtime has no MuJoCo or MJX dependency. MuJoCo is an optional
host-only source for `adapt_mujoco_model` and a reference for the comparisons
below; MJX remains a separate provider-backed plant described in the
[native robotics guide](guides_robotics.md).

## Native capability under qualification

| Concern | Native owner | Admitted profile |
| --- | --- | --- |
| Model | `NativeRobotModel` | 3-D fixed-base tree; fixed, hinge and prismatic joints; positive-mass fixed root |
| Inertia | `prepare_articulated_mass` | Configuration-bound ABA factors; true-dual inverse-mass and mass actions; vector and explicit multiple RHS |
| Actuation | `ScalarJointMotorPlan`, `ScalarJointDampingPlan` | Stateless motors (command clip, signed gear, effort clip) and executed viscous damping |
| Stops | `ReducedJointStopPlan` | Normal-only hinge/prismatic stops on lifted coordinates, coupled with contact |
| Geometry | `RobotCollisionGeometryPlan`, `ArticulatedCollisionGeometry` | Positive-radius body spheres and capsules; static planes |
| Contact law | `PreparedArticulatedContactStep` | Position-level inelastic hard contact; zero restitution and compliance; isotropic static/dynamic Coulomb |
| Plant | `PreparedArticulatedPlant` | Complete-state payload; full requested interval or rollback; explicit case axes |
| Control | `PlantControlDynamics` | Complete-state scan carry; endpoint or full retained history |
| Rank evidence | `RigidConstraintRankAudit("certified-forest")` | Opt-in forest certificate for maximal-coordinate graphs; dense SVD remains the default |

Floating bases, ball joints, loops, tendons, stateful actuators, rolling,
spinning or adhesive laws, restitution in the native step, mesh/box/cylinder
collision in the articulated plant, topology mutation and multi-GPU execution
are outside this profile. Refusals happen at preparation or adaptation; there is
no fallback to different physics.

## Comparison tiers

`tools/qualify_native_mujoco.py` compiles each fixture once and adapts that
exact `MjModel`. Every tier compares native and provider quantities at
identical mapped states and inputs, or refuses.

| Tier | Valid claim | Never claimed |
| --- | --- | --- |
| `smooth-model-mapping` | Body-origin, COM and site poses; site Jacobians; virtual work; mass matrix and mass actions; inverse dynamics, bias and forward dynamics where the provider has no active constraint | Contact or limit behavior |
| `smooth-actuation` | Generalized actuator effort, clamps and actuator source power; damping effort when damping is executed | PD/servo equivalence |
| `matched-stepping` | One-step configuration, velocity and clock agreement at sampled states where the provider has no active constraint, only when the source integrator is Euler, the source compiles no contact pair (excluded contact is also refused), limits are not transformed into native stops, damping is not treated implicitly (`eulerdamp` disabled or zero damping), and actuation is enabled | Any step under different laws, integrators or damping treatment; such requests are refused |
| `different-native-contact-physics` | Same compiled geometry and signed distances at identical states; native plant invariants under the native law; MuJoCo checked only against its own rest invariant | Equal trajectories, forces, impulses or complementarity residuals |
| `fixed-branch-ad` | Native site-position JVP/VJP against provider site Jacobians and the adjoint identity in the shared smooth domain | Derivatives through contact, stop, winding or subdivision events |
| `mjx-smooth-model-mapping` | COM and site positions, bias, actuator effort and forward dynamics against `mjx.forward` (JAX implementation, CPU) when a release-matched `mujoco`/`mujoco-mjx` pair is available | MJX contact or stepping parity |

`matched_stepping_refusal` returns the reason a one-step comparison is invalid.
The profile's `invalid_requests` deliberately ask for matched stepping where it
is invalid (soft versus hard contact, regularized limits versus hard stops, and
RK4 versus native semi-implicit velocity Euler); each must be refused. An
unexpected admission fails the run.

## MuJoCo smooth compatibility versus native contact transformation

The adapter produces two different kinds of result and the qualification keeps
them separate.

**Smooth compatibility** is a mapping of the same model content. Compiled body
transforms, COM offsets, inertial quaternions and principal inertias become
native COM-centered bodies without a second parallel-axis shift. Hinge and
slide joints keep their axes and anchors; the native lifted coordinate is
`qpos - qpos0` and the velocity is `qvel`. Stateless joint motors (fixed gain,
zero bias, scalar gear) keep the order control clamp, gain, force clamp,
transmission, with native gear `gain · gear` and effort limit `|gear| · F` for a
symmetric `forcerange` `[-F, F]`. Viscous damping is executed. Body-origin and
site frames and the admitted sensors keep their names, quantities, units and
stages. These are the quantities of the smooth tiers, and only they can agree
within the profile's numerical tolerances.

**Native contact transformation** is a declared change of physical law, not a
mapping. MuJoCo's regularized soft contact (`solref`, `solimp`, `condim`,
margins, gaps, friction mixing and priority) and its regularized limits are not
translated into native parameters. With `MujocoNativeContactTransformation`,
colliding spheres, capsules and world planes become native primitives under the
inelastic Coulomb hard-contact law with an explicit `RobotContactMaterialTable`;
with `MujocoJointStopTransformation`, limited joints become native hard stops on
`[range_low - qpos0, range_high - qpos0]`; `MujocoContactExclusion` removes
contact. Each records an interpretation-changing transformed or dropped loss in
the `AdapterReport`, waived only by the caller's stated rationale, and
`MujocoModelEvidence.contact_mode`/`joint_limit_mode` name the executed law. Pair
filtering (explicit pairs, weld and parent filtering, excludes,
`contype`/`conaffinity`) is reproduced as native body-pair exclusions where it
is all-or-none between two native bodies; mixed geom-level admission between the
same two bodies is refused. After a transformation, geometry is the same but the
dynamics are not, so only the `different-native-contact-physics` tier applies.

## Fixtures and tolerances

`tests/fixtures/robotics/native_mujoco_profiles.json` declares the provider
release the fixtures were authored for, the deterministic state seed and count,
per-dtype tolerances by quantity (positions, rotations, Jacobians, relative
mass/effort/acceleration/actuation/power, step state, signed distance, adjoint
identity), the MJCF fixtures under `tests/fixtures/robotics/native_mujoco/` with
their source SHA-256 and requested tiers, and the invalid requests. A source
hash mismatch fails the run; the compiled-model identity is recorded per run
from `mj_saveModel` bytes. There is no compatibility-generation field.

| Fixture | Adaptation | Tiers |
| --- | --- | --- |
| `arm_smooth` | No contact or limit policy | smooth mapping, actuation, matched stepping, fixed-branch AD |
| `arm_contact` | Native inelastic Coulomb contact with declared materials | smooth mapping, different native contact physics |
| `arm_limited` | Native hard stops in place of regularized limits | smooth mapping |

The `arm_contact` fixture also declares its native rollout: step policy,
certificate, path and solver budgets, search backend (`lbvh`) with an
`exhaustive` reference backend, a refined time step, and physical limits. Its
contact tier requires an admissible native reset, full-interval success of each
step, non-penetration, ledger identity, nonpositive physical contact work and
energy surplus, nonnegative correction loss, no energy creation beyond source
work, established contact, native rest on contact, rest agreement under time
step refinement, and search-backend parity; MuJoCo's own simulation must come to
rest under its own law. The rest-gap difference between the laws is reported,
not judged. A contact fixture without a declared native rollout reports the
tier as `not-evaluated` with its geometry comparison as a prerequisite only;
matching geometry is never a contact-physics pass.

## Running and reading a qualification

```bash
python -m tools.qualify_native_mujoco --output native-mujoco-qualification.json
```

`--profile` selects another profile and `--repeats` the timing repeats. The run
uses float64. The report records the runtime identity, JAX backend and devices,
provider availability and versions, profile hash, and per fixture the source
hash, MuJoCo version, compiled-model hash, native model, prepared, evidence and
report identities, every adapter loss, the executed laws, integrator, timestep,
gravity and option flags, and the tolerance set.

Runtime provenance uses `QualificationRuntimeIdentity` and the shared benchmark
source/build, driver, installed-package, device and numerical-configuration
observations. Failed identity capture aborts qualification rather than emitting
a qualified record with missing identity. The retained execution `uv.lock`
participates in source/build identity; this is not an attestation that every
optional-extra branch of a universal dependency lock is resolvable.

Tier status is `passed`, `failed`, `refused`, `not-evaluated`,
`provider-unavailable` or `provider-unsupported`; every comparison keeps its
measured error, error kind and tolerance. The report status is
`provider-unavailable` (exit code 2) when host MuJoCo is missing, `failed`
(exit code 1) when any tier, geometry prerequisite, source hash or invalid
request fails, and otherwise `qualified-evaluated-tiers`, with every unevaluated
tier listed in `not_passed_unevaluated`. An adaptation refusal is reported as
`adaptation-refused`.

Timings compare JIT native forward dynamics with host `mj_forward` on the same
state. They measure different work on different devices and are not a speed
claim.

## Native-only evidence

Native correctness does not depend on a provider. The owning suites are:

- articulated inertia: `tests/unit/discretization/test_articulated_mass.py`,
  `test_reduced_dynamics.py`;
- device routes, analytic witnesses and trajectory bounds:
  `tests/unit/discretization/contact/test_device_contact_routes.py`,
  `test_analytic_contact_witnesses.py`, `test_contact_trajectory_bounds.py`;
- operator cone and history: `tests/unit/applications/contact/test_operator_contact_cone.py`,
  `test_contact_manifold.py`;
- coupled step and plant: `tests/unit/applications/contact/test_articulated_step.py`,
  `tests/unit/applications/test_articulated_plant.py`,
  `tests/integration/test_native_articulated_contact.py`;
- model and adapters: `tests/unit/applications/test_native_robot_model.py`,
  `test_robot_urdf.py`, `tests/integration/test_native_mujoco_model.py`,
  `tests/integration/test_native_mujoco_qualification.py`;
- control: `tests/unit/control/test_plant_control_dynamics.py`;
- rank evidence: `tests/unit/linalg/test_forest_bounds.py`,
  `tests/unit/discretization/test_rigid_constraint_rank.py`.

References are independent: analytic one- and two-link systems, small dense
mass and constraint matrices, analytic sphere/capsule/plane geometry, dense
oracles for forest bounds, and host MuJoCo only where the model and law match.
The optimized path never provides its own reference.

Performance is measured by the owning tools with phases kept separate:
`tools/robotics_articulation_benchmarks.py` (preparation, mass binding,
repeated and multiple-RHS actions, contact step and rollout),
`tools/contact_search_benchmarks.py` (canonical epochs through kinematics and
refusal rates), `tools/contact_closure_benchmarks.py` (operator cone and
derivative actions against the explicit dense authority) and
`tools/rigid_constraint_benchmarks.py` (rank audit, certification, inconclusive
and fallback rates). Unsuccessful, inconclusive and resource-refused cases are
reported with timings and are never counted as speedups. Same-model smooth
performance and different-contact-law throughput are separate reports.

Use `--native-execution-only` in the articulation campaign when varying only
case capacity or horizon and separately qualified scalar component/derivative
measurements need not be repeated. It still measures reset, the public step
and complete-state rollout, and labels omitted diagnostic phases
`not-requested`; omission is never derivative admission. Cold campaigns require
an inactive persistent compilation cache, selected before process launch.

## Derivative admission

Native derivatives are qualified only on successful fixed branches: a
single-segment step with no contact or stop activation, no winding increment,
positive feature margins and branch margins above the policy factor, evaluated
with exactly the step, actuation and gravity bound to the result. With contact
or stop blocks, regularity is proved rather than inferred from a successful
linear solve: the explicit bounded `SVDSolvePolicy` in
`ArticulatedContactStepPolicy.derivative_regularity` (or `regularity=` for the
standalone cone sensitivity) must certify the coupled fixed-branch `(v1, p)`
Jacobian (respectively the cone Jacobian) as nonsingular with fixed rank before
any solve; without it every contact derivative is refused. Cone derivatives also require a recertified branch away
from the apex and stick/slip boundaries, and friction derivatives are refused at
a frictionless Coulomb boundary. Elsewhere `step_jvp`, `step_vjp` and the cone
JVP/VJP return zero with explicit refusal evidence. Search, route birth or
death, winding, chart cuts, activation and subdivision are discrete; JIT
compatibility does not imply differentiability through them.

## Known limitations

These boundaries hold for the current implementation and must be kept in mind
when reading any qualification or benchmark record:

- Complete-plant qualification uses float64. Mass, search and standalone cone
  paths also run in float32, but that does not establish end-to-end float32
  plant qualification or performance on another accelerator. Large float32
  trees can refuse smooth inverse reconstruction at an explicit tolerance budget
  while their mass actions pass, and float32 cone derivatives can be refused by
  the explicit SVD rank policy.
- Highly coupled cone problems can exhaust the declared iteration budget; the
  result is a failure, not a smaller certified response. A separated positive
  response spectrum alone is not a global Coulomb uniqueness proof.
- Forest rank bounds can be inconclusive for long chains; only an explicitly
  admitted dense fallback then decides rank. Whole-step convergence refusals are
  reported under both audit methods rather than presented as speedups.
- Constant-rotation paths use the native-FK endpoint hull and ancestry speed
  bounds use body-wise accumulation. The enclosure guarantee is
  `ENCLOSURE_CONSERVATIVE` with modeled rounding, not a directed-rounding proof
  for arbitrary tree depth.
- The compiled contact step is not a fused, device-resident iteration schedule:
  compiled conditionals and error checks emit many kernels, graph launches and
  small device-to-host copies per step. The fused articulated sweeps cover
  inertia factorization, factor substitution and reverse body-force reduction;
  substantial work remains outside them.
- Coarse articulated sweeps and compact kinematics reduce the kernel schedule;
  that is not a throughput claim and not a uniform latency improvement. Some
  complete-plant rollout, mass-binding and derivative phases are slower than in
  the preceding native implementation. No speed superiority over a provider is
  claimed.
- Compiled forward queries reserve their result storage before private work.
  This is a per-query private-work guarantee, not whole-program allocation
  isolation: a later query's output allocation can still hold earlier
  sibling or caller work before its reservation.
- Prepared contact and articulation representations changed; reprepare
  checkpoints, tangent/cotangent templates and compiled artifacts rather than
  reusing an old representation.
- Instrumented trace durations are not throughput measurements, and provider
  peak-memory counters include compilation and are not phase-local peaks.
  Compiler buffers, logical retained arrays and provider allocator observations
  remain distinct evidence.

## Recording requirements

Record the exact command, code revision, provider versions, device, dtype,
capacities, tolerances, requested law, geometry and derivative profile, and
every pass, failure, refusal, skip and not-evaluated reason. Reuse evidence
only while those inputs are unchanged. Until a recorded run exists for a given
identity, this guide is a procedure, not evidence of MuJoCo equivalence or
improved throughput.

See the [robotics API reference](api/applications/robotics.md), the
[contact API reference](api/applications/contact.md), the
[contact discretization API reference](api/discretization/contact.md) and
`examples/native_articulated_contact.py`.

# Robotics muscle routes and MJX projections

Phydrax separates native geometry from provider geometry. `FixedBodyRoutePlan` owns piecewise-linear routes through body-fixed points; the analytic wrap families below own only their declared primitive branches. `MJXPreparedMuscleProjection` instead projects fields computed by a same-release compiled MuJoCo/MJX model, including provider wrapping. These are alternative geometry authorities for a given route; their forces must not be combined.

## Native fixed body-attached routes

`FixedBodyRoutePlan` stores route names, body IDs, route masks, and point offsets as static compressed-row topology. Every route has at least two points and all capacities are fixed at preparation. `PreparedFixedBodyRoute.local_positions_m` remains a dynamic JAX leaf, so attachment coordinates can be calibrated without changing topology or IDs.

A prepared route uses the existing `PreparedReducedArticulation.frame_transform` API to map every body-local point into the world frame. It returns:

- world points and segment vectors in m;
- segment and route lengths in m;
- route length rates in m/s from an exact JVP;
- a `FunctionLinearOperator` mapping generalized velocity to length rate;
- the exact transpose tensile-force pullback.

Input tension is positive tensile in N. If `J_L` maps generalized velocity to route extension rate, the generalized load is

\[
Q = -J_L^T T,
\]

so articulation power is `qdot · Q = -T · Ldot`: a shortening tensile route does positive work on the articulation. `FixedBodyRoutePullbackEvidence` records both sides and their residual. Disabled, non-finite, zero-length, or compressive route rows contribute zero load and fail the corresponding evidence rather than changing shape. The smooth fixed-point route supports local JVP/VJP; a zero-length segment is outside its differentiable domain.

This native route has no obstacle or contact branch. Run `examples/robotics_fixed_body_muscle_route.py`; qualify derivative and virtual-power identities with `tools/qualify_fixed_body_route.py`.

## Prepared MJX built-in muscle projections

`MJXAdapter.prepare_muscle_projection()` discovers actuators compiled with MuJoCo’s built-in muscle gain, bias, and activation dynamics. The all-muscle selection follows compiled actuator order; `MJXMuscleProjectionPlan(names)` preserves an explicit unique name order. Preparation validates one distinct activation state and a finite increasing actuator length range for every discovered built-in muscle.

`scatter_control(complete_control, independent_excitation)` requires the complete model control vector. Passing `MJXAdapter.control(state)` preserves the source epoch binding required by the atomic plant step. The method replaces only selected muscle entries, validates dimensionless excitation in `[0, 1]`, and preserves every non-muscle actuator control. This avoids an implicit zero-control policy.

`snapshot()` gathers fixed-shape projections:

| Field | Canonical quantity | Unit | Provider meaning |
|---|---|---|---|
| `activation` | `muscle_activation` | 1 | MuJoCo built-in muscle activation state |
| `length_m` | `musculotendon_length` | m | compiled actuator transmission length |
| `velocity_m_per_s` | `musculotendon_velocity` | m/s | positive transmission extension rate |
| `raw_force_N` | `raw_provider_force` | N | raw signed `mjData.actuator_force`; built-in MuJoCo muscle pulling is negative |

`raw_force_N` has atomic `force_owner="provider-native"`. It is not normalized, not converted to positive tensile force, and must not be multiplied by De Groote–Fregly, D1, Shorten, or another native force law. The negative-pulling statement is limited to MuJoCo’s documented built-in muscle convention; it is not a universal provider-force sign rule.

Length, velocity, and force are forward-derived. A step increments the payload state epoch but preserves the prior forward epoch, so snapshots are explicitly stale until `MJXAdapter.refresh()` runs `mjx.forward`. Failed steps retain the complete accepted `PlantRuntimeState` source through the standard atomic plant transaction; a subsequent successful refresh makes only its accepted complete payload current. Activation itself is state-current, but a snapshot succeeds only when all four fields are finite and the forward-derived fields are current.

The provider contract is documented in MuJoCo 3.12’s [Muscles](https://mujoco.readthedocs.io/en/3.12.0/modeling.html#muscles) section: actuator length is the transmission length, actuator velocity is its rate, and built-in muscle actuator force is negative when pulling. `tools/qualify_mjx_muscle_projection.py` compares all four projected fields against host MuJoCo from the same qualified release.

## Bounded analytic sphere and planar-cylinder wrapping

`SphereRouteWrapPlan` independently implements the fixed-branch tangent and arc
geometry documented by OpenSim Core at commit
`86b30588374650fbaf012a345a836a64f6855522`. `PlanarCylinderRouteWrapPlan`
implements the same lateral-surface construction for endpoints in one common axial
plane. Both return fixed-capacity surface samples, tangent points, surface and total
length, branch/event margin, surface/tangency residuals, and explicit status.

`sense="short"` and `sense="long"` are different prepared branch identities. Local
JVP/VJP is supported only away from endpoint contact, chord tangency, tangent-pair
ties, and short/long branch changes. Endpoints inside the primitive and degenerate
required routes fail. A chord that does not require wrapping is a successful direct
route with `applied=False`.

The cylinder fidelity is deliberately planar. Unequal endpoint axial coordinates
require the source-specific helical tangent adjustment and are rejected with
`NONPLANAR_CYLINDER_ROUTE`; no approximate helix is substituted. A common axial
plane outside the declared finite lateral length is a successful direct route.
End-cap contact is not modeled. This planar identity remains unchanged; the
separate lateral-cylinder family below admits unequal axial coordinates.

Run `examples/robotics_analytic_wrap.py`; qualify geometric residuals and the
fixed-branch directional derivative with `tools/qualify_analytic_route_wrap.py`.

## Source-qualified three-dimensional lateral cylinder

`OpenSimCylinderRouteWrapPlan` is a separate single-obstacle geometry authority.
It pins [OpenSim `WrapCylinder.cpp`](https://github.com/opensim-org/opensim-core/blob/86b30588374650fbaf012a345a836a64f6855522/OpenSim/Simulation/Wrap/WrapCylinder.cpp)
at the same revision, with raw SHA-256
`ce01766de755cd78ae2b21a271809d5c988082e87a50e9ebf5302c9282662580`
and Apache-2.0 source license. Its **independent numerical realization** solves
the source lateral-helix and endpoint-tangency equations exactly on the unrolled
cylinder. It does not claim bitwise or tolerance parity with OpenSim's iterative,
display-segment-dependent tangent adjustment or its quadrant-selection policy.

The origin, oriented axis, radius and lateral length are explicit SI geometry;
preparation binds their numeric content identity. Geometry leaves remain dynamic
for local calibration, while accepted state records the geometry binding and
rejects stale geometry. There is one obstacle and exactly three candidate slots:
direct, positive-axis winding, negative-axis winding. Each lateral candidate has
zero extra turns. `side="shortest"` selects the shorter complete lateral path;
`side="positive"` and `"negative"` prescribe an oriented branch. These are
different identities. A prescribed opposite side changes both tangent points,
not merely the direction of a surface arc.

Let the two projected free tangent lengths be `a` and `b`, the circumferential
arc length be `c = radius × abs(angle)`, and endpoint axial separation be `dz`.
The unrolled path has length `h = a + c + b`. Minimizing the sum of the two free
segments and the source helical surface length makes all three axial slopes
equal to `dz / h`. Thus contact heights are `z_start + a × dz / h` and
`z_end − b × dz / h`, total length is `sqrt(h² + dz²)`, and wall length is
`c × sqrt(1 + (dz/h)²)`. The objective in the two contact heights is convex
for nondegenerate tangent spans. This supplies a stationary/global minimum
**within each lateral branch** without a nonlinear iteration. Comparing the
two branch costs gives the shortest zero-extra-winding lateral route.
Surface samples are visualization only and never determine mechanical length.

The lifecycle is explicit:

```python
from phydrax.applications.robotics import OpenSimCylinderRouteWrapPlan

prepared = OpenSimCylinderRouteWrapPlan().prepare(origin, axis, radius, length)
source = prepared.initial_state()
candidate = prepared.propose(source, endpoints)  # endpoints: (2, 3), metres
accepted = prepared.commit(candidate, source)
fixed = prepared.evaluate_fixed_branch(accepted, endpoints)
loads, power = prepared.tensile_force_pullback(
    accepted, endpoints, endpoint_velocities, native_tension,
    force_owner="native-tension",
)
```

Discrete branch selection is accepted state, not a hidden cache. Candidates
report every candidate cost/feasibility, the shortest lateral branch/cost gap,
selected excess length, contact/rim margins, three-dimensional tangent-direction
continuity and radius residuals. A branch transition requires explicit commit.
There is no hysteresis. Ties, contact onset, endpoint contact/inside-radius,
unsupported rims and nonfinite geometry fail closed. Failed, stale or foreign
candidates preserve every accepted state leaf and counter. A fixed-branch
Jacobian has endpoint velocity shape `(2, 3)` and length-rate shape `(1,)`;
JVP/VJP is admitted only by `fixed_branch_gradient_supported`. Candidate
selection and topology transitions are not a differentiable contract.

The route does not generate tension or a separate contact force. Its endpoint
loads are `−J_Lᵀ T`, and its power evidence checks
`sum(endpoint_velocity × endpoint_load) = −T × length_rate`.
Pull those endpoint loads through body kinematics once if the attachments move.
Uncommitted branch transitions, invalid geometry and compressive/nonfinite
tension produce zero load. `force_owner="provider-native"` is rejected: an MJX
raw actuator force remains exclusively on the provider path.

### Remaining cap/rim and obstacle gates

Shortest-path evidence above **does not mean shortest path around a capped
finite solid**. Both lateral tangent points must lie strictly inside the
declared axial extent; otherwise this family returns
`CAP_OR_RIM_UNSUPPORTED`, never an approximate cap path or a longer branch
fallback. A straight chord entirely above or below an end is independently
certified free and can succeed without a cap model.

The pinned OpenSim source checks whether both preliminary tangent points are
beyond its display length, but supplies no cap traversal or rim normal-cone
equations. This family deliberately does not repeat that test as a physical
finite-cylinder claim. The [MuJoCo 3.12 spatial-tendon specification](https://mujoco.readthedocs.io/en/3.12.0/XMLreference.html#tendon-spatial)
treats wrapping cylinders as infinite and requires separating sites between
multiple obstacle geoms to avoid an iterative solve. Its `sidesite` rule is
not a source for finite cap/rim shortest paths, arbitrary obstacle ordering,
biological hysteresis, or native dynamic-body obstacle transactions.
No cap/rim or multi-obstacle API is admitted here; those require a selected,
pinned complete geometry/topology specification and independent validation
cases. Provider-native routes remain a distinct authority for their own
supported geometries, not an oracle for finite caps.

Run `examples/robotics_opensim_cylinder_wrap.py` for candidate/commit, unequal
axial tangency, fixed-branch power and transition rejection. The dedicated
`tests/unit/applications/test_robot_opensim_cylinder_wrap.py` uses an independent
polar-tangent/axial-stationarity oracle and covers rollback, side selection,
rigid-frame covariance, JIT/vmap/JVP/VJP and ownership. Run
`python -m benchmarks.robotics_opensim_cylinder_wrap --smoke` for separately
measured compilation/execution and declared retained-array memory, candidate,
geometric and power evidence. These drivers do not constitute an OpenSim
executable replay or a held-out physiological validation claim.

Run `examples/robotics_mjx_muscle_projection.py` with the qualified optional MuJoCo/MJX pair.

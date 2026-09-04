# Robotics muscle routes and MJX projections

Phydrax exposes two distinct geometry authorities. `FixedBodyRoutePlan` owns native, piecewise-linear routes through body-fixed points. `MJXPreparedMuscleProjection` only projects fields computed by a same-release compiled MuJoCo/MJX model, including provider wrapping. They are alternatives for route geometry; their forces must not be combined.

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
End-cap contact is not modeled. MuJoCo/OpenSim provider routes remain authoritative
for unsupported three-dimensional cylinder or dynamic obstacle cases.

Run `examples/robotics_analytic_wrap.py`; qualify geometric residuals and the
fixed-branch directional derivative with `tools/qualify_analytic_route_wrap.py`.

Run `examples/robotics_mjx_muscle_projection.py` with the qualified optional MuJoCo/MJX pair.

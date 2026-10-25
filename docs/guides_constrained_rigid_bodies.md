# Constrained rigid-body dynamics

`RigidConstraintDynamicsPlan` advances one fixed-capacity two- or three-dimensional rigid-body population with globally coupled bilateral joints. The implementation is a mass-metric projected kick--drift--kick method. It is not a contact solver, an XPBD compatibility layer, or a claim of symplectic/RATTLE equivalence.

## Configuration and topology

Bodies come from a prepared `RigidBodySetPlan`. Spatial orientations are scalar-first Hamilton quaternions mapping body vectors to world vectors; planar orientations are scalar principal angles. Angular velocity and torque use the corresponding world-frame convention. A fixed body is an ordinary active body whose `fixed_mask` entry is true. Joints to the world use such a fixed body rather than a sentinel index.

Joint plans use stable particle IDs:

- `BallJointSetPlan` stores reference world anchors. In two dimensions it is the native revolute anchor relation.
- `FixedJointSetPlan` stores the complete relative transform.
- `HingeJointSetPlan` is spatial-only and leaves axial rotation free.
- `PrismaticJointSetPlan` leaves one axial translation free.
- `DistanceJointSetPlan` enforces one positive reference length.

`RigidJointGraphPlan.prepare(bodies, reference)` resolves IDs, converts reference geometry to body-local data, and builds a canonical stable row layout. Topology and reference data are immutable within the prepared epoch.

## One constrained step

The unconstrained predictor reuses the rigid-body half kick and Lie-group drift. At the predicted pose, the position solver uses only the three translational and three rotational coordinates of mobile bodies. It globally solves the hard-constraint closest-point problem

```text
minimize  1/2 delta^T M delta
subject to C(Retract(q_predicted, delta)) = 0
```

through its KKT residual. `NewtonKrylov` supplies the matrix-free nonlinear solve and `implicit_root_result` supplies mathematical derivatives at a successful regular root. The reported position constraint values are independently re-evaluated from the reconstructed physical pose.

Velocity is reconstructed from the accepted pose displacement, the closing load is evaluated, and the closing half kick is applied. A full matrix-free saddle solve then projects the final mobile twist:

```text
[ M  J* ] [delta_u] = [      0 ]
[ J   0 ] [ impulse]   [ -J u* ]
```

The final `J(q) u` value is independently re-evaluated. The implementation does not silently add compliance, drop constraint rows, or substitute a minimum-norm solve for an unresolved KKT system.

The principal SO(3) logarithm defines pose displacement. A step whose rotational displacement reaches the declared near-pi chart boundary is rejected rather than wrapped or clipped. Hinge axes additionally retain a positive alignment margin so the anti-aligned branch cannot masquerade as the intended hinge configuration.

## Solver policy and multipliers

`RigidConstraintSolverPlan` owns nonlinear and linear methods, work limits, a characteristic length, componentwise physical tolerances, chart/alignment margins, and invariant tolerances. Translation and rotation have different physical scales; the original componentwise residuals remain available even though scaled constraint coordinates are used by the nonlinear solve.

Every nonempty step audits the rank of the joint Jacobian at the closing-kick
kinematics. The audited operator is the unweighted Jacobian of the
characteristic-length scaled joint residuals with respect to the mobile
translation and rotation coordinates, not the mass-weighted Schur operator. A step
requires full numerical row rank under the strict rule
`sigma > rank_tolerance * sigma_max`, even when the iterative KKT residual is
compatible. The KKT solve itself remains matrix-free; no rank audit becomes a hidden
pseudoinverse or regularization. `RigidConstraintSolverPlan(rank_audit=...)` selects
the route with a `RigidConstraintRankAudit`:

- `"general-dense-svd"` (default) materializes the operator and consumes the native
  exact SVD rank evidence for any joint graph.
- `"certified-forest"` (opt-in) admits only acyclic joint graphs with at most one
  fixed body per connected component; duplicate joints between one body pair count
  as a cycle. It certifies full row rank from verified bounds without a dense matrix
  (see [Certified forest rank audit](#certified-forest-rank-audit)). A failed
  certificate is inconclusive, never a rank-deficiency claim, and rejects the step
  with `RANK_INCONCLUSIVE` unless `dense_fallback` admitted the dense SVD. The
  fallback is planned under its `SVDResourcePolicy` during preparation; an
  unaffordable fallback or unsupported topology raises before tracing.

`RigidConstraintDiagnostics.rank` is a `RigidConstraintRankEvidence`: status
(certified, measured full rank, measured deficiency, dense failure, inconclusive,
empty), executed route, rank lower/upper bounds, the dense condition estimate of an
executed SVD, and, on the forest route, the complete certificate. The certificate
reports a singular-value lower bound, an operator-norm upper bound, a condition
upper bound (never a computed condition number), right-inverse and residual bounds,
the cutoff threshold and its separation, and its numeric binding. A graph without
rows keeps rank zero and condition one. `PreparedRigidConstraintDynamics.constraint_rank_evidence(kinematics)`
runs the audit alone.

`RigidConstraintState` carries position- and velocity-multiplier guesses. They are numerical warm starts tied to the prepared topology, not material state. Position multipliers are projection coordinates, not calibrated forces. Velocity multipliers are impulse coordinates for the executed projection.

## Transactional results

`RigidConstraintStepResult` contains both `candidate_state` and `accepted_state`. Ordinary numerical failure is fail-closed: the candidate remains inspectable, while the accepted state, including warm starts, remains exactly the previous state.

A successful step requires all of the following:

- finite positive step size and finite input/load data;
- successful position and velocity solver results;
- physical position, stationarity, and velocity residuals below their declared tolerances;
- valid quaternion norms, SO(3) chart margin, and hinge alignment margin;
- unchanged fixed-body poses and velocities;
- finite candidate state;
- full row rank of the joint Jacobian, measured or certified by the selected route;
- no kinetic-energy increase beyond tolerance in the isolated mass-orthogonal velocity projection.

`RigidConstraintEvaluation` retains both solver results, both physical residual trees, multipliers, loads, numerical diagnostics, rejection bits, and prepared identity. Derivatives are meaningful only when the step is successful and locally valid; solver-status, rank, hinge-branch, or chart changes are not smooth events.

## Certified forest rank audit

Root every component at its fixed body (the edge to it carries no parent columns)
or, for a free component, at its tree center. Each joint becomes the incoming row
block `A_w x_w + B_w x_parent` of its child body `w`. The blocks are the columns of
the audited operator itself: one operator action per depth-parity color and
coordinate isolates every endpoint column, and preparation checks the assembled
action against the operator on a probe before tracing.

For each block the certificate stores an approximate local right inverse `Q_w` of
`A_w` and the coefficient `C_w = Q_w B_w` actually held in memory. They define the
right-inverse candidate `R`: `x_root = 0` and `x_w = Q_w y_w - C_w x_parent`.
Setting a free root increment to zero only constructs `R`; it does not fix the
body. With verified bounds `a_w >= ||Q_w||`, `b_w >= ||C_w||`,
`u_w >= ||A_w Q_w - I||` and `v_w >= ||B_w - A_w C_w||`, the block-norm domination
matrices of `R` and of `E = J R - I` give

```text
h_w = a_w + b_w h_parent           (0 at roots and anchored parents)
t_v = 1 + sum_children b_w t_w
d_v = sum_children (v_w + b_w d_w)
||R|| <= V   = sqrt(max_w h_w * max_w a_w t_w)
||E|| <= eps = sqrt(max_w (u_w + v_w h_parent) * max_w (u_w + a_w d_w))
```

because the row sums of the dominating matrix of `R` are `h`, its column sums are
`a t`, and `||M|| <= sqrt(||M||_1 ||M||_inf)`. Row block `w` of `E` is
`(A_w Q_w - I) y_w + (B_w - A_w C_w) x_parent`, which gives the two residual sums.
If `eps < 1`, `R (I + E)^-1` is an exact right inverse of norm at most
`V / (1 - eps)`, so `sigma_min(J) >= L = (1 - eps) / V`. The Schur row/column-sum
test of the blocks gives `sigma_max(J) <= U`. Full row rank is certified only when
`L > rank_tolerance * U`, which implies the strict dense rule for the same operator.

Every bound is outward rounded: each floating operation group adds
`16 k eps |x| + 8 k tiny` for its `k` operations, which covers reduction
reassociation, fused multiply-add, flush-to-zero, and device division or square
root within four units in the last place. Products use explicit IEEE arithmetic
rather than reduced-precision matrix units. The recurrences use pointer doubling,
`Sum_k B^k = Prod_j (I + B^(2^j))`, so device work is `O(n log depth)`. Local full
rank, unit pivots, or a small residual alone prove nothing: long chains, large lever
arms, and unequal component scales correctly make the certificate inconclusive when
`L` cannot clear the threshold. The certificate concerns the operator assembled
from the represented blocks; agreement with a separately computed dense SVD also
requires a separation larger than that SVD's own backward error.

## Scope boundaries

Compliance, damping, motors, limits, hard contact, restitution, Coulomb friction, breakage, fixed-capacity topology, rods, shells, transient FEM, and rigid--deformable/MPM coupling use separate physical plans described in the [extended mechanics guide](guides_extended_mechanics.md). They share geometry, solver, and candidate/accepted substrates without changing the bilateral joint model into a universal constraint type.

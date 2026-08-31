# Constrained rigid-body dynamics

`RigidConstraintDynamicsPlan` advances one static three-dimensional rigid-body population with globally coupled fixed, ball, and hinge constraints. The implementation is a mass-metric projected kick--drift--kick method. It is not a contact solver, an XPBD compatibility layer, or a claim of symplectic/RATTLE equivalence.

## Configuration and topology

Bodies come from a prepared `RigidBodySetPlan`. Three-dimensional orientations are scalar-first Hamilton quaternions mapping body vectors to world vectors; angular velocity and torque are world-frame vectors. A fixed body is an ordinary active body whose `fixed_mask` entry is true. Joints to the world use such a fixed body rather than a sentinel index.

Joint plans use the stable particle IDs owned by the rigid bodies:

- `BallJointSetPlan` stores reference world anchors and removes three relative translational degrees of freedom.
- `FixedJointSetPlan` stores the complete relative transform from the reference kinematics and removes six degrees of freedom.
- `HingeJointSetPlan` stores reference world anchors and directed axes. It removes anchor translation and the two transverse relative rotations while leaving axial rotation free.

`RigidJointGraphPlan.prepare(bodies, reference)` resolves IDs and converts reference geometry to body-local data once. Topology, endpoint IDs, reference anchors, and reference axes are not runtime-differentiable state. The first implementation is three-dimensional, fixed-capacity, and static-topology only.

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

Every nonempty step independently materializes the joint Jacobian for the native
dense-SVD rank audit. A graph whose row rank falls below its declared row count is
rejected even when the iterative KKT residual is compatible. The KKT solve itself
remains matrix-free; the rank audit is an explicit first-implementation cost rather
than a hidden pseudoinverse or regularization.

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
- no kinetic-energy increase beyond tolerance in the isolated mass-orthogonal velocity projection.

`RigidConstraintEvaluation` retains both solver results, both physical residual trees, multipliers, loads, numerical diagnostics, rejection bits, and prepared identity. Derivatives are meaningful only when the step is successful and locally valid; solver-status, rank, hinge-branch, or chart changes are not smooth events.

## Scope boundaries

The current contract deliberately excludes contact, friction, restitution, joint limits, motors, compliance, damping, breakage, dynamic topology, two-dimensional joints, and deformable bodies. DEM contact remains under the particle contact APIs. Constraint forces or impulses should not be fed into contact or control code without a separately derived coupling contract.

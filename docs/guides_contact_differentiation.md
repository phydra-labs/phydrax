# Contact differentiation and topology changes

Contact derivatives are local derivatives of one accepted discrete interface
program. Search, route birth/death, closest-feature choice, cone active-set
changes, CCD, capacity growth, remeshing, and lag/evolution refresh are not
ordinary smooth operations.

## Participant derivatives

Every `AbstractContactParticipant` provides positions, velocities, and a force
pullback. `ParticipantDualityEvidence` checks the JVP/VJP identity between a
mechanics direction and a surface force. Nonlinear rigid or articulated maps may
provide explicit trajectory bounds independently of their derivative map.

## Smooth closure derivatives

`contact_closure_gap_jvp` and `contact_closure_gap_vjp` differentiate potential
and normal traction with respect to fixed-route gaps. Evidence includes the
minimum gap/feature branch margin, primal closure success, finite derivative,
and branch qualification.

## Cone derivatives

`contact_cone_solution_jvp` differentiates the converged fixed-iteration cone
map with respect to free contact velocity and effective mass. A derivative is
qualified only away from the cone apex and stick/slip boundary. Exact active-set
or primal-dual KKT derivatives can use the same `ContactConeProgram` without
changing route identity.

## Mortar derivatives

`mortar_gap_jvp` differentiates fixed-route augmented mortar traction. The
multiplier update remains candidate state; it is committed only after the outer
mechanics solve and augmentation convergence succeed.

## Proxy refinement and remeshing

`ContactProxyTransfer` maps proxy vertex fields through declared affine parent
weights. `ContactStateTransferPlan` maps route history and preserves irreversible
damage and wear by taking at least the maximum parent value. Missing parents,
duplicate routes, or non-affine transfer weights reject the transfer.

## Long trajectories

Temporal differentiation should checkpoint accepted mechanics, contact route
state, candidate epochs, and formulation state together. Replaying only mechanics
positions is insufficient because friction, damage, wear, multiplier, and
capacity decisions affect the accepted stationarity problem.

A derivative result must never be presented as qualified when:

- primal or adjoint solves fail;
- a candidate epoch is incomplete;
- a route or closest-feature margin is exhausted;
- a cone route lies on a stick/slip/apex boundary;
- remeshing lacks complete state transfer;
- distributed route ownership is ambiguous;
- a closure component declares differentiation unavailable.

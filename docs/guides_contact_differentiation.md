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

`prepare_contact_cone_sensitivity` classifies every block of a certified
`ContactConeResult` (separating, sticking, sliding, normal-active for
normal-only blocks, frictionless boundary for zero-friction Coulomb blocks) and
prepares one matrix-free linearization of the fixed-branch residual of the same
block law that the primal solved. `contact_cone_solution_jvp` and
`contact_cone_solution_vjp` reuse that linearization: the JVP solves the
implicit equation with restarted GMRES and the VJP solves the transposed
system. The implicit Jacobian is generally nonsymmetric.

A derivative is admitted only with a native regularity proof of that Jacobian.
The caller supplies an explicit bounded `SVDSolvePolicy` as `regularity`;
`phydrax.linalg.certify_rectangular_rank` then certifies full, locally fixed
rank, and the `RectangularRankCertificate` is retained in the preparation.
Without that policy, or when the rank is deficient (for example a sticking
block whose tangential response rows vanish, or duplicated support witnesses),
every derivative is refused; a successful Krylov solve for one right-hand side
is never taken as regularity evidence. Evidence reports `jacobian_regular` and
`regularity_margin`.

The tangent is structured, not a dense matrix. `ContactConeTangent` carries the
response action `(dW) p*` at the solution impulse together with free-velocity,
normal-bias, compliance and friction tangents; geometry and inertia parameters
enter through `(dW) p*` and the free velocity computed by their owners. The VJP
returns a `ContactConeCotangent` whose `response_action` cotangent `m`
represents `dW -> m · (dW p*)`. Static friction does not enter fixed-branch
equations, so its derivative is exactly zero on admitted branches. Padded
components of normal-only blocks keep the equation `p = 0`.

Derivatives are refused, with a zero tangent and explicit evidence, for stale
or uncertified results, cone apices, stick/slip boundaries, unproven
regularity, branch margins below `margin_factor` certificate tolerances, and
blocks that the geometry owner marks inadmissible (for example nonunique
support witnesses). `margin_factor` and `geometry_admissible` are applied on
every call, including reuse of a prepared sensitivity. A frictionless Coulomb
block in contact lies on the friction-parameter boundary: a JVP whose friction
tangent touches it is refused (`friction_boundary_refused`), and a VJP is
refused unless `friction_cotangent=False`, in which case the friction
cotangents are not computed and are `None`. A successful primal does not admit
derivatives through branch or route changes. Admission is decided before any
derivative work: a refused JVP or VJP never evaluates the parameter
linearization or runs the GMRES/transpose solve, and reports
`linear_successful=False`.

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

# Commercial K-way contact and implicit operators

## Simultaneous contact

`KWayMPMContactPlan` builds deterministic fixed-capacity field pairs and solves all
normal complementarity conditions from one nodal state. It currently supports up
to three simultaneously occupied fields, which closes the prior two-field boundary
without claiming unbounded local cardinality.

The normal system uses a Fischer–Burmeister residual. Tangential impulses are
projected simultaneously under the selected sharp or smooth Coulomb law. Essential
velocity components enter the same solve and produce explicit multipliers.

Evidence includes normal/tangential impulses, modes, complementarity, friction-cone
and equality residuals, action/reaction, dissipation, iterations, and convergence.

`MPMRigidActorState` and `apply_rigid_actor_reactions` accumulate impulses and
moments across every contact node before updating one shared body velocity. A
finite-mass rigid actor is never independently advanced node by node.

## Implicit topology

`MPMImplicitUnknownLayout` fixes field/node/component DOFs, contact multipliers, and
rigid DOFs. `MPMImplicitTopologyPlan` fingerprints route, block, field, contact,
material-branch, and topology-generation journals.

The base implicit MPM residual now evaluates the material algorithmic tangent in
every residual call; a failed tangent invalidates the nonlinear state rather than
remaining diagnostic-only.

`linearize_kway_contact` supplies bounded numerical generalized actions for the
converged contact map. Smooth contact remains a smooth-model derivative; sharp
contact is exposed through generalized-derivative evidence rather than a universal
classical gradient.

## Moving domains

`MPMRouteSupersetPlan` freezes one route/dedup topology and differentiates weights,
physical gradients, offsets, and assignment input. The result contains JVP and
transpose actions for position, deformation, and domain state. A route/index/mask
change or insufficient margin rejects and requests an outer topology epoch.

## Compact operators

`MPMCompactImplicitOperator` maps a dense residual, JVP, and transpose action through
`BlockSparseMPMNodalStoragePlan` and reports dense/compact defects.
`MPMSparseContactOperator` and `MPMSparsePhaseFieldOperator` provide compact contact
and phase-field actions. Block-Jacobi and two-level multigrid plans remain explicit,
qualified operator choices.

Nonlinear convergence is bounded and fail-closed; no universal global convergence is
claimed for finite-strain/contact/plastic/fracture systems.

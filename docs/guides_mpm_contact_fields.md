# MPM contact, friction, and nodal fields

## Implicit shared-unknown contact

The nodal impulse paths below remain explicit contact methods. They are not an
inner stage of a monolithic Newton solve. For implicit MPM–ABD/FEM contact, bind
an accepted-history-frozen MPM residual/JVP mechanics block and its geometry to
the [monolithic contact plan](guides_monolithic_contact.md). Freeze particle/grid
interpolation weights and transfer routes for the prepared step; return trial
particle/material/grid history without private solves or early commit.
`prepare_mpm_contact_participant` is geometry-only and does not itself provide
this mechanics block. Contact cross blocks are assembled in the shared root.
Search/envelope, CCD, domain, root, and friction-lag failures roll back all
participants together. Non-associated material residuals retain nonsymmetric
tangents and residual merit rather than an invented contact energy.

`prepare_implicit_mpm_contact_participant(prepared, participant_id=...)` binds the
pure prepared MPM residual. Its unknown is grid displacement (initially zero)
relative to frozen accepted particles; its history is `MPMRuntimeState`.
Bind `.particle_displacement_operator` through the existing geometry preparer
with accepted particle positions. The MPM mechanics participant is
residual-merit-only. Reprepare the pure residual and participant at **every**
accepted boundary: stale time, accepted-step, or topology-generation evidence
refuses the next solve. F-bar and axisymmetric paths without an admitted affine
trajectory bound are refused rather than treated as linear sample trajectories.

## Rigid geometry contact

`RigidMPMContactPlan` evaluates a compiled geometry signed distance and boundary
normal at occupied grid nodes. It projects approaching relative normal velocity and
limits tangential impulse by either:

- `SharpCoulombMPMFrictionPlan` for branchwise stick/slip;
- `SmoothCoulombMPMFrictionPlan` for a declared regularized surrogate.

The result records contact mask/mode, impulse, work, frictional dissipation, contact
step limit, and normal reliability. A moving wall supplies a JAX-pure velocity
provider with a stable identity.

Prescribed velocity is an essential constraint, not contact. Compilation rejects a
prepared overlap between prescribed components and the rigid contact band. MUSL
reapplies contact and essential constraints after its second momentum transfer.

Sharp contact and Coulomb stick/slip are piecewise differentiable. Smooth contact is
the exact derivative of the smoothed model only.

## Multiple nodal fields

`MPMNodalFieldPlan` owns fixed field identities and structural particle field slots.
All `MPMGridState` arrays carry a leading field axis, including the exact single-field
`K = 1` migration.

MPM independently transfers each field's mass, momentum, force, mass gradient, and
velocity state. `KWayMPMContactPlan` constructs every deterministic occupied-field
pair and solves their normal complementarity simultaneously with optional essential
velocity rows and sharp or smooth Coulomb friction. A particle gathers only its
owning field.

The commercial kernel supports a fixed prepared cardinality and currently admits up
to three simultaneously occupied fields per node. Larger local cardinality is an
explicit unsupported claim tuple rather than silently falling back to pairwise order.

Material identity, body identity, velocity-field identity, and topology generation
are separate state. `MPMMaterialBank` stores disjoint material selections and tuple
histories without padding heterogeneous constitutive state widths.

Evidence includes per-field grid state, cross-field leakage checks, contact
frictional dissipation, and action/reaction defect.

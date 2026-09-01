# MPM contact, friction, and nodal fields

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

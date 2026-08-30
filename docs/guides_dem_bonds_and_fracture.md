# DEM bonds and fracture

Bonds are permanent sparse material relations, not transient contact candidates. `FixedBondGraphPlan` resolves stable endpoint IDs once and retains independent stable bond IDs, local anchors, reference vectors, cross-sections, stiffnesses, and damping.

`evaluate_bonds` computes axial, shear, bending, and twisting response in the current rigid-body pose. Bond force and moment reduce directly to owners even when no contact candidate exists. Stored energy and global force/torque residuals remain explicit.

`MixedModeBondDamagePlan` advances monotone scalar damage between declared initiation and failure loading. Damage never heals. Fracture energy grows monotonically and `break_step` is written exactly once when an intact bond fails. Compression/contact after failure remains a separate contact-law decision.

## Topology events

`TopologyEventPlan` fixes owner, child, and event capacities. `split_preallocated_owner` deactivates one source owner and activates preallocated child slots only when mass, linear momentum, angular momentum, IDs, capacity, and finite-state checks all pass. Failure returns the unchanged accepted pool and record.

Topology choices are discrete stopped-gradient events. Replays use event IDs, source/child IDs, causal bond IDs, conservation residuals, and the plan fingerprint. Dynamic allocation, remeshing, and arbitrary fragment creation are unsupported.

# Molecular coarse-graining

PhydraX treats a coarse bead as an active molecular particle, not as a fictitious
chemical element. `element_mask` separates chemical-element identity from active
particle support, while `atom_type_ids` supplies learned or classical interaction
types.

PaiNN and NequIP declare an `AtomisticSpeciesKind`:

- `ATOMIC_NUMBER` requires every active particle to be a chemical element;
- `ATOM_TYPE_ID` embeds explicit interaction or bead types.

The type convention is part of model architecture identity. `maximum_species_id`
therefore replaces the element-specific `maximum_atomic_number` constructor keyword.

## Fixed center-of-mass maps

`MolecularCoarseMapPlan` accepts stable bead IDs, bead type IDs, and one bead index per
active fine particle. The initial contract is a strict disjoint partition:

- every active fine particle belongs to exactly one bead;
- inactive padding uses assignment `-1`;
- every bead is nonempty;
- a bead cannot span molecule or region identities;
- coarse topology is supplied explicitly and is never inferred from distance.

Preparation constructs a non-element `AtomisticSystemPlan` for the beads. Bead masses,
charges, positions, momenta, and instantaneous force labels are respectively mass or
additive reductions of the member particles.

For periodic frames, explicit image counts are authoritative. Without image counts,
the mapper uses anchor-relative minimum images and reports a positive uniqueness
margin. Ambiguous groups fail rather than averaging wrapped coordinates.

## Force matching

`CoarseForceMatchingProblem` maps fine batches and instantaneous forces to a coarse
batch. Optional analytic prior forces are subtracted only when accompanied by a stable
prior ID. `fit_coarse_potential` requires an `ATOM_TYPE_ID` potential and delegates the
actual force-only optimization to `fit_atomistic_potential`.

The fitted scalar potential approximates an equilibrium potential of mean force. The
reported projected and residual force scales are not kinetic-error estimates. Correct
coarse kinetics generally requires separately calibrated friction, memory, or a
generalized Langevin model.

A runtime coarse potential should combine the selected learned residual with the same
analytic prior identified by the force-matching problem. Changing the prior invalidates
the training provenance.

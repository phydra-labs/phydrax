# MPM fracture and sparse storage

## Diffuse phase-field fracture

`PhaseFieldNeoHookeanMPMConstitutivePlan` stores particle damage and irreversible
tensile-energy history. Its spectral Hencky split degrades tensile stress/energy while
retaining compressive response.

`PreparedMPMPhaseFieldDynamics` executes one transactional mechanics/damage macro
step:

1. attempt explicit mechanics at accepted damage;
2. update tensile history;
3. volume-project damage/history to the nodal grid;
4. solve the bounded AT2 phase-field equation with `d_new >= d_old`;
5. gather damage to particles and reevaluate material stress;
6. commit mechanics, damage, and history together.

Evidence reports damage residual, irreversibility, damage increment, and fracture
energy. This is diffuse degradation with fixed particle topology; it does not claim
independent sharp crack-face velocity or contact.

## Sharp alternatives

`MPMFieldPartitionFracturePlan` creates a fixed-capacity topology epoch by assigning
high-damage particles to crack-side velocity fields. Existing two-field contact can
then handle closure/friction.

`CPICFracturePlan` is a separate alternative. It applies particle-node compatibility
tags and supplies particle affine ghost velocities on incompatible routes so APIC
moments remain well posed.

Field duplication and CPIC must not be applied to the same crack: that would suppress
transfer twice and define a different method.

## Active blocks

`MPMActiveBlockPlan` derives block activation from all valid routes, dilates a fixed
block halo, carries current/previous union, produces a dense active-node mask, and
fails before grid update on capacity overflow.

## Compact storage

`BlockSparseMPMNodalStoragePlan` stores fixed-capacity compact blocks, maps logical
nodes/routes to compact slots, and packs/unpacks arbitrary trailing field payloads.
`DenseMPMNodalStoragePlan` is the semantic reference. Dense/compact values must agree
exactly on active nodes.

Activation, compaction, and topology epochs are piecewise structural decisions.
Rematerialized replay requires identical route, block-ID/slot, field, and topology
digests.

The initial compact adapter qualifies explicit payload storage. Compact implicit and
phase-field operators require their own neighbor/preconditioner adapters and are not
claimed automatically.

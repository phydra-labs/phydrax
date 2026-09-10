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

## Sparse nodal storage

`BlockSparseMPMNodalStoragePlan` binds a shared
`SparseBlockTopologyPlan`. The topology is constructed from valid logical
particle-grid routes, stores only canonical block keys and dense local node
IDs, and maps the existing `GatherStencil` into compact storage. There is no
full logical block mask, active-node mask, or MPM-specific page table.

Pass the storage plan as `nodal_storage` when compiling the material-point
problem. A splat prepared against `TensorGridPlan.prepare_index_space` avoids
materializing the logical tensor grid entirely. Mass, momentum, internal and
external force, grid contact, prescribed boundary values, and G2P then execute
on the compact storage-node axis. `DenseMPMNodalStoragePlan` remains the
semantic reference.

Block capacity, storage-node capacity, logical grid-node count, route count,
and topology overflow are separate evidence. A topology candidate is complete
before nodal mutation. A rejected numerical attempt retains the previous
accepted block keys and generation. Post-advection routes must resolve in the
same attempted topology or the attempt rejects.

The compact realization supports one or multiple velocity fields, K-way field
contact, and the prepared implicit grid-velocity solve. Implicit Newton
iterations freeze one compact topology and solve over storage-node unknowns.
Diffuse phase-field evolution constructs compact logical-neighbor routes and
accepts only a complete dependency support. A missing required neighbor
rejects; it is never interpreted as zero.

Activation, compaction, and topology epochs are piecewise structural
decisions. Rematerialized replay requires identical route, logical block,
storage, and topology digests.

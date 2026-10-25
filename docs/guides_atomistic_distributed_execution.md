# Distributed atomistic execution

PhydraX runs one periodic atomistic system across many owners through a single
contract: an `AtomisticDomainPlan` (static admission), a
`PreparedAtomisticDomainRuntime` (one bound layered-model revision) and an
`AtomisticDomainState` (the complete continuation). Two spatial strategies,
distributed reciprocal-space work on slab or pencil FFT layouts, distributed
dynamics with constraints, data-parallel and data x domain training, durable
restart, stage transport, and stage x domain campaigns all share that contract.

Every claim on this page is scoped by what was observed. Lane-reference owners
and forced host-CPU devices are functional oracles of the collective program.
They are not accelerator, scaling, or multi-host evidence. Multi-host execution,
specific accelerators and interconnects, transport providers, and
pretrained-model fidelity are separate qualification gates; see
[Qualification and gates](#qualification-and-gates) and
[Explicit nonclaims](#explicit-nonclaims). An implemented route on this page is a
candidate until its own gate is observed: a C01-C22 closure classification
records implementation, observed evidence, and external prerequisites
separately, and never authorizes a release.

## Domain plan and ownership {#domain-plan-and-ownership}

The plan binds five parts:

- A `phx.discretization.spatial.DistributedOwnershipPlan`. Its
  `MortonAddressPlan` addresses the unit fractional box `[0, 1)^3` with the
  cell's periodic axes. Owner `r` is device `r` of a one-dimensional
  `ExecutionGroup` mesh (`shard_map`). With `owner_lanes`, owners are instead
  named `vmap` lanes of one device.
- A `phx.discretization.FractionalOwnerPartition` that assigns one fractional
  region per owner in the declared partition of the **current** cell.
  Ownership is never guessed from Cartesian cell diagonals.
- One `AtomisticDomainRolePlan` per neighborhood role. Each role has its own
  requirement (cutoff and declared nearest-neighbor selection), complete
  `PeriodicImageStencil` covering `cutoff + skin`, `StreamedRelationPlan`
  tiling, and capacities: `alias_capacity` image aliases per owner pair,
  `edge_capacity` candidate edges per owner, and `halo_capacity` deduplicated
  source columns per owner pair.
- The `strategy`, an owner-order `AtomisticDomainReductionPolicy`
  (`"fast"`, `"deterministic"`, or `"compensated"`), `migration_capacity`,
  `message_capacity_bytes`, the optional graph-partition and reciprocal plans,
  and `dynamic_cell`.
- Optional canonical domain `terms=(...)` (Lennard-Jones, PME/Ewald Coulomb,
  LJ-PME and FourierD3 dispersion, induced-dipole polarization), each on its own
  roles; see [Canonical domain terms](#canonical-domain-terms).

The layered model's role (`term_key="model"`) takes its cutoff and selection
from `model.requirements.neighborhood(AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH)`;
each domain term's roles carry the term's own requirements with
`term_key=term.term_key`. `prepare_atomistic_domain(plan, model=None)` refuses
a role whose requirement differs from the model's or term's requirement. A
terms-only runtime (no model) is a classical distributed program.

Every capacity is explicit. Exceeding one refuses the topology, migration, or
message; nothing is truncated.

```python
import jax
import numpy as np
import phydrax as phx
from phydrax.execution import ExecutionRuntime

phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))

vectors = np.array([[2.2, 0.0, 0.0], [0.6, 2.5, 0.0], [0.3, 0.4, 2.8]])
cutoff, skin, owners, local = 2.4, 0.3, 2, 8
model = phx.nn.atomistic.MACEPotential(
    phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond().scale,
    phx.nn.atomistic.MACEArchitecture(
        species=(1, 8), cutoff=cutoff, radial_basis_count=4, cutoff_power=5,
        channel_count=4, hidden_degree=1, edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2), radial_widths=(8,), readout_width=4,
        average_neighbor_count=8.0,
    ),
    atomic_energies=np.array([-0.3, -1.1]),
    key=jax.random.key(0),
)
cell = phx.discretization.PeriodicCell(vectors)
address = phx.discretization.spatial.MortonAddressPlan(
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
)
runtime = ExecutionRuntime.current()
ownership = phx.discretization.spatial.DistributedOwnershipPlan(
    address, runtime.child_groups(runtime.root_group.spec.device_count)[0],
    local, owner_lanes=owners,                      # lane reference: an oracle
)
requirement = model.requirements.neighborhood(
    phx.atomistic.AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
)
role = phx.atomistic.AtomisticDomainRolePlan(
    requirement,
    cell.image_stencil(requirement.bounded_cutoff() + skin, maximum_image_count=4096),
    streaming=phx.sparse.StreamedRelationPlan(
        receiver_tile=4, edge_tile=32,
        channel_capacity=model.configuration.message_payload_width(),
        accumulation="deterministic",
    ),
    skin=skin, edge_capacity=512, alias_capacity=512, halo_capacity=8,
)
plan = phx.atomistic.AtomisticDomainPlan(
    ownership,
    phx.discretization.FractionalOwnerPartition(cell, (owners, 1, 1)),
    (role,),
    migration_capacity=8,
    message_capacity_bytes=1 << 20,
    reduction=phx.atomistic.AtomisticDomainReductionPolicy("deterministic"),
)
domain = phx.atomistic.prepare_atomistic_domain(plan, model)

numbers = np.array([1, 8, 1, 8, 1, 8], dtype=np.int32)
positions = np.random.default_rng(4).uniform(0.0, 1.0, (6, 3)) @ vectors
species = np.asarray(model.species_indices(numbers, np.ones(6, dtype=bool)))
state = domain.ingress_global(positions, species, rng_key=jax.random.key(7))
evaluation = domain.evaluate(state)              # energy, forces, strain, stress
if not bool(evaluation.successful):
    raise RuntimeError(f"domain evaluation refused: {evaluation.reasons}")
```

## Halo and graph-partition strategies {#halo-and-graph-partition-strategies}

Both strategies run the same layered model through the strategy-neutral
owner-region executor (`owner_atom_energies`, `owner_layer_gradients`).
Communication happens per layer; a single depth-times-cutoff ghost expansion
is never used. Repeated images of one source stay distinct edges, and nonzero
self images are included. A declared nearest-neighbor selection (for example
UMA `max_neighbors`) is applied inside each owner, at every evaluation, to the
complete candidate set. The selection order is the single-device one:
distance, then source stable ID, then image shift.

**Halo domain decomposition** (`AtomisticDomainStrategy.HALO`, the default).
Each owner discovers its receivers' edges from periodic image aliases
exchanged in the current cell. An edge `(receiver row, source column, n)`
carries `d = x_receiver - x_source + n @ H`. Before each interaction,
`HaloColumnExchange` gathers the owned source payload into deduplicated halo
columns through the `DistributedHaloPlan`. Its custom-VJP adjoint returns each
column cotangent to its owner exactly once. A payload larger than
`message_capacity_bytes` is refused at trace time with
`AtomisticExchangeCapacityError`.

**Replicated-geometry graph partition**
(`AtomisticDomainStrategy.GRAPH_PARTITION` with an
`AtomisticGraphPartitionPlan`). Every owner holds the replicated owner-blocked
geometry, runs one bounded native route search per role, and keeps only its
own receivers. The search is an `AtomisticCellListImageSearch` or an
`AtomisticClusterTileImageSearch`; the dense and tiled all-pairs reference
searches are refused.
Source payloads cross owners through `ReplicatedColumnExchange`, a tiled
all-gather whose exact transpose is the tiled
`JaxCollectiveProvider.reduce_scatter`. No owner keeps a reduced full-length
gradient buffer, and no `custom_vjp` blocks forward-mode derivatives.

Every replicated buffer is charged against `replicated_memory_bytes`: the
search geometry, gathered positions, and each layer's gathered columns. The
charges are reported by `graph_partition_memory_evidence`. An over-budget
plan raises `AtomisticGraphPartitionResourceError` (with `.evidence`) before
execution. Capacity failures (cell occupancy, edges per owner, receiver
degree, image count) are reported apart from scientific failures (non-finite
or stale geometry, an incomplete stencil margin).

`ReplicatedColumnExchange` and `reduce_scatter` are declared by the native JAX
provider only. `Mpi4JaxCollectiveProvider` declares no reduce-scatter, and the
graph-partition route is not emulated on top of it.

Halo decomposition combined with data parallelism is not presented as a
replacement for the graph partition. Their relative performance is a
measurement, not an assumption.

## Ingress, evaluation, certificates and migration

- `ingress_global(positions, species, *, rng_key, ...)` is host ingress. It
  wraps coordinates into the current cell, assigns owners, distributes every
  per-atom leaf in logical order, and builds every role topology. A refused
  first topology, a singular cell, or a refused constraint import raises with
  its evidence.
- `ingress_process_local(positions, species, *, rng_key, active, stable_ids,
  logical_indices, logical_count, ...)` takes only this process's rows (whole
  owner blocks, in any owner assignment). It assembles them with
  `make_global_array_from_process_local_data`, adopts them with
  `DistributedPointLayout.from_blocks`, and migrates every continuation leaf
  on device to its fractional owner. There is no global host gather; the only
  host synchronizations are scalar audits.
- `evaluate(state, *, forces=True, stress=True)` returns the energy and its
  split (layered-model `local_energy` and `global_energy`, plus
  `term_energies` in `plan.terms` order), owner energies, owner-blocked atom
  energies and forces, `dE/d strain` (row cell `H' = H @ F.T`) including every
  term's strain partial, stress for fully periodic 3D cells, system contexts,
  the certificate, the status (with per-term capacity and scientific status),
  the `AtomisticStepRejectionReason` bitmask, and `successful`. Any failure
  poisons every numerical output with NaN. `atom_energy_complete` is
  structural: it is true only when the model declares no global payload or
  system energy and the plan has no terms, and is never inferred from a
  numerical zero.
- `certificate(state)` reports the per-role reuse certificate:
  `2 max|x - x_ref| + max_{n in S} |n (H - H_ref)| <= skin`, plus stencil
  completeness for the cutoff in the current cell. A plan without
  `dynamic_cell` refuses any cell change.
- `rebuild(state)` is one global, traceable transaction. It wraps, migrates
  every per-atom leaf (positions, momenta, masses, species, image counts,
  force-cache rows, `atom_payload`), re-anchors constraint rows, and rebuilds
  every role. It commits only if every owner, every role, the cell solve, and
  the constraints admit the candidate; otherwise the accepted state is
  returned bitwise, with the attempt's evidence.
- `with_dynamics(...)` advances dynamical leaves within the current epoch.
  `rebind_model(model, state)` binds a new model revision and marks the force
  cache stale without consuming a step. `to_packet` and `from_packet` move
  per-process host fragments for stage transport.
- Plan, model-revision and run identities travel as dynamic identity words, so one
  compiled executable serves every material and parameter revision of a layout;
  `evaluate` refuses a state bound to another plan or revision numerically
  (`status.identity_bound`, `POTENTIAL`).

Integer route discovery, owner assignment, and topology epochs are not
differentiated; derivatives hold within one accepted epoch.

## System reductions of layered models

Global conditions run as typed system stages of the `AtomisticLayeredModel`
protocol. Owned receivers produce `system_partials`, and the executor combines
them with `owner_system_sum` into the replicated `system_context` consumed by
the next declared operation. The transpose sums each owner's local cotangent
once and returns it once to the owned producers. It never multiplies by the
owner count, never reduces ghost energies twice, and keeps parameter and cell
cotangents.

| Family | Distributed system state |
| --- | --- |
| Native MACE | No system stage and no global payload; the numerical contract is unchanged. |
| Conditioned MACE (joint embedding) | Case-level conditioning bound to owner-local rows (`atom_capacity == local_capacity`). Per-atom condition fields are refused; carry them as `atom_payload`. |
| AIMNet2 | Per-pass charge normalization per system and channel, reduced exactly once. |
| UMA (layered) | Composition, condition, MoLE and channel-balance reductions. The keyed roll gauge uses the evaluation key and per-edge route identities, so a route's roll does not depend on its owner. |
| MACE-LES | Latent charges as the global payload of the distributed latent-Ewald epilogue (next section). |

A position-force route does not imply cell, stress, HVP, or mixed-derivative
support for another family; each family declares its own locality and
derivatives.

## Distributed reciprocal-space work

`AtomisticReciprocalDistributionPlan(definition, *, reuse_reach, strategy=...,
schedule="slab", pencil_grid=None, ...)` distributes the work of a canonical,
term-owned scientific definition: PME, Ewald, multipole, LJ-PME, FourierD3, or
latent Ewald (`AtomisticReciprocalKind`). `prepare_reciprocal_distribution`
binds it to the ownership and partition; the partition must use the term's own
cell. Physics (influence, splines, Nyquist convention, Ewald split) stays in the
definition, and this layer only distributes it:

1. Owned charges, moments, or channels are spread into bounded local patches.
   Patch coverage comes from the current cell and the reuse reach
   (`r * ||H^{-1}[:, a]||`, exact for skew cells).
2. Patch cells are routed with bounded `all_to_all` packets to the owners of
   the canonical physical FFT layout and summed there: x-planes to x-slab
   owners (`schedule="slab"`, capacity `packet_planes`), or `(x, y)`-lines to
   the pencil owners (`schedule="pencil"`, `pencil_grid=(px, py)`, capacity
   `packet_lines`), where owner `r` holds x-block `r % px` and y-block
   `r // px`.
3. The native distributed transform (`DistributedSpectralExecutionPlan.forward_local`)
   produces each owner's modal block (slab: y-rows; pencil: z-slab `r`), and the
   term's kernel is evaluated on it. Slabs run on the ownership mesh axis (its
   lane-reference topology for lane owners). Pencils run on the owner-region
   spectral topology (`SpectralMeshTopology.owner_region`) that factors the
   same one-dimensional owner axis into the `(px, py)` grid, built from the
   ownership group's own device list and checked against its owner-to-process
   map, so the canonical pencil transpose is one all-to-all over the owner axis.
   There is no second owner model, host gather, or replicated mesh.
4. System terms (background, boundary dipoles, anchors) are reduced once.
5. `inverse_local` and the exact transpose of the packet route return the
   conjugate field.
6. The field is gathered at owned atoms as the local adjoint of spreading.

The pencil schedule needs `pencil_grid` with `px * py` equal to the owner count
and a mesh with `K0` divisible by `px`, `K1` by `py`, and `K2` by the owner
count (the modal z-slabs); slabs need `K0` and `K1` divisible by the owner
count. It is refused with explicit modes, with
`strategy="replicated-mesh"`, and with a slab-only option (`packet_planes`);
slab plans refuse `pencil_grid` and `packet_lines`. `packet_lines` may not
exceed its exact bound `min(p0, K0 / px) * min(p1, K1 / py)` for patch extent
`p` and mesh `K`. Preparation checks that every transfer block equals the
canonical local layout shape. The prepared `ReciprocalDistributionReport`
records the layout identity: `schedule`, `spectral_grid`, `owner_coordinates`
and `owner_devices` (`(process, device id)` per owner), the physical and modal
layout ids and local block shapes, `spectral_topology_id`, and
`collective_order`, the exact execution order of the transfer and spectral
collectives. The plan identity binds `schedule`, `pencil_grid`, and
`packet_lines`, so slab and pencil plans of one definition never share an
identity.

Ewald without a mesh distributes bounded mode chunks with their own
reduce-scatter. `owner_reciprocal_energy` returns each owner's energy share
plus `ReciprocalDistributionEvidence`. In that evidence, capacity status
(`patch_covered`, `packets_fit`) is kept apart from scientific status (cell,
finiteness, support, neutrality). `owner_reciprocal_actions` returns
potentials and forces at owned atoms. A domain plan with `reciprocal=...` adds
the model's own global epilogue to every evaluation, including strain. The
binding is checked: MACE-LES requires `plan.reciprocal` of kind
`LATENT_EWALD` built from its own latent definition (same definition and
scale identity), and any other model, or a terms-only runtime, refuses
`plan.reciprocal`. Coulomb reciprocal work of a classical program belongs to
an `AtomisticDomainEwaldTerm` instead.

`strategy="replicated-mesh"` is an explicitly named, bounded comparison: the
deposited mesh is owner-summed and transformed fully on every owner. It is not
a distributed FFT. Byte-budget refusals of any strategy raise
`ReciprocalDistributionCapacityError`; layout and option refusals raise
`ValueError`.

Owner-local polarization (`prepare_owner_polarization`,
`solve_owner_polarization`) solves `(alpha^-1 - T_u) u = E_p` with the native
distributed PCG. It carries global convergence and residual status,
warm-start state, and implicit stationary sensitivity. A declared TCG solver
runs exactly its order with algorithmic sensitivity and is never reported as
converged. No route certifies a caller-supplied reciprocal result: every
reciprocal energy on this page is computed by the distributed term itself.

## Canonical domain terms {#canonical-domain-terms}

`AtomisticDomainPlan(..., terms=(...))` composes canonical prepared
single-device terms with an optional layered model. Each term executes on the
domain role topology of its own requirement (`term_key` = the term name),
uses the canonical term's own pair kernel or reciprocal definition, and keeps
its fixed per-atom inputs (atom types, charges, multipole site data) as
owner-blocked continuation payload that migrates with its atoms.

| Term | Execution |
| --- | --- |
| `AtomisticDomainLennardJonesTerm(prepared_lj)` | canonical Lennard-Jones on its minimum-image role; every unordered pair counted once from its two directed routes |
| `AtomisticDomainEwaldTerm(prepared_pme_or_ewald, reciprocal_plan, *, exception_capacity=None)` | real-space pairs on its role, topology exceptions on the exception support topology, reciprocal/self/background/boundary energy through the distributed reciprocal definition of the same prepared term |
| `AtomisticDomainLennardJonesPMETerm(prepared_lj_pme, reciprocal_plan)` | screened real-space pairs and the exception correction on the term's interaction-site pair role with the canonical `lennard_jones_pme_pair_energy`; reciprocal and self energies through the distributed mesh of the same prepared term, with site-type rows and the anchor flag as owner-blocked payload |
| `AtomisticDomainFourierD3Term(prepared_fourier_d3, reciprocal_plan)` | modified coordination numbers per owner on the term's directed coordination role (`row_coordination_numbers`), per-atom coefficients from `row_coefficients`, and the reciprocal lattice sum plus coefficient-dependent self energy through the distributed mesh; coordination and coefficient derivatives return once through each owner's energy share |
| `AtomisticDomainPolarizationTerm(prepared, owner_polarization_plan, *, exception_capacity=None)` | induced dipoles solved owner-locally (native distributed PCG or a declared TCG order) from explicit site data, real space on its image-pair role, scaled pairs on the exception support topology |

Domain rows are physical atoms. `AtomisticDomainLennardJonesPMETerm` is admitted
only when the system's interaction sites are a proven identity image of its
particles (no virtual sites), so its site pairs are exactly the atom pairs; a
system with virtual sites is refused at term construction.
`require_domain_requirement` admits the interaction-site pair role on the
strength of that proof; other interaction-site requirements are refused. Both
dispersion terms require a reciprocal plan that distributes the same prepared
term (`definition.prepared_id`), on either FFT schedule.

**Minimum-image certification.** Minimum-image roles certify at every
evaluation that no pair has two images, and no atom a self image, inside the
cutoff. Image ambiguity is a scientific failure that poisons the evaluation.

**Exception support topology.** For Ewald/PME and polarization terms, listed
exception and scaled pairs act on their current minimum image independently of
any cutoff, as in the canonical terms. (LJ-PME instead applies its topology
scales inside the pair cutoff on its own interaction-site pair role, exactly as
its single-device pair relation, and takes no `exception_capacity`.)
`domain_exception_topology` splits the replicated pair table across
owners, resolves endpoints through the bounded stable-ID directory, sends both
directed routes to their receivers' owners (weight ½ each), and gathers the
partners through a dedicated `DistributedHaloPlan`. It is rebuilt at every
evaluation with its own `exception_capacity`, which is required whenever the
system lists such pairs. Directory, route and halo overflow are capacity
evidence; a pair naming an absent atom is a scientific failure.

**Model-field terms.** The `AtomisticLayeredModel` protocol exposes
`owner_fields`. AIMNet2 returns the predicted `CHARGES` (the channel sum of
the last equilibrated charges, zero on padding); MACE, MACE-LES and UMA
return none. An `AtomisticDomainEwaldTerm` whose charge field is `CHARGES`
takes its charges from the model through an `OwnerFieldTerm` evaluated inside
the model readout, so the charge cotangent returns through the model exactly
once and predicted-charge AIMNet2 feeds distributed PME. A runtime without the
producing model refuses such a term. Moment or polarizability fields are
refused: no layered model produces them.

**Strategies.** Lennard-Jones, Ewald, LJ-PME, and FourierD3 terms run on both
halo and graph partition (graph partition gathers their inputs through
`ReplicatedColumnExchange`). Polarization is halo-only: its owner multipole
relation binds a physical halo plan, so the graph-partition strategy refuses
it, and it requires a periodic (Ewald or particle-mesh) multipole route.

Training predictions (`owner_predictions`) include the terms with fixed
inputs; their parameter partials are zero.

## Distributed dynamics and constraints

`AtomisticDomainDynamicsPlan(domain, integrator, ensemble, *, constraints=None,
barostat=None, strain_response=None)` runs the P5 schedules over a domain:

- Integrators: velocity Verlet, BAOAB, Nose-Hoover chains, CSVR, deterministic
  rescaling, generalized Langevin, and MTK NPT/NPH. GLE memory `(N, 3, K)` is
  owner-blocked atom payload that migrates with its atom.
- The scheduled isotropic Monte Carlo volume move
  (`IsotropicMonteCarloBarostatPlan`, `monte_carlo_volume_move`).
- Fixed atoms (`mobile_count` with the migrating `mobile` payload) and
  molecular-center volume scaling (`molecule_count` with migrating `molecule`
  labels).

`AtomisticDomainEnsemblePlan` declares the ensemble, atom and mobile counts,
the momentum restriction, and the volume convention. `atom_payload(...)`
returns the payload the method needs at ingress. `PreparedAtomisticDomainDynamics`
then provides `initialize`, `prime`, `step` / `step_detailed`, and
`monte_carlo_volume_move`.

Kinetic energy, kinetic tensor, momentum, degrees of freedom, virial and inner
products are native owner collectives (`owner_kinetic_energy`, ...). They are
computed over the declared reduction, and the static degrees of freedom
(`3 mobile - constraints - 3 [center of mass]`) are checked against exact
collective counts at every transition.

Constraints have their own support topology and capacity
(`AtomisticDomainConstraintPlan`: `row_capacity`, `halo_capacity`,
`directory_capacity`, `migration_capacity`). They are not bounded by the
Verlet skin. Rows are owner-blocked by anchor owner, migrate in every rebuild,
and are solved with the native constrained solve under global status.
Binding and global rebuild preserve `ROUTE_CAPACITY` for directory, row-migration
and halo overflow; a refused constraint commit does not relabel that evidence
as `CONSTRAINT`. Missing members, inconsistent components and failed numerical
constraint projection remain scientific refusals.

Refusals mirror the P5 admission rules and fire before a trajectory starts:
the per-method ensemble admission; constrained MTK; a center-of-mass
restriction with fixed atoms or under per-particle stochastic thermostats;
MTK cell dynamics without a fully periodic 3x3 cell, a `dynamic_cell` domain
plan or the strain response; a Monte Carlo volume move that is not NPT, whose
integrator is not a canonical NVT particle method, that competes with an MTK
cell owner, that lacks a `dynamic_cell` plan, or that scales a constrained
system atomically instead of by molecular centers; and an ensemble without
kinetic degrees of freedom. RNG and counter rules, and rejection and rollback,
match the single-device method.

## Data-parallel and data x domain training

`AtomisticTrainingDistributionPlan(execution_group, /, *, data_axis="data",
reduction=None, data_lanes=None, domain=None)` binds a named data axis. With `domain`, it also
binds a second (owner) mesh axis of the domain plan. Data shards are
group-local (`ExecutionGroupRank`), never the whole process world, and carry
immutable global epoch and draw identities.

`AtomisticDataParallelReduction` sums objective numerators and supports over
the data axis **before** normalization; empty-support shards are inert.
`bind_parameters` returns each rank's parameter contribution exactly once,
with no rank-count or owner-count scaling.

`AtomisticDomainTrainingShard.from_domain_state`,
`atomistic_domain_training_objectives`, and `run_distributed_training_attempt`
execute the shared training kernel on domain rows.

`"deterministic"` and `"compensated"` reductions are reproducible for one fixed
layout. An ordered sum of already-reduced shard totals is **not** partition
invariant: across different shard counts, results agree only within the
declared roundoff bound. Device data x domain execution needs a group whose
mesh axes are exactly `(data_axis, owner_axis)`, where the owner axis size equals
the domain owner count; both axes may be larger than one, and every data row then
executes one domain system. The plan's `collective_schedule` (structure
agreement, objective numerator/support and failure counts on the data axis, then
the parameter cotangent on the owner axis and the data axis) enters its identity.
Lane-reference data shards compose only with lane-reference domains. The
`data-domain-training` gate (both mesh axes larger than one, unequal and empty
support, an independent single-device reference, and a durable restart) is the
evidence for this composition; until it is observed, data x domain training with
both axes larger than one is an unqualified candidate.

## Durability, restart and stage transport

- `publish_domain_checkpoint(repository, runtime, state, *, checkpoint_id,
  writer_id, ...)` is collective. Every process publishes its canonical
  addressable shards as authenticated row-region records. The coordinator then
  re-derives complete coverage and writes the single global commit. Without
  that commit nothing restores: `IncompleteDistributedCheckpointError`. Each
  shard records its lifecycle writer slot (`process_index`, the global process
  index for a whole-job publication) and the physical `source_process_index`
  that held it.
- `restore_domain_checkpoint(...)` performs a same-topology restart that
  restores every leaf bit for bit. Continuation is bitwise only where the
  reductions are deterministic. A changed topology is refused with
  `AtomisticDomainCheckpointError`.
- `migrate_domain_checkpoint(..., policy=TopologyRestartPolicy(...),
  relative_tolerance=...)` performs a changed-topology continuation. It is a
  tolerance-class restart that reads region records only and re-ingresses
  through `ingress_process_local`. Replicated payload, controller, ledger and
  conditioning leaves are restored bit for bit as device arrays placed and
  committed like the template's leaves (typed keys re-wrapped), never as host
  arrays. The returned `AtomisticDomainMigrationEvidence` records source and
  target execution IDs, owner and process counts, rows, read bytes, and the
  verified RNG lineage.
- `checkpoint_domain_state` / `restore_domain_state` is an in-memory host
  checkpoint of a fully addressable state. Restore preserves both the template's
  shardings and its placement commitment, including auxiliary sections and typed
  keys. Uncommitted local cell/state values remain movable alongside owner-sharded
  arrays; deserialization does not pin them to a single device.

Durable records bind `scientific_id` and `hamiltonian_revision_id` independently
of placement and capacity. The scientific identity covers model and term
definitions, role requirements, reciprocal physics, and the declared cell
policy; the revision binds the numerical Hamiltonian parameters. Both are
cached as dynamic identity words in the prepared runtime. Migration requires
both identities to match, including for terms-only programs. Missing identities
refuse rather than being inferred from caller labels or model weights.

Publication places local fully replicated leaves on their owning execution
group before choosing canonical shard writers. It does not mutate the live
continuation. Same-topology restore preserves the template's commitment for
local leaves as well as the numerical values of every shard.

Shard inventories are authenticated repository chunks, with separate byte,
nesting and member limits supplied through `ArrayArchiveLimits`. Large inventories
do not bypass the fixed metadata-value ceiling. The
[distributed execution API](api/execution.md) describes the canonical wire
representation and the required republication of retired checkpoint artifacts.
The repository's whole artifact-manifest byte budget is a separate allocation:
one process holding all stage groups needs room for all their chunk records.
Changing that explicit allocation does not enlarge the generic metadata-value
security ceiling.

Publication also carries named `sections` (global array pytrees such as a
training-kernel state), per-process `process_sections` (for example a
stage-transport state with credits and pending packets), loader cursors and
the training-distribution identity. Process sections are bound to their
process topology: a changed-topology migration refuses them, so drain the
transport before changing topology.

Overlap restore reads only the authenticated row-region records it needs; no
whole-file checksum is skipped. Same-topology restart and changed-topology
migration are different operations with different guarantees.

Stage pipelines use `AtomisticStageTransportPlan` with
`AtomisticTransportStage` (explicit `ExecutionGroup` children, for example
`process_subset_group_spec`) and `AtomisticTransportEdge`. Every round runs one
fixed root-group schedule:

1. a control all-gather of credits, outbox headers, and drain/cancel flags;
2. a fixed-slot payload all-to-all with SHA-256 per fragment;
3. a receipt all-gather.

A grant commits only when every process acknowledges it, so group transfers
are atomic. A full sink parks upstream packets bitwise. Child stage
collectives are entered only through `stage_group`, which refuses
non-members. Rank loss is fail-coupled by the launcher: the job terminates
within its deadline and restarts from the last durable commit. No collective
rollback after a rank disappears is promised.

### Stage x domain campaigns {#stage-x-domain-campaigns}

`AtomisticDomainCampaignPlan(campaign_id, stages, tasks, rng_root=0,
sink="campaign-outputs", outbox_capacity=1, inbox_capacity=1, lane_slots=1,
maximum_rounds=4096, maximum_round_bytes=1 << 26)` runs native distributed
domain dynamics inside a linear multistage campaign. Each `AtomisticDomainStage`
names an explicit child `ExecutionGroupSpec` of one root group, a stage program
(`AtomisticDomainStageProgram`: called only on member processes with the live
child group, returning `PreparedAtomisticDomainDynamics`), a declared
topology-neutral `program_id`, a visit `budget` of accepted steps, and the
`segment` of steps one round may execute. `AtomisticDomainCampaignTask` wraps an
`AtomisticCampaignTask` whose packet carries positions, species, the model's
species indices, optional momenta and cell, plus explicit float64 masses and the
ensemble's `mobile`/`molecules` labels.

- **Binding.** `plan.prepare(root)` is a root-group collective. The root group
  must span every process of the job on a one-dimensional mesh. Each stage's
  domain must own exactly its child group with device owners; a lane-reference
  stage domain is refused.
- **Rounds.** `run_round(state, *, sink=None, cancel=False)` lets every member
  stage admit a task when it holds none (the entry stage from the task list,
  later stages from their inbox), execute one jitted segment of its resident's
  native domain dynamics, and deposit a finished resident. A completed visit
  continues to the next stage over a process-local edge carrying every member's
  `to_packet` fragment (one atomic group transfer, re-ingressed with
  `from_packet` and primed by the next stage); the last stage's completed visits
  and every failed or capacity-refused visit leave as a replicated
  `AtomisticDomainStageReport` for the campaign sink. Each stage holds at most
  one resident task. Task keys are `fold_in(key(rng_root), task.ordinal)`,
  never rank-derived. `run(state, *, sink=None, checkpoint=None, ...)` repeats
  rounds until consensus termination within `maximum_rounds`.
- **Backpressure.** A finished resident whose outbox is full stays parked
  without new physics (`AtomisticDomainCampaignRound.parked`); grants follow
  destination inbox credits in FIFO order; a sink that returns no receipt
  leaves the report parked for a later round. The transport's sink host
  supplies the `AbstractAtomisticCampaignSink`, and publication is idempotent by
  the campaign's semantic output key: a replayed transfer republishes the
  identical payload, and a different payload under a published key is refused.
- **Complete-state checkpoint.** `checkpoint(state, repository, checkpoint_id, *,
  writer_id, limits=...)` (or `run(..., checkpoint=AtomisticDomainCampaignCheckpointPolicy(
  repository, every, prefix, writer_id, limits=...))`) is one lifecycle global commit of the
  transport state (credits, parked and committed packets, sequences, receipt
  counter, transfer digest), the campaign controller (entry cursor, epoch,
  membership, resident task, visit progress, outcome, reasons), the prepared
  runtime, revision, dynamics and topology-independent scientific identities,
  and, for every resident, both the
  complete distributed domain state (every leaf, including the force cache with
  owner energies, every role topology, controllers, ledger, and typed RNG key)
  and its encoded domain packet. Publication reads the accepted state and
  changes nothing; the caller continues with the same state.
  `limits` is the native `ArrayArchiveLimits` admission policy. Size its member
  and manifest bounds for the complete addressable state shards, not merely the
  encoded packets; the generic archive default is not enlarged implicitly.
  Use the same admitted bounds at restore and migration.
- **Same-profile exact restore.** `restore(repository, checkpoint_id, limits=...)` requires
  the checkpoint's execution identity. Each process reads only its own blocks
  and restores every resident's domain state leaf for leaf into a structure
  template, admitted only when the recorded structure identity matches. The
  continuation is bitwise on the same topology and numerical profile; a changed
  topology is refused with `AtomisticDomainCampaignCheckpointError`.
  Local single-device leaves retain the template's uncommitted placement, so
  a compiled segment can place them alongside owner-sharded arrays rather than
  inheriting an unintended single-device commitment from deserialization.
- **Tolerance-class migration.** `migrate(repository, checkpoint_id, limits=...)` requires
  the same campaign and stage programs on a different execution topology
  (process count, stage groups, owners, or capacities). It reads only the
  authenticated regions it needs, re-fragments pending continuations by the
  process-local edge rule (`PreparedAtomisticDomainRuntime.merge_packets`) in
  FIFO order, and resumes residents on their new stage groups. Already committed
  receiving-inbox transfers remain committed; pending sending-outbox transfers
  are re-issued without squeezing both into the sender's capacity. Scientific
  admission binds each stage's Hamiltonian, integrator, ensemble, schedule,
  strain-response and constraint definition. Changed physics refuses even when
  caller labels match. Scientific state, visit progress, RNG lineage and parked
  report evaluation keys are preserved; the trajectory continues within
  roundoff, not bitwise. `AtomisticDomainCampaignMigration` records source and
  target execution identities and process counts, the source round and transfer
  digest, residents, re-issued items, and read bytes. Calling `migrate` on the
  publishing topology is refused (use `restore`).

Restore and migration are distinct operations with distinct guarantees; neither
is inferred from the other, and no bitwise equality across backends (for example
GPU to CPU) is claimed. The `stage-domain-campaign` gate is the evidence for this
section; until it is observed, stage x domain campaigns are an unqualified
candidate.

## Envelope refusals

| Condition | Refusal |
| --- | --- |
| Counting other than directed or half-counted pairs, exclusions other than none or topology exceptions, a receiver subset, or `cutoff=None` | `ValueError` from `AtomisticDomainRolePlan` |
| Stencil not the complete box, translating a nonperiodic axis, or not covering `cutoff + skin` | `ValueError` from `AtomisticDomainPlan` |
| Model requirement differs from the plan's model role | `ValueError` from `prepare_atomistic_domain` |
| Per-atom or non-owner-bound conditioning | `ValueError` at ingress |
| Cell change on a static-cell plan; expired certificate; stale owner epoch | evaluation fails with reason bits; `with_dynamics` refuses the cell |
| Alias, edge, halo, migration, or receive overflow | refused topology or transaction (capacity bits, never truncation) |
| Halo message above `message_capacity_bytes` | `AtomisticExchangeCapacityError` at trace time |
| Replicated graph-partition memory above budget | `AtomisticGraphPartitionResourceError` |
| Patch, packet, FFT, or replicated-mesh bytes above budget | `ReciprocalDistributionCapacityError` |
| Graph partition on a provider without reduce-scatter | not declared; no emulation |
| MTK or barostat without `dynamic_cell`; constrained MTK | `ValueError` before the trajectory |
| Same-topology restore of a changed topology | `AtomisticDomainCheckpointError` |
| Restore without a global commit | `IncompleteDistributedCheckpointError` |
| Transport capacity, schema, or divergence | `AtomisticTransportCapacityError`, `AtomisticTransportSchemaError`, `AtomisticTransportDivergenceError` |
| Device data axis with lane-reference domain ownership | `ValueError` from `AtomisticTrainingDistributionPlan` |
| Term role whose requirement differs from the term's | `ValueError` from `prepare_atomistic_domain` |
| `plan.reciprocal` that is not the model's own latent definition, or on a terms-only runtime | `ValueError` from `prepare_atomistic_domain` |
| Ewald/PME or polarization term with listed exception pairs and no `exception_capacity` | `ValueError` from the term |
| Exception directory, route or halo overflow | term capacity status false (never truncation) |
| Minimum-image ambiguity; exception naming an absent atom | term scientific status false |
| Predicted-charge term without the producing model; moment/polarizability fields | `ValueError` |
| Polarization term on the graph-partition strategy, or with a non-periodic multipole route | `ValueError` |
| LJ-PME term whose interaction sites are not a proven identity image of the atoms | `ValueError` from `AtomisticDomainLennardJonesPMETerm` |
| Dispersion or Ewald term whose reciprocal plan distributes another prepared term | `ValueError` from the term |
| Pencil schedule without a grid, with `px * py` not the owner count, an indivisible mesh, explicit modes, or the replicated mesh; a slab/pencil option on the other schedule | `ValueError` from the reciprocal plan or its preparation |
| `packet_lines` above its exact bound | `ValueError`; packet bytes over budget raise `ReciprocalDistributionCapacityError` |
| Process sections in a changed-topology migration | `AtomisticDomainCheckpointError` (drain first) |
| Campaign root group not spanning the job, or a stage domain not owning exactly its child group with device owners | `ValueError` from `prepare` |
| Campaign `restore` on another topology, or `migrate` on the same topology | `AtomisticDomainCampaignCheckpointError` |

## Qualification and gates

`tools/atomistic_distributed_qualification.py` reports each gate as `PASS`
only when it was observed in that invocation. Otherwise it reports
`NOT_OBSERVED` with the reason, or `FAIL`. A gate never inherits another
gate's evidence. Runtime identity and observed host topology come from the
live `ExecutionRuntime` inventory (Linux boot identities), and resource
admission uses `resolve_execution_plan`.

| Gate | Observed when |
| --- | --- |
| `lane-reference` | lane owners (2 and 3, halo and graph partition) match the independent single-device global MACE oracle; a numerical oracle only |
| `single-process-physical-devices` | one process admits two or more physical accelerators; forced or host-CPU devices record functional evidence and stay `NOT_OBSERVED` |
| `multi-process-one-host` | two coupled `launch_local_processes` ranks on one boot identity match the oracle with process-local ingress |
| `multi-host` | `--observe-current-job` on every rank of an externally launched job with two or more distinct boot identities, admitted with `host_count=2` |
| `killed-rank-fail-coupled` | a SIGKILLed rank fails the job before its deadline and no global commit exists (domain publication and transport probe) |
| `same-topology-restart` | every continuation leaf restores bitwise on the publishing topology |
| `changed-topology-restart` | a 2x2-owner checkpoint continues on 2x1 owners with lineage evidence |
| `stage-transport-subgroup` | the two-process subgroup probe commits and restores bitwise |
| `stage-domain-campaign` | a backpressured two-stage native domain-dynamics campaign over process-subset child groups: credit and sink parking without physics, a killed rank during a commit, bitwise same-topology restart, and changed-topology migration of the committed campaign |
| `data-domain-training` | device data x domain training with both mesh axes larger than one, unequal and empty support, an independent single-device objective and gradient reference, and a durable restart |
| `model-family-domains` | AIMNet2, UMA, MACE-LES, terms-only Lennard-Jones/PME with the LJ-PME dispersion term, and FourierD3 (slab and pencil) domain routes match independent single-device programs on both strategies, with dynamic-cell MTK trajectories; undeclared routes must be natively refused |

The parity, restart, `stage-domain-campaign`, and `data-domain-training` gates
run the two-layer native MACE domain of the smoke driver (native MACE domain
dynamics in both campaign stages, one MACE domain plan per data row). The other
model families, the classical and dispersion terms, and the pencil schedule are
observed only through `model-family-domains`. Until a gate reports `PASS` in a
retained record, the routes it covers stay implemented but unqualified.

```bash
# Local gates (forced CPU devices: functional, never accelerator evidence).
python -m tools.atomistic_distributed_qualification \
    --output benchmarks/atomistic_distributed_qualification.json
# The stage x domain, data x domain and model-family gates alone.
python -m tools.atomistic_distributed_qualification \
    --gates stage-domain-campaign data-domain-training model-family-domains \
    --devices-per-process 2 --timeout-seconds 900 \
    --output benchmarks/atomistic_distributed_campaign_qualification.json
# Inside an externally launched multi-host job, on every rank.
python -m tools.atomistic_distributed_qualification --observe-current-job --gates multi-host \
    --output benchmarks/atomistic_distributed_multi_host.json
# One coupled smoke scenario.
python -m tools.atomistic_distributed_smoke --scenario parity --strategy graph-partition \
    --directory .tmp/atomistic-smoke
# Multi-process integration tests (two processes, forced CPU devices), one file at a time.
pytest tests/integration/atomistic/test_distributed_processes.py
pytest tests/integration/atomistic/test_stage_domain_campaign.py
pytest tests/integration/atomistic/test_distributed_family_processes.py
pytest tests/integration/atomistic/test_distributed_training_processes.py
```

`benchmarks/distributed_atomistic_scaling.py` records phase-separated strong
(`--campaigns strong --total-atoms N`) and weak (`--campaigns weak
--atoms-per-owner N`) rows. Each row covers one family (`mace`,
`mace-conditioned`, `mace-les`, `aimnet2`, `uma`, or the terms-only
`lj-pme` classical program of canonical Lennard-Jones and PME terms),
strategy, and owner execution, and the driver measures:

- preparation (model, plan, runtime, ingress);
- lowering, compilation, first, and warm evaluation, including compiler
  memory and the compiled program's ordered collectives with operand bytes;
- forced owner-crossing rebuild and the post-migration evaluation;
- a distributed velocity-Verlet step;
- a data x domain training attempt, successful only for the canonical accepted
  update outcome, with rule acceptance and support/finite evidence (not applicable
  for a terms-only runtime, which has no trainable model);
- collective publication and bitwise restore, with declared archive limits and
  the actual published shard count.

`--reciprocal pme pme-replicated-mesh ewald` adds standalone rows: patch and
packet resources, energy and force phases, and the isolated distributed FFT
round trip, checked against `np.fft.fftn` on the canonical blocks.
Both forward and round-trip errors must be finite and within the predeclared
float64 tolerance, `1e-12 * max(1, largest reference magnitude)`; returning without
an exception is not an FFT correctness gate.
`--pencil-grids PXxPY ...` adds distributed-FFT PME rows on the canonical pencil
layout (owners `PX * PY`); pencil rows record the layout report and, for device
owners, the traced collective program. One-process PME rows record their
relative parity with the single-device term for energy, forces, and cell
gradient. `--reciprocal-only` records only reciprocal rows. Each row's
`evidence_scope` names the observed platform and owner devices, so forced-CPU and
lane rows are labeled functional only. Refused, out-of-memory, and failed rows
and workloads keep their status and traceback; nothing is dropped.
`--local-processes P --devices-per-process D` relaunches the driver as coupled
workers, with one owner per device of every process.

Use `--checkpoint-members` and `--checkpoint-inventory-bytes` to admit the
complete shard inventory. Generic archive defaults are unchanged; publication and
restore use the same explicit limits, which are retained even in refusal records.
`--workloads publication` measures that lifecycle separately, without repeating
numerical evaluation after a resource refusal.

```bash
python -m benchmarks.distributed_atomistic_scaling --owners lanes --owner-counts 1 2 \
    --families mace mace-conditioned aimnet2 uma mace-les lj-pme \
    --strategies halo graph-partition \
    --workloads evaluate rebuild md-step training-step publication --reciprocal pme ewald \
    --checkpoint-members 2048 --checkpoint-inventory-bytes 8388608
python -m benchmarks.distributed_atomistic_scaling --owners devices --owner-counts 4 \
    --local-processes 2 --devices-per-process 2 --workloads evaluate rebuild publication \
    --checkpoint-members 2048 --checkpoint-inventory-bytes 8388608
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=6 python -m benchmarks.distributed_atomistic_scaling \
    --owners devices --reciprocal-only --pencil-grids 2x2 2x3 3x2 --repeats 2 \
    --output benchmarks/distributed_atomistic_pencil_cpu.json
```

## Explicit nonclaims

- Lane-reference owners and forced host-CPU devices prove the collective
  program, not accelerator performance, scaling efficiency, or multi-host
  operation.
- Two processes on one host are not multi-host evidence. Multi-host operation
  requires two or more distinct observed boot identities in the live
  inventory.
- No accelerator vendor, GPU model, interconnect, MPI provider, or multi-host
  transport is assumed present. A missing gate is `NOT_OBSERVED`, never an
  implicit pass.
- Parity is checked against the same randomly initialized native model on
  one device; pretrained-model and source-artifact fidelity are separate
  gates.
- Changed-topology continuation preserves scientific state and RNG lineage
  within a declared tolerance; future floating-point trajectories are not
  bitwise identical. Bitwise partition invariance of training reductions is
  not offered. The same holds for stage x domain campaign migration.
- Bitwise restore is a same-topology, same-numerical-profile guarantee. No
  bitwise equality between backends (for example a GPU run restored or
  replayed on CPU) is claimed.
- Rank failure terminates and restarts the coupled job. No in-job rollback
  after rank loss is promised.
- Pencil reciprocal layouts across several physical accelerators or hosts have
  not been observed; a host with one accelerator can only run the trivial `1x1`
  pencil. Forced-CPU pencil grids are functional evidence of the collective
  program only.
- Data x domain training, stage x domain campaigns, and the LJ-PME and FourierD3
  domain terms are implemented; their multi-process gates above are the only
  evidence that can qualify them, and none is implied by unit tests of the
  separate data-parallel or domain routes.

## Historical measurements

The following campaigns were recorded with the predecessor driver,
`benchmarks/distributed_mace_scaling.py` (now
`benchmarks/distributed_atomistic_scaling.py`), on the owner-local MACE runtime
that preceded the `AtomisticDomain*` contract. Their JSON records are kept
unchanged as historical evidence. They describe only that code revision,
hardware, and workload.

The A100-SXM4-80GB NVLink campaign used two-layer native-JAX MACE, JAX/JAXlib
0.11.2, CUDA 13, and five synchronized warmed repeats with no competing
workloads. The table reports evaluation medians in milliseconds, not complete
simulation-step latency.

| Campaign | Atoms on one GPU | Atoms on two GPUs | One GPU | Two GPUs |
| --- | ---: | ---: | ---: | ---: |
| Fixed total | 128 | 128 | 45.121 | 42.014 |
| Fixed total | 512 | 512 | 170.104 | 107.035 |
| Fixed total | 2048 | 2048 | 739.855 | 363.964 |
| Fixed per owner | 48 | 96 | 22.466 | 35.134 |
| Fixed per owner | 192 | 384 | 65.717 | 91.258 |
| Fixed per owner | 768 | 1536 | 259.061 | 275.383 |

`benchmarks/distributed_mace_scaling.json` retains the largest fixed-total and
fixed-per-owner cases. For 2048 atoms on two GPUs:

| Phase | Seconds |
| --- | ---: |
| Preparation | 14.566 |
| Lowering | 4.439 |
| Compilation | 16.250 |
| Rebuild | 7.291 |
| Public post-migration call | 21.780 |

The four-H100 80GB HBM3 NV18 campaign used the same route with driver
580.126.09, CUDA 13.4.92, JAX/JAXlib 0.11.2, and `NCCL_NVLS_ENABLE=0`
(`benchmarks/h100_qualification.json`):

| Campaign | Atoms per owner or fixed total | One GPU | Two GPUs | Four GPUs |
| --- | ---: | ---: | ---: | ---: |
| Fixed total | 128 | 35.096 ms | 42.870 ms | 37.243 ms |
| Fixed total | 2048 | 507.424 ms | 299.158 ms | 184.993 ms |
| Fixed per owner | 48 | 15.007 ms | 37.571 ms | 50.771 ms |
| Fixed per owner | 768 | 202.896 ms | 221.797 ms | 245.744 ms |

Both campaigns ran in a single process and are not distributed Pallas or
multi-host evidence. Small workloads did not amortize communication.

## Old to new API map

| Former API | Domain contract |
| --- | --- |
| `OwnerLocalAtomisticPlan` | `AtomisticDomainPlan` + `AtomisticDomainRolePlan` |
| `prepare_owner_local_atomistic` | `PreparedAtomisticDomainRuntime.ingress_global` / `ingress_process_local` |
| `evaluate_owner_local_atomistic` | `PreparedAtomisticDomainRuntime.evaluate` |
| `rebuild_owner_local_atomistic` | `PreparedAtomisticDomainRuntime.rebuild` |
| `rebind_owner_local_model` | `PreparedAtomisticDomainRuntime.rebind_model` |
| `owner_local_loss_gradient` | `PreparedAtomisticDomainRuntime.loss_gradient` (force reference is now owner-blocked) |
| `OwnerLocalAtomisticState` / `Topology` / `TopologyEvidence` / `ExecutionStatus` / `Evaluation` / `Transition` / `LossGradient` | `AtomisticDomainState` / `AtomisticDomainTopology` / `AtomisticDomainTopologyEvidence` / `AtomisticDomainExecutionStatus` / `AtomisticDomainEvaluation` / `AtomisticDomainTransition` / `AtomisticDomainLossGradient` |
| `checkpoint_owner_local_atomistic`, `restore_owner_local_atomistic`, `OwnerLocalAtomisticCheckpoint`, slab checkpoint names | `checkpoint_domain_state` / `restore_domain_state` / `AtomisticDomainCheckpoint`; durable `publish_domain_checkpoint` / `restore_domain_checkpoint` / `migrate_domain_checkpoint` |
| `DistributedAtomisticPlan`, `PreparedDistributedAtomisticRuntime`, `DistributedAtomisticState`, `DistributedSpatialDecomposition`, `DistributedOutputMask`, `DistributedExecutionMode`, `DistributedCollectiveOperations` | `AtomisticDomainPlan` / `prepare_atomistic_domain` / `AtomisticDomainState`; native collectives of the ownership plan (no caller-supplied collective callables) |
| `propose_distributed_migration`, `commit_distributed_migration`, `migrate_distributed_atomistic`, `DistributedMigrationCandidate` | one atomic `rebuild` transaction |
| `exchange_distributed_halos`, `reverse_halo_force_return`, `reverse_distributed_halo_force_return` | per-layer `HaloColumnExchange` / `ReplicatedColumnExchange` inside the executor |
| `evaluate_distributed_atomistic`, `DistributedAtomisticEvaluation`, `DistributedExecutionStatus`, `DistributedPhase`, `DistributedPhaseEvidence`, `distributed_domain_evidence`, `DistributedDomainEvidence` | `evaluate`, `AtomisticDomainEvaluation`, topology and certificate evidence |
| `DistributedReductionPolicy` / `DistributedReductionMode` | `AtomisticDomainReductionPolicy` / `AtomisticDomainReductionMode` |
| `distributed_thermodynamic_reduction` | `owner_kinetic_energy`, `owner_kinetic_tensor`, `owner_linear_momentum`, ... |
| `distributed_constraint_projection` | `AtomisticDomainConstraintPlan` + `solve_domain_constraints` |
| `DistributedPMEPlan`, `PreparedDistributedPME`, `DistributedReciprocalEvidence`, `certify_distributed_reciprocal`, `distributed_particle_mesh_electrostatics` | `AtomisticReciprocalDistributionPlan`, `prepare_reciprocal_distribution`, `owner_reciprocal_energy` / `owner_reciprocal_actions`, `ReciprocalDistributionEvidence` (computed, never certified) |
| `DistributedPolarizationPlan`, `PreparedDistributedPolarization`, `DistributedPolarizationEvidence`, `certify_distributed_polarization` | `OwnerPolarizationPlan`, `prepare_owner_polarization`, `solve_owner_polarization` |
| `halo_short_range_evaluate` | removed; only a labeled test-local reference remains |
| `benchmarks/distributed_mace_scaling.py` | `benchmarks/distributed_atomistic_scaling.py` (old JSON kept as history) |

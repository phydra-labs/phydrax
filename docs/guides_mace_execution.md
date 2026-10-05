# Native MACE execution

`phydrax.nn.atomistic.MACEPotential` is a native, trainable implementation of the
standard real-agnostic MACE architecture family. It runs on the same atomistic
structures, graph topologies, prediction, training, dynamics, artifact, and
deployment owners as PaiNN and NequIP. This guide covers the architecture and
basis contract, exact, tabulated, folded, and accelerated execution routes,
derivative continuity, capacity, hardware, and rights boundaries.

!!! warning "Unreleased candidate"
    All MACE execution routes are unreleased candidates. The capability catalog
    registers four candidate profiles with `released=False`:
    `atomistic.mace-inference.native-exact.profile`, `native-tabulated`,
    `pallas-cuda`, and `distributed`. Each profile lists the evidence gates it
    still needs: numerical and derivative references, source-checkpoint
    fidelity, capacity scaling, operations restart, rights and security,
    scientific validation, and independent review. No independent signed release
    has been obtained. A catalog entry is a declared obligation, not a release or
    a measured result.

## Admitted architecture

`MACEArchitecture` declares one model completely. There is no implicit default
width, degree, interaction depth, or head.

- `interactions` accepts `"real-agnostic"`, `"real-agnostic-residual"`,
  `"real-agnostic-density"`, and `"real-agnostic-density-residual"`. Depth is
  the declared tuple length. One-, two-, and three-layer models are covered by
  tests, and the depth is not fixed at two.
- `correlations` gives the symmetric-product body order of each interaction.
  `hidden_degree <= edge_degree` sets the irreps
  `C x (l, (-1)^l), l <= hidden_degree` carried between layers. Edge harmonics
  reach `edge_degree`, and `channel_count` is `C`.
- `radial_basis_count`, `cutoff_power`, and `radial_widths` declare the Bessel
  basis, polynomial envelope, and bias-free radial MLP. `distance_transform="agnesi"`
  with `agnesi_parameters=(a, q, p)` selects the source's Agnesi transform.
  `cutoff_placement` puts the envelope either on the embedding (`"embedding"`)
  or on the radial-network output weights (`"weights"`).
- `heads`, `head`, and `energy_scaling` (`"unscaled"` or `"scale-shift"`) select
  multihead readouts. The model's `atomic_energies` has shape `(heads, species)`.
  `energy_scale`/`energy_shift` are given per head, and only for scale-shift
  models.
- `pair_repulsion=True` adds ZBL repulsion to the first layer. Both ZBL and
  Agnesi require atomic-number species.
- `readout_correlation` is admitted only for one interaction. It adds the
  separate invariant readout product (reshape, symmetric contraction of that
  correlation, product linear, nonlinear readout). This is a distinct stage,
  not a two-layer model with its second interaction removed.
  `last_readout_only` and `agnostic_product` reproduce the corresponding
  source options.

Readout energy assembly follows the source placement: per-node interaction
energy is the ZBL share plus every layer readout. Scale-shift models apply the
selected head's `scale * interaction + shift` to that sum. The reference energy
`E0[head, species]` is added afterwards and is never scaled.

The following are permanent nonclaims of this implementation, not missing
standard-MACE support:

- MACE-MH-1's nonlinear interaction architecture;
- MACEField polarization or field response;
- arbitrary custom e3nn modules;
- checkpoint conversion without a declared architecture;
- unrestricted angular degree or body order;
- differentiability through neighbor discovery, image enumeration, or
  ownership changes;
- the universal accuracy of any pretrained model;
- the capacity, speed, and scaling figures reported for the reference
  implementation (for example 11.24M atoms, 3.1–5.0x, and 93.8% efficiency).
  These are external reported results, not acceptance targets or promises.

## Construct and evaluate a native model

```python
import jax
import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
architecture = phx.nn.atomistic.MACEArchitecture(
    species=(1, 8),
    cutoff=3.0,
    radial_basis_count=4,
    cutoff_power=5,
    channel_count=4,
    hidden_degree=1,
    edge_degree=2,
    interactions=("real-agnostic", "real-agnostic-residual"),
    correlations=(2, 2),
    radial_widths=(8,),
    readout_width=4,
    average_neighbor_count=2.0,
)
model = phx.nn.atomistic.MACEPotential(
    units.scale,
    architecture,
    atomic_energies=jnp.asarray([[-1.0, -2.0]], dtype=jnp.float64),
    key=jax.random.key(0),
)
water = phx.atomistic.AtomicStructure(
    jnp.asarray([8, 1, 1]),
    jnp.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    jnp.asarray([15.999, 1.008, 1.008]),
    units.scale,
    cell=jnp.asarray([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]]),
    periodic_axes=jnp.asarray([True, True, True]),
)
batch = phx.atomistic.AtomisticBatch.from_structure(water)
execution = phx.atomistic.AtomisticGraphExecutionPlan(
    32,
    backend="particle",
    streamed=phx.sparse.StreamedRelationPlan(
        receiver_tile=16, edge_tile=128, accumulation="deterministic"
    ),
    image_capacity=phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=1024,
        maximum_degree=64,
        maximum_images=343,
    ),
)
prediction = phx.atomistic.energy_and_forces(model, batch, execution, compute_stress=True)
```

The model above is a random native initialization. It demonstrates the
workflow and has no physical accuracy. Construction with only `key` follows the
source family's random initialization. The trusted source converter supplies
all of `embedding`, `geometry`, and `layers` instead. Both routes run the same
owning validators, which also run after artifact restore.

## Basis, parity, and parameter fidelity

Hidden, edge, and target features use `O3IrrepLayout`, an ordered real-irrep
layout with explicit degree, parity, and multiplicity. Edge directions use
fully normalized real Cartesian harmonics (`RealCartesianHarmonics`) with no
polar-chart singularity. Each interaction couples features and harmonics with
an `uvu` `O3TensorProductPlan`. The same coupling owner also serves the
physical Cartesian `O3Representation` used by NequIP and the meshfree laws.
The two layouts are different contracts, not aliases. Equal block sizes never
imply a basis map.

The symmetric product keeps the source parameterization:

- `SymmetricContraction` holds the original species-conditioned weights `W` as
  parameter leaves, with shape `(species, paths, channels)` per output block
  and correlation.
- `SymmetricContractionBasis` holds the fixed coupling projection `P`. It
  records its origin (`"native-coupling"` or `"imported-source"`), source
  identity, equivariance residual, and admitted tolerance. Including the
  tolerance intentionally changes basis, plan, architecture, and artifact
  identities; old identities are not compatibility aliases. Restore validates
  actual coefficients and routing, not only recorded labels. Dense scientific
  replay has a separate preallocation entry bound.
  Native construction admits dense coupling-family and projection workspace
  before generation or SVD. Fixed source tensors must be real; complex values
  are refused rather than projected onto their real part.

- `SymmetricContractionPlan` shares one sparse product graph across the
  correlation orders. It bounds its node count before allocation.
- `MergedSymmetricContraction` binds the folded coefficients `C = P W` to one
  `W` revision as fixed data. Optimizing merged coefficients would enlarge the
  source parameter space, so training always uses `SymmetricContraction`. A
  binding against changed weights raises `StaleSymmetricContractionBinding`.

`mace_potential_from_source` reconstructs a model from one exact source tensor
inventory. Every orthogonal map `degree_transforms[l]` satisfies
`x_native = T_l @ x_source` for real degree-`l` components. The source's
`wigner_3j(l1, l2, l3)` tables in `couplings` certify each coupling path's
native sign, and coupling path scales are restricted to basis signs of
magnitude one. The native graph displacement is `r_receiver - r_source + n @ H`.
The source convention is read from the admitted provider and transformed
explicitly, including odd-degree signs. It is never guessed from shapes. See
the [interoperability guide](guides_atomistic_interop.md) for the trusted
converter that calls this owner.

## Graph topology and streamed layers

Topology preparation and numerical differentiation are separate boundaries:

- `prepare_atomistic_graph_topology(batch, execution, cutoff=..., skin=...)` runs
  once on the host with concrete positions and cells. It searches directed
  image routes within `cutoff + skin` with explicit integer shifts, stable
  receiver-major route IDs, case isolation, and separately charged
  `ParticleImageCapacity` budgets. It also prepares the streamed schedule once
  per topology epoch. `backend="particle"` uses the case-batched fractional cell
  list. `backend="dense"` is the bounded named all-pairs-times-stencil reference
  and requires `maximum_dense_atoms`.
- `bind_atomistic_graph` and `atomistic_energy_derivatives` only compute
  geometry `d_e = x[receiver] - x[sender] + n_e @ H[case_e]`, cutoff masks, and
  derivatives. They never discover, sort, or hash topology, so they are valid
  under `jit`, JVP/VJP, and mixed derivatives.

Runtime topology reuse binds exact endpoints, validity masks, image shifts,
stable route identities, and the attached streamed schedule. A matching shape
or relation schema is not a geometry certificate. Image states retain their
discovery positions and lattice; using a different cell requires an owning
completeness certificate or an exact rebuild.

Each MACE layer streams over that prepared relation. Every directed edge
evaluates radial weights, real harmonics, the `uvu` coupling of its sender's
`linear_up` features, a scalar density, and, in the first layer, the ZBL share.
Each receiver then runs its complete epilogue exactly once after its final
fragment: `linear_down`, neighbor or density normalization, species
self-connection or residual, symmetric product, product linear, and readout.
High-degree receivers are split across fragments, and their seeded
high/correction accumulator is carried between fragments. Only inter-layer
node states (`MACESourceRows`, the halo-exchanged payload of distributed
execution) and readout energies leave a receiver tile.

`MACELayerUpdate.resources` reports the substrate's declared persistent
schedule, fragment, receiver, spill/replay, and cotangent bytes.
`kernel_resources` reports the accelerated kernel's per-fragment bind bytes.
Neither is a compiler measurement. Compiled-executable memory is separate
evidence from `phydrax.execution` memory estimates. Requested graph-wide
outputs remain O(edges × width) and are charged. There is no constant total
memory claim and no predicted per-atom capacity. See
[sparse spatial hierarchies](guides_sparse_spatial_hierarchies.md) for the
streamed relation contract.

## Energy, forces, stress, and derivatives

`energy_and_forces(..., compute_stress=True)` returns forces as the negative
position gradient and stress as the tensile-positive Cartesian tensor
`sym(dE/d strain) / |det H|`. Both come from the same scalar energy. Row
lattice vectors deform as `H @ F.T` with `F = I + strain` at fixed fractional
coordinates and fixed integer images. Pressure is `-trace(stress) / 3`.
Self-image edges can give zero net positional force and nonzero stress.
`AtomisticProvenance.stress_conventions` names each case's meaning:

- `CAUCHY_TENSION_POSITIVE` for fully periodic cells;
- `CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME` for partially periodic cases, whose
  full invertible 3×3 cell is the caller's declared embedding. That value is a
  cell-volume-normalized stress, not an intrinsic surface or line stress.

`stress_case_mask` requests stress for selected cases of a mixed
finite/periodic batch. Unrequested cases return NaN with
`stress_available=False`, never a valid-looking zero. A failed case poisons
both values and derivatives with NaN rather than returning a successful zero
gradient.

Supported derivative paths on a frozen topology: energy; E/F/S; coordinate and
parameter JVP/VJP; mixed force-loss and stress-loss parameter gradients, which
are the parameter gradients of first coordinate/strain derivatives; and
coordinate HVP for the exact network. Continuity is part of the route:

- `PolynomialCutoff` has value, slope, and curvature zero at the cutoff. Its join
  is C2, and the third derivative jumps there.
- Tabulated radial tables admit first coordinate derivatives only, because their
  cutoff join is C1 (clamped zero slope). Hessians and force/stress training use
  the exact network.
- PaiNN and NequIP keep their cosine cutoffs. Their spatial second derivative is
  piecewise at the cutoff, and no checkpoint cutoff is smoothed silently.

## Training

`MACEPotential` trains with the shared `fit_atomistic_potential` on energy,
force, and stress labels (see
[atomistic learning](guides_atomistic.md#energy-force-and-stress-training)).
Every accepted update changes the canonical `NumericRevision` from
`atomistic_potential_revision`. Folded, merged, and tabulated bindings of the
previous revision then refuse.

## Prepared frozen inference

`prepare_mace_potential` binds one exact model at its current revision for
inference. It is a host boundary over concrete parameters and folds only
genuinely linear, species-local stages:

- first-layer `linear_up(embedding)` and residual rows become per-species rows;
- a non-residual `self_connection(linear_down(m) / average)` becomes one
  per-species matrix per target block;
- symmetric products bind merged `C = P W` coefficients;
- with `radial="tabulated"`, radial networks become smooth tables for the
  active ordered species pairs.

Folding changes rounding order. The prepared form therefore has its own
`prepared_id` and records `source_revision_id`. `validate()` reruns the exact
model's validators, recomputes every fold, and raises `StaleMACEPreparation`
when the embedded model drifted. `active_species` bounds the folded and tabulated
species data. Particles of other species refuse at evaluation instead of being
mapped to another element. Training and `energy_and_forces` use the exact
`MACEPotential`. The prepared form evaluates a bound graph through `graph_energy`
and is a registered artifact type.

```python
topology = phx.atomistic.prepare_atomistic_graph_topology(batch, execution, cutoff=3.0)
prepared = phx.nn.atomistic.prepare_mace_potential(
    model,
    radial="tabulated",
    tables=phx.nn.atomistic.RadialTableDeclaration(0.3, 2048, layout="projected-width"),
)
prepared.validate()
policy = phx.nn.atomistic.RadialTableQualificationPolicy(
    value_tolerance=1.0e-6, first_derivative_tolerance=1.0e-4
)
qualifications = [
    phx.nn.atomistic.qualify_radial_tables(
        layer.radial_table,
        model.geometry.radial,
        model.layers[index].interaction.radial,
        policy,
    )
    for index, layer in enumerate(prepared.layers)
]
graph = phx.atomistic.bind_atomistic_graph(
    topology, execution, batch.positions.reshape((-1, 3)), cutoff=3.0, cell_vectors=batch.cells
)
energy, atom_energy = prepared.graph_energy(
    batch.atomic_numbers,
    batch.atom_mask,
    batch.atom_cases,
    batch.case_count,
    batch.atom_capacity,
    graph,
)
```

In one local run, the tabulated energy of this random model differed from the
exact energy by `5.4e-12` eV. Both layers passed the declared value/first-
derivative policy with admitted derivative order 1; maximum logarithmic first-
derivative errors were `1.08e-7` and `1.01e-7`. This is a bounded local table
qualification, not a second-derivative or material-accuracy claim. Other models
and table resolutions need their own held-out qualification. Call
`require_passed()` before relying on a tabulated realization; a small energy
defect alone is insufficient.

Tabulation is a different numerical identity with the same learned parameters.
It is not qualified automatically:

- `RadialTableDeclaration(grid_min, node_count, layout=...)` declares uniform
  support on `[grid_min, cutoff]`. Radii below the support refuse, and radii
  beyond the cutoff are exactly zero. There is no invisible exact fallback
  under the tabulated name.
- Slopes come from a not-a-knot/clamped cubic-spline solve, not from the
  network's JVP. `RadialSpeciesBinding` keeps the full model species domain and
  tabulates every ordered active pair without assuming symmetry.
  Atom-type domains may contain identifier `0`; atomic-number domains retain
  their own positive-element contract. Negative and out-of-domain table
  indices are unsupported, not wrapped or clipped onto another species pair.
- `select_radial_table_layout` chooses `"projected-width"` or `"embedding-width"`
  tables by an explicit `"minimum-table-bytes"` or `"minimum-edge-work"`
  objective and returns `RadialTableResources` for both. Embedding-width tables
  are admissible only when the final radial layer is linear without
  postprocessing.
- `qualify_radial_tables` compares the exact network and the table on held-out
  radii for every active row. Its `RadialTableQualificationPolicy` tolerances
  are fixed before any result is collected. `RadialTableQualification` retains
  every failure, the cutoff slope, second-derivative jumps, refusal evidence,
  and `admitted_derivative_order`. These errors are in internal radial feature
  units and do not bound energy, force, or stress errors. End-to-end E/F/S and
  NVE comparisons are separate evidence.

## Accelerated coupling

`phydrax.backends.atomistic` owns admission of the accelerated MACE edge
coupling. The kernels are written once for Pallas Mosaic GPU with 128-lane
warpgroup channel tiles:

- `"cuda"` compiles them for an NVIDIA device with compute capability at least
  9.0. `pallas_atomistic_availability("cuda")` refuses other devices with a
  reason. No CUDA hardware qualification has been recorded: this target has no
  measured correctness, performance, or memory tuple.
- `"cpu_interpret"` runs the same kernel bodies through the Mosaic GPU
  interpreter on CPU. It is semantic reference verification only. It is never
  GPU correctness, performance, or memory evidence.
- No target substitutes ordinary JAX for a kernel. HIP and TPU acceleration are
  not implemented.

```python
coupling = phx.nn.atomistic.MACEAcceleratedCoupling(
    phx.nn.atomistic.MACEKernelPlan(
        target="cpu_interpret",
        precision="float64",
        receiver_tile=2,
        edge_tile=4,
        channel_tile=128,
        reduction_programs=2,
        fragment_budget_bytes=1 << 26,
    )
)
interpreted = model.with_acceleration(coupling)
accelerated = phx.atomistic.energy_and_forces(interpreted, batch, execution)
```

Acceleration is an execution policy, not architecture. It never enters
`architecture_id` or the numeric revision, and every parameter keeps its source
role, so the accelerated route remains trainable. Admission and execution rules:

`MACEEdgeCouplingSpec` is prepared from the `O3TensorProduct`, not only its plan.
Its static `coefficient_support` describes the actual executed fixed tables,
including admitted imported entries at native zeros. The values remain dynamic
numerical leaves; stale support after a table mutation is refused.

- The streamed relation still owns tiling, seeded spill, and the single
  receiver epilogue.
- The kernel plan's `accumulation` must match the streamed schedule's.
  `MACEKernelPlan` defaults to `"deterministic"`, so the example's execution plan
  declares `accumulation="deterministic"`. A mismatched stream refuses.
- `fragment_budget_bytes` admits the declared bind bytes of one fragment
  (`MACEFragmentResources`). An oversized fragment is refused, never split
  inside the kernel. `require_fragment_routing` checks the routing ownership
  contract once per prepared topology epoch.
- One JAX primitive carries JVP, transpose, and batching rules over the
  source, radial, harmonic, and coefficient kernels, up to the plan's
  `maximum_derivative_order` (default 2). Transformations outside that envelope
  raise `MACECouplingDerivativeError`.
- `MACEKernelEvidence` records the admission, executable signature, declared
  fragment resources, and topology identity of the selected route.
  `prepare_mace_potential(..., edge_coupling=...)` applies the same kernel to a
  prepared realization.

## Distributed execution

Owner-local multilayer execution applies the same layer kernels to owned
receivers with halo source rows per layer. Reverse cotangents are returned in
reverse layer order. See
[distributed atomistic execution](guides_atomistic_distributed_execution.md).
The local owner-lane oracle is a numerical check, not hardware qualification.
Actual execution on two forced CPU devices is distinct from multi-host or GPU
tuples, which are not qualified.

## Source checkpoints and rights

The trusted checkpoint converter, its isolated pinned provider interpreter, and
the explicit trust and archive bounds are described in the
[interoperability guide](guides_atomistic_interop.md). Only that provider
subprocess imports torch, e3nn, or mace-torch. Native artifacts load without them.

Rights are separate from bytes:

- No checkpoint weights are bundled with Phydrax.
- The provider code, each weight file, and any data terms carry independent
  licenses. A readable URL or a matching SHA-256 grants no redistribution
  right and does not establish trust in a pickle.
- MIT-licensed foundation rows (MPA-0 medium and the MP-0b, MP-0b2, and MP-0b3
  standard variants) may be supplied locally for a fidelity campaign.
- OMAT-0, OFF23, and MH-0 weights are distributed under the Academic Software
  License and remain rights-blocked here. They were not downloaded.

See the official terms in the
[mace-foundations README](https://github.com/ACEsuit/mace-foundations) and the
[mace-off README](https://github.com/ACEsuit/mace-off). Per-row source fidelity
is not published by this guide. A model name the converter recognizes is not an
admitted row.

## Identity and persistence

- `MACEArchitecture.architecture_id` content-addresses the declaration.
- `atomistic_potential_revision` covers only parameter leaves.
- A model artifact's `content_id` also covers fixed bases, coefficients, and
  tables.
- `prepared_id` names a prepared realization.

`write_atomistic_model_artifact` and `read_atomistic_model_artifact` persist
exact and prepared models without pickle. Restore reruns the owning scientific
validators and recomputes every identity. See
[atomistic learning](guides_atomistic.md#native-model-artifacts-and-restarts).

# Sparse spatial hierarchies

Phydrax separates three spatial objects that share Morton addressing but carry
different scientific semantics:

- `MortonPointHierarchyPlan` builds an occupied point quadtree or octree. Empty
  space is absent. Leaves own contiguous ranges in canonical Morton order.
- `SparseVoxelGridPlan` stores fixed-resolution voxel samples in aligned bricks.
  Missing samples are unsupported unless the field declares a constant
  background.
- `AdaptiveDyadicGridPlan` stores mixed-resolution control volumes. Covering
  topologies partition the root box exactly and may enforce face-based 2:1
  balance.

A point cluster, a voxel sample, and an adaptive control volume are not
interchangeable. Only integer addressing, ordering, bounds, capacities, and
topology evidence are shared.

## Shared active-key and block substrates

`KeyGroupPlan` is the fixed-capacity active-container authority used by
particle cells, route reductions, sparse blocks, and other keyed worklists. It
sorts explicitly by key and stable item ID, stores only active group keys, and
returns reversible item permutations, group starts/counts, binary lookup, and
independent group/member overflow evidence. Logical key space size does not
allocate a dense reverse map.

`TensorGridPlan.prepare_index_space(bounds)` prepares structured axes and
point/interval entity layouts without materializing tensor-product points,
measures, or boundary masks. `TensorIndexLayout` evaluates coordinates,
measure weights, and physical-boundary membership only at requested logical
IDs. Dense materialization is explicit.

`SparseBlockTopologyPlan` combines one tensor index layout with canonical
active keys. It stores fixed-capacity outer blocks and dense local sites,
returns logical-to-storage lookup and topology transitions, and never treats a
storage slot as physical identity. Closure offsets are explicit physical or
route-support requirements; execution scratch halos are separate.

`RelationExecutionPlan` groups existing `EdgeRelation` or `RowRelation` routes
by target. Fast, canonical deterministic, and compensated sums share one
prepared execution order. Invalid routes are numerically inert, and compact
target output is available without allocating the full logical target space.
It reduces route values that already exist; `StreamedRelationPlan` (below)
evaluates a nonlinear per-route callback and a receiver epilogue inside bounded
tiles over the same relations and the same seeded reducers.

These are execution substrates, not a new field language. Each domain retains
its own inactive-state, boundary, conservation, and topology-transition
semantics.

## Streamed nonlinear relations

`phydrax.sparse.StreamedRelationPlan` executes an additive nonlinear relation
without materializing graph-wide per-route messages. For every admitted route it
evaluates one pure edge callback on that route's source, receiver, and edge
rows; it adds the returned message into its receiver's seeded accumulator; and
after the receiver's final route it applies one complete receiver epilogue to
the finished aggregate. Only receiver outputs and explicitly requested edge
outputs leave a tile.

### Prepared schedules

`plan.prepare(relation, owner_id=..., epoch=..., stable_route_ids=...,
receiver_valid=..., source_valid=...)` accepts an `EdgeRelation` or a
`RowRelation` (lowered through `as_edge_relation()`) and returns a
`PreparedStreamedRelation`. Its `StreamedSchedule` is receiver-major: events
are ordered by receiver and then by stable route ID, with reversible permutation
and row offsets. `transpose=True` additionally prepares an independent
source-major schedule, and `prepared.transpose()` streams into relation sources
over it. Swapping endpoints does not reuse or re-sort the target-major order.

Tiles are filled greedily. A tile owns at most `receiver_tile` receivers and
`edge_tile` route lanes. A receiver whose events exceed the remaining lanes is
split into fragments across consecutive tiles. Its seeded high/correction
accumulator (`KeyGroupAccumulation`) is carried from one fragment to the next.
Its epilogue is committed once, after the final fragment, so a nonlinear epilogue
always sees the complete sum. Chunk subtotals are never summed independently,
which is why compensated accumulation keeps its correction across fragment
boundaries. Rows are never truncated. On concrete host topology the exact tile
sequence is prepared; on traced topology inside a compiled rebuild the static
tile bound is `receivers // R + capacity // F + 1`, and an incomplete schedule
is reported by evidence instead.

Requested tiles are upper bounds. Each direction is prepared at effective widths
`R = max(min(receiver_tile, receivers in that direction), 1)` and
`F = max(min(edge_tile, route capacity), 1)`. These are recorded on the schedule
and resolved once from static extents, never from route content, so small
relations carry no padding and the plan identity is unchanged.
`channel_capacity` caps the elements of one event's message plus edge output,
and of one receiver output. A payload above it is refused before evaluation.

Event semantics are explicit:

- Duplicate routes with equal endpoints remain distinct events. Each one is
  evaluated and accumulated separately. There is no CSR-style coalescing of
  nonlinear routes.
- Invalid routes, routes into or out of an endpoint masked by
  `receiver_valid`/`source_valid`, and routes masked by the runtime
  `edge_active` argument are absent. Padded and inactive lanes replay a usable
  lane of the same fragment, so callbacks see only admitted, domain-safe
  inputs, and are masked from accumulation. A fragment with no usable lane skips
  its callbacks with `lax.cond`. Under `vmap` that skip becomes a select, so a
  batched caller's slot-zero route data must itself be domain-safe.
- An empty row, or a row whose routes are all inactive, still receives the
  epilogue of a zero aggregate. A receiver masked by `receiver_valid` commits
  zero.
- Event order is deterministic: receiver-major and then by stable route ID, and
  tiles follow that order. With `accumulation="deterministic"`, permuting route
  storage under fixed `stable_route_ids` reproduces the result bit for bit.
  `"compensated"` streams the same order with an error-free correction term.
  `"fast"` uses the shared segment-sum reducer.

### Callback contracts and payloads

`StreamedPayloadSpec(message, output, edge_output=None)` declares one event's
additive message, one receiver's output, and an optional per-event edge output.
Each leaf is a `jax.ShapeDtypeStruct` with no leading axis. Callbacks are never
probed: traced results must match the declared structure, shape, and dtype, or
evaluation is refused. Messages accumulate in their declared dtype.

- `StreamedEdgeFunction`: `(parameters, source, receiver, edge) -> message`,
  or `(message, edge_output)` when edge outputs are declared. It sees exactly
  one event and must not depend on other events.
- `StreamedReceiverEpilogue`: `(parameters, receiver, aggregate) -> output`.
  It runs once per receiver on the complete aggregate.
- `StreamedFragmentAggregator`: `(parameters, sources, receivers, edges,
  fragment, seed) -> accumulations` (or `(accumulations, edge_output)`). It
  replaces the per-event callback when a consumer owns a specialized
  whole-fragment accumulation. `prepared.evaluate_fragments(...)` calls it once
  per fragment with a prepared `StreamedFragment`, which carries receiver-slot
  lane ranges, fragment-local distinct sources, source-major lane lists, and
  `lane_use`. The aggregator also receives the seeded accumulator of every
  slot. `StreamedFragment.reduce` is the substrate's canonical seeded additive
  reduction for per-lane values. The substrate still owns tiling, spill across
  fragments, the single epilogue, replay, and output commits.
  `prepared.fragments()` returns every fragment's routing, stacked by tile.

Callables that carry arrays remain ordinary dynamic PyTree leaves of
`parameters`. They are not hidden static captures.

Requested edge outputs are explicit graph-wide outputs that lead with the route
shape. They are zero on unused routes and charged in
`StreamedRelationResources.output_bytes`. Their storage is
O(routes × edge-output width) by construction, and returning them does not drop
their cotangent.

### Identity, evidence, and declared resources

`StreamedTopologyBinding` separates topology identity from schedule identity.
`owner_id` names the topology owner's schema, and `epoch` is its dynamic integer
topology epoch, a device leaf. `content_id` fingerprints the concrete route
indices, masks, and stable IDs. It is `None` for traced topology, whose identity
is the owner and epoch. `binding_id` fingerprints `owner_id` and `content_id`.
Equal shapes or capacities never imply the same topology.
`PreparedStreamedRelation.execution_id` combines `plan.plan_id`, `binding_id`,
the direction, the endpoint and route shapes, and the prepared tile counts.
Owners that now embed a `StreamedRelationPlan` in their own fingerprint (for
example `AtomisticGraphExecutionPlan.plan_id`) produce identities that differ
from earlier source revisions. Rebuild dependent artifacts rather than comparing
them with historical IDs.

`StreamedRelationEvidence` reports `active_routes`, `evaluated_routes`,
`committed_receivers`, `tiles_used`, `fragmented_receivers`,
`maximum_receiver_degree`, `duplicate_stable_ids`, `schedule_complete`, and
`finite`. `successful` requires a complete schedule without duplicate stable
IDs, together with finite accumulations and committed outputs. Duplicate stable
route IDs are refused during host preparation. A failed evaluation multiplies
every inexact output by NaN and links that NaN to every differentiable input,
including parameters and callback arrays the outputs do not read (for example a
parameter used only by a failed message under a residual-only epilogue), so its
values and derivatives of every order are invalid rather than silently zero.

`StreamedRelationResources` (also available before evaluation through
`prepared.resources(...)`) declares the logical bytes the substrate owns:

- persistent schedules;
- one edge fragment and one receiver tile of workspace;
- the spill carry and retained replay boundaries;
- admitted outputs and their staging;
- one live cotangent accumulator for differentiable inputs: parameters, source,
  receiver and edge rows, and the dynamic array leaves of array-bearing
  callbacks (pass them as `callbacks=` to `prepared.resources(...)`).

It excludes callback-internal intermediates and compiler temporaries. Under
`replay="full"`, `replay_boundary_bytes` is `None` (undeclared). These declared
bytes are not a compiled-memory bound. Certify a transformed executable
(forward, reverse, force-loss, HVP) separately with
`phx.execution.compiled_memory_estimate`; sampled peaks are observations.
Streaming bounds the edge workspace by the tiles, not by the route count. Total
memory is still not O(1): schedules, node states, requested edge outputs, and
cotangent accumulators scale with the graph.

### Differentiation

The reference execution is ordinary JAX. Forward, JVP, VJP, reverse-over-reverse
(force-loss parameter gradients), and forward-over-reverse (coordinate HVPs) are
JAX's own derivatives of the tiled primal. Each tile body is rematerialized
inside the shared checkpointed-scan replay boundary (`replay="step"`, `"block"`
with `replay_block_size`, or `"full"`), with its own nested checkpoint, so a
second reverse pass also replays tiles. Mixed coordinate/parameter derivatives
need no custom first-order rule. Integer schedules, permutations, and epochs are
discrete topology and are not differentiated.

The shared seeded reducers keep the derivatives of valid zero-valued events.
Primal zero events remain exact no-ops in the high/correction state. Their
additive tangent, however, follows route validity rather than whether the
numerical value is zero. For an active event `theta * x + x * x` at `x = 0`,
the reduction therefore keeps coordinate derivative `theta` and mixed derivative
one. Opposite events that cancel to a zero seeded subtotal keep their parameter
tangent. Zero events also keep their curvature under forward-over-reverse and
reverse-over-reverse transforms (HVPs and force-loss mixed gradients) in every
accumulation mode. Structural padding stays inert through its mask. The same
holds for `reduce_key_groups` in fast, deterministic, and compensated modes,
seeded or unseeded.

### Example

```python
import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx

# Five sources stream into three receivers. Receiver 1 has degree four, so an
# edge tile of two splits it into fragments; receiver 0 receives a duplicate
# route pair; receiver 2 has no routes.
relation = phx.sparse.EdgeRelation(
    np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int32),
    np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int32),
    source_size=5,
    target_size=3,
)
plan = phx.sparse.StreamedRelationPlan(
    receiver_tile=2,
    edge_tile=2,
    channel_capacity=16,
    accumulation="compensated",
)
prepared = plan.prepare(relation, owner_id="docs-streamed-example")

payload = phx.sparse.StreamedPayloadSpec(
    message={"s": jax.ShapeDtypeStruct((2,), jnp.float64)},
    output=jax.ShapeDtypeStruct((2,), jnp.float64),
    edge_output=jax.ShapeDtypeStruct((), jnp.float64),
)


def edge_function(parameters, source, receiver, edge):
    # One event, no leading edge axis: returns (message, requested edge output).
    displacement = receiver["x"] - source["x"]
    distance = jnp.sqrt(jnp.sum(displacement**2) + edge**2)
    weight = jnp.exp(-parameters["decay"] * distance)
    return {"s": weight * source["h"]}, weight


def epilogue(parameters, receiver, aggregate):
    # Runs once per receiver, after its final fragment, on the complete sum.
    return jnp.tanh(aggregate["s"] @ parameters["mix"] + receiver["h"])


key_x, key_h = jax.random.split(jax.random.key(0))
parameters = {"decay": jnp.asarray(0.7), "mix": 0.5 * jnp.eye(2)}
positions = jax.random.normal(key_x, (5, 3))
source_features = jax.random.normal(key_h, (5, 2))
receivers = {"x": positions[:3], "h": jnp.zeros((3, 2))}
edges = jnp.full((6,), 0.1)

result = prepared.evaluate(
    payload,
    edge_function,
    epilogue,
    parameters,
    {"x": positions, "h": source_features},
    receivers,
    edges,
)
assert bool(result.evidence.successful)
assert int(result.evidence.fragmented_receivers) == 1
receiver_outputs = result.receiver_outputs  # (3, 2); row 2 is tanh(receivers["h"][2])
edge_weights = result.edge_outputs  # (6,), one requested value per route


def energy(parameters, positions):
    streamed = prepared.evaluate(
        payload,
        edge_function,
        epilogue,
        parameters,
        {"x": positions, "h": source_features},
        receivers,
        edges,
    )
    return jnp.sum(streamed.receiver_outputs**2) + jnp.sum(streamed.edge_outputs)


@jax.jit
def forces(parameters, positions):
    return -jax.grad(energy, argnums=1)(parameters, positions)


def force_loss(parameters):
    return jnp.sum(forces(parameters, positions) ** 2)


parameter_gradient = jax.jit(jax.grad(force_loss))(parameters)
```

The force-loss gradient differentiates through the source coordinate gradient
of the streamed relation (reverse over reverse) with ordinary JAX. Checked
against an independent dense per-route reference, the energy, forces, and
force-loss parameter gradient match it to rounding.

### Consumers on the shared route

The following consumers run their per-route work through prepared streamed
relations. Each exposes an `execution: StreamedRelationPlan | None` argument
where it is public:

- `phydrax.graph.EquivariantGraphConvolution`: one message per route. Each
  receiver normalizes by its complete incoming weight only after its last route.
- `phydrax.graph.GraphKernelIntegral` with `reduction="sum"` or `"mean"`.
  Count and measure normalization happen in the receiver epilogue. `"max"` and
  `"min"` remain `phydrax.sparse.route_reduce` reductions.
- `phydrax.graph.MeshGraphNetBlock` and `MeshGraphNet`: all processor steps
  share one schedule prepared from the input graph's relation. Updated edge
  latents are a requested graph-wide edge output, so they are charged.
  Only the MLP hidden activations are bounded by the edge tile.
- Graph callbacks entering these operators (`radial_fn`, `kernel_fn`) must be
  wrapped in `phydrax.graph.RouteLocal`. Graph-wide callbacks, such as a
  normalization over all edges, are refused rather than reinterpreted per route.
  `GraphAttentionOperator` and `GraphNeuralOperator` keep their explicit
  route-reduction ownership.
- Meshfree conservation solves
  (`prepare_meshfree_conservation_solve` and
  `prepare_meshfree_coupled_conservation_solve`, `execution=`): each edge
  constitutive law is evaluated once per canonical edge. The streamed receiver
  sum carries `+f` to the second endpoint, and the returned per-edge flux
  carries `-f` to the first, so action and reaction never depend on law parity.
- Atomistic PaiNN, NequIP, and native MACE message passing use the schedule
  prepared once per atomistic graph topology epoch. MACE can also select an
  accelerated whole-fragment aggregator. See [Atomistic learning](guides_atomistic.md)
  and [Native MACE execution](guides_mace_execution.md).

An observed before/after energy and force comparison covers the migrated PaiNN and NequIP
consumers. This is regression evidence for those consumers, not a general
performance or capacity claim for the streamed route.


## Morton addressing

`MortonAddressPlan(lower, upper, maximum_depth)` owns the half-open physical
box `[lower, upper)`. Nonperiodic points on the upper boundary are outside the
domain. Periodic axes wrap the upper boundary to the lower boundary.
Nonfinite or nonperiodic out-of-domain points remain invalid; they are never
clipped into a valid cell.

The canonical code uses at most 63 bits:

- depth 63 in one dimension;
- depth 31 in two dimensions;
- depth 21 in three dimensions.

Topology construction sorts by validity, Morton code, and stable physical ID.
Integer codes, permutations, refinement decisions, and relation routes are
discrete stop-gradient state.

## Point hierarchies

```python
address = phx.discretization.MortonAddressPlan(
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    maximum_depth=8,
)
plan = phx.discretization.MortonPointHierarchyPlan(
    address,
    point_capacity=positions.shape[0],
    node_capacity=9 * positions.shape[0],
    target_leaf_occupancy=8,
)
hierarchy = plan.build(
    positions,
    active_mask=active,
    stable_ids=particle_ids,
)
```

The packed state contains logical/storage permutations, occupied prefixes,
parents, ordered children, leaf item ranges, and physical cell bounds. Every
active point occurs in exactly one leaf range. Coincident points remain in one
range even when its occupancy exceeds the preferred target.

`refresh` refits when cell membership and stable IDs remain unchanged. A
rebuild produces a complete candidate and accepts it atomically. Invalid
points, duplicate stable IDs, or node-capacity exhaustion leave the previous
accepted hierarchy authoritative.

## Compact plane execution and exact point queries

`MortonNeighborQueryPlan` and `MortonRadiusRelationPlan` sort sources once by
their canonical Morton code and stable ID. Every occupied cell at every prefix
level is then one contiguous span of that order, located by binary search over
the sorted codes, so no per-level structure is allocated. Each target visits
the `3^d` stencil around its own coarse cell, skips stencil cells farther than
the active search bound, and packs at most `maximum_candidates` sources into a
fixed-width buffer. Distances and selection run only over that buffer, and
targets are processed in bounded chunks (`target_chunk_size`), so the working
set is bounded by chunk size times candidate capacity rather than by the
target-by-source product.

The k-nearest-neighbor query starts at the finest level whose visited stencil
holds enough sources. Its selection is certified when the k-th distance (or the
radius cap) is strictly smaller than the distance from the target to the
nearest unvisited region, after a conservative floating-point margin. An
uncertified row retries once at the coarser level whose cells are wider than
that distance, visiting only the cells within it. The radius relation uses the
finest level whose cells are wider than the radius. Neighbors are ordered by
squared distance and then stable ID; radius pairs are ordered by target and
source stable ID, or by the smaller and larger ID for `pair_once`.

Every result carries a per-target `status` (`MortonNeighborQueryStatus`):
`COMPLETE`, `INACTIVE_TARGET`, `INVALID_TARGET`, `INVALID_SOURCES`,
`CANDIDATE_OVERFLOW`, or `UNCERTIFIED`. Rows that are not complete return no
neighbors, and evidence reports the required candidate width, overflowing and
uncertified row counts, and source/target validity. A possibly wrong neighbor
is never returned as valid.

```python
query_plan = phx.discretization.spatial.MortonNeighborQueryPlan(
    address,
    source_capacity=source.shape[-2],
    target_capacity=target.shape[-2],
    maximum_neighbors=16,
    maximum_candidates=512,
)
neighbors = phx.graph.query_neighbors(
    source,
    target,
    max_neighbors=16,
    plan=query_plan,
)
```

The dense query remains the default. Supplying a plan is explicit because the
crossover depends on point count, clustering, query/source ratio, device, and
candidate capacity. `neighbors.evidence` identifies the realization and
reports exactness, finite input, completeness, and success. Discrete selected
indices are stopped topology; relative vectors and distances are recomputed
with native minimum-image JAX geometry and remain branchwise differentiable.

Pairwise interaction kernels use a compact Morton plane schedule instead:
leaves contain bounded contiguous Morton-order ranges, and each coarser plane
groups contiguous children without allocating every occupied octree prefix.
`MortonPlaneInteractionPlan` performs deterministic dual-tree refinement over
those planes. A geometric opening policy accepts unequal-size node pairs
for far translation and opens the remainder until exact leaf completion. Every
ordered point pair is covered by exactly one accepted ancestor or one leaf
route. Queue, far-route, and near-route requirements are counted before their
fixed-capacity products are accepted. Node expansion radii are finite powers
of two that enclose their tight point bounds; scale exponent range and failures
remain visible in schedule evidence.

Plane schedules also support distinct source and target supports. Each schedule
retains its own logical permutation, padded node AABBs, and child ranges;
`MortonPlaneInteractionPlan` produces source-node to target-node routes without
assuming equal capacities or shared storage. Prepared integral operators expand
reference AABBs by their declared displacement envelope, retain the route
topology while points move within that envelope, and fail stale when a point
leaves its assigned padded node. Optional node-radius and interaction-cutoff
policies force refinement where a kernel's local resolution or compact support
requires it.


`PeriodicFoFFinderPlan` applies the same exact-link contract through explicit
`direct`, `cell_list`, or `morton_plane` realizations. Group slots are ordered
by each component's minimum stable particle ID. Group and link capacities,
topology completeness, convergence, finite arithmetic, and success are
reported independently; overflow never publishes a partial catalog as
successful.

`MortonRadiusRelationPlan` separately controls candidate capacity, pair
capacity, and open or closed radius boundaries. Candidate or pair exhaustion
invalidates every route; the exact required pair count is reported whenever no
candidate buffer overflowed. The result also exposes the occupied coarse cells
as logical cell slots, counts, and offsets into the Morton storage order.

`DistributedNeighborQueryPlan` and `DistributedRadiusQueryPlan` run over a
`DistributedPointLayout` with arbitrary uneven owners. Each target first
queries its own owner, bounds its search radius by that local result and the
published owner populations, and is sent only to owners whose (periodic)
source boxes intersect the certified ball; answers merge by distance, stable
ID, and owner. Targets are never replicated: communication is bounded by
`maximum_remote_owners × halo_capacity` targets per owner pair, and overflow is
a per-target refusal. `DistributedMortonNeighborQueryPlan` applies the same
query to contiguous logical shards and returns globally indexed rows in
logical target order. The owner-box shell needs one pass but its tightness
depends on how compact the owner regions are; no locality-optimal
repartitioning is implied.

`ParticleOctreePlan3D` uses this substrate. Barnes--Hut uses a batched
branchless walk over compact occupied nodes at moderate capacities and a
bounded stack for larger supports. Both descend through every node containing
the target and evaluate near leaves directly. `opening_angle=0` is the direct
leaf authority. The reported opening indicator is geometric; it is not
presented as a relative-force error certificate.

Extended point primitives use `MortonPrimitiveBoundsPlan`. It keeps center
ownership unchanged while reducing per-item AABBs through leaves and internal
nodes. Surfels use this view because a tangent footprint may cross its center's
Morton cell. See [Surfels](guides_surfels.md).

## Sparse voxel fields

```python
voxel_plan = phx.discretization.SparseVoxelGridPlan(
    address,
    brick_size=4,
    brick_capacity=maximum_bricks,
)
grid = voxel_plan.prepare(active_integer_coordinates)
field = phx.discretization.SparseVoxelField(
    grid,
    brick_values,
    background_mode="unsupported",
)
samples = field.sample_multilinear(query_points)
```

Topology and values are separate. `SparseVoxelField` values remain trainable;
the grid is nontrainable topology state. Nearest and multilinear queries return
storage routes, weights, complete-stencil evidence, and support masks.
Multilinear deposition uses the same stencil authority and deposits nothing
for an incomplete unsupported stencil.

`background_mode="constant"` supplies an explicit value for absent voxels.
There is no implicit-zero convention and no hidden renormalization of partial
stencils.

`VoxelGeometrySamplingPlan` samples a `CompiledGeometry` onto an existing
sparse topology. Sampling always downgrades exact signed-distance claims to an
approximate, piecewise-smooth field. When supplied an
`ExactSDFEnclosureCertificate`, it reports cells whose sign is certified by a
Lipschitz enclosure. Unresolved cells remain explicit.

## Adaptive dyadic cells

`AdaptiveDyadicGridPlan` starts from a root leaf or a validated leaf set.
Refinement allocates complete child families. Coarsening requires all active
siblings. Optional balance closure refines coarse face neighbors until adjacent
leaf levels differ by at most one.

An adaptation returns `DyadicTopologyTransition`:

- requested and accepted refinement/coarsening counts;
- balance-induced refinements;
- maximum-depth rejections;
- required capacity;
- candidate acceptance.

Any failed invariant or capacity requirement preserves the previous topology.
Stable cell identity is the `(level, Morton prefix)` pair, not the storage slot.

`DyadicCellTransferPlan` distinguishes cell averages from cell contents.
Average restriction is volume weighted; content restriction is additive.
Piecewise-constant prolongation preserves both the represented average and the
global integral.

## Finite volume

`DyadicFiniteVolumePlan` lowers accepted covering leaves into explicit face
geometry. Same-level faces produce one route. Coarse/fine interfaces are split
into fine subfaces, with one integrated flux scattered to both adjacent cells
with opposite signs. The resulting `DyadicFiniteVolumeDiscretization` uses the
existing unstructured explicit-face conservation runtime and boundary policies.

Tree lookup is not performed in a time-step kernel. Cell, face, quadrature, and
boundary routes are materialized once per accepted topology epoch.

## Differentiation

Spatial topology is discrete. Gradients are defined branchwise while codes,
leaf membership, active masks, and relation routes remain fixed. Coordinates,
particle payloads, and voxel values remain differentiable through realized
moment, interpolation, flux, and transfer calculations. Crossing a cell or
adaptation boundary begins a new topology epoch; it is not smoothed implicitly.

## Choosing a substrate

Use a dense tensor grid for dense regular stencils. Use a tensor index space
when logical structured addressing is required without a dense field. Use the
occupied-key cell list for uniform short-range particle interactions. Use LBVH
for dynamic primitive broad phase. Use a Morton point hierarchy for clustered
or long-range particles. Add primitive bounds for finite-support points such
as surfels. Use a sparse voxel grid for sparse fixed-resolution samples. Use a
sparse block topology for block-major fields on a virtual structured layout.
Use a dyadic topology when physical cell resolution and conservative
coarse/fine interfaces are part of the model. Use `RelationExecutionPlan` or
`route_reduce` to reduce route values that already exist. Use
`StreamedRelationPlan` when a nonlinear per-route function feeds an additive
receiver update and the per-route messages should not be materialized
graph-wide.

## Qualification tools

- `tools/spatial_hierarchy_benchmarks.py` records point-tree storage,
  preparation/evaluation timing, interaction counts, and direct-reference error.
- `tools/spatial_query_benchmarks.py` compares dense and Morton exact query
  timing, schedule storage, candidate evidence, and exact logical routes.
- `tools/sparse_voxel_benchmarks.py` records brick occupancy, topology storage,
  support coverage, and sampling timing for dense and narrow-band layouts.
- `tools/dyadic_amr_benchmarks.py` records adaptation and balance work,
  coarse/fine face lowering, finite-volume timing, and constant-state defect.
- `tools/surfel_substrate_benchmarks.py` records surfel realization, hierarchy,
  primitive-bound refit, and ray-query behavior.
- `tools/surfel_voxel_benchmarks.py` records bounded overlap routes, local
  implicit reconstruction, and plane error.
- `tools/sparse_execution_benchmarks.py` records active-key grouping,
  prepared relation reduction, the streamed relation runner against its
  materialized nonlinear reference, tile-major rasterization, and compact LBM,
  MPM, and FLIP execution.
- `benchmarks/streamed_relation_scaling.py` separates schedule preparation,
  lowering, compilation, first execution, and warm repeats. It covers forward,
  reverse, force-loss parameter-gradient, and coordinate-HVP transforms while
  varying edge count, degree skew, channel width, and tile capacities. It
  records XLA compiler byte estimates, sampled peaks (observations, not bounds),
  declared `StreamedRelationResources`, and capacity refusals. No result from it
  is a published performance claim.

Small systems should continue to use direct or dense authorities when the
measured crossover favors them.

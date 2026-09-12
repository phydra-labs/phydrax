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

These are execution substrates, not a new field language. Each domain retains
its own inactive-state, boundary, conservation, and topology-transition
semantics.

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

The canonical point hierarchy is not also the most compact execution layout
for every point-search kernel. `MortonNeighborQueryPlan` and
`MortonRadiusRelationPlan` lower the same validated Morton point order into a
private plane schedule. Leaves contain bounded contiguous Morton-order ranges;
each coarser plane groups contiguous children without allocating every occupied
octree prefix. Tight node bounds, parent/child ranges, logical permutations,
and required-versus-allocated resource evidence remain explicit.

The plane schedule has no independent physical identity. Algorithms return
logical source indices through `RowRelation`, exact radius routes through
`EdgeRelation`, or an owning domain result. Coincident points and points that
share the deepest integer Morton code are split into stable-ID execution tiles
with identical geometry. Their worst-case direct work is visible through
terminal-bucket and candidate evidence.

```python
query_plan = phx.discretization.spatial.MortonNeighborQueryPlan(
    address,
    source_capacity=source.shape[-2],
    target_capacity=target.shape[-2],
    maximum_neighbors=16,
    maximum_candidates=source.shape[-2],
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

`MortonPlaneInteractionPlan` performs deterministic dual-tree refinement over
the compact planes. A geometric opening policy accepts unequal-size node pairs
for far translation and opens the remainder until exact leaf completion. Every
ordered point pair is covered by exactly one accepted ancestor or one leaf
route. Queue, far-route, and near-route requirements are counted before their
fixed-capacity products are accepted. Node expansion radii are finite powers
of two that enclose their tight point bounds; scale exponent range and failures
remain visible in schedule evidence.


`PeriodicFoFFinderPlan` applies the same exact-link contract through explicit
`direct`, `cell_list`, or `morton_plane` realizations. Group slots are ordered
by each component's minimum stable particle ID. Group and link capacities,
topology completeness, convergence, finite arithmetic, and success are
reported independently; overflow never publishes a partial catalogue as
successful.

`MortonRadiusRelationPlan` separately controls pair capacity and open or closed
radius boundaries. Capacity exhaustion invalidates every truncated route and
reports the exact required pair count.

`DistributedMortonNeighborQueryPlan` shards sources, gathers target coordinates,
computes each shard's exact local top-k set, and globally merges those sets by
distance and stable ID. It returns globally indexed, target-sharded rows. This
portable authority communicates all targets and `shard_count × target_count ×
local_k` candidate summaries; it does not claim the communication complexity of
a fully Morton-repartitioned multi-host tree.

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
coarse/fine interfaces are part of the model.

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
  prepared relation reduction, tile-major rasterization, and compact LBM,
  MPM, and FLIP execution.

Small systems should continue to use direct or dense authorities when the
measured crossover favors them.

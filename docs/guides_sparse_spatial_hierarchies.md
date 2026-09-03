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

Use a dense tensor grid for dense regular stencils. Use a flat cell list for
uniform short-range particle interactions. Use LBVH for dynamic primitive broad
phase. Use a Morton point hierarchy for clustered or long-range particles. Add
primitive bounds for finite-support points such as surfels. Use a sparse voxel
grid for sparse fixed-resolution fields. Use a dyadic topology when physical
cell resolution and conservative coarse/fine interfaces are part of the model.

## Qualification tools

- `tools/spatial_hierarchy_benchmarks.py` records point-tree storage,
  preparation/evaluation timing, interaction counts, and direct-reference error.
- `tools/sparse_voxel_benchmarks.py` records brick occupancy, topology storage,
  support coverage, and sampling timing for dense and narrow-band layouts.
- `tools/dyadic_amr_benchmarks.py` records adaptation and balance work,
  coarse/fine face lowering, finite-volume timing, and constant-state defect.
- `tools/surfel_substrate_benchmarks.py` records surfel realization, hierarchy,
  primitive-bound refit, and ray-query behavior.
- `tools/surfel_voxel_benchmarks.py` records bounded overlap routes, local
  implicit reconstruction, and plane error.

Small systems should continue to use direct or dense authorities when the
measured crossover favors them.

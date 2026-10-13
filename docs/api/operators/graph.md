# Graph operators

Graph operators act on `DomainFunction`s over `GraphDomain` components. They
preserve Phydrax's usual residual path: an operator builds a residual
`DomainFunction`, and `ResidualPenalty` samples the relevant graph component
through an explicit integration source.

## Reductions and adjacency

::: phydrax.operators.graph_degree

---

::: phydrax.operators.neighbor_aggregate

---

::: phydrax.operators.graph_laplacian

## Graph calculus

::: phydrax.operators.graph_gradient

---

::: phydrax.operators.graph_divergence

---

::: phydrax.operators.graph_incidence_laplacian

## Physics residual builders

These helpers compose graph calculus operators into common residual forms. They
return ordinary `DomainFunction`s, so they can be used directly as operators in
`Residual` conditions.

::: phydrax.operators.graph_poisson_residual

---

::: phydrax.operators.graph_diffusion_residual

---

::: phydrax.operators.graph_conservation_residual

---

::: phydrax.operators.graph_advection_diffusion_residual

---

::: phydrax.operators.graph_heat_residual

---

::: phydrax.operators.graph_euler_residual

## Grounded-solve effective resistance

`EffectiveResistancePlan` prepares a canonical undirected
`phydrax.sparse.EdgeRelation` and a fixed list of query nodes. Valid routes
must have `source_indices < target_indices`; parallel conductors are allowed,
but self-loops and reversed routes are not. Conductances are dynamic,
differentiable values, one per route, and must be finite and strictly positive
on valid routes.

The plan grounds the minimum-index node of each connected component and solves
the reduced sparse Laplacian, without forming a dense pseudoinverse. The default
policy uses PCG with refreshable Jacobi preconditioning; callers can supply a
`phydrax.linalg.LinearSolvePolicy`. This is a **grounded-solve** route, not a
claim of exact arithmetic: retain the solve evidence when interpreting values
or gradients.

`plan.solve(conductances)` returns an `EffectiveResistanceResult` with a
`resistance` matrix of shape `(q, q)` in query-node order and a `connected`
mask. On `EffectiveResistanceStatus.SUCCESS`, pairs in different components
have resistance `+inf` and `connected=False`; disconnection is not a solve
failure. Invalid conductances produce `INVALID_CONDUCTANCE`, and failed solves
produce `SOLVE_FAILED`. Resistance values are `NaN` unless the overall status
is `SUCCESS`. The result also exposes `solve_status`, `residual_norm`,
`iterations`, and `derivative_regular`; check this evidence before using a
conductance sensitivity.

Preparation accepts `dtype` (default `jnp.float64`), solver-internal
`resources: SolveResourcePolicy | None`, and an outer `workspace_bytes` budget
(default 512 MiB). It bounds a logical outer-array inventory for both primal
execution and the implicit conductance derivative, including retained output.
Preparation chooses `plan.block_size` using the stricter inventory and refuses
the plan before allocating result arrays if even one RHS column cannot fit.
Native per-column solve workspace and Krylov constraints from
`LinearSolvePolicy.resources` can further reduce the block size; solver
internals remain under the native solve-cost contract, not this outer budget.

`plan.resource_estimate` is a `dict[str, int]` with explicit byte quantities:

| Key | Meaning |
| --- | --- |
| `output_bytes` | Floating-point result retained by the consumer: `q * q * dtype.itemsize` here, or `sketch_dimension * n * dtype.itemsize` for a sketch. |
| `primal_workspace_bytes` | Retained output, mapped query/embedding values and solve evidence, RHS/potential/query blocks, and sketch row assembly/weights when applicable. |
| `derivative_workspace_bytes` | Primal inventory plus output cotangent, all retained reduced solutions, query/potential cotangent and adjoint blocks, and accumulated/per-block conductance gradients; sketches also retain signs and RHS cotangents. |
| `budgeted_workspace_bytes` | `max(primal_workspace_bytes, derivative_workspace_bytes)`, always at most the supplied `workspace_bytes`. |

The remaining integer fields disaggregate these arrays. `rhs_bytes`,
`potential_bytes`, and `query_block_bytes` describe the current block;
`retained_query_bytes`, `retained_evidence_bytes`, `retained_potential_bytes`,
and (for sketches) `retained_sign_bytes` account for all mapped columns,
including the padded final block. `conductance_gradient_bytes` is one value
per relation-capacity slot; `edge_gradient_block_bytes` accounts for per-column
edge-gradient contributions. `output_bytes` does **not** include row assembly
or derivative storage. Immutable topology, caller inputs, native solver
internals, and downstream objectives are outside this outer inventory.

These are conservative logical array inventories, not a bound on physical
device peak memory: compiler/XLA temporaries, fusion, and buffer reuse are
outside the planning guarantee. Compiler temporary memory is measurable only
after lowering/compilation and is recorded separately for the primal and
derivative by the scaling benchmark. `plan.plan_id` identifies
relation indices/validity, query nodes, component roots, policy, dtype, and
block size, never conductance values. A changed budget changes the ID only if
it changes the selected block size.

::: phydrax.graph.EffectiveResistancePlan

---

::: phydrax.graph.EffectiveResistanceResult

---

::: phydrax.graph.EffectiveResistanceStatus

## Sketched effective resistance

`SketchedEffectiveResistancePlan` prepares the same canonical relation with an
explicit `sketch_dimension`. Its `.embed(conductances, key)` method requires a
typed key created by `jax.random.key`, not a legacy `jax.random.PRNGKey`.
Rademacher signs are addressed by sketch row using `fold_in`, so changing RHS
block size does not change the random sketch.

The returned `SketchedResistanceEmbedding` stores `embedding` with shape
`(sketch_dimension, n)` and `component_label`. Its
`.resistance(sources, targets)` method computes squared embedding distances
for node pairs, returning `+inf` across components on success and `NaN` on
failure. The embedding retains `status`, `solve_status`, `residual_norm`,
`iterations`, `derivative_regular`, and `sketch_dimension`. Approximation error
from random projection is separate from numerical linear-solve error.

The sketch plan accepts the same solve policy, dtype, solver resources, and
outer workspace budget as the grounded-solve plan. The retained embedding is
included in both logical workspace estimates; the stricter implicit-derivative
estimate determines blocking. It avoids a quadratic all-pairs output, but its
retained embedding still costs `sketch_dimension * n` floating-point values.
RHS/solution blocks have `n - component_count` coordinates per column, while
embedding blocks have `n` values per row (not `sketch_dimension`).
Choose the dimension explicitly; there is no automatic epsilon-to-dimension
planning API. See the
[worked resistance recipe](../../cookbook/graph_physics.md#effective-resistance-and-conductance-sensitivity)
for sensitivity, typed-key usage, and the qualitative projection guarantee.

::: phydrax.graph.SketchedEffectiveResistancePlan

---

::: phydrax.graph.SketchedResistanceEmbedding

## Randomized shortest paths and free energy

`RandomizedShortestPathPlan` prepares a directed `phydrax.sparse.EdgeRelation`
(route source to route target is one transition; include both orientations for
undirected graphs) and a fixed list of query nodes, used both as path sources and
as absorbing targets. Its `.solve(affinities, costs, inverse_temperature)`
method takes float64 per-route positive affinities, nonnegative path costs and a
positive scalar `theta`. Affinities define the reference random walk by
per-source max-shifted log normalization and a deterministic sum of exponentials,
so finite positive affinities never overflow a value-space row sum. Common
scaling of a source's affinities does not change the walk. Impassable transitions
are expressed through relation validity, never zero affinity. Self-loop
transitions are refused at construction.

For target `t`, the free energy of the Boltzmann bag of hitting paths from node
`i` is `phi_i = -(1/theta) log sum_paths prod p exp(-theta * cost)`, and
`expected_cost` is the expected cost of those paths. `free_energy[a, b]` and
`expected_cost[a, b]` describe paths from `nodes[a]` to `nodes[b]`; they are
`+inf` where `nodes[a]` cannot reach `nodes[b]` (`reachable` is `False`), zero on
the diagonal, and NaN in a failed target's column. `free_energy_distance` and
`expected_cost_dissimilarity` are the symmetrized `(M + Mᵀ) / 2` forms. As
`theta -> inf` the free energy tends to the shortest-path cost; as `theta -> 0`
the expected cost tends to the reference walk's expected hitting cost, and when
some paths escape into nodes that cannot reach the target, the free energy
diverges like `-log(hit probability) / theta`.

Partition functions span hundreds of decades across a graph at moderate
`theta`, so the plan never solves for them. It solves the soft Bellman equation
for `phi` itself with native Newton: the Jacobian `I - Pi` is the transient
generator of the optimal (Doob-transformed) walk. This avoids partition-function
dynamic range, not all ill-conditioning: near-closed transient classes can make
the generator nearly singular, particularly at small `theta`. Rows near unit
mass use a `log1p`/`expm1` form to reduce cancellation, but finite precision and
solve tolerances still limit accuracy. The retained accuracy/conditioning gate
samples path50, grid49, random40, and directed40 over `theta = 1e-9` through
`1e3`; that measured envelope is not a guarantee for every graph or temperature.
Successful values must be interpreted with the returned nonlinear/linear status
and residual evidence. Each target's Newton solve starts from the
pointwise minimum of two provable upper bounds: the single best augmented path
(exact as `theta -> inf`, computed with `phydrax.combinatorial.ShortestPathFieldPlan`)
and the Jensen bound `m - log(h) / theta` from the hitting probability and
conditional hitting cost (exact to first order as `theta -> 0`). Seeds carry no
derivative. One Newton correction at the converged stopped root carries the
first-order implicit-function derivative `-J⁻¹ dG` through the linear solve.
Only `linear_policy.differentiation.mode == "mathematical"` is admitted, because
operator and right-hand-side derivatives are both required. Free energies and
expected costs support first-order JVP, VJP, and gradients with respect to
affinities, costs, and `theta`; `d(theta * phi) / d theta` equals expected cost.
Plan and result expose immutable `maximum_derivative_order = 1` evidence without
additional numerical device leaves. Nested JVP/VJP/grad and Hessian transforms
raise `ValueError` with `DERIVATIVE_UNSUPPORTED` rather than returning incomplete
higher derivatives. The first-order contract is included in plan identity.

Construction prepares the Newton and walk-generator solve templates once on the
host, including symbolic factorization; `.solve` only refreshes their numerics.
The plan can therefore be closed over or passed as a traced argument to compiled
code (`eqx.filter_jit`) without host re-planning.

The default `linear_policy` is GMRES with a refreshable native sparse LU
preconditioner (approximate-minimum-degree ordering). Its factor storage is
charged exactly to `SolveResourcePolicy.preconditioner_bytes`, and it runs
unchanged on any JAX device. Native numeric factorization is sequential in the
number of pivots, so per-target cost grows quickly beyond roughly `10^4` nodes.
On CPU, larger graphs can pass
`LinearSolvePolicy(SparseLU(provider="jax-cpu"), resources=SolveResourcePolicy(factorization_bytes=...))`.
SuperLU chooses its ordering internally, so the planner charges a conservative
dense factor bound; the caller must raise that budget explicitly, and SuperLU's
actual memory is managed by the provider rather than bounded by Phydrax
planning. The result retains per-target `nonlinear_status`, `newton_iterations`,
`residual_norm`, and `linear_status`; `status` is `INVALID_INPUT` for
inadmissible values (no solve runs), `NONLINEAR_FAILED` or `LINEAR_SOLVE_FAILED`
when any target fails. `workspace_bytes` budgets retained outputs, reach masks,
and logical per-target derivative arrays; compiler temporaries and native solver
internals are outside that estimate. `seed_maximum_work` caps the total seed work
across all query targets (default `268_435_456`), and `seed_maximum_host_bytes`
caps aggregate host callback bytes across those targets (default `268_435_456`).
`resource_estimate` separates total `seed_work`, sequential callback peak
`seed_host_peak_bytes`, and cumulative allocation/traffic estimate
`seed_host_aggregate_bytes`; aggregate bytes are not peak live memory.
The device reference seed has no host callback bytes. These budgets are admitted
at construction before execution. `benchmarks/rsp_free_energy_gate.py` retains
the qualification record, including failed partition-function routes as
negative controls.

::: phydrax.graph.RandomizedShortestPathPlan

---

::: phydrax.graph.RandomizedShortestPathResult

---

::: phydrax.graph.RandomizedShortestPathStatus

## Exact geometric query neighborhoods

`query_neighbors` returns fixed-width `RowRelation` routes, minimum-image
relative vectors, squared distances, distances, counts, and
`QueryNeighborhoodEvidence`. With default `target_chunk_size=None` and
`source_chunk_size=None`, `query_workspace_bytes` (4 MiB) derives both tiles
from the case count, coordinate dimension, float64 pair vectors/distances,
mask/sort keys, and top-k merge state. Full-fit neighborhoods execute the
whole-axis parallel sort directly, without tile loops; larger problems use
the exact two-axis top-k merge. The default admits the quiet qualification's
256-point small query while bounding larger pair tables.
Positive, non-boolean integer chunks override the respective automatic tile;
both explicit chunks bypass budget admission. Evidence reports `backend="dense"`
or `"blocked"`, the executed source/target tile extents, and the workspace budget.
The budget excludes required target-by-k output and compiler scratch, and
clamps to at least one pair. Equal distances retain source-index order.
`batched_knn_query_graph` and `batched_knn_graph` expose and forward the same
policy. If `max_neighbors` is omitted without a plan, k equals the source count,
so required output can still be quadratic.
Supplying a `MortonNeighborQueryPlan` explicitly selects coarse Morton-cell
execution with a bounded candidate buffer per target while preserving logical
indices, source masks, target masks, stable equal-distance ordering, optional
self exclusion, and per-axis periodicity. Radius boundaries are inclusive.
The tile arguments are validated with a supplied plan but do not override its
candidate capacity or execution policy.

```python
address = phx.discretization.spatial.MortonAddressPlan(
    (0.0, 0.0),
    (1.0, 1.0),
    20,
    periodic_axes=(True, False),
)
plan = phx.discretization.spatial.MortonNeighborQueryPlan(
    address,
    source_capacity=source.shape[-2],
    target_capacity=target.shape[-2],
    maximum_neighbors=8,
)
neighborhood = phx.graph.query_neighbors(
    source,
    target,
    max_neighbors=8,
    periodic_lengths=(1.0, None),
    plan=plan,
)
```

Candidate capacity is a completeness bound, not an approximation knob. A
target whose visited cells overflow the buffer, or whose k-th distance cannot
be certified against the visited region, returns no routes and sets
`evidence.complete` and `evidence.successful` false.
The Pallas distance realization is explicit and retains a native JVP; the
portable JAX realization remains the default.

::: phydrax.graph.query_neighbors

---

::: phydrax.graph.QueryNeighborhood

---

::: phydrax.graph.QueryNeighborhoodEvidence

## GraphIR model blocks

These are executable `GraphIR -> GraphIR` blocks in `phydrax.graph`. They can be
used directly, placed inside process steppers, or exposed as `DomainFunction`s
with `GraphDomain.GraphModel(...)`, `GraphDatasetDomain.GraphModel(...)`, or
`GraphTrajectoryDatasetDomain.GraphModel(...)`. Autoregressive graph processes
can be exposed with `GraphDomain.GraphRolloutModel(...)` to supervise multi-step
predictions through the same term machinery. Graph model wrappers can install
node, edge, and graph-global `DomainFunction` inputs before executing the block,
which lets learned operators consume state fields, coefficient fields, and case
parameters through the same residual path. `GraphIR` topology
and query-graph geometry stored inside these blocks are fixed solver state;
learned arrays in the surrounding model remain trainable.

::: phydrax.graph.GraphKernelIntegral

---

::: phydrax.graph.GraphAttentionOperator

---

::: phydrax.graph.GraphNeuralOperator

---

::: phydrax.graph.GraphDiffusion

---

::: phydrax.graph.GraphFiniteVolumeDivergence

---

::: phydrax.graph.GraphFiniteVolumeDiffusion

---

::: phydrax.graph.GraphProcessor

---

::: phydrax.graph.RepeatedGraphProcessor

---

::: phydrax.domain.graph.GraphModel

## Graph process wrappers

Graph process wrappers turn graph vector fields into one-step or multi-step
operators. `GraphRolloutModel` keeps the sampled graph entity axis first and
returns rollout time as an unnamed trailing axis, so predicted graph trajectories
can be used directly in `ResidualPenalty` losses.

::: phydrax.graph.EulerGraphStepper

---

::: phydrax.graph.RK4GraphStepper

---

::: phydrax.graph.AutoregressiveGraphRollout

---

::: phydrax.domain.graph.GraphRolloutModel

## Equivariant graph operators

Euclidean graph operators read positions from mapping-valued graph nodes and
construct invariant edge features or equivariant vector updates. The equivariant
convolution writes scalar and vector node payloads, so either output can be
exposed with `GraphDomain.GraphModel(...)`.

::: phydrax.graph.euclidean_edge_features

---

::: phydrax.graph.gaussian_radial_basis

---

::: phydrax.graph.EquivariantGraphConvolution

## Typed and relational graph operators

Typed graph helpers read integer type ids from mapping-valued node or edge
payloads. Type components select heterogeneous graph subsets, while
`RelationalGraphConvolution` applies relation-specific message weights.

::: phydrax.graph.node_type_indices

---

::: phydrax.graph.edge_type_indices

---

::: phydrax.graph.typed_nodes_component

---

::: phydrax.graph.typed_edges_component

---

::: phydrax.graph.RelationalGraphConvolution

## Hypergraph operators

Hypergraphs are represented as typed bipartite `GraphIR` objects: original
entities and hyperedge entities are both graph nodes, and incidence relations are
typed graph edges.

::: phydrax.graph.hypergraph_to_bipartite_graph

---

::: phydrax.graph.incidence_to_bipartite_graph

---

::: phydrax.graph.HypergraphBipartiteGraph

---

::: phydrax.graph.HypergraphConvolution

## Topological graph operators

Simplicial complexes are represented as typed `GraphIR` objects: 0-cells,
1-cells, and 2-cells are graph nodes, while signed boundary/incidence maps are
typed graph edges. Hodge operators can then be used directly or wrapped with
`GraphDomain.GraphModel(...)` for physics residuals over vertex, edge, or face
cells.

::: phydrax.graph.triangle_mesh_to_simplicial_graph

---

::: phydrax.graph.SimplicialComplexGraph


### Metric cochain complexes and DEC

`CochainComplexIR` is the graph execution view of a canonical
`phydrax.discretization.CellComplexTopology` and `CochainDiscretization`. It packs
sparse signed incidences, primal and dual measures, diagonal Hodge stars, boundary
masks, cell coordinates, and an optional precomputed harmonic subspace into
`GraphIR`. This is a lowering of an admitted cochain realization, not a second
complex implementation. It accepts diagonal Hodges; sparse Gram realizations
execute through the exterior/linalg protocol instead.

Functional DEC operators and `GraphIR -> GraphIR` wrappers implement d, δ and
split/full Δ. `boundary="absolute"` or `boundary="relative"` selects the
active subcomplex. Canonical topology and reorientation live in discretization;
use `reorient_cochain(values, signs, cell_axis=...)` with an explicit cell axis.

`graph_to_cochain_complex` is the canonical graph-to-DEC bridge. Its explicit
`GraphEdgeSemantics` distinguishes reciprocal directed storage from an
undirected-once edge list, resolves parallel conductances deterministically, and
rejects ambiguous asymmetric or padded inputs. It creates a degree-one cochain
complex with the requested node probability measure.

::: phydrax.graph.CochainComplexIR

---

::: phydrax.exterior.ComplexBoundary

---

::: phydrax.linalg.harmonic_subspace

---

::: phydrax.discretization.reorient_cochain

---

::: phydrax.discretization.reorient_cell_complex

---

::: phydrax.graph.graph_to_cochain_complex

---

::: phydrax.graph.cochain_exterior_derivative

---

::: phydrax.graph.cochain_codifferential

---

::: phydrax.graph.cochain_hodge_laplacian

---

::: phydrax.graph.cochain_harmonic_projection

### Continuous differential forms to cochains

`DeRhamBridge` requires explicit oriented cell parameterizations and quadrature.
Coordinates alone are not sufficient integration data.
`validate_de_rham_commutation` compares integration of dα with d of its cochain.

::: phydrax.exterior.CellParameterization

::: phydrax.exterior.DeRhamBridge

::: phydrax.exterior.integrate_form

::: phydrax.exterior.validate_de_rham_commutation

### Typed cochain fields and domain-level DEC

`FormType` is the shared scientific contract for graph domains, neural operators
and residual programs; representation is declared separately. Domain-level DEC
operators adapt labeled `DomainFunction` carriers to the canonical calculus,
preserving degree, twist and realization identity.

::: phydrax.exterior.FormType

---

::: phydrax.operators.cochain_exterior_derivative

---

::: phydrax.operators.cochain_codifferential

---

::: phydrax.operators.cochain_hodge_laplacian

---

::: phydrax.operators.cochain_harmonic_projection

### Shared residual programs and metric reduction

`CochainResidualProgram` declares named input/output cochain schemas around one
full-complex residual callable. The same program can be bound to a
`phydrax.terms.CochainResidualTerm` for fixed-complex PINNs or operator training.
Its fingerprint includes the canonical callable identity and every field
semantic: StrictModule and plain module-level residual functions are identified
by content, while opaque callables (closures, lambdas, methods, partials) must
declare `residual_semantic_id` and `residual_numeric_id`; omitting them raises
`TypeError`.

`cochain_metric_reduce` first reduces each nonempty graph segment and then
averages segments. `graph_mean` is an arithmetic cell mean, `metric_mean` is a
Hodge-star-normalized mean, and `metric_sum` retains Hodge-star mass. Entity
masks exclude padding; optional segment weights compose graph-time quadrature.

::: phydrax.graph.CochainResidualProgram

---

::: phydrax.graph.cochain_metric_reduce

---

## Spectral graph and cochain operators

`phydrax.exterior.hodge_laplacian_eigenbasis` and `hodge_sector_spectra` operate
on realizations through their Hilbert complexes rather than graph-owned dense
algebra. Absolute/relative selection excludes inactive coordinates from spectra
and harmonic counts. Eigen and residual evidence comes from the linalg owner.

::: phydrax.exterior.HodgeSectorSpectra

::: phydrax.exterior.hodge_laplacian_eigenbasis

::: phydrax.exterior.hodge_sector_spectra

Sparse polynomial and Chebyshev filters provide the complementary scalable path
that applies a spectral graph operator without an eigendecomposition.

::: phydrax.graph.graph_adjacency_apply

---

::: phydrax.graph.graph_laplacian_apply

---

::: phydrax.graph.GraphLaplacianOperator

---

::: phydrax.graph.GraphPolynomialFilter

---

::: phydrax.graph.GraphChebyshevFilter

## Native fixed-topology message passing

Phydrax-native graph layers pass messages over `GraphIR.edge_relation()`, a
`phydrax.sparse.EdgeRelation` whose routes are the graph edges and whose
validity mask is `edge_mask`. Routes with `edge_mask=False` are inert in every
gather, reduction, softmax, and degree normalization, so padded or boundary
routes change neither node outputs nor gradients.

Layers whose per-route message feeds an additive receiver update run on a
prepared `phydrax.sparse.StreamedRelationPlan`. Each accepts an
`execution: StreamedRelationPlan | None` argument. Each route's callback is
evaluated once inside bounded tiles. Every receiver applies its normalization or
update once to its complete aggregate after its last route, and the per-route
messages are not materialized graph-wide:

- `EquivariantGraphConvolution` (normalization by complete incoming weight);
- `GraphKernelIntegral` with `reduction="sum"` or `"mean"` (count and measure
  normalization in the receiver epilogue);
- `MeshGraphNetBlock`/`MeshGraphNet` (all processor steps share one schedule).
  Updated edge latents are a requested graph-wide edge output and are charged
  as such; only the MLP hidden activations are bounded by the edge tile.

Their graph callbacks (`radial_fn`, `kernel_fn`) must be declared per route with
`RouteLocal`. A graph-wide callback is refused rather than evaluated per route.
See [Streamed nonlinear relations](../../guides_sparse_spatial_hierarchies.md#streamed-nonlinear-relations)
for schedule, evidence, and resource semantics.

The remaining native layers gather node payloads onto routes with
`gather_routes` and reduce messages onto targets with `route_reduce` (`"sum"`,
`"mean"`, `"max"`, or `"min"`). These include `GraphKernelIntegral` with
`reduction="max"` or `"min"`, `GraphNeuralOperator` (select the reduction with
`reduction=`), `GraphAttentionOperator` (its softmax keeps explicit
route-reduction ownership), `RelationalGraphConvolution`,
`HypergraphConvolution`, the DEC operators, cluster pooling, and the edge-index
layers `GCNConv`, `SAGEConv`, `GINConv`, and `MessagePassing` (`aggr="add"` is
the route `"sum"`; empty targets reduce to zero).

The jraph-compatible family (`GraphNetwork`, `InteractionNetwork`,
`RelationNetwork`, `DeepSets`, `GraphNetGAT`, `GraphConvolution`, and
`phydrax.graph.compat.jraph`) keeps pluggable aggregators with the jraph callback
contract `(data, segment_ids, num_segments)` and `segment_sum` defaults. One-off
index utilities such as `degree`, `coalesce`, and `Data` batching also keep
direct segment reductions: they build or count topology once rather than
repeatedly reducing over a prepared relation.

`facet_adjacency` bridges a prepared mesh discretization to the same relation.
An `UnstructuredFiniteVolumeDiscretization` contributes one owner-to-neighbor
route per face; boundary faces (neighbor sentinel `-1`) and inactive faces keep
their slot with safe index `0` and `valid=False`, so face payloads stay aligned
with routes. A `FiniteElementDiscretization` contributes its interior-facet
integration domain. `FacetAdjacency.topology_id` identifies the valid
`(facet, owner, neighbor)` triples on the named cell and facet entity sets, so
finite-volume and finite-element discretizations of one mesh share it.
`GraphIR.from_edge_relation` builds a graph whose `edge_relation()` reproduces
the relation and whose `edge_mask` is its validity mask, so a learned simulator
and the physical residual reduce over one topology:

```python
adjacency = phx.graph.facet_adjacency(finite_volume)
graph = phx.graph.GraphIR.from_edge_relation(
    adjacency.relation,
    nodes={"centers": finite_volume.cell_centers},
    edges={"flux": face_flux},
)
residual = phx.graph.GraphFiniteVolumeDivergence(normalize_by_volume=False)(graph)
```

::: phydrax.graph.facet_adjacency

---

::: phydrax.graph.FacetAdjacency

---

::: phydrax.graph.RouteLocal

## Learned graph simulator architectures

These blocks package common graph SciML model structure while preserving the
same `GraphIR -> GraphIR` surface as the lower-level operators.

::: phydrax.graph.RowMLP

---

::: phydrax.graph.MeshGraphNetBlock

---

::: phydrax.graph.MeshGraphNet

## Multiscale graph operators

Cluster pooling and multiscale blocks expose a coarse-graph path for long-range
interactions and hierarchy-aware graph neural operators.

::: phydrax.graph.pool_graph_by_cluster

---

::: phydrax.graph.unpool_nodes_by_cluster

---

::: phydrax.graph.GraphClusterPool

---

::: phydrax.graph.GraphMultiscaleBlock

## Mesh calculus

Mesh-calculus helpers build geometry-aware `GraphIR` objects and executable
cotangent operators for triangular surface meshes. `MeshCotangentLaplacian`
requires an explicit sign: `"neighbor_minus_self"` approximates the
negative-semidefinite differential Laplacian \(\Delta\), while
`"self_minus_neighbor"` is the positive-semidefinite stiffness convention
\(-\Delta\).

::: phydrax.graph.mesh_to_cotangent_graph

---

::: phydrax.graph.mesh_cotangent_weights

---

::: phydrax.graph.mesh_lumped_vertex_areas

---

::: phydrax.graph.mesh_face_areas

---

::: phydrax.graph.mesh_face_normals

---

::: phydrax.graph.mesh_vertex_normals

---

::: phydrax.graph.MeshCotangentLaplacian

## Derived graph structures

Derived graph structures convert one topology into another while preserving the
`GraphIR` execution contract. Line graphs support edge/flux dynamics, and mesh
dual graphs support face- or cell-centered finite-volume operators.

Line-graph transitions are joined through endpoint-indexed edge lists rather
than testing every edge pair. Transitions remain ordered by
`(source_edge_index, target_edge_index)`; shared-node connectivity records the
smallest common endpoint, including parallel edges and self-loops. High-valence
graphs may still require quadratic result storage because those transitions
are part of the requested graph.

::: phydrax.graph.line_graph

---

::: phydrax.graph.LineGraph

---

::: phydrax.graph.mesh_to_dual_graph

---

::: phydrax.graph.MeshDualGraph

## Geometry graph construction

Host radius and kNN builders enumerate candidates in source tiles of 256
and retain only result indices and the current top-k. Bipartite outputs retain
target-major ordering (radius source-index order; kNN distance/index order);
homogeneous point-cloud outputs retain lexicographic sender/receiver ordering.
Radius inclusion and periodic minimum-image semantics are unchanged.

::: phydrax.graph.radius_graph

---

::: phydrax.graph.knn_graph

---

::: phydrax.graph.radius_query_graph

---

::: phydrax.graph.knn_query_graph

---

::: phydrax.graph.query_graph_from_edges

---

::: phydrax.graph.mollified_kernel_weight

---

::: phydrax.graph.QueryGraph

## Multi-graph transfer operators

Query-graph transfer operators move fields from one graph topology to another
through a fixed source-to-target query graph. This is the encode/decode bridge
for graph neural operator and multi-resolution graph pipelines.

::: phydrax.graph.query_graph_with_source_features

---

::: phydrax.graph.query_target_features

---

::: phydrax.graph.QueryGraphOperator

---

::: phydrax.graph.query_encode_process_decode

---

::: phydrax.graph.GraphEncodeProcessDecode

---

::: phydrax.graph.mesh_to_geometry_graph

---

::: phydrax.graph.point_cloud_to_graph

---

::: phydrax.graph.GeometryGraph

## Matrix gauge links and ordered holonomy

::: phydrax.graph.MatrixGaugeLinkSpace

::: phydrax.graph.gauge_transform_links

::: phydrax.discretization.ordered_path_transport

::: phydrax.graph.closed_path_trace

Non-Abelian paths use
`phydrax.discretization.OrientedEdgePathPlan`; ordered cell boundaries use
`CellBoundaryPathPlan`. Incidence alone is never treated as multiplication
order. Matrix gauge-link spaces reference the canonical edge
`DiscreteFieldSpace` and pointwise Lie-group geometry rather than introducing
another topology or field owner.

## Weighted graph partitioning

`partition_graph` splits a `WeightedCSRGraph` (symmetric, nonnegative integer
vertex and edge weights, canonical sorted rows) into `GraphPartitionPlan.part_count`
parts with the native deterministic multilevel k-way route: heavy-edge matching
coarsening, recursive greedy-growing bisection of the coarsest graph, and
boundary Fiduccia-Mattheyses refinement in which no move exceeds a part
capacity. Part `p` targets `W * share_p` of the total vertex weight `W`; its
capacity is `max(floor(maximum_imbalance * t_p), ceil(t_p))`. The default
`require_nonempty` policy gives every part a vertex. `GraphPartitionEvidence`
is measured from the returned parts: weighted edge cut, cut-edge and boundary
vertex counts, part weights against targets and capacities, the imbalance and
its lower bound, indivisible heavy vertices, a determinism statement, the
backend identity and `GraphPartitionWork` counters. An unbalanced outcome is a
status (`indivisible_vertex_overload` or `balance_not_reached`), never a
silently relaxed capacity, and the cut is not claimed optimal. A `work_limit`
bounds adjacency visits; exhausting it is a resource refusal.
`MeshPartitionKind.GRAPH` distributes meshes through this route.

::: phydrax.graph.WeightedCSRGraph

---

::: phydrax.graph.GraphPartitionPlan

---

::: phydrax.graph.partition_graph

---

::: phydrax.graph.GraphPartitionResult

---

::: phydrax.graph.GraphPartitionEvidence

---

::: phydrax.graph.GraphPartitionWork

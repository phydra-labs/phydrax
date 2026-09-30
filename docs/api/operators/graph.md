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

## Exact geometric query neighborhoods

`query_neighbors` returns fixed-width `RowRelation` routes, minimum-image
relative vectors, squared distances, distances, counts, and
`QueryNeighborhoodEvidence`. The default implementation is the dense exact
authority. Supplying a `MortonNeighborQueryPlan` selects coarse Morton-cell
execution with a bounded candidate buffer per target while preserving logical
indices, source masks, target masks, stable equal-distance ordering, optional
self exclusion, and per-axis periodicity.

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
validity mask is `edge_mask`. Node payloads are gathered onto routes with
`gather_routes` and messages are reduced onto targets with `route_reduce`
(`"sum"`, `"mean"`, `"max"`, or `"min"`). Routes with `edge_mask=False` are
inert in every gather, reduction, softmax, and degree normalization, so padded or
boundary routes change neither node outputs nor gradients. This contract covers
`MeshGraphNet`, `GraphAttentionOperator`, `GraphKernelIntegral` and
`GraphNeuralOperator` (select the reduction with `reduction=`),
`EquivariantGraphConvolution`, `RelationalGraphConvolution`,
`HypergraphConvolution`, the DEC operators, cluster
pooling, and the edge-index layers `GCNConv`, `SAGEConv`, `GINConv`, and
`MessagePassing` (`aggr="add"` is the route `"sum"`; empty targets reduce to
zero).

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

::: phydrax.graph.line_graph

---

::: phydrax.graph.LineGraph

---

::: phydrax.graph.mesh_to_dual_graph

---

::: phydrax.graph.MeshDualGraph

## Geometry graph construction

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

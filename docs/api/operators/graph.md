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

---

::: phydrax.graph.SimplicialHodgeLaplacian

### Metric cochain complexes and DEC

`CochainComplexIR` stores a canonical oriented cell complex: sparse signed
incidences, primal and dual measures, diagonal Hodge stars, boundary masks,
cell coordinates, and an optional precomputed harmonic subspace. Constructors
validate chain-complex identities and positive metric data before the object can
reach compiled execution.

The functional DEC operators and their `GraphIR -> GraphIR` wrappers implement
the exterior derivative, metric codifferential, split/full Hodge Laplacian, and
metric harmonic projection. `triangle_mesh_to_cochain_complex` always builds
the complete oriented topology; boundary behavior belongs to the consuming
operator. `CochainBoundaryPolicy("absolute" | "relative")` selects that
operator's active subcomplex. `reorient_cochain_complex` changes the oriented
cell basis without changing the represented physical complex; use
`reorient_cochain` to transform signed coefficient arrays consistently.

::: phydrax.graph.CochainComplexIR

---

::: phydrax.graph.CochainBoundaryPolicy

---

::: phydrax.graph.triangle_mesh_to_cochain_complex

---

::: phydrax.graph.compute_harmonic_subspace

---

::: phydrax.graph.reorient_cochain

---

::: phydrax.graph.reorient_cochain_complex

---

::: phydrax.graph.cochain_exterior_derivative

---

::: phydrax.graph.cochain_codifferential

---

::: phydrax.graph.cochain_hodge_laplacian

---

::: phydrax.graph.cochain_harmonic_projection

---

### Typed cochain fields and domain-level DEC

`CochainFieldSpec` is the shared semantic contract used by graph domains,
cochain neural operators, and residual programs. The domain-level DEC functions
accept a declared cochain `DomainFunction`, preserve or change its degree as
mathematically required, and return another `DomainFunction`. They execute the
same sparse kernels as the array-level functions above.

::: phydrax.graph.CochainFieldSpec

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
Its fingerprint includes the explicit callable identity and every field
semantic.

`cochain_metric_reduce` first reduces each nonempty graph segment and then
averages segments. `graph_mean` is an arithmetic cell mean, `metric_mean` is a
Hodge-star-normalized mean, and `metric_sum` retains Hodge-star mass. Entity
masks exclude padding; optional segment weights compose graph-time quadrature.

::: phydrax.graph.CochainResidualProgram

---

::: phydrax.graph.cochain_metric_reduce

---

## Spectral graph operators

Sparse polynomial and Chebyshev filters provide a scalable spectral graph
operator path without dense eigendecompositions.

::: phydrax.graph.graph_adjacency_apply

---

::: phydrax.graph.graph_laplacian_apply

---

::: phydrax.graph.GraphLaplacianOperator

---

::: phydrax.graph.GraphPolynomialFilter

---

::: phydrax.graph.GraphChebyshevFilter

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

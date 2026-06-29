# Physics-informed graph residuals

This recipe shows the graph-domain version of the standard Phydrax pattern:
define fields on a domain, compose operators into a residual, then turn that
residual into a constraint term.

Graphs are useful when the physical support is an irregular mesh, particle
system, point cloud, circuit, molecular graph, or any finite topology where
continuous coordinates are not the only natural indexing structure.

## Graph domain

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        edges=jnp.array([[2.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )

    domain = phx.domain.GraphDomain(graph, measure="count")
    structure = phx.domain.ProductStructure((("graph",),))
    n_nodes = graph.num_nodes
    n_edges = graph.num_edges
    n_boundary_nodes = 2

    nodes = domain.component({"graph": phx.domain.Nodes()})
    edges = domain.component({"graph": phx.domain.Edges()})
    boundary_nodes = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def edge_weight(edge):
        return edge[0]

    node_batch = nodes.sample(3, structure=structure)
    edge_batch = edges.sample(2, structure=structure)
    boundary_batch = boundary_nodes.sample(n_boundary_nodes, structure=structure)

    grad_u = phx.operators.graph_gradient(u)
    weighted_grad_u = phx.operators.graph_gradient(u, weight=edge_weight)
    lap_u = phx.operators.graph_incidence_laplacian(u)

    assert jnp.allclose(grad_u(edge_batch).data, jnp.array([1.0, 2.0]))
    assert jnp.allclose(weighted_grad_u(edge_batch).data, jnp.array([2.0, 6.0]))
    assert jnp.allclose(lap_u(node_batch).data, jnp.array([-1.0, -1.0, 2.0]))
    assert jnp.allclose(lap_u(boundary_batch).data, jnp.array([-1.0, 2.0]))

    diffusion = phx.constraints.FunctionalConstraint.from_operator(
        component=nodes,
        operator=phx.operators.graph_incidence_laplacian,
        constraint_vars="u",
        num_points=n_nodes,
        structure=structure,
    )

    constant_gradient = phx.constraints.FunctionalConstraint.from_operator(
        component=edges,
        operator=phx.operators.graph_gradient,
        constraint_vars="u",
        num_points=n_edges,
        structure=structure,
    )

    boundary_diffusion = phx.constraints.FunctionalConstraint.from_operator(
        component=boundary_nodes,
        operator=phx.operators.graph_incidence_laplacian,
        constraint_vars="u",
        num_points=n_boundary_nodes,
        structure=structure,
    )

    @domain.Function("graph")
    def constant_u(node):
        del node
        return 1.0

    assert diffusion.loss({"u": constant_u}) < 1e-12
    assert constant_gradient.loss({"u": constant_u}) < 1e-12
    assert boundary_diffusion.loss({"u": constant_u}) < 1e-12
    ```

## What this enables

The graph residual above is a finite-topology analogue of
\(\nabla\cdot(k\nabla u)\). A learned graph model can be exposed as a
`DomainFunction` with `GraphDomain.GraphModel(...)`, then constrained with the
same `FunctionalConstraint` and `FunctionalSolver` machinery used for continuous
PDEs.

Use `graph_gradient` for node-to-edge differences, `graph_divergence` for
edge-to-node conservation terms, and `graph_incidence_laplacian` for the
incidence-form `div(grad(u))`.

## Hard graph value constraints

Use `enforce_graph_values` when a graph node, edge, or graph-global subset should
satisfy a value constraint by construction instead of through a penalty term.
The result is still an ordinary `DomainFunction`, so downstream graph residuals
and constraints see the enforced values.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.ProductStructure((("graph",),))
    nodes = domain.component({"graph": phx.domain.Nodes()})
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    boundary_count = 2

    @domain.Function("graph")
    def u(node):
        return node[0]

    hard_u = phx.constraints.enforce_graph_values(u, boundary, target=5.0)
    batch = nodes.sample(graph.num_nodes, structure=structure)
    assert jnp.allclose(hard_u(batch).data, jnp.array([5.0, 1.0, 5.0]))

    bc = phx.constraints.FunctionalConstraint.from_operator(
        component=boundary,
        operator=lambda f: f - 5.0,
        constraint_vars="u",
        num_points=boundary_count,
        structure=structure,
    )
    assert bc.loss({"u": hard_u}) < 1e-12

    term = phx.solver.SingleFieldEnforcedConstraint(
        "u",
        boundary,
        lambda f: phx.constraints.enforce_graph_values(f, boundary, target=5.0),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": u},
        constraints=(),
        constraint_terms=[term],
    )
    solver_u = solver.ansatz_functions().get("u")
    assert jnp.allclose(solver_u(batch).data, jnp.array([5.0, 1.0, 5.0]))
    ```

## Named residual builders

Common graph physics residuals are available as named operator compositions.
They return normal `DomainFunction`s and are meant to be used with
`FunctionalConstraint.from_operator(...)`.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.ProductStructure((("graph",),))
    node_count = graph.num_nodes
    nodes = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def constant_u(node):
        del node
        return 1.0

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=nodes,
        operator=phx.operators.graph_poisson_residual,
        constraint_vars="u",
        num_points=node_count,
        structure=structure,
    )
    assert constraint.loss({"u": constant_u}) < 1e-12
    ```

## Graph neural operator blocks

Executable `GraphIR -> GraphIR` blocks can be wrapped as `DomainFunction`s. This
keeps learned graph operators and physics-encoded operators inside the same
constraint path.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        edges=jnp.array([[2.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.ProductStructure((("graph",),))
    nodes = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def u(node):
        del node
        return 1.0

    def residual(f):
        return domain.GraphModel(phx.graph.GraphDiffusion(), input_fn=f)

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=nodes,
        operator=residual,
        constraint_vars="u",
        num_points=graph.num_nodes,
        structure=structure,
    )

    assert constraint.loss({"u": u}) < 1e-12
    ```

## Edge-biased graph attention

Sparse attention can be used as a long-range graph neural operator while still
returning an ordinary `GraphIR`. Edge payloads can bias attention logits, and
the result can be exposed as a `DomainFunction` with `GraphDomain.GraphModel`.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes={"features": jnp.array([[1.0], [3.0], [0.0]])},
        edges={"bias": jnp.array([0.0, jnp.log(3.0)])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    target_nodes = domain.component({"graph": phx.domain.NodeSet([2])})
    target_count = 1
    target_batch = target_nodes.sample(
        target_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @domain.Function("graph")
    def u(node):
        return node.get("features")[0]

    attention = domain.GraphModel(
        phx.graph.GraphAttentionOperator(
            logit_fn=lambda edges, sent, recv, globals_: jnp.zeros((sent.shape[0],)),
            input_key="u",
            output_key="attn",
            edge_bias_key="bias",
        ),
        input_fn=u,
        input_key="u",
        output_key="attn",
    )

    assert jnp.allclose(jnp.ravel(attention(target_batch).data), jnp.array([2.5]))
    ```

## Query graphs and graph neural operators

Graph neural operators often evaluate a source field at separate target/query
points. A `QueryGraph` represents this as a typed bipartite `GraphIR`; source
points and target points are graph nodes, and radius or kNN query edges carry
relative coordinates, distances, and optional mollified kernel weights.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    domain = phx.domain.GraphDomain(bundle.graph)
    target_count = 1
    target_cells = domain.component({"graph": bundle.target_nodes_component()})
    target_batch = target_cells.sample(
        target_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @domain.Function("graph")
    def u(point):
        return point.get("features")[0]

    gno = domain.GraphModel(
        phx.graph.GraphNeuralOperator(
            input_key="u",
            output_key="gno",
            edge_weight_key=None,
            normalize=False,
            target_node_type=bundle.target_type,
        ),
        input_fn=u,
        input_key="u",
        output_key="gno",
    )

    assert jnp.allclose(jnp.ravel(gno(target_batch).data), jnp.array([4.0]))
    ```

## Multi-graph transfer

A fixed `QueryGraph` can serve as an operator bridge between a source graph and
a separate target topology. The transfer result is itself a `GraphIR`, so target
nodes can be sampled by `GraphDomain` for losses and downstream operators.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    source_graph = phx.graph.GraphIR(
        nodes={
            "positions": jnp.array([[0.0], [1.0]]),
            "features": jnp.array([[1.0], [3.0]]),
        },
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    query = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        weight_kind=None,
    )
    transfer = phx.graph.QueryGraphOperator(
        query,
        source_key="features",
        input_key="u",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )
    query_out = transfer(source_graph)
    assert jnp.allclose(
        jnp.ravel(phx.graph.query_target_features(query_out, query, "out")),
        jnp.array([4.0]),
    )

    target_domain = phx.domain.GraphDomain(query_out)
    target_nodes = target_domain.component({"graph": query.target_nodes_component()})
    target_count = 1
    target_batch = target_nodes.sample(
        target_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @target_domain.Function("graph")
    def prediction(point):
        return point.get("out")[0]

    assert jnp.allclose(prediction(target_batch).data, jnp.array([4.0]))
    ```

## Encode-process-decode query operators

Source-to-latent and latent-to-target query graphs can be composed into a
GNO-style encode-process-decode operator. The latent processor is any
`GraphIR -> GraphIR` block.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    source_graph = phx.graph.GraphIR(
        nodes={"features": jnp.array([[1.0], [3.0]])},
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    encoder_query = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        weight_kind=None,
    )
    decoder_query = phx.graph.radius_query_graph(
        jnp.array([[0.5]]),
        jnp.array([[0.25]]),
        radius=1.0,
        weight_kind=None,
    )
    pipeline = phx.graph.query_encode_process_decode(
        encoder_query,
        decoder_query,
        source_key="features",
        latent_key="latent",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )
    decoded = pipeline(source_graph)
    assert jnp.allclose(
        jnp.ravel(phx.graph.query_target_features(decoded, decoder_query, "out")),
        jnp.array([4.0]),
    )
    ```

## Derived graph structures

Line graphs turn original edges into graph nodes, which is useful for flux- or
edge-centered dynamics. Mesh dual graphs turn triangular faces into graph nodes,
which is useful for cell-centered finite-volume workflows.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    primal = phx.graph.GraphIR(
        edges={"features": jnp.array([[1.0], [3.0]])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    edge_graph = phx.graph.line_graph(primal)
    edge_domain = phx.domain.GraphDomain(edge_graph.graph)
    edge_nodes = edge_domain.component({"graph": edge_graph.original_edges_component()})
    edge_count = edge_graph.graph.num_nodes
    edge_batch = edge_nodes.sample(
        edge_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @edge_domain.Function("graph")
    def flux_state(edge_node):
        return edge_node.get("features")[0]

    edge_diffusion = edge_domain.GraphModel(
        phx.graph.GraphDiffusion(),
        input_fn=flux_state,
    )
    assert jnp.allclose(edge_diffusion(edge_batch).data, jnp.array([-2.0, 2.0]))

    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    dual = phx.graph.mesh_to_dual_graph(vertices, faces)
    assert dual.graph.num_nodes == 2
    assert dual.graph.num_edges == 2
    ```

## Finite-volume graph operators

Face- or cell-centered dual graphs can use conservative finite-volume blocks.
`GraphFiniteVolumeDivergence` maps edge fluxes to cells, while
`GraphFiniteVolumeDiffusion` constructs diffusive edge fluxes from cell values,
edge distances, and cell areas.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    dual = phx.graph.mesh_to_dual_graph(vertices, faces)
    dual_domain = phx.domain.GraphDomain(dual.graph)
    face_cells = dual_domain.component({"graph": dual.face_nodes_component()})
    face_count = dual.graph.num_nodes
    face_batch = face_cells.sample(
        face_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )
    face_values = jnp.array([1.0, 3.0])

    @dual_domain.Function("graph")
    def u(face):
        return face_values[face.get("face_index")]

    diffusion = dual_domain.GraphModel(
        phx.graph.GraphFiniteVolumeDiffusion(input_key="u", output_key="du"),
        input_fn=u,
        input_key="u",
        output_key="du",
    )
    du = diffusion(face_batch).data
    assert du.shape == (face_count, 1)
    assert jnp.allclose(jnp.sum(du[:, 0] * dual.graph.nodes.get("area")[:, 0]), 0.0)
    ```

## Graph model side inputs

Graph models can evaluate node, edge, and graph-global `DomainFunction` inputs
before running the `GraphIR -> GraphIR` block. This is the common path for
coefficient fields, metric terms, boundary tags, or case parameters used by a
physics-informed graph operator.

Graph topology and geometry carried by `GraphIR` values, including query graphs
stored inside graph operators, are treated as fixed solver state. Trainable
Equinox/JAX arrays in the surrounding graph model remain optimizer parameters.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    class WeightedDiffusion:
        def __call__(self, graph):
            nodes = dict(graph.nodes)
            u = nodes["u"]
            k = graph.edges["k"]
            scale = jnp.squeeze(graph.globals["scale"])
            flux = scale * k * (u[graph.receivers] - u[graph.senders])
            incoming = phx.graph.segment_sum(flux, graph.receivers, graph.num_nodes)
            outgoing = phx.graph.segment_sum(flux, graph.senders, graph.num_nodes)
            nodes["residual"] = incoming - outgoing
            return graph.replace(nodes=nodes, validate=False)

    side_input_graph = phx.graph.GraphIR(
        nodes={"x": jnp.array([[0.0], [1.0], [2.0]])},
        edges={"coefficient": jnp.array([2.0, 4.0])},
        globals={"case_scale": jnp.array([0.5])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    side_input_domain = phx.domain.GraphDomain(side_input_graph)
    side_input_nodes = side_input_domain.component({"graph": phx.domain.Nodes()})
    side_input_structure = phx.domain.ProductStructure((("graph",),))

    @side_input_domain.Function("graph")
    def side_input_u(node):
        del node
        return 1.0

    @side_input_domain.Function("graph")
    def side_input_k(edge):
        return edge["coefficient"]

    @side_input_domain.Function("graph")
    def side_input_scale(case):
        return case["case_scale"]

    def side_input_residual(f):
        return side_input_domain.GraphModel(
            WeightedDiffusion(),
            input_fn=f,
            input_key="u",
            edge_input_fn=side_input_k,
            edge_input_key="k",
            global_input_fn=side_input_scale,
            global_input_key="scale",
            output_key="residual",
        )

    side_input_constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=side_input_nodes,
        operator=side_input_residual,
        constraint_vars="u",
        num_points=side_input_graph.num_nodes,
        structure=side_input_structure,
    )
    assert side_input_constraint.loss({"u": side_input_u}) < 1e-12
    ```

## Equivariant graph operators

When a graph carries Euclidean positions, scalar fields can generate
translation-invariant scalar outputs and rotation-equivariant vector outputs.
This is useful for force-like, flux-like, and geometry-aware simulator terms.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes={
            "positions": jnp.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            "features": jnp.array([[1.0], [2.0], [3.0]]),
        },
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )
    graph = phx.graph.euclidean_edge_features(graph)
    domain = phx.domain.GraphDomain(graph)
    nodes = domain.component({"graph": phx.domain.Nodes()})
    node_count = graph.num_nodes
    batch = nodes.sample(
        node_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @domain.Function("graph")
    def u(node):
        return node.get("features")[0]

    vector_model = domain.GraphModel(
        phx.graph.EquivariantGraphConvolution(
            input_key="u",
            scalar_output_key="scalar",
            vector_output_key="vector",
        ),
        input_fn=u,
        input_key="u",
        output_key="vector",
    )

    vector_field = vector_model(batch)
    assert vector_field.data.shape == (node_count, 2, 1)
    ```

## Spectral graph filters

Polynomial and Chebyshev filters apply sparse powers of graph adjacency or
Laplacian operators. They are useful when a graph operator should behave like a
spectral neural operator while remaining local, sparse, and batchable.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [3.0]]),
        edges={"weight": jnp.array([1.0, 1.0])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    nodes = domain.component({"graph": phx.domain.Nodes()})
    batch = nodes.sample(2, structure=phx.domain.ProductStructure((("graph",),)))

    @domain.Function("graph")
    def u(node):
        return node[0]

    lap = phx.graph.GraphLaplacianOperator(
        weight_key="weight",
        normalization="none",
    )
    assert jnp.allclose(lap(graph).nodes[:, 0], jnp.array([-2.0, 2.0]))

    spectral = domain.GraphModel(
        phx.graph.GraphChebyshevFilter(jnp.array([1.0]), weight_key="weight"),
        input_fn=u,
    )
    assert jnp.allclose(spectral(batch).data, jnp.array([1.0, 3.0]))
    ```

## Typed and relational graphs

Heterogeneous graphs store integer type ids as ordinary node or edge payloads.
Type-based components select subsets for constraints, and relation-specific
operators can use edge types for typed message passing.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes={
            "features": jnp.array([[1.0], [2.0], [3.0]]),
            "type": jnp.array([0, 1, 0], dtype=jnp.int32),
        },
        edges={
            "features": jnp.array([[0.5], [1.5], [2.5]]),
            "type": jnp.array([0, 1, 0], dtype=jnp.int32),
        },
        senders=jnp.array([0, 2, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 1, 0], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph, measure="count")
    structure = phx.domain.ProductStructure((("graph",),))

    typed_nodes = domain.component({"graph": phx.domain.NodeType(1)})
    typed_batch = typed_nodes.sample(1, structure=structure)
    typed_payload = typed_batch.points.get("graph")
    assert jnp.allclose(jnp.ravel(typed_payload.get("features").data), jnp.array([2.0]))

    conv = phx.graph.RelationalGraphConvolution(
        jnp.array([10.0, 100.0]),
        input_key="features",
        output_key="updated",
    )
    out = conv(graph)
    assert jnp.allclose(jnp.ravel(out.nodes.get("updated")), jnp.array([20.0, 310.0, 0.0]))
    ```

## Hypergraphs as bipartite graphs

Higher-order interactions can be represented by turning each hyperedge into an
auxiliary graph node. The result is a typed bipartite `GraphIR`, so graph-domain
components, graph models, and constraints remain unchanged.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    bundle = phx.graph.hypergraph_to_bipartite_graph(
        ([0, 1], [1, 2]),
        node_features=jnp.array([[1.0], [2.0], [3.0]]),
    )
    domain = phx.domain.GraphDomain(bundle.graph)
    original_nodes = domain.component({"graph": bundle.original_nodes_component()})
    batch = original_nodes.sample(
        3,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @domain.Function("graph")
    def u(node):
        return node.get("features")[0]

    model = domain.GraphModel(
        phx.graph.HypergraphConvolution(input_key="u", output_key="u_next"),
        input_fn=u,
        input_key="u",
        output_key="u_next",
    )

    assert jnp.allclose(jnp.ravel(model(batch).data), jnp.array([1.5, 2.0, 2.5]))
    ```

## Simplicial complexes and Hodge operators

Triangular meshes can also be lifted into a signed simplicial complex. Vertices,
edge cells, and face cells become typed graph nodes, and boundary/incidence
maps become typed graph edges. This exposes discrete exterior-calculus style
operators through the same graph-model and constraint path.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    bundle = phx.graph.triangle_mesh_to_simplicial_graph(
        jnp.array([[0, 1, 2]], dtype=jnp.int32)
    )
    domain = phx.domain.GraphDomain(bundle.graph)
    structure = phx.domain.ProductStructure((("graph",),))
    vertex_cells = domain.component({"graph": bundle.vertex_cells_component()})
    vertex_count = 3
    vertex_batch = vertex_cells.sample(vertex_count, structure=structure)
    vertex_values = jnp.array([0.0, 1.0, 0.0])

    @domain.Function("graph")
    def u(cell):
        return jnp.where(
            cell.get("cell_dim") == 0,
            vertex_values[cell.get("local_index")],
            0.0,
        )

    hodge_l0 = domain.GraphModel(
        phx.graph.SimplicialHodgeLaplacian(0, input_key="u", output_key="lap_u"),
        input_fn=u,
        input_key="u",
        output_key="lap_u",
    )
    assert jnp.allclose(hodge_l0(vertex_batch).data, jnp.array([-1.0, 2.0, -1.0]))

    @domain.Function("graph")
    def constant_u(cell):
        del cell
        return 1.0

    def residual(f):
        return domain.GraphModel(
            phx.graph.SimplicialHodgeLaplacian(0, input_key="u", output_key="lap_u"),
            input_fn=f,
            input_key="u",
            output_key="lap_u",
        )

    hodge_constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=vertex_cells,
        operator=residual,
        constraint_vars="u",
        num_points=vertex_count,
        structure=structure,
    )
    assert hodge_constraint.loss({"u": constant_u}) < 1e-12
    ```

## MeshGraphNet and multiscale blocks

MeshGraphNet-style encoder-process-decoder models and coarse-graph blocks use
the same graph-model wrapper. This keeps learned simulators, hierarchy-aware
operators, and physics residuals on one execution path.

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        edges=jnp.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, -1.0, 1.0],
            ]
        ),
        senders=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 3, 0], dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([4], dtype=jnp.int32),
    )

    model = phx.graph.MeshGraphNet(
        node_in_size=2,
        edge_in_size=3,
        node_out_size=1,
        latent_size=8,
        hidden_size=8,
        processor_steps=2,
        key=jr.key(0),
    )
    out = model(graph)
    assert out.nodes.shape == (4, 1)

    coarse = phx.graph.pool_graph_by_cluster(
        graph,
        jnp.array([0, 0, 1, 1], dtype=jnp.int32),
    )
    assert coarse.nodes.shape == (2, 2)

    def coarse_shift(g):
        return g.replace(nodes=g.nodes + 1.0, validate=False)

    multiscale = phx.graph.GraphMultiscaleBlock(
        jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        coarse_shift,
        residual=False,
    )
    lifted = multiscale(graph)
    assert lifted.nodes.shape == graph.nodes.shape
    ```

## Graph families

For operator-learning workflows, `GraphDatasetDomain` samples graph cases and
materializes the requested node, edge, or global component over the sampled
batched topology.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph0 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        edges=jnp.array([[1.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        edges=jnp.array([[1.0], [1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )

    case_count = 2
    domain = phx.domain.GraphDatasetDomain((graph0, graph1))
    domain = domain.with_layout(domain.layout_for_batch_size(case_count, multiple=2))
    structure = phx.domain.ProductStructure((("graph",),))

    boundary = domain.component({"graph": phx.domain.BoundaryNodes([1])})
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=structure,
    )

    @domain.Function("graph")
    def u(node):
        del node
        return 1.0

    residual = phx.operators.graph_incidence_laplacian(u)
    assert jnp.allclose(residual(batch).data, jnp.zeros((case_count,)))

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=boundary,
        operator=phx.operators.graph_incidence_laplacian,
        constraint_vars="u",
        num_points=case_count,
        structure=structure,
    )
    assert constraint.loss({"u": u}) < 1e-12
    ```

## Supervised graph operator data

Graph-family targets can be exposed as fixed `DomainFunction`s or used directly
as supervised constraints. Targets are provided per graph case and stay aligned
with sampled graph entities, including explicit node/edge subsets and repeated
cases.

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import phydrax as phx

    graph0 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        edges=jnp.array([[1.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        edges=jnp.array([[1.0], [1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    targets = (
        10.0 + 2.0 * graph0.nodes[:, 0],
        10.0 + 2.0 * graph1.nodes[:, 0],
    )

    domain = phx.domain.GraphDatasetDomain((graph0, graph1))
    nodes = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def u(node):
        return 10.0 + 2.0 * node[0]

    target_fn = phx.constraints.GraphTarget(domain, targets)
    batch = domain.points_from_indices(
        [1, 0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=phx.domain.ProductStructure((("graph",),)),
    )
    assert jnp.allclose(target_fn(batch).data, jnp.array([18.0, 12.0, 18.0]))

    constraint = phx.constraints.GraphSupervisedConstraint(
        "u",
        nodes,
        targets,
        num_cases=8,
    )
    assert constraint.loss({"u": u}, key=jr.key(0)) < 1e-12

    trajectory = phx.domain.GraphTrajectoryDatasetDomain(
        (graph0, graph1),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    values = []
    for graph, length in zip(trajectory.graphs, trajectory.lengths.tolist(), strict=True):
        times = trajectory.start + trajectory.dt * jnp.arange(int(length))
        node_values = jnp.expand_dims(graph.nodes[:, 0], axis=0)
        time_values = jnp.expand_dims(times, axis=1)
        values.append(node_values + 2.0 * time_values)

    signal = phx.constraints.GraphTrajectorySignal(
        trajectory,
        tuple(values),
        interpolation="linear",
    )
    component = trajectory.component(
        {"graph": phx.domain.BoundaryNodes([1]), "t": phx.domain.Interior()}
    )
    trajectory_batch = trajectory.points_from_case_time(
        [0, 1],
        [0.25, 0.75],
        component=component,
        structure=phx.domain.ProductStructure((("graph", "t"),)),
    )
    assert jnp.allclose(signal(trajectory_batch).data, jnp.array([1.5, 5.5]))
    ```

## Geometry features

Mesh and point-cloud constructors can attach coordinate-aware feature payloads
and return graph subset metadata for boundary/interface constraints.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    bundle = phx.graph.mesh_to_geometry_graph(vertices, faces)
    domain = phx.domain.GraphDomain(bundle.graph)
    boundary_count = int(bundle.boundary_nodes.shape[0])
    boundary = domain.component({"graph": bundle.boundary_nodes_component()})
    batch = boundary.sample(
        boundary_count,
        structure=phx.domain.ProductStructure((("graph",),)),
    )

    @domain.Function("graph")
    def coordinate_sum(node):
        return jnp.sum(node["positions"], axis=-1)

    assert jnp.allclose(coordinate_sum(batch).data, jnp.array([0.0, 1.0, 1.0]))
    assert "distance" in bundle.graph.edges
    ```

## Mesh cotangent calculus

Triangular meshes can also be converted into geometry graphs carrying
cotangent edge weights and lumped vertex masses. `MeshCotangentLaplacian` is a
`GraphIR -> GraphIR` block, so it can be wrapped as a Phydrax residual while
preserving mesh metadata in the graph payload.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    bundle = phx.graph.mesh_to_cotangent_graph(vertices, faces)
    domain = phx.domain.GraphDomain(bundle.graph)
    nodes = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.ProductStructure((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 1.0

    def residual(f):
        return domain.GraphModel(
            phx.graph.MeshCotangentLaplacian(input_key="u", output_key="lap_u"),
            input_fn=f,
            input_key="u",
            output_key="lap_u",
        )

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=nodes,
        operator=residual,
        constraint_vars="u",
        num_points=bundle.graph.num_nodes,
        structure=structure,
    )

    assert constraint.loss({"u": u}) < 1e-12
    assert "cotangent_weight" in bundle.graph.edges
    assert "mass" in bundle.graph.nodes
    ```

## Graph process steppers

Graph process helpers wrap `GraphIR -> GraphIR` vector fields as one-step
integrators and autoregressive rollouts. One-step process models can be exposed
with `GraphDomain.GraphModel(...)`; multi-step process predictions can be
exposed with `GraphDomain.GraphRolloutModel(...)` and compared by ordinary
constraints.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        edges=jnp.array([[1.0], [1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )

    def unit_node_rate(g):
        return g.replace(nodes=jnp.ones_like(g.nodes), edges=None, validate=False)

    stepper = phx.graph.EulerGraphStepper(unit_node_rate, dt=0.25)
    next_graph = stepper(graph)
    nodes = phx.graph.rollout_features(stepper, graph, steps=2, feature="nodes")

    assert jnp.allclose(next_graph.nodes[:, 0], jnp.array([0.25, 1.25, 3.25]))
    assert nodes.shape == (3, 3, 1)

    domain = phx.domain.GraphDomain(graph)
    graph_nodes = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.ProductStructure((("graph",),))

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def target(node):
        return node[0] + 0.25 * jnp.arange(3.0)

    def residual(rollout):
        return rollout - target

    rollout_model = domain.GraphRolloutModel(stepper, steps=2, input_fn=u)
    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=graph_nodes,
        operator=residual,
        constraint_vars="rollout",
        num_points=graph.num_nodes,
        structure=structure,
    )

    assert constraint.loss({"rollout": rollout_model}) < 1e-12
    ```

## Graph trajectories

`GraphTrajectoryDatasetDomain` couples a graph-family case with a valid time for
that case. The sampled `GraphBatch` carries graph entity rows and a `t` field on
the same axis, so temporal graph residuals can use ordinary `DomainFunction`
arguments.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    graph0 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        edges=jnp.array([[1.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        edges=jnp.array([[1.0], [1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )

    domain = phx.domain.GraphTrajectoryDatasetDomain(
        (graph0, graph1),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    structure = phx.domain.ProductStructure((("graph", "t"),))
    case_count = 2

    edges_at_start = domain.component(
        {"graph": phx.domain.EdgeSet([0]), "t": phx.domain.FixedStart()}
    )

    @domain.Function("graph", "t")
    def u(node, t):
        del node, t
        return 1.0

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=edges_at_start,
        operator=phx.operators.graph_gradient,
        constraint_vars="u",
        num_points=case_count,
        structure=structure,
    )

    assert constraint.loss({"u": u}) < 1e-12
    ```

# Graph Domains

Graph domains make sparse graphs available through the same `DomainFunction`,
sampling, constraint, and solver contracts used by the rest of Phydrax. A graph
domain samples graph entities, not dense coordinate points: node, edge, and
graph-global values are carried with a `GraphIR` topology so graph operators and
message-passing models can evaluate on the sampled batch.

Use `GraphDomain` for one fixed graph, `GraphDatasetDomain` for a finite family of
graphs, and `GraphTrajectoryDatasetDomain` when each graph case also owns a
time-series length. In all cases the graph payload remains non-trainable domain
state.

## Single Graphs

`GraphDomain` exposes the nodes, edges, or globals of one `GraphIR` as a finite
measure space. Whole-entity components such as `Nodes()`, `Edges()`, and
`Globals()` select all entities of that kind. Explicit subsets select local entity
indices or entity types.

```python
import jax.numpy as jnp
import phydrax as phx
import jax.numpy as jnp

graph = phx.graph.GraphIR(
    nodes=jnp.asarray([[0.0], [1.0], [2.0]]),
    senders=jnp.asarray([0, 1], dtype=jnp.int32),
    receivers=jnp.asarray([1, 2], dtype=jnp.int32),
    n_node=jnp.asarray([3], dtype=jnp.int32),
    n_edge=jnp.asarray([2], dtype=jnp.int32),
)
graphs = (graph, graph)
lengths = jnp.asarray([2, 3], dtype=jnp.int32)


graph = phx.graph.GraphIR(
    nodes=jnp.asarray([[0.0], [1.0]]),
    edges=jnp.asarray([[1.0]]),
    senders=jnp.asarray([0], dtype=jnp.int32),
    receivers=jnp.asarray([1], dtype=jnp.int32),
    globals=jnp.asarray([[0.0]]),
    n_node=jnp.asarray([2], dtype=jnp.int32),
    n_edge=jnp.asarray([1], dtype=jnp.int32),
)
other_graph = phx.graph.GraphIR(
    nodes=jnp.asarray([[0.0], [1.0], [2.0]]),
    edges=jnp.asarray([[1.0], [1.0]]),
    senders=jnp.asarray([0, 1], dtype=jnp.int32),
    receivers=jnp.asarray([1, 2], dtype=jnp.int32),
    globals=jnp.asarray([[1.0]]),
    n_node=jnp.asarray([3], dtype=jnp.int32),
    n_edge=jnp.asarray([2], dtype=jnp.int32),
)
graphs = (graph, other_graph)
lengths = jnp.asarray([3, 5], dtype=jnp.int32)

graph_domain = phx.domain.GraphDomain(graph, label="mesh")
nodes = graph_domain.component({"mesh": phx.domain.Nodes()})
```

::: phydrax.domain.GraphDomain
    options:
        members:
            - __init__
            - label
            - measure_mode
            - num_nodes
            - num_edges
            - num_graphs
            - component_size
            - component_measure
            - sample_component
            - GraphModel
            - GraphRolloutModel
            - equivalent

## Graph Datasets

`GraphDatasetDomain` samples graph cases from a finite collection, batches their
topology into one `GraphIR`, and then materializes the requested entities. Layouts
are optional static padding plans for JIT-stable graph batches.

```python
domain = phx.domain.GraphDatasetDomain(graphs, label="graph")
layout = domain.layout_for_batch_size(16)
domain = domain.with_layout(layout)
```

::: phydrax.domain.GraphDatasetDomain
    options:
        members:
            - __init__
            - label
            - size
            - measure_mode
            - layout
            - layout_for_batch_size
            - with_layout
            - sample_indices
            - component_size
            - component_measure
            - sample_component
            - points_from_indices
            - GraphModel
            - GraphRolloutModel
            - equivalent

## Graph Trajectories

`GraphTrajectoryDatasetDomain` couples a graph-family axis with a time axis. Each
graph case has a valid trajectory length on a shared uniform time grid. Sampling
keeps `(graph, t)` paired so fixed-start, fixed-end, boundary, and interior time
components are interpreted per graph case.

```python
domain = phx.domain.GraphTrajectoryDatasetDomain(
    graphs,
    lengths,
    dt=0.1,
    graph_label="graph",
    time_label="t",
)
component = domain.component({"graph": phx.domain.Nodes(), "t": phx.domain.Interior()})
```

::: phydrax.domain.GraphTrajectoryDatasetDomain
    options:
        members:
            - __init__
            - labels
            - graph_label
            - time_label
            - measure_mode
            - sampling_mode
            - layout
            - size
            - max_length
            - total_observations
            - durations
            - end_times
            - factor
            - layout_for_batch_size
            - with_layout
            - observation_times
            - points_from_case_time
            - GraphModel
            - GraphRolloutModel
            - equivalent

## Batches

`GraphBatch` is returned by graph-domain sampling. It behaves like a mapping from
domain labels to `coordax.Field` trees, and also carries the batched `GraphIR`
topology required by graph operators and neural graph models.

::: phydrax.domain.GraphBatch
    options:
        members: []

## Graph Entity Components

Use these markers with `domain.component(...)` to choose which graph entities a
constraint, function, or sample is defined on.

::: phydrax.domain.Nodes
    options:
        members: []

---

::: phydrax.domain.Edges
    options:
        members: []

---

::: phydrax.domain.Globals
    options:
        members: []

### Cochain cells and typed fields

`CochainCells(degree, region=...)` selects all, interior, or geometric boundary
cells of one degree from the node representation of a canonical cochain
complex. Selection remains local to each graph in dataset and trajectory
batches and excludes padded nodes.

`as_cochain_field` attaches a `CochainFieldSpec` to a graph-backed
`DomainFunction` and masks every other degree to zero. `cochain_field_spec`
recovers that declaration for validation by DEC operators, constraints, and
hard enforcement.

::: phydrax.domain.CochainCells
    options:
        members:
            - __init__

---

::: phydrax.domain.as_cochain_field

---

::: phydrax.domain.cochain_field_spec

## Explicit Graph Subsets

Explicit subsets are local to each graph case. For a `GraphDatasetDomain`, the
same local subset selector is applied to every sampled graph. Type selectors
expect mapping-valued graph payloads with an integer type field such as
`graph.nodes["type"]` or `graph.edges["type"]`.

::: phydrax.domain.NodeSet
    options:
        members: []

---

::: phydrax.domain.EdgeSet
    options:
        members: []

---

::: phydrax.domain.NodeType
    options:
        members: []

---

::: phydrax.domain.EdgeType
    options:
        members: []

---

::: phydrax.domain.BoundaryNodes
    options:
        members: []

---

::: phydrax.domain.InteriorNodes
    options:
        members: []

---

::: phydrax.domain.BoundaryEdges
    options:
        members: []

---

::: phydrax.domain.InterfaceEdges
    options:
        members: []

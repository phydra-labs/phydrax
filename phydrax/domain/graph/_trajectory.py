#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping, Sequence
from typing import Literal

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ...graph import batch_graphs, GraphIR, LayoutPlan
from .._components import Boundary, Fixed, FixedEnd, FixedStart, Interior
from .._domain import _AbstractDomain, _AbstractUnaryDomain
from .._scalar import ScalarInterval
from .._structure import _validate_label, NumPoints, ProductStructure
from ._batch import GRAPH_ENTITY_INDEX_KEY, GRAPH_GRAPH_INDEX_KEY, GraphBatch
from ._components import graph_component_kind
from ._dataset import (
    _component_indices_for_graph,
    _entity_payload,
    _graph_ids_for_kind,
    _offsets_for_kind,
    _take_tree,
    _to_axis_fields,
    GRAPH_DATASET_INDEX_KEY,
    GRAPH_ENTITY_OFFSET_KEY,
    GRAPH_SAMPLE_INDEX_KEY,
    GraphDatasetDomain,
)


GraphTrajectoryMeasure = Literal[
    "case_time_probability",
    "time_integral_average",
    "time_integral_sum",
]
GraphTrajectorySampling = Literal["case_time_uniform", "observation_uniform"]

GRAPH_TRAJECTORY_TIME_INDEX_KEY = "__phydrax_graph_trajectory_time_index__"


def _as_lengths(lengths: ArrayLike, n: int, /) -> Array:
    arr = jnp.asarray(lengths)
    if arr.ndim != 1:
        raise ValueError(f"lengths must have shape (N,), got {arr.shape}.")
    if int(arr.shape[0]) != n:
        raise ValueError(f"lengths must have length {n}, got {int(arr.shape[0])}.")
    arr_i = arr.astype(jnp.int32)
    if bool(jnp.any(arr_i <= 0)):
        raise ValueError("All graph trajectory lengths must be positive.")
    if bool(jnp.any(arr_i != arr)):
        raise ValueError("Graph trajectory lengths must be integer-valued.")
    return arr_i


def _as_scalar(name: str, value: ArrayLike, /) -> Array:
    arr = jnp.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {arr.shape}.")
    return arr.reshape(())


def _as_sample_count(num_points: NumPoints, /) -> int:
    if isinstance(num_points, int):
        n = int(num_points)
    else:
        if len(num_points) != 1:
            raise ValueError(
                "GraphTrajectoryDatasetDomain requires paired graph-time sampling."
            )
        n = int(num_points[0])
    if n < 0:
        raise ValueError("num_points must be non-negative.")
    return n


def _single_axis_for_graph_trajectory(
    domain: "GraphTrajectoryDatasetDomain",
    structure: ProductStructure,
    /,
) -> tuple[ProductStructure, str]:
    structure = structure.canonicalize(domain.labels, fixed_labels=frozenset())
    if len(structure.blocks) != 1:
        raise ValueError(
            "GraphTrajectoryDatasetDomain sampling requires graph and time in one "
            "paired ProductStructure block."
        )
    block = frozenset(structure.blocks[0])
    if block != frozenset(domain.labels):
        raise ValueError(
            "GraphTrajectoryDatasetDomain sampling requires a paired block containing "
            f"{domain.labels}."
        )
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("Graph trajectory ProductStructure must be canonicalized.")
    return structure, axis_names[0]


def _sample_cases_uniform(
    domain: "GraphTrajectoryDatasetDomain", n: int, key: Key[Array, ""], /
) -> Array:
    return jr.randint(key, shape=(n,), minval=0, maxval=domain.size, dtype=jnp.int32)


def _sample_valid_cases(valid: Array, n: int, key: Key[Array, ""], /) -> Array:
    valid_np = np.asarray(valid, dtype=bool)
    if not bool(valid_np.any()):
        raise ValueError("No graph trajectories are valid for this fixed time.")
    valid_idx = jnp.asarray(np.nonzero(valid_np)[0], dtype=jnp.int32)
    draw = jr.randint(key, shape=(n,), minval=0, maxval=int(valid_idx.shape[0]))
    return valid_idx[draw]


def _component_times(
    domain: "GraphTrajectoryDatasetDomain",
    component,
    case_indices: Array,
    n: int,
    key: Key[Array, ""],
    /,
) -> tuple[Array, Array]:
    comp = component.spec.component_for(domain.time_label)
    lengths = domain.lengths[case_indices]
    durations = (lengths.astype(float) - 1.0) * domain.dt

    if isinstance(comp, Interior):
        if domain.sampling_mode == "observation_uniform":
            obs_idx = jr.randint(
                key,
                shape=(n,),
                minval=0,
                maxval=domain.total_observations,
                dtype=jnp.int32,
            )
            time_idx = domain.flat_time_indices[obs_idx]
            return domain.observation_times(domain.flat_case_indices[obs_idx], time_idx), time_idx
        u = jr.uniform(key, shape=(n,))
        times = domain.start + u * durations
        time_idx = jnp.rint((times - domain.start) / domain.dt).astype(jnp.int32)
        time_idx = jnp.clip(time_idx, 0, lengths - 1)
        return times, time_idx

    if isinstance(comp, FixedStart):
        return jnp.full((n,), domain.start, dtype=float), jnp.zeros((n,), dtype=jnp.int32)

    if isinstance(comp, FixedEnd):
        time_idx = lengths - 1
        return domain.end_times[case_indices], time_idx

    if isinstance(comp, Fixed):
        value = jnp.asarray(comp.value, dtype=float).reshape(())
        time_idx = jnp.rint((value - domain.start) / domain.dt).astype(jnp.int32)
        return (
            jnp.full((n,), value, dtype=float),
            jnp.full((n,), time_idx, dtype=jnp.int32),
        )

    if isinstance(comp, Boundary):
        pick_end = jr.bernoulli(key, p=0.5, shape=(n,))
        time_idx = jnp.where(pick_end, lengths - 1, 0).astype(jnp.int32)
        times = jnp.where(pick_end, domain.end_times[case_indices], domain.start)
        return times, time_idx

    raise TypeError(f"Unsupported graph trajectory time component {type(comp).__name__}.")


class GraphTrajectoryDatasetDomain(_AbstractDomain):
    """A coupled graph-family and time domain for graph trajectories.

    Each graph case owns a valid trajectory length on a shared uniform time grid.
    Sampling keeps graph and time labels paired, so time components such as
    `FixedEnd()` are interpreted relative to the selected graph case.
    """

    graphs: tuple[GraphIR, ...]
    lengths: Array
    dt: Array
    start: Array
    _graph_label: str
    _time_label: str
    _measure: GraphTrajectoryMeasure
    _sampling: GraphTrajectorySampling
    _layout: LayoutPlan | None
    _graph_factor: GraphDatasetDomain
    _time_factor: ScalarInterval
    _max_length: int
    _total_observations: int
    _flat_case_indices: Array
    _flat_time_indices: Array

    def __init__(
        self,
        graphs: Sequence[GraphIR],
        lengths: ArrayLike,
        /,
        *,
        dt: ArrayLike,
        start: ArrayLike = 0.0,
        graph_label: str = "graph",
        time_label: str = "t",
        measure: GraphTrajectoryMeasure = "case_time_probability",
        sampling: GraphTrajectorySampling = "case_time_uniform",
        layout: LayoutPlan | None = None,
        validate: bool = True,
    ):
        """Create a finite graph-trajectory domain.

        Parameters:
            graphs: Graph cases sampled by the domain.
            lengths: Valid time-step count for each graph case.
            dt: Uniform time spacing shared by all graph cases.
            start: Shared start time.
            graph_label: Label used for sampled graph entities.
            time_label: Label used for sampled times.
            measure: Measure mode for graph-time reductions.
            sampling: Strategy for drawing interior graph-time samples.
            layout: Optional static graph-batch padding plan.
            validate: Validate each `GraphIR` before storing it.
        """
        _validate_label(str(graph_label))
        _validate_label(str(time_label))
        if str(graph_label) == str(time_label):
            raise ValueError("graph_label and time_label must be distinct.")
        if len(graphs) == 0:
            raise ValueError("GraphTrajectoryDatasetDomain requires at least one graph.")

        graphs_tuple = tuple(graphs)
        for graph in graphs_tuple:
            if not isinstance(graph, GraphIR):
                raise TypeError(
                    "GraphTrajectoryDatasetDomain expects phydrax.graph.GraphIR values."
                )
            if validate:
                graph.validate()

        lengths_arr = _as_lengths(lengths, len(graphs_tuple))
        dt_arr = _as_scalar("dt", dt)
        if bool(dt_arr <= 0):
            raise ValueError("dt must be positive.")
        start_arr = _as_scalar("start", start)

        if measure not in (
            "case_time_probability",
            "time_integral_average",
            "time_integral_sum",
        ):
            raise ValueError(
                "measure must be 'case_time_probability', "
                "'time_integral_average', or 'time_integral_sum'."
            )
        if sampling not in ("case_time_uniform", "observation_uniform"):
            raise ValueError(
                "sampling must be 'case_time_uniform' or 'observation_uniform'."
            )

        flat_case_parts: list[Array] = []
        flat_time_parts: list[Array] = []
        for i, length in enumerate(list(map(int, lengths_arr.tolist()))):
            flat_case_parts.append(jnp.full((length,), i, dtype=jnp.int32))
            flat_time_parts.append(jnp.arange(length, dtype=jnp.int32))

        max_length = int(jnp.max(lengths_arr))
        support_steps = max(max_length - 1, 1)
        end = start_arr + dt_arr * float(support_steps)

        self.graphs = graphs_tuple
        self.lengths = lengths_arr
        self.dt = dt_arr
        self.start = start_arr
        self._graph_label = str(graph_label)
        self._time_label = str(time_label)
        self._measure = measure
        self._sampling = sampling
        self._layout = layout
        self._graph_factor = GraphDatasetDomain(
            graphs_tuple,
            label=str(graph_label),
            measure="probability",
            layout=layout,
            validate=False,
        )
        self._time_factor = ScalarInterval(float(start_arr), float(end), label=str(time_label))
        self._max_length = max_length
        self._total_observations = int(jnp.sum(lengths_arr))
        self._flat_case_indices = jnp.concatenate(flat_case_parts, axis=0)
        self._flat_time_indices = jnp.concatenate(flat_time_parts, axis=0)

    @property
    def labels(self) -> tuple[str, str]:
        """Graph and time labels owned by the domain."""
        return (self._graph_label, self._time_label)

    @property
    def graph_label(self) -> str:
        """Label used for graph entity payloads."""
        return self._graph_label

    @property
    def time_label(self) -> str:
        """Label used for sampled trajectory times."""
        return self._time_label

    @property
    def measure_mode(self) -> GraphTrajectoryMeasure:
        """Measure mode used for graph-time reductions."""
        return self._measure

    @property
    def sampling_mode(self) -> GraphTrajectorySampling:
        """Sampling strategy used for interior graph-time points."""
        return self._sampling

    @property
    def layout(self) -> LayoutPlan | None:
        """Static graph-batch layout, if sampled batches are padded."""
        return self._layout

    @property
    def size(self) -> int:
        """Number of graph trajectory cases."""
        return len(self.graphs)

    @property
    def max_length(self) -> int:
        """Maximum valid time-step count across graph cases."""
        return int(self._max_length)

    @property
    def total_observations(self) -> int:
        """Total number of valid graph-time observations across cases."""
        return int(self._total_observations)

    @property
    def flat_case_indices(self) -> Array:
        """Flat case index for each valid graph-time observation."""
        return self._flat_case_indices

    @property
    def flat_time_indices(self) -> Array:
        """Flat local time index for each valid graph-time observation."""
        return self._flat_time_indices

    @property
    def durations(self) -> Array:
        """Per-case trajectory duration, `(length - 1) * dt`."""
        return (self.lengths.astype(float) - 1.0) * self.dt

    @property
    def end_times(self) -> Array:
        """Per-case final valid time."""
        return self.start + self.durations

    def factor(self, label: str, /) -> _AbstractUnaryDomain:
        """Return the unary graph or time factor for `label`."""
        if label == self._graph_label:
            return self._graph_factor
        if label == self._time_label:
            return self._time_factor
        raise KeyError(f"Label {label!r} not in domain {self.labels}.")

    def layout_for_batch_size(
        self,
        num_cases: int,
        /,
        *,
        multiple: int = 1,
    ) -> LayoutPlan:
        """Return a worst-case static graph layout for `num_cases` samples."""
        return self._graph_factor.layout_for_batch_size(num_cases, multiple=multiple)

    def with_layout(self, layout: LayoutPlan | None, /) -> "GraphTrajectoryDatasetDomain":
        """Return a copy that packs sampled graph batches with `layout`."""
        return GraphTrajectoryDatasetDomain(
            self.graphs,
            self.lengths,
            dt=self.dt,
            start=self.start,
            graph_label=self.graph_label,
            time_label=self.time_label,
            measure=self.measure_mode,
            sampling=self.sampling_mode,
            layout=layout,
            validate=False,
        )

    def observation_times(self, case_indices: ArrayLike, time_indices: ArrayLike, /) -> Array:
        """Convert local time indices to physical times."""
        del case_indices
        return self.start + self.dt * jnp.asarray(time_indices, dtype=float)

    def points_from_case_time(
        self,
        case_indices: ArrayLike,
        times: ArrayLike,
        /,
        *,
        component,
        structure: ProductStructure | None = None,
        time_indices: ArrayLike | None = None,
    ) -> GraphBatch:
        """Materialize graph-time samples by explicit case and time arrays.

        `case_indices` and `times` must have the same leading length. The selected
        graph component is expanded to all matching entities for each sampled
        graph-time pair.
        """
        structure_in = structure or ProductStructure((self.labels,))
        structure_out, axis = _single_axis_for_graph_trajectory(self, structure_in)
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
        time_arr = jnp.asarray(times, dtype=float).reshape((-1,))
        if int(case_idx.shape[0]) != int(time_arr.shape[0]):
            raise ValueError("case_indices and times must have the same length.")
        if int(case_idx.shape[0]) == 0:
            raise ValueError("Graph trajectory graph batches must be non-empty.")
        case_np = np.asarray(case_idx)
        if np.any(case_np < 0) or np.any(case_np >= self.size):
            raise ValueError(f"Graph trajectory case indices must be in [0, {self.size}).")
        if time_indices is None:
            time_idx = jnp.rint((time_arr - self.start) / self.dt).astype(jnp.int32)
        else:
            time_idx = jnp.asarray(time_indices, dtype=jnp.int32).reshape((-1,))
            if int(time_idx.shape[0]) != int(case_idx.shape[0]):
                raise ValueError("time_indices must have the same length as case_indices.")

        graph_component = component.spec.component_for(self.graph_label)
        kind = graph_component_kind(graph_component)
        selected_graphs = tuple(self.graphs[int(i)] for i in case_np.tolist())
        real_batched = batch_graphs(selected_graphs, validate=True)
        batched = self._layout.pack(real_batched) if self._layout is not None else real_batched
        offsets = _offsets_for_kind(selected_graphs, kind)

        entity_parts: list[Array] = []
        dataset_parts: list[Array] = []
        sample_parts: list[Array] = []
        offset_parts: list[Array] = []
        time_parts: list[Array] = []
        time_index_parts: list[Array] = []
        for pos, (case_index, graph, offset) in enumerate(
            zip(case_np.tolist(), selected_graphs, offsets, strict=True)
        ):
            local = _component_indices_for_graph(graph, graph_component, kind)
            global_indices = local + jnp.asarray(offset, dtype=jnp.int32)
            n_entities = int(local.shape[0])
            entity_parts.append(global_indices)
            dataset_parts.append(jnp.full((n_entities,), int(case_index), dtype=jnp.int32))
            sample_parts.append(jnp.full((n_entities,), int(pos), dtype=jnp.int32))
            offset_parts.append(jnp.full((n_entities,), int(offset), dtype=jnp.int32))
            time_parts.append(jnp.full((n_entities,), time_arr[pos], dtype=float))
            time_index_parts.append(jnp.full((n_entities,), time_idx[pos], dtype=jnp.int32))

        entity_indices = jnp.concatenate(entity_parts, axis=0)
        dataset_indices = jnp.concatenate(dataset_parts, axis=0)
        sample_indices = jnp.concatenate(sample_parts, axis=0)
        entity_offsets = jnp.concatenate(offset_parts, axis=0)
        entity_times = jnp.concatenate(time_parts, axis=0)
        entity_time_indices = jnp.concatenate(time_index_parts, axis=0)

        payload = _take_tree(_entity_payload(batched, kind), entity_indices)
        graph_ids = _graph_ids_for_kind(real_batched, kind)[entity_indices]
        points = {
            self.graph_label: _to_axis_fields(payload, axis),
            self.time_label: cx.Field(entity_times, dims=(axis,)),
            GRAPH_ENTITY_INDEX_KEY: cx.Field(entity_indices, dims=(axis,)),
            GRAPH_GRAPH_INDEX_KEY: cx.Field(graph_ids, dims=(axis,)),
            GRAPH_DATASET_INDEX_KEY: cx.Field(dataset_indices, dims=(axis,)),
            GRAPH_SAMPLE_INDEX_KEY: cx.Field(sample_indices, dims=(axis,)),
            GRAPH_ENTITY_OFFSET_KEY: cx.Field(entity_offsets, dims=(axis,)),
            GRAPH_TRAJECTORY_TIME_INDEX_KEY: cx.Field(entity_time_indices, dims=(axis,)),
        }
        return GraphBatch(
            points=frozendict(points),
            structure=structure_out,
            graph=batched,
            graph_label=self.graph_label,
            component_kind=kind,
        )

    def GraphModel(
        self,
        model,
        /,
        *,
        input_fn=None,
        edge_input_fn=None,
        global_input_fn=None,
        output: Literal["nodes", "edges", "globals"] = "nodes",
        input_key: str | None = None,
        edge_input_key: str | None = None,
        global_input_key: str | None = None,
        output_key: str | None = None,
    ):
        """Wrap a `GraphIR -> GraphIR` model as a graph-time `DomainFunction`.

        The model sees the sampled batched graph topology and may also consume the
        sampled time label through its input functions.
        """
        from ...domain._function import DomainFunction
        from ...nn import GraphModel

        return DomainFunction(
            domain=self,
            deps=(self.graph_label, self.time_label),
            func=GraphModel(
                model,
                input_fn=input_fn,
                edge_input_fn=edge_input_fn,
                global_input_fn=global_input_fn,
                output=output,
                input_key=input_key,
                edge_input_key=edge_input_key,
                global_input_key=global_input_key,
                output_key=output_key,
            ),
        )

    def GraphRolloutModel(
        self,
        stepper,
        /,
        *,
        steps: int,
        include_initial: bool = True,
        feature: Literal["nodes", "edges", "globals"] = "nodes",
        input_fn=None,
        edge_input_fn=None,
        global_input_fn=None,
        input_key: str | None = None,
        edge_input_key: str | None = None,
        global_input_key: str | None = None,
        output_key: str | None = None,
    ):
        """Wrap an autoregressive graph rollout as a graph-time `DomainFunction`.

        Use this for graph sequence models whose state is advanced by repeatedly
        applying a one-step graph stepper.
        """
        from ...domain._function import DomainFunction
        from ...nn import GraphRolloutModel

        return DomainFunction(
            domain=self,
            deps=(self.graph_label, self.time_label),
            func=GraphRolloutModel(
                stepper,
                steps=steps,
                include_initial=include_initial,
                feature=feature,
                input_fn=input_fn,
                edge_input_fn=edge_input_fn,
                global_input_fn=global_input_fn,
                input_key=input_key,
                edge_input_key=edge_input_key,
                global_input_key=global_input_key,
                output_key=output_key,
            ),
        )

    def equivalent(self, other: object, /) -> bool:
        """Return whether another domain has the same public graph-time shape."""
        if not isinstance(other, GraphTrajectoryDatasetDomain):
            return False
        if self.labels != other.labels:
            return False
        if self.measure_mode != other.measure_mode:
            return False
        if self.sampling_mode != other.sampling_mode:
            return False
        if self.size != other.size:
            return False
        if bool(jnp.any(self.lengths != other.lengths)):
            return False
        if bool(jnp.any(self.dt != other.dt)):
            return False
        if bool(jnp.any(self.start != other.start)):
            return False
        return self._graph_factor.equivalent(other._graph_factor)


def sample_graph_trajectory_component(
    component,
    num_points: NumPoints,
    *,
    structure: ProductStructure,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> GraphBatch:
    del sampler
    domain = component.domain
    if not isinstance(domain, GraphTrajectoryDatasetDomain):
        raise TypeError(
            "sample_graph_trajectory_component requires a GraphTrajectoryDatasetDomain."
        )

    n = _as_sample_count(num_points)
    structure_out, _axis = _single_axis_for_graph_trajectory(domain, structure)
    if n == 0:
        raise ValueError("GraphTrajectoryDatasetDomain requires at least one sample.")

    case_key, time_key = jr.split(key)
    time_comp = component.spec.component_for(domain.time_label)
    if isinstance(time_comp, Fixed):
        fixed_value = jnp.asarray(time_comp.value, dtype=float).reshape(())
        valid = (domain.start <= fixed_value) & (fixed_value <= domain.end_times)
        case_indices = _sample_valid_cases(valid, n, case_key)
    elif domain.sampling_mode == "observation_uniform" and isinstance(
        time_comp, Interior
    ):
        obs_idx = jr.randint(
            time_key,
            shape=(n,),
            minval=0,
            maxval=domain.total_observations,
            dtype=jnp.int32,
        )
        case_indices = domain.flat_case_indices[obs_idx]
        time_indices = domain.flat_time_indices[obs_idx]
        times = domain.observation_times(case_indices, time_indices)
        return domain.points_from_case_time(
            case_indices,
            times,
            component=component,
            structure=structure_out,
            time_indices=time_indices,
        )
    else:
        case_indices = _sample_cases_uniform(domain, n, case_key)

    times, time_indices = _component_times(domain, component, case_indices, n, time_key)
    return domain.points_from_case_time(
        case_indices,
        times,
        component=component,
        structure=structure_out,
        time_indices=time_indices,
    )


def graph_trajectory_component_measure(component, /) -> Array | None:
    domain = component.domain
    if not isinstance(domain, GraphTrajectoryDatasetDomain):
        return None

    graph_comp = component.spec.component_for(domain.graph_label)
    graph_measure = domain._graph_factor.component_measure(graph_comp)
    time_comp = component.spec.component_for(domain.time_label)
    point_mass = isinstance(time_comp, (FixedStart, FixedEnd, Fixed))
    boundary = isinstance(time_comp, Boundary)

    if domain.measure_mode == "case_time_probability":
        return jnp.asarray(1.0, dtype=float)
    if point_mass:
        time_measure = jnp.asarray(1.0, dtype=float)
    elif boundary:
        time_measure = jnp.asarray(2.0, dtype=float)
    else:
        time_measure = jnp.mean(domain.durations)
    if domain.measure_mode == "time_integral_sum":
        time_measure = time_measure * float(domain.size)
    return jnp.asarray(graph_measure, dtype=float) * jnp.asarray(time_measure, dtype=float)


def graph_trajectory_default_quadrature_total_weight(
    component, batch: GraphBatch, /
) -> cx.Field | None:
    domain = component.domain
    if not isinstance(domain, GraphTrajectoryDatasetDomain):
        return None
    _structure, axis = _single_axis_for_graph_trajectory(domain, batch.structure)
    case_field = batch.points.get(GRAPH_DATASET_INDEX_KEY)
    if not isinstance(case_field, cx.Field):
        return None
    case_idx = jnp.asarray(case_field.data, dtype=jnp.int32)
    n = int(case_idx.shape[0])
    if n == 0:
        return cx.Field(jnp.zeros((0,), dtype=float), dims=(axis,))

    time_comp = component.spec.component_for(domain.time_label)
    point_mass = isinstance(time_comp, (FixedStart, FixedEnd, Fixed))
    boundary = isinstance(time_comp, Boundary)
    durations = domain.durations[case_idx]

    if domain.measure_mode == "case_time_probability":
        per_entity = jnp.ones((n,), dtype=float)
    elif point_mass:
        per_entity = jnp.ones((n,), dtype=float)
    elif boundary:
        per_entity = jnp.full((n,), 2.0, dtype=float)
    else:
        per_entity = durations
    if domain.measure_mode == "time_integral_sum":
        per_entity = per_entity * float(domain.size)
    return cx.Field(per_entity / float(n), dims=(axis,))


def graph_trajectory_quadrature_weights_by_axis(
    component, batch: GraphBatch, /
) -> Mapping[str, cx.Field] | None:
    domain = component.domain
    if not isinstance(domain, GraphTrajectoryDatasetDomain):
        return None
    _structure, axis = _single_axis_for_graph_trajectory(domain, batch.structure)
    weight = graph_trajectory_default_quadrature_total_weight(component, batch)
    if weight is None:
        return None
    return {axis: weight}


__all__ = [
    "GRAPH_TRAJECTORY_TIME_INDEX_KEY",
    "GraphTrajectoryDatasetDomain",
    "GraphTrajectoryMeasure",
    "GraphTrajectorySampling",
    "graph_trajectory_component_measure",
    "graph_trajectory_default_quadrature_total_weight",
    "graph_trajectory_quadrature_weights_by_axis",
    "sample_graph_trajectory_component",
]

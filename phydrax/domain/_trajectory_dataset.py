#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from ._components import (
    Boundary,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ._dataset import DatasetDomain
from ._domain import _AbstractDomain, _AbstractUnaryDomain
from ._scalar import ScalarInterval
from ._structure import _validate_label, NumPoints, PointsBatch, ProductStructure


TrajectoryMeasure = Literal[
    "case_time_probability",
    "time_integral_average",
    "time_integral_sum",
]
TrajectorySampling = Literal["case_time_uniform", "observation_uniform"]

TRAJECTORY_CASE_INDEX_KEY = "__phydrax_trajectory_case_index"
TRAJECTORY_TIME_INDEX_KEY = "__phydrax_trajectory_time_index"


def _tree_leading_axis_size(tree: PyTree[ArrayLike], /) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("TrajectoryDatasetDomain requires at least one input leaf.")

    first = jnp.asarray(leaves[0])
    if first.ndim == 0:
        raise ValueError("Trajectory input leaves must have a leading dataset axis.")
    n = int(first.shape[0])
    if n <= 0:
        raise ValueError("Trajectory input leading axis must be non-empty.")

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.ndim == 0:
            raise ValueError("Trajectory input leaves must have a leading dataset axis.")
        if int(arr.shape[0]) != n:
            raise ValueError(
                "TrajectoryDatasetDomain requires all input leaves to share the same "
                f"leading axis; got {int(arr.shape[0])} and {n}."
            )
    return n


def _as_lengths(lengths: ArrayLike, n: int, /) -> Array:
    arr = jnp.asarray(lengths)
    if arr.ndim != 1:
        raise ValueError(f"lengths must have shape (N,), got {arr.shape}.")
    if int(arr.shape[0]) != n:
        raise ValueError(f"lengths must have length {n}, got {int(arr.shape[0])}.")
    arr_i = arr.astype(jnp.int32)
    if bool(jnp.any(arr_i <= 0)):
        raise ValueError("All trajectory lengths must be positive.")
    if bool(jnp.any(arr_i != arr)):
        raise ValueError("Trajectory lengths must be integer-valued.")
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
                "TrajectoryDatasetDomain requires paired sampling with one sample count."
            )
        n = int(num_points[0])
    if n < 0:
        raise ValueError("num_points must be non-negative.")
    return n


def _single_axis_for_trajectory(
    domain: "TrajectoryDatasetDomain",
    structure: ProductStructure,
    /,
) -> tuple[ProductStructure, str]:
    structure = structure.canonicalize(domain.labels, fixed_labels=frozenset())
    if len(structure.blocks) != 1:
        raise ValueError(
            "TrajectoryDatasetDomain sampling requires the data and time labels to be "
            "sampled in one paired block."
        )
    block = frozenset(structure.blocks[0])
    if block != frozenset(domain.labels):
        raise ValueError(
            "TrajectoryDatasetDomain sampling requires a paired block containing "
            f"{domain.labels}."
        )
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("Trajectory ProductStructure must be canonicalized.")
    return structure, axis_names[0]


class TrajectoryDatasetDomain(_AbstractDomain):
    """A coupled finite-function and time domain for ragged trajectory data.

    Each dataset row represents one input/function/parameterization. The same row also
    owns a valid time interval sampled at constant spacing `dt`, with per-row length
    supplied by `lengths`.

    The domain exposes two labels, by default `("data", "t")`, but samples them as a
    coupled pair. This is the right domain for models `u(data, t)` when each dataset
    element has its own trajectory horizon.
    """

    inputs: PyTree[Array]
    lengths: Array
    dt: Array
    start: Array
    _data_label: str
    _time_label: str
    _measure: TrajectoryMeasure
    _sampling: TrajectorySampling
    _data_factor: DatasetDomain
    _time_factor: ScalarInterval
    _max_length: int
    _total_observations: int
    _flat_case_indices: Array
    _flat_time_indices: Array

    def __init__(
        self,
        inputs: PyTree[ArrayLike],
        lengths: ArrayLike,
        /,
        *,
        dt: ArrayLike,
        start: ArrayLike = 0.0,
        data_label: str = "data",
        time_label: str = "t",
        measure: TrajectoryMeasure = "case_time_probability",
        sampling: TrajectorySampling = "case_time_uniform",
    ):
        """Create a finite dataset of row-conditioned trajectories.

        Parameters:
            inputs: Per-case input PyTree with a shared leading case axis.
            lengths: Valid time-step count for each case.
            dt: Uniform time spacing shared by all cases.
            start: Shared start time.
            data_label: Label used for sampled input rows.
            time_label: Label used for sampled times.
            measure: Measure mode for coupled case-time reductions.
            sampling: Strategy for drawing interior case-time samples.
        """
        _validate_label(str(data_label))
        _validate_label(str(time_label))
        if str(data_label) == str(time_label):
            raise ValueError("data_label and time_label must be distinct.")

        arrays = jax.tree_util.tree_map(lambda x: jnp.asarray(x), inputs)
        n = _tree_leading_axis_size(arrays)
        lengths_arr = _as_lengths(lengths, n)
        dt_arr = _as_scalar("dt", dt)
        if bool(dt_arr <= 0):
            raise ValueError("dt must be positive.")
        start_arr = _as_scalar("start", start)

        measure_str = str(measure)
        if measure_str not in (
            "case_time_probability",
            "time_integral_average",
            "time_integral_sum",
        ):
            raise ValueError(
                "measure must be one of 'case_time_probability', "
                "'time_integral_average', or 'time_integral_sum'."
            )
        measure_value: TrajectoryMeasure
        if measure_str == "case_time_probability":
            measure_value = "case_time_probability"
        elif measure_str == "time_integral_average":
            measure_value = "time_integral_average"
        else:
            measure_value = "time_integral_sum"

        sampling_str = str(sampling)
        if sampling_str not in ("case_time_uniform", "observation_uniform"):
            raise ValueError(
                "sampling must be either 'case_time_uniform' or 'observation_uniform'."
            )
        sampling_value: TrajectorySampling
        if sampling_str == "case_time_uniform":
            sampling_value = "case_time_uniform"
        else:
            sampling_value = "observation_uniform"

        max_length = int(jnp.max(lengths_arr))
        total_observations = int(jnp.sum(lengths_arr))
        flat_case_parts: list[Array] = []
        flat_time_parts: list[Array] = []
        for i, length in enumerate(list(map(int, lengths_arr.tolist()))):
            flat_case_parts.append(jnp.full((length,), i, dtype=jnp.int32))
            flat_time_parts.append(jnp.arange(length, dtype=jnp.int32))
        # ScalarInterval requires a non-degenerate interval. Single-sample trajectories
        # still expose a tiny support interval for metadata; actual samples stay at start.
        support_steps = max(max_length - 1, 1)
        end = start_arr + dt_arr * float(support_steps)

        self.inputs = arrays
        self.lengths = lengths_arr
        self.dt = dt_arr
        self.start = start_arr
        self._data_label = str(data_label)
        self._time_label = str(time_label)
        self._measure = measure_value
        self._sampling = sampling_value
        self._data_factor = DatasetDomain(
            arrays, label=str(data_label), measure="probability"
        )
        self._time_factor = ScalarInterval(
            float(start_arr), float(end), label=str(time_label)
        )
        self._max_length = max_length
        self._total_observations = total_observations
        self._flat_case_indices = jnp.concatenate(flat_case_parts, axis=0)
        self._flat_time_indices = jnp.concatenate(flat_time_parts, axis=0)

    @property
    def labels(self) -> tuple[str, ...]:
        """Data and time labels owned by the domain."""
        return (self._data_label, self._time_label)

    @property
    def data_label(self) -> str:
        """Label used for sampled input rows."""
        return self._data_label

    @property
    def time_label(self) -> str:
        """Label used for sampled trajectory times."""
        return self._time_label

    @property
    def measure_mode(self) -> TrajectoryMeasure:
        """Measure mode used for coupled case-time reductions."""
        return self._measure

    @property
    def sampling_mode(self) -> TrajectorySampling:
        """Sampling strategy used for interior case-time points."""
        return self._sampling

    @property
    def size(self) -> int:
        """Number of trajectory cases."""
        return int(self.lengths.shape[0])

    @property
    def max_length(self) -> int:
        """Maximum valid time-step count across cases."""
        return int(self._max_length)

    @property
    def total_observations(self) -> int:
        """Total number of valid time observations across all cases."""
        return int(self._total_observations)

    @property
    def flat_case_indices(self) -> Array:
        """Flat case index for each valid time observation."""
        return self._flat_case_indices

    @property
    def flat_time_indices(self) -> Array:
        """Flat local time index for each valid time observation."""
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
        """Return the unary data or time factor for `label`."""
        if label == self._data_label:
            return self._data_factor
        if label == self._time_label:
            return self._time_factor
        raise KeyError(f"Label {label!r} not in domain {self.labels}.")

    def equivalent(self, other: object, /) -> bool:
        """Return whether another domain has the same public trajectory shape."""
        if not isinstance(other, TrajectoryDatasetDomain):
            return False
        if self.labels != other.labels:
            return False
        if self.measure_mode != other.measure_mode:
            return False
        if self.sampling_mode != other.sampling_mode:
            return False
        if self.max_length != other.max_length:
            return False
        if self.size != other.size:
            return False
        if bool(jnp.any(self.lengths != other.lengths)):
            return False
        if bool(jnp.any(self.dt != other.dt)):
            return False
        if bool(jnp.any(self.start != other.start)):
            return False
        if self.inputs_tree_structure() != other.inputs_tree_structure():
            return False
        leaves_a = jax.tree_util.tree_leaves(self.inputs)
        leaves_b = jax.tree_util.tree_leaves(other.inputs)
        for a, b in zip(leaves_a, leaves_b, strict=True):
            arr_a = jnp.asarray(a)
            arr_b = jnp.asarray(b)
            if arr_a.shape != arr_b.shape:
                return False
            if arr_a.dtype != arr_b.dtype:
                return False
        return True

    def inputs_tree_structure(self) -> Any:
        """Return the PyTree structure of the stored per-case inputs."""
        return jax.tree_util.tree_structure(self.inputs)

    def input_rows(self, case_indices: ArrayLike, /) -> PyTree[Array]:
        """Return input rows for explicit case indices."""
        idx = jnp.asarray(case_indices, dtype=jnp.int32)
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a)[idx], self.inputs)

    def observation_times(
        self, case_indices: ArrayLike, time_indices: ArrayLike, /
    ) -> Array:
        """Convert local time indices to physical times."""
        del case_indices
        return self.start + self.dt * jnp.asarray(time_indices, dtype=float)

    def points_from_case_time(
        self,
        case_indices: ArrayLike,
        times: ArrayLike,
        /,
        *,
        structure: ProductStructure | None = None,
        time_indices: ArrayLike | None = None,
    ) -> PointsBatch:
        """Materialize paired case-time samples as a `PointsBatch`."""
        structure_in = structure or ProductStructure((self.labels,))
        structure_, axis = _single_axis_for_trajectory(self, structure_in)
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
        time_arr = jnp.asarray(times, dtype=float).reshape((-1,))
        if int(case_idx.shape[0]) != int(time_arr.shape[0]):
            raise ValueError("case_indices and times must have the same length.")
        if time_indices is None:
            time_idx = jnp.rint((time_arr - self.start) / self.dt).astype(jnp.int32)
        else:
            time_idx = jnp.asarray(time_indices, dtype=jnp.int32).reshape((-1,))
            if int(time_idx.shape[0]) != int(case_idx.shape[0]):
                raise ValueError(
                    "time_indices must have the same length as case_indices."
                )

        data_samples = self.input_rows(case_idx)

        def _to_field(v: ArrayLike):
            arr = jnp.asarray(v)
            if arr.ndim == 0:
                raise ValueError(
                    "Trajectory input rows must retain a leading sample axis."
                )
            return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

        points: dict[str, Any] = {
            self._data_label: jax.tree_util.tree_map(_to_field, data_samples),
            self._time_label: cx.Field(time_arr, dims=(axis,)),
            TRAJECTORY_CASE_INDEX_KEY: cx.Field(case_idx, dims=(axis,)),
            TRAJECTORY_TIME_INDEX_KEY: cx.Field(time_idx, dims=(axis,)),
        }
        return PointsBatch(points=frozendict(points), structure=structure_)


def _flat_observation_indices(domain: TrajectoryDatasetDomain, /) -> tuple[Array, Array]:
    return domain.flat_case_indices, domain.flat_time_indices


def _sample_cases_uniform(
    domain: TrajectoryDatasetDomain, n: int, key: Key[Array, ""], /
) -> Array:
    return jr.randint(key, shape=(n,), minval=0, maxval=domain.size, dtype=jnp.int32)


def _sample_valid_cases(
    valid: Array,
    n: int,
    key: Key[Array, ""],
    /,
) -> Array:
    valid_f = jnp.asarray(valid, dtype=float)
    valid_count = jnp.sum(valid_f)
    checked_count = eqx.error_if(
        valid_count,
        valid_count <= 0,
        "No trajectories are valid for this fixed time component.",
    )
    probs = valid_f / checked_count
    return jr.choice(key, int(valid.shape[0]), shape=(n,), p=probs).astype(jnp.int32)


def _component_times(
    domain: TrajectoryDatasetDomain,
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
            flat_cases, flat_times = _flat_observation_indices(domain)
            obs_idx = jr.randint(
                key,
                shape=(n,),
                minval=0,
                maxval=domain.total_observations,
                dtype=jnp.int32,
            )
            time_idx = flat_times[obs_idx]
            return domain.observation_times(flat_cases[obs_idx], time_idx), time_idx
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

    raise TypeError(f"Unsupported trajectory time component {type(comp).__name__}.")


def sample_trajectory_component(
    component,
    num_points: NumPoints,
    *,
    structure: ProductStructure,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> PointsBatch:
    del sampler
    domain = component.domain
    if not isinstance(domain, TrajectoryDatasetDomain):
        raise TypeError("sample_trajectory_component requires a TrajectoryDatasetDomain.")

    data_comp = component.spec.component_for(domain.data_label)
    if not isinstance(data_comp, Interior):
        raise TypeError(
            "TrajectoryDatasetDomain supports only Interior() for the data label."
        )

    n = _as_sample_count(num_points)
    structure_, _axis = _single_axis_for_trajectory(domain, structure)
    if n == 0:
        return domain.points_from_case_time(
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=float),
            structure=structure_,
            time_indices=jnp.zeros((0,), dtype=jnp.int32),
        )

    case_key, time_key = jr.split(key)
    time_comp = component.spec.component_for(domain.time_label)
    if isinstance(time_comp, Fixed):
        fixed_value = jnp.asarray(time_comp.value, dtype=float).reshape(())
        valid = (domain.start <= fixed_value) & (fixed_value <= domain.end_times)
        case_indices = _sample_valid_cases(valid, n, case_key)
    elif domain.sampling_mode == "observation_uniform" and isinstance(
        time_comp, Interior
    ):
        flat_cases, flat_times = _flat_observation_indices(domain)
        obs_idx = jr.randint(
            time_key,
            shape=(n,),
            minval=0,
            maxval=domain.total_observations,
            dtype=jnp.int32,
        )
        case_indices = flat_cases[obs_idx]
        times = domain.observation_times(case_indices, flat_times[obs_idx])
        return domain.points_from_case_time(
            case_indices,
            times,
            structure=structure_,
            time_indices=flat_times[obs_idx],
        )
    else:
        case_indices = _sample_cases_uniform(domain, n, case_key)

    times, time_indices = _component_times(domain, component, case_indices, n, time_key)
    return domain.points_from_case_time(
        case_indices,
        times,
        structure=structure_,
        time_indices=time_indices,
    )


def trajectory_default_quadrature_total_weight(
    component, batch: PointsBatch, /
) -> cx.Field | None:
    domain = component.domain
    if not isinstance(domain, TrajectoryDatasetDomain):
        return None
    structure, axis = _single_axis_for_trajectory(domain, batch.structure)
    del structure
    if TRAJECTORY_CASE_INDEX_KEY not in batch:
        return None
    case_field = batch[TRAJECTORY_CASE_INDEX_KEY]
    if not isinstance(case_field, cx.Field):
        raise TypeError("Trajectory case indices must be stored as a coordax.Field.")
    case_idx = jnp.asarray(case_field.data, dtype=jnp.int32)
    n = int(case_idx.shape[0])
    if n == 0:
        return cx.Field(jnp.zeros((0,), dtype=float), dims=(axis,))

    time_comp = component.spec.component_for(domain.time_label)
    point_mass = isinstance(time_comp, (FixedStart, FixedEnd, Fixed))
    boundary = isinstance(time_comp, Boundary)
    durations = domain.durations[case_idx]

    if domain.measure_mode == "case_time_probability":
        per_sample = jnp.ones((n,), dtype=float)
    elif point_mass:
        per_sample = jnp.ones((n,), dtype=float)
    elif boundary:
        per_sample = jnp.full((n,), 2.0, dtype=float)
    else:
        per_sample = durations

    if domain.measure_mode == "time_integral_sum":
        per_sample = per_sample * float(domain.size)

    return cx.Field(per_sample / float(n), dims=(axis,))


def trajectory_component_measure(component, /) -> Array | None:
    domain = component.domain
    if not isinstance(domain, TrajectoryDatasetDomain):
        return None

    time_comp = component.spec.component_for(domain.time_label)
    point_mass = isinstance(time_comp, (FixedStart, FixedEnd, Fixed))
    boundary = isinstance(time_comp, Boundary)

    if domain.measure_mode == "case_time_probability":
        return jnp.asarray(1.0, dtype=float)

    if point_mass:
        measure = jnp.asarray(1.0, dtype=float)
    elif boundary:
        measure = jnp.asarray(2.0, dtype=float)
    else:
        measure = jnp.mean(domain.durations)

    if domain.measure_mode == "time_integral_sum":
        measure = measure * float(domain.size)
    return jnp.asarray(measure, dtype=float)


def trajectory_quadrature_weights_by_axis(
    component, batch: PointsBatch, /
) -> Mapping[str, cx.Field] | None:
    domain = component.domain
    if not isinstance(domain, TrajectoryDatasetDomain):
        return None
    _structure, axis = _single_axis_for_trajectory(domain, batch.structure)
    w = trajectory_default_quadrature_total_weight(component, batch)
    if w is None:
        return None
    return {axis: w}


__all__ = [
    "TRAJECTORY_CASE_INDEX_KEY",
    "TRAJECTORY_TIME_INDEX_KEY",
    "TrajectoryDatasetDomain",
    "TrajectoryMeasure",
    "TrajectorySampling",
    "sample_trajectory_component",
    "trajectory_component_measure",
    "trajectory_default_quadrature_total_weight",
    "trajectory_quadrature_weights_by_axis",
]

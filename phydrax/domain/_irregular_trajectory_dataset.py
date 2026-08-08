#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from ._coordinate import CoordinateSpec
from ._dataset import DatasetDomain
from ._domain import JointFactor
from ._factor_component import FactorComponent
from ._scalar import ScalarInterval
from ._selection import Boundary, Fixed, FixedEnd, FixedStart, Interior, Selection
from ._structure import _validate_label, NumPoints, PointBatch, SampleLayout
from ._trajectory_dataset import (
    _bind_trajectory_component,
    TRAJECTORY_CASE_INDEX_KEY,
    TRAJECTORY_TIME_INDEX_KEY,
    TrajectoryMeasure,
    TrajectorySampling,
)


if TYPE_CHECKING:
    from ._function import DomainFunction


def _tree_leading_axis_size(tree: PyTree[ArrayLike], /) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(
            "IrregularTrajectoryDatasetDomain requires at least one input leaf."
        )

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
                "IrregularTrajectoryDatasetDomain requires all input leaves to share "
                f"the same leading axis; got {int(arr.shape[0])} and {n}."
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


def _as_times(times: ArrayLike, lengths: Array, n: int, /) -> Array:
    arr = jnp.asarray(times, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"times must have shape (N, T_max), got {arr.shape}.")
    if int(arr.shape[0]) != n:
        raise ValueError(f"times leading axis must be N={n}, got {arr.shape[0]}.")
    max_length = int(jnp.max(lengths))
    if int(arr.shape[1]) < max_length:
        raise ValueError(
            f"times second axis must be at least max(lengths)={max_length}, "
            f"got {arr.shape[1]}."
        )

    for case_id, length in enumerate(list(map(int, lengths.tolist()))):
        valid = arr[case_id, :length]
        if not bool(jnp.all(jnp.isfinite(valid))):
            raise ValueError(f"times for case {case_id} contain non-finite values.")
        if length > 1 and not bool(jnp.all(jnp.diff(valid) > 0.0)):
            raise ValueError(
                f"times for case {case_id} must be strictly increasing on the valid segment."
            )
    return arr


def _as_sample_count(num_points: NumPoints, /) -> int:
    if isinstance(num_points, int):
        n = int(num_points)
    else:
        if len(num_points) != 1:
            raise ValueError(
                "IrregularTrajectoryDatasetDomain requires paired sampling with one sample count."
            )
        n = int(num_points[0])
    if n < 0:
        raise ValueError("num_points must be non-negative.")
    return n


def _single_axis_for_trajectory(
    domain: "IrregularTrajectoryDatasetDomain",
    structure: SampleLayout,
    /,
) -> tuple[SampleLayout, str]:
    structure = structure.canonicalize(domain.labels, fixed_labels=frozenset())
    if len(structure.blocks) != 1:
        raise ValueError(
            "IrregularTrajectoryDatasetDomain sampling requires the data and time "
            "labels to be sampled in one paired block."
        )
    block = frozenset(structure.blocks[0])
    if block != frozenset(domain.labels):
        raise ValueError(
            "IrregularTrajectoryDatasetDomain sampling requires a paired block "
            f"containing {domain.labels}."
        )
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("Irregular trajectory SampleLayout must be canonicalized.")
    return structure, axis_names[0]


class IrregularTrajectoryDatasetDomain(JointFactor):
    """A coupled finite-function and irregular time domain for ragged trajectories.

    Each dataset row owns explicit strictly increasing observation times. Sampling
    keeps `data` and `t` paired, so every sampled time is valid for the selected
    dataset row even when cases have different non-uniform time grids.
    """

    inputs: PyTree[Array]
    times: Array
    lengths: Array
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
    _node_widths: Array

    def __init__(
        self,
        inputs: PyTree[ArrayLike],
        times: ArrayLike,
        lengths: ArrayLike,
        /,
        *,
        data_label: str = "data",
        time_label: str = "t",
        measure: TrajectoryMeasure = "case_time_probability",
        sampling: TrajectorySampling = "case_time_uniform",
    ):
        """Create a finite dataset of row-conditioned irregular trajectories.

        Parameters:
            inputs: Per-case input PyTree with a shared leading case axis.
            times: Padded physical time table with shape `(cases, max_length)`.
            lengths: Valid time-step count for each case.
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
        times_arr = _as_times(times, lengths_arr, n)

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
        if measure_str == "case_time_probability":
            measure_value: TrajectoryMeasure = "case_time_probability"
        elif measure_str == "time_integral_average":
            measure_value = "time_integral_average"
        else:
            measure_value = "time_integral_sum"

        sampling_str = str(sampling)
        if sampling_str not in ("case_time_uniform", "observation_uniform"):
            raise ValueError(
                "sampling must be either 'case_time_uniform' or 'observation_uniform'."
            )
        if sampling_str == "case_time_uniform":
            sampling_value: TrajectorySampling = "case_time_uniform"
        else:
            sampling_value = "observation_uniform"

        max_length = int(jnp.max(lengths_arr))
        total_observations = int(jnp.sum(lengths_arr))
        flat_case_parts: list[Array] = []
        flat_time_parts: list[Array] = []
        for i, length in enumerate(list(map(int, lengths_arr.tolist()))):
            flat_case_parts.append(jnp.full((length,), i, dtype=jnp.int32))
            flat_time_parts.append(jnp.arange(length, dtype=jnp.int32))

        start_times = times_arr[jnp.arange(n), 0]
        end_times = times_arr[jnp.arange(n), lengths_arr - 1]
        t_min = float(jnp.min(start_times))
        t_max = float(jnp.max(end_times))
        if t_max <= t_min:
            t_max = t_min + 1.0

        self.inputs = arrays
        self.times = times_arr
        self.lengths = lengths_arr
        self._data_label = str(data_label)
        self._time_label = str(time_label)
        self._measure = measure_value
        self._sampling = sampling_value
        self._data_factor = DatasetDomain(
            arrays, label=str(data_label), measure="probability"
        )
        self._time_factor = ScalarInterval(t_min, t_max, label=str(time_label))
        self._max_length = max_length
        self._total_observations = total_observations
        self._flat_case_indices = jnp.concatenate(flat_case_parts, axis=0)
        self._flat_time_indices = jnp.concatenate(flat_time_parts, axis=0)
        self._node_widths = _node_widths(times_arr, lengths_arr)

    @property
    def labels(self) -> tuple[str, ...]:
        """Data and time labels owned by the domain."""
        return (self._data_label, self._time_label)

    @property
    def coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        return (
            CoordinateSpec(None, kind="pytree", differentiable=False, dtype=None),
            CoordinateSpec((), kind="scalar", differentiable=True),
        )

    def bind_component(
        self,
        selections: Mapping[str, Selection],
        /,
    ) -> FactorComponent:
        return _bind_trajectory_component(self, selections)

    def _replace_labels(
        self,
        labels: tuple[str, ...],
        /,
    ) -> "IrregularTrajectoryDatasetDomain":
        updated = eqx.tree_at(
            lambda factor: (factor._data_label, factor._time_label),
            self,
            labels,
        )
        return eqx.tree_at(
            lambda factor: (factor._data_factor, factor._time_factor),
            updated,
            (
                updated._data_factor.relabel(labels[0]),
                updated._time_factor.relabel(labels[1]),
            ),
        )

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

    def field(self, values: ArrayLike, /) -> "DomainFunction":
        """Expose case-aligned target values at every sampled trajectory time."""
        from ._observation import indexed_field

        return indexed_field(
            self,
            values,
            size=self.size,
            index_key=TRAJECTORY_CASE_INDEX_KEY,
            owner="IrregularTrajectoryDatasetDomain.field",
        )

    def signal(
        self,
        values: ArrayLike,
        /,
        *,
        interpolation: Literal["nearest", "linear", "cubic_hermite"] = "linear",
    ) -> "DomainFunction":
        """Expose padded trajectory observations as an interpolated field."""
        from ._trajectory_signal import TrajectorySignal

        return TrajectorySignal(self, values, interpolation=interpolation)

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
    def start_times(self) -> Array:
        """Per-case first valid time."""
        return self.times[jnp.arange(self.size), 0]

    @property
    def end_times(self) -> Array:
        """Per-case final valid time."""
        return self.times[jnp.arange(self.size), self.lengths - 1]

    @property
    def durations(self) -> Array:
        """Per-case time span, `end_times - start_times`."""
        return self.end_times - self.start_times

    @property
    def node_widths(self) -> Array:
        """Trapezoid-style quadrature widths for each valid time node."""
        return self._node_widths

    def _same_factor_support(self, other: object, /) -> bool:
        """Return whether another domain has the same public trajectory shape."""
        if not isinstance(other, IrregularTrajectoryDatasetDomain):
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
        if bool(jnp.any(self.times != other.times)):
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
        """Gather physical times by case and local time index."""
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32)
        time_idx = jnp.asarray(time_indices, dtype=jnp.int32)
        return self.times[case_idx, time_idx]

    def lower_time_indices(self, case_indices: ArrayLike, times: ArrayLike, /) -> Array:
        """Return the lower bracketing time index for each requested time."""
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
        t = jnp.asarray(times, dtype=float).reshape((-1,))
        rows = self.times[case_idx]
        lengths = self.lengths[case_idx]
        mask = jnp.arange(self.times.shape[1])[None, :] < lengths[:, None]
        counts = jnp.sum((rows <= t[:, None]) & mask, axis=1)
        return jnp.clip(counts - 1, 0, lengths - 1).astype(jnp.int32)

    def nearest_time_indices(self, case_indices: ArrayLike, times: ArrayLike, /) -> Array:
        """Return the nearest valid time index for each requested time."""
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
        t = jnp.asarray(times, dtype=float).reshape((-1,))
        lower = self.lower_time_indices(case_idx, t)
        lengths = self.lengths[case_idx]
        upper = jnp.minimum(lower + 1, lengths - 1)
        lower_t = self.times[case_idx, lower]
        upper_t = self.times[case_idx, upper]
        use_upper = jnp.abs(upper_t - t) < jnp.abs(t - lower_t)
        return jnp.where(use_upper, upper, lower).astype(jnp.int32)

    def bracketing_time_indices(
        self, case_indices: ArrayLike, times: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        """Return lower index, upper index, and interpolation fraction for times."""
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
        t = jnp.asarray(times, dtype=float).reshape((-1,))
        lengths = self.lengths[case_idx]
        lower_raw = self.lower_time_indices(case_idx, t)
        lower_max = jnp.maximum(lengths - 2, 0)
        lower = jnp.minimum(lower_raw, lower_max)
        upper = jnp.minimum(lower + 1, lengths - 1)
        t0 = self.times[case_idx, lower]
        t1 = self.times[case_idx, upper]
        denom = jnp.where(lengths > 1, t1 - t0, 1.0)
        frac = jnp.where(lengths > 1, (t - t0) / denom, 0.0)
        frac = jnp.clip(frac, 0.0, 1.0)
        return lower.astype(jnp.int32), upper.astype(jnp.int32), frac

    def points_from_case_time(
        self,
        case_indices: ArrayLike,
        times: ArrayLike,
        /,
        *,
        structure: SampleLayout | None = None,
        time_indices: ArrayLike | None = None,
    ) -> PointBatch:
        """Materialize paired case-time samples as a `PointBatch`."""
        structure_in = structure or SampleLayout((self.labels,))
        structure_, axis = _single_axis_for_trajectory(self, structure_in)
        case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
        time_arr = jnp.asarray(times, dtype=float).reshape((-1,))
        if int(case_idx.shape[0]) != int(time_arr.shape[0]):
            raise ValueError("case_indices and times must have the same length.")
        if time_indices is None:
            time_idx = self.lower_time_indices(case_idx, time_arr)
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
        return PointBatch(points=frozendict(points), structure=structure_)


def _node_widths(times: Array, lengths: Array, /) -> Array:
    n = int(times.shape[0])
    t_max = int(times.shape[1])
    widths = jnp.zeros_like(times)
    for case_id, length in enumerate(list(map(int, lengths.tolist()))):
        if length <= 1:
            continue
        row = times[case_id, :length]
        diffs = jnp.diff(row)
        w = jnp.zeros((length,), dtype=times.dtype)
        w = w.at[0].set(0.5 * diffs[0])
        w = w.at[length - 1].set(0.5 * diffs[-1])
        if length > 2:
            interior = 0.5 * (diffs[:-1] + diffs[1:])
            w = w.at[1 : length - 1].set(interior)
        widths = widths.at[case_id, :length].set(w)
    return widths.reshape((n, t_max))


def _flat_observation_indices(
    domain: IrregularTrajectoryDatasetDomain, /
) -> tuple[Array, Array]:
    return domain.flat_case_indices, domain.flat_time_indices


def _sample_cases_uniform(
    domain: IrregularTrajectoryDatasetDomain, n: int, key: Key[Array, ""], /
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
    domain: IrregularTrajectoryDatasetDomain,
    component,
    case_indices: Array,
    n: int,
    key: Key[Array, ""],
    /,
) -> tuple[Array, Array]:
    comp = component.spec.selection_for(domain.time_label)
    lengths = domain.lengths[case_indices]
    start_times = domain.start_times[case_indices]
    end_times = domain.end_times[case_indices]
    durations = end_times - start_times

    if isinstance(comp, Interior):
        u = jr.uniform(key, shape=(n,))
        times = start_times + u * durations
        time_idx = domain.lower_time_indices(case_indices, times)
        return times, time_idx

    if isinstance(comp, FixedStart):
        return start_times, jnp.zeros((n,), dtype=jnp.int32)

    if isinstance(comp, FixedEnd):
        return end_times, lengths - 1

    if isinstance(comp, Fixed):
        value = jnp.asarray(comp.value, dtype=float).reshape(())
        times = jnp.full((n,), value, dtype=float)
        return times, domain.lower_time_indices(case_indices, times)

    if isinstance(comp, Boundary):
        pick_end = jr.bernoulli(key, p=0.5, shape=(n,))
        time_idx = jnp.where(pick_end, lengths - 1, 0).astype(jnp.int32)
        times = jnp.where(pick_end, end_times, start_times)
        return times, time_idx

    raise TypeError(
        f"Unsupported irregular trajectory time component {type(comp).__name__}."
    )


def sample_irregular_trajectory_component(
    component,
    num_points: NumPoints,
    *,
    structure: SampleLayout,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> PointBatch:
    del sampler
    domain = component.domain
    if not isinstance(domain, IrregularTrajectoryDatasetDomain):
        raise TypeError(
            "sample_irregular_trajectory_component requires an "
            "IrregularTrajectoryDatasetDomain."
        )

    data_comp = component.spec.selection_for(domain.data_label)
    if not isinstance(data_comp, Interior):
        raise TypeError(
            "IrregularTrajectoryDatasetDomain supports only Interior() for the data label."
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
    time_comp = component.spec.selection_for(domain.time_label)
    if isinstance(time_comp, Fixed):
        fixed_value = jnp.asarray(time_comp.value, dtype=float).reshape(())
        valid = (domain.start_times <= fixed_value) & (fixed_value <= domain.end_times)
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
        time_indices = flat_times[obs_idx]
        times = domain.observation_times(case_indices, time_indices)
        return domain.points_from_case_time(
            case_indices,
            times,
            structure=structure_,
            time_indices=time_indices,
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


def irregular_trajectory_default_quadrature_total_weight(
    component, batch: PointBatch, /
) -> cx.Field | None:
    domain = component.domain
    if not isinstance(domain, IrregularTrajectoryDatasetDomain):
        return None
    structure, axis = _single_axis_for_trajectory(domain, batch.structure)
    del structure
    if TRAJECTORY_CASE_INDEX_KEY not in batch:
        return None
    case_field = batch[TRAJECTORY_CASE_INDEX_KEY]
    time_field = batch[TRAJECTORY_TIME_INDEX_KEY]
    if not isinstance(case_field, cx.Field):
        raise TypeError("Trajectory case indices must be stored as a coordax.Field.")
    if not isinstance(time_field, cx.Field):
        raise TypeError("Trajectory time indices must be stored as a coordax.Field.")
    case_idx = jnp.asarray(case_field.data, dtype=jnp.int32)
    time_idx = jnp.asarray(time_field.data, dtype=jnp.int32)
    n = int(case_idx.shape[0])
    if n == 0:
        return cx.Field(jnp.zeros((0,), dtype=float), dims=(axis,))

    time_comp = component.spec.selection_for(domain.time_label)
    point_mass = isinstance(time_comp, (FixedStart, FixedEnd, Fixed))
    boundary = isinstance(time_comp, Boundary)
    durations = domain.durations[case_idx]

    if domain.measure_mode == "case_time_probability":
        per_sample = jnp.ones((n,), dtype=float)
    elif point_mass:
        per_sample = jnp.ones((n,), dtype=float)
    elif boundary:
        per_sample = jnp.full((n,), 2.0, dtype=float)
    elif domain.sampling_mode == "observation_uniform" and isinstance(
        time_comp, Interior
    ):
        node_width = domain.node_widths[case_idx, time_idx]
        scale = float(domain.total_observations)
        if domain.measure_mode == "time_integral_average":
            scale = scale / float(domain.size)
        per_sample = scale * node_width
    else:
        per_sample = durations

    if domain.measure_mode == "time_integral_sum" and not (
        domain.sampling_mode == "observation_uniform" and isinstance(time_comp, Interior)
    ):
        per_sample = per_sample * float(domain.size)

    return cx.Field(per_sample / float(n), dims=(axis,))


def irregular_trajectory_quadrature_weights_by_axis(
    component, batch: PointBatch, /
) -> dict[str, cx.Field] | None:
    domain = component.domain
    if not isinstance(domain, IrregularTrajectoryDatasetDomain):
        return None
    _structure, axis = _single_axis_for_trajectory(domain, batch.structure)
    w = irregular_trajectory_default_quadrature_total_weight(component, batch)
    if w is None:
        return None
    return {axis: w}


__all__ = [
    "IrregularTrajectoryDatasetDomain",
    "sample_irregular_trajectory_component",
    "irregular_trajectory_default_quadrature_total_weight",
    "irregular_trajectory_quadrature_weights_by_axis",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._numerics import normalize_least_squares_design
from ..._strict import AbstractAttribute, StrictModule
from .._layout import InputLayout, StateLayout
from .._trajectory import TrajectoryData
from ._features import AbstractFeatureLibrary


SINDyFormulationKind: TypeAlias = Literal[
    "strong", "discrete", "integral", "weak", "implicit", "pde"
]
WindowBoundary: TypeAlias = Literal["drop", "partial"]
WindowQuadrature: TypeAlias = Literal["left", "trapezoid"]


class SINDyDesignDiagnostics(StrictModule):
    """Pre-fit rank and conditioning evidence for one equation design."""

    singular_values: Array
    sample_count: Array
    rank: Array
    condition_number: Array
    weight_sum: Array


class SINDyDesign(StrictModule):
    """A reusable masked equation design, separate from sparse regression."""

    matrix: Array
    target: Array
    valid: Array
    weights: Array
    coordinates: Array
    case_index: Array
    window_start: Array
    window_end: Array
    diagnostics: SINDyDesignDiagnostics
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    formulation: SINDyFormulationKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    library_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    design_id: str = eqx.field(static=True)

    @property
    def num_rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.target.shape[1])


class AbstractSINDyFormulation(StrictModule):
    """Static policy lowering trajectory data and a library into equations."""

    formulation: AbstractAttribute[SINDyFormulationKind]
    formulation_id: AbstractAttribute[str]

    @abc.abstractmethod
    def build(
        self,
        data: TrajectoryData,
        library: AbstractFeatureLibrary,
        /,
    ) -> SINDyDesign:
        raise NotImplementedError


class SINDyProblem(StrictModule):
    """Trajectory, feature dictionary, and equation formulation before fitting."""

    data: TrajectoryData
    library: AbstractFeatureLibrary
    formulation: AbstractSINDyFormulation

    def __init__(
        self,
        *,
        data: TrajectoryData,
        library: AbstractFeatureLibrary,
        formulation: AbstractSINDyFormulation,
    ):
        if not isinstance(data, TrajectoryData):
            raise TypeError("data must be TrajectoryData.")
        if not isinstance(library, AbstractFeatureLibrary):
            raise TypeError("library must be an AbstractFeatureLibrary.")
        if not isinstance(formulation, AbstractSINDyFormulation):
            raise TypeError("formulation must be an AbstractSINDyFormulation.")
        _validate_library(data, library)
        self.data = data
        self.library = library
        self.formulation = formulation

    def build_design(self) -> SINDyDesign:
        return self.formulation.build(self.data, self.library)


def build_sindy_design(problem: SINDyProblem, /) -> SINDyDesign:
    """Build a reusable design without selecting or fitting a sparse regressor."""
    if not isinstance(problem, SINDyProblem):
        raise TypeError("problem must be a SINDyProblem.")
    return problem.build_design()


def _validate_library(data: TrajectoryData, library: AbstractFeatureLibrary, /) -> None:
    if library.state_layout.layout_id != data.state_layout.layout_id:
        raise ValueError("Feature library and trajectory must use the same state layout.")
    expected_input = None if data.input_layout is None else data.input_layout.layout_id
    actual_input = (
        None if library.input_layout is None else library.input_layout.layout_id
    )
    if expected_input != actual_input:
        raise ValueError("Feature library and trajectory must use the same input layout.")


def _time_values(data: TrajectoryData, values: Array, index, /) -> Array:
    return values[(slice(None),) * len(data.case_shape) + (index,)]


def _flatten_state(values: Array, layout: StateLayout, /) -> Array:
    if not layout.shape:
        return values[..., None]
    prefix = values.shape[: values.ndim - len(layout.shape)]
    return values.reshape(prefix + (layout.size,))


def _row_metadata(case_count: int, starts: Sequence[int], ends: Sequence[int], /):
    row_count = len(starts)
    return (
        jnp.repeat(jnp.arange(case_count, dtype=jnp.int32), row_count),
        jnp.tile(jnp.asarray(starts, dtype=jnp.int32), case_count),
        jnp.tile(jnp.asarray(ends, dtype=jnp.int32), case_count),
    )


def _make_equation_design(
    *,
    matrix: Array,
    target: Array,
    valid: Array,
    weights: Array,
    coordinates: Array,
    case_index: Array,
    window_start: Array,
    window_end: Array,
    state_layout: StateLayout,
    input_layout: InputLayout | None,
    feature_names: tuple[str, ...],
    output_names: tuple[str, ...],
    formulation: SINDyFormulationKind,
    source_id: str,
    coordinate_id: str,
    library_id: str,
    formulation_id: str,
) -> SINDyDesign:
    matrix_values = matrix.reshape((-1, len(feature_names)))
    target_values = target.reshape((-1, len(output_names)))
    valid_values = valid.reshape((-1,))
    weight_values = jnp.where(valid_values, weights.reshape((-1,)), 0.0)
    matrix_values = jnp.where(valid_values[:, None], matrix_values, 0.0)
    target_values = jnp.where(valid_values[:, None], target_values, 0.0)
    normalized = normalize_least_squares_design(
        matrix_values,
        mask=valid_values,
        weights=weight_values,
        max_features=len(feature_names),
    )
    design_id = (
        "sindy-design:"
        f"{canonical_fingerprint({'source': source_id, 'library': library_id, 'formulation': formulation_id})}"
    )
    return SINDyDesign(
        matrix=matrix_values,
        target=target_values,
        valid=valid_values,
        weights=weight_values,
        coordinates=coordinates.reshape((-1,)),
        case_index=case_index.reshape((-1,)),
        window_start=window_start.reshape((-1,)),
        window_end=window_end.reshape((-1,)),
        diagnostics=SINDyDesignDiagnostics(
            singular_values=normalized.singular_values,
            sample_count=normalized.sample_count,
            rank=normalized.rank,
            condition_number=normalized.condition_number,
            weight_sum=normalized.weight_sum,
        ),
        state_layout=state_layout,
        input_layout=input_layout,
        feature_names=feature_names,
        output_names=output_names,
        formulation=formulation,
        source_id=source_id,
        coordinate_id=coordinate_id,
        library_id=library_id,
        formulation_id=formulation_id,
        design_id=design_id,
    )


def _make_design(
    *,
    matrix: Array,
    target: Array,
    valid: Array,
    weights: Array,
    coordinates: Array,
    case_index: Array,
    window_start: Array,
    window_end: Array,
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    formulation: SINDyFormulationKind,
    formulation_id: str,
) -> SINDyDesign:
    return _make_equation_design(
        matrix=matrix,
        target=target,
        valid=valid,
        weights=weights,
        coordinates=coordinates,
        case_index=case_index,
        window_start=window_start,
        window_end=window_end,
        state_layout=data.state_layout,
        input_layout=data.input_layout,
        feature_names=library.feature_names,
        output_names=data.state_layout.component_names,
        formulation=formulation,
        source_id=data.source_id,
        coordinate_id=data.coordinate_id,
        library_id=library.library_id,
        formulation_id=formulation_id,
    )


def _required_input_valid(data: TrajectoryData, /) -> Array:
    if data.input_valid is None:
        raise RuntimeError("Trajectory inputs are missing their validity mask.")
    return data.input_valid


def _sample_inputs(data: TrajectoryData, count: int, /):
    if data.inputs is None:
        return None, jnp.ones(data.case_shape + (count,), dtype=bool)
    return (
        _time_values(data, data.inputs, slice(0, count)),
        _required_input_valid(data)[..., :count],
    )


class StrongSINDyFormulation(AbstractSINDyFormulation):
    """Pointwise derivative equations from explicitly attached derivatives."""

    formulation: SINDyFormulationKind = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(self):
        self.formulation = "strong"
        self.formulation_id = "strong:pointwise-derivative"

    def build(
        self, data: TrajectoryData, library: AbstractFeatureLibrary, /
    ) -> SINDyDesign:
        _validate_library(data, library)
        if data.derivatives is None or data.derivative_valid is None:
            raise ValueError(
                "Strong SINDy requires derivatives with an explicit derivative_valid mask."
            )
        count = (
            data.capacity
            if data.inputs is None or data.input_alignment == "samples"
            else data.capacity - 1
        )
        states = _time_values(data, data.states, slice(0, count))
        inputs, input_valid = _sample_inputs(data, count)
        evaluation = library.evaluate(states, inputs)
        valid = (
            data.sample_valid[..., :count]
            & data.derivative_valid[..., :count]
            & input_valid
            & evaluation.valid
        )
        target = _flatten_state(
            _time_values(data, data.derivatives, slice(0, count)),
            data.state_layout,
        )
        case_index, starts, ends = _row_metadata(
            data.num_cases, tuple(range(count)), tuple(range(count))
        )
        return _make_design(
            matrix=evaluation.values,
            target=target,
            valid=valid,
            weights=data.weights[..., :count],
            coordinates=data.coordinates[..., :count],
            case_index=case_index,
            window_start=starts,
            window_end=ends,
            data=data,
            library=library,
            formulation=self.formulation,
            formulation_id=self.formulation_id,
        )


class DiscreteSINDyFormulation(AbstractSINDyFormulation):
    """Direct next-state map equations with a declared integer lag."""

    formulation: SINDyFormulationKind = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    lag: int = eqx.field(static=True)

    def __init__(self, *, lag: int = 1):
        resolved_lag = int(lag)
        if resolved_lag < 1:
            raise ValueError("lag must be positive.")
        self.formulation = "discrete"
        self.lag = resolved_lag
        self.formulation_id = f"discrete:next-state:lag={resolved_lag}"

    def build(
        self, data: TrajectoryData, library: AbstractFeatureLibrary, /
    ) -> SINDyDesign:
        _validate_library(data, library)
        transitions = data.transitions(self.lag)
        evaluation = library.evaluate(transitions.source_states, transitions.inputs)
        valid = transitions.valid & evaluation.valid
        target = _flatten_state(transitions.target_states, data.state_layout)
        count = data.capacity - self.lag
        case_index, starts, ends = _row_metadata(
            data.num_cases,
            tuple(range(count)),
            tuple(range(self.lag, data.capacity)),
        )
        return _make_design(
            matrix=evaluation.values,
            target=target,
            valid=valid,
            weights=transitions.weights,
            coordinates=transitions.source_coordinates,
            case_index=case_index,
            window_start=starts,
            window_end=ends,
            data=data,
            library=library,
            formulation=self.formulation,
            formulation_id=self.formulation_id,
        )


def _window_endpoints(
    capacity: int,
    window_size: int,
    stride: int,
    boundary: WindowBoundary,
    /,
) -> tuple[tuple[int, int], ...]:
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive.")
    starts = range(0, capacity - 1, stride)
    if boundary == "drop":
        windows = tuple(
            (start, start + window_size)
            for start in starts
            if start + window_size < capacity
        )
    elif boundary == "partial":
        windows = tuple(
            (start, min(start + window_size, capacity - 1)) for start in starts
        )
    else:
        raise ValueError("boundary must be 'drop' or 'partial'.")
    if not windows:
        raise ValueError(
            "Window policy produces no equations for this trajectory capacity."
        )
    return windows


def _interval_features(
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    interval: int,
    /,
):
    source_state = _time_values(data, data.states, interval)
    target_state = _time_values(data, data.states, interval + 1)
    if data.inputs is None:
        source_input = None
        target_input = None
        input_valid = jnp.ones(data.case_shape, dtype=bool)
    elif data.input_alignment == "samples":
        source_input = _time_values(data, data.inputs, interval)
        target_input = _time_values(data, data.inputs, interval + 1)
        input_valid = (
            _required_input_valid(data)[..., interval]
            & _required_input_valid(data)[..., interval + 1]
        )
    else:
        source_input = _time_values(data, data.inputs, interval)
        target_input = source_input
        input_valid = _required_input_valid(data)[..., interval]
    source = library.evaluate(source_state, source_input)
    target = library.evaluate(target_state, target_input)
    valid = (
        data.transition_valid[..., interval] & input_valid & source.valid & target.valid
    )
    return source.values, target.values, valid


def _window_feature_integral(
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    start: int,
    end: int,
    /,
    *,
    quadrature: WindowQuadrature,
    test_order: int | None,
):
    duration = data.coordinates[..., end] - data.coordinates[..., start]
    safe_duration = jnp.where(duration > 0.0, duration, 1.0)
    integral = jnp.zeros(data.case_shape + (library.num_features,))
    target_integral = jnp.zeros(data.case_shape + (data.state_layout.size,))
    valid = jnp.ones(data.case_shape, dtype=bool)
    equation_weight = jnp.zeros(data.case_shape, dtype=data.coordinates.dtype)
    for interval in range(start, end):
        left, right, interval_valid = _interval_features(data, library, interval)
        left_time = data.coordinates[..., interval]
        right_time = data.coordinates[..., interval + 1]
        dt = right_time - left_time
        if test_order is None:
            left_test = jnp.ones_like(dt)
            right_test = jnp.ones_like(dt)
            left_derivative = jnp.zeros_like(dt)
            right_derivative = jnp.zeros_like(dt)
        else:
            left_coordinate = (left_time - data.coordinates[..., start]) / safe_duration
            right_coordinate = (right_time - data.coordinates[..., start]) / safe_duration
            left_base = left_coordinate * (1.0 - left_coordinate)
            right_base = right_coordinate * (1.0 - right_coordinate)
            left_test = left_base**test_order
            right_test = right_base**test_order
            left_derivative = (
                test_order
                * left_base ** (test_order - 1)
                * (1.0 - 2.0 * left_coordinate)
                / safe_duration
            )
            right_derivative = (
                test_order
                * right_base ** (test_order - 1)
                * (1.0 - 2.0 * right_coordinate)
                / safe_duration
            )
        if quadrature == "trapezoid":
            feature_integrand = 0.5 * (
                left_test[..., None] * left + right_test[..., None] * right
            )
            left_state = _flatten_state(
                _time_values(data, data.states, interval),
                data.state_layout,
            )
            right_state = _flatten_state(
                _time_values(data, data.states, interval + 1),
                data.state_layout,
            )
            derivative_integrand = 0.5 * (
                left_derivative[..., None] * left_state
                + right_derivative[..., None] * right_state
            )
        elif quadrature == "left":
            feature_integrand = left_test[..., None] * left
            derivative_integrand = left_derivative[..., None] * _flatten_state(
                _time_values(data, data.states, interval),
                data.state_layout,
            )
        else:
            raise ValueError("quadrature must be 'left' or 'trapezoid'.")
        integral = integral + dt[..., None] * feature_integrand
        target_integral = target_integral - dt[..., None] * derivative_integrand
        valid = valid & interval_valid & (dt > 0.0)
        equation_weight = equation_weight + jnp.sqrt(
            data.weights[..., interval] * data.weights[..., interval + 1]
        )
    equation_weight = equation_weight / float(end - start)
    return integral, target_integral, valid, equation_weight


class IntegralSINDyFormulation(AbstractSINDyFormulation):
    """Windowed integral identities using endpoint state differences."""

    formulation: SINDyFormulationKind = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    quadrature: WindowQuadrature = eqx.field(static=True)
    boundary: WindowBoundary = eqx.field(static=True)

    def __init__(
        self,
        *,
        window_size: int,
        stride: int = 1,
        quadrature: WindowQuadrature = "trapezoid",
        boundary: WindowBoundary = "drop",
    ):
        if quadrature not in ("left", "trapezoid"):
            raise ValueError("quadrature must be 'left' or 'trapezoid'.")
        if boundary not in ("drop", "partial"):
            raise ValueError("boundary must be 'drop' or 'partial'.")
        if int(window_size) < 1 or int(stride) < 1:
            raise ValueError("window_size and stride must be positive.")
        self.formulation = "integral"
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.quadrature = quadrature
        self.boundary = boundary
        self.formulation_id = (
            f"integral:window={self.window_size}:stride={self.stride}:"
            f"quadrature={quadrature}:boundary={boundary}"
        )

    def build(
        self, data: TrajectoryData, library: AbstractFeatureLibrary, /
    ) -> SINDyDesign:
        _validate_library(data, library)
        windows = _window_endpoints(
            data.capacity,
            self.window_size,
            self.stride,
            self.boundary,
        )
        matrices = []
        targets = []
        masks = []
        weights = []
        coordinates = []
        for start, end in windows:
            integral, _, valid, weight = _window_feature_integral(
                data,
                library,
                start,
                end,
                quadrature=self.quadrature,
                test_order=None,
            )
            source = _flatten_state(
                _time_values(data, data.states, start),
                data.state_layout,
            )
            target = _flatten_state(
                _time_values(data, data.states, end),
                data.state_layout,
            )
            matrices.append(integral)
            targets.append(target - source)
            masks.append(valid)
            weights.append(weight)
            coordinates.append(
                0.5 * (data.coordinates[..., start] + data.coordinates[..., end])
            )
        case_axis = len(data.case_shape)
        matrix = jnp.stack(tuple(matrices), axis=case_axis)
        target = jnp.stack(tuple(targets), axis=case_axis)
        valid = jnp.stack(tuple(masks), axis=case_axis)
        weight = jnp.stack(tuple(weights), axis=case_axis)
        coordinate = jnp.stack(tuple(coordinates), axis=case_axis)
        starts = tuple(start for start, _ in windows)
        ends = tuple(end for _, end in windows)
        case_index, row_start, row_end = _row_metadata(data.num_cases, starts, ends)
        return _make_design(
            matrix=matrix,
            target=target,
            valid=valid,
            weights=weight,
            coordinates=coordinate,
            case_index=case_index,
            window_start=row_start,
            window_end=row_end,
            data=data,
            library=library,
            formulation=self.formulation,
            formulation_id=self.formulation_id,
        )


class WeakSINDyFormulation(AbstractSINDyFormulation):
    """Compactly supported polynomial test-function identities on windows."""

    formulation: SINDyFormulationKind = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    test_orders: tuple[int, ...] = eqx.field(static=True)
    quadrature: WindowQuadrature = eqx.field(static=True)
    boundary: WindowBoundary = eqx.field(static=True)

    def __init__(
        self,
        *,
        window_size: int,
        stride: int = 1,
        test_orders: Sequence[int] = (1,),
        quadrature: WindowQuadrature = "trapezoid",
        boundary: WindowBoundary = "drop",
    ):
        orders = tuple(int(order) for order in test_orders)
        if (
            not orders
            or any(order < 1 for order in orders)
            or len(set(orders)) != len(orders)
        ):
            raise ValueError("test_orders must contain unique positive integers.")
        if quadrature not in ("left", "trapezoid"):
            raise ValueError("quadrature must be 'left' or 'trapezoid'.")
        if boundary not in ("drop", "partial"):
            raise ValueError("boundary must be 'drop' or 'partial'.")
        if int(window_size) < 1 or int(stride) < 1:
            raise ValueError("window_size and stride must be positive.")
        self.formulation = "weak"
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.test_orders = orders
        self.quadrature = quadrature
        self.boundary = boundary
        self.formulation_id = (
            f"weak:window={self.window_size}:stride={self.stride}:"
            f"orders={orders}:quadrature={quadrature}:boundary={boundary}"
        )

    def build(
        self, data: TrajectoryData, library: AbstractFeatureLibrary, /
    ) -> SINDyDesign:
        _validate_library(data, library)
        windows = _window_endpoints(
            data.capacity,
            self.window_size,
            self.stride,
            self.boundary,
        )
        matrices = []
        targets = []
        masks = []
        weights = []
        coordinates = []
        starts = []
        ends = []
        for start, end in windows:
            for order in self.test_orders:
                integral, target, valid, weight = _window_feature_integral(
                    data,
                    library,
                    start,
                    end,
                    quadrature=self.quadrature,
                    test_order=order,
                )
                matrices.append(integral)
                targets.append(target)
                masks.append(valid)
                weights.append(weight)
                coordinates.append(
                    0.5 * (data.coordinates[..., start] + data.coordinates[..., end])
                )
                starts.append(start)
                ends.append(end)
        case_axis = len(data.case_shape)
        matrix = jnp.stack(tuple(matrices), axis=case_axis)
        target = jnp.stack(tuple(targets), axis=case_axis)
        valid = jnp.stack(tuple(masks), axis=case_axis)
        weight = jnp.stack(tuple(weights), axis=case_axis)
        coordinate = jnp.stack(tuple(coordinates), axis=case_axis)
        case_index, row_start, row_end = _row_metadata(
            data.num_cases, tuple(starts), tuple(ends)
        )
        return _make_design(
            matrix=matrix,
            target=target,
            valid=valid,
            weights=weight,
            coordinates=coordinate,
            case_index=case_index,
            window_start=row_start,
            window_end=row_end,
            data=data,
            library=library,
            formulation=self.formulation,
            formulation_id=self.formulation_id,
        )


__all__ = [
    "AbstractSINDyFormulation",
    "DiscreteSINDyFormulation",
    "IntegralSINDyFormulation",
    "SINDyDesign",
    "SINDyDesignDiagnostics",
    "SINDyFormulationKind",
    "SINDyProblem",
    "StrongSINDyFormulation",
    "WeakSINDyFormulation",
    "WindowBoundary",
    "WindowQuadrature",
    "build_sindy_design",
]

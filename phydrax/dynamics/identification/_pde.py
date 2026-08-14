#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics import weighted_total_degree_indices
from ..._strict import AbstractAttribute, StrictModule
from .._layout import StateLayout
from ._sindy_design import _make_equation_design, SINDyDesign
from ._sparse_regression import AbstractSparseRegression, SparseRegressionResult


def _shape_tail(values: Array, shape: tuple[int, ...], /) -> tuple[int, ...]:
    return values.shape if not shape else values.shape[: -len(shape)]


def _quadrature_weights(coordinates: tuple[Array, ...], /) -> Array:
    shape = tuple(int(coordinate.size) for coordinate in coordinates)
    result = jnp.ones(shape)
    for axis, coordinate in enumerate(coordinates):
        widths = jnp.empty_like(coordinate)
        widths = widths.at[0].set(0.5 * (coordinate[1] - coordinate[0]))
        widths = widths.at[-1].set(0.5 * (coordinate[-1] - coordinate[-2]))
        if coordinate.size > 2:
            widths = widths.at[1:-1].set(0.5 * (coordinate[2:] - coordinate[:-2]))
        axis_shape = [1] * len(coordinates)
        axis_shape[axis] = int(coordinate.size)
        result = result * widths.reshape(tuple(axis_shape))
    return result


class StructuredPDEData(StrictModule):
    """Fields sampled on a shared tensor grid with explicit cases, masks, and measure."""

    coordinates: tuple[Array, ...]
    values: Array
    sample_valid: Array
    weights: Array
    state_layout: StateLayout
    coordinate_names: tuple[str, ...] = eqx.field(static=True)
    time_axis: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: Sequence[ArrayLike],
        values: ArrayLike,
        /,
        *,
        state_layout: StateLayout,
        coordinate_names: Sequence[str],
        time_axis: int = 0,
        sample_valid: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        source_id: str,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        coordinate_values = tuple(jnp.asarray(item) for item in coordinates)
        names = tuple(str(name) for name in coordinate_names)
        if not coordinate_values or len(names) != len(coordinate_values):
            raise ValueError("coordinate_names must name every tensor-grid axis.")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("coordinate_names must be non-empty and unique.")
        for coordinate in coordinate_values:
            if coordinate.ndim != 1 or coordinate.size < 3:
                raise ValueError(
                    "Each PDE coordinate axis must be a rank-one array with at least three points."
                )
            host = np.asarray(coordinate, dtype=float)
            if not np.all(np.isfinite(host)) or not np.all(np.diff(host) > 0.0):
                raise ValueError(
                    "PDE coordinates must be finite and strictly increasing."
                )
        resolved_time_axis = int(time_axis)
        if resolved_time_axis < 0 or resolved_time_axis >= len(coordinate_values):
            raise ValueError("time_axis is out of range.")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string.")
        field_values = jnp.asarray(values)
        if not jnp.issubdtype(field_values.dtype, jnp.inexact):
            field_values = field_values.astype(float)
        batch_shape = _shape_tail(field_values, state_layout.shape)
        grid_shape = tuple(int(item.size) for item in coordinate_values)
        if (
            len(batch_shape) < len(grid_shape)
            or tuple(batch_shape[-len(grid_shape) :]) != grid_shape
        ):
            raise ValueError(
                "values must end in tensor-grid axes followed by the state layout."
            )
        case_shape = tuple(batch_shape[: -len(grid_shape)])
        sample_shape = case_shape + grid_shape
        flat_values = field_values.reshape(sample_shape + (state_layout.size,))
        finite = jnp.all(jnp.isfinite(flat_values), axis=-1)
        if sample_valid is None:
            valid = finite
        else:
            requested = jnp.asarray(sample_valid, dtype=bool)
            if requested.shape != sample_shape:
                raise ValueError(
                    f"sample_valid must have shape {sample_shape}; got {requested.shape}."
                )
            valid = requested & finite
        if weights is None:
            grid_weights = _quadrature_weights(coordinate_values)
            weight_values = jnp.broadcast_to(grid_weights, sample_shape)
        else:
            weight_values = jnp.asarray(weights)
            if weight_values.shape != sample_shape:
                raise ValueError(
                    f"weights must have shape {sample_shape}; got {weight_values.shape}."
                )
            if jnp.issubdtype(weight_values.dtype, jnp.complexfloating):
                raise TypeError("weights must be real-valued.")
        finite_weights = jnp.isfinite(weight_values) & (weight_values >= 0.0)
        valid = valid & finite_weights & (weight_values > 0.0)
        expanded_valid = valid.reshape(sample_shape + (1,) * len(state_layout.shape))
        self.coordinates = coordinate_values
        self.values = jnp.where(expanded_valid, field_values, 0.0)
        self.sample_valid = valid
        self.weights = jnp.where(valid, weight_values, 0.0)
        self.state_layout = state_layout
        self.coordinate_names = names
        self.time_axis = resolved_time_axis
        self.case_shape = case_shape
        self.grid_shape = grid_shape
        self.source_id = source_id
        self.dataset_id = "pde-data:" + canonical_fingerprint(
            {
                "source": source_id,
                "coordinates": tuple(
                    np.asarray(value).tolist() for value in coordinate_values
                ),
                "state_layout": state_layout.layout_id,
                "case_shape": case_shape,
            }
        )

    @property
    def num_cases(self) -> int:
        return prod(self.case_shape) if self.case_shape else 1

    @property
    def num_axes(self) -> int:
        return len(self.grid_shape)

    @property
    def flat_values(self) -> Array:
        return self.values.reshape(
            self.case_shape + self.grid_shape + (self.state_layout.size,)
        )


class PDEDerivative(StrictModule):
    """One component and tensor-grid derivative multi-index."""

    component: int = eqx.field(static=True)
    orders: tuple[int, ...] = eqx.field(static=True)
    name: str | None = eqx.field(static=True)

    def __init__(
        self,
        component: int,
        orders: Sequence[int],
        /,
        *,
        name: str | None = None,
    ):
        resolved_orders = tuple(int(order) for order in orders)
        if not resolved_orders or any(order < 0 for order in resolved_orders):
            raise ValueError("orders must be a non-empty nonnegative multi-index.")
        resolved_name = None if name is None else str(name)
        if resolved_name == "":
            raise ValueError("name must be non-empty or None.")
        self.component = int(component)
        self.orders = resolved_orders
        self.name = resolved_name


class PDEDerivativeEvaluation(StrictModule):
    values: Array
    valid: Array
    derivative: PDEDerivative
    method_id: str = eqx.field(static=True)


class AbstractPDEDerivative(StrictModule):
    """Derivative policy over one structured field component."""

    method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        data: StructuredPDEData,
        derivative: PDEDerivative,
        /,
    ) -> PDEDerivativeEvaluation:
        raise NotImplementedError


def _erode_axis(mask: Array, axis: int, /) -> Array:
    left = jnp.roll(mask, 1, axis=axis)
    right = jnp.roll(mask, -1, axis=axis)
    eroded = mask & left & right
    lower: list[slice | int] = [slice(None)] * mask.ndim
    upper: list[slice | int] = [slice(None)] * mask.ndim
    lower[axis] = 0
    upper[axis] = -1
    eroded = eroded.at[tuple(lower)].set(False)
    return eroded.at[tuple(upper)].set(False)


class FiniteDifferencePDEDerivative(AbstractPDEDerivative):
    """Second-order centered tensor-grid differentiation with stencil-valid masks."""

    method_id: str = eqx.field(static=True)
    max_total_order: int = eqx.field(static=True)

    def __init__(self, *, max_total_order: int = 4):
        maximum = int(max_total_order)
        if maximum < 1:
            raise ValueError("max_total_order must be positive.")
        self.max_total_order = maximum
        self.method_id = f"finite-difference:gradient:max-order={maximum}"

    def evaluate(
        self,
        data: StructuredPDEData,
        derivative: PDEDerivative,
        /,
    ) -> PDEDerivativeEvaluation:
        if not isinstance(data, StructuredPDEData):
            raise TypeError("data must be StructuredPDEData.")
        if not isinstance(derivative, PDEDerivative):
            raise TypeError("derivative must be a PDEDerivative.")
        if derivative.component < 0 or derivative.component >= data.state_layout.size:
            raise ValueError("PDE derivative component is out of range.")
        if len(derivative.orders) != data.num_axes:
            raise ValueError("PDE derivative orders must name every coordinate axis.")
        if sum(derivative.orders) > self.max_total_order:
            raise ValueError(
                f"Derivative total order exceeds max_total_order={self.max_total_order}."
            )
        values = data.flat_values[..., derivative.component]
        valid = data.sample_valid
        case_rank = len(data.case_shape)
        for grid_axis, order in enumerate(derivative.orders):
            array_axis = case_rank + grid_axis
            for _ in range(order):
                gradient = jnp.gradient(
                    values,
                    data.coordinates[grid_axis],
                    axis=array_axis,
                )
                if isinstance(gradient, list):
                    raise RuntimeError(
                        "Single-axis finite differences returned multiple gradients."
                    )
                values = gradient
                valid = _erode_axis(valid, array_axis)
        finite = jnp.isfinite(values)
        valid = valid & finite
        return PDEDerivativeEvaluation(
            values=jnp.where(valid, values, 0.0),
            valid=valid,
            derivative=derivative,
            method_id=self.method_id,
        )


class PDELibraryTerm(StrictModule):
    """A state monomial optionally multiplying one field derivative."""

    state_powers: tuple[int, ...] = eqx.field(static=True)
    derivative: PDEDerivative | None
    name: str = eqx.field(static=True)

    def __init__(
        self,
        state_powers: Sequence[int],
        /,
        *,
        derivative: PDEDerivative | None = None,
        name: str,
    ):
        powers = tuple(int(power) for power in state_powers)
        if not powers or any(power < 0 for power in powers):
            raise ValueError("state_powers must be a non-empty nonnegative tuple.")
        if derivative is not None and not isinstance(derivative, PDEDerivative):
            raise TypeError("derivative must be a PDEDerivative or None.")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string.")
        self.state_powers = powers
        self.derivative = derivative
        self.name = name


class AbstractPDEFeatureLibrary(StrictModule):
    """Ordered PDE-FIND term contract."""

    terms: AbstractAttribute[tuple[PDELibraryTerm, ...]]
    feature_names: AbstractAttribute[tuple[str, ...]]
    library_id: AbstractAttribute[str]

    @property
    def num_features(self) -> int:
        return len(self.terms)


class PDEFeatureLibrary(AbstractPDEFeatureLibrary):
    """Explicit ordered PDE-FIND terms without hidden equation generation."""

    terms: tuple[PDELibraryTerm, ...]
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[PDELibraryTerm],
        /,
        *,
        library_id: str | None = None,
    ):
        resolved = tuple(terms)
        if not resolved or any(not isinstance(term, PDELibraryTerm) for term in resolved):
            raise TypeError("terms must contain PDELibraryTerm instances.")
        names = tuple(term.name for term in resolved)
        if len(set(names)) != len(names):
            raise ValueError("PDE feature names must be unique.")
        identifier = (
            "pde-library:"
            + canonical_fingerprint(
                tuple(
                    (
                        term.state_powers,
                        None
                        if term.derivative is None
                        else (term.derivative.component, term.derivative.orders),
                        term.name,
                    )
                    for term in resolved
                )
            )
            if library_id is None
            else str(library_id)
        )
        if not identifier:
            raise ValueError("library_id must be non-empty.")
        self.terms = resolved
        self.feature_names = names
        self.library_id = identifier

    @property
    def num_features(self) -> int:
        return len(self.terms)


def _dense_indices(dimension: int, degree: int, /) -> tuple[tuple[int, ...], ...]:
    sparse = weighted_total_degree_indices(dimension, degree + 1)
    return tuple(
        tuple(dict(index).get(axis, 0) for axis in range(dimension)) for index in sparse
    )


def _monomial_name(powers: tuple[int, ...], names: tuple[str, ...], /) -> str:
    factors = []
    for power, name in zip(powers, names, strict=True):
        if power == 1:
            factors.append(name)
        elif power > 1:
            factors.append(f"{name}^{power}")
    return "1" if not factors else " * ".join(factors)


def _derivative_name(
    component_name: str,
    orders: tuple[int, ...],
    coordinate_names: tuple[str, ...],
    /,
) -> str:
    total = sum(orders)
    denominator = " ".join(
        f"d{name}" if order == 1 else f"d{name}^{order}"
        for name, order in zip(coordinate_names, orders, strict=True)
        if order
    )
    return (
        f"d({component_name})/{denominator}"
        if total == 1
        else f"d^{total}({component_name})/{denominator}"
    )


class PolynomialPDELibrary(AbstractPDEFeatureLibrary):
    """Canonical polynomial-state and spatial-derivative PDE-FIND dictionary."""

    terms: tuple[PDELibraryTerm, ...]
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    spatial_derivative_order: int = eqx.field(static=True)

    def __init__(
        self,
        state_layout: StateLayout,
        coordinate_names: Sequence[str],
        /,
        *,
        time_axis: int = 0,
        polynomial_degree: int = 2,
        spatial_derivative_order: int = 2,
        include_mixed_derivatives: bool = False,
        include_interactions: bool = True,
        max_features: int = 4096,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        names = tuple(str(name) for name in coordinate_names)
        resolved_time = int(time_axis)
        if len(names) < 2 or resolved_time < 0 or resolved_time >= len(names):
            raise ValueError(
                "PDE libraries require one time and at least one spatial axis."
            )
        degree = int(polynomial_degree)
        derivative_order = int(spatial_derivative_order)
        if degree < 0 or derivative_order < 1:
            raise ValueError(
                "polynomial_degree must be nonnegative and spatial_derivative_order positive."
            )
        monomials = _dense_indices(state_layout.size, degree)
        spatial_axes = tuple(axis for axis in range(len(names)) if axis != resolved_time)
        if include_mixed_derivatives:
            spatial_indices = tuple(
                index
                for index in _dense_indices(len(spatial_axes), derivative_order)
                if sum(index) > 0
            )
        else:
            spatial_indices = tuple(
                tuple(
                    order if local_axis == selected else 0
                    for local_axis in range(len(spatial_axes))
                )
                for selected in range(len(spatial_axes))
                for order in range(1, derivative_order + 1)
            )
        derivative_specs = []
        for component in range(state_layout.size):
            for spatial_index in spatial_indices:
                full = [0] * len(names)
                for local_axis, grid_axis in enumerate(spatial_axes):
                    full[grid_axis] = spatial_index[local_axis]
                derivative_specs.append(PDEDerivative(component, tuple(full)))
        interaction_monomials = monomials if include_interactions else (monomials[0],)
        count = len(monomials) + len(interaction_monomials) * len(derivative_specs)
        if count > int(max_features):
            raise ValueError(
                f"PDE library would contain {count} features; max_features={int(max_features)}."
            )
        terms = []
        for powers in monomials:
            terms.append(
                PDELibraryTerm(
                    powers,
                    name=_monomial_name(powers, state_layout.component_names),
                )
            )
        for derivative in derivative_specs:
            derivative_label = _derivative_name(
                state_layout.component_names[derivative.component],
                derivative.orders,
                names,
            )
            for powers in interaction_monomials:
                monomial_label = _monomial_name(powers, state_layout.component_names)
                label = (
                    derivative_label
                    if monomial_label == "1"
                    else f"({monomial_label}) * ({derivative_label})"
                )
                terms.append(
                    PDELibraryTerm(
                        powers,
                        derivative=derivative,
                        name=label,
                    )
                )
        library = PDEFeatureLibrary(terms)
        self.terms = library.terms
        self.feature_names = library.feature_names
        self.library_id = library.library_id
        self.polynomial_degree = degree
        self.spatial_derivative_order = derivative_order


class PDEIdentificationProblem(StrictModule):
    """Structured-grid PDE data, target derivatives, dictionary, and derivative policy."""

    data: StructuredPDEData
    library: AbstractPDEFeatureLibrary
    targets: tuple[PDEDerivative, ...]
    derivative: AbstractPDEDerivative
    target_names: tuple[str, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        data: StructuredPDEData,
        library: AbstractPDEFeatureLibrary,
        targets: Sequence[PDEDerivative] | None = None,
        derivative: AbstractPDEDerivative | None = None,
    ):
        if not isinstance(data, StructuredPDEData):
            raise TypeError("data must be StructuredPDEData.")
        if not isinstance(library, AbstractPDEFeatureLibrary):
            raise TypeError("library must be an AbstractPDEFeatureLibrary.")
        for term in library.terms:
            if len(term.state_powers) != data.state_layout.size:
                raise ValueError("Every PDE term must name each state component power.")
            if term.derivative is not None:
                _validate_derivative(data, term.derivative)
        if targets is None:
            time_orders = tuple(
                1 if axis == data.time_axis else 0 for axis in range(data.num_axes)
            )
            resolved_targets = tuple(
                PDEDerivative(component, time_orders)
                for component in range(data.state_layout.size)
            )
        else:
            resolved_targets = tuple(targets)
        if not resolved_targets or any(
            not isinstance(target, PDEDerivative) for target in resolved_targets
        ):
            raise TypeError("targets must contain PDEDerivative instances.")
        for target in resolved_targets:
            _validate_derivative(data, target)
        derivative_policy = (
            FiniteDifferencePDEDerivative() if derivative is None else derivative
        )
        if not isinstance(derivative_policy, AbstractPDEDerivative):
            raise TypeError("derivative must be an AbstractPDEDerivative.")
        target_names = tuple(
            target.name
            if target.name is not None
            else _derivative_name(
                data.state_layout.component_names[target.component],
                target.orders,
                data.coordinate_names,
            )
            for target in resolved_targets
        )
        if len(set(target_names)) != len(target_names):
            raise ValueError("PDE target names must be unique.")
        self.data = data
        self.library = library
        self.targets = resolved_targets
        self.derivative = derivative_policy
        self.target_names = target_names
        self.problem_id = "pde-problem:" + canonical_fingerprint(
            {
                "data": data.dataset_id,
                "library": library.library_id,
                "targets": tuple(
                    (target.component, target.orders) for target in resolved_targets
                ),
                "derivative": derivative_policy.method_id,
            }
        )

    def build_design(self) -> SINDyDesign:
        fields = self.data.flat_values
        derivative_cache: dict[tuple[int, tuple[int, ...]], PDEDerivativeEvaluation] = {}

        def evaluate_derivative(specification: PDEDerivative) -> PDEDerivativeEvaluation:
            key = (specification.component, specification.orders)
            if key not in derivative_cache:
                derivative_cache[key] = self.derivative.evaluate(self.data, specification)
            return derivative_cache[key]

        feature_values = []
        feature_valid = []
        for term in self.library.terms:
            monomial = jnp.ones(
                self.data.case_shape + self.data.grid_shape,
                dtype=fields.dtype,
            )
            for component, power in enumerate(term.state_powers):
                if power:
                    monomial = monomial * fields[..., component] ** power
            valid = self.data.sample_valid & jnp.isfinite(monomial)
            if term.derivative is not None:
                evaluated = evaluate_derivative(term.derivative)
                monomial = monomial * evaluated.values
                valid = valid & evaluated.valid
            feature_values.append(jnp.where(valid, monomial, 0.0))
            feature_valid.append(valid)
        target_evaluations = tuple(evaluate_derivative(target) for target in self.targets)
        matrix = jnp.stack(tuple(feature_values), axis=-1)
        target = jnp.stack(
            tuple(evaluation.values for evaluation in target_evaluations),
            axis=-1,
        )
        valid = self.data.sample_valid & jnp.all(
            jnp.stack(tuple(feature_valid), axis=-1), axis=-1
        )
        valid = valid & jnp.all(
            jnp.stack(
                tuple(evaluation.valid for evaluation in target_evaluations),
                axis=-1,
            ),
            axis=-1,
        )
        metadata_shape = self.data.case_shape + self.data.grid_shape
        time_size = self.data.grid_shape[self.data.time_axis]
        time_index_shape = [1] * len(metadata_shape)
        time_index_shape[len(self.data.case_shape) + self.data.time_axis] = time_size
        time_indices = jnp.broadcast_to(
            jnp.arange(time_size, dtype=jnp.int32).reshape(tuple(time_index_shape)),
            metadata_shape,
        )
        time_coordinates = jnp.broadcast_to(
            self.data.coordinates[self.data.time_axis].reshape(tuple(time_index_shape)),
            metadata_shape,
        )
        if self.data.case_shape:
            case_shape = self.data.case_shape + (1,) * self.data.num_axes
            case_index = jnp.broadcast_to(
                jnp.arange(self.data.num_cases, dtype=jnp.int32).reshape(case_shape),
                metadata_shape,
            )
        else:
            case_index = jnp.zeros(metadata_shape, dtype=jnp.int32)
        output_layout = StateLayout(
            (len(self.targets),), component_names=self.target_names
        )
        formulation_id = (
            f"pde-find:derivative={self.derivative.method_id}:"
            f"targets={tuple((target.component, target.orders) for target in self.targets)}"
        )
        return _make_equation_design(
            matrix=matrix,
            target=target,
            valid=valid,
            weights=self.data.weights,
            coordinates=time_coordinates,
            case_index=case_index,
            window_start=time_indices,
            window_end=time_indices,
            state_layout=output_layout,
            input_layout=None,
            feature_names=self.library.feature_names,
            output_names=self.target_names,
            formulation="pde",
            source_id=self.data.source_id,
            coordinate_id=self.data.coordinate_names[self.data.time_axis],
            library_id=self.library.library_id,
            formulation_id=formulation_id,
        )


def _validate_derivative(data: StructuredPDEData, derivative: PDEDerivative, /) -> None:
    if derivative.component < 0 or derivative.component >= data.state_layout.size:
        raise ValueError("PDE derivative component is out of range.")
    if len(derivative.orders) != data.num_axes:
        raise ValueError("PDE derivative orders must name every coordinate axis.")


class PDEIdentificationResult(StrictModule):
    """Sparse PDE coefficients and all strong-form design and solver evidence."""

    coefficients: Array
    support: Array
    design: SINDyDesign
    regression: SparseRegressionResult
    problem: PDEIdentificationProblem
    valid: Array
    method_id: str = eqx.field(static=True)

    def equations(self, *, digits: int = 6, active_only: bool = True) -> tuple[str, ...]:
        equations = []
        values = np.asarray(self.coefficients)
        supports = np.asarray(self.support)
        for output, target_name in enumerate(self.design.output_names):
            terms = []
            for feature, feature_name in enumerate(self.design.feature_names):
                if active_only and not supports[output, feature]:
                    continue
                coefficient = values[output, feature]
                terms.append(f"{coefficient:.{digits}g} * {feature_name}")
            equations.append(f"{target_name} = " + (" + ".join(terms) if terms else "0"))
        return tuple(equations)


def fit_pde_find(
    problem: PDEIdentificationProblem,
    regressor: AbstractSparseRegression,
    /,
) -> PDEIdentificationResult:
    """Build one PDE-FIND design and fit the declared sparse regressor."""
    if not isinstance(problem, PDEIdentificationProblem):
        raise TypeError("problem must be a PDEIdentificationProblem.")
    if not isinstance(regressor, AbstractSparseRegression):
        raise TypeError("regressor must be an AbstractSparseRegression.")
    design = problem.build_design()
    regression = regressor.fit(design)
    return PDEIdentificationResult(
        coefficients=regression.coefficients,
        support=regression.support,
        design=design,
        regression=regression,
        problem=problem,
        valid=regression.successful,
        method_id=f"pde-find:{regression.method_id}",
    )


__all__ = [
    "AbstractPDEFeatureLibrary",
    "AbstractPDEDerivative",
    "FiniteDifferencePDEDerivative",
    "PDEDerivative",
    "PDEDerivativeEvaluation",
    "PDEFeatureLibrary",
    "PDEIdentificationProblem",
    "PDEIdentificationResult",
    "PDELibraryTerm",
    "PolynomialPDELibrary",
    "StructuredPDEData",
    "fit_pde_find",
]

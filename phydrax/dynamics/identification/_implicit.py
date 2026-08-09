#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from .._layout import InputLayout, StateLayout
from .._trajectory import TrajectoryData
from ._features import AbstractFeatureLibrary, FeatureEvaluation, PolynomialFeatureLibrary
from ._sindy_design import (
    _make_equation_design,
    _row_metadata,
    _sample_inputs,
    _time_values,
)
from ._sparse_regression import AbstractSparseRegression, SparseRegressionResult


class AbstractImplicitFeatureLibrary(StrictModule):
    """Feature dictionary over state, derivative, and optional input values."""

    state_layout: AbstractAttribute[StateLayout]
    input_layout: AbstractAttribute[InputLayout | None]
    feature_names: AbstractAttribute[tuple[str, ...]]
    library_id: AbstractAttribute[str]

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @abc.abstractmethod
    def evaluate(
        self,
        states: ArrayLike,
        derivatives: ArrayLike,
        inputs: ArrayLike | None = None,
        /,
    ) -> FeatureEvaluation:
        raise NotImplementedError


class ImplicitFeatureLibrary(AbstractImplicitFeatureLibrary):
    """Adapt any library over a concatenated flattened state and derivative."""

    base: AbstractFeatureLibrary
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractFeatureLibrary,
        /,
        *,
        state_layout: StateLayout,
    ):
        if not isinstance(base, AbstractFeatureLibrary):
            raise TypeError("base must be an AbstractFeatureLibrary.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if base.state_layout.size != 2 * state_layout.size:
            raise ValueError(
                "The base library state layout must contain the flattened state and derivative."
            )
        self.base = base
        self.state_layout = state_layout
        self.input_layout = base.input_layout
        self.feature_names = base.feature_names
        self.library_id = "implicit:" + canonical_fingerprint(
            {
                "base": base.library_id,
                "state_layout": state_layout.layout_id,
            }
        )

    def evaluate(
        self,
        states: ArrayLike,
        derivatives: ArrayLike,
        inputs: ArrayLike | None = None,
        /,
    ) -> FeatureEvaluation:
        state_values = jnp.asarray(states)
        derivative_values = jnp.asarray(derivatives)
        rank = len(self.state_layout.shape)
        expected_tail = self.state_layout.shape
        if rank:
            valid_state_shape = (
                state_values.ndim >= rank
                and tuple(state_values.shape[-rank:]) == expected_tail
            )
            batch_shape = state_values.shape[:-rank]
        else:
            valid_state_shape = state_values.ndim >= 0
            batch_shape = state_values.shape
        if not valid_state_shape or derivative_values.shape != state_values.shape:
            raise ValueError(
                "states and derivatives must have identical shapes ending in the state layout."
            )
        state_flat = state_values.reshape(batch_shape + (self.state_layout.size,))
        derivative_flat = derivative_values.reshape(
            batch_shape + (self.state_layout.size,)
        )
        augmented = jnp.concatenate((state_flat, derivative_flat), axis=-1).reshape(
            batch_shape + self.base.state_layout.shape
        )
        evaluation = self.base.evaluate(augmented, inputs)
        return FeatureEvaluation(
            values=evaluation.values,
            valid=evaluation.valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


class PolynomialImplicitFeatureLibrary(AbstractImplicitFeatureLibrary):
    """Weighted-total-degree monomials in state, derivative, and optional input."""

    adapter: ImplicitFeatureLibrary
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)

    def __init__(
        self,
        state_layout: StateLayout,
        /,
        *,
        degree: int = 2,
        input_layout: InputLayout | None = None,
        include_bias: bool = True,
        interaction_only: bool = False,
        max_features: int = 4096,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        augmented_names = state_layout.component_names + tuple(
            f"d({name})/dcoordinate" for name in state_layout.component_names
        )
        augmented_layout = StateLayout(
            (2 * state_layout.size,), component_names=augmented_names
        )
        base = PolynomialFeatureLibrary(
            augmented_layout,
            input_layout=input_layout,
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only,
            max_features=max_features,
        )
        adapter = ImplicitFeatureLibrary(base, state_layout=state_layout)
        self.adapter = adapter
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.feature_names = adapter.feature_names
        self.library_id = adapter.library_id
        self.degree = int(degree)

    def evaluate(
        self,
        states: ArrayLike,
        derivatives: ArrayLike,
        inputs: ArrayLike | None = None,
        /,
    ) -> FeatureEvaluation:
        return self.adapter.evaluate(states, derivatives, inputs)


class ImplicitSINDyProblem(StrictModule):
    """Homogeneous feature equations formed from state and attached derivatives."""

    data: TrajectoryData
    library: AbstractImplicitFeatureLibrary
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        data: TrajectoryData,
        library: AbstractImplicitFeatureLibrary,
    ):
        if not isinstance(data, TrajectoryData):
            raise TypeError("data must be TrajectoryData.")
        if not isinstance(library, AbstractImplicitFeatureLibrary):
            raise TypeError("library must be an AbstractImplicitFeatureLibrary.")
        if data.derivatives is None or data.derivative_valid is None:
            raise ValueError("Implicit SINDy requires attached derivatives and validity.")
        if library.state_layout.layout_id != data.state_layout.layout_id:
            raise ValueError("Implicit library and trajectory state layouts must match.")
        expected_input = (
            None if data.input_layout is None else data.input_layout.layout_id
        )
        actual_input = (
            None if library.input_layout is None else library.input_layout.layout_id
        )
        if expected_input != actual_input:
            raise ValueError("Implicit library and trajectory input layouts must match.")
        self.data = data
        self.library = library
        self.problem_id = "implicit-problem:" + canonical_fingerprint(
            {"dataset": data.dataset_id, "library": library.library_id}
        )

    def homogeneous_design(
        self,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        count = (
            self.data.capacity
            if self.data.inputs is None or self.data.input_alignment == "samples"
            else self.data.capacity - 1
        )
        states = _time_values(self.data, self.data.states, slice(0, count))
        derivatives = _time_values(self.data, self.data.derivatives, slice(0, count))
        inputs, input_valid = _sample_inputs(self.data, count)
        evaluation = self.library.evaluate(states, derivatives, inputs)
        valid = (
            self.data.sample_valid[..., :count]
            & self.data.derivative_valid[..., :count]
            & input_valid
            & evaluation.valid
        )
        case_index, starts, ends = _row_metadata(
            self.data.num_cases, tuple(range(count)), tuple(range(count))
        )
        return (
            evaluation.values.reshape((-1, self.library.num_features)),
            valid.reshape((-1,)),
            self.data.weights[..., :count].reshape((-1,)),
            self.data.coordinates[..., :count].reshape((-1,)),
            case_index,
            starts,
            ends,
        )


class ImplicitSINDyCandidate(StrictModule):
    """One normalized target-feature regression and its homogeneous equation."""

    coefficients: Array
    support: Array
    residual: Array
    residual_norm: Array
    score: Array
    valid: Array
    regression: SparseRegressionResult
    target_index: int = eqx.field(static=True)
    target_name: str = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)

    def equation(self, *, digits: int = 6) -> str:
        terms = []
        for coefficient, name in zip(
            np.asarray(self.coefficients), self.feature_names, strict=True
        ):
            if coefficient != 0.0:
                terms.append(f"{coefficient:.{digits}g} * {name}")
        return " + ".join(terms) + " = 0"


class ImplicitSINDyResult(StrictModule):
    """Ranked target-feature equations with invalid candidates retained."""

    candidates: tuple[ImplicitSINDyCandidate, ...]
    scores: Array
    candidate_valid: Array
    selected_index: Array
    feature_names: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.selected_index >= 0

    @property
    def selected(self) -> ImplicitSINDyCandidate:
        index = int(self.selected_index)
        if index < 0:
            raise ValueError("No valid implicit equation candidate was found.")
        return self.candidates[index]


def _target_indices(
    targets: Sequence[int | str] | None,
    names: tuple[str, ...],
    /,
) -> tuple[int, ...]:
    if targets is None:
        return tuple(range(len(names)))
    resolved = []
    for target in targets:
        if isinstance(target, str):
            if target not in names:
                raise ValueError(f"Unknown implicit target feature {target!r}.")
            index = names.index(target)
        else:
            index = int(target)
            if index < 0 or index >= len(names):
                raise ValueError("Implicit target index is out of range.")
        resolved.append(index)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("targets must resolve to unique feature indices.")
    return tuple(resolved)


def fit_implicit_sindy(
    problem: ImplicitSINDyProblem,
    regressor: AbstractSparseRegression,
    /,
    *,
    targets: Sequence[int | str] | None = None,
    complexity_weight: float = 0.0,
) -> ImplicitSINDyResult:
    """Search target-feature normalizations without admitting the zero equation."""
    if not isinstance(problem, ImplicitSINDyProblem):
        raise TypeError("problem must be an ImplicitSINDyProblem.")
    if not isinstance(regressor, AbstractSparseRegression):
        raise TypeError("regressor must be an AbstractSparseRegression.")
    penalty = float(complexity_weight)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("complexity_weight must be finite and nonnegative.")
    matrix, valid, weights, coordinates, cases, starts, ends = (
        problem.homogeneous_design()
    )
    target_indices = _target_indices(targets, problem.library.feature_names)
    output_layout = StateLayout((1,), component_names=("implicit_equation",))
    candidates = []
    scores = []
    candidate_valid = []
    weight_sum = jnp.maximum(jnp.sum(jnp.where(valid, weights, 0.0)), 1.0)
    for target_index in target_indices:
        feature_indices = tuple(
            index
            for index in range(problem.library.num_features)
            if index != target_index
        )
        feature_names = tuple(
            problem.library.feature_names[index] for index in feature_indices
        )
        target_name = problem.library.feature_names[target_index]
        target = matrix[:, target_index : target_index + 1]
        design = _make_equation_design(
            matrix=matrix[:, feature_indices],
            target=target,
            valid=valid,
            weights=weights,
            coordinates=coordinates,
            case_index=cases,
            window_start=starts,
            window_end=ends,
            state_layout=output_layout,
            input_layout=problem.data.input_layout,
            feature_names=feature_names,
            output_names=("implicit_equation",),
            formulation="implicit",
            source_id=problem.data.source_id,
            coordinate_id=problem.data.coordinate_id,
            library_id=problem.library.library_id,
            formulation_id=f"implicit:target={target_name}",
        )
        regression = regressor.fit(design)
        equation = jnp.zeros(
            (problem.library.num_features,), dtype=regression.coefficients.dtype
        )
        equation = equation.at[target_index].set(1.0)
        equation = equation.at[jnp.asarray(feature_indices)].set(
            -regression.coefficients[0]
        )
        residual = matrix @ equation
        residual_norm = jnp.sqrt(
            jnp.sum(jnp.where(valid, weights * jnp.abs(residual) ** 2, 0.0))
        )
        target_energy = jnp.sum(
            jnp.where(valid, weights * jnp.abs(target[:, 0]) ** 2, 0.0)
        )
        relative_error = residual_norm**2 / jnp.maximum(
            target_energy, jnp.finfo(residual_norm.dtype).tiny
        )
        support = equation != 0.0
        complexity = jnp.sum(support).astype(relative_error.dtype)
        valid_candidate = (
            regression.successful
            & jnp.isfinite(relative_error)
            & (target_energy > jnp.finfo(target_energy.dtype).eps * weight_sum)
            & (complexity > 1)
        )
        score = jnp.where(
            valid_candidate,
            relative_error + penalty * complexity,
            jnp.inf,
        )
        candidates.append(
            ImplicitSINDyCandidate(
                coefficients=equation,
                support=support,
                residual=jnp.where(valid, residual, 0.0),
                residual_norm=residual_norm,
                score=score,
                valid=valid_candidate,
                regression=regression,
                target_index=target_index,
                target_name=target_name,
                feature_names=problem.library.feature_names,
            )
        )
        scores.append(score)
        candidate_valid.append(valid_candidate)
    score_values = jnp.stack(tuple(scores))
    valid_values = jnp.stack(tuple(candidate_valid))
    selected = jnp.where(jnp.any(valid_values), jnp.argmin(score_values), -1).astype(
        jnp.int32
    )
    return ImplicitSINDyResult(
        candidates=tuple(candidates),
        scores=score_values,
        candidate_valid=valid_values,
        selected_index=selected,
        feature_names=problem.library.feature_names,
        method_id=f"implicit-sindy:{candidates[0].regression.method_id}",
        problem_id=problem.problem_id,
    )


__all__ = [
    "AbstractImplicitFeatureLibrary",
    "ImplicitFeatureLibrary",
    "ImplicitSINDyCandidate",
    "ImplicitSINDyProblem",
    "ImplicitSINDyResult",
    "PolynomialImplicitFeatureLibrary",
    "fit_implicit_sindy",
]

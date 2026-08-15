#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...metrix import EuclideanStateGeometry
from .._layout import InputLayout, StateLayout
from .._system import ContinuousSystem, DiscreteSystem
from ._features import AbstractFeatureLibrary
from ._sindy_design import SINDyDesign, SINDyProblem
from ._sparse_regression import AbstractSparseRegression, SparseRegressionResult


class _IdentifiedSINDyLaw(StrictModule):
    library: AbstractFeatureLibrary
    coefficients: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    controlled: bool = eqx.field(static=True)

    def __call__(
        self,
        coordinate: Array,
        state: Array,
        inputs_or_args,
        args=None,
    ) -> Array:
        del coordinate, args
        inputs = inputs_or_args if self.controlled else None
        features = self.library.evaluate(state, inputs).values
        return (self.coefficients @ features).reshape(self.state_shape)


class SINDyResult(StrictModule):
    """Sparse physical equations, diagnostics, and executable system conversion."""

    coefficients: Array
    support: Array
    regression: SparseRegressionResult
    design: SINDyDesign
    library: AbstractFeatureLibrary
    state_layout: StateLayout
    input_layout: InputLayout | None
    valid: Array
    status: Array
    formulation: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def predict_design(self) -> Array:
        return self.design.matrix @ self.coefficients.T

    def evaluate(
        self,
        states: ArrayLike,
        inputs: ArrayLike | None = None,
        /,
    ) -> Array:
        evaluation = self.library.evaluate(states, inputs)
        values = oe.contract("of,...f->...o", self.coefficients, evaluation.values)
        return values.reshape(evaluation.valid.shape + self.state_layout.shape)

    def render_equations(
        self,
        /,
        *,
        precision: int = 6,
        active_only: bool = True,
    ) -> tuple[str, ...]:
        digits = int(precision)
        if digits < 1:
            raise ValueError("precision must be positive.")
        coefficients = jnp.asarray(self.coefficients)
        equations = []
        derivative = self.formulation != "discrete"
        for output, name in enumerate(self.design.output_names):
            terms = []
            for feature, feature_name in enumerate(self.design.feature_names):
                coefficient = float(coefficients[output, feature])
                if active_only and not bool(self.support[output, feature]):
                    continue
                terms.append(f"{coefficient:.{digits}g} * {feature_name}")
            left = (
                f"d{name}/d{self.design.coordinate_id}" if derivative else f"{name}[k+1]"
            )
            equations.append(f"{left} = {' + '.join(terms) if terms else '0'}")
        return tuple(equations)

    def to_system(
        self,
        /,
        *,
        system_id: str | None = None,
    ) -> ContinuousSystem | DiscreteSystem:
        if not bool(self.valid):
            raise ValueError("Cannot construct a system from an invalid SINDy result.")
        if not isinstance(self.state_layout.geometry, EuclideanStateGeometry):
            raise ValueError(
                "Unstructured SINDy coefficients define an ambient Euclidean law; "
                "use a structured manifold formulation for non-Euclidean state layouts."
            )
        identifier = (
            f"identified-sindy:{self.source_id}:{self.method_id}"
            if system_id is None
            else system_id
        )
        law = _IdentifiedSINDyLaw(
            library=self.library,
            coefficients=self.coefficients,
            state_shape=self.state_layout.shape,
            controlled=self.input_layout is not None,
        )
        system_type = (
            DiscreteSystem if self.formulation == "discrete" else ContinuousSystem
        )
        return system_type(
            law,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=identifier,
        )


def _result_from_regression(
    problem: SINDyProblem,
    design: SINDyDesign,
    regression: SparseRegressionResult,
    /,
) -> SINDyResult:
    return SINDyResult(
        coefficients=regression.coefficients,
        support=regression.support,
        regression=regression,
        design=design,
        library=problem.library,
        state_layout=problem.data.state_layout,
        input_layout=problem.data.input_layout,
        valid=regression.successful,
        status=regression.status,
        formulation=design.formulation,
        source_id=problem.data.source_id,
        method_id=f"sindy:{design.formulation}:{regression.method_id}",
    )


def fit_sindy(
    problem: SINDyProblem,
    regressor: AbstractSparseRegression,
    /,
) -> SINDyResult:
    """Build equations, fit one declared sparse regressor, and retain all evidence."""
    if not isinstance(problem, SINDyProblem):
        raise TypeError("problem must be a SINDyProblem.")
    if not isinstance(regressor, AbstractSparseRegression):
        raise TypeError("regressor must be an AbstractSparseRegression.")
    design = problem.build_design()
    regression = regressor.fit(design)
    return _result_from_regression(problem, design, regression)


__all__ = ["SINDyResult", "fit_sindy"]

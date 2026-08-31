#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    PseudospectralMethodPlan,
    TensorSpectralDiscretization,
)
from ._ir import PDEExpression, PDEProblemIR
from ._spectral_compile import (
    _SpectralEvaluator,
    SpectralStateLayout,
)
from ._validate import infer_expression_type, validate_pde_ir


SpectralResidualScope: TypeAlias = Literal["full", "retained"]
SpectralConditionHandling: TypeAlias = Literal["reject", "external"]


class SpectralResidualCompilationReport(StrictModule, NonTrainableState):
    """Exactness, capacity, measure, and condition evidence for one residual."""

    trial_shape: tuple[int, ...] = eqx.field(static=True)
    evaluation_shape: tuple[int, ...] = eqx.field(static=True)
    equation_names: tuple[str, ...] = eqx.field(static=True)
    scope: SpectralResidualScope = eqx.field(static=True)
    condition_handling: SpectralConditionHandling = eqx.field(static=True)
    maximum_polynomial_degree: int | None = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    measure: str = eqx.field(static=True)
    evaluation_modes: int = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        trial_shape: Sequence[int],
        evaluation_shape: Sequence[int],
        equation_names: Sequence[str],
        scope: SpectralResidualScope,
        condition_handling: SpectralConditionHandling,
        maximum_polynomial_degree: int | None,
        exact: bool,
        coefficient_itemsize: int,
    ):
        trial = tuple(int(value) for value in trial_shape)
        evaluation = tuple(int(value) for value in evaluation_shape)
        names = tuple(str(name) for name in equation_names)
        count = 1
        for size in evaluation:
            count *= size
        self.trial_shape = trial
        self.evaluation_shape = evaluation
        self.equation_names = names
        self.scope = scope
        self.condition_handling = condition_handling
        self.maximum_polynomial_degree = maximum_polynomial_degree
        self.exact = bool(exact)
        self.measure = "prepared-physical-quadrature"
        self.evaluation_modes = count
        self.coefficient_bytes = count * int(coefficient_itemsize)
        self.report_id = canonical_fingerprint(
            {
                "kind": "spectral-residual-compilation-report",
                "trial_shape": list(trial),
                "evaluation_shape": list(evaluation),
                "equations": list(names),
                "scope": scope,
                "condition_handling": condition_handling,
                "maximum_polynomial_degree": maximum_polynomial_degree,
                "exact": bool(exact),
                "measure": self.measure,
                "evaluation_modes": count,
                "coefficient_bytes": self.coefficient_bytes,
            }
        )


class CompiledSpectralResidual(StrictModule):
    """All-coordinate spectral PDE residual and measured scalar objective."""

    layout: SpectralStateLayout
    discretization: TensorSpectralDiscretization
    method: Any
    evaluator: _SpectralEvaluator
    equation_scales: Array
    report: SpectralResidualCompilationReport
    equation_names: tuple[str, ...] = eqx.field(static=True)
    equation_components: tuple[int, ...] = eqx.field(static=True)
    scope: SpectralResidualScope = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)

    def __init__(
        self,
        layout: SpectralStateLayout,
        discretization: TensorSpectralDiscretization,
        method: Any,
        evaluator: _SpectralEvaluator,
        equation_scales: ArrayLike,
        report: SpectralResidualCompilationReport,
        /,
        *,
        equation_names: Sequence[str],
        equation_components: Sequence[int],
        compilation_id: str,
        source_hash: str,
        scope: SpectralResidualScope,
    ):
        names = tuple(str(name) for name in equation_names)
        components = tuple(int(value) for value in equation_components)
        scales = jnp.asarray(
            equation_scales, dtype=discretization.quadrature_weights.dtype
        )
        if scales.shape != (len(names),):
            raise ValueError("equation_scales must contain one value per equation.")
        self.layout = layout
        self.discretization = discretization
        self.method = method
        self.evaluator = evaluator
        self.equation_scales = scales
        self.report = report
        self.equation_names = names
        self.equation_components = components
        self.scope = scope
        self.compilation_id = str(compilation_id)
        self.source_hash = str(source_hash)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.layout.state_shape

    @property
    def evaluation(self) -> TensorSpectralDiscretization:
        return self.method.dealiasing.evaluation

    def project_state(self, values: ArrayLike | Mapping[str, ArrayLike], /) -> Array:
        physical = (
            values
            if isinstance(values, Mapping)
            else self.layout.unpack(values, physical=True)
        )
        coefficients = {
            name: self.discretization.project(physical[name])
            for name in self.layout.field_names
        }
        return self.layout.pack(coefficients)

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        coefficients = self.layout.unpack(state)
        physical = {
            name: self.discretization.reconstruct(coefficients[name])
            for name in self.layout.field_names
        }
        return self.layout.pack(physical, physical=True)

    def physical_residuals(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, ...]:
        value = jnp.asarray(state)
        time = jnp.asarray(0.0, dtype=value.real.dtype)
        outputs = self.evaluator.physical_outputs(time, value, args)
        if self.scope == "full":
            return outputs
        return tuple(
            self.discretization.reconstruct(self.method.dealiasing.project(output))
            for output in outputs
        )

    def residual_coefficients(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, ...]:
        outputs = self.physical_residuals(state, args)
        space = self.evaluation if self.scope == "full" else self.discretization
        return tuple(space.project(output) for output in outputs)

    def residual_energy(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        outputs = self.physical_residuals(state, args)
        space = self.evaluation if self.scope == "full" else self.discretization
        weights = space.quadrature_weights.astype(space.plan.precision.reduction_dtype)
        total = jnp.asarray(0.0, dtype=weights.dtype)
        for output, scale in zip(outputs, self.equation_scales, strict=True):
            value = output.astype(space.plan.precision.reduction_dtype) / scale
            weight_shape = weights.shape + (1,) * (value.ndim - weights.ndim)
            density = jnp.real(jnp.conj(value) * value)
            total = total + jnp.sum(weights.reshape(weight_shape) * density)
        return total

    def __call__(self, state: ArrayLike, args: Any = None, /) -> Array:
        return self.residual_energy(state, args)


def _residual_degree(
    expression: PDEExpression,
    functional_parameters: frozenset[str],
    /,
) -> int | None:
    if expression.op == "field":
        return 1
    if expression.op == "parameter":
        return 1 if expression.symbol in functional_parameters else 0
    if expression.op == "coordinate":
        return 1
    if expression.op == "constant":
        return 0
    degrees = tuple(
        _residual_degree(argument, functional_parameters) for argument in expression.args
    )
    if any(value is None for value in degrees):
        return None
    finite = tuple(int(value) for value in degrees if value is not None)
    if expression.op in (
        "add",
        "negate",
        "component",
        "derivative",
        "gradient",
        "divergence",
        "curl",
        "laplacian",
        "integral",
    ):
        return max(finite, default=0)
    if expression.op in ("multiply", "dot"):
        return sum(finite)
    if expression.op == "divide":
        return finite[0] if finite[1] == 0 else None
    if expression.op == "power":
        exponent = expression.args[1]
        if (
            exponent.op != "constant"
            or exponent.value is None
            or not float(exponent.value).is_integer()
        ):
            return None
        power = int(exponent.value)
        return finite[0] * power if power >= 0 else None
    if expression.op in ("sin", "cos", "exp", "log", "sqrt"):
        return 0 if finite[0] == 0 else None
    return None


def _coordinate_representation_exact(
    expression: PDEExpression,
    coordinate_axes: Mapping[str, tuple[int, ...]],
    discretization: TensorSpectralDiscretization,
    /,
) -> bool:
    if expression.op == "coordinate":
        assert expression.symbol is not None
        return all(
            discretization.axes[axis].family in ("chebyshev", "legendre")
            for axis in coordinate_axes[expression.symbol]
        )
    return all(
        _coordinate_representation_exact(argument, coordinate_axes, discretization)
        for argument in expression.args
    )


def _all_coordinate_axes(
    problem: PDEProblemIR,
    discretization: TensorSpectralDiscretization,
    /,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if sum(coordinate.size for coordinate in problem.coordinates) != len(
        discretization.axes
    ):
        raise ValueError("PDE coordinate size must match spectral tensor rank.")
    output = []
    offset = 0
    for coordinate in problem.coordinates:
        axes = tuple(range(offset, offset + coordinate.size))
        for axis in axes:
            prepared = discretization.axes[axis]
            if coordinate.periodic != prepared.periodic:
                raise ValueError(
                    f"PDE coordinate {coordinate.name!r} periodicity does not match "
                    f"spectral basis {prepared.family!r}."
                )
            if coordinate.bounds is not None and not jnp.allclose(
                jnp.asarray(coordinate.bounds), prepared.bounds
            ):
                raise ValueError(
                    f"PDE coordinate {coordinate.name!r} bounds do not match "
                    "the spectral basis."
                )
        output.append((coordinate.name, axes))
        offset += coordinate.size
    return tuple(output)


def _interior_region_axes(
    problem: PDEProblemIR,
    coordinate_axes: tuple[tuple[str, tuple[int, ...]], ...],
    /,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    lookup = dict(coordinate_axes)
    output = []
    for region in problem.regions:
        if region.kind != "interior" or region.component is not None:
            continue
        axes = tuple(
            axis
            for coordinate in region.coordinates
            if coordinate in lookup
            for axis in lookup[coordinate]
        )
        if axes:
            output.append((region.name, axes))
    return tuple(output)


def compile_spectral_residual(
    problem: PDEProblemIR,
    discretization: TensorSpectralDiscretization,
    method: PseudospectralMethodPlan,
    /,
    *,
    equations: Sequence[str] | None = None,
    parameter_values: Mapping[str, Any] | None = None,
    equation_scales: Mapping[str, float] | None = None,
    scope: SpectralResidualScope = "full",
    require_exact: bool = True,
    condition_handling: SpectralConditionHandling = "reject",
) -> CompiledSpectralResidual:
    """Compile selected PDE equalities into an all-coordinate residual objective."""
    if not isinstance(problem, PDEProblemIR):
        raise TypeError("problem must be a PDEProblemIR.")
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    if not isinstance(method, PseudospectralMethodPlan):
        raise TypeError("method must be a PseudospectralMethodPlan.")
    if scope not in ("full", "retained"):
        raise ValueError("scope must be 'full' or 'retained'.")
    if condition_handling not in ("reject", "external"):
        raise ValueError("condition_handling must be 'reject' or 'external'.")
    validate_pde_ir(problem)
    if problem.conditions and condition_handling == "reject":
        raise ValueError(
            "PDE conditions require explicit external hard-condition handling."
        )
    if any(
        field.coordinates
        and tuple(field.coordinates)
        != tuple(coordinate.name for coordinate in problem.coordinates)
        for field in problem.fields
    ):
        raise ValueError(
            "Spectral residual fields must share the complete compiled coordinates."
        )
    requested = (
        tuple(equation.name for equation in problem.equations)
        if equations is None
        else tuple(str(name) for name in equations)
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("equations must select unique non-empty PDE equation names.")
    by_name = {equation.name: equation for equation in problem.equations}
    unknown_equations = set(requested) - set(by_name)
    if unknown_equations:
        raise KeyError(f"Unknown PDE equations: {tuple(sorted(unknown_equations))}.")
    selected = tuple(by_name[name] for name in requested)
    expressions = tuple(equation.residual for equation in selected)
    coordinate_axes = _all_coordinate_axes(problem, discretization)
    types = tuple(
        infer_expression_type(expression, problem) for expression in expressions
    )
    functional = frozenset(
        parameter.name for parameter in problem.parameters if parameter.functional
    )
    degrees = tuple(
        _residual_degree(expression, functional) for expression in expressions
    )
    required_degree = (
        None
        if any(value is None for value in degrees)
        else max(1, max(int(value) for value in degrees if value is not None))
    )
    nonlinear = any(value is None or value > 1 for value in degrees)
    prepared_method = method.prepare(
        discretization,
        required_polynomial_degree=required_degree,
        nonlinear=nonlinear,
    )
    exact = bool(prepared_method.dealiasing.report.exact) and all(
        _coordinate_representation_exact(
            expression,
            dict(coordinate_axes),
            discretization,
        )
        for expression in expressions
    )
    if scope == "full" and nonlinear:
        exact = exact and prepared_method.dealiasing.report.kind == "closure"
    if require_exact and not exact:
        raise ValueError(
            "The selected spectral realization cannot certify the requested "
            "full residual objective."
        )
    supplied = {} if parameter_values is None else dict(parameter_values)
    unknown_parameters = set(supplied) - {
        parameter.name for parameter in problem.parameters
    }
    if unknown_parameters:
        raise KeyError(
            f"Unknown spectral PDE parameter values: {tuple(sorted(unknown_parameters))}."
        )
    defaults = tuple(
        supplied.get(parameter.name, parameter.value) for parameter in problem.parameters
    )
    parameter_fingerprints = {
        parameter.name: (
            None if value is None else array_tree_fingerprint(jnp.asarray(value))
        )
        for parameter, value in zip(problem.parameters, defaults, strict=True)
    }
    scales_by_name = {} if equation_scales is None else dict(equation_scales)
    if len(requested) == 1 and not scales_by_name:
        scales_by_name = {requested[0]: 1.0}
    if set(scales_by_name) != set(requested):
        raise ValueError("equation_scales must exactly cover selected equations.")
    scales = tuple(float(scales_by_name[name]) for name in requested)
    if any(not isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("equation_scales must be finite and positive.")
    layout = SpectralStateLayout(problem.fields, discretization)
    evaluator = _SpectralEvaluator(
        problem,
        expressions,
        requested,
        tuple(value.components for value in types),
        layout,
        discretization,
        prepared_method,
        defaults,
        coordinate_axes,
        None,
        _interior_region_axes(problem, coordinate_axes),
    )
    report = SpectralResidualCompilationReport(
        trial_shape=discretization.modal_shape,
        evaluation_shape=prepared_method.dealiasing.evaluation.modal_shape,
        equation_names=requested,
        scope=scope,
        condition_handling=condition_handling,
        maximum_polynomial_degree=required_degree,
        exact=exact,
        coefficient_itemsize=jnp.dtype(
            discretization.plan.precision.coefficient_dtype
        ).itemsize,
    )
    compilation_id = canonical_fingerprint(
        {
            "kind": "spectral-residual-compiler",
            "problem": problem.canonical_hash,
            "discretization": discretization.prepared_id,
            "method": prepared_method.prepared_id,
            "equations": list(requested),
            "scales": list(scales),
            "scope": scope,
            "require_exact": bool(require_exact),
            "condition_handling": condition_handling,
            "parameters": parameter_fingerprints,
            "report": report.report_id,
        }
    )
    return CompiledSpectralResidual(
        layout,
        discretization,
        prepared_method,
        evaluator,
        jnp.asarray(scales),
        report,
        equation_names=requested,
        equation_components=tuple(value.components for value in types),
        compilation_id=compilation_id,
        source_hash=problem.canonical_hash,
        scope=scope,
    )


__all__ = [
    "CompiledSpectralResidual",
    "SpectralConditionHandling",
    "SpectralResidualCompilationReport",
    "SpectralResidualScope",
    "compile_spectral_residual",
]

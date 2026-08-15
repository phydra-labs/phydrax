#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._policies import FailurePolicy
from ._self_adjoint_spectrum import (
    prepare_self_adjoint_spectrum,
    PreparedSelfAdjointSpectrum,
    SelfAdjointSpectrumPlan,
    SelfAdjointSpectrumPolicy,
    SelfAdjointSpectrumStatus,
)
from ._spectral_derivatives import (
    density_from_projector,
    perturbation_in_eigenbasis,
)


SpectralFunctionDifferentiation: TypeAlias = Literal["none", "frechet"]


class AbstractSpectralFunction(StrictModule):
    """Stable scalar function and divided-difference contract."""

    function_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def value(self, eigenvalue: ArrayLike, /) -> Array:
        """Evaluate the scalar function on real eigenvalues."""

    @abc.abstractmethod
    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        """Evaluate the scalar derivative."""

    def divided_difference(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate a stable Loewner divided difference."""
        return _stable_divided_difference(self, left, right)

    @abc.abstractmethod
    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        """Return whether values and uncertainty lie inside the differentiable domain."""


class PolynomialSpectralFunction(AbstractSpectralFunction):
    """Polynomial f(x) = Σ cᵢ xⁱ with trainable coefficients."""

    coefficients: Array

    def __init__(self, coefficients: ArrayLike, /):
        values = jnp.asarray(coefficients)
        if values.ndim != 1 or values.shape[0] == 0:
            raise ValueError("coefficients must be a non-empty rank-one array.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Polynomial coefficients must have a real floating dtype.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Polynomial coefficients must be finite.",
        )
        self.coefficients = values
        self.function_id = f"polynomial:{values.shape[0]}:{values.dtype}"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        x = jnp.asarray(eigenvalue)
        result = jnp.zeros_like(x)
        for coefficient in reversed(self.coefficients):
            result = result * x + coefficient
        return result

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        x = jnp.asarray(eigenvalue)
        result = jnp.zeros_like(x)
        for degree in range(self.coefficients.shape[0] - 1, 0, -1):
            result = result * x + degree * self.coefficients[degree]
        return result

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        del uncertainty
        values = jnp.asarray(eigenvalues)
        return (
            jnp.all(jnp.isfinite(values), axis=-1)
            & jnp.all(jnp.isfinite(self.coefficients))
        )


class FermiDiracSpectralFunction(AbstractSpectralFunction):
    """Fermi–Dirac occupation f(x) = sigmoid((μ - x) / T)."""

    chemical_potential: Array
    temperature: Array

    def __init__(
        self,
        chemical_potential: ArrayLike,
        temperature: ArrayLike,
        /,
    ):
        chemical = _real_scalar(chemical_potential, "chemical_potential")
        thermal = _real_scalar(temperature, "temperature")
        if chemical.dtype != thermal.dtype:
            raise TypeError("chemical_potential and temperature must share a dtype.")
        thermal = eqx.error_if(
            thermal,
            thermal <= 0,
            "Fermi-Dirac temperature must be strictly positive.",
        )
        self.chemical_potential = chemical
        self.temperature = thermal
        self.function_id = f"fermi-dirac:{chemical.dtype}"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        x = jnp.asarray(eigenvalue)
        return jax.nn.sigmoid((self.chemical_potential - x) / self.temperature)

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        occupation = self.value(eigenvalue)
        return -occupation * (1 - occupation) / self.temperature

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        del uncertainty
        values = jnp.asarray(eigenvalues)
        return (
            jnp.all(jnp.isfinite(values), axis=-1)
            & jnp.isfinite(self.chemical_potential)
            & jnp.isfinite(self.temperature)
            & (self.temperature > 0)
        )


class ExponentialSpectralFunction(AbstractSpectralFunction):
    """Matrix exponential spectral function."""

    def __init__(self):
        self.function_id = "exponential"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        return jnp.exp(jnp.asarray(eigenvalue))

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        return self.value(eigenvalue)

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        del uncertainty
        return jnp.all(jnp.isfinite(jnp.asarray(eigenvalues)), axis=-1)


class LogarithmSpectralFunction(AbstractSpectralFunction):
    """Principal real matrix logarithm on certified positive spectra."""

    def __init__(self):
        self.function_id = "logarithm"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        return jnp.log(jnp.asarray(eigenvalue))

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        return jnp.reciprocal(jnp.asarray(eigenvalue))

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        values = jnp.asarray(eigenvalues)
        error = jnp.broadcast_to(jnp.asarray(uncertainty), values.shape)
        return jnp.all(jnp.isfinite(values), axis=-1) & jnp.all(
            values - error > 0,
            axis=-1,
        )


class SquareRootSpectralFunction(AbstractSpectralFunction):
    """Principal real square root on certified positive spectra."""

    def __init__(self):
        self.function_id = "square-root"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        return jnp.sqrt(jnp.asarray(eigenvalue))

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        return 0.5 / jnp.sqrt(jnp.asarray(eigenvalue))

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        values = jnp.asarray(eigenvalues)
        error = jnp.broadcast_to(jnp.asarray(uncertainty), values.shape)
        return jnp.all(jnp.isfinite(values), axis=-1) & jnp.all(
            values - error > 0,
            axis=-1,
        )


class InverseSquareRootSpectralFunction(AbstractSpectralFunction):
    """Inverse principal square root on certified positive spectra."""

    def __init__(self):
        self.function_id = "inverse-square-root"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        return jax.lax.rsqrt(jnp.asarray(eigenvalue))

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        x = jnp.asarray(eigenvalue)
        return -0.5 * jax.lax.rsqrt(x) / x

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        values = jnp.asarray(eigenvalues)
        error = jnp.broadcast_to(jnp.asarray(uncertainty), values.shape)
        return jnp.all(jnp.isfinite(values), axis=-1) & jnp.all(
            values - error > 0,
            axis=-1,
        )


class FractionalPowerSpectralFunction(AbstractSpectralFunction):
    """Real scalar power with explicit real-domain certification."""

    power: float = eqx.field(static=True)
    integer_power: bool = eqx.field(static=True)

    def __init__(self, power: float, /):
        exponent = float(power)
        if not math.isfinite(exponent):
            raise ValueError("power must be finite.")
        self.power = exponent
        self.integer_power = exponent.is_integer()
        self.function_id = f"fractional-power:{exponent.hex()}"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        return jnp.asarray(eigenvalue) ** self.power

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        x = jnp.asarray(eigenvalue)
        return self.power * x ** (self.power - 1)

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        values = jnp.asarray(eigenvalues)
        error = jnp.broadcast_to(jnp.asarray(uncertainty), values.shape)
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        if not self.integer_power:
            return finite & jnp.all(values - error > 0, axis=-1)
        if self.power < 0:
            return finite & jnp.all(jnp.abs(values) - error > 0, axis=-1)
        return finite


class ResolventSpectralFunction(AbstractSpectralFunction):
    """Resolvent f(x) = 1 / (x - shift) away from certified poles."""

    shift: Array

    def __init__(self, shift: ArrayLike, /):
        value = _real_scalar(shift, "shift")
        self.shift = value
        self.function_id = f"resolvent:{value.dtype}"

    def value(self, eigenvalue: ArrayLike, /) -> Array:
        return jnp.reciprocal(jnp.asarray(eigenvalue) - self.shift)

    def derivative(self, eigenvalue: ArrayLike, /) -> Array:
        difference = jnp.asarray(eigenvalue) - self.shift
        return -jnp.reciprocal(difference * difference)

    def validate_domain(self, eigenvalues: ArrayLike, uncertainty: ArrayLike = 0, /) -> Array:
        values = jnp.asarray(eigenvalues)
        error = jnp.broadcast_to(jnp.asarray(uncertainty), values.shape)
        return (
            jnp.all(jnp.isfinite(values), axis=-1)
            & jnp.isfinite(self.shift)
            & jnp.all(jnp.abs(values - self.shift) - error > 0, axis=-1)
        )


class SelfAdjointSpectralOperatorStatus(IntEnum):
    """Status of a smooth self-adjoint spectral operator."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    DOMAIN_ERROR = 2
    NONFINITE = 3
    OPERATOR_RESIDUAL_TOO_LARGE = 4


class SelfAdjointSpectralOperatorPolicy(StrictModule):
    """Differentiation, residual, and failure requirements."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    differentiation: SpectralFunctionDifferentiation = eqx.field(static=True)
    failure: FailurePolicy

    def __init__(
        self,
        *,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        differentiation: SpectralFunctionDifferentiation = "none",
        failure: FailurePolicy | None = None,
    ):
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (relative, absolute)
        ):
            raise ValueError("Spectral operator tolerances must be finite and non-negative.")
        if differentiation not in ("none", "frechet"):
            raise ValueError("differentiation must be 'none' or 'frechet'.")
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.differentiation = differentiation
        self.failure = failure_


class SelfAdjointSpectralOperatorDiagnostics(StrictModule):
    """Function-domain, reconstruction, metric, and finite-value evidence."""

    function_values: Array
    eigenvalue_uncertainty: Array
    reconstruction_residual: Array
    metric_self_adjointness_error: Array
    density_identity_error: Array
    finite: Array
    domain_valid: Array
    converged: Array


class SelfAdjointSpectralOperatorProvenance(StrictModule):
    """Source spectrum and scalar-function identity."""

    problem_id: str = eqx.field(static=True)
    spectrum_plan_id: str = eqx.field(static=True)
    function_id: str = eqx.field(static=True)
    differentiation: SpectralFunctionDifferentiation = eqx.field(static=True)
    method: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class SelfAdjointSpectralOperator(StrictModule):
    """Matrix function, density kernel, trace, and spectral evidence."""

    operator: Array
    density_kernel: Array
    trace: Array
    status: Array
    diagnostics: SelfAdjointSpectralOperatorDiagnostics
    provenance: SelfAdjointSpectralOperatorProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SelfAdjointSpectralOperatorStatus.SUCCESS)

    def apply_coordinates(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != (self.operator.shape[1],):
            raise ValueError("vector must match the spectral operator dimension.")
        return self.operator @ value


def self_adjoint_spectral_operator(
    spectrum_or_problem,
    function: AbstractSpectralFunction,
    /,
    *,
    policy: SelfAdjointSpectralOperatorPolicy | None = None,
    spectrum_policy: SelfAdjointSpectrumPolicy | SelfAdjointSpectrumPlan | None = None,
) -> SelfAdjointSpectralOperator:
    """Evaluate a smooth scalar function of one self-adjoint operator pencil."""
    if not isinstance(function, AbstractSpectralFunction):
        raise TypeError("function must implement AbstractSpectralFunction.")
    selected_policy = (
        SelfAdjointSpectralOperatorPolicy() if policy is None else policy
    )
    if not isinstance(selected_policy, SelfAdjointSpectralOperatorPolicy):
        raise TypeError("policy must be a SelfAdjointSpectralOperatorPolicy or None.")
    if isinstance(spectrum_or_problem, PreparedSelfAdjointSpectrum):
        if spectrum_policy is not None:
            raise ValueError("spectrum_policy must be omitted for prepared spectrum state.")
        spectrum = spectrum_or_problem
    else:
        spectrum = prepare_self_adjoint_spectrum(spectrum_or_problem, spectrum_policy)
    values = spectrum.eigenvalues
    uncertainty = _eigenvalue_uncertainty(spectrum)
    domain_valid = function.validate_domain(values, uncertainty)
    function_values = function.value(values)
    operator = (
        spectrum.eigenvectors
        * function_values.astype(spectrum.eigenvectors.dtype)[..., None, :]
    ) @ spectrum.inverse_basis
    density = density_from_projector(operator, spectrum.paired_metric)
    diagnostics, status = _spectral_operator_evidence(
        spectrum,
        function_values,
        uncertainty,
        domain_valid,
        operator,
        density,
        selected_policy,
    )
    derivative_valid = status == int(SelfAdjointSpectralOperatorStatus.SUCCESS)
    if selected_policy.differentiation == "frechet":
        operator = jax.lax.cond(
            jnp.all(derivative_valid),
            lambda value: _attach_spectral_operator_derivative(
                spectrum.problem,
                function,
                value,
                spectrum.eigenvalues,
                spectrum.eigenvectors,
                spectrum.inverse_basis,
            ),
            jax.lax.stop_gradient,
            operator,
        )
        density = jax.lax.cond(
            jnp.all(derivative_valid),
            lambda value: _attach_spectral_density_derivative(
                spectrum.problem,
                function,
                value,
                jax.lax.stop_gradient(operator),
                spectrum.paired_metric,
                spectrum.eigenvalues,
                spectrum.eigenvectors,
                spectrum.inverse_basis,
            ),
            jax.lax.stop_gradient,
            density,
        )
    else:
        operator = jax.lax.stop_gradient(operator)
        density = jax.lax.stop_gradient(density)
    if selected_policy.failure.mode == "error":
        operator = eqx.error_if(
            operator,
            jnp.any(status != int(SelfAdjointSpectralOperatorStatus.SUCCESS)),
            "Self-adjoint spectral operator did not satisfy its numerical contract.",
        )
    return SelfAdjointSpectralOperator(
        operator=operator,
        density_kernel=density,
        trace=jnp.real(jnp.trace(operator, axis1=-2, axis2=-1)),
        status=status,
        diagnostics=diagnostics,
        provenance=SelfAdjointSpectralOperatorProvenance(
            problem_id=spectrum.problem.problem_id,
            spectrum_plan_id=spectrum.plan.plan_id,
            function_id=function.function_id,
            differentiation=selected_policy.differentiation,
            method="Loewner divided-difference Frechet derivative",
            numeric_version=spectrum.eigen_prepared.numeric_version,
        ),
    )


@eqx.filter_custom_jvp
def _attach_spectral_operator_derivative(
    problem,
    function,
    operator,
    eigenvalues,
    eigenvectors,
    inverse_basis,
):
    del problem, function, eigenvalues, eigenvectors, inverse_basis
    return operator


@_attach_spectral_operator_derivative.def_jvp
def _spectral_operator_jvp(primals, tangents):
    problem, function, operator, eigenvalues, eigenvectors, inverse_basis = primals
    problem_tangent, function_tangent, _, _, _, _ = tangents
    derivative, _, _ = _spectral_operator_tangent(
        problem,
        problem_tangent,
        function,
        function_tangent,
        eigenvalues,
        eigenvectors,
        inverse_basis,
    )
    return operator, derivative


@eqx.filter_custom_jvp
def _attach_spectral_density_derivative(
    problem,
    function,
    density,
    operator,
    paired_metric,
    eigenvalues,
    eigenvectors,
    inverse_basis,
):
    del (
        problem,
        function,
        operator,
        paired_metric,
        eigenvalues,
        eigenvectors,
        inverse_basis,
    )
    return density


@_attach_spectral_density_derivative.def_jvp
def _spectral_density_jvp(primals, tangents):
    (
        problem,
        function,
        density,
        operator,
        paired_metric,
        eigenvalues,
        eigenvectors,
        inverse_basis,
    ) = primals
    problem_tangent, function_tangent, _, _, _, _, _, _ = tangents
    operator_derivative, paired_metric_tangent, _ = _spectral_operator_tangent(
        problem,
        problem_tangent,
        function,
        function_tangent,
        eigenvalues,
        eigenvectors,
        inverse_basis,
    )
    right_hand_side = operator_derivative - density @ paired_metric_tangent
    derivative = jnp.swapaxes(
        jnp.linalg.solve(
            jnp.swapaxes(paired_metric, -1, -2),
            jnp.swapaxes(right_hand_side, -1, -2),
        ),
        -1,
        -2,
    )
    return density, derivative


def _spectral_operator_tangent(
    problem,
    problem_tangent,
    function,
    function_tangent,
    eigenvalues,
    eigenvectors,
    inverse_basis,
):
    perturbation, paired_metric_tangent = perturbation_in_eigenbasis(
        problem,
        problem_tangent,
        eigenvalues,
        eigenvectors,
    )
    left = eigenvalues[..., :, None]
    right = eigenvalues[..., None, :]
    loewner = function.divided_difference(left, right).astype(eigenvectors.dtype)
    function_values, parameter_tangent = eqx.filter_jvp(
        lambda current: current.value(eigenvalues),
        (function,),
        (function_tangent,),
    )
    if parameter_tangent is None:
        parameter_tangent = jnp.zeros_like(function_values)
    derivative_in_basis = (
        loewner * perturbation
        + parameter_tangent.astype(eigenvectors.dtype)[..., None, :]
        * jnp.eye(eigenvalues.shape[-1], dtype=eigenvectors.dtype)
    )
    derivative = eigenvectors @ derivative_in_basis @ inverse_basis
    return derivative, paired_metric_tangent, derivative_in_basis


def _spectral_operator_evidence(
    spectrum,
    function_values,
    uncertainty,
    domain_valid,
    operator,
    density,
    policy,
):
    expected_images = (
        spectrum.eigenvectors
        * function_values.astype(spectrum.eigenvectors.dtype)[..., None, :]
    )
    reconstruction = jnp.linalg.norm(
        operator @ spectrum.eigenvectors - expected_images,
        axis=(-2, -1),
    )
    metric_adjointness = jnp.linalg.norm(
        jnp.conj(jnp.swapaxes(operator, -1, -2)) @ spectrum.paired_metric
        - spectrum.paired_metric @ operator,
        axis=(-2, -1),
    )
    density_identity = jnp.linalg.norm(
        operator - density @ spectrum.paired_metric,
        axis=(-2, -1),
    )
    residual = reconstruction + metric_adjointness + density_identity
    scale = (
        jnp.linalg.norm(operator, axis=(-2, -1))
        + jnp.linalg.norm(function_values, axis=-1)
        + 1
    )
    tolerance = policy.absolute_tolerance + policy.relative_tolerance * scale
    finite = (
        jnp.all(jnp.isfinite(function_values), axis=-1)
        & jnp.all(jnp.isfinite(operator), axis=(-2, -1))
        & jnp.all(jnp.isfinite(density), axis=(-2, -1))
        & jnp.isfinite(residual)
    )
    source_success = spectrum.status == int(SelfAdjointSpectrumStatus.SUCCESS)
    residual_ok = residual <= tolerance
    status = jnp.where(
        ~source_success,
        int(SelfAdjointSpectralOperatorStatus.SOURCE_FAILURE),
        jnp.where(
            ~domain_valid,
            int(SelfAdjointSpectralOperatorStatus.DOMAIN_ERROR),
            jnp.where(
                ~finite,
                int(SelfAdjointSpectralOperatorStatus.NONFINITE),
                jnp.where(
                    ~residual_ok,
                    int(SelfAdjointSpectralOperatorStatus.OPERATOR_RESIDUAL_TOO_LARGE),
                    int(SelfAdjointSpectralOperatorStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = SelfAdjointSpectralOperatorDiagnostics(
        function_values=function_values,
        eigenvalue_uncertainty=uncertainty,
        reconstruction_residual=reconstruction,
        metric_self_adjointness_error=metric_adjointness,
        density_identity_error=density_identity,
        finite=finite,
        domain_valid=domain_valid,
        converged=status == int(SelfAdjointSpectralOperatorStatus.SUCCESS),
    )
    return diagnostics, status

def _eigenvalue_uncertainty(spectrum):
    scale = jnp.maximum(jnp.abs(spectrum.eigenvalues), 1)
    return 4 * jnp.maximum(
        spectrum.source_diagnostics.residual_norms,
        spectrum.source_diagnostics.relative_residuals * scale,
    )


def _stable_divided_difference(function, left, right):
    x = jnp.asarray(left)
    y = jnp.asarray(right)
    difference = x - y
    scale = jnp.maximum(jnp.maximum(jnp.abs(x), jnp.abs(y)), 1)
    close = jnp.abs(difference) <= jnp.sqrt(jnp.finfo(x.dtype).eps) * scale
    safe_difference = jnp.where(close, 1, difference)
    quotient = (function.value(x) - function.value(y)) / safe_difference
    midpoint_derivative = function.derivative((x + y) / 2)
    return jnp.where(close, midpoint_derivative, quotient)


def _real_scalar(value, name):
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating dtype.")
    return eqx.error_if(array, ~jnp.isfinite(array), f"{name} must be finite.")


__all__ = [
    "AbstractSpectralFunction",
    "ExponentialSpectralFunction",
    "FermiDiracSpectralFunction",
    "FractionalPowerSpectralFunction",
    "InverseSquareRootSpectralFunction",
    "LogarithmSpectralFunction",
    "PolynomialSpectralFunction",
    "ResolventSpectralFunction",
    "SelfAdjointSpectralOperator",
    "SelfAdjointSpectralOperatorDiagnostics",
    "SelfAdjointSpectralOperatorPolicy",
    "SelfAdjointSpectralOperatorProvenance",
    "SelfAdjointSpectralOperatorStatus",
    "SpectralFunctionDifferentiation",
    "SquareRootSpectralFunction",
    "self_adjoint_spectral_operator",
]

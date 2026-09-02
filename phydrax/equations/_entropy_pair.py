#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix import CoordinateChart, HessianGeometry
from ..metrix._utils import _pointwise_jacfwd
from ..metrix._validation import MetricValidationReport, validate_metric
from ._hyperbolic_systems import (
    AbstractConservationSystem,
    CompressibleNavierStokesSystem,
    EulerSystem,
)


EntropyFunction = Callable[[Array], Array]
EntropyVariablesFunction = Callable[[Array], Array]
EntropyFluxFunction = Callable[[Array, int, Any], Array]
AdmissibilityFunction = Callable[[Array], Array]


def _state_array(
    value: ArrayLike,
    component_count: int,
    name: str,
    /,
) -> Array:
    state = jnp.asarray(value)
    if state.ndim < 1 or state.shape[-1] != component_count:
        raise ValueError(
            f"{name} must have trailing component dimension {component_count}; "
            f"got {state.shape}."
        )
    if not jnp.issubdtype(state.dtype, jnp.floating):
        raise TypeError(f"{name} must use real floating-point coordinates.")
    return state


def _scalar_output(value: ArrayLike, expected_shape: tuple[int, ...], name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must return shape {expected_shape}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must return real floating-point values.")
    return array


def _variable_output(
    value: ArrayLike,
    expected_shape: tuple[int, ...],
    name: str,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must return state shape {expected_shape}; got {array.shape}."
        )
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must return real floating-point values.")
    return array


def _axis(axis: int, dimension: int, /) -> int:
    axis_ = int(axis)
    if axis_ != axis:
        raise TypeError("Entropy flux axis must be an integer.")
    if not 0 <= axis_ < dimension:
        raise ValueError(f"Entropy flux axis must lie in [0, {dimension}).")
    return axis_


class ConvexEntropyPair(StrictModule, NonTrainableState):
    """A domain-aware convex entropy and entropy-flux pair for one system."""

    system: AbstractConservationSystem
    entropy_function: EntropyFunction
    entropy_variables_function: EntropyVariablesFunction
    entropy_flux_function: EntropyFluxFunction
    admissible_function: AdmissibilityFunction
    state_chart: CoordinateChart
    entropy_id: str = eqx.field(static=True)
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AbstractConservationSystem,
        entropy: EntropyFunction,
        entropy_variables: EntropyVariablesFunction,
        entropy_flux: EntropyFluxFunction,
        admissible: AdmissibilityFunction,
        /,
        *,
        entropy_id: str,
    ):
        if not isinstance(system, AbstractConservationSystem):
            raise TypeError("system must be an AbstractConservationSystem.")
        for function, name in (
            (entropy, "entropy"),
            (entropy_variables, "entropy_variables"),
            (entropy_flux, "entropy_flux"),
            (admissible, "admissible"),
        ):
            if not callable(function):
                raise TypeError(f"{name} must be callable.")
        identifier = str(entropy_id)
        if not identifier:
            raise ValueError("entropy_id must be non-empty.")
        chart = CoordinateChart(
            f"entropy:{identifier}",
            system.component_names,
        )
        pair_id = canonical_fingerprint(
            {
                "kind": "convex-entropy-pair",
                "system": system.system_id,
                "entropy_id": identifier,
                "chart": chart.name,
                "coordinates": chart.coordinates,
            }
        )
        self.system = system
        self.entropy_function = entropy
        self.entropy_variables_function = entropy_variables
        self.entropy_flux_function = entropy_flux
        self.admissible_function = admissible
        self.state_chart = chart
        self.entropy_id = identifier
        self.pair_id = pair_id

    @property
    def dimension(self) -> int:
        return self.system.dimension

    @property
    def component_count(self) -> int:
        return self.system.component_count

    def _state(self, value: ArrayLike, name: str, /) -> Array:
        state = _state_array(value, self.component_count, name)
        valid_array = self.admissible(state)
        if valid_array.shape != state.shape[:-1]:
            raise ValueError(
                "Entropy admissibility must return the state leading shape; "
                f"got {valid_array.shape} for {state.shape}."
            )
        return eqx.error_if(
            state,
            jnp.any(~valid_array),
            f"{name} contains states outside entropy pair {self.pair_id!r}.",
        )

    def _state_unchecked(self, value: ArrayLike, name: str, /) -> Array:
        return _state_array(value, self.component_count, name)

    def admissible(self, state: ArrayLike, /) -> Array:
        value = self._state_unchecked(state, "Entropy state")
        valid = jnp.asarray(self.admissible_function(value))
        if valid.dtype != jnp.bool_:
            raise TypeError(
                "Entropy admissibility predicates must return Boolean values."
            )
        if valid.shape != value.shape[:-1]:
            raise ValueError(
                "Entropy admissibility must return the state leading shape; "
                f"got {valid.shape} for {value.shape}."
            )
        return valid & jnp.all(jnp.isfinite(value), axis=-1)

    def _entropy_unchecked(self, state: Array, /) -> Array:
        return _scalar_output(
            self.entropy_function(state),
            state.shape[:-1],
            "Entropy function",
        )

    def entropy(self, state: ArrayLike, /) -> Array:
        return self._entropy_unchecked(self._state(state, "Entropy state"))

    def _entropy_variables_unchecked(self, state: Array, /) -> Array:
        return _variable_output(
            self.entropy_variables_function(state),
            state.shape,
            "Entropy variables",
        )

    def entropy_variables(self, state: ArrayLike, /) -> Array:
        return self._entropy_variables_unchecked(
            self._state(state, "Entropy-variable state")
        )

    def _entropy_flux_unchecked(
        self,
        state: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        axis_ = _axis(axis, self.dimension)
        return _scalar_output(
            self.entropy_flux_function(state, axis_, args),
            state.shape[:-1],
            "Entropy flux",
        )

    def entropy_flux(
        self,
        state: ArrayLike,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        value = self._state(state, "Entropy-flux state")
        return self._entropy_flux_unchecked(value, axis, args)

    def normal_entropy_flux(
        self,
        state: ArrayLike,
        normal: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Contract the entropy-flux vector with an arbitrary physical normal."""
        value = self._state(state, "Normal entropy-flux state")
        normal_ = jnp.asarray(normal)
        if normal_.shape[-1:] != (self.dimension,):
            raise ValueError("Entropy-flux normal has the wrong dimension.")
        fluxes = jnp.stack(
            tuple(
                self._entropy_flux_unchecked(value, axis, args)
                for axis in range(self.dimension)
            ),
            axis=-1,
        )
        return ein.contract("...d,...d->...", fluxes, normal_, backend="jax")

    def relative_entropy(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        left_value = self._state(left, "Left relative-entropy state")
        right_value = self._state(right, "Right relative-entropy state")
        if left_value.shape != right_value.shape:
            raise ValueError("Relative-entropy states must have equal shapes.")
        return (
            self._entropy_unchecked(left_value)
            - self._entropy_unchecked(right_value)
            - jnp.sum(
                self._entropy_variables_unchecked(right_value)
                * (left_value - right_value),
                axis=-1,
            )
        )

    def symmetrizer_action(
        self,
        state: ArrayLike,
        direction: ArrayLike,
        /,
    ) -> Array:
        value = self._state(state, "Symmetrizer state")
        vector = _state_array(direction, self.component_count, "Symmetrizer direction")
        if value.shape != vector.shape:
            raise ValueError("Symmetrizer direction must match the state shape.")

        def action(point: Array, tangent: Array, /) -> Array:
            return jax.jvp(
                self.entropy_variables_function,
                (point,),
                (tangent,),
            )[1]

        if value.ndim == 1:
            return _variable_output(action(value, vector), value.shape, "Symmetrizer")
        leading_shape = value.shape[:-1]
        flat_value = value.reshape((-1, self.component_count))
        flat_vector = vector.reshape((-1, self.component_count))
        result = jax.vmap(action)(flat_value, flat_vector)
        return result.reshape(leading_shape + (self.component_count,))

    def _symmetrizer_action_unchecked(
        self,
        state: Array,
        direction: Array,
        /,
    ) -> Array:
        def action(point: Array, tangent: Array, /) -> Array:
            return jax.jvp(
                self.entropy_variables_function,
                (point,),
                (tangent,),
            )[1]

        if state.ndim == 1:
            return action(state, direction)
        leading_shape = state.shape[:-1]
        flat_state = state.reshape((-1, self.component_count))
        flat_direction = direction.reshape((-1, self.component_count))
        result = jax.vmap(action)(flat_state, flat_direction)
        return result.reshape(leading_shape + (self.component_count,))

    def entropy_potential(
        self,
        state: ArrayLike,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        value = self._state(state, "Entropy-potential state")
        axis_ = _axis(axis, self.dimension)
        return self._entropy_potential_unchecked(value, axis_, args)

    def _entropy_potential_unchecked(
        self,
        state: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        flux = _variable_output(
            self.system.physical_flux(state, axis, args),
            state.shape,
            "Conservation-system physical flux",
        )
        return ein.contract(
            "...i,...i->...",
            self._entropy_variables_unchecked(state),
            flux,
        ) - self._entropy_flux_unchecked(state, axis, args)

    def normal_entropy_potential(
        self,
        state: ArrayLike,
        normal: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Return v·(F·n) − q·n for an arbitrary physical normal."""
        value = self._state(state, "Normal entropy-potential state")
        normal_ = jnp.asarray(normal)
        if normal_.shape[-1:] != (self.dimension,):
            raise ValueError("Entropy-potential normal has the wrong dimension.")
        fluxes = jnp.stack(
            tuple(
                _variable_output(
                    self.system.physical_flux(value, axis, args),
                    value.shape,
                    "Conservation-system physical flux",
                )
                for axis in range(self.dimension)
            ),
            axis=-1,
        )
        physical = ein.contract("...id,...d->...i", fluxes, normal_, backend="jax")
        return ein.contract(
            "...i,...i->...",
            self._entropy_variables_unchecked(value),
            physical,
            backend="jax",
        ) - self.normal_entropy_flux(value, normal_, args)

    def interface_entropy_residual(
        self,
        left: ArrayLike,
        right: ArrayLike,
        numerical_flux: ArrayLike,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        left_value = self._state(left, "Left interface state")
        right_value = self._state(right, "Right interface state")
        if left_value.shape != right_value.shape:
            raise ValueError("Interface states must have equal shapes.")
        flux = _state_array(numerical_flux, self.component_count, "Numerical flux")
        if flux.shape != left_value.shape:
            raise ValueError("Numerical flux must match the interface state shape.")
        axis_ = _axis(axis, self.dimension)
        variables_left = self._entropy_variables_unchecked(left_value)
        variables_right = self._entropy_variables_unchecked(right_value)
        return ein.contract(
            "...i,...i->...",
            variables_right - variables_left,
            flux,
        ) - (
            self._entropy_potential_unchecked(right_value, axis_, args)
            - self._entropy_potential_unchecked(left_value, axis_, args)
        )

    def normal_interface_entropy_residual(
        self,
        left: ArrayLike,
        right: ArrayLike,
        numerical_flux: ArrayLike,
        normal: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Tadmor interface residual for one arbitrary-normal shared flux."""
        left_value = self._state(left, "Left normal-interface state")
        right_value = self._state(right, "Right normal-interface state")
        if left_value.shape != right_value.shape:
            raise ValueError("Normal-interface states must have equal shapes.")
        flux = _state_array(numerical_flux, self.component_count, "Numerical flux")
        if flux.shape != left_value.shape:
            raise ValueError("Numerical flux must match the interface state shape.")
        variables_left = self._entropy_variables_unchecked(left_value)
        variables_right = self._entropy_variables_unchecked(right_value)
        return ein.contract(
            "...i,...i->...",
            variables_right - variables_left,
            flux,
            backend="jax",
        ) - (
            self.normal_entropy_potential(right_value, normal, args)
            - self.normal_entropy_potential(left_value, normal, args)
        )

    def hessian_geometry(self) -> HessianGeometry:
        return HessianGeometry(self._entropy_point, chart=self.state_chart)

    def _entropy_point(self, state: Array, /) -> Array:
        return _scalar_output(
            self.entropy_function(state),
            (),
            "Pointwise entropy",
        )


class ConvexEntropyValidationReport(StrictModule):
    """Representative-state validation evidence for one convex entropy pair."""

    valid: Array
    finite: Array
    admissible: Array
    maximum_entropy_variable_residual: Array
    maximum_flux_compatibility_residual: Array
    maximum_flux_symmetrizer_asymmetry: Array
    maximum_symmetrizer_asymmetry: Array
    minimum_relative_entropy: Array
    maximum_diagonal_relative_entropy: Array
    metric_validation: MetricValidationReport
    axes: tuple[int, ...] = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        *,
        valid: ArrayLike,
        finite: ArrayLike,
        admissible: ArrayLike,
        maximum_entropy_variable_residual: ArrayLike,
        maximum_flux_compatibility_residual: ArrayLike,
        maximum_flux_symmetrizer_asymmetry: ArrayLike,
        maximum_symmetrizer_asymmetry: ArrayLike,
        minimum_relative_entropy: ArrayLike,
        maximum_diagonal_relative_entropy: ArrayLike,
        metric_validation: MetricValidationReport,
        axes: Sequence[int],
        precision_evidence: PrecisionEvidenceEnvelope,
    ):
        if not isinstance(metric_validation, MetricValidationReport):
            raise TypeError("metric_validation must be a MetricValidationReport.")
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be a PrecisionEvidenceEnvelope.")
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.admissible = jnp.asarray(admissible, dtype=bool)
        self.maximum_entropy_variable_residual = jnp.asarray(
            maximum_entropy_variable_residual
        )
        self.maximum_flux_compatibility_residual = jnp.asarray(
            maximum_flux_compatibility_residual
        )
        self.maximum_flux_symmetrizer_asymmetry = jnp.asarray(
            maximum_flux_symmetrizer_asymmetry
        )
        self.maximum_symmetrizer_asymmetry = jnp.asarray(maximum_symmetrizer_asymmetry)
        self.minimum_relative_entropy = jnp.asarray(minimum_relative_entropy)
        self.maximum_diagonal_relative_entropy = jnp.asarray(
            maximum_diagonal_relative_entropy
        )
        self.metric_validation = metric_validation
        self.axes = tuple(int(axis) for axis in axes)
        self.precision_evidence = precision_evidence


def _precision_policy(
    precision: GeometryPrecisionPolicy | None,
    /,
) -> GeometryPrecisionPolicy:
    policy = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(policy, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    return policy


def _maximum_abs(value: Array, precision: GeometryPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.max(jnp.abs(value), initial=0.0))


def _axis_tuple(axes: Sequence[int] | None, dimension: int, /) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(dimension))
    result = tuple(int(axis) for axis in axes)
    if not result:
        raise ValueError("Entropy axes must be non-empty.")
    if len(set(result)) != len(result):
        raise ValueError("Entropy axes must be unique.")
    if any(axis < 0 or axis >= dimension for axis in result):
        raise ValueError(f"Entropy axes must lie in [0, {dimension}).")
    return result


def validate_convex_entropy_pair(
    pair: ConvexEntropyPair,
    states: ArrayLike,
    /,
    *,
    comparison_states: ArrayLike | None = None,
    axes: Sequence[int] | None = None,
    args: Any = None,
    variable_tolerance: float = 1e-8,
    flux_tolerance: float = 1e-8,
    symmetry_tolerance: float = 1e-8,
    relative_entropy_tolerance: float = 1e-8,
    eigenvalue_floor: float = 0.0,
    maximum_condition_number: float | None = None,
    raise_on_error: bool = True,
    precision: GeometryPrecisionPolicy | None = None,
) -> ConvexEntropyValidationReport:
    """Validate entropy variables, fluxes, convexity, and relative entropy."""
    if not isinstance(pair, ConvexEntropyPair):
        raise TypeError("pair must be a ConvexEntropyPair.")
    tolerances = (
        variable_tolerance,
        flux_tolerance,
        symmetry_tolerance,
        relative_entropy_tolerance,
    )
    if any(not isfinite(float(value)) or float(value) < 0.0 for value in tolerances):
        raise ValueError("Entropy validation tolerances must be finite and non-negative.")
    policy = _precision_policy(precision)
    original = _state_array(states, pair.component_count, "Entropy validation states")
    policy.validate_coordinates(original)
    state = policy.compute(original)
    if comparison_states is None:
        comparison_original = original
        comparison = state
    else:
        comparison_original = _state_array(
            comparison_states,
            pair.component_count,
            "Entropy comparison states",
        )
        if comparison_original.shape != original.shape:
            raise ValueError("Entropy comparison states must match the state shape.")
        policy.validate_coordinates(comparison_original)
        comparison = policy.compute(comparison_original)
    selected_axes = _axis_tuple(axes, pair.dimension)
    admissible = jnp.all(pair.admissible(state)) & jnp.all(pair.admissible(comparison))

    entropy = pair._entropy_unchecked(state)
    variables = pair._entropy_variables_unchecked(state)
    gradient = _pointwise_jacfwd(pair._entropy_point, state, pair.component_count)
    maximum_entropy_variable_residual = _maximum_abs(
        variables - gradient,
        policy,
    )

    hessian = _pointwise_jacfwd(
        pair.entropy_variables_function,
        state,
        pair.component_count,
    )
    hessian_asymmetry = hessian - jnp.swapaxes(hessian, -1, -2)
    maximum_symmetrizer_asymmetry = _maximum_abs(hessian_asymmetry, policy)

    maximum_flux_compatibility_residual = policy.decision(jnp.asarray(0.0))
    maximum_flux_symmetrizer_asymmetry = policy.decision(jnp.asarray(0.0))
    for axis in selected_axes:
        flux_jacobian = _pointwise_jacfwd(
            lambda point, axis_=axis: pair.system.physical_flux(point, axis_, args),
            state,
            pair.component_count,
        )
        flux_gradient = _pointwise_jacfwd(
            lambda point, axis_=axis: pair._entropy_flux_unchecked(
                point,
                axis_,
                args,
            ),
            state,
            pair.component_count,
        )
        compatibility = ein.contract("...i,...ij->...j", variables, flux_jacobian)
        maximum_flux_compatibility_residual = jnp.maximum(
            maximum_flux_compatibility_residual,
            _maximum_abs(flux_gradient - compatibility, policy),
        )
        symmetrized_flux = ein.contract("...ik,...kj->...ij", hessian, flux_jacobian)
        maximum_flux_symmetrizer_asymmetry = jnp.maximum(
            maximum_flux_symmetrizer_asymmetry,
            _maximum_abs(
                symmetrized_flux - jnp.swapaxes(symmetrized_flux, -1, -2),
                policy,
            ),
        )

    comparison_entropy = pair._entropy_unchecked(comparison)
    comparison_variables = pair._entropy_variables_unchecked(comparison)
    relative_entropy = (
        entropy
        - comparison_entropy
        - jnp.sum(
            comparison_variables * (state - comparison),
            axis=-1,
        )
    )
    diagonal = (
        entropy
        - entropy
        - jnp.sum(
            variables * (state - state),
            axis=-1,
        )
    )
    minimum_relative_entropy = policy.decision(jnp.min(relative_entropy))
    maximum_diagonal_relative_entropy = _maximum_abs(diagonal, policy)

    metric_validation = validate_metric(
        pair.hessian_geometry().metric(),
        original,
        symmetry_tolerance=symmetry_tolerance,
        eigenvalue_floor=eigenvalue_floor,
        maximum_condition_number=maximum_condition_number,
        raise_on_error=False,
        precision=policy,
    )
    finite = jnp.all(
        jnp.stack(
            (
                jnp.all(jnp.isfinite(entropy)),
                jnp.all(jnp.isfinite(variables)),
                jnp.all(jnp.isfinite(gradient)),
                jnp.all(jnp.isfinite(hessian)),
                jnp.all(jnp.isfinite(comparison_entropy)),
                jnp.all(jnp.isfinite(comparison_variables)),
                jnp.all(jnp.isfinite(relative_entropy)),
                jnp.all(jnp.isfinite(diagonal)),
                jnp.isfinite(maximum_entropy_variable_residual),
                jnp.isfinite(maximum_flux_compatibility_residual),
                jnp.isfinite(maximum_flux_symmetrizer_asymmetry),
                jnp.isfinite(maximum_symmetrizer_asymmetry),
                jnp.isfinite(minimum_relative_entropy),
                jnp.isfinite(maximum_diagonal_relative_entropy),
                metric_validation.finite,
            )
        )
    )
    valid = (
        finite
        & admissible
        & metric_validation.valid
        & (
            maximum_entropy_variable_residual
            <= jnp.asarray(
                variable_tolerance,
                dtype=maximum_entropy_variable_residual.dtype,
            )
        )
        & (
            maximum_flux_compatibility_residual
            <= jnp.asarray(
                flux_tolerance,
                dtype=maximum_flux_compatibility_residual.dtype,
            )
        )
        & (
            maximum_flux_symmetrizer_asymmetry
            <= jnp.asarray(
                symmetry_tolerance,
                dtype=maximum_flux_symmetrizer_asymmetry.dtype,
            )
        )
        & (
            maximum_symmetrizer_asymmetry
            <= jnp.asarray(
                symmetry_tolerance,
                dtype=maximum_symmetrizer_asymmetry.dtype,
            )
        )
        & (
            minimum_relative_entropy
            >= -jnp.asarray(
                relative_entropy_tolerance,
                dtype=minimum_relative_entropy.dtype,
            )
        )
        & (
            maximum_diagonal_relative_entropy
            <= jnp.asarray(
                relative_entropy_tolerance,
                dtype=maximum_diagonal_relative_entropy.dtype,
            )
        )
    )
    report = ConvexEntropyValidationReport(
        valid=valid,
        finite=finite,
        admissible=admissible,
        maximum_entropy_variable_residual=maximum_entropy_variable_residual,
        maximum_flux_compatibility_residual=maximum_flux_compatibility_residual,
        maximum_flux_symmetrizer_asymmetry=maximum_flux_symmetrizer_asymmetry,
        maximum_symmetrizer_asymmetry=maximum_symmetrizer_asymmetry,
        minimum_relative_entropy=minimum_relative_entropy,
        maximum_diagonal_relative_entropy=maximum_diagonal_relative_entropy,
        metric_validation=metric_validation,
        axes=selected_axes,
        precision_evidence=policy.evidence_for(
            original,
            children={
                "comparison": policy.evidence_for(comparison_original),
                "metric": metric_validation.precision_evidence,
            },
        ),
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Convex entropy pair validation failed: "
            f"finite={bool(jax.device_get(finite))}, "
            f"admissible={bool(jax.device_get(admissible))}, "
            "maximum_entropy_variable_residual="
            f"{float(jax.device_get(maximum_entropy_variable_residual))}, "
            "maximum_flux_compatibility_residual="
            f"{float(jax.device_get(maximum_flux_compatibility_residual))}, "
            "maximum_flux_symmetrizer_asymmetry="
            f"{float(jax.device_get(maximum_flux_symmetrizer_asymmetry))}, "
            "maximum_symmetrizer_asymmetry="
            f"{float(jax.device_get(maximum_symmetrizer_asymmetry))}, "
            f"minimum_relative_entropy={float(jax.device_get(minimum_relative_entropy))}."
        )
    return report


def _euler_entropy(
    system: EulerSystem | CompressibleNavierStokesSystem, state: ArrayLike, /
) -> Array:
    value = jnp.asarray(state)
    density = value[..., 0]
    pressure = system.pressure(value)
    thermodynamic_entropy = jnp.log(pressure) - system.gamma * jnp.log(density)
    return -density * thermodynamic_entropy / (system.gamma - 1.0)


def _euler_entropy_flux(
    system: EulerSystem | CompressibleNavierStokesSystem,
    state: ArrayLike,
    axis: int,
    /,
) -> Array:
    value = jnp.asarray(state)
    density = value[..., 0]
    velocity = value[..., 1 : 1 + system.dimension] / density[..., None]
    return velocity[..., _axis(axis, system.dimension)] * _euler_entropy(system, value)


class _IdealGasEntropyPotential(StrictModule, NonTrainableState):
    system: EulerSystem | CompressibleNavierStokesSystem

    def __init__(self, system: EulerSystem | CompressibleNavierStokesSystem, /):
        self.system = system

    def __call__(self, state: Array, /) -> Array:
        return _euler_entropy(self.system, state)


class _IdealGasEntropyFlux(StrictModule, NonTrainableState):
    system: EulerSystem | CompressibleNavierStokesSystem

    def __init__(self, system: EulerSystem | CompressibleNavierStokesSystem, /):
        self.system = system

    def __call__(self, state: Array, axis: int, _args: Any = None, /) -> Array:
        return _euler_entropy_flux(self.system, state, axis)


def ideal_gas_euler_entropy_pair(
    system: EulerSystem | CompressibleNavierStokesSystem,
    /,
) -> ConvexEntropyPair:
    """Return the mathematical ideal-gas entropy pair matching Euler variables."""
    if not isinstance(system, (EulerSystem, CompressibleNavierStokesSystem)):
        raise TypeError(
            "ideal_gas_euler_entropy_pair requires EulerSystem or "
            "CompressibleNavierStokesSystem."
        )
    return ConvexEntropyPair(
        system,
        _IdealGasEntropyPotential(system),
        system.entropy_variables,
        _IdealGasEntropyFlux(system),
        system.admissible,
        entropy_id="ideal-gas-mathematical-entropy",
    )


__all__ = [
    "ConvexEntropyPair",
    "ConvexEntropyValidationReport",
    "ideal_gas_euler_entropy_pair",
    "validate_convex_entropy_pair",
]

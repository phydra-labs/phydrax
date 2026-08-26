#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._method import (
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
    SpectralDifferentiabilityPolicy,
)
from ._space import TensorSpectralDiscretization


if TYPE_CHECKING:
    from ...equations import ConvexEntropyPair


class SpectralConservationMethodPlan(StrictModule, NonTrainableState):
    """Periodic conservative flux projection and optional entropy diagnostics."""

    pseudospectral: PseudospectralMethodPlan
    flux_polynomial_degree: int | None = eqx.field(static=True)
    entropy_diagnostics: bool = eqx.field(static=True)
    differentiability: SpectralDifferentiabilityPolicy = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        pseudospectral: PseudospectralMethodPlan,
        /,
        *,
        flux_polynomial_degree: int | None = None,
        entropy_diagnostics: bool = False,
        differentiability: SpectralDifferentiabilityPolicy = "smooth_discrete",
    ):
        if not isinstance(pseudospectral, PseudospectralMethodPlan):
            raise TypeError("pseudospectral must be a PseudospectralMethodPlan.")
        degree = None if flux_polynomial_degree is None else int(flux_polynomial_degree)
        if degree is not None and degree < 1:
            raise ValueError("flux_polynomial_degree must be positive or None.")
        if differentiability not in (
            "smooth_discrete",
            "branchwise",
            "smooth_surrogate",
            "unsupported",
        ):
            raise ValueError("Unknown spectral differentiability policy.")
        self.pseudospectral = pseudospectral
        self.flux_polynomial_degree = degree
        self.entropy_diagnostics = bool(entropy_diagnostics)
        self.differentiability = differentiability
        self.method_id = canonical_fingerprint(
            {
                "kind": "spectral-conservation-method",
                "pseudospectral": pseudospectral.method_id,
                "flux_polynomial_degree": degree,
                "entropy_diagnostics": bool(entropy_diagnostics),
                "differentiability": differentiability,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
    ) -> "PreparedSpectralConservationMethod":
        if not all(axis.periodic for axis in discretization.axes):
            raise ValueError("Spectral conservation initially requires periodic axes.")
        nonlinear = self.flux_polynomial_degree is None or self.flux_polynomial_degree > 1
        prepared = self.pseudospectral.prepare(
            discretization,
            required_polynomial_degree=self.flux_polynomial_degree,
            nonlinear=nonlinear,
        )
        return PreparedSpectralConservationMethod(self, discretization, prepared)


class PreparedSpectralConservationMethod(StrictModule, NonTrainableState):
    plan: SpectralConservationMethodPlan
    discretization: TensorSpectralDiscretization
    pseudospectral: PreparedPseudospectralMethod
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralConservationMethodPlan,
        discretization: TensorSpectralDiscretization,
        pseudospectral: PreparedPseudospectralMethod,
        /,
    ):
        self.plan = plan
        self.discretization = discretization
        self.pseudospectral = pseudospectral
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-conservation-method",
                "plan": plan.method_id,
                "discretization": discretization.prepared_id,
                "pseudospectral": pseudospectral.prepared_id,
            }
        )


class SpectralEntropyDiagnostics(StrictModule):
    pair_id: str = eqx.field(static=True)
    total_entropy: Array
    semidiscrete_entropy_rate: Array
    source_entropy_rate: Array
    convective_entropy_rate: Array
    admissible: Array
    precision_evidence: PrecisionEvidenceEnvelope


class SpectralConservationDiagnostics(StrictModule):
    total_integral: Array
    semidiscrete_integral_rate: Array
    source_integral: Array
    conservation_defect: Array
    entropy: SpectralEntropyDiagnostics | None
    precision_evidence: PrecisionEvidenceEnvelope
    method_id: str = eqx.field(static=True)


class PreparedSpectralConservationDynamics(StrictModule):
    """Pure periodic conservative pseudospectral semidiscretization."""

    system: Any
    discretization: TensorSpectralDiscretization
    method: PreparedSpectralConservationMethod
    entropy_pair: Any
    source: Any = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: TensorSpectralDiscretization,
        method: SpectralConservationMethodPlan,
        /,
        *,
        source: Any = None,
        entropy_pair: "ConvexEntropyPair | None" = None,
    ):
        from ...equations import AbstractConservationSystem, ConvexEntropyPair

        if not isinstance(system, AbstractConservationSystem):
            raise TypeError("system must be an AbstractConservationSystem.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if system.dimension != len(discretization.axes):
            raise ValueError("Conservation-system dimension must match spectral rank.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        if entropy_pair is not None:
            if not isinstance(entropy_pair, ConvexEntropyPair):
                raise TypeError("entropy_pair must be a ConvexEntropyPair or None.")
            if entropy_pair.system.system_id != system.system_id:
                raise ValueError("entropy_pair must target the conservation system.")
        prepared = method.prepare(discretization)
        self.system = system
        self.discretization = discretization
        self.method = prepared
        self.entropy_pair = entropy_pair
        self.source = source
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-conservation-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": prepared.prepared_id,
                "entropy_pair": None if entropy_pair is None else entropy_pair.pair_id,
                "source": None if source is None else repr(source),
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.discretization.modal_shape + (self.system.component_count,)

    def _validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Spectral conservation state must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        return value

    def _physical_source(
        self,
        time: Array,
        physical_state: Array,
        args: Any,
        /,
    ) -> Array:
        if self.source is None:
            return jnp.zeros_like(physical_state)
        evaluation = self.method.pseudospectral.dealiasing.evaluation
        points = evaluation.points.reshape(
            evaluation.physical_shape + (len(evaluation.axes),)
        )
        value = jnp.asarray(self.source(time, physical_state, points, args))
        if value.shape != physical_state.shape:
            raise ValueError("Spectral conservation source must match physical state.")
        return value

    def residual_parts(
        self,
        time: Array,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        coefficients = self._validate_state(state)
        dealiasing = self.method.pseudospectral.dealiasing
        physical = dealiasing.reconstruct(coefficients)
        convective = jnp.zeros_like(coefficients)
        for axis in range(len(self.discretization.axes)):
            flux = self.system.physical_flux(physical, axis, args)
            flux_coefficients = dealiasing.project(flux)
            convective = convective - self.discretization.modal_derivative(
                flux_coefficients,
                axis=axis,
                order=1,
            )
        source = dealiasing.project(self._physical_source(time, physical, args))
        return convective, source

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        convective, source = self.residual_parts(time, state, args)
        return convective + source

    def residual_with_diagnostics(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, SpectralConservationDiagnostics]:
        coefficients = self._validate_state(state)
        convective, source = self.residual_parts(time, coefficients, args)
        residual = convective + source
        physical_state = self.discretization.reconstruct(coefficients)
        physical_residual = self.discretization.reconstruct(residual)
        physical_source = self.discretization.reconstruct(source)
        weights = self.discretization.quadrature_weights[..., None]
        total_integral = jnp.sum(
            weights * physical_state, axis=tuple(range(len(self.discretization.axes)))
        )
        residual_integral = jnp.sum(
            weights * physical_residual, axis=tuple(range(len(self.discretization.axes)))
        )
        source_integral = jnp.sum(
            weights * physical_source, axis=tuple(range(len(self.discretization.axes)))
        )
        entropy = None
        if self.entropy_pair is not None:
            pair = self.entropy_pair
            entropy_variables = pair.entropy_variables(physical_state)
            convective_physical = self.discretization.reconstruct(convective)
            convective_density = oe.contract(
                "...i,...i->...",
                entropy_variables,
                convective_physical,
            )
            source_density = oe.contract(
                "...i,...i->...",
                entropy_variables,
                physical_source,
            )
            scalar_weights = self.discretization.quadrature_weights
            convective_rate = jnp.sum(scalar_weights * convective_density)
            source_rate = jnp.sum(scalar_weights * source_density)
            entropy = SpectralEntropyDiagnostics(
                pair_id=pair.pair_id,
                total_entropy=jnp.sum(scalar_weights * pair.entropy(physical_state)),
                semidiscrete_entropy_rate=convective_rate + source_rate,
                source_entropy_rate=source_rate,
                convective_entropy_rate=convective_rate,
                admissible=jnp.all(pair.admissible(physical_state)),
                precision_evidence=self.discretization.precision_evidence,
            )
        diagnostics = SpectralConservationDiagnostics(
            total_integral=total_integral,
            semidiscrete_integral_rate=residual_integral,
            source_integral=source_integral,
            conservation_defect=residual_integral - source_integral,
            entropy=entropy,
            precision_evidence=self.discretization.precision_evidence,
            method_id=self.method.plan.method_id,
        )
        return residual, diagnostics

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        value = self._validate_state(state)
        return jax.linearize(lambda candidate: self(time, candidate, args), value)


__all__ = [
    "PreparedSpectralConservationDynamics",
    "PreparedSpectralConservationMethod",
    "SpectralConservationDiagnostics",
    "SpectralConservationMethodPlan",
    "SpectralEntropyDiagnostics",
]

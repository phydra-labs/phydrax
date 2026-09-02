#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._riemann import (
    _normal_ale_inputs,
    AbstractArbitraryNormalNumericalFluxPlan,
    HLLFluxPlan,
    NumericalFluxResult,
)
from ...equations._hyperbolic_systems import (
    CompressibleNavierStokesSystem,
    EulerSystem,
)
from ._contracts import AllSpeedCompressiblePolicy, ShockResolvingPolicy
from ._system import (
    MaterialCompressibleNavierStokesSystem,
    MaterialEulerSystem,
)


def _hll_flux(
    left: Array,
    right: Array,
    left_flux: Array,
    right_flux: Array,
    lower: Array,
    upper: Array,
    /,
) -> NumericalFluxResult:
    lower_ = jnp.minimum(jnp.asarray(lower), 0.0)
    upper_ = jnp.maximum(jnp.asarray(upper), 0.0)
    denominator = upper_ - lower_
    zero_width = denominator == 0.0
    central = 0.5 * (left_flux + right_flux)
    middle = (
        upper_[..., None] * left_flux
        - lower_[..., None] * right_flux
        + (lower_ * upper_)[..., None] * (right - left)
    ) / jnp.where(zero_width, 1.0, denominator)[..., None]
    upwind = jnp.where(
        (lower_ >= 0.0)[..., None],
        left_flux,
        jnp.where((upper_ <= 0.0)[..., None], right_flux, middle),
    )
    flux = jnp.where(zero_width[..., None], central, upwind)
    return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower_), jnp.abs(upper_)))


class AllSpeedHLLFluxPlan(AbstractArbitraryNormalNumericalFluxPlan, NonTrainableState):
    """HLL flux whose acoustic dissipation follows one explicit all-speed policy."""

    policy: AllSpeedCompressiblePolicy
    flux_id: str = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)

    def __init__(self, policy: AllSpeedCompressiblePolicy, /):
        if not isinstance(policy, AllSpeedCompressiblePolicy):
            raise TypeError("policy must be AllSpeedCompressiblePolicy.")
        self.policy = policy
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint(
            {
                "kind": "all-speed-hll-flux",
                "policy": policy.policy_id,
                "signal_scaling": "center-plus-minus-scaled-half-width",
                "normal_ale_contract": (
                    "physical-normal-flux-minus-grid-transport-relative-waves-v1"
                ),
            }
        )

    def _scaled_bounds(self, lower: Array, upper: Array, /) -> tuple[Array, Array]:
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper)
        center = 0.5 * (lower_ + upper_)
        acoustic = 0.5 * (upper_ - lower_)
        acoustic = eqx.error_if(
            acoustic,
            jnp.any(~jnp.isfinite(center) | ~jnp.isfinite(acoustic) | (acoustic < 0.0)),
            "All-speed signal bounds must be finite and ordered.",
        )
        scaled_acoustic = self.policy.scaled_acoustic_speed(
            jnp.abs(center),
            jnp.maximum(acoustic, jnp.finfo(acoustic.dtype).tiny),
        )
        return center - scaled_acoustic, center + scaled_acoustic

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        lower, upper = system.signal_bounds(left_, right_, int(axis), args)
        scaled_lower, scaled_upper = self._scaled_bounds(lower, upper)
        return _hll_flux(
            left_,
            right_,
            system.physical_flux(left_, int(axis), args),
            system.physical_flux(right_, int(axis), args),
            scaled_lower,
            scaled_upper,
        )

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        normal_ = jnp.asarray(normal)
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        scaled_lower, scaled_upper = self._scaled_bounds(lower, upper)
        return _hll_flux(
            left_,
            right_,
            system.physical_normal_flux(left_, normal_, args),
            system.physical_normal_flux(right_, normal_, args),
            scaled_lower,
            scaled_upper,
        )

    def normal_ale_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        grid_normal_velocity: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_, right_, normal_, grid_velocity = _normal_ale_inputs(
            left, right, normal, grid_normal_velocity
        )
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        center = 0.5 * (jnp.asarray(lower) + jnp.asarray(upper))
        acoustic = 0.5 * (jnp.asarray(upper) - jnp.asarray(lower))
        relative_center = center - grid_velocity
        scaled_acoustic = self.policy.scaled_acoustic_speed(
            jnp.abs(relative_center),
            jnp.maximum(acoustic, jnp.finfo(acoustic.dtype).tiny),
        )
        scaled_lower = relative_center - scaled_acoustic
        scaled_upper = relative_center + scaled_acoustic
        return _hll_flux(
            left_,
            right_,
            system.physical_normal_flux(left_, normal_, args)
            - grid_velocity[..., None] * left_,
            system.physical_normal_flux(right_, normal_, args)
            - grid_velocity[..., None] * right_,
            scaled_lower,
            scaled_upper,
        )


class ShockAwareAllSpeedFluxPlan(
    AbstractArbitraryNormalNumericalFluxPlan, NonTrainableState
):
    """All-speed primary flux with explicit pressure-sensor Einfeldt dispatch."""

    policy: ShockResolvingPolicy
    primary: AllSpeedHLLFluxPlan
    generic_fallback: HLLFluxPlan
    flux_id: str = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)

    def __init__(self, policy: ShockResolvingPolicy, /):
        if not isinstance(policy, ShockResolvingPolicy):
            raise TypeError("policy must be ShockResolvingPolicy.")
        self.policy = policy
        self.primary = AllSpeedHLLFluxPlan(policy.all_speed)
        self.generic_fallback = HLLFluxPlan()
        self.differentiability = "branchwise"
        self.flux_id = canonical_fingerprint(
            {
                "kind": "shock-aware-all-speed-flux",
                "policy": policy.policy_id,
                "primary": self.primary.flux_id,
                "fallback": policy.fallback_flux.flux_id,
                "generic_fallback": self.generic_fallback.flux_id,
                "sensor": "relative-pressure-jump",
            }
        )

    def _fallback_plan(self, system: Any, /):
        if isinstance(system, (EulerSystem, CompressibleNavierStokesSystem)):
            return self.policy.fallback_flux
        if isinstance(
            system,
            (MaterialEulerSystem, MaterialCompressibleNavierStokesSystem),
        ):
            return self.generic_fallback
        raise TypeError(
            "Shock-aware all-speed flux requires a supported compressible system."
        )

    @staticmethod
    def _sensor(system: Any, left: Array, right: Array, /) -> Array:
        left_pressure = jnp.asarray(system.pressure(left))
        right_pressure = jnp.asarray(system.pressure(right))
        scale = jnp.maximum(
            jnp.abs(left_pressure) + jnp.abs(right_pressure),
            jnp.finfo(left_pressure.dtype).tiny,
        )
        return jnp.abs(right_pressure - left_pressure) / scale

    def _select(
        self,
        system: Any,
        left: Array,
        right: Array,
        primary: NumericalFluxResult,
        fallback: NumericalFluxResult,
        /,
    ) -> NumericalFluxResult:
        sensor = self._sensor(system, left, right)
        admissible = jnp.asarray(system.admissible(left), dtype=bool) & jnp.asarray(
            system.admissible(right), dtype=bool
        )
        primary_finite = jnp.all(
            jnp.isfinite(primary.normal_flux), axis=-1
        ) & jnp.isfinite(primary.max_speed)
        ledger = self.policy.ledger(sensor, admissible, primary_finite)
        selected_flux = self.policy.select_flux(
            primary.normal_flux, fallback.normal_flux, ledger
        )
        selected_speed = jnp.where(
            ledger.fallback_used, fallback.max_speed, primary.max_speed
        )
        return NumericalFluxResult(
            selected_flux,
            selected_speed,
            fallback_activated=ledger.fallback_used,
        )

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        primary = self.primary.face_flux(system, left_, right_, axis, args)
        fallback = self._fallback_plan(system).face_flux(
            system, left_, right_, axis, args
        )
        return self._select(system, left_, right_, primary, fallback)

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        primary = self.primary.normal_face_flux(system, left_, right_, normal, args)
        fallback = self._fallback_plan(system).normal_face_flux(
            system, left_, right_, normal, args
        )
        return self._select(system, left_, right_, primary, fallback)

    def normal_ale_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        grid_normal_velocity: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        primary = self.primary.normal_ale_face_flux(
            system,
            left_,
            right_,
            normal,
            grid_normal_velocity,
            args,
        )
        fallback = self._fallback_plan(system).normal_ale_face_flux(
            system,
            left_,
            right_,
            normal,
            grid_normal_velocity,
            args,
        )
        return self._select(system, left_, right_, primary, fallback)


__all__ = ["AllSpeedHLLFluxPlan", "ShockAwareAllSpeedFluxPlan"]

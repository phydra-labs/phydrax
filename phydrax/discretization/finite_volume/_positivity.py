#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._mapped import MappedFiniteVolumeDiscretization
from ._riemann import AbstractNumericalFluxPlan, NumericalFluxResult
from ._structured import FiniteVolumeDiscretization


class EinfeldtHLLFluxPlan(AbstractNumericalFluxPlan):
    """Monotone HLL fallback with Roe-enlarged Einfeldt signal bounds."""

    def __init__(self):
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint({"kind": "einfeldt-hll-flux"})

    def _bounds(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any,
        /,
    ) -> tuple[Array, Array]:
        lower, upper = system.signal_bounds(left, right, axis, args)
        left_matrix, right_matrix, roe_speeds = system.eigensystem(
            left, right, axis, args
        )
        del left_matrix, right_matrix
        lower = jnp.minimum(lower, jnp.min(roe_speeds, axis=-1))
        upper = jnp.maximum(upper, jnp.max(roe_speeds, axis=-1))
        return jnp.minimum(lower, 0.0), jnp.maximum(upper, 0.0)

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        lower, upper = self._bounds(system, left, right, int(axis), args)
        left_flux = system.physical_flux(left, int(axis), args)
        right_flux = system.physical_flux(right, int(axis), args)
        denominator = upper - lower
        flux = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right - left)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, flux),
        )
        return NumericalFluxResult(
            flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper))
        )


class FiniteVolumeAdmissibilityReport(StrictModule):
    high_order_valid: Array
    fallback_valid: Array
    blend_factor: Array
    activated: Array
    minimum_density: Array
    limited_state_valid: Array
    secondary_reduction_applied: Array
    secondary_reduction_factor: Array


class PositivityBlendResult(StrictModule):
    state: Array
    report: FiniteVolumeAdmissibilityReport
    normal_fluxes: tuple[Array, ...]
    integrated_fluxes: tuple[Array, ...]
    face_blend_factors: tuple[Array, ...]


class FluxPositivityPlan(StrictModule, NonTrainableState):
    """Conservative global blending against a monotone fallback stage."""

    fallback_flux: EinfeldtHLLFluxPlan
    iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, iterations: int = 32, /):
        iterations_ = int(iterations)
        if iterations_ <= 0:
            raise ValueError("Positivity blending iterations must be positive.")
        self.fallback_flux = EinfeldtHLLFluxPlan()
        self.iterations = iterations_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-volume-flux-positivity",
                "fallback": self.fallback_flux.flux_id,
                "iterations": iterations_,
            }
        )

    def limit_candidate(
        self,
        system: Any,
        high_order_state: Array,
        fallback_state: Array,
        /,
    ) -> PositivityBlendResult:
        high = jnp.asarray(high_order_state)
        fallback = jnp.asarray(fallback_state)
        if high.shape != fallback.shape:
            raise ValueError("High-order and fallback candidates must have equal shape.")
        high_valid = jnp.all(system.admissible(high))
        fallback_valid = jnp.all(system.admissible(fallback))
        direction = high - fallback

        def body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            valid = jnp.all(system.admissible(fallback + midpoint * direction))
            return jnp.where(valid, midpoint, lower), jnp.where(
                valid, upper, midpoint
            )

        lower, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            body,
            (
                jnp.asarray(0.0, dtype=high.dtype),
                jnp.asarray(1.0, dtype=high.dtype),
            ),
        )
        blend = jnp.where(high_valid, 1.0, lower)
        blend = jnp.where(fallback_valid, blend, 0.0)
        state = fallback + blend * direction
        return PositivityBlendResult(
            state=state,
            report=FiniteVolumeAdmissibilityReport(
                high_order_valid=high_valid,
                fallback_valid=fallback_valid,
                blend_factor=blend,
                activated=blend < 1.0,
                minimum_density=jnp.min(state[..., 0]),
                limited_state_valid=jnp.all(system.admissible(state)),
                secondary_reduction_applied=jnp.asarray(False),
                secondary_reduction_factor=jnp.asarray(
                    1.0, dtype=state.dtype
                ),
            ),
            normal_fluxes=(),
            integrated_fluxes=(),
            face_blend_factors=(),
        )
    def limit_face_fluxes(
        self,
        system: Any,
        base_state: Array,
        high_order_fluxes: tuple[Array, ...],
        fallback_fluxes: tuple[Array, ...],
        common_residual: Array,
        step_size: Array,
        discretization: FiniteVolumeDiscretization
        | MappedFiniteVolumeDiscretization,
        /,
    ) -> PositivityBlendResult:
        if len(high_order_fluxes) != len(fallback_fluxes):
            raise ValueError("High-order and fallback face fluxes must align.")

        def residual(fluxes):
            output = jnp.zeros_like(base_state)
            for axis, flux in enumerate(fluxes):
                integrated = flux * discretization.face_measures[axis][..., None]
                if discretization.grid.structured_axes[axis].periodic:
                    difference = jnp.roll(integrated, -1, axis=axis) - integrated
                else:
                    lower: list[slice | int] = [slice(None)] * integrated.ndim
                    upper: list[slice | int] = [slice(None)] * integrated.ndim
                    lower[axis] = slice(0, integrated.shape[axis] - 1)
                    upper[axis] = slice(1, integrated.shape[axis])
                    difference = integrated[tuple(upper)] - integrated[tuple(lower)]
                output = output - difference / discretization.cell_volumes[..., None]
            return output

        low_residual = residual(fallback_fluxes) + common_residual
        high_residual = residual(high_order_fluxes) + common_residual
        fallback_state = base_state + step_size * low_residual
        high_state = base_state + step_size * high_residual
        direction = high_state - fallback_state
        fallback_valid_cells = system.admissible(fallback_state)
        high_valid_cells = system.admissible(high_state)

        def body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate = fallback_state + midpoint[..., None] * direction
            valid = system.admissible(candidate)
            return jnp.where(valid, midpoint, lower), jnp.where(
                valid, upper, midpoint
            )

        cell_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            body,
            (
                jnp.zeros(base_state.shape[:-1], dtype=base_state.dtype),
                jnp.ones(base_state.shape[:-1], dtype=base_state.dtype),
            ),
        )
        cell_factor = jnp.where(high_valid_cells, 1.0, cell_factor)
        cell_factor = jnp.where(fallback_valid_cells, cell_factor, 0.0)
        face_factors = []
        limited_fluxes = []
        for axis, (high_flux, low_flux) in enumerate(
            zip(high_order_fluxes, fallback_fluxes, strict=True)
        ):
            if discretization.grid.structured_axes[axis].periodic:
                factor = jnp.minimum(
                    jnp.roll(cell_factor, 1, axis=axis), cell_factor
                )
            else:
                moved = jnp.moveaxis(cell_factor, axis, 0)
                interior = jnp.minimum(moved[:-1], moved[1:])
                factor = jnp.moveaxis(
                    jnp.concatenate((moved[:1], interior, moved[-1:])), 0, axis
                )
            face_factors.append(factor)
            limited_fluxes.append(
                low_flux + factor[..., None] * (high_flux - low_flux)
            )
        preliminary_fluxes = tuple(limited_fluxes)
        preliminary_state = base_state + step_size * (
            residual(preliminary_fluxes) + common_residual
        )
        preliminary_valid = jnp.all(system.admissible(preliminary_state))

        def secondary_body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate_fluxes = tuple(
                low_flux
                + midpoint * (preliminary_flux - low_flux)
                for preliminary_flux, low_flux in zip(
                    preliminary_fluxes, fallback_fluxes, strict=True
                )
            )
            candidate_state = base_state + step_size * (
                residual(candidate_fluxes) + common_residual
            )
            valid = jnp.all(system.admissible(candidate_state))
            return jnp.where(valid, midpoint, lower), jnp.where(
                valid, upper, midpoint
            )

        secondary_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            secondary_body,
            (
                jnp.asarray(0.0, dtype=base_state.dtype),
                jnp.asarray(1.0, dtype=base_state.dtype),
            ),
        )
        secondary_factor = jnp.where(
            preliminary_valid, 1.0, secondary_factor
        )
        secondary_factor = jnp.where(
            jnp.all(fallback_valid_cells), secondary_factor, 0.0
        )
        limited_fluxes_ = tuple(
            low_flux
            + secondary_factor * (preliminary_flux - low_flux)
            for preliminary_flux, low_flux in zip(
                preliminary_fluxes, fallback_fluxes, strict=True
            )
        )
        limited_state = base_state + step_size * (
            residual(limited_fluxes_) + common_residual
        )
        limited_valid = jnp.all(system.admissible(limited_state))
        final_face_factors = tuple(
            secondary_factor * factor for factor in face_factors
        )
        integrated_fluxes = tuple(
            flux * measure[..., None]
            for flux, measure in zip(
                limited_fluxes_, discretization.face_measures, strict=True
            )
        )
        return PositivityBlendResult(
            state=limited_state,
            report=FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.all(high_valid_cells),
                fallback_valid=jnp.all(fallback_valid_cells),
                blend_factor=jnp.minimum(
                    jnp.min(cell_factor), secondary_factor
                ),
                activated=jnp.any(cell_factor < 1.0)
                | (secondary_factor < 1.0),
                minimum_density=jnp.min(limited_state[..., 0]),
                limited_state_valid=limited_valid,
                secondary_reduction_applied=~preliminary_valid,
                secondary_reduction_factor=secondary_factor,
            ),
            normal_fluxes=limited_fluxes_,
            integrated_fluxes=integrated_fluxes,
            face_blend_factors=final_face_factors,
        )


__all__ = [
    "EinfeldtHLLFluxPlan",
    "FiniteVolumeAdmissibilityReport",
    "FluxPositivityPlan",
    "PositivityBlendResult",
]

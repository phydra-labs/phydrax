#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._flux_ledger import FiniteVolumeStageFluxRateLedger
from ._mapped import MappedFiniteVolumeDiscretization
from ._riemann import (
    _NORMAL_ALE_CONTRACT,
    _normal_ale_inputs,
    AbstractNumericalFluxPlan,
    NumericalFluxResult,
)
from ._shallow_water import ShallowWaterBalancedFaceResult
from ._structured import FiniteVolumeDiscretization
from ._triangle_fv import TriangleFiniteVolumeDiscretization
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class EinfeldtHLLFluxPlan(AbstractNumericalFluxPlan):
    """Monotone HLL fallback with Roe-enlarged Einfeldt signal bounds."""

    def __init__(self):
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint(
            {"kind": "einfeldt-hll-flux", "normal_ale_contract": _NORMAL_ALE_CONTRACT}
        )

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

    def _normal_bounds(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any,
        /,
    ) -> tuple[Array, Array]:
        material = getattr(system, "material", None)
        if (
            system.component_count != system.dimension + 2
            or material is None
            or not hasattr(material, "gamma")
        ):
            raise TypeError(
                "Einfeldt normal flux requires an Euler-compatible state layout."
            )
        left_primitive = system.conserved_to_primitive(left)
        right_primitive = system.conserved_to_primitive(right)
        left_root = jnp.sqrt(left_primitive[..., 0])
        right_root = jnp.sqrt(right_primitive[..., 0])
        denominator = left_root + right_root
        roe_velocity = (
            left_root[..., None] * left_primitive[..., 1:-1]
            + right_root[..., None] * right_primitive[..., 1:-1]
        ) / denominator[..., None]
        left_enthalpy = (left[..., -1] + left_primitive[..., -1]) / left_primitive[..., 0]
        right_enthalpy = (right[..., -1] + right_primitive[..., -1]) / right_primitive[
            ..., 0
        ]
        roe_enthalpy = (
            left_root * left_enthalpy + right_root * right_enthalpy
        ) / denominator
        roe_speed_squared = jnp.sum(roe_velocity**2, axis=-1)
        roe_sound = jnp.sqrt(
            jnp.maximum(
                (material.gamma - 1.0) * (roe_enthalpy - 0.5 * roe_speed_squared),
                jnp.finfo(left.dtype).tiny,
            )
        )
        roe_normal_velocity = jnp.sum(roe_velocity * normal, axis=-1)
        lower, upper = system.normal_signal_bounds(left, right, normal, args)
        return (
            jnp.minimum(jnp.asarray(lower), roe_normal_velocity - roe_sound),
            jnp.maximum(jnp.asarray(upper), roe_normal_velocity + roe_sound),
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
        return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper)))

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
        lower, upper = self._normal_bounds(system, left_, right_, normal_, args)
        lower = jnp.minimum(lower, 0.0)
        upper = jnp.maximum(upper, 0.0)
        left_flux = system.physical_normal_flux(left_, normal_, args)
        right_flux = system.physical_normal_flux(right_, normal_, args)
        denominator = upper - lower
        flux = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right_ - left_)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, flux),
        )
        return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper)))

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
        lower, upper = self._normal_bounds(system, left_, right_, normal_, args)
        lower = jnp.minimum(lower - grid_velocity, 0.0)
        upper = jnp.maximum(upper - grid_velocity, 0.0)
        left_flux = (
            system.physical_normal_flux(left_, normal_, args)
            - grid_velocity[..., None] * left_
        )
        right_flux = (
            system.physical_normal_flux(right_, normal_, args)
            - grid_velocity[..., None] * right_
        )
        denominator = upper - lower
        middle = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right_ - left_)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, middle),
        )
        return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper)))


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


class BalancedPositivityBlendResult(StrictModule):
    """Positivity-limited shallow-water face contributions."""

    state: Array
    report: FiniteVolumeAdmissibilityReport
    contributions: tuple[ShallowWaterBalancedFaceResult, ...]
    normal_fluxes: tuple[Array, ...]
    integrated_fluxes: tuple[Array, ...]
    face_blend_factors: tuple[Array, ...]


class StageRatePositivityResult(StrictModule):
    """A positivity-limited stage rate and its target-volume Euler content."""

    euler_content: Array
    euler_cell_average: Array
    ledger: FiniteVolumeStageFluxRateLedger
    report: FiniteVolumeAdmissibilityReport
    face_blend_factors: tuple[Array, ...]


class FluxPositivityPlan(StrictModule, NonTrainableState):
    """Conservative global blending against a monotone fallback stage."""

    fallback_flux: AbstractNumericalFluxPlan
    iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        iterations: int = 32,
        /,
        *,
        fallback_flux: AbstractNumericalFluxPlan | None = None,
    ):
        iterations_ = int(iterations)
        if iterations_ <= 0:
            raise ValueError("Positivity blending iterations must be positive.")
        fallback = EinfeldtHLLFluxPlan() if fallback_flux is None else fallback_flux
        if not isinstance(fallback, AbstractNumericalFluxPlan):
            raise TypeError("fallback_flux must be a numerical flux plan.")
        self.fallback_flux = fallback
        self.iterations = iterations_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-volume-flux-positivity",
                "fallback": fallback.flux_id,
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
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

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
                secondary_reduction_factor=jnp.asarray(1.0, dtype=state.dtype),
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
        discretization: (
            FiniteVolumeDiscretization
            | MappedFiniteVolumeDiscretization
            | TriangleFiniteVolumeDiscretization
            | UnstructuredFiniteVolumeDiscretization
        ),
        /,
    ) -> PositivityBlendResult:
        if len(high_order_fluxes) != len(fallback_fluxes):
            raise ValueError("High-order and fallback face fluxes must align.")

        def residual(fluxes):
            output = jnp.zeros_like(base_state)
            if isinstance(
                discretization,
                (
                    TriangleFiniteVolumeDiscretization,
                    UnstructuredFiniteVolumeDiscretization,
                ),
            ):
                if len(fluxes) != 1:
                    raise ValueError("Triangle positivity requires one face-flux block.")
                face_measures = discretization.face_measures.astype(base_state.dtype)
                cell_volumes = discretization.cell_volumes.astype(base_state.dtype)
                integrated = fluxes[0] * face_measures[:, None]
                output = output.at[discretization.owner_cells].add(-integrated)
                neighbour = discretization.neighbour_cells
                output = output.at[jnp.maximum(neighbour, 0)].add(
                    jnp.where((neighbour >= 0)[:, None], integrated, 0.0)
                )
                return output / cell_volumes[:, None]
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
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

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
        if isinstance(
            discretization,
            (
                TriangleFiniteVolumeDiscretization,
                UnstructuredFiniteVolumeDiscretization,
            ),
        ):
            owner_factor = cell_factor[discretization.owner_cells]
            neighbour = discretization.neighbour_cells
            neighbour_factor = cell_factor[jnp.maximum(neighbour, 0)]
            factor = jnp.where(
                neighbour >= 0,
                jnp.minimum(owner_factor, neighbour_factor),
                owner_factor,
            )
            face_factors.append(factor)
            limited_fluxes.append(
                fallback_fluxes[0]
                + factor[..., None] * (high_order_fluxes[0] - fallback_fluxes[0])
            )
        else:
            for axis, (high_flux, low_flux) in enumerate(
                zip(high_order_fluxes, fallback_fluxes, strict=True)
            ):
                if discretization.grid.structured_axes[axis].periodic:
                    factor = jnp.minimum(
                        jnp.roll(cell_factor, 1, axis=axis),
                        cell_factor,
                    )
                else:
                    moved = jnp.moveaxis(cell_factor, axis, 0)
                    interior = jnp.minimum(moved[:-1], moved[1:])
                    factor = jnp.moveaxis(
                        jnp.concatenate((moved[:1], interior, moved[-1:])),
                        0,
                        axis,
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
                low_flux + midpoint * (preliminary_flux - low_flux)
                for preliminary_flux, low_flux in zip(
                    preliminary_fluxes, fallback_fluxes, strict=True
                )
            )
            candidate_state = base_state + step_size * (
                residual(candidate_fluxes) + common_residual
            )
            valid = jnp.all(system.admissible(candidate_state))
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        secondary_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            secondary_body,
            (
                jnp.asarray(0.0, dtype=base_state.dtype),
                jnp.asarray(1.0, dtype=base_state.dtype),
            ),
        )
        secondary_factor = jnp.where(preliminary_valid, 1.0, secondary_factor)
        secondary_factor = jnp.where(jnp.all(fallback_valid_cells), secondary_factor, 0.0)
        limited_fluxes_ = tuple(
            low_flux + secondary_factor * (preliminary_flux - low_flux)
            for preliminary_flux, low_flux in zip(
                preliminary_fluxes, fallback_fluxes, strict=True
            )
        )
        limited_state = base_state + step_size * (
            residual(limited_fluxes_) + common_residual
        )
        limited_valid = jnp.all(system.admissible(limited_state))
        final_face_factors = tuple(secondary_factor * factor for factor in face_factors)
        if isinstance(
            discretization,
            (
                TriangleFiniteVolumeDiscretization,
                UnstructuredFiniteVolumeDiscretization,
            ),
        ):
            integrated_fluxes = (
                limited_fluxes_[0]
                * discretization.face_measures.astype(base_state.dtype)[:, None],
            )
        else:
            integrated_fluxes = tuple(
                flux * measure[..., None]
                for flux, measure in zip(
                    limited_fluxes_,
                    discretization.face_measures,
                    strict=True,
                )
            )
        return PositivityBlendResult(
            state=limited_state,
            report=FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.all(high_valid_cells),
                fallback_valid=jnp.all(fallback_valid_cells),
                blend_factor=jnp.minimum(jnp.min(cell_factor), secondary_factor),
                activated=jnp.any(cell_factor < 1.0) | (secondary_factor < 1.0),
                minimum_density=jnp.min(limited_state[..., 0]),
                limited_state_valid=limited_valid,
                secondary_reduction_applied=~preliminary_valid,
                secondary_reduction_factor=secondary_factor,
            ),
            normal_fluxes=limited_fluxes_,
            integrated_fluxes=integrated_fluxes,
            face_blend_factors=final_face_factors,
        )

    def limit_balanced_face_contributions(
        self,
        system: Any,
        base_state: Array,
        high_order: tuple[ShallowWaterBalancedFaceResult, ...],
        fallback: tuple[ShallowWaterBalancedFaceResult, ...],
        common_residual: Array,
        step_size: Array,
        discretization: FiniteVolumeDiscretization,
        /,
    ) -> BalancedPositivityBlendResult:
        """Blend transport and both bed corrections with one face factor."""
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError(
                "Balanced shallow-water positivity requires Cartesian finite volumes."
            )
        if len(high_order) != len(fallback) or len(high_order) != len(
            discretization.cell_shape
        ):
            raise ValueError(
                "Balanced high-order and fallback face blocks must align by axis."
            )
        base = jnp.asarray(base_state)
        common = jnp.asarray(common_residual, dtype=base.dtype)
        dt = jnp.asarray(step_size, dtype=base.dtype).reshape(())

        def residual(
            contributions: tuple[ShallowWaterBalancedFaceResult, ...],
        ) -> Array:
            output = jnp.zeros_like(base)
            for axis, contribution in enumerate(contributions):
                measure = discretization.face_measures[axis].astype(base.dtype)[..., None]
                left_integrated = contribution.left_flux.astype(base.dtype) * measure
                right_integrated = contribution.right_flux.astype(base.dtype) * measure
                if discretization.grid.structured_axes[axis].periodic:
                    difference = (
                        jnp.roll(left_integrated, -1, axis=axis) - right_integrated
                    )
                else:
                    lower = [slice(None)] * right_integrated.ndim
                    upper = [slice(None)] * left_integrated.ndim
                    lower[axis] = slice(0, right_integrated.shape[axis] - 1)
                    upper[axis] = slice(1, left_integrated.shape[axis])
                    difference = (
                        left_integrated[tuple(upper)] - right_integrated[tuple(lower)]
                    )
                output = output - difference / discretization.cell_volumes[
                    ..., None
                ].astype(base.dtype)
            return output

        fallback_state = base + dt * (residual(fallback) + common)
        high_state = base + dt * (residual(high_order) + common)
        fallback_valid_cells = system.admissible(fallback_state)
        high_valid_cells = system.admissible(high_state)
        direction = high_state - fallback_state

        def local_body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate = fallback_state + midpoint[..., None] * direction
            valid = system.admissible(candidate)
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        cell_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            local_body,
            (
                jnp.zeros(base.shape[:-1], dtype=base.dtype),
                jnp.ones(base.shape[:-1], dtype=base.dtype),
            ),
        )
        cell_factor = jnp.where(high_valid_cells, 1.0, cell_factor)
        cell_factor = jnp.where(fallback_valid_cells, cell_factor, 0.0)

        face_factors = []
        preliminary = []
        for axis, (high, low) in enumerate(zip(high_order, fallback, strict=True)):
            if discretization.grid.structured_axes[axis].periodic:
                factor = jnp.minimum(jnp.roll(cell_factor, 1, axis=axis), cell_factor)
            else:
                moved = jnp.moveaxis(cell_factor, axis, 0)
                interior = jnp.minimum(moved[:-1], moved[1:])
                factor = jnp.moveaxis(
                    jnp.concatenate((moved[:1], interior, moved[-1:])),
                    0,
                    axis,
                )
            face_factors.append(factor)
            preliminary.append(
                ShallowWaterBalancedFaceResult(
                    low.normal_flux
                    + factor[..., None] * (high.normal_flux - low.normal_flux),
                    low.left_correction
                    + factor[..., None] * (high.left_correction - low.left_correction),
                    low.right_correction
                    + factor[..., None] * (high.right_correction - low.right_correction),
                    jnp.maximum(low.max_speed, high.max_speed),
                    high.reconstructed_left,
                    high.reconstructed_right,
                    low.dry_face & high.dry_face,
                )
            )
        preliminary_tuple = tuple(preliminary)
        preliminary_state = base + dt * (residual(preliminary_tuple) + common)
        preliminary_valid = jnp.all(system.admissible(preliminary_state))

        def global_body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate_contributions = tuple(
                ShallowWaterBalancedFaceResult(
                    low.normal_flux
                    + midpoint * (candidate.normal_flux - low.normal_flux),
                    low.left_correction
                    + midpoint * (candidate.left_correction - low.left_correction),
                    low.right_correction
                    + midpoint * (candidate.right_correction - low.right_correction),
                    candidate.max_speed,
                    candidate.reconstructed_left,
                    candidate.reconstructed_right,
                    candidate.dry_face,
                )
                for candidate, low in zip(preliminary_tuple, fallback, strict=True)
            )
            candidate_state = base + dt * (residual(candidate_contributions) + common)
            valid = jnp.all(system.admissible(candidate_state))
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        secondary_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            global_body,
            (
                jnp.asarray(0.0, dtype=base.dtype),
                jnp.asarray(1.0, dtype=base.dtype),
            ),
        )
        fallback_valid = jnp.all(fallback_valid_cells)
        secondary_factor = jnp.where(preliminary_valid, 1.0, secondary_factor)
        secondary_factor = jnp.where(fallback_valid, secondary_factor, 0.0)
        limited = tuple(
            ShallowWaterBalancedFaceResult(
                low.normal_flux
                + secondary_factor * (candidate.normal_flux - low.normal_flux),
                low.left_correction
                + secondary_factor * (candidate.left_correction - low.left_correction),
                low.right_correction
                + secondary_factor * (candidate.right_correction - low.right_correction),
                candidate.max_speed,
                candidate.reconstructed_left,
                candidate.reconstructed_right,
                candidate.dry_face,
            )
            for candidate, low in zip(preliminary_tuple, fallback, strict=True)
        )
        limited_state = base + dt * (residual(limited) + common)
        limited_valid = jnp.all(system.admissible(limited_state))
        normal_fluxes = tuple(contribution.normal_flux for contribution in limited)
        integrated_fluxes = tuple(
            contribution.normal_flux * measure[..., None]
            for contribution, measure in zip(
                limited,
                discretization.face_measures,
                strict=True,
            )
        )
        return BalancedPositivityBlendResult(
            state=limited_state,
            report=FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.all(high_valid_cells),
                fallback_valid=fallback_valid,
                blend_factor=jnp.minimum(jnp.min(cell_factor), secondary_factor),
                activated=jnp.any(cell_factor < 1.0) | (secondary_factor < 1.0),
                minimum_density=jnp.min(limited_state[..., 0]),
                limited_state_valid=limited_valid,
                secondary_reduction_applied=~preliminary_valid,
                secondary_reduction_factor=secondary_factor,
            ),
            contributions=limited,
            normal_fluxes=normal_fluxes,
            integrated_fluxes=integrated_fluxes,
            face_blend_factors=tuple(
                secondary_factor * factor for factor in face_factors
            ),
        )

    def limit_stage_rate_ledgers(
        self,
        system: Any,
        base_content: ArrayLike,
        high_order_ledger: FiniteVolumeStageFluxRateLedger,
        fallback_ledger: FiniteVolumeStageFluxRateLedger,
        local_euler_increment: ArrayLike,
        target_cell_volumes: ArrayLike,
        /,
    ) -> StageRatePositivityResult:
        """Blend aligned content rates and test the Euler target as cell averages."""

        if not isinstance(high_order_ledger, FiniteVolumeStageFluxRateLedger):
            raise TypeError("high_order_ledger must be FiniteVolumeStageFluxRateLedger.")
        if not isinstance(fallback_ledger, FiniteVolumeStageFluxRateLedger):
            raise TypeError("fallback_ledger must be FiniteVolumeStageFluxRateLedger.")
        high = high_order_ledger
        fallback = fallback_ledger
        if (
            high.cell_count != fallback.cell_count
            or high.component_shape != fallback.component_shape
            or high.geometry_family_id != fallback.geometry_family_id
            or high.geometry_layout_id != fallback.geometry_layout_id
            or high.evidence_policy_id != fallback.evidence_policy_id
            or high.topology_epoch_id != fallback.topology_epoch_id
        ):
            raise ValueError(
                "High-order and fallback stage ledgers must share cell shape, "
                "geometry family/layout, evidence policy, and topology epoch."
            )
        if len(high.blocks) != len(fallback.blocks):
            raise ValueError("High-order and fallback stage block layouts must align.")
        if any(
            (
                high_block.block_id,
                high_block.block_kind,
                high_block.route_id,
                high_block.rate_block_id,
            )
            != (
                fallback_block.block_id,
                fallback_block.block_kind,
                fallback_block.route_id,
                fallback_block.rate_block_id,
            )
            for high_block, fallback_block in zip(
                high.blocks,
                fallback.blocks,
                strict=True,
            )
        ):
            raise ValueError(
                "High-order and fallback stage blocks must have identical routes "
                "and rate policies."
            )

        content = jnp.asarray(base_content)
        expected_shape = (high.cell_count, *high.component_shape)
        if content.shape != expected_shape:
            raise ValueError(
                "base_content must match the stage ledger cell/component shape."
            )
        if not jnp.issubdtype(content.dtype, jnp.inexact):
            raise TypeError("base_content must have a real inexact dtype.")
        target_volumes = jnp.asarray(target_cell_volumes, dtype=content.dtype)
        if target_volumes.shape != (high.cell_count,):
            raise ValueError("target_cell_volumes must have exact shape (cell_count,).")
        increment = jnp.asarray(local_euler_increment, dtype=content.dtype)
        if increment.shape != ():
            raise ValueError("local_euler_increment must be scalar.")
        content = eqx.error_if(
            content,
            ~jnp.isfinite(increment)
            | (increment <= 0.0)
            | jnp.any(~jnp.isfinite(content))
            | jnp.any(~jnp.isfinite(target_volumes))
            | jnp.any(high.active_cell_mask & (target_volumes <= 0.0))
            | jnp.any((~high.active_cell_mask) & (target_volumes != 0.0)),
            "Euler content/increment and supplied target volumes must be finite, "
            "with a positive increment, positive active volumes, and zero inactive "
            "volumes.",
        )
        content = eqx.error_if(
            content,
            (high.geometry_version != fallback.geometry_version)
            | (high.evidence_version != fallback.evidence_version)
            | jnp.any(high.active_cell_mask != fallback.active_cell_mask)
            | jnp.any(high.source_rate != fallback.source_rate),
            "High-order and fallback ledgers must have identical dynamic geometry, "
            "evidence, active masks, and common source rate.",
        )
        content = eqx.error_if(
            content,
            jnp.any((~high.active_cell_mask)[:, None] & (content != 0.0)),
            "Inactive base content must be exactly zero.",
        )
        for high_block, fallback_block in zip(
            high.blocks,
            fallback.blocks,
            strict=True,
        ):
            content = eqx.error_if(
                content,
                jnp.any(high_block.active_mask != fallback_block.active_mask),
                "High-order and fallback face activity masks must be identical.",
            )

        active = high.active_cell_mask
        active_components = active[:, None]
        safe_volumes = jnp.where(
            active,
            target_volumes,
            jnp.ones_like(target_volumes),
        )

        def cell_average(candidate_content: Array, /) -> Array:
            average = candidate_content / safe_volumes[:, None]
            return jnp.where(active_components, average, jnp.zeros_like(average))

        any_active = jnp.any(active)
        first_active = jnp.argmax(active.astype(jnp.int32))

        def admissible_active_cells(candidate_average: Array, /) -> Array:
            def evaluate(_):
                seed = candidate_average[first_active]
                safe_average = jnp.where(
                    active_components,
                    candidate_average,
                    seed[None, :],
                )
                return jnp.where(
                    active,
                    system.admissible(safe_average),
                    True,
                )

            return jax.lax.cond(
                any_active,
                evaluate,
                lambda _: jnp.ones((high.cell_count,), dtype=bool),
                operand=None,
            )

        fallback_content = content + increment * fallback.scatter_content_rate()
        high_content = content + increment * high.scatter_content_rate()
        fallback_average = cell_average(fallback_content)
        high_average = cell_average(high_content)
        fallback_valid_cells = admissible_active_cells(fallback_average)
        high_valid_cells = admissible_active_cells(high_average)
        direction = high_content - fallback_content

        def local_body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate_content = fallback_content + midpoint[:, None] * direction
            valid = admissible_active_cells(cell_average(candidate_content))
            return jnp.where(valid, midpoint, lower), jnp.where(
                valid,
                upper,
                midpoint,
            )

        cell_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            local_body,
            (
                jnp.zeros((high.cell_count,), dtype=content.dtype),
                jnp.ones((high.cell_count,), dtype=content.dtype),
            ),
        )
        cell_factor = jnp.where(high_valid_cells, 1.0, cell_factor)
        cell_factor = jnp.where(fallback_valid_cells, cell_factor, 0.0)
        cell_factor = jnp.where(active, cell_factor, 1.0)

        preliminary_blocks = []
        face_factors = []
        for high_block, fallback_block in zip(
            high.blocks,
            fallback.blocks,
            strict=True,
        ):
            owner_factor = cell_factor[high_block.owner_cells]
            neighbour = high_block.neighbour_cells
            neighbour_factor = cell_factor[jnp.maximum(neighbour, 0)]
            factor = jnp.where(
                neighbour >= 0,
                jnp.minimum(owner_factor, neighbour_factor),
                owner_factor,
            )
            factor = jnp.where(high_block.active_mask, factor, 0.0)
            face_factors.append(factor)
            preliminary_blocks.append(
                fallback_block.with_flux_rate(
                    fallback_block.flux_rate
                    + factor[:, None] * (high_block.flux_rate - fallback_block.flux_rate)
                )
            )

        def make_ledger(blocks) -> FiniteVolumeStageFluxRateLedger:
            return FiniteVolumeStageFluxRateLedger(
                tuple(blocks),
                high.source_rate,
                high.active_cell_mask,
                geometry_layout_id=high.geometry_layout_id,
                geometry_family_id=high.geometry_family_id,
                geometry_version=high.geometry_version,
                evidence_policy_id=high.evidence_policy_id,
                evidence_version=high.evidence_version,
                topology_epoch_id=high.topology_epoch_id,
            )

        preliminary_ledger = make_ledger(preliminary_blocks)
        preliminary_content = (
            content + increment * preliminary_ledger.scatter_content_rate()
        )
        preliminary_valid = jnp.all(
            admissible_active_cells(cell_average(preliminary_content))
        )

        def secondary_body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate_blocks = tuple(
                fallback_block.with_flux_rate(
                    fallback_block.flux_rate
                    + midpoint * (preliminary_block.flux_rate - fallback_block.flux_rate)
                )
                for preliminary_block, fallback_block in zip(
                    preliminary_blocks,
                    fallback.blocks,
                    strict=True,
                )
            )
            candidate_ledger = make_ledger(candidate_blocks)
            candidate_content = (
                content + increment * candidate_ledger.scatter_content_rate()
            )
            valid = jnp.all(admissible_active_cells(cell_average(candidate_content)))
            return jnp.where(valid, midpoint, lower), jnp.where(
                valid,
                upper,
                midpoint,
            )

        secondary_factor, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            secondary_body,
            (
                jnp.asarray(0.0, dtype=content.dtype),
                jnp.asarray(1.0, dtype=content.dtype),
            ),
        )
        fallback_valid = jnp.all(fallback_valid_cells)
        secondary_factor = jnp.where(
            preliminary_valid,
            1.0,
            secondary_factor,
        )
        secondary_factor = jnp.where(fallback_valid, secondary_factor, 0.0)
        limited_blocks = tuple(
            fallback_block.with_flux_rate(
                fallback_block.flux_rate
                + secondary_factor
                * (preliminary_block.flux_rate - fallback_block.flux_rate)
            )
            for preliminary_block, fallback_block in zip(
                preliminary_blocks,
                fallback.blocks,
                strict=True,
            )
        )
        limited_ledger = make_ledger(limited_blocks)
        limited_content = content + increment * limited_ledger.scatter_content_rate()
        limited_average = cell_average(limited_content)
        limited_valid = jnp.all(admissible_active_cells(limited_average))
        final_face_factors = tuple(secondary_factor * factor for factor in face_factors)
        active_density = jnp.where(
            active,
            limited_average[..., 0],
            jnp.asarray(jnp.inf, dtype=limited_average.dtype),
        )
        return StageRatePositivityResult(
            euler_content=limited_content,
            euler_cell_average=limited_average,
            ledger=limited_ledger,
            report=FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.all(high_valid_cells),
                fallback_valid=fallback_valid,
                blend_factor=jnp.minimum(
                    jnp.min(jnp.where(active, cell_factor, 1.0)),
                    secondary_factor,
                ),
                activated=jnp.any(jnp.where(active, cell_factor < 1.0, False))
                | (secondary_factor < 1.0),
                minimum_density=jnp.min(active_density),
                limited_state_valid=limited_valid,
                secondary_reduction_applied=~preliminary_valid,
                secondary_reduction_factor=secondary_factor,
            ),
            face_blend_factors=final_face_factors,
        )


__all__ = [
    "BalancedPositivityBlendResult",
    "EinfeldtHLLFluxPlan",
    "FiniteVolumeAdmissibilityReport",
    "FluxPositivityPlan",
    "PositivityBlendResult",
    "StageRatePositivityResult",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg._dense_inverse import dense_inverse
from ._halo import PreparedFiniteVolumeHaloPlan
from ._mapped import MappedFiniteVolumeDiscretization
from ._physical_boundaries import PrescribedHeatFluxWallBoundary
from ._structured import FiniteVolumeDiscretization


def _cell_to_faces(values: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    if periodic:
        faces = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _ghosted_center_gradient(
    values: Array,
    coordinates: Array,
    axis: int,
    depth: int,
    interior_count: int,
    /,
) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    denominator = (
        coordinates[depth + 1 : depth + interior_count + 1]
        - coordinates[depth - 1 : depth + interior_count - 1]
    )
    shape = (interior_count,) + (1,) * (moved.ndim - 1)
    gradient = (
        moved[depth + 1 : depth + interior_count + 1]
        - moved[depth - 1 : depth + interior_count - 1]
    ) / denominator.reshape(shape)
    return jnp.moveaxis(gradient, 0, axis)


def _structured_conserved_gradient(
    system: Any,
    time: Array,
    state: Array,
    discretization: FiniteVolumeDiscretization,
    halo: PreparedFiniteVolumeHaloPlan,
    args: Any,
    /,
) -> Array:
    derivatives = []
    for axis in range(system.dimension):
        ghosted = halo.materialize_axis(system, time, state, axis, args)
        derivatives.append(
            _ghosted_center_gradient(
                ghosted.values,
                ghosted.axis_coordinates,
                axis,
                ghosted.depth,
                discretization.cell_shape[axis],
            )
        )
    return jnp.stack(tuple(derivatives), axis=-1)


def _mapped_conserved_gradient(
    system: Any,
    time: Array,
    state: Array,
    discretization: MappedFiniteVolumeDiscretization,
    halo: PreparedFiniteVolumeHaloPlan,
    args: Any,
    /,
) -> Array:
    dimension = system.dimension
    matrix = jnp.zeros(
        discretization.cell_shape + (dimension, dimension), dtype=state.dtype
    )
    right_hand_side = jnp.zeros(
        discretization.cell_shape + (system.component_count, dimension),
        dtype=state.dtype,
    )
    for axis in range(dimension):
        ghosted = halo.materialize_axis(system, time, state, axis, args)
        moved_values = jnp.moveaxis(ghosted.values, axis, 0)
        moved_centers = jnp.moveaxis(ghosted.physical_centers, axis, 0)
        depth = ghosted.depth
        count = discretization.cell_shape[axis]
        current_values = moved_values[depth : depth + count]
        current_centers = moved_centers[depth : depth + count]
        for offset in (-1, 1):
            neighbor_values = moved_values[depth + offset : depth + count + offset]
            neighbor_centers = moved_centers[depth + offset : depth + count + offset]
            displacement = jnp.moveaxis(neighbor_centers - current_centers, 0, axis)
            difference = jnp.moveaxis(neighbor_values - current_values, 0, axis)
            matrix = matrix + displacement[..., :, None] * displacement[..., None, :]
            right_hand_side = (
                right_hand_side + difference[..., :, None] * displacement[..., None, :]
            )
    regularization = jnp.finfo(state.dtype).eps * jnp.eye(dimension, dtype=state.dtype)
    inverse = dense_inverse(matrix + regularization, positive_definite=True)
    gradient = ein.contract(
        "...ij,...jc->...ic",
        inverse,
        jnp.swapaxes(right_hand_side, -1, -2),
    )
    return jnp.swapaxes(gradient, -1, -2)


class FiniteVolumeDiffusionEvaluation(StrictModule):
    """Shared face fluxes and gradient-coupled cell source."""

    face_fluxes: tuple[Array, ...]
    cell_source: Array
    source_step: Array
    finite: Array
    successful: Array


class ViscousStabilityReport(StrictModule):
    maximum_diffusive_rate: Array
    selected_step: Array
    limiting_cell_flat_index: Array
    minimum_source_step: Array


class ViscousFluxPlan(StrictModule, NonTrainableState):
    """Conservative projection of an equation-owned diffusive tensor."""

    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.plan_id = canonical_fingerprint({"kind": "equation-owned-viscous-flux"})

    @staticmethod
    def _check_system(system: Any, /) -> None:
        from ...equations._hyperbolic_systems import AbstractEntropyDiffusionSystem

        if not isinstance(system, AbstractEntropyDiffusionSystem):
            raise TypeError("ViscousFluxPlan requires an AbstractEntropyDiffusionSystem.")

    def conserved_gradient(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any = None,
        /,
    ) -> Array:
        self._check_system(system)
        value = jnp.asarray(state)
        if value.shape != discretization.cell_shape + (system.component_count,):
            raise ValueError("Viscous state does not match the prepared FV shape.")
        if isinstance(discretization, MappedFiniteVolumeDiscretization):
            periodic_axes = tuple(
                axis
                for axis, structured_axis in enumerate(
                    discretization.grid.structured_axes
                )
                if structured_axis.periodic
            )
            seam_axes = frozenset(seam.axis for seam in discretization.periodic_seams)
            missing = tuple(axis for axis in periodic_axes if axis not in seam_axes)
            if missing:
                raise ValueError(
                    "Mapped periodic viscous flux requires prepared isometry evidence "
                    f"for axes {missing!r}."
                )
            return _mapped_conserved_gradient(
                system, time, value, discretization, halo, args
            )
        return _structured_conserved_gradient(
            system, time, value, discretization, halo, args
        )

    def _apply_prescribed_heat_flux(
        self,
        system: Any,
        time: Array,
        state: Array,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        fluxes: tuple[Array, ...],
        args: Any,
        /,
    ) -> tuple[Array, ...]:
        output = list(fluxes)
        for axis, pair in enumerate(halo.plan.boundaries.pairs):
            if pair is None:
                continue
            for side, boundary in (("lower", pair.lower), ("upper", pair.upper)):
                if not isinstance(boundary, PrescribedHeatFluxWallBoundary):
                    continue
                face_index = 0 if side == "lower" else output[axis].shape[axis] - 1
                cell_index = 0 if side == "lower" else state.shape[axis] - 1
                interior = jnp.take(state, cell_index, axis=axis)
                coordinates = jnp.take(
                    discretization.face_centers[axis], face_index, axis=axis
                )
                normal = discretization.outward_normal(axis, side)
                outward_heat = boundary.normal_heat_flux(
                    time, interior, coordinates, normal, args
                )
                face_flux = jnp.take(output[axis], face_index, axis=axis)
                traction = face_flux[..., system.momentum_slice]
                mechanical = jnp.sum(boundary.wall_velocity * traction, axis=-1)
                sign = -1.0 if side == "lower" else 1.0
                replacement = face_flux.at[..., system.energy_index].set(
                    mechanical + sign * outward_heat
                )
                index: list[slice | int] = [slice(None)] * output[axis].ndim
                index[axis] = face_index
                output[axis] = output[axis].at[tuple(index)].set(replacement)
        return tuple(output)

    def evaluate(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any = None,
        /,
    ) -> FiniteVolumeDiffusionEvaluation:
        value = jnp.asarray(state)
        gradient = self.conserved_gradient(
            system, time, value, discretization, halo, args
        )
        equation = system.diffusion_evaluation(value, gradient, args)
        output = []
        mapped = isinstance(discretization, MappedFiniteVolumeDiscretization)
        for axis in range(system.dimension):
            periodic = discretization.grid.structured_axes[axis].periodic
            tensor = _cell_to_faces(equation.flux, axis, periodic)
            if mapped:
                normal = (
                    discretization.face_area_vectors[axis]
                    / discretization.face_measures[axis][..., None]
                )
                output.append(ein.contract("...cd,...d->...c", tensor, normal))
            else:
                output.append(tensor[..., axis])
        fluxes = self._apply_prescribed_heat_flux(
            system, time, value, discretization, halo, tuple(output), args
        )
        finite = equation.finite & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(flux)) for flux in fluxes))
        )
        return FiniteVolumeDiffusionEvaluation(
            fluxes,
            equation.source,
            equation.source_step,
            finite,
            equation.successful & finite,
        )

    def face_fluxes(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any = None,
        /,
    ) -> tuple[Array, ...]:
        return self.evaluate(system, time, state, discretization, halo, args).face_fluxes

    @staticmethod
    def _face_distances(
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        axis: int,
        /,
    ) -> Array:
        measure = discretization.face_measures[axis]
        periodic = discretization.grid.structured_axes[axis].periodic
        if periodic:
            widths = discretization.grid.structured_axes[axis].interval_widths
            distance = 0.5 * (widths + jnp.roll(widths, 1))
            shape = [1] * measure.ndim
            shape[axis] = distance.size
            return jnp.broadcast_to(distance.reshape(tuple(shape)), measure.shape)
        centers = jnp.moveaxis(discretization.cell_centers, axis, 0)
        face_centers = jnp.moveaxis(discretization.face_centers[axis], axis, 0)
        normals = jnp.moveaxis(
            discretization.face_area_vectors[axis] / measure[..., None], axis, 0
        )
        lower_distance = 2.0 * jnp.abs(
            jnp.sum((centers[0] - face_centers[0]) * normals[0], axis=-1)
        )
        upper_distance = 2.0 * jnp.abs(
            jnp.sum((face_centers[-1] - centers[-1]) * normals[-1], axis=-1)
        )
        interior_distance = jnp.abs(
            jnp.sum((centers[1:] - centers[:-1]) * normals[1:-1], axis=-1)
        )
        return jnp.moveaxis(
            jnp.concatenate(
                (
                    lower_distance[None, ...],
                    interior_distance,
                    upper_distance[None, ...],
                ),
                axis=0,
            ),
            0,
            axis,
        )

    def stability_report(
        self,
        system: Any,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        args: Any = None,
        /,
        *,
        safety: float = 0.45,
        time: Array | None = None,
        halo: PreparedFiniteVolumeHaloPlan | None = None,
    ) -> ViscousStabilityReport:
        self._check_system(system)
        value = jnp.asarray(state)
        diffusivity = jnp.asarray(system.maximum_diffusivity(value, args))
        diffusivity = jnp.broadcast_to(diffusivity, discretization.cell_shape)
        cell_rate = jnp.zeros(discretization.cell_shape, dtype=value.dtype)
        for axis in range(system.dimension):
            periodic = discretization.grid.structured_axes[axis].periodic
            face_diffusivity = _cell_to_faces(diffusivity, axis, periodic)
            measure = discretization.face_measures[axis]
            distance = self._face_distances(discretization, axis)
            distance = eqx.error_if(
                distance,
                jnp.any(~jnp.isfinite(distance) | (distance <= 0.0)),
                "Viscous stability requires finite positive face distance.",
            )
            weight = 2.0 * measure * face_diffusivity / distance
            if periodic:
                contribution = weight + jnp.roll(weight, -1, axis=axis)
            else:
                lower: list[slice | int] = [slice(None)] * measure.ndim
                upper: list[slice | int] = [slice(None)] * measure.ndim
                lower[axis] = slice(0, measure.shape[axis] - 1)
                upper[axis] = slice(1, measure.shape[axis])
                contribution = weight[tuple(lower)] + weight[tuple(upper)]
            cell_rate = cell_rate + contribution / discretization.cell_volumes
        maximum_rate = jnp.max(cell_rate)
        spatial_step = jnp.where(
            maximum_rate > 0.0,
            jnp.asarray(safety, dtype=value.dtype) / maximum_rate,
            jnp.asarray(jnp.inf, dtype=value.dtype),
        )
        source_step = jnp.asarray(jnp.inf, dtype=value.dtype)
        if time is not None and halo is not None:
            source_step = jnp.min(
                self.evaluate(
                    system,
                    time,
                    value,
                    discretization,
                    halo,
                    args,
                ).source_step
            )
        return ViscousStabilityReport(
            maximum_rate,
            jnp.minimum(spatial_step, source_step),
            jnp.argmax(cell_rate),
            source_step,
        )

    def stable_step(
        self,
        system: Any,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        args: Any = None,
        /,
        *,
        safety: float = 0.45,
        time: Array | None = None,
        halo: PreparedFiniteVolumeHaloPlan | None = None,
    ) -> Array:
        return self.stability_report(
            system,
            state,
            discretization,
            args,
            safety=safety,
            time=time,
            halo=halo,
        ).selected_step

    def residual_from_evaluation(
        self,
        evaluation: FiniteVolumeDiffusionEvaluation,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        /,
    ) -> Array:
        if not isinstance(evaluation, FiniteVolumeDiffusionEvaluation):
            raise TypeError("evaluation must be FiniteVolumeDiffusionEvaluation.")
        residual = evaluation.cell_source
        for axis, flux in enumerate(evaluation.face_fluxes):
            integrated = flux * discretization.face_measures[axis][..., None]
            if discretization.grid.structured_axes[axis].periodic:
                difference = jnp.roll(integrated, -1, axis=axis) - integrated
            else:
                lower: list[slice | int] = [slice(None)] * integrated.ndim
                upper: list[slice | int] = [slice(None)] * integrated.ndim
                lower[axis] = slice(0, integrated.shape[axis] - 1)
                upper[axis] = slice(1, integrated.shape[axis])
                difference = integrated[tuple(upper)] - integrated[tuple(lower)]
            residual = residual + difference / discretization.cell_volumes[..., None]
        return residual

    def residual(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any = None,
        /,
    ) -> Array:
        evaluation = self.evaluate(system, time, state, discretization, halo, args)
        return self.residual_from_evaluation(evaluation, discretization)


__all__ = [
    "FiniteVolumeDiffusionEvaluation",
    "ViscousFluxPlan",
    "ViscousStabilityReport",
]

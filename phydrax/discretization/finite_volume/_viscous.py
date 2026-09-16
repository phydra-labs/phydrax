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
from ._rarefied_wall import MaxwellSmoluchowskiContinuumWallPlan
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

    def _apply_physical_wall_flux(
        self,
        system: Any,
        time: Array,
        state: Array,
        conserved_gradient: Array,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        fluxes: tuple[Array, ...],
        args: Any,
        /,
    ) -> tuple[tuple[Array, ...], Array]:
        output = list(fluxes)
        successful = jnp.asarray(True)
        for axis, pair in enumerate(halo.plan.boundaries.pairs):
            if pair is None:
                continue
            for side, boundary in (("lower", pair.lower), ("upper", pair.upper)):
                face_index = 0 if side == "lower" else output[axis].shape[axis] - 1
                cell_index = 0 if side == "lower" else state.shape[axis] - 1
                interior = jnp.take(state, cell_index, axis=axis)
                interior_gradient = jnp.take(conserved_gradient, cell_index, axis=axis)
                coordinates = jnp.take(
                    discretization.face_centers[axis], face_index, axis=axis
                )
                normal = discretization.outward_normal(axis, side)
                face_flux = jnp.take(output[axis], face_index, axis=axis)
                sign = -1.0 if side == "lower" else 1.0
                if isinstance(boundary, MaxwellSmoluchowskiContinuumWallPlan):
                    full_distance = self._face_distances(discretization, axis)
                    wall_distance = 0.5 * jnp.take(full_distance, face_index, axis=axis)
                    evaluation = boundary.evaluate_normal_flux(
                        system,
                        interior,
                        interior_gradient,
                        wall_distance,
                        normal,
                        args,
                    )
                    replacement = sign * evaluation.normal_diffusive_flux
                    successful = successful & evaluation.header.globally_eligible
                elif isinstance(boundary, PrescribedHeatFluxWallBoundary):
                    outward_heat = boundary.normal_heat_flux(
                        time, interior, coordinates, normal, args
                    )
                    traction = face_flux[..., system.momentum_slice]
                    mechanical = jnp.sum(boundary.wall_velocity * traction, axis=-1)
                    replacement = face_flux.at[..., system.energy_index].set(
                        mechanical + sign * outward_heat
                    )
                else:
                    continue
                index: list[slice | int] = [slice(None)] * output[axis].ndim
                index[axis] = face_index
                output[axis] = output[axis].at[tuple(index)].set(replacement)
        return tuple(output), successful

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
        fluxes, wall_successful = self._apply_physical_wall_flux(
            system,
            time,
            value,
            gradient,
            discretization,
            halo,
            tuple(output),
            args,
        )
        finite = equation.finite & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(flux)) for flux in fluxes))
        )
        return FiniteVolumeDiffusionEvaluation(
            fluxes,
            equation.source,
            equation.source_step,
            finite,
            equation.successful & wall_successful & finite,
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

    def unstructured_conserved_gradient(
        self,
        system: Any,
        state: ArrayLike,
        discretization,
        /,
    ) -> Array:
        """Least-squares conserved gradient on polygonal/polyhedral FV graphs."""

        from ._unstructured import UnstructuredFiniteVolumeDiscretization

        self._check_system(system)
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Unstructured viscous gradients require prepared unstructured FV."
            )
        value = jnp.asarray(state)
        if value.shape != discretization.state_shape:
            raise ValueError("Unstructured viscous state has incompatible shape.")
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        internal = neighbour >= 0
        internal_owner = owner[internal]
        internal_neighbour = neighbour[internal]
        displacement = (
            discretization.cell_centers[internal_neighbour]
            - discretization.cell_centers[internal_owner]
        )
        difference = value[internal_neighbour] - value[internal_owner]
        matrix_terms = displacement[..., :, None] * displacement[..., None, :]
        rhs_terms = difference[..., :, None] * displacement[..., None, :]
        dimension = discretization.cell_dimension
        matrix = jnp.zeros(
            (discretization.cell_count, dimension, dimension), dtype=value.dtype
        )
        right_hand_side = jnp.zeros(
            (discretization.cell_count, system.component_count, dimension),
            dtype=value.dtype,
        )
        matrix = matrix.at[internal_owner].add(matrix_terms)
        matrix = matrix.at[internal_neighbour].add(matrix_terms)
        right_hand_side = right_hand_side.at[internal_owner].add(rhs_terms)
        right_hand_side = right_hand_side.at[internal_neighbour].add(rhs_terms)
        length_scale = jnp.maximum(
            jnp.max(jnp.abs(discretization.cell_centers), axis=-1),
            jnp.asarray(1.0, dtype=value.dtype),
        )
        regularization = (
            64.0
            * jnp.finfo(value.dtype).eps
            * length_scale[:, None, None] ** 2
            * jnp.eye(dimension, dtype=value.dtype)[None, :, :]
        )
        inverse = dense_inverse(matrix + regularization, positive_definite=True)
        return ein.contract("nij,ncj->nci", inverse, right_hand_side)

    def evaluate_unstructured(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization,
        args: Any = None,
        /,
        *,
        boundary_normal_flux: ArrayLike | None = None,
    ) -> FiniteVolumeDiffusionEvaluation:
        """Evaluate equation-owned diffusion on arbitrary explicit FV faces."""

        from ._unstructured import UnstructuredFiniteVolumeDiscretization

        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Unstructured viscous evaluation requires prepared unstructured FV."
            )
        value = jnp.asarray(state)
        gradient = self.unstructured_conserved_gradient(
            system,
            value,
            discretization,
        )
        equation = system.diffusion_evaluation(value, gradient, args)
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        neighbour_tensor = equation.flux[safe_neighbour]
        tensor = 0.5 * (equation.flux[owner] + neighbour_tensor)
        normal = discretization.area_vectors / discretization.face_measures[..., None]
        face_flux = ein.contract("fcd,fd->fc", tensor, normal)
        boundary = neighbour < 0
        owner_flux = ein.contract("fcd,fd->fc", equation.flux[owner], normal)
        face_flux = jnp.where(boundary[:, None], owner_flux, face_flux)
        if boundary_normal_flux is not None:
            prescribed = jnp.asarray(boundary_normal_flux)
            if prescribed.shape != face_flux.shape:
                raise ValueError(
                    "boundary_normal_flux must contain one component flux per face."
                )
            face_flux = jnp.where(boundary[:, None], prescribed, face_flux)
        finite = equation.finite & jnp.all(jnp.isfinite(face_flux))
        return FiniteVolumeDiffusionEvaluation(
            (face_flux,),
            equation.source,
            equation.source_step,
            finite,
            equation.successful & finite,
        )

    def unstructured_residual_from_evaluation(
        self,
        evaluation: FiniteVolumeDiffusionEvaluation,
        discretization,
        /,
    ) -> Array:
        """Scatter one owner-oriented unstructured diffusive face ledger."""

        from ._unstructured import UnstructuredFiniteVolumeDiscretization

        if not isinstance(evaluation, FiniteVolumeDiffusionEvaluation) or not isinstance(
            discretization, UnstructuredFiniteVolumeDiscretization
        ):
            raise TypeError(
                "Unstructured viscous residual requires evaluation and discretization."
            )
        if len(evaluation.face_fluxes) != 1:
            raise ValueError("Unstructured viscous evaluation requires one face block.")
        integrated = evaluation.face_fluxes[0] * discretization.face_measures[..., None]
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        residual = jnp.asarray(evaluation.cell_source)
        residual = residual.at[owner].add(
            integrated / discretization.cell_volumes[owner, None]
        )
        internal = neighbour >= 0
        residual = residual.at[neighbour[internal]].add(
            -integrated[internal] / discretization.cell_volumes[neighbour[internal], None]
        )
        return residual

    def unstructured_residual(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization,
        args: Any = None,
        /,
        *,
        boundary_normal_flux: ArrayLike | None = None,
    ) -> Array:
        evaluation = self.evaluate_unstructured(
            system,
            time,
            state,
            discretization,
            args,
            boundary_normal_flux=boundary_normal_flux,
        )
        return self.unstructured_residual_from_evaluation(evaluation, discretization)

    def unstructured_stability_report(
        self,
        system: Any,
        state: ArrayLike,
        discretization,
        args: Any = None,
        /,
        *,
        safety: float = 0.45,
    ) -> ViscousStabilityReport:
        """Conservative graph-based explicit diffusion step bound."""

        from ._unstructured import UnstructuredFiniteVolumeDiscretization

        self._check_system(system)
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Unstructured viscous stability requires prepared unstructured FV."
            )
        value = jnp.asarray(state)
        diffusivity = jnp.broadcast_to(
            jnp.asarray(system.maximum_diffusivity(value, args)),
            (discretization.cell_count,),
        )
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        face_diffusivity = jnp.where(
            neighbour >= 0,
            0.5 * (diffusivity[owner] + diffusivity[safe_neighbour]),
            diffusivity[owner],
        )
        owner_distance = jnp.abs(
            jnp.sum(
                (discretization.face_centers - discretization.cell_centers[owner])
                * (discretization.area_vectors / discretization.face_measures[:, None]),
                axis=-1,
            )
        )
        neighbour_distance = jnp.where(
            neighbour >= 0,
            jnp.abs(
                jnp.sum(
                    (
                        discretization.cell_centers[safe_neighbour]
                        - discretization.face_centers
                    )
                    * (
                        discretization.area_vectors
                        / discretization.face_measures[:, None]
                    ),
                    axis=-1,
                )
            ),
            owner_distance,
        )
        distance = owner_distance + neighbour_distance
        distance = eqx.error_if(
            distance,
            jnp.any(~jnp.isfinite(distance) | (distance <= 0.0)),
            "Unstructured viscous stability requires positive face distances.",
        )
        weight = 2.0 * discretization.face_measures * face_diffusivity / distance
        rate = jnp.zeros((discretization.cell_count,), dtype=value.dtype)
        rate = rate.at[owner].add(weight)
        internal = neighbour >= 0
        rate = rate.at[neighbour[internal]].add(weight[internal])
        rate = rate / discretization.cell_volumes
        maximum_rate = jnp.max(rate)
        step = jnp.where(
            maximum_rate > 0.0,
            jnp.asarray(safety, dtype=value.dtype) / maximum_rate,
            jnp.asarray(jnp.inf, dtype=value.dtype),
        )
        return ViscousStabilityReport(
            maximum_rate,
            step,
            jnp.argmax(rate),
            jnp.asarray(jnp.inf, dtype=value.dtype),
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nudged elastic bands, saddle qualification, and intrinsic reaction paths."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure, AtomisticSystemPlan
from .._optimization import _require_structure_matches_system
from .._surface import AbstractPreparedPotentialEnergySurface
from ..vibration._harmonic import StationaryPointKind, VibrationalAnalysisResult


class ReactionPathResult(StrictModule, NonTrainableState):
    images: Array
    energies: Array
    neb_forces: Array
    iterations: Array
    provider_evaluations: Array
    successful: Array
    climbing_image: int = eqx.field(static=True)
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        images: ArrayLike,
        energies: ArrayLike,
        neb_forces: ArrayLike,
        iterations: int,
        provider_evaluations: int,
        successful: ArrayLike,
        /,
        *,
        climbing_image: int,
        source_result_ids: tuple[str, ...],
        plan_id: str,
    ):
        images_ = jnp.asarray(images)
        energies_ = jnp.asarray(energies, dtype=images_.dtype)
        forces_ = jnp.asarray(neb_forces, dtype=images_.dtype)
        if images_.ndim != 3 or images_.shape[2] != 3:
            raise ValueError("Reaction path images must have shape (image, atom, 3).")
        if energies_.shape != (images_.shape[0],) or forces_.shape != images_.shape:
            raise ValueError("Reaction path energies and forces must align with images.")
        climbing = int(climbing_image)
        if climbing <= 0 or climbing >= images_.shape[0] - 1:
            raise ValueError("Climbing image must be an interior image.")
        sources = tuple(str(value).strip() for value in source_result_ids)
        if not sources or any(not value for value in sources):
            raise ValueError("Reaction path source result IDs must be non-empty.")
        self.images = images_
        self.energies = energies_
        self.neb_forces = forces_
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.provider_evaluations = jnp.asarray(
            provider_evaluations, dtype=jnp.int32
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.climbing_image = climbing
        self.source_result_ids = sources
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "reaction-path-result",
                "plan": self.plan_id,
                "climbing_image": climbing,
                "sources": list(sources),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "images": np.asarray(images_),
                        "energies": np.asarray(energies_),
                        "forces": np.asarray(forces_),
                        "iterations": np.asarray(self.iterations),
                        "provider_evaluations": np.asarray(self.provider_evaluations),
                    }
                ),
            }
        )

    @property
    def transition_state_positions(self) -> Array:
        return self.images[self.climbing_image]


class NudgedElasticBandPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    image_count: int = eqx.field(static=True)
    spring_constant: float = eqx.field(static=True)
    initial_step_size: float = eqx.field(static=True)
    maximum_step_size: float = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    climbing_start: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        /,
        *,
        image_count: int = 7,
        spring_constant: float = 1.0,
        initial_step_size: float = 0.02,
        maximum_step_size: float = 0.2,
        force_tolerance: float = 1.0e-3,
        maximum_steps: int = 500,
        climbing_start: int = 20,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if not isinstance(surface, AbstractPreparedPotentialEnergySurface):
            raise TypeError("surface must be a prepared potential-energy surface.")
        if surface.system_id != system.system_id or not surface.capabilities.forces:
            raise ValueError("NEB requires a force-capable surface for the same system.")
        if not np.any(
            np.asarray(system.active_mask, dtype=np.bool_)
            & np.asarray(system.mobile_mask, dtype=np.bool_)
        ):
            raise ValueError("NEB requires at least one active mobile atom.")
        images = int(image_count)
        steps = int(maximum_steps)
        climb = int(climbing_start)
        values = tuple(
            float(value)
            for value in (
                spring_constant,
                initial_step_size,
                maximum_step_size,
                force_tolerance,
            )
        )
        if images < 3 or steps <= 0 or climb < 0:
            raise ValueError("NEB image/work counts are invalid.")
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "NEB spring, step, and force values must be positive finite."
            )
        if values[1] > values[2]:
            raise ValueError("initial_step_size cannot exceed maximum_step_size.")
        self.system = system
        self.surface = surface
        self.image_count = images
        (
            self.spring_constant,
            self.initial_step_size,
            self.maximum_step_size,
            self.force_tolerance,
        ) = values
        self.maximum_steps = steps
        self.climbing_start = climb
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nudged-elastic-band-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "image_count": images,
                "spring_constant": values[0],
                "initial_step_size": values[1],
                "maximum_step_size": values[2],
                "force_tolerance": values[3],
                "maximum_steps": steps,
                "climbing_start": climb,
            }
        )

    def run(
        self,
        reactant: AtomicStructure,
        product: AtomicStructure,
        /,
        *,
        initial_images: ArrayLike | None = None,
    ) -> ReactionPathResult:
        _require_structure_matches_system(reactant, self.system)
        _require_structure_matches_system(product, self.system)
        if not np.array_equal(
            np.asarray(reactant.particle_ids), np.asarray(product.particle_ids)
        ):
            raise ValueError("Reaction endpoints must preserve stable particle order.")
        left = np.asarray(
            reactant.positions,
            dtype=np.dtype(self.system.coordinate_dtype),
        )
        right = np.asarray(product.positions, dtype=left.dtype)
        active = np.asarray(self.system.active_mask, dtype=np.bool_)
        movable = np.asarray(self.system.mobile_mask, dtype=np.bool_)
        path_mask = active & movable
        fixed_active = active & ~movable
        if not np.array_equal(
            left[fixed_active],
            right[fixed_active],
            equal_nan=True,
        ):
            raise ValueError(
                "Reaction endpoints must agree at every nonmobile active atom."
            )
        if initial_images is None:
            fractions = np.linspace(0.0, 1.0, self.image_count)
            images = (
                left[None, :, :] + fractions[:, None, None] * (right - left)[None, :, :]
            )
        else:
            images = np.asarray(initial_images, dtype=left.dtype).copy()
            if images.shape != (self.image_count, *left.shape):
                raise ValueError("initial_images has an incompatible shape.")
            if not np.array_equal(
                images[0][active],
                left[active],
                equal_nan=True,
            ) or not np.array_equal(
                images[-1][active],
                right[active],
                equal_nan=True,
            ):
                raise ValueError(
                    "initial_images must preserve active endpoint coordinates."
                )
            if not np.array_equal(
                images[:, fixed_active],
                np.broadcast_to(
                    left[fixed_active],
                    (self.image_count, int(np.count_nonzero(fixed_active)), 3),
                ),
                equal_nan=True,
            ):
                raise ValueError(
                    "initial_images must keep nonmobile active coordinates fixed."
                )
        cell = None if reactant.cell is None else np.asarray(reactant.cell)
        velocities = np.zeros_like(images)
        time_step = self.initial_step_size
        alpha = 0.1
        positive_steps = 0
        sources: list[str] = []
        evaluations = 0
        final_energies = np.full((self.image_count,), np.nan)
        final_forces = np.zeros_like(images)
        climbing = max(1, self.image_count // 2)
        successful = False
        completed_steps = 0
        for step in range(self.maximum_steps):
            evaluated = tuple(self.surface.evaluate(image, cell) for image in images)
            evaluations += self.image_count
            sources.extend(value.source_result_id for value in evaluated)
            if not all(bool(value.successful) for value in evaluated):
                break
            energies = np.asarray([float(value.energy) for value in evaluated])
            physical_forces = np.asarray(
                [np.asarray(value.forces) for value in evaluated]
            )
            climbing = int(np.argmax(energies[1:-1])) + 1
            neb_forces = np.zeros_like(physical_forces)
            for image in range(1, self.image_count - 1):
                previous = images[image] - images[image - 1]
                following = images[image + 1] - images[image]
                previous_energy = energies[image - 1]
                energy = energies[image]
                following_energy = energies[image + 1]
                if following_energy > energy > previous_energy:
                    tangent = following
                elif following_energy < energy < previous_energy:
                    tangent = previous
                else:
                    high = max(
                        abs(following_energy - energy), abs(previous_energy - energy)
                    )
                    low = min(
                        abs(following_energy - energy), abs(previous_energy - energy)
                    )
                    tangent = (
                        following * high + previous * low
                        if following_energy > previous_energy
                        else following * low + previous * high
                    )
                tangent = np.array(tangent, copy=True)
                tangent[~path_mask] = 0.0
                tangent_norm = float(np.sqrt(np.sum(tangent * tangent)))
                if not isfinite(tangent_norm) or tangent_norm <= 0.0:
                    raise ValueError("NEB encountered a zero or nonfinite tangent.")
                tangent /= tangent_norm
                force = np.array(physical_forces[image], copy=True)
                force[~path_mask] = 0.0
                parallel = float(np.sum(force * tangent)) * tangent
                perpendicular = force - parallel
                spring = (
                    self.spring_constant
                    * (
                        float(np.sqrt(np.sum(following[path_mask] ** 2)))
                        - float(np.sqrt(np.sum(previous[path_mask] ** 2)))
                    )
                    * tangent
                )
                neb_forces[image] = perpendicular + spring
                if step >= self.climbing_start and image == climbing:
                    neb_forces[image] = force - 2.0 * parallel
                neb_forces[image, ~path_mask] = 0.0
            maximum_force = float(np.max(np.abs(neb_forces[1:-1, path_mask])))
            final_energies = energies
            final_forces = neb_forces
            completed_steps = step + 1
            if step >= self.climbing_start and maximum_force <= self.force_tolerance:
                successful = True
                break
            interior_velocity = velocities[1:-1]
            interior_force = neb_forces[1:-1]
            interior_velocity += time_step * interior_force
            power = float(np.sum(interior_velocity * interior_force))
            velocity_norm = float(np.sqrt(np.sum(interior_velocity * interior_velocity)))
            force_norm = float(np.sqrt(np.sum(interior_force * interior_force)))
            if power > 0.0:
                positive_steps += 1
                if velocity_norm > 0.0 and force_norm > 0.0:
                    interior_velocity[:] = (
                        (1.0 - alpha) * interior_velocity
                        + alpha * interior_force * velocity_norm / force_norm
                    )
                if positive_steps > 5:
                    time_step = min(1.1 * time_step, self.maximum_step_size)
                    alpha *= 0.99
            else:
                positive_steps = 0
                time_step *= 0.5
                alpha = 0.1
                interior_velocity.fill(0.0)
            images[1:-1] += time_step * interior_velocity
            images[0] = left
            images[-1] = right
        return ReactionPathResult(
            images,
            final_energies,
            final_forces,
            completed_steps,
            evaluations,
            successful,
            climbing_image=climbing,
            source_result_ids=tuple(sources),
            plan_id=self.plan_id,
        )


class ReactionPathQualificationResult(StrictModule, NonTrainableState):
    tangent_overlap: Array
    successful: Array
    path_result_id: str = eqx.field(static=True)
    vibration_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: ReactionPathResult,
        vibration: VibrationalAnalysisResult,
        masses: ArrayLike,
        /,
        *,
        minimum_overlap: float = 0.5,
    ):
        if vibration.stationary_point is not StationaryPointKind.FIRST_ORDER_SADDLE:
            overlap = 0.0
            successful = False
        else:
            index = path.climbing_image
            tangent = np.asarray(path.images[index + 1] - path.images[index - 1])
            imaginary_index = int(np.flatnonzero(np.asarray(vibration.imaginary_mask))[0])
            mode = np.asarray(vibration.normal_modes)[..., imaginary_index]
            root_mass = np.sqrt(np.asarray(masses))[:, None]
            tangent_mass = root_mass * tangent
            mode_mass = root_mass * mode
            denominator = float(
                np.sqrt(
                    np.sum(tangent_mass * tangent_mass) * np.sum(mode_mass * mode_mass)
                )
            )
            overlap = (
                0.0
                if denominator == 0.0
                else abs(float(np.sum(tangent_mass * mode_mass))) / denominator
            )
            successful = (
                bool(path.successful)
                and bool(vibration.successful)
                and overlap >= float(minimum_overlap)
            )
        self.tangent_overlap = jnp.asarray(overlap)
        self.successful = jnp.asarray(successful, dtype=jnp.bool_)
        self.path_result_id = path.result_id
        self.vibration_result_id = vibration.result_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "reaction-path-qualification",
                "path": path.result_id,
                "vibration": vibration.result_id,
                "tangent_overlap": overlap,
                "successful": successful,
            }
        )


class IntrinsicReactionCoordinateResult(StrictModule, NonTrainableState):
    forward: Array
    reverse: Array
    forward_energies: Array
    reverse_energies: Array
    successful: Array
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: ArrayLike,
        reverse: ArrayLike,
        forward_energies: ArrayLike,
        reverse_energies: ArrayLike,
        successful: ArrayLike,
        source_result_ids: tuple[str, ...],
        /,
    ):
        forward_ = jnp.asarray(forward)
        reverse_ = jnp.asarray(reverse, dtype=forward_.dtype)
        forward_energy = jnp.asarray(forward_energies, dtype=forward_.dtype)
        reverse_energy = jnp.asarray(reverse_energies, dtype=forward_.dtype)
        if forward_.ndim != 3 or reverse_.ndim != 3:
            raise ValueError("IRC branches must have shape (step, atom, 3).")
        if forward_energy.shape != (forward_.shape[0],) or reverse_energy.shape != (
            reverse_.shape[0],
        ):
            raise ValueError("IRC energies must align with branch steps.")
        self.forward = forward_
        self.reverse = reverse_
        self.forward_energies = forward_energy
        self.reverse_energies = reverse_energy
        self.successful = jnp.asarray(successful, dtype=jnp.bool_)
        self.source_result_ids = source_result_ids
        self.result_id = canonical_fingerprint(
            {
                "kind": "intrinsic-reaction-coordinate-result",
                "sources": list(source_result_ids),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "forward": np.asarray(forward_),
                        "reverse": np.asarray(reverse_),
                        "forward_energies": np.asarray(forward_energy),
                        "reverse_energies": np.asarray(reverse_energy),
                    }
                ),
            }
        )


class IntrinsicReactionCoordinatePlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    step_size: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    corrector_iterations: int = eqx.field(static=True)
    corrector_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        /,
        *,
        step_size: float = 0.02,
        maximum_steps: int = 200,
        force_tolerance: float = 1.0e-3,
        corrector_iterations: int = 8,
        corrector_tolerance: float = 1.0e-4,
    ):
        if surface.system_id != system.system_id or not surface.capabilities.forces:
            raise ValueError("IRC requires a force-capable surface for the same system.")
        step = float(step_size)
        tolerance = float(force_tolerance)
        steps = int(maximum_steps)
        corrector_steps = int(corrector_iterations)
        corrector = float(corrector_tolerance)
        if (
            not isfinite(step)
            or step <= 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or not isfinite(corrector)
            or corrector <= 0.0
        ):
            raise ValueError("IRC step and tolerances must be positive finite.")
        if steps <= 0 or corrector_steps < 0:
            raise ValueError(
                "IRC steps must be positive and corrector_iterations non-negative."
            )
        self.system = system
        self.surface = surface
        self.step_size = step
        self.maximum_steps = steps
        self.force_tolerance = tolerance
        self.corrector_iterations = corrector_steps
        self.corrector_tolerance = corrector
        self.plan_id = canonical_fingerprint(
            {
                "kind": "intrinsic-reaction-coordinate-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "step_size": step,
                "maximum_steps": steps,
                "force_tolerance": tolerance,
                "corrector_iterations": corrector_steps,
                "corrector_tolerance": corrector,
            }
        )

    def run(
        self,
        transition_state: AtomicStructure,
        imaginary_mode: ArrayLike,
        /,
    ) -> IntrinsicReactionCoordinateResult:
        _require_structure_matches_system(transition_state, self.system)
        mode = np.array(
            imaginary_mode,
            dtype=np.dtype(self.system.coordinate_dtype),
            copy=True,
        )
        if mode.shape != transition_state.positions.shape:
            raise ValueError("imaginary_mode must align with transition-state positions.")
        masses = np.asarray(self.system.masses)[:, None]
        active = np.asarray(self.system.active_mask, dtype=np.bool_)
        movable = np.asarray(self.system.mobile_mask, dtype=np.bool_)
        mode[~movable] = 0.0
        norm = float(np.sqrt(np.sum(masses[active] * mode[active] ** 2)))
        if not isfinite(norm) or norm <= 0.0:
            raise ValueError("imaginary_mode has zero or nonfinite mass-weighted norm.")
        mode /= norm
        cell = (
            None if transition_state.cell is None else np.asarray(transition_state.cell)
        )
        source_ids: list[str] = []

        def branch(direction: float):
            position = (
                np.asarray(transition_state.positions).copy()
                + direction * self.step_size * mode
            )
            positions = [position.copy()]
            energies = []
            converged = False
            for iteration in range(self.maximum_steps):
                value = self.surface.evaluate(position, cell)
                source_ids.append(value.source_result_id)
                energies.append(float(value.energy))
                if not bool(value.successful):
                    break
                force = np.asarray(value.forces).copy()
                force[~movable] = 0.0
                maximum = float(np.max(np.abs(force[active])))
                if maximum <= self.force_tolerance:
                    converged = True
                    break
                mass_direction = force / masses
                mass_direction[~movable] = 0.0
                direction_norm = float(
                    np.sqrt(np.sum(masses[active] * mass_direction[active] ** 2))
                )
                if not isfinite(direction_norm) or direction_norm <= 0.0:
                    break
                origin = position.copy()
                candidate = origin + self.step_size * mass_direction / direction_norm
                for _ in range(self.corrector_iterations):
                    corrected = self.surface.evaluate(candidate, cell)
                    source_ids.append(corrected.source_result_id)
                    if not bool(corrected.successful):
                        break
                    radial = candidate - origin
                    radial[~movable] = 0.0
                    radial_norm = float(
                        np.sqrt(np.sum(masses[active] * radial[active] ** 2))
                    )
                    if not isfinite(radial_norm) or radial_norm <= 0.0:
                        break
                    tangent = radial / radial_norm
                    corrected_direction = np.asarray(corrected.forces) / masses
                    corrected_direction[~movable] = 0.0
                    projection = float(
                        np.sum(
                            masses[active] * corrected_direction[active] * tangent[active]
                        )
                    )
                    perpendicular = corrected_direction - projection * tangent
                    perpendicular_norm = float(
                        np.sqrt(np.sum(masses[active] * perpendicular[active] ** 2))
                    )
                    if perpendicular_norm <= self.corrector_tolerance:
                        break
                    candidate = candidate + 0.5 * self.step_size * perpendicular
                    radial = candidate - origin
                    radial_norm = float(
                        np.sqrt(np.sum(masses[active] * radial[active] ** 2))
                    )
                    candidate = origin + self.step_size * radial / radial_norm
                position = candidate
                if iteration + 1 < self.maximum_steps:
                    positions.append(position.copy())
            return np.asarray(positions), np.asarray(energies), converged

        forward, forward_energy, forward_success = branch(1.0)
        reverse, reverse_energy, reverse_success = branch(-1.0)
        return IntrinsicReactionCoordinateResult(
            forward,
            reverse,
            forward_energy,
            reverse_energy,
            forward_success and reverse_success,
            tuple(source_ids),
        )


__all__ = [
    "IntrinsicReactionCoordinatePlan",
    "IntrinsicReactionCoordinateResult",
    "NudgedElasticBandPlan",
    "ReactionPathQualificationResult",
    "ReactionPathResult",
]

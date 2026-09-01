#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._reference import (
    MemberKinematics,
    MemberNetworkDefinition,
    rotation_vector_matrix,
)


def _wrapped_angle(value: Array, /) -> Array:
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def _so3_log(rotation: Array, /) -> Array:
    trace = jnp.trace(rotation, axis1=-2, axis2=-1)
    cosine = jnp.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    angle = jnp.arccos(cosine)
    vector = jnp.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        axis=-1,
    )
    sine = jnp.sin(angle)
    scale = jnp.where(
        jnp.abs(sine) > 1.0e-8,
        angle / (2.0 * sine),
        0.5 + angle * angle / 12.0,
    )
    return scale[..., None] * vector


def _frame_from_chord_director(chord: Array, director: Array, /) -> tuple[Array, Array]:
    length = jnp.sqrt(jnp.sum(chord * chord, axis=-1))
    first = chord / jnp.maximum(length[:, None], jnp.finfo(chord.dtype).tiny)
    projected = director - jnp.sum(director * first, axis=-1)[:, None] * first
    margin = jnp.sqrt(jnp.sum(projected * projected, axis=-1))
    second = projected / jnp.maximum(margin[:, None], jnp.finfo(chord.dtype).tiny)
    third = jnp.cross(first, second)
    return jnp.stack((first, second, third), axis=-1), margin


class AxialConstitutiveState(StrictModule):
    """Member strain energy, force, tangent, and unilateral evidence."""

    effective_rest_length: Array
    strain: Array
    energy: Array
    axial_force: Array
    tangent: Array
    active: Array
    switching_margin: Array
    valid: Array


class AbstractAxialLaw(StrictModule, NonTrainableState):
    law_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        length: Array,
        rest_length: Array,
        axial_rigidity: Array,
        initial_strain: Array,
        thermal_strain: Array,
        /,
    ) -> AxialConstitutiveState:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_rest_length(
        self,
        length: Array,
        axial_force: Array,
        axial_rigidity: Array,
        /,
    ) -> Array:
        raise NotImplementedError


class LinearAxialLaw(AbstractAxialLaw):
    """Engineering-strain elastic axial law carrying tension or compression."""

    def __init__(self, *, law_id: str = "axial-law:linear-elastic"):
        self.law_id = str(law_id)

    def evaluate(
        self,
        length: Array,
        rest_length: Array,
        axial_rigidity: Array,
        initial_strain: Array,
        thermal_strain: Array,
        /,
    ) -> AxialConstitutiveState:
        effective = rest_length * (1.0 + initial_strain + thermal_strain)
        extension = length - effective
        strain = extension / rest_length
        tangent = axial_rigidity / rest_length
        force = tangent * extension
        energy = 0.5 * tangent * extension**2
        valid = (
            jnp.isfinite(length)
            & jnp.isfinite(effective)
            & (length > 0.0)
            & (effective > 0.0)
            & (axial_rigidity > 0.0)
        )
        return AxialConstitutiveState(
            effective,
            strain,
            energy,
            force,
            tangent,
            jnp.ones_like(force, dtype=bool),
            jnp.abs(extension),
            valid,
        )

    def inverse_rest_length(
        self,
        length: Array,
        axial_force: Array,
        axial_rigidity: Array,
        /,
    ) -> Array:
        return axial_rigidity * length / (axial_rigidity + axial_force)


class TensionOnlyCableLaw(AbstractAxialLaw):
    """Exact positive-part elastic cable energy with active-set evidence."""

    activation_tolerance: float = eqx.field(static=True)
    deactivation_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        activation_tolerance: float = 1.0e-10,
        deactivation_tolerance: float = 1.0e-10,
        law_id: str = "axial-law:tension-only",
    ):
        values = (float(activation_tolerance), float(deactivation_tolerance))
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Cable active-set tolerances must be finite and nonnegative."
            )
        self.activation_tolerance, self.deactivation_tolerance = values
        self.law_id = str(law_id)

    def evaluate(
        self,
        length: Array,
        rest_length: Array,
        axial_rigidity: Array,
        initial_strain: Array,
        thermal_strain: Array,
        /,
    ) -> AxialConstitutiveState:
        effective = rest_length * (1.0 + initial_strain + thermal_strain)
        extension = length - effective
        active = extension > 0.0
        positive = jnp.maximum(extension, 0.0)
        tangent_value = axial_rigidity / rest_length
        force = tangent_value * positive
        energy = 0.5 * tangent_value * positive**2
        strain = positive / rest_length
        tangent = jnp.where(active, tangent_value, 0.0)
        valid = (
            jnp.isfinite(length)
            & jnp.isfinite(effective)
            & (length > 0.0)
            & (effective > 0.0)
            & (axial_rigidity > 0.0)
        )
        return AxialConstitutiveState(
            effective,
            strain,
            energy,
            force,
            tangent,
            active,
            jnp.abs(extension),
            valid,
        )

    def evaluate_active(
        self,
        length: Array,
        rest_length: Array,
        axial_rigidity: Array,
        initial_strain: Array,
        thermal_strain: Array,
        active: Array,
        /,
    ) -> AxialConstitutiveState:
        effective = rest_length * (1.0 + initial_strain + thermal_strain)
        extension = length - effective
        tangent_value = axial_rigidity / rest_length
        force = jnp.where(active, tangent_value * extension, 0.0)
        energy = jnp.where(active, 0.5 * tangent_value * extension**2, 0.0)
        strain = jnp.where(active, extension / rest_length, 0.0)
        tangent = jnp.where(active, tangent_value, 0.0)
        valid = (
            jnp.isfinite(length)
            & jnp.isfinite(effective)
            & (length > 0.0)
            & (effective > 0.0)
            & (axial_rigidity > 0.0)
        )
        return AxialConstitutiveState(
            effective,
            strain,
            energy,
            force,
            tangent,
            active,
            jnp.abs(extension),
            valid,
        )

    def inverse_rest_length(
        self,
        length: Array,
        axial_force: Array,
        axial_rigidity: Array,
        /,
    ) -> Array:
        force = eqx.error_if(
            axial_force,
            jnp.any(axial_force < 0.0),
            "Tension-only cable targets may not contain compression.",
        )
        return axial_rigidity * length / (axial_rigidity + force)


def _evaluate_axial_law(
    law: AbstractAxialLaw,
    length: Array,
    rest_length: Array,
    axial_rigidity: Array,
    initial_strain: Array,
    thermal_strain: Array,
    active: Array,
    /,
) -> AxialConstitutiveState:
    if isinstance(law, TensionOnlyCableLaw):
        return law.evaluate_active(
            length,
            rest_length,
            axial_rigidity,
            initial_strain,
            thermal_strain,
            active,
        )
    return law.evaluate(
        length,
        rest_length,
        axial_rigidity,
        initial_strain,
        thermal_strain,
    )


class MemberBlockEvaluation(StrictModule):
    """One block's energy and member-resultant arrays."""

    energy: Array
    member_indices: Array
    axial_force: Array
    shear: Array
    bending_moment: Array
    torsion: Array
    active: Array
    unilateral: Array
    switching_margin: Array
    valid: Array


class AbstractMemberBlock(StrictModule, NonTrainableState):
    block_id: str = eqx.field(static=True)
    member_indices: Array

    @abc.abstractmethod
    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberBlockEvaluation:
        raise NotImplementedError


class AxialMemberBlock(AbstractMemberBlock):
    """Translation-only truss or cable members."""

    law: AbstractAxialLaw

    def __init__(
        self,
        member_indices: ArrayLike,
        law: AbstractAxialLaw | None = None,
        /,
        *,
        block_id: str | None = None,
    ):
        indices = jnp.asarray(member_indices, dtype=jnp.int32)
        if indices.ndim != 1:
            raise ValueError("member_indices must be rank one.")
        law_ = LinearAxialLaw() if law is None else law
        if not isinstance(law_, AbstractAxialLaw):
            raise TypeError("law must be an AbstractAxialLaw.")
        self.member_indices = indices
        self.law = law_
        self.block_id = str(
            block_id
            or canonical_fingerprint(
                {
                    "kind": "axial-member-block",
                    "members": array_tree_fingerprint(indices),
                    "law": law_.law_id,
                }
            )
        )

    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberBlockEvaluation:
        structure = definition.structure
        indices = self.member_indices
        senders = structure.senders[indices]
        receivers = structure.receivers[indices]
        vectors = kinematics.positions[receivers] - kinematics.positions[senders]
        lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        properties = definition.properties.structural_arrays()
        reference = definition.reference
        materials = definition.properties.materials
        thermal_coefficients = jnp.asarray(
            [value.thermal_expansion for value in materials]
        )[definition.properties.member_material[indices]]
        state = _evaluate_axial_law(
            self.law,
            lengths,
            reference.rest_lengths[indices],
            properties["young"][indices] * properties["area"][indices],
            reference.initial_strain[indices],
            thermal_coefficients * reference.initial_temperature[indices],
            reference.cable_active[indices],
        )
        dimension = definition.structure.dimension
        return MemberBlockEvaluation(
            jnp.sum(state.energy),
            indices,
            state.axial_force,
            jnp.zeros((indices.size, max(dimension - 1, 1)), dtype=lengths.dtype),
            jnp.zeros((indices.size, max(dimension - 1, 1)), dtype=lengths.dtype),
            jnp.zeros((indices.size,), dtype=lengths.dtype),
            state.active,
            jnp.full(
                state.active.shape,
                isinstance(self.law, TensionOnlyCableLaw),
                dtype=bool,
            ),
            state.switching_margin,
            jnp.all(state.valid),
        )


class CorotationalFrameBlock(AbstractMemberBlock):
    """Objective 2-D/3-D corotational Timoshenko frame members."""

    axial_law: AbstractAxialLaw
    director_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        member_indices: ArrayLike,
        /,
        *,
        axial_law: AbstractAxialLaw | None = None,
        director_tolerance: float = 1.0e-10,
        block_id: str | None = None,
    ):
        indices = jnp.asarray(member_indices, dtype=jnp.int32)
        if indices.ndim != 1:
            raise ValueError("member_indices must be rank one.")
        law = LinearAxialLaw() if axial_law is None else axial_law
        if not isinstance(law, AbstractAxialLaw):
            raise TypeError("axial_law must be an AbstractAxialLaw.")
        tolerance = float(director_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("director_tolerance must be finite and positive.")
        self.member_indices = indices
        self.axial_law = law
        self.director_tolerance = tolerance
        self.block_id = str(
            block_id
            or canonical_fingerprint(
                {
                    "kind": "corotational-frame-block",
                    "members": array_tree_fingerprint(indices),
                    "law": law.law_id,
                    "director_tolerance": tolerance,
                }
            )
        )

    @staticmethod
    def _timoshenko_energy(
        first: Array,
        second: Array,
        elastic_bending: Array,
        shear_rigidity: Array,
        length: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        ratio = (
            12.0
            * elastic_bending
            / jnp.maximum(shear_rigidity * length**2, jnp.finfo(length.dtype).tiny)
        )
        common = elastic_bending / (length * (1.0 + ratio))
        diagonal = (4.0 + ratio) * common
        coupling = (2.0 - ratio) * common
        energy = 0.5 * (
            diagonal * (first**2 + second**2) + 2.0 * coupling * first * second
        )
        moment_first = diagonal * first + coupling * second
        moment_second = coupling * first + diagonal * second
        return energy, moment_first, moment_second

    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberBlockEvaluation:
        structure = definition.structure
        reference = definition.reference
        properties = definition.properties.structural_arrays()
        indices = self.member_indices
        senders = structure.senders[indices]
        receivers = structure.receivers[indices]
        current_vector = kinematics.positions[receivers] - kinematics.positions[senders]
        reference_vector = reference.positions[receivers] - reference.positions[senders]
        lengths = jnp.sqrt(jnp.sum(current_vector * current_vector, axis=-1))
        axial = _evaluate_axial_law(
            self.axial_law,
            lengths,
            reference.rest_lengths[indices],
            properties["young"][indices] * properties["area"][indices],
            reference.initial_strain[indices],
            jnp.zeros_like(lengths),
            reference.cable_active[indices],
        )
        if structure.dimension == 2:
            current_angle = jnp.arctan2(current_vector[:, 1], current_vector[:, 0])
            reference_angle = jnp.arctan2(reference_vector[:, 1], reference_vector[:, 0])
            chord_change = _wrapped_angle(current_angle - reference_angle)
            first = _wrapped_angle(
                kinematics.rotation_vectors[senders, 0]
                - reference.rotation_vectors[senders, 0]
                - chord_change
            )
            second = _wrapped_angle(
                kinematics.rotation_vectors[receivers, 0]
                - reference.rotation_vectors[receivers, 0]
                - chord_change
            )
            bending, moment_first, moment_second = self._timoshenko_energy(
                first,
                second,
                properties["young"][indices] * properties["inertia_z"][indices],
                properties["shear"][indices] * properties["shear_area_y"][indices],
                lengths,
            )
            bending_moment = 0.5 * (moment_first + moment_second)[:, None]
            shear = ((moment_first + moment_second) / lengths)[:, None]
            torsion = jnp.zeros_like(lengths)
            frame_valid = jnp.ones_like(lengths, dtype=bool)
        else:
            current_rotations = rotation_vector_matrix(kinematics.rotation_vectors)
            reference_rotations = rotation_vector_matrix(reference.rotation_vectors)
            current_director = 0.5 * (
                current_rotations[senders, :, 1] + current_rotations[receivers, :, 1]
            )
            reference_director = 0.5 * (
                reference_rotations[senders, :, 1] + reference_rotations[receivers, :, 1]
            )
            current_frame, current_margin = _frame_from_chord_director(
                current_vector, current_director
            )
            reference_frame, reference_margin = _frame_from_chord_director(
                reference_vector, reference_director
            )
            local_reference_first = (
                jnp.swapaxes(reference_frame, -1, -2) @ reference_rotations[senders]
            )
            local_reference_second = (
                jnp.swapaxes(reference_frame, -1, -2) @ reference_rotations[receivers]
            )
            local_current_first = (
                jnp.swapaxes(current_frame, -1, -2) @ current_rotations[senders]
            )
            local_current_second = (
                jnp.swapaxes(current_frame, -1, -2) @ current_rotations[receivers]
            )
            first = _so3_log(
                local_current_first @ jnp.swapaxes(local_reference_first, -1, -2)
            )
            second = _so3_log(
                local_current_second @ jnp.swapaxes(local_reference_second, -1, -2)
            )
            bending_y, moment_y_first, moment_y_second = self._timoshenko_energy(
                first[:, 1],
                second[:, 1],
                properties["young"][indices] * properties["inertia_y"][indices],
                properties["shear"][indices] * properties["shear_area_z"][indices],
                lengths,
            )
            bending_z, moment_z_first, moment_z_second = self._timoshenko_energy(
                first[:, 2],
                second[:, 2],
                properties["young"][indices] * properties["inertia_z"][indices],
                properties["shear"][indices] * properties["shear_area_y"][indices],
                lengths,
            )
            twist = second[:, 0] - first[:, 0]
            torsion_stiffness = (
                properties["shear"][indices] * properties["torsion"][indices] / lengths
            )
            torsion_energy = 0.5 * torsion_stiffness * twist**2
            torsion = torsion_stiffness * twist
            bending = bending_y + bending_z + torsion_energy
            bending_moment = jnp.stack(
                (
                    0.5 * (moment_y_first + moment_y_second),
                    0.5 * (moment_z_first + moment_z_second),
                ),
                axis=-1,
            )
            shear = jnp.stack(
                (
                    (moment_z_first + moment_z_second) / lengths,
                    (moment_y_first + moment_y_second) / lengths,
                ),
                axis=-1,
            )
            frame_valid = (current_margin > self.director_tolerance) & (
                reference_margin > self.director_tolerance
            )
        return MemberBlockEvaluation(
            jnp.sum(axial.energy + bending),
            indices,
            axial.axial_force,
            shear,
            bending_moment,
            torsion,
            axial.active,
            jnp.full(
                axial.active.shape,
                isinstance(self.axial_law, TensionOnlyCableLaw),
                dtype=bool,
            ),
            axial.switching_margin,
            jnp.all(axial.valid & frame_valid),
        )


class DiscreteRodBlock(AbstractMemberBlock):
    """Stretching, discrete-curvature bending, and nodal-twist chain energy."""

    node_indices: Array
    axial_law: AbstractAxialLaw

    def __init__(
        self,
        node_indices: ArrayLike,
        member_indices: ArrayLike,
        /,
        *,
        axial_law: AbstractAxialLaw | None = None,
        block_id: str | None = None,
    ):
        nodes = jnp.asarray(node_indices, dtype=jnp.int32)
        members = jnp.asarray(member_indices, dtype=jnp.int32)
        if nodes.ndim != 1 or members.ndim != 1 or nodes.size != members.size + 1:
            raise ValueError(
                "A discrete rod requires one ordered node chain and its edges."
            )
        law = LinearAxialLaw() if axial_law is None else axial_law
        self.node_indices = nodes
        self.member_indices = members
        self.axial_law = law
        self.block_id = str(
            block_id
            or canonical_fingerprint(
                {
                    "kind": "discrete-rod-block",
                    "nodes": array_tree_fingerprint(nodes),
                    "members": array_tree_fingerprint(members),
                    "law": law.law_id,
                }
            )
        )

    @staticmethod
    def _curvature(points: Array, /) -> tuple[Array, Array]:
        edges = points[1:] - points[:-1]
        lengths = jnp.sqrt(jnp.sum(edges * edges, axis=-1))
        tangents = edges / jnp.maximum(lengths[:, None], jnp.finfo(points.dtype).tiny)
        cross = jnp.cross(tangents[:-1], tangents[1:])
        denominator = 1.0 + jnp.sum(tangents[:-1] * tangents[1:], axis=-1)
        curvature = (
            2.0 * cross / jnp.maximum(denominator[:, None], jnp.finfo(points.dtype).eps)
        )
        dual = 0.5 * (lengths[:-1] + lengths[1:])
        return curvature / dual[:, None], lengths

    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberBlockEvaluation:
        points = kinematics.positions[self.node_indices]
        reference_points = definition.reference.positions[self.node_indices]
        curvature, lengths = self._curvature(points)
        reference_curvature, _ = self._curvature(reference_points)
        indices = self.member_indices
        properties = definition.properties.structural_arrays()
        axial = _evaluate_axial_law(
            self.axial_law,
            lengths,
            definition.reference.rest_lengths[indices],
            properties["young"][indices] * properties["area"][indices],
            definition.reference.initial_strain[indices],
            jnp.zeros_like(lengths),
            definition.reference.cable_active[indices],
        )
        bending_rigidity = 0.5 * (
            properties["young"][indices[:-1]]
            * (
                properties["inertia_y"][indices[:-1]]
                + properties["inertia_z"][indices[:-1]]
            )
            + properties["young"][indices[1:]]
            * (
                properties["inertia_y"][indices[1:]]
                + properties["inertia_z"][indices[1:]]
            )
        )
        dual = 0.5 * (lengths[:-1] + lengths[1:])
        curvature_difference = curvature - reference_curvature
        bending_energy = (
            0.5 * bending_rigidity * dual * jnp.sum(curvature_difference**2, axis=-1)
        )
        if definition.structure.dimension == 3:
            rotations = kinematics.rotation_vectors[self.node_indices]
            reference_rotations = definition.reference.rotation_vectors[self.node_indices]
            edge_twist = 0.5 * jnp.sum(
                (rotations[1:] + rotations[:-1])
                * (points[1:] - points[:-1])
                / lengths[:, None],
                axis=-1,
            )
            reference_edge_twist = 0.5 * jnp.sum(
                (reference_rotations[1:] + reference_rotations[:-1])
                * (reference_points[1:] - reference_points[:-1])
                / definition.reference.rest_lengths[indices, None],
                axis=-1,
            )
            twist_difference = jnp.diff(edge_twist - reference_edge_twist)
            torsion_rigidity = 0.5 * (
                properties["shear"][indices[:-1]] * properties["torsion"][indices[:-1]]
                + properties["shear"][indices[1:]] * properties["torsion"][indices[1:]]
            )
            torsion_energy = 0.5 * torsion_rigidity / dual * twist_difference**2
            torsion_result = jnp.pad(torsion_rigidity / dual * twist_difference, (1, 0))
        else:
            torsion_energy = jnp.zeros_like(bending_energy)
            torsion_result = jnp.zeros_like(lengths)
        moment = jnp.pad(
            bending_rigidity[:, None] * curvature_difference,
            ((1, 0), (0, 0)),
        )
        return MemberBlockEvaluation(
            jnp.sum(axial.energy) + jnp.sum(bending_energy + torsion_energy),
            indices,
            axial.axial_force,
            jnp.zeros((indices.size, 2), dtype=points.dtype),
            moment,
            torsion_result,
            axial.active,
            jnp.full(
                axial.active.shape,
                isinstance(self.axial_law, TensionOnlyCableLaw),
                dtype=bool,
            ),
            axial.switching_margin,
            jnp.all(axial.valid)
            & jnp.all(jnp.isfinite(curvature))
            & jnp.all(
                1.0
                + jnp.sum(
                    (points[1:-1] - points[:-2]) * (points[2:] - points[1:-1]), axis=-1
                )
                > -1.0
            ),
        )


class HingeBendingBlock(AbstractMemberBlock):
    """Discrete surface hinge energy on explicit oriented vertex quadruples."""

    hinges: Array
    stiffness: Array
    rest_angle: Array

    def __init__(
        self,
        hinges: ArrayLike,
        stiffness: ArrayLike,
        rest_angle: ArrayLike,
        /,
        *,
        block_id: str | None = None,
    ):
        hinges_ = jnp.asarray(hinges, dtype=jnp.int32)
        stiffness_ = jnp.asarray(stiffness)
        rest_ = jnp.asarray(rest_angle, dtype=stiffness_.dtype)
        if hinges_.ndim != 2 or hinges_.shape[1] != 4:
            raise ValueError("hinges must have shape (count, 4).")
        if stiffness_.shape != (hinges_.shape[0],) or rest_.shape != stiffness_.shape:
            raise ValueError("Hinge stiffness/rest angles must match hinge count.")
        if bool(jnp.any(~jnp.isfinite(stiffness_) | (stiffness_ <= 0.0))):
            raise ValueError("Hinge stiffness must be finite and positive.")
        self.hinges = hinges_
        self.stiffness = stiffness_
        self.rest_angle = rest_
        self.member_indices = jnp.empty((0,), dtype=jnp.int32)
        self.block_id = str(
            block_id
            or canonical_fingerprint(
                {
                    "kind": "hinge-bending-block",
                    "hinges": array_tree_fingerprint(hinges_),
                    "stiffness": array_tree_fingerprint(stiffness_),
                    "rest": array_tree_fingerprint(rest_),
                }
            )
        )

    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberBlockEvaluation:
        first, second, left, right = (
            kinematics.positions[self.hinges[:, index]] for index in range(4)
        )
        edge = second - first
        edge_length = jnp.sqrt(jnp.sum(edge * edge, axis=-1))
        edge_unit = edge / jnp.maximum(edge_length[:, None], jnp.finfo(edge.dtype).tiny)
        normal_left = jnp.cross(second - first, left - first)
        normal_right = jnp.cross(right - first, second - first)
        left_norm = jnp.sqrt(jnp.sum(normal_left * normal_left, axis=-1))
        right_norm = jnp.sqrt(jnp.sum(normal_right * normal_right, axis=-1))
        normal_left = normal_left / jnp.maximum(
            left_norm[:, None], jnp.finfo(edge.dtype).tiny
        )
        normal_right = normal_right / jnp.maximum(
            right_norm[:, None], jnp.finfo(edge.dtype).tiny
        )
        sine = jnp.sum(edge_unit * jnp.cross(normal_left, normal_right), axis=-1)
        cosine = jnp.sum(normal_left * normal_right, axis=-1)
        angle = jnp.arctan2(sine, cosine)
        difference = _wrapped_angle(angle - self.rest_angle)
        energy = 0.5 * self.stiffness * difference**2
        count = self.hinges.shape[0]
        return MemberBlockEvaluation(
            jnp.sum(energy),
            self.member_indices,
            jnp.empty((0,), dtype=edge.dtype),
            jnp.empty((0, 2), dtype=edge.dtype),
            jnp.empty((0, 2), dtype=edge.dtype),
            jnp.empty((0,), dtype=edge.dtype),
            jnp.empty((0,), dtype=bool),
            jnp.empty((0,), dtype=bool),
            jnp.empty((0,), dtype=edge.dtype),
            jnp.all(edge_length > 0.0)
            & jnp.all(left_norm > 0.0)
            & jnp.all(right_norm > 0.0)
            & jnp.all(jnp.isfinite(energy))
            & (count >= 0),
        )


class MemberNetworkAssembly(StrictModule, NonTrainableState):
    """Static sum of homogeneous member and hinge mechanics blocks."""

    blocks: tuple[AbstractMemberBlock, ...]
    assembly_id: str = eqx.field(static=True)

    def __init__(self, blocks: Sequence[AbstractMemberBlock], /):
        blocks_ = tuple(blocks)
        if not blocks_ or any(
            not isinstance(block, AbstractMemberBlock) for block in blocks_
        ):
            raise TypeError("blocks must contain AbstractMemberBlock values.")
        occupied: list[int] = []
        for block in blocks_:
            occupied.extend(np.asarray(block.member_indices).tolist())
        if len(occupied) != len(set(occupied)):
            raise ValueError("A member may belong to only one constitutive member block.")
        self.blocks = blocks_
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "member-network-assembly",
                "blocks": [block.block_id for block in blocks_],
            }
        )

    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberNetworkAssemblyState:
        count = definition.structure.member_count
        dtype = kinematics.positions.dtype
        axial = jnp.zeros((count,), dtype=dtype)
        shear = jnp.zeros(
            (count, max(definition.structure.dimension - 1, 1)), dtype=dtype
        )
        moment = jnp.zeros_like(shear)
        torsion = jnp.zeros((count,), dtype=dtype)
        active = jnp.zeros((count,), dtype=bool)
        unilateral = jnp.zeros((count,), dtype=bool)
        margin = jnp.full((count,), jnp.inf, dtype=dtype)
        energy = jnp.asarray(0.0, dtype=dtype)
        valid = jnp.asarray(True)
        evaluations = []
        for block in self.blocks:
            evaluated = block.evaluate(definition, kinematics)
            evaluations.append(evaluated)
            energy = energy + evaluated.energy
            valid = valid & evaluated.valid
            indices = evaluated.member_indices
            axial = axial.at[indices].set(evaluated.axial_force)
            shear = shear.at[indices].set(evaluated.shear)
            moment = moment.at[indices].set(evaluated.bending_moment)
            torsion = torsion.at[indices].set(evaluated.torsion)
            active = active.at[indices].set(evaluated.active)
            unilateral = unilateral.at[indices].set(evaluated.unilateral)
            margin = margin.at[indices].set(evaluated.switching_margin)
        return MemberNetworkAssemblyState(
            energy,
            axial,
            shear,
            moment,
            torsion,
            active,
            unilateral,
            margin,
            valid,
            tuple(evaluations),
        )


class MemberNetworkAssemblyState(StrictModule):
    energy: Array
    axial_force: Array
    shear: Array
    bending_moment: Array
    torsion: Array
    active: Array
    unilateral: Array
    switching_margin: Array
    valid: Array
    blocks: tuple[MemberBlockEvaluation, ...]


__all__ = [
    "AbstractAxialLaw",
    "AbstractMemberBlock",
    "AxialConstitutiveState",
    "AxialMemberBlock",
    "CorotationalFrameBlock",
    "DiscreteRodBlock",
    "HingeBendingBlock",
    "LinearAxialLaw",
    "MemberBlockEvaluation",
    "MemberNetworkAssembly",
    "MemberNetworkAssemblyState",
    "TensionOnlyCableLaw",
]

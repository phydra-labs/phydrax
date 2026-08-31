#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from .._force_density_topology import ForceDensityStructure
from ._properties import MemberPropertyMap


def rotation_vector_matrix(rotation_vector: ArrayLike, /) -> Array:
    """Return SO(2)/SO(3) matrices from scalar or three-vector rotation charts."""
    value = jnp.asarray(rotation_vector)
    if value.shape[-1:] == (1,):
        angle = value[..., 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        return jnp.stack(
            (
                jnp.stack((cosine, -sine), axis=-1),
                jnp.stack((sine, cosine), axis=-1),
            ),
            axis=-2,
        )
    if value.shape[-1:] != (3,):
        raise ValueError("Rotation vectors must end in one or three coordinates.")
    angle = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    safe = jnp.maximum(angle, jnp.finfo(value.dtype).eps)
    axis = value / safe
    x, y, z = (axis[..., index] for index in range(3))
    zero = jnp.zeros_like(x)
    skew = jnp.stack(
        (
            jnp.stack((zero, -z, y), axis=-1),
            jnp.stack((z, zero, -x), axis=-1),
            jnp.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )
    identity = jnp.broadcast_to(jnp.eye(3, dtype=value.dtype), skew.shape)
    sine = jnp.sin(angle)[..., None]
    cosine = jnp.cos(angle)[..., None]
    matrix = identity + sine * skew + (1.0 - cosine) * (skew @ skew)
    near = angle[..., 0] <= jnp.sqrt(jnp.finfo(value.dtype).eps)
    return jnp.where(near[..., None, None], identity + skew, matrix)


class MemberReferenceState(StrictModule, NonTrainableState):
    """Stress-free geometry, orientation, curvature, and installation metadata."""

    positions: Array
    rest_lengths: Array
    rotation_vectors: Array
    rest_curvature: Array
    rest_twist: Array
    initial_strain: Array
    initial_temperature: Array
    installation_stage: Array
    cable_active: Array
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        /,
        *,
        rest_lengths: ArrayLike | None = None,
        rotation_vectors: ArrayLike | None = None,
        rest_curvature: ArrayLike | None = None,
        rest_twist: ArrayLike | None = None,
        initial_strain: ArrayLike | None = None,
        initial_temperature: ArrayLike | None = None,
        installation_stage: ArrayLike | None = None,
        cable_active: ArrayLike | None = None,
        reference_id: str | None = None,
    ):
        xyz = jnp.asarray(positions)
        expected = (structure.node_count, structure.dimension)
        if xyz.shape != expected or not jnp.issubdtype(xyz.dtype, jnp.inexact):
            raise TypeError(f"positions must be a real array with shape {expected}.")
        vectors = xyz[structure.receivers] - xyz[structure.senders]
        geometric_lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        lengths = geometric_lengths if rest_lengths is None else jnp.asarray(rest_lengths)
        if lengths.shape != (structure.member_count,):
            raise ValueError("rest_lengths must contain one value per member.")
        if bool(
            jnp.any(structure.member_valid & (~jnp.isfinite(lengths) | (lengths <= 0.0)))
        ):
            raise ValueError("Active rest lengths must be finite and positive.")
        rotation_dimension = 1 if structure.dimension == 2 else 3
        rotations = (
            jnp.zeros((structure.node_count, rotation_dimension), dtype=xyz.dtype)
            if rotation_vectors is None
            else jnp.asarray(rotation_vectors, dtype=xyz.dtype)
        )
        if rotations.shape != (structure.node_count, rotation_dimension):
            raise ValueError("rotation_vectors do not match node rotation coordinates.")
        curvature = (
            jnp.zeros((structure.member_count, rotation_dimension), dtype=xyz.dtype)
            if rest_curvature is None
            else jnp.asarray(rest_curvature, dtype=xyz.dtype)
        )
        twist = (
            jnp.zeros((structure.member_count,), dtype=xyz.dtype)
            if rest_twist is None
            else jnp.asarray(rest_twist, dtype=xyz.dtype)
        )
        strain = (
            jnp.zeros((structure.member_count,), dtype=xyz.dtype)
            if initial_strain is None
            else jnp.asarray(initial_strain, dtype=xyz.dtype)
        )
        temperature = (
            jnp.zeros((structure.member_count,), dtype=xyz.dtype)
            if initial_temperature is None
            else jnp.asarray(initial_temperature, dtype=xyz.dtype)
        )
        stage = (
            jnp.zeros((structure.member_count,), dtype=jnp.int32)
            if installation_stage is None
            else jnp.asarray(installation_stage, dtype=jnp.int32)
        )
        active = (
            jnp.ones((structure.member_count,), dtype=bool)
            if cable_active is None
            else jnp.asarray(cable_active, dtype=bool)
        )
        if (
            curvature.shape != (structure.member_count, rotation_dimension)
            or twist.shape != (structure.member_count,)
            or strain.shape != (structure.member_count,)
            or temperature.shape != (structure.member_count,)
            or stage.shape != (structure.member_count,)
            or active.shape != (structure.member_count,)
        ):
            raise ValueError("Member reference fields do not match member axes.")
        if bool(
            jnp.any(~jnp.isfinite(xyz))
            | jnp.any(~jnp.isfinite(rotations))
            | jnp.any(~jnp.isfinite(curvature))
            | jnp.any(~jnp.isfinite(twist))
            | jnp.any(~jnp.isfinite(strain))
            | jnp.any(~jnp.isfinite(temperature))
        ):
            raise ValueError("Reference state values must be finite.")
        self.positions = xyz
        self.rest_lengths = lengths.astype(xyz.dtype)
        self.rotation_vectors = rotations
        self.rest_curvature = curvature
        self.rest_twist = twist
        self.initial_strain = strain
        self.initial_temperature = temperature
        self.installation_stage = stage
        self.cable_active = active
        self.reference_id = str(
            reference_id
            or canonical_fingerprint(
                {
                    "kind": "member-reference-state",
                    "structure": structure.structure_id,
                    "arrays": array_tree_fingerprint(
                        (
                            xyz,
                            lengths,
                            rotations,
                            curvature,
                            twist,
                            strain,
                            temperature,
                            stage,
                            active,
                        )
                    ),
                }
            )
        )


class MemberDOFLayout(StrictModule, NonTrainableState):
    """Translation coordinates from ForceDensityStructure plus nodal rotations."""

    structure: ForceDensityStructure
    rotation_constrained: Array
    free_rotation_indices: Array
    constrained_rotation_indices: Array
    rotation_dimension: int = eqx.field(static=True)
    translation_size: int = eqx.field(static=True)
    rotation_size: int = eqx.field(static=True)
    reduced_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure: ForceDensityStructure,
        /,
        *,
        rotation_constrained: ArrayLike | None = None,
    ):
        if structure.dimension not in (2, 3):
            raise ValueError("Member networks currently require dimension two or three.")
        rotation_dimension = 1 if structure.dimension == 2 else 3
        constrained = (
            jnp.zeros((structure.node_count, rotation_dimension), dtype=bool)
            if rotation_constrained is None
            else jnp.asarray(rotation_constrained, dtype=bool)
        )
        if constrained.shape != (structure.node_count, rotation_dimension):
            raise ValueError("rotation_constrained has the wrong shape.")
        invalid = ~structure.node_valid[:, None]
        constrained = constrained | invalid
        flat = np.asarray(constrained).reshape((-1,))
        free = np.flatnonzero(~flat).astype(np.int32)
        fixed = np.flatnonzero(flat).astype(np.int32)
        translation_size = structure.free_dof_count
        rotation_size = free.size
        self.structure = structure
        self.rotation_constrained = constrained
        self.free_rotation_indices = jnp.asarray(free)
        self.constrained_rotation_indices = jnp.asarray(fixed)
        self.rotation_dimension = rotation_dimension
        self.translation_size = translation_size
        self.rotation_size = rotation_size
        self.reduced_size = translation_size + rotation_size
        self.layout_id = canonical_fingerprint(
            {
                "kind": "member-dof-layout",
                "structure": structure.structure_id,
                "rotation_constraints": array_tree_fingerprint(constrained),
            }
        )

    def reduce(
        self,
        positions: ArrayLike,
        rotation_vectors: ArrayLike,
        /,
    ) -> Array:
        rotation = jnp.asarray(rotation_vectors)
        expected = (self.structure.node_count, self.rotation_dimension)
        if rotation.shape != expected:
            raise ValueError(f"rotation_vectors must have shape {expected}.")
        translations = self.structure.reduce(positions)
        rotations = rotation.reshape((-1,))[self.free_rotation_indices]
        return jnp.concatenate((translations, rotations))

    def expand(
        self,
        reduced: ArrayLike,
        prescribed_positions: ArrayLike,
        prescribed_rotations: ArrayLike,
        /,
    ) -> MemberKinematics:
        value = jnp.asarray(reduced)
        if value.shape != (self.reduced_size,):
            raise ValueError("reduced member-network state has the wrong shape.")
        translations = value[: self.translation_size]
        rotations = value[self.translation_size :]
        positions = self.structure.expand(translations, prescribed_positions)
        prescribed = jnp.asarray(prescribed_rotations)
        if prescribed.shape != (self.constrained_rotation_indices.size,):
            raise ValueError("prescribed_rotations has the wrong shape.")
        full = jnp.zeros(
            (self.structure.node_count * self.rotation_dimension,), dtype=value.dtype
        )
        full = full.at[self.constrained_rotation_indices].set(
            prescribed, unique_indices=True
        )
        full = full.at[self.free_rotation_indices].set(rotations, unique_indices=True)
        rotation_vectors = full.reshape(
            (self.structure.node_count, self.rotation_dimension)
        )
        return MemberKinematics(positions, rotation_vectors)

    def prescribed_rotations(self, rotations: ArrayLike, /) -> Array:
        value = jnp.asarray(rotations)
        expected = (self.structure.node_count, self.rotation_dimension)
        if value.shape != expected:
            raise ValueError(f"rotations must have shape {expected}.")
        return value.reshape((-1,))[self.constrained_rotation_indices]


class MemberKinematics(StrictModule):
    """Current nodal positions and rotation-chart coordinates."""

    positions: Array
    rotation_vectors: Array

    @property
    def rotation_matrices(self) -> Array:
        return rotation_vector_matrix(self.rotation_vectors)


class MemberNetworkDefinition(StrictModule, NonTrainableState):
    """Fixed member topology, reference state, properties, and DOF layout."""

    structure: ForceDensityStructure
    reference: MemberReferenceState
    properties: MemberPropertyMap
    dofs: MemberDOFLayout
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure: ForceDensityStructure,
        reference: MemberReferenceState,
        properties: MemberPropertyMap,
        dofs: MemberDOFLayout,
        /,
    ):
        if properties.member_count != structure.member_count:
            raise ValueError("Member properties must match the structure member count.")
        if dofs.structure.structure_id != structure.structure_id:
            raise ValueError("DOF layout and structure identities do not match.")
        self.structure = structure
        self.reference = reference
        self.properties = properties
        self.dofs = dofs
        self.definition_id = canonical_fingerprint(
            {
                "kind": "member-network-definition",
                "structure": structure.structure_id,
                "reference": reference.reference_id,
                "properties": properties.mapping_id,
                "dofs": dofs.layout_id,
            }
        )


__all__ = [
    "MemberDOFLayout",
    "MemberKinematics",
    "MemberNetworkDefinition",
    "MemberReferenceState",
    "rotation_vector_matrix",
]

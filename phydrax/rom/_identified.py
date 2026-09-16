#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import ContinuousSystem, DiscreteSystem, StateLayout, TrajectoryData
from ._basis import ReducedBasisArtifact


def _encode_flat(basis: ReducedBasisArtifact, values: Array, offset: Array, /) -> Array:
    if not basis.subspace.orthonormal:
        raise ValueError("Reduced trajectory projection requires an orthonormal basis.")
    centered = values - offset
    columns = basis.basis_matrix
    space = basis.subspace.space

    def encode_one(value):
        vector = space.unflatten(value)
        return jax.vmap(
            lambda column: space.inner(space.unflatten(column), vector),
            in_axes=1,
        )(columns)

    flattened = centered.reshape((-1, space.size))
    encoded = jax.vmap(encode_one)(flattened)
    return encoded.reshape(centered.shape[:-1] + (basis.rank,))


def project_trajectory_data(
    data: TrajectoryData,
    basis: ReducedBasisArtifact,
    offset: ArrayLike,
    /,
    *,
    source_id: str,
) -> TrajectoryData:
    """Project already-partitioned trajectories into one linear reduced chart."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if not isinstance(basis, ReducedBasisArtifact) or basis.role != "state":
        raise TypeError("basis must be a state ReducedBasisArtifact.")
    if data.state_layout.size != basis.subspace.space.size:
        raise ValueError("Trajectory state layout and reduced basis size do not match.")
    offset_ = jnp.asarray(offset, dtype=data.states.dtype).reshape((-1,))
    if offset_.shape != (basis.subspace.space.size,):
        raise ValueError("offset must flatten to the basis full-space size.")
    state_flat = data.states.reshape(data.case_shape + (data.capacity, -1))
    reduced_states = _encode_flat(basis, state_flat, offset_)
    reduced_derivatives = (
        None
        if data.derivatives is None
        else _encode_flat(
            basis,
            data.derivatives.reshape(data.case_shape + (data.capacity, -1)),
            jnp.zeros_like(offset_),
        )
    )
    layout = StateLayout(
        (basis.rank,),
        component_names=tuple(f"z{index}" for index in range(basis.rank)),
        layout_id=canonical_fingerprint(
            {
                "kind": "reduced-state-layout",
                "basis": basis.artifact_id,
                "source_layout": data.state_layout.layout_id,
            }
        ),
    )
    return TrajectoryData(
        data.coordinates,
        reduced_states,
        state_layout=layout,
        sample_valid=data.sample_valid,
        transition_valid=data.transition_valid,
        reset_mask=data.reset_mask,
        weights=data.weights,
        inputs=data.inputs,
        input_layout=data.input_layout,
        input_valid=data.input_valid,
        input_alignment=(
            "transitions" if data.input_alignment is None else data.input_alignment
        ),
        derivatives=reduced_derivatives,
        derivative_valid=data.derivative_valid,
        case_axes=data.case_axes,
        case_axis_roles=data.case_axis_roles,
        coordinate_id=data.coordinate_id,
        coordinate_kind=data.coordinate_kind,
        source_id=source_id,
        dataset_id=canonical_fingerprint(
            {
                "kind": "projected-trajectory-data",
                "source": data.dataset_id,
                "basis": basis.artifact_id,
                "offset": array_tree_fingerprint(offset_)["sha256"],
            }
        ),
    )


class IdentifiedReducedDynamics(StrictModule, NonTrainableState):
    """Linear representation composed with one executable identified system."""

    basis: ReducedBasisArtifact
    offset: Array
    system: ContinuousSystem | DiscreteSystem
    identification_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: ReducedBasisArtifact,
        offset: ArrayLike,
        system: ContinuousSystem | DiscreteSystem,
        /,
        *,
        identification_id: str,
        partition_id: str,
    ):
        if not isinstance(basis, ReducedBasisArtifact) or basis.role != "state":
            raise TypeError("basis must be a state ReducedBasisArtifact.")
        if not isinstance(system, (ContinuousSystem, DiscreteSystem)):
            raise TypeError("system must be ContinuousSystem or DiscreteSystem.")
        if system.state_layout.size != basis.rank:
            raise ValueError(
                "Identified system state size must equal reduced basis rank."
            )
        offset_ = jnp.asarray(offset, dtype=basis.basis_matrix.dtype).reshape((-1,))
        if offset_.shape != (basis.subspace.space.size,):
            raise ValueError("offset must flatten to the basis full-space size.")
        identification = str(identification_id)
        partition = str(partition_id)
        if not identification or not partition:
            raise ValueError("identification_id and partition_id must be non-empty.")
        self.basis = basis
        self.offset = offset_
        self.system = system
        self.identification_id = identification
        self.partition_id = partition
        self.model_id = canonical_fingerprint(
            {
                "kind": "identified-reduced-dynamics",
                "basis": basis.artifact_id,
                "offset": array_tree_fingerprint(offset_)["sha256"],
                "system": system.system_id,
                "identification": identification,
                "partition": partition,
            }
        )

    def encode(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state, dtype=self.offset.dtype).reshape((-1,))
        if value.shape != self.offset.shape:
            raise ValueError("state must flatten to the basis full-space size.")
        return _encode_flat(self.basis, value, self.offset)

    def decode(self, reduced: ArrayLike, /) -> Array:
        value = jnp.asarray(reduced, dtype=self.offset.dtype)
        if value.shape[-1:] != (self.basis.rank,):
            raise ValueError("reduced values must end in the reduced-state axis.")
        reconstructed = contract("...r,nr->...n", value, self.basis.basis_matrix)
        return reconstructed + self.offset


__all__ = [
    "IdentifiedReducedDynamics",
    "project_trajectory_data",
]

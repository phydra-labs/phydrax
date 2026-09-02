#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._abelian import AbelianLeg, AbelianTensor, AbelianTensorLayout
from ._abelian_core import (
    AbelianMatrixProductState,
    canonicalize_abelian_mps,
)
from ._precision import TensorNetworkPrecisionPolicy


class AbelianTensorTruncationEvidence(StrictModule):
    retained_rank: int = eqx.field(static=True)
    available_rank: int = eqx.field(static=True)
    per_sector_retained_ranks: Array
    selected_modes: Array
    discarded_weight: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)


class AbelianTEBDEvidence(StrictModule):
    discarded_weights: Array
    cumulative_discarded_weight: Array
    norm_residual: Array
    valid: Array
    trotter_order: int = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)


def _offsets(leg: AbelianLeg):
    starts = []
    start = 0
    for capacity in leg.capacities:
        starts.append(start)
        start += capacity
    return tuple(starts)


def _gate_conservation_residual(gate, left_leg, right_leg):
    left_offsets = _offsets(left_leg)
    right_offsets = _offsets(right_leg)
    residual = jnp.asarray(0.0, dtype=gate.real.dtype)
    group = left_leg.group
    for left_out, left_out_charge in enumerate(left_leg.charges):
        for right_out, right_out_charge in enumerate(right_leg.charges):
            output_charge = group.add(left_out_charge, right_out_charge)
            for left_in, left_in_charge in enumerate(left_leg.charges):
                for right_in, right_in_charge in enumerate(right_leg.charges):
                    if group.add(left_in_charge, right_in_charge) == output_charge:
                        continue
                    block = gate[
                        left_offsets[left_out] : left_offsets[left_out]
                        + left_leg.capacities[left_out],
                        right_offsets[right_out] : right_offsets[right_out]
                        + right_leg.capacities[right_out],
                        left_offsets[left_in] : left_offsets[left_in]
                        + left_leg.capacities[left_in],
                        right_offsets[right_in] : right_offsets[right_in]
                        + right_leg.capacities[right_in],
                    ]
                    residual = jnp.maximum(residual, jnp.max(jnp.abs(block)))
    return residual


def apply_abelian_two_site_gate(
    state: AbelianMatrixProductState,
    left_site: int,
    gate: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = True,
    conservation_tolerance: float = 1e-10,
) -> tuple[AbelianMatrixProductState, AbelianTensorTruncationEvidence]:
    if not isinstance(state, AbelianMatrixProductState):
        raise TypeError("state must be AbelianMatrixProductState.")
    site = int(left_site)
    if not 0 <= site < state.site_count - 1:
        raise ValueError("Two-site gate index is outside the Abelian MPS.")
    capacity = int(maximum_bond_dimension)
    if capacity < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    precision = state.precision
    left_tensor = state.tensors[site]
    right_tensor = state.tensors[site + 1]
    left_physical = left_tensor.layout.legs[1]
    middle = left_tensor.layout.legs[2]
    right_physical = right_tensor.layout.legs[1]
    gate_ = precision.contraction(jnp.asarray(gate))
    expected = (
        left_physical.size,
        right_physical.size,
        left_physical.size,
        right_physical.size,
    )
    if gate_.shape != expected:
        raise ValueError("Abelian two-site gate shape is invalid.")
    gate_ = eqx.error_if(
        gate_,
        _gate_conservation_residual(gate_, left_physical, right_physical)
        > float(conservation_tolerance),
        "Two-site gate violates the declared Abelian charge conservation.",
    )
    left_dense = precision.contraction(left_tensor.to_dense())
    right_dense = precision.contraction(right_tensor.to_dense())
    theta = oe.contract("lpi,iqr->lpqr", left_dense, right_dense)
    theta = oe.contract("abij,lijr->labr", gate_, theta)
    left_virtual = left_tensor.layout.legs[0]
    right_virtual = right_tensor.layout.legs[2]
    left_offsets = _offsets(left_virtual)
    left_physical_offsets = _offsets(left_physical)
    right_physical_offsets = _offsets(right_physical)
    right_offsets = _offsets(right_virtual)
    matrix_dense = precision.factorization(
        theta.reshape(
            (
                left_virtual.size * left_physical.size,
                right_physical.size * right_virtual.size,
            )
        )
    )
    decompositions = []
    all_singular_values = []
    total_available = 0
    for middle_sector, middle_charge in enumerate(middle.charges):
        row_routes = []
        row_indices = []
        for left_sector, left_charge in enumerate(left_virtual.charges):
            for physical_sector, physical_charge in enumerate(left_physical.charges):
                if left_virtual.group.add(left_charge, physical_charge) != middle_charge:
                    continue
                indices = [
                    (left_offsets[left_sector] + left_index) * left_physical.size
                    + left_physical_offsets[physical_sector]
                    + physical_index
                    for left_index in range(left_virtual.capacities[left_sector])
                    for physical_index in range(left_physical.capacities[physical_sector])
                ]
                row_routes.append((left_sector, physical_sector, len(indices)))
                row_indices.extend(indices)
        column_routes = []
        column_indices = []
        for physical_sector, physical_charge in enumerate(right_physical.charges):
            for right_sector, right_charge in enumerate(right_virtual.charges):
                if left_virtual.group.add(middle_charge, physical_charge) != right_charge:
                    continue
                indices = [
                    (right_physical_offsets[physical_sector] + physical_index)
                    * right_virtual.size
                    + right_offsets[right_sector]
                    + right_index
                    for physical_index in range(
                        right_physical.capacities[physical_sector]
                    )
                    for right_index in range(right_virtual.capacities[right_sector])
                ]
                column_routes.append((physical_sector, right_sector, len(indices)))
                column_indices.extend(indices)
        sector_matrix = matrix_dense[
            jnp.ix_(jnp.asarray(row_indices), jnp.asarray(column_indices))
        ]
        u, singular_values, vh = jnp.linalg.svd(sector_matrix, full_matrices=False)
        sector_capacity = middle.capacities[middle_sector]
        available = min(int(singular_values.shape[0]), sector_capacity)
        u_pad = jnp.zeros((sector_matrix.shape[0], sector_capacity), dtype=u.dtype)
        vh_pad = jnp.zeros((sector_capacity, sector_matrix.shape[1]), dtype=vh.dtype)
        s_pad = jnp.zeros((sector_capacity,), dtype=singular_values.dtype)
        u_pad = u_pad.at[:, :available].set(u[:, :available])
        vh_pad = vh_pad.at[:available, :].set(vh[:available, :])
        s_pad = s_pad.at[:available].set(singular_values[:available])
        decompositions.append((row_routes, column_routes, u_pad, s_pad, vh_pad))
        all_singular_values.append(s_pad)
        total_available += available
    spectrum = jnp.concatenate(all_singular_values)
    retained = min(capacity, total_available)
    order = jnp.argsort(-jnp.abs(spectrum), stable=True)
    selected = jnp.zeros(spectrum.shape, dtype=bool).at[order[:retained]].set(True)
    discarded = precision.decision(
        precision.sum(jnp.where(selected, 0.0, jnp.abs(spectrum) ** 2))
    )
    left_blocks = {
        sector: jnp.zeros(shape, dtype=theta.dtype)
        for sector, shape in zip(
            left_tensor.layout.sectors, left_tensor.layout.block_shapes, strict=True
        )
    }
    right_blocks = {
        sector: jnp.zeros(shape, dtype=theta.dtype)
        for sector, shape in zip(
            right_tensor.layout.sectors, right_tensor.layout.block_shapes, strict=True
        )
    }
    retained_per_sector = []
    cursor_spectrum = 0
    for middle_sector, decomposition in enumerate(decompositions):
        row_routes, column_routes, u, singular_values, vh = decomposition
        sector_capacity = middle.capacities[middle_sector]
        sector_mask = selected[cursor_spectrum : cursor_spectrum + sector_capacity]
        retained_per_sector.append(jnp.sum(sector_mask.astype(jnp.int32)))
        u = u * sector_mask[None, :]
        weighted = (singular_values * sector_mask)[:, None] * vh
        cursor = 0
        for left_sector, physical_sector, size in row_routes:
            shape = left_blocks[(left_sector, physical_sector, middle_sector)].shape
            left_blocks[(left_sector, physical_sector, middle_sector)] = u[
                cursor : cursor + size
            ].reshape(shape)
            cursor += size
        cursor = 0
        for physical_sector, right_sector, size in column_routes:
            shape = right_blocks[(middle_sector, physical_sector, right_sector)].shape
            right_blocks[(middle_sector, physical_sector, right_sector)] = weighted[
                :, cursor : cursor + size
            ].reshape(shape)
            cursor += size
        cursor_spectrum += sector_capacity
    active_counts = jnp.stack(retained_per_sector)
    new_middle_left = middle.with_active(active_counts)
    new_middle_right = new_middle_left.dual()
    left_legs = list(left_tensor.layout.legs)
    left_legs[2] = new_middle_left
    right_legs = list(right_tensor.layout.legs)
    right_legs[0] = new_middle_right
    left_layout = AbelianTensorLayout(
        tuple(left_legs), total_charge=left_tensor.layout.total_charge
    )
    right_layout = AbelianTensorLayout(
        tuple(right_legs), total_charge=right_tensor.layout.total_charge
    )
    new_left = AbelianTensor(
        left_layout,
        tuple(left_blocks[sector] for sector in left_layout.sectors),
        precision=precision,
    )
    new_right = AbelianTensor(
        right_layout,
        tuple(right_blocks[sector] for sector in right_layout.sectors),
        precision=precision,
    )
    tensors = list(state.tensors)
    tensors[site] = new_left
    tensors[site + 1] = new_right
    result = AbelianMatrixProductState(tuple(tensors))
    if normalize:
        result = result.normalized()
    precision_evidence = precision.evidence_for(
        tuple(block for tensor in state.tensors for block in tensor.blocks),
        children={"input-state": state.precision_evidence},
        output_value=tuple(block for tensor in result.tensors for block in tensor.blocks),
    )
    evidence = AbelianTensorTruncationEvidence(
        retained,
        total_available,
        active_counts,
        selected,
        discarded,
        jnp.isfinite(discarded) & (discarded >= 0.0),
        precision_evidence,
        precision.policy_id,
    )
    return result, evidence


class AbelianNearestNeighborHamiltonian(StrictModule):
    terms: tuple[Array, ...]
    physical_legs: tuple[AbelianLeg, ...]
    hamiltonian_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[ArrayLike],
        physical_legs: Sequence[AbelianLeg],
        /,
        *,
        hamiltonian_id: str,
        conservation_tolerance: float = 1e-10,
    ):
        legs = tuple(physical_legs)
        values = tuple(jnp.asarray(term) for term in terms)
        if len(values) != len(legs) - 1 or not legs:
            raise ValueError("One Abelian two-site term is required per bond.")
        if any(leg.orientation != 1 for leg in legs):
            raise ValueError(
                "Abelian Hamiltonian physical legs must be outward oriented."
            )
        for index, term in enumerate(values):
            size = legs[index].size * legs[index + 1].size
            if term.shape != (size, size):
                raise ValueError("Abelian Hamiltonian term shape is invalid.")
            tensor = term.reshape((legs[index].size, legs[index + 1].size) * 2)
            residual = _gate_conservation_residual(tensor, legs[index], legs[index + 1])
            if not bool(
                jnp.all(jnp.isfinite(term))
                & jnp.allclose(term, jnp.conj(term.T), rtol=1e-10, atol=1e-12)
                & (residual <= conservation_tolerance)
            ):
                raise ValueError(
                    "Abelian Hamiltonian terms must be finite, Hermitian, and conserving."
                )
        identifier = str(hamiltonian_id)
        if not identifier:
            raise ValueError("hamiltonian_id must be nonempty.")
        self.terms = values
        self.physical_legs = legs
        self.hamiltonian_id = identifier

    def gate(self, bond: int, step: ArrayLike, /) -> Array:
        index = int(bond)
        left = self.physical_legs[index].size
        right = self.physical_legs[index + 1].size
        return jsp.linalg.expm(-1j * jnp.asarray(step) * self.terms[index]).reshape(
            (left, right, left, right)
        )


def abelian_tebd_step(
    state: AbelianMatrixProductState,
    hamiltonian: AbelianNearestNeighborHamiltonian,
    step_size: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
    order: int = 2,
    normalize: bool = True,
) -> tuple[AbelianMatrixProductState, AbelianTEBDEvidence]:
    if tuple(tensor.layout.legs[1].leg_id for tensor in state.tensors) != tuple(
        leg.leg_id for leg in hamiltonian.physical_legs
    ):
        raise ValueError("Abelian MPS and Hamiltonian physical legs differ.")
    if order not in (1, 2):
        raise ValueError("Abelian TEBD order must be one or two.")
    step = jnp.asarray(step_size, dtype=state.tensors[0].blocks[0].real.dtype)
    step = eqx.error_if(
        step,
        (~jnp.isfinite(step)) | (step < 0.0),
        "Abelian TEBD step must be finite and nonnegative.",
    )
    current = state
    records = []

    def layer(value, bonds, local_step):
        local_records = []
        for bond in bonds:
            value, evidence = apply_abelian_two_site_gate(
                value,
                bond,
                hamiltonian.gate(bond, local_step),
                maximum_bond_dimension=maximum_bond_dimension,
                normalize=normalize,
            )
            local_records.append(evidence)
        return value, local_records

    layers = (
        (
            (range(0, state.site_count - 1, 2), step),
            (range(1, state.site_count - 1, 2), step),
        )
        if order == 1
        else (
            (range(0, state.site_count - 1, 2), 0.5 * step),
            (range(1, state.site_count - 1, 2), step),
            (range(0, state.site_count - 1, 2), 0.5 * step),
        )
    )
    for bonds, local_step in layers:
        current, local = layer(current, bonds, local_step)
        records.extend(local)
    current = canonicalize_abelian_mps(
        current, center=state.site_count // 2, normalize=normalize
    )
    weights = (
        jnp.stack([record.discarded_weight for record in records])
        if records
        else jnp.zeros((0,), dtype=step.dtype)
    )
    cumulative = jnp.sum(weights)
    norm_residual = jnp.abs(current.norm() - 1.0)
    return current, AbelianTEBDEvidence(
        weights,
        cumulative,
        norm_residual,
        jnp.all(jnp.isfinite(weights)) & jnp.isfinite(norm_residual),
        order,
        state.precision.policy_id,
    )


def abelian_product_mps(
    local_states: Sequence[ArrayLike],
    physical_legs: Sequence[AbelianLeg],
    local_charge_ordinals: Sequence[int],
    /,
    *,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> AbelianMatrixProductState:
    values = tuple(jnp.asarray(value) for value in local_states)
    legs = tuple(physical_legs)
    ordinals = tuple(int(value) for value in local_charge_ordinals)
    if not values or len(values) != len(legs) or len(values) != len(ordinals):
        raise ValueError(
            "Product-state values, physical legs, and charge ordinals must align."
        )
    group = legs[0].group
    precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
    cumulative = group.zero
    tensors = []
    for value, physical, ordinal in zip(values, legs, ordinals, strict=True):
        if physical.orientation != 1 or physical.group.group_id != group.group_id:
            raise ValueError("Product-state physical legs are incompatible.")
        if value.shape != (physical.size,) or not 0 <= ordinal < len(physical.charges):
            raise ValueError("Product-state value or charge ordinal is invalid.")
        left_charge = cumulative
        cumulative = group.add(cumulative, physical.charges[ordinal])
        left_leg = AbelianLeg(group, (left_charge,), (1,), orientation=1)
        right_leg = AbelianLeg(group, (cumulative,), (1,), orientation=-1)
        layout = AbelianTensorLayout((left_leg, physical, right_leg))
        dense = value[None, :, None]
        tensors.append(AbelianTensor.from_dense(layout, dense, precision=precision_))
    return AbelianMatrixProductState(tuple(tensors))


__all__ = [
    "AbelianNearestNeighborHamiltonian",
    "AbelianTEBDEvidence",
    "AbelianTensorTruncationEvidence",
    "abelian_product_mps",
    "abelian_tebd_step",
    "apply_abelian_two_site_gate",
]

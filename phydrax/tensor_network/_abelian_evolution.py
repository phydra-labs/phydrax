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
    abelian_mps_inner,
    AbelianMatrixProductOperator,
    AbelianMatrixProductState,
    add_abelian_mps,
    apply_abelian_mpo,
    canonicalize_abelian_mps,
    compress_abelian_mps,
    scale_abelian_mps,
)
from ._precision import TensorNetworkPrecisionPolicy


class AbelianTensorTruncationEvidence(StrictModule):
    retained_rank: int = eqx.field(static=True)
    available_rank: int = eqx.field(static=True)
    per_sector_retained_ranks: Array
    selected_modes: Array
    discarded_weight: Array
    overflow_discarded_weight: Array
    protected_sectors_satisfied: Array
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
    protected_charges: Sequence[Sequence[int]] = (),
    prepared_routes: tuple[
        tuple[
            tuple[tuple[int, int, int], ...],
            tuple[tuple[int, int, int], ...],
        ],
        ...,
    ]
    | None = None,
) -> tuple[AbelianMatrixProductState, AbelianTensorTruncationEvidence]:
    """Apply a conserving gate through sector blocks and truncate globally."""

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
    protected = tuple(
        left_physical.group.normalize(charge) for charge in protected_charges
    )
    if len(set(protected)) != len(protected):
        raise ValueError("Protected Abelian charges must be unique.")
    left_virtual = left_tensor.layout.legs[0]
    right_virtual = right_tensor.layout.legs[2]
    left_physical_offsets = _offsets(left_physical)
    right_physical_offsets = _offsets(right_physical)
    decompositions = []
    retained_spectra = []
    overflow_weight = jnp.asarray(0.0, dtype=gate_.real.dtype)
    total_available = 0
    group = middle.group
    if prepared_routes is not None and len(prepared_routes) != len(middle.charges):
        raise ValueError("Prepared Abelian gate routes do not match the bond.")
    for middle_sector, middle_charge in enumerate(middle.charges):
        if prepared_routes is None:
            row_routes = tuple(
                (
                    left_sector,
                    physical_sector,
                    left_virtual.capacities[left_sector]
                    * left_physical.capacities[physical_sector],
                )
                for left_sector, left_charge in enumerate(left_virtual.charges)
                for physical_sector, physical_charge in enumerate(left_physical.charges)
                if group.add(left_charge, physical_charge) == middle_charge
            )
            column_routes = tuple(
                (
                    physical_sector,
                    right_sector,
                    right_physical.capacities[physical_sector]
                    * right_virtual.capacities[right_sector],
                )
                for physical_sector, physical_charge in enumerate(right_physical.charges)
                for right_sector, right_charge in enumerate(right_virtual.charges)
                if group.add(middle_charge, physical_charge) == right_charge
            )
        else:
            row_routes, column_routes = prepared_routes[middle_sector]
        row_count = sum(route[2] for route in row_routes)
        column_count = sum(route[2] for route in column_routes)
        sector_matrix = jnp.zeros(
            (row_count, column_count), dtype=jnp.result_type(gate_, left_tensor.blocks[0])
        )
        row_cursor = 0
        for left_out_sector, left_out_physical, row_size in row_routes:
            column_cursor = 0
            for right_out_physical, right_out_sector, column_size in column_routes:
                contribution = jnp.zeros(
                    (row_size, column_size), dtype=sector_matrix.dtype
                )
                for left_index, left_source_sector in enumerate(
                    left_tensor.layout.sectors
                ):
                    if (
                        left_source_sector[0] != left_out_sector
                        or left_source_sector[2] != middle_sector
                    ):
                        continue
                    for right_index, right_source_sector in enumerate(
                        right_tensor.layout.sectors
                    ):
                        if (
                            right_source_sector[0] != middle_sector
                            or right_source_sector[2] != right_out_sector
                        ):
                            continue
                        left_in_physical = left_source_sector[1]
                        right_in_physical = right_source_sector[1]
                        gate_block = gate_[
                            left_physical_offsets[
                                left_out_physical
                            ] : left_physical_offsets[left_out_physical]
                            + left_physical.capacities[left_out_physical],
                            right_physical_offsets[
                                right_out_physical
                            ] : right_physical_offsets[right_out_physical]
                            + right_physical.capacities[right_out_physical],
                            left_physical_offsets[
                                left_in_physical
                            ] : left_physical_offsets[left_in_physical]
                            + left_physical.capacities[left_in_physical],
                            right_physical_offsets[
                                right_in_physical
                            ] : right_physical_offsets[right_in_physical]
                            + right_physical.capacities[right_in_physical],
                        ]
                        local = oe.contract(
                            "abij,lix,xjr->labr",
                            gate_block,
                            precision.contraction(left_tensor.blocks[left_index]),
                            precision.contraction(right_tensor.blocks[right_index]),
                        ).reshape((row_size, column_size))
                        contribution = contribution + local
                sector_matrix = sector_matrix.at[
                    row_cursor : row_cursor + row_size,
                    column_cursor : column_cursor + column_size,
                ].set(contribution)
                column_cursor += column_size
            row_cursor += row_size
        u, singular_values, vh = jnp.linalg.svd(
            precision.factorization(sector_matrix), full_matrices=False
        )
        sector_capacity = middle.capacities[middle_sector]
        stored = min(int(singular_values.shape[0]), sector_capacity)
        u_pad = jnp.zeros((row_count, sector_capacity), dtype=u.dtype)
        vh_pad = jnp.zeros((sector_capacity, column_count), dtype=vh.dtype)
        s_pad = jnp.zeros((sector_capacity,), dtype=singular_values.dtype)
        u_pad = u_pad.at[:, :stored].set(u[:, :stored])
        vh_pad = vh_pad.at[:stored, :].set(vh[:stored, :])
        s_pad = s_pad.at[:stored].set(singular_values[:stored])
        overflow_weight = overflow_weight + precision.sum(
            jnp.abs(singular_values[stored:]) ** 2
        )
        decompositions.append((row_routes, column_routes, u_pad, s_pad, vh_pad))
        retained_spectra.append(s_pad)
        total_available += int(singular_values.shape[0])
    spectrum = jnp.concatenate(retained_spectra)
    retained_limit = min(capacity, int(spectrum.shape[0]))
    selected = jnp.zeros(spectrum.shape, dtype=bool)
    sector_starts = []
    cursor = 0
    protected_selected = 0
    for sector, charge in enumerate(middle.charges):
        sector_starts.append(cursor)
        if charge in protected and middle.capacities[sector] > 0:
            local = retained_spectra[sector]
            local_index = jnp.argmax(jnp.abs(local))
            if protected_selected < retained_limit:
                selected = selected.at[cursor + local_index].set(True)
                protected_selected += 1
        cursor += middle.capacities[sector]
    order = jnp.argsort(-jnp.abs(spectrum), stable=True)
    remaining = retained_limit - protected_selected
    positions = jnp.nonzero(
        ~selected[order],
        size=spectrum.shape[0],
        fill_value=spectrum.shape[0],
    )[0]
    safe_positions = jnp.minimum(positions[:remaining], spectrum.shape[0] - 1)
    chosen = order[safe_positions]
    selected = selected.at[chosen].set(positions[:remaining] < spectrum.shape[0])
    retained = retained_limit
    discarded_within = precision.sum(jnp.where(selected, 0.0, jnp.abs(spectrum) ** 2))
    discarded = precision.decision(discarded_within + overflow_weight)
    left_blocks = [
        jnp.zeros(shape, dtype=gate_.dtype) for shape in left_tensor.layout.block_shapes
    ]
    right_blocks = [
        jnp.zeros(shape, dtype=gate_.dtype) for shape in right_tensor.layout.block_shapes
    ]
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
            key = (left_sector, physical_sector, middle_sector)
            if key in left_tensor.layout.sectors:
                index = left_tensor.layout.sectors.index(key)
                left_blocks[index] = u[cursor : cursor + size].reshape(
                    left_blocks[index].shape
                )
            cursor += size
        cursor = 0
        for physical_sector, right_sector, size in column_routes:
            key = (middle_sector, physical_sector, right_sector)
            if key in right_tensor.layout.sectors:
                index = right_tensor.layout.sectors.index(key)
                right_blocks[index] = weighted[:, cursor : cursor + size].reshape(
                    right_blocks[index].shape
                )
            cursor += size
        cursor_spectrum += sector_capacity
    active_counts = jnp.stack(retained_per_sector)
    protected_checks = tuple(
        (
            active_counts[middle.charges.index(charge)] > 0
            if charge in middle.charges
            else jnp.asarray(False)
        )
        for charge in protected
    )
    protected_satisfied = (
        jnp.all(jnp.stack(protected_checks)) if protected_checks else jnp.asarray(True)
    )
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
    new_left = AbelianTensor(left_layout, tuple(left_blocks), precision=precision)
    new_right = AbelianTensor(right_layout, tuple(right_blocks), precision=precision)
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
        precision.decision(overflow_weight),
        protected_satisfied,
        jnp.isfinite(discarded) & (discarded >= 0.0) & protected_satisfied,
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


class AbelianDMRGEvidence(StrictModule):
    energies: Array
    residual_norms: Array
    discarded_weights: Array
    charge_drifts: Array
    protected_sectors_satisfied: Array
    converged: Array
    valid: Array
    sweep_count: int = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)


class AbelianTDVPEvidence(StrictModule):
    times: Array
    norm_residuals: Array
    energy_values: Array
    discarded_weights: Array
    charge_drifts: Array
    protected_sectors_satisfied: Array
    valid: Array
    step_count: int = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)


def _abelian_energy(
    state: AbelianMatrixProductState,
    hamiltonian: AbelianMatrixProductOperator,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[Array, AbelianMatrixProductState, Array]:
    applied, compression = apply_abelian_mpo(
        hamiltonian,
        state,
        maximum_bond_dimension=maximum_bond_dimension,
        normalize=False,
    )
    denominator = abelian_mps_inner(state, state)
    energy = abelian_mps_inner(state, applied) / denominator
    return jnp.real(energy), applied, compression.accumulated_discarded_weight


def abelian_finite_dmrg(
    initial_state: AbelianMatrixProductState,
    hamiltonian: AbelianMatrixProductOperator,
    /,
    *,
    maximum_sweeps: int,
    maximum_bond_dimension: int,
    descent_step: ArrayLike,
    residual_tolerance: float = 1e-8,
    protected_charges: Sequence[Sequence[int]] = (),
) -> tuple[AbelianMatrixProductState, AbelianDMRGEvidence]:
    """Bounded variational residual sweeps in the fixed total-charge sector."""

    sweeps = int(maximum_sweeps)
    if sweeps < 1:
        raise ValueError("maximum_sweeps must be positive.")
    capacity = int(maximum_bond_dimension)
    if capacity < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    step = jnp.asarray(descent_step, dtype=initial_state.norm().dtype)
    step = eqx.error_if(
        step,
        (~jnp.isfinite(step)) | (step <= 0),
        "Abelian DMRG descent_step must be finite and positive.",
    )
    current = canonicalize_abelian_mps(
        initial_state, center=initial_state.site_count // 2, normalize=True
    )
    energies = []
    residuals = []
    losses = []
    drifts = []
    protected_ok = []
    initial_charge = current.total_charge
    for _ in range(sweeps):
        energy, applied, application_loss = _abelian_energy(
            current,
            hamiltonian,
            maximum_bond_dimension=capacity,
        )
        residual = add_abelian_mps(applied, scale_abelian_mps(current, -energy))
        residual_norm = jnp.sqrt(
            jnp.maximum(jnp.real(abelian_mps_inner(residual, residual)), 0.0)
        )
        trial = add_abelian_mps(current, scale_abelian_mps(residual, -step))
        current, compression = compress_abelian_mps(
            trial,
            maximum_bond_dimension=capacity,
            normalize=True,
            protected_charges=protected_charges,
        )
        energies.append(energy)
        residuals.append(residual_norm)
        losses.append(application_loss + compression.accumulated_discarded_weight)
        drifts.append(
            jnp.asarray(
                0.0 if current.total_charge == initial_charge else 1.0,
                dtype=energy.dtype,
            )
        )
        protected_ok.append(compression.protected_sectors_satisfied)
    energy_values = jnp.stack(energies)
    residual_values = jnp.stack(residuals)
    loss_values = jnp.stack(losses)
    drift_values = jnp.stack(drifts)
    protected_values = jnp.stack(protected_ok)
    converged = residual_values[-1] <= float(residual_tolerance)
    valid = (
        jnp.all(jnp.isfinite(energy_values))
        & jnp.all(jnp.isfinite(residual_values))
        & jnp.all(jnp.isfinite(loss_values))
        & jnp.all(drift_values == 0)
        & jnp.all(protected_values)
    )
    return current, AbelianDMRGEvidence(
        energy_values,
        residual_values,
        loss_values,
        drift_values,
        protected_values,
        converged,
        valid,
        sweeps,
        current.precision.policy_id,
    )


def abelian_finite_tdvp(
    initial_state: AbelianMatrixProductState,
    hamiltonian: AbelianMatrixProductOperator,
    step_size: ArrayLike,
    /,
    *,
    step_count: int,
    maximum_bond_dimension: int,
    imaginary_time: bool = False,
    normalize: bool = True,
    protected_charges: Sequence[Sequence[int]] = (),
) -> tuple[AbelianMatrixProductState, AbelianTDVPEvidence]:
    """Second-order midpoint evolution with block-native MPO applications."""

    count = int(step_count)
    if count < 1:
        raise ValueError("step_count must be positive.")
    capacity = int(maximum_bond_dimension)
    if capacity < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    step = jnp.asarray(step_size, dtype=initial_state.norm().dtype)
    step = eqx.error_if(
        step,
        (~jnp.isfinite(step)) | (step <= 0),
        "Abelian TDVP step_size must be finite and positive.",
    )
    factor = jnp.asarray(-1.0 if imaginary_time else -1.0j)
    current = initial_state.normalized() if normalize else initial_state
    initial_charge = current.total_charge
    times = []
    norm_residuals = []
    energies = []
    losses = []
    drifts = []
    protected_ok = []
    for index in range(count):
        energy, first_action, first_loss = _abelian_energy(
            current,
            hamiltonian,
            maximum_bond_dimension=capacity,
        )
        midpoint = add_abelian_mps(
            current, scale_abelian_mps(first_action, factor * step * 0.5)
        )
        midpoint, midpoint_compression = compress_abelian_mps(
            midpoint,
            maximum_bond_dimension=capacity,
            normalize=False,
            protected_charges=protected_charges,
        )
        _, second_action, second_loss = _abelian_energy(
            midpoint,
            hamiltonian,
            maximum_bond_dimension=capacity,
        )
        trial = add_abelian_mps(current, scale_abelian_mps(second_action, factor * step))
        current, final_compression = compress_abelian_mps(
            trial,
            maximum_bond_dimension=capacity,
            normalize=normalize,
            protected_charges=protected_charges,
        )
        norm_residual = jnp.abs(current.norm() - 1.0) if normalize else jnp.asarray(0.0)
        times.append((index + 1) * step)
        norm_residuals.append(norm_residual)
        energies.append(energy)
        losses.append(
            first_loss
            + second_loss
            + midpoint_compression.accumulated_discarded_weight
            + final_compression.accumulated_discarded_weight
        )
        drifts.append(
            jnp.asarray(
                0.0 if current.total_charge == initial_charge else 1.0,
                dtype=energy.dtype,
            )
        )
        protected_ok.append(
            midpoint_compression.protected_sectors_satisfied
            & final_compression.protected_sectors_satisfied
        )
    time_values = jnp.stack(times)
    norm_values = jnp.stack(norm_residuals)
    energy_values = jnp.stack(energies)
    loss_values = jnp.stack(losses)
    drift_values = jnp.stack(drifts)
    protected_values = jnp.stack(protected_ok)
    valid = (
        jnp.all(jnp.isfinite(time_values))
        & jnp.all(jnp.isfinite(norm_values))
        & jnp.all(jnp.isfinite(energy_values))
        & jnp.all(jnp.isfinite(loss_values))
        & jnp.all(drift_values == 0)
        & jnp.all(protected_values)
    )
    return current, AbelianTDVPEvidence(
        time_values,
        norm_values,
        energy_values,
        loss_values,
        drift_values,
        protected_values,
        valid,
        count,
        current.precision.policy_id,
    )


__all__ = [
    "AbelianDMRGEvidence",
    "AbelianNearestNeighborHamiltonian",
    "AbelianTDVPEvidence",
    "AbelianTEBDEvidence",
    "AbelianTensorTruncationEvidence",
    "abelian_finite_dmrg",
    "abelian_finite_tdvp",
    "abelian_product_mps",
    "abelian_tebd_step",
    "apply_abelian_two_site_gate",
]

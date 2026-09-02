#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._abelian import AbelianCharge, AbelianLeg, AbelianTensor, AbelianTensorLayout
from ._core import MatrixProductOperator, MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy


def _same_leg_catalogue(left: AbelianLeg, right: AbelianLeg, /) -> bool:
    return (
        left.group.group_id == right.group.group_id
        and left.charges == right.charges
        and left.capacities == right.capacities
    )


def _boundary_leg(leg: AbelianLeg, charge: AbelianCharge, /) -> bool:
    return len(leg.charges) == 1 and leg.charges[0] == charge and leg.capacities == (1,)


class AbelianMatrixProductState(StrictModule):
    tensors: tuple[AbelianTensor, ...]
    precision: TensorNetworkPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    physical_dimensions: tuple[int, ...] = eqx.field(static=True)
    total_charge: AbelianCharge = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(self, tensors: Sequence[AbelianTensor], /):
        values = tuple(tensors)
        if not values or any(not isinstance(tensor, AbelianTensor) for tensor in values):
            raise TypeError(
                "tensors must be a nonempty sequence of AbelianTensor values."
            )
        precision = values[0].precision
        group = values[0].layout.legs[0].group
        for tensor in values:
            if len(tensor.layout.legs) != 3:
                raise ValueError("Abelian MPS site tensors require three legs.")
            left, physical, right = tensor.layout.legs
            if (left.orientation, physical.orientation, right.orientation) != (1, 1, -1):
                raise ValueError("Abelian MPS legs must have orientations (+,+,-).")
            if tensor.layout.total_charge != group.zero:
                raise ValueError("Abelian MPS site tensors must be charge invariant.")
            if tensor.precision.policy_id != precision.policy_id:
                raise ValueError("Abelian MPS precision policies must match.")
        for first, second in pairwise(values):
            if not first.layout.legs[2].dual_compatible(second.layout.legs[0]):
                raise ValueError(
                    "Adjacent Abelian MPS virtual legs must be dual compatible."
                )
        if not _boundary_leg(values[0].layout.legs[0], group.zero):
            raise ValueError(
                "The left Abelian MPS boundary must be neutral and one dimensional."
            )
        final_leg = values[-1].layout.legs[2]
        if len(final_leg.charges) != 1 or final_leg.capacities != (1,):
            raise ValueError(
                "The right Abelian MPS boundary must contain one charge sector."
            )
        self.tensors = values
        self.precision = precision
        self.precision_evidence = precision.evidence_for(
            tuple(block for tensor in values for block in tensor.blocks)
        )
        self.site_count = len(values)
        self.physical_dimensions = tuple(tensor.layout.legs[1].size for tensor in values)
        self.total_charge = final_leg.charges[0]
        self.structure_id = canonical_fingerprint(
            {
                "kind": "abelian-matrix-product-state",
                "layouts": tuple(tensor.layout.layout_id for tensor in values),
                "precision": precision.policy_id,
            }
        )

    def to_dense(self, /, *, maximum_elements: int = 1_000_000) -> Array:
        dense = MatrixProductState(
            tuple(tensor.to_dense() for tensor in self.tensors),
            precision=self.precision,
        )
        return dense.to_dense(maximum_elements=maximum_elements)

    def norm(self) -> Array:
        return self.precision.decision(jnp.sqrt(jnp.real(abelian_mps_inner(self, self))))

    def normalized(self) -> AbelianMatrixProductState:
        norm = self.norm()
        norm = eqx.error_if(
            norm,
            ~jnp.isfinite(norm) | (norm <= 0.0),
            "Abelian MPS norm must be finite and positive.",
        )
        first = self.tensors[0]
        tensors = (
            AbelianTensor(
                first.layout,
                tuple(block / norm for block in first.blocks),
                precision=self.precision,
            ),
        ) + self.tensors[1:]
        return AbelianMatrixProductState(tensors)


class AbelianMatrixProductOperator(StrictModule):
    tensors: tuple[AbelianTensor, ...]
    precision: TensorNetworkPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    output_dimensions: tuple[int, ...] = eqx.field(static=True)
    input_dimensions: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(self, tensors: Sequence[AbelianTensor], /):
        values = tuple(tensors)
        if not values or any(not isinstance(tensor, AbelianTensor) for tensor in values):
            raise TypeError(
                "tensors must be a nonempty sequence of AbelianTensor values."
            )
        precision = values[0].precision
        group = values[0].layout.legs[0].group
        for tensor in values:
            if len(tensor.layout.legs) != 4:
                raise ValueError("Abelian MPO site tensors require four legs.")
            orientations = tuple(leg.orientation for leg in tensor.layout.legs)
            if orientations != (1, -1, 1, -1):
                raise ValueError("Abelian MPO legs must have orientations (+,-,+,-).")
            if tensor.layout.total_charge != group.zero:
                raise ValueError("Abelian MPO site tensors must be charge invariant.")
            if tensor.precision.policy_id != precision.policy_id:
                raise ValueError("Abelian MPO precision policies must match.")
        for first, second in pairwise(values):
            if not first.layout.legs[3].dual_compatible(second.layout.legs[0]):
                raise ValueError(
                    "Adjacent Abelian MPO virtual legs must be dual compatible."
                )
        if not _boundary_leg(values[0].layout.legs[0], group.zero) or not _boundary_leg(
            values[-1].layout.legs[3], group.zero
        ):
            raise ValueError(
                "Abelian MPO boundaries must be neutral and one dimensional."
            )
        self.tensors = values
        self.precision = precision
        self.precision_evidence = precision.evidence_for(
            tuple(block for tensor in values for block in tensor.blocks)
        )
        self.site_count = len(values)
        self.output_dimensions = tuple(tensor.layout.legs[1].size for tensor in values)
        self.input_dimensions = tuple(tensor.layout.legs[2].size for tensor in values)
        self.structure_id = canonical_fingerprint(
            {
                "kind": "abelian-matrix-product-operator",
                "layouts": tuple(tensor.layout.layout_id for tensor in values),
                "precision": precision.policy_id,
            }
        )

    def to_dense(self, /, *, maximum_elements: int = 1_000_000) -> Array:
        dense = MatrixProductOperator(
            tuple(tensor.to_dense() for tensor in self.tensors),
            precision=self.precision,
        )
        return dense.to_dense(maximum_elements=maximum_elements)


def _block_map(tensor: AbelianTensor):
    return dict(zip(tensor.layout.sectors, tensor.blocks, strict=True))


def abelian_mps_inner(
    left: AbelianMatrixProductState,
    right: AbelianMatrixProductState,
    /,
) -> Array:
    if not isinstance(left, AbelianMatrixProductState) or not isinstance(
        right, AbelianMatrixProductState
    ):
        raise TypeError("left and right must be AbelianMatrixProductState values.")
    if (
        left.site_count != right.site_count
        or left.total_charge != right.total_charge
        or left.precision.policy_id != right.precision.policy_id
        or any(
            not _matching_physical_basis(first.layout.legs[1], second.layout.legs[1])
            for first, second in zip(left.tensors, right.tensors, strict=True)
        )
    ):
        raise ValueError(
            "Abelian MPS site counts, total charge, physical bases, and precision "
            "must match for inner products."
        )
    precision = left.precision
    environment = {(0, 0): jnp.ones((1, 1), dtype=left.tensors[0].blocks[0].dtype)}
    for left_tensor, right_tensor in zip(left.tensors, right.tensors, strict=True):
        next_environment = {}
        for (left_charge, right_charge), value in environment.items():
            for left_sector, left_block in zip(
                left_tensor.layout.sectors, left_tensor.blocks, strict=True
            ):
                if left_sector[0] != left_charge:
                    continue
                for right_sector, right_block in zip(
                    right_tensor.layout.sectors, right_tensor.blocks, strict=True
                ):
                    if (
                        right_sector[0] != right_charge
                        or right_sector[1] != left_sector[1]
                    ):
                        continue
                    key = (left_sector[2], right_sector[2])
                    contribution = oe.contract(
                        "ab,api,bpj->ij",
                        precision.accumulation(value),
                        jnp.conj(precision.accumulation(left_block)),
                        precision.accumulation(right_block),
                    )
                    next_environment[key] = (
                        next_environment[key] + contribution
                        if key in next_environment
                        else contribution
                    )
        environment = next_environment
    return precision.output(environment[(0, 0)].reshape(()))


def abelian_mps_one_site_expectation(
    state: AbelianMatrixProductState,
    site: int,
    operator: ArrayLike,
    /,
) -> Array:
    site_ = int(site)
    if not 0 <= site_ < state.site_count:
        raise ValueError("Expectation site is outside the Abelian MPS.")
    physical_leg = state.tensors[site_].layout.legs[1]
    value = jnp.asarray(operator)
    if value.shape != (physical_leg.size, physical_leg.size):
        raise ValueError(
            "One-site operator shape does not match the Abelian physical leg."
        )
    offsets = []
    start = 0
    for capacity in physical_leg.capacities:
        offsets.append(start)
        start += capacity
    precision = state.precision
    environment = {(0, 0): jnp.ones((1, 1), dtype=state.tensors[0].blocks[0].dtype)}
    for index, tensor in enumerate(state.tensors):
        next_environment = {}
        for (bra_charge, ket_charge), env in environment.items():
            for bra_sector, bra_block in zip(
                tensor.layout.sectors, tensor.blocks, strict=True
            ):
                if bra_sector[0] != bra_charge:
                    continue
                for ket_sector, ket_block in zip(
                    tensor.layout.sectors, tensor.blocks, strict=True
                ):
                    if ket_sector[0] != ket_charge:
                        continue
                    if index != site_ and bra_sector[1] != ket_sector[1]:
                        continue
                    if index == site_:
                        row = bra_sector[1]
                        column = ket_sector[1]
                        local = value[
                            offsets[row] : offsets[row] + physical_leg.capacities[row],
                            offsets[column] : offsets[column]
                            + physical_leg.capacities[column],
                        ]
                    else:
                        local = jnp.eye(bra_block.shape[1], dtype=value.dtype)
                    key = (bra_sector[2], ket_sector[2])
                    contribution = oe.contract(
                        "ab,api,pq,bqj->ij",
                        precision.accumulation(env),
                        jnp.conj(precision.accumulation(bra_block)),
                        precision.accumulation(local),
                        precision.accumulation(ket_block),
                    )
                    next_environment[key] = (
                        next_environment[key] + contribution
                        if key in next_environment
                        else contribution
                    )
        environment = next_environment
    return precision.output(environment[(0, 0)].reshape(()))


def _replace_blocks(tensor, blocks):
    return AbelianTensor(tensor.layout, tuple(blocks), precision=tensor.precision)


def canonicalize_abelian_mps(
    state: AbelianMatrixProductState,
    /,
    *,
    center: int,
    normalize: bool = True,
    prepared_routes: tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        tuple[tuple[tuple[int, ...], ...], ...],
    ]
    | None = None,
) -> AbelianMatrixProductState:
    center_ = int(center)
    if not 0 <= center_ < state.site_count:
        raise ValueError("center is outside the Abelian MPS.")
    precision = state.precision
    tensors = list(state.tensors)
    if prepared_routes is not None:
        left_prepared, right_prepared = prepared_routes
        if (
            len(left_prepared) != state.site_count
            or len(right_prepared) != state.site_count
        ):
            raise ValueError(
                "Prepared Abelian canonicalization routes do not match state."
            )
    else:
        left_prepared = tuple(
            tuple(
                tuple(
                    index
                    for index, sector in enumerate(tensor.layout.sectors)
                    if sector[0] == charge
                )
                for charge in range(len(tensor.layout.legs[0].charges))
            )
            for tensor in state.tensors
        )
        right_prepared = tuple(
            tuple(
                tuple(
                    index
                    for index, sector in enumerate(tensor.layout.sectors)
                    if sector[2] == charge
                )
                for charge in range(len(tensor.layout.legs[2].charges))
            )
            for tensor in state.tensors
        )
    tensors = list(state.tensors)
    for site in range(center_):
        tensor = tensors[site]
        blocks = list(tensor.blocks)
        right_leg = tensor.layout.legs[2]
        transfers = []
        for right_sector, routes in enumerate(right_prepared[site]):
            if not routes:
                transfers.append(
                    jnp.zeros(
                        (
                            right_leg.capacities[right_sector],
                            right_leg.capacities[right_sector],
                        ),
                        dtype=blocks[0].dtype,
                    )
                )
                continue
            row_sizes = [
                blocks[index].shape[0] * blocks[index].shape[1] for index in routes
            ]
            matrix = jnp.concatenate(
                [
                    blocks[index].reshape((row_sizes[position], -1))
                    for position, index in enumerate(routes)
                ],
                axis=0,
            )
            q, r = jnp.linalg.qr(precision.factorization(matrix))
            capacity = right_leg.capacities[right_sector]
            rank = q.shape[1]
            q_pad = (
                jnp.zeros((matrix.shape[0], capacity), dtype=q.dtype).at[:, :rank].set(q)
            )
            r_pad = jnp.zeros((capacity, capacity), dtype=r.dtype).at[:rank, :].set(r)
            active = jnp.arange(capacity) < right_leg.active_degeneracies[right_sector]
            q_pad = jnp.where(active[None, :], q_pad, 0)
            r_pad = jnp.where(active[:, None] & active[None, :], r_pad, 0)
            cursor = 0
            for size, index in zip(row_sizes, routes, strict=True):
                blocks[index] = q_pad[cursor : cursor + size].reshape(blocks[index].shape)
                cursor += size
            transfers.append(r_pad)
        tensors[site] = _replace_blocks(tensor, blocks)
        next_tensor = tensors[site + 1]
        next_blocks = [
            oe.contract("ab,bpr->apr", transfers[sector[0]], block)
            for sector, block in zip(
                next_tensor.layout.sectors, next_tensor.blocks, strict=True
            )
        ]
        tensors[site + 1] = _replace_blocks(next_tensor, next_blocks)
    for site in range(state.site_count - 1, center_, -1):
        tensor = tensors[site]
        blocks = list(tensor.blocks)
        left_leg = tensor.layout.legs[0]
        transfers = []
        for left_sector, routes in enumerate(left_prepared[site]):
            if not routes:
                transfers.append(
                    jnp.zeros(
                        (
                            left_leg.capacities[left_sector],
                            left_leg.capacities[left_sector],
                        ),
                        dtype=blocks[0].dtype,
                    )
                )
                continue
            column_sizes = [
                blocks[index].shape[1] * blocks[index].shape[2] for index in routes
            ]
            matrix = jnp.concatenate(
                [blocks[index].reshape((blocks[index].shape[0], -1)) for index in routes],
                axis=1,
            )
            q, r = jnp.linalg.qr(precision.factorization(matrix.T))
            capacity = left_leg.capacities[left_sector]
            rank = q.shape[1]
            q_pad = (
                jnp.zeros((matrix.shape[1], capacity), dtype=q.dtype).at[:, :rank].set(q)
            )
            r_pad = jnp.zeros((capacity, capacity), dtype=r.dtype).at[:rank, :].set(r)
            active = jnp.arange(capacity) < left_leg.active_degeneracies[left_sector]
            q_pad = jnp.where(active[None, :], q_pad, 0)
            r_pad = jnp.where(active[:, None] & active[None, :], r_pad, 0)
            canonical = q_pad.T
            cursor = 0
            for size, index in zip(column_sizes, routes, strict=True):
                blocks[index] = canonical[:, cursor : cursor + size].reshape(
                    blocks[index].shape
                )
                cursor += size
            transfers.append(r_pad.T)
        tensors[site] = _replace_blocks(tensor, blocks)
        previous = tensors[site - 1]
        previous_blocks = [
            oe.contract("lpa,ab->lpb", block, transfers[sector[2]])
            for sector, block in zip(
                previous.layout.sectors, previous.blocks, strict=True
            )
        ]
        tensors[site - 1] = _replace_blocks(previous, previous_blocks)
    result = AbelianMatrixProductState(tuple(tensors))
    return result.normalized() if normalize else result


class AbelianChainCompressionEvidence(StrictModule):
    """Finite ordered loss record for an Abelian chain compression."""

    truncations: tuple[object, ...]
    accumulated_discarded_weight: Array
    maximum_discarded_weight: Array
    protected_sectors_satisfied: Array
    valid: Array
    precision_policy_id: str = eqx.field(static=True)


class AbelianEnvironmentEvidence(StrictModule):
    """Expectation and norm residual produced by block-native environments."""

    expectation: Array
    state_norm: Array
    imaginary_residual: Array
    valid: Array
    precision_policy_id: str = eqx.field(static=True)


def _matching_physical_basis(left: AbelianLeg, right: AbelianLeg, /) -> bool:
    return (
        left.group.group_id == right.group.group_id
        and left.charges == right.charges
        and left.capacities == right.capacities
    )


def _merged_leg(
    first: AbelianLeg,
    second: AbelianLeg,
    /,
    *,
    orientation: int,
    subtract: bool,
) -> tuple[AbelianLeg, tuple[tuple[int, int], ...]]:
    if first.group.group_id != second.group.group_id:
        raise ValueError("Composite Abelian legs must use the same group.")
    group = first.group
    charges: list[AbelianCharge] = []
    capacities: list[int] = []
    active: list[Array] = []
    routes = []
    for first_ordinal, first_charge in enumerate(first.charges):
        for second_ordinal, second_charge in enumerate(second.charges):
            charge = (
                group.subtract(first_charge, second_charge)
                if subtract
                else group.add(first_charge, second_charge)
            )
            size = first.capacities[first_ordinal] * second.capacities[second_ordinal]
            active_size = (
                first.active_degeneracies[first_ordinal]
                * second.active_degeneracies[second_ordinal]
            )
            if charge in charges:
                ordinal = charges.index(charge)
                offset = active[ordinal]
                capacities[ordinal] += size
                active[ordinal] = active[ordinal] + active_size
            else:
                ordinal = len(charges)
                offset = 0
                charges.append(charge)
                capacities.append(size)
                active.append(active_size)
            routes.append((ordinal, offset))
    return (
        AbelianLeg(
            group,
            tuple(charges),
            tuple(capacities),
            orientation=orientation,
            active_degeneracies=jnp.stack(active),
        ),
        tuple(routes),
    )


def _direct_sum_leg(
    left: AbelianLeg, right: AbelianLeg, /
) -> tuple[AbelianLeg, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    if (
        left.group.group_id != right.group.group_id
        or left.orientation != right.orientation
    ):
        raise ValueError("Direct-sum Abelian legs must share group and orientation.")
    charges: list[AbelianCharge] = []
    capacities: list[int] = []
    active: list[Array] = []
    left_routes = []
    right_routes = []
    for source, destination in ((left, left_routes), (right, right_routes)):
        for ordinal, charge in enumerate(source.charges):
            if charge in charges:
                result_ordinal = charges.index(charge)
                offset = active[result_ordinal]
                capacities[result_ordinal] += source.capacities[ordinal]
                active[result_ordinal] = (
                    active[result_ordinal] + source.active_degeneracies[ordinal]
                )
            else:
                result_ordinal = len(charges)
                offset = 0
                charges.append(charge)
                capacities.append(source.capacities[ordinal])
                active.append(source.active_degeneracies[ordinal])
            destination.append((result_ordinal, offset))
    return (
        AbelianLeg(
            left.group,
            tuple(charges),
            tuple(capacities),
            orientation=left.orientation,
            active_degeneracies=jnp.stack(active),
        ),
        tuple(left_routes),
        tuple(right_routes),
    )


def scale_abelian_mps(
    state: AbelianMatrixProductState, scalar: ArrayLike, /
) -> AbelianMatrixProductState:
    if not isinstance(state, AbelianMatrixProductState):
        raise TypeError("state must be AbelianMatrixProductState.")
    value = jnp.asarray(scalar)
    if value.shape != ():
        raise ValueError("Abelian MPS scale must be scalar.")
    first = state.tensors[0]
    scaled = AbelianTensor(
        first.layout,
        tuple(value * block for block in first.blocks),
        precision=state.precision,
    )
    return AbelianMatrixProductState((scaled,) + state.tensors[1:])


def scale_abelian_mpo(
    operator: AbelianMatrixProductOperator, scalar: ArrayLike, /
) -> AbelianMatrixProductOperator:
    if not isinstance(operator, AbelianMatrixProductOperator):
        raise TypeError("operator must be AbelianMatrixProductOperator.")
    value = jnp.asarray(scalar)
    if value.shape != ():
        raise ValueError("Abelian MPO scale must be scalar.")
    first = operator.tensors[0]
    scaled = AbelianTensor(
        first.layout,
        tuple(value * block for block in first.blocks),
        precision=operator.precision,
    )
    return AbelianMatrixProductOperator((scaled,) + operator.tensors[1:])


def adjoint_abelian_mpo(
    operator: AbelianMatrixProductOperator, /
) -> AbelianMatrixProductOperator:
    if not isinstance(operator, AbelianMatrixProductOperator):
        raise TypeError("operator must be AbelianMatrixProductOperator.")
    tensors = []
    for tensor in operator.tensors:
        left, output, input_, right = tensor.layout.legs
        layout = AbelianTensorLayout(
            (left, input_.dual(), output.dual(), right),
            total_charge=tensor.layout.total_charge,
        )
        blocks = []
        for sector in layout.sectors:
            original = (sector[0], sector[2], sector[1], sector[3])
            index = tensor.layout.sectors.index(original)
            blocks.append(jnp.swapaxes(jnp.conj(tensor.blocks[index]), 1, 2))
        tensors.append(AbelianTensor(layout, tuple(blocks), precision=operator.precision))
    return AbelianMatrixProductOperator(tuple(tensors))


def _add_abelian_chains(left, right, /, *, operator: bool):
    expected_type = (
        AbelianMatrixProductOperator if operator else AbelianMatrixProductState
    )
    if not isinstance(left, expected_type) or not isinstance(right, expected_type):
        raise TypeError("Abelian chain addition requires matching chain values.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("Abelian chain precision policies must match.")
    if left.site_count != right.site_count:
        raise ValueError("Abelian chain site counts must match.")
    if operator:
        dimensions_match = (
            left.output_dimensions == right.output_dimensions
            and left.input_dimensions == right.input_dimensions
        )
        right_axis = 3
        physical_axes = (1, 2)
    else:
        dimensions_match = (
            left.physical_dimensions == right.physical_dimensions
            and left.total_charge == right.total_charge
        )
        right_axis = 2
        physical_axes = (1,)
    if not dimensions_match:
        raise ValueError("Abelian chain physical dimensions or charges differ.")
    for left_tensor, right_tensor in zip(left.tensors, right.tensors, strict=True):
        for axis in physical_axes:
            if not _matching_physical_basis(
                left_tensor.layout.legs[axis], right_tensor.layout.legs[axis]
            ):
                raise ValueError("Abelian chain physical bases must match.")
    bonds = []
    left_bond_routes = []
    right_bond_routes = []
    for site in range(left.site_count - 1):
        leg, first_routes, second_routes = _direct_sum_leg(
            left.tensors[site].layout.legs[right_axis],
            right.tensors[site].layout.legs[right_axis],
        )
        bonds.append(leg)
        left_bond_routes.append(first_routes)
        right_bond_routes.append(second_routes)
    tensors = []
    for site, (first, second) in enumerate(zip(left.tensors, right.tensors, strict=True)):
        legs = list(first.layout.legs)
        if site > 0:
            legs[0] = bonds[site - 1].dual()
        if site < left.site_count - 1:
            legs[right_axis] = bonds[site]
        layout = AbelianTensorLayout(tuple(legs), total_charge=first.layout.total_charge)
        dtype = jnp.result_type(first.blocks[0], second.blocks[0])
        blocks = [jnp.zeros(shape, dtype=dtype) for shape in layout.block_shapes]
        for source, routes, side in (
            (first, left_bond_routes, 0),
            (second, right_bond_routes, 1),
        ):
            for sector, block in zip(source.layout.sectors, source.blocks, strict=True):
                result_sector = list(sector)
                indices = [jnp.arange(size) for size in block.shape]
                if site > 0:
                    ordinal, offset = routes[site - 1][sector[0]]
                    result_sector[0] = ordinal
                    indices[0] = offset + jnp.arange(block.shape[0])
                if site < left.site_count - 1:
                    bond_routes = (
                        left_bond_routes[site] if side == 0 else right_bond_routes[site]
                    )
                    ordinal, offset = bond_routes[sector[right_axis]]
                    result_sector[right_axis] = ordinal
                    indices[right_axis] = offset + jnp.arange(block.shape[right_axis])
                result_index = layout.sectors.index(tuple(result_sector))
                blocks[result_index] = (
                    blocks[result_index].at[jnp.ix_(*indices)].add(block)
                )
        tensors.append(AbelianTensor(layout, tuple(blocks), precision=left.precision))
    return expected_type(tuple(tensors))


def add_abelian_mps(
    left: AbelianMatrixProductState, right: AbelianMatrixProductState, /
) -> AbelianMatrixProductState:
    return _add_abelian_chains(left, right, operator=False)


def add_abelian_mpo(
    left: AbelianMatrixProductOperator,
    right: AbelianMatrixProductOperator,
    /,
) -> AbelianMatrixProductOperator:
    return _add_abelian_chains(left, right, operator=True)


def apply_abelian_mpo_exact(
    operator: AbelianMatrixProductOperator,
    state: AbelianMatrixProductState,
    /,
) -> AbelianMatrixProductState:
    """Apply an MPO through charge-routed blocks before optional compression."""

    if not isinstance(operator, AbelianMatrixProductOperator) or not isinstance(
        state, AbelianMatrixProductState
    ):
        raise TypeError("operator and state must be Abelian MPO and MPS values.")
    if (
        operator.site_count != state.site_count
        or operator.input_dimensions != state.physical_dimensions
    ):
        raise ValueError("Abelian MPO inputs must match Abelian MPS dimensions.")
    if operator.precision.policy_id != state.precision.policy_id:
        raise ValueError("Abelian MPO and MPS precision policies must match.")
    tensors = []
    for op_tensor, state_tensor in zip(operator.tensors, state.tensors, strict=True):
        op_left, op_output, op_input, op_right = op_tensor.layout.legs
        state_left, state_input, state_right = state_tensor.layout.legs
        if not _matching_physical_basis(op_input, state_input):
            raise ValueError("Abelian MPO input and state physical bases differ.")
        left_leg, left_routes = _merged_leg(
            state_left, op_left, orientation=1, subtract=True
        )
        right_leg, right_routes = _merged_leg(
            state_right, op_right, orientation=-1, subtract=True
        )
        output_leg = op_output.dual()
        layout = AbelianTensorLayout((left_leg, output_leg, right_leg))
        dtype = jnp.result_type(op_tensor.blocks[0], state_tensor.blocks[0])
        blocks = [jnp.zeros(shape, dtype=dtype) for shape in layout.block_shapes]
        op_left_count = len(op_left.charges)
        op_right_count = len(op_right.charges)
        for op_sector, op_block in zip(
            op_tensor.layout.sectors, op_tensor.blocks, strict=True
        ):
            for state_sector, state_block in zip(
                state_tensor.layout.sectors, state_tensor.blocks, strict=True
            ):
                if op_sector[2] != state_sector[1]:
                    continue
                left_route = state_sector[0] * op_left_count + op_sector[0]
                right_route = state_sector[2] * op_right_count + op_sector[3]
                left_ordinal, left_offset = left_routes[left_route]
                right_ordinal, right_offset = right_routes[right_route]
                result_sector = (
                    left_ordinal,
                    op_sector[1],
                    right_ordinal,
                )
                result_index = layout.sectors.index(result_sector)
                combined = oe.contract(
                    "aoib,cid->caodb",
                    state.precision.contraction(op_block),
                    state.precision.contraction(state_block),
                ).reshape(
                    (
                        state_block.shape[0] * op_block.shape[0],
                        op_block.shape[1],
                        state_block.shape[2] * op_block.shape[3],
                    )
                )
                left_op_capacity = op_left.capacities[op_sector[0]]
                left_state_active = state_left.active_degeneracies[state_sector[0]]
                left_op_active = op_left.active_degeneracies[op_sector[0]]
                left_state_index = jnp.arange(combined.shape[0]) // left_op_capacity
                left_op_index = jnp.arange(combined.shape[0]) % left_op_capacity
                left_valid = (left_state_index < left_state_active) & (
                    left_op_index < left_op_active
                )
                left_indices = (
                    left_offset + left_state_index * left_op_active + left_op_index
                )
                right_op_capacity = op_right.capacities[op_sector[3]]
                right_state_active = state_right.active_degeneracies[state_sector[2]]
                right_op_active = op_right.active_degeneracies[op_sector[3]]
                right_state_index = jnp.arange(combined.shape[2]) // right_op_capacity
                right_op_index = jnp.arange(combined.shape[2]) % right_op_capacity
                right_valid = (right_state_index < right_state_active) & (
                    right_op_index < right_op_active
                )
                right_indices = (
                    right_offset + right_state_index * right_op_active + right_op_index
                )
                masked = combined * left_valid[:, None, None] * right_valid[None, None, :]
                blocks[result_index] = (
                    blocks[result_index]
                    .at[
                        jnp.ix_(
                            left_indices,
                            jnp.arange(combined.shape[1]),
                            right_indices,
                        )
                    ]
                    .add(masked)
                )
        tensors.append(AbelianTensor(layout, tuple(blocks), precision=state.precision))
    return AbelianMatrixProductState(tuple(tensors))


def compose_abelian_mpo_exact(
    left: AbelianMatrixProductOperator,
    right: AbelianMatrixProductOperator,
    /,
) -> AbelianMatrixProductOperator:
    """Compose MPO blocks with merged charge-product virtual allocations."""

    if not isinstance(left, AbelianMatrixProductOperator) or not isinstance(
        right, AbelianMatrixProductOperator
    ):
        raise TypeError("left and right must be AbelianMatrixProductOperator values.")
    if (
        left.site_count != right.site_count
        or left.input_dimensions != right.output_dimensions
    ):
        raise ValueError("Abelian MPO composition dimensions do not match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("Abelian MPO precision policies must match.")
    tensors = []
    for first, second in zip(left.tensors, right.tensors, strict=True):
        first_left, output, middle_left, first_right = first.layout.legs
        second_left, middle_right, input_, second_right = second.layout.legs
        if not middle_left.dual_compatible(middle_right):
            raise ValueError("Composed Abelian MPO physical bases must be dual.")
        left_leg, left_routes = _merged_leg(
            first_left, second_left, orientation=1, subtract=False
        )
        right_leg, right_routes = _merged_leg(
            first_right, second_right, orientation=-1, subtract=False
        )
        layout = AbelianTensorLayout((left_leg, output, input_, right_leg))
        dtype = jnp.result_type(first.blocks[0], second.blocks[0])
        blocks = [jnp.zeros(shape, dtype=dtype) for shape in layout.block_shapes]
        second_left_count = len(second_left.charges)
        second_right_count = len(second_right.charges)
        for first_sector, first_block in zip(
            first.layout.sectors, first.blocks, strict=True
        ):
            for second_sector, second_block in zip(
                second.layout.sectors, second.blocks, strict=True
            ):
                if first_sector[2] != second_sector[1]:
                    continue
                left_route = first_sector[0] * second_left_count + second_sector[0]
                right_route = first_sector[3] * second_right_count + second_sector[3]
                left_ordinal, left_offset = left_routes[left_route]
                right_ordinal, right_offset = right_routes[right_route]
                result_sector = (
                    left_ordinal,
                    first_sector[1],
                    second_sector[2],
                    right_ordinal,
                )
                result_index = layout.sectors.index(result_sector)
                combined = oe.contract(
                    "aomb,cmid->acoibd",
                    left.precision.contraction(first_block),
                    left.precision.contraction(second_block),
                ).reshape(
                    (
                        first_block.shape[0] * second_block.shape[0],
                        first_block.shape[1],
                        second_block.shape[2],
                        first_block.shape[3] * second_block.shape[3],
                    )
                )
                left_second_capacity = second_left.capacities[second_sector[0]]
                left_first_active = first_left.active_degeneracies[first_sector[0]]
                left_second_active = second_left.active_degeneracies[second_sector[0]]
                left_first_index = jnp.arange(combined.shape[0]) // left_second_capacity
                left_second_index = jnp.arange(combined.shape[0]) % left_second_capacity
                left_valid = (left_first_index < left_first_active) & (
                    left_second_index < left_second_active
                )
                left_indices = (
                    left_offset
                    + left_first_index * left_second_active
                    + left_second_index
                )
                right_second_capacity = second_right.capacities[second_sector[3]]
                right_first_active = first_right.active_degeneracies[first_sector[3]]
                right_second_active = second_right.active_degeneracies[second_sector[3]]
                right_first_index = jnp.arange(combined.shape[3]) // right_second_capacity
                right_second_index = jnp.arange(combined.shape[3]) % right_second_capacity
                right_valid = (right_first_index < right_first_active) & (
                    right_second_index < right_second_active
                )
                right_indices = (
                    right_offset
                    + right_first_index * right_second_active
                    + right_second_index
                )
                masked = (
                    combined
                    * left_valid[:, None, None, None]
                    * right_valid[None, None, None, :]
                )
                blocks[result_index] = (
                    blocks[result_index]
                    .at[
                        jnp.ix_(
                            left_indices,
                            jnp.arange(combined.shape[1]),
                            jnp.arange(combined.shape[2]),
                            right_indices,
                        )
                    ]
                    .add(masked)
                )
        tensors.append(AbelianTensor(layout, tuple(blocks), precision=left.precision))
    return AbelianMatrixProductOperator(tuple(tensors))


def compress_abelian_mps(
    state: AbelianMatrixProductState,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = False,
    protected_charges: Sequence[Sequence[int]] = (),
) -> tuple[AbelianMatrixProductState, AbelianChainCompressionEvidence]:
    """Compress each bond using conserving identity gates and global selection."""

    from ._abelian_evolution import apply_abelian_two_site_gate

    capacity = int(maximum_bond_dimension)
    if capacity < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    current = canonicalize_abelian_mps(state, center=0, normalize=False)
    records = []
    for site in range(state.site_count - 1):
        left_size = current.physical_dimensions[site]
        right_size = current.physical_dimensions[site + 1]
        gate = jnp.eye(
            left_size * right_size, dtype=current.tensors[site].blocks[0].dtype
        ).reshape((left_size, right_size, left_size, right_size))
        current, evidence = apply_abelian_two_site_gate(
            current,
            site,
            gate,
            maximum_bond_dimension=capacity,
            normalize=False,
            protected_charges=protected_charges,
        )
        records.append(evidence)
    if normalize:
        current = current.normalized()
    weights = (
        jnp.stack([record.discarded_weight for record in records])
        if records
        else jnp.zeros((0,), dtype=current.norm().dtype)
    )
    protected = (
        jnp.all(jnp.stack([record.protected_sectors_satisfied for record in records]))
        if records
        else jnp.asarray(True)
    )
    evidence = AbelianChainCompressionEvidence(
        tuple(records),
        jnp.sum(weights),
        jnp.max(weights) if records else jnp.asarray(0.0, dtype=current.norm().dtype),
        protected,
        jnp.all(jnp.isfinite(weights)) & protected,
        current.precision.policy_id,
    )
    return current, evidence


def _mpo_as_fused_abelian_mps(
    operator: AbelianMatrixProductOperator, /
) -> tuple[
    AbelianMatrixProductState,
    tuple[tuple[AbelianLeg, tuple[tuple[int, Array], ...]], ...],
]:
    tensors = []
    physical_records = []
    for tensor in operator.tensors:
        left, output, input_, right = tensor.layout.legs
        fused, routes = _merged_leg(input_, output, orientation=1, subtract=True)
        layout = AbelianTensorLayout((left, fused, right))
        dtype = tensor.blocks[0].dtype
        blocks = [jnp.zeros(shape, dtype=dtype) for shape in layout.block_shapes]
        output_count = len(output.charges)
        for sector, block in zip(tensor.layout.sectors, tensor.blocks, strict=True):
            route_index = sector[2] * output_count + sector[1]
            fused_ordinal, offset = routes[route_index]
            result_sector = (sector[0], fused_ordinal, sector[3])
            result_index = layout.sectors.index(result_sector)
            input_capacity = input_.capacities[sector[2]]
            output_capacity = output.capacities[sector[1]]
            input_active = input_.active_degeneracies[sector[2]]
            output_active = output.active_degeneracies[sector[1]]
            input_index = jnp.arange(input_capacity * output_capacity) // output_capacity
            output_index = jnp.arange(input_capacity * output_capacity) % output_capacity
            valid = (input_index < input_active) & (output_index < output_active)
            target = offset + input_index * output_active + output_index
            values = jnp.transpose(block, (0, 2, 1, 3)).reshape(
                (
                    block.shape[0],
                    input_capacity * output_capacity,
                    block.shape[3],
                )
            )
            values = values * valid[None, :, None]
            blocks[result_index] = (
                blocks[result_index]
                .at[
                    jnp.ix_(
                        jnp.arange(block.shape[0]),
                        target,
                        jnp.arange(block.shape[3]),
                    )
                ]
                .add(values)
            )
        tensors.append(AbelianTensor(layout, tuple(blocks), precision=operator.precision))
        physical_records.append((fused, routes))
    return AbelianMatrixProductState(tuple(tensors)), tuple(physical_records)


def _fused_abelian_mps_as_mpo(
    state: AbelianMatrixProductState,
    template: AbelianMatrixProductOperator,
    physical_records: tuple[tuple[AbelianLeg, tuple[tuple[int, Array], ...]], ...],
    /,
) -> AbelianMatrixProductOperator:
    tensors = []
    for state_tensor, template_tensor, (_, routes) in zip(
        state.tensors, template.tensors, physical_records, strict=True
    ):
        _, output, input_, _ = template_tensor.layout.legs
        left = state_tensor.layout.legs[0]
        right = state_tensor.layout.legs[2]
        layout = AbelianTensorLayout((left, output, input_, right))
        dtype = state_tensor.blocks[0].dtype
        blocks = [jnp.zeros(shape, dtype=dtype) for shape in layout.block_shapes]
        output_count = len(output.charges)
        for result_index, sector in enumerate(layout.sectors):
            route_index = sector[2] * output_count + sector[1]
            fused_ordinal, offset = routes[route_index]
            source_sector = (sector[0], fused_ordinal, sector[3])
            if source_sector not in state_tensor.layout.sectors:
                continue
            source_index = state_tensor.layout.sectors.index(source_sector)
            source = state_tensor.blocks[source_index]
            input_capacity = input_.capacities[sector[2]]
            output_capacity = output.capacities[sector[1]]
            input_active = input_.active_degeneracies[sector[2]]
            output_active = output.active_degeneracies[sector[1]]
            input_index = jnp.arange(input_capacity * output_capacity) // output_capacity
            output_index = jnp.arange(input_capacity * output_capacity) % output_capacity
            valid = (input_index < input_active) & (output_index < output_active)
            source_positions = offset + input_index * output_active + output_index
            values = source[:, source_positions, :] * valid[None, :, None]
            blocks[result_index] = jnp.transpose(
                values.reshape(
                    (
                        source.shape[0],
                        input_capacity,
                        output_capacity,
                        source.shape[2],
                    )
                ),
                (0, 2, 1, 3),
            )
        tensors.append(AbelianTensor(layout, tuple(blocks), precision=state.precision))
    return AbelianMatrixProductOperator(tuple(tensors))


def compress_abelian_mpo(
    operator: AbelianMatrixProductOperator,
    /,
    *,
    maximum_bond_dimension: int,
    protected_charges: Sequence[Sequence[int]] = (),
) -> tuple[AbelianMatrixProductOperator, AbelianChainCompressionEvidence]:
    """Compress an MPO by fusing its covariant physical pair without densifying."""

    if not isinstance(operator, AbelianMatrixProductOperator):
        raise TypeError("operator must be AbelianMatrixProductOperator.")
    fused, records = _mpo_as_fused_abelian_mps(operator)
    compressed, evidence = compress_abelian_mps(
        fused,
        maximum_bond_dimension=maximum_bond_dimension,
        normalize=False,
        protected_charges=protected_charges,
    )
    return _fused_abelian_mps_as_mpo(compressed, operator, records), evidence


def apply_abelian_mpo(
    operator: AbelianMatrixProductOperator,
    state: AbelianMatrixProductState,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = False,
    protected_charges: Sequence[Sequence[int]] = (),
) -> tuple[AbelianMatrixProductState, AbelianChainCompressionEvidence]:
    exact = apply_abelian_mpo_exact(operator, state)
    return compress_abelian_mps(
        exact,
        maximum_bond_dimension=maximum_bond_dimension,
        normalize=normalize,
        protected_charges=protected_charges,
    )


def compose_abelian_mpo(
    left: AbelianMatrixProductOperator,
    right: AbelianMatrixProductOperator,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[AbelianMatrixProductOperator, AbelianChainCompressionEvidence]:
    exact = compose_abelian_mpo_exact(left, right)
    return compress_abelian_mpo(exact, maximum_bond_dimension=maximum_bond_dimension)


def adjoint_abelian_tensor(tensor: AbelianTensor, /) -> AbelianTensor:
    if not isinstance(tensor, AbelianTensor):
        raise TypeError("tensor must be AbelianTensor.")
    group = tensor.layout.legs[0].group
    layout = AbelianTensorLayout(
        tuple(leg.dual() for leg in tensor.layout.legs),
        total_charge=group.negate(tensor.layout.total_charge),
    )
    blocks = tuple(
        jnp.conj(tensor.blocks[tensor.layout.sectors.index(sector)])
        for sector in layout.sectors
    )
    return AbelianTensor(layout, blocks, precision=tensor.precision)


def add_abelian_tensors(left: AbelianTensor, right: AbelianTensor, /) -> AbelianTensor:
    if not isinstance(left, AbelianTensor) or not isinstance(right, AbelianTensor):
        raise TypeError("left and right must be AbelianTensor values.")
    if left.layout.layout_id != right.layout.layout_id:
        raise ValueError("Abelian tensor layouts must match for addition.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("Abelian tensor precision policies must match.")
    return AbelianTensor(
        left.layout,
        tuple(
            first + second
            for first, second in zip(left.blocks, right.blocks, strict=True)
        ),
        precision=left.precision,
    )


def scale_abelian_tensor(tensor: AbelianTensor, scalar: ArrayLike, /) -> AbelianTensor:
    if not isinstance(tensor, AbelianTensor):
        raise TypeError("tensor must be AbelianTensor.")
    value = jnp.asarray(scalar)
    if value.shape != ():
        raise ValueError("Abelian tensor scale must be scalar.")
    return AbelianTensor(
        tensor.layout,
        tuple(value * block for block in tensor.blocks),
        precision=tensor.precision,
    )


def abelian_mpo_environment_evidence(
    state: AbelianMatrixProductState,
    operator: AbelianMatrixProductOperator,
    /,
    *,
    maximum_bond_dimension: int,
) -> AbelianEnvironmentEvidence:
    applied, compression = apply_abelian_mpo(
        operator,
        state,
        maximum_bond_dimension=maximum_bond_dimension,
        normalize=False,
    )
    expectation = abelian_mps_inner(state, applied)
    norm = abelian_mps_inner(state, state)
    residual = jnp.abs(jnp.imag(expectation))
    return AbelianEnvironmentEvidence(
        expectation,
        norm,
        residual,
        compression.valid
        & jnp.isfinite(jnp.real(expectation))
        & jnp.isfinite(jnp.imag(expectation))
        & jnp.isfinite(norm),
        state.precision.policy_id,
    )


__all__ = [
    "AbelianChainCompressionEvidence",
    "AbelianEnvironmentEvidence",
    "AbelianMatrixProductOperator",
    "AbelianMatrixProductState",
    "abelian_mpo_environment_evidence",
    "abelian_mps_inner",
    "abelian_mps_one_site_expectation",
    "add_abelian_tensors",
    "add_abelian_mpo",
    "add_abelian_mps",
    "adjoint_abelian_tensor",
    "adjoint_abelian_mpo",
    "apply_abelian_mpo",
    "apply_abelian_mpo_exact",
    "canonicalize_abelian_mps",
    "compose_abelian_mpo",
    "compose_abelian_mpo_exact",
    "compress_abelian_mpo",
    "compress_abelian_mps",
    "scale_abelian_mpo",
    "scale_abelian_tensor",
    "scale_abelian_mps",
]

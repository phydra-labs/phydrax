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
from ._abelian import AbelianCharge, AbelianLeg, AbelianTensor
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
    if left.structure_id != right.structure_id:
        raise ValueError("Abelian MPS structures must match for inner products.")
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
) -> AbelianMatrixProductState:
    center_ = int(center)
    if not 0 <= center_ < state.site_count:
        raise ValueError("center is outside the Abelian MPS.")
    precision = state.precision
    tensors = list(state.tensors)
    for site in range(center_):
        tensor = tensors[site]
        blocks = list(tensor.blocks)
        right_leg = tensor.layout.legs[2]
        transfers = []
        for right_sector in range(len(right_leg.charges)):
            routes = [
                index
                for index, sector in enumerate(tensor.layout.sectors)
                if sector[2] == right_sector
            ]
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
        for left_sector in range(len(left_leg.charges)):
            routes = [
                index
                for index, sector in enumerate(tensor.layout.sectors)
                if sector[0] == left_sector
            ]
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


__all__ = [
    "AbelianMatrixProductOperator",
    "AbelianMatrixProductState",
    "abelian_mps_inner",
    "abelian_mps_one_site_expectation",
    "canonicalize_abelian_mps",
]

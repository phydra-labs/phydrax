#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tensor-product commuting cell/node/face/edge transfers for patch AMR."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import adjoint, ArraySpace, transpose
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ._entities import VariablePatchEntityComplex


class CompatibleEntityTransferEvidence(StrictModule, NonTrainableState):
    """Capacity, roundtrip, constant, and commuting evidence for one degree."""

    degree: int = eqx.field(static=True)
    prolongation_routes: int = eqx.field(static=True)
    restriction_routes: int = eqx.field(static=True)
    route_capacity: int = eqx.field(static=True)
    constant_defect: float = eqx.field(static=True)
    roundtrip_defect: float = eqx.field(static=True)
    commuting_defect: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        prolongation_routes: int,
        restriction_routes: int,
        route_capacity: int,
        constant_defect: float,
        roundtrip_defect: float,
        commuting_defect: float,
        /,
    ):
        defects = (
            float(constant_defect),
            float(roundtrip_defect),
            float(commuting_defect),
        )
        if (
            int(degree) < 0
            or min(int(prolongation_routes), int(restriction_routes)) < 0
            or max(int(prolongation_routes), int(restriction_routes))
            > int(route_capacity)
            or any(not np.isfinite(value) or value < 0.0 for value in defects)
        ):
            raise ValueError("Compatible entity transfer evidence is invalid.")
        self.degree = int(degree)
        self.prolongation_routes = int(prolongation_routes)
        self.restriction_routes = int(restriction_routes)
        self.route_capacity = int(route_capacity)
        self.constant_defect = defects[0]
        self.roundtrip_defect = defects[1]
        self.commuting_defect = defects[2]
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "compatible-entity-transfer-evidence",
                "degree": int(degree),
                "prolongation_routes": int(prolongation_routes),
                "restriction_routes": int(restriction_routes),
                "route_capacity": int(route_capacity),
                "defects": defects,
            }
        )


class CompatibleEntityTransfer(StrictModule, NonTrainableState):
    degree: int = eqx.field(static=True)
    prolongation: SparseCoordinateOperator
    restriction: SparseCoordinateOperator
    dual_pullback: object
    hilbert_adjoint: object
    evidence: CompatibleEntityTransferEvidence
    transfer_id: str = eqx.field(static=True)


def _canonical_coordinate(
    orientation: tuple[int, ...],
    coordinate: tuple[int, ...],
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> tuple[int, ...] | None:
    result = list(coordinate)
    tangent = frozenset(orientation)
    for axis, extent in enumerate(shape):
        upper = extent - 1 if axis in tangent else extent
        if periodic[axis] and axis not in tangent:
            result[axis] %= extent
        elif result[axis] < 0 or result[axis] > upper:
            return None
    return tuple(result)


def _pad_operator(
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    coefficients: Sequence[float],
    capacity: int,
    source_space: ArraySpace,
    target_space: ArraySpace,
    operator_id: str,
    /,
) -> SparseCoordinateOperator:
    route_count = len(source_indices)
    if route_count > capacity:
        raise ValueError("Compatible entity transfer route capacity is exceeded.")
    source = np.zeros((capacity,), dtype=np.int32)
    target = np.zeros((capacity,), dtype=np.int32)
    weights = np.zeros((capacity,), dtype=np.float64)
    valid = np.zeros((capacity,), dtype=np.bool_)
    source[:route_count] = source_indices
    target[:route_count] = target_indices
    weights[:route_count] = coefficients
    valid[:route_count] = True
    return SparseCoordinateOperator(
        EdgeRelation(
            source,
            target,
            source_size=source_space.size,
            target_size=target_space.size,
            valid=valid,
        ),
        jnp.asarray(weights, dtype=source_space.dtype),
        source=source_space,
        target=target_space,
        operator_id=operator_id,
    )


def _scipy_matrix(operator: SparseCoordinateOperator, /):
    valid = np.asarray(operator.relation.valid, dtype=np.bool_)
    source = np.asarray(operator.relation.source_indices)[valid]
    target = np.asarray(operator.relation.target_indices)[valid]
    coefficients = np.asarray(operator.coefficients)[valid]
    return sp.coo_matrix(
        (coefficients, (target, source)),
        shape=(operator.target.size, operator.source.size),
    ).tocsr()


def _prolongation_routes(
    coarse_keys,
    fine_keys,
    ratio: int,
    coarse_shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
):
    coarse_by_key = {key: index for index, key in enumerate(coarse_keys)}
    source = []
    target = []
    coefficients = []
    for fine_index, (orientation, fine_coordinate) in enumerate(fine_keys):
        tangent = frozenset(orientation)
        options = []
        tangent_scale = ratio ** (-len(orientation))
        for axis, value in enumerate(fine_coordinate):
            lower = value // ratio
            remainder = value % ratio
            if axis in tangent or remainder == 0:
                options.append(((lower, 1.0),))
            else:
                fraction = remainder / ratio
                upper = lower + 1
                options.append(((lower, 1.0 - fraction), (upper, fraction)))
        for combination in product(*options):
            coordinate = tuple(value for value, _ in combination)
            coordinate = _canonical_coordinate(
                orientation,
                coordinate,
                coarse_shape,
                periodic,
            )
            if coordinate is None:
                continue
            coarse_index = coarse_by_key.get((orientation, coordinate))
            if coarse_index is None:
                continue
            source.append(coarse_index)
            target.append(fine_index)
            coefficients.append(
                tangent_scale * np.prod([weight for _, weight in combination])
            )
    return source, target, coefficients


def _restriction_routes(
    coarse_keys,
    fine_keys,
    ratio: int,
    fine_shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
):
    fine_by_key = {key: index for index, key in enumerate(fine_keys)}
    source = []
    target = []
    coefficients = []
    for coarse_index, (orientation, coarse_coordinate) in enumerate(coarse_keys):
        tangent = tuple(orientation)
        offsets = tuple(product(range(ratio), repeat=len(tangent))) or ((),)
        for offset in offsets:
            coordinate = tuple(
                value * ratio + (offset[tangent.index(axis)] if axis in tangent else 0)
                for axis, value in enumerate(coarse_coordinate)
            )
            coordinate = _canonical_coordinate(
                orientation,
                coordinate,
                fine_shape,
                periodic,
            )
            if coordinate is None:
                continue
            fine_index = fine_by_key.get((orientation, coordinate))
            if fine_index is None:
                continue
            source.append(fine_index)
            target.append(coarse_index)
            coefficients.append(1.0)
    return source, target, coefficients


class CompatibleEntityTransferFamily(StrictModule, NonTrainableState):
    """Joint tensor-product transfer satisfying the discrete de Rham commutator."""

    coarse: VariablePatchEntityComplex
    fine: VariablePatchEntityComplex
    refinement_ratio: int = eqx.field(static=True)
    transfers: tuple[CompatibleEntityTransfer, ...]
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: VariablePatchEntityComplex,
        fine: VariablePatchEntityComplex,
        refinement_ratio: int,
        route_capacities: Sequence[int],
        /,
        *,
        dtype=jnp.float64,
    ):
        ratio = int(refinement_ratio)
        capacities = tuple(route_capacities)
        dimension = coarse.complex.dimension
        if (
            not isinstance(coarse, VariablePatchEntityComplex)
            or not isinstance(fine, VariablePatchEntityComplex)
            or fine.level != coarse.level + 1
            or fine.complex.dimension != dimension
            or ratio <= 1
            or len(capacities) != dimension + 1
            or any(value <= 0 for value in capacities)
        ):
            raise ValueError("Compatible entity transfer family inputs are invalid.")
        dtype_ = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype_, jnp.inexact):
            raise TypeError("Compatible entity transfer dtype must be inexact.")
        coarse_shape = coarse.topology.plan.global_cell_shapes[coarse.level]
        fine_shape = fine.topology.plan.global_cell_shapes[fine.level]
        periodic = coarse.topology.plan.periodic_axes
        raw = []
        for degree, route_capacity in enumerate(capacities):
            coarse_keys = tuple(
                key for key in coarse.entity_keys[degree] if key is not None
            )
            fine_keys = tuple(key for key in fine.entity_keys[degree] if key is not None)
            coarse_space = ArraySpace((coarse.capacity[degree],), dtype=dtype_)
            fine_space = ArraySpace((fine.capacity[degree],), dtype=dtype_)
            p_source, p_target, p_weights = _prolongation_routes(
                coarse_keys,
                fine_keys,
                ratio,
                coarse_shape,
                periodic,
            )
            r_source, r_target, r_weights = _restriction_routes(
                coarse_keys,
                fine_keys,
                ratio,
                fine_shape,
                periodic,
            )
            prolongation = _pad_operator(
                p_source,
                p_target,
                p_weights,
                route_capacity,
                coarse_space,
                fine_space,
                canonical_fingerprint(
                    {
                        "kind": "compatible-entity-prolongation",
                        "coarse": coarse.complex_id,
                        "fine": fine.complex_id,
                        "degree": degree,
                        "routes": {
                            "source": array_tree_fingerprint(p_source),
                            "target": array_tree_fingerprint(p_target),
                            "weights": array_tree_fingerprint(p_weights),
                        },
                    }
                ),
            )
            restriction = _pad_operator(
                r_source,
                r_target,
                r_weights,
                route_capacity,
                fine_space,
                coarse_space,
                canonical_fingerprint(
                    {
                        "kind": "compatible-entity-restriction",
                        "prolongation": prolongation.operator_id,
                        "routes": {
                            "source": array_tree_fingerprint(r_source),
                            "target": array_tree_fingerprint(r_target),
                        },
                    }
                ),
            )
            raw.append(
                (
                    prolongation,
                    restriction,
                    len(p_source),
                    len(r_source),
                    route_capacity,
                )
            )
        prolongation_matrices = tuple(_scipy_matrix(value[0]) for value in raw)
        restriction_matrices = tuple(_scipy_matrix(value[1]) for value in raw)
        transfers = []
        tolerance = 1.0e-12
        for degree, (
            prolongation,
            restriction,
            prolongation_count,
            restriction_count,
            capacity,
        ) in enumerate(raw):
            coarse_active = np.asarray(
                coarse.complex.entities(degree).active_mask, dtype=np.bool_
            )
            fine_active = np.asarray(
                fine.complex.entities(degree).active_mask, dtype=np.bool_
            )
            if degree == 0:
                constant = prolongation_matrices[degree] @ coarse_active.astype("float64")
                constant_defect = float(
                    np.max(np.abs(constant[fine_active] - 1.0), initial=0.0)
                )
            else:
                constant_defect = 0.0
            roundtrip = (
                restriction_matrices[degree] @ prolongation_matrices[degree]
            ).toarray()
            identity = np.eye(coarse.capacity[degree])
            supported_coarse = (
                restriction_matrices[degree].getnnz(axis=1) > 0
            ) & coarse_active
            active_roundtrip = np.abs((roundtrip - identity)[supported_coarse])
            roundtrip_defect = float(
                0.0 if active_roundtrip.size == 0 else np.max(active_roundtrip)
            )
            if degree < dimension:
                coarse_derivative = (
                    coarse.complex.incidences[degree].scipy_boundary().transpose()
                )
                fine_derivative = (
                    fine.complex.incidences[degree].scipy_boundary().transpose()
                )
                commutator = (
                    fine_derivative @ prolongation_matrices[degree]
                    - prolongation_matrices[degree + 1] @ coarse_derivative
                )
                commuting_defect = float(np.max(np.abs(commutator.data), initial=0.0))
            else:
                commuting_defect = 0.0
            if max(constant_defect, roundtrip_defect, commuting_defect) > tolerance:
                raise ValueError(
                    "Variable patch entity transfer failed constant/roundtrip/commuting qualification."
                )
            evidence = CompatibleEntityTransferEvidence(
                degree,
                prolongation_count,
                restriction_count,
                capacity,
                constant_defect,
                roundtrip_defect,
                commuting_defect,
            )
            transfer_id = canonical_fingerprint(
                {
                    "kind": "compatible-variable-patch-entity-transfer",
                    "degree": degree,
                    "prolongation": prolongation.operator_id,
                    "restriction": restriction.operator_id,
                    "evidence": evidence.evidence_id,
                }
            )
            transfers.append(
                CompatibleEntityTransfer(
                    degree,
                    prolongation,
                    restriction,
                    transpose(prolongation),
                    adjoint(prolongation),
                    evidence,
                    transfer_id,
                )
            )
        self.coarse = coarse
        self.fine = fine
        self.refinement_ratio = ratio
        self.transfers = tuple(transfers)
        self.family_id = canonical_fingerprint(
            {
                "kind": "compatible-variable-patch-entity-transfer-family",
                "coarse": coarse.complex_id,
                "fine": fine.complex_id,
                "ratio": ratio,
                "transfers": [transfer.transfer_id for transfer in transfers],
            }
        )

    def transfer(self, degree: int, /) -> CompatibleEntityTransfer:
        index = int(degree)
        if index < 0 or index >= len(self.transfers):
            raise ValueError("Compatible entity transfer degree is out of range.")
        return self.transfers[index]


__all__ = [
    "CompatibleEntityTransfer",
    "CompatibleEntityTransferEvidence",
    "CompatibleEntityTransferFamily",
]

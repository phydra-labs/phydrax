#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific coordinate-scope identity shared by preparation caches."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import final, TYPE_CHECKING

import jax
import numpy as np

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from ._coordinate_operands import (
    ContainerOwners,
    coordinate_input_owner_fields,
    CoordinateOperandProfile,
    OwnerFields,
    require_coordinate_input_containers,
    require_coordinate_input_owners,
)


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh
    from ._coordinate_budget import CoordinateEnclosureBudget


def _geometry_id(geometry: CellGeometrySpec, /) -> str:
    # Coordinate identity → geometry validity → source enclosure → coordinate identity.
    from ._cell_geometry_validity import cell_geometry_id

    return cell_geometry_id(geometry)


def coordinate_source_bank_key(geometry: CellGeometrySpec, /) -> tuple[str, str]:
    """Bind the source bank to its declared geometry and current numerical tree."""
    geometry_id = _geometry_id(geometry)
    live_tree_id = (
        geometry.storage_id
        if geometry.storage_id is not None
        else canonical_fingerprint(array_tree_fingerprint(geometry))
    )
    return geometry_id, live_tree_id


def coordinate_scope_key(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[str, str, str]:
    """Scientific source ownership and every live coordinate/reference bank."""

    geometry_id = _geometry_id(geometry)
    live_tree_id = (
        geometry.storage_id
        if geometry.storage_id is not None
        else canonical_fingerprint(array_tree_fingerprint((mesh, geometry)))
    )
    return mesh.mesh_id, geometry_id, live_tree_id


@dataclass(frozen=True, slots=True)
class CoordinateIdentityInputs:
    """Actual owners and banks authenticating a previously computed identity.

    ``shared`` is the separately captured rest of the owner graph; rechecking
    these inputs always rechecks it as well.
    """

    owner_fields: OwnerFields
    containers: ContainerOwners
    immutable_banks: tuple[jax.Array, ...]
    mutable_banks: tuple[np.ndarray, ...]
    mutable_write_flags: tuple[bool, ...]
    mutable_content_id: str | None
    shared: CoordinateIdentityInputs | None = None


@dataclass(frozen=True, slots=True)
class ElementIdentityAdmission:
    """One exact original-formula identity owned by one explicit request budget."""

    owner: object
    identity: tuple[str, str]
    inputs: CoordinateIdentityInputs


@final
class OperationElementAdmission:
    """An element admission its authenticating pure operation carries to callees.

    Only that operation constructs and closes it. A carried token proves an
    unchanged owner only while the operation runs, for its exact owner, and
    while it is still the budget's registered admission; otherwise callees
    authenticate in full.
    """

    __slots__ = ("_admission", "_live")

    def __init__(self, admission: ElementIdentityAdmission, /) -> None:
        self._admission = admission
        self._live = True

    def close(self) -> None:
        self._live = False

    def identity(
        self, owner: object, budget: CoordinateEnclosureBudget, /
    ) -> tuple[str, str] | None:
        """The authenticated identity of ``owner``, or ``None`` when not current."""
        admission = self._admission
        if (
            self._live
            and admission.owner is owner
            and budget.element_identity_admission_cache.get(id(owner)) is admission
        ):
            return admission.identity
        return None


def _mutable_coordinate_banks_id(
    banks: tuple[np.ndarray, ...], budget: CoordinateEnclosureBudget, /
) -> str | None:
    if not banks:
        return None
    # This is the same logical-byte/shape/dtype authentication used by stored
    # record admissions. It is only a mutable-bank seal, never the element ID.
    with budget.temporary_scope():
        budget.reserve(
            sum(1 + bank.size for bank in banks),
            256 + 128 * len(banks) + 2 * max(bank.nbytes for bank in banks),
        )
        return logical_array_value_collection_digest(
            {f"mutable/{index:06d}": bank for index, bank in enumerate(banks)}
        )


def _coordinate_identity_inputs(
    owner: object,
    shared: CoordinateIdentityInputs | None,
    budget: CoordinateEnclosureBudget,
    /,
) -> CoordinateIdentityInputs | None:
    arrays: list[jax.Array | np.ndarray] = []
    containers: list[tuple[list[object], tuple[object, ...]]] = []
    known = (
        ()
        if shared is None
        else (
            CoordinateOperandProfile(
                (),
                MappingProxyType({id(item[0]): item[0] for item in shared.owner_fields}),
                shared.owner_fields,
                0,
            ),
        )
    )
    owners = coordinate_input_owner_fields(
        (owner,), budget, known=known, arrays=arrays, containers=containers
    )
    if owners is None:
        return None
    immutable = tuple(bank for bank in arrays if isinstance(bank, jax.Array))
    mutable = tuple(bank for bank in arrays if isinstance(bank, np.ndarray))
    budget.reserve(len(immutable))
    if any(bank.is_deleted() for bank in immutable):
        raise ValueError("element_identity must retain live actual JAX banks")
    return CoordinateIdentityInputs(
        owners,
        tuple(containers),
        immutable,
        mutable,
        tuple(bank.flags.writeable for bank in mutable),
        _mutable_coordinate_banks_id(mutable, budget),
        shared,
    )


def coordinate_identity_inputs(
    owner: object, budget: CoordinateEnclosureBudget, /
) -> CoordinateIdentityInputs | None:
    """Capture the shared actual-field invariant, with no wrapper transparency."""
    return _coordinate_identity_inputs(owner, None, budget)


def operation_coordinate_identity_inputs(
    owner: object,
    shared: CoordinateIdentityInputs,
    budget: CoordinateEnclosureBudget,
    /,
) -> CoordinateIdentityInputs | None:
    """Capture ``owner`` whose remaining graph ``shared`` this operation captured.

    ``shared`` must have been captured by the same pure operation on the same
    budget. Its owners are skipped by identity, not rescanned; the returned
    inputs retain it, so any later reuse rechecks the complete graph.
    """
    return _coordinate_identity_inputs(owner, shared, budget)


def require_coordinate_identity_inputs(
    inputs: CoordinateIdentityInputs,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    verified: dict[int, CoordinateIdentityInputs] | None = None,
) -> bool:
    """Recheck complete current owners and NumPy content, and actual JAX liveness.

    ``verified`` belongs to one pure operation and is dropped with it. An
    inputs graph (an owner's own capture or a shared one) that this operation
    already rechecked successfully is not rechecked again; it is recorded only
    after its complete recheck succeeds.
    """
    if verified is not None and verified.get(id(inputs)) is inputs:
        return True
    if not (
        require_coordinate_input_owners(inputs.owner_fields, budget)
        and require_coordinate_input_containers(inputs.containers, budget)
    ):
        return False
    budget.reserve(len(inputs.immutable_banks))
    if any(bank.is_deleted() for bank in inputs.immutable_banks):
        raise ValueError("element_identity must retain live actual JAX banks")
    if not (
        tuple(bank.flags.writeable for bank in inputs.mutable_banks)
        == inputs.mutable_write_flags
        and _mutable_coordinate_banks_id(inputs.mutable_banks, budget)
        == inputs.mutable_content_id
        and (
            inputs.shared is None
            or require_coordinate_identity_inputs(inputs.shared, budget, verified=verified)
        )
    ):
        return False
    if verified is not None:
        verified[id(inputs)] = inputs
    return True

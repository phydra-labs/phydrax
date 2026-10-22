#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific coordinate-scope identity shared by preparation caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    """Actual owners and banks authenticating a previously computed identity."""

    owner_fields: OwnerFields
    containers: ContainerOwners
    immutable_banks: tuple[jax.Array, ...]
    mutable_banks: tuple[np.ndarray, ...]
    mutable_write_flags: tuple[bool, ...]
    mutable_content_id: str | None


@dataclass(frozen=True, slots=True)
class ElementIdentityAdmission:
    """One exact original-formula identity owned by one explicit request budget."""

    owner: object
    identity: tuple[str, str]
    inputs: CoordinateIdentityInputs


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


def coordinate_identity_inputs(
    owner: object, budget: CoordinateEnclosureBudget, /
) -> CoordinateIdentityInputs | None:
    """Capture the shared actual-field invariant, with no wrapper transparency."""
    arrays: list[jax.Array | np.ndarray] = []
    containers: list[tuple[list[object], tuple[object, ...]]] = []
    owners = coordinate_input_owner_fields(
        (owner,), budget, arrays=arrays, containers=containers
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
    )


def require_coordinate_identity_inputs(
    inputs: CoordinateIdentityInputs, budget: CoordinateEnclosureBudget, /
) -> bool:
    """Recheck complete current owners and NumPy content, and actual JAX liveness."""
    if not (
        require_coordinate_input_owners(inputs.owner_fields, budget)
        and require_coordinate_input_containers(inputs.containers, budget)
    ):
        return False
    budget.reserve(len(inputs.immutable_banks))
    if any(bank.is_deleted() for bank in inputs.immutable_banks):
        raise ValueError("element_identity must retain live actual JAX banks")
    return (
        tuple(bank.flags.writeable for bank in inputs.mutable_banks)
        == inputs.mutable_write_flags
        and _mutable_coordinate_banks_id(inputs.mutable_banks, budget)
        == inputs.mutable_content_id
    )

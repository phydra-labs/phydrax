#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellMesh,
    FiniteElementAdaptationMap,
    FiniteElementTransferBundle,
    refine_triangles_local,
)
from ..equations import FiniteElementMaterialTransaction
from ._finite_element_schedule import FiniteElementAcceptedState


class FiniteElementTopologyResult(StrictModule, NonTrainableState):
    state: FiniteElementAcceptedState
    mesh: CellMesh
    adaptation: FiniteElementAdaptationMap | None
    transfer: FiniteElementTransferBundle | None
    committed: Array
    diagnostics: object


class FiniteElementTopologyTransaction(StrictModule, NonTrainableState):
    """Build and certify a local-mesh candidate before atomic promotion."""

    certify: Callable
    material_transfer: Callable | None
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        certify: Callable,
        /,
        *,
        material_transfer: Callable | None = None,
        transaction_id: str = "finite-element-topology-transaction",
    ):
        if not callable(certify):
            raise TypeError("certify must be callable.")
        if material_transfer is not None and not callable(material_transfer):
            raise TypeError("material_transfer must be callable or None.")
        identifier = str(transaction_id)
        if not identifier:
            raise ValueError("transaction_id must be non-empty.")
        self.certify = certify
        self.material_transfer = material_transfer
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-topology-transaction",
                "declared_id": identifier,
                "has_material_transfer": material_transfer is not None,
            }
        )

    def execute(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        marked_cell_ids: ArrayLike,
        args: object = None,
        /,
    ) -> FiniteElementTopologyResult:
        if not isinstance(accepted, FiniteElementAcceptedState):
            raise TypeError("accepted must be FiniteElementAcceptedState.")
        if mesh.topology_id != accepted.topology_id:
            raise ValueError("Accepted state and topology transaction mesh disagree.")
        candidate_mesh, adaptation, transfer = refine_triangles_local(
            mesh,
            marked_cell_ids,
            numeric_version=f"adapted-{accepted.state_version + 1}",
        )
        candidate_fields = []
        for field in accepted.fields:
            if field.shape[0] != transfer.primal.shape[1]:
                raise ValueError(
                    "Automatic topology transfer currently requires vertex P1 fields."
                )
            candidate_fields.append(
                jnp.tensordot(transfer.primal, field, axes=((1,), (0,)))
            )
        candidate_materials: FiniteElementMaterialTransaction | None
        if accepted.materials is None:
            candidate_materials = None
        elif self.material_transfer is None:
            return FiniteElementTopologyResult(
                state=accepted,
                mesh=mesh,
                adaptation=None,
                transfer=None,
                committed=jnp.asarray(False),
                diagnostics="material-transfer-policy-required",
            )
        else:
            transferred = self.material_transfer(
                accepted.materials,
                adaptation,
                args,
            )
            if not isinstance(transferred, FiniteElementMaterialTransaction):
                return FiniteElementTopologyResult(
                    state=accepted,
                    mesh=mesh,
                    adaptation=None,
                    transfer=None,
                    committed=jnp.asarray(False),
                    diagnostics="material-transfer-rejected",
                )
            candidate_materials = transferred
        certified = self.certify(
            candidate_mesh,
            tuple(candidate_fields),
            candidate_materials,
            adaptation,
            args,
        )
        if not bool(jnp.asarray(certified)):
            return FiniteElementTopologyResult(
                state=accepted,
                mesh=mesh,
                adaptation=None,
                transfer=None,
                committed=jnp.asarray(False),
                diagnostics="candidate-certification-rejected",
            )
        promoted = FiniteElementAcceptedState(
            candidate_fields,
            accepted.time,
            accepted.step,
            candidate_mesh.topology_id,
            f"{accepted.prepared_id}:adapted:{adaptation.adaptation_id}",
            f"{accepted.compilation_id}:adapted:{adaptation.adaptation_id}",
            materials=candidate_materials,
            schedule_cursor=accepted.schedule_cursor,
            state_version=accepted.state_version + 1,
        )
        return FiniteElementTopologyResult(
            state=promoted,
            mesh=candidate_mesh,
            adaptation=adaptation,
            transfer=transfer,
            committed=jnp.asarray(True),
            diagnostics="committed",
        )


__all__ = ["FiniteElementTopologyResult", "FiniteElementTopologyTransaction"]

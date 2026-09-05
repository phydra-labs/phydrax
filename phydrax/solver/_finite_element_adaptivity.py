#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, FiniteElementTransferBundle
from ..discretization.fem import (
    FiniteElementHPEpoch,
    FiniteElementHPGeometry,
    FiniteElementHPTopology,
    FiniteElementHPTransaction,
    prepare_finite_element_hp_epoch,
)
from ..equations import MaterialTransaction
from ..meshing import CellMeshTransition
from ._finite_element_schedule import FiniteElementAcceptedState


class FiniteElementTopologyResult(StrictModule, NonTrainableState):
    state: FiniteElementAcceptedState
    mesh: CellMesh
    transition: CellMeshTransition | None
    transfer: FiniteElementTransferBundle | None
    committed: Array
    diagnostics: object


class FiniteElementHPTopologyResult(StrictModule, NonTrainableState):
    state: FiniteElementAcceptedState
    epoch: FiniteElementHPEpoch
    transaction: FiniteElementHPTransaction | None
    auxiliary_state: tuple[tuple[str, Array], ...]
    integrator_state: tuple[tuple[str, Array], ...]
    committed: Array
    diagnostics: object


class FiniteElementTopologyTransaction(StrictModule, NonTrainableState):
    """Build and certify a local-mesh candidate before atomic promotion."""

    certify: Callable
    material_transfer: Callable | None
    field_transfer: Callable | None
    history_transfer: Callable | None
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        certify: Callable,
        /,
        *,
        material_transfer: Callable | None = None,
        field_transfer: Callable | None = None,
        history_transfer: Callable | None = None,
        transaction_id: str = "finite-element-topology-transaction",
    ):
        if not callable(certify):
            raise TypeError("certify must be callable.")
        if material_transfer is not None and not callable(material_transfer):
            raise TypeError("material_transfer must be callable or None.")
        if field_transfer is not None and not callable(field_transfer):
            raise TypeError("field_transfer must be callable or None.")
        if history_transfer is not None and not callable(history_transfer):
            raise TypeError("history_transfer must be callable or None.")
        identifier = str(transaction_id)
        if not identifier:
            raise ValueError("transaction_id must be non-empty.")
        self.certify = certify
        self.material_transfer = material_transfer
        self.field_transfer = field_transfer
        self.history_transfer = history_transfer
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-topology-transaction",
                "declared_id": identifier,
                "has_material_transfer": material_transfer is not None,
                "has_field_transfer": field_transfer is not None,
                "has_history_transfer": history_transfer is not None,
            }
        )

    def execute(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        transition: CellMeshTransition,
        transfer: FiniteElementTransferBundle,
        args: object = None,
        /,
    ) -> FiniteElementTopologyResult:
        if not isinstance(accepted, FiniteElementAcceptedState):
            raise TypeError("accepted must be FiniteElementAcceptedState.")
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if not isinstance(transition, CellMeshTransition):
            raise TypeError("transition must be CellMeshTransition.")
        if not isinstance(transfer, FiniteElementTransferBundle):
            raise TypeError("transfer must be FiniteElementTransferBundle.")
        if (
            mesh.topology_id != accepted.topology_id
            or transition.source_mesh_id != mesh.mesh_id
            or transition.source_topology_id != mesh.topology_id
        ):
            raise ValueError("Accepted state and topology transition disagree.")
        candidate_mesh = transition.target.mesh
        candidate_fields = []
        for field in accepted.fields:
            if field.shape[0] != transfer.primal.shape[1]:
                raise ValueError(
                    "Automatic topology transfer currently requires vertex P1 fields."
                )
            candidate_fields.append(
                jnp.tensordot(transfer.primal, field, axes=((1,), (0,)))
            )
        candidate_materials: MaterialTransaction | None
        if accepted.materials is None:
            candidate_materials = None
        elif self.material_transfer is None:
            return FiniteElementTopologyResult(
                state=accepted,
                mesh=mesh,
                transition=None,
                transfer=None,
                committed=jnp.asarray(False),
                diagnostics="material-transfer-policy-required",
            )
        else:
            transferred = self.material_transfer(
                accepted.materials,
                transition.lineage,
                args,
            )
            if not isinstance(transferred, MaterialTransaction):
                return FiniteElementTopologyResult(
                    state=accepted,
                    mesh=mesh,
                    transition=None,
                    transfer=None,
                    committed=jnp.asarray(False),
                    diagnostics="material-transfer-rejected",
                )
            candidate_materials = transferred
        certified = self.certify(
            candidate_mesh,
            tuple(candidate_fields),
            candidate_materials,
            transition.lineage,
            args,
        )
        if not bool(jnp.asarray(certified)):
            return FiniteElementTopologyResult(
                state=accepted,
                mesh=mesh,
                transition=None,
                transfer=None,
                committed=jnp.asarray(False),
                diagnostics="candidate-certification-rejected",
            )
        promoted = FiniteElementAcceptedState(
            candidate_fields,
            accepted.time,
            accepted.step,
            candidate_mesh.topology_id,
            f"{accepted.prepared_id}:transition:{transition.transition_id}",
            f"{accepted.compilation_id}:transition:{transition.transition_id}",
            materials=candidate_materials,
            schedule_cursor=accepted.schedule_cursor,
            state_version=accepted.state_version + 1,
        )
        return FiniteElementTopologyResult(
            state=promoted,
            mesh=candidate_mesh,
            transition=transition,
            transfer=transfer,
            committed=jnp.asarray(True),
            diagnostics="committed",
        )

    def execute_hp(
        self,
        accepted: FiniteElementAcceptedState,
        transaction: FiniteElementHPTransaction,
        args: object = None,
        /,
        *,
        auxiliary_state: Sequence[tuple[str, ArrayLike]] = (),
        integrator_state: Sequence[tuple[str, ArrayLike]] = (),
    ) -> FiniteElementHPTopologyResult:
        """Transfer, certify, and atomically promote one prepared hp candidate."""

        if not isinstance(accepted, FiniteElementAcceptedState):
            raise TypeError("accepted must be FiniteElementAcceptedState.")
        if not isinstance(transaction, FiniteElementHPTransaction):
            raise TypeError("transaction must be FiniteElementHPTransaction.")
        if accepted.topology_id != transaction.accepted.topology.topology_id:
            raise ValueError("Accepted state and hp transaction topology disagree.")
        transfers = transaction.p_transfers + transaction.h_transfers
        if self.field_transfer is None:
            if len(transfers) != len(accepted.fields):
                raise ValueError(
                    "Automatic hp state transfer requires one transfer per field."
                )
            candidate_fields = tuple(
                transfer.apply_mass_projection(field)
                for transfer, field in zip(transfers, accepted.fields, strict=True)
            )
        else:
            candidate_fields = tuple(
                jnp.asarray(value)
                for value in self.field_transfer(
                    accepted.fields,
                    transaction,
                    args,
                )
            )
        candidate_materials: MaterialTransaction | None
        if accepted.materials is None:
            candidate_materials = None
        elif self.material_transfer is None:
            return FiniteElementHPTopologyResult(
                accepted,
                transaction.accepted,
                None,
                tuple((str(name), jnp.asarray(value)) for name, value in auxiliary_state),
                tuple(
                    (str(name), jnp.asarray(value)) for name, value in integrator_state
                ),
                jnp.asarray(False),
                "material-transfer-policy-required",
            )
        else:
            candidate_materials = self.material_transfer(
                accepted.materials,
                transaction,
                args,
            )
            if not isinstance(candidate_materials, MaterialTransaction):
                return FiniteElementHPTopologyResult(
                    accepted,
                    transaction.accepted,
                    None,
                    tuple(
                        (str(name), jnp.asarray(value)) for name, value in auxiliary_state
                    ),
                    tuple(
                        (str(name), jnp.asarray(value))
                        for name, value in integrator_state
                    ),
                    jnp.asarray(False),
                    "material-transfer-rejected",
                )
        auxiliary = tuple(
            (str(name), jnp.asarray(value)) for name, value in auxiliary_state
        )
        integrator = tuple(
            (str(name), jnp.asarray(value)) for name, value in integrator_state
        )
        if self.history_transfer is not None:
            transferred_history = self.history_transfer(
                auxiliary,
                integrator,
                transaction,
                args,
            )
            auxiliary = tuple(
                (str(name), jnp.asarray(value)) for name, value in transferred_history[0]
            )
            integrator = tuple(
                (str(name), jnp.asarray(value)) for name, value in transferred_history[1]
            )
        elif auxiliary or integrator:
            return FiniteElementHPTopologyResult(
                accepted,
                transaction.accepted,
                None,
                auxiliary,
                integrator,
                jnp.asarray(False),
                "history-transfer-policy-required",
            )
        certified = self.certify(
            transaction.candidate,
            candidate_fields,
            candidate_materials,
            transaction,
            args,
        )
        if not bool(jnp.asarray(certified)):
            return FiniteElementHPTopologyResult(
                accepted,
                transaction.accepted,
                None,
                auxiliary,
                integrator,
                jnp.asarray(False),
                "candidate-certification-rejected",
            )
        promoted = FiniteElementAcceptedState(
            candidate_fields,
            accepted.time,
            accepted.step,
            transaction.candidate.topology.topology_id,
            transaction.candidate.epoch_id,
            f"{accepted.compilation_id}:hp:{transaction.transaction_id}",
            materials=candidate_materials,
            schedule_cursor=accepted.schedule_cursor,
            state_version=accepted.state_version + 1,
        )
        return FiniteElementHPTopologyResult(
            promoted,
            transaction.candidate,
            transaction,
            auxiliary,
            integrator,
            jnp.asarray(True),
            "committed",
        )


def write_finite_element_hp_epoch(
    path: str | Path,
    epoch: FiniteElementHPEpoch,
    /,
) -> None:
    """Persist the canonical forest and geometry needed to reconstruct one hp epoch."""

    if not isinstance(epoch, FiniteElementHPEpoch):
        raise TypeError("epoch must be FiniteElementHPEpoch.")
    field_name = ""
    conformity = "H1"
    component_shape: tuple[int, ...] = ()
    if epoch.discretization is not None:
        field_name = epoch.discretization.field_spaces[0].name
        conformity = epoch.discretization.elements[0][0].conformity
        component_shape = tuple(
            int(value)
            for value in epoch.discretization.field_spaces[0]
            .vector_space.structure()
            .shape[1:]
        )
    metadata = {
        "cell_kind": epoch.topology.cell_kind,
        "topology_id": epoch.topology.topology_id,
        "field_name": field_name,
        "conformity": conformity,
        "component_shape": list(component_shape),
    }
    np.savez(
        Path(path),
        metadata=np.asarray(json.dumps(metadata)),
        cell_global_ids=np.asarray(epoch.topology.cell_global_ids),
        allocated=np.asarray(epoch.topology.allocated),
        active=np.asarray(epoch.topology.active),
        cell_degrees=np.asarray(epoch.topology.cell_degrees),
        root_cell_ids=np.asarray(epoch.topology.root_cell_ids),
        path_codes=np.asarray(epoch.topology.path_codes),
        levels=np.asarray(epoch.topology.levels),
        parent_slots=np.asarray(epoch.topology.parent_slots),
        child_slots=np.asarray(epoch.topology.child_slots),
        child_valid=np.asarray(epoch.topology.child_valid),
        cell_vertices=np.asarray(epoch.geometry.cell_vertices),
        reference_lower=np.asarray(epoch.geometry.reference_lower),
        reference_upper=np.asarray(epoch.geometry.reference_upper),
        allow_pickle=False,
    )


def read_finite_element_hp_epoch(path: str | Path, /) -> FiniteElementHPEpoch:
    """Reconstruct one canonical hp epoch from `write_finite_element_hp_epoch`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        topology = FiniteElementHPTopology(
            metadata["cell_kind"],
            metadata["topology_id"],
            archive["cell_global_ids"],
            archive["allocated"],
            archive["active"],
            archive["cell_degrees"],
            root_cell_ids=archive["root_cell_ids"],
            path_codes=archive["path_codes"],
            levels=archive["levels"],
            parent_slots=archive["parent_slots"],
            child_slots=archive["child_slots"],
            child_valid=archive["child_valid"],
        )
        geometry = FiniteElementHPGeometry(
            topology,
            archive["cell_vertices"],
            archive["reference_lower"],
            archive["reference_upper"],
        )
    field_name = str(metadata["field_name"])
    if field_name:
        return prepare_finite_element_hp_epoch(
            topology,
            geometry,
            field_name,
            conformity=metadata["conformity"],
            component_shape=tuple(metadata["component_shape"]),
        )
    from ..discretization.fem import (
        finite_element_hp_interface_plan,
        hp_active_cell_mesh,
    )

    mesh, _, _ = hp_active_cell_mesh(topology, geometry)
    return FiniteElementHPEpoch(
        mesh,
        topology,
        geometry,
        finite_element_hp_interface_plan(topology, geometry),
    )


__all__ = [
    "FiniteElementHPTopologyResult",
    "read_finite_element_hp_epoch",
    "FiniteElementTopologyResult",
    "FiniteElementTopologyTransaction",
    "write_finite_element_hp_epoch",
]

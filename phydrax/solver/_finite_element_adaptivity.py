#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    current_native_execution_scope,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellMesh,
    FieldEpochTransition,
    TopologyEpoch,
    TopologyEpochTransition,
)
from ..discretization._cell_geometry_transition_types import MaterialChartDeformation
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._nested_reference import PreparedNestedReference
from ..discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementFieldTransfer,
    FiniteElementHPEpoch,
    FiniteElementHPGeometry,
    FiniteElementHPTopology,
    FiniteElementHPTransaction,
    FiniteElementPlan,
    prepare_finite_element_hp_epoch,
    prepare_l2_projection_target,
    prepare_projection_field_transfer,
    prepare_source_realization_field_transfer,
    SourceRealizationFieldSemantics,
)
from ..discretization.fem._topology_transfer import (
    prepare_shared_nested_field_transfer,
)
from ..discretization.finite_volume._remap_evidence import (
    MappedSurfaceChartRemapEvidence,
    PreparedUnstructuredConservativeRemap,
)
from ..equations import MaterialTransaction
from ..geometry import CommonRefinementPolicy, prepare_common_refinement
from ..lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionRole,
    CompositionTransport,
)
from ..meshing import CellMeshingResult, EntityLineageKind, MeshAdaptationResult
from ..meshing._adaptation import GeometryRealizationMeshAdaptation
from ..meshing._decision import PhysicalErrorEvidence, SolverAwareDecision
from ..meshing._measurements import NativeExecutionRecord
from ..meshing._native_preparation import native_preparation_ledger
from ..typing import checked, parse
from ._finite_element_schedule import FiniteElementAcceptedState


class MaterialTopologyTransferResult(StrictModule, NonTrainableState):
    """Material state and the prepared remap owners used to produce it."""

    materials: MaterialTransaction
    remaps: tuple[PreparedUnstructuredConservativeRemap, ...]


_OWNER = "finite-element-topology-transaction"
_PARTITION = "serial"


class FiniteElementTopologyResult(StrictModule, NonTrainableState):
    """Promoted or retained state of one mesh-adaptation rebind.

    ``adaptation`` is set only when committed. ``receipt`` is the
    ``CompositionRebindReceipt`` of the staged rebind (``None`` when refused
    before staging); ``transfers`` are the prepared field transfers with their
    semantics, geometry binding, and evidence.
    Chart transactions additionally retain ``material_remaps`` and bind all
    consumer reports to ``execution_evidence`` only after their shared
    execution allowance ends, including physical certification and publication.
    """

    state: FiniteElementAcceptedState
    mesh: CellMesh
    adaptation: MeshAdaptationResult | None
    committed: Array
    receipt: CompositionRebindReceipt | None
    transfers: tuple[FiniteElementFieldTransfer, ...]
    diagnostics: str = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord | None = None
    material_remaps: tuple[PreparedUnstructuredConservativeRemap, ...] = ()


def _space_realizes_result(
    space: FiniteElementDiscretization,
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> bool:
    mesh = result.mesh
    geometry = result.geometry
    elements, routes, values = geometry.resolve(mesh, budget=budget)
    return (
        space.mesh.topology_id == mesh.topology_id
        and tuple(element.element_id for element in space.coordinate_elements)
        == tuple(element.element_id for element in elements)
        and all(
            np.array_equal(np.asarray(actual), np.asarray(expected))
            for actual, expected in zip(space.coordinate_dofs, routes, strict=True)
        )
        and np.array_equal(
            np.asarray(space.default_runtime.coordinates).view(np.uint64),
            np.asarray(values).view(np.uint64),
        )
    )


def _topology_decision_admission(
    adaptation: MeshAdaptationResult,
    decision: SolverAwareDecision,
    has_reanalysis: bool,
    /,
) -> str | None:
    """Validate the selected physical action before any transfer preparation."""
    from ..meshing._adaptation import (
        MarkedMeshAdaptation,
        MetricMeshAdaptation,
        RelocationMeshAdaptation,
    )
    from ..meshing._decision import AdaptationAction

    selected = decision.require_selected(
        adaptation.source.result_id,
        adaptation.result_id,
        adaptation.target.result_id,
    )
    match adaptation.request:
        case MarkedMeshAdaptation():
            action = AdaptationAction.H
        case MetricMeshAdaptation():
            action = AdaptationAction.METRIC
        case RelocationMeshAdaptation():
            action = AdaptationAction.RELOCATION
        case GeometryRealizationMeshAdaptation():
            action = AdaptationAction.GEOMETRY_ORDER
        case _:
            raise ValueError(
                f"MeshAdaptationResult.request must support physical decision admission; got {type(adaptation.request).__name__}"
            )
    if (
        selected.action != action
        or selected.feasibility.route_id != adaptation.route.value
    ):
        raise ValueError(
            f"SolverAwareDecision.selected must match action={action!r} and route={adaptation.route.value!r}; got action={selected.action!r}, route={selected.feasibility.route_id!r}"
        )
    if not adaptation.status.converged:
        return "native-admission-rejected"
    if not has_reanalysis:
        return "independent-reanalysis-required"
    return None


def _require_topology_candidate_layouts(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    adaptation: MeshAdaptationResult,
    decision: SolverAwareDecision,
    compiled_layout_id: str | None,
    /,
) -> None:
    """Bind the prepared coordinate maps and compiled layouts to admission."""
    if any(
        prepared.default_runtime.geometry_layout_id != result.geometry.geometry_layout_id
        or not np.array_equal(
            prepared.default_runtime.coordinates, result.geometry.coordinates
        )
        for prepared, result in (
            (source, adaptation.source),
            (target, adaptation.target),
        )
    ):
        raise ValueError(
            f"FiniteElementTopologyTransaction.spaces must preserve admitted coordinate maps; got source_layout={source.default_runtime.geometry_layout_id!r}, target_layout={target.default_runtime.geometry_layout_id!r}"
        )
    selected = decision.selected
    if selected is None:
        raise ValueError(
            f"SolverAwareDecision.selected must be present for native execution; got {selected!r}"
        )
    feasibility = selected.feasibility
    layouts = tuple(
        sorted((space.name, space.layout.layout_id) for space in target.field_spaces)
    )
    if (
        feasibility.cell_families
        != tuple(sorted({block.cell_kind for block in adaptation.target.mesh.blocks}))
        or feasibility.geometry_layout_id != adaptation.target.geometry.geometry_layout_id
        or feasibility.field_layouts != layouts
        or feasibility.dofs != sum(space.layout.size for space in target.field_spaces)
        or compiled_layout_id is None
        or feasibility.compiled_layout_id != compiled_layout_id
        or feasibility.geometry_certificate_id != adaptation.target.audit.report_id
        or feasibility.topology_certificate_id != adaptation.target.audit.report_id
    ):
        raise ValueError(
            f"SolverAwareDecision.selected.feasibility must match the candidate certified compiled layouts; got {feasibility!r}, compiled_layout_id={compiled_layout_id!r}"
        )


def _adopt_topology_material_remaps(
    materials: MaterialTopologyTransferResult | MaterialTransaction | None,
    adaptation: MeshAdaptationResult,
    material_remaps: list[PreparedUnstructuredConservativeRemap] | None,
    source_workspace: NativeHostStorageWorkspace | None,
    /,
) -> MaterialTransaction | None:
    """Validate borrowed remap ownership before retaining material storage."""
    match materials:
        case MaterialTopologyTransferResult():
            if material_remaps is None:
                if materials.remaps:
                    raise ValueError(
                        f"FiniteElementTopologyTransaction.material_remaps must provide a chart transaction scope for remap owners; got {material_remaps!r}"
                    )
            else:
                geometry_transition = adaptation.geometry_transition
                if geometry_transition is None:
                    raise ValueError(
                        f"MeshAdaptationResult.geometry_transition must be present for material remap owners; got {geometry_transition!r}"
                    )
                for owner in materials.remaps:
                    match owner.route:
                        case MappedSurfaceChartRemapEvidence() as chart:
                            if (
                                chart.deformation.deformation_id
                                != geometry_transition.chart_correspondence_id
                            ):
                                raise ValueError(
                                    f"MappedSurfaceChartRemapEvidence.deformation.deformation_id must equal {geometry_transition.chart_correspondence_id!r}; got {chart.deformation.deformation_id!r}"
                                )
                        case _:
                            raise ValueError(
                                f"PreparedUnstructuredConservativeRemap.route must bind the chart transition; got {type(owner.route).__name__}"
                            )
                    if owner.execution_evidence is not None:
                        raise ValueError(
                            f"PreparedUnstructuredConservativeRemap.execution_evidence must be None while borrowing the chart transaction; got {owner.execution_evidence!r}"
                        )
                material_remaps.extend(materials.remaps)
                workspace = source_workspace
                if workspace is None:
                    raise RuntimeError(
                        f"FiniteElementTopologyTransaction.source_workspace must be present for material remap owners; got {workspace!r}"
                    )
                workspace.retain_owner(materials)
            return materials.materials
        case MaterialTransaction() | None:
            if material_remaps is not None and materials is not None:
                raise TypeError(
                    f"FiniteElementTopologyTransaction.material_transfer.result must be MaterialTopologyTransferResult for chart callbacks; got {type(materials).__name__}"
                )
            return materials
        case _:
            assert_never(materials)


def _refresh_topology_fields(
    callback: Callable[..., Sequence[ArrayLike]],
    candidate_fields: tuple[Array, ...],
    target: FiniteElementDiscretization,
    adaptation: MeshAdaptationResult,
    args: object,
    /,
) -> tuple[Array, ...] | str:
    """Refresh transferred values while preserving layout-first refusal order."""
    refreshed = tuple(
        jnp.asarray(value) for value in callback(candidate_fields, adaptation, args)
    )
    if len(refreshed) != len(target.field_spaces) or any(
        value.shape != space.vector_space.structure().shape
        for value, space in zip(refreshed, target.field_spaces, strict=True)
    ):
        return "field-refresh-layout-rejected"
    if any(not bool(jnp.all(jnp.isfinite(value))) for value in refreshed):
        return "field-refresh-nonfinite"
    return refreshed


@final
class PreparedFiniteElementTopologyTransition(StrictModule, NonTrainableState):
    """Reusable immutable spaces and operators for one topology transition."""

    source: FiniteElementDiscretization
    target: FiniteElementDiscretization
    transfers: tuple[FiniteElementFieldTransfer, ...]
    transaction_id: str = eqx.field(static=True)
    adaptation_id: str = eqx.field(static=True)
    source_result_id: str = eqx.field(static=True)
    target_result_id: str = eqx.field(static=True)
    field_spec_ids: tuple[str, ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transaction: FiniteElementTopologyTransaction,
        adaptation: MeshAdaptationResult,
        source: FiniteElementDiscretization,
        target: FiniteElementDiscretization,
        transfers: Sequence[FiniteElementFieldTransfer],
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        prepared = tuple(transfers)
        source_fields = _fields_for_mesh(transaction.fields, adaptation.source.mesh)
        target_fields = _fields_for_mesh(transaction.fields, adaptation.target.mesh)
        source_ids = tuple(field.field_spec_id for field in source_fields)
        target_ids = tuple(field.field_spec_id for field in target_fields)
        if (
            adaptation.transition is None
            or not _space_realizes_result(source, adaptation.source, budget=budget)
            or not _space_realizes_result(target, adaptation.target, budget=budget)
            or tuple(field.field_spec_id for field in source.construction_plan.fields)
            != source_ids
            or tuple(field.field_spec_id for field in target.construction_plan.fields)
            != target_ids
            or len(prepared) != len(transaction.fields)
            or any(
                transfer.source_field.field_space_id
                != source.field_spaces[index].field_space_id
                or transfer.target_field.field_space_id
                != target.field_spaces[index].field_space_id
                or not transfer.evidence.passed
                for index, transfer in enumerate(prepared)
            )
        ):
            raise ValueError(
                f"PreparedFiniteElementTopologyTransition.transfers must preserve mesh, geometry, ordered field basis/orientation, and passing evidence; got source={source.mesh.topology_id}, target={target.mesh.topology_id}, fields={source_ids!r}, transfers={len(prepared)}"
            )
        self.source = source
        self.target = target
        self.transfers = prepared
        self.transaction_id = transaction.transaction_id
        self.adaptation_id = adaptation.result_id
        self.source_result_id = adaptation.source.result_id
        self.target_result_id = adaptation.target.result_id
        self.field_spec_ids = tuple(field.field_spec_id for field in transaction.fields)
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-topology-transition",
                "transaction": transaction.transaction_id,
                "adaptation": adaptation.result_id,
                "source": source.prepared_id,
                "target": target.prepared_id,
                "declared_fields": self.field_spec_ids,
                "source_fields": source_ids,
                "target_fields": target_ids,
                "transfers": [value.transfer_id for value in prepared],
            }
        )

    def require(
        self,
        transaction: FiniteElementTopologyTransaction,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        adaptation: MeshAdaptationResult,
        /,
    ) -> None:
        if (
            transaction.transaction_id != self.transaction_id
            or adaptation.result_id != self.adaptation_id
            or adaptation.source.result_id != self.source_result_id
            or adaptation.target.result_id != self.target_result_id
            or mesh.mesh_id != self.source.mesh.mesh_id
            or accepted.topology_id != mesh.topology_id
            or accepted.prepared_id != self.source.prepared_id
            or tuple(field.field_spec_id for field in transaction.fields)
            != self.field_spec_ids
            or len(accepted.fields) != len(self.source.field_spaces)
            or any(
                value.shape != space.vector_space.structure().shape
                for value, space in zip(
                    accepted.fields, self.source.field_spaces, strict=True
                )
            )
        ):
            raise ValueError(
                f"PreparedFiniteElementTopologyTransition.transaction must match the accepted mesh, revision, field order, and adaptation; got transaction={transaction.transaction_id!r}, adaptation={adaptation.result_id!r}, fields={len(accepted.fields)}"
            )


class FiniteElementHPTopologyResult(StrictModule, NonTrainableState):
    state: FiniteElementAcceptedState
    epoch: FiniteElementHPEpoch
    transaction: FiniteElementHPTransaction | None
    auxiliary_state: tuple[tuple[str, Array], ...]
    integrator_state: tuple[tuple[str, Array], ...]
    committed: Array
    receipt: CompositionRebindReceipt | None
    diagnostics: str = eqx.field(static=True)


@checked
def refinement_parent_cells(adaptation: MeshAdaptationResult, /) -> np.ndarray | None:
    """Return parent cell indices for a nesting adaptation, or ``None`` otherwise.

    Entry ``t`` is the source cell (concatenated block order) whose preserved or
    refined lineage produced target cell ``t``. Coarsening, created cells, and
    unknown (remeshing) lineage have no single parent per target cell.
    """

    transition = adaptation.transition
    if transition is None:
        raise ValueError(
            f"refinement_parent_cells.adaptation.transition must be present; got {transition!r}"
        )
    source = adaptation.source.mesh
    target = adaptation.target.mesh
    cells = transition.lineage.entity_lineage(target.topological_dimension)
    kinds = np.asarray(cells.relation_kinds)
    inheriting = (kinds == int(EntityLineageKind.PRESERVED)) | (
        kinds == int(EntityLineageKind.REFINED_FROM)
    )
    if not np.all(inheriting) or np.asarray(cells.created_target_ids).size:
        return None
    children = np.asarray(cells.target_global_ids, dtype=np.int64)
    parents = np.asarray(cells.source_global_ids, dtype=np.int64)
    target_ids = np.concatenate([np.asarray(block.global_ids) for block in target.blocks])
    source_ids = np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
    if np.unique(children).size != children.size or not np.array_equal(
        np.sort(children), np.sort(target_ids)
    ):
        return None
    source_order = np.argsort(source_ids)
    child_order = np.argsort(children)
    parent_ids = parents[child_order][np.argsort(np.argsort(target_ids))]
    positions = np.searchsorted(source_ids, parent_ids, sorter=source_order)
    if np.any(positions >= source_ids.size) or not np.array_equal(
        source_ids[source_order[np.minimum(positions, source_ids.size - 1)]],
        parent_ids,
    ):
        return None
    return source_order[positions]


def _fields_for_mesh(
    fields: Sequence[FiniteElementFieldSpec],
    mesh: CellMesh,
) -> tuple[FiniteElementFieldSpec, ...]:
    """Rebind declared element descriptors when native blocks are regrouped."""
    output = []
    for field in fields:
        if not field.block_names or set(field.block_names) == {
            block.name for block in mesh.blocks
        }:
            field.resolve(mesh)
            output.append(field)
            continue
        by_kind = {}
        for element in field.elements:
            prior = by_kind.get(element.cell_kind)
            if prior is not None and prior.element_id != element.element_id:
                raise ValueError(
                    f"FiniteElementFieldSpec.elements must unambiguously declare each regrouped cell family; got cell_kind={element.cell_kind!r}, element_ids={(prior.element_id, element.element_id)!r}"
                )
            by_kind[element.cell_kind] = element
        if any(block.cell_kind not in by_kind for block in mesh.blocks):
            raise ValueError(
                f"FiniteElementFieldSpec.elements must declare every target cell family; got declared={tuple(by_kind)!r}, target={tuple(block.cell_kind for block in mesh.blocks)!r}"
            )
        output.append(
            FiniteElementFieldSpec(
                field.name,
                {block.name: by_kind[block.cell_kind] for block in mesh.blocks},
                component_shape=field.component_shape,
            )
        )
    return tuple(output)


def _mesh_entry(mesh: CellMesh, /) -> CompositionEntry:
    return CompositionEntry(
        mesh,
        entry_id="mesh",
        role="topology",
        owner_id=_OWNER,
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="finite-element-mesh",
    )


def _field_semantics(field: FiniteElementFieldSpec, /) -> dict[str, object]:
    """Field identity that native block regrouping cannot change or alias.

    A declaration with exactly one element per cell kind is exactly the case
    `_fields_for_mesh` may rebind onto regrouped native blocks, so its meaning
    is its per-family element. Heterogeneous block- or region-specific
    declarations retain their full block-bound specification identity; their
    regrouping is refused by `_fields_for_mesh` and is never aliased here.
    """
    families: dict[str, set[str]] = {}
    for element in field.elements:
        families.setdefault(element.cell_kind, set()).add(element.element_id)
    if any(len(identifiers) != 1 for identifiers in families.values()):
        return {"block_bound": field.field_spec_id}
    return {
        "name": field.name,
        "component_shape": list(field.component_shape),
        "families": sorted(
            (kind, identifiers.pop()) for kind, identifiers in families.items()
        ),
    }


def _fields_semantics(fields: tuple[FiniteElementFieldSpec, ...], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "finite-element-field-set",
            "fields": [_field_semantics(field) for field in fields],
        }
    )


def _discretization_entry(
    discretization: FiniteElementDiscretization,
    fields: tuple[FiniteElementFieldSpec, ...],
    mesh: CompositionEntry,
    /,
) -> CompositionEntry:
    fields = _fields_for_mesh(fields, discretization.mesh)
    return CompositionEntry(
        discretization,
        entry_id="discretization",
        role="discretization",
        owner_id=_OWNER,
        structure_id=discretization.plan_id,
        revision_id=discretization.prepared_id,
        semantics_id=_fields_semantics(fields),
        dependencies=(mesh.binding("structure"),),
    )


def _state_entry(
    value: ArrayLike,
    entry_id: str,
    semantics_id: str,
    epoch_id: str,
    dependency: CompositionEntry,
    /,
    *,
    role: CompositionRole = "physical-state",
) -> CompositionEntry:
    array = jnp.asarray(value)
    return CompositionEntry(
        array,
        entry_id=entry_id,
        role=role,
        owner_id=_OWNER,
        structure_id=epoch_id,
        revision_id=canonical_fingerprint(
            {
                "kind": "finite-element-state-revision",
                "entry": entry_id,
                "epoch": epoch_id,
                "value": array_tree_fingerprint(np.asarray(array)),
            }
        ),
        semantics_id=semantics_id,
        dependencies=(dependency.binding("structure"),),
    )


def _material_entry(
    materials: MaterialTransaction, epoch_id: str, mesh: CompositionEntry, /
) -> CompositionEntry:
    return CompositionEntry(
        materials,
        entry_id="materials",
        role="history",
        owner_id=_OWNER,
        structure_id=epoch_id,
        revision_id=materials.transaction_id,
        semantics_id="finite-element-materials",
        dependencies=(mesh.binding("structure"),),
    )


def _policy_transport(
    source: CompositionEntry,
    target: CompositionEntry,
    route_id: str,
    successful: ArrayLike,
    /,
) -> CompositionTransport:
    """Consumer-policy transport (materials, histories) with its own success."""

    return CompositionTransport(
        "physical-remap",
        (source.entry_id,),
        (target,),
        source_structure_ids=(source.structure_id,),
        route_id=route_id,
        successful=jnp.asarray(successful, dtype=jnp.bool_),
    )


class FiniteElementTopologyTransaction(StrictModule, NonTrainableState):
    """Transfer, certify, and atomically promote one certified mesh adaptation.

    ``fields`` declares the finite-element field (family, degree, DOF
    functionals) of every accepted field, in order; accepted arrays must realize
    exactly those spaces, and the transfer route is chosen from them, never from
    array widths. Nested refinement moves every field through
    :func:`~phydrax.discretization.fem.prepare_nested_field_transfer` on the
    lineage parent indices (H1/L2 Lagrange, H(curl), H(div)); non-nested
    adaptations move every field through the Galerkin L2 projection on a
    certified common refinement (``projection_policy``), scalar Lagrange fields
    directly and H(curl)/H(div) fields through their covariant/contravariant
    Piola maps with certified reproduction and content evidence. Materials cross
    through ``material_transfer`` with the lineage.
    Chart material callbacks return ``MaterialTopologyTransferResult`` with
    every FV remap used by the material update. These owners share the
    native work, memory and deadline with all FE field roles and
    physical reanalysis; the returned transaction result owns their finalized
    immutable evidence. Ordinary non-chart callbacks return ``MaterialTransaction``.
    Chart callbacks additionally receive ``scope``, ``budget`` and
    ``source_workspace`` explicitly. The work ticket is settled before the
    callback; retained coordinate storage is separate from source ownership.
    Numerical/source banks used by callbacks but not already owned by accepted
    state, adaptation or ``args`` are declared explicitly in ``callback_owners``.
    Opaque Python callbacks themselves are never traversed as numerical owners.
    Bounded surface-chart adaptations require explicit ``surface_chart_semantics``
    for each scalar field: intensive values follow material chart correspondence;
    conservative densities preserve the old physical-area inventory. The
    prepared correspondence, not an identifier or corner overlap, supplies this
    action. Vector/Piola surface transfers are not implied by that scalar route.
    With ``field_transfer`` the declared semantics also bind each refreshed
    field: a conservative density keeps the remap's content obligation on its
    final value, while an intensive or material-compatible field is a separate
    re-solve transition that cannot inherit that badge; its remap image is still
    certified and its solved value is admitted only through ``certify``.
    Undeclared refreshed fields keep the conservative obligation.

    The mesh, the reprepared discretization, every field, and the materials are
    staged as one ``CompositionRebind`` and published by
    ``commit_composition_rebind`` only when ``certify`` accepts the candidate and
    every transport's evidence passes; otherwise the accepted mesh and state are
    returned unchanged with the refused ``CompositionRebindReceipt``.
    """

    certify: Callable
    fields: tuple[FiniteElementFieldSpec, ...]
    material_transfer: (
        Callable[..., MaterialTransaction | MaterialTopologyTransferResult] | None
    )
    field_transfer: Callable | None
    history_transfer: Callable | None
    projection_policy: CommonRefinementPolicy
    surface_chart_semantics: tuple[tuple[str, str], ...] = eqx.field(static=True)
    composition_rebind: Callable[[CompositionRebind], CompositionRebind] | None
    callback_owners: tuple[object, ...]
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        certify: Callable,
        /,
        *,
        fields: FiniteElementFieldSpec | Sequence[FiniteElementFieldSpec] = (),
        material_transfer: Callable[
            ..., MaterialTransaction | MaterialTopologyTransferResult
        ]
        | None = None,
        field_transfer: Callable | None = None,
        history_transfer: Callable | None = None,
        projection_policy: CommonRefinementPolicy | None = None,
        surface_chart_semantics: Mapping[str, SourceRealizationFieldSemantics]
        | None = None,
        composition_rebind: Callable[[CompositionRebind], CompositionRebind]
        | None = None,
        callback_owners: Sequence[object] = (),
        transaction_id: str = "finite-element-topology-transaction",
    ) -> None:
        if not callable(certify):
            raise TypeError("certify must be callable.")
        specs = (fields,) if isinstance(fields, FiniteElementFieldSpec) else tuple(fields)
        if any(not isinstance(field, FiniteElementFieldSpec) for field in specs):
            raise TypeError(
                f"FiniteElementTopologyTransaction.fields must contain FiniteElementFieldSpec values; got {tuple(type(field).__name__ for field in specs)!r}"
            )
        if material_transfer is not None and not callable(material_transfer):
            raise TypeError("material_transfer must be callable or None.")
        if field_transfer is not None and not callable(field_transfer):
            raise TypeError("field_transfer must be callable or None.")
        if history_transfer is not None and not callable(history_transfer):
            raise TypeError("history_transfer must be callable or None.")
        policy = (
            CommonRefinementPolicy(overlap_simplices=True)
            if projection_policy is None
            else projection_policy
        )
        if not isinstance(policy, CommonRefinementPolicy):
            raise TypeError(
                f"FiniteElementTopologyTransaction.projection_policy must be CommonRefinementPolicy or None; got {type(projection_policy).__name__}"
            )
        if composition_rebind is not None and not callable(composition_rebind):
            raise TypeError(
                f"FiniteElementTopologyTransaction.composition_rebind must be callable or None; got {composition_rebind!r}"
            )
        if surface_chart_semantics is not None and not isinstance(
            surface_chart_semantics, Mapping
        ):
            raise TypeError(
                f"FiniteElementTopologyTransaction.surface_chart_semantics must be a field-name mapping or None; got {type(surface_chart_semantics).__name__}"
            )
        chart_semantics = tuple(
            sorted(
                (
                    str(name),
                    parse(
                        semantics,
                        SourceRealizationFieldSemantics,
                        "surface_chart_semantics",
                    ),
                )
                for name, semantics in (surface_chart_semantics or {}).items()
            )
        )
        declared_names = {field.name for field in specs}
        if any(name not in declared_names for name, _ in chart_semantics):
            raise ValueError(
                f"FiniteElementTopologyTransaction.surface_chart_semantics must name declared fields; got names={tuple(name for name, _ in chart_semantics)!r}, declared={tuple(sorted(declared_names))!r}"
            )
        if not policy.overlap_simplices:
            raise ValueError(
                f"FiniteElementTopologyTransaction.projection_policy.overlap_simplices must be True; got {policy.overlap_simplices!r}"
            )
        identifier = str(transaction_id)
        if not identifier:
            raise ValueError("transaction_id must be non-empty.")
        self.certify = certify
        self.fields = specs
        self.material_transfer = material_transfer
        self.field_transfer = field_transfer
        self.history_transfer = history_transfer
        self.projection_policy = policy
        self.surface_chart_semantics = chart_semantics
        self.composition_rebind = composition_rebind
        self.callback_owners = tuple(callback_owners)
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-topology-transaction",
                "declared_id": identifier,
                "fields": [field.field_spec_id for field in specs],
                "projection_policy": policy.policy_id,
                "surface_chart_semantics": chart_semantics,
                "has_material_transfer": material_transfer is not None,
                "has_field_transfer": field_transfer is not None,
                "has_history_transfer": history_transfer is not None,
                "has_composition_rebind": composition_rebind is not None,
            }
        )

    def prepare_transition(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        adaptation: MeshAdaptationResult,
        /,
        *,
        source: FiniteElementDiscretization | None = None,
        target: FiniteElementDiscretization | None = None,
        field_transfers: Sequence[FiniteElementFieldTransfer] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedFiniteElementTopologyTransition:
        """Prepare reusable spaces and field operators for one nested transition."""
        if (source is None) != (target is None):
            raise ValueError(
                f"FiniteElementTopologyTransaction.prepare_transition.spaces must provide both resident spaces or neither; got source={type(source).__name__}, target={type(target).__name__}"
            )
        geometry_transition = adaptation.geometry_transition
        if (
            geometry_transition is not None
            and geometry_transition.evidence.kind == "bounded_chart_deformation"
        ):
            raise ValueError(
                f"FiniteElementTopologyTransaction.prepare_transition.geometry_transition must prepare inside its native execution scope; got kind={geometry_transition.evidence.kind!r}"
            )
        if source is None or target is None:
            source_, target_ = self._prepared_pair(accepted, adaptation, budget=budget)
        else:
            source_, target_ = source, target
        transfers = (
            self._field_transfers(source_, target_, adaptation, budget=budget)
            if field_transfers is None
            else tuple(field_transfers)
        )
        if isinstance(transfers, str):
            raise ValueError(
                f"FiniteElementTopologyTransaction.prepare_transition.transfers must be admitted; got {transfers!r}"
            )
        prepared = PreparedFiniteElementTopologyTransition(
            self, adaptation, source_, target_, transfers, budget=budget
        )
        prepared.require(self, accepted, mesh, adaptation)
        return prepared

    def _retained(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        diagnostics: str,
        /,
        *,
        receipt: CompositionRebindReceipt | None = None,
        transfers: tuple[FiniteElementFieldTransfer, ...] = (),
    ) -> FiniteElementTopologyResult:
        return FiniteElementTopologyResult(
            state=accepted,
            mesh=mesh,
            adaptation=None,
            committed=jnp.asarray(False),
            receipt=receipt,
            transfers=transfers,
            diagnostics=diagnostics,
        )

    def _surface_chart_field_transfers(
        self,
        source: FiniteElementDiscretization,
        target: FiniteElementDiscretization,
        correspondence: MaterialChartDeformation,
        adaptation: MeshAdaptationResult,
        /,
        *,
        source_workspace: NativeHostStorageWorkspace | None,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[FiniteElementFieldTransfer, ...] | str:
        from phydrax.discretization._sphere_chart_records import (
            PreparedSphereChartDeformation,
        )

        from ..discretization._surface_chart_records import (
            PreparedSurfaceChartDeformation,
        )
        from ..discretization.fem._sphere_chart_transfer import (
            prepare_sphere_chart_compatible_transfer,
            prepare_sphere_chart_field_transfer,
        )
        from ..discretization.fem._surface_chart_compatible import (
            prepare_surface_chart_compatible_transfer,
        )
        from ..discretization.fem._surface_chart_transfer import (
            prepare_surface_chart_field_transfer,
        )
        from ..meshing._tetra_metric import MetricRemeshingEvidence

        match correspondence:
            case PreparedSurfaceChartDeformation() | PreparedSphereChartDeformation():
                pass
            case _:
                return "surface-chart-correspondence-unsupported"

        semantics = dict(self.surface_chart_semantics)
        if set(semantics) != {field.name for field in self.fields}:
            return "surface-chart-field-semantics-required"
        if (
            not isinstance(adaptation.evidence, MetricRemeshingEvidence)
            or adaptation.evidence.execution_evidence is None
        ):
            raise ValueError(
                f"MeshAdaptationResult.evidence must include a MetricRemeshingEvidence execution record; got {adaptation.evidence!r}"
            )
        previous = adaptation.evidence.execution_evidence
        previous.require_valid()
        ledger = budget
        transfers: list[FiniteElementFieldTransfer] = []
        if source_workspace is None:
            raise RuntimeError(
                f"FiniteElementTopologyTransaction.source_workspace must provide a persistent reservation separate from algebra scratch; got {source_workspace!r}"
            )
        # One correspondence proof serves every field transfer of this request.
        validation = (
            correspondence.require_bound(
                source.mesh,
                correspondence.source_geometry,
                target.mesh,
                correspondence.target_geometry,
                budget=ledger,
            )
            if isinstance(correspondence, PreparedSphereChartDeformation)
            else None
        )
        prepared_fields: dict[
            tuple[str, str, tuple[tuple[str, str], ...]],
            tuple[FiniteElementFieldTransfer, ...],
        ] | None = (
            validation.derive("sphere-transaction-field-transfers", dict)
            if validation is not None
            else None
        )
        prepared_key = (
            source.prepared_id, target.prepared_id, self.surface_chart_semantics
        )
        if prepared_fields is not None and prepared_key in prepared_fields:
            return prepared_fields[prepared_key]
        for field in self.fields:
            semantic = semantics[field.name]
            work = min(
                ledger.maximum_work_units - ledger.work_units,
                ledger.ticket.remaining_work_units,
            )
            storage = ledger.maximum_memory_bytes
            ledger.admit_work_bound(work)
            if work <= 0:
                raise ValueError(
                    f"FiniteElementTopologyTransaction.work must be positive for material field transfer; got {work}"
                )
            with ledger.bound_stage(work, storage):
                match correspondence:
                    case PreparedSphereChartDeformation():
                        if semantic == "material-compatible":
                            transfer = prepare_sphere_chart_compatible_transfer(
                                source,
                                target,
                                correspondence,
                                field_name=field.name,
                                source_geometry=correspondence.source_geometry,
                                target_geometry=correspondence.target_geometry,
                                maximum_work=work,
                                maximum_storage_bytes=storage,
                                budget=ledger,
                                validation=validation,
                            ).field_transfer
                        elif semantic in ("intensive", "conservative-density"):
                            transfer = prepare_sphere_chart_field_transfer(
                                source,
                                target,
                                correspondence,
                                field_name=field.name,
                                source_geometry=correspondence.source_geometry,
                                target_geometry=correspondence.target_geometry,
                                semantics=semantic,
                                maximum_work=work,
                                budget=ledger,
                                validation=validation,
                            )
                        else:
                            raise ValueError(
                                f"FiniteElementTopologyTransaction.surface_chart_semantics must declare a supported sphere material field semantic; got {semantic!r}"
                            )
                    case PreparedSurfaceChartDeformation():
                        if semantic == "material-compatible":
                            transfer = prepare_surface_chart_compatible_transfer(
                                source,
                                target,
                                correspondence,
                                field_name=field.name,
                                source_geometry=correspondence.source_geometry,
                                target_geometry=correspondence.target_geometry,
                                maximum_work=work,
                                budget=ledger,
                            ).field_transfer
                        elif semantic in ("intensive", "conservative-density"):
                            transfer = prepare_surface_chart_field_transfer(
                                source,
                                target,
                                correspondence,
                                field_name=field.name,
                                source_geometry=correspondence.source_geometry,
                                target_geometry=correspondence.target_geometry,
                                semantics=semantic,
                                maximum_work=work,
                                budget=ledger,
                            )
                        else:
                            raise ValueError(
                                f"FiniteElementTopologyTransaction.surface_chart_semantics must declare a supported surface material field semantic; got {semantic!r}"
                            )
                    case _:
                        assert_never(correspondence)

                transfers.append(transfer)
        result = tuple(transfers)
        if prepared_fields is not None:
            # This proof token lives until the complete request ends, including
            # a rejected trial followed by an accepted trial. Keep the actual
            # numerical owners admitted for that same lifetime.
            ledger.retain_owner(result)
            prepared_fields[prepared_key] = result
        return result

    def _source_realization_field_transfers(
        self,
        source: FiniteElementDiscretization,
        target: FiniteElementDiscretization,
        adaptation: MeshAdaptationResult,
        /,
    ) -> tuple[FiniteElementFieldTransfer, ...] | str:
        request = adaptation.request
        if not isinstance(request, GeometryRealizationMeshAdaptation):
            raise TypeError(
                f"MeshAdaptationResult.request must be GeometryRealizationMeshAdaptation for material realization transfer; got {type(request).__name__}"
            )
        meanings = dict(self.surface_chart_semantics)
        if any(field.name not in meanings for field in self.fields):
            return "source-realization-field-semantics-required"
        remaining = (
            adaptation.policy.limits.maximum_work_units
            - 4 * request.certificate_limits.maximum_work_units
            - request.geometry_realization.transition.evidence.evaluation_count
        )
        if remaining < 2 * len(self.fields):
            return "source-realization-field-work-budget"
        return tuple(
            prepare_source_realization_field_transfer(
                source,
                target,
                request.geometry_realization,
                field_name=field.name,
                source_geometry=adaptation.source.geometry,
                target_geometry=adaptation.target.geometry,
                semantics=parse(
                    meanings[field.name],
                    SourceRealizationFieldSemantics,
                    "surface_chart_semantics",
                ),
                maximum_work=remaining // len(self.fields),
                maximum_storage_bytes=adaptation.policy.limits.maximum_scratch_bytes,
            )
            for field in self.fields
        )

    def _field_transfers(
        self,
        source: FiniteElementDiscretization,
        target: FiniteElementDiscretization,
        adaptation: MeshAdaptationResult,
        /,
        *,
        source_workspace: NativeHostStorageWorkspace | None = None,
        scope: NativeExecutionScope | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[FiniteElementFieldTransfer, ...] | str:
        """One transfer per declared field, or the refusal reason."""

        geometry_transition = adaptation.geometry_transition
        if isinstance(adaptation.request, GeometryRealizationMeshAdaptation):
            return self._source_realization_field_transfers(source, target, adaptation)
        if (
            geometry_transition is not None
            and geometry_transition.evidence.kind == "bounded_chart_deformation"
        ):
            correspondence = geometry_transition.chart_deformation
            if (
                correspondence is None
                or geometry_transition.chart_correspondence_id
                != correspondence.deformation_id
            ):
                return "surface-chart-correspondence-missing"
            if scope is None or budget is None:
                raise RuntimeError(
                    f"FiniteElementTopologyTransaction.execution must provide scope and budget for chart field roles; got scope={scope!r}, budget={budget!r}"
                )
            return self._surface_chart_field_transfers(
                source,
                target,
                correspondence,
                adaptation,
                source_workspace=source_workspace,
                scope=scope,
                budget=budget,
            )
        parents = refinement_parent_cells(adaptation)
        coarsening = adaptation.coarsening_witnesses
        if (
            parents is not None
            or coarsening is not None
            or adaptation.geometry_transition is not None
        ):
            prepared_by_layout: dict[
                tuple[str, str, tuple[str, ...], tuple[str, ...]],
                FiniteElementFieldTransfer,
            ] = {}
            # One transition's nested pieces are proved once and passed
            # explicitly to every later field route that reads them; each
            # field still owns its ticket and imports their charges.
            references: list[PreparedNestedReference] = []
            transfers: list[FiniteElementFieldTransfer] = []
            for field in self.fields:
                source_index = source.field_index(field.name)
                target_index = target.field_index(field.name)
                key = (
                    source.dof_maps[source_index].dof_map_id,
                    target.dof_maps[target_index].dof_map_id,
                    tuple(
                        element.element_id for element in source.elements[source_index]
                    ),
                    tuple(
                        element.element_id for element in target.elements[target_index]
                    ),
                )
                retained = prepared_by_layout.get(key)
                if retained is None:
                    retained, reference = prepare_shared_nested_field_transfer(
                        source,
                        target,
                        parents,
                        field_name=field.name,
                        parent_reference_vertices=(
                            adaptation.parent_reference_vertices
                            if parents is not None
                            else None
                        ),
                        parent_reference_homogeneous=(
                            adaptation.parent_reference_homogeneous
                            if parents is not None
                            else None
                        ),
                        source_geometry=adaptation.source.geometry,
                        target_geometry=adaptation.target.geometry,
                        geometry_transition=adaptation.geometry_transition,
                        coarsening_witnesses=coarsening,
                        prepared=tuple(references),
                        budget=budget,
                    )
                    if reference is not None and all(
                        reference is not item for item in references
                    ):
                        references.append(reference)
                    prepared_by_layout[key] = retained
                    transfers.append(retained)
                    continue
                transfers.append(
                    FiniteElementFieldTransfer(
                        retained.transfer,
                        source.field_spaces[source_index],
                        target.field_spaces[target_index],
                        retained.geometry,
                        retained.evidence,
                        source_measures=retained.source_measures,
                        target_measures=retained.target_measures,
                    )
                )
            return tuple(transfers)
        refinement = prepare_common_refinement(
            source.mesh, target.mesh, policy=self.projection_policy
        )
        if not refinement.succeeded:
            return "projection-coverage-rejected"
        return tuple(
            prepare_projection_field_transfer(
                source,
                prepare_l2_projection_target(target, field_name=field.name),
                refinement,
                field_name=field.name,
            )
            for field in self.fields
        )

    def _prepared_pair(
        self,
        accepted: FiniteElementAcceptedState,
        adaptation: MeshAdaptationResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[FiniteElementDiscretization, FiniteElementDiscretization]:
        if not self.fields:
            raise ValueError(
                f"FiniteElementTopologyTransaction.fields must declare accepted FE fields; got {self.fields!r}"
            )
        if len(self.fields) != len(accepted.fields):
            raise ValueError(
                f"FiniteElementTopologyTransaction.fields must contain one declaration per accepted field; got declared={len(self.fields)}, accepted={len(accepted.fields)}"
            )
        source = FiniteElementPlan(
            adaptation.source.mesh,
            _fields_for_mesh(self.fields, adaptation.source.mesh),
            coordinate_spec=adaptation.source.geometry,
        ).prepare(budget=budget)
        for value, space in zip(accepted.fields, source.field_spaces, strict=True):
            if value.shape != space.vector_space.structure().shape:
                raise ValueError(
                    f"FiniteElementAcceptedState.fields.{space.name}.shape must equal {space.vector_space.structure().shape!r}; got {value.shape!r}"
                )
        return source, FiniteElementPlan(
            adaptation.target.mesh,
            _fields_for_mesh(self.fields, adaptation.target.mesh),
            coordinate_spec=adaptation.target.geometry,
        ).prepare(budget=budget)

    def _transfer_materials(
        self,
        accepted: FiniteElementAcceptedState,
        lineage: object,
        args: object,
        /,
        *,
        source_workspace: NativeHostStorageWorkspace | None = None,
        scope: NativeExecutionScope | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> MaterialTransaction | MaterialTopologyTransferResult | None | str:
        if accepted.materials is None:
            return None
        if self.material_transfer is None:
            return "material-transfer-policy-required"
        if source_workspace is None:
            transferred = self.material_transfer(accepted.materials, lineage, args)
        else:
            if scope is None or budget is None:
                raise RuntimeError(
                    f"FiniteElementTopologyTransaction.execution must provide scope and budget for chart material roles; got scope={scope!r}, budget={budget!r}"
                )
            transferred = self.material_transfer(
                accepted.materials,
                lineage,
                args,
                source_workspace=source_workspace,
                scope=scope,
                budget=budget,
            )
        if not isinstance(
            transferred, (MaterialTransaction, MaterialTopologyTransferResult)
        ):
            return "material-transfer-rejected"
        return transferred

    @checked
    def execute(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        adaptation: MeshAdaptationResult,
        args: object = None,
        /,
        *,
        decision: SolverAwareDecision | None = None,
        reanalysis: Callable[
            [CellMesh, tuple[Array, ...], MaterialTransaction | None, object],
            PhysicalErrorEvidence,
        ]
        | None = None,
        compiled_layout_id: str | None = None,
        preparation: PreparedFiniteElementTopologyTransition | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> FiniteElementTopologyResult:
        geometry_transition = adaptation.geometry_transition
        if (
            geometry_transition is None
            or geometry_transition.evidence.kind != "bounded_chart_deformation"
        ):
            return self._execute(
                accepted,
                mesh,
                adaptation,
                args,
                decision=decision,
                reanalysis=reanalysis,
                compiled_layout_id=compiled_layout_id,
                preparation=preparation,
                scope=current_native_execution_scope() if budget is not None else None,
                budget=budget,
            )
        if preparation is not None:
            raise ValueError(
                f"FiniteElementTopologyTransaction.preparation must be None for chart transitions prepared inside their native scope; got {preparation.preparation_id!r}"
            )
        from ..discretization.fem._topology_transfer import FiniteElementTransferEvidence
        from ..discretization.finite_volume._automatic_remap import bind_remap_execution
        from ..meshing._tetra_metric import MetricRemeshingEvidence
        from ..meshing._volume_generation import native_volume_source_execution_budget

        if (
            not isinstance(adaptation.evidence, MetricRemeshingEvidence)
            or adaptation.evidence.execution_evidence is None
        ):
            raise ValueError(
                f"MeshAdaptationResult.evidence must include a MetricRemeshingEvidence source execution record; got {adaptation.evidence!r}"
            )
        previous = adaptation.evidence.execution_evidence
        previous.require_valid()
        owner_id = previous.owner_id
        if owner_id is None:
            raise ValueError(
                f"NativeExecutionRecord.owner_id must identify the chart transaction source; got {owner_id!r}"
            )
        limits = adaptation.policy.limits
        owners: list[PreparedUnstructuredConservativeRemap] = []
        active_scope = current_native_execution_scope()
        if active_scope is not None:
            if budget is None:
                raise ValueError(
                    "FiniteElementTopologyTransaction.budget must identify the active request ledger for a chart transition; got None"
                )
            if not budget.ticket.settled and not budget.ticket.native_accounting:
                raise ValueError(
                    "FiniteElementTopologyTransaction.budget must charge the active native request; got a live nonnative ticket"
                )
            with active_scope.host_workspace() as workspace:
                workspace.retain_owner(
                    (
                        accepted,
                        mesh,
                        adaptation,
                        args,
                        decision,
                        self.callback_owners,
                        geometry_transition.chart_deformation,
                    )
                )
                result = self._execute(
                    accepted,
                    mesh,
                    adaptation,
                    args,
                    decision=decision,
                    reanalysis=reanalysis,
                    compiled_layout_id=compiled_layout_id,
                    material_remaps=owners,
                    source_workspace=workspace,
                    scope=active_scope,
                    budget=budget,
                )
                workspace.retain_owner((result, tuple(owners)))
            # The caller binds the actual ended request receipt after its
            # surrounding native scope exits. No intermediate receipt exists.
            return FiniteElementTopologyResult(
                state=result.state,
                mesh=result.mesh,
                adaptation=result.adaptation,
                committed=result.committed,
                receipt=result.receipt,
                transfers=result.transfers,
                diagnostics=result.diagnostics,
                material_remaps=tuple(owners),
            )
        if budget is not None:
            raise ValueError(
                "FiniteElementTopologyTransaction.budget must belong to an active native request for a chart transition; got a standalone ledger"
            )
        with NativeExecutionScope(limits.native_limits()) as native:
            with (
                native_volume_source_execution_budget(
                    limits, previous, owner_id, scope=native
                ),
                native.host_workspace() as workspace,
            ):
                workspace.retain_owner(
                    (
                        accepted,
                        mesh,
                        adaptation,
                        args,
                        decision,
                        self.callback_owners,
                        geometry_transition.chart_deformation,
                    )
                )
                initial = HostBatchTicket(0)
                ledger = CoordinateEnclosureBudget(
                    initial,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                initial.settle()
                with ledger.retained_storage(
                    lambda bound: native.batch(0, bytes_upper=bound)
                ):
                    result = self._execute(
                        accepted,
                        mesh,
                        adaptation,
                        args,
                        decision=decision,
                        reanalysis=reanalysis,
                        compiled_layout_id=compiled_layout_id,
                        material_remaps=owners,
                        source_workspace=workspace,
                        scope=native,
                        budget=ledger,
                    )
                    workspace.retain_owner((result, tuple(owners)))
                overlap = native_preparation_ledger(native).overlap(native, previous)
        if native.evidence is None:
            raise RuntimeError(
                f"NativeExecutionScope.evidence must contain the ended chart transaction record; got {native.evidence!r}"
            )
        execution = NativeExecutionRecord(
            overlap.normalize(native.evidence),
            owner_id=previous.owner_id,
            preparation_evidence=previous,
        )
        transfers = []
        for transfer in result.transfers:
            original = transfer.evidence
            evidence = FiniteElementTransferEvidence(
                dict(original.defects),
                original.tolerance,
                estimates=dict(original.estimates),
                bounds=dict(original.bounds),
                execution_evidence=execution,
            )
            transfers.append(
                FiniteElementFieldTransfer(
                    transfer.transfer,
                    transfer.source_field,
                    transfer.target_field,
                    transfer.geometry,
                    evidence,
                    source_measures=transfer.source_measures,
                    target_measures=transfer.target_measures,
                )
            )
        return FiniteElementTopologyResult(
            state=result.state,
            mesh=result.mesh,
            adaptation=result.adaptation,
            committed=result.committed,
            receipt=result.receipt,
            transfers=tuple(transfers),
            diagnostics=result.diagnostics,
            execution_evidence=execution,
            material_remaps=tuple(
                bind_remap_execution(owner, execution) for owner in owners
            ),
        )

    @checked
    def _execute(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        adaptation: MeshAdaptationResult,
        args: object = None,
        /,
        *,
        decision: SolverAwareDecision | None = None,
        reanalysis: Callable[
            [CellMesh, tuple[Array, ...], MaterialTransaction | None, object],
            PhysicalErrorEvidence,
        ]
        | None = None,
        compiled_layout_id: str | None = None,
        material_remaps: list[PreparedUnstructuredConservativeRemap] | None = None,
        source_workspace: NativeHostStorageWorkspace | None = None,
        preparation: PreparedFiniteElementTopologyTransition | None = None,
        scope: NativeExecutionScope | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> FiniteElementTopologyResult:
        if (
            mesh.topology_id != accepted.topology_id
            or adaptation.source.mesh.mesh_id != mesh.mesh_id
        ):
            raise ValueError("Accepted state and mesh adaptation disagree.")
        source_realization = isinstance(
            adaptation.request, GeometryRealizationMeshAdaptation
        )
        if source_realization and reanalysis is None:
            return self._retained(accepted, mesh, "independent-reanalysis-required")
        if decision is not None:
            reason = _topology_decision_admission(
                adaptation, decision, reanalysis is not None
            )
            if reason is not None:
                return self._retained(accepted, mesh, reason)
        transition = adaptation.transition
        if transition is None:
            return self._retained(accepted, mesh, "adaptation-unchanged")
        candidate_mesh = adaptation.target.mesh
        if preparation is None:
            if budget is None:
                source, target = self._prepared_pair(accepted, adaptation)
                transfers = self._field_transfers(source, target, adaptation)
            else:
                if scope is None:
                    raise RuntimeError(
                        f"FiniteElementTopologyTransaction.scope must be present for chart field preparation; got {scope!r}"
                    )
                with (
                    budget.bind_ticket(scope.ticket_view())
                    if budget.ticket.settled
                    else budget.activate()
                ):
                    source, target = self._prepared_pair(
                        accepted, adaptation, budget=budget
                    )
                    transfers = self._field_transfers(
                        source,
                        target,
                        adaptation,
                        source_workspace=source_workspace,
                        scope=scope,
                        budget=budget,
                    )
                if source_workspace is not None:
                    source_workspace.retain_owner((source, target))
            if isinstance(transfers, str):
                return self._retained(accepted, mesh, transfers)
        else:
            preparation.require(self, accepted, mesh, adaptation)
            source, target = preparation.source, preparation.target
            transfers = preparation.transfers
        if decision is not None:
            _require_topology_candidate_layouts(
                source, target, adaptation, decision, compiled_layout_id
            )
        materials = self._transfer_materials(
            accepted,
            transition.lineage,
            args,
            source_workspace=source_workspace,
            scope=scope,
            budget=budget,
        )
        if isinstance(materials, str):
            return self._retained(accepted, mesh, materials, transfers=transfers)
        materials = _adopt_topology_material_remaps(
            materials, adaptation, material_remaps, source_workspace
        )
        source_epoch = TopologyEpoch(
            accepted.state_version,
            cell_geometry_id(adaptation.source.geometry),
            mesh.topology_id,
            _PARTITION,
        )
        target_epoch = TopologyEpoch(
            accepted.state_version + 1,
            cell_geometry_id(adaptation.target.geometry),
            candidate_mesh.topology_id,
            _PARTITION,
        )
        transitions = tuple(
            transfer.epoch_transition(source_epoch, target_epoch)
            for transfer in transfers
        )
        candidate_fields = tuple(
            item.apply(value).values.reshape(space.vector_space.structure().shape)
            for item, value, space in zip(
                transitions, accepted.fields, target.field_spaces, strict=True
            )
        )
        transferred_fields = candidate_fields
        if self.field_transfer is not None:
            refreshed = _refresh_topology_fields(
                self.field_transfer, candidate_fields, target, adaptation, args
            )
            if isinstance(refreshed, str):
                return self._retained(accepted, mesh, refreshed, transfers=transfers)
            candidate_fields = refreshed
        certified = bool(
            jnp.asarray(
                self.certify(
                    candidate_mesh,
                    candidate_fields,
                    materials,
                    transition.lineage,
                    args,
                )
            )
        )
        if reanalysis is not None and (decision is not None or source_realization):
            actual = reanalysis(candidate_mesh, candidate_fields, materials, args)
            if not isinstance(actual, PhysicalErrorEvidence):
                raise TypeError(
                    f"FiniteElementTopologyTransaction.reanalysis.result must be PhysicalErrorEvidence; got {type(actual).__name__}"
                )
            certified = certified and actual.revision_id == adaptation.target.result_id
            if decision is not None:
                certified = certified and not decision.reanalysis_issues(actual)
        # Source-realization trials expose a staged CompositionRebindReceipt, never a publication
        # authorized by an opaque Boolean instead of the physical decision.
        if source_realization and decision is None:
            certified = False
        receipt = self._commit(
            accepted,
            (source, target),
            (source_epoch, target_epoch),
            transitions,
            candidate_fields,
            materials,
            transition.lineage.lineage_id,
            certified,
            decision,
            transferred_fields,
        )
        if not receipt.published:
            reason = (
                "transfer-evidence-rejected"
                if not all(receipt.transport_accepted)
                else "geometry-realization-decision-required"
                if source_realization and decision is None
                else "candidate-certification-rejected"
            )
            return self._retained(
                accepted, mesh, reason, receipt=receipt, transfers=transfers
            )
        promoted = FiniteElementAcceptedState(
            candidate_fields,
            accepted.time,
            accepted.step,
            candidate_mesh.topology_id,
            target.prepared_id,
            compiled_layout_id
            if compiled_layout_id is not None
            else f"{accepted.compilation_id}:transition:{transition.transition_id}",
            materials=materials,
            schedule_cursor=accepted.schedule_cursor,
            state_version=accepted.state_version + 1,
            transition_id=receipt.receipt_id,
        )
        return FiniteElementTopologyResult(
            state=promoted,
            mesh=candidate_mesh,
            adaptation=adaptation,
            committed=jnp.asarray(True),
            receipt=receipt,
            transfers=transfers,
            diagnostics="committed",
        )

    def _refreshed_field_transport(
        self,
        transition: TopologyEpochTransition | FieldEpochTransition,
        source: CompositionEntry,
        carried: CompositionEntry,
        solved: CompositionEntry,
        semantics: SourceRealizationFieldSemantics | None,
        /,
    ) -> CompositionTransport:
        """Certify the remap image, then stage the independently checked solve.

        The solved field is not falsely labeled as the remap image. An explicit
        conservative density requires the remap's content ledger; an
        unspecified field retains any content obligation of its prepared remap.
        A declared intensive/material-compatible re-solve is
        a different physical transition: it reports no inherited content and is
        accepted only with its certified remap image; physical reanalysis gates
        the single accepted composition boundary.
        """
        remap = transition.composition_transport(source, carried)
        # The owner transaction ID already binds every declared field semantics.
        route_id = canonical_fingerprint(
            {
                "kind": "finite-element-remap-reanalysis",
                "remap": remap.transport_id,
                "owner": self.transaction_id,
                "source": source.record_id,
                "solved": solved.record_id,
            }
        )
        successful = remap.successful & jnp.all(jnp.isfinite(solved.value))
        if semantics == "conservative-density" and not remap.conservative:
            # A checked non-conservative high-order/Piola interpolation is not
            # a physical density remap merely because a caller requested one.
            successful = jnp.asarray(False)
        match semantics:
            case "intensive" | "material-compatible":
                return CompositionTransport(
                    remap.kind,
                    remap.source_entry_ids,
                    (solved,),
                    source_structure_ids=remap.source_structure_ids,
                    route_id=route_id,
                    successful=successful,
                )
            case "conservative-density" | None:
                target_content = (
                    jnp.vdot(
                        transition.target_measures, jnp.asarray(solved.value).reshape(-1)
                    )[None]
                    if isinstance(transition, TopologyEpochTransition)
                    else None
                )
                return CompositionTransport(
                    remap.kind,
                    remap.source_entry_ids,
                    (solved,),
                    source_structure_ids=remap.source_structure_ids,
                    route_id=route_id,
                    successful=successful,
                    source_content=remap.source_content,
                    target_content=target_content,
                    content_tolerance=remap.content_tolerance,
                )
            case invalid:
                assert_never(invalid)

    def _commit(
        self,
        accepted: FiniteElementAcceptedState,
        discretizations: tuple[FiniteElementDiscretization, FiniteElementDiscretization],
        epochs: tuple[TopologyEpoch, TopologyEpoch],
        transitions: tuple[TopologyEpochTransition | FieldEpochTransition, ...],
        candidate_fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        lineage_id: str,
        certified: bool,
        /,
        decision: SolverAwareDecision | None,
        transferred_fields: tuple[Array, ...],
    ) -> CompositionRebindReceipt:
        """Stage mesh, discretization, fields, and materials as one rebind."""

        source, target = discretizations
        source_epoch, target_epoch = epochs
        source_specs = _fields_for_mesh(self.fields, source.mesh)
        target_specs = _fields_for_mesh(self.fields, target.mesh)
        mesh_entries = (_mesh_entry(source.mesh), _mesh_entry(target.mesh))
        discretization_entries = tuple(
            _discretization_entry(item, self.fields, entry)
            for item, entry in zip(discretizations, mesh_entries, strict=True)
        )
        source_fields = tuple(
            _state_entry(
                value,
                f"field/{field.name}",
                f"finite-element-field:{canonical_fingerprint(_field_semantics(field))}",
                source_epoch.epoch_id,
                discretization_entries[0],
            )
            for field, value in zip(source_specs, accepted.fields, strict=True)
        )
        target_fields = tuple(
            _state_entry(
                value,
                f"field/{field.name}",
                f"finite-element-field:{canonical_fingerprint(_field_semantics(field))}",
                target_epoch.epoch_id,
                discretization_entries[1],
            )
            for field, value in zip(target_specs, candidate_fields, strict=True)
        )
        if self.field_transfer is None:
            transports = [
                item.composition_transport(source_entry, target_entry)
                for item, source_entry, target_entry in zip(
                    transitions, source_fields, target_fields, strict=True
                )
            ]
        else:
            intermediate_fields = tuple(
                _state_entry(
                    value,
                    f"field/{field.name}",
                    f"finite-element-field:{canonical_fingerprint(_field_semantics(field))}",
                    target_epoch.epoch_id,
                    discretization_entries[1],
                )
                for field, value in zip(target_specs, transferred_fields, strict=True)
            )
            declared = dict(self.surface_chart_semantics)
            transports = [
                self._refreshed_field_transport(
                    item,
                    before,
                    carried,
                    solved,
                    None
                    if field.name not in declared
                    else parse(
                        declared[field.name],
                        SourceRealizationFieldSemantics,
                        "surface_chart_semantics",
                    ),
                )
                for field, item, before, carried, solved in zip(
                    self.fields,
                    transitions,
                    source_fields,
                    intermediate_fields,
                    target_fields,
                    strict=True,
                )
            ]
        source_entries = [*mesh_entries[:1], discretization_entries[0], *source_fields]
        if accepted.materials is not None and materials is not None:
            source_materials = _material_entry(
                accepted.materials, source_epoch.epoch_id, mesh_entries[0]
            )
            source_entries.append(source_materials)
            transports.append(
                _policy_transport(
                    source_materials,
                    _material_entry(materials, target_epoch.epoch_id, mesh_entries[1]),
                    canonical_fingerprint(
                        {
                            "kind": "finite-element-material-transfer",
                            "lineage": lineage_id,
                            "source": accepted.materials.transaction_id,
                            "target": materials.transaction_id,
                        }
                    ),
                    True,
                )
            )
        rebind = CompositionRebind(
            Composition(source_entries, boundary_id=accepted.accepted_id),
            reprepare=(mesh_entries[1], discretization_entries[1]),
            transports=transports,
        )
        return commit_composition_rebind(
            self._complete_rebind(rebind, decision), accepted_boundary=certified
        )

    def _complete_rebind(
        self, core: CompositionRebind, decision: SolverAwareDecision | None, /
    ) -> CompositionRebind:
        """Attach running artifacts/state without weakening any core transport."""
        rebind = (
            core if self.composition_rebind is None else self.composition_rebind(core)
        )
        if not isinstance(rebind, CompositionRebind):
            raise TypeError(
                f"FiniteElementTopologyTransaction.composition_rebind.result must be CompositionRebind; got {type(rebind).__name__}"
            )
        if rebind.source.boundary_id != core.source.boundary_id:
            raise ValueError(
                f"CompositionRebind.source.boundary_id must equal {core.source.boundary_id!r}; got {rebind.source.boundary_id!r}"
            )
        for original, extended in (
            (core.source, rebind.source),
            (core.candidate, rebind.candidate),
        ):
            for entry in original.entries:
                if (
                    entry.entry_id not in extended.entry_ids
                    or extended.entry(entry.entry_id).record_id != entry.record_id
                ):
                    raise ValueError(
                        f"CompositionRebind.entries must retain core mesh/state artifacts; got entry_id={entry.entry_id!r}, extended_entry_ids={extended.entry_ids!r}"
                    )
        if not {item.transport_id for item in core.transports}.issubset(
            item.transport_id for item in rebind.transports
        ):
            raise ValueError(
                f"CompositionRebind.transports must retain core field/material transports; got {tuple(item.transport_id for item in rebind.transports)!r}"
            )
        if decision is not None:
            selected = decision.selected
            if selected is None:
                raise ValueError(
                    f"SolverAwareDecision.selected must be present to publish a composition; got {selected!r}"
                )
            selected.feasibility.require_rebind(rebind)
        return rebind

    def _hp_retained(
        self,
        accepted: FiniteElementAcceptedState,
        transaction: FiniteElementHPTransaction,
        auxiliary: tuple[tuple[str, Array], ...],
        integrator: tuple[tuple[str, Array], ...],
        diagnostics: str,
        /,
        *,
        receipt: CompositionRebindReceipt | None = None,
    ) -> FiniteElementHPTopologyResult:
        return FiniteElementHPTopologyResult(
            accepted,
            transaction.accepted,
            None,
            auxiliary,
            integrator,
            jnp.asarray(False),
            receipt,
            diagnostics,
        )

    @checked
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
        """Transfer, certify, and atomically promote one prepared hp candidate.

        The hp epoch, every field, the materials, and the named auxiliary and
        integrator histories are staged as one ``CompositionRebind``; field
        transports succeed only with finite values and the hp transaction's own
        admissibility, geometry, and conservation evidence.
        """

        if accepted.topology_id != transaction.accepted.topology.topology_id:
            raise ValueError("Accepted state and hp transaction topology disagree.")
        auxiliary = tuple(
            (str(name), jnp.asarray(value)) for name, value in auxiliary_state
        )
        integrator = tuple(
            (str(name), jnp.asarray(value)) for name, value in integrator_state
        )
        transfers = transaction.p_transfers + transaction.h_transfers
        if self.field_transfer is None:
            if len(transfers) != len(accepted.fields):
                raise ValueError(
                    "Automatic hp state transfer requires one transfer per field."
                )
            candidate_fields = tuple(
                transfer.apply_l2_projection(field)
                for transfer, field in zip(transfers, accepted.fields, strict=True)
            )
        else:
            candidate_fields = tuple(
                jnp.asarray(value)
                for value in self.field_transfer(accepted.fields, transaction, args)
            )
        if len(candidate_fields) != len(accepted.fields):
            raise ValueError(
                f"FiniteElementTopologyTransaction.field_transfer.result must contain one field per accepted field; got returned={len(candidate_fields)}, accepted={len(accepted.fields)}"
            )
        candidate_materials: MaterialTransaction | None = None
        if accepted.materials is not None:
            if self.material_transfer is None:
                return self._hp_retained(
                    accepted,
                    transaction,
                    auxiliary,
                    integrator,
                    "material-transfer-policy-required",
                )
            transferred = self.material_transfer(accepted.materials, transaction, args)
            if not isinstance(transferred, MaterialTransaction):
                return self._hp_retained(
                    accepted,
                    transaction,
                    auxiliary,
                    integrator,
                    "material-transfer-rejected",
                )
            candidate_materials = transferred
        candidate_auxiliary, candidate_integrator = auxiliary, integrator
        if self.history_transfer is not None:
            transferred_history = self.history_transfer(
                auxiliary, integrator, transaction, args
            )
            candidate_auxiliary = tuple(
                (str(name), jnp.asarray(value)) for name, value in transferred_history[0]
            )
            candidate_integrator = tuple(
                (str(name), jnp.asarray(value)) for name, value in transferred_history[1]
            )
            if [name for name, _ in candidate_auxiliary] != [
                name for name, _ in auxiliary
            ] or [name for name, _ in candidate_integrator] != [
                name for name, _ in integrator
            ]:
                return self._hp_retained(
                    accepted,
                    transaction,
                    auxiliary,
                    integrator,
                    "history-transfer-rejected",
                )
        elif auxiliary or integrator:
            return self._hp_retained(
                accepted,
                transaction,
                auxiliary,
                integrator,
                "history-transfer-policy-required",
            )
        certified = bool(
            jnp.asarray(
                self.certify(
                    transaction.candidate,
                    candidate_fields,
                    candidate_materials,
                    transaction,
                    args,
                )
            )
        )
        if transaction.hp_decision is not None:
            decision = transaction.hp_decision.solver_decision
            if decision is not None:
                certified = (
                    certified
                    and transaction.reanalysis is not None
                    and not decision.reanalysis_issues(transaction.reanalysis)
                )
        receipt = self._commit_hp(
            accepted,
            transaction,
            candidate_fields,
            candidate_materials,
            (auxiliary, integrator),
            (candidate_auxiliary, candidate_integrator),
            certified,
        )
        if not receipt.published:
            reason = (
                "transfer-evidence-rejected"
                if not all(receipt.transport_accepted)
                else "candidate-certification-rejected"
            )
            return self._hp_retained(
                accepted, transaction, auxiliary, integrator, reason, receipt=receipt
            )
        promoted = FiniteElementAcceptedState(
            candidate_fields,
            accepted.time,
            accepted.step,
            transaction.candidate.topology.topology_id,
            transaction.candidate.epoch_id,
            transaction.compiled_layout_id
            if transaction.compiled_layout_id is not None
            else f"{accepted.compilation_id}:hp:{transaction.transaction_id}",
            materials=candidate_materials,
            schedule_cursor=accepted.schedule_cursor,
            state_version=accepted.state_version + 1,
            transition_id=receipt.receipt_id,
        )
        return FiniteElementHPTopologyResult(
            promoted,
            transaction.candidate,
            transaction,
            candidate_auxiliary,
            candidate_integrator,
            jnp.asarray(True),
            receipt,
            "committed",
        )

    def _commit_hp(
        self,
        accepted: FiniteElementAcceptedState,
        transaction: FiniteElementHPTransaction,
        candidate_fields: tuple[Array, ...],
        candidate_materials: MaterialTransaction | None,
        histories: tuple[tuple[tuple[str, Array], ...], ...],
        candidate_histories: tuple[tuple[tuple[str, Array], ...], ...],
        certified: bool,
        /,
    ) -> CompositionRebindReceipt:
        """Stage the hp epoch, fields, materials, and histories as one rebind."""

        epochs = (transaction.accepted, transaction.candidate)
        epoch_entries = tuple(
            CompositionEntry(
                epoch,
                entry_id="hp-epoch",
                role="topology",
                owner_id=_OWNER,
                structure_id=epoch.topology.topology_id,
                revision_id=epoch.epoch_id,
                semantics_id="finite-element-hp-epoch",
            )
            for epoch in epochs
        )
        evidence = (
            jnp.all(jnp.asarray(transaction.admissible))
            & jnp.all(jnp.asarray(transaction.geometry_valid))
            & jnp.all(
                jnp.abs(jnp.asarray(transaction.conservation_error))
                <= transaction.conservation_tolerance
            )
        )
        source_entries: list[CompositionEntry] = [epoch_entries[0]]
        transports: list[CompositionTransport] = []

        def stage(
            entry_id: str,
            semantics_id: str,
            before: ArrayLike,
            after: ArrayLike,
            role: CompositionRole,
            successful: Array,
        ) -> None:
            source_entry = _state_entry(
                before,
                entry_id,
                semantics_id,
                epochs[0].epoch_id,
                epoch_entries[0],
                role=role,
            )
            target_entry = _state_entry(
                after,
                entry_id,
                semantics_id,
                epochs[1].epoch_id,
                epoch_entries[1],
                role=role,
            )
            source_entries.append(source_entry)
            transports.append(
                _policy_transport(
                    source_entry,
                    target_entry,
                    f"{transaction.transaction_id}:{entry_id}",
                    successful & jnp.all(jnp.isfinite(target_entry.value)),
                )
            )

        for index, (before, after) in enumerate(
            zip(accepted.fields, candidate_fields, strict=True)
        ):
            stage(
                f"field/{index}",
                f"finite-element-hp-field:{index}",
                before,
                after,
                "physical-state",
                evidence,
            )
        for kind, before_states, after_states in zip(
            ("auxiliary", "integrator"), histories, candidate_histories, strict=True
        ):
            for (name, before), (_, after) in zip(
                before_states, after_states, strict=True
            ):
                stage(
                    f"{kind}/{name}",
                    f"finite-element-hp-{kind}:{name}",
                    before,
                    after,
                    "history",
                    jnp.asarray(True),
                )
        if accepted.materials is not None and candidate_materials is not None:
            source_materials = _material_entry(
                accepted.materials, epochs[0].epoch_id, epoch_entries[0]
            )
            source_entries.append(source_materials)
            transports.append(
                _policy_transport(
                    source_materials,
                    _material_entry(
                        candidate_materials, epochs[1].epoch_id, epoch_entries[1]
                    ),
                    f"{transaction.transaction_id}:materials",
                    True,
                )
            )
        rebind = CompositionRebind(
            Composition(source_entries, boundary_id=accepted.accepted_id),
            reprepare=(epoch_entries[1],),
            transports=transports,
        )
        decision = (
            None
            if transaction.hp_decision is None
            else transaction.hp_decision.solver_decision
        )
        return commit_composition_rebind(
            self._complete_rebind(rebind, decision), accepted_boundary=certified
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
            epoch.discretization.field_spaces[0].vector_space.structure().shape[1:]
        )
    metadata = {
        "kind": "finite-element-hp-epoch",
        "cell_kind": epoch.topology.cell_kind,
        "topology_id": epoch.topology.topology_id,
        "field_name": field_name,
        "conformity": conformity,
        "component_shape": list(component_shape),
    }
    write_array_archive(
        path,
        manifest=metadata,
        arrays={
            "cell_global_ids": np.asarray(epoch.topology.cell_global_ids),
            "allocated": np.asarray(epoch.topology.allocated),
            "active": np.asarray(epoch.topology.active),
            "cell_degrees": np.asarray(epoch.topology.cell_degrees),
            "root_cell_ids": np.asarray(epoch.topology.root_cell_ids),
            "path_codes": np.asarray(epoch.topology.path_codes),
            "levels": np.asarray(epoch.topology.levels),
            "parent_slots": np.asarray(epoch.topology.parent_slots),
            "child_slots": np.asarray(epoch.topology.child_slots),
            "child_valid": np.asarray(epoch.topology.child_valid),
            "cell_vertices": np.asarray(epoch.geometry.cell_vertices),
            "reference_lower": np.asarray(epoch.geometry.reference_lower),
            "reference_upper": np.asarray(epoch.geometry.reference_upper),
        },
    )


def read_finite_element_hp_epoch(path: str | Path, /) -> FiniteElementHPEpoch:
    """Reconstruct one canonical hp epoch from `write_finite_element_hp_epoch`."""

    metadata, arrays = read_array_archive(path)
    if metadata.get("kind") != "finite-element-hp-epoch":
        raise ValueError("Archive is not a finite-element hp epoch.")
    topology = FiniteElementHPTopology(
        metadata["cell_kind"],
        metadata["topology_id"],
        arrays["cell_global_ids"],
        arrays["allocated"],
        arrays["active"],
        arrays["cell_degrees"],
        root_cell_ids=arrays["root_cell_ids"],
        path_codes=arrays["path_codes"],
        levels=arrays["levels"],
        parent_slots=arrays["parent_slots"],
        child_slots=arrays["child_slots"],
        child_valid=arrays["child_valid"],
    )
    geometry = FiniteElementHPGeometry(
        topology,
        arrays["cell_vertices"],
        arrays["reference_lower"],
        arrays["reference_upper"],
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
    "MaterialTopologyTransferResult",
    "refinement_parent_cells",
    "write_finite_element_hp_epoch",
]

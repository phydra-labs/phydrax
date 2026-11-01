#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from pathlib import Path
from typing import assert_never, final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._array_archive import read_array_archive, write_array_archive
from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._compensated import compensated_sum_chunks
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._nested_reference_types import (
    NestedReferenceWitnesses,
    ReferenceCellMapWitnesses,
)
from ..discretization._view_support import OwnerLocalCellGeometry
from ..discretization.fem import (
    FiniteElementDeRhamComplex,
    FiniteElementHPEpoch,
    FiniteElementHPGeometry,
    FiniteElementHPTopology,
    FiniteElementHPTransaction,
    FiniteElementPlan,
    FiniteElementTransferResult,
    prepare_finite_element_hp_epoch,
)
from ..discretization.fem._de_rham import PeriodicFiniteElementTransfer
from ..discretization.fem._distributed import (
    finite_element_dof_identity_plan,
    FiniteElementDofIdentityPlan,
    FiniteElementDofOwnership,
)
from ..discretization.fem._generic import (
    FiniteElementDiscretization,
    FiniteElementDofMap,
)
from ..discretization.fem._topology_transfer import (
    _measure_defect_bound,
    FiniteElementTopologyTransfer,
)
from ..discretization.finite_volume._automatic_remap import (
    prepare_unstructured_conservative_remap,
)
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..equations import (
    CompiledFiniteElementProblem,
    MaterialIntegrationPlan,
    MaterialTransaction,
)
from ..geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy
from ..graph._partition import GraphPartitionPlan
from ..lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionDependency,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionTransport,
)
from ..linalg import AbstractLinearOperator
from ..meshing import MeshAdaptationResult
from ..meshing._distribution import (
    _owner_group_allgather,
    _owner_group_differing_declarations,
    _owner_group_fail_if_any,
    _owner_local_state_value_identity,
    OwnerLocalGraphKeyedStateTransfer,
    OwnerLocalGraphMigrationEvidence,
    OwnerLocalGraphPartitionEvidence,
    OwnerLocalTopologyTransferRoute,
)
from ..meshing._lineage import EntityLineageKind, MeshLineage, VertexInterpolationStencil
from ..sparse import EdgeRelation, SparseLinearMap
from ..typing import checked
from ._finite_element_schedule import (
    FiniteElementAcceptedFieldLayout,
    FiniteElementAcceptedState,
    FiniteElementFieldTransferSemantics,
    FiniteElementRestartManifest,
    FiniteElementRestartStateAssociation,
    FiniteElementRestartStateLayout,
    FiniteElementRestartStateRole,
)


if TYPE_CHECKING:
    from ..meshing._device_adaptation import (
        OwnerLocalAdaptivePartTransition,
        OwnerLocalPeriodicAdaptiveResult,
    )

type FiniteElementFieldTransferContext = (
    MeshAdaptationResult
    | FiniteElementHPTransaction
    | OwnerLocalAdaptivePartTransition
    | OwnerLocalPeriodicAdaptiveResult
    | None
)


type FiniteElementFieldTransferAction = Callable[
    [
        Array,
        FiniteElementAcceptedFieldLayout,
        FiniteElementFieldTransferContext,
        object,
    ],
    ArrayLike,
]


def _reference_maps_for_adaptation(
    adaptation: MeshAdaptationResult, /
) -> tuple[NestedReferenceWitnesses | ReferenceCellMapWitnesses, ...] | None:
    transition = adaptation.transition
    if transition is None or transition.geometry_transition is None:
        return None
    exact = transition.geometry_transition
    target_ids = {
        int(identifier)
        for block in adaptation.target.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    direct = exact.reference_map
    if direct is None:
        direct_maps = ()
    elif isinstance(direct, ReferenceCellMapWitnesses):
        direct_maps = (direct,)
    else:
        direct_maps = tuple(direct)
    covered = {
        int(identifier)
        for witness in direct_maps
        for identifier in np.asarray(witness.target_cell_ids, dtype=np.int64)
    }
    if covered == target_ids:
        return direct_maps
    if exact.refinement is not None:
        refined_ids = set(
            np.asarray(exact.refinement.fine_cell_ids, dtype=np.int64).tolist()
        )
        if (
            not covered.intersection(refined_ids)
            and covered.union(refined_ids) == target_ids
        ):
            return (*direct_maps, exact.refinement)
    return None


def _vertex_p1_field_layout(
    layout: FiniteElementAcceptedFieldLayout,
    mesh: CellMesh,
    /,
) -> bool:
    if (
        layout.storage_kind != "canonical_dofs"
        or layout.periodic_quotient
        or layout.form_identity is not None
    ):
        return False
    try:
        elements = layout.field_spec.resolve(mesh)
    except (TypeError, ValueError, NotImplementedError):
        return False
    return all(
        element.form_basis is None
        and element.conformity == "H1"
        and element.degree == 1
        and element.continuity == "conforming"
        and element.representation == "point_value"
        and sum(len(values) for values in element.entity_dofs[0])
        == element.local_dof_count
        and all(not entity for level in element.entity_dofs[1:] for entity in level)
        for element in elements
    )


def _has_complete_vertex_ancestry(
    adaptation: MeshAdaptationResult,
    source: CellMesh,
    target: CellMesh,
    /,
) -> bool:
    transition = adaptation.transition
    if transition is None or not isinstance(
        transition.vertex_stencil, VertexInterpolationStencil
    ):
        return False
    stencil = transition.vertex_stencil
    if (
        not stencil.preserves_constants
        or stencil.source_entity_set_id != source.entity_set(0).entity_set_id
        or stencil.target_entity_set_id != target.entity_set(0).entity_set_id
    ):
        return False
    targets = np.asarray(stencil.target_global_ids, dtype=np.int64)
    expected_targets = np.asarray(target.vertex_global_ids, dtype=np.int64)
    sources = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    source_ids = set(int(value) for value in np.asarray(source.vertex_global_ids))
    if (
        targets.ndim != 1
        or np.unique(targets).size != targets.size
        or not np.array_equal(np.sort(targets), np.sort(expected_targets))
        or sources.ndim != 2
        or valid.shape != sources.shape
        or sources.shape[0] != targets.size
    ):
        return False
    return all(
        np.any(valid[row])
        and all(int(identifier) in source_ids for identifier in sources[row, valid[row]])
        for row in range(targets.size)
    )


def _has_unparented_target_cells(
    adaptation: MeshAdaptationResult, target: CellMesh, /
) -> bool:
    transition = adaptation.transition
    if transition is None:
        return True
    try:
        lineage = transition.lineage.entity_lineage(target.topological_dimension)
    except KeyError:
        return True
    target_ids = {
        int(identifier)
        for block in target.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    related = set(int(identifier) for identifier in np.asarray(lineage.target_global_ids))
    created = set(
        int(identifier) for identifier in np.asarray(lineage.created_target_ids)
    )
    return bool(created.intersection(target_ids)) or not target_ids <= related.union(
        created
    )


def _canonical_field_dof_counts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    layouts: Sequence[FiniteElementAcceptedFieldLayout],
    /,
) -> tuple[int, ...]:
    plan = FiniteElementPlan(
        mesh,
        tuple(layout.field_spec for layout in layouts),
        coordinate_spec=geometry,
    )
    discretization = plan.prepare()
    return tuple(int(value.global_dof_count) for value in discretization.dof_maps)


def _apply_form_operator(operator: AbstractLinearOperator, values: Array, /) -> Array:
    if values.ndim != 1 or values.shape[0] != operator.source.size:
        raise ValueError(
            "Accepted field shape does not match its declared source FE space."
        )
    return operator.mv(values)


def _finite_element_transfer(
    layout: FiniteElementAcceptedFieldLayout,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    reference_maps: tuple[NestedReferenceWitnesses | ReferenceCellMapWitnesses, ...]
    | None,
    geometry_transition: CellGeometryTransition | None,
    integration_policy: CurvedOverlapIntegrationPolicy | None,
    /,
) -> tuple[FiniteElementTransferResult, PeriodicFiniteElementTransfer | None]:
    identity = layout.form_identity
    if identity is None or layout.storage_kind != "canonical_dofs":
        raise ValueError("The accepted layout has no canonical de Rham degree identity.")
    dimension, _, order, family, twist = identity
    if (
        dimension != source_mesh.topological_dimension
        or dimension != target_mesh.topological_dimension
    ):
        raise ValueError(
            "Accepted de Rham form dimension differs from its mesh endpoints."
        )
    layout.field_spec.resolve(source_mesh)
    layout.field_spec.resolve(target_mesh)
    source_coordinate_spec = (
        None if geometry_transition is None else geometry_transition.source_geometry
    )
    target_coordinate_spec = (
        None if geometry_transition is None else geometry_transition.target_geometry
    )
    source_complex = FiniteElementDeRhamComplex(
        source_mesh,
        family=family,
        order=order,
        twist=twist,
        coordinate_spec=source_coordinate_spec,
    )
    target_complex = FiniteElementDeRhamComplex(
        target_mesh,
        family=family,
        order=order,
        twist=twist,
        coordinate_spec=target_coordinate_spec,
    )
    if layout.periodic_quotient:
        transfer = source_complex.transfer_periodic_quotient(
            target_complex,
            reference_maps=reference_maps,
            geometry_transition=geometry_transition,
            integration_policy=integration_policy,
        )
        return transfer.quotient_transfer_result, transfer
    return (
        source_complex.transfer(
            target_complex,
            reference_maps=reference_maps,
            geometry_transition=geometry_transition,
            integration_policy=integration_policy,
        ),
        None,
    )


def _hp_field_layouts_match(
    epoch: FiniteElementHPEpoch,
    layouts: Sequence[FiniteElementAcceptedFieldLayout],
    /,
) -> bool:
    discretization = epoch.discretization
    if discretization is None or len(layouts) != len(discretization.field_spaces):
        return False
    for index, (layout, space) in enumerate(
        zip(layouts, discretization.field_spaces, strict=True)
    ):
        if (
            layout.storage_kind != "hp_cell_slots"
            or layout.name != space.name
            or layout.component_shape != tuple(space.vector_space.structure().shape[1:])
        ):
            return False
        try:
            declared = layout.field_spec.resolve(epoch.mesh)
        except (TypeError, ValueError, NotImplementedError):
            return False
        actual = discretization.elements[index]
        if len(declared) != len(actual) or any(
            (left.conformity, left.value_spec.value_spec_id, left.representation)
            != (right.conformity, right.value_spec.value_spec_id, right.representation)
            for left, right in zip(declared, actual, strict=True)
        ):
            return False
    return True


@final
class FiniteElementConservativeContentEvidence(StrictModule, NonTrainableState):
    """Audited overlap coverage and measure action for one cell-average field."""

    layout_id: str = eqx.field(static=True)
    remap_plan_id: str = eqx.field(static=True)
    coverage_evidence_id: str = eqx.field(static=True)
    transfer: FiniteElementTopologyTransfer
    coverage_complete: bool = eqx.field(static=True)
    source_measures: Array
    target_measures: Array
    source_integral: Array
    target_integral: Array
    conservation_defect: Array
    conservation_bound: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout_id: str,
        remap_plan_id: str,
        coverage_evidence_id: str,
        transfer: FiniteElementTopologyTransfer,
        coverage_complete: bool,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        source_integral: ArrayLike,
        target_integral: ArrayLike,
        conservation_defect: ArrayLike,
        conservation_bound: float,
        /,
    ) -> None:
        layout = str(layout_id).strip()
        remap = str(remap_plan_id).strip()
        coverage = str(coverage_evidence_id).strip()
        if not layout or not remap or not coverage:
            raise ValueError(
                "Conservative content evidence identities must be non-empty."
            )
        if not isinstance(transfer, FiniteElementTopologyTransfer):
            raise TypeError("transfer must be FiniteElementTopologyTransfer.")
        if not transfer.conservative:
            raise ValueError(
                "Conservative content evidence requires a certified transfer."
            )
        if not isinstance(coverage_complete, (bool, np.bool_)) or not coverage_complete:
            raise ValueError(
                "Conservative content evidence requires complete overlap coverage."
            )
        source_mass = np.asarray(source_measures, dtype=np.float64)
        target_mass = np.asarray(target_measures, dtype=np.float64)
        source_integral_ = jnp.asarray(source_integral)
        target_integral_ = jnp.asarray(target_integral)
        defect = jnp.asarray(conservation_defect)
        if (
            source_mass.shape != (transfer.source_size,)
            or target_mass.shape != (transfer.target_size,)
            or np.any(~np.isfinite(source_mass))
            or np.any(~np.isfinite(target_mass))
            or np.any(source_mass <= 0.0)
            or np.any(target_mass <= 0.0)
            or source_integral_.shape != target_integral_.shape
            or source_integral_.shape != defect.shape
            or np.any(~np.isfinite(np.asarray(source_integral_)))
            or np.any(~np.isfinite(np.asarray(target_integral_)))
            or np.any(~np.isfinite(np.asarray(defect)))
        ):
            raise ValueError("Conservative content evidence arrays are malformed.")
        bound = float(conservation_bound)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("conservation_bound must be finite and non-negative.")
        if np.any(np.abs(np.asarray(defect)) > bound):
            raise ValueError(
                "Conservative content defect exceeds its certified action bound."
            )
        self.layout_id = layout
        self.remap_plan_id = remap
        self.coverage_evidence_id = coverage
        self.transfer = transfer
        self.coverage_complete = True
        self.source_measures = jnp.asarray(source_mass)
        self.target_measures = jnp.asarray(target_mass)
        self.source_integral = source_integral_
        self.target_integral = target_integral_
        self.conservation_defect = defect
        self.conservation_bound = bound
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "finite-element-conservative-cell-average-evidence",
                "layout_id": layout,
                "remap_plan_id": remap,
                "coverage_evidence_id": coverage,
                "transfer_id": transfer.transfer_id,
                "source_measures": array_tree_fingerprint(source_mass),
                "target_measures": array_tree_fingerprint(target_mass),
                "source_integral": array_tree_fingerprint(np.asarray(source_integral_)),
                "target_integral": array_tree_fingerprint(np.asarray(target_integral_)),
                "conservation_defect": array_tree_fingerprint(np.asarray(defect)),
                "conservation_bound": bound,
            }
        )


def _unstructured_finite_volume_geometry(
    mesh: CellMesh, field_name: str, /
) -> UnstructuredFiniteVolumeDiscretization:
    kinds = {block.cell_kind for block in mesh.blocks}
    if mesh.topological_dimension == 2 and kinds == {"triangle"}:
        connectivity: dict[str, np.ndarray] = {
            "triangles": np.concatenate(
                tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks)
            )
        }
    elif mesh.topological_dimension == 3 and kinds == {"tetrahedron"}:
        connectivity = {
            "tetrahedra": np.concatenate(
                tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks)
            )
        }
    else:
        raise NotImplementedError(
            "Conservative cell averages require simplex FV geometry."
        )
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    return UnstructuredFiniteVolumePlan(
        np.asarray(mesh.coordinates, dtype=np.float64),
        triangles=connectivity.get("triangles"),
        tetrahedra=connectivity.get("tetrahedra"),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        cell_global_ids=cell_ids,
        field_name=field_name,
    ).prepare(numeric_version=mesh.numeric_version)


def _cell_average_dof_rows(
    mesh: CellMesh,
    dof_map: FiniteElementDofMap,
    finite_volume: UnstructuredFiniteVolumeDiscretization,
    /,
) -> np.ndarray:
    if len(dof_map.cell_dofs) != len(mesh.blocks):
        raise ValueError("DG0 DOF maps do not align with mesh blocks.")
    dof_by_cell: dict[int, int] = {}
    for block, cell_dofs in zip(mesh.blocks, dof_map.cell_dofs, strict=True):
        routes = np.asarray(cell_dofs, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        if routes.shape != (identifiers.size, 1):
            raise ValueError("Conservative cell averages require one DOF per cell.")
        for identifier, dof in zip(identifiers.tolist(), routes[:, 0], strict=True):
            if int(identifier) in dof_by_cell:
                raise ValueError("Cell-average source cell IDs must be unique.")
            dof_by_cell[int(identifier)] = int(dof)
    finite_volume_ids = np.asarray(finite_volume.cell_global_ids, dtype=np.int64)
    try:
        rows = np.asarray(
            [dof_by_cell[int(identifier)] for identifier in finite_volume_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Finite-volume overlap remap changed cell identities."
        ) from error
    dof_count = int(dof_map.global_dof_count)
    if (
        rows.size != dof_count
        or np.unique(rows).size != dof_count
        or not np.array_equal(np.sort(rows), np.arange(dof_count, dtype=np.int64))
    ):
        raise ValueError("DG0 cell DOFs must form one canonical route per cell.")
    return rows


def _conservative_cell_average_transfer(
    layout: FiniteElementAcceptedFieldLayout,
    field: Array,
    adaptation: MeshAdaptationResult,
    /,
) -> tuple[Array, FiniteElementConservativeContentEvidence]:
    source_result, target_result = adaptation.source, adaptation.target
    source_mesh, target_mesh = source_result.mesh, target_result.mesh
    from ..meshing._adaptation import _piecewise_linear_geometry

    if not _piecewise_linear_geometry(source_mesh, source_result.geometry) or not (
        _piecewise_linear_geometry(target_mesh, target_result.geometry)
    ):
        raise ValueError("Conservative cell averages require piecewise-linear geometry.")
    source_fv = _unstructured_finite_volume_geometry(source_mesh, layout.name)
    target_fv = _unstructured_finite_volume_geometry(target_mesh, layout.name)
    source_dof_map = (
        FiniteElementPlan(
            source_mesh,
            (layout.field_spec,),
        )
        .prepare()
        .dof_maps[0]
    )
    target_dof_map = (
        FiniteElementPlan(
            target_mesh,
            (layout.field_spec,),
        )
        .prepare()
        .dof_maps[0]
    )
    source_dof_rows = _cell_average_dof_rows(source_mesh, source_dof_map, source_fv)
    target_dof_rows = _cell_average_dof_rows(target_mesh, target_dof_map, target_fv)
    source_dof_count = int(source_dof_map.global_dof_count)
    target_dof_count = int(target_dof_map.global_dof_count)
    values = jnp.asarray(field)
    if (
        values.ndim == 0
        or values.shape[0] != source_dof_count
        or tuple(values.shape[1:]) != layout.component_shape
        or not jnp.issubdtype(values.dtype, jnp.floating)
        or not bool(jnp.all(jnp.isfinite(values)))
    ):
        raise ValueError(
            "Conservative cell-average fields must be finite real DG0 values."
        )
    prepared = prepare_unstructured_conservative_remap(
        source_fv,
        target_fv,
        provenance=f"finite-element-cell-average:{layout.layout_id}",
    )
    if not prepared.succeeded or prepared.plan is None:
        raise ValueError(
            f"Physical overlap did not certify complete coverage: {prepared.reason}"
        )
    remap = prepared.plan
    coverage_complete = bool(np.asarray(remap.report.coverage_complete))
    if not remap.require_complete or not coverage_complete:
        raise ValueError(
            "Conservative cell-average remap lacks complete overlap coverage."
        )
    source_cells = np.asarray(remap.source_indices, dtype=np.int64)
    target_cells = np.asarray(remap.target_routes, dtype=np.int64)
    source_indices = source_dof_rows[source_cells]
    target_indices = target_dof_rows[target_cells]
    overlap = np.asarray(remap.intersection_measures, dtype=np.float64)
    source_cell_measures = np.asarray(source_fv.cell_volumes, dtype=np.float64)
    target_cell_measures = np.asarray(target_fv.cell_volumes, dtype=np.float64)
    target_denominators = target_cell_measures[target_cells]
    if (
        np.any(~np.isfinite(overlap))
        or np.any(overlap < 0.0)
        or np.any(~np.isfinite(target_denominators))
        or np.any(target_denominators <= 0.0)
    ):
        raise ValueError("Physical overlap measures are not finite and positive.")
    coefficients = overlap / target_denominators
    relation = EdgeRelation(
        source_indices,
        target_indices,
        source_size=source_dof_count,
        target_size=target_dof_count,
    )
    primal = SparseLinearMap(
        relation,
        coefficients,
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-conservative-cell-average-overlap",
                "layout": layout.layout_id,
                "remap": remap.plan_id,
            }
        ),
    )
    source_measures = np.zeros((source_dof_count,), dtype=np.float64)
    target_measures = np.zeros((target_dof_count,), dtype=np.float64)
    source_measures[source_dof_rows] = source_cell_measures
    target_measures[target_dof_rows] = target_cell_measures
    transfer = FiniteElementTopologyTransfer(
        primal,
        source_mesh.topology_id,
        target_mesh.topology_id,
        preserves_constants=True,
        conservative=True,
        positivity_preserving=True,
        source_measures=source_measures,
        target_measures=target_measures,
    )
    target_values = transfer.apply(values)
    source_cell_values = jnp.asarray(values[source_dof_rows], dtype=jnp.float64)
    target_cell_values = jnp.asarray(target_values[target_dof_rows], dtype=jnp.float64)
    conservation_defect = remap.conservation_defect(
        source_cell_values, target_cell_values
    )
    trailing = (1,) * (source_cell_values.ndim - 1)
    source_content = np.asarray(
        source_cell_values
        * jnp.asarray(source_cell_measures, dtype=jnp.float64).reshape((-1,) + trailing),
        dtype=np.float64,
    )
    target_content = np.asarray(
        target_cell_values
        * jnp.asarray(target_cell_measures, dtype=jnp.float64).reshape((-1,) + trailing),
        dtype=np.float64,
    )
    conservation_bound = _measure_defect_bound(
        np.dtype(np.float64),
        transfer.action_condition,
        source_content,
        target_content,
    )
    source_integral = compensated_sum_chunks(
        (source_content,), output_ndim=source_cell_values.ndim - 1
    )
    target_integral = compensated_sum_chunks(
        (target_content,), output_ndim=target_cell_values.ndim - 1
    )
    evidence = FiniteElementConservativeContentEvidence(
        layout.layout_id,
        remap.plan_id,
        remap.coverage_evidence_id,
        transfer,
        coverage_complete,
        source_measures,
        target_measures,
        source_integral,
        target_integral,
        conservation_defect,
        conservation_bound,
    )
    return target_values, evidence


class FiniteElementTopologyResult(StrictModule, NonTrainableState):
    """Promoted state and certified mesh, form, and conservative transfers."""

    state: FiniteElementAcceptedState
    mesh: CellMesh
    adaptation: MeshAdaptationResult | None
    committed: Array
    diagnostics: object
    periodic_transfers: tuple[PeriodicFiniteElementTransfer, ...] = ()
    field_transfers: tuple[FiniteElementTransferResult, ...] = ()
    conservative_transfers: tuple[FiniteElementConservativeContentEvidence, ...] = ()


class FiniteElementHPTopologyResult(StrictModule, NonTrainableState):
    state: FiniteElementAcceptedState
    epoch: FiniteElementHPEpoch
    transaction: FiniteElementHPTransaction | None
    auxiliary_state: tuple[tuple[str, Array], ...]
    integrator_state: tuple[tuple[str, Array], ...]
    committed: Array
    diagnostics: object


class FiniteElementRepartitionResult(StrictModule, NonTrainableState):
    """Atomic accepted FE ownership rebind and its target restart state."""

    state: FiniteElementAcceptedState
    manifest: FiniteElementRestartManifest
    committed: Array
    diagnostics: str
    rebind_receipt: CompositionRebindReceipt | None
    field_transfers: tuple[OwnerLocalGraphKeyedStateTransfer, ...]
    payload_transfers: tuple[OwnerLocalGraphKeyedStateTransfer, ...]
    target_identity_plans: tuple[FiniteElementDofIdentityPlan, ...]


@final
class FiniteElementOwnerLocalFieldTransferEvidence(StrictModule, NonTrainableState):
    """Owner-local FE field transform bound to exact source and target rows."""

    field_layout_id: str = eqx.field(static=True)
    semantics: FiniteElementFieldTransferSemantics = eqx.field(static=True)
    adaptive_result_id: str = eqx.field(static=True)
    vertex_stencil_id: str | None = eqx.field(static=True)
    lineage_id: str | None = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    target_plan_id: str = eqx.field(static=True)
    source_structure_id: str = eqx.field(static=True)
    target_structure_id: str = eqx.field(static=True)
    source_owner_map_id: str = eqx.field(static=True)
    target_owner_map_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    source_content_root: str = eqx.field(static=True)
    target_content_root: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    source_integral: Array | None
    target_integral: Array | None
    conservation_defect: Array | None
    conservation_bound: float | None = eqx.field(static=True)
    conservation_evidence_id: str | None = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    target_entity_keys: Array
    target_values: Array
    successful: Array

    def __init__(
        self,
        field_layout_id: str,
        semantics: FiniteElementFieldTransferSemantics,
        adaptive_result_id: str,
        vertex_stencil_id: str | None,
        lineage_id: str | None,
        source_plan_id: str,
        target_plan_id: str,
        source_structure_id: str,
        target_structure_id: str,
        source_owner_map_id: str,
        target_owner_map_id: str,
        action_id: str,
        source_content_root: str,
        target_content_root: str,
        target_entity_keys: ArrayLike,
        target_values: ArrayLike,
        /,
        *,
        conservative: bool = False,
        source_integral: ArrayLike | None = None,
        target_integral: ArrayLike | None = None,
        conservation_defect: ArrayLike | None = None,
        conservation_bound: float | None = None,
    ) -> None:
        identities = (
            field_layout_id,
            adaptive_result_id,
            source_plan_id,
            target_plan_id,
            source_structure_id,
            target_structure_id,
            source_owner_map_id,
            target_owner_map_id,
            action_id,
            source_content_root,
            target_content_root,
        )
        keys = np.asarray(target_entity_keys)
        values = jnp.asarray(target_values)
        if (
            any(not isinstance(value, str) or not value for value in identities)
            or (
                vertex_stencil_id is not None
                and (not isinstance(vertex_stencil_id, str) or not vertex_stencil_id)
            )
            or not isinstance(semantics, FiniteElementFieldTransferSemantics)
            or keys.dtype != np.dtype(np.int64)
            or keys.ndim != 2
            or keys.shape[1] != 3
            or np.any(keys < 0)
            or np.unique(keys, axis=0).shape[0] != keys.shape[0]
            or values.ndim == 0
            or values.shape[0] != keys.shape[0]
            or not jnp.issubdtype(values.dtype, jnp.inexact)
            or not bool(np.asarray(jnp.all(jnp.isfinite(values))))
        ):
            raise ValueError("Owner-local FE field transfer evidence is malformed.")
        if lineage_id is not None and (not isinstance(lineage_id, str) or not lineage_id):
            raise ValueError("Owner-local FE transfer lineage identity is malformed.")
        if not isinstance(conservative, (bool, np.bool_)):
            raise TypeError("conservative must be bool.")
        if conservative:
            if (
                source_integral is None
                or target_integral is None
                or conservation_defect is None
                or conservation_bound is None
            ):
                raise ValueError(
                    "Conservative owner-local FE transfer needs integral evidence."
                )
            source_integral_ = jnp.asarray(source_integral)
            target_integral_ = jnp.asarray(target_integral)
            defect_ = jnp.asarray(conservation_defect)
            bound = float(conservation_bound)
            source_host = np.asarray(source_integral_)
            target_host = np.asarray(target_integral_)
            defect_host = np.asarray(defect_)
            if (
                source_host.shape != target_host.shape
                or source_host.shape != defect_host.shape
                or not np.issubdtype(source_host.dtype, np.inexact)
                or not np.issubdtype(target_host.dtype, np.inexact)
                or not np.issubdtype(defect_host.dtype, np.inexact)
                or np.any(~np.isfinite(source_host))
                or np.any(~np.isfinite(target_host))
                or np.any(~np.isfinite(defect_host))
                or not np.array_equal(defect_host, target_host - source_host)
                or not np.isfinite(bound)
                or bound < 0.0
                or np.any(np.abs(defect_host) > bound)
            ):
                raise ValueError(
                    "Conservative owner-local FE integral evidence is invalid."
                )
            conservation_id = canonical_fingerprint(
                {
                    "kind": "finite-element-owner-local-conservation",
                    "field_layout": field_layout_id,
                    "adaptive_result": adaptive_result_id,
                    "source_plan": source_plan_id,
                    "target_plan": target_plan_id,
                    "source_integral": array_tree_fingerprint(source_host),
                    "target_integral": array_tree_fingerprint(target_host),
                    "defect": array_tree_fingerprint(defect_host),
                    "bound": bound,
                }
            )
        else:
            if any(
                value is not None
                for value in (
                    source_integral,
                    target_integral,
                    conservation_defect,
                    conservation_bound,
                )
            ):
                raise ValueError(
                    "Nonconservative transfers cannot carry integral claims."
                )
            source_integral_ = target_integral_ = defect_ = None
            bound = None
            conservation_id = None
        self.field_layout_id = field_layout_id
        self.semantics = semantics
        self.adaptive_result_id = adaptive_result_id
        self.vertex_stencil_id = vertex_stencil_id
        self.lineage_id = lineage_id
        self.source_plan_id = source_plan_id
        self.target_plan_id = target_plan_id
        self.source_structure_id = source_structure_id
        self.target_structure_id = target_structure_id
        self.source_owner_map_id = source_owner_map_id
        self.target_owner_map_id = target_owner_map_id
        self.action_id = action_id
        self.source_content_root = source_content_root
        self.target_content_root = target_content_root
        self.target_entity_keys = jnp.asarray(keys)
        self.target_values = values
        self.conservative = bool(conservative)
        self.source_integral = source_integral_
        self.target_integral = target_integral_
        self.conservation_defect = defect_
        self.conservation_bound = bound
        self.conservation_evidence_id = conservation_id
        self.successful = jnp.asarray(True, dtype=jnp.bool_)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-owner-local-topology-field-transfer",
                "field_layout": field_layout_id,
                "semantics": semantics.value,
                "adaptive_result": adaptive_result_id,
                "vertex_stencil": vertex_stencil_id,
                "lineage": lineage_id,
                "source_plan": source_plan_id,
                "target_plan": target_plan_id,
                "source_structure": source_structure_id,
                "target_structure": target_structure_id,
                "source_owner_map": source_owner_map_id,
                "target_owner_map": target_owner_map_id,
                "action": action_id,
                "source_content": source_content_root,
                "target_content": target_content_root,
                "target_keys": array_tree_fingerprint(keys),
                "target_values": array_tree_fingerprint(np.asarray(values)),
                "conservation_evidence": conservation_id,
            }
        )

    def composition_transport(
        self,
        source: CompositionEntry,
        target: CompositionEntry,
        /,
        *,
        target_revision_id: str,
    ) -> CompositionTransport:
        if (
            not isinstance(source, CompositionEntry)
            or not isinstance(target, CompositionEntry)
            or source.entry_id != target.entry_id
            or source.structure_id != self.source_structure_id
            or array_tree_fingerprint(np.asarray(target.value))
            != array_tree_fingerprint(np.asarray(self.target_values))
            or target.revision_id != target_revision_id
        ):
            raise ValueError(
                "Owner-local FE transfer entries differ from their evidence."
            )
        # Adaptive transfers remap content onto new entities. Conservative fields
        # report their allreduced integrals and roundoff bound; the exact source
        # and target roots stay bound through the transfer route identity.
        source_integral = self.source_integral
        bound = self.conservation_bound
        conservative = (
            self.conservative and source_integral is not None and bound is not None
        )
        return CompositionTransport(
            "physical-remap",
            (source.entry_id,),
            (target,),
            source_structure_ids=(source.structure_id,),
            route_id=self.transfer_id,
            successful=self.successful,
            source_content=source_integral if conservative else None,
            target_content=self.target_integral if conservative else None,
            content_tolerance=(
                jnp.full(jnp.shape(source_integral), bound, dtype=jnp.float64)
                if conservative and source_integral is not None and bound is not None
                else None
            ),
        )


class FiniteElementOwnerLocalTopologyResult(StrictModule, NonTrainableState):
    """Atomic owner-local FE topology transition and restart-state result."""

    state: FiniteElementAcceptedState
    manifest: FiniteElementRestartManifest
    committed: Array
    diagnostics: str
    rebind_receipt: CompositionRebindReceipt | None
    field_transfers: tuple[FiniteElementOwnerLocalFieldTransferEvidence, ...]
    payload_transfers: tuple[
        FiniteElementOwnerLocalFieldTransferEvidence | OwnerLocalGraphKeyedStateTransfer,
        ...,
    ]
    target_identity_plans: tuple[FiniteElementDofIdentityPlan, ...]
    adaptive_result_id: str | None = eqx.field(static=True)
    vertex_stencil_id: str | None = eqx.field(static=True)
    lineage_id: str | None = eqx.field(static=True, default=None)


@checked
def partition_owner_local_finite_element_graph(
    source_view: OwnerLocalCellGeometry,
    source_compiled: CompiledFiniteElementProblem,
    plan: GraphPartitionPlan,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str = "finite_element_owner_local_graph",
) -> OwnerLocalGraphPartitionEvidence:
    """Partition current owner rows using integer FE work, never physical measure."""
    import jax

    from ..discretization._cell_mesh import EmptyLocalCellMeshAuthority
    from ..graph._distributed_partition import (
        DistributedWeightedCSRGraph,
        partition_graph_distributed,
    )

    if (
        not isinstance(source_view, OwnerLocalCellGeometry)
        or not isinstance(source_compiled, CompiledFiniteElementProblem)
        or not isinstance(plan, GraphPartitionPlan)
        or not isinstance(execution_group, ExecutionGroup)
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise TypeError("FE graph partitioning requires typed current source inputs.")
    members = execution_group.spec.process_indices
    if (
        not execution_group.is_member
        or len(execution_group.devices) != len(members)
        or tuple(device.process_index for device in execution_group.devices) != members
        or len(members) != plan.part_count
    ):
        raise ValueError("FE graph partition inputs differ from their owner group.")
    rank = members.index(jax.process_index())
    if source_view.owner_rank != rank or source_view.periodic_topology_id is None:
        raise ValueError("FE graph partition requires a current periodic owner view.")
    if not source_view.periodic_face_coverage_id:
        raise ValueError("FE graph partition requires its exact periodic face coverage.")
    source_view.require_owner_directory_current(
        execution_group,
        axis_name=f"{axis_name}_directory",
    )
    discretization = source_compiled.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("FE graph work weights require a prepared FE discretization.")
    local_mesh, local_geometry = source_view.reconstruct_local_geometry()
    compiled_mesh = discretization.mesh
    if local_mesh is None:
        authority = compiled_mesh.empty_local_authority
        if (
            local_geometry is not None
            or not isinstance(authority, EmptyLocalCellMeshAuthority)
            or authority.source_topology_id != source_view.source_topology_id
            or authority.source_geometry_id != source_view.source_geometry_id
            or authority.distribution_id != source_view.distribution_id
            or authority.local_rank != rank
            or authority.part_count != len(members)
        ):
            raise ValueError(
                "Empty FE graph source lacks its current partition authority."
            )
    elif (
        local_geometry is None
        or compiled_mesh.mesh_id != local_mesh.mesh_id
        or source_view.local_mesh_id != local_mesh.mesh_id
    ):
        raise ValueError(
            "FE graph discretization differs from its exact local source mesh."
        )
    owned_ids = np.asarray(source_view.owned_cell_ids, dtype=np.int64)
    cell_ids = np.asarray(source_view.cell_ids, dtype=np.int64)
    owned = np.asarray(source_view.owned, dtype=np.bool_)
    expected_owned_ids = np.sort(cell_ids[owned], kind="stable")
    if not np.array_equal(owned_ids, expected_owned_ids) or not np.array_equal(
        np.asarray(source_view.owned_cell_owner_ranks, dtype=np.int32),
        np.full(owned_ids.shape, rank, dtype=np.int32),
    ):
        raise ValueError("FE work rows differ from the current owned cell directory.")
    work_by_cell: dict[int, int] = {}
    if not discretization.dof_maps:
        raise ValueError("FE graph work requires at least one accepted field.")
    for dof_map in discretization.dof_maps:
        if len(dof_map.cell_dofs) != len(compiled_mesh.blocks):
            raise ValueError("FE DOF maps omit local cell blocks.")
        components = int(np.prod(dof_map.component_shape, dtype=np.int64))
        components = max(1, components)
        for block, routes in zip(compiled_mesh.blocks, dof_map.cell_dofs, strict=True):
            cell_routes = np.asarray(routes)
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            if (
                cell_routes.ndim != 2
                or cell_routes.shape[0] != identifiers.size
                or identifiers.ndim != 1
                or np.unique(identifiers).size != identifiers.size
            ):
                raise ValueError("FE element work rows are malformed.")
            work = int(cell_routes.shape[1]) * components
            if identifiers.size and work <= 0:
                raise ValueError("FE element work per cell must be positive.")
            for identifier in identifiers:
                cell_id = int(identifier)
                work_by_cell[cell_id] = work_by_cell.get(cell_id, 0) + work
    try:
        vertex_weights = np.asarray(
            [work_by_cell[int(identifier)] for identifier in owned_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("FE work weights omit an owned graph cell.") from error
    if vertex_weights.size and np.any(vertex_weights <= 0):
        raise ValueError("FE graph work weights must be positive integers.")
    graph = DistributedWeightedCSRGraph(
        source_view.cell_neighbor_offsets,
        source_view.cell_neighbor_ids,
        entity_ids=owned_ids,
        owner_ranks=source_view.owned_cell_owner_ranks,
        neighbor_owner_ranks=source_view.cell_neighbor_owner_ranks,
        edge_weights=source_view.cell_neighbor_weights,
        vertex_weights=vertex_weights,
    )
    result = partition_graph_distributed(
        graph,
        plan,
        execution_group,
        axis_name=axis_name,
    )
    if not result.feasible:
        raise ValueError(
            "FE graph partition did not reach the plan's part capacities; "
            f"part weights {np.asarray(result.part_weights).tolist()} exceed "
            f"capacities {np.asarray(result.capacity_weights).tolist()}."
        )
    if (
        not np.array_equal(np.asarray(result.entity_ids, dtype=np.int64), owned_ids)
        or np.asarray(result.owners).shape != owned_ids.shape
        or result.plan_id != plan.plan_id
    ):
        raise ValueError("FE graph result does not bind its integer-work owner rows.")
    return OwnerLocalGraphPartitionEvidence(
        result,
        source_view.local_view_id,
        source_view.periodic_face_coverage_id,
        source_view.distribution_id,
    )


@dataclass(slots=True)
class _FiniteElementRepartitionContext:
    source_manifest: FiniteElementRestartManifest
    source_compiled: CompiledFiniteElementProblem
    target_compiled: CompiledFiniteElementProblem
    source_plans: tuple[FiniteElementDofIdentityPlan, ...]
    target_plans: tuple[FiniteElementDofIdentityPlan, ...]
    migration: OwnerLocalGraphMigrationEvidence | OwnerLocalTopologyTransferRoute
    source_composition: Composition
    execution_group: ExecutionGroup
    source_geometry_view: OwnerLocalCellGeometry
    target_geometry_view: OwnerLocalCellGeometry
    state_entry_id: str
    geometry_entry_id: str
    field_ids: tuple[tuple[str, str], ...]
    material_entry_id: str | None
    auxiliary_ids: tuple[tuple[str, str], ...]
    integrator_ids: tuple[tuple[str, str], ...]
    target_auxiliary_layouts: tuple[FiniteElementRestartStateLayout, ...] | None
    target_integrator_layouts: tuple[FiniteElementRestartStateLayout, ...] | None
    target_material_layouts: tuple[FiniteElementRestartStateLayout, ...] | None
    source_coordinate_spec: CellGeometrySpec | None = dataclass_field(init=False)
    target_coordinate_spec: CellGeometrySpec | None = dataclass_field(init=False)
    source_material_plan: MaterialIntegrationPlan | None
    target_material_plan: MaterialIntegrationPlan | None
    retained: tuple[str, ...]
    invalidated: tuple[str, ...]
    reprepared: tuple[CompositionEntry, ...]
    rank: int
    parts: int
    axis_name: str
    topology_change: bool = False
    adaptive_result: object | None = None
    adaptive_result_id: str | None = None
    vertex_stencil: VertexInterpolationStencil | None = None
    target_global_geometry_id: str = ""
    target_local_fields: list[Array] = dataclass_field(default_factory=list)
    global_geometry_id: str = ""
    source_state: FiniteElementAcceptedState = dataclass_field(init=False)
    source_discretization: FiniteElementDiscretization = dataclass_field(init=False)
    target_discretization: FiniteElementDiscretization = dataclass_field(init=False)
    source_mesh: CellMesh = dataclass_field(init=False)
    target_mesh: CellMesh = dataclass_field(init=False)
    source_auxiliary: tuple[tuple[str, Array], ...] = dataclass_field(init=False)
    source_integrator: tuple[tuple[str, Array], ...] = dataclass_field(init=False)
    source_auxiliary_layouts: tuple[FiniteElementRestartStateLayout, ...] | None = (
        dataclass_field(init=False)
    )
    source_integrator_layouts: tuple[FiniteElementRestartStateLayout, ...] | None = (
        dataclass_field(init=False)
    )
    source_materials: MaterialTransaction | None = dataclass_field(init=False)
    field_source_entries: list[CompositionEntry] = dataclass_field(default_factory=list)
    field_transfers: list[OwnerLocalGraphKeyedStateTransfer] = dataclass_field(
        default_factory=list
    )
    topology_field_transfers: list[FiniteElementOwnerLocalFieldTransferEvidence] = (
        dataclass_field(default_factory=list)
    )
    payload_transfers: list[OwnerLocalGraphKeyedStateTransfer] = dataclass_field(
        default_factory=list
    )
    topology_payload_transfers: list[FiniteElementOwnerLocalFieldTransferEvidence] = (
        dataclass_field(default_factory=list)
    )
    payload_records: list[
        tuple[
            str,
            str,
            CompositionEntry,
            FiniteElementRestartStateLayout,
            Array,
            OwnerLocalGraphKeyedStateTransfer
            | FiniteElementOwnerLocalFieldTransferEvidence
            | None,
            str,
        ]
    ] = dataclass_field(default_factory=list)
    payload_roots: list[tuple[str, str, str]] = dataclass_field(default_factory=list)
    state_entry: CompositionEntry | None = None
    material_entry: CompositionEntry | None = None
    geometry_entry: CompositionEntry | None = None
    state_value: object = None
    schema_id: str = ""
    global_geometry_id: str = ""
    material_root: str | None = None
    target_materials: MaterialTransaction | None = None
    auxiliary_values: tuple[tuple[str, Array], ...] = ()
    integrator_values: tuple[tuple[str, Array], ...] = ()
    target_state: FiniteElementAcceptedState | None = None
    target_manifest: FiniteElementRestartManifest | None = None
    target_state_entry: CompositionEntry | None = None
    transports: list[CompositionTransport] = dataclass_field(default_factory=list)
    staged_entries: dict[str, CompositionEntry] = dataclass_field(default_factory=dict)
    rebind: CompositionRebind | None = None

    def __post_init__(self) -> None:
        self.source_state = self.source_manifest.state
        source_discretization = self.source_compiled.discretization
        target_discretization = self.target_compiled.discretization
        if not isinstance(
            source_discretization, FiniteElementDiscretization
        ) or not isinstance(target_discretization, FiniteElementDiscretization):
            raise TypeError(
                "FE repartition requires prepared FiniteElementDiscretization authorities."
            )
        self.source_discretization = source_discretization
        self.target_discretization = target_discretization
        self.source_mesh = source_discretization.mesh
        self.target_mesh = target_discretization.mesh
        self.source_auxiliary = self.source_manifest.auxiliary_state
        self.source_integrator = self.source_manifest.integrator_state
        self.source_auxiliary_layouts = self.source_manifest.auxiliary_layouts
        self.source_integrator_layouts = self.source_manifest.integrator_layouts
        self.source_materials = self.source_state.materials
        self.target_materials = self.source_materials
        self.source_coordinate_spec = None
        self.target_coordinate_spec = None


@dataclass(slots=True)
class _FiniteElementOwnerLocalFieldActionContext:
    repartition: _FiniteElementRepartitionContext
    layout: FiniteElementAcceptedFieldLayout
    source_plan: FiniteElementDofIdentityPlan
    target_plan: FiniteElementDofIdentityPlan
    source_ownership: FiniteElementDofOwnership
    target_ownership: FiniteElementDofOwnership
    source_entry: CompositionEntry
    source_values: Array
    adaptive_result: FiniteElementFieldTransferContext | None
    vertex_stencil: VertexInterpolationStencil | None
    field_index: int

    def request_source_rows(
        self,
        entity_keys: ArrayLike,
        /,
        *,
        axis_name: str,
    ) -> Array:
        context = self.repartition
        keys = np.asarray(entity_keys)
        if keys.dtype != np.dtype(np.int64) or keys.ndim != 2 or keys.shape[1] != 3:
            raise ValueError(
                "FE transfer source requests need canonical three-column keys."
            )
        # The owned source rows stay on their owners (an exact key bijection);
        # requested rows, repeated or remote, are then fetched from those owners.
        owned_keys = np.asarray(self.source_ownership.dof_keys, dtype=np.int64)
        transfer = context.migration.transfer_keyed_values(
            _owner_local_keyed_declaration(context, self.source_entry),
            owned_keys,
            self.source_ownership.owner_ranks,
            self.source_values,
            owned_keys,
            np.full((owned_keys.shape[0],), context.rank, dtype=np.int32),
            context.execution_group,
            target_structure_id=canonical_fingerprint(
                {
                    "kind": "finite-element-owner-local-source-row-request",
                    "layout": self.layout.layout_id,
                    "identity": self.source_plan.identity_id,
                }
            ),
            key_identity_id=self.source_plan.identity_id,
            axis_name=axis_name,
            value_fiber_shape=tuple(int(size) for size in self.source_values.shape[1:]),
        )
        return transfer.gather_target_local_values(
            keys,
            context.execution_group,
            axis_name=f"{axis_name}_local",
        )


def _owner_local_target_field_structure(
    context: _FiniteElementRepartitionContext,
    layout: FiniteElementAcceptedFieldLayout,
    ownership: FiniteElementDofOwnership,
    /,
) -> str:
    """Group-wide structure of one owned FE field on the target placement.

    Every owner, including an empty one, declares the same structure: the field
    layout, canonical DOF identity, global target topology/geometry and owner
    directory. Rank-local ownership rows bind the accepted target state instead.
    """
    return canonical_fingerprint(
        {
            "kind": "owner-local-finite-element-owned-field",
            "layout": layout.layout_id,
            "identity": ownership.identity_id,
            "target_topology": context.target_geometry_view.source_topology_id,
            "target_geometry": context.target_global_geometry_id,
            "target_directory": context.migration.target_directory_id,
        }
    )


def _owner_local_keyed_revision(
    context: _FiniteElementRepartitionContext, entry_id: str, /
) -> str:
    """Group-wide revision under which one rank's FE shard enters a keyed transfer."""
    state = context.source_state
    return canonical_fingerprint(
        {
            "kind": "finite-element-owner-local-keyed-declaration",
            "entry": entry_id,
            "state_entry": context.state_entry_id,
            "source_topology": context.source_geometry_view.source_topology_id,
            "source_geometry": context.global_geometry_id,
            "target_topology": context.target_geometry_view.source_topology_id,
            "target_geometry": context.target_global_geometry_id,
            "step": state.step,
            "cursor": state.schedule_cursor,
            "version": state.state_version,
        }
    )


def _owner_local_keyed_declaration(
    context: _FiniteElementRepartitionContext, entry: CompositionEntry, /
) -> CompositionEntry:
    """Present one rank's FE shard entry under its group-wide keyed declaration.

    Owner-local FE shard entries carry rank-local structure and revision IDs
    (local ownership rows, local prepared spaces). Graph keyed transfers require
    one declaration on every owner, so the shard keeps its entry, role, owner,
    semantics and value, and takes a structure/revision derived from the global
    accepted placement. The composition transport binds the actual shard entry.
    """
    return CompositionEntry(
        entry.value,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=canonical_fingerprint(
            {
                "kind": "finite-element-owner-local-keyed-structure",
                "entry": entry.entry_id,
                "semantics": entry.semantics_id,
                "source_topology": context.source_geometry_view.source_topology_id,
                "source_geometry": context.global_geometry_id,
            }
        ),
        revision_id=_owner_local_keyed_revision(context, entry.entry_id),
        semantics_id=entry.semantics_id,
    )


def _finite_element_keyed_transport(
    transfer: OwnerLocalGraphKeyedStateTransfer
    | FiniteElementOwnerLocalFieldTransferEvidence,
    source: CompositionEntry,
    target: CompositionEntry,
    /,
    *,
    target_revision_id: str,
) -> CompositionTransport:
    """Bind the actual shard entries to a keyed transfer of their declaration."""
    if not isinstance(transfer, OwnerLocalGraphKeyedStateTransfer):
        return transfer.composition_transport(
            source, target, target_revision_id=target_revision_id
        )
    if (
        source.entry_id != transfer.source_entry_id
        or target.entry_id != source.entry_id
        or source.role != transfer.source_role
        or target.role != source.role
        or source.owner_id != transfer.source_owner_id
        or target.owner_id != source.owner_id
        or source.semantics_id != transfer.source_semantics_id
        or target.semantics_id != source.semantics_id
        or target.revision_id != target_revision_id
        or _owner_local_state_value_identity(source.value) != transfer.source_value_id
        or _owner_local_state_value_identity(target.value) != transfer.target_value_id
    ):
        raise ValueError("Owner-local FE keyed transport entries differ from their rows.")
    return CompositionTransport(
        "ownership-migration",
        (source.entry_id,),
        (target,),
        source_structure_ids=(source.structure_id,),
        route_id=canonical_fingerprint(
            {
                "kind": "finite-element-owner-local-keyed-transport",
                "transfer": transfer.transfer_id,
                "source_entry": source.record_id,
                "target_entry": target.record_id,
            }
        ),
        successful=transfer.successful,
        source_content_root=transfer.source_content_root,
        target_content_root=transfer.target_content_root,
    )


def _owner_local_gather_target_field(
    context: _FiniteElementRepartitionContext,
    source_entry: CompositionEntry,
    layout: FiniteElementAcceptedFieldLayout,
    target_plan: FiniteElementDofIdentityPlan,
    target_ownership: FiniteElementDofOwnership,
    target_values: Array,
    target_structure_id: str,
    axis_name: str,
    /,
) -> Array:
    entry_id = f"{source_entry.entry_id}:target-local:{layout.layout_id}"
    candidate_entry = CompositionEntry(
        target_values,
        entry_id=entry_id,
        role=source_entry.role,
        owner_id=source_entry.owner_id,
        structure_id=target_structure_id,
        revision_id=_owner_local_keyed_revision(context, entry_id),
        semantics_id=layout.layout_id,
    )
    # Owned target rows stay on their owners; the local plan's resident rows,
    # including ghosts owned elsewhere, are then fetched from those owners.
    owned_keys = np.asarray(target_ownership.dof_keys, dtype=np.int64)
    owned = context.migration.transfer_keyed_values(
        candidate_entry,
        owned_keys,
        target_ownership.owner_ranks,
        target_values,
        owned_keys,
        np.full((owned_keys.shape[0],), context.rank, dtype=np.int32),
        context.execution_group,
        target_structure_id=target_structure_id,
        key_identity_id=target_plan.identity_id,
        axis_name=axis_name,
        value_fiber_shape=tuple(int(size) for size in target_values.shape[1:]),
    )
    local_values = owned.gather_target_local_values(
        np.asarray(target_plan.dof_keys, dtype=np.int64),
        context.execution_group,
        axis_name=f"{axis_name}_local",
    )
    if local_values.shape[0] != target_plan.dof_keys.shape[0]:
        raise ValueError("Owner-local target FE field gather changed its row count.")
    return local_values


def _owner_local_cell_measure_map(view: OwnerLocalCellGeometry, /) -> dict[int, float]:
    measures: dict[int, float] = {}
    for block in view.blocks:
        identifiers = np.asarray(block.cell_ids, dtype=np.int64)
        values = np.asarray(block.cell_measures, dtype=np.float64)
        if (
            identifiers.ndim != 1
            or values.shape != identifiers.shape
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or any(int(identifier) in measures for identifier in identifiers)
        ):
            raise ValueError("Owner-local FE cell measure rows are malformed.")
        measures.update(
            (int(identifier), float(value))
            for identifier, value in zip(identifiers, values, strict=True)
        )
    return measures


def _owner_local_cell_content(
    view: OwnerLocalCellGeometry,
    keys: ArrayLike,
    values: ArrayLike,
    /,
    *,
    axis_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    key_array = np.asarray(keys)
    value_array = np.asarray(values)
    if (
        key_array.dtype != np.dtype(np.int64)
        or key_array.ndim != 2
        or key_array.shape[1] != 3
        or value_array.ndim == 0
        or value_array.shape[0] != key_array.shape[0]
        or np.any(key_array[:, 0] != view.topological_dimension)
        or np.any(key_array[:, 2] != 0)
        or not np.issubdtype(value_array.dtype, np.inexact)
        or np.any(~np.isfinite(value_array))
    ):
        raise ValueError("Conservative P0 field rows do not match their cell keys.")
    measure_by_id = _owner_local_cell_measure_map(view)
    try:
        measures = np.asarray(
            [measure_by_id[int(identifier)] for identifier in key_array[:, 1]],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError(
            "Conservative P0 rows lack current physical cell measures."
        ) from error
    trailing = (1,) * (value_array.ndim - 1)
    content = np.asarray(
        value_array.astype(np.result_type(value_array.dtype, np.float64), copy=False)
        * measures.reshape((-1,) + trailing),
        dtype=np.result_type(value_array.dtype, np.float64),
    )
    if content.shape[0]:
        integral = np.asarray(
            compensated_sum_chunks(
                (content,),
                output_ndim=value_array.ndim - 1,
            )
        )
        absolute = np.asarray(
            compensated_sum_chunks(
                (np.abs(content),),
                output_ndim=value_array.ndim - 1,
            ),
            dtype=np.float64,
        )
    else:
        integral = np.zeros(value_array.shape[1:], dtype=content.dtype)
        absolute = np.zeros(value_array.shape[1:], dtype=np.float64)
    if np.any(~np.isfinite(integral)) or np.any(~np.isfinite(absolute)):
        raise ValueError(f"{axis_name} conservative P0 integral is non-finite.")
    return integral, absolute


def _owner_local_conservative_integrals(
    context: _FiniteElementRepartitionContext,
    source_view: OwnerLocalCellGeometry,
    source_keys: ArrayLike,
    source_values: ArrayLike,
    target_view: OwnerLocalCellGeometry,
    target_keys: ArrayLike,
    target_values: ArrayLike,
    /,
    *,
    axis_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    local_error = None
    try:
        source_local, source_abs_local = _owner_local_cell_content(
            source_view, source_keys, source_values, axis_name=f"{axis_name}_source"
        )
        target_local, target_abs_local = _owner_local_cell_content(
            target_view, target_keys, target_values, axis_name=f"{axis_name}_target"
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        context.execution_group,
        f"{axis_name}_content_preflight",
        "FE conservative content evaluation",
    )

    def gather(value: np.ndarray, name: str, /) -> np.ndarray:
        # Scalar fibers carry a unit leading axis: owner metadata is never 0-d.
        array = np.asarray(value)
        rows = np.asarray(
            _owner_group_allgather(
                array[None, ...], context.execution_group, f"{axis_name}_{name}"
            )
        )
        return rows.reshape((rows.shape[0], *array.shape))

    source_rows = gather(source_local, "source_integral")
    target_rows = gather(target_local, "target_integral")
    source_abs_rows = gather(source_abs_local, "source_absolute")
    target_abs_rows = gather(target_abs_local, "target_absolute")
    source_integral = np.asarray(
        compensated_sum_chunks((source_rows,), output_ndim=source_local.ndim)
    )
    target_integral = np.asarray(
        compensated_sum_chunks((target_rows,), output_ndim=target_local.ndim)
    )
    total_absolute = np.asarray(
        compensated_sum_chunks(
            (source_abs_rows + target_abs_rows,),
            output_ndim=source_abs_local.ndim,
        ),
        dtype=np.float64,
    )
    counts = _owner_group_allgather(
        np.asarray(
            (
                np.asarray(source_keys).shape[0],
                np.asarray(target_keys).shape[0],
            ),
            dtype=np.int64,
        ),
        context.execution_group,
        f"{axis_name}_row_counts",
    )
    terms = int(np.sum(counts, dtype=np.int64)) * 2 + 2 * context.parts
    epsilon = np.finfo(np.float64).eps
    scaled_roundoff = terms * epsilon
    if scaled_roundoff >= 1.0:
        raise ValueError("Conservative FE content reduction exceeds its roundoff model.")
    bound = float(
        (scaled_roundoff / (1.0 - scaled_roundoff)) * np.max(total_absolute, initial=0.0)
    )
    defect = target_integral - source_integral
    if np.any(np.abs(defect) > bound):
        raise ValueError("Owner-local conservative FE content is not preserved.")
    return source_integral, target_integral, defect, bound


def _finite_element_owner_local_action_id(
    action: FiniteElementFieldTransferAction | None,
    default_id: str,
    /,
) -> str:
    if action is None:
        return default_id
    action_type = type(action)
    module = getattr(action, "__module__", action_type.__module__)
    name = getattr(action, "__qualname__", action_type.__qualname__)
    return canonical_fingerprint(
        {
            "kind": "finite-element-owner-local-field-action",
            "module": str(module),
            "name": str(name),
            "default": default_id,
        }
    )


def _owner_local_cell_average_rows(
    metadata: "_FiniteElementOwnerLocalAdaptiveMetadata",
    dimension: int,
    target_ids: np.ndarray,
    /,
) -> tuple[tuple[tuple[int, Fraction], ...], ...]:
    """Build exact physical-overlap rows from accepted nested cell lineage."""
    if metadata.lineage is None:
        if (
            metadata.source_cell_ids.size
            or metadata.target_cell_ids.size
            or target_ids.size
        ):
            raise ValueError("Nonempty conservative FE transfer requires cell lineage.")
        return ()
    source_measures = dict(
        zip(
            map(int, metadata.source_cell_ids), metadata.source_cell_measures, strict=True
        )
    )
    target_measures = dict(
        zip(
            map(int, metadata.target_cell_ids), metadata.target_cell_measures, strict=True
        )
    )
    if (
        len(source_measures) != metadata.source_cell_ids.size
        or len(target_measures) != metadata.target_cell_ids.size
        or any(
            not isinstance(value, Fraction) or value <= 0
            for value in (*source_measures.values(), *target_measures.values())
        )
    ):
        raise ValueError("Conservative FE lineage requires positive exact cell measures.")
    lineage = metadata.lineage.entity_lineage(dimension)
    source_coverage = {identifier: Fraction(0) for identifier in source_measures}
    target_coverage = {identifier: Fraction(0) for identifier in target_measures}
    rows: dict[int, list[tuple[int, Fraction]]] = {
        identifier: [] for identifier in target_measures
    }
    seen: set[tuple[int, int]] = set()
    for source, target, kind in zip(
        np.asarray(lineage.source_global_ids, dtype=np.int64),
        np.asarray(lineage.target_global_ids, dtype=np.int64),
        np.asarray(lineage.relation_kinds, dtype=np.int32),
        strict=True,
    ):
        source_id, target_id = int(source), int(target)
        if (
            source_id not in source_measures
            or target_id not in target_measures
            or (source_id, target_id) in seen
        ):
            raise ValueError(
                "Conservative FE cell lineage escapes or duplicates its domain."
            )
        seen.add((source_id, target_id))
        source_measure = source_measures[source_id]
        target_measure = target_measures[target_id]
        if kind == int(EntityLineageKind.PRESERVED):
            if source_measure != target_measure:
                raise ValueError("Preserved FE cell changed its exact physical measure.")
            overlap = source_measure
        elif kind == int(EntityLineageKind.REFINED_FROM):
            overlap = target_measure
        elif kind == int(EntityLineageKind.COARSENED_INTO):
            overlap = source_measure
        else:
            raise ValueError("FE cell lineage lacks a nested physical-overlap contract.")
        source_coverage[source_id] += overlap
        target_coverage[target_id] += overlap
        rows[target_id].append((source_id, overlap / target_measure))
    if source_coverage != source_measures or target_coverage != target_measures:
        raise ValueError(
            "Conservative FE lineage does not exactly cover source and target."
        )
    return tuple(tuple(sorted(rows[int(identifier)])) for identifier in target_ids)


def _transfer_finite_element_owner_local_adaptive_field(
    transaction: "FiniteElementTopologyTransaction",
    context: _FiniteElementRepartitionContext,
    layout: FiniteElementAcceptedFieldLayout,
    source_plan: FiniteElementDofIdentityPlan,
    target_plan: FiniteElementDofIdentityPlan,
    source_ownership: FiniteElementDofOwnership,
    target_ownership: FiniteElementDofOwnership,
    source_entry: CompositionEntry,
    source_values: Array,
    metadata: "_FiniteElementOwnerLocalAdaptiveMetadata",
    /,
    *,
    target_structure_id: str,
    action: FiniteElementFieldTransferAction | None,
    field_index: int,
    axis_name: str,
) -> tuple[FiniteElementOwnerLocalFieldTransferEvidence, Array]:
    source_mesh = context.source_mesh
    target_mesh = context.target_mesh
    source_map = context.source_discretization.dof_maps[field_index]
    target_map = context.target_discretization.dof_maps[field_index]
    vertex_transfer = _vertex_p1_field_layout(
        layout, source_mesh
    ) and _vertex_p1_field_layout(layout, target_mesh)
    conservative_cell_average = (
        layout.transfer_semantics
        is FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE
    )
    local_error = None
    source_keys = np.asarray(source_ownership.dof_keys, dtype=np.int64)
    target_keys = np.asarray(target_ownership.dof_keys, dtype=np.int64)
    target_rows: tuple[tuple[tuple[int, Fraction], ...], ...] = ()
    if conservative_cell_average:
        dimension = context.target_geometry_view.topological_dimension
        if (
            source_map.association != "cell"
            or target_map.association != "cell"
            or (
                source_keys.size
                and (
                    np.any(source_keys[:, 0] != dimension)
                    or np.any(source_keys[:, 2] != 0)
                )
            )
            or (
                target_keys.size
                and (
                    np.any(target_keys[:, 0] != dimension)
                    or np.any(target_keys[:, 2] != 0)
                )
            )
        ):
            local_error = "Conservative FE cell-average ancestry or keys are malformed."
        else:
            try:
                target_rows = _owner_local_cell_average_rows(
                    metadata, dimension, target_keys[:, 1]
                )
            except (ValueError, KeyError) as error:
                local_error = str(error)
    elif vertex_transfer:
        stencil = metadata.vertex_stencil
        if stencil is None:
            if target_keys.shape[0]:
                local_error = "Nonempty P1 FE target lacks its interpolation stencil."
        else:
            stencil_target_ids = np.asarray(stencil.target_global_ids, dtype=np.int64)
            if (
                not stencil.preserves_constants
                or np.unique(stencil_target_ids).size != stencil_target_ids.size
                or (
                    target_keys.size
                    and (
                        np.any(target_keys[:, 0] != 0)
                        or np.any(target_keys[:, 2] != 0)
                        or not set(target_keys[:, 1].tolist()).issubset(
                            set(stencil_target_ids.tolist())
                        )
                    )
                )
            ):
                local_error = "P1 stencil omits an owned target FE vertex."
    elif not vertex_transfer and action is None:
        local_error = "Declared owner-local FE field transfer action is required."
    elif (
        layout.transfer_semantics
        is FiniteElementFieldTransferSemantics.MATERIAL_FOLLOWING_CONTENT
        and metadata.lineage is None
    ):
        local_error = "Material-following FE fields require accepted cell lineage."
    try:
        _owner_group_fail_if_any(
            local_error,
            context.execution_group,
            f"{axis_name}_preflight",
            "owner-local FE adaptive field transfer",
        )
    except Exception as error:
        raise ValueError(f"{type(error).__name__}: {error}") from error

    action_context = _FiniteElementOwnerLocalFieldActionContext(
        context,
        layout,
        source_plan,
        target_plan,
        source_ownership,
        target_ownership,
        source_entry,
        source_values,
        metadata.adaptive_result,
        metadata.vertex_stencil,
        field_index,
    )
    conservative = False
    source_integral: np.ndarray | None = None
    target_integral: np.ndarray | None = None
    conservation_defect: np.ndarray | None = None
    conservation_bound: float | None = None
    action_id = _finite_element_owner_local_action_id(
        action,
        f"field-layout:{layout.layout_id}",
    )

    if conservative_cell_average:
        unique_origins = np.asarray(
            sorted({source for row in target_rows for source, _ in row}), dtype=np.int64
        )
        request_keys = np.zeros((unique_origins.size, 3), dtype=np.int64)
        request_keys[:, 0] = dimension
        request_keys[:, 1] = unique_origins
        parent_values = action_context.request_source_rows(
            request_keys,
            axis_name=f"{axis_name}_source_parents",
        )
        origin_row = {
            int(identifier): row for row, identifier in enumerate(unique_origins.tolist())
        }
        target_values = (
            jnp.stack(
                tuple(
                    sum(
                        (
                            parent_values[origin_row[source]] * float(weight)
                            for source, weight in row
                        ),
                        jnp.zeros(source_values.shape[1:], dtype=source_values.dtype),
                    )
                    for row in target_rows
                ),
                axis=0,
            )
            if target_rows
            else jnp.empty(
                (0,) + tuple(int(size) for size in source_values.shape[1:]),
                dtype=source_values.dtype,
            )
        )
        target_local_values = _owner_local_gather_target_field(
            context,
            source_entry,
            layout,
            target_plan,
            target_ownership,
            target_values,
            target_structure_id,
            f"{axis_name}_target_local",
        )
        source_integral, target_integral, conservation_defect, conservation_bound = (
            _owner_local_conservative_integrals(
                context,
                context.source_geometry_view,
                source_keys,
                source_values,
                context.target_geometry_view,
                target_keys,
                target_values,
                axis_name=f"{axis_name}_content",
            )
        )
        conservative = True
        action_id = f"cell-lineage:{context.adaptive_result_id}"
    elif vertex_transfer:
        stencil = metadata.vertex_stencil
        if stencil is None:
            source_vertex_ids = np.empty((0,), dtype=np.int64)
            request_keys = np.empty((0, 3), dtype=np.int64)
        else:
            if not stencil.preserves_constants:
                raise ValueError(
                    "Owner-local P1 transfer stencil does not preserve constants."
                )
            stencil_sources = np.asarray(stencil.source_global_ids, dtype=np.int64)
            stencil_valid = np.asarray(stencil.valid, dtype=np.bool_)
            source_vertex_ids = np.unique(stencil_sources[stencil_valid])
            request_keys = np.zeros((source_vertex_ids.size, 3), dtype=np.int64)
            request_keys[:, 1] = source_vertex_ids
        source_vertex_values = action_context.request_source_rows(
            request_keys,
            axis_name=f"{axis_name}_vertex_sources",
        )
        if stencil is None:
            target_values = jnp.empty(
                (target_keys.shape[0],)
                + tuple(int(size) for size in source_values.shape[1:]),
                dtype=source_values.dtype,
            )
        else:
            transferred_vertices = stencil.apply(
                source_vertex_ids,
                source_vertex_values,
            )
            target_vertex_ids = np.asarray(stencil.target_global_ids, dtype=np.int64)
            target_row_by_id = {
                int(identifier): row
                for row, identifier in enumerate(target_vertex_ids.tolist())
            }
            try:
                target_owned_rows = tuple(
                    transferred_vertices[target_row_by_id[int(identifier)]]
                    for identifier in target_keys[:, 1].tolist()
                )
            except KeyError as error:
                raise ValueError("P1 stencil omits an owned target FE vertex.") from error
            target_values = (
                jnp.stack(target_owned_rows, axis=0)
                if target_owned_rows
                else jnp.empty(
                    (0,) + tuple(int(size) for size in source_values.shape[1:]),
                    dtype=source_values.dtype,
                )
            )
        target_local_values = _owner_local_gather_target_field(
            context,
            source_entry,
            layout,
            target_plan,
            target_ownership,
            target_values,
            target_structure_id,
            f"{axis_name}_target_local",
        )
        action_id = (
            "empty-owner-p1-interpolation"
            if stencil is None
            else f"vertex-stencil:{stencil.stencil_id}"
        )
    else:
        source_local_values = action_context.request_source_rows(
            source_plan.dof_keys,
            axis_name=f"{axis_name}_source_local",
        )
        if action is None:
            raise ValueError(
                "Owner-local FE transfer action disappeared after admission."
            )
        target_values = jnp.asarray(
            action(
                source_local_values,
                layout,
                metadata.adaptive_result,
                action_context,
            )
        )
        if (
            target_values.ndim == 0
            or target_values.shape[0] != target_ownership.dof_keys.shape[0]
            or tuple(target_values.shape[1:])
            != tuple(int(size) for size in source_values.shape[1:])
            or not jnp.issubdtype(target_values.dtype, jnp.inexact)
            or not bool(np.asarray(jnp.all(jnp.isfinite(target_values))))
        ):
            raise ValueError("Owner-local FE action changed the declared field shape.")
        target_local_values = _owner_local_gather_target_field(
            context,
            source_entry,
            layout,
            target_plan,
            target_ownership,
            target_values,
            target_structure_id,
            f"{axis_name}_target_local",
        )
        action_id = _finite_element_owner_local_action_id(
            action, f"field-layout:{layout.layout_id}"
        )

    target_values = jnp.asarray(target_values)
    expected_shape = (
        target_ownership.dof_keys.shape[0],
        *tuple(int(size) for size in source_values.shape[1:]),
    )
    local_expected_shape = (
        target_plan.dof_keys.shape[0],
        *tuple(int(size) for size in source_values.shape[1:]),
    )
    if (
        target_values.shape != expected_shape
        or target_local_values.shape != local_expected_shape
        or not bool(np.asarray(jnp.all(jnp.isfinite(target_values))))
        or not bool(np.asarray(jnp.all(jnp.isfinite(target_local_values))))
    ):
        raise ValueError(
            "Owner-local FE topology transfer produced malformed target rows."
        )
    source_root = canonical_fingerprint(
        {
            "kind": "finite-element-owner-local-source-field-content",
            "entry": source_entry.record_id,
            "layout": layout.layout_id,
            "plan": source_plan.plan_id,
            "values": array_tree_fingerprint(np.asarray(source_values)),
        }
    )
    target_root = canonical_fingerprint(
        {
            "kind": "finite-element-owner-local-target-field-content",
            "entry": source_entry.entry_id,
            "layout": layout.layout_id,
            "plan": target_plan.plan_id,
            "values": array_tree_fingerprint(np.asarray(target_values)),
        }
    )
    evidence = FiniteElementOwnerLocalFieldTransferEvidence(
        layout.layout_id,
        layout.transfer_semantics,
        context.adaptive_result_id or context.migration.evidence_id,
        None if metadata.vertex_stencil is None else metadata.vertex_stencil.stencil_id,
        None if metadata.lineage is None else metadata.lineage.lineage_id,
        source_plan.plan_id,
        target_plan.plan_id,
        source_entry.structure_id,
        target_structure_id,
        source_ownership.ownership_id,
        target_ownership.ownership_id,
        action_id,
        source_root,
        target_root,
        target_ownership.dof_keys,
        target_values,
        conservative=conservative,
        source_integral=source_integral,
        target_integral=target_integral,
        conservation_defect=conservation_defect,
        conservation_bound=conservation_bound,
    )
    return evidence, target_local_values


@dataclass(frozen=True, slots=True)
class _FiniteElementOwnerLocalAdaptiveMetadata:
    result_id: str | None
    candidate_id: str | None
    lineage: MeshLineage | None
    vertex_stencil: VertexInterpolationStencil | None
    source_cell_ids: np.ndarray
    target_cell_ids: np.ndarray
    source_cell_measures: tuple[Fraction, ...]
    target_cell_measures: tuple[Fraction, ...]
    adaptive_result: FiniteElementFieldTransferContext | None


def _finite_element_owner_local_adaptive_metadata(
    adaptive_result: object | None, /
) -> _FiniteElementOwnerLocalAdaptiveMetadata:
    from ..meshing._device_adaptation import (
        OwnerLocalAdaptivePartTransition,
        OwnerLocalPeriodicAdaptiveResult,
    )

    empty = np.empty((0,), dtype=np.int64)
    if adaptive_result is None:
        return _FiniteElementOwnerLocalAdaptiveMetadata(
            None, None, None, None, empty, empty, (), (), None
        )
    if type(adaptive_result) is OwnerLocalPeriodicAdaptiveResult:
        if not adaptive_result.status.converged or adaptive_result.refusal is not None:
            raise ValueError("Owner-local periodic adaptation was not fully accepted.")
        return _FiniteElementOwnerLocalAdaptiveMetadata(
            adaptive_result.result_id,
            adaptive_result.candidate_id,
            adaptive_result.lineage,
            adaptive_result.vertex_stencil,
            np.asarray(adaptive_result.source_cell_ids, dtype=np.int64),
            np.asarray(adaptive_result.target_cell_ids, dtype=np.int64),
            tuple(adaptive_result.source_cell_measures),
            tuple(adaptive_result.target_cell_measures),
            adaptive_result,
        )
    if type(adaptive_result) is OwnerLocalAdaptivePartTransition:
        if not adaptive_result.transition_id:
            raise ValueError("Owner-local adaptive transition has no identity.")
        return _FiniteElementOwnerLocalAdaptiveMetadata(
            adaptive_result.transition_id,
            None,
            adaptive_result.lineage,
            adaptive_result.vertex_stencil,
            np.asarray(adaptive_result.source_cell_ids, dtype=np.int64),
            np.asarray(adaptive_result.target_cell_ids, dtype=np.int64),
            tuple(adaptive_result.source_cell_measures),
            tuple(adaptive_result.target_cell_measures),
            adaptive_result,
        )
    raise TypeError(
        "adaptive_result must be OwnerLocalPeriodicAdaptiveResult, "
        "OwnerLocalAdaptivePartTransition, or None on an empty owner."
    )


def _finite_element_repartition_result(
    context: _FiniteElementRepartitionContext,
    state: FiniteElementAcceptedState,
    manifest: FiniteElementRestartManifest,
    committed: bool,
    diagnostics: str,
    receipt: CompositionRebindReceipt | None = None,
    /,
) -> FiniteElementRepartitionResult:
    return FiniteElementRepartitionResult(
        state,
        manifest,
        jnp.asarray(committed, dtype=jnp.bool_),
        diagnostics,
        receipt,
        tuple(context.field_transfers),
        tuple(context.payload_transfers),
        context.target_plans,
    )


def _refused_finite_element_repartition(
    context: _FiniteElementRepartitionContext, diagnostics: str, /
) -> FiniteElementRepartitionResult:
    return _finite_element_repartition_result(
        context, context.source_state, context.source_manifest, False, diagnostics
    )


def _finite_element_repartition_collective_error(
    context: _FiniteElementRepartitionContext,
    local_error: str | None,
    phase: str,
    /,
) -> str | None:
    try:
        _owner_group_fail_if_any(
            local_error,
            context.execution_group,
            f"{context.axis_name}_{phase}",
            "FE repartition admission",
        )
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _validate_finite_element_repartition_collectively(
    context: _FiniteElementRepartitionContext,
    phase: str,
    validate: Callable[[], None],
    /,
) -> str | None:
    local_error = None
    try:
        validate()
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    return _finite_element_repartition_collective_error(context, local_error, phase)


def _finite_element_restart_array_identity(value: ArrayLike, /) -> str:
    import jax

    array = jnp.asarray(value)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        return canonical_fingerprint(
            {
                "kind": "finite-element-typed-key",
                "implementation": str(jax.random.key_impl(array)),
                "data": array_tree_fingerprint(jax.random.key_data(array)),
            }
        )
    return canonical_fingerprint(array_tree_fingerprint(np.asarray(array)))


def _finite_element_materials_match(
    candidate: MaterialTransaction | None,
    expected: MaterialTransaction | None,
    /,
) -> bool:
    if candidate is None or expected is None:
        return candidate is expected
    return (
        candidate.layout_id == expected.layout_id
        and candidate.transaction_id == expected.transaction_id
        and len(candidate.states) == len(expected.states)
        and all(
            left.site_id.key == right.site_id.key
            and left.model_id == right.model_id
            and left.state_version == right.state_version
            and _finite_element_restart_array_identity(left.committed)
            == _finite_element_restart_array_identity(right.committed)
            and _finite_element_restart_array_identity(left.trial)
            == _finite_element_restart_array_identity(right.trial)
            for left, right in zip(candidate.states, expected.states, strict=True)
        )
    )


def _finite_element_accepted_states_match(
    candidate: FiniteElementAcceptedState,
    expected: FiniteElementAcceptedState,
    /,
) -> bool:
    return (
        candidate.accepted_id == expected.accepted_id
        and candidate.topology_id == expected.topology_id
        and candidate.prepared_id == expected.prepared_id
        and candidate.compilation_id == expected.compilation_id
        and candidate.step == expected.step
        and candidate.schedule_cursor == expected.schedule_cursor
        and candidate.state_version == expected.state_version
        and _finite_element_restart_array_identity(candidate.time)
        == _finite_element_restart_array_identity(expected.time)
        and tuple(layout.layout_id for layout in candidate.field_layouts)
        == tuple(layout.layout_id for layout in expected.field_layouts)
        and tuple(
            _finite_element_restart_array_identity(value) for value in candidate.fields
        )
        == tuple(
            _finite_element_restart_array_identity(value) for value in expected.fields
        )
        and (
            None
            if candidate.field_ownership is None
            else tuple(item.ownership_id for item in candidate.field_ownership)
        )
        == (
            None
            if expected.field_ownership is None
            else tuple(item.ownership_id for item in expected.field_ownership)
        )
        and _finite_element_materials_match(candidate.materials, expected.materials)
    )


def _finite_element_restart_layout_identity(
    layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
    /,
) -> tuple[tuple[str, str, str, str | None, str | None], ...] | None:
    if layouts is None:
        return None
    return tuple(
        (
            item.layout_id,
            item.role.value,
            item.association.value,
            None if item.ownership is None else item.ownership.ownership_id,
            item.prng_impl,
        )
        for item in layouts
    )


def _finite_element_restart_layout_schema(
    layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
    /,
) -> tuple[tuple[str, str], ...] | None:
    if layouts is None:
        return None
    return tuple((layout.name, layout.layout_id) for layout in layouts)


def _finite_element_restart_manifests_match(
    candidate: FiniteElementRestartManifest,
    expected: FiniteElementRestartManifest,
    /,
) -> bool:
    return (
        candidate.manifest_id == expected.manifest_id
        and _finite_element_accepted_states_match(candidate.state, expected.state)
        and tuple(
            (name, _finite_element_restart_array_identity(value))
            for name, value in candidate.auxiliary_state
        )
        == tuple(
            (name, _finite_element_restart_array_identity(value))
            for name, value in expected.auxiliary_state
        )
        and tuple(
            (name, _finite_element_restart_array_identity(value))
            for name, value in candidate.integrator_state
        )
        == tuple(
            (name, _finite_element_restart_array_identity(value))
            for name, value in expected.integrator_state
        )
        and _finite_element_restart_layout_identity(candidate.auxiliary_layouts)
        == _finite_element_restart_layout_identity(expected.auxiliary_layouts)
        and _finite_element_restart_layout_identity(candidate.integrator_layouts)
        == _finite_element_restart_layout_identity(expected.integrator_layouts)
        and _finite_element_restart_layout_identity(candidate.material_layouts)
        == _finite_element_restart_layout_identity(expected.material_layouts)
    )


def _finite_element_prepared_geometry_matches(
    discretization: FiniteElementDiscretization,
    coordinate_spec: CellGeometrySpec,
    /,
) -> bool:
    if not isinstance(coordinate_spec, CellGeometrySpec):
        return False
    elements, routes, coordinates = coordinate_spec.resolve(discretization.mesh)
    return (
        tuple(item.element_id for item in elements)
        == tuple(item.element_id for item in discretization.coordinate_elements)
        and tuple(array_tree_fingerprint(value) for value in routes)
        == tuple(
            array_tree_fingerprint(value) for value in discretization.coordinate_dofs
        )
        and array_tree_fingerprint(coordinates)
        == array_tree_fingerprint(discretization.default_runtime.coordinates)
    )


def _finite_element_identity_plan_matches(
    context: _FiniteElementRepartitionContext,
    discretization: FiniteElementDiscretization,
    plan: FiniteElementDofIdentityPlan,
    field_index: int,
    layout: FiniteElementAcceptedFieldLayout,
    coordinate_spec: CellGeometrySpec | None,
    /,
) -> bool:
    if layout.periodic_quotient and coordinate_spec is None:
        return False
    mesh = discretization.mesh
    local_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    local_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.asarray(plan.cell_ids, dtype=np.int64)
    cell_owner_ranks = np.asarray(plan.cell_owner_ranks, dtype=np.int32)
    vertex_ids = np.asarray(plan.vertex_ids, dtype=np.int64)
    vertex_owner_ranks = np.asarray(plan.vertex_owner_ranks, dtype=np.int32)
    if not np.array_equal(cell_ids, local_cell_ids) or not np.array_equal(
        vertex_ids, local_vertex_ids
    ):
        return False
    expected = finite_element_dof_identity_plan(
        discretization,
        field_index,
        field_layout_id=layout.layout_id,
        distribution_id=plan.distribution_id,
        topology_id=plan.topology_id,
        geometry_id=plan.geometry_id,
        local_rank=context.rank,
        part_count=context.parts,
        cell_ids=cell_ids,
        cell_owner_ranks=cell_owner_ranks,
        vertex_ids=vertex_ids,
        vertex_owner_ranks=vertex_owner_ranks,
        periodic_quotient=layout.periodic_quotient,
        coordinate_spec=coordinate_spec,
    )
    return (
        plan.dof_map_id == expected.dof_map_id
        and plan.identity_id == expected.identity_id
        and plan.orientation_action_id == expected.orientation_action_id
        and np.array_equal(np.asarray(plan.dof_keys), np.asarray(expected.dof_keys))
        and np.array_equal(np.asarray(plan.owner_ranks), np.asarray(expected.owner_ranks))
        and np.array_equal(
            np.asarray(plan.cell_owner_ranks),
            np.asarray(expected.cell_owner_ranks),
        )
        and np.array_equal(
            np.asarray(plan.vertex_owner_ranks),
            np.asarray(expected.vertex_owner_ranks),
        )
    )


def _validate_finite_element_repartition_authority(
    context: _FiniteElementRepartitionContext, /
) -> None:
    state = context.source_state
    source = context.source_discretization
    if (
        len(context.source_plans) != len(state.fields)
        or len(context.target_plans) != len(state.fields)
        or len(state.field_layouts) != len(state.fields)
        or state.field_ownership is None
        or len(state.field_ownership) != len(state.fields)
        or state.prepared_id != source.prepared_id
        or state.compilation_id != context.source_compiled.compilation_id
        or context.source_compiled.form.form_id != context.target_compiled.form.form_id
        or context.source_compiled.form.field_names
        != context.target_compiled.form.field_names
        or tuple(layout.name for layout in state.field_layouts)
        != context.source_compiled.form.field_names
        or context.migration.group_id != context.execution_group.spec.group_id
        or context.migration.local_rank != context.rank
        or context.migration.part_count != context.parts
        or not context.axis_name.strip()
        or any(not isinstance(item, CompositionEntry) for item in context.reprepared)
    ):
        raise ValueError("Source/target FE repartition authority is inconsistent.")

    source_mesh = context.source_mesh
    target_mesh = context.target_mesh
    source_blocks = tuple(
        (block.name, block.cell_kind, block.arity, block.topological_dimension)
        for block in source_mesh.blocks
    )
    target_blocks = tuple(
        (block.name, block.cell_kind, block.arity, block.topological_dimension)
        for block in target_mesh.blocks
    )
    if (
        source_mesh.topological_dimension != target_mesh.topological_dimension
        or source_mesh.ambient_dimension != target_mesh.ambient_dimension
        or source_blocks != target_blocks
    ):
        raise ValueError(
            "Source and target FE carriers need one explicit shared cell-family schema."
        )
    source_view = context.source_geometry_view
    target_view = context.target_geometry_view
    if not isinstance(source_view, OwnerLocalCellGeometry) or not isinstance(
        target_view, OwnerLocalCellGeometry
    ):
        raise ValueError(
            "Source/target compact FE geometry views differ from the current graph "
            "authority: view-type."
        )
    # source_map_id/source_context_id bind each rank's resident source bank (an
    # exact local power map, or a halo-expanded periodic quotient); both
    # placements must share the authored source identity, not those rank banks.
    stale = tuple(
        name
        for name, failed in (
            ("source-owner", source_view.owner_rank != context.rank),
            ("target-owner", target_view.owner_rank != context.rank),
            (
                "source-directory",
                source_view.distribution_id != context.migration.source_directory_id,
            ),
            (
                "target-directory",
                target_view.distribution_id != context.migration.target_directory_id,
            ),
            ("source-topology", source_view.source_topology_id != state.topology_id),
            ("target-topology", target_view.source_topology_id != state.topology_id),
            (
                "geometry",
                source_view.source_geometry_id != target_view.source_geometry_id,
            ),
            ("source", source_view.source_id != target_view.source_id),
            ("revision", source_view.source_revision != target_view.source_revision),
        )
        if failed
    )
    if stale:
        raise ValueError(
            "Source/target compact FE geometry views differ from the current graph "
            f"authority: {', '.join(stale)}."
        )
    source_view.require_owner_directory_current(
        context.execution_group,
        axis_name=f"{context.axis_name}_source_geometry_directory",
    )
    target_view.require_owner_directory_current(
        context.execution_group,
        axis_name=f"{context.axis_name}_target_geometry_directory",
    )

    def coordinate_spec(
        view: OwnerLocalCellGeometry,
        discretization: FiniteElementDiscretization,
        label: str,
        /,
    ) -> CellGeometrySpec | None:
        mesh, geometry = view.reconstruct_local_geometry()
        if mesh is None:
            if (
                sum(block.cell_count for block in discretization.mesh.blocks) != 0
                or view.cell_ids.size != 0
                or view.vertex_ids.size != 0
            ):
                raise ValueError(
                    f"Empty {label} FE view differs from its compiled carrier."
                )
            return None
        if (
            mesh.mesh_id != discretization.mesh.mesh_id
            or not isinstance(geometry, CellGeometrySpec)
            or not _finite_element_prepared_geometry_matches(discretization, geometry)
        ):
            raise ValueError(
                f"{label} FE compiled geometry differs from its compact owner view."
            )
        return geometry

    context.source_coordinate_spec = coordinate_spec(
        source_view, context.source_discretization, "source"
    )
    context.target_coordinate_spec = coordinate_spec(
        target_view, context.target_discretization, "target"
    )
    periodic = any(layout.periodic_quotient for layout in state.field_layouts)
    if periodic and (
        context.source_coordinate_spec is None or context.target_coordinate_spec is None
    ):
        raise ValueError(
            "Periodic FE repartition requires exact source and target coordinate specs."
        )


def _finite_element_owner_local_mesh_rows_match_view(
    candidate: CellMesh | None,
    view: OwnerLocalCellGeometry,
    /,
) -> bool:
    actual, _ = view.reconstruct_local_geometry()
    if candidate is None:
        return view.owned_cell_ids.size == 0
    if (
        actual is None
        or candidate.topological_dimension != actual.topological_dimension
        or candidate.ambient_dimension != actual.ambient_dimension
    ):
        return False
    actual_vertices = np.asarray(actual.vertex_global_ids, dtype=np.int64)
    actual_coordinates = np.asarray(actual.coordinates, dtype=np.float64)
    actual_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(actual_vertices.tolist())
    }
    candidate_vertices = np.asarray(candidate.vertex_global_ids, dtype=np.int64)
    candidate_coordinates = np.asarray(candidate.coordinates, dtype=np.float64)
    for identifier, coordinates in zip(
        candidate_vertices, candidate_coordinates, strict=True
    ):
        row = actual_vertex_rows.get(int(identifier))
        if row is None or not np.array_equal(coordinates, actual_coordinates[row]):
            return False

    def cell_signatures(mesh: CellMesh, /) -> dict[int, tuple[str, str, tuple[int, ...]]]:
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        result: dict[int, tuple[str, str, tuple[int, ...]]] = {}
        for block in mesh.blocks:
            cell_ids = np.asarray(block.global_ids, dtype=np.int64)
            vertices = np.asarray(block.vertices, dtype=np.int64).reshape(
                (cell_ids.size, block.arity)
            )
            for identifier, row in zip(cell_ids, vertices, strict=True):
                key = int(identifier)
                if key in result:
                    raise ValueError("Owner-local mesh repeats a global cell identity.")
                result[key] = (
                    block.name,
                    block.cell_kind,
                    tuple(sorted(int(vertex_ids[int(index)]) for index in row)),
                )
        return result

    candidate_cells = cell_signatures(candidate)
    actual_cells = cell_signatures(actual)
    return all(
        actual_cells.get(identifier) == signature
        for identifier, signature in candidate_cells.items()
    )


def _finite_element_adapted_in_place_after_migration(
    context: _FiniteElementRepartitionContext, /
) -> bool:
    """Whether the FE source view is the migrated shard an adaptation refined in place.

    The target view's migration evidence either moves the FE source view onto
    the adapted target, or it is the earlier migration that produced the
    adaptive source shard, with the adaptation applied in place on that
    placement. The shard's part identity also binds the adaptation
    specification, so the FE source view is matched to it by its owner
    directory, topology and geometry, which the part identity is derived from.
    """
    from ..meshing._device_adaptation import OwnerLocalPeriodicAdaptiveResult

    source_view = context.source_geometry_view
    migration = context.migration
    adaptive = context.adaptive_result
    return bool(
        type(adaptive) is OwnerLocalPeriodicAdaptiveResult
        and isinstance(source_view, OwnerLocalCellGeometry)
        and isinstance(migration, OwnerLocalGraphMigrationEvidence)
        and migration.target_directory_id == source_view.distribution_id
        and migration.target_part_id == adaptive.source_part_id
        and adaptive.source_distribution_id == source_view.distribution_id
        and adaptive.source_topology_id == source_view.source_topology_id
        and adaptive.source_geometry_id == source_view.source_geometry_id
    )


def _validate_finite_element_owner_local_topology_authority(
    context: _FiniteElementRepartitionContext, /
) -> None:

    from ..discretization._cell_mesh import EmptyLocalCellMeshAuthority
    from ..meshing._assembly import MeshPartManifest
    from ..meshing._device_adaptation import (
        _require_owner_local_periodic_transfer,
        OwnerLocalAdaptivePartTransition,
        OwnerLocalPeriodicAdaptiveResult,
    )

    state = context.source_state
    source_view = context.source_geometry_view
    target_view = context.target_geometry_view
    migration = context.migration
    field_ownership = state.field_ownership
    if not field_ownership:
        raise ValueError("Owner-local FE topology transition lacks DOF ownership.")
    in_place_after_migration = _finite_element_adapted_in_place_after_migration(context)
    if (
        not context.topology_change
        or not isinstance(source_view, OwnerLocalCellGeometry)
        or not isinstance(target_view, OwnerLocalCellGeometry)
    ):
        raise ValueError(
            "Owner-local FE topology transition authority is stale: it needs compact "
            "source/target views."
        )
    stale = tuple(
        name
        for name, failed in (
            (
                "migration-evidence",
                isinstance(migration, OwnerLocalGraphMigrationEvidence)
                and migration is not target_view.graph_migration_evidence,
            ),
            ("group", migration.group_id != context.execution_group.spec.group_id),
            ("rank", migration.local_rank != context.rank),
            ("part-count", migration.part_count != context.parts),
            (
                "migration-directories",
                not in_place_after_migration
                and (
                    migration.source_directory_id != source_view.distribution_id
                    or migration.target_directory_id != target_view.distribution_id
                    or (
                        isinstance(migration, OwnerLocalGraphMigrationEvidence)
                        and migration.target_part_id != target_view.part_id
                    )
                ),
            ),
            (
                "view-owner",
                source_view.owner_rank != context.rank
                or target_view.owner_rank != context.rank,
            ),
            ("source-topology", source_view.source_topology_id != state.topology_id),
            (
                "ownership-directory",
                source_view.distribution_id != field_ownership[0].distribution_id,
            ),
            ("unchanged-topology", target_view.source_topology_id == state.topology_id),
            (
                "infeasible-migration",
                isinstance(migration, OwnerLocalGraphMigrationEvidence)
                and not migration.result.feasible,
            ),
            ("axis", not context.axis_name.strip()),
        )
        if failed
    )
    if stale:
        raise ValueError(
            f"Owner-local FE topology transition authority is stale: {', '.join(stale)}."
        )
    source_view.require_owner_directory_current(
        context.execution_group,
        axis_name=f"{context.axis_name}_source_directory",
    )
    target_view.require_owner_directory_current(
        context.execution_group,
        axis_name=f"{context.axis_name}_target_directory",
    )
    if not context.source_manifest.state.field_ownership:
        raise ValueError("Owner-local FE adaptation needs source DOF ownership.")

    def coordinate_spec(
        view: OwnerLocalCellGeometry,
        discretization: FiniteElementDiscretization,
        label: str,
        /,
    ) -> CellGeometrySpec | None:
        mesh, geometry = view.reconstruct_local_geometry()
        if mesh is None:
            authority = discretization.mesh.empty_local_authority
            if (
                sum(block.cell_count for block in discretization.mesh.blocks) != 0
                or view.cell_ids.size != 0
                or view.vertex_ids.size != 0
                or not isinstance(authority, EmptyLocalCellMeshAuthority)
                or authority.source_topology_id != view.source_topology_id
                or authority.source_geometry_id != view.source_geometry_id
                or authority.distribution_id != view.distribution_id
                or authority.group_id != context.execution_group.spec.group_id
                or authority.local_rank != context.rank
                or authority.part_count != context.parts
            ):
                raise ValueError(f"Empty {label} FE view lacks current local authority.")
            return None
        if (
            mesh.mesh_id != discretization.mesh.mesh_id
            or not isinstance(geometry, CellGeometrySpec)
            or not _finite_element_prepared_geometry_matches(discretization, geometry)
        ):
            raise ValueError(
                f"{label} FE compiled geometry differs from its current compact view."
            )
        return geometry

    context.source_coordinate_spec = coordinate_spec(
        source_view, context.source_discretization, "source"
    )
    context.target_coordinate_spec = coordinate_spec(
        target_view, context.target_discretization, "target"
    )
    geometry_entries = tuple(
        entry
        for entry in context.reprepared
        if entry.entry_id == context.geometry_entry_id
    )
    if len(geometry_entries) != 1:
        raise ValueError("FE topology adaptation needs one target geometry manifest.")
    target_geometry_entry = geometry_entries[0]
    target_manifest = target_geometry_entry.value
    if (
        target_geometry_entry.role != "discretization"
        or not isinstance(target_manifest, MeshPartManifest)
        or target_manifest.source_topology_id != target_view.source_topology_id
        or target_manifest.source_geometry_id != target_view.source_geometry_id
        or target_manifest.part_id != target_view.part_id
        or target_manifest.distribution_id != target_view.distribution_id
        or target_geometry_entry.structure_id != target_view.source_geometry_id
        or target_geometry_entry.revision_id != target_manifest.manifest_id
    ):
        raise ValueError(
            "Target FE geometry manifest differs from its compact authority."
        )
    target_source_entry = _finite_element_geometry_source_entry(
        target_view,
        (*context.reprepared, *context.source_composition.entries),
    )
    if not _finite_element_geometry_binds_source(
        target_geometry_entry, target_source_entry
    ):
        raise ValueError("Target FE geometry manifest lacks its current source binding.")
    context.target_global_geometry_id = target_manifest.source_geometry_id

    metadata = _finite_element_owner_local_adaptive_metadata(context.adaptive_result)
    context.adaptive_result_id = metadata.result_id or migration.evidence_id
    context.vertex_stencil = metadata.vertex_stencil
    if context.adaptive_result is None:
        if (
            source_view.cell_ids.size
            or source_view.vertex_ids.size
            or target_view.cell_ids.size
            or target_view.vertex_ids.size
        ):
            raise ValueError("A nonempty FE topology transition lacks adaptive evidence.")
    elif type(context.adaptive_result) is OwnerLocalPeriodicAdaptiveResult:
        adaptive = context.adaptive_result
        differing = tuple(
            name
            for name, differs in (
                (
                    "source-distribution",
                    adaptive.source_distribution_id != source_view.distribution_id,
                ),
                (
                    "target-distribution",
                    adaptive.target_distribution_id != target_view.distribution_id,
                ),
                (
                    "source-topology",
                    adaptive.source_topology_id != source_view.source_topology_id,
                ),
                (
                    "source-geometry",
                    adaptive.source_geometry_id != source_view.source_geometry_id,
                ),
                (
                    "target-topology",
                    adaptive.target_topology_id != target_view.source_topology_id,
                ),
                (
                    "target-geometry",
                    adaptive.target_geometry_id != target_view.source_geometry_id,
                ),
                (
                    "target-local-view",
                    adaptive.target_local_view_id != target_view.local_view_id,
                ),
                (
                    "target-owner-directory",
                    adaptive.target_owner_directory_id != target_view.distribution_id,
                ),
                (
                    "target-view-result",
                    target_view.adaptive_result is None
                    or target_view.adaptive_result.result_id != adaptive.result_id,
                ),
                # The adaptive result binds this owner's target cells; the
                # compact view adds read-only face-neighbor halo cells around
                # them, so its resident vertices cover the adaptive vertices.
                (
                    "target-cells",
                    not np.array_equal(
                        np.sort(metadata.target_cell_ids),
                        np.sort(np.asarray(target_view.owned_cell_ids, dtype=np.int64)),
                    ),
                ),
                (
                    "target-vertices",
                    not np.all(
                        np.isin(
                            np.asarray(adaptive.target_vertex_ids, dtype=np.int64),
                            np.asarray(target_view.vertex_ids, dtype=np.int64),
                        )
                    ),
                ),
                (
                    "graph-plan",
                    isinstance(migration, OwnerLocalGraphMigrationEvidence)
                    and adaptive.graph_partition_plan is not None
                    and adaptive.graph_partition_plan.plan_id != migration.result.plan_id,
                ),
                (
                    "target-face-coverage",
                    adaptive.target_periodic_face_coverage_id
                    != target_view.periodic_face_coverage_id,
                ),
                (
                    "target-periodic-topology",
                    adaptive.target_periodic_topology_id
                    != target_view.periodic_topology_id,
                ),
            )
            if differs
        )
        if differing:
            raise ValueError(
                "Periodic FE adaptive evidence differs from the target view: "
                f"{', '.join(differing)}."
            )
        _require_owner_local_periodic_transfer(adaptive, source_view, target_view)
    elif type(context.adaptive_result) is OwnerLocalAdaptivePartTransition:
        adaptive = context.adaptive_result
        source_owned_ids = np.sort(np.asarray(source_view.owned_cell_ids, dtype=np.int64))
        target_owned_ids = np.sort(np.asarray(target_view.owned_cell_ids, dtype=np.int64))
        if (
            adaptive.part_index != context.rank
            or adaptive.source_distribution_id != source_view.distribution_id
            or adaptive.source_part_id != source_view.part_id
            or adaptive.source_topology_id != source_view.source_topology_id
            or adaptive.source_geometry_id != source_view.source_geometry_id
            or not np.array_equal(np.sort(metadata.source_cell_ids), source_owned_ids)
            or not np.array_equal(np.sort(metadata.target_cell_ids), target_owned_ids)
            or not _finite_element_owner_local_mesh_rows_match_view(
                adaptive.source_mesh, source_view
            )
            or not _finite_element_owner_local_mesh_rows_match_view(
                adaptive.target_mesh, target_view
            )
            or (
                adaptive.vertex_stencil is not None
                and (
                    adaptive.source_mesh is None
                    or adaptive.target_mesh is None
                    or adaptive.vertex_stencil.source_entity_set_id
                    != adaptive.source_mesh.entity_set(0).entity_set_id
                    or adaptive.vertex_stencil.target_entity_set_id
                    != adaptive.target_mesh.entity_set(0).entity_set_id
                )
            )
        ):
            raise ValueError("Generic FE adaptive transition differs from its views.")
    else:
        raise TypeError("Owner-local FE adaptation has an unsupported result type.")

    if isinstance(migration, OwnerLocalTopologyTransferRoute):
        if not np.array_equal(
            np.sort(metadata.source_cell_ids),
            np.sort(np.asarray(source_view.owned_cell_ids, dtype=np.int64)),
        ) or not np.array_equal(
            np.sort(metadata.target_cell_ids),
            np.sort(np.asarray(target_view.owned_cell_ids, dtype=np.int64)),
        ):
            raise ValueError(
                "Topology-only transfer changed owner-local lineage placement."
            )
    else:
        target_graph_ids = np.asarray(migration.result.entity_ids, dtype=np.int64)
        target_graph_owners = np.asarray(migration.result.owners, dtype=np.int32)
        target_graph_rows = np.sort(
            target_graph_ids[target_graph_owners == context.rank], kind="stable"
        )
        migrated_view = source_view if in_place_after_migration else target_view
        target_owned_rows = np.sort(
            np.asarray(migrated_view.owned_cell_ids, dtype=np.int64)
        )
        migrated_rows = np.sort(np.asarray(migration.target_cell_ids, dtype=np.int64))
        if (
            not np.array_equal(migrated_rows, target_owned_rows)
            or np.setdiff1d(target_graph_rows, migrated_rows).size
        ):
            raise ValueError(
                "FE target graph partition rows differ from current ownership."
            )


def _validate_finite_element_repartition_identity_plans(
    context: _FiniteElementRepartitionContext, /
) -> None:
    migration = context.migration
    if not isinstance(migration, OwnerLocalGraphMigrationEvidence):
        raise TypeError(
            "Repartition identity validation requires graph migration evidence."
        )
    state = context.source_state
    if any(
        not isinstance(plan, FiniteElementDofIdentityPlan)
        for plan in context.source_plans
    ):
        raise TypeError("source_identity_plans must be finite-element identity plans.")
    if any(
        not isinstance(plan, FiniteElementDofIdentityPlan)
        for plan in context.target_plans
    ):
        raise TypeError("target_identity_plans must be finite-element identity plans.")
    if any(layout.storage_kind != "canonical_dofs" for layout in state.field_layouts):
        raise ValueError("Owner-local FE repartition requires canonical DOF fields.")
    geometry_entry = context.geometry_entry
    if geometry_entry is None:
        raise ValueError("FE identity plans lack their source geometry entry.")
    source_cell_owners: dict[int, int] = {}
    source_vertex_owners: dict[int, int] = {}
    target_cell_owners: dict[int, int] = {}
    target_vertex_owners: dict[int, int] = {}

    def owner_rows(identifiers: ArrayLike, owner_ranks: ArrayLike, /) -> dict[int, int]:
        return dict(
            zip(
                np.asarray(identifiers, dtype=np.int64).tolist(),
                np.asarray(owner_ranks, dtype=np.int32).tolist(),
                strict=True,
            )
        )

    from ..discretization._cell_mesh import EmptyLocalCellMeshAuthority

    for mesh, directory_id in (
        (context.source_mesh, context.migration.source_directory_id),
        (context.target_mesh, context.migration.target_directory_id),
    ):
        resident_cells = sum(block.cell_count for block in mesh.blocks)
        authority = mesh.empty_local_authority
        if resident_cells == 0:
            if (
                not isinstance(authority, EmptyLocalCellMeshAuthority)
                or authority.source_topology_id != state.topology_id
                or authority.source_geometry_id != context.global_geometry_id
                or authority.distribution_id != directory_id
                or authority.group_id != context.execution_group.spec.group_id
                or authority.local_rank != context.rank
                or authority.part_count != context.parts
                or authority.topological_dimension != mesh.topological_dimension
                or authority.ambient_dimension != mesh.ambient_dimension
                or len(mesh.blocks) != 1
                or mesh.blocks[0].name != authority.block_name
                or mesh.blocks[0].cell_kind != authority.cell_kind
            ):
                raise ValueError(
                    "Zero-resident FE carrier lacks current global and partition authority."
                )
        elif authority is not None:
            raise ValueError("Resident FE cells cannot carry an empty-local authority.")
    from ..meshing._assembly import MeshPartManifest

    source_view = context.source_geometry_view
    target_view = context.target_geometry_view
    source_manifest = geometry_entry.value
    target_geometry_entries = tuple(
        entry
        for entry in context.reprepared
        if entry.entry_id == context.geometry_entry_id
    )
    if (
        not isinstance(source_manifest, MeshPartManifest)
        or len(target_geometry_entries) != 1
    ):
        raise ValueError(
            "FE repartition requires current source and target compact geometry manifests."
        )
    target_geometry_entry = target_geometry_entries[0]
    target_manifest = target_geometry_entry.value
    if (
        target_geometry_entry.role != "discretization"
        or not isinstance(target_manifest, MeshPartManifest)
        or source_manifest.source_topology_id != state.topology_id
        or target_manifest.source_topology_id != state.topology_id
        or source_manifest.source_geometry_id != context.global_geometry_id
        or target_manifest.source_geometry_id != context.global_geometry_id
        or source_manifest.part_id != source_view.part_id
        or target_manifest.part_id != target_view.part_id
        or source_manifest.distribution_id != context.migration.source_directory_id
        or target_manifest.distribution_id != context.migration.target_directory_id
        or source_view.distribution_id != context.migration.source_directory_id
        or target_view.distribution_id != context.migration.target_directory_id
        or source_view.source_topology_id != state.topology_id
        or target_view.source_topology_id != state.topology_id
        or source_view.source_geometry_id != context.global_geometry_id
        or target_view.source_geometry_id != context.global_geometry_id
        or target_geometry_entry.structure_id != context.global_geometry_id
        or target_geometry_entry.revision_id != target_manifest.manifest_id
    ):
        raise ValueError(
            "Compact FE geometry manifests/views differ from current global or directory authority."
        )
    target_source_entry = _finite_element_geometry_source_entry(
        target_view,
        (*context.reprepared, *context.source_composition.entries),
    )
    if not _finite_element_geometry_binds_source(
        target_geometry_entry, target_source_entry
    ):
        raise ValueError(
            "Target FE geometry manifest lacks its current global source authority."
        )
    source_cell_owners = owner_rows(source_view.cell_ids, source_view.cell_owner_ranks)
    source_vertex_owners = owner_rows(
        source_view.vertex_ids, source_view.vertex_owner_ranks
    )
    target_cell_owners = owner_rows(target_view.cell_ids, target_view.cell_owner_ranks)
    target_vertex_owners = owner_rows(
        target_view.vertex_ids, target_view.vertex_owner_ranks
    )
    graph_ids = np.asarray(migration.result.entity_ids, dtype=np.int64)
    graph_owners = np.asarray(migration.result.owners, dtype=np.int32)
    source_owned_rows = np.asarray(source_view.cell_ids, dtype=np.int64)[
        np.asarray(source_view.owned, dtype=np.bool_)
    ]
    source_graph_rows = np.sort(source_owned_rows, kind="stable")
    # The partition result lists only the rows this rank held before migration;
    # the rows it receives are bound by the migration evidence. Kept rows must
    # appear among them, and the evidence must equal the target view's rows.
    kept_graph_rows = np.sort(graph_ids[graph_owners == context.rank], kind="stable")
    migration_rows = np.sort(
        np.asarray(migration.target_cell_ids, dtype=np.int64), kind="stable"
    )
    target_owned_rows = np.asarray(target_view.cell_ids, dtype=np.int64)[
        np.asarray(target_view.owned, dtype=np.bool_)
    ]
    target_directory_owned = np.sort(target_owned_rows, kind="stable")
    mismatched = tuple(
        f"{name} ({left.size} vs {right.size} rows)"
        for name, left, right in (
            ("graph-source", np.sort(graph_ids, kind="stable"), source_graph_rows),
            ("migration-target", migration_rows, target_directory_owned),
        )
        if not np.array_equal(left, right)
    )
    if not np.all(np.isin(kept_graph_rows, migration_rows)):
        mismatched = (*mismatched, "graph-kept")
    if mismatched:
        raise ValueError(
            "Graph migration rows differ from the current compact source or target "
            f"view: {', '.join(mismatched)}."
        )
    for index, (layout, source_plan, target_plan, ownership) in enumerate(
        zip(
            state.field_layouts,
            context.source_plans,
            context.target_plans,
            state.field_ownership or (),
            strict=True,
        )
    ):
        source_map = context.source_discretization.dof_maps[index]
        target_map = context.target_discretization.dof_maps[index]
        directory_owner_mismatch = (
            owner_rows(source_plan.cell_ids, source_plan.cell_owner_ranks)
            != source_cell_owners
            or owner_rows(source_plan.vertex_ids, source_plan.vertex_owner_ranks)
            != source_vertex_owners
            or owner_rows(target_plan.cell_ids, target_plan.cell_owner_ranks)
            != target_cell_owners
            or owner_rows(target_plan.vertex_ids, target_plan.vertex_owner_ranks)
            != target_vertex_owners
        )
        if directory_owner_mismatch or (
            source_plan.field_layout_id != layout.layout_id
            or target_plan.field_layout_id != layout.layout_id
            or source_plan.identity_id != target_plan.identity_id
            or not _finite_element_identity_plan_matches(
                context,
                context.source_discretization,
                source_plan,
                index,
                layout,
                context.source_coordinate_spec,
            )
            or not _finite_element_identity_plan_matches(
                context,
                context.target_discretization,
                target_plan,
                index,
                layout,
                context.target_coordinate_spec,
            )
            or source_plan.topology_id != state.topology_id
            or target_plan.topology_id != state.topology_id
            or source_plan.geometry_id != context.global_geometry_id
            or target_plan.geometry_id != context.global_geometry_id
            or source_plan.distribution_id != context.migration.source_directory_id
            or target_plan.distribution_id != context.migration.target_directory_id
            or source_plan.local_rank != context.rank
            or target_plan.local_rank != context.rank
            or source_plan.part_count != context.parts
            or target_plan.part_count != context.parts
            or source_plan.dof_keys.shape != (source_map.global_dof_count, 3)
            or target_plan.dof_keys.shape != (target_map.global_dof_count, 3)
            or ownership.ownership_id != source_plan.ownership().ownership_id
        ):
            raise ValueError(
                f"FE field {layout.name!r} identity/ownership plan is stale."
            )


def _validate_finite_element_owner_local_topology_identity_plans(
    context: _FiniteElementRepartitionContext, /
) -> None:
    state = context.source_state
    if (
        state.field_ownership is None
        or len(context.source_plans) != len(state.fields)
        or len(context.target_plans) != len(state.fields)
        or len(state.field_layouts) != len(state.fields)
    ):
        raise ValueError("Owner-local FE topology transition lacks field identity plans.")
    if any(
        not isinstance(plan, FiniteElementDofIdentityPlan)
        for plan in (*context.source_plans, *context.target_plans)
    ):
        raise TypeError("Owner-local FE topology plans must be typed identity plans.")

    def owner_rows(identifiers: ArrayLike, owner_ranks: ArrayLike, /) -> dict[int, int]:
        return dict(
            zip(
                np.asarray(identifiers, dtype=np.int64).tolist(),
                np.asarray(owner_ranks, dtype=np.int32).tolist(),
                strict=True,
            )
        )

    source_view = context.source_geometry_view
    target_view = context.target_geometry_view
    source_cell_owners = owner_rows(source_view.cell_ids, source_view.cell_owner_ranks)
    source_vertex_owners = owner_rows(
        source_view.vertex_ids, source_view.vertex_owner_ranks
    )
    target_cell_owners = owner_rows(target_view.cell_ids, target_view.cell_owner_ranks)
    target_vertex_owners = owner_rows(
        target_view.vertex_ids, target_view.vertex_owner_ranks
    )
    for index, (layout, source_plan, target_plan, source_ownership) in enumerate(
        zip(
            state.field_layouts,
            context.source_plans,
            context.target_plans,
            state.field_ownership,
            strict=True,
        )
    ):
        source_map = context.source_discretization.dof_maps[index]
        target_map = context.target_discretization.dof_maps[index]
        if (
            layout.storage_kind != "canonical_dofs"
            or source_plan.field_layout_id != layout.layout_id
            or target_plan.field_layout_id != layout.layout_id
            or owner_rows(source_plan.cell_ids, source_plan.cell_owner_ranks)
            != source_cell_owners
            or owner_rows(source_plan.vertex_ids, source_plan.vertex_owner_ranks)
            != source_vertex_owners
            or owner_rows(target_plan.cell_ids, target_plan.cell_owner_ranks)
            != target_cell_owners
            or owner_rows(target_plan.vertex_ids, target_plan.vertex_owner_ranks)
            != target_vertex_owners
            or source_plan.topology_id != source_view.source_topology_id
            or source_plan.geometry_id != source_view.source_geometry_id
            or source_plan.distribution_id != source_view.distribution_id
            or target_plan.topology_id != target_view.source_topology_id
            or target_plan.geometry_id != target_view.source_geometry_id
            or target_plan.distribution_id != target_view.distribution_id
            or source_plan.local_rank != context.rank
            or target_plan.local_rank != context.rank
            or source_plan.part_count != context.parts
            or target_plan.part_count != context.parts
            or source_plan.dof_keys.shape != (source_map.global_dof_count, 3)
            or target_plan.dof_keys.shape != (target_map.global_dof_count, 3)
            or source_ownership.ownership_id != source_plan.ownership().ownership_id
            or not _finite_element_identity_plan_matches(
                context,
                context.source_discretization,
                source_plan,
                index,
                layout,
                context.source_coordinate_spec,
            )
            or not _finite_element_identity_plan_matches(
                context,
                context.target_discretization,
                target_plan,
                index,
                layout,
                context.target_coordinate_spec,
            )
        ):
            raise ValueError(
                f"Owner-local FE field {layout.name!r} identity plan is stale."
            )


def _finite_element_geometry_source_entry(
    view: OwnerLocalCellGeometry,
    entries: Sequence[CompositionEntry],
    /,
) -> CompositionEntry:
    from ..discretization._cell_geometry import CellGeometryCoordinateMap
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ..meshing._assembly import MeshPart
    from ..meshing._result import CellMeshingResult
    from ..meshing.providers._native_periodic import NativePeriodicSource

    def source_authority_matches(value: object, /) -> bool:
        match value:
            case NativePeriodicSource():
                value.validate_restored()
                return (
                    value.source_id == view.source_id
                    and value.source_revision == view.source_revision
                    and value.binding_id == view.source_context_id
                    and view.source_map_id is None
                )
            case ExactPowerCellGeometrySource():
                context = value.periodic_source_context
                context_id = None if context is None else context.context_id
                return (
                    value.source_id == view.source_id
                    and value.source_revision == view.source_revision
                    and value.source_map_id == view.source_map_id
                    and context_id == view.source_context_id
                )
            case CellGeometryCoordinateMap():
                return (
                    value.source_id == view.source_id
                    and value.source_revision == view.source_revision
                    and value.source_map_id == view.source_map_id
                    and view.source_context_id is None
                )
            case None:
                return view.source_map_id is None and view.source_context_id is None
            case _:
                return False

    matches: list[CompositionEntry] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.entry_id in seen:
            continue
        seen.add(entry.entry_id)
        if entry.role != "topology":
            continue
        if view.source_value is not None:
            value = view.source_value
            authority = view.source_geometry_authority
            # A graph-migrated view keeps its authored source data while the
            # migrated source binding (the view's source context) lives in its
            # geometry authority, so the authored source then binds only the
            # source identity.
            authored_matches = (
                source_authority_matches(value)
                if authority is None or authority is value
                else (
                    isinstance(value, NativePeriodicSource)
                    and isinstance(authority, NativePeriodicSource)
                    and value.source_id == view.source_id
                    and value.source_revision == view.source_revision
                )
            )
            if (
                entry.value is not value
                or entry.structure_id != view.source_mesh_id
                or entry.revision_id != view.source_revision
                or not authored_matches
                or not source_authority_matches(authority)
            ):
                continue
        else:
            value = entry.value
            if isinstance(value, MeshPart):
                if value.part_id != view.part_id or entry.revision_id != view.part_id:
                    continue
                result = value.carrier
            elif isinstance(value, CellMeshingResult):
                result = value
                if entry.revision_id != view.part_id:
                    continue
            else:
                continue
            if (
                not isinstance(result, CellMeshingResult)
                or entry.structure_id != result.mesh.mesh_id
                or result.mesh.mesh_id != view.source_mesh_id
                or result.mesh.topology_id != view.source_topology_id
                or cell_geometry_id(result.geometry) != view.source_geometry_id
                or result.geometry.source_map_id != view.source_map_id
                or not source_authority_matches(result.geometry.source_map)
            ):
                continue
        matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            "Compact FE geometry view lacks one current source geometry authority entry."
        )
    return matches[0]


def _finite_element_geometry_binds_source(
    geometry_entry: CompositionEntry, source_entry: CompositionEntry, /
) -> bool:
    return all(
        any(
            dependency.entry_id == source_entry.entry_id
            and dependency.facet == facet
            and dependency.bound_id == value
            for dependency in geometry_entry.dependencies
        )
        for facet, value in (
            ("structure", source_entry.structure_id),
            ("revision", source_entry.revision_id),
            ("semantics", source_entry.semantics_id),
        )
    )


def _validate_finite_element_repartition_source_composition(
    context: _FiniteElementRepartitionContext, /
) -> None:
    state = context.source_state
    if (
        not context.state_entry_id
        or len(context.field_ids) != len(state.field_layouts)
        or tuple(name for name, _ in context.field_ids)
        != tuple(layout.name for layout in state.field_layouts)
        or len({entry_id for _, entry_id in context.field_ids}) != len(context.field_ids)
    ):
        raise ValueError("FE field entry IDs do not match accepted field layouts.")
    state_entry = context.source_composition.entry(context.state_entry_id)
    state_value = state_entry.value
    if isinstance(state_value, FiniteElementRestartManifest):
        valid_value = _finite_element_restart_manifests_match(
            state_value, context.source_manifest
        )
        expected_revision = context.source_manifest.manifest_id
    elif isinstance(state_value, FiniteElementAcceptedState):
        valid_value = _finite_element_accepted_states_match(state_value, state)
        expected_revision = state.accepted_id
    else:
        raise TypeError(
            "Composition state entry must be an accepted FE state or manifest."
        )
    if not valid_value or state_entry.revision_id != expected_revision:
        raise ValueError(
            "Composition state entry is stale relative to its restart manifest."
        )
    from ..meshing._assembly import MeshPartManifest

    geometry_entry = context.source_composition.entry(context.geometry_entry_id)
    manifest = geometry_entry.value
    source_view = context.source_geometry_view
    if not isinstance(manifest, MeshPartManifest):
        raise ValueError(
            "Accepted FE state lacks its current compact global geometry authority: "
            "the geometry entry is not a mesh-part manifest."
        )
    migrated_directory_id = (
        context.migration.target_directory_id
        if _finite_element_adapted_in_place_after_migration(context)
        else context.migration.source_directory_id
    )
    stale = tuple(
        name
        for name, failed in (
            ("state-structure", state_entry.structure_id != state.topology_id),
            ("geometry-role", geometry_entry.role != "discretization"),
            ("manifest-topology", manifest.source_topology_id != state.topology_id),
            ("manifest-geometry-missing", not manifest.source_geometry_id),
            ("manifest-part", manifest.part_id != source_view.part_id),
            (
                "manifest-geometry",
                manifest.source_geometry_id != source_view.source_geometry_id,
            ),
            (
                "manifest-view-directory",
                manifest.distribution_id != source_view.distribution_id,
            ),
            (
                "manifest-migration-directory",
                manifest.distribution_id != migrated_directory_id,
            ),
            (
                "geometry-structure",
                geometry_entry.structure_id != manifest.source_geometry_id,
            ),
            ("geometry-revision", geometry_entry.revision_id != manifest.manifest_id),
            (
                "state-geometry-dependency",
                not any(
                    dependency.entry_id == geometry_entry.entry_id
                    and dependency.facet == "structure"
                    and dependency.bound_id == geometry_entry.structure_id
                    for dependency in state_entry.dependencies
                ),
            ),
        )
        if failed
    )
    if stale:
        raise ValueError(
            "Accepted FE state lacks its current compact global geometry authority: "
            f"{', '.join(stale)}."
        )
    source_source_entry = _finite_element_geometry_source_entry(
        source_view, context.source_composition.entries
    )
    if not _finite_element_geometry_binds_source(geometry_entry, source_source_entry):
        raise ValueError(
            "Accepted FE geometry manifest lacks its actual source root binding."
        )
    context.geometry_entry = geometry_entry
    context.global_geometry_id = manifest.source_geometry_id
    context.state_entry = state_entry
    context.state_value = state_value
    for index, (layout, (_, entry_id)) in enumerate(
        zip(state.field_layouts, context.field_ids, strict=True)
    ):
        entry = context.source_composition.entry(entry_id)
        if (
            entry.role != state_entry.role
            or entry.owner_id != state_entry.owner_id
            or entry.structure_id != state_entry.structure_id
            or entry.revision_id != state.accepted_id
            or entry.semantics_id != layout.layout_id
            or array_tree_fingerprint(np.asarray(entry.value))
            != array_tree_fingerprint(np.asarray(state.fields[index]))
        ):
            raise ValueError(
                f"FE field sidecar {layout.name!r} is not its accepted field reference."
            )
        context.field_source_entries.append(entry)


def _validate_finite_element_restart_bank_names(
    context: _FiniteElementRepartitionContext, /
) -> None:
    auxiliary_names = tuple(name for name, _ in context.source_auxiliary)
    integrator_names = tuple(name for name, _ in context.source_integrator)
    if (
        len(context.auxiliary_ids) != len(auxiliary_names)
        or tuple(name for name, _ in context.auxiliary_ids) != auxiliary_names
        or len({entry_id for _, entry_id in context.auxiliary_ids})
        != len(context.auxiliary_ids)
        or len(context.integrator_ids) != len(integrator_names)
        or tuple(name for name, _ in context.integrator_ids) != integrator_names
        or len({entry_id for _, entry_id in context.integrator_ids})
        != len(context.integrator_ids)
    ):
        raise ValueError("FE restart bank entry IDs do not match manifest names.")
    for values, source_layouts, target_layouts, names, label in (
        (
            context.source_auxiliary,
            context.source_auxiliary_layouts,
            context.target_auxiliary_layouts,
            auxiliary_names,
            "Auxiliary",
        ),
        (
            context.source_integrator,
            context.source_integrator_layouts,
            context.target_integrator_layouts,
            integrator_names,
            "Integrator",
        ),
    ):
        if not values and (
            source_layouts not in (None, ()) or target_layouts not in (None, ())
        ):
            raise ValueError(f"Empty {label} state has unexpected restart layouts.")
        if values and (source_layouts is None or target_layouts is None):
            raise ValueError(f"FE repartition requires explicit {label.lower()} layouts.")
        if (
            source_layouts is not None
            and tuple(layout.name for layout in source_layouts) != names
        ):
            raise ValueError(f"{label} source layouts differ from payload names.")
        if (
            target_layouts is not None
            and tuple(layout.name for layout in target_layouts) != names
        ):
            raise ValueError(f"{label} target layouts differ from payload names.")
    declared_layouts = (
        *(
            ()
            if context.source_auxiliary_layouts is None
            else context.source_auxiliary_layouts
        ),
        *(
            ()
            if context.source_integrator_layouts is None
            else context.source_integrator_layouts
        ),
        *(
            ()
            if context.target_auxiliary_layouts is None
            else context.target_auxiliary_layouts
        ),
        *(
            ()
            if context.target_integrator_layouts is None
            else context.target_integrator_layouts
        ),
        *(
            ()
            if context.source_manifest.material_layouts is None
            else context.source_manifest.material_layouts
        ),
        *(
            ()
            if context.target_material_layouts is None
            else context.target_material_layouts
        ),
    )
    if any(
        not isinstance(layout, FiniteElementRestartStateLayout)
        for layout in declared_layouts
    ):
        raise TypeError("FE restart layouts must be explicit typed layouts.")


def _validate_finite_element_material_contract(
    context: _FiniteElementRepartitionContext, /
) -> None:
    materials = context.source_materials
    source_layouts = context.source_manifest.material_layouts
    target_layouts = context.target_material_layouts
    if (materials is None) != (context.material_entry_id is None):
        raise ValueError(
            "Accepted FE material state requires exactly one Composition root."
        )
    if materials is None:
        if source_layouts not in (None, ()) or target_layouts not in (None, ()):
            raise ValueError("Material layouts exist without accepted material state.")
        if (
            context.source_material_plan is not None
            or context.target_material_plan is not None
        ):
            if (
                context.source_material_plan is None
                or context.target_material_plan is None
                or context.source_material_plan.layout_id
                != context.target_material_plan.layout_id
            ):
                raise ValueError("FE material plans differ across repartition.")
        return
    if source_layouts is None or target_layouts is None:
        raise ValueError(
            "Replicated material state requires explicit source/target layouts."
        )
    site_names = tuple(state.site_id.key for state in materials.states)
    if (
        tuple(layout.name for layout in source_layouts) != site_names
        or tuple(layout.name for layout in target_layouts) != site_names
    ):
        raise ValueError("Material layouts differ from replicated transaction rows.")
    source_plan = context.source_material_plan
    target_plan = context.target_material_plan
    if source_plan is None and target_plan is None:
        pass
    elif (
        isinstance(source_plan, MaterialIntegrationPlan)
        and isinstance(target_plan, MaterialIntegrationPlan)
        and source_plan.layout_id == target_plan.layout_id
    ):
        source_plan.validate(materials)
        target_plan.validate(materials)
        if (
            tuple(site.key for site in source_plan.site_ids) != site_names
            or tuple(site.key for site in target_plan.site_ids) != site_names
        ):
            raise ValueError("Material plans differ from replicated transaction rows.")
    else:
        raise ValueError(
            "Material repartition requires matching plans or no plan on either side."
        )
    all_layouts = (*source_layouts, *target_layouts)
    if any(
        layout.role is not FiniteElementRestartStateRole.MODEL_STATE
        or layout.association is not FiniteElementRestartStateAssociation.REPLICATED
        or layout.ownership is not None
        for layout in all_layouts
    ):
        raise ValueError(
            "Owned material restart rows require an owner-aware material plan."
        )
    if tuple(layout.layout_id for layout in source_layouts) != tuple(
        layout.layout_id for layout in target_layouts
    ):
        raise ValueError("Replicated material layouts changed.")


def _validate_finite_element_material_entry(
    context: _FiniteElementRepartitionContext, /
) -> None:
    if context.material_entry_id is None:
        return
    entry = context.source_composition.entry(context.material_entry_id)
    expected = context.source_materials
    if (
        expected is None
        or not isinstance(entry.value, MaterialTransaction)
        or not _finite_element_materials_match(entry.value, expected)
        or entry.role != "model-state"
        or entry.structure_id != expected.layout_id
        or entry.revision_id != expected.transaction_id
    ):
        raise ValueError("Composition material root differs from accepted FE state.")
    context.material_entry = entry


def _finite_element_repartition_schema_id(
    context: _FiniteElementRepartitionContext, /
) -> str:
    reprepare_ids = tuple(entry.entry_id for entry in context.reprepared)
    if len(set(reprepare_ids)) != len(reprepare_ids):
        raise ValueError("Target FE reprepare entries contain duplicate IDs.")
    return canonical_fingerprint(
        {
            "kind": "finite-element-repartition-bank-schema",
            "state_entry": context.state_entry_id,
            "topology": context.source_state.topology_id,
            "geometry": context.global_geometry_id,
            "field_entries": context.field_ids,
            "auxiliary_entries": context.auxiliary_ids,
            "integrator_entries": context.integrator_ids,
            "material_entry": context.material_entry_id,
            "source_material_present": context.source_materials is not None,
            "retained_entries": tuple(sorted(context.retained)),
            "invalidated_entries": tuple(sorted(context.invalidated)),
            "reprepare_entries": tuple(sorted(reprepare_ids)),
            "fields": [
                (layout.name, layout.layout_id)
                for layout in context.source_state.field_layouts
            ],
            "auxiliary": _finite_element_restart_layout_schema(
                context.source_auxiliary_layouts
            ),
            "target_auxiliary": _finite_element_restart_layout_schema(
                context.target_auxiliary_layouts
            ),
            "integrator": _finite_element_restart_layout_schema(
                context.source_integrator_layouts
            ),
            "target_integrator": _finite_element_restart_layout_schema(
                context.target_integrator_layouts
            ),
            "material_plan": (
                None
                if context.source_material_plan is None
                else context.source_material_plan.layout_id
            ),
            "target_material_plan": (
                None
                if context.target_material_plan is None
                else context.target_material_plan.layout_id
            ),
            "source_material_layouts": _finite_element_restart_layout_schema(
                context.source_manifest.material_layouts
            ),
            "target_material_layouts": _finite_element_restart_layout_schema(
                context.target_material_layouts
            ),
        }
    )


def _validate_finite_element_owner_local_topology_preflight(
    context: _FiniteElementRepartitionContext, /
) -> None:
    source_view = context.source_geometry_view
    target_view = context.target_geometry_view
    migration = context.migration
    if (
        not context.topology_change
        or not isinstance(source_view, OwnerLocalCellGeometry)
        or not isinstance(target_view, OwnerLocalCellGeometry)
        or not isinstance(
            migration, (OwnerLocalGraphMigrationEvidence, OwnerLocalTopologyTransferRoute)
        )
    ):
        raise ValueError(
            "Owner-local FE topology transition inputs are stale: it needs compact "
            "source/target views and graph migration evidence."
        )
    ownership = context.source_state.field_ownership
    stale = tuple(
        name
        for name, failed in (
            (
                "migration-evidence",
                isinstance(migration, OwnerLocalGraphMigrationEvidence)
                and migration is not target_view.graph_migration_evidence,
            ),
            ("group", migration.group_id != context.execution_group.spec.group_id),
            ("rank", migration.local_rank != context.rank),
            ("part-count", migration.part_count != context.parts),
            (
                "migration-directories",
                not _finite_element_adapted_in_place_after_migration(context)
                and (
                    migration.source_directory_id != source_view.distribution_id
                    or migration.target_directory_id != target_view.distribution_id
                    or (
                        isinstance(migration, OwnerLocalGraphMigrationEvidence)
                        and migration.target_part_id != target_view.part_id
                    )
                ),
            ),
            (
                "view-owner",
                source_view.owner_rank != context.rank
                or target_view.owner_rank != context.rank,
            ),
            (
                "form",
                context.source_compiled.form.form_id
                != context.target_compiled.form.form_id
                or context.source_compiled.form.field_names
                != context.target_compiled.form.field_names,
            ),
            (
                "field-layouts",
                tuple(layout.name for layout in context.source_state.field_layouts)
                != context.source_compiled.form.field_names,
            ),
            ("field-ownership", ownership is None),
            (
                "source-topology",
                source_view.source_topology_id != context.source_state.topology_id,
            ),
            (
                "ownership-directory",
                ownership is not None
                and source_view.distribution_id != ownership[0].distribution_id,
            ),
            (
                "unchanged-topology",
                target_view.source_topology_id == context.source_state.topology_id,
            ),
            ("axis", not context.axis_name.strip()),
        )
        if failed
    )
    if stale:
        raise ValueError(
            f"Owner-local FE topology transition inputs are stale: {', '.join(stale)}."
        )


def _finite_element_owner_local_topology_schema(
    context: _FiniteElementRepartitionContext, /
) -> dict[str, object]:
    """Group-global declaration schema; rank-local candidate IDs stay out of it."""
    return {
        "kind": "finite-element-owner-local-topology-schema",
        "state_entry": context.state_entry_id,
        "source_topology": context.source_geometry_view.source_topology_id,
        "target_topology": context.target_geometry_view.source_topology_id,
        "source_geometry": context.global_geometry_id,
        "target_geometry": context.target_global_geometry_id,
        "source_directory": context.migration.source_directory_id,
        "target_directory": context.migration.target_directory_id,
        "graph_plan": migration.result.plan_id
        if isinstance((migration := context.migration), OwnerLocalGraphMigrationEvidence)
        else None,
        "graph_result": migration.result.result_id
        if isinstance(migration, OwnerLocalGraphMigrationEvidence)
        else None,
        "field_entries": context.field_ids,
        "auxiliary_entries": context.auxiliary_ids,
        "integrator_entries": context.integrator_ids,
        "material_entry": context.material_entry_id,
        "retained_entries": tuple(sorted(context.retained)),
        "invalidated_entries": tuple(sorted(context.invalidated)),
        "reprepare_entries": tuple(
            sorted(entry.entry_id for entry in context.reprepared)
        ),
        "field_layouts": tuple(
            (layout.name, layout.layout_id, layout.transfer_semantics.value)
            for layout in context.source_state.field_layouts
        ),
        "source_auxiliary": _finite_element_restart_layout_schema(
            context.source_auxiliary_layouts
        ),
        "target_auxiliary": _finite_element_restart_layout_schema(
            context.target_auxiliary_layouts
        ),
        "source_integrator": _finite_element_restart_layout_schema(
            context.source_integrator_layouts
        ),
        "target_integrator": _finite_element_restart_layout_schema(
            context.target_integrator_layouts
        ),
        "source_material_plan": (
            None
            if context.source_material_plan is None
            else context.source_material_plan.layout_id
        ),
        "target_material_plan": (
            None
            if context.target_material_plan is None
            else context.target_material_plan.layout_id
        ),
        "source_material_layouts": _finite_element_restart_layout_schema(
            context.source_manifest.material_layouts
        ),
        "target_material_layouts": _finite_element_restart_layout_schema(
            context.target_material_layouts
        ),
    }


def _admit_finite_element_owner_local_topology(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    error = _validate_finite_element_repartition_collectively(
        context,
        "source_composition",
        lambda: _validate_finite_element_repartition_source_composition(context),
    )
    if error is not None:
        return error
    error = _validate_finite_element_repartition_collectively(
        context,
        "topology_preflight",
        lambda: _validate_finite_element_owner_local_topology_preflight(context),
    )
    if error is not None:
        return error
    error = _validate_finite_element_repartition_collectively(
        context,
        "topology_authority",
        lambda: _validate_finite_element_owner_local_topology_authority(context),
    )
    if error is not None:
        return error

    def validate_schema() -> None:
        _validate_finite_element_owner_local_topology_identity_plans(context)
        _validate_finite_element_restart_bank_names(context)
        _validate_finite_element_material_contract(context)
        _validate_finite_element_material_entry(context)
        context.schema_id = canonical_fingerprint(
            _finite_element_owner_local_topology_schema(context)
        )

    error = _validate_finite_element_repartition_collectively(
        context, "topology_schema", validate_schema
    )
    if error is not None:
        return error
    rows = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(context.schema_id), dtype=np.uint8),
        context.execution_group,
        f"{context.axis_name}_topology_schema",
    )
    if all(
        np.asarray(row, dtype=np.uint8).tobytes().hex() == context.schema_id
        for row in rows
    ):
        return None
    differing = _owner_group_differing_declarations(
        _finite_element_owner_local_topology_schema(context),
        context.execution_group,
        f"{context.axis_name}_topology_schema_parts",
    )
    return (
        "Owner-local FE topology field/bank schemas differ across ranks: "
        f"{', '.join(differing)}."
    )


def _admit_finite_element_repartition(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    def validate() -> None:
        _validate_finite_element_repartition_authority(context)
        _validate_finite_element_repartition_source_composition(context)
        _validate_finite_element_repartition_identity_plans(context)
        _validate_finite_element_restart_bank_names(context)
        _validate_finite_element_material_contract(context)
        _validate_finite_element_material_entry(context)
        context.schema_id = _finite_element_repartition_schema_id(context)

    error = _validate_finite_element_repartition_collectively(
        context, "source_admission", validate
    )
    if error is not None:
        return error
    rows = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(context.schema_id), dtype=np.uint8),
        context.execution_group,
        f"{context.axis_name}_schema",
    )
    if any(
        np.asarray(row, dtype=np.uint8).tobytes().hex() != context.schema_id
        for row in rows
    ):
        return "FE repartition bank/layout schemas differ across ranks."
    return None


def _migrate_finite_element_owner_local_adaptive_fields(
    transaction: "FiniteElementTopologyTransaction",
    context: _FiniteElementRepartitionContext,
    /,
) -> str | None:
    metadata = _finite_element_owner_local_adaptive_metadata(context.adaptive_result)
    state = context.source_state
    if state.field_ownership is None:
        return "Accepted FE source state lacks owner-local field rows."
    for index, (
        layout,
        source_plan,
        target_plan,
        source_ownership,
        source_entry,
    ) in enumerate(
        zip(
            state.field_layouts,
            context.source_plans,
            context.target_plans,
            state.field_ownership,
            context.field_source_entries,
            strict=True,
        )
    ):
        local_error = None
        action: FiniteElementFieldTransferAction | None = None
        target_ownership: FiniteElementDofOwnership | None = None
        target_structure_id = ""
        try:
            action = transaction._field_transfer_action(layout.transfer_semantics)
            target_ownership = target_plan.ownership()
            target_structure_id = _owner_local_target_field_structure(
                context, layout, target_ownership
            )
            if layout.storage_kind != "canonical_dofs":
                raise ValueError(
                    "Owner-local topology transfer needs canonical FE DOF rows."
                )
            if (
                layout.transfer_semantics
                is FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE
            ):
                if (
                    context.target_geometry_view.topological_dimension
                    != context.source_geometry_view.topological_dimension
                    or context.target_geometry_view.coordinates.shape[1]
                    != context.source_geometry_view.coordinates.shape[1]
                ):
                    raise ValueError(
                        "Conservative P0 adaptation changed physical dimension."
                    )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        error = _finite_element_repartition_collective_error(
            context, local_error, f"adaptive_field_{index}_preflight"
        )
        if error is not None:
            return error
        if target_ownership is None:
            return "Owner-local FE target field ownership was not constructed."
        try:
            evidence, local_values = _transfer_finite_element_owner_local_adaptive_field(
                transaction,
                context,
                layout,
                source_plan,
                target_plan,
                source_ownership,
                target_ownership,
                source_entry,
                state.fields[index],
                metadata,
                target_structure_id=target_structure_id,
                action=action,
                field_index=index,
                axis_name=f"{context.axis_name}_adaptive_field_{index}",
            )
            local_error = (
                None
                if local_values.shape[0] == target_plan.dof_keys.shape[0]
                else "Owner-local FE transfer omitted target consumer rows."
            )
        except Exception as error:
            evidence = None
            local_values = None
            local_error = f"{type(error).__name__}: {error}"
        error = _finite_element_repartition_collective_error(
            context, local_error, f"adaptive_field_{index}_transfer"
        )
        if error is not None or evidence is None or local_values is None:
            return error or "Owner-local FE field transfer produced no evidence."
        context.topology_field_transfers.append(evidence)
        context.target_local_fields.append(local_values)
    return None


def _validate_finite_element_owner_local_target_consumer(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    local_error = None
    try:
        if len(context.target_local_fields) != len(context.target_plans):
            raise ValueError("Owner-local FE target fields differ from target plans.")
        for layout, plan, values in zip(
            context.source_state.field_layouts,
            context.target_plans,
            context.target_local_fields,
            strict=True,
        ):
            if (
                values.ndim == 0
                or values.shape[0] != plan.dof_keys.shape[0]
                or tuple(values.shape[1:]) != layout.component_shape
                or not bool(np.asarray(jnp.all(jnp.isfinite(values))))
            ):
                raise ValueError(
                    f"Target FE field {layout.name!r} differs from its compiled local map."
                )
        full_state: object = (
            context.target_local_fields[0]
            if len(context.target_local_fields) == 1
            else tuple(context.target_local_fields)
        )
        context.target_compiled.full_space.validate(full_state)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    return _finite_element_repartition_collective_error(
        context, local_error, "adaptive_target_consumer"
    )


def _migrate_finite_element_fields(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    try:
        state = context.source_state
        if state.field_ownership is None or context.state_entry is None:
            raise ValueError(
                "Accepted FE source authority lost its typed state ownership."
            )
        for index, (layout, source_plan, target_plan, source_ownership) in enumerate(
            zip(
                state.field_layouts,
                context.source_plans,
                context.target_plans,
                state.field_ownership,
                strict=True,
            )
        ):
            target_ownership = target_plan.ownership()
            source_entry = context.field_source_entries[index]
            key_identity = canonical_fingerprint(
                {
                    "kind": "finite-element-canonical-dof-state",
                    "field_layout": layout.layout_id,
                    "source_identity": source_plan.identity_id,
                    "target_identity": target_plan.identity_id,
                    "source_topology": context.source_geometry_view.source_topology_id,
                    "source_geometry": context.global_geometry_id,
                }
            )
            target_structure = _owner_local_target_field_structure(
                context, layout, target_ownership
            )
            transfer = context.migration.transfer_keyed_values(
                _owner_local_keyed_declaration(context, source_entry),
                source_ownership.dof_keys,
                source_ownership.owner_ranks,
                state.fields[index],
                target_ownership.dof_keys,
                target_ownership.owner_ranks,
                context.execution_group,
                target_structure_id=target_structure,
                key_identity_id=key_identity,
                axis_name=f"{context.axis_name}_field_{index}",
                value_fiber_shape=layout.component_shape,
            )
            local_error = None
            if not isinstance(transfer, OwnerLocalGraphKeyedStateTransfer):
                local_error = "Graph migration returned no typed FE field transfer."
            elif transfer.target_values.shape[0] != target_ownership.dof_keys.shape[0]:
                local_error = "Migrated FE field rows differ from target ownership."
            error = _finite_element_repartition_collective_error(
                context, local_error, f"field_transfer_{index}"
            )
            if error is not None:
                return error
            context.field_transfers.append(transfer)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    else:
        local_error = None
    return _finite_element_repartition_collective_error(
        context, local_error, "field_transfer"
    )


def _validate_finite_element_target_consumer(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    local_fields: list[Array] = []
    local_error = None
    try:
        for index, (plan, transfer) in enumerate(
            zip(context.target_plans, context.field_transfers, strict=True)
        ):
            values = transfer.gather_target_local_values(
                plan.dof_keys,
                context.execution_group,
                axis_name=f"{context.axis_name}_target_local_field_{index}",
            )
            local_error = (
                None
                if values.shape[0] == plan.dof_keys.shape[0]
                else "Target FE local gather differs from its actual DOF map."
            )
            error = _finite_element_repartition_collective_error(
                context, local_error, f"target_local_field_{index}"
            )
            if error is not None:
                return error
            local_fields.append(values)
        full_state: object = (
            local_fields[0] if len(local_fields) == 1 else tuple(local_fields)
        )
        context.target_compiled.full_space.validate(full_state)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    else:
        local_error = None
    return _finite_element_repartition_collective_error(
        context, local_error, "target_compiled_consumer"
    )


def _finite_element_payload_identity(
    value: ArrayLike, layout: FiniteElementRestartStateLayout, /
) -> str:
    import jax

    array = jnp.asarray(value)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        return canonical_fingerprint(
            {
                "kind": "finite-element-typed-key",
                "implementation": str(jax.random.key_impl(array)),
                "data": array_tree_fingerprint(jax.random.key_data(array)),
            }
        )
    return canonical_fingerprint(
        {
            "kind": "finite-element-restart-payload",
            "layout": layout.layout_id,
            "array": array_tree_fingerprint(np.asarray(array)),
        }
    )


def _finite_element_ownership_matches_plan(
    context: _FiniteElementRepartitionContext,
    ownership: FiniteElementDofOwnership,
    plans: tuple[FiniteElementDofIdentityPlan, ...],
    directory_id: str,
    /,
) -> bool:
    for plan in plans:
        plan_keys = np.asarray(plan.dof_keys, dtype=np.int64)
        owned_keys = np.asarray(ownership.dof_keys, dtype=np.int64)
        if (
            ownership.field_layout_id == plan.field_layout_id
            and ownership.identity_id == plan.identity_id
            and ownership.orientation_action_id == plan.orientation_action_id
            and ownership.distribution_id == directory_id
            and ownership.topology_id == plan.topology_id
            and ownership.geometry_id == plan.geometry_id
            and ownership.local_rank == context.rank
            and ownership.part_count == context.parts
            and set(tuple(row) for row in owned_keys.tolist()).issubset(
                set(tuple(row) for row in plan_keys.tolist())
            )
        ):
            return True
    return False


def _validate_finite_element_payload_bank(
    context: _FiniteElementRepartitionContext,
    payloads: tuple[tuple[str, Array], ...],
    source_layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
    target_layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
    entry_ids: dict[str, str],
    label: str,
    /,
) -> None:
    if not payloads:
        if source_layouts not in (None, ()) or target_layouts not in (None, ()):
            raise ValueError(f"Empty {label} state has unexpected restart layouts.")
        return
    if source_layouts is None or target_layouts is None:
        raise ValueError(f"{label} payload requires explicit source/target layouts.")
    source_by_name = {layout.name: layout for layout in source_layouts}
    target_by_name = {layout.name: layout for layout in target_layouts}
    names = {name for name, _ in payloads}
    if (
        len(source_by_name) != len(source_layouts)
        or len(target_by_name) != len(target_layouts)
        or set(source_by_name) != set(target_by_name)
        or set(source_by_name) != names
        or set(entry_ids) != names
    ):
        raise ValueError(f"{label} payload/layout/entry names disagree.")
    state_entry = context.state_entry
    if state_entry is None:
        raise ValueError("FE payload source lacks its accepted state entry.")
    for name, value in payloads:
        source_layout = source_by_name[name]
        target_layout = target_by_name[name]
        if (
            source_layout.role != target_layout.role
            or source_layout.association != target_layout.association
            or source_layout.prng_impl != target_layout.prng_impl
        ):
            raise ValueError(
                f"{label} payload {name!r} changes its explicit role/layout."
            )
        source_layout.validate_value(value)
        entry = context.source_composition.entry(entry_ids[name])
        if (
            entry.role != source_layout.role.value
            or entry.owner_id != state_entry.owner_id
            or entry.structure_id != source_layout.layout_id
            or entry.revision_id != context.source_manifest.manifest_id
            or entry.semantics_id != source_layout.semantics_id
            or _finite_element_payload_identity(entry.value, source_layout)
            != _finite_element_payload_identity(value, source_layout)
        ):
            raise ValueError(
                f"{label} Composition entry {entry_ids[name]!r} differs from its manifest payload."
            )
        if source_layout.association is FiniteElementRestartStateAssociation.OWNED_ROWS:
            source_ownership = source_layout.ownership
            target_ownership = target_layout.ownership
            if (
                not isinstance(source_ownership, FiniteElementDofOwnership)
                or not isinstance(target_ownership, FiniteElementDofOwnership)
                or not _finite_element_ownership_matches_plan(
                    context,
                    source_ownership,
                    context.source_plans,
                    context.source_geometry_view.distribution_id,
                )
                or not _finite_element_ownership_matches_plan(
                    context,
                    target_ownership,
                    context.target_plans,
                    context.target_geometry_view.distribution_id,
                )
            ):
                raise ValueError(
                    f"{label} payload {name!r} lacks exact source/target DOF ownership."
                )
        elif source_layout.ownership is not None or target_layout.ownership is not None:
            raise ValueError(
                f"Replicated {label} payload {name!r} cannot carry DOF ownership."
            )


def _validate_finite_element_target_payload(
    context: _FiniteElementRepartitionContext,
    value: ArrayLike,
    layout: FiniteElementRestartStateLayout,
    phase: str,
    /,
) -> Array:
    target_value: Array | None = None
    local_error = None
    try:
        target_value = layout.validate_value(value)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    error = _finite_element_repartition_collective_error(context, local_error, phase)
    if error is not None or target_value is None:
        raise ValueError(error or "FE target payload validation produced no value.")
    return target_value


def _transfer_finite_element_payload_bank(
    context: _FiniteElementRepartitionContext,
    payloads: tuple[tuple[str, Array], ...],
    source_layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
    target_layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
    entry_ids: dict[str, str],
    label: str,
    /,
    *,
    transaction: "FiniteElementTopologyTransaction | None" = None,
) -> tuple[tuple[str, Array], ...]:
    if not payloads:
        return ()
    source_by_name = {layout.name: layout for layout in source_layouts or ()}
    target_by_name = {layout.name: layout for layout in target_layouts or ()}
    target_values: list[tuple[str, Array]] = []
    for name, source_value in payloads:
        source_layout = source_by_name[name]
        target_layout = target_by_name[name]
        source_entry = context.source_composition.entry(entry_ids[name])
        value_id = ""
        transfer: (
            OwnerLocalGraphKeyedStateTransfer
            | FiniteElementOwnerLocalFieldTransferEvidence
            | None
        )
        if source_layout.association is FiniteElementRestartStateAssociation.OWNED_ROWS:
            source_ownership = source_layout.ownership
            target_ownership = target_layout.ownership
            if not isinstance(
                source_ownership, FiniteElementDofOwnership
            ) or not isinstance(target_ownership, FiniteElementDofOwnership):
                raise TypeError("Owned FE restart payloads need typed DOF ownership.")
            if context.topology_change:
                if transaction is None:
                    raise ValueError(
                        "Owner-local topology history transfer needs its transaction."
                    )
                field_indices = tuple(
                    index
                    for index, layout in enumerate(context.source_state.field_layouts)
                    if layout.layout_id == source_ownership.field_layout_id
                )
                if len(field_indices) != 1:
                    raise ValueError(
                        f"Owned {label} payload {name!r} lacks one FE field semantic."
                    )
                field_index = field_indices[0]
                field_layout = context.source_state.field_layouts[field_index]
                evidence, _ = _transfer_finite_element_owner_local_adaptive_field(
                    transaction,
                    context,
                    field_layout,
                    context.source_plans[field_index],
                    context.target_plans[field_index],
                    source_ownership,
                    target_ownership,
                    source_entry,
                    source_value,
                    _finite_element_owner_local_adaptive_metadata(
                        context.adaptive_result
                    ),
                    target_structure_id=target_layout.layout_id,
                    action=transaction._field_transfer_action(
                        field_layout.transfer_semantics
                    ),
                    field_index=field_index,
                    axis_name=f"{context.axis_name}_{label}_{name}",
                )
                transfer = evidence
                target_value = _validate_finite_element_target_payload(
                    context,
                    evidence.target_values,
                    target_layout,
                    f"{context.axis_name}_{label}_{name}_target",
                )
                context.topology_payload_transfers.append(evidence)
                context.payload_roots.append(
                    (
                        f"{label}:{name}",
                        evidence.source_content_root,
                        evidence.target_content_root,
                    )
                )
            else:
                key_identity = canonical_fingerprint(
                    {
                        "kind": "finite-element-restart-owned-payload",
                        "source_layout": source_layout.layout_id,
                        "target_layout": target_layout.layout_id,
                        "field_layout": source_ownership.field_layout_id,
                        "source_identity": source_ownership.identity_id,
                        "target_identity": target_ownership.identity_id,
                    }
                )
                transfer = context.migration.transfer_keyed_values(
                    _owner_local_keyed_declaration(context, source_entry),
                    source_ownership.dof_keys,
                    source_ownership.owner_ranks,
                    source_value,
                    target_ownership.dof_keys,
                    target_ownership.owner_ranks,
                    context.execution_group,
                    target_structure_id=target_layout.layout_id,
                    key_identity_id=key_identity,
                    axis_name=f"{context.axis_name}_{label}_{name}",
                    value_fiber_shape=tuple(int(size) for size in source_value.shape[1:]),
                )
                target_value = _validate_finite_element_target_payload(
                    context,
                    transfer.target_values,
                    target_layout,
                    f"{context.axis_name}_{label}_{name}_target",
                )
                context.payload_transfers.append(transfer)
                context.payload_roots.append(
                    (
                        f"{label}:{name}",
                        transfer.source_content_root,
                        transfer.target_content_root,
                    )
                )
        else:
            value_id = _finite_element_payload_identity(source_value, source_layout)
            gathered = _owner_group_allgather(
                np.frombuffer(bytes.fromhex(value_id), dtype=np.uint8),
                context.execution_group,
                f"{context.axis_name}_{label}_{name}_replicas",
            )
            if any(
                np.asarray(row, dtype=np.uint8).tobytes().hex() != value_id
                for row in gathered
            ):
                raise ValueError(
                    f"Replicated {label} payload {name!r} differs across owners."
                )
            transfer = None
            target_value = _validate_finite_element_target_payload(
                context,
                source_value,
                target_layout,
                f"{context.axis_name}_{label}_{name}_replica_target",
            )
            context.payload_roots.append((f"{label}:{name}", value_id, value_id))
        context.payload_records.append(
            (
                label,
                name,
                source_entry,
                target_layout,
                target_value,
                transfer,
                value_id,
            )
        )
        target_values.append((name, target_value))
    return tuple(target_values)


def _migrate_finite_element_restart_banks(
    context: _FiniteElementRepartitionContext,
    /,
    *,
    transaction: "FiniteElementTopologyTransaction | None" = None,
) -> str | None:
    auxiliary_ids = dict(context.auxiliary_ids)
    integrator_ids = dict(context.integrator_ids)

    def validate() -> None:
        _validate_finite_element_payload_bank(
            context,
            context.source_auxiliary,
            context.source_auxiliary_layouts,
            context.target_auxiliary_layouts,
            auxiliary_ids,
            "auxiliary",
        )
        _validate_finite_element_payload_bank(
            context,
            context.source_integrator,
            context.source_integrator_layouts,
            context.target_integrator_layouts,
            integrator_ids,
            "integrator",
        )

    error = _validate_finite_element_repartition_collectively(
        context, "restart_payload_admission", validate
    )
    if error is not None:
        return error
    try:
        context.auxiliary_values = _transfer_finite_element_payload_bank(
            context,
            context.source_auxiliary,
            context.source_auxiliary_layouts,
            context.target_auxiliary_layouts,
            auxiliary_ids,
            "auxiliary",
            transaction=transaction,
        )
        context.integrator_values = _transfer_finite_element_payload_bank(
            context,
            context.source_integrator,
            context.source_integrator_layouts,
            context.target_integrator_layouts,
            integrator_ids,
            "integrator",
            transaction=transaction,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        context.auxiliary_values = context.source_auxiliary
        context.integrator_values = context.source_integrator
    else:
        local_error = None
    return _finite_element_repartition_collective_error(
        context, local_error, "restart_payloads"
    )


def _qualify_finite_element_material_state(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    try:
        materials = context.source_materials
        if materials is not None:
            source_plan = context.source_material_plan
            target_plan = context.target_material_plan
            source_layouts = context.source_manifest.material_layouts
            target_layouts = context.target_material_layouts
            if source_layouts is None or target_layouts is None:
                raise ValueError(
                    "Replicated material state requires explicit source/target layouts."
                )
            names = tuple(state.site_id.key for state in materials.states)
            if (
                tuple(layout.name for layout in source_layouts) != names
                or tuple(layout.name for layout in target_layouts) != names
            ):
                raise ValueError(
                    "Material layouts differ from replicated transaction rows."
                )
            if source_plan is None and target_plan is None:
                plan_id = None
            elif (
                isinstance(source_plan, MaterialIntegrationPlan)
                and isinstance(target_plan, MaterialIntegrationPlan)
                and source_plan.layout_id == target_plan.layout_id
            ):
                source_plan.validate(materials)
                target_plan.validate(materials)
                if (
                    tuple(site.key for site in source_plan.site_ids) != names
                    or tuple(site.key for site in target_plan.site_ids) != names
                ):
                    raise ValueError(
                        "Material plans differ from replicated transaction rows."
                    )
                plan_id = source_plan.layout_id
            else:
                raise ValueError(
                    "Material repartition requires matching plans or no plan on either side."
                )
            context.material_root = canonical_fingerprint(
                {
                    "kind": "finite-element-replicated-material-state",
                    "plan": plan_id,
                    "states": [
                        (
                            state.site_id.key,
                            state.model_id,
                            state.state_version,
                            array_tree_fingerprint(state.committed),
                            array_tree_fingerprint(state.trial),
                        )
                        for state in materials.states
                    ],
                }
            )
            rows = _owner_group_allgather(
                np.frombuffer(bytes.fromhex(context.material_root), dtype=np.uint8),
                context.execution_group,
                f"{context.axis_name}_replicated_materials",
            )
            if any(
                np.asarray(row, dtype=np.uint8).tobytes().hex() != context.material_root
                for row in rows
            ):
                raise ValueError(
                    "Replicated FE material transaction differs across owners."
                )
        elif context.source_manifest.material_layouts not in (None, ()):
            raise ValueError("Material layouts exist without accepted material state.")
        elif context.target_material_layouts not in (None, ()):
            raise ValueError(
                "Target material layouts exist without accepted material state."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    else:
        local_error = None
    return _finite_element_repartition_collective_error(
        context, local_error, "material_state"
    )


def _build_finite_element_repartition_target_state(
    context: _FiniteElementRepartitionContext, /
) -> str | None:
    try:
        field_ownership = tuple(plan.ownership() for plan in context.target_plans)
        transfers = (
            context.topology_field_transfers
            if context.topology_change
            else context.field_transfers
        )
        target_topology_id = (
            context.target_geometry_view.source_topology_id
            if context.topology_change
            else context.source_state.topology_id
        )
        target_state_version = (
            context.source_state.state_version + 1
            if context.topology_change
            else context.source_state.state_version
        )
        context.target_state = FiniteElementAcceptedState(
            tuple(transfer.target_values for transfer in transfers),
            context.source_state.time,
            context.source_state.step,
            target_topology_id,
            context.target_discretization.prepared_id,
            context.target_compiled.compilation_id,
            field_layouts=context.source_state.field_layouts,
            materials=context.target_materials,
            schedule_cursor=context.source_state.schedule_cursor,
            state_version=target_state_version,
            field_ownership=field_ownership,
        )
        context.target_manifest = FiniteElementRestartManifest(
            context.target_state,
            auxiliary_state=context.auxiliary_values,
            integrator_state=context.integrator_values,
            auxiliary_layouts=context.target_auxiliary_layouts,
            integrator_layouts=context.target_integrator_layouts,
            material_layouts=context.target_material_layouts,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    else:
        local_error = None
    return _finite_element_repartition_collective_error(
        context, local_error, "target_state"
    )


def _finite_element_target_dependencies(
    context: _FiniteElementRepartitionContext,
    entry: CompositionEntry,
    staged_entries: dict[str, CompositionEntry],
    /,
) -> tuple[CompositionDependency, ...]:
    bindings = []
    for dependency in entry.dependencies:
        target = staged_entries.get(dependency.entry_id)
        if target is None:
            target = context.source_composition.entry(dependency.entry_id)
        bindings.append(target.binding(dependency.facet))
    return tuple(bindings)


def _active_finite_element_field_transfers(
    context: _FiniteElementRepartitionContext, /
) -> Sequence[
    OwnerLocalGraphKeyedStateTransfer | FiniteElementOwnerLocalFieldTransferEvidence
]:
    return (
        context.topology_field_transfers
        if context.topology_change
        else context.field_transfers
    )


def _active_finite_element_payload_transfers(
    context: _FiniteElementRepartitionContext, /
) -> Sequence[
    OwnerLocalGraphKeyedStateTransfer | FiniteElementOwnerLocalFieldTransferEvidence
]:
    return (
        context.topology_payload_transfers
        if context.topology_change
        else context.payload_transfers
    )


def _stage_finite_element_state_transport(
    context: _FiniteElementRepartitionContext, /
) -> None:
    target_state = context.target_state
    target_manifest = context.target_manifest
    source_entry = context.state_entry
    if (
        target_state is None
        or target_manifest is None
        or source_entry is None
        or context.state_value is None
    ):
        raise ValueError("FE repartition target or source state is incomplete.")
    staged = {entry.entry_id: entry for entry in context.reprepared}
    target_state_value: object
    if isinstance(context.state_value, FiniteElementRestartManifest):
        target_state_value = target_manifest
        target_revision = target_manifest.manifest_id
    else:
        target_state_value = target_state
        target_revision = target_state.accepted_id
    target_state_entry = CompositionEntry(
        target_state_value,
        entry_id=source_entry.entry_id,
        role=source_entry.role,
        owner_id=source_entry.owner_id,
        structure_id=target_state.topology_id,
        revision_id=target_revision,
        semantics_id=source_entry.semantics_id,
        dependencies=_finite_element_target_dependencies(context, source_entry, staged),
    )
    staged[source_entry.entry_id] = target_state_entry
    transports: list[CompositionTransport] = []
    field_transfers = _active_finite_element_field_transfers(context)
    # Field sidecars bind the accepted state they belong to: its structure and
    # accepted ID (the source-composition contract every later transition checks).
    for source_field, transfer in zip(
        context.field_source_entries, field_transfers, strict=True
    ):
        target_field = CompositionEntry(
            transfer.target_values,
            entry_id=source_field.entry_id,
            role=source_field.role,
            owner_id=source_field.owner_id,
            structure_id=target_state_entry.structure_id,
            revision_id=target_state.accepted_id,
            semantics_id=source_field.semantics_id,
            dependencies=_finite_element_target_dependencies(
                context, source_field, staged
            ),
        )
        transports.append(
            _finite_element_keyed_transport(
                transfer,
                source_field,
                target_field,
                target_revision_id=target_state.accepted_id,
            )
        )
        staged[source_field.entry_id] = target_field
    context.target_state_entry = target_state_entry
    context.staged_entries = staged
    context.transports = transports


def _stage_finite_element_aggregate_transport(
    context: _FiniteElementRepartitionContext, /
) -> None:
    state = context.source_state
    target_state = context.target_state
    target_manifest = context.target_manifest
    state_entry = context.state_entry
    target_state_entry = context.target_state_entry
    if (
        target_state is None
        or target_manifest is None
        or state_entry is None
        or target_state_entry is None
    ):
        raise ValueError("FE repartition state transport is incomplete.")
    field_transfers = _active_finite_element_field_transfers(context)
    payload_transfers = _active_finite_element_payload_transfers(context)
    root_kind = (
        "finite-element-owner-local-topology-state-root"
        if context.topology_change
        else "finite-element-repartition-state-root"
    )
    source_root = canonical_fingerprint(
        {
            "kind": root_kind,
            "fields": tuple(
                (layout.layout_id, transfer.source_content_root)
                for layout, transfer in zip(
                    state.field_layouts, field_transfers, strict=True
                )
            ),
            "materials": context.material_root,
            "payloads": tuple(
                (name, source_root) for name, source_root, _ in context.payload_roots
            ),
            "time": array_tree_fingerprint(state.time),
            "step": state.step,
            "schedule_cursor": state.schedule_cursor,
            "state_version": state.state_version,
        }
    )
    target_root = canonical_fingerprint(
        {
            "kind": root_kind,
            "fields": tuple(
                (layout.layout_id, transfer.target_content_root)
                for layout, transfer in zip(
                    state.field_layouts, field_transfers, strict=True
                )
            ),
            "materials": context.material_root,
            "payloads": tuple(
                (name, target_root) for name, _, target_root in context.payload_roots
            ),
            "time": array_tree_fingerprint(target_state.time),
            "step": target_state.step,
            "schedule_cursor": target_state.schedule_cursor,
            "state_version": target_state.state_version,
        }
    )
    route_id = canonical_fingerprint(
        {
            "kind": (
                "finite-element-owner-local-topology-rebind"
                if context.topology_change
                else "finite-element-accepted-state-owner-repartition"
            ),
            "migration": context.migration.evidence_id,
            "adaptive_result": context.adaptive_result_id,
            "source_manifest": context.source_manifest.manifest_id,
            "target_manifest": target_manifest.manifest_id,
            "fields": [transfer.transfer_id for transfer in field_transfers],
            "payloads": [transfer.transfer_id for transfer in payload_transfers],
            "source_owner_maps": [
                transfer.source_owner_map_id
                for transfer in (*field_transfers, *payload_transfers)
            ],
            "target_owner_maps": [
                transfer.target_owner_map_id
                for transfer in (*field_transfers, *payload_transfers)
            ],
            "source_root": source_root,
            "target_root": target_root,
        }
    )
    # A topology change remaps the accepted state onto new entities: content is
    # not row-preserved, so the per-field evidence carries the conservation
    # claims. A fixed-topology repartition moves exact rows and pairs its roots.
    context.transports.append(
        CompositionTransport(
            "physical-remap",
            (state_entry.entry_id,),
            (target_state_entry,),
            source_structure_ids=(state_entry.structure_id,),
            route_id=route_id,
            successful=jnp.asarray(True),
        )
        if context.topology_change
        else CompositionTransport(
            "ownership-migration",
            (state_entry.entry_id,),
            (target_state_entry,),
            source_structure_ids=(state_entry.structure_id,),
            route_id=route_id,
            successful=jnp.asarray(True),
            source_content_root=source_root,
            target_content_root=target_root,
        )
    )


def _stage_finite_element_material_transport(
    context: _FiniteElementRepartitionContext, /
) -> None:
    source = context.material_entry
    materials = context.target_materials
    if source is None or materials is None:
        return
    target = CompositionEntry(
        materials,
        entry_id=source.entry_id,
        role=source.role,
        owner_id=source.owner_id,
        structure_id=materials.layout_id,
        revision_id=materials.transaction_id,
        semantics_id=source.semantics_id,
        dependencies=_finite_element_target_dependencies(
            context, source, context.staged_entries
        ),
    )
    plan_id = (
        None
        if context.source_material_plan is None
        else context.source_material_plan.layout_id
    )
    context.transports.append(
        CompositionTransport(
            "ownership-migration",
            (source.entry_id,),
            (target,),
            source_structure_ids=(source.structure_id,),
            route_id=canonical_fingerprint(
                {
                    "kind": "finite-element-material-state-rebind",
                    "migration": context.migration.evidence_id,
                    "material_plan": plan_id,
                    "material_root": context.material_root,
                }
            ),
            successful=jnp.asarray(True),
            source_content_root=context.material_root,
            target_content_root=context.material_root,
        )
    )
    context.staged_entries[source.entry_id] = target


def _stage_finite_element_payload_transports(
    context: _FiniteElementRepartitionContext, /
) -> None:
    target_manifest = context.target_manifest
    if target_manifest is None:
        raise ValueError("FE repartition target manifest is missing.")
    for (
        label,
        name,
        source,
        target_layout,
        target_value,
        transfer,
        value_id,
    ) in context.payload_records:
        target = CompositionEntry(
            target_value,
            entry_id=source.entry_id,
            role=source.role,
            owner_id=source.owner_id,
            structure_id=target_layout.layout_id,
            revision_id=target_manifest.manifest_id,
            semantics_id=source.semantics_id,
            dependencies=_finite_element_target_dependencies(
                context, source, context.staged_entries
            ),
        )
        if transfer is not None:
            transport = _finite_element_keyed_transport(
                transfer, source, target, target_revision_id=target_manifest.manifest_id
            )
        else:
            transport = CompositionTransport(
                "ownership-migration",
                (source.entry_id,),
                (target,),
                source_structure_ids=(source.structure_id,),
                route_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-replicated-payload",
                        "migration": context.migration.evidence_id,
                        "bank": f"{label}:{name}",
                        "layout": target_layout.layout_id,
                        "value": value_id,
                    }
                ),
                successful=jnp.asarray(True),
                source_content_root=value_id,
                target_content_root=value_id,
            )
        context.transports.append(transport)
        context.staged_entries[source.entry_id] = target


def _prepare_finite_element_composition_rebind(
    context: _FiniteElementRepartitionContext, /
) -> None:
    _stage_finite_element_state_transport(context)
    _stage_finite_element_aggregate_transport(context)
    _stage_finite_element_material_transport(context)
    _stage_finite_element_payload_transports(context)
    context.rebind = CompositionRebind(
        context.source_composition,
        retain=context.retained,
        refresh=(),
        reprepare=context.reprepared,
        transports=context.transports,
        invalidate=context.invalidated,
    )


def _commit_finite_element_composition_rebind(
    context: _FiniteElementRepartitionContext, /
) -> CompositionRebindReceipt | str:
    rebind = context.rebind
    if rebind is None:
        return "FE composition rebind produced no candidate."
    receipt = commit_composition_rebind(rebind, accepted_boundary=True)
    rows = _owner_group_allgather(
        np.asarray([bool(receipt.published)], dtype=np.bool_),
        context.execution_group,
        f"{context.axis_name}_commit",
    )
    if not bool(np.all(rows)):
        return "FE composition rebind was refused on an owner."
    return receipt


class FiniteElementTopologyTransaction(StrictModule, NonTrainableState):
    """Transfer FE state through accepted mesh and hp transitions.

    Per-field callbacks receive ``(field, layout, context, args)``; context is a
    mesh, hp, or owner-local adaptive result, or ``None`` for an empty owner.
    The accepted field layout selects the callback semantics. Compatible form
    transfers retain their exact reference/orientation contract.
    """

    certify: Callable
    material_transfer: Callable | None
    material_field_transfer: FiniteElementFieldTransferAction | None
    eulerian_field_transfer: FiniteElementFieldTransferAction | None
    nested_field_transfer: FiniteElementFieldTransferAction | None
    nonconservative_field_transfer: FiniteElementFieldTransferAction | None
    history_transfer: Callable | None
    integration_policy: CurvedOverlapIntegrationPolicy | None
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        certify: Callable,
        /,
        *,
        material_transfer: Callable | None = None,
        material_field_transfer: FiniteElementFieldTransferAction | None = None,
        eulerian_field_transfer: FiniteElementFieldTransferAction | None = None,
        nested_field_transfer: FiniteElementFieldTransferAction | None = None,
        nonconservative_field_transfer: FiniteElementFieldTransferAction | None = None,
        history_transfer: Callable | None = None,
        integration_policy: CurvedOverlapIntegrationPolicy | None = None,
        transaction_id: str = "finite-element-topology-transaction",
    ) -> None:
        if not callable(certify):
            raise TypeError("certify must be callable.")
        if material_transfer is not None and not callable(material_transfer):
            raise TypeError("material_transfer must be callable or None.")
        for name, action in (
            ("material_field_transfer", material_field_transfer),
            ("eulerian_field_transfer", eulerian_field_transfer),
            ("nested_field_transfer", nested_field_transfer),
            ("nonconservative_field_transfer", nonconservative_field_transfer),
        ):
            if action is not None and not callable(action):
                raise TypeError(f"{name} must be callable or None.")
        if history_transfer is not None and not callable(history_transfer):
            raise TypeError("history_transfer must be callable or None.")
        if integration_policy is not None and not isinstance(
            integration_policy, CurvedOverlapIntegrationPolicy
        ):
            raise TypeError(
                "integration_policy must be CurvedOverlapIntegrationPolicy or None."
            )
        identifier = str(transaction_id)
        if not identifier:
            raise ValueError("transaction_id must be non-empty.")
        self.certify = certify
        self.material_transfer = material_transfer
        self.material_field_transfer = material_field_transfer
        self.eulerian_field_transfer = eulerian_field_transfer
        self.nested_field_transfer = nested_field_transfer
        self.nonconservative_field_transfer = nonconservative_field_transfer
        self.history_transfer = history_transfer
        self.integration_policy = integration_policy
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-topology-transaction",
                "declared_id": identifier,
                "has_material_transfer": material_transfer is not None,
                "field_actions": (
                    material_field_transfer is not None,
                    eulerian_field_transfer is not None,
                    nested_field_transfer is not None,
                    nonconservative_field_transfer is not None,
                ),
                "has_history_transfer": history_transfer is not None,
                "integration_policy": (
                    None if integration_policy is None else integration_policy.policy_id
                ),
            }
        )

    def _field_transfer_action(
        self, semantics: FiniteElementFieldTransferSemantics, /
    ) -> FiniteElementFieldTransferAction | None:
        match semantics:
            case FiniteElementFieldTransferSemantics.MATERIAL_FOLLOWING_CONTENT:
                return self.material_field_transfer
            case FiniteElementFieldTransferSemantics.EULERIAN_PHYSICAL_OVERLAP:
                return self.eulerian_field_transfer
            case FiniteElementFieldTransferSemantics.NESTED_FE:
                return self.nested_field_transfer
            case FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION:
                return self.nonconservative_field_transfer
            case FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE:
                return None
            case other:
                assert_never(other)

    def _retained(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        diagnostics: str,
        /,
        *,
        periodic_transfers: Sequence[PeriodicFiniteElementTransfer] = (),
        field_transfers: Sequence[FiniteElementTransferResult] = (),
        conservative_transfers: Sequence[FiniteElementConservativeContentEvidence] = (),
    ) -> FiniteElementTopologyResult:
        return FiniteElementTopologyResult(
            state=accepted,
            mesh=mesh,
            adaptation=None,
            committed=jnp.asarray(False),
            diagnostics=diagnostics,
            periodic_transfers=tuple(periodic_transfers),
            field_transfers=tuple(field_transfers),
            conservative_transfers=tuple(conservative_transfers),
        )

    @checked
    def repartition(
        self,
        source_manifest: FiniteElementRestartManifest,
        source_compiled: CompiledFiniteElementProblem,
        target_compiled: CompiledFiniteElementProblem,
        source_identity_plans: Sequence[FiniteElementDofIdentityPlan],
        target_identity_plans: Sequence[FiniteElementDofIdentityPlan],
        migration_evidence: OwnerLocalGraphMigrationEvidence,
        source_composition: Composition,
        execution_group: ExecutionGroup,
        source_geometry_view: OwnerLocalCellGeometry,
        target_geometry_view: OwnerLocalCellGeometry,
        /,
        *,
        state_entry_id: str,
        geometry_entry_id: str,
        field_entry_ids: Sequence[tuple[str, str]],
        material_entry_id: str | None,
        auxiliary_entry_ids: Sequence[tuple[str, str]] = (),
        integrator_entry_ids: Sequence[tuple[str, str]] = (),
        target_auxiliary_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
        target_integrator_layouts: Sequence[FiniteElementRestartStateLayout]
        | None = None,
        target_material_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
        source_material_plan: MaterialIntegrationPlan | None = None,
        target_material_plan: MaterialIntegrationPlan | None = None,
        retain_entry_ids: Sequence[str] = (),
        invalidate_entry_ids: Sequence[str] = (),
        reprepare_entries: Sequence[CompositionEntry] = (),
        axis_name: str = "finite_element_repartition",
    ) -> FiniteElementRepartitionResult:
        """Commit a pure owner-local FE state migration and lifecycle rebind."""
        import jax

        if not isinstance(
            source_compiled, CompiledFiniteElementProblem
        ) or not isinstance(target_compiled, CompiledFiniteElementProblem):
            raise TypeError("Source and target authority must be compiled FE problems.")
        parts = len(execution_group.spec.process_indices)
        try:
            rank = execution_group.spec.process_indices.index(jax.process_index())
        except ValueError as error:
            raise ValueError(
                "FE repartition rank is outside the execution group."
            ) from error
        context = _FiniteElementRepartitionContext(
            source_manifest=source_manifest,
            source_compiled=source_compiled,
            target_compiled=target_compiled,
            source_plans=tuple(source_identity_plans),
            target_plans=tuple(target_identity_plans),
            migration=migration_evidence,
            source_composition=source_composition,
            execution_group=execution_group,
            source_geometry_view=source_geometry_view,
            target_geometry_view=target_geometry_view,
            state_entry_id=state_entry_id,
            geometry_entry_id=geometry_entry_id,
            field_ids=tuple(field_entry_ids),
            material_entry_id=material_entry_id,
            auxiliary_ids=tuple(auxiliary_entry_ids),
            integrator_ids=tuple(integrator_entry_ids),
            target_auxiliary_layouts=(
                None
                if target_auxiliary_layouts is None
                else tuple(target_auxiliary_layouts)
            ),
            target_integrator_layouts=(
                None
                if target_integrator_layouts is None
                else tuple(target_integrator_layouts)
            ),
            target_material_layouts=(
                None
                if target_material_layouts is None
                else tuple(target_material_layouts)
            ),
            source_material_plan=source_material_plan,
            target_material_plan=target_material_plan,
            retained=tuple(retain_entry_ids),
            invalidated=tuple(invalidate_entry_ids),
            reprepared=tuple(reprepare_entries),
            rank=rank,
            parts=parts,
            axis_name=axis_name,
        )

        def refused(diagnostics: str) -> FiniteElementRepartitionResult:
            return _refused_finite_element_repartition(context, diagnostics)

        error = _admit_finite_element_repartition(context)
        if error is not None:
            return refused(error)
        error = _migrate_finite_element_fields(context)
        if error is not None:
            return refused(error)
        error = _validate_finite_element_target_consumer(context)
        if error is not None:
            return refused(error)
        error = _migrate_finite_element_restart_banks(context)
        if error is not None:
            return refused(error)
        error = _qualify_finite_element_material_state(context)
        if error is not None:
            return refused(error)
        error = _build_finite_element_repartition_target_state(context)
        if error is not None:
            return refused(error)
        error = _validate_finite_element_repartition_collectively(
            context,
            "composition_rebind",
            lambda: _prepare_finite_element_composition_rebind(context),
        )
        if error is not None:
            return refused(error)
        receipt = _commit_finite_element_composition_rebind(context)
        if isinstance(receipt, str):
            return refused(receipt)
        if context.target_state is None or context.target_manifest is None:
            return refused("FE repartition target state was not constructed.")
        return _finite_element_repartition_result(
            context,
            context.target_state,
            context.target_manifest,
            True,
            "committed",
            receipt,
        )

    @checked
    def adapt_owner_local(
        self,
        source_manifest: FiniteElementRestartManifest,
        source_compiled: CompiledFiniteElementProblem,
        target_compiled: CompiledFiniteElementProblem,
        source_identity_plans: Sequence[FiniteElementDofIdentityPlan],
        target_identity_plans: Sequence[FiniteElementDofIdentityPlan],
        adaptive_result: object | None,
        source_composition: Composition,
        execution_group: ExecutionGroup,
        source_geometry_view: OwnerLocalCellGeometry,
        target_geometry_view: OwnerLocalCellGeometry,
        /,
        *,
        state_entry_id: str,
        geometry_entry_id: str,
        field_entry_ids: Sequence[tuple[str, str]],
        material_entry_id: str | None,
        auxiliary_entry_ids: Sequence[tuple[str, str]] = (),
        integrator_entry_ids: Sequence[tuple[str, str]] = (),
        target_auxiliary_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
        target_integrator_layouts: Sequence[FiniteElementRestartStateLayout]
        | None = None,
        target_material_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
        source_material_plan: MaterialIntegrationPlan | None = None,
        target_material_plan: MaterialIntegrationPlan | None = None,
        retain_entry_ids: Sequence[str] = (),
        invalidate_entry_ids: Sequence[str] = (),
        reprepare_entries: Sequence[CompositionEntry] = (),
        axis_name: str = "finite_element_owner_local_topology",
    ) -> FiniteElementOwnerLocalTopologyResult:
        """Atomically transfer accepted owner-local FE state through an adaptive topology."""
        import jax

        members = execution_group.spec.process_indices
        try:
            rank = members.index(jax.process_index())
        except ValueError as error:
            raise ValueError(
                "FE topology adaptation rank is outside the execution group."
            ) from error
        local_error = None
        migration: (
            OwnerLocalGraphMigrationEvidence | OwnerLocalTopologyTransferRoute | None
        ) = None
        source_plans: tuple[FiniteElementDofIdentityPlan, ...] = ()
        target_plans: tuple[FiniteElementDofIdentityPlan, ...] = ()
        field_ids: tuple[tuple[str, str], ...] = ()
        auxiliary_ids: tuple[tuple[str, str], ...] = ()
        integrator_ids: tuple[tuple[str, str], ...] = ()
        retained_ids: tuple[str, ...] = ()
        invalidated_ids: tuple[str, ...] = ()
        target_reprepared: tuple[CompositionEntry, ...] = ()
        target_auxiliary: tuple[FiniteElementRestartStateLayout, ...] | None = None
        target_integrator: tuple[FiniteElementRestartStateLayout, ...] | None = None
        target_material: tuple[FiniteElementRestartStateLayout, ...] | None = None
        try:
            if (
                not isinstance(source_manifest, FiniteElementRestartManifest)
                or not isinstance(source_compiled, CompiledFiniteElementProblem)
                or not isinstance(target_compiled, CompiledFiniteElementProblem)
                or not isinstance(
                    source_compiled.discretization, FiniteElementDiscretization
                )
                or not isinstance(
                    target_compiled.discretization, FiniteElementDiscretization
                )
                or not isinstance(source_composition, Composition)
                or not isinstance(source_geometry_view, OwnerLocalCellGeometry)
                or not isinstance(target_geometry_view, OwnerLocalCellGeometry)
                or not source_manifest.state.field_ownership
                or not isinstance(state_entry_id, str)
                or not state_entry_id.strip()
                or not isinstance(geometry_entry_id, str)
                or not geometry_entry_id.strip()
                or not isinstance(material_entry_id, (str, type(None)))
                or not isinstance(axis_name, str)
                or not axis_name.strip()
            ):
                raise TypeError(
                    "Owner-local FE topology transaction inputs are malformed."
                )
            migration = target_geometry_view.graph_migration_evidence
            if migration is None:
                migration = OwnerLocalTopologyTransferRoute(
                    source_geometry_view.distribution_id,
                    target_geometry_view.distribution_id,
                    execution_group.spec.group_id,
                    rank,
                    len(members),
                )
            elif not isinstance(migration, OwnerLocalGraphMigrationEvidence):
                raise TypeError(
                    "Target FE topology view has invalid graph migration evidence."
                )
            source_plans = tuple(source_identity_plans)
            target_plans = tuple(target_identity_plans)
            field_ids = tuple(field_entry_ids)
            auxiliary_ids = tuple(auxiliary_entry_ids)
            integrator_ids = tuple(integrator_entry_ids)
            retained_ids = tuple(retain_entry_ids)
            invalidated_ids = tuple(invalidate_entry_ids)
            target_reprepared = tuple(reprepare_entries)
            target_auxiliary = (
                None
                if target_auxiliary_layouts is None
                else tuple(target_auxiliary_layouts)
            )
            target_integrator = (
                None
                if target_integrator_layouts is None
                else tuple(target_integrator_layouts)
            )
            target_material = (
                None
                if target_material_layouts is None
                else tuple(target_material_layouts)
            )
            if (
                len(source_plans) != len(source_manifest.state.fields)
                or len(target_plans) != len(source_manifest.state.fields)
                or len(field_ids) != len(source_manifest.state.fields)
                or any(
                    not isinstance(entry, CompositionEntry) for entry in target_reprepared
                )
            ):
                raise ValueError(
                    "Owner-local FE source/target plan counts are inconsistent."
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            "finite_element_owner_local_topology_inputs",
            "owner-local FE topology transaction",
        )
        if migration is None:
            raise RuntimeError(
                "FE topology input collective accepted without migration evidence."
            )
        context = _FiniteElementRepartitionContext(
            source_manifest=source_manifest,
            source_compiled=source_compiled,
            target_compiled=target_compiled,
            source_plans=source_plans,
            target_plans=target_plans,
            migration=migration,
            source_composition=source_composition,
            execution_group=execution_group,
            source_geometry_view=source_geometry_view,
            target_geometry_view=target_geometry_view,
            state_entry_id=state_entry_id,
            geometry_entry_id=geometry_entry_id,
            field_ids=field_ids,
            material_entry_id=material_entry_id,
            auxiliary_ids=auxiliary_ids,
            integrator_ids=integrator_ids,
            target_auxiliary_layouts=target_auxiliary,
            target_integrator_layouts=target_integrator,
            target_material_layouts=target_material,
            source_material_plan=source_material_plan,
            target_material_plan=target_material_plan,
            retained=retained_ids,
            invalidated=invalidated_ids,
            reprepared=target_reprepared,
            rank=rank,
            parts=len(members),
            axis_name=axis_name,
            topology_change=True,
            adaptive_result=adaptive_result,
        )

        def refused(diagnostics: str) -> FiniteElementOwnerLocalTopologyResult:
            return FiniteElementOwnerLocalTopologyResult(
                context.source_state,
                context.source_manifest,
                jnp.asarray(False, dtype=jnp.bool_),
                diagnostics,
                None,
                tuple(context.topology_field_transfers),
                tuple(context.topology_payload_transfers),
                context.target_plans,
                context.adaptive_result_id,
                None
                if context.vertex_stencil is None
                else context.vertex_stencil.stencil_id,
            )

        error = _admit_finite_element_owner_local_topology(context)
        if error is not None:
            return refused(error)
        error = _migrate_finite_element_owner_local_adaptive_fields(self, context)
        if error is not None:
            return refused(error)
        error = _validate_finite_element_owner_local_target_consumer(context)
        if error is not None:
            return refused(error)
        error = _migrate_finite_element_restart_banks(context, transaction=self)
        if error is not None:
            return refused(error)
        error = _qualify_finite_element_material_state(context)
        if error is not None:
            return refused(error)
        metadata = _finite_element_owner_local_adaptive_metadata(context.adaptive_result)

        def certify_candidate() -> None:
            certified = jnp.asarray(
                self.certify(
                    context.target_mesh,
                    tuple(context.target_local_fields),
                    context.target_materials,
                    metadata.lineage,
                    context,
                )
            )
            if certified.shape != () or certified.dtype != jnp.bool_:
                raise ValueError("FE topology certification must return a scalar bool.")
            if not bool(np.asarray(certified)):
                raise ValueError("FE topology candidate certification refused.")

        error = _validate_finite_element_repartition_collectively(
            context, "candidate_certification", certify_candidate
        )
        if error is not None:
            return refused(error)
        error = _build_finite_element_repartition_target_state(context)
        if error is not None:
            return refused(error)
        error = _validate_finite_element_repartition_collectively(
            context,
            "composition_rebind",
            lambda: _prepare_finite_element_composition_rebind(context),
        )
        if error is not None:
            return refused(error)
        receipt = _commit_finite_element_composition_rebind(context)
        if isinstance(receipt, str):
            return refused(receipt)
        if context.target_state is None or context.target_manifest is None:
            return refused("Owner-local FE topology target state was not constructed.")
        return FiniteElementOwnerLocalTopologyResult(
            context.target_state,
            context.target_manifest,
            jnp.asarray(True, dtype=jnp.bool_),
            "committed",
            receipt,
            tuple(context.topology_field_transfers),
            tuple(context.topology_payload_transfers),
            context.target_plans,
            context.adaptive_result_id,
            None if context.vertex_stencil is None else context.vertex_stencil.stencil_id,
            None if metadata.lineage is None else metadata.lineage.lineage_id,
        )

    @checked
    def execute(
        self,
        accepted: FiniteElementAcceptedState,
        mesh: CellMesh,
        adaptation: MeshAdaptationResult,
        args: object = None,
        /,
    ) -> FiniteElementTopologyResult:
        if (
            mesh.topology_id != accepted.topology_id
            or adaptation.source.mesh.mesh_id != mesh.mesh_id
        ):
            raise ValueError("Accepted state and mesh adaptation disagree.")
        transition = adaptation.transition
        if transition is None:
            return self._retained(accepted, mesh, "adaptation-unchanged")
        if accepted.field_ownership is not None:
            return self._retained(
                accepted,
                mesh,
                "owner-local-accepted-fields-require-pure-repartition",
            )
        candidate_mesh = adaptation.target.mesh
        reference_maps = _reference_maps_for_adaptation(adaptation)
        geometry_transition = transition.geometry_transition
        complete_vertex_ancestry = _has_complete_vertex_ancestry(
            adaptation, mesh, candidate_mesh
        )
        unparented_target_cells = _has_unparented_target_cells(adaptation, candidate_mesh)
        if reference_maps is None and mesh.mesh_id != candidate_mesh.mesh_id:
            geometry_transition = None
        candidate_fields: list[Array | None] = [None] * len(accepted.fields)
        periodic_transfers: list[PeriodicFiniteElementTransfer] = []
        field_transfers: list[FiniteElementTransferResult] = []
        conservative_transfers: list[FiniteElementConservativeContentEvidence] = []
        transfer_cache: dict[
            tuple[int, int, int, str, str, bool],
            tuple[FiniteElementTransferResult, PeriodicFiniteElementTransfer | None],
        ] = {}
        explicit_indices: list[int] = []
        source_periodic = mesh.periodic_topology is not None
        target_periodic = candidate_mesh.periodic_topology is not None
        vertex_transfer = adaptation.transfer
        for index, (field, layout) in enumerate(
            zip(accepted.fields, accepted.field_layouts, strict=True)
        ):
            semantics = layout.transfer_semantics
            form_identity = layout.form_identity
            match semantics:
                case FiniteElementFieldTransferSemantics.MATERIAL_FOLLOWING_CONTENT:
                    if transition.volume_material_chain is None:
                        return self._retained(
                            accepted,
                            mesh,
                            "material-following-field-requires-validated-material-chain",
                            periodic_transfers=periodic_transfers,
                            field_transfers=field_transfers,
                        )
                    if self.material_field_transfer is None:
                        return self._retained(
                            accepted,
                            mesh,
                            "material-following-field-transfer-policy-required",
                            periodic_transfers=periodic_transfers,
                            field_transfers=field_transfers,
                        )
                    explicit_indices.append(index)
                    continue
                case FiniteElementFieldTransferSemantics.EULERIAN_PHYSICAL_OVERLAP:
                    if self.eulerian_field_transfer is None:
                        return self._retained(
                            accepted,
                            mesh,
                            "eulerian-physical-overlap-transfer-policy-required",
                            periodic_transfers=periodic_transfers,
                            field_transfers=field_transfers,
                        )
                    explicit_indices.append(index)
                    continue
                case FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE:
                    try:
                        transferred, evidence = _conservative_cell_average_transfer(
                            layout, field, adaptation
                        )
                    except (TypeError, ValueError, NotImplementedError) as error:
                        return self._retained(
                            accepted,
                            mesh,
                            f"conservative-cell-average-transfer-refused: {error}",
                            periodic_transfers=periodic_transfers,
                            field_transfers=field_transfers,
                            conservative_transfers=conservative_transfers,
                        )
                    candidate_fields[index] = transferred
                    conservative_transfers.append(evidence)
                    continue
                case FiniteElementFieldTransferSemantics.NESTED_FE:
                    if form_identity is None:
                        if self.nested_field_transfer is None:
                            return self._retained(
                                accepted,
                                mesh,
                                "nested-fe-field-transfer-policy-required",
                                periodic_transfers=periodic_transfers,
                                field_transfers=field_transfers,
                            )
                        explicit_indices.append(index)
                        continue
                case FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION:
                    pass
                case other:
                    assert_never(other)
            if layout.storage_kind == "hp_cell_slots":
                explicit_indices.append(index)
                continue
            if semantics is FiniteElementFieldTransferSemantics.NESTED_FE:
                if form_identity is None:
                    return self._retained(
                        accepted,
                        mesh,
                        "nested-fe-form-identity-unresolved",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                if layout.periodic_quotient and not (source_periodic and target_periodic):
                    return self._retained(
                        accepted,
                        mesh,
                        "periodic-quotient-layout-requires-periodic-endpoints",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                dimension, form_degree, order, family, twist = form_identity
                key = (
                    dimension,
                    form_degree,
                    order,
                    family,
                    twist,
                    layout.periodic_quotient,
                )
                cached = transfer_cache.get(key)
                if cached is None:
                    try:
                        if (
                            reference_maps is None
                            and geometry_transition is None
                            and mesh.mesh_id != candidate_mesh.mesh_id
                        ):
                            raise ValueError(
                                "Accepted de Rham h-transfer requires exact reference maps."
                            )
                        cached = _finite_element_transfer(
                            layout,
                            mesh,
                            candidate_mesh,
                            reference_maps,
                            geometry_transition,
                            self.integration_policy,
                        )
                        transfer_cache[key] = cached
                    except (TypeError, ValueError, NotImplementedError) as error:
                        if layout.periodic_quotient:
                            return self._retained(
                                accepted,
                                mesh,
                                f"periodic-quotient-transfer-refused: {error}",
                                periodic_transfers=periodic_transfers,
                                field_transfers=field_transfers,
                            )
                        return self._retained(
                            accepted,
                            mesh,
                            f"nested-fe-transfer-refused: {error}",
                            periodic_transfers=periodic_transfers,
                            field_transfers=field_transfers,
                        )
                transfer_result, quotient_transfer = cached
                if all(
                    item.transfer_id != transfer_result.transfer_id
                    for item in field_transfers
                ):
                    field_transfers.append(transfer_result)
                try:
                    candidate_fields[index] = _apply_form_operator(
                        transfer_result.complex_map.map(form_degree), field
                    )
                except (TypeError, ValueError, NotImplementedError) as error:
                    return self._retained(
                        accepted,
                        mesh,
                        f"declared-de-rham-layout-transfer-refused: {error}",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                if quotient_transfer is not None and all(
                    item.transfer_id != quotient_transfer.transfer_id
                    for item in periodic_transfers
                ):
                    periodic_transfers.append(quotient_transfer)
                continue
            if (
                _vertex_p1_field_layout(layout, mesh)
                and _vertex_p1_field_layout(layout, candidate_mesh)
                and vertex_transfer is not None
            ):
                if (
                    not complete_vertex_ancestry
                    or unparented_target_cells
                    or not vertex_transfer.preserves_constants
                    or not vertex_transfer.preserves_linear
                ):
                    explicit_indices.append(index)
                    continue
                if field.shape[0] != vertex_transfer.source_size:
                    return self._retained(
                        accepted,
                        mesh,
                        "declared-vertex-p1-layout-does-not-match-source-transfer",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                candidate_fields[index] = vertex_transfer.apply(field)
                continue
            explicit_indices.append(index)

        if explicit_indices:
            for index in explicit_indices:
                layout = accepted.field_layouts[index]
                action = self._field_transfer_action(layout.transfer_semantics)
                if action is None:
                    return self._retained(
                        accepted,
                        mesh,
                        "declared-field-transfer-action-required",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                try:
                    transferred = jnp.asarray(
                        action(accepted.fields[index], layout, adaptation, args)
                    )
                except (TypeError, ValueError, NotImplementedError) as error:
                    return self._retained(
                        accepted,
                        mesh,
                        f"declared-field-transfer-refused: {error}",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                if (
                    transferred.ndim == 0
                    or transferred.shape[1:] != accepted.fields[index].shape[1:]
                ):
                    return self._retained(
                        accepted,
                        mesh,
                        "declared-field-transfer-changed-payload-shape",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                candidate_fields[index] = transferred
            canonical_indices = [
                index
                for index in explicit_indices
                if accepted.field_layouts[index].storage_kind == "canonical_dofs"
            ]
            if canonical_indices:
                source_layouts = tuple(
                    accepted.field_layouts[index] for index in canonical_indices
                )
                try:
                    source_counts = _canonical_field_dof_counts(
                        mesh, adaptation.source.geometry, source_layouts
                    )
                    target_counts = _canonical_field_dof_counts(
                        candidate_mesh, adaptation.target.geometry, source_layouts
                    )
                except (TypeError, ValueError, NotImplementedError) as error:
                    return self._retained(
                        accepted,
                        mesh,
                        f"explicit-field-layout-unresolved: {error}",
                        periodic_transfers=periodic_transfers,
                        field_transfers=field_transfers,
                    )
                for index, source_count, target_count in zip(
                    canonical_indices, source_counts, target_counts, strict=True
                ):
                    candidate = candidate_fields[index]
                    if (
                        accepted.fields[index].shape[0] != source_count
                        or candidate is None
                        or candidate.shape[0] != target_count
                    ):
                        return self._retained(
                            accepted,
                            mesh,
                            "explicit-field-transfer-does-not-match-declared-layout",
                            periodic_transfers=periodic_transfers,
                            field_transfers=field_transfers,
                        )

        if any(value is None for value in candidate_fields):
            return self._retained(
                accepted,
                mesh,
                "field-transfer-left-an-unmapped-field",
                periodic_transfers=periodic_transfers,
                field_transfers=field_transfers,
            )
        completed_fields = tuple(value for value in candidate_fields if value is not None)
        candidate_materials: MaterialTransaction | None
        if accepted.materials is None:
            candidate_materials = None
        elif self.material_transfer is None:
            return self._retained(
                accepted,
                mesh,
                "material-transfer-policy-required",
                periodic_transfers=periodic_transfers,
                field_transfers=field_transfers,
            )
        else:
            transferred = self.material_transfer(
                accepted.materials,
                transition.lineage,
                args,
            )
            if not isinstance(transferred, MaterialTransaction):
                return self._retained(
                    accepted,
                    mesh,
                    "material-transfer-rejected",
                    periodic_transfers=periodic_transfers,
                    field_transfers=field_transfers,
                )
            candidate_materials = transferred
        certified = self.certify(
            candidate_mesh,
            completed_fields,
            candidate_materials,
            transition.lineage,
            args,
        )
        if not bool(jnp.asarray(certified)):
            return self._retained(
                accepted,
                mesh,
                "candidate-certification-rejected",
                periodic_transfers=periodic_transfers,
                field_transfers=field_transfers,
                conservative_transfers=conservative_transfers,
            )
        promoted = FiniteElementAcceptedState(
            completed_fields,
            accepted.time,
            accepted.step,
            candidate_mesh.topology_id,
            f"{accepted.prepared_id}:transition:{transition.transition_id}",
            f"{accepted.compilation_id}:transition:{transition.transition_id}",
            field_layouts=accepted.field_layouts,
            materials=candidate_materials,
            schedule_cursor=accepted.schedule_cursor,
            state_version=accepted.state_version + 1,
        )
        return FiniteElementTopologyResult(
            state=promoted,
            mesh=candidate_mesh,
            adaptation=adaptation,
            committed=jnp.asarray(True),
            diagnostics="committed",
            periodic_transfers=tuple(periodic_transfers),
            field_transfers=tuple(field_transfers),
            conservative_transfers=tuple(conservative_transfers),
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
        """Transfer, certify, and atomically promote one prepared hp candidate."""

        if accepted.topology_id != transaction.accepted.topology.topology_id:
            raise ValueError("Accepted state and hp transaction topology disagree.")
        transfers = transaction.p_transfers + transaction.h_transfers

        def retained(diagnostics: str, /) -> FiniteElementHPTopologyResult:
            return FiniteElementHPTopologyResult(
                accepted,
                transaction.accepted,
                None,
                tuple((str(name), jnp.asarray(value)) for name, value in auxiliary_state),
                tuple(
                    (str(name), jnp.asarray(value)) for name, value in integrator_state
                ),
                jnp.asarray(False),
                diagnostics,
            )

        if accepted.field_ownership is not None:
            return retained("owner-local-accepted-fields-require-pure-repartition")

        for layout in accepted.field_layouts:
            match layout.transfer_semantics:
                case FiniteElementFieldTransferSemantics.MATERIAL_FOLLOWING_CONTENT:
                    return retained(
                        "material-following-hp-transfer-requires-validated-material-chain"
                    )
                case FiniteElementFieldTransferSemantics.EULERIAN_PHYSICAL_OVERLAP:
                    if self.eulerian_field_transfer is None:
                        return retained(
                            "eulerian-physical-overlap-transfer-policy-required"
                        )
                case FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION:
                    if self.nonconservative_field_transfer is None:
                        return retained(
                            "nonconservative-interpolation-transfer-policy-required"
                        )
                case FiniteElementFieldTransferSemantics.NESTED_FE:
                    if (
                        layout.storage_kind != "hp_cell_slots"
                        and self.nested_field_transfer is None
                    ):
                        return retained("nested-fe-field-transfer-policy-required")
                case FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE:
                    return retained(
                        "conservative-cell-average-hp-storage-is-not-supported"
                    )
                case other:
                    assert_never(other)
        all_nested_hp = all(
            layout.storage_kind == "hp_cell_slots"
            and layout.transfer_semantics is FiniteElementFieldTransferSemantics.NESTED_FE
            for layout in accepted.field_layouts
        )
        if all_nested_hp:
            if len(transfers) != len(accepted.fields):
                raise ValueError(
                    "Automatic hp state transfer requires one transfer per field."
                )
            if not _hp_field_layouts_match(transaction.accepted, accepted.field_layouts):
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
                    "hp-field-layout-mismatch",
                )
            candidate_fields = tuple(
                transfer.apply_l2_projection(field)
                for transfer, field in zip(transfers, accepted.fields, strict=True)
            )
        else:
            if any(
                layout.storage_kind == "hp_cell_slots"
                for layout in accepted.field_layouts
            ):
                return retained("mixed-hp-cell-slot-transfer-is-not-supported")
            candidate_rows: list[Array] = []
            for field, layout in zip(
                accepted.fields, accepted.field_layouts, strict=True
            ):
                action = self._field_transfer_action(layout.transfer_semantics)
                if action is None:
                    return retained("declared-hp-field-transfer-action-required")
                try:
                    candidate = jnp.asarray(action(field, layout, transaction, args))
                except (TypeError, ValueError, NotImplementedError) as error:
                    return retained(f"declared-hp-field-transfer-refused: {error}")
                if candidate.ndim == 0 or candidate.shape[1:] != field.shape[1:]:
                    return retained("declared-hp-field-transfer-changed-payload-shape")
                candidate_rows.append(candidate)
            candidate_fields = tuple(candidate_rows)
        if len(candidate_fields) != len(accepted.fields):
            return FiniteElementHPTopologyResult(
                accepted,
                transaction.accepted,
                None,
                tuple((str(name), jnp.asarray(value)) for name, value in auxiliary_state),
                tuple(
                    (str(name), jnp.asarray(value)) for name, value in integrator_state
                ),
                jnp.asarray(False),
                "hp-field-transfer-count-mismatch",
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
            field_layouts=accepted.field_layouts,
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
    "FiniteElementConservativeContentEvidence",
    "FiniteElementHPTopologyResult",
    "FiniteElementOwnerLocalFieldTransferEvidence",
    "FiniteElementOwnerLocalTopologyResult",
    "FiniteElementRepartitionResult",
    "read_finite_element_hp_epoch",
    "partition_owner_local_finite_element_graph",
    "FiniteElementTopologyResult",
    "FiniteElementTopologyTransaction",
    "write_finite_element_hp_epoch",
]

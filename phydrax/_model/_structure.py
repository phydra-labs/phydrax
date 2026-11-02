#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import dataclasses
import enum
import json
import math
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike


if TYPE_CHECKING:
    from .._physical import SpatialCoordinateContract
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._exact_algebraic_geometry import RadicalElement
    from ..meshing._association import GeometryAssociation

from .._archive_columns import (
    ExactTensorArchiveRecipe,
    integer_column,
    integer_column_values,
    IntegerRowsArchiveRecipe,
    TextColumnArchiveRecipe,
    TextRowsArchiveRecipe,
)
from .._array_archive import (
    _read_npy_metadata,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
)
from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import mark_strict_initialized, Strict
from .._typing_plan import validate_tree
from ._artifacts import artifact_value, artifact_value_id, register_artifact_value


_PLC_TEXT_COLUMNS = frozenset(
    {"vertex_ids", "facet_ids", "segment_ids", "plc_edge_ids", "region_ids"}
)
_PLC_PATH_COLUMNS = frozenset(
    {
        "vertex_occurrence_paths",
        "facet_occurrence_paths",
        "segment_occurrence_paths",
        "plc_edge_occurrence_paths",
        "region_occurrence_paths",
    }
)
_PLC_INTEGER_COLUMNS = frozenset(
    {
        "vertex_entity_ids",
        "facet_entity_ids",
        "segment_entity_ids",
        "plc_edge_entity_ids",
        "region_entity_ids",
    }
)
_PLC_INTEGER_ROWS = frozenset({"plc_edge_vertices", "plc_edge_facets"})


class PiecewiseLinearComplexArchiveRecipe(NamedTuple):
    """Exact PLC fields, with static identity and incidence banks in columns."""

    fields: dict[str, object]

    def decode(self) -> Any:
        """Decode stored fields; scientific ownership is restored separately."""
        from ..meshing._plc_source import PiecewiseLinearComplex

        expected = {field.name for field in dataclasses.fields(PiecewiseLinearComplex)}
        if set(self.fields) != expected:
            raise ValueError("Stored PLC recipe has an incomplete field inventory.")
        fields = dict(self.fields)
        for name in _PLC_TEXT_COLUMNS:
            column = fields[name]
            if type(column) is not TextColumnArchiveRecipe:
                raise TypeError("PLC identity columns must be exact text columns.")
            fields[name] = column.decode()
        for name in _PLC_PATH_COLUMNS:
            column = fields[name]
            if type(column) is not TextRowsArchiveRecipe:
                raise TypeError("PLC occurrence paths must be exact text rows.")
            fields[name] = column.decode()
        for name in _PLC_INTEGER_COLUMNS:
            fields[name] = integer_column_values(fields[name])
        for name in _PLC_INTEGER_ROWS:
            column = fields[name]
            if type(column) is not IntegerRowsArchiveRecipe:
                raise TypeError("PLC incidence must be exact integer rows.")
            fields[name] = column.decode()
        return _restore_dataclass(PiecewiseLinearComplex, fields)


class CADRevisionArchiveRecipe(NamedTuple):
    revision_id: str
    source_id: str
    provenance_id: str
    occurrences: TextRowsArchiveRecipe

    def restore(self) -> Any:
        from ..geometry._cad_revision import CADOccurrence, CADRevision

        rows = self.occurrences.decode()
        if any(len(row) < 6 or row[4] not in ("-1", "1") for row in rows):
            raise ValueError("Invalid column-encoded CAD occurrence.")
        return CADRevision(
            self.revision_id,
            self.source_id,
            tuple(
                CADOccurrence(
                    self.revision_id,
                    row[0],
                    row[1],
                    row[2],
                    tuple(row[5:]),
                    row[3] or None,
                    int(row[4]),
                )
                for row in rows
            ),
            self.provenance_id,
        )


class CADTransactionArchiveRecipe(NamedTuple):
    transaction_id: str
    source_revision_id: str
    target_revision_id: str
    correspondences: TextRowsArchiveRecipe
    unresolved_source_ids: frozenset[str]
    unresolved_target_ids: frozenset[str]
    coverage: object

    def restore(self) -> Any:
        from ..geometry._cad_revision import (
            AssociationCoverageEvidence,
            OccurrenceCorrespondence,
            OccurrenceCorrespondenceTransaction,
        )

        if not isinstance(self.coverage, AssociationCoverageEvidence):
            raise TypeError("CAD transaction requires coverage evidence.")
        rows = self.correspondences.decode()
        if any(len(row) != 4 or row[3] not in ("", "-1", "0", "1") for row in rows):
            raise ValueError("Invalid column-encoded CAD correspondence.")
        return OccurrenceCorrespondenceTransaction(
            self.transaction_id,
            self.source_revision_id,
            self.target_revision_id,
            tuple(
                OccurrenceCorrespondence(
                    row[0], row[1], row[2], None if not row[3] else int(row[3])
                )
                for row in rows
            ),
            self.unresolved_source_ids,
            self.unresolved_target_ids,
            self.coverage,
        )


class CADGraphArchiveRecipe(NamedTuple):
    source_revision: object
    target_revision: object
    transaction: object
    graph_id: str

    def restore(self) -> Any:
        from ..geometry._cad_revision import (
            AssociationGraph,
            CADRevision,
            OccurrenceCorrespondenceTransaction,
        )

        if (
            not isinstance(self.source_revision, CADRevision)
            or not isinstance(self.target_revision, CADRevision)
            or not isinstance(self.transaction, OccurrenceCorrespondenceTransaction)
        ):
            raise TypeError("CAD graph requires revisions and a transaction.")

        graph = AssociationGraph(
            self.source_revision, self.target_revision, self.transaction
        )
        if graph.graph_id != self.graph_id:
            raise ValueError("Stored CAD graph identity changed.")
        return graph


class PartitionOperandArchiveRecipe(NamedTuple):
    operand_id: str
    source: object
    role: object
    selection: object
    target_region_ids: tuple[str, ...]


class PartitionPlanArchiveRecipe(NamedTuple):
    coordinate_contract: object
    embedding: object
    operands: tuple[PartitionOperandArchiveRecipe, ...]
    precedence: tuple[str, ...]
    plan_id: str
    topological_dimension: int


class AdaptiveImplicitBoundarySourceArchiveRecipe(NamedTuple):
    surface: object
    expected_source_id: str
    expected_source_revision: str
    frame_id: str


class ParametricCurveBoundarySourceArchiveRecipe(NamedTuple):
    curves: tuple[object, ...]
    parameter_ranges: tuple[tuple[float, float], ...]
    source_id: str
    source_revision: str
    curve_entity_ids: tuple[str, ...]
    occurrence_paths: tuple[tuple[str, ...], ...]
    chart_ids: tuple[str, ...]
    expected_query_id: str


class MeshingDomainArchiveRecipe(NamedTuple):
    patches: tuple[object, ...]
    curves: tuple[object, ...]
    corner_count: int
    source_id: str
    source_revision: str
    regions: tuple[object, ...]
    tolerance: float
    accuracy: str
    source_indices: tuple[tuple[int, ...], ...]
    source_occurrences: tuple[tuple[tuple[str, ...], ...], ...]
    region_source_indices: tuple[int, ...]
    region_source_occurrences: tuple[tuple[str, ...], ...]
    source_kinds: tuple[str, ...]
    authority_id: str
    brep_authority: object | None
    corner_points: object
    corner_points_authored: bool
    expected_domain_id: str


class MeshingDomainBoundarySourceArchiveRecipe(NamedTuple):
    domain: MeshingDomainArchiveRecipe
    patches: tuple[int, ...]
    resolution: int
    boundary_occurrences: object | None
    boundary_cell_ids: object | None
    boundary_patch_ids: object | None
    expected_source_scope_id: str


class SurfaceSourceChartsArchiveRecipe(NamedTuple):
    domain: MeshingDomainArchiveRecipe
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    root_parameters: np.ndarray
    # Exact chart corners as canonical rational text; source pieces as int rows.
    root_parameter_exact: ExactTensorArchiveRecipe
    root_patches: np.ndarray
    associations: tuple[GeometryAssociation, ...]
    boundary_source: MeshingDomainBoundarySourceArchiveRecipe
    source_root_cell_ids: np.ndarray
    boundary_occurrences: np.ndarray
    source_piece_ids: IntegerRowsArchiveRecipe
    expected_source_chart_id: str


class PreparedSurfaceSourceSupportArchiveRecipe(NamedTuple):
    """Constructor inputs of a prepared surface support; root banks are recomputed."""

    domain: MeshingDomainArchiveRecipe
    original: object
    root_parameters: np.ndarray
    root_boundary_source: MeshingDomainBoundarySourceArchiveRecipe
    expected_support_id: str


class SurfaceNativeRestrictionBoundarySourceArchiveRecipe(NamedTuple):
    """Exact restriction query inputs, reconstructed through its proof owner."""

    root: MeshingDomainBoundarySourceArchiveRecipe
    support: PreparedSurfaceSourceSupportArchiveRecipe
    mesh: CellMesh
    geometry: CellGeometrySpec
    source_id: str
    source_revision: str
    source_scope_id: str


class SurfaceChartBoundarySourceArchiveRecipe(NamedTuple):
    root: object
    support: object
    deformation: object
    predecessor_fidelity: object
    expected_source_id: str
    expected_source_revision: str
    expected_source_query_id: str
    expected_source_scope_id: str
    expected_domain_id: str
    expected_support_id: str
    expected_deformation_domain_id: str
    expected_deformation_id: str
    expected_predecessor_fidelity_id: str
    expected_source_profile_id: str | None
    expected_source_profile_revision: str | None
    expected_source_atlas_id: str | None
    expected_target_atlas_id: str | None


class SurfaceSubdivisionBoundarySourceArchiveRecipe(NamedTuple):
    root: MeshingDomainBoundarySourceArchiveRecipe
    support: object
    lineages: tuple[object, ...]
    vertex_stencils: tuple[object, ...]
    targets: tuple[CellMesh, ...]
    geometry: CellGeometrySpec
    source_id: str
    source_revision: str
    expected_boundary_occurrence_id: str


class SphereMaterialCellAtlasArchiveRecipe(NamedTuple):
    source_profile: object
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: object
    source_embedding: object
    expected_atlas_id: str


class PreparedSphereChartDeformationArchiveRecipe(NamedTuple):
    source_atlas: SphereMaterialCellAtlasArchiveRecipe
    target_atlas: SphereMaterialCellAtlasArchiveRecipe
    expected_deformation_id: str


class ReconstructedTriangleBoundarySourceArchiveRecipe(NamedTuple):
    source: object
    expected_source_id: str
    expected_source_revision: str


class NativeImplicitBoundaryQueryArchiveRecipe(NamedTuple):
    query: AdaptiveImplicitBoundarySourceArchiveRecipe
    source_id: str
    source_revision: str
    frame_id: str
    authoring_profile: AnalyticImplicitProfileArchiveRecipe | None
    expected_query_id: str


class NativePlcSourceArchiveRecipe(NamedTuple):
    complex: object
    source_id: str
    source_revision: str
    mapped_domain: object | None
    fidelity_source: object | None
    maximum_deviation: float | None
    plc_vertex_root_global_ids: object | None
    expected_binding_id: str


class NativeMappedHexSourceArchiveRecipe(NamedTuple):
    reference: object
    domain: object
    plc_vertex_root_global_ids: object
    expected_source_id: str
    expected_source_revision: str
    expected_binding_id: str
    expected_relation_id: str


class MappedReferenceBoundarySourceArchiveRecipe(NamedTuple):
    authority: NativeMappedHexSourceArchiveRecipe
    face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...]
    edge_scope_keys: tuple[tuple[str, tuple[str, ...]], ...]
    expected_source_query_id: str


class SurfaceEnvelopeSourceArchiveRecipe(NamedTuple):
    original: object
    source_id: str
    coordinate_contract: object
    offset: float
    spacing: float
    certificate_spacing: float
    maximum_deviation: float
    expected_source_revision: str
    expected_envelope_id: str


class PiecewiseLinearBoundarySourceArchiveRecipe(NamedTuple):
    """Owner recipe plus the selected source scope of one PWL boundary stratum.

    Source rows, entity keys, occurrence paths, and chart IDs are recomputed by
    the owner from this selection; ``expected_source_query_id`` fingerprints the
    selection and its entity keys, so restores compare one canonical ID instead
    of storing per-entity ancestry text in the manifest.
    """

    owner: SurfaceEnvelopeSourceArchiveRecipe
    entity_dimension: int
    entity_ids: object
    expected_source_query_id: str


class AnalyticImplicitProfileArchiveRecipe(NamedTuple):
    geometry: object
    coordinate_contract: object
    tube_radius: float
    cover_radius: float
    source_id: str
    source_revision: str
    expected_profile_id: str


class FiniteElementAcceptedStateArchiveRecipe(NamedTuple):
    """Stored state payload plus canonical layout recipes for FE cold restore."""

    fields: tuple[object, ...]
    time: object
    step: int
    topology_id: str
    prepared_id: str
    compilation_id: str
    field_layouts: tuple[object, ...]
    field_ownership: tuple[object, ...] | None
    materials: object | None
    schedule_cursor: int
    state_version: int
    expected_accepted_id: str


class FiniteVolumeRuntimeStateArchiveRecipe(NamedTuple):
    """Typed accepted FV state fields bound to stable source-row identity."""

    content_state: object
    topology_journal_record: Mapping[str, Any]
    topology_journal_arrays: Mapping[str, np.ndarray]
    accepted_step: Array
    step_size: Array
    last_status: Array
    controller_state: Array
    integrator_state: Array
    output_cursor: Array
    sliding_shift: Array
    sliding_event_id: str | None
    sliding_coupling_id: str | None
    source_cell_ids: Array
    expected_runtime_id: str
    expected_epoch_id: str
    expected_state_id: str


class FiniteVolumeOwnerLocalCellStateBankArchiveRecipe(NamedTuple):
    """Stable-keyed external cell-state bank layout with an array key column."""

    entry_id: str
    cell_ids: ArrayLike
    cell_axis: int


class FiniteVolumeOversetPartSpecArchiveRecipe(NamedTuple):
    """Callable-free owner-local overset part declaration."""

    part_name: str
    wall: object | None
    boundary: object | None
    excluded: object | None
    solid_index: int
    priority: int
    owner_rank: int
    image_rotation: object | None
    image_translation: object | None
    expected_spec_id: str


class FiniteVolumeOversetPolicyArchiveRecipe(NamedTuple):
    """Exact scientific overset selection policy."""

    fringe_layers: int
    donor_priority: str
    expected_policy_id: str


class FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe(NamedTuple):
    """Callable-free local source/target inputs for a cold FV repartition."""

    group_id: str
    process_indices: tuple[int, ...]
    device_keys: tuple[tuple[int, int], ...]
    mesh_axes: tuple[tuple[str, int], ...]
    parent_group_id: str | None
    process_host_ids: tuple[tuple[int, str], ...]
    part_names: tuple[str, ...]
    donor_part_name: str
    receptor_part_name: str
    source_views: tuple[object, ...]
    source_specifications: tuple[object, ...]
    source_overset_specs: tuple[object, ...]
    source_overset_policy: object
    target_overset_specs: tuple[object, ...]
    target_graph_partition_plan: object
    target_specification: object
    target_view: object
    source_topology_epoch_record: object
    source_topology_artifacts_record: object
    target_topology_epoch_record: object
    target_topology_artifacts_record: object
    source_boundary_patch_names: tuple[str, ...]
    target_boundary_patch_names: tuple[str, ...]
    system_kind: str
    spatial_dimension: int
    expected_system_id: str
    problem_name: str
    field_name: str
    source_id: str | None
    reconstruction_kind: str
    flux_kind: str
    expected_method_id: str
    flux_smooth_epsilon: float
    material_gamma: float
    material_gas_constant: float
    material_density_floor: float
    material_pressure_floor: float
    precision_policy: object
    positivity_iterations: int
    positivity_fallback_kind: str
    step_cfl: float
    step_reduction_factor: float
    step_minimum_step_size: float
    boundary_condition_kind: str
    maximum_retries: int
    topology_event_capacity: int
    topology_event_policy: str
    owner_local_axis_name: str
    expected_target_plan_id: str
    expected_source_dynamics_id: str
    expected_source_boundary_set_id: str
    expected_target_boundary_set_id: str
    expected_positivity_id: str
    expected_source_runtime_id: str
    expected_source_epoch_id: str
    expected_precision_policy_id: str
    expected_step_policy_id: str
    expected_source_artifacts_id: str
    expected_source_connectivity_id: str
    expected_target_distribution_id: str
    expected_target_connectivity_id: str
    expected_target_dynamics_id: str
    expected_target_epoch_id: str
    expected_target_artifacts_id: str


class SparseCoordinateOperatorArchiveRecipe(NamedTuple):
    relation: object
    coefficients: object
    source: object
    target: object
    properties: object
    operator_id: str
    accumulation_dtype: object
    block_shape: tuple[int, int] | None


# Entries of a periodic FE transfer archive root: the cover mesh and transfer
# settings are authoritative; the periodic and distributed transfers are derived.
PERIODIC_FINITE_ELEMENT_TRANSFER_ROOT_ENTRY_IDS = frozenset(
    {
        "mesh/periodic-topology",
        "transfer/settings",
        "transfer/periodic",
        "transfer/distributed",
    }
)


class PeriodicFiniteElementTransferArchiveRecipe(NamedTuple):
    """Identities a periodic FE transfer root recomputes from its mesh/settings.

    Quotient maps, lifts, collapses and their entry, commuting, closure and
    deck-descent evidence are derived data; the restore rebuilds them and
    requires these stored identities.
    """

    transfer_id: str
    map_id: str
    lifted_evidence_id: str
    quotient_evidence_id: str


class DistributedFiniteElementTransferArchiveRecipe(NamedTuple):
    """Per-degree DOF owner rows of a distributed periodic FE transfer.

    The sparse owner plans are rebuilt from the periodic quotient transfer,
    these rows and the root's part count; ``transfer_id`` binds the result.
    """

    source_owners: tuple[object, ...]
    target_owners: tuple[object, ...]
    transfer_id: str


class _FiniteElementSpecArchiveProjection(NamedTuple):
    """Distinguish a projected executable FE element from a stored layout recipe."""

    recipe: Any

    def rebuild(self, /) -> Any:
        _, recipe_type = _finite_element_archive_types()
        if type(self.recipe) is not recipe_type:
            raise TypeError(
                "FiniteElementSpec projection lost its canonical element recipe."
            )
        element = self.recipe.rebuild()
        element_type, _ = _finite_element_archive_types()
        if type(element) is not element_type:
            raise TypeError(
                "FiniteElementSpec recipe rebuilt an unexpected element type."
            )
        return element


def _finite_element_archive_types() -> tuple[type, type]:
    from ..discretization.fem._reference import FiniteElementSpec
    from ..solver._finite_element_schedule import FiniteElementAcceptedElementRecipe

    register_artifact_value(
        "phydrax.discretization.fem:FiniteElementSpec", FiniteElementSpec
    )
    register_artifact_value(
        "phydrax.solver:FiniteElementAcceptedElementRecipe",
        FiniteElementAcceptedElementRecipe,
    )
    return FiniteElementSpec, FiniteElementAcceptedElementRecipe


_RADICAL_TYPE_ID = "phydrax.discretization:RadicalElement"
_RADICAL_TYPE: type[RadicalElement] | None = None


def _radical_element_type() -> type[RadicalElement]:
    global _RADICAL_TYPE
    if _RADICAL_TYPE is None:
        from ..discretization._exact_algebraic_geometry import RadicalElement

        register_artifact_value(_RADICAL_TYPE_ID, RadicalElement)
        _RADICAL_TYPE = RadicalElement
    return _RADICAL_TYPE


def _is_radical_element(value: Any, /) -> bool:
    value_type = type(value)
    if (
        value_type.__module__ != "phydrax.discretization._exact_algebraic_geometry"
        or value_type.__name__ != "RadicalElement"
    ):
        return False
    return value_type is _radical_element_type()


_RATIONAL_POLYNOMIAL_TYPE_ID = "phydrax.discretization:RationalPolynomial"
_RATIONAL_POLYNOMIAL_TYPE: type | None = None


def _rational_polynomial_type() -> type:
    global _RATIONAL_POLYNOMIAL_TYPE
    if _RATIONAL_POLYNOMIAL_TYPE is None:
        from ..discretization._coordinate_enclosure import RationalPolynomial

        register_artifact_value(_RATIONAL_POLYNOMIAL_TYPE_ID, RationalPolynomial)
        _RATIONAL_POLYNOMIAL_TYPE = RationalPolynomial
    return _RATIONAL_POLYNOMIAL_TYPE


def _is_rational_polynomial(value: Any, /) -> bool:
    value_type = type(value)
    if (
        value_type.__module__ != "phydrax.discretization._coordinate_enclosure"
        or value_type.__name__ != "RationalPolynomial"
    ):
        return False
    return value_type is _rational_polynomial_type()


_FORM_TYPE_ID = "phydrax.exterior:FormType"
_FORM_VALUE_SPEC_ID = "phydrax.exterior:FormValueSpec"


def _form_value_types() -> tuple[type, type]:
    from ..exterior._form_type import FormType, FormValueSpec

    register_artifact_value(_FORM_TYPE_ID, FormType)
    register_artifact_value(_FORM_VALUE_SPEC_ID, FormValueSpec)
    return FormType, FormValueSpec


def _strict_value_field_names(value_type: type, /) -> tuple[str, ...] | None:
    form_type, form_value_spec = _form_value_types()
    if value_type is form_type:
        return (
            "dimension",
            "degree",
            "twist",
            "fiber_shape",
            "ambient_dimension",
            "form_type_id",
        )
    if value_type is form_value_spec:
        return ("form_type", "proxy", "value_spec_id")
    return None


def _form_value_recipe(value: Any, /, *, path: str) -> dict[str, Any] | None:
    if type(value).__module__ != "phydrax.exterior._form_type":
        return None
    form_type, form_value_spec = _form_value_types()
    if type(value) is form_type:
        rebuilt = form_type(
            value.dimension,
            value.degree,
            twist=value.twist,
            fiber_shape=value.fiber_shape,
            ambient_dimension=value.ambient_dimension,
        )
        if rebuilt.form_type_id != value.form_type_id:
            raise ValueError(f"{path} FormType identity is not canonical.")
        fields = {
            "dimension": value.dimension,
            "degree": value.degree,
            "twist": value.twist,
            "fiber_shape": value.fiber_shape,
            "ambient_dimension": value.ambient_dimension,
            "form_type_id": value.form_type_id,
        }
    elif type(value) is form_value_spec:
        rebuilt = form_value_spec(value.form_type, proxy=value.proxy)
        if rebuilt.value_spec_id != value.value_spec_id:
            raise ValueError(f"{path} FormValueSpec identity is not canonical.")
        fields = {
            "form_type": value.form_type,
            "proxy": value.proxy,
            "value_spec_id": value.value_spec_id,
        }
    else:
        return None
    return {
        "kind": "strict_value",
        "type": artifact_value_id(type(value)),
        "fields": {
            name: model_structure_recipe(field_value, path=f"{path}.{name}")
            for name, field_value in fields.items()
        },
    }


def _restore_strict_value(value_type: type, fields: Mapping[str, Any], /) -> Any:
    expected = _strict_value_field_names(value_type)
    if expected is None:
        raise TypeError("Stored strict value type is not an admitted form value.")
    if set(fields) != set(expected):
        raise ValueError("Stored strict value fields do not match their exact type.")
    form_type, form_value_spec = _form_value_types()
    if value_type is form_type:
        restored = form_type(
            fields["dimension"],
            fields["degree"],
            twist=fields["twist"],
            fiber_shape=fields["fiber_shape"],
            ambient_dimension=fields["ambient_dimension"],
        )
        if restored.form_type_id != fields["form_type_id"]:
            raise ValueError("Stored FormType identity does not reproduce.")
        return restored
    if value_type is form_value_spec:
        source_type = fields["form_type"]
        if type(source_type) is not form_type:
            raise TypeError("Stored FormValueSpec requires its exact FormType owner.")
        restored = form_value_spec(source_type, proxy=fields["proxy"])
        if restored.value_spec_id != fields["value_spec_id"]:
            raise ValueError("Stored FormValueSpec identity does not reproduce.")
        return restored
    raise TypeError("Stored strict value type is not an admitted form value.")


if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _is_prng_key(value: Any, /) -> bool:
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    )


def _preflight_serialized_leaf(
    file: Any,
    shape: tuple[int, ...],
    dtype: Any,
    /,
) -> int:
    start = file.tell()
    try:
        payload_dtype, payload_shape, elements = _read_npy_metadata(
            file,
            None,
            DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )
    except ArrayArchiveCorruptionError as error:
        raise ValueError("Serialized model leaf metadata is invalid.") from error
    expected_dtype = np.dtype(dtype)
    if payload_shape != shape or payload_dtype != expected_dtype:
        raise ValueError(
            "Serialized model leaf shape or dtype does not match its recipe."
        )
    payload_end = file.tell() + elements * payload_dtype.itemsize
    file.seek(start)
    return payload_end


def _canonical_recipe_payload(recipe: Mapping[str, Any], /) -> bytes:
    try:
        return json.dumps(
            recipe,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("Model artifact recipe is not canonical finite JSON.") from error


def _source_query_archive_recipe(value: Any, /) -> object | None:
    """Project query owners onto their exact inputs, excluding prepared caches."""
    if type(value).__module__ not in {
        "phydrax.geometry._curve_boundary",
        "phydrax.geometry._domain_boundary",
        "phydrax.geometry._meshing_domain",
        "phydrax.geometry._surface_source_support",
        "phydrax.geometry._piecewise_linear_boundary",
        "phydrax.geometry.implicit._adaptive_discovery",
        "phydrax.geometry.reconstruction._boundary",
        "phydrax.geometry.reconstruction._core",
        "phydrax.geometry.implicit._analytic_profile",
        "phydrax.meshing.providers._native_implicit",
        "phydrax.geometry._mapped_reference_boundary",
        "phydrax.meshing.providers._native_sources",
        "phydrax.meshing._surface_envelope",
        "phydrax.meshing._surface_boundary_sources",
        "phydrax.geometry._sphere_material_atlas",
        "phydrax.discretization._sphere_chart_records",
    }:
        return None
    from ..discretization._sphere_chart_records import PreparedSphereChartDeformation
    from ..geometry._curve_boundary import ParametricCurveBoundarySource
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
    from ..geometry._source_boundary_records import SourceBoundaryQuery
    from ..geometry._sphere_material_atlas import SphereMaterialCellAtlas
    from ..geometry._surface_source_support import (
        PreparedSurfaceSourceSupport,
        SurfaceNativeRestrictionBoundarySource,
        SurfaceSourceCharts,
    )
    from ..geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
        AdaptiveImplicitSurface,
    )
    from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
    from ..geometry.reconstruction._boundary import ReconstructedTriangleBoundarySource
    from ..geometry.reconstruction._core import (
        ReconstructedGeometrySource,
        ReconstructedGeometrySourceArchiveRecipe,
    )
    from ..meshing._plc_source import PiecewiseLinearComplex
    from ..meshing._surface_boundary_sources import (
        _boundary_occurrence_identity,
        SurfaceChartBoundarySource,
        SurfaceSubdivisionBoundarySource,
    )
    from ..meshing._surface_envelope import SurfaceEnvelopeSource
    from ..meshing.providers._native_sources import NativeMappedHexSource, NativePlcSource

    if type(value) is AnalyticImplicitProfile:
        return AnalyticImplicitProfileArchiveRecipe(
            value.geometry,
            value.coordinate_contract,
            value.tube_radius,
            value.cover_radius,
            value.source_id,
            value.source_revision,
            value.profile_id,
        )
    if type(value) is AdaptiveImplicitSurface:
        return value.archive_recipe
    if type(value) is AdaptiveImplicitBoundarySource:
        value.require_current()
        return AdaptiveImplicitBoundarySourceArchiveRecipe(
            value.surface.archive_recipe,
            value.source_id,
            value.source_revision,
            value.frame_id,
        )
    if type(value) is ParametricCurveBoundarySource:
        value.require_current()
        return ParametricCurveBoundarySourceArchiveRecipe(
            value.curves,
            value.parameter_ranges,
            value.source_id,
            value.source_revision,
            value.curve_entity_ids,
            value.occurrence_paths,
            value.chart_ids,
            value.query_id,
        )
    if type(value) is ReconstructedGeometrySource:
        return value.archive_recipe
    if type(value) is ReconstructedTriangleBoundarySource:
        return ReconstructedTriangleBoundarySourceArchiveRecipe(
            value.source.archive_recipe,
            value.source_id,
            value.source_revision,
        )
    if type(value) is MeshingDomain:
        value.require_current(value.source_id, value.source_revision)
        return MeshingDomainArchiveRecipe(
            value.patches,
            value.curves,
            int(value.corner_points.shape[0]),
            value.source_id,
            value.source_revision,
            value.regions,
            value.tolerance,
            value.accuracy,
            value.source_indices,
            value.source_occurrences,
            value.region_source_indices,
            value.region_source_occurrences,
            value.source_kinds,
            value.authority_id,
            value.brep_authority,
            value.corner_points,
            value.corner_points_authored,
            value.domain_id,
        )
    if type(value) is MeshingDomainBoundarySource:
        value.require_current()
        domain_recipe = _source_query_archive_recipe(value.domain)
        if type(domain_recipe) is not MeshingDomainArchiveRecipe:
            raise TypeError(
                "Meshing-domain boundary source lost its exact domain recipe."
            )
        return MeshingDomainBoundarySourceArchiveRecipe(
            domain_recipe,
            value.patches,
            value.resolution,
            value.boundary_occurrences,
            value.boundary_cell_ids,
            value.boundary_patch_ids,
            value.source_scope_id,
        )
    if type(value) is SurfaceSourceCharts:
        value.require_current()
        domain_recipe = _source_query_archive_recipe(value.domain)
        boundary_recipe = _source_query_archive_recipe(value.boundary_source)
        if (
            type(domain_recipe) is not MeshingDomainArchiveRecipe
            or type(boundary_recipe) is not MeshingDomainBoundarySourceArchiveRecipe
        ):
            raise TypeError("Surface source charts lost their exact source recipes.")
        return SurfaceSourceChartsArchiveRecipe(
            domain_recipe,
            value.mesh,
            value.geometry,
            value.coordinate_contract,
            value.root_parameters,
            ExactTensorArchiveRecipe.encode(value.root_parameter_exact, rational=True),
            value.root_patches,
            value.associations,
            boundary_recipe,
            value.source_root_cell_ids,
            value.boundary_occurrences,
            IntegerRowsArchiveRecipe.encode(value.source_piece_ids),
            value.source_chart_id,
        )
    if type(value) is PreparedSurfaceSourceSupport:
        value.require_current()
        domain_recipe = _source_query_archive_recipe(value.domain)
        boundary_recipe = _source_query_archive_recipe(value.root_boundary_source)
        if (
            type(domain_recipe) is not MeshingDomainArchiveRecipe
            or type(boundary_recipe) is not MeshingDomainBoundarySourceArchiveRecipe
        ):
            raise TypeError("Prepared surface support lost its exact source recipes.")
        return PreparedSurfaceSourceSupportArchiveRecipe(
            domain_recipe,
            value.original,
            value.root_parameters,
            boundary_recipe,
            value.support_id,
        )
    if type(value) is SurfaceNativeRestrictionBoundarySource:
        value.require_current()
        root_recipe = _source_query_archive_recipe(value.root)
        support_recipe = _source_query_archive_recipe(value.support)
        if (
            type(root_recipe) is not MeshingDomainBoundarySourceArchiveRecipe
            or type(support_recipe) is not PreparedSurfaceSourceSupportArchiveRecipe
        ):
            raise TypeError("Native restriction query lost its exact source recipes.")
        return SurfaceNativeRestrictionBoundarySourceArchiveRecipe(
            root_recipe,
            support_recipe,
            value.mesh,
            value.geometry,
            value.source_id,
            value.source_revision,
            value.source_scope_id,
        )
    if type(value) is SurfaceEnvelopeSource:
        value.require_current()
        original_recipe = _source_query_archive_recipe(value.original)
        if type(original_recipe) is not ReconstructedGeometrySourceArchiveRecipe:
            raise TypeError(
                "Surface envelope lost its exact reconstructed source recipe."
            )
        return SurfaceEnvelopeSourceArchiveRecipe(
            original_recipe,
            value.source_id,
            value.coordinate_contract,
            value.offset,
            value.spacing,
            value.certificate_spacing,
            value.maximum_deviation,
            value.source_revision,
            value.envelope_id,
        )
    if type(value) is PiecewiseLinearBoundarySource:
        value.require_current()
        owner_recipe = _source_query_archive_recipe(value.owner)
        if type(owner_recipe) is not SurfaceEnvelopeSourceArchiveRecipe:
            raise TypeError(
                "Piecewise-linear boundary query lost its exact envelope owner."
            )
        return PiecewiseLinearBoundarySourceArchiveRecipe(
            owner_recipe,
            value.entity_dimension,
            value.entity_ids,
            value.source_query_id,
        )
    if type(value) is NativePlcSource:
        complex_ = PiecewiseLinearComplex.from_stored_fields(value.complex)
        rebuilt = NativePlcSource(
            complex_,
            value.source_id,
            value.source_revision,
            mapped_domain=value.mapped_domain,
            fidelity_source=value.fidelity_source,
            maximum_deviation=value.maximum_deviation,
            plc_vertex_root_global_ids=value.plc_vertex_root_global_ids,
        )
        if rebuilt.binding_id != value.binding_id:
            raise ValueError(
                "Native PLC source binding identity differs from its exact inputs."
            )
        fidelity_recipe = (
            None
            if value.fidelity_source is None
            else _source_query_archive_recipe(value.fidelity_source)
        )
        if value.fidelity_source is not None and fidelity_recipe is None:
            raise TypeError("Native PLC source lost its exact fidelity-query recipe.")
        return NativePlcSourceArchiveRecipe(
            complex_,
            value.source_id,
            value.source_revision,
            value.mapped_domain,
            fidelity_recipe,
            value.maximum_deviation,
            value.plc_vertex_root_global_ids,
            value.binding_id,
        )
    if type(value) is SurfaceSubdivisionBoundarySource:
        value.require_current()
        root_recipe = _source_query_archive_recipe(value.root)
        if type(root_recipe) is not MeshingDomainBoundarySourceArchiveRecipe:
            raise TypeError("Surface subdivision lost its exact root boundary recipe.")
        return SurfaceSubdivisionBoundarySourceArchiveRecipe(
            root_recipe,
            value.support,
            value.lineages,
            value.vertex_stencils,
            value.targets,
            value.geometry,
            value.source_id,
            value.source_revision,
            _boundary_occurrence_identity(value.current_boundary_source),
        )
    if type(value) is SphereMaterialCellAtlas:
        value.require_bound(value.source_mesh, value.source_geometry)
        profile_recipe = _source_query_archive_recipe(value.source_profile)
        if type(profile_recipe) is not AnalyticImplicitProfileArchiveRecipe:
            raise TypeError(
                "Sphere material atlas lost its exact analytic profile recipe."
            )
        return SphereMaterialCellAtlasArchiveRecipe(
            profile_recipe,
            value.source_mesh,
            value.source_geometry,
            value.source_validity,
            value.source_embedding,
            value.atlas_id,
        )
    if type(value) is PreparedSphereChartDeformation:
        value.require_bound(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
        )
        source_recipe = _source_query_archive_recipe(value.source_atlas)
        target_recipe = _source_query_archive_recipe(value.target_atlas)
        if (
            type(source_recipe) is not SphereMaterialCellAtlasArchiveRecipe
            or type(target_recipe) is not SphereMaterialCellAtlasArchiveRecipe
        ):
            raise TypeError("Sphere deformation lost one of its exact atlas recipes.")
        return PreparedSphereChartDeformationArchiveRecipe(
            source_recipe, target_recipe, value.deformation_id
        )
    if type(value) is NativeMappedHexSource:
        value.validate_restored()
        reference_recipe = _source_query_archive_recipe(value.reference)
        if type(reference_recipe) is not NativePlcSourceArchiveRecipe:
            raise TypeError(
                "Mapped hex authority lost its exact native PLC reference recipe."
            )
        return NativeMappedHexSourceArchiveRecipe(
            reference_recipe,
            value.domain,
            value.root_relation.vertex_root_global_ids,
            value.source_id,
            value.source_revision,
            value.binding_id,
            value.root_relation.relation_id,
        )
    if type(value) is SurfaceChartBoundarySource:
        value.require_current()
        from ..discretization._sphere_chart_records import (
            PreparedSphereChartDeformation,
        )
        from ..meshing._surface_boundary_sources import (
            SurfaceSubdivisionBoundarySource,
        )

        deformation = value.deformation
        source_profile_id = None
        source_profile_revision = None
        source_atlas_id = target_atlas_id = None
        if type(deformation) is PreparedSphereChartDeformation:
            source_atlas = deformation.source_atlas
            target_atlas = deformation.target_atlas
            source_profile_id = source_atlas.source_profile.source_id
            source_profile_revision = source_atlas.source_profile.source_revision
            source_atlas_id = source_atlas.atlas_id
            target_atlas_id = target_atlas.atlas_id
        if isinstance(value.root, MeshingDomainBoundarySource):
            root_scope_id = value.root.source_scope_id
        elif isinstance(value.root, SurfaceSubdivisionBoundarySource):
            root_scope_id = value.root.current_boundary_source.source_scope_id
        else:
            raise TypeError(
                "Surface chart boundary source has an unregistered root query."
            )
        return SurfaceChartBoundarySourceArchiveRecipe(
            value.root,
            value.support,
            deformation,
            value.predecessor_fidelity,
            value.source_id,
            value.source_revision,
            value.source_query_id,
            root_scope_id,
            value.domain.domain_id,
            value.support.support_id,
            deformation.domain_id,
            deformation.deformation_id,
            value.predecessor_fidelity.certificate_id,
            source_profile_id,
            source_profile_revision,
            source_atlas_id,
            target_atlas_id,
        )
    if type(value) is MappedReferenceBoundarySource:
        value.require_current()
        authority_recipe = _source_query_archive_recipe(value.authority)
        if type(authority_recipe) is not NativeMappedHexSourceArchiveRecipe:
            raise TypeError("Mapped boundary query lost its exact root authority recipe.")
        return MappedReferenceBoundarySourceArchiveRecipe(
            authority_recipe,
            value.face_scope_keys,
            value.edge_scope_keys,
            value.source_query_id,
        )
    if type(value).__module__ == "phydrax.meshing.providers._native_implicit":
        from ..meshing.providers._native_implicit import _NativeImplicitBoundaryQuery

        if type(value) is _NativeImplicitBoundaryQuery:
            value.require_current()
            query_recipe = _source_query_archive_recipe(value.query)
            if type(query_recipe) is not AdaptiveImplicitBoundarySourceArchiveRecipe:
                raise TypeError("Native implicit query lost its adaptive source recipe.")
            profile_recipe: AnalyticImplicitProfileArchiveRecipe | None = None
            if value.authoring_profile is not None:
                recipe = _source_query_archive_recipe(value.authoring_profile)
                if type(recipe) is not AnalyticImplicitProfileArchiveRecipe:
                    raise TypeError(
                        "Native implicit query lost its exact authoring-profile recipe."
                    )
                profile_recipe = recipe
            return NativeImplicitBoundaryQueryArchiveRecipe(
                query_recipe,
                value.source_id,
                value.source_revision,
                value.frame_id,
                profile_recipe,
                value.query_id,
            )
    if isinstance(value, SourceBoundaryQuery):
        raise TypeError(
            f"No explicit source-query archive recipe is registered for {type(value).__name__}."
        )
    return None


def _finite_element_accepted_state_archive_recipe(
    value: Any, /
) -> FiniteElementAcceptedStateArchiveRecipe | None:
    if type(value).__module__ != "phydrax.solver._finite_element_schedule":
        return None
    from ..solver._finite_element_schedule import FiniteElementAcceptedState

    if type(value) is not FiniteElementAcceptedState:
        return None
    return FiniteElementAcceptedStateArchiveRecipe(
        value.fields,
        value.time,
        value.step,
        value.topology_id,
        value.prepared_id,
        value.compilation_id,
        tuple(layout.recipe() for layout in value.field_layouts),
        value.field_ownership,
        value.materials,
        value.schedule_cursor,
        value.state_version,
        value.accepted_id,
    )


def _periodic_finite_element_transfer_root_archive_projection(
    value: Any, /
) -> Any | None:
    """Store a periodic FE transfer root by its authoritative inputs only.

    The root retains the cover mesh, the transfer settings and the per-degree
    DOF owner rows. Every quotient operator, its evidence and the distributed
    sparse plans derive from them, so the transfer entries keep identities
    and owner rows, and the restore recomputes and requires them.
    """
    if (
        type(value).__module__ != "phydrax.lifecycle._composition_rebind"
        or type(value).__name__ != "Composition"
        or frozenset(value.entry_ids) != PERIODIC_FINITE_ELEMENT_TRANSFER_ROOT_ENTRY_IDS
    ):
        return None
    from ..discretization.fem._de_rham import (
        DistributedFiniteElementTransfer,
        PeriodicFiniteElementTransfer,
    )
    from ..lifecycle._composition_rebind import Composition, CompositionEntry

    entries = {entry.entry_id: entry for entry in value.entries}
    periodic = entries["transfer/periodic"].value
    distributed = entries["transfer/distributed"].value
    if (
        type(periodic) is PeriodicFiniteElementTransferArchiveRecipe
        and type(distributed) is DistributedFiniteElementTransferArchiveRecipe
    ):
        return None
    if (
        type(periodic) is not PeriodicFiniteElementTransfer
        or type(distributed) is not DistributedFiniteElementTransfer
    ):
        raise TypeError("Periodic FE transfer root must retain its typed transfers.")
    if (
        distributed.transfer_result.evidence_id
        != periodic.quotient_transfer_result.evidence_id
    ):
        raise ValueError(
            "Distributed FE transfer must distribute its root's periodic quotient."
        )
    compact = {
        "transfer/periodic": PeriodicFiniteElementTransferArchiveRecipe(
            periodic.transfer_id,
            periodic.complex_map.map_id,
            periodic.lifted_transfer_result.evidence_id,
            periodic.quotient_transfer_result.evidence_id,
        ),
        "transfer/distributed": DistributedFiniteElementTransferArchiveRecipe(
            tuple(np.asarray(item.source_owner) for item in distributed.degree_maps),
            tuple(np.asarray(item.target_owner) for item in distributed.degree_maps),
            distributed.transfer_id,
        ),
    }
    return Composition(
        tuple(
            CompositionEntry(
                compact.get(entry.entry_id, entry.value),
                entry_id=entry.entry_id,
                role=entry.role,
                owner_id=entry.owner_id,
                structure_id=entry.structure_id,
                revision_id=entry.revision_id,
                semantics_id=entry.semantics_id,
                dependencies=entry.dependencies,
            )
            for entry in value.entries
        ),
        boundary_id=value.boundary_id,
    )


def _finite_volume_runtime_state_archive_recipe(
    value: Any, /
) -> FiniteVolumeRuntimeStateArchiveRecipe | None:
    if (
        type(value).__module__ != "phydrax.solver._finite_volume_runtime"
        or type(value).__name__ != "FiniteVolumeValidatedRuntimeState"
    ):
        return None
    from ..solver._finite_volume_runtime import (
        FiniteVolumeRuntimeState,
        FiniteVolumeValidatedRuntimeState,
    )
    from ..solver._finite_volume_topology_events import (
        FiniteVolumeTopologyEventJournal,
    )

    if type(value) is not FiniteVolumeValidatedRuntimeState:
        raise TypeError(
            "Finite-volume archive projection requires its exact validated token."
        )
    state = value.runtime_state
    if type(state) is not FiniteVolumeRuntimeState:
        raise TypeError("Finite-volume validated token lost its canonical runtime state.")
    journal = state.topology_journal
    if type(journal) is not FiniteVolumeTopologyEventJournal:
        raise TypeError(
            "Finite-volume runtime state lost its canonical topology journal."
        )
    return FiniteVolumeRuntimeStateArchiveRecipe(
        state.content_state,
        journal.to_archive_record(),
        journal.archive_arrays(),
        state.accepted_step,
        state.step_size,
        state.last_status,
        state.controller_state,
        state.integrator_state,
        state.output_cursor,
        state.sliding_shift,
        state.sliding_event_id,
        state.sliding_coupling_id,
        value.source_cell_ids,
        value.runtime_id,
        value.epoch_id,
        value.state_id,
    )


def _finite_volume_cell_state_bank_archive_recipe(
    value: Any, /
) -> FiniteVolumeOwnerLocalCellStateBankArchiveRecipe | None:
    if (
        type(value).__module__ != "phydrax.solver._finite_volume_runtime"
        or type(value).__name__ != "FiniteVolumeOwnerLocalCellStateBankSpec"
    ):
        return None
    from ..solver._finite_volume_runtime import (
        FiniteVolumeOwnerLocalCellStateBankSpec,
    )

    if type(value) is not FiniteVolumeOwnerLocalCellStateBankSpec:
        raise TypeError("FV cell-bank archive projection requires its exact owner type.")
    return FiniteVolumeOwnerLocalCellStateBankArchiveRecipe(
        value.entry_id,
        np.asarray(value.cell_ids, dtype=np.int64),
        value.cell_axis,
    )


def _finite_volume_overset_part_spec_archive_recipe(
    value: Any, /
) -> FiniteVolumeOversetPartSpecArchiveRecipe | None:
    if (
        type(value).__module__ != "phydrax.meshing._overset"
        or type(value).__name__ != "OversetPartSpec"
    ):
        return None
    from ..meshing._overset import OversetPartSpec

    if type(value) is not OversetPartSpec:
        raise TypeError("FV archive requires the exact owner-local OversetPartSpec.")
    if value.solid_query is not None:
        raise TypeError(
            "FV archive context cannot persist an unregistered solid callback."
        )
    return FiniteVolumeOversetPartSpecArchiveRecipe(
        value.part_name,
        value.wall,
        value.boundary,
        value.excluded,
        value.solid_index,
        value.priority,
        value.owner_rank,
        value.image_rotation,
        value.image_translation,
        value.spec_id,
    )


def _finite_volume_overset_policy_archive_recipe(
    value: Any, /
) -> FiniteVolumeOversetPolicyArchiveRecipe | None:
    if (
        type(value).__module__ != "phydrax.meshing._overset"
        or type(value).__name__ != "OversetPolicy"
    ):
        return None
    from ..meshing._overset import OversetPolicy

    if type(value) is not OversetPolicy:
        raise TypeError("FV archive policy must use its exact owner type.")
    return FiniteVolumeOversetPolicyArchiveRecipe(
        value.fringe_layers,
        value.donor_priority,
        value.policy_id,
    )


def _finite_element_spec_archive_recipe(
    value: Any, /
) -> _FiniteElementSpecArchiveProjection | None:
    if type(value).__module__ != "phydrax.discretization.fem._reference":
        return None
    element_type, recipe_type = _finite_element_archive_types()
    if type(value) is not element_type:
        raise TypeError(
            "Finite-element source maps require their exact registered element."
        )
    from ..solver._finite_element_schedule import _accepted_element_recipe

    recipe = _accepted_element_recipe(value)
    if type(recipe) is not recipe_type:
        raise TypeError("Finite-element source map lost its canonical element recipe.")
    return _FiniteElementSpecArchiveProjection(recipe)


def _conic_root_line_endpoint_witness_archive_recipe(value: Any, /) -> Any | None:
    if (
        type(value).__module__ != "phydrax.geometry.brep._intersection"
        or type(value).__name__ != "ConicRootLineEndpointWitness"
    ):
        return None
    from ..geometry.brep._intersection import ConicRootLineEndpointWitness
    from ..interchange._cad_archive import ConicRootLineEndpointWitnessArchiveRecipe

    if type(value) is not ConicRootLineEndpointWitness:
        raise TypeError(
            "Conic endpoint archive projection requires its exact owner type."
        )
    return ConicRootLineEndpointWitnessArchiveRecipe(
        object.__getattribute__(value, "source_curve"),
        object.__getattribute__(value, "source_pcurve"),
        object.__getattribute__(value, "source_region"),
        object.__getattribute__(value, "parameter_exact"),
        object.__getattribute__(value, "source_interval"),
        object.__getattribute__(value, "trim_curve"),
        object.__getattribute__(value, "trim_interval"),
    )


def _cell_quality_report_archive_recipe(value: Any, /) -> object | None:
    if (
        type(value).__module__ != "phydrax.meshing._quality"
        or type(value).__name__ != "CellQualityReport"
    ):
        return None
    from ..meshing._quality import (
        CellQualityReport,
        CellQualityReportArchiveRecipe,
    )

    if type(value) is not CellQualityReport:
        raise TypeError("Quality archive projection requires CellQualityReport.")
    recipe = CellQualityReportArchiveRecipe.from_report(value)
    register_artifact_value(
        "phydrax.meshing:CellQualityReportArchiveRecipe", type(recipe)
    )
    return recipe


# Records whose archive form is a row-column recipe (``_row_record_archive_recipe``).
_ROW_RECORD_TYPES = frozenset(
    {
        ("phydrax.meshing._association", "GeometryAssociation"),
        ("phydrax.meshing._plc_source", "PiecewiseLinearComplex"),
        ("phydrax.geometry._embedding_state", "MeshEmbeddingWitness"),
        ("phydrax.geometry.surface._contracts", "SurfaceMetadata"),
        ("phydrax.discretization._surface_chart_records", "SurfaceChartWitness"),
    }
)


def model_archive_root_projection(value: Any, /) -> Any:
    """Project an archive root; a row record root keeps its registered type.

    Row-record column recipes replace nested records only. An archive root must
    stay its registered dataclass, so a row-record root projects its fields.
    """
    from ..meshing._archive_layer_columns import (
        is_layer_archive_record,
        layer_record_archive_recipe,
    )

    if (
        type(value).__module__ == "phydrax.geometry._source_fidelity_certificate"
        and type(value).__name__ == "SourceFidelityCertificate"
    ):
        row = layer_record_archive_recipe(value)
        if row is None:
            raise TypeError("Source fidelity archive root must use its canonical type.")
        fields = {
            name: model_archive_projection(field) for name, field in row.fields.items()
        }
        return _restore_dataclass(type(value), fields)

    if (type(value).__module__, type(value).__name__) not in _ROW_RECORD_TYPES and not (
        dataclasses.is_dataclass(value) and is_layer_archive_record(value)
    ):
        return model_archive_projection(value)
    fields = {
        field.name: model_archive_projection(object.__getattribute__(value, field.name))
        for field in dataclasses.fields(value)
    }
    if all(
        fields[field.name] is object.__getattribute__(value, field.name)
        for field in dataclasses.fields(value)
    ):
        return value
    return _restore_dataclass(type(value), fields)


def model_structure_matches_recipe(value: Any, recipe: Mapping[str, Any], /) -> bool:
    """Compare a fidelity root one projected field at a time, without retaining copies."""
    if (
        type(value).__module__ != "phydrax.geometry._source_fidelity_certificate"
        or type(value).__name__ != "SourceFidelityCertificate"
    ):
        return model_structure_recipe(value) == recipe
    from ..meshing._archive_layer_columns import (
        _COLUMNS,
        _encode_column,
        is_layer_archive_record,
    )

    if not is_layer_archive_record(value):
        return False
    names = tuple(field.name for field in dataclasses.fields(value))
    fields = recipe.get("fields")
    if (
        set(recipe) != {"kind", "type", "fields"}
        or recipe.get("kind") != "dataclass"
        or recipe.get("type") != artifact_value_id(type(value))
        or not isinstance(fields, dict)
        or set(fields) != set(names)
    ):
        return False
    columns = _COLUMNS["SourceFidelityCertificate"]
    for name in names:
        field = object.__getattribute__(value, name)
        if name in columns:
            field = _encode_column(field, columns[name])
        projected = model_archive_projection(field)
        if model_structure_recipe(projected, path=f"model.{name}") != fields[name]:
            return False
    return True


def _row_record_archive_recipe(value: Any, /) -> object | None:
    """Project records with one static value per mesh entity to column recipes."""
    module, name = type(value).__module__, type(value).__name__
    from ..meshing._archive_layer_columns import layer_record_archive_recipe

    layer_recipe = layer_record_archive_recipe(value)
    if layer_recipe is not None:
        return layer_recipe
    if (
        module == "phydrax.geometry.design._continuation"
        and name == "SketchContinuationReference"
    ):
        from ..geometry.design._continuation import (
            SketchContinuationReference,
            SketchContinuationReferenceArchiveRecipe,
        )

        if type(value) is not SketchContinuationReference:
            raise TypeError("Sketch continuation projection requires its owner type.")
        return SketchContinuationReferenceArchiveRecipe.from_reference(value)
    if module == "phydrax.meshing._association" and name == "GeometryAssociation":
        from ..meshing._association import (
            GeometryAssociation,
            GeometryAssociationArchiveRecipe,
        )

        if type(value) is not GeometryAssociation:
            raise TypeError("Association archive projection requires its owner type.")
        return GeometryAssociationArchiveRecipe.from_association(value)
    if module == "phydrax.geometry._embedding_state" and name == "MeshEmbeddingWitness":
        from ..geometry._embedding_state import (
            MeshEmbeddingWitness,
            MeshEmbeddingWitnessArchiveRecipe,
        )

        if type(value) is not MeshEmbeddingWitness:
            raise TypeError("Witness archive projection requires its owner type.")
        return MeshEmbeddingWitnessArchiveRecipe.from_witness(value)
    if module == "phydrax.geometry.surface._contracts" and name == "SurfaceMetadata":
        from ..geometry.surface._contracts import (
            SurfaceMetadata,
            SurfaceMetadataArchiveRecipe,
        )

        if type(value) is not SurfaceMetadata:
            raise TypeError("Surface metadata projection requires its owner type.")
        return SurfaceMetadataArchiveRecipe.from_metadata(value)
    if (
        module == "phydrax.discretization._surface_chart_records"
        and name == "SurfaceChartWitness"
    ):
        from ..discretization._surface_chart_records import (
            SurfaceChartWitness,
            SurfaceChartWitnessArchiveRecipe,
        )

        if type(value) is not SurfaceChartWitness:
            raise TypeError("Surface chart witness projection requires its owner type.")
        return SurfaceChartWitnessArchiveRecipe.from_witness(value)
    return None


def _row_record_recipe_types() -> frozenset[type]:
    from ..discretization._surface_chart_records import SurfaceChartWitnessArchiveRecipe
    from ..geometry._embedding_state import MeshEmbeddingWitnessArchiveRecipe
    from ..geometry.design._continuation import SketchContinuationReferenceArchiveRecipe
    from ..geometry.surface._contracts import SurfaceMetadataArchiveRecipe
    from ..meshing._archive_layer_columns import LayerRecordArchiveRecipe
    from ..meshing._association import GeometryAssociationArchiveRecipe

    return frozenset(
        {
            CADRevisionArchiveRecipe,
            CADTransactionArchiveRecipe,
            CADGraphArchiveRecipe,
            GeometryAssociationArchiveRecipe,
            LayerRecordArchiveRecipe,
            MeshEmbeddingWitnessArchiveRecipe,
            SurfaceChartWitnessArchiveRecipe,
            SketchContinuationReferenceArchiveRecipe,
            SurfaceMetadataArchiveRecipe,
        }
    )


def restore_row_record_recipes(value: Any, /) -> Any:
    """Rebuild column-archived row records in a freshly restored model tree.

    Each record is rebuilt by its owning constructor and must reproduce its
    stored identity. Containers created by the restore are updated in place or
    rebuilt; shared nodes stay shared.
    """
    recipe_types = _row_record_recipe_types()
    memo: dict[int, Any] = {}

    def visit(node: Any, /) -> Any:
        identity = id(node)
        if identity in memo:
            return memo[identity]
        node_type = type(node)
        if node_type in recipe_types:
            fields = tuple(visit(item) for item in node)
            result = node_type(*fields).restore()
        elif (
            node is None
            or isinstance(
                node,
                (str, bytes, bool, int, float, complex, Fraction, enum.Enum, type),
            )
            or isinstance(node, (jax.Array, np.ndarray, np.generic))
        ):
            result = node
        elif dataclasses.is_dataclass(node):
            for field in dataclasses.fields(node):
                child = object.__getattribute__(node, field.name)
                restored = visit(child)
                if restored is not child:
                    object.__setattr__(node, field.name, restored)
            result = node
        elif isinstance(node, tuple):
            items = tuple(visit(item) for item in node)
            if all(a is b for a, b in zip(items, node, strict=True)):
                result = node
            elif "_fields" in vars(node_type):
                result = node_type(*items)
            else:
                result = items
        elif isinstance(node, list):
            for index, item in enumerate(node):
                node[index] = visit(item)
            result = node
        elif isinstance(node, (set, frozenset)):
            items = [visit(item) for item in node]
            result = (
                node
                if all(a is b for a, b in zip(items, node, strict=True))
                else node_type(items)
            )
        elif isinstance(node, Mapping):
            pairs = [(key, visit(item)) for key, item in node.items()]
            if all(item is node[key] for key, item in pairs):
                result = node
            elif isinstance(node, frozendict):
                result = frozendict(dict(pairs))
            elif isinstance(node, MappingProxyType):
                result = MappingProxyType(dict(pairs))
            else:
                result = dict(pairs)
        else:
            result = node
        memo[identity] = result
        return result

    result = visit(value)
    if (
        type(value).__module__ == "phydrax.geometry._source_fidelity_certificate"
        and type(value).__name__ == "SourceFidelityCertificate"
    ):
        from ..meshing._archive_layer_columns import LayerRecordArchiveRecipe

        fields = {
            field.name: object.__getattribute__(result, field.name)
            for field in dataclasses.fields(result)
        }
        return LayerRecordArchiveRecipe(type(result).__name__, fields).restore()
    return result


def model_archive_projection(value: Any, /) -> Any:
    """Return a structural archive view with query caches replaced by owner inputs."""
    if (
        type(value).__module__ == "phydrax.meshing._plc_source"
        and type(value).__name__ == "PiecewiseLinearComplex"
    ):
        from ..meshing._plc_source import PiecewiseLinearComplex

        if type(value) is not PiecewiseLinearComplex:
            raise TypeError("PLC archive projection requires the canonical type.")
        fields: dict[str, object] = {}
        for field in dataclasses.fields(value):
            item = object.__getattribute__(value, field.name)
            if field.name in _PLC_TEXT_COLUMNS:
                fields[field.name] = TextColumnArchiveRecipe.encode(item)
            elif field.name in _PLC_PATH_COLUMNS:
                fields[field.name] = TextRowsArchiveRecipe.encode(item)
            elif field.name in _PLC_INTEGER_COLUMNS:
                fields[field.name] = integer_column(item)
            elif field.name in _PLC_INTEGER_ROWS:
                fields[field.name] = IntegerRowsArchiveRecipe.encode(item)
            else:
                fields[field.name] = model_archive_projection(item)
        return PiecewiseLinearComplexArchiveRecipe(fields)
    if type(value).__module__ == "phydrax.geometry._cad_revision":
        from ..geometry._cad_revision import (
            AssociationGraph,
            CADRevision,
            OccurrenceCorrespondenceTransaction,
        )

        if type(value) is AssociationGraph:
            return CADGraphArchiveRecipe(
                model_archive_projection(value.source_revision),
                model_archive_projection(value.target_revision),
                model_archive_projection(value.transaction),
                value.graph_id,
            )

        if type(value) is CADRevision:
            return CADRevisionArchiveRecipe(
                value.revision_id,
                value.source_id,
                value.provenance_id,
                TextRowsArchiveRecipe.encode(
                    tuple(
                        (
                            row.occurrence_id,
                            row.entity_id,
                            row.kind,
                            row.parent_occurrence_id or "",
                            str(row.orientation),
                            *row.path,
                        )
                        for row in value.occurrences
                    )
                ),
            )
        if type(value) is OccurrenceCorrespondenceTransaction:
            return CADTransactionArchiveRecipe(
                value.transaction_id,
                value.source_revision_id,
                value.target_revision_id,
                TextRowsArchiveRecipe.encode(
                    tuple(
                        (
                            row.source_occurrence_id,
                            row.target_occurrence_id,
                            row.evidence_id,
                            ""
                            if row.orientation_transform is None
                            else str(row.orientation_transform),
                        )
                        for row in value.correspondences
                    )
                ),
                value.unresolved_source_ids,
                value.unresolved_target_ids,
                value.coverage,
            )
    if (type(value).__module__, type(value).__name__) in {
        ("phydrax.geometry.brep._partition", "BRepPartitionPlan"),
        ("phydrax.geometry.brep._planar", "PlanarPartitionPlan"),
    }:
        from ..geometry.brep._partition import BRepPartitionPlan
        from ..geometry.brep._planar import PlanarPartitionPlan

        if type(value) not in (BRepPartitionPlan, PlanarPartitionPlan):
            raise TypeError("Partition archive projection requires the canonical plan.")
        planar = type(value) is PlanarPartitionPlan
        return PartitionPlanArchiveRecipe(
            value.coordinate_contract,
            value.embedding if planar else None,
            tuple(
                PartitionOperandArchiveRecipe(
                    operand.operand_id,
                    model_archive_projection(operand.source if planar else operand.model),
                    operand.role.value,
                    operand.selection,
                    operand.target_region_ids,
                )
                for operand in value.operands
            ),
            value.policy.region_precedence,
            value.plan_id,
            value.topological_dimension,
        )
    if (
        type(value).__module__ == "phydrax.geometry.brep._partition"
        and type(value).__name__ == "BRepPartitionResult"
    ):
        from ..geometry.brep._partition import (
            BRepPartitionResult,
            BRepPartitionResultArchiveRecipe,
        )

        if type(value) is not BRepPartitionResult:
            raise TypeError("Partition archive projection requires the canonical type.")
        return BRepPartitionResultArchiveRecipe(
            model_archive_projection(value.model),
            model_archive_projection(value.revision),
            model_archive_projection(value.association_graph),
            value.regions,
            value.patches,
            value.report,
            value.topological_dimension,
            model_archive_projection(value.operand_association_graphs),
            model_archive_projection(value.plan),
            value.tessellation_policy,
        )
    if (
        type(value).__module__ == "phydrax.geometry.implicit._source_charts"
        and type(value).__name__ == "ImplicitRootChartEvidence"
    ):
        from ..geometry.implicit._source_charts import ImplicitRootChartEvidence

        if type(value) is not ImplicitRootChartEvidence:
            raise TypeError(
                "Implicit root-chart evidence must use its canonical record type."
            )
        return {
            name: model_archive_projection(object.__getattribute__(value, name))
            for name in (
                "row",
                "target_cell_global_id",
                "target_entity_global_id",
                "target_local_entity_index",
                "source_cell_key",
                "root_edge_keys",
                "status",
                "field_residual_bound",
                "target_to_root_lower_bound",
                "target_to_root_bound",
                "source_normal_deviation_bounds",
                "chart_jacobian_lower",
                "chart_injectivity_margin",
                "source_gradient_lower",
                "detail",
            )
        }

    state_recipe = _finite_element_accepted_state_archive_recipe(value)
    if state_recipe is not None:
        return model_archive_projection(state_recipe)
    periodic_root = _periodic_finite_element_transfer_root_archive_projection(value)
    if periodic_root is not None:
        return model_archive_projection(periodic_root)
    fv_state_recipe = _finite_volume_runtime_state_archive_recipe(value)
    if fv_state_recipe is not None:
        return model_archive_projection(fv_state_recipe)
    bank_recipe = _finite_volume_cell_state_bank_archive_recipe(value)
    if bank_recipe is not None:
        return model_archive_projection(bank_recipe)
    overset_part_recipe = _finite_volume_overset_part_spec_archive_recipe(value)
    if overset_part_recipe is not None:
        return model_archive_projection(overset_part_recipe)
    overset_policy_recipe = _finite_volume_overset_policy_archive_recipe(value)
    if overset_policy_recipe is not None:
        return model_archive_projection(overset_policy_recipe)
    element_recipe = _finite_element_spec_archive_recipe(value)
    if element_recipe is not None:
        return model_archive_projection(element_recipe)
    quality_recipe = _cell_quality_report_archive_recipe(value)
    if quality_recipe is not None:
        return model_archive_projection(quality_recipe)
    row_recipe = _row_record_archive_recipe(value)
    if row_recipe is not None:
        return model_archive_projection(row_recipe)
    conic_endpoint_recipe = _conic_root_line_endpoint_witness_archive_recipe(value)
    if conic_endpoint_recipe is not None:
        return model_archive_projection(conic_endpoint_recipe)
    recipe = _source_query_archive_recipe(value)
    if recipe is not None:
        return model_archive_projection(recipe)
    if (
        type(value).__module__ == "phydrax.linalg._spaces"
        and type(value).__name__ == "ArraySpace"
    ):
        from ..linalg._spaces import ArraySpace

        if type(value) is not ArraySpace:
            raise TypeError("Array-space archive projection requires the canonical type.")
        recipe = value.archive_recipe()
        return type(recipe)(
            recipe.shape,
            recipe.dtype,
            model_archive_projection(recipe.pairing),
            recipe.space_id,
        )
    if (
        type(value).__module__ == "phydrax.linalg._operator_pairing"
        and type(value).__name__ == "OperatorPairing"
    ):
        from ..linalg._operator_pairing import OperatorPairing

        if type(value) is not OperatorPairing:
            raise TypeError(
                "Operator-pairing archive projection requires the canonical type."
            )
        recipe = value.archive_recipe()
        return type(recipe)(
            model_archive_projection(recipe.operator),
            model_archive_projection(recipe.solve_policy),
            recipe.pairing_id,
            recipe.operator_id,
            recipe.operator_content_fingerprint,
            recipe.solve_policy_fingerprint,
        )
    if (
        type(value).__module__ == "phydrax.sparse._linear"
        and type(value).__name__ == "SparseCoordinateOperator"
    ):
        from ..sparse._linear import SparseCoordinateOperator

        if type(value) is not SparseCoordinateOperator:
            raise TypeError(
                "Sparse archive projection requires its canonical operator type."
            )
        return SparseCoordinateOperatorArchiveRecipe(
            model_archive_projection(value.relation),
            value.coefficients,
            model_archive_projection(value.source),
            model_archive_projection(value.target),
            model_archive_projection(value.properties),
            value.operator_id,
            value.accumulation_dtype,
            value.block_shape,
        )
    if isinstance(value, (jax.Array, np.ndarray)) or _is_prng_key(value):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: model_archive_projection(
                object.__getattribute__(value, field.name)
            )
            for field in dataclasses.fields(value)
        }
        if all(
            fields[field.name] is object.__getattribute__(value, field.name)
            for field in dataclasses.fields(value)
        ):
            return value
        return _restore_dataclass(type(value), fields)
    if isinstance(value, tuple) and "_fields" in vars(type(value)):
        values = tuple(model_archive_projection(item) for item in value)
        return (
            value
            if all(a is b for a, b in zip(values, value, strict=True))
            else type(value)(*values)
        )
    if isinstance(value, tuple):
        values = tuple(model_archive_projection(item) for item in value)
        return (
            value if all(a is b for a, b in zip(values, value, strict=True)) else values
        )
    if isinstance(value, list):
        values = [model_archive_projection(item) for item in value]
        return (
            value if all(a is b for a, b in zip(values, value, strict=True)) else values
        )
    if isinstance(value, (set, frozenset)):
        values = tuple(model_archive_projection(item) for item in value)
        if all(any(projected is original for original in value) for projected in values):
            return value
        projected_set = set(values)
        return frozenset(projected_set) if isinstance(value, frozenset) else projected_set
    if isinstance(value, Mapping):
        pairs = tuple(
            (model_archive_projection(key), model_archive_projection(item))
            for key, item in value.items()
        )
        if all(
            new_key is key and new_item is item
            for (new_key, new_item), (key, item) in zip(pairs, value.items(), strict=True)
        ):
            return value
        if isinstance(value, frozendict):
            return frozendict(pairs)
        if isinstance(value, MappingProxyType):
            return MappingProxyType(dict(pairs))
        return dict(pairs)
    return value


def _recipe_contains_array(recipe: Mapping[str, Any], /) -> bool:
    pending = [recipe]
    while pending:
        node = pending.pop()
        kind = node["kind"]
        if kind in ("array", "prng_key"):
            return True
        if kind == "slice":
            pending.extend((node["start"], node["stop"], node["step"]))
        elif kind in ("dataclass", "strict_value"):
            pending.extend(node["fields"].values())
        elif kind == "finite_element_spec":
            pending.append(node["recipe"])
        elif kind in ("tuple", "list", "set", "frozenset", "namedtuple"):
            pending.extend(node["items"])
        elif kind in ("mapping", "frozendict", "mappingproxy"):
            pending.extend(item for pair in node["items"] for item in pair)
        elif kind == "rational_polynomial":
            pending.extend((node["numerator"], node["denominator"]))
    return False


def serialize_model_leaf(file: BinaryIO, value: Any, /) -> None:
    """Serialize one model leaf with canonical, pickle-free NPY encoding."""
    if _is_prng_key(value):
        np.save(file, np.asarray(jr.key_data(value)), allow_pickle=False)
        return
    if eqx.is_array_like(value):
        array = np.asarray(value)
        _recipe_array_spec(
            list(array.shape),
            str(array.dtype),
            DEFAULT_ARRAY_ARCHIVE_LIMITS,
            "serialized model leaf",
        )
        payload = np.ascontiguousarray(array) if array.ndim else array
        np.save(file, payload, allow_pickle=False)
        return
    eqx.default_serialise_filter_spec(file, value)


def deserialize_model_leaf(file: BinaryIO, value: Any, /) -> Any:
    """Deserialize one model leaf after bounded NPY metadata admission."""
    if isinstance(value, _RecipeArrayDescriptor):
        _preflight_serialized_leaf(file, value.shape, value.dtype)
        payload = np.load(
            file,
            allow_pickle=False,
            max_header_size=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_npy_header_bytes,
        )
        if value.backend == "prng_key":
            return jr.wrap_key_data(
                jnp.asarray(payload, dtype=jnp.uint32),
                impl=value.implementation,
            )
        if value.backend == "jax":
            return jnp.asarray(payload)
        return np.array(payload, copy=True, order="C")
    if _is_prng_key(value):
        key_data = jr.key_data(value)
        _preflight_serialized_leaf(
            file,
            tuple(key_data.shape),
            np.dtype(np.uint32),
        )
        data = jnp.asarray(
            np.load(
                file,
                allow_pickle=False,
                max_header_size=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_npy_header_bytes,
            ),
            dtype=jnp.uint32,
        )
        return jr.wrap_key_data(data, impl=str(jr.key_impl(value)))
    if eqx.is_array_like(value):
        if isinstance(value, (jax.Array, np.ndarray)):
            shape = tuple(value.shape)
            dtype = value.dtype
        else:
            scalar = np.asarray(value)
            shape = tuple(scalar.shape)
            dtype = scalar.dtype
        _preflight_serialized_leaf(file, shape, dtype)
    return eqx.default_deserialise_filter_spec(file, value)


def _registered_type_id(value_type: type, path: str, /) -> str:
    """Return a structured value type's artifact identity, naming it if absent."""
    try:
        return artifact_value_id(value_type)
    except TypeError as error:
        raise TypeError(
            f"{value_type.__module__}.{value_type.__qualname__} at {path} has no "
            "canonical artifact registration."
        ) from error


def model_structure_recipe(
    value: Any,
    /,
    *,
    path: str = "model",
) -> dict[str, Any]:
    """Encode static model structure without embedding Python module paths."""
    if path == "model":
        value = model_archive_root_projection(value)
    if _is_prng_key(value):
        data = np.asarray(jr.key_data(value))
        return {
            "kind": "prng_key",
            "shape": list(data.shape),
            "implementation": str(jr.key_impl(value)),
        }
    if isinstance(value, (jax.Array, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{path} contains an object-dtype array.")
        return {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "backend": "jax" if isinstance(value, jax.Array) else "numpy",
        }
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": artifact_value_id(type(value)),
            "name": value.name,
        }
    if _is_radical_element(value):
        radical = _radical_element_type().from_stored_fields(value)
        return {
            "kind": "radical",
            "type": artifact_value_id(type(radical)),
            "terms": [
                [radicand, coefficient.numerator, coefficient.denominator]
                for radicand, coefficient in radical.terms
            ],
        }
    if _is_rational_polynomial(value):
        polynomial = _rational_polynomial_type()(value.numerator, value.denominator)
        if (
            polynomial.numerator != value.numerator
            or polynomial.denominator != value.denominator
        ):
            raise ValueError(f"{path} rational polynomial is not canonical.")
        return {
            "kind": "rational_polynomial",
            "type": artifact_value_id(type(polynomial)),
            "numerator": model_structure_recipe(
                polynomial.numerator, path=f"{path}.numerator"
            ),
            "denominator": model_structure_recipe(
                polynomial.denominator, path=f"{path}.denominator"
            ),
        }
    if isinstance(value, Fraction):
        return {
            "kind": "fraction",
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite static float.")
        return {"kind": "literal", "value": value}
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{path} contains a non-finite static complex value.")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return model_structure_recipe(value.item(), path=path)
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": model_structure_recipe(value.start, path=f"{path}.start"),
            "stop": model_structure_recipe(value.stop, path=f"{path}.stop"),
            "step": model_structure_recipe(value.step, path=f"{path}.step"),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, type):
        return {"kind": "type", "value": artifact_value_id(value)}
    if type(value) is _FiniteElementSpecArchiveProjection:
        element_type, recipe_type = _finite_element_archive_types()
        if type(value.recipe) is not recipe_type:
            raise TypeError(
                "FiniteElementSpec projection must retain its exact owner recipe."
            )
        return {
            "kind": "finite_element_spec",
            "type": artifact_value_id(element_type),
            "recipe": model_structure_recipe(value.recipe, path=f"{path}.recipe"),
        }
    if isinstance(value, tuple) and "_fields" in vars(type(value)):
        return {
            "kind": "namedtuple",
            "type": _registered_type_id(type(value), path),
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    strict_value = _form_value_recipe(value, path=path)
    if strict_value is not None:
        return strict_value
    if dataclasses.is_dataclass(value) and not isinstance(value, frozendict):
        return {
            "kind": "dataclass",
            "type": _registered_type_id(type(value), path),
            "fields": {
                field.name: model_structure_recipe(
                    object.__getattribute__(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        encoded_items = [
            model_structure_recipe(item, path=f"{path}.item") for item in value
        ]
        encoded_items.sort(key=_canonical_recipe_payload)
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": encoded_items,
        }
    if isinstance(value, Mapping):
        mapping_kind = (
            "frozendict"
            if isinstance(value, frozendict)
            else "mappingproxy"
            if isinstance(value, MappingProxyType)
            else "mapping"
        )
        encoded_items = [
            (
                model_structure_recipe(key, path=f"{path}.key"),
                model_structure_recipe(item, path=f"{path}.value"),
            )
            for key, item in value.items()
        ]
        encoded_items.sort(key=lambda pair: _canonical_recipe_payload(pair[0]))
        key_payloads = [_canonical_recipe_payload(key) for key, _ in encoded_items]
        if len(set(key_payloads)) != len(key_payloads):
            raise TypeError(f"{path} contains keys with the same canonical encoding.")
        return {
            "kind": mapping_kind,
            "items": [[key, item] for key, item in encoded_items],
        }
    if callable(value):
        return {"kind": "callable", "value": artifact_value_id(value)}
    raise TypeError(
        f"Portable model artifact cannot represent {path} of type {type(value).__name__}."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ModelRecipeArray:
    """One exact dynamic array leaf declared by a model structure recipe."""

    name: str
    tree_index: int
    path: str
    shape: tuple[int, ...]
    dtype: str
    backend: str
    implementation: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class _RecipeArrayDescriptor:
    shape: tuple[int, ...]
    dtype: str
    backend: str
    implementation: str | None = None


_PRNG_IMPLEMENTATION_WORDS = {
    "threefry2x32": 2,
    "rbg": 4,
    "unsafe_rbg": 4,
}


def _recipe_fields(
    recipe: Mapping[str, Any],
    expected: set[str],
    path: str,
    /,
) -> None:
    if set(recipe) != expected:
        raise ValueError(
            f"{path} must use exactly {sorted(expected)}; found {sorted(recipe)}."
        )


def _recipe_array_spec(
    shape_value: Any,
    dtype_value: Any,
    limits: ArrayArchiveLimits | None,
    path: str,
    /,
) -> tuple[tuple[int, ...], np.dtype[Any], int, int]:
    if not isinstance(shape_value, list) or any(
        type(extent) is not int or extent < 0 for extent in shape_value
    ):
        raise ValueError(f"{path} shape must contain exact nonnegative integers.")
    shape = tuple(shape_value)
    if limits is not None and len(shape) > limits.max_array_rank:
        raise ValueError(f"{path} exceeds the recipe array-rank limit.")
    try:
        dtype = np.dtype(dtype_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} dtype is invalid.") from error
    if limits is None:
        invalid_dtype = (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.metadata is not None
            or dtype.kind not in {"b", "i", "u", "f", "c"}
        )
    else:
        invalid_dtype = (
            dtype.hasobject
            or (
                (dtype.fields is not None or dtype.subdtype is not None)
                and not limits.allow_structured_dtypes
            )
            or dtype.metadata is not None
            or dtype.kind not in limits.allowed_dtype_kinds
            or dtype.itemsize > limits.max_dtype_itemsize
        )
    if invalid_dtype:
        raise ValueError(f"{path} dtype is not admitted by recipe policy.")
    maximum_addressable = int(np.iinfo(np.intp).max)
    elements = 1
    for extent in shape:
        if extent > maximum_addressable:
            raise ValueError(f"{path} exceeds addressable array extent.")
        if limits is not None and extent > limits.max_axis_length:
            raise ValueError(f"{path} exceeds the recipe axis-length limit.")
        if extent and elements > maximum_addressable // extent:
            raise ValueError(f"{path} exceeds addressable array elements.")
        if (
            limits is not None
            and extent
            and elements > limits.max_array_elements // extent
        ):
            raise ValueError(f"{path} exceeds the recipe element limit.")
        elements *= extent
    if limits is not None and elements > limits.max_array_elements:
        raise ValueError(f"{path} exceeds the recipe element limit.")
    byte_count = elements * dtype.itemsize
    if byte_count > maximum_addressable:
        raise ValueError(f"{path} exceeds addressable array bytes.")
    if limits is not None and byte_count > limits.max_member_bytes:
        raise ValueError(f"{path} exceeds the recipe array-byte limit.")
    return shape, dtype, elements, byte_count


def validate_model_structure_recipe(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> None:
    """Validate a complete recipe and every allocation bound without allocating."""
    if limits is not None and not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance.")
    pending: list[tuple[Any, int, str]] = [(recipe, 1, "recipe")]
    canonical_sets: list[tuple[str, list[Mapping[str, Any]]]] = []
    canonical_mappings: list[tuple[str, list[Mapping[str, Any]]]] = []
    total_elements = 0
    total_bytes = 0
    node_count = 0
    while pending:
        node, depth, path = pending.pop()
        node_count += 1
        if limits is not None and node_count > limits.max_manifest_bytes:
            raise ValueError("Model artifact recipe exceeds its node limit.")
        if limits is not None and depth > limits.max_manifest_nesting:
            raise ValueError("Model artifact recipe exceeds its nesting limit.")
        if not isinstance(node, dict):
            raise TypeError(f"{path} must be a recipe object.")
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path} has no valid recipe kind.")

        if kind == "array":
            _recipe_fields(node, {"kind", "shape", "dtype", "backend"}, path)
            if node["backend"] not in ("jax", "numpy"):
                raise ValueError(f"{path} has an invalid array backend.")
            _, _, elements, byte_count = _recipe_array_spec(
                node["shape"], node["dtype"], limits, path
            )
        elif kind == "prng_key":
            _recipe_fields(node, {"kind", "shape", "implementation"}, path)
            implementation = node["implementation"]
            if (
                not isinstance(implementation, str)
                or implementation not in _PRNG_IMPLEMENTATION_WORDS
            ):
                raise ValueError(f"{path} has an unsupported PRNG implementation.")
            shape, _, elements, byte_count = _recipe_array_spec(
                node["shape"], np.dtype(np.uint32).str, limits, path
            )
            if not shape or shape[-1] != _PRNG_IMPLEMENTATION_WORDS[implementation]:
                raise ValueError(f"{path} PRNG key-data shape is incompatible.")
        else:
            elements = 0
            byte_count = 0
        if elements:
            maximum_addressable = int(np.iinfo(np.intp).max)
            if limits is None:
                if total_elements > maximum_addressable - elements:
                    raise ValueError("Model recipe exceeds addressable total elements.")
                if total_bytes > maximum_addressable - byte_count:
                    raise ValueError("Model recipe exceeds addressable total bytes.")
            else:
                if total_elements > limits.max_total_array_elements - elements:
                    raise ValueError(
                        "Model artifact recipe exceeds its total element limit."
                    )
                if total_bytes > limits.max_aggregate_bytes - byte_count:
                    raise ValueError(
                        "Model artifact recipe exceeds its total array-byte limit."
                    )
            total_elements += elements
            total_bytes += byte_count

        if kind in ("array", "prng_key"):
            continue
        if kind == "literal":
            _recipe_fields(node, {"kind", "value"}, path)
            value = node["value"]
            if value is not None and type(value) not in (str, bool, int, float):
                raise TypeError(f"{path} literal has an invalid type.")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"{path} literal must be finite.")
            continue
        if kind == "fraction":
            _recipe_fields(node, {"kind", "numerator", "denominator"}, path)
            numerator = node["numerator"]
            denominator = node["denominator"]
            if (
                type(numerator) is not int
                or type(denominator) is not int
                or denominator <= 0
                or math.gcd(numerator, denominator) != 1
            ):
                raise ValueError(f"{path} fraction must be a reduced exact rational.")
            continue
        if kind == "radical":
            _recipe_fields(node, {"kind", "type", "terms"}, path)
            radical_type = _radical_element_type()
            if (
                node["type"] != _RADICAL_TYPE_ID
                or artifact_value(node["type"]) is not radical_type
            ):
                raise TypeError(
                    f"{path} radical type is not the registered exact scalar."
                )
            terms = node["terms"]
            if not isinstance(terms, list):
                raise TypeError(f"{path} radical terms must be a list.")
            previous_radicand = 0
            for term in terms:
                if (
                    not isinstance(term, list)
                    or len(term) != 3
                    or type(term[0]) is not int
                    or term[0] <= previous_radicand
                    or type(term[1]) is not int
                    or type(term[2]) is not int
                    or term[1] == 0
                    or term[2] <= 0
                    or math.gcd(term[1], term[2]) != 1
                ):
                    raise ValueError(f"{path} radical terms are not canonical rationals.")
                previous_radicand = term[0]
            continue
        if kind == "rational_polynomial":
            _recipe_fields(node, {"kind", "type", "numerator", "denominator"}, path)
            polynomial_type = _rational_polynomial_type()
            if (
                node["type"] != _RATIONAL_POLYNOMIAL_TYPE_ID
                or artifact_value(node["type"]) is not polynomial_type
            ):
                raise TypeError(
                    f"{path} rational polynomial type is not the registered exact scalar."
                )
            for name in ("numerator", "denominator"):
                child = node[name]
                if not isinstance(child, dict) or child.get("kind") != "mappingproxy":
                    raise TypeError(
                        f"{path}.{name} must be an immutable exact polynomial map."
                    )
                pending.append((child, depth + 1, f"{path}.{name}"))
            continue
        if kind == "complex":
            _recipe_fields(node, {"kind", "real", "imag"}, path)
            components = (node["real"], node["imag"])
            if any(
                type(component) not in (int, float) or not np.isfinite(component)
                for component in components
            ):
                raise ValueError(f"{path} complex components must be finite numbers.")
            continue
        if kind == "dtype":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} dtype value must be text.")
            _recipe_array_spec([], node["value"], limits, path)
            continue
        if kind == "bytes":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} bytes value must be base64 text.")
            try:
                decoded = base64.b64decode(node["value"].encode("ascii"), validate=True)
            except (UnicodeEncodeError, ValueError) as error:
                raise ValueError(f"{path} bytes value is invalid base64.") from error
            if limits is not None and len(decoded) > limits.max_member_bytes:
                raise ValueError(f"{path} bytes value exceeds the static-byte limit.")
            continue
        if kind == "path":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} path value must be text.")
            continue
        if kind == "slice":
            _recipe_fields(node, {"kind", "start", "stop", "step"}, path)
            pending.extend(
                (
                    (node["start"], depth + 1, f"{path}.start"),
                    (node["stop"], depth + 1, f"{path}.stop"),
                    (node["step"], depth + 1, f"{path}.step"),
                )
            )
            continue
        if kind == "ellipsis":
            _recipe_fields(node, {"kind"}, path)
            continue
        if kind == "enum":
            _recipe_fields(node, {"kind", "type", "name"}, path)
            if not isinstance(node["type"], str) or not isinstance(node["name"], str):
                raise TypeError(f"{path} enum identity must contain text.")
            enum_type = artifact_value(node["type"])
            if not isinstance(enum_type, type) or not issubclass(enum_type, enum.Enum):
                raise TypeError(f"{path} enum identity does not resolve to an enum.")
            if node["name"] not in enum_type.__members__:
                raise ValueError(f"{path} enum member is unknown.")
            continue
        if kind in ("type", "callable"):
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} artifact identity must be text.")
            resolved = artifact_value(node["value"])
            if kind == "type" and not isinstance(resolved, type):
                raise TypeError(f"{path} does not resolve to a type.")
            if kind == "callable" and not callable(resolved):
                raise TypeError(f"{path} does not resolve to a callable.")
            continue
        if kind == "namedtuple":
            _recipe_fields(node, {"kind", "type", "items"}, path)
            cls = artifact_value(node["type"]) if isinstance(node["type"], str) else None
            if not isinstance(cls, type) or "_fields" not in vars(cls):
                raise TypeError(f"{path} identity does not resolve to a named tuple.")
            items = node["items"]
            if not isinstance(items, list) or len(items) != len(cls._fields):
                raise ValueError(f"{path} named-tuple fields are invalid.")
            pending.extend(
                (item, depth + 1, f"{path}.items[{index}]")
                for index, item in enumerate(items)
            )
            continue
        if kind == "finite_element_spec":
            _recipe_fields(node, {"kind", "type", "recipe"}, path)
            element_type, recipe_type = _finite_element_archive_types()
            type_id = node["type"]
            if (
                not isinstance(type_id, str)
                or type_id != artifact_value_id(element_type)
                or artifact_value(type_id) is not element_type
            ):
                raise TypeError(f"{path} FE projection has an unregistered element type.")
            recipe = node["recipe"]
            if (
                not isinstance(recipe, dict)
                or recipe.get("kind") != "dataclass"
                or recipe.get("type") != artifact_value_id(recipe_type)
                or artifact_value(recipe["type"]) is not recipe_type
            ):
                raise TypeError(f"{path} FE projection lost its exact element recipe.")
            pending.append((recipe, depth + 1, f"{path}.recipe"))
            continue
        if kind == "dataclass":
            _recipe_fields(node, {"kind", "type", "fields"}, path)
            cls = artifact_value(node["type"]) if isinstance(node["type"], str) else None
            fields = node["fields"]
            if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
                raise TypeError(f"{path} identity does not resolve to a dataclass.")
            if not isinstance(fields, dict):
                raise TypeError(f"{path} dataclass fields must be an object.")
            expected = tuple(field.name for field in dataclasses.fields(cls))
            if set(fields) != set(expected):
                raise ValueError(f"{path} dataclass fields do not match their type.")
            pending.extend(
                (fields[name], depth + 1, f"{path}.fields.{name}") for name in expected
            )
            continue
        if kind == "strict_value":
            _recipe_fields(node, {"kind", "type", "fields"}, path)
            type_id = node["type"]
            if not isinstance(type_id, str):
                raise TypeError(f"{path} strict value identity must be text.")
            value_type = artifact_value(type_id)
            if (
                not isinstance(value_type, type)
                or artifact_value_id(value_type) != type_id
            ):
                raise TypeError(
                    f"{path} strict value type is not canonically registered."
                )
            expected = _strict_value_field_names(value_type)
            fields = node["fields"]
            if expected is None:
                raise TypeError(f"{path} strict value type is not admitted.")
            if not isinstance(fields, dict):
                raise TypeError(f"{path} strict value fields must be an object.")
            if set(fields) != set(expected):
                raise ValueError(f"{path} strict value fields do not match their type.")
            pending.extend(
                (fields[name], depth + 1, f"{path}.fields.{name}") for name in expected
            )
            continue
        if kind in ("tuple", "list", "set", "frozenset"):
            _recipe_fields(node, {"kind", "items"}, path)
            items = node["items"]
            if not isinstance(items, list):
                raise TypeError(f"{path} items must be a list.")
            if kind in ("set", "frozenset"):
                canonical_sets.append((path, items))
            pending.extend(
                (item, depth + 1, f"{path}.items[{index}]")
                for index, item in enumerate(items)
            )
            continue
        if kind in ("mapping", "frozendict", "mappingproxy"):
            _recipe_fields(node, {"kind", "items"}, path)
            items = node["items"]
            if not isinstance(items, list) or any(
                not isinstance(pair, list) or len(pair) != 2 for pair in items
            ):
                raise TypeError(f"{path} mapping items must be key/value pairs.")
            keys = [pair[0] for pair in items]
            canonical_mappings.append((path, keys))
            for index, (key, value) in enumerate(items):
                pending.append((key, depth + 1, f"{path}.items[{index}].key"))
                pending.append((value, depth + 1, f"{path}.items[{index}].value"))
            continue
        raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")

    for path, items in canonical_sets:
        payloads = [_canonical_recipe_payload(item) for item in items]
        if payloads != sorted(payloads) or len(payloads) != len(set(payloads)):
            raise ValueError(f"{path} set items are not uniquely canonical.")
        if any(_recipe_contains_array(item) for item in items):
            raise ValueError(f"{path} set items cannot contain array recipes.")
    for path, keys in canonical_mappings:
        payloads = [_canonical_recipe_payload(key) for key in keys]
        if payloads != sorted(payloads) or len(payloads) != len(set(payloads)):
            raise ValueError(f"{path} mapping keys are not uniquely canonical.")
        if any(_recipe_contains_array(key) for key in keys):
            raise ValueError(f"{path} mapping keys cannot contain array recipes.")


def _restore_dataclass(cls: type[DataclassInstance], fields: Mapping[str, Any], /) -> Any:
    instance = object.__new__(cls)
    for field in dataclasses.fields(cls):
        object.__setattr__(instance, field.name, fields[field.name])
    if issubclass(cls, Strict):
        mark_strict_initialized(instance)
    return instance


def _recipe_children(recipe: Mapping[str, Any], /) -> tuple[Any, ...]:
    kind = recipe["kind"]
    if kind == "slice":
        return recipe["start"], recipe["stop"], recipe["step"]
    if kind in ("dataclass", "strict_value"):
        return tuple(recipe["fields"].values())
    if kind == "finite_element_spec":
        return (recipe["recipe"],)
    if kind == "rational_polynomial":
        return recipe["numerator"], recipe["denominator"]
    if kind in ("tuple", "list", "set", "frozenset", "namedtuple"):
        return tuple(recipe["items"])
    if kind in ("mapping", "frozendict", "mappingproxy"):
        return tuple(child for pair in recipe["items"] for child in (pair[0], pair[1]))
    return ()


def _restore_recipe(recipe: Mapping[str, Any], array_factory: Any, /) -> Any:
    """Restore validated recipe trees with an explicit stack, never recursion."""
    pending: list[tuple[Mapping[str, Any], bool]] = [(recipe, False)]
    state: dict[int, bool] = {}
    order: list[Mapping[str, Any]] = []
    while pending:
        node, expanded = pending.pop()
        identity = id(node)
        if expanded:
            state[identity] = True
            order.append(node)
            continue
        if identity in state:
            if not state[identity]:
                raise ValueError("Model recipe contains a cyclic container graph.")
            continue
        state[identity] = False
        pending.append((node, True))
        for child in reversed(_recipe_children(node)):
            child_identity = id(child)
            if child_identity in state:
                if not state[child_identity]:
                    raise ValueError("Model recipe contains a cyclic container graph.")
                continue
            pending.append((child, False))

    restored: dict[int, Any] = {}
    for node in order:
        kind = node["kind"]
        identity = id(node)
        if kind in ("array", "prng_key"):
            value = array_factory(node)
        elif kind == "literal":
            value = node["value"]
        elif kind == "fraction":
            value = Fraction(node["numerator"], node["denominator"])
        elif kind == "complex":
            value = complex(node["real"], node["imag"])
        elif kind == "dtype":
            value = np.dtype(node["value"])
        elif kind == "bytes":
            value = base64.b64decode(node["value"].encode("ascii"), validate=True)
        elif kind == "path":
            value = Path(node["value"])
        elif kind == "slice":
            value = slice(*(restored[id(child)] for child in _recipe_children(node)))
        elif kind == "ellipsis":
            value = Ellipsis
        elif kind == "enum":
            value = artifact_value(node["type"])[node["name"]]
        elif kind in ("type", "callable"):
            value = artifact_value(node["value"])
        elif kind == "namedtuple":
            cls = artifact_value(node["type"])
            value = cls(*(restored[id(child)] for child in node["items"]))
        elif kind == "radical":
            cls = artifact_value(node["type"])
            terms = tuple(
                (radicand, Fraction(numerator, denominator))
                for radicand, numerator, denominator in node["terms"]
            )
            value = cls(terms=terms)
            if value.terms != terms:
                raise ValueError(
                    "Stored radical terms are not in canonical square-free form."
                )
        elif kind == "rational_polynomial":
            cls = artifact_value(node["type"])
            value = cls(
                restored[id(node["numerator"])],
                restored[id(node["denominator"])],
            )
        elif kind == "finite_element_spec":
            _, recipe_type = _finite_element_archive_types()
            element_recipe = restored[id(node["recipe"])]
            if type(element_recipe) is not recipe_type:
                raise TypeError("FE projection did not restore its exact element recipe.")
            value = _FiniteElementSpecArchiveProjection(element_recipe)
        elif kind == "strict_value":
            value_type = artifact_value(node["type"])
            fields = {name: restored[id(child)] for name, child in node["fields"].items()}
            value = _restore_strict_value(value_type, fields)
        elif kind == "dataclass":
            fields = {name: restored[id(child)] for name, child in node["fields"].items()}
            value = _restore_dataclass(artifact_value(node["type"]), fields)
        elif kind in ("tuple", "list", "set", "frozenset"):
            items = [restored[id(child)] for child in node["items"]]
            if kind == "tuple":
                value = tuple(items)
            elif kind == "list":
                value = items
            elif kind == "set":
                value = set(items)
            else:
                value = frozenset(items)
        elif kind in ("mapping", "frozendict", "mappingproxy"):
            items = node["items"]
            value = {restored[id(key)]: restored[id(item)] for key, item in items}
            if kind == "frozendict":
                value = frozendict(value)
            elif kind == "mappingproxy":
                value = MappingProxyType(value)
        else:
            raise ValueError(f"Unknown model recipe kind {kind!r}.")
        restored[identity] = value
    return restored[id(recipe)]


def _array_descriptor(recipe: Mapping[str, Any], /) -> _RecipeArrayDescriptor:
    if recipe["kind"] == "prng_key":
        return _RecipeArrayDescriptor(
            tuple(recipe["shape"]),
            np.dtype(np.uint32).str,
            "prng_key",
            recipe["implementation"],
        )
    return _RecipeArrayDescriptor(
        tuple(recipe["shape"]),
        np.dtype(recipe["dtype"]).str,
        recipe["backend"],
    )


def _descriptor_tree(
    recipe: Mapping[str, Any],
    limits: ArrayArchiveLimits | None,
    /,
) -> Any:
    validate_model_structure_recipe(recipe, limits=limits)
    return _restore_recipe(recipe, _array_descriptor)


def model_recipe_template(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Construct a nonallocating structural template for serialized leaves."""

    return _descriptor_tree(recipe, limits)


def _serialized_leaf_specification(
    value: Any,
    /,
) -> tuple[tuple[int, ...], np.dtype[Any]] | None:
    if isinstance(value, _RecipeArrayDescriptor):
        return value.shape, np.dtype(value.dtype)
    if _is_prng_key(value):
        data = jr.key_data(value)
        return tuple(data.shape), np.dtype(np.uint32)
    if eqx.is_array_like(value):
        if isinstance(value, (jax.Array, np.ndarray)):
            return tuple(value.shape), np.dtype(value.dtype)
        scalar = np.asarray(value)
        return tuple(scalar.shape), scalar.dtype
    return None


def preflight_model_tree_serialization(file: Any, template: Any, /) -> None:
    """Bind every serialized NPY leaf to a template before device allocation."""

    start = file.tell()
    file.seek(0, 2)
    payload_end = file.tell()
    file.seek(start)
    for leaf in jax.tree_util.tree_leaves(template):
        specification = _serialized_leaf_specification(leaf)
        if specification is None:
            continue
        shape, dtype = specification
        leaf_end = _preflight_serialized_leaf(file, shape, dtype)
        if leaf_end > payload_end:
            raise ValueError("Serialized model leaf payload is truncated.")
        file.seek(leaf_end)
    if file.tell() != payload_end:
        raise ValueError(
            "Serialized model leaf inventory has trailing or missing payloads."
        )
    file.seek(start)


def model_recipe_array_inventory(
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[ModelRecipeArray, ...]:
    """Return the exact dynamic array inventory without allocating arrays."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Recipe array prefix must be non-empty.")

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(
        _descriptor_tree(recipe, limits)
    )
    return _model_recipe_array_inventory_from_leaves(recipe, path_leaves, prefix)


def _model_recipe_array_inventory_from_leaves(
    recipe: Mapping[str, Any],
    path_leaves: list[tuple[Any, Any]],
    prefix: str,
    /,
) -> tuple[ModelRecipeArray, ...]:
    """Inventory already-validated descriptor leaves without rebuilding the tree."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Recipe array prefix must be non-empty.")
    inventory: list[ModelRecipeArray] = []
    for tree_index, (path, leaf) in enumerate(path_leaves):
        if not isinstance(leaf, _RecipeArrayDescriptor):
            continue
        inventory.append(
            ModelRecipeArray(
                name=f"{prefix}/{len(inventory):06d}",
                tree_index=tree_index,
                path=jax.tree_util.keystr(path) or "<root>",
                shape=leaf.shape,
                dtype=leaf.dtype,
                backend=leaf.backend,
                implementation=leaf.implementation,
            )
        )
    if len(inventory) != _recipe_array_count(recipe):
        raise TypeError(
            "Model recipe arrays must be PyTree leaves; a registered value stores "
            "arrays inside a non-PyTree node."
        )
    return tuple(inventory)


def _recipe_array_count(recipe: Mapping[str, Any], /) -> int:
    count = 0
    pending: list[Any] = [recipe]
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            if node.get("kind") in ("array", "prng_key"):
                count += 1
                continue
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return count


def pack_model_array_tree(
    value: Any,
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> dict[str, np.ndarray]:
    """Extract the exact dynamic arrays declared by a canonical recipe."""
    value = model_archive_root_projection(value)

    descriptor_tree = _descriptor_tree(recipe, limits)
    descriptor_leaves, descriptor_definition = jax.tree_util.tree_flatten_with_path(
        descriptor_tree
    )
    value_leaves, value_definition = jax.tree_util.tree_flatten_with_path(value)
    if descriptor_definition != value_definition or len(descriptor_leaves) != len(
        value_leaves
    ):
        raise TypeError("Value PyTree does not match its model structure recipe.")
    inventory = _model_recipe_array_inventory_from_leaves(
        recipe, descriptor_leaves, prefix
    )
    arrays: dict[str, np.ndarray] = {}
    for entry in inventory:
        expected_path, descriptor = descriptor_leaves[entry.tree_index]
        actual_path, leaf = value_leaves[entry.tree_index]
        if expected_path != actual_path or not isinstance(
            descriptor, _RecipeArrayDescriptor
        ):
            raise TypeError("Value array paths do not match their model recipe.")
        if descriptor.backend == "prng_key":
            if not _is_prng_key(leaf):
                raise TypeError("Value PRNG leaf does not match its model recipe.")
            payload = np.asarray(jr.key_data(leaf))
            if str(jr.key_impl(leaf)) != descriptor.implementation:
                raise TypeError("Value PRNG implementation changed from its recipe.")
        else:
            if not isinstance(leaf, (jax.Array, np.ndarray)):
                raise TypeError("Value array leaf does not match its model recipe.")
            if descriptor.backend == "jax" and not isinstance(leaf, jax.Array):
                raise TypeError("Value array backend changed from its model recipe.")
            if descriptor.backend == "numpy" and not isinstance(leaf, np.ndarray):
                raise TypeError("Value array backend changed from its model recipe.")
            payload = np.asarray(leaf)
        if payload.shape != descriptor.shape or payload.dtype.str != descriptor.dtype:
            raise TypeError("Value array shape or dtype changed from its model recipe.")
        arrays[entry.name] = payload
    return arrays


def model_from_array_recipe(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    validate: bool = True,
    restore_rows: bool = True,
) -> Any:
    """Restore an exact array inventory, optionally deferring owner validation."""

    descriptor_tree = _descriptor_tree(recipe, limits)
    path_leaves, definition = jax.tree_util.tree_flatten_with_path(descriptor_tree)
    inventory = _model_recipe_array_inventory_from_leaves(recipe, path_leaves, prefix)
    if set(arrays) != {entry.name for entry in inventory}:
        raise ValueError("Model recipe arrays do not match its exact leaf inventory.")
    admitted: dict[str, np.ndarray] = {}
    for entry in inventory:
        payload = np.asarray(arrays[entry.name])
        if payload.shape != entry.shape or payload.dtype.str != entry.dtype:
            raise ValueError("Model recipe array shape or dtype is inconsistent.")
        admitted[entry.name] = payload

    restored = [leaf for _, leaf in path_leaves]
    for entry in inventory:
        descriptor = restored[entry.tree_index]
        if not isinstance(descriptor, _RecipeArrayDescriptor):
            raise RuntimeError("Model recipe inventory lost its array descriptor.")
        payload = admitted[entry.name]
        if entry.backend == "prng_key":
            restored[entry.tree_index] = jr.wrap_key_data(
                jnp.asarray(payload, dtype=jnp.uint32),
                impl=entry.implementation,
            )
        elif entry.backend == "jax":
            restored[entry.tree_index] = jnp.asarray(payload)
        else:
            restored[entry.tree_index] = np.array(payload, copy=True, order="C")
    model = jax.tree_util.tree_unflatten(definition, restored)
    if restore_rows:
        # Scientific authority is still delegated to the owning callback.
        model = restore_row_record_recipes(model)
    # Other owner recipes may occupy fields whose live values have a different
    # type; the scientific restore callback reconstructs and validates those.
    if validate:
        validate_tree(model)
    return model


def model_from_structure_recipe(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Construct a bounded array-zeroed template from a validated recipe."""

    validate_model_structure_recipe(recipe, limits=limits)

    def zeros(node: Mapping[str, Any], /) -> Any:
        if node["kind"] == "prng_key":
            return jr.wrap_key_data(
                jnp.zeros(tuple(node["shape"]), dtype=jnp.uint32),
                impl=node["implementation"],
            )
        dtype = np.dtype(node["dtype"])
        if node["backend"] == "jax":
            return jnp.zeros(tuple(node["shape"]), dtype=dtype)
        return np.zeros(tuple(node["shape"]), dtype=dtype)

    return _restore_recipe(recipe, zeros)


# Shared archive nodes ---------------------------------------------------------------
#
# A native model archive stores each distinct object subtree once. Every object
# node (and every container holding arrays) has a content key: the canonical
# fingerprint of its recipe with child subtrees replaced by their keys and array
# descriptors bound to their payload content. In canonical traversal order the
# first occurrence of a key is written in full; later equal nodes are written as
# ``{"kind": "shared", "key": ...}`` and their arrays are not stored again. Keys
# depend only on content, so archives are independent of object aliasing.

_SHARED_RECIPE_KIND = "shared"
_SHARED_OBJECT_KINDS = frozenset(
    {"dataclass", "namedtuple", "strict_value", "finite_element_spec"}
)
_ARRAY_RECIPE_KINDS = frozenset({"array", "prng_key"})


@dataclasses.dataclass(frozen=True, slots=True)
class _RecipeArrayNode:
    index: int


def _recipe_array_nodes(recipe: Mapping[str, Any], /) -> list[Mapping[str, Any]]:
    """Array recipe nodes in exact inventory (PyTree leaf) order.

    A node reached through several parents repeats once per occurrence.
    """
    nodes: list[Mapping[str, Any]] = []

    def marker(node: Mapping[str, Any], /) -> _RecipeArrayNode:
        nodes.append(node)
        return _RecipeArrayNode(len(nodes) - 1)

    return [
        nodes[leaf.index]
        for leaf in jax.tree_util.tree_leaves(_restore_recipe(recipe, marker))
        if isinstance(leaf, _RecipeArrayNode)
    ]


def _map_recipe_children(node: Mapping[str, Any], transform: Any, /) -> dict[str, Any]:
    """Copy one recipe node, replacing direct children in canonical order."""
    kind = node["kind"]
    mapped = dict(node)
    if kind == "slice":
        for name in ("start", "stop", "step"):
            mapped[name] = transform(node[name])
    elif kind in ("dataclass", "strict_value"):
        fields = node["fields"]
        if not isinstance(fields, dict):
            raise ValueError("Recipe fields must be a mapping.")
        mapped["fields"] = {name: transform(fields[name]) for name in sorted(fields)}
    elif kind == "finite_element_spec":
        mapped["recipe"] = transform(node["recipe"])
    elif kind == "rational_polynomial":
        mapped["numerator"] = transform(node["numerator"])
        mapped["denominator"] = transform(node["denominator"])
    elif kind in ("tuple", "list", "set", "frozenset", "namedtuple"):
        mapped["items"] = [transform(child) for child in node["items"]]
    elif kind in ("mapping", "frozendict", "mappingproxy"):
        mapped["items"] = [
            [transform(key), transform(item)] for key, item in node["items"]
        ]
    return mapped


@dataclasses.dataclass(slots=True)
class _RecipeContent:
    payloads: dict[int, np.ndarray]
    keys: dict[int, str] = dataclasses.field(default_factory=dict)
    with_arrays: set[int] = dataclasses.field(default_factory=set)

    def digest(self, node: Mapping[str, Any], /) -> None:
        """Key one node whose composite and array children are already keyed."""
        identity = id(node)
        if node["kind"] in _ARRAY_RECIPE_KINDS:
            self.with_arrays.add(identity)
            content: Any = {"node": dict(node), "payload": self.payloads[identity]}
        else:

            def child_content(child: Mapping[str, Any], /) -> Any:
                child_identity = id(child)
                if child_identity in self.keys:
                    if child_identity in self.with_arrays:
                        self.with_arrays.add(identity)
                    return {"key": self.keys[child_identity]}
                return child

            content = {"node": _map_recipe_children(node, child_content)}
        self.keys[identity] = canonical_fingerprint(
            {"kind": "model-archive-recipe-node", **content}
        )

    def shareable(self, node: Mapping[str, Any], /) -> bool:
        return (
            node["kind"] in _SHARED_OBJECT_KINDS
            or (
                node["kind"] in ("tuple", "list", "set", "frozenset")
                and len(node["items"]) >= 2
            )
            or (node["kind"] not in _ARRAY_RECIPE_KINDS and id(node) in self.with_arrays)
        )


def _is_recipe_leaf(node: Mapping[str, Any], /) -> bool:
    return node["kind"] not in _ARRAY_RECIPE_KINDS and not _recipe_children(node)


def share_model_archive_recipe(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    /,
    *,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Write each distinct object subtree and array bank once.

    ``arrays`` is the exact per-occurrence inventory of ``recipe``. The result is
    the canonical shared manifest recipe and its array members, named in
    canonical traversal order.
    """
    nodes = _recipe_array_nodes(recipe)
    names = [f"{prefix}/{index:06d}" for index in range(len(nodes))]
    if set(arrays) != set(names):
        raise ValueError("Model recipe arrays do not match its exact leaf inventory.")
    content = _RecipeContent({})
    for node, name in zip(nodes, names, strict=True):
        content.payloads.setdefault(id(node), np.asarray(arrays[name]))

    def digest(node: Mapping[str, Any], /) -> None:
        if id(node) in content.keys:
            return
        for child in _recipe_children(node):
            if not _is_recipe_leaf(child):
                digest(child)
        content.digest(node)

    digest(recipe)
    written: set[str] = set()
    members: dict[str, np.ndarray] = {}

    def emit(node: Mapping[str, Any], /) -> dict[str, Any]:
        identity = id(node)
        if node["kind"] in _ARRAY_RECIPE_KINDS:
            members[f"{prefix}/{len(members):06d}"] = content.payloads[identity]
            return dict(node)
        if _is_recipe_leaf(node):
            return dict(node)
        if content.shareable(node):
            key = content.keys[identity]
            if key in written:
                return {"kind": _SHARED_RECIPE_KIND, "key": key}
            written.add(key)
        return _map_recipe_children(node, emit)

    return emit(recipe), members


def expand_shared_model_archive_recipe(
    shared: Mapping[str, Any],
    members: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Resolve a canonical shared manifest into its recipe and array inventory.

    Shared references resolve only to an earlier complete subtree with the same
    recomputed content key, so a changed definition is refused for every
    referrer. Equal subtrees written twice, unused members, or unresolved keys
    are refused as noncanonical. The validated recipe reuses one node per key.
    """
    content = _RecipeContent({})
    definitions: dict[str, dict[str, Any]] = {}
    consumed = [0]

    def expand(node: Any, /) -> dict[str, Any]:
        if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
            raise ValueError("Shared model recipe nodes must be recipe objects.")
        kind = node["kind"]
        if kind == _SHARED_RECIPE_KIND:
            if set(node) != {"kind", "key"} or not isinstance(node["key"], str):
                raise ValueError("Shared model recipe references are not canonical.")
            definition = definitions.get(node["key"])
            if definition is None:
                raise ValueError(
                    "Shared model recipe reference has no earlier equal definition."
                )
            return definition
        if kind in _ARRAY_RECIPE_KINDS:
            name = f"{prefix}/{consumed[0]:06d}"
            consumed[0] += 1
            if name not in members:
                raise ValueError("Shared model recipe array member is missing.")
            expanded = dict(node)
            content.payloads[id(expanded)] = np.asarray(members[name])
            content.digest(expanded)
            return expanded
        if _is_recipe_leaf(node):
            return dict(node)
        expanded = _map_recipe_children(node, expand)
        content.digest(expanded)
        if content.shareable(expanded):
            key = content.keys[id(expanded)]
            if key in definitions:
                # Earlier archive writers did not share array-free containers.
                # Their repeated definitions remain admissible; references still
                # resolve only to an earlier content-identical definition.
                if (
                    expanded["kind"] in _SHARED_OBJECT_KINDS
                    or id(expanded) in content.with_arrays
                ):
                    raise ValueError("Shared model recipe repeats an equal subtree.")
            else:
                definitions[key] = expanded
        return expanded

    try:
        recipe = expand(shared)
    except (AttributeError, KeyError, TypeError, RecursionError) as error:
        raise ValueError("Shared model recipe is malformed.") from error
    if consumed[0] != len(members):
        raise ValueError("Shared model archive has unused array members.")
    validate_model_structure_recipe(recipe, limits=limits)
    arrays = {
        f"{prefix}/{index:06d}": content.payloads[id(node)]
        for index, node in enumerate(_recipe_array_nodes(recipe))
    }
    return recipe, arrays


__all__ = [
    "expand_shared_model_archive_recipe",
    "share_model_archive_recipe",
    "deserialize_model_leaf",
    "model_from_array_recipe",
    "model_from_structure_recipe",
    "model_recipe_template",
    "model_recipe_array_inventory",
    "ModelRecipeArray",
    "model_structure_recipe",
    "pack_model_array_tree",
    "serialize_model_leaf",
    "preflight_model_tree_serialization",
    "validate_model_structure_recipe",
]

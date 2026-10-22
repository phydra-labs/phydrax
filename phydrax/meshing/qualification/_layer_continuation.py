#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Accepted layer/core lifecycle state: fields, finite volumes, history, PDE.

One accepted state of the native layer/core lineage carries

* H1 Q2, DG1, H(curl) degree 2 and H(div) degree 2 coefficient fields, moved
  across each archived adaptation through its own parent, geometry-transition
  and coarsening witnesses with adjoint and physical-content evidence;
* the finite-volume cell average of the declared two-material density, moved
  through the mapped nested conservative remap that preserves every material
  inventory;
* a harmonic diffusion solution, an irreversible damage history and the
  committed material state, carried by the atomic finite-element topology
  transaction whose changed-PDE certificate must roll back unchanged and whose
  target is certified against an independent solve on the adapted state.
* when the producer also owns the unit-box P1 ``x+y+z`` solve, its actual
  original coefficient bank is retained separately as ``original-u`` in every
  accepted epoch, transferred from the archived source bank and checked against
  an independent P1 solve and mapped physical L2 target in the cold process.

A warm producer authors the initial state once, continues it through every
adaptation it executes and checkpoints each accepted state as a
``MeshingAcceptedEpoch``. ``continue_layer_core_lifecycle`` restores the
archived state preceding the final adaptation in a fresh process, continues
it through that archived adaptation on the same route, solves the continued
state independently and compares it like-for-like with the producer's
archived warm continuation of the same step.
Scientific restoration, state preparation, continuation and comparison share
one current request allowance; archived execution receipts remain history.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import cast, final, NamedTuple, NotRequired, TypedDict
from weakref import ref, ReferenceType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax import (
    BranchDifferentiationPolicy,
    model_array_bytes,
    NonTrainableState,
    StrictModule,
)
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._execution_sampling import compiled_memory_estimate, CompiledMemoryEstimate
from phydrax._fingerprint import logical_array_value_collection_digest
from phydrax._frozendict import frozendict
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization import (
    CellMesh,
    FieldTracePolicy,
    NestedReferenceWitnesses,
    PiecewiseConstantReconstruction,
    PolynomialComposedCellGeometryElement,
    reference_cell_topology,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_identity import (
    coordinate_identity_inputs,
    CoordinateIdentityInputs,
    require_coordinate_identity_inputs,
)
from phydrax.discretization._coordinate_operands import PreparedCoordinateOperands
from phydrax.discretization.fem import (
    dirichlet_constraint,
    discontinuous_element,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementFieldTransfer,
    FiniteElementPlan,
    FiniteElementSpec,
    FiniteElementTransferDiscretization,
    form_element,
    lagrange_element,
    prepare_nested_compatible_field_transfers,
    prepare_nested_field_transfer,
)
from phydrax.discretization.fem._reference_rules import degree_aware_reference_rule
from phydrax.discretization.fem._topology_transfer import (
    prepare_shared_nested_field_transfer,
)
from phydrax.discretization.finite_volume import (
    prepare_unstructured_conservative_remap,
    UnstructuredConservativeRemapPlan,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.equations import (
    compile_finite_element_problem,
    CompiledFiniteElementProblem,
    DiffusionAction,
    FiniteElementForm,
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from phydrax.geometry._domain_coverage import DomainCoverageCertificate
from phydrax.geometry._embedding_state import coordinate_certificate_request
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    MeshingFiniteVolumeFieldDeclaration,
    prepare_meshing_field_index_binding,
    read_meshing_source_execution_controls,
)
from phydrax.lifecycle._meshing_sources import (
    _read_meshing_source_archive,
    _restore_meshing_source_archive,
    meshing_source_owner_reconstruction_scope,
)
from phydrax.linalg import DenseLU, LinearSolvePolicy, LinearSystem, solve
from phydrax.meshing import (
    CellMeshingResult,
    MeshAdaptationResult,
    MeshAdaptationStatus,
    MeshingLimits,
    MeshLineage,
    MeshTransitionKind,
    NativeExecutionRecord,
    NativeMeshingOptions,
    VolumeMeshingSpec,
)
from phydrax.meshing._measurements import (
    native_publication_resource_binding,
    NativeExecutionInterchange,
    NativePublicationResourceBinding,
)
from phydrax.meshing._native_preparation import native_preparation_ledger
from phydrax.meshing._volume_generation import native_volume_source_execution_budget
from phydrax.meshing.providers import NativeLayerCoreSource
from phydrax.solver import FiniteElementAcceptedState, FiniteElementTopologyTransaction
from phydrax.typing import checked
from phydrax.units import KILOGRAM_PER_CUBIC_METER, ONE


class FiniteVolumeFieldTemplate(TypedDict):
    name: str
    space_family: str
    degree: int
    continuity: str
    component_shape: tuple[int, ...]
    component_names: tuple[str, ...]
    role: str


def layer_core_finite_volume_field_templates() -> tuple[FiniteVolumeFieldTemplate, ...]:
    """Return the cell-average material-density declaration."""
    return (
        {
            "name": "state",
            "space_family": "finite-volume",
            "degree": 0,
            "continuity": "cell-local",
            "component_shape": (1,),
            "component_names": ("value",),
            "role": "conservative-density",
        },
    )


class LayerCoreFieldContinuation(TypedDict):
    """One coefficient field moved across one archived adaptation."""

    field: str
    dofs: list[int]
    adjoint_error: float
    compiled_error: float
    physical_error: float | None
    physical_scope: str
    defects: dict[str, float]


class LayerCoreVolumeContinuation(TypedDict):
    """The accepted finite-volume density moved across one archived adaptation."""

    cells: list[int]
    region_inventories: list[float]
    defects: dict[str, float]


class LayerCoreHistoryStage(TypedDict):
    phase: str
    cells: int
    changed_pde_rolled_back: bool
    pde_defect: float
    damage_defect: float
    inventory_defect: float


class LayerCoreCompilerEvidence(TypedDict):
    """XLA plan and measured executions of one accepted-epoch harmonic solve."""

    temporary_bytes: int | None
    output_bytes: int | None
    argument_bytes: int | None
    generated_code_bytes: int | None
    memory_status: str
    compilation_count: int
    compilation_count_scope: str
    warm_samples_seconds: list[float]
    warm_recompilation_count: int
    execution_scope: str
    assembly_binding_id: str


class LayerCoreResultIdentity(TypedDict):
    result_id: str
    topology_id: str
    vertices: int
    cells: int
    audit_passed: bool
    resource_guarantees: dict[str, list[str]]
    execution_evidence: NativeExecutionInterchange | None
    resource_binding: NativePublicationResourceBinding | None


class LayerCoreSolveEpoch(TypedDict):
    """One accepted state's PDE solve, its publication runtime, and retained arrays."""

    epoch: int
    name: str
    result: LayerCoreResultIdentity
    dofs: int
    compiler: LayerCoreCompilerEvidence
    stages_seconds: dict[str, float]
    retained_bytes: int


class LayerCoreStepContinuation(TypedDict):
    """Every accepted array of one state continued across one adaptation."""

    phase: str
    adaptation_id: str
    source_epoch: int
    target_epoch: int
    fields: list[LayerCoreFieldContinuation]
    finite_volume: LayerCoreVolumeContinuation
    history: LayerCoreHistoryStage
    solve: LayerCoreSolveEpoch
    stages_seconds: dict[str, float]
    original_scalar: NotRequired[dict[str, object]]


class LayerCoreEpochIdentity(TypedDict):
    """Archived identity of one accepted state and its numerical banks."""

    epoch: int
    epoch_id: str
    result_id: str
    transition_id: str | None
    field_banks_id: str


LAYER_CORE_FIELD_ROLES = MappingProxyType(
    {
        "H1": "intensive",
        "DG": "conservative-density",
        "Hcurl": "compatible-cochain",
        "Hdiv": "compatible-cochain",
    }
)
LAYER_CORE_FIELD_KINDS = tuple(LAYER_CORE_FIELD_ROLES)
LAYER_CORE_REGION_DENSITIES = (2.0, 3.0)


class _HarmonicSolve(NamedTuple):
    """One harmonic solve, its values and, for accepted epochs, its program."""

    space: FiniteElementDiscretization
    problem: CompiledFiniteElementProblem
    system: tuple[LinearSystem, object]
    values: Array
    executable: Callable[..., tuple[Array, Array]] | None
    memory: CompiledMemoryEstimate | None
    stages_seconds: dict[str, float]


class _PreparedLayerCoreFieldTemplates(NamedTuple):
    """Actual reference declarations retained by one continuation owner."""

    templates: tuple[tuple[str, FiniteElementFieldSpec], ...]
    inputs: CoordinateIdentityInputs
    factories: tuple[ReferenceType[object], ...]
    roles: tuple[tuple[str, str], ...]


@final
class LayerCoreLifecyclePlans(StrictModule):
    """Immutable preparations, optionally owned by one live request's budget."""

    transfer_spaces: dict[
        str, tuple[CellMeshingResult, FiniteElementTransferDiscretization]
    ] = eqx.field(default_factory=dict)
    history_spaces: dict[str, tuple[CellMeshingResult, FiniteElementDiscretization]] = (
        eqx.field(default_factory=dict)
    )
    original_scalar_spaces: dict[
        str, tuple[CellMeshingResult, FiniteElementDiscretization]
    ] = eqx.field(default_factory=dict)
    field_transfers: dict[tuple[str, str], FiniteElementFieldTransfer] = eqx.field(
        default_factory=dict
    )
    finite_volumes: dict[
        str, tuple[CellMeshingResult, UnstructuredFiniteVolumeDiscretization]
    ] = eqx.field(default_factory=dict)
    remaps: dict[tuple[str, str], UnstructuredConservativeRemapPlan] = eqx.field(
        default_factory=dict
    )
    diffusion_solutions: dict[tuple[str, float], _HarmonicSolve] = eqx.field(
        default_factory=dict
    )
    _field_templates: dict[str, _PreparedLayerCoreFieldTemplates] = eqx.field(
        default_factory=dict
    )
    _request_budget: list[CoordinateEnclosureBudget | None] = eqx.field(
        static=True, default_factory=lambda: [None]
    )

    def __init__(self, *, budget: CoordinateEnclosureBudget | None = None) -> None:
        self.transfer_spaces = {}
        self.history_spaces = {}
        self.field_transfers = {}
        self.original_scalar_spaces = {}
        self.finite_volumes = {}
        self.remaps = {}
        self.diffusion_solutions = {}
        self._field_templates = {}
        self._request_budget = [budget]

    @property
    def budget(self) -> CoordinateEnclosureBudget | None:
        return self._request_budget[0]

    @contextmanager
    def bind_budget(
        self, budget: CoordinateEnclosureBudget, /
    ) -> Iterator[LayerCoreLifecyclePlans]:
        previous = self.budget
        if previous is not None and previous is not budget:
            raise ValueError("LayerCoreLifecyclePlans must not replace another live request budget")
        self._request_budget[0] = budget
        try:
            yield self
        finally:
            self._request_budget[0] = previous


@final
class LayerCoreAcceptedState(StrictModule):
    """Every accepted array of one layer/core lifecycle epoch.

    ``fields`` holds the H1, DG, H(curl) and H(div) coefficients in
    ``LAYER_CORE_FIELD_KINDS`` order, ``density`` the finite-volume cell average
    of the declared material in finite-volume cell order, and ``history`` the
    transaction state whose fields are the harmonic solution and the damage
    history and whose materials commit that same density in mesh cell order.
    ``epoch`` is the accepted state version.
    ``original_scalar``, when present, is the unit-box producer's actual P1
    ``x+y+z`` solution, separate from the canonical Q2 harmonic history.
    """

    result: CellMeshingResult
    fields: tuple[Array, ...]
    density: Array
    history: FiniteElementAcceptedState
    original_scalar: Array | None
    epoch: int = eqx.field(static=True)

    def __init__(
        self,
        result: CellMeshingResult,
        fields: tuple[Array, ...],
        density: Array,
        history: FiniteElementAcceptedState,
        epoch: int,
        *,
        original_scalar: Array | None = None,
    ) -> None:
        if len(fields) != len(LAYER_CORE_FIELD_KINDS):
            raise ValueError(
                f"LayerCoreAcceptedState.fields must hold one bank per {LAYER_CORE_FIELD_KINDS!r}; got {len(fields)}"
            )
        if (
            history.topology_id != result.mesh.topology_id
            or history.state_version != epoch
            or len(history.fields) != 2
            or history.materials is None
        ):
            raise ValueError(
                f"LayerCoreAcceptedState.history must bind epoch {epoch} of topology {result.mesh.topology_id!r} with two fields and materials; got {(history.topology_id, history.state_version, len(history.fields), history.materials is not None)!r}"
            )
        self.result = result
        self.fields = tuple(jnp.asarray(value) for value in fields)
        self.density = jnp.asarray(density)
        self.history = history
        self.epoch = int(epoch)
        self.original_scalar = (
            None if original_scalar is None else jnp.asarray(original_scalar)
        )


class LayerCoreMaterialDeclaration(TypedDict):
    source_id: str
    source_revision: str
    domain_id: str
    region_ids: tuple[str, ...]
    densities: tuple[float, ...]
    site_id: str


@checked
def layer_core_material_declaration(
    source: CellMeshingResult,
    /,
    *,
    densities: tuple[float, ...] = LAYER_CORE_REGION_DENSITIES,
    site_id: str = "two-material-density",
) -> LayerCoreMaterialDeclaration:
    report = source.certification
    if (
        report is None
        or not isinstance(report.coverage, DomainCoverageCertificate)
        or report.request.domain is None
    ):
        raise ValueError(
            f"material.source_certification must include coverage and a domain; got {report!r}"
        )
    declaration: LayerCoreMaterialDeclaration = {
        "source_id": report.request.domain.source_id,
        "source_revision": report.request.domain.source_revision,
        "domain_id": report.request.domain.domain_id,
        "region_ids": report.coverage.region_ids,
        "densities": densities,
        "site_id": site_id,
    }
    layer_core_material_values(source, declaration)
    return declaration


@checked
def layer_core_material_values(
    result: CellMeshingResult, declaration: LayerCoreMaterialDeclaration, /
) -> np.ndarray:
    report = result.certification
    if (
        report is None
        or not report.passed
        or not isinstance(report.coverage, DomainCoverageCertificate)
        or report.coverage.status != "certified"
        or report.request.domain is None
    ):
        raise ValueError(
            f"material.result_certification must pass with certified coverage and a domain; got {report!r}"
        )
    if (
        report.request.domain.source_id,
        report.request.domain.source_revision,
        report.coverage.domain_id,
        report.coverage.region_ids,
    ) != (
        declaration["source_id"],
        declaration["source_revision"],
        declaration["domain_id"],
        declaration["region_ids"],
    ):
        raise ValueError(
            f"material.declaration must bind the result source and region namespace; got {declaration!r} for {(report.request.domain.source_id, report.request.domain.source_revision, report.coverage.domain_id, report.coverage.region_ids)!r}"
        )
    values = np.asarray(declaration["densities"], dtype=np.float64)
    if (
        values.shape != (len(declaration["region_ids"]),)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            f"material.densities must be finite and positive with shape {(len(declaration['region_ids']),)!r}; got {values!r}"
        )
    MaterialSiteId(declaration["site_id"])
    labels = layer_core_region_labels(result)
    if np.any(labels < 0) or np.any(labels >= values.size):
        raise ValueError(
            f"material.cell_regions must index the density bank of size {values.size}; got {labels!r}"
        )
    return values


def accepted_layer_core_lineage(records: object, /) -> tuple[MeshingAcceptedEpoch, ...]:
    """The archived accepted epochs of alternating refinements and inverse coarsenings.

    Epoch ``0`` owns the generation; every later epoch owns the adaptation that
    produced it, so the lineage is ``tuple(epoch.transition for epoch in
    epochs[1:])``.
    """
    if not isinstance(records, Mapping):
        raise TypeError(
            f"closure.records must be a named record mapping; got {type(records).__name__}"
        )
    accepted = records["accepted_data"]
    if not isinstance(accepted, Mapping) or set(accepted) != {"accepted_epochs"}:
        raise ValueError(
            f"closure.accepted_data must retain exactly the accepted epochs; got {accepted!r}"
        )
    epochs = cast(tuple[MeshingAcceptedEpoch, ...], accepted["accepted_epochs"])
    if (
        type(epochs) is not tuple
        or len(epochs) < 3
        or len(epochs) % 2 != 1
        or any(type(epoch) is not MeshingAcceptedEpoch for epoch in epochs)
    ):
        raise TypeError(
            f"closure.accepted_epochs must be an odd tuple of at least three MeshingAcceptedEpoch records; got {epochs!r}"
        )
    target = records["accepted_target"]
    if (
        not isinstance(target, CellMeshingResult)
        or target.result_id != epochs[-1].carrier.result_id
    ):
        raise ValueError(
            f"closure.accepted_target must be the final accepted carrier {epochs[-1].carrier.result_id!r}; got {target!r}"
        )
    if epochs[0].transition is not None:
        raise ValueError(
            f"closure.accepted_epochs[0] must own the generation without a transition; got {epochs[0].transition!r}"
        )
    for index, (previous, epoch) in enumerate(zip(epochs, epochs[1:]), start=1):
        stage = epoch.transition
        coarsening = index % 2 == 0
        if (
            not isinstance(stage, MeshAdaptationResult)
            or stage.source.result_id != previous.carrier.result_id
            or stage.target.result_id != epoch.carrier.result_id
            or _is_coarsening(stage) != coarsening
        ):
            raise ValueError(
                f"closure.accepted_epochs[{index}].transition must {'coarsen' if coarsening else 'refine'} epoch {index - 1}; got {stage!r}"
            )
        if stage.status is not MeshAdaptationStatus.COMPLETE:
            raise ValueError(
                f"adaptation.status must be MeshAdaptationStatus.COMPLETE; got {stage.status!r}"
            )
    for epoch in epochs:
        if ("fe/original-u" in epoch.fields) != (
            "fe/original-u" in epochs[0].fields
        ):
            raise ValueError(
                "continuation.original_scalar must be retained in every accepted epoch or absent from every epoch"
            )
        certification = epoch.carrier.certification
        if certification is None or not certification.passed:
            raise ValueError(
                f"continuation.certification must pass; got {certification!r}"
            )
    return epochs


def _is_coarsening(adaptation: MeshAdaptationResult, /) -> bool:
    """Whether the adaptation is an inverse coarsening, else a nested refinement."""
    transition = adaptation.transition
    if transition is None or transition.transition_kind not in (
        MeshTransitionKind.REFINE,
        MeshTransitionKind.COARSEN,
    ):
        raise ValueError(
            f"adaptation.transition must be a nested refinement or inverse coarsening; got {None if transition is None else transition.transition_kind!r}"
        )
    return transition.transition_kind is MeshTransitionKind.COARSEN


def _cell_ids(result: CellMeshingResult, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )


@checked
def layer_core_cell_regions(result: CellMeshingResult, /) -> dict[int, int]:
    report = result.certification
    if report is None or report.request.cell_regions is None:
        raise ValueError(
            f"continuation.cell_regions must be present in certification; got {report!r}"
        )
    labels = np.asarray(report.request.cell_regions, dtype=np.int64)
    return dict(zip(_cell_ids(result).tolist(), labels.tolist(), strict=True))


def _field_template(
    kind: str,
    cell_kind: str,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[str, FiniteElementFieldSpec]:
    """Build caller-owned element records; only their owners cache host tables.

    FiniteElementSpec can carry device arrays, so continuation must not retain
    these records in a process-global cache.
    """
    match kind:
        case "H1":
            element = lagrange_element(cell_kind, 2)
        case "DG":
            element = discontinuous_element(cell_kind, 1)
        case "Hcurl":
            element = form_element(
                cell_kind,
                1,
                2,
                family=(
                    "tensor-trimmed"
                    if cell_kind in ("quadrilateral", "hexahedron")
                    else "trimmed"
                ),
                twist="untwisted",
                proxy="circulation",
                budget=budget,
            )
        case "Hdiv":
            element = form_element(
                cell_kind,
                reference_cell_topology(cell_kind).dimension - 1,
                2,
                family=(
                    "tensor-trimmed"
                    if cell_kind in ("quadrilateral", "hexahedron")
                    else "trimmed"
                ),
                twist="twisted",
                proxy="flux",
                budget=budget,
            )
        case _:
            raise ValueError(
                f"continuation.field_kind must be one of {LAYER_CORE_FIELD_KINDS!r}; got {kind!r}"
            )
    return LAYER_CORE_FIELD_ROLES[kind], FiniteElementFieldSpec(kind, element)


def layer_core_field_templates(
    cell_kind: str,
    /,
    *,
    kinds: tuple[str, ...] = LAYER_CORE_FIELD_KINDS,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Return mesh-free H1, DG, H(curl), and H(div) transfer declarations."""
    return tuple(_field_template(kind, cell_kind, budget=budget) for kind in kinds)


def _prepared_layer_core_field_templates(
    cell_kind: str, plans: LayerCoreLifecyclePlans, /
) -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Reuse mesh-free declarations after complete paid current-input checks."""
    budget = plans.budget
    if budget is None:
        return layer_core_field_templates(cell_kind)
    with budget.temporary_scope():
        budget.reserve(
            5 + 2 * len(LAYER_CORE_FIELD_KINDS),
            1024 + 128 * len(LAYER_CORE_FIELD_KINDS),
        )
        factories = (
            _field_template, lagrange_element, discontinuous_element,
            form_element, reference_cell_topology,
        )
        roles = tuple(
            (kind, LAYER_CORE_FIELD_ROLES[kind]) for kind in LAYER_CORE_FIELD_KINDS
        )
        cached = plans._field_templates.get(cell_kind)
        if cached is not None:
            if (
                len(cached.factories) != len(factories)
                or any(old() is not new for old, new in zip(cached.factories, factories, strict=True))
                or cached.roles != roles
                or not require_coordinate_identity_inputs(cached.inputs, budget)
            ):
                raise ValueError(
                    "LayerCoreLifecyclePlans.field_templates must retain the original "
                    "declarations, factories and complete current reference banks."
                )
            budget.retain_owner(cached)
            return cached.templates
        templates = layer_core_field_templates(cell_kind, budget=budget)
        inputs = coordinate_identity_inputs(templates, budget)
        if inputs is not None:
            prepared = _PreparedLayerCoreFieldTemplates(
                templates, inputs, tuple(ref(factory) for factory in factories), roles,
            )
            budget.retain_owner(prepared)
            plans._field_templates[cell_kind] = prepared
        return templates


@checked
def layer_core_field_elements(
    result: CellMeshingResult, kind: str, /
) -> dict[str, FiniteElementSpec]:
    if kind not in LAYER_CORE_FIELD_KINDS:
        raise ValueError(
            f"continuation.field_kind must be one of {LAYER_CORE_FIELD_KINDS!r}; got {kind!r}"
        )
    templates = {
        cell_kind: {
            field.name: field
            for _, field in layer_core_field_templates(cell_kind, kinds=(kind,))
        }
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    return {
        block.name: templates[block.cell_kind][kind].elements[0]
        for block in result.mesh.blocks
    }


type _FieldSpace = FiniteElementDiscretization | FiniteElementTransferDiscretization


def _transfer_space(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    reuse_prepared: FiniteElementTransferDiscretization | None = None,
    reuse_source: CellMeshingResult | None = None,
) -> FiniteElementTransferDiscretization:
    if (reuse_prepared is None) != (reuse_source is None):
        raise ValueError(
            f"transfer.reuse_preparation must be supplied together with reuse_source; got {(reuse_prepared is not None, reuse_source is not None)!r}"
        )
    templates = {
        cell_kind: {
            field.name: field
            for _, field in layer_core_field_templates(cell_kind, budget=plans.budget)
        }
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    fields = tuple(
        FiniteElementFieldSpec(
            field.name,
            {
                block.name: templates[block.cell_kind][field.name].elements[0]
                for block in result.mesh.blocks
            },
            component_shape=field.component_shape,
        )
        for field in next(iter(templates.values())).values()
    )
    plan = FiniteElementPlan(
        result.mesh,
        fields,
        coordinate_spec=result.geometry,
        budget=plans.budget,
    )
    retained = plans.transfer_spaces.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError(
                f"transfer.cached_result_id must equal {result.result_id!r}; got {retained_result.result_id!r}"
            )
        retained_space.require_exact_plan(plan, budget=plans.budget)
        return retained_space
    if reuse_source is not None:
        require_restored_layer_core_result_identity(
            reuse_source, result, budget=plans.budget
        )
    prepared = plan.prepare_transfer(
        reuse_prepared=reuse_prepared, budget=plans.budget
    )
    plans.transfer_spaces[result.result_id] = (result, prepared)
    return prepared


type _CellRoute = tuple[int, FiniteElementSpec, np.ndarray, np.ndarray]


def _cells(space: _FieldSpace, /, *, field_name: str | None = None) -> list[_CellRoute]:
    index = 0 if field_name is None else space.field_index(field_name)
    rows: list[_CellRoute] = []
    for block, element, routes, transforms in zip(
        space.mesh.blocks,
        space.elements[index],
        space.dof_maps[index].cell_dofs,
        space.dof_maps[index].cell_transforms,
        strict=True,
    ):
        for identifier, route, transform in zip(
            np.asarray(block.global_ids).tolist(),
            np.asarray(routes),
            np.asarray(transforms),
            strict=True,
        ):
            rows.append((identifier, element, route, transform))
    return rows


@checked
def layer_core_scalar_content(
    space: _FieldSpace,
    values: np.ndarray,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    """Return the physical integral through mapped coordinate Jacobians."""
    index = 0 if field_name is None else space.field_index(field_name)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    runtime = np.asarray(space.default_runtime.coordinates)
    for element, coordinate, routes, geometry_routes in zip(
        space.elements[index],
        space.coordinate_elements,
        space.dof_maps[index].cell_dofs,
        space.coordinate_dofs,
        strict=True,
    ):
        # Refined and restored blocks keep the source basis restricted or
        # composed onto each child; they are tabulated, not reinterpolated.
        if not isinstance(
            coordinate,
            (
                FiniteElementSpec,
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                SplineCellGeometryElement,
            ),
        ):
            raise TypeError(
                f"continuation.coordinate_element must support coordinate tabulation; got {type(coordinate).__name__}"
            )
        points, weights = degree_aware_reference_rule(element.cell_kind, 8)
        basis = np.asarray(element.tabulate(points)[0])
        gradients = np.asarray(coordinate.tabulate(points)[1])
        for route, geometry_route in zip(
            np.asarray(routes), np.asarray(geometry_routes), strict=True
        ):
            determinant = np.linalg.det(
                np.einsum("qna,nd->qda", gradients, runtime[geometry_route])
            )
            if np.any(determinant <= 0.0):
                raise ValueError(
                    f"continuation.jacobian_determinants must be positive; got {determinant!r}"
                )
            total += np.einsum(
                "q,qn,np->p", np.asarray(weights) * determinant, basis, values[route]
            )
    return total


@checked
def layer_core_divergence_content(
    space: _FieldSpace,
    values: np.ndarray,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    """Return total flux content using reference-space contravariant Piola integration."""
    total = np.zeros(values.shape[1:], dtype=np.float64)
    tabulations: dict[
        int, tuple[FiniteElementSpec, np.ndarray, np.ndarray]
    ] = {}
    for _, element, route, transform in _cells(space, field_name=field_name):
        prepared = tabulations.get(id(element))
        if prepared is None:
            points, weights = degree_aware_reference_rule(element.cell_kind, 8)
            divergence = np.trace(
                np.asarray(element.tabulate(points)[1]), axis1=-2, axis2=-1
            )
            prepared = element, np.asarray(weights), divergence
            tabulations[id(element)] = prepared
        _, weights, divergence = prepared
        total += np.einsum(
            "q,qn,np->p", np.asarray(weights), divergence, transform @ values[route]
        )
    return total


def _line_content(
    element: FiniteElementSpec,
    coefficients: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(8)
    points = first + ((nodes + 1.0) / 2.0)[:, None] * (second - first)
    basis = np.asarray(element.tabulate(points)[0])
    return np.einsum("q,qnc,np,c->p", weights / 2.0, basis, coefficients, second - first)


@checked
def layer_core_edge_circulation_defect(
    fine: _FieldSpace,
    coarse: _FieldSpace,
    fine_values: np.ndarray,
    coarse_values: np.ndarray,
    coarsening: MeshAdaptationResult,
    /,
    *,
    field_name: str | None = None,
) -> float:
    """Every coarse reference edge circulation equals its fine sub-edge sum."""
    witnesses = coarsening.coarsening_witnesses
    if witnesses is None:
        raise ValueError(
            "coarsening.coarsening_witnesses must retain nested reference data; got None"
        )
    joins = {
        identifier: (parent, np.asarray(reference))
        for identifier, parent, reference in zip(
            np.asarray(witnesses.fine_cell_ids).tolist(),
            np.asarray(witnesses.coarse_cell_ids).tolist(),
            np.asarray(witnesses.fine_reference_vertices),
            strict=True,
        )
    }
    fine_cells = _cells(fine, field_name=field_name)
    defect = 0.0
    for identifier, element, route, transform in _cells(coarse, field_name=field_name):
        topology = reference_cell_topology(element.cell_kind)
        vertices = np.asarray(topology.vertices)
        children = []
        for child, fine_element, fine_route, fine_transform in fine_cells:
            child_topology = reference_cell_topology(fine_element.cell_kind)
            parent, image = joins.get(child, (child, np.asarray(child_topology.vertices)))
            if parent == identifier:
                children.append((fine_element, fine_route, fine_transform, image))
        if not children:
            raise ValueError(
                f"coarsening.fine_children must be nonempty for coarse cell {identifier}; got {children!r}"
            )
        for first, second in topology.entities[1]:
            start, tangent = vertices[first], vertices[second] - vertices[first]
            expected = np.zeros(fine_values.shape[1:], dtype=np.float64)
            supports: set[tuple[tuple[float, ...], ...]] = set()
            for fine_element, fine_route, fine_transform, image in children:
                local = reference_cell_topology(fine_element.cell_kind)
                local_vertices = np.asarray(local.vertices)
                for a, b in local.entities[1]:
                    endpoints = image[[a, b]]
                    parameters = (endpoints - start) @ tangent / (tangent @ tangent)
                    if (
                        np.max(
                            np.abs(endpoints - (start + parameters[:, None] * tangent))
                        )
                        > 1.0e-12
                        or np.min(parameters) < -1.0e-12
                        or np.max(parameters) > 1.0 + 1.0e-12
                    ):
                        continue
                    support = tuple(sorted(tuple(point.tolist()) for point in endpoints))
                    if support in supports:
                        continue
                    supports.add(support)
                    sign = 1.0 if parameters[1] > parameters[0] else -1.0
                    expected += sign * _line_content(
                        fine_element,
                        fine_transform @ fine_values[fine_route],
                        local_vertices[a],
                        local_vertices[b],
                    )
            actual = _line_content(
                element,
                transform @ coarse_values[route],
                vertices[first],
                vertices[second],
            )
            defect = max(defect, float(np.max(np.abs(actual - expected))))
    return defect


@checked
def layer_core_initial_values(
    space: _FieldSpace,
    kind: str,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    index = 0 if field_name is None else space.field_index(field_name)
    points = np.asarray(space.dof_maps[index].dof_coordinates)
    if kind not in ("H1", "DG"):
        return np.random.default_rng(19).normal(
            size=(space.dof_maps[index].global_dof_count, 2)
        )
    # A physical affine field pulls back through the quadratic fiber graph into
    # H1 Q2 exactly. A physical quadratic would compose to degree four and is
    # not an admissible polynomial-reproduction oracle for this curved geometry.
    return np.column_stack(
        (
            1.0 + points[:, 1] + points[:, 2],
            0.5 + 0.125 * points[:, 1] + 0.1 * points[:, 2],
        )
    )


class _NestedRoute(NamedTuple):
    """The adaptation's own parent charts or coarsening witnesses."""

    parents: np.ndarray | None
    parent_reference_vertices: np.ndarray | None
    parent_reference_homogeneous: np.ndarray | None
    coarsening_witnesses: NestedReferenceWitnesses | None


def _nested_route(adaptation: MeshAdaptationResult, /) -> _NestedRoute:
    if _is_coarsening(adaptation):
        return _NestedRoute(None, None, None, adaptation.coarsening_witnesses)
    return _NestedRoute(
        adaptation.parent_cells,
        adaptation.parent_reference_vertices,
        adaptation.parent_reference_homogeneous,
        None,
    )


def _field_transfers(
    adaptation: MeshAdaptationResult,
    first: FiniteElementTransferDiscretization,
    second: FiniteElementTransferDiscretization,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> dict[str, FiniteElementFieldTransfer]:
    """Prepare every field transfer of one adaptation once per process."""
    route = _nested_route(adaptation)
    transfers: dict[str, FiniteElementFieldTransfer] = {}
    rooted_reference = None
    for kind, role in LAYER_CORE_FIELD_ROLES.items():
        if role not in ("intensive", "conservative-density"):
            continue
        key = (adaptation.result_id, kind)
        retained = plans.field_transfers.get(key)
        if retained is None:
            transfer, reference = prepare_shared_nested_field_transfer(
                first,
                second,
                route.parents,
                field_name=kind,
                parent_reference_vertices=route.parent_reference_vertices,
                parent_reference_homogeneous=route.parent_reference_homogeneous,
                source_geometry=adaptation.source.geometry,
                target_geometry=adaptation.target.geometry,
                geometry_transition=adaptation.geometry_transition,
                coarsening_witnesses=route.coarsening_witnesses,
                budget=plans.budget,
            )
            if kind == "DG":
                rooted_reference = reference
            transfers[kind] = transfer
        else:
            transfers[kind] = retained
        plans.field_transfers[key] = transfers[kind]
    compatible = tuple(
        kind
        for kind, role in LAYER_CORE_FIELD_ROLES.items()
        if role == "compatible-cochain"
    )
    retained_compatible = tuple(
        plans.field_transfers.get((adaptation.result_id, kind)) for kind in compatible
    )
    prepared = (
        prepare_nested_compatible_field_transfers(
            first,
            second,
            route.parents,
            field_names=compatible,
            parent_reference_vertices=route.parent_reference_vertices,
            parent_reference_homogeneous=route.parent_reference_homogeneous,
            source_geometry=adaptation.source.geometry,
            target_geometry=adaptation.target.geometry,
            geometry_transition=adaptation.geometry_transition,
            # The geometry transition already owns the full inverse witnesses.
            # Use the same binding as DG's rooted preparation, rather than
            # redundantly passing those witnesses as a second input.
            coarsening_witnesses=None,
            nested_reference=rooted_reference,
            budget=plans.budget,
        )
        if any(value is None for value in retained_compatible)
        else retained_compatible
    )
    for kind, transfer in zip(compatible, prepared, strict=True):
        if transfer is None:
            raise ValueError(
                f"continuation.compatible_transfer must be prepared for {kind}; got None"
            )
        transfers[kind] = transfer
        plans.field_transfers[(adaptation.result_id, kind)] = transfer
    return transfers


def _field_physical_error(
    kind: str,
    adaptation: MeshAdaptationResult,
    first: FiniteElementTransferDiscretization,
    second: FiniteElementTransferDiscretization,
    transfer: FiniteElementFieldTransfer,
    source_values: np.ndarray,
    target_values: np.ndarray,
    /,
) -> tuple[float | None, str]:
    """Physical evidence of one transferred field against its own invariant."""
    if kind == "H1":
        # The accepted H1 state is the authored affine field; mapped Q2
        # reproduces it exactly on every nested carrier.
        expected = layer_core_initial_values(second, kind, field_name=kind)[:, 0]
        return (
            float(np.max(np.abs(target_values - expected))),
            "affine-H1Q2-reproduction",
        )
    if _is_coarsening(adaptation):
        # The inverse is a content-preserving projection of every fine field.
        arbitrary = np.random.default_rng(47).normal(
            size=(first.dof_maps[first.field_index(kind)].global_dof_count, 2)
        )
        projected = np.asarray(transfer.transfer.apply(arbitrary))
        match kind:
            case "DG":
                return (
                    float(
                        np.max(
                            np.abs(
                                layer_core_scalar_content(
                                    second, projected, field_name=kind
                                )
                                - layer_core_scalar_content(
                                    first, arbitrary, field_name=kind
                                )
                            )
                        )
                    ),
                    "arbitrary-fine-scalar-content-projection",
                )
            case "Hdiv":
                return (
                    float(
                        np.max(
                            np.abs(
                                layer_core_divergence_content(
                                    second, projected, field_name=kind
                                )
                                - layer_core_divergence_content(
                                    first, arbitrary, field_name=kind
                                )
                            )
                        )
                    ),
                    "arbitrary-fine-flux-content-projection",
                )
            case _:
                return (
                    layer_core_edge_circulation_defect(
                        first,
                        second,
                        arbitrary,
                        projected,
                        adaptation,
                        field_name=kind,
                    ),
                    "arbitrary-fine-edge-circulation-projection",
                )
    match kind:
        case "DG":
            return (
                float(
                    np.max(
                        np.abs(
                            layer_core_scalar_content(
                                second, target_values[:, None], field_name=kind
                            )
                            - layer_core_scalar_content(
                                first, source_values[:, None], field_name=kind
                            )
                        )
                    )
                ),
                "accepted-scalar-content-prolongation",
            )
        case "Hdiv":
            return (
                float(
                    np.max(
                        np.abs(
                            layer_core_divergence_content(
                                second, target_values[:, None], field_name=kind
                            )
                            - layer_core_divergence_content(
                                first, source_values[:, None], field_name=kind
                            )
                        )
                    )
                ),
                "accepted-flux-content-prolongation",
            )
        case _:
            # Edge circulation splitting is witnessed by coarsening charts only.
            return None, "edge-circulation-witnessed-on-inverse-coarsening"


@checked
def continue_layer_core_fields(
    values: tuple[Array, ...],
    adaptation: MeshAdaptationResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    restores: CellMeshingResult | None = None,
) -> tuple[tuple[Array, ...], list[LayerCoreFieldContinuation]]:
    """Move accepted H1/DG/H(curl)/H(div) coefficients across one archived adaptation.

    ``values`` are the accepted coefficients on ``adaptation.source`` in
    ``LAYER_CORE_FIELD_KINDS`` order. ``restores`` names an already prepared
    carrier that the target restores bitwise; its transfer preparation is then
    reused after the restored identity is proved.
    """
    first = _transfer_space(adaptation.source, plans=plans)
    reused = None if restores is None else plans.transfer_spaces.get(restores.result_id)
    second = _transfer_space(
        adaptation.target,
        plans=plans,
        reuse_prepared=None if reused is None else reused[1],
        reuse_source=None if reused is None else restores,
    )
    transfers = _field_transfers(adaptation, first, second, plans=plans)
    continued: list[Array] = []
    summaries: list[LayerCoreFieldContinuation] = []
    for kind, value in zip(LAYER_CORE_FIELD_KINDS, values, strict=True):
        transfer = transfers[kind]
        if not transfer.evidence.passed:
            raise ValueError(
                f"continuation.transfer_evidence must pass for {kind}; got {dict(transfer.evidence.defects)!r}"
            )
        source_values = np.asarray(value)
        target_values = np.asarray(transfer.transfer.apply(source_values))
        dual = np.random.default_rng(29).normal(size=target_values.shape)
        adjoint = (
            float(np.vdot(target_values, dual)),
            float(np.vdot(source_values, np.asarray(transfer.transfer.pullback(dual)))),
        )
        np.testing.assert_allclose(*adjoint, rtol=2.0e-11, atol=2.0e-11)
        compiled = np.asarray(
            eqx.filter_jit(transfer.transfer.apply)(jnp.asarray(source_values))
        )
        np.testing.assert_allclose(compiled, target_values, rtol=2.0e-13, atol=2.0e-13)
        physical, scope = _field_physical_error(
            kind, adaptation, first, second, transfer, source_values, target_values
        )
        if physical is not None and physical > 3.0e-11:
            raise ValueError(
                f"continuation.physical_defect must be at most 3.0e-11 for {kind}; got {physical:.3e} ({scope})"
            )
        summaries.append(
            {
                "field": kind,
                "dofs": [
                    first.dof_maps[first.field_index(kind)].global_dof_count,
                    second.dof_maps[second.field_index(kind)].global_dof_count,
                ],
                "adjoint_error": abs(adjoint[0] - adjoint[1]),
                "compiled_error": float(np.max(np.abs(compiled - target_values))),
                "physical_error": physical,
                "physical_scope": scope,
                "defects": dict(transfer.evidence.defects),
            }
        )
        continued.append(jnp.asarray(target_values))
    return tuple(continued), summaries


def _finite_volume(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> UnstructuredFiniteVolumeDiscretization:
    retained = plans.finite_volumes.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError(
                f"finite_volume.cached_result_id must equal {result.result_id!r}; got {retained_result.result_id!r}"
            )
        return retained_space
    report = result.certification
    if report is None or report.embedding is None:
        raise ValueError(
            f"finite_volume.embedding must be present in certification; got {report!r}"
        )
    periodic = result.mesh.periodic_topology is not None
    periodic_geometry = result.geometry if periodic else None
    periodic_embedding = report.embedding if periodic else None
    template = layer_core_finite_volume_field_templates()[0]
    space = UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        field_name=template["name"],
        component_names=template["component_names"],
        periodic_geometry=periodic_geometry,
        periodic_embedding=periodic_embedding,
        budget=plans.budget,
    ).prepare(
        cell_geometry=result.geometry,
        periodic_embedding=periodic_embedding,
        budget=plans.budget,
    )
    plans.finite_volumes[result.result_id] = (result, space)
    return space


@checked
def prepare_layer_core_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    stage: MeshAdaptationResult,
    provenance: str,
    /,
    *,
    plans: LayerCoreLifecyclePlans | None = None,
) -> UnstructuredConservativeRemapPlan:
    plans_ = LayerCoreLifecyclePlans() if plans is None else plans
    key = (stage.result_id, provenance)
    if key in plans_.remaps:
        plan = plans_.remaps[key]
        if (
            plan.source_topology_id != source.topology_id
            or plan.source_geometry_id != source.geometry_id
            or plan.target_topology_id != target.topology_id
            or plan.target_geometry_id != target.geometry_id
        ):
            raise ValueError(
                f"remap.cached_epoch must match source and target topology and geometry; got {(plan.source_topology_id, plan.source_geometry_id, plan.target_topology_id, plan.target_geometry_id)!r} for {(source.topology_id, source.geometry_id, target.topology_id, target.geometry_id)!r}"
            )
        return plan
    remap = prepare_unstructured_conservative_remap(
        source,
        target,
        provenance=provenance,
        source_geometry=stage.source.geometry,
        target_geometry=stage.target.geometry,
        geometry_transition=stage.geometry_transition,
        budget=plans_.budget,
    )
    if remap.plan is None:
        raise ValueError(
            f"remap.plan must be accepted for {provenance!r}; got {remap.status!r}: {remap.reason}"
        )
    plans_.remaps[key] = remap.plan
    return remap.plan


def _inventories(
    volumes: np.ndarray,
    densities: np.ndarray,
    gids: np.ndarray,
    regions: Mapping[int, int],
    region_count: int,
    /,
) -> np.ndarray:
    labels = np.asarray([regions[int(identifier)] for identifier in gids], dtype=np.int64)
    return np.bincount(labels, weights=volumes * densities, minlength=region_count)


@checked
def layer_core_initial_density(
    result: CellMeshingResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
) -> Array:
    """The declared material density as finite-volume cell averages, in FV order."""
    densities = layer_core_material_values(result, material)
    regions = layer_core_cell_regions(result)
    space = _finite_volume(result, plans=plans)
    return jnp.asarray(
        [
            densities[regions[int(identifier)]]
            for identifier in np.asarray(space.cell_global_ids)
        ],
        dtype=jnp.float64,
    )


@checked
def continue_layer_core_finite_volume(
    density: Array,
    adaptation: MeshAdaptationResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
) -> tuple[Array, LayerCoreVolumeContinuation]:
    """Conservatively remap the accepted material density across one adaptation."""
    region_count = layer_core_material_values(adaptation.source, material).size
    layer_core_material_values(adaptation.target, material)
    first = _finite_volume(adaptation.source, plans=plans)
    second = _finite_volume(adaptation.target, plans=plans)
    remap = prepare_layer_core_remap(
        first,
        second,
        adaptation,
        "layer-core-inverse-coarsening"
        if _is_coarsening(adaptation)
        else "layer-core-refinement",
        plans=plans,
    )
    gids = [
        np.asarray(space.cell_global_ids, dtype=np.int64) for space in (first, second)
    ]
    regions = [
        layer_core_cell_regions(result)
        for result in (adaptation.source, adaptation.target)
    ]
    volumes = [
        np.asarray(space.cell_volumes, dtype=np.float64) for space in (first, second)
    ]
    source_density = np.asarray(density, dtype=np.float64)
    if source_density.shape != volumes[0].shape:
        raise ValueError(
            f"finite_volume.density must hold one value per source cell {volumes[0].shape!r}; got {source_density.shape!r}"
        )
    content = (volumes[0] * source_density)[:, None]
    target_content = np.asarray(remap.apply_content(jnp.asarray(content)))
    target_density = target_content[:, 0] / volumes[1]
    source_inventories = _inventories(
        volumes[0], source_density, gids[0], regions[0], region_count
    )
    target_inventories = _inventories(
        volumes[1], target_density, gids[1], regions[1], region_count
    )
    arbitrary = (
        np.random.default_rng(53).uniform(1.0, 4.0, size=(volumes[0].size, 1))
        * volumes[0][:, None]
    )
    defects = {
        "region_inventory": float(
            np.max(np.abs(target_inventories - source_inventories))
        ),
        "arbitrary_inventory": float(
            abs(
                np.sum(np.asarray(remap.apply_content(jnp.asarray(arbitrary))))
                - np.sum(arbitrary)
            )
        ),
    }
    scale = float(np.sum(np.abs(content)))
    if max(defects.values()) > 1.0e-13 * max(1.0, scale):
        raise ValueError(
            f"finite_volume.conservation_defects must be at most {1.0e-13 * max(1.0, scale)!r}; got {defects!r}"
        )
    return jnp.asarray(target_density), {
        "cells": [value.size for value in gids],
        "region_inventories": target_inventories.tolist(),
        "defects": defects,
    }


@final
class _HarmonicBoundary(StrictModule, NonTrainableState):
    offset: Array

    def __init__(self, offset: Array) -> None:
        self.offset = offset

    def __call__(self, points: Array) -> Array:
        return points[..., 1] + points[..., 2] + self.offset


def _damage(points: Array, args: object) -> Array:
    """A bounded physical affine history represented exactly by mapped Q2."""
    del args
    return 0.5 + 0.125 * points[..., 1] + 0.1 * points[..., 2]


@eqx.filter_jit
def _harmonic_solution(
    problem: CompiledFiniteElementProblem,
    operator: LinearSystem,
    rhs: object,
    /,
) -> tuple[Array, Array]:
    """Factor, solve and expand one assembled harmonic diffusion system."""
    result = solve(operator, rhs, policy=LinearSolvePolicy(DenseLU()))
    return problem.expand(result.value), result.successful


def _solve_diffusion(
    space: FiniteElementDiscretization,
    offset: float,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    compiled: bool,
) -> tuple[_HarmonicSolve, bool]:
    """Return the retained solve, or assemble and solve it once.

    Accepted-state solves are lowered and compiled ahead of time for their
    epoch evidence; changed-PDE refusal witnesses execute eagerly. The boolean
    reports whether this call compiled a program, so an epoch that reuses a
    restored source space never claims a second compilation.
    """
    key = (space.prepared_id, float(offset))
    retained = plans.diffusion_solutions.get(key)
    if retained is not None:
        retained.space.require_exact_plan(
            space.construction_plan,
            numeric_version=space.numeric_version,
            budget=plans.budget,
        )
        if compiled and retained.executable is None:
            raise ValueError(
                f"diffusion.retained_solve must be compiled for an accepted epoch; got an eager witness for {key!r}"
            )
        return retained, False
    problem = compile_finite_element_problem(
        FiniteElementForm(
            "layer-core-harmonic-diffusion",
            "u",
            (DiffusionAction("u", 1.0),),
        ),
        space,
        constraint=dirichlet_constraint(space, "u"),
        dirichlet_values=_HarmonicBoundary(jnp.asarray(offset, dtype=jnp.float64)),
        budget=plans.budget,
    )
    started = perf_counter()
    operator, rhs = problem.linear_system()
    stages = {"solver_assembly": perf_counter() - started}
    executable = None
    memory = None
    if compiled:
        started = perf_counter()
        # Equinox's decorator typing omits its runtime lowering interface.
        lowered = _harmonic_solution.lower(  # ty: ignore[unresolved-attribute]
            problem, operator, rhs
        )
        stages["solver_lowering"] = perf_counter() - started
        started = perf_counter()
        executable = lowered.compile()
        stages["solver_compile"] = perf_counter() - started
        memory = compiled_memory_estimate(executable.compiled)
        started = perf_counter()
        values, successful = jax.block_until_ready(executable(problem, operator, rhs))
        stages["solver_first_execution"] = perf_counter() - started
    else:
        result = solve(operator, rhs, policy=LinearSolvePolicy(DenseLU()))
        values, successful = problem.expand(result.value), result.successful
    if not bool(jnp.all(successful)):
        raise ValueError(
            f"diffusion.solve_successful must be true for every solve; got {successful!r}"
        )
    error = float(
        jnp.max(
            jnp.abs(
                values
                - _HarmonicBoundary(jnp.asarray(offset, dtype=jnp.float64))(
                    space.dof_maps[0].dof_coordinates
                )
            )
        )
    )
    if error > 1.0e-9:
        raise ValueError(
            f"diffusion.harmonic_error must be at most 1.0e-9; got {error:.3e}"
        )
    harmonic = _HarmonicSolve(
        space, problem, (operator, rhs), values, executable, memory, stages
    )
    plans.diffusion_solutions[key] = harmonic
    return harmonic, compiled


def _published_result_identity(result: CellMeshingResult, /) -> LayerCoreResultIdentity:
    """Identify a publication with its declared resource guarantees and receipt."""
    return {
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "vertices": int(result.mesh.coordinates.shape[0]),
        "cells": int(sum(block.cell_count for block in result.mesh.blocks)),
        "audit_passed": bool(result.audit.passed),
        "resource_guarantees": {
            "enforced": list(result.runtime.enforced_limits),
            "unenforced": list(result.runtime.unenforced_limits),
        },
        "resource_binding": native_publication_resource_binding(
            result.result_id, result.runtime.runtime_id, result.execution_evidence,
            enforced_limits=result.runtime.enforced_limits,
            unenforced_limits=result.runtime.unenforced_limits,
        ),
        "execution_evidence": (
            None
            if result.execution_evidence is None
            else result.execution_evidence.to_record()
        ),
    }


def _harmonic_epoch(
    epoch: int,
    name: str,
    result: CellMeshingResult,
    harmonic: _HarmonicSolve,
    /,
    *,
    compiled: bool,
    repeats: int,
) -> LayerCoreSolveEpoch:
    """Measure warm executions of the accepted state's compiled solve."""
    operator, rhs = harmonic.system
    if harmonic.executable is None:
        raise ValueError(
            "diffusion.epoch_solve must be ahead-of-time compiled; got an eager witness"
        )
    executable = harmonic.executable
    warm: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        _, successful = jax.block_until_ready(executable(harmonic.problem, operator, rhs))
        warm.append(perf_counter() - started)
        if not bool(jnp.all(successful)):
            raise ValueError(
                f"diffusion.warm_solve_successful must be true for every execution; got {successful!r}"
            )
    memory = harmonic.memory
    stages = dict(harmonic.stages_seconds) if compiled else {}
    stages["solver_warm_execution"] = float(np.median(warm))
    return {
        "epoch": epoch,
        "name": name,
        "result": _published_result_identity(result),
        "dofs": int(harmonic.space.dof_maps[0].global_dof_count),
        "compiler": {
            "temporary_bytes": None if memory is None else memory.temporary_bytes,
            "output_bytes": None if memory is None else memory.output_bytes,
            "argument_bytes": None if memory is None else memory.argument_bytes,
            "generated_code_bytes": (
                None if memory is None else memory.generated_code_bytes
            ),
            "memory_status": "unavailable" if memory is None else "measured",
            "compilation_count": 1 if compiled else 0,
            "compilation_count_scope": "explicit-AOT-compile-invocations",
            "warm_samples_seconds": warm,
            "warm_recompilation_count": 0,
            "execution_scope": (
                "accepted-epoch-assembled-harmonic-dense-LU-solve-and-expand"
                if compiled
                else "accepted-epoch-restored-source-program-dense-LU-solve-and-expand"
            ),
            "assembly_binding_id": harmonic.problem.compilation_id,
        },
        "stages_seconds": stages,
        "retained_bytes": model_array_bytes(
            (result, harmonic.space, harmonic.problem, harmonic.values)
        ),
    }


def _original_scalar_exact(points: Array) -> Array:
    """The unchanged unit-box P1 harmonic target, not the Q2 history target."""
    return points[..., 0] + points[..., 1] + points[..., 2]


def layer_core_original_scalar_field_template(
    cell_kind: str, /
) -> FiniteElementFieldSpec:
    """The concrete supplemental declaration of the original unit-box P1 solve."""
    return FiniteElementFieldSpec("original-u", lagrange_element(cell_kind, 1))


def _original_scalar_space(
    result: CellMeshingResult, /, *, plans: LayerCoreLifecyclePlans
) -> FiniteElementDiscretization:
    field = FiniteElementFieldSpec(
        "original-u",
        {
            block.name: layer_core_original_scalar_field_template(
                block.cell_kind
            ).elements[0]
            for block in result.mesh.blocks
        },
    )
    plan = FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry, budget=plans.budget
    )
    retained = plans.original_scalar_spaces.get(result.result_id)
    if retained is not None:
        retained[1].require_exact_plan(plan, budget=plans.budget)
        return retained[1]
    prepared = plan.prepare(budget=plans.budget)
    plans.original_scalar_spaces[result.result_id] = (result, prepared)
    return prepared


def _original_scalar_physical(
    result: CellMeshingResult,
    values: Array,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> dict[str, float]:
    space = _original_scalar_space(result, plans=plans)
    expected_shape = (space.dof_maps[0].global_dof_count,)
    if values.shape != expected_shape:
        raise ValueError(
            f"original_scalar.shape must equal the actual P1 layout {expected_shape!r}; got {values.shape!r}"
        )
    squared, integral, volume = 0.0, 0.0, 0.0
    for geometry, dofs in zip(
        space.evaluate_geometry("original-u", result.geometry.coordinates),
        space.dof_maps[0].cell_dofs,
        strict=True,
    ):
        evaluated = np.asarray(values)[np.asarray(dofs)] @ np.asarray(
            geometry.basis_values
        ).T
        weights = np.asarray(geometry.physical_weights, dtype=np.float64)
        exact = np.asarray(_original_scalar_exact(geometry.physical_points))
        squared += float(np.sum(weights * (evaluated - exact) ** 2))
        integral += float(np.sum(weights * evaluated))
        volume += float(np.sum(weights))
    physical = {
        "l2_error": float(np.sqrt(max(0.0, squared))),
        "state_integral": integral,
        "mapped_volume": volume,
        "analytic_volume_defect": volume - 1.0,
        "nodal_maximum_error": float(
            np.max(
                np.abs(
                    np.asarray(values)
                    - np.asarray(
                        _original_scalar_exact(space.dof_maps[0].dof_coordinates)
                    )
                )
            )
        ),
    }
    if (
        not all(np.isfinite(value) for value in physical.values())
        or abs(physical["analytic_volume_defect"]) > 1.0e-10
        or physical["nodal_maximum_error"] > 1.0e-9
    ):
        raise ValueError(
            f"original_scalar.physical must retain the analytic unit-box P1 target and volume; got {physical!r}"
        )
    return physical


def _original_scalar_independent_solve(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    epoch: int,
    repeats: int,
) -> tuple[Array, LayerCoreSolveEpoch]:
    """Solve the archived original P1 problem independently in the cold process."""
    space = _original_scalar_space(result, plans=plans)
    problem = compile_finite_element_problem(
        FiniteElementForm(
            "unit-box-original-P1-diffusion",
            "original-u",
            (DiffusionAction("original-u", 1.0),),
        ),
        space,
        constraint=dirichlet_constraint(space, "original-u"),
        dirichlet_values=_original_scalar_exact,
        budget=plans.budget,
    )
    started = perf_counter()
    operator, rhs = problem.linear_system()
    stages = {"solver_assembly": perf_counter() - started}
    started = perf_counter()
    lowered = _harmonic_solution.lower(  # ty: ignore[unresolved-attribute]
        problem, operator, rhs
    )
    stages["solver_lowering"] = perf_counter() - started
    started = perf_counter()
    executable = lowered.compile()
    stages["solver_compile"] = perf_counter() - started
    started = perf_counter()
    values, successful = jax.block_until_ready(executable(problem, operator, rhs))
    stages["solver_first_execution"] = perf_counter() - started
    if not bool(jnp.all(successful)):
        raise ValueError(
            "original_scalar.independent_solve must succeed; got an unsuccessful solve"
        )
    harmonic = _HarmonicSolve(
        space,
        problem,
        (operator, rhs),
        values,
        executable,
        compiled_memory_estimate(executable.compiled),
        stages,
    )
    return values, _harmonic_epoch(
        epoch, "original-u", result, harmonic, compiled=True, repeats=repeats
    )


def _continue_original_scalar(
    state: LayerCoreAcceptedState,
    adaptation: MeshAdaptationResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    accepted_values: Array | None,
    repeats: int,
) -> tuple[Array, dict[str, object]]:
    """Transfer the actual original bank; retain warm solves or solve independently cold."""
    if state.original_scalar is None:
        raise ValueError("original_scalar.source must retain its actual accepted bank")
    first = _original_scalar_space(adaptation.source, plans=plans)
    second = _original_scalar_space(adaptation.target, plans=plans)
    before = _original_scalar_physical(
        adaptation.source, state.original_scalar, plans=plans
    )
    coarsening = _is_coarsening(adaptation)
    transfer = prepare_nested_field_transfer(
        first,
        second,
        None if coarsening else adaptation.parent_cells,
        field_name="original-u",
        parent_reference_vertices=(
            None if coarsening else adaptation.parent_reference_vertices
        ),
        parent_reference_homogeneous=(
            None if coarsening else adaptation.parent_reference_homogeneous
        ),
        source_geometry=adaptation.source.geometry,
        target_geometry=adaptation.target.geometry,
        geometry_transition=adaptation.geometry_transition,
        budget=plans.budget,
    )
    if not transfer.evidence.passed:
        raise ValueError("original_scalar.transfer must pass its nested P1 certificate")
    carried = jax.block_until_ready(transfer.transfer.apply(state.original_scalar))
    physical = _original_scalar_physical(adaptation.target, carried, plans=plans)
    inventory_defect = physical["state_integral"] - before["state_integral"]
    if abs(inventory_defect) > 1.0e-9:
        raise ValueError(
            f"original_scalar.inventory_defect must be at most 1.0e-9; got {inventory_defect!r}"
        )
    independent_solve: LayerCoreSolveEpoch | None = None
    if accepted_values is None:
        compared, independent_solve = _original_scalar_independent_solve(
            adaptation.target, plans=plans, epoch=state.epoch + 1, repeats=repeats
        )
    else:
        compared = jnp.asarray(accepted_values)
    compared_physical = _original_scalar_physical(
        adaptation.target, compared, plans=plans
    )
    defect = float(jnp.max(jnp.abs(carried - compared)))
    if not np.isfinite(defect) or defect > 1.0e-9:
        raise ValueError(
            f"original_scalar.continuation must equal the actual warm or independent cold solve within 1.0e-9; got {defect!r}"
        )
    # Warm checkpoints retain the exact arrays produced by the original solver;
    # cold checkpoints retain the actual transfer, checked by an independent solve.
    accepted = carried if accepted_values is None else compared
    return accepted, {
        "field": "original-u",
        "degree": 1,
        "dofs": [
            first.dof_maps[0].global_dof_count, second.dof_maps[0].global_dof_count
        ],
        "physical": physical if accepted_values is None else compared_physical,
        "state_inventory_defect": inventory_defect,
        "solve_comparison_defect": defect,
        "accepted_bank_origin": (
            "actual-archived-P1-state-transfer"
            if accepted_values is None
            else "actual-original-warm-P1-solver-values"
        ),
        "independent_solve": independent_solve,
        "independent_physical": (
            compared_physical if accepted_values is None else None
        ),
    }


def _original_scalar_comparison(
    state: LayerCoreAcceptedState,
    reference: MeshingAcceptedEpoch,
    /,
) -> dict[str, float]:
    present = "fe/original-u" in reference.fields
    if (state.original_scalar is not None) != present:
        raise ValueError(
            "original_scalar.comparison must retain the same declared bank on both states"
        )
    if state.original_scalar is None:
        return {}
    archived = jnp.asarray(reference.fields["fe/original-u"])
    if state.original_scalar.shape != archived.shape:
        raise ValueError(
            f"original_scalar.comparison shapes must agree; got {state.original_scalar.shape!r} and {archived.shape!r}"
        )
    defect = float(jnp.max(jnp.abs(state.original_scalar - archived)))
    if not np.isfinite(defect) or defect > 1.0e-9:
        raise ValueError(
            f"original_scalar.warm_cold_defect must be at most 1.0e-9; got {defect!r}"
        )
    return {"original-u": defect}


@final
class _DensityTransport(StrictModule, NonTrainableState):
    plan: UnstructuredConservativeRemapPlan
    source_volumes: Array
    target_volumes: Array
    source_gather: Array
    target_gather: Array
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UnstructuredConservativeRemapPlan,
        source_volumes: Array,
        target_volumes: Array,
        source_gather: Array,
        target_gather: Array,
        site_id: str,
    ) -> None:
        self.plan = plan
        self.source_volumes = source_volumes
        self.target_volumes = target_volumes
        self.source_gather = source_gather
        self.target_gather = target_gather
        self.site_id = site_id

    def __call__(
        self, before: MaterialTransaction, lineage: MeshLineage, args: object
    ) -> MaterialTransaction:
        del lineage, args
        state = before.state(self.site_id)
        content = self.source_volumes * state.committed[self.source_gather]
        transferred = (
            self.plan.apply_content(content[:, None])[:, 0] / self.target_volumes
        )
        return MaterialTransaction(
            (
                MaterialState(
                    state.site_id,
                    state.model_id,
                    transferred[self.target_gather],
                    state_version=state.state_version + 1,
                ),
            )
        )


@final
class _PhysicalCertification(StrictModule, NonTrainableState):
    expected_solution: Array
    expected_damage: Array
    measures: Array
    region_labels: Array
    inventories: Array
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        solution: Array,
        damage: Array,
        measures: Array,
        regions: Array,
        inventories: Array,
        site_id: str,
    ) -> None:
        self.expected_solution = solution
        self.expected_damage = damage
        self.measures = measures
        self.region_labels = regions
        self.inventories = inventories
        self.site_id = site_id

    def __call__(
        self,
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        lineage: MeshLineage,
        args: object,
    ) -> bool:
        del mesh, lineage, args
        if materials is None:
            return False
        inventory = layer_core_region_inventory(
            self.measures * materials.state(self.site_id).committed,
            self.region_labels,
            self.inventories.shape[0],
        )
        solution_defect = jnp.max(jnp.abs(fields[0] - self.expected_solution))
        damage_defect = jnp.max(jnp.abs(fields[1] - self.expected_damage))
        inventory_defect = jnp.max(jnp.abs(inventory - self.inventories))
        return bool(
            jnp.all(jnp.isfinite(fields[0]))
            & jnp.all(jnp.isfinite(fields[1]))
            & (solution_defect <= 1.0e-9)
            & (damage_defect <= 1.0e-9)
            & jnp.all((fields[1] >= 0.0) & (fields[1] <= 1.0))
            & (inventory_defect <= 1.0e-12)
        )


@checked
def layer_core_cell_measures(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> np.ndarray:
    """Certified mapped cell measures in mesh block order."""
    space = _finite_volume(result, plans=plans)
    rows = dict(
        zip(
            np.asarray(space.cell_global_ids).tolist(),
            np.asarray(space.cell_volumes).tolist(),
            strict=True,
        )
    )
    return np.asarray(
        [rows[identifier] for identifier in _cell_ids(result).tolist()], dtype=np.float64
    )


@checked
def layer_core_region_labels(result: CellMeshingResult, /) -> np.ndarray:
    report = result.certification
    if report is None or report.request.cell_regions is None:
        raise ValueError(
            f"continuation.cell_regions must be present in certification; got {report!r}"
        )
    cell_ids = _cell_ids(result)
    if not np.array_equal(np.asarray(report.request.cell_global_ids), cell_ids):
        raise ValueError(
            f"material.cell_global_ids must equal mesh block cell order {tuple(np.asarray(cell_ids).tolist())!r}; got {tuple(np.asarray(report.request.cell_global_ids).tolist())!r}"
        )
    return np.asarray(report.request.cell_regions, dtype=np.int64)


def layer_core_region_inventory(
    content: Array, labels: Array | np.ndarray, region_count: int, /
) -> Array:
    return (
        jnp.zeros((region_count,), dtype=jnp.float64)
        .at[jnp.asarray(labels, dtype=jnp.int32)]
        .add(content)
    )


def _density_transport(
    adaptation: MeshAdaptationResult,
    site_id: str,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> _DensityTransport:
    first = _finite_volume(adaptation.source, plans=plans)
    second = _finite_volume(adaptation.target, plans=plans)
    provenance = (
        "layer-core-inverse-coarsening"
        if _is_coarsening(adaptation)
        else "layer-core-refinement"
    )
    plan = prepare_layer_core_remap(
        first,
        second,
        adaptation,
        provenance,
        plans=plans,
    )
    source_rows = {
        identifier: row
        for row, identifier in enumerate(_cell_ids(adaptation.source).tolist())
    }
    source_gather = np.asarray(
        [
            source_rows[int(identifier)]
            for identifier in np.asarray(first.cell_global_ids)
        ],
        dtype=np.int64,
    )
    target_gather = _finite_volume_rows(adaptation.target, plans=plans)
    constant = plan.apply_content(first.cell_volumes[:, None])[:, 0] / second.cell_volumes
    if not np.allclose(constant, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"material.remapped_constant must equal one within 1.0e-12; got {constant!r}"
        )
    return _DensityTransport(
        plan,
        first.cell_volumes,
        second.cell_volumes,
        jnp.asarray(source_gather, dtype=jnp.int32),
        jnp.asarray(target_gather, dtype=jnp.int32),
        site_id,
    )


def _finite_volume_rows(
    result: CellMeshingResult, /, *, plans: LayerCoreLifecyclePlans
) -> np.ndarray:
    """The finite-volume row of every mesh cell, in mesh block order."""
    space = _finite_volume(result, plans=plans)
    rows = {
        identifier: row
        for row, identifier in enumerate(np.asarray(space.cell_global_ids).tolist())
    }
    return np.asarray(
        [rows[int(identifier)] for identifier in _cell_ids(result)], dtype=np.int64
    )


def _same_coordinate_map(
    source: CellMeshingResult,
    restored: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> bool:
    """Bitwise-equal elements, routes, coefficients and periodic source map.

    The restored geometry additionally records the coarsening restriction that
    produced it; that provenance is not part of the coordinate map.
    """
    certification = source.certification
    if certification is None:
        raise ValueError(
            "qualification.coordinate_map_certificate must retain source request limits; got None"
        )
    source_geometry, restored_geometry = (source.geometry, restored.geometry)
    with coordinate_certificate_request(
        certification.request.limits, budget=budget
    ) as budget:
        return (
            source_geometry.block_names == restored_geometry.block_names
            and all(
                (
                    first.element_id == second.element_id
                    for first, second in zip(
                        source_geometry.elements, restored_geometry.elements, strict=True
                    )
                )
            )
            and all(
                (
                    np.array_equal(np.asarray(first), np.asarray(second))
                    for first, second in zip(
                        source_geometry.geometry_dofs,
                        restored_geometry.geometry_dofs,
                        strict=True,
                    )
                )
            )
            and np.array_equal(
                np.asarray(source_geometry.coordinates).view(np.uint64),
                np.asarray(restored_geometry.coordinates).view(np.uint64),
            )
            and (
                source_geometry.source_coordinates(budget=budget)
                == restored_geometry.source_coordinates(budget=budget)
            )
            and bool(
                eqx.tree_equal(
                    source_geometry.periodic_source, restored_geometry.periodic_source
                )
            )
        )


@checked
def require_restored_layer_core_result_identity(
    source: CellMeshingResult,
    restored: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    source_report = source.certification
    restored_report = restored.certification
    if source_report is None or restored_report is None:
        raise ValueError(
            f"restoration.certification must be present for source and restored results; got {(source_report, restored_report)!r}"
        )
    source_request = source_report.request
    restored_request = restored_report.request
    if (
        source.mesh.topology_id != restored.mesh.topology_id
        or tuple(block.block_id for block in source.mesh.blocks)
        != tuple(block.block_id for block in restored.mesh.blocks)
        or not np.array_equal(
            np.asarray(source.mesh.vertex_global_ids),
            np.asarray(restored.mesh.vertex_global_ids),
        )
        or not np.array_equal(
            np.asarray(source.mesh.coordinates).view(np.uint64),
            np.asarray(restored.mesh.coordinates).view(np.uint64),
        )
        or (
            source_request.source_id,
            source_request.source_revision,
            source_request.domain_id,
        )
        != (
            restored_request.source_id,
            restored_request.source_revision,
            restored_request.domain_id,
        )
        or not _same_coordinate_map(source, restored, budget=budget)
    ):
        raise ValueError(
            f"restoration.result_identity must preserve mesh, coordinate map, and source revision; got restored result {restored.result_id!r} for source {source.result_id!r}"
        )


def layer_core_history_field_templates(
    cell_kind: str, /
) -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Return the quadratic diffusion and irreversible damage-history declarations."""
    element = lagrange_element(cell_kind, 2)
    return (
        ("history:intensive", FiniteElementFieldSpec("u", element)),
        (
            "history:material-history",
            FiniteElementFieldSpec("damage-history", element),
        ),
    )


def _history_fields(result: CellMeshingResult, /) -> tuple[FiniteElementFieldSpec, ...]:
    templates = {
        cell_kind: {
            field.name: field
            for _, field in layer_core_history_field_templates(cell_kind)
        }
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    return tuple(
        FiniteElementFieldSpec(
            field.name,
            {
                block.name: templates[block.cell_kind][field.name].elements[0]
                for block in result.mesh.blocks
            },
            component_shape=field.component_shape,
        )
        for field in next(iter(templates.values())).values()
    )


def _history_space(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    reuse_source: CellMeshingResult | None = None,
) -> FiniteElementDiscretization:
    plan = FiniteElementPlan(
        result.mesh,
        _history_fields(result),
        coordinate_spec=result.geometry,
        budget=plans.budget,
    )
    retained = plans.history_spaces.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError(
                f"history.cached_result_id must equal {result.result_id!r}; got {retained_result.result_id!r}"
            )
        retained_space.require_exact_plan(plan, budget=plans.budget)
        return retained_space
    if reuse_source is not None:
        require_restored_layer_core_result_identity(
            reuse_source, result, budget=plans.budget
        )
        source_entry = plans.history_spaces.get(reuse_source.result_id)
        if source_entry is None:
            raise ValueError(
                f"history.source_preparation must be cached for {reuse_source.result_id!r}; got None"
            )
        source_space = source_entry[1]
        source_space.require_exact_plan(plan, budget=plans.budget)
        plans.history_spaces[result.result_id] = (result, source_space)
        return source_space
    transfer_entry = plans.transfer_spaces.get(result.result_id)
    if transfer_entry is None:
        prepared = plan.prepare(budget=plans.budget)
    else:
        transfer_space = transfer_entry[1]
        h1 = transfer_space.field_index("H1")
        dof_map = transfer_space.dof_maps[h1]
        prepared = FiniteElementDiscretization(
            plan, dof_maps=(dof_map, dof_map), budget=plans.budget
        )
    plans.history_spaces[result.result_id] = (result, prepared)
    return prepared


def _history_field_transfers(
    adaptation: MeshAdaptationResult,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    plans: LayerCoreLifecyclePlans,
    /,
) -> tuple[FiniteElementFieldTransfer, ...] | None:
    retained = plans.field_transfers.get((adaptation.result_id, "H1"))
    if retained is None:
        return None
    return tuple(
        FiniteElementFieldTransfer(
            retained.transfer,
            source.field_spaces[index],
            target.field_spaces[index],
            retained.geometry,
            retained.evidence,
            source_measures=retained.source_measures,
            target_measures=retained.target_measures,
        )
        for index in range(len(source.field_spaces))
    )


def _accepted_history(
    result: CellMeshingResult,
    solution: Array,
    damage: Array,
    density: Array,
    epoch: int,
    compilation_id: str,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
) -> FiniteElementAcceptedState:
    """The transaction state whose materials commit the accepted FV density."""
    space = _history_space(result, plans=plans)
    rows = _finite_volume_rows(result, plans=plans)
    materials = MaterialTransaction(
        (
            MaterialState(
                MaterialSiteId(material["site_id"]),
                "physical-cell-average-density",
                jnp.asarray(density)[rows],
                state_version=epoch,
            ),
        )
    )
    return FiniteElementAcceptedState(
        (solution, damage),
        0.0,
        0,
        result.mesh.topology_id,
        space.prepared_id,
        compilation_id,
        materials=materials,
        state_version=epoch,
    )


@checked
def initial_layer_core_state(
    source: CellMeshingResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    solver_repeats: int = 1,
    original_scalar: Array | None = None,
) -> tuple[LayerCoreAcceptedState, LayerCoreSolveEpoch]:
    """Author the producer's initial accepted state and solve its harmonic PDE.

    H1 and DG hold the affine field ``1 + y + z`` exactly representable on the
    mapped carrier; H(curl) and H(div) hold one fixed normal coefficient draw;
    the density is the declared material; the damage history is the affine
    bounded field; the harmonic solution is compiled ahead of time.
    """
    if isinstance(solver_repeats, bool) or solver_repeats < 1:
        raise ValueError(
            f"history.solver_repeats must be a positive integer; got {solver_repeats!r}"
        )
    first = _transfer_space(source, plans=plans)
    fields = tuple(
        jnp.asarray(layer_core_initial_values(first, kind, field_name=kind)[:, 0])
        for kind in LAYER_CORE_FIELD_KINDS
    )
    density = layer_core_initial_density(source, material=material, plans=plans)
    space = _history_space(source, plans=plans)
    harmonic, compiled = _solve_diffusion(space, 0.0, plans=plans, compiled=True)
    epoch = _harmonic_epoch(
        0, "initial", source, harmonic, compiled=compiled, repeats=solver_repeats
    )
    history = _accepted_history(
        source,
        harmonic.values,
        space.project("damage-history", _damage),
        density,
        0,
        harmonic.problem.compilation_id,
        material=material,
        plans=plans,
    )
    if original_scalar is not None:
        _original_scalar_physical(source, original_scalar, plans=plans)
    return LayerCoreAcceptedState(
        source, fields, density, history, 0, original_scalar=original_scalar
    ), epoch


@checked
def continue_layer_core_history(
    accepted: FiniteElementAcceptedState,
    adaptation: MeshAdaptationResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    solver_repeats: int = 1,
    restores: CellMeshingResult | None = None,
) -> tuple[FiniteElementAcceptedState, LayerCoreHistoryStage, LayerCoreSolveEpoch]:
    """Atomic PDE, damage-history and material transaction with changed-PDE rollback.

    The target harmonic problem is solved independently, compiled ahead of
    time and re-executed ``solver_repeats`` times; the transferred solution
    commits only when it matches that solve and the material inventory and
    damage history are preserved.
    """
    if isinstance(solver_repeats, bool) or solver_repeats < 1:
        raise ValueError(
            f"history.solver_repeats must be a positive integer; got {solver_repeats!r}"
        )
    source, target = adaptation.source, adaptation.target
    region_count = layer_core_material_values(source, material).size
    layer_core_material_values(target, material)
    site_id = material["site_id"]
    if accepted.materials is None:
        raise ValueError(
            "history.accepted_materials must retain material state; got None"
        )
    fields = _history_fields(source)
    source_space = _history_space(source, plans=plans)
    target_space = _history_space(
        target,
        plans=plans,
        reuse_source=(
            restores
            if restores is not None and restores.result_id in plans.history_spaces
            else None
        ),
    )
    target_harmonic, target_compiled = _solve_diffusion(
        target_space, 0.0, plans=plans, compiled=True
    )
    independent = target_harmonic.values
    witness, _ = _solve_diffusion(target_space, 0.1, plans=plans, compiled=False)
    expected_damage = target_space.project("damage-history", _damage)
    regions = layer_core_region_labels(source)
    inventories = layer_core_region_inventory(
        jnp.asarray(layer_core_cell_measures(source, plans=plans))
        * accepted.materials.state(site_id).committed,
        regions,
        region_count,
    )
    target_measures = layer_core_cell_measures(target, plans=plans)
    target_regions = layer_core_region_labels(target)
    transport = _density_transport(adaptation, site_id, plans=plans)

    def transaction(solution: Array, /) -> FiniteElementTopologyTransaction:
        return FiniteElementTopologyTransaction(
            _PhysicalCertification(
                solution,
                expected_damage,
                jnp.asarray(target_measures, dtype=jnp.float64),
                jnp.asarray(target_regions, dtype=jnp.int32),
                inventories,
                site_id,
            ),
            fields=fields,
            material_transfer=transport,
        )

    committing = transaction(independent)
    preparation = committing.prepare_transition(
        accepted,
        source.mesh,
        adaptation,
        source=source_space,
        target=target_space,
        field_transfers=_history_field_transfers(
            adaptation, source_space, target_space, plans
        ),
        budget=plans.budget,
    )
    refused = transaction(witness.values).execute(
        accepted,
        source.mesh,
        adaptation,
        preparation=preparation,
        budget=plans.budget,
    )
    if (
        bool(refused.committed)
        or refused.state is not accepted
        or refused.mesh is not source.mesh
    ):
        raise ValueError(
            f"history.changed_pde_rollback must retain accepted state and mesh without committing; got {(bool(refused.committed), refused.state is accepted, refused.mesh is source.mesh)!r}"
        )
    committed = committing.execute(
        accepted,
        source.mesh,
        adaptation,
        compiled_layout_id=target_harmonic.problem.compilation_id,
        preparation=preparation,
        budget=plans.budget,
    )
    if (
        not bool(committed.committed)
        or committed.receipt is None
        or not committed.receipt.published
    ):
        raise ValueError(
            f"history.transaction must commit with a published transaction record; got {(bool(committed.committed), committed.receipt)!r}: {committed.diagnostics}"
        )
    if not all(transfer.evidence.passed for transfer in committed.transfers):
        raise ValueError(
            f"history.transfer_evidence must pass for every transfer; got {tuple(transfer.evidence.passed for transfer in committed.transfers)!r}"
        )
    state = committed.state
    if state.materials is None:
        raise ValueError(
            "history.committed_materials must retain material state; got None"
        )
    inventory = layer_core_region_inventory(
        jnp.asarray(target_measures) * state.materials.state(site_id).committed,
        target_regions,
        region_count,
    )
    phase = "coarsen" if _is_coarsening(adaptation) else "refine"
    stage: LayerCoreHistoryStage = {
        "phase": phase,
        "cells": int(_cell_ids(target).size),
        "changed_pde_rolled_back": True,
        "pde_defect": float(jnp.max(jnp.abs(state.fields[0] - independent))),
        "damage_defect": float(jnp.max(jnp.abs(state.fields[1] - expected_damage))),
        "inventory_defect": float(jnp.max(jnp.abs(inventory - inventories))),
    }
    epoch = _harmonic_epoch(
        state.state_version,
        phase,
        target,
        target_harmonic,
        compiled=target_compiled,
        repeats=solver_repeats,
    )
    return state, stage, epoch


def _run_continuation_consumers(
    state: LayerCoreAcceptedState,
    adaptation: MeshAdaptationResult,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    stages_seconds: dict[str, float],
    /,
    *,
    solver_repeats: int,
    restores: CellMeshingResult | None,
) -> tuple[
    tuple[tuple[Array, ...], list[LayerCoreFieldContinuation]],
    tuple[Array, LayerCoreVolumeContinuation],
    tuple[FiniteElementAcceptedState, LayerCoreHistoryStage, LayerCoreSolveEpoch],
]:
    """Keep all current source preparations and live storage on their owning ledger."""

    def run_fields() -> tuple[tuple[Array, ...], list[LayerCoreFieldContinuation]]:
        started = perf_counter()
        try:
            return continue_layer_core_fields(
                state.fields, adaptation, plans=plans, restores=restores
            )
        finally:
            stages_seconds["compatible_field_continuation"] = perf_counter() - started

    def run_volume() -> tuple[Array, LayerCoreVolumeContinuation]:
        started = perf_counter()
        try:
            return continue_layer_core_finite_volume(
                state.density, adaptation, material=material, plans=plans
            )
        finally:
            stages_seconds["finite_volume_continuation"] = perf_counter() - started

    def run_history() -> tuple[
        FiniteElementAcceptedState, LayerCoreHistoryStage, LayerCoreSolveEpoch
    ]:
        started = perf_counter()
        try:
            return continue_layer_core_history(
                state.history,
                adaptation,
                material=material,
                plans=plans,
                solver_repeats=solver_repeats,
                restores=restores,
            )
        finally:
            stages_seconds["history_and_pde_continuation"] = perf_counter() - started

    return run_fields(), run_volume(), run_history()


@checked
def continue_layer_core_state(
    state: LayerCoreAcceptedState,
    adaptation: MeshAdaptationResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    execution: NativeExecutionScope,
    solver_repeats: int = 1,
    restores: CellMeshingResult | None = None,
    original_scalar: Array | None = None,
) -> tuple[LayerCoreAcceptedState, LayerCoreStepContinuation]:
    """Continue every accepted array of one state across one archived adaptation.

    Warm producers and cold consumers run this same route. ``restores`` names
    an already prepared carrier that a coarsening target restores bitwise.
    """
    if plans.budget is None:
        return _with_layer_core_plan_budget(
            execution,
            plans,
            lambda: continue_layer_core_state(
                state,
                adaptation,
                material=material,
                plans=plans,
                execution=execution,
                solver_repeats=solver_repeats,
                restores=restores,
                original_scalar=original_scalar,
            ),
        )
    if getattr(plans.budget.ticket, "scope", None) is not execution:
        raise ValueError("continuation.plans must use the actual current native request budget")
    if state.result.result_id != adaptation.source.result_id:
        raise ValueError(
            f"continuation.state must belong to the adaptation source {adaptation.source.result_id!r}; got {state.result.result_id!r}"
        )
    if original_scalar is not None and state.original_scalar is None:
        raise ValueError(
            "original_scalar.continuation cannot introduce an unseeded original bank"
        )
    stages_seconds: dict[str, float] = {}
    (fields, field_summaries), (density, volume), (history, stage, solve) = (
        _run_continuation_consumers(
            state,
            adaptation,
            material,
            plans,
            stages_seconds,
            solver_repeats=solver_repeats,
            restores=restores,
        )
    )
    target = adaptation.target
    committed = (
        None
        if history.materials is None
        else history.materials.state(material["site_id"]).committed
    )
    rows = _finite_volume_rows(target, plans=plans)
    # The FV continuation and the transaction's material transport share one
    # remap plan; one accepted density is checkpointed for both.
    if committed is None or not np.array_equal(
        np.asarray(committed), np.asarray(density)[rows]
    ):
        raise ValueError(
            "continuation.committed_material must equal the continued finite-volume density bitwise; got differing material state"
        )
    scalar, scalar_summary = None, None
    if state.original_scalar is not None:
        started = perf_counter()
        scalar, scalar_summary = _continue_original_scalar(
            state,
            adaptation,
            plans=plans,
            accepted_values=original_scalar,
            repeats=solver_repeats,
        )
        stages_seconds["original_scalar_continuation"] = perf_counter() - started
    continued = LayerCoreAcceptedState(
        target, fields, density, history, history.state_version, original_scalar=scalar
    )
    step: LayerCoreStepContinuation = {
        "phase": stage["phase"],
        "adaptation_id": adaptation.result_id,
        "source_epoch": state.epoch,
        "target_epoch": continued.epoch,
        "fields": field_summaries,
        "finite_volume": volume,
        "history": stage,
        "solve": solve,
        "stages_seconds": stages_seconds,
    }
    if scalar_summary is not None:
        step["original_scalar"] = scalar_summary
    return continued, step


def layer_core_continuation_execution[T](
    limits: MeshingLimits,
    previous: NativeExecutionRecord,
    call: Callable[[NativeExecutionScope], T],
    /,
    *,
    import_preparation: bool = True,
) -> tuple[T, NativeExecutionRecord]:
    """Run one continuation with current preparation or historical provenance.

    Cold reconstruction starts a fresh request: its archived receipt is history,
    not work or elapsed time to import into the current allowance.
    """
    if previous.owner_id is None:
        raise ValueError(
            "continuation.execution_owner_id must retain the archived execution owner; got None"
        )
    with (
        NativeExecutionScope(limits.native_limits()) as execution,
        (
            native_volume_source_execution_budget(
                limits, previous, previous.owner_id, scope=execution
            )
            if import_preparation
            else nullcontext()
        ),
    ):
        if not import_preparation:
            previous.require_valid()
            native_preparation_ledger(execution).record_history(execution, previous)
        value = call(execution)
        overlap = (
            native_preparation_ledger(execution).overlap(execution, previous)
            if import_preparation
            else None
        )
    if execution.evidence is None:
        raise RuntimeError(
            "continuation.execution_evidence must retain ended scope evidence; got None"
        )
    return value, NativeExecutionRecord(
        execution.evidence if overlap is None else overlap.normalize(execution.evidence),
        preparation_evidence=previous if import_preparation else None,
        history_evidence=() if import_preparation else (previous,),
        owner_id=previous.owner_id,
    )


def _layer_core_current_preparation(
    source: CellMeshingResult,
    current: NativeExecutionRecord | None,
    /,
) -> NativeExecutionRecord:
    base = source.execution_evidence
    if base is None:
        raise ValueError("continuation.source must retain its actual native request receipt")
    if current is None:
        return base
    current.require_valid()
    phase: NativeExecutionRecord | None = current
    while phase is not None:
        if phase is base:
            return current
        phase = phase.preparation_evidence
    raise ValueError(
        "continuation.current_preparation must retain the actual source request, not an equal owner or historical receipt"
    )


def _with_layer_core_plan_budget[T](
    execution: NativeExecutionScope,
    plans: LayerCoreLifecyclePlans,
    call: Callable[[], T],
    /,
    *,
    prepared_operands: PreparedCoordinateOperands | None = None,
    source: CellMeshingResult | None = None,
    paid_receipt: NativeExecutionRecord | None = None,
) -> T:
    with execution.ticket_view() as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=execution.limits.maximum_work_units,
            maximum_memory_bytes=execution.limits.maximum_managed_bytes,
        )
        if prepared_operands is not None:
            if source is None or paid_receipt is None:
                raise ValueError("Layer continuation operands require their actual source and paid receipt.")
            prepared_operands.require(
                source.mesh, source.geometry, paid_receipt, budget,
                coordinate_contract=source.coordinate_contract,
            )
        with (
            budget.activate(),
            plans.bind_budget(budget),
            meshing_source_owner_reconstruction_scope(budget=budget),
        ):
            return call()


@checked
def warm_layer_core_initial_state(
    source: CellMeshingResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    limits: MeshingLimits,
    part_name: str,
    solver_repeats: int = 1,
    original_scalar: Array | None = None,
    current_preparation: NativeExecutionRecord | None = None,
    prepared_operands: PreparedCoordinateOperands | None = None,
) -> tuple[
    LayerCoreAcceptedState,
    MeshingAcceptedEpoch,
    LayerCoreSolveEpoch,
    NativeExecutionRecord,
]:
    """Author, solve and checkpoint a producer's initial accepted state."""
    previous = _layer_core_current_preparation(source, current_preparation)
    def author() -> tuple[
        LayerCoreAcceptedState, MeshingAcceptedEpoch, LayerCoreSolveEpoch
    ]:
        state, solve = initial_layer_core_state(
            source,
            material=material,
            plans=plans,
            solver_repeats=solver_repeats,
            original_scalar=original_scalar,
        )
        return state, layer_core_accepted_epoch(state, part_name=part_name, plans=plans), solve

    (state, epoch, solve), record = layer_core_continuation_execution(
        limits,
        previous,
        lambda execution: _with_layer_core_plan_budget(
            execution, plans, author,
            prepared_operands=prepared_operands,
            source=source, paid_receipt=previous,
        ),
    )
    return (
        state,
        epoch,
        solve,
        record,
    )


@checked
def warm_layer_core_step(
    state: LayerCoreAcceptedState,
    adaptation: MeshAdaptationResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    limits: MeshingLimits,
    part_name: str,
    solver_repeats: int = 1,
    restores: CellMeshingResult | None = None,
    original_scalar: Array | None = None,
    current_preparation: NativeExecutionRecord | None = None,
    prepared_operands: PreparedCoordinateOperands | None = None,
) -> tuple[
    LayerCoreAcceptedState,
    MeshingAcceptedEpoch,
    LayerCoreStepContinuation,
    NativeExecutionRecord,
]:
    """Continue one accepted state warm across one adaptation and checkpoint it."""
    previous = _layer_core_current_preparation(
        adaptation.target, current_preparation
    )

    def continue_and_checkpoint(execution: NativeExecutionScope) -> tuple[
        LayerCoreAcceptedState, MeshingAcceptedEpoch, LayerCoreStepContinuation
    ]:
        def continue_state() -> tuple[
            LayerCoreAcceptedState, MeshingAcceptedEpoch, LayerCoreStepContinuation
        ]:
            continued, step = continue_layer_core_state(
                state,
                adaptation,
                material=material,
                plans=plans,
                execution=execution,
                solver_repeats=solver_repeats,
                restores=restores,
                original_scalar=original_scalar,
            )
            epoch = layer_core_accepted_epoch(
                continued, part_name=part_name, plans=plans, transition=adaptation
            )
            return continued, epoch, step

        return _with_layer_core_plan_budget(
            execution, plans, continue_state,
            prepared_operands=prepared_operands,
            source=adaptation.target, paid_receipt=previous,
        )

    (continued, epoch, step), record = layer_core_continuation_execution(
        limits, previous, continue_and_checkpoint
    )
    return (
        continued,
        epoch,
        step,
        record,
    )


@checked
def layer_core_lineage_records(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    epochs: tuple[MeshingAcceptedEpoch, ...],
    /,
) -> dict[str, object]:
    """The canonical accepted-epoch closure roles of one layer/core lineage.

    The final epoch carrier is the accepted target whose certification inputs,
    report and associations the closure binds; epoch ``0`` owns the generation.
    """
    target = epochs[-1].carrier
    certification = target.certification
    if certification is None:
        raise ValueError(
            f"closure.accepted_target must retain its certification; got {target.result_id!r}"
        )
    return {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": target.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "accepted_target": target,
        "accepted_data": {"accepted_epochs": epochs},
    }


_STATE_EPOCH_ENTRY = "fe/accepted-epoch"
_FINITE_VOLUME_ENTRY = "fv/density"
_FINITE_VOLUME_EPOCH_ENTRY = "fv/accepted-epoch"
_HISTORY_FIELDS = ("u", "damage-history")


def _state_banks(state: LayerCoreAcceptedState, /) -> dict[str, Array | np.ndarray]:
    """Every accepted numerical bank of one state under its archive entry name."""
    epoch = np.asarray(state.epoch, dtype=np.int64)
    return {
        **{
            f"fe/{kind}": value
            for kind, value in zip(LAYER_CORE_FIELD_KINDS, state.fields, strict=True)
        },
        **{
            f"fe/{name}": value
            for name, value in zip(_HISTORY_FIELDS, state.history.fields, strict=True)
        },
        **(
            {"fe/original-u": state.original_scalar}
            if state.original_scalar is not None else {}
        ),
        _STATE_EPOCH_ENTRY: epoch,
        _FINITE_VOLUME_ENTRY: state.density[:, None],
        _FINITE_VOLUME_EPOCH_ENTRY: epoch,
    }


def _state_spaces(
    result: CellMeshingResult, /, *, plans: LayerCoreLifecyclePlans
) -> tuple[
    FiniteElementTransferDiscretization,
    FiniteElementDiscretization,
    UnstructuredFiniteVolumeDiscretization,
]:
    return (
        _transfer_space(result, plans=plans),
        _history_space(result, plans=plans),
        _finite_volume(result, plans=plans),
    )


@checked
def layer_core_accepted_epoch(
    state: LayerCoreAcceptedState,
    /,
    *,
    part_name: str,
    plans: LayerCoreLifecyclePlans,
    transition: MeshAdaptationResult | None = None,
) -> MeshingAcceptedEpoch:
    """Checkpoint every accepted array of one state against its prepared layouts."""
    result = state.result
    transfer, history, volume = _state_spaces(result, plans=plans)
    spaces = {space.name: space for space in transfer.field_spaces} | {
        space.name: space for space in history.field_spaces
    }
    banks = _state_banks(state)
    names = (*LAYER_CORE_FIELD_KINDS, *_HISTORY_FIELDS)
    original_fields: tuple[FiniteElementFieldSpec, ...] = ()
    if state.original_scalar is not None:
        original = _original_scalar_space(result, plans=plans)
        spaces["original-u"] = original.field_spaces[0]
        names += ("original-u",)
        original_fields = original.construction_plan.fields
    finite_element = MeshingFiniteElementFieldDeclaration(
        part_name=part_name,
        topology_id=result.mesh.topology_id,
        geometry_layout_id=result.geometry.geometry_layout_id,
        field_space_ids=frozendict({name: spaces[name].field_space_id for name in names}),
        value_units=frozendict({name: (ONE,) for name in names}),
        maximum_derivative_orders=frozendict({name: 1 for name in names}),
        branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=tuple(
            MeshingFieldStateRole(
                f"fe/{name}",
                name,
                "coefficients",
                tuple(banks[f"fe/{name}"].shape),
                np.dtype(banks[f"fe/{name}"].dtype),
                state.epoch,
                index_binding=prepare_meshing_field_index_binding(
                    result, "field/global_coefficient_ids", field_space=spaces[name]
                ),
            )
            for name in names
        )
        + (
            MeshingFieldStateRole(
                _STATE_EPOCH_ENTRY,
                names[0],
                "state-epoch",
                (),
                np.dtype(np.int64),
                state.epoch,
            ),
        ),
        history_policy="accepted-state",
        numeric_version=transfer.numeric_version,
        finite_element_fields=(
            *transfer.construction_plan.fields,
            *history.construction_plan.fields,
            *original_fields,
        ),
        precision_policy=transfer.precision_policy,
    )
    template = layer_core_finite_volume_field_templates()[0]
    cell_space = volume.field_spaces[0]
    reconstruction = PiecewiseConstantReconstruction()
    finite_volume = MeshingFiniteVolumeFieldDeclaration(
        part_name=part_name,
        topology_id=result.mesh.topology_id,
        geometry_layout_id=result.geometry.geometry_layout_id,
        field_space_ids=frozendict({template["name"]: cell_space.field_space_id}),
        value_units=frozendict({template["name"]: (KILOGRAM_PER_CUBIC_METER,)}),
        maximum_derivative_orders=frozendict({template["name"]: 0}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=(
            MeshingFieldStateRole(
                _FINITE_VOLUME_ENTRY,
                template["name"],
                "coefficients",
                tuple(banks[_FINITE_VOLUME_ENTRY].shape),
                np.dtype(banks[_FINITE_VOLUME_ENTRY].dtype),
                state.epoch,
                index_binding=prepare_meshing_field_index_binding(
                    result, "field/global_coefficient_ids", field_space=cell_space
                ),
            ),
            MeshingFieldStateRole(
                _FINITE_VOLUME_EPOCH_ENTRY,
                template["name"],
                "state-epoch",
                (),
                np.dtype(np.int64),
                state.epoch,
            ),
        ),
        history_policy="accepted-state",
        numeric_version=volume.numeric_version,
        finite_volume_field_name=template["name"],
        finite_volume_component_names=template["component_names"],
        finite_volume_coordinate_policy="accepted-map",
        reconstruction_policy=reconstruction,
        reconstruction_policy_id=reconstruction.plan_id,
    )
    return MeshingAcceptedEpoch(
        result, (finite_element, finite_volume), banks, transition=transition
    )


@checked
def layer_core_epoch_identity(epoch: MeshingAcceptedEpoch, /) -> LayerCoreEpochIdentity:
    """Archived identity of one accepted epoch and the digest of its banks."""
    return {
        "epoch": int(np.asarray(epoch.fields[_STATE_EPOCH_ENTRY])),
        "epoch_id": epoch.epoch_id,
        "result_id": epoch.carrier.result_id,
        "transition_id": None if epoch.transition is None else epoch.transition.result_id,
        "field_banks_id": logical_array_value_collection_digest(dict(epoch.fields)),
    }


@checked
def restore_layer_core_state(
    epoch: MeshingAcceptedEpoch,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
) -> LayerCoreAcceptedState:
    """Rebuild one accepted state from its archived banks, never from new values.

    Every declared layout must be the layout this process continues on, and
    the rebuilt state's banks must digest to the archived banks.
    """
    result = epoch.carrier
    transfer, history, volume = _state_spaces(result, plans=plans)
    actual = {space.name: space.field_space_id for space in transfer.field_spaces} | {
        space.name: space.field_space_id for space in history.field_spaces
    }
    if "fe/original-u" in epoch.fields:
        original = _original_scalar_space(result, plans=plans)
        actual["original-u"] = original.field_spaces[0].field_space_id
    finite_element, finite_volume = epoch.declarations
    if dict(finite_element.field_space_ids) != {
        name: actual[name] for name in finite_element.field_space_ids
    } or dict(finite_volume.field_space_ids) != {
        layer_core_finite_volume_field_templates()[0]["name"]: volume.field_spaces[
            0
        ].field_space_id
    }:
        raise ValueError(
            f"restoration.field_layouts must equal this process's prepared layouts; got {(dict(finite_element.field_space_ids), dict(finite_volume.field_space_ids))!r}"
        )
    banks = epoch.fields
    index = int(np.asarray(banks[_STATE_EPOCH_ENTRY]))
    density = jnp.asarray(banks[_FINITE_VOLUME_ENTRY])[:, 0]
    solution, damage = (jnp.asarray(banks[f"fe/{name}"]) for name in _HISTORY_FIELDS)
    state = LayerCoreAcceptedState(
        result,
        tuple(jnp.asarray(banks[f"fe/{kind}"]) for kind in LAYER_CORE_FIELD_KINDS),
        density,
        _accepted_history(
            result,
            solution,
            damage,
            density,
            index,
            f"archived-accepted-epoch:{epoch.epoch_id}",
            material=material,
            plans=plans,
        ),
        index,
        original_scalar=(
            jnp.asarray(banks["fe/original-u"]) if "fe/original-u" in banks else None
        ),
    )
    if state.original_scalar is not None:
        _original_scalar_physical(result, state.original_scalar, plans=plans)
    if logical_array_value_collection_digest(
        _state_banks(state)
    ) != logical_array_value_collection_digest(dict(banks)):
        raise ValueError(
            f"restoration.state_banks must equal archived epoch {epoch.epoch_id!r}; got differing arrays"
        )
    return state


def _epoch_index(epoch: MeshingAcceptedEpoch, /) -> int:
    return int(np.asarray(epoch.fields[_STATE_EPOCH_ENTRY]))


@checked
def layer_core_continuation_defects(
    continued: LayerCoreAcceptedState,
    archived: MeshingAcceptedEpoch,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
) -> dict[str, float]:
    """Compare a continued state like-for-like with the archived continuation of its step.

    Coefficients agree within the original restoration tolerance, the PDE
    solution and damage history within the history tolerance, and the density
    and its region inventories within the inventory tolerance.
    """
    if (
        continued.result.result_id != archived.carrier.result_id
        or continued.epoch != _epoch_index(archived)
    ):
        raise ValueError(
            f"comparison.states must continue the same step; got {(continued.result.result_id, continued.epoch)!r} and {(archived.carrier.result_id, _epoch_index(archived))!r}"
        )
    banks = _state_banks(continued)
    defects: dict[str, float] = {}
    defects.update(_original_scalar_comparison(continued, archived))
    for kind in LAYER_CORE_FIELD_KINDS:
        first, second = (
            np.asarray(banks[f"fe/{kind}"]),
            np.asarray(archived.fields[f"fe/{kind}"]),
        )
        np.testing.assert_allclose(first, second, rtol=2.0e-11, atol=2.0e-11)
        defects[kind] = float(np.max(np.abs(first - second)))
    for name in _HISTORY_FIELDS:
        defects[name] = float(
            np.max(
                np.abs(
                    np.asarray(banks[f"fe/{name}"])
                    - np.asarray(archived.fields[f"fe/{name}"])
                )
            )
        )
        if defects[name] > 1.0e-9:
            raise ValueError(
                f"comparison.{name} must agree within 1.0e-9; got {defects[name]:.3e}"
            )
    result = continued.result
    region_count = layer_core_material_values(result, material).size
    space = _finite_volume(result, plans=plans)
    regions = layer_core_cell_regions(result)
    volumes = np.asarray(space.cell_volumes, dtype=np.float64)
    gids = np.asarray(space.cell_global_ids, dtype=np.int64)
    density = np.asarray(continued.density)
    archived_density = np.asarray(archived.fields[_FINITE_VOLUME_ENTRY])[:, 0]
    defects["density"] = float(np.max(np.abs(density - archived_density)))
    defects["region_inventory"] = float(
        np.max(
            np.abs(
                _inventories(volumes, density, gids, regions, region_count)
                - _inventories(volumes, archived_density, gids, regions, region_count)
            )
        )
    )
    if max(defects["density"], defects["region_inventory"]) > 1.0e-12:
        raise ValueError(
            f"comparison.material must agree within 1.0e-12; got {(defects['density'], defects['region_inventory'])!r}"
        )
    return defects


@checked
def layer_core_restoration_defects(
    restored: LayerCoreAcceptedState,
    reference: MeshingAcceptedEpoch,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> dict[str, float]:
    """Restoration of the accepted state preceding a refinement by its inverse."""
    carrier = reference.carrier
    require_restored_layer_core_result_identity(
        carrier, restored.result, budget=plans.budget
    )
    if restored.result.mesh.topology_id != carrier.mesh.topology_id:
        raise ValueError(
            f"coarsening.topology_id must equal source topology {carrier.mesh.topology_id!r}; got {restored.result.mesh.topology_id!r}"
        )
    banks = _state_banks(restored)
    defects: dict[str, float] = {}
    defects.update(_original_scalar_comparison(restored, reference))
    for kind in LAYER_CORE_FIELD_KINDS:
        first, second = (
            np.asarray(banks[f"fe/{kind}"]),
            np.asarray(reference.fields[f"fe/{kind}"]),
        )
        np.testing.assert_allclose(first, second, rtol=2.0e-11, atol=2.0e-11)
        defects[kind] = float(np.max(np.abs(first - second)))
    defects["damage-history"] = float(
        np.max(
            np.abs(
                np.asarray(banks["fe/damage-history"])
                - np.asarray(reference.fields["fe/damage-history"])
            )
        )
    )
    if defects["damage-history"] > 1.0e-9:
        raise ValueError(
            f"history.restored_damage must equal the preceding damage within 1.0e-9; got {defects['damage-history']:.3e}"
        )
    first = _finite_volume(carrier, plans=plans)
    last = _finite_volume(restored.result, plans=plans)
    rows = {
        identifier: row
        for row, identifier in enumerate(np.asarray(first.cell_global_ids).tolist())
    }
    order = np.asarray(
        [rows[identifier] for identifier in np.asarray(last.cell_global_ids).tolist()],
        dtype=np.int64,
    )
    volumes = np.asarray(first.cell_volumes, dtype=np.float64)
    content = volumes * np.asarray(reference.fields[_FINITE_VOLUME_ENTRY])[:, 0]
    defects["restored_cell_content"] = float(
        np.max(
            np.abs(
                np.asarray(last.cell_volumes) * np.asarray(restored.density)
                - content[order]
            )
        )
    )
    defects["restored_cell_volume"] = float(
        np.max(np.abs(np.asarray(last.cell_volumes) - volumes[order]))
    )
    scale = float(np.sum(np.abs(content)))
    if max(
        defects["restored_cell_content"], defects["restored_cell_volume"]
    ) > 1.0e-13 * max(1.0, scale):
        raise ValueError(
            f"finite_volume.restoration_defects must be at most {1.0e-13 * max(1.0, scale)!r}; got {defects!r}"
        )
    return defects


@final
class LayerCoreLifecycleReport(StrictModule):
    """Restored archive identities, the cold step and its comparison with the warm step."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    material_declaration: LayerCoreMaterialDeclaration = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_result: LayerCoreResultIdentity = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    coarsened_geometry_id: str = eqx.field(static=True)
    restored_coordinate_map_bitwise: bool = eqx.field(static=True)
    adaptation_ids: tuple[str, ...] = eqx.field(static=True)
    cells: tuple[int, ...] = eqx.field(static=True)
    stages_seconds: tuple[tuple[str, float], ...] = eqx.field(static=True)
    restored_epochs: tuple[LayerCoreEpochIdentity, ...] = eqx.field(static=True)
    continued_from: LayerCoreEpochIdentity = eqx.field(static=True)
    continuation: LayerCoreStepContinuation = eqx.field(static=True)
    warm_continuation_defects: tuple[tuple[str, float], ...] = eqx.field(static=True)
    restoration_defects: tuple[tuple[str, float], ...] = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord
    limits_id: str = eqx.field(static=True)

    def to_records(self) -> tuple[dict[str, object], ...]:
        """Return canonical diagnostic records and the optional actual P1 consumer."""
        records: tuple[dict[str, object], ...] = (
            {
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "material_declaration": self.material_declaration,
                "source_mesh_id": self.source_mesh_id,
                "source_result": self.source_result,
                "source_geometry_id": self.source_geometry_id,
                "coarsened_geometry_id": self.coarsened_geometry_id,
                "restored_coordinate_map_bitwise": self.restored_coordinate_map_bitwise,
                "adaptation_ids": list(self.adaptation_ids),
                "cells": list(self.cells),
                "stages_seconds": dict(self.stages_seconds),
                "restored_epochs": list(self.restored_epochs),
                "restored_epoch_ids": [
                    epoch["epoch_id"] for epoch in self.restored_epochs
                ],
                "continued_from": self.continued_from,
                "warm_continuation_defects": dict(self.warm_continuation_defects),
                "restoration_defects": dict(self.restoration_defects),
            },
            {"fields": self.continuation["fields"]},
            {"finite_volume": self.continuation["finite_volume"]},
            {
                "history": {
                    "phase": self.continuation["phase"],
                    "adaptation_id": self.continuation["adaptation_id"],
                    "source_epoch": self.continuation["source_epoch"],
                    "target_epoch": self.continuation["target_epoch"],
                    "stages": [self.continuation["history"]],
                    "epochs": [self.continuation["solve"]],
                }
            },
            {
                "execution_evidence": self.execution_evidence.to_record(),
                "limits_id": self.limits_id,
            },
        )
        if "original_scalar" in self.continuation:
            scalar_record: dict[str, object] = {
                "original_scalar": self.continuation["original_scalar"]
            }
            records += (scalar_record,)
        return records


def continue_layer_core_lifecycle(
    archive_path: str | Path,
    content_id: str,
    /,
    *,
    region_densities: tuple[float, ...] = LAYER_CORE_REGION_DENSITIES,
    material_site: str = "two-material-density",
    solver_repeats: int = 1,
) -> LayerCoreLifecycleReport:
    """Cold-continue the archived accepted state through its final archived adaptation.

    The state preceding the final inverse coarsening is rebuilt from its
    archived banks, continued on the warm producer's route under the source
    domain limits, solved independently, and compared like-for-like with the
    archived warm continuation; its restoration of the state preceding the
    final refinement is checked as well. The declared material must equal the
    archived initial density bitwise. The cold harmonic solve is re-executed
    ``solver_repeats`` times for its epoch evidence.

    Source controls are authenticated before opening the current request. One
    native allowance and coordinate ledger then own scientific restoration,
    all continued consumers and the final coordinate-map comparison.
    """
    stages_seconds: dict[str, float] = {}
    phase_started = perf_counter()
    archive = _read_meshing_source_archive(archive_path, expected_content_id=content_id)
    stages_seconds["archive_member_authentication"] = perf_counter() - phase_started
    phase_started = perf_counter()
    controls = read_meshing_source_execution_controls(
        archive_path,
        expected_content_id=content_id,
        _authenticated_archive=archive,
    )
    stages_seconds["execution_control_restoration"] = perf_counter() - phase_started
    previous = controls.execution_evidence

    def restore_and_continue(
        execution: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[
        tuple[MeshingAcceptedEpoch, ...],
        NativeLayerCoreSource,
        LayerCoreMaterialDeclaration,
        LayerCoreAcceptedState,
        LayerCoreStepContinuation,
        dict[str, float],
        dict[str, float],
        bool,
    ]:
        phase_started = perf_counter()
        records = _restore_meshing_source_archive(
            archive, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS, budget=budget
        )
        stages_seconds["scientific_owner_restoration"] = perf_counter() - phase_started
        epochs = accepted_layer_core_lineage(records)
        authored = records["generation_source"]
        specification = records["generation_specification"]
        if not isinstance(authored, NativeLayerCoreSource):
            raise TypeError(
                f"closure.generation_source must be a NativeLayerCoreSource; got {type(authored).__name__}"
            )
        if (
            authored.source_id,
            authored.source_revision,
            authored.binding_id,
            specification.specification_id,
        ) != (
            controls.source_id,
            controls.source_revision,
            controls.source_binding_id,
            controls.specification_id,
        ):
            raise ValueError(
                f"continuation.source_controls must match authenticated controls; got {(authored.source_id, authored.source_revision, authored.binding_id, specification.specification_id)!r} for {(controls.source_id, controls.source_revision, controls.source_binding_id, controls.specification_id)!r}"
            )
        source = epochs[0].carrier
        material = layer_core_material_declaration(
            source, densities=region_densities, site_id=material_site
        )
        plans = LayerCoreLifecyclePlans(budget=budget)
        final = epochs[-1].transition
        if not isinstance(final, MeshAdaptationResult):
            raise ValueError(
                f"closure.accepted_epochs[-1].transition must own the final coarsening adaptation; got {type(final).__name__}"
            )
        phase_started = perf_counter()
        declared = layer_core_initial_density(source, material=material, plans=plans)
        # Epoch 0 of a carried lineage holds the restored density of earlier
        # cycles, so the declaration is checked under the inventory tolerance.
        declared_defect = float(
            np.max(
                np.abs(
                    np.asarray(epochs[0].fields[_FINITE_VOLUME_ENTRY])[:, 0]
                    - np.asarray(declared)
                )
            )
        )
        if declared_defect > 1.0e-12:
            raise ValueError(
                f"continuation.material_declaration must equal the archived initial density within 1.0e-12; got {declared_defect:.3e} for {material!r}"
            )
        preceding = restore_layer_core_state(epochs[-2], material=material, plans=plans)
        stages_seconds["accepted_state_restoration"] = perf_counter() - phase_started
        phase_started = perf_counter()
        cold, step = continue_layer_core_state(
            preceding,
            final,
            material=material,
            plans=plans,
            execution=execution,
            solver_repeats=solver_repeats,
        )
        stages_seconds["cold_state_continuation"] = perf_counter() - phase_started
        stages_seconds.update(
            {f"cold_{name}": seconds for name, seconds in step["stages_seconds"].items()}
        )
        phase_started = perf_counter()
        compared = layer_core_continuation_defects(
            cold, epochs[-1], material=material, plans=plans
        )
        restored = layer_core_restoration_defects(cold, epochs[-3], plans=plans)
        same_map = _same_coordinate_map(
            epochs[-3].carrier, cold.result, budget=budget
        )
        stages_seconds["warm_cold_comparison"] = perf_counter() - phase_started
        return epochs, authored, material, cold, step, compared, restored, same_map

    def continue_archive(execution: NativeExecutionScope) -> tuple[
        tuple[MeshingAcceptedEpoch, ...],
        NativeLayerCoreSource,
        LayerCoreMaterialDeclaration,
        LayerCoreAcceptedState,
        LayerCoreStepContinuation,
        dict[str, float],
        dict[str, float],
        bool,
    ]:
        with execution.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=controls.limits.maximum_work_units,
                maximum_memory_bytes=controls.limits.maximum_scratch_bytes,
            )
            with budget.activate(), meshing_source_owner_reconstruction_scope(budget=budget):
                return restore_and_continue(execution, budget)

    (epochs, authored, material, cold, step, compared, restored, same_map), latest = (
        layer_core_continuation_execution(
            controls.limits, previous, continue_archive, import_preparation=False
        )
    )
    source = epochs[0].carrier
    return LayerCoreLifecycleReport(
        source_id=authored.source_id,
        source_revision=authored.source_revision,
        material_declaration=material,
        source_mesh_id=source.mesh.mesh_id,
        source_result=_published_result_identity(source),
        source_geometry_id=cell_geometry_id(source.geometry),
        coarsened_geometry_id=cell_geometry_id(cold.result.geometry),
        restored_coordinate_map_bitwise=same_map,
        adaptation_ids=tuple(
            epoch.transition.result_id
            for epoch in epochs[1:]
            if epoch.transition is not None
        ),
        cells=tuple(_cell_ids(epoch.carrier).size for epoch in epochs),
        stages_seconds=tuple(sorted(stages_seconds.items())),
        restored_epochs=tuple(layer_core_epoch_identity(epoch) for epoch in epochs),
        continued_from=layer_core_epoch_identity(epochs[-2]),
        continuation=step,
        warm_continuation_defects=tuple(sorted(compared.items())),
        restoration_defects=tuple(sorted(restored.items())),
        execution_evidence=latest,
        limits_id=controls.limits.limits_id,
    )

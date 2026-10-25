#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Synthetic substrate checks and native compartment transport evidence.

Run ``JAX_ENABLE_X64=1 python -m tools.neurofluid_qualification
--scenario native-generated-transport --size 2 --repeats 3 --steps 5 --dt 0.01``.
The native scenario generates the bulk mesh from occupied labeled voxels; it
never substitutes the handbuilt tetrahedron used by the synthetic H(div) check.
Acceptance compares each discrete local/material inventory equation and the
continuous-time transport solution against an independent host assembly.
One warm native request includes source authoring, publication, scientific
consumers and archive admission. It ends before the distinct original-cap cold
request; both retain the same whole-workflow expiry. The archive retains the
actual preceding ended publication receipt explicitly as a prefix. The workflow
record binds the complete ended warm receipt to that archive and scientific state.
Compiler estimates and process-lifetime/device allocation peaks are distinct.
An absent native route or kernel is a prerequisite failure, not a comparison
engine fallback. Targets are requested objectives, not claims of qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, NamedTuple, Required, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    CompilationTiming,
    compiler_evidence,
    CompilerEvidence,
    DurationDistribution,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import array_tree_fingerprint, ArrayArchiveLimits
from phydrax._meshcore import (
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    NativeParallelExecution,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle import (
    MeshingSourceClosure,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    recertify_restored_meshing_source,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_field_records import (
    _require_execution_endpoint_extension,
)
from phydrax.lifecycle._meshing_sources import meshing_source_owner_reconstruction_scope
from phydrax.meshing import (
    meshcore_identity,
    meshcore_runtime_identity,
    MeshcoreStatus,
    NativeExecutionInterchange,
    NativeExecutionRecord,
    NativeMeshingPhaseMeasurement,
)
from phydrax.meshing._certified_unit_cells import affine_image_geometry
from phydrax.meshing._measurements import (
    native_publication_resource_binding,
    NativePublicationResourceBinding,
)
from phydrax.meshing._native_preparation import (
    prepare_native_completed_source_issue,
    stage_native_source_publication,
)
from phydrax.meshing.providers._native_publication import bind_native_execution_result
from phydrax.meshing.qualification._native_lifecycle import execute_native_request
from tools._meshing_cases import python_source_revision
from tools.meshing_qualification._runtime import (
    ColdControlPacket,
    ColdWorkerReport,
    Deadline,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    prepare_cold_worker,
    qualification_failure,
    StageLedger,
)


class _TransportGeometry(TypedDict):
    compartment_ids: tuple[str, ...]
    voxel_counts: dict[str, int]
    compartment_volumes: list[float]
    expected_compartment_volumes: list[float]
    volume_error: float
    interface_area: float
    interface_areas: dict[str, float]
    expected_interface_areas: dict[str, float]
    interface_area_error: float
    interface_facets: int
    interface_facets_by_pair: dict[str, int]
    adjacency_pairs: tuple[tuple[str, str], ...]
    voxel_adjacency_pairs: tuple[tuple[str, str], ...]
    junction_edge_length: float
    expected_junction_edge_length: float
    junction_length_error: float
    quadruple_line_length: float
    expected_quadruple_line_length: float
    quadruple_line_length_error: float
    quadruple_points: int
    quadruple_points_recovered: int
    quintuple_points: int
    quintuple_points_recovered: int
    material_containment_violations: int
    boundary_flux_consistency_error: float
    volume_flux_divergence_error: float
    declared_parameter_error: float
    affine_matrix: list[list[float]]
    geometry_tolerance: float


class _TransportInvariants(_TransportGeometry):
    accepted_steps: list[bool]
    independent_inventory_initial: float
    independent_inventory_final: float
    inventory_error: float
    local_balance_error: float
    interface_inventory_balance_error: float
    material_interface_flux_samples: dict[str, list[float]]
    bulk_network_exchange_samples: list[list[float]]
    external_loss_samples: list[float]
    boundary_inflow_rate: float
    boundary_outflow_samples: list[float]
    removal_samples: list[float]
    integrated_boundary_inflow: float
    integrated_boundary_outflow: float
    integrated_removal: float
    initial_state_error: float
    compartment_inventory_final: list[float]
    network_inventory_final: float
    reservoir_inventory_final: float
    transfer_error: float
    exchange_support: dict[str, object]
    minimum_concentration: float
    physical_error: float
    physical_reference: str
    successful: bool


class _TransportIdentities(TypedDict, total=False):
    mesh_id: str
    result_id: str
    request_id: str
    region_evidence_id: str
    coverage_id: str
    case_revision: str
    network_id: str
    transport_plan_id: str
    runtime_id: str
    transfer_id: str
    meshcore: str
    provider: str


class _TransportTiming(TypedDict, total=False):
    source_preparation_seconds: float
    geometry_preparation_seconds: float
    generation_publication_seconds: float
    readmission_seconds: float
    transport_transfer_preparation_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    first_execution_seconds: float
    warmed: dict[str, int | float | list[float] | None]
    independent_certification_seconds: float
    end_to_end_seconds: float
    explicit_primary_transport_compilations: int
    elapsed_before_failure_seconds: float


class _TransportMemory(TypedDict):
    compiler: dict[str, object]
    logical_retained_bytes: int
    logical_output_bytes: int
    process_peak_resident_bytes: int | None
    process_peak_scope: str
    device: list[dict[str, object]]


class _TransportLifecycle(TypedDict):
    passed: bool
    field_names: tuple[str, ...]
    material_parameter_id: str
    accepted_history_steps: int
    AD: dict[str, float]
    hot_continuation_seconds: float
    hot_continuation_relative_physical_error: float
    hot_continuation_inventory_defect: float
    hot_continuation_local_balance_error: float
    atomic_rollback: dict[str, bool | int]
    cold: dict[str, object]


class _NativeTransportContext(TypedDict, total=False):
    case: str
    qualification_role: str
    native_phase_measurements: list[dict[str, object]]
    source: dict[str, str]
    native_execution: NativeExecutionInterchange
    warm_execution_binding: dict[str, str]
    identities: _TransportIdentities
    cold_process: dict[str, Any]
    size_request: dict[str, object]


class NativeTransportRecord(_NativeTransportContext, total=False):
    """Transport evidence; success-only fields are absent on phase failure."""

    successful: Required[bool]
    scenario: str
    status: str
    failure: dict[str, object]
    controls: dict[str, int | float]
    targets: dict[str, float]
    units: dict[str, str]
    timing: _TransportTiming
    memory: _TransportMemory
    invariants: _TransportInvariants
    lifecycle: _TransportLifecycle
    time_convergence: dict[str, object]
    transport_case: dict[str, object]
    native_execution_scope: str
    original_native_limits: dict[str, object]
    publication_resource_guarantees: dict[str, list[str]]
    publication_resource_binding: NativePublicationResourceBinding | None
    comparison: dict[str, str]
    native_build_identity: dict[str, str]
    loaded_python_source_digest: str
    build_identity: dict[str, object]
    runtime_identity: dict[str, object]


type _TransportTrajectory = tuple[
    phx.equations.MixedDimensionalTransportState,
    tuple[phx.equations.MixedDimensionalTransportState, jax.Array],
]


class _TransportGeneration(TypedDict):
    transport_case: NativeTransportCase
    source: phx.geometry.CompartmentMeshingSource
    network: phx.discretization.PreparedMetricNetwork
    request: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    plan: phx.meshing.NativeMeshingPlan
    result: phx.meshing.CellMeshingResult
    native_execution: NativeExecutionInterchange
    native_phases: list[dict[str, object]]
    source_seconds: float
    geometry_preparation: float
    generation: float


class _TransportAcceptedExecution(TypedDict):
    case: phx.applications.neurofluid.NeurofluidCase
    readmitted: phx.applications.neurofluid.NeurofluidCase
    transport_plan: phx.applications.neurofluid.NeurofluidTransportPlan
    runtime: phx.equations.PreparedMixedDimensionalTransport
    evidence: phx.meshing.RegionMeshingEvidence
    initial: phx.equations.MixedDimensionalTransportState
    first: _TransportTrajectory
    admission: float
    preparation: float
    compilation: CompilationTiming
    first_seconds: float
    warmed: DurationDistribution
    compiler: CompilerEvidence
    certification: float
    invariants: _TransportInvariants
    lifecycle: _TransportLifecycle
    time_convergence: dict[str, object]


def manifest(payload: bytes) -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "qualification-image",
        checksum_algorithm="sha256",
        checksum=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )


def _qualified_label_volume(
    values: np.ndarray,
    affine: phx.imaging.ImageIndexAffine,
    /,
    *,
    asset_id: str,
    ontology_id: str,
    label_definitions: tuple[phx.imaging.LabelDefinition, ...],
    transformation_id: str,
) -> phx.imaging.LabelVolume:
    asset = phx.imaging.MedicalImageAsset(
        asset_id,
        "segmentation",
        values,
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "qualification-deid", "subject-0", "synthetic", True, True, True
        ),
        (manifest(values.tobytes()),),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id=transformation_id,
        ),
    )
    return phx.imaging.LabelVolume(
        asset,
        phx.imaging.LabelOntology(ontology_id, "synthetic", "1", label_definitions),
    )


def _segmentation(
    size: int,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[
    phx.SpatialCoordinateContract,
    phx.imaging.LabelVolume,
    phx.geometry.CompartmentComplex,
    phx.imaging.CompartmentSurfaceResult,
]:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="qualification",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    label_definitions = (
        phx.imaging.LabelDefinition(1, "first-label", "First"),
        phx.imaging.LabelDefinition(2, "second-label", "Second"),
    )
    if coordinate_budget is None:
        values = np.ones((size, size, size), dtype=np.int16)
        values[size // 2 :] = 2
        labels = _qualified_label_volume(
            values,
            affine,
            asset_id="qualification-labels",
            ontology_id="qualification-labels",
            label_definitions=label_definitions,
            transformation_id="synthetic-qualification-generator",
        )
    else:
        voxel_count = size**3
        # Raw int16 values, manifest payload, and LabelVolume's selected-value
        # copy coexist; retained values plus validity mask remain charged.
        scratch_bytes = 8 * voxel_count + 16384
        result_bytes = 3 * voxel_count + 16384
        work_units = 8 * voxel_count + 256
        coordinate_budget.ticket.checkpoint()
        with coordinate_budget.temporary_scope() as lifetime:
            coordinate_budget.admit_work_bound(
                work_units + 64,
                storage_upper=scratch_bytes + result_bytes,
            )
            coordinate_budget.reserve(
                work_units, scratch_bytes + result_bytes
            )
            values = np.ones((size, size, size), dtype=np.int16)
            values[size // 2 :] = 2
            labels = _qualified_label_volume(
                values,
                affine,
                asset_id="qualification-labels",
                ontology_id="qualification-labels",
                label_definitions=label_definitions,
                transformation_id="synthetic-qualification-generator",
            )
            with coordinate_budget.live_storage() as storage:
                storage.promote_temporary_storage(lifetime, result_bytes)
                coordinate_budget.retain_owner(
                    labels, temporary_storage_credit=storage
                )
            del values
            coordinate_budget.release_temporary_storage(scratch_bytes)
            coordinate_budget.ticket.checkpoint()
    definitions = (
        phx.geometry.CompartmentDefinition(
            "first", ("first-label",), "material", allowed_neighbor_ids=("second",)
        ),
        phx.geometry.CompartmentDefinition(
            "second", ("second-label",), "material", allowed_neighbor_ids=("first",)
        ),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "first-second", "first", "second", "exchange"
    )
    complex_ = phx.imaging.build_compartment_complex(
        labels, definitions, (interface,), budget=coordinate_budget
    )
    surfaces = phx.imaging.extract_compartment_surfaces(
        labels, complex_, budget=coordinate_budget
    )
    return contract, labels, complex_, surfaces


def qualify() -> dict[str, object]:
    """Qualify the explicit synthetic substrate scenario, not native generation."""
    _, _, complex_, surfaces = _segmentation(2)
    element = phx.discretization.fem.form_element(
        "tetrahedron", 2, 1, family="trimmed", twist="twisted", proxy="flux"
    )
    centers = np.asarray(
        (
            (1 / 3, 1 / 3, 0.0),
            (1 / 3, 0.0, 1 / 3),
            (1 / 3, 1 / 3, 1 / 3),
            (0.0, 1 / 3, 1 / 3),
        )
    )
    values_rt, _ = element.tabulate(centers)
    normals = np.asarray(
        ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 1.0, 1.0), (-1.0, 0.0, 0.0))
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    areas = np.asarray((0.5, 0.5, np.sqrt(3.0) / 2.0, 0.5))
    flux = np.asarray(
        [
            [
                areas[face] * np.dot(values_rt[face, basis], normals[face])
                for basis in range(4)
            ]
            for face in range(4)
        ]
    )
    _, center_gradients = element.tabulate(np.full((1, 3), 0.25, dtype=np.float64))
    integrated_divergence = (
        np.trace(np.asarray(center_gradients[0]), axis1=-2, axis2=-1) / 6.0
    )
    flux_error = float(np.max(np.abs(np.sum(flux, axis=0) - integrated_divergence)))
    tetrahedron = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        np.asarray(((0, 1, 2, 3),)),
    )
    boundary_face_id = int(np.asarray(tetrahedron.entity_set(2).entity_ids)[0])
    normal_boundary = phx.equations.fem.HDivNormalBoundaryCondition(
        np.asarray((boundary_face_id,)),
        resistance=2.0,
        prescribed_flux=1.0,
    )
    hdiv = phx.equations.fem.HDivStokesPlan(
        tetrahedron,
        phx.discretization.PressureGaugePolicy("mean-zero"),
        normal_boundaries=(normal_boundary,),
    ).prepare()
    zero_velocity, pressure, multiplier = hdiv.state_space.zeros()
    # The adjoint is not a unit-flux lift: its normalization depends on the
    # prepared BDM basis. Solve the outward-flux constraint instead.
    flux_factorization = phx.linalg.factorize(
        hdiv.normal_flux_operator, phx.linalg.FactorizationPolicy("svd")
    )
    flux_lift = flux_factorization.solve(hdiv.normal_flux_target)
    velocity = flux_lift.value
    normal_flow_error = float(
        np.max(np.abs(np.asarray(hdiv.normal_flux(velocity)) - 1.0))
    )
    resistance_power_error = abs(
        float(
            np.vdot(
                np.asarray(velocity),
                np.asarray(hdiv.normal_resistance_operator.mv(velocity)),
            )
        )
        - 2.0
    )
    constraint_error = float(
        np.max(np.abs(np.asarray(hdiv.residual((velocity, pressure, multiplier))[2])))
    )
    contracts = {
        "adjacency": bool(complex_.adjacency.successful),
        "rt0_flux": flux_error <= 1.0e-12,
        "hdiv_preparation": bool(hdiv.evidence.successful),
        "normal_flux_lift": bool(flux_lift.successful),
        "normal_flow": normal_flow_error <= 1.0e-12,
        "resistance_power": resistance_power_error <= 1.0e-12,
        "normal_constraint": constraint_error <= 1.0e-12,
    }
    record: dict[str, object] = {
        "scenario": "synthetic",
        "compartments": len(complex_.compartments),
        "interfaces": len(surfaces.surfaces),
        "interface_triangles": surfaces.surfaces[0].surface.mesh.entity_set(2).count,
        "rt0_flux_error": flux_error,
        "bdm2_velocity_dofs": zero_velocity.size,
        "dg1_pressure_dofs": pressure.size,
        "normal_flow_error": normal_flow_error,
        "resistance_power_error": resistance_power_error,
        "normal_constraint_error": constraint_error,
        "adjacency_successful": contracts["adjacency"],
        "hdiv_preparation_successful": contracts["hdiv_preparation"],
        "normal_flux_lift": {
            "status": int(flux_lift.status),
            "successful": bool(flux_lift.successful),
            "residual_norm": float(flux_lift.diagnostics.residual_norm),
            "normal_residual_norm": float(flux_lift.diagnostics.normal_residual_norm),
            "rank": int(flux_lift.diagnostics.rank),
        },
        "successful": all(contracts.values()),
    }
    if not record["successful"]:
        record["status"] = "failed"
        record["failure"] = {
            "phase": "synthetic-invariants",
            "message": "Neurofluid synthetic qualification failed.",
            "failed_contracts": [
                name for name, successful in contracts.items() if not successful
            ],
        }
    return record


def _native_source(
    size: int,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[
    phx.geometry.CompartmentMeshingSource,
    phx.discretization.PreparedMetricNetwork,
]:
    contract, labels, compartments, interfaces = _segmentation(
        size, coordinate_budget=coordinate_budget
    )

    def author() -> tuple[
        phx.geometry.CompartmentMeshingSource,
        phx.discretization.PreparedMetricNetwork,
    ]:
        lower, upper = -0.5, size - 0.5
        points = np.asarray(
            (
                (lower, lower, lower),
                (upper, lower, lower),
                (upper, upper, lower),
                (lower, upper, lower),
                (lower, lower, upper),
                (upper, lower, upper),
                (upper, upper, upper),
                (lower, upper, upper),
            ),
            dtype=np.float64,
        )
        triangles = np.asarray(
            (
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (3, 7, 6),
                (3, 6, 2),
                (0, 4, 7),
                (0, 7, 3),
                (1, 2, 6),
                (1, 6, 5),
            ),
            dtype=np.int64,
        )
        outer = phx.geometry.SurfaceModel.from_triangles(
            points,
            triangles,
            phx.geometry.SurfaceMetadata(
                source_id="qualification-outer",
                source_revision=labels.label_volume_id,
                coordinate_contract=contract,
                provenance=("synthetic",),
            ),
        )
        source = phx.geometry.CompartmentMeshingSource(
            labels, compartments, outer, interfaces, budget=coordinate_budget
        )
        network = phx.discretization.MetricNetworkPlan.from_arrays(
            np.asarray(((0.0, 0.0, 0.0), (size - 1.0,) * 3), dtype=np.float64),
            np.asarray(((0, 1),), dtype=np.int64),
            contract,
            areas=np.asarray((0.01,), dtype=np.float64),
            perimeters=np.asarray((0.2,), dtype=np.float64),
            root_vertex_ids=np.asarray((0,), dtype=np.int64),
            tip_vertex_ids=np.asarray((1,), dtype=np.int64),
        ).prepare()
        return source, network

    if coordinate_budget is None:
        return author()
    # Eight outer vertices, twelve fixed facets, and a two-node metric edge.
    work_units = 2048
    scratch_bytes = 16384
    result_bytes = 65536
    coordinate_budget.ticket.checkpoint()
    with coordinate_budget.temporary_scope() as lifetime:
        coordinate_budget.admit_work_bound(
            work_units + 128,
            storage_upper=scratch_bytes + result_bytes,
        )
        coordinate_budget.reserve(work_units, scratch_bytes + result_bytes)
        source, network = author()
        with coordinate_budget.live_storage() as storage:
            storage.promote_temporary_storage(lifetime, result_bytes)
            coordinate_budget.retain_owner(
                (source, network), temporary_storage_credit=storage
            )
        coordinate_budget.release_temporary_storage(scratch_bytes)
        coordinate_budget.ticket.checkpoint()
    return source, network


class _CompartmentPhysics(NamedTuple):
    """Declared transport coefficients and initial state of one compartment."""

    porosity: float
    diffusivity: float
    removal_rate: float
    initial_concentration: float


class _NetworkExchangePhysics(NamedTuple):
    """Declared network, bulk-network exchange and tip-reservoir physics.

    ``exchange_coefficients`` holds one permeability per network node and
    ``reservoir_volumes``/``reservoir_coefficients`` one value per tip;
    ``network_initial`` and ``reservoir_initial`` are the initial states.
    Each node samples the bulk on a circle of ``averaging_radius`` normal to
    its tangent at ``averaging_angular_points`` equally weighted angles.
    """

    network_diffusivity: tuple[float, ...]
    network_volume_flow: tuple[float, ...]
    exchange_coefficients: tuple[float, ...]
    averaging_radius: float
    averaging_angular_points: int
    reservoir_volumes: tuple[float, ...]
    reservoir_coefficients: tuple[float, ...]
    network_initial: tuple[float, ...]
    reservoir_initial: tuple[float, ...]


# One diffusive edge with a permeable node at each end and a tip reservoir.
_NETWORK_EXCHANGE_PHYSICS = _NetworkExchangePhysics(
    network_diffusivity=(0.01,),
    network_volume_flow=(0.0,),
    exchange_coefficients=(0.5, 0.5),
    averaging_radius=0.01,
    # The product's CircleAverageKernel default; the stencil is recomputed
    # independently from this declaration.
    averaging_angular_points=12,
    reservoir_volumes=(0.003,),
    reservoir_coefficients=(0.2,),
    network_initial=(1.0, 1.5),
    reservoir_initial=(0.5,),
)


class NativeTransportCase(NamedTuple):
    """Declared image interpretation, source, network and per-material physics.

    Units are millimeter, second and millimolar. ``velocity`` is one uniform,
    divergence-free bulk velocity; boundary faces with inward volume flux carry
    ``inflow_concentration``. ``exchange`` declares the network, exchange and
    reservoir physics. ``size`` is the image edge count retained by the
    transport checkpoint.
    """

    name: str
    role: str
    source: phx.geometry.CompartmentMeshingSource
    network: phx.discretization.PreparedMetricNetwork
    physics: dict[str, _CompartmentPhysics]
    velocity: tuple[float, float, float]
    inflow_concentration: float
    exchange: _NetworkExchangePhysics
    size: int


NATIVE_TRANSPORT_CASE_ROLES: dict[str, str] = {
    "two-slab": "routine",
    "oblique-junction-transport": "combined",
}

# Sheared, anisotropic voxel axes with a physical offset in millimeters.
_JUNCTION_AFFINE = np.asarray(
    (
        (0.9, 0.15, 0.1, 1.5),
        (0.0, 0.8, 0.2, -2.0),
        (0.05, 0.0, 1.1, 0.75),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
_JUNCTION_DEPTH = 5
_JUNCTION_LABELS = (
    (1, "alpha"),
    (2, "beta"),
    (3, "gamma"),
    (4, "delta"),
    (5, "cap"),
    (6, "tiny"),
)
# Face-sharing material pairs of the authored image; the diagonal quadrant
# pairs meet only along the quadruple line and are not interfaces.
_JUNCTION_INTERFACES = (
    ("alpha", "beta"),
    ("alpha", "cap"),
    ("alpha", "gamma"),
    ("beta", "cap"),
    ("beta", "delta"),
    ("cap", "delta"),
    ("cap", "gamma"),
    ("cap", "tiny"),
    ("delta", "gamma"),
)
_JUNCTION_PHYSICS = {
    "alpha": _CompartmentPhysics(0.8, 0.01, 0.0, 3.0),
    "beta": _CompartmentPhysics(0.7, 0.02, 0.0, 2.0),
    "gamma": _CompartmentPhysics(0.9, 0.005, 0.0, 1.5),
    "delta": _CompartmentPhysics(0.6, 0.015, 0.0, 2.5),
    "cap": _CompartmentPhysics(0.75, 0.01, 0.05, 1.0),
    "tiny": _CompartmentPhysics(0.5, 0.03, 0.5, 4.0),
}
_JUNCTION_VELOCITY = (0.02, 0.01, -0.015)
_JUNCTION_INFLOW_CONCENTRATION = 0.8
# Outward triangles of the unit index box whose corners follow
# itertools.product((0, 1), repeat=3) for a positively oriented affine.
_INDEX_BOX_TRIANGLES = np.asarray(
    (
        (0, 1, 3),
        (0, 3, 2),
        (4, 7, 5),
        (4, 6, 7),
        (0, 5, 1),
        (0, 4, 5),
        (2, 7, 6),
        (2, 3, 7),
        (0, 6, 4),
        (0, 2, 6),
        (1, 7, 3),
        (1, 5, 7),
    ),
    dtype=np.int64,
)


def _junction_source(
    size: int,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[
    phx.geometry.CompartmentMeshingSource,
    phx.discretization.PreparedMetricNetwork,
]:
    """Author four quadrant materials under a cap with one enclosed voxel material.

    The quadrants meet along a quadruple line; quadrant pairs and the cap meet
    along triple lines and a quintuple point. The one-voxel ``tiny`` material is
    required. Occupied voxels are interpreted under an oblique anisotropic affine.
    """

    if size < 4:
        raise ValueError(
            f"transport.oblique-junction-transport.size must be at least four voxels; got {size!r}"
        )
    if coordinate_budget is None:
        # A standalone source author owns one bounded authoring operation.
        # The qualification pipeline always supplies its complete-request ledger.
        limits = phx.meshing.MeshingLimits()
        with NativeExecutionScope(limits.native_limits()) as source_scope:
            with source_scope.batch(
                limits.maximum_work_units,
                queries_upper=limits.maximum_geometry_queries,
                bytes_upper=limits.maximum_scratch_bytes,
                cavity_upper=limits.maximum_cavity_cells,
            ) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                with ticket.native_target(), budget.activate():
                    return _junction_source(size, coordinate_budget=budget)
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="qualification",
    )
    affine = phx.imaging.ImageIndexAffine(
        _JUNCTION_AFFINE, "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    plane_count = size * size
    voxel_count = plane_count * _JUNCTION_DEPTH
    raw_bytes = 2 * voxel_count
    # meshgrid, nested where outputs, and their two predicates coexist.
    quadrant_scratch = 48 * plane_count + 4096
    # The serialized payload and LabelVolume selected-value scan are 2N each,
    # beyond the still-live raw int16 image.
    label_scratch = 4 * voxel_count + 8192
    label_bytes = 3 * voxel_count + 16384
    quadrant_work = 8 * plane_count + voxel_count
    label_work = 8 * voxel_count + 256
    label_definitions = tuple(
        phx.imaging.LabelDefinition(value, f"{name}-label", name.capitalize())
        for value, name in _JUNCTION_LABELS
    )
    coordinate_budget.ticket.checkpoint()
    with coordinate_budget.temporary_scope() as lifetime:
        coordinate_budget.admit_work_bound(
            quadrant_work + label_work + 64,
            storage_upper=raw_bytes
            + max(quadrant_scratch, label_scratch + label_bytes),
        )
        coordinate_budget.reserve(quadrant_work, raw_bytes + quadrant_scratch)
        half = size // 2
        rows, columns = np.meshgrid(
            np.arange(size), np.arange(size), indexing="ij"
        )
        quadrants = np.where(
            rows < half,
            np.where(columns < half, 1, 2),
            np.where(columns < half, 3, 4),
        )
        values = np.empty((size, size, _JUNCTION_DEPTH), dtype=np.int16)
        values[:, :, :2] = quadrants[:, :, None]
        values[:, :, 2:] = 5
        values[size - 2, 1, 3] = 6
        del rows, columns, quadrants
        coordinate_budget.release_temporary_storage(quadrant_scratch)
        coordinate_budget.ticket.checkpoint()
        coordinate_budget.reserve(
            label_work, label_scratch + label_bytes
        )
        labels = _qualified_label_volume(
            values,
            affine,
            asset_id="qualification-junction-labels",
            ontology_id="qualification-junction-labels",
            label_definitions=label_definitions,
            transformation_id="synthetic-junction-qualification-generator",
        )
        with coordinate_budget.live_storage() as storage:
            storage.promote_temporary_storage(lifetime, label_bytes)
            coordinate_budget.retain_owner(
                labels, temporary_storage_credit=storage
            )
        del values
        coordinate_budget.release_temporary_storage(raw_bytes + label_scratch)
        coordinate_budget.ticket.checkpoint()
    definitions = tuple(
        phx.geometry.CompartmentDefinition(
            name,
            (f"{name}-label",),
            "material",
            allowed_neighbor_ids=tuple(
                sorted(
                    other
                    for pair in _JUNCTION_INTERFACES
                    if name in pair
                    for other in pair
                    if other != name
                )
            ),
        )
        for _, name in _JUNCTION_LABELS
    )
    interfaces = tuple(
        phx.geometry.CompartmentInterfaceDefinition(
            f"{first}-{second}", first, second, "exchange"
        )
        for first, second in _JUNCTION_INTERFACES
    )
    complex_ = phx.imaging.build_compartment_complex(
        labels, definitions, interfaces, budget=coordinate_budget
    )
    surfaces = phx.imaging.extract_compartment_surfaces(
        labels, complex_, budget=coordinate_budget
    )

    def author_outer() -> tuple[
        phx.geometry.CompartmentMeshingSource,
        phx.discretization.PreparedMetricNetwork,
    ]:
        corners = np.asarray(tuple(product((0, 1), repeat=3)), dtype=np.float64)
        corners = corners * np.asarray(labels.asset.values.shape, dtype=np.float64) - 0.5
        triangles = (
            _INDEX_BOX_TRIANGLES
            if np.linalg.det(_JUNCTION_AFFINE[:3, :3]) > 0.0
            else _INDEX_BOX_TRIANGLES[:, ::-1]
        )
        outer_geometry = affine_image_geometry(
            phx.discretization.CellMesh.from_triangles(
                np.asarray(affine.index_to_world(corners), dtype=np.float64),
                triangles,
            ),
            _JUNCTION_AFFINE,
            tuple(tuple(Fraction(float(value)) for value in row) for row in corners),
            budget=coordinate_budget,
        )
        outer = phx.geometry.SurfaceModel.from_triangles(
            np.asarray(outer_geometry.coordinates, dtype=np.float64),
            triangles,
            phx.geometry.SurfaceMetadata(
                source_id="qualification-junction-outer-exact-affine",
                source_revision=labels.label_volume_id,
                coordinate_contract=contract,
                provenance=("synthetic",),
            ),
        )
        outer_geometry.resolve(outer.mesh, budget=coordinate_budget)
        source = phx.geometry.CompartmentMeshingSource(
            labels,
            complex_,
            outer,
            surfaces,
            outer_geometry=outer_geometry,
            budget=coordinate_budget,
        )
        # Root in the alpha quadrant, tip in the cap; both voxel centers are
        # farther than the averaging radius from every material interface.
        nodes = np.asarray(
            affine.index_to_world(
                np.asarray(
                    ((1.0, 1.0, 0.0), (size - 2.0, size - 2.0, 3.0))
                )
            ),
            dtype=np.float64,
        )
        network = phx.discretization.MetricNetworkPlan.from_arrays(
            nodes,
            np.asarray(((0, 1),), dtype=np.int64),
            contract,
            areas=np.asarray((0.01,), dtype=np.float64),
            perimeters=np.asarray((0.2,), dtype=np.float64),
            root_vertex_ids=np.asarray((0,), dtype=np.int64),
            tip_vertex_ids=np.asarray((1,), dtype=np.int64),
        ).prepare()
        return source, network

    # The independent outer box has eight corners, twelve facets and one
    # two-node network edge; exact affine work is separately priced above.
    work_units = 2048
    scratch_bytes = 16384
    result_bytes = 65536
    coordinate_budget.ticket.checkpoint()
    with coordinate_budget.temporary_scope() as lifetime:
        coordinate_budget.admit_work_bound(
            work_units + 512,
            storage_upper=scratch_bytes + result_bytes,
        )
        coordinate_budget.reserve(work_units, scratch_bytes + result_bytes)
        source, network = author_outer()
        with coordinate_budget.live_storage() as storage:
            storage.promote_temporary_storage(lifetime, result_bytes)
            coordinate_budget.retain_owner(
                (source, network), temporary_storage_credit=storage
            )
        coordinate_budget.release_temporary_storage(scratch_bytes)
        coordinate_budget.ticket.checkpoint()
    return source, network


def native_transport_case(
    case: str,
    size: int,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> NativeTransportCase:
    """Resolve one declared transport case; unknown names are refused."""
    role = NATIVE_TRANSPORT_CASE_ROLES.get(case)
    if role is None:
        raise ValueError(
            f"transport.case must name a declared case {tuple(NATIVE_TRANSPORT_CASE_ROLES)!r}; got {case!r}"
        )
    if case == "two-slab":
        source, network = _native_source(size, coordinate_budget=coordinate_budget)
        return NativeTransportCase(
            case,
            role,
            source,
            network,
            {
                "first": _CompartmentPhysics(0.8, 0.01, 0.0, 3.0),
                "second": _CompartmentPhysics(0.8, 0.01, 0.0, 2.0),
            },
            (0.0, 0.0, 0.0),
            0.0,
            _NETWORK_EXCHANGE_PHYSICS,
            size,
        )
    source, network = _junction_source(size, coordinate_budget=coordinate_budget)
    return NativeTransportCase(
        case,
        role,
        source,
        network,
        dict(_JUNCTION_PHYSICS),
        _JUNCTION_VELOCITY,
        _JUNCTION_INFLOW_CONCENTRATION,
        _NETWORK_EXCHANGE_PHYSICS,
        size,
    )


def _declared_transport_case(
    source: phx.geometry.CompartmentMeshingSource,
    size: int,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> NativeTransportCase:
    """Recover the declared case with the same scientific source and label revision."""
    for name in NATIVE_TRANSPORT_CASE_ROLES:
        if name == "oblique-junction-transport" and size < 4:
            continue
        declared = native_transport_case(
            name, size, coordinate_budget=coordinate_budget
        )
        if (
            declared.source.source_id == source.source_id
            and declared.source.source_revision == source.source_revision
        ):
            return declared
    raise RuntimeError(
        "transport.cold_source must be the source of a declared transport case; "
        f"got source_id={source.source_id!r}, source_revision={source.source_revision!r}, "
        f"size={size!r}"
    )


def _boundary_faces(
    mesh: phx.discretization.CellMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return owner, outward unit normal and area of boundary faces in face order."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise TypeError(
            f"transport.connectivity must be TetrahedralConnectivity; got {type(connectivity).__name__}"
        )
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    incidence = np.zeros(faces.shape[0], dtype=np.int64)
    owner = np.full(faces.shape[0], -1, dtype=np.int64)
    for cell, row in enumerate(np.asarray(connectivity.cell_faces, dtype=np.int64)):
        incidence[row] += 1
        owner[row] = cell
    rows = np.flatnonzero(incidence == 1)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[faces[rows]]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    normal /= norm[:, None]
    outward = np.mean(corners, axis=1) - np.mean(points[cells[owner[rows]]], axis=1)
    normal *= np.where(np.einsum("ij,ij->i", normal, outward) < 0.0, -1.0, 1.0)[:, None]
    return owner[rows], normal, 0.5 * norm


def _transport_parameters(
    result: phx.meshing.CellMeshingResult,
    case: NativeTransportCase,
) -> phx.applications.neurofluid.NeurofluidTransportParameters:
    """Assign declared material physics by certified region evidence."""
    evidence = result.region_evidence
    count = result.mesh.entity_set(3).count
    if (
        evidence is None
        or len(evidence.cell_region_ids) != count
        or set(evidence.cell_region_ids) != set(case.physics)
    ):
        raise RuntimeError(
            "transport.parameters must assign declared physics to every certified compartment; "
            f"got regions={None if evidence is None else sorted(set(evidence.cell_region_ids))!r}, "
            f"declared={sorted(case.physics)!r}"
        )
    physics = [case.physics[region] for region in evidence.cell_region_ids]
    _, normal, area = _boundary_faces(result.mesh)
    boundary_count = np.count_nonzero(
        np.asarray(result.mesh.entity_set(2).subset("boundary").mask)
    )
    if area.size != boundary_count:
        raise RuntimeError(
            "transport.boundary_faces must match the mesh boundary entity set; "
            f"got {area.size}, expected {boundary_count}"
        )
    velocity = np.asarray(case.velocity, dtype=np.float64)
    flux = area * (normal @ velocity)
    flux = np.where(flux == 0.0, 0.0, flux)
    return phx.applications.neurofluid.NeurofluidTransportParameters(
        porosity=np.asarray([value.porosity for value in physics], dtype=np.float64),
        bulk_diffusivity=np.asarray(
            [value.diffusivity for value in physics], dtype=np.float64
        ),
        bulk_velocity=np.broadcast_to(velocity, (count, 3)).copy(),
        bulk_boundary_volume_flux=flux,
        bulk_boundary_inflow_concentration=np.where(
            flux < 0.0, case.inflow_concentration, 0.0
        ),
        bulk_removal_rate=np.asarray(
            [value.removal_rate for value in physics], dtype=np.float64
        ),
        units=phx.applications.neurofluid.NeurofluidTransportUnits(
            phx.units.MILLIMETER, phx.units.SECOND, phx.units.MILLIMOLAR
        ),
        **_network_parameter_values(case.exchange),
    )


def _network_parameter_values(exchange: _NetworkExchangePhysics) -> dict[str, Any]:
    """Product network/exchange/reservoir parameters of the declared physics."""
    return {
        "network_diffusivity": np.asarray(exchange.network_diffusivity, dtype=np.float64),
        "network_volume_flow": np.asarray(exchange.network_volume_flow, dtype=np.float64),
        "exchange_coefficients": np.asarray(
            exchange.exchange_coefficients, dtype=np.float64
        ),
        "averaging_radius": float(exchange.averaging_radius),
        "reservoir_volumes": np.asarray(exchange.reservoir_volumes, dtype=np.float64),
        "reservoir_coefficients": np.asarray(
            exchange.reservoir_coefficients, dtype=np.float64
        ),
    }


def _network_parameter_error(
    exchange: _NetworkExchangePhysics,
    parameters: phx.applications.neurofluid.NeurofluidTransportParameters,
) -> float:
    """Largest deviation of product network controls from the declared ones.

    A shape mismatch is an unbounded deviation.
    """
    error = 0.0
    for name, declared in _network_parameter_values(exchange).items():
        expected = np.asarray(declared, dtype=np.float64)
        actual = np.asarray(getattr(parameters, name), dtype=np.float64)
        if actual.shape != expected.shape:
            return math.inf
        error = max(error, float(np.max(np.abs(actual - expected), initial=0.0)))
    return error


class _VoxelReference(NamedTuple):
    """Exact occupied-voxel geometry derived from the source image alone."""

    compartments: tuple[str, ...]
    voxels: np.ndarray
    linear: np.ndarray
    offset: np.ndarray
    counts: np.ndarray
    volumes: np.ndarray
    interface_areas: dict[tuple[int, int], float]
    junction_length: float
    quadruple_line_length: float
    quadruple_points: np.ndarray
    quintuple_points: np.ndarray


def _distinct_compartments(stack: np.ndarray) -> np.ndarray:
    """Count distinct nonnegative compartment indices along the last axis."""
    ordered = np.sort(stack, axis=-1)
    fresh = np.concatenate(
        (
            ordered[..., :1] >= 0,
            (ordered[..., 1:] != ordered[..., :-1]) & (ordered[..., 1:] >= 0),
        ),
        axis=-1,
    )
    return np.count_nonzero(fresh, axis=-1)


def _voxel_complex(
    compartments: tuple[str, ...],
    voxels: np.ndarray,
    linear: np.ndarray,
    offset: np.ndarray,
) -> _VoxelReference:
    """Measure materials, interfaces and junctions of an indexed voxel image.

    ``voxels`` holds compartment indices, ``-1`` outside the domain. Voxel ``i``
    occupies the index box ``[i - 1/2, i + 1/2]`` mapped by
    ``x = linear @ i + offset``. Volumes are ``|det linear|`` per voxel; a face
    normal to index axis ``a`` has the area of the cross product of the two
    transformed edge vectors spanning it. Only face-sharing voxels form an
    interface, so diagonal neighbours meeting along an edge contribute none.
    Junction lines are voxel edges around which at least three (quadruple
    lines: four) materials meet. Junction points are the zero-dimensional
    strata among voxel corners shared by at least four (five) materials: a
    corner whose eight incident labels (exterior included) repeat across it
    along some index axis lies inside a straight junction line, carries no
    geometric feature, and need not be a mesh vertex. The exterior is not a
    material.
    """
    counts = np.bincount(voxels[voxels >= 0], minlength=len(compartments))
    areas: dict[tuple[int, int], float] = {}
    for axis in range(3):
        first, second = (value for value in range(3) if value != axis)
        face = float(np.linalg.norm(np.cross(linear[:, first], linear[:, second])))
        moved = np.moveaxis(voxels, axis, 0)
        lower, upper = moved[:-1], moved[1:]
        mask = (lower >= 0) & (upper >= 0) & (lower != upper)
        pairs = np.stack(
            (
                np.minimum(lower[mask], upper[mask]),
                np.maximum(lower[mask], upper[mask]),
            ),
            axis=1,
        )
        if pairs.size:
            unique, multiplicity = np.unique(pairs, axis=0, return_counts=True)
            for (low, high), number in zip(unique, multiplicity, strict=True):
                key = (int(low), int(high))
                areas[key] = areas.get(key, 0.0) + int(number) * face
    padded = np.pad(voxels, 1, constant_values=-1)
    triple_length = 0.0
    quadruple_length = 0.0
    for axis in range(3):
        moved = np.moveaxis(padded, axis, 0)[1:-1]
        around = _distinct_compartments(
            np.stack(
                (
                    moved[:, :-1, :-1],
                    moved[:, 1:, :-1],
                    moved[:, :-1, 1:],
                    moved[:, 1:, 1:],
                ),
                axis=-1,
            )
        )
        edge = float(np.linalg.norm(linear[:, axis]))
        triple_length += np.count_nonzero(around >= 3) * edge
        quadruple_length += np.count_nonzero(around >= 4) * edge
    shape = voxels.shape
    incident_labels = np.stack(
        [
            padded[
                first : first + shape[0] + 1,
                second : second + shape[1] + 1,
                third : third + shape[2] + 1,
            ]
            for first, second, third in product((0, 1), repeat=3)
        ],
        axis=-1,
    )
    incident = _distinct_compartments(incident_labels)
    block = incident_labels.reshape(incident_labels.shape[:3] + (2, 2, 2))
    extruded = (
        np.all(block[..., 0, :, :] == block[..., 1, :, :], axis=(-2, -1))
        | np.all(block[..., :, 0, :] == block[..., :, 1, :], axis=(-2, -1))
        | np.all(block[..., :, :, 0] == block[..., :, :, 1], axis=(-2, -1))
    )

    def corners(minimum: int) -> np.ndarray:
        lattice = np.argwhere((incident >= minimum) & ~extruded).astype(np.float64)
        return (lattice - 0.5) @ linear.T + offset

    return _VoxelReference(
        compartments,
        voxels,
        linear,
        offset,
        counts,
        counts * abs(float(np.linalg.det(linear))),
        areas,
        float(triple_length),
        float(quadruple_length),
        corners(4),
        corners(5),
    )


def _voxel_reference(source: phx.geometry.CompartmentMeshingSource) -> _VoxelReference:
    """Measure materials, interfaces and junctions of the occupied-voxel image."""
    labels = source.labels
    ontology = {value.label_id: int(value.value) for value in labels.ontology.labels}
    definitions = source.compartments.compartments
    compartments = tuple(sorted(value.compartment_id for value in definitions))
    data = np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask, dtype=np.bool_)
    voxels = np.full(data.shape, -1, dtype=np.int64)
    for definition in definitions:
        index = compartments.index(definition.compartment_id)
        for label_id in definition.label_ids:
            voxels[valid & (data == ontology[label_id])] = index
    matrix = np.asarray(labels.asset.spatial_affine.matrix, dtype=np.float64)
    return _voxel_complex(compartments, voxels, matrix[:3, :3], matrix[:3, 3])


class _IndependentBulk(NamedTuple):
    """Independently assembled bulk transport and its material decomposition."""

    reference: _VoxelReference
    centers: np.ndarray
    regions: np.ndarray
    mass: np.ndarray
    matrix: np.ndarray
    forcing: np.ndarray
    pair_flux: dict[tuple[int, int], np.ndarray]
    outflow: np.ndarray
    inflow: np.ndarray
    removal: np.ndarray
    diameters: np.ndarray
    geometry: _TransportGeometry


class _BulkAssembly(NamedTuple):
    """Cell-centered ``mass * du/dt + matrix @ u + forcing = 0`` and its parts."""

    volumes: np.ndarray
    centers: np.ndarray
    mass: np.ndarray
    matrix: np.ndarray
    forcing: np.ndarray
    divergence: np.ndarray
    pair_flux: dict[tuple[int, int], np.ndarray]
    pair_area: dict[tuple[int, int], float]
    pair_count: dict[tuple[int, int], int]
    outflow: np.ndarray
    inflow: np.ndarray
    removal: np.ndarray
    boundary_flux: np.ndarray


def _assemble_bulk(
    points: np.ndarray,
    cells: np.ndarray,
    regions: np.ndarray,
    porosity: np.ndarray,
    diffusivity: np.ndarray,
    removal_rate: np.ndarray,
    velocity: np.ndarray,
    inflow_concentration: float,
    face_order: dict[tuple[int, ...], int] | None = None,
) -> _BulkAssembly:
    """Assemble porous advection-diffusion-removal on tetrahedra from first principles.

    Storage is ``porosity * volume``. Each interior face carries the two-point
    diffusive flux with the harmonic mean of the two cell diffusivities over
    the centroid separation projected on the face normal, plus first-order
    upwind advection of the face volume flux ``area * v . n`` of the uniform
    velocity ``v``. Boundary faces with outward volume flux remove
    ``flux * u``; inward faces add ``-flux * inflow_concentration``. Removal is
    ``porosity * volume * rate * u``. ``pair_flux[(a, b)]`` is the linear
    functional of the state giving the total flux from material ``a`` to
    ``b``. Boundary rows follow ``face_order`` (sorted vertex tuple to face
    row), else sorted vertex tuples.
    """
    count = len(cells)
    tetrahedra = points[cells]
    volumes = np.abs(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])) / 6.0
    centers = np.mean(tetrahedra, axis=1)
    velocity = np.asarray(velocity, dtype=np.float64)
    faces: dict[tuple[int, ...], list[int]] = {}
    for cell, vertices in enumerate(cells.tolist()):
        for face in combinations(sorted(vertices), 3):
            faces.setdefault(face, []).append(cell)
    mass = porosity * volumes
    matrix = np.zeros((count, count), dtype=np.float64)
    forcing = np.zeros(count, dtype=np.float64)
    divergence = np.zeros(count, dtype=np.float64)
    pair_flux: dict[tuple[int, int], np.ndarray] = {}
    pair_area: dict[tuple[int, int], float] = {}
    pair_count: dict[tuple[int, int], int] = {}
    boundary: list[tuple[Any, int, tuple[int, ...]]] = []
    for face, adjacent in faces.items():
        if len(adjacent) == 1:
            order = face if face_order is None else face_order[face]
            boundary.append((order, adjacent[0], face))
            continue
        if len(adjacent) != 2:
            raise RuntimeError(
                "transport.face_incidence must contain at most two cells per face; "
                f"got {len(adjacent)}"
            )
        owner, neighbor = adjacent
        vertices = points[np.asarray(face, dtype=np.int64)]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        norm = float(np.linalg.norm(normal))
        unit = normal / norm
        if np.dot(unit, centers[neighbor] - centers[owner]) < 0.0:
            unit = -unit
        area = 0.5 * norm
        distance = float(np.dot(centers[neighbor] - centers[owner], unit))
        total = diffusivity[owner] + diffusivity[neighbor]
        transmissibility = (
            area * 2.0 * diffusivity[owner] * diffusivity[neighbor] / total / distance
            if total > 0.0
            else 0.0
        )
        flux = area * float(np.dot(velocity, unit))
        divergence[owner] += flux
        divergence[neighbor] -= flux
        # Outflow from owner into neighbor as a linear functional of the state.
        terms = [(owner, transmissibility), (neighbor, -transmissibility)]
        terms.append((owner, flux) if flux >= 0.0 else (neighbor, flux))
        row: np.ndarray | None = None
        sign = 1.0
        if regions[owner] != regions[neighbor]:
            key = (
                int(min(regions[owner], regions[neighbor])),
                int(max(regions[owner], regions[neighbor])),
            )
            sign = 1.0 if regions[owner] == key[0] else -1.0
            row = pair_flux.setdefault(key, np.zeros(count, dtype=np.float64))
            pair_area[key] = pair_area.get(key, 0.0) + area
            pair_count[key] = pair_count.get(key, 0) + 1
        for column, value in terms:
            matrix[owner, column] += value
            matrix[neighbor, column] -= value
            if row is not None:
                row[column] += sign * value
    boundary.sort(key=lambda item: item[0])
    outflow = np.zeros(count, dtype=np.float64)
    inflow = np.zeros(count, dtype=np.float64)
    boundary_flux = np.zeros(len(boundary), dtype=np.float64)
    for index, (_, owner, face) in enumerate(boundary):
        vertices = points[np.asarray(face, dtype=np.int64)]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        norm = float(np.linalg.norm(normal))
        unit = normal / norm
        if np.dot(unit, np.mean(vertices, axis=0) - centers[owner]) < 0.0:
            unit = -unit
        flux = 0.5 * norm * float(np.dot(velocity, unit))
        boundary_flux[index] = flux
        divergence[owner] += flux
        if flux >= 0.0:
            matrix[owner, owner] += flux
            outflow[owner] += flux
        else:
            forcing[owner] += flux * inflow_concentration
            inflow[owner] -= flux * inflow_concentration
    removal = mass * removal_rate
    matrix[np.diag_indices(count)] += removal
    return _BulkAssembly(
        volumes,
        centers,
        mass,
        matrix,
        forcing,
        divergence,
        pair_flux,
        pair_area,
        pair_count,
        outflow,
        inflow,
        removal,
        boundary_flux,
    )


def _material_containment_violations(
    reference: _VoxelReference,
    points: np.ndarray,
    cells: np.ndarray,
    centers: np.ndarray,
    regions: np.ndarray,
) -> int:
    """Count tetrahedra whose centroid or a vertex leaves their material voxels."""
    inverse = np.linalg.inv(reference.linear)
    shape = np.asarray(reference.voxels.shape)
    centroid = np.floor((centers - reference.offset) @ inverse.T + 0.5).astype(np.int64)
    inside = np.all((centroid >= 0) & (centroid < shape), axis=1)
    clipped = np.clip(centroid, 0, shape - 1)
    violations = int(np.count_nonzero(~inside)) + int(
        np.count_nonzero(inside & (reference.voxels[tuple(clipped.T)] != regions))
    )
    index = (points - reference.offset) @ inverse.T
    low = np.clip(np.floor(index + 0.5 - 1.0e-9).astype(np.int64), 0, shape - 1)
    high = np.clip(np.floor(index + 0.5 + 1.0e-9).astype(np.int64), 0, shape - 1)
    for cell, region in enumerate(regions.tolist()):
        for vertex in cells[cell].tolist():
            closure = reference.voxels[
                low[vertex, 0] : high[vertex, 0] + 1,
                low[vertex, 1] : high[vertex, 1] + 1,
                low[vertex, 2] : high[vertex, 2] + 1,
            ]
            violations += int(not np.any(closure == region))
    return violations


def _pair_name(compartments: tuple[str, ...], key: tuple[int, int]) -> str:
    return f"{compartments[key[0]]}|{compartments[key[1]]}"


def _independent_bulk(
    source: phx.geometry.CompartmentMeshingSource,
    result: phx.meshing.CellMeshingResult,
    case: NativeTransportCase,
    parameters: phx.applications.neurofluid.NeurofluidTransportParameters,
) -> _IndependentBulk:
    """Recover the voxel complex and assemble declared material transport.

    Each tetrahedron takes the declared physics of its source material, and
    boundary fluxes come from the mesh faces and the declared velocity;
    nothing is read from the product's assembly. Geometry is compared with
    the occupied-voxel image: material volumes, pairwise interface areas and
    facet counts, adjacency, triple and quadruple junction line lengths, and
    quadruple and quintuple junction points. The parameters handed to the
    product must equal the declared ones.
    """
    evidence = result.region_evidence
    if evidence is None:
        raise RuntimeError(
            "transport.region_evidence must contain compartment assignments; got None"
        )
    reference = _voxel_reference(source)
    compartments = reference.compartments
    if set(evidence.cell_region_ids) != set(compartments) or set(case.physics) != set(
        compartments
    ):
        raise RuntimeError(
            "transport.cell_region_ids and declared physics must cover every source compartment; "
            f"got regions={tuple(sorted(set(evidence.cell_region_ids)))!r}, "
            f"physics={tuple(sorted(case.physics))!r}, expected {compartments!r}"
        )
    position = {name: index for index, name in enumerate(compartments)}
    regions = np.asarray(
        [position[name] for name in evidence.cell_region_ids], dtype=np.int64
    )
    cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    tetrahedra = points[cells]
    centers = np.mean(tetrahedra, axis=1)
    diameters = np.max(
        np.linalg.norm(tetrahedra[:, :, None] - tetrahedra[:, None, :], axis=-1),
        axis=(1, 2),
    )
    count = len(cells)
    scale = max(1.0, float(np.sum(reference.volumes)))
    tolerance = 1.0e-10 * scale
    violations = _material_containment_violations(
        reference, points, cells, centers, regions
    )
    if violations:
        # Physics is keyed by the source voxel material, which the certified
        # assignment only names once every tetrahedron stays inside it.
        raise RuntimeError(
            "transport.material_coverage must keep every tetrahedron inside its source material; "
            f"got containment_violations={violations}"
        )
    physics = [case.physics[compartments[region]] for region in regions.tolist()]
    porosity = np.asarray([value.porosity for value in physics], dtype=np.float64)
    diffusivity = np.asarray([value.diffusivity for value in physics], dtype=np.float64)
    removal_rate = np.asarray(
        [value.removal_rate for value in physics], dtype=np.float64
    )
    velocity = np.asarray(case.velocity, dtype=np.float64)
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise RuntimeError(
            f"transport.connectivity must be TetrahedralConnectivity; got {type(connectivity).__name__}"
        )
    face_order = {
        tuple(sorted(row)): index
        for index, row in enumerate(
            np.asarray(connectivity.faces, dtype=np.int64).tolist()
        )
    }
    assembly = _assemble_bulk(
        points,
        cells,
        regions,
        porosity,
        diffusivity,
        removal_rate,
        velocity,
        float(case.inflow_concentration),
        face_order,
    )
    edge_regions: dict[tuple[int, ...], set[int]] = {}
    vertex_regions: dict[int, set[int]] = {}
    for cell, vertices in enumerate(cells.tolist()):
        region = int(regions[cell])
        ordered = sorted(vertices)
        for edge in combinations(ordered, 2):
            edge_regions.setdefault(edge, set()).add(region)
        for vertex in ordered:
            vertex_regions.setdefault(vertex, set()).add(region)

    def line_length(minimum: int) -> float:
        return float(
            sum(
                np.linalg.norm(points[edge[1]] - points[edge[0]])
                for edge, owners in edge_regions.items()
                if len(owners) >= minimum
            )
        )

    def recovered(minimum: int, expected: np.ndarray) -> int:
        found = points[
            [vertex for vertex, owners in vertex_regions.items() if len(owners) >= minimum]
        ].reshape((-1, 3))
        return sum(
            bool(
                found.size
                and np.min(np.linalg.norm(found - point, axis=1)) <= tolerance
            )
            for point in expected
        )

    junction_length = line_length(3)
    quadruple_length = line_length(4)
    quadruple_recovered = recovered(4, reference.quadruple_points)
    quintuple_recovered = recovered(5, reference.quintuple_points)
    boundary_flux = np.asarray(parameters.bulk_boundary_volume_flux, dtype=np.float64)
    boundary_inflow = np.asarray(
        parameters.bulk_boundary_inflow_concentration, dtype=np.float64
    )
    if (
        boundary_flux.shape != assembly.boundary_flux.shape
        or boundary_inflow.shape != assembly.boundary_flux.shape
        or np.asarray(parameters.porosity).shape != (count,)
        or np.asarray(parameters.bulk_diffusivity).shape != (count,)
    ):
        raise RuntimeError(
            "transport.parameters must hold one scalar coefficient per cell and one "
            "boundary value per independent boundary face; "
            f"got boundary={(boundary_flux.shape, boundary_inflow.shape)!r}, "
            f"expected {assembly.boundary_flux.shape!r}"
        )
    consistency = float(
        np.max(np.abs(boundary_flux - assembly.boundary_flux), initial=0.0)
    )
    # Inflow is compared as the injected rate so a face with vanishing normal
    # velocity cannot flip between inflow and outflow on rounding alone.
    declared_inflow = np.where(
        assembly.boundary_flux < 0.0,
        assembly.boundary_flux * float(case.inflow_concentration),
        0.0,
    )
    parameter_error = max(
        float(np.max(np.abs(np.asarray(parameters.porosity) - porosity))),
        float(np.max(np.abs(np.asarray(parameters.bulk_diffusivity) - diffusivity))),
        float(np.max(np.abs(np.asarray(parameters.bulk_removal_rate) - removal_rate))),
        float(np.max(np.abs(np.asarray(parameters.bulk_velocity) - velocity))),
        float(
            np.max(
                np.abs(
                    np.where(boundary_flux < 0.0, boundary_flux * boundary_inflow, 0.0)
                    - declared_inflow
                ),
                initial=0.0,
            )
        ),
        _network_parameter_error(case.exchange, parameters),
    )
    region_volumes = np.bincount(
        regions, weights=assembly.volumes, minlength=len(compartments)
    )
    volume_error = float(np.max(np.abs(region_volumes - reference.volumes)))
    pair_area, pair_count = assembly.pair_area, assembly.pair_count
    keys = sorted(set(pair_area) | set(reference.interface_areas))
    area_error = max(
        (
            abs(pair_area.get(key, 0.0) - reference.interface_areas.get(key, 0.0))
            for key in keys
        ),
        default=0.0,
    )
    evidence_counts: dict[tuple[int, int], int] = {}
    for _, _, _, first, second in evidence.interface_facets:
        key = (
            min(position[first], position[second]),
            max(position[first], position[second]),
        )
        evidence_counts[key] = evidence_counts.get(key, 0) + 1
    voxel_pairs = tuple(
        (compartments[first], compartments[second])
        for first, second in sorted(reference.interface_areas)
    )
    junction_error = abs(junction_length - reference.junction_length)
    quadruple_error = abs(quadruple_length - reference.quadruple_line_length)
    divergence_error = float(np.max(np.abs(assembly.divergence))) if count else 0.0
    if (
        volume_error > tolerance
        or area_error > tolerance
        or evidence_counts != pair_count
        or evidence.adjacency_pairs != voxel_pairs
        or junction_error > tolerance * max(1.0, reference.junction_length)
        or quadruple_error > tolerance * max(1.0, reference.quadruple_line_length)
        or quadruple_recovered != len(reference.quadruple_points)
        or quintuple_recovered != len(reference.quintuple_points)
        or consistency > tolerance
        or parameter_error > tolerance
        or divergence_error > tolerance
    ):
        raise RuntimeError(
            "transport.material_coverage must match the occupied-voxel complex and declared physics; "
            f"got volume_error={volume_error}, interface_area_error={area_error}, "
            f"facet_counts={evidence_counts!r}/{pair_count!r}, "
            f"adjacency={evidence.adjacency_pairs!r}/{voxel_pairs!r}, "
            f"junction_length_error={junction_error}, "
            f"quadruple_line_length_error={quadruple_error}, "
            f"quadruple_points={quadruple_recovered}/{len(reference.quadruple_points)}, "
            f"quintuple_points={quintuple_recovered}/{len(reference.quintuple_points)}, "
            f"boundary_flux_error={consistency}, declared_parameter_error={parameter_error}, "
            f"volume_flux_divergence={divergence_error}"
        )
    geometry: _TransportGeometry = {
        "compartment_ids": compartments,
        "voxel_counts": {
            name: int(value)
            for name, value in zip(compartments, reference.counts, strict=True)
        },
        "compartment_volumes": region_volumes.tolist(),
        "expected_compartment_volumes": reference.volumes.tolist(),
        "volume_error": volume_error,
        "interface_area": float(sum(pair_area.values())),
        "interface_areas": {
            _pair_name(compartments, key): value for key, value in pair_area.items()
        },
        "expected_interface_areas": {
            _pair_name(compartments, key): value
            for key, value in reference.interface_areas.items()
        },
        "interface_area_error": area_error,
        "interface_facets": len(evidence.interface_facets),
        "interface_facets_by_pair": {
            _pair_name(compartments, key): value for key, value in pair_count.items()
        },
        "adjacency_pairs": evidence.adjacency_pairs,
        "voxel_adjacency_pairs": voxel_pairs,
        "junction_edge_length": junction_length,
        "expected_junction_edge_length": reference.junction_length,
        "junction_length_error": junction_error,
        "quadruple_line_length": quadruple_length,
        "expected_quadruple_line_length": reference.quadruple_line_length,
        "quadruple_line_length_error": quadruple_error,
        "quadruple_points": len(reference.quadruple_points),
        "quadruple_points_recovered": int(quadruple_recovered),
        "quintuple_points": len(reference.quintuple_points),
        "quintuple_points_recovered": int(quintuple_recovered),
        "material_containment_violations": violations,
        "boundary_flux_consistency_error": consistency,
        "volume_flux_divergence_error": divergence_error,
        "declared_parameter_error": parameter_error,
        "affine_matrix": np.asarray(
            source.labels.asset.spatial_affine.matrix, dtype=np.float64
        ).tolist(),
        "geometry_tolerance": tolerance,
    }
    return _IndependentBulk(
        reference,
        centers,
        regions,
        assembly.mass,
        assembly.matrix,
        assembly.forcing,
        assembly.pair_flux,
        assembly.outflow,
        assembly.inflow,
        assembly.removal,
        diameters,
        geometry,
    )


class _TransportReference(NamedTuple):
    """Independent affine system ``mass * du/dt + matrix @ u + forcing = 0``."""

    mass: np.ndarray
    matrix: np.ndarray
    forcing: np.ndarray


class _ExchangeSupportSite(TypedDict):
    """Independent support facts of one averaging site."""

    node_compartment: str | None
    support_cells: int
    minimum_weight: float
    weight_sum: float
    single_material: bool
    within_radius: bool


class _ExchangeSupport(TypedDict):
    """Independent exchange support and its stencil comparison."""

    sites: list[_ExchangeSupportSite]
    stencil_error: float
    stencil_tolerance: float
    unlocated_points: int
    shared_face_points: int
    angular_points: int


class _IndependentSystem(NamedTuple):
    bulk: _IndependentBulk
    reference: _TransportReference
    average: np.ndarray
    exchange_weights: np.ndarray
    node_compartments: tuple[int, ...]
    exchange_support: _ExchangeSupport


def _couple_network(
    bulk_mass: np.ndarray,
    bulk_matrix: np.ndarray,
    bulk_forcing: np.ndarray,
    average: np.ndarray,
    nodes: np.ndarray,
    area: float,
    tip: int,
    exchange: _NetworkExchangePhysics,
) -> tuple[_TransportReference, np.ndarray]:
    """Couple a bulk system to one declared diffusive edge and tip reservoir.

    Each node stores ``area * length / 2``; the edge conducts
    ``diffusivity * area / length``. Node ``k`` exchanges
    ``coefficient_k * node_mass_k * (average_k @ u_bulk - u_k)`` with the bulk
    through the averaging rows ``average``; the tip node exchanges
    ``volume * coefficient * (u_tip - u_reservoir)`` with the reservoir. Every
    coupling is a symmetric ``w * d d^T`` with ``d`` the difference functional,
    so it conserves inventory whenever each averaging row sums to one.
    """
    count = bulk_mass.size
    if (
        nodes.shape != (2, 3)
        or average.shape != (2, count)
        or len(exchange.network_diffusivity) != 1
        or len(exchange.exchange_coefficients) != 2
        or len(exchange.reservoir_volumes) != 1
        or len(exchange.reservoir_coefficients) != 1
        or any(value != 0.0 for value in exchange.network_volume_flow)
    ):
        raise RuntimeError(
            "transport.oracle admits one diffusive two-node network edge with one tip reservoir; "
            f"got nodes={nodes.shape!r}, average={average.shape!r}, exchange={exchange!r}"
        )
    length = float(np.linalg.norm(nodes[1] - nodes[0]))
    network_mass = np.full(2, area * length / 2.0, dtype=np.float64)
    reservoir_volume = float(exchange.reservoir_volumes[0])
    reservoir_coefficient = float(exchange.reservoir_coefficients[0])
    mass = np.concatenate(
        (bulk_mass, network_mass, np.asarray((reservoir_volume,), dtype=np.float64))
    )
    matrix = np.zeros((len(mass), len(mass)), dtype=np.float64)
    matrix[:count, :count] = bulk_matrix
    weight = float(exchange.network_diffusivity[0]) * area / length
    matrix[count : count + 2, count : count + 2] = weight * np.asarray(
        ((1.0, -1.0), (-1.0, 1.0)), dtype=np.float64
    )
    exchange_weights = (
        np.asarray(exchange.exchange_coefficients, dtype=np.float64) * network_mass
    )
    for target in range(2):
        difference = np.zeros(len(mass), dtype=np.float64)
        difference[:count] = average[target]
        difference[count + target] = -1.0
        matrix += exchange_weights[target] * np.outer(difference, difference)
    difference = np.zeros(len(mass), dtype=np.float64)
    difference[count + tip], difference[-1] = 1.0, -1.0
    matrix += reservoir_volume * reservoir_coefficient * np.outer(difference, difference)
    forcing = np.concatenate((bulk_forcing, np.zeros(3, dtype=np.float64)))
    return _TransportReference(mass, matrix, forcing), exchange_weights


# Product and independent circle-average rows are sums of equal 1/n weights.
EXCHANGE_STENCIL_TOLERANCE = 1.0e-12
# Barycentric slack admitting a circle point on a shared face to every incident cell.
_LOCATION_SLACK = 1.0e-9


def _node_tangents(
    nodes: np.ndarray, senders: np.ndarray, receivers: np.ndarray
) -> np.ndarray:
    """Normalized sum of the unit directions of the edges incident to each node."""
    direction = nodes[receivers] - nodes[senders]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    tangents = np.zeros_like(nodes)
    np.add.at(tangents, senders, direction)
    np.add.at(tangents, receivers, direction)
    return tangents / np.linalg.norm(tangents, axis=1)[:, None]


def _circle_samples(
    centers: np.ndarray, tangents: np.ndarray, radius: float, angular_points: int
) -> np.ndarray:
    """Equally spaced points on the circle of ``radius`` normal to each tangent.

    The in-plane frame is ``e1 = t x r / |t x r|`` and ``e2 = t x e1`` with
    ``r = z`` unless ``|t_z| >= 0.9`` (then ``r = y``); point ``k`` sits at
    angle ``2 pi k / n``. Reversing ``t`` maps the point set onto itself.
    """
    unit = tangents / np.linalg.norm(tangents, axis=1)[:, None]
    helper = np.where(
        (np.abs(unit[:, 2]) < 0.9)[:, None],
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((0.0, 1.0, 0.0)),
    )
    first = np.cross(unit, helper)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(unit, first)
    angle = 2.0 * np.pi * np.arange(angular_points) / angular_points
    return centers[:, None, :] + radius * (
        np.cos(angle)[None, :, None] * first[:, None, :]
        + np.sin(angle)[None, :, None] * second[:, None, :]
    )


def _containing_cells(
    points: np.ndarray, cells: np.ndarray, samples: np.ndarray
) -> list[np.ndarray]:
    """Cells whose closed tetrahedron holds each sample, by barycentric coordinates."""
    corners = points[cells]
    edges = np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1))
    inverse = np.linalg.inv(edges)
    containing = []
    for sample in samples:
        local = np.einsum("cij,cj->ci", inverse, sample - corners[:, 0])
        barycentric = np.concatenate((1.0 - local.sum(axis=1)[:, None], local), axis=1)
        containing.append(
            np.flatnonzero(np.min(barycentric, axis=1) >= -_LOCATION_SLACK)
        )
    return containing


class _ExchangeAverage(NamedTuple):
    """Independently located circle-average rows and their comparison."""

    rows: np.ndarray
    unlocated_points: int
    shared_face_points: int
    stencil_error: float


def _exchange_average(
    points: np.ndarray,
    cells: np.ndarray,
    centers: np.ndarray,
    tangents: np.ndarray,
    radius: float,
    angular_points: int,
    product_rows: np.ndarray,
) -> _ExchangeAverage:
    """Recompute the cell-wise circle average and compare the product's rows.

    Every circle point carries weight ``1 / angular_points`` on the cell that
    contains it. A point on a face shared by several cells may be owned by any
    of them: after every interior point is placed, each shared point goes to
    the containing cell with the most product weight still unexplained, so a
    valid product choice is reproduced while a weight on any cell that holds
    no circle point, or a wrong total, remains a mismatch.
    """
    samples = _circle_samples(centers, tangents, radius, angular_points)
    rows = np.zeros((len(centers), len(cells)), dtype=np.float64)
    comparable = product_rows.shape == rows.shape
    preferred = product_rows if comparable else rows
    weight = 1.0 / angular_points
    unlocated = 0
    shared = 0
    for site, site_samples in enumerate(samples):
        ambiguous = []
        for containing in _containing_cells(points, cells, site_samples):
            if containing.size == 0:
                unlocated += 1
            elif containing.size == 1:
                rows[site, int(containing[0])] += weight
            else:
                ambiguous.append(containing)
        shared += len(ambiguous)
        for containing in ambiguous:
            remaining = preferred[site, containing] - rows[site, containing]
            rows[site, int(containing[int(np.argmax(remaining))])] += weight
    error = float(np.max(np.abs(rows - product_rows))) if comparable else math.inf
    return _ExchangeAverage(rows, unlocated, shared, error)


def _independent_system(
    source: phx.geometry.CompartmentMeshingSource,
    result: phx.meshing.CellMeshingResult,
    case: NativeTransportCase,
    runtime: Any,
    network: phx.discretization.PreparedMetricNetwork,
    parameters: phx.applications.neurofluid.NeurofluidTransportParameters,
) -> _IndependentSystem:
    """Couple the bulk with the declared two-node network and tip reservoir.

    Network geometry and every network, exchange and reservoir coefficient
    come from the declared case. The circle-average stencil is recomputed by
    independent point location from the mesh, declared nodes, edge tangents
    and averaging radius; the product's stencil must match it.
    """
    bulk = _independent_bulk(source, result, case, parameters)
    declared = case.network
    if network.network_id != declared.network_id:
        raise RuntimeError(
            "transport.network must be the declared case network; "
            f"got network_id={network.network_id!r}, expected {declared.network_id!r}"
        )
    count = bulk.mass.size
    transfer = runtime.exchange.transfer
    stencil = transfer.sampling.stencil
    indices = np.asarray(stencil.relation.source_indices)
    weights = np.asarray(stencil.weights)
    valid = np.asarray(stencil.relation.valid)
    quadrature = np.asarray(transfer.quadrature_weights)
    sites = quadrature.shape[0]
    nodes = np.asarray(declared.mesh.coordinates, dtype=np.float64)
    tips = np.flatnonzero(np.asarray(declared.tip_mask))
    angular_points = int(case.exchange.averaging_angular_points)
    if sites != len(nodes) or quadrature.shape[1] != angular_points or tips.size != 1:
        raise RuntimeError(
            "transport.oracle admits one declared circle per network node and one tip reservoir; "
            f"got sites={sites}, nodes={len(nodes)}, circle_points={quadrature.shape[1]}, "
            f"declared={angular_points}, tips={tips.tolist()!r}"
        )
    routes = indices.reshape((sites, quadrature.shape[1], -1))
    route_weights = (weights * valid).reshape(routes.shape) * quadrature[:, :, None]
    product_average = np.zeros((sites, count), dtype=np.float64)
    for target in range(sites):
        np.add.at(
            product_average[target],
            routes[target].ravel(),
            route_weights[target].ravel(),
        )
    radius = float(case.exchange.averaging_radius)
    located = _exchange_average(
        np.asarray(result.mesh.coordinates, dtype=np.float64),
        np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks]),
        nodes,
        _node_tangents(
            nodes,
            np.asarray(declared.senders, dtype=np.int64),
            np.asarray(declared.receivers, dtype=np.int64),
        ),
        radius,
        angular_points,
        product_average,
    )
    if located.unlocated_points or located.stencil_error > EXCHANGE_STENCIL_TOLERANCE:
        raise RuntimeError(
            "transport.exchange_stencil must equal the independently located circle average; "
            f"got unlocated_points={located.unlocated_points}, "
            f"stencil_error={located.stencil_error}, allowance={EXCHANGE_STENCIL_TOLERANCE}"
        )
    average = located.rows
    reference = bulk.reference
    node_voxels = np.floor(
        (nodes - reference.offset) @ np.linalg.inv(reference.linear).T + 0.5
    ).astype(np.int64)
    shape = np.asarray(reference.voxels.shape)
    node_compartments = tuple(
        int(reference.voxels[tuple(voxel)])
        if np.all((voxel >= 0) & (voxel < shape))
        else -1
        for voxel in node_voxels
    )
    support_rows: list[_ExchangeSupportSite] = []
    for target in range(sites):
        support = np.flatnonzero(average[target] != 0.0)
        distance = np.linalg.norm(bulk.centers[support] - nodes[target], axis=1)
        support_rows.append(
            {
                "node_compartment": None
                if node_compartments[target] < 0
                else reference.compartments[node_compartments[target]],
                "support_cells": int(support.size),
                "minimum_weight": float(np.min(route_weights[target])),
                "weight_sum": float(np.sum(average[target])),
                "single_material": bool(
                    support.size
                    and node_compartments[target] >= 0
                    and np.all(bulk.regions[support] == node_compartments[target])
                ),
                "within_radius": bool(
                    np.all(distance <= radius + bulk.diameters[support])
                ),
            }
        )
    system, exchange_weights = _couple_network(
        bulk.mass,
        bulk.matrix,
        bulk.forcing,
        average,
        nodes,
        float(np.asarray(declared.areas, dtype=np.float64)[0]),
        int(tips[0]),
        case.exchange,
    )
    return _IndependentSystem(
        bulk,
        system,
        average,
        exchange_weights,
        node_compartments,
        {
            "sites": support_rows,
            "stencil_error": located.stencil_error,
            "stencil_tolerance": EXCHANGE_STENCIL_TOLERANCE,
            "unlocated_points": located.unlocated_points,
            "shared_face_points": located.shared_face_points,
            "angular_points": angular_points,
        },
    )


def _exact_transport(
    reference: _TransportReference, state: np.ndarray, time: float
) -> np.ndarray:
    """Evaluate the full affine exponential action without forming its dense exponential."""
    count = reference.mass.size
    generator = np.zeros((count + 1, count + 1), dtype=np.float64)
    generator[:count, :count] = -reference.matrix / reference.mass[:, None]
    generator[:count, count] = -reference.forcing / reference.mass
    return expm_multiply(csr_matrix(generator) * time, np.append(state, 1.0))[:count]


def _backward_euler(
    reference: _TransportReference,
    state: np.ndarray,
    *,
    dt: float,
    steps: int,
    homogeneous: bool,
) -> np.ndarray:
    """Apply the independent backward-Euler propagator; tangents omit forcing."""
    factor = lu_factor(np.diag(reference.mass) + dt * reference.matrix)
    forcing = 0.0 if homogeneous else dt * reference.forcing
    value = np.asarray(state, dtype=np.float64)
    for _ in range(steps):
        value = lu_solve(factor, reference.mass * value - forcing)
    return value


def _relative_mass_error(
    reference: _TransportReference, actual: np.ndarray, expected: np.ndarray
) -> float:
    root = np.sqrt(reference.mass)
    return float(
        np.linalg.norm(root * (actual - expected)) / np.linalg.norm(root * expected)
    )


def _transport_history_balance_error(
    reference: _TransportReference,
    initial: np.ndarray,
    states: np.ndarray,
    *,
    dt: float,
) -> float:
    """Check every mixed-carrier step against the independent affine equations."""
    before = np.concatenate((initial[None], states[:-1]), axis=0)
    residuals = (
        reference.mass * (states - before) / dt
        + states @ reference.matrix.T
        + reference.forcing
    )
    return float(np.max(np.abs(residuals)))


def _transport_invariants(
    source: phx.geometry.CompartmentMeshingSource,
    result: phx.meshing.CellMeshingResult,
    case: NativeTransportCase,
    runtime: Any,
    network: phx.discretization.PreparedMetricNetwork,
    parameters: phx.applications.neurofluid.NeurofluidTransportParameters,
    initial: Any,
    history: Any,
    accepted: Any,
    *,
    steps: int,
    dt: float,
    target_error: float,
    balance_tolerance: float,
) -> tuple[_TransportInvariants, _TransportReference]:
    """Close local, material, interface and total inventories against the oracle."""
    system = _independent_system(source, result, case, runtime, network, parameters)
    bulk, reference = system.bulk, system.reference
    mass = reference.mass
    average = system.average
    count = len(initial.bulk)
    first = np.concatenate(
        (
            np.asarray(initial.bulk),
            np.asarray(initial.network),
            np.asarray(initial.reservoirs),
        )
    )
    states = np.concatenate(
        (
            np.asarray(history.bulk),
            np.asarray(history.network),
            np.asarray(history.reservoirs),
        ),
        axis=1,
    )
    balance_error = _transport_history_balance_error(reference, first, states, dt=dt)
    bulk_states = states[:, :count]
    outflow_samples = bulk_states @ bulk.outflow
    removal_samples = bulk_states @ bulk.removal
    inflow_rate = float(np.sum(bulk.inflow))
    external = outflow_samples + removal_samples - inflow_rate
    independent_inventory = states @ mass
    inventory_error = float(
        np.max(
            np.abs(independent_inventory - first @ mass + dt * np.cumsum(external))
        )
    )
    initial_bulk = np.asarray(initial.bulk, dtype=np.float64)
    declared_initial = np.asarray(
        [
            case.physics[bulk.reference.compartments[region]].initial_concentration
            for region in bulk.regions.tolist()
        ],
        dtype=np.float64,
    )
    declared_initial = np.concatenate(
        (
            declared_initial,
            np.asarray(case.exchange.network_initial, dtype=np.float64),
            np.asarray(case.exchange.reservoir_initial, dtype=np.float64),
        )
    )
    initial_state_error = (
        float(np.max(np.abs(first - declared_initial)))
        if first.shape == declared_initial.shape
        else math.inf
    )
    constants: dict[int, float] = {}
    for index in range(len(bulk.reference.compartments)):
        values = initial_bulk[bulk.regions == index]
        if values.size and np.ptp(values) == 0.0:
            constants[index] = float(values[0])
    if any(value not in constants for value in system.node_compartments):
        raise RuntimeError(
            "transport.exchange nodes must lie in materials with uniform initial state; "
            f"got node compartments {system.node_compartments!r}"
        )
    expected_average = np.asarray(
        [constants[value] for value in system.node_compartments], dtype=np.float64
    )
    source_transfer = runtime.exchange.transfer.average(initial.bulk)
    transfer_error = float(
        max(
            np.max(np.abs(np.asarray(source_transfer.values) - expected_average)),
            np.max(np.abs(average @ initial_bulk - expected_average)),
        )
    )
    support = system.exchange_support
    support_valid = all(
        row["single_material"]
        and row["within_radius"]
        and row["minimum_weight"] >= 0.0
        and abs(row["weight_sum"] - 1.0) <= balance_tolerance
        for row in support["sites"]
    )
    exchange_rates = (bulk_states @ average.T - states[:, count : count + 2]) * (
        system.exchange_weights
    )
    before = np.concatenate((first[None], states[:-1]), axis=0)
    changes = (bulk_states - before[:, :count]) * mass[:count] / dt
    compartments = bulk.reference.compartments
    interface_balance = 0.0
    compartment_final = []
    for index in range(len(compartments)):
        mask = bulk.regions == index
        crossing = np.zeros(len(states), dtype=np.float64)
        for (low, high), row in bulk.pair_flux.items():
            if low == index:
                crossing += bulk_states @ row
            elif high == index:
                crossing -= bulk_states @ row
        closure = (
            changes[:, mask].sum(axis=1)
            + crossing
            + bulk_states[:, mask] @ bulk.outflow[mask]
            - float(np.sum(bulk.inflow[mask]))
            + bulk_states[:, mask] @ bulk.removal[mask]
            + exchange_rates @ average[:, mask].sum(axis=1)
        )
        interface_balance = max(interface_balance, float(np.max(np.abs(closure))))
        compartment_final.append(float(states[-1, :count][mask] @ mass[:count][mask]))
    exact = _exact_transport(reference, first, steps * dt)
    physical_error = _relative_mass_error(reference, states[-1], exact)
    minimum = float(np.min(states))
    successful = bool(
        np.all(np.asarray(accepted))
        and initial_state_error == 0.0
        and bool(source_transfer.evidence.successful)
        and np.all(np.isfinite(states))
        and minimum >= -balance_tolerance
        and inventory_error <= balance_tolerance
        and balance_error <= balance_tolerance
        and interface_balance <= balance_tolerance
        and transfer_error <= balance_tolerance
        and support_valid
        and physical_error <= target_error
    )
    return {
        **bulk.geometry,
        "accepted_steps": np.asarray(accepted).tolist(),
        "independent_inventory_initial": float(first @ mass),
        "independent_inventory_final": float(independent_inventory[-1]),
        "inventory_error": inventory_error,
        "local_balance_error": balance_error,
        "interface_inventory_balance_error": interface_balance,
        "material_interface_flux_samples": {
            _pair_name(compartments, key): (bulk_states @ row).tolist()
            for key, row in bulk.pair_flux.items()
        },
        "bulk_network_exchange_samples": exchange_rates.tolist(),
        "external_loss_samples": external.tolist(),
        "boundary_inflow_rate": inflow_rate,
        "boundary_outflow_samples": outflow_samples.tolist(),
        "removal_samples": removal_samples.tolist(),
        # Inventory moved by each external term over the solved trajectory;
        # a term is observable by the closure only above the balance tolerance.
        "integrated_boundary_inflow": float(dt * len(states) * inflow_rate),
        "integrated_boundary_outflow": float(dt * np.sum(outflow_samples)),
        "integrated_removal": float(dt * np.sum(removal_samples)),
        "initial_state_error": initial_state_error,
        "compartment_inventory_final": compartment_final,
        "network_inventory_final": float(
            states[-1, count : count + 2] @ mass[count : count + 2]
        ),
        "reservoir_inventory_final": float(states[-1, -1] * mass[-1]),
        "transfer_error": transfer_error,
        "exchange_support": {**support, "valid": support_valid},
        "minimum_concentration": minimum,
        "physical_error": physical_error,
        "physical_reference": (
            "host augmented-exponential solution of the independently assembled "
            "affine bulk/network/reservoir transport (advection, diffusion, "
            "removal, boundary inflow/outflow, exchange)"
        ),
        "successful": successful,
    }, reference


TIME_CONVERGENCE_REFINEMENTS = (2, 4)
# Backward Euler is first order; each halving must show it against the exact solution.
TIME_CONVERGENCE_MINIMUM_ORDER = 0.75


def _refined_final_state(
    runtime: Any, initial: Any, *, steps: int, dt: float
) -> tuple[np.ndarray, bool]:
    """Advance the native runtime over the same horizon with a refined step."""
    final, (_, accepted) = jax.block_until_ready(
        jax.jit(lambda owner, state: _transport_loop(owner, state, steps=steps, dt=dt))(
            runtime, initial
        )
    )
    return np.asarray(runtime.flatten(final)), bool(jnp.all(accepted))


def _time_convergence(
    runtime: Any,
    initial: Any,
    reference: _TransportReference,
    physical_error: float,
    *,
    steps: int,
    dt: float,
    phase: Callable[[str], None],
) -> dict[str, object]:
    """Measure backward-Euler errors against the exact affine solution as dt halves."""
    exact = _exact_transport(
        reference, np.asarray(runtime.flatten(initial)), steps * dt
    )
    widths, errors, accepted = [dt], [physical_error], True
    for factor in TIME_CONVERGENCE_REFINEMENTS:
        phase(f"time-convergence-dt-over-{factor}")
        final, admitted = _refined_final_state(
            runtime, initial, steps=steps * factor, dt=dt / factor
        )
        accepted = accepted and admitted
        widths.append(dt / factor)
        errors.append(_relative_mass_error(reference, final, exact))
    orders = [
        math.log(previous / following) / math.log(coarse / fine)
        if previous > 0.0 and following > 0.0
        else None
        for previous, following, coarse, fine in zip(
            errors, errors[1:], widths, widths[1:], strict=False
        )
    ]
    strictly_decreasing = all(b < a for a, b in zip(errors, errors[1:]))
    order_met = all(
        order is not None and order >= TIME_CONVERGENCE_MINIMUM_ORDER for order in orders
    )
    return {
        "horizon_seconds": steps * dt,
        "step_widths_seconds": widths,
        "relative_errors": errors,
        "observed_orders": orders,
        "required_minimum_order": TIME_CONVERGENCE_MINIMUM_ORDER,
        "all_steps_accepted": accepted,
        "strictly_decreasing": strictly_decreasing,
        "order_met": order_met,
        "reference": "same augmented-exponential affine solution as physical_error",
        "passed": accepted and strictly_decreasing and order_met,
    }


def _transport_loop(runtime: Any, state: Any, *, steps: int, dt: float) -> Any:
    def advance(current: Any, unused: Any) -> Any:
        del unused
        step = runtime.step_backward_euler(current, dt)
        return step.state, (step.state, step.accepted)

    return jax.lax.scan(advance, state, None, length=steps)


def _network_source_plan(network: Any) -> phx.discretization.MetricNetworkPlan:
    vertex_ids = np.asarray(network.mesh.vertex_global_ids, dtype=np.int64)
    plan = phx.discretization.MetricNetworkPlan(
        network.mesh,
        network.coordinate_contract,
        np.asarray(network.areas),
        np.asarray(network.perimeters),
        vertex_ids[np.asarray(network.root_mask)],
        vertex_ids[np.asarray(network.tip_mask)],
    )
    if plan.prepare().network_id != network.network_id:
        raise RuntimeError(
            "transport.network_id must survive reconstruction from its constructor facts; "
            "got a different prepared identity"
        )
    return plan


def _transport_continuation_error(continued: Any, expected: Any) -> float:
    """Compare every continued field and history without hiding nonfinite defects."""
    error = 0.0
    for first, second in zip(
        jax.tree.leaves(continued), jax.tree.leaves(expected), strict=True
    ):
        if np.asarray(first).dtype.kind == "b":
            continue
        defect = float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
        if not np.isfinite(defect):
            return math.inf
        error = max(error, defect)
    return error


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Bootstrap a distinct original-cap request before restoring any source owner."""
    controls = read_meshing_source_execution_controls(
        packet.archive_path, expected_content_id=packet.content_id
    )
    carrier: list[phx.meshing.CellMeshingResult] = []

    def restore(budget: CoordinateEnclosureBudget) -> ColdWorkerReport:
        return _continue_cold_transport(packet, budget, carrier)

    renewed = execute_native_request(
        controls.execution_evidence,
        controls.limits,
        restore,
        target=lambda _: carrier[0],
        failure_factory=qualification_failure,
        workflow_remaining_seconds=packet.deadline.remaining(),
        observe_managed_admission=True,
    )
    execution = renewed.result.execution_evidence
    if execution is None:
        raise RuntimeError("transport.cold_request must retain its ended execution record.")
    renewed.value.record["native_execution"] = execution.to_record()
    renewed.value.record["native_execution_scope"] = (
        "fresh original-cap reader/reproof/preparation/continued-transport request"
    )
    return renewed.value


def _continue_cold_transport(
    packet: ColdControlPacket,
    budget: CoordinateEnclosureBudget,
    carrier: list[phx.meshing.CellMeshingResult],
) -> ColdWorkerReport:
    started = perf_counter()
    records = read_meshing_source_closure(
        packet.archive_path, expected_content_id=packet.content_id, budget=budget
    )
    read_seconds = perf_counter() - started
    packet.deadline.remaining()
    content_id = packet.content_id
    if array_tree_fingerprint(records) != packet.controls["expected_identity"]:
        raise RuntimeError(
            "transport.cold_closure must preserve source, material, fields, history, and clocks; "
            "got a different content fingerprint"
        )
    checkpoint = records["accepted_data"]["neurofluid_transport"]
    source, result = checkpoint.source, checkpoint.result
    carrier.append(result)
    source.validate_source_integrity(budget=budget)
    result.region_evidence.require_source(source.compartments)
    if result.certification is None:
        raise RuntimeError(
            "transport.cold_certification must retain the complete source theorem; got None"
        )
    renewed = recertify_restored_meshing_source(
        result.certification.request,
        result.mesh,
        result.geometry,
        result.audit,
        archive_limits=ArrayArchiveLimits(),
    )
    renewed.require_passed()
    network = checkpoint.network_plan.prepare()
    case = phx.applications.neurofluid.NeurofluidCase(
        checkpoint.case_id, source.labels, source.compartments, result, network
    )
    plan = phx.applications.neurofluid.NeurofluidTransportPlan(
        case, checkpoint.parameters
    )
    runtime = plan.prepare()
    if (case.case_revision, plan.plan_id, runtime.runtime_id) != checkpoint.physics_ids:
        raise RuntimeError(
            "transport.cold_physics must preserve case, plan, and runtime identities; "
            "got different reconstructed identities"
        )
    steps, dt = checkpoint.steps, checkpoint.dt
    execute = jax.jit(
        lambda owner, state: _transport_loop(owner, state, steps=steps, dt=dt)
    )
    continued = jax.block_until_ready(execute(runtime, checkpoint.final))
    if not bool(jnp.all(continued[1][1])):
        raise RuntimeError(
            "transport.cold_continuation must accept every physical step; got a rejected step"
        )
    error = _transport_continuation_error(continued, checkpoint.continued)
    if not np.isfinite(error) or error > checkpoint.balance_tolerance:
        raise RuntimeError(
            "transport.cold_continuation must match the accepted hot fields; "
            f"got error={error}, allowance={checkpoint.balance_tolerance}"
        )
    restored, reference = _transport_invariants(
        source,
        result,
        _declared_transport_case(
            source, checkpoint.size, coordinate_budget=budget
        ),
        runtime,
        network,
        checkpoint.parameters,
        checkpoint.initial,
        checkpoint.history,
        checkpoint.accepted,
        steps=steps,
        dt=dt,
        target_error=checkpoint.target_error,
        balance_tolerance=checkpoint.balance_tolerance,
    )
    if not restored["successful"]:
        raise RuntimeError(
            "transport.cold_history must pass the independent physical gates; "
            f"got {restored!r}"
        )
    continued_balance_error = _transport_history_balance_error(
        reference,
        np.asarray(runtime.flatten(checkpoint.final)),
        np.concatenate(
            [
                np.asarray(getattr(continued[1][0], name))
                for name in ("bulk", "network", "reservoirs")
            ],
            axis=1,
        ),
        dt=dt,
    )
    if (
        not np.isfinite(continued_balance_error)
        or continued_balance_error > checkpoint.balance_tolerance
    ):
        raise RuntimeError(
            "transport.cold_continued_history must balance every physical step; "
            f"got local_balance_error={continued_balance_error}"
        )
    with TemporaryDirectory(prefix="phydrax-neurofluid-cold-rearchive-") as directory:
        replay = write_meshing_source_closure(Path(directory) / "replayed.zip", records)
        if replay.content_id != content_id:
            raise RuntimeError(
                "transport.cold_rearchive must preserve the scientific closure; "
                f"got content_id={replay.content_id!r}, expected={content_id!r}"
            )
    packet.deadline.remaining()
    record = {
        "passed": True,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        **(
            {"outer_geometry_id": cell_geometry_id(source.outer_geometry)}
            if source.outer_geometry is not None
            else {}
        ),
        "result_id": result.result_id,
        "source_report_id": renewed.report_id,
        "network_id": network.network_id,
        "parameter_id": checkpoint.parameters.parameter_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "allfield_history_bitwise_preserved": True,
        "continued_max_error": error,
        "continued_local_balance_error": continued_balance_error,
        "accepted_step_cursor": checkpoint.step_cursor,
        "accepted_time": checkpoint.time,
        "continued_step_cursor": checkpoint.step_cursor + steps,
        "continued_time": checkpoint.time + steps * dt,
        "prepared_runtime_serialized": False,
        "default_max_members": ArrayArchiveLimits().max_members,
        "default_max_nesting": ArrayArchiveLimits().max_manifest_nesting,
        "default_max_rank": ArrayArchiveLimits().max_array_rank,
    }
    return ColdWorkerReport(record, read_seconds=read_seconds)


def _publish_transport_checkpoint(
    records: MeshingSourceClosure,
    *,
    phase: Callable[[str], None],
    context: _NativeTransportContext,
    deadline: Deadline,
    end_warm_request: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Admit publication before ending warm ownership, then start independent cold work."""

    ledger_record: dict[str, Any] = {}
    ledger = StageLedger(
        ledger_record,
        deadline=deadline,
        cold_phase="fresh-default-cold-allfield-history-continued-native-PDE",
    )
    try:
        with (
            TemporaryDirectory(prefix="phydrax-neurofluid-source-state-") as directory,
            prepare_cold_worker(
                "tools.neurofluid_qualification", deadline=deadline, ledger=ledger
            ) as cold,
        ):
            receipt = write_meshing_source_closure(Path(directory) / "state.zip", records)
            expected_identity = array_tree_fingerprint(records)
            if end_warm_request is not None:
                end_warm_request(receipt.content_id)
            phase("fresh-default-cold-allfield-history-continued-native-PDE")
            evidence = cold.execute(
                receipt.path,
                receipt.content_id,
                {"expected_identity": expected_identity},
            )
            evidence["content_id"] = receipt.content_id
            evidence["process"] = ledger_record["cold_process"]
            return evidence
    finally:
        if "cold_process" in ledger_record:
            context["cold_process"] = ledger_record["cold_process"]


def _transport_lifecycle(
    source: Any,
    result: Any,
    case: Any,
    network: Any,
    parameters: Any,
    runtime: Any,
    initial: Any,
    first: Any,
    execute: Any,
    loop: Any,
    reference: _TransportReference,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    size: int,
    steps: int,
    dt: float,
    target_error: float,
    balance_tolerance: float,
    phase: Callable[[str], None],
) -> tuple[
    _TransportLifecycle,
    phx.applications.neurofluid.NeurofluidTransportCheckpoint,
]:

    final, (history, accepted) = first
    mass = reference.mass
    tangent = runtime.unflatten(
        jnp.sin(jnp.arange(sum(runtime.sizes), dtype=jnp.float64) + 1.0)
    )
    phase("fixed-epoch-native-transport-AD")

    def derivative(owner: Any, state: Any, direction: Any) -> Any:
        return jax.jvp(lambda value: loop(owner, value)[0], (state,), (direction,))

    ad = jax.jit(derivative)
    ad_executable, ad_compilation = measure_lower_and_compile(
        lambda: ad.lower(runtime, initial, tangent), lambda lowered: lowered.compile()
    )
    (ad_primal, ad_value), ad_seconds = measure_synchronized(
        lambda: ad_executable(runtime, initial, tangent)
    )
    tangent_flat = np.asarray(runtime.flatten(tangent))
    expected = _backward_euler(
        reference, tangent_flat, dt=dt, steps=steps, homogeneous=True
    )
    actual = np.asarray(runtime.flatten(ad_value))
    error = _relative_mass_error(reference, actual, expected)
    # Open boundaries and removal change the tangent inventory; the
    # independent propagator fixes the admissible change.
    inventory_defect = float(abs(mass @ (actual - expected)))
    primal_defect = float(
        np.max(
            np.abs(
                np.asarray(runtime.flatten(ad_primal))
                - np.asarray(runtime.flatten(final))
            )
        )
    )
    if (
        not np.isfinite(error)
        or error > target_error
        or inventory_defect > balance_tolerance
        or primal_defect > balance_tolerance
    ):
        raise RuntimeError(
            "transport.AD must pass independent physical and inventory gates; "
            f"got error={error}, inventory_defect={inventory_defect}, primal_defect={primal_defect}"
        )
    phase("continued-hot-native-transport")
    continued, continuation_seconds = measure_synchronized(
        lambda: execute(runtime, final)
    )
    if not bool(jnp.all(continued[1][1])):
        raise RuntimeError(
            "transport.hot_continuation must accept every physical step; got a rejected step"
        )
    final_flat = np.asarray(runtime.flatten(final))
    expected_continued = _backward_euler(
        reference, final_flat, dt=dt, steps=steps, homogeneous=False
    )
    continued_flat = np.asarray(runtime.flatten(continued[0]))
    continued_error = _relative_mass_error(
        reference, continued_flat, expected_continued
    )
    continued_inventory_defect = float(
        abs(mass @ (continued_flat - expected_continued))
    )
    continued_balance_error = _transport_history_balance_error(
        reference,
        final_flat,
        np.concatenate(
            [
                np.asarray(getattr(continued[1][0], name))
                for name in ("bulk", "network", "reservoirs")
            ],
            axis=1,
        ),
        dt=dt,
    )
    if (
        not np.isfinite(continued_error)
        or continued_error > target_error
        or continued_inventory_defect > balance_tolerance
        or not np.isfinite(continued_balance_error)
        or continued_balance_error > balance_tolerance
    ):
        raise RuntimeError(
            "transport.hot_continuation must pass independent physical and inventory gates; "
            f"got error={continued_error}, inventory_defect={continued_inventory_defect}, "
            f"local_balance_error={continued_balance_error}"
        )
    phase("allfield-history-native-resource-rollback")
    before = array_tree_fingerprint((initial, history, accepted, final, parameters))
    negative_policy = phx.linalg.LinearSolvePolicy(
        resources=phx.linalg.SolveResourcePolicy(
            factorization_bytes=0,
            workspace_bytes=0,
            krylov_basis_bytes=0,
            preconditioner_bytes=0,
            recycling_state_bytes=0,
        )
    )
    rejected = jax.jit(
        lambda state: runtime.step_backward_euler(state, dt, policy=negative_policy)
    )(final)
    rejected = jax.block_until_ready(rejected)
    if bool(rejected.accepted):
        raise RuntimeError(
            "transport.storage_refusal must reject a step with no solver storage; got accepted"
        )
    if array_tree_fingerprint(
        (initial, history, accepted, final, parameters)
    ) != before or any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for old, new in zip(
            jax.tree.leaves(final), jax.tree.leaves(rejected.state), strict=True
        )
    ):
        raise RuntimeError(
            "transport.storage_refusal must preserve accepted fields, material, and history; "
            "got changed state"
        )
    phase("default-source-allfield-material-history-checkpoint")
    checkpoint = phx.applications.neurofluid.NeurofluidTransportCheckpoint(
        source=source,
        result=result,
        network_plan=_network_source_plan(network),
        parameters=parameters,
        initial=initial,
        history=history,
        accepted=accepted,
        final=final,
        continued=continued,
        ad_tangent=tangent,
        ad_value=ad_value,
        case_id=case.case_id,
        physics_ids=(
            case.case_revision,
            phx.applications.neurofluid.NeurofluidTransportPlan(case, parameters).plan_id,
            runtime.runtime_id,
        ),
        size=size,
        steps=steps,
        dt=dt,
        step_cursor=steps,
        time=steps * dt,
        target_error=target_error,
        balance_tolerance=balance_tolerance,
        budget=coordinate_budget,
    )
    return {
        "passed": True,
        "field_names": tuple(
            template["name"] for template in native_transport_field_templates()
        ),
        "material_parameter_id": parameters.parameter_id,
        "accepted_history_steps": steps,
        "AD": {
            "relative_discrete_physical_error": error,
            "inventory_defect": inventory_defect,
            "primal_defect": primal_defect,
            "lowering_seconds": ad_compilation.lowering_seconds,
            "compilation_seconds": ad_compilation.compilation_seconds,
            "execution_seconds": ad_seconds,
        },
        "hot_continuation_seconds": continuation_seconds,
        "hot_continuation_relative_physical_error": continued_error,
        "hot_continuation_inventory_defect": continued_inventory_defect,
        "hot_continuation_local_balance_error": continued_balance_error,
        "atomic_rollback": {
            "passed": True,
            "negative_linear_status": int(rejected.linear_status),
            "positive_source_physics_controls_unchanged": True,
        },
        "cold": {},
    }, checkpoint


# User-approved hard sizing request of the shared labeled-image volume: the
# unchanged unit target must hold for both p50 and p95 within +-50 %.
NATIVE_IMAGE_SIZE_COMPLIANCE = phx.meshing.SizeCompliancePolicy(
    relative_tolerance=0.5, target_statistics=("p50", "p95")
)
# The former inherited zero-tolerance request. It is retained only as
# unqualified stress evidence and a compliance-boundary regression; it is not
# a positive profile and its refusal is not an infeasibility claim.
FORMER_EXACT_IMAGE_SIZE_COMPLIANCE = phx.meshing.SizeCompliancePolicy(
    relative_tolerance=0.0, target_statistics=("p50", "p95")
)


def _size_request_record(
    policy: phx.meshing.SizeCompliancePolicy, /
) -> dict[str, object]:
    """Record the hard unit-size request and whether it is the qualified one."""
    return {
        "target_size": 1.0,
        "strength": "HARD",
        "target_statistics": list(policy.target_statistics),
        "relative_tolerance": policy.relative_tolerance,
        "absolute_tolerance": policy.absolute_tolerance,
        "policy_id": policy.policy_id,
        "qualification": (
            "approved-hard-request"
            if policy.policy_id == NATIVE_IMAGE_SIZE_COMPLIANCE.policy_id
            else "unqualified-stress-request"
        ),
    }


def _prepare_transport_generation(
    *,
    case: str,
    size: int,
    maximum_cells: int,
    maximum_vertices: int,
    size_compliance: phx.meshing.SizeCompliancePolicy,
    phase: Callable[[str], None],
    context: _NativeTransportContext,
    deadline: Deadline,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
    source_workspace: NativeHostStorageWorkspace | None = None,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
) -> _TransportGeneration:
    """Author and publish under one original native request allowance."""
    limits = phx.meshing.MeshingLimits(
        maximum_cells=maximum_cells, maximum_vertices=maximum_vertices,
        **({"maximum_work_units": maximum_work_units} if maximum_work_units is not None else {}),
        **({"maximum_geometry_queries": maximum_geometry_queries} if maximum_geometry_queries is not None else {}),
    )
    with (
        NativeParallelExecution(None, deadline.remaining()),
        NativeExecutionScope(limits.native_limits()) as source_scope,
        (
            source_scope.host_workspace()
            if source_workspace is None else nullcontext(source_workspace)
        ) as source_storage,
    ):
        started = perf_counter()
        native_phases: list[dict[str, object]] = []
        context["native_phase_measurements"] = native_phases
    
        def record_native_phase(measurement: NativeMeshingPhaseMeasurement) -> None:
            native_phases.append(asdict(measurement))
    
        if coordinate_budget is not None:
            transport_case = native_transport_case(
                case, size, coordinate_budget=coordinate_budget
            )
        else:
            with source_scope.batch(
                limits.maximum_work_units,
                queries_upper=limits.maximum_geometry_queries,
                bytes_upper=limits.maximum_scratch_bytes,
                cavity_upper=limits.maximum_cavity_cells,
            ) as source_ticket:
                source_budget = CoordinateEnclosureBudget(
                    source_ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                with source_ticket.native_target(), source_budget.activate():
                    transport_case = native_transport_case(
                        case, size, coordinate_budget=source_budget
                    )
        source_storage.retain_owner(transport_case)
        source, network = transport_case.source, transport_case.network
        source_seconds = perf_counter() - started
        context["source"] = {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "source_complex_id": source.compartments.complex_id,
            "coordinate_contract_id": source.coordinate_contract.spatial_id,
            "interpretation": source.interpretation,
            **(
                {"outer_geometry_id": cell_geometry_id(source.outer_geometry)}
                if source.outer_geometry is not None
                else {}
            ),
        }
        phase("request-preparation")
        scope = phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            f"{source.source_id}:boundary",
            np.asarray((0,), dtype=np.int64),
        )
        request = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope, 1.0, strength=phx.meshing.SizeControlStrength.HARD
                ),
            ),
            size_compliance=size_compliance,
            limits=limits,
        )
        context["size_request"] = _size_request_record(size_compliance)
        phase("geometry-preparation")
        started = perf_counter()
        options = phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source,
            request,
            coordinate_contract=source.coordinate_contract,
            record_phase=record_native_phase,
        )
        geometry_preparation = perf_counter() - started
        phase("generation-publication")
        result, generation = measure_synchronized(
            lambda: plan.execute(record_phase=record_native_phase)
        )
        execution = result.execution_evidence
        if execution is None:
            raise RuntimeError(
                "transport.generation.execution_evidence must contain completed resource evidence; got None"
            )
        execution.require_valid()
        if int(execution.status) != int(MeshcoreStatus.OK):
            raise RuntimeError(
                "transport.generation.status must report successful native execution; "
                f"got {execution.status!r}"
            )
        source_storage.retain_owner((plan, result))
    native_execution = NativeExecutionRecord(
        source_scope.completed_evidence(),
        owner_id=plan.plan_id,
        consumer_evidence=execution,
    ).to_record()
    context["native_execution"] = native_execution
    context["identities"] = {
        "mesh_id": result.mesh.mesh_id,
        "result_id": result.result_id,
        "request_id": request.specification_id,
        "provider": result.provider.name,
    }
    return {
        "transport_case": transport_case,
        "source": source,
        "network": network,
        "request": request,
        "options": options,
        "plan": plan,
        "result": result,
        "native_execution": native_execution,
        "native_phases": native_phases,
        "source_seconds": source_seconds,
        "geometry_preparation": geometry_preparation,
        "generation": generation,
    }


def _report_native_transport(
    generated: _TransportGeneration,
    executed: _TransportAcceptedExecution,
    *,
    campaign_started: float,
    size: int,
    repeats: int,
    steps: int,
    dt: float,
    target_error: float,
    balance_tolerance: float,
    maximum_cells: int,
    maximum_vertices: int,
    oracle_capacity: int,
    phase: Callable[[str], None],
    warm_execution_binding: dict[str, str],
) -> NativeTransportRecord:
    """Report physical acceptance, allocation evidence, and loaded build identity."""
    source, network = generated["source"], generated["network"]
    transport_case = generated["transport_case"]
    request, plan, result = generated["request"], generated["plan"], generated["result"]
    native_execution, native_phases = (
        generated["native_execution"],
        generated["native_phases"],
    )
    source_seconds = generated["source_seconds"]
    geometry_preparation = generated["geometry_preparation"]
    generation = generated["generation"]
    case, readmitted = executed["case"], executed["readmitted"]
    transport_plan, runtime = executed["transport_plan"], executed["runtime"]
    evidence, initial, first = (
        executed["evidence"],
        executed["initial"],
        executed["first"],
    )
    admission, preparation = executed["admission"], executed["preparation"]
    compilation, first_seconds = executed["compilation"], executed["first_seconds"]
    warmed, compiler = executed["warmed"], executed["compiler"]
    certification = executed["certification"]
    invariants, lifecycle = executed["invariants"], executed["lifecycle"]
    time_convergence = executed["time_convergence"]
    phase("memory-evidence")
    peak = peak_resident_bytes()
    device_memory: list[dict[str, object]] = []
    for device in jax.local_devices():
        stats = device.memory_stats()
        device_memory.append(
            {
                "device": str(device),
                "peak_bytes": None if stats is None else stats.get("peak_bytes_in_use"),
                "unavailable_reason": (
                    "Backend does not expose device allocation peak."
                    if stats is None or "peak_bytes_in_use" not in stats
                    else None
                ),
            }
        )
    record: NativeTransportRecord = {
        "scenario": "native-generated-transport",
        "case": transport_case.name,
        "qualification_role": transport_case.role,
        "transport_case": {
            "image_shape": list(np.asarray(source.labels.asset.values).shape),
            "affine_matrix": np.asarray(
                source.labels.asset.spatial_affine.matrix, dtype=np.float64
            ).tolist(),
            "compartment_physics": {
                name: value._asdict() for name, value in transport_case.physics.items()
            },
            "velocity_millimeter_per_second": list(transport_case.velocity),
            "boundary_inflow_concentration": transport_case.inflow_concentration,
            "network_exchange_physics": transport_case.exchange._asdict(),
            "network_nodes": np.asarray(network.mesh.coordinates).tolist(),
        },
        "controls": {
            "size": size,
            "maximum_cells": maximum_cells,
            "maximum_vertices": maximum_vertices,
            "oracle_capacity": oracle_capacity,
            "steps": steps,
            "dt_seconds": dt,
            "repeats": repeats,
        },
        "targets": {
            "physical_error": target_error,
            "balance_tolerance": balance_tolerance,
        },
        "size_request": _size_request_record(request.size_compliance),
        "source": {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "source_complex_id": source.compartments.complex_id,
            "coordinate_contract_id": source.coordinate_contract.spatial_id,
            "interpretation": source.interpretation,
            "origin": "synthetic-labeled-image",
            **(
                {"outer_geometry_id": cell_geometry_id(source.outer_geometry)}
                if source.outer_geometry is not None
                else {}
            ),
        },
        "units": {
            "length": "millimeter",
            "time": "second",
            "concentration": "millimolar",
            "inventory": "millimolar*millimeter^3",
        },
        "identities": {
            "mesh_id": result.mesh.mesh_id,
            "result_id": result.result_id,
            "request_id": request.specification_id,
            "region_evidence_id": evidence.evidence_id,
            "coverage_id": evidence.coverage_id,
            "case_revision": readmitted.case_revision,
            "network_id": network.network_id,
            "transport_plan_id": transport_plan.plan_id,
            "runtime_id": runtime.runtime_id,
            "transfer_id": runtime.exchange.transfer.transfer_id,
            "meshcore": meshcore_identity(),
            "provider": result.provider.name,
        },
        "timing": {
            "source_preparation_seconds": source_seconds,
            "geometry_preparation_seconds": geometry_preparation,
            "generation_publication_seconds": generation,
            "readmission_seconds": admission,
            "transport_transfer_preparation_seconds": preparation,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_execution_seconds": first_seconds,
            "warmed": warmed.to_seconds_dict(),
            "independent_certification_seconds": certification,
            "end_to_end_seconds": perf_counter() - campaign_started,
            "explicit_primary_transport_compilations": 1,
        },
        "memory": {
            "compiler": asdict(compiler),
            "logical_retained_bytes": logical_array_bytes(
                (source, plan, result, case, runtime, initial)
            ),
            "logical_output_bytes": logical_array_bytes(first),
            "process_peak_resident_bytes": peak,
            "process_peak_scope": f"{PEAK_RESIDENT_SCOPE};includes-compilation-and-independent-certification",
            "device": device_memory,
        },
        "invariants": invariants,
        "lifecycle": lifecycle,
        "time_convergence": time_convergence,
        "native_execution": native_execution,
        "native_execution_scope": "complete ended source-authoring/generation/science/FE-FV/archive-admission request; independent cold work and time belong to its separate fresh request; archived publication receipt is a prefix only",
        "warm_execution_binding": warm_execution_binding,
        "original_native_limits": asdict(request.limits),
        "publication_resource_guarantees": {
            "enforced": list(result.runtime.enforced_limits),
            "unenforced": list(result.runtime.unenforced_limits),
        },
        "publication_resource_binding": native_publication_resource_binding(
            result.result_id, result.runtime.runtime_id, result.execution_evidence,
            enforced_limits=result.runtime.enforced_limits,
            unenforced_limits=result.runtime.unenforced_limits,
        ),
        "successful": bool(
            invariants["successful"]
            and lifecycle["passed"]
            and time_convergence["passed"] is True
        ),
        "comparison": {"status": "not-requested"},
    }
    record["native_phase_measurements"] = native_phases
    phase("runtime-build-identity")
    root = Path(__file__).resolve().parents[1]
    record["native_build_identity"] = meshcore_runtime_identity()
    record["loaded_python_source_digest"] = python_source_revision(
        Path(phx.__file__).resolve().parent
    )
    record["build_identity"] = capture_benchmark_identity(
        root,
        Path(__file__).resolve(),
        (*record.keys(), "build_identity", "runtime_identity"),
    ).to_dict()
    environment = capture_environment()
    record["runtime_identity"] = {
        "fingerprint": environment.fingerprint,
        "backend": environment.backend,
        "x64_enabled": environment.x64_enabled,
        "platform": environment.platform,
        "machine": environment.machine,
        "devices": [device.to_dict() for device in environment.devices],
        "package_fingerprint": environment.package_fingerprint,
    }
    return record


class NativeTransportFieldTemplate(TypedDict):
    name: str
    space_family: str
    degree: int
    continuity: str
    component_shape: tuple[int, ...]
    role: str


def native_transport_field_templates() -> tuple[NativeTransportFieldTemplate, ...]:
    """Declare the bulk, network, and reservoir concentration fields."""
    return (
        {
            "name": "bulk",
            "space_family": "finite-volume",
            "degree": 0,
            "continuity": "cell-local",
            "component_shape": (),
            "role": "conservative-density",
        },
        {
            "name": "network",
            "space_family": "metric-network",
            "degree": 1,
            "continuity": "continuous",
            "component_shape": (),
            "role": "conservative-density",
        },
        {
            "name": "reservoirs",
            "space_family": "reservoir",
            "degree": 0,
            "continuity": "discrete",
            "component_shape": (),
            "role": "conservative-density",
        },
    )


def _run_warm_native_transport(
    *,
    case: str = "two-slab",
    size: int = 2,
    repeats: int = 3,
    steps: int = 5,
    dt: float = 0.01,
    target_error: float = 0.05,
    balance_tolerance: float = 1.0e-9,
    maximum_cells: int = 4096,
    maximum_vertices: int = 4096,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
    oracle_capacity: int = 2048,
    phase: Callable[[str], None],
    context: _NativeTransportContext,
    deadline: Deadline,
    coordinate_budget: CoordinateEnclosureBudget,
    source_workspace: NativeHostStorageWorkspace,
) -> tuple[
    _TransportGeneration,
    _TransportAcceptedExecution,
    phx.applications.neurofluid.NeurofluidTransportCheckpoint,
]:
    """Generate, re-admit, transfer, solve, and independently certify one case."""
    generated = _prepare_transport_generation(
        case=case,
        size=size,
        maximum_cells=maximum_cells,
        maximum_vertices=maximum_vertices,
        maximum_work_units=maximum_work_units,
        maximum_geometry_queries=maximum_geometry_queries,
        size_compliance=NATIVE_IMAGE_SIZE_COMPLIANCE,
        phase=phase,
        context=context,
        deadline=deadline,
        coordinate_budget=coordinate_budget,
        source_workspace=source_workspace,
    )
    transport_case = generated["transport_case"]
    source, network = generated["source"], generated["network"]
    result = generated["result"]
    phase("oracle-capacity-admission")
    count = result.mesh.entity_set(3).count
    if count + 3 > oracle_capacity:
        raise ValueError(
            "transport.oracle_capacity must admit the generated transport state; "
            f"got requested={count + 3}, limit={oracle_capacity}"
        )
    phase("case-readmission")
    started = perf_counter()
    neurofluid_case = phx.applications.neurofluid.NeurofluidCase(
        "native-generated-transport", source.labels, source.compartments, result, network
    )
    readmitted = phx.applications.neurofluid.NeurofluidCase(
        neurofluid_case.case_id,
        neurofluid_case.segmentation,
        neurofluid_case.compartments,
        neurofluid_case.bulk_mesh,
        neurofluid_case.network,
    )
    admission = perf_counter() - started
    phase("transport-transfer-preparation")
    parameters = _transport_parameters(result, transport_case)
    transport_plan = phx.applications.neurofluid.NeurofluidTransportPlan(
        readmitted, parameters
    )
    runtime, preparation = measure_synchronized(transport_plan.prepare)
    evidence = result.region_evidence
    if evidence is None:
        raise RuntimeError(
            "transport.region_evidence must contain compartment assignments; got None"
        )
    initial_values = {
        "bulk": jnp.asarray(
            [
                transport_case.physics[region].initial_concentration
                for region in evidence.cell_region_ids
            ],
            dtype=jnp.float64,
        ),
        "network": jnp.asarray(transport_case.exchange.network_initial, dtype=jnp.float64),
        "reservoirs": jnp.asarray(
            transport_case.exchange.reservoir_initial, dtype=jnp.float64
        ),
    }
    initial = phx.equations.MixedDimensionalTransportState(
        **{
            template["name"]: initial_values[template["name"]]
            for template in native_transport_field_templates()
        }
    )

    def loop(current_runtime: Any, state: Any) -> Any:
        return _transport_loop(current_runtime, state, steps=steps, dt=dt)

    execute = jax.jit(loop)

    def lower() -> Any:
        phase("lowering")
        return execute.lower(runtime, initial)

    def compile_lowered(lowered: Any) -> Any:
        phase("compilation")
        return lowered.compile()

    executable, compilation = measure_lower_and_compile(lower, compile_lowered)
    phase("first-execution")
    first, first_seconds = measure_synchronized(lambda: executable(runtime, initial))
    phase("warmed-execution")
    _, warmed = measure_repeated(
        lambda: executable(runtime, initial), warmup=0, repeats=repeats
    )
    phase("compiler-evidence")
    compiler = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    phase("independent-certification")
    started = perf_counter()
    _, (history, accepted) = first
    invariants, reference = _transport_invariants(
        source,
        result,
        transport_case,
        runtime,
        network,
        parameters,
        initial,
        history,
        accepted,
        steps=steps,
        dt=dt,
        target_error=target_error,
        balance_tolerance=balance_tolerance,
    )
    certification = perf_counter() - started
    time_convergence = _time_convergence(
        runtime,
        initial,
        reference,
        invariants["physical_error"],
        steps=steps,
        dt=dt,
        phase=phase,
    )
    lifecycle, checkpoint = _transport_lifecycle(
        source,
        result,
        readmitted,
        network,
        parameters,
        runtime,
        initial,
        first,
        executable,
        loop,
        reference,
        coordinate_budget=coordinate_budget,
        size=transport_case.size,
        steps=steps,
        dt=dt,
        target_error=target_error,
        balance_tolerance=balance_tolerance,
        phase=phase,
    )
    executed: _TransportAcceptedExecution = {
        "case": neurofluid_case,
        "readmitted": readmitted,
        "transport_plan": transport_plan,
        "runtime": runtime,
        "evidence": evidence,
        "initial": initial,
        "first": first,
        "admission": admission,
        "preparation": preparation,
        "compilation": compilation,
        "first_seconds": first_seconds,
        "warmed": warmed,
        "compiler": compiler,
        "certification": certification,
        "invariants": invariants,
        "lifecycle": lifecycle,
        "time_convergence": time_convergence,
    }
    return generated, executed, checkpoint


def _execute_native_transport(
    *,
    case: str,
    size: int,
    repeats: int,
    steps: int,
    dt: float,
    target_error: float,
    balance_tolerance: float,
    maximum_cells: int,
    maximum_vertices: int,
    oracle_capacity: int,
    phase: Callable[[str], None],
    context: _NativeTransportContext,
    deadline: Deadline,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
) -> NativeTransportRecord:
    """Admit the archive in the warm request and end it before independent cold."""
    campaign_started = perf_counter()
    limits = phx.meshing.MeshingLimits(
        maximum_cells=maximum_cells, maximum_vertices=maximum_vertices,
        **({"maximum_work_units": maximum_work_units} if maximum_work_units is not None else {}),
        **({"maximum_geometry_queries": maximum_geometry_queries} if maximum_geometry_queries is not None else {}),
    )
    with ExitStack() as warm:
        warm.enter_context(NativeParallelExecution(None, deadline.remaining()))
        scope = warm.enter_context(NativeExecutionScope(limits.native_limits()))
        scope.observe_managed_admission()
        request_view = warm.enter_context(scope.ticket_view())
        warm.enter_context(request_view.native_target())
        storage = warm.enter_context(scope.host_workspace())
        initial = scope.batch(0)
        initial.settle()
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        warm.enter_context(budget.bind_ticket(scope.ticket_view()))
        warm.enter_context(meshing_source_owner_reconstruction_scope(budget=budget))
        generated, executed, checkpoint = _run_warm_native_transport(
            case=case, size=size, repeats=repeats, steps=steps, dt=dt,
            target_error=target_error, balance_tolerance=balance_tolerance,
            maximum_cells=maximum_cells, maximum_vertices=maximum_vertices,
            maximum_work_units=maximum_work_units,
            maximum_geometry_queries=maximum_geometry_queries,
            oracle_capacity=oracle_capacity, phase=phase, context=context,
            deadline=deadline, coordinate_budget=budget, source_workspace=storage,
        )
        storage.retain_owner((generated["source"], generated["result"], checkpoint))
        archived_result = generated["result"]
        certification, publication_receipt = (
            archived_result.certification, archived_result.execution_evidence
        )
        if certification is None or publication_receipt is None:
            raise RuntimeError(
                "transport.archive must retain its ended publication receipt and source certification."
            )
        records = MeshingSourceClosure.from_records(
            {
                "certification_inputs": certification.request,
                "report": certification,
                "associations": archived_result.associations,
                "generation_part": phx.meshing.MeshPart(
                    "image", archived_result,
                    coordinate_contract=archived_result.coordinate_contract,
                ),
                "generation_source": generated["source"],
                "generation_specification": generated["request"],
                "generation_options": generated["options"],
                "accepted_data": {"neurofluid_transport": checkpoint},
            }
        )

        def end_warm_request(content_id: str) -> None:
            # The archive's ended mesh publication is an honest prefix, not a
            # snapshot of this still-running request. Close every paid owner
            # before constructing the actual complete receipt or starting cold.
            stage_native_source_publication(scope, archived_result, budget=budget)
            issue = prepare_native_completed_source_issue(
                scope, archived_result, owner_id=publication_receipt.owner_id,
                budget=budget,
            )
            warm.close()
            result = bind_native_execution_result(
                archived_result, scope.completed_evidence(), limits, scope.operation_started
            )
            issue.complete(result)
            register_meshing_source_artifacts()
            _require_execution_endpoint_extension(archived_result, result)
            execution = result.execution_evidence
            if execution is None or execution.owner_id is None:
                raise RuntimeError("transport.warm_request must retain its owner-bound ended receipt.")
            generated["result"] = result
            generated["native_execution"] = execution.to_record()
            context["native_execution"] = generated["native_execution"]
            binding = {
                "kind": "native-transport-archive-execution-binding",
                "archive_content_id": content_id,
                "source_id": generated["source"].source_id,
                "source_revision": generated["source"].source_revision,
                "result_id": result.result_id,
                "request_owner_id": execution.owner_id,
                "specification_id": generated["request"].specification_id,
                "limits_id": limits.limits_id,
                "full_request_receipt_id": execution.receipt_hex(),
                "archived_prefix_receipt_id": publication_receipt.receipt_hex(),
                "archived_receipt_scope": "ended-mesh-publication-prefix",
                "full_receipt_scope": "complete-ended-warm-native-transport-request",
            }
            binding["binding_id"] = phx.canonical_fingerprint(binding)
            context["warm_execution_binding"] = binding

        executed["lifecycle"]["cold"] = _publish_transport_checkpoint(
            records, phase=phase, context=context, deadline=deadline,
            end_warm_request=end_warm_request,
        )
        executed["lifecycle"]["cold"]["archive_execution_scope"] = (
            "PREFIX: actual ended mesh publication; complete ended warm request "
            "is retained in workflow native_execution and warm_execution_binding"
        )
    if "warm_execution_binding" not in context:
        raise RuntimeError("transport.archive must bind its complete ended warm request.")
    return _report_native_transport(
        generated, executed, campaign_started=campaign_started, size=size,
        repeats=repeats, steps=steps, dt=dt, target_error=target_error,
        balance_tolerance=balance_tolerance, maximum_cells=maximum_cells,
        maximum_vertices=maximum_vertices, oracle_capacity=oracle_capacity, phase=phase,
        warm_execution_binding=context["warm_execution_binding"],
    )


def native_transport(
    *,
    case: str = "two-slab",
    size: int = 2,
    repeats: int = 3,
    steps: int = 5,
    dt: float = 0.01,
    target_error: float = 0.05,
    balance_tolerance: float = 1.0e-9,
    maximum_cells: int = 4096,
    maximum_vertices: int = 4096,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
    oracle_capacity: int = 2048,
    phase_callback: Callable[[str], None] | None = None,
    timeout: float = 120.0,
) -> NativeTransportRecord:
    """Preserve failed native attempts and their failure phases without fallback."""
    started = perf_counter()
    deadline = Deadline(started + timeout)
    context: _NativeTransportContext = {"case": case}
    if case in NATIVE_TRANSPORT_CASE_ROLES:
        context["qualification_role"] = NATIVE_TRANSPORT_CASE_ROLES[case]
    current_phase = "source-preparation"

    def phase(name: str) -> None:
        nonlocal current_phase
        current_phase = name
        deadline.remaining()
        if phase_callback is not None:
            phase_callback(name)

    try:
        phase(current_phase)
        record = _execute_native_transport(
            case=case,
            size=size,
            repeats=repeats,
            steps=steps,
            dt=dt,
            target_error=target_error,
            balance_tolerance=balance_tolerance,
            maximum_cells=maximum_cells,
            maximum_vertices=maximum_vertices,
            maximum_work_units=maximum_work_units,
            maximum_geometry_queries=maximum_geometry_queries,
            oracle_capacity=oracle_capacity,
            phase=phase,
            context=context,
            deadline=deadline,
        )
        phase("workflow-complete")
        record["timing"]["end_to_end_seconds"] = perf_counter() - started
        return record
    except Exception as error:
        failure: dict[str, object] = {
            "phase": current_phase,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(error),
        }
        if isinstance(error, phx.meshing.MeshingFailure):
            evidence = error.evidence
            failure["meshing_evidence"] = {
                "category": evidence.category.value,
                "message": evidence.message,
                "provider_code": evidence.provider_code,
                "stage": evidence.stage,
                "entity_ids": evidence.entity_ids,
                "locations": evidence.locations,
                "requested": evidence.requested,
                "achieved": evidence.achieved,
                "checkpoint_id": evidence.checkpoint_id,
                "evidence_id": evidence.evidence_id,
            }
        return {
            "scenario": "native-generated-transport",
            "successful": False,
            "status": "failed",
            "failure": failure,
            **context,
            "controls": {
                "size": size,
                "repeats": repeats,
                "steps": steps,
                "dt_seconds": dt,
                "maximum_cells": maximum_cells,
                "maximum_vertices": maximum_vertices,
                "oracle_capacity": oracle_capacity,
            },
            "targets": {
                "physical_error": target_error,
                "balance_tolerance": balance_tolerance,
            },
            "timing": {"elapsed_before_failure_seconds": perf_counter() - started},
            "comparison": {"status": "not-requested"},
        }


def add_native_arguments(
    parser: argparse.ArgumentParser, *, default_scenario: str
) -> None:
    parser.add_argument(
        "--scenario",
        choices=("synthetic", "native-generated-transport"),
        default=default_scenario,
    )
    parser.add_argument(
        "--case",
        choices=tuple(NATIVE_TRANSPORT_CASE_ROLES),
        default="two-slab",
        help="Declared labeled-image transport case.",
    )
    parser.add_argument(
        "--size", type=int, default=2, help="Occupied-voxel image edge count."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Transport step width in seconds."
    )
    parser.add_argument(
        "--target-error",
        type=float,
        default=0.05,
        help="Relative mass-weighted error against independent continuous-time transport.",
    )
    parser.add_argument("--balance-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--maximum-cells", type=int, default=4096)
    parser.add_argument("--maximum-vertices", type=int, default=4096)
    parser.add_argument(
        "--oracle-capacity",
        type=int,
        default=2048,
        help="Maximum state width for the independent dense scientific oracle.",
    )


def validate_native_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if (
        args.size < 2
        or min(
            args.repeats,
            args.steps,
            args.maximum_cells,
            args.maximum_vertices,
            args.oracle_capacity,
        )
        < 1
    ):
        parser.error(
            "size must exceed one; repeats, steps, and capacities must be positive"
        )
    if args.case == "oblique-junction-transport" and args.size < 4:
        parser.error("oblique-junction-transport size must be at least four")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (args.dt, args.target_error, args.balance_tolerance)
    ):
        parser.error(
            "dt, target-error, and balance-tolerance must be finite and positive"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_arguments(parser, default_scenario="synthetic")
    args = parser.parse_args()
    validate_native_arguments(parser, args)
    if args.scenario == "synthetic":
        record = qualify()
    else:
        record = native_transport(
            case=args.case,
            size=args.size,
            repeats=args.repeats,
            steps=args.steps,
            dt=args.dt,
            target_error=args.target_error,
            balance_tolerance=args.balance_tolerance,
            maximum_cells=args.maximum_cells,
            maximum_vertices=args.maximum_vertices,
            oracle_capacity=args.oracle_capacity,
        )
    print(json.dumps(record, indent=2))
    if not record["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

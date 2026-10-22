"""White-box tests of native publication resource and audit declarations."""

import dataclasses
from typing import get_args

import numpy as np
import pytest

import phydrax as phx
from phydrax import model_array_bytes
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import MeshcoreError, MeshcoreStatus
from phydrax.meshing import (
    CellMeshAuditDisposition,
    CellMeshAuditPolicy,
    MeshingDerivativeMode,
    MeshingLimits,
)
from phydrax.meshing.providers._native_route_policy import (
    publication_policy,
    PublicationRoute,
)
from tests._support.cell_meshes import triangle_grid
from tests._support.mixed_adaptation import red_shared_face_source


# Corpus gates and limit probes read each declaration as `maximum_<name>`.
_LIMITS = frozenset(
    member.name.removeprefix("maximum_")
    for member in dataclasses.fields(MeshingLimits)
    if member.name.startswith("maximum_")
)
# Implicit publication additionally refuses its grid, surface and projection
# capacities; these are route capabilities, not request limits.
_ROUTE_CAPABILITIES = frozenset(("grid_capacity", "surface_capacity", "projection"))


@pytest.mark.parametrize("route", get_args(PublicationRoute))
def test_published_route_guarantees_account_for_every_request_limit(
    route: PublicationRoute,
) -> None:
    policy = publication_policy(route)
    # Native publication refuses excess retained data on every route.
    enforced = {*policy.enforced_limits, "data_bytes"}
    unenforced = set(policy.unenforced_limits) - {"data_bytes"}
    assert len(set(policy.enforced_limits)) == len(policy.enforced_limits)
    bases = {
        name.partition(":")[0] for name in (enforced | unenforced) - _ROUTE_CAPABILITIES
    }
    assert bases == _LIMITS
    # A bare enforced name is a full guarantee; partial coverage stays qualified.
    assert not {name for name in enforced if ":" not in name} & {
        name.partition(":")[0] for name in unenforced
    }


def _refine_all(
    source: phx.meshing.CellMeshingResult, limits: MeshingLimits, /
) -> phx.meshing.MeshAdaptationResult:
    cells = np.concatenate([np.asarray(block.global_ids) for block in source.mesh.blocks])
    request = phx.meshing.MarkedMeshAdaptation(cells, np.empty((0,), dtype=np.int64))
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION, limits=limits
    )
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(source, request, policy=policy)
    )


def _triangle_source() -> phx.meshing.CellMeshingResult:
    return phx.meshing.certify_cell_mesh(
        triangle_grid(2, 2), phx.SpatialCoordinateContract.si()
    )


def test_bisection_adaptation_publishes_its_proven_resource_guarantees() -> None:
    adaptation = _refine_all(_triangle_source(), MeshingLimits())
    runtime = adaptation.target.runtime
    # An affine carrier queries no source, so every request limit is enforced.
    assert {name.partition(":")[0] for name in runtime.enforced_limits} >= _LIMITS
    assert runtime.unenforced_limits == ()
    # The ended adaptation settlement record reaches both consumer-visible targets.
    receipt = adaptation.target.execution_evidence
    assert receipt is not None
    assert receipt.owner_id == adaptation.prepared_id
    assert adaptation.transition is not None
    assert adaptation.transition.target.result_id == adaptation.target.result_id
    # The transition view is a rebuilt module carrying the same ended record.
    assert tree_content_equal(adaptation.transition.target.execution_evidence, receipt)


@pytest.mark.parametrize("name", ("edges", "faces", "connectivity_entries", "data_bytes"))
def test_bisection_adaptation_refuses_a_target_beyond_its_published_limits(
    name: str,
) -> None:
    source = _triangle_source()
    mesh = source.mesh
    # The source fits each allowance exactly; only the refined target exceeds it.
    allowance = {
        "edges": mesh.entity_set(1).entity_ids.size,
        "faces": mesh.entity_set(2).entity_ids.size,
        "connectivity_entries": sum(block.vertices.size for block in mesh.blocks)
        + sum(
            incidence.relation.source_indices.size
            + incidence.relation.target_indices.size
            for incidence in mesh.topology.incidences
        ),
        "data_bytes": model_array_bytes(source),
    }[name]
    with pytest.raises(phx.meshing.MeshingFailure) as refused:
        _refine_all(source, MeshingLimits(**{f"maximum_{name}": allowance}))
    failure = refused.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    achieved = dict(failure.evidence.achieved)
    measured = "retained_data_bytes" if name == "data_bytes" else name
    assert achieved[measured] > dict(failure.evidence.requested)[f"maximum_{name}"]


def _red_mixed(limits: MeshingLimits, /) -> phx.meshing.MeshAdaptationResult:
    mesh, geometry = red_shared_face_source(False, False)
    source = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED, limits=limits
    )
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((11, 37), dtype=np.int64)),
            policy=policy,
        )
    )


def test_mixed_adaptation_admits_its_template_cavity_under_the_request() -> None:
    limits = {"maximum_cells": 100_000, "maximum_work_units": 20_000_000}
    adaptation = _red_mixed(MeshingLimits(**limits))
    runtime = adaptation.target.runtime
    assert {name.partition(":")[0] for name in runtime.enforced_limits} >= _LIMITS
    assert runtime.unenforced_limits == ()
    # Two marked cells retire at least two source cells in one template edit.
    with pytest.raises(MeshcoreError) as refused:
        _red_mixed(MeshingLimits(**limits, maximum_cavity_cells=1))
    assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize(
    ("route", "watertight", "association"),
    [
        ("dual", CellMeshAuditDisposition.REJECT, True),
        ("grid", CellMeshAuditDisposition.REJECT, True),
        ("polyhedral", CellMeshAuditDisposition.REJECT, True),
        ("structured_surface", CellMeshAuditDisposition.SKIP, False),
        ("structured_volume", CellMeshAuditDisposition.REJECT, False),
        ("layer_core", CellMeshAuditDisposition.REJECT, True),
        ("mapped_layer_core", CellMeshAuditDisposition.REJECT, True),
        ("curve", CellMeshAuditDisposition.SKIP, True),
        ("planar", CellMeshAuditDisposition.REJECT, True),
        ("parametric_surface", CellMeshAuditDisposition.REJECT, True),
        ("source_quads", CellMeshAuditDisposition.REJECT, True),
        ("implicit", CellMeshAuditDisposition.REJECT, True),
        ("volume", CellMeshAuditDisposition.REJECT, True),
        ("compartment", CellMeshAuditDisposition.REJECT, True),
        ("periodic", CellMeshAuditDisposition.SKIP, False),
        ("periodic_stratified", CellMeshAuditDisposition.SKIP, False),
    ],
    ids=(
        "dual",
        "grid",
        "polyhedral",
        "structured-surface",
        "structured-volume",
        "layer-core",
        "mapped-layer-core",
        "curve",
        "planar",
        "parametric-surface",
        "source-quads",
        "implicit",
        "volume",
        "compartment",
        "periodic",
        "periodic-stratified",
    ),
)
def test_route_audit_obligations_are_explicit(
    route: PublicationRoute, watertight: CellMeshAuditDisposition, association: bool
) -> None:
    policy = publication_policy(route)
    assert policy.audit_policy.watertight_boundary is watertight
    assert policy.audit_policy.require_complete_association is association


def test_unknown_publication_route_fails_closed() -> None:
    with pytest.raises(ValueError, match="publication.route"):
        publication_policy("unsupported")  # ty: ignore[invalid-argument-type]


def test_route_audit_override_preserves_limits_and_requested_dispositions() -> None:
    audit = CellMeshAuditPolicy(
        require_complete_association=True,
        nonmanifold=CellMeshAuditDisposition.RECORD,
        watertight_boundary=CellMeshAuditDisposition.SKIP,
    )
    baseline = publication_policy("parametric_surface")
    selected = publication_policy("parametric_surface", audit_policy=audit)
    assert selected.audit_policy is audit
    assert selected.enforced_limits == baseline.enforced_limits
    assert selected.unenforced_limits == baseline.unenforced_limits


@pytest.mark.parametrize(
    "mode",
    [
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        MeshingDerivativeMode.FIXED_ROUTE_PIECEWISE,
    ],
    ids=["nondifferentiable", "fixed-route-piecewise"],
)
def test_derivative_declaration_reaches_publication_policy_unchanged(
    mode: MeshingDerivativeMode,
) -> None:
    policy = publication_policy("implicit", derivative_mode=mode)
    assert policy.derivative_mode is mode


def test_audit_override_rejects_wrong_nominal_kind() -> None:
    with pytest.raises(TypeError, match="audit_policy"):
        publication_policy("dual", audit_policy="not-an-audit")  # ty: ignore[invalid-argument-type]

"""White-box tests of native publication resource and audit declarations."""

import pytest

from phydrax.meshing import (
    CellMeshAuditDisposition,
    CellMeshAuditPolicy,
    MeshingDerivativeMode,
)
from phydrax.meshing.providers._native_route_policy import (
    publication_policy,
    PublicationRoute,
)


@pytest.mark.parametrize(
    ("route", "enforced", "unenforced"),
    [
        (
            "dual",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "wall_seconds",
            ),
            (
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
                "data_bytes",
            ),
        ),
        (
            "grid",
            ("vertices", "cells", "connectivity_entries", "wall_seconds"),
            (
                "edges",
                "faces",
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
                "data_bytes",
            ),
        ),
        (
            "polyhedral",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "work_units",
                "wall_seconds",
            ),
            ("cavity_cells", "geometry_queries", "scratch_bytes", "data_bytes"),
        ),
        (
            "structured_volume",
            (
                "maximum_vertices",
                "maximum_cells",
                "maximum_edges",
                "maximum_faces",
                "maximum_connectivity_entries",
                "maximum_data_bytes",
                "maximum_wall_seconds",
            ),
            (
                "maximum_work_units",
                "maximum_cavity_cells",
                "maximum_geometry_queries",
                "maximum_scratch_bytes",
            ),
        ),
        (
            "structured_surface",
            (
                "maximum_vertices",
                "maximum_cells",
                "maximum_edges",
                "maximum_faces",
                "maximum_connectivity_entries",
                "maximum_data_bytes",
                "maximum_wall_seconds",
            ),
            (
                "maximum_work_units",
                "maximum_cavity_cells",
                "maximum_geometry_queries",
                "maximum_scratch_bytes",
            ),
        ),
        (
            "layer_core",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "wall_time",
            ),
            ("native_workspace",),
        ),
        (
            "mapped_layer_core",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "wall_time",
            ),
            ("native_workspace",),
        ),
        (
            "curve",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "work_units",
                "geometry_queries",
                "wall_seconds",
            ),
            ("cavity_cells", "scratch_bytes"),
        ),
        (
            "planar",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "work_units",
                "geometry_queries",
                "wall_seconds",
            ),
            ("cavity_cells",),
        ),
        (
            "parametric_surface",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "work_units",
                "geometry_queries",
                "wall_seconds",
                "cavity_cells",
                "scratch_bytes",
            ),
            (),
        ),
        (
            "source_quads",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "wall_seconds",
            ),
            ("work_units", "geometry_queries", "scratch_bytes", "cavity_cells"),
        ),
        (
            "implicit",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "geometry_queries",
                "scratch_bytes",
                "wall_seconds",
                "grid_capacity",
                "surface_capacity",
                "projection",
            ),
            ("work_units", "cavity_cells"),
        ),
        (
            "compartment",
            (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "work_units:native_and_metered_host",
                "wall_seconds:native_and_host_boundaries",
                "cavity_cells:native",
                "geometry_queries:native_and_source_batches",
                "scratch_bytes:managed_native_and_bounded_host",
            ),
            (
                "scratch_bytes:unmanaged_host_device_compiler",
                "work_units:unmetered_host_device_compiler",
                "wall_seconds:nonpreemptible_host_device_compiler",
            ),
        ),
        (
            "periodic",
            (
                "vertices",
                "cells",
                "edges",
                "faces",
                "connectivity_entries",
                "data_bytes",
                "scratch_bytes",
                "work_units",
                "wall_seconds",
                "geometry_queries",
            ),
            ("cavity_cells",),
        ),
        (
            "periodic_stratified",
            (
                "vertices",
                "cells",
                "edges",
                "faces",
                "connectivity_entries",
                "data_bytes",
                "scratch_bytes",
                "work_units",
                "wall_seconds",
                "cavity_cells",
                "geometry_queries",
            ),
            (),
        ),
    ],
    ids=(
        "dual",
        "grid",
        "polyhedral",
        "structured-volume",
        "structured-surface",
        "layer-core",
        "mapped-layer-core",
        "curve",
        "planar",
        "parametric-surface",
        "source-quads",
        "implicit",
        "compartment",
        "periodic",
        "periodic-stratified",
    ),
)
def test_route_resource_declarations_preserve_order(
    route: PublicationRoute, enforced: tuple[str, ...], unenforced: tuple[str, ...]
) -> None:
    policy = publication_policy(route)
    assert policy.enforced_limits == enforced
    assert policy.unenforced_limits == unenforced
    assert not set(policy.enforced_limits) & set(policy.unenforced_limits)
    assert policy.derivative_mode is MeshingDerivativeMode.NONDIFFERENTIABLE


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

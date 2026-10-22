#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Publication audit and resource declarations for native route families."""

from typing import assert_never, final, Literal

import equinox as eqx

from phydrax._strict import StrictModule
from phydrax.meshing._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from phydrax.meshing._contracts import MeshingDerivativeMode
from phydrax.typing import checked, parse


PublicationRoute = Literal[
    "dual",
    "grid",
    "polyhedral",
    "structured_surface",
    "structured_volume",
    "layer_core",
    "mapped_layer_core",
    "curve",
    "planar",
    "parametric_surface",
    "source_quads",
    "implicit",
    "implicit_adaptive",
    "volume",
    "compartment",
    "periodic",
    "periodic_stratified",
]


@final
class NativePublicationPolicy(StrictModule):
    audit_policy: CellMeshAuditPolicy
    derivative_mode: MeshingDerivativeMode = eqx.field(static=True)
    enforced_limits: tuple[str, ...] = eqx.field(static=True)
    unenforced_limits: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        audit_policy: CellMeshAuditPolicy,
        derivative_mode: MeshingDerivativeMode,
        enforced_limits: tuple[str, ...],
        unenforced_limits: tuple[str, ...],
    ) -> None:
        mode = parse(
            derivative_mode,
            MeshingDerivativeMode,
            "NativePublicationPolicy.derivative_mode",
        )
        self.audit_policy = audit_policy
        self.derivative_mode = mode
        self.enforced_limits = enforced_limits
        self.unenforced_limits = unenforced_limits


# Keep this declarative route table explicit so audit defaults and limit evidence stay paired.
@checked
def publication_policy(
    route: PublicationRoute,
    /,
    *,
    audit_policy: CellMeshAuditPolicy | None = None,
    derivative_mode: MeshingDerivativeMode = MeshingDerivativeMode.NONDIFFERENTIABLE,
) -> NativePublicationPolicy:
    """Declare the audit and resource evidence retained by a native route."""
    route = parse(route, PublicationRoute, "publication.route")
    match route:
        case "dual":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "wall_seconds",
            )
            unenforced = (
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
                "data_bytes",
            )
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "grid":
            enforced = ("vertices", "cells", "connectivity_entries", "wall_seconds")
            unenforced = (
                "edges",
                "faces",
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
                "data_bytes",
            )
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "polyhedral":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
                "data_bytes",
                "wall_seconds",
            )
            # Native allocations and the concrete bounded host owners share
            # one request pool, including source preparation and publication.
            # This is not a claim to cap process RSS or compiler/device memory.
            unenforced = ("scratch_bytes:unmanaged_host_device_compiler",)
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "structured_surface" | "structured_volume":
            # Preflight and publication check entity and data capacities and
            # deadlines. Work, scratch and query tickets cover only edge-arc
            # certification, so those limits are not declared route-wide.
            enforced = (
                "vertices",
                "cells",
                "edges",
                "faces",
                "connectivity_entries",
                "data_bytes",
                "wall_seconds",
            )
            unenforced = (
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
            )
            audit = CellMeshAuditPolicy(
                watertight_boundary=CellMeshAuditDisposition.REJECT
                if route == "structured_volume"
                else CellMeshAuditDisposition.SKIP
            )
        case "layer_core" | "mapped_layer_core":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "wall_seconds",
            )
            # Work and scratch tickets meter source-seam preparation only; no
            # route-wide work, cavity, query or scratch enforcement is claimed.
            unenforced = (
                "work_units",
                "cavity_cells",
                "geometry_queries",
                "scratch_bytes",
            )
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "curve" | "planar" | "parametric_surface":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "work_units",
                "geometry_queries",
                "wall_seconds",
            )
            match route:
                case "curve" | "planar":
                    # Host coordinate ledgers are bounded; no route-wide scratch
                    # proof covers native construction storage.
                    unenforced = ("cavity_cells", "scratch_bytes")
                case "parametric_surface":
                    enforced += ("cavity_cells", "scratch_bytes")
                    unenforced = ()
                case invalid:
                    assert_never(invalid)
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.SKIP
                if route == "curve"
                else CellMeshAuditDisposition.REJECT,
            )
        case "source_quads":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "wall_seconds",
            )
            unenforced = (
                "work_units",
                "geometry_queries",
                "scratch_bytes",
                "cavity_cells",
            )
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "implicit" | "implicit_adaptive":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "geometry_queries",
                "cavity_cells",
                "scratch_bytes",
                "wall_seconds",
                "grid_capacity",
                "surface_capacity",
                "projection",
            )
            match route:
                case "implicit":
                    enforced += ("work_units",)
                    unenforced = ()
                case "implicit_adaptive":
                    unenforced = ("work_units",)
                case invalid:
                    assert_never(invalid)
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "volume":
            enforced = (
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "scratch_bytes",
                "work_units",
                "geometry_queries",
                "cavity_cells",
                "wall_seconds",
            )
            unenforced = ()
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "compartment":
            enforced = (
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
            )
            unenforced = (
                "scratch_bytes:unmanaged_host_device_compiler",
                "work_units:unmetered_host_device_compiler",
                "wall_seconds:nonpreemptible_host_device_compiler",
            )
            audit = CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=CellMeshAuditDisposition.REJECT,
            )
        case "periodic" | "periodic_stratified":
            enforced = (
                "vertices",
                "cells",
                "edges",
                "faces",
                "connectivity_entries",
                "data_bytes",
                "scratch_bytes",
                "work_units",
                "wall_seconds",
            )
            match route:
                case "periodic":
                    enforced += ("geometry_queries",)
                    unenforced = ("cavity_cells",)
                case "periodic_stratified":
                    enforced += ("cavity_cells", "geometry_queries")
                    unenforced = ()
                case invalid:
                    assert_never(invalid)
            audit = CellMeshAuditPolicy()
        case invalid:
            assert_never(invalid)
    return NativePublicationPolicy(
        audit if audit_policy is None else audit_policy,
        derivative_mode,
        enforced,
        unenforced,
    )

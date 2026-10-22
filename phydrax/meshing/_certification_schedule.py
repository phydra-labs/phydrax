#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static route selectors and mandatory certification schedules."""

from __future__ import annotations

from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse


MeshCertificationCheck: TypeAlias = Literal[
    "cell_validity",
    "topology",
    "quality",
    "global_embedding",
    "domain_coverage",
    "source_fidelity",
]


MeshCertificationRoute: TypeAlias = Literal[
    "volume_plc",
    "volume_implicit",
    "volume_layer",
    "volume_cad",
    "mapped_volume",
    "surface",
    "curve",
    "periodic",
]


def route_checks(
    route: MeshCertificationRoute, /
) -> tuple[tuple[MeshCertificationCheck, ...], tuple[str, ...]]:
    """Required certificates and audit checks that may not be skipped."""

    common: tuple[MeshCertificationCheck, ...] = (
        "cell_validity",
        "topology",
        "quality",
        "global_embedding",
    )
    welded = ("coincident_vertices", "collapsed_cells", "duplicate_cells")
    manifold = (
        "nonmanifold_facets",
        "nonmanifold_edges",
        "nonmanifold_vertices",
        "inconsistent_orientation",
        "self_intersection",
    )
    match route:
        case "volume_plc":
            return (*common, "domain_coverage"), (*welded, *manifold, "open_boundary")
        case "volume_implicit" | "volume_layer" | "volume_cad" | "mapped_volume":
            return (
                (*common, "domain_coverage", "source_fidelity"),
                (*welded, *manifold, "open_boundary"),
            )
        case "surface":
            # Surfaces may be intentionally open; closure is not applicable.
            return (*common, "source_fidelity"), (*welded, *manifold)
        case "curve":
            return common, (*welded, "inconsistent_orientation")
        case "periodic":
            return common, (*welded, *manifold)
        case _:
            assert_never(route)


@final
class MeshCertificationSchedule(StrictModule, NonTrainableState):
    """Checks a route must certify and audit checks it may not skip."""

    route: MeshCertificationRoute = eqx.field(static=True)
    required_checks: tuple[MeshCertificationCheck, ...] = eqx.field(static=True)
    required_audit_checks: tuple[str, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, route: MeshCertificationRoute, /) -> None:
        route_ = parse(route, MeshCertificationRoute, "route")
        required, audit_checks = route_checks(route_)
        self.route = route_
        self.required_checks = required
        self.required_audit_checks = audit_checks
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mesh-certification-schedule",
                "route": route_,
                "required": required,
                "audit_checks": audit_checks,
            }
        )

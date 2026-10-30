#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Closed scientific route selectors and mandatory certification checks."""

from __future__ import annotations

from typing import assert_never, Literal

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse


type MeshCertificationCheck = Literal[
    "cell_validity",
    "topology",
    "quality",
    "global_embedding",
    "domain_coverage",
    "source_fidelity",
    "protected_features",
    "junctions",
]
type MeshCertificationRoute = Literal[
    "volume_plc",
    "volume_implicit",
    "volume_layer",
    "volume_cad",
    "mapped_volume",
    "surface",
    "curve",
    "periodic",
    "planar",
]


_COMMON_CHECKS: tuple[MeshCertificationCheck, ...] = (
    "cell_validity",
    "topology",
    "quality",
    "global_embedding",
)
_VOLUME_AUDIT = (
    "coincident_vertices",
    "collapsed_cells",
    "duplicate_cells",
    "nonmanifold_facets",
    "nonmanifold_edges",
    "nonmanifold_vertices",
    "inconsistent_orientation",
    "self_intersection",
    "open_boundary",
)


def route_checks(
    route: MeshCertificationRoute, /
) -> tuple[tuple[MeshCertificationCheck, ...], tuple[str, ...]]:
    """Return mandatory proof obligations and non-skippable audit checks."""
    match route:
        case "volume_plc":
            return (*_COMMON_CHECKS, "domain_coverage"), _VOLUME_AUDIT
        case "volume_implicit" | "volume_layer" | "volume_cad" | "mapped_volume":
            return (*_COMMON_CHECKS, "domain_coverage", "source_fidelity"), _VOLUME_AUDIT
        case "surface":
            return (*_COMMON_CHECKS, "source_fidelity"), _VOLUME_AUDIT[:-1]
        case "curve":
            return _COMMON_CHECKS, ("coincident_vertices", "collapsed_cells", "inconsistent_orientation")
        case "periodic":
            return _COMMON_CHECKS, _VOLUME_AUDIT[:-1]
        case "planar":
            return (
                *_COMMON_CHECKS,
                "domain_coverage",
                "protected_features",
                "junctions",
            ), _VOLUME_AUDIT[:-1]
        case _:
            assert_never(route)


class MeshCertificationSchedule(StrictModule, NonTrainableState):
    """Mathematical checks one native route must certify."""

    route: MeshCertificationRoute
    required_checks: tuple[MeshCertificationCheck, ...]
    required_audit_checks: tuple[str, ...]
    schedule_id: str

    def __init__(self, route: MeshCertificationRoute, /) -> None:
        route_ = parse(route, MeshCertificationRoute, "route")
        checks, audit = route_checks(route_)
        self.route = route_
        self.required_checks = checks
        self.required_audit_checks = audit
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mesh-certification-schedule",
                "route": route_,
                "required": checks,
                "audit_checks": audit,
            }
        )


__all__ = [
    "MeshCertificationCheck",
    "MeshCertificationRoute",
    "MeshCertificationSchedule",
    "route_checks",
]

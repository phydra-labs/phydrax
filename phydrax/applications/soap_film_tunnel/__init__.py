#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quasi-2D gravity-driven soap-film tunnel on the surface plug-flow film route."""

from ._geometry import SoapFilmTunnelGeometry, SoapFilmTunnelMesh
from ._profiles import soap_film_tunnel_candidate_profiles
from ._tunnel import (
    CylinderWakeReference,
    PreparedSoapFilmTunnel,
    SheddingStatus,
    SoapFilmInflow,
    SoapFilmTunnelEvidence,
    SoapFilmTunnelPlan,
    SoapFilmTunnelResult,
    SoapFilmTunnelScales,
    SoapFilmTunnelState,
    SoapFilmTunnelStepResult,
    SoapFilmWireKind,
    StrouhalEstimate,
)


__all__ = [
    "CylinderWakeReference",
    "PreparedSoapFilmTunnel",
    "SheddingStatus",
    "SoapFilmInflow",
    "SoapFilmTunnelEvidence",
    "SoapFilmTunnelGeometry",
    "SoapFilmTunnelMesh",
    "SoapFilmTunnelPlan",
    "SoapFilmTunnelResult",
    "SoapFilmTunnelScales",
    "SoapFilmTunnelState",
    "SoapFilmTunnelStepResult",
    "SoapFilmWireKind",
    "StrouhalEstimate",
    "soap_film_tunnel_candidate_profiles",
]

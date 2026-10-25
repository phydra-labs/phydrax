"""Ephemeral preparation mappings passed through one BRep request."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .._atlas import TrimArcRefinements, TrimSourceArcs
    from ._intersection import (
        BranchRootEndpoint,
        CurveSurfaceIntersectionRoot,
        Prepared,
        TrimIntersectionRoot,
        TrimRootEndpoint,
        TripleSurfaceIntersectionRoot,
    )
    from ._intersection_curve import (
        IntersectionCurve,
        IntersectionJetPreparation,
        PreparedChartAtlas,
    )
    from ._query_types import FaceCertificate


@dataclass(slots=True)
class TrimRequestPreparations:
    """Retain programs and source identities for an explicitly owned request."""

    trim_roots: dict[
        int, tuple[TrimIntersectionRoot, Prepared | None, dict[int, np.ndarray]]
    ] = field(default_factory=dict)
    curve_surface_roots: dict[
        int,
        tuple[CurveSurfaceIntersectionRoot, Prepared | None, dict[int, np.ndarray]],
    ] = field(default_factory=dict)
    triple_roots: dict[
        int,
        tuple[TripleSurfaceIntersectionRoot, Prepared | None, dict[int, np.ndarray]],
    ] = field(default_factory=dict)
    root_endpoints: dict[
        int,
        tuple[
            TrimRootEndpoint | BranchRootEndpoint,
            dict[int, tuple[float, float]],
        ],
    ] = field(default_factory=dict)
    branch_objects: dict[int, tuple[IntersectionCurve, IntersectionJetPreparation]] = (
        field(default_factory=dict)
    )
    branch_identities: dict[tuple[str, str], IntersectionJetPreparation] = field(
        default_factory=dict
    )
    pair_programs: dict[tuple[str, int], Prepared] = field(default_factory=dict)
    source_programs: dict[tuple[str, str, int, int], Prepared] = field(
        default_factory=dict
    )
    interval_certificates: OrderedDict[str, tuple[FaceCertificate, ...]] = field(
        default_factory=OrderedDict
    )
    trim_arc_refinements: dict[tuple[int, int], TrimArcRefinements] = field(
        default_factory=dict
    )
    trim_source_arcs: dict[tuple[object, ...], TrimSourceArcs] = field(
        default_factory=dict
    )
    # Concrete chart rows per retained intersection curve (``id`` of the curve
    # the record strongly references), reused only while that curve's bank
    # objects and immutable layouts are unchanged.
    chart_atlases: dict[int, PreparedChartAtlas] = field(default_factory=dict)

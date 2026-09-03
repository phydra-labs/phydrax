#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx

from phydrax.enforcement._geometry_support import (
    BoundaryCover,
    BoundaryPatch,
    BoundarySide,
)
from phydrax.linalg import AbstractLinearOperator

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._trace_extension import (
    DiscreteTraceCorrectionProvider,
    PreparedTraceExtension,
)


InterfaceGauge = Literal["minimum_energy", "minus_only", "plus_only"]


class OrientedInterfaceSupport(StrictModule, NonTrainableState):
    """One physical interface with an authoritative minus-to-plus orientation."""

    cover: BoundaryCover
    minus_patch: BoundaryPatch
    plus_patch: BoundaryPatch
    minus_region: str = eqx.field(static=True)
    plus_region: str = eqx.field(static=True)
    common_trace_space_id: str = eqx.field(static=True)
    orientation_certificate_id: str = eqx.field(static=True)
    owner_certificate_id: str = eqx.field(static=True)
    normal_opposition_tolerance: float = eqx.field(static=True)
    normal_opposition_error: float = eqx.field(static=True)
    stability_owner_ids: tuple[str, ...] = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        cover: BoundaryCover,
        /,
        *,
        minus_region: str,
        plus_region: str,
        common_trace_space_id: str,
        orientation_certificate_id: str,
        owner_certificate_id: str,
        normal_opposition_error: float,
        normal_opposition_tolerance: float,
        stability_owner_ids: tuple[str, ...] = (),
    ):
        if not isinstance(cover, BoundaryCover) or len(cover.patches) != 2:
            raise ValueError(
                "An oriented interface cover must contain exactly two patches."
            )
        minus, plus = cover.patches
        if minus.side is not BoundarySide.MINUS or plus.side is not BoundarySide.PLUS:
            raise ValueError("Interface cover patches must be ordered minus, plus.")
        minus_name = str(minus_region)
        plus_name = str(plus_region)
        if not minus_name or not plus_name or minus_name == plus_name:
            raise ValueError("Interface regions must be distinct non-empty names.")
        trace_space = str(common_trace_space_id)
        orientation_id = str(orientation_certificate_id)
        owner_id = str(owner_certificate_id)
        if not trace_space or not orientation_id or not owner_id:
            raise ValueError("Interface trace, orientation, and owner IDs are required.")
        error = float(normal_opposition_error)
        tolerance = float(normal_opposition_tolerance)
        if (
            not math.isfinite(error)
            or not math.isfinite(tolerance)
            or error < 0.0
            or tolerance < 0.0
            or error > tolerance
        ):
            raise ValueError(
                "Interface normals do not satisfy the declared opposition bound."
            )
        if not any(set(junction.patch_indices) == {0, 1} for junction in cover.junctions):
            raise ValueError("The two interface patches require exact junction evidence.")
        stability = tuple(str(value) for value in stability_owner_ids)
        if any(not value for value in stability):
            raise ValueError("stability_owner_ids must contain non-empty IDs.")
        self.cover = cover
        self.minus_patch = minus
        self.plus_patch = plus
        self.minus_region = minus_name
        self.plus_region = plus_name
        self.common_trace_space_id = trace_space
        self.orientation_certificate_id = orientation_id
        self.owner_certificate_id = owner_id
        self.normal_opposition_error = error
        self.normal_opposition_tolerance = tolerance
        self.stability_owner_ids = stability
        self.interface_id = canonical_fingerprint(
            {
                "kind": "oriented-interface-support-v1",
                "cover": cover.cover_id,
                "minus_region": minus_name,
                "plus_region": plus_name,
                "trace_space": trace_space,
                "orientation": orientation_id,
                "owner": owner_id,
                "stability_owners": stability,
            }
        )


class TwoSidedInterfaceCorrectionProvider(StrictModule, NonTrainableState):
    """Joint plus/minus right inverse for value and flux interface rows.

    The candidate and trace operators both act on the complete product field.
    Consequently no side is a sequential pivot, and a common flux correction can
    be pulled back equal-and-opposite by the representation that owns it.
    """

    trace_operator: AbstractLinearOperator
    candidate_operator: AbstractLinearOperator
    support: OrientedInterfaceSupport
    preservation_operator: AbstractLinearOperator | None
    gauge: InterfaceGauge = eqx.field(static=True)
    gauge_certificate_id: str | None = eqx.field(static=True)
    construction_certificate_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        trace_operator: AbstractLinearOperator,
        candidate_operator: AbstractLinearOperator,
        support: OrientedInterfaceSupport,
        /,
        *,
        construction_certificate_id: str,
        gauge: InterfaceGauge = "minimum_energy",
        gauge_certificate_id: str | None = None,
        preservation_operator: AbstractLinearOperator | None = None,
    ):
        if not isinstance(trace_operator, AbstractLinearOperator) or not isinstance(
            candidate_operator, AbstractLinearOperator
        ):
            raise TypeError(
                "Interface trace and candidates must be native linear operators."
            )
        if not isinstance(support, OrientedInterfaceSupport):
            raise TypeError("support must be OrientedInterfaceSupport.")
        if not trace_operator.source.compatible(candidate_operator.target):
            raise ValueError(
                "Interface candidates must map into the complete product field space."
            )
        if trace_operator.target.space_id != support.common_trace_space_id:
            raise ValueError(
                "Interface trace target does not match the declared common trace space."
            )
        gauge_ = str(gauge)
        if gauge_ not in ("minimum_energy", "minus_only", "plus_only"):
            raise ValueError("Unknown two-sided interface correction gauge.")
        gauge_id = None if gauge_certificate_id is None else str(gauge_certificate_id)
        if gauge_ != "minimum_energy" and (gauge_id is None or not gauge_id):
            raise ValueError(
                "One-sided interface gauges require an explicit certificate."
            )
        construction = str(construction_certificate_id)
        if not construction:
            raise ValueError("construction_certificate_id must be non-empty.")
        if preservation_operator is not None:
            if not isinstance(preservation_operator, AbstractLinearOperator):
                raise TypeError("preservation_operator must be linear or None.")
            if not preservation_operator.source.compatible(trace_operator.source):
                raise ValueError(
                    "Interface preservation operator acts on the wrong space."
                )
        self.trace_operator = trace_operator
        self.candidate_operator = candidate_operator
        self.support = support
        self.preservation_operator = preservation_operator
        self.gauge = gauge_  # type: ignore[assignment]
        self.gauge_certificate_id = gauge_id
        self.construction_certificate_id = construction
        self.provider_id = canonical_fingerprint(
            {
                "kind": "two-sided-interface-correction-v1",
                "trace": trace_operator.operator_id,
                "candidate": candidate_operator.operator_id,
                "support": support.interface_id,
                "construction": construction,
                "gauge": gauge_,
                "gauge_certificate": gauge_id,
                "preservation": (
                    None
                    if preservation_operator is None
                    else preservation_operator.operator_id
                ),
            }
        )

    def prepare(
        self, /, *, rank=None, resources=None, numeric_version=0
    ) -> PreparedTraceExtension:
        preservation_id = (
            None
            if self.preservation_operator is None
            else canonical_fingerprint(
                {
                    "kind": "interface-preservation-v1",
                    "operator": self.preservation_operator.operator_id,
                    "interface": self.support.interface_id,
                    "orientation": self.support.orientation_certificate_id,
                    "owner": self.support.owner_certificate_id,
                }
            )
        )
        construction_id = canonical_fingerprint(
            {
                "kind": "two-sided-interface-system-v1",
                "provider": self.provider_id,
                "construction": self.construction_certificate_id,
                "interface": self.support.interface_id,
                "gauge": self.gauge,
                "gauge_certificate": self.gauge_certificate_id,
            }
        )
        provider = DiscreteTraceCorrectionProvider(
            self.trace_operator,
            self.candidate_operator,
            self.support.cover,
            representation_id=f"interface/{self.support.interface_id}",
            representation_certificate_id=construction_id,
            preservation_operator=self.preservation_operator,
            preservation_certificate_id=preservation_id,
            stability_owner_ids=self.support.stability_owner_ids,
        )
        return provider.prepare(
            rank=rank,
            resources=resources,
            numeric_version=numeric_version,
        )


__all__ = [
    "InterfaceGauge",
    "OrientedInterfaceSupport",
    "TwoSidedInterfaceCorrectionProvider",
]

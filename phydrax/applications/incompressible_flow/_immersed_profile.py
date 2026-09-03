#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification._evidence import QualificationMatrix
from ...qualification._registry import CapabilityProfile, SupportTuple
from ._immersed_support import (
    DEFORMABLE_CONTACT_SUPPORT_TUPLE,
    FIXED_TOPOLOGY_SHARP_SUPPORT_TUPLE,
    FREE_RIGID_MARKER_SUPPORT_TUPLE,
    LBM_BODY_SUPPORT_TUPLE,
    PRESCRIBED_MARKER_SUPPORT_TUPLE,
    RESOLVED_CFD_DEM_SUPPORT_TUPLE,
)


IMMERSED_REFERENCE_CASES = (
    "manufactured-loads",
    "fixed-cylinder",
    "moving-cylinder",
    "fixed-sphere",
    "moving-sphere",
    "added-mass",
    "free-settling",
    "flexible-contact-state",
    "sharp-certificate",
)


class ImmersedDNSQualificationProfile(StrictModule, NonTrainableState):
    """Unsigned candidate envelope for current immersed DNS owner routes."""

    prescribed_marker: SupportTuple
    free_rigid_marker: SupportTuple
    fixed_topology_sharp: SupportTuple
    deformable_contact: SupportTuple
    lbm_body: SupportTuple
    resolved_cfd_dem: SupportTuple
    capability_profile: CapabilityProfile
    qualification_matrix: QualificationMatrix
    load_tolerance: float = eqx.field(static=True)
    reference_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    sharp_measure_tolerance: float = eqx.field(static=True)
    marker_condition_limit: float = eqx.field(static=True)
    required_reference_cases: tuple[str, ...] = eqx.field(static=True)
    released: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        load_tolerance: float = 1.0e-8,
        reference_tolerance: float = 5.0e-2,
        conservation_tolerance: float = 1.0e-8,
        sharp_measure_tolerance: float = 1.0e-8,
        marker_condition_limit: float = 1.0e10,
    ):
        tolerances = np.asarray(
            (
                load_tolerance,
                reference_tolerance,
                conservation_tolerance,
                sharp_measure_tolerance,
                marker_condition_limit,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(tolerances)) or np.any(tolerances <= 0.0):
            raise ValueError("Immersed DNS qualification tolerances must be positive.")
        if float(marker_condition_limit) <= 1.0:
            raise ValueError("marker_condition_limit must be greater than one.")
        supports = (
            PRESCRIBED_MARKER_SUPPORT_TUPLE,
            FREE_RIGID_MARKER_SUPPORT_TUPLE,
            FIXED_TOPOLOGY_SHARP_SUPPORT_TUPLE,
            DEFORMABLE_CONTACT_SUPPORT_TUPLE,
            LBM_BODY_SUPPORT_TUPLE,
            RESOLVED_CFD_DEM_SUPPORT_TUPLE,
        )
        capability = CapabilityProfile(
            "immersed-dns.candidate",
            "phydrax",
            "candidate",
            supports,
            released=False,
        )
        matrix = QualificationMatrix(
            {
                f"reference.{case}": {
                    "evidence_kind": "reference",
                    "criterion_id": case,
                }
                for case in IMMERSED_REFERENCE_CASES
            }
        )
        self.prescribed_marker = supports[0]
        self.free_rigid_marker = supports[1]
        self.fixed_topology_sharp = supports[2]
        self.deformable_contact = supports[3]
        self.lbm_body = supports[4]
        self.resolved_cfd_dem = supports[5]
        self.capability_profile = capability
        self.qualification_matrix = matrix
        self.load_tolerance = float(load_tolerance)
        self.reference_tolerance = float(reference_tolerance)
        self.conservation_tolerance = float(conservation_tolerance)
        self.sharp_measure_tolerance = float(sharp_measure_tolerance)
        self.marker_condition_limit = float(marker_condition_limit)
        self.required_reference_cases = IMMERSED_REFERENCE_CASES
        self.released = False
        self.profile_id = canonical_fingerprint(
            {
                "kind": "immersed-dns-qualification-profile",
                "capability_profile": capability.profile_id,
                "qualification_matrix": matrix.matrix_id,
                "load_tolerance": self.load_tolerance,
                "reference_tolerance": self.reference_tolerance,
                "conservation_tolerance": self.conservation_tolerance,
                "sharp_measure_tolerance": self.sharp_measure_tolerance,
                "marker_condition_limit": self.marker_condition_limit,
                "required_reference_cases": self.required_reference_cases,
                "released": False,
            }
        )

    @property
    def support_tuples(self) -> tuple[SupportTuple, ...]:
        return (
            self.prescribed_marker,
            self.free_rigid_marker,
            self.fixed_topology_sharp,
            self.deformable_contact,
            self.lbm_body,
            self.resolved_cfd_dem,
        )

    def supports(self, support_tuple: SupportTuple, /) -> bool:
        if not isinstance(support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        return any(
            support.support_tuple_id == support_tuple.support_tuple_id
            for support in self.support_tuples
        )

    def tolerance_for(self, case_id: str, /) -> float:
        case = str(case_id)
        if case not in self.required_reference_cases:
            raise KeyError(f"Unknown immersed reference case {case!r}.")
        if case == "manufactured-loads":
            return self.load_tolerance
        if case == "flexible-contact-state":
            return self.conservation_tolerance
        if case == "sharp-certificate":
            return self.sharp_measure_tolerance
        return self.reference_tolerance


__all__ = ["IMMERSED_REFERENCE_CASES", "ImmersedDNSQualificationProfile"]

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._local_variational import AbstractPreparedLocalDiscretization
from ...discretization.iga._certificate import LocalGeometryCertificate
from ._iga_solids import _identifier, _local_certificate, _prepared
from ._rod_dynamics import RodPlan


class IGARodPlan(StrictModule, NonTrainableState):
    """Certified spline-centerline assignment over an existing rod mechanics plan."""

    prepared: AbstractPreparedLocalDiscretization
    rod: RodPlan
    reference_certificate: LocalGeometryCertificate
    displacement_field: str = eqx.field(static=True)
    rotation_field: str | None = eqx.field(static=True)
    reference_certificate_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(
        static=True, default="IGA.Structures.Rod.SinglePatch.Untrimmed"
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: AbstractPreparedLocalDiscretization,
        rod: RodPlan,
        reference_certificate: LocalGeometryCertificate,
        /,
        *,
        displacement_field: str = "u",
        rotation_field: str | None = None,
    ):
        prepared_ = _prepared(prepared)
        if not isinstance(rod, RodPlan):
            raise TypeError("rod must be a RodPlan.")
        certificate = _local_certificate(reference_certificate)
        self.prepared = prepared_
        self.rod = rod
        self.reference_certificate = certificate
        self.displacement_field = _identifier(displacement_field, "displacement_field")
        self.rotation_field = (
            None
            if rotation_field is None
            else _identifier(rotation_field, "rotation_field")
        )
        if self.rotation_field == self.displacement_field:
            raise ValueError("rod displacement and rotation fields must differ.")
        self.reference_certificate_id = certificate.certificate_id
        self.profile_id = "IGA.Structures.Rod.SinglePatch.Untrimmed"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-rod-plan",
                "profile": self.profile_id,
                "prepared": prepared_.prepared_id,
                "rod": rod.plan_id,
                "u": self.displacement_field,
                "rotation": self.rotation_field,
                "certificate": certificate.certificate_id,
            }
        )


__all__ = ["IGARodPlan"]

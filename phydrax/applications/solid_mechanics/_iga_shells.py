# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._local_variational import AbstractPreparedLocalDiscretization
from ...discretization.iga._certificate import SurfaceEmbeddingCertificate
from ._iga_solids import _identifier, _prepared


def _surface(value: SurfaceEmbeddingCertificate, /) -> SurfaceEmbeddingCertificate:
    if not isinstance(value, SurfaceEmbeddingCertificate):
        raise TypeError("surface_certificate must be a SurfaceEmbeddingCertificate.")
    if not bool(value.passed):
        raise ValueError("surface_certificate did not pass; shell lowering fails closed.")
    return value


class IGAKirchhoffLovePlan(StrictModule, NonTrainableState):
    """Certified KL shell field/material assignment; no shell tensor algebra."""

    prepared: AbstractPreparedLocalDiscretization
    surface_certificate: SurfaceEmbeddingCertificate
    displacement_field: str = eqx.field(static=True)
    surface_certificate_id: str = eqx.field(static=True)
    material_plan_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(
        static=True, default="iga.structures.shell.kl.single-patch.untrimmed"
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: AbstractPreparedLocalDiscretization,
        surface_certificate: SurfaceEmbeddingCertificate,
        /,
        *,
        displacement_field: str = "u",
        material_plan_id: str,
    ):
        prepared_ = _prepared(prepared)
        certificate = _surface(surface_certificate)
        self.prepared = prepared_
        self.surface_certificate = certificate
        self.displacement_field = _identifier(displacement_field, "displacement_field")
        self.surface_certificate_id = certificate.certificate_id
        self.material_plan_id = _identifier(material_plan_id, "material_plan_id")
        self.profile_id = "iga.structures.shell.kl.single-patch.untrimmed"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-kirchhoff-love-plan",
                "profile": self.profile_id,
                "prepared": prepared_.prepared_id,
                "field": self.displacement_field,
                "surface_certificate": certificate.certificate_id,
                "material": self.material_plan_id,
            }
        )


class IGAReissnerMindlinPlan(StrictModule, NonTrainableState):
    """Certified tangent-rotation RM shell assignment; shear law stays external."""

    prepared: AbstractPreparedLocalDiscretization
    surface_certificate: SurfaceEmbeddingCertificate
    displacement_field: str = eqx.field(static=True)
    rotation_field: str = eqx.field(static=True)
    tangent_frame_id: str = eqx.field(static=True)
    shear_policy_id: str = eqx.field(static=True)
    surface_certificate_id: str = eqx.field(static=True)
    material_plan_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(
        static=True,
        default="iga.structures.shell.rm.tangential-rotation.single-patch.untrimmed",
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: AbstractPreparedLocalDiscretization,
        surface_certificate: SurfaceEmbeddingCertificate,
        /,
        *,
        displacement_field: str = "u",
        rotation_field: str = "theta",
        tangent_frame_id: str,
        shear_policy_id: str,
        material_plan_id: str,
    ):
        prepared_ = _prepared(prepared)
        certificate = _surface(surface_certificate)
        displacement = _identifier(displacement_field, "displacement_field")
        rotation = _identifier(rotation_field, "rotation_field")
        if displacement == rotation:
            raise ValueError("RM displacement and rotation fields must differ.")
        self.prepared = prepared_
        self.surface_certificate = certificate
        self.displacement_field = displacement
        self.rotation_field = rotation
        self.tangent_frame_id = _identifier(tangent_frame_id, "tangent_frame_id")
        self.shear_policy_id = _identifier(shear_policy_id, "shear_policy_id")
        self.surface_certificate_id = certificate.certificate_id
        self.material_plan_id = _identifier(material_plan_id, "material_plan_id")
        self.profile_id = (
            "iga.structures.shell.rm.tangential-rotation.single-patch.untrimmed"
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-reissner-mindlin-plan",
                "profile": self.profile_id,
                "prepared": prepared_.prepared_id,
                "u": displacement,
                "rotation": rotation,
                "frame": self.tangent_frame_id,
                "shear": self.shear_policy_id,
                "surface_certificate": certificate.certificate_id,
                "material": self.material_plan_id,
            }
        )


__all__ = ["IGAKirchhoffLovePlan", "IGAReissnerMindlinPlan"]

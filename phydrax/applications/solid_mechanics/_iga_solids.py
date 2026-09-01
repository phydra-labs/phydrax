#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._local_variational import AbstractPreparedLocalDiscretization
from ...discretization.iga._certificate import (
    DeformedJacobianCertificate,
    GlobalInjectivityCertificate,
    LocalGeometryCertificate,
)
from ...operators.mechanics import HyperelasticResponse
from ._mixed_hyperelastic import MixedHyperelasticLaw


IGASolidDimensionalMode = Literal["plane_strain", "plane_stress", "axisymmetric", "3d"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _prepared(
    value: AbstractPreparedLocalDiscretization, /
) -> AbstractPreparedLocalDiscretization:
    if not isinstance(value, AbstractPreparedLocalDiscretization):
        raise TypeError("prepared must be an AbstractPreparedLocalDiscretization.")
    return value


def _local_certificate(value: LocalGeometryCertificate, /) -> LocalGeometryCertificate:
    if not isinstance(value, LocalGeometryCertificate):
        raise TypeError("reference_certificate must be a LocalGeometryCertificate.")
    if not bool(value.passed):
        raise ValueError("reference_certificate did not pass; IGA lowering fails closed.")
    return value


class IGASolidFormulation(StrictModule, NonTrainableState):
    """Certified IGA field assignment for an existing linear or hyperelastic law.

    The adapter owns no tensor algebra: constitutive evaluation remains with the
    supplied mechanics material object and discretization remains with ``prepared``.
    """

    prepared: AbstractPreparedLocalDiscretization
    material: MixedHyperelasticLaw | HyperelasticResponse | object
    reference_certificate: LocalGeometryCertificate
    deformed_certificate: (
        DeformedJacobianCertificate | GlobalInjectivityCertificate | None
    )
    displacement_field: str = eqx.field(static=True)
    pressure_field: str | None = eqx.field(static=True)
    dimensional_mode: IGASolidDimensionalMode = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    reference_certificate_id: str = eqx.field(static=True)
    deformed_certificate_policy_id: str | None = eqx.field(static=True)
    material_plan_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: AbstractPreparedLocalDiscretization,
        material: MixedHyperelasticLaw | HyperelasticResponse | object,
        reference_certificate: LocalGeometryCertificate,
        /,
        *,
        displacement_field: str,
        dimensional_mode: IGASolidDimensionalMode,
        profile_id: str,
        material_plan_id: str,
        pressure_field: str | None = None,
        deformed_certificate: DeformedJacobianCertificate
        | GlobalInjectivityCertificate
        | None = None,
    ):
        if material is None:
            raise TypeError("material must be an existing mechanics material object.")
        prepared_ = _prepared(prepared)
        certificate = _local_certificate(reference_certificate)
        if dimensional_mode not in ("plane_strain", "plane_stress", "axisymmetric", "3d"):
            raise ValueError("dimensional_mode is unsupported.")
        if deformed_certificate is not None:
            if not isinstance(
                deformed_certificate,
                (DeformedJacobianCertificate, GlobalInjectivityCertificate),
            ):
                raise TypeError(
                    "deformed_certificate is not an IGA deformation certificate."
                )
            if not bool(deformed_certificate.passed):
                raise ValueError(
                    "deformed_certificate did not pass; IGA lowering fails closed."
                )
        self.prepared = prepared_
        self.material = material
        self.reference_certificate = certificate
        self.deformed_certificate = deformed_certificate
        self.displacement_field = _identifier(displacement_field, "displacement_field")
        self.pressure_field = (
            None
            if pressure_field is None
            else _identifier(pressure_field, "pressure_field")
        )
        self.dimensional_mode = dimensional_mode
        self.profile_id = _identifier(profile_id, "profile_id")
        self.reference_certificate_id = certificate.certificate_id
        self.deformed_certificate_policy_id = (
            None if deformed_certificate is None else deformed_certificate.certificate_id
        )
        self.material_plan_id = _identifier(material_plan_id, "material_plan_id")
        self.formulation_id = canonical_fingerprint(
            {
                "kind": "iga-solid-formulation",
                "profile": self.profile_id,
                "prepared": prepared_.prepared_id,
                "displacement": self.displacement_field,
                "pressure": self.pressure_field,
                "mode": dimensional_mode,
                "reference_certificate": certificate.certificate_id,
                "deformed_certificate": self.deformed_certificate_policy_id,
                "material": self.material_plan_id,
            }
        )


def iga_linear_solid(
    prepared: AbstractPreparedLocalDiscretization,
    material: object,
    reference_certificate: LocalGeometryCertificate,
    /,
    *,
    displacement_field: str = "u",
    dimensional_mode: IGASolidDimensionalMode = "3d",
    profile_id: str = "IGA.Structures.Solid.Linear.SinglePatch.Untrimmed",
    material_plan_id: str,
) -> IGASolidFormulation:
    return IGASolidFormulation(
        prepared,
        material,
        reference_certificate,
        displacement_field=displacement_field,
        dimensional_mode=dimensional_mode,
        profile_id=profile_id,
        material_plan_id=material_plan_id,
    )


def iga_hyperelastic_solid(
    prepared: AbstractPreparedLocalDiscretization,
    material: MixedHyperelasticLaw | HyperelasticResponse,
    reference_certificate: LocalGeometryCertificate,
    /,
    *,
    displacement_field: str = "u",
    pressure_field: str | None = None,
    dimensional_mode: IGASolidDimensionalMode = "3d",
    profile_id: str = "IGA.Structures.Solid.Hyperelastic.SinglePatch.Untrimmed",
    material_plan_id: str,
    deformed_certificate: DeformedJacobianCertificate
    | GlobalInjectivityCertificate
    | None = None,
) -> IGASolidFormulation:
    return IGASolidFormulation(
        prepared,
        material,
        reference_certificate,
        displacement_field=displacement_field,
        pressure_field=pressure_field,
        dimensional_mode=dimensional_mode,
        profile_id=profile_id,
        material_plan_id=material_plan_id,
        deformed_certificate=deformed_certificate,
    )


__all__ = [
    "IGASolidDimensionalMode",
    "IGASolidFormulation",
    "iga_hyperelastic_solid",
    "iga_hyperelastic_u_p_solid",
    "iga_linear_solid",
]


def iga_hyperelastic_u_p_solid(
    prepared: AbstractPreparedLocalDiscretization,
    material: MixedHyperelasticLaw,
    reference_certificate: LocalGeometryCertificate,
    /,
    *,
    displacement_field: str = "u",
    pressure_field: str = "p",
    dimensional_mode: IGASolidDimensionalMode = "3d",
    material_plan_id: str,
    deformed_certificate: DeformedJacobianCertificate
    | GlobalInjectivityCertificate
    | None = None,
) -> IGASolidFormulation:
    """Return the certified displacement-pressure adapter for an existing mixed law."""
    return IGASolidFormulation(
        prepared,
        material,
        reference_certificate,
        displacement_field=displacement_field,
        pressure_field=pressure_field,
        dimensional_mode=dimensional_mode,
        profile_id="IGA.Structures.Solid.Hyperelastic.UP.SinglePatch.Untrimmed",
        material_plan_id=material_plan_id,
        deformed_certificate=deformed_certificate,
    )

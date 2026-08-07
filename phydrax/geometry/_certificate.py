#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ZeroSetAccuracy(str, Enum):
    """Accuracy guarantee for the represented boundary zero set."""

    EXACT = "exact"
    TOLERANCE_BOUND = "tolerance_bound"
    APPROXIMATE = "approximate"


class SignReliability(str, Enum):
    """Reliability of negative-inside/positive-outside classification."""

    RELIABLE = "reliable"
    LOCAL = "local"
    UNRELIABLE = "unreliable"


class DistanceSemantics(str, Enum):
    """Meaning of a scalar boundary field's magnitude."""

    EXACT = "exact_signed_distance"
    APPROXIMATE = "approximate_distance"
    LEVEL_SET = "boundary_level_set"


class FieldRegularity(str, Enum):
    """Regularity class guaranteed by a field construction."""

    SMOOTH = "smooth"
    PIECEWISE_SMOOTH = "piecewise_smooth"
    NONSMOOTH = "nonsmooth"


@dataclass(frozen=True, slots=True)
class FieldCertificate:
    """Machine-readable guarantees carried by a boundary-defining field."""

    zero_set_accuracy: ZeroSetAccuracy
    sign_reliability: SignReliability
    distance_semantics: DistanceSemantics
    regularity: FieldRegularity
    safe_step_factor: float | None
    validity_region: str
    parameter_differentiable: bool
    provenance: tuple[str, ...] = ()

    @property
    def is_signed_distance(self) -> bool:
        return self.distance_semantics is DistanceSemantics.EXACT

    def translated(self) -> FieldCertificate:
        """Return the unchanged guarantees with translation provenance."""
        return replace(self, provenance=(*self.provenance, "rigid_translation"))


_EXACT_SDF_CERTIFICATE = FieldCertificate(
    zero_set_accuracy=ZeroSetAccuracy.EXACT,
    sign_reliability=SignReliability.RELIABLE,
    distance_semantics=DistanceSemantics.EXACT,
    regularity=FieldRegularity.PIECEWISE_SMOOTH,
    safe_step_factor=1.0,
    validity_region="all_space",
    parameter_differentiable=True,
    provenance=("analytic",),
)


def exact_signed_distance_certificate(*, smooth: bool) -> FieldCertificate:
    """Return the canonical certificate for an analytic signed distance."""
    regularity = FieldRegularity.SMOOTH if smooth else FieldRegularity.PIECEWISE_SMOOTH
    return replace(_EXACT_SDF_CERTIFICATE, regularity=regularity)


def sharp_union_certificate(
    certificates: tuple[FieldCertificate, ...],
) -> FieldCertificate:
    """Propagate guarantees through a sharp negative-inside union."""
    if not certificates:
        raise ValueError("A sharp union requires at least one field certificate.")

    zero_rank = {
        ZeroSetAccuracy.EXACT: 0,
        ZeroSetAccuracy.TOLERANCE_BOUND: 1,
        ZeroSetAccuracy.APPROXIMATE: 2,
    }
    sign_rank = {
        SignReliability.RELIABLE: 0,
        SignReliability.LOCAL: 1,
        SignReliability.UNRELIABLE: 2,
    }
    zero_set_accuracy = max(
        (certificate.zero_set_accuracy for certificate in certificates),
        key=zero_rank.__getitem__,
    )
    sign_reliability = max(
        (certificate.sign_reliability for certificate in certificates),
        key=sign_rank.__getitem__,
    )
    return FieldCertificate(
        zero_set_accuracy=zero_set_accuracy,
        sign_reliability=sign_reliability,
        distance_semantics=DistanceSemantics.LEVEL_SET,
        regularity=FieldRegularity.NONSMOOTH,
        safe_step_factor=None,
        validity_region="all_space",
        parameter_differentiable=all(
            certificate.parameter_differentiable for certificate in certificates
        ),
        provenance=(
            *(entry for certificate in certificates for entry in certificate.provenance),
            "sharp_union",
        ),
    )


__all__ = [
    "DistanceSemantics",
    "FieldCertificate",
    "FieldRegularity",
    "SignReliability",
    "ZeroSetAccuracy",
    "exact_signed_distance_certificate",
    "sharp_union_certificate",
]

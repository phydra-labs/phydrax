#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real

from .._validation import optional_identifier


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
    """Machine-readable guarantees carried by a boundary-defining field.

    `lipschitz_upper_bound` bounds the field's Lipschitz constant and
    `evaluation_error` bounds the absolute error of an evaluated field value;
    both are `None` when undeclared. `topology_identity` is the canonical
    identifier of the certified region topology (`None` when uncertified).
    """

    zero_set_accuracy: ZeroSetAccuracy
    sign_reliability: SignReliability
    distance_semantics: DistanceSemantics
    regularity: FieldRegularity
    safe_step_factor: float | None
    validity_region: str
    parameter_differentiable: bool
    provenance: tuple[str, ...] = ()
    lipschitz_upper_bound: float | None = None
    evaluation_error: float | None = None
    topology_identity: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("lipschitz_upper_bound", self.lipschitz_upper_bound),
            ("evaluation_error", self.evaluation_error),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"FieldCertificate.{name} must be finite and non-negative when declared."
                )
        optional_identifier(self.topology_identity, "FieldCertificate.topology_identity")

    @property
    def is_signed_distance(self) -> bool:
        return self.distance_semantics is DistanceSemantics.EXACT

    def translated(self) -> FieldCertificate:
        """Return the unchanged guarantees with translation provenance."""
        return replace(self, provenance=(*self.provenance, "rigid_translation"))


@dataclass(frozen=True, slots=True)
class ExactSDFEnclosureCertificate:
    """Global Lipschitz enclosure qualification of an exact SDF field certificate.

    The field certificate owns the declared evaluation error and Lipschitz
    upper bound. This certificate qualifies interval sign classification. It
    does not by itself claim exact cell measures: boxes intersecting the zero
    set remain as explicit lower/upper measure uncertainty.
    """

    field: FieldCertificate

    def __post_init__(self) -> None:
        if not isinstance(self.field, FieldCertificate):
            raise TypeError(
                "ExactSDFEnclosureCertificate.field must be a FieldCertificate."
            )
        if (
            self.field.zero_set_accuracy is not ZeroSetAccuracy.EXACT
            or self.field.sign_reliability is not SignReliability.RELIABLE
            or self.field.distance_semantics is not DistanceSemantics.EXACT
            or self.field.validity_region != "all_space"
        ):
            raise ValueError(
                "Exact-SDF measure enclosure requires a globally reliable exact signed-distance certificate."
            )
        if (
            self.field.evaluation_error is None
            or self.field.lipschitz_upper_bound is None
            or self.field.lipschitz_upper_bound < 1.0
        ):
            raise ValueError(
                "Exact-SDF measure enclosure requires a declared evaluation error and "
                "a declared Lipschitz upper bound of at least one."
            )

    @property
    def certifies_global_enclosure(self) -> bool:
        return True


_EXACT_SDF_CERTIFICATE = FieldCertificate(
    zero_set_accuracy=ZeroSetAccuracy.EXACT,
    sign_reliability=SignReliability.RELIABLE,
    distance_semantics=DistanceSemantics.EXACT,
    regularity=FieldRegularity.PIECEWISE_SMOOTH,
    safe_step_factor=1.0,
    validity_region="all_space",
    parameter_differentiable=True,
    provenance=("analytic",),
    lipschitz_upper_bound=1.0,
    evaluation_error=0.0,
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
    "ExactSDFEnclosureCertificate",
    "DistanceSemantics",
    "FieldCertificate",
    "FieldRegularity",
    "SignReliability",
    "ZeroSetAccuracy",
    "exact_signed_distance_certificate",
    "sharp_union_certificate",
]

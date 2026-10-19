#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, TypeAlias

from .._validation import nonnegative_real_scalar, optional_identifier
from ..typing import checked, parse


FieldBoundOrigin: TypeAlias = Literal["declared", "established", "sampled"]
"""Provenance of a field's Lipschitz and evaluation-error bounds.

``established`` bounds are derived by the field representation (for example an
analytic signed distance); ``declared`` bounds are supplied by the caller and
are trusted, not verified; ``sampled`` bounds are estimated from samples and
bound nothing between them.
"""


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
    both are `None` when undeclared. `bound_origin` distinguishes bounds derived
    by the representation from caller-declared or sampled bounds.
    `topology_identity` identifies the certified region topology
    (`None` when uncertified).
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
    bound_origin: FieldBoundOrigin = "declared"

    def __post_init__(self) -> None:
        for name, value in (
            ("lipschitz_upper_bound", self.lipschitz_upper_bound),
            ("evaluation_error", self.evaluation_error),
        ):
            if value is not None:
                nonnegative_real_scalar(value, f"FieldCertificate.{name}")
        optional_identifier(self.topology_identity, "FieldCertificate.topology_identity")
        parse(self.bound_origin, FieldBoundOrigin, "FieldCertificate.bound_origin")

    @property
    def is_signed_distance(self) -> bool:
        return self.distance_semantics is DistanceSemantics.EXACT

    @property
    def bounds_established(self) -> bool:
        """Whether the field representation establishes both bounds."""
        return (
            self.bound_origin == "established"
            and self.lipschitz_upper_bound is not None
            and self.evaluation_error is not None
        )

    def translated(self) -> FieldCertificate:
        """Return the unchanged guarantees with translation provenance."""
        return replace(self, provenance=(*self.provenance, "rigid_translation"))


@dataclass(frozen=True, slots=True)
class ExactSDFEnclosureCertificate:
    """Qualify global Lipschitz enclosures for a signed-distance field certificate.

    The field certificate owns the evaluation error and Lipschitz upper bound.
    This certificate qualifies interval sign classification only when those
    bounds are established by the field representation
    (``certifies_global_enclosure``); caller-declared or sampled bounds leave the
    enclosure unqualified. Boxes intersecting the zero set retain explicit
    lower/upper measure uncertainty.
    """

    field: FieldCertificate

    @checked
    def __init__(self, field: FieldCertificate) -> None:
        object.__setattr__(self, "field", field)
        if (
            self.field.zero_set_accuracy is not ZeroSetAccuracy.EXACT
            or self.field.sign_reliability is not SignReliability.RELIABLE
            or self.field.distance_semantics is not DistanceSemantics.EXACT
            or self.field.validity_region != "all_space"
        ):
            raise ValueError(
                f"ExactSDFEnclosureCertificate.field must describe a globally reliable signed distance; got {self.field!r}"
            )
        if (
            self.field.evaluation_error is None
            or self.field.lipschitz_upper_bound is None
            or self.field.lipschitz_upper_bound < 1.0
        ):
            raise ValueError(
                f"ExactSDFEnclosureCertificate.field must declare evaluation error and Lipschitz bound >= 1; got {self.field!r}"
            )

    @property
    def certifies_global_enclosure(self) -> bool:
        return self.field.bounds_established


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
    bound_origin="established",
)


def exact_signed_distance_certificate(*, smooth: bool) -> FieldCertificate:
    """Return analytic signed-distance guarantees."""
    regularity = FieldRegularity.SMOOTH if smooth else FieldRegularity.PIECEWISE_SMOOTH
    return replace(_EXACT_SDF_CERTIFICATE, regularity=regularity)


def sharp_union_certificate(
    certificates: tuple[FieldCertificate, ...],
) -> FieldCertificate:
    """Propagate guarantees through a sharp negative-inside union."""
    if not certificates:
        raise ValueError("sharp_union_certificate.certificates must be nonempty; got ()")

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


@checked
def level_set_interpolation_certificate(source: FieldCertificate, /) -> FieldCertificate:
    """Downgrade a sampled field to sparse multilinear level-set guarantees.

    Interpolation does not preserve the signed-distance identity, source zero
    sets, enclosure bounds, or topology identity. Sign reliability is at most
    local within populated sparse stencils and no stronger than the source's.
    Derivatives are piecewise smooth and no safe sphere-tracing step is claimed.
    """
    return FieldCertificate(
        zero_set_accuracy=ZeroSetAccuracy.APPROXIMATE,
        sign_reliability=(
            SignReliability.UNRELIABLE
            if source.sign_reliability is SignReliability.UNRELIABLE
            else SignReliability.LOCAL
        ),
        distance_semantics=DistanceSemantics.LEVEL_SET,
        regularity=FieldRegularity.PIECEWISE_SMOOTH,
        safe_step_factor=None,
        validity_region="sparse_voxel_support",
        parameter_differentiable=source.parameter_differentiable,
        provenance=(*source.provenance, "multilinear_level_set_interpolation"),
    )


__all__ = [
    "ExactSDFEnclosureCertificate",
    "DistanceSemantics",
    "FieldBoundOrigin",
    "FieldCertificate",
    "FieldRegularity",
    "SignReliability",
    "ZeroSetAccuracy",
    "exact_signed_distance_certificate",
    "level_set_interpolation_certificate",
    "sharp_union_certificate",
]

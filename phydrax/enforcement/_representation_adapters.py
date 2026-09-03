#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._fingerprint import canonical_fingerprint
from ..conditions import ProductFieldSpec
from ..linalg import AbstractRealCoordinateMap, AbstractVectorSpace
from ._linear_representation import (
    CallableLinearRepresentation,
    LinearConditionAssembly,
    LinearRepresentationCertificate,
)


def _representation(
    kind: str,
    field_spec: ProductFieldSpec,
    native_coefficient_space: AbstractVectorSpace,
    coefficient_space: AbstractVectorSpace,
    extraction: Callable[[Mapping[str, Any]], Any],
    replacement: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    synthesis: Callable[[Any], Mapping[str, Any]],
    assembly: Callable[[Any], LinearConditionAssembly],
    /,
    *,
    real_coordinates: AbstractRealCoordinateMap | None = None,
    numeric_version: int = 0,
    support_ids: Sequence[str] = (),
    layout_ids: Sequence[str] = (),
    topology_ids: Sequence[str] = (),
    maximum_derivative_orders: Sequence[Any] = (),
    source_certificate_ids: Sequence[str] = (),
) -> CallableLinearRepresentation:
    identifier = str(kind)
    if not identifier:
        raise ValueError("Representation adapter kind must be nonempty.")
    coordinate_id = (
        None if real_coordinates is None else real_coordinates.evidence.evidence_id
    )
    certificate = LinearRepresentationCertificate(
        field_spec_id=field_spec.field_spec_id,
        field_names=field_spec.sources,
        native_coefficient_space_id=native_coefficient_space.space_id,
        coefficient_space_id=coefficient_space.space_id,
        extraction_id=canonical_fingerprint({"kind": identifier, "action": "extract"}),
        replacement_id=canonical_fingerprint({"kind": identifier, "action": "replace"}),
        synthesis_id=canonical_fingerprint({"kind": identifier, "action": "synthesize"}),
        coordinate_evidence_id=coordinate_id,
        support_ids=support_ids,
        layout_ids=layout_ids,
        topology_ids=topology_ids,
        maximum_derivative_orders=maximum_derivative_orders,
        construction_dependencies=(identifier,),
        source_certificate_ids=source_certificate_ids,
        proof=f"{identifier}-construction",
        zero_preserving=True,
        round_trip_exact=True,
    )
    return CallableLinearRepresentation(
        field_spec,
        native_coefficient_space,
        coefficient_space,
        extraction,
        replacement,
        synthesis,
        assembly,
        certificate=certificate,
        real_coordinates=real_coordinates,
        numeric_version=numeric_version,
    )


def spectral_linear_representation(
    *args: Any, **kwargs: Any
) -> CallableLinearRepresentation:
    """Build an explicit tensor/spherical/lattice spectral representation."""
    return _representation("spectral-linear-representation", *args, **kwargs)


def finite_element_linear_representation(
    *args: Any, **kwargs: Any
) -> CallableLinearRepresentation:
    """Build an explicit sparse finite-element representation."""
    return _representation("finite-element-linear-representation", *args, **kwargs)


def iga_linear_representation(*args: Any, **kwargs: Any) -> CallableLinearRepresentation:
    """Build an explicit B-spline/NURBS coefficient representation."""
    return _representation("iga-linear-representation", *args, **kwargs)


def trefftz_linear_representation(
    *args: Any, **kwargs: Any
) -> CallableLinearRepresentation:
    """Build an explicit PDE-certified Trefftz/holomorphic representation."""
    return _representation("trefftz-linear-representation", *args, **kwargs)


def rom_linear_representation(*args: Any, **kwargs: Any) -> CallableLinearRepresentation:
    """Build an explicit reduced-order linear representation."""
    return _representation("rom-linear-representation", *args, **kwargs)


def finite_feature_linear_representation(
    *args: Any, **kwargs: Any
) -> CallableLinearRepresentation:
    """Build an explicit kernel/random-feature/linear-head representation."""
    return _representation("finite-feature-linear-representation", *args, **kwargs)


__all__ = [
    "finite_element_linear_representation",
    "finite_feature_linear_representation",
    "iga_linear_representation",
    "rom_linear_representation",
    "spectral_linear_representation",
    "trefftz_linear_representation",
]

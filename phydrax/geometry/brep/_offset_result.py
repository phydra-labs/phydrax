#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Offset-intent rounding evidence, separate from intrinsic carrier gaps."""

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...typing import checked, parse
from ._model import BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range


BRepOffsetVolumeStatus: TypeAlias = Literal["certified", "unresolved"]


@final
class BRepOffsetEvidence(StrictModule):
    """Distance bounds against the rational offset intent in model entity order.

    Plane and vertex bounds use the model length unit. Volume enclosures use its
    cube and include coefficient rounding, not only quadrature estimates.
    """

    face_rounding_bounds: Array | None
    vertex_rounding_bounds: Array | None
    volume_enclosure: Array | None
    volume_status: BRepOffsetVolumeStatus = eqx.field(static=True)
    volume_reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_rounding_bounds: ArrayLike | None,
        vertex_rounding_bounds: ArrayLike | None,
        volume_enclosure: ArrayLike | None,
        volume_reason: str = "",
    ) -> None:
        faces = (
            None
            if face_rounding_bounds is None
            else jnp.asarray(face_rounding_bounds, dtype=jnp.float64)
        )
        vertices = (
            None
            if vertex_rounding_bounds is None
            else jnp.asarray(vertex_rounding_bounds, dtype=jnp.float64)
        )
        enclosure = (
            None
            if volume_enclosure is None
            else jnp.asarray(volume_enclosure, dtype=jnp.float64)
        )
        for name, values in (("face", faces), ("vertex", vertices)):
            if values is None:
                continue
            host = np.asarray(values)
            if host.ndim != 1 or np.any(~np.isfinite(host)) or np.any(host < 0.0):
                raise ValueError(
                    f"BRepOffsetEvidence.rounding_bounds must be finite nonnegative rows; got invalid {name} values"
                )
        if enclosure is not None:
            host = np.asarray(enclosure)
            if host.shape != (2,) or np.any(~np.isfinite(host)) or host[0] > host[1]:
                raise ValueError(
                    "BRepOffsetEvidence.volume_enclosure must have finite ordered endpoints; got an invalid interval"
                )
        self.face_rounding_bounds = faces
        self.vertex_rounding_bounds = vertices
        self.volume_enclosure = enclosure
        self.volume_status = parse(
            "unresolved" if enclosure is None else "certified",
            BRepOffsetVolumeStatus,
            "offset.volume_status",
        )
        self.volume_reason = volume_reason


@final
class BRepOffsetResult(StrictModule):
    """Published offset model and its certified operation-intent evidence."""

    model: BRepModel
    evidence: BRepOffsetEvidence
    parameter_range: CertifiedParameterRange | None

    @checked
    def __init__(
        self,
        model: BRepModel,
        evidence: BRepOffsetEvidence,
        /,
        *,
        parameter_range: CertifiedParameterRange | None = None,
    ) -> None:
        geometry = model.require_geometry()
        face_bounds = evidence.face_rounding_bounds
        vertex_bounds = evidence.vertex_rounding_bounds
        if (face_bounds is not None and face_bounds.shape != (len(model.patches),)) or (
            vertex_bounds is not None
            and vertex_bounds.shape != (geometry.vertex_points.shape[0],)
        ):
            raise ValueError(
                "BRepOffsetResult.evidence must align with model face and vertex rows; got different counts"
            )
        self.model = model
        self.evidence = evidence
        self.parameter_range = parameter_range


def exact_operation_offset_result(
    model: BRepModel,
    /,
    *,
    include_parameter_range: bool = False,
) -> BRepOffsetResult:
    """Keep native normal-operation trees separate from rational-plane intent proofs."""
    return BRepOffsetResult(
        model,
        BRepOffsetEvidence(
            face_rounding_bounds=None,
            vertex_rounding_bounds=None,
            volume_enclosure=None,
            volume_reason="smooth native offset intent has no rational-plane rounding proof; volume requires the certified mass-property query",
        ),
        parameter_range=(
            unknown_parameter_range(
                "distance", "smooth offset family has no certified parameter interval"
            )
            if include_parameter_range
            else None
        ),
    )

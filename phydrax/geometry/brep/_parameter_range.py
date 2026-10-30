#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified construction parameter intervals and refusal evidence."""

from __future__ import annotations

from math import isnan
from typing import Literal, TypeAlias, final

import equinox as eqx

from ..._strict import StrictModule
from ...typing import parse
from ._model import BRepEntityId


ParameterRangeEndStatus: TypeAlias = Literal[
    "attained_exactly", "certified_inner_bound", "unknown"
]
ParameterRangeConstraint: TypeAlias = Literal[
    "parameter_domain",
    "face_vanishes",
    "trim_vanishes",
    "edges_collide",
    "edge_reverses",
    "reach",
    "det_j_sign",
    "unknown",
]


@final
class CertifiedParameterRange(StrictModule):
    """Open certified interval with independently justified endpoints."""

    parameter: str = eqx.field(static=True)
    interval: tuple[float | None, float | None] = eqx.field(static=True)
    end_status: tuple[ParameterRangeEndStatus, ParameterRangeEndStatus] = eqx.field(static=True)
    constraints: tuple[ParameterRangeConstraint, ParameterRangeConstraint] = eqx.field(static=True)
    entity_ids: tuple[tuple[BRepEntityId, ...], tuple[BRepEntityId, ...]] = eqx.field(static=True)
    evidence: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        parameter: str,
        interval: tuple[float | None, float | None],
        *,
        end_status: tuple[ParameterRangeEndStatus, ParameterRangeEndStatus],
        constraints: tuple[ParameterRangeConstraint, ParameterRangeConstraint],
        entity_ids: tuple[tuple[BRepEntityId, ...], tuple[BRepEntityId, ...]] = ((), ()),
        evidence: tuple[str, ...],
    ) -> None:
        if not isinstance(parameter, str) or not parameter or not evidence:
            raise ValueError("A parameter range requires a named parameter and proof evidence.")
        if any(
            len(values) != 2 for values in (interval, end_status, constraints, entity_ids)
        ):
            raise ValueError("A parameter range must provide exactly two endpoint records.")
        statuses = tuple(
            parse(value, ParameterRangeEndStatus, "range.end_status")
            for value in end_status
        )
        bindings = tuple(
            parse(value, ParameterRangeConstraint, "range.constraint")
            for value in constraints
        )
        for value, status in zip(interval, statuses, strict=True):
            if (value is None) != (status == "unknown") or (
                value is not None and isnan(value)
            ):
                raise ValueError("Unknown range ends must use None and certified ends cannot be NaN.")
        lower, upper = interval
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("A certified parameter interval must have strictly ordered endpoints.")
        if any(
            not isinstance(entity, BRepEntityId)
            for row in entity_ids
            for entity in row
        ):
            raise TypeError("Parameter-range entity bindings must be BRepEntityId values.")
        evidence_ = tuple(evidence)
        if any(not isinstance(item, str) or not item for item in evidence_):
            raise ValueError("Parameter-range evidence entries must be nonempty strings.")
        self.parameter = parameter
        self.interval = interval
        self.end_status = (statuses[0], statuses[1])
        self.constraints = (bindings[0], bindings[1])
        self.entity_ids = (tuple(entity_ids[0]), tuple(entity_ids[1]))
        self.evidence = evidence_


def unknown_parameter_range(parameter: str, reason: str, /) -> CertifiedParameterRange:
    """Retain unknown endpoints explicitly rather than substituting a guess."""
    return CertifiedParameterRange(
        parameter,
        (None, None),
        end_status=("unknown", "unknown"),
        constraints=("unknown", "unknown"),
        evidence=(reason,),
    )


def attach_parameter_range(
    error: ValueError, parameter_range: CertifiedParameterRange, /
) -> ValueError:
    """Attach typed interval evidence without changing the refusal itself."""
    if not isinstance(error, ValueError) or not isinstance(
        parameter_range, CertifiedParameterRange
    ):
        raise TypeError("attach_parameter_range requires a ValueError and range certificate.")
    error.__dict__["parameter_range"] = parameter_range
    return error


def refusal_parameter_range(error: ValueError, /) -> CertifiedParameterRange:
    """Return a refusal's certified parameter range or reject missing evidence."""
    value = error.__dict__.get("parameter_range")
    if not isinstance(value, CertifiedParameterRange):
        raise ValueError("Refusal has no certified parameter range.")
    return value


__all__ = [
    "CertifiedParameterRange",
    "ParameterRangeConstraint",
    "ParameterRangeEndStatus",
    "attach_parameter_range",
    "refusal_parameter_range",
    "unknown_parameter_range",
]

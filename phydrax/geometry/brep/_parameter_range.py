#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified construction parameter intervals and refusal evidence."""

from __future__ import annotations

from math import isnan
from typing import final, Literal, TypeAlias

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
    """An open certified interval, with independently justified endpoints.

    An ``attained_exactly`` endpoint names the limiting value, not admissibility
    at that value. ``None`` is an unknown endpoint, never an unbounded feasible end;
    a proved unbounded end is represented by signed infinity. Evidence describes
    the proof and its family, not a numerical estimate.
    """

    parameter: str = eqx.field(static=True)
    interval: tuple[float | None, float | None] = eqx.field(static=True)
    end_status: tuple[ParameterRangeEndStatus, ParameterRangeEndStatus] = eqx.field(
        static=True
    )
    constraints: tuple[ParameterRangeConstraint, ParameterRangeConstraint] = eqx.field(
        static=True
    )
    entity_ids: tuple[tuple[BRepEntityId, ...], tuple[BRepEntityId, ...]] = eqx.field(
        static=True
    )
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
        if not parameter or not evidence:
            raise ValueError(
                "CertifiedParameterRange.parameter must accompany proof evidence; got an empty parameter or evidence"
            )
        if any(
            len(values) != 2 for values in (interval, end_status, constraints, entity_ids)
        ):
            raise ValueError(
                "CertifiedParameterRange.endpoints must each have two entries; got a different count"
            )
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
                raise ValueError(
                    "CertifiedParameterRange.interval must use None for unknown ends and non-NaN certified ends; got inconsistent endpoint evidence"
                )
        lower, upper = interval
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                "CertifiedParameterRange.interval must have strictly ordered endpoints; got an empty or reversed interval"
            )
        self.parameter = parameter
        self.interval = interval
        self.end_status = (statuses[0], statuses[1])
        self.constraints = (bindings[0], bindings[1])
        self.entity_ids = entity_ids
        self.evidence = evidence


def unknown_parameter_range(parameter: str, reason: str, /) -> CertifiedParameterRange:
    """Retain an explicit unknown certificate for an unsupported family."""
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
    """Enrich an existing refusal without changing its type, args, or message."""
    error.__dict__["parameter_range"] = parameter_range
    return error


def refusal_parameter_range(error: ValueError, /) -> CertifiedParameterRange:
    """Read the typed certificate attached to a construction refusal."""
    value = error.__dict__.get("parameter_range")
    if not isinstance(value, CertifiedParameterRange):
        raise ValueError(
            "refusal_parameter_range.evidence must contain a certified parameter range; got absent evidence"
        )
    return value

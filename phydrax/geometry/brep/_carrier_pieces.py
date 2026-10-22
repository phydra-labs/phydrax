#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Describe interval-evaluable source pieces without importing carrier classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from typing import Generic, TypeVar

import equinox as eqx
import jax
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint
from ..._host_storage import declared_object_storage_bytes
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked


_CarrierT = TypeVar("_CarrierT", bound=StrictModule, covariant=True)


@checked
def carrier_payload(carrier: StrictModule, /) -> dict[str, object]:
    """Fingerprint a curve/surface carrier from its family, metadata, and arrays."""
    return {
        "family": type(carrier).__name__,
        "structure": str(jax.tree_util.tree_structure(carrier)),
        "arrays": array_tree_fingerprint(carrier),
    }


def admit_carrier_fingerprint(
    source: object,
    budget: CoordinateEnclosureBudget,
    ancestors: tuple[int, ...] = (),
) -> None:
    """Admit live payload metadata and host transfers before fingerprinting.

    Carrier payloads retain paths and PyTree static construction metadata, not
    whole-array encodings. Each nesting level can appear in every descendant's
    path; the structural allowance covers those paths, escaped JSON and its
    simultaneously live UTF-8 representation. Array transfers are additional.
    This is byte admission only, not a new scientific work charge.
    """
    depth = len(ancestors) + 1
    budget.reserve(0, 32 * depth * declared_object_storage_bytes(source))
    if id(source) in ancestors:
        raise ValueError("Correspondence fingerprint source must be acyclic.")
    ancestors = (*ancestors, id(source))
    if isinstance(source, (Array, np.ndarray)):
        budget.reserve(0, source.nbytes)
    elif is_dataclass(source) and not isinstance(source, type):
        for field in fields(source):
            admit_carrier_fingerprint(field.name, budget, ancestors)
            admit_carrier_fingerprint(getattr(source, field.name), budget, ancestors)
    elif isinstance(source, dict):
        for key, value in source.items():
            admit_carrier_fingerprint(key, budget, ancestors)
            admit_carrier_fingerprint(value, budget, ancestors)
    elif isinstance(source, (tuple, list, set, frozenset)):
        for value in source:
            admit_carrier_fingerprint(value, budget, ancestors)
    elif isinstance(source, Fraction):
        admit_carrier_fingerprint(source.numerator, budget, ancestors)
        admit_carrier_fingerprint(source.denominator, budget, ancestors)


class CoefficientEnclosureSource(ABC):
    """Provide source coefficient intervals to the numerical interval interpreter."""

    @abstractmethod
    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return nominal coefficient arrays and their lower and upper bounds."""
        raise NotImplementedError


class CurveCoordinateResult(StrictModule):
    """Expose coordinates from a coupled curve result without adding stored fields."""

    __strict_abstract__ = True

    point: eqx.AbstractVar[Array]


class RootCurveCoordinateSource(ABC):
    """Identify a curve whose coordinate evaluation executes a coupled root solve."""

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> CurveCoordinateResult:
        """Return coordinates together with the coupled root's numerical evidence."""
        raise NotImplementedError


def child_coefficient_enclosures(
    system: StrictModule, /
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Collect construction intervals from descendant sources, excluding the root."""
    children = jax.tree_util.tree_leaves(
        system,
        is_leaf=lambda item: (
            item is not system and isinstance(item, CoefficientEnclosureSource)
        ),
    )
    return tuple(
        bound
        for child in children
        if isinstance(child, CoefficientEnclosureSource)
        for bound in child.coefficient_enclosures()
    )


class AffineTrimChordEvidence(StrictModule):
    """Enclose affine trim endpoints in the stored normalized chart coordinates."""

    nominal_start: tuple[Fraction, Fraction] = eqx.field(static=True)
    nominal_end: tuple[Fraction, Fraction] = eqx.field(static=True)
    start_lower: np.ndarray
    start_upper: np.ndarray
    end_lower: np.ndarray
    end_upper: np.ndarray
    source_id: str = eqx.field(static=True)
    endpoint_root_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        nominal_start: tuple[Fraction, Fraction],
        nominal_end: tuple[Fraction, Fraction],
        start_lower: np.ndarray,
        start_upper: np.ndarray,
        end_lower: np.ndarray,
        end_upper: np.ndarray,
        source_id: str,
        endpoint_root_ids: tuple[str, ...],
    ) -> None:
        boxes = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (start_lower, start_upper, end_lower, end_upper)
        )
        for nominal, lower, upper in (
            (nominal_start, boxes[0], boxes[1]),
            (nominal_end, boxes[2], boxes[3]),
        ):
            if (
                lower.shape != (2,)
                or upper.shape != (2,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > upper)
                or any(
                    Fraction(float(lower[axis])) > value
                    or Fraction(float(upper[axis])) < value
                    for axis, value in enumerate(nominal)
                )
            ):
                raise ValueError(
                    "AffineTrimChordEvidence.endpoints must have finite ordered two-coordinate bounds containing their rational nominal endpoints; "
                    f"got nominal={nominal!r}, lower={lower!r}, upper={upper!r}."
                )
        if not source_id:
            raise ValueError(
                f"AffineTrimChordEvidence.source_id must be nonempty; got {source_id!r}."
            )
        self.nominal_start, self.nominal_end = nominal_start, nominal_end
        self.start_lower, self.start_upper, self.end_lower, self.end_upper = boxes
        self.source_id = source_id
        self.endpoint_root_ids = endpoint_root_ids


@dataclass(frozen=True, slots=True)
class SurfacePiece(Generic[_CarrierT]):
    """Retain one source evaluator, closed parameter box, and deterministic index."""

    evaluator: _CarrierT
    lower: np.ndarray
    upper: np.ndarray
    index: int


@dataclass(frozen=True, slots=True)
class CurvePiece(Generic[_CarrierT]):
    """Retain one source evaluator, closed parameter interval, and deterministic index."""

    evaluator: _CarrierT
    lower: float
    upper: float
    index: int


__all__ = [
    "admit_carrier_fingerprint",
    "AffineTrimChordEvidence",
    "carrier_payload",
    "child_coefficient_enclosures",
    "CoefficientEnclosureSource",
    "CurveCoordinateResult",
    "CurvePiece",
    "RootCurveCoordinateSource",
    "SurfacePiece",
]

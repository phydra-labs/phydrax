#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small source-piece records and coefficient identity helpers for B-Rep math."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
import jax
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule

_CarrierT = TypeVar("_CarrierT", bound=StrictModule, covariant=True)


def carrier_payload(carrier: StrictModule, /) -> dict[str, object]:
    """Fingerprint the complete immutable source definition, not query state."""
    identity = getattr(carrier, "identity_payload", None)
    return {
        "type": f"{type(carrier).__module__}.{type(carrier).__qualname__}",
        "structure": str(jax.tree_util.tree_structure(carrier)),
        "arrays": array_tree_fingerprint(carrier),
        "definition": identity() if callable(identity) else None,
    }


class CoefficientEnclosureSource(ABC):
    """Optional explicit source coefficients for interval source evaluators."""

    @abstractmethod
    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        raise NotImplementedError



@dataclass(frozen=True, slots=True)
class SurfacePiece(Generic[_CarrierT]):
    evaluator: _CarrierT
    lower: np.ndarray
    upper: np.ndarray
    index: int


@dataclass(frozen=True, slots=True)
class CurvePiece(Generic[_CarrierT]):
    evaluator: _CarrierT
    lower: float
    upper: float
    index: int


class AffineTrimChordEvidence(StrictModule):
    """Exact rational endpoints with separate outward numerical images."""

    nominal_start: tuple[object, object] = eqx.field(static=True)
    nominal_end: tuple[object, object] = eqx.field(static=True)
    start_lower: np.ndarray
    start_upper: np.ndarray
    end_lower: np.ndarray
    end_upper: np.ndarray
    source_id: str = eqx.field(static=True)
    endpoint_root_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        nominal_start: tuple[object, object],
        nominal_end: tuple[object, object],
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
        if any(value.shape != (2,) or not np.all(np.isfinite(value)) for value in boxes):
            raise ValueError("Affine trim endpoint enclosures must be finite 2-vectors.")
        if np.any(boxes[0] > boxes[1]) or np.any(boxes[2] > boxes[3]):
            raise ValueError("Affine trim endpoint enclosures must be ordered.")
        if not source_id:
            raise ValueError("Affine trim chord source_id must be nonempty.")
        self.nominal_start, self.nominal_end = nominal_start, nominal_end
        self.start_lower, self.start_upper, self.end_lower, self.end_upper = boxes
        self.source_id = source_id
        self.endpoint_root_ids = tuple(endpoint_root_ids)


def child_coefficient_enclosures(
    system: StrictModule, /
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Return concrete floating leaves as exact represented coefficient boxes."""
    return tuple(
        (array, array, array)
        for leaf in jax.tree_util.tree_leaves(system)
        if isinstance(leaf, (np.ndarray, np.generic))
        for array in (np.asarray(leaf, dtype=np.float64),)
        if np.all(np.isfinite(array))
    )


__all__ = [
    "AffineTrimChordEvidence",
    "CoefficientEnclosureSource",
    "CurvePiece",
    "SurfacePiece",
    "carrier_payload",
    "child_coefficient_enclosures",
]

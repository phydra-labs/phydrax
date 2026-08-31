#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class TopologyResourceError(RuntimeError):
    """An exact topology calculation exceeded an explicit resource budget."""


class TopologyResourcePolicy(StrictModule, NonTrainableState):
    """Fail-closed resource limits for exact topology preprocessing."""

    max_cells: int = eqx.field(static=True)
    max_boundary_nonzeros: int = eqx.field(static=True)
    max_reduction_entries: int = eqx.field(static=True)
    max_operations: int = eqx.field(static=True)
    max_representative_entries: int = eqx.field(static=True)
    max_rational_bit_length: int = eqx.field(static=True)
    max_packed_intervals: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_cells: int = 100_000,
        max_boundary_nonzeros: int = 2_000_000,
        max_reduction_entries: int = 5_000_000,
        max_operations: int = 100_000_000,
        max_representative_entries: int = 5_000_000,
        max_rational_bit_length: int = 16_384,
        max_packed_intervals: int = 1_000_000,
    ):
        values = {
            "max_cells": int(max_cells),
            "max_boundary_nonzeros": int(max_boundary_nonzeros),
            "max_reduction_entries": int(max_reduction_entries),
            "max_operations": int(max_operations),
            "max_representative_entries": int(max_representative_entries),
            "max_rational_bit_length": int(max_rational_bit_length),
            "max_packed_intervals": int(max_packed_intervals),
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("Topology resource limits must be positive.")
        for name, value in values.items():
            setattr(self, name, value)
        self.policy_id = canonical_fingerprint(
            {"kind": "topology-resource-policy", **values}
        )


class TopologyReductionEvidence(StrictModule, NonTrainableState):
    """Auditable exact-algebra work and verification evidence."""

    algorithm: str = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)
    counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    verified: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        algorithm: str,
        coefficient_id: str,
        counts: Mapping[str, int] | tuple[tuple[str, int], ...],
        /,
        *,
        exact: bool = True,
        verified: bool = True,
    ):
        algorithm_ = str(algorithm)
        coefficient_id_ = str(coefficient_id)
        if not algorithm_ or not coefficient_id_:
            raise ValueError("Reduction algorithm and coefficient identity are required.")
        items = tuple(counts.items()) if isinstance(counts, Mapping) else tuple(counts)
        counts_ = tuple(sorted((str(name), int(value)) for name, value in items))
        if any(not name or value < 0 for name, value in counts_):
            raise ValueError("Reduction counts must be named non-negative integers.")
        if len({name for name, _ in counts_}) != len(counts_):
            raise ValueError("Reduction count names must be unique.")
        self.algorithm = algorithm_
        self.coefficient_id = coefficient_id_
        self.counts = counts_
        self.exact = bool(exact)
        self.verified = bool(verified)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "topology-reduction-evidence",
                "algorithm": algorithm_,
                "coefficient": coefficient_id_,
                "counts": [list(value) for value in counts_],
                "exact": bool(exact),
                "verified": bool(verified),
            }
        )


__all__ = [
    "TopologyReductionEvidence",
    "TopologyResourceError",
    "TopologyResourcePolicy",
]

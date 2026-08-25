#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._resources import CliffordResourceEvidence


class CliffordProductEvidence(StrictModule, NonTrainableState):
    """Structural and resource evidence for one exact prepared blade product."""

    algebra_id: str = eqx.field(static=True)
    left_layout_id: str = eqx.field(static=True)
    right_layout_id: str = eqx.field(static=True)
    output_layout_id: str = eqx.field(static=True)
    product_kind: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    structural_zero_count: int = eqx.field(static=True)
    exact_closure: bool = eqx.field(static=True)
    resource_evidence: CliffordResourceEvidence
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        algebra_id: str,
        left_layout_id: str,
        right_layout_id: str,
        output_layout_id: str,
        product_kind: str,
        backend: str,
        term_count: int,
        structural_zero_count: int,
        exact_closure: bool,
        resource_evidence: CliffordResourceEvidence,
    ):
        identifiers = tuple(
            str(value)
            for value in (
                algebra_id,
                left_layout_id,
                right_layout_id,
                output_layout_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Clifford product evidence IDs must be non-empty.")
        if backend not in ("dense", "sparse"):
            raise ValueError("Clifford product backend must be 'dense' or 'sparse'.")
        terms = int(term_count)
        zeros = int(structural_zero_count)
        if terms < 0 or zeros < 0:
            raise ValueError("Clifford product evidence counts must be nonnegative.")
        if not isinstance(resource_evidence, CliffordResourceEvidence):
            raise TypeError("resource_evidence must be CliffordResourceEvidence.")
        self.algebra_id = identifiers[0]
        self.left_layout_id = identifiers[1]
        self.right_layout_id = identifiers[2]
        self.output_layout_id = identifiers[3]
        self.product_kind = str(product_kind)
        self.backend = backend
        self.term_count = terms
        self.structural_zero_count = zeros
        self.exact_closure = bool(exact_closure)
        self.resource_evidence = resource_evidence
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "clifford-product-evidence-v1",
                "algebra": identifiers[0],
                "left": identifiers[1],
                "right": identifiers[2],
                "output": identifiers[3],
                "product": str(product_kind),
                "backend": backend,
                "terms": terms,
                "structural_zeros": zeros,
                "exact_closure": bool(exact_closure),
                "resources": resource_evidence.evidence_id,
            }
        )


__all__ = ["CliffordProductEvidence"]

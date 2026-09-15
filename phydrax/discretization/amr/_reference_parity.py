#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provenance-bound independent reference comparisons for block AMR."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class BlockAMRReferenceParityEvidence(StrictModule, NonTrainableState):
    """Observable-wise mixed-tolerance defects against an independent artifact."""

    names: tuple[str, ...] = eqx.field(static=True)
    maximum_absolute_defects: tuple[float, ...] = eqx.field(static=True)
    maximum_tolerance_ratios: tuple[float, ...] = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class BlockAMRReferenceParityPlan(StrictModule, NonTrainableState):
    """Immutable reference values with provider/revision and mixed tolerances."""

    names: tuple[str, ...] = eqx.field(static=True)
    reference_values: tuple[Array, ...]
    provider_id: str = eqx.field(static=True)
    provider_revision: str = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_values: Mapping[str, ArrayLike],
        /,
        *,
        provider_id: str,
        provider_revision: str,
        absolute_tolerance: float,
        relative_tolerance: float,
    ):
        entries = tuple(
            sorted(
                (str(name), np.asarray(value)) for name, value in reference_values.items()
            )
        )
        provider = str(provider_id)
        revision = str(provider_revision)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not entries
            or any(
                not name or value.size == 0 or np.any(~np.isfinite(value))
                for name, value in entries
            )
            or not provider
            or not revision
            or not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
            or (absolute == 0.0 and relative == 0.0)
        ):
            raise ValueError("Block-AMR reference artifact or tolerances are invalid.")
        self.names = tuple(name for name, _ in entries)
        self.reference_values = tuple(jnp.asarray(value) for _, value in entries)
        self.provider_id = provider
        self.provider_revision = revision
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "block-amr-reference-parity-artifact",
                "provider": provider,
                "revision": revision,
                "values": {
                    name: array_tree_fingerprint(value) for name, value in entries
                },
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def compare(
        self,
        candidate_values: Mapping[str, ArrayLike],
        /,
    ) -> BlockAMRReferenceParityEvidence:
        candidate = {
            str(name): jnp.asarray(value) for name, value in candidate_values.items()
        }
        if set(candidate) != set(self.names):
            raise ValueError(
                "Reference parity candidates must match observable names exactly."
            )
        defects = []
        ratios = []
        for name, reference in zip(self.names, self.reference_values, strict=True):
            value = candidate[name]
            if value.shape != reference.shape:
                raise ValueError(f"Reference parity observable {name!r} changed shape.")
            difference = jnp.abs(value - reference.astype(value.dtype))
            tolerance = self.absolute_tolerance + self.relative_tolerance * jnp.maximum(
                jnp.abs(value), jnp.abs(reference.astype(value.dtype))
            )
            defects.append(float(jnp.max(difference)))
            ratios.append(float(jnp.max(difference / tolerance)))
        passed = all(ratio <= 1.0 for ratio in ratios)
        return BlockAMRReferenceParityEvidence(
            names=self.names,
            maximum_absolute_defects=tuple(defects),
            maximum_tolerance_ratios=tuple(ratios),
            passed=passed,
            artifact_id=self.artifact_id,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "block-amr-reference-parity-evidence",
                    "artifact": self.artifact_id,
                    "defects": defects,
                    "ratios": ratios,
                    "passed": passed,
                }
            ),
        )


__all__ = ["BlockAMRReferenceParityEvidence", "BlockAMRReferenceParityPlan"]

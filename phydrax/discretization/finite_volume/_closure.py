#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ConservativeFaceClosurePlan(StrictModule, NonTrainableState):
    """Static shared-face correction whose trainable parameters live in ``args``."""

    correction: Callable = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        correction: Callable,
        /,
        *,
        closure_id: str,
        consistency_tolerance: float = 1e-10,
        differentiability: str = "smooth_discrete",
    ):
        if not callable(correction):
            raise TypeError("correction must be callable.")
        identifier = str(closure_id)
        tolerance = float(consistency_tolerance)
        if (
            not identifier
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or differentiability
            not in ("smooth_discrete", "branchwise", "smooth_surrogate")
        ):
            raise ValueError("Conservative face closure metadata is invalid.")
        self.correction = correction
        self.consistency_tolerance = tolerance
        self.differentiability = differentiability
        self.closure_id = canonical_fingerprint(
            {
                "kind": "conservative-face-closure",
                "declared_id": identifier,
                "consistency_tolerance": tolerance,
                "differentiability": differentiability,
            }
        )

    def apply(
        self,
        system: Any,
        left: Array,
        right: Array,
        baseline_flux: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        names = tuple(system.component_names)
        if any(name.startswith("magnetic_") for name in names):
            raise ValueError(
                "Cell-face closures are unsupported for constrained MHD until they "
                "also provide compatible edge electromotive corrections."
            )
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        baseline = jnp.asarray(baseline_flux)
        correction = jnp.asarray(
            self.correction(system, left_, right_, baseline, int(axis), args)
        )
        if correction.shape != baseline.shape:
            raise ValueError("Face closure correction must match baseline flux shape.")
        correction = eqx.error_if(
            correction,
            jnp.any(~jnp.isfinite(correction)),
            "Face closure produced a nonfinite correction.",
        )
        equal = jnp.max(jnp.abs(left_ - right_), axis=-1) <= self.consistency_tolerance
        consistency_defect = jnp.max(
            jnp.where(equal[..., None], jnp.abs(correction), 0.0), initial=0.0
        )
        correction = eqx.error_if(
            correction,
            consistency_defect > self.consistency_tolerance,
            "Face closure violates equal-state consistency.",
        )
        return baseline + correction


__all__ = ["ConservativeFaceClosurePlan"]

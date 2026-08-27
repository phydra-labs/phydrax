#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FiniteElementPrecisionPolicy(StrictModule, NonTrainableState):
    """Explicit geometry/evaluation/accumulation precision for FE kernels."""

    geometry_dtype: str = eqx.field(static=True)
    evaluation_dtype: str = eqx.field(static=True)
    compensated_accumulation: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: str = "float64",
        evaluation_dtype: str = "float64",
        compensated_accumulation: bool = True,
    ):
        geometry = str(geometry_dtype)
        evaluation = str(evaluation_dtype)
        if jnp.dtype(geometry) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise ValueError("geometry_dtype must be float32 or float64.")
        if jnp.dtype(evaluation) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise ValueError("evaluation_dtype must be float32 or float64.")
        self.geometry_dtype = geometry
        self.evaluation_dtype = evaluation
        self.compensated_accumulation = bool(compensated_accumulation)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-precision-policy",
                "geometry_dtype": geometry,
                "evaluation_dtype": evaluation,
                "compensated_accumulation": bool(compensated_accumulation),
            }
        )


__all__ = ["FiniteElementPrecisionPolicy"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
    RealPrecisionDType,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


ParticleRealization: TypeAlias = Literal["dense_pairs", "cell_edge_list"]
ParticleAccumulation: TypeAlias = Literal["fast", "deterministic", "compensated"]


class ParticleExecutionPolicy(StrictModule, NonTrainableState):
    """Execution realization and reduction semantics for particle interactions."""

    realization: ParticleRealization = eqx.field(static=True)
    accumulation: ParticleAccumulation = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        realization: ParticleRealization = "dense_pairs",
        accumulation: ParticleAccumulation = "deterministic",
    ):
        if realization not in ("dense_pairs", "cell_edge_list"):
            raise ValueError("realization must be 'dense_pairs' or 'cell_edge_list'.")
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError(
                "accumulation must be 'fast', 'deterministic', or 'compensated'."
            )
        self.realization = realization
        self.accumulation = accumulation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "particle-execution-policy",
                "realization": realization,
                "accumulation": accumulation,
            }
        )


class ParticlePrecisionPolicy(StrictModule, NonTrainableState):
    """Coordinate, pair-evaluation, accumulation, and output precision."""

    geometry_dtype: RealPrecisionDType = eqx.field(static=True)
    evaluation_dtype: RealPrecisionDType = eqx.field(static=True)
    accumulation_dtype: RealPrecisionDType = eqx.field(static=True)
    certification_dtype: RealPrecisionDType = eqx.field(static=True)
    output_dtype: RealPrecisionDType = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: Any = "float64",
        evaluation_dtype: Any = "float64",
        accumulation_dtype: Any | None = None,
        certification_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        geometry = real_precision_dtype_name(geometry_dtype)
        evaluation = real_precision_dtype_name(evaluation_dtype)
        accumulation = real_precision_dtype_name(
            evaluation if accumulation_dtype is None else accumulation_dtype
        )
        certification = real_precision_dtype_name(
            accumulation if certification_dtype is None else certification_dtype
        )
        output = real_precision_dtype_name(
            evaluation if output_dtype is None else output_dtype
        )
        self.geometry_dtype = geometry
        self.evaluation_dtype = evaluation
        self.accumulation_dtype = accumulation
        self.certification_dtype = certification
        self.output_dtype = output
        self.policy_id = canonical_fingerprint(
            {
                "kind": "particle-precision-policy",
                "geometry_dtype": geometry,
                "evaluation_dtype": evaluation,
                "accumulation_dtype": accumulation,
                "certification_dtype": certification,
                "output_dtype": output,
            }
        )

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "particle",
            {
                "storage": self.evaluation_dtype,
                "basis": self.geometry_dtype,
                "compute": self.evaluation_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.certification_dtype,
                "output": self.output_dtype,
            },
        )

    def geometry(self, value: Any, /):
        return jnp.asarray(value, dtype=self.geometry_dtype)

    def evaluation(self, value: Any, /):
        return jnp.asarray(value, dtype=self.evaluation_dtype)

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(self.accumulation_dtype)

    def certification(self, value: Any, /):
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(self.certification_dtype)

    def output(self, value: Any, /):
        return jnp.asarray(value, dtype=self.output_dtype)

    def evidence(self) -> PrecisionEvidenceEnvelope:
        resolution = PrecisionResolution(
            self.request,
            "phydrax-particle",
            {
                "storage": self.evaluation_dtype,
                "basis": self.geometry_dtype,
                "compute": self.evaluation_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.certification_dtype,
                "output": self.output_dtype,
            },
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


__all__ = ["ParticleExecutionPolicy", "ParticlePrecisionPolicy"]

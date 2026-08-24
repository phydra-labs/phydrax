#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..geometry.complex import HypersurfaceKahlerGeometry, ProjectiveHypersurface
from ._calabi_yau import CalabiYauMetricResult


class CalabiYauMetricArtifact(StrictModule):
    """Frozen Ricci-flat metric candidate with replayable scientific evidence."""

    potential_model: Any
    normalization: Array
    objective_history: Array
    residual_history: Array
    positivity_history: Array
    hypersurface_id: str = eqx.field(static=True)
    projective_dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    precision: GeometryPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope
    precision_policy_id: str = eqx.field(static=True)
    schema_version: int = eqx.field(static=True)

    def __init__(
        self,
        potential_model: Any,
        normalization: ArrayLike,
        /,
        *,
        hypersurface_id: str,
        projective_dimension: int,
        degree: int,
        objective_history: ArrayLike,
        residual_history: ArrayLike,
        positivity_history: ArrayLike,
        precision: GeometryPrecisionPolicy,
        precision_evidence: PrecisionEvidenceEnvelope,
        precision_policy_id: str,
        schema_version: int = 2,
    ):
        self.potential_model = potential_model
        self.normalization = jnp.asarray(normalization)
        self.hypersurface_id = str(hypersurface_id)
        self.projective_dimension = int(projective_dimension)
        self.degree = int(degree)
        self.objective_history = jnp.asarray(objective_history)
        self.residual_history = jnp.asarray(residual_history)
        self.positivity_history = jnp.asarray(positivity_history)
        if not isinstance(precision, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy.")
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        if precision.policy_id != str(precision_policy_id):
            raise ValueError("Calabi-Yau artifact precision identity changed.")
        self.precision = precision
        self.precision_evidence = precision_evidence
        self.precision_policy_id = precision.policy_id
        self.schema_version = int(schema_version)

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "hypersurface_id": self.hypersurface_id,
            "projective_dimension": self.projective_dimension,
            "degree": self.degree,
            "precision_policy_id": self.precision_policy_id,
            "precision_evidence_id": self.precision_evidence.evidence_id,
            "normalization": float(self.normalization),
            "iteration_count": int(self.objective_history.shape[0]),
        }

    def evaluate(
        self,
        hypersurface: ProjectiveHypersurface,
        homogeneous_point: ArrayLike,
        /,
    ):
        if (
            hypersurface.hypersurface_id != self.hypersurface_id
            or hypersurface.projective_dimension != self.projective_dimension
            or hypersurface.degree != self.degree
        ):
            raise ValueError("Hypersurface metadata does not match the artifact.")
        return HypersurfaceKahlerGeometry(
            hypersurface,
            self.potential_model,
            normalization=self.normalization,
        ).evaluate(homogeneous_point)


def freeze_calabi_yau_result(
    result: CalabiYauMetricResult,
    hypersurface: ProjectiveHypersurface,
    /,
) -> CalabiYauMetricArtifact:
    if result.hypersurface_id != hypersurface.hypersurface_id:
        raise ValueError("Result and hypersurface identities do not match.")
    return CalabiYauMetricArtifact(
        result.potential_model,
        result.normalization,
        hypersurface_id=hypersurface.hypersurface_id,
        projective_dimension=hypersurface.projective_dimension,
        degree=hypersurface.degree,
        objective_history=result.objective_history,
        residual_history=result.residual_history,
        positivity_history=result.positivity_history,
        precision=result.precision,
        precision_evidence=result.precision_evidence,
        precision_policy_id=result.precision.policy_id,
        schema_version=2,
    )


__all__ = ["CalabiYauMetricArtifact", "freeze_calabi_yau_result"]

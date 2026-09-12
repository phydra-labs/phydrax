#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy.typing import ArrayLike, NDArray

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._runtime import ROMArtifact


@dataclass(frozen=True, slots=True)
class EmpiricalInterpolationPlan:
    maximum_condition: float = 1.0e10
    minimum_residual: float = 1.0e-12
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        condition = float(self.maximum_condition)
        residual = float(self.minimum_residual)
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("maximum_condition must be finite and exceed one.")
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("minimum_residual must be finite and nonnegative.")
        object.__setattr__(self, "maximum_condition", condition)
        object.__setattr__(self, "minimum_residual", residual)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "empirical-interpolation-plan",
                    "maximum_condition": condition,
                    "minimum_residual": residual,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class EmpiricalInterpolationArtifact:
    source_artifact_id: str
    node_indices: NDArray[np.integer]
    interpolation_matrix: NDArray[np.floating]
    reconstruction_matrix: NDArray[np.floating]
    condition_number: float
    maximum_reproduction_error: float
    plan_id: str
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = str(self.source_artifact_id).strip()
        plan = str(self.plan_id).strip()
        nodes = np.array(self.node_indices, dtype=np.int32, copy=True)
        interpolation = np.array(self.interpolation_matrix, dtype=float, copy=True)
        reconstruction = np.array(self.reconstruction_matrix, dtype=float, copy=True)
        condition = float(self.condition_number)
        error = float(self.maximum_reproduction_error)
        if not source or not plan:
            raise ValueError(
                "Empirical interpolation source and plan IDs must be non-empty."
            )
        if nodes.ndim != 1 or nodes.size == 0 or np.unique(nodes).size != nodes.size:
            raise ValueError(
                "Empirical interpolation nodes must be unique and non-empty."
            )
        if interpolation.shape != (nodes.size, nodes.size):
            raise ValueError("Interpolation matrix must be square on selected nodes.")
        if reconstruction.ndim != 2 or reconstruction.shape[1] != nodes.size:
            raise ValueError("Reconstruction matrix must end in the node axis.")
        if (
            np.any(nodes < 0)
            or np.any(nodes >= reconstruction.shape[0])
            or np.any(~np.isfinite(interpolation))
            or np.any(~np.isfinite(reconstruction))
            or not np.isfinite(condition)
            or condition < 1.0
            or not np.isfinite(error)
            or error < 0.0
        ):
            raise ValueError("Empirical interpolation artifact values are invalid.")
        nodes.setflags(write=False)
        interpolation.setflags(write=False)
        reconstruction.setflags(write=False)
        object.__setattr__(self, "source_artifact_id", source)
        object.__setattr__(self, "plan_id", plan)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "interpolation_matrix", interpolation)
        object.__setattr__(self, "reconstruction_matrix", reconstruction)
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(self, "maximum_reproduction_error", error)
        object.__setattr__(
            self,
            "artifact_id",
            canonical_fingerprint(
                {
                    "kind": "empirical-interpolation-artifact",
                    "source": source,
                    "plan": plan,
                    "content": array_tree_fingerprint(
                        {
                            "nodes": nodes,
                            "interpolation": interpolation,
                            "reconstruction": reconstruction,
                        }
                    )["sha256"],
                    "condition_number": condition,
                    "maximum_reproduction_error": error,
                }
            ),
        )

    def prepare(self) -> PreparedEmpiricalInterpolation:
        return PreparedEmpiricalInterpolation(self)


class PreparedEmpiricalInterpolation(StrictModule, NonTrainableState):
    node_indices: Array
    reconstruction_matrix: Array
    artifact_id: str = eqx.field(static=True)
    source_artifact_id: str = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    maximum_reproduction_error: float = eqx.field(static=True)

    def __init__(self, artifact: EmpiricalInterpolationArtifact, /):
        if not isinstance(artifact, EmpiricalInterpolationArtifact):
            raise TypeError("artifact must be EmpiricalInterpolationArtifact.")
        self.node_indices = jnp.asarray(artifact.node_indices, dtype=jnp.int32)
        self.reconstruction_matrix = jnp.asarray(artifact.reconstruction_matrix)
        self.artifact_id = artifact.artifact_id
        self.source_artifact_id = artifact.source_artifact_id
        self.condition_number = artifact.condition_number
        self.maximum_reproduction_error = artifact.maximum_reproduction_error

    def interpolate(self, node_values: ArrayLike, /) -> Array:
        values = jnp.asarray(node_values)
        if values.shape[-1:] != (int(self.node_indices.size),):
            raise ValueError("Node values must end in the empirical node axis.")
        return contract("ij,...j->...i", self.reconstruction_matrix, values)


def prepare_empirical_interpolation(
    artifact: ROMArtifact,
    plan: EmpiricalInterpolationPlan | None = None,
    /,
) -> EmpiricalInterpolationArtifact:
    if not isinstance(artifact, ROMArtifact):
        raise TypeError("artifact must be ROMArtifact.")
    policy = EmpiricalInterpolationPlan() if plan is None else plan
    if not isinstance(policy, EmpiricalInterpolationPlan):
        raise TypeError("plan must be EmpiricalInterpolationPlan or None.")
    basis = np.asarray(artifact.basis, dtype=float)
    rank = int(basis.shape[1])
    nodes = [int(np.argmax(np.abs(basis[:, 0])))]
    if abs(basis[nodes[0], 0]) <= policy.minimum_residual:
        raise ValueError("First empirical basis vector has no resolvable node.")
    for column in range(1, rank):
        selected_basis = basis[np.asarray(nodes), :column]
        selected_value = basis[np.asarray(nodes), column]
        coefficients = np.linalg.solve(selected_basis, selected_value)
        residual = basis[:, column] - basis[:, :column] @ coefficients
        node = int(np.argmax(np.abs(residual)))
        if node in nodes or abs(residual[node]) <= policy.minimum_residual:
            raise ValueError("Empirical interpolation lost rank during node selection.")
        nodes.append(node)
    node_indices = np.asarray(nodes, dtype=np.int32)
    interpolation = basis[node_indices, :]
    condition = float(np.linalg.cond(interpolation))
    if not np.isfinite(condition) or condition > policy.maximum_condition:
        raise ValueError("Empirical interpolation matrix exceeds the condition limit.")
    reconstruction = np.linalg.solve(interpolation.T, basis.T).T
    reproduced = reconstruction @ interpolation
    error = float(np.max(np.abs(reproduced - basis)))
    return EmpiricalInterpolationArtifact(
        artifact.artifact_id,
        node_indices,
        interpolation,
        reconstruction,
        condition,
        error,
        policy.plan_id,
    )


__all__ = [
    "EmpiricalInterpolationArtifact",
    "EmpiricalInterpolationPlan",
    "PreparedEmpiricalInterpolation",
    "prepare_empirical_interpolation",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import NumericRevision
from ..linalg import DualSpace
from ._affine import (
    AffineLinearROMEvaluation,
    AffineLinearROMProblem,
    PreparedAffineLinearROM,
)
from ._production import ROMResourcePolicy


class CertificationStatus(IntEnum):
    SUCCESS = 0
    BASE_EVALUATION_INVALID = 1
    STABILITY_OUT_OF_SCOPE = 2
    NONFINITE = 3


class StabilityBoundEvaluation(StrictModule, NonTrainableState):
    bound: Array
    valid: Array
    status: Array
    certificate_id: str = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    error_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)


class AbstractStabilityBoundEvaluator(StrictModule, NonTrainableState):
    family_id: eqx.AbstractVar[str]
    error_space_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    certificate_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def __call__(self, inputs: PyTree[Array], /) -> StabilityBoundEvaluation:
        raise NotImplementedError


class ArrayAffineStabilityBound(AbstractStabilityBoundEvaluator):
    """Certified positive affine lower bound over one parameter box."""

    weights: Array
    offset: Array
    lower: Array
    upper: Array
    input_size: int = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    error_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        offset: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        family_id: str,
        error_space_id: str,
        support_id: str,
        evidence_id: str,
    ):
        weights_ = jnp.asarray(weights)
        offset_ = jnp.asarray(offset, dtype=weights_.dtype)
        lower_ = jnp.asarray(lower, dtype=weights_.dtype)
        upper_ = jnp.asarray(upper, dtype=weights_.dtype)
        if weights_.ndim != 1 or offset_.shape != ():
            raise ValueError("Stability weights must be a vector and offset a scalar.")
        if lower_.shape != weights_.shape or upper_.shape != weights_.shape:
            raise ValueError("Stability support bounds must match the input width.")
        host = tuple(np.asarray(value) for value in (weights_, offset_, lower_, upper_))
        if any(not np.all(np.isfinite(value)) for value in host):
            raise ValueError("Stability-bound values must be finite.")
        if np.any(host[2] > host[3]):
            raise ValueError("Stability lower support cannot exceed upper support.")
        minimum = float(
            host[1]
            + np.sum(np.where(host[0] >= 0.0, host[0] * host[2], host[0] * host[3]))
        )
        if minimum <= 0.0:
            raise ValueError("Certified affine stability bound must stay positive.")
        family = str(family_id)
        error_space = str(error_space_id)
        support = str(support_id)
        evidence = str(evidence_id)
        if not family or not error_space or not support or not evidence:
            raise ValueError("Stability certificate identities must be non-empty.")
        self.weights = weights_
        self.offset = offset_
        self.lower = lower_
        self.upper = upper_
        self.input_size = weights_.size
        self.family_id = family
        self.error_space_id = error_space
        self.support_id = support
        self.evidence_id = evidence
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "array-affine-stability-lower-bound",
                "family": family,
                "error_space": error_space,
                "support": support,
                "evidence": evidence,
                "content": array_tree_fingerprint(host)["sha256"],
            }
        )

    def __call__(self, inputs: PyTree[Array], /) -> StabilityBoundEvaluation:
        value = jnp.asarray(inputs)
        if value.shape != (self.input_size,):
            raise ValueError(
                f"Stability-bound inputs must have shape {(self.input_size,)}."
            )
        finite = jnp.all(jnp.isfinite(value))
        inside = jnp.all((value >= self.lower) & (value <= self.upper))
        bound = self.offset + self.weights @ value
        valid = finite & inside & jnp.isfinite(bound) & (bound > 0.0)
        return StabilityBoundEvaluation(
            jnp.where(valid, bound, jnp.nan),
            valid,
            jnp.where(valid, 0, 1).astype(jnp.int32),
            self.certificate_id,
            self.family_id,
            self.error_space_id,
            self.support_id,
        )


class ResidualDualNormArtifact(StrictModule, NonTrainableState):
    """Factored exact dual norm over affine residual atoms."""

    factor: Array
    gram: Array
    numerical_rank: Array
    reconstruction_defect: Array
    roundoff_bound: Array
    numeric_revision: NumericRevision
    family_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    residual_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    rhs_terms: int = eqx.field(static=True)
    operator_terms: int = eqx.field(static=True)
    lift_terms: int = eqx.field(static=True)
    reduced_rank: int = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    @property
    def atom_count(self) -> int:
        return self.factor.shape[1]

    def coefficients(
        self,
        evaluation: AffineLinearROMEvaluation,
        /,
    ) -> Array:
        coefficient = evaluation.coefficients
        rhs = coefficient.right_hand_side
        lift = (
            jnp.reshape(
                coefficient.operator[:, None] * coefficient.lift[None, :],
                (-1,),
            )
            if self.lift_terms
            else jnp.zeros((0,), dtype=rhs.dtype)
        )
        state = jnp.reshape(
            coefficient.operator[:, None] * evaluation.reduced_state[None, :],
            (-1,),
        )
        result = jnp.concatenate((rhs, lift, state))
        if result.shape != (self.atom_count,):
            raise ValueError("Residual coefficient layout does not match its artifact.")
        return result

    def norm(self, evaluation: AffineLinearROMEvaluation, /) -> Array:
        values = self.factor @ self.coefficients(evaluation)
        return jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(values, values)), 0.0))


class AffineROMCertification(StrictModule, NonTrainableState):
    residual_norm: ResidualDualNormArtifact
    stability: AbstractStabilityBoundEvaluator
    certification_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual_norm: ResidualDualNormArtifact,
        stability: AbstractStabilityBoundEvaluator,
        /,
    ):
        if not isinstance(residual_norm, ResidualDualNormArtifact):
            raise TypeError("residual_norm must be a ResidualDualNormArtifact.")
        if not isinstance(stability, AbstractStabilityBoundEvaluator):
            raise TypeError("stability must be an AbstractStabilityBoundEvaluator.")
        if stability.family_id != residual_norm.family_id:
            raise ValueError(
                "Stability and residual artifacts must bind one operator family."
            )
        if stability.error_space_id != residual_norm.residual_space_id:
            raise ValueError(
                "Stability error space must match the residual dual-norm space."
            )
        if stability.support_id != residual_norm.support_id:
            raise ValueError("Stability and residual support identities must match.")
        self.residual_norm = residual_norm
        self.stability = stability
        self.certification_id = canonical_fingerprint(
            {
                "kind": "affine-rom-certification",
                "residual_norm": residual_norm.artifact_id,
                "stability": stability.certificate_id,
            }
        )


class CertifiedAffineROMEvaluation(StrictModule, NonTrainableState):
    evaluation: AffineLinearROMEvaluation
    residual_dual_norm: Array
    stability_lower_bound: Array
    absolute_state_error_bound: Array
    valid: Array
    status: Array
    certification_id: str = eqx.field(static=True)


def prepare_residual_dual_norm(
    problem: AffineLinearROMProblem,
    model: PreparedAffineLinearROM,
    /,
    *,
    resource_policy: ROMResourcePolicy | None = None,
) -> ResidualDualNormArtifact:
    """Prepare the exact full residual dual norm for one affine ROM family."""
    if not isinstance(problem, AffineLinearROMProblem):
        raise TypeError("problem must be an AffineLinearROMProblem.")
    if not isinstance(model, PreparedAffineLinearROM):
        raise TypeError("model must be a PreparedAffineLinearROM.")
    if problem.family_id != model.family_id:
        raise ValueError("Problem and prepared model must share one affine family.")
    residual_space = DualSpace(problem.reduction.test.full_space)
    atoms = list(problem.right_hand_side_terms)
    for operator in problem.operator_terms:
        for lift in problem.lift_terms:
            atoms.append(
                jax.tree.map(
                    jnp.negative,
                    residual_space.validate(operator.mv(lift)),
                )
            )
    trial_coordinates = jnp.eye(
        problem.reduction.rank,
        dtype=model.reduced_operator_terms.dtype,
    )
    trial_vectors = problem.reduction.trial.prolongation.mv_block(trial_coordinates)
    for operator in problem.operator_terms:
        images = operator.mv_block(trial_vectors)
        atoms.extend(
            residual_space.unflatten(-images[:, column])
            for column in range(problem.reduction.rank)
        )
    policy = ROMResourcePolicy() if resource_policy is None else resource_policy
    if not isinstance(policy, ROMResourcePolicy):
        raise TypeError("resource_policy must be a ROMResourcePolicy or None.")
    workspace = 16 * len(atoms) * len(atoms)
    if not policy.admit(
        full_dimension=problem.reduction.test.full_space.size,
        reduced_dimension=problem.reduction.trial_rank,
        affine_terms=len(problem.operator_terms),
        residual_atoms=len(atoms),
        workspace_bytes=max(workspace, 1),
    ):
        raise ValueError("Residual norm preparation exceeds the ROM resource policy.")
    gram = jnp.stack(
        tuple(
            jnp.stack(tuple(residual_space.inner(left, right) for right in atoms))
            for left in atoms
        )
    )
    gram = 0.5 * (gram + jnp.conj(gram.T))
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    largest = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    cutoff = jnp.finfo(eigenvalues.dtype).eps * len(atoms) * largest
    positive = eigenvalues > cutoff
    clipped = jnp.where(positive, eigenvalues, 0.0)
    factor = jnp.sqrt(clipped)[:, None] * jnp.conj(eigenvectors.T)
    reconstructed = jnp.conj(factor.T) @ factor
    defect = jnp.max(jnp.abs(reconstructed - gram))
    roundoff = (
        jnp.finfo(gram.real.dtype).eps
        * len(atoms)
        * jnp.maximum(jnp.linalg.norm(gram), 1.0)
    )
    content_digest = array_tree_fingerprint({"factor": factor, "gram": gram})["sha256"]
    revision = NumericRevision(content_digest, label="affine-residual-dual-norm")
    artifact_id = canonical_fingerprint(
        {
            "kind": "affine-residual-dual-norm",
            "family": problem.family_id,
            "reduction": problem.reduction.reduction_id,
            "space": problem.reduction.test.full_space.space_id,
            "support": problem.reduction.support_id,
            "measure": problem.reduction.test_measure_id,
            "geometry": problem.reduction.geometry_id,
            "rhs_terms": len(problem.right_hand_side_terms),
            "operator_terms": len(problem.operator_terms),
            "lift_terms": len(problem.lift_terms),
            "rank": problem.reduction.rank,
            "revision": revision.revision_id,
        }
    )
    return ResidualDualNormArtifact(
        factor,
        gram,
        jnp.sum(positive).astype(jnp.int32),
        defect,
        roundoff,
        revision,
        problem.family_id,
        problem.reduction.reduction_id,
        problem.reduction.test.full_space.space_id,
        problem.reduction.support_id,
        problem.reduction.test_measure_id,
        problem.reduction.geometry_id,
        len(problem.right_hand_side_terms),
        len(problem.operator_terms),
        len(problem.lift_terms),
        problem.reduction.rank,
        artifact_id,
    )


def certify_affine_rom_evaluation(
    model: PreparedAffineLinearROM,
    evaluation: AffineLinearROMEvaluation,
    inputs: PyTree[Array],
    certification: AffineROMCertification,
    /,
) -> CertifiedAffineROMEvaluation:
    if not isinstance(model, PreparedAffineLinearROM):
        raise TypeError("model must be a PreparedAffineLinearROM.")
    if not isinstance(evaluation, AffineLinearROMEvaluation):
        raise TypeError("evaluation must be an AffineLinearROMEvaluation.")
    if not isinstance(certification, AffineROMCertification):
        raise TypeError("certification must be an AffineROMCertification.")
    if evaluation.model_id != model.model_id:
        raise ValueError("Evaluation does not belong to the supplied ROM model.")
    residual_artifact = certification.residual_norm
    if (
        residual_artifact.family_id != model.family_id
        or residual_artifact.reduction_id != model.reduction.reduction_id
        or residual_artifact.support_id != model.reduction.support_id
        or residual_artifact.measure_id != model.reduction.test_measure_id
        or residual_artifact.geometry_id != model.reduction.geometry_id
    ):
        raise ValueError("Certification identities do not match the prepared ROM.")
    stability = certification.stability(inputs)
    residual = residual_artifact.norm(evaluation)
    valid = evaluation.valid & stability.valid & jnp.isfinite(residual)
    bound = jnp.where(valid, residual / stability.bound, jnp.nan)
    status = jnp.where(
        ~evaluation.valid,
        int(CertificationStatus.BASE_EVALUATION_INVALID),
        jnp.where(
            ~stability.valid,
            int(CertificationStatus.STABILITY_OUT_OF_SCOPE),
            jnp.where(
                jnp.isfinite(residual),
                int(CertificationStatus.SUCCESS),
                int(CertificationStatus.NONFINITE),
            ),
        ),
    ).astype(jnp.int32)
    return CertifiedAffineROMEvaluation(
        evaluation,
        residual,
        stability.bound,
        bound,
        valid,
        status,
        certification.certification_id,
    )


__all__ = [
    "AbstractStabilityBoundEvaluator",
    "AffineROMCertification",
    "ArrayAffineStabilityBound",
    "CertificationStatus",
    "CertifiedAffineROMEvaluation",
    "ResidualDualNormArtifact",
    "StabilityBoundEvaluation",
    "certify_affine_rom_evaluation",
    "prepare_residual_dual_norm",
]

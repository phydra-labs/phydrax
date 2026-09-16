#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import NumericRevision
from ..linalg import (
    AbstractLinearOperator,
    DenseLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve,
)
from ._reduction import TrialTestReduction


class RectangularLinearROMProblem(StrictModule, NonTrainableState):
    reduction: TrialTestReduction
    operator: AbstractLinearOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: TrialTestReduction,
        operator: AbstractLinearOperator,
        /,
        *,
        problem_id: str,
    ):
        if not isinstance(reduction, TrialTestReduction):
            raise TypeError("reduction must be TrialTestReduction.")
        if reduction.test_rank < reduction.trial_rank:
            raise ValueError("Rectangular reduction requires test rank >= trial rank.")
        reduction.validate_operator(operator)
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.reduction = reduction
        self.operator = operator
        self.problem_id = identifier

    def solve(
        self,
        right_hand_side,
        /,
        *,
        policy: LinearSolvePolicy | None = None,
    ):
        reduced_matrix = self.reduction.project_operator(self.operator)
        reduced_rhs = self.reduction.test.reduced_space.flatten(
            self.reduction.project_covector(right_hand_side)
        )
        operator = DenseLinearOperator(
            reduced_matrix,
            source=self.reduction.trial.reduced_space,
            target=self.reduction.test.reduced_space,
            operator_id=f"{self.problem_id}:rectangular-operator",
        )
        return solve(
            LeastSquaresProblem(
                operator,
                problem_id=f"{self.problem_id}:least-squares",
            ),
            reduced_rhs,
            policy=policy,
        )


class ReducedInfSupEvidence(StrictModule, NonTrainableState):
    singular_values: Array
    lower_bound: Array
    valid: Array
    velocity_basis_id: str = eqx.field(static=True)
    pressure_basis_id: str = eqx.field(static=True)
    divergence_operator_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduced_divergence: ArrayLike,
        /,
        *,
        velocity_basis_id: str,
        pressure_basis_id: str,
        divergence_operator_id: str,
        minimum_accepted: float,
    ):
        matrix = jnp.asarray(reduced_divergence)
        if matrix.ndim != 2:
            raise ValueError("reduced_divergence must be a matrix.")
        singular = jnp.linalg.svd(matrix, compute_uv=False)
        lower = jnp.min(singular)
        accepted = float(minimum_accepted)
        if not np.isfinite(accepted) or accepted <= 0.0:
            raise ValueError("minimum_accepted must be finite and positive.")
        identifiers = tuple(
            str(value)
            for value in (
                velocity_basis_id,
                pressure_basis_id,
                divergence_operator_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Inf-sup identities must be non-empty.")
        self.singular_values = singular
        self.lower_bound = lower
        self.valid = jnp.isfinite(lower) & (lower >= accepted)
        self.velocity_basis_id = identifiers[0]
        self.pressure_basis_id = identifiers[1]
        self.divergence_operator_id = identifiers[2]
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "reduced-inf-sup-evidence",
                "velocity_basis": identifiers[0],
                "pressure_basis": identifiers[1],
                "divergence": identifiers[2],
                "minimum_accepted": accepted,
                "matrix": array_tree_fingerprint(matrix)["sha256"],
            }
        )


class DescriptorStructureEvidence(StrictModule, NonTrainableState):
    finite_dimension: int = eqx.field(static=True)
    algebraic_dimension: int = eqx.field(static=True)
    index: int = eqx.field(static=True)
    regular: bool = eqx.field(static=True)
    impulse_free: bool = eqx.field(static=True)
    pencil_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        finite_dimension: int,
        algebraic_dimension: int,
        index: int,
        regular: bool,
        impulse_free: bool,
        pencil_id: str,
    ):
        finite = int(finite_dimension)
        algebraic = int(algebraic_dimension)
        index_ = int(index)
        pencil = str(pencil_id)
        if finite <= 0 or algebraic < 0 or index_ < 0 or not pencil:
            raise ValueError("Descriptor structure values are invalid.")
        self.finite_dimension = finite
        self.algebraic_dimension = algebraic
        self.index = index_
        self.regular = bool(regular)
        self.impulse_free = bool(impulse_free)
        self.pencil_id = pencil
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "descriptor-structure-evidence",
                "finite_dimension": finite,
                "algebraic_dimension": algebraic,
                "index": index_,
                "regular": self.regular,
                "impulse_free": self.impulse_free,
                "pencil": pencil,
            }
        )

    @property
    def index_one_supported(self) -> bool:
        return self.regular and self.impulse_free and self.index <= 1


class IndexOneDescriptorReduction(StrictModule, NonTrainableState):
    """Schur reduction of a semi-explicit index-one linear descriptor system."""

    differential_matrix: Array
    coupling_da: Array
    coupling_ad: Array
    algebraic_matrix: Array
    evidence: DescriptorStructureEvidence
    numeric_revision: NumericRevision
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        differential_matrix: ArrayLike,
        coupling_da: ArrayLike,
        coupling_ad: ArrayLike,
        algebraic_matrix: ArrayLike,
        evidence: DescriptorStructureEvidence,
        /,
    ):
        if (
            not isinstance(evidence, DescriptorStructureEvidence)
            or not evidence.index_one_supported
        ):
            raise ValueError(
                "Descriptor evidence must establish a regular impulse-free index-one system."
            )
        add = jnp.asarray(differential_matrix)
        ada = jnp.asarray(coupling_da)
        aad = jnp.asarray(coupling_ad)
        aaa = jnp.asarray(algebraic_matrix)
        nd = evidence.finite_dimension
        na = evidence.algebraic_dimension
        if (
            add.shape != (nd, nd)
            or ada.shape != (nd, na)
            or aad.shape != (na, nd)
            or aaa.shape != (na, na)
        ):
            raise ValueError("Descriptor blocks do not match the structure evidence.")
        singular = np.linalg.svd(np.asarray(aaa), compute_uv=False)
        if singular.size and singular[-1] <= np.finfo(singular.dtype).eps * max(
            float(singular[0]), 1.0
        ):
            raise ValueError("Index-one algebraic block must be nonsingular.")
        content = {"add": add, "ada": ada, "aad": aad, "aaa": aaa}
        digest = array_tree_fingerprint(content)["sha256"]
        self.differential_matrix = add
        self.coupling_da = ada
        self.coupling_ad = aad
        self.algebraic_matrix = aaa
        self.evidence = evidence
        self.numeric_revision = NumericRevision(digest, label="index-one-descriptor")
        self.model_id = canonical_fingerprint(
            {
                "kind": "index-one-descriptor-reduction",
                "structure": evidence.evidence_id,
                "revision": self.numeric_revision.revision_id,
            }
        )

    def reduced_differential_matrix(self) -> Array:
        solved = jnp.linalg.solve(self.algebraic_matrix, self.coupling_ad)
        return self.differential_matrix - self.coupling_da @ solved

    def reconstruct_algebraic(
        self, differential_state: ArrayLike, forcing: ArrayLike | None = None
    ) -> Array:
        state = jnp.asarray(differential_state)
        source = (
            jnp.zeros((self.evidence.algebraic_dimension,), dtype=state.dtype)
            if forcing is None
            else jnp.asarray(forcing)
        )
        return jnp.linalg.solve(
            self.algebraic_matrix,
            source - self.coupling_ad @ state,
        )


__all__ = [
    "DescriptorStructureEvidence",
    "IndexOneDescriptorReduction",
    "RectangularLinearROMProblem",
    "ReducedInfSupEvidence",
]

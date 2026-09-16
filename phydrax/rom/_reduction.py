#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    ConstraintMap,
    DenseLinearOperator,
    DualSpace,
)
from ._basis import ReducedBasisArtifact


class TrialTestReduction(StrictModule, NonTrainableState):
    """Fixed-reference square or rectangular Petrov–Galerkin maps."""

    trial: ConstraintMap
    test: ConstraintMap
    trial_basis: ReducedBasisArtifact
    test_basis: ReducedBasisArtifact
    trial_state_contract_id: str = eqx.field(static=True)
    test_state_contract_id: str = eqx.field(static=True)
    trial_measure_id: str = eqx.field(static=True)
    test_measure_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)

    def __init__(
        self,
        trial: ConstraintMap,
        test: ConstraintMap,
        trial_basis: ReducedBasisArtifact,
        test_basis: ReducedBasisArtifact,
        /,
    ):
        if not isinstance(trial, ConstraintMap) or not isinstance(test, ConstraintMap):
            raise TypeError("trial and test must be ConstraintMap values.")
        if not isinstance(trial_basis, ReducedBasisArtifact) or not isinstance(
            test_basis, ReducedBasisArtifact
        ):
            raise TypeError(
                "trial_basis and test_basis must be ReducedBasisArtifact values."
            )
        if trial_basis.role != "state" or test_basis.role != "state":
            raise ValueError("Trial and test bases must use role='state'.")
        if not trial.full_space.compatible(trial_basis.subspace.space):
            raise ValueError("Trial constraint full space must match its basis space.")
        if not test.full_space.compatible(test_basis.subspace.space):
            raise ValueError("Test constraint full space must match its basis space.")
        if trial_basis.support_id != test_basis.support_id:
            raise ValueError("Trial and test bases must share physical support.")
        if trial_basis.geometry_id != test_basis.geometry_id:
            raise ValueError("Trial and test bases must share reference geometry.")
        self.trial = trial
        self.test = test
        self.trial_basis = trial_basis
        self.test_basis = test_basis
        self.trial_state_contract_id = trial_basis.state_contract_id
        self.test_state_contract_id = test_basis.state_contract_id
        self.trial_measure_id = trial_basis.measure_id
        self.test_measure_id = test_basis.measure_id
        self.support_id = trial_basis.support_id
        self.geometry_id = trial_basis.geometry_id
        self.reduction_id = canonical_fingerprint(
            {
                "kind": "trial-test-reduction",
                "trial": trial.constraint_id,
                "test": test.constraint_id,
                "trial_basis": trial_basis.artifact_id,
                "test_basis": test_basis.artifact_id,
                "trial_state_contract": self.trial_state_contract_id,
                "test_state_contract": self.test_state_contract_id,
                "trial_measure": self.trial_measure_id,
                "test_measure": self.test_measure_id,
                "support": self.support_id,
                "geometry": self.geometry_id,
            }
        )

    @property
    def rank(self) -> int:
        return self.trial.reduced_space.size

    @property
    def trial_rank(self) -> int:
        return self.trial.reduced_space.size

    @property
    def test_rank(self) -> int:
        return self.test.reduced_space.size

    @property
    def square(self) -> bool:
        return self.trial_rank == self.test_rank

    def validate_operator(self, operator: AbstractLinearOperator, /) -> None:
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("Affine ROM operator terms must be unbatched.")
        if not operator.source.compatible(self.trial.full_space):
            raise ValueError("Operator source must match the trial full space.")
        if not operator.target.compatible(DualSpace(self.test.full_space)):
            raise ValueError("Operator target must be the dual of the test full space.")

    def project_operator(self, operator: AbstractLinearOperator, /) -> Array:
        self.validate_operator(operator)
        trial_coordinates = jnp.eye(
            self.trial_rank,
            dtype=self.trial_basis.basis_matrix.dtype,
        )
        trial_vectors = self.trial.prolongation.mv_block(trial_coordinates)
        residuals = operator.mv_block(trial_vectors)
        return self.test.dual_pullback.mv_block(residuals)

    def project_covector(self, covector, /):
        return self.test.pullback_dual(DualSpace(self.test.full_space).validate(covector))

    def project_lift_action(self, operator: AbstractLinearOperator, lift, /):
        self.validate_operator(operator)
        value = self.trial.full_space.validate(lift)
        return self.test.pullback_dual(operator.mv(value))

    def reconstruct(self, reduced, lift, /):
        return self.trial.expand(reduced, lift)


def trial_test_reduction_from_bases(
    trial_basis: ReducedBasisArtifact,
    test_basis: ReducedBasisArtifact | None = None,
    /,
) -> TrialTestReduction:
    """Construct fixed-reference trial/test maps from physical basis artifacts."""
    if not isinstance(trial_basis, ReducedBasisArtifact):
        raise TypeError("trial_basis must be a ReducedBasisArtifact.")
    resolved_test = trial_basis if test_basis is None else test_basis
    if not isinstance(resolved_test, ReducedBasisArtifact):
        raise TypeError("test_basis must be a ReducedBasisArtifact or None.")

    def constraint(artifact: ReducedBasisArtifact, kind: str) -> ConstraintMap:
        reduced = ArraySpace(
            (artifact.rank,),
            dtype=artifact.basis_matrix.dtype,
            space_id=f"{artifact.artifact_id}:{kind}-coordinates",
        )
        prolongation = DenseLinearOperator(
            artifact.basis_matrix,
            source=reduced,
            target=artifact.subspace.space,
            operator_id=f"{artifact.artifact_id}:{kind}-prolongation",
        )
        return ConstraintMap(
            artifact.subspace.space,
            reduced,
            prolongation,
            constraint_id=f"{artifact.artifact_id}:{kind}-constraint",
        )

    return TrialTestReduction(
        constraint(trial_basis, "trial"),
        constraint(resolved_test, "test"),
        trial_basis,
        resolved_test,
    )


__all__ = ["TrialTestReduction", "trial_test_reduction_from_bases"]

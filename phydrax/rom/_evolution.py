#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import DAEStructure, DifferentialAlgebraicSystem
from ..lifecycle import NumericRevision
from ..linalg import AbstractLinearOperator, DualSpace
from ._reduction import TrialTestReduction


class AffineEvolutionROMProblem(StrictModule, NonTrainableState):
    reduction: TrialTestReduction
    mass_terms: tuple[AbstractLinearOperator, ...]
    operator_terms: tuple[AbstractLinearOperator, ...]
    right_hand_side_terms: tuple[PyTree[Array], ...]
    lift_terms: tuple[PyTree[Array], ...]
    mass_term_ids: tuple[str, ...] = eqx.field(static=True)
    operator_term_ids: tuple[str, ...] = eqx.field(static=True)
    right_hand_side_term_ids: tuple[str, ...] = eqx.field(static=True)
    lift_term_ids: tuple[str, ...] = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: TrialTestReduction,
        mass_terms: Sequence[AbstractLinearOperator],
        operator_terms: Sequence[AbstractLinearOperator],
        right_hand_side_terms: Sequence[PyTree[Array]],
        /,
        *,
        mass_term_ids: Sequence[str],
        operator_term_ids: Sequence[str],
        right_hand_side_term_ids: Sequence[str],
        lift_terms: Sequence[PyTree[Array]] = (),
        lift_term_ids: Sequence[str] = (),
        source_artifact_ids: Sequence[str],
    ):
        if not isinstance(reduction, TrialTestReduction) or not reduction.square:
            raise ValueError("Affine evolution currently requires a square reduction.")
        masses = tuple(mass_terms)
        operators = tuple(operator_terms)
        rhs = tuple(right_hand_side_terms)
        lifts = tuple(lift_terms)
        mass_ids = tuple(str(value) for value in mass_term_ids)
        operator_ids = tuple(str(value) for value in operator_term_ids)
        rhs_ids = tuple(str(value) for value in right_hand_side_term_ids)
        lift_ids = tuple(str(value) for value in lift_term_ids)
        if (
            not masses
            or not operators
            or not rhs
            or len(masses) != len(mass_ids)
            or len(operators) != len(operator_ids)
            or len(rhs) != len(rhs_ids)
            or len(lifts) != len(lift_ids)
        ):
            raise ValueError(
                "Affine evolution terms and IDs must be non-empty and aligned."
            )
        for term in (*masses, *operators):
            reduction.validate_operator(term)
        residual_space = DualSpace(reduction.test.full_space)
        rhs = tuple(residual_space.validate(value) for value in rhs)
        lifts = tuple(reduction.trial.full_space.validate(value) for value in lifts)
        sources = tuple(str(value) for value in source_artifact_ids)
        if not sources or any(not value for value in sources):
            raise ValueError("source_artifact_ids must be non-empty.")
        self.reduction = reduction
        self.mass_terms = masses
        self.operator_terms = operators
        self.right_hand_side_terms = rhs
        self.lift_terms = lifts
        self.mass_term_ids = mass_ids
        self.operator_term_ids = operator_ids
        self.right_hand_side_term_ids = rhs_ids
        self.lift_term_ids = lift_ids
        self.family_id = canonical_fingerprint(
            {
                "kind": "affine-evolution-rom-family",
                "reduction": reduction.reduction_id,
                "mass": list(mass_ids),
                "operator": list(operator_ids),
                "rhs": list(rhs_ids),
                "lift": list(lift_ids),
                "sources": list(sources),
            }
        )


class PreparedAffineEvolutionROM(StrictModule, NonTrainableState):
    reduction: TrialTestReduction
    reduced_mass_terms: Array
    reduced_operator_terms: Array
    reduced_right_hand_side_terms: Array
    reduced_mass_lift_terms: Array
    reduced_operator_lift_terms: Array
    lift_coordinates: Array
    numeric_revision: NumericRevision
    family_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def bind(
        self,
        mass_coefficients: ArrayLike,
        operator_coefficients: ArrayLike,
        right_hand_side_coefficients: ArrayLike,
        /,
        *,
        lift_coefficients: ArrayLike | None = None,
        lift_rate_coefficients: ArrayLike | None = None,
    ) -> DifferentialAlgebraicSystem:
        mass_values = jnp.asarray(mass_coefficients)
        operator_values = jnp.asarray(operator_coefficients)
        rhs_values = jnp.asarray(right_hand_side_coefficients)
        lift_values = (
            jnp.zeros((self.lift_coordinates.shape[0],), dtype=mass_values.dtype)
            if lift_coefficients is None
            else jnp.asarray(lift_coefficients)
        )
        lift_rates = (
            jnp.zeros_like(lift_values)
            if lift_rate_coefficients is None
            else jnp.asarray(lift_rate_coefficients)
        )
        expected = (
            (self.reduced_mass_terms.shape[0],),
            (self.reduced_operator_terms.shape[0],),
            (self.reduced_right_hand_side_terms.shape[0],),
            (self.lift_coordinates.shape[0],),
        )
        actual = (
            mass_values.shape,
            operator_values.shape,
            rhs_values.shape,
            lift_values.shape,
        )
        if actual != expected or lift_rates.shape != expected[3]:
            raise ValueError(
                "Affine evolution coefficient shapes do not match the model."
            )
        mass = contract("q,qij->ij", mass_values, self.reduced_mass_terms)
        operator = contract("q,qij->ij", operator_values, self.reduced_operator_terms)
        rhs = contract("q,qi->i", rhs_values, self.reduced_right_hand_side_terms)
        if self.lift_coordinates.shape[0]:
            rhs = rhs - contract(
                "q,p,qpi->i",
                operator_values,
                lift_values,
                self.reduced_operator_lift_terms,
            )
            rhs = rhs - contract(
                "q,p,qpi->i",
                mass_values,
                lift_rates,
                self.reduced_mass_lift_terms,
            )

        def residual(time, state, state_rate, args):
            del time, args
            return mass @ state_rate + operator @ state - rhs

        return DifferentialAlgebraicSystem(
            residual,
            state_shape=(self.reduction.trial_rank,),
            structure=DAEStructure(("differential",) * self.reduction.trial_rank),
            system_id=canonical_fingerprint(
                {
                    "kind": "bound-affine-evolution-rom",
                    "model": self.model_id,
                    "coefficients": array_tree_fingerprint(
                        {
                            "mass": mass_values,
                            "operator": operator_values,
                            "rhs": rhs_values,
                            "lift": lift_values,
                            "lift_rate": lift_rates,
                        }
                    )["sha256"],
                }
            ),
        )


def prepare_affine_evolution_rom(
    problem: AffineEvolutionROMProblem,
    /,
) -> PreparedAffineEvolutionROM:
    if not isinstance(problem, AffineEvolutionROMProblem):
        raise TypeError("problem must be an AffineEvolutionROMProblem.")
    reduction = problem.reduction
    residual_space = DualSpace(reduction.test.reduced_space)
    masses = jnp.stack(
        tuple(reduction.project_operator(term) for term in problem.mass_terms)
    )
    operators = jnp.stack(
        tuple(reduction.project_operator(term) for term in problem.operator_terms)
    )
    rhs = jnp.stack(
        tuple(
            residual_space.flatten(reduction.project_covector(value))
            for value in problem.right_hand_side_terms
        )
    )
    if problem.lift_terms:
        mass_lift = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        residual_space.flatten(reduction.project_lift_action(term, lift))
                        for lift in problem.lift_terms
                    )
                )
                for term in problem.mass_terms
            )
        )
        operator_lift = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        residual_space.flatten(reduction.project_lift_action(term, lift))
                        for lift in problem.lift_terms
                    )
                )
                for term in problem.operator_terms
            )
        )
        lift_coordinates = jnp.stack(
            tuple(
                reduction.trial.full_space.flatten(value) for value in problem.lift_terms
            )
        )
    else:
        mass_lift = jnp.zeros((len(problem.mass_terms), 0, reduction.test_rank))
        operator_lift = jnp.zeros((len(problem.operator_terms), 0, reduction.test_rank))
        lift_coordinates = jnp.zeros((0, reduction.trial.full_space.size))
    content = {
        "mass": masses,
        "operator": operators,
        "rhs": rhs,
        "mass_lift": mass_lift,
        "operator_lift": operator_lift,
        "lift": lift_coordinates,
    }
    digest = array_tree_fingerprint(content)["sha256"]
    revision = NumericRevision(digest, label="affine-evolution-rom")
    model_id = canonical_fingerprint(
        {
            "kind": "prepared-affine-evolution-rom",
            "family": problem.family_id,
            "revision": revision.revision_id,
        }
    )
    return PreparedAffineEvolutionROM(
        reduction,
        masses,
        operators,
        rhs,
        mass_lift,
        operator_lift,
        lift_coordinates,
        revision,
        problem.family_id,
        model_id,
    )


__all__ = [
    "AffineEvolutionROMProblem",
    "PreparedAffineEvolutionROM",
    "prepare_affine_evolution_rom",
]

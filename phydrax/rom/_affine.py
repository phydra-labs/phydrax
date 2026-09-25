#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    DenseLinearOperator,
    DualSpace,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
)
from ._production import ROMAdmissionEvidence, ROMAdmissionStatus
from ._reduction import TrialTestReduction


class AffineCoefficientStatus(IntEnum):
    SUCCESS = 0
    OUT_OF_SUPPORT = 1
    NONFINITE = 2


class AffineLinearROMStatus(IntEnum):
    SUCCESS = 0
    UNSUPPORTED_INPUT = 1
    SOLVE_FAILED = 2


class AffineCoefficientEvaluation(StrictModule, NonTrainableState):
    operator: Array
    right_hand_side: Array
    lift: Array
    valid: Array
    status: Array
    coefficient_map_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: ArrayLike,
        right_hand_side: ArrayLike,
        lift: ArrayLike,
        valid: Any,
        status: Any,
        /,
        *,
        coefficient_map_id: str,
        support_id: str,
    ):
        self.operator = jnp.asarray(operator)
        self.right_hand_side = jnp.asarray(right_hand_side)
        self.lift = jnp.asarray(lift)
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.coefficient_map_id = str(coefficient_map_id)
        self.support_id = str(support_id)


class AbstractAffineCoefficientMap(StrictModule, NonTrainableState):
    """Portable parameter-to-affine-coefficient contract."""

    input_contract_id: eqx.AbstractVar[str]
    unit_contract_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    operator_term_ids: eqx.AbstractVar[tuple[str, ...]]
    right_hand_side_term_ids: eqx.AbstractVar[tuple[str, ...]]
    lift_term_ids: eqx.AbstractVar[tuple[str, ...]]
    coefficient_map_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def __call__(self, inputs: PyTree[Array], /) -> AffineCoefficientEvaluation:
        raise NotImplementedError


class ArrayAffineCoefficientMap(AbstractAffineCoefficientMap):
    """Affine coefficient map over one bounded parameter vector."""

    operator_matrix: Array
    operator_offset: Array
    right_hand_side_matrix: Array
    right_hand_side_offset: Array
    lift_matrix: Array
    lift_offset: Array
    lower: Array
    upper: Array
    input_size: int = eqx.field(static=True)
    input_contract_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    operator_term_ids: tuple[str, ...] = eqx.field(static=True)
    right_hand_side_term_ids: tuple[str, ...] = eqx.field(static=True)
    lift_term_ids: tuple[str, ...] = eqx.field(static=True)
    coefficient_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator_matrix: ArrayLike,
        operator_offset: ArrayLike,
        right_hand_side_matrix: ArrayLike,
        right_hand_side_offset: ArrayLike,
        /,
        *,
        operator_term_ids: Sequence[str],
        right_hand_side_term_ids: Sequence[str],
        lift_matrix: ArrayLike | None = None,
        lift_offset: ArrayLike | None = None,
        lift_term_ids: Sequence[str] = (),
        lower: ArrayLike,
        upper: ArrayLike,
        input_contract_id: str,
        unit_contract_id: str,
        support_id: str,
    ):
        operator = jnp.asarray(operator_matrix)
        operator_base = jnp.asarray(operator_offset)
        rhs = jnp.asarray(right_hand_side_matrix)
        rhs_base = jnp.asarray(right_hand_side_offset)
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper)
        if operator.ndim != 2 or rhs.ndim != 2:
            raise ValueError("Coefficient matrices must be two-dimensional.")
        input_size = operator.shape[1]
        if rhs.shape[1] != input_size:
            raise ValueError("All coefficient matrices must share one input width.")
        lift_ids = tuple(str(item) for item in lift_term_ids)
        lift = (
            jnp.zeros((0, input_size), dtype=operator.dtype)
            if lift_matrix is None
            else jnp.asarray(lift_matrix)
        )
        lift_base = (
            jnp.zeros((0,), dtype=operator.dtype)
            if lift_offset is None
            else jnp.asarray(lift_offset)
        )
        if lift.shape != (len(lift_ids), input_size):
            raise ValueError("lift_matrix shape must match lift terms and input width.")
        operator_ids = tuple(str(item) for item in operator_term_ids)
        rhs_ids = tuple(str(item) for item in right_hand_side_term_ids)
        if operator.shape[0] != len(operator_ids) or operator_base.shape != (
            len(operator_ids),
        ):
            raise ValueError("Operator coefficient arrays must match operator term IDs.")
        if rhs.shape[0] != len(rhs_ids) or rhs_base.shape != (len(rhs_ids),):
            raise ValueError("RHS coefficient arrays must match RHS term IDs.")
        if lift_base.shape != (len(lift_ids),):
            raise ValueError("lift_offset must match lift_term_ids.")
        if lower_.shape != (input_size,) or upper_.shape != (input_size,):
            raise ValueError("lower and upper must match the coefficient input width.")
        if any(not item for item in (*operator_ids, *rhs_ids, *lift_ids)):
            raise ValueError("Affine term IDs must be non-empty.")
        if (
            len(set(operator_ids)) != len(operator_ids)
            or len(set(rhs_ids)) != len(rhs_ids)
            or len(set(lift_ids)) != len(lift_ids)
        ):
            raise ValueError("Affine term IDs must be unique within each term family.")
        host_arrays = tuple(
            np.asarray(value)
            for value in (
                operator,
                operator_base,
                rhs,
                rhs_base,
                lift,
                lift_base,
                lower_,
                upper_,
            )
        )
        if any(not np.all(np.isfinite(value)) for value in host_arrays):
            raise ValueError("Affine coefficient-map arrays must be finite.")
        if np.any(np.asarray(lower_) > np.asarray(upper_)):
            raise ValueError("lower cannot exceed upper.")
        input_contract = str(input_contract_id)
        unit_contract = str(unit_contract_id)
        support = str(support_id)
        if not input_contract or not unit_contract or not support:
            raise ValueError(
                "Coefficient input, unit, and support IDs must be non-empty."
            )
        self.operator_matrix = operator
        self.operator_offset = operator_base
        self.right_hand_side_matrix = rhs
        self.right_hand_side_offset = rhs_base
        self.lift_matrix = lift
        self.lift_offset = lift_base
        self.lower = lower_
        self.upper = upper_
        self.input_size = input_size
        self.input_contract_id = input_contract
        self.unit_contract_id = unit_contract
        self.support_id = support
        self.operator_term_ids = operator_ids
        self.right_hand_side_term_ids = rhs_ids
        self.lift_term_ids = lift_ids
        self.coefficient_map_id = canonical_fingerprint(
            {
                "kind": "array-affine-coefficient-map",
                "input_contract": input_contract,
                "unit_contract": unit_contract,
                "support": support,
                "operator_terms": list(operator_ids),
                "rhs_terms": list(rhs_ids),
                "lift_terms": list(lift_ids),
                "content": array_tree_fingerprint(host_arrays)["sha256"],
            }
        )

    def __call__(self, inputs: PyTree[Array], /) -> AffineCoefficientEvaluation:
        value = jnp.asarray(inputs)
        if value.shape[-1:] != (self.input_size,):
            raise ValueError(
                f"Affine coefficient inputs must end in shape {(self.input_size,)}."
            )
        finite_input = jnp.all(jnp.isfinite(value), axis=-1)
        inside = jnp.all((value >= self.lower) & (value <= self.upper), axis=-1)
        operator = self.operator_offset + contract(
            "qi,...i->...q", self.operator_matrix, value
        )
        rhs = self.right_hand_side_offset + contract(
            "qi,...i->...q", self.right_hand_side_matrix, value
        )
        lift = self.lift_offset + contract("pi,...i->...p", self.lift_matrix, value)
        finite_output = (
            jnp.all(jnp.isfinite(operator), axis=-1)
            & jnp.all(jnp.isfinite(rhs), axis=-1)
            & jnp.all(jnp.isfinite(lift), axis=-1)
        )
        valid = finite_input & inside & finite_output
        status = jnp.where(
            ~finite_input | ~finite_output,
            int(AffineCoefficientStatus.NONFINITE),
            jnp.where(
                inside,
                int(AffineCoefficientStatus.SUCCESS),
                int(AffineCoefficientStatus.OUT_OF_SUPPORT),
            ),
        )
        return AffineCoefficientEvaluation(
            operator,
            rhs,
            lift,
            valid,
            status,
            coefficient_map_id=self.coefficient_map_id,
            support_id=self.support_id,
        )


class AffineLinearROMProblem(StrictModule, NonTrainableState):
    """One fixed-reference affine variational problem before projection."""

    reduction: TrialTestReduction
    operator_terms: tuple[AbstractLinearOperator, ...]
    right_hand_side_terms: tuple[PyTree[Array], ...]
    lift_terms: tuple[PyTree[Array], ...]
    observations: tuple[tuple[str, AbstractLinearOperator], ...]
    operator_term_ids: tuple[str, ...] = eqx.field(static=True)
    right_hand_side_term_ids: tuple[str, ...] = eqx.field(static=True)
    lift_term_ids: tuple[str, ...] = eqx.field(static=True)
    source_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: TrialTestReduction,
        operator_terms: Sequence[AbstractLinearOperator],
        right_hand_side_terms: Sequence[PyTree[Array]],
        /,
        *,
        operator_term_ids: Sequence[str],
        right_hand_side_term_ids: Sequence[str],
        lift_terms: Sequence[PyTree[Array]] = (),
        lift_term_ids: Sequence[str] = (),
        observations: Sequence[tuple[str, AbstractLinearOperator]] = (),
        source_artifact_ids: Sequence[str],
        family_id: str | None = None,
    ):
        if not isinstance(reduction, TrialTestReduction):
            raise TypeError("reduction must be a TrialTestReduction.")
        if not reduction.square:
            raise ValueError(
                "AffineLinearROMProblem requires equal trial and test ranks; "
                "use the rectangular least-squares route otherwise."
            )
        operators = tuple(operator_terms)
        rhs_terms = tuple(right_hand_side_terms)
        lifts = tuple(lift_terms)
        operator_ids = tuple(str(item) for item in operator_term_ids)
        rhs_ids = tuple(str(item) for item in right_hand_side_term_ids)
        lift_ids = tuple(str(item) for item in lift_term_ids)
        if not operators or len(operators) != len(operator_ids):
            raise ValueError(
                "operator_terms and operator_term_ids must be non-empty and aligned."
            )
        if not rhs_terms or len(rhs_terms) != len(rhs_ids):
            raise ValueError(
                "right_hand_side_terms and IDs must be non-empty and aligned."
            )
        if len(lifts) != len(lift_ids):
            raise ValueError("lift_terms and lift_term_ids must be aligned.")
        for operator in operators:
            reduction.validate_operator(operator)
        test_dual = DualSpace(reduction.test.full_space)
        rhs_terms = tuple(test_dual.validate(value) for value in rhs_terms)
        lifts = tuple(reduction.trial.full_space.validate(value) for value in lifts)
        observation_items = tuple(
            (str(name), operator) for name, operator in observations
        )
        if len({name for name, _ in observation_items}) != len(observation_items) or any(
            not name for name, _ in observation_items
        ):
            raise ValueError("Observation names must be unique and non-empty.")
        for _, observation in observation_items:
            if not isinstance(observation, AbstractLinearOperator):
                raise TypeError("Observation maps must be AbstractLinearOperator values.")
            if observation.batch_shape or not observation.source.compatible(
                reduction.trial.full_space
            ):
                raise ValueError(
                    "Observation maps must be unbatched and source-compatible with the trial space."
                )
        sources = tuple(str(item) for item in source_artifact_ids)
        if (
            not sources
            or any(not item for item in sources)
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("source_artifact_ids must be unique and non-empty.")
        resolved_family = (
            canonical_fingerprint(
                {
                    "kind": "affine-linear-rom-family",
                    "reduction": reduction.reduction_id,
                    "operators": list(operator_ids),
                    "rhs": list(rhs_ids),
                    "lifts": list(lift_ids),
                    "observations": [
                        [name, operator.operator_id]
                        for name, operator in observation_items
                    ],
                    "sources": list(sources),
                }
            )
            if family_id is None
            else str(family_id)
        )
        if not resolved_family:
            raise ValueError("family_id must be non-empty.")
        self.reduction = reduction
        self.operator_terms = operators
        self.right_hand_side_terms = rhs_terms
        self.lift_terms = lifts
        self.observations = observation_items
        self.operator_term_ids = operator_ids
        self.right_hand_side_term_ids = rhs_ids
        self.lift_term_ids = lift_ids
        self.source_artifact_ids = sources
        self.family_id = resolved_family


class PreparedAffineObservation(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    output_map: Array
    lift_values: Array
    output_space: Any
    observation_id: str = eqx.field(static=True)

    def evaluate(self, reduced: Array, lift_coefficients: Array, /):
        coordinates = self.output_map @ reduced
        if self.lift_values.shape[0]:
            coordinates = coordinates + lift_coefficients @ self.lift_values
        return self.output_space.unflatten(coordinates)


class NamedROMObservation(StrictModule):
    value: PyTree[Array]
    name: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)


class AffineLinearROMEvaluation(StrictModule):
    reduced_state: Array
    reconstructed_state: PyTree[Array] | None
    observations: tuple[NamedROMObservation, ...]
    reduced_residual: Array
    coefficients: AffineCoefficientEvaluation
    solve_result: LinearSolveResult | None
    valid: Array
    status: Array
    model_id: str = eqx.field(static=True)
    reconstruction_performed: bool = eqx.field(static=True)


class PreparedAffineLinearROM(StrictModule, NonTrainableState):
    reduction: TrialTestReduction
    coefficient_map: AbstractAffineCoefficientMap
    reduced_operator_terms: Array
    reduced_right_hand_side_terms: Array
    reduced_lift_cross_terms: Array
    lift_coordinates: Array
    observations: tuple[PreparedAffineObservation, ...]
    linear_solve: PreparedLinearSolve
    numeric_revision: NumericRevision
    family_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def admit(self, inputs: PyTree[Array], /) -> ROMAdmissionEvidence:
        coefficients = self.coefficient_map(inputs)
        status = jnp.where(
            coefficients.valid,
            int(ROMAdmissionStatus.ADMITTED),
            jnp.where(
                coefficients.status == int(AffineCoefficientStatus.NONFINITE),
                int(ROMAdmissionStatus.INPUT_SCHEMA),
                int(ROMAdmissionStatus.PARAMETER_SUPPORT),
            ),
        )
        return ROMAdmissionEvidence(
            coefficients.valid,
            status,
            jnp.where(coefficients.valid, 0.0, jnp.inf),
            support_id=coefficients.support_id,
            evidence_ids=(coefficients.coefficient_map_id, self.model_id),
        )

    def evaluate(
        self,
        inputs: PyTree[Array],
        /,
        *,
        reconstruct: bool = False,
    ) -> AffineLinearROMEvaluation:
        coefficients = self.coefficient_map(inputs)
        if coefficients.valid.ndim != 0:
            raise ValueError(
                "evaluate is the unbatched host convenience; use admit followed by "
                "evaluate_admitted for batched execution."
            )
        rank = self.reduction.rank
        dtype = self.reduced_operator_terms.dtype
        if not bool(np.asarray(coefficients.valid)):
            return AffineLinearROMEvaluation(
                jnp.zeros((rank,), dtype=dtype),
                None,
                (),
                jnp.zeros((rank,), dtype=dtype),
                coefficients,
                None,
                coefficients.valid,
                jnp.asarray(
                    int(AffineLinearROMStatus.UNSUPPORTED_INPUT), dtype=jnp.int32
                ),
                self.model_id,
                False,
            )
        return self._evaluate_coefficients(coefficients, reconstruct=reconstruct)

    def evaluate_admitted(
        self,
        inputs: PyTree[Array],
        /,
        *,
        reconstruct: bool = False,
    ) -> AffineLinearROMEvaluation:
        """Evaluate one query already admitted by the bound coefficient map."""
        value = jnp.asarray(inputs)
        if value.ndim > 1:
            return eqx.filter_vmap(
                lambda row: self.evaluate_admitted(row, reconstruct=reconstruct)
            )(value)
        coefficients = self.coefficient_map(inputs)
        guarded = eqx.error_if(
            coefficients.operator,
            ~coefficients.valid,
            "evaluate_admitted received an unsupported affine ROM input.",
        )
        coefficients = eqx.tree_at(
            lambda value: value.operator,
            coefficients,
            guarded,
        )
        return self._evaluate_coefficients(coefficients, reconstruct=reconstruct)

    def _evaluate_coefficients(
        self,
        coefficients: AffineCoefficientEvaluation,
        /,
        *,
        reconstruct: bool,
    ) -> AffineLinearROMEvaluation:
        matrix = contract(
            "q,qij->ij",
            coefficients.operator,
            self.reduced_operator_terms,
        )
        right_hand_side = contract(
            "q,qi->i",
            coefficients.right_hand_side,
            self.reduced_right_hand_side_terms,
        )
        if self.lift_coordinates.shape[0]:
            right_hand_side = right_hand_side - contract(
                "q,p,qpi->i",
                coefficients.operator,
                coefficients.lift,
                self.reduced_lift_cross_terms,
            )
        operator = DenseLinearOperator(
            matrix,
            source=self.reduction.trial.reduced_space,
            target=DualSpace(self.reduction.test.reduced_space),
            operator_id=f"{self.family_id}:reduced-operator",
        )
        system = LinearSystem(
            operator,
            problem_id=f"{self.family_id}:online-system",
        )
        result = solve(refresh(self.linear_solve, system), right_hand_side)
        reduced = jnp.asarray(result.value)
        residual = matrix @ reduced - right_hand_side
        lift = self.reduction.trial.full_space.zeros()
        if self.lift_coordinates.shape[0]:
            lift = self.reduction.trial.full_space.unflatten(
                coefficients.lift @ self.lift_coordinates
            )
        reconstructed = self.reduction.reconstruct(reduced, lift) if reconstruct else None
        observation_values = tuple(
            NamedROMObservation(
                observation.evaluate(reduced, coefficients.lift),
                observation.name,
                observation.observation_id,
            )
            for observation in self.observations
        )
        valid = coefficients.valid & result.successful & jnp.all(jnp.isfinite(residual))
        status = jnp.where(
            valid,
            int(AffineLinearROMStatus.SUCCESS),
            int(AffineLinearROMStatus.SOLVE_FAILED),
        ).astype(jnp.int32)
        return AffineLinearROMEvaluation(
            reduced,
            reconstructed,
            observation_values,
            residual,
            coefficients,
            result,
            valid,
            status,
            self.model_id,
            bool(reconstruct),
        )


class AffineLinearROMPlan(StrictModule, NonTrainableState):
    solve_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, solve_policy: LinearSolvePolicy | None = None):
        policy = LinearSolvePolicy() if solve_policy is None else solve_policy
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be a LinearSolvePolicy or None.")
        self.solve_policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "affine-linear-rom-plan",
                "solve_policy": repr(policy),
            }
        )

    def prepare(
        self,
        problem: AffineLinearROMProblem,
        coefficient_map: AbstractAffineCoefficientMap,
        /,
    ) -> PreparedAffineLinearROM:
        if not isinstance(problem, AffineLinearROMProblem):
            raise TypeError("problem must be an AffineLinearROMProblem.")
        if not isinstance(coefficient_map, AbstractAffineCoefficientMap):
            raise TypeError("coefficient_map must be an AbstractAffineCoefficientMap.")
        if coefficient_map.operator_term_ids != problem.operator_term_ids:
            raise ValueError("Coefficient-map operator term ordering does not match.")
        if coefficient_map.right_hand_side_term_ids != problem.right_hand_side_term_ids:
            raise ValueError("Coefficient-map RHS term ordering does not match.")
        if coefficient_map.lift_term_ids != problem.lift_term_ids:
            raise ValueError("Coefficient-map lift term ordering does not match.")
        reduced_operators = jnp.stack(
            tuple(
                problem.reduction.project_operator(operator)
                for operator in problem.operator_terms
            )
        )
        reduced_dual = DualSpace(problem.reduction.test.reduced_space)
        reduced_rhs = jnp.stack(
            tuple(
                reduced_dual.flatten(problem.reduction.project_covector(value))
                for value in problem.right_hand_side_terms
            )
        )
        if problem.lift_terms:
            lift_cross = jnp.stack(
                tuple(
                    jnp.stack(
                        tuple(
                            reduced_dual.flatten(
                                problem.reduction.project_lift_action(operator, lift)
                            )
                            for lift in problem.lift_terms
                        )
                    )
                    for operator in problem.operator_terms
                )
            )
            lift_coordinates = jnp.stack(
                tuple(
                    problem.reduction.trial.full_space.flatten(lift)
                    for lift in problem.lift_terms
                )
            )
        else:
            lift_cross = jnp.zeros(
                (len(problem.operator_terms), 0, problem.reduction.rank),
                dtype=reduced_operators.dtype,
            )
            lift_coordinates = jnp.zeros(
                (0, problem.reduction.trial.full_space.size),
                dtype=reduced_operators.dtype,
            )
        trial_coordinates = jnp.eye(
            problem.reduction.rank,
            dtype=reduced_operators.dtype,
        )
        trial_vectors = problem.reduction.trial.prolongation.mv_block(trial_coordinates)
        observations = []
        for name, observation in problem.observations:
            output_map = observation.mv_block(trial_vectors)
            lift_values = (
                jnp.stack(
                    tuple(
                        observation.target.flatten(observation.mv(lift))
                        for lift in problem.lift_terms
                    )
                )
                if problem.lift_terms
                else jnp.zeros(
                    (0, observation.target.size),
                    dtype=output_map.dtype,
                )
            )
            observations.append(
                PreparedAffineObservation(
                    name,
                    output_map,
                    lift_values,
                    observation.target,
                    canonical_fingerprint(
                        {
                            "kind": "prepared-affine-observation",
                            "name": name,
                            "operator": observation.operator_id,
                            "reduction": problem.reduction.reduction_id,
                        }
                    ),
                )
            )
        template_operator = DenseLinearOperator(
            jnp.eye(problem.reduction.rank, dtype=reduced_operators.dtype),
            source=problem.reduction.trial.reduced_space,
            target=DualSpace(problem.reduction.test.reduced_space),
            operator_id=f"{problem.family_id}:reduced-operator",
        )
        linear_solve = prepare(
            LinearSystem(
                template_operator,
                problem_id=f"{problem.family_id}:online-system",
            ),
            self.solve_policy,
        )
        content = {
            "operator": reduced_operators,
            "rhs": reduced_rhs,
            "lift_cross": lift_cross,
            "lift_coordinates": lift_coordinates,
            "observations": tuple(
                (item.output_map, item.lift_values) for item in observations
            ),
        }
        revision = NumericRevision(
            SemanticProvenance(
                {
                    "kind": "affine-linear-rom",
                    "family": problem.family_id,
                    "reduction": problem.reduction.reduction_id,
                    "coefficient_map": coefficient_map.coefficient_map_id,
                }
            ),
            content,
        )
        model_id = canonical_fingerprint(
            {
                "kind": "prepared-affine-linear-rom",
                "family": problem.family_id,
                "reduction": problem.reduction.reduction_id,
                "coefficient_map": coefficient_map.coefficient_map_id,
                "support": coefficient_map.support_id,
                "plan": self.plan_id,
                "revision": revision.revision_id,
            }
        )
        return PreparedAffineLinearROM(
            problem.reduction,
            coefficient_map,
            reduced_operators,
            reduced_rhs,
            lift_cross,
            lift_coordinates,
            tuple(observations),
            linear_solve,
            revision,
            problem.family_id,
            model_id,
        )


def prepare_affine_linear_rom(
    problem: AffineLinearROMProblem,
    coefficient_map: AbstractAffineCoefficientMap,
    /,
    *,
    plan: AffineLinearROMPlan | None = None,
) -> PreparedAffineLinearROM:
    policy = AffineLinearROMPlan() if plan is None else plan
    if not isinstance(policy, AffineLinearROMPlan):
        raise TypeError("plan must be an AffineLinearROMPlan or None.")
    return policy.prepare(problem, coefficient_map)


__all__ = [
    "AbstractAffineCoefficientMap",
    "AffineCoefficientEvaluation",
    "AffineCoefficientStatus",
    "AffineLinearROMEvaluation",
    "AffineLinearROMPlan",
    "AffineLinearROMProblem",
    "AffineLinearROMStatus",
    "ArrayAffineCoefficientMap",
    "NamedROMObservation",
    "PreparedAffineLinearROM",
    "PreparedAffineObservation",
    "prepare_affine_linear_rom",
]

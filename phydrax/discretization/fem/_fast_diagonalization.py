#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    AbstractVectorSpace,
    ArraySpace,
    DenseLinearOperator,
    DiagonalPairing,
    FailurePolicy,
    KroneckerSumLinearOperator,
    LinearCapabilityError,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    OperatorProperties,
    PreconditionerCostEstimate,
    PreconditionerProperties,
    prepare,
    PreparedLinearSolve,
    solve,
    StructuredDirect,
)


class FastDiagonalizationEligibility(StrictModule):
    """Static acceptance decision and complete rejection evidence."""

    eligible: bool = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    axis_sizes: tuple[int, ...] = eqx.field(static=True)
    eligibility_id: str = eqx.field(static=True)

    def __init__(
        self,
        eligible: bool,
        reasons: tuple[str, ...],
        axis_sizes: tuple[int, ...],
        /,
    ):
        reasons_ = tuple(str(reason) for reason in reasons)
        sizes = tuple(int(size) for size in axis_sizes)
        if any(not reason for reason in reasons_) or any(size < 1 for size in sizes):
            raise ValueError("Fast-diagonalization eligibility evidence is invalid.")
        accepted = bool(eligible)
        if accepted == bool(reasons_):
            raise ValueError(
                "Eligible fast diagonalization has no rejection reasons and "
                "ineligible fast diagonalization has at least one."
            )
        self.eligible = accepted
        self.reasons = reasons_
        self.axis_sizes = sizes
        self.eligibility_id = canonical_fingerprint(
            {
                "kind": "tensor-fast-diagonalization-eligibility",
                "eligible": accepted,
                "reasons": list(reasons_),
                "axis_sizes": list(sizes),
            }
        )


class TensorFastDiagonalizationPreconditioner(AbstractPreconditioner):
    """Physical tensor inverse delegated to Phydrax structured direct solve."""

    prepared: PreparedLinearSolve
    physical_mass_diagonal: Array
    eligibility: FastDiagonalizationEligibility
    builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedLinearSolve,
        physical_mass_diagonal: ArrayLike,
        space: AbstractVectorSpace,
        eligibility: FastDiagonalizationEligibility,
        builder_id: str,
        /,
    ):
        if not isinstance(prepared, PreparedLinearSolve):
            raise TypeError("prepared must be a PreparedLinearSolve.")
        structured_operator = prepared.problem.operator
        if not isinstance(structured_operator, KroneckerSumLinearOperator):
            raise TypeError(
                "Prepared fast diagonalization requires a Kronecker-sum operator."
            )
        factor_matrices = []
        for factor in structured_operator.factors:
            if not isinstance(factor, DenseLinearOperator):
                raise TypeError(
                    "Prepared fast diagonalization requires dense axis factors."
                )
            factor_matrices.append(factor.matrix)
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        if not isinstance(eligibility, FastDiagonalizationEligibility) or not (
            eligibility.eligible
        ):
            raise ValueError("A prepared FDM action requires positive eligibility.")
        diagonal = jnp.asarray(physical_mass_diagonal)
        if diagonal.shape != (space.size,):
            raise ValueError("Tensor physical mass diagonal has the wrong shape.")
        identifier = str(builder_id)
        if not identifier:
            raise ValueError("builder_id must be non-empty.")
        self.prepared = prepared
        self.physical_mass_diagonal = diagonal
        self.eligibility = eligibility
        self.builder_id = identifier
        self.space = space
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "tensor-fast-diagonalization-preconditioner",
                "builder": identifier,
                "prepared_plan": prepared.plan.plan_id,
                "eligibility": eligibility.eligibility_id,
                "space": space.space_id,
                "numeric": array_tree_fingerprint(
                    (
                        diagonal,
                        tuple(factor_matrices),
                    )
                ),
            }
        )

    def apply(
        self,
        residual: PyTree,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        del iteration
        coordinates = self.space.flatten(self.space.validate(residual))
        structured = self.prepared.problem.operator.source
        weighted = coordinates / self.physical_mass_diagonal
        right_hand_side = structured.unflatten(weighted)
        result = solve(self.prepared, right_hand_side)
        value = eqx.error_if(
            result.value,
            ~jnp.all(result.successful),
            "Tensor fast-diagonalization solve failed.",
        )
        return self.space.unflatten(structured.flatten(value))


class TensorFastDiagonalizationBuilder(AbstractPreconditionerBuilder):
    """Build a quad/hex separable mass-diffusion inverse without dense tensors."""

    mass_operators: tuple[AbstractLinearOperator, ...]
    stiffness_operators: tuple[AbstractLinearOperator, ...]
    diffusion: tuple[float, ...] = eqx.field(static=True)
    reaction: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass_operators: tuple[AbstractLinearOperator, ...],
        stiffness_operators: tuple[AbstractLinearOperator, ...],
        /,
        *,
        diffusion: tuple[float, ...] | None = None,
        reaction: float = 0.0,
    ):
        masses = tuple(mass_operators)
        stiffnesses = tuple(stiffness_operators)
        if len(masses) not in (2, 3) or len(stiffnesses) != len(masses):
            raise ValueError(
                "Fast diagonalization requires two quad axes or three hex axes."
            )
        if not all(
            isinstance(operator, AbstractLinearOperator)
            for operator in (*masses, *stiffnesses)
        ):
            raise TypeError("Fast-diagonalization factors must be linear operators.")
        coefficients = (
            (1.0,) * len(masses)
            if diffusion is None
            else tuple(float(value) for value in diffusion)
        )
        reaction_ = float(reaction)
        if (
            len(coefficients) != len(masses)
            or any(not isfinite(value) or value < 0.0 for value in coefficients)
            or not isfinite(reaction_)
            or reaction_ < 0.0
        ):
            raise ValueError(
                "Fast-diagonalization diffusion/reaction coefficients must be "
                "finite and non-negative."
            )
        self.mass_operators = masses
        self.stiffness_operators = stiffnesses
        self.diffusion = coefficients
        self.reaction = reaction_
        self._builder_id = canonical_fingerprint(
            {
                "kind": "tensor-fast-diagonalization-builder",
                "mass_operators": [operator.operator_id for operator in masses],
                "stiffness_operators": [operator.operator_id for operator in stiffnesses],
                "diffusion": list(coefficients),
                "reaction": reaction_,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def _structural_reasons(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        reasons = []
        if not isinstance(setup_operator, AbstractLinearOperator):
            raise TypeError("setup_operator must be an AbstractLinearOperator.")
        axis_sizes = tuple(operator.source.size for operator in self.mass_operators)
        if setup_operator.batch_shape or not setup_operator.source.compatible(
            setup_operator.target
        ):
            reasons.append("setup operator is not an unbatched endomorphism")
        if setup_operator.source.size != prod(axis_sizes):
            reasons.append("setup space size does not equal the tensor DOF count")
        for axis, (mass, stiffness) in enumerate(
            zip(self.mass_operators, self.stiffness_operators, strict=True)
        ):
            if (
                mass.batch_shape
                or stiffness.batch_shape
                or not mass.source.compatible(mass.target)
                or not stiffness.source.compatible(stiffness.target)
                or mass.source.size != stiffness.source.size
            ):
                reasons.append(f"axis {axis} factors are not matching endomorphisms")
            if (
                not mass.capabilities.materialize
                or not stiffness.capabilities.materialize
            ):
                reasons.append(f"axis {axis} factors lack materialization capability")
            if not mass.properties.certifies("diagonal"):
                reasons.append(f"axis {axis} mass is not certified diagonal")
            if not mass.properties.certifies("positive_definite"):
                reasons.append(f"axis {axis} mass is not certified positive definite")
            if not stiffness.properties.certifies("positive_semidefinite"):
                reasons.append(
                    f"axis {axis} stiffness is not certified positive semidefinite"
                )
        nonsingular = self.reaction > 0.0 or any(
            coefficient > 0.0 and stiffness.properties.certifies("positive_definite")
            for coefficient, stiffness in zip(
                self.diffusion,
                self.stiffness_operators,
                strict=True,
            )
        )
        if not nonsingular:
            reasons.append("positive modal denominators are not certified")
        return tuple(reasons), axis_sizes

    def _materialized_factors(
        self,
        policy: MaterializationPolicy,
        /,
    ) -> tuple[tuple[Array, Array, Array], ...]:
        factors = []
        for mass, stiffness in zip(
            self.mass_operators,
            self.stiffness_operators,
            strict=True,
        ):
            mass_matrix = materialize(mass, policy)
            factors.append(
                (
                    mass_matrix,
                    materialize(stiffness, policy),
                    jnp.diag(mass_matrix),
                )
            )
        return tuple(factors)

    @staticmethod
    def _numerical_reasons(
        factors: tuple[tuple[Array, Array, Array], ...],
        /,
    ) -> tuple[str, ...]:
        reasons = []
        for axis, (mass, stiffness, diagonal) in enumerate(factors):
            mass_host = np.asarray(mass)
            stiffness_host = np.asarray(stiffness)
            diagonal_host = np.real(np.asarray(diagonal))
            real_dtype = diagonal_host.dtype
            scale = max(float(np.max(np.abs(mass_host))), 1.0)
            tolerance = np.finfo(real_dtype).eps * mass_host.shape[0] * scale
            off_diagonal = mass_host - np.diag(np.diag(mass_host))
            if not np.all(np.isfinite(mass_host)) or not np.all(
                np.isfinite(stiffness_host)
            ):
                reasons.append(f"axis {axis} factors contain non-finite entries")
            if np.max(np.abs(off_diagonal)) > tolerance:
                reasons.append(f"axis {axis} mass has nonzero off-diagonal entries")
            if np.any(np.abs(np.imag(np.diag(mass_host))) > tolerance) or np.any(
                diagonal_host <= 0.0
            ):
                reasons.append(f"axis {axis} mass diagonal is not real and positive")
            stiffness_scale = max(float(np.max(np.abs(stiffness_host))), 1.0)
            stiffness_tolerance = (
                np.finfo(stiffness_host.real.dtype).eps
                * stiffness_host.shape[0]
                * stiffness_scale
            )
            if (
                np.max(np.abs(stiffness_host - np.conj(stiffness_host.T)))
                > stiffness_tolerance
            ):
                reasons.append(f"axis {axis} stiffness is not numerically Hermitian")
        return tuple(reasons)

    def eligibility_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> FastDiagonalizationEligibility:
        reasons, axis_sizes = self._structural_reasons(setup_operator)
        if materialization is None:
            reasons = (*reasons, "an explicit materialization policy is required")
        else:
            if not isinstance(materialization, MaterializationPolicy):
                raise TypeError(
                    "materialization must be a MaterializationPolicy or None."
                )
            try:
                factors = self._materialized_factors(materialization)
            except LinearCapabilityError as error:
                reasons = (*reasons, str(error))
            else:
                reasons = (*reasons, *self._numerical_reasons(factors))
        return FastDiagonalizationEligibility(not reasons, reasons, axis_sizes)

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        reasons, _ = self._structural_reasons(setup_operator)
        if reasons:
            raise ValueError(
                "Fast-diagonalization properties are unavailable: " + "; ".join(reasons)
            )
        return PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        evidence = self.eligibility_for(
            setup_operator,
            materialization=materialization,
        )
        itemsize = max(
            np.dtype(leaf.dtype).itemsize
            for operator in self.mass_operators
            for leaf in jax.tree.leaves(operator.source.structure())
        )
        factor_entries = sum(2 * size * size + size for size in evidence.axis_sizes)
        tensor_size = prod(evidence.axis_sizes)
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=(factor_entries + tensor_size) * itemsize,
            preparation_workspace_bytes=sum(
                3 * size * size * itemsize for size in evidence.axis_sizes
            ),
            apply_workspace_bytes_per_rhs=4 * tensor_size * itemsize,
            accepted=evidence.eligible,
            reason=(
                "separable diagonal-mass quad/hex fast diagonalization"
                if evidence.eligible
                else "; ".join(evidence.reasons)
            ),
        )

    def _structured_operator(
        self,
        factors: tuple[tuple[Array, Array, Array], ...],
        /,
    ) -> tuple[KroneckerSumLinearOperator, Array]:
        dimension = len(factors)
        axis_operators = []
        for axis, ((_, stiffness, mass_diagonal), coefficient) in enumerate(
            zip(factors, self.diffusion, strict=True)
        ):
            axis_space = ArraySpace(
                (mass_diagonal.size,),
                dtype=stiffness.dtype,
                pairing=DiagonalPairing(jnp.real(mass_diagonal)),
            )
            matrix = coefficient * stiffness / mass_diagonal[:, None] + (
                self.reaction / dimension
            ) * jnp.eye(mass_diagonal.size, dtype=stiffness.dtype)
            positive_definite = self.reaction > 0.0 or (
                coefficient > 0.0
                and self.stiffness_operators[axis].properties.certifies(
                    "positive_definite"
                )
            )
            axis_operators.append(
                DenseLinearOperator(
                    matrix,
                    source=axis_space,
                    target=axis_space,
                    properties=OperatorProperties(
                        self_adjoint=True,
                        positive_semidefinite=True,
                        positive_definite=positive_definite,
                        evidence={
                            "self_adjoint": "transformed",
                            "positive_semidefinite": "transformed",
                            **(
                                {"positive_definite": "transformed"}
                                if positive_definite
                                else {}
                            ),
                        },
                    ),
                    operator_id=f"fdm-axis/{self.builder_id}/{axis}",
                )
            )
        structured = KroneckerSumLinearOperator(
            tuple(axis_operators),
            operator_id=f"fdm-kronecker-sum/{self.builder_id}",
        )
        ones = jnp.ones(
            structured.source.structure().shape,
            dtype=structured.source.structure().dtype,
        )
        physical_mass = structured.source.flatten(structured.source.riesz(ones))
        return structured, physical_mass

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(materialization, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        structural_reasons, axis_sizes = self._structural_reasons(setup_operator)
        try:
            factors = self._materialized_factors(materialization)
        except LinearCapabilityError as error:
            structural_reasons = (*structural_reasons, str(error))
            factors = ()
        numerical_reasons = () if not factors else self._numerical_reasons(factors)
        reasons = (*structural_reasons, *numerical_reasons)
        evidence = FastDiagonalizationEligibility(not reasons, reasons, axis_sizes)
        if not evidence.eligible:
            raise ValueError(
                "Fast diagonalization is ineligible: " + "; ".join(evidence.reasons)
            )
        structured, physical_mass = self._structured_operator(factors)
        prepared = prepare(
            LinearSystem(structured),
            LinearSolvePolicy(
                StructuredDirect(),
                materialization=materialization,
                failure=FailurePolicy("status"),
            ),
        )
        return TensorFastDiagonalizationPreconditioner(
            prepared,
            physical_mass,
            setup_operator.source,
            evidence,
            self.builder_id,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, TensorFastDiagonalizationPreconditioner):
            raise TypeError(
                "FDM refresh requires a TensorFastDiagonalizationPreconditioner."
            )
        if preconditioner.builder_id != self.builder_id:
            raise ValueError("FDM refresh must preserve the builder identity.")
        return self.prepare(setup_operator, materialization=materialization)


__all__ = [
    "FastDiagonalizationEligibility",
    "TensorFastDiagonalizationBuilder",
    "TensorFastDiagonalizationPreconditioner",
]

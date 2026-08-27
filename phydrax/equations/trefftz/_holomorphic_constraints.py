#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from operator import index
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._holomorphic import (
    ComplexAffineNormalization,
    HolomorphicJet,
    HolomorphicMapCertificate,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    RankPolicy,
    SolveResourcePolicy,
)
from ._holomorphic import _derivative_coefficients, _horner


HolomorphicConstraintComponent = Literal["real", "imaginary"]


def _finite_real_scalar(value: ArrayLike, name: str, /) -> float:
    raw = np.asarray(value)
    if raw.shape != () or np.iscomplexobj(raw):
        raise TypeError(f"{name} must be one real scalar.")
    resolved = float(raw)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


class HolomorphicPointConstraint(StrictModule, NonTrainableState):
    """One real linear value/normal-derivative constraint at a complex point."""

    coordinate: Array
    normal: Array
    target: Array
    component: HolomorphicConstraintComponent = eqx.field(static=True)
    value_weight: float = eqx.field(static=True)
    normal_weight: float = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: ArrayLike,
        target: ArrayLike,
        /,
        *,
        component: HolomorphicConstraintComponent = "real",
        normal: ArrayLike = (0.0, 0.0),
        value_weight: float = 1.0,
        normal_weight: float = 0.0,
    ):
        coordinate_raw = np.asarray(coordinate)
        if coordinate_raw.shape != ():
            raise ValueError("Holomorphic point constraint coordinate must be scalar.")
        coordinate_ = complex(coordinate_raw)
        if not math.isfinite(abs(coordinate_)):
            raise ValueError("Holomorphic point constraint coordinate must be finite.")
        if component not in ("real", "imaginary"):
            raise ValueError(
                "Holomorphic constraint component must be real or imaginary."
            )
        normal_raw = np.asarray(normal)
        if normal_raw.shape != (2,) or np.iscomplexobj(normal_raw):
            raise TypeError(
                "Holomorphic point constraint normal must be real shape (2,)."
            )
        normal_ = np.asarray(normal_raw, dtype=np.float64)
        if not np.all(np.isfinite(normal_)):
            raise ValueError("Holomorphic point constraint normal must be finite.")
        target_ = _finite_real_scalar(target, "target")
        value_weight_ = _finite_real_scalar(value_weight, "value_weight")
        normal_weight_ = _finite_real_scalar(normal_weight, "normal_weight")
        if value_weight_ == 0.0 and normal_weight_ == 0.0:
            raise ValueError("A holomorphic point constraint requires a nonzero weight.")
        if normal_weight_ != 0.0 and not np.any(normal_ != 0.0):
            raise ValueError("Normal-derivative constraints require a nonzero normal.")
        coordinate_array = jnp.asarray(coordinate_, dtype=jnp.complex128)
        normal_array = jnp.asarray(normal_, dtype=jnp.float64)
        target_array = jnp.asarray(target_, dtype=jnp.float64)
        self.coordinate = coordinate_array
        self.normal = normal_array
        self.target = target_array
        self.component = component
        self.value_weight = value_weight_
        self.normal_weight = normal_weight_
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "holomorphic-point-constraint",
                "coordinate": array_tree_fingerprint(coordinate_array),
                "normal": array_tree_fingerprint(normal_array),
                "target": array_tree_fingerprint(target_array),
                "component": component,
                "value_weight": value_weight_,
                "normal_weight": normal_weight_,
            }
        )

    @classmethod
    def dirichlet(
        cls,
        coordinate: ArrayLike,
        target: ArrayLike,
        /,
        *,
        component: HolomorphicConstraintComponent = "real",
    ) -> HolomorphicPointConstraint:
        """Constrain one real or imaginary potential value."""
        return cls(coordinate, target, component=component)

    @classmethod
    def normal_derivative(
        cls,
        coordinate: ArrayLike,
        normal: ArrayLike,
        target: ArrayLike,
        /,
        *,
        component: HolomorphicConstraintComponent = "real",
    ) -> HolomorphicPointConstraint:
        """Constrain one supplied-normal directional derivative."""
        return cls(
            coordinate,
            target,
            component=component,
            normal=normal,
            value_weight=0.0,
            normal_weight=1.0,
        )

    @classmethod
    def robin(
        cls,
        coordinate: ArrayLike,
        normal: ArrayLike,
        target: ArrayLike,
        /,
        *,
        value_weight: float,
        normal_weight: float,
        component: HolomorphicConstraintComponent = "real",
    ) -> HolomorphicPointConstraint:
        """Constrain a weighted value plus supplied-normal derivative."""
        return cls(
            coordinate,
            target,
            component=component,
            normal=normal,
            value_weight=value_weight,
            normal_weight=normal_weight,
        )


def _constraint_row(
    constraint: HolomorphicPointConstraint,
    maximum_degree: int,
    normalization: ComplexAffineNormalization,
    /,
) -> np.ndarray:
    coordinate = complex(np.asarray(constraint.coordinate))
    center = complex(np.asarray(normalization.center)[0])
    scale = complex(np.asarray(normalization.matrix)[0, 0])
    normalized = scale * (coordinate - center)
    powers = np.asarray(
        [normalized**power for power in range(maximum_degree + 1)],
        dtype=np.complex128,
    )
    derivatives = np.zeros_like(powers)
    for power in range(1, maximum_degree + 1):
        derivatives[power] = power * scale * normalized ** (power - 1)
    normal = np.asarray(constraint.normal)
    direction = complex(float(normal[0]), float(normal[1]))
    functional = (
        constraint.value_weight * powers
        + constraint.normal_weight * direction * derivatives
    )
    if constraint.component == "real":
        return np.concatenate((np.real(functional), -np.imag(functional)))
    return np.concatenate((np.imag(functional), np.real(functional)))


def _canonical_columns(values: np.ndarray, /) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


class HolomorphicConstraintEvidence(StrictModule, NonTrainableState):
    """Numerical rank, lift, and nullspace evidence for prepared constraints."""

    singular_values: Array
    lift_residual_norm: Array
    nullspace_residual_norm: Array
    lift_tolerance: Array
    nullspace_tolerance: Array
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)
    factorization_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        singular_values: ArrayLike,
        lift_residual_norm: ArrayLike,
        nullspace_residual_norm: ArrayLike,
        lift_tolerance: ArrayLike,
        nullspace_tolerance: ArrayLike,
        rank: int,
        nullity: int,
        factorization_id: str,
        plan_id: str,
    ):
        singular_values_ = jnp.asarray(singular_values)
        scalar_values = tuple(
            jnp.asarray(value)
            for value in (
                lift_residual_norm,
                nullspace_residual_norm,
                lift_tolerance,
                nullspace_tolerance,
            )
        )
        if singular_values_.ndim != 1 or any(
            value.shape != () for value in scalar_values
        ):
            raise ValueError("Holomorphic constraint evidence has invalid array shapes.")
        if not bool(jnp.all(jnp.isfinite(singular_values_))) or any(
            not bool(jnp.isfinite(value)) for value in scalar_values
        ):
            raise ValueError("Holomorphic constraint evidence must be finite.")
        rank_ = int(rank)
        nullity_ = int(nullity)
        factorization_id_ = str(factorization_id)
        plan_id_ = str(plan_id)
        if rank_ < 0 or nullity_ < 0:
            raise ValueError(
                "Holomorphic constraint rank and nullity must be nonnegative."
            )
        if not factorization_id_ or not plan_id_:
            raise ValueError(
                "Holomorphic constraint evidence identifiers must be nonempty."
            )
        (
            self.lift_residual_norm,
            self.nullspace_residual_norm,
            self.lift_tolerance,
            self.nullspace_tolerance,
        ) = scalar_values
        self.singular_values = singular_values_
        self.rank = rank_
        self.nullity = nullity_
        self.factorization_id = factorization_id_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "holomorphic-constraint-evidence",
                "plan": plan_id_,
                "factorization": factorization_id_,
                "rank": rank_,
                "nullity": nullity_,
                "singular_values": array_tree_fingerprint(singular_values_),
                "lift_residual_norm": array_tree_fingerprint(scalar_values[0]),
                "nullspace_residual_norm": array_tree_fingerprint(scalar_values[1]),
                "lift_tolerance": array_tree_fingerprint(scalar_values[2]),
                "nullspace_tolerance": array_tree_fingerprint(scalar_values[3]),
            }
        )


class HolomorphicPolynomialConstraintPlan(StrictModule, NonTrainableState):
    """Prepare an affine real-coordinate polynomial coefficient constraint map."""

    normalization: ComplexAffineNormalization
    constraints: tuple[HolomorphicPointConstraint, ...]
    maximum_degree: int = eqx.field(static=True)
    rank_cutoff: float | None = eqx.field(static=True)
    maximum_factor_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_degree: int,
        constraints: Sequence[HolomorphicPointConstraint],
        /,
        *,
        normalization: ComplexAffineNormalization | None = None,
        rank_cutoff: float | None = None,
        maximum_factor_bytes: int = 512 * 1024**2,
        maximum_workspace_bytes: int = 512 * 1024**2,
    ):
        degree = int(maximum_degree)
        if degree < 0:
            raise ValueError("Holomorphic constraint degree must be nonnegative.")
        constraints_ = tuple(constraints)
        if not constraints_ or not all(
            isinstance(constraint, HolomorphicPointConstraint)
            for constraint in constraints_
        ):
            raise TypeError("constraints must contain HolomorphicPointConstraint values.")
        normalization_ = (
            ComplexAffineNormalization.identity(1)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, ComplexAffineNormalization):
            raise TypeError("normalization must be ComplexAffineNormalization or None.")
        if normalization_.dimension != 1:
            raise ValueError("Polynomial constraints require one complex input.")
        cutoff = None if rank_cutoff is None else float(rank_cutoff)
        if cutoff is not None and (not math.isfinite(cutoff) or cutoff < 0.0):
            raise ValueError("rank_cutoff must be nonnegative and finite or None.")
        if isinstance(maximum_factor_bytes, bool) or isinstance(
            maximum_workspace_bytes, bool
        ):
            raise TypeError("Holomorphic constraint resource budgets must be integers.")
        factor_bytes = index(maximum_factor_bytes)
        workspace_bytes = index(maximum_workspace_bytes)
        if factor_bytes <= 0 or workspace_bytes <= 0:
            raise ValueError("Holomorphic constraint resource budgets must be positive.")
        self.maximum_degree = degree
        self.normalization = normalization_
        self.constraints = constraints_
        self.rank_cutoff = cutoff
        self.maximum_factor_bytes = factor_bytes
        self.maximum_workspace_bytes = workspace_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "holomorphic-polynomial-constraint-plan",
                "maximum_degree": degree,
                "normalization": normalization_.normalization_id,
                "constraints": [constraint.constraint_id for constraint in constraints_],
                "rank_cutoff": cutoff,
                "maximum_factor_bytes": factor_bytes,
                "maximum_workspace_bytes": workspace_bytes,
            }
        )

    @property
    def coefficient_count(self) -> int:
        return 2 * (self.maximum_degree + 1)

    def prepare(self, /) -> PreparedHolomorphicPolynomialConstraints:
        matrix_numpy = np.stack(
            tuple(
                _constraint_row(constraint, self.maximum_degree, self.normalization)
                for constraint in self.constraints
            )
        )
        target_numpy = np.asarray(
            [float(np.asarray(constraint.target)) for constraint in self.constraints],
            dtype=np.float64,
        )
        row_norms = np.linalg.norm(matrix_numpy, axis=1)
        row_tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * max(self.coefficient_count, 1)
            * max(float(np.linalg.norm(matrix_numpy)), 1.0)
        )
        if np.any(row_norms <= row_tolerance):
            raise ValueError(
                "At least one point constraint is identically zero on the polynomial basis."
            )
        matrix = jnp.asarray(matrix_numpy, dtype=jnp.float64)
        target = jnp.asarray(target_numpy, dtype=jnp.float64)
        factorization = factorize(
            DenseLinearOperator(matrix),
            FactorizationPolicy(
                "svd",
                rank=RankPolicy(relative_cutoff=self.rank_cutoff),
                resources=SolveResourcePolicy(
                    factorization_bytes=self.maximum_factor_bytes,
                    workspace_bytes=self.maximum_workspace_bytes,
                ),
            ),
        )
        result = factorization.solve(target)
        particular = jnp.asarray(result.value, dtype=matrix.dtype)
        rank = int(np.asarray(factorization.rank()))
        nullity = self.coefficient_count - rank
        nullspace = factorization.right_nullspace()
        if int(np.asarray(nullspace.dimension)) != nullity:
            raise RuntimeError(
                "Holomorphic constraint nullspace dimension is inconsistent."
            )
        nullspace_basis = jnp.asarray(
            _canonical_columns(np.asarray(nullspace.basis[:, :nullity])),
            dtype=matrix.dtype,
        )
        lift_residual = matrix @ particular - target
        nullspace_residual = matrix @ nullspace_basis
        epsilon = np.finfo(np.asarray(matrix).dtype).eps
        matrix_norm = float(np.linalg.norm(np.asarray(matrix)))
        lift_scale = max(
            matrix_norm * float(np.linalg.norm(np.asarray(particular))),
            float(np.linalg.norm(target_numpy)),
            1.0,
        )
        nullspace_scale = max(
            matrix_norm * float(np.linalg.norm(np.asarray(nullspace_basis))),
            1.0,
        )
        dimension_scale = max(matrix.shape)
        lift_tolerance = jnp.asarray(
            512.0 * epsilon * dimension_scale * lift_scale,
            dtype=matrix.dtype,
        )
        nullspace_tolerance = jnp.asarray(
            512.0 * epsilon * dimension_scale * nullspace_scale,
            dtype=matrix.dtype,
        )
        lift_residual_norm = jnp.linalg.norm(lift_residual)
        nullspace_residual_norm = jnp.linalg.norm(nullspace_residual)
        if not bool(jnp.all(jnp.isfinite(particular))) or not bool(
            jnp.all(jnp.isfinite(nullspace_basis))
        ):
            raise RuntimeError(
                "Holomorphic constraint preparation produced nonfinite data."
            )
        if not bool(lift_residual_norm <= lift_tolerance):
            raise ValueError("Holomorphic polynomial constraints are inconsistent.")
        if not bool(nullspace_residual_norm <= nullspace_tolerance):
            raise RuntimeError(
                "Holomorphic constraint nullspace failed its residual check."
            )
        if not bool(result.successful):
            raise RuntimeError(
                "Holomorphic constraint minimum-norm solve did not converge."
            )
        evidence = HolomorphicConstraintEvidence(
            singular_values=factorization.singular_values(),
            lift_residual_norm=lift_residual_norm,
            nullspace_residual_norm=nullspace_residual_norm,
            lift_tolerance=lift_tolerance,
            nullspace_tolerance=nullspace_tolerance,
            rank=rank,
            nullity=nullity,
            factorization_id=factorization.factorization_id,
            plan_id=self.plan_id,
        )
        return PreparedHolomorphicPolynomialConstraints(
            self,
            constraint_matrix=matrix,
            target=target,
            particular_coefficients=particular,
            nullspace_basis=nullspace_basis,
            evidence=evidence,
        )


class PreparedHolomorphicPolynomialConstraints(StrictModule, NonTrainableState):
    """Prepared minimum-norm lift and nullspace map for polynomial coefficients."""

    plan: HolomorphicPolynomialConstraintPlan
    constraint_matrix: Array
    target: Array
    particular_coefficients: Array
    nullspace_basis: Array
    evidence: HolomorphicConstraintEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HolomorphicPolynomialConstraintPlan,
        /,
        *,
        constraint_matrix: ArrayLike,
        target: ArrayLike,
        particular_coefficients: ArrayLike,
        nullspace_basis: ArrayLike,
        evidence: HolomorphicConstraintEvidence,
    ):
        if not isinstance(plan, HolomorphicPolynomialConstraintPlan):
            raise TypeError("plan must be a HolomorphicPolynomialConstraintPlan.")
        if not isinstance(evidence, HolomorphicConstraintEvidence):
            raise TypeError("evidence must be HolomorphicConstraintEvidence.")
        matrix = jnp.asarray(constraint_matrix)
        target_ = jnp.asarray(target)
        particular = jnp.asarray(particular_coefficients)
        nullspace = jnp.asarray(nullspace_basis)
        expected_matrix_shape = (len(plan.constraints), plan.coefficient_count)
        if matrix.shape != expected_matrix_shape:
            raise ValueError("Prepared holomorphic constraint matrix has invalid shape.")
        if target_.shape != (len(plan.constraints),):
            raise ValueError("Prepared holomorphic constraint target has invalid shape.")
        if particular.shape != (plan.coefficient_count,):
            raise ValueError("Prepared holomorphic lift has invalid shape.")
        if nullspace.shape != (plan.coefficient_count, evidence.nullity):
            raise ValueError("Prepared holomorphic nullspace has invalid shape.")
        if any(
            jnp.iscomplexobj(value) for value in (matrix, target_, particular, nullspace)
        ):
            raise TypeError(
                "Prepared holomorphic coefficient maps must be real Cartesian."
            )
        if not all(
            bool(jnp.all(jnp.isfinite(value)))
            for value in (matrix, target_, particular, nullspace)
        ):
            raise ValueError("Prepared holomorphic coefficient maps must be finite.")
        self.plan = plan
        self.constraint_matrix = matrix
        self.target = target_
        self.particular_coefficients = particular
        self.nullspace_basis = nullspace
        self.evidence = evidence
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-holomorphic-polynomial-constraints",
                "plan": plan.plan_id,
                "evidence": evidence.evidence_id,
                "particular_coefficients": array_tree_fingerprint(particular),
                "nullspace_basis": array_tree_fingerprint(nullspace),
            }
        )

    def coefficient_vector(self, free_coordinates: ArrayLike, /) -> Array:
        """Map free real coordinates to all real Cartesian polynomial coefficients."""
        free = jnp.asarray(free_coordinates)
        if free.shape != (self.evidence.nullity,):
            raise ValueError(
                "Free holomorphic coordinates must match the prepared nullity."
            )
        if jnp.iscomplexobj(free):
            raise TypeError("Free holomorphic coordinates must be real.")
        dtype = jnp.result_type(
            free.dtype,
            self.particular_coefficients.dtype,
            self.nullspace_basis.dtype,
        )
        return self.particular_coefficients.astype(dtype) + self.nullspace_basis.astype(
            dtype
        ) @ free.astype(dtype)

    def residual(self, free_coordinates: ArrayLike, /) -> Array:
        """Evaluate the prepared finite constraint residual."""
        coefficients = self.coefficient_vector(free_coordinates)
        return self.constraint_matrix @ coefficients - self.target


class ConstrainedHolomorphicPolynomialPotential(StrictModule):
    """Scalar holomorphic polynomial parameterized inside an affine constraint set."""

    __hash__ = object.__hash__

    free_coordinates: Array
    constraints: PreparedHolomorphicPolynomialConstraints
    branches: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        constraints: PreparedHolomorphicPolynomialConstraints,
        /,
        *,
        initial_free_coordinates: ArrayLike | None = None,
    ):
        if not isinstance(constraints, PreparedHolomorphicPolynomialConstraints):
            raise TypeError(
                "constraints must be PreparedHolomorphicPolynomialConstraints."
            )
        free = (
            jnp.zeros(
                (constraints.evidence.nullity,),
                dtype=constraints.particular_coefficients.dtype,
            )
            if initial_free_coordinates is None
            else jnp.asarray(initial_free_coordinates)
        )
        if free.shape != (constraints.evidence.nullity,):
            raise ValueError("initial_free_coordinates must match constraint nullity.")
        if jnp.iscomplexobj(free):
            raise TypeError("initial_free_coordinates must be real Cartesian.")
        if not bool(jnp.all(jnp.isfinite(free))):
            raise ValueError("initial_free_coordinates must be finite.")
        homogeneous = bool(jnp.all(constraints.target == 0.0))
        coverage = "finite-subspace" if homogeneous else "finite-parametric-family"
        self.free_coordinates = free
        self.constraints = constraints
        self.branches = 1
        self.maximum_degree = constraints.plan.maximum_degree
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=1,
            complex_output_size=1,
            construction="affine-constrained-complex-polynomial-horner",
            normalization_id=constraints.plan.normalization.normalization_id,
            maximum_derivative_order=max(self.maximum_degree, 4),
            operations=(
                "real-affine-coefficient-map",
                "complex-affine",
                "complex-polynomial",
            ),
            parameter_mode="real-cartesian-nullspace",
            parameter_coverage=coverage,
            linear_in_parameters=homogeneous,
            construction_dependencies=(constraints.prepared_id,),
        )

    @property
    def coefficient_vector(self) -> Array:
        return self.constraints.coefficient_vector(self.free_coordinates)

    @property
    def coefficients(self) -> Array:
        count = self.maximum_degree + 1
        values = self.coefficient_vector
        return (values[:count] + 1j * values[count:]).reshape((1, count))

    def _normalized_scalar(self, coordinate: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinate)
        if value.shape == ():
            vector = value.reshape((1,))
        elif value.shape == (1,):
            vector = value
        else:
            raise ValueError(
                "ConstrainedHolomorphicPolynomialPotential expects one complex scalar."
            )
        return self.constraints.plan.normalization(vector)[0]

    def __call__(self, coordinate: ArrayLike, /) -> Array:
        return _horner(self.coefficients, self._normalized_scalar(coordinate))

    def jet(self, coordinate: ArrayLike, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if order_ < 0:
            raise ValueError("Holomorphic jet order must be nonnegative.")
        normalized = self._normalized_scalar(coordinate)
        scale = self.constraints.plan.normalization.matrix[0, 0]
        coefficients = self.coefficients
        value = _horner(coefficients, normalized)
        derivatives = tuple(
            _horner(_derivative_coefficients(coefficients, current), normalized)
            * scale**current
            for current in range(1, order_ + 1)
        )
        return HolomorphicJet(value, derivatives)

    def constraint_residual(self) -> Array:
        """Evaluate every prepared finite coefficient constraint."""
        return self.constraints.residual(self.free_coordinates)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


__all__ = [
    "ConstrainedHolomorphicPolynomialPotential",
    "HolomorphicConstraintComponent",
    "HolomorphicConstraintEvidence",
    "HolomorphicPointConstraint",
    "HolomorphicPolynomialConstraintPlan",
    "PreparedHolomorphicPolynomialConstraints",
]

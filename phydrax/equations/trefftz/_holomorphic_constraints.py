#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product
from operator import index
from typing import Any, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._holomorphic import (
    HolomorphicJet,
    HolomorphicMapCertificate,
    HolomorphicPotentialProvider,
)
from ..._holomorphic_linear import (
    HolomorphicLinearFrame,
    HolomorphicLinearFrameCertificate,
    HolomorphicMultiIndexSet,
    HolomorphicMultiJet,
    MultivariableHolomorphicPotentialProvider,
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


HolomorphicConstraintComponent = Literal["real", "imaginary"]


def _finite_real_scalar(value: ArrayLike, name: str, /) -> float:
    raw = np.asarray(value)
    if raw.shape != () or np.iscomplexobj(raw):
        raise TypeError(f"{name} must be one real scalar.")
    resolved = float(raw)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _coordinate_vector(value: ArrayLike, /) -> Array:
    raw = np.asarray(value, dtype=np.complex128)
    if raw.shape == ():
        raw = raw.reshape((1,))
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError(
            "Holomorphic point coordinates must be one finite complex vector."
        )
    return jnp.asarray(raw)


def _component_weight(component: HolomorphicConstraintComponent, /) -> complex:
    if component == "real":
        return 1.0 + 0.0j
    if component == "imaginary":
        return -1.0j
    raise ValueError("Holomorphic constraint component must be real or imaginary.")


def _canonical_columns(values: np.ndarray, /) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


class HolomorphicJetFunctionalTerm(StrictModule, NonTrainableState):
    """One weighted real part of one complex output derivative."""

    weight: Array
    output_index: int = eqx.field(static=True)
    derivative_multi_index: tuple[int, ...] = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        output_index: int,
        derivative_multi_index: Sequence[int],
        weight: ArrayLike = 1.0 + 0.0j,
        /,
    ):
        output = int(output_index)
        derivative = tuple(int(item) for item in derivative_multi_index)
        weight_raw = np.asarray(weight)
        if output < 0 or not derivative or any(item < 0 for item in derivative):
            raise ValueError("Holomorphic functional term indices are invalid.")
        if weight_raw.shape != ():
            raise ValueError("Holomorphic functional term weight must be scalar.")
        weight_ = complex(weight_raw)
        if not math.isfinite(abs(weight_)) or weight_ == 0.0j:
            raise ValueError(
                "Holomorphic functional term weight must be finite and nonzero."
            )
        weight_array = jnp.asarray(weight_, dtype=jnp.complex128)
        self.weight = weight_array
        self.output_index = output
        self.derivative_multi_index = derivative
        self.term_id = canonical_fingerprint(
            {
                "kind": "holomorphic-jet-functional-term",
                "output_index": output,
                "derivative_multi_index": list(derivative),
                "weight": array_tree_fingerprint(weight_array),
            }
        )


@runtime_checkable
class HolomorphicLinearFunctional(Protocol):
    functional_id: str

    def assemble_row(self, frame: HolomorphicLinearFrame, /) -> Array: ...


class HolomorphicPointFunctional(StrictModule, NonTrainableState):
    """One explicit real-linear functional of holomorphic point jets."""

    coordinate: Array
    terms: tuple[HolomorphicJetFunctionalTerm, ...]
    construction: str = eqx.field(static=True)
    construction_dependencies: tuple[str, ...] = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: ArrayLike,
        terms: Sequence[HolomorphicJetFunctionalTerm],
        /,
        *,
        construction: str = "complex-jet-real-linear-functional",
        construction_dependencies: Sequence[str] = (),
    ):
        coordinate_ = _coordinate_vector(coordinate)
        terms_ = tuple(terms)
        if not terms_ or not all(
            isinstance(term, HolomorphicJetFunctionalTerm) for term in terms_
        ):
            raise TypeError("terms must contain HolomorphicJetFunctionalTerm values.")
        if any(
            len(term.derivative_multi_index) != int(coordinate_.size) for term in terms_
        ):
            raise ValueError(
                "Functional derivative dimensions must match the coordinate."
            )
        construction_ = str(construction)
        dependencies = tuple(str(value) for value in construction_dependencies)
        if not construction_ or any(not value for value in dependencies):
            raise ValueError("Holomorphic functional identifiers must be nonempty.")
        self.coordinate = coordinate_
        self.terms = terms_
        self.construction = construction_
        self.construction_dependencies = dependencies
        self.functional_id = canonical_fingerprint(
            {
                "kind": "holomorphic-point-functional",
                "coordinate": array_tree_fingerprint(coordinate_),
                "terms": [term.term_id for term in terms_],
                "construction": construction_,
                "construction_dependencies": list(dependencies),
            }
        )

    @classmethod
    def value(
        cls,
        coordinate: ArrayLike,
        /,
        *,
        component: HolomorphicConstraintComponent = "real",
        output_index: int = 0,
        weight: float = 1.0,
    ) -> HolomorphicPointFunctional:
        coordinates = _coordinate_vector(coordinate)
        scale = _finite_real_scalar(weight, "weight")
        return cls(
            coordinates,
            (
                HolomorphicJetFunctionalTerm(
                    output_index,
                    (0,) * int(coordinates.size),
                    scale * _component_weight(component),
                ),
            ),
        )

    @classmethod
    def normal_derivative(
        cls,
        coordinate: ArrayLike,
        normal: ArrayLike,
        /,
        *,
        component: HolomorphicConstraintComponent = "real",
        output_index: int = 0,
        weight: float = 1.0,
    ) -> HolomorphicPointFunctional:
        coordinates = _coordinate_vector(coordinate)
        dimension = int(coordinates.size)
        normal_raw = np.asarray(normal)
        if normal_raw.shape != (2 * dimension,) or np.iscomplexobj(normal_raw):
            raise TypeError(f"Holomorphic normal must be real shape ({2 * dimension},).")
        normal_ = np.asarray(normal_raw, dtype=np.float64)
        if not np.all(np.isfinite(normal_)) or not np.any(normal_ != 0.0):
            raise ValueError("Holomorphic normal must be finite and nonzero.")
        scale = _finite_real_scalar(weight, "weight")
        component_weight = _component_weight(component)
        terms = []
        for axis in range(dimension):
            derivative = tuple(
                1 if current == axis else 0 for current in range(dimension)
            )
            direction = complex(normal_[axis], normal_[dimension + axis])
            if direction != 0.0j:
                terms.append(
                    HolomorphicJetFunctionalTerm(
                        output_index,
                        derivative,
                        scale * component_weight * direction,
                    )
                )
        return cls(coordinates, terms)

    @classmethod
    def robin(
        cls,
        coordinate: ArrayLike,
        normal: ArrayLike,
        /,
        *,
        value_weight: float,
        normal_weight: float,
        component: HolomorphicConstraintComponent = "real",
        output_index: int = 0,
    ) -> HolomorphicPointFunctional:
        value_scale = _finite_real_scalar(value_weight, "value_weight")
        normal_scale = _finite_real_scalar(normal_weight, "normal_weight")
        if value_scale == 0.0 and normal_scale == 0.0:
            raise ValueError("Robin functionals require one nonzero weight.")
        coordinates = _coordinate_vector(coordinate)
        terms: list[HolomorphicJetFunctionalTerm] = []
        if value_scale != 0.0:
            terms.extend(
                cls.value(
                    coordinates,
                    component=component,
                    output_index=output_index,
                    weight=value_scale,
                ).terms
            )
        if normal_scale != 0.0:
            terms.extend(
                cls.normal_derivative(
                    coordinates,
                    normal,
                    component=component,
                    output_index=output_index,
                    weight=normal_scale,
                ).terms
            )
        return cls(coordinates, terms)

    def assemble_row(self, frame: HolomorphicLinearFrame, /) -> Array:
        if not isinstance(frame, HolomorphicLinearFrame):
            raise TypeError("frame must implement HolomorphicLinearFrame.")
        certificate = frame.linear_frame_certificate()
        if int(self.coordinate.size) != certificate.complex_input_size:
            raise ValueError("Functional and holomorphic frame input dimensions differ.")
        row = jnp.zeros((certificate.real_coefficient_count,), dtype=jnp.float64)
        for term in self.terms:
            if term.output_index >= certificate.complex_output_size:
                raise ValueError("Functional output index exceeds the frame output size.")
            if sum(term.derivative_multi_index) > certificate.maximum_derivative_order:
                raise ValueError(
                    "Functional derivative order exceeds the frame evidence."
                )
            basis = frame.basis_derivative(
                self.coordinate,
                term.derivative_multi_index,
            )
            if basis.shape != (
                certificate.complex_output_size,
                certificate.real_coefficient_count,
            ):
                raise ValueError("Holomorphic frame returned an invalid basis shape.")
            row = row + jnp.real(term.weight * basis[term.output_index])
        return row

    def evaluate_provider(self, provider: Any, /) -> Array:
        dimension = int(self.coordinate.size)
        if dimension == 1:
            if not isinstance(provider, HolomorphicPotentialProvider):
                raise TypeError("Scalar functional provider lacks holomorphic jets.")
            maximum = max(sum(term.derivative_multi_index) for term in self.terms)
            jet = provider.jet(self.coordinate[0], maximum)
            result = jnp.asarray(0.0, dtype=jnp.result_type(jet.value.real, float))
            for term in self.terms:
                derivative = jet.derivative(term.derivative_multi_index[0])
                result = result + jnp.real(term.weight * derivative[term.output_index])
            return result
        if not isinstance(provider, MultivariableHolomorphicPotentialProvider):
            raise TypeError(
                "Multivariable functional provider lacks holomorphic multijets."
            )
        indices = {(0,) * dimension}
        for term in self.terms:
            indices.update(
                tuple(value)
                for value in product(
                    *(range(maximum + 1) for maximum in term.derivative_multi_index)
                )
            )
        index_set = HolomorphicMultiIndexSet(
            dimension,
            tuple(indices),
            require_downward_closed=True,
        )
        jet = provider.multi_jet(self.coordinate, index_set)
        result = jnp.asarray(0.0, dtype=jnp.result_type(jet.value.real, float))
        for term in self.terms:
            derivative = jet.derivative(term.derivative_multi_index)
            result = result + jnp.real(term.weight * derivative[term.output_index])
        return result


class HolomorphicConstraintOperatorEvidence(StrictModule, NonTrainableState):
    """Rank, right-inverse, and nullspace evidence for a functional operator."""

    singular_values: Array
    right_inverse_residual_norm: Array
    nullspace_residual_norm: Array
    right_inverse_tolerance: Array
    nullspace_tolerance: Array
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)
    factorization_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        singular_values: ArrayLike,
        right_inverse_residual_norm: ArrayLike,
        nullspace_residual_norm: ArrayLike,
        right_inverse_tolerance: ArrayLike,
        nullspace_tolerance: ArrayLike,
        rank: int,
        nullity: int,
        factorization_id: str,
        plan_id: str,
    ):
        singular_values_ = jnp.asarray(singular_values)
        scalars = tuple(
            jnp.asarray(value)
            for value in (
                right_inverse_residual_norm,
                nullspace_residual_norm,
                right_inverse_tolerance,
                nullspace_tolerance,
            )
        )
        if singular_values_.ndim != 1 or any(value.shape != () for value in scalars):
            raise ValueError("Constraint operator evidence has invalid array shapes.")
        if not bool(jnp.all(jnp.isfinite(singular_values_))) or any(
            not bool(jnp.isfinite(value)) for value in scalars
        ):
            raise ValueError("Constraint operator evidence must be finite.")
        rank_ = int(rank)
        nullity_ = int(nullity)
        factorization_id_ = str(factorization_id)
        plan_id_ = str(plan_id)
        if rank_ < 0 or nullity_ < 0 or not factorization_id_ or not plan_id_:
            raise ValueError("Constraint operator evidence metadata is invalid.")
        self.singular_values = singular_values_
        (
            self.right_inverse_residual_norm,
            self.nullspace_residual_norm,
            self.right_inverse_tolerance,
            self.nullspace_tolerance,
        ) = scalars
        self.rank = rank_
        self.nullity = nullity_
        self.factorization_id = factorization_id_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "holomorphic-constraint-operator-evidence",
                "plan": plan_id_,
                "factorization": factorization_id_,
                "rank": rank_,
                "nullity": nullity_,
                "singular_values": array_tree_fingerprint(singular_values_),
                "right_inverse_residual": array_tree_fingerprint(scalars[0]),
                "nullspace_residual": array_tree_fingerprint(scalars[1]),
                "right_inverse_tolerance": array_tree_fingerprint(scalars[2]),
                "nullspace_tolerance": array_tree_fingerprint(scalars[3]),
            }
        )


class HolomorphicConstraintOperatorPlan(StrictModule, NonTrainableState):
    """Target-independent real-linear constraints on a holomorphic frame."""

    frame: Any
    functionals: tuple[Any, ...]
    rank_cutoff: float | None = eqx.field(static=True)
    maximum_factor_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: HolomorphicLinearFrame,
        functionals: Sequence[HolomorphicLinearFunctional],
        /,
        *,
        rank_cutoff: float | None = None,
        maximum_factor_bytes: int = 512 * 1024**2,
        maximum_workspace_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(frame, HolomorphicLinearFrame):
            raise TypeError("frame must implement HolomorphicLinearFrame.")
        functionals_ = tuple(functionals)
        if not functionals_ or not all(
            isinstance(functional, HolomorphicLinearFunctional)
            for functional in functionals_
        ):
            raise TypeError("functionals must implement HolomorphicLinearFunctional.")
        cutoff = None if rank_cutoff is None else float(rank_cutoff)
        if cutoff is not None and (not math.isfinite(cutoff) or cutoff < 0.0):
            raise ValueError("rank_cutoff must be nonnegative and finite or None.")
        if isinstance(maximum_factor_bytes, bool) or isinstance(
            maximum_workspace_bytes, bool
        ):
            raise TypeError("Constraint operator resource budgets must be integers.")
        factor_bytes = index(maximum_factor_bytes)
        workspace_bytes = index(maximum_workspace_bytes)
        if factor_bytes <= 0 or workspace_bytes <= 0:
            raise ValueError("Constraint operator resource budgets must be positive.")
        certificate = frame.linear_frame_certificate()
        self.frame = frame
        self.functionals = functionals_
        self.rank_cutoff = cutoff
        self.maximum_factor_bytes = factor_bytes
        self.maximum_workspace_bytes = workspace_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "holomorphic-constraint-operator-plan",
                "frame": certificate.frame_id,
                "functionals": [functional.functional_id for functional in functionals_],
                "rank_cutoff": cutoff,
                "maximum_factor_bytes": factor_bytes,
                "maximum_workspace_bytes": workspace_bytes,
            }
        )

    def prepare(self, /) -> PreparedHolomorphicConstraintOperator:
        matrix = jnp.stack(
            tuple(functional.assemble_row(self.frame) for functional in self.functionals)
        ).astype(jnp.float64)
        certificate = self.frame.linear_frame_certificate()
        row_norms = jnp.linalg.norm(matrix, axis=1)
        row_tolerance = (
            256.0
            * jnp.finfo(matrix.dtype).eps
            * max(certificate.real_coefficient_count, 1)
            * jnp.maximum(jnp.linalg.norm(matrix), 1.0)
        )
        if bool(jnp.any(row_norms <= row_tolerance)):
            raise ValueError(
                "At least one functional is identically zero on the holomorphic frame."
            )
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
        target_count = len(self.functionals)
        identity = jnp.eye(target_count, dtype=matrix.dtype)
        right_columns = []
        for column in range(target_count):
            result = factorization.solve(identity[:, column])
            value = jnp.asarray(result.value, dtype=matrix.dtype)
            if not bool(jnp.all(jnp.isfinite(value))):
                raise RuntimeError(
                    "Constraint right-inverse solve produced nonfinite data."
                )
            right_columns.append(value)
        right_inverse = jnp.stack(tuple(right_columns), axis=1)
        rank = int(np.asarray(factorization.rank()))
        nullity = certificate.real_coefficient_count - rank
        nullspace = factorization.right_nullspace()
        if int(np.asarray(nullspace.dimension)) != nullity:
            raise RuntimeError("Constraint nullspace dimension is inconsistent.")
        nullspace_basis = jnp.asarray(
            _canonical_columns(np.asarray(nullspace.basis[:, :nullity])),
            dtype=matrix.dtype,
        )
        right_residual = matrix @ right_inverse @ matrix - matrix
        nullspace_residual = matrix @ nullspace_basis
        epsilon = jnp.finfo(matrix.dtype).eps
        dimension_scale = max(matrix.shape)
        right_tolerance = (
            512.0
            * epsilon
            * dimension_scale
            * jnp.maximum(
                jnp.linalg.norm(matrix)
                * jnp.maximum(jnp.linalg.norm(right_inverse @ matrix), 1.0),
                1.0,
            )
        )
        nullspace_tolerance = (
            512.0
            * epsilon
            * dimension_scale
            * jnp.maximum(
                jnp.linalg.norm(matrix)
                * jnp.maximum(jnp.linalg.norm(nullspace_basis), 1.0),
                1.0,
            )
        )
        right_residual_norm = jnp.linalg.norm(right_residual)
        nullspace_residual_norm = jnp.linalg.norm(nullspace_residual)
        if not bool(right_residual_norm <= right_tolerance):
            raise RuntimeError("Constraint right inverse failed its residual check.")
        if not bool(nullspace_residual_norm <= nullspace_tolerance):
            raise RuntimeError("Constraint nullspace failed its residual check.")
        evidence = HolomorphicConstraintOperatorEvidence(
            singular_values=factorization.singular_values(),
            right_inverse_residual_norm=right_residual_norm,
            nullspace_residual_norm=nullspace_residual_norm,
            right_inverse_tolerance=right_tolerance,
            nullspace_tolerance=nullspace_tolerance,
            rank=rank,
            nullity=nullity,
            factorization_id=factorization.factorization_id,
            plan_id=self.plan_id,
        )
        return PreparedHolomorphicConstraintOperator(
            self,
            constraint_matrix=matrix,
            right_inverse=right_inverse,
            nullspace_basis=nullspace_basis,
            evidence=evidence,
        )


class PreparedHolomorphicConstraintOperator(StrictModule, NonTrainableState):
    """Reusable right inverse and nullspace for one functional operator."""

    plan: HolomorphicConstraintOperatorPlan
    constraint_matrix: Array
    right_inverse: Array
    nullspace_basis: Array
    evidence: HolomorphicConstraintOperatorEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HolomorphicConstraintOperatorPlan,
        /,
        *,
        constraint_matrix: ArrayLike,
        right_inverse: ArrayLike,
        nullspace_basis: ArrayLike,
        evidence: HolomorphicConstraintOperatorEvidence,
    ):
        if not isinstance(plan, HolomorphicConstraintOperatorPlan):
            raise TypeError("plan must be HolomorphicConstraintOperatorPlan.")
        if not isinstance(evidence, HolomorphicConstraintOperatorEvidence):
            raise TypeError("evidence must be HolomorphicConstraintOperatorEvidence.")
        matrix = jnp.asarray(constraint_matrix)
        right = jnp.asarray(right_inverse)
        nullspace = jnp.asarray(nullspace_basis)
        coefficient_count = plan.frame.linear_frame_certificate().real_coefficient_count
        target_count = len(plan.functionals)
        if matrix.shape != (target_count, coefficient_count):
            raise ValueError("Prepared constraint matrix has invalid shape.")
        if right.shape != (coefficient_count, target_count):
            raise ValueError("Prepared constraint right inverse has invalid shape.")
        if nullspace.shape != (coefficient_count, evidence.nullity):
            raise ValueError("Prepared constraint nullspace has invalid shape.")
        if any(jnp.iscomplexobj(value) for value in (matrix, right, nullspace)):
            raise TypeError("Prepared constraint operators must be real Cartesian.")
        self.plan = plan
        self.constraint_matrix = matrix
        self.right_inverse = right
        self.nullspace_basis = nullspace
        self.evidence = evidence
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-holomorphic-constraint-operator",
                "plan": plan.plan_id,
                "evidence": evidence.evidence_id,
                "constraint_matrix": array_tree_fingerprint(matrix),
                "right_inverse": array_tree_fingerprint(right),
                "nullspace_basis": array_tree_fingerprint(nullspace),
            }
        )

    @property
    def target_count(self) -> int:
        return int(self.constraint_matrix.shape[0])

    def minimum_norm_coefficients(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets)
        if values.shape[-1:] != (self.target_count,):
            raise ValueError("Constraint targets must end with the functional count.")
        if jnp.iscomplexobj(values):
            raise TypeError("Constraint targets must be real.")
        return values @ jnp.swapaxes(self.right_inverse, -1, -2)

    def target_residual(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets)
        coefficients = self.minimum_norm_coefficients(values)
        return coefficients @ jnp.swapaxes(self.constraint_matrix, -1, -2) - values

    def affine_map(self, target: ArrayLike, /) -> HolomorphicAffineCoefficientMap:
        return HolomorphicAffineCoefficientMap(self, target)


class HolomorphicConstraintLiftEvidence(StrictModule, NonTrainableState):
    """Consistency evidence for one target-specific minimum-norm lift."""

    residual_norm: Array
    tolerance: Array
    consistent: Array
    lift_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_norm: ArrayLike,
        tolerance: ArrayLike,
        target: ArrayLike,
        prepared_id: str,
    ):
        residual = jnp.asarray(residual_norm)
        tolerance_ = jnp.asarray(tolerance)
        if residual.shape != () or tolerance_.shape != ():
            raise ValueError("Constraint lift evidence values must be scalar.")
        consistent = jnp.isfinite(residual) & (residual <= tolerance_)
        self.residual_norm = residual
        self.tolerance = tolerance_
        self.consistent = consistent
        self.lift_id = canonical_fingerprint(
            {
                "kind": "holomorphic-constraint-lift-evidence",
                "prepared_operator": str(prepared_id),
                "target": array_tree_fingerprint(jnp.asarray(target)),
                "residual_norm": array_tree_fingerprint(residual),
                "tolerance": array_tree_fingerprint(tolerance_),
            }
        )


class HolomorphicAffineCoefficientMap(StrictModule, NonTrainableState):
    """One target-specific lift plus the reusable homogeneous nullspace."""

    operator: PreparedHolomorphicConstraintOperator
    target: Array
    particular_coefficients: Array
    evidence: HolomorphicConstraintLiftEvidence
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: PreparedHolomorphicConstraintOperator,
        target: ArrayLike,
        /,
    ):
        if not isinstance(operator, PreparedHolomorphicConstraintOperator):
            raise TypeError("operator must be PreparedHolomorphicConstraintOperator.")
        target_ = jnp.asarray(target)
        if target_.shape != (operator.target_count,) or jnp.iscomplexobj(target_):
            raise ValueError(
                "Affine constraint target must be one real functional vector."
            )
        particular = operator.minimum_norm_coefficients(target_)
        residual = operator.constraint_matrix @ particular - target_
        residual_norm = jnp.linalg.norm(residual)
        epsilon = jnp.finfo(particular.dtype).eps
        scale = jnp.maximum(
            jnp.linalg.norm(operator.constraint_matrix)
            * jnp.maximum(jnp.linalg.norm(particular), 1.0),
            jnp.maximum(jnp.linalg.norm(target_), 1.0),
        )
        tolerance = 512.0 * epsilon * max(operator.constraint_matrix.shape) * scale
        evidence = HolomorphicConstraintLiftEvidence(
            residual_norm=residual_norm,
            tolerance=tolerance,
            target=target_,
            prepared_id=operator.prepared_id,
        )
        if not bool(evidence.consistent):
            raise ValueError("Holomorphic constraint target is inconsistent.")
        self.operator = operator
        self.target = target_
        self.particular_coefficients = particular
        self.evidence = evidence
        self.map_id = canonical_fingerprint(
            {
                "kind": "holomorphic-affine-coefficient-map",
                "operator": operator.prepared_id,
                "lift": evidence.lift_id,
                "particular_coefficients": array_tree_fingerprint(particular),
            }
        )

    @property
    def nullity(self) -> int:
        return self.operator.evidence.nullity

    def coefficient_vector(self, free_coordinates: ArrayLike, /) -> Array:
        free = jnp.asarray(free_coordinates)
        if free.shape != (self.nullity,) or jnp.iscomplexobj(free):
            raise ValueError("Free coordinates must be one real nullspace vector.")
        return self.particular_coefficients + self.operator.nullspace_basis @ free

    def residual(self, free_coordinates: ArrayLike, /) -> Array:
        coefficients = self.coefficient_vector(free_coordinates)
        return self.operator.constraint_matrix @ coefficients - self.target


class ConstrainedHolomorphicPotential(StrictModule):
    """Holomorphic frame parameterized inside one affine coefficient set."""

    __hash__ = object.__hash__

    free_coordinates: Array
    coefficient_map: HolomorphicAffineCoefficientMap
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        coefficient_map: HolomorphicAffineCoefficientMap,
        /,
        *,
        initial_free_coordinates: ArrayLike | None = None,
    ):
        if not isinstance(coefficient_map, HolomorphicAffineCoefficientMap):
            raise TypeError("coefficient_map must be HolomorphicAffineCoefficientMap.")
        free = (
            jnp.zeros(
                (coefficient_map.nullity,),
                dtype=coefficient_map.particular_coefficients.dtype,
            )
            if initial_free_coordinates is None
            else jnp.asarray(initial_free_coordinates)
        )
        if free.shape != (coefficient_map.nullity,) or jnp.iscomplexobj(free):
            raise ValueError(
                "Initial free coordinates must be one real nullspace vector."
            )
        if not bool(jnp.all(jnp.isfinite(free))):
            raise ValueError("Initial free coordinates must be finite.")
        frame = coefficient_map.operator.plan.frame
        frame_certificate = frame.linear_frame_certificate()
        if not isinstance(frame_certificate, HolomorphicLinearFrameCertificate):
            raise TypeError(
                "ConstrainedHolomorphicPotential requires a globally holomorphic frame."
            )
        homogeneous = bool(jnp.all(coefficient_map.target == 0.0))
        self.free_coordinates = free
        self.coefficient_map = coefficient_map
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=frame_certificate.complex_input_size,
            complex_output_size=frame_certificate.complex_output_size,
            construction="affine-constrained-holomorphic-linear-frame",
            normalization_id=frame_certificate.normalization_id,
            maximum_derivative_order=frame_certificate.maximum_derivative_order,
            operations=("real-affine-coefficient-map", "holomorphic-linear-frame"),
            parameter_mode="real-cartesian-nullspace",
            parameter_coverage=(
                "finite-subspace" if homogeneous else "finite-parametric-family"
            ),
            linear_in_parameters=homogeneous,
            construction_dependencies=(
                frame_certificate.frame_id,
                coefficient_map.map_id,
            ),
        )

    @property
    def frame(self) -> HolomorphicLinearFrame:
        return self.coefficient_map.operator.plan.frame

    @property
    def coefficient_vector(self) -> Array:
        return self.coefficient_map.coefficient_vector(self.free_coordinates)

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        dimension = self._certificate.complex_input_size
        basis = self.frame.basis_derivative(coordinates, (0,) * dimension)
        return basis @ self.coefficient_vector

    def jet(self, coordinate: ArrayLike, order: int, /) -> HolomorphicJet:
        if self._certificate.complex_input_size != 1:
            raise ValueError("Scalar holomorphic jets require one complex input.")
        order_ = int(order)
        if order_ < 0 or order_ > self._certificate.maximum_derivative_order:
            raise ValueError("Requested holomorphic jet order is unavailable.")
        value = self(coordinate)
        derivatives = tuple(
            self.frame.basis_derivative(coordinate, (current,)) @ self.coefficient_vector
            for current in range(1, order_ + 1)
        )
        return HolomorphicJet(value, derivatives)

    def multi_jet(
        self,
        coordinates: ArrayLike,
        index_set: HolomorphicMultiIndexSet,
        /,
    ) -> HolomorphicMultiJet:
        if not isinstance(index_set, HolomorphicMultiIndexSet):
            raise TypeError("index_set must be HolomorphicMultiIndexSet.")
        if index_set.complex_dimension != self._certificate.complex_input_size:
            raise ValueError("Multijet and constrained potential dimensions differ.")
        if index_set.maximum_total_order > self._certificate.maximum_derivative_order:
            raise ValueError("Requested holomorphic multijet order is unavailable.")
        coefficients = self.coefficient_vector
        value = (
            self.frame.basis_derivative(
                coordinates,
                (0,) * index_set.complex_dimension,
            )
            @ coefficients
        )
        derivatives = tuple(
            self.frame.basis_derivative(coordinates, derivative) @ coefficients
            for derivative in index_set.nonzero_indices
        )
        return HolomorphicMultiJet(value, derivatives, index_set)

    def constraint_residual(self) -> Array:
        return self.coefficient_map.residual(self.free_coordinates)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


class HolomorphicProjectionState(StrictModule):
    """Parameter-dependent correction coefficients reusable across query points."""

    functional_values: Array
    correction_coefficients: Array

    def __init__(
        self,
        functional_values: ArrayLike,
        correction_coefficients: ArrayLike,
        /,
    ):
        values = jnp.asarray(functional_values)
        correction = jnp.asarray(correction_coefficients)
        if values.ndim != 1 or correction.ndim != 1:
            raise ValueError("Holomorphic projection state arrays must be vectors.")
        self.functional_values = values
        self.correction_coefficients = correction


class HolomorphicConstraintProjector(StrictModule, NonTrainableState):
    """Fixed cardinal lift for exact finite constraints on nonlinear providers."""

    operator: PreparedHolomorphicConstraintOperator
    projector_id: str = eqx.field(static=True)

    def __init__(self, operator: PreparedHolomorphicConstraintOperator, /):
        if not isinstance(operator, PreparedHolomorphicConstraintOperator):
            raise TypeError("operator must be PreparedHolomorphicConstraintOperator.")
        if operator.evidence.rank != operator.target_count:
            raise ValueError("Cardinal projection requires full row rank.")
        self.operator = operator
        self.projector_id = canonical_fingerprint(
            {
                "kind": "holomorphic-constraint-projector",
                "operator": operator.prepared_id,
            }
        )

    def project(
        self,
        provider: HolomorphicPotentialProvider,
        target: ArrayLike,
        /,
    ) -> ProjectedHolomorphicPotential:
        return ProjectedHolomorphicPotential(
            provider,
            self,
            self.operator.affine_map(target),
        )


class ProjectedHolomorphicPotential(StrictModule):
    """Certified provider corrected by one fixed holomorphic cardinal lift."""

    __hash__ = object.__hash__

    provider: Any
    projector: HolomorphicConstraintProjector
    coefficient_map: HolomorphicAffineCoefficientMap
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        provider: HolomorphicPotentialProvider,
        projector: HolomorphicConstraintProjector,
        coefficient_map: HolomorphicAffineCoefficientMap,
        /,
    ):
        if not isinstance(provider, HolomorphicPotentialProvider):
            raise TypeError("provider must implement HolomorphicPotentialProvider.")
        if not isinstance(projector, HolomorphicConstraintProjector):
            raise TypeError("projector must be HolomorphicConstraintProjector.")
        if coefficient_map.operator.prepared_id != projector.operator.prepared_id:
            raise ValueError("Projection target map and projector are incompatible.")
        child = provider.holomorphic_certificate()
        frame = projector.operator.plan.frame.linear_frame_certificate()
        if (
            child.complex_input_size != frame.complex_input_size
            or child.complex_output_size != frame.complex_output_size
        ):
            raise ValueError("Projection provider and cardinal frame dimensions differ.")
        maximum_functional_order = max(
            sum(term.derivative_multi_index)
            for functional in projector.operator.plan.functionals
            for term in functional.terms
        )
        if maximum_functional_order > child.maximum_derivative_order:
            raise ValueError("Projection provider lacks required functional derivatives.")
        homogeneous = bool(jnp.all(coefficient_map.target == 0.0))
        linear = homogeneous and child.linear_in_parameters
        self.provider = provider
        self.projector = projector
        self.coefficient_map = coefficient_map
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=child.complex_input_size,
            complex_output_size=child.complex_output_size,
            construction="cardinal-projected-holomorphic-provider",
            normalization_id=canonical_fingerprint(
                {
                    "kind": "projected-holomorphic-normalization",
                    "child": child.normalization_id,
                    "frame": frame.normalization_id,
                }
            ),
            maximum_derivative_order=min(
                child.maximum_derivative_order,
                frame.maximum_derivative_order,
            ),
            operations=tuple(dict.fromkeys(child.operations))
            + ("holomorphic-cardinal-lift",),
            parameter_mode=child.parameter_mode,
            parameter_coverage=(
                "finite-subspace"
                if linear and child.parameter_coverage == "finite-subspace"
                else "finite-parametric-family"
            ),
            linear_in_parameters=linear,
            construction_dependencies=(
                child.certificate_id,
                projector.projector_id,
                coefficient_map.map_id,
            ),
        )

    @property
    def frame(self) -> HolomorphicLinearFrame:
        return self.projector.operator.plan.frame

    def prepare_projection(self, /) -> HolomorphicProjectionState:
        values = jnp.stack(
            tuple(
                functional.evaluate_provider(self.provider)
                for functional in self.projector.operator.plan.functionals
            )
        )
        residual = self.coefficient_map.target - values
        correction = self.projector.operator.minimum_norm_coefficients(residual)
        return HolomorphicProjectionState(values, correction)

    def evaluate_with_state(
        self,
        coordinates: ArrayLike,
        state: HolomorphicProjectionState,
        /,
    ) -> Array:
        if not isinstance(state, HolomorphicProjectionState):
            raise TypeError("state must be HolomorphicProjectionState.")
        dimension = self._certificate.complex_input_size
        basis = self.frame.basis_derivative(coordinates, (0,) * dimension)
        return self.provider(coordinates) + basis @ state.correction_coefficients

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        return self.evaluate_with_state(coordinates, self.prepare_projection())

    def jet(self, coordinate: ArrayLike, order: int, /) -> HolomorphicJet:
        if self._certificate.complex_input_size != 1:
            raise ValueError("Scalar projected jets require one complex input.")
        state = self.prepare_projection()
        child = self.provider.jet(coordinate, order)
        derivatives = tuple(
            child.derivative(current)
            + self.frame.basis_derivative(coordinate, (current,))
            @ state.correction_coefficients
            for current in range(1, order + 1)
        )
        return HolomorphicJet(
            self.evaluate_with_state(coordinate, state),
            derivatives,
        )

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


__all__ = [
    "ConstrainedHolomorphicPotential",
    "HolomorphicAffineCoefficientMap",
    "HolomorphicConstraintComponent",
    "HolomorphicConstraintLiftEvidence",
    "HolomorphicConstraintOperatorEvidence",
    "HolomorphicConstraintOperatorPlan",
    "HolomorphicConstraintProjector",
    "HolomorphicJetFunctionalTerm",
    "HolomorphicLinearFunctional",
    "HolomorphicPointFunctional",
    "HolomorphicProjectionState",
    "PreparedHolomorphicConstraintOperator",
    "ProjectedHolomorphicPotential",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    FunctionLinearOperator,
    saddle_point_operator,
    ScaledLinearOperator,
)
from .._cell_mesh import CellMesh
from ._generic import (
    FiniteElementCoordinateSpec,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ._reference import lagrange_element


if TYPE_CHECKING:
    from ...equations import CompiledFiniteElementProblem, FiniteElementForm


PressureGaugeMode: TypeAlias = Literal["mean-zero", "pinned", "none"]
MixedConstraintFormulation: TypeAlias = Literal["exact", "finite-bulk"]
MixedPressureStabilizationKind: TypeAlias = Literal[
    "none",
    "pressure-laplacian",
]


class PressureGaugeEvidence(StrictModule):
    residual: Array
    scale: Array
    finite: Array
    valid: Array
    mode: PressureGaugeMode = eqx.field(static=True)


class PressureGaugePolicy(StrictModule, NonTrainableState):
    """Explicit pressure gauge independent of the eventual pressure-space size."""

    weights: Array | None
    pinned_dof: int | None = eqx.field(static=True)
    mode: PressureGaugeMode = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: PressureGaugeMode,
        /,
        *,
        weights: ArrayLike | None = None,
        pinned_dof: int | None = None,
        tolerance: float = 1.0e-10,
    ):
        mode_ = str(mode)
        if mode_ not in ("mean-zero", "pinned", "none"):
            raise ValueError("Pressure gauge mode must be mean-zero, pinned, or none.")
        limit = float(tolerance)
        if not isfinite(limit) or limit < 0.0:
            raise ValueError("Pressure gauge tolerance must be finite and nonnegative.")
        if mode_ == "mean-zero":
            if pinned_dof is not None:
                raise ValueError("Mean-zero pressure gauges do not accept pinned_dof.")
            weights_ = None if weights is None else np.asarray(weights, dtype=float)
            if weights_ is not None and (
                weights_.ndim != 1
                or np.any(~np.isfinite(weights_))
                or np.any(weights_ <= 0.0)
                or float(np.sum(weights_)) <= limit
            ):
                raise ValueError(
                    "Mean-zero pressure weights must be one positive finite vector."
                )
            pin = None
        elif mode_ == "pinned":
            if weights is not None:
                raise ValueError("Pinned pressure gauges do not accept weights.")
            pin = 0 if pinned_dof is None else int(pinned_dof)
            if pin < 0:
                raise ValueError("pinned_dof must be nonnegative.")
            weights_ = None
        else:
            if weights is not None or pinned_dof is not None:
                raise ValueError("The no-gauge policy accepts no gauge data.")
            pin = None
            weights_ = None
        self.weights = None if weights_ is None else jnp.asarray(weights_)
        self.pinned_dof = pin
        self.mode = mode_  # type: ignore[assignment]
        self.tolerance = limit
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "mixed-pressure-gauge",
                "mode": mode_,
                "weights": None if weights_ is None else array_tree_fingerprint(weights_),
                "pinned_dof": pin,
                "tolerance": limit.hex(),
            }
        )

    def _weights(self, size: int, dtype, /) -> Array:
        if size <= 0:
            raise ValueError("Pressure gauge requires a nonempty pressure vector.")
        if self.mode == "mean-zero":
            weights = (
                jnp.ones((size,), dtype=dtype)
                if self.weights is None
                else jnp.asarray(self.weights, dtype=dtype)
            )
            if weights.shape != (size,):
                raise ValueError(
                    "Pressure gauge weights do not match the pressure space."
                )
            return weights
        if self.mode == "pinned":
            if self.pinned_dof is None or self.pinned_dof >= size:
                raise ValueError("pinned_dof lies outside the pressure space.")
            return jnp.zeros((size,), dtype=dtype).at[self.pinned_dof].set(1.0)
        return jnp.zeros((size,), dtype=dtype)

    def residual(self, pressure: ArrayLike, /) -> Array:
        value = jnp.asarray(pressure)
        if value.ndim != 1:
            raise ValueError("Pressure gauge requires one rank-1 pressure vector.")
        if self.mode == "none":
            return jnp.asarray(0.0, dtype=value.dtype)
        return jnp.sum(self._weights(value.size, value.dtype) * value)

    def project(self, pressure: ArrayLike, /) -> Array:
        value = jnp.asarray(pressure)
        if value.ndim != 1:
            raise ValueError("Pressure gauge requires one rank-1 pressure vector.")
        if self.mode == "none":
            return value
        weights = self._weights(value.size, value.dtype)
        return value - self.residual(value) / jnp.sum(weights)

    def evidence(self, pressure: ArrayLike, /) -> PressureGaugeEvidence:
        value = jnp.asarray(pressure)
        if value.ndim != 1:
            raise ValueError("Pressure gauge requires one rank-1 pressure vector.")
        residual = self.residual(value)
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=value.dtype),
            jnp.max(jnp.abs(value)),
        )
        finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual)
        valid = finite & (
            jnp.asarray(True)
            if self.mode == "none"
            else jnp.abs(residual) <= self.tolerance * scale
        )
        return PressureGaugeEvidence(residual, scale, finite, valid, self.mode)

    def diagnostic_vector(self, size: int, dtype=float, /) -> np.ndarray:
        return np.asarray(self._weights(size, dtype))


class MixedPressureStabilization(StrictModule, NonTrainableState):
    """Declared pressure stabilization; stable mixed plans accept only none."""

    kind: MixedPressureStabilizationKind = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    stabilization_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MixedPressureStabilizationKind = "none",
        /,
        *,
        coefficient: float = 0.0,
    ):
        kind_ = str(kind)
        if kind_ not in ("none", "pressure-laplacian"):
            raise ValueError("Unknown mixed-pressure stabilization.")
        coefficient_ = float(coefficient)
        if not isfinite(coefficient_) or coefficient_ < 0.0:
            raise ValueError("Stabilization coefficient must be finite and nonnegative.")
        if (kind_ == "none") != (coefficient_ == 0.0):
            raise ValueError(
                "No stabilization requires zero coefficient and pressure-laplacian requires a positive coefficient."
            )
        self.kind = kind_  # type: ignore[assignment]
        self.coefficient = coefficient_
        self.stabilization_id = canonical_fingerprint(
            {
                "kind": "mixed-pressure-stabilization",
                "stabilization": kind_,
                "coefficient": coefficient_.hex(),
            }
        )


class MixedInfSupEvidence(StrictModule, NonTrainableState):
    """Algebraic LBB, gauge, and off-diagonal adjoint evidence."""

    displacement_dimension: int = eqx.field(static=True)
    pressure_dimension: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    minimum_stable_rank: int = eqx.field(static=True)
    pressure_nullity: int = eqx.field(static=True)
    inf_sup_constant: float = eqx.field(static=True)
    largest_discarded_singular_value: float = eqx.field(static=True)
    gauge_nullspace_coupling: float = eqx.field(static=True)
    adjoint_sign: int = eqx.field(static=True)
    adjoint_defect: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    gauge_resolves_nullspace: bool = eqx.field(static=True)
    stabilization_absent: bool = eqx.field(static=True)
    stable: bool = eqx.field(static=True)
    locking_safe: bool = eqx.field(static=True)


class MixedFiniteElementSpaceEvidence(StrictModule, NonTrainableState):
    pair_names: tuple[str, ...] = eqx.field(static=True)
    displacement_degree: int = eqx.field(static=True)
    pressure_degree: int = eqx.field(static=True)
    displacement_components: int = eqx.field(static=True)
    lbb_conforming: bool = eqx.field(static=True)
    stabilization_absent: bool = eqx.field(static=True)
    stabilization_refused: bool = eqx.field(static=True)
    locking_safe: bool = eqx.field(static=True)


class MixedFiniteElementConstraintEvaluation(StrictModule):
    residual: tuple[Array, Array]
    displacement_residual: Array
    constraint_residual: Array
    gauged_pressure: Array
    gauge: PressureGaugeEvidence
    inf_sup: MixedInfSupEvidence
    finite: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)


class PreparedMixedFiniteElementConstraint(StrictModule, NonTrainableState):
    """Compiled mixed root with certified saddle blocks and explicit gauge."""

    problem: CompiledFiniteElementProblem
    discretization: FiniteElementDiscretization
    operator: BlockLinearOperator
    primal_operator: AbstractLinearOperator
    constraint_operator: AbstractLinearOperator
    pressure_operator: AbstractLinearOperator | None
    gauge: PressureGaugePolicy
    inf_sup: MixedInfSupEvidence
    spaces: MixedFiniteElementSpaceEvidence
    displacement_index: int = eqx.field(static=True)
    pressure_index: int = eqx.field(static=True)
    constraint_sign: int = eqx.field(static=True)
    formulation: MixedConstraintFormulation = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def evaluate(
        self,
        state: tuple[ArrayLike, ArrayLike],
        args: object = None,
        /,
    ) -> MixedFiniteElementConstraintEvaluation:
        values = self.problem.state_space.validate(state)
        gauged_pressure = self.gauge.project(values[self.pressure_index])
        evaluated = list(values)
        evaluated[self.pressure_index] = gauged_pressure
        raw = self.problem.residual(tuple(evaluated), args)
        displacement = self.primal_operator.target.inverse_riesz(
            raw[self.displacement_index]
        )
        constraint = self.constraint_sign * self.constraint_operator.target.inverse_riesz(
            raw[self.pressure_index]
        )
        gauge_evidence = self.gauge.evidence(gauged_pressure)
        finite = (
            jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(constraint))
            & jnp.all(jnp.isfinite(gauged_pressure))
        )
        valid = finite & gauge_evidence.valid & jnp.asarray(self.inf_sup.stable)
        return MixedFiniteElementConstraintEvaluation(
            (displacement, constraint),
            displacement,
            constraint,
            gauged_pressure,
            gauge_evidence,
            self.inf_sup,
            finite,
            valid,
            self.prepared_id,
        )


class MixedFiniteElementConstraintPlan(StrictModule, NonTrainableState):
    """Taylor-Hood/Q2-Q1 preparation with no unverified stabilization path."""

    mesh: CellMesh
    coordinate_spec: FiniteElementCoordinateSpec | None
    gauge: PressureGaugePolicy
    stabilization: MixedPressureStabilization
    displacement_field: str = eqx.field(static=True)
    pressure_field: str = eqx.field(static=True)
    bulk_modulus: float | None = eqx.field(static=True)
    formulation: MixedConstraintFormulation = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        gauge: PressureGaugePolicy,
        /,
        *,
        coordinate_spec: FiniteElementCoordinateSpec | None = None,
        displacement_field: str = "u",
        pressure_field: str = "p",
        bulk_modulus: float | None = None,
        stabilization: MixedPressureStabilization | None = None,
        rank_tolerance: float = 1.0e-10,
        plan_id: str | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if not isinstance(gauge, PressureGaugePolicy):
            raise TypeError("gauge must be PressureGaugePolicy.")
        stabilization_ = (
            MixedPressureStabilization() if stabilization is None else stabilization
        )
        if not isinstance(stabilization_, MixedPressureStabilization):
            raise TypeError("stabilization must be MixedPressureStabilization or None.")
        if coordinate_spec is not None:
            if not isinstance(coordinate_spec, FiniteElementCoordinateSpec):
                raise TypeError(
                    "coordinate_spec must be FiniteElementCoordinateSpec or None."
                )
            coordinate_spec.resolve(mesh)
        if stabilization_.kind != "none":
            raise ValueError(
                "Taylor-Hood/Q2-Q1 preparation refuses unverified pressure stabilization."
            )
        displacement = str(displacement_field)
        pressure = str(pressure_field)
        if not displacement or not pressure or displacement == pressure:
            raise ValueError("Mixed field names must be distinct and non-empty.")
        tolerance = float(rank_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("rank_tolerance must be positive and finite.")
        if bulk_modulus is None:
            bulk = None
            formulation: MixedConstraintFormulation = "exact"
            if gauge.mode == "none":
                raise ValueError("Exact mixed pressure requires an explicit gauge.")
        else:
            bulk = float(bulk_modulus)
            if not isfinite(bulk) or bulk <= 0.0:
                raise ValueError("bulk_modulus must be positive and finite.")
            formulation = "finite-bulk"
            if gauge.mode != "none":
                raise ValueError(
                    "Finite-bulk pressure must use the explicit no-gauge policy."
                )
        payload = {
            "kind": "mixed-finite-element-constraint-plan",
            "mesh": mesh.mesh_id,
            "displacement_field": displacement,
            "pressure_field": pressure,
            "formulation": formulation,
            "bulk_modulus": None if bulk is None else bulk.hex(),
            "gauge": gauge.gauge_id,
            "stabilization": stabilization_.stabilization_id,
            "rank_tolerance": tolerance.hex(),
        }
        if coordinate_spec is not None:
            payload["coordinate_spec"] = {
                "id": coordinate_spec.coordinate_spec_id,
                "coordinates": array_tree_fingerprint(coordinate_spec.coordinates),
            }
        generated = canonical_fingerprint(payload)
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty or None.")
        self.mesh = mesh
        self.coordinate_spec = coordinate_spec
        self.gauge = gauge
        self.stabilization = stabilization_
        self.displacement_field = displacement
        self.pressure_field = pressure
        self.bulk_modulus = bulk
        self.formulation = formulation
        self.rank_tolerance = tolerance
        self.plan_id = identifier

    def _space_plan(self, /) -> tuple[FiniteElementPlan, MixedFiniteElementSpaceEvidence]:
        ambient_dimension = self.mesh.ambient_dimension
        displacement_elements = {}
        pressure_elements = {}
        pair_names = []
        for block in self.mesh.blocks:
            if block.cell_kind in ("triangle", "tetrahedron"):
                pair_name = "taylor-hood"
            elif block.cell_kind in ("quadrilateral", "hexahedron"):
                pair_name = "q2-q1"
            else:
                raise ValueError("Mixed FE preparation requires simplex or tensor cells.")
            displacement_elements[block.name] = lagrange_element(block.cell_kind, 2)
            pressure_elements[block.name] = lagrange_element(block.cell_kind, 1)
            pair_names.append(pair_name)
        fields = (
            FiniteElementFieldSpec(
                self.displacement_field,
                displacement_elements,
                component_shape=(ambient_dimension,),
            ),
            FiniteElementFieldSpec(self.pressure_field, pressure_elements),
        )
        evidence = MixedFiniteElementSpaceEvidence(
            tuple(pair_names),
            2,
            1,
            ambient_dimension,
            True,
            True,
            True,
            True,
        )
        return FiniteElementPlan(
            self.mesh, fields, coordinate_spec=self.coordinate_spec
        ), evidence

    def prepare(
        self,
        form: FiniteElementForm,
        /,
        *,
        initial_state: tuple[ArrayLike, ArrayLike] | None = None,
        args: object = None,
    ) -> PreparedMixedFiniteElementConstraint:
        from ...equations._finite_element_variational import (
            compile_finite_element_problem,
            CompiledFiniteElementProblem,
            FiniteElementForm,
        )

        if not isinstance(form, FiniteElementForm):
            raise TypeError("form must be FiniteElementForm.")
        if form.field_names != (self.displacement_field, self.pressure_field):
            raise ValueError("Mixed form fields must match the plan field order exactly.")
        finite_element_plan, spaces = self._space_plan()
        discretization = finite_element_plan.prepare()
        problem = compile_finite_element_problem(form, discretization)
        if not isinstance(problem, CompiledFiniteElementProblem):
            raise TypeError("Mixed form compilation did not return an FE problem.")
        state = problem.state_space.zeros() if initial_state is None else initial_state
        return _prepare_compiled_mixed_constraint(
            problem,
            discretization,
            state,
            self,
            spaces,
            args=args,
        )


def mixed_inf_sup_diagnostic(
    constraint_matrix: ArrayLike,
    coupling_matrix: ArrayLike,
    gauge: PressureGaugePolicy,
    /,
    *,
    formulation: MixedConstraintFormulation,
    rank_tolerance: float = 1.0e-10,
) -> MixedInfSupEvidence:
    """Diagnose an assembled divergence block without concealing instability."""

    constraint = np.asarray(constraint_matrix)
    coupling = np.asarray(coupling_matrix)
    if constraint.ndim != 2 or coupling.shape != constraint.T.shape:
        raise ValueError(
            "Mixed constraint and coupling matrices must be transposes in shape."
        )
    if np.any(~np.isfinite(constraint)) or np.any(~np.isfinite(coupling)):
        raise ValueError("Mixed constraint and coupling matrices must be finite.")
    if not isinstance(gauge, PressureGaugePolicy):
        raise TypeError("gauge must be PressureGaugePolicy.")
    if formulation not in ("exact", "finite-bulk"):
        raise ValueError("Unknown mixed constraint formulation.")
    tolerance = float(rank_tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive and finite.")
    pressure_dimension, displacement_dimension = constraint.shape
    if pressure_dimension < 1 or displacement_dimension < 1:
        raise ValueError("Mixed constraint matrices must be nonempty.")
    left_vectors, singular_values, _ = np.linalg.svd(constraint, full_matrices=True)
    largest = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    threshold = tolerance * largest
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    minimum_rank = max(pressure_dimension - 1, 0)
    nullity = pressure_dimension - numerical_rank
    inf_sup = float(singular_values[numerical_rank - 1]) if numerical_rank else 0.0
    largest_discarded = (
        float(singular_values[numerical_rank])
        if numerical_rank < singular_values.size
        else 0.0
    )
    null_basis = left_vectors[:, numerical_rank:]
    if null_basis.shape[1] == 0:
        gauge_coupling = 1.0
        gauge_resolves = True
    elif gauge.mode == "none":
        gauge_coupling = 0.0
        gauge_resolves = formulation == "finite-bulk" and nullity <= 1
    else:
        gauge_vector = gauge.diagnostic_vector(
            pressure_dimension, constraint.dtype
        ).reshape((1, pressure_dimension))
        gauge_singular_values = np.linalg.svd(gauge_vector @ null_basis, compute_uv=False)
        gauge_coupling = (
            float(np.min(gauge_singular_values)) if gauge_singular_values.size else 0.0
        )
        gauge_resolves = nullity <= 1 and gauge_coupling > tolerance
    plus_defect = float(np.max(np.abs(coupling - constraint.T), initial=0.0))
    minus_defect = float(np.max(np.abs(coupling + constraint.T), initial=0.0))
    sign = 1 if plus_defect <= minus_defect else -1
    adjoint_defect = min(plus_defect, minus_defect)
    adjoint_scale = max(
        float(np.max(np.abs(coupling), initial=0.0)),
        float(np.max(np.abs(constraint), initial=0.0)),
        1.0,
    )
    stable = (
        numerical_rank >= minimum_rank
        and nullity <= 1
        and inf_sup > threshold
        and gauge_resolves
        and adjoint_defect <= tolerance * adjoint_scale
    )
    return MixedInfSupEvidence(
        displacement_dimension,
        pressure_dimension,
        numerical_rank,
        minimum_rank,
        nullity,
        inf_sup,
        largest_discarded,
        gauge_coupling,
        sign,
        adjoint_defect,
        tolerance,
        gauge_resolves,
        True,
        stable,
        stable,
    )


def _primalized_block(
    operator: AbstractLinearOperator,
    target,
    /,
) -> FunctionLinearOperator:
    return FunctionLinearOperator(
        lambda value: target.inverse_riesz(operator.mv(value)),
        source=operator.source,
        target=target,
        operator_id=canonical_fingerprint(
            {
                "kind": "primalized-mixed-finite-element-block",
                "operator": operator.operator_id,
                "target": target.space_id,
            }
        ),
    )


def _dense_operator(operator: AbstractLinearOperator, /) -> np.ndarray:
    columns = []
    for index in range(operator.source.size):
        coordinate = np.zeros(
            (operator.source.size,),
            dtype=operator.source.flatten(operator.source.zeros()).dtype,
        )
        coordinate[index] = 1.0
        vector = operator.source.unflatten(jnp.asarray(coordinate))
        columns.append(np.asarray(operator.target.flatten(operator.mv(vector))))
    if not columns:
        return np.empty((operator.target.size, 0))
    return np.stack(columns, axis=1)


def _prepare_compiled_mixed_constraint(
    problem: CompiledFiniteElementProblem,
    discretization: FiniteElementDiscretization,
    state: tuple[ArrayLike, ArrayLike],
    plan: MixedFiniteElementConstraintPlan,
    spaces: MixedFiniteElementSpaceEvidence,
    /,
    *,
    args: object,
) -> PreparedMixedFiniteElementConstraint:
    fields = problem.form.field_names
    if fields != (plan.displacement_field, plan.pressure_field):
        raise ValueError("Compiled mixed fields do not match the preparation plan.")
    if not isinstance(problem.state_space, BlockSpace):
        raise ValueError("Mixed constraints require a block state space.")
    values = problem.state_space.validate(state)
    displacement_index = 0
    pressure_index = 1
    block = problem.block_linearization_operator(values, args)
    primal_raw = block.blocks[displacement_index][displacement_index]
    constraint_raw = block.blocks[pressure_index][displacement_index]
    coupling_raw = block.blocks[displacement_index][pressure_index]
    pressure_raw = block.blocks[pressure_index][pressure_index]
    if primal_raw is None or constraint_raw is None or coupling_raw is None:
        raise ValueError("Mixed hyperelastic form is missing a required saddle block.")
    displacement_space = problem.state_space.spaces[displacement_index]
    pressure_space = problem.state_space.spaces[pressure_index]
    if not isinstance(pressure_space, ArraySpace) or len(pressure_space.shape) != 1:
        raise ValueError("Mixed pressure must use one rank-1 array field.")
    primal = _primalized_block(primal_raw, displacement_space)
    raw_constraint = _primalized_block(constraint_raw, pressure_space)
    raw_coupling = _primalized_block(coupling_raw, displacement_space)
    if plan.formulation == "exact" and pressure_raw is not None:
        raise ValueError(
            "Exact mixed pressure refuses a diagonal block that would hide stabilization or bulk compliance."
        )
    if plan.formulation == "finite-bulk" and pressure_raw is None:
        raise ValueError(
            "Finite-bulk mixed pressure requires its physical compliance block."
        )
    constraint_matrix = _dense_operator(raw_constraint)
    coupling_matrix = _dense_operator(raw_coupling)
    inf_sup = mixed_inf_sup_diagnostic(
        constraint_matrix,
        coupling_matrix,
        plan.gauge,
        formulation=plan.formulation,
        rank_tolerance=plan.rank_tolerance,
    )
    if not inf_sup.stable:
        raise ValueError(
            "Mixed displacement-pressure spaces fail the requested LBB/gauge/adjoint evidence."
        )
    constraint = (
        raw_constraint
        if inf_sup.adjoint_sign == 1
        else ScaledLinearOperator(raw_constraint, -1.0)
    )
    pressure_operator = (
        None
        if pressure_raw is None
        else ScaledLinearOperator(
            _primalized_block(pressure_raw, pressure_space),
            -float(inf_sup.adjoint_sign),
        )
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-mixed-finite-element-constraint",
            "plan": plan.plan_id,
            "compilation": problem.compilation_id,
            "linearization": block.operator_id,
            "constraint_sign": inf_sup.adjoint_sign,
        }
    )
    operator = saddle_point_operator(
        primal,
        constraint,
        pressure_operator,
        operator_id=f"{prepared_id}:saddle",
    )
    return PreparedMixedFiniteElementConstraint(
        problem,
        discretization,
        operator,
        primal,
        constraint,
        pressure_operator,
        plan.gauge,
        inf_sup,
        spaces,
        displacement_index,
        pressure_index,
        inf_sup.adjoint_sign,
        plan.formulation,
        prepared_id,
    )


__all__ = [
    "MixedConstraintFormulation",
    "MixedFiniteElementConstraintEvaluation",
    "MixedFiniteElementConstraintPlan",
    "MixedFiniteElementSpaceEvidence",
    "MixedInfSupEvidence",
    "MixedPressureStabilization",
    "MixedPressureStabilizationKind",
    "PreparedMixedFiniteElementConstraint",
    "PressureGaugeEvidence",
    "PressureGaugeMode",
    "PressureGaugePolicy",
    "mixed_inf_sup_diagnostic",
]

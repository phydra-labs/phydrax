#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import sqrt
from operator import index
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._programming._cones import ProductCone, ZeroCone
from .._programming._problem import ConicProgram
from .._programming._psd_cone import PositiveSemidefiniteCone
from ._basis import (
    dense_monomial_count,
    DenseLocalizingBasis,
    DenseMomentBasis,
    DenseMonomialBasis,
)
from ._problem import (
    _equation_degrees,
    _equation_terms,
    PolynomialOptimizationProblem,
)


class PolynomialRelaxationStatus(IntEnum):
    """Host planning status, separate from any downstream solver status."""

    READY = 0
    INSUFFICIENT_ORDER = 1
    RESOURCE_REJECTED = 2
    STRUCTURALLY_INFEASIBLE = 3
    INVALID_DATA = 4


def _positive_limit(value: int, name: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = index(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


class PolynomialRelaxationResources(StrictModule):
    """Hard host-materialization limits for one dense moment relaxation."""

    max_moments: int = eqx.field(static=True)
    max_psd_blocks: int = eqx.field(static=True)
    max_psd_matrix_size: int = eqx.field(static=True)
    max_conic_rows: int = eqx.field(static=True)
    max_dense_entries: int = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_moments: int = 10_000,
        max_psd_blocks: int = 256,
        max_psd_matrix_size: int = 512,
        max_conic_rows: int = 1_000_000,
        max_dense_entries: int = 50_000_000,
    ):
        limits = {
            "max_moments": _positive_limit(max_moments, "max_moments"),
            "max_psd_blocks": _positive_limit(max_psd_blocks, "max_psd_blocks"),
            "max_psd_matrix_size": _positive_limit(
                max_psd_matrix_size, "max_psd_matrix_size"
            ),
            "max_conic_rows": _positive_limit(max_conic_rows, "max_conic_rows"),
            "max_dense_entries": _positive_limit(max_dense_entries, "max_dense_entries"),
        }
        self.max_moments = limits["max_moments"]
        self.max_psd_blocks = limits["max_psd_blocks"]
        self.max_psd_matrix_size = limits["max_psd_matrix_size"]
        self.max_conic_rows = limits["max_conic_rows"]
        self.max_dense_entries = limits["max_dense_entries"]
        self.resource_id = canonical_fingerprint(
            {"kind": "polynomial-relaxation-resources", **limits}
        )


class PolynomialRelaxationEstimate(StrictModule):
    """Exact dense topology counts computed without allocating relaxation arrays."""

    moment_count: int = eqx.field(static=True)
    zero_rows: int = eqx.field(static=True)
    conic_rows: int = eqx.field(static=True)
    psd_blocks: int = eqx.field(static=True)
    moment_matrix_size: int = eqx.field(static=True)
    localizing_matrix_sizes: tuple[int, ...] = eqx.field(static=True)
    largest_psd_matrix: int = eqx.field(static=True)
    dense_constraint_entries: int = eqx.field(static=True)


class PolynomialRelaxationPlan(StrictModule):
    """Validated relaxation order, resource decision, and explicit planning evidence."""

    problem: PolynomialOptimizationProblem
    resources: PolynomialRelaxationResources
    estimate: PolynomialRelaxationEstimate
    order: int = eqx.field(static=True)
    required_order: int = eqx.field(static=True)
    status: PolynomialRelaxationStatus = eqx.field(static=True)
    message: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def ready(self) -> bool:
        return self.status == PolynomialRelaxationStatus.READY


class PolynomialRelaxationTemplate(StrictModule):
    """Reusable dense basis and cone topology for numeric coefficient bindings."""

    plan: PolynomialRelaxationPlan
    moment_basis: DenseMomentBasis
    equality_multiplier_bases: tuple[DenseMonomialBasis | None, ...]
    localizing_bases: tuple[DenseLocalizingBasis, ...]
    equality_row_slices: tuple[slice, ...] = eqx.field(static=True)
    zero_slice: slice = eqx.field(static=True)
    moment_psd_slice: slice = eqx.field(static=True)
    localizing_psd_slices: tuple[slice, ...] = eqx.field(static=True)
    template_id: str = eqx.field(static=True)


class PreparedPolynomialRelaxation(StrictModule):
    """One numeric polynomial binding compiled to the canonical conic contract."""

    template: PolynomialRelaxationTemplate
    problem: PolynomialOptimizationProblem
    program: ConicProgram
    numeric_version: int = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @property
    def plan(self) -> PolynomialRelaxationPlan:
        return self.template.plan

    @property
    def conic_program(self) -> ConicProgram:
        return self.program


def _constant_infeasibility(problem: PolynomialOptimizationProblem, /) -> str | None:
    systems = (
        ("equality", problem.equalities, True),
        ("inequality", problem.inequalities, False),
    )
    for kind, system, equality in systems:
        if system is None:
            continue
        coefficients = np.asarray(system.coefficients)
        if not np.all(np.isfinite(coefficients)):
            return f"{kind} system has non-finite coefficients"
        for equation in range(system.support.equation_count):
            exponents, selected = _equation_terms(system, equation)
            if exponents.shape[0] and np.all(exponents == 0):
                value = np.sum(np.asarray(selected))
                if (equality and value != 0.0) or (not equality and value < 0.0):
                    label = system.support.equation_labels[equation]
                    return f"{kind} {label!r} is an infeasible constant polynomial"
    for name, system in (
        ("objective", problem.objective),
        ("equalities", problem.equalities),
        ("inequalities", problem.inequalities),
    ):
        if system is not None and not np.all(
            np.isfinite(np.asarray(system.coefficients))
        ):
            return f"{name} has non-finite coefficients"
    return None


def _estimate(
    problem: PolynomialOptimizationProblem,
    order: int,
    equality_degrees: tuple[int, ...],
    inequality_degrees: tuple[int, ...],
    /,
) -> PolynomialRelaxationEstimate:
    variables = problem.variable_count
    moment_count = dense_monomial_count(variables, 2 * order)
    moment_size = dense_monomial_count(variables, order)
    equality_rows = 0
    if problem.equalities is not None:
        for equation, degree in enumerate(equality_degrees):
            terms, _ = _equation_terms(problem.equalities, equation)
            if terms.shape[0]:
                multiplier_degree = max(2 * order - degree, 0)
                equality_rows += dense_monomial_count(variables, multiplier_degree)
    localizing_sizes = tuple(
        dense_monomial_count(variables, max(order - (degree + 1) // 2, 0))
        for degree in inequality_degrees
    )
    zero_rows = 1 + equality_rows
    psd_dimensions = (moment_size, *localizing_sizes)
    conic_rows = zero_rows + sum(size * (size + 1) // 2 for size in psd_dimensions)
    return PolynomialRelaxationEstimate(
        moment_count,
        zero_rows,
        conic_rows,
        len(psd_dimensions),
        moment_size,
        localizing_sizes,
        max(psd_dimensions),
        conic_rows * moment_count,
    )


def plan_polynomial_relaxation(
    problem: PolynomialOptimizationProblem,
    order: int,
    /,
    *,
    resources: PolynomialRelaxationResources | None = None,
) -> PolynomialRelaxationPlan:
    """Plan a dense moment relaxation without materializing its conic matrix."""

    if not isinstance(problem, PolynomialOptimizationProblem):
        raise TypeError("problem must be a PolynomialOptimizationProblem.")
    if isinstance(order, bool):
        raise TypeError("order must be an integer.")
    relaxation_order = index(order)
    if relaxation_order < 0:
        raise ValueError("order must be nonnegative.")
    limits = PolynomialRelaxationResources() if resources is None else resources
    if not isinstance(limits, PolynomialRelaxationResources):
        raise TypeError("resources must be PolynomialRelaxationResources or None.")

    objective_degree = _equation_degrees(problem.objective)[0]
    equality_degrees = (
        () if problem.equalities is None else _equation_degrees(problem.equalities)
    )
    inequality_degrees = (
        () if problem.inequalities is None else _equation_degrees(problem.inequalities)
    )
    required_order = max(
        (degree + 1) // 2
        for degree in (objective_degree, *equality_degrees, *inequality_degrees)
    )
    estimate = _estimate(
        problem,
        relaxation_order,
        equality_degrees,
        inequality_degrees,
    )
    infeasibility = _constant_infeasibility(problem)
    if infeasibility is not None and "non-finite" in infeasibility:
        status = PolynomialRelaxationStatus.INVALID_DATA
        message = str(infeasibility)
    elif infeasibility is not None:
        status = PolynomialRelaxationStatus.STRUCTURALLY_INFEASIBLE
        message = infeasibility
    elif relaxation_order < required_order:
        status = PolynomialRelaxationStatus.INSUFFICIENT_ORDER
        message = (
            f"relaxation order {relaxation_order} is below the support-required "
            f"order {required_order}"
        )
    else:
        violations = (
            (
                estimate.moment_count > limits.max_moments,
                f"moment count {estimate.moment_count} exceeds {limits.max_moments}",
            ),
            (
                estimate.psd_blocks > limits.max_psd_blocks,
                f"PSD block count {estimate.psd_blocks} exceeds {limits.max_psd_blocks}",
            ),
            (
                estimate.largest_psd_matrix > limits.max_psd_matrix_size,
                "largest PSD matrix size "
                f"{estimate.largest_psd_matrix} exceeds {limits.max_psd_matrix_size}",
            ),
            (
                estimate.conic_rows > limits.max_conic_rows,
                f"conic row count {estimate.conic_rows} exceeds {limits.max_conic_rows}",
            ),
            (
                estimate.dense_constraint_entries > limits.max_dense_entries,
                "dense constraint entries "
                f"{estimate.dense_constraint_entries} exceed {limits.max_dense_entries}",
            ),
        )
        rejected = next((text for condition, text in violations if condition), None)
        if rejected is None:
            status = PolynomialRelaxationStatus.READY
            message = "dense moment relaxation topology accepted"
        else:
            status = PolynomialRelaxationStatus.RESOURCE_REJECTED
            message = rejected
    plan_id = canonical_fingerprint(
        {
            "kind": "polynomial-relaxation-plan",
            "structure": problem.structure_id,
            "order": relaxation_order,
            "resources": limits.resource_id,
            "required_order": required_order,
            "status": int(status),
        }
    )
    return PolynomialRelaxationPlan(
        problem,
        limits,
        estimate,
        relaxation_order,
        required_order,
        status,
        message,
        plan_id,
    )


def _pack_slice_size(matrix_size: int, /) -> int:
    return matrix_size * (matrix_size + 1) // 2


def prepare_polynomial_relaxation_template(
    plan: PolynomialRelaxationPlan,
    /,
) -> PolynomialRelaxationTemplate:
    if not isinstance(plan, PolynomialRelaxationPlan):
        raise TypeError("plan must be a PolynomialRelaxationPlan.")
    if not plan.ready:
        raise ValueError(
            f"Cannot prepare polynomial relaxation with status {plan.status.name}: "
            f"{plan.message}."
        )
    problem = plan.problem
    moments = DenseMomentBasis(problem.variable_count, plan.order)

    equality_bases: list[DenseMonomialBasis | None] = []
    equality_slices: list[slice] = []
    cursor = 1
    if problem.equalities is not None:
        degrees = _equation_degrees(problem.equalities)
        for equation, degree in enumerate(degrees):
            terms, _ = _equation_terms(problem.equalities, equation)
            basis = (
                None
                if terms.shape[0] == 0
                else DenseMonomialBasis(problem.variable_count, 2 * plan.order - degree)
            )
            equality_bases.append(basis)
            rows = 0 if basis is None else basis.size
            equality_slices.append(slice(cursor, cursor + rows))
            cursor += rows
    zero_slice = slice(0, cursor)

    localizing: list[DenseLocalizingBasis] = []
    if problem.inequalities is not None:
        for equation in range(problem.inequalities.support.equation_count):
            exponents, _ = _equation_terms(problem.inequalities, equation)
            localizing.append(DenseLocalizingBasis(moments, exponents))

    moment_start = cursor
    cursor += _pack_slice_size(moments.matrix_size)
    moment_slice = slice(moment_start, cursor)
    localizing_slices: list[slice] = []
    for basis in localizing:
        start = cursor
        cursor += _pack_slice_size(basis.matrix_size)
        localizing_slices.append(slice(start, cursor))
    if cursor != plan.estimate.conic_rows:
        raise RuntimeError(
            "Prepared polynomial topology disagrees with its resource estimate."
        )
    template_id = canonical_fingerprint(
        {
            "kind": "polynomial-relaxation-template",
            "structure": problem.structure_id,
            "order": plan.order,
            "moment_basis": moments.basis_id,
            "equality_bases": [
                None if basis is None else basis.basis_id for basis in equality_bases
            ],
            "localizing_bases": [basis.basis_id for basis in localizing],
        }
    )
    return PolynomialRelaxationTemplate(
        plan,
        moments,
        tuple(equality_bases),
        tuple(localizing),
        tuple(equality_slices),
        zero_slice,
        moment_slice,
        tuple(localizing_slices),
        template_id,
    )


def _moment_indices(basis: DenseMomentBasis, exponents: np.ndarray, /) -> tuple[int, ...]:
    lookup = {
        exponent: position
        for position, exponent in enumerate(basis.moments.exponent_tuples)
    }
    return tuple(lookup[tuple(int(value) for value in row)] for row in exponents)


def _polynomial_vector(
    basis: DenseMomentBasis,
    exponents: np.ndarray,
    coefficients: Array,
    dtype: Any,
    /,
) -> Array:
    vector = jnp.zeros((basis.moment_count,), dtype=dtype)
    if exponents.shape[0] == 0:
        return vector
    positions = jnp.asarray(_moment_indices(basis, exponents), dtype=jnp.int32)
    return vector.at[positions].add(coefficients.astype(dtype))


def _symmetric_pack_map(
    entry_indices: np.ndarray,
    moment_count: int,
    dtype: Any,
    /,
) -> Array:
    matrix_size = int(entry_indices.shape[0])
    packed = jnp.zeros((_pack_slice_size(matrix_size), moment_count), dtype=dtype)
    cursor = 0
    for column in range(matrix_size):
        for row in range(column + 1):
            scale = 1.0 if row == column else sqrt(2.0)
            packed = packed.at[cursor, int(entry_indices[row, column])].set(scale)
            cursor += 1
    return packed


def _compile_numeric(
    template: PolynomialRelaxationTemplate,
    problem: PolynomialOptimizationProblem,
    /,
) -> ConicProgram:
    coefficient_dtypes = [problem.objective.coefficients.dtype]
    if problem.equalities is not None:
        coefficient_dtypes.append(problem.equalities.coefficients.dtype)
    if problem.inequalities is not None:
        coefficient_dtypes.append(problem.inequalities.coefficients.dtype)
    dtype = jnp.result_type(*coefficient_dtypes, jnp.float32)
    basis = template.moment_basis

    objective_exponents, objective_coefficients = _equation_terms(problem.objective, 0)
    objective = _polynomial_vector(
        basis,
        objective_exponents,
        objective_coefficients,
        dtype,
    )
    matrix = jnp.zeros(
        (template.plan.estimate.conic_rows, basis.moment_count), dtype=dtype
    )
    rhs = jnp.zeros((template.plan.estimate.conic_rows,), dtype=dtype)
    zero_index = basis.moments.index((0,) * problem.variable_count)
    matrix = matrix.at[0, zero_index].set(1.0)
    rhs = rhs.at[0].set(1.0)

    if problem.equalities is not None:
        moment_lookup = {
            exponent: position
            for position, exponent in enumerate(basis.moments.exponent_tuples)
        }
        for equation, (multiplier_basis, row_slice) in enumerate(
            zip(
                template.equality_multiplier_bases,
                template.equality_row_slices,
                strict=True,
            )
        ):
            if multiplier_basis is None:
                continue
            exponents, coefficients = _equation_terms(problem.equalities, equation)
            rows = jnp.zeros((multiplier_basis.size, basis.moment_count), dtype=dtype)
            for row, multiplier in enumerate(multiplier_basis.exponent_tuples):
                positions = tuple(
                    moment_lookup[
                        tuple(
                            left + right
                            for left, right in zip(multiplier, term, strict=True)
                        )
                    ]
                    for term in exponents
                )
                rows = rows.at[row, jnp.asarray(positions, dtype=jnp.int32)].add(
                    coefficients.astype(dtype)
                )
            matrix = matrix.at[row_slice].set(rows)

    moment_map = _symmetric_pack_map(
        np.asarray(basis.entry_indices),
        basis.moment_count,
        dtype,
    )
    matrix = matrix.at[template.moment_psd_slice].set(-moment_map)

    if problem.inequalities is not None:
        for equation, (localizing_basis, row_slice) in enumerate(
            zip(
                template.localizing_bases,
                template.localizing_psd_slices,
                strict=True,
            )
        ):
            _, coefficients = _equation_terms(problem.inequalities, equation)
            entry_indices = np.asarray(localizing_basis.entry_term_indices)
            localizing_map = jnp.zeros(
                (
                    _pack_slice_size(localizing_basis.matrix_size),
                    basis.moment_count,
                ),
                dtype=dtype,
            )
            cursor = 0
            for column in range(localizing_basis.matrix_size):
                for row in range(column + 1):
                    scale = 1.0 if row == column else sqrt(2.0)
                    for term in range(localizing_basis.term_count):
                        localizing_map = localizing_map.at[
                            cursor, int(entry_indices[row, column, term])
                        ].add(scale * coefficients[term].astype(dtype))
                    cursor += 1
            matrix = matrix.at[row_slice].set(-localizing_map)

    cones = [
        ZeroCone(template.plan.estimate.zero_rows),
        PositiveSemidefiniteCone(basis.matrix_size),
    ]
    cones.extend(
        PositiveSemidefiniteCone(localizing.matrix_size)
        for localizing in template.localizing_bases
    )
    return ConicProgram(
        None,
        objective,
        matrix,
        rhs,
        ProductCone(tuple(cones)),
        problem_id=f"polynomial-moment-relaxation:{template.template_id}",
        convexity_evidence="linear image into explicit PSD and zero cones",
    )


def bind_polynomial_relaxation_numeric(
    template: PolynomialRelaxationTemplate,
    problem: PolynomialOptimizationProblem,
    /,
    *,
    numeric_version: int = 0,
) -> PreparedPolynomialRelaxation:
    if not isinstance(template, PolynomialRelaxationTemplate):
        raise TypeError("template must be a PolynomialRelaxationTemplate.")
    if not isinstance(problem, PolynomialOptimizationProblem):
        raise TypeError("problem must be a PolynomialOptimizationProblem.")
    if problem.structure_id != template.plan.problem.structure_id:
        raise ValueError("Numeric binding must preserve polynomial support topology.")
    if isinstance(numeric_version, bool):
        raise TypeError("numeric_version must be an integer.")
    version = index(numeric_version)
    if version < 0:
        raise ValueError("numeric_version must be nonnegative.")
    rebound_plan = plan_polynomial_relaxation(
        problem,
        template.plan.order,
        resources=template.plan.resources,
    )
    if not rebound_plan.ready:
        raise ValueError(
            f"Numeric binding is not admissible ({rebound_plan.status.name}): "
            f"{rebound_plan.message}."
        )
    program = _compile_numeric(template, problem)
    binding_id = canonical_fingerprint(
        {
            "kind": "prepared-polynomial-relaxation",
            "template": template.template_id,
            "problem": problem.problem_id,
            "numeric_version": version,
        }
    )
    return PreparedPolynomialRelaxation(
        template,
        problem,
        program,
        version,
        binding_id,
    )


def prepare_polynomial_relaxation(
    problem_or_plan: PolynomialOptimizationProblem | PolynomialRelaxationPlan,
    order: int | None = None,
    /,
    *,
    resources: PolynomialRelaxationResources | None = None,
) -> PreparedPolynomialRelaxation:
    """Plan, materialize, and numerically bind one dense moment relaxation."""

    if isinstance(problem_or_plan, PolynomialRelaxationPlan):
        if order is not None or resources is not None:
            raise ValueError("order and resources must be omitted with an existing plan.")
        plan = problem_or_plan
    else:
        if not isinstance(problem_or_plan, PolynomialOptimizationProblem):
            raise TypeError(
                "problem_or_plan must be PolynomialOptimizationProblem or "
                "PolynomialRelaxationPlan."
            )
        if order is None:
            raise ValueError("order is required when preparing from a problem.")
        plan = plan_polynomial_relaxation(
            problem_or_plan,
            order,
            resources=resources,
        )
    template = prepare_polynomial_relaxation_template(plan)
    return bind_polynomial_relaxation_numeric(template, plan.problem)


def compile_polynomial_relaxation(
    plan: PolynomialRelaxationPlan,
    /,
) -> ConicProgram:
    """Compile a ready plan directly to an existing canonical ``ConicProgram``."""

    return prepare_polynomial_relaxation(plan).program


def refresh_polynomial_relaxation(
    prepared: PreparedPolynomialRelaxation,
    problem: PolynomialOptimizationProblem,
    /,
) -> PreparedPolynomialRelaxation:
    """Refresh numeric coefficients while preserving every basis and cone block."""

    if not isinstance(prepared, PreparedPolynomialRelaxation):
        raise TypeError("prepared must be a PreparedPolynomialRelaxation.")
    refreshed = bind_polynomial_relaxation_numeric(
        prepared.template,
        problem,
        numeric_version=prepared.numeric_version + 1,
    )
    if refreshed.program.structure_id != prepared.program.structure_id:
        raise ValueError(
            "Numeric refresh must preserve the compiled conic dtype and topology."
        )
    return refreshed


__all__ = [
    "PolynomialRelaxationEstimate",
    "PolynomialRelaxationPlan",
    "PolynomialRelaxationResources",
    "PolynomialRelaxationStatus",
    "PolynomialRelaxationTemplate",
    "PreparedPolynomialRelaxation",
    "bind_polynomial_relaxation_numeric",
    "compile_polynomial_relaxation",
    "plan_polynomial_relaxation",
    "prepare_polynomial_relaxation",
    "prepare_polynomial_relaxation_template",
    "refresh_polynomial_relaxation",
]

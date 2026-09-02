#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...backends._availability import import_backend_module
from ...backends.cvxpy import cvxpy_availability
from ...linalg import AbstractSparseLinearOperator, OperatorProperties
from ...sparse import EdgeRelation, SparseLinearMap
from ._cones import (
    NonnegativeCone,
    ProductCone,
    RotatedSecondOrderCone,
    SecondOrderCone,
    ZeroCone,
)
from ._exponential_cone import ExponentialCone
from ._power_cone import PowerCone
from ._problem import ConicProgram
from ._psd_cone import PositiveSemidefiniteCone
from ._quadratic import ConvexProgramResult


@dataclass(frozen=True, slots=True)
class CVXPYVariableSlice:
    variable_id: int
    start: int
    stop: int
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CVXPYConstraintSlice:
    constraint_id: int
    start: int
    stop: int
    shape: tuple[int, ...]


class CVXPYProgramBinding(StrictModule):
    """Host canonical program and exact CVXPY coordinate recovery metadata."""

    program: ConicProgram
    problem: Any = eqx.field(static=True)
    solving_chain: Any = eqx.field(static=True)
    inverse_data: Any = eqx.field(static=True)
    variable_slices: tuple[CVXPYVariableSlice, ...] = eqx.field(static=True)
    parameter_topology_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    numeric_fingerprint: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)
    numeric_binding_id: str = eqx.field(static=True)
    canonical_variable_id: int = eqx.field(static=True, default=-1)
    constraint_slices: tuple[CVXPYConstraintSlice, ...] = eqx.field(
        static=True, default=()
    )

    def prepare(self, policy: Any = None, /):
        """Prepare this exact numeric import for provenance-bound execution."""
        fingerprint = _program_numeric_fingerprint(self.program)
        expected_binding = _numeric_binding_id(
            self.binding_id, fingerprint, self.numeric_version
        )
        if (
            fingerprint != self.numeric_fingerprint
            or expected_binding != self.numeric_binding_id
        ):
            raise ValueError("CVXPY numeric binding is stale.")
        from ._lifecycle import prepare_convex_template, PreparedConvexProgram

        template = prepare_convex_template(self.program, policy)
        return PreparedConvexProgram(
            self.program,
            template,
            numeric_version=self.numeric_version,
            numeric_binding_id=self.numeric_binding_id,
        )


def _module():
    return import_backend_module(
        cvxpy_availability(), "optimization.canonical-convex-model", "cvxpy"
    )


def _sparse_map(matrix, /, *, properties=None):
    coo = matrix.tocoo()
    relation = EdgeRelation(
        jnp.asarray(coo.col, dtype=jnp.int32),
        jnp.asarray(coo.row, dtype=jnp.int32),
        source_size=int(matrix.shape[1]),
        target_size=int(matrix.shape[0]),
    )
    return SparseLinearMap(
        relation,
        jnp.asarray(coo.data, dtype=jnp.float64),
        properties=properties,
    )


def _cones(dims):
    cones = []
    if int(dims.zero):
        cones.append(ZeroCone(int(dims.zero)))
    if int(dims.nonneg):
        cones.append(NonnegativeCone(int(dims.nonneg)))
    cones.extend(SecondOrderCone(int(size)) for size in dims.soc)
    cones.extend(PositiveSemidefiniteCone(int(size)) for size in dims.psd)
    cones.extend(ExponentialCone() for _ in range(int(dims.exp)))
    cones.extend(PowerCone(float(exponent)) for exponent in dims.p3d)
    if not cones:
        raise ValueError("CVXPY canonicalization produced no supported cone rows.")
    return ProductCone(tuple(cones))


def _parameter_topology(problem) -> str:
    return canonical_fingerprint(
        {
            "kind": "cvxpy-parameter-topology",
            "parameters": [
                (int(parameter.id), tuple(int(size) for size in parameter.shape))
                for parameter in problem.parameters()
            ],
        }
    )


def _program_numeric_fingerprint(program: ConicProgram, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "cvxpy-canonical-numeric-program",
            "structure": program.structure_id,
            "arrays": array_tree_fingerprint(program),
        }
    )


def _numeric_binding_id(
    binding_id: str,
    numeric_fingerprint: str,
    numeric_version: int,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "cvxpy-numeric-binding",
            "binding": binding_id,
            "numeric_fingerprint": numeric_fingerprint,
            "numeric_version": numeric_version,
        }
    )


def import_cvxpy_problem(problem: Any, /) -> CVXPYProgramBinding:
    """Import the supported real continuous LP/QP/product-cone canonical subset."""
    cp = _module()
    if not isinstance(problem, cp.Problem):
        raise TypeError("problem must be a cvxpy.Problem.")
    for variable in problem.variables():
        attributes = variable.attributes
        if attributes.get("boolean") or attributes.get("integer"):
            raise TypeError("Mixed-integer CVXPY models require MixedIntegerProgram.")
        if (
            attributes.get("complex")
            or attributes.get("imag")
            or attributes.get("hermitian")
        ):
            raise TypeError("Complex CVXPY canonicalization is unsupported.")
    data, chain, inverse = problem.get_problem_data(cp.CLARABEL)
    if "A" not in data or "b" not in data or "c" not in data or "dims" not in data:
        raise TypeError("CVXPY problem did not canonicalize to assembled conic data.")
    matrix = data["A"].tocsr()
    linear = np.asarray(data["c"], dtype=np.float64)
    rhs = np.asarray(data["b"], dtype=np.float64)
    quadratic = data.get("P")
    quadratic_operator = None
    if quadratic is not None and int(quadratic.nnz):
        quadratic_operator = _sparse_map(
            quadratic.tocsr(),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "transformed",
                    "positive_semidefinite": "transformed",
                },
            ),
        )
    canonical = ConicProgram(
        quadratic_operator,
        jnp.asarray(linear),
        _sparse_map(matrix),
        jnp.asarray(rhs),
        _cones(data["dims"]),
        problem_id="cvxpy-canonical-program",
        convexity_evidence="cvxpy-dcp-canonicalization",
    )
    parameter_problem = data.get("param_prob")
    slices = []
    if parameter_problem is not None:
        columns = parameter_problem.var_id_to_col
        for variable in problem.variables():
            start = int(columns[variable.id])
            size = int(variable.size)
            slices.append(
                CVXPYVariableSlice(
                    int(variable.id), start, start + size, tuple(variable.shape)
                )
            )
    constraint_slices = []
    canonical_variable_id = -1
    if parameter_problem is not None:
        canonical_variable_id = int(parameter_problem.x.id)
        offset = 0
        for constraint in parameter_problem.constraints:
            size = int(constraint.size)
            constraint_slices.append(
                CVXPYConstraintSlice(
                    int(constraint.id),
                    offset,
                    offset + size,
                    tuple(constraint.shape),
                )
            )
            offset += size
        if offset != canonical.num_constraints:
            raise ValueError("CVXPY canonical dual map does not cover every cone row.")
    topology = _parameter_topology(problem)
    binding_id = canonical_fingerprint(
        {
            "kind": "cvxpy-program-binding",
            "program": canonical.structure_id,
            "topology": topology,
            "canonical_variable": canonical_variable_id,
            "variables": [
                (item.variable_id, item.start, item.stop, item.shape) for item in slices
            ],
            "constraints": [
                (item.constraint_id, item.start, item.stop, item.shape)
                for item in constraint_slices
            ],
        }
    )
    numeric_fingerprint = _program_numeric_fingerprint(canonical)
    numeric_version = 0
    return CVXPYProgramBinding(
        canonical,
        problem,
        chain,
        inverse,
        tuple(slices),
        topology,
        binding_id,
        numeric_fingerprint,
        numeric_version,
        _numeric_binding_id(binding_id, numeric_fingerprint, numeric_version),
        canonical_variable_id,
        tuple(constraint_slices),
    )


def _dense_host(operator):
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        import scipy.sparse as sp

        if storage.batch_shape:
            raise ValueError("CVXPY export requires unbatched sparse storage.")
        return sp.csr_matrix(
            (
                np.asarray(storage.values),
                np.asarray(storage.indices),
                np.asarray(storage.indptr),
            ),
            shape=storage.shape,
        )
    return np.asarray(operator)


def export_cvxpy_program(program: ConicProgram, /):
    """Export supported native cone blocks to an explicit CVXPY Problem."""
    cp = _module()
    if not isinstance(program, ConicProgram) or program.batch_shape:
        raise ValueError("CVXPY export requires one unbatched ConicProgram.")
    x = cp.Variable(program.num_variables)
    objective = np.asarray(program.linear) @ x
    if program.quadratic is not None:
        objective = objective + 0.5 * cp.quad_form(x, _dense_host(program.quadratic))
    matrix = _dense_host(program.constraint_matrix)
    slack = np.asarray(program.constraint_rhs) - matrix @ x
    blocks = (
        program.cone.cones if isinstance(program.cone, ProductCone) else (program.cone,)
    )
    slices = (
        program.cone.slices
        if isinstance(program.cone, ProductCone)
        else (slice(0, program.num_constraints),)
    )
    constraints = []
    for cone, block_slice in zip(blocks, slices, strict=True):
        value = slack[block_slice]
        if isinstance(cone, ZeroCone):
            constraints.append(value == 0)
        elif isinstance(cone, NonnegativeCone):
            constraints.append(value >= 0)
        elif isinstance(cone, SecondOrderCone):
            constraints.append(cp.SOC(value[0], value[1:]))
        elif isinstance(cone, RotatedSecondOrderCone):
            transformed = jnp.asarray(cone._to_soc(jnp.eye(cone.dimension)))
            transformed_value = np.asarray(transformed) @ value
            constraints.append(cp.SOC(transformed_value[0], transformed_value[1:]))
        elif isinstance(cone, ExponentialCone):
            constraints.append(cp.ExpCone(value[0], value[1], value[2]))
        elif isinstance(cone, PowerCone):
            constraints.append(cp.PowCone3D(value[0], value[1], value[2], cone.exponent))
        else:
            raise TypeError(f"CVXPY export does not support {type(cone).__name__}.")
    lower, upper = np.asarray(program.lower_bounds), np.asarray(program.upper_bounds)
    if np.any(np.isfinite(lower)):
        constraints.append(x[np.isfinite(lower)] >= lower[np.isfinite(lower)])
    if np.any(np.isfinite(upper)):
        constraints.append(x[np.isfinite(upper)] <= upper[np.isfinite(upper)])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    imported = import_cvxpy_problem(problem)
    binding_id = canonical_fingerprint(
        {
            "kind": "cvxpy-export",
            "program": program.structure_id,
            "cvxpy_binding": imported.binding_id,
        }
    )
    numeric_fingerprint = _program_numeric_fingerprint(program)
    binding = CVXPYProgramBinding(
        program,
        imported.problem,
        imported.solving_chain,
        imported.inverse_data,
        imported.variable_slices,
        imported.parameter_topology_id,
        binding_id,
        numeric_fingerprint,
        0,
        _numeric_binding_id(binding_id, numeric_fingerprint, 0),
        imported.canonical_variable_id,
        imported.constraint_slices,
    )
    return problem, binding


def refresh_cvxpy_program(
    binding: CVXPYProgramBinding,
    /,
) -> CVXPYProgramBinding:
    """Recanonicalize changed parameters only when topology/maps remain exact."""
    if not isinstance(binding, CVXPYProgramBinding):
        raise TypeError("binding must be a CVXPYProgramBinding.")
    refreshed = import_cvxpy_problem(binding.problem)
    if refreshed.parameter_topology_id != binding.parameter_topology_id:
        raise ValueError("CVXPY parameter topology changed during numeric refresh.")
    if refreshed.program.structure_id != binding.program.structure_id:
        raise ValueError("CVXPY recanonicalization changed sparse/cone topology.")
    if refreshed.binding_id != binding.binding_id:
        raise ValueError("CVXPY recovery binding changed during numeric refresh.")
    if refreshed.variable_slices != binding.variable_slices:
        raise ValueError("CVXPY variable recovery maps changed during refresh.")
    if (
        refreshed.canonical_variable_id != binding.canonical_variable_id
        or refreshed.constraint_slices != binding.constraint_slices
    ):
        raise ValueError("CVXPY canonical inverse maps changed during refresh.")
    numeric_version = binding.numeric_version + 1
    return CVXPYProgramBinding(
        refreshed.program,
        refreshed.problem,
        refreshed.solving_chain,
        refreshed.inverse_data,
        refreshed.variable_slices,
        refreshed.parameter_topology_id,
        refreshed.binding_id,
        refreshed.numeric_fingerprint,
        numeric_version,
        _numeric_binding_id(
            refreshed.binding_id,
            refreshed.numeric_fingerprint,
            numeric_version,
        ),
        refreshed.canonical_variable_id,
        refreshed.constraint_slices,
    )


def restore_cvxpy_solution(
    binding: CVXPYProgramBinding,
    result: ConvexProgramResult,
    /,
) -> dict[int, np.ndarray]:
    """Restore original CVXPY variable values from one matching canonical result."""
    if not isinstance(binding, CVXPYProgramBinding):
        raise TypeError("binding must be a CVXPYProgramBinding.")
    if not isinstance(result, ConvexProgramResult):
        raise TypeError("result must be a ConvexProgramResult.")
    if not bool(np.asarray(result.successful)):
        raise ValueError(
            "CVXPY solution restoration requires a successful optimal result."
        )
    current_fingerprint = _program_numeric_fingerprint(binding.program)
    current_binding_id = _numeric_binding_id(
        binding.binding_id,
        current_fingerprint,
        binding.numeric_version,
    )
    if (
        current_fingerprint != binding.numeric_fingerprint
        or current_binding_id != binding.numeric_binding_id
    ):
        raise ValueError("CVXPY numeric binding is stale.")
    provenance = result.provenance
    if (
        provenance.problem_id != binding.program.problem_id
        or provenance.structure_id != binding.program.structure_id
        or int(np.asarray(provenance.numeric_version)) != binding.numeric_version
        or provenance.numeric_binding_id != binding.numeric_binding_id
    ):
        raise ValueError("Convex result does not match the CVXPY numeric binding.")

    primal = np.asarray(result.primal)
    if primal.shape != (binding.program.num_variables,):
        raise ValueError("Canonical primal shape does not match the CVXPY inverse map.")
    dual = np.asarray(result.cone_dual)
    if dual.shape != (binding.program.num_constraints,):
        raise ValueError("Canonical dual shape does not match the CVXPY inverse map.")

    restored = {
        item.variable_id: primal[item.start : item.stop].reshape(item.shape, order="F")
        for item in binding.variable_slices
    }
    if binding.solving_chain is not None and binding.canonical_variable_id >= 0:
        from cvxpy.reductions.solution import Solution

        dual_vars = {
            item.constraint_id: dual[item.start : item.stop].reshape(
                item.shape, order="F"
            )
            for item in binding.constraint_slices
        }
        canonical_solution = Solution(
            "optimal",
            float(np.asarray(result.objective)),
            {binding.canonical_variable_id: primal},
            dual_vars,
            {},
        )
        original_solution = canonical_solution
        reductions = binding.solving_chain.reductions[:-1]
        inverse_data = binding.inverse_data[:-1]
        for reduction, inverse_item in reversed(
            tuple(zip(reductions, inverse_data, strict=True))
        ):
            original_solution = reduction.invert(original_solution, inverse_item)
        restored = {
            int(variable_id): np.asarray(value)
            for variable_id, value in original_solution.primal_vars.items()
        }
        binding.problem.unpack(original_solution)
    else:
        variables = {
            int(variable.id): variable for variable in binding.problem.variables()
        }
        for variable_id, value in restored.items():
            if variable_id in variables:
                variables[variable_id].value = value
    return restored


__all__ = [
    "CVXPYProgramBinding",
    "CVXPYVariableSlice",
    "CVXPYConstraintSlice",
    "export_cvxpy_program",
    "refresh_cvxpy_program",
    "import_cvxpy_problem",
    "restore_cvxpy_solution",
]

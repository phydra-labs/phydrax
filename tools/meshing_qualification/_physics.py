# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Shared manufactured diffusion, independent integral, and compiler evidence owners."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Any, NamedTuple, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tools.compile_counter import CompileCounter
from tools.meshing_qualification._runtime import (
    _CorpusJitEntry,
    _measured,
    qualification_failure,
)
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


def _manufactured_value(points: Any) -> Any:
    return jnp.sum(points * points, axis=-1)


def _manufactured_source(points: Any, parameters: Any) -> Any:
    del parameters
    return jnp.full(points.shape[:-1], -2.0 * points.shape[-1], dtype=points.dtype)


def _material_manufactured_value(points: Any) -> Any:
    x = points[..., 0]
    return jnp.where(x <= 1.0, x * x, 0.5 * x * x + 0.5)


def _material_manufactured_source(points: Any, context: Any) -> Any:
    del context
    return jnp.full(points.shape[:-1], -2.0, dtype=points.dtype)


def _prepare_corpus_solver(
    result: phx.meshing.CellMeshingResult,
    *,
    material_x: bool = True,
    solution_degree: int = 1,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[Any, Any, Any]:

    mesh = result.mesh
    family = "triangle" if mesh.topological_dimension == 2 else "tetrahedron"
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(family, solution_degree)
        ),
        coordinate_spec=result.geometry,
        budget=budget,
    ).prepare(budget=budget)
    if mesh.topological_dimension == 3 and material_x:
        zones = {
            zone.name: zone
            for zone in result.zones
            if zone.role is phx.meshing.MeshZoneRole.REGION
        }
        if set(zones) != {"left", "right"}:
            raise qualification_failure(
                "_prepare_corpus_solver",
                ("The material diffusion consumer requires both source regions."),
                {"zones": zones},
            )
        diffusion = tuple(
            phx.equations.DiffusionAction(
                "u",
                conductivity,
                action_id=f"{name}-volume-conductivity",
                domain=space.integration_domain(
                    "cell", phx.meshing.resolve_mesh_scope(mesh, zones[name].scope)
                ),
            )
            for name, conductivity in (("left", 1.0), ("right", 2.0))
        )
        forcing = _material_manufactured_source
        exact = _material_manufactured_value
        coefficient_id = "material-flux-continuous-quadratic"
    else:
        diffusion = (phx.equations.DiffusionAction("u", 1.0),)
        forcing = _manufactured_source
        exact = _manufactured_value
        coefficient_id = "quadratic-exact-laplacian"
    form = phx.equations.FiniteElementForm(
        "native-corpus-manufactured-diffusion",
        "u",
        (
            *diffusion,
            phx.equations.SourceAction(
                "u", phx.equations.coefficient(forcing, coefficient_id=coefficient_id)
            ),
        ),
    )
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=exact, budget=budget
    )
    return space, problem, constraint.free_dofs


@eqx.filter_jit
def _solve_corpus_problem(
    problem: Any,
    seed: Any,
    policy: phx.linalg.LinearSolvePolicy | None = None,
) -> tuple[Any, Any]:
    system, rhs = problem.linear_system()
    residual = rhs - system.operator.mv(seed)
    solved = phx.linalg.solve(system, residual, policy=policy)
    return problem.expand(seed + solved.value), solved


class CorpusPreconditioning(StrEnum):
    """Explicit corpus solve preconditioning, added to the caller's authored policy.

    ``NONE`` solves with the authored policy unchanged. ``EXACT_JACOBI`` binds a
    canonical linalg diagonal inverse of the exact FE system diagonal; the
    authored method, tolerances, step limits and true-residual stop test are kept.
    The changed algorithm and preconditioner identity are recorded in evidence.
    """

    NONE = "none"
    EXACT_JACOBI = "exact-jacobi"


class CorpusPreconditionerEvidence(TypedDict):
    kind: str
    preconditioner_id: str
    diagonal_method: str
    diagonal_operator_id: str
    size: int
    minimum_diagonal: float
    maximum_diagonal: float
    positive_definite: str


class _CorpusEpochSystem(NamedTuple):
    """One assembled system retaining its immutable source problem."""

    problem: Any
    system: phx.linalg.LinearSystem
    rhs: Any


@eqx.filter_jit
def _assemble_corpus_epoch_system(
    problem: Any,
    preconditioning: CorpusPreconditioning,
) -> tuple[phx.linalg.LinearSystem, Any, Any]:
    system, rhs = problem.linear_system()
    diagonal = (
        problem.exact_diagonal()
        if preconditioning is CorpusPreconditioning.EXACT_JACOBI
        else None
    )
    return system, rhs, diagonal


def _prepare_corpus_epoch_system(
    problem: Any, preconditioning: CorpusPreconditioning
) -> tuple[_CorpusEpochSystem, Any]:
    system, rhs, diagonal = _assemble_corpus_epoch_system(problem, preconditioning)
    return _CorpusEpochSystem(problem, system, rhs), diagonal


def _preconditioner_refusal(reason: str, values: dict[str, Any]) -> RuntimeError:
    return qualification_failure(
        "_corpus_solve_policy",
        json.dumps(
            {"kind": "native-solver-preconditioner-refusal", "reason": reason},
            sort_keys=True,
        ),
        values,
    )


def _corpus_solve_policy(
    epoch: _CorpusEpochSystem,
    diagonal: Any,
    preconditioning: CorpusPreconditioning,
    policy: phx.linalg.LinearSolvePolicy | None,
) -> tuple[phx.linalg.LinearSolvePolicy | None, CorpusPreconditionerEvidence | None]:
    """Bind the requested prepared preconditioner to the caller's authored policy."""
    if preconditioning is CorpusPreconditioning.NONE:
        return policy, None
    base = phx.linalg.LinearSolvePolicy() if policy is None else policy
    if base.preconditioning is not None:
        raise ValueError(
            "_corpus_solve_policy.policy must not declare preconditioning when "
            f"{preconditioning.value!r} is requested; got an existing preconditioning policy"
        )
    space = epoch.system.operator.source
    if not isinstance(space, phx.linalg.ArraySpace) or not isinstance(
        space.pairing, phx.linalg.EuclideanPairing
    ):
        # The FE diagonal is the system diagonal only when inverse Riesz is identity.
        raise _preconditioner_refusal(
            "system-space-not-euclidean", {"space": type(space).__name__}
        )
    identifier = (
        f"corpus-exact-jacobi:{diagonal.diagonal_id}:{epoch.system.operator.operator_id}"
    )
    # The canonical owner refuses nonfinite/zero entries and verifies positivity.
    action = phx.linalg.DiagonalPreconditioner(
        diagonal.diagonal, space=space, preconditioner_id=identifier
    )
    if not action.properties.certifies("positive_definite"):
        raise _preconditioner_refusal(
            "positive-definite-claim-unverified", {"preconditioner": identifier}
        )
    prepared = eqx.tree_at(
        lambda selected: selected.preconditioning,
        base,
        phx.linalg.PreconditioningPolicy(action),
        is_leaf=lambda value: value is None,
    )
    values = np.asarray(diagonal.diagonal)
    return prepared, {
        "kind": preconditioning.value,
        "preconditioner_id": action.preconditioner_id,
        "diagonal_method": diagonal.method,
        "diagonal_operator_id": diagonal.operator_id,
        "size": int(values.size),
        "minimum_diagonal": float(values.min()),
        "maximum_diagonal": float(values.max()),
        "positive_definite": "verified",
    }


def _ready_policy[T](
    prepared: tuple[phx.linalg.LinearSolvePolicy | None, T],
) -> tuple[phx.linalg.LinearSolvePolicy | None, T]:
    jax.block_until_ready(prepared[0])
    return prepared


@eqx.filter_jit
def _solve_corpus_epoch_system(
    epoch: _CorpusEpochSystem,
    seed: Any,
    policy: phx.linalg.LinearSolvePolicy | None = None,
) -> tuple[Any, Any]:
    residual = epoch.rhs - epoch.system.operator.mv(seed)
    solved = phx.linalg.solve(epoch.system, residual, policy=policy)
    return epoch.problem.expand(seed + solved.value), solved


def _physical_inventory(
    space: phx.discretization.FiniteElementDiscretization, values: Any
) -> float:
    """Integrate actual routed P1/P2 nodal fields by independent simplex moments."""
    mesh = space.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    d = mesh.topological_dimension
    inventory = 0.0
    for block, element, route, transform in zip(
        mesh.blocks,
        space.elements[0],
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        strict=True,
    ):
        local = np.einsum("cij,cj->ci", np.asarray(transform), np.asarray(values)[route])
        vertices = [dofs[0] for dofs in element.entity_dofs[0]]
        vertex_sum = np.sum(local[:, vertices], axis=1)
        if element.degree == 1:
            average = vertex_sum / (d + 1)
        elif element.degree == 2:
            edges = [dofs[0] for dofs in element.entity_dofs[1]]
            # E[lambda_i(2lambda_i-1)] = (2-d)/((d+1)(d+2));
            # E[4lambda_i lambda_j] = 4/((d+1)(d+2)).
            average = (
                (2 - d) * vertex_sum + 4.0 * np.sum(local[:, edges], axis=1)
            ) / ((d + 1) * (d + 2))
        else:
            raise ValueError("corpus.physical_inventory.space must use simplex P1 or P2")
        measures = _simplex_measures(points, np.asarray(block.vertices))
        inventory += float(np.sum(measures * average))
    return inventory


def _physical_error(
    mesh: phx.discretization.CellMesh,
    values: Any,
    *,
    material_x: bool = False,
    space: phx.discretization.FiniteElementDiscretization | None = None,
) -> float:
    """Exact independent affine-simplex integral using the actual FE DOF routes.

    P2 uses the nodal barycentric basis, not solver quadrature or tabulation.
    The no-space route retains the original vertex-associated P1 contract.
    """

    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = points[cells]
    d = mesh.topological_dimension
    measures = _simplex_measures(points, cells)
    # u_h - |x|² is a barycentric polynomial of degree two. The normalized
    # Dirichlet(1,...,1) moments integrate its square exactly, without sharing
    # FEM quadrature, tabulation or assembly.
    if space is None:
        if np.asarray(values).shape != (points.shape[0],):
            raise ValueError(
                "corpus.physical_error.values must supply the actual FE space for nonvertex DOFs"
            )
        nodal = np.asarray(values)[cells]
    elif all(element.degree == 1 for element in space.elements[0]):
        nodal = np.concatenate(
            [
                np.einsum(
                    "cij,cj->ci", np.asarray(transform), np.asarray(values)[route]
                )[:, [dofs[0] for dofs in element.entity_dofs[0]]]
                for element, route, transform in zip(
                    space.elements[0],
                    space.dof_maps[0].cell_dofs,
                    space.dof_maps[0].cell_transforms,
                    strict=True,
                )
            ]
        )
    else:
        coefficient = _quadratic_error_coefficients(mesh, values, space, material_x)
        return _quadratic_coefficient_l2(measures, coefficient, d)
    if material_x:
        right = corners.mean(axis=1)[:, 0] > 1.0
        gram = corners[:, :, 0, None] * corners[:, None, :, 0]
        gram = (
            gram * np.where(right, 0.5, 1.0)[:, None, None]
            + np.where(right, 0.5, 0.0)[:, None, None]
        )
    else:
        gram = corners @ np.swapaxes(corners, -1, -2)
    coefficient = -gram
    coefficient += 0.5 * (nodal[:, :, None] + nodal[:, None, :])
    return _quadratic_coefficient_l2(measures, coefficient, d)


def _quadratic_error_coefficients(
    mesh: phx.discretization.CellMesh,
    values: Any,
    space: phx.discretization.FiniteElementDiscretization,
    material_x: bool,
) -> np.ndarray:
    """Construct u_h-q in homogeneous barycentric degree-two coefficients."""
    coefficients = []
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    for block, element, route, transform in zip(
        mesh.blocks,
        space.elements[0],
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        strict=True,
    ):
        family = "triangle" if mesh.topological_dimension == 2 else "tetrahedron"
        if element.element_id != phx.discretization.lagrange_element(family, 2).element_id:
            raise ValueError("corpus.physical_error.space must use scalar simplex P1 or P2")
        corners = points[np.asarray(block.vertices)]
        local = np.einsum("cij,cj->ci", np.asarray(transform), np.asarray(values)[route])
        reference = np.asarray(element.reference_nodes, dtype=np.float64)
        barycentric = np.column_stack((1.0 - reference.sum(axis=1), reference))
        nodes = np.einsum("li,cid->cld", barycentric, corners)
        if material_x:
            x = nodes[..., 0]
            exact = np.where(x <= 1.0, x * x, 0.5 * x * x + 0.5)
        else:
            exact = np.sum(nodes * nodes, axis=-1)
        defect = local - exact
        vertices = np.asarray([dofs[0] for dofs in element.entity_dofs[0]])
        vertex_defect = defect[:, vertices]
        # phi_i=lambda_i(2lambda_i-1), phi_ij=4lambda_i lambda_j.
        # Homogenize the linear terms with sum(lambda)=1.
        coefficient = -0.5 * (
            vertex_defect[:, :, None] + vertex_defect[:, None, :]
        )
        for i in range(corners.shape[1]):
            coefficient[:, i, i] = vertex_defect[:, i]
        edges = phx.discretization.reference_cell_topology(family).entities[1]
        for (i, j), dofs in zip(edges, element.entity_dofs[1], strict=True):
            coefficient[:, i, j] += 2.0 * defect[:, dofs[0]]
            coefficient[:, j, i] = coefficient[:, i, j]
        coefficients.append(coefficient)
    return np.concatenate(coefficients)


def _quadratic_coefficient_l2(
    measures: np.ndarray, coefficient: np.ndarray, d: int
) -> float:
    """Integrate squared barycentric polynomials by exact Dirichlet moments."""
    n = d + 1
    integral = np.zeros(coefficient.shape[0], dtype=np.float64)
    denominator = math.factorial(d + 4) / math.factorial(d)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    exponents = np.bincount((i, j, k, l), minlength=n)
                    moment = (
                        np.prod([math.factorial(int(e)) for e in exponents]) / denominator
                    )
                    integral += coefficient[:, i, j] * coefficient[:, k, l] * moment
    return float(np.sqrt(max(0.0, float(np.sum(measures * integral)))))


type CorpusSolverNumeric = int | float | list[CorpusSolverNumeric]


class CorpusSolverEvidence(TypedDict):
    successful: bool
    status: CorpusSolverNumeric
    iterations: CorpusSolverNumeric
    matvec_count: CorpusSolverNumeric
    residual_norm: CorpusSolverNumeric
    relative_residual: CorpusSolverNumeric
    rank: CorpusSolverNumeric
    condition_estimate: CorpusSolverNumeric | None
    condition_status: str
    backend: str
    method: str
    method_reason: str
    preconditioner_id: str | None
    preconditioner_plan_id: str | None
    preconditioning_side: str | None
    preconditioner_refresh: str | None


class CorpusCompilerEvidence(TypedDict):
    temporary_bytes: int | None
    output_bytes: int | None
    argument_bytes: int | None
    generated_code_bytes: int | None
    memory_status: str
    compilation_count: int
    warm_samples_seconds: list[float]
    compilation_count_scope: str
    first_execution_compilation_count: int
    warm_recompilation_count: int
    execution_scope: str
    assembly_seconds: float
    assembly_invocations: int
    assembly_compilation_count: int
    assembly_compile_seconds: float
    assembly_binding_id: str
    preconditioning: str
    preconditioner: CorpusPreconditionerEvidence | None
    preconditioner_preparation_seconds: float | None
    preconditioner_preparation_compilation_count: int | None
    solver: CorpusSolverEvidence


def _compiled_solver_stages(
    problem: Any,
    seed: Any,
    repeats: int,
    *,
    policy: phx.linalg.LinearSolvePolicy | None = None,
    preconditioning: CorpusPreconditioning = CorpusPreconditioning.NONE,
) -> tuple[Any, CorpusCompilerEvidence, dict[str, float]]:

    with CompileCounter(stage="epoch_system_assembly") as assembly_compiler:
        (epoch, diagonal), assembly = _measured(
            lambda: jax.block_until_ready(
                _prepare_corpus_epoch_system(problem, preconditioning)
            )
        )
    preparation_compiler: CompileCounter | None = None
    preparation_seconds: float | None = None
    if preconditioning is CorpusPreconditioning.NONE:
        solve_policy, preconditioner = policy, None
    else:
        with CompileCounter(
            stage="epoch_preconditioner_preparation"
        ) as preparation_compiler:
            (solve_policy, preconditioner), preparation = _measured(
                lambda: _ready_policy(
                    _corpus_solve_policy(epoch, diagonal, preconditioning, policy)
                )
            )
        preparation_seconds = preparation["wall_seconds"]
    entry = _solve_corpus_epoch_system
    if not isinstance(entry, _CorpusJitEntry):
        raise TypeError(
            f"_compiled_solver_stages.entry must expose the compiled _CorpusJitEntry lowering boundary; got {type(entry).__name__!r}."
        )
    lowered, lowering = _measured(lambda: entry.lower(epoch, seed, solve_policy))
    compiled, compilation = _measured(lowered.compile)
    memory = compiled.compiled.memory_analysis()
    with CompileCounter(stage="solver_first_execution") as first_compiler:
        first, first_stage = _measured(
            lambda: jax.block_until_ready(compiled(epoch, seed, solve_policy))
        )
    warm = []
    with CompileCounter(stage="solver_warm_execution") as warm_compiler:
        for _ in range(repeats):
            _, stage = _measured(
                lambda: jax.block_until_ready(compiled(epoch, seed, solve_policy))
            )
            warm.append(stage["wall_seconds"])
    values, solved = first
    condition = np.asarray(solved.diagnostics.condition_estimate)
    condition_available = bool(np.all(np.isfinite(condition)))
    successful = bool(np.all(np.asarray(solved.successful)))
    provenance = solved.provenance
    compiler: CorpusCompilerEvidence = {
        "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        "output_bytes": None if memory is None else memory.output_size_in_bytes,
        "argument_bytes": None if memory is None else memory.argument_size_in_bytes,
        "generated_code_bytes": None
        if memory is None
        else memory.generated_code_size_in_bytes,
        "memory_status": "unavailable" if memory is None else "measured",
        "compilation_count": 1,
        "warm_samples_seconds": warm,
        "compilation_count_scope": "explicit-AOT-compile-invocations",
        "first_execution_compilation_count": first_compiler.count,
        "warm_recompilation_count": warm_compiler.count,
        "execution_scope": "epoch-preassembled-system-residual-solve-and-expand",
        "assembly_seconds": assembly["wall_seconds"],
        "assembly_invocations": 1,
        "assembly_compilation_count": assembly_compiler.count,
        "assembly_compile_seconds": assembly_compiler.seconds,
        "assembly_binding_id": problem.compilation_id,
        "preconditioning": preconditioning.value,
        "preconditioner": preconditioner,
        "preconditioner_preparation_seconds": preparation_seconds,
        "preconditioner_preparation_compilation_count": (
            None if preparation_compiler is None else preparation_compiler.count
        ),
        "solver": {
            "successful": successful,
            "status": np.asarray(solved.status).tolist(),
            "iterations": np.asarray(solved.diagnostics.iterations).tolist(),
            "matvec_count": np.asarray(solved.diagnostics.matvec_count).tolist(),
            "residual_norm": np.asarray(solved.diagnostics.residual_norm).tolist(),
            "relative_residual": np.asarray(
                solved.diagnostics.relative_residual
            ).tolist(),
            "rank": np.asarray(solved.diagnostics.rank).tolist(),
            # The solver's own estimate; a preconditioned Krylov estimate is never
            # substituted for the original operator's condition.
            "condition_estimate": condition.tolist() if condition_available else None,
            "condition_status": "reported" if condition_available else "unavailable",
            "backend": provenance.backend,
            "method": provenance.method,
            "method_reason": provenance.reason,
            "preconditioner_id": provenance.preconditioner_id,
            "preconditioner_plan_id": provenance.preconditioner_plan_id,
            "preconditioning_side": provenance.preconditioning_side,
            "preconditioner_refresh": provenance.preconditioner_refresh,
        },
    }
    stages = {
        "solver_assembly": assembly["wall_seconds"],
        "solver_lowering": lowering["wall_seconds"],
        "solver_compile": compilation["wall_seconds"],
        "solver_first_execution": first_stage["wall_seconds"],
        "solver_warm_execution": float(np.median(warm)),
    }
    if preparation_seconds is not None:
        stages["solver_preconditioner_preparation"] = preparation_seconds
    if not successful or warm_compiler.count or first_compiler.count:
        raise qualification_failure(
            "_compiled_solver_stages",
            (
                json.dumps(
                    {
                        "kind": (
                            "native-solver-failure"
                            if not successful
                            else "native-solver-recompilation"
                        ),
                        "message": (
                            "Manufactured diffusion solver did not converge."
                            if not successful
                            else "The AOT-compiled solve recompiled during execution."
                        ),
                        "compiler": compiler,
                        "stages_seconds": stages,
                    },
                    sort_keys=True,
                )
            ),
            {"successful": successful},
        )
    return values, compiler, stages

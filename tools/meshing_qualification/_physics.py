# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Shared manufactured diffusion, independent integral, and compiler evidence owners."""

from __future__ import annotations

import json
import math
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
    result: phx.meshing.CellMeshingResult, *, material_x: bool = True
) -> tuple[Any, Any, Any]:

    mesh = result.mesh
    family = "triangle" if mesh.topological_dimension == 2 else "tetrahedron"
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(family, 1)
        ),
        coordinate_spec=result.geometry,
    ).prepare()
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
        form, space, constraint=constraint, dirichlet_values=exact
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


class _CorpusEpochSystem(NamedTuple):
    """One assembled system retaining its immutable source problem."""

    problem: Any
    system: phx.linalg.LinearSystem
    rhs: Any


@eqx.filter_jit
def _assemble_corpus_epoch_system(
    problem: Any,
) -> tuple[phx.linalg.LinearSystem, Any]:
    return problem.linear_system()


def _prepare_corpus_epoch_system(problem: Any) -> _CorpusEpochSystem:
    system, rhs = _assemble_corpus_epoch_system(problem)
    return _CorpusEpochSystem(problem, system, rhs)


@eqx.filter_jit
def _solve_corpus_epoch_system(
    epoch: _CorpusEpochSystem,
    seed: Any,
    policy: phx.linalg.LinearSolvePolicy | None = None,
) -> tuple[Any, Any]:
    residual = epoch.rhs - epoch.system.operator.mv(seed)
    solved = phx.linalg.solve(epoch.system, residual, policy=policy)
    return epoch.problem.expand(seed + solved.value), solved


def _physical_error(
    mesh: phx.discretization.CellMesh,
    values: Any,
    *,
    material_x: bool = False,
) -> float:
    """Independent P1-versus-quadratic L2 integral on affine simplices."""

    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = points[cells]
    d = mesh.topological_dimension
    measures = _simplex_measures(points, cells)
    # u_h - |x|² is a barycentric polynomial of degree two. The normalized
    # Dirichlet(1,...,1) moments integrate its square exactly, without sharing
    # FEM quadrature, tabulation or assembly.
    nodal = np.asarray(values)[cells]
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
    n = d + 1
    integral = np.zeros(cells.shape[0], dtype=np.float64)
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
    residual_norm: CorpusSolverNumeric
    relative_residual: CorpusSolverNumeric
    rank: CorpusSolverNumeric
    condition_estimate: CorpusSolverNumeric | None
    condition_status: str


class CorpusCompilerEvidence(TypedDict):
    temporary_bytes: int | None
    output_bytes: int | None
    argument_bytes: int | None
    generated_code_bytes: int | None
    memory_status: str
    compilation_count: int
    warm_samples_seconds: list[float]
    compilation_count_scope: str
    warm_recompilation_count: int
    execution_scope: str
    assembly_seconds: float
    assembly_invocations: int
    assembly_compilation_count: int
    assembly_compile_seconds: float
    assembly_binding_id: str
    solver: CorpusSolverEvidence


def _compiled_solver_stages(
    problem: Any,
    seed: Any,
    repeats: int,
    *,
    policy: phx.linalg.LinearSolvePolicy | None = None,
) -> tuple[Any, CorpusCompilerEvidence, dict[str, float]]:

    with CompileCounter(stage="epoch_system_assembly") as assembly_compiler:
        epoch, assembly = _measured(
            lambda: jax.block_until_ready(_prepare_corpus_epoch_system(problem))
        )
    entry = _solve_corpus_epoch_system
    if not isinstance(entry, _CorpusJitEntry):
        raise TypeError(
            f"_compiled_solver_stages.entry must expose the compiled _CorpusJitEntry lowering boundary; got {type(entry).__name__!r}."
        )
    lowered, lowering = _measured(lambda: entry.lower(epoch, seed, policy))
    compiled, compilation = _measured(lowered.compile)
    memory = compiled.compiled.memory_analysis()
    first, first_stage = _measured(
        lambda: jax.block_until_ready(compiled(epoch, seed, policy))
    )
    warm = []
    for _ in range(repeats):
        _, stage = _measured(lambda: jax.block_until_ready(compiled(epoch, seed, policy)))
        warm.append(stage["wall_seconds"])
    values, solved = first
    condition = np.asarray(solved.diagnostics.condition_estimate)
    condition_available = bool(np.all(np.isfinite(condition)))
    successful = bool(np.all(np.asarray(solved.successful)))
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
        "warm_recompilation_count": 0,
        "execution_scope": "epoch-preassembled-system-residual-solve-and-expand",
        "assembly_seconds": assembly["wall_seconds"],
        "assembly_invocations": 1,
        "assembly_compilation_count": assembly_compiler.count,
        "assembly_compile_seconds": assembly_compiler.seconds,
        "assembly_binding_id": problem.compilation_id,
        "solver": {
            "successful": successful,
            "status": np.asarray(solved.status).tolist(),
            "iterations": np.asarray(solved.diagnostics.iterations).tolist(),
            "residual_norm": np.asarray(solved.diagnostics.residual_norm).tolist(),
            "relative_residual": np.asarray(
                solved.diagnostics.relative_residual
            ).tolist(),
            "rank": np.asarray(solved.diagnostics.rank).tolist(),
            "condition_estimate": condition.tolist() if condition_available else None,
            "condition_status": "reported" if condition_available else "unavailable",
        },
    }
    stages = {
        "solver_assembly": assembly["wall_seconds"],
        "solver_lowering": lowering["wall_seconds"],
        "solver_compile": compilation["wall_seconds"],
        "solver_first_execution": first_stage["wall_seconds"],
        "solver_warm_execution": float(np.median(warm)),
    }
    if not successful:
        raise qualification_failure(
            "_compiled_solver_stages",
            (
                json.dumps(
                    {
                        "kind": "native-solver-failure",
                        "message": "Manufactured diffusion solver did not converge.",
                        "compiler": compiler,
                        "stages_seconds": stages,
                    },
                    sort_keys=True,
                )
            ),
            {"successful": successful},
        )
    return values, compiler, stages

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    P1FiniteElementDiscretization,
)
from ..linalg import LinearSolvePolicy, LinearSolveResult


class VariationalProblemIR(StrictModule):
    """Scalar diffusion weak form with explicit volume and boundary functionals."""

    source: Callable[[Array], ArrayLike]
    dirichlet: Callable[[Array], ArrayLike]
    neumann: Callable[[Array], ArrayLike] | None
    diffusion: Array
    name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        field_name: str,
        /,
        *,
        diffusion: ArrayLike,
        source: Callable[[Array], ArrayLike],
        dirichlet: Callable[[Array], ArrayLike],
        neumann: Callable[[Array], ArrayLike] | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        field = str(field_name)
        if not name_ or not field:
            raise ValueError("Variational problem and field names must be non-empty.")
        coefficient = jnp.asarray(diffusion)
        if coefficient.shape != ():
            raise ValueError("Variational diffusion must be scalar.")
        coefficient = eqx.error_if(
            coefficient,
            ~jnp.isfinite(coefficient) | (coefficient <= 0.0),
            "Variational diffusion must be finite and positive.",
        )
        if not callable(source) or not callable(dirichlet):
            raise TypeError("source and dirichlet must be callable.")
        if neumann is not None and not callable(neumann):
            raise TypeError("neumann must be callable or None.")
        self.source = source
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.diffusion = coefficient
        self.name = name_
        self.field_name = field
        self.problem_id = (
            canonical_fingerprint(
                {
                    "kind": "scalar-diffusion-variational-problem",
                    "name": name_,
                    "field": field,
                    "diffusion_shape": list(coefficient.shape),
                    "source": repr(source),
                    "dirichlet": repr(dirichlet),
                    "neumann": None if neumann is None else repr(neumann),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty.")


class CompiledVariationalProblem(StrictModule):
    """P1 weak-form load and boundary realization with full provenance."""

    problem: VariationalProblemIR
    discretization: P1FiniteElementDiscretization
    right_hand_side: Array
    dirichlet_values: Array
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: VariationalProblemIR,
        discretization: P1FiniteElementDiscretization,
        right_hand_side: Array,
        dirichlet_values: Array,
        /,
    ):
        if not isinstance(problem, VariationalProblemIR):
            raise TypeError("problem must be a VariationalProblemIR.")
        if not isinstance(discretization, P1FiniteElementDiscretization):
            raise TypeError("discretization must be a P1FiniteElementDiscretization.")
        if problem.field_name != discretization.field_spaces[0].name:
            raise ValueError(
                "Variational field name must match the finite element space."
            )
        rhs = jnp.asarray(right_hand_side)
        boundary = jnp.asarray(dirichlet_values)
        expected = (int(discretization.vertices.shape[0]),)
        if rhs.shape != expected or boundary.shape != expected:
            raise ValueError("Compiled variational arrays must match the vertex count.")
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-p1-variational-problem",
                "problem": problem.problem_id,
                "discretization": discretization.prepared_id,
            }
        )
        form_key = DiscretizationKey(
            "variational_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        self.problem = problem
        self.discretization = discretization
        self.right_hand_side = rhs
        self.dirichlet_values = boundary
        self.discretization_bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    form_key,
                    "compiled-variational-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.compilation_id = compilation_id

    def solve(
        self,
        /,
        *,
        policy: LinearSolvePolicy | None = None,
    ) -> tuple[Array, LinearSolveResult]:
        return self.discretization.solve_poisson(
            self.right_hand_side / self.problem.diffusion,
            dirichlet_values=self.dirichlet_values,
            policy=policy,
        )


def compile_variational_problem(
    problem: VariationalProblemIR,
    discretization: P1FiniteElementDiscretization,
    /,
) -> CompiledVariationalProblem:
    """Lower a scalar diffusion weak form onto one prepared P1 space."""
    if not isinstance(problem, VariationalProblemIR):
        raise TypeError("problem must be a VariationalProblemIR.")
    if not isinstance(discretization, P1FiniteElementDiscretization):
        raise TypeError(
            "No variational lowering is registered for this discretization type."
        )
    volume_samples = jnp.asarray(problem.source(discretization.quadrature_points))
    expected_volume = (int(discretization.faces.shape[0]), 3)
    if volume_samples.shape != expected_volume:
        raise ValueError(
            f"Variational source must return shape {expected_volume}; "
            f"got {volume_samples.shape}."
        )
    right_hand_side = discretization.assemble_load(volume_samples)
    boundary_edges = discretization.boundary_edges
    if problem.neumann is not None and int(boundary_edges.shape[0]):
        midpoints = 0.5 * (
            discretization.vertices[boundary_edges[:, 0]]
            + discretization.vertices[boundary_edges[:, 1]]
        )
        neumann_values = jnp.asarray(problem.neumann(midpoints))
        if neumann_values.shape != (int(boundary_edges.shape[0]),):
            raise ValueError(
                "Neumann functional must return one value per boundary edge."
            )
        right_hand_side = right_hand_side + discretization.assemble_boundary_load(
            neumann_values
        )
    dirichlet_values = jnp.asarray(problem.dirichlet(discretization.vertices))
    if dirichlet_values.shape == ():
        dirichlet_values = jnp.full(
            (int(discretization.vertices.shape[0]),),
            dirichlet_values,
            dtype=discretization.vertices.dtype,
        )
    if dirichlet_values.shape != (int(discretization.vertices.shape[0]),):
        raise ValueError(
            "Dirichlet functional must return scalar or one value per vertex."
        )
    return CompiledVariationalProblem(
        problem,
        discretization,
        right_hand_side,
        dirichlet_values,
    )


__all__ = [
    "CompiledVariationalProblem",
    "VariationalProblemIR",
    "compile_variational_problem",
]

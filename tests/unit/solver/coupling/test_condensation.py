#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact static condensation and field-split preconditioning of coupled problems.

Two P2 finite-element regions of ``[0, 2] x [0, 1]`` are tied by a side-trace
mortar. References are independent of the condensation: the uncondensed
dense-LU solve of the assembled coupled system, host central differences of
the condensed primal, and NumPy ranks of the materialized pivot blocks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from tests.unit.solver.coupling._cases import (
    build_region,
    couple,
    Coupled,
    dense_policy,
    interface_binding,
    plate_cover,
    Region,
    RegionSpec,
    SMOOTH,
    transmission_law,
)


cpl = phx.solver.coupling
la = phx.linalg

COMPONENTS = (("left", "u"), ("right", "u"))
MULTIPLIER = ("gamma", "multiplier")
NOMINAL = np.asarray([1.3, 0.7])


def _flat(prepared: cpl.PreparedCoupledProblem, state: Any) -> np.ndarray:
    return np.asarray(prepared.state_space.flatten(state))


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(value - reference)) / np.max(np.abs(reference)))


@pytest.fixture(scope="module")
def mortar() -> Coupled:
    return couple(
        build_region(RegionSpec("left", "fe", 0.0, 1.0, 4, 2), SMOOTH),
        build_region(RegionSpec("right", "fe", 1.0, 2.0, 4, 2), SMOOTH),
        "mortar-side-trace",
    )


# The left owner reads its diffusivity (operator) and source scale (load) from
# runtime user arguments; the right owner keeps the manufactured problem.
def _diffusivity(points: Array, context: Any) -> Array:
    return context.user_args["kappa"] * (1.0 + 0.25 * points[..., 1])


def _source(points: Array, context: Any) -> Array:
    return context.user_args["source"] * SMOOTH.source(points)


def _runtime_region(region: Region) -> Region:
    problem = region.problem
    if not isinstance(problem, phx.equations.CompiledFiniteElementProblem):
        raise TypeError("The runtime owner is a finite-element region.")
    discretization = problem.discretization
    if not isinstance(discretization, phx.discretization.FiniteElementDiscretization):
        raise TypeError("The runtime owner is a finite-element region.")
    fixed = np.ones(region.dof_points.shape[0], dtype=np.bool_)
    fixed[region.free_rows] = False
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "poisson-runtime",
            "u",
            (
                phx.equations.DiffusionAction(
                    "u",
                    phx.equations.coefficient(_diffusivity, coefficient_id="kappa"),
                ),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(_source, coefficient_id="source")
                ),
            ),
        ),
        discretization,
        constraint=phx.discretization.dirichlet_constraint(
            discretization, "u", boundary_mask=fixed
        ),
        dirichlet_values=lambda points: SMOOTH.value(jnp.asarray(points)),
    )
    return dataclasses.replace(
        region,
        component=cpl.VariationalComponent(region.spec.name, compiled, field="u"),
        problem=compiled,
    )


def _parameters(theta: Array) -> dict[str, object]:
    return {"kappa": theta[0], "source": theta[1]}


def _parameter(name: str, role: cpl.ParameterRole) -> cpl.ParameterBinding:
    """Scalar input ``name`` of the left owner (operator or load, by role)."""
    port = phx.ValuePort(
        name, event_shape=(), component_ids=("value",), representation="scalar"
    )
    surface = (
        phx.DerivativeSurface.PHYSICAL_PARAMETER
        if role == "coefficient"
        else phx.DerivativeSurface.SOLVER_ARGUMENT
    )
    return cpl.ParameterBinding(
        name,
        port,
        targets=(cpl.RuntimeInput("left", name),),
        role=role,
        derivative=surface,
    )


@pytest.fixture(scope="module")
def runtime() -> cpl.PreparedCoupledProblem:
    left = _runtime_region(build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 2), SMOOTH))
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, 3, 2), SMOOTH)
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    plan = cpl.CoupledProblemPlan(
        "runtime-regions",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=(transmission_law(left, right, binding, "mortar-side-trace"),),
        parameters=(_parameter("kappa", "coefficient"), _parameter("source", "source")),
    )
    return cpl.prepare_coupled_problem(
        plan, interface_owners=(cover,), parameters=_parameters(jnp.asarray(NOMINAL))
    )


@pytest.mark.parametrize(
    "pivot",
    [COMPONENTS, (("left", "u"),)],
    ids=["components-pivot-multiplier-retained", "left-pivot"],
)
def test_exact_condensation_reproduces_the_uncondensed_direct_solution(
    mortar: Coupled, pivot: tuple[tuple[str, str], ...]
) -> None:
    prepared = mortar.prepared
    direct = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    condensed = cpl.condense_coupled_problem(prepared, pivot=pivot)
    result = cpl.solve_condensed_problem(condensed)
    evidence = result.evidence
    reference = _flat(prepared, direct.state)

    assert bool(direct.accepted)
    assert _relative(_flat(prepared, result.solution.state), reference) <= 1.0e-10
    assert bool(result.accepted) and bool(result.solution.native_successful)
    assert all(bool(item.accepted) for item in result.solution.components)
    assert bool(result.solution.interface("gamma").accepted(result.solution.tolerance))
    assert evidence.pivot_paths == pivot
    assert evidence.pivot_size + evidence.retained_size == prepared.state_space.size
    assert int(evidence.rank) == evidence.pivot_size
    assert bool(evidence.full_rank)
    assert int(evidence.pivot_status) == int(la.LinearSolveStatus.SUCCESS)
    # One batched pivot solve per retained column plus the pivot load.
    assert np.asarray(result.elimination.status).shape == (evidence.retained_size + 1,)
    assert np.all(np.asarray(result.elimination.successful))
    assert bool(result.schur.successful) and bool(result.reconstruction.successful)


def _state_map(
    solve: Callable[[dict[str, object]], Any], prepared: cpl.PreparedCoupledProblem
) -> Callable[[Array], Array]:
    def state(theta: Array) -> Array:
        return prepared.state_space.flatten(solve(_parameters(theta)).state)

    return state


@pytest.mark.parametrize(
    "direction",
    [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])],
    ids=["operator-diffusivity", "load-source"],
)
def test_condensed_parameter_derivatives_match_the_uncondensed_solution_map(
    runtime: cpl.PreparedCoupledProblem, direction: np.ndarray
) -> None:
    prepared = runtime
    condensed = cpl.condense_coupled_problem(
        prepared, pivot=COMPONENTS, parameters=_parameters(jnp.asarray(NOMINAL))
    )
    # The uncondensed mortar system is indefinite; its exact reference derivative
    # reuses the primal factors.
    policy = la.LinearSolvePolicy(
        la.DenseLU(),
        differentiation=la.DifferentiationPolicy("mathematical"),
        derivative_solve=la.LinearDerivativeSolvePolicy(route="primal-factors"),
    )

    def condensed_state(theta: Array) -> Array:
        result = cpl.solve_condensed_problem(condensed, parameters=_parameters(theta))
        return prepared.state_space.flatten(result.solution.state)

    uncondensed_state = _state_map(
        lambda parameters: cpl.solve_coupled_problem(
            prepared, parameters=parameters, policy=policy
        ),
        prepared,
    )
    theta = jnp.asarray(NOMINAL)
    tangent = jnp.asarray(direction)
    _, condensed_jvp = jax.jvp(condensed_state, (theta,), (tangent,))
    _, uncondensed_jvp = jax.jvp(uncondensed_state, (theta,), (tangent,))
    step = 1.0e-4
    difference = (
        np.asarray(condensed_state(theta + step * tangent))
        - np.asarray(condensed_state(theta - step * tangent))
    ) / (2.0 * step)
    weights = jnp.asarray(
        np.random.default_rng(3).standard_normal(prepared.state_space.size)
    )
    _, pullback = jax.vjp(condensed_state, theta)
    (condensed_vjp,) = pullback(weights)
    _, reference_pullback = jax.vjp(uncondensed_state, theta)
    (uncondensed_vjp,) = reference_pullback(weights)
    result = cpl.solve_condensed_problem(condensed, parameters=_parameters(theta))

    assert bool(result.accepted) and bool(result.derivative_valid)
    assert condensed.derivative_capability.admits("arguments")
    assert np.max(np.abs(np.asarray(condensed_jvp))) > 1.0e-3
    assert _relative(np.asarray(condensed_jvp), np.asarray(uncondensed_jvp)) <= 1.0e-9
    assert _relative(np.asarray(condensed_jvp), difference) <= 1.0e-6
    np.testing.assert_allclose(
        np.asarray(condensed_vjp), np.asarray(uncondensed_vjp), rtol=1.0e-9
    )
    np.testing.assert_allclose(
        float(jnp.dot(weights, condensed_jvp)),
        float(jnp.dot(condensed_vjp, tangent)),
        rtol=1.0e-10,
    )


def test_rank_deficient_pivot_is_refused_with_its_evidence() -> None:
    # The pure-Neumann right owner is singular on its own; the mortar ties it to
    # the Dirichlet-anchored left owner, so the coupled system is regular.
    coupled = couple(
        build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 1), SMOOTH),
        build_region(RegionSpec("right", "fe", 1.0, 2.0, 3, 1, dirichlet="none"), SMOOTH),
        "mortar-side-trace",
    )
    prepared = coupled.prepared
    selection = la.BlockSelection(prepared.state_space, (("pivot", (("right", "u"),)),))
    system, _ = prepared.linear_system()
    pivot = np.asarray(
        la.materialize(
            la.select_block_operator(system.operator, selection, selection),
            la.MaterializationPolicy(),
        )
    )
    size = pivot.shape[0]
    whole = np.asarray(la.materialize(system.operator, la.MaterializationPolicy()))

    assert prepared.nullspace_policy is None
    assert np.linalg.matrix_rank(pivot) == size - 1
    assert np.linalg.matrix_rank(whole) == whole.shape[0]
    with pytest.raises(
        ValueError,
        match=rf"rank-deficient pivot: numerical rank {size - 1} of pivot size {size}",
    ):
        cpl.condense_coupled_problem(
            prepared, pivot=(("right", "u"),), factorization=la.FactorizationPolicy("svd")
        )
    # The regular complement pivot of the same problem is admitted.
    admitted = cpl.condense_coupled_problem(
        prepared, pivot=(("left", "u"),), factorization=la.FactorizationPolicy("svd")
    )
    assert bool(cpl.solve_condensed_problem(admitted).accepted)


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"pivot": ()}, "non-empty pivot"),
        ({"pivot": (*COMPONENTS, MULTIPLIER)}, "non-empty retained set"),
        ({"pivot": (("left", "v"),)}, r"unknown solve paths \[\('left', 'v'\)\]"),
        (
            {
                "pivot": COMPONENTS,
                "materialization": la.MaterializationPolicy(max_entries=100),
            },
            r"pivot block needs 12544 dense entries",
        ),
        (
            {"pivot": COMPONENTS, "condition_limit": 1.0e12},
            "factorization 'lu' reports no condition estimate",
        ),
        (
            {
                "pivot": COMPONENTS,
                "factorization": la.FactorizationPolicy("svd"),
                "condition_limit": 10.0,
            },
            r"ill-conditioned pivot: .* exceeds condition_limit 1\.000000e\+01",
        ),
    ],
    ids=[
        "empty-pivot",
        "empty-retained",
        "unknown-path",
        "materialization-budget",
        "unknown-condition",
        "condition-limit",
    ],
)
def test_unqualified_condensations_are_refused(
    mortar: Coupled, keywords: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cpl.condense_coupled_problem(mortar.prepared, **keywords)


def test_condensation_of_a_gauged_kernel_is_refused() -> None:
    coupled = couple(
        build_region(RegionSpec("left", "fe", 0.0, 1.0, 2, 1, dirichlet="none"), SMOOTH),
        build_region(RegionSpec("right", "fe", 1.0, 2.0, 2, 1, dirichlet="none"), SMOOTH),
        "mortar-side-trace",
        gauge=cpl.CoupledGauge(compatibility="project"),
    )
    with pytest.raises(ValueError, match="declared kernel"):
        cpl.condense_coupled_problem(coupled.prepared, pivot=COMPONENTS)


@pytest.mark.parametrize("mode", ["algorithmic", "rhs-only"])
def test_partial_or_unrolled_pivot_derivatives_are_refused(
    mortar: Coupled, mode: la.DifferentiationMode
) -> None:
    factorization = la.FactorizationPolicy(
        "lu", differentiation=la.DifferentiationPolicy(mode)
    )
    with pytest.raises(
        ValueError, match=f"refuses factorization differentiation mode '{mode}'"
    ):
        cpl.condense_coupled_problem(
            mortar.prepared, pivot=COMPONENTS, factorization=factorization
        )


def test_stopped_condensation_refuses_argument_derivatives(
    runtime: cpl.PreparedCoupledProblem,
) -> None:
    prepared = runtime
    condensed = cpl.condense_coupled_problem(
        prepared,
        pivot=COMPONENTS,
        parameters=_parameters(jnp.asarray(NOMINAL)),
        factorization=la.FactorizationPolicy(
            "lu", differentiation=la.DifferentiationPolicy("none")
        ),
    )

    def state(theta: Array) -> Array:
        result = cpl.solve_condensed_problem(condensed, parameters=_parameters(theta))
        return prepared.state_space.flatten(result.solution.state)

    theta = jnp.asarray(NOMINAL)
    result = cpl.solve_condensed_problem(condensed, parameters=_parameters(theta))

    assert bool(result.accepted) and not bool(result.derivative_valid)
    assert not condensed.derivative_capability.admits("arguments")
    with pytest.raises(ValueError, match="derivative-unsupported"):
        jax.jvp(state, (theta,), (jnp.asarray([1.0, 0.0]),))
    with pytest.raises(ValueError, match="derivative-unsupported"):
        jax.grad(lambda value: jnp.sum(state(value)))(theta)


def _field_split(prepared: cpl.PreparedCoupledProblem) -> la.PreconditioningPolicy:
    """Upper block factorization whose pivot action is an approximate Jacobi sweep."""
    selection = la.BlockSelection(
        prepared.state_space,
        (("primal", COMPONENTS), ("multiplier", (MULTIPLIER,))),
    )
    return la.PreconditioningPolicy(
        la.AdditiveSubspaceCorrectionBuilder(
            (
                la.SubspaceCorrectionTerm(
                    la.BlockRestrictionLinearOperator(selection),
                    la.BlockProlongationLinearOperator(selection),
                    la.BlockFactorizationPreconditionerBuilder(
                        la.JacobiPreconditionerBuilder(),
                        la.DenseInversePreconditionerBuilder(),
                        "upper",
                    ),
                ),
            )
        )
    )


def _krylov(
    prepared: cpl.PreparedCoupledProblem,
    max_steps: int,
    preconditioning: la.PreconditioningPolicy | None,
) -> cpl.CoupledSolution:
    return cpl.solve_coupled_problem(
        prepared,
        policy=la.LinearSolvePolicy(
            la.FGMRES(restart=200),
            tolerance=la.TolerancePolicy(
                relative=1.0e-11, absolute=1.0e-14, max_steps=max_steps
            ),
            preconditioning=preconditioning,
        ),
    )


def _iterations(solution: cpl.CoupledSolution) -> int:
    linear = solution.linear
    if linear is None:
        raise AssertionError("An affine coupled solve reports its linear result.")
    return int(np.asarray(linear.diagnostics.iterations))


def test_approximate_field_split_preconditions_the_original_coupled_system(
    mortar: Coupled,
) -> None:
    prepared = mortar.prepared
    direct = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    plain = _krylov(prepared, 400, None)
    split = _krylov(prepared, 400, _field_split(prepared))
    reference = _flat(prepared, direct.state)

    assert bool(split.native_successful) and bool(split.accepted)
    assert _relative(_flat(prepared, split.state), reference) <= 1.0e-8
    assert _iterations(split) < _iterations(plain)


def test_certification_not_preconditioner_success_gates_acceptance(
    mortar: Coupled,
) -> None:
    starved = _krylov(mortar.prepared, 2, _field_split(mortar.prepared))

    assert not bool(starved.native_successful)
    assert not bool(starved.accepted)

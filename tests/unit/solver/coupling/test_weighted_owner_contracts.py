#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Kernel, identification, solver-argument, and identity contracts of prepared problems.

The owner is a small dense affine system ``A(p) u = b`` whose coordinates are
paired by a non-Euclidean ``DiagonalPairing`` (weights ``w``), so the coupled
linear system identifies rows with states through ``M^{-1} = diag(1 / w)``.
Every reference is computed on the host from the declared matrices with NumPy:
the kernels and compatibility conditions of ``A``, dense coordinate transposes,
and explicit minimum-norm solutions.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


cpl = phx.solver.coupling

_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0, 3.0])
# Weighted path-graph Laplacian: symmetric, kernel and left kernel = span(1).
_LAPLACIAN = np.asarray(
    [
        [2.0, -2.0, 0.0, 0.0, 0.0],
        [-2.0, 3.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 4.0, -3.0, 0.0],
        [0.0, 0.0, -3.0, 5.0, -2.0],
        [0.0, 0.0, 0.0, -2.0, 2.0],
    ]
)
_ONES = np.ones((5, 1)) / np.sqrt(5.0)


class _WeightedOwner(cpl.AbstractSpatialComponent):
    """``(base + p**2 coupling) u = load`` on ``DiagonalPairing(weights)`` coordinates."""

    name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    space: cpl.ComponentSpace = eqx.field(static=True)
    state_blocks: tuple[cpl.ComponentBlock, ...]
    row_blocks: tuple[cpl.ComponentBlock, ...]
    fields: tuple[cpl.ComponentField, ...]
    base: Array
    coupling: Array
    kernel: Array | None

    def __init__(
        self,
        name: str,
        base: np.ndarray,
        /,
        *,
        coupling: np.ndarray | None = None,
        kernel: np.ndarray | None = None,
    ) -> None:
        state = phx.linalg.ArraySpace(
            (base.shape[0],), pairing=phx.linalg.DiagonalPairing(jnp.asarray(_WEIGHTS))
        )
        self.name = name
        self.owner_id = f"weighted-owner-{name}"
        self.space = "full"
        self.state_blocks = (cpl.ComponentBlock("u", state),)
        self.row_blocks = (cpl.ComponentBlock("u", phx.linalg.DualSpace(state)),)
        self.fields = (
            cpl.ComponentField(
                "u",
                state_block="u",
                row_block="u",
                full_space=state,
                constraint=None,
                free_rows=None,
            ),
        )
        self.base = jnp.asarray(base)
        self.coupling = (
            jnp.zeros_like(self.base) if coupling is None else jnp.asarray(coupling)
        )
        self.kernel = None if kernel is None else jnp.asarray(kernel)

    def matrix(self, args: object, /) -> Array:
        if not isinstance(args, Mapping):
            raise TypeError("The weighted owner reads its runtime inputs from a mapping.")
        return self.base + args["p"] ** 2 * self.coupling

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        if not isinstance(args, Mapping):
            raise TypeError("The weighted owner reads its runtime inputs from a mapping.")
        return (self.matrix(args) @ state[0] - args["load"],)

    def linear_operator(self, args: object, /) -> phx.linalg.BlockLinearOperator:
        operator = phx.linalg.DenseLinearOperator(
            self.matrix(args),
            source=self.state_blocks[0].space,
            target=self.row_blocks[0].space,
        )
        return phx.linalg.BlockLinearOperator(
            ((operator,),), source=self.state_space, target=self.row_space
        )

    def lift(self, field: str, args: object, /) -> Array:
        del args
        return self.field(field).full_space.zeros()

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        return None if self.kernel is None else (self.kernel,)

    def boundary_impositions(self) -> tuple[phx.discretization.BoundaryImposition, ...]:
        return ()

    def field_space_id(self, field: str, /) -> str:
        return self.field(field).full_space.space_id


def _plan(
    *owners: _WeightedOwner,
    gauge: cpl.CoupledGauge | None = None,
    parameters: tuple[cpl.ParameterBinding, ...] = (),
    resources: cpl.CoupledResourcePolicy | None = None,
) -> cpl.CoupledProblemPlan:
    return cpl.CoupledProblemPlan(
        "weighted",
        components=owners,
        bindings=(),
        laws=(),
        gauge=gauge,
        parameters=parameters,
        resources=resources,
    )


def _arguments(load: np.ndarray, p: float = 0.0) -> dict[str, object]:
    return {"load": jnp.asarray(load), "p": jnp.asarray(p)}


def _dense(operator: phx.linalg.AbstractLinearOperator, transpose: bool, /) -> np.ndarray:
    space = operator.target if transpose else operator.source
    action = operator.transpose_mv if transpose else operator.mv
    output = operator.source if transpose else operator.target
    return np.stack(
        [
            np.asarray(output.flatten(action(space.unflatten(column))))
            for column in jnp.eye(space.size)
        ],
        axis=1,
    )


def test_identified_system_publishes_its_exact_coordinate_transpose() -> None:
    """``K = M^{-1} A`` has transpose ``A^T M^{-T}``, never ``A^T M``."""
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((5, 5)) + 5.0 * np.eye(5)
    owners = (_WeightedOwner("a", matrix), _WeightedOwner("b", matrix))
    arguments = {owner.name: _arguments(np.zeros(5)) for owner in owners}
    grouped = cpl.CoupledResourcePolicy(
        execution=cpl.CoupledExecutionPolicy(lane_capacity=2)
    )
    identified = np.diag(1.0 / _WEIGHTS) @ matrix
    expected = np.kron(np.eye(2), identified)

    for resources in (None, grouped):
        prepared = cpl.prepare_coupled_problem(
            _plan(*owners, resources=resources), arguments=arguments
        )
        operator = prepared.linear_system(arguments)[0].operator
        np.testing.assert_allclose(_dense(operator, False), expected, atol=1.0e-13)
        np.testing.assert_allclose(_dense(operator, True), expected.T, atol=1.0e-13)
    assert prepared.worksets is not None
    assert prepared.worksets.grouped_components == frozenset({"a", "b"})


def test_gauged_kernel_accepts_exactly_the_compatible_loads() -> None:
    """Compatibility is ``1^T b = 0`` for the row load ``b``, not ``1^T M^{-1} b = 0``."""
    owner = _WeightedOwner("graph", _LAPLACIAN, kernel=_ONES)
    compatible = np.asarray([1.0, -2.0, 0.5, 3.0, -2.5])
    # 1^T M^{-1} b = 0 but 1^T b != 0: compatible only for the misidentified kernel.
    misidentified = _WEIGHTS * compatible
    assert abs(np.sum(compatible)) < 1.0e-14
    assert abs(np.sum(misidentified)) > 1.0
    prepared = cpl.prepare_coupled_problem(
        _plan(owner, gauge=cpl.CoupledGauge(compatibility="error")),
        arguments={"graph": _arguments(compatible)},
    )
    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())

    solution = cpl.solve_coupled_problem(
        prepared, arguments={"graph": _arguments(compatible)}, policy=policy
    )
    u = np.asarray(solution.field("graph", "u"))
    np.testing.assert_allclose(_LAPLACIAN @ u, compatible, atol=1.0e-12)
    # Dense LU reports the singular gauged pivot; the compatibility test and the
    # original equations are what this contract certifies.
    assert solution.linear is not None
    assert float(solution.linear.diagnostics.compatibility_residual) <= 1.0e-14
    assert all(bool(item.accepted) for item in solution.components)
    with pytest.raises(eqx.EquinoxRuntimeError, match="incompatible with the declared"):
        cpl.solve_coupled_problem(
            prepared, arguments={"graph": _arguments(misidentified)}, policy=policy
        )


def test_left_kernel_without_a_right_kernel_is_refused() -> None:
    """``A = L D``: ``A^T 1 = 0`` while ``A 1 = L d != 0``; the kernel pair is unequal."""
    scaling = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    owner = _WeightedOwner("graph", _LAPLACIAN @ scaling, kernel=_ONES)
    assert np.linalg.norm((_LAPLACIAN @ scaling).T @ _ONES) < 1.0e-12
    assert np.linalg.norm(_LAPLACIAN @ scaling @ _ONES) > 1.0

    with pytest.raises(
        ValueError, match="0-dimensional right kernel but a 1-dimensional"
    ):
        cpl.prepare_coupled_problem(
            _plan(owner, gauge=cpl.CoupledGauge()),
            arguments={"graph": _arguments(np.zeros(5))},
        )


def _solver_argument(name: str) -> cpl.ParameterBinding:
    port = phx.ValuePort(
        name, event_shape=(), component_ids=("value",), representation="scalar"
    )
    return cpl.ParameterBinding(
        name,
        port,
        targets=(cpl.RuntimeInput("owner", name),),
        role="control",
        derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
    )


@pytest.mark.parametrize("reference", [0.0, 0.7], ids=["stationary", "generic"])
def test_solver_argument_entering_the_operator_is_refused_at_every_reference(
    reference: float,
) -> None:
    """``A(p) = A0 + p^2 C`` enters the operator although ``dA/dp = 0`` at ``p = 0``."""
    owner = _WeightedOwner("owner", 5.0 * np.eye(5), coupling=np.ones((5, 5)))
    arguments = {"owner": {"load": jnp.ones(5)}}

    with pytest.raises(ValueError, match="enters the coupled operator"):
        cpl.prepare_coupled_problem(
            _plan(owner, parameters=(_solver_argument("p"),)),
            arguments=arguments,
            parameters={"p": jnp.asarray(reference)},
        )


def test_solver_argument_confined_to_the_load_is_admitted_under_rhs_only() -> None:
    owner = _WeightedOwner("owner", 5.0 * np.eye(5) + np.ones((5, 5)))
    prepared = cpl.prepare_coupled_problem(
        _plan(owner, parameters=(_solver_argument("load"),)),
        arguments={"owner": {"p": jnp.asarray(0.0)}},
        parameters={"load": jnp.asarray(0.0)},
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("rhs-only"),
    )
    matrix = 5.0 * np.eye(5) + np.ones((5, 5))

    def total(load: Array) -> Array:
        solution = cpl.solve_coupled_problem(
            prepared,
            arguments={"owner": {"p": jnp.asarray(0.0)}},
            parameters={"load": load},
            policy=policy,
        )
        return jnp.sum(solution.field("owner", "u"))

    assert prepared.derivative_capability(policy).admits("load")
    # d/dg sum(A^{-1} g 1) = 1^T A^{-1} 1 for the scalar load g broadcast to every row.
    expected = np.sum(np.linalg.solve(matrix, np.ones(5)))
    assert float(jax.grad(total)(jnp.asarray(0.3))) == pytest.approx(expected, rel=1e-12)


def test_problem_identity_distinguishes_gauge_resources_and_parameter_semantics() -> None:
    owner = _WeightedOwner("graph", _LAPLACIAN, kernel=_ONES)
    arguments = {"graph": _arguments(np.zeros(5))}

    def identity(plan: cpl.CoupledProblemPlan) -> str:
        return cpl.prepare_coupled_problem(plan, arguments=arguments).problem_id

    reference = identity(_plan(owner, gauge=cpl.CoupledGauge()))
    variants = {
        "compatibility": _plan(owner, gauge=cpl.CoupledGauge(compatibility="project")),
        "kernel-tolerance": _plan(
            owner,
            gauge=cpl.CoupledGauge(),
            resources=cpl.CoupledResourcePolicy(kernel_tolerance=1.0e-9),
        ),
    }
    assert identity(_plan(owner, gauge=cpl.CoupledGauge())) == reference
    for name, plan in variants.items():
        assert identity(plan) != reference, name

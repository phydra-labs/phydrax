from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _filter():
    centers = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
    return phx.applications.solid_mechanics.DensityFilterPlan(
        centers,
        jnp.ones((4,)),
        0.4,
    ).prepare()


class _DiagonalStateSolver(phx.optim.AbstractStateSolver):
    load: jax.Array
    modulus: Callable = eqx.field(static=True)

    def __init__(self, load, modulus):
        self.load = jnp.asarray(load)
        self.modulus = modulus

    @property
    def method_id(self) -> str:
        return "test-diagonal-state"

    def solve(self, problem, design, initial_state, /, *, args):
        del initial_state
        state = self.load / self.modulus(design)
        residual = problem.residual(state, design, args)
        return phx.optim.StateEquationResult(
            state,
            residual,
            phx.optim.OptimizationStatus.SUCCESS,
            phx.optim.OptimizationDiagnostics(residual_evaluations=1),
        )


def test_sparse_density_filter_preserves_constants_and_transpose_pairing():
    density_filter = _filter()
    values = jnp.asarray((0.1, 0.3, 0.7, 0.9))
    cotangent = jnp.asarray((0.4, -0.2, 0.5, 0.1))

    assert density_filter.constant_residual <= 1.0e-12
    assert jnp.allclose(density_filter.apply(jnp.ones((4,))), 1.0)
    assert jnp.vdot(density_filter.apply(values), cotangent) == pytest.approx(
        float(jnp.vdot(values, density_filter.operator.transpose_mv(cotangent)))
    )


def test_simp_interpolation_has_exact_endpoints_and_finite_gradient():
    interpolation = phx.applications.solid_mechanics.SIMPInterpolation(
        10.0,
        minimum_modulus=0.1,
        penalty=3.0,
    )
    values = interpolation(jnp.asarray((0.0, 1.0, 0.5)))
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(10.0)
    assert jax.grad(lambda density: interpolation(density))(jnp.asarray(0.5)) > 0.0


def test_fixed_mesh_compliance_topology_optimization_respects_volume():
    load = jnp.asarray((2.0, 1.5, 0.7, 0.4))
    interpolation = phx.applications.solid_mechanics.SIMPInterpolation(
        1.0,
        minimum_modulus=0.05,
        penalty=1.0,
    )
    density_filter = _filter()
    problem = phx.applications.solid_mechanics.ComplianceTopologyProblem(
        lambda state, modulus, _: modulus * state - load,
        load,
        density_filter,
        interpolation,
        0.5,
        _DiagonalStateSolver(
            load,
            lambda density: interpolation(density_filter.apply(density)),
        ),
        problem_id="four-cell-compliance",
    )
    initial_density = jnp.full((4,), 0.5)
    initial_state = load / interpolation(density_filter.apply(initial_density))
    result = phx.applications.solid_mechanics.solve_topology_optimization(
        problem,
        initial_state,
        initial_density,
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=5.0e-5,
            relative_optimality=0.0,
            absolute_step=1.0e-10,
            relative_step=0.0,
            maximum_steps=100,
        ),
    )

    assert result.state_design.successful
    assert result.volume_ratio <= 0.5 + 5.0e-5
    assert jnp.all((result.physical_density >= 1.0e-3) & (result.physical_density <= 1.0))
    assert result.physical_density[0] > result.physical_density[-1]
    assert result.state_design.objective < jnp.vdot(load, initial_state)


def test_topology_reanalysis_separates_discretization_control():
    state_design = phx.optim.StateDesignResult(
        jnp.asarray((1.0,)),
        jnp.asarray((0.5,)),
        10.0,
        None,
        None,
        phx.optim.OptimizationStatus.SUCCESS,
        phx.optim.OptimizationDiagnostics(),
        phx.optim.OptimizationProvenance(
            problem_id="test",
            method="test",
            backend="phydrax",
            globalization="none",
            matrix_free=True,
        ),
    )
    result = phx.applications.solid_mechanics.TopologyOptimizationResult(
        state_design,
        jnp.asarray((0.5,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.5,)),
        jnp.asarray(0.5),
        "test",
    )

    report = phx.applications.solid_mechanics.reanalyse_topology_design(
        result,
        lambda density: phx.applications.solid_mechanics.DensityTransferResult(
            jnp.repeat(density, 2),
            jnp.asarray(1.0e-8),
        ),
        lambda density: jnp.asarray(12.0),
        uniform_coarse_compliance=8.0,
        uniform_reference_compliance=9.6,
    )
    assert report.discretization_ratio == pytest.approx(1.2)
    assert report.excess_stiffness_overreport == pytest.approx(0.0)
    assert report.transfer_measure_error == pytest.approx(1.0e-8)

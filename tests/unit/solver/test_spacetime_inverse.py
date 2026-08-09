#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import phydrax as phx


class _ConstantRawLapse(eqx.Module):
    def __call__(self, coordinates):
        minimum = jnp.asarray(1e-6, dtype=coordinates.dtype)
        return jnp.log(jnp.expm1(1.0 - minimum))


class _ZeroShift(eqx.Module):
    def __call__(self, coordinates):
        return jnp.zeros((3,), dtype=coordinates.dtype)


class _IsotropicRawFactor(eqx.Module):
    expansion: jax.Array
    baseline: float = eqx.field(static=True)

    def __init__(self, expansion):
        self.expansion = jnp.asarray(expansion)
        self.baseline = 0.4

    def __call__(self, coordinates):
        diagonal = self.baseline + self.expansion * coordinates[0]
        return jnp.eye(3, dtype=coordinates.dtype) * diagonal


def _parameterization(expansion, chart):
    return phx.metrix.ADMParameterization(
        _ConstantRawLapse(),
        _ZeroShift(),
        _IsotropicRawFactor(expansion),
        chart=chart,
    )


def test_functional_solver_recovers_one_signature_safe_spacetime_from_shared_observables():
    domain = phx.domain.HyperRectangle([-1.0] * 4, [1.0] * 4, label="x")
    component = domain.component()
    chart = phx.metrix.CoordinateChart("inverse_adm", ("t", "x", "y", "z"))
    target_expansion = jnp.asarray(0.35)
    initial_parameterization = _parameterization(-0.1, chart)
    target_parameterization = _parameterization(target_expansion, chart)
    metric_field = phx.operators.as_lorentzian_metric_field(
        domain,
        initial_parameterization.metric(),
        var="x",
    )
    target_metric = target_parameterization.metric()
    target_metric_field = phx.operators.as_lorentzian_metric_field(
        domain,
        target_metric,
        var="x",
    )
    target_curvature = phx.operators.domain_scalar_curvature(
        domain,
        target_metric,
        var="x",
    )

    def scalar_curvature_observable(candidate_field):
        candidate_metric = phx.operators.lorentzian_metric_from_field(
            candidate_field,
            chart=chart,
            var="x",
        )
        return phx.operators.domain_scalar_curvature(
            candidate_field.domain,
            candidate_metric,
            var="x",
        )

    metric_observation = phx.conditions.Observation(
        "metric",
        component,
        target_metric_field,
        label="metric-data",
    )
    curvature_observation = phx.conditions.Observation(
        "metric",
        component,
        target_curvature,
        operator=scalar_curvature_observable,
        label="curvature-data",
    )
    points = jnp.array(
        [
            [0.0, -0.4, 0.1, 0.2],
            [0.3, 0.2, -0.3, 0.1],
            [0.6, 0.1, 0.2, -0.2],
            [0.9, -0.2, -0.1, 0.3],
        ]
    )
    batch = component.points(points)
    source = phx.integration.fixed(
        phx.integration.from_samples(
            phx.integration.mean_over(component),
            batch,
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"metric": metric_field},
        terms=(
            phx.terms.ObservationPenalty(metric_observation, source),
            phx.terms.ObservationPenalty(
                curvature_observation,
                source,
                scale=0.1,
            ),
        ),
    )
    trainable_leaves = jax.tree.leaves(solver.trainable_functions())
    initial_loss = solver.loss()

    assert len(trainable_leaves) == 1
    assert trainable_leaves[0].shape == ()

    trained = solver.solve(
        num_iter=80,
        optim=optax.adam(0.04),
        keep_best=True,
        jit=True,
        log_every=0,
    )
    trained_field = trained["metric"]
    trained_metric = phx.operators.lorentzian_metric_from_field(
        trained_field,
        chart=chart,
        var="x",
    )
    final_loss = trained.loss()
    final_matrices = trained_metric(points)
    expected_matrices = target_metric(points)
    final_curvature = phx.operators.domain_scalar_curvature(
        domain,
        trained_metric,
        var="x",
    ).func(points)
    expected_curvature = target_curvature.func(points)
    adm_report = phx.metrix.validate_adm_decomposition(
        phx.metrix.decompose_adm_metric(trained_metric, points),
        reference_metric=final_matrices,
    )

    assert final_loss < 1e-5 * initial_loss
    assert jnp.allclose(final_matrices, expected_matrices, rtol=2e-3, atol=2e-4)
    assert jnp.allclose(final_curvature, expected_curvature, rtol=2e-3, atol=2e-4)
    assert bool(adm_report.valid)
    assert adm_report.minimum_lapse > 0.0
    assert adm_report.minimum_spatial_eigenvalue > 0.0

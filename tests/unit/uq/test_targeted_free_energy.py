#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import Affine

import phydrax as phx


class _AffineBijector(phx.uq.AbstractBijector):
    log_scale: jax.Array
    shift: jax.Array

    def __init__(self, log_scale=0.0, shift=0.0):
        self.log_scale = jnp.asarray([log_scale], dtype=jnp.float64)
        self.shift = jnp.asarray([shift], dtype=jnp.float64)

    def forward_shape(self, raw_shape, /):
        if tuple(raw_shape) != (1,):
            raise ValueError("shape")
        return (1,)

    def inverse_shape(self, physical_shape, /):
        return self.forward_shape(physical_shape)

    def forward(self, value, /):
        return jnp.exp(self.log_scale) * jnp.asarray(value) + self.shift

    def inverse(self, value, /):
        return (jnp.asarray(value) - self.shift) / jnp.exp(self.log_scale)

    def forward_log_det_jacobian(self, value, /):
        del value
        return self.log_scale


def _potentials():
    source = phx.uq.CallableReducedPotential(
        lambda value: 0.5 * value[0] ** 2, (1,), "unit-normal"
    )
    target = phx.uq.CallableReducedPotential(
        lambda value: 0.5 * ((value[0] - 2.0) / 0.5) ** 2,
        (1,),
        "shifted-normal",
    )
    return source, target


def test_exact_affine_map_produces_constant_generalized_work():
    source, target = _potentials()
    mapping = phx.uq.TargetedMapPlan(
        _AffineBijector(jnp.log(0.5), 2.0),
        (1,),
        architecture_id="affine-one-dimensional",
    )
    problem = phx.uq.TargetedFreeEnergyProblem(source, target, mapping)
    samples = jnp.linspace(-2.0, 2.0, 17)[:, None]
    targets = 0.5 * samples + 2.0

    evaluation = phx.uq.evaluate_targeted_work(problem, samples, target_samples=targets)
    estimate = phx.uq.free_energy_perturbation(evaluation.forward_work)

    assert bool(evaluation.valid & estimate.converged)
    np.testing.assert_allclose(evaluation.forward_work, jnp.log(2.0), atol=1.0e-12)
    np.testing.assert_allclose(evaluation.reverse_work, -jnp.log(2.0), atol=1.0e-12)
    assert jnp.max(evaluation.forward_roundtrip_residual) < 1.0e-12


def test_flowjax_adapter_and_com_chart_roundtrip_exactly():
    adapter = phx.uq.FlowJAXBijectionAdapter(
        Affine(jnp.zeros((3,)), jnp.ones((3,))),
        architecture_id="identity-affine",
    )
    chart = phx.uq.CenterOfMassPreservingBijector(adapter, [1.0, 3.0])
    positions = jnp.asarray([[0.2, -0.1, 0.3], [1.2, 0.4, -0.2]])

    mapped = chart.forward(positions)
    restored = chart.inverse(mapped)

    np.testing.assert_allclose(restored, positions, atol=1.0e-12)
    original_com = jnp.sum(jnp.asarray([1.0, 3.0])[:, None] * positions, axis=0) / 4.0
    mapped_com = jnp.sum(jnp.asarray([1.0, 3.0])[:, None] * mapped, axis=0) / 4.0
    np.testing.assert_allclose(mapped_com, original_com, atol=1.0e-12)
    assert jnp.allclose(chart.forward_log_det_jacobian(positions), 0.0)


def test_targeted_map_training_improves_affine_overlap():
    source, target = _potentials()
    mapping = phx.uq.TargetedMapPlan(
        _AffineBijector(), (1,), architecture_id="trainable-affine"
    )
    problem = phx.uq.TargetedFreeEnergyProblem(source, target, mapping)
    source_samples = jax.random.normal(jax.random.key(0), (128, 1))
    target_samples = 0.5 * jax.random.normal(jax.random.key(1), (128, 1)) + 2.0
    initial = phx.uq.evaluate_targeted_work(
        problem, source_samples, target_samples=target_samples
    )

    result = phx.uq.fit_targeted_free_energy_map(
        problem,
        source_samples,
        jax.random.key(2),
        target_samples=target_samples,
        policy=phx.uq.TargetedMapTrainingPolicy(
            maximum_steps=60,
            learning_rate=5.0e-2,
            validation_interval=10,
        ),
    )
    fitted_problem = phx.uq.TargetedFreeEnergyProblem(source, target, result.mapping)
    fitted = phx.uq.evaluate_targeted_work(
        fitted_problem, source_samples, target_samples=target_samples
    )

    assert bool(result.valid & fitted.valid)
    assert jnp.var(fitted.forward_work) < jnp.var(initial.forward_work)
    assert result.forward_effective_samples > 0.8 * source_samples.shape[0]

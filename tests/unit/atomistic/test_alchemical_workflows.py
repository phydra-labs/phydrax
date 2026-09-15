import jax
import jax.numpy as jnp
import numpy as np

from phydrax.atomistic._elastic_network import ElasticNetworkPlan
from phydrax.atomistic._external_field import (
    ExternalFieldBoundaryPolicy,
    GriddedExternalFieldPlan,
)
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._units import AtomisticUnitSystem


def _elastic_system():
    return AtomisticSystemPlan(
        [10, 20, 30], [6, 6, 6], [12.0, 12.0, 12.0], AtomisticUnitSystem.reduced()
    ).prepare()


def test_elastic_network_rigid_invariance_and_force_energy_parity():
    system = _elastic_system()
    reference = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    network = ElasticNetworkPlan(1.5, 3.0, 3).prepare(system, reference)
    angle = 0.43
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rigid = reference @ rotation.T + jnp.asarray([2.0, -0.4, 1.1])
    invariant = network.evaluate(rigid)
    np.testing.assert_allclose(invariant.energy, 0.0, atol=2.0e-30)
    np.testing.assert_allclose(invariant.forces, 0.0, atol=5.0e-15)
    displaced = reference.at[1, 0].add(0.12)
    evaluation = network.evaluate(displaced)
    gradient = jax.grad(lambda value: network.evaluate(value).energy)(displaced)
    np.testing.assert_allclose(evaluation.forces, -gradient, rtol=1.0e-12, atol=1.0e-12)
    assert network.preparation.edge_count == 3


def _linear_field(policy):
    grid = jnp.indices((3, 4, 5), dtype=float)
    scalar = grid[0] + 2.0 * grid[1] - 0.5 * grid[2]
    return GriddedExternalFieldPlan(
        [1.0, -1.0, 2.0],
        [0.5, 2.0, 0.25],
        scalar,
        boundary_policy=policy,
        coordinate_frame="laboratory",
        coordinate_unit="length",
        value_unit="energy",
    ).prepare()


def test_scalar_field_interpolation_gradient_and_conservative_force():
    field = _linear_field(ExternalFieldBoundaryPolicy.FAIL)
    point = jnp.asarray([[1.35, 1.4, 2.6]])
    evaluation = field.evaluate(point)
    expected = 0.7 + 2.0 * 1.2 - 0.5 * 2.4
    np.testing.assert_allclose(evaluation.values, [expected], atol=1.0e-12)
    np.testing.assert_allclose(evaluation.jacobian, [[2.0, 1.0, -2.0]], atol=1.0e-12)
    force = field.energy_and_forces(point, coupling=[3.0])
    np.testing.assert_allclose(force.forces, [[-6.0, -3.0, 6.0]], atol=1.0e-12)
    gradient = jax.grad(lambda value: field.energy_and_forces(value).energy)(point)
    np.testing.assert_allclose(force.forces / 3.0, -gradient, atol=1.0e-12)


def test_vector_field_and_boundary_policies_report_domain_evidence():
    grid = jnp.indices((2, 2, 2), dtype=float)
    vector = jnp.stack((grid[0], 2.0 * grid[1], 3.0 * grid[2]), axis=-1)
    periodic = GriddedExternalFieldPlan(
        [0.0] * 3,
        [1.0] * 3,
        vector,
        boundary_policy=ExternalFieldBoundaryPolicy.PERIODIC,
        coordinate_frame="lab",
        coordinate_unit="x",
        value_unit="vector",
    ).prepare()
    wrapped = periodic.evaluate(jnp.asarray([[2.25, 0.5, 0.5]]))
    reference = periodic.evaluate(jnp.asarray([[0.25, 0.5, 0.5]]))
    np.testing.assert_allclose(wrapped.values, reference.values)
    assert bool(wrapped.evidence.out_of_domain[0])
    assert bool(wrapped.evidence.successful)
    clamped = _linear_field(ExternalFieldBoundaryPolicy.CLAMP).evaluate(
        jnp.asarray([[8.0, 1.0, 2.5]])
    )
    assert bool(clamped.evidence.out_of_domain[0])
    assert float(clamped.jacobian[0, 0]) == 0.0
    assert bool(clamped.evidence.successful)
    failed = _linear_field(ExternalFieldBoundaryPolicy.FAIL).evaluate(
        jnp.asarray([[8.0, 1.0, 2.5]])
    )
    assert bool(failed.evidence.out_of_domain[0])
    assert int(failed.evidence.out_of_domain_count) == 1
    assert not bool(failed.evidence.successful)
    assert bool(jnp.isnan(failed.values[0]))


def _padded_elastic_inputs():
    system = AtomisticSystemPlan(
        [10, 20, 30],
        [6, 6, 0],
        [12.0, 12.0, 1.0],
        AtomisticUnitSystem.reduced(),
        active_mask=[True, True, False],
    ).prepare()
    reference = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [jnp.nan, jnp.nan, jnp.nan]]
    )
    return system, reference


def test_elastic_reference_ignores_and_canonicalizes_inactive_padding():
    system, reference = _padded_elastic_inputs()
    plan = ElasticNetworkPlan(1.1, 2.0, 3)
    from_nan = plan.prepare(system, reference)
    from_infinity = plan.prepare(
        system,
        reference.at[2].set(jnp.asarray([jnp.inf, -jnp.inf, jnp.nan])),
    )
    assert from_nan.prepared_id == from_infinity.prepared_id
    assert from_nan.preparation.edge_count == 1


def test_elastic_padded_routes_do_not_scatter_nan_forces():
    system, reference = _padded_elastic_inputs()
    network = ElasticNetworkPlan(1.1, 2.0, 3).prepare(system, reference)
    evaluation = network.evaluate(reference)
    assert bool(evaluation.successful)
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))
    np.testing.assert_allclose(evaluation.forces[2], 0.0, atol=0.0)


def test_elastic_collapsed_valid_edge_fails_closed_with_finite_output():
    system, reference = _padded_elastic_inputs()
    network = ElasticNetworkPlan(1.1, 2.0, 1).prepare(system, reference)
    collapsed = reference.at[1].set(reference[0])
    evaluation = network.evaluate(collapsed)
    assert bool(evaluation.finite)
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))
    assert not bool(evaluation.successful)

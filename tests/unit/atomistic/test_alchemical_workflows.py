import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._alchemical import (
    AlchemicalEndpointPlan,
    AlchemicalTransformationPlan,
    LambdaSchedulePlan,
)
from phydrax.atomistic._elastic_network import ElasticNetworkPlan
from phydrax.atomistic._external_field import (
    ExternalFieldBoundaryPolicy,
    GriddedExternalFieldPlan,
)
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.uq import ReducedPotentialSamples


UNITS = AtomisticUnitSystem.reduced()


def _endpoint_pair(*, omit_final_bond=False):
    endpoint_a = AlchemicalEndpointPlan(
        [10, 20],
        [0, 1],
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([1.0, 1.1]),
        jnp.asarray([0.2, 0.3]),
        bond_particle_ids=[[10, 20]],
        bond_stiffness=jnp.asarray([2.0]),
        bond_equilibrium_lengths=jnp.asarray([1.0]),
        units=UNITS,
    )
    endpoint_b = AlchemicalEndpointPlan(
        [200, 100],
        [3, 2],
        jnp.asarray([-0.5, 0.5]),
        jnp.asarray([1.3, 1.2]),
        jnp.asarray([0.1, 0.15]),
        bond_particle_ids=None if omit_final_bond else [[100, 200]],
        bond_stiffness=None if omit_final_bond else jnp.asarray([4.0]),
        bond_equilibrium_lengths=None if omit_final_bond else jnp.asarray([1.5]),
        units=UNITS,
    )
    return endpoint_a, endpoint_b


def _transformation():
    endpoint_a, endpoint_b = _endpoint_pair()
    return AlchemicalTransformationPlan(
        endpoint_a,
        endpoint_b,
        atom_mapping=[[20, 200], [10, 100]],
        atom_capacity=2,
        bond_capacity=1,
        schedule=LambdaSchedulePlan.linear(5),
    ).prepare()


def _manual_endpoint_energy(distance, charge, sigma, epsilon, stiffness, equilibrium):
    sigma_pair = 0.5 * (sigma[0] + sigma[1])
    epsilon_pair = np.sqrt(epsilon[0] * epsilon[1])
    ratio6 = (sigma_pair / distance) ** 6
    return (
        0.5 * stiffness * (distance - equilibrium) ** 2
        + charge[0] * charge[1] / distance
        + 4.0 * epsilon_pair * (ratio6**2 - ratio6)
    )


def test_endpoint_exactness_and_stable_mapping():
    prepared = _transformation()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]])
    first, final = prepared.evaluate(positions, 0.0), prepared.evaluate(positions, 1.0)
    expected_a = _manual_endpoint_energy(
        1.25, [1.0, -1.0], [1.0, 1.1], [0.2, 0.3], 2.0, 1.0
    )
    expected_b = _manual_endpoint_energy(
        1.25, [0.5, -0.5], [1.2, 1.3], [0.15, 0.1], 4.0, 1.5
    )
    np.testing.assert_allclose(first.energy, expected_a, rtol=1.0e-12)
    np.testing.assert_allclose(final.energy, expected_b, rtol=1.0e-12)
    np.testing.assert_array_equal(prepared.endpoint_particle_ids[:, 0], [10, 100])
    np.testing.assert_array_equal(prepared.endpoint_particle_ids[:, 1], [20, 200])
    np.testing.assert_array_equal(prepared.atom_type_ids[1], [2, 3])


def test_alchemical_endpoint_reduced_potentials_feed_targeted_work():
    transformation = _transformation()
    source = phx.uq.AlchemicalEndpointReducedPotential(transformation, 0.0)
    target = phx.uq.AlchemicalEndpointReducedPotential(transformation, 1.0)
    mapping = phx.uq.TargetedMapPlan(
        phx.uq.IdentityBijector(), source.event_shape, architecture_id="identity"
    )
    problem = phx.uq.TargetedFreeEnergyProblem(source, target, mapping)
    positions = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]],
        ]
    )

    evaluation = phx.uq.evaluate_targeted_work(problem, positions)

    assert bool(evaluation.valid)
    assert jnp.all(jnp.isfinite(evaluation.forward_work))
    expected = jax.vmap(
        lambda value: (
            transformation.evaluate(value, 1.0).energy
            - transformation.evaluate(value, 0.0).energy
        )
    )(positions)
    np.testing.assert_allclose(evaluation.forward_work, expected)


def test_mapping_capacity_and_mapped_core_topology_rejection():
    endpoint_a, endpoint_b = _endpoint_pair(omit_final_bond=True)
    plan = AlchemicalTransformationPlan(
        endpoint_a,
        endpoint_b,
        atom_mapping=[[10, 100], [20, 200]],
        atom_capacity=2,
        bond_capacity=1,
    )
    with pytest.raises(ValueError, match="compatible bonded topology"):
        plan.prepare()
    with pytest.raises(ValueError, match="atom capacity"):
        AlchemicalTransformationPlan(
            endpoint_a,
            endpoint_b,
            atom_mapping=[[10, 100]],
            atom_capacity=2,
            bond_capacity=1,
        )


def test_dummy_soft_core_is_finite_at_coincident_coordinates():
    endpoint_a = AlchemicalEndpointPlan(
        [1, 2],
        [0, 0],
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0.2, 0.2]),
        units=UNITS,
    )
    endpoint_b = AlchemicalEndpointPlan(
        [1, 2],
        [0, 0],
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0.2, 0.0]),
        dummy_mask=[False, True],
        units=UNITS,
    )
    prepared = AlchemicalTransformationPlan(
        endpoint_a, endpoint_b, atom_capacity=2, bond_capacity=0
    ).prepare()
    evaluation = eqx.filter_jit(prepared.evaluate)(jnp.zeros((2, 3)), 0.5)
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))


def test_component_dudlambda_matches_finite_difference():
    prepared = _transformation()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.31, 0.1, 0.0]])
    center = prepared.evaluate(positions, 0.37)
    step = 1.0e-5
    plus, minus = (
        prepared.evaluate(positions, 0.37 + step),
        prepared.evaluate(positions, 0.37 - step),
    )
    finite_difference = (plus.component_energies - minus.component_energies) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        center.component_dudlambda, finite_difference, rtol=2.0e-5, atol=2.0e-6
    )
    np.testing.assert_allclose(center.dudlambda, jnp.sum(center.component_dudlambda))


def test_cross_evaluation_matrix_and_cycle_closure():
    prepared = _transformation()
    samples = jnp.asarray(
        [[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0]]]
    )
    cross = eqx.filter_jit(prepared.cross_evaluate)(samples, None)
    assert cross.values.shape == (5, 2)
    assert cross.component_energies.shape == (5, 2, 3)
    assert bool(cross.successful)
    reduced = ReducedPotentialSamples(cross.values, [1, 1, 0, 0, 0], [0, 1])
    np.testing.assert_allclose(reduced.values, cross.values)
    cycle = prepared.cycle_work(samples[0], [0.0, 0.4, 1.0, 0.2, 0.0])
    np.testing.assert_allclose(cycle.work, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(cycle.component_work, 0.0, atol=2.0e-14)
    assert bool(cycle.successful)


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


def test_soft_core_tracks_lj_and_charge_activity_not_only_dummy_status():
    interacting = AlchemicalEndpointPlan(
        [1, 2],
        [0, 0],
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0.2, 0.2]),
        units=UNITS,
    )
    noninteracting = AlchemicalEndpointPlan(
        [1, 2],
        [0, 0],
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0.0, 0.0]),
        units=UNITS,
    )
    prepared = AlchemicalTransformationPlan(
        interacting,
        noninteracting,
        atom_capacity=2,
        bond_capacity=0,
    ).prepare()
    endpoint = prepared.evaluate(jnp.zeros((2, 3)), 1.0)
    np.testing.assert_allclose(endpoint.energy, 0.0, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(endpoint.forces)))
    assert bool(jnp.all(jnp.isfinite(endpoint.component_dudlambda)))
    assert bool(endpoint.successful)


def test_zero_weight_disappearing_bond_has_defined_gradient():
    bonded = AlchemicalEndpointPlan(
        [1, 2],
        [0, 0],
        jnp.zeros(2),
        jnp.ones(2),
        jnp.zeros(2),
        bond_particle_ids=[[1, 2]],
        bond_stiffness=jnp.asarray([3.0]),
        bond_equilibrium_lengths=jnp.asarray([1.0]),
        units=UNITS,
    )
    dummy = AlchemicalEndpointPlan(
        [1, 2],
        [0, 0],
        jnp.zeros(2),
        jnp.ones(2),
        jnp.zeros(2),
        dummy_mask=[False, True],
        units=UNITS,
    )
    prepared = AlchemicalTransformationPlan(
        bonded, dummy, atom_capacity=2, bond_capacity=1
    ).prepare()
    positions = jnp.zeros((2, 3))
    absent = prepared.evaluate(positions, 1.0)
    np.testing.assert_allclose(absent.energy, 0.0, atol=0.0)
    np.testing.assert_allclose(absent.forces, 0.0, atol=0.0)
    assert bool(absent.successful)
    collapsed = prepared.evaluate(positions, 0.0)
    assert bool(collapsed.finite)
    assert bool(jnp.all(jnp.isfinite(collapsed.forces)))
    assert not bool(collapsed.successful)


def test_interpolation_rejects_nonscalar_lambda():
    with pytest.raises(ValueError, match="must be scalar"):
        _transformation().interpolate(jnp.asarray([0.25, 0.75]))


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

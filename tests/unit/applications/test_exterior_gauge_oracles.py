# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.lattice_field._distributed_qcd import (
    DistributedGaugeTheoryPlan,
    DistributedHMCPlan,
    prepare_distributed_hmc,
)
from phydrax.applications.lattice_field._hamiltonian_gauge import (
    periodic_schwinger_gauss_network,
    periodic_schwinger_hamiltonian,
    PeriodicSchwingerModel,
)
from phydrax.applications.lattice_field._qcd_observables import (
    HypercubicGaugeObservablePlan,
    measure_hypercubic_gauge_observables,
    measure_wilson_gauge_action,
)
from phydrax.applications.lattice_field._qcd_recipes import build_su3_gauge_geometry
from phydrax.applications.lattice_field._z2_gauge import (
    prepare_z2_gauss_sector,
    Z2GaugeModel,
)
from phydrax.discretization._cell_complex import cubical_cell_complex
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan
from phydrax.discretization._lattice_distribution import LatticeDecompositionPlan
from phydrax.discretization._oriented_path import (
    ordered_path_transport,
    prepare_cell_boundary_paths,
)
from phydrax.discretization._topology import TensorTopology
from phydrax.graph._charged_scalar_gauge import ChargedScalarGaugePlan
from phydrax.graph._matrix_gauge import MatrixGaugeLinkSpace
from phydrax.linalg._materialization import MaterializationPolicy
from phydrax.metrix._complex_matrix_manifold import SpecialUnitaryGroup
from phydrax.operators.path_integral._wilson_gauge import WilsonGaugeAction
from phydrax.operators.quantum._gauge_constraints import (
    GaussSectorResourcePolicy,
    prepare_sparse_physical_sector,
)
from phydrax.operators.quantum._subspaces import BasisStateSubspace
from phydrax.sampling._compact_group_hamiltonian import (
    _kinetic,
    _trajectory,
    _value_and_gradient,
)
from phydrax.solver._local_hamiltonian import materialize_local_hamiltonian


def test_distributed_wilson_force_is_negative_action_derivative() -> None:
    shape = (3, 2)
    decomposition = LatticeDecompositionPlan(shape, (3, 1))
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("x", "y"), shape, periodic=(True, True)),
        jnp.ones((2,), dtype=jnp.complex128),
    )
    geometry = build_su3_gauge_geometry(boundary, decomposition=decomposition)
    group = geometry.link_space.group
    coordinates = (
        jnp.sin(
            jnp.arange(geometry.link_space.num_edges * 8, dtype=jnp.float64).reshape(
                (-1, 8)
            )
        )
        * 0.17
    )
    links = jax.vmap(lambda x: group.exp(group.hat(x)))(coordinates)
    beta = 2.7
    action = WilsonGaugeAction(
        geometry.link_space, geometry.plaquettes, plaquette_couplings=beta
    )
    gauge = DistributedGaugeTheoryPlan(
        decomposition, beta, forward_edges=geometry.transport.forward_edges
    ).prepare()
    force = gauge.gauge_force_from_graph(geometry.transport, geometry.staples, links)
    assert force.successful
    # Independent orthonormal su(3) directions for the Frobenius Lie metric.
    generators: list[np.ndarray] = []
    for first in range(3):
        for second in range(first + 1, 3):
            real = np.zeros((3, 3), dtype=np.complex128)
            real[first, second], real[second, first] = 1, -1
            imaginary = np.zeros((3, 3), dtype=np.complex128)
            imaginary[first, second] = imaginary[second, first] = 1j
            generators.extend((real / np.sqrt(2), imaginary / np.sqrt(2)))
    generators.extend(
        (
            1j * np.diag((1, -1, 0)) / np.sqrt(2),
            1j * np.diag((1, 1, -2)) / np.sqrt(6),
        )
    )
    basis = jnp.asarray(np.stack(generators))

    def varied_action(amounts: jax.Array) -> jax.Array:
        algebra = jnp.sum(amounts[..., None, None] * basis[None, ...], axis=1)
        changed = jax.vmap(group.exp)(algebra) @ links
        return action.canonical_action(changed)

    derivative = jax.grad(varied_action)(
        jnp.zeros((geometry.link_space.num_edges, 8), dtype=jnp.float64)
    )
    components = -jnp.real(
        jnp.trace(basis[None, ...] @ force.value[:, None, ...], axis1=-2, axis2=-1)
    )
    np.testing.assert_allclose(components, -derivative, atol=2e-12, rtol=2e-12)
    generic_force = gauge.gauge_force(links, geometry.staples.staples(links))
    native_force = (
        jnp.zeros_like(links)
        .at[geometry.transport.forward_edges]
        .set(generic_force.value)
    )
    generic_components = -jnp.real(
        jnp.trace(basis[None, ...] @ native_force[:, None, ...], axis1=-2, axis2=-1)
    )
    np.testing.assert_allclose(generic_components, -derivative, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        jnp.sum(force.local_values, axis=0), force.value, atol=1e-13
    )
    np.testing.assert_allclose(
        gauge.gauge_action(links).value, action.canonical_action(links), atol=1e-12
    )


def test_schwinger_gauss_sectors_are_oriented_electric_divergence() -> None:
    model = PeriodicSchwingerModel(
        2, maximum_flux=1, lattice_spacing=0.8, mass=0.3, gauge_coupling=0.9
    )
    network = periodic_schwinger_gauss_network(model)
    generators = np.asarray(network.dense_generators(maximum_elements=3000))[:, 0]
    levels = np.asarray(model.link_space.electric_levels)
    occupations = np.asarray(tuple(np.ndindex((3, 3, 2, 2))), dtype=np.int32)
    electric = levels[occupations[:, :2]]
    # Boundary=head-tail; physical electric divergence=-boundary.
    expected = np.stack(
        (
            electric[:, 0] - electric[:, 1] - occupations[:, 2] + 1,
            electric[:, 1] - electric[:, 0] - occupations[:, 3],
        ),
        axis=1,
    )
    observed = np.stack(tuple(np.real(np.diag(g)) for g in generators), axis=1)
    np.testing.assert_array_equal(observed, expected)
    sector = prepare_sparse_physical_sector(
        network,
        resources=GaussSectorResourcePolicy(
            maximum_hilbert_dimension=100,
            maximum_dense_elements=10000,
            maximum_basis_states=100,
        ),
    )
    indices = np.flatnonzero(np.all(expected == 0, axis=1))
    assert isinstance(sector.subspace, BasisStateSubspace)
    np.testing.assert_array_equal(sector.subspace.basis_indices, indices)
    hamiltonian = np.asarray(
        materialize_local_hamiltonian(
            periodic_schwinger_hamiltonian(model),
            policy=MaterializationPolicy(max_entries=2000, max_bytes=1 << 20),
        )
    )
    physical = hamiltonian[np.ix_(indices, indices)]
    assert np.max(np.abs(physical - np.diag(np.diag(physical)))) > 0.1
    for generator in generators:
        np.testing.assert_allclose(
            generator @ hamiltonian - hamiltonian @ generator, 0.0, atol=1e-13
        )


def test_wilson_density_has_coupled_abelian_flux_normalization() -> None:
    shape = (3, 4)
    theta = 2 * np.pi / np.prod(shape)
    beta = 2.3
    links = np.broadcast_to(np.eye(2, dtype=np.complex128), shape + (2, 2, 2)).copy()
    for x, y in np.ndindex(shape):
        phase = np.exp(1j * theta * x)
        links[x, y, 1] = np.diag((phase, phase.conjugate()))
        if x == shape[0] - 1:
            seam = np.exp(-1j * theta * shape[0] * y)
            links[x, y, 0] = np.diag((seam, seam.conjugate()))
    plan = HypercubicGaugeObservablePlan(shape, color_components=2)
    measured = measure_hypercubic_gauge_observables(
        plan, links, measure_topology=False, beta=beta
    )
    cells = cubical_cell_complex(shape, periodic=True)
    space = MatrixGaugeLinkSpace(cells.topology, SpecialUnitaryGroup(2))
    action = WilsonGaugeAction(
        space, prepare_cell_boundary_paths(cells.topology), plaquette_couplings=beta
    )
    native_links = (
        jnp.asarray(links)
        .reshape((-1, 2, 2, 2))
        .transpose(1, 0, 2, 3)
        .reshape((-1, 2, 2))
    )
    generic = measure_wilson_gauge_action(action, native_links)
    expected = beta * (1 - np.cos(theta))
    np.testing.assert_allclose(measured.mean_plaquette, np.cos(theta), atol=1e-13)
    np.testing.assert_allclose(measured.wilson_action_density, expected, atol=1e-13)
    np.testing.assert_allclose(generic.wilson_action_density, expected, atol=1e-13)
    np.testing.assert_allclose(
        action.canonical_action(native_links), np.prod(shape) * expected, atol=1e-12
    )


def test_charged_scalar_uses_supplied_edge_orientation_and_sparse_boundary() -> None:
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    faces = np.asarray(((2, 1, 0),), dtype=np.int32)
    edges = np.asarray(((2, 0), (1, 0), (2, 1)), dtype=np.int32)
    plan = ChargedScalarGaugePlan(vertices, faces, edges, coupling=0.7)
    potential = jnp.asarray((0.2, -0.4, 0.3))
    state = plan.state(jnp.asarray((1 + 0.1j, 0.2 + 0.7j, -0.4 + 0.2j)), potential)
    # Face traversal 2->1->0->2, in the user-provided edge orientations.
    np.testing.assert_allclose(
        plan.curvature(state), (potential[2] + potential[1] - potential[0],)
    )
    parameter = jnp.asarray((0.6, -0.2, 0.3))
    assert plan.validate_transform(state, parameter).successful


def test_one_site_torus_keeps_group_order_but_binary_boundary_cancels() -> None:
    cells = cubical_cell_complex((1, 1), periodic=True)
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("x", "y"), (1, 1), periodic=(True, True)),
        jnp.ones((2,), dtype=jnp.complex128),
    )
    geometry = build_su3_gauge_geometry(boundary)
    group = geometry.link_space.group
    first = group.exp(group.hat(jnp.asarray((0.2, 0.1, 0.3, 0, 0, 0, 0, 0))))
    second = group.exp(group.hat(jnp.asarray((-0.1, 0.2, 0, 0.4, 0, 0, 0, 0))))
    links = jnp.stack((first, second))
    expected = first @ second @ jnp.conj(first.T) @ jnp.conj(second.T)
    observed = ordered_path_transport(geometry.plaquettes, links)[0]
    np.testing.assert_allclose(observed, expected, atol=1e-13)
    assert np.max(np.abs(np.asarray(expected) - np.eye(3))) > 0.01
    model = Z2GaugeModel(cells.topology, electric_coupling=1, magnetic_coupling=1)
    sector = prepare_z2_gauss_sector(model, maximum_basis_states=4)
    assert sector.constraint_rank == 0
    assert sector.expected_dimension == 4
    np.testing.assert_array_equal(sector.subspace.basis_indices, np.arange(4))


def test_native_qcd_hmc_trajectory_is_reversible_with_second_order_energy() -> None:
    shape = (2, 2)
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("x", "y"), shape, periodic=(True, True)),
        jnp.ones((2,), dtype=jnp.complex128),
    )
    decomposition = LatticeDecompositionPlan(shape, (2, 1))
    geometry = build_su3_gauge_geometry(boundary, decomposition=decomposition)
    group = geometry.link_space.group
    coordinates = 0.1 * jnp.sin(jnp.arange(64, dtype=jnp.float64).reshape((8, 8)))
    links = jax.vmap(lambda x: group.exp(group.hat(x)))(coordinates)
    gauge = DistributedGaugeTheoryPlan(
        decomposition, 2.3, forward_edges=geometry.transport.forward_edges
    ).prepare()
    momentum = 0.2 * jnp.cos(jnp.arange(64, dtype=jnp.float64).reshape((8, 8)))
    errors: list[float] = []
    for step_size, steps in ((0.04, 3), (0.02, 6)):
        kernel = prepare_distributed_hmc(
            DistributedHMCPlan(step_size=step_size, leapfrog_steps=steps), gauge, links
        ).kernel
        log_value, gradient = _value_and_gradient(kernel, links)
        initial_energy = -log_value + _kinetic(kernel, momentum)
        position, reverse_momentum, next_gradient, value, used, nonfinite, nonmember = (
            _trajectory(kernel, links, gradient, momentum)
        )
        assert used == steps
        assert not nonfinite
        assert not nonmember
        energy = -value + _kinetic(kernel, reverse_momentum)
        errors.append(float(jnp.abs(energy - initial_energy)))
        restored, restored_momentum, _, _, _, failed, outside = _trajectory(
            kernel, position, next_gradient, reverse_momentum
        )
        assert not failed
        assert not outside
        np.testing.assert_allclose(restored, links, atol=2e-12)
        np.testing.assert_allclose(restored_momentum, momentum, atol=2e-12)
    assert errors[1] < 0.4 * errors[0]

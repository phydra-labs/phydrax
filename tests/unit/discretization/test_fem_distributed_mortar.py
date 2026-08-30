#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization.fem._distributed import (
    DistributedFiniteElementMortarPlan,
    finite_element_partition_workset_plan,
    FiniteElementFacetOwnershipPlan,
    FiniteElementHaloPlan,
    FiniteElementPartition,
    PartitionedFiniteElementDofMap,
)
from phydrax.discretization.fem._generic import FiniteElementDofMap
from phydrax.discretization.fem._mortar import serial_finite_element_mortar_plan
from phydrax.discretization.fem._reference import lagrange_element


def _unequal_reversed_mortar():
    left_nodes = np.linspace(-1.0, 1.0, 4)
    right_nodes = np.linspace(-1.0, 1.0, 2)
    mortar_nodes = left_nodes
    quadrature, weights = np.polynomial.legendre.leggauss(6)
    physical = np.stack((quadrature, 0.2 * quadrature**2, 0.1 * quadrature**3), axis=1)
    measure = np.sqrt(1.0 + (0.4 * quadrature) ** 2 + (0.3 * quadrature**2) ** 2)
    return serial_finite_element_mortar_plan(
        left_nodes,
        right_nodes,
        mortar_nodes,
        quadrature,
        weights,
        right_orientation=np.asarray([1, 0], dtype=np.int32),
        declared_reproduction_degree=1,
        left_physical_coordinates=physical,
        right_physical_coordinates=physical,
        coordinate_measure=measure,
        interface_id="unequal-reversed-curved",
    )


def _child_mortar(child_index: int):
    nodes = np.linspace(-1.0, 1.0, 3)
    quadrature, weights = np.polynomial.legendre.leggauss(5)
    mapped_nodes = 0.5 * (nodes + (-1.0 if child_index == 0 else 1.0))
    mapped_quadrature = 0.5 * (quadrature + (-1.0 if child_index == 0 else 1.0))
    return serial_finite_element_mortar_plan(
        nodes,
        nodes,
        nodes,
        quadrature,
        weights,
        left_evaluation_points=mapped_quadrature,
        left_polynomial_coordinates=nodes,
        right_polynomial_coordinates=mapped_nodes,
        mortar_polynomial_coordinates=mapped_nodes,
        polynomial_evaluation_points=mapped_quadrature,
        right_orientation=(
            np.asarray([2, 1, 0], dtype=np.int32) if child_index == 1 else None
        ),
        declared_reproduction_degree=2,
        coordinate_measure=np.full(quadrature.shape, 0.5),
        interface_id="coarse-to-two-children",
        child_index=child_index,
        child_count=2,
    )


def test_owned_halo_worksets_dependencies_and_deterministic_halo_actions():
    partition = FiniteElementPartition(np.asarray([0, 0, 1, 1]), 2)
    facets = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    worksets = finite_element_partition_workset_plan(
        partition,
        facets,
        cell_global_ids=np.asarray([40, 10, 30, 20]),
    )

    assert np.array_equal(np.asarray(worksets.owned_cells[0, :2]), [1, 0])
    assert np.array_equal(np.asarray(worksets.owned_cells[1, :2]), [3, 2])
    assert np.array_equal(
        np.asarray(worksets.dependencies), [[False, True], [True, False]]
    )
    assert np.array_equal(
        np.asarray(worksets.completions), np.asarray(worksets.dependencies).T
    )
    assert len(worksets.completion_ids) == 2
    assert worksets.dependency_ids[0] == (worksets.completion_ids[1],)

    halo = FiniteElementHaloPlan(
        np.asarray([[0, 2, -1], [1, 3, 4]], dtype=np.int32),
        valid=np.asarray([[True, True, False], [True, True, True]]),
        owner_columns=np.asarray([0, 1], dtype=np.int32),
    )
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    assert jnp.allclose(
        halo.sum_contributions(values), jnp.asarray([4.0, 11.0, 4.0, 11.0, 11.0])
    )
    assert jnp.allclose(
        halo.update_replicas(values), jnp.asarray([1.0, 4.0, 1.0, 4.0, 4.0])
    )
    assert jnp.allclose(
        halo.update_pullback(jnp.ones_like(values)),
        jnp.asarray([2.0, 0.0, 0.0, 3.0, 0.0]),
    )
    assert halo.reduction_semantics == "replica-columns-left-to-right"


def test_global_inner_and_dual_pullback_are_partition_independent():
    mesh = CellMesh.from_triangles(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
    )
    dof_map = FiniteElementDofMap(mesh, (lagrange_element("triangle", 1),))
    identifiers = jnp.asarray([10, 20, 30, 40], dtype=jnp.int32)
    serial = PartitionedFiniteElementDofMap(
        dof_map, identifiers, jnp.ones((4,), dtype=bool)
    )
    first = PartitionedFiniteElementDofMap(
        dof_map, identifiers, jnp.asarray([True, True, False, False])
    )
    second = PartitionedFiniteElementDofMap(
        dof_map, identifiers, jnp.asarray([False, False, True, True])
    )
    left = jnp.asarray([1.0, -2.0, 3.0, 0.5])
    right = jnp.asarray([0.25, 4.0, -1.0, 2.0])
    local_dual = jnp.asarray([2.0, 3.0, 5.0, 7.0])

    assert jnp.allclose(
        first.global_inner(left, right) + second.global_inner(left, right),
        serial.global_inner(left, right),
    )
    assert jnp.allclose(
        first.pullback_global(local_dual) + second.pullback_global(local_dual),
        serial.pullback_global(local_dual),
    )


def test_one_rank_and_partitioned_facet_reductions_are_identical_and_once_owned():
    facets = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    cell_ids = np.asarray([40, 10, 30, 20], dtype=np.int64)
    facet_ids = np.asarray([300, 100, 200], dtype=np.int64)
    one_rank = FiniteElementFacetOwnershipPlan(
        FiniteElementPartition(np.zeros((4,), dtype=np.int32), 1),
        facets,
        cell_global_ids=cell_ids,
        facet_global_ids=facet_ids,
    )
    partitioned = FiniteElementFacetOwnershipPlan(
        FiniteElementPartition(np.asarray([0, 0, 1, 1]), 2),
        facets,
        cell_global_ids=cell_ids,
        facet_global_ids=facet_ids,
    )
    flux = jnp.asarray([1.5, -2.0, 3.0])

    partition_sum = partitioned.route_partition(0, flux) + partitioned.route_partition(
        1, flux
    )
    assert jnp.allclose(partition_sum, one_rank.route_equal_opposite(flux))
    assert jnp.allclose(partition_sum, partitioned.route_equal_opposite(flux))
    assert np.array_equal(
        np.sum(np.asarray(partitioned.evaluation_mask), axis=0), np.ones((3,))
    )
    assert jnp.allclose(jnp.sum(partition_sum), 0.0)


def test_unequal_p_reversed_curved_mortar_reproduces_and_preserves_adjoint_roles():
    mortar = _unequal_reversed_mortar()
    quadrature = mortar.quadrature_points[:, 0]
    left_nodes = jnp.linspace(-1.0, 1.0, 4)
    right_local = (1.0 + 2.0 * jnp.linspace(-1.0, 1.0, 2))[::-1]

    assert mortar.evidence.constant_reproduced
    assert mortar.evidence.declared_polynomials_reproduced
    assert mortar.evidence.coordinates_compatible
    assert jnp.allclose(
        mortar.interpolate_left(1.0 + 2.0 * left_nodes), 1.0 + 2.0 * quadrature
    )
    assert jnp.allclose(mortar.interpolate_right(right_local), 1.0 + 2.0 * quadrature)

    trace = jnp.asarray([0.3, -0.2, 0.8, 1.1])
    quadrature_dual = jnp.linspace(-0.5, 0.7, mortar.quadrature_points.shape[0])
    assert jnp.allclose(
        jnp.vdot(mortar.interpolate_left(trace), quadrature_dual),
        jnp.vdot(trace, mortar.pullback_left_raw(quadrature_dual)),
    )

    mortar_value = jnp.asarray([0.2, 0.4, -0.1, 0.7])
    left_adjoint = mortar.pairing_adjoint_to_left(mortar_value)
    left_pairing = jnp.vdot(trace, mortar.left_mass @ left_adjoint)
    physical_pairing = jnp.vdot(
        mortar.interpolate_left(trace),
        mortar.physical_weights * (mortar.mortar_interpolation @ mortar_value),
    )
    assert jnp.allclose(left_pairing, physical_pairing)
    assert jnp.allclose(mortar.mass_project_left(jnp.ones((4,))), jnp.ones((4,)))

    flux = 1.0 + quadrature**2
    left_flux, right_flux = mortar.conservative_flux_contributions(flux)
    assert jnp.allclose(jnp.sum(left_flux), mortar.integrated_flux(flux))
    assert jnp.allclose(jnp.sum(right_flux), -mortar.integrated_flux(flux))
    assert jnp.allclose(mortar.conservation_residual(flux), 0.0)


def test_two_to_one_children_and_distributed_mortar_equal_serial_once_evaluation():
    children = (_child_mortar(0), _child_mortar(1))
    assert all(child.evidence.declared_polynomials_reproduced for child in children)
    assert jnp.allclose(
        sum((child.integrated_flux(jnp.ones((5,))) for child in children)), 2.0
    )

    ownership = FiniteElementFacetOwnershipPlan(
        FiniteElementPartition(np.asarray([0, 1, 1]), 2),
        np.asarray([[0, 1], [0, 2]], dtype=np.int32),
        cell_global_ids=np.asarray([10, 20, 30]),
        facet_global_ids=np.asarray([100, 101]),
    )
    distributed = DistributedFiniteElementMortarPlan(
        ownership, children, np.asarray([0, 1], dtype=np.int32)
    )
    fluxes = (jnp.linspace(0.5, 1.5, 5), jnp.linspace(1.5, 0.5, 5))
    serial = distributed.conservative_flux_contributions(fluxes)
    part_zero = distributed.conservative_flux_contributions(fluxes, part=0)
    part_one = distributed.conservative_flux_contributions(fluxes, part=1)

    for serial_pair, zero_pair, one_pair in zip(serial, part_zero, part_one, strict=True):
        assert jnp.allclose(zero_pair[0] + one_pair[0], serial_pair[0])
        assert jnp.allclose(zero_pair[1] + one_pair[1], serial_pair[1])
        assert jnp.allclose(jnp.sum(serial_pair[0]) + jnp.sum(serial_pair[1]), 0.0)
    assert np.array_equal(
        np.sum(np.asarray(ownership.evaluation_mask), axis=0), np.ones((2,))
    )

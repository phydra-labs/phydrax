#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellMeshStorage,
    CellPartition,
    coordinate_lagrange_element,
    DistributedHaloPlan,
    PeriodicCell,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
    PolygonalConnectivity,
    prepare_owner_local_halo_plans,
    resolve_owner_local_field_values,
)
from phydrax.discretization.fem import (
    discontinuous_element,
    execute_owner_local_finite_element_mass,
    execute_owner_local_finite_element_operator,
    execute_owner_local_finite_element_transfer,
    finite_element_dof_identity_keys,
    FiniteElementDofOwnershipPlan,
    FiniteElementFieldSpec,
    FiniteElementGlobalDofOwnership,
    FiniteElementPlan,
    form_element,
    lagrange_element,
    lower_distributed_finite_element_phases,
    map_owner_local_finite_element,
    owner_local_finite_element_transfer_support,
    OwnerLocalFiniteElementDiscretization,
    partition_cells_cost_aware,
    prepare_nested_field_transfer,
    prepare_owner_local_finite_element,
    prepare_owner_local_finite_element_transfer,
)
from phydrax.linalg import DistributedKrylovPolicy
from tests._support.cell_meshes import storage
from tests._support.forced_devices import run_on_forced_devices


requires_two_devices = pytest.mark.skipif(
    len(jax.local_devices()) < 2, reason="Requires two real JAX devices."
)


# Empirical `test_owner_local_p1_collective_mass_stiffness_and_solve`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_P1_CONSTANT_STIFFNESS_ATOL = 1.0e-12
# Empirical `test_owner_local_p1_collective_mass_stiffness_and_solve`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_P1_COLLECTIVE_PAIRING_ATOL = 1.0e-12
# Empirical `test_owner_local_p1_mass_uses_original_curved_coordinate_bank_by_default`: matched max |delta|=1.11022e-16; joint allowance fraction=2.13162e-09, margin=100%.
_CURVED_P1_MASS_ATOL = 1.0e-13
# Empirical `test_owner_local_p1_mass_uses_original_curved_coordinate_bank_by_default`: matched max |delta|=1.11022e-16; joint allowance fraction=0.00111022, margin=99.889%.
_CURVED_P1_CONSTANT_STIFFNESS_ATOL = 1.0e-13
# Empirical `test_owner_local_p1_mass_uses_original_curved_coordinate_bank_by_default`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CURVED_P1_SOLVE_VALUES_ATOL = 1.0e-12
# Empirical `test_owner_local_p1_collective_mass_stiffness_and_solve`: matched max |delta|=2.22045e-16; joint allowance fraction=1.90317e-09, margin=100%.
_P1_OWNED_MASS_IMAGE_ATOL = 1.0e-12
# Empirical `test_owner_local_p1_collective_mass_stiffness_and_solve`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_P1_OWNED_STIFFNESS_IMAGE_ATOL = 1.0e-12
# Empirical `test_owner_local_p1_collective_mass_stiffness_and_solve`: matched max |delta|=5.9508e-14; joint allowance fraction=2.9385e-07, margin=100%.
_P1_OWNED_SOLVE_VALUES_ATOL = 1.0e-9


def test_scientific_dof_keys_follow_ids_not_coordinates_or_local_rows() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    identifiers = np.asarray((41, 73, 109), dtype=np.int64)

    def keys(order: np.ndarray, shift: float) -> dict[int, str]:
        inverse = np.argsort(order)
        mesh = CellMesh(
            points[order] + shift,
            (
                CellBlock(
                    "triangle",
                    "triangle",
                    inverse[None, :],
                    global_ids=np.asarray((211,)),
                ),
            ),
            vertex_global_ids=identifiers[order],
        )
        plan = FiniteElementPlan(
            mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
        )
        prepared = plan.prepare()
        return dict(
            zip(
                identifiers[order].tolist(),
                finite_element_dof_identity_keys(plan, prepared.dof_maps[0]),
                strict=True,
            )
        )

    assert keys(np.asarray((0, 1, 2)), 0.0) == keys(np.asarray((2, 0, 1)), 7.0)


@pytest.mark.parametrize("family", ("H1", "Hdiv", "Hcurl"))
@pytest.mark.parametrize("identification", ("translation", "rotation"))
def test_global_periodic_dof_owner_is_derived_from_incident_cells(
    family: str,
    identification: str,
    tmp_path: Path,
) -> None:
    group: PeriodicCell | PeriodicIsometryGroup
    if identification == "translation":
        points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        vertices = np.asarray(((0, 2, 3), (0, 1, 2)), dtype=np.int32)
        group = PeriodicCell(np.eye(2))
        representatives = np.zeros((4,), dtype=np.int64)
        shifts = points.astype(np.int64)
    else:
        points = np.asarray(((1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0)))
        vertices = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int32)
        group = PeriodicIsometryGroup(
            np.asarray((((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))
        )
        representatives = np.asarray((0, 1, 0, 1), dtype=np.int64)
        shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
    block = CellBlock(
        "periodic",
        "triangle",
        vertices,
        global_ids=np.asarray((71, 43), dtype=np.int64),
    )
    lifted = CellMesh(points, (block,))
    periodic = PeriodicMeshTopology(
        lifted,
        group,
        representatives,
        shifts,
    )
    mesh = CellMesh(points, (block,), periodic_topology=periodic)
    element = (
        lagrange_element("triangle", 3)
        if family == "H1"
        else form_element(
            "triangle",
            1,
            2,
            twist="twisted" if family == "Hdiv" else "untwisted",
            proxy="flux" if family == "Hdiv" else "circulation",
        )
    )
    plan = FiniteElementPlan(mesh, FiniteElementFieldSpec("u", element))
    authority = FiniteElementGlobalDofOwnership(
        plan, CellPartition(np.asarray((1, 0)), 2)
    )
    repartitioned = FiniteElementGlobalDofOwnership(
        plan, CellPartition(np.asarray((0, 1)), 2)
    )
    assert authority.scientific_keys == repartitioned.scientific_keys
    np.testing.assert_array_equal(authority.global_ids, repartitioned.global_ids)
    expected = np.full((len(authority.scientific_keys),), 2, dtype=np.int32)
    for cell, routes in enumerate(np.asarray(authority.source_dof_map.cell_dofs[0])):
        identifiers = np.asarray(authority.source_to_global)[routes]
        np.minimum.at(expected, identifiers, np.full(identifiers.shape, (1, 0)[cell]))
    np.testing.assert_array_equal(authority.owners, expected)
    assert np.all(expected < 2)

    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    receipt = write_meshing_source_closure(
        tmp_path / "actual-periodic-owner", (authority, repartitioned)
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    for original, loaded in zip((authority, repartitioned), restored, strict=True):
        loaded.validate_restored()
        assert loaded.plan_id == original.plan_id
        assert loaded.scientific_keys == original.scientific_keys
        np.testing.assert_array_equal(
            loaded.source_lifted_frames, original.source_lifted_frames
        )
        np.testing.assert_array_equal(loaded.owners, original.owners)
    bad = eqx.tree_at(lambda value: value.owners, authority, 1 - authority.owners)
    with pytest.raises(ValueError, match="restored_state"):
        bad.validate_restored()
    record = run_on_forced_devices(
        "tests._support.discretization_process_scenarios",
        "cold_periodic_owners",
        (str(receipt.path), receipt.content_id),
        devices=1,
        env={},
        timeout_seconds=120,
    )
    assert record["owner_keys_equal"]


def test_cost_partition_and_phases_are_exactly_once_and_conservative() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            )
        ),
        np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32),
    )
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("u", discontinuous_element("triangle", 2)),
    ).prepare()
    cost_partition = partition_cells_cost_aware(discretization, 2)
    phases = lower_distributed_finite_element_phases(discretization, cost_partition)

    assert cost_partition.evidence.imbalance_ratio >= 1.0
    assert phases.phase_names == (
        "owned-local",
        "halo-update",
        "interface",
        "contribution-sum",
    )
    cell_values = jnp.asarray(((1.5, -0.25), (2.0, 0.75)))
    partition_total = sum(
        (
            phases.local_contribution(part, cell_values)
            for part in range(phases.partition.part_count)
        ),
        start=jnp.zeros((2,)),
    )
    np.testing.assert_allclose(partition_total, jnp.sum(cell_values, axis=0))

    masks = jnp.stack(
        tuple(phases.interface_mask(part) for part in range(phases.partition.part_count))
    )
    np.testing.assert_array_equal(jnp.sum(masks, axis=0), 1)
    flux = jnp.asarray(((0.4, -0.2),))
    serial = phases.facet_ownership.route_equal_opposite(flux)
    partitioned = sum(
        (
            phases.facet_ownership.route_partition(part, flux)
            for part in range(phases.partition.part_count)
        ),
        start=jnp.zeros_like(serial),
    )
    np.testing.assert_allclose(partitioned, serial)
    np.testing.assert_allclose(jnp.sum(partitioned, axis=0), 0.0)


def _owner_local_simplex_source() -> CellMesh:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    identifiers = np.asarray((10, 20, 30, 40, 50, 60), dtype=np.int64)
    block = CellBlock(
        "triangles",
        "triangle",
        np.asarray(((0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)), dtype=np.int32),
        global_ids=np.asarray((100, 200, 300, 400), dtype=np.int64),
    )
    probe = CellMesh(points, (block,), vertex_global_ids=identifiers)
    connectivity = probe.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            "simplex_fixture.mesh.connectivity must be PolygonalConnectivity; got a different connectivity kind"
        )
    endpoints = np.sort(identifiers[np.asarray(connectivity.edges)], axis=1)
    return CellMesh(
        points,
        (block,),
        vertex_global_ids=identifiers,
        entity_global_ids={1: endpoints[:, 0] * 100 + endpoints[:, 1]},
    )


def _owner_local_simplex_pair(
    devices: tuple[jax.Device, ...],
) -> tuple[OwnerLocalFiniteElementDiscretization, ...]:
    """sharded logical arrays; preparation sees only each five-row closure."""
    sharding = NamedSharding(
        Mesh(np.asarray(devices, dtype=object), ("parts",)), PartitionSpec("parts")
    )
    coordinate_shards = (
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), dtype=jnp.float64),
        jnp.asarray(((0.0, 1.0), (1.0, 1.0), (2.0, 1.0)), dtype=jnp.float64),
    )
    cell_shards = (
        jnp.asarray(((0, 1, 3), (1, 4, 3)), dtype=jnp.int32),
        jnp.asarray(((1, 2, 4), (2, 5, 4)), dtype=jnp.int32),
    )
    logical_arrays = (
        (
            "cell_vertices",
            jax.make_array_from_single_device_arrays(
                (4, 3),
                sharding,
                [
                    jax.device_put(value, device)
                    for value, device in zip(cell_shards, devices, strict=True)
                ],
            ),
        ),
        (
            "coordinates",
            jax.make_array_from_single_device_arrays(
                (6, 2),
                sharding,
                [
                    jax.device_put(value, device)
                    for value, device in zip(coordinate_shards, devices, strict=True)
                ],
            ),
        ),
    )
    evidence = canonical_fingerprint(
        {"kind": "test-simplex-pair-ownership", "vertices": 6, "cells": 4}
    )
    topology = _owner_local_simplex_source().topology_id
    geometry = canonical_fingerprint({"kind": "test-two-unit-squares"})
    vertex_owner = {10: 0, 20: 0, 30: 1, 40: 0, 50: 1, 60: 1}
    meshes = []
    for rank, (points, vertices, ids, cell_ids, cell_owners) in enumerate(
        (
            (
                ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
                ((0, 1, 3), (1, 4, 3), (1, 2, 4)),
                (10, 20, 30, 40, 50),
                (100, 200, 300),
                (0, 0, 1),
            ),
            (
                ((1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)),
                ((0, 3, 2), (0, 1, 3), (1, 4, 3)),
                (20, 30, 40, 50, 60),
                (200, 300, 400),
                (0, 1, 1),
            ),
        )
    ):
        local = CellMesh(
            np.asarray(points, dtype=np.float64),
            (
                CellBlock(
                    "triangles",
                    "triangle",
                    np.asarray(vertices, dtype=np.int32),
                    global_ids=np.asarray(cell_ids, dtype=np.int64),
                ),
            ),
            vertex_global_ids=np.asarray(ids, dtype=np.int64),
        )
        connectivity = local.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError(
                f"fixture.connectivity must be a PolygonalConnectivity; got {type(connectivity).__name__}"
            )
        edge_vertices = np.asarray(connectivity.edges)
        endpoints = np.sort(np.asarray(ids, dtype=np.int64)[edge_vertices], axis=1)
        edge_ids = endpoints[:, 0] * 100 + endpoints[:, 1]
        entity_ids = (
            np.asarray(ids, dtype=np.int64),
            edge_ids,
            np.asarray(cell_ids, dtype=np.int64),
        )
        owners = (
            np.asarray([vertex_owner[value] for value in ids], dtype=np.int32),
            np.asarray(
                [vertex_owner[int(left)] for left, _ in endpoints], dtype=np.int32
            ),
            np.asarray(cell_owners, dtype=np.int32),
        )
        storage = CellMeshStorage(
            (6, 9, 4),
            entity_ids,
            owners,
            partition_index=rank,
            partition_count=2,
            logical_topology_id=topology,
            logical_geometry_id=geometry,
            evidence_id=evidence,
            logical_arrays=logical_arrays,
            local_coordinates=local.coordinates,
            local_blocks=local.blocks,
        )
        mesh = CellMesh(
            local.coordinates,
            local.blocks,
            storage=storage,
        )
        meshes.append(mesh)
    halos = prepare_owner_local_halo_plans(
        meshes,
        0,
        devices=devices,
        axis_name="parts",
        message_capacity=2,
    )
    return tuple(
        prepare_owner_local_finite_element(
            FiniteElementPlan(
                mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
            ),
            halo,
            axis_name="parts",
        )
        for mesh, halo in zip(meshes, halos, strict=True)
    )


def _mapped_owner_local_programs(
    programs: tuple[OwnerLocalFiniteElementDiscretization, ...],
    callback: Callable[[OwnerLocalFiniteElementDiscretization], Array],
    devices: tuple[jax.Device, ...],
) -> Array:
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget:
        return map_owner_local_finite_element(
            programs,
            (),
            lambda program: (callback(program),),
            devices,
            budget=budget,
        )[0]


@requires_two_devices
def test_dof_ownership_projection_reuses_prepared_space_and_rejects_wrong_owner() -> None:

    programs = _owner_local_simplex_pair(tuple(jax.local_devices()[:2]))
    plans = tuple(
        FiniteElementPlan(
            program.local_discretization.mesh,
            FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
        )
        for program in programs
    )
    authority = FiniteElementGlobalDofOwnership(
        FiniteElementPlan(
            _owner_local_simplex_source(),
            FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
        ),
        CellPartition(np.asarray((0, 0, 1, 1)), 2),
    )
    packets = authority.prepare_closures(plans, message_capacity=3)
    for plan, (local, witness, halo) in zip(plans, packets, strict=True):
        projected = prepare_owner_local_finite_element(
            plan,
            halo,
            axis_name="parts",
            ownership=witness,
            local_discretization=local,
        )
        assert projected.local_discretization is local
        assert projected.global_dof_count == 6
        wrong = eqx.tree_at(lambda value: value.local_owned, halo, ~halo.local_owned)
        with pytest.raises(ValueError, match="dof_ownership.halo"):
            FiniteElementDofOwnershipPlan(plan, local.dof_maps[0], authority, wrong)
    with pytest.raises(ValueError, match="global_dof_ownership.message_capacity"):
        authority.prepare_closures(plans, message_capacity=2)


@requires_two_devices
def test_source_owned_field_transfer_matches_global_primal_and_paired_pullbacks() -> None:

    devices = tuple(jax.local_devices()[:2])
    closures = _owner_local_simplex_pair(devices)
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    source_plan = FiniteElementPlan(_owner_local_simplex_source(), field)
    source = source_plan.prepare()
    authority = FiniteElementGlobalDofOwnership(
        source_plan,
        CellPartition(np.asarray((0, 0, 1, 1)), 2),
        source_discretization=source,
    )
    transfer = prepare_nested_field_transfer(
        source,
        source,
        np.arange(4, dtype=np.int64),
        field_name="u",
    )
    assert transfer.evidence.passed
    support = owner_local_finite_element_transfer_support(transfer, authority, authority)
    mesh = source_plan.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            "transfer_fixture.mesh.connectivity must provide triangle incidence; got a different connectivity kind"
        )
    vertex_owners = np.asarray(authority.owners)[np.asarray(authority.source_to_global)]
    edge_owners = np.min(vertex_owners[np.asarray(connectivity.edges)], axis=1)
    ids = tuple(np.asarray(entities.entity_ids) for entities in mesh.topology.entity_sets)
    native_owners = (vertex_owners, edge_owners, np.asarray((0, 0, 1, 1), dtype=np.int32))
    original_storage = closures[0].local_discretization.mesh.storage
    if original_storage is None:
        raise TypeError(
            "transfer_fixture.mesh.logical_arrays must retain source arrays; got None"
        )
    local_plans = []
    for rank, required in enumerate(support):
        required_cells = authority.supporting_cells(required)
        assert set(np.asarray(required_cells).tolist()).issubset(set(ids[-1].tolist()))
        storage = CellMeshStorage(
            tuple(entities.count for entities in mesh.topology.entity_sets),
            ids,
            native_owners,
            partition_index=rank,
            partition_count=2,
            logical_topology_id=mesh.topology_id,
            logical_geometry_id=original_storage.logical_geometry_id,
            evidence_id=original_storage.evidence_id,
            logical_arrays=original_storage.logical_arrays,
            local_coordinates=mesh.coordinates,
            local_blocks=mesh.blocks,
        )
        local_plans.append(
            FiniteElementPlan(
                CellMesh(mesh.coordinates, mesh.blocks, storage=storage),
                field,
            )
        )
    programs = authority.prepare_discretizations(
        local_plans, axis_name="parts", message_capacity=4
    )
    for rank, required in enumerate(support):
        assert set(np.asarray(required).tolist()).issubset(
            set(np.asarray(programs[rank].global_ids).tolist())
        )
        assert set(np.asarray(authority.supporting_cells(required)).tolist()).issubset(
            {100, 200, 300, 400}
        )
    paired = prepare_owner_local_finite_element_transfer(
        transfer,
        authority,
        authority,
        programs,
        programs,
    )
    points = source.dof_maps[0].dof_coordinates
    values = jnp.stack((1.0 + points[:, 0], 2.0 - points[:, 1]), axis=1)
    duals = jnp.stack((points[:, 0] - 0.5, points[:, 1] + 0.25), axis=1)
    rows = tuple(authority.global_to_source[program.global_ids] for program in programs)
    forward = execute_owner_local_finite_element_transfer(
        paired,
        tuple(values[index] for index in rows),
        devices=devices,
    )
    transpose = execute_owner_local_finite_element_transfer(
        paired,
        tuple(duals[index] for index in rows),
        devices=devices,
        direction="transpose",
    )
    adjoint = execute_owner_local_finite_element_transfer(
        paired,
        tuple(duals[index] for index in rows),
        devices=devices,
        direction="adjoint",
    )
    expected_forward = transfer.transfer.apply(values)
    expected_transpose = transfer.transfer.pullback(duals)
    expected_adjoint = transfer.transfer.primal.adjoint_mv_block(duals)
    for rank, (program, index) in enumerate(zip(programs, rows, strict=True)):
        mask = program.owned_mask[:, None]
        np.testing.assert_allclose(
            np.asarray(forward)[rank], jnp.where(mask, expected_forward[index], 0)
        )
        np.testing.assert_allclose(
            np.asarray(transpose)[rank], jnp.where(mask, expected_transpose[index], 0)
        )
        np.testing.assert_allclose(
            np.asarray(adjoint)[rank], jnp.where(mask, expected_adjoint[index], 0)
        )


@requires_two_devices
def test_owner_local_p1_collective_mass_stiffness_and_solve() -> None:

    devices = tuple(jax.local_devices()[:2])
    programs = _owner_local_simplex_pair(devices)
    assert programs[0].global_space_id == programs[1].global_space_id
    assert programs[0].local_space_id != programs[1].local_space_id
    assert all(
        program.local_dof_count == 5 and program.global_dof_count == 6
        for program in programs
    )
    assert all(
        program.local_discretization.mesh.coordinates.shape == (5, 2)
        for program in programs
    )
    policy = DistributedKrylovPolicy(
        16, relative_tolerance=1.0e-10, absolute_tolerance=1.0e-12
    )

    affine_fields = tuple(
        jnp.where(
            program.owned_mask,
            1.0
            + program.local_discretization.mesh.coordinates[:, 0]
            + 2.0 * program.local_discretization.mesh.coordinates[:, 1],
            0,
        )
        for program in programs
    )
    rhs = execute_owner_local_finite_element_operator(
        programs,
        affine_fields,
        "mass",
        devices=devices,
    )
    local_rhs = tuple(shard.data[0] for shard in rhs.addressable_shards)
    solved = execute_owner_local_finite_element_mass(
        programs,
        local_rhs,
        policy,
        devices=devices,
        initial=tuple(0.5 * value for value in affine_fields),
    )
    values, accepted = solved.value, solved.accepted
    constant_stiffness = execute_owner_local_finite_element_operator(
        programs,
        tuple(jnp.where(program.owned_mask, 1.0, 0.0) for program in programs),
        "stiffness",
        devices=devices,
    )
    affine_stiffness = execute_owner_local_finite_element_operator(
        programs,
        affine_fields,
        "stiffness",
        devices=devices,
    )
    pairing = _mapped_owner_local_programs(
        programs,
        lambda program: program.pairing.inner(
            program.global_ids / 10.0, program.global_ids / 10.0
        ),
        devices,
    )
    np.testing.assert_allclose(constant_stiffness, 0.0, atol=_P1_CONSTANT_STIFFNESS_ATOL)
    np.testing.assert_array_equal(accepted, True)
    expected_mass = {
        10: 7 / 24,
        20: 30 / 24,
        30: 27 / 24,
        40: 21 / 24,
        50: 42 / 24,
        60: 17 / 24,
    }
    expected_stiffness = {10: -1.5, 20: -2.0, 30: -0.5, 40: 0.5, 50: 2.0, 60: 1.5}
    for rank, program in enumerate(programs):
        ids = np.asarray(program.global_ids)
        owned = np.asarray(program.owned_mask)
        points = np.asarray(program.local_discretization.mesh.coordinates)
        np.testing.assert_allclose(
            rhs[rank],
            [
                expected_mass[int(value)] if flag else 0
                for value, flag in zip(ids, owned, strict=True)
            ],
            atol=_P1_OWNED_MASS_IMAGE_ATOL,
        )
        np.testing.assert_allclose(
            affine_stiffness[rank],
            [
                expected_stiffness[int(value)] if flag else 0
                for value, flag in zip(ids, owned, strict=True)
            ],
            atol=_P1_OWNED_STIFFNESS_IMAGE_ATOL,
        )
        np.testing.assert_allclose(
            values[rank],
            np.where(owned, 1 + points[:, 0] + 2 * points[:, 1], 0),
            atol=_P1_OWNED_SOLVE_VALUES_ATOL,
        )
        with pytest.raises(ValueError, match="bounded export"):
            program.halo.pack_reference(jnp.ones((6,), dtype=jnp.float64))
    np.testing.assert_allclose(pairing, 91.0, atol=_P1_COLLECTIVE_PAIRING_ATOL)
    assert solved.condition_estimate is None
    np.testing.assert_array_equal(solved.status, 0)
    np.testing.assert_array_less(solved.residual_norm, 1.0e-10)
    unfinished = execute_owner_local_finite_element_mass(
        programs,
        local_rhs,
        DistributedKrylovPolicy(1, relative_tolerance=1.0e-10),
        devices=devices,
    )
    np.testing.assert_array_equal(unfinished.status, 1)
    np.testing.assert_array_equal(unfinished.accepted, False)
    with pytest.raises(ValueError, match="execution.buckets"):
        execute_owner_local_finite_element_operator(
            programs,
            (affine_fields[0], affine_fields[1][:-1]),
            "mass",
            devices=devices,
        )


@requires_two_devices
def test_owner_local_collective_rejects_mismatched_stable_id_routes() -> None:

    devices = tuple(jax.local_devices()[:2])
    left, right = _owner_local_simplex_pair(devices)
    # Locally valid route slots, but the peer requests the two shared IDs reversed.
    bad_halo = DistributedHaloPlan.from_owner_local(
        2,
        local_global_ids=right.global_ids,
        local_owned=right.owned_mask,
        global_entity_count=6,
        partition_index=1,
        phase_send_indices=right.halo.phase_send_indices,
        phase_receive_indices=np.asarray(((2, 0),), dtype=np.int32),
        phase_send_valid=right.halo.phase_send_valid,
        phase_receive_valid=right.halo.phase_receive_valid,
        permutations=right.halo.permutations,
        evidence_id=right.distribution_evidence_id,
    )
    bad = eqx.tree_at(lambda program: program.halo, right, bad_halo)
    certificates = _mapped_owner_local_programs(
        (left, bad), lambda program: program.collective_certificate(), devices
    )
    np.testing.assert_array_equal(certificates, False)
    with pytest.raises(ValueError, match="halo_preparation.message_capacity"):
        prepare_owner_local_halo_plans(
            (left.local_discretization.mesh, right.local_discretization.mesh),
            0,
            devices=devices,
            axis_name="parts",
            message_capacity=1,
        )


@requires_two_devices
def test_predecessor_fields_resolve_remote_ids_in_transfer_order_after_device_move() -> (
    None
):

    devices = tuple(jax.local_devices()[:2])
    programs = _owner_local_simplex_pair(devices)
    meshes = tuple(program.local_discretization.mesh for program in programs)
    # Carried values are arbitrary independent components, with deliberately
    # poisoned ghosts. The next predecessor closure asks for previously absent IDs.
    fields = (
        jnp.asarray(
            (
                (2.0, 7.0, 11.0),
                (3.0, 13.0, 17.0),
                (-999.0, -999.0, -999.0),
                (5.0, 19.0, 23.0),
                (-999.0, -999.0, -999.0),
            ),
            dtype=jnp.float64,
        ),
        jnp.asarray(
            (
                (-999.0, -999.0, -999.0),
                (31.0, 37.0, 41.0),
                (-999.0, -999.0, -999.0),
                (43.0, 47.0, 53.0),
                (59.0, 61.0, 67.0),
            ),
            dtype=jnp.float64,
        ),
    )
    fields = tuple(
        jax.device_put(field, device)
        for field, device in zip(fields, devices, strict=True)
    )
    queries = (
        np.asarray((60, 40, 30, 10), dtype=np.int64),
        np.asarray((10, 50, 20), dtype=np.int64),
    )
    owners = (
        np.asarray((1, 0, 1, 0), dtype=np.int32),
        np.asarray((0, 1, 0), dtype=np.int32),
    )
    moved = tuple(reversed(devices))
    resolved = resolve_owner_local_field_values(
        meshes,
        fields,
        queries,
        owners,
        0,
        source_topology_id=meshes[0].topology_id,
        devices=moved,
        axis_name="parts",
        message_capacity=2,
    )
    expected = (
        np.asarray(
            ((59.0, 61.0, 67.0), (5.0, 19.0, 23.0), (31.0, 37.0, 41.0), (2.0, 7.0, 11.0))
        ),
        np.asarray(((2.0, 7.0, 11.0), (43.0, 47.0, 53.0), (3.0, 13.0, 17.0))),
    )
    for value, target, device in zip(resolved, expected, moved, strict=True):
        np.testing.assert_array_equal(value, target)
        assert value.devices() == {device}
    with pytest.raises(
        ValueError, match="resolve_owner_local_field_values.query_global_ids"
    ):
        resolve_owner_local_field_values(
            meshes,
            fields,
            (np.asarray((999,), dtype=np.int64), queries[1]),
            (np.asarray((1,), dtype=np.int32), owners[1]),
            0,
            source_topology_id=meshes[0].topology_id,
            devices=moved,
            axis_name="parts",
            message_capacity=2,
        )


@requires_two_devices
def test_owner_local_p1_mass_uses_original_curved_coordinate_bank_by_default() -> None:

    coordinate_element = coordinate_lagrange_element("triangle", 2)
    coefficients = np.asarray(coordinate_element.reference_nodes).copy()
    coefficients[:, 1] += 0.125 * coefficients[:, 0] * coefficients[:, 1]
    root = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "curved",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((37,), dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": coordinate_element},
        {
            "curved": np.arange(coordinate_element.local_dof_count, dtype=np.int32)[
                None, :
            ]
        },
        coefficients,
    )
    owner = storage(root, geometry)
    ghost = CellMeshStorage(
        owner.global_entity_counts,
        owner.entity_global_ids,
        owner.entity_owner,
        partition_index=1,
        partition_count=2,
        logical_topology_id=owner.logical_topology_id,
        logical_geometry_id=owner.logical_geometry_id,
        logical_coordinate_geometry_id=owner.logical_coordinate_geometry_id,
        evidence_id=owner.evidence_id,
        logical_arrays=owner.logical_arrays,
        local_coordinates=root.coordinates,
        local_blocks=root.blocks,
        global_coordinate_count=owner.global_coordinate_count,
        coordinate_global_ids=owner.coordinate_global_ids,
        coordinate_owner=owner.coordinate_owner,
        local_geometry=geometry,
    )
    meshes = tuple(
        CellMesh(root.coordinates, root.blocks, storage=storage)
        for storage in (owner, ghost)
    )
    devices = tuple(jax.local_devices()[:2])
    halos = prepare_owner_local_halo_plans(
        meshes,
        0,
        devices=devices,
        axis_name="parts",
        message_capacity=3,
    )
    programs = tuple(
        prepare_owner_local_finite_element(
            FiniteElementPlan(
                mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
            ),
            halo,
            axis_name="parts",
        )
        for mesh, halo in zip(meshes, halos, strict=True)
    )
    constants = tuple(jnp.where(program.owned_mask, 1.0, 0.0) for program in programs)
    mass = execute_owner_local_finite_element_operator(
        programs,
        constants,
        "mass",
        devices=devices,
    )
    # det(Dx) = 1 + reference_x/8: its triangle integral is 25/48,
    # whereas silently replacing the source bank by sampled corners gives 1/2.
    # The 1e-13 bounds are 451 float64 ulps at unit scale for the small
    # polynomial quadrature and cell-assembly paths; the PCG comparison uses
    # its 1e-12 relative stopping threshold at this unit-scale constant field.
    np.testing.assert_allclose(jnp.sum(mass), 25.0 / 48.0, atol=_CURVED_P1_MASS_ATOL)
    stiffness = execute_owner_local_finite_element_operator(
        programs,
        constants,
        "stiffness",
        devices=devices,
    )
    np.testing.assert_allclose(stiffness, 0.0, atol=_CURVED_P1_CONSTANT_STIFFNESS_ATOL)
    solved = execute_owner_local_finite_element_mass(
        programs,
        tuple(shard.data[0] for shard in mass.addressable_shards),
        DistributedKrylovPolicy(10, relative_tolerance=1.0e-12),
        devices=devices,
    )
    np.testing.assert_allclose(
        solved.value, np.stack(constants), atol=_CURVED_P1_SOLVE_VALUES_ATOL
    )
    np.testing.assert_array_equal(solved.accepted, True)


@pytest.fixture
def single_partition_halo() -> DistributedHaloPlan:
    """Prepare two owned rows without peer communication."""
    indices = np.empty((0, 1), dtype=np.int32)
    valid = np.empty((0, 1), dtype=np.bool_)
    return DistributedHaloPlan.from_owner_local(
        1,
        local_global_ids=np.asarray((41, 73), dtype=np.int64),
        local_owned=np.asarray((True, True), dtype=np.bool_),
        global_entity_count=2,
        partition_index=0,
        phase_send_indices=indices,
        phase_receive_indices=indices,
        phase_send_valid=valid,
        phase_receive_valid=valid,
        permutations=(),
        evidence_id="single-partition-halo",
    )


def test_owner_local_halo_stored_fields_preserve_identity_and_routing(
    single_partition_halo: DistributedHaloPlan,
) -> None:
    rebuilt = DistributedHaloPlan.from_stored_fields(single_partition_halo)
    assert rebuilt.plan_id == single_partition_halo.plan_id
    assert rebuilt.evidence_id == single_partition_halo.evidence_id
    assert rebuilt.permutations == single_partition_halo.permutations
    np.testing.assert_array_equal(
        rebuilt.local_global_ids, single_partition_halo.local_global_ids
    )
    np.testing.assert_array_equal(rebuilt.local_owned, single_partition_halo.local_owned)
    np.testing.assert_array_equal(rebuilt.local_valid, single_partition_halo.local_valid)
    np.testing.assert_array_equal(
        rebuilt.phase_send_indices, single_partition_halo.phase_send_indices
    )
    np.testing.assert_array_equal(
        rebuilt.phase_receive_indices, single_partition_halo.phase_receive_indices
    )


def test_owner_local_halo_stored_fields_reject_corrupted_mask_shape(
    single_partition_halo: DistributedHaloPlan,
) -> None:
    corrupted = eqx.tree_at(
        lambda value: value.local_owned,
        single_partition_halo,
        jnp.asarray((True,), dtype=jnp.bool_),
    )
    with pytest.raises(ValueError, match="DistributedHaloPlan.local_global_ids"):
        DistributedHaloPlan.from_stored_fields(corrupted)


def test_owner_local_halo_stored_fields_reject_corrupted_id_dtype(
    single_partition_halo: DistributedHaloPlan,
) -> None:
    corrupted = eqx.tree_at(
        lambda value: value.local_global_ids,
        single_partition_halo,
        single_partition_halo.local_global_ids.astype(jnp.float64),
    )
    with pytest.raises(TypeError, match="local_global_ids"):
        DistributedHaloPlan.from_stored_fields(corrupted)


@requires_two_devices
def test_owner_local_execution_view_preserves_scientific_source_identity() -> None:
    source = _owner_local_simplex_pair(tuple(jax.local_devices()[:2]))[0]
    capacities = (
        source.local_dof_count + 1,
        source.cell_owned.size + 1,
        source.mass.relation.capacity + 1,
        source.stiffness.relation.capacity + 1,
        source.halo.message_capacity + 1,
    )
    view = OwnerLocalFiniteElementDiscretization.from_execution_source(source, capacities)
    assert view.execution_source is source
    assert view.execution_capacities == capacities
    assert view.local_discretization is source.local_discretization
    assert view.global_space_id == source.global_space_id
    assert view.local_space_id == source.local_space_id
    assert view.prepared_id == source.prepared_id
    assert view.execution_vector_shape == (capacities[0],) + source.component_shape
    np.testing.assert_array_equal(
        view.global_ids[: source.local_dof_count], source.global_ids
    )
    np.testing.assert_array_equal(view.owned_mask[-1], False)
    np.testing.assert_array_equal(view.cell_owned[-1], False)
    view.validate_restored()


def test_owner_local_halo_rejects_missing_evidence_metadata(
    single_partition_halo: DistributedHaloPlan,
) -> None:
    halo = single_partition_halo
    with pytest.raises(ValueError, match="DistributedHaloPlan.local_global_ids"):
        DistributedHaloPlan.from_owner_local(
            halo.part_count,
            local_global_ids=halo.local_global_ids,
            local_owned=halo.local_owned,
            global_entity_count=halo.entity_count,
            partition_index=0,
            phase_send_indices=halo.phase_send_indices,
            phase_receive_indices=halo.phase_receive_indices,
            phase_send_valid=halo.phase_send_valid,
            phase_receive_valid=halo.phase_receive_valid,
            permutations=halo.permutations,
            evidence_id=None,  # ty: ignore[invalid-argument-type]
        )

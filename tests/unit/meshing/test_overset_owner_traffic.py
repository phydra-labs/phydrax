# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path


def test_actual_two_process_sparse_donor_fringe_action_and_transpose() -> None:
    from phydrax.execution import (
        launch_local_processes,
        LocalProcessLaunchError,
        LocalProcessLaunchPlan,
    )

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ)
    environment.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        PHYDRAX_CPU_COLLECTIVES="gloo",
    )
    try:
        results = launch_local_processes(
            (sys.executable, str(Path(__file__).resolve()), "worker"),
            LocalProcessLaunchPlan(2, f"127.0.0.1:{port}"),
            environment=environment,
        )
    except LocalProcessLaunchError as error:
        for result in error.results:
            print(result.stdout.decode(), end="")
            print(result.stderr.decode(), end="", file=sys.stderr)
        raise
    records = [json.loads(result.stdout.decode().splitlines()[-1]) for result in results]
    assert [record["rank"] for record in records] == [0, 1]
    assert all(
        record["processes"] == 2 and record["collective_permute"] for record in records
    )
    assert all(not record["all_gather"] for record in records)
    assert all(record["nonaddressable_routes"] for record in records)
    for profile, capacity in (
        ("fe", 6),
        ("fe_vector", 12),
        ("fv", 1),
        ("fe_image_vector", 12),
        ("fe_image_invariant", 12),
        ("fe_affine_vector", 12),
    ):
        assert all(record[f"{profile}_packet_capacity"] == capacity for record in records)
    assert records[0]["fe_served"] == records[1]["fe_requested"] == 6
    assert records[0]["fe_vector_served"] == records[1]["fe_vector_requested"] == 12
    assert (
        records[0]["fe_image_vector_served"]
        == records[1]["fe_image_vector_requested"]
        == 12
    )
    assert (
        records[0]["fe_image_invariant_served"]
        == records[1]["fe_image_invariant_requested"]
        == 12
    )
    assert (
        records[0]["fe_affine_vector_served"]
        == records[1]["fe_affine_vector_requested"]
        == 12
    )
    assert records[0]["fv_served"] == records[1]["fv_requested"] == 1
    for record in records:
        assert record["max_action_error"] < 1e-11
        assert record["max_transpose_error"] < 1e-11
        assert record["max_duality_error"] < 1e-11
        assert record["bounded_refusal"] and record["missing_owner_refusal"]
        assert record["nonrigid_polar_refusal"]
    print(json.dumps(records, sort_keys=True))


def _worker() -> None:
    """Keep both ranks' bootstrap, image profiles and collective refusal probes
    in one explicit call order; no branch may reorder the shared communication
    phases before either process reports its isolated local evidence.
    """
    from phydrax.execution import initialize_from_environment

    initialize_from_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec

    from phydrax import SpatialCoordinateContract
    from phydrax._execution_runtime import ExecutionRuntime
    from phydrax.discretization import CellBlock, CellMesh
    from phydrax.discretization._distributed_field import make_owner_local_field_array
    from phydrax.discretization.fem import (
        FiniteElementFieldSpec,
        FiniteElementPlan,
        lagrange_element,
    )
    from phydrax.discretization.finite_volume import (
        PiecewiseConstantReconstruction,
        prepare_finite_volume_field_reconstruction,
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.meshing import certify_cell_mesh
    from phydrax.meshing._assembly import MeshAssembly, MeshPart
    from phydrax.meshing._overset import (
        OversetPartSpec,
        OversetPolicy,
        prepare_overset_connectivity,
        prepare_overset_field_transfer,
    )
    from phydrax.meshing._overset_exchange import _owner_local_overset_action

    rank = jax.process_index()
    source_mesh = CellMesh(
        np.asarray(((0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)), dtype=np.float64),
        (
            CellBlock(
                "cells", "triangle", np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32)
            ),
        ),
    )
    target_mesh = CellMesh(
        np.asarray(((0.3, 0.3), (1.1, 0.3), (0.3, 1.1)), dtype=np.float64),
        (CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),),
    )
    source_result = certify_cell_mesh(source_mesh, SpatialCoordinateContract.si())
    source = MeshPart("donor", source_result)
    target = MeshPart(
        "fringe", certify_cell_mesh(target_mesh, SpatialCoordinateContract.si())
    )
    boundary = target.scope(1, np.asarray(target_mesh.entity_set(1).entity_ids))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec("donor", owner_rank=0),
            OversetPartSpec("fringe", boundary=boundary, owner_rank=1),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float64)
    affine_matrix = np.asarray(((0.6, -0.8), (0.8, 0.6)), dtype=np.float64)
    translation = np.asarray((6.0, -2.0), dtype=np.float64)
    image_connectivities = {}
    for image_name, matrix in (("rigid", rotation), ("affine", affine_matrix)):
        image_target_mesh = CellMesh(
            np.asarray(target_mesh.coordinates) @ matrix.T + translation,
            (CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),),
        )
        image_target = MeshPart(
            "fringe", certify_cell_mesh(image_target_mesh, SpatialCoordinateContract.si())
        )
        image_connectivities[image_name] = prepare_overset_connectivity(
            MeshAssembly((source, image_target)),
            (
                OversetPartSpec(
                    "donor",
                    owner_rank=0,
                    image_rotation=matrix,
                    image_translation=translation,
                ),
                OversetPartSpec(
                    "fringe",
                    owner_rank=1,
                    boundary=image_target.scope(
                        1, np.asarray(image_target_mesh.entity_set(1).entity_ids)
                    ),
                ),
            ),
            policy=OversetPolicy(fringe_layers=1),
        )
    group = ExecutionRuntime.current(mesh_axes=(("parts", 2),)).root_group
    fe = FiniteElementPlan(
        source_mesh,
        FiniteElementFieldSpec("u", lagrange_element("triangle", 2)),
        coordinate_spec=source_result.geometry,
    ).prepare()
    fe_vector = FiniteElementPlan(
        source_mesh,
        FiniteElementFieldSpec(
            "u", lagrange_element("triangle", 2), component_shape=(2,)
        ),
        coordinate_spec=source_result.geometry,
    ).prepare()
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(source_mesh).prepare()
    fv_owner = prepare_finite_volume_field_reconstruction(
        fv, PiecewiseConstantReconstruction()
    )
    nodes = np.asarray(fe.dof_maps[0].dof_coordinates)
    fe_values = (
        jnp.asarray(3 + nodes[:, 0] ** 2 + nodes[:, 0] * nodes[:, 1] - 2 * nodes[:, 1])
        if rank == 0
        else None
    )
    fe_vector_values = (
        jnp.stack((fe_values, 1 + nodes[:, 0] - nodes[:, 1] ** 2), axis=1)
        if fe_values is not None
        else None
    )
    fv_values = (
        jnp.asarray((4.2, -9.1), dtype=jnp.float64).reshape(fv_owner.coefficient_shape)
        if rank == 0
        else None
    )
    evidence = {
        "rank": rank,
        "processes": jax.process_count(),
        "max_action_error": 0.0,
        "max_transpose_error": 0.0,
        "max_duality_error": 0.0,
        "all_gather": False,
        "collective_permute": False,
        "nonaddressable_routes": True,
        "bounded_refusal": False,
        "missing_owner_refusal": False,
        "nonrigid_polar_refusal": False,
    }
    try:
        prepare_overset_field_transfer(
            image_connectivities["affine"],
            {"donor": fe_vector},
            "u",
            value_action="polar-vector",
        )
    except ValueError:
        evidence["nonrigid_polar_refusal"] = True
    for profile, owner, values in (
        ("fe", fe, fe_values),
        ("fe_vector", fe_vector, fe_vector_values),
        ("fv", fv_owner, fv_values),
        ("fe_image_vector", fe_vector, fe_vector_values),
        ("fe_image_invariant", fe_vector, fe_vector_values),
        ("fe_affine_vector", fe_vector, fe_vector_values),
    ):
        registration = (
            image_connectivities["affine"]
            if profile == "fe_affine_vector"
            else image_connectivities["rigid"]
            if profile.startswith("fe_image_")
            else connectivity
        )
        action = (
            "polar-vector"
            if profile == "fe_image_vector"
            else "contravariant-vector"
            if profile == "fe_affine_vector"
            else "invariant"
            if profile == "fe_image_invariant"
            else None
        )
        transfer = prepare_overset_field_transfer(
            registration,
            {"donor": owner},
            "u",
            value_action=action,
        )
        scalar_count = int(np.prod(transfer.coefficient_shapes[0]))
        stable_ids = 2**40 + 17 * np.arange(scalar_count, dtype=np.int64)
        exchange = transfer.prepare_owner_local_exchange(
            {"donor": stable_ids},
            group,
            message_capacity=12,
        )
        evidence["nonaddressable_routes"] = (
            evidence["nonaddressable_routes"]
            and not exchange.route_weights.is_fully_addressable
            and not exchange.send_indices.is_fully_addressable
            and len(exchange.route_weights.addressable_shards) == 1
        )
        evidence[f"{profile}_packet_capacity"] = exchange.message_capacity
        local_coefficients = {} if values is None else {"donor": values}
        result = exchange.apply(local_coefficients)
        for name, result_values in result.items():
            points = np.asarray(target_mesh.coordinates)[
                np.asarray(registration.receptors_of("fringe").receptor_rows)
            ]
            polynomial = (
                3 + points[:, 0] ** 2 + points[:, 0] * points[:, 1] - 2 * points[:, 1]
            )
            if name != "fringe":
                expected = np.empty(result_values.shape, dtype=np.float64)
            elif profile in (
                "fe_vector",
                "fe_image_vector",
                "fe_image_invariant",
                "fe_affine_vector",
            ):
                expected = np.stack(
                    (polynomial, 1 + points[:, 0] - points[:, 1] ** 2), axis=1
                )
                if profile in ("fe_image_vector", "fe_affine_vector"):
                    matrix = affine_matrix if profile == "fe_affine_vector" else rotation
                    expected = expected @ matrix.T
            elif profile == "fe":
                expected = polynomial
            else:
                expected = np.full(result_values.shape, 4.2, dtype=np.float64)
            error = np.max(np.abs(np.asarray(result_values) - expected), initial=0.0)
            evidence["max_action_error"] = max(evidence["max_action_error"], float(error))
        refreshed = exchange.apply({} if values is None else {"donor": values + 1.25})
        for name, result_values in result.items():
            matrix = affine_matrix if profile == "fe_affine_vector" else rotation
            delta = (
                np.full(result_values.shape, 1.25, dtype=np.float64) @ matrix.T
                if profile in ("fe_image_vector", "fe_affine_vector") and name == "fringe"
                else 1.25
            )
            error = np.max(
                np.abs(np.asarray(refreshed[name]) - np.asarray(result_values) - delta),
                initial=0.0,
            )
            evidence["max_action_error"] = max(evidence["max_action_error"], float(error))
        dual = {}
        for receptor in transfer.receptors:
            shape = (receptor.receptor_ids.size, *transfer.value_shape)
            dual[receptor.part_name] = (
                jnp.arange(int(np.prod(shape)), dtype=jnp.float64).reshape(shape) + 0.7
            )
        local_dual = {name: dual[name] for name in result}
        scattered = exchange.transpose(local_dual)
        serial_scattered = transfer.transpose(dual) if rank == 0 else {}
        for name, result_values in scattered.items():
            error = np.max(
                np.abs(np.asarray(result_values) - np.asarray(serial_scattered[name])),
                initial=0.0,
            )
            evidence["max_transpose_error"] = max(
                evidence["max_transpose_error"], float(error)
            )
        left, right = (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        for name, result_values in result.items():
            left = left + jnp.vdot(result_values, local_dual[name])
        for name, coefficients in local_coefficients.items():
            right = right + jnp.vdot(coefficients, scattered[name])
        products = make_owner_local_field_array(
            (jnp.stack((left, right)),),
            (rank,),
            group.devices,
            axis_name="parts",
        )
        pairing = jax.shard_map(
            lambda local: jax.lax.psum(local[0], "parts"),
            mesh=group.mesh,
            in_specs=PartitionSpec("parts"),
            out_specs=PartitionSpec(),
            check_vma=False,
        )(products)
        pairing = np.asarray(pairing)
        evidence["max_duality_error"] = max(
            evidence["max_duality_error"], float(abs(pairing[0] - pairing[1]))
        )
        # Lower the exercised dynamic route, not query preparation or a reference
        # implementation, to publish actual communication resource evidence.
        for transpose in (False, True):
            input_capacity = (
                exchange.output_capacity if transpose else exchange.owned_capacity
            )
            local_input = make_owner_local_field_array(
                (jnp.zeros(input_capacity, dtype=jnp.float64),),
                (rank,),
                group.devices,
                axis_name="parts",
            )
            # Equinox types filter_jit as Callable, omitting its real lower interface.
            lowered = _owner_local_overset_action.lower(exchange, local_input, transpose)  # ty: ignore[unresolved-attribute]
            hlo = lowered.lowered.compiler_ir(dialect="hlo").as_hlo_text()
            evidence["all_gather"] = evidence["all_gather"] or "all-gather" in hlo
            evidence["collective_permute"] = (
                evidence["collective_permute"] or "collective-permute" in hlo
            )
        evidence[f"{profile}_requested"] = exchange.requested_support_count
        evidence[f"{profile}_served"] = exchange.served_support_count
        if profile == "fe":
            try:
                transfer.prepare_owner_local_exchange(
                    {"donor": stable_ids}, group, message_capacity=5
                )
            except ValueError:
                evidence["bounded_refusal"] = True
            wrong_ids = stable_ids + (1 if rank == 0 else 0)
            try:
                transfer.prepare_owner_local_exchange(
                    {"donor": wrong_ids}, group, message_capacity=6
                )
            except ValueError:
                evidence["missing_owner_refusal"] = True
    print(json.dumps(evidence, sort_keys=True), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    _worker()

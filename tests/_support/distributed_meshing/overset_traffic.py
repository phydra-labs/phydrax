# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Sparse donor/fringe action, transpose, image, and refusal evidence."""

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from phydrax._execution_runtime import ExecutionGroup, ExecutionRuntime
from phydrax.discretization import PreparedFieldReconstruction
from phydrax.discretization._distributed_field import make_owner_local_field_array
from phydrax.discretization.fem import FiniteElementDiscretization
from phydrax.execution import runtime_info
from phydrax.meshing._overset import (
    OversetConnectivity,
    prepare_overset_field_transfer,
    PreparedOversetFieldTransfer,
)
from phydrax.meshing._overset_exchange import (
    owner_local_overset_action,
    PreparedOwnerLocalOversetExchange,
)
from tests._support.distributed_meshing.overset_traffic_builders import (
    build_traffic_case,
    OversetTrafficCase,
)


def _global_pairing(local: jax.Array) -> jax.Array:
    return jax.lax.psum(local[0], "parts")


def _action_evidence(
    case: OversetTrafficCase,
    profile: str,
    registration: OversetConnectivity,
    transfer: PreparedOversetFieldTransfer,
    exchange: PreparedOwnerLocalOversetExchange,
    values: jax.Array | None,
    group: ExecutionGroup,
    rank: int,
    evidence: dict[str, Any],
) -> None:
    target_mesh = case.target_mesh
    rotation = case.rotation
    affine_matrix = case.affine_matrix
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
        _global_pairing,
        mesh=group.mesh,
        in_specs=PartitionSpec("parts"),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(products)
    pairing = np.asarray(pairing)
    evidence["max_duality_error"] = max(
        evidence["max_duality_error"], float(abs(pairing[0] - pairing[1]))
    )


def _exercise_profile(
    case: OversetTrafficCase,
    profile: str,
    owner: FiniteElementDiscretization | PreparedFieldReconstruction,
    values: jax.Array | None,
    group: ExecutionGroup,
    rank: int,
    evidence: dict[str, Any],
) -> None:
    image_connectivities = case.image_connectivities
    connectivity = case.connectivity
    registration = (
        image_connectivities["affine"]
        if profile == "fe_affine_vector"
        else image_connectivities["rigid"]
        if profile.startswith("fe_image_")
        else connectivity
    )
    action: Literal["polar-vector", "contravariant-vector", "invariant"] | None = (
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
    _action_evidence(
        case, profile, registration, transfer, exchange, values, group, rank, evidence
    )
    # Lower the exercised dynamic route to retain communication resource evidence.
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
        lowered = owner_local_overset_action.lower(exchange, local_input, transpose)  # ty: ignore[unresolved-attribute]
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


def sparse_donor_fringe_traffic() -> dict[str, Any]:
    """Return both numerical and communication evidence after rank shutdown.

    Both ranks execute image profiles and collective refusal probes in the same
    order before either publishes its local evidence.
    """
    runtime_info()
    try:
        rank = jax.process_index()
        case = build_traffic_case(rank)
        group = ExecutionRuntime.current(mesh_axes=(("parts", 2),)).root_group
        evidence: dict[str, Any] = {
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
                case.image_connectivities["affine"],
                {"donor": case.fe_vector},
                "u",
                value_action="polar-vector",
            )
        except ValueError:
            evidence["nonrigid_polar_refusal"] = True
        for profile, owner, values in (
            ("fe", case.fe, case.fe_values),
            ("fe_vector", case.fe_vector, case.fe_vector_values),
            ("fv", case.fv_owner, case.fv_values),
            ("fe_image_vector", case.fe_vector, case.fe_vector_values),
            ("fe_image_invariant", case.fe_vector, case.fe_vector_values),
            ("fe_affine_vector", case.fe_vector, case.fe_vector_values),
        ):
            _exercise_profile(case, profile, owner, values, group, rank, evidence)
        return evidence
    finally:
        jax.distributed.shutdown()

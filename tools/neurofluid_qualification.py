#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Synthetic substrate checks and genuine native compartment transport evidence.

Run ``JAX_ENABLE_X64=1 python -m tools.neurofluid_qualification
--scenario native-generated-transport --size 2 --repeats 3 --steps 5 --dt 0.01``.
The native scenario generates the bulk mesh from occupied labeled voxels; it
never substitutes the handbuilt tetrahedron used by the synthetic H(div) check.
Acceptance compares each discrete local/material inventory equation and the
continuous-time transport solution against an independent host assembly.
Compiler estimates and process-lifetime/device allocation peaks are distinct.
An absent native route or kernel is a prerequisite failure, not a comparison
engine fallback. Targets are requested objectives, not claims of qualification.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._fingerprint import array_tree_fingerprint
from phydrax._meshcore import meshcore_identity, meshcore_runtime_identity, MeshcoreStatus
from phydrax.meshing._measurements import NativeMeshingPhaseMeasurement
from tools._meshing_cases import python_source_revision


def manifest(payload: bytes) -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "qualification-image",
        checksum_algorithm="sha256",
        checksum=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )


def _segmentation(
    size: int,
) -> tuple[
    phx.SpatialCoordinateContract,
    phx.imaging.LabelVolume,
    phx.geometry.CompartmentComplex,
    phx.imaging.CompartmentSurfaceResult,
]:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="qualification",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    values = np.ones((size, size, size), dtype=np.int16)
    values[size // 2 :] = 2
    asset = phx.imaging.MedicalImageAsset(
        "qualification-labels",
        "segmentation",
        values,
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "qualification-deid", "subject-0", "synthetic", True, True, True
        ),
        (manifest(values.tobytes()),),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-qualification-generator",
        ),
    )
    labels = phx.imaging.LabelVolume(
        asset,
        phx.imaging.LabelOntology(
            "qualification-labels",
            "synthetic",
            "1",
            (
                phx.imaging.LabelDefinition(1, "first-label", "First"),
                phx.imaging.LabelDefinition(2, "second-label", "Second"),
            ),
        ),
    )
    definitions = (
        phx.geometry.CompartmentDefinition(
            "first", ("first-label",), "material", allowed_neighbor_ids=("second",)
        ),
        phx.geometry.CompartmentDefinition(
            "second", ("second-label",), "material", allowed_neighbor_ids=("first",)
        ),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "first-second", "first", "second", "exchange"
    )
    complex_ = phx.imaging.build_compartment_complex(labels, definitions, (interface,))
    surfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)
    return contract, labels, complex_, surfaces


def qualify() -> dict[str, object]:
    """Qualify the explicit synthetic substrate scenario, not native generation."""
    _, _, complex_, surfaces = _segmentation(2)
    element = phx.discretization.fem.form_element(
        "tetrahedron", 2, 1, family="trimmed", twist="twisted", proxy="flux"
    )
    centers = np.asarray(
        (
            (1 / 3, 1 / 3, 0.0),
            (1 / 3, 0.0, 1 / 3),
            (1 / 3, 1 / 3, 1 / 3),
            (0.0, 1 / 3, 1 / 3),
        )
    )
    values_rt, _ = element.tabulate(centers)
    normals = np.asarray(
        ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 1.0, 1.0), (-1.0, 0.0, 0.0))
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    areas = np.asarray((0.5, 0.5, np.sqrt(3.0) / 2.0, 0.5))
    flux = np.asarray(
        [
            [
                areas[face] * np.dot(values_rt[face, basis], normals[face])
                for basis in range(4)
            ]
            for face in range(4)
        ]
    )
    _, center_gradients = element.tabulate(np.full((1, 3), 0.25, dtype=np.float64))
    integrated_divergence = (
        np.trace(np.asarray(center_gradients[0]), axis1=-2, axis2=-1) / 6.0
    )
    flux_error = float(np.max(np.abs(np.sum(flux, axis=0) - integrated_divergence)))
    tetrahedron = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        np.asarray(((0, 1, 2, 3),)),
    )
    boundary_face_id = int(np.asarray(tetrahedron.entity_set(2).entity_ids)[0])
    normal_boundary = phx.equations.fem.HDivNormalBoundaryCondition(
        np.asarray((boundary_face_id,)),
        resistance=2.0,
        prescribed_flux=1.0,
    )
    hdiv = phx.equations.fem.HDivStokesPlan(
        tetrahedron,
        phx.discretization.PressureGaugePolicy("mean-zero"),
        normal_boundaries=(normal_boundary,),
    ).prepare()
    zero_velocity, pressure, multiplier = hdiv.state_space.zeros()
    velocity = hdiv.normal_flux_operator.adjoint_mv(np.asarray((1.0 / 6.0,)))
    normal_flow_error = float(
        np.max(np.abs(np.asarray(hdiv.normal_flux(velocity)) - 1.0))
    )
    resistance_power_error = abs(
        float(
            np.vdot(
                np.asarray(velocity),
                np.asarray(hdiv.normal_resistance_operator.mv(velocity)),
            )
        )
        - 2.0
    )
    constraint_error = float(
        np.max(np.abs(np.asarray(hdiv.residual((velocity, pressure, multiplier))[2])))
    )
    if (
        not complex_.adjacency.successful
        or flux_error > 1.0e-12
        or not bool(hdiv.evidence.successful)
        or normal_flow_error > 1.0e-12
        or resistance_power_error > 1.0e-12
        or constraint_error > 1.0e-12
    ):
        raise RuntimeError("Neurofluid synthetic qualification failed.")
    return {
        "scenario": "synthetic",
        "compartments": len(complex_.compartments),
        "interfaces": len(surfaces.surfaces),
        "interface_triangles": surfaces.surfaces[0].surface.mesh.entity_set(2).count,
        "rt0_flux_error": flux_error,
        "bdm2_velocity_dofs": zero_velocity.size,
        "dg1_pressure_dofs": pressure.size,
        "normal_flow_error": normal_flow_error,
        "resistance_power_error": resistance_power_error,
        "normal_constraint_error": constraint_error,
        "successful": True,
    }


def _native_source(
    size: int,
) -> tuple[
    phx.geometry.CompartmentMeshingSource,
    phx.discretization.PreparedMetricNetwork,
]:
    contract, labels, compartments, interfaces = _segmentation(size)
    lower, upper = -0.5, size - 0.5
    points = np.asarray(
        (
            (lower, lower, lower),
            (upper, lower, lower),
            (upper, upper, lower),
            (lower, upper, lower),
            (lower, lower, upper),
            (upper, lower, upper),
            (upper, upper, upper),
            (lower, upper, upper),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int64,
    )
    outer = phx.geometry.SurfaceModel.from_triangles(
        points,
        triangles,
        phx.geometry.SurfaceMetadata(
            source_id="qualification-outer",
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=("synthetic",),
        ),
    )
    source = phx.geometry.CompartmentMeshingSource(
        labels, compartments, outer, interfaces
    )
    network = phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((0.0, 0.0, 0.0), (size - 1.0,) * 3), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int64),
        contract,
        areas=np.asarray((0.01,), dtype=np.float64),
        perimeters=np.asarray((0.2,), dtype=np.float64),
        root_vertex_ids=np.asarray((0,), dtype=np.int64),
        tip_vertex_ids=np.asarray((1,), dtype=np.int64),
    ).prepare()
    return source, network


def _transport_parameters(
    result: phx.meshing.CellMeshingResult,
) -> phx.applications.neurofluid.NeurofluidTransportParameters:
    count = result.mesh.entity_set(3).count
    boundary_count = np.count_nonzero(
        np.asarray(result.mesh.entity_set(2).subset("boundary").mask)
    )
    return phx.applications.neurofluid.NeurofluidTransportParameters(
        porosity=np.full(count, 0.8, dtype=np.float64),
        bulk_diffusivity=np.full(count, 0.01, dtype=np.float64),
        bulk_velocity=np.zeros((count, 3), dtype=np.float64),
        bulk_boundary_volume_flux=np.zeros(boundary_count, dtype=np.float64),
        bulk_boundary_inflow_concentration=np.zeros(boundary_count, dtype=np.float64),
        bulk_removal_rate=np.zeros(count, dtype=np.float64),
        network_diffusivity=np.asarray((0.01,), dtype=np.float64),
        units=phx.applications.neurofluid.NeurofluidTransportUnits(
            phx.units.MILLIMETER, phx.units.SECOND, phx.units.MILLIMOLAR
        ),
        network_volume_flow=np.zeros(1, dtype=np.float64),
        exchange_coefficients=np.full(2, 0.5, dtype=np.float64),
        averaging_radius=0.01,
        reservoir_volumes=np.asarray((0.003,), dtype=np.float64),
        reservoir_coefficients=np.asarray((0.2,), dtype=np.float64),
    )


def _independent_bulk(result: Any, size: int) -> tuple[Any, Any, Any, dict[str, object]]:
    """Integrate tetrahedra and build material fluxes from vertex incidence."""
    evidence = result.region_evidence
    if evidence is None:
        raise RuntimeError("Native compartment route omitted required region evidence.")
    cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    tetrahedra = points[cells]
    volumes = np.abs(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])) / 6.0
    centers = np.mean(tetrahedra, axis=1)
    regions = np.asarray(
        [0 if region == "first" else 1 for region in evidence.cell_region_ids],
        dtype=np.int64,
    )
    if set(evidence.cell_region_ids) != {"first", "second"}:
        raise RuntimeError("Native route lost authoritative compartment identities.")
    split = size // 2 - 0.5
    tolerance = 1.0e-10
    if np.any(tetrahedra[regions == 0, :, 0] > split + tolerance) or np.any(
        tetrahedra[regions == 1, :, 0] < split - tolerance
    ):
        raise RuntimeError(
            "A generated tetrahedron crosses the source material interface."
        )
    region_volumes = np.bincount(regions, weights=volumes, minlength=2)
    expected = np.asarray((size // 2, size - size // 2), dtype=np.float64) * size**2
    volume_error = float(np.max(np.abs(region_volumes - expected)))
    faces: dict[tuple[int, ...], list[int]] = {}
    for cell_index, vertices in enumerate(cells):
        for face in combinations(vertices, 3):
            faces.setdefault(tuple(sorted(int(vertex) for vertex in face)), []).append(
                cell_index
            )
    stiffness = np.zeros((len(cells), len(cells)), dtype=np.float64)
    interface = np.zeros(len(cells), dtype=np.float64)
    interface_area = 0.0
    interface_count = 0
    for face, adjacent in faces.items():
        if len(adjacent) == 1:
            continue
        if len(adjacent) != 2:
            raise RuntimeError("Generated compartment mesh has nonmanifold faces.")
        owner, neighbor = adjacent
        vertices = points[np.asarray(face, dtype=np.int64)]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        norm = np.linalg.norm(normal)
        distance = abs(np.dot(centers[neighbor] - centers[owner], normal / norm))
        weight = 0.01 * 0.5 * norm / distance
        stiffness[owner, owner] += weight
        stiffness[neighbor, neighbor] += weight
        stiffness[owner, neighbor] -= weight
        stiffness[neighbor, owner] -= weight
        if regions[owner] != regions[neighbor]:
            first, second = (
                (owner, neighbor) if regions[owner] == 0 else (neighbor, owner)
            )
            interface[first] += weight
            interface[second] -= weight
            interface_area += 0.5 * norm
            interface_count += 1
    if (
        volume_error > tolerance
        or abs(interface_area - size**2) > tolerance
        or interface_count != len(evidence.interface_facets)
        or evidence.adjacency_pairs != (("first", "second"),)
    ):
        raise RuntimeError("Independent material coverage or interface adjacency failed.")
    return (
        volumes * 0.8,
        stiffness,
        interface,
        {
            "compartment_volumes": region_volumes.tolist(),
            "volume_error": volume_error,
            "interface_area": interface_area,
            "interface_facets": interface_count,
            "adjacency_pairs": evidence.adjacency_pairs,
        },
    )


def _independent_system(
    result: Any,
    runtime: Any,
    size: int,
) -> tuple[Any, Any, Any, Any, dict[str, object]]:
    bulk_mass, bulk_matrix, interface, geometry = _independent_bulk(result, size)
    transfer = runtime.exchange.transfer
    stencil = transfer.sampling.stencil
    indices = np.asarray(stencil.relation.source_indices)
    weights = np.asarray(stencil.weights)
    valid = np.asarray(stencil.relation.valid)
    quadrature = np.asarray(transfer.quadrature_weights)
    routes = indices.reshape((2, quadrature.shape[1], -1))
    route_weights = (weights * valid).reshape(routes.shape) * quadrature[:, :, None]
    average = np.zeros((2, len(bulk_mass)), dtype=np.float64)
    for target in range(2):
        np.add.at(average[target], routes[target].ravel(), route_weights[target].ravel())
    length = np.sqrt(3.0) * (size - 1)
    network_mass = np.full(2, 0.01 * length / 2.0, dtype=np.float64)
    mass = np.concatenate(
        (bulk_mass, network_mass, np.asarray((0.003,), dtype=np.float64))
    )
    count = len(bulk_mass)
    matrix = np.zeros((len(mass), len(mass)), dtype=np.float64)
    matrix[:count, :count] = bulk_matrix
    network_weight = 0.01 * 0.01 / length
    matrix[count : count + 2, count : count + 2] = network_weight * np.asarray(
        ((1.0, -1.0), (-1.0, 1.0)), dtype=np.float64
    )
    for target in range(2):
        difference = np.zeros(len(mass), dtype=np.float64)
        difference[:count] = average[target]
        difference[count + target] = -1.0
        matrix += 0.5 * network_mass[target] * np.outer(difference, difference)
    difference = np.zeros(len(mass), dtype=np.float64)
    difference[count + 1], difference[-1] = 1.0, -1.0
    matrix += 0.003 * 0.2 * np.outer(difference, difference)
    return mass, matrix, average, interface, geometry


def _transport_invariants(
    result: Any,
    runtime: Any,
    initial: Any,
    history: Any,
    accepted: Any,
    *,
    size: int,
    steps: int,
    dt: float,
    target_error: float,
    balance_tolerance: float,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    mass, matrix, average, interface, geometry = _independent_system(
        result, runtime, size
    )
    count = len(initial.bulk)
    first = np.concatenate(
        (
            np.asarray(initial.bulk),
            np.asarray(initial.network),
            np.asarray(initial.reservoirs),
        )
    )
    states = np.concatenate(
        (
            np.asarray(history.bulk),
            np.asarray(history.network),
            np.asarray(history.reservoirs),
        ),
        axis=1,
    )
    before = np.concatenate((first[None], states[:-1]), axis=0)
    residuals = mass * (states - before) / dt + states @ matrix.T
    independent_inventory = states @ mass
    inventory_error = float(np.max(np.abs(independent_inventory - first @ mass)))
    balance_error = float(np.max(np.abs(residuals)))
    source_transfer = runtime.exchange.transfer.average(initial.bulk)
    # Both averaging circles lie wholly inside their authoritative material;
    # the initial field is constant there, independent of prepared route weights.
    expected_average = np.asarray((3.0, 2.0), dtype=np.float64)
    transfer_error = float(
        max(
            np.max(np.abs(np.asarray(source_transfer.values) - expected_average)),
            np.max(np.abs(average @ first[:count] - expected_average)),
        )
    )
    exchange_rates = (states[:, :count] @ average.T - states[:, count : count + 2]) * (
        0.5 * mass[count : count + 2]
    )
    region_mask = np.asarray(
        [region == "first" for region in result.region_evidence.cell_region_ids],
        dtype=np.bool_,
    )
    material_flux = states[:, :count] @ interface
    region_exchange = exchange_rates @ average[:, region_mask].sum(axis=1)
    region_inventory_change = (
        (states[:, :count] - before[:, :count])[:, region_mask]
        @ mass[:count][region_mask]
        / dt
    )
    interface_balance = float(
        np.max(np.abs(region_inventory_change + material_flux + region_exchange))
    )
    root_mass = np.sqrt(mass)
    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix / root_mass[:, None] / root_mass[None, :]
    )
    exact = (
        eigenvectors
        @ (np.exp(-steps * dt * eigenvalues) * (eigenvectors.T @ (root_mass * first)))
    ) / root_mass
    physical_error = float(
        np.linalg.norm(root_mass * (states[-1] - exact))
        / np.linalg.norm(root_mass * exact)
    )
    minimum = float(np.min(states))
    successful = bool(
        np.all(np.asarray(accepted))
        and bool(source_transfer.evidence.successful)
        and np.all(np.isfinite(states))
        and minimum >= -balance_tolerance
        and inventory_error <= balance_tolerance
        and balance_error <= balance_tolerance
        and interface_balance <= balance_tolerance
        and transfer_error <= balance_tolerance
        and physical_error <= target_error
    )
    return {
        **geometry,
        "accepted_steps": np.asarray(accepted).tolist(),
        "independent_inventory_initial": float(first @ mass),
        "independent_inventory_final": float(independent_inventory[-1]),
        "inventory_error": inventory_error,
        "local_balance_error": balance_error,
        "interface_inventory_balance_error": interface_balance,
        "material_interface_flux_samples": material_flux.tolist(),
        "bulk_network_exchange_samples": exchange_rates.tolist(),
        "compartment_ids": ("first", "second"),
        "compartment_inventory_final": [
            float(states[-1, :count][region_mask] @ mass[:count][region_mask]),
            float(states[-1, :count][~region_mask] @ mass[:count][~region_mask]),
        ],
        "network_inventory_final": float(
            states[-1, count : count + 2] @ mass[count : count + 2]
        ),
        "reservoir_inventory_final": float(states[-1, -1] * mass[-1]),
        "transfer_error": transfer_error,
        "minimum_concentration": minimum,
        "physical_error": physical_error,
        "physical_reference": "host symmetric mass-scaled eigensolution of independently assembled closed transport",
        "successful": successful,
    }, (mass, matrix, eigenvalues, eigenvectors)


def _transport_loop(runtime: Any, state: Any, *, steps: int, dt: float) -> Any:
    def advance(current: Any, unused: Any) -> Any:
        del unused
        step = runtime.step_backward_euler(current, dt)
        return step.state, (step.state, step.accepted)

    return jax.lax.scan(advance, state, None, length=steps)


def _network_source_plan(network: Any) -> phx.discretization.MetricNetworkPlan:
    vertex_ids = np.asarray(network.mesh.vertex_global_ids, dtype=np.int64)
    plan = phx.discretization.MetricNetworkPlan(
        network.mesh,
        network.coordinate_contract,
        np.asarray(network.areas),
        np.asarray(network.perimeters),
        vertex_ids[np.asarray(network.root_mask)],
        vertex_ids[np.asarray(network.tip_mask)],
    )
    if plan.prepare().network_id != network.network_id:
        raise RuntimeError(
            "The original network constructor facts do not reproduce its scientific identity."
        )
    return plan


def _transport_cold_worker(
    path: str, content_id: str, expected: str
) -> dict[str, object]:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        recertify_restored_meshing_source,
        write_meshing_source_closure,
    )

    records = read_meshing_source_closure(path, expected_content_id=content_id)
    if array_tree_fingerprint(records) != json.loads(expected):
        raise RuntimeError(
            "Cold transport lost original source, material, allfield, history or clock facts."
        )
    checkpoint = records["accepted_data"]["neurofluid_transport"]
    source, result = checkpoint.source, checkpoint.result
    source.validate_source_integrity()
    result.region_evidence.require_source(source.compartments)
    if result.certification is None:
        raise RuntimeError(
            "Cold transport requires the original complete source theorem."
        )
    renewed = recertify_restored_meshing_source(
        result.certification.request,
        result.mesh,
        result.geometry,
        result.audit,
        archive_limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    renewed.require_passed()
    network = checkpoint.network_plan.prepare()
    case = phx.applications.neurofluid.NeurofluidCase(
        checkpoint.case_id, source.labels, source.compartments, result, network
    )
    plan = phx.applications.neurofluid.NeurofluidTransportPlan(
        case, checkpoint.parameters
    )
    runtime = plan.prepare()
    if (case.case_revision, plan.plan_id, runtime.runtime_id) != checkpoint.physics_ids:
        raise RuntimeError(
            "Cold preparation changed the original source-qualified physics identity."
        )
    steps, dt = checkpoint.steps, checkpoint.dt
    execute = jax.jit(
        lambda owner, state: _transport_loop(owner, state, steps=steps, dt=dt)
    )
    continued = jax.block_until_ready(execute(runtime, checkpoint.final))
    if not bool(jnp.all(continued[1][1])):
        raise RuntimeError(
            "The cold continued native PDE rejected an original physical step."
        )
    hot = checkpoint.continued
    error = max(
        float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
        for first, second in zip(
            jax.tree.leaves(continued), jax.tree.leaves(hot), strict=True
        )
        if np.asarray(first).dtype.kind != "b"
    )
    if error > checkpoint.balance_tolerance:
        raise RuntimeError(
            "Cold continued native physics differs from the actual hot accepted fields."
        )
    restored, _ = _transport_invariants(
        result,
        runtime,
        checkpoint.initial,
        checkpoint.history,
        checkpoint.accepted,
        size=checkpoint.size,
        steps=steps,
        dt=dt,
        target_error=checkpoint.target_error,
        balance_tolerance=checkpoint.balance_tolerance,
    )
    if not restored["successful"]:
        raise RuntimeError(
            "Cold material and full accepted history failed the original independent physical gates."
        )
    with TemporaryDirectory(prefix="phydrax-neurofluid-cold-rearchive-") as directory:
        replay = write_meshing_source_closure(Path(directory) / "replayed.zip", records)
        if replay.content_id != content_id:
            raise RuntimeError(
                "Cold source/allfield/history rearchive changed the canonical scientific closure."
            )
    return {
        "passed": True,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "result_id": result.result_id,
        "source_report_id": renewed.report_id,
        "network_id": network.network_id,
        "parameter_id": checkpoint.parameters.parameter_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "allfield_history_bitwise_preserved": True,
        "continued_max_error": error,
        "accepted_step_cursor": checkpoint.step_cursor,
        "accepted_time": checkpoint.time,
        "continued_step_cursor": checkpoint.step_cursor + steps,
        "continued_time": checkpoint.time + steps * dt,
        "prepared_runtime_serialized": False,
        "default_max_members": DEFAULT_ARRAY_ARCHIVE_LIMITS.max_members,
        "default_max_nesting": DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_nesting,
        "default_max_rank": DEFAULT_ARRAY_ARCHIVE_LIMITS.max_array_rank,
    }


def _transport_lifecycle(
    source: Any,
    request: Any,
    options: Any,
    result: Any,
    case: Any,
    network: Any,
    parameters: Any,
    runtime: Any,
    initial: Any,
    first: Any,
    execute: Any,
    loop: Any,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    size: int,
    steps: int,
    dt: float,
    target_error: float,
    balance_tolerance: float,
    phase: Callable[[str], None],
    context: dict[str, object],
) -> dict[str, object]:
    from phydrax.lifecycle._meshing_sources import write_meshing_source_closure

    final, (history, accepted) = first
    mass, _, eigenvalues, eigenvectors = reference
    root_mass = np.sqrt(mass)
    tangent = runtime.unflatten(
        jnp.sin(jnp.arange(sum(runtime.sizes), dtype=jnp.float64) + 1.0)
    )
    phase("fixed-epoch-native-transport-AD")

    def derivative(owner: Any, state: Any, direction: Any) -> Any:
        return jax.jvp(lambda value: loop(owner, value)[0], (state,), (direction,))

    ad = jax.jit(derivative)
    ad_executable, ad_compilation = measure_lower_and_compile(
        lambda: ad.lower(runtime, initial, tangent), lambda lowered: lowered.compile()
    )
    (ad_primal, ad_value), ad_seconds = measure_synchronized(
        lambda: ad_executable(runtime, initial, tangent)
    )
    tangent_flat = np.asarray(runtime.flatten(tangent))
    expected = (
        eigenvectors
        @ (
            (1.0 + dt * eigenvalues) ** (-steps)
            * (eigenvectors.T @ (root_mass * tangent_flat))
        )
    ) / root_mass
    actual = np.asarray(runtime.flatten(ad_value))
    error = float(
        np.linalg.norm(root_mass * (actual - expected))
        / np.linalg.norm(root_mass * expected)
    )
    inventory_defect = float(abs(mass @ (actual - tangent_flat)))
    primal_defect = float(
        np.max(
            np.abs(
                np.asarray(runtime.flatten(ad_primal))
                - np.asarray(runtime.flatten(final))
            )
        )
    )
    if (
        not np.isfinite(error)
        or error > target_error
        or inventory_defect > balance_tolerance
        or primal_defect > balance_tolerance
    ):
        raise RuntimeError(
            "Actual native PDE AD failed the unchanged independent physical/inventory gates."
        )
    phase("continued-hot-native-transport")
    continued, continuation_seconds = measure_synchronized(
        lambda: execute(runtime, final)
    )
    if not bool(jnp.all(continued[1][1])):
        raise RuntimeError(
            "The continued original native transport rejected a physical step."
        )
    final_flat = np.asarray(runtime.flatten(final))
    expected_continued = (
        eigenvectors
        @ (
            (1.0 + dt * eigenvalues) ** (-steps)
            * (eigenvectors.T @ (root_mass * final_flat))
        )
    ) / root_mass
    continued_flat = np.asarray(runtime.flatten(continued[0]))
    continued_error = float(
        np.linalg.norm(root_mass * (continued_flat - expected_continued))
        / np.linalg.norm(root_mass * expected_continued)
    )
    continued_inventory_defect = float(abs(mass @ (continued_flat - final_flat)))
    if (
        not np.isfinite(continued_error)
        or continued_error > target_error
        or continued_inventory_defect > balance_tolerance
    ):
        raise RuntimeError(
            "Continued native PDE failed the original independent physical/inventory gates."
        )
    phase("allfield-history-native-resource-rollback")
    before = array_tree_fingerprint((initial, history, accepted, final, parameters))
    negative_policy = phx.linalg.LinearSolvePolicy(
        resources=phx.linalg.SolveResourcePolicy(
            factorization_bytes=0,
            workspace_bytes=0,
            krylov_basis_bytes=0,
            preconditioner_bytes=0,
            recycling_state_bytes=0,
        )
    )
    rejected = jax.jit(
        lambda state: runtime.step_backward_euler(state, dt, policy=negative_policy)
    )(final)
    rejected = jax.block_until_ready(rejected)
    if bool(rejected.accepted):
        raise RuntimeError(
            "A native PDE step with no solver storage was incorrectly accepted."
        )
    if array_tree_fingerprint(
        (initial, history, accepted, final, parameters)
    ) != before or any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for old, new in zip(
            jax.tree.leaves(final), jax.tree.leaves(rejected.state), strict=True
        )
    ):
        raise RuntimeError(
            "Native refusal failed atomic preservation of all accepted fields/material/history."
        )
    phase("default-source-allfield-material-history-checkpoint")
    checkpoint = phx.applications.neurofluid.NeurofluidTransportCheckpoint(
        source=source,
        result=result,
        network_plan=_network_source_plan(network),
        parameters=parameters,
        initial=initial,
        history=history,
        accepted=accepted,
        final=final,
        continued=continued,
        ad_tangent=tangent,
        ad_value=ad_value,
        case_id=case.case_id,
        physics_ids=(
            case.case_revision,
            phx.applications.neurofluid.NeurofluidTransportPlan(case, parameters).plan_id,
            runtime.runtime_id,
        ),
        size=size,
        steps=steps,
        dt=dt,
        step_cursor=steps,
        time=steps * dt,
        target_error=target_error,
        balance_tolerance=balance_tolerance,
    )
    records = {
        "certification_inputs": result.certification.request,
        "report": result.certification,
        "associations": result.associations,
        "generation_part": phx.meshing.MeshPart(
            "image", result, coordinate_contract=result.coordinate_contract
        ),
        "generation_source": source,
        "generation_specification": request,
        "generation_options": options,
        "accepted_target": result,
        "accepted_data": {"neurofluid_transport": checkpoint},
    }
    with TemporaryDirectory(prefix="phydrax-neurofluid-source-state-") as directory:
        receipt = write_meshing_source_closure(Path(directory) / "state.zip", records)
        expected_identity = json.dumps(array_tree_fingerprint(records), sort_keys=True)
        phase("fresh-default-cold-allfield-history-continued-native-PDE")
        script = (
            "import json,sys; from tools.neurofluid_qualification import _transport_cold_worker; "
            "print(json.dumps(_transport_cold_worker(sys.argv[1],sys.argv[2],sys.argv[3]),sort_keys=True))"
        )
        started = perf_counter()
        cold = subprocess.run(
            (
                sys.executable,
                "-c",
                script,
                str(receipt.path),
                receipt.content_id,
                expected_identity,
            ),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        context["cold_process"] = {
            "returncode": cold.returncode,
            "stdout": cold.stdout,
            "stderr": cold.stderr,
        }
        if cold.returncode != 0:
            raise RuntimeError(
                "Fresh original source/allfield/history continuation failed: "
                + cold.stderr
            )
        cold_evidence = json.loads(cold.stdout)
        cold_evidence.update(
            content_id=receipt.content_id, elapsed_seconds=perf_counter() - started
        )
    return {
        "passed": True,
        "field_names": ("bulk", "network", "reservoirs"),
        "material_parameter_id": parameters.parameter_id,
        "accepted_history_steps": steps,
        "AD": {
            "relative_discrete_physical_error": error,
            "inventory_defect": inventory_defect,
            "primal_defect": primal_defect,
            "lowering_seconds": ad_compilation.lowering_seconds,
            "compilation_seconds": ad_compilation.compilation_seconds,
            "execution_seconds": ad_seconds,
        },
        "hot_continuation_seconds": continuation_seconds,
        "hot_continuation_relative_physical_error": continued_error,
        "hot_continuation_inventory_defect": continued_inventory_defect,
        "atomic_rollback": {
            "passed": True,
            "negative_linear_status": int(rejected.linear_status),
            "positive_source_physics_controls_unchanged": True,
        },
        "cold": cold_evidence,
    }


def _execute_native_transport(
    *,
    size: int = 2,
    repeats: int = 3,
    steps: int = 5,
    dt: float = 0.01,
    target_error: float = 0.05,
    balance_tolerance: float = 1.0e-9,
    maximum_cells: int = 4096,
    maximum_vertices: int = 4096,
    oracle_capacity: int = 2048,
    phase: Callable[[str], None],
    context: dict[str, object],
) -> dict[str, object]:
    """Generate, re-admit, transfer, solve, and independently certify one case."""
    campaign_started = perf_counter()
    started = perf_counter()
    native_phases: list[dict[str, object]] = []
    context["native_phase_measurements"] = native_phases

    def record_native_phase(measurement: NativeMeshingPhaseMeasurement) -> None:
        native_phases.append(asdict(measurement))

    source, network = _native_source(size)
    source_seconds = perf_counter() - started
    context["source"] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_complex_id": source.compartments.complex_id,
        "coordinate_contract_id": source.coordinate_contract.spatial_id,
    }
    phase("request-preparation")
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{source.source_id}:boundary",
        np.asarray((0,), dtype=np.int64),
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(scope, 1.0),),
        limits=phx.meshing.MeshingLimits(
            maximum_cells=maximum_cells,
            maximum_vertices=maximum_vertices,
        ),
    )
    phase("geometry-preparation")
    started = perf_counter()
    options = phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source,
        request,
        coordinate_contract=source.coordinate_contract,
        record_phase=record_native_phase,
    )
    geometry_preparation = perf_counter() - started
    phase("generation-publication")
    result, generation = measure_synchronized(
        lambda: plan.execute(record_phase=record_native_phase)
    )
    execution = result.execution_evidence
    if execution is None:
        raise RuntimeError(
            "Original native transport requires its actual ended generation resource receipt."
        )
    execution.require_valid()
    if int(execution.status) != int(MeshcoreStatus.OK):
        raise RuntimeError(
            "Original generation did not end with a passing native resource status."
        )
    native_execution = execution.to_record()
    context["native_execution"] = native_execution
    context["identities"] = {
        "mesh_id": result.mesh.mesh_id,
        "result_id": result.result_id,
        "request_id": request.specification_id,
        "provider": result.provider.name,
    }
    phase("oracle-capacity-admission")
    count = result.mesh.entity_set(3).count
    if count + 3 > oracle_capacity:
        raise ValueError(
            "Generated transport state exceeds the declared independent dense oracle capacity."
        )
    phase("case-readmission")
    started = perf_counter()
    case = phx.applications.neurofluid.NeurofluidCase(
        "native-generated-transport", source.labels, source.compartments, result, network
    )
    readmitted = phx.applications.neurofluid.NeurofluidCase(
        case.case_id, case.segmentation, case.compartments, case.bulk_mesh, case.network
    )
    admission = perf_counter() - started
    phase("transport-transfer-preparation")
    parameters = _transport_parameters(result)
    transport_plan = phx.applications.neurofluid.NeurofluidTransportPlan(
        readmitted, parameters
    )
    runtime, preparation = measure_synchronized(transport_plan.prepare)
    evidence = result.region_evidence
    if evidence is None:
        raise RuntimeError(
            "Generated compartment transport requires exact region evidence."
        )
    initial = phx.equations.MixedDimensionalTransportState(
        jnp.asarray(
            [3.0 if region == "first" else 2.0 for region in evidence.cell_region_ids],
            dtype=jnp.float64,
        ),
        jnp.asarray((1.0, 1.5), dtype=jnp.float64),
        jnp.asarray((0.5,), dtype=jnp.float64),
    )

    def loop(current_runtime: Any, state: Any) -> Any:
        return _transport_loop(current_runtime, state, steps=steps, dt=dt)

    execute = jax.jit(loop)

    def lower() -> Any:
        phase("lowering")
        return execute.lower(runtime, initial)

    def compile_lowered(lowered: Any) -> Any:
        phase("compilation")
        return lowered.compile()

    executable, compilation = measure_lower_and_compile(lower, compile_lowered)
    phase("first-execution")
    first, first_seconds = measure_synchronized(lambda: executable(runtime, initial))
    phase("warmed-execution")
    _, warmed = measure_repeated(
        lambda: executable(runtime, initial), warmup=0, repeats=repeats
    )
    phase("compiler-evidence")
    compiler = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    phase("independent-certification")
    started = perf_counter()
    _, (history, accepted) = first
    invariants, reference = _transport_invariants(
        result,
        runtime,
        initial,
        history,
        accepted,
        size=size,
        steps=steps,
        dt=dt,
        target_error=target_error,
        balance_tolerance=balance_tolerance,
    )
    certification = perf_counter() - started
    lifecycle = _transport_lifecycle(
        source,
        request,
        options,
        result,
        readmitted,
        network,
        parameters,
        runtime,
        initial,
        first,
        executable,
        loop,
        reference,
        size=size,
        steps=steps,
        dt=dt,
        target_error=target_error,
        balance_tolerance=balance_tolerance,
        phase=phase,
        context=context,
    )
    phase("memory-evidence")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    device_memory = []
    for device in jax.local_devices():
        stats = device.memory_stats()
        device_memory.append(
            {
                "device": str(device),
                "peak_bytes": None if stats is None else stats.get("peak_bytes_in_use"),
                "unavailable_reason": (
                    "Backend does not expose device allocation peak."
                    if stats is None or "peak_bytes_in_use" not in stats
                    else None
                ),
            }
        )
    record = {
        "scenario": "native-generated-transport",
        "controls": {
            "size": size,
            "maximum_cells": maximum_cells,
            "maximum_vertices": maximum_vertices,
            "oracle_capacity": oracle_capacity,
            "steps": steps,
            "dt_seconds": dt,
            "repeats": repeats,
        },
        "targets": {
            "physical_error": target_error,
            "balance_tolerance": balance_tolerance,
        },
        "source": {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "source_complex_id": source.compartments.complex_id,
            "coordinate_contract_id": source.coordinate_contract.spatial_id,
            "interpretation": "occupied-voxel-cells",
            "origin": "synthetic-labeled-image",
        },
        "units": {
            "length": "millimeter",
            "time": "second",
            "concentration": "millimolar",
            "inventory": "millimolar*millimeter^3",
        },
        "identities": {
            "mesh_id": result.mesh.mesh_id,
            "result_id": result.result_id,
            "request_id": request.specification_id,
            "region_evidence_id": evidence.evidence_id,
            "coverage_id": evidence.coverage_id,
            "case_revision": readmitted.case_revision,
            "network_id": network.network_id,
            "transport_plan_id": transport_plan.plan_id,
            "runtime_id": runtime.runtime_id,
            "transfer_id": runtime.exchange.transfer.transfer_id,
            "meshcore": meshcore_identity(),
            "provider": result.provider.name,
        },
        "timing": {
            "source_preparation_seconds": source_seconds,
            "geometry_preparation_seconds": geometry_preparation,
            "generation_publication_seconds": generation,
            "readmission_seconds": admission,
            "transport_transfer_preparation_seconds": preparation,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_execution_seconds": first_seconds,
            "warmed": warmed.to_seconds_dict(),
            "independent_certification_seconds": certification,
            "end_to_end_seconds": perf_counter() - campaign_started,
            "explicit_primary_transport_compilations": 1,
        },
        "memory": {
            "compiler": asdict(compiler),
            "logical_retained_bytes": logical_array_bytes(
                (source, plan, result, case, runtime, initial)
            ),
            "logical_output_bytes": logical_array_bytes(first),
            "process_peak_resident_bytes": peak
            * (1 if sys.platform == "darwin" else 1024),
            "process_peak_scope": "process-lifetime high-water mark, includes compilation and independent certification",
            "device": device_memory,
        },
        "invariants": invariants,
        "lifecycle": lifecycle,
        "native_execution": native_execution,
        "native_execution_scope": "actual ended generation operation; not compiler/device allocation measurement",
        "original_native_limits": asdict(request.limits),
        "successful": bool(invariants["successful"] and lifecycle["passed"]),
        "comparison": {"status": "not-requested"},
    }
    record["native_phase_measurements"] = native_phases
    phase("runtime-build-identity")
    root = Path(__file__).resolve().parents[1]
    record["native_build_identity"] = meshcore_runtime_identity()
    record["loaded_python_source_digest"] = python_source_revision(
        Path(phx.__file__).resolve().parent
    )
    record["build_identity"] = capture_benchmark_identity(
        root,
        Path(__file__).resolve(),
        (*record.keys(), "build_identity", "runtime_identity"),
    ).to_dict()
    environment = capture_environment()
    record["runtime_identity"] = {
        "fingerprint": environment.fingerprint,
        "backend": environment.backend,
        "x64_enabled": environment.x64_enabled,
        "platform": environment.platform,
        "machine": environment.machine,
        "devices": [device.to_dict() for device in environment.devices],
        "package_fingerprint": environment.package_fingerprint,
    }
    return record


def native_transport(
    *,
    size: int = 2,
    repeats: int = 3,
    steps: int = 5,
    dt: float = 0.01,
    target_error: float = 0.05,
    balance_tolerance: float = 1.0e-9,
    maximum_cells: int = 4096,
    maximum_vertices: int = 4096,
    oracle_capacity: int = 2048,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Preserve failed native attempts and exact failure phases without fallback."""
    started = perf_counter()
    context: dict[str, object] = {}
    current_phase = "source-preparation"

    def phase(name: str) -> None:
        nonlocal current_phase
        current_phase = name
        if phase_callback is not None:
            phase_callback(name)

    try:
        phase(current_phase)
        return _execute_native_transport(
            size=size,
            repeats=repeats,
            steps=steps,
            dt=dt,
            target_error=target_error,
            balance_tolerance=balance_tolerance,
            maximum_cells=maximum_cells,
            maximum_vertices=maximum_vertices,
            oracle_capacity=oracle_capacity,
            phase=phase,
            context=context,
        )
    except Exception as error:
        failure: dict[str, object] = {
            "phase": current_phase,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(error),
        }
        if isinstance(error, phx.meshing.MeshingFailure):
            evidence = error.evidence
            failure["meshing_evidence"] = {
                "category": evidence.category.value,
                "message": evidence.message,
                "provider_code": evidence.provider_code,
                "stage": evidence.stage,
                "entity_ids": evidence.entity_ids,
                "locations": evidence.locations,
                "requested": evidence.requested,
                "achieved": evidence.achieved,
                "checkpoint_id": evidence.checkpoint_id,
                "evidence_id": evidence.evidence_id,
            }
        return {
            "scenario": "native-generated-transport",
            "successful": False,
            "status": "failed",
            "failure": failure,
            **context,
            "controls": {
                "size": size,
                "repeats": repeats,
                "steps": steps,
                "dt_seconds": dt,
                "maximum_cells": maximum_cells,
                "maximum_vertices": maximum_vertices,
                "oracle_capacity": oracle_capacity,
            },
            "targets": {
                "physical_error": target_error,
                "balance_tolerance": balance_tolerance,
            },
            "timing": {"elapsed_before_failure_seconds": perf_counter() - started},
            "comparison": {"status": "not-requested"},
        }


def add_native_arguments(
    parser: argparse.ArgumentParser, *, default_scenario: str
) -> None:
    parser.add_argument(
        "--scenario",
        choices=("synthetic", "native-generated-transport"),
        default=default_scenario,
    )
    parser.add_argument(
        "--size", type=int, default=2, help="Occupied-voxel cube edge count."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Transport step width in seconds."
    )
    parser.add_argument(
        "--target-error",
        type=float,
        default=0.05,
        help="Relative mass-weighted error against independent continuous-time transport.",
    )
    parser.add_argument("--balance-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--maximum-cells", type=int, default=4096)
    parser.add_argument("--maximum-vertices", type=int, default=4096)
    parser.add_argument(
        "--oracle-capacity",
        type=int,
        default=2048,
        help="Maximum state width for the independent dense scientific oracle.",
    )


def validate_native_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if (
        args.size < 2
        or min(
            args.repeats,
            args.steps,
            args.maximum_cells,
            args.maximum_vertices,
            args.oracle_capacity,
        )
        < 1
    ):
        parser.error(
            "size must exceed one; repeats, steps, and capacities must be positive"
        )
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (args.dt, args.target_error, args.balance_tolerance)
    ):
        parser.error(
            "dt, target-error, and balance-tolerance must be finite and positive"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_arguments(parser, default_scenario="synthetic")
    args = parser.parse_args()
    validate_native_arguments(parser, args)
    if args.scenario == "synthetic":
        record = qualify()
    else:
        record = native_transport(
            size=args.size,
            repeats=args.repeats,
            steps=args.steps,
            dt=args.dt,
            target_error=args.target_error,
            balance_tolerance=args.balance_tolerance,
            maximum_cells=args.maximum_cells,
            maximum_vertices=args.maximum_vertices,
            oracle_capacity=args.oracle_capacity,
        )
    print(json.dumps(record, indent=2))
    if not record["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

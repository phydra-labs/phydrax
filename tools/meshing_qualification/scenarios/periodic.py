# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples.native_surface_meshing import zero_mean_constraint
from phydrax import (
    ArrayArchiveLimits,
    canonical_fingerprint,
    model_array_bytes,
    NonTrainableState,
    StrictModule,
)
from phydrax.lifecycle import (
    meshing_source_owner_reconstruction_scope,
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    recertify_restored_meshing_source,
    validate_meshing_accepted_epochs,
    write_meshing_source_closure,
)
from phydrax.meshing import NativeMeshingPhaseMeasurement
from phydrax.meshing.qualification import (
    periodic_field_templates,
    prepare_periodic_accepted_epoch,
)
from tools.meshing_qualification._controls import (
    _periodic_qualification_requirements,
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import _compiled_solver_stages
from tools.meshing_qualification._records import (
    _phase_recordings,
    _quality_summary,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    run_cold_worker,
    StageLedger,
)
from tools.meshing_qualification.scenarios.bisection import (
    _bisection_evidence,
    _bisection_lineage,
)


def _periodic_qualification_fixture(
    args: QualificationControls,
) -> tuple[
    phx.meshing.NativePeriodicSource,
    phx.meshing.SurfaceMeshingSpec,
    phx.meshing.NativeMeshingOptions,
    dict[str, Any],
]:
    """Freeze a skew torus source, including represented materials and feature orbits."""

    M, D = phx.meshing, phx.discretization
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    cell = D.PeriodicCell(vectors)
    count = args.resolution
    lattice = np.stack(
        np.meshgrid(
            np.arange(count, dtype=np.int64),
            np.arange(count, dtype=np.int64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 2))
    points = (lattice / count) @ vectors
    payload: dict[str, Any] = {
        "case": args.periodic_case,
        "vectors": vectors.tolist(),
        "fractional_seed_numerators": lattice.tolist(),
        "denominator": count,
    }
    regions = np.asarray((0,), dtype=np.int64)
    features: tuple[phx.meshing.ProtectedFeature, ...] = ()
    controls: tuple[phx.meshing.RegionControl, ...] = ()
    if args.periodic_case == "skew-lattice":
        domain = M.PeriodicPointOrbits(points, cell)
        source = M.NativePeriodicSource(
            domain, "qualification-skew-torus", canonical_fingerprint(payload)
        )
    elif args.periodic_case == "skew-material-feature":
        corners: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        assignments: list[int] = []
        for i in range(count):
            for j in range(count):
                for triangle in (
                    ((i, j), (i + 1, j), (i + 1, j + 1)),
                    ((i, j), (i + 1, j + 1), (i, j + 1)),
                ):
                    corners.append(triangle)
                    assignments.append(0 if i < count // 2 else 1)
        integer_corners = np.asarray(corners, dtype=np.int64)
        simplices = (integer_corners[..., 0] % count) * count + integer_corners[
            ..., 1
        ] % count
        domain = M.publish_periodic_simplices(
            points, simplices, integer_corners // count, cell
        )
        payload.update(
            {
                "simplices": simplices.tolist(),
                "shifts": (integer_corners // count).tolist(),
                "cell_regions": assignments,
            }
        )
        source = M.NativePeriodicSource(
            domain,
            "qualification-skew-material-torus",
            canonical_fingerprint(payload),
            cell_regions=np.asarray(assignments, dtype=np.int64),
        )
        regions = np.asarray((0, 1), dtype=np.int64)
        connectivity = domain.connectivity
        periodic = domain.periodic_topology
        if not isinstance(connectivity, D.PolygonalConnectivity) or periodic is None:
            raise qualification_failure(
                "_periodic_qualification_fixture",
                (
                    "The represented periodic fixture lacks its triangle connectivity or quotient."
                ),
                {"D": D, "connectivity": connectivity, "periodic": periodic},
            )
        edges = np.asarray(connectivity.edges)
        fractional = np.asarray(domain.coordinates) @ np.linalg.solve(
            vectors, np.eye(2, dtype=np.float64)
        )
        x = fractional[edges, 0]
        interface = (x[:, 0] == x[:, 1]) & (
            (x[:, 0] % 1.0 == 0.0) | (x[:, 0] % 1.0 == 0.5)
        )
        quotient_edges = np.unique(np.asarray(periodic.orbits(1)[0])[interface])
        features = (
            M.ProtectedFeature(
                M.MeshingScope(
                    source.source_id,
                    source.source_revision,
                    M.MeshingEntityKind.GEOMETRY,
                    1,
                    "represented-material-interface-orbits",
                    quotient_edges,
                ),
                M.FeatureKind.MATERIAL_INTERFACE,
            ),
        )
        controls = tuple(
            M.RegionControl(
                M.MeshingScope(
                    source.source_id,
                    source.source_revision,
                    M.MeshingEntityKind.GEOMETRY,
                    2,
                    "represented-periodic-regions",
                    np.asarray((region,), dtype=np.int64),
                ),
                f"material-{region}",
                f"material-{region}",
                M.RegionRole.USER,
            )
            for region in regions
        )
    else:
        raise ValueError(
            f"periodic.case must name a supported qualification case; got {args.periodic_case!r}"
        )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "represented-periodic-regions",
        regions,
    )
    capacity = args.capacity
    limits = capacity_meshing_limits(
        capacity,
        args.timeout,
        cells=8,
        edges=8,
        faces=8,
        connectivity_entries=64,
        data_bytes=1024,
        scratch_bytes=4096,
        work_units=2048,
        cavity_cells=1,
        geometry_queries=64,
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            M.UniformSizeControl(
                scope,
                1.0 / count,
                maximum_size=1.5 / count,
                strength=M.SizeControlStrength.HARD,
            ),
        ),
        protected_features=features,
        region_controls=controls,
        limits=limits,
    )
    return (
        source,
        specification,
        M.NativeMeshingOptions("periodic_delaunay"),
        {
            **payload,
            "source_revision": source.source_revision,
            "source_binding_id": source.binding_id,
            "provenance": "PHYDRA original dyadic-skew translational torus fixture",
            "expected_admission": "positive",
            "expected_measure": 1.0,
            "expected_material_measures": {"material-0": 0.5, "material-1": 0.5}
            if controls
            else {"0": 1.0},
            "expected_interface_length": 2.0 * float(np.linalg.norm(vectors[1]))
            if features
            else None,
            "protected_feature_ids": [feature.feature_id for feature in features],
            "resource_limits": {
                "maximum_vertices": capacity,
                "maximum_cells": 8 * capacity,
                "maximum_edges": 8 * capacity,
                "maximum_faces": 8 * capacity,
                "maximum_connectivity_entries": 64 * capacity,
                "maximum_data_bytes": 1024 * capacity,
                "maximum_scratch_bytes": 4096 * capacity,
                "maximum_work_units": 2048 * capacity,
                "maximum_cavity_cells": capacity,
                "maximum_geometry_queries": 64 * capacity,
                "maximum_wall_seconds": args.timeout,
            },
            "frozen_size_control": {
                "target_size": 1.0 / count,
                "maximum_size": 1.5 / count,
                "strength": "hard",
            },
        },
    )


def _periodic_qualification_acceptance(
    result: phx.meshing.CellMeshingResult,
    case: dict[str, Any],
) -> dict[str, object]:
    """Check quotient closure, signed chain maps, material inventory, and feature coverage."""

    mesh = result.mesh
    periodic = mesh.periodic_topology
    if periodic is None or not result.audit.passed or not result.compliance.passed:
        raise qualification_failure(
            "_periodic_qualification_acceptance",
            ("Periodic provider/transaction did not publish an accepted quotient."),
            {"periodic": periodic, "result": result},
        )
    if result.certification is None or not result.certification.passed:
        raise qualification_failure(
            "_periodic_qualification_acceptance",
            ("Periodic publication lacks accepted independent certification."),
            {"result": result},
        )
    periodic.require_lift(mesh.coordinates)
    declared = result.geometry if result.geometry.exact_source is not None else None
    orbit_measures = phx.discretization.periodic_orbit_measures(mesh, geometry=declared)
    measure = float(orbit_measures.total_measure)
    defects = []
    for degree in range(1, mesh.topological_dimension + 1):
        values = (
            np.arange(periodic.quotient.entities(degree - 1).count, dtype=np.float64)
            + 0.375
        )
        first = (
            mesh.topology.incidences[degree - 1]
            .exterior_derivative()
            .mv(periodic.identification(degree - 1).mv(values))
        )
        second = periodic.identification(degree).mv(
            periodic.quotient.incidences[degree - 1].exterior_derivative().mv(values)
        )
        defects.append(float(np.max(np.abs(np.asarray(first - second)), initial=0.0)))
    boundary = int(
        np.count_nonzero(
            np.asarray(periodic.quotient.entities(1).subset("boundary").mask)
        )
    )
    if (
        periodic.euler_characteristic != 0
        or boundary
        or abs(measure - 1.0) > 1.0e-10
        or max(defects) > 1.0e-10
    ):
        raise qualification_failure(
            "_periodic_qualification_acceptance",
            (
                "Periodic quotient failed closure, measure, or signed incidence acceptance."
            ),
            {
                "boundary": boundary,
                "defects": defects,
                "measure": measure,
                "periodic": periodic,
            },
        )
    measures = np.asarray(orbit_measures.orbit_measures)
    cells = np.asarray(mesh.entity_set(mesh.topological_dimension).entity_ids)
    material_measures: dict[str, float] = {}
    for zone in result.zones:
        if zone.material_id is None:
            raise qualification_failure(
                "_periodic_qualification_acceptance",
                ("An accepted periodic region lacks its declared material identity."),
                {"zone": zone},
            )
        material_measures[zone.material_id] = float(
            np.sum(measures[np.isin(cells, np.asarray(zone.scope.entity_ids))])
        )
    expected = case["expected_material_measures"]
    if set(material_measures) != set(expected) or any(
        abs(material_measures[name] - expected[name]) > 1.0e-10 for name in expected
    ):
        raise qualification_failure(
            "_periodic_qualification_acceptance",
            ("Periodic material identities or one-orbit inventories changed."),
            {"expected": expected, "material_measures": material_measures},
        )
    feature_lengths: dict[str, float] = {}
    leaders = np.asarray(periodic.orbit_representatives(1))
    edge_ids = np.asarray(mesh.topology.entities(1).entity_ids)
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise qualification_failure(
            "_periodic_qualification_acceptance",
            ("Periodic triangle acceptance requires polygonal connectivity."),
            {"connectivity": connectivity},
        )
    edge_points = np.asarray(mesh.coordinates)[np.asarray(connectivity.edges)]
    lengths = np.linalg.norm(edge_points[:, 1] - edge_points[:, 0], axis=1)
    orbit = np.asarray(periodic.orbits(1)[0])
    for feature_id in case["protected_feature_ids"]:
        matching = [
            label
            for label in result.labels
            if label.name == f"periodic-feature:{feature_id}"
        ]
        if len(matching) != 1:
            raise qualification_failure(
                "_periodic_qualification_acceptance",
                ("A declared periodic material interface label was lost."),
                {"matching": matching},
            )
        mask = np.isin(edge_ids, np.asarray(matching[0].scope.entity_ids))
        if not np.array_equal(mask, np.isin(orbit, orbit[mask])):
            raise qualification_failure(
                "_periodic_qualification_acceptance",
                ("Protected interface membership is not orbit closed."),
                {"mask": mask, "orbit": orbit},
            )
        feature_lengths[feature_id] = float(np.sum(lengths[leaders[mask[leaders]]]))
        if abs(feature_lengths[feature_id] - case["expected_interface_length"]) > 1.0e-10:
            raise qualification_failure(
                "_periodic_qualification_acceptance",
                ("Protected interface one-orbit length changed."),
                {
                    "case": case,
                    "feature_id": feature_id,
                    "feature_lengths": feature_lengths,
                },
            )
    return {
        "passed": True,
        "quotient_id": periodic.periodic_topology_id,
        "quotient_counts": [entities.count for entities in periodic.quotient.entity_sets],
        "euler_characteristic": periodic.euler_characteristic,
        "boundary_facets": boundary,
        "measure": measure,
        "signed_chain_map_defects": defects,
        "material_measures": material_measures,
        "feature_lengths": feature_lengths,
        "quality": _quality_summary(result.quality),
        "certification_id": result.certification.report_id,
        "compliance_id": result.compliance.report_id,
    }


def _periodic_qualification_exact(
    points: jax.Array | np.ndarray,
    dual: jax.Array | np.ndarray,
) -> jax.Array:
    fractional = points @ dual
    return jnp.cos(2.0 * jnp.pi * fractional[..., 0]) + 0.5 * jnp.sin(
        2.0 * jnp.pi * fractional[..., 1]
    )


class _PeriodicQualificationForcing(StrictModule, NonTrainableState):
    """One source-addressed callable; lattice coefficients remain dynamic leaves."""

    dual: jax.Array
    eigenvalues: jax.Array

    def __init__(self, dual: jax.Array, /) -> None:
        self.dual = dual
        self.eigenvalues = 4.0 * jnp.pi**2 * jnp.sum(dual * dual, axis=0)

    def __call__(self, points: jax.Array, context: object) -> jax.Array:
        del context
        fractional = points @ self.dual
        return self.eigenvalues[0] * jnp.cos(
            2.0 * jnp.pi * fractional[..., 0]
        ) + 0.5 * self.eigenvalues[1] * jnp.sin(2.0 * jnp.pi * fractional[..., 1])


def _periodic_qualification_spaces(
    result: phx.meshing.CellMeshingResult,
    case: dict[str, Any],
) -> tuple[
    dict[str, phx.discretization.FiniteElementDiscretization],
    phx.equations.CompiledFiniteElementProblem,
    np.ndarray,
    np.ndarray,
]:
    """Prepare separate H1/H(curl)/H(div) spaces and the mean-zero scalar PDE."""

    D = phx.discretization
    spaces = {
        field.name: D.FiniteElementPlan(
            result.mesh,
            field,
            coordinate_spec=result.geometry,
        ).prepare()
        for _, field in periodic_field_templates()
    }
    constraint, weights = zero_mean_constraint(spaces["u"])
    dual = jnp.asarray(
        np.linalg.solve(
            np.asarray(case["vectors"], dtype=np.float64), np.eye(2, dtype=np.float64)
        )
    )
    forcing = _PeriodicQualificationForcing(dual)

    form = phx.equations.FiniteElementForm(
        "qualification-skew-periodic-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    forcing,
                    coefficient_id=canonical_fingerprint(
                        {
                            "kind": "skew-torus-manufactured-forcing",
                            "vectors": case["vectors"],
                        }
                    ),
                ),
            ),
        ),
        # Unit diffusion is SPD on the connected mesh's positive-mass mean-zero subspace.
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, spaces["u"], constraint=constraint
    )
    kept = np.flatnonzero(np.arange(weights.size, dtype=np.int64) != np.argmax(weights))
    return spaces, problem, kept, weights


def _periodic_qualification_error(
    space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
    case: dict[str, Any],
) -> float:
    """Independent high-order physical quadrature of the periodic manufactured error."""
    rule = phx.integration.ReferenceTriangleRule(
        phx.integration.GaussLegendreRule(8)
    ).materialize()
    dual = np.linalg.solve(
        np.asarray(case["vectors"], dtype=np.float64), np.eye(2, dtype=np.float64)
    )
    squared_error = 0.0
    for block_index in range(len(space.mesh.blocks)):
        geometry = space.evaluate_block_geometry(
            "u", block_index, space.default_runtime.coordinates, rule.points, rule.weights
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[block_index])
        evaluated = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), np.asarray(values)[routes]
        )
        exact = np.asarray(
            _periodic_qualification_exact(np.asarray(geometry.physical_points), dual)
        )
        squared_error += float(
            np.sum(np.asarray(geometry.physical_weights) * (evaluated - exact) ** 2)
        )
    return float(np.sqrt(squared_error))


def _periodic_qualification_cell_errors(
    space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical squared-error integrals and measures in block-cell order."""
    rule = phx.integration.ReferenceTriangleRule(
        phx.integration.GaussLegendreRule(8)
    ).materialize()
    dual = np.linalg.solve(
        np.asarray(case["vectors"], dtype=np.float64), np.eye(2, dtype=np.float64)
    )
    squared_errors = []
    measures = []
    for block_index in range(len(space.mesh.blocks)):
        geometry = space.evaluate_block_geometry(
            "u", block_index, space.default_runtime.coordinates, rule.points, rule.weights
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[block_index])
        evaluated = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), np.asarray(values)[routes]
        )
        exact = np.asarray(
            _periodic_qualification_exact(np.asarray(geometry.physical_points), dual)
        )
        physical_weights = np.asarray(geometry.physical_weights)
        squared_errors.append(np.sum(physical_weights * (evaluated - exact) ** 2, axis=1))
        measures.append(np.sum(physical_weights, axis=1))
    return np.concatenate(squared_errors), np.concatenate(measures)


def _periodic_qualification_coarsening_marks(
    squared_errors: np.ndarray,
    measures: np.ndarray,
    parents: np.ndarray,
    cell_ids: np.ndarray,
    target_error: float,
) -> np.ndarray:
    """Mark siblings whose coarse physical error permits equidistributed L2 error.

    The retained Galerkin error is an a-posteriori marking indicator, not an
    acceptance certificate for a new solve. Conformity closes the marks in the
    periodic adaptation owner; the final solve and cold restore independently
    enforce the unchanged global physical error target.
    """
    eligible = (
        np.ones(squared_errors.size, dtype=np.bool_)
        if np.sum(squared_errors) <= target_error**2
        else squared_errors <= target_error**2 * measures / np.sum(measures)
    )
    return cell_ids[eligible[parents]]


def _periodic_qualification_initial_fields(
    spaces: dict[str, phx.discretization.FiniteElementDiscretization],
    case: dict[str, Any],
) -> dict[str, jax.Array]:
    """Declare periodic compatible moment fields once, retaining them after adaptation."""
    dual = jnp.asarray(
        np.linalg.solve(
            np.asarray(case["vectors"], dtype=np.float64), np.eye(2, dtype=np.float64)
        )
    )

    def vector(points: jax.Array, context: object) -> jax.Array:
        del context
        fractional = points @ dual
        return jnp.stack(
            (
                1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * fractional[..., 1]),
                -0.4 + 0.1 * jnp.sin(2.0 * jnp.pi * fractional[..., 0]),
            ),
            axis=-1,
        )

    def flux(points: jax.Array, context: object) -> jax.Array:
        del context
        fractional = points @ dual
        return jnp.stack(
            (
                0.3 + 0.15 * jnp.sin(2.0 * jnp.pi * fractional[..., 0]),
                0.8 + 0.2 * jnp.cos(2.0 * jnp.pi * fractional[..., 1]),
            ),
            axis=-1,
        )

    return {
        "vector": spaces["vector"].project("vector", vector),
        "flux": spaces["flux"].project("flux", flux),
    }


def _periodic_qualification_transfer(
    adaptation: phx.meshing.MeshAdaptationResult,
    source_spaces: dict[str, phx.discretization.FiniteElementDiscretization],
    target_spaces: dict[str, phx.discretization.FiniteElementDiscretization],
    state: dict[str, jax.Array],
    operation: str,
    transition: dict[str, Any],
    record: dict[str, Any],
    index: int,
    *,
    deadline: Deadline,
) -> dict[str, jax.Array]:
    """Apply certified quotient-sized field transfers to the accepted state."""

    _ledger = StageLedger(record, deadline=deadline)
    D = phx.discretization
    target_state: dict[str, jax.Array] = {}
    transfer_records: dict[str, Any] = {}
    transition["transfers"] = transfer_records
    for name in ("u", "vector", "flux"):
        source, target = source_spaces[name], target_spaces[name]

        if _ledger.run(
            f"field_transfer_prepare_{name}_transition_{index}",
            lambda: operation == "refine",
        ):
            transfer, measured = _ledger.run(
                record.get("active_phase", "prepare_nested_field_transfer"),
                lambda: _measured(
                    lambda: D.prepare_nested_field_transfer(
                        source,
                        target,
                        adaptation.parent_cells,
                        field_name=name,
                        parent_reference_vertices=adaptation.parent_reference_vertices,
                        source_geometry=adaptation.source.geometry,
                        target_geometry=adaptation.target.geometry,
                    )
                ),
            )
        elif operation == "coarsen":
            transfer, measured = _ledger.run(
                record.get("active_phase", "prepare_nested_field_transfer"),
                lambda: _measured(
                    lambda: D.prepare_nested_field_transfer(
                        source,
                        target,
                        None,
                        field_name=name,
                        source_geometry=adaptation.source.geometry,
                        target_geometry=adaptation.target.geometry,
                        coarsening_witnesses=adaptation.coarsening_witnesses,
                    )
                ),
            )
        else:
            raise ValueError(
                f"periodic.operation must be refine or coarsen; got {operation!r}"
            )
        record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
        evidence = transfer.evidence
        field_record: dict[str, Any] = {
            "transfer_id": transfer.transfer_id,
            "evidence_id": evidence.evidence_id,
            "passed": evidence.passed,
            "defects": dict(evidence.defects),
            "estimates": dict(evidence.estimates),
            "bounds": dict(evidence.bounds),
            "tolerance": evidence.tolerance,
            "semantics": transfer.transfer.semantics,
            "preserves_constants": transfer.transfer.preserves_constants,
            "conservative": transfer.transfer.conservative,
            "source_dofs": source.dof_maps[0].global_dof_count,
            "target_dofs": target.dof_maps[0].global_dof_count,
            "preparation": measured,
            "retained_bytes": model_array_bytes(transfer),
        }
        transfer_records[name] = field_record
        if not evidence.passed:
            raise qualification_failure(
                "_periodic_qualification_transfer",
                (f"Periodic {name} field transfer failed its scientific certificate."),
                {"evidence": evidence},
            )
        if (
            name in ("vector", "flux")
            and evidence.defect("commuting") > evidence.tolerance
        ):
            raise qualification_failure(
                "_periodic_qualification_transfer",
                (
                    f"Periodic {name} field transfer does not commute with its differential."
                ),
                {"evidence": evidence, "name": name},
            )

        values, measured = _ledger.run(
            f"field_transfer_execute_{name}_transition_{index}",
            lambda: _measured(
                lambda: jax.block_until_ready(transfer.transfer.apply(state[name]))
            ),
        )
        record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
        field_record["execution"] = measured
        if not np.all(np.isfinite(np.asarray(values))):
            raise qualification_failure(
                "_periodic_qualification_transfer",
                (
                    f"Periodic {name} transfer produced nonfinite accepted-state coefficients."
                ),
                {"values": values},
            )
        if name == "u":
            if (
                not transfer.transfer.conservative
                or transfer.source_measures is None
                or transfer.target_measures is None
            ):
                raise qualification_failure(
                    "_periodic_qualification_transfer",
                    (
                        "Periodic scalar continuation requires certified conservative coarsening/refinement."
                    ),
                    {"transfer": transfer},
                )
            before = float(np.asarray(transfer.source_measures) @ np.asarray(state[name]))
            after = float(np.asarray(transfer.target_measures) @ np.asarray(values))
            field_record["state_inventory"] = {
                "source": before,
                "target": after,
                "defect": after - before,
            }
            if abs(after - before) > 1.0e-10 * max(1.0, abs(before)):
                raise qualification_failure(
                    "_periodic_qualification_transfer",
                    ("Periodic scalar transfer changed the accepted-state inventory."),
                    {"after": after, "before": before},
                )
        target_state[name] = values
    return target_state


def _periodic_qualification_solve(
    result: phx.meshing.CellMeshingResult,
    spaces: dict[str, phx.discretization.FiniteElementDiscretization],
    problem: phx.equations.CompiledFiniteElementProblem,
    kept: np.ndarray,
    weights: np.ndarray,
    state: dict[str, jax.Array],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> None:
    """Accept the continued scalar solve without resetting compatible states."""

    _ledger = StageLedger(record, deadline=Deadline(started + args.timeout))
    epoch = len(record["epochs"])

    seed = _ledger.run(
        f"solver_execution_epoch_{epoch}",
        lambda: (
            jnp.zeros((kept.size,), dtype=result.mesh.coordinates.dtype)
            if "u" not in state
            else jnp.asarray(state["u"])[kept]
        ),
    )
    seed_error = (
        None
        if "u" not in state
        else _periodic_qualification_error(spaces["u"], state["u"], record["case"])
    )
    values, compiler, timings = _compiled_solver_stages(problem, seed, args.repeats)
    record["stages_seconds"].update(
        {f"{name}_epoch_{epoch}": value for name, value in timings.items()}
    )
    state["u"] = values
    mean = float(weights @ np.asarray(values) / np.sum(weights))
    error = _periodic_qualification_error(spaces["u"], values, record["case"])
    field_norms = {
        name: float(
            np.sqrt(
                max(
                    0.0,
                    float(
                        np.asarray(state[name])
                        @ (spaces[name].mass.to_scipy() @ np.asarray(state[name]))
                    ),
                )
            )
        )
        for name in ("vector", "flux")
    }
    if (
        abs(mean) > 1.0e-10
        or not np.isfinite(error)
        or not all(np.isfinite(list(field_norms.values())))
    ):
        raise qualification_failure(
            "_periodic_qualification_solve",
            (
                "Continued periodic state failed mean, physical error, or compatible field acceptance."
            ),
            {"error": error, "field_norms": field_norms, "mean": mean},
        )
    if error <= args.target_error and record["time_to_target_error_seconds"] is None:
        record["time_to_target_error_seconds"] = perf_counter() - started
    record["epochs"].append(
        {
            "epoch": epoch,
            "result": _result_identity(result),
            "l2_error": error,
            "mean": mean,
            "compatible_field_l2_norms": field_norms,
            "seed_origin": "zero"
            if seed_error is None
            else "actual-transferred-accepted-scalar-state",
            "transferred_state_l2_error": seed_error,
            "compiler": compiler,
            "dofs": {
                name: space.dof_maps[0].global_dof_count for name, space in spaces.items()
            },
            "retained_bytes": model_array_bytes((result, spaces, problem, state)),
        }
    )


@meshing_source_owner_reconstruction_scope()
def _periodic_qualification_cold_process(
    path: str,
    content_id: str,
    expected: dict[str, Any],
    case: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    """Renew the entire accepted chain and continue its final real physical state."""

    started = perf_counter()
    restored = read_meshing_source_closure(
        Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    epochs = validate_meshing_accepted_epochs(restored)
    identities = {
        "source": restored["generation_source"].binding_id,
        "specification": restored["generation_specification"].specification_id,
        "options": restored["generation_options"].options_id,
        "epochs": [epoch.epoch_id for epoch in epochs],
        "carriers": [_result_identity(epoch.carrier) for epoch in epochs],
    }
    identities = json.loads(json.dumps(identities))
    if identities != expected:
        raise qualification_failure(
            "_periodic_qualification_cold_process",
            ("Cold periodic restart changed source, accepted history or raw fields."),
            {"expected": expected, "identities": identities},
        )
    renewed = []
    for epoch in epochs:
        carrier = epoch.carrier
        certification = carrier.certification
        if certification is None:
            raise qualification_failure(
                "_periodic_qualification_cold_process",
                ("Cold periodic carrier lost its accepted source certificate."),
                {"certification": certification},
            )
        report = recertify_restored_meshing_source(
            certification.request,
            carrier.mesh,
            carrier.geometry,
            carrier.audit,
            archive_limits=ArrayArchiveLimits(),
        )
        report.require_passed()
        if report.report_id != certification.report_id:
            raise qualification_failure(
                "_periodic_qualification_cold_process",
                ("Cold periodic source renewal changed the original accepted theorem."),
                {"certification": certification, "report": report},
            )
        acceptance = _periodic_qualification_acceptance(carrier, case)
        for declaration in epoch.declarations:
            prepare_meshing_field_owner(declaration, carrier)
        renewed.append(
            {
                "epoch_id": epoch.epoch_id,
                "independent_acceptance": acceptance,
                "report_id": report.report_id,
            }
        )
    final = epochs[-1]
    spaces, problem, kept, weights = _periodic_qualification_spaces(final.carrier, case)
    state = {
        name: jnp.asarray(final.fields[f"periodic:{name}"])
        for name in ("u", "vector", "flux")
    }
    record = {
        "epochs": [],
        "stages_seconds": {},
        "case": case,
        "time_to_target_error_seconds": None,
    }
    args = QualificationControls.from_record(controls)
    _periodic_qualification_solve(
        final.carrier, spaces, problem, kept, weights, state, args, record, started
    )
    return {
        "process_id": os.getpid(),
        "runtime": _runtime_identity(),
        "identities": identities,
        "renewed_epochs": renewed,
        "continued_solver_epoch": record["epochs"][-1],
        "wall_seconds": perf_counter() - started,
    }


def _periodic_qualification_checkpoint(
    source: Any,
    specification: Any,
    options: Any,
    epochs: list[Any],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> None:
    """Publish the source and final accepted history, then restart cold."""

    _ledger = StageLedger(record, deadline=Deadline(started + args.timeout))

    limits = ArrayArchiveLimits()
    policy = asdict(limits)
    policy["allowed_dtype_kinds"] = sorted(limits.allowed_dtype_kinds)
    result = epochs[-1].carrier
    certification = result.certification
    if certification is None:
        raise qualification_failure(
            "_periodic_qualification_checkpoint",
            ("The accepted periodic generation lacks its source certificate."),
            {"certification": certification},
        )
    records = {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": result.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "accepted_target": epochs[-1].carrier,
        "accepted_data": {"accepted_epochs": tuple(epochs)},
    }
    evidence: dict[str, Any] = {
        "archive_limits": policy,
        "prepared_fe_caches_serialized": False,
    }
    record["checkpoint"] = evidence
    with TemporaryDirectory(prefix="phydrax-native-periodic-source-") as directory:
        receipt, timing = _ledger.run(
            "native_checkpoint_publication",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "periodic.checkpoint.zip", records, limits=limits
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        evidence["content_id"] = receipt.content_id
        evidence["container_bytes"] = receipt.path.stat().st_size
        remaining = args.timeout - (perf_counter() - started)
        if remaining <= 0.0:
            raise TimeoutError(
                f"periodic.remaining_seconds must be positive before cold restart; got {remaining!r}"
            )
        expected = {
            "source": source.binding_id,
            "specification": specification.specification_id,
            "options": options.options_id,
            "epochs": [epoch.epoch_id for epoch in epochs],
            "carriers": [_result_identity(epoch.carrier) for epoch in epochs],
        }
        expected = json.loads(json.dumps(expected))
        controls = {**args.to_record(), "timeout": remaining}
        payload = Path(directory) / "cold-input.json"
        payload.write_text(
            json.dumps(
                {"expected": expected, "case": record["case"], "controls": controls}
            )
        )
        # The scenario module owns its worker command.

        remaining = _ledger.run(
            "native_checkpoint_fresh_process_restart",
            lambda: args.timeout - (perf_counter() - started),
        )
        if remaining <= 0.0:
            raise TimeoutError(
                f"periodic.remaining_seconds must be positive before cold restart; got {remaining!r}"
            )
        try:
            cold = run_cold_worker(
                "scenarios.periodic",
                (str(receipt.path), receipt.content_id, str(payload)),
                deadline=Deadline(started + args.timeout),
                ledger=StageLedger(
                    record,
                    deadline=Deadline(started + args.timeout),
                    cold_phase="native_checkpoint_fresh_process_restart",
                ),
            )
        except ColdWorkerTimeout as failure:
            evidence.update(
                {
                    "timeout_seconds": remaining,
                    "stdout": failure.stdout,
                    "stderr": failure.stderr,
                }
            )
            raise TimeoutError(
                f"periodic.cold_restart must finish within {remaining!r} seconds; got {type(failure).__name__}"
            ) from failure
        evidence.update(
            {
                "returncode": 0,
                "stdout": json.dumps(cold),
                "stderr": "",
            }
        )
        # Worker return-code evidence is retained by StageLedger.
        # Parsed and process-validated by run_cold_worker.
        if cold["process_id"] == os.getpid() or cold["identities"] != expected:
            raise qualification_failure(
                "_periodic_qualification_checkpoint",
                ("Periodic restart lacks fresh-process accepted-history identity."),
                {"cold": cold, "expected": expected},
            )
        cold["continued_solver_epoch"]["epoch"] = len(record["epochs"])
        cold["continued_solver_epoch"]["seed_origin"] = (
            "actual-cold-restored-accepted-scalar-state"
        )
        evidence["cold_restart"] = cold
        record["epochs"].append(cold["continued_solver_epoch"])
        if (
            cold["continued_solver_epoch"]["l2_error"] <= args.target_error
            and record["time_to_target_error_seconds"] is None
        ):
            record["time_to_target_error_seconds"] = perf_counter() - started


def _periodic_qualification_epochs(
    result: phx.meshing.CellMeshingResult,
    source: phx.meshing.NativePeriodicSource,
    specification: phx.meshing.SurfaceMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> None:
    """Run accuracy-controlled orbit refinement/coarsening through transferred state.

    Refine until the unchanged physical L2 target is reached. Coarsening marks
    sibling families only where their retained coarse-solve error indicator
    permits the target error density; periodic conformity remains owner-enforced.
    Every coarsening removes cells, but need not restore the initial topology.
    Every accepted carrier keeps its varying-DOF state and transition.
    """

    _ledger = StageLedger(record, deadline=Deadline(started + args.timeout))

    solver_preparation, measured = _ledger.run(
        "solver_prepare_epoch_0",
        lambda: _measured(lambda: _periodic_qualification_spaces(result, record["case"])),
    )
    spaces, problem, kept, weights = solver_preparation
    record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
    state, measured = _ledger.run(
        record.get("active_phase", "block_until_ready"),
        lambda: _measured(
            lambda: jax.block_until_ready(
                _periodic_qualification_initial_fields(spaces, record["case"])
            )
        ),
    )
    record["stages_seconds"]["compatible_initial_state_preparation"] = measured[
        "wall_seconds"
    ]
    _periodic_qualification_solve(
        result, spaces, problem, kept, weights, state, args, record, started
    )
    accepted_epochs = [
        prepare_periodic_accepted_epoch(
            result, spaces, state, 0, failure_factory=qualification_failure
        )
    ]
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=specification.limits,
        association_transfer=phx.meshing.PeriodicAssociationTransfer(
            source, specification
        ),
    )
    history = None
    for cycle in range(args.adaptation_rounds):
        operation = "refine"
        coarse_squared_errors = np.empty((0,), dtype=np.float64)
        coarse_measures = np.empty((0,), dtype=np.float64)
        while operation is not None:
            if perf_counter() - started >= args.timeout:
                raise TimeoutError(
                    f"periodic.elapsed_seconds must be below {args.timeout!r}; got {perf_counter() - started!r}"
                )
            index = len(record["transitions"])
            ids = np.asarray(
                result.mesh.entity_set(result.mesh.topological_dimension).entity_ids,
                dtype=np.int64,
            )
            empty = np.empty((0,), dtype=np.int64)
            if operation == "refine":
                coarse_indicators, measured = _ledger.run(
                    f"accuracy_indicator_transition_{index}",
                    lambda: _measured(
                        lambda: _periodic_qualification_cell_errors(
                            spaces["u"], state["u"], record["case"]
                        )
                    ),
                )
                coarse_squared_errors, coarse_measures = coarse_indicators
                record["stages_seconds"][record["active_phase"]] = measured[
                    "wall_seconds"
                ]
                coarsen_ids = empty
            else:
                if history is None:
                    raise ValueError(
                        "periodic.history must retain the refined sibling families; got None"
                    )
                coarsen_ids = _periodic_qualification_coarsening_marks(
                    coarse_squared_errors,
                    coarse_measures,
                    np.asarray(history.parent_cells),
                    ids,
                    args.target_error,
                )
            request = phx.meshing.MarkedMeshAdaptation(
                ids if operation == "refine" else empty,
                coarsen_ids,
                hierarchy=history,
            )

            adaptation_preparation, measured = _ledger.run(
                f"{operation}_prepare_cycle_{cycle}",
                lambda: _measured(
                    lambda: phx.meshing.prepare_mesh_adaptation(
                        result, request, policy=policy
                    )
                ),
            )
            record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
            if operation == "coarsen":
                coarsen_ids, measured = _ledger.run(
                    f"coarsen_select_cycle_{cycle}",
                    lambda: _measured(
                        lambda: phx.meshing.periodic_coarsening_cell_ids(
                            adaptation_preparation
                        )
                    ),
                )
                record["stages_seconds"][record["active_phase"]] = measured[
                    "wall_seconds"
                ]
                if coarsen_ids.size == 0:
                    raise qualification_failure(
                        "_periodic_qualification_epochs",
                        ("No accuracy-admissible periodic sibling family can coarsen."),
                        {"coarse_squared_errors": coarse_squared_errors},
                    )
                if not np.array_equal(coarsen_ids, np.asarray(request.coarsen_cell_ids)):
                    request = phx.meshing.MarkedMeshAdaptation(
                        empty, coarsen_ids, hierarchy=history
                    )
                    adaptation_preparation, measured = _ledger.run(
                        f"coarsen_closed_prepare_cycle_{cycle}",
                        lambda: _measured(
                            lambda: phx.meshing.prepare_mesh_adaptation(
                                result, request, policy=policy
                            )
                        ),
                    )
                    record["stages_seconds"][record["active_phase"]] = measured[
                        "wall_seconds"
                    ]

            adaptation, measured = _ledger.run(
                f"{operation}_execute_cycle_{cycle}",
                lambda: _measured(
                    lambda: phx.meshing.execute_mesh_adaptation(adaptation_preparation)
                ),
            )
            record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
            transition: dict[str, Any] = {
                "cycle": cycle,
                "operation": operation,
                "status": adaptation.status.value,
                "source": _result_identity(result),
                "target": _result_identity(adaptation.target),
                "evidence": (
                    None
                    if adaptation.evidence is None
                    else _bisection_evidence(adaptation.evidence)
                ),
                "retained_bytes": model_array_bytes(adaptation),
                "accuracy_control": {
                    "indicator": "retained-coarse-Galerkin-cell-physical-L2",
                    "coarse_l2_error": float(np.sqrt(np.sum(coarse_squared_errors))),
                    "target_l2_error": args.target_error,
                    "marked_coarsen_cells": coarsen_ids.size,
                    "coarsening_requires_initial_topology": False,
                },
            }
            record["transitions"].append(transition)
            if adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE:
                raise qualification_failure(
                    "_periodic_qualification_epochs",
                    (
                        "A required periodic orbit transaction was not completely accepted."
                    ),
                    {"adaptation": adaptation},
                )
            if adaptation.lineage is None or not isinstance(
                adaptation.hierarchy, phx.meshing.PeriodicRefinement
            ):
                raise qualification_failure(
                    "_periodic_qualification_epochs",
                    (
                        "An accepted periodic transaction lacks lineage or retained orbit history."
                    ),
                    {"adaptation": adaptation},
                )
            transition["lineage"] = _bisection_lineage(adaptation)
            history = adaptation.hierarchy
            transition["orbit_history_id"] = history.refinement_id
            transition["retired_history_id"] = (
                None if history.retired is None else history.retired.refinement_id
            )
            target = adaptation.target

            acceptance, measured = _ledger.run(
                f"independent_acceptance_transition_{index}",
                lambda: _measured(
                    lambda: _periodic_qualification_acceptance(target, record["case"])
                ),
            )
            transition["independent_acceptance"] = acceptance
            record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
            if operation == "coarsen" and sum(
                block.cell_count for block in target.mesh.blocks
            ) >= sum(block.cell_count for block in result.mesh.blocks):
                raise qualification_failure(
                    "_periodic_qualification_epochs",
                    ("Accuracy-controlled coarsening did not remove any cells."),
                    {"operation": operation, "source": result, "target": target},
                )

            solver_preparation, measured = _ledger.run(
                f"solver_prepare_epoch_{len(record['epochs'])}",
                lambda: _measured(
                    lambda: _periodic_qualification_spaces(target, record["case"])
                ),
            )
            target_spaces, problem, kept, weights = solver_preparation
            record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
            state = _periodic_qualification_transfer(
                adaptation,
                spaces,
                target_spaces,
                state,
                operation,
                transition,
                record,
                index,
                deadline=Deadline(started + args.timeout),
            )
            transferred_mean = float(weights @ np.asarray(state["u"]) / np.sum(weights))
            transition["transferred_scalar_mean"] = transferred_mean
            if abs(transferred_mean) > 1.0e-10:
                raise qualification_failure(
                    "_periodic_qualification_epochs",
                    (
                        "Scalar state transfer changed the accepted periodic mean-zero inventory."
                    ),
                    {"transferred_mean": transferred_mean},
                )
            result, spaces = target, target_spaces
            _periodic_qualification_solve(
                result, spaces, problem, kept, weights, state, args, record, started
            )
            accepted_epochs.append(
                prepare_periodic_accepted_epoch(
                    result,
                    spaces,
                    state,
                    len(accepted_epochs),
                    transition=adaptation,
                    failure_factory=qualification_failure,
                )
            )
            operation = (
                (
                    "coarsen"
                    if record["epochs"][-1]["l2_error"] <= args.target_error
                    else "refine"
                )
                if operation == "refine"
                else None
            )
    _periodic_qualification_checkpoint(
        source, specification, options, accepted_epochs, args, record, started
    )


def _periodic_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Run native source publication and accuracy-controlled conservative orbit cycles."""

    if (
        args.resolution < 4
        or args.capacity < 1
        or args.repeats < 1
        or args.adaptation_rounds < 1
        or not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
        or (args.periodic_case == "skew-material-feature" and args.resolution % 2)
    ):
        raise ValueError(
            f"periodic.controls must have resolution >= 4 (even for materials), positive budgets and targets, and at least one cycle; got resolution={args.resolution!r}, capacity={args.capacity!r}, repeats={args.repeats!r}, rounds={args.adaptation_rounds!r}, timeout={args.timeout!r}, target_error={args.target_error!r}, case={args.periodic_case!r}"
        )
    started = perf_counter()
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-periodic-lifecycle",
        "engine": "native",
        "status": "failed",
        "runtime": _runtime_identity(),
        "expected_admission": "positive",
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "adaptation_cycles": args.adaptation_rounds,
            "warm_repeats": args.repeats,
        },
        "stages_seconds": {},
        "epochs": [],
        "transitions": [],
        "active_phase": "source_preparation",
        "physical_error": None,
        "physical_error_scope": "last-accepted-solver-epoch",
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "independent_scientific_certification": "unassessed",
        "manufactured_solution": {
            "exact": "cos(2pi fractional_x)+0.5 sin(2pi fractional_y)",
            "diffusion": 1.0,
            "gauge": "physical-mass-weighted-zero-mean",
            "norm": "physical-L2-8-point-tensor-triangle-quadrature",
        },
        "nonclaims": _periodic_qualification_requirements()["nonclaims"],
    }
    _ledger = StageLedger(record, deadline=Deadline(started + args.timeout))
    try:
        fixture, measured = _ledger.run(
            record.get("active_phase", "_periodic_qualification_fixture"),
            lambda: _measured(lambda: _periodic_qualification_fixture(args)),
        )
        source, specification, options, case = fixture
        record["case"] = case
        record["stages_seconds"]["source_preparation"] = measured["wall_seconds"]
        record.update(
            {
                "source_binding_id": source.binding_id,
                "request_id": specification.specification_id,
                "limits_id": specification.limits.limits_id,
                "native_options_id": options.options_id,
                "request": {
                    "specification_id": specification.specification_id,
                    "limits_id": specification.limits.limits_id,
                    "feature_ids": case["protected_feature_ids"],
                },
            }
        )
        provider = phx.meshing.NativeMeshingProvider(options)

        plan, measured = _ledger.run(
            "geometry_constraint_preparation",
            lambda: _measured(
                lambda: provider.plan(
                    source,
                    specification,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=phases.append,
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
        record["prepared_id"] = plan.plan_id

        result, measured = _ledger.run(
            "generation_publication_certification",
            lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
        )
        record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
        record["generation"] = _result_identity(result)

        acceptance, measured = _ledger.run(
            "independent_acceptance",
            lambda: _measured(lambda: _periodic_qualification_acceptance(result, case)),
        )
        record["independent_acceptance"] = acceptance
        record["stages_seconds"][record["active_phase"]] = measured["wall_seconds"]
        _periodic_qualification_epochs(
            result, source, specification, options, args, record, started
        )
        final_error = record["epochs"][-1]["l2_error"]
        record.update(
            {
                "physical_error": final_error,
                "target_error_reached": final_error <= args.target_error,
                "status": "passed" if final_error <= args.target_error else "failed",
                "independent_scientific_certification": "passed",
                "implemented_admitted_workflow": "periodic-provider-checkpoint-restart-orbit-refine-coarsen-H1-Hcurl-Hdiv-Poisson",
            }
        )
        if record["status"] == "failed":
            record["failure"] = (
                "Continued periodic Poisson solve missed the frozen final physical-error target."
            )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
        if (
            record["transitions"]
            and record["transitions"][-1]["operation"] == "coarsen"
            and record["active_phase"].startswith("field_transfer_prepare")
        ):
            record["positive_prerequisite"] = (
                "Canonical reference-chart coarsening must admit conservative scalar H1 and "
                "compatible H(curl)/H(div) projection with inventory and commuting certificates."
            )
    finally:
        record["native_phases"] = _phase_recordings(phases)
        if record["epochs"]:
            record["physical_error"] = record["epochs"][-1]["l2_error"]
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
        record["process_peak_scope"] = "process-lifetime"
        record["device_memory"] = [
            {
                "device_id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "status": "unavailable" if statistics is None else "backend-reported",
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "memory_statistics": statistics,
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
        if record["status"] != "passed":
            record["failure_phase"] = record["active_phase"]
        if record["end_to_end_seconds"] > args.timeout:
            record["status"] = "timeout"
            record["failure_phase"] = record["active_phase"]
        del record["active_phase"]
    return record


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    p = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    return _periodic_qualification_cold_process(
        argv[0], argv[1], p["expected"], p["case"], p["controls"]
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Run accuracy-controlled periodic refinement/coarsening with frozen host controls."""
    return _periodic_qualification_lifecycle(args)

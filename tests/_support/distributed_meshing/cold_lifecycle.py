# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Cold restoration and continuation using native meshing owners."""

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax.discretization import ExactPlcCellGeometrySource
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization.finite_volume._hybrid_diffusion import HybridMimeticDiffusion
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._periodic_power_preparation import PeriodicPowerPreparation
from phydrax.geometry.brep._model import BRepModel
from phydrax.lifecycle._meshing_field_records import (
    prepare_meshing_field_dynamics,
    prepare_meshing_field_owner,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    recertify_restored_meshing_source,
    require_same_source_evidence,
)
from phydrax.meshing._bisection import BisectionHierarchy
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.cold_lifecycle_identities import (
    array_record,
    coordinate_record,
    mesh_record,
    plc_source_record,
)


def uniform_source_inverse(path: str, content_id: str) -> dict[str, Any]:
    original, fine, hierarchy, policy = read_meshing_source_closure(
        path, expected_content_id=content_id
    )
    before_work = {
        name: int(np.asarray(value))
        for name, value in fine.execution_evidence.work_counters.items()
    }
    lineage = hierarchy.uniform_refinement
    request = phx.meshing.MarkedMeshAdaptation(
        (),
        np.sort(
            np.concatenate([np.asarray(block.global_ids) for block in fine.mesh.blocks])
        ),
        hierarchy=hierarchy,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(fine, request, policy=policy)
    )
    result_hierarchy = result.hierarchy
    if not isinstance(result_hierarchy, BisectionHierarchy):
        raise AssertionError(
            "cold_lifecycle.result_hierarchy must be a bisection hierarchy; "
            f"got {type(result_hierarchy).__name__}"
        )
    execution_evidence = result.target.execution_evidence
    if execution_evidence is None:
        raise AssertionError(
            "cold_lifecycle.execution_evidence must contain native work and "
            "storage witnesses; got None"
        )
    return {
        "status": result.status.value,
        "original": mesh_record(original.mesh),
        "target": mesh_record(result.target.mesh),
        "original_entity_ids": [
            array_record(original.mesh.entity_set(degree).entity_ids)
            for degree in range(original.mesh.topological_dimension + 1)
        ],
        "target_entity_ids": [
            array_record(result.target.mesh.entity_set(degree).entity_ids)
            for degree in range(original.mesh.topological_dimension + 1)
        ],
        "tags": array_record(result_hierarchy.tags),
        "parent_tags": array_record(lineage.parent_tags),
        "generations": array_record(result_hierarchy.generations),
        "parent_levels": array_record(lineage.parent_levels),
        "fine_work": {
            name: int(np.asarray(value))
            for name, value in fine.execution_evidence.work_counters.items()
        },
        "before_work": before_work,
        "total_work_units": int(execution_evidence.total_work_units),
        "host_storage_peak_bytes_upper": int(
            execution_evidence.host_storage_peak_bytes_upper
        ),
        "uniform_refinement_absent": result_hierarchy.uniform_refinement is None,
    }


def decimal_plc_inverse(path: str, content_id: str, route: str) -> dict[str, Any]:
    original, fine, hierarchy = read_meshing_source_closure(
        path, expected_content_id=content_id
    )
    children = np.concatenate(
        [np.asarray(block.global_ids) for block in fine.mesh.blocks]
    )
    original_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in original.mesh.blocks]
    )
    marks = children[~np.isin(children, original_ids)]
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute(route),
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            fine,
            phx.meshing.MarkedMeshAdaptation((), marks, hierarchy=hierarchy),
            policy=policy,
        )
    )
    original_source = original.geometry.exact_source
    target_source = result.target.geometry.exact_source
    if not isinstance(original_source, ExactPlcCellGeometrySource):
        raise TypeError(
            f"cold_lifecycle.original_source must be an ExactPlcCellGeometrySource; got {type(original_source).__name__}"
        )
    if not isinstance(target_source, ExactPlcCellGeometrySource):
        raise TypeError(
            f"cold_lifecycle.target_source must be an ExactPlcCellGeometrySource; got {type(target_source).__name__}"
        )
    with coordinate_reference_budget(
        queries_upper=(
            cell_geometry_preparation_query_upper(original.geometry)
            + cell_geometry_preparation_query_upper(result.target.geometry)
        ),
    ) as budget:
        return {
            "status": result.status.value,
            "original": mesh_record(original.mesh),
            "target": mesh_record(result.target.mesh),
            "original_source_coordinates": coordinate_record(
                original.geometry.source_coordinates(budget=budget)
            ),
            "target_source_coordinates": coordinate_record(
                result.target.geometry.source_coordinates(budget=budget)
            ),
            "original_source": plc_source_record(original_source),
            "target_source": plc_source_record(target_source),
        }


def curved_brep_domain(path: str, content_id: str) -> dict[str, Any]:
    domain = read_meshing_source_closure(path, expected_content_id=content_id)
    return {
        "domain_type_exact": type(domain) is MeshingDomain,
        "brep_type_exact": type(domain.brep_authority) is BRepModel,
        "identity": [
            domain.source_id,
            domain.source_revision,
            domain.domain_id,
            domain.authority_id,
            type(domain.curve_atlas.mapping).__name__,
            domain.brep_authority.model_id,
        ],
    }


def periodic_power_preparation(path: str, content_id: str) -> dict[str, Any]:
    restored = read_meshing_source_closure(path, expected_content_id=content_id)
    owner = restored[0]
    owner.validate_restored()
    return {
        "type_exact": type(owner) is PeriodicPowerPreparation,
        "alias": owner is restored[1],
    }


def periodic_finite_volume(path: str, content_id: str) -> dict[str, Any]:
    restored = read_meshing_source_closure(path, expected_content_id=content_id)
    publication = restored["generation_part"].carrier
    report = recertify_restored_meshing_source(
        restored["certification_inputs"],
        publication.mesh,
        publication.geometry,
        publication.audit,
    )
    old, current = require_same_source_evidence(
        restored["report"],
        report,
        publication.mesh,
        publication.geometry,
        publication.audit,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    declaration = restored["field_declarations"]["generation"]
    owner, _ = prepare_meshing_field_owner(declaration, publication)
    if not isinstance(owner, UnstructuredFiniteVolumeDiscretization):
        raise AssertionError(
            "cold_lifecycle.field_owner must be an unstructured finite-volume "
            f"discretization; got {type(owner).__name__}"
        )
    dynamics = prepare_meshing_field_dynamics(declaration, publication)
    arrays = restored["accepted_data"]["fields"]
    diffusion = HybridMimeticDiffusion(owner)
    return {
        "old_report_alias": old is restored["report"],
        "current_report_alias": current is report,
        "local_fluxes": array_record(
            diffusion.local_fluxes(
                arrays["scalar_state"][:, 0],
                jnp.ones(diffusion.face_count, dtype=jnp.float64),
                jnp.eye(3, dtype=jnp.float64),
            )
        ),
        "scalar_history": array_record(arrays["scalar_history"]),
        "density": array_record(arrays["density"]),
        "cell_count": owner.cell_count,
        "quotient_face_count": owner.face_measures.size,
        "physical_face_count": publication.mesh.connectivity.face_count,
        "vector_rate": array_record(
            dynamics(jnp.asarray(0.04, dtype=jnp.float64), arrays["vector_state"])
        ),
        "vector_state": array_record(arrays["vector_state"]),
        "vector_history": array_record(arrays["vector_history"]),
    }

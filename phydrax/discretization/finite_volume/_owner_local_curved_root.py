from __future__ import annotations

from collections import defaultdict
from typing import Any, TYPE_CHECKING

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
)


if TYPE_CHECKING:
    from ...geometry._common_root_volume_overlap import (
        CurvedOverlapIntegrationEvidence,
        CurvedOverlapIntegrationPolicy,
    )


_REQUEST_SCHEMA = "owner-local-curved-root-map-request"
_RESULT_SCHEMA = "owner-local-curved-root-overlap-result"


def _directory_cell_owners(
    view: Any,
    identifiers: np.ndarray,
    execution_group: Any,
    axis: str,
    /,
) -> np.ndarray:
    return np.asarray(
        view.resolve_owner_ranks(
            np.asarray(identifiers, dtype=np.int64),
            execution_group,
            directory_kind="mesh-cells",
            axis_name=axis,
        ),
        dtype=np.int32,
    )


def _diameter(bounds: np.ndarray, /) -> float:
    return float(np.linalg.norm(bounds[1] - bounds[0]))


def _measure_limit(
    measured: float,
    error: float,
    bounds: np.ndarray,
    integration_policy: Any,
    /,
) -> float:
    from ._owner_local_remap import _COVERAGE_ROUNDOFF_ULPS

    return (
        integration_policy.relative_tolerance * float(measured)
        + integration_policy.absolute_tolerance
        + float(error)
        + _COVERAGE_ROUNDOFF_ULPS
        * np.finfo(np.float64).eps
        * _diameter(np.asarray(bounds, dtype=np.float64)) ** 3
    )


def _merge_bounds(existing: np.ndarray | None, candidate: np.ndarray, /) -> np.ndarray:
    bounds = np.asarray(candidate, dtype=np.float64)
    if (
        bounds.shape != (2, 3)
        or np.any(~np.isfinite(bounds))
        or np.any(bounds[0] > bounds[1])
    ):
        raise ValueError(
            "Curved root-owner integration returned malformed coordinate bounds."
        )
    if existing is None:
        return bounds.copy()
    previous = np.asarray(existing, dtype=np.float64)
    return np.stack(
        (np.minimum(previous[0], bounds[0]), np.maximum(previous[1], bounds[1]))
    )


def _bundle_integration_evidence(
    evidence_values: tuple[CurvedOverlapIntegrationEvidence, ...],
    integration_policy: CurvedOverlapIntegrationPolicy,
    /,
) -> tuple[CurvedOverlapIntegrationEvidence, ...] | None:
    from ...geometry._common_root_volume_overlap import CurvedOverlapIntegrationEvidence

    evidence = tuple(
        sorted(
            (value for value in evidence_values if value is not None),
            key=lambda value: value.evidence_id,
        )
    )
    if not evidence:
        return None
    if any(
        not isinstance(value, CurvedOverlapIntegrationEvidence)
        or value.integration_policy.policy_id != integration_policy.policy_id
        for value in evidence
    ):
        raise ValueError(
            "Curved root-owner integration evidence uses incompatible policies."
        )
    return evidence


def _check_packet_measure(
    identifier: int,
    measured: float,
    error: float,
    bounds: np.ndarray,
    declared: float,
    integration_policy: Any,
    stage: str,
    /,
) -> None:
    if (
        not np.isfinite(measured)
        or not np.isfinite(declared)
        or measured <= 0.0
        or declared <= 0.0
    ):
        raise ValueError(
            f"Curved root-owner {stage} measure is malformed for cell {identifier}."
        )
    limit = _measure_limit(measured, error, bounds, integration_policy)
    if abs(float(measured) - float(declared)) > limit:
        raise ValueError(
            f"Curved root-owner {stage} geometry disagrees with accepted cell {identifier}."
        )


def _validate_request_record(
    record: Any,
    sender: int,
    local_rank: int,
    root_role: str,
    transition_id: str,
    source_binding: tuple[Any, ...],
    target_binding: tuple[Any, ...],
    root_binding: tuple[Any, ...],
    transition_maps: tuple[Any, ...],
    /,
) -> tuple[int, int]:
    from . import _owner_local_remap as owner_local

    if not isinstance(record, tuple) or len(record) != 14 or record[0] != _REQUEST_SCHEMA:
        raise ValueError("Curved root-map request has an unknown packet schema.")
    (
        _,
        packet_source_binding,
        packet_target_binding,
        packet_root_binding,
        packet_transition_id,
        packet_role,
        root_id,
        root_owner,
        endpoint_role,
        endpoint_id,
        endpoint_owner,
        vertex_ids,
        declared_measure,
        witness,
    ) = record
    expected_endpoint_role = "target" if root_role == "source" else "source"
    if (
        packet_source_binding != source_binding
        or packet_target_binding != target_binding
        or packet_root_binding != root_binding
        or packet_transition_id != transition_id
        or packet_role != root_role
        or endpoint_role != expected_endpoint_role
        or int(root_owner) != local_rank
        or int(endpoint_owner) != sender
        or int(root_id) < 0
        or int(endpoint_id) < 0
        or not np.isfinite(float(declared_measure))
        or float(declared_measure) <= 0.0
    ):
        raise ValueError(
            "Curved root-map request does not bind current endpoint authorities."
        )
    if (
        not np.array_equal(
            np.asarray(witness.target_cell_ids, dtype=np.int64),
            np.asarray((endpoint_id,), dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(witness.source_cell_ids, dtype=np.int64),
            np.asarray((root_id,), dtype=np.int64),
        )
        or not owner_local._same_reference_cell_map(
            owner_local._reference_map_row_for_cell(transition_maps, int(endpoint_id)),
            witness,
        )
    ):
        raise ValueError(
            "Curved root-map request differs from its accepted exact transition row."
        )
    endpoint_vertices = np.asarray(vertex_ids, dtype=np.int64)
    if endpoint_vertices.shape != (4,) or np.unique(endpoint_vertices).size != 4:
        raise ValueError("Curved root-map request lost its stable tetrahedron vertices.")
    return int(root_id), int(endpoint_id)


def _validate_result_record(
    record: Any,
    sender: int,
    root_role: str,
    transition_id: str,
    source_binding: tuple[Any, ...],
    target_binding: tuple[Any, ...],
    root_binding: tuple[Any, ...],
    transition_maps: tuple[Any, ...],
    target_view: Any,
    target_owned_ids: np.ndarray,
    target_declared_by_id: dict[int, float],
    local_rank: int,
    /,
) -> None:
    from ...geometry._common_root_volume_overlap import CurvedOverlapIntegrationEvidence
    from ...geometry._supermesh import PreparedCommonRefinement
    from . import _owner_local_remap as owner_local

    if not isinstance(record, tuple) or len(record) != 13 or record[0] != _RESULT_SCHEMA:
        raise ValueError("Curved root-overlap result has an unknown packet schema.")
    (
        _,
        packet_source_binding,
        packet_target_binding,
        packet_root_binding,
        packet_transition_id,
        packet_role,
        root_id,
        root_owner,
        endpoint_owner,
        requests,
        refinement,
        integration,
        root_native_measure,
    ) = record
    if (
        packet_source_binding != source_binding
        or packet_target_binding != target_binding
        or packet_root_binding != root_binding
        or packet_transition_id != transition_id
        or packet_role != root_role
        or int(root_id) < 0
        or not np.isfinite(float(root_native_measure))
        or float(root_native_measure) <= 0.0
        or not isinstance(requests, tuple)
        or not isinstance(refinement, PreparedCommonRefinement)
        or not isinstance(integration, CurvedOverlapIntegrationEvidence)
    ):
        raise ValueError("Curved root-overlap result is not bound to its routed request.")
    endpoint_ids = set()
    for request in requests:
        request_sender = int(request[10])
        if root_role == "source":
            if request_sender != local_rank or int(endpoint_owner) != local_rank:
                raise ValueError(
                    "Curved refinement result returned to a non-owning target rank."
                )
            endpoint_id = int(request[9])
            if endpoint_id not in target_declared_by_id or not np.any(
                np.asarray(target_owned_ids, dtype=np.int64) == endpoint_id
            ):
                raise ValueError(
                    "Curved root-overlap result names a non-owned target cell."
                )
            if not np.array_equal(
                np.asarray(request[11], dtype=np.int64),
                owner_local._owner_cell_vertex_ids(target_view, endpoint_id),
            ):
                raise ValueError(
                    "Curved refinement result lost the current target vertex binding."
                )
            if float(request[12]) != target_declared_by_id[endpoint_id]:
                raise ValueError(
                    "Curved refinement result changed the target cell measure binding."
                )
        elif int(root_owner) != local_rank or not np.any(
            np.asarray(target_owned_ids, dtype=np.int64) == int(root_id)
        ):
            raise ValueError(
                "Curved coarsening result is not retained by its target root owner."
            )
        if (
            int(request[6]) != int(root_id)
            or int(request[7]) != int(root_owner)
            or int(request[10]) != int(endpoint_owner)
            or request[1] != source_binding
            or request[2] != target_binding
            or request[3] != root_binding
            or request[4] != transition_id
            or request[5] != root_role
        ):
            raise ValueError(
                "Curved root-overlap result contains a stale request identity."
            )
        endpoint_id = int(request[9])
        if not owner_local._same_reference_cell_map(
            owner_local._reference_map_row_for_cell(transition_maps, endpoint_id),
            request[13],
        ):
            raise ValueError("Curved root-overlap result lost its exact transition map.")
        endpoint_ids.add(endpoint_id)
    source_ids = set(
        np.asarray(refinement.source_cell_global_ids, dtype=np.int64).tolist()
    )
    target_ids = set(
        np.asarray(refinement.target_cell_global_ids, dtype=np.int64).tolist()
    )
    if root_role == "source":
        if source_ids != {int(root_id)} or target_ids != endpoint_ids:
            raise ValueError(
                "Curved refinement result changed stable root/target cell IDs."
            )
    elif source_ids != endpoint_ids or target_ids != {int(root_id)}:
        raise ValueError("Curved coarsening result changed stable source/root cell IDs.")


def _aggregate_root_results(
    result_records: tuple[Any, ...],
    root_role: str,
    target_ids: np.ndarray,
    target_measures: np.ndarray,
    target_errors: dict[int, float],
    target_bounds: dict[int, np.ndarray],
    source_self_errors: dict[int, float],
    source_self_bounds: dict[int, np.ndarray],
    source_self_integration: CurvedOverlapIntegrationEvidence | None,
    target_self_integration: CurvedOverlapIntegrationEvidence | None,
    tolerance: float,
    require_complete: bool,
    integration_policy: CurvedOverlapIntegrationPolicy,
    source_binding: tuple[Any, ...],
    target_binding: tuple[Any, ...],
    transition_id: str,
    /,
) -> tuple[
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[CurvedOverlapIntegrationEvidence, ...] | None,
]:
    from ...geometry import _supermesh as supermesh_owner
    from . import _owner_local_remap as owner_local

    target_ids_array = np.asarray(target_ids, dtype=np.int64)
    target_measure_array = np.asarray(target_measures, dtype=np.float64)
    if target_measure_array.shape != target_ids_array.shape:
        raise ValueError("Curved root results lost target native measure rows.")
    target_index = {
        int(identifier): row for row, identifier in enumerate(target_ids_array)
    }
    source_measures_by_id: dict[int, float] = {}
    source_owners_by_id: dict[int, int] = {}
    source_errors_by_id = dict(source_self_errors)
    source_bounds_by_id = {
        int(identifier): np.asarray(bounds, dtype=np.float64).copy()
        for identifier, bounds in source_self_bounds.items()
    }
    target_errors_by_id = dict(target_errors)
    target_bounds_by_id = {
        int(identifier): np.asarray(bounds, dtype=np.float64).copy()
        for identifier, bounds in target_bounds.items()
    }
    entries: list[tuple[int, int, float, np.ndarray, np.ndarray, float]] = []
    unresolved_uncertain = unresolved_invalid = unresolved_failures = 0
    for record in result_records:
        refinement = record[10]
        integration = record[11]
        unresolved_uncertain += refinement.evidence.uncertain_predicate_count
        unresolved_invalid += refinement.evidence.invalid_cell_count
        unresolved_failures += refinement.evidence.intersection_failure_count
        requests = record[9]
        endpoint_owner_by_id = {int(request[9]): int(request[10]) for request in requests}
        source_global_ids = np.asarray(refinement.source_cell_global_ids, dtype=np.int64)
        target_global_ids = np.asarray(refinement.target_cell_global_ids, dtype=np.int64)
        source_native = np.asarray(refinement.source_measures, dtype=np.float64)
        target_native = np.asarray(refinement.target_measures, dtype=np.float64)
        source_errors = np.asarray(integration.source_errors, dtype=np.float64)[:, 0]
        target_errors = np.asarray(integration.target_errors, dtype=np.float64)[:, 0]
        source_coordinate_bounds = np.asarray(
            integration.source_coordinate_bounds, dtype=np.float64
        )
        target_coordinate_bounds = np.asarray(
            integration.target_coordinate_bounds, dtype=np.float64
        )
        canonical_source_measures = (
            {int(record[6]): float(record[12])}
            if root_role == "source"
            else {int(request[9]): float(request[12]) for request in requests}
        )
        for row, identifier in enumerate(source_global_ids.tolist()):
            key = int(identifier)
            owner = (
                int(record[7]) if root_role == "source" else endpoint_owner_by_id.get(key)
            )
            if owner is None or key not in canonical_source_measures:
                raise ValueError(
                    "Curved root result lost its source-cell owner/measure binding."
                )
            canonical_measure = canonical_source_measures[key]
            _check_packet_measure(
                key,
                float(source_native[row]),
                float(source_errors[row]),
                source_coordinate_bounds[row],
                canonical_measure,
                integration_policy,
                "candidate source",
            )
            previous = source_measures_by_id.get(key)
            error = max(source_errors_by_id.get(key, 0.0), float(source_errors[row]))
            bounds = _merge_bounds(
                source_bounds_by_id.get(key), source_coordinate_bounds[row]
            )
            if previous is not None and abs(
                previous - canonical_measure
            ) > _measure_limit(
                max(previous, canonical_measure), error, bounds, integration_policy
            ):
                raise ValueError("Repeated curved root source measures disagree.")
            source_measures_by_id[key] = canonical_measure
            source_bounds_by_id[key] = bounds
            source_owners_by_id[key] = owner
            source_errors_by_id[key] = error
        for row, identifier in enumerate(target_global_ids.tolist()):
            key = int(identifier)
            if key not in target_index:
                raise ValueError("Curved root result names a non-owned target cell.")
            target_bounds_by_id[key] = _merge_bounds(
                target_bounds_by_id.get(key), target_coordinate_bounds[row]
            )
            target_errors_by_id[key] = max(
                target_errors_by_id.get(key, 0.0), float(target_errors[row])
            )
            expected = float(target_measure_array[target_index[key]])
            if abs(expected - float(target_native[row])) > _measure_limit(
                float(target_native[row]),
                float(target_errors[row]),
                target_coordinate_bounds[row],
                integration_policy,
            ):
                raise ValueError(
                    "Curved root target measure differs from its accepted local map."
                )
        source_cells = np.asarray(refinement.source_cells, dtype=np.int32)
        target_cells = np.asarray(refinement.target_cells, dtype=np.int32)
        volumes = np.asarray(refinement.volumes, dtype=np.float64)
        first_moments = np.asarray(refinement.first_moments, dtype=np.float64)
        if refinement.second_moments is None:
            raise ValueError("Curved root overlaps require second-moment evidence.")
        second_moments = np.asarray(refinement.second_moments, dtype=np.float64)
        overlap_errors = np.asarray(integration.overlap_errors, dtype=np.float64)[:, 0]
        if overlap_errors.shape != volumes.shape:
            raise ValueError(
                "Curved root overlap errors do not align with overlap entries."
            )
        for entry, (source_row, target_row, volume) in enumerate(
            zip(source_cells, target_cells, volumes, strict=True)
        ):
            source_id = int(source_global_ids[int(source_row)])
            target_id = int(target_global_ids[int(target_row)])
            output_target = target_index.get(target_id)
            entries.append(
                (
                    output_target if output_target is not None else -1,
                    source_id,
                    float(volume),
                    first_moments[entry],
                    second_moments[entry],
                    float(overlap_errors[entry]),
                )
            )
    source_ids_array = np.asarray(sorted(source_measures_by_id), dtype=np.int64)
    source_index = {
        int(identifier): row for row, identifier in enumerate(source_ids_array)
    }
    if any(target_row < 0 for target_row, *_ in entries):
        raise ValueError("Curved root overlap entry lost its target stable ID.")
    entries.sort(key=lambda item: (item[0], item[1]))
    target_cells_array = np.asarray([item[0] for item in entries], dtype=np.int32)
    source_cells_array = np.asarray(
        [source_index[item[1]] for item in entries], dtype=np.int32
    )
    volumes_array = np.asarray([item[2] for item in entries], dtype=np.float64)
    first_moments_array = np.asarray(
        [item[3] for item in entries], dtype=np.float64
    ).reshape((-1, 3))
    second_moments_array = np.asarray(
        [item[4] for item in entries], dtype=np.float64
    ).reshape((-1, 3, 3))
    overlap_errors_array = np.asarray([item[5] for item in entries], dtype=np.float64)
    source_measure_array = np.asarray(
        [source_measures_by_id[int(identifier)] for identifier in source_ids_array],
        dtype=np.float64,
    )
    source_owner_array = np.asarray(
        [source_owners_by_id[int(identifier)] for identifier in source_ids_array],
        dtype=np.int32,
    )
    target_covered = np.bincount(
        target_cells_array, weights=volumes_array, minlength=target_ids_array.size
    )
    source_covered = np.bincount(
        source_cells_array, weights=volumes_array, minlength=source_ids_array.size
    )
    source_defects = source_covered - source_measure_array
    target_defects = target_covered - target_measure_array
    source_overlap_errors = np.bincount(
        source_cells_array, weights=overlap_errors_array, minlength=source_ids_array.size
    )
    target_overlap_errors = np.bincount(
        target_cells_array, weights=overlap_errors_array, minlength=target_ids_array.size
    )
    source_limits = np.asarray(
        [
            tolerance * source_measure_array[row]
            + source_errors_by_id[int(identifier)]
            + source_overlap_errors[row]
            + owner_local._COVERAGE_ROUNDOFF_ULPS
            * np.finfo(np.float64).eps
            * _diameter(source_bounds_by_id[int(identifier)]) ** 3
            for row, identifier in enumerate(source_ids_array.tolist())
        ],
        dtype=np.float64,
    )
    target_limits = np.asarray(
        [
            tolerance * target_measure_array[row]
            + target_errors_by_id[int(identifier)]
            + target_overlap_errors[row]
            + owner_local._COVERAGE_ROUNDOFF_ULPS
            * np.finfo(np.float64).eps
            * _diameter(target_bounds_by_id[int(identifier)]) ** 3
            for row, identifier in enumerate(target_ids_array.tolist())
        ],
        dtype=np.float64,
    )
    policy = CommonRefinementPolicy(
        coverage=(
            CommonRefinementCoverage.TARGET
            if require_complete
            else CommonRefinementCoverage.PARTIAL
        ),
        coverage_tolerance=tolerance,
        second_moments=True,
    )
    status, reason = supermesh_owner._coverage_status(
        source_defects,
        target_defects,
        source_limits,
        target_limits,
        policy,
    )
    fatal_statuses = tuple(
        refinement.status
        for record in result_records
        for refinement in (record[10],)
        if refinement.status
        in (
            CommonRefinementStatus.INVALID_GEOMETRY,
            CommonRefinementStatus.PREDICATE_UNCERTAIN,
            CommonRefinementStatus.INTERSECTION_FAILURE,
            CommonRefinementStatus.DOUBLE_COVERAGE,
        )
    )
    if fatal_statuses:
        priority = {
            CommonRefinementStatus.SUCCESS: 0,
            CommonRefinementStatus.COVERAGE_GAP: 1,
            CommonRefinementStatus.DOUBLE_COVERAGE: 2,
            CommonRefinementStatus.INTERSECTION_FAILURE: 3,
            CommonRefinementStatus.PREDICATE_UNCERTAIN: 4,
            CommonRefinementStatus.INVALID_GEOMETRY: 5,
        }
        status = max(fatal_statuses, key=priority.__getitem__)
        reason = f"curved root-owner subrefinement reported {status.name}"
    counts = supermesh_owner._Counts(
        len(result_records),
        int(volumes_array.size),
        0,
        unresolved_uncertain,
        unresolved_invalid,
        unresolved_failures,
    )
    evidence = CommonRefinementEvidence(
        status=status,
        reason=reason,
        source_coverage_defects=source_defects,
        target_coverage_defects=target_defects,
        source_measures=source_measure_array,
        target_measures=target_measure_array,
        source_tolerances=source_limits,
        target_tolerances=target_limits,
        counts=counts,
    )
    offsets = np.zeros((target_ids_array.size + 1,), dtype=np.int32)
    np.cumsum(
        np.bincount(target_cells_array, minlength=target_ids_array.size),
        out=offsets[1:],
    )
    refinement_id = canonical_fingerprint(
        {
            "kind": "owner-local-curved-root-refinement",
            "source_authority": source_binding,
            "target_authority": target_binding,
            "transition": transition_id,
            "source_ids": array_tree_fingerprint(source_ids_array),
            "target_ids": array_tree_fingerprint(target_ids_array),
            "source_cells": array_tree_fingerprint(source_cells_array),
            "target_cells": array_tree_fingerprint(target_cells_array),
            "volumes": array_tree_fingerprint(volumes_array),
            "first_moments": array_tree_fingerprint(first_moments_array),
            "second_moments": array_tree_fingerprint(second_moments_array),
            "evidence": evidence.evidence_id,
            "root_witnesses": tuple(
                sorted(record[11].witness_id for record in result_records)
            ),
        }
    )
    local_refinement = owner_local._OwnerLocalCurvedRefinement(
        status,
        source_ids_array,
        target_ids_array,
        source_measure_array,
        target_measure_array,
        offsets,
        target_cells_array,
        source_cells_array,
        volumes_array,
        first_moments_array,
        second_moments_array,
        evidence,
        refinement_id,
        int(target_ids_array.size),
    )
    integration_bundle = _bundle_integration_evidence(
        tuple(result[11] for result in result_records)
        + tuple(
            value
            for value in (source_self_integration, target_self_integration)
            if value is not None
        ),
        integration_policy,
    )
    return (
        local_refinement,
        source_ids_array,
        source_owner_array,
        source_measure_array,
        overlap_errors_array,
        integration_bundle,
    )


def prepare_owner_local_curved_root_refinement(
    source_view: Any,
    target_view: Any,
    source_kind: str,
    target_kind: str,
    source_ids: np.ndarray,
    source_owners: np.ndarray,
    source_declared: np.ndarray,
    target_ids: np.ndarray,
    target_owned_ids: np.ndarray,
    target_declared: np.ndarray,
    source_owned_ids: np.ndarray,
    transition: Any,
    tolerance: float,
    require_complete: bool,
    execution_group: Any,
    mesh: Any,
    axis: str,
    local_rank: int,
    owner_count: int,
    /,
) -> tuple[Any, ...]:
    from ...geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy
    from ...geometry._supermesh import prepare_compact_cell_common_refinement
    from . import _owner_local_remap as owner_local

    integration_policy = CurvedOverlapIntegrationPolicy()
    coverage_policy = CommonRefinementPolicy(
        coverage=(
            CommonRefinementCoverage.TARGET
            if require_complete
            else CommonRefinementCoverage.PARTIAL
        ),
        coverage_tolerance=tolerance,
        second_moments=True,
    )
    setup_error: Exception | None = None
    try:
        if source_kind != "tetrahedron" or target_kind != "tetrahedron":
            raise ValueError(
                "Curved owner-local common-root overlap requires tetrahedra."
            )
        if transition is None:
            raise ValueError(
                "Curved owner-local overlap requires a canonical typed transition."
            )
        empty_ids = np.empty((0,), dtype=np.int64)
        empty_vertices: tuple[np.ndarray, ...] = ()
        empty_refinement = prepare_compact_cell_common_refinement(
            empty_vertices,
            empty_ids,
            empty_vertices,
            empty_ids,
            source_cell_kind=source_kind,
            target_cell_kind=target_kind,
            policy=coverage_policy,
        )
        (
            source_geometry_refinement,
            source_self_integration,
            source_native,
            source_errors,
            source_bounds,
        ) = owner_local._curved_self_cell_geometry(
            source_view, source_kind, empty_refinement, tolerance, integration_policy
        )
        (
            target_geometry_refinement,
            target_self_integration,
            target_native,
            target_errors,
            target_bounds,
        ) = owner_local._curved_self_cell_geometry(
            target_view, target_kind, empty_refinement, tolerance, integration_policy
        )
        source_owned_measures = np.asarray(
            [source_native[int(identifier)] for identifier in source_owned_ids],
            dtype=np.float64,
        )
        target_owned_measures = np.asarray(
            [target_native[int(identifier)] for identifier in target_owned_ids],
            dtype=np.float64,
        )
        source_local_errors = np.asarray(
            [source_errors[int(identifier)] for identifier in source_owned_ids],
            dtype=np.float64,
        )
        source_local_bounds = np.asarray(
            [source_bounds[int(identifier)] for identifier in source_owned_ids],
            dtype=np.float64,
        ).reshape((-1, 2, 3))
    except Exception as caught:
        setup_error = caught
    owner_local._synchronize_error(
        setup_error, mesh, axis, owner_count, "curved local endpoint certification"
    )

    map_error: Exception | None = None
    local_map_rows = []
    try:
        transition_maps = owner_local._transition_reference_maps(transition)
        if transition_maps is None:
            raise ValueError(
                "Curved owner-local overlap has no exact transition map bank."
            )
        root_role, map_groups = transition_maps
        if root_role not in ("source", "target"):
            raise ValueError(
                "Curved owner-local transition has an unknown root endpoint."
            )
        root_view = source_view if root_role == "source" else target_view
        endpoint_view = target_view if root_role == "source" else source_view
        endpoint_role = "target" if root_role == "source" else "source"
        endpoint_ids = np.asarray(
            target_owned_ids if root_role == "source" else source_owned_ids,
            dtype=np.int64,
        )
        endpoint_id_bank = np.asarray(
            target_ids if root_role == "source" else source_ids,
            dtype=np.int64,
        )
        endpoint_measure_bank = np.asarray(
            target_declared if root_role == "source" else source_declared,
            dtype=np.float64,
        )
        source_owner_by_id = (
            {
                int(identifier): int(owner)
                for identifier, owner in zip(source_ids, source_owners, strict=True)
            }
            if root_role == "target"
            else {}
        )
        endpoint_measures_by_id = {
            int(identifier): float(measure)
            for identifier, measure in zip(
                endpoint_id_bank, endpoint_measure_bank, strict=True
            )
        }
        source_binding = owner_local._owner_view_authority_identity(source_view)
        target_binding = owner_local._owner_view_authority_identity(target_view)
        root_binding = source_binding if root_role == "source" else target_binding
        transition_id = transition.transition_id
        for endpoint_id in endpoint_ids:
            endpoint_identifier = int(endpoint_id)
            if (
                endpoint_role == "source"
                and source_owner_by_id.get(endpoint_identifier) != local_rank
            ):
                raise ValueError(
                    "Coarsening map sender does not own its current source cell."
                )
            witness = owner_local._reference_map_row_for_cell(
                map_groups, endpoint_identifier
            )
            if witness is None:
                raise ValueError(
                    f"Owned curved {endpoint_role} cell {endpoint_identifier} lacks an exact root map."
                )
            if (
                witness.target_cell_kind != "tetrahedron"
                or witness.source_cell_kind != "tetrahedron"
                or not np.array_equal(
                    np.asarray(witness.target_cell_ids, dtype=np.int64),
                    np.asarray((endpoint_identifier,), dtype=np.int64),
                )
            ):
                raise ValueError("Curved root map does not bind its local endpoint cell.")
            vertex_ids = owner_local._owner_cell_vertex_ids(
                endpoint_view, endpoint_identifier
            )
            measure = endpoint_measures_by_id.get(endpoint_identifier)
            if measure is None or not np.isfinite(measure) or measure <= 0.0:
                raise ValueError(
                    "Curved endpoint has no accepted positive native measure."
                )
            local_map_rows.append(
                (
                    endpoint_identifier,
                    int(np.asarray(witness.source_cell_ids, dtype=np.int64)[0]),
                    vertex_ids,
                    float(measure),
                    witness,
                )
            )
    except Exception as caught:
        map_error = caught
    owner_local._synchronize_error(
        map_error, mesh, axis, owner_count, "curved endpoint exact-map authorization"
    )
    root_ids_for_lookup = np.asarray(
        [record[1] for record in local_map_rows], dtype=np.int64
    )
    root_owners = np.asarray(
        root_view.resolve_owner_ranks(
            root_ids_for_lookup,
            execution_group,
            directory_kind="mesh-cells",
            axis_name=axis,
        ),
        dtype=np.int32,
    )
    outgoing_requests: list[list[Any]] = [[] for _ in range(owner_count)]
    local_requests = []
    for row, root_owner in zip(local_map_rows, root_owners, strict=True):
        endpoint_identifier, root_identifier, vertex_ids, measure, witness = row
        record = (
            _REQUEST_SCHEMA,
            source_binding,
            target_binding,
            root_binding,
            transition_id,
            root_role,
            root_identifier,
            int(root_owner),
            endpoint_role,
            endpoint_identifier,
            local_rank,
            np.asarray(vertex_ids, dtype=np.int64),
            measure,
            witness,
        )
        if int(root_owner) == local_rank:
            local_requests.append(record)
        else:
            outgoing_requests[int(root_owner)].append(record)
    local_requests.sort(key=lambda record: (int(record[6]), int(record[9])))
    for records in outgoing_requests:
        records.sort(key=lambda record: (int(record[6]), int(record[9])))
    local_request_records, incoming_requests = (
        owner_local._exchange_curved_root_map_packets(
            tuple(tuple(records) for records in outgoing_requests),
            local_rank,
            execution_group,
            mesh,
            axis,
            owner_count,
        )
    )
    del local_request_records
    tagged_requests = [(local_rank, record) for record in local_requests]
    tagged_requests.extend(
        (sender, record)
        for sender, records in incoming_requests.items()
        for record in records
    )
    request_error: Exception | None = None
    try:
        for sender, record in tagged_requests:
            _validate_request_record(
                record,
                sender,
                local_rank,
                root_role,
                transition_id,
                source_binding,
                target_binding,
                root_binding,
                map_groups,
            )
    except Exception as caught:
        request_error = caught
    owner_local._synchronize_error(
        request_error, mesh, axis, owner_count, "curved root-map receiver validation"
    )
    root_owners_verified = _directory_cell_owners(
        root_view,
        np.asarray([record[6] for _, record in tagged_requests], dtype=np.int64),
        execution_group,
        axis,
    )
    endpoint_owners_verified = _directory_cell_owners(
        endpoint_view,
        np.asarray([record[9] for _, record in tagged_requests], dtype=np.int64),
        execution_group,
        axis,
    )
    directory_error = None
    for index, (sender, record) in enumerate(tagged_requests):
        if (
            int(root_owners_verified[index]) != local_rank
            or int(endpoint_owners_verified[index]) != int(record[10])
            or int(record[10]) != sender
        ):
            directory_error = ValueError(
                "Curved root-map packet owner ranks disagree with accepted directories."
            )
            break
    owner_local._synchronize_error(
        directory_error, mesh, axis, owner_count, "curved root-map directory binding"
    )

    requests_by_root_and_endpoint_owner: dict[tuple[int, int], list[tuple[int, Any]]] = (
        defaultdict(list)
    )
    for sender, record in tagged_requests:
        requests_by_root_and_endpoint_owner[(int(record[6]), int(record[10]))].append(
            (sender, record)
        )
    root_native = source_native if root_role == "source" else target_native
    local_results = []
    outgoing_results: list[list[Any]] = [[] for _ in range(owner_count)]
    root_processing_error: Exception | None = None
    try:
        for (root_id, endpoint_owner), tagged_group in sorted(
            requests_by_root_and_endpoint_owner.items()
        ):
            tagged_group.sort(key=lambda item: int(item[1][9]))
            group_records = tuple(record for _, record in tagged_group)
            map_rows = tuple(
                (record[13], np.asarray(record[11], dtype=np.int64))
                for record in group_records
            )
            if root_id not in root_native:
                raise ValueError(f"Declared root owner lacks exact root cell {root_id}.")
            refinement, integration = owner_local._root_owner_overlap(
                root_view,
                map_rows,
                root_role,
                tolerance,
                False,
                CurvedOverlapIntegrationPolicy(),
            )
            source_ids_result = np.asarray(
                refinement.source_cell_global_ids, dtype=np.int64
            )
            target_ids_result = np.asarray(
                refinement.target_cell_global_ids, dtype=np.int64
            )
            if root_role == "source":
                if set(source_ids_result.tolist()) != {root_id} or set(
                    target_ids_result.tolist()
                ) != {int(record[9]) for record in group_records}:
                    raise ValueError(
                        "Curved refinement result changed root/target stable IDs."
                    )
                root_measures = np.asarray(refinement.source_measures, dtype=np.float64)
                root_errors = np.asarray(integration.source_errors, dtype=np.float64)[
                    :, 0
                ]
                root_bounds = np.asarray(
                    integration.source_coordinate_bounds, dtype=np.float64
                )
                endpoint_ids_result = target_ids_result
                endpoint_measures = np.asarray(
                    refinement.target_measures, dtype=np.float64
                )
                endpoint_errors = np.asarray(integration.target_errors, dtype=np.float64)[
                    :, 0
                ]
                endpoint_bounds = np.asarray(
                    integration.target_coordinate_bounds, dtype=np.float64
                )
                root_measure_by_id = root_native
                endpoint_declared_by_id = {
                    int(record[9]): float(record[12]) for record in group_records
                }
            else:
                if set(source_ids_result.tolist()) != {
                    int(record[9]) for record in group_records
                } or set(target_ids_result.tolist()) != {root_id}:
                    raise ValueError(
                        "Curved coarsening result changed source/root stable IDs."
                    )
                root_measures = np.asarray(refinement.target_measures, dtype=np.float64)
                root_errors = np.asarray(integration.target_errors, dtype=np.float64)[
                    :, 0
                ]
                root_bounds = np.asarray(
                    integration.target_coordinate_bounds, dtype=np.float64
                )
                endpoint_ids_result = source_ids_result
                endpoint_measures = np.asarray(
                    refinement.source_measures, dtype=np.float64
                )
                endpoint_errors = np.asarray(integration.source_errors, dtype=np.float64)[
                    :, 0
                ]
                endpoint_bounds = np.asarray(
                    integration.source_coordinate_bounds, dtype=np.float64
                )
                root_measure_by_id = root_native
                endpoint_declared_by_id = {
                    int(record[9]): float(record[12]) for record in group_records
                }
            root_rows = np.asarray(
                refinement.source_cell_global_ids
                if root_role == "source"
                else refinement.target_cell_global_ids,
                dtype=np.int64,
            )
            for identifier, measured, error, bounds in zip(
                root_rows, root_measures, root_errors, root_bounds, strict=True
            ):
                _check_packet_measure(
                    int(identifier),
                    float(measured),
                    float(error),
                    bounds,
                    float(root_measure_by_id[int(identifier)]),
                    CurvedOverlapIntegrationPolicy(),
                    "resident root",
                )
            for identifier, measured, error, bounds in zip(
                endpoint_ids_result,
                endpoint_measures,
                endpoint_errors,
                endpoint_bounds,
                strict=True,
            ):
                _check_packet_measure(
                    int(identifier),
                    float(measured),
                    float(error),
                    bounds,
                    endpoint_declared_by_id[int(identifier)],
                    CurvedOverlapIntegrationPolicy(),
                    "remote endpoint",
                )
            result_record = (
                _RESULT_SCHEMA,
                source_binding,
                target_binding,
                root_binding,
                transition_id,
                root_role,
                root_id,
                local_rank,
                endpoint_owner,
                group_records,
                refinement,
                integration,
                float(root_native[root_id]),
            )
            if root_role == "target" or endpoint_owner == local_rank:
                local_results.append(result_record)
            else:
                outgoing_results[endpoint_owner].append(result_record)
    except Exception as caught:
        root_processing_error = caught
    owner_local._synchronize_error(
        root_processing_error, mesh, axis, owner_count, "curved root-owner integration"
    )
    if root_role == "source":
        _, incoming_results = owner_local._exchange_curved_root_map_packets(
            tuple(tuple(records) for records in outgoing_results),
            local_rank,
            execution_group,
            mesh,
            axis,
            owner_count,
        )
        tagged_results = [(local_rank, record) for record in local_results]
        tagged_results.extend(
            (sender, record)
            for sender, records in incoming_results.items()
            for record in records
        )
    else:
        tagged_results = [(local_rank, record) for record in local_results]
    tagged_results.sort(
        key=lambda item: (
            int(item[1][6]),
            int(item[1][8]),
            tuple(sorted(int(request[9]) for request in item[1][9])),
        )
    )
    result_root_owners = _directory_cell_owners(
        root_view,
        np.asarray([record[6] for _, record in tagged_results], dtype=np.int64),
        execution_group,
        axis,
    )
    result_directory_error = None
    for row, (sender, record) in enumerate(tagged_results):
        if int(result_root_owners[row]) != sender or int(record[7]) != sender:
            result_directory_error = ValueError(
                "Curved root-overlap result sender differs from the accepted root-cell owner."
            )
            break
    owner_local._synchronize_error(
        result_directory_error,
        mesh,
        axis,
        owner_count,
        "curved root-overlap result owner binding",
    )

    result_error: Exception | None = None
    target_declared_by_id = {
        int(identifier): float(measure)
        for identifier, measure in zip(target_ids, target_declared, strict=True)
    }
    try:
        for sender, record in tagged_results:
            _validate_result_record(
                record,
                sender,
                root_role,
                transition_id,
                source_binding,
                target_binding,
                root_binding,
                map_groups,
                target_view,
                np.asarray(target_owned_ids, dtype=np.int64),
                target_declared_by_id,
                local_rank,
            )
    except Exception as caught:
        result_error = caught
    owner_local._synchronize_error(
        result_error, mesh, axis, owner_count, "curved root-overlap result validation"
    )
    result_records = tuple(record for _, record in tagged_results)
    aggregation_error: Exception | None = None
    aggregate_values: tuple[Any, ...] | None = None
    try:
        aggregate_values = _aggregate_root_results(
            result_records,
            root_role,
            np.asarray(target_owned_ids, dtype=np.int64),
            target_owned_measures,
            target_errors,
            target_bounds,
            source_errors,
            source_bounds,
            source_self_integration,
            target_self_integration,
            tolerance,
            require_complete,
            integration_policy,
            source_binding,
            target_binding,
            transition_id,
        )
    except Exception as caught:
        aggregation_error = caught
    owner_local._synchronize_error(
        aggregation_error,
        mesh,
        axis,
        owner_count,
        "curved root-owner overlap aggregation",
    )
    if aggregate_values is None:
        raise RuntimeError("Curved root-owner aggregation omitted its evidence.")
    (
        local_refinement,
        candidate_ids,
        candidate_owners,
        candidate_measures,
        overlap_errors,
        integration_bundle,
    ) = aggregate_values
    candidate_periodic_ids = np.full(candidate_ids.shape, -1, dtype=np.int64)
    candidate_windings = np.empty((candidate_ids.size, 0), dtype=np.int64)
    candidate_vertices: tuple[np.ndarray, ...] = ()
    candidate_faces = None
    source_local_errors = np.asarray(
        [source_errors[int(identifier)] for identifier in source_owned_ids],
        dtype=np.float64,
    )
    source_local_bounds = np.asarray(
        [source_bounds[int(identifier)] for identifier in source_owned_ids],
        dtype=np.float64,
    ).reshape((-1, 2, 3))
    return (
        local_refinement,
        source_geometry_refinement,
        target_geometry_refinement,
        integration_bundle,
        candidate_ids,
        candidate_owners,
        candidate_measures,
        candidate_vertices,
        candidate_periodic_ids,
        candidate_windings,
        candidate_faces,
        source_owned_measures,
        target_owned_measures,
        source_local_errors,
        source_local_bounds,
        overlap_errors,
    )

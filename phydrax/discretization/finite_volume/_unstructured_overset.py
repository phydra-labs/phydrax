#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class UnstructuredOversetReport(StrictModule):
    maximum_receptor_coverage_defect: Array
    donor_overlap_measure: Array
    receptor_overlap_measure: Array
    maximum_donor_covered_fraction: Array
    receptor_count: Array
    hole_count: Array
    donor_hole_count: Array
    donor_count: Array
    donor_eligible_count: Array
    union_volume_measure: Array
    union_volume: Array
    union_volume_defect: Array
    union_volume_certificate: Array
    coverage_defect: Array
    coverage_status_mask: Array
    coverage_status: str = eqx.field(static=True)
    tolerance_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)


class UnstructuredOversetPlan(StrictModule, NonTrainableState):
    """Immutable conservative donor/receptor overlap map for one mesh epoch.

    ``receptor_cells`` and ``donor_indices`` are local indices into the two
    prepared discretizations. The corresponding global IDs are retained in
    the artifact so that a route map remains auditable when a mesh's local
    ordering changes. Certified interface geometry additionally carries
    explicit physical ``receptor_face_ids``. All route endpoints must be active,
    non-hole cells; receptor routes must additionally be fringe cells.
    """

    donor_topology_id: str = eqx.field(static=True)
    donor_geometry_id: str = eqx.field(static=True)
    receptor_topology_id: str = eqx.field(static=True)
    receptor_geometry_id: str = eqx.field(static=True)
    donor_global_ids: Array
    receptor_global_ids: Array
    donor_route_global_ids: Array
    receptor_route_global_ids: Array
    donor_active_mask: Array
    donor_hole_mask: Array
    donor_fringe_mask: Array
    donor_eligible: Array
    donor_eligibility: Array
    donor_eligible_mask: Array
    receptor_active_mask: Array
    receptor_hole_mask: Array
    receptor_fringe_mask: Array
    receptor_cells: Array
    receptor_mask: Array
    active_mask: Array
    hole_mask: Array
    fringe_mask: Array
    receptor_offsets: Array
    donor_indices: Array
    receptor_routes: Array
    overlap_measures: Array
    receptor_volumes: Array
    donor_covered_measures: Array
    union_volume: Array
    union_volume_certificate: Array
    receptor_face_ids: Array | None
    receptor_face_points: Array | None
    receptor_face_normals: Array | None
    receptor_face_measures: Array | None
    receptor_face_cells: Array | None
    report: UnstructuredOversetReport
    face_artifact_id: str | None = eqx.field(static=True)
    interpolation_policy: str = eqx.field(static=True)
    bounded_interpolation: bool = eqx.field(static=True)
    coverage_status: str = eqx.field(static=True)
    coverage_status_mask: Array
    tolerance: float = eqx.field(static=True)
    tolerance_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    identity: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        donor: UnstructuredFiniteVolumeDiscretization,
        receptor: UnstructuredFiniteVolumeDiscretization,
        receptor_cells: ArrayLike,
        receptor_offsets: ArrayLike,
        donor_indices: ArrayLike,
        overlap_measures: ArrayLike,
        /,
        *,
        hole_mask: ArrayLike | None = None,
        tolerance: float = 1e-10,
        tolerance_id: str | None = None,
        epoch_id: str | int = "0",
        donor_active_mask: ArrayLike | None = None,
        donor_hole_mask: ArrayLike | None = None,
        donor_fringe_mask: ArrayLike | None = None,
        receptor_active_mask: ArrayLike | None = None,
        receptor_hole_mask: ArrayLike | None = None,
        receptor_fringe_mask: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        fringe_mask: ArrayLike | None = None,
        receptor_mask: ArrayLike | None = None,
        donor_eligible: ArrayLike | None = None,
        donor_eligibility: ArrayLike | None = None,
        donor_eligible_mask: ArrayLike | None = None,
        donor_global_ids: ArrayLike | None = None,
        receptor_global_ids: ArrayLike | None = None,
        donor_route_global_ids: ArrayLike | None = None,
        receptor_route_global_ids: ArrayLike | None = None,
        donor_indices_are_global_ids: bool = False,
        receptor_face_ids: ArrayLike | None = None,
        receptor_face_points: ArrayLike | None = None,
        receptor_face_normals: ArrayLike | None = None,
        receptor_face_measures: ArrayLike | None = None,
        receptor_face_cells: ArrayLike | None = None,
        face_artifact_id: str | None = None,
        receptor_cells_are_global_ids: bool = False,
        union_volume_certificate: ArrayLike | None = None,
        interpolation_policy: str = "conservative",
        policy: str | None = None,
        conservation_policy: str | None = None,
        coverage_policy: str | None = None,
        bounded_interpolation: bool | None = None,
        bounded: bool | None = None,
    ):
        if not isinstance(
            donor, UnstructuredFiniteVolumeDiscretization
        ) or not isinstance(receptor, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Overset donor and receptor must be unstructured FV geometry."
            )

        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Overset tolerance must be positive and finite.")
        epoch_ = str(epoch_id)
        if not epoch_:
            raise ValueError("Overset epoch_id must be non-empty.")
        if tolerance_id is None:
            tolerance_id_ = canonical_fingerprint(
                {"kind": "overset-tolerance", "value": tolerance_}
            )
        else:
            tolerance_id_ = str(tolerance_id)
            if not tolerance_id_:
                raise ValueError("Overset tolerance_id must be non-empty.")

        policy_values = []
        if interpolation_policy != "conservative":
            policy_values.append(str(interpolation_policy).strip().lower())
        if policy is not None:
            policy_values.append(str(policy).strip().lower())
        if conservation_policy is not None:
            policy_values.append(str(conservation_policy).strip().lower())
        if coverage_policy is not None:
            policy_values.append(str(coverage_policy).strip().lower())
        bounded_flags = [
            bool(value) for value in (bounded_interpolation, bounded) if value is not None
        ]
        if bounded_flags:
            policy_values.append(
                "conservative_bounded" if bounded_flags[0] else "conservative"
            )
        if not policy_values:
            policy_values = ["conservative"]
        if any(value != policy_values[0] for value in policy_values[1:]):
            raise ValueError("Overset interpolation policies must agree.")
        policy_ = policy_values[0].replace("-", "_")
        if policy_ in {"conservative_unbounded", "unbounded"}:
            policy_ = "conservative"
        elif policy_ in {
            "bounded",
            "bounded_conservative",
            "conservative_bounded",
        }:
            policy_ = "conservative_bounded"
        if policy_ not in {"conservative", "conservative_bounded"}:
            raise ValueError(
                "Overset policy must be conservative; nonconservative policies "
                "are not supported."
            )
        bounded_ = policy_ == "conservative_bounded"
        requested_flags = [
            bool(value) for value in (bounded_interpolation, bounded) if value is not None
        ]
        if requested_flags and any(value != bounded_ for value in requested_flags):
            raise ValueError(
                "bounded interpolation flags conflict with the overset policy."
            )

        donor_mesh_ids = np.asarray(donor.cell_global_ids, dtype=np.int64)
        receptor_mesh_ids = np.asarray(receptor.cell_global_ids, dtype=np.int64)
        if donor_mesh_ids.shape != (donor.cell_count,) or receptor_mesh_ids.shape != (
            receptor.cell_count,
        ):
            raise ValueError("Overset mesh cell global IDs are malformed.")

        receptors_raw = np.asarray(receptor_cells)
        if receptors_raw.ndim != 1 or receptors_raw.dtype.kind not in "iu":
            raise ValueError("receptor_cells must be a one-dimensional integer array.")
        receptors_raw = receptors_raw.astype(np.int64)
        receptor_positions = {
            int(identifier): index for index, identifier in enumerate(receptor_mesh_ids)
        }
        receptor_values_are_global = receptor_cells_are_global_ids or np.any(
            (receptors_raw < 0) | (receptors_raw >= receptor.cell_count)
        )
        if receptor_values_are_global:
            try:
                receptors = np.asarray(
                    [receptor_positions[int(identifier)] for identifier in receptors_raw],
                    dtype=np.int32,
                )
            except KeyError as error:
                raise ValueError(
                    "receptor_cells contains an unknown receptor index or global ID."
                ) from error
        else:
            receptors = receptors_raw.astype(np.int32)
        if (
            receptors.ndim != 1
            or np.any(receptors < 0)
            or np.any(receptors >= receptor.cell_count)
            or np.unique(receptors).size != receptors.size
            or receptors.size == 0
        ):
            raise ValueError("receptor_cells must contain unique valid cell indices.")

        offsets = np.asarray(receptor_offsets)
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu":
            raise ValueError("receptor_offsets must be a one-dimensional integer array.")
        offsets = offsets.astype(np.int32)
        donors_raw = np.asarray(donor_indices)
        if donors_raw.ndim != 1 or donors_raw.dtype.kind not in "iu":
            raise ValueError("donor_indices must be a one-dimensional integer array.")
        donors_raw = donors_raw.astype(np.int64)
        measures = np.asarray(overlap_measures, dtype=float)
        if offsets.shape != (receptors.size + 1,):
            raise ValueError("receptor_offsets must contain one CSR row per receptor.")
        if (
            offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or offsets[-1] != donors_raw.size
            or measures.shape != donors_raw.shape
        ):
            raise ValueError("Overset CSR routes are inconsistent.")
        if np.any(~np.isfinite(measures)) or np.any(measures <= 0.0):
            raise ValueError("Overset overlap measures must be positive and finite.")
        if donors_raw.size == 0:
            raise ValueError("Overset routes cannot be empty.")

        donor_positions = {
            int(identifier): index for index, identifier in enumerate(donor_mesh_ids)
        }
        if donor_indices_are_global_ids or np.any(
            (donors_raw < 0) | (donors_raw >= donor.cell_count)
        ):
            try:
                donors = np.asarray(
                    [donor_positions[int(identifier)] for identifier in donors_raw],
                    dtype=np.int32,
                )
            except KeyError as error:
                raise ValueError(
                    "donor_indices contains an unknown donor index or global ID."
                ) from error
        else:
            donors = donors_raw.astype(np.int32)
        if donor_route_global_ids is not None:
            route_ids = np.asarray(donor_route_global_ids)
            if route_ids.shape != donors.shape or route_ids.dtype.kind not in "iu":
                raise ValueError("donor_route_global_ids must match donor routes.")
            try:
                mapped = np.asarray(
                    [donor_positions[int(identifier)] for identifier in route_ids],
                    dtype=np.int32,
                )
            except KeyError as error:
                raise ValueError(
                    "donor_route_global_ids contains an unknown ID."
                ) from error
            if not np.array_equal(mapped, donors):
                raise ValueError(
                    "donor_indices and donor_route_global_ids identify different cells."
                )
        if receptor_route_global_ids is not None:
            route_ids = np.asarray(receptor_route_global_ids)
            expected = receptors[np.repeat(np.arange(receptors.size), np.diff(offsets))]
            if route_ids.shape != expected.shape or route_ids.dtype.kind not in "iu":
                raise ValueError("receptor_route_global_ids must match receptor routes.")
            if not np.array_equal(
                receptor_mesh_ids[expected], route_ids.astype(np.int64)
            ):
                raise ValueError(
                    "receptor_cells and receptor_route_global_ids identify different cells."
                )

        def _mask(
            name: str, value: ArrayLike | None, count: int, default: bool
        ) -> np.ndarray:
            if value is None:
                return np.full((count,), default, dtype=bool)
            result = np.asarray(value, dtype=bool)
            if result.shape != (count,):
                raise ValueError(f"{name} must contain one flag per cell.")
            return result

        donor_active = _mask(
            "donor_active_mask", donor_active_mask, donor.cell_count, True
        )
        donor_hole_value = donor_hole_mask
        if (
            donor_hole_value is None
            and hole_mask is not None
            and donor.cell_count == receptor.cell_count
            and donor.prepared_id == receptor.prepared_id
        ):
            donor_hole_value = hole_mask
        donor_holes = _mask("donor_hole_mask", donor_hole_value, donor.cell_count, False)
        donor_fringe = _mask(
            "donor_fringe_mask", donor_fringe_mask, donor.cell_count, False
        )
        donor_default_eligible = donor_active & ~donor_holes
        eligibility_values = [
            value
            for value in (donor_eligible, donor_eligibility, donor_eligible_mask)
            if value is not None
        ]
        if len(eligibility_values) > 1:
            reference = np.asarray(eligibility_values[0], dtype=bool)
            if any(
                not np.array_equal(reference, np.asarray(value, dtype=bool))
                for value in eligibility_values[1:]
            ):
                raise ValueError(
                    "donor eligibility aliases must identify the same cells."
                )
        eligibility_input = eligibility_values[0] if eligibility_values else None
        if eligibility_input is None:
            donor_eligible_ = donor_default_eligible
        else:
            donor_eligible_ = np.asarray(eligibility_input, dtype=bool)
            if donor_eligible_.shape != (donor.cell_count,):
                raise ValueError(
                    "donor_eligibility must contain one flag per donor cell."
                )
            if np.any(donor_eligible_ & ~donor_default_eligible):
                raise ValueError(
                    "donor_eligibility cannot include inactive or hole donor cells."
                )

        if active_mask is not None and receptor_active_mask is not None:
            if not np.array_equal(
                np.asarray(active_mask, dtype=bool),
                np.asarray(receptor_active_mask, dtype=bool),
            ):
                raise ValueError("active_mask and receptor_active_mask must agree.")
        receptor_active = _mask(
            "receptor_active_mask",
            receptor_active_mask if receptor_active_mask is not None else active_mask,
            receptor.cell_count,
            True,
        )
        if hole_mask is not None and receptor_hole_mask is not None:
            if not np.array_equal(
                np.asarray(hole_mask, dtype=bool),
                np.asarray(receptor_hole_mask, dtype=bool),
            ):
                raise ValueError("hole_mask and receptor_hole_mask must agree.")
        receptor_holes = _mask(
            "receptor_hole_mask",
            receptor_hole_mask if receptor_hole_mask is not None else hole_mask,
            receptor.cell_count,
            False,
        )
        route_mask = np.zeros((receptor.cell_count,), dtype=bool)
        route_mask[receptors] = True
        if receptor_mask is not None:
            provided_receptor_mask = np.asarray(receptor_mask, dtype=bool)
            if provided_receptor_mask.shape != route_mask.shape or not np.array_equal(
                provided_receptor_mask, route_mask
            ):
                raise ValueError(
                    "receptor_mask must identify exactly the routed receptor cells."
                )
        if fringe_mask is not None and receptor_fringe_mask is not None:
            if not np.array_equal(
                np.asarray(fringe_mask, dtype=bool),
                np.asarray(receptor_fringe_mask, dtype=bool),
            ):
                raise ValueError("fringe_mask and receptor_fringe_mask must agree.")
        receptor_fringe = _mask(
            "receptor_fringe_mask",
            receptor_fringe_mask if receptor_fringe_mask is not None else fringe_mask,
            receptor.cell_count,
            False,
        )
        if receptor_fringe_mask is None and fringe_mask is None:
            receptor_fringe = route_mask.copy()
        if np.any(receptor_holes & route_mask):
            raise ValueError("Overset receptor cells cannot also be holes.")
        if np.any(route_mask & ~receptor_active):
            raise ValueError("Overset routes cannot target inactive receptor cells.")
        if np.any(route_mask & ~receptor_fringe):
            raise ValueError("Overset routes must target fringe receptor cells.")
        if np.any(donors_raw.size and ~donor_eligible_[donors]):
            raise ValueError(
                "Overset routes cannot use inactive, hole, or ineligible donors."
            )

        if donor_global_ids is None:
            donor_ids = donor_mesh_ids.copy()
        else:
            donor_ids = np.asarray(donor_global_ids)
            if donor_ids.shape != (donor.cell_count,) or donor_ids.dtype.kind not in "iu":
                raise ValueError(
                    "donor_global_ids must contain one integer ID per donor cell."
                )
            donor_ids = donor_ids.astype(np.int64)
            if not np.array_equal(donor_ids, donor_mesh_ids):
                raise ValueError("donor_global_ids do not match the donor geometry.")
        if receptor_global_ids is None:
            receptor_ids = receptor_mesh_ids.copy()
        else:
            receptor_ids = np.asarray(receptor_global_ids)
            if (
                receptor_ids.shape != (receptor.cell_count,)
                or receptor_ids.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "receptor_global_ids must contain one integer ID per receptor cell."
                )
            receptor_ids = receptor_ids.astype(np.int64)
            if not np.array_equal(receptor_ids, receptor_mesh_ids):
                raise ValueError(
                    "receptor_global_ids do not match the receptor geometry."
                )

        routes = np.repeat(np.arange(receptors.size, dtype=np.int32), np.diff(offsets))
        receptor_coverage = np.bincount(
            routes, weights=measures, minlength=receptors.size
        )
        receptor_volumes = np.asarray(receptor.cell_volumes, dtype=float)[receptors]
        if np.any(~np.isfinite(receptor_volumes)) or np.any(receptor_volumes <= 0.0):
            raise ValueError("Overset receptor volumes must be positive and finite.")
        defect = receptor_coverage - receptor_volumes
        coverage_limit = tolerance_ * np.maximum(receptor_volumes, 1e-14)
        if np.any(np.abs(defect) > coverage_limit):
            raise ValueError("Overset overlap coverage is incomplete.")
        coverage_status_mask = np.abs(defect) <= coverage_limit

        donor_coverage = np.bincount(donors, weights=measures, minlength=donor.cell_count)
        donor_volumes = np.asarray(donor.cell_volumes, dtype=float)
        if np.any(~np.isfinite(donor_volumes)) or np.any(donor_volumes <= 0.0):
            raise ValueError("Overset donor volumes must be positive and finite.")
        if np.any(
            donor_coverage - donor_volumes > tolerance_ * np.maximum(donor_volumes, 1e-14)
        ):
            raise ValueError("Overset overlap double-counts donor cell measure.")
        if union_volume_certificate is None:
            union_certificate = donor_coverage.copy()
        else:
            union_certificate = np.asarray(union_volume_certificate, dtype=float)
            if union_certificate.shape != donor_volumes.shape:
                raise ValueError(
                    "union_volume_certificate must contain one value per donor cell."
                )
            if np.any(~np.isfinite(union_certificate)) or np.any(
                union_certificate < -tolerance_ * np.maximum(donor_volumes, 1e-14)
            ):
                raise ValueError("Overset union-volume certificate is invalid.")
            if np.any(
                np.abs(union_certificate - donor_coverage)
                > tolerance_ * np.maximum(donor_volumes, 1e-14)
            ):
                raise ValueError(
                    "Overset union-volume certificate does not match route measure."
                )
        union_defect = union_certificate - donor_coverage
        covered_fraction = donor_coverage / donor_volumes
        coverage_status_ = "complete"
        donor_route_ids = donor_ids[donors]
        receptor_route_ids = receptor_ids[receptors][routes]
        face_values = (
            receptor_face_ids,
            receptor_face_points,
            receptor_face_normals,
            receptor_face_measures,
            receptor_face_cells,
        )
        if any(value is not None for value in face_values):
            if any(value is None for value in face_values):
                raise ValueError(
                    "Overset receptor face artifacts require IDs, points, normals, "
                    "measures, and cells together."
                )
            face_ids_raw = np.asarray(receptor_face_ids)
            face_points = np.asarray(receptor_face_points, dtype=float)
            face_normals = np.asarray(receptor_face_normals, dtype=float)
            face_measures = np.asarray(receptor_face_measures, dtype=float)
            face_count = int(face_ids_raw.size)
            if (
                face_ids_raw.ndim != 1
                or face_ids_raw.dtype.kind not in "iu"
                or face_count == 0
                or face_points.ndim != 3
                or face_points.shape[0] != face_count
                or face_points.shape[1] <= 0
                or face_points.shape[2] != receptor.cell_dimension
                or face_normals.shape != face_points.shape
                or face_measures.shape != face_points.shape[:2]
            ):
                raise ValueError(
                    "Overset receptor face artifacts must have shapes "
                    "(F,), (F,Q,d), (F,Q,d), (F,Q), and (F,)."
                )
            face_ids = face_ids_raw.astype(np.int64)
            physical_face_count = int(np.asarray(receptor.face_measures).size)
            if (
                np.any(face_ids < 0)
                or np.any(face_ids >= physical_face_count)
                or np.unique(face_ids).size != face_count
            ):
                raise ValueError(
                    "Overset receptor_face_ids must identify unique physical faces."
                )
            face_cells_raw = np.asarray(receptor_face_cells)
            if (
                face_cells_raw.shape != (face_count,)
                or face_cells_raw.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "Overset receptor face artifacts must have shapes "
                    "(F,), (F,Q,d), (F,Q,d), (F,Q), and (F,)."
                )
            face_cells_raw = face_cells_raw.astype(np.int64)
            if np.all((face_cells_raw >= 0) & (face_cells_raw < receptor.cell_count)):
                face_cells = face_cells_raw.astype(np.int32)
            else:
                try:
                    face_cells = np.asarray(
                        [
                            receptor_positions[int(identifier)]
                            for identifier in face_cells_raw
                        ],
                        dtype=np.int32,
                    )
                except KeyError as error:
                    raise ValueError(
                        "Overset receptor face cells contain an unknown cell ID."
                    ) from error
            routed_cell_set = frozenset(int(value) for value in receptors)
            face_cell_set = frozenset(int(value) for value in face_cells)
            if face_cell_set != routed_cell_set:
                raise ValueError(
                    "Overset receptor face cells must cover exactly the routed "
                    "receptor cells."
                )
            physical_owners = np.asarray(receptor.owner_cells, dtype=np.int32)[face_ids]
            physical_neighbours = np.asarray(receptor.neighbour_cells, dtype=np.int32)[
                face_ids
            ]
            owner_or_neighbour = (physical_owners == face_cells) | (
                physical_neighbours == face_cells
            )
            if not np.all(owner_or_neighbour):
                raise ValueError(
                    "Overset receptor face IDs are not incident to their routed cells."
                )
            reference_points = np.asarray(receptor.face_quadrature_points)[face_ids]
            reference_measures = np.asarray(receptor.face_quadrature_weights)[face_ids]
            reference_vectors = np.asarray(receptor.area_vectors)[face_ids]
            reference_face_measures = np.asarray(receptor.face_measures)[face_ids]
            orientation = np.where(physical_owners == face_cells, 1.0, -1.0)
            reference_normals = orientation[:, None] * (
                reference_vectors / reference_face_measures[:, None]
            )
            reference_normals = np.broadcast_to(
                reference_normals[:, None, :], face_normals.shape
            )
            if (
                reference_points.shape != face_points.shape
                or reference_measures.shape != face_measures.shape
                or not np.allclose(
                    face_points, reference_points, rtol=tolerance_, atol=tolerance_
                )
                or not np.allclose(
                    face_normals, reference_normals, rtol=tolerance_, atol=tolerance_
                )
                or not np.allclose(
                    face_measures, reference_measures, rtol=tolerance_, atol=tolerance_
                )
            ):
                raise ValueError(
                    "Overset receptor face artifacts are stale or do not match the "
                    "identified physical faces."
                )
            normal_norms = np.linalg.norm(face_normals, axis=-1)
            if (
                np.any(~np.isfinite(face_points))
                or np.any(~np.isfinite(face_normals))
                or np.any(~np.isfinite(face_measures))
                or np.any(~np.isfinite(normal_norms))
                or np.any(np.abs(normal_norms - 1.0) > tolerance_)
                or np.any(face_measures <= 0.0)
            ):
                raise ValueError(
                    "Overset receptor face artifacts require finite unit normals "
                    "and positive measures."
                )
            face_artifact_id_ = (
                canonical_fingerprint(
                    {
                        "kind": "unstructured-overset-face-artifact",
                        "donor_geometry": donor.geometry_id,
                        "receptor_geometry": receptor.geometry_id,
                        "epoch_id": epoch_,
                        "face_ids": array_tree_fingerprint(face_ids),
                        "points": array_tree_fingerprint(face_points),
                        "normals": array_tree_fingerprint(face_normals),
                        "measures": array_tree_fingerprint(face_measures),
                        "cells": array_tree_fingerprint(face_cells),
                    }
                )
                if face_artifact_id is None
                else str(face_artifact_id)
            )
            if not face_artifact_id_:
                raise ValueError("face_artifact_id must be non-empty.")
        else:
            if face_artifact_id is not None:
                raise ValueError(
                    "face_artifact_id requires a complete receptor face artifact."
                )
            face_ids = face_points = face_normals = face_measures = face_cells = None
            face_artifact_id_ = None

        self.donor_topology_id = donor.topology_id
        self.donor_geometry_id = donor.geometry_id
        self.receptor_topology_id = receptor.topology_id
        self.receptor_geometry_id = receptor.geometry_id
        self.donor_global_ids = jnp.asarray(donor_ids)
        self.donor_route_global_ids = jnp.asarray(donor_route_ids)
        self.receptor_route_global_ids = jnp.asarray(receptor_route_ids)
        self.receptor_global_ids = jnp.asarray(receptor_ids)
        self.donor_active_mask = jnp.asarray(donor_active)
        self.donor_hole_mask = jnp.asarray(donor_holes)
        self.donor_fringe_mask = jnp.asarray(donor_fringe)
        self.donor_eligibility = jnp.asarray(donor_eligible_)
        self.donor_eligible = jnp.asarray(donor_eligible_)
        self.donor_eligible_mask = jnp.asarray(donor_eligible_)
        self.receptor_active_mask = jnp.asarray(receptor_active)
        self.union_volume = jnp.asarray(union_certificate)
        self.receptor_hole_mask = jnp.asarray(receptor_holes)
        self.receptor_fringe_mask = jnp.asarray(receptor_fringe)
        self.receptor_cells = jnp.asarray(receptors)
        self.receptor_mask = jnp.asarray(route_mask)
        self.active_mask = jnp.asarray(receptor_active)
        self.hole_mask = jnp.asarray(receptor_holes)
        self.fringe_mask = jnp.asarray(receptor_fringe)
        self.receptor_offsets = jnp.asarray(offsets)
        self.donor_indices = jnp.asarray(donors)
        self.receptor_routes = jnp.asarray(routes)
        self.overlap_measures = jnp.asarray(measures)
        self.receptor_volumes = jnp.asarray(receptor_volumes)
        self.donor_covered_measures = jnp.asarray(donor_coverage)
        self.union_volume_certificate = jnp.asarray(union_certificate)
        self.receptor_face_ids = (
            None if face_ids is None else jnp.asarray(face_ids, dtype=jnp.int32)
        )
        self.receptor_face_points = (
            None if face_points is None else jnp.asarray(face_points)
        )
        self.receptor_face_normals = (
            None if face_normals is None else jnp.asarray(face_normals)
        )
        self.receptor_face_measures = (
            None if face_measures is None else jnp.asarray(face_measures)
        )
        self.receptor_face_cells = None if face_cells is None else jnp.asarray(face_cells)
        self.face_artifact_id = face_artifact_id_
        self.interpolation_policy = policy_
        self.coverage_status_mask = jnp.asarray(coverage_status_mask)
        self.bounded_interpolation = bounded_
        self.coverage_status = coverage_status_
        self.tolerance = tolerance_
        self.tolerance_id = tolerance_id_
        self.epoch_id = epoch_
        self.report = UnstructuredOversetReport(
            maximum_receptor_coverage_defect=jnp.asarray(
                np.max(np.abs(defect)) if defect.size else 0.0
            ),
            donor_overlap_measure=jnp.asarray(np.sum(donor_coverage)),
            receptor_overlap_measure=jnp.asarray(np.sum(receptor_coverage)),
            maximum_donor_covered_fraction=jnp.asarray(
                np.max(covered_fraction) if covered_fraction.size else 0.0
            ),
            receptor_count=jnp.asarray(receptors.size, dtype=jnp.int32),
            hole_count=jnp.asarray(np.sum(receptor_holes), dtype=jnp.int32),
            donor_hole_count=jnp.asarray(np.sum(donor_holes), dtype=jnp.int32),
            donor_count=jnp.asarray(donor.cell_count, dtype=jnp.int32),
            donor_eligible_count=jnp.asarray(np.sum(donor_eligible_), dtype=jnp.int32),
            coverage_status_mask=jnp.asarray(coverage_status_mask),
            union_volume_measure=jnp.asarray(np.sum(union_certificate)),
            union_volume=jnp.asarray(np.sum(union_certificate)),
            union_volume_defect=jnp.asarray(np.sum(union_defect)),
            union_volume_certificate=jnp.asarray(union_certificate),
            coverage_defect=jnp.asarray(defect),
            coverage_status=coverage_status_,
            tolerance_id=tolerance_id_,
            epoch_id=epoch_,
        )
        identity = canonical_fingerprint(
            {
                "kind": "unstructured-overset-overlap",
                "donor_topology": donor.topology_id,
                "donor_geometry": donor.geometry_id,
                "receptor_topology": receptor.topology_id,
                "receptor_geometry": receptor.geometry_id,
                "donor_global_ids": array_tree_fingerprint(donor_ids),
                "receptor_global_ids": array_tree_fingerprint(receptor_ids),
                "donor_active_mask": array_tree_fingerprint(donor_active),
                "donor_hole_mask": array_tree_fingerprint(donor_holes),
                "donor_fringe_mask": array_tree_fingerprint(donor_fringe),
                "donor_eligible": array_tree_fingerprint(donor_eligible_),
                "receptor_active_mask": array_tree_fingerprint(receptor_active),
                "receptor_hole_mask": array_tree_fingerprint(receptor_holes),
                "receptor_fringe_mask": array_tree_fingerprint(receptor_fringe),
                "receptor_cells": array_tree_fingerprint(receptors),
                "receptor_offsets": array_tree_fingerprint(offsets),
                "donor_route_global_ids": array_tree_fingerprint(donor_route_ids),
                "receptor_route_global_ids": array_tree_fingerprint(receptor_route_ids),
                "donor_indices": array_tree_fingerprint(donors),
                "overlap_measures": array_tree_fingerprint(measures),
                "union_volume_certificate": array_tree_fingerprint(union_certificate),
                "face_artifact_id": face_artifact_id_,
                "face_ids": (
                    None if face_ids is None else array_tree_fingerprint(face_ids)
                ),
                "face_points": (
                    None if face_points is None else array_tree_fingerprint(face_points)
                ),
                "face_normals": (
                    None if face_normals is None else array_tree_fingerprint(face_normals)
                ),
                "face_measures": (
                    None
                    if face_measures is None
                    else array_tree_fingerprint(face_measures)
                ),
                "face_cells": (
                    None if face_cells is None else array_tree_fingerprint(face_cells)
                ),
                "policy": policy_,
                "tolerance": tolerance_,
                "tolerance_id": tolerance_id_,
                "epoch_id": epoch_,
            }
        )
        self.identity = identity
        self.plan_id = identity

    def validate_geometry(
        self,
        donor: UnstructuredFiniteVolumeDiscretization,
        receptor: UnstructuredFiniteVolumeDiscretization,
        /,
    ) -> None:
        """Reject use of this immutable map with a stale mesh geometry."""
        if not isinstance(
            donor, UnstructuredFiniteVolumeDiscretization
        ) or not isinstance(receptor, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Overset geometry must be unstructured FV geometry.")
        if (
            donor.topology_id != self.donor_topology_id
            or donor.geometry_id != self.donor_geometry_id
        ):
            raise ValueError("Overset donor geometry is stale.")
        if (
            receptor.topology_id != self.receptor_topology_id
            or receptor.geometry_id != self.receptor_geometry_id
        ):
            raise ValueError("Overset receptor geometry is stale.")

    def interpolate(self, donor_cell_averages: ArrayLike, /) -> Array:
        donor = jnp.asarray(donor_cell_averages)
        if donor.ndim < 1 or donor.shape[0] != self.donor_eligible.size:
            raise ValueError("Overset donor values must begin with donor cell count.")
        trailing = (1,) * (donor.ndim - 1)
        weighted = donor[self.donor_indices] * self.overlap_measures.astype(
            donor.dtype
        ).reshape((-1,) + trailing)
        receptor = jnp.zeros(
            (self.receptor_cells.size,) + donor.shape[1:], dtype=donor.dtype
        )
        receptor = receptor.at[self.receptor_routes].add(weighted)
        interpolated = receptor / self.receptor_volumes.astype(donor.dtype).reshape(
            (-1,) + trailing
        )
        if self.bounded_interpolation:
            lower = jnp.full(receptor.shape, jnp.inf, dtype=interpolated.dtype)
            upper = jnp.full(receptor.shape, -jnp.inf, dtype=interpolated.dtype)
            donor_routes = donor[self.donor_indices]
            lower = lower.at[self.receptor_routes].min(donor_routes)
            upper = upper.at[self.receptor_routes].max(donor_routes)
            interpolated = jnp.minimum(jnp.maximum(interpolated, lower), upper)
        return interpolated

    def apply(
        self, receptor_cell_averages: ArrayLike, donor_cell_averages: ArrayLike, /
    ) -> Array:
        receptor = jnp.asarray(receptor_cell_averages)
        if receptor.ndim < 1 or receptor.shape[0] != self.receptor_mask.size:
            raise ValueError(
                "Overset receptor values must begin with receptor cell count."
            )
        interpolated = self.interpolate(donor_cell_averages)
        return receptor.at[self.receptor_cells].set(interpolated)

    def conservation_defect(
        self, donor_cell_averages: ArrayLike, interpolated_receptor: ArrayLike, /
    ) -> Array:
        donor = jnp.asarray(donor_cell_averages)
        receptor = jnp.asarray(interpolated_receptor)
        if donor.ndim < 1 or donor.shape[0] != self.donor_eligible.size:
            raise ValueError("Overset donor values must begin with donor cell count.")
        if receptor.ndim < 1:
            raise ValueError("Overset receptor values must have a cell axis.")
        if receptor.shape[0] == self.receptor_mask.size:
            receptor = receptor[self.receptor_cells]
        elif receptor.shape[0] != self.receptor_cells.size:
            raise ValueError(
                "Overset receptor values must contain either all cells or routed cells."
            )
        trailing = (1,) * (donor.ndim - 1)
        donor_integral = jnp.sum(
            donor[self.donor_indices]
            * self.overlap_measures.astype(donor.dtype).reshape((-1,) + trailing),
            axis=0,
        )
        receptor_integral = jnp.sum(
            receptor
            * self.receptor_volumes.astype(receptor.dtype).reshape(
                (-1,) + (1,) * (receptor.ndim - 1)
            ),
            axis=0,
        )
        return receptor_integral - donor_integral


class PeriodicSlidingCoupling(StrictModule, NonTrainableState):
    """One immutable, conservative overlap map for a frozen stage interval.

    The map is deliberately a value object.  A new shift always creates a new
    coupling and therefore cannot change a map captured by an in-flight stage
    or retry.
    """

    left_routes: Array
    right_routes: Array
    overlap_measures: Array
    left_measures: Array
    right_measures: Array
    normalized_shift: float = eqx.field(static=True)
    shift_precision: int = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    coverage_error: Array
    coverage_passed: Array
    passed: Array
    status: Array
    conservation_defect: Array

    def interpolate_left_to_right(self, left_values: ArrayLike, /) -> Array:
        left = jnp.asarray(left_values)
        if left.shape[0] != self.left_measures.size:
            raise ValueError("Sliding values must begin with left interval count.")
        trailing = (1,) * (left.ndim - 1)
        weighted = left[self.left_routes] * self.overlap_measures.astype(
            left.dtype
        ).reshape((-1,) + trailing)
        right = jnp.zeros((self.right_measures.size,) + left.shape[1:], dtype=left.dtype)
        right = right.at[self.right_routes].add(weighted)
        return right / self.right_measures.astype(left.dtype).reshape((-1,) + trailing)

    def right_integrated_flux(self, left_flux_density: ArrayLike, /) -> Array:
        left = jnp.asarray(left_flux_density)
        if left.shape[0] != self.left_measures.size:
            raise ValueError("Sliding flux must begin with left interval count.")
        trailing = (1,) * (left.ndim - 1)
        contributions = -left[self.left_routes] * self.overlap_measures.astype(
            left.dtype
        ).reshape((-1,) + trailing)
        right = jnp.zeros((self.right_measures.size,) + left.shape[1:], dtype=left.dtype)
        return right.at[self.right_routes].add(contributions)

    def integrated_seam_flux(
        self, left_flux_density: ArrayLike, step_size: ArrayLike = 1.0, /
    ) -> tuple[Array, Array]:
        """Return left/right integrated seam fluxes with exact opposite signs.

        ``left_flux_density`` is the only numerical flux input.  The returned
        pair is an interface ledger, not an additional source term.
        """
        left = jnp.asarray(left_flux_density)
        step = jnp.asarray(step_size, dtype=left.dtype).reshape(())
        left_integrated = (
            left
            * self.left_measures.astype(left.dtype).reshape(
                (-1,) + (1,) * (left.ndim - 1)
            )
            * step
        )
        right_integrated = self.right_integrated_flux(left) * step
        return left_integrated, right_integrated

    def flux_conservation_defect(
        self, left_flux_density: ArrayLike, right_integrated_flux: ArrayLike, /
    ) -> Array:
        left = jnp.asarray(left_flux_density)
        right = jnp.asarray(right_integrated_flux)
        left_integrated = jnp.sum(
            left
            * self.left_measures.astype(left.dtype).reshape(
                (-1,) + (1,) * (left.ndim - 1)
            ),
            axis=0,
        )
        return left_integrated + jnp.sum(right, axis=0)


class PeriodicSlidingRefreshArtifact(StrictModule, NonTrainableState):
    """Typed accepted-boundary artifact for one frozen sliding-map refresh."""

    content_state: Any
    remap: PeriodicSlidingCoupling
    metrics: PeriodicSlidingCoupling
    evidence: PeriodicSlidingCoupling
    status: Array
    result_id: str = eqx.field(static=True)


class PeriodicSlidingInterfacePlan(StrictModule, NonTrainableState):
    """Accepted-step periodic overlap rebuild for a one-dimensional sliding seam.

    ``coupling`` is a pure host-side rebuild.  It is intentionally not callable
    from a stage kernel: a runtime captures one returned coupling for all three
    stages and all retries, then requests the next one only after acceptance.
    """

    left_breaks: Array
    right_breaks: Array
    period: float = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)
    shift_precision: int = eqx.field(static=True)
    coverage_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_breaks: ArrayLike,
        right_breaks: ArrayLike,
        period: float,
        /,
        *,
        interface_id: str,
        shift_precision: int = 14,
        coverage_tolerance: float = 1e-12,
    ):
        period_ = float(period)
        left = np.asarray(left_breaks, dtype=float)
        right = np.asarray(right_breaks, dtype=float)
        precision = int(shift_precision)
        tolerance = float(coverage_tolerance)
        if not np.isfinite(period_) or period_ <= 0.0:
            raise ValueError("Sliding period must be positive and finite.")
        if (
            isinstance(shift_precision, bool)
            or not isinstance(shift_precision, (int, np.integer))
            or precision < 0
            or precision > 17
        ):
            raise ValueError("shift_precision must be an integer in [0, 17].")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("coverage_tolerance must be positive and finite.")
        for name, breaks in (("left", left), ("right", right)):
            if (
                breaks.ndim != 1
                or breaks.size < 2
                or np.any(~np.isfinite(breaks))
                or not np.isclose(breaks[0], 0.0)
                or not np.isclose(breaks[-1], period_)
                or np.any(np.diff(breaks) <= 0.0)
            ):
                raise ValueError(
                    f"Sliding {name} breaks must strictly partition [0, period]."
                )
        identifier = str(interface_id)
        if not identifier:
            raise ValueError("interface_id must be non-empty.")
        self.left_breaks = jnp.asarray(left)
        self.right_breaks = jnp.asarray(right)
        self.period = period_
        self.interface_id = identifier
        self.shift_precision = precision
        self.coverage_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-sliding-interface",
                "schema_version": 2,
                "left_breaks": array_tree_fingerprint(left),
                "right_breaks": array_tree_fingerprint(right),
                "period": period_,
                "interface_id": identifier,
                "shift_precision": precision,
                "coverage_tolerance": tolerance,
            }
        )

    def evaluate_shift(self, time: ArrayLike, args: object = None, /) -> float:
        """Evaluate a boundary shift without mutating the frozen coupling.

        Callers may provide a scalar ``args`` value, a mapping containing
        ``sliding_shift``, or a callable under that key.  The value is evaluated
        only by the accepted-step host transaction; stage kernels never call it.
        """
        value: object = args
        if isinstance(args, dict):
            for key in ("sliding_shift", "motion_shift", "shift"):
                if key in args:
                    value = args[key]
                    break
        if callable(value):
            provider = cast(Callable[[ArrayLike, object], ArrayLike], value)
            value = provider(time, args)
        if value is None:
            value = 0.0
        return self.normalize_shift(cast(ArrayLike, value))

    def normalize_shift(self, shift: ArrayLike, /) -> float:
        """Normalize a physical shift deterministically into ``[0, period)``."""
        value = float(np.asarray(shift).reshape(()))
        if not np.isfinite(value):
            raise ValueError("Sliding shift must be finite.")
        normalized = round((value % self.period) / self.period, self.shift_precision)
        normalized *= self.period
        if np.isclose(
            normalized,
            self.period,
            rtol=0.0,
            atol=self.coverage_tolerance,
        ):
            normalized = 0.0
        return float(normalized)

    def coupling(self, shift: ArrayLike, /) -> PeriodicSlidingCoupling:
        """Build and certify the overlap map for one accepted-boundary shift."""
        normalized_shift = self.normalize_shift(shift)
        left = np.asarray(self.left_breaks)
        right = np.asarray(self.right_breaks)
        left_routes = []
        right_routes = []
        overlap = []
        for right_cell in range(right.size - 1):
            start = right[right_cell] + normalized_shift
            stop = right[right_cell + 1] + normalized_shift
            pieces = []
            if stop <= self.period:
                pieces.append((start, stop))
            elif start >= self.period:
                pieces.append((start - self.period, stop - self.period))
            else:
                pieces.append((start, self.period))
                pieces.append((0.0, stop - self.period))
            for piece_start, piece_stop in pieces:
                for left_cell in range(left.size - 1):
                    measure = max(
                        0.0,
                        min(piece_stop, left[left_cell + 1])
                        - max(piece_start, left[left_cell]),
                    )
                    if measure > 0.0:
                        left_routes.append(left_cell)
                        right_routes.append(right_cell)
                        overlap.append(measure)
        left_routes_ = np.asarray(left_routes, dtype=np.int32)
        right_routes_ = np.asarray(right_routes, dtype=np.int32)
        overlap_ = np.asarray(overlap, dtype=float)
        left_measures = np.diff(left)
        right_measures = np.diff(right)
        left_coverage = np.bincount(
            left_routes_, weights=overlap_, minlength=left_measures.size
        )
        right_coverage = np.bincount(
            right_routes_, weights=overlap_, minlength=right_measures.size
        )
        coverage_error = float(
            max(
                np.max(np.abs(left_coverage - left_measures), initial=0.0),
                np.max(np.abs(right_coverage - right_measures), initial=0.0),
            )
        )
        coverage_passed = bool(coverage_error <= self.coverage_tolerance)
        if not coverage_passed:
            raise ValueError(
                "Sliding overlap rebuild failed periodic coverage "
                f"(maximum defect={coverage_error:.17g})."
            )
        evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-sliding-overlap-evidence",
                "plan": self.plan_id,
                "shift_hex": float(normalized_shift).hex(),
                "shift_precision": self.shift_precision,
                "coverage_tolerance": self.coverage_tolerance,
                "coverage_error": float(coverage_error).hex(),
                "left_coverage": array_tree_fingerprint(left_coverage),
                "right_coverage": array_tree_fingerprint(right_coverage),
            }
        )
        coupling_id = canonical_fingerprint(
            {
                "kind": "periodic-sliding-coupling",
                "schema_version": 2,
                "plan": self.plan_id,
                "shift_hex": float(normalized_shift).hex(),
                "shift_precision": self.shift_precision,
                "left_routes": array_tree_fingerprint(left_routes_),
                "right_routes": array_tree_fingerprint(right_routes_),
                "overlap": array_tree_fingerprint(overlap_),
                "evidence": evidence_id,
            }
        )
        return PeriodicSlidingCoupling(
            left_routes=jnp.asarray(left_routes_),
            right_routes=jnp.asarray(right_routes_),
            overlap_measures=jnp.asarray(overlap_),
            left_measures=jnp.asarray(left_measures),
            right_measures=jnp.asarray(right_measures),
            normalized_shift=normalized_shift,
            shift_precision=self.shift_precision,
            coupling_id=coupling_id,
            evidence_id=evidence_id,
            coverage_error=jnp.asarray(coverage_error),
            coverage_passed=jnp.asarray(coverage_passed),
            passed=jnp.asarray(coverage_passed),
            status=jnp.asarray(0 if coverage_passed else 1, dtype=jnp.int32),
            conservation_defect=jnp.asarray(0.0),
        )


__all__ = [
    "PeriodicSlidingCoupling",
    "PeriodicSlidingInterfacePlan",
    "UnstructuredOversetPlan",
    "UnstructuredOversetReport",
]

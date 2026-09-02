#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...linalg import ArraySpace, DiagonalPairing
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySet
from ..spatial import DyadicCellTopology
from ..spatial._morton import _morton_decode_host, _morton_encode_host
from ._geometry_protocol import FiniteVolumeFaceBlock


class DyadicFiniteVolumeDiscretization(AbstractPreparedDiscretization):
    """Explicit conservative face geometry lowered from dyadic leaves."""

    topology: DyadicCellTopology
    topology_leaf_slots: Array
    cell_centers: Array
    cell_volumes: Array
    cell_quadrature_points: Array
    cell_quadrature_weights: Array
    cell_quadrature_valid: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    face_quadrature_points: Array
    face_quadrature_weights: Array
    owner_cells: Array
    owner_signs: Array
    neighbour_cells: Array
    boundary_patch_ids: Array
    face_block: FiniteVolumeFaceBlock
    face_blocks: tuple[FiniteVolumeFaceBlock, ...]
    cell_space: DiscreteFieldSpace
    face_space: DiscreteFieldSpace
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    preparation: PreparationReport
    boundary_patch_names: tuple[str, ...] = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    cell_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.cell_centers.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.face_centers.shape[0])

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.cell_count, self.component_count


class DyadicFiniteVolumePlan(AbstractDiscretizationPlan):
    """Lower a qualified dyadic leaf partition to explicit finite-volume faces."""

    topology: DyadicCellTopology
    component_names: tuple[str, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: DyadicCellTopology,
        /,
        *,
        component_names: Sequence[str] = ("value",),
        field_name: str = "state",
    ) -> None:
        if not isinstance(topology, DyadicCellTopology):
            raise TypeError("topology must be DyadicCellTopology.")
        if not bool(topology.evidence.successful) or not bool(topology.evidence.covering):
            raise ValueError(
                "Dyadic finite volume requires a successful covering topology."
            )
        components = tuple(str(value).strip() for value in component_names)
        field = str(field_name).strip()
        if not components or any(not value for value in components):
            raise ValueError("component_names must be nonempty strings.")
        if len(set(components)) != len(components):
            raise ValueError("component_names must be unique.")
        if not field:
            raise ValueError("field_name must be nonempty.")
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.MATRIX_FREE,
        )
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "component_names", components)
        object.__setattr__(self, "field_name", field)
        object.__setattr__(
            self,
            "key",
            DiscretizationKey("dyadic_finite_volume", DiscretizationRole.PHYSICAL),
        )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "dyadic-finite-volume-plan",
                    "topology": topology.topology_id,
                    "component_names": list(components),
                    "field_name": field,
                }
            ),
        )

    def prepare(self, *, numeric_version: str = "0") -> DyadicFiniteVolumeDiscretization:
        topology = self.topology
        leaf_slots = np.flatnonzero(np.asarray(topology.leaf_active)).astype(np.int32)
        lower = np.asarray(topology.cell_lower)[leaf_slots]
        upper = np.asarray(topology.cell_upper)[leaf_slots]
        centers = np.asarray(topology.cell_centers)[leaf_slots]
        volumes = np.asarray(topology.cell_volumes)[leaf_slots]
        dimension = topology.address_plan.dimension
        domain_lower = np.asarray(topology.address_plan.lower)
        domain_upper = np.asarray(topology.address_plan.upper)
        periodic = topology.address_plan.periodic_axes
        scale = max(float(np.max(domain_upper - domain_lower)), 1.0)
        tolerance = 128.0 * np.finfo(float).eps * scale
        faces: list[tuple[int, int, int, np.ndarray, np.ndarray, float]] = []
        leaf_levels = np.asarray(topology.levels, dtype=np.int32)[leaf_slots]
        leaf_prefixes = np.asarray(topology.prefixes, dtype=np.uint64)[leaf_slots]
        key_to_cell = {
            (int(leaf_levels[cell]), int(leaf_prefixes[cell])): cell
            for cell in range(leaf_slots.size)
        }
        resolution = topology.address_plan.resolution
        maximum_depth = topology.address_plan.maximum_depth

        def containing_cell(coordinate: tuple[int, ...]) -> int | None:
            for level in range(maximum_depth, -1, -1):
                shift = maximum_depth - level
                level_coordinate = tuple(value >> shift for value in coordinate)
                prefix = _morton_encode_host(level_coordinate, dimension, level)
                cell = key_to_cell.get((level, prefix))
                if cell is not None:
                    return cell
            return None

        patch_names: list[str] = []
        patch_by_axis_side: dict[tuple[int, int], int] = {}
        axis_names = ("x", "y", "z")
        for axis in range(dimension):
            if periodic[axis]:
                continue
            for side, label in ((-1, "min"), (1, "max")):
                patch_by_axis_side[(axis, side)] = len(patch_names)
                patch_names.append(f"{axis_names[axis]}_{label}")

        for owner in range(leaf_slots.size):
            level = int(leaf_levels[owner])
            prefix = int(leaf_prefixes[owner])
            level_coordinate = _morton_decode_host(prefix, dimension, level)
            width = 1 << (maximum_depth - level)
            integer_lower = tuple(value * width for value in level_coordinate)
            for axis in range(dimension):
                transverse = tuple(value for value in range(dimension) if value != axis)
                area = float(
                    np.prod(
                        [
                            upper[owner, value] - lower[owner, value]
                            for value in transverse
                        ]
                    )
                )
                if not periodic[axis] and integer_lower[axis] == 0:
                    face_center = centers[owner].copy()
                    face_center[axis] = domain_lower[axis]
                    normal = np.zeros((dimension,), dtype=float)
                    normal[axis] = -1.0
                    faces.append(
                        (
                            owner,
                            -1,
                            patch_by_axis_side[(axis, -1)],
                            face_center,
                            normal,
                            area,
                        )
                    )

                positive_coordinate = integer_lower[axis] + width
                if not periodic[axis] and positive_coordinate == resolution:
                    face_center = centers[owner].copy()
                    face_center[axis] = domain_upper[axis]
                    normal = np.zeros((dimension,), dtype=float)
                    normal[axis] = 1.0
                    faces.append(
                        (
                            owner,
                            -1,
                            patch_by_axis_side[(axis, 1)],
                            face_center,
                            normal,
                            area,
                        )
                    )
                    continue

                sample = [integer_lower[value] + width // 2 for value in range(dimension)]
                sample[axis] = (
                    0 if positive_coordinate == resolution else positive_coordinate
                )
                neighbour = containing_cell(tuple(sample))
                if neighbour is None:
                    raise ValueError(
                        "Covering dyadic topology has an unresolved face neighbour."
                    )
                neighbours = {neighbour}
                if int(leaf_levels[neighbour]) > level:
                    half_width = width // 2
                    neighbours = set()
                    for bits in product((0, 1), repeat=dimension - 1):
                        fine_sample = list(sample)
                        for transverse_axis, bit in zip(transverse, bits, strict=True):
                            fine_sample[transverse_axis] = (
                                integer_lower[transverse_axis]
                                + bit * half_width
                                + half_width // 2
                            )
                        fine_neighbour = containing_cell(tuple(fine_sample))
                        if fine_neighbour is None:
                            raise ValueError(
                                "Balanced dyadic topology has an unresolved "
                                "fine face neighbour."
                            )
                        neighbours.add(fine_neighbour)
                for neighbour in sorted(neighbours):
                    overlap_lower = np.maximum(lower[owner], lower[neighbour])
                    overlap_upper = np.minimum(upper[owner], upper[neighbour])
                    if any(
                        overlap_upper[value] - overlap_lower[value] <= tolerance
                        for value in transverse
                    ):
                        raise ValueError(
                            "Dyadic face neighbours have no transverse overlap."
                        )
                    face_center = centers[owner].copy()
                    for value in transverse:
                        face_center[value] = 0.5 * (
                            overlap_lower[value] + overlap_upper[value]
                        )
                    face_center[axis] = upper[owner, axis]
                    normal = np.zeros((dimension,), dtype=float)
                    normal[axis] = 1.0
                    face_area = float(
                        np.prod(
                            [
                                overlap_upper[value] - overlap_lower[value]
                                for value in transverse
                            ]
                        )
                    )
                    faces.append(
                        (
                            owner,
                            neighbour,
                            -1,
                            face_center,
                            normal,
                            face_area,
                        )
                    )
        faces.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                *tuple(float(value) for value in item[3]),
            )
        )
        owner = np.asarray([item[0] for item in faces], dtype=np.int32)
        neighbour = np.asarray([item[1] for item in faces], dtype=np.int32)
        patch_ids = np.asarray([item[2] for item in faces], dtype=np.int32)
        face_centers = np.asarray([item[3] for item in faces], dtype=float)
        normals = np.asarray([item[4] for item in faces], dtype=float)
        measures = np.asarray([item[5] for item in faces], dtype=float)
        area_vectors = normals * measures[:, None]
        face_ids = np.arange(len(faces), dtype=np.int32)
        geometry_id = canonical_fingerprint(
            {
                "kind": "dyadic-finite-volume-geometry",
                "topology": topology.topology_id,
                "owners": array_tree_fingerprint(owner),
                "neighbours": array_tree_fingerprint(neighbour),
                "areas": array_tree_fingerprint(measures),
            }
        )
        face_block = FiniteVolumeFaceBlock(
            face_ids=jnp.asarray(face_ids),
            owner_cells=jnp.asarray(owner),
            neighbour_cells=jnp.asarray(neighbour),
            boundary_patch_ids=jnp.asarray(patch_ids),
            face_centers=jnp.asarray(face_centers),
            area_vectors=jnp.asarray(area_vectors),
            face_measures=jnp.asarray(measures),
            active_mask=jnp.ones((len(faces),), dtype=bool),
            block_id=canonical_fingerprint(
                {
                    "kind": "dyadic-finite-volume-face-block",
                    "geometry": geometry_id,
                }
            ),
        )
        cell_count = int(leaf_slots.size)
        face_count = int(len(faces))
        cell_entities = EntitySet(
            "dyadic_cells", dimension, np.arange(cell_count, dtype=np.int64)
        )
        face_entities = EntitySet(
            "dyadic_faces", dimension - 1, np.arange(face_count, dtype=np.int64)
        )
        support = DiscreteSupport(topology, dimension, geometry_id)
        components = len(self.component_names)
        cell_shape = (cell_count, components)
        cell_space = DiscreteFieldSpace(
            self.field_name,
            support.support_id,
            EntityDofLayout(
                cell_entities.entity_set_id,
                cell_count,
                cell_count,
                component_shape=(components,),
            ),
            ArraySpace(
                cell_shape,
                pairing=DiagonalPairing(
                    jnp.broadcast_to(jnp.asarray(volumes)[:, None], cell_shape)
                ),
            ),
            representation="cell_average",
            conformity="discontinuous",
            reconstruction_id=canonical_fingerprint(
                {"kind": "dyadic-cell-average", "plan": self.plan_id}
            ),
        )
        face_shape = (face_count, components)
        face_space = DiscreteFieldSpace(
            f"{self.field_name}_face_flux",
            support.support_id,
            EntityDofLayout(
                face_entities.entity_set_id,
                face_count,
                face_count,
                component_shape=(components,),
            ),
            ArraySpace(
                face_shape,
                pairing=DiagonalPairing(
                    jnp.broadcast_to(jnp.asarray(measures)[:, None], face_shape)
                ),
            ),
            representation="flux_moment",
            conformity="Hdiv",
            trace_space_id=cell_space.field_space_id,
        )
        preparation = PreparationReport(
            capabilities=self.capabilities,
            diagnostics=(
                "dyadic leaf measures are positive",
                "coarse-fine faces are decomposed conservatively",
                "face area vectors point outward from owners",
                "boundary patches are complete",
            ),
            resource_counts={
                "cells": cell_count,
                "faces": face_count,
                "boundary_faces": int(np.sum(neighbour < 0)),
            },
        )
        measure_metadata = (
            DiscreteMeasure(
                "dyadic_cell_measure",
                support.support_id,
                cell_entities.entity_set_id,
                jnp.asarray(volumes),
            ),
            DiscreteMeasure(
                "dyadic_face_measure",
                support.support_id,
                face_entities.entity_set_id,
                jnp.asarray(measures),
            ),
        )
        spaces, measure_values, capabilities = validate_prepared_metadata(
            key=self.key,
            support=support,
            field_spaces=(cell_space, face_space),
            measures=measure_metadata,
            capabilities=self.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be nonempty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dyadic-finite-volume",
                "plan": self.plan_id,
                "geometry": geometry_id,
                "numeric_version": version,
            }
        )
        return DyadicFiniteVolumeDiscretization(
            topology=topology,
            topology_leaf_slots=jnp.asarray(leaf_slots),
            cell_centers=jnp.asarray(centers),
            cell_volumes=jnp.asarray(volumes),
            cell_quadrature_points=jnp.asarray(centers[:, None, :]),
            cell_quadrature_weights=jnp.asarray(volumes[:, None]),
            cell_quadrature_valid=jnp.ones((leaf_slots.size, 1), dtype=bool),
            face_centers=jnp.asarray(face_centers),
            area_vectors=jnp.asarray(area_vectors),
            face_measures=jnp.asarray(measures),
            face_quadrature_points=jnp.asarray(face_centers[:, None, :]),
            face_quadrature_weights=jnp.asarray(measures[:, None]),
            owner_cells=jnp.asarray(owner),
            owner_signs=jnp.ones((len(faces),), dtype=jnp.int8),
            neighbour_cells=jnp.asarray(neighbour),
            boundary_patch_ids=jnp.asarray(patch_ids),
            face_block=face_block,
            face_blocks=(face_block,),
            cell_space=cell_space,
            face_space=face_space,
            key=self.key,
            support=support,
            field_spaces=spaces,
            measures=measure_values,
            capabilities=capabilities,
            preparation=preparation,
            boundary_patch_names=tuple(patch_names),
            component_names=self.component_names,
            cell_dimension=dimension,
            topology_id=topology.topology_id,
            geometry_id=geometry_id,
            plan_id=self.plan_id,
            prepared_id=prepared_id,
            numeric_version=version,
        )


__all__ = ["DyadicFiniteVolumeDiscretization", "DyadicFiniteVolumePlan"]

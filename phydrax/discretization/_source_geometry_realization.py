# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Material-reference source realization with retained physical inventory evidence."""

from __future__ import annotations

import math
from contextlib import nullcontext
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry import _global_embedding as imported_geometry__global_embedding
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import (
    checked,
    Float64,
)
from . import (
    _cell_geometry_transition_types as transition_types,
    _mapped_geometry_measure as mapped_measure_owner,
)
from ._cell_geometry import (
    CellGeometrySpec,
    owned_coordinate_preparation,
)
from ._cell_geometry_transition_budget import charged_geometry_transition
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    expression_add,
    expression_bounds,
    expression_scale,
)


_SIMPLEX_DIMENSIONS = {"interval": 1, "triangle": 2, "tetrahedron": 3}


_SOURCE_GEOMETRY_REALIZATION_TOKEN = object()


class SourceGeometryRealization(StrictModule, NonTrainableState):
    """A declared material-reference correspondence between different domains.

    Both embeddings are certified independently. Measures and their absolute
    errors refer to physical cells, not to an inferred common physical domain.
    """

    __strict_contract__ = True

    transition: transition_types.CellGeometryTransition
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    source_cell_measures: Float64[transition_types.GeometryTargetCellDim]
    target_cell_measures: Float64[transition_types.GeometryTargetCellDim]
    source_measure_errors: Float64[transition_types.GeometryTargetCellDim]
    target_measure_errors: Float64[transition_types.GeometryTargetCellDim]
    peak_expression_bytes: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transition: transition_types.CellGeometryTransition,
        source_embedding: GlobalEmbeddingCertificate,
        target_embedding: GlobalEmbeddingCertificate,
        source_cell_measures: Float64[transition_types.GeometryTargetCellDim],
        target_cell_measures: Float64[transition_types.GeometryTargetCellDim],
        source_measure_errors: Float64[transition_types.GeometryTargetCellDim],
        target_measure_errors: Float64[transition_types.GeometryTargetCellDim],
        peak_expression_bytes: int,
        /,
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _SOURCE_GEOMETRY_REALIZATION_TOKEN:
            raise TypeError(
                "source_realization.construction must use prepare_source_geometry_realization; got missing preparation token"
            )
        self.transition = transition
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_cell_measures = source_cell_measures
        self.target_cell_measures = target_cell_measures
        self.source_measure_errors = source_measure_errors
        self.target_measure_errors = target_measure_errors
        self.peak_expression_bytes = peak_expression_bytes
        self.realization_id = canonical_fingerprint(
            {
                "kind": "source-geometry-realization-measures",
                "transition": transition.transition_id,
                "source_measures": array_tree_fingerprint(source_cell_measures),
                "target_measures": array_tree_fingerprint(target_cell_measures),
                "source_errors": array_tree_fingerprint(source_measure_errors),
                "target_errors": array_tree_fingerprint(target_measure_errors),
            }
        )
        self.require_current()

    def require_current(self, /) -> None:
        """Validate retained numerical measure evidence after reconstruction."""
        expected = canonical_fingerprint(
            {
                "kind": "source-geometry-realization-measures",
                "transition": self.transition.transition_id,
                "source_measures": array_tree_fingerprint(self.source_cell_measures),
                "target_measures": array_tree_fingerprint(self.target_cell_measures),
                "source_errors": array_tree_fingerprint(self.source_measure_errors),
                "target_errors": array_tree_fingerprint(self.target_measure_errors),
            }
        )
        if expected != self.realization_id:
            raise ValueError(
                "source_realization.measures must equal the prepared physical measures; got different measure arrays"
            )
        source, target = (
            np.asarray(self.source_cell_measures),
            np.asarray(self.target_cell_measures),
        )
        old_errors, new_errors = (
            np.asarray(self.source_measure_errors),
            np.asarray(self.target_measure_errors),
        )
        count = self.transition.target_cell_ids.shape[0]
        if any(
            value.shape != (count,) for value in (source, target, old_errors, new_errors)
        ) or (
            not all(
                np.all(np.isfinite(value))
                for value in (source, target, old_errors, new_errors)
            )
            or np.any(old_errors < 0)
            or np.any(new_errors < 0)
            or np.any(source <= old_errors)
            or np.any(target <= new_errors)
        ):
            raise ValueError(
                "source_realization.measures must have positive finite quantitative enclosures; got invalid physical inventory bounds"
            )


@checked
def require_realization_reference_complex(source: CellMesh, target: CellMesh, /) -> None:
    if source.ambient_dimension != target.ambient_dimension:
        raise ValueError(
            "source_realization.ambient_dimension must retain the declared coordinate space; got different ambient dimensions"
        )
    if source.topology_id != target.topology_id:
        raise ValueError(
            "source_realization.reference_complex must be common to both endpoints; got different reference complexes"
        )
    if len(source.blocks) != len(target.blocks) or not np.array_equal(
        np.asarray(source.vertex_global_ids), np.asarray(target.vertex_global_ids)
    ):
        raise ValueError(
            "source_realization.vertex_ids must retain reference vertex identities; got different vertex IDs"
        )
    for old, new in zip(source.blocks, target.blocks, strict=True):
        if (
            (old.name, old.cell_kind) != (new.name, new.cell_kind)
            or not np.array_equal(np.asarray(old.global_ids), np.asarray(new.global_ids))
            or not np.array_equal(np.asarray(old.vertices), np.asarray(new.vertices))
        ):
            raise ValueError(
                "source_realization.cell_ids must retain oriented reference-cell identities; got different cell IDs or orientations"
            )


def _realization_displacement_bound(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    maximum = 0.0
    changed = False
    old_cells = mapped_measure_owner.mapped_geometry_cells(
        source, source_geometry, budget=budget
    )
    new_cells = mapped_measure_owner.mapped_geometry_cells(
        target, target_geometry, budget=budget
    )
    for (old_element, old_values), (new_element, new_values) in zip(
        old_cells, new_cells, strict=True
    ):
        with budget.temporary_scope():
            old = mapped_measure_owner.mapped_coordinate_expressions(
                old_element, old_values, budget=budget
            )
            new = mapped_measure_owner.mapped_coordinate_expressions(
                new_element, new_values, budget=budget
            )
            domain = (
                "simplex"
                if old_element.cell_kind in _SIMPLEX_DIMENSIONS
                else "prism"
                if old_element.cell_kind == "prism"
                else "box"
            )
            component_bounds = []
            for first, second in zip(old, new, strict=True):
                difference = expression_add(
                    second, expression_scale(first, -1, budget=budget), budget=budget
                )
                changed = changed or bool(difference)
                lower, upper = expression_bounds(
                    difference,
                    domain,
                    source.topological_dimension,
                    budget=budget,
                )
                component_bounds.append(max(abs(lower), abs(upper)))
            squared = sum(
                (Fraction(value) ** 2 for value in component_bounds), Fraction(0)
            )
            bound = float(fraction_sqrt_interval(squared, budget=budget)[1])
            maximum = max(maximum, float(np.nextafter(bound, math.inf)))
    if not changed:
        raise ValueError(
            "source_realization.displacement must change the coordinate source law; got absent displacement expression"
        )
    return maximum


@checked
def prepare_source_geometry_realization(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    source_embedding: imported_geometry__global_embedding.GlobalEmbeddingCertificate,
    target_embedding: imported_geometry__global_embedding.GlobalEmbeddingCertificate,
    policy: transition_types.CellGeometryTransitionPolicy,
    maximum_storage_bytes: int,
    budget: CoordinateEnclosureBudget | None = None,
) -> SourceGeometryRealization:
    """Certify an explicit same-reference material map, not physical equality."""

    if policy.reconstruction != "source_realization":
        raise ValueError(
            "source_realization.policy must select source_realization; got different reconstruction policy"
        )
    source_embedding.binding.require(source_mesh, source_geometry)
    target_embedding.binding.require(target_mesh, target_geometry)
    if source_embedding.status != "certified" or target_embedding.status != "certified":
        raise ValueError(
            "source_realization.embedding must certify both physical domains; got unaccepted embedding certificates"
        )
    require_realization_reference_complex(source_mesh, target_mesh)
    if cell_geometry_id(source_geometry) == cell_geometry_id(target_geometry):
        raise ValueError(
            "source_realization.geometry_id must change the coordinate map; got equal geometry IDs"
        )
    if maximum_storage_bytes < 1 or policy.maximum_evaluations < 4:
        raise ValueError(
            "source_realization.limits must give positive storage and work budgets; got nonpositive limit"
        )
    integration_limit = policy.maximum_evaluations
    with (
        (
            nullcontext(budget)
            if budget is not None
            else owned_coordinate_preparation(
                policy.maximum_evaluations,
                maximum_storage_bytes,
                maximum_geometry_queries=policy.maximum_evaluations,
            )
        ) as budget,
        charged_geometry_transition(policy, budget=budget),
    ):
        ledger = budget
        with charged_geometry_transition(policy, budget=ledger):
            bound = _realization_displacement_bound(
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
                budget=ledger,
            )
            source_measures, source_errors, old_exact = (
                mapped_measure_owner.certified_cell_measures(
                    source_mesh,
                    source_geometry,
                    maximum_work=integration_limit,
                    budget=budget,
                )
            )
            target_measures, target_errors, new_exact = (
                mapped_measure_owner.certified_cell_measures(
                    target_mesh,
                    target_geometry,
                    maximum_work=integration_limit,
                    budget=budget,
                )
            )
        if bound > policy.reconstruction_tolerance:
            raise transition_types.CellGeometryTransitionError(
                "approximation_bound",
                "source_realization.displacement_bound must fit the full-map displacement budget; got excess certified displacement",
                measured=bound,
                limit=policy.reconstruction_tolerance,
            )
        source_total, target_total = (
            math.fsum(source_measures),
            math.fsum(target_measures),
        )
        evidence = transition_types.CellGeometryTransitionEvidence(
            "source_realization",
            exact=False,
            node_count=target_geometry.coordinates.shape[0],
            evaluation_count=ledger.work_units,
            containment_defect=0.0,
            continuity_residual=0.0,
            approximation_bound=bound,
            approximation_tolerance=policy.reconstruction_tolerance,
            rounding_slack=0.0,
            source_measure=source_total,
            target_measure=target_total,
            measure_exact=old_exact and new_exact,
            coverage_defect=None,
            source_measure_error_bound=math.fsum(source_errors)
            + abs(float(np.spacing(source_total))),
            target_measure_error_bound=math.fsum(target_errors)
            + abs(float(np.spacing(target_total))),
        )
        ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
        )
        node_cells = np.full(target_geometry.coordinates.shape[0], -1, dtype=np.int64)
        node_locals = np.full(node_cells.shape, -1, dtype=np.int32)
        offset = 0
        for routes in target_geometry.resolve(target_mesh)[1]:
            for row, route in enumerate(np.asarray(routes, dtype=np.int64)):
                route = np.array(route, dtype=np.int64, copy=False)
                if route.ndim != 1:
                    raise ValueError(
                        "source_realization.routes must have rank-one rows; got incompatible route array"
                    )
                fresh = node_cells[route] < 0
                node_cells[route[fresh]] = offset + row
                node_locals[route[fresh]] = np.flatnonzero(fresh)
            offset += routes.shape[0]
        if np.any(node_cells < 0):
            raise ValueError(
                "source_realization.routes must reference every coordinate coefficient; got unreferenced coefficients"
            )
        dimension = target_mesh.topological_dimension
        empty = jnp.zeros((0,), dtype=jnp.int64)
        transition_id = canonical_fingerprint(
            {
                "kind": "source-geometry-realization",
                "source": cell_geometry_id(source_geometry),
                "target": cell_geometry_id(target_geometry),
                "topology": target_mesh.topology_id,
                "source_embedding": source_embedding.certificate_id,
                "target_embedding": target_embedding.certificate_id,
                "evidence": evidence.evidence_id,
                "policy": policy.policy_id,
            }
        )
        transition = transition_types.CellGeometryTransition(
            source_topology_id=source_mesh.topology_id,
            target_topology_id=target_mesh.topology_id,
            source_geometry_id=cell_geometry_id(source_geometry),
            target_geometry_id=cell_geometry_id(target_geometry),
            geometry=target_geometry,
            vertex_coordinates=target_mesh.coordinates,
            target_cell_ids=jnp.asarray(ids),
            parent_cell_ids=jnp.full(ids.shape, -1, dtype=jnp.int64),
            parent_reference_vertices=jnp.zeros(
                (ids.size, dimension + 1, dimension), dtype=jnp.float64
            ),
            coarsened_cell_ids=empty,
            coarsened_into_ids=empty,
            coarsened_reference_vertices=jnp.zeros(
                (0, dimension + 1, dimension), dtype=jnp.float64
            ),
            node_owner_cells=jnp.asarray(node_cells),
            node_owner_locals=jnp.asarray(node_locals),
            evidence=evidence,
            policy_id=policy.policy_id,
            transition_id=transition_id,
        )
        return SourceGeometryRealization(
            transition,
            source_embedding,
            target_embedding,
            jnp.asarray(source_measures),
            jnp.asarray(target_measures),
            jnp.asarray(source_errors),
            jnp.asarray(target_errors),
            ledger.peak_bytes_upper,
            _construction_token=_SOURCE_GEOMETRY_REALIZATION_TOKEN,
        )

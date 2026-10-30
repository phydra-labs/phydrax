from __future__ import annotations

import math
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import checked
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    add,
    polynomial_bounds,
    scale,
)
from ._mapped_geometry_measure import (
    bernstein_reference_domain,
    certified_cell_measures,
    mapped_coordinate_expressions,
    mapped_geometry_cells,
)
from ._reference_cell import reference_cell_topology


_SOURCE_GEOMETRY_REALIZATION_TOKEN = object()


class SourceGeometryRealization(StrictModule, NonTrainableState):
    """Independent source/target embeddings with a shared reference-cell law."""

    transition: CellGeometryTransition
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    source_cell_measures: Array
    target_cell_measures: Array
    source_measure_errors: Array
    target_measure_errors: Array
    maximum_displacement: float = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transition: CellGeometryTransition,
        source_embedding: GlobalEmbeddingCertificate,
        target_embedding: GlobalEmbeddingCertificate,
        source_cell_measures: Array,
        target_cell_measures: Array,
        source_measure_errors: Array,
        target_measure_errors: Array,
        maximum_displacement: float,
        /,
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _SOURCE_GEOMETRY_REALIZATION_TOKEN:
            raise TypeError(
                "SourceGeometryRealization must be issued by prepare_source_geometry_realization."
            )
        self.transition = transition
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_cell_measures = source_cell_measures
        self.target_cell_measures = target_cell_measures
        self.source_measure_errors = source_measure_errors
        self.target_measure_errors = target_measure_errors
        self.maximum_displacement = maximum_displacement
        self.realization_id = canonical_fingerprint(
            {
                "kind": "source-geometry-realization",
                "transition": transition.transition_id,
                "source_embedding": source_embedding.certificate_id,
                "target_embedding": target_embedding.certificate_id,
                "source_measures": array_tree_fingerprint(source_cell_measures),
                "target_measures": array_tree_fingerprint(target_cell_measures),
                "source_errors": array_tree_fingerprint(source_measure_errors),
                "target_errors": array_tree_fingerprint(target_measure_errors),
                "maximum_displacement": maximum_displacement,
            }
        )
        self.require_current()

    def require_current(self, /) -> None:
        self.transition.require_bound(
            self.transition.source_mesh,
            self.transition.source_geometry,
            self.transition.target_mesh,
            self.transition.target_geometry,
        )
        self.source_embedding.require_current(
            self.transition.source_mesh, self.transition.source_geometry
        )
        self.target_embedding.require_current(
            self.transition.target_mesh, self.transition.target_geometry
        )
        if (
            self.source_embedding.status != "certified"
            or self.target_embedding.status != "certified"
            or self.transition.evidence.kind != "source_realization"
            or self.transition.evidence.approximation_bound != self.maximum_displacement
        ):
            raise ValueError(
                "Source realization requires current certified endpoint proofs."
            )
        source_count = self.source_embedding.cell_count
        target_count = self.target_embedding.cell_count
        arrays = tuple(
            np.asarray(value)
            for value in (
                self.source_cell_measures,
                self.target_cell_measures,
                self.source_measure_errors,
                self.target_measure_errors,
            )
        )
        if (
            arrays[0].shape != (source_count,)
            or arrays[2].shape != (source_count,)
            or arrays[1].shape != (target_count,)
            or arrays[3].shape != (target_count,)
            or any(not np.all(np.isfinite(value)) for value in arrays)
            or np.any(arrays[0] <= arrays[2])
            or np.any(arrays[1] <= arrays[3])
            or np.any(arrays[2] < 0.0)
            or np.any(arrays[3] < 0.0)
        ):
            raise ValueError(
                "Source realization measures require positive finite enclosures."
            )
        expected = canonical_fingerprint(
            {
                "kind": "source-geometry-realization",
                "transition": self.transition.transition_id,
                "source_embedding": self.source_embedding.certificate_id,
                "target_embedding": self.target_embedding.certificate_id,
                "source_measures": array_tree_fingerprint(self.source_cell_measures),
                "target_measures": array_tree_fingerprint(self.target_cell_measures),
                "source_errors": array_tree_fingerprint(self.source_measure_errors),
                "target_errors": array_tree_fingerprint(self.target_measure_errors),
                "maximum_displacement": self.maximum_displacement,
            }
        )
        if expected != self.realization_id:
            raise ValueError(
                "Source realization evidence differs from its retained measures."
            )


def _require_common_reference_complex(source: CellMesh, target: CellMesh, /) -> None:
    if (
        source.topological_dimension != target.topological_dimension
        or source.ambient_dimension != target.ambient_dimension
        or source.topology_id != target.topology_id
        or not np.array_equal(
            np.asarray(source.vertex_global_ids), np.asarray(target.vertex_global_ids)
        )
        or len(source.blocks) != len(target.blocks)
    ):
        raise ValueError("Source realization endpoints must share one reference complex.")
    for old, new in zip(source.blocks, target.blocks, strict=True):
        if (
            (old.name, old.cell_kind) != (new.name, new.cell_kind)
            or not np.array_equal(np.asarray(old.global_ids), np.asarray(new.global_ids))
            or not np.array_equal(np.asarray(old.vertices), np.asarray(new.vertices))
        ):
            raise ValueError(
                "Source realization must preserve oriented reference-cell identity."
            )


def _displacement_bound(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
) -> float:
    source_cells = mapped_geometry_cells(source_mesh, source_geometry)
    target_cells = mapped_geometry_cells(target_mesh, target_geometry)
    if len(source_cells) != len(target_cells):
        raise ValueError("Source realization must preserve cell correspondence.")
    maximum = Fraction(0)
    for source_cell, target_cell in zip(source_cells, target_cells, strict=True):
        old_element, old_local, old_row = source_cell
        new_element, new_local, new_row = target_cell
        if old_element.cell_kind != new_element.cell_kind:
            raise ValueError("Source realization must preserve reference cell kinds.")
        old = mapped_coordinate_expressions(old_element, old_local, cell_row=old_row)
        new = mapped_coordinate_expressions(new_element, new_local, cell_row=new_row)
        dimension = reference_cell_topology(old_element.cell_kind).dimension
        domain = bernstein_reference_domain(old_element.cell_kind)
        component_bounds = []
        for old_component, new_component in zip(old, new, strict=True):
            difference = add(new_component, scale(old_component, -1))
            lower, upper = polynomial_bounds(difference, domain, dimension)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(
                    "Source realization displacement has no finite enclosure."
                )
            component_bounds.append(Fraction.from_float(max(abs(lower), abs(upper))))
        squared = sum((value * value for value in component_bounds), Fraction(0))
        maximum = max(maximum, fraction_sqrt_interval(squared)[1])
    bound = float(maximum)
    if not math.isfinite(bound):
        raise ValueError("Source realization displacement is not representable.")
    while Fraction.from_float(bound) < maximum:
        bound = float(np.nextafter(bound, math.inf))
    return bound


@checked
def prepare_source_geometry_realization(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    source_embedding: GlobalEmbeddingCertificate,
    target_embedding: GlobalEmbeddingCertificate,
    maximum_displacement: float,
    absolute_measure_tolerance: float = 1e-12,
    relative_measure_tolerance: float = 1e-12,
) -> SourceGeometryRealization:
    """Certify explicit cross-domain correspondence on one common reference mesh."""
    if not isinstance(source_embedding, GlobalEmbeddingCertificate) or not isinstance(
        target_embedding, GlobalEmbeddingCertificate
    ):
        raise TypeError("Source realization requires independent embedding certificates.")
    if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError("Source realization endpoints must be CellMesh values.")
    if not isinstance(source_geometry, CellGeometrySpec) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError("Source realization endpoints must retain CellGeometrySpec maps.")
    _require_common_reference_complex(source_mesh, target_mesh)
    if cell_geometry_id(source_geometry) == cell_geometry_id(target_geometry):
        raise ValueError("Source realization must declare a changed coordinate map.")
    source_embedding.require_current(source_mesh, source_geometry)
    target_embedding.require_current(target_mesh, target_geometry)
    if (
        source_embedding.status != "certified"
        or target_embedding.status != "certified"
        or not source_embedding.validity.all_certified
        or not target_embedding.validity.all_certified
    ):
        raise ValueError("Source realization requires certified endpoint embeddings.")
    tolerance = finite_real_scalar(maximum_displacement, "maximum_displacement")
    if tolerance < 0.0:
        raise ValueError("maximum_displacement must be nonnegative.")
    displacement = _displacement_bound(
        source_mesh, source_geometry, target_mesh, target_geometry
    )
    if displacement > tolerance:
        raise ValueError(
            f"Source realization displacement bound {displacement} exceeds {tolerance}."
        )
    source_measures = certified_cell_measures(
        source_mesh,
        source_geometry,
        absolute_tolerance=absolute_measure_tolerance,
        relative_tolerance=relative_measure_tolerance,
    )
    target_measures = certified_cell_measures(
        target_mesh,
        target_geometry,
        absolute_tolerance=absolute_measure_tolerance,
        relative_tolerance=relative_measure_tolerance,
    )
    evidence = CellGeometryTransitionEvidence(
        "source_realization", approximation_bound=displacement
    )
    transition = CellGeometryTransition(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        evidence=evidence,
    )
    return SourceGeometryRealization(
        transition,
        source_embedding,
        target_embedding,
        jnp.asarray(source_measures.values),
        jnp.asarray(target_measures.values),
        jnp.asarray(source_measures.absolute_errors),
        jnp.asarray(target_measures.absolute_errors),
        displacement,
        _construction_token=_SOURCE_GEOMETRY_REALIZATION_TOKEN,
    )


__all__ = ["SourceGeometryRealization", "prepare_source_geometry_realization"]

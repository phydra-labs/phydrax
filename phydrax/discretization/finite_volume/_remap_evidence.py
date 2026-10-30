from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from .._cell_geometry_transition_types import CellGeometryTransition
from .._sphere_chart_records import PreparedSphereChartDeformation
from .._surface_chart_records import PreparedSurfaceChartDeformation
from ._unstructured_remap import UnstructuredRemapReport


type MappedSurfaceDeformation = (
    PreparedSurfaceChartDeformation | PreparedSphereChartDeformation
)


@final
class MappedNestedRemapEvidence(StrictModule, NonTrainableState):
    """Physical coverage report bound to an exact nested geometry witness."""

    transition: CellGeometryTransition
    report: UnstructuredRemapReport
    reference_witness_id: str = eqx.field(static=True)
    geometry_error_bound: float = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transition: CellGeometryTransition,
        report: UnstructuredRemapReport,
        geometry_error_bound: float,
        reason: str,
        /,
    ) -> None:
        if not isinstance(transition, CellGeometryTransition):
            raise TypeError("transition must be a CellGeometryTransition.")
        if not isinstance(report, UnstructuredRemapReport):
            raise TypeError("report must be an UnstructuredRemapReport.")
        bound = float(geometry_error_bound)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("geometry_error_bound must be finite and nonnegative.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        witness_id = (
            transition.reference_map_id
            or transition.refinement_id
            or transition.coarsening_id
        )
        if witness_id is None:
            raise ValueError(
                "Mapped nested evidence requires an exact reference witness."
            )
        self.transition = transition
        self.report = report
        self.reference_witness_id = witness_id
        self.geometry_error_bound = bound
        self.reason = reason.strip()
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mapped-nested-remap-evidence",
                "transition": transition.transition_id,
                "witness": witness_id,
                "geometry_error_bound": bound,
                "report": array_tree_fingerprint(report),
                "reason": self.reason,
            }
        )

    @property
    def passed(self) -> bool:
        return bool(np.asarray(self.report.coverage_complete)) and np.isfinite(
            self.geometry_error_bound
        )


@final
class MappedSurfaceChartRemapEvidence(StrictModule, NonTrainableState):
    """Retain chart-material contents and the physical coverage report."""

    deformation: MappedSurfaceDeformation
    physical_contents: Array
    target_measures: Array
    report: UnstructuredRemapReport
    reason: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        deformation: MappedSurfaceDeformation,
        physical_contents: ArrayLike,
        target_measures: ArrayLike,
        report: UnstructuredRemapReport,
        /,
        *,
        reason: str,
    ) -> None:
        if not isinstance(
            deformation,
            (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation),
        ):
            raise TypeError("deformation must be a prepared surface or sphere chart map.")
        if not isinstance(report, UnstructuredRemapReport):
            raise TypeError("report must be an UnstructuredRemapReport.")
        contents = jnp.asarray(physical_contents)
        measures = jnp.asarray(target_measures)
        source_count = sum(block.cell_count for block in deformation.source_mesh.blocks)
        target_count = sum(block.cell_count for block in deformation.target_mesh.blocks)
        if (
            contents.ndim == 0
            or contents.shape[0] != source_count
            or measures.shape != (target_count,)
            or np.any(~np.isfinite(np.asarray(contents)))
            or np.any(~np.isfinite(np.asarray(measures)))
            or np.any(np.asarray(measures) <= 0.0)
        ):
            raise ValueError(
                "Mapped surface content and target measures must bind the chart endpoints."
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        self.deformation = deformation
        self.physical_contents = contents
        self.target_measures = measures
        self.report = report
        self.reason = reason.strip()
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mapped-surface-chart-remap-evidence",
                "deformation": deformation.deformation_id,
                "physical_contents": array_tree_fingerprint(contents),
                "target_measures": array_tree_fingerprint(measures),
                "report": array_tree_fingerprint(report),
                "reason": self.reason,
            }
        )

    @property
    def passed(self) -> bool:
        return (
            bool(np.asarray(self.report.coverage_complete))
            and self.deformation.source_domain_coverage == "certified"
            and self.deformation.target_domain_coverage == "certified"
        )


@final
class RemapPreparationFailure(StrictModule, NonTrainableState):
    """Retain a failed route's endpoint identities and refusal reason."""

    source_discretization_id: str = eqx.field(static=True)
    target_discretization_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_discretization_id: str,
        target_discretization_id: str,
        reason: str,
        /,
    ) -> None:
        for value, name in (
            (source_discretization_id, "source_discretization_id"),
            (target_discretization_id, "target_discretization_id"),
            (reason, "reason"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        self.source_discretization_id = source_discretization_id.strip()
        self.target_discretization_id = target_discretization_id.strip()
        self.reason = reason.strip()
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "unstructured-remap-preparation-failure",
                "source": self.source_discretization_id,
                "target": self.target_discretization_id,
                "reason": self.reason,
            }
        )

    @property
    def passed(self) -> bool:
        return False


__all__ = [
    "MappedNestedRemapEvidence",
    "MappedSurfaceChartRemapEvidence",
    "RemapPreparationFailure",
]

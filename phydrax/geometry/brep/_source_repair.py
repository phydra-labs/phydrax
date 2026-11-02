#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit, certified completion and repair of external B-Rep source data.

The operation is a geometry-changing opt-in.  It selects and certifies a
bounded completion of the authored incidence and chart sheets, rather than
asserting that an originally gapped trim already described a closed region.
The input model and its import report remain unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import inf, isfinite, nextafter
from numbers import Real

from ..._fingerprint import canonical_fingerprint
from ...linalg._hermitian_spectral import fraction_sqrt_interval
from ._model import (
    _certify_edge_face_gap_bounds,
    _validate_degenerate_coedge_poles,
    BRepEntityId,
    BRepModel,
)


@dataclass(frozen=True, slots=True)
class BRepSourceRepairResult:
    """Repaired source and evidence for its explicit trim interpretation.

    Displacement bounds refer to the declared completed interpretation, not to
    an asserted filled region of the unmodified, potentially gapped source.
    Connector bounds additionally quantify its departure from authored uses.
    """

    model: BRepModel
    source_revision: str
    target_revision: str
    max_displacement: float
    certificate_id: str
    solid_map: dict[BRepEntityId, BRepEntityId]
    face_map: dict[BRepEntityId, BRepEntityId]
    edge_map: dict[BRepEntityId, BRepEntityId]
    vertex_map: dict[BRepEntityId, BRepEntityId]
    changes_interpretation: bool
    completion_semantics: str
    completion_connector_bound: float
    face_source_to_target_bound: float
    face_target_to_source_bound: float
    completion_evidence: object


def _upper_float(value: Fraction, /) -> float:
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("A repair displacement bound is not finite.") from error
    if not isfinite(number):
        raise ValueError("A repair displacement bound is not finite.")
    return nextafter(number, inf) if Fraction.from_float(number) < value else number


def _occurrence_scale(model: BRepModel, /) -> Fraction:
    """Bound all exact occurrence linear maps, including the definition itself."""
    scale = Fraction(1)
    for occurrence in model.geometry.assembly_graph.iter_occurrences():
        matrix = occurrence.placement.matrix
        gram = tuple(
            tuple(
                sum((matrix[k][i] * matrix[k][j] for k in range(3)), Fraction(0))
                for j in range(3)
            )
            for i in range(3)
        )
        squared = max(sum((abs(value) for value in row), Fraction(0)) for row in gram)
        scale = max(scale, fraction_sqrt_interval(squared)[1])
    return scale


def _require_exact_closed(model: BRepModel, /) -> None:
    geometry = model.geometry
    if not geometry.solid_shells or geometry.edge_vertex_gap_bounds is not None:
        raise ValueError("A repaired source must have exactly incident closed solids.")
    _validate_degenerate_coedge_poles(geometry, model.patches)
    gaps = _certify_edge_face_gap_bounds(
        geometry,
        model.patches,
        parameter_bounds=model.parameter_bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    if gaps is None or any(gap != 0.0 for gap in gaps):
        raise ValueError("Repaired coedges require exact full-range surface incidence.")
    for shells in geometry.solid_shells:
        for shell in shells:
            uses: dict[int, list[int]] = defaultdict(list)
            for face, orientation in zip(
                geometry.shell_faces[shell],
                geometry.shell_orientations[shell],
                strict=True,
            ):
                for loop in geometry.face_loops[face]:
                    for coedge in loop:
                        uses[geometry.coedge_edges[coedge]].append(
                            orientation
                            * int(model.orientation[face])
                            * geometry.coedge_senses[coedge]
                        )
            if not uses or any(sorted(signs) != [-1, 1] for signs in uses.values()):
                raise ValueError("Repaired shell incidence is not closed and oriented.")


def _entity_map(
    source: BRepModel, target: BRepModel, kind: str, /
) -> dict[BRepEntityId, BRepEntityId]:
    old = getattr(source, f"{kind}_ids")
    new = getattr(target, f"{kind}_ids")
    if len(old) != len(new):
        raise ValueError("Repair must preserve authored entity identities and ordering.")
    return dict(zip(old, new, strict=True))


def repair_brep_source(model: BRepModel, *, tolerance: float) -> BRepSourceRepairResult:
    """Complete authored trims and repair their source carriers within ``tolerance``.

    The explicitly selected interpretation uses the authored incidence, chart
    sheets and winding to complete supported simple closed strip and annulus
    boundaries.  Its connector and directed face bounds are recorded separately;
    neither completion nor its filled region is claimed to have been exact in
    the original source.  Ambiguous, unsupported, or unproved cases refuse
    without modifying the original model.
    """
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a finite positive real number.")
    budget_float = float(tolerance)
    if not isfinite(budget_float) or budget_float <= 0:
        raise ValueError("tolerance must be a finite positive real number.")
    budget = (
        Fraction(tolerance)
        if isinstance(tolerance, (Fraction, int))
        else Fraction.from_float(budget_float)
    )
    model.require_current()
    revision = model.source_revision

    from ._source_repair_completion import certify_source_completion
    from ._source_repair_target import build_repaired_target

    target = build_repaired_target(model)
    if not isinstance(target, BRepModel) or target.source_id != model.source_id:
        raise ValueError("Repaired source must preserve its source owner.")
    if target.source_revision == revision:
        raise ValueError("Repair did not create a new source revision.")
    source_geometry, target_geometry = model.geometry, target.geometry
    for name in (
        "edge_vertices",
        "coedge_edges",
        "coedge_senses",
        "face_loops",
        "shell_faces",
        "shell_orientations",
        "solid_shells",
    ):
        if getattr(source_geometry, name) != getattr(target_geometry, name):
            raise ValueError("Repair must preserve authored incidence and occurrences.")
    if (
        source_geometry.assembly_graph.identity_payload()
        != target_geometry.assembly_graph.identity_payload()
    ):
        raise ValueError("Repair must preserve the exact assembly occurrence graph.")
    target.require_current()
    _require_exact_closed(target)
    evidence = certify_source_completion(model, target, tolerance=budget)
    local = (
        evidence.max_vertex,
        evidence.max_edge,
        evidence.max_connector,
        evidence.face_source_to_target,
        evidence.face_target_to_source,
    )
    if any(not isinstance(value, Fraction) or value < 0 for value in local):
        raise ValueError(
            "Repair evidence must provide nonnegative exact rational bounds."
        )
    scale = _occurrence_scale(model)
    bounds = tuple(value * scale for value in local)
    achieved = max(bounds)
    if achieved > budget:
        raise ValueError("Certified source repair exceeds the displacement tolerance.")
    if Fraction.from_float(_upper_float(achieved)) > budget:
        raise ValueError(
            "The certified bound cannot be represented within the requested tolerance."
        )
    maps = tuple(
        _entity_map(model, target, kind) for kind in ("solid", "face", "edge", "vertex")
    )
    semantics = "bounded-authored-incidence: simple closed strips and annuli on certified chart sheets"
    certificate = canonical_fingerprint(
        {
            "kind": "brep-source-repair-completion",
            "source": revision,
            "target": target.source_revision,
            "tolerance": (budget.numerator, budget.denominator),
            "changes_interpretation": True,
            "semantics": semantics,
            "completion": evidence.completion_payload,
            "local_bounds": tuple(
                (value.numerator, value.denominator) for value in local
            ),
            "occurrence_scale": (scale.numerator, scale.denominator),
            "maps": tuple(
                tuple((a.index, b.index) for a, b in mapping.items()) for mapping in maps
            ),
        }
    )
    model.require_current()
    return BRepSourceRepairResult(
        target,
        revision,
        target.source_revision,
        _upper_float(achieved),
        certificate,
        maps[0],
        maps[1],
        maps[2],
        maps[3],
        True,
        semantics,
        _upper_float(bounds[2]),
        _upper_float(bounds[3]),
        _upper_float(bounds[4]),
        evidence.completion_payload,
    )

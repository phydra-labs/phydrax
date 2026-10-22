# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native moving-overset qualification with independent oracles and a cold restart.

Every registered profile executes the same chain: native source generation, hole
cutting and donor search, rigid source motion with a refreshed registration, the
declared interpolative state transfer, an accepted composite overset Poisson update,
a separately certified conservative cell-average route, atomic refusals, and a
fresh-process restart. Independent oracles recompute blanking, fringes and donor
admissibility from exact rational predicates, measure the manufactured composite
solution error, and check discrete flux balance. The steady FE field has no invented
time history. The separately accepted ALE route retains real density, material
fractions and spatial history in a second source-authoritative checkpoint; both
readers, renewed proofs, and the continued PDE solve consume the same original
cold-request allowance. Mandatory completion is derived
from the recorded obligations, never assigned.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from traceback import format_exception
from typing import (
    Any,
    cast,
    final,
    NamedTuple,
    Protocol,
    runtime_checkable,
    TypedDict,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import (
    array_tree_fingerprint,
    ArrayArchiveLimits,
    canonical_fingerprint,
    model_array_bytes,
    StrictModule,
)
from phydrax._frozendict import frozendict
from phydrax._meshcore import current_native_execution_scope
from phydrax.discretization import (
    PolygonalConnectivity,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import (
    CoordinateOperandCollector,
    PreparedCoordinateOperands,
)
from phydrax.discretization.finite_volume._remap_evidence import (
    PreparedUnstructuredConservativeRemap,
    RemapPreparationFailure,
)
from phydrax.discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from phydrax.geometry import (
    CommonRefinementEvidence,
)
from phydrax.lifecycle import (
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    CouplingSearchStatus,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    NativeMeshingPhaseMeasurement,
    OversetCellStatus,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_conservative_state_transport,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
)
from phydrax.meshing.providers._native_planar_domain import graded_size_compliance
from phydrax.meshing.qualification import (
    deform_planar_interior,
    execute_native_consumer,
    execute_native_request,
    field_declarations,
    field_entries,
    field_plans,
    poisson_solve_policy,
    prepare_poisson_update,
    rigid_planar_successor,
)
from tools.compile_counter import CompileCounter
from tools.meshing_qualification._controls import (
    _overset_qualification_requirements,
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _quality_summary,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _LoweredProgram,
    _measured,
    _runtime_identity,
    ColdControlPacket,
    ColdWorkerReport,
    ColdWorkerTimeout,
    Deadline,
    epoch_stage_name,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    prepare_cold_worker,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.metric_adaptation import (
    _mesh_triangles,
)
from tools.meshing_qualification.scenarios.triangulation import (
    _simplex_measures,
)


_BACKGROUND = "background"
_BACKGROUND_HALF_WIDTH = 3.0
_MANUFACTURED_SOLUTION = "u=5+x+y-(x^2+y^2)/4, -Laplace(u)=1"
_ERROR_QUANTITY = "independent-manufactured-composite-overset-L2-error"
_SCHWARZ_MAXIMUM_ITERATIONS = 64
_INTERFACE_TOLERANCE = 1.0e-8
_RESIDUAL_TOLERANCE = 1.0e-9
_LIFT_TOLERANCE = 1.0e-12
_BALANCE_TOLERANCE = 1.0e-8
_CONTENT_TOLERANCE = 1.0e-9
_ACCEPTED_BOUNDARY = "accepted-poisson-step"
_COLD_PHASE = "fresh_process_native_restart_ownership_continued_composite_pde"
# Shewchuk's orient2d static filter for binary64 with unit roundoff 2^-53.
_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0
_ORIENT_BOUND = (3.0 + 16.0 * _UNIT_ROUNDOFF) * _UNIT_ROUNDOFF


# ------------------------------------------------------------------ profiles


class _OversetBody(NamedTuple):
    """One authored protected square-annulus body and its translation schedule."""

    name: str
    center: tuple[float, float]
    half_width: float
    cavity_half_width: float
    refinement: int
    offsets: tuple[tuple[float, float], ...]
    offset_rule: str

    def offset(self, step: int, /) -> np.ndarray:
        match self.offset_rule:
            case "cycle":
                value = self.offsets[step % len(self.offsets)]
            case "first-then-repeat-last":
                value = self.offsets[min(step, len(self.offsets) - 1)]
            case rule:
                raise ValueError(
                    f"overset.body.offset_rule must be 'cycle' or 'first-then-repeat-last'; got {rule!r}"
                )
        return np.asarray(value, dtype=np.float64)

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "center": list(self.center),
            "half_width": self.half_width,
            "cavity_half_width": self.cavity_half_width,
            "refinement": self.refinement,
            "offsets": [list(value) for value in self.offsets],
            "offset_rule": self.offset_rule,
        }


class _OversetProfile(NamedTuple):
    name: str
    fringe_layers: int
    bodies: tuple[_OversetBody, ...]

    def record(self) -> dict[str, object]:
        return {
            "fringe_layers": self.fringe_layers,
            "bodies": [body.record() for body in self.bodies],
        }


_OVERSET_PROFILES: dict[str, _OversetProfile] = {
    "routine": _OversetProfile(
        "routine",
        1,
        (
            _OversetBody(
                "moving",
                (0.0, 0.0),
                2.25,
                0.25,
                1,
                ((0.55, 0.0), (0.05, 0.0)),
                "first-then-repeat-last",
            ),
        ),
    ),
    "larger": _OversetProfile(
        "larger",
        1,
        (
            _OversetBody(
                "port", (-1.5, 0.0), 1.25, 0.25, 2, ((0.0, 0.5), (0.0, -0.5)), "cycle"
            ),
            _OversetBody(
                "starboard",
                (1.5, 0.0),
                1.25,
                0.25,
                2,
                ((0.0, -0.5), (0.0, 0.5)),
                "cycle",
            ),
        ),
    ),
    "combined": _OversetProfile(
        "combined",
        2,
        (
            _OversetBody(
                "alpha",
                (-1.15, -0.3),
                1.7,
                0.3,
                1,
                ((0.0, 0.45), (0.0, -0.45)),
                "cycle",
            ),
            _OversetBody(
                "beta",
                (1.55, 0.5),
                1.3,
                0.2,
                2,
                ((-0.05, -0.4), (0.05, 0.4)),
                "cycle",
            ),
        ),
    ),
}


def _overset_profile(args: QualificationControls) -> _OversetProfile:
    """Select the authored profile and refuse drift from its frozen requirements."""
    name = args.overset_profile
    profile = _OVERSET_PROFILES.get(name)
    if profile is None:
        raise ValueError(
            f"overset.profile must be one of {sorted(_OVERSET_PROFILES)!r}; got {name!r}"
        )
    geometry = _overset_qualification_requirements().get("geometry")
    frozen = (
        geometry.get("profiles", {}).get(name) if isinstance(geometry, dict) else None
    )
    if (
        not isinstance(geometry, dict)
        or geometry.get("background_half_width") != _BACKGROUND_HALF_WIDTH
        or frozen != profile.record()
    ):
        raise ValueError(
            f"overset.requirements.geometry must freeze the authored {name!r} profile; got {frozen!r}"
        )
    return profile


def _body_areas(body: _OversetBody | None, /) -> dict[str, float | None]:
    """Analytic represented area and boundary fluxes of the manufactured solution."""
    if body is None:
        width = 2.0 * _BACKGROUND_HALF_WIDTH
        # The exterior outward flux equals the integral of Laplace(u) = -1.
        return {"domain": width * width, "wall_flux": None, "exterior_flux": -width**2}
    outer, cavity = 2.0 * body.half_width, 2.0 * body.cavity_half_width
    # The wall normal points into the cavity: the flux equals the cavity area.
    return {
        "domain": outer * outer - cavity * cavity,
        "wall_flux": cavity * cavity,
        "exterior_flux": None,
    }


# ------------------------------------------------------------ exact predicates


def _exact_orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> np.ndarray:
    """Signs of orient2d(a, b, c), filtered and exactly resolved when uncertain."""
    a, b, c = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
    )
    left = (a[..., 0] - c[..., 0]) * (b[..., 1] - c[..., 1])
    right = (a[..., 1] - c[..., 1]) * (b[..., 0] - c[..., 0])
    determinant = left - right
    bound = _ORIENT_BOUND * (np.abs(left) + np.abs(right))
    signs = np.sign(determinant).astype(np.int8)
    for index in zip(*np.nonzero(~(np.abs(determinant) > bound)), strict=True):
        ax, ay = (Fraction(float(value)) for value in a[index])
        bx, by = (Fraction(float(value)) for value in b[index])
        cx, cy = (Fraction(float(value)) for value in c[index])
        exact = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
        signs[index] = (exact > 0) - (exact < 0)
    return signs


def _counterclockwise(triangles: np.ndarray, /) -> np.ndarray:
    """Exactly orient nondegenerate triangles counterclockwise."""
    signs = _exact_orientation(triangles[:, 0], triangles[:, 1], triangles[:, 2])
    if np.any(signs == 0):
        raise qualification_failure(
            "_counterclockwise",
            ("Overset oracle received an exactly degenerate triangle."),
            {"degenerate": int(np.count_nonzero(signs == 0))},
        )
    oriented = triangles.copy()
    flipped = signs < 0
    oriented[flipped, 1], oriented[flipped, 2] = (
        triangles[flipped, 2],
        triangles[flipped, 1],
    )
    return oriented


def _wall_rectangle(points: np.ndarray, edges: np.ndarray, /) -> tuple[float, ...]:
    """Read the represented wall as one closed axis-aligned rectangle, or refuse."""
    corners = points[edges]
    lower, upper = np.min(corners, axis=(0, 1)), np.max(corners, axis=(0, 1))
    horizontal = corners[:, 0, 1] == corners[:, 1, 1]
    vertical = corners[:, 0, 0] == corners[:, 1, 0]
    on_sides = np.where(
        horizontal,
        np.isin(corners[:, 0, 1], (lower[1], upper[1])),
        np.isin(corners[:, 0, 0], (lower[0], upper[0])),
    )
    length = float(np.sum(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)))
    perimeter = 2.0 * float(np.sum(upper - lower))
    if (
        not edges.size
        or not np.all(horizontal ^ vertical)
        or not np.all(on_sides)
        or abs(length - perimeter) > 1.0e-12 * max(1.0, perimeter)
    ):
        raise qualification_failure(
            "_wall_rectangle",
            (
                "Independent hole oracle admits only closed axis-aligned rectangular walls."
            ),
            {"length": length, "perimeter": perimeter},
        )
    return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))


def _points_in_rectangle(points: np.ndarray, rectangle: tuple[float, ...], /) -> Any:
    xl, yl, xu, yu = rectangle
    return (
        (points[:, 0] >= xl)
        & (points[:, 0] <= xu)
        & (points[:, 1] >= yl)
        & (points[:, 1] <= yu)
    )


def _triangles_touch_rectangle(
    triangles: np.ndarray, rectangle: tuple[float, ...], /
) -> np.ndarray:
    """Closed triangle/rectangle intersection by exact separating axes."""
    xl, yl, xu, yu = rectangle
    lower, upper = np.min(triangles, axis=1), np.max(triangles, axis=1)
    touching = ~(
        (upper[:, 0] < xl) | (lower[:, 0] > xu) | (upper[:, 1] < yl) | (lower[:, 1] > yu)
    )
    rows = np.flatnonzero(touching)
    if not rows.size:
        return touching
    oriented = _counterclockwise(triangles[rows])
    corners = np.asarray(((xl, yl), (xu, yl), (xu, yu), (xl, yu)), dtype=np.float64)
    separated = np.zeros(rows.size, dtype=np.bool_)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        signs = _exact_orientation(
            oriented[:, None, first], oriented[:, None, second], corners[None, :, :]
        )
        separated |= np.all(signs < 0, axis=1)
    touching[rows] = ~separated
    return touching


def _scope_edge_rows(mesh: phx.discretization.CellMesh, scope: Any, /) -> np.ndarray:
    """Vertex rows of the mesh edges named by one declared edge scope."""
    if scope.entity_dimension != 1:
        raise ValueError(
            f"overset.scope.entity_dimension must be 1; got {scope.entity_dimension!r}"
        )
    identifiers = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    requested = np.unique(np.asarray(scope.entity_ids, dtype=np.int64))
    selected = np.isin(identifiers, requested)
    if int(np.count_nonzero(selected)) != requested.size:
        raise ValueError(
            f"overset.scope.entity_ids must name existing mesh edges; got {requested.size} IDs with {int(np.count_nonzero(selected))} matches"
        )
    incidence = mesh.connectivity
    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError("Overset edge scopes require polygonal connectivity.")
    return np.asarray(incidence.edges, dtype=np.int64)[selected]


def _oracle_fringe_layers(
    cells: np.ndarray,
    hole: np.ndarray,
    cell_hole: np.ndarray,
    boundary: np.ndarray,
    protected: np.ndarray,
    layers: int,
    /,
) -> np.ndarray:
    """The declared fringe law driven by independently computed holes and cuts."""
    touched = np.zeros(hole.shape, dtype=np.bool_)
    touched[cells[cell_hole].reshape((-1,))] = True
    frontier = (touched | boundary) & ~hole
    layer = np.where(frontier, 1, 0).astype(np.int32)
    for depth in range(2, layers + 1):
        reached = np.any(frontier[cells], axis=1) & ~cell_hole
        grown = np.zeros(hole.shape, dtype=np.bool_)
        grown[cells[reached].reshape((-1,))] = True
        frontier = grown & ~hole & (layer == 0)
        layer[frontier] = depth
    layer[protected & ~hole] = 0
    return layer


class _PartLayout(NamedTuple):
    points: np.ndarray
    cells: np.ndarray
    cell_ids: np.ndarray


def _part_carrier(part: phx.meshing.MeshPart, /) -> phx.meshing.CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, phx.meshing.CellMeshingResult):
        raise TypeError("Native overset qualification requires a cell meshing result.")
    return carrier


def _part_certification(
    part: phx.meshing.MeshPart, /
) -> phx.meshing.MeshCertificationReport:
    certificate = _part_carrier(part).certification
    if certificate is None:
        raise ValueError("Native overset qualification requires certified carriers.")
    return certificate


def _planar_source(plan: phx.meshing.NativeMeshingPlan, /) -> phx.meshing.NativePlanarSource:
    source = plan.source
    if not isinstance(source, phx.meshing.NativePlanarSource):
        raise TypeError("Native overset qualification requires a planar source.")
    return source


def _part_layout(part: Any, /) -> _PartLayout:
    mesh = _part_carrier(part).mesh
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    return _PartLayout(
        np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(_mesh_triangles(mesh), dtype=np.int64),
        cell_ids,
    )


def _gamma(count: int, /) -> float:
    """Higham's ``gamma_n = n u / (1 - n u)`` for binary64 unit roundoff ``u``."""
    product = count * _UNIT_ROUNDOFF
    if not 0.0 <= product < 1.0:
        raise ValueError(f"overset.gamma.count must keep n*u below one; got {count!r}")
    return product / (1.0 - product)


# The oracle's admissible measure rules and locator decisions are authored here.
# A product with a longer measure rule or different locator tolerances is refused;
# product output cannot redefine the independent oracle's decisions.
_MAXIMUM_MEASURE_RULE_POINTS = 16
_MAXIMUM_REFERENCE_TOLERANCE = 1.0e-10
_MAXIMUM_RESIDUAL_TOLERANCE = 1.0e-10


class _DonorChart(NamedTuple):
    """One donor part's affine triangle charts and its admissible cells."""

    points: np.ndarray
    cells: np.ndarray
    cell_ids: np.ndarray
    admissible: np.ndarray


class _ReceptorDonors(NamedTuple):
    """Reported donor outcome of one receptor part in receptor-slot order."""

    sites: np.ndarray
    status: np.ndarray
    donor_parts: np.ndarray
    donor_cells: np.ndarray
    candidate_parts: np.ndarray
    residuals: np.ndarray
    measures: np.ndarray


def _coordinate_sums(corners: np.ndarray, /) -> np.ndarray:
    """``S = max_a sum_i |x_{i,a}|`` over the three corners of every triangle."""
    return np.max(np.sum(np.abs(corners), axis=1), axis=-1)


def _measure_bound(scale: float, area: float, /) -> float:
    """Binary64 forward-error bound of an affine-triangle measure rule.

    The chart Jacobian entries are three-term sums ``sum_i x_i g_i`` with exact
    reference gradients ``g`` in ``{-1, 0, 1}``: ``|dJ| <= gamma_2 S``. A direct
    determinant then errs by at most ``gamma_16 S^2``; a Gram ``sqrt(det(J^T J))``
    evaluation errs by at most ``gamma_64 S^4 / |det J|`` (perturbation divided by
    ``sqrt(a) + sqrt(b) >= |det J|``). Since ``|det J| <= 2 S^2`` both are below
    ``gamma_80 S^4 / |det J|``. Multiplying by at most
    ``_MAXIMUM_MEASURE_RULE_POINTS`` rounded reference weights summing exactly to
    ``1/2`` and summing them adds a relative ``gamma_{points + 2}``. The final factor
    dominates evaluating this bound.
    """
    determinant = 2.0 * area
    relative = _gamma(_MAXIMUM_MEASURE_RULE_POINTS + 2)
    return (
        (1.0 + relative) * _gamma(80) * scale**4 / (2.0 * determinant)
        + relative * area
    ) * (1.0 + _gamma(4))


def _reference_band(
    corners: np.ndarray, sites_norm: float, residual_tolerance: float, /
) -> np.ndarray:
    """Upper bound on the owner's computed minus exact barycentric margin.

    The owner's reference point satisfies ``|J xi - (p - x0)| <= residual_tolerance``
    up to the binary64 residual evaluation error ``gamma_32 (S + 3 |p|)``; mapping
    that physical defect through ``J^{-1}`` bounds each reference coordinate, and
    the third barycentric ``1 - xi_0 - xi_1`` doubles it.
    """
    jacobian = np.stack(
        (corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1
    )
    inverse = np.linalg.inv(jacobian)
    norm = np.max(np.sum(np.abs(inverse), axis=-1), axis=-1)
    defect = residual_tolerance + _gamma(32) * (
        _coordinate_sums(corners) + 3.0 * sites_norm
    )
    return 2.0 * norm * defect * (1.0 + _gamma(16))


def _exact_barycentric(
    corners: np.ndarray, site: np.ndarray, /
) -> tuple[Fraction, Fraction]:
    """Exact rational minimum barycentric and measure of binary64 inputs."""
    a, b, c = (tuple(Fraction(float(value)) for value in row) for row in corners)
    p = tuple(Fraction(float(value)) for value in site)

    def orient(
        first: tuple[Fraction, ...],
        second: tuple[Fraction, ...],
        third: tuple[Fraction, ...],
    ) -> Fraction:
        return (first[0] - third[0]) * (second[1] - third[1]) - (
            first[1] - third[1]
        ) * (second[0] - third[0])

    total = orient(a, b, c)
    if total == 0:
        raise qualification_failure(
            "_exact_barycentric",
            ("Overset donor oracle received an exactly degenerate donor triangle."),
            {"corners": corners.tolist()},
        )
    margin = min(orient(p, b, c), orient(a, p, c), orient(a, b, p)) / total
    return margin, abs(total) / 2


_DONOR_FAULTS = (
    "oracle_orphans",
    "registration_orphans",
    "chosen_not_containing",
    "chosen_not_admissible",
    "not_finest",
    "candidate_part_mismatches",
    "measure_discrepancy_violations",
    "residual_violations",
)


def _donor_verdict(
    receptor: _ReceptorDonors,
    charts: dict[int, _DonorChart],
    /,
) -> dict[str, Any]:
    """Independent finest-admissible-donor verdict of one receptor part.

    Exact rational barycentrics classify each receptor/admissible-cell pair against
    the owner's reference tolerance ``tau`` widened by the derived band ``rho``:
    certainly located (``margin >= rho - tau``), possibly located
    (``margin >= -tau - rho``) or certainly not. Native measures must agree with
    exact areas within ``_measure_bound``; the chosen donor must be possibly
    located, admissible, and no coarser than any certainly located competitor by
    more than both bounds; the reported competing-part count must lie between the
    certain and possible part counts; and the reported inverse-map residual must
    meet the owner's residual tolerance.
    """
    counts = dict.fromkeys(("receptors", "measure_ties", "band_pairs", *_DONOR_FAULTS), 0)
    sites = np.asarray(receptor.sites, dtype=np.float64)
    sites_norm = float(np.max(np.abs(sites), initial=0.0))
    certain: list[list[tuple[int, int]]] = [[] for _ in range(sites.shape[0])]
    possible: list[list[tuple[int, int]]] = [[] for _ in range(sites.shape[0])]
    exact: dict[tuple[int, int], tuple[Fraction, float]] = {}
    rows_by_id = {
        part: {int(identifier): row for row, identifier in enumerate(chart.cell_ids)}
        for part, chart in charts.items()
    }
    worst_ratio = 0.0
    for part, chart in charts.items():
        cells = np.flatnonzero(chart.admissible)
        if not cells.size or not sites.size:
            continue
        corners = chart.points[chart.cells[cells]]
        band = _reference_band(corners, sites_norm, _MAXIMUM_RESIDUAL_TOLERANCE)
        edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=-1)
        dilation = (
            8.0 * np.max(edges, axis=1) * (_MAXIMUM_REFERENCE_TOLERANCE + band)
        )
        lower = np.min(corners, axis=1) - dilation[:, None]
        upper = np.max(corners, axis=1) + dilation[:, None]
        for start in range(0, sites.shape[0], 128):
            chunk = sites[start : start + 128]
            boxed = np.all(
                (chunk[:, None, :] >= lower[None]) & (chunk[:, None, :] <= upper[None]),
                axis=-1,
            )
            for local, pair in zip(*np.nonzero(boxed), strict=True):
                slot, row = start + int(local), int(cells[pair])
                margin, area = _exact_barycentric(corners[pair], sites[slot])
                tau, rho = (
                    Fraction(_MAXIMUM_REFERENCE_TOLERANCE),
                    Fraction(float(band[pair])),
                )
                if margin < -tau - rho:
                    continue
                if (part, row) not in exact:
                    exact[(part, row)] = (
                        area,
                        _measure_bound(
                            float(_coordinate_sums(corners[pair : pair + 1])[0]),
                            float(area),
                        ),
                    )
                possible[slot].append((part, row))
                if margin >= rho - tau:
                    certain[slot].append((part, row))
                else:
                    counts["band_pairs"] += 1
    for slot in range(sites.shape[0]):
        counts["receptors"] += 1
        if int(receptor.status[slot]) != int(CouplingSearchStatus.FOUND):
            counts["registration_orphans"] += 1
            counts["oracle_orphans"] += int(bool(certain[slot]))
            continue
        donor = int(receptor.donor_parts[slot])
        row = rows_by_id.get(donor, {}).get(int(receptor.donor_cells[slot]))
        if row is None or not charts[donor].admissible[row]:
            counts["chosen_not_admissible"] += 1
            continue
        if (donor, row) not in possible[slot]:
            counts["chosen_not_containing"] += 1
            continue
        area, bound = exact[(donor, row)]
        discrepancy = abs(Fraction(float(receptor.measures[slot])) - area)
        worst_ratio = max(worst_ratio, float(discrepancy) / bound)
        counts["measure_discrepancy_violations"] += int(discrepancy > Fraction(bound))
        residual = float(receptor.residuals[slot])
        corners = charts[donor].points[charts[donor].cells[row]][None]
        residual_bound = _MAXIMUM_RESIDUAL_TOLERANCE + _gamma(32) * float(
            _coordinate_sums(corners)[0] + 3.0 * sites_norm
        )
        counts["residual_violations"] += int(
            not np.isfinite(residual) or residual > residual_bound
        )
        low = area - Fraction(bound)
        if certain[slot] and low > min(
            exact[key][0] + Fraction(exact[key][1]) for key in certain[slot]
        ):
            counts["not_finest"] += 1
        counts["measure_ties"] += int(
            sum(
                exact[key][0] - Fraction(exact[key][1]) <= area + Fraction(bound)
                and exact[key][0] + Fraction(exact[key][1]) >= low
                for key in possible[slot]
            )
            > 1
        )
        least = len({part for part, _ in certain[slot]})
        most = len({part for part, _ in possible[slot]})
        counts["candidate_part_mismatches"] += int(
            not least <= int(receptor.candidate_parts[slot]) <= most
        )
    return {
        **counts,
        "maximum_measure_discrepancy_to_bound_ratio": worst_ratio,
        "exact": all(counts[key] == 0 for key in _DONOR_FAULTS),
    }


def _overset_donor_oracle(
    connectivity: phx.meshing.OversetConnectivity,
    layouts: dict[str, _PartLayout],
    /,
) -> dict[str, object]:
    """Apply the independent donor verdict to every receptor part of one epoch."""
    parts = connectivity.assembly.parts
    names = tuple(part.name for part in parts)
    if connectivity.policy.donor_priority != "finest-donor" or any(
        spec.priority != 0 for spec in connectivity.specs
    ):
        raise qualification_failure(
            "_overset_donor_oracle",
            ("Independent donor oracle covers the equal-priority finest-donor policy."),
            {"policy": connectivity.policy.donor_priority},
        )
    charts: dict[int, _DonorChart] = {}
    for index, (name, geometry) in enumerate(
        zip(names, connectivity.geometries, strict=True)
    ):
        layout = layouts[name]
        owner = geometry.discretization
        rule = (
            0
            if owner is None
            else max(
                int(np.asarray(block.reference_weights).shape[-1])
                for block in owner.block_geometries[0]
            )
        )
        references = tuple(
            locator.policy.reference_tolerance for locator in geometry.locators
        )
        residuals = tuple(
            locator.policy.residual_tolerance for locator in geometry.locators
        )
        if (
            owner is None
            or np.asarray(geometry.cell_measures).dtype != np.float64
            or layout.points.shape[1] != 2
            or not 1 <= rule <= _MAXIMUM_MEASURE_RULE_POINTS
            or not references
            or any(value != _MAXIMUM_REFERENCE_TOLERANCE for value in references)
            or any(value != _MAXIMUM_RESIDUAL_TOLERANCE for value in residuals)
        ):
            raise qualification_failure(
                "_overset_donor_oracle",
                (
                    "Independent donor oracle covers binary64 planar affine FE charts "
                    "with the authored measure-rule bound and exact locator controls."
                ),
                {
                    "part": name,
                    "rule_points": rule,
                    "reference_tolerances": references,
                    "residual_tolerances": residuals,
                },
            )
        charts[index] = _DonorChart(
            layout.points,
            layout.cells,
            layout.cell_ids,
            np.asarray(connectivity.blanking_of(name).cell_status)
            == int(OversetCellStatus.ACTIVE),
        )
    verdicts: dict[str, dict[str, Any]] = {}
    for index, evidence in enumerate(connectivity.receptors):
        name = names[index]
        verdicts[name] = _donor_verdict(
            _ReceptorDonors(
                layouts[name].points[np.asarray(evidence.receptor_rows, dtype=np.int64)],
                np.asarray(evidence.status),
                np.asarray(evidence.donor_parts),
                np.asarray(evidence.donor_cells),
                np.asarray(evidence.candidate_parts),
                np.asarray(evidence.residuals),
                np.asarray(evidence.donor_measures),
            ),
            {part: chart for part, chart in charts.items() if part != index},
        )
    totals = {
        key: sum(int(item[key]) for item in verdicts.values())
        for key in ("receptors", "measure_ties", "band_pairs", *_DONOR_FAULTS)
    }
    return {
        **totals,
        "parts": verdicts,
        "maximum_measure_discrepancy_to_bound_ratio": max(
            (
                item["maximum_measure_discrepancy_to_bound_ratio"]
                for item in verdicts.values()
            ),
            default=0.0,
        ),
        "exact": all(item["exact"] is True for item in verdicts.values()),
        "method": (
            "exact-rational-barycentric-owner-tolerance-band-finest-admissible-donor"
        ),
    }


def _overset_registration_oracle(
    connectivity: phx.meshing.OversetConnectivity,
) -> dict[str, object]:
    """Recompute holes, cut cells, fringes, conflicts and donors from geometry."""
    parts = connectivity.assembly.parts
    specs = connectivity.specs
    if any(
        spec.image_rotation is not None
        or spec.excluded is not None
        or spec.solid_query is not None
        for spec in specs
    ):
        raise qualification_failure(
            "_overset_registration_oracle",
            ("Independent hole oracle covers unposed discretized-wall registrations."),
            {"specs": [spec.spec_id for spec in specs]},
        )
    layouts = {part.name: _part_layout(part) for part in parts}
    rectangles = {
        spec.part_name: _wall_rectangle(
            layouts[spec.part_name].points,
            _scope_edge_rows(_part_carrier(part).mesh, spec.wall),
        )
        for part, spec in zip(parts, specs, strict=True)
        if spec.wall is not None
    }
    per_part: dict[str, dict[str, int]] = {}
    for part, spec, blanking in zip(parts, specs, connectivity.blanking, strict=True):
        layout = layouts[part.name]
        hole = np.zeros(layout.points.shape[0], dtype=np.bool_)
        cell_hole = np.zeros(layout.cells.shape[0], dtype=np.bool_)
        for owner, rectangle in rectangles.items():
            if owner == part.name:
                continue
            hole |= _points_in_rectangle(layout.points, rectangle)
            cell_hole |= _triangles_touch_rectangle(
                layout.points[layout.cells], rectangle
            )
        cell_hole |= np.any(hole[layout.cells], axis=1)
        boundary = np.zeros_like(hole)
        if spec.boundary is not None:
            boundary[
                _scope_edge_rows(_part_carrier(part).mesh, spec.boundary).reshape((-1,))
            ] = True
        protected = np.zeros_like(hole)
        if spec.wall is not None:
            protected[_scope_edge_rows(_part_carrier(part).mesh, spec.wall).reshape((-1,))] = (
                True
            )
        layer = _oracle_fringe_layers(
            layout.cells,
            hole,
            cell_hole,
            boundary,
            protected,
            connectivity.policy.fringe_layers,
        )
        vertices = np.select(
            (hole, layer > 0),
            (int(OversetVertexStatus.HOLE), int(OversetVertexStatus.RECEPTOR)),
            int(OversetVertexStatus.ACTIVE),
        )
        cells = np.select(
            (cell_hole, np.any(layer[layout.cells] > 0, axis=1)),
            (int(OversetCellStatus.HOLE), int(OversetCellStatus.FRINGE)),
            int(OversetCellStatus.ACTIVE),
        )
        conflicts = protected & (vertices != int(OversetVertexStatus.ACTIVE))
        per_part[part.name] = {
            "vertex_status_mismatches": int(
                np.count_nonzero(vertices != np.asarray(blanking.vertex_status))
            ),
            "cell_status_mismatches": int(
                np.count_nonzero(cells != np.asarray(blanking.cell_status))
            ),
            "fringe_layer_mismatches": int(
                np.count_nonzero(layer != np.asarray(blanking.fringe_layer))
            ),
            "protected_conflict_mismatches": int(
                np.count_nonzero(conflicts != np.asarray(blanking.protected_conflict))
            ),
            "oracle_protected_conflicts": int(np.count_nonzero(conflicts)),
            "oracle_hole_vertices": int(np.count_nonzero(hole)),
            "oracle_hole_cells": int(np.count_nonzero(cell_hole)),
            "oracle_receptors": int(np.count_nonzero(layer > 0)),
        }
    donors = _overset_donor_oracle(connectivity, layouts)
    blanking_exact = all(
        item[key] == 0
        for item in per_part.values()
        for key in (
            "vertex_status_mismatches",
            "cell_status_mismatches",
            "fringe_layer_mismatches",
            "protected_conflict_mismatches",
            "oracle_protected_conflicts",
        )
    )
    return {
        "method": "exact-closed-rectangle-wall-and-separating-axis-cut-oracle",
        "wall_rectangles": {name: list(value) for name, value in rectangles.items()},
        "parts": per_part,
        "donors": donors,
        "exact": blanking_exact and donors["exact"] is True,
    }


# ---------------------------------------------------------------- manufactured


def _manufactured(points: np.ndarray, /) -> np.ndarray:
    return (
        5.0 + points[:, 0] + points[:, 1] - 0.25 * np.sum(points * points, axis=1)
    )


def _quadratic_moments() -> np.ndarray:
    """Normalized Dirichlet moments of four barycentric factors on a triangle."""
    moments = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for indices in itertools.product(range(3), repeat=4):
        exponents = np.bincount(indices, minlength=3)
        moments[indices] = (
            math.prod(math.factorial(int(value)) for value in exponents)
            * math.factorial(2)
            / math.factorial(6)
        )
    return moments


_MOMENTS = _quadratic_moments()


def _manufactured_l2_error(
    points: np.ndarray, cells: np.ndarray, values: np.ndarray, /
) -> float:
    """Exact L2 norm of P1 minus the quadratic manufactured solution."""
    if not cells.size:
        return 0.0
    corners = points[cells]
    nodal = np.asarray(values, dtype=np.float64)[cells]
    affine = 5.0 + corners[..., 0] + corners[..., 1]
    gram = corners @ np.swapaxes(corners, -1, -2)
    defect = nodal - affine
    coefficient = 0.5 * (defect[:, :, None] + defect[:, None, :]) + 0.25 * gram
    integral = np.einsum("tij,tkl,ijkl->t", coefficient, coefficient, _MOMENTS)
    measures = np.abs(_simplex_measures(points, cells))
    return float(np.sqrt(max(0.0, float(np.sum(measures * integral)))))


# ------------------------------------------------------------- native sources


def _overset_qualification_source(
    name: str,
    resolution: int,
    capacity: int,
    timeout: float,
    body: _OversetBody | None,
    *,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
    maximum_scratch_bytes: int | None = None,
) -> tuple[Any, Any, Any, dict[str, object]]:
    """Author a provider input with protected source wall/exterior obligations."""
    if (body is None) != (name == _BACKGROUND) or (
        body is not None and body.name != name
    ):
        raise ValueError(
            f"overset.source.name must identify the background or its authored body; got {name!r}"
        )
    center, half, refinement = (
        ((0.0, 0.0), _BACKGROUND_HALF_WIDTH, 1)
        if body is None
        else (body.center, body.half_width, body.refinement)
    )
    cx, cy = center
    vertices = np.asarray(
        (
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ),
        dtype=np.float64,
    )
    loops: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3),)
    if body is not None:
        cavity = body.cavity_half_width
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(
                    (
                        (cx - cavity, cy - cavity),
                        (cx - cavity, cy + cavity),
                        (cx + cavity, cy + cavity),
                        (cx + cavity, cy - cavity),
                    ),
                    dtype=np.float64,
                ),
            )
        )
        loops = (*loops, (4, 5, 6, 7))
    region = phx.geometry.PlanarMeshRegion(vertices, loops, feature_id=name)
    revision = canonical_fingerprint(
        {
            "kind": "overset-authored-planar-source",
            "name": name,
            "vertices": vertices.tolist(),
            "loops": loops,
        }
    )
    source = phx.meshing.NativePlanarSource(region, revision)
    scope = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "overset-fluid-region",
        np.asarray((0,), dtype=np.int64),
    )
    exterior = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        1,
        "overset-exterior",
        np.arange(4, dtype=np.int64),
    )
    protected = [
        phx.meshing.ProtectedFeature(
            exterior, phx.meshing.FeatureKind.CURVE, maximum_deviation=1.0e-10, hard=True
        )
    ]
    scopes: dict[str, object] = {"fluid": scope.scope_id, "boundary": exterior.scope_id}
    if body is not None:
        wall = phx.meshing.MeshingScope(
            name,
            revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            1,
            "overset-cavity-wall",
            np.arange(4, 8, dtype=np.int64),
        )
        protected.append(
            phx.meshing.ProtectedFeature(
                wall, phx.meshing.FeatureKind.CURVE, maximum_deviation=1.0e-10, hard=True
            )
        )
        scopes["cavity_wall"] = wall.scope_id
    limits = capacity_meshing_limits(
        capacity,
        timeout,
        maximum_work_units=maximum_work_units,
        maximum_geometry_queries=maximum_geometry_queries,
        scratch_bytes_limit=maximum_scratch_bytes,
        edges=8,
        faces=8,
        cells=8,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=32,
        cavity_cells=1,
        geometry_queries=64,
        scratch_bytes=4096,
    )
    size = 0.8 / (resolution * refinement)
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=tuple(protected),
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=np.radians(25.0), hard=True
        ),
        limits=limits,
    )
    return (
        source,
        request,
        phx.meshing.NativeMeshingOptions("planar_constrained_delaunay"),
        scopes,
    )


def _source_edge_ids(
    part: phx.meshing.MeshPart,
    source: phx.meshing.NativePlanarSource,
    edges: tuple[int, ...],
) -> np.ndarray:
    """Mesh edge IDs the published association binds to named source loop edges.

    ``NativePlanarSource`` identifies loop edge ``i`` as ``<revision>:edge:<i>``;
    the selection reads that association, never names or coordinates.
    """
    carrier = _part_carrier(part)
    edge_set = carrier.mesh.entity_set(1)
    names = {f"{source.source_revision}:edge:{edge}": edge for edge in edges}
    selected: list[int] = []
    covered: set[int] = set()
    for association in carrier.associations:
        if (
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != source.source_id
            or association.source_revision != source.source_revision
            or association.target_entity_set_id != edge_set.entity_set_id
        ):
            continue
        roles = association.source_entities.roles()
        if roles is None:
            raise ValueError(
                "overset.association.source_entity_roles must be explicit; got None"
            )
        for identifier, entity, role in zip(
            np.asarray(association.target_global_ids),
            association.source_entities.entity_ids(),
            roles,
            strict=True,
        ):
            if role is GeometrySourceEntityRole.EDGE and entity in names:
                selected.append(int(identifier))
                covered.add(names[entity])
    if covered != set(edges):
        raise qualification_failure(
            "_source_edge_ids",
            ("Overset wall/boundary scopes require associations of every source edge."),
            {"covered": sorted(covered), "requested": sorted(edges)},
        )
    return np.unique(np.asarray(selected, dtype=np.int64))


def _overset_qualification_part(
    name: str,
    body: _OversetBody | None,
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    deadline: Deadline,
) -> tuple[phx.meshing.MeshPart, phx.meshing.NativeMeshingPlan]:
    """Execute a real source/provider plan and independently recount its domain."""

    _ledger = StageLedger(record, deadline=deadline)
    source, request, options, scopes = _ledger.run(
        f"{name}:source_preparation",
        lambda: _overset_qualification_source(
            name, args.resolution, args.capacity, args.timeout, body,
            maximum_work_units=args.maximum_work_units,
            maximum_geometry_queries=args.maximum_geometry_queries,
            maximum_scratch_bytes=args.maximum_scratch_bytes,
        ),
    )
    evidence: dict[str, Any] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "source_scopes": scopes,
        "request_id": request.specification_id,
        "limits_id": request.limits.limits_id,
        "quality_target_id": request.quality_target.target_id,
        "protected_feature_ids": [
            feature.feature_id for feature in request.protected_features
        ],
        "source_vertices": np.asarray(source.region.vertices).tolist(),
        "authored_body": None if body is None else body.record(),
    }
    record["parts"][name] = evidence
    plan = _ledger.run(
        f"{name}:geometry_constraint_preparation",
        lambda: phx.meshing.NativeMeshingProvider(options).plan(
            source,
            request,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            record_phase=phases.append,
        ),
    )
    evidence["prepared_id"] = plan.plan_id
    result = _ledger.run(
        f"{name}:generation_publication_certification",
        lambda: plan.execute(record_phase=phases.append),
    )
    evidence["generation"] = _result_identity(result)
    if (
        result.certification is None
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_overset_qualification_part",
            ("Overset provider carrier lacks passing source/control certification."),
            {"result": result},
        )
    cells = _mesh_triangles(result.mesh)
    points = np.asarray(result.mesh.coordinates)
    areas = _simplex_measures(points, cells)
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            f"overset.connectivity must be PolygonalConnectivity; got {type(connectivity).__name__}"
        )
    analytic = _body_areas(body)
    boundary = np.asarray(connectivity.boundary_edges)
    edges = np.asarray(connectivity.edges)[boundary]
    corners = points[edges]
    lengths = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    if body is None:
        center, cavity = np.zeros(2), None
        outer_expected, wall_expected = 8.0 * _BACKGROUND_HALF_WIDTH, 0.0
        inner = np.zeros(edges.shape[0], dtype=np.bool_)
    else:
        center, cavity = np.asarray(body.center), body.cavity_half_width
        outer_expected, wall_expected = 8.0 * body.half_width, 8.0 * cavity
        inner = np.all(np.abs(corners - center) <= cavity + 1.0e-10, axis=(1, 2))
    measured_area = float(np.sum(areas))
    outer_length = float(np.sum(lengths[~inner]))
    wall_length = float(np.sum(lengths[inner]))
    domain_area = analytic["domain"]
    if domain_area is None:
        raise ValueError("Overset authored domain area must be present.")
    if np.any(areas <= 0.0) or abs(measured_area - domain_area) > 1.0e-9:
        raise qualification_failure(
            "_overset_qualification_part",
            (
                "Overset source area or triangle orientation disagrees with the analytic fixture."
            ),
            {"areas": areas, "expected": analytic["domain"], "measured": measured_area},
        )
    if abs(outer_length - outer_expected) > 1.0e-9:
        raise qualification_failure(
            "_overset_qualification_part",
            ("Overset source exterior is not covered exactly once."),
            {"name": name, "outer_length": outer_length},
        )
    if cavity is not None and (
        abs(wall_length - wall_expected) > 1.0e-9
        or np.any(np.all(np.abs(points[cells].mean(axis=1) - center) < cavity, axis=1))
    ):
        raise qualification_failure(
            "_overset_qualification_part",
            ("Overset protected cavity wall or excluded cavity was lost."),
            {"name": name, "wall_length": wall_length},
        )
    part = phx.meshing.MeshPart(name, result)
    identifiers = np.asarray(result.mesh.entity_set(1).entity_ids, dtype=np.int64)
    geometric = {
        "wall": np.sort(identifiers[boundary][inner]),
        "exterior": np.sort(identifiers[boundary][~inner]),
    }
    if body is not None:
        associated = {
            "wall": _source_edge_ids(part, source, (4, 5, 6, 7)),
            "exterior": _source_edge_ids(part, source, (0, 1, 2, 3)),
        }
        if any(
            not np.array_equal(associated[key], geometric[key])
            for key in ("wall", "exterior")
        ):
            raise qualification_failure(
                "_overset_qualification_part",
                (
                    "Source-associated wall/exterior edges disagree with the independent geometric recount."
                ),
                {"name": name},
            )
    evidence["independent_acceptance"] = {
        "expected_area": analytic["domain"],
        "measured_area": measured_area,
        "outer_boundary_length": outer_length,
        "cavity_wall_length": wall_length,
        "association_wall_edges": int(geometric["wall"].size),
        "association_exterior_edges": int(geometric["exterior"].size),
        "quality": _quality_summary(
            phx.meshing.summarize_cell_quality(
                phx.meshing.evaluate_cell_quality(result.mesh)
            )
        ),
        "certification_id": result.certification.report_id,
        "requested": dict(result.compliance.requested),
        "achieved": dict(result.compliance.achieved),
    }
    record["attempts"].append(
        {
            "phase": record["active_phase"],
            "status": "passed",
            "source_binding_id": source.binding_id,
            "result_id": result.result_id,
        }
    )
    return part, plan


def _overset_qualification_connectivity(
    background: phx.meshing.MeshPart,
    bodies: dict[str, tuple[phx.meshing.MeshPart, phx.meshing.NativeMeshingPlan]],
    profile: _OversetProfile,
    args: QualificationControls,
) -> phx.meshing.OversetConnectivity:
    """Bind source-associated cavity walls and artificial boundaries of every body."""
    specs = [OversetPartSpec(background.name)]
    parts = [background]
    for body in profile.bodies:
        part, plan = bodies[body.name]
        source = plan.source
        if not isinstance(source, phx.meshing.NativePlanarSource):
            raise TypeError(
                f"overset.plan.source must be NativePlanarSource; got {type(source).__name__}"
            )
        specs.append(
            OversetPartSpec(
                body.name,
                wall=part.scope(1, _source_edge_ids(part, source, (4, 5, 6, 7))),
                boundary=part.scope(1, _source_edge_ids(part, source, (0, 1, 2, 3))),
            )
        )
        parts.append(part)
    return prepare_overset_connectivity(
        phx.meshing.MeshAssembly(tuple(parts)),
        tuple(specs),
        policy=OversetPolicy(
            fringe_layers=profile.fringe_layers,
            maximum_wall_candidate_pairs=args.overset_wall_pairs,
            maximum_donor_candidate_pairs=args.overset_donor_pairs,
        ),
    )


def _overset_qualification_registration_evidence(connectivity: Any) -> dict[str, object]:
    """Expose blanking/search statuses, including incomplete registrations."""

    return {
        "connectivity_id": connectivity.connectivity_id,
        "epoch": connectivity.epoch,
        "predecessor_id": connectivity.predecessor_id,
        "policy_id": connectivity.policy.policy_id,
        "complete": connectivity.complete,
        "orphan_count": connectivity.orphan_count,
        "protected_wall_conflict_count": connectivity.conflict_count,
        "ambiguous_count": connectivity.ambiguous_count,
        "wall_candidate_pairs": connectivity.wall_candidate_pairs,
        "parts": [
            {
                "part": item.part_name,
                "blanking_id": item.blanking_id,
                "hole_vertex_ids": item.vertex_ids_with(
                    OversetVertexStatus.HOLE
                ).tolist(),
                "vertex_status": np.asarray(item.vertex_status).tolist(),
                "cell_status": np.asarray(item.cell_status).tolist(),
            }
            for item in connectivity.blanking
        ],
        "receptors": [
            {
                "part": item.part_name,
                "evidence_id": item.evidence_id,
                "orphan_ids": item.orphan_ids.tolist(),
                "status": np.asarray(item.status).tolist(),
                "receptor_ids": np.asarray(item.receptor_ids).tolist(),
                "donor_parts": np.asarray(item.donor_parts).tolist(),
                "donor_cells": np.asarray(item.donor_cells).tolist(),
                "candidate_parts": np.asarray(item.candidate_parts).tolist(),
                "residuals": [
                    float(value) if np.isfinite(value) else None
                    for value in np.asarray(item.residuals)
                ],
                "donor_measures": [
                    float(value) if np.isfinite(value) else None
                    for value in np.asarray(item.donor_measures)
                ],
            }
            for item in connectivity.receptors
        ],
        "donor_packet_ids": [packet.packet_id for packet in connectivity.packets],
    }


def _overset_require_oracle(oracle: dict[str, object], epoch: int, /) -> None:
    if oracle["exact"] is not True:
        raise qualification_failure(
            "_overset_registration_oracle",
            (
                "Native hole cutting, fringes or donors disagree with the independent oracle."
            ),
            {"epoch": epoch, "oracle": oracle},
        )


# --------------------------------------------------------- composite Poisson


@final
class _OversetComposite(StrictModule):
    """Composite overset Poisson system over every part's unconstrained rows.

    Each part constrains its physical boundary and blanked rows to their data. Its
    remaining rows are unknowns: ordinary FE equations of the part's compiled
    problem, except receptor rows, which are the exact interpolation rows
    ``u_r - I_r(u_donors) = 0`` of the epoch's prepared field query. The operator
    is matrix-free; the right-hand side carries the FE loads and the interpolated
    donor Dirichlet lifts.
    """

    operators: tuple[Any, ...]
    loads: tuple[jax.Array, ...]
    prolongations: tuple[Any, ...]
    lifts: tuple[jax.Array, ...]
    positions: tuple[jax.Array, ...]
    coupling: phx.linalg.AbstractLinearOperator
    donor_names: tuple[str, ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @property
    def space(self) -> phx.linalg.ArraySpace:
        return phx.linalg.ArraySpace(
            (self.offsets[-1],), dtype=jnp.float64,
            space_id=f"{self.operator_id}:unknowns",
        )

    def split(self, value: jax.Array, /) -> tuple[jax.Array, ...]:
        return tuple(
            value[start:stop]
            for start, stop in zip(self.offsets[:-1], self.offsets[1:], strict=True)
        )

    def _interpolate(self, full: tuple[jax.Array, ...], /) -> dict[str, jax.Array]:
        values = dict(zip(self.names, full, strict=True))
        return self.coupling.mv({name: values[name] for name in self.donor_names})

    def homogeneous(self, value: jax.Array, /) -> jax.Array:
        parts = self.split(value)
        interpolated = self._interpolate(
            tuple(
                prolongation.mv(part)
                for prolongation, part in zip(self.prolongations, parts, strict=True)
            )
        )
        return jnp.concatenate(
            [
                operator.mv(part)
                .at[position]
                .set(part[position] - interpolated[name])
                for name, operator, part, position in zip(
                    self.names, self.operators, parts, self.positions, strict=True
                )
            ]
        )

    def operator(self) -> phx.linalg.FunctionLinearOperator:
        return phx.linalg.FunctionLinearOperator(
            self.homogeneous,
            source=self.space,
            target=self.space,
            operator_id=self.operator_id,
        )

    def right_hand_side(self) -> jax.Array:
        interpolated = self._interpolate(self.lifts)
        return jnp.concatenate(
            [
                load.at[position].set(interpolated[name])
                for name, load, position in zip(
                    self.names, self.loads, self.positions, strict=True
                )
            ]
        )

    def expand(self, value: jax.Array, /) -> dict[str, jax.Array]:
        return {
            name: prolongation.mv(part) + lift
            for name, prolongation, lift, part in zip(
                self.names, self.prolongations, self.lifts, self.split(value), strict=True
            )
        }


@runtime_checkable
class _OversetJitEntry[T](Protocol):
    def lower(self, *args: object, **kwargs: object) -> _LoweredProgram[T]: ...


@eqx.filter_jit
def _solve_overset_composite(
    composite: _OversetComposite,
    seed: jax.Array,
    policy: phx.linalg.LinearSolvePolicy,
) -> tuple[dict[str, jax.Array], phx.linalg.LinearSolveResult]:
    """Native preconditioned Krylov solve of the composite overset system."""
    result = phx.linalg.solve(
        phx.linalg.LinearSystem(composite.operator()),
        composite.right_hand_side(),
        policy=policy,
        initial_guess=seed,
    )
    return composite.expand(result.value), result


@eqx.filter_jit
def _overset_full_residual(problem: Any, values: jax.Array) -> jax.Array:
    """Matrix-free FE residual, independent of the composite solve operator.

    One AOT executable per part and epoch evaluates the right-hand side at the
    lift and afterwards checks the solved state.
    """
    return problem.full_residual(values)


def _overset_system_operator(problem: Any, /) -> phx.linalg.FunctionLinearOperator:
    """The constrained affine FE system action of ``problem.linear_system()``.

    The action is realized inside the consuming compilation, so no separate
    per-epoch assembly program is compiled.
    """
    space = problem.state_space
    return phx.linalg.FunctionLinearOperator(
        lambda value: space.inverse_riesz(problem.affine_operator().mv(value)),
        source=space,
        target=space,
        operator_id=f"overset-system:{problem.compilation_id}",
    )


class _OversetPartRows(NamedTuple):
    constraint: Any
    free: np.ndarray
    receptor_rows: np.ndarray
    physical_rows: np.ndarray
    physical_values: jax.Array
    hole: np.ndarray
    wall_rows: np.ndarray
    exterior_rows: np.ndarray


class _OversetFactorEvidence(TypedDict):
    size: int
    pattern_entries: int
    colors: int
    block_relative_error: float
    block_verified: bool
    factor_entries: int
    factor_storage_bytes_upper: int
    factor_symbolic_work: int
    reused_symbolic_plan: bool
    symbolic_plan_id: str
    factor_status: int
    minimum_pivot: float


class _OversetSymbolicPart(NamedTuple):
    pattern_id: str
    plan: phx.linalg.SparseFactorizationPlan


class _OversetSolver(NamedTuple):
    composite: _OversetComposite
    policy: phx.linalg.LinearSolvePolicy
    compiled: Any
    rows: dict[str, _OversetPartRows]
    analytic: dict[str, dict[str, float | None]]
    factorizations: dict[str, _OversetFactorEvidence]
    symbolic_parts: dict[str, _OversetSymbolicPart]
    compiler: dict[str, Any]
    stages: dict[str, float]
    problems: tuple[Any, ...]
    residuals: tuple[Any, ...]
    route: Any


def _solver_evidence(solved: Any, /) -> dict[str, object]:
    condition = np.asarray(solved.diagnostics.condition_estimate)
    available = bool(np.all(np.isfinite(condition)))
    return {
        "successful": bool(np.all(np.asarray(solved.successful))),
        "status": np.asarray(solved.status).tolist(),
        "iterations": np.asarray(solved.diagnostics.iterations).tolist(),
        "residual_norm": np.asarray(solved.diagnostics.residual_norm).tolist(),
        "relative_residual": np.asarray(solved.diagnostics.relative_residual).tolist(),
        "rank": np.asarray(solved.diagnostics.rank).tolist(),
        "condition_estimate": condition.tolist() if available else None,
        "condition_status": "reported" if available else "unavailable",
    }


def _overset_part_rows(
    connectivity: phx.meshing.OversetConnectivity,
    name: str,
    field: phx.discretization.FiniteElementDiscretization,
) -> _OversetPartRows:
    """Classify one part's P1 rows into physical, blanked, receptor and free rows."""
    mesh = _part_carrier(connectivity.assembly.part(name)).mesh
    dof_map = field.dof_maps[field.field_index("u")]
    coordinates = np.asarray(dof_map.dof_coordinates, dtype=np.float64)
    if coordinates.shape != np.asarray(mesh.coordinates).shape or not np.array_equal(
        coordinates, np.asarray(mesh.coordinates)
    ):
        raise qualification_failure(
            "_overset_part_rows",
            ("Overset P1 DOFs must coincide with the registered mesh vertex rows."),
            {"name": name},
        )
    status = np.asarray(connectivity.blanking_of(name).vertex_status)
    hole = status == int(OversetVertexStatus.HOLE)
    receptor_rows = np.asarray(
        connectivity.receptors_of(name).receptor_rows, dtype=np.int64
    )
    receptor = np.zeros(hole.shape, dtype=np.bool_)
    receptor[receptor_rows] = True
    if not np.array_equal(receptor, status == int(OversetVertexStatus.RECEPTOR)):
        raise qualification_failure(
            "_overset_part_rows",
            ("Receptor evidence disagrees with the registered fringe classification."),
            {"name": name},
        )
    physical = np.asarray(dof_map.boundary_dof_mask, dtype=np.bool_) & ~receptor & ~hole
    if not np.any(physical):
        raise qualification_failure(
            "_overset_part_rows",
            ("Every overset part requires physical Dirichlet data."),
            {"name": name},
        )
    spec = connectivity.specs[connectivity.part_index(name)]
    wall = np.zeros(hole.shape, dtype=np.bool_)
    if spec.wall is not None:
        wall[_scope_edge_rows(mesh, spec.wall).reshape((-1,))] = True
    constraint = phx.discretization.dirichlet_constraint(
        field, "u", boundary_mask=physical | hole
    )
    physical_rows = np.flatnonzero(physical)
    return _OversetPartRows(
        constraint,
        np.asarray(constraint.free_dofs, dtype=np.int64),
        receptor_rows,
        physical_rows,
        jnp.asarray(_manufactured(coordinates)[physical_rows]),
        hole,
        np.flatnonzero(physical & wall),
        np.flatnonzero(physical & ~wall),
    )


def _overset_block_pattern(cells: np.ndarray, free: np.ndarray, /) -> Any:
    """P1 vertex adjacency of one part restricted to its unconstrained rows."""
    local = np.full(int(np.max(cells, initial=-1)) + 1, -1, dtype=np.int64)
    local[free] = np.arange(free.size, dtype=np.int64)
    mapped = local[cells]
    targets = np.repeat(mapped, cells.shape[1], axis=1).reshape((-1,))
    sources = np.tile(mapped, (1, cells.shape[1])).reshape((-1,))
    keep = (targets >= 0) & (sources >= 0)
    pairs = np.unique(
        np.concatenate(
            (
                np.stack((targets[keep], sources[keep]), axis=1),
                np.stack((np.arange(free.size),) * 2, axis=1),
            )
        ),
        axis=0,
    )
    return phx.sparse.EdgeRelation(
        pairs[:, 1], pairs[:, 0], source_size=free.size, target_size=free.size
    )


@eqx.filter_jit
def _overset_block_coefficients(
    plan: phx.sparse.SparseDerivativePlan,
    zero: jax.Array,
    key: jax.Array,
) -> tuple[phx.sparse.SparseCoordinateOperator, phx.sparse.SparseDerivativeVerification]:
    """Evaluate and verify one block in one native numerical compilation."""
    return plan.operator(zero), phx.sparse.verify_sparse_derivative(plan, zero, key=key)


def _overset_part_term(
    composite: _OversetComposite,
    index: int,
    cells: np.ndarray,
    free: np.ndarray,
    materialization: phx.linalg.MaterializationPolicy,
    /,
    *,
    previous: _OversetSymbolicPart | None = None,
) -> tuple[
    phx.linalg.SubspaceCorrectionTerm, _OversetFactorEvidence,
    _OversetSymbolicPart, phx.sparse.SparseCoordinateOperator,
]:
    """Verified sparse factorization of one part's exact composite diagonal block."""
    start, stop = composite.offsets[index], composite.offsets[index + 1]
    space = composite.space
    local = composite.operators[index].source
    if not isinstance(local, phx.linalg.ArraySpace) or local.size != stop - start:
        raise TypeError("Composite part blocks require the owning scalar FE coordinate space.")
    name = composite.names[index]
    restriction = phx.linalg.FunctionLinearOperator(
        lambda value: value[start:stop],
        source=space,
        target=local,
        transpose_action=lambda value: space.zeros().at[start:stop].set(value),
        operator_id=f"{composite.operator_id}:restrict:{composite.names[index]}",
    )
    prolongation = phx.linalg.FunctionLinearOperator(
        lambda value: space.zeros().at[start:stop].set(value),
        source=local,
        target=space,
        transpose_action=lambda value: value[start:stop],
        operator_id=f"{composite.operator_id}:prolong:{composite.names[index]}",
    )

    # The local action's numerical leaves are explicit plan arguments rather
    # than constants captured by the derivative program.
    arguments, structure = eqx.partition(
        (composite.operators[index], composite.positions[index]), eqx.is_array
    )

    def block(value: jax.Array, dynamic: object) -> jax.Array:
        # All coupling packets read another part. Therefore restriction of A
        # after this part's prolongation is its FE action with identity receptor
        # rows; evaluating the other parts and their zero donor inputs is dead work.
        operator, position = cast(
            tuple[phx.linalg.AbstractLinearOperator, jax.Array],
            eqx.combine(dynamic, structure),
        )
        return operator.mv(value).at[position].set(value[position])

    zero = local.zeros()
    plan = phx.sparse.compile_sparse_jacobian(
        block,
        zero,
        source=local,
        target=local,
        sample_args=arguments,
        structure=_overset_block_pattern(cells, free),
        compiler="native",
        plan_id=f"{composite.operator_id}:block:{composite.names[index]}",
    )
    local_operator, verification = _overset_block_coefficients(
        plan, zero, jax.random.key(index)
    )
    relative = float(verification.maximum_relative_error)
    if not bool(verification.passed):
        raise qualification_failure(
            "_overset_part_term",
            ("The declared P1 pattern misses entries of a composite part block."),
            {"part": composite.names[index], "relative_error": relative},
        )
    reusable = previous is not None and previous.pattern_id == plan.pattern.pattern_id
    if reusable and previous is not None:
        symbolic = previous.plan
        if (
            symbolic.factor_nnz > materialization.max_entries
            or symbolic.factor_storage_bytes_upper > materialization.max_bytes
        ):
            raise qualification_failure(
                "_overset_part_term",
                ("Retained symbolic part factor exceeds the remaining shared cap."),
                {"part": name, "factor_bytes": symbolic.factor_storage_bytes_upper},
            )
        # The native builder independently revalidates the exact CSR pattern and
        # binds the retained symbolic plan to this epoch's numerical operator.
        builder = phx.linalg.SparseFactorizationPreconditionerBuilder(
            symbolic.policy, prepared_plan=symbolic, setup_operator=local_operator,
        )
    else:
        builder = phx.linalg.SparseFactorizationPreconditionerBuilder(
            phx.linalg.SparseFactorizationPolicy(
                "lu", ordering="approximate-minimum-degree",
                max_factor_nnz=materialization.max_entries,
                max_factor_bytes=materialization.max_bytes,
            )
        )
    solver = builder.prepare(local_operator, materialization=materialization)
    factorization = solver.factorization
    status = int(factorization.status)
    if status != 0:
        raise qualification_failure(
            "_overset_part_term",
            ("The sparse factorization of a composite part block failed."),
            {"part": composite.names[index], "status": status},
        )
    evidence: _OversetFactorEvidence = {
        "size": stop - start,
        "pattern_entries": plan.nnz,
        "colors": plan.num_colors,
        "block_relative_error": relative,
        "block_verified": True,
        "factor_entries": int(factorization.plan.factor_nnz),
        "factor_storage_bytes_upper": int(factorization.plan.factor_storage_bytes_upper),
        "factor_symbolic_work": 0 if reusable else int(factorization.plan.symbolic_work),
        "reused_symbolic_plan": reusable,
        "symbolic_plan_id": factorization.plan.plan_id,
        "factor_status": status,
        "minimum_pivot": float(factorization.diagnostics.minimum_pivot),
    }
    return (
        phx.linalg.SubspaceCorrectionTerm(restriction, prolongation, solver),
        evidence,
        _OversetSymbolicPart(plan.pattern.pattern_id, factorization.plan),
        local_operator,
    )


def _overset_query_operator(route: Any, /) -> phx.linalg.FunctionLinearOperator:
    """Lower the canonical query action without retaining its proof graph in JIT inputs."""
    if not route.coefficient_linear:
        raise ValueError("Composite overset Poisson requires coefficient-linear queries.")
    source = phx.linalg.PyTreeSpace(
        {
            name: jax.ShapeDtypeStruct(shape, jnp.float64)
            for name, shape in zip(route.donor_names, route.coefficient_shapes, strict=True)
        },
        space_id=f"{route.transfer_id}:donor-coefficients",
    )
    target = phx.linalg.PyTreeSpace(
        {
            item.part_name: jax.ShapeDtypeStruct(
                (item.receptor_ids.size, *route.value_shape), jnp.float64
            )
            for item in route.receptors
        },
        space_id=f"{route.transfer_id}:receptor-values",
    )
    return phx.linalg.FunctionLinearOperator(
        route.apply, source=source, target=target,
        transpose_action=route.transpose, operator_id=route.transfer_id,
    )


def _overset_prepare_solver(
    connectivity: phx.meshing.OversetConnectivity,
    fields: dict[str, Any],
    seeds: dict[str, jax.Array],
    route: Any,
    analytic: dict[str, dict[str, float | None]],
    capacity: int,
    *,
    previous: _OversetSolver | None = None,
    prepare_part_blocks: bool = True,
) -> _OversetSolver:
    """Compile the physical composite system with explicitly requested part blocks."""
    names = tuple(part.name for part in connectivity.assembly.parts)
    routed = {
        item.part_name: np.asarray(item.receptor_rows, dtype=np.int64)
        for item in route.receptors
    }
    if route.previous_id is not None:
        raise qualification_failure(
            "_overset_prepare_solver",
            ("Composite coupling must use the epoch's own fringe receptor route."),
            {"previous_id": route.previous_id},
        )
    if any(packet.donor_part == packet.receptor_part for packet in route.packets):
        raise ValueError("Overset part blocks require the native no-self-donor invariant.")
    residual_entry = _overset_full_residual
    if not isinstance(residual_entry, _OversetJitEntry):
        raise TypeError("Composite residual must expose the canonical JIT lowering entry.")

    def prepare() -> tuple[
        dict[str, _OversetPartRows], _OversetComposite, float, tuple[Any, ...],
        tuple[Any, ...],
    ]:
        rows: dict[str, _OversetPartRows] = {}
        problems, residuals, operators, loads, prolongations, lifts, positions = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        absolute = math.inf
        for name in names:
            part = _overset_part_rows(connectivity, name, fields[name])
            if not np.array_equal(
                routed.get(name, part.receptor_rows), part.receptor_rows
            ):
                raise qualification_failure(
                    "_overset_prepare_solver",
                    ("Composite route receptors disagree with the registration."),
                    {"name": name},
                )
            rows[name] = part
            data = np.array(seeds[name], dtype=np.float64)
            data[part.physical_rows] = np.asarray(part.physical_values)
            problem = prepare_poisson_update(
                fields[name], jax.device_put(data), constraint=part.constraint
            )
            lift = jnp.asarray(problem.lift)
            residual = residual_entry.lower(problem, lift).compile()
            # Affine FE system: A z = -Pᵀ R(P 0 + g), exactly the right-hand side of
            # ``problem.linear_system()``, evaluated by the same executable that
            # later checks the solved state. The Dirichlet prolongation P is the
            # free-row selection (reduced coordinates are ``values[free]``).
            load = problem.state_space.inverse_riesz(
                jax.device_put(-np.asarray(residual(problem, lift))[part.free])
            )
            position = np.searchsorted(part.free, part.receptor_rows)
            if part.receptor_rows.size and not np.array_equal(
                part.free[np.minimum(position, part.free.size - 1)],
                part.receptor_rows,
            ):
                raise qualification_failure(
                    "_overset_prepare_solver",
                    ("Receptor rows must be unconstrained composite unknowns."),
                    {"name": name},
                )
            problems.append(problem)
            residuals.append(residual)
            operators.append(_overset_system_operator(problem))
            loads.append(jnp.asarray(load))
            prolongations.append(part.constraint.constraint_map.prolongation)
            lifts.append(lift)
            positions.append(jnp.asarray(position))
            absolute = min(
                absolute, poisson_solve_policy(fields[name]).tolerance.absolute
            )
        offsets = tuple(
            int(value)
            for value in np.concatenate(
                ([0], np.cumsum([rows[name].free.size for name in names]))
            )
        )
        composite = _OversetComposite(
            tuple(operators),
            tuple(loads),
            tuple(prolongations),
            tuple(lifts),
            tuple(positions),
            _overset_query_operator(route),
            route.donor_names,
            names,
            offsets,
            f"overset-composite:{connectivity.connectivity_id}:{route.transfer_id}",
        )
        return rows, composite, absolute, tuple(problems), tuple(residuals)

    with CompileCounter(stage="overset_composite_preparation") as preparation_compiler:
        (rows, composite, absolute, problems, residuals), preparation = _measured(prepare)
    materialization = phx.linalg.MaterializationPolicy(
        max_entries=64 * capacity,
        max_bytes=4096 * capacity,
        max_workspace_bytes=4096 * capacity,
    )

    def factor() -> tuple[
        _OversetComposite, Any, dict[str, _OversetFactorEvidence],
        dict[str, _OversetSymbolicPart],
    ]:
        terms, evidence, symbolic_parts = [], {}, {}
        local_actions = []
        remaining_entries = materialization.max_entries
        remaining_bytes = materialization.max_bytes
        for index, name in enumerate(names):
            if remaining_entries <= 0 or remaining_bytes <= 0:
                raise qualification_failure(
                    "_overset_prepare_solver",
                    ("Composite part factors exhausted the shared storage cap."),
                    {"parts": evidence},
                )
            term, record, symbolic, local_action = _overset_part_term(
                composite,
                index,
                _part_layout(connectivity.assembly.part(name)).cells,
                rows[name].free,
                phx.linalg.MaterializationPolicy(
                    max_entries=remaining_entries,
                    max_bytes=remaining_bytes,
                    max_workspace_bytes=materialization.max_workspace_bytes,
                ),
                previous=None if previous is None else previous.symbolic_parts.get(name),
            )
            terms.append(term)
            local_actions.append(local_action)
            evidence[name] = record
            symbolic_parts[name] = symbolic
            remaining_entries -= record["factor_entries"]
            remaining_bytes -= record["factor_storage_bytes_upper"]
        # The global operator stays matrix-free: compose the already verified
        # local sparse actions with native cross-part queries. Reuse the very
        # same blocks admitted for preconditioning instead of repeating FE
        # quadrature in each global/preconditioner action.
        realized = eqx.tree_at(
            lambda value: value.operators, composite, tuple(local_actions)
        )
        preconditioner = phx.linalg.MultiplicativeSubspaceCorrectionBuilder(
            tuple(terms)
        ).prepare(realized.operator(), materialization=materialization)
        return realized, preconditioner, evidence, symbolic_parts

    preconditioner = None
    factorizations: dict[str, _OversetFactorEvidence] = {}
    symbolic_parts: dict[str, _OversetSymbolicPart] = {}
    assembly = {"wall_seconds": 0.0}
    factor_compilation_count, factor_compile_seconds = 0, 0.0
    if prepare_part_blocks:
        with CompileCounter(stage="overset_composite_factorization") as factor_compiler:
            (composite, preconditioner, factorizations, symbolic_parts), assembly = _measured(factor)
        factor_compilation_count = factor_compiler.count
        factor_compile_seconds = factor_compiler.seconds
    # The qualification differentiates no composite solve: request no derivative
    # map instead of tracing an unused implicit adjoint of every epoch's system.
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=_SCHWARZ_MAXIMUM_ITERATIONS),
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0, absolute=absolute, max_steps=_SCHWARZ_MAXIMUM_ITERATIONS
        ),
        materialization=materialization,
        differentiation=phx.linalg.DifferentiationPolicy("none"),
        resources=phx.linalg.SolveResourcePolicy(
            factorization_bytes=materialization.max_bytes,
            workspace_bytes=materialization.max_workspace_bytes,
            krylov_basis_bytes=materialization.max_bytes,
            preconditioner_bytes=materialization.max_bytes,
        ),
        preconditioning=None if preconditioner is None else
        phx.linalg.PreconditioningPolicy(preconditioner, side="right"),
    )
    seed = jax.device_put(
        np.concatenate([np.asarray(seeds[name])[rows[name].free] for name in names])
    )
    entry = _solve_overset_composite
    if not isinstance(entry, _OversetJitEntry):
        raise TypeError("Composite solver must expose the canonical JIT lowering entry.")
    lowered, lowering = _measured(
        lambda: entry.lower(composite, seed, policy)
    )
    compiled, compilation = _measured(lowered.compile)
    memory = compiled.compiled.memory_analysis()
    compiler: dict[str, Any] = {
        "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        "output_bytes": None if memory is None else memory.output_size_in_bytes,
        "argument_bytes": None if memory is None else memory.argument_size_in_bytes,
        "generated_code_bytes": None
        if memory is None
        else memory.generated_code_size_in_bytes,
        "memory_status": "unavailable" if memory is None else "measured",
        "compilation_count": 1,
        "warm_samples_seconds": [],
        "compilation_count_scope": "explicit-AOT-compile-invocations",
        "warm_recompilation_count": 0,
        "execution_scope": (
            "overset-composite-right-preconditioned-GMRES-multiplicative-part-blocks"
            if prepare_part_blocks else "overset-composite-unpreconditioned-GMRES"
        ),
        "local_action_realization": (
            "verified-native-sparse-part-blocks" if prepare_part_blocks
            else "matrix-free-finite-element"
        ),
        "assembly_seconds": assembly["wall_seconds"],
        "assembly_invocations": int(prepare_part_blocks),
        "assembly_status": "prepared" if prepare_part_blocks else "not-requested",
        "assembly_compilation_count": factor_compilation_count,
        "assembly_compile_seconds": factor_compile_seconds,
        "preparation_compilation_count": preparation_compiler.count,
        "preparation_compile_seconds": preparation_compiler.seconds,
        "assembly_binding_id": composite.operator_id,
        "solver": None,
    }
    return _OversetSolver(
        composite,
        policy,
        compiled,
        rows,
        analytic,
        factorizations,
        symbolic_parts,
        compiler,
        {
            "solver_preparation": preparation["wall_seconds"],
            "solver_assembly": assembly["wall_seconds"],
            "solver_lowering": lowering["wall_seconds"],
            "solver_compile": compilation["wall_seconds"],
        },
        problems,
        residuals,
        route,
    )


def _flux_record(discrete: float, analytic: float | None, /) -> dict[str, object] | None:
    if analytic is None:
        return None
    return {
        "discrete_consistent_reaction": discrete,
        "analytic": analytic,
        "relative_error": abs(discrete - analytic) / abs(analytic),
    }


def _overset_composite_update(
    connectivity: phx.meshing.OversetConnectivity,
    values: dict[str, jax.Array],
    solver: _OversetSolver,
    *,
    reused: bool,
) -> tuple[dict[str, jax.Array], dict[str, Any], float]:
    """Solve the composite overset Poisson update and check it independently."""
    composite, rows = solver.composite, solver.rows
    if (
        solver.route.connectivity.connectivity_id != connectivity.connectivity_id
        or composite.coupling.operator_id != solver.route.transfer_id
    ):
        raise qualification_failure(
            "_overset_composite_update",
            ("Composite numerical actions must bind the current registration and field query."),
            {"registration": connectivity.connectivity_id, "query": solver.route.transfer_id},
        )
    names = composite.names
    seed = jax.device_put(
        np.concatenate([np.asarray(values[name])[rows[name].free] for name in names])
    )
    started = perf_counter()
    current, solved = solver.compiled(composite, seed, solver.policy)
    current = jax.block_until_ready(current)
    execution = perf_counter() - started
    evidence = _solver_evidence(solved)
    iterations = int(np.max(np.asarray(solved.diagnostics.iterations)))
    preconditioned = solver.policy.preconditioning is not None
    applications = iterations if preconditioned else 0
    if evidence["successful"] is not True:
        raise qualification_failure(
            "_overset_composite_update",
            ("Native composite overset solve did not converge within its cap."),
            {
                "solver": evidence,
                "part_factorized_solves": {name: applications for name in names},
                "total_part_factorized_solves": applications * len(names),
                "preconditioner_composite_actions": applications * (len(names) - 1),
            },
        )
    filled = solver.route.apply(
        {name: current[name] for name in solver.route.donor_names}
    )
    interface = max(
        (
            float(
                np.max(
                    np.abs(
                        np.asarray(current[name])[rows[name].receptor_rows]
                        - np.asarray(filled[name])
                    )
                )
            )
            for name in names
            if rows[name].receptor_rows.size
        ),
        default=0.0,
    )
    reports: dict[str, Any] = {}
    passed = interface <= _INTERFACE_TOLERANCE
    for name, problem, evaluate in zip(
        names, solver.problems, solver.residuals, strict=True
    ):
        part = rows[name]
        residual = np.asarray(jax.block_until_ready(evaluate(problem, current[name])))
        state = np.asarray(current[name])
        equations = np.setdiff1d(part.free, part.receptor_rows)
        constrained = np.asarray(part.constraint.constrained_dofs)
        free_residual = float(np.max(np.abs(residual[equations]), initial=0.0))
        lift_error = float(
            np.max(np.abs(state[constrained] - np.asarray(problem.lift)[constrained]))
        )
        boundary_error = float(
            np.max(np.abs(state[part.physical_rows] - np.asarray(part.physical_values)))
        )
        area = solver.analytic[name]["domain"]
        if area is None:
            raise ValueError(f"overset.analytic.domain must be set for {name!r}")
        balance = abs(float(np.sum(residual)) + area)
        layout = _part_layout(connectivity.assembly.part(name))
        active = np.asarray(connectivity.blanking_of(name).cell_status) != int(
            OversetCellStatus.HOLE
        )
        active_vertices = np.unique(layout.cells[active].reshape((-1,)))
        error = _manufactured_l2_error(layout.points, layout.cells[active], state)
        nodal = float(
            np.max(
                np.abs(
                    state[active_vertices]
                    - _manufactured(layout.points[active_vertices])
                )
            )
        )
        part_passed = (
            bool(np.all(np.isfinite(residual)))
            and free_residual <= _RESIDUAL_TOLERANCE
            and lift_error <= _LIFT_TOLERANCE
            and boundary_error <= _LIFT_TOLERANCE
            and balance <= _BALANCE_TOLERANCE * max(1.0, area)
            and math.isfinite(error)
        )
        passed = passed and part_passed
        reports[name] = {
            "passed": part_passed,
            "factorization": solver.factorizations.get(name),
            "matrix_free_free_residual_maximum": free_residual,
            "constrained_lift_maximum_error": lift_error,
            "physical_boundary_maximum_error": boundary_error,
            "flux_balance_defect": balance,
            "analytic_domain_area": area,
            "wall_flux": _flux_record(
                float(np.sum(residual[part.wall_rows])),
                solver.analytic[name]["wall_flux"],
            ),
            "exterior_flux": _flux_record(
                float(np.sum(residual[part.exterior_rows])),
                solver.analytic[name]["exterior_flux"],
            ),
            "manufactured_l2_error": error,
            "manufactured_nodal_maximum_error": nodal,
            "measured_cells": int(np.count_nonzero(active)),
            "unknown_rows": int(part.free.size),
            "constrained_rows": int(constrained.size),
            "receptor_rows": int(part.receptor_rows.size),
            "hole_rows": int(np.count_nonzero(part.hole)),
            "retained_solver_bytes": model_array_bytes(problem),
        }
    tolerance = solver.policy.tolerance
    report = {
        "passed": passed,
        "method": (
            "native-GMRES-right-preconditioned-multiplicative-subspace-correction"
            if preconditioned else "native-GMRES-unpreconditioned"
        ),
        "local_action_realization": solver.compiler["local_action_realization"],
        "manufactured_solution": _MANUFACTURED_SOLUTION,
        "error_quantity": _ERROR_QUANTITY,
        "reused_compiled_executable": reused,
        "solver": evidence,
        "execution_seconds": execution,
        "iteration": {
            "semantics": (
                "Krylov iterations of one composite system, not fixed-point sweeps. "
                "When part-block preconditioning is requested, each application "
                "executes one forward multiplicative sweep and applies each part "
                "factorization once with one composite action between parts."
            ),
            "krylov_iterations": iterations,
            "maximum_iterations": tolerance.max_steps,
            "absolute_tolerance": tolerance.absolute,
            "relative_tolerance": tolerance.relative,
            "parts": len(names),
            "part_factorized_solves_per_sweep": len(names) if preconditioned else 0,
            "composite_actions_per_sweep": len(names) - 1 if preconditioned else 0,
            "preconditioner_applications": applications,
            "part_factorized_solves": {name: applications for name in names},
            "total_part_factorized_solves": applications * len(names),
            "preconditioner_composite_actions": applications * (len(names) - 1),
            "work_count_scope": "executed early-exit native Arnoldi steps",
            "unknowns": composite.offsets[-1],
        },
        "interface_consistency_maximum_error": interface,
        "parts": reports,
        "physical_error": max(item["manufactured_l2_error"] for item in reports.values()),
    }
    if not passed:
        raise qualification_failure(
            "_overset_composite_update",
            (
                "Composite overset update failed residual, lift, interface or flux-balance checks."
            ),
            {"report": report},
        )
    return current, report, execution


def _overset_epoch_stages(
    solver: _OversetSolver, execution: float, epoch: int, /
) -> dict[str, float]:
    """Attribute the composite solve to the canonical epoch stage keys."""
    stages = {**solver.stages, "solver_first_execution": execution}
    return {epoch_stage_name(stage, epoch): seconds for stage, seconds in stages.items()}


# ------------------------------------------------------- interpolative route


def _overset_transfer_oracles(
    route: Any,
    candidate: phx.meshing.OversetConnectivity,
    fields: dict[str, Any],
    values: dict[str, jax.Array],
    step: int,
) -> dict[str, object]:
    """Constants, affine reproduction, adjoint duality and measured nonconservation."""
    names = route.donor_names
    coordinates = {
        name: np.asarray(
            fields[name].dof_maps[fields[name].field_index("u")].dof_coordinates
        )
        for name in names
    }
    sites = {
        item.part_name: np.asarray(
            _part_carrier(candidate.assembly.part(item.part_name)).mesh.coordinates
        )[np.asarray(item.receptor_rows)]
        for item in route.receptors
    }
    constant = route.apply(
        {
            name: jax.device_put(np.ones(values[name].shape, dtype=values[name].dtype))
            for name in names
        }
    )
    constant_error = max(
        (
            float(np.max(np.abs(np.asarray(value) - 1.0)))
            for value in constant.values()
            if value.size
        ),
        default=0.0,
    )
    affine = route.apply(
        {
            name: jnp.asarray(1.25 + 0.75 * points[:, 0] - 0.5 * points[:, 1])
            for name, points in coordinates.items()
        }
    )
    affine_error, affine_scale = 0.0, 1.0
    for name, points in sites.items():
        if not points.size:
            continue
        expected = 1.25 + 0.75 * points[:, 0] - 0.5 * points[:, 1]
        affine_error = max(
            affine_error, float(np.max(np.abs(np.asarray(affine[name]) - expected)))
        )
        affine_scale = max(affine_scale, float(np.max(np.abs(expected))))
    generator = np.random.default_rng(20261020 + step)
    duality = route.duality_evidence(
        {
            name: jnp.asarray(generator.standard_normal(values[name].shape))
            for name in names
        },
        {
            item.part_name: jnp.asarray(
                generator.standard_normal((item.receptor_ids.size,))
            )
            for item in route.receptors
        },
        tolerance=1.0e-12,
    )
    quadratic = route.apply(
        {name: jnp.asarray(np.sum(points * points, axis=1)) for name, points in coordinates.items()}
    )
    inventory_defect, jensen_minimum = 0.0, math.inf
    for item in route.receptors:
        points = sites[item.part_name]
        if not points.size:
            continue
        part = candidate.assembly.part(item.part_name)
        mesh = _part_carrier(part).mesh
        cells = _mesh_triangles(mesh)
        areas = np.abs(_simplex_measures(np.asarray(mesh.coordinates), cells))
        lumped = np.zeros(np.asarray(mesh.coordinates).shape[0], dtype=np.float64)
        np.add.at(lumped, cells.reshape((-1,)), np.repeat(areas / 3.0, 3))
        gap = np.asarray(quadratic[item.part_name]) - np.sum(points * points, axis=1)
        inventory_defect += float(np.sum(lumped[np.asarray(item.receptor_rows)] * gap))
        jensen_minimum = min(jensen_minimum, float(np.min(gap)))
    passed = (
        route.conservative is False
        and constant_error <= 1.0e-12
        and affine_error <= 1.0e-11 * affine_scale
        and bool(duality.valid)
        and inventory_defect > 0.0
        and jensen_minimum >= -1.0e-12
    )
    return {
        "passed": passed,
        "declared_conservative": route.conservative,
        "constant_reproduction_maximum_error": constant_error,
        "affine_reproduction_maximum_error": affine_error,
        "affine_scale": affine_scale,
        "transpose_duality": {
            "primal_pairing": float(duality.primal_pairing),
            "transpose_pairing": float(duality.transpose_pairing),
            "residual": float(duality.residual),
            "valid": bool(duality.valid),
        },
        "quadratic_receptor_lumped_inventory_defect": inventory_defect,
        "quadratic_minimum_interpolation_gap": jensen_minimum
        if np.isfinite(jensen_minimum)
        else None,
        "partition_of_unity_is_not_conservation": inventory_defect > 0.0
        and constant_error <= 1.0e-12,
    }


# --------------------------------------------------- conservative ALE route


def _triangle_minimum_angles(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    corners = points[cells]
    angles = []
    for vertex in range(3):
        first = corners[:, (vertex + 1) % 3] - corners[:, vertex]
        second = corners[:, (vertex + 2) % 3] - corners[:, vertex]
        cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        angles.append(np.arctan2(cross, np.sum(first * second, axis=1)))
    return np.min(np.stack(angles, axis=1), axis=1)


def _overset_ale_coordinates(
    mesh: phx.discretization.CellMesh,
    specification: phx.meshing.SurfaceMeshingSpec,
    direction: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Request-limited interior ALE displacement following the first body motion.

    ``x + A psi(x) s`` with ``psi = ((1-(x/R)^2)(1-(y/R)^2))^2`` vanishing on the
    fixed exterior and ``A = A0 2^-k`` for the least ``k`` that keeps every triangle
    positive, consumes at most half of the generated minimum-angle margin, and
    preserves every authored hard size statistic and edge bound.
    """
    incidence = mesh.connectivity
    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError(
            f"overset.ale.connectivity must be PolygonalConnectivity; got {type(incidence).__name__}"
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    boundary = np.unique(
        np.asarray(incidence.edges)[np.asarray(incidence.boundary_edges)]
    )
    scaled = points / _BACKGROUND_HALF_WIDTH
    bump = ((1.0 - scaled[:, 0] ** 2) * (1.0 - scaled[:, 1] ** 2)) ** 2
    bump[boundary] = 0.0
    unit = bump[:, None] * direction[None, :]
    cells = np.asarray(_mesh_triangles(mesh), dtype=np.int64)
    quality = specification.quality_target
    if quality is None:
        raise ValueError("overset.ale requires the authored hard quality target")
    minimum_angle = quality.minimum_angle
    edges = np.asarray(incidence.edges, dtype=np.int64)
    controls: list[phx.meshing.UniformSizeControl] = []
    for control in specification.size_controls:
        if not isinstance(control, phx.meshing.UniformSizeControl):
            raise TypeError("overset.ale requires the authored uniform size controls")
        controls.append(control)
    generated = float(np.min(_triangle_minimum_angles(points, cells)))
    margin = generated - minimum_angle
    if not np.isfinite(margin) or margin <= 0.0:
        raise qualification_failure(
            "_overset_ale_coordinates",
            ("Generated background quality leaves no ALE angle margin."),
            {"generated": generated, "minimum_angle": minimum_angle},
        )
    floor = minimum_angle + 0.5 * margin
    for halvings in range(64):
        scale = amplitude * 2.0**-halvings
        moved = points + scale * unit
        moved[boundary] = points[boundary]
        deformed = float(np.min(_triangle_minimum_angles(moved, cells)))
        lengths = np.linalg.norm(moved[edges[:, 1]] - moved[edges[:, 0]], axis=1)
        size_issues = [
            issue
            for control in controls
            for issue in graded_size_compliance(
                control,
                specification.size_compliance,
                lengths,
                np.full(lengths.shape, control.target_size),
                1.0,
            )[2]
        ]
        if (
            np.all(_simplex_measures(moved, cells) > 0.0)
            and deformed >= floor
            and not size_issues
        ):
            break
    else:
        raise qualification_failure(
            "_overset_ale_coordinates",
            ("No representable ALE amplitude preserves the authored quality and size controls."),
            {"margin": margin},
        )
    displaced = np.any(moved != points, axis=1)
    interior = np.setdiff1d(np.arange(points.shape[0]), boundary)
    if not np.any(displaced[interior]):
        raise qualification_failure(
            "_overset_ale_coordinates",
            ("Quality-limited ALE amplitude produced no represented interior motion."),
            {"scale": scale},
        )
    return moved, {
        "law": "x+A*psi(x)*s, psi=((1-(x/R)^2)(1-(y/R)^2))^2, A=A0*2^-k",
        "direction": direction.tolist(),
        "requested_amplitude": amplitude,
        "halvings": halvings,
        "amplitude": scale,
        "hard_minimum_angle": minimum_angle,
        "generated_minimum_angle": generated,
        "deformed_minimum_angle": deformed,
        "consumed_margin_fraction_bound": 0.5,
        "size_compliance_policy_id": specification.size_compliance.policy_id,
        "size_control_ids": [control.control_id for control in controls],
        "size_compliance_issues": size_issues,
        "deformed_maximum_edge": float(np.max(lengths)),
        "maximum_displacement": float(np.max(np.abs(moved - points))),
        "interior_vertices": int(interior.size),
        "displaced_interior_vertices": int(np.count_nonzero(displaced[interior])),
    }


class _OversetConservative(NamedTuple):
    target_states: tuple[phx.lifecycle.CompositionEntry, ...]
    source: phx.meshing.CellMeshingResult
    source_states: tuple[phx.lifecycle.CompositionEntry, ...]
    motion: phx.meshing.CellMeshTransition
    prepared_operands: PreparedCoordinateOperands | None


def _overset_common_refinement_policy(
    pair_capacity: int, maximum_scratch_bytes: int,
) -> phx.geometry.CommonRefinementPolicy:
    """Use the same authored scratch owner for warm and archive-restored remaps."""
    return phx.geometry.CommonRefinementPolicy(
        maximum_candidate_pairs=pair_capacity,
        maximum_accepted_pairs=pair_capacity,
        maximum_memory_bytes=maximum_scratch_bytes,
        coverage_tolerance=1.0e-10,
    )


def _overset_qualification_conservative(
    connectivity: phx.meshing.OversetConnectivity,
    plan: phx.meshing.NativeMeshingPlan,
    profile: _OversetProfile,
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    deadline: Deadline,
) -> _OversetConservative:
    """Deform the background interior and commit physical/material/history content."""

    _ledger = StageLedger(record, deadline=deadline)
    background = connectivity.assembly.part(_BACKGROUND)
    source_carrier = background.carrier
    if (
        not isinstance(source_carrier, phx.meshing.CellMeshingResult)
        or source_carrier.certification is None
    ):
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("Moving overset inventory requires a certified background carrier."),
            {"carrier": type(source_carrier).__name__},
        )
    specification = plan.specification
    if not isinstance(specification, phx.meshing.SurfaceMeshingSpec):
        raise TypeError("Planar ALE requires a surface meshing specification.")
    quality = specification.quality_target
    if quality is None:
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("ALE motion requires the hard background quality target."),
            {"plan": plan.plan_id},
        )
    first = profile.bodies[0].offset(0)
    amplitude = float(np.linalg.norm(first))
    coordinates, ale = _ledger.run(
        "conservative_quality_limited_ale_law",
        lambda: _overset_ale_coordinates(
            source_carrier.mesh, specification, first / amplitude, amplitude
        ),
    )
    motions: list[phx.meshing.CellMeshTransition] = []
    coordinate_operands = CoordinateOperandCollector(capture_measures=True)
    try:
        target = _ledger.run(
            "separate_conservative_native_motion_publication",
            lambda: deform_planar_interior(
                background,
                coordinates,
                motion_id=f"qualification-conservative-ale-{profile.name}",
                plan=plan,
                record_phase=phases.append,
                record_motion=motions.append,
                coordinate_operands=coordinate_operands,
            ),
        )
    finally:
        if coordinate_operands.pending:
            coordinate_operands.abort()
    target_carrier = target.carrier
    if (
        not isinstance(target_carrier, phx.meshing.CellMeshingResult)
        or target_carrier.certification is None
    ):
        raise qualification_failure(
            "_overset_qualification_conservative",
            ("Moving overset inventory requires successor source certification."),
            {"carrier": type(target_carrier).__name__},
        )
    previous = connectivity
    candidate = _ledger.run(
        "separate_conservative_reregistration",
        lambda: previous.reregister({_BACKGROUND: target}),
    )
    candidate.require_complete()
    source_entry = {
        entry.entry_id: entry for entry in previous.composition_entries()
    }[f"overset:part:{background.name}"]
    target_entry = {
        entry.entry_id: entry for entry in candidate.composition_entries()
    }[f"overset:part:{target.name}"]
    source_mesh, target_mesh = source_carrier.mesh, target_carrier.mesh
    source_cells = _mesh_triangles(source_mesh)
    source_points = np.asarray(source_mesh.coordinates)
    centers = source_points[source_cells].mean(axis=1)
    density = jnp.asarray(2.0 + centers[:, 0] / 8.0 - centers[:, 1] / 10.0)
    fraction = jnp.asarray(0.5 + centers[:, 0] / 20.0)
    states = tuple(
        phx.lifecycle.CompositionEntry(
            value,
            entry_id=f"conservative:{name}",
            role=role,
            owner_id="qualification-native-moving-inventory",
            structure_id=source_entry.structure_id,
            revision_id=f"accepted:{name}",
            semantics_id=meaning,
            dependencies=(source_entry.binding("revision"),),
        )
        for name, role, meaning, value in zip(
            ("density", "materials", "history"),
            ("physical-state", "model-state", "history"),
            (
                "normalized-cell-density",
                "two-material-volume-fractions",
                "spatial-density-history",
            ),
            (
                density[:, None],
                jnp.stack((fraction, 1.0 - fraction), axis=1),
                jnp.stack((density, density**2), axis=1),
            ),
            strict=True,
        )
    )
    composition = phx.lifecycle.Composition(
        (*previous.composition_entries(), *states),
        boundary_id="accepted-conservative-state",
    )
    policy = _overset_common_refinement_policy(
        args.overset_overlap_pairs, specification.limits.maximum_scratch_bytes,
    )
    prepared, transports = _ledger.run(
        "separate_moving_conservative_state_transport",
        lambda: prepare_overset_conservative_state_transport(
            source_entry,
            target_entry,
            states,
            content_tolerance=_CONTENT_TOLERANCE,
            policy=policy,
        ),
    )
    evidence, remap = prepared.evidence, prepared.plan
    if not isinstance(evidence, CommonRefinementEvidence) or remap is None:
        raise TypeError(
            f"overset.transport must include CommonRefinementEvidence and a remap route; got evidence={type(evidence).__name__}, remap={remap!r}"
        )

    def commit() -> tuple[Any, Any, Any]:
        motion = prepare_overset_motion_rebind(
            composition, previous, candidate, transports=transports
        )
        tamper = transports[0]
        if tamper.target_content is None or tamper.source_content is None:
            raise qualification_failure(
                "_overset_qualification_conservative",
                ("Conservative transport must report source and target content."),
                {"transport": tamper.transport_id},
            )
        nonconserving = phx.lifecycle.CompositionTransport(
            "physical-remap",
            tamper.source_entry_ids,
            tamper.targets,
            source_structure_ids=tamper.source_structure_ids,
            route_id=tamper.route_id,
            successful=tamper.successful,
            source_content=tamper.source_content,
            target_content=tamper.target_content
            + 1.0e-6 * jnp.maximum(1.0, jnp.abs(tamper.target_content)),
            content_tolerance=tamper.content_tolerance,
        )
        tampered = prepare_overset_motion_rebind(
            composition,
            previous,
            candidate,
            transports=(nonconserving, *transports[1:]),
        ).commit(accepted_boundary=True)
        return (
            motion.commit(accepted_boundary=False),
            tampered,
            motion.commit(accepted_boundary=True),
        )

    refused, tampered, receipt = _ledger.run(
        "separate_conservative_rebind_and_refusals", commit
    )
    refusals_atomic = (
        not refused.published
        and refused.composition is composition
        and not tampered.published
        and tampered.composition is composition
    )
    if not refusals_atomic or not receipt.published:
        raise qualification_failure(
            "_overset_qualification_conservative",
            (
                "Moving conservative state violated atomic refusal or accepted publication."
            ),
            {"receipt": receipt, "refused": refused, "tampered": tampered},
        )
    source_measures = _simplex_measures(source_points, source_cells)
    target_cells = _mesh_triangles(target_mesh)
    target_points = np.asarray(target_mesh.coordinates)
    target_measures = _simplex_measures(target_points, target_cells)
    inventories = []
    for old, new in zip(
        states, tuple(item.targets[0] for item in transports), strict=True
    ):
        trailing = (1,) * (old.value.ndim - 1)
        old_content = np.sum(
            source_measures.reshape((-1,) + trailing) * np.asarray(old.value), axis=0
        )
        new_content = np.sum(
            target_measures.reshape((-1,) + trailing) * np.asarray(new.value), axis=0
        )
        defect = float(np.max(np.abs(old_content - new_content)))
        if defect > _CONTENT_TOLERANCE or new.semantics_id != old.semantics_id:
            raise qualification_failure(
                "_overset_qualification_conservative",
                (
                    "Moving conservative state lost physical/material/history content or identity."
                ),
                {"defect": defect, "entry": old.entry_id},
            )
        inventories.append(
            {
                "entry_id": old.entry_id,
                "role": old.role,
                "semantics_id": old.semantics_id,
                "source": np.asarray(old_content).tolist(),
                "target": np.asarray(new_content).tolist(),
                "independent_inventory_absolute_error": defect,
            }
        )
    material = receipt.composition.entry("conservative:materials").value
    target_density = np.asarray(
        receipt.composition.entry("conservative:density").value
    )[:, 0]
    authority = plan.source
    if not isinstance(authority, phx.meshing.NativePlanarSource):
        raise TypeError(
            f"overset.plan.source must be NativePlanarSource; got {type(authority).__name__}"
        )
    edge_points = np.asarray(authority.region.vertices)[
        np.asarray(authority.region.edges)
    ]
    first_points, second_points = edge_points[:, 0], edge_points[:, 1]
    cross = first_points[:, 0] * second_points[:, 1] - second_points[:, 0] * first_points[:, 1]
    analytic_inventory = float(
        np.sum(cross)
        + np.sum((first_points[:, 0] + second_points[:, 0]) * cross) / 48.0
        - np.sum((first_points[:, 1] + second_points[:, 1]) * cross) / 60.0
    )
    source_inventory = float(np.sum(source_measures * np.asarray(density)))
    constant_error = float(jnp.max(jnp.abs(remap.apply(jnp.ones_like(density)) - 1.0)))
    target_centers = target_points[target_cells].mean(axis=1)
    exact_target = 2.0 + target_centers[:, 0] / 8.0 - target_centers[:, 1] / 10.0

    def longest(points: np.ndarray, cells: np.ndarray) -> float:
        corners = points[cells]
        return float(
            np.max(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2))
        )

    # A complete first-order overlap average is a convex combination of source
    # averages whose cells meet the target cell: the centroid shift is bounded by
    # the two longest edges, so the linear density error is bounded by |grad rho|.
    accuracy_bound = math.hypot(1.0 / 8.0, 1.0 / 10.0) * (
        longest(source_points, source_cells) + longest(target_points, target_cells)
    )
    accuracy_error = float(np.max(np.abs(target_density - exact_target)))
    density_values = np.asarray(density)
    monotone = bool(
        np.all(target_density >= float(np.min(density_values)) - 1.0e-12)
        and np.all(target_density <= float(np.max(density_values)) + 1.0e-12)
    )
    coverage = (
        evidence.source_gap_count,
        evidence.target_gap_count,
        evidence.source_double_count,
        evidence.target_double_count,
    )
    certified = (
        all(count == 0 for count in coverage)
        and abs(source_inventory - analytic_inventory) <= _CONTENT_TOLERANCE
        and constant_error <= 1.0e-10
        and float(jnp.max(jnp.abs(jnp.sum(material, axis=1) - 1.0))) <= 1.0e-10
        and float(jnp.min(material)) >= 0.0
        and float(np.min(target_density)) > 0.0
        and monotone
        and accuracy_error <= accuracy_bound + 1.0e-12
    )
    record["conservative_profile"] = {
        "profile": "moving-native-background-ALE-cell-average-complete-overlap",
        "moving_fe_transfer_certified": False,
        "moving_cell_average_transfer_certified": certified,
        "certified": certified,
        "physical_boundary_moved": False,
        "ale": ale,
        "source_geometry_id": source_mesh.geometry_id,
        "target_geometry_id": target_mesh.geometry_id,
        "policy_id": policy.policy_id,
        "remap_id": prepared.remap_id,
        "status": prepared.status.name,
        "evidence_id": evidence.evidence_id,
        "candidate_pair_count": evidence.candidate_pair_count,
        "accepted_pair_count": evidence.accepted_pair_count,
        "piece_pair_count": evidence.piece_pair_count,
        "source_gap_count": evidence.source_gap_count,
        "target_gap_count": evidence.target_gap_count,
        "source_double_count": evidence.source_double_count,
        "target_double_count": evidence.target_double_count,
        "retained_bytes": evidence.retained_bytes,
        "working_bytes": evidence.working_bytes,
        "inventories": inventories,
        "transport_ids": [item.transport_id for item in transports],
        "analytic_source_inventory": analytic_inventory,
        "source_inventory": source_inventory,
        "minimum_target_average": float(np.min(target_density)),
        "monotone_bounds_preserved": monotone,
        "linear_density_accuracy_error": accuracy_error,
        "linear_density_accuracy_bound": accuracy_bound,
        "constant_reproduction_error": constant_error,
        "boundary_refusal_preserved": refused.composition is composition,
        "nonconserving_transport_refused": not tampered.published,
        "refusals_atomic": refusals_atomic,
        "accepted_composition_id": receipt.composition.composition_id,
        "source_certification_id": source_carrier.certification.report_id,
        "target_certification_id": target_carrier.certification.report_id,
        "registration_epoch": candidate.epoch,
    }
    if not certified:
        raise qualification_failure(
            "_overset_qualification_conservative",
            (
                "Moving conservative inventory failed coverage, content, constants, bounds or accuracy."
            ),
            {"profile": record["conservative_profile"]},
        )
    if len(motions) != 1:
        raise qualification_failure(
            "overset.conservative.motion",
            "Accepted ALE publication must retain its one actual source-to-target motion transition.",
            {"recorded_transitions": len(motions)},
        )
    geometry_motion = motions[0].geometry_transition
    if geometry_motion is None:
        raise ValueError("overset.conservative.motion must retain its actual source-authoritative coordinate transition")
    record["conservative_profile"]["motion_transition_id"] = motions[0].transition_id
    record["conservative_profile"]["geometry_motion_transition_id"] = geometry_motion.transition_id
    record["conservative_profile"]["source_geometry_id"] = geometry_motion.source_geometry_id
    record["conservative_profile"]["target_geometry_id"] = geometry_motion.target_geometry_id
    record["conservative_profile"]["motion_preserves_topology"] = (
        motions[0].source_topology_id == target_carrier.mesh.topology_id
    )
    prepared_operands = coordinate_operands.for_result(
        target_carrier.mesh, target_carrier.geometry
    )
    record["conservative_profile"]["prepared_numeric_operands_issued"] = prepared_operands is not None
    return _OversetConservative(
        tuple(
            receipt.composition.entry(f"conservative:{name}")
            for name in ("density", "materials", "history")
        ),
        source_carrier, states, motions[0], prepared_operands,
    )


# ------------------------------------------------------------ epochs/motion


class _OversetAccepted(NamedTuple):
    connectivity: phx.meshing.OversetConnectivity
    composition: phx.lifecycle.Composition
    plans: dict[str, Any]
    fields: dict[str, Any]
    values: dict[str, jax.Array]
    solver: _OversetSolver
    route: Any
    revision: str


def _overset_qualification_fields(
    connectivity: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain the declared physical plans without inferring them from coefficients."""

    plans = field_plans(connectivity)
    return plans, {name: plan.prepare() for name, plan in plans.items()}


def _overset_accept_epoch(
    connectivity: phx.meshing.OversetConnectivity,
    plans: dict[str, Any],
    fields: dict[str, Any],
    seeds: dict[str, jax.Array],
    label: str,
    epoch: dict[str, Any],
    profile: _OversetProfile,
    record: dict[str, Any],
    deadline: Deadline,
    *,
    previous: _OversetSolver | None = None,
    reuse: phx.meshing.PreparedOversetFieldTransfer | None = None,
) -> _OversetAccepted:
    """Run the composite PDE update of one registration epoch and publish it."""
    _ledger = StageLedger(record, deadline=deadline)
    route = _ledger.run(
        f"{label}:coupling_field_query",
        lambda: prepare_overset_field_transfer(connectivity, fields, "u", reuse=reuse),
    )
    analytic = {
        _BACKGROUND: _body_areas(None),
        **{body.name: _body_areas(body) for body in profile.bodies},
    }
    solver = _ledger.run(
        f"{label}:composite_solver_preparation",
        lambda: _overset_prepare_solver(
            connectivity, fields, seeds, route, analytic,
            int(record["control_capacities"]["entity_capacity"]),
            previous=previous,
        ),
    )
    values, report, execution = _ledger.run(
        f"{label}:composite_schwarz_poisson",
        lambda: _overset_composite_update(connectivity, seeds, solver, reused=False),
    )
    record["stages_seconds"].update(
        _overset_epoch_stages(solver, execution, epoch["epoch"])
    )
    epoch["compiler"] = {**solver.compiler, "solver": report["solver"]}
    epoch["poisson"] = report
    epoch["retained_bytes"] = model_array_bytes(
        (connectivity, fields, values, solver.composite, solver.policy, solver.problems, solver.route)
    )
    revision = f"{label}-poisson"
    prepared, states = field_entries(connectivity, fields, values, revision)
    composition = phx.lifecycle.Composition(
        (*connectivity.composition_entries(), *prepared, *states),
        boundary_id=_ACCEPTED_BOUNDARY,
    )
    epoch["accepted_composition_id"] = composition.composition_id
    return _OversetAccepted(
        connectivity, composition, plans, fields, values, solver, route, revision
    )


def _overset_qualification_motion(
    accepted: _OversetAccepted,
    successors: dict[str, phx.meshing.MeshPart],
    step: int,
    profile: _OversetProfile,
    record: dict[str, Any],
    deadline: Deadline,
) -> _OversetAccepted:
    """Refresh holes/donors, transfer accepted state, refuse faults, then update."""

    _ledger = StageLedger(record, deadline=deadline)
    connectivity, composition = accepted.connectivity, accepted.composition
    candidate = _ledger.run(
        f"motion-{step}:wall_cutting_donor_refresh",
        lambda: connectivity.reregister(successors),
    )
    epoch: dict[str, Any] = {
        # Initial geometry is epoch zero; each motion publishes its successor.
        "epoch": step + 1,
        "step": step,
        "offsets": {body.name: body.offset(step).tolist() for body in profile.bodies},
        "source_successors": {
            name: {
                "result_id": _part_carrier(part).result_id,
                "certification_id": _part_certification(part).report_id,
            }
            for name, part in successors.items()
        },
        "registration": _overset_qualification_registration_evidence(candidate),
    }
    record["epochs"].append(epoch)
    candidate.require_complete()
    oracle = _ledger.run(
        f"motion-{step}:independent_registration_oracle",
        lambda: _overset_registration_oracle(candidate),
    )
    epoch["independent_registration_oracle"] = oracle
    _overset_require_oracle(oracle, epoch["epoch"])
    reused = {packet.packet_id for packet in connectivity.packets} & {
        packet.packet_id for packet in candidate.packets
    }
    revisions = {part.name: part.part_id for part in candidate.assembly.parts}
    epoch["motion_refresh"] = {
        "predecessor_id": candidate.predecessor_id,
        "connectivity_id": candidate.connectivity_id,
        "reused_donor_packets": len(reused),
        "packets_bound_to_successor_revisions": all(
            packet.donor_revision == revisions[packet.donor_part]
            and packet.receptor_revision == revisions[packet.receptor_part]
            for packet in candidate.packets
        ),
        "refreshed": not reused
        and candidate.predecessor_id == connectivity.connectivity_id
        and candidate.epoch == connectivity.epoch + 1
        and candidate.connectivity_id != connectivity.connectivity_id,
    }
    if epoch["motion_refresh"]["refreshed"] is not True:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("Motion reused donor evidence or skipped a registration epoch."),
            {"refresh": epoch["motion_refresh"]},
        )
    next_plans, next_fields = _ledger.run(
        f"motion-{step}:field_and_transfer_preparation",
        lambda: _overset_qualification_fields(candidate),
    )
    route = _ledger.run(
        f"motion-{step}:prepared_field_query",
        lambda: prepare_overset_field_transfer(
            candidate, next_fields, "u", previous=connectivity, reuse=accepted.route
        ),
    )
    coefficients = accepted.values
    receptors = _ledger.run(
        f"motion-{step}:state_transfer",
        lambda: jax.block_until_ready(
            route.apply({name: coefficients[name] for name in route.donor_names})
        ),
    )
    transfer = _ledger.run(
        f"motion-{step}:independent_transfer_oracles",
        lambda: _overset_transfer_oracles(
            route, candidate, next_fields, coefficients, step
        ),
    )
    if transfer["passed"] is not True:
        raise qualification_failure(
            "_overset_qualification_motion",
            (
                "FE interpolation lost constants/affine reproduction/duality or claimed conservation."
            ),
            {"transfer": transfer},
        )
    transported = dict(coefficients)
    for item in route.receptors:
        transported[item.part_name] = (
            transported[item.part_name]
            .at[np.asarray(item.receptor_rows)]
            .set(receptors[item.part_name])
        )
    prepared, states = field_entries(
        candidate, next_fields, transported, f"motion-{step}"
    )
    old_states = tuple(
        composition.entry(f"temperature:{name}") for name in sorted(coefficients)
    )
    transport = phx.lifecycle.CompositionTransport(
        "physical-remap",
        tuple(entry.entry_id for entry in old_states),
        states,
        source_structure_ids=tuple(entry.structure_id for entry in old_states),
        route_id=route.transfer_id,
        successful=True,
    )
    failed = phx.lifecycle.CompositionTransport(
        "physical-remap",
        transport.source_entry_ids,
        transport.targets,
        source_structure_ids=transport.source_structure_ids,
        route_id=transport.route_id,
        successful=False,
    )

    def rebind(transports: tuple[Any, ...]) -> Any:
        return prepare_overset_motion_rebind(
            composition,
            connectivity,
            candidate,
            transports=transports,
            reprepare=prepared,
        )

    motion = _ledger.run(
        f"motion-{step}:composition_rebind", lambda: rebind((transport,))
    )
    refused = _ledger.run(
        f"motion-{step}:refused_boundary",
        lambda: motion.commit(accepted_boundary=False),
    )
    failed_receipt = _ledger.run(
        f"motion-{step}:failed_transport_refusal",
        lambda: rebind((failed,)).commit(accepted_boundary=True),
    )
    atomic = (
        not refused.published
        and refused.composition is composition
        and not failed_receipt.published
        and failed_receipt.composition is composition
    )
    if not atomic:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("Refused overset motion changed the accepted Composition."),
            {"failed": failed_receipt, "refused": refused},
        )
    receipt = _ledger.run(
        f"motion-{step}:accepted_boundary",
        lambda: motion.commit(accepted_boundary=True),
    )
    if not receipt.published:
        raise qualification_failure(
            "_overset_qualification_motion",
            ("Source-aware motion failed its accepted Composition boundary."),
            {"receipt": receipt},
        )
    epoch["rollback"] = {
        "boundary_refusal_preserved": refused.composition is composition,
        "failed_transport_refused": not failed_receipt.published,
        "atomic": atomic,
    }
    epoch["transfer"] = {
        "transfer_id": route.transfer_id,
        "prepared_queries": [
            {
                "query_id": query.query_id,
                "complete": query.complete,
                "coverage": query.coverage,
                "status": np.asarray(query.evidence.status).tolist(),
                "support_count": np.asarray(query.evidence.support_count).tolist(),
                "conditioning": [
                    float(value) if np.isfinite(value) else None
                    for value in np.asarray(query.evidence.conditioning)
                ],
                "unreported_conditioning_is_unavailable": True,
            }
            for query in route.queries
        ],
        "previous_connectivity_id": route.previous_id,
        "conservative": route.conservative,
        "oracles": transfer,
        "oracles_passed": transfer["passed"],
        "retained_transfer_bytes": model_array_bytes(route),
        "accepted_boundary_published": receipt.published,
        "target_composition_id": receipt.composition.composition_id,
    }
    next_accepted = _overset_accept_epoch(
        candidate,
        next_plans,
        next_fields,
        transported,
        f"motion-{step}",
        epoch,
        profile,
        record,
        deadline,
        previous=accepted.solver,
        reuse=route,
    )
    record["attempts"].append(
        {
            "phase": record["active_phase"],
            "status": "passed",
            "connectivity_id": candidate.connectivity_id,
        }
    )
    return next_accepted


def _overset_refused_overreach(
    accepted: _OversetAccepted,
    generation_plans: dict[str, phx.meshing.NativeMeshingPlan],
    profile: _OversetProfile,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    deadline: Deadline,
) -> None:
    """Move one certified body across the background boundary; refusal is atomic."""
    _ledger = StageLedger(record, deadline=deadline)
    body = profile.bodies[0]
    plan = generation_plans[body.name]
    source = plan.source
    if not isinstance(source, phx.meshing.NativePlanarSource):
        raise TypeError(
            f"overset.plan.source must be NativePlanarSource; got {type(source).__name__}"
        )
    outer = np.asarray(source.region.vertices)[:4]
    gaps = np.asarray(
        (
            _BACKGROUND_HALF_WIDTH + float(np.min(outer[:, 0])),
            _BACKGROUND_HALF_WIDTH - float(np.max(outer[:, 0])),
            _BACKGROUND_HALF_WIDTH + float(np.min(outer[:, 1])),
            _BACKGROUND_HALF_WIDTH - float(np.max(outer[:, 1])),
        )
    )
    side = int(np.argmin(gaps))
    reach = float(gaps[side]) + 0.25
    offset = np.zeros(2, dtype=np.float64)
    offset[side // 2] = -reach if side % 2 == 0 else reach
    connectivity, composition = accepted.connectivity, accepted.composition
    moved, _ = _ledger.run(
        "refused_overreach_motion:source_successor",
        lambda: rigid_planar_successor(
            connectivity.assembly.part(body.name),
            offset,
            motion_id=f"qualification-refused-overreach-{profile.name}",
            plan=plan,
            record_phase=phases.append,
        ),
    )
    candidate = _ledger.run(
        "refused_overreach_motion:wall_cutting_donor_refresh",
        lambda: connectivity.reregister({body.name: moved}),
    )
    fields = dict(accepted.fields)
    fields[body.name] = field_plans(candidate)[body.name].prepare()
    try:
        prepare_overset_field_transfer(candidate, fields, "u", previous=connectivity)
    except OversetConnectivityError as failure:
        route_refusal: str | None = str(failure)
    else:
        route_refusal = None
    prepared, states = field_entries(
        candidate, fields, accepted.values, "refused-overreach"
    )
    old_states = tuple(
        composition.entry(f"temperature:{name}") for name in sorted(accepted.values)
    )

    def commit() -> Any:
        return prepare_overset_motion_rebind(
            composition,
            connectivity,
            candidate,
            transports=(
                phx.lifecycle.CompositionTransport(
                    "physical-remap",
                    tuple(entry.entry_id for entry in old_states),
                    states,
                    source_structure_ids=tuple(
                        entry.structure_id for entry in old_states
                    ),
                    route_id="refused-overreach-material-attached",
                    successful=True,
                ),
            ),
            reprepare=prepared,
        ).commit(accepted_boundary=True)

    receipt = _ledger.run("refused_overreach_motion:incomplete_commit", commit)
    atomic = (
        not candidate.complete
        and route_refusal is not None
        and not receipt.published
        and receipt.composition is composition
    )
    record["refused_overreach_motion"] = {
        "body": body.name,
        "offset": offset.tolist(),
        "registration_complete": candidate.complete,
        "orphan_count": candidate.orphan_count,
        "protected_wall_conflict_count": candidate.conflict_count,
        "field_route_refusal": route_refusal,
        "commit_published": receipt.published,
        "atomic": atomic,
    }
    if not atomic:
        raise qualification_failure(
            "_overset_refused_overreach",
            ("Incomplete overreaching registration was not refused atomically."),
            {"evidence": record["refused_overreach_motion"]},
        )


# ------------------------------------------------------ checkpoint/restart


def _overset_archive_limits(capacity: int, /) -> ArrayArchiveLimits:
    byte_budget = 4096 * capacity
    return ArrayArchiveLimits(
        max_container_bytes=byte_budget,
        max_aggregate_bytes=byte_budget,
        max_member_bytes=byte_budget,
        max_manifest_bytes=1024 * capacity,
        max_members=64 * capacity,
        max_central_directory_bytes=1024 * capacity,
        max_axis_length=64 * capacity,
        max_array_elements=64 * capacity,
        max_total_array_elements=512 * capacity,
        max_manifest_nesting=128,
    )


def _overset_owner_identities(
    connectivity: phx.meshing.OversetConnectivity,
    fields: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, object]:
    names = tuple(sorted(values))
    return {
        "connectivity_id": connectivity.connectivity_id,
        "parts": list(names),
        "carriers": {
            name: {
                "result_id": _part_carrier(connectivity.assembly.part(name)).result_id,
                "certification_id": _part_certification(
                    connectivity.assembly.part(name)
                ).report_id,
            }
            for name in names
        },
        "field_space_ids": {
            name: {space.name: space.field_space_id for space in fields[name].field_spaces}
            for name in names
        },
        "values_fingerprint": array_tree_fingerprint(
            tuple(np.asarray(values[name]) for name in names)
        ),
    }


def _overset_restored_analytic(
    connectivity: phx.meshing.OversetConnectivity, profile: dict[str, Any]
) -> dict[str, dict[str, float | None]]:
    bodies = {
        item["name"]: _OversetBody(
            item["name"],
            tuple(item["center"]),
            item["half_width"],
            item["cavity_half_width"],
            item["refinement"],
            tuple(tuple(value) for value in item["offsets"]),
            item["offset_rule"],
        )
        for item in profile["bodies"]
    }
    return {
        part.name: _body_areas(bodies.get(part.name))
        for part in connectivity.assembly.parts
    }


class _OversetColdRemapRefusal(RuntimeError):
    """Carry the actual refused route across the cold process reporting boundary."""

    def __init__(
        self,
        remap: PreparedUnstructuredConservativeRemap,
        budget: CoordinateEnclosureBudget | None,
    ) -> None:
        evidence = remap.evidence
        self._scope = current_native_execution_scope()
        self.report: dict[str, Any] = {
            "status": remap.status.name,
            "reason": remap.reason,
            "remap_id": remap.remap_id,
            "route_type": type(remap.route).__name__,
            "evidence_type": type(evidence).__name__,
            "plan_present": remap.plan is not None,
            "plan_requires_complete": None if remap.plan is None else remap.plan.require_complete,
            "execution_evidence": None if remap.execution_evidence is None
            else remap.execution_evidence.to_record(),
            "coordinate_work_units": None if budget is None else budget.work_units,
            "coordinate_peak_bytes_upper": None if budget is None else budget.peak_bytes_upper,
        }
        if isinstance(evidence, CommonRefinementEvidence):
            self.report["coverage"] = {
                name: getattr(evidence, name)
                for name in (
                    "reason", "source_gap_count", "target_gap_count",
                    "source_double_count", "target_double_count",
                    "maximum_relative_source_defect", "maximum_relative_target_defect",
                    "candidate_pair_count", "accepted_pair_count", "piece_pair_count",
                    "uncertain_predicate_count", "invalid_cell_count",
                    "intersection_failure_count", "retained_bytes", "working_bytes",
                    "evidence_id",
                )
            }
            self.report["coverage"]["status"] = evidence.status.name
            self.report["coverage"].update({
                name: np.asarray(getattr(evidence, name)).tolist()
                for name in (
                    "source_coverage_defects", "target_coverage_defects",
                    "source_coverage_tolerances", "target_coverage_tolerances",
                )
            })
        elif isinstance(evidence, RemapPreparationFailure):
            self.report["preparation_failure"] = {
                "reason": evidence.reason,
                "source_discretization_id": evidence.source_discretization_id,
                "target_discretization_id": evidence.target_discretization_id,
                "logical_exact_work_units": int(np.asarray(evidence.logical_exact_work_units)),
                "host_storage_upper_bytes": int(np.asarray(evidence.host_storage_upper_bytes)),
                "execution_evidence": evidence.execution_evidence.to_record(),
            }
        super().__init__(
            "overset.cold.conservative requires an actual complete-overlap remap; "
            f"got status={remap.status.name}, reason={remap.reason!r}, "
            f"route={type(remap.route).__name__}, plan_present={remap.plan is not None}"
        )

    def to_record(self) -> dict[str, Any]:
        """Retain actual ended request counters without issuing an accepted owner."""
        return {
            **self.report,
            "request_execution_evidence": None
            if self._scope is None or self._scope.evidence is None
            else asdict(self._scope.completed_evidence()),
        }


def _overset_require_complete_cold_remap(
    remap: PreparedUnstructuredConservativeRemap,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[UnstructuredConservativeRemapPlan, CommonRefinementEvidence]:
    """Admit only the real complete route; preserve all refusal evidence."""
    overlap = remap.evidence
    if (
        remap.status is phx.geometry.CommonRefinementStatus.SUCCESS
        and remap.plan is not None
        and remap.plan.require_complete
        and isinstance(overlap, CommonRefinementEvidence)
        and overlap.status is phx.geometry.CommonRefinementStatus.SUCCESS
        and not any((
            overlap.source_gap_count, overlap.target_gap_count,
            overlap.source_double_count, overlap.target_double_count,
        ))
    ):
        return remap.plan, overlap
    raise _OversetColdRemapRefusal(remap, budget)


def _overset_qualification_cold_worker(
    path: Path,
    content_id: str,
    payload: dict[str, Any],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
    """Restore, independently prove ownership and solve inside one cold request."""
    started = perf_counter()
    expected = payload["expected"]
    limits = _overset_archive_limits(int(payload["capacity"]))
    restored = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=limits, budget=budget
    )
    rebuilt = restored["registration"].prepare()
    if not isinstance(rebuilt, phx.meshing.OversetConnectivity):
        raise TypeError(
            f"overset.cold.rebuilt must be OversetConnectivity; got {type(rebuilt).__name__}"
        )
    rebuilt.require_complete()
    names = tuple(sorted(part.name for part in rebuilt.assembly.parts))
    for name in names:
        source = restored["generation_sources"][name]
        specification = restored["generation_specifications"][name]
        if (
            source.binding_id != expected["source_binding_ids"][name]
            or specification.specification_id != expected["specification_ids"][name]
        ):
            raise qualification_failure(
                "_overset_qualification_cold_worker",
                ("Cold checkpoint changed the authored source recipe or hard request."),
                {"name": name},
            )
    fields: dict[str, Any] = {}
    for name in names:
        prepared, _ = prepare_meshing_field_owner(
            restored["field_declarations"][name],
            rebuilt.assembly.part(name).carrier,
            budget=budget,
        )
        if not isinstance(prepared, phx.discretization.FiniteElementDiscretization):
            raise TypeError(
                f"overset.cold.prepared must be FiniteElementDiscretization; got {type(prepared).__name__}"
            )
        fields[name] = prepared
    values = {
        name: jnp.asarray(restored["accepted_data"]["fields"][f"temperature:{name}"])
        for name in names
    }
    identities = _overset_owner_identities(rebuilt, fields, values)
    entries, states = field_entries(rebuilt, fields, values, payload["revision"])
    composition = phx.lifecycle.Composition(
        (*rebuilt.composition_entries(), *entries, *states),
        boundary_id=_ACCEPTED_BOUNDARY,
    )
    owned = (
        identities
        == {key: expected[key] for key in identities}
        and composition.composition_id == expected["composition_id"]
    )
    if not owned:
        raise qualification_failure(
            "_overset_qualification_cold_worker",
            ("Cold restart did not reproduce the accepted owner composition."),
            {"composition_id": composition.composition_id, "identities": identities},
        )
    conservative_packet = payload["conservative"]
    conservative_restored = read_meshing_source_closure(
        Path(conservative_packet["archive_path"]),
        expected_content_id=conservative_packet["content_id"],
        limits=limits,
        budget=budget,
    )
    conservative_epochs = phx.lifecycle.validate_meshing_accepted_epochs(
        conservative_restored
    )
    if len(conservative_epochs) != 2:
        raise ValueError("overset.cold.conservative requires the actual source and accepted motion epochs")
    source_epoch, target_epoch = conservative_epochs
    conservative_carrier = target_epoch.carrier
    conservative_values = target_epoch.fields
    conservative_expected = conservative_packet["expected"]
    bitwise = {
        name: array_tree_fingerprint(value)
        == conservative_expected["field_fingerprints"].get(name)
        for name, value in conservative_values.items()
    }
    measures = _simplex_measures(
        np.asarray(conservative_carrier.mesh.coordinates),
        _mesh_triangles(conservative_carrier.mesh),
    )
    inventory_errors = {
        name: float(np.max(np.abs(
            np.sum(measures[:, None] * np.asarray(conservative_values[name]), axis=0)
            - np.asarray(inventory)
        )))
        for name, inventory in conservative_expected["inventories"].items()
    }
    remap = prepare_overset_conservative_remap(
        phx.meshing.MeshPart(_BACKGROUND, source_epoch.carrier),
        phx.meshing.MeshPart(_BACKGROUND, conservative_carrier),
        policy=_overset_common_refinement_policy(
            int(payload["overset_overlap_pairs"]), budget.maximum_memory_bytes,
        ),
        budget=budget,
    )
    remap_plan, overlap = _overset_require_complete_cold_remap(remap, budget=budget)
    replay_errors = {
        name: float(np.max(np.abs(
            np.asarray(remap_plan.apply(jnp.asarray(source_epoch.fields[name])))
            - np.asarray(value)
        )))
        for name, value in conservative_values.items()
    }
    source_measures = _simplex_measures(
        np.asarray(source_epoch.carrier.mesh.coordinates),
        _mesh_triangles(source_epoch.carrier.mesh),
    )
    conservation_errors = {
        name: float(np.max(np.abs(
            np.sum(source_measures[:, None] * np.asarray(source_epoch.fields[name]), axis=0)
            - np.sum(measures[:, None] * np.asarray(value), axis=0)
        )))
        for name, value in conservative_values.items()
    }
    coverage = (
        overlap.source_gap_count, overlap.target_gap_count,
        overlap.source_double_count, overlap.target_double_count,
    )
    conservative_passed = (
        source_epoch.carrier.result_id == conservative_expected["source_result_id"]
        and conservative_carrier.result_id == conservative_expected["target_result_id"]
        and [epoch.epoch_id for epoch in conservative_epochs]
        == conservative_expected["epoch_ids"]
        and set(bitwise) == set(conservative_expected["field_fingerprints"])
        and all(bitwise.values())
        and all(error <= _CONTENT_TOLERANCE for error in inventory_errors.values())
        and all(error <= _CONTENT_TOLERANCE for error in conservation_errors.values())
        and all(error <= 1.0e-12 for error in replay_errors.values())
        and all(count == 0 for count in coverage)
    )
    if not conservative_passed:
        raise qualification_failure(
            "overset.cold.conservative",
            "Accepted ALE density, material fractions and spatial history must restore bitwise and replay the real complete-overlap transfer with unchanged inventories.",
            {"bitwise": bitwise, "inventory_errors": inventory_errors,
             "transfer_replay_errors": replay_errors, "coverage_counts": coverage},
        )
    restore_seconds = perf_counter() - started
    route = prepare_overset_field_transfer(rebuilt, fields, "u")
    solver = _overset_prepare_solver(
        rebuilt, fields, values, route,
        _overset_restored_analytic(rebuilt, payload["profile"]),
        int(payload["capacity"]),
        # The checkpoint is the accepted initial guess, not a convergence verdict.
        # Native GMRES still evaluates and solves the physical system at the same
        # tolerance/cap; cold startup does not pre-factor an already solved state.
        prepare_part_blocks=False,
    )
    continued, report, _ = _overset_composite_update(
        rebuilt, values, solver, reused=False,
    )
    return _part_carrier(rebuilt.assembly.part(_BACKGROUND)), ColdWorkerReport(
        record={
            "passed": report["passed"]
            and report["physical_error"] <= float(payload["target_error"])
            and conservative_passed,
            "runtime": _runtime_identity(),
            "owner_identities_reproduced": owned,
            "composition_id": composition.composition_id,
            "identities": identities,
            "restored_registration_epoch": rebuilt.epoch,
            "prepared_fe_or_overset_caches_serialized": False,
            "restore_and_ownership_seconds": restore_seconds,
            "conservative": {
                "passed": conservative_passed,
                "fields_bitwise_reproduced": bitwise,
                "independent_inventory_absolute_errors": inventory_errors,
                "source_to_target_inventory_absolute_errors": conservation_errors,
                "transfer_replay_maximum_errors": replay_errors,
                "coverage_counts": list(coverage),
                "source_result_id": source_epoch.carrier.result_id,
                "target_result_id": conservative_carrier.result_id,
                "epoch_ids": [epoch.epoch_id for epoch in conservative_epochs],
            },
            "compiler": {**solver.compiler, "solver": report["solver"]},
            "preparation_stages_seconds": solver.stages,
            "continued": report,
            "continued_values": {
                name: np.asarray(continued[name]).tolist() for name in names
            },
        },
        read_seconds=restore_seconds,
    )


def _overset_conservative_checkpoint(
    accepted: _OversetConservative,
    plan: phx.meshing.NativeMeshingPlan,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain actual generation and accepted motion, never relabel motion generation."""
    from phydrax.lifecycle._meshing_field_records import MeshingGeometryMotionTransition

    def epoch(
        carrier: phx.meshing.CellMeshingResult,
        states: tuple[phx.lifecycle.CompositionEntry, ...],
        number: int,
        motion: phx.meshing.CellMeshTransition | None = None,
    ) -> phx.lifecycle.MeshingAcceptedEpoch:
        reconstruction = phx.discretization.PiecewiseConstantReconstruction()
        owner = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            carrier.mesh, field_name="density", component_names=("density",), budget=budget
        ).prepare(
            numeric_version=carrier.mesh.numeric_version,
            cell_geometry=carrier.geometry,
            budget=budget,
        )
        cells = phx.lifecycle.prepare_meshing_field_index_binding(
            carrier, "mesh/cell_ids"
        )
        coefficients = phx.lifecycle.prepare_meshing_field_index_binding(
            carrier, "field/global_coefficient_ids", field_space=owner.cell_space
        )
        declaration = phx.lifecycle.MeshingFiniteVolumeFieldDeclaration(
            part_name=_BACKGROUND,
            topology_id=carrier.mesh.topology_id,
            geometry_layout_id=carrier.geometry.geometry_layout_id,
            field_space_ids=frozendict({"density": owner.cell_space.field_space_id}),
            value_units=frozendict({"density": (phx.units.ONE,)}),
            maximum_derivative_orders=frozendict({"density": 0}),
            branch_policy=phx.BranchDifferentiationPolicy.SMOOTH,
            trace_policy=phx.discretization.FieldTracePolicy("cell-sided"),
            state_roles=tuple(
                phx.lifecycle.MeshingFieldStateRole(
                    state.entry_id,
                    "density",
                    "coefficients" if index == 0 else "material-history",
                    tuple(state.value.shape),
                    np.dtype(state.value.dtype),
                    number,
                    value_units=None if index == 0 else (phx.units.ONE, phx.units.ONE),
                    index_binding=coefficients if index == 0 else cells,
                )
                for index, state in enumerate(states)
            ),
            history_policy="material-history",
            numeric_version=owner.numeric_version,
            finite_volume_field_name="density",
            finite_volume_component_names=("density",),
            finite_volume_coordinate_policy="accepted-map",
            reconstruction_policy=reconstruction,
            reconstruction_policy_id=reconstruction.plan_id,
        )
        return phx.lifecycle.MeshingAcceptedEpoch(
            carrier, (declaration,),
            {state.entry_id: state.value for state in states},
            transition=None if motion is None else MeshingGeometryMotionTransition(
                accepted.source, carrier, motion, budget=budget
            ),
        )

    states = accepted.target_states
    initial = epoch(accepted.source, accepted.source_states, 0)
    final = epoch(accepted.motion.target, states, 1, accepted.motion)
    certificate = accepted.motion.target.certification
    if certificate is None:
        raise ValueError("overset.conservative.checkpoint requires actual target certification")
    records = {
        "certification_inputs": certificate.request,
        "report": certificate,
        "associations": accepted.motion.target.associations,
        "generation_source": plan.source,
        "generation_specification": plan.specification,
        "generation_options": plan.options,
        "accepted_target": accepted.motion.target,
        "accepted_data": {"accepted_epochs": (initial, final)},
    }
    measures = _simplex_measures(
        np.asarray(accepted.motion.target.mesh.coordinates),
        _mesh_triangles(accepted.motion.target.mesh),
    )
    return records, {
        "source_result_id": accepted.source.result_id,
        "target_result_id": accepted.motion.target.result_id,
        "motion_transition_id": accepted.motion.transition_id,
        "epoch_ids": [initial.epoch_id, final.epoch_id],
        "field_fingerprints": {
            state.entry_id: array_tree_fingerprint(state.value) for state in states
        },
        "inventories": {
            state.entry_id: np.sum(
                measures[:, None] * np.asarray(state.value), axis=0
            ).tolist()
            for state in states
        },
    }


def _overset_qualification_restart(
    accepted: _OversetAccepted,
    generation_plans: dict[str, phx.meshing.NativeMeshingPlan],
    conservative: _OversetConservative,
    profile: _OversetProfile,
    args: QualificationControls,
    record: dict[str, Any],
    deadline: Deadline,
) -> None:
    """Publish the accepted epoch, restart it cold, and compare its continuation."""

    _ledger = StageLedger(record, deadline=deadline)
    connectivity = accepted.connectivity
    limits = _overset_archive_limits(args.capacity)
    policy = asdict(limits)
    policy["allowed_dtype_kinds"] = sorted(limits.allowed_dtype_kinds)
    names = tuple(sorted(accepted.values))
    states = {
        name: accepted.composition.entry(f"temperature:{name}") for name in names
    }
    background_part = connectivity.assembly.part(_BACKGROUND)
    background = _part_carrier(background_part)
    background_certificate = _part_certification(background_part)
    declarations = field_declarations(
        connectivity,
        accepted.plans,
        accepted.fields,
        {name: states[name].value for name in names},
    )
    records = {
        "certification_inputs": background_certificate.request,
        "report": background_certificate,
        "associations": background.associations,
        "registration": connectivity.registration(),
        "primary_generation_part": _BACKGROUND,
        "generation_sources": {name: generation_plans[name].source for name in names},
        "generation_specifications": {
            name: generation_plans[name].specification for name in names
        },
        "generation_options": {name: generation_plans[name].options for name in names},
        "field_declarations": declarations,
        "accepted_data": {
            "fields": {states[name].entry_id: states[name].value for name in names}
        },
    }
    expected = {
        **_overset_owner_identities(
            connectivity,
            accepted.fields,
            {name: states[name].value for name in names},
        ),
        "composition_id": accepted.composition.composition_id,
        "source_binding_ids": {
            name: _planar_source(generation_plans[name]).binding_id for name in names
        },
        "specification_ids": {
            name: generation_plans[name].specification.specification_id
            for name in names
        },
    }
    evidence: dict[str, Any] = {
        "archive_limits": policy,
        "source_authority_stripped": False,
        "prepared_fe_or_overset_caches_serialized": False,
        "physical_state_bindings": [
            {
                "part_name": name,
                "entry_id": states[name].entry_id,
                "structure_id": states[name].structure_id,
                "revision_id": states[name].revision_id,
                "semantics_id": states[name].semantics_id,
                "owner_id": states[name].owner_id,
            }
            for name in names
        ],
        "status": "unreached",
    }
    record["checkpoint"] = evidence
    with (
        TemporaryDirectory(prefix="phydrax-native-overset-source-") as directory,
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.overset",
            deadline=deadline,
            ledger=StageLedger(record, deadline=deadline, cold_phase=_COLD_PHASE),
        ) as cold_worker,
    ):
        def publish_conservative(budget: CoordinateEnclosureBudget) -> tuple[Any, dict[str, Any]]:
            from phydrax.lifecycle._meshing_sources import (
                meshing_source_owner_reconstruction_scope,
            )

            try:
                with meshing_source_owner_reconstruction_scope(
                    budget=budget, archive_limits=limits
                ):
                    conservative_records, expected_conservative = _ledger.run(
                        "conservative_checkpoint:actual_field_declarations",
                        lambda: _overset_conservative_checkpoint(
                            conservative, generation_plans[_BACKGROUND], budget=budget
                        ),
                    )
                    publication = _ledger.run(
                        "conservative_checkpoint:source_authoritative_write",
                        lambda: write_meshing_source_closure(
                            Path(directory) / "conservative.checkpoint.zip",
                            conservative_records, limits=limits,
                        ),
                    )
            except (RuntimeError, ValueError, TypeError, OSError) as failure:
                raw_refusal = getattr(failure, "batch_refusal_evidence", None)
                evidence["conservative_publication_failure"] = {
                    "phase": record["active_phase"],
                    "exception_type": type(failure).__name__,
                    "callback_coordinate_work_units": budget.work_units,
                    "raw_batch_refusal_evidence": None if raw_refusal is None
                    else np.asarray(raw_refusal).tolist(),
                    "resource_axis_not_inferred_from_status": True,
                }
                raise
            return publication, expected_conservative

        receipt = _ledger.run(
            "native_checkpoint_publication",
            lambda: write_meshing_source_closure(
                Path(directory) / "overset.checkpoint.zip", records, limits=limits
            ),
        )
        evidence["content_id"] = receipt.content_id
        evidence["container_bytes"] = receipt.path.stat().st_size
        try:
            conservative_publication = _ledger.run(
                "conservative_checkpoint_publication",
                lambda: execute_native_consumer(
                    conservative.motion.target,
                    generation_plans[_BACKGROUND].specification.limits,
                    publish_conservative,
                    failure_factory=qualification_failure,
                    prepared_operands=conservative.prepared_operands,
                ),
            )
        except (RuntimeError, ValueError, TypeError, OSError) as failure:
            # Scope exit attaches the actual ended native counters. Capture
            # after that boundary, not only inside the still-live callback.
            work = getattr(failure, "work_counters", None)
            memory = getattr(failure, "memory_counters", None)
            raw_refusal = getattr(failure, "batch_refusal_evidence", None)
            diagnostic = evidence.setdefault("conservative_publication_failure", {})
            diagnostic.update({
                "phase": record["active_phase"],
                "exception_type": type(failure).__name__,
                "native_work_counters": None if work is None else dict(work.items()),
                "native_memory_counters": None if memory is None else dict(memory.items()),
                "raw_batch_refusal_evidence": None if raw_refusal is None
                else np.asarray(raw_refusal).tolist(),
                "batch_requested_upper": getattr(failure, "batch_requested_upper", None),
                "traceback": "".join(format_exception(failure)),
                "resource_axis_not_inferred_from_status": True,
            })
            raise
        conservative_receipt, conservative_expected = conservative_publication.value
        ended = conservative_publication.result.execution_evidence
        if ended is None:
            raise ValueError("overset.conservative.checkpoint requires the real ended publication-consumer receipt")
        evidence["conservative_publication_execution"] = {
            "owner_id": ended.owner_id,
            "work_units": int(np.asarray(ended.total_work_units)),
            "geometry_queries": int(np.asarray(ended.total_geometry_queries)),
            "receipt_fingerprint": array_tree_fingerprint(ended),
            "original_request_allowance_renewed": False,
            "archive_snapshots_preceding_native_motion_receipt": True,
        }
        evidence["conservative_content_id"] = conservative_receipt.content_id
        evidence["conservative_container_bytes"] = conservative_receipt.path.stat().st_size
        try:
            cold = cold_worker.execute(
                receipt.path,
                receipt.content_id,
                {
                    "expected": expected,
                    "revision": accepted.revision,
                    "capacity": args.capacity,
                    "target_error": args.target_error,
                    "profile": profile.record(),
                    "overset_overlap_pairs": args.overset_overlap_pairs,
                    "maximum_scratch_bytes": generation_plans[
                        _BACKGROUND
                    ].specification.limits.maximum_scratch_bytes,
                    "conservative": {
                        "archive_path": str(conservative_receipt.path),
                        "content_id": conservative_receipt.content_id,
                        "expected": conservative_expected,
                    },
                },
            )
        except ColdWorkerTimeout as failure:
            raise TimeoutError(
                f"overset.cold_restart must finish within the remaining wall budget; got {type(failure).__name__}"
            ) from failure
    if "conservative_remap_refusal" in cold:
        evidence["status"] = "published"
        record["restart"] = {
            "passed": False,
            "process_id": cold["process_id"],
            "conservative_remap_refusal": cold["conservative_remap_refusal"],
        }
        raise qualification_failure(
            "overset.cold.conservative",
            "The cold route must produce a real complete-overlap remap.",
            cold["conservative_remap_refusal"],
        )
    evidence["status"] = "passed"
    uninterrupted, reference, _ = _ledger.run(
        "uninterrupted_composite_schwarz_poisson",
        lambda: _overset_composite_update(
            connectivity, accepted.values, accepted.solver, reused=True,
        ),
    )
    continuation = max(
        float(
            np.max(
                np.abs(
                    np.asarray(cold["continued_values"][name], dtype=np.float64)
                    - np.asarray(uninterrupted[name])
                )
            )
        )
        for name in names
    )
    parent_runtime = record["runtime"]["build"]
    passed = (
        cold.get("passed") is True
        and cold.get("owner_identities_reproduced") is True
        and (cold.get("conservative") or {}).get("passed") is True
        and (cold.get("cold_request") or {}).get("reader_proof_solve_shared_budget") is True
        and cold.get("composition_id") == accepted.composition.composition_id
        and cold.get("runtime", {}).get("build") == parent_runtime
        and reference["passed"] is True
        and math.isfinite(continuation)
        and continuation <= 1.0e-12
    )
    record["restart"] = {
        "checkpoint_content_id": receipt.content_id,
        "cold_process_id": cold.get("process_id"),
        "cold_runtime_matches_parent_build": cold.get("runtime", {}).get("build")
        == parent_runtime,
        "owner_identities_reproduced": cold.get("owner_identities_reproduced"),
        "composition_id": cold.get("composition_id"),
        "restored_registration_epoch": cold.get("restored_registration_epoch"),
        "restore_and_ownership_seconds": cold.get("restore_and_ownership_seconds"),
        "conservative": cold.get("conservative"),
        "cold_request": cold.get("cold_request"),
        "cold_request_result": cold.get("cold_request_result"),
        "continued_composite_poisson": cold.get("continued"),
        "uninterrupted_composite_poisson": reference,
        "continuation_maximum_error": continuation if np.isfinite(continuation) else None,
        "passed": passed,
    }
    if not passed:
        raise qualification_failure(
            "_overset_qualification_restart",
            (
                "Cold restart ownership or continued solve disagreed with the uninterrupted update."
            ),
            {"restart": {key: value for key, value in record["restart"].items()}},
        )


# ---------------------------------------------------------- obligations


def _overset_workflow_obligations(
    record: dict[str, Any], args: QualificationControls, elapsed: float
) -> dict[str, bool]:
    """Derive each mandatory moving-overset obligation from recorded evidence."""
    epochs = [item for item in record.get("epochs", []) if isinstance(item, dict)]
    motions = [item for item in epochs if item.get("epoch", 0) > 0]
    conservative = record.get("conservative_profile") or {}
    overreach = record.get("refused_overreach_motion") or {}
    restart = record.get("restart") or {}
    target = args.target_error

    def oracle_exact(item: dict[str, Any]) -> bool:
        oracle = item.get("independent_registration_oracle") or {}
        registration = item.get("registration") or {}
        return oracle.get("exact") is True and registration.get("complete") is True

    def pde(item: dict[str, Any]) -> bool:
        report = item.get("poisson") or {}
        error = report.get("physical_error")
        return (
            report.get("passed") is True
            and isinstance(error, float)
            and math.isfinite(error)
            and error <= target
        )

    cold = restart.get("continued_composite_poisson") or {}
    cold_error = cold.get("physical_error")
    return {
        "native_hole_cutting_and_donor_search": bool(epochs)
        and all(oracle_exact(item) for item in epochs),
        "motion_refresh": len(motions) >= 2
        and all(
            (item.get("motion_refresh") or {}).get("refreshed") is True
            for item in motions
        ),
        "interpolative_transfer": len(motions) >= 2
        and all(
            (item.get("transfer") or {}).get("oracles_passed") is True
            for item in motions
        ),
        "separately_certified_conservative_transfer": conservative.get("certified")
        is True,
        "accepted_pde_update": bool(epochs) and all(pde(item) for item in epochs),
        "atomic_refusals": len(motions) >= 2
        and all(
            (item.get("rollback") or {}).get("atomic") is True for item in motions
        )
        and conservative.get("refusals_atomic") is True
        and overreach.get("atomic") is True,
        "cold_restart_ownership_continuation": restart.get("passed") is True
        and (restart.get("conservative") or {}).get("passed") is True
        and (restart.get("cold_request") or {}).get("reader_proof_solve_shared_budget") is True
        and isinstance(cold_error, float)
        and cold_error <= target,
        "wall_budget": elapsed <= args.timeout,
    }


# ------------------------------------------------------------ execution


def _overset_qualification_execute(
    args: QualificationControls,
    profile: _OversetProfile,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    started: float,
    deadline: Deadline,
) -> None:
    """Drive every native caller handoff without substituting certified raw grids."""

    _ledger = StageLedger(record, deadline=deadline)
    background, background_plan = _overset_qualification_part(
        _BACKGROUND, None, args, record, phases, deadline
    )
    bodies = {
        body.name: _overset_qualification_part(
            body.name, body, args, record, phases, deadline
        )
        for body in profile.bodies
    }
    generation_plans = {
        _BACKGROUND: background_plan,
        **{name: plan for name, (_, plan) in bodies.items()},
    }
    connectivity = _ledger.run(
        "native_wall_cutting_donor_search",
        lambda: _overset_qualification_connectivity(background, bodies, profile, args),
    )
    record["connectivity"] = _overset_qualification_registration_evidence(connectivity)
    connectivity.require_complete()
    initial: dict[str, Any] = {
        "epoch": 0,
        "name": "initial",
        # The complete initial registration evidence is record["connectivity"].
        "registration": {
            "connectivity_id": connectivity.connectivity_id,
            "complete": connectivity.complete,
            "orphan_count": connectivity.orphan_count,
            "protected_wall_conflict_count": connectivity.conflict_count,
        },
    }
    record["epochs"].append(initial)
    oracle = _ledger.run(
        "initial_independent_registration_oracle",
        lambda: _overset_registration_oracle(connectivity),
    )
    initial["independent_registration_oracle"] = oracle
    _overset_require_oracle(oracle, 0)
    conservative = _overset_qualification_conservative(
        connectivity, background_plan, profile, args, record, phases, deadline
    )
    physical_plans, discretizations = _ledger.run(
        "initial_actual_fe_preparation",
        lambda: _overset_qualification_fields(connectivity),
    )
    record["generation_recipe_bindings"] = {
        name: {
            "source_binding_id": _planar_source(plan).binding_id,
            "request_id": plan.specification.specification_id,
            "plan_id": plan.plan_id,
        }
        for name, plan in generation_plans.items()
    }
    # The initial iterate is affine, not the manufactured solution: only physical
    # walls and the exterior carry manufactured Dirichlet data.
    seeds = {
        name: 5.0 + jnp.sum(field.dof_maps[0].dof_coordinates, axis=1)
        for name, field in discretizations.items()
    }
    accepted = _overset_accept_epoch(
        connectivity,
        physical_plans,
        discretizations,
        seeds,
        "initial",
        initial,
        profile,
        record,
        deadline,
    )
    record["physical_error"] = initial["poisson"]["physical_error"]
    if record["physical_error"] <= args.target_error:
        record["time_to_target_error_seconds"] = perf_counter() - started
    record["initial_composition_id"] = accepted.composition.composition_id
    record["motion_sources"] = []
    for step in range(args.overset_motion_steps):
        if perf_counter() - started > args.timeout:
            raise TimeoutError(
                f"overset.elapsed_seconds must not exceed {args.timeout!r} before motion; got {perf_counter() - started!r}"
            )
        successors: dict[str, phx.meshing.MeshPart] = {}
        for body in profile.bodies:
            successor, moved_plan = _ledger.run(
                f"motion-{step}:{body.name}:source_successor",
                lambda body=body: rigid_planar_successor(
                    accepted.connectivity.assembly.part(body.name),
                    body.offset(step),
                    motion_id=f"qualification-accepted-motion-{step}-{body.name}",
                    plan=generation_plans[body.name],
                    record_phase=phases.append,
                ),
            )
            generation_plans[body.name] = moved_plan
            successors[body.name] = successor
            record["motion_sources"].append(
                {
                    "step": step,
                    "body": body.name,
                    "source_binding_id": _planar_source(moved_plan).binding_id,
                    "source_revision": moved_plan.source.source_revision,
                    "request_id": moved_plan.specification.specification_id,
                    "limits_id": moved_plan.specification.limits.limits_id,
                    "prepared_id": moved_plan.plan_id,
                    "result_id": _part_carrier(successor).result_id,
                }
            )
        accepted = _overset_qualification_motion(
            accepted, successors, step, profile, record, deadline
        )
    record["physical_error"] = max(
        epoch["poisson"]["physical_error"] for epoch in record["epochs"]
    )
    record["final_connectivity_id"] = accepted.connectivity.connectivity_id
    record["final_composition_id"] = accepted.composition.composition_id
    record["retained_registration_bytes"] = model_array_bytes(accepted.connectivity)
    _overset_refused_overreach(
        accepted, generation_plans, profile, record, phases, deadline
    )
    _overset_qualification_restart(
        accepted, generation_plans, conservative, profile, args, record, deadline
    )
    record["physical_error"] = max(
        record["physical_error"],
        record["restart"]["continued_composite_poisson"]["physical_error"],
    )
    record["target_error_reached"] = record["physical_error"] <= args.target_error
    if record["target_error_reached"] and perf_counter() - started <= args.timeout:
        # Every epoch oracle, transfer, refusal, update and cold continuation passed
        # its independent check before reaching here.
        record["status"] = "passed"
        record["case_contract_completion"] = "complete"
        record["independent_scientific_certification"] = "passed"
    else:
        record["failure"] = (
            "The manufactured composite error target was not reached within the wall budget."
        )


def _overset_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Native sources → hole/donor refresh → transfers/rebind → Schwarz PDE → cold restart."""

    if (
        any(
            value < 1
            for value in (
                args.resolution,
                args.capacity,
                args.repeats,
                args.overset_wall_pairs,
                args.overset_donor_pairs,
                args.overset_overlap_pairs,
            )
        )
        or args.overset_motion_steps < 2
    ):
        raise ValueError(
            f"overset.controls must have positive capacities, repeats, and resolution with at least two motion steps; got resolution={args.resolution!r}, capacity={args.capacity!r}, repeats={args.repeats!r}, wall_pairs={args.overset_wall_pairs!r}, donor_pairs={args.overset_donor_pairs!r}, overlap_pairs={args.overset_overlap_pairs!r}, motion_steps={args.overset_motion_steps!r}"
        )
    if (
        not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"overset.controls must have finite positive timeout and error target; got timeout={args.timeout!r}, target_error={args.target_error!r}"
        )
    profile = _overset_profile(args)
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record: dict[str, Any] = {
        "scenario": "native-overset-lifecycle",
        "profile": profile.name,
        "authored_profile": profile.record(),
        "status": "failed",
        "runtime": _runtime_identity(),
        "engine": "native",
        "expected_admission": "positive",
        "physics_preparation_owner": (
            f"{prepare_poisson_update.__module__}.{prepare_poisson_update.__qualname__}"
        ),
        "requirements": _overset_qualification_requirements(),
        "control_capacities": {
            "overset_profile": profile.name,
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "motion_steps": args.overset_motion_steps,
            "fringe_layers": profile.fringe_layers,
            "maximum_wall_candidate_pairs": args.overset_wall_pairs,
            "maximum_donor_candidate_pairs": args.overset_donor_pairs,
            "maximum_overlap_candidate_pairs": args.overset_overlap_pairs,
            "maximum_overlap_accepted_pairs": args.overset_overlap_pairs,
            "maximum_overlap_memory_bytes": 4096 * args.capacity,
            "maximum_composite_krylov_iterations": _SCHWARZ_MAXIMUM_ITERATIONS,
        },
        "targets": {
            _ERROR_QUANTITY: args.target_error,
            "wall_seconds": args.timeout,
        },
        "parts": {},
        "stages_seconds": {},
        "epochs": [],
        "attempts": [],
        "physical_error": None,
        "time_to_target_error_seconds": None,
        "positive_prerequisite": None,
        "workflow_obligations": None,
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "full_corpus_completion": "unassessed",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "optional_external_baseline": "not-requested",
        "active_phase": "source_preparation",
    }
    phases: list[NativeMeshingPhaseMeasurement] = []
    try:
        _overset_qualification_execute(args, profile, record, phases, started, deadline)
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except OversetConnectivityError as failure:
        record_failure(record, failure, record["active_phase"])
        record["failed_registration"] = _overset_qualification_registration_evidence(
            failure.connectivity
        )
    except (RuntimeError, ValueError, TypeError, OSError) as failure:
        record_failure(record, failure, record["active_phase"])
    finally:
        elapsed = perf_counter() - started
        record["end_to_end_seconds"] = elapsed
        record["native_phases"] = _phase_recordings(phases)
        record["peak_process_resident_bytes"] = peak_resident_bytes()
        record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
        record["device_memory"] = [
            {
                "device_id": device.id,
                "platform": device.platform,
                "status": "unavailable" if statistics is None else "backend-reported",
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "memory_statistics": statistics,
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
        if elapsed > args.timeout:
            record["status"] = "timeout"
            record["time_to_target_error_seconds"] = None
            record["wall_budget_evidence"] = {
                "requested_wall_seconds": args.timeout,
                "achieved_wall_seconds": elapsed,
                "phase": record["active_phase"],
                "status": "timeout",
            }
        obligations = _overset_workflow_obligations(record, args, elapsed)
        record["workflow_obligations"] = obligations
        record["mandatory_workflow_completion"] = (
            "complete"
            if record["status"] == "passed" and all(obligations.values())
            else "incomplete"
        )
        if record["status"] != "passed":
            record["failure_phase"] = record["active_phase"]
            record["case_contract_completion"] = "incomplete"
            record["independent_scientific_certification"] = "unassessed"
            record["attempts"].append(
                {
                    "phase": record["active_phase"],
                    "status": record["status"],
                    "failure": record.get("failure"),
                }
            )
            record["positive_prerequisite"] = {
                "phase": record["active_phase"],
                "detail": record.get("failure"),
                "source_authority_stripped": False,
                "safe_refusal_is_positive": False,
                "owner": (
                    "native-source-archive-owner"
                    if any(
                        stage in record["active_phase"]
                        for stage in ("checkpoint", "restart", "fresh_process")
                    )
                    else "native-overset-owner"
                    if any(
                        stage in record["active_phase"]
                        for stage in ("motion", "oracle", "conservative", "overreach")
                    )
                    else "native-source-owner"
                ),
            }
        del record["active_phase"]
    return record


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Bootstrap original controls before the one budgeted reader/proof/solve."""
    began = perf_counter()
    execution = read_meshing_source_execution_controls(
        packet.archive_path,
        expected_content_id=packet.content_id,
        limits=_overset_archive_limits(int(packet.controls["capacity"])),
    )
    conservative_packet = packet.controls["conservative"]
    conservative_execution = read_meshing_source_execution_controls(
        Path(conservative_packet["archive_path"]),
        expected_content_id=conservative_packet["content_id"],
        limits=_overset_archive_limits(int(packet.controls["capacity"])),
    )
    requested_scratch = packet.controls["maximum_scratch_bytes"]
    if (
        type(requested_scratch) is not int
        or requested_scratch <= 0
        or requested_scratch != execution.limits.maximum_scratch_bytes
        or requested_scratch != conservative_execution.limits.maximum_scratch_bytes
    ):
        raise qualification_failure(
            "overset.cold.original_controls",
            "Cold remap scratch must equal both archived original request limits.",
            {
                "maximum_scratch_bytes": requested_scratch,
                "main_scratch_bytes": execution.limits.maximum_scratch_bytes,
                "conservative_scratch_bytes": conservative_execution.limits.maximum_scratch_bytes,
            },
        )
    expected = packet.controls["expected"]
    if any(
        controls.limits.limits_id != execution.limits.limits_id
        or controls.specification_id != expected["specification_ids"][_BACKGROUND]
        or controls.source_binding_id != expected["source_binding_ids"][_BACKGROUND]
        for controls in (execution, conservative_execution)
    ):
        raise qualification_failure(
            "overset.cold.original_controls",
            "Both accepted routes must retain the same original background hard request and source binding.",
            {"main_specification_id": execution.specification_id,
             "conservative_specification_id": conservative_execution.specification_id},
        )
    control_read_seconds = perf_counter() - began
    try:
        continued = execute_native_request(
            execution.execution_evidence,
            execution.limits,
            lambda budget: _overset_qualification_cold_worker(
                packet.archive_path, packet.content_id, packet.controls, budget=budget
            ),
            target=lambda value: value[0],
            failure_factory=qualification_failure,
            workflow_remaining_seconds=packet.deadline.remaining(),
        )
    except _OversetColdRemapRefusal as failure:
        # Report only after the real request has unwound. No accepted target or
        # execution receipt is issued, and the parent records scientific failure.
        return ColdWorkerReport(
            record={"passed": False, "conservative_remap_refusal": failure.to_record()},
            read_seconds=control_read_seconds,
        )
    _, report = continued.value
    receipt = continued.result.execution_evidence
    if receipt is None:
        raise qualification_failure(
            "overset.cold.execution",
            "The cold reader, proof and solve must retain one ended execution receipt.",
            {"result": continued.result},
        )
    return ColdWorkerReport(
        record={
            **report.record,
            "cold_request_result": _result_identity(continued.result),
            "cold_request": {
                "history_owner_id": execution.execution_evidence.owner_id,
                "owner_id": receipt.owner_id,
                "work_units": int(np.asarray(receipt.total_work_units)),
                "geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
                "receipt_fingerprint": array_tree_fingerprint(receipt),
                "original_limits_id": execution.limits.limits_id,
                "conservative_history_owner_id": conservative_execution.execution_evidence.owner_id,
                "reader_proof_solve_shared_budget": True,
            },
        },
        read_seconds=control_read_seconds + report.read_seconds,
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _overset_qualification_lifecycle(args)

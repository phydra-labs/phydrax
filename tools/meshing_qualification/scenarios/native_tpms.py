"""Native TPMS lattice qualification (plan section 14.8).

Profiles, each against an independent oracle:

* ``gyroid-sheet-cell``: quotient topology of the sheet ``{|F| <= tau}``; exact
  periodic face pairing of the fill; the certified volume-fraction enclosure
  against an independent quadrature (exact in ``x``, converged Gauss-Legendre
  in ``(y, z)``); homogenized conductivity with the cubic point group proved
  from the table and the cubic-symmetry defect measured;
* ``schwarz-p-network``: volume fraction 1/2 proved by the half-lattice shift
  ``F(u + 1/2) = -F(u)`` and enclosed by the cover; network and complement
  quotient component counts;
* ``schwarz-p-singular``: a level inside a critical-value enclosure is refused
  with the certified separation evidence;
* ``diamond-cylinder-specimen``: certified creases and caps, a watertight
  oriented solid exported as 3MF, linear-elastic compression with a discrete
  kinematic upper bound and an affine patch test, and volume/porosity against
  an exact-per-line oracle within the certified fidelity tube;
* ``periodic-cold-restore``: an independent process restores the Schwarz P
  network cell, recomputes its identities, certification, quotient components
  and measure, and re-solves the homogenized conductivity bit for bit.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.solid_mechanics import PeriodicCellProblem
from phydrax.discretization import PeriodicCell
from phydrax.discretization._periodic_exact import apply_periodic
from phydrax.discretization.fem import PeriodicSimplexOperatorPattern
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
    PeriodicImplicitSurface,
)
from phydrax.geometry.surface import SurfaceExportPolicy
from phydrax.linalg import SparseFactorizationPolicy
from phydrax.meshing import (
    construct_lattice_specimen_surface,
    export_lattice_solid,
    LatticeSpecimenVolume,
    mesh_lattice_specimen,
    mesh_periodic_trigonometric_region,
    PeriodicImplicitVolume,
)
from phydrax.meshing._sizing import SizeControlStrength
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.units import METER

from .._runtime import require_meshcore, runtime_identity


_G = phx.geometry
_ROOT_TOLERANCE = 1.0e-8
# Gyroid sheet half width: the band +-tau must be separated from every
# critical value (the gyroid's nearest nonzero critical values are +-1).
_GYROID_TAU = 0.3
_GYROID_DIAMETER = 0.1
_SCHWARZ_P_DIAMETER = 0.2
# Diamond (Schwarz D) network in a cylinder: the specimen of the unit suite.
_CYLINDER_CENTER = (0.13, 0.21, 0.07)
_CYLINDER_AXIS = (0.0, 0.0, 0.4)
_CYLINDER_RADIUS = 0.25
_SPECIMEN_DIAMETER = 0.125
_SPECIMEN_ROOT_TOLERANCE = 1.0e-9
# Isotropic solid for the compression: Young's modulus 1, Poisson ratio 0.3.
_YOUNG, _POISSON = 1.0, 0.3
_COMPRESSION_STRAIN = 1.0e-3
# Affine patch-test and solver residual acceptance (relative, binary64 solve).
_PATCH_TOLERANCE = 1.0e-9
_RESIDUAL_TOLERANCE = 1.0e-9
# Declared budget of the compression's native sparse Cholesky (the free block
# of the Dirichlet-eliminated coupled vector stiffness, nested dissection).
_ELASTIC_FACTORIZATION = SparseFactorizationPolicy(
    "cholesky",
    ordering="nested-dissection",
    max_factor_nnz=2_000_000,
    max_factor_bytes=256 * 1024 * 1024,
)
# Cubic-symmetry defect of the discrete homogenized tensor: a measured
# discretization consequence (the exact tensor is proved isotropic); the
# acceptance is the unit suite's, not relaxed.
_CUBIC_DEFECT_ACCEPTANCE = 1.0e-2
# Declared symbolic-analysis budget of the reverse Cuthill-McKee comparison
# (no numeric factorization is performed with it).
_RCM_COMPARISON_FACTOR_NNZ = 16_000_000


def _policy(diameter: float, root_tolerance: float = _ROOT_TOLERANCE) -> Any:
    return AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=diameter, root_tolerance=root_tolerance
    )


def _timed[T](timings: dict[str, float], name: str, operation: Callable[[], T]) -> T:
    started = perf_counter()
    value = operation()
    timings[name] = perf_counter() - started
    return value


# ---------------------------------------------------------------------------
# Independent oracles and checks


def _signed_permutations() -> list[tuple[tuple[int, ...], ...]]:
    matrices = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrices.append(
                tuple(
                    tuple(
                        signs[row] if column == order[row] else 0 for column in range(3)
                    )
                    for row in range(3)
                )
            )
    return matrices


def _table_point_group(
    table: Any, accepted_signs: tuple[int, ...]
) -> list[tuple[tuple[int, ...], ...]]:
    """Linear parts of the quarter-lattice maps ``u -> A u + b`` proved on the table.

    ``symmetry_sign`` proves ``F(A u + b) = sigma F(u)`` exactly from the
    rational table; the map preserves the region for each accepted ``sigma``.
    """
    quarters = [Fraction(k, 4) for k in range(4)]
    group = []
    for matrix in _signed_permutations():
        for shift in itertools.product(quarters, repeat=3):
            if table.symmetry_sign(matrix, shift) in accepted_signs:
                group.append(matrix)
                break
    return group


def _invariant_symmetric_dimension(
    matrices: list[tuple[tuple[int, ...], ...]],
) -> int:
    """Dimension of symmetric 3x3 tensors ``S`` with ``A S A^T = S`` for every ``A``."""
    basis = []
    for i in range(3):
        for j in range(i, 3):
            unit = np.zeros((3, 3))
            unit[i, j] = unit[j, i] = 1.0
            basis.append(unit)
    rows = []
    for matrix in matrices:
        a = np.asarray(matrix, dtype=np.float64)
        rows.append(
            np.stack([(a @ unit @ a.T - unit).reshape(-1) for unit in basis], axis=1)
        )
    stacked = np.concatenate(rows)
    # Entries are small integers: the rank is exact in binary64.
    return len(basis) - int(np.linalg.matrix_rank(stacked))


def _gyroid_band_fraction(tau: float, panels: int, order: int = 8) -> float:
    """Volume fraction of ``{|F| <= tau}`` for the gyroid, exact along ``x``.

    For fixed ``(y, z)``, ``F = R sin(X + phi) + C`` with ``R^2 = cos^2 Y +
    sin^2 Z`` and ``C = sin Y cos Z``; the fraction of a period with
    ``a <= sin <= b`` is ``(asin b - asin a) / pi`` after clipping to [-1, 1].
    The ``(y, z)`` integral uses composite Gauss-Legendre on ``panels^2``
    squares of the unit period.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    starts = np.arange(panels) / panels
    points = (starts[:, None] + 0.5 * (nodes[None, :] + 1.0) / panels).reshape(-1)
    point_weights = np.tile(0.5 * weights / panels, panels)
    angle_y, angle_z = np.meshgrid(
        2.0 * math.pi * points, 2.0 * math.pi * points, indexing="ij"
    )
    weight = np.outer(point_weights, point_weights)
    radius = np.hypot(np.cos(angle_y), np.sin(angle_z))
    offset = np.sin(angle_y) * np.cos(angle_z)
    upper = np.clip((tau - offset) / radius, -1.0, 1.0)
    lower = np.clip((-tau - offset) / radius, -1.0, 1.0)
    fraction = (np.arcsin(upper) - np.arcsin(lower)) / math.pi
    return float(np.sum(weight * fraction))


def _cell_face_pairs(volume: PeriodicImplicitVolume, axis: int, /) -> int:
    """Count boundary cap triangles; require each upper face to be the lower's translate."""
    mesh = volume.mesh
    topology = mesh.periodic_topology
    if topology is None:
        raise RuntimeError("The periodic fill lost its periodic topology.")
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    lower, upper = volume.reduced_domain[:, axis]
    in_plane = [index for index in range(3) if index != axis]
    counts: dict[frozenset[int], int] = {}
    tetrahedra = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    for cell in tetrahedra.tolist():
        for omitted in range(4):
            face = frozenset(cell[index] for index in range(4) if index != omitted)
            counts[face] = counts.get(face, 0) + 1
    sides: dict[float, dict[frozenset[tuple[int, ...]], tuple[bytes, ...]]] = {
        float(lower): {},
        float(upper): {},
    }
    step = np.zeros(3, dtype=np.int64)
    step[axis] = 1
    # Unit cubic cell: reduced and Cartesian coordinates coincide.
    for face, count in counts.items():
        rows = sorted(face)
        plane = coordinates[rows, axis]
        if count != 1 or not np.all(plane == plane[0]) or float(plane[0]) not in sides:
            continue
        moved = step if plane[0] == lower else np.zeros(3, dtype=np.int64)
        key = frozenset(
            (int(representatives[row]), *(shifts[row] + moved).tolist()) for row in rows
        )
        sides[float(plane[0])][key] = tuple(
            sorted(coordinates[row, in_plane].tobytes() for row in rows)
        )
    if not sides[float(lower)] or sides[float(lower)] != sides[float(upper)]:
        raise RuntimeError(f"Cell faces normal to axis {axis} are not exact translates.")
    return 2 * len(sides[float(lower)])


def _exact_lift(surface: PeriodicImplicitSurface) -> int:
    """Translates on upper cell faces are their representatives moved exactly."""
    mesh = surface.mesh
    if mesh is None or mesh.periodic_topology is None:
        raise RuntimeError("The quotient discovery is unresolved.")
    topology = mesh.periodic_topology
    coordinates = np.asarray(mesh.coordinates)
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    lifted = apply_periodic(surface.cell, coordinates[representatives], shifts)
    if not np.array_equal(coordinates, lifted):
        raise RuntimeError("A lifted contour vertex is not its exact lattice translate.")
    return int(np.count_nonzero(np.any(shifts != 0, axis=1)))


def _topology_record(surface: PeriodicImplicitSurface) -> dict[str, Any]:
    topology = surface.topology
    if topology is None:
        raise RuntimeError("Quotient discovery left unresolved boxes.")
    return {
        "vertices": topology.vertex_count,
        "edges": topology.edge_count,
        "faces": topology.face_count,
        "euler_characteristic": topology.euler_characteristic,
        "component_euler_characteristics": list(topology.component_euler_characteristics),
        "component_genera": list(topology.component_genera),
        "closed": topology.closed,
        "orientable": topology.orientable,
    }


def _discover(source: Any, diameter: float, source_id: str) -> PeriodicImplicitSurface:
    surface = discover_periodic_trigonometric_surface(
        source, policy=_policy(diameter), source_id=source_id
    )
    if not surface.resolved:
        raise RuntimeError(f"Quotient discovery of {source_id} stayed unresolved.")
    return surface


def _fill_record(volume: PeriodicImplicitVolume) -> dict[str, Any]:
    result = volume.result
    result.audit.require_passed()
    mesh = volume.mesh
    return {
        "result_id": result.result_id,
        "vertices": int(mesh.coordinates.shape[0]),
        "tetrahedra": sum(block.cell_count for block in mesh.blocks),
        "audit_passed": result.audit.passed,
        "quotient_region_components": volume.quotient_region_components,
        "cap_triangles": volume.cap_triangle_count,
        "measure": volume.measure,
        "soft_shape_target_unmet_count": dict(
            volume.construction.compliance.achieved
        ).get("soft_shape_target_unmet_count"),
    }


def _conductivity(mesh: Any) -> Any:
    problem = PeriodicCellProblem(mesh, physics="conductivity")
    return problem, problem.solve(1.0)


def _conductivity_record(problem: Any, result: Any) -> dict[str, Any]:
    tensor = np.asarray(result.effective_tensor, dtype=np.float64)
    phi = float(result.volume_fraction)
    mean = float(np.trace(tensor)) / 3.0
    upper = 2.0 * phi / (3.0 - phi)
    record = {
        "effective_tensor": tensor.tolist(),
        "isotropic_part": mean,
        "volume_fraction": phi,
        "component_winding_ranks": list(problem.space.component_winding_ranks),
        "accepted": bool(result.accepted),
        "symmetry_defect": float(result.symmetry_defect),
        "energy_defect": float(result.energy_defect),
        "cubic_symmetry_defect": float(result.cubic_symmetry_defect),
        "hashin_shtrikman_void_upper": upper,
        "hill_mandel_power_defects": [
            float(audit.power_defect) for audit in result.hill_mandel
        ],
    }
    if not record["accepted"]:
        raise RuntimeError("Homogenized conductivity failed its Hill-Mandel audit.")
    if not 0.0 < mean < upper:
        raise RuntimeError(
            "Homogenized conductivity leaves the Hashin-Shtrikman void bracket."
        )
    return record


# ---------------------------------------------------------------------------
# Profiles


def _gyroid_sheet_cell() -> dict[str, Any]:
    timings: dict[str, float] = {}
    table = _G.TrigonometricTable.family("gyroid")
    source = _timed(
        timings,
        "source_seconds",
        lambda: _G.TrigonometricLevelSet(
            table,
            PeriodicCell(np.eye(3)),
            region="sheet",
            sheet=_G.TrigonometricSheetBand(_GYROID_TAU),
        ),
    )
    surface = _timed(
        timings,
        "discovery_seconds",
        lambda: _discover(source, _GYROID_DIAMETER, "gyroid-sheet"),
    )
    topology = _topology_record(surface)
    # Each band level +-tau is separated from every critical value, so each
    # boundary sheet is isotopic to the gyroid's conventional cell: genus 5
    # (Schoen, NASA TN D-5541, 1970).
    if topology["component_genera"] != [5, 5] or not (
        topology["closed"] and topology["orientable"]
    ):
        raise RuntimeError(f"Unexpected gyroid sheet quotient topology {topology}.")
    lifted_translates = _exact_lift(surface)
    metrics = surface.lattice_metrics()
    fine = _timed(
        timings,
        "oracle_seconds",
        lambda: _gyroid_band_fraction(_GYROID_TAU, 512),
    )
    coarse = _gyroid_band_fraction(_GYROID_TAU, 256)
    oracle_error = abs(fine - coarse)
    lower, upper = metrics.solid_fraction_lower, metrics.solid_fraction_upper
    if not lower + oracle_error <= fine <= upper - oracle_error:
        raise RuntimeError(
            f"Independent sheet fraction {fine} lies outside the enclosure [{lower}, {upper}]."
        )
    volume = _timed(
        timings,
        "fill_seconds",
        lambda: mesh_periodic_trigonometric_region(
            surface,
            # The sheet (half-width 0.3) is about one default target edge
            # thick, so the hard default shape targets fail at this resolution
            # (1138 radius-edge and 995 dihedral rows); they pass only at
            # maximum_edge_length=0.03 (2-3 edges across: 155705 cells, about
            # 2.5x the wall and 7x the peak memory). This profile keeps the
            # default resolution and enforces cell validity only.
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=None,
                minimum_dihedral_degrees=0.0,
                shape_strength=SizeControlStrength.SOFT,
            ),
        ),
    )
    fill = _fill_record(volume)
    pairs = [_cell_face_pairs(volume, axis) for axis in range(3)]
    if sum(pairs) != volume.cap_triangle_count:
        raise RuntimeError("Cell-face cap triangles are not all exactly paired.")
    if volume.quotient_region_components != 1:
        raise RuntimeError("The gyroid sheet quotient region is not connected.")
    group = _table_point_group(table, (1, -1))
    invariant = _invariant_symmetric_dimension(group)
    if invariant != 1:
        raise RuntimeError("The proved gyroid point group does not force isotropy.")
    problem, result = _timed(
        timings, "homogenization_seconds", lambda: _conductivity(volume.mesh)
    )
    conduction = _conductivity_record(problem, result)
    # Symbolic comparison only: the solve uses the default nested-dissection
    # Cholesky under the library's default budget; the reverse Cuthill-McKee
    # analysis declares its own budget so its fill can be reported.
    bandwidth = PeriodicSimplexOperatorPattern(
        problem.space,
        components=1,
        factorization=SparseFactorizationPolicy(
            "cholesky",
            ordering="reverse-cuthill-mckee",
            max_factor_nnz=_RCM_COMPARISON_FACTOR_NNZ,
        ),
    )
    orderings = {
        "nested_dissection_factor_nnz": int(problem.pattern.plan.factor_nnz),
        "reverse_cuthill_mckee_factor_nnz": int(bandwidth.plan.factor_nnz),
        "default_max_factor_nnz": SparseFactorizationPolicy().max_factor_nnz,
    }
    if conduction["cubic_symmetry_defect"] > _CUBIC_DEFECT_ACCEPTANCE:
        raise RuntimeError("The measured cubic-symmetry defect exceeds its acceptance.")
    if abs(conduction["volume_fraction"] - volume.measure) > 1.0e-12 * volume.measure:
        raise RuntimeError("Homogenization measured a different region volume.")
    return {
        "status": "passed",
        "family": "gyroid",
        "region": "sheet",
        "half_width": _GYROID_TAU,
        "discovery": {
            "relative_cell_diameter": surface.policy.relative_cell_diameter,
            "target_level": surface.target_level,
            "root_error": surface.root_error,
            "topology": topology,
            "lifted_translate_vertices": lifted_translates,
        },
        "volume_fraction": {
            "certified_lower": lower,
            "certified_upper": upper,
            "independent_quadrature": fine,
            "quadrature_self_convergence": oracle_error,
            "quadrature": "exact in x; composite 8-point Gauss-Legendre, 512^2 panels in (y, z)",
            "mesh_measure": volume.measure,
            "measured_surface_area": metrics.measured_surface_area,
            "vertex_root_distance": metrics.vertex_root_distance,
        },
        "fill": {**fill, "paired_cap_triangles_per_axis": pairs},
        "conductivity": {
            **conduction,
            "proved_point_group_order": len(group),
            "invariant_symmetric_tensor_dimension": invariant,
            "factorization": orderings,
            "cubic_defect_acceptance": _CUBIC_DEFECT_ACCEPTANCE,
        },
        "timings": timings,
    }


def _schwarz_p(
    region: Literal["network", "complement"],
) -> tuple[PeriodicImplicitSurface, PeriodicImplicitVolume]:
    source = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3)), region=region
    )
    surface = _discover(source, _SCHWARZ_P_DIAMETER, f"schwarz_p-{region}")
    return surface, mesh_periodic_trigonometric_region(surface)


def _schwarz_p_network() -> tuple[dict[str, Any], PeriodicImplicitVolume]:
    timings: dict[str, float] = {}
    table = _G.TrigonometricTable.family("schwarz_p")
    half = Fraction(1, 2)
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    sign = table.symmetry_sign(identity, (half, half, half))
    if sign != -1:
        raise RuntimeError("The half-lattice shift does not prove F(u + 1/2) = -F(u).")
    network, network_volume = _timed(
        timings, "network_seconds", lambda: _schwarz_p("network")
    )
    complement, complement_volume = _timed(
        timings, "complement_seconds", lambda: _schwarz_p("complement")
    )
    records: dict[str, dict[str, Any]] = {}
    for name, surface, volume in (
        ("network", network, network_volume),
        ("complement", complement, complement_volume),
    ):
        metrics = surface.lattice_metrics()
        if not metrics.solid_fraction_lower <= 0.5 <= metrics.solid_fraction_upper:
            raise RuntimeError(f"The {name} enclosure misses the proved fraction 1/2.")
        problem, result = _conductivity(volume.mesh)
        records[name] = {
            "topology": _topology_record(surface),
            "certified_fraction": [
                metrics.solid_fraction_lower,
                metrics.solid_fraction_upper,
            ],
            "fill": _fill_record(volume),
            "conductivity": _conductivity_record(problem, result),
        }
        # Each labyrinth is one quotient component winding along all axes.
        if volume.quotient_region_components != 1 or list(
            problem.space.component_winding_ranks
        ) != [3]:
            raise RuntimeError(f"The Schwarz P {name} is not one percolating component.")
        if records[name]["topology"]["component_genera"] != [3]:
            raise RuntimeError("The Schwarz P surface is not genus 3 per cubic cell.")
    total = network_volume.measure + complement_volume.measure
    return {
        "status": "passed",
        "family": "schwarz_p",
        "level": 0.0,
        "symmetry_proof": {
            "map": "u -> u + (1/2, 1/2, 1/2)",
            "sign": sign,
            "proved_fraction": 0.5,
        },
        "network": records["network"],
        "complement": records["complement"],
        "network_plus_complement_measure": total,
        "timings": timings,
    }, network_volume


def _schwarz_p_singular() -> dict[str, Any]:
    table = _G.TrigonometricTable.family("schwarz_p")
    levels = _G.certify_trigonometric_critical_levels(table)
    refusals = []
    for level in (1.0, -1.0, 3.0):
        enclosure = levels.separate(level).critical_levels.enclosure_meeting(level)
        try:
            _G.TrigonometricLevelSet(table, PeriodicCell(np.eye(3)), level=level)
        except _G.TrigonometricSingularLevelError as error:
            separation = error.separation
            refusals.append(
                {
                    "level": level,
                    "critical_value_enclosure": None
                    if enclosure is None
                    else list(enclosure),
                    "outcome": separation.outcome,
                    "evidence_value": None
                    if separation.evidence_value is None
                    else list(separation.evidence_value),
                }
            )
        else:
            raise RuntimeError(
                f"Schwarz P at the critical level {level} was not refused."
            )
        if enclosure is None or not enclosure[0] <= level <= enclosure[1]:
            raise RuntimeError("The refused level lies outside its critical enclosure.")
    regular = _G.TrigonometricLevelSet(table, PeriodicCell(np.eye(3)), level=0.5)
    return {
        "status": "passed",
        "family": "schwarz_p",
        "refusals": refusals,
        "regular_level_accepted": regular.level,
        "regular_level_separation": regular.critical_levels.separation(0.5),
    }


def _schwarz_d_length(
    x: np.ndarray, y: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Exact length of ``{F <= 0}`` for Schwarz D on vertical segments.

    ``F = R sin(Z + phi)`` with ``R cos phi = cos(X - Y)`` and ``R sin phi =
    sin(X + Y)``, ``(X, Y, Z) = 2 pi (x, y, z)``; ``sin <= 0`` on half of
    every period.
    """
    angle_x, angle_y = 2.0 * math.pi * x, 2.0 * math.pi * y
    phase = np.arctan2(np.sin(angle_x + angle_y), np.cos(angle_x - angle_y))

    def below(theta: np.ndarray) -> np.ndarray:
        turns = np.floor(theta / (2.0 * math.pi))
        rest = theta - 2.0 * math.pi * turns
        return math.pi * turns + np.clip(rest - math.pi, 0.0, math.pi)

    return (below(2.0 * math.pi * high + phase) - below(2.0 * math.pi * low + phase)) / (
        2.0 * math.pi
    )


def _cylinder_network_volume(count: int) -> float:
    """Gauss-Legendre in the radius, periodic trapezoid in the angle, exact in z."""
    nodes, weights = np.polynomial.legendre.leggauss(count)
    radii = 0.5 * _CYLINDER_RADIUS * (nodes + 1.0)
    angles = 2.0 * math.pi * np.arange(2 * count) / (2 * count)
    r, theta = np.meshgrid(radii, angles, indexing="ij")
    weight = (0.5 * _CYLINDER_RADIUS * weights * radii)[:, None] * (
        2.0 * math.pi / (2 * count)
    )
    low = np.full(r.shape, _CYLINDER_CENTER[2])
    return float(
        np.sum(
            weight
            * _schwarz_d_length(
                _CYLINDER_CENTER[0] + r * np.cos(theta),
                _CYLINDER_CENTER[1] + r * np.sin(theta),
                low,
                low + _CYLINDER_AXIS[2],
            )
        )
    )


def _cap_rows(volume: LatticeSpecimenVolume, name: str, dof: np.ndarray) -> np.ndarray:
    """DOF rows of the cap vertices named ``name``, matched bit for bit."""
    surface = volume.surface
    facet = surface.facet_names.index(name)
    triangles = np.asarray(surface.triangles, dtype=np.int64)[
        np.asarray(surface.triangle_facets) == facet
    ]
    points = np.asarray(surface.vertices, dtype=np.float64)[np.unique(triangles)]
    index = {row.tobytes(): position for position, row in enumerate(dof)}
    rows = [index.get(point.tobytes()) for point in points]
    if any(row is None for row in rows):
        raise RuntimeError(f"A {name} cap vertex is not an FE degree of freedom.")
    return np.asarray(rows, dtype=np.int64)


def _elastic_problem(space: Any, mask: np.ndarray, values: np.ndarray) -> Any:
    lame = _YOUNG * _POISSON / ((1.0 + _POISSON) * (1.0 - 2.0 * _POISSON))
    shear = _YOUNG / (2.0 * (1.0 + _POISSON))
    constraint = phx.discretization.dirichlet_constraint(space, "u", boundary_mask=mask)
    return phx.equations.compile_finite_element_problem(
        phx.equations.fem.linear_elasticity_form("u", lame, shear),
        space,
        constraint=constraint,
        dirichlet_values=values,
    )


def _solve(problem: Any) -> tuple[np.ndarray, float]:
    """Sparse SPD Cholesky of the constrained elastic operator, residual measured.

    The elasticity form lowers to the coupled vector stiffness; the cap
    constraints are eliminated sparsely and the free block is factored by the
    native Cholesky under ``_ELASTIC_FACTORIZATION``. A successful factor with
    positive pivots is the SPD evidence of the eliminated operator.
    """
    system, rhs = problem.linear_system(storage="sparse")
    factor = phx.linalg.factorize_sparse(system.operator, _ELASTIC_FACTORIZATION)
    if int(factor.status) != int(phx.linalg.SparseFactorizationStatus.SUCCESS) or (
        int(factor.diagnostics.rejected_pivots) != 0
    ):
        raise RuntimeError("The constrained elastic operator failed SPD Cholesky.")
    solved = factor.solve(rhs)
    if not bool(solved.success):
        raise RuntimeError("The elastic Cholesky solve did not succeed.")
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    scale = max(1.0, float(jnp.linalg.norm(rhs)))
    return np.asarray(problem.expand(solved.value), dtype=np.float64), residual / scale


def _specimen_compression(volume: LatticeSpecimenVolume) -> dict[str, Any]:
    """Linear-elastic compression between bonded platens, with discrete oracles.

    Base cap fixed, top cap displaced by ``-strain * height`` along the axis.
    The uniaxial-strain field ``u = (0, 0, -strain (z - z0))`` is P1 and meets
    every Dirichlet value, so the FE strain energy cannot exceed its energy
    ``(lambda + 2 mu) strain^2 |V| / 2`` (minimum potential energy over the
    same constrained space). An affine field imposed on the whole boundary is
    reproduced at the interior nodes (patch test).
    """
    result = volume.result
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("tetrahedron", 1), component_shape=(3,)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    dof = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    count = dof.shape[0]
    base = _cap_rows(volume, "base", dof)
    top = _cap_rows(volume, "top", dof)
    height = _CYLINDER_AXIS[2]
    z0 = _CYLINDER_CENTER[2]
    trial = np.zeros((count, 3))
    trial[:, 2] = -_COMPRESSION_STRAIN * (dof[:, 2] - z0)
    mask = np.zeros(count, dtype=np.bool_)
    mask[base] = mask[top] = True
    problem = _elastic_problem(space, mask, np.where(mask[:, None], trial, 0.0))
    displacement, residual = _solve(problem)
    unconstrained = phx.equations.compile_finite_element_problem(
        phx.equations.fem.linear_elasticity_form(
            "u",
            _YOUNG * _POISSON / ((1.0 + _POISSON) * (1.0 - 2.0 * _POISSON)),
            _YOUNG / (2.0 * (1.0 + _POISSON)),
        ),
        space,
    )
    force = np.asarray(
        unconstrained.residual(jnp.asarray(displacement)), dtype=np.float64
    )
    energy = 0.5 * float(np.sum(force * displacement))
    trial_force = np.asarray(unconstrained.residual(jnp.asarray(trial)), dtype=np.float64)
    trial_energy = 0.5 * float(np.sum(trial_force * trial))
    lame = _YOUNG * _POISSON / ((1.0 + _POISSON) * (1.0 - 2.0 * _POISSON))
    shear = _YOUNG / (2.0 * (1.0 + _POISSON))
    analytic_trial = 0.5 * (lame + 2.0 * shear) * _COMPRESSION_STRAIN**2 * volume.measure
    top_reaction = float(np.sum(force[top, 2]))
    base_reaction = float(np.sum(force[base, 2]))
    free_force = float(np.max(np.abs(force[~mask]), initial=0.0))
    delta = _COMPRESSION_STRAIN * height
    work = -0.5 * top_reaction * delta
    stiffness = -top_reaction / delta
    if residual > _RESIDUAL_TOLERANCE:
        raise RuntimeError("The compression solve residual exceeds its acceptance.")
    if not 0.0 < energy <= trial_energy:
        raise RuntimeError("FE energy violates the discrete kinematic upper bound.")
    # The assembled energy of the constant-strain trial is closed form.
    if abs(trial_energy - analytic_trial) > _PATCH_TOLERANCE * analytic_trial:
        raise RuntimeError("Assembled uniaxial-strain energy differs from closed form.")
    # Patch test: an affine displacement on every boundary node.
    gradient = np.asarray(((1.0, 0.2, -0.3), (0.1, -0.5, 0.25), (0.4, 0.0, 0.7))) * 1e-3
    affine = dof @ gradient.T + np.asarray((1e-3, -2e-3, 5e-4))
    boundary = np.asarray(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    patch, patch_residual = _solve(
        _elastic_problem(space, boundary, np.where(boundary[:, None], affine, 0.0))
    )
    patch_error = float(np.max(np.abs(patch - affine))) / float(np.max(np.abs(affine)))
    if patch_error > _PATCH_TOLERANCE:
        raise RuntimeError("The affine patch test is not reproduced.")
    return {
        "young_modulus": _YOUNG,
        "poisson_ratio": _POISSON,
        "strain": _COMPRESSION_STRAIN,
        "platen_displacement": delta,
        "dofs": int(3 * count),
        "base_cap_nodes": int(base.size),
        "top_cap_nodes": int(top.size),
        "normalized_residual": residual,
        "top_reaction": top_reaction,
        "base_reaction": base_reaction,
        "maximum_free_node_force": free_force,
        "axial_stiffness": stiffness,
        "strain_energy": energy,
        "reaction_work": work,
        "kinematic_upper_bound_energy": trial_energy,
        "kinematic_upper_bound_analytic": analytic_trial,
        "energy_over_bound": energy / trial_energy,
        "apparent_modulus": stiffness * height / (math.pi * _CYLINDER_RADIUS**2),
        "patch_test_relative_error": patch_error,
        "patch_test_residual": patch_residual,
        "interior_patch_nodes": int(np.count_nonzero(~boundary)),
    }


def _diamond_cylinder_specimen(directory: Path) -> dict[str, Any]:
    timings: dict[str, float] = {}
    network = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_d"), PeriodicCell(np.eye(3))
    )
    envelope = _G.Cylinder(_CYLINDER_CENTER, _CYLINDER_AXIS, _CYLINDER_RADIUS)
    surface = _timed(
        timings,
        "surface_seconds",
        lambda: construct_lattice_specimen_surface(
            _G.Intersection(network, envelope),
            policy=_policy(_SPECIMEN_DIAMETER, _SPECIMEN_ROOT_TOLERANCE),
            source_id="diamond-cylinder",
        ),
    )
    volume = _timed(timings, "fill_seconds", lambda: mesh_lattice_specimen(surface))
    if not volume.certified_solid:
        raise RuntimeError("The diamond cylinder specimen is not a certified solid.")
    if surface.crease_segments.shape[0] == 0:
        raise RuntimeError("The specimen has no certified creases.")
    corners = np.asarray(surface.vertices)[np.asarray(surface.triangles)]
    area = float(
        0.5
        * np.sum(
            np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
        )
    )
    oracle = _timed(timings, "oracle_seconds", lambda: _cylinder_network_volume(384))
    oracle_coarse = _cylinder_network_volume(192)
    fidelity = max(surface.implicit_face_distance, surface.cap_deviation)
    tube = area * fidelity
    error = abs(volume.measure - oracle)
    if not math.isfinite(fidelity) or error > tube:
        raise RuntimeError("The specimen volume leaves its certified fidelity tube.")
    envelope_volume = math.pi * _CYLINDER_RADIUS**2 * _CYLINDER_AXIS[2]
    porosity_oracle = 1.0 - oracle / envelope_volume
    exported = _timed(
        timings,
        "export_seconds",
        lambda: export_lattice_solid(
            directory / "diamond-cylinder.3mf", volume, SurfaceExportPolicy(METER)
        ),
    )
    compression = _timed(
        timings, "compression_seconds", lambda: _specimen_compression(volume)
    )
    names = surface.facet_names
    facets = np.asarray(surface.triangle_facets)
    return {
        "status": "passed",
        "family": "schwarz_d",
        "envelope": {
            "kind": "cylinder",
            "center": list(_CYLINDER_CENTER),
            "axis": list(_CYLINDER_AXIS),
            "radius": _CYLINDER_RADIUS,
        },
        "creases_and_caps": {
            "facet_triangles": {
                name: int(np.count_nonzero(facets == index))
                for index, name in enumerate(names)
            },
            "crease_segments": int(surface.crease_segments.shape[0]),
            "envelope_edge_segments": int(surface.envelope_edge_segments.shape[0]),
            "crease_root_distance": surface.crease_root_distance,
            "cap_deviation": surface.cap_deviation,
            "implicit_face_distance": surface.implicit_face_distance,
        },
        "solid": {
            "certified_solid": volume.certified_solid,
            "boundary_matches_surface": volume.boundary_matches_surface,
            "covers_surface": volume.covers_surface,
            "result_id": volume.result.result_id,
            "vertices": int(volume.mesh.coordinates.shape[0]),
            "tetrahedra": sum(block.cell_count for block in volume.mesh.blocks),
            "export_3mf": {
                "certificate_id": exported.certificate.certificate_id,
                "artifact_digest": exported.export.report.artifact_digest,
                "lossy": exported.lossy,
                "components": exported.audit.component_count,
                "enclosed_volume": exported.enclosed_volume,
            },
        },
        "volume_oracle": {
            "mesh_measure": volume.measure,
            "independent_volume": oracle,
            "independent_self_convergence": abs(oracle - oracle_coarse),
            "absolute_error": error,
            "fidelity_tube": tube,
            "boundary_area": area,
            "porosity": volume.porosity,
            "independent_porosity": porosity_oracle,
            "porosity_error": abs(volume.porosity - porosity_oracle),
            "porosity_tube": tube / envelope_volume,
        },
        "compression": compression,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Independent-process cold restore


def _cold_identity(volume_result: Any) -> dict[str, Any]:
    mesh = volume_result.mesh
    topology = mesh.periodic_topology
    return {
        "result_id": volume_result.result_id,
        "mesh_id": mesh.mesh_id,
        "topology_id": mesh.topology_id,
        "periodic_topology_id": topology.periodic_topology_id,
        "audit_report_id": volume_result.audit.report_id,
        "certification_id": None
        if volume_result.certification is None
        else volume_result.certification.report_id,
    }


def _quotient_measure(mesh: Any) -> float:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    total = 0.0
    for block in mesh.blocks:
        cells = coordinates[np.asarray(block.vertices, dtype=np.int64)]
        total += float(np.sum(np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])))) / 6.0
    return total


def _read_cold(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Fresh-process restore: identities, certification, physics and export."""
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
    )
    from phydrax.meshing._periodic_implicit_volume import _quotient_components
    from phydrax.meshing._result import CellMeshingResult

    result = read_model_artifact(
        directory / "schwarz-p-network.phx",
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    identity = _cold_identity(result)
    if identity != expected["identity"]:
        raise RuntimeError("Cold TPMS restore changed its accepted identities.")
    result.audit.require_passed()
    if result.certification is not None:
        result.certification.require_passed()
        result.certification.revalidate(result.mesh, result.geometry)
    mesh = result.mesh
    topology = mesh.periodic_topology
    components = _quotient_components(mesh, topology)
    measure = _quotient_measure(mesh)
    problem, solved = _conductivity(mesh)
    tensor = np.asarray(solved.effective_tensor, dtype=np.float64)
    stored = np.asarray(expected["effective_tensor"], dtype=np.float64)
    return {
        "pid": os.getpid(),
        "identity": identity,
        "quotient_region_components": components,
        "measure": measure,
        "measure_bitwise": measure == expected["measure"],
        "effective_tensor": tensor.tolist(),
        "effective_tensor_bitwise": bool(np.array_equal(tensor, stored)),
        "effective_tensor_max_difference": float(np.max(np.abs(tensor - stored))),
        "component_winding_ranks": list(problem.space.component_winding_ranks),
        "accepted": bool(solved.accepted),
    }


def _periodic_cold_restore(volume: PeriodicImplicitVolume) -> dict[str, Any]:
    from phydrax.meshing._model_codec import write_model_artifact

    result = volume.result
    _, solved = _conductivity(volume.mesh)
    expected = {
        "identity": _cold_identity(result),
        "measure": _quotient_measure(volume.mesh),
        "effective_tensor": np.asarray(solved.effective_tensor).tolist(),
    }
    with tempfile.TemporaryDirectory(prefix="phydrax-tpms-cold-") as temporary:
        directory = Path(temporary)
        write_model_artifact(directory / "schwarz-p-network.phx", result, limits=None)
        archive_bytes = (directory / "schwarz-p-network.phx").stat().st_size
        reader = subprocess.run(
            (
                sys.executable,
                "-m",
                __name__,
                "read",
                str(directory),
                json.dumps(expected),
            ),
            capture_output=True,
            text=True,
            check=False,
        )
    if reader.returncode:
        raise RuntimeError(f"TPMS cold reader failed:\n{reader.stdout}\n{reader.stderr}")
    restored = json.loads(reader.stdout.strip().splitlines()[-1])
    if restored["pid"] == os.getpid():
        raise RuntimeError("TPMS cold writer and reader must be separate processes.")
    if not (
        restored["measure_bitwise"]
        and restored["effective_tensor_bitwise"]
        and restored["accepted"]
        and restored["quotient_region_components"] == volume.quotient_region_components
    ):
        raise RuntimeError("The cold restore does not reproduce the accepted lattice.")
    return {
        "status": "passed",
        "producer_pid": os.getpid(),
        "archive_bytes": archive_bytes,
        "expected": expected,
        "consumer": restored,
    }


# ---------------------------------------------------------------------------
# Scenario


def _profile(name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = perf_counter()
    try:
        record = run()
    except Exception as error:
        record = {
            "status": "failed",
            "error_type": type(error).__name__,
            "detail": str(error),
            "traceback": traceback.format_exc(),
        }
    return {"profile": name, **record, "wall_seconds": perf_counter() - started}


def native_tpms() -> dict[str, object]:
    meshcore = require_meshcore()
    held: dict[str, PeriodicImplicitVolume] = {}

    def network() -> dict[str, Any]:
        record, volume = _schwarz_p_network()
        held["network"] = volume
        return record

    def cold() -> dict[str, Any]:
        volume = held.get("network")
        if volume is None:
            raise RuntimeError("The Schwarz P network profile produced no lattice.")
        return _periodic_cold_restore(volume)

    with tempfile.TemporaryDirectory(prefix="phydrax-tpms-") as temporary:
        profiles = (
            _profile("gyroid-sheet-cell", _gyroid_sheet_cell),
            _profile("schwarz-p-network", network),
            _profile("schwarz-p-singular", _schwarz_p_singular),
            _profile(
                "diamond-cylinder-specimen",
                lambda: _diamond_cylinder_specimen(Path(temporary)),
            ),
            _profile("periodic-cold-restore", cold),
        )
    passed = all(profile["status"] == "passed" for profile in profiles)
    return {
        "status": "passed" if passed else "failed",
        "workflow": "trigonometric nodal sources -> quotient discovery -> periodic fill "
        "-> homogenization; clipped specimens -> certified solid -> FE and 3MF export",
        "profile_order": [profile["profile"] for profile in profiles],
        "profiles": list(profiles),
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }


def _standalone_specimen() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phydrax-tpms-") as temporary:
        return _diamond_cylinder_specimen(Path(temporary))


_PROFILES: dict[str, Callable[[], dict[str, Any]]] = {
    "gyroid-sheet-cell": _gyroid_sheet_cell,
    "schwarz-p-network": lambda: _schwarz_p_network()[0],
    "schwarz-p-singular": _schwarz_p_singular,
    "diamond-cylinder-specimen": _standalone_specimen,
    "periodic-cold-restore": lambda: _periodic_cold_restore(_schwarz_p("network")[1]),
}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "read":
        print(json.dumps(_read_cold(Path(sys.argv[2]), json.loads(sys.argv[3]))))
    elif len(sys.argv) == 3 and sys.argv[1] == "profile":
        print(
            json.dumps(
                _profile(sys.argv[2], _PROFILES[sys.argv[2]]),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        raise SystemExit(
            "Usage: native_tpms read <directory> <expected> | profile <name>"
        )

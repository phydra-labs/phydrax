#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.discretization._periodic_exact import apply_periodic
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
    PeriodicImplicitSurface,
)


_G = phx.geometry
_HALF = Fraction(1, 2)
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_INVERSION = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))


def _discover(
    family: _G.TrigonometricFamily,
    diameter: float,
    vectors: np.ndarray | None = None,
) -> PeriodicImplicitSurface:
    table = _G.TrigonometricTable.family(family)
    cell = PeriodicCell(np.eye(3) if vectors is None else vectors)
    source = _G.TrigonometricLevelSet(table, cell)
    return discover_periodic_trigonometric_surface(
        source,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=diameter, root_tolerance=1.0e-8
        ),
        source_id=f"{family}-quotient",
    )


def _assert_exact_lift(surface: PeriodicImplicitSurface) -> None:
    mesh = surface.mesh
    assert mesh is not None and mesh.periodic_topology is not None
    topology = mesh.periodic_topology
    coordinates = np.asarray(mesh.coordinates)
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    # Translates on upper cell faces are their representatives moved by the
    # exact lattice action, bit for bit.
    assert np.any(shifts != 0)
    np.testing.assert_array_equal(
        coordinates, apply_periodic(surface.cell, coordinates[representatives], shifts)
    )


@pytest.mark.parametrize(
    ("family", "diameter", "genus", "symmetry"),
    (
        ("schwarz_p", 0.2, 3, (_IDENTITY, (_HALF, _HALF, _HALF))),
        ("gyroid", 0.1, 5, (_INVERSION, (0, 0, 0))),
        ("schwarz_d", 0.1, 9, (_INVERSION, (0, 0, 0))),
    ),
)
def test_quotient_discovery_reports_cited_cubic_cell_genus_and_half_fraction(
    family: _G.TrigonometricFamily,
    diameter: float,
    genus: int,
    symmetry: tuple[tuple[tuple[int, int, int], ...], tuple[int | Fraction, ...]],
) -> None:
    surface = _discover(family, diameter)

    # Conventional cubic cells: P genus 3, G genus 5 (two primitive cells of
    # genus 3), D genus 9 (four primitive cells); Schoen, NASA TN D-5541 (1970).
    assert surface.resolved and surface.topology is not None
    assert not surface.evidence.unresolved_issues.size
    topology = surface.topology
    assert topology.closed and topology.orientable
    assert topology.component_count == 1
    assert topology.component_genera == (genus,)
    assert topology.euler_characteristic == 2 - 2 * genus
    _assert_exact_lift(surface)

    # A measure-preserving lattice map with F o g = -F proves fraction 1/2,
    # which the mesh-independent cover encloses.
    table = _G.TrigonometricTable.family(family)
    assert table.symmetry_sign(*symmetry) == -1
    metrics = surface.lattice_metrics()
    assert metrics.solid_fraction_lower <= 0.5 <= metrics.solid_fraction_upper
    assert metrics.porosity_lower <= 0.5 <= metrics.porosity_upper


def test_quotient_discovery_on_a_skew_cell_keeps_topology_and_exact_lifts() -> None:
    vectors = np.asarray(((1.3, 0.0, 0.0), (0.4, 0.9, 0.0), (-0.2, 0.3, 1.7)))
    surface = _discover("schwarz_p", 0.2, vectors)

    assert surface.topology is not None
    assert surface.topology.component_genera == (3,)
    _assert_exact_lift(surface)
    metrics = surface.lattice_metrics()
    assert metrics.cell_volume == pytest.approx(abs(np.linalg.det(vectors)))


def _independent_schwarz_p_area(points_per_axis: int, half_width: float) -> float:
    """Coarea quadrature of the area: |{|F| < e}| integrand |grad F| / (2 e)."""
    u = (np.arange(points_per_axis) + 0.5) / points_per_axis
    x, y, z = np.meshgrid(*(2.0 * np.pi * u,) * 3, indexing="ij")
    field = np.cos(x) + np.cos(y) + np.cos(z)
    gradient = 2.0 * np.pi * np.sqrt(np.sin(x) ** 2 + np.sin(y) ** 2 + np.sin(z) ** 2)
    band = np.abs(field) < half_width
    return float(np.sum(gradient[band]) / points_per_axis**3 / (2.0 * half_width))


def test_lattice_metrics_report_measured_area_against_independent_quadrature() -> None:
    surface = _discover("schwarz_p", 0.1)
    metrics = surface.lattice_metrics()
    reference = _independent_schwarz_p_area(256, 0.05)

    # Measured, not enclosed: the contour area matches an independent coarea
    # quadrature of the nodal surface to the discretization level.
    assert metrics.measured_surface_area == pytest.approx(reference, rel=0.02)
    assert metrics.measured_specific_surface == pytest.approx(
        metrics.measured_surface_area / metrics.cell_volume
    )
    low, high = metrics.measured_solid_specific_surface
    assert low <= metrics.measured_surface_area / 0.5 <= high
    assert 0.0 < metrics.vertex_root_distance <= 1.0e-8


def test_singular_quotient_level_is_refused_before_discovery() -> None:
    table = _G.TrigonometricTable.family("schwarz_p")
    with pytest.raises(_G.TrigonometricSingularLevelError) as refusal:
        _G.TrigonometricLevelSet(table, PeriodicCell(np.eye(3)), level=1.0)
    assert refusal.value.separation.outcome == "quarter_point"

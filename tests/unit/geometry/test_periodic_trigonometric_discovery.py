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
    # The identity lattice has unit spectral stretch, up to outward rounding.
    assert Fraction(metrics.vertex_root_distance) >= Fraction(surface.root_error)
    assert Fraction(metrics.vertex_root_distance) <= Fraction(surface.root_error) * (
        1 + Fraction(1, 2**48)
    )
    assert metrics.porosity_lower <= 0.5 <= metrics.porosity_upper


def test_quotient_discovery_on_a_skew_cell_keeps_topology_and_exact_lifts() -> None:
    vectors = np.asarray(((1.3, 0.0, 0.0), (0.4, 0.9, 0.0), (-0.2, 0.3, 1.7)))
    surface = _discover("schwarz_p", 0.2, vectors)

    assert surface.topology is not None
    assert surface.topology.component_genera == (3,)
    _assert_exact_lift(surface)
    metrics = surface.lattice_metrics()
    assert metrics.cell_volume == pytest.approx(abs(np.linalg.det(vectors)))
    _assert_stretch_certificate(surface, vectors, metrics.vertex_root_distance)


def _assert_stretch_certificate(
    surface: PeriodicImplicitSurface, vectors: np.ndarray, distance: float
) -> None:
    # Exact principal-minor certificate that distance² I - error² V Vᵀ is PSD.
    distance2 = Fraction(distance) ** 2
    error2 = Fraction(surface.root_error) ** 2
    exact = [[Fraction(float(value)) for value in row] for row in vectors]
    matrix = [
        [
            (distance2 if i == j else Fraction(0))
            - error2 * sum((exact[i][k] * exact[j][k] for k in range(3)), Fraction(0))
            for j in range(3)
        ]
        for i in range(3)
    ]
    assert all(matrix[i][i] >= 0 for i in range(3))
    assert all(
        matrix[i][i] * matrix[j][j] - matrix[i][j] ** 2 >= 0
        for i in range(3)
        for j in range(i + 1, 3)
    )
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] ** 2)
        - matrix[0][1] * (matrix[0][1] * matrix[2][2] - matrix[1][2] * matrix[0][2])
        + matrix[0][2] * (matrix[0][1] * matrix[1][2] - matrix[1][1] * matrix[0][2])
    )
    assert determinant >= 0


def test_lattice_metrics_stretch_for_underestimated_spectral_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bracket the spectral norm with exact principal minors in binary64.

    Bisection finds adjacent binary64 numbers around the exact largest singular
    value. The patched NumPy spectral norm returns the lower number, independent
    of LAPACK; the old one-step metric is proved too small by exact arithmetic.
    """
    vectors = np.asarray(
        (
            (-2.21875, -1.8125, -1.6875),
            (-0.78125, 0.1875, 1.21875),
            (-2.6875, -2.3125, 2.21875),
        ),
        dtype=np.float64,
    )
    discovered = _discover("schwarz_p", 0.2, vectors)
    # Widening the certified error is valid and makes the rounding case fixed.
    error = float.fromhex("0x1.0000000000002p+0")
    assert discovered.root_error <= error
    surface = PeriodicImplicitSurface(
        discovered.source,
        discovered.mesh,
        discovered.topology,
        discovered.cover,
        discovered.evidence,
        discovered.reduced_domain,
        None,
        {
            "corner_ids": discovered.scaffold_corner_ids,
            "corner_signs": discovered.scaffold_corner_signs,
            "root_edge_ids": discovered.scaffold_root_edge_ids,
            "triangle_root_edge_rows": discovered.scaffold_triangle_root_edge_rows,
        },
        discovered.vertex_root_rows,
        target_level=discovered.target_level,
        root_error=error,
        policy=discovered.policy,
        source_id=discovered.source_id,
    )
    exact = [[Fraction(float(value)) for value in row] for row in vectors]

    def upper_bound(value: Fraction) -> bool:
        matrix = [
            [
                (value**2 if i == j else Fraction(0))
                - sum((exact[i][k] * exact[j][k] for k in range(3)), Fraction(0))
                for j in range(3)
            ]
            for i in range(3)
        ]
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] ** 2)
            - matrix[0][1] * (matrix[0][1] * matrix[2][2] - matrix[1][2] * matrix[0][2])
            + matrix[0][2] * (matrix[0][1] * matrix[1][2] - matrix[1][1] * matrix[0][2])
        )
        return (
            all(matrix[i][i] >= 0 for i in range(3))
            and all(
                matrix[i][i] * matrix[j][j] - matrix[i][j] ** 2 >= 0
                for i in range(3)
                for j in range(i + 1, 3)
            )
            and determinant >= 0
        )

    lower, upper = 4.0, 5.0
    assert not upper_bound(Fraction(lower)) and upper_bound(Fraction(upper))
    while np.nextafter(lower, np.inf) < upper:
        middle = (lower + upper) / 2.0
        if upper_bound(Fraction(middle)):
            upper = middle
        else:
            lower = middle
    assert not upper_bound(Fraction(lower)) and upper_bound(Fraction(upper))
    old_distance = float(np.nextafter(lower * error, np.inf))
    assert not upper_bound(Fraction(old_distance) / Fraction(error))
    original_norm = np.linalg.norm
    monkeypatch.setattr(
        np.linalg,
        "norm",
        lambda values, *args, **kwargs: (
            lower
            if kwargs.get("ord") == 2 and values.shape == (3, 3)
            else original_norm(values, *args, **kwargs)
        ),
    )
    distance = surface.lattice_metrics().vertex_root_distance
    _assert_stretch_certificate(surface, vectors, distance)


def test_lattice_metrics_tight_axis_aligned_stretch() -> None:
    surface = _discover("schwarz_p", 0.2, np.diag([2.0, 1.0, 0.5]))
    distance = surface.lattice_metrics().vertex_root_distance
    assert Fraction(distance) >= 2 * Fraction(surface.root_error)
    assert Fraction(distance) <= 2 * Fraction(surface.root_error) * (
        1 + Fraction(1, 2**48)
    )


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

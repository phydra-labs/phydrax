import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing._audit_topology import (
    _Complex,
    _host_topology_counts,
    _native_topology_counts,
    _supplemented_star_complex,
    _TopologyCounts,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="the native welded-topology route requires meshcore"
)

# Mixed blocks expose facet keys shared by different face shapes, whose links
# both routes refuse; single kinds keep the star checks decided.
_KINDS = (
    (1, ("interval",)),
    (2, ("triangle",)),
    (2, ("quadrilateral",)),
    (2, ("triangle", "quadrilateral")),
    (3, ("tetrahedron",)),
    (3, ("hexahedron",)),
    (3, ("prism",)),
    (3, ("tetrahedron", "hexahedron", "prism")),
)
_ARITY = {
    "interval": 2,
    "triangle": 3,
    "quadrilateral": 4,
    "tetrahedron": 4,
    "hexahedron": 8,
    "prism": 6,
}


def _empty() -> _Complex:
    return _Complex(
        np.empty((0, 1), dtype=np.int64),
        np.empty((0, 1), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int8),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )


def _blocks(
    generator: np.random.Generator, kinds: tuple[str, ...], vertices: int, cells: int
) -> tuple[tuple[str, np.ndarray], ...]:
    blocks = []
    for kind in kinds:
        count = int(generator.integers(1, cells))
        rows = generator.integers(0, vertices, size=(count, _ARITY[kind]))
        # Adjacent repeats, reversed copies and exact duplicates exercise
        # collapsed, inconsistent and duplicated facets.
        repeats = generator.integers(0, count, size=count // 3)
        rows = np.concatenate((rows, rows[repeats], rows[repeats][:, ::-1]))
        blocks.append((kind, rows.astype(np.int64)))
    return tuple(blocks)


def _outcome(
    route: Callable[..., _TopologyCounts],
    arguments: tuple[object, ...],
    options: dict[str, bool],
) -> tuple[object, ...]:
    try:
        counts = route(*arguments, **options)
    except ValueError as error:
        # Both routes refuse facet links whose incidences no cell lists.
        assert str(error).endswith("Welded star links reference unknown incidences.")
        return ("refused",)
    fields = dataclasses.astuple(counts)
    return (*fields[:-1], counts.boundary.tolist())


@pytest.mark.parametrize(("dimension", "kinds"), _KINDS)
@pytest.mark.parametrize("seed", range(8))
def test_native_welded_counts_match_host_on_random_complexes(
    dimension: int, kinds: tuple[str, ...], seed: int
) -> None:
    generator = np.random.default_rng(seed)
    vertices = int(generator.integers(4, 14))
    complex_ = _supplemented_star_complex(
        _empty(), _blocks(generator, kinds, vertices, 12), dimension
    )
    supplement = _blocks(generator, kinds, vertices + 4, 6)
    for star in (complex_, _supplemented_star_complex(complex_, supplement, dimension)):
        for scope in (None, np.unique(generator.integers(0, vertices, size=3))):
            for embedded, manifold in ((False, dimension >= 2), (True, False)):
                arguments = (complex_, star, scope, dimension)
                options = {
                    "embedded": embedded,
                    "check_manifold": manifold,
                    "check_watertight": True,
                }
                assert _outcome(_native_topology_counts, arguments, options) == (
                    _outcome(_host_topology_counts, arguments, options)
                )


def test_padded_keys_group_beside_a_face_welded_to_key_zero() -> None:
    # A quadrilateral face welded entirely onto key 0 sorts beside padded
    # triangle keys; both routes must still group equal keys.
    rows = np.asarray(
        ((0, 0, 1, 0, 0, 2), (3, 4, 5, 6, 7, 8), (3, 5, 4, 6, 8, 7)), dtype=np.int64
    )
    complex_ = _supplemented_star_complex(_empty(), (("prism", rows),), 3)
    options = {"embedded": True, "check_manifold": False, "check_watertight": True}
    arguments = (complex_, complex_, None, 3)
    native = _outcome(_native_topology_counts, arguments, options)
    assert native == _outcome(_host_topology_counts, arguments, options)
    assert native[4] == 3


def test_links_beyond_every_incidence_are_refused_by_both_routes() -> None:
    rows = np.asarray(
        ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 8, 10, 9, 11)), dtype=np.int64
    )
    complex_ = _supplemented_star_complex(_empty(), (("hexahedron", rows),), 3)
    options = {"embedded": False, "check_manifold": True, "check_watertight": True}
    arguments = (complex_, complex_, None, 3)
    assert _outcome(_native_topology_counts, arguments, options) == ("refused",)
    assert _outcome(_host_topology_counts, arguments, options) == ("refused",)


def _kuhn_cube(size: int) -> phx.discretization.CellMesh:
    grid = np.arange(size + 1, dtype=np.float64) / size
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    i, j, k = (
        axis.ravel() for axis in np.meshgrid(*(np.arange(size),) * 3, indexing="ij")
    )
    corners = [
        ((i + a) * (size + 1) + (j + b)) * (size + 1) + (k + c)
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]
    cells = np.concatenate(
        [
            np.stack((corners[0], corners[p], corners[q], corners[7]), axis=1)
            for p, q in ((4, 6), (4, 5), (2, 6), (2, 3), (1, 5), (1, 3))
        ]
    )
    a, b, c, d = (points[cells[:, index]] for index in range(4))
    negative = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0.0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "tetrahedron", cells),)
    )


def _pinched_mesh() -> phx.discretization.CellMesh:
    # Two tetrahedra sharing one vertex through coincident vertex rows, and a
    # third one stacked on the first by a duplicated vertex set.
    points = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (0, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
            (1, 0, 0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3), (4, 6, 5, 7), (4, 8, 2, 3)), dtype=np.int32)
    return phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "tetrahedron", cells),)
    )


@pytest.mark.parametrize("mesh_factory", (lambda: _kuhn_cube(3), _pinched_mesh))
def test_welded_audit_report_is_route_independent(
    mesh_factory: Callable[[], phx.discretization.CellMesh],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._audit_topology as topology

    mesh = mesh_factory()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    native = phx.meshing.audit_cell_mesh(mesh, geometry)
    monkeypatch.setattr(topology, "meshcore_available", lambda: False)
    host = phx.meshing.audit_cell_mesh(mesh, geometry)
    assert native.report_id == host.report_id
    assert native.check_counts == host.check_counts

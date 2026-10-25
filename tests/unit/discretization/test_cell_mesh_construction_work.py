import numpy as np
import pytest

from phydrax.discretization._cell_mesh import (
    cell_block_construction_bounds,
    cell_block_construction_work,
    cell_mesh_construction_bounds,
    cell_mesh_construction_work,
    cell_mesh_reuse_work,
    CellBlock,
    CellMesh,
)


_ROUTES = [
    (2, 1, [("interval", 4, 2)]),
    (3, 2, [("triangle", 4, 3), ("quadrilateral", 3, 4)]),
    (4, 2, [("polygon", 3, 5), ("polygon", 2, 5)]),
    (5, 3, [("tetrahedron", 6, 4)]),
    (8, 3, [("hexahedron", 5, 8)]),
    (9, 3, [("prism", 2, 6), ("pyramid", 3, 5)]),
    (6, 4, [("simplex:4", 3, 5)]),
]


@pytest.mark.parametrize(("vertices", "ambient", "blocks"), _ROUTES)
def test_work_is_monotone_in_counts(
    vertices: int, ambient: int, blocks: list[tuple[str, int, int]]
) -> None:
    base = cell_mesh_construction_work(vertices, ambient, blocks)
    assert base > 0
    grown = [(kind, 2 * count, arity) for kind, count, arity in blocks]
    assert cell_mesh_construction_work(2 * vertices, ambient, grown) > base


def test_block_work_is_monotone() -> None:
    assert cell_block_construction_work(4, 3) < cell_block_construction_work(8, 3)
    assert cell_block_construction_work(4, 3) < cell_block_construction_work(4, 6)


_HEX = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]]
_REALIZED = [
    (2, "interval", [[0, 1], [1, 2], [2, 3]], 4),
    (2, "triangle", [[0, 1, 2], [0, 2, 3], [1, 4, 2]], 5),
    (2, "simplex:2", [[0, 1, 2], [0, 2, 3], [1, 4, 2]], 5),
    (2, "quadrilateral", [[0, 1, 4, 3], [1, 2, 5, 4]], 6),
    (3, "tetrahedron", [[0, 1, 2, 3], [1, 2, 3, 4]], 5),
    (3, "simplex:3", [[0, 1, 2, 3], [1, 2, 3, 4]], 5),
    (3, "hexahedron", _HEX, 12),
    (3, "prism", [[0, 1, 2, 3, 4, 5], [1, 6, 2, 4, 7, 5]], 8),
]


@pytest.mark.parametrize(("ambient", "kind", "rows", "vertices"), _REALIZED)
def test_realized_mesh_never_exceeds_a_priori_work(
    ambient: int, kind: str, rows: list[list[int]], vertices: int
) -> None:
    rng = np.random.default_rng(0)
    coordinates = rng.standard_normal((vertices, ambient)).astype(np.float64)
    cells = np.asarray(rows, dtype=np.int32)
    mesh = CellMesh(coordinates, (CellBlock("cells", kind, cells),))
    blocks = [(kind, cells.shape[0], cells.shape[1])]
    a_priori = cell_mesh_construction_work(vertices, ambient, blocks)
    assert cell_mesh_construction_work(vertices, ambient, blocks, realized=mesh) <= (
        a_priori
    )
    assert cell_mesh_reuse_work(vertices, ambient, blocks) < a_priori


@pytest.mark.parametrize(
    ("vertices", "ambient", "blocks"),
    [
        (0, 2, [("triangle", 1, 3)]),
        (3, 2, []),
        (3, 3, [("polyhedron", 1, 4)]),
        (3, 2, [("triangle", 1, 4)]),
        (4, 3, [("triangle", 1, 3), ("tetrahedron", 1, 4)]),
        (4, 3, [("simplex:2", 1, 3), ("quadrilateral", 1, 4)]),
    ],
)
def test_refusals_mirror_byte_bounds(
    vertices: int, ambient: int, blocks: list[tuple[str, int, int]]
) -> None:
    with pytest.raises(ValueError) as byte_error:
        cell_mesh_construction_bounds(vertices, ambient, blocks)
    with pytest.raises(ValueError) as work_error:
        cell_mesh_construction_work(vertices, ambient, blocks)
    assert str(work_error.value) == str(byte_error.value).replace(
        "cell_mesh_construction_bounds", "cell_mesh_construction_work"
    )


def test_block_refusal_mirrors_byte_bound() -> None:
    with pytest.raises(ValueError, match="positive cell count"):
        cell_block_construction_bounds(0, 3)
    with pytest.raises(ValueError, match="positive cell count"):
        cell_block_construction_work(0, 3)

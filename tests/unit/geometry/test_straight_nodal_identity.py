from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh, reference_cell_topology
from phydrax.meshing._curving import _straight_geometry


@pytest.mark.parametrize(
    ("kind", "degree", "expected"),
    (
        (
            "triangle",
            2,
            "b764e531842f0af289a6b658dca9d67256b3c34b22a571e37f06e18da3f12ffe",
        ),
        (
            "triangle",
            3,
            "e8517c412af3f3d34dd4cf154a8674e4386509a6ac6560004ecc198455245695",
        ),
        (
            "quadrilateral",
            2,
            "5325bf4185090ae3f69cbe23aa017a356969842f1241848c5ed357ee5d2f6c37",
        ),
        (
            "quadrilateral",
            3,
            "4e77c72db6eb702ebe9d8e18a216841803482d576541581408b9af74b15c0577",
        ),
        (
            "tetrahedron",
            2,
            "21d14c5d662e53021fdd11be56f4b9eda31cdc8fc17327a61b9efaaafa029426",
        ),
        (
            "tetrahedron",
            3,
            "afb6473ce3d79914930f686b12c86e99b1f9458596276835daa732843c1146a3",
        ),
        ("prism", 2, "a46ff4ffeeaaf6de59690b5758ca25788df41c4e8918e4d7820528dd3fd72117"),
        ("prism", 3, "c41b0476b29b8d20bed26f763af6cb94387cc93873afa19644cc634471695680"),
        (
            "pyramid",
            2,
            "ea6f45539acf5163d9707169a2f87455a732ff6ee0805efa84100cc9bf945402",
        ),
        (
            "pyramid",
            3,
            "43f81e1f319c4f44fd11aba8f163b7b18fc31c6e4ff9fb9d7849da1a2d1a3e1a",
        ),
        (
            "hexahedron",
            2,
            "4b9b758244b4bd1cd79914db660b123f22cb6483c60a033fba9e26d62006a978",
        ),
        (
            "hexahedron",
            3,
            "db91a4ce9d6cb2484aa6b472763893fa30a0ac097ee4d32c1b879df677bcc916",
        ),
    ),
)
def test_straight_nodal_source_identity(kind: str, degree: int, expected: str) -> None:
    points = np.asarray(
        reference_cell_topology(kind).vertices, dtype=np.float64
    ) * np.float64(1.25)
    block = CellBlock(
        'a"é\\z',
        kind,
        np.arange(len(points), dtype=np.int64)[None, :],
        global_ids=np.asarray((43,), dtype=np.int64),
    )
    mesh = CellMesh(points, (block,))

    assert _straight_geometry(mesh, degree).source_map_id == expected

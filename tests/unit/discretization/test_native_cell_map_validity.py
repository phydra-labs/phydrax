"""The native root proof of P1 tetrahedra, Q1 hexahedra and P1 prisms.

``phx_mc_cell_map_validity`` decides the per-cell proof of
``certify_cell_geometry_validity`` where its root proof suffices; the reference
is the same certificate computed with every cell on the Python proof. Statuses,
directed endpoints, depths, map digests and certificate identities must agree
bit for bit, including inverted, degenerate and sub-floor cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import phydrax._meshcore as meshcore_module
import phydrax.discretization._cell_geometry_validity as validity_module
from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.geometry._mesh_certificates import _same_tree_content


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="phydrax-meshcore unavailable"
)

_CUBE = np.asarray(
    [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=np.float64
)
_HEX = np.asarray((0, 1, 3, 2, 4, 5, 7, 6), dtype=np.int32)


def _tetrahedra() -> CellMesh:
    # Disjoint cells: random, inverted, flat, coincident vertices, tiny and
    # huge scales, exact zeros and a wide exponent spread inside one cell.
    generator = np.random.default_rng(11)
    cells = [generator.normal(size=(4, 3)) for _ in range(24)]
    cells.append(_CUBE[[0, 1, 2, 4]])
    cells.append(_CUBE[[0, 2, 1, 4]])
    cells.append(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float))
    cells.append(np.asarray([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float))
    cells.append(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.3, 2.0**-30]]))
    cells.extend(generator.normal(size=(4, 3)) * scale for scale in (2.0**-60, 1.0e8))
    spread = generator.normal(size=(4, 3))
    spread[0] *= 2.0**-200
    cells.append(spread)
    points = np.concatenate(cells)
    return CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.arange(points.shape[0], dtype=np.int32).reshape(-1, 4),
            ),
        ),
    )


def _hexahedra() -> CellMesh:
    # Perturbed (multilinear) boxes, an exactly affine one, an inverted one, a
    # twisted one needing subdivision and badly distorted ones.
    generator = np.random.default_rng(5)
    cells = [_CUBE + generator.normal(size=(8, 3)) * 0.08 for _ in range(16)]
    cells.extend(_CUBE + generator.normal(size=(8, 3)) * 0.4 for _ in range(8))
    cells.append(_CUBE * np.asarray((2.0, 0.5, 0.25)))
    inverted = _CUBE.copy()
    inverted[:, 0] = 1.0 - inverted[:, 0]
    cells.append(inverted)
    twisted = _CUBE.copy()
    angle = 2.5 * twisted[:, 2]
    x, y = twisted[:, 0] - 0.5, twisted[:, 1] - 0.5
    twisted[:, 0] = x * np.cos(angle) - y * np.sin(angle) + 0.5
    twisted[:, 1] = x * np.sin(angle) + y * np.cos(angle) + 0.5
    cells.append(twisted)
    points = np.concatenate(cells)
    routes = (np.arange(len(cells), dtype=np.int32)[:, None] * 8 + _HEX).astype(np.int32)
    return CellMesh(points, (CellBlock("cells", "hexahedron", routes),))


def _python_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(meshcore_module, "meshcore_available", lambda: False)


_WEDGE = np.asarray(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64
)


def _prisms() -> CellMesh:
    # Perturbed (nonaffine) wedges, thin layer-like wedges over a curved
    # wall, an exact translate (affine), an inverted, a collapsed and a
    # twisted wedge needing subdivision.
    generator = np.random.default_rng(7)
    cells = [_WEDGE + generator.normal(size=(6, 3)) * 0.08 for _ in range(16)]
    cells.extend(_WEDGE + generator.normal(size=(6, 3)) * 0.35 for _ in range(8))
    for _ in range(8):
        bottom = generator.normal(size=(3, 3))
        bottom /= np.linalg.norm(bottom, axis=1, keepdims=True)
        if np.linalg.det(bottom) < 0.0:
            bottom = bottom[::-1]
        cells.append(np.concatenate((bottom, bottom * (1.0 - 0.0123))))
    cells.append(_WEDGE * np.asarray((2.0, 0.5, 0.25)) + np.asarray((0.1, 0.0, 0.3)))
    inverted = _WEDGE.copy()
    inverted[:, 2] = 1.0 - inverted[:, 2]
    cells.append(inverted)
    collapsed = _WEDGE.copy()
    collapsed[3] = collapsed[0]
    cells.append(collapsed)
    twisted = _WEDGE.copy()
    angle = 2.4 * twisted[:, 2]
    x, y = twisted[:, 0] - 1.0 / 3.0, twisted[:, 1] - 1.0 / 3.0
    twisted[:, 0] = x * np.cos(angle) - y * np.sin(angle) + 1.0 / 3.0
    twisted[:, 1] = x * np.sin(angle) + y * np.cos(angle) + 1.0 / 3.0
    cells.append(twisted)
    points = np.concatenate(cells)
    routes = np.arange(points.shape[0], dtype=np.int32).reshape(-1, 6)
    return CellMesh(points, (CellBlock("cells", "prism", routes),))


def _decided(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    decided = [0]
    original = validity_module._native_cell_map_proofs

    def counted(*args: Any) -> Any:
        result = original(*args)
        decided[0] += int(np.count_nonzero(result[0]))
        return result

    monkeypatch.setattr(validity_module, "_native_cell_map_proofs", counted)
    return decided


def _assert_same(
    native: CellValidityCertificate, rational: CellValidityCertificate
) -> None:
    assert native.certificate_id == rational.certificate_id
    assert native.cell_map_digests == rational.cell_map_digests
    assert _same_tree_content(native, rational)


@pytest.mark.parametrize("build", (_tetrahedra, _hexahedra, _prisms))
@pytest.mark.parametrize("floor", (None, 0.3))
def test_native_root_proof_reproduces_the_rational_certificate(
    monkeypatch: pytest.MonkeyPatch, build: Any, floor: float | None
) -> None:
    mesh = build()
    policy = (
        None if floor is None else CellValidityPolicy(relative_determinant_floor=floor)
    )
    with monkeypatch.context() as patch:
        _python_only(patch)
        rational = certify_cell_geometry_validity(mesh, policy=policy)
    decided = _decided(monkeypatch)
    native = certify_cell_geometry_validity(mesh, policy=policy)

    _assert_same(native, rational)
    assert decided[0] > 0
    status = np.asarray(native.status)
    assert np.any(status == CellValidityStatus.CERTIFIED_VALID)
    assert np.any(status == CellValidityStatus.INVALID)


def test_inverted_and_degenerate_cells_are_refused_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _tetrahedra()
    decided = _decided(monkeypatch)
    certificate = certify_cell_geometry_validity(mesh)
    status = np.asarray(certificate.status)
    # The inverted unit cell, the flat cell and the coincident-vertex cell.
    assert status[25] == status[26] == status[27] == CellValidityStatus.INVALID
    assert status[24] == CellValidityStatus.CERTIFIED_VALID
    # Every cell but the 2**-200 exponent spread (beyond the native exact
    # width, so the Python proof decides it) is decided natively.
    assert decided[0] == status.size - 1


def test_the_python_proof_takes_every_undecided_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A twisted hexahedron needs subdivision; the native root proof leaves it.
    mesh = _hexahedra()
    decided = _decided(monkeypatch)
    certificate = certify_cell_geometry_validity(mesh)
    depth = np.asarray(certificate.depth)
    assert depth[-1] > 0
    assert decided[0] < depth.size

"""Prove-once reuse of exact surface restriction proofs and spline root bases."""

from typing import Any

import jax
import numpy as np
import pytest

import phydrax.discretization._cell_geometry as cell_geometry
import phydrax.geometry._surface_source_support as surface_support
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import (
    coordinate_source_basis,
    elevate_spline_extraction,
    SplineCellGeometryElement,
)
from phydrax.discretization._coordinate_enclosure import expression_equal
from phydrax.geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    SurfaceNativeRestrictionBoundarySource,
)
from phydrax.meshing.providers import NativeMeshingProvider
from tools.meshing_qualification.scenarios.native_family_matrix import (
    _CONTRACT,
    _quarter_extrusion,
    _quarter_surface_case,
)


def _same_rows(
    first: surface_support.SurfaceRestrictionRows,
    second: surface_support.SurfaceRestrictionRows,
) -> bool:
    left, right = first._asdict(), second._asdict()
    return all(
        np.array_equal(left[name], right[name])
        if isinstance(left[name], np.ndarray)
        else left[name] == right[name]
        for name in left
    )


def test_restriction_proof_is_reused_only_for_its_exact_live_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _quarter_surface_case(4)
    result = (
        NativeMeshingProvider(case.options)
        .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    assert result.certification is not None
    source = result.certification.inputs.source
    assert isinstance(source, SurfaceNativeRestrictionBoundarySource)
    support, mesh, geometry = source.support, source.mesh, source.geometry
    proof_maps: list[None] = []
    original = surface_support.coordinate_map_expressions

    def counted(*args: Any, **kwargs: Any) -> Any:
        proof_maps.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(surface_support, "coordinate_map_expressions", counted)
    # The source proved these live operands at construction.
    proved = support.prove_native_restrictions(mesh, geometry)
    source.require_current()
    assert not proof_maps
    # Callers own their rows; mutating them never reaches the proof record.
    proved.cell_ids[0] = -1
    again = support.prove_native_restrictions(mesh, geometry)
    assert int(again.cell_ids[0]) != -1
    # A content-equal rebuild is a different live object and is proved in full.
    rebuilt = jax.tree_util.tree_map(lambda leaf: leaf, mesh)
    assert rebuilt is not mesh
    replayed = support.prove_native_restrictions(rebuilt, geometry)
    assert proof_maps
    assert _same_rows(replayed, again)
    count = len(proof_maps)
    support.prove_native_restrictions(rebuilt, geometry)
    assert len(proof_maps) == count
    # A refused proof is never recorded: it refuses again on every call.
    for _ in range(2):
        with pytest.raises(ValueError, match="chart witness"):
            support.prove_native_restrictions(
                mesh,
                geometry,
                chart_witness=object(),  # ty: ignore[invalid-argument-type]
            )


def test_spline_source_basis_is_bounded_and_keyed_by_exact_content() -> None:
    domain = _quarter_extrusion()
    atlas = prepare_surface_source_root_atlas(
        domain, (0,), SpatialCoordinateContract.si()
    )
    elements, _, _ = atlas.geometry.resolve(atlas.mesh)
    root = elements[0]
    assert isinstance(root, SplineCellGeometryElement)
    cell_geometry._SPLINE_SOURCE_BASES.clear()
    fresh = cell_geometry._root_source_basis(root)
    first = coordinate_source_basis(root)
    copy = jax.tree_util.tree_map(lambda leaf: leaf, root)
    second = coordinate_source_basis(copy)
    assert len(cell_geometry._SPLINE_SOURCE_BASES) == 1
    assert all(
        expression_equal(a, b) and expression_equal(a, c)
        for a, b, c in zip(fresh, first, second, strict=True)
    )
    elevated = elevate_spline_extraction(root, (3, 3))
    coordinate_source_basis(elevated)
    assert len(cell_geometry._SPLINE_SOURCE_BASES) == 2
    limit = cell_geometry._SPLINE_SOURCE_BASIS_LIMIT
    cell_geometry._SPLINE_SOURCE_BASES.clear()
    for index in range(limit):
        cell_geometry._SPLINE_SOURCE_BASES[("filler", index)] = ()
    coordinate_source_basis(root)
    assert len(cell_geometry._SPLINE_SOURCE_BASES) == limit
    assert ("filler", 0) not in cell_geometry._SPLINE_SOURCE_BASES
    cell_geometry._SPLINE_SOURCE_BASES.clear()

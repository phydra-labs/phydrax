#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


gdstk = pytest.importorskip("gdstk")

from phydrax import SpatialCoordinateContract
from phydrax.interchange import (
    AdapterStatus,
    decode_layout_bytes,
    LayoutAdapterError,
    LayoutFormat,
    LayoutImportPolicy,
    LayoutLayerKey,
    ResourceLimits,
)
from phydrax.units import MICROMETER, MILLIMETER


_LIMITS = ResourceLimits(
    max_bytes=1_000_000,
    max_depth=8,
    max_nodes=1_000,
    max_attributes=100,
    max_losses=100,
)


def _write(library, path: Path, format: LayoutFormat) -> bytes:
    if format is LayoutFormat.GDSII:
        library.write_gds(path)
    else:
        library.write_oas(path)
    return path.read_bytes()


def _policy(
    format: LayoutFormat,
    *,
    unit=MICROMETER,
    top_cell: str | None = None,
    path_tolerance: float | None = None,
    waived_loss_paths: tuple[str, ...] = (),
    limits: ResourceLimits = _LIMITS,
) -> LayoutImportPolicy:
    return LayoutImportPolicy(
        format,
        SpatialCoordinateContract(unit),
        limits,
        top_cell=top_cell,
        path_tolerance=path_tolerance,
        waived_loss_paths=waived_loss_paths,
    )


def test_gdsii_and_oasis_keep_distinct_provenance_for_equal_physical_regions(
    tmp_path: Path,
):
    child = gdstk.Cell("CHILD")
    child.add(gdstk.rectangle((0, 0), (2, 1), layer=4, datatype=7))
    top = gdstk.Cell("TOP")
    top.add(
        gdstk.Reference(
            child,
            origin=(10, 0),
            columns=2,
            rows=1,
            spacing=(3, 0),
        )
    )
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    library.add(top, child)

    gds = decode_layout_bytes(
        _write(library, tmp_path / "layout.gds", LayoutFormat.GDSII),
        _policy(LayoutFormat.GDSII),
    )
    oasis = decode_layout_bytes(
        _write(library, tmp_path / "layout.oas", LayoutFormat.OASIS),
        _policy(LayoutFormat.OASIS),
    )

    assert gds.model == oasis.model
    assert gds.model.regions[0] == oasis.model.regions[0]
    assert gds.source_digest != oasis.source_digest
    assert gds.model.regions[0].provenance_id != oasis.model.regions[0].provenance_id
    assert [region.layer for region in gds.model.regions] == [
        LayoutLayerKey(4, 7),
        LayoutLayerKey(4, 7),
    ]
    assert gds.model.regions[0].occurrence_path == (
        "TOP",
        "reference:0:CHILD:rep:0",
        "polygon:0:rep:0",
    )
    minima = [
        np.asarray(region.geometry.vertices).min(axis=0) for region in gds.model.regions
    ]
    assert np.allclose(minima, ((10, 0), (13, 0)))


def test_source_user_units_are_explicitly_converted_to_target_contract(tmp_path: Path):
    top = gdstk.Cell("TOP")
    top.add(gdstk.rectangle((0, 0), (1000, 500), layer=1, datatype=2))
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    library.add(top)
    data = _write(library, tmp_path / "units.gds", LayoutFormat.GDSII)

    result = decode_layout_bytes(
        data,
        _policy(LayoutFormat.GDSII, unit=MILLIMETER),
    )

    vertices = np.asarray(result.model.regions[0].geometry.vertices)
    assert np.allclose(vertices.min(axis=0), (0, 0))
    assert np.allclose(vertices.max(axis=0), (1, 0.5))
    assert result.source_user_unit_meters == pytest.approx(1e-6)
    assert result.source_database_unit_meters == pytest.approx(1e-9)
    assert result.source_digest == result.resource_manifest.content_sha256
    assert result.report.coordinate_mapping[0].startswith("source user unit")


def test_ambiguous_top_cell_requires_explicit_selection(tmp_path: Path):
    first = gdstk.Cell("FIRST")
    first.add(gdstk.rectangle((0, 0), (1, 1)))
    second = gdstk.Cell("SECOND")
    second.add(gdstk.rectangle((2, 0), (3, 1)))
    library = gdstk.Library()
    library.add(first, second)
    data = _write(library, tmp_path / "ambiguous.gds", LayoutFormat.GDSII)

    with pytest.raises(LayoutAdapterError) as raised:
        decode_layout_bytes(data, _policy(LayoutFormat.GDSII))
    assert raised.value.status is AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC

    selected = decode_layout_bytes(
        data,
        _policy(LayoutFormat.GDSII, top_cell="SECOND"),
    )
    assert selected.top_cell == "SECOND"
    assert len(selected.model.regions) == 1


def test_supported_path_requires_tolerance_and_an_explicit_loss_waiver(tmp_path: Path):
    top = gdstk.Cell("TOP")
    top.add(
        gdstk.FlexPath(
            ((0, 0), (1, 0), (1, 1)),
            0.2,
            layer=9,
            datatype=3,
            simple_path=True,
        )
    )
    library = gdstk.Library()
    library.add(top)
    data = _write(library, tmp_path / "path.gds", LayoutFormat.GDSII)

    with pytest.raises(LayoutAdapterError) as missing_tolerance:
        decode_layout_bytes(data, _policy(LayoutFormat.GDSII))
    assert missing_tolerance.value.status is AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC

    with pytest.raises(LayoutAdapterError) as unwaived:
        decode_layout_bytes(
            data,
            _policy(LayoutFormat.GDSII, path_tolerance=1e-3),
        )
    assert unwaived.value.report is not None
    assert [loss.path for loss in unwaived.value.report.losses] == ["TOP/path:0"]

    result = decode_layout_bytes(
        data,
        _policy(
            LayoutFormat.GDSII,
            path_tolerance=1e-3,
            waived_loss_paths=("TOP/path:0",),
        ),
    )
    assert result.report.status is AdapterStatus.DECLARED_LOSS
    assert result.report.valid
    assert result.model.regions[0].layer == LayoutLayerKey(9, 3)


def test_hierarchy_cycles_and_repetition_resource_excess_fail_closed(tmp_path: Path):
    first = gdstk.Cell("FIRST")
    second = gdstk.Cell("SECOND")
    first.add(gdstk.Reference(second))
    second.add(gdstk.Reference(first))
    cyclic = gdstk.Library()
    cyclic.add(first, second)
    cycle_data = _write(cyclic, tmp_path / "cycle.gds", LayoutFormat.GDSII)

    with pytest.raises(LayoutAdapterError) as cycle:
        decode_layout_bytes(
            cycle_data,
            _policy(LayoutFormat.GDSII, top_cell="FIRST"),
        )
    assert cycle.value.status is AdapterStatus.MALFORMED_SOURCE

    child = gdstk.Cell("CHILD")
    child.add(gdstk.rectangle((0, 0), (1, 1)))
    top = gdstk.Cell("TOP")
    top.add(gdstk.Reference(child, columns=50, rows=1, spacing=(2, 0)))
    repeated = gdstk.Library()
    repeated.add(top, child)
    repeated_data = _write(repeated, tmp_path / "repeated.gds", LayoutFormat.GDSII)
    small_limits = ResourceLimits(1_000_000, 8, 20, 100, 100)

    with pytest.raises(LayoutAdapterError) as excessive:
        decode_layout_bytes(
            repeated_data,
            _policy(LayoutFormat.GDSII, limits=small_limits),
        )
    assert excessive.value.status is AdapterStatus.MALFORMED_SOURCE


def test_missing_optional_dependency_is_a_typed_adapter_failure(monkeypatch):
    import phydrax.interchange._layout as layout_module

    def missing(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(layout_module, "import_module", missing)
    with pytest.raises(LayoutAdapterError) as raised:
        decode_layout_bytes(b"not parsed", _policy(LayoutFormat.GDSII))
    assert raised.value.status is AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE
    assert raised.value.resource_manifest is not None

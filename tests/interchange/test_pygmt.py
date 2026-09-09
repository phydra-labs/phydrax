#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import importlib
import json
import os

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.interchange._geospatial import GeospatialContract, QualifiedGeospatialGrid
from phydrax.interchange._pygmt import (
    export_geospatial_grid,
    GeospatialDependencyError,
    render_geospatial_grid,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import METER


def _asymmetric_grid(*, registration="pixel"):
    contract = GeospatialContract.local_cartesian(
        SpatialCoordinateContract(METER, reference_frame="synthetic-survey"),
        vertical_datum="synthetic-benchmark",
        registration=registration,
        mask_semantics="valid_true",
    )
    return QualifiedGeospatialGrid(
        [14.0, 12.0, 10.0],
        [8.0, 5.0],
        [[19.0, 12.0, 11.0], [45.0, 38.0, 31.0]],
        contract,
        value_unit=METER,
        valid=[[True, False, True], [True, True, True]],
        value_role="vertical",
    )


def test_xarray_export_preserves_sample_locations_masked_payload_and_metadata():
    pytest.importorskip("xarray")
    grid = _asymmetric_grid()
    exported = export_geospatial_grid(grid)
    data = exported.dataarray
    assert exported.report.valid
    assert exported.report.status == AdapterStatus.LOSSLESS
    assert data.sel(x=10.0, y=5.0).item() == 31.0
    assert data.sel(x=14.0, y=8.0).item() == 19.0
    assert data.sel(x=12.0, y=8.0).item() == 12.0
    assert not data.coords["valid"].sel(x=12.0, y=8.0).item()
    np.testing.assert_array_equal(data.x, grid.x)
    np.testing.assert_array_equal(data.y, grid.y)
    assert (
        json.loads(data.attrs["phydrax_geospatial_contract"])["vertical_datum"]
        == "synthetic-benchmark"
    )
    assert tuple(data.attrs["region"]) == (9.0, 15.0, 3.5, 9.5)
    data.loc[dict(x=10.0, y=5.0)] = 999.0
    assert grid.values[1, 2] == 31.0


def test_missing_optional_package_fails_without_creating_a_map(tmp_path, monkeypatch):
    original = importlib.import_module

    def missing_xarray(name, package=None):
        if name == "xarray":
            raise ModuleNotFoundError("xarray is unavailable", name="xarray")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_xarray)
    output = tmp_path / "unavailable.png"
    with pytest.raises(GeospatialDependencyError) as failure:
        render_geospatial_grid(_asymmetric_grid(), output)
    assert not output.exists()
    assert not failure.value.report.valid
    assert failure.value.report.status == AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE


def test_rendering_boundary_rejects_remote_inputs_and_cartesian_map_reprojection(
    tmp_path,
):
    grid = _asymmetric_grid()
    with pytest.raises(ValueError):
        render_geospatial_grid(grid, tmp_path / "remote.png", cmap="@remote_palette")
    with pytest.raises(ValueError):
        render_geospatial_grid(grid, tmp_path / "reprojected.png", projection="M12c")
    assert not (tmp_path / "remote.png").exists()
    assert not (tmp_path / "reprojected.png").exists()


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_TEST_PYGMT") != "1",
    reason="explicit real PyGMT/GMT/Ghostscript qualification is opt-in",
)
def test_real_gmt_observes_registration_orientation_and_rendered_resource(tmp_path):
    pygmt = pytest.importorskip("pygmt")
    for registration, expected_bounds, expected_registration in (
        ("pixel", [9.0, 15.0, 3.5, 9.5], 1),
        ("gridline", [10.0, 14.0, 5.0, 8.0], 0),
    ):
        grid = _asymmetric_grid(registration=registration)
        exported = export_geospatial_grid(grid, for_pygmt=True)
        # Observe the actual GMT consumer, not only xarray accessor fields.
        info = np.fromstring(pygmt.grdinfo(exported.dataarray, per_column="n"), sep=" ")
        np.testing.assert_allclose(info[:4], expected_bounds)
        np.testing.assert_allclose(info[6:10], [2, 3, 3, 2])
        assert int(info[-2]) == expected_registration
        assert int(info[-1]) == 0
        assert exported.dataarray.sel(x=10, y=5).item() == 31
        assert exported.dataarray.sel(x=14, y=8).item() == 19
        assert np.isnan(exported.dataarray.sel(x=12, y=8).item())
        result = render_geospatial_grid(
            grid, tmp_path / (registration + ".png"), frame=False
        )
        payload = (tmp_path / (registration + ".png")).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(payload).hexdigest() == result.manifest.content_sha256
        assert result.report.valid
        assert result.provenance["remote_download"] is False
        assert result.provenance["contract"]["registration"] == registration

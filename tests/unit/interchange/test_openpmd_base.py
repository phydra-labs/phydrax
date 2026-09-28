#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits, ResourceReadError
from phydrax.interchange._openpmd_base import (
    ELECTRIC_FIELD_DIMENSION,
    LENGTH_DIMENSION,
    OPENPMD_LASER_ENVELOPE_REVISION,
    OPENPMD_MESH_REVISION,
    OPENPMD_PARTICLE_REVISION,
    OpenPMDIterationTime,
    OpenPMDUnit,
    OpenPMDUnsupportedError,
    preflight_hdf5,
    read_grid_unit_si,
    read_iteration,
    read_record_unit,
    read_series_root,
    TIME_DIMENSION,
    write_iteration,
    write_record_unit,
    write_series_root,
)


# Exact SI values (2019 SI redefinition): elementary charge and speed of light.
_ELEMENTARY_CHARGE = 1.602176634e-19
_SPEED_OF_LIGHT = 299792458.0
_CHARGE_DIMENSION = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def _limits(
    *, max_depth: int = 12, max_nodes: int = 10_000, max_attributes: int = 256
) -> ResourceLimits:
    return ResourceLimits(1 << 24, max_depth, max_nodes, max_attributes, 0)


def _mesh_series(path: Path, **overrides: object) -> Path:
    """A minimal openPMD 1.1.0 group-based mesh file written independently."""
    with h5py.File(path, "w") as handle:
        attributes = {
            "openPMD": np.bytes_("1.1.0"),
            "openPMDextension": np.uint32(0),
            "basePath": np.bytes_("/data/%T/"),
            "meshesPath": np.bytes_("meshes/"),
            "iterationEncoding": np.bytes_("groupBased"),
            "iterationFormat": np.bytes_("/data/%T/"),
            **overrides,
        }
        for name, value in attributes.items():
            if value is not None:
                handle.attrs[name] = value
        iteration = handle.create_group("data/100")
        iteration.attrs["time"] = np.float64(250.0)
        iteration.attrs["dt"] = np.float64(2.5)
        iteration.attrs["timeUnitSI"] = np.float64(1.0e-16)
        rho = iteration.create_group("meshes").create_dataset(
            "rho", data=np.arange(12.0).reshape(3, 4)
        )
        rho.attrs["unitDimension"] = np.asarray((-3.0, 0.0, 1.0, 1.0, 0, 0, 0))
        rho.attrs["unitSI"] = np.float64(_ELEMENTARY_CHARGE * 1.0e24)
        rho.attrs["gridUnitSI"] = np.float64(1.0e-6)
    return path


def test_record_unit_converts_stored_values_to_si_and_back(tmp_path: Path) -> None:
    path = tmp_path / "charge.h5"
    stored = np.asarray([1.0, -2.0, 0.5])
    with h5py.File(path, "w") as handle:
        record = handle.create_dataset("charge", data=stored)
        write_record_unit(
            record.attrs,
            record.attrs,
            OpenPMDUnit(_ELEMENTARY_CHARGE, _CHARGE_DIMENSION),
        )
    with h5py.File(path, "r") as handle:
        record = handle["charge"]
        unit = read_record_unit(record.attrs, record.attrs, _CHARGE_DIMENSION, 0.0)
        si = unit.to_si(np.asarray(record[()]))
    np.testing.assert_allclose(si, stored * 1.602176634e-19, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(unit.from_si(si), stored, rtol=1e-15, atol=0.0)


def test_record_unit_refuses_mismatched_dimension_and_invalid_scale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.h5"
    with h5py.File(path, "w") as handle:
        record = handle.create_dataset("E", data=np.zeros(2))
        write_record_unit(
            record.attrs, record.attrs, OpenPMDUnit(1.0, ELECTRIC_FIELD_DIMENSION)
        )
        record.attrs["unitSI"] = np.float64(-1.0)
    with h5py.File(path, "r") as handle:
        attributes = handle["E"].attrs
        with pytest.raises(ValueError, match="unitDimension"):
            # A magnetic field (kg A^-1 s^-2) is not an electric field.
            read_record_unit(
                attributes, attributes, (0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0), 1e-12
            )
        with pytest.raises(ValueError, match="unitSI must be finite and positive"):
            read_record_unit(attributes, attributes, ELECTRIC_FIELD_DIMENSION, 1e-12)
    with pytest.raises(ValueError, match="seven finite"):
        OpenPMDUnit(1.0, (1.0, 0.0, 0.0))  # ty: ignore[invalid-argument-type]


def test_velocity_record_in_units_of_c_converts_exactly() -> None:
    velocity = OpenPMDUnit(_SPEED_OF_LIGHT, (1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0))
    beta = np.asarray([0.0, 0.5, 0.999])
    np.testing.assert_array_equal(velocity.to_si(beta), beta * 299792458.0)


def test_mesh_revision_admits_release_series_and_iteration_time(
    tmp_path: Path,
) -> None:
    path = _mesh_series(tmp_path / "mesh.h5")
    with h5py.File(path, "r") as handle:
        root = read_series_root(handle, OPENPMD_MESH_REVISION)
        group, time = read_iteration(handle, 100)
        grid_unit = read_grid_unit_si(
            group["meshes/rho"].attrs,
            OPENPMD_MESH_REVISION,
            (LENGTH_DIMENSION, LENGTH_DIMENSION),
            0.0,
        )
    assert root.iteration_encoding == "groupBased"
    assert root.meshes_path == "meshes/"
    assert root.particles_path is None
    assert root.extensions == ()
    assert time.time_seconds == pytest.approx(2.5e-14, rel=1e-15)
    assert time.dt_seconds == pytest.approx(2.5e-16, rel=1e-15)
    # openPMD 1.1.0 shares one scalar gridUnitSI across all axes.
    np.testing.assert_array_equal(grid_unit, [1.0e-6, 1.0e-6])


def test_revisions_refuse_other_standard_versions(tmp_path: Path) -> None:
    path = _mesh_series(tmp_path / "mesh.h5")
    with h5py.File(path, "r") as handle:
        with pytest.raises(OpenPMDUnsupportedError, match=r"openPMD 1\.1\.0"):
            read_series_root(handle, OPENPMD_LASER_ENVELOPE_REVISION)
    draft = _mesh_series(tmp_path / "draft.h5", openPMD=np.bytes_("2.0.0"))
    with h5py.File(draft, "r") as handle:
        with pytest.raises(OpenPMDUnsupportedError, match=r"openPMD 2\.0\.0"):
            read_series_root(handle, OPENPMD_PARTICLE_REVISION)


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        pytest.param(np.uint32(0), (), id="base-standard"),
        pytest.param(np.uint32(1), ("ED-PIC",), id="ed-pic"),
    ],
)
def test_release_extension_bitmask_is_decoded(
    tmp_path: Path, mask: np.uint32, expected: tuple[str, ...]
) -> None:
    path = _mesh_series(tmp_path / "mesh.h5", openPMDextension=mask)
    with h5py.File(path, "r") as handle:
        assert read_series_root(handle, OPENPMD_MESH_REVISION).extensions == expected


def test_release_extension_bitmask_refuses_unknown_bits_and_text(
    tmp_path: Path,
) -> None:
    unknown = _mesh_series(tmp_path / "unknown.h5", openPMDextension=np.uint32(6))
    with h5py.File(unknown, "r") as handle:
        with pytest.raises(OpenPMDUnsupportedError, match="0x6"):
            read_series_root(handle, OPENPMD_MESH_REVISION)
    text = _mesh_series(tmp_path / "text.h5", openPMDextension=np.bytes_("ED-PIC"))
    with h5py.File(text, "r") as handle:
        with pytest.raises(ValueError, match="bitmask"):
            read_series_root(handle, OPENPMD_MESH_REVISION)


def test_draft_revision_requires_its_named_extension(tmp_path: Path) -> None:
    path = _mesh_series(
        tmp_path / "draft.h5",
        openPMD=np.bytes_("2.0.0"),
        openPMDextension=np.bytes_("SpeciesType"),
    )
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="LaserEnvelope extension is not declared"):
            read_series_root(handle, OPENPMD_LASER_ENVELOPE_REVISION)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"basePath": np.bytes_("/fields/%T/")}, "basePath", id="base"),
        pytest.param(
            {"iterationFormat": np.bytes_("/data/it%T/")},
            "iterationFormat",
            id="group-format",
        ),
        pytest.param(
            {
                "iterationEncoding": np.bytes_("fileBased"),
                "iterationFormat": np.bytes_("run/data_%T.h5"),
            },
            "file name",
            id="file-format",
        ),
        pytest.param(
            {
                "iterationEncoding": np.bytes_("fileBased"),
                "iterationFormat": np.bytes_("data_%T_%06T.h5"),
            },
            "file name",
            id="file-format-two-placeholders",
        ),
        pytest.param({"meshesPath": np.bytes_("/meshes/")}, "relative", id="meshes"),
        pytest.param({"openPMDextension": None}, "openPMDextension", id="extension"),
    ],
)
def test_series_root_refuses_malformed_metadata(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    path = _mesh_series(tmp_path / "bad.h5", **overrides)
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match=message):
            read_series_root(handle, OPENPMD_MESH_REVISION)


@pytest.mark.parametrize("pattern", ["data_%T.h5", "openpmd_%06T"])
def test_file_based_series_is_admitted(tmp_path: Path, pattern: str) -> None:
    path = _mesh_series(
        tmp_path / "data_100.h5",
        iterationEncoding=np.bytes_("fileBased"),
        iterationFormat=np.bytes_(pattern),
    )
    with h5py.File(path, "r") as handle:
        root = read_series_root(handle, OPENPMD_MESH_REVISION)
    assert root.iteration_encoding == "fileBased"
    assert root.iteration_format == pattern


def test_grid_units_follow_the_revision_layout(tmp_path: Path) -> None:
    path = tmp_path / "grid.h5"
    with h5py.File(path, "w") as handle:
        release = handle.create_dataset("release", data=np.zeros((2, 2)))
        release.attrs["gridUnitSI"] = np.float64(1.0e-6)
        draft = handle.create_dataset("draft", data=np.zeros((2, 2)))
        draft.attrs["gridUnitSI"] = np.asarray([1.0e-6, 1.0e-15])
        draft.attrs["gridUnitDimension"] = np.concatenate(
            [LENGTH_DIMENSION, TIME_DIMENSION]
        )
    axes = (LENGTH_DIMENSION, TIME_DIMENSION)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(
            read_grid_unit_si(
                handle["draft"].attrs, OPENPMD_LASER_ENVELOPE_REVISION, axes, 0.0
            ),
            [1.0e-6, 1.0e-15],
        )
        with pytest.raises(OpenPMDUnsupportedError, match="grid-axis unit dimensions"):
            read_grid_unit_si(
                handle["draft"].attrs,
                OPENPMD_LASER_ENVELOPE_REVISION,
                (TIME_DIMENSION, LENGTH_DIMENSION),
                0.0,
            )
        # openPMD 1.1.0 has no per-axis dimensions, so a time axis is refused.
        with pytest.raises(OpenPMDUnsupportedError, match="non-length grid axes"):
            read_grid_unit_si(handle["release"].attrs, OPENPMD_MESH_REVISION, axes, 0.0)


def test_written_series_root_and_iteration_read_back(tmp_path: Path) -> None:
    path = tmp_path / "written.h5"
    with h5py.File(path, "w") as handle:
        write_series_root(
            handle,
            OPENPMD_PARTICLE_REVISION,
            meshes_path=None,
            particles_path="particles/",
        )
        write_iteration(handle, 7, OpenPMDIterationTime(3.0, 0.5, 1.0e-15))
        handle.create_group("data/7/particles")
    with h5py.File(path, "r") as handle:
        root = read_series_root(handle, OPENPMD_PARTICLE_REVISION)
        _, time = read_iteration(handle, 7)
        with pytest.raises(ValueError, match="iteration is absent"):
            read_iteration(handle, 8)
    assert root.particles_path == "particles/"
    assert root.meshes_path is None
    assert time.time_seconds == pytest.approx(3.0e-15, rel=1e-15)


def test_preflight_inventories_every_payload_without_reading(tmp_path: Path) -> None:
    path = _mesh_series(tmp_path / "mesh.h5")
    with h5py.File(path, "r") as handle:
        inventory = preflight_hdf5(handle, _limits())
        # Root, data, data/100, meshes, rho: five objects; rho holds 12 float64.
        assert inventory.object_count == 5
        assert inventory.maximum_depth == 4
        assert set(inventory.datasets) == {"data/100/meshes/rho"}
        assert inventory.decoded_bytes == 12 * 8
        assert inventory.decoded_elements == 12
        inventory.require_budget(_limits(), 12 * 8 + 12 * 16, 12 * 16)
        with pytest.raises(ResourceReadError, match="maximum_decoded_bytes"):
            inventory.require_budget(_limits(), 12 * 8 + 12 * 16 - 1, 12 * 16)
        with pytest.raises(ResourceReadError, match="element count"):
            inventory.require_budget(_limits(max_nodes=16), 1 << 20, 0)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        pytest.param(_limits(max_depth=3), "structure exceeds", id="depth"),
        pytest.param(_limits(max_nodes=4), "structure exceeds", id="nodes"),
        pytest.param(_limits(max_attributes=8), "attributes exceed", id="attributes"),
    ],
)
def test_preflight_refuses_structures_beyond_bounds(
    tmp_path: Path, limits: ResourceLimits, message: str
) -> None:
    path = _mesh_series(tmp_path / "mesh.h5")
    with h5py.File(path, "r") as handle:
        with pytest.raises(ResourceReadError, match=message):
            preflight_hdf5(handle, limits)


def test_preflight_refuses_aliases_links_and_unadmitted_filters(
    tmp_path: Path,
) -> None:
    alias = _mesh_series(tmp_path / "alias.h5")
    with h5py.File(alias, "r+") as handle:
        handle["data/100/meshes/alias"] = handle["data/100/meshes/rho"]
    soft = _mesh_series(tmp_path / "soft.h5")
    with h5py.File(soft, "r+") as handle:
        handle["data/100/meshes/soft"] = h5py.SoftLink("/data/100/meshes/rho")
    filtered = _mesh_series(tmp_path / "lzf.h5")
    with h5py.File(filtered, "r+") as handle:
        handle.create_dataset("data/100/meshes/lzf", data=np.ones(64), compression="lzf")
    with h5py.File(alias, "r") as handle:
        with pytest.raises(OpenPMDUnsupportedError, match="hard-link aliases"):
            preflight_hdf5(handle, _limits())
    with h5py.File(soft, "r") as handle:
        with pytest.raises(ValueError, match="soft and external links"):
            preflight_hdf5(handle, _limits())
    with h5py.File(filtered, "r") as handle:
        with pytest.raises(OpenPMDUnsupportedError, match="filters"):
            preflight_hdf5(handle, _limits())

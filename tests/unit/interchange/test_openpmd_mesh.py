#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pytest

from phydrax import DimensionalScaleContract, ElectromagneticScaleContract
from phydrax._external_resource import (
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from phydrax._external_runtime import pin_executable
from phydrax.interchange import (
    convert_openpmd_hdf5_to_adios2,
    OpenPMDADIOS2Provider,
    OpenPMDMeshError,
    OpenPMDMeshImportPolicy,
    OpenPMDMeshIteration,
    OpenPMDMeshRecord,
    read_openpmd_adios2,
    read_openpmd_meshes_hdf5,
    write_openpmd_meshes_hdf5,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import CHARGE, KILOGRAM, LENGTH, TIME, UnitDefinition


# CODATA 2022 (NIST SP 961, May 2024), written independently of the scale owner.
_ELECTRON_MASS = 9.1093837139e-31
_ELEMENTARY_CHARGE = 1.602176634e-19
_LIGHT = 299_792_458.0
_SI = ElectromagneticScaleContract.si()
_EDGE = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
_FACE = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))


def _limits(max_bytes: int = 4_000_000, max_nodes: int = 200_000) -> ResourceLimits:
    return ResourceLimits(max_bytes, 12, max_nodes, 16_384, 1)


def _resource(path: Path, limits: ResourceLimits | None = None) -> BoundedResource:
    return read_bounded_resource(
        path.name,
        trusted_root=path.parent,
        limits=_limits() if limits is None else limits,
    )


def _micrometre_units() -> ElectromagneticScaleContract:
    """Electron-normalized code units: c = e = m_e = 1 and a 1 µm length unit."""
    length = Fraction(1, 10**6)
    time = length / _SI.speed_of_light
    mass, charge, relativity = _SI.electron_mass, _SI.elementary_charge, _SI.relativity
    return ElectromagneticScaleContract.code_units(
        DimensionalScaleContract(
            UnitDefinition("L0", LENGTH, "si", length),
            UnitDefinition("m_e", KILOGRAM.dimension, "si", mass),
            UnitDefinition("T0", TIME, "si", time),
        ),
        UnitDefinition("q_e", CHARGE, "si", charge),
        gravitational_constant=relativity.gravitational_constant
        * mass
        * time**2
        / length**3,
        speed_of_light=1,
        reduced_planck_constant=_SI.reduced_planck_constant * time / (mass * length**2),
        boltzmann_constant=relativity.boltzmann_constant * time**2 / (mass * length**2),
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=_SI.vacuum_permittivity
        * length**3
        * mass
        / (charge**2 * time**2),
        constant_set_id="codata-2022",
    )


def _cartesian_iteration() -> OpenPMDMeshIteration:
    """Staggered 3-D E, B, J, rho records of one leapfrog PIC iteration (SI)."""
    rng = np.random.default_rng(3)
    shape = (4, 3, 5)
    labels = ("x", "y", "z")
    spacing, offset = (1.0e-6, 2.0e-6, 0.5e-6), (0.0, -1.0e-6, 3.0e-6)

    def vector(magnitude: float) -> tuple[np.ndarray, ...]:
        return tuple(rng.normal(size=shape) * magnitude for _ in range(3))

    records = (
        OpenPMDMeshRecord(
            "rho",
            "cartesian",
            labels,
            spacing,
            offset,
            (rng.normal(size=shape),),
            ((0.0,) * 3,),
        ),
        OpenPMDMeshRecord(
            "J", "cartesian", labels, spacing, offset, vector(1.0e12), _EDGE, -0.5e-16
        ),
        OpenPMDMeshRecord(
            "B", "cartesian", labels, spacing, offset, vector(1.0e2), _FACE
        ),
        OpenPMDMeshRecord(
            "E", "cartesian", labels, spacing, offset, vector(1.0e9), _EDGE
        ),
    )
    return OpenPMDMeshIteration(7, 1.5e-15, 1.0e-16, records)


def _export(directory: Path, iteration: OpenPMDMeshIteration | None = None) -> Path:
    return write_openpmd_meshes_hdf5(
        directory,
        _cartesian_iteration() if iteration is None else iteration,
        scale=_SI,
        limits=_limits(),
    ).path


def _refusal(
    path: Path, policy: OpenPMDMeshImportPolicy, limits: ResourceLimits | None = None
) -> OpenPMDMeshError:
    with pytest.raises(OpenPMDMeshError) as caught:
        read_openpmd_meshes_hdf5(_resource(path, limits), policy, scale=_SI)
    assert caught.value.report.status == caught.value.status
    assert not caught.value.report.valid
    return caught.value


def test_cartesian_roundtrip_preserves_staggered_records_as_a_file_based_member(
    tmp_path: Path,
) -> None:
    source = _cartesian_iteration()
    exported = write_openpmd_meshes_hdf5(tmp_path, source, scale=_SI, limits=_limits())
    assert exported.path == tmp_path / "fields_7.h5"
    assert exported.report.status == AdapterStatus.LOSSLESS
    with h5py.File(exported.path, "r") as handle:
        assert handle.attrs["iterationEncoding"] == b"fileBased"
        assert handle.attrs["iterationFormat"] == b"fields_%T.h5"
        assert handle["data/7/meshes/E"].attrs["geometry"] == b"cartesian"

    imported = read_openpmd_meshes_hdf5(
        _resource(exported.path), OpenPMDMeshImportPolicy(7), scale=_SI
    )
    iteration = imported.iteration
    assert imported.report.status == AdapterStatus.LOSSLESS
    assert (iteration.iteration, iteration.time, iteration.dt) == (7, 1.5e-15, 1.0e-16)
    assert tuple(value.name for value in iteration.records) == ("E", "B", "J", "rho")
    for name in ("E", "B", "J", "rho"):
        expected, actual = source.record(name), iteration.record(name)
        assert actual.record_id == expected.record_id
        assert actual.positions == expected.positions
        assert actual.time_offset == expected.time_offset
        for left, right in zip(actual.components, expected.components, strict=True):
            np.testing.assert_array_equal(left, right)


def test_stored_axis_order_is_canonicalized_to_x_y_z(tmp_path: Path) -> None:
    # Independent Fortran-ordered 2-D record: the dataset is stored [z][x] as
    # seen from C, while axisLabels, gridSpacing, and position list x first.
    x = 0.1 + 0.2 * np.arange(3)
    z = -1.0 + 0.5 * np.arange(4)
    values = np.add.outer(10.0 * z, x)  # stored[k, i] = 10 z_k + x_i
    path = tmp_path / "fortran.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "openPMD": np.bytes_("1.1.0"),
                "openPMDextension": np.uint32(0),
                "basePath": np.bytes_("/data/%T/"),
                "meshesPath": np.bytes_("meshes/"),
                "iterationEncoding": np.bytes_("groupBased"),
                "iterationFormat": np.bytes_("/data/%T/"),
            }
        )
        iteration = handle.create_group("data/3")
        iteration.attrs.update({"time": 0.0, "dt": 1.0, "timeUnitSI": 1.0})
        rho = iteration.create_dataset("meshes/rho", data=values)
        rho.attrs.update(
            {
                "geometry": np.bytes_("cartesian"),
                "dataOrder": np.bytes_("F"),
                "axisLabels": np.asarray([b"x", b"z"]),
                "gridSpacing": np.asarray([0.2, 0.5]),
                "gridGlobalOffset": np.asarray([0.1, -1.0]),
                "gridUnitSI": 1.0,
                "position": np.asarray([0.25, 0.0]),
                "unitSI": 1.0,
                "unitDimension": np.asarray([-3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                "timeOffset": 0.0,
            }
        )

    record = read_openpmd_meshes_hdf5(
        _resource(path), OpenPMDMeshImportPolicy(3, records=("rho",)), scale=_SI
    ).iteration.record("rho")
    assert record.axis_labels == ("x", "z")
    assert record.grid_spacing == (0.2, 0.5)
    assert record.grid_global_offset == (0.1, -1.0)
    assert record.positions == ((0.25, 0.0),)
    np.testing.assert_array_equal(record.components[0], np.add.outer(x, 10.0 * z))


def _normalized_file(path: Path) -> None:
    """Hand-written code output: E in E0 units, µm grid, fs times, record offset."""
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "openPMD": np.bytes_("1.1.0"),
                "openPMDextension": np.uint32(1),
                "basePath": np.bytes_("/data/%T/"),
                "meshesPath": np.bytes_("fields/"),
                "iterationEncoding": np.bytes_("fileBased"),
                "iterationFormat": np.bytes_("run_%T"),
            }
        )
        iteration = handle.create_group("data/40")
        iteration.attrs.update({"time": 40.0, "dt": 0.1, "timeUnitSI": 1.0e-15})
        electric = iteration.create_group("fields/E")
        electric.attrs.update(
            {
                "geometry": np.bytes_("cartesian"),
                "dataOrder": np.bytes_("C"),
                "axisLabels": np.asarray([b"x", b"y"]),
                "gridSpacing": np.asarray([0.5, 0.25], dtype=np.float32),
                "gridGlobalOffset": np.asarray([-2.0, 10.0]),
                "gridUnitSI": 1.0e-6,
                "unitDimension": np.asarray([1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0]),
                "timeOffset": np.float32(0.05),
            }
        )
        for index, name in enumerate(("x", "y", "z")):
            component = electric.create_dataset(
                name, data=(index + 1.0) * np.arange(6.0).reshape(2, 3)
            )
            component.attrs.update({"unitSI": 3.2e12, "position": [0.5, 0.0]})


def test_stored_units_convert_through_unit_si_to_any_bound_scale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_40.h5"
    _normalized_file(path)
    policy = OpenPMDMeshImportPolicy(40, records=("E",))

    si = read_openpmd_meshes_hdf5(_resource(path), policy, scale=_SI).iteration
    electric = si.record("E")
    np.testing.assert_allclose(si.time, 40.0e-15, rtol=1e-15)
    np.testing.assert_allclose(si.dt, 0.1e-15, rtol=1e-15)
    np.testing.assert_allclose(electric.grid_spacing, (0.5e-6, 0.25e-6), rtol=1e-15)
    np.testing.assert_allclose(electric.grid_global_offset, (-2.0e-6, 10.0e-6))
    np.testing.assert_allclose(electric.time_offset, 0.05e-15, rtol=1e-7)
    np.testing.assert_allclose(
        electric.components[2], 3.0 * 3.2e12 * np.arange(6.0).reshape(2, 3), rtol=1e-15
    )

    # Code units: L0 = 1 µm, T0 = L0 / c, E unit = m_e c^2 / (e L0).
    code = read_openpmd_meshes_hdf5(
        _resource(path), policy, scale=_micrometre_units()
    ).iteration
    field_unit = _ELECTRON_MASS * _LIGHT**2 / (_ELEMENTARY_CHARGE * 1.0e-6)
    np.testing.assert_allclose(code.time, 40.0e-15 * _LIGHT / 1.0e-6, rtol=1e-12)
    np.testing.assert_allclose(code.record("E").grid_spacing, (0.5, 0.25), rtol=1e-12)
    np.testing.assert_allclose(
        code.record("E").components[0],
        3.2e12 * np.arange(6.0).reshape(2, 3) / field_unit,
        rtol=1e-12,
    )


def test_theta_mode_records_use_the_openpmd_mode_plane_layout(tmp_path: Path) -> None:
    r = 0.5 + np.arange(4.0)
    z = np.arange(5.0)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    # F(r, θ, z) = a + c1 cos θ + s1 sin θ + c2 cos 2θ + s2 sin 2θ.
    coefficients = (rr + zz, rr * zz, np.sin(zz) * rr, np.cos(rr), rr**2 - zz)
    planes = np.stack(coefficients)
    components = (planes, 2.0 * planes, -planes)
    record = OpenPMDMeshRecord(
        "B",
        "thetaMode",
        ("r", "z"),
        (1.0e-6, 2.0e-6),
        (0.0, 5.0e-6),
        components,
        ((0.5, 0.0), (0.0, 0.0), (0.5, 0.5)),
    )
    assert record.mode_count == 3
    assert record.component_names == ("r", "t", "z")
    path = write_openpmd_meshes_hdf5(
        tmp_path,
        OpenPMDMeshIteration(2, 0.0, 1.0e-16, (record,)),
        scale=_SI,
        limits=_limits(),
    ).path

    theta = np.linspace(0.0, 2.0 * np.pi, 7)
    expected = (
        coefficients[0][..., None]
        + coefficients[1][..., None] * np.cos(theta)
        + coefficients[2][..., None] * np.sin(theta)
        + coefficients[3][..., None] * np.cos(2.0 * theta)
        + coefficients[4][..., None] * np.sin(2.0 * theta)
    )
    with h5py.File(path, "r") as handle:
        stored = handle["data/2/meshes/B"]
        assert stored.attrs["geometryParameters"] == b"m=3;imag=+"
        assert [bytes(value) for value in stored.attrs["axisLabels"]] == [b"r", b"z"]
        raw = stored["t"][()]
    # Standard reconstruction from the raw planes: mode 0, then (Re, Im) per m.
    synthesized = raw[0][..., None] + sum(
        raw[2 * m - 1][..., None] * np.cos(m * theta)
        + raw[2 * m][..., None] * np.sin(m * theta)
        for m in (1, 2)
    )
    np.testing.assert_allclose(synthesized, 2.0 * expected, rtol=1e-14, atol=1e-13)

    imported = read_openpmd_meshes_hdf5(
        _resource(path), OpenPMDMeshImportPolicy(2, records=("B",)), scale=_SI
    ).iteration.record("B")
    assert imported.record_id == record.record_id


def _replace_dataset(
    path: Path, name: str, transform: Callable[[np.ndarray], np.ndarray]
) -> None:
    with h5py.File(path, "r+") as handle:
        item = handle[name]
        attributes = dict(item.attrs)
        values = transform(item[()])
        del handle[name]
        replaced = handle.create_dataset(name, data=values)
        replaced.attrs.update(attributes)


def _set_attribute(path: Path, name: str, key: str, value: object) -> None:
    with h5py.File(path, "r+") as handle:
        handle[name].attrs[key] = value


def _theta_file(path: Path, parameters: bytes, planes: int) -> None:
    record = OpenPMDMeshRecord(
        "rho",
        "thetaMode",
        ("r", "z"),
        (1.0, 1.0),
        (0.0, 0.0),
        (np.arange(12.0).reshape(3, 2, 2),),
        ((0.0, 0.0),),
    )
    write_openpmd_meshes_hdf5(
        path.parent,
        OpenPMDMeshIteration(0, 0.0, 1.0, (record,)),
        scale=_SI,
        limits=_limits(),
        series=path.stem.removesuffix("_0"),
    )
    _set_attribute(path, "data/0/meshes/rho", "geometryParameters", parameters)
    _replace_dataset(
        path,
        "data/0/meshes/rho",
        lambda values: np.arange(4.0 * planes).reshape(planes, 2, 2),
    )


def _truncate_component(path: Path) -> None:
    _replace_dataset(path, "data/7/meshes/E/y", lambda values: values[:, :-1])


def _truncate_bytes(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _wrong_dimension(path: Path) -> None:
    _set_attribute(
        path, "data/7/meshes/B", "unitDimension", np.asarray([1.0, 1, -3, -1, 0, 0, 0])
    )


def _cylindrical(path: Path) -> None:
    _set_attribute(path, "data/7/meshes/E", "geometry", np.bytes_("cylindrical"))


def _standard_two(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        handle.attrs["openPMD"] = np.bytes_("2.0.0")


def _missing_record(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        del handle["data/7/meshes/J"]


def _missing_component(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        del handle["data/7/meshes/B/z"]


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (_truncate_component, AdapterStatus.MALFORMED_SOURCE),
        (_truncate_bytes, AdapterStatus.MALFORMED_SOURCE),
        (_wrong_dimension, AdapterStatus.MALFORMED_SOURCE),
        (_missing_record, AdapterStatus.MALFORMED_SOURCE),
        (_cylindrical, AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC),
        (_standard_two, AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC),
        (_missing_component, AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC),
    ],
    ids=[
        "truncated-component",
        "truncated-image",
        "unit-dimension",
        "missing-record",
        "cylindrical-geometry",
        "openpmd-2",
        "missing-component",
    ],
)
def test_malformed_and_unsupported_cartesian_sources_are_refused(
    tmp_path: Path, mutate: Callable[[Path], None], status: AdapterStatus
) -> None:
    path = _export(tmp_path)
    mutate(path)
    assert _refusal(path, OpenPMDMeshImportPolicy(7)).status == status


@pytest.mark.parametrize(
    ("parameters", "planes", "status"),
    [
        (b"m=2;imag=-", 3, AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC),
        (b"m=2;imag=+", 4, AdapterStatus.MALFORMED_SOURCE),
        (b"imag=+", 3, AdapterStatus.MALFORMED_SOURCE),
    ],
    ids=["negative-imaginary", "mode-plane-count", "missing-mode-count"],
)
def test_theta_mode_conventions_are_enforced(
    tmp_path: Path, parameters: bytes, planes: int, status: AdapterStatus
) -> None:
    path = tmp_path / "theta_0.h5"
    _theta_file(path, parameters, planes)
    assert _refusal(path, OpenPMDMeshImportPolicy(0, records=("rho",))).status == status


def test_decoded_payload_and_structure_budgets_refuse_before_reading(
    tmp_path: Path,
) -> None:
    path = _export(tmp_path)
    budget = _refusal(path, OpenPMDMeshImportPolicy(7, maximum_decoded_bytes=4_096))
    assert budget.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "maximum_decoded_bytes" in str(budget)
    nodes = _refusal(path, OpenPMDMeshImportPolicy(7), _limits(max_nodes=64))
    assert nodes.status == AdapterStatus.INCONSISTENT_SOURCE


def test_export_refuses_beyond_limits_without_publishing(tmp_path: Path) -> None:
    with pytest.raises(ResourceReadError):
        write_openpmd_meshes_hdf5(
            tmp_path, _cartesian_iteration(), scale=_SI, limits=_limits(max_bytes=2_048)
        )
    assert list(tmp_path.iterdir()) == []


def _openpmd_pipe() -> OpenPMDADIOS2Provider:
    executable = shutil.which(os.environ.get("PHYDRAX_OPENPMD_PIPE", "openpmd-pipe"))
    version = os.environ.get("PHYDRAX_OPENPMD_API_VERSION")
    if executable is None or version is None:
        pytest.skip(
            "requires openPMD-api openpmd-pipe (set PHYDRAX_OPENPMD_PIPE and "
            "PHYDRAX_OPENPMD_API_VERSION)"
        )
    return OpenPMDADIOS2Provider(
        pin_executable(executable, version=version, license_id="LGPL-3.0-or-later")
    )


def test_adios2_provider_route_round_trips_through_openpmd_api(tmp_path: Path) -> None:
    provider = _openpmd_pipe()
    source = _cartesian_iteration()
    exported = write_openpmd_meshes_hdf5(tmp_path, source, scale=_SI, limits=_limits())
    (tmp_path / "bp").mkdir()
    artifact = convert_openpmd_hdf5_to_adios2(
        provider, exported.resource, tmp_path / "bp", limits=_limits()
    )
    assert artifact.path == tmp_path / "bp" / "fields_7.bp"
    assert sorted(value.path for value in artifact.files) == [
        "fields_7.bp/data.0",
        "fields_7.bp/md.0",
        "fields_7.bp/md.idx",
    ]
    image = read_openpmd_adios2(provider, artifact.path, limits=_limits())
    imported = read_openpmd_meshes_hdf5(image, OpenPMDMeshImportPolicy(7), scale=_SI)
    for name in ("E", "B", "J", "rho"):
        assert imported.iteration.record(name).record_id == source.record(name).record_id

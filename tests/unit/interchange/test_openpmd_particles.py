#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
)
from phydrax.interchange import (
    OpenPMDParticleTrackError,
    OpenPMDParticleTrackImportPolicy,
    OpenPMDParticleTrackSelection,
    read_openpmd_particle_tracks_hdf5,
    write_openpmd_particle_tracks_hdf5,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import CHARGE, KILOGRAM, LENGTH, TIME, UnitDefinition


# CODATA 2022 (NIST SP 961, May 2024), written independently of the scale owner.
_ELECTRON_MASS = 9.1093837139e-31
_ELEMENTARY_CHARGE = 1.602176634e-19
_LIGHT = 299_792_458.0
_SI = ElectromagneticScaleContract.si()


def _limits(max_bytes: int = 4_000_000, max_nodes: int = 200_000) -> ResourceLimits:
    return ResourceLimits(max_bytes, 12, max_nodes, 16_384, 1)


def _resource(path: Path, limits: ResourceLimits | None = None) -> BoundedResource:
    return read_bounded_resource(
        path.name,
        trusted_root=path.parent,
        limits=_limits() if limits is None else limits,
    )


def _policy(
    identities: tuple[int, ...] | None = None,
) -> OpenPMDParticleTrackImportPolicy:
    return OpenPMDParticleTrackImportPolicy(
        OpenPMDParticleTrackSelection("electrons", identities=identities)
    )


def _trajectory(*, accelerations: bool = False) -> ChargedTrajectory:
    """Four electron lanes with unordered identities and distinct weights."""
    rng = np.random.default_rng(7)
    samples, lanes = 6, 4
    times = np.linspace(0.0, 5.0e-13, samples)
    velocities = rng.normal(size=(samples, lanes, 3)) * 1.0e8
    return ChargedTrajectory(
        times,
        rng.normal(size=(samples, lanes, 3)) * 1.0e-6,
        velocities,
        np.full(lanes, -_ELEMENTARY_CHARGE),
        np.asarray([2.0, 1.0, 4.0, 3.0]),
        np.ones((samples, lanes), dtype=np.bool_),
        (
            np.asarray([1, 0, 0, 2], dtype=np.uint32),
            np.asarray([5, 9, 3, 0], dtype=np.uint32),
        ),
        proper_accelerations=velocities * 1.0e12 if accelerations else None,
    )


def _identity_order(trajectory: ChargedTrajectory) -> np.ndarray:
    identities = (
        np.asarray(trajectory.id_hi, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(trajectory.id_lo, dtype=np.uint64)
    return np.argsort(identities)


def _export(path: Path, trajectory: ChargedTrajectory | None = None) -> Path:
    source = _trajectory() if trajectory is None else trajectory
    write_openpmd_particle_tracks_hdf5(
        path,
        source,
        np.full(source.particle_count, _ELECTRON_MASS),
        scale=_SI,
        species="electrons",
        limits=_limits(),
    )
    return path


def _rewrite_particles(
    path: Path, iteration: int, select: Callable[[str, np.ndarray], np.ndarray]
) -> None:
    """Replace every stored component of one iteration by ``select(name, values)``."""
    with h5py.File(path, "r+") as handle:
        species = handle[f"data/{iteration}/particles/electrons"]
        names: list[str] = []
        species.visit(names.append)
        for name in names:
            item = species[name]
            if isinstance(item, h5py.Dataset):
                attributes = dict(item.attrs)
                values = select(name, item[()])
                del species[name]
                replaced = species.create_dataset(name, data=values)
                for key, value in attributes.items():
                    replaced.attrs[key] = value
            elif "shape" in item.attrs:
                count = int(item.attrs["shape"][0])
                kept = select(name, np.arange(count)).shape[0]
                item.attrs["shape"] = np.asarray([kept], dtype=np.uint64)


def _refusal(
    path: Path, policy: OpenPMDParticleTrackImportPolicy
) -> OpenPMDParticleTrackError:
    with pytest.raises(OpenPMDParticleTrackError) as caught:
        read_openpmd_particle_tracks_hdf5(_resource(path), policy, scale=_SI)
    assert caught.value.report.status == caught.value.status
    assert not caught.value.report.valid
    return caught.value


def test_roundtrip_preserves_tracks_in_ascending_identity_order(tmp_path: Path) -> None:
    source = _trajectory()
    masses = np.asarray([1.0, 2.0, 3.0, 4.0]) * _ELECTRON_MASS
    exported = write_openpmd_particle_tracks_hdf5(
        tmp_path / "tracks.h5",
        source,
        masses,
        scale=_SI,
        species="electrons",
        limits=_limits(),
        iterations=range(10, 70, 10),
    )
    imported = read_openpmd_particle_tracks_hdf5(
        _resource(exported.path), _policy(), scale=_SI
    )

    order = _identity_order(source)
    trajectory = imported.trajectory
    assert exported.report.status == AdapterStatus.LOSSLESS
    assert imported.report.status == AdapterStatus.LOSSLESS
    assert imported.report.valid
    assert imported.iterations == (10, 20, 30, 40, 50, 60)
    assert np.array_equal(trajectory.id_hi, np.asarray(source.id_hi)[order])
    assert np.array_equal(trajectory.id_lo, np.asarray(source.id_lo)[order])
    assert np.asarray(trajectory.id_lo).tolist() == [3, 9, 5, 0]
    np.testing.assert_allclose(trajectory.times, source.times, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        trajectory.positions, np.asarray(source.positions)[:, order], rtol=1e-15
    )
    np.testing.assert_allclose(
        trajectory.proper_velocities,
        np.asarray(source.proper_velocities)[:, order],
        rtol=1e-14,
    )
    np.testing.assert_array_equal(trajectory.charges, np.asarray(source.charges)[order])
    np.testing.assert_array_equal(
        trajectory.multiplicities, np.asarray(source.multiplicities)[order]
    )
    np.testing.assert_allclose(imported.masses, masses[order], rtol=1e-15)
    assert bool(np.all(np.asarray(trajectory.active)))


def test_export_declares_dropped_accelerations_and_refuses_unencodable_lanes(
    tmp_path: Path,
) -> None:
    exported = write_openpmd_particle_tracks_hdf5(
        tmp_path / "hermite.h5",
        _trajectory(accelerations=True),
        np.full(4, _ELECTRON_MASS),
        scale=_SI,
        species="electrons",
        limits=_limits(),
    )
    assert exported.report.status == AdapterStatus.DECLARED_LOSS
    assert [loss.path for loss in exported.report.losses] == ["proper_accelerations"]

    source = _trajectory()
    active = np.ones((6, 4), dtype=np.bool_)
    active[3, 1] = False
    inactive = ChargedTrajectory(
        source.times,
        source.positions,
        source.proper_velocities,
        source.charges,
        source.multiplicities,
        active,
        (source.id_hi, source.id_lo),
    )
    with pytest.raises(ValueError, match="Inactive samples"):
        _export(tmp_path / "inactive.h5", inactive)
    lane_times = np.repeat(np.asarray(source.times)[:, :1], 4, axis=1)
    lane_times[2, 3] += 1.0e-15
    staggered = ChargedTrajectory(
        lane_times,
        source.positions,
        source.proper_velocities,
        source.charges,
        source.multiplicities,
        source.active,
        (source.id_hi, source.id_lo),
    )
    with pytest.raises(ValueError, match="per-lane times"):
        _export(tmp_path / "staggered.h5", staggered)


def _code_units(length_si: Fraction) -> ElectromagneticScaleContract:
    """Electron-normalized code units: c = e = m_e = 1, length unit ``length_si``."""
    light = _SI.speed_of_light
    time_si = length_si / light
    mass_si = _SI.electron_mass
    charge_si = _SI.elementary_charge
    relativity = _SI.relativity
    return ElectromagneticScaleContract.code_units(
        DimensionalScaleContract(
            UnitDefinition("L0", LENGTH, "si", length_si),
            UnitDefinition("m_e", KILOGRAM.dimension, "si", mass_si),
            UnitDefinition("T0", TIME, "si", time_si),
        ),
        UnitDefinition("q_e", CHARGE, "si", charge_si),
        gravitational_constant=relativity.gravitational_constant
        * mass_si
        * time_si**2
        / length_si**3,
        speed_of_light=1,
        reduced_planck_constant=_SI.reduced_planck_constant
        * time_si
        / (mass_si * length_si**2),
        boltzmann_constant=relativity.boltzmann_constant
        * time_si**2
        / (mass_si * length_si**2),
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=_SI.vacuum_permittivity
        * length_si**3
        * mass_si
        / (charge_si**2 * time_si**2),
        constant_set_id="codata-2022",
    )


# Independent PIC-style encoding (hand-written openPMD 1.1.0 + ED-PIC): positions
# in micrometres relative to a constant offset in millimetres, macroparticle
# momenta in m_e c (macroWeighted, weightingPower 1), charge in e and mass in m_e
# as constant components, and times in femtoseconds with a record timeOffset.
_CELL_POSITIONS = np.asarray(
    [
        [[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]],
        [[0.75, -0.5, 2.5], [2.5, 0.5, -0.25]],
        [[1.25, 0.0, 3.25], [2.0, 1.0, 0.5]],
    ]
)
_OFFSET_MM = np.asarray([1.0, -2.0, 0.5])
_MACRO_MOMENTA = np.asarray(
    [
        [[3.0, 0.0, 30.0], [0.0, -4.0, 12.0]],
        [[2.0, 1.0, 31.0], [1.0, -4.0, 11.0]],
        [[1.0, 2.0, 32.0], [2.0, -3.0, 10.0]],
    ]
)
_WEIGHTS = np.asarray([3.0, 0.5])
_IDENTITIES = np.asarray([2**40 + 5, 17], dtype=np.uint64)
_ITERATION_FS = np.asarray([0.0, 2.0, 4.0])
_TIME_OFFSET_FS = 0.5
_UNIT_DIMENSIONS = {
    "position": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "positionOffset": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "momentum": (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "charge": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    "mass": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "weighting": (0.0,) * 7,
    "id": (0.0,) * 7,
}


def _pic_record(
    item: h5py.Group | h5py.Dataset, name: str, macro: int, power: float
) -> None:
    item.attrs["unitDimension"] = np.asarray(_UNIT_DIMENSIONS[name])
    item.attrs["timeOffset"] = np.float64(_TIME_OFFSET_FS)
    item.attrs["macroWeighted"] = np.uint32(macro)
    item.attrs["weightingPower"] = np.float64(power)


def _write_pic_series(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["openPMD"] = np.bytes_("1.1.0")
        handle.attrs["openPMDextension"] = np.uint32(1)
        handle.attrs["basePath"] = np.bytes_("/data/%T/")
        handle.attrs["particlesPath"] = np.bytes_("particles/")
        handle.attrs["iterationEncoding"] = np.bytes_("groupBased")
        handle.attrs["iterationFormat"] = np.bytes_("/data/%T/")
        for sample, time in enumerate(_ITERATION_FS):
            iteration = handle.create_group(f"data/{100 * sample}")
            iteration.attrs["time"] = np.float64(time)
            iteration.attrs["dt"] = np.float64(2.0)
            iteration.attrs["timeUnitSI"] = np.float64(1.0e-15)
            species = iteration.create_group("particles/electrons")
            # Storage order reversed on odd samples: identity alone matches lanes.
            order = slice(None, None, -1 if sample % 2 else 1)
            for name, values, unit, power in (
                ("position", _CELL_POSITIONS[sample], 1.0e-6, 0.0),
                ("momentum", _MACRO_MOMENTA[sample], _ELECTRON_MASS * _LIGHT, 1.0),
            ):
                record = species.create_group(name)
                _pic_record(record, name, int(name == "momentum"), power)
                for axis, label in enumerate("xyz"):
                    component = record.create_dataset(label, data=values[order, axis])
                    component.attrs["unitSI"] = np.float64(unit)
            offset = species.create_group("positionOffset")
            _pic_record(offset, "positionOffset", 0, 0.0)
            for axis, label in enumerate("xyz"):
                component = offset.create_group(label)
                component.attrs["value"] = np.float64(_OFFSET_MM[axis])
                component.attrs["shape"] = np.asarray([2], dtype=np.uint64)
                component.attrs["unitSI"] = np.float64(1.0e-3)
            for name, value, unit, macro, power in (
                ("charge", -1.0, _ELEMENTARY_CHARGE, 0, 1.0),
                ("mass", 1.0, _ELECTRON_MASS, 0, 1.0),
            ):
                constant = species.create_group(name)
                constant.attrs["value"] = np.float64(value)
                constant.attrs["shape"] = np.asarray([2], dtype=np.uint64)
                constant.attrs["unitSI"] = np.float64(unit)
                _pic_record(constant, name, macro, power)
            for name, data, macro, power in (
                ("weighting", _WEIGHTS[order], 1, 1.0),
                ("id", _IDENTITIES[order], 0, 0.0),
            ):
                dataset = species.create_dataset(name, data=data)
                dataset.attrs["unitSI"] = np.float64(1.0)
                _pic_record(dataset, name, macro, power)


def _pic_si_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lanes in ascending identity (17, then 2**40 + 5): SI time, position, u."""
    lanes = np.asarray([1, 0], dtype=np.intp)
    times = (_ITERATION_FS + _TIME_OFFSET_FS) * 1.0e-15
    positions = _CELL_POSITIONS[:, lanes] * 1.0e-6 + _OFFSET_MM * 1.0e-3
    per_particle = _MACRO_MOMENTA[:, lanes] / _WEIGHTS[lanes][None, :, None]
    velocities = per_particle * _LIGHT  # p / m_e with p in m_e c
    return times, positions, velocities


def test_import_converts_record_units_macroweighting_and_scale(tmp_path: Path) -> None:
    path = tmp_path / "pic.h5"
    _write_pic_series(path)
    times, positions, velocities = _pic_si_reference()

    si = read_openpmd_particle_tracks_hdf5(_resource(path), _policy(), scale=_SI)
    trajectory = si.trajectory
    assert si.report.status == AdapterStatus.LOSSLESS
    assert si.iterations == (0, 100, 200)
    assert np.asarray(trajectory.id_lo).tolist() == [17, 5]
    assert np.asarray(trajectory.id_hi).tolist() == [0, 2**8]
    np.testing.assert_allclose(trajectory.times[:, 0], times, rtol=1e-15)
    np.testing.assert_allclose(trajectory.positions, positions, rtol=1e-14)
    np.testing.assert_allclose(trajectory.proper_velocities, velocities, rtol=1e-14)
    np.testing.assert_allclose(trajectory.charges, -_ELEMENTARY_CHARGE, rtol=1e-15)
    np.testing.assert_array_equal(trajectory.multiplicities, [0.5, 3.0])
    np.testing.assert_allclose(si.masses, _ELECTRON_MASS, rtol=1e-15)

    # Electron-normalized micrometre units (c = e = m_e = 1).
    code = read_openpmd_particle_tracks_hdf5(
        _resource(path), _policy(), scale=_code_units(Fraction(1, 10**6))
    )
    time_unit = 1.0e-6 / _LIGHT
    np.testing.assert_allclose(code.trajectory.times[:, 0], times / time_unit, rtol=1e-14)
    np.testing.assert_allclose(code.trajectory.positions, positions / 1.0e-6, rtol=1e-14)
    np.testing.assert_allclose(
        code.trajectory.proper_velocities, velocities / _LIGHT, rtol=1e-14
    )
    np.testing.assert_allclose(code.trajectory.charges, -1.0, rtol=1e-15)
    np.testing.assert_allclose(code.masses, 1.0, rtol=1e-15)


def test_identity_reorder_leaves_imported_tracks_unchanged(tmp_path: Path) -> None:
    reference = read_openpmd_particle_tracks_hdf5(
        _resource(_export(tmp_path / "ordered.h5")), _policy(), scale=_SI
    )
    shuffled = _export(tmp_path / "shuffled.h5")
    rng = np.random.default_rng(3)
    for iteration in range(6):
        permutation = rng.permutation(4)
        _rewrite_particles(shuffled, iteration, lambda _, values: values[permutation])
    permuted = read_openpmd_particle_tracks_hdf5(
        _resource(shuffled), _policy(), scale=_SI
    )

    shuffled_lanes, ordered_lanes = permuted.trajectory, reference.trajectory
    for shuffled_array, ordered_array in (
        (shuffled_lanes.times, ordered_lanes.times),
        (shuffled_lanes.positions, ordered_lanes.positions),
        (shuffled_lanes.proper_velocities, ordered_lanes.proper_velocities),
        (shuffled_lanes.charges, ordered_lanes.charges),
        (shuffled_lanes.multiplicities, ordered_lanes.multiplicities),
        (shuffled_lanes.id_hi, ordered_lanes.id_hi),
        (shuffled_lanes.id_lo, ordered_lanes.id_lo),
    ):
        np.testing.assert_array_equal(shuffled_array, ordered_array)
    np.testing.assert_array_equal(permuted.masses, reference.masses)
    assert permuted.report.target_id == reference.report.target_id
    assert permuted.report.source_id != reference.report.source_id


def test_explicit_identity_subset_follows_only_selected_particles(tmp_path: Path) -> None:
    path = _export(tmp_path / "subset.h5")
    # Particle 9 (index 1) is lost after the second iteration.
    for iteration in range(2, 6):
        _rewrite_particles(path, iteration, lambda _, values: values[[0, 2, 3]])
    selected = read_openpmd_particle_tracks_hdf5(
        _resource(path), _policy(identities=(2**33, 3, 2**32 + 5)), scale=_SI
    )
    assert np.asarray(selected.trajectory.id_lo).tolist() == [3, 5, 0]
    assert np.asarray(selected.trajectory.id_hi).tolist() == [0, 1, 2]

    missing = _refusal(path, _policy())
    assert missing.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "missing" in str(missing)
    lost = _refusal(path, _policy(identities=(9,)))
    assert lost.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "[9]" in str(lost)


def test_truncated_record_is_refused_before_payload_decoding(tmp_path: Path) -> None:
    path = _export(tmp_path / "truncated.h5")
    _rewrite_particles(
        path,
        4,
        lambda name, values: values[:-1] if name == "momentum/x" else values,
    )
    refused = _refusal(path, _policy())
    assert refused.status == AdapterStatus.MALFORMED_SOURCE
    assert "truncated" in str(refused)


def test_missing_identity_record_is_refused_as_unsupported(tmp_path: Path) -> None:
    path = _export(tmp_path / "anonymous.h5")
    with h5py.File(path, "r+") as handle:
        del handle["data/1/particles/electrons/id"]
    refused = _refusal(path, _policy())
    assert refused.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert "identities are missing" in str(refused)


def test_repeated_identity_is_refused(tmp_path: Path) -> None:
    path = _export(tmp_path / "repeated.h5")
    _rewrite_particles(
        path,
        2,
        lambda name, values: values[[0, 0, 2, 3]] if name == "id" else values,
    )
    refused = _refusal(path, _policy())
    assert refused.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "repeat" in str(refused)


def test_nonmonotonic_iteration_time_is_refused(tmp_path: Path) -> None:
    path = _export(tmp_path / "time.h5")
    with h5py.File(path, "r+") as handle:
        handle["data/3"].attrs["time"] = handle["data/1"].attrs["time"]
    refused = _refusal(path, _policy())
    assert refused.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "nonmonotonic" in str(refused)


def test_changing_particle_charge_between_iterations_is_refused(tmp_path: Path) -> None:
    path = _export(tmp_path / "charge.h5")
    with h5py.File(path, "r+") as handle:
        handle["data/2/particles/electrons/charge"].attrs["value"] = np.float64(
            -2.0 * _ELEMENTARY_CHARGE
        )
    refused = _refusal(path, _policy())
    assert refused.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "charge changes" in str(refused)


@pytest.mark.parametrize(
    ("record", "dimension"),
    [
        ("position", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ("momentum", (1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)),
        ("charge", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ],
    ids=["position-as-mass", "momentum-as-velocity", "charge-as-current"],
)
def test_unit_dimension_mismatch_is_refused(
    tmp_path: Path, record: str, dimension: tuple[float, ...]
) -> None:
    path = _export(tmp_path / "units.h5")
    with h5py.File(path, "r+") as handle:
        handle[f"data/0/particles/electrons/{record}"].attrs["unitDimension"] = (
            np.asarray(dimension)
        )
    refused = _refusal(path, _policy())
    assert refused.status == AdapterStatus.MALFORMED_SOURCE
    assert "unitDimension" in str(refused)


def test_decoded_resource_overflow_is_refused_before_payload_read(
    tmp_path: Path,
) -> None:
    path = _export(tmp_path / "bounded.h5")
    restrictive = OpenPMDParticleTrackImportPolicy(
        OpenPMDParticleTrackSelection("electrons"), maximum_decoded_bytes=2048
    )
    refused = _refusal(path, restrictive)
    assert refused.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "maximum_decoded_bytes" in str(refused)

    with pytest.raises(OpenPMDParticleTrackError) as nodes:
        read_openpmd_particle_tracks_hdf5(
            _resource(path, _limits(max_nodes=64)), _policy(), scale=_SI
        )
    assert nodes.value.status == AdapterStatus.INCONSISTENT_SOURCE

    destination = tmp_path / "oversize.h5"
    with pytest.raises(ResourceReadError):
        write_openpmd_particle_tracks_hdf5(
            destination,
            _trajectory(),
            np.full(4, _ELECTRON_MASS),
            scale=_SI,
            species="electrons",
            limits=ResourceLimits(1024, 12, 200_000, 4096, 0),
        )
    assert not destination.exists()


def test_trajectory_radiation_consumes_imported_tracks(tmp_path: Path) -> None:
    # One nonrelativistic cyclotron period (beta = 0.01) of two counter-phased
    # electrons written in PIC code units, radiated from the imported lanes and
    # from the same orbit built directly in SI.
    samples, beta, radius = 97, 0.01, 1.0e-3
    omega0 = beta * _LIGHT / radius
    times = np.linspace(0.0, 2.0 * np.pi / omega0, samples)
    phases = omega0 * times[:, None] + np.asarray([0.0, np.pi])[None, :]
    positions = radius * np.stack(
        [np.cos(phases), np.sin(phases), np.zeros_like(phases)], axis=-1
    )
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    velocities = (
        gamma
        * beta
        * _LIGHT
        * np.stack([-np.sin(phases), np.cos(phases), np.zeros_like(phases)], axis=-1)
    )
    direct = ChargedTrajectory(
        times,
        positions,
        velocities,
        np.full(2, -_ELEMENTARY_CHARGE),
        np.ones(2),
        np.ones((samples, 2), dtype=np.bool_),
        (np.zeros(2, dtype=np.uint32), np.asarray([4, 8], dtype=np.uint32)),
    )
    code = _code_units(Fraction(1, 10**3))
    path = tmp_path / "orbit.h5"
    write_openpmd_particle_tracks_hdf5(
        path,
        direct,
        np.full(2, _ELECTRON_MASS),
        scale=_SI,
        species="electrons",
        limits=_limits(),
    )
    imported = read_openpmd_particle_tracks_hdf5(_resource(path), _policy(), scale=code)

    observers = RadiationObserverPlan(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), np.asarray([0.0, 1.0, 0.0])
    )
    frequencies = omega0 * np.asarray([1.0, 2.0])
    time_unit = 1.0e-3 / _LIGHT
    spectra = []
    for scale, trajectory, factor in (
        (_SI, direct, 1.0),
        (code, imported.trajectory, time_unit),
    ):
        plan = TrajectoryRadiationPlan(
            scale,
            observers,
            frequencies * factor,
            coherence="incoherent",
            route="segment-exact",
            emission="truncated",
        )
        spectra.append(np.asarray(plan.prepare().evaluate(trajectory).spectral_energy))
    energy_unit = code.unit_si_map()["energy"][0]
    # d²W/(dω dΩ) carries energy × time.
    np.testing.assert_allclose(
        spectra[1] * energy_unit * time_unit, spectra[0], rtol=1e-9
    )
    assert np.all(spectra[0][0] > 0.0)

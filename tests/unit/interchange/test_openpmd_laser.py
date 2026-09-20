#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._external_resource import (
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from phydrax.discretization import (
    AxisDiscretization,
    AxisDomain,
    PreparedTensorGrid,
    TensorGridPlan,
    UniformAxisSpec,
)
from phydrax.geometry import RigidFrame
from phydrax.interchange._openpmd_laser import (
    OpenPMDLaserEnvelopeError,
    OpenPMDLaserEnvelopeImportPolicy,
    OpenPMDLaserEnvelopeProfile,
    read_openpmd_laser_envelope_hdf5,
    write_openpmd_laser_envelope_hdf5,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.optics.wave._envelope import PulseEnvelopeField
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._pulse_time import PulseTimeSpace


def _limits(max_bytes=2_000_000, max_nodes=100_000):
    return ResourceLimits(max_bytes, 12, max_nodes, 128, 0)


def _field(*, polarization="tangential"):
    plane_grid = TensorGridPlan(
        (UniformAxisSpec(7), UniformAxisSpec(9)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[-2.0, -3.0], [2.0, 3.0]]))
    time_grid = TensorGridPlan((UniformAxisSpec(11),), axis_names=("time",)).prepare(
        jnp.asarray([[-4.0], [4.0]])
    )
    plane = PlaneFieldSpace(plane_grid, RigidFrame.identity(3), "finite-window")
    time = PulseTimeSpace(time_grid, topology="finite-window")
    coordinates = plane.transverse_coordinates
    scalar = jnp.exp(
        -0.5
        * (
            coordinates[..., 0, None] ** 2
            + 0.5 * coordinates[..., 1, None] ** 2
            + 0.25 * time.coordinates[None, None, :] ** 2
        )
    ) * jnp.exp(0.2j * time.coordinates[None, None, :])
    if polarization == "scalar":
        values = scalar
    else:
        jones = jnp.asarray([1.0, 1.0j]) / jnp.sqrt(2.0)
        values = scalar[..., None] * jones
    return PulseEnvelopeField(
        plane,
        time,
        values,
        2.4e15,
        0.0,
        polarization=polarization,
    )


def _resource(path: Path, limits=None):
    return read_bounded_resource(
        path.name,
        trusted_root=path.parent,
        limits=_limits() if limits is None else limits,
    )


def test_hdf5_roundtrip_preserves_temporal_complex_electric_envelope(tmp_path: Path):
    profile = OpenPMDLaserEnvelopeProfile(
        record_name="driver",
        iteration=7,
        axis_labels=("t", "y", "x"),
        polarization=(1.0, 1.0j),
    )
    source = _field()
    path = tmp_path / "laser.h5"
    exported = write_openpmd_laser_envelope_hdf5(path, source, profile, limits=_limits())
    imported = read_openpmd_laser_envelope_hdf5(
        _resource(path),
        OpenPMDLaserEnvelopeImportPolicy(profile),
        frame=RigidFrame.identity(3),
        longitudinal_coordinate=0.0,
    )

    assert exported.report.status == AdapterStatus.LOSSLESS
    assert imported.report.status == AdapterStatus.LOSSLESS
    assert imported.report.valid
    assert imported.field.polarization == "tangential"
    assert imported.field.values.shape == source.values.shape
    assert jnp.allclose(imported.field.values, source.values, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(
        imported.field.plane_space.coordinate_axes[0],
        source.plane_space.coordinate_axes[0],
    )
    assert jnp.allclose(
        imported.field.time_space.coordinates, source.time_space.coordinates
    )
    assert imported.field.carrier_angular_frequency == source.carrier_angular_frequency
    assert imported.field.longitudinal_coordinate == source.longitudinal_coordinate

    path.unlink()
    assert jnp.isfinite(jnp.sum(jnp.abs(imported.field.values)))
    assert imported.resource.data


def test_scalar_envelope_roundtrip_expands_declared_polarization(tmp_path: Path):
    profile = OpenPMDLaserEnvelopeProfile(polarization=(0.6, 0.8j))
    source = _field(polarization="scalar")
    path = tmp_path / "scalar.h5"
    write_openpmd_laser_envelope_hdf5(path, source, profile, limits=_limits())

    imported = read_openpmd_laser_envelope_hdf5(
        _resource(path),
        OpenPMDLaserEnvelopeImportPolicy(profile),
        frame=RigidFrame.identity(3),
        longitudinal_coordinate=0.0,
    )
    expected = source.values[..., None] * jnp.asarray(profile.polarization)
    assert jnp.allclose(imported.field.values, expected, rtol=2e-6, atol=2e-6)


def test_export_rejects_tangential_field_without_constant_profile_polarization(
    tmp_path: Path,
):
    source = _field()
    wrong = OpenPMDLaserEnvelopeProfile(polarization=(1.0, 0.0))

    with pytest.raises(ValueError, match="not factorizable"):
        write_openpmd_laser_envelope_hdf5(
            tmp_path / "wrong.h5", source, wrong, limits=_limits()
        )
    located = PulseEnvelopeField(
        source.plane_space,
        source.time_space,
        source.values,
        source.carrier_angular_frequency,
        1.0,
        polarization=source.polarization,
    )
    with pytest.raises(ValueError, match="nonzero longitudinal coordinate"):
        write_openpmd_laser_envelope_hdf5(
            tmp_path / "located.h5",
            located,
            OpenPMDLaserEnvelopeProfile(polarization=(1.0, 1.0j)),
            limits=_limits(),
        )


def test_export_rejects_nonuniform_plane_axes(tmp_path: Path):
    source = _field(polarization="scalar")
    nonuniform_axis = AxisDiscretization(
        nodes=jnp.asarray([-2.0, -0.7, 0.0, 2.0]),
        quad_weights=jnp.asarray([0.65, 1.0, 1.35, 1.0]),
        basis="nonuniform",
        domain=AxisDomain.interval(-2.0, 2.0),
        lower_endpoint_included=True,
        upper_endpoint_included=True,
    )
    uniform_axis = UniformAxisSpec(9).materialize(jnp.asarray(-3.0), jnp.asarray(3.0))
    plane = PlaneFieldSpace(
        PreparedTensorGrid((nonuniform_axis, uniform_axis), axis_names=("x", "y")),
        RigidFrame.identity(3),
        "finite-window",
    )
    values = jnp.ones(plane.shape + source.time_space.shape, dtype="complex128")
    nonuniform = PulseEnvelopeField(
        plane,
        source.time_space,
        values,
        source.carrier_angular_frequency,
        source.longitudinal_coordinate,
    )

    with pytest.raises(ValueError, match="uniform point axes"):
        write_openpmd_laser_envelope_hdf5(
            tmp_path / "nonuniform.h5",
            nonuniform,
            OpenPMDLaserEnvelopeProfile(),
            limits=_limits(),
        )


def test_import_rejects_vector_potential_theta_and_malformed_metadata_with_reports(
    tmp_path: Path,
):
    profile = OpenPMDLaserEnvelopeProfile()
    policy = OpenPMDLaserEnvelopeImportPolicy(profile)
    cases = (
        (
            "vector.h5",
            "envelopeField",
            np.bytes_("normalized_vector_potential"),
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        ),
        (
            "theta.h5",
            "geometry",
            np.bytes_("thetaMode"),
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        ),
    )
    for name, attribute, value, status in cases:
        path = tmp_path / name
        write_openpmd_laser_envelope_hdf5(
            path, _field(polarization="scalar"), profile, limits=_limits()
        )
        with h5py.File(path, "r+") as handle:
            handle[f"data/0/meshes/{profile.record_name}"].attrs[attribute] = value
        with pytest.raises(OpenPMDLaserEnvelopeError) as caught:
            read_openpmd_laser_envelope_hdf5(
                _resource(path),
                policy,
                frame=RigidFrame.identity(3),
                longitudinal_coordinate=0.0,
            )
        assert caught.value.status == status
        assert caught.value.report.status == status
        assert not caught.value.report.valid

    malformed = tmp_path / "malformed.h5"
    write_openpmd_laser_envelope_hdf5(
        malformed,
        _field(polarization="scalar"),
        profile,
        limits=_limits(),
    )
    with h5py.File(malformed, "r+") as handle:
        del handle[f"data/0/meshes/{profile.record_name}"].attrs["angularFrequency"]
    with pytest.raises(OpenPMDLaserEnvelopeError) as caught:
        read_openpmd_laser_envelope_hdf5(
            _resource(malformed),
            policy,
            frame=RigidFrame.identity(3),
            longitudinal_coordinate=0.0,
        )
    assert caught.value.status == AdapterStatus.MALFORMED_SOURCE
    assert not caught.value.report.valid


def test_import_preflights_decoded_resources_before_payload_read(tmp_path: Path):
    profile = OpenPMDLaserEnvelopeProfile()
    path = tmp_path / "bounded.h5"
    write_openpmd_laser_envelope_hdf5(
        path,
        _field(polarization="scalar"),
        profile,
        limits=_limits(),
    )
    resource = _resource(path)
    restrictive = OpenPMDLaserEnvelopeImportPolicy(profile, maximum_decoded_bytes=16)

    with pytest.raises(OpenPMDLaserEnvelopeError) as caught:
        read_openpmd_laser_envelope_hdf5(
            resource,
            restrictive,
            frame=RigidFrame.identity(3),
            longitudinal_coordinate=0.0,
        )
    assert caught.value.status == AdapterStatus.INCONSISTENT_SOURCE
    assert "maximum_decoded_bytes" in str(caught.value)
    assert not caught.value.report.valid

    too_few_nodes = _limits(max_nodes=16)
    limited_resource = _resource(path, too_few_nodes)
    with pytest.raises(OpenPMDLaserEnvelopeError) as nodes:
        read_openpmd_laser_envelope_hdf5(
            limited_resource,
            OpenPMDLaserEnvelopeImportPolicy(profile),
            frame=RigidFrame.identity(3),
            longitudinal_coordinate=0.0,
        )
    assert nodes.value.status == AdapterStatus.INCONSISTENT_SOURCE


def test_export_fails_closed_before_publishing_oversize_hdf5(tmp_path: Path):
    destination = tmp_path / "too-small.h5"
    with pytest.raises(ResourceReadError):
        write_openpmd_laser_envelope_hdf5(
            destination,
            _field(polarization="scalar"),
            OpenPMDLaserEnvelopeProfile(),
            limits=ResourceLimits(256, 12, 100_000, 128, 0),
        )
    assert not destination.exists()

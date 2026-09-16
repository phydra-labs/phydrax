#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._kinematics import LorentzFrame
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)


def _scale(*, quantum_constants_explicit: bool = True):
    return RelativityScaleContract(
        DimensionalScaleContract.si(),
        1,
        3,
        2,
        1,
        quantum_constants_explicit,
    )


def _units(*, convention: RelativityConvention | None = None):
    return RelativisticUnitContract(
        _scale(),
        RelativityConvention(metric_signature="mostly_minus")
        if convention is None
        else convention,
    )


def _geometry(units, *, snapshot_token=7, valid=True):
    return ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(valid),
        snapshot_token=jnp.asarray(snapshot_token),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat-adm",
    )


def _tetrad(units, *, vectors=None):
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    metric = minkowski_metric(chart, convention="mostly_minus")
    return orthonormal_tetrad(
        metric,
        jnp.eye(4) if vectors is None else vectors,
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="inertial-observer",
    )


def _frame(*, snapshot_token=7, time=0.5, scale_factor=0.75):
    units = _units()
    return LocalRelativisticFramePlan(
        _geometry(units, snapshot_token=snapshot_token),
        _tetrad(units),
        units,
        jnp.asarray((time, 0.0, 0.0, 0.0)),
        jnp.asarray(time),
        jnp.asarray(scale_factor),
        observer_id="comoving-observer",
        orientation_id="right-handed-future",
    )


def test_physical_natural_and_constant_bridges_round_trip_without_hidden_c_or_hbar():
    units = _units()
    energy = jnp.asarray((10.0, 17.0))
    momentum = jnp.asarray(((1.0, -2.0, 0.5), (2.0, 0.0, -1.0)))

    four_momentum = units.assemble_four_momentum(energy, momentum)
    restored_energy, restored_momentum = units.split_four_momentum(four_momentum)
    natural = units.to_natural_four_momentum(four_momentum, jnp.asarray((2.0, 4.0)))
    restored_four_momentum = units.from_natural_four_momentum(
        natural,
        jnp.asarray((2.0, 4.0)),
    )

    np.testing.assert_allclose(four_momentum[..., 1:], 3.0 * momentum)
    np.testing.assert_allclose(restored_energy, energy)
    np.testing.assert_allclose(restored_momentum, momentum)
    np.testing.assert_allclose(restored_four_momentum, four_momentum)
    np.testing.assert_allclose(
        units.rest_energy_to_mass(units.mass_to_rest_energy(jnp.asarray(5.0))),
        5.0,
    )
    np.testing.assert_allclose(
        units.energy_to_wave_number(units.wave_number_to_energy(jnp.asarray(7.0))),
        7.0,
    )
    np.testing.assert_allclose(
        units.energy_to_angular_frequency(
            units.angular_frequency_to_energy(jnp.asarray(11.0))
        ),
        11.0,
    )


def test_lorentz_scalar_and_mass_shell_are_invariant_in_E_cp_components():
    units = _units()
    mass = jnp.asarray(2.0)
    spatial_momentum = jnp.asarray((0.5, -0.25, 0.75))
    rest_energy = units.mass_to_rest_energy(mass)
    cp = 3.0 * spatial_momentum
    energy = jnp.sqrt(rest_energy**2 + jnp.sum(cp**2))
    four_momentum = units.assemble_four_momentum(energy, spatial_momentum)
    boost = LorentzFrame.boost(jnp.asarray((0.2, -0.1, 0.05)))
    transformed = boost.apply(four_momentum).value

    np.testing.assert_allclose(
        units.lorentz_scalar(transformed, transformed),
        units.lorentz_scalar(four_momentum, four_momentum),
        rtol=2.0e-6,
    )
    assert bool(units.mass_shell_admissible(four_momentum, mass))
    assert bool(units.mass_shell_admissible(1.0e-9 * four_momentum, 1.0e-9 * mass))
    assert not bool(
        units.mass_shell_admissible(
            four_momentum.at[0].add(1.0),
            mass,
            relative_tolerance=1.0e-8,
        )
    )


def test_metric_signature_is_explicit_in_scalar_and_contract_identity():
    mostly_minus = _units()
    mostly_plus = _units(convention=RelativityConvention(metric_signature="mostly_plus"))
    rest = jnp.asarray((2.0, 0.0, 0.0, 0.0))

    np.testing.assert_allclose(mostly_minus.lorentz_scalar(rest, rest), 4.0)
    np.testing.assert_allclose(mostly_plus.lorentz_scalar(rest, rest), -4.0)
    assert mostly_minus.contract_id != mostly_plus.contract_id


def test_local_frame_round_trip_and_identity_bind_snapshot_time_and_scale():
    frame = _frame()
    coordinate = jnp.asarray((9.0, 1.0, -2.0, 3.0))

    local = frame.coordinate_to_local(coordinate)
    restored = frame.local_to_coordinate(local)
    later = _frame(time=0.6)
    new_snapshot = _frame(snapshot_token=8)

    np.testing.assert_allclose(local, coordinate)
    np.testing.assert_allclose(restored, coordinate)
    assert bool(frame.admissible)
    assert frame.frame_id == later.frame_id
    assert frame.frame_id == new_snapshot.frame_id
    assert frame.realization_id() != later.realization_id()
    assert frame.realization_id() != new_snapshot.realization_id()
    assert int(frame.frame_token) == 7
    assert int(new_snapshot.frame_token) == 8
    assert frame.units.contract_id == _units().contract_id


def test_eulerian_frame_from_arbitrary_adm_is_jax_safe_and_invertible():
    units = _units()
    spatial_host = (
        np.asarray(((2.0, 0.1, 0.0), (0.1, 1.5, 0.05), (0.0, 0.05, 0.75))) * 1.0e-12
    )
    spatial = jnp.asarray(spatial_host)
    geometry = ADMGridGeometry(
        jnp.asarray(0.8),
        jnp.asarray((0.1, -0.05, 0.02)),
        spatial,
        jnp.asarray(np.linalg.inv(spatial_host)),
        jnp.asarray(np.sqrt(np.linalg.det(spatial_host))),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(11),
        chart_id="curved-adm",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="z4c-stage",
    )

    build = eqx.filter_jit(
        lambda value: LocalRelativisticFramePlan.from_adm(
            value,
            units,
            jnp.asarray((0.4, 0.0, 0.0, 0.0)),
            jnp.asarray(0.4),
            jnp.asarray(0.9),
            observer_id="eulerian-observer",
            orientation_id="future-right-handed",
        )
    )
    frame = build(geometry)
    coordinate = jnp.asarray((5.0, 0.4, -0.2, 0.1))
    restored = eqx.filter_jit(
        lambda plan, value: plan.local_to_coordinate(plan.coordinate_to_local(value))
    )(frame, coordinate)

    np.testing.assert_allclose(restored, coordinate, rtol=2.0e-5, atol=2.0e-5)
    assert bool(frame.admissible)
    assert bool(frame.tetrad.qualified)
    assert int(frame.frame_token) == 11


def test_invalid_unit_and_physical_domain_inputs_are_refused():
    convention = RelativityConvention(metric_signature="mostly_minus")
    with pytest.raises(ValueError, match="explicitly declared c and hbar"):
        RelativisticUnitContract(
            _scale(quantum_constants_explicit=False),
            convention,
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="reference_energy"):
        _units().to_natural_four_momentum(jnp.ones((4,)), jnp.asarray(0.0))
    with pytest.raises(eqx.EquinoxRuntimeError, match="future-directed"):
        _units().invariant_phase_space_weight(jnp.asarray((0.0, 0.0, 0.0, 0.0)))
    with pytest.raises(eqx.EquinoxRuntimeError, match="rest_mass"):
        _units().mass_shell_residual(jnp.ones((4,)), jnp.asarray(-1.0))


@pytest.mark.parametrize(
    ("keyword", "invalid", "message"),
    (
        ("phase_space_normalization", "flat-d3p", "phase-space normalization"),
        ("s_matrix_normalization", "implicit-delta", "S-matrix normalization"),
        ("spin_normalization", "implicit-average", "spin normalization"),
        ("color_normalization", "implicit-average", "color normalization"),
        ("polarization_normalization", "implicit-gauge", "polarization normalization"),
        (
            "identical_particle_normalization",
            "implicit-symmetry",
            "identical-particle normalization",
        ),
    ),
)
def test_unknown_scattering_normalization_is_refused(keyword, invalid, message):
    with pytest.raises(ValueError, match=message):
        RelativisticUnitContract(
            _scale(),
            RelativityConvention(metric_signature="mostly_minus"),
            **{keyword: invalid},
        )


def test_invalid_tetrad_geometry_or_scale_binding_is_refused():
    units = _units()
    bad_tetrad = _tetrad(units, vectors=2.0 * jnp.eye(4))
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="valid ADM, tetrad, time, and scale",
    ):
        LocalRelativisticFramePlan(
            _geometry(units),
            bad_tetrad,
            units,
            jnp.zeros((4,)),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            observer_id="observer",
            orientation_id="orientation",
        )

    other_units = RelativisticUnitContract(
        RelativityScaleContract(
            DimensionalScaleContract.si(),
            1,
            4,
            2,
            1,
        ),
        units.convention,
    )
    with pytest.raises(ValueError, match="different scales"):
        LocalRelativisticFramePlan(
            _geometry(units),
            _tetrad(units),
            other_units,
            jnp.zeros((4,)),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            observer_id="observer",
            orientation_id="orientation",
        )


def test_contract_identity_covers_every_normalization_convention():
    units = _units()
    alternative = RelativisticUnitContract(
        units.scale,
        units.convention,
        spin_normalization="sum-all",
        color_normalization="density-matrix-explicit",
        polarization_normalization="covariant-gauge-with-ward-check",
        identical_particle_normalization="ordered-labeled-final-state",
    )

    assert units.contract_id != alternative.contract_id
    assert units.energy_unit.unit_id == alternative.energy_unit.unit_id
    assert units.momentum_unit.unit_id == alternative.momentum_unit.unit_id
    assert units.four_vector_convention == "contravariant-(E,c*p)"
    assert units.phase_space_normalization == "lorentz-invariant-2E"
    assert units.s_matrix_normalization == "covariant-delta"

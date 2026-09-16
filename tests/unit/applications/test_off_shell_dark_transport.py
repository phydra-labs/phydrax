#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import QuantumKineticState
from phydrax.applications.curved_spacetime_qft._off_shell_transport import (
    breit_wigner_off_shell_state,
    off_shell_moments,
    OffShellTransportPlan,
    quasiparticle_off_shell_state,
)
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


def _support():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(2),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="off-shell-test-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="off-shell-test-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="off-shell-test-observer",
        orientation_id="right-handed-future",
    )
    species = (
        DarkSectorSpeciesPlan(
            "off-shell-dark-fermion",
            1.0,
            charge_names=("dark-charge",),
            charges=(1.0,),
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    return QuantumKineticState(
        jnp.asarray([[[0.2]]]),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True,)),
        units=units,
        frame=frame,
    )


def _plan():
    nodes = np.linspace(-18.0, 22.0, 4001)
    spacing = nodes[1] - nodes[0]
    weights = np.full(nodes.shape, spacing)
    weights[[0, -1]] *= 0.5
    return OffShellTransportPlan(
        _support(),
        nodes,
        weights,
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        energy_quadrature_id="symmetric-trapezoid-4001",
        momentum_quadrature_id="single-on-shell-mode",
        spectral_tolerance=7.0e-3,
        dyson_tolerance=2.0e-12,
        kms_tolerance=2.0e-12,
    )


def test_breit_wigner_moments_sum_rule_dyson_and_kms_evidence():
    plan = _plan()
    state = breit_wigner_off_shell_state(
        plan,
        jnp.asarray([[2.0]]),
        0.2,
        inverse_temperature=0.4,
        chemical_potentials=jnp.asarray((0.1,)),
    )
    momentum = jnp.asarray([[[2.0, 0.3, 0.0, 0.0]]])
    moments = off_shell_moments(plan, state, momentum)

    assert bool(state.evidence.causal)
    assert bool(state.evidence.positive)
    assert bool(state.evidence.sum_rule_satisfied)
    assert bool(state.evidence.dyson_satisfied)
    assert bool(state.evidence.kms_satisfied)
    assert bool(state.evidence.valid)
    np.testing.assert_allclose(moments.spectral_weight, 1.0, atol=7.0e-3)
    np.testing.assert_allclose(
        moments.spectral_energy_moment / moments.spectral_weight,
        2.0,
        atol=2.0e-2,
    )
    assert state.energy_quadrature_id == plan.energy_quadrature_id
    assert state.frame_id == plan.frame_id
    assert state.unit_contract_id == plan.unit_contract_id
    np.testing.assert_array_equal(state.frame_token, plan.frame.frame_token)
    np.testing.assert_array_equal(state.frame_time, plan.frame.time)
    np.testing.assert_array_equal(state.frame_scale_factor, plan.frame.scale_factor)
    assert state.frame_realization_id == plan.frame.realization_id()


def test_narrow_width_limit_converges_to_on_shell_fermi_occupation():
    plan = _plan()
    pole = jnp.asarray([[2.0]])
    beta = 0.4
    chemical = jnp.asarray((0.1,))
    broad = breit_wigner_off_shell_state(
        plan,
        pole,
        0.3,
        inverse_temperature=beta,
        chemical_potentials=chemical,
    )
    narrow = breit_wigner_off_shell_state(
        plan,
        pole,
        0.05,
        inverse_temperature=beta,
        chemical_potentials=chemical,
    )
    momentum = jnp.asarray([[[2.0, 0.0, 0.0, 0.0]]])
    broad_number = off_shell_moments(plan, broad, momentum).occupation_number[0, 0, 0]
    narrow_number = off_shell_moments(plan, narrow, momentum).occupation_number[0, 0, 0]
    on_shell = 1.0 / (np.exp(beta * (2.0 - 0.1)) + 1.0)

    assert abs(float(narrow_number) - on_shell) < abs(float(broad_number) - on_shell)
    np.testing.assert_allclose(narrow_number, on_shell, atol=2.0e-3)


def test_negative_width_is_refused_instead_of_clipped():
    plan = _plan()
    shape = plan.spectral_shape
    zeros = jnp.zeros(shape)
    with pytest.raises(Exception, match="causality"):
        quasiparticle_off_shell_state(
            plan,
            jnp.ones(shape),
            jnp.zeros(shape),
            zeros,
            -jnp.ones(shape),
            zeros,
            zeros,
            jnp.asarray([[2.0]]),
            inverse_temperature=1.0,
            chemical_potentials=jnp.asarray((0.0,)),
            time=0.0,
        )

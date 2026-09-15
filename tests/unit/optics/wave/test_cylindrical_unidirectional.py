#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.artifacts import ArtifactManifest
from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.discretization.spectral import CylindricalHankelPlan
from phydrax.optics.materials import (
    AngularFrequencyValidity,
    ConstantRefractiveIndex,
    RefractiveIndexProvenance,
)
from phydrax.optics.wave._cylindrical import (
    CylindricalAnalyticPulseField,
    CylindricalUnidirectionalPropagationPlan,
    CylindricalUnidirectionalPropagationStatus,
    prepare_cylindrical_unidirectional_propagation,
    propagate_cylindrical_unidirectional,
)
from phydrax.optics.wave._nonlinear_response import InstantaneousScalarSusceptibility
from phydrax.optics.wave._pulse_time import PulseTimeSpace


jax.config.update("jax_enable_x64", True)


def _provenance() -> RefractiveIndexProvenance:
    manifest = ArtifactManifest(
        artifact_id="cylindrical-propagation-test-law",
        producer="phydrax-tests",
        version="1",
        sha256="0" * 64,
        byte_size=0,
        source_uri="generated://cylindrical-test-law",
        license_id="LicenseRef-PHYDRA",
        model="analytic cylindrical refractive-index test law",
        coverage="unit-test frequencies",
    )
    return RefractiveIndexProvenance(manifest, record_id="cylindrical-test-law")


def _law() -> ConstantRefractiveIndex:
    return ConstantRefractiveIndex(
        1.5,
        validity=AngularFrequencyValidity(0.5, 40.0),
        reference_wave_speed=1.0,
        provenance=_provenance(),
        law_id="cylindrical-constant-test-index",
    )


def _spaces(radial_count: int = 12, temporal_count: int = 64, radius: float = 20.0):
    hankel = CylindricalHankelPlan(radius, radial_count, order=0).prepare()
    temporal_grid = TensorGridPlan(
        (FourierAxisSpec(temporal_count),), axis_names=("time",)
    ).prepare(jnp.asarray([[0.0], [2.0 * jnp.pi]]))
    return hankel, PulseTimeSpace(temporal_grid, topology="periodic-cell")


def _plan(hankel, time_space, omega: float, **overrides):
    options = {
        "step_count": 4,
        "dealias_fraction": 1.0,
        "edge_guard_fraction": 0.01,
        "maximum_spectral_edge_fraction": 1.0,
        "maximum_analytic_signal_defect": 1.0,
        "maximum_hermitian_reconstruction_defect": 1.0,
        "maximum_nonlinear_rejected_fraction": 1.0,
        "maximum_refinement_error": 1.0,
        "maximum_backward_wave_estimate": 1.0,
        "maximum_radial_boundary_fraction": 1.0,
        "maximum_radial_high_mode_fraction": 1.0,
        "maximum_longitudinal_cutoff_fraction": 1.0,
    }
    options.update(overrides)
    return CylindricalUnidirectionalPropagationPlan(hankel, time_space, omega, **options)


def _mode_field(hankel, time_space, temporal_mode: int, radial_mode: int):
    radial_spectrum = jnp.zeros(hankel.plan.radial_count, dtype=jnp.complex128)
    radial_spectrum = radial_spectrum.at[radial_mode].set(1.0)
    radial_values = hankel.inverse(radial_spectrum)
    carrier = jnp.exp(-1j * temporal_mode * time_space.coordinates)
    return CylindricalAnalyticPulseField(
        hankel,
        time_space,
        radial_values[:, None] * carrier[None, :],
        float(temporal_mode),
        0.0,
    )


def test_linear_hankel_modes_gain_the_analytic_longitudinal_phase():
    hankel, time_space = _spaces()
    temporal_mode = 8
    radial_mode = 2
    distance = 0.37
    field = _mode_field(hankel, time_space, temporal_mode, radial_mode)
    prepared = prepare_cylindrical_unidirectional_propagation(
        _plan(hankel, time_space, float(temporal_mode)), _law()
    )
    result = propagate_cylindrical_unidirectional(
        prepared,
        field,
        InstantaneousScalarSusceptibility(),
        distance,
    )
    radial_values = field.values[:, 0]
    initial_modes = hankel.forward(radial_values)
    modal_phase = jnp.exp(
        1j * prepared.longitudinal_wavenumbers[:, temporal_mode] * distance
    )
    expected_radial = hankel.inverse(initial_modes * modal_phase)
    carrier = jnp.exp(-1j * temporal_mode * time_space.coordinates)
    expected = expected_radial[:, None] * carrier[None, :]

    assert jnp.allclose(result.field.values, expected, rtol=2.0e-10, atol=2.0e-11)
    assert result.status == int(CylindricalUnidirectionalPropagationStatus.SUCCESS)
    assert result.evidence.hankel.prepared_id == hankel.prepared_id
    assert result.evidence.input_radial_measure > 0.0
    assert result.evidence.output_radial_measure > 0.0
    assert result.evidence.radial_measure_relative_change < 1.0e-5
    assert result.response_evaluation.successful


def test_temporal_edge_violation_has_explicit_cylindrical_status():
    hankel, time_space = _spaces()
    field = _mode_field(hankel, time_space, 31, 0)
    plan = _plan(
        hankel,
        time_space,
        31.0,
        edge_guard_fraction=0.1,
        maximum_spectral_edge_fraction=1.0e-6,
    )
    result = propagate_cylindrical_unidirectional(
        plan.prepare(_law()),
        field,
        InstantaneousScalarSusceptibility(),
        0.0,
    )

    assert result.evidence.spectral_edge_fraction > 0.99
    assert result.status == int(
        CylindricalUnidirectionalPropagationStatus.SPECTRAL_EDGE_LIMIT
    )


def test_nonzero_azimuthal_order_is_rejected_for_optical_propagation():
    hankel = CylindricalHankelPlan(10.0, 8, order=1).prepare()
    temporal_grid = TensorGridPlan((FourierAxisSpec(32),), axis_names=("time",)).prepare(
        jnp.asarray([[0.0], [2.0 * jnp.pi]])
    )
    time_space = PulseTimeSpace(temporal_grid, topology="periodic-cell")

    with pytest.raises(ValueError, match="only m=0"):
        CylindricalUnidirectionalPropagationPlan(hankel, time_space, 5.0, step_count=2)


def test_radial_boundary_and_high_mode_evidence_have_distinct_statuses():
    hankel, time_space = _spaces(radius=8.0)
    carrier = jnp.exp(-10j * time_space.coordinates)
    boundary_values = jnp.zeros(
        (hankel.plan.radial_count, time_space.size), dtype=jnp.complex128
    )
    boundary_values = boundary_values.at[-1, :].set(carrier)
    boundary_field = CylindricalAnalyticPulseField(
        hankel, time_space, boundary_values, 10.0, 0.0
    )
    boundary_plan = _plan(
        hankel,
        time_space,
        10.0,
        maximum_radial_boundary_fraction=1.0e-8,
    )
    boundary = propagate_cylindrical_unidirectional(
        boundary_plan.prepare(_law()),
        boundary_field,
        InstantaneousScalarSusceptibility(),
        0.0,
    )

    assert boundary.evidence.radial_boundary_fraction > 0.99
    assert boundary.status == int(
        CylindricalUnidirectionalPropagationStatus.RADIAL_BOUNDARY_LIMIT
    )

    high_field = _mode_field(hankel, time_space, 10, hankel.plan.radial_count - 1)
    high_plan = _plan(
        hankel,
        time_space,
        10.0,
        maximum_radial_high_mode_fraction=1.0e-8,
    )
    high = propagate_cylindrical_unidirectional(
        high_plan.prepare(_law()),
        high_field,
        InstantaneousScalarSusceptibility(),
        0.0,
    )

    assert high.evidence.radial_high_mode_fraction > 0.99
    assert high.status == int(
        CylindricalUnidirectionalPropagationStatus.RADIAL_HIGH_MODE_LIMIT
    )


def test_cutoff_and_resource_limits_fail_closed_without_changing_the_field():
    hankel, time_space = _spaces(radius=4.0)
    cutoff_field = _mode_field(hankel, time_space, 1, hankel.plan.radial_count - 1)
    cutoff_plan = _plan(
        hankel,
        time_space,
        1.0,
        maximum_longitudinal_cutoff_fraction=1.0e-8,
    )
    cutoff = propagate_cylindrical_unidirectional(
        cutoff_plan.prepare(_law()),
        cutoff_field,
        InstantaneousScalarSusceptibility(),
        0.1,
    )

    assert cutoff.evidence.longitudinal_cutoff_fraction > 0.99
    assert cutoff.status == int(
        CylindricalUnidirectionalPropagationStatus.LONGITUDINAL_CUTOFF_LIMIT
    )

    resource_field = _mode_field(hankel, time_space, 8, 1)
    resource_plan = _plan(
        hankel,
        time_space,
        8.0,
        maximum_workspace_bytes=1,
    )
    resource = propagate_cylindrical_unidirectional(
        resource_plan.prepare(_law()),
        resource_field,
        InstantaneousScalarSusceptibility(),
        0.3,
    )

    assert resource.evidence.workspace_bytes > resource.evidence.maximum_workspace_bytes
    assert resource.status == int(
        CylindricalUnidirectionalPropagationStatus.RESOURCE_LIMIT
    )
    assert not resource.successful
    assert jnp.array_equal(resource.field.values, resource_field.values)
    assert (
        resource.field.longitudinal_coordinate == resource_field.longitudinal_coordinate
    )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.artifacts import ArtifactManifest
from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.geometry import RigidFrame
from phydrax.optics.materials import (
    AngularFrequencyValidity,
    CauchyRefractiveIndex,
    ConstantRefractiveIndex,
    medium_wavenumber,
    RefractiveIndexProvenance,
)
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._nonlinear_response import (
    AnalyticPulseField,
    InstantaneousScalarSusceptibility,
)
from phydrax.optics.wave._unidirectional import (
    prepare_unidirectional_propagation,
    propagate_unidirectional,
    UnidirectionalPropagationPlan,
    UnidirectionalPropagationStatus,
)


jax.config.update("jax_enable_x64", True)


def _provenance() -> RefractiveIndexProvenance:
    manifest = ArtifactManifest(
        artifact_id="nonlinear-optics-test-law",
        producer="phydrax-tests",
        version="1",
        sha256="0" * 64,
        byte_size=0,
        source_uri="generated://analytic-test-law",
        license_id="LicenseRef-PHYDRA",
        model="analytic refractive-index test law",
        coverage="unit-test frequencies",
    )
    return RefractiveIndexProvenance(manifest, record_id="analytic-test-law")


def _grids(spatial_count: int = 4, temporal_count: int = 64):
    plane_grid = TensorGridPlan(
        (FourierAxisSpec(spatial_count), FourierAxisSpec(spatial_count)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray([[-50.0, -50.0], [50.0, 50.0]]))
    temporal_grid = TensorGridPlan(
        (FourierAxisSpec(temporal_count),), axis_names=("time",)
    ).prepare(jnp.asarray([[0.0], [2.0 * jnp.pi]]))
    space = PlaneFieldSpace(plane_grid, RigidFrame.identity(3), "periodic-cell")
    return space, temporal_grid


def _constant_law(index=1.5):
    return ConstantRefractiveIndex(
        index,
        validity=AngularFrequencyValidity(0.5, 40.0),
        reference_wave_speed=1.0,
        provenance=_provenance(),
        law_id="constant-test-index",
    )


def _cauchy_law():
    return CauchyRefractiveIndex(
        jnp.asarray([1.4, 0.012]),
        1.0,
        validity=AngularFrequencyValidity(0.5, 40.0),
        reference_wave_speed=1.0,
        provenance=_provenance(),
        law_id="dispersive-test-index",
    )


def _mode_field(space, temporal_grid, mode: int, amplitude=1.0):
    time = temporal_grid.axes[0].nodes
    omega = float(mode)
    values = amplitude * jnp.exp(-1j * omega * time)
    values = jnp.broadcast_to(values, space.shape + temporal_grid.shape)
    return AnalyticPulseField(
        space, temporal_grid, values, omega, 0.0, polarization="scalar"
    )


def _plan(space, temporal_grid, omega, steps, **overrides):
    options = {
        "dealias_fraction": 1.0,
        "edge_guard_fraction": 0.05,
        "maximum_spectral_edge_fraction": 1.0,
        "maximum_analytic_signal_defect": 1.0,
        "maximum_nonlinear_rejected_fraction": 1.0,
        "maximum_refinement_error": 1.0,
        "maximum_backward_wave_estimate": 1.0,
    }
    options.update(overrides)
    return UnidirectionalPropagationPlan(
        space,
        temporal_grid,
        omega,
        polarization="scalar",
        step_count=steps,
        **options,
    )


def test_zero_susceptibility_has_linear_parity_and_dispersive_phase():
    space, temporal_grid = _grids()
    mode = 9
    field = _mode_field(space, temporal_grid, mode)
    law = _cauchy_law()
    plan = _plan(space, temporal_grid, float(mode), 8)
    prepared = prepare_unidirectional_propagation(plan, law)
    distance = jnp.asarray(0.37)

    result = propagate_unidirectional(
        prepared,
        field,
        InstantaneousScalarSusceptibility(0.0, 0.0),
        distance,
    )
    expected_phase = jnp.exp(1j * medium_wavenumber(law, mode) * distance)

    assert jnp.allclose(
        result.field.values, field.values * expected_phase, rtol=1.0e-10, atol=1.0e-11
    )
    assert result.evidence.fixed_step_refinement_error < 1.0e-11


def test_scalar_kerr_matches_b_integral_and_has_fourth_order_refinement():
    space, temporal_grid = _grids()
    mode = 12
    amplitude = 0.8
    chi3 = 0.1
    distance = 4.0
    field = _mode_field(space, temporal_grid, mode, amplitude)
    law = _constant_law()
    susceptibility = InstantaneousScalarSusceptibility(0.0, chi3)
    index = 1.5
    wave_number = index * mode
    nonlinear_wave_number = 3.0 * chi3 * mode**2 * amplitude**2 / (8.0 * wave_number)
    expected = field.values * jnp.exp(
        1j * (wave_number + nonlinear_wave_number) * distance
    )

    errors = []
    for steps in (4, 8, 16):
        prepared = prepare_unidirectional_propagation(
            _plan(
                space,
                temporal_grid,
                float(mode),
                steps,
                dealias_fraction=0.5,
            ),
            law,
        )
        result = propagate_unidirectional(prepared, field, susceptibility, distance)
        errors.append(
            jnp.linalg.norm(result.field.values - expected) / jnp.linalg.norm(expected)
        )

    assert errors[-1] < 1.0e-5
    assert errors[0] / errors[1] >= 8.0
    assert errors[1] / errors[2] >= 8.0


def test_phase_matched_chi2_preserves_manley_rowe_energy():
    space, temporal_grid = _grids()
    time = temporal_grid.axes[0].nodes
    fundamental_mode = 5
    values = 0.9 * jnp.exp(-1j * fundamental_mode * time) + 0.15j * jnp.exp(
        -2j * fundamental_mode * time
    )
    values = jnp.broadcast_to(values, space.shape + temporal_grid.shape)
    field = AnalyticPulseField(
        space,
        temporal_grid,
        values,
        float(fundamental_mode),
        0.0,
        polarization="scalar",
    )
    prepared = prepare_unidirectional_propagation(
        _plan(space, temporal_grid, float(fundamental_mode), 32),
        _constant_law(),
    )
    result = propagate_unidirectional(
        prepared,
        field,
        InstantaneousScalarSusceptibility(0.03, 0.0),
        0.5,
    )
    initial_spectrum = jnp.fft.ifft(field.values, axis=2, norm="ortho")
    final_spectrum = jnp.fft.ifft(result.field.values, axis=2, norm="ortho")
    initial_energy = jnp.sum(jnp.abs(initial_spectrum) ** 2)
    final_energy = jnp.sum(jnp.abs(final_spectrum) ** 2)

    # Constant n gives k(2 omega)=2 k(omega), and the conserved spectral
    # electric energy is the two-photon Manley--Rowe invariant in this scaling.
    assert jnp.abs(final_energy / initial_energy - 1.0) < 1.0e-4


def test_spectral_edge_violation_has_explicit_status():
    space, temporal_grid = _grids()
    field = _mode_field(space, temporal_grid, 31)
    plan = _plan(
        space,
        temporal_grid,
        31.0,
        4,
        edge_guard_fraction=0.1,
        maximum_spectral_edge_fraction=1.0e-4,
    )
    result = propagate_unidirectional(
        prepare_unidirectional_propagation(plan, _constant_law()),
        field,
        InstantaneousScalarSusceptibility(),
        0.1,
    )

    assert result.status == int(UnidirectionalPropagationStatus.SPECTRAL_EDGE_LIMIT)
    assert not result.successful
    assert result.evidence.spectral_edge_fraction > 0.99


def test_execution_is_deterministic_and_has_smooth_runtime_gradients():
    space, temporal_grid = _grids(4, 32)
    mode = 6
    law = _cauchy_law()
    plan = _plan(space, temporal_grid, float(mode), 4)
    prepared = prepare_unidirectional_propagation(plan, law)
    susceptibility = InstantaneousScalarSusceptibility(0.0, 0.02)
    field = _mode_field(space, temporal_grid, mode, 0.4)

    first = propagate_unidirectional(prepared, field, susceptibility, 0.2)
    second = propagate_unidirectional(prepared, field, susceptibility, 0.2)
    assert jnp.array_equal(first.field.values, second.field.values)
    assert jnp.array_equal(first.status, second.status)

    carrier = field.values / 0.4

    def objective(amplitude, chi3, distance):
        varied_field = AnalyticPulseField(
            space,
            temporal_grid,
            amplitude * carrier,
            float(mode),
            0.0,
            polarization="scalar",
        )
        varied_response = InstantaneousScalarSusceptibility(0.0, chi3)
        propagated = propagate_unidirectional(
            prepared, varied_field, varied_response, distance
        )
        return jnp.real(propagated.field.values[0, 0, 0])

    gradients = jax.grad(objective, argnums=(0, 1, 2))(0.4, 0.02, 0.2)
    assert all(jnp.isfinite(value) for value in gradients)
    assert all(value != 0.0 for value in gradients)

    def dispersion_objective(coefficients):
        varied_law = eqx.tree_at(
            lambda selected: selected.coefficients, law, coefficients
        )
        varied_prepared = prepare_unidirectional_propagation(plan, varied_law)
        propagated = propagate_unidirectional(varied_prepared, field, susceptibility, 0.2)
        return jnp.real(propagated.field.values[0, 0, 0])

    dispersion_gradient = jax.grad(dispersion_objective)(law.coefficients)
    assert jnp.all(jnp.isfinite(dispersion_gradient))
    assert jnp.any(dispersion_gradient != 0.0)

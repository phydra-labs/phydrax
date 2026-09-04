import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.applications.cosmology._native_boltzmann import (
    _line_of_sight_transfers,
)


cosmology = phx.applications.cosmology


def _prepared(*, scale_count=40, line_of_sight_quadrature_tolerance=1.0e-2):
    scale_contract = cosmology.CosmologyScaleContract(
        cosmology.CODE_COSMOLOGY_SCALE.length_unit,
        cosmology.CODE_COSMOLOGY_SCALE.mass_unit,
        cosmology.CODE_COSMOLOGY_SCALE.time_unit,
    )
    background = cosmology.FLRWBackground(
        1.0,
        0.3,
        radiation_density=1.0e-3,
        scale=scale_contract,
    )
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="scalar-thermodynamics",
        numerical_policy_id="fixed-grid",
        physics_policy_id="synthetic-visibility",
        scale_id=scale_contract.scale_id,
        source_kind="native",
        differentiation="native-parameter",
    )
    scale = jnp.geomspace(1.0e-3, 1.0, scale_count)
    visibility = jnp.exp(-0.5 * ((scale - 0.1) / 0.03) ** 2)
    thermodynamics = cosmology.ThermodynamicsHistory(
        scale,
        jnp.ones_like(scale),
        1.0 / scale,
        2.0 * visibility,
        visibility,
        scale_contract,
        provenance,
        background.realization,
    )
    layout = cosmology.ScalarHierarchyLayout(
        photon_order=3,
        polarization_order=3,
        relic_order=3,
    )
    transitions = cosmology.ApproximationTransitionPolicy(
        tight_coupling_exit=0.02,
        radiation_streaming_entry=0.3,
        overlap_tolerance=1.0e30,
    )
    return cosmology.ScalarEinsteinBoltzmannPlan(
        background,
        thermodynamics,
        jnp.asarray([0.05, 0.1, 0.2]),
        layout,
        transitions,
        jnp.asarray([2, 3]),
        constraint_tolerance=1.0e30,
        overlap_tolerance=1.0e30,
        tail_tolerance=1.0e30,
        line_of_sight_quadrature_tolerance=line_of_sight_quadrature_tolerance,
    ).prepare()


def test_native_scalar_solver_generates_operators_transfer_and_tt_te_ee():
    prepared = _prepared()
    result = prepared.solve(jnp.asarray([2.0e-9, 2.0e-9, 2.0e-9]))
    assert result.transfer.states.shape[:2] == (40, 3)
    assert result.temperature_source.shape == (3, 40)
    assert result.polarization_source.shape == (3, 40)
    assert result.temperature_transfer.shape == (2, 3)
    assert result.polarization_transfer.shape == (2, 3)
    assert result.transfer_table.descriptor.fields == ("cold_baryon", "total_matter")
    assert result.transfer_table.descriptor.gauge == "synchronous"
    assert result.cmb_spectra.modes == ("scalar",)
    assert result.cmb_spectra.lensing_state == "unlensed"
    assert result.cmb_spectra.spectra.shape == (1, 2, 4, 4)
    assert jnp.allclose(
        result.cmb_spectra.spectra,
        jnp.swapaxes(result.cmb_spectra.spectra, -1, -2),
    )
    assert bool(result.evidence.transition_schedule_valid)
    assert bool(result.evidence.finite)
    assert bool(result.successful)


def test_line_of_sight_error_includes_polarization_underresolution():
    delta_time = jnp.asarray([1.0, 1.0])
    radial = jnp.ones((1, 1, 3))
    temperature_source = jnp.full((3, 1), 2.0)
    polarization_source = jnp.asarray([[0.0], [0.0], [2.0]])

    temperature, polarization, error, finite = _line_of_sight_transfers(
        delta_time,
        temperature_source,
        polarization_source,
        radial,
    )

    assert bool(finite)
    assert jnp.allclose(temperature, 4.0)
    assert jnp.allclose(polarization, 1.0)
    assert jnp.allclose(error, 1.0)
    assert not bool(error <= 0.5)

    _, _, nonfinite_error, nonfinite = _line_of_sight_transfers(
        delta_time,
        temperature_source,
        polarization_source.at[-1, 0].set(jnp.nan),
        radial,
    )
    assert not bool(nonfinite)
    assert not bool(jnp.isfinite(nonfinite_error))


def test_native_scalar_solver_gates_success_on_line_of_sight_error():
    primordial = jnp.asarray([2.0e-9, 2.0e-9, 2.0e-9])
    tolerance = 5.0e-3
    resolved = _prepared(line_of_sight_quadrature_tolerance=tolerance)
    reference = _prepared(line_of_sight_quadrature_tolerance=1.0)
    resolved_result = resolved.solve(primordial)
    reference_result = reference.solve(primordial)

    assert resolved.plan.plan_id != reference.plan.plan_id
    assert resolved.prepared_id != reference.prepared_id
    assert bool(resolved_result.evidence.successful)
    assert bool(resolved_result.successful)
    assert resolved_result.evidence.line_of_sight_quadrature_error <= tolerance
    assert jnp.allclose(
        resolved_result.cmb_spectra.spectra,
        reference_result.cmb_spectra.spectra,
    )

    underresolved_result = _prepared(
        scale_count=8, line_of_sight_quadrature_tolerance=tolerance
    ).solve(primordial)
    assert bool(underresolved_result.evidence.finite)
    assert underresolved_result.evidence.line_of_sight_quadrature_error > tolerance
    assert not bool(underresolved_result.evidence.successful)
    assert not bool(underresolved_result.successful)

    prepared = _prepared(line_of_sight_quadrature_tolerance=tolerance)
    nonfinite = eqx.tree_at(
        lambda candidate: candidate.conformal_times,
        prepared,
        prepared.conformal_times.at[1].set(jnp.nan),
    )
    with pytest.raises(ValueError):
        nonfinite.solve(primordial)


def test_native_scalar_solver_is_jittable_and_parameter_differentiable():
    prepared = _prepared()
    solve = jax.jit(prepared.solve)
    result = solve(jnp.asarray([2.0e-9, 2.0e-9, 2.0e-9]))
    assert bool(result.evidence.finite)

    def amplitude_response(amplitude):
        output = prepared.solve(amplitude * jnp.ones((3,)))
        return output.cmb_spectra.spectra[0, 0, 0, 0]

    derivative = jax.grad(amplitude_response)(jnp.asarray(2.0e-9))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0

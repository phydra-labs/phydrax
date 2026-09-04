#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.artifacts import ArtifactManifest
from phydrax.optics.materials import (
    AngularFrequencyValidity,
    CauchyRefractiveIndex,
    ConstantRefractiveIndex,
    evaluate_refractive_index,
    LorentzDrudeRefractiveIndex,
    lower_to_frequency_maxwell_material,
    lower_to_geometric_index,
    medium_wavenumber,
    RefractiveIndexProvenance,
    SellmeierRefractiveIndex,
    TabulatedComplexRefractiveIndex,
)


def _provenance() -> RefractiveIndexProvenance:
    manifest = ArtifactManifest(
        artifact_id="unit-test-material-record",
        producer="phydrax-tests",
        version="1",
        sha256="0" * 64,
        byte_size=0,
        source_uri="memory://unit-test-material-record",
        license_id="internal-test",
        model="scalar-refractive-index",
        coverage="synthetic parameters",
    )
    return RefractiveIndexProvenance(manifest, record_id="record-0")


def _validity(*, extrapolation: str = "reject") -> AngularFrequencyValidity:
    return AngularFrequencyValidity(1.0, 10.0, extrapolation=extrapolation)


def _constant(
    value: complex | float,
    *,
    extrapolation: str = "reject",
) -> ConstantRefractiveIndex:
    return ConstantRefractiveIndex(
        value,
        validity=_validity(extrapolation=extrapolation),
        reference_wave_speed=3.0,
        provenance=_provenance(),
        law_id="constant",
    )


def test_cauchy_and_sellmeier_reference_formulas_use_angular_frequency() -> None:
    wave_speed = 3.0e8
    wavelength = 0.5e-6
    omega = 2.0 * np.pi * wave_speed / wavelength
    validity = AngularFrequencyValidity(0.5 * omega, 1.5 * omega)
    cauchy = CauchyRefractiveIndex(
        jnp.asarray([1.5, 0.01]),
        1.0e-6,
        validity=validity,
        reference_wave_speed=wave_speed,
        provenance=_provenance(),
        law_id="cauchy",
    )
    sellmeier = SellmeierRefractiveIndex(
        jnp.asarray([1.0]),
        jnp.asarray([0.2e-6]),
        validity=validity,
        reference_wave_speed=wave_speed,
        provenance=_provenance(),
        law_id="sellmeier",
    )
    np.testing.assert_allclose(
        evaluate_refractive_index(cauchy, omega).refractive_index,
        1.54,
        rtol=2e-6,
    )
    expected_sellmeier = np.sqrt(1.0 + wavelength**2 / (wavelength**2 - (0.2e-6) ** 2))
    np.testing.assert_allclose(
        evaluate_refractive_index(sellmeier, omega).refractive_index,
        expected_sellmeier,
        rtol=2e-6,
    )


def test_lorentz_drude_formula_selects_passive_square_root() -> None:
    law = LorentzDrudeRefractiveIndex(
        4.0,
        jnp.asarray([3.0, 2.0]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([0.0, 0.5]),
        validity=_validity(),
        reference_wave_speed=3.0,
        provenance=_provenance(),
        law_id="lorentz-drude",
    )
    omega = 2.0
    epsilon = 4.0 + 3.0 / (1.0 - omega**2) + 2.0 / (-(omega**2) - 1j * 0.5 * omega)
    value = evaluate_refractive_index(law, omega).refractive_index
    np.testing.assert_allclose(value**2, epsilon, rtol=2e-6)
    assert float(jnp.imag(value)) >= 0.0
    assert law.passive_branch == "positive-imaginary"


def test_tabulated_complex_interpolation_and_linear_continuation() -> None:
    law = TabulatedComplexRefractiveIndex(
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([1.0 + 0.1j, 2.0 + 0.2j, 4.0 + 0.4j]),
        validity=AngularFrequencyValidity(1.0, 3.0, extrapolation="continue"),
        reference_wave_speed=3.0,
        provenance=_provenance(),
        law_id="table",
        passive_branch="positive-imaginary",
    )
    evaluation = evaluate_refractive_index(law, jnp.asarray([1.5, 3.5]))
    np.testing.assert_allclose(
        evaluation.refractive_index,
        np.asarray([1.5 + 0.15j, 5.0 + 0.5j]),
    )
    np.testing.assert_array_equal(evaluation.status, np.asarray([0, 2]))
    np.testing.assert_array_equal(evaluation.accepted, np.asarray([True, True]))


def test_validity_rejection_and_clamping_are_status_bearing() -> None:
    rejected = evaluate_refractive_index(_constant(1.5), jnp.asarray([0.0, 0.5, 2.0]))
    np.testing.assert_array_equal(rejected.accepted, np.asarray([False, False, True]))
    np.testing.assert_array_equal(rejected.status, np.asarray([3, 3, 0]))
    np.testing.assert_array_equal(
        rejected.extrapolated, np.asarray([False, False, False])
    )
    assert np.isnan(np.asarray(rejected.refractive_index[:2])).all()

    clamped = evaluate_refractive_index(
        _constant(1.5, extrapolation="clamp"), jnp.asarray([0.5, 11.0])
    )
    np.testing.assert_array_equal(clamped.accepted, np.asarray([True, True]))
    np.testing.assert_array_equal(clamped.status, np.asarray([1, 1]))
    np.testing.assert_array_equal(clamped.extrapolated, np.asarray([True, True]))
    np.testing.assert_allclose(clamped.evaluated_angular_frequency, [1.0, 10.0])


def test_medium_wavenumber_uses_explicit_reference_wave_speed() -> None:
    law = _constant(1.5)
    np.testing.assert_allclose(medium_wavenumber(law, 6.0), 3.0)


def test_cauchy_frequency_gradient_matches_closed_form() -> None:
    law = CauchyRefractiveIndex(
        jnp.asarray([1.5, 0.01]),
        2.0,
        validity=_validity(),
        reference_wave_speed=3.0,
        provenance=_provenance(),
        law_id="differentiable-cauchy",
    )

    def evaluated_index(omega: jax.Array) -> jax.Array:
        return jnp.real(evaluate_refractive_index(law, omega).refractive_index)

    omega = jnp.asarray(4.0)
    expected = 2.0 * 0.01 * (2.0 / (2.0 * np.pi * 3.0)) ** 2 * omega
    np.testing.assert_allclose(jax.grad(evaluated_index)(omega), expected, rtol=2e-6)


def test_geometric_lowering_rejects_loss_without_dropping_it() -> None:
    lowering = lower_to_geometric_index(_constant(1.5 + 0.01j), 2.0)
    assert not bool(lowering.accepted)
    assert int(lowering.status) == 2
    assert np.isnan(float(lowering.refractive_index))
    np.testing.assert_allclose(lowering.imaginary_magnitude, 0.01)


def test_maxwell_lowering_is_isotropic_nonmagnetic_epsilon_n_squared() -> None:
    law = _constant(1.5 + 0.1j)
    material = lower_to_frequency_maxwell_material(law, 2.0, material_id="sampled-index")
    np.testing.assert_allclose(material.permittivity, (1.5 + 0.1j) ** 2)
    np.testing.assert_allclose(material.permeability, 1.0)
    np.testing.assert_allclose(material.magnetoelectric_xi, 0.0)
    np.testing.assert_allclose(material.magnetoelectric_zeta, 0.0)
    assert material.reciprocal is True
    assert material.passive is True
    assert material.origin_evidence_id == law.provenance.provenance_id


def test_maxwell_lowering_derives_passivity_from_epsilon_not_index_sheet() -> None:
    law = _constant(-1.0 + 0.1j)
    material = lower_to_frequency_maxwell_material(
        law, 2.0, material_id="negative-index-active-epsilon"
    )

    np.testing.assert_allclose(material.permittivity, (-1.0 + 0.1j) ** 2)
    assert float(jnp.imag(material.permittivity)) < 0.0
    assert material.passive is False


def test_passive_tabulated_branch_rejects_gain_samples() -> None:
    with pytest.raises(ValueError, match="Im.n."):
        TabulatedComplexRefractiveIndex(
            jnp.asarray([1.0, 2.0]),
            jnp.asarray([1.0 - 0.1j, 1.1 + 0.1j]),
            validity=AngularFrequencyValidity(1.0, 2.0),
            reference_wave_speed=3.0,
            provenance=_provenance(),
            law_id="gain-table",
            passive_branch="positive-imaginary",
        )


def test_passive_constant_uses_positive_real_tie_break() -> None:
    with pytest.raises(ValueError, match="Re.n."):
        ConstantRefractiveIndex(
            -1.0,
            validity=_validity(),
            reference_wave_speed=3.0,
            provenance=_provenance(),
            law_id="wrong-sheet",
            passive_branch="positive-imaginary",
        )

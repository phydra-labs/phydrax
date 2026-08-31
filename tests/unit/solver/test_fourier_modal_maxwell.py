from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _harmonics():
    return LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )


def _boundary_policy() -> fm.BoundaryCascadePolicy:
    return fm.BoundaryCascadePolicy(
        doublings=6,
        initializer_order=7,
        paired_error=True,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-10,
    )


def test_fresnel_interface_complex_amplitudes_and_power() -> None:
    harmonics = _harmonics()
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    dielectric = fm.FrequencyMaxwellMaterial(4.0, material_id="dielectric")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (),
        fm.HomogeneousMaxwellPort(dielectric, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem)
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "te",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    assert int(result.status) == int(fm.FourierModalSolveStatus.SUCCESS)
    assert float(result.reflected_power[0]) == pytest.approx(
        1.0 / 9.0, rel=2e-6, abs=2e-7
    )
    assert float(result.transmitted_power[0]) == pytest.approx(
        8.0 / 9.0, rel=2e-6, abs=2e-7
    )


def test_lossless_film_conserves_power_and_reconstructs_fields() -> None:
    harmonics = _harmonics()
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    film_material = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    layer = fm.FourierModalLayer(
        film_material,
        0.2,
        fm.DirectFourierFactorizationPlan(),
        layer_id="film",
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (layer,),
        fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(
        problem,
        fm.FourierModalSolvePolicy(boundary=_boundary_policy()),
    )
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "tm",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    np.testing.assert_allclose(
        np.asarray(result.reflected_power + result.transmitted_power),
        np.asarray(result.incident_power),
        rtol=1e-7,
        atol=1e-9,
    )
    field = fm.fields_in_layer(prepared, result, 0, 0.1)
    assert field.electric_field.shape == harmonics.sample_shape + (3, 1)
    assert field.magnetic_field.shape == harmonics.sample_shape + (3, 1)
    assert bool(jnp.all(jnp.isfinite(field.electric_field)))
    farfield = fm.diffraction_order_far_field(prepared, result)
    assert farfield.power.shape == (1, 2, 1)
    assert bool(jnp.all(farfield.propagating))


def test_full_tensor_layer_operator_matches_finite_contract() -> None:
    harmonics = _harmonics()
    epsilon = jnp.asarray(
        (
            (2.0 + 0.0j, 0.0, 0.2),
            (0.0, 2.4 + 0.0j, 0.0),
            (0.2, 0.0, 2.8 + 0.0j),
        )
    )
    material = fm.FrequencyMaxwellMaterial(epsilon, material_id="anisotropic")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.2, 0.0)),
    )
    assert operator.matrix.shape == (4, 4)
    assert bool(operator.diagnostics.finite)
    assert float(operator.diagnostics.constitutive_residual) < 1e-10


def test_zero_thickness_boundary_is_identity() -> None:
    harmonics = _harmonics()
    material = fm.FrequencyMaxwellMaterial(2.0, material_id="uniform")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.0, 0.0)),
    )
    relation = fm.prepare_layer_boundary(operator, 0.0, _boundary_policy())
    np.testing.assert_allclose(np.asarray(relation.a), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.asarray(relation.d), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.asarray(relation.b), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(relation.c), 0.0, atol=1e-12)

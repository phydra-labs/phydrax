from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spectral import BrillouinZonePlan, LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _source_problem():
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    host = fm.FrequencyMaxwellMaterial(2.25, material_id="host")
    left = fm.FourierModalLayer(
        host,
        0.1,
        fm.DirectFourierFactorizationPlan(),
        layer_id="left-half",
    )
    right = fm.FourierModalLayer(
        host,
        0.1,
        fm.DirectFourierFactorizationPlan(),
        layer_id="right-half",
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (left, fm.FourierModalSourcePlane("source"), right),
        fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
    )
    policy = fm.FourierModalSolvePolicy(
        boundary=fm.BoundaryCascadePolicy(
            doublings=6,
            initializer_order=7,
            paired_error=False,
            relative_tolerance=1e-7,
        )
    )
    return harmonics, problem, policy


def test_internal_current_emits_to_both_ports_and_many_rhs_match() -> None:
    harmonics, problem, policy = _source_problem()
    prepared = fm.prepare_fourier_modal_maxwell(problem, policy)
    coefficient = fm.point_source_coefficients(
        harmonics,
        problem.bloch_wavevector,
        jnp.asarray((0.25, 0.0)),
    )
    current = jnp.zeros((3, 1, 2), dtype=jnp.complex128)
    current = current.at[1, 0, :].set(jnp.asarray((coefficient[0], 2.0 * coefficient[0])))
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    excitation = fm.FourierModalExcitation(
        zero,
        zero,
        source_ids=("source",),
        electric_currents=(current,),
        magnetic_currents=(jnp.zeros_like(current),),
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    assert bool(jnp.all(result.reflected_power > 0.0))
    assert bool(jnp.all(result.transmitted_power > 0.0))
    np.testing.assert_allclose(
        np.asarray(result.right_outgoing[:, 1]),
        2.0 * np.asarray(result.right_outgoing[:, 0]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_thickness_refresh_reuses_material_and_operator() -> None:
    _, problem, policy = _source_problem()
    prepared = fm.prepare_fourier_modal_maxwell(problem, policy)
    left, source, right = problem.elements
    assert isinstance(left, fm.FourierModalLayer)
    assert isinstance(right, fm.FourierModalLayer)
    updated_left = fm.FourierModalLayer(
        left.material,
        0.11,
        left.factorization,
        layer_id=left.layer_id,
    )
    updated = fm.FourierModalMaxwellProblem(
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
        problem.superstrate,
        (updated_left, source, right),
        problem.substrate,
        numeric_version="thickness",
    )
    refreshed = fm.refresh_fourier_modal_maxwell(
        prepared,
        updated,
        fm.FourierModalRefreshSpec(("thickness", "unchanged")),
    )
    old_layers = tuple(
        value
        for value in prepared.elements
        if isinstance(value, fm.PreparedFourierModalLayer)
    )
    new_layers = tuple(
        value
        for value in refreshed.elements
        if isinstance(value, fm.PreparedFourierModalLayer)
    )
    np.testing.assert_allclose(
        np.asarray(new_layers[0].operator.matrix),
        np.asarray(old_layers[0].operator.matrix),
    )
    assert refreshed.refresh_count == 1


def test_brillouin_field_and_power_reductions_are_explicit() -> None:
    harmonics, _, _ = _source_problem()
    rule = BrillouinZonePlan((3,)).prepare(harmonics)
    values = jnp.asarray(((1.0,), (2.0,), (4.0,)))
    expected = jnp.mean(values, axis=0)
    np.testing.assert_allclose(
        np.asarray(fm.integrate_brillouin_fields(values, rule)),
        np.asarray(expected),
    )
    np.testing.assert_allclose(
        np.asarray(fm.integrate_brillouin_power(values, rule)),
        np.asarray(expected),
    )


def test_translation_refresh_reuses_shared_base_convolution() -> None:
    _, problem, policy = _source_problem()
    prepared = fm.prepare_fourier_modal_maxwell(problem, policy)
    old_layers = tuple(
        value
        for value in prepared.elements
        if isinstance(value, fm.PreparedFourierModalLayer)
    )
    assert old_layers[0].base_material is old_layers[1].base_material
    left, source, right = problem.elements
    assert isinstance(left, fm.FourierModalLayer)
    translated_left = fm.FourierModalLayer(
        left.material,
        left.thickness,
        left.factorization,
        translation=jnp.asarray((0.2, 0.0)),
        layer_id=left.layer_id,
    )
    updated = fm.FourierModalMaxwellProblem(
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
        problem.superstrate,
        (translated_left, source, right),
        problem.substrate,
        numeric_version="translation",
    )
    refreshed = fm.refresh_fourier_modal_maxwell(
        prepared,
        updated,
        fm.FourierModalRefreshSpec(("translation", "unchanged")),
    )
    translated_layers = tuple(
        value
        for value in refreshed.elements
        if isinstance(value, fm.PreparedFourierModalLayer)
    )
    assert translated_layers[0].base_material is old_layers[0].base_material
    expected = fm.translate_prepared_fourier_material(
        old_layers[0].base_material,
        problem.harmonics,
        jnp.asarray((0.2, 0.0)),
    )
    np.testing.assert_allclose(
        np.asarray(translated_layers[0].material.permittivity),
        np.asarray(expected.permittivity),
    )


def test_brillouin_case_batch_preserves_case_and_rhs_axes() -> None:
    harmonics, _, _ = _source_problem()
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="batch-vacuum")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (),
        fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
    )
    rule = BrillouinZonePlan((2,)).prepare(harmonics)
    prepared = fm.prepare_brillouin_zone_maxwell(problem, rule)
    excitations = tuple(
        fm.plane_wave_excitation(
            case.scattering,
            harmonics.plan.layout.mode_ids[0],
            "te",
        )
        for case in prepared.cases
    )
    result = fm.solve_fourier_modal_case_batch(prepared, excitations)
    assert result.right_outgoing.shape == (2, 2, 1)
    assert result.status.shape == (2,)
    assert bool(jnp.all(result.status == int(fm.FourierModalSolveStatus.SUCCESS)))

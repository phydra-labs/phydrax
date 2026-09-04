#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _inputs(gradient, widths=(1.0, 1.0, 1.0)):
    return phx.equations.AlgebraicLESInputs(
        jnp.asarray(gradient),
        phx.equations.LESFilterScale(jnp.asarray(widths)),
    )


def _resolved_filter(
    name="cell-average",
    *,
    family="implicit-grid-volume",
    axis_names=("x", "y", "z"),
    topology="tensor-product",
    boundary_class="periodic",
    scale_rule="volume-equivalent",
    commutation_status="unmodeled",
    repeated_filter_semantics="unmodeled",
):
    return phx.equations.ResolvedLESFilter(
        name,
        family=family,
        axis_names=axis_names,
        topology=topology,
        boundary_class=boundary_class,
        scale_rule=scale_rule,
        commutation_status=commutation_status,
        repeated_filter_semantics=repeated_filter_semantics,
    )


def _provenance(
    filter_name="cell-average",
    discretization_id="grid-a",
    regime="wall-resolved",
    *,
    source_kind="user",
    evidence_ids=(),
):
    return phx.equations.LESParameterProvenance(
        _resolved_filter(filter_name),
        discretization_id,
        regime,
        source_kind=source_kind,
        evidence_ids=evidence_ids,
    )


def test_les_filter_scale_and_provenance_validate_physical_semantics():
    widths = jnp.asarray(((1.0, 2.0, 4.0), (3.0, 6.0, 12.0)))
    scale = phx.equations.LESFilterScale(widths)
    np.testing.assert_allclose(scale.directional_widths, widths)
    np.testing.assert_allclose(scale.equivalent_width, np.cbrt((8.0, 216.0)))

    for invalid in (
        jnp.ones((2,)),
        jnp.ones((3, 1)),
        jnp.asarray((1.0, 0.0, 1.0)),
    ):
        with pytest.raises(ValueError):
            phx.equations.LESFilterScale(invalid)
    with pytest.raises(ValueError):
        phx.equations.LESFilterScale(jnp.asarray((1.0, jnp.inf, 1.0)))
    with pytest.raises(TypeError):
        _resolved_filter(object())

    resolved_filter = _resolved_filter()
    assert resolved_filter.dimension == 3
    assert resolved_filter.axis_names == ("x", "y", "z")
    assert resolved_filter.filter_id == _resolved_filter().filter_id
    assert resolved_filter.filter_id != _resolved_filter(name="box").filter_id
    assert (
        resolved_filter.filter_id
        != _resolved_filter(axis_names=("z", "x", "y")).filter_id
    )
    assert (
        resolved_filter.filter_id != _resolved_filter(topology="unstructured").filter_id
    )
    assert (
        resolved_filter.filter_id
        != _resolved_filter(boundary_class="wall-bounded").filter_id
    )
    assert (
        resolved_filter.filter_id
        != _resolved_filter(commutation_status="modeled").filter_id
    )
    assert (
        resolved_filter.filter_id
        != _resolved_filter(repeated_filter_semantics="composed").filter_id
    )
    explicit_filter = _resolved_filter(
        family="explicit-filter", scale_rule="kernel-equivalent"
    )
    assert resolved_filter.filter_id != explicit_filter.filter_id
    sharp = _resolved_filter(
        family="sharp-fourier-projection",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    assert sharp.family == "sharp-fourier-projection"
    with pytest.raises(ValueError):
        _resolved_filter(
            family="sharp-fourier-projection",
            scale_rule="cutoff-equivalent",
        )

    first = _provenance()
    assert first.provenance_id == _provenance().provenance_id
    assert first.provenance_id != _provenance(filter_name="box").provenance_id
    assert first.provenance_id != _provenance(discretization_id="grid-b").provenance_id
    assert first.provenance_id != _provenance(regime="wall-modelled").provenance_id
    literature = _provenance(
        source_kind="literature", evidence_ids=("doi:10.example/les",)
    )
    assert first.provenance_id != literature.provenance_id
    assert literature.evidence_ids == ("doi:10.example/les",)
    with pytest.raises(ValueError):
        _provenance(source_kind="a-priori")
    with pytest.raises(TypeError):
        phx.equations.LESParameterProvenance(
            object(),
            "grid-a",
            "resolved",
            source_kind="user",
            evidence_ids=(),
        )


def test_model_and_prepared_identities_separate_formula_from_binding():
    model = phx.equations.SmagorinskyLESPlan(jnp.asarray(0.17))
    changed = phx.equations.SmagorinskyLESPlan(jnp.asarray(0.23))
    assert model.model_id == changed.model_id
    assert model.model_id != phx.equations.WALELESPlan(0.17).model_id

    provenance = _provenance()
    prepared = model.prepare(provenance)
    assert isinstance(prepared, phx.equations.PreparedAlgebraicLESModel)
    assert prepared.coefficient == pytest.approx(0.17)
    assert prepared.model_id == model.model_id
    assert prepared.provenance is provenance
    assert prepared.prepared_id == model.prepare(provenance).prepared_id
    assert prepared.prepared_id != changed.prepare(provenance).prepared_id
    assert (
        prepared.prepared_id
        != model.prepare(_provenance(regime="wall-modelled")).prepared_id
    )

    for model_type in (
        phx.equations.SmagorinskyLESPlan,
        phx.equations.WALELESPlan,
        phx.equations.VremanLESPlan,
        phx.equations.AMDLESPlan,
    ):
        with pytest.raises(TypeError):
            model_type()
        with pytest.raises(ValueError):
            model_type(-0.1)
        with pytest.raises(ValueError):
            model_type(jnp.asarray((0.1, 0.2)))


def test_smagorinsky_formula_uses_full_strain_and_volume_equivalent_width():
    gradient = np.diag((1.0, -0.5, 0.25))
    widths = np.asarray((2.0, 3.0, 4.0))
    coefficient = 0.2
    result = phx.equations.SmagorinskyLESPlan(coefficient).evaluate(
        _inputs(gradient, widths)
    )

    strain_squared = np.sum(gradient * gradient)
    delta = np.cbrt(np.prod(widths))
    expected_viscosity = (coefficient * delta) ** 2 * np.sqrt(2.0 * strain_squared)
    deviatoric = gradient - np.trace(gradient) * np.eye(3) / 3.0
    expected_stress = -2.0 * expected_viscosity * deviatoric
    np.testing.assert_allclose(result.kinematic_viscosity, expected_viscosity, rtol=2e-6)
    np.testing.assert_allclose(
        result.specific_deviatoric_stress, expected_stress, rtol=2e-6
    )
    np.testing.assert_allclose(
        result.energy_transfer, -np.sum(expected_stress * gradient), rtol=2e-6
    )


def test_wale_formula_value_uses_full_strain_denominator():
    gradient = np.asarray(((0.4, -0.2, 0.1), (0.3, -0.1, 0.2), (-0.2, 0.5, -0.3)))
    coefficient = 0.31
    width = 0.12
    result = phx.equations.WALELESPlan(coefficient).evaluate(
        _inputs(gradient, (width, width, width))
    )

    strain = 0.5 * (gradient + gradient.T)
    squared = gradient @ gradient
    symmetric_squared = 0.5 * (squared + squared.T)
    deviatoric_squared = symmetric_squared - np.trace(symmetric_squared) * np.eye(3) / 3.0
    strain_invariant = np.sum(strain * strain)
    squared_invariant = np.sum(deviatoric_squared * deviatoric_squared)
    expected = (
        (coefficient * width) ** 2
        * squared_invariant**1.5
        / (strain_invariant**2.5 + squared_invariant**1.25)
    )
    np.testing.assert_allclose(result.kinematic_viscosity, expected, rtol=3e-6)


def test_vreman_directional_metric_and_amd_positive_branch_values():
    gradient = np.diag((1.0, 2.0, 0.0))
    widths = np.asarray((1.5, 0.75, 4.0))
    coefficient = 0.08
    result = phx.equations.VremanLESPlan(coefficient).evaluate(_inputs(gradient, widths))
    beta_diagonal = (np.diag(gradient) * widths) ** 2
    invariant = (
        beta_diagonal[0] * beta_diagonal[1]
        + beta_diagonal[0] * beta_diagonal[2]
        + beta_diagonal[1] * beta_diagonal[2]
    )
    expected = coefficient * np.sqrt(invariant / np.sum(gradient * gradient))
    np.testing.assert_allclose(result.kinematic_viscosity, expected, rtol=2e-6)

    compression = np.diag((-2.0, 0.0, 0.0))
    amd = phx.equations.AMDLESPlan(0.3).evaluate(_inputs(compression, (0.4, 0.8, 1.2)))
    expected_amd = 0.3 * (0.4**2 * 2.0**2 * (4.0 / 3.0)) / 2.0**2
    np.testing.assert_allclose(amd.kinematic_viscosity, expected_amd, rtol=2e-6)


def test_exact_zero_branches_have_finite_zero_jvps():
    zero = jnp.zeros((3, 3))
    direction = jnp.asarray(((0.2, -0.1, 0.3), (0.4, -0.2, 0.1), (-0.3, 0.2, 0.5)))
    scale = phx.equations.LESFilterScale(jnp.asarray((0.2, 0.3, 0.4)))
    model_types = (
        phx.equations.SmagorinskyLESPlan,
        phx.equations.WALELESPlan,
        phx.equations.VremanLESPlan,
        phx.equations.AMDLESPlan,
    )
    for model_type in model_types:
        model = model_type(0.2)

        def viscosity(gradient):
            inputs = phx.equations.AlgebraicLESInputs(gradient, scale)
            return model.evaluate(inputs).kinematic_viscosity

        primal, tangent = jax.jvp(viscosity, (zero,), (direction,))
        assert primal == 0.0
        assert tangent == 0.0
        assert jnp.isfinite(tangent)
        result = model.evaluate(phx.equations.AlgebraicLESInputs(zero, scale))
        np.testing.assert_array_equal(
            result.specific_deviatoric_stress, jnp.zeros((3, 3))
        )
        assert result.energy_transfer == 0.0

    isotropic = jnp.eye(3) * 0.5
    smagorinsky = phx.equations.SmagorinskyLESPlan(0.2).evaluate(
        phx.equations.AlgebraicLESInputs(isotropic, scale)
    )
    assert smagorinsky.kinematic_viscosity > 0.0
    np.testing.assert_array_equal(
        smagorinsky.specific_deviatoric_stress, jnp.zeros((3, 3))
    )
    assert smagorinsky.energy_transfer == 0.0
    assert (
        phx.equations.WALELESPlan(0.2)
        .evaluate(phx.equations.AlgebraicLESInputs(isotropic, scale))
        .kinematic_viscosity
        == 0.0
    )
    assert (
        phx.equations.AMDLESPlan(0.2)
        .evaluate(phx.equations.AlgebraicLESInputs(isotropic, scale))
        .kinematic_viscosity
        == 0.0
    )

    rank_one = jnp.asarray(((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert (
        phx.equations.VremanLESPlan(0.2)
        .evaluate(phx.equations.AlgebraicLESInputs(rank_one, scale))
        .kinematic_viscosity
        == 0.0
    )
    expansion = jnp.diag(jnp.asarray((1.0, 0.0, 0.0)))
    assert (
        phx.equations.AMDLESPlan(0.2)
        .evaluate(phx.equations.AlgebraicLESInputs(expansion, scale))
        .kinematic_viscosity
        == 0.0
    )


def test_stress_is_symmetric_trace_free_and_has_the_declared_sign():
    gradient = jnp.asarray(((-1.0, 0.2, 0.0), (0.1, -0.5, 0.3), (0.0, -0.1, 0.2)))
    inputs = _inputs(gradient, (0.2, 0.35, 0.5))
    for model in (
        phx.equations.SmagorinskyLESPlan(0.16),
        phx.equations.WALELESPlan(0.32),
        phx.equations.VremanLESPlan(0.07),
        phx.equations.AMDLESPlan(0.3),
    ):
        result = model.evaluate(inputs)
        stress = result.specific_deviatoric_stress
        np.testing.assert_allclose(stress, stress.T, atol=2e-7)
        np.testing.assert_allclose(jnp.trace(stress), 0.0, atol=2e-7)
        strain = 0.5 * (gradient + gradient.T)
        np.testing.assert_allclose(
            result.energy_transfer, -jnp.sum(stress * strain), rtol=2e-6, atol=2e-7
        )
        assert result.energy_transfer >= -2e-7


def test_coordinate_permutation_preserves_scalar_results_and_permutes_stress():
    gradient = jnp.asarray(((-0.7, 0.4, 0.1), (-0.2, -0.3, 0.5), (0.2, -0.1, -0.6)))
    widths = jnp.asarray((0.12, 0.25, 0.4))
    permutation = np.asarray((2, 0, 1))
    permuted_gradient = gradient[permutation][:, permutation]
    permuted_widths = widths[permutation]

    for model in (
        phx.equations.SmagorinskyLESPlan(0.16),
        phx.equations.WALELESPlan(0.32),
        phx.equations.VremanLESPlan(0.07),
        phx.equations.AMDLESPlan(0.3),
    ):
        original = model.evaluate(_inputs(gradient, widths))
        permuted = model.evaluate(_inputs(permuted_gradient, permuted_widths))
        np.testing.assert_allclose(
            permuted.kinematic_viscosity,
            original.kinematic_viscosity,
            rtol=4e-6,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            permuted.energy_transfer,
            original.energy_transfer,
            rtol=4e-6,
            atol=2e-8,
        )
        expected_stress = original.specific_deviatoric_stress[permutation][:, permutation]
        np.testing.assert_allclose(
            permuted.specific_deviatoric_stress,
            expected_stress,
            rtol=4e-6,
            atol=2e-8,
        )


def test_coefficient_and_width_scaling_match_each_formula():
    gradient = jnp.asarray(((-0.6, 0.3, 0.2), (0.1, -0.4, 0.5), (-0.2, 0.1, 0.3)))
    widths = jnp.asarray((0.13, 0.21, 0.34))
    coefficient = 0.17
    factor = 1.7
    cases = (
        (phx.equations.SmagorinskyLESPlan, 2),
        (phx.equations.WALELESPlan, 2),
        (phx.equations.VremanLESPlan, 1),
        (phx.equations.AMDLESPlan, 1),
    )
    for model_type, coefficient_power in cases:
        baseline = (
            model_type(coefficient)
            .evaluate(_inputs(gradient, widths))
            .kinematic_viscosity
        )
        coefficient_scaled = (
            model_type(coefficient * factor)
            .evaluate(_inputs(gradient, widths))
            .kinematic_viscosity
        )
        width_scaled = (
            model_type(coefficient)
            .evaluate(_inputs(gradient, widths * factor))
            .kinematic_viscosity
        )
        np.testing.assert_allclose(
            coefficient_scaled, baseline * factor**coefficient_power, rtol=5e-6, atol=2e-8
        )
        np.testing.assert_allclose(
            width_scaled, baseline * factor**2, rtol=5e-6, atol=2e-8
        )


def test_jit_and_jvp_cover_dynamic_coefficients_and_widths():
    gradient = jnp.asarray(((-0.6, 0.3, 0.2), (0.1, -0.4, 0.5), (-0.2, 0.1, 0.3)))
    scale = phx.equations.LESFilterScale(jnp.asarray((0.13, 0.21, 0.34)))
    prepared = phx.equations.VremanLESPlan(0.07).prepare(_provenance())

    compiled = jax.jit(
        lambda value: (
            prepared.evaluate(
                phx.equations.AlgebraicLESInputs(value, scale)
            ).kinematic_viscosity
        )
    )
    np.testing.assert_allclose(
        compiled(gradient),
        prepared.evaluate(
            phx.equations.AlgebraicLESInputs(gradient, scale)
        ).kinematic_viscosity,
        rtol=2e-6,
    )

    inputs = phx.equations.AlgebraicLESInputs(gradient, scale)

    def coefficient_response(coefficient):
        return (
            phx.equations.SmagorinskyLESPlan(coefficient)
            .evaluate(inputs)
            .kinematic_viscosity
        )

    primal, tangent = jax.jvp(
        coefficient_response, (jnp.asarray(0.2),), (jnp.asarray(1.0),)
    )
    np.testing.assert_allclose(tangent, 2.0 * primal / 0.2, rtol=3e-6)

    def width_response(widths):
        local = phx.equations.AlgebraicLESInputs(
            gradient, phx.equations.LESFilterScale(widths)
        )
        return phx.equations.VremanLESPlan(0.07).evaluate(local).kinematic_viscosity

    width_primal, width_tangent = jax.jvp(
        width_response,
        (jnp.asarray((0.13, 0.21, 0.34)),),
        (jnp.asarray((0.01, -0.02, 0.03)),),
    )
    assert jnp.isfinite(width_primal)
    assert jnp.isfinite(width_tangent)

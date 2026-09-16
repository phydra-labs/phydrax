#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.geometry.complex import (
    assess_complex_structure_family,
    ComplexStructureFamilyPlan,
    ProjectiveLineSamples,
    TrainableHomogeneousHypersurface,
)
from phydrax.integration import (
    CalabiYauModuliObservablePlan,
    evaluate_calabi_yau_moduli_observables,
    PreparedCalabiYauModuliSamples,
    ProjectiveMeasureTarget,
)
from phydrax.metrix import (
    chern_character_form,
    integrate_top_characteristic_form,
)


def test_complex_structure_family_preserves_pivot_and_evaluates_deformations():
    base = TrainableHomogeneousHypersurface(
        ((2, 0), (1, 1), (0, 2)),
        jnp.asarray((1.0 + 0.0j, 0.0j, 1.0 + 0.0j)),
        pivot=0,
        family_id="quadratic-cp1-family",
    )
    family = ComplexStructureFamilyPlan(
        base,
        ("cross-term",),
        ((0.0, 1.0, 0.0),),
    )
    evidence = assess_complex_structure_family(family)
    assert bool(evidence.accepted)
    values = family.evaluate_deformations(jnp.asarray(((1.0, 2.0),)))
    np.testing.assert_allclose(values, ((2.0,),))
    deformed = family.deformed(jnp.asarray((0.25 + 0.0j,)))
    np.testing.assert_allclose(deformed.coefficients, (1.0, 0.25, 1.0))


def test_chern_character_form_and_sampled_number_use_explicit_convention():
    curvature = jnp.asarray(
        [
            [[[-2.0j * np.pi]]],
            [[[-4.0j * np.pi]]],
        ]
    )
    form = chern_character_form(
        curvature,
        1,
        2,
        source_id="u1-curvature-control",
    )
    np.testing.assert_allclose(form.coefficients[..., 0], (1.0, 2.0), atol=1e-12)
    evidence = integrate_top_characteristic_form(
        form,
        jnp.asarray((0.25, 0.75)),
        2.0,
        measure_id="two-point-control",
    )
    np.testing.assert_allclose(evidence.normalized_value, 1.75)
    np.testing.assert_allclose(evidence.physical_value, 3.5)
    assert bool(evidence.accepted)
    assert "not-an-exact" in evidence.claim

    curvature_four = jnp.zeros((1, 6, 1, 1), dtype=jnp.complex128)
    curvature_four = curvature_four.at[0, 0, 0, 0].set(-2.0j * np.pi)
    curvature_four = curvature_four.at[0, 5, 0, 0].set(-2.0j * np.pi)
    second_character = chern_character_form(
        curvature_four,
        2,
        4,
        source_id="u1-second-character-control",
    )
    np.testing.assert_allclose(second_character.coefficients[0, 0], 1.0, atol=1e-12)


def _measure():
    points = jnp.asarray(
        (
            (1.0 + 0.0j, 0.0j),
            (0.0j, 1.0 + 0.0j),
            (1.0 + 0.0j, 1.0 + 0.0j),
            (1.0 + 0.0j, -1.0 + 0.0j),
        )
    )
    samples = ProjectiveLineSamples(
        homogeneous_points=points,
        chart_indices=(0, 1, 0, 0),
        pivot_indices=(1, 0, 1, 1),
        polynomial_residuals=jnp.zeros((4,)),
        smoothness_margins=jnp.ones((4,)),
        valid=jnp.ones((4,), dtype=bool),
        line_ids=(0, 1, 2, 3),
        root_ids=(0, 0, 0, 0),
    )
    return ProjectiveMeasureTarget(
        samples,
        jnp.zeros((4,)),
        measure_kind="canonical",
    )


def test_sampled_harmonic_moduli_observables_report_wp_yukawa_and_batch_errors():
    measure = _measure()
    representatives = jnp.asarray(
        (
            ((1.0 + 0.0j,), (0.0j,)),
            ((0.0j,), (1.0 + 0.0j,)),
            ((1.0 + 0.0j,), (0.0j,)),
            ((0.0j,), (1.0 + 0.0j,)),
        )
    )
    yukawa = jnp.zeros((4, 2, 2, 2), dtype=jnp.complex128)
    yukawa = yukawa.at[:, 0, 0, 0].set(1.0)
    yukawa = yukawa.at[:, 1, 1, 1].set(2.0)
    plan = CalabiYauModuliObservablePlan(
        ("u", "v"),
        representative_kind="harmonic",
        representative_source_id="analytic-orthogonal-control",
        batch_count=2,
    )
    result = evaluate_calabi_yau_moduli_observables(
        PreparedCalabiYauModuliSamples(plan, measure, representatives, yukawa)
    )
    np.testing.assert_allclose(result.weil_petersson_metric, 0.5 * jnp.eye(2))
    np.testing.assert_allclose(result.yukawa_couplings[0, 0, 0], 1.0)
    np.testing.assert_allclose(result.yukawa_couplings[1, 1, 1], 2.0)
    np.testing.assert_allclose(result.hermiticity_residual, 0.0)
    np.testing.assert_allclose(result.yukawa_symmetry_residual, 0.0)
    assert bool(result.accepted)
    assert bool(result.authoritative)


def test_algebraic_representatives_never_promote_to_harmonic_authority():
    measure = _measure()
    representatives = jnp.ones((4, 1, 1), dtype=jnp.complex128)
    yukawa = jnp.ones((4, 1, 1, 1), dtype=jnp.complex128)
    plan = CalabiYauModuliObservablePlan(
        ("u",),
        representative_kind="algebraic",
        representative_source_id="normal-deformation-proxy",
        batch_count=2,
    )
    result = evaluate_calabi_yau_moduli_observables(
        PreparedCalabiYauModuliSamples(plan, measure, representatives, yukawa)
    )
    assert bool(result.accepted)
    assert not bool(result.authoritative)
    assert "not-harmonic" in result.claim

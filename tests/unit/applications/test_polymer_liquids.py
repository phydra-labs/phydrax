import jax.numpy as jnp
import numpy as np

from phydrax.applications import polymer_liquids as pl


def _transform(count=32):
    return pl.IsotropicRadialTransformPlan(count, 8.0).prepare()


def test_isotropic_radial_transform_has_exact_declared_discrete_inverse():
    transform = _transform()
    values = jnp.exp(-(transform.radii**2))
    evidence = pl.radial_transform_evidence(transform, values, tolerance=1.0e-11)
    assert evidence.successful
    np.testing.assert_allclose(
        transform.inverse(transform.forward(values)), values, rtol=1.0e-11, atol=1.0e-14
    )
    assert float(transform.radii[0]) > 0.0
    assert float(transform.wave_numbers[0]) > 0.0


def test_one_site_ideal_hnc_prism_is_exact_fixed_point():
    transform = _transform(16)
    mixture = pl.SiteMixturePlan(("A",), [0.2])
    form_factor = pl.SequenceFormFactorPlan("gaussian-chain", [0], 1.0, site_count=1)
    potential = pl.SitePairPotentialPlan(
        transform.radii,
        jnp.zeros((1, 1, transform.plan.count)),
        source_id="ideal-control",
    )
    prepared = pl.PRISMPlan(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.HNC), maximum_iterations=20
    ).prepare(transform, mixture, form_factor, potential)

    result = pl.solve_prism(prepared)

    assert result.successful
    np.testing.assert_allclose(result.evaluation.gamma, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        result.evaluation.closure.radial_distribution, 1.0, atol=1.0e-13
    )
    np.testing.assert_allclose(result.evaluation.oz.structure_factor, 1.0, atol=1.0e-13)
    assert np.all(np.asarray(result.evaluation.oz.linear_status) == 0)


def test_multisite_sequence_form_factors_retain_symmetry_and_psd():
    wave = _transform(12).wave_numbers
    for kind in ("gaussian-chain", "freely-jointed-chain", "gaussian-ring"):
        omega = pl.SequenceFormFactorPlan(kind, [0, 1, 0, 1], 0.8, site_count=2).evaluate(
            wave
        )
        np.testing.assert_allclose(omega, np.swapaxes(omega, -1, -2))
        assert np.min(np.linalg.eigvalsh(np.asarray(omega))) >= -1.0e-12


def test_prism_closures_enforce_domains_without_clipping():
    radii = jnp.asarray([0.25, 0.75, 1.25])
    gamma = jnp.full((1, 1, 3), 0.2)
    beta_potential = jnp.full((1, 1, 3), 0.4)

    hnc = pl.evaluate_prism_closure(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.HNC),
        radii,
        gamma,
        beta_potential,
    )
    py = pl.evaluate_prism_closure(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.PERCUS_YEVICK),
        radii,
        gamma,
        beta_potential,
    )
    msa = pl.evaluate_prism_closure(
        pl.PRISMClosurePlan(
            pl.PRISMClosureKind.MEAN_SPHERICAL,
            hard_core_diameters=[[1.0]],
        ),
        radii,
        gamma,
        beta_potential,
    )
    ms = pl.evaluate_prism_closure(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.MARTYNOV_SARKISOV),
        radii,
        gamma,
        beta_potential,
    )
    assert hnc.successful & py.successful & msa.successful & ms.successful
    np.testing.assert_allclose(msa.radial_distribution[..., :2], 0.0)

    invalid = pl.evaluate_prism_closure(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.MARTYNOV_SARKISOV),
        radii,
        jnp.full((1, 1, 3), -0.6),
        beta_potential,
    )
    assert not invalid.successful
    assert float(invalid.domain_margin) < 0.0


def test_prism_implicit_root_and_density_continuation_retain_certification():
    transform = _transform(8)
    mixture = pl.SiteMixturePlan(("A",), [0.1])
    form_factor = pl.SequenceFormFactorPlan("gaussian-chain", [0], 1.0, site_count=1)
    potential = pl.SitePairPotentialPlan(
        transform.radii,
        jnp.zeros((1, 1, transform.plan.count)),
        source_id="ideal-continuation-control",
    )
    prepared = pl.PRISMPlan(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.HNC),
        maximum_iterations=20,
    ).prepare(transform, mixture, form_factor, potential)
    initial = pl.solve_prism(prepared)
    implicit = pl.solve_prism_implicit(prepared, jnp.zeros((1, 1, transform.plan.count)))
    assert initial.successful & implicit.successful

    continuation = pl.continue_prism_density(
        prepared,
        initial,
        1.0,
        1.1,
        pl.PRISMDensityContinuationPlan(
            minimum_density_scale=0.5,
            maximum_density_scale=1.5,
            initial_step=0.05,
            maximum_step=0.1,
            maximum_steps=8,
        ),
    )
    assert continuation.successful
    np.testing.assert_allclose(continuation.continuation.points[-1].coordinate, 1.1)

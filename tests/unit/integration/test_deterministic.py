import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _interval():
    return phx.domain.ScalarInterval(0.0, 1.0, label="x")


def test_reference_cell_rules_integrate_constant_to_cell_measure():
    rule = phx.integration.GaussLegendreRule(7)
    reference_rules = (
        (phx.integration.ReferenceIntervalRule(rule), 1.0),
        (phx.integration.ReferenceTriangleRule(rule), 0.5),
        (phx.integration.ReferenceQuadrilateralRule(rule), 1.0),
        (phx.integration.ReferenceTetrahedronRule(rule), 1.0 / 6.0),
        (phx.integration.ReferenceHexahedronRule(rule), 1.0),
    )

    for reference_rule, expected_measure in reference_rules:
        data = reference_rule.materialize()
        assert jnp.allclose(jnp.sum(data.weights), expected_measure, atol=1e-12)


def test_fixed_integral_mean_and_density_have_distinct_measure_semantics():
    domain = _interval()
    component = domain.component()
    x_squared = domain.Function("x")(lambda x: x**2)
    log_density = domain.Function("x")(lambda x: jnp.log1p(x))
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(12))

    integral = phx.integration.integrate(x_squared, phx.integration.over(component), plan)
    mean = phx.integration.integrate(
        x_squared, phx.integration.mean_over(component), plan
    )
    unnormalized = phx.integration.integrate(
        x_squared,
        phx.integration.density(phx.integration.over(component), log_density),
        plan,
    )
    normalized = phx.integration.integrate(
        x_squared,
        phx.integration.normalized_density(phx.integration.over(component), log_density),
        plan,
    )

    assert jnp.allclose(integral.value.data, 1.0 / 3.0, atol=1e-12)
    assert jnp.allclose(mean.value.data, 1.0 / 3.0, atol=1e-12)
    assert jnp.allclose(unnormalized.value.data, 7.0 / 12.0, atol=1e-12)
    assert jnp.allclose(normalized.value.data, 7.0 / 18.0, atol=1e-12)
    assert integral.error_estimate is None


def test_materialize_reduce_reuses_exactly_the_same_realization():
    domain = _interval()
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(10))
    realization = phx.integration.materialize(target, plan)
    linear = domain.Function("x")(lambda x: x)
    quadratic = domain.Function("x")(lambda x: x**2)

    first = phx.integration.reduce(linear, realization)
    second = phx.integration.reduce(quadratic, realization)

    assert first.num_evaluations == second.num_evaluations == 10
    assert jnp.allclose(first.value.data, 0.5, atol=1e-12)
    assert jnp.allclose(second.value.data, 1.0 / 3.0, atol=1e-12)


def test_from_samples_attaches_target_measure_without_resampling():
    domain = _interval()
    component = domain.component()
    points = component.sample(
        4096,
        structure=phx.domain.ProductStructure((("x",),)),
        sampler="uniform",
        key=jr.key(4),
    )
    realization = phx.integration.from_samples(
        phx.integration.over(component), points, key=jr.key(5)
    )
    function = domain.Function("x")(lambda x: x**2)

    first = phx.integration.reduce(function, realization)
    second = phx.integration.reduce(function, realization)

    assert first.value.data == second.value.data
    assert jnp.allclose(first.value.data, 1.0 / 3.0, atol=2e-2)


def test_scalar_boundary_and_interior_factors_form_one_product_rule():
    space = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    time = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = phx.domain.ProductDomain(space, time)
    component = domain.component({"x": phx.domain.Boundary(), "t": phx.domain.Interior()})
    function = domain.Function("x", "t")(lambda x, t: x + t)

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )

    assert jnp.allclose(estimate.value.data, 6.0, atol=1e-12)
    assert estimate.num_evaluations == 16


def test_scalar_boundary_product_preserves_probability_measure():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Uniform(0.0, 2.0),
        label="z",
    )
    interval = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    domain = phx.domain.ProductDomain(probability, interval)
    component = domain.component({"z": phx.domain.Interior(), "x": phx.domain.Boundary()})

    estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(5)),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 2.0, atol=1e-12)
    assert estimate.num_evaluations == 10


def test_sparse_grid_reports_level_difference_not_statistical_error():
    domain = _interval()
    function = domain.Function("x")(lambda x: x**4)

    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        phx.integration.SparseGridPlan(1, 4),
    )
    estimate = phx.integration.reduce(function, realization)
    current_count = realization.batch.batch.weights.data.size
    assert realization.batch.previous is not None
    previous_count = realization.batch.previous.weights.data.size

    assert jnp.allclose(estimate.value.data, 0.2, atol=1e-12)
    assert estimate.error_kind == "sparse-grid-level-difference"
    assert estimate.diagnostics.level_difference is not None
    assert estimate.num_evaluations == current_count + previous_count == 14
    assert estimate.diagnostics.num_evaluations == 14
    assert estimate.diagnostics.num_unique_nodes == current_count == 9


def test_mapped_triangle_preserves_output_field_semantics():
    rule = phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(6))
    target = phx.integration.mapped(
        rule,
        lambda reference: reference,
        lambda reference: jnp.ones((reference.shape[0],)),
    )

    estimate = phx.integration.integrate(
        lambda point: point[:, 0] + point[:, 1],
        target,
        phx.integration.CellQuadraturePlan(rule),
    )

    assert estimate.value.dims == ()
    assert jnp.allclose(estimate.value.data, 1.0 / 3.0, atol=1e-12)


def test_output_pytrees_reduce_leafwise_with_structure_and_dtype_preserved():
    domain = _interval()
    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(10)),
    )
    integrands = {
        "moments": (
            domain.Function("x")(lambda x: x),
            domain.Function("x")(lambda x: x**2),
        ),
        "complex": domain.Function("x")(lambda x: (1.0 + 2.0j) * x),
    }

    estimate = phx.integration.reduce(integrands, realization)

    assert set(estimate.value) == {"complex", "moments"}
    assert jnp.allclose(estimate.value["moments"][0].data, 0.5, atol=1e-12)
    assert jnp.allclose(estimate.value["moments"][1].data, 1.0 / 3.0, atol=1e-12)
    assert jnp.allclose(estimate.value["complex"].data, 0.5 + 1.0j, atol=1e-12)
    assert jnp.issubdtype(estimate.value["complex"].data.dtype, jnp.complexfloating)
    assert estimate.error_estimate is None


def test_reusable_realization_supports_jit_vmap_jvp_grad_and_complex_values():
    domain = _interval()
    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(12)),
    )

    def objective(scale):
        function = domain.Function("x")(lambda x: scale * x**2 + 1j * scale * x)
        return phx.integration.reduce(function, realization).value.data

    value = jax.jit(objective)(3.0)
    values = jax.vmap(objective)(jnp.asarray([1.0, 2.0, 3.0]))
    tangent = jax.jvp(lambda scale: objective(scale).real, (3.0,), (1.0,))[1]
    gradient = jax.grad(lambda scale: objective(scale).real)(3.0)

    assert jnp.allclose(value, 1.0 + 1.5j, atol=1e-12)
    assert jnp.allclose(
        values,
        jnp.asarray([1.0 / 3.0 + 0.5j, 2.0 / 3.0 + 1.0j, 1.0 + 1.5j]),
        atol=1e-12,
    )
    assert jnp.allclose(tangent, 1.0 / 3.0, atol=1e-12)
    assert jnp.allclose(gradient, 1.0 / 3.0, atol=1e-12)


def test_deterministic_and_randomized_key_contracts_are_explicit():
    domain = _interval()
    target = phx.integration.over(domain.component())

    with pytest.raises(ValueError, match="does not consume key"):
        phx.integration.materialize(
            target, phx.integration.FixedQuadraturePlan(), key=jr.key(0)
        )
    with pytest.raises(ValueError, match="requires key"):
        phx.integration.materialize(target, phx.integration.MonteCarloPlan(16))


def test_zero_density_reports_invalid_normalization_mass():
    domain = _interval()
    target = phx.integration.normalized_density(
        phx.integration.over(domain.component()),
        domain.Function()(lambda: -jnp.inf),
    )

    estimate = phx.integration.integrate(
        1.0,
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(6)),
    )

    assert estimate.status == int(
        phx.integration.IntegrationStatus.INVALID_NORMALIZATION_MASS
    )


@pytest.mark.parametrize("use_union", [False, True])
def test_fixed_density_rejects_zero_normalized_component_mass(use_union):
    domain = _interval()
    left_empty = domain.component(where_all=lambda x: x < 0.0)
    if use_union:
        right_empty = domain.component(where_all=lambda x: x > 1.0)
        component = phx.domain.DomainComponentUnion((left_empty, right_empty))
    else:
        component = left_empty
    target = phx.integration.density(
        phx.integration.mean_over(component),
        domain.Function()(lambda: 0.0),
    )

    estimate = phx.integration.integrate(
        1.0,
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(6)),
    )

    assert estimate.status == int(
        phx.integration.IntegrationStatus.INVALID_NORMALIZATION_MASS
    )
    assert not estimate.successful
    assert not jnp.all(jnp.isfinite(estimate.value.data))


def test_anisotropic_sparse_grid_preserves_constant_measure():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    domain = phx.domain.ProductDomain(x, y)

    estimate = phx.integration.integrate(
        1.0,
        phx.integration.over(domain.component()),
        phx.integration.SparseGridPlan(2, 2, anisotropy=(2, 1)),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 1.0, atol=1e-12)


def test_mapped_density_preserves_density_and_normalization_semantics():
    rule = phx.integration.ReferenceIntervalRule(phx.integration.GaussLegendreRule(8))
    base = phx.integration.mapped(
        rule,
        lambda reference: reference[:, 0],
        lambda reference: jnp.ones(reference.shape[0]),
    )
    target = phx.integration.normalized_density(base, lambda x: jnp.log1p(x))

    estimate = phx.integration.integrate(
        lambda x: x,
        target,
        phx.integration.CellQuadraturePlan(rule),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 5.0 / 9.0, atol=1e-12)


def test_mapped_target_mass_rescales_the_physical_measure():
    rule = phx.integration.ReferenceIntervalRule(phx.integration.GaussLegendreRule(8))
    target = phx.integration.mapped(
        rule,
        lambda reference: reference[:, 0],
        lambda reference: jnp.ones(reference.shape[0]),
        target_mass=jnp.asarray(2.0),
    )

    estimate = phx.integration.integrate(
        1.0,
        target,
        phx.integration.CellQuadraturePlan(rule),
    )

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, 2.0, atol=1e-12)
    assert jnp.allclose(estimate.diagnostics.target_mass, 2.0, atol=1e-12)


def test_mapped_finite_operand_product_overflow_is_nonfinite_integrand():
    rule = phx.integration.ReferenceIntervalRule(phx.integration.GaussLegendreRule(2))
    target = phx.integration.mapped(
        rule,
        lambda reference: reference[:, 0],
        lambda reference: jnp.ones(reference.shape[0]),
        target_mass=jnp.asarray(4.0),
    )
    maximum = jnp.finfo(jnp.asarray(1.0).dtype).max

    estimate = phx.integration.integrate(
        maximum,
        target,
        phx.integration.CellQuadraturePlan(rule),
    )

    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)
    assert not estimate.successful
    assert not jnp.all(jnp.isfinite(estimate.value.data))


def test_mapped_finite_normalized_quotient_overflow_is_nonfinite_integrand():
    rule = phx.integration.ReferenceIntervalRule(phx.integration.GaussLegendreRule(2))
    base = phx.integration.mapped(
        rule,
        lambda reference: reference[:, 0],
        lambda reference: jnp.ones(reference.shape[0]),
    )
    target = phx.integration.normalized_density(base, lambda x: 0.0)
    dtype = jnp.asarray(1.0).dtype
    epsilon = jnp.finfo(dtype).eps
    scale = jnp.finfo(dtype).max * epsilon
    batch = phx.integration.MappedIntegrationBatch(
        jnp.zeros((2, 1), dtype=dtype),
        jnp.arange(2, dtype=dtype),
        jnp.asarray([1.0, -(1.0 - epsilon)], dtype=dtype),
    )
    realization = phx.integration.IntegrationRealization(
        target,
        phx.integration.CellQuadraturePlan(rule),
        batch,
        None,
    )

    estimate = phx.integration.reduce(
        jnp.asarray([scale, -scale], dtype=dtype),
        realization,
    )

    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)
    assert not estimate.successful
    assert not jnp.all(jnp.isfinite(estimate.value.data))


def test_fixed_probability_rejects_nonfinite_integrands():
    probability = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")

    estimate = phx.integration.integrate(
        jnp.asarray(jnp.nan),
        phx.integration.expectation(probability),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )

    assert estimate.status == int(phx.integration.IntegrationStatus.NONFINITE_INTEGRAND)
    assert not estimate.successful


def test_density_preserves_a_normalized_component_base_across_plans():
    domain = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    target = phx.integration.density(
        phx.integration.mean_over(domain.component()),
        domain.Function()(lambda: 0.0),
    )
    estimates = (
        phx.integration.integrate(
            1.0,
            target,
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
        ),
        phx.integration.integrate(
            1.0,
            target,
            phx.integration.AdaptiveQuadraturePlan(
                absolute_tolerance=1e-10,
                relative_tolerance=1e-10,
            ),
        ),
        phx.integration.integrate(
            1.0,
            target,
            phx.integration.MonteCarloPlan(64),
            key=jr.key(13),
        ),
    )

    assert all(estimate.successful for estimate in estimates)
    assert all(
        jnp.allclose(estimate.value.data, 1.0, atol=1e-12) for estimate in estimates
    )


def test_probability_component_uses_probability_measure():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Uniform(0.0, 2.0),
        label="z",
    )
    function = probability.Function("z")(lambda z: z)
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8))

    mass = phx.integration.integrate(
        1.0,
        phx.integration.over(probability.component()),
        plan,
    )
    mean = phx.integration.integrate(
        function,
        phx.integration.over(probability.component()),
        plan,
    )

    assert mass.successful
    assert mean.successful
    assert jnp.allclose(mass.value.data, 1.0, atol=1e-12)
    assert jnp.allclose(mean.value.data, 1.0, atol=1e-12)


def test_partial_axis_integration_does_not_apply_unreduced_geometry_weights():
    geometry = phx.domain.geometry2d.Circle(center=(0.0, 0.0), radius=1.0)
    time = phx.domain.TimeInterval(0.0, 3.0)
    domain = geometry @ time
    component = domain.component()
    axis = phx.domain.LegendreAxisSpec(5)
    grid = phx.domain.GridSpec((axis, axis), cut_cell_order=2)
    points = component.sample_coord_separable(
        {"x": grid},
        num_points=5,
        dense_structure=phx.domain.ProductStructure((("t",),)),
        key=jr.key(14),
    )
    realization = phx.integration.from_samples(
        phx.integration.over(component, axes="t"),
        points,
    )

    estimate = phx.integration.reduce(1.0, realization)
    mask = points.coord_mask_by_label["x"].data

    assert estimate.successful
    assert jnp.allclose(estimate.value.data, jnp.where(mask, 3.0, 0.0), atol=1e-12)


def test_sparse_grid_rejects_boundary_component_selectors():
    domain = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    target = phx.integration.over(domain.component({"x": phx.domain.Boundary()}))

    with pytest.raises(TypeError, match=r"Interior\(\)"):
        phx.integration.materialize(
            target,
            phx.integration.SparseGridPlan(1, 2),
        )


def test_sparse_grid_requires_complete_coupled_axes_with_fixed_labels():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 1.0, label="y")
    t = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = phx.domain.ProductDomain(x, y, t)
    component = domain.component({"t": phx.domain.FixedStart()})
    plan = phx.integration.SparseGridPlan(2, 2)

    with pytest.raises(ValueError, match="one coupled axis"):
        phx.integration.materialize(
            phx.integration.over(component, axes=("x",)),
            plan,
        )

    realization = phx.integration.materialize(
        phx.integration.over(component, axes=("x", "y")),
        plan,
    )

    assert realization.batch.batch.points.points["t"].dims == ()
    assert realization.batch.batch.points.points["t"].data == 0.0

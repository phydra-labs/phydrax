#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic source identity, rejection rollback, and local operator compatibility."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.domain import DomainFunction
from phydrax.enforcement import (
    ConditionEvaluationContext,
    EnforcementOptions,
    EnforcementSpec,
    PeriodicResourcePolicy,
    prepare_periodic_projection,
    PreparedPeriodicProjection,
    RealizationLifecycleState,
    RealizationStatus,
)


def _projection() -> tuple[DomainFunction, PreparedPeriodicProjection]:
    space = phx.domain.Interval1d(0.0, 1.0)
    field = space.Function("x")(lambda x: x[0] ** 2)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    prepared = prepare_periodic_projection(
        {"u": field}, (phx.conditions.Periodic("u", seam),), route="analytic"
    )
    return field, prepared


def test_rejected_periodic_attempt_preserves_committed_coordinates() -> None:
    field, prepared = _projection()
    accepted = prepared.realize(
        {"u": field},
        context=ConditionEvaluationContext(
            prepared.condition, accepted_step=2, parameter_revision=3
        ),
    )
    other = phx.conditions.Periodic(
        "u", prepared.declarations[0].pairing, target=1.0
    ).as_condition()
    rejected = prepared.realize(
        {"u": field},
        accepted.state,
        context=ConditionEvaluationContext(
            other, accepted_step=4, parameter_revision=5, attempt=1
        ),
    )

    assert accepted.successful
    assert rejected.status is RealizationStatus.INVALID_INPUT
    assert rejected.fields is None
    assert rejected.state.generation == accepted.state.generation
    assert rejected.state.accepted_step == 2
    assert rejected.state.parameter_revision == 3
    assert rejected.state.realization_stamp is accepted.state.realization_stamp
    assert rejected.state.last_failure is not None
    assert rejected.state.last_failure.accepted_step == 4
    assert rejected.state.last_failure.attempt == 1


def test_runtime_field_support_mismatch_is_a_failed_realization() -> None:
    _, prepared = _projection()
    other = phx.domain.Interval1d(0.0, 2.0)
    field = other.Function("x")(lambda x: x[0] ** 2)
    result = prepared.realize(
        {"u": field},
        RealizationLifecycleState.initial(),
        context=ConditionEvaluationContext(prepared.condition, accepted_step=3),
    )

    assert result.status is RealizationStatus.INVALID_INPUT
    assert result.fields is None
    assert result.stamp is None
    assert result.state.accepted_step == 0


def test_same_coordinate_cannot_evade_joint_rows_with_different_ids() -> None:
    field, prepared = _projection()
    second = phx.domain.PeriodicIdentification(
        field.domain, "x", identification_id="another-name"
    )
    declarations = (
        prepared.declarations[0],
        phx.conditions.Periodic("u", second.pairing(), target=1.0),
    )

    with pytest.raises(ValueError):
        prepare_periodic_projection({"u": field}, declarations, route="analytic")


def test_independent_field_remains_periodic_under_analytic_realization() -> None:
    field, prepared = _projection()
    constant = field.domain.Function()(3.0)
    independent = prepare_periodic_projection(
        {"u": constant}, prepared.declarations, route="analytic"
    )
    result = independent.realize(
        {"u": constant}, context=ConditionEvaluationContext(independent.condition)
    )
    assert result.successful
    assert result.fields is not None
    batch = field.domain.component().points(
        {"x": jnp.asarray([[0.0], [0.25], [1.0]], dtype=jnp.float64)}
    )
    np.testing.assert_allclose(result.fields["u"](batch).data, 3.0, rtol=0.0, atol=1e-14)


def test_affine_robin_compatibility_uses_the_boundary_operator() -> None:
    space = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.Interval1d(0.0, 1.0, label="y")
    field = space.Function("x", "y")(lambda x, y: 1.0 + x[0] + y[0])
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    prepared = prepare_periodic_projection(
        {"u": field},
        (phx.conditions.Periodic("u", seam, transport=0.5, target=1.0),),
        route="analytic",
    )
    wall = phx.conditions.Robin(
        "u",
        space.component({"y": phx.domain.Boundary()}),
        dirichlet_coefficient=2.0,
        neumann_coefficient=1.0,
        target=2.0,
        var="y",
    )

    # J(g)=1, but B(h)=2. Comparing J(g) with h would incorrectly admit this wall.
    with pytest.raises(ValueError):
        phx.enforcement.compile(
            {"u": field},
            (
                EnforcementSpec(wall),
                EnforcementSpec(prepared.condition, realization=prepared),
            ),
            options=EnforcementOptions(num_reference=128),
        )


def test_constant_affine_jump_preserves_homogeneous_neumann_walls() -> None:
    space = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.Interval1d(0.0, 1.0, label="y")
    field = space.Function("x", "y")(lambda x, y: 1.0 + x[0] + y[0])
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    declaration = phx.conditions.Periodic("u", seam, target=1.0)
    prepared = prepare_periodic_projection({"u": field}, (declaration,), route="analytic")
    wall = phx.conditions.Neumann(
        "u", space.component({"y": phx.domain.Boundary()}), target=0.0, var="y"
    )
    program = phx.enforcement.compile(
        {"u": field},
        (
            EnforcementSpec(wall),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=128),
    )
    fields = program.apply({"u": field})
    wall_points = wall.on.sample(phx.domain.PointSampling(8), key=jr.key(7))
    seam_points = seam.component.sample(phx.domain.PointSampling(8), key=jr.key(8))

    np.testing.assert_allclose(wall.residual(fields)(wall_points).data, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        declaration.residual(fields)(seam_points).data, 0.0, atol=1e-12
    )


def test_fixed_target_participates_in_periodic_identity() -> None:
    field, prepared = _projection()
    alternate = phx.conditions.Periodic("u", prepared.declarations[0].pairing, target=1.0)
    replacement = prepare_periodic_projection(
        {"u": field}, (alternate,), route="analytic"
    )
    assert replacement.condition.condition_id != prepared.condition.condition_id
    assert replacement.realization_id != prepared.realization_id
    points = field.domain.component().points(
        {"x": jnp.asarray([[0.0], [1.0]], dtype=jnp.float64)}
    )
    result = replacement.realize(
        {"u": field}, context=ConditionEvaluationContext(replacement.condition)
    )
    assert result.fields is not None
    values = result.fields["u"](points).data
    np.testing.assert_allclose(values[-1] - values[0], 1.0, atol=1e-12)


def test_endpoint_budget_counts_every_declared_jet() -> None:
    space = phx.domain.HyperRectangle(jnp.zeros(2), jnp.ones(2))
    field = space.Function("x")(lambda x: jnp.sin(x[0]) + x[0] * x[1])
    declarations = tuple(
        phx.conditions.Periodic(
            "u",
            phx.domain.PeriodicIdentification(space, "x", component=axis).pairing(),
            order=order,
        )
        for axis in (0, 1)
        for order in (0, 1)
    )
    # Two axes, each requesting both endpoint values and slopes: (1+4)^2 = 25.
    with pytest.raises(ValueError):
        prepare_periodic_projection(
            {"u": field},
            declarations,
            route="analytic",
            resources=PeriodicResourcePolicy(maximum_endpoint_evaluations=24),
        )
    prepared = prepare_periodic_projection(
        {"u": field},
        declarations,
        route="analytic",
        resources=PeriodicResourcePolicy(maximum_endpoint_evaluations=25),
    )
    assert prepared.evidence.endpoint_evaluations == 25
    result = prepared.realize(
        {"u": field}, context=ConditionEvaluationContext(prepared.condition)
    )
    assert result.fields is not None
    for declaration in declarations:
        points = declaration.on.sample(phx.domain.PointSampling(4), key=jr.key(2))
        np.testing.assert_allclose(
            declaration.residual(result.fields)(points).data, 0.0, atol=1e-12
        )


def test_value_only_resource_policy_allows_zero_maximum_order() -> None:
    field, prepared = _projection()
    restricted = prepare_periodic_projection(
        {"u": field},
        prepared.declarations,
        route="analytic",
        resources=PeriodicResourcePolicy(maximum_order=0),
    )
    result = restricted.realize(
        {"u": field}, context=ConditionEvaluationContext(restricted.condition)
    )
    assert result.fields is not None
    declaration = restricted.declarations[0]
    points = declaration.on.sample(phx.domain.PointSampling(1), key=jr.key(0))
    np.testing.assert_allclose(
        declaration.residual(result.fields)(points).data, 0.0, atol=1e-12
    )


def test_small_nonzero_transport_commutator_is_not_exact_evidence() -> None:
    space = phx.domain.HyperRectangle(jnp.zeros(2), jnp.ones(2))
    field = space.Function()(jnp.asarray([0.0, 1e13], dtype=jnp.float64))
    value = phx.conditions.ArrayCodomain.from_shape((2,))
    first = phx.conditions.EventLinearMap(jnp.diag(jnp.asarray([1.0, 2.0])))
    second = phx.conditions.EventLinearMap(
        jnp.asarray([[1.0, 5e-13], [0.0, 1.0]], dtype=jnp.float64)
    )
    declarations = (
        phx.conditions.Periodic(
            "u",
            phx.domain.PeriodicIdentification(space, "x", component=0).pairing(),
            transport=first,
            value=value,
        ),
        phx.conditions.Periodic(
            "u",
            phx.domain.PeriodicIdentification(space, "x", component=1).pairing(),
            transport=second,
            value=value,
        ),
    )
    with pytest.raises(ValueError):
        prepare_periodic_projection({"u": field}, declarations, route="analytic")


@pytest.mark.parametrize(
    "coefficients", [(1.0, 2.0), (2.0, 2.0)], ids=["noncommuting", "scalar-map"]
)
def test_eventwise_robin_wall_requires_transport_commutation(
    coefficients: tuple[float, float],
) -> None:
    space = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.Interval1d(0.0, 1.0, label="y")
    field = space.Function("x", "y")(
        lambda x, y: jnp.stack([1.0 + x[0] + y[0], 2.0 + 2.0 * x[0] + y[0]])
    )
    transport = phx.conditions.EventLinearMap(
        jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)
    )
    declaration = phx.conditions.Periodic(
        "u",
        phx.domain.PeriodicIdentification(space, "x").pairing(),
        transport=transport,
        value=phx.conditions.ArrayCodomain.from_shape((2,)),
    )
    prepared = prepare_periodic_projection({"u": field}, (declaration,), route="analytic")
    wall = phx.conditions.Robin(
        "u",
        space.component({"y": phx.domain.Boundary()}),
        dirichlet_coefficient=jnp.asarray(coefficients, dtype=jnp.float64),
        neumann_coefficient=1.0,
        target=0.0,
        var="y",
    )
    specs = (
        EnforcementSpec(wall),
        EnforcementSpec(prepared.condition, realization=prepared),
    )
    options = EnforcementOptions(num_reference=128)
    if coefficients[0] != coefficients[1]:
        with pytest.raises(ValueError):
            phx.enforcement.compile({"u": field}, specs, options=options)
        return
    program = phx.enforcement.compile({"u": field}, specs, options=options)
    result = program.apply({"u": field})
    points = wall.on.sample(phx.domain.PointSampling(8), key=jr.key(3))
    np.testing.assert_allclose(wall.residual(result)(points).data, 0.0, atol=1e-11)


@pytest.mark.parametrize("functional", [False, True], ids=["constant", "field"])
def test_real_event_refuses_complex_affine_target(functional: bool) -> None:
    field, prepared = _projection()
    target = field.domain.Function()(1.0j) if functional else 1.0j
    with pytest.raises(ValueError):
        phx.conditions.Periodic(
            "u",
            prepared.declarations[0].pairing,
            target=target,
            value=phx.conditions.ArrayCodomain(dtype=jnp.float64),
        )


def test_construction_checks_the_output_event_not_only_the_input_period() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    embedding = phx.nn.layers.ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=1,
        coordinate=0,
        period=1.0,
        modes=(1,),
    )
    field = space.Model("x")(embedding)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    with pytest.raises(ValueError):
        prepare_periodic_projection(
            {"u": field}, (phx.conditions.Periodic("u", seam),), route="construction"
        )


def test_construction_requires_exact_identification_period() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    embedding = phx.nn.layers.ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=1,
        coordinate=0,
        period=1.0 + 5e-13,
        modes=(10**12,),
    )
    field = space.Model("x")(embedding)
    declaration = phx.conditions.Periodic(
        "u",
        phx.domain.PeriodicIdentification(space, "x").pairing(),
        value=phx.conditions.ArrayCodomain.from_shape((2,)),
    )
    with pytest.raises(ValueError):
        prepare_periodic_projection({"u": field}, (declaration,), route="construction")


def test_compiled_construction_cannot_bypass_fundamental_support_validation() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    embedding = phx.nn.layers.ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=1, coordinate=0, period=1.0, modes=(1,)
    )
    field = space.Model("x")(embedding)
    declaration = phx.conditions.Periodic(
        "u",
        phx.domain.PeriodicIdentification(space, "x").pairing(),
        value=phx.conditions.ArrayCodomain.from_shape((2,)),
    )
    prepared = prepare_periodic_projection(
        {"u": field}, (declaration,), route="construction"
    )
    other = phx.domain.Interval1d(0.0, 2.0).Model("x")(embedding)

    def realize(
        projection: PreparedPeriodicProjection, value: DomainFunction
    ) -> phx.enforcement.FieldRealizationResult:
        return projection.realize(
            {"u": value}, context=ConditionEvaluationContext(projection.condition)
        )

    compiled = eqx.filter_jit(realize)
    valid = compiled(prepared, field)
    assert valid.status is RealizationStatus.UNCHANGED
    eager = realize(prepared, other)
    assert eager.status is RealizationStatus.INVALID_INPUT
    assert eager.fields is None
    with pytest.raises(eqx.EquinoxRuntimeError):
        compiled(prepared, other)


def test_periodic_admission_requires_a_typed_condition() -> None:
    _, prepared = _projection()
    with pytest.raises(TypeError):
        prepared.admission(None)  # ty: ignore[invalid-argument-type]


def test_realization_context_requires_a_typed_condition() -> None:
    with pytest.raises(TypeError):
        ConditionEvaluationContext(None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("state", "context", "argument"),
    [
        pytest.param(None, None, "context", id="missing-context"),
        pytest.param(object(), None, "state", id="invalid-lifecycle-state"),
    ],
)
def test_periodic_realization_refuses_invalid_boundary_kinds(
    state: object, context: object, argument: str
) -> None:
    field, prepared = _projection()
    if argument == "state":
        context = ConditionEvaluationContext(prepared.condition)
    with pytest.raises(TypeError):
        prepared.realize(
            {"u": field},
            state,  # ty: ignore[invalid-argument-type]
            context=context,  # ty: ignore[invalid-argument-type]
        )

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared periodic seam projections and their composition admission.

Independent oracles: the centered Bernoulli endpoint lift
``p_k(x) = L^k B_{k+1}((x-a)/L) / (k+1)!`` with textbook Bernoulli polynomials,
for which ``J_j p_k = delta_jk``; the elementary minimum-norm coefficient update
``c + M^+ (g - M c)``; and direct JAX differentiation of the unprojected base
field. Tolerances are float64 round-off scaled by the field magnitudes.
"""

import math
from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._model import PERIODIC_INPUT_CERTIFICATE_KEY
from phydrax.conditions import (
    Condition,
    FieldCodomain,
    FieldSpec,
    Periodic,
    ProductFieldSpec,
)
from phydrax.domain import (
    Boundary,
    Domain,
    DomainFunction,
    FixedStart,
    HyperRectangle,
    Interval1d,
    PeriodicIdentification,
    physical_boundary,
    PointSampling,
    PointwiseEvaluator,
    TimeInterval,
)
from phydrax.enforcement import (
    CallableCorrectionChart,
    ConditionEvaluationContext,
    EnforcementOptions,
    EnforcementSpec,
    finite_feature_linear_representation,
    InteriorAnchors,
    LinearAssemblyEvidence,
    LinearConditionAssembly,
    NonlinearFieldRetraction,
    PeriodicResourcePolicy,
    prepare_periodic_projection,
    PreparedPeriodicProjection,
    RealizationStatus,
)
from phydrax.linalg import ArraySpace, DenseLinearOperator
from phydrax.nn import PeriodicInputCertificate
from phydrax.nn.models import MLP


LENGTH = 2.0
SPACE = Interval1d(0.0, LENGTH)
IDENTIFICATION = PeriodicIdentification(SPACE, "x")
PAIRING = IDENTIFICATION.pairing()
QUERY = jnp.asarray([0.0, 0.37, 1.1, 1.73, LENGTH])


def _base(x: Array) -> Array:
    return jnp.sin(1.3 * x[0]) + 0.3 * x[0] ** 2 + 0.1 * x[0] ** 3


BASE = SPACE.Function("x")(_base)


def _bernoulli(order: int, s: Array) -> Array:
    match order:
        case 1:
            return s - 0.5
        case 2:
            return s**2 - s + 1.0 / 6.0
        case 3:
            return s**3 - 1.5 * s**2 + 0.5 * s
        case 4:
            return s**4 - 2.0 * s**3 + s**2 - 1.0 / 30.0
        case _:
            raise ValueError(f"No reference Bernoulli polynomial of order {order}.")


def _lift(order: int, s: Array, length: float) -> Array:
    return length**order * _bernoulli(order + 1, s) / math.factorial(order + 1)


def _derivative(
    function: Callable[[Array], Array], order: int, axis: int = 0
) -> Callable[[Array], Array]:
    for _ in range(order):
        function = (lambda f: lambda p: jax.jacfwd(f)(p)[axis])(function)
    return function


def _project_axis(
    function: Callable[[Array], Array],
    axis: int,
    lower: float,
    length: float,
    orders: tuple[int, ...],
) -> Callable[[Array], Array]:
    """Independent centered-Bernoulli projection along one Cartesian axis."""

    def projected(point: Array) -> Array:
        upper_point = point.at[axis].set(lower + length)
        lower_point = point.at[axis].set(lower)
        s = (point[axis] - lower) / length
        value = function(point)
        for order in orders:
            jet = _derivative(function, order, axis)
            value = value - _lift(order, s, length) * (
                jet(upper_point) - jet(lower_point)
            )
        return value

    return projected


def _realize(
    prepared: PreparedPeriodicProjection, fields: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = prepared.realize(
        fields, context=ConditionEvaluationContext(prepared.condition)
    )
    assert result.successful, result.message
    assert result.fields is not None
    return result.fields


def _values(field: DomainFunction, domain: Domain, points: Array) -> Array:
    batch = domain.component().points({"x": jnp.reshape(points, (points.shape[0], -1))})
    return jnp.asarray(field(batch).data).reshape((-1,))


def _pointwise(field: DomainFunction, domain: Domain) -> Callable[[Array], Array]:
    def value(point: Array) -> Array:
        return _values(field, domain, jnp.reshape(point, (1, -1)))[0]

    return value


def _jumps(function: Callable[[Array], Array], orders: range) -> np.ndarray:
    lower, upper = jnp.asarray([0.0]), jnp.asarray([LENGTH])
    return np.asarray(
        [
            _derivative(function, order)(upper) - _derivative(function, order)(lower)
            for order in orders
        ]
    )


def _seam(*orders: int) -> tuple[Periodic, ...]:
    return tuple(Periodic("u", PAIRING, order=order) for order in orders)


@pytest.mark.parametrize(
    "orders",
    [(0,), (0, 1), (0, 1, 2, 3), (2,)],
    ids=["value", "value-slope", "jets-0-to-3", "sparse-second-jet"],
)
def test_analytic_projection_matches_centered_bernoulli_oracle(
    orders: tuple[int, ...],
) -> None:
    prepared = prepare_periodic_projection({"u": BASE}, _seam(*orders), route="analytic")
    projected = _realize(prepared, {"u": BASE})["u"]

    oracle = _project_axis(_base, 0, 0.0, LENGTH, orders)
    expected = jnp.stack([oracle(jnp.reshape(x, (1,))) for x in QUERY])
    np.testing.assert_allclose(
        _values(projected, SPACE, QUERY), expected, rtol=0.0, atol=1e-13
    )
    assert prepared.evidence.scope == "continuum"
    assert prepared.evidence.required_regularity == (("u", "x", max(orders)),)


def test_value_projection_does_not_claim_undeclared_jets() -> None:
    prepared = prepare_periodic_projection({"u": BASE}, _seam(0), route="analytic")
    projected = _pointwise(_realize(prepared, {"u": BASE})["u"], SPACE)

    before = _jumps(_base, range(2))
    after = _jumps(projected, range(2))
    assert abs(after[0]) < 1e-14
    assert abs(before[1]) > 1e-2
    np.testing.assert_allclose(after[1], before[1], rtol=0.0, atol=1e-13)


def test_projection_is_idempotent_and_keeps_periodic_fields() -> None:
    declarations = _seam(0, 1)
    prepared = prepare_periodic_projection({"u": BASE}, declarations, route="analytic")
    once = _realize(prepared, {"u": BASE})
    twice = _realize(prepared, once)
    np.testing.assert_allclose(
        _values(twice["u"], SPACE, QUERY),
        _values(once["u"], SPACE, QUERY),
        rtol=0.0,
        atol=1e-14,
    )

    periodic = SPACE.Function("x")(
        lambda x: jnp.sin(2.0 * jnp.pi * x[0] / LENGTH) + jnp.cos(jnp.pi * x[0])
    )
    kept = _realize(prepared, {"u": periodic})["u"]
    np.testing.assert_allclose(
        _values(kept, SPACE, QUERY),
        _values(periodic, SPACE, QUERY),
        rtol=0.0,
        atol=1e-14,
    )


def test_projected_coordinate_derivative_matches_finite_differences() -> None:
    prepared = prepare_periodic_projection({"u": BASE}, _seam(0, 1), route="analytic")
    projected = _pointwise(_realize(prepared, {"u": BASE})["u"], SPACE)
    step = 1e-5
    for x in (0.37, 1.1, 1.73):
        point = jnp.asarray([x])
        derivative = jax.jvp(projected, (point,), (jnp.ones((1,)),))[1]
        central = (projected(point + step) - projected(point - step)) / (2.0 * step)
        np.testing.assert_allclose(derivative, central, rtol=0.0, atol=1e-8)


def test_parameter_gradient_keeps_endpoint_model_dependence() -> None:
    model = MLP(
        in_size=1,
        out_size="scalar",
        width_size=8,
        depth=2,
        activation=jnp.tanh,
        key=jr.key(0),
    )
    prepared = prepare_periodic_projection(
        {"u": SPACE.Model("x")(model)}, _seam(0, 1), route="analytic"
    )
    x0 = jnp.asarray([0.37])

    def projected(candidate: MLP) -> Array:
        field = _realize(prepared, {"u": SPACE.Model("x")(candidate)})["u"]
        return _pointwise(field, SPACE)(x0)

    def oracle(candidate: MLP) -> Array:
        def value(point: Array) -> Array:
            return jnp.reshape(candidate(point), ())

        return _project_axis(value, 0, 0.0, LENGTH, (0, 1))(x0)

    gradient = eqx.filter_grad(projected)(model)
    reference = eqx.filter_grad(oracle)(model)
    for leaf, expected in zip(
        jax.tree.leaves(eqx.filter(gradient, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(reference, eqx.is_inexact_array)),
        strict=True,
    ):
        np.testing.assert_allclose(leaf, expected, rtol=0.0, atol=1e-12)

    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    gradient_parameters = eqx.filter(gradient, eqx.is_inexact_array)
    keys = iter(jr.split(jr.key(1), len(jax.tree.leaves(parameters))))
    direction = jax.tree.map(lambda leaf: jr.normal(next(keys), leaf.shape), parameters)
    step = 1e-6

    def shifted(sign: float) -> Array:
        moved = jax.tree.map(lambda p, d: p + sign * step * d, parameters, direction)
        return projected(eqx.combine(moved, static))

    central = (shifted(1.0) - shifted(-1.0)) / (2.0 * step)
    directional = sum(
        jnp.sum(g * d)
        for g, d in zip(
            jax.tree.leaves(gradient_parameters), jax.tree.leaves(direction), strict=True
        )
    )
    np.testing.assert_allclose(directional, central, rtol=1e-6, atol=1e-8)


def test_two_axis_box_projection_matches_composed_oracle() -> None:
    box = HyperRectangle(jnp.zeros(2), jnp.asarray([1.0, 2.0]))
    seams = (
        PeriodicIdentification(box, "x", component=0),
        PeriodicIdentification(box, "x", component=1),
    )

    def base(point: Array) -> Array:
        return jnp.sin(1.1 * point[0] + 0.4 * point[1]) + point[0] * point[1] ** 2

    field = box.Function("x")(base)
    declarations = tuple(
        Periodic("v", seam.pairing(), order=order) for seam in seams for order in (0, 1)
    )
    prepared = prepare_periodic_projection({"v": field}, declarations, route="analytic")
    projected = _realize(prepared, {"v": field})["v"]

    oracle = _project_axis(_project_axis(base, 0, 0.0, 1.0, (0, 1)), 1, 0.0, 2.0, (0, 1))
    points = jnp.asarray(
        [[0.0, 0.0], [1.0, 2.0], [0.0, 2.0], [0.3, 0.0], [1.0, 0.7], [0.45, 1.3]]
    )
    np.testing.assert_allclose(
        _values(projected, box, points),
        jnp.stack([oracle(point) for point in points]),
        rtol=0.0,
        atol=1e-12,
    )
    assert tuple(axis.identification_id for axis in prepared.evidence.axes) == tuple(
        seam.identification_id for seam in seams
    )
    assert prepared.evidence.endpoint_evaluations == 25


_COMPLEX_BASE = SPACE.Function("x")(
    lambda x: jax.lax.complex(_base(x), jnp.cos(0.9 * x[0]) + 0.2 * x[0])
)


@pytest.mark.parametrize(
    ("field", "transport", "target"),
    [
        (BASE, -1.0, 0.0),
        (_COMPLEX_BASE, complex(np.exp(0.7j)), 0.0),
        (_COMPLEX_BASE, complex(np.exp(1e-6j)), 0.0),
        (BASE, 1.0, 0.5),
    ],
    ids=["antiperiodic", "bloch", "bloch-near-identity", "affine-jump"],
)
@pytest.mark.strict_jax
def test_transported_projection_satisfies_its_seam_relation(
    field: DomainFunction, transport: complex, target: float
) -> None:
    declarations = (
        Periodic("u", PAIRING, transport=transport, target=target),
        Periodic("u", PAIRING, order=1, transport=transport),
    )
    prepared = prepare_periodic_projection({"u": field}, declarations, route="analytic")
    projected = _pointwise(_realize(prepared, {"u": field})["u"], SPACE)
    lower, upper = jnp.asarray([0.0]), jnp.asarray([LENGTH])

    value = projected(upper) - transport * projected(lower)
    slope = jax.jacfwd(projected)(upper)[0] - transport * jax.jacfwd(projected)(lower)[0]
    assert projected(lower).dtype == _values(field, SPACE, lower).dtype
    assert abs(complex(value) - target) < 1e-13
    assert abs(complex(slope)) < 1e-13
    (axis,) = prepared.evidence.axes
    assert axis.rank == axis.rows == 2
    assert np.isfinite(axis.condition_number)
    assert axis.condition_number < 10.0


def test_linearly_dependent_declarations_are_refused() -> None:
    declarations = (
        Periodic("u", PAIRING, label="first"),
        Periodic("u", PAIRING, label="second"),
    )
    with pytest.raises(ValueError, match="linearly dependent"):
        prepare_periodic_projection({"u": BASE}, declarations, route="analytic")


_BOX = HyperRectangle(jnp.zeros(2), jnp.ones(2))
_BOX_X = PeriodicIdentification(_BOX, "x", component=0)
_BOX_Y = PeriodicIdentification(_BOX, "x", component=1)
_BOX_FIELD = _BOX.Function("x")(lambda x: jnp.sin(x[0]) + x[0] * x[1])


@pytest.mark.parametrize(
    ("fields", "declarations", "resources", "message"),
    [
        (
            {"u": BASE},
            (Periodic("u", PAIRING, order=3),),
            PeriodicResourcePolicy(maximum_order=2),
            "jet order 3 exceeds",
        ),
        (
            {"u": BASE},
            _seam(0, 1),
            PeriodicResourcePolicy(maximum_rows=1),
            "2 rows exceeds",
        ),
        (
            {"v": _BOX_FIELD},
            (Periodic("v", _BOX_X.pairing()), Periodic("v", _BOX_Y.pairing())),
            PeriodicResourcePolicy(maximum_axes=1),
            "2 identified coordinates",
        ),
        (
            {"v": _BOX_FIELD},
            (Periodic("v", _BOX_X.pairing()), Periodic("v", _BOX_Y.pairing())),
            PeriodicResourcePolicy(maximum_endpoint_evaluations=8),
            "9 endpoint evaluations",
        ),
    ],
    ids=["jet-order", "endpoint-rows", "identified-axes", "endpoint-evaluations"],
)
def test_resource_limits_are_refused(
    fields: Mapping[str, DomainFunction],
    declarations: tuple[Periodic, ...],
    resources: PeriodicResourcePolicy,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_periodic_projection(
            fields, declarations, route="analytic", resources=resources
        )


@pytest.mark.parametrize(
    ("declarations", "message"),
    [
        (
            (
                Periodic("v", _BOX_X.pairing(), transport=-1.0, target=1.0),
                Periodic("v", _BOX_Y.pairing(), target=1.0),
            ),
            "Incompatible periodic seam targets",
        ),
        (
            (
                Periodic(
                    "v",
                    _BOX_X.pairing(),
                    target=_BOX.Function("x")(lambda x: x[1]),
                ),
                Periodic("v", _BOX_Y.pairing()),
            ),
            "constant affine targets",
        ),
    ],
    ids=["conflicting-constant-jumps", "uncertifiable-varying-jump"],
)
def test_incompatible_corner_targets_are_refused(
    declarations: tuple[Periodic, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_periodic_projection({"v": _BOX_FIELD}, declarations, route="analytic")


def test_compatible_corner_affine_jumps_are_enforced() -> None:
    declarations = (
        Periodic("v", _BOX_X.pairing(), target=1.0),
        Periodic("v", _BOX_Y.pairing(), target=-0.5),
    )
    prepared = prepare_periodic_projection(
        {"v": _BOX_FIELD}, declarations, route="analytic"
    )
    projected = _realize(prepared, {"v": _BOX_FIELD})["v"]
    transverse = jnp.asarray([0.0, 0.3, 1.0])
    x_jump = _values(
        projected, _BOX, jnp.stack([jnp.ones(3), transverse], axis=-1)
    ) - _values(projected, _BOX, jnp.stack([jnp.zeros(3), transverse], axis=-1))
    y_jump = _values(
        projected, _BOX, jnp.stack([transverse, jnp.ones(3)], axis=-1)
    ) - _values(projected, _BOX, jnp.stack([transverse, jnp.zeros(3)], axis=-1))
    np.testing.assert_allclose(x_jump, 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(y_jump, -0.5, rtol=0.0, atol=1e-14)


def test_certified_trial_field_is_refused_by_analytic_route() -> None:
    box = HyperRectangle(-jnp.ones(2), jnp.ones(2))
    trial = box.Model("x")(
        phx.equations.LinearTrefftzField(
            phx.equations.HarmonicPolynomialBasis(2, 2),
            initial_scale=0.1,
            key=jr.key(4),
        )
    )
    seam = PeriodicIdentification(box, "x", component=0).pairing()
    with pytest.raises(ValueError, match="certified exact PDE trial field"):
        prepare_periodic_projection(
            {"u": trial}, (Periodic("u", seam),), route="analytic"
        )


_POWERS = jnp.arange(4)
_NODES = jnp.asarray([0.1, 0.6, 1.2, 1.9])
# Value and first-derivative endpoint jumps of the monomials x^k on [0, L].
_JUMP_ROWS = np.asarray(
    [
        [0.0, LENGTH, LENGTH**2, LENGTH**3],
        [0.0, 0.0, 2.0 * LENGTH, 3.0 * LENGTH**2],
    ]
)


def _cubic(coefficients: Array) -> DomainFunction:
    return SPACE.Function("x")(lambda x: jnp.sum(coefficients * x[0] ** _POWERS))


def _cubic_coefficients(field: DomainFunction) -> Array:
    vandermonde = _NODES[:, None] ** _POWERS[None, :]
    return jnp.linalg.solve(vandermonde, _values(field, SPACE, _NODES))


def _cubic_representation() -> Any:
    coefficients = ArraySpace((4,), dtype="float64")
    operator = DenseLinearOperator(
        jnp.asarray(_JUMP_ROWS), source=coefficients, operator_id="cubic-seam-jets"
    )
    seam = PAIRING.component.sample(PointSampling(1), key=jr.key(0))
    holder: dict[str, Any] = {}

    def assemble(bound: Any) -> LinearConditionAssembly:
        representation = holder["representation"]
        evidence = LinearAssemblyEvidence(
            bound_condition_id=bound.bound_id,
            operator_id=operator.operator_id,
            coefficient_space_id=coefficients.space_id,
            codomain_id="cubic-seam-jets",
            quantifier_id="pointwise",
            representation_id=representation.representation_id,
            prepared_id=representation.prepared_id,
            row_shape=(2,),
            row_dtype="float64",
            support_id=PAIRING.pairing_id,
            geometry_revision="interval-0-2",
            assembly_method="analytic-endpoint-rows",
            exactness="exact",
            numeric_fingerprint="cubic-seam-jets",
            tolerance=1e-12,
        )
        return LinearConditionAssembly(
            operator,
            evidence,
            codomain_coordinates=lambda targets: jnp.stack(
                [jnp.reshape(target(seam).data, ()) for target in targets]
            ),
        )

    representation = finite_feature_linear_representation(
        ProductFieldSpec((FieldSpec("u", FieldCodomain(SPACE.component())),)),
        coefficients,
        coefficients,
        lambda values: _cubic_coefficients(values["u"]),
        lambda values, c: {"u": _cubic(c)},
        lambda c: {"u": _cubic(c)},
        assemble,
    )
    holder["representation"] = representation
    return representation


def test_coefficient_route_eliminates_exact_seam_rows() -> None:
    initial = np.asarray([0.2, -0.4, 0.3, 0.15])
    field = _cubic(jnp.asarray(initial))
    declarations = (
        Periodic("u", PAIRING, target=0.5),
        Periodic("u", PAIRING, order=1),
    )
    prepared = prepare_periodic_projection(
        {"u": field},
        declarations,
        route="coefficient",
        representation=_cubic_representation(),
    )
    result = prepared.realize(
        {"u": field},
        context=ConditionEvaluationContext(prepared.condition, exact_required=True),
    )

    assert result.status is RealizationStatus.SUCCESS
    assert result.fields is not None
    assert result.stamp is not None and result.stamp.exact
    assert prepared.evidence.scope == "finite-representation"
    expected = initial + np.linalg.pinv(_JUMP_ROWS) @ (
        np.asarray([0.5, 0.0]) - _JUMP_ROWS @ initial
    )
    np.testing.assert_allclose(
        _cubic_coefficients(result.fields["u"]), expected, rtol=0.0, atol=1e-12
    )
    projected = _pointwise(result.fields["u"], SPACE)
    np.testing.assert_allclose(_jumps(projected, range(2)), [0.5, 0.0], atol=1e-12)


def _certified(period: float) -> DomainFunction:
    certificate = PeriodicInputCertificate(
        input_size=1,
        periodic_inputs=((0, period),),
        regularity=phx.DerivativeRegularity.smooth(),
    )
    return DomainFunction(
        domain=SPACE,
        deps=("x",),
        func=PointwiseEvaluator(
            lambda x: jnp.sin(jnp.pi * x[0]) + 0.3 * jnp.cos(2.0 * jnp.pi * x[0])
        ),
        metadata={PERIODIC_INPUT_CERTIFICATE_KEY: certificate},
    )


def test_construction_route_admits_certified_periodic_model() -> None:
    field = _certified(LENGTH)
    prepared = prepare_periodic_projection(
        {"u": field}, _seam(0, 1), route="construction"
    )
    result = prepared.realize(
        {"u": field}, context=ConditionEvaluationContext(prepared.condition)
    )

    assert result.status is RealizationStatus.UNCHANGED
    assert result.fields is not None and result.fields["u"] is field
    assert prepared.evidence.scope == "structural"
    certificate = field.metadata[PERIODIC_INPUT_CERTIFICATE_KEY]
    assert prepared.evidence.construction_certificates == (certificate.certificate_id,)
    assert prepared.admission(prepared.condition).writes == ()


@pytest.mark.parametrize(
    ("field", "declarations", "message"),
    [
        (
            _certified(LENGTH),
            (Periodic("u", PAIRING, transport=-1.0),),
            "homogeneous identity-transport",
        ),
        (
            _certified(LENGTH),
            (Periodic("u", PAIRING, target=0.5),),
            "homogeneous identity-transport",
        ),
        (BASE, _seam(0), "carries no PeriodicInputCertificate"),
        (_certified(1.0), _seam(0), "does not certify period"),
    ],
    ids=["antiperiodic", "affine-jump", "missing-certificate", "wrong-period"],
)
def test_construction_route_refuses_uncertified_relations(
    field: DomainFunction, declarations: tuple[Periodic, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_periodic_projection({"u": field}, declarations, route="construction")


def test_compiler_binds_preservation_of_compatible_walls_and_initial() -> None:
    domain = HyperRectangle(jnp.zeros(2), jnp.ones(2)) @ TimeInterval(0.0, 0.5)
    seam = PeriodicIdentification(domain, "x", component=0)
    lower_wall, upper_wall = physical_boundary(domain, (seam,))[:2]
    model = MLP(
        in_size=3,
        out_size="scalar",
        width_size=8,
        depth=2,
        activation=jnp.tanh,
        key=jr.key(0),
    )
    field = domain.Model("x", "t")(model)
    declarations = (
        Periodic("u", seam.pairing(), label="value"),
        Periodic("u", seam.pairing(), order=1, label="slope"),
    )
    prepared = prepare_periodic_projection(
        {"u": field}, declarations, route="analytic", data_compatibility="probed"
    )
    walls = (
        phx.conditions.Dirichlet("u", lower_wall, target=0.0, label="lower-wall"),
        phx.conditions.Dirichlet("u", upper_wall, target=0.0, label="upper-wall"),
    )
    initial = phx.conditions.Initial(
        "u",
        domain.component({"t": FixedStart()}),
        target=domain.Function("x")(
            lambda x: jnp.sin(2.0 * jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
        ),
    )
    program = phx.enforcement.compile(
        {"u": field},
        (
            *(EnforcementSpec(wall) for wall in walls),
            EnforcementSpec(initial),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=256),
    )

    (bound,) = program.realization_specs
    assert isinstance(bound.realization, PreparedPeriodicProjection)
    records = {
        record.contract_id: record.status
        for record in bound.realization.evidence.preserved
    }
    assert records == {
        "lower-wall": "certified",
        "upper-wall": "certified",
        "Initial:u:order-0": "probed",
    }
    enforced = program.apply({"u": field})
    for condition in (*declarations, *walls, initial):
        batch = condition.on.sample(PointSampling(5), key=jr.key(3))
        residual = jnp.asarray(condition.residual(enforced)(batch).data)
        assert float(jnp.max(jnp.abs(residual))) < 1e-12


def test_unlabeled_walls_bind_distinct_preserved_contracts() -> None:
    declarations = (Periodic("u", _BOX_X.pairing()),)
    prepared = prepare_periodic_projection(
        {"u": _BOX_FIELD}, declarations, route="analytic"
    )
    walls = physical_boundary(_BOX, (_BOX_X,))
    program = phx.enforcement.compile(
        {"u": _BOX_FIELD},
        (
            *(
                EnforcementSpec(phx.conditions.Dirichlet("u", wall, target=0.0))
                for wall in walls
            ),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=256),
    )

    (bound,) = program.realization_specs
    assert isinstance(bound.realization, PreparedPeriodicProjection)
    assert isinstance(bound.condition, Condition)
    admission = bound.realization.admission(bound.condition)
    assert len(admission.preserves) == len(walls) == 2
    assert len(set(admission.preserves)) == 2
    assert set(admission.establishes) >= {
        declaration.condition_id for declaration in declarations
    }


def _refused_composition(case: str) -> dict[str, Any]:
    walls = physical_boundary(_BOX, (_BOX_X,))
    fields: dict[str, DomainFunction] = {"u": _BOX_FIELD}
    declarations: tuple[Periodic, ...] = (Periodic("u", _BOX_X.pairing()),)
    specs: tuple[EnforcementSpec, ...] = ()
    interior: tuple[InteriorAnchors, ...] = ()
    match case:
        case "seam-wall":
            specs = (
                EnforcementSpec(
                    phx.conditions.Dirichlet(
                        "u", _BOX.component({"x": Boundary()}), target=0.0
                    )
                ),
            )
        case "nonperiodic-wall-data" | "uncertified-wall-data":
            specs = (
                EnforcementSpec(
                    phx.conditions.Dirichlet(
                        "u", walls[0], target=_BOX.Function("x")(lambda x: x[0])
                    )
                ),
            )
        case "antiperiodic-constant-wall":
            declarations = (Periodic("u", _BOX_X.pairing(), transport=-1.0),)
            specs = (
                EnforcementSpec(phx.conditions.Dirichlet("u", walls[0], target=1.0)),
            )
        case "interior-anchors":
            interior = (
                InteriorAnchors(
                    "u", points={"x": jnp.asarray([[0.5, 0.5]])}, values=jnp.ones(1)
                ),
            )
        case "opaque-chart-writes":
            fields["v"] = _BOX.Function("x")(lambda x: x[0])
            chart = CallableCorrectionChart(
                lambda values: jnp.zeros(()),
                lambda values, coordinates: dict(values),
                chart_id="opaque-chart",
            )
            other = Periodic("v", _BOX_X.pairing()).as_condition()
            specs = (EnforcementSpec(other, realization=NonlinearFieldRetraction(chart)),)
        case _:
            raise ValueError(case)
    prepared = prepare_periodic_projection(
        {"u": fields["u"]},
        declarations,
        route="analytic",
        data_compatibility="probed" if case == "nonperiodic-wall-data" else "certified",
    )
    return {
        "functions": fields,
        "specs": (*specs, EnforcementSpec(prepared.condition, realization=prepared)),
        "interior": interior,
    }


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("seam-wall", "a seam is not a wall"),
        ("nonperiodic-wall-data", "incompatible with its periodic seams"),
        ("uncertified-wall-data", "seam compatibility cannot be proven"),
        ("antiperiodic-constant-wall", "incompatible with periodic declaration"),
        ("interior-anchors", "InteriorAnchors on periodic field"),
        ("opaque-chart-writes", "may write periodically enforced fields"),
    ],
    ids=[
        "seam-wall",
        "nonperiodic-wall-data",
        "uncertified-wall-data",
        "antiperiodic-constant-wall",
        "interior-anchors",
        "opaque-chart-writes",
    ],
)
def test_compiler_refuses_nonpreserving_compositions(case: str, message: str) -> None:
    composition = _refused_composition(case)
    with pytest.raises(ValueError, match=message):
        phx.enforcement.compile(
            composition["functions"],
            composition["specs"],
            interior=composition["interior"],
            options=EnforcementOptions(num_reference=256),
        )


def test_noncommuting_event_transports_are_refused() -> None:
    value = phx.conditions.ArrayCodomain.from_shape((2,))
    field = _BOX.Function("x")(lambda x: jnp.stack((1.0 + x[0], 2.0 + x[1])))
    declarations = (
        Periodic(
            "u",
            _BOX_X.pairing(),
            transport=phx.conditions.EventLinearMap(jnp.diag(jnp.asarray([1.0, -1.0]))),
            value=value,
        ),
        Periodic(
            "u",
            _BOX_Y.pairing(),
            transport=phx.conditions.EventLinearMap(
                jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
            ),
            value=value,
        ),
    )

    with pytest.raises(ValueError, match="do not commute"):
        prepare_periodic_projection({"u": field}, declarations, route="analytic")


def test_hard_preparation_requires_the_identification_face_maps() -> None:
    lower = IDENTIFICATION.face_map("lower")
    forged = phx.domain.PairedSupport(
        IDENTIFICATION.face("lower"),
        lower,
        lower,
        pairing_id="forged-seam",
        left_patch_id="fundamental-domain",
        right_patch_id="fundamental-domain",
        normal=IDENTIFICATION.normal(),
        topology="periodic-interface",
        identification=IDENTIFICATION,
    )

    with pytest.raises(ValueError, match="identification's face maps"):
        prepare_periodic_projection(
            {"u": BASE}, (Periodic("u", forged, target=1.0),), route="analytic"
        )


def test_construction_route_rechecks_the_realized_fields() -> None:
    prepared = prepare_periodic_projection(
        {"u": _certified(LENGTH)}, _seam(0), route="construction"
    )
    result = prepared.realize(
        {"u": BASE}, context=ConditionEvaluationContext(prepared.condition)
    )

    assert result.status is RealizationStatus.UNSUPPORTED
    assert result.fields is None
    assert "PeriodicInputCertificate" in (result.message or "")


_PLANE = Interval1d(0.0, 1.0) @ Interval1d(0.0, 1.0, label="y")
_PLANE_X = PeriodicIdentification(_PLANE, "x")
_PLANE_FIELD = _PLANE.Function("x", "y")(lambda x, y: jnp.sin(x[0]) + x[0] * y[0])


def _robin_program(coefficient: Any) -> Any:
    prepared = prepare_periodic_projection(
        {"u": _PLANE_FIELD}, (Periodic("u", _PLANE_X.pairing()),), route="analytic"
    )
    wall = phx.conditions.Robin(
        "u",
        _PLANE.component({"y": Boundary()}),
        dirichlet_coefficient=coefficient,
        neumann_coefficient=1.0,
        var="y",
        label="robin-wall",
    )
    return phx.enforcement.compile(
        {"u": _PLANE_FIELD},
        (
            EnforcementSpec(wall),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=256),
    )


def test_robin_wall_with_coefficient_along_the_seam_is_refused() -> None:
    coefficient = _PLANE.Function("x")(lambda x: 1.0 + x[0])
    with pytest.raises(ValueError, match="boundary-operator coefficients"):
        _robin_program(coefficient)


def test_constant_robin_wall_is_preserved_exactly() -> None:
    program = _robin_program(2.0)
    (bound,) = program.realization_specs
    assert isinstance(bound.realization, PreparedPeriodicProjection)
    assert bound.realization.evidence.preserved_ids == ("robin-wall",)


def test_construction_route_refuses_local_hard_contracts() -> None:
    domain = SPACE @ TimeInterval(0.0, 1.0)
    certificate = PeriodicInputCertificate(
        input_size=2,
        periodic_inputs=((0, LENGTH),),
        regularity=phx.DerivativeRegularity.smooth(),
    )
    field = DomainFunction(
        domain=domain,
        deps=("x", "t"),
        func=PointwiseEvaluator(lambda x, t: jnp.sin(jnp.pi * x[0]) * (1.0 + t)),
        metadata={PERIODIC_INPUT_CERTIFICATE_KEY: certificate},
    )
    identification = PeriodicIdentification(domain, "x")
    prepared = prepare_periodic_projection(
        {"u": field}, (Periodic("u", identification.pairing()),), route="construction"
    )
    initial = phx.conditions.Initial(
        "u", domain.component({"t": FixedStart()}), target=0.0
    )

    with pytest.raises(ValueError, match="does not preserve local hard contracts"):
        phx.enforcement.compile(
            {"u": field},
            (
                EnforcementSpec(initial),
                EnforcementSpec(prepared.condition, realization=prepared),
            ),
            options=EnforcementOptions(num_reference=256),
        )


def test_probed_contracts_are_not_claimed_preserved() -> None:
    domain = HyperRectangle(jnp.zeros(2), jnp.ones(2)) @ TimeInterval(0.0, 0.5)
    seam = PeriodicIdentification(domain, "x", component=0)
    lower_wall, upper_wall = physical_boundary(domain, (seam,))[:2]
    field = domain.Function("x", "t")(lambda x, t: jnp.cos(x[0]) * x[1] + t)
    prepared = prepare_periodic_projection(
        {"u": field},
        (Periodic("u", seam.pairing()),),
        route="analytic",
        data_compatibility="probed",
    )
    transverse = domain.Function("t")(lambda t: jnp.sin(3.0 * t))
    walls = (
        phx.conditions.Dirichlet("u", lower_wall, target=transverse, label="time-wall"),
        phx.conditions.Dirichlet(
            "u",
            upper_wall,
            target=domain.Function("x")(lambda x: jnp.cos(2.0 * jnp.pi * x[0])),
            label="periodic-wall",
        ),
    )
    program = phx.enforcement.compile(
        {"u": field},
        (
            *(EnforcementSpec(wall) for wall in walls),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=256),
    )

    (bound,) = program.realization_specs
    assert isinstance(bound.realization, PreparedPeriodicProjection)
    assert isinstance(bound.condition, Condition)
    statuses = {
        record.contract_id: record.status
        for record in bound.realization.evidence.preserved
    }
    assert statuses == {"time-wall": "certified", "periodic-wall": "probed"}
    assert bound.realization.admission(bound.condition).preserves == ("time-wall",)


def test_declared_event_shape_refuses_a_scalar_field() -> None:
    declaration = Periodic(
        "u", PAIRING, value=phx.conditions.ArrayCodomain.from_shape((2,))
    )
    batch = PAIRING.component.sample(PointSampling(2), key=jr.key(0))
    with pytest.raises(ValueError, match="event shape"):
        declaration.residual({"u": BASE})(batch)


@pytest.mark.strict_jax
def test_single_precision_field_residual_is_strict_clean() -> None:
    field = SPACE.Function("x")(lambda x: (x[0] * (LENGTH - x[0])).astype(jnp.float32))
    batch = PAIRING.component.sample(PointSampling(2), key=jr.key(0))
    residual = jnp.asarray(Periodic("u", PAIRING).residual({"u": field})(batch).data)

    assert residual.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(residual))) == 0.0


def test_construction_route_realizes_inside_jit() -> None:
    field = _certified(LENGTH)
    prepared = prepare_periodic_projection({"u": field}, _seam(0), route="construction")
    context = ConditionEvaluationContext(prepared.condition)
    query = SPACE.component().points({"x": QUERY[:, None]})

    @jax.jit
    def evaluate(shift: Array) -> Array:
        result = prepared.realize({"u": field}, context=context)
        assert result.fields is not None
        return jnp.asarray(result.fields["u"](query).data) + shift

    np.testing.assert_allclose(
        evaluate(jnp.asarray(0.0)), _values(field, SPACE, QUERY), rtol=0.0, atol=1e-15
    )

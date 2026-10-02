#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transported periodic trace relations across identified seams.

Every oracle is the closed-form relation ``T[target](upper) - Gamma S[source](lower)
- g`` evaluated by hand on manufactured nonperiodic fields whose values vary
along the seam (transverse coordinate), so a wrong-side pullback, a collapsed
transverse coordinate, or differentiation after the face projection is visible.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


# Box ``[0, 2] x [-1, 3]`` identified along component 1: source face ``x_1 = -1``,
# target face ``x_1 = 3``; ``x_0`` is the transverse seam coordinate.
_BOX = phx.domain.HyperRectangle(np.asarray([0.0, -1.0]), np.asarray([2.0, 3.0]))
_LOWER = -1.0
_UPPER = 3.0
_SEAM = phx.domain.PeriodicIdentification(_BOX, "x", component=1).pairing()


def _u(x0: np.ndarray, x1: float) -> np.ndarray:
    return (1.0 + x0**2) * (x1**3 + 0.3 * x1)


def _du(x0: np.ndarray, x1: float) -> np.ndarray:
    return (1.0 + x0**2) * (3.0 * x1**2 + 0.3)


def _v(x0: np.ndarray, x1: float) -> np.ndarray:
    return np.cos(x0) * x1**2


_U = _BOX.Function("x")(lambda x: (1.0 + x[0] ** 2) * (x[1] ** 3 + 0.3 * x[1]))
_V = _BOX.Function("x")(lambda x: jnp.cos(x[0]) * x[1] ** 2)


def _seam_batch(seam: Any, count: int = 7) -> Any:
    return seam.component.sample(phx.domain.PointSampling(count), key=jr.key(0))


def _transverse(batch: Any) -> np.ndarray:
    return np.asarray(batch.points["x"].data)[:, 0]


def _space_time_case(order: int) -> tuple[Any, Any, Callable[[Any], np.ndarray]]:
    # ``u(x, t) = (1 + t^2) (x^3 + 0.3 x)`` on ``[0, 2] x [0, 1]`` identified in x:
    # ``u(2, t) - u(0, t) = 8.6 (1 + t^2)``, ``u_x(2, t) - u_x(0, t) = 12 (1 + t^2)``.
    domain = phx.domain.Interval1d(0.0, 2.0) @ phx.domain.TimeInterval(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(domain, "x").pairing()
    field = domain.Function("x", "t")(
        lambda x, t: (1.0 + t**2) * (x[0] ** 3 + 0.3 * x[0])
    )
    jump = 8.6 if order == 0 else 12.0

    def oracle(batch: Any) -> np.ndarray:
        t = np.asarray(batch.points["t"].data).reshape(-1)
        return jump * (1.0 + t**2)

    return seam, field, oracle


def _box_case(order: int) -> tuple[Any, Any, Callable[[Any], np.ndarray]]:
    jet = _u if order == 0 else _du

    def oracle(batch: Any) -> np.ndarray:
        x0 = _transverse(batch)
        return jet(x0, _UPPER) - jet(x0, _LOWER)

    return _SEAM, _U, oracle


@pytest.mark.parametrize(
    ("build", "order"),
    (
        pytest.param(_box_case, 0, id="box-value"),
        pytest.param(_box_case, 1, id="box-derivative"),
        pytest.param(_space_time_case, 0, id="space-time-value"),
        pytest.param(_space_time_case, 1, id="space-time-derivative"),
    ),
)
def test_same_field_residual_matches_the_two_physical_traces(
    build: Callable[[int], tuple[Any, Any, Callable[[Any], np.ndarray]]], order: int
) -> None:
    seam, field, oracle = build(order)
    batch = _seam_batch(seam)
    residual = phx.conditions.Periodic("u", seam, order=order).residual({"u": field})

    np.testing.assert_allclose(residual(batch).data, oracle(batch), rtol=1e-12)


@pytest.mark.parametrize(
    ("options", "oracle"),
    (
        pytest.param(
            {"transport": -1.0},
            lambda x0: _u(x0, _UPPER) + _u(x0, _LOWER),
            id="antiperiodic",
        ),
        pytest.param(
            {"target": 2.5},
            lambda x0: _u(x0, _UPPER) - _u(x0, _LOWER) - 2.5,
            id="affine-constant-jump",
        ),
        pytest.param(
            # A field target is evaluated on the seam through the source map.
            {"target": _BOX.Function("x")(lambda x: jnp.sin(x[0]) + x[1])},
            lambda x0: _u(x0, _UPPER) - _u(x0, _LOWER) - (np.sin(x0) + _LOWER),
            id="affine-field-jump",
        ),
        pytest.param(
            {"transport": 0.5, "order": 1},
            lambda x0: _du(x0, _UPPER) - 0.5 * _du(x0, _LOWER),
            id="scaled-derivative",
        ),
    ),
)
def test_scalar_transport_and_affine_target_follow_the_declared_relation(
    options: dict[str, Any], oracle: Callable[[np.ndarray], np.ndarray]
) -> None:
    batch = _seam_batch(_SEAM)
    residual = phx.conditions.Periodic("u", _SEAM, **options).residual({"u": _U})

    np.testing.assert_allclose(
        residual(batch).data, oracle(_transverse(batch)), rtol=1e-12
    )


def _bloch(x0: np.ndarray, x1: float) -> np.ndarray:
    return (1.0 + x0**2) * np.exp(1j * x1) * (1.0 + 0.2j * x1)


def _bloch_derivative(x0: np.ndarray, x1: float) -> np.ndarray:
    return (1.0 + x0**2) * np.exp(1j * x1) * (1.2j - 0.2 * x1)


_PHASE = np.exp(0.7j)


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    ("options", "oracle"),
    (
        pytest.param(
            {"transport": jnp.asarray(_PHASE, dtype=jnp.complex128)},
            lambda x0: _bloch(x0, _UPPER) - _PHASE * _bloch(x0, _LOWER),
            id="bloch-phase",
        ),
        pytest.param(
            {"transport": -1.0, "target": 0.5},
            lambda x0: _bloch(x0, _UPPER) + _bloch(x0, _LOWER) - 0.5,
            id="real-antiperiodic-affine",
        ),
        pytest.param(
            {
                "transport": jnp.asarray(_PHASE, dtype=jnp.complex128),
                "trace_actions": (
                    phx.conditions.JetAction({1: 2.0}),
                    phx.conditions.JetAction({1: 2.0}),
                ),
            },
            lambda x0: (
                2.0 * _bloch_derivative(x0, _UPPER)
                - 2.0 * _PHASE * _bloch_derivative(x0, _LOWER)
            ),
            id="bloch-real-flux",
        ),
    ),
)
def test_complex_field_seams_promote_real_constants_explicitly(
    options: dict[str, Any], oracle: Callable[[np.ndarray], np.ndarray]
) -> None:
    def bloch(x: Any) -> Any:
        x0, x1 = x.astype(jnp.complex128)
        return (1.0 + x0**2) * jnp.exp(1j * x1) * (1.0 + 0.2j * x1)

    batch = _seam_batch(_SEAM)
    residual = phx.conditions.Periodic("u", _SEAM, **options).residual(
        {"u": _BOX.Function("x")(bloch)}
    )(batch)

    assert residual.data.dtype == jnp.complex128
    np.testing.assert_allclose(residual.data, oracle(_transverse(batch)), rtol=1e-12)


def test_event_linear_map_transports_the_source_event() -> None:
    matrix = np.asarray([[0.0, 1.0], [-1.0, 0.5]])
    field = _BOX.Function("x")(
        lambda x: jnp.stack((x[0] * x[1] ** 2, jnp.cos(x[1]) + x[0]))
    )

    def exact(x0: np.ndarray, x1: float) -> np.ndarray:
        return np.stack((x0 * x1**2, np.cos(x1) + x0), axis=-1)

    batch = _seam_batch(_SEAM)
    x0 = _transverse(batch)
    residual = phx.conditions.Periodic(
        "w",
        _SEAM,
        transport=phx.conditions.EventLinearMap(matrix),
        value=phx.conditions.ArrayCodomain.from_shape((2,)),
    ).residual({"w": field})(batch)

    np.testing.assert_allclose(
        residual.data,
        exact(x0, _UPPER) - exact(x0, _LOWER) @ matrix.T,
        rtol=1e-12,
    )


def test_jet_action_pair_matches_constitutive_flux() -> None:
    # Source flux ``2 du/dx_1``; target ``0.5 u + 3 du/dx_1``.
    condition = phx.conditions.Periodic(
        "u",
        _SEAM,
        trace_actions=(
            phx.conditions.JetAction({1: 2.0}),
            phx.conditions.JetAction({0: 0.5, 1: 3.0}),
        ),
    )
    batch = _seam_batch(_SEAM)
    x0 = _transverse(batch)

    np.testing.assert_allclose(
        condition.residual({"u": _U})(batch).data,
        0.5 * _u(x0, _UPPER) + 3.0 * _du(x0, _UPPER) - 2.0 * _du(x0, _LOWER),
        rtol=1e-12,
    )


def _distinct_source() -> Any:
    # Source ``v`` on the lower face, target ``u`` on the upper face.
    return phx.conditions.Periodic(("v", "u"), _SEAM, transport=2.0, target=0.25)


def test_distinct_source_and_target_fields_keep_their_roles() -> None:
    batch = _seam_batch(_SEAM)
    x0 = _transverse(batch)

    np.testing.assert_allclose(
        _distinct_source().residual({"u": _U, "v": _V})(batch).data,
        _u(x0, _UPPER) - 2.0 * _v(x0, _LOWER) - 0.25,
        rtol=1e-12,
    )


def test_lowering_is_a_certified_linear_action_with_separate_target() -> None:
    condition = _distinct_source().as_condition()
    batch = _seam_batch(_SEAM)
    x0 = _transverse(batch)
    shifted = {
        "u": _BOX.Function("x")(lambda x: x[0] - x[1]),
        "v": _BOX.Function("x")(lambda x: x[0] * x[1]),
    }

    assert condition.fields.sources == ("v", "u")
    for spec in condition.fields.fields:
        # A self-seam binds the source fields on the whole fundamental domain.
        assert isinstance(
            spec.codomain.support.spec.selection_for("x"), phx.domain.Interior
        )
    assert condition.operator.capabilities.is_linear
    assert isinstance(condition.relation, phx.conditions.Equality)
    np.testing.assert_allclose(condition.relation.target(batch).data, 0.25)

    action = condition.operator.apply({"u": _U, "v": _V})(batch).data
    np.testing.assert_allclose(action, _u(x0, _UPPER) - 2.0 * _v(x0, _LOWER), rtol=1e-12)
    combined = condition.operator.apply({"u": _U + shifted["u"], "v": _V + shifted["v"]})(
        batch
    ).data
    separate = action + condition.operator.apply(shifted)(batch).data
    np.testing.assert_allclose(combined, separate, rtol=1e-12)


def test_lowering_refuses_reordered_source_fields() -> None:
    support = phx.conditions.FieldCodomain(_BOX.component())
    reordered = (
        phx.conditions.FieldSpec("u", support),
        phx.conditions.FieldSpec("v", support),
    )
    with pytest.raises(ValueError, match="declared field order"):
        _distinct_source().as_condition(fields=reordered)


def _interface() -> Any:
    cover = phx.domain.cartesian_subdomain_cover(_BOX, "x", (1, 2))
    return cover.pairings[0]


@pytest.mark.parametrize(
    ("build", "error", "match"),
    (
        pytest.param(
            lambda: phx.conditions.Periodic("u", _interface()),
            ValueError,
            "periodic-interface pairing",
            id="non-periodic-pairing",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic("u", _SEAM.component),
            TypeError,
            "PairedSupport",
            id="not-a-pairing",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic("u", _SEAM, transport=np.asarray([1.0, 2.0])),
            ValueError,
            "must be scalar",
            id="vector-transport",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic("u", _SEAM, transport=0.0),
            ValueError,
            "finite and nonzero",
            id="zero-transport",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic(
                "w", _SEAM, transport=phx.conditions.EventLinearMap(np.eye(2))
            ),
            ValueError,
            "rank-one value codomain",
            id="event-map-on-scalar-value",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic(
                "w",
                _SEAM,
                transport=phx.conditions.EventLinearMap(np.eye(3)),
                value=phx.conditions.ArrayCodomain.from_shape((2,)),
            ),
            ValueError,
            "rank-one value codomain",
            id="event-map-size-mismatch",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic("u", _SEAM, target=np.asarray([1.0, 2.0])),
            ValueError,
            "does not match the value codomain",
            id="target-shape",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic("u", _SEAM, target=np.inf),
            ValueError,
            "must be finite",
            id="nonfinite-target",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic(
                "u",
                _SEAM,
                target=phx.domain.TimeInterval(0.0, 1.0).Function("t")(lambda t: t),
            ),
            ValueError,
            "seam domain",
            id="foreign-target",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic(
                "u",
                _SEAM,
                order=1,
                trace_actions=(
                    phx.conditions.JetAction({1: 1.0}),
                    phx.conditions.JetAction({1: 1.0}),
                ),
            ),
            ValueError,
            "use order=0",
            id="trace-actions-with-order",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic(
                "u",
                _SEAM,
                trace_actions=(  # ty: ignore[invalid-argument-type]
                    phx.conditions.JetAction({1: 1.0}),
                ),
            ),
            TypeError,
            "JetAction pair",
            id="incomplete-trace-actions",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic(("u", "u"), _SEAM),
            ValueError,
            "binds its field once",
            id="repeated-field",
        ),
        pytest.param(
            lambda: phx.conditions.Periodic("u", _SEAM, order=-1),
            ValueError,
            "nonnegative integer",
            id="negative-order",
        ),
    ),
)
def test_periodic_refuses_invalid_declaration(
    build: Callable[[], Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        build()


@pytest.mark.parametrize(
    ("terms", "error", "match"),
    (
        pytest.param({}, TypeError, "nonempty", id="empty"),
        pytest.param({-1: 1.0}, ValueError, "nonnegative", id="negative-order"),
        pytest.param({1: 0.0}, ValueError, "nonzero", id="zero-coefficient"),
        pytest.param({1: np.nan}, ValueError, "finite", id="nonfinite-coefficient"),
    ),
)
def test_jet_action_refuses_invalid_terms(
    terms: dict[int, float], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        phx.conditions.JetAction(terms)


def test_jet_action_refuses_non_field_and_non_identification_inputs() -> None:
    action = phx.conditions.JetAction.derivative(1)
    identification = phx.domain.PeriodicIdentification(_BOX, "x", component=1)

    with pytest.raises(TypeError):
        action.apply(0, identification)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        action.apply(_U, 0)  # ty: ignore[invalid-argument-type]


@pytest.mark.strict_jax
def test_complex_jet_action_on_real_field_is_strict_clean() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    declaration = phx.conditions.Periodic(
        "u",
        seam,
        trace_actions=(
            phx.conditions.JetAction({1: 1j}),
            phx.conditions.JetAction({0: 1.0}),
        ),
    )
    field = space.Function("x")(lambda x: x[0] ** 2)
    batch = seam.component.sample(phx.domain.PointSampling(2), key=jr.key(0))
    residual = jnp.asarray(declaration.residual({"u": field})(batch).data)

    # trace_actions are (source, target): T[u](1) - S[u](0) with S = i d/dx and
    # T = identity gives u(1) - i u'(0) = 1 - 0.
    assert residual.dtype == jnp.complex128
    np.testing.assert_allclose(residual, 1.0, rtol=0.0, atol=1e-15)


def test_lowering_refuses_an_incompatible_codomain_override() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    declaration = phx.conditions.Periodic("u", seam)
    override = phx.conditions.FieldCodomain(
        seam.component, phx.conditions.ArrayCodomain.from_shape((2,))
    )

    with pytest.raises(ValueError, match="codomain override"):
        declaration.as_condition(codomain=override)


def test_declared_dtype_refuses_a_trace_of_another_dtype() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    declaration = phx.conditions.Periodic(
        "u", seam, value=phx.conditions.ArrayCodomain(dtype=jnp.complex128)
    )
    field = space.Function("x")(lambda x: x[0])
    batch = seam.component.sample(phx.domain.PointSampling(2), key=jr.key(0))

    with pytest.raises(ValueError, match="dtype"):
        declaration.residual({"u": field})(batch)


def test_real_declared_dtype_refuses_complex_transport() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()

    with pytest.raises(ValueError, match="real declared value dtype"):
        phx.conditions.Periodic(
            "u",
            seam,
            transport=1j,
            value=phx.conditions.ArrayCodomain(dtype=jnp.float64),
        )


@pytest.mark.strict_jax
def test_seam_constants_adopt_the_declared_field_precision() -> None:
    space = phx.domain.Interval1d(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(space, "x").pairing()
    declaration = phx.conditions.Periodic(
        "u",
        seam,
        transport=1j,
        value=phx.conditions.ArrayCodomain(dtype=jnp.complex64),
    )
    field = space.Function("x")(
        lambda x: (
            jnp.asarray(x[0], dtype=jnp.complex64)
            + jnp.asarray(1.0j, dtype=jnp.complex64)
        )
    )
    batch = seam.component.sample(phx.domain.PointSampling(2), key=jr.key(0))
    residual = jnp.asarray(declaration.residual({"u": field})(batch).data)

    # u(1) - i u(0) = (1 + i) - i * i = 2 + i.
    assert residual.dtype == jnp.complex64
    np.testing.assert_allclose(residual, 2.0 + 1.0j, rtol=0.0, atol=1e-6)

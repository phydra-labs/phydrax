from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.nn import PeriodicInputCertificate
from phydrax.nn.layers import (
    Dropout,
    ExplicitFourierFeatureEmbeddings,
    HybridFourierFeatureEmbeddings,
    MultiscaleFourierFeatureEmbeddings,
    RandomFourierFeatureEmbeddings,
    TrainableFourierFeatureEmbeddings,
)
from phydrax.nn.models import MLP, Sequential


KEY = "periodic_input_certificate"
PERIOD = 2.0


def _periodic_embedding() -> ExplicitFourierFeatureEmbeddings:
    return ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=1,
        coordinate=0,
        period=PERIOD,
        modes=(1, 2, 3),
        phases=0.4,
        include_constant=True,
    )


def _bound_composite(activation: Callable[[Array], Array]) -> phx.domain.DomainFunction:
    embedding = _periodic_embedding()
    mlp = MLP(
        in_size=embedding.out_size,
        out_size="scalar",
        width_size=8,
        depth=2,
        activation=activation,
        key=jr.key(3),
    )
    return phx.domain.Interval1d(0.0, PERIOD).Model("x")(Sequential((embedding, mlp)))


def test_periodic_modes_certify_their_coordinate() -> None:
    embedding = ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=3,
        coordinate=1,
        period=PERIOD,
        modes=(np.int64(1), 2, 5),
        passthrough=(0, 2),
    )
    certificate = embedding.model_metadata()[KEY]

    assert isinstance(certificate, PeriodicInputCertificate)
    assert certificate.input_size == 3
    assert certificate.periodic_inputs == ((1, PERIOD),)
    assert certificate.period_of(1) == PERIOD
    assert certificate.period_of(0) is None
    assert certificate.supports_order(4)


def test_explicit_lattice_wavevectors_certify_declared_inputs() -> None:
    embedding = ExplicitFourierFeatureEmbeddings(
        in_size=2,
        wavevectors=jnp.asarray(
            [[jnp.pi, 1.3], [-2.0 * jnp.pi, 0.7], [0.0, 2.0]], dtype=jnp.float64
        ),
        periodic_inputs={0: PERIOD},
    )
    point = jnp.asarray([0.3, -0.8], dtype=jnp.float64)
    shifted = point.at[0].add(PERIOD)

    assert embedding.model_metadata()[KEY].periodic_inputs == ((0, PERIOD),)
    np.testing.assert_allclose(embedding(shifted), embedding(point), atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(embedding)(shifted), jax.jacfwd(embedding)(point), atol=1e-12
    )


@pytest.mark.parametrize(
    "embedding",
    [
        pytest.param(
            ExplicitFourierFeatureEmbeddings(
                in_size=2, wavevectors=jnp.asarray([[jnp.pi, 0.0]], dtype=jnp.float64)
            ),
            id="explicit-undeclared",
        ),
        pytest.param(
            RandomFourierFeatureEmbeddings(in_size=2, out_size=8, key=jr.key(0)),
            id="random",
        ),
        pytest.param(
            TrainableFourierFeatureEmbeddings(in_size=2, out_size=8, key=jr.key(1)),
            id="trainable",
        ),
        pytest.param(
            MultiscaleFourierFeatureEmbeddings(in_size=2, scales=(1.0, 2.0)),
            id="multiscale",
        ),
        pytest.param(
            HybridFourierFeatureEmbeddings(
                in_size=2,
                deterministic_wavevectors=jnp.asarray([[jnp.pi, 0.0]], dtype=jnp.float64),
                random_out_size=4,
                key=jr.key(2),
            ),
            id="hybrid",
        ),
        pytest.param(
            MLP(in_size=2, out_size="scalar", width_size=4, depth=1, key=jr.key(4)),
            id="mlp",
        ),
    ],
)
def test_uncertified_models_bind_without_a_periodic_certificate(
    embedding: phx.AbstractArrayModel,
) -> None:
    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    field = domain.Model("x", "t")(embedding)

    assert KEY not in field.metadata
    assert all(
        record[0] != "periodic-input"
        for record in embedding.model_execution_contract().certificates
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: ExplicitFourierFeatureEmbeddings.from_periodic_modes(
                in_size=2, coordinate=0, period=PERIOD, modes=(1,), passthrough=(0,)
            ),
            id="from-periodic-modes",
        ),
        pytest.param(
            lambda: ExplicitFourierFeatureEmbeddings(
                in_size=2,
                wavevectors=jnp.asarray([[jnp.pi, 0.0]], dtype=jnp.float64),
                passthrough=(0,),
                periodic_inputs={0: PERIOD},
            ),
            id="explicit",
        ),
    ],
)
def test_passthrough_of_a_certified_coordinate_is_refused(
    build: Callable[[], ExplicitFourierFeatureEmbeddings],
) -> None:
    with pytest.raises(ValueError, match="not periodic"):
        build()


@pytest.mark.parametrize(
    ("modes", "error"),
    [
        pytest.param((1.5,), TypeError, id="non-integral"),
        pytest.param((2.0,), TypeError, id="integral-float"),
        pytest.param((True,), TypeError, id="bool"),
        pytest.param((0,), ValueError, id="zero"),
        pytest.param((1, 1), ValueError, id="duplicate"),
        pytest.param((), ValueError, id="empty"),
    ],
)
def test_periodic_modes_must_be_distinct_positive_integers(
    modes: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ExplicitFourierFeatureEmbeddings.from_periodic_modes(
            in_size=1,
            coordinate=0,
            period=PERIOD,
            modes=modes,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("periodic_inputs", "error"),
    [
        pytest.param({0: PERIOD * 1.001}, ValueError, id="off-lattice"),
        pytest.param({0: -PERIOD}, ValueError, id="negative-period"),
        pytest.param({0: float("inf")}, ValueError, id="infinite-period"),
        pytest.param({2: PERIOD}, ValueError, id="index-out-of-range"),
        pytest.param({}, ValueError, id="empty"),
    ],
)
def test_invalid_periodic_declarations_are_refused(
    periodic_inputs: dict[int, float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ExplicitFourierFeatureEmbeddings(
            in_size=2,
            wavevectors=jnp.asarray([[jnp.pi, 0.0], [3.0 * jnp.pi, 1.0]]),
            periodic_inputs=periodic_inputs,
        )


def test_bound_embedding_maps_certificate_to_flat_model_input() -> None:
    embedding = ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=2, coordinate=0, period=PERIOD, modes=(1, 4), passthrough=(1,)
    )
    domain = phx.domain.Interval1d(0.0, PERIOD) @ phx.domain.TimeInterval(0.0, 1.0)
    field = domain.Model("x", "t")(embedding)
    certificate = field.metadata[KEY]

    assert certificate.certificate_id == embedding.model_metadata()[KEY].certificate_id
    assert ("periodic-input", certificate.certificate_id) in {
        record[:2] for record in embedding.model_execution_contract().certificates
    }
    t = jnp.asarray(0.25, dtype=jnp.float64)
    x = jnp.asarray([0.3], dtype=jnp.float64)
    np.testing.assert_allclose(field.func(x + PERIOD, t), field.func(x, t), atol=1e-12)


def test_bound_composite_is_periodic_with_composite_regularity() -> None:
    field = _bound_composite(jax.nn.tanh)
    certificate = field.metadata[KEY]
    x = jnp.asarray([0.37], dtype=jnp.float64)
    shifted = x + PERIOD

    assert certificate.periodic_inputs == ((0, PERIOD),)
    assert certificate.supports_order(3)
    np.testing.assert_allclose(field.func(shifted), field.func(x), atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(field.func)(shifted), jax.jacfwd(field.func)(x), atol=1e-10
    )


def test_composite_binds_on_multiple_labels_with_flat_first_stage_packing() -> None:
    embedding = ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=2, coordinate=0, period=PERIOD, modes=(1, 2), passthrough=(1,)
    )
    model = Sequential(
        (
            embedding,
            MLP(
                in_size=embedding.out_size,
                out_size="scalar",
                width_size=8,
                depth=2,
                key=jr.key(8),
            ),
        )
    )
    domain = phx.domain.Interval1d(0.0, PERIOD) @ phx.domain.TimeInterval(0.0, 1.0)
    field = domain.Model("x", "t")(model)
    certificate = field.metadata[KEY]
    x = jnp.asarray([0.3], dtype=jnp.float64)
    t = jnp.asarray(0.25, dtype=jnp.float64)

    assert certificate.input_size == 2
    assert certificate.periodic_inputs == ((0, PERIOD),)
    np.testing.assert_allclose(
        field.func(x, t),
        model(jnp.asarray([0.3, 0.25], dtype=jnp.float64)),
        atol=1e-12,
    )
    np.testing.assert_allclose(field.func(x + PERIOD, t), field.func(x, t), atol=1e-12)
    assert not np.allclose(field.func(x, t + 0.5), field.func(x, t))


def test_composite_certificate_regularity_is_the_composite_regularity() -> None:
    embedding = _periodic_embedding()
    model = Sequential(
        (
            embedding,
            MLP(
                in_size=embedding.out_size,
                out_size="scalar",
                width_size=4,
                depth=1,
                activation=jax.nn.relu,
                key=jr.key(5),
            ),
        )
    )
    certificate = model.model_metadata()[KEY]

    assert (
        certificate.regularity == model.model_execution_contract().derivative.regularity
    )
    assert certificate.supports_order(0)
    assert not certificate.supports_order(1)
    assert certificate.periodic_inputs == ((0, PERIOD),)


def test_composite_with_undeclared_regularity_carries_no_certificate() -> None:
    field = _bound_composite(lambda value: value * jnp.sin(value))

    assert KEY not in field.metadata


def test_composite_with_uncertified_first_stage_carries_no_certificate() -> None:
    model = Sequential(
        (
            MLP(in_size=1, out_size=4, width_size=4, depth=1, key=jr.key(6)),
            MLP(in_size=4, out_size="scalar", width_size=4, depth=1, key=jr.key(7)),
        )
    )

    assert KEY not in model.model_metadata()


def test_transformed_fields_drop_the_periodic_certificate() -> None:
    field = _bound_composite(jax.nn.tanh).with_metadata(note="kept")
    domain = phx.domain.Interval1d(0.0, PERIOD)

    assert KEY in field.metadata
    assert KEY not in (field + 1.0).metadata
    pulled = phx.operators.pullback(field, {}, domain=domain)
    derivative = phx.operators.grad(field, var="x")
    assert KEY not in pulled.metadata
    assert KEY not in derivative.metadata
    assert pulled.metadata["note"] == "kept"
    assert derivative.metadata["note"] == "kept"


@pytest.mark.parametrize("harmonic", (1.0 + 1e-15, 1e14 + 0.25))
def test_near_integer_wavevectors_do_not_certify_exact_periodicity(
    harmonic: float,
) -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        ExplicitFourierFeatureEmbeddings(
            in_size=1,
            wavevectors=jnp.asarray(
                [[2.0 * jnp.pi * harmonic / PERIOD]], dtype=jnp.float64
            ),
            periodic_inputs={0: PERIOD},
        )


@pytest.mark.parametrize("period,modes", ((1e-308, (1,)), (PERIOD, (2**53 + 1,))))
def test_unrepresentable_periodic_modes_are_refused(
    period: float, modes: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        ExplicitFourierFeatureEmbeddings.from_periodic_modes(
            in_size=1, coordinate=0, period=period, modes=modes
        )


@pytest.mark.parametrize("leaf", ("embedding_matrix", "phases"))
def test_changed_fixed_fourier_data_cannot_execute_a_stale_certificate(
    leaf: str,
) -> None:
    embedding = _periodic_embedding()
    if leaf == "embedding_matrix":
        changed = eqx.tree_at(
            lambda model: model.embedding_matrix,
            embedding,
            embedding.embedding_matrix.at[0, 0].set(1.0),
        )
    else:
        changed = eqx.tree_at(
            lambda model: model.phases, embedding, embedding.phases.at[0].set(jnp.nan)
        )
    point = jnp.asarray([0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        eqx.filter_jit(embedding)(point), embedding(point), atol=1e-12
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="Certified Fourier"):
        eqx.filter_jit(changed)(point)


def test_uncertified_coordinate_can_change_without_breaking_the_lattice() -> None:
    embedding = ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=2, coordinate=0, period=PERIOD, modes=(1, 2)
    )
    changed = eqx.tree_at(
        lambda model: model.embedding_matrix,
        embedding,
        embedding.embedding_matrix.at[:, 1].set(
            jnp.asarray([1.3, 0.7], dtype=jnp.float64)
        ),
    )
    point = jnp.asarray([0.3, -0.7], dtype=jnp.float64)
    evaluate = eqx.filter_jit(changed)
    np.testing.assert_allclose(
        evaluate(point.at[0].add(PERIOD)), evaluate(point), atol=1e-12
    )


def test_bound_certificate_requires_matching_packed_input_size() -> None:
    embedding = ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=2, coordinate=0, period=PERIOD, modes=(1, 2)
    )
    with pytest.raises(ValueError, match="input_size"):
        phx.domain.Interval1d(0.0, PERIOD).Model("x")(embedding)


def test_metadata_annotations_cannot_replace_existing_construction_evidence() -> None:
    field = _bound_composite(jax.nn.tanh)
    wrong_period = PeriodicInputCertificate(
        input_size=1,
        periodic_inputs=((0, 3.0),),
        regularity=phx.DerivativeRegularity.smooth(),
    )
    with pytest.raises(ValueError, match="Cannot replace source-bound"):
        field.with_metadata(periodic_input_certificate=wrong_period)
    annotated = field.with_metadata(note="retained")
    point = jnp.asarray([0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        annotated.func(point + PERIOD), annotated.func(point), atol=1e-12
    )


def test_current_evaluator_metadata_reflects_replaced_model_regularity() -> None:
    field = _bound_composite(jax.nn.tanh)
    replacement = _bound_composite(jax.nn.relu)
    assert isinstance(replacement.func, phx.domain.ConcatenatedModelEvaluator)
    changed_model = replacement.func.raw_model
    changed = eqx.tree_at(lambda item: item.func.raw_model, field, changed_model)
    stored_certificate = changed.metadata[KEY]
    current_certificate = changed.func.model_metadata()[KEY]
    assert stored_certificate.supports_order(1)
    assert not current_certificate.supports_order(1)
    point = jnp.asarray([0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        changed.func(point + PERIOD), changed.func(point), atol=1e-12
    )


def test_nonperiodic_generator_cannot_inherit_observable_construction_evidence() -> None:
    domain = phx.domain.Interval1d(0.0, PERIOD)
    observable = phx.domain.DomainFunction(
        domain=domain,
        deps=("x",),
        func=phx.domain.PointwiseEvaluator(lambda x: jnp.sin(jnp.pi * x[0])),
        metadata={
            KEY: PeriodicInputCertificate(
                input_size=1,
                periodic_inputs=((0, PERIOD),),
                regularity=phx.DerivativeRegularity.smooth(),
            ),
            "note": "retained",
        },
    )
    drift = domain.Function("x")(lambda x: x)
    generator = phx.operators.kolmogorov_generator(observable, drift, var="x")
    point = jnp.asarray([0.25], dtype=jnp.float64)
    np.testing.assert_allclose(generator.func(point), 0.25 * jnp.pi / jnp.sqrt(2.0))
    np.testing.assert_allclose(
        generator.func(point + PERIOD), 2.25 * jnp.pi / jnp.sqrt(2.0)
    )
    assert generator.metadata["note"] == "retained"
    condition = phx.conditions.Periodic(
        "u", phx.domain.PeriodicIdentification(domain, "x").pairing()
    )
    with pytest.raises(ValueError, match="carries no PeriodicInputCertificate"):
        phx.enforcement.prepare_periodic_projection(
            {"u": generator}, (condition,), route="construction"
        )


def test_stochastic_pipeline_is_certified_only_in_its_inference_state() -> None:
    embedding = _periodic_embedding()
    model = Sequential((embedding, Dropout(embedding.out_size, p=0.5)))
    assert KEY not in model.model_metadata()
    inference = eqx.nn.inference_mode(model)
    assert inference.model_metadata()[KEY].supports_order(2)
    point = jnp.asarray([0.3], dtype=jnp.float64)
    evaluate = eqx.filter_jit(inference)
    np.testing.assert_allclose(evaluate(point + PERIOD), evaluate(point), atol=1e-12)
    assert KEY not in eqx.nn.inference_mode(inference, value=False).model_metadata()


@pytest.mark.parametrize("variable", (True, False), ids=("variable", "clipped"))
def test_delays_cannot_inherit_source_periodicity_evidence(variable: bool) -> None:
    domain = phx.domain.TimeInterval(0.0, PERIOD)
    source = phx.domain.DomainFunction(
        domain=domain,
        deps=("t",),
        func=phx.domain.PointwiseEvaluator(lambda t: jnp.cos(jnp.pi * t)),
        metadata={
            KEY: PeriodicInputCertificate(
                input_size=1,
                periodic_inputs=((0, PERIOD),),
                regularity=phx.DerivativeRegularity.smooth(),
            ),
            "note": "retained",
        },
    )
    if variable:
        tau = domain.Function("t")(lambda t: 0.5 * t)
        delayed = phx.operators.delay_operator(source, tau, time_var="t")
        expected_upper = -1.0
    else:
        delayed = phx.operators.delay_operator(
            source, 0.5, time_var="t", clip_time_min=0.0
        )
        expected_upper = 0.0
    lower = jnp.array(0.0, dtype=jnp.float64)
    upper = jnp.array(PERIOD, dtype=jnp.float64)
    np.testing.assert_allclose(delayed.func(lower), 1.0, atol=1e-12)
    np.testing.assert_allclose(delayed.func(upper), expected_upper, atol=1e-12)
    assert delayed.metadata["note"] == "retained"
    condition = phx.conditions.Periodic(
        "u", phx.domain.PeriodicIdentification(domain, "t").pairing()
    )
    with pytest.raises(ValueError, match="carries no PeriodicInputCertificate"):
        phx.enforcement.prepare_periodic_projection(
            {"u": delayed}, (condition,), route="construction"
        )


def test_taylor_partial_does_not_inherit_source_construction_evidence() -> None:
    domain = phx.domain.Interval1d(0.0, PERIOD)
    field = domain.Function("x")(lambda x: jnp.sin(jnp.pi * x[0])).with_metadata(
        periodic_input_certificate=PeriodicInputCertificate(
            input_size=1,
            periodic_inputs=((0, PERIOD),),
            regularity=phx.DerivativeRegularity.smooth(),
        ),
        note="retained",
    )
    derivative = phx.operators.partial_n(field, var="x", axis=0, order=2, backend="jet")
    point = jnp.asarray([0.25], dtype=jnp.float64)
    np.testing.assert_allclose(
        derivative.func(point),
        -(jnp.pi**2) * jnp.sin(jnp.pi * point[0]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert derivative.metadata["note"] == "retained"
    condition = phx.conditions.Periodic(
        "u", phx.domain.PeriodicIdentification(domain, "x").pairing()
    )
    with pytest.raises(ValueError):
        phx.enforcement.prepare_periodic_projection(
            {"u": derivative}, (condition,), route="construction"
        )

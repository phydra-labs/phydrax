import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _ScaledSourceOperator(phx.nn.operator.AbstractOperatorModel):
    operator_architecture = "FNO"

    scale: jnp.ndarray
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, scale=0.0):
        self.scale = jnp.asarray(scale)
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        source = batch.input("forcing")
        assert source.values is not None
        return self.scale * source.values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _problem():
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("x",))
    source = phx.equations.PDEParameter("source", functional=True)
    u = phx.equations.PDEExpression.field("u")
    forcing = phx.equations.PDEExpression.parameter("source")
    return phx.equations.PDEProblemIR(
        (x,),
        (field,),
        parameters=(source,),
        equations=(phx.equations.PDEEquation("poisson", -u.laplacian("x"), forcing),),
    )


def _dataset(count=12):
    nodes = jnp.linspace(0.0, 1.0, count, endpoint=False)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=jnp.full((count,), 1.0 / count),
        basis="fourier",
        periodic=True,
    )
    amplitudes = jnp.asarray((0.75, 1.25))[:, None]
    forcing = amplitudes * jnp.sin(2.0 * jnp.pi * nodes)[None, :]
    samples = phx.nn.operator.FunctionSamples(values=forcing, axes=(axis,))
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    batch = phx.nn.operator.OperatorBatch(
        inputs={"forcing": samples},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch)
    return phx.nn.operator.training.OperatorDataset(batch, targets)


def _compiled(count=12):
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    return phx.equations.compile_spectral_residual(
        _problem(),
        space,
        phx.discretization.PseudospectralMethodPlan(),
    )


def _loss_value(term, model, dataset):
    prediction = model.predict(dataset.batch)
    context = phx.nn.operator.training.OperatorLossContext(
        prediction,
        dataset.batch,
        dataset.targets,
        prediction,
        dataset.batch,
        dataset.targets,
    )
    return term(
        model,
        prediction,
        dataset.batch,
        dataset.targets,
        key=jr.key(0),
        step=jnp.asarray(0),
        training=False,
        context=context,
    )


def test_spectral_pde_loss_is_targetless_differentiable_and_fingerprinted():
    dataset = _dataset()
    compiled = _compiled()
    term = phx.nn.operator.training.SpectralPDEResidualLoss(
        "spectral_poisson",
        compiled,
        {"u": "output"},
        {"source": "forcing"},
    )
    model = _ScaledSourceOperator()
    initial = _loss_value(term, model, dataset)
    gradient = eqx.filter_grad(lambda candidate: _loss_value(term, candidate, dataset))(
        model
    )

    assert initial > 0.0
    assert jnp.isfinite(initial)
    assert jnp.isfinite(gradient.scale)
    assert gradient.scale != 0.0
    assert (
        term.fingerprint
        == phx.nn.operator.training.SpectralPDEResidualLoss(
            "spectral_poisson",
            compiled,
            {"u": "output"},
            {"source": "forcing"},
        ).fingerprint
    )
    assert (
        term.fingerprint
        != phx.nn.operator.training.SpectralPDEResidualLoss(
            "spectral_poisson",
            compiled,
            {"u": "output"},
            {"source": "forcing"},
            weight=2.0,
        ).fingerprint
    )


def test_targetless_operator_fit_reduces_spectral_residual():
    dataset = _dataset()
    term = phx.nn.operator.training.SpectralPDEResidualLoss(
        "spectral_poisson",
        _compiled(),
        {"u": "output"},
        {"source": "forcing"},
    )
    model = _ScaledSourceOperator()
    initial = _loss_value(term, model, dataset)
    result = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        loss_terms=(term,),
        normalization=None,
        learning_rate=1e-3,
        batch_size=2,
        steps=4,
        epochs=4,
        shuffle=False,
        seed=7,
    )
    final = _loss_value(term, result.execution_model, dataset)

    assert final < initial

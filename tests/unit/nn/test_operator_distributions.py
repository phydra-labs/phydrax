import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _batch(*, cases=2, size=6, masked=False):
    nodes = jnp.linspace(0.0, 1.0, size, endpoint=False)
    axis = phx.nn.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=jnp.full((size,), 1.0 / size),
        periodic=True,
    )
    values = jnp.stack(
        tuple(jnp.sin(2.0 * jnp.pi * nodes + case) for case in range(cases))
    )
    mask = None
    if masked:
        mask = jnp.arange(size) < size - 1
    return phx.nn.OperatorBatch(
        inputs={"state": phx.nn.FunctionSamples(values=values, axes=(axis,))},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,), mask=mask)},
        case_axes=("case",),
    )


def test_gaussian_distribution_matches_dense_log_density_and_masks_samples():
    batch = _batch(masked=True)
    query = batch.require_single_query()
    mean = jnp.arange(12, dtype=float).reshape((2, 6)) / 10.0
    scale = jnp.full((2, 6), 0.3)
    factors = jnp.stack(
        (
            jnp.linspace(-0.2, 0.2, 6),
            jnp.linspace(0.1, 0.3, 6),
        ),
        axis=-1,
    )
    factors = jnp.broadcast_to(factors, (2, 6, 2))
    distribution = phx.nn.GaussianOperatorDistribution(
        mean=mean,
        scale=scale,
        factors=factors,
        query=query,
        output_spec=phx.nn.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
        uncertainty_source="process",
    )
    target = mean + 0.05
    active = slice(0, 5)
    covariance = distribution.dense_covariance()
    expected = jax.vmap(
        lambda value, center, matrix: jax.scipy.stats.multivariate_normal.logpdf(
            value[active],
            center[active],
            matrix[active, active],
        )
    )(target, mean, covariance)
    samples = distribution.sample(jr.key(1), (3, 4))

    assert isinstance(distribution, phx.nn.AbstractOperatorDistribution)
    assert distribution.event_shape == (6,)
    assert distribution.rank == 2
    assert distribution.uncertainty_source == "process"
    assert distribution.log_prob(target).shape == (2,)
    assert jnp.allclose(distribution.log_prob(target), expected, rtol=1e-5, atol=1e-6)
    assert samples.shape == (3, 4, 2, 6)
    assert jnp.array_equal(samples[..., -1], jnp.zeros((3, 4, 2)))
    assert jnp.array_equal(distribution.marginal_variance()[:, -1], jnp.zeros((2,)))
    with pytest.raises(ValueError, match="target must have shape"):
        distribution.log_prob(target[:, :-1])


def test_fixed_scale_gaussian_operator_has_coherent_process_distribution_and_gradient():
    batch = _batch()
    base = phx.nn.FNO(
        n_modes=(2,),
        in_channels="scalar",
        out_channels=3,
        width=6,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(3),
    )
    model = phx.nn.GaussianFunctionOperator(
        base,
        out_channels="scalar",
        factor_rank=2,
        scale_mode="fixed",
        fixed_scale=0.07,
        factor_scale=0.4,
        uncertainty_source="process",
    )
    distribution = model.distribution(batch)
    samples = model.sample(batch, num_samples=5, key=jr.key(4))
    target = jnp.zeros((2, 6))
    loss = phx.nn.operator_distribution_nll(model, batch, target)
    gradients = eqx.filter_grad(
        lambda candidate: phx.nn.operator_distribution_nll(candidate, batch, target)
    )(model)
    leaves = tuple(
        leaf for leaf in jax.tree_util.tree_leaves(gradients) if eqx.is_array(leaf)
    )

    assert isinstance(model, phx.nn.AbstractProbabilisticOperatorModel)
    assert jnp.array_equal(distribution.scale, jnp.full((2, 6), 0.07))
    assert distribution.factors.shape == (2, 6, 2)
    assert distribution.uncertainty_source == "process"
    configuration = dict(model.operator_contract.configuration)
    assert configuration["wrapped_architecture"] == "FNO"
    assert configuration["factor_rank"] == 2
    assert configuration["scale_mode"] == "fixed"
    assert configuration["uncertainty_source"] == "process"
    assert model.operator_contract.capabilities == base.operator_contract.capabilities
    assert samples.shape == (5, 2, 6)
    assert jnp.isfinite(loss)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_gaussian_operator_parameter_contract_distinguishes_fixed_and_learned_scale():
    fixed_base = phx.nn.FNO(
        n_modes=(2,),
        in_channels="scalar",
        out_channels=2,
        width=4,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(10),
    )
    phx.nn.GaussianFunctionOperator(
        fixed_base,
        factor_rank=1,
        scale_mode="fixed",
    )
    with pytest.raises(ValueError, match="must emit 3"):
        phx.nn.GaussianFunctionOperator(
            fixed_base,
            factor_rank=1,
            scale_mode="learned",
        )
    with pytest.raises(ValueError, match="scale_mode"):
        phx.nn.GaussianFunctionOperator(
            fixed_base,
            factor_rank=1,
            scale_mode="invalid",
        )
    with pytest.raises(ValueError, match="finite"):
        phx.nn.GaussianFunctionOperator(
            fixed_base,
            factor_rank=1,
            scale_mode="fixed",
            fixed_scale=float("nan"),
        )

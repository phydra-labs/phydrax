#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.models.architectures._hofno import (
    _dealiased_spectral_resample,
    _ProjectedProductFourierMixer,
)


def _identity_quadratic_mixer(aliasing):
    mixer = _ProjectedProductFourierMixer(
        channels=1,
        n_modes=(4,),
        interaction_order=2,
        factor_bias=False,
        spectral_channel_mixing="depthwise",
        aliasing=aliasing,
        key=jr.key(0),
    )
    return eqx.tree_at(
        lambda model: (model.projection.weight, model.spectral.weight),
        mixer,
        (
            jnp.ones_like(mixer.projection.weight),
            jnp.ones_like(mixer.spectral.weight),
        ),
    )


def _grid_batch(*, periodic=True, mask=None):
    nodes = jnp.arange(8, dtype=float) / 8.0
    axis = phx.nn.OperatorAxis("x", nodes, basis="fourier", periodic=periodic)
    source = phx.nn.FunctionSamples(
        values=jnp.stack(
            (
                jnp.sin(2.0 * jnp.pi * nodes),
                jnp.cos(2.0 * jnp.pi * nodes),
            )
        ),
        axes=(axis,),
        mask=mask,
    )
    return phx.nn.OperatorBatch(
        inputs={"source": source},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
    )


def test_dealiased_projected_product_removes_folded_retained_mode():
    nodes = jnp.arange(16, dtype=float) / 16.0
    values = jnp.cos(2.0 * jnp.pi * 7.0 * nodes)[:, None]
    collocation = _identity_quadratic_mixer("collocation")(values)[:, 0]
    dealiased = _identity_quadratic_mixer("dealiased")(values)[:, 0]
    collocation_spectrum = jnp.fft.rfft(collocation, norm="ortho")
    dealiased_spectrum = jnp.fft.rfft(dealiased, norm="ortho")

    assert jnp.abs(collocation_spectrum[2]) > 0.5
    assert jnp.abs(dealiased_spectrum[2]) < 1e-10
    assert jnp.max(jnp.abs(dealiased - jnp.mean(dealiased))) < 1e-10


def test_dealiased_resampling_preserves_even_grid_nyquist_mode():
    values = ((-1.0) ** jnp.arange(16, dtype=float))[:, None]
    oversampled = _dealiased_spectral_resample(values, (21,))
    restored = _dealiased_spectral_resample(oversampled, (16,))

    assert restored.shape == values.shape
    assert jnp.allclose(restored, values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_hofno_has_finite_nd_output_and_parameter_gradients(ndim):
    size = 6
    nodes = jnp.arange(size, dtype=float) / size
    values = jr.normal(jr.key(10 + ndim), (size,) * ndim)
    model = phx.nn.HOFNO(
        n_modes=(2,) * ndim,
        width=3,
        depth=1,
        ffn_expansion=2,
        key=jr.key(ndim),
    )

    output = model((values,) + (nodes,) * ndim)
    gradient = eqx.filter_grad(
        lambda candidate: jnp.sum(candidate((values,) + (nodes,) * ndim) ** 2)
    )(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradient, eqx.is_inexact_array))

    assert output.shape == values.shape
    assert jnp.all(jnp.isfinite(output))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_hofno_scan_matches_loop_under_jit():
    nodes = jnp.arange(8, dtype=float) / 8.0
    x, y = jnp.meshgrid(nodes, nodes, indexing="ij")
    values = jnp.sin(2.0 * jnp.pi * x) + jnp.cos(2.0 * jnp.pi * y)
    options = dict(
        n_modes=(3, 3),
        width=4,
        depth=2,
        ffn_expansion=2,
        key=jr.key(9),
    )
    loop = phx.nn.HOFNO(**options, scan=False)
    scanned = phx.nn.HOFNO(**options, scan=True)
    execute = eqx.filter_jit(lambda model, field, axis: model((field, axis, axis)))

    assert jnp.allclose(
        execute(loop, values, nodes),
        execute(scanned, values, nodes),
        rtol=1e-10,
        atol=1e-10,
    )


def test_hofno_runtime_and_registry_enforce_periodic_all_valid_contract():
    model = phx.nn.HOFNO(
        n_modes=(3,),
        width=4,
        depth=1,
        ffn_expansion=2,
        interaction_order=3,
        source_key="source",
        key=jr.key(4),
    )
    output = model(_grid_batch())
    status = phx.nn.operator_architecture_status("higher order Fourier neural operator")
    configuration = dict(model.operator_contract.configuration)

    assert output.shape == (2, 8)
    assert status.tier == "experimental"
    assert not status.recommendation_eligible
    assert status.capabilities.axis_requirement == "periodic_uniform"
    assert status.capabilities.masks == "all_valid_only"
    assert model.operator_contract.architecture == "HOFNO"
    assert configuration["interaction_order"] == 3
    assert configuration["aliasing"] == "dealiased"

    nonperiodic = _grid_batch(periodic=False)
    report = phx.nn.validate_operator_architecture("HOFNO", nonperiodic)
    assert "NONPERIODIC_AXIS" in report.codes
    with pytest.raises(ValueError, match="requires periodic"):
        model(nonperiodic)

    mask = jnp.ones((2, 8), dtype=bool).at[0, 0].set(False)
    with pytest.raises(eqx.EquinoxRuntimeError, match="all-valid source masks"):
        model(_grid_batch(mask=mask))
    with pytest.raises(ValueError, match="requires domain_padding=0"):
        phx.nn.HOFNO(n_modes=(3,), domain_padding=0.1, key=jr.key(5))

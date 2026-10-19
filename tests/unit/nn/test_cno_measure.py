import hashlib
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.operator.architectures import ConvolutionAxisPolicy
from phydrax.nn.operator.architectures.spectral._cno import _coordinate_features
from tests._support.periodic_operator_cases import trained_operator


def _batch(
    values: Any, axis: Any, *, source_mask: Any, query_mask: Any, quadrature: Any
) -> Any:
    source = phx.nn.operator.FunctionSamples(
        values=values,
        axes=(axis,),
        mask=source_mask,
        quadrature_weights=quadrature,
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(axis,),
        mask=query_mask,
        quadrature_weights=quadrature,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"source": source}, queries={"query": query}
    )


def test_cno_masks_invalid_sources_before_lifting_and_zeros_invalid_queries() -> None:
    nodes = jnp.arange(9, dtype="float64") / 9
    quadrature = jnp.linspace(0.5, 1.5, 9)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=quadrature,
        basis="fourier",
        periodic=True,
    )
    source_mask = jnp.asarray([True, True, False, True, True, False, True, True, True])
    query_mask = jnp.asarray([True, True, True, True, True, True, True, True, False])
    values = jnp.sin(2.0 * jnp.pi * nodes)
    invalid_nan = jnp.where(source_mask, values, jnp.nan)
    invalid_large = jnp.where(source_mask, values, 1e6)
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=4,
        depth=2,
        oversample_factor=1,
        key=jr.key(2),
    )

    nan_output = eqx.filter_jit(model)(
        _batch(
            invalid_nan,
            axis,
            source_mask=source_mask,
            query_mask=query_mask,
            quadrature=quadrature,
        )
    )
    large_output = model(
        _batch(
            invalid_large,
            axis,
            source_mask=source_mask,
            query_mask=query_mask,
            quadrature=quadrature,
        )
    )

    assert jnp.all(jnp.isfinite(nan_output))
    assert jnp.allclose(nan_output, large_output, atol=1e-6, rtol=1e-6)
    assert nan_output[-1] == 0.0


def test_cno_uniform_measure_full_support_matches_array_evaluation() -> None:
    nodes = -1.0 + 2.0 * jnp.arange(11, dtype="float64") / 11
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=jnp.full(11, 0.2),
        basis="fourier",
        periodic=True,
    )
    values = jnp.cos(3.0 * nodes)
    mask = jnp.ones(11, dtype="bool")
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(5),
    )

    batch_output = model(
        _batch(
            values,
            axis,
            source_mask=mask,
            query_mask=mask,
            quadrature=jnp.full(11, 0.2),
        )
    )
    array_output = model((values, nodes))

    assert jnp.allclose(batch_output, array_output, atol=2e-6, rtol=2e-6)


def test_periodic_coordinate_features_have_equal_interior_and_seam_chords() -> None:
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(13, dtype="float64") / 13,
        basis="fourier",
        periodic=True,
    )
    features = _coordinate_features((axis,), ())
    closed = jnp.concatenate((features, features[:1]), axis=0)
    chords = jnp.linalg.norm(jnp.diff(closed, axis=0), axis=-1)

    assert features.shape == (13, 2)
    assert jnp.allclose(chords, jnp.full_like(chords, chords[0]))
    assert jnp.allclose(features[0], jnp.asarray([0.0, 1.0]))


def test_cno_rejects_axes_outside_periodic_uniform_fourier_contract() -> None:
    for axis in (
        phx.nn.operator.OperatorAxis("x", jnp.arange(5.0) / 5),
        phx.nn.operator.OperatorAxis(
            "x", jnp.arange(5.0) / 5, basis="legendre", periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.0, 0.25, jnp.nan, 0.75]), periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.0, 0.25, 0.25, 0.75]), periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.75, 0.5, 0.25, 0.0]), periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.0, 0.4, 0.7, 0.9]), periodic=True
        ),
        phx.nn.operator.OperatorAxis("x", jnp.asarray([0.0]), periodic=True),
    ):
        model = phx.nn.operator.architectures.CNO(
            spatial_ndim=1,
            width=3,
            depth=1,
            oversample_factor=1,
            key=jr.key(20),
        )
        values = jnp.ones((axis.size,))
        axis = phx.nn.operator.OperatorAxis(
            axis.name,
            axis.nodes,
            quadrature_weights=jnp.ones((axis.size,)),
            basis=axis.basis,
            periodic=axis.periodic,
        )
        batch = phx.nn.operator.OperatorBatch(
            inputs={
                "source": phx.nn.operator.FunctionSamples(values=values, axes=(axis,))
            },
            queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        )

        with pytest.raises(ValueError):
            model(batch)


@pytest.mark.parametrize("architecture", ("CNO", "UNO"), ids=("cno", "uno"))
def test_configured_support_plan_owns_axis_admission(architecture: str) -> None:
    def model(plan: Any) -> Any:
        match architecture:
            case "CNO":
                return phx.nn.operator.architectures.CNO(
                    spatial_ndim=1, width=2, depth=1, support_plan=plan, key=jr.key(3)
                )
            case _:
                return phx.nn.operator.architectures.UNO(
                    spatial_ndim=1, widths=(2, 2), support_plan=plan, key=jr.key(3)
                )

    def batch(nodes: Any) -> Any:
        axis = phx.nn.operator.OperatorAxis(
            "x", nodes, quadrature_weights=jnp.full(nodes.shape, 1.0 / nodes.size)
        )
        return _batch(
            jnp.ones(nodes.shape),
            axis,
            source_mask=None,
            query_mask=None,
            quadrature=axis.quadrature_weights,
        )

    uniform = batch((jnp.arange(8, dtype=jnp.float64) + 0.5) / 8)
    graded = batch(jnp.asarray([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95]))
    bounded = model(
        phx.nn.operator.architectures.ConvolutionSupportPlan(("dirichlet_sine",))
    )
    periodic = model(None)

    bounded.operator_contract.validate(uniform).require_runtime()
    with pytest.raises(ValueError, match="NONUNIFORM_AXIS"):
        bounded.operator_contract.validate(graded).require_runtime()
    with pytest.raises(ValueError, match="NONPERIODIC_AXIS"):
        periodic.operator_contract.validate(uniform).require_runtime()


@pytest.mark.parametrize(
    "policy", ("dirichlet_sine", "neumann_cosine"), ids=("sine", "cosine")
)
def test_bounded_extension_keeps_physical_measure_nonnegative(
    policy: ConvolutionAxisPolicy,
) -> None:
    layer = phx.nn.operator.architectures.AntiAliasedConvND(
        spatial_ndim=1,
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        activation=None,
        circular=False,
        extension_modes=(policy,),
        key=jr.key(4),
    )
    values = jnp.sin(jnp.pi * (jnp.arange(8, dtype=jnp.float64) + 0.5) / 8)[:, None]

    weighted = layer(values, quadrature=jnp.full((8,), 0.125, dtype=jnp.float64))
    unweighted = layer(values)

    # A uniform physical measure cancels in the measure-normalized convolution.
    assert jnp.allclose(weighted, unweighted, rtol=1e-13, atol=1e-13)


def test_cno_rejects_noncoincident_source_and_query_axes() -> None:
    source_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(7, dtype="float64") / 7,
        basis="fourier",
        periodic=True,
        quadrature_weights=jnp.ones((7,)),
    )
    query_axis = phx.nn.operator.OperatorAxis(
        "x",
        0.1 + jnp.arange(7, dtype="float64") / 7,
        basis="fourier",
        periodic=True,
        quadrature_weights=jnp.ones((7,)),
    )
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(21),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones(7),
                axes=(source_axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(query_axis,),
            )
        },
    )

    with pytest.raises(ValueError, match="exactly coincident"):
        model(batch)


def test_cno_uses_circular_measure_layers_and_periodic_feature_width() -> None:
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=2,
        in_channels=2,
        width=4,
        depth=2,
        oversample_factor=1,
        key=jr.key(22),
    )

    assert model.lift.in_size == 6
    assert all(block.first.circular and block.second.circular for block in model.blocks)


def test_cno_oversampled_activation_uses_explicit_periodic_spatial_axes() -> None:
    layer = phx.nn.operator.architectures.AntiAliasedConvND(
        spatial_ndim=1,
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        oversample_factor=2,
        circular=True,
        key=jr.key(25),
    )
    values = jnp.ones((8, 1))

    output = layer(values, quadrature=jnp.full((8,), 1.0 / 8.0))

    assert output.shape == (8, 2)
    assert jnp.all(jnp.isfinite(output))


def test_cno_rejects_missing_physical_source_quadrature() -> None:
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(7, dtype="float64") / 7,
        basis="fourier",
        periodic=True,
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((7,)),
                axes=(axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
            )
        },
    )
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(23),
    )

    with pytest.raises(ValueError, match="physical source quadrature"):
        model(batch)


def test_uno_transports_masked_measure_and_zeros_invalid_queries() -> None:
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(7, dtype="float64") / 7,
        quadrature_weights=jnp.ones((7,)),
        basis="fourier",
        periodic=True,
    )
    source_mask = jnp.asarray([True, True, False, True, True, True, True])
    query_mask = jnp.asarray([True, True, True, True, True, False, True])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((7,)),
                axes=(axis,),
                mask=source_mask,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
                mask=query_mask,
            )
        },
    )
    model = phx.nn.operator.architectures.UNO(
        spatial_ndim=1,
        widths=(3, 4),
        oversample_factor=1,
        key=jr.key(24),
    )

    output = model(batch)

    assert output.shape == (7,)
    assert jnp.all(jnp.isfinite(output))
    assert output[2] == 0.0
    assert output[5] == 0.0
    assert jnp.any(output[source_mask & query_mask] != 0.0)


@pytest.mark.strict_jax
@pytest.mark.parametrize("architecture", ("CNO", "UNO"), ids=("cno", "uno"))
def test_bias_free_linear_operator_superposition_and_artifact(
    architecture: str, tmp_path: Path
) -> None:
    nodes = jnp.arange(12, dtype=jnp.float64) / 12
    quadrature = jnp.linspace(0.5, 1.5, 12, dtype=jnp.float64)
    axis = phx.nn.operator.OperatorAxis(
        "x", nodes, quadrature_weights=quadrature, basis="fourier", periodic=True
    )
    source_mask = jnp.asarray(
        [True, True, False, True, True, True, True, False, True, True, True, True]
    )
    query_mask = jnp.asarray(
        [True, True, True, True, True, True, True, True, True, True, True, False]
    )
    model = (
        phx.nn.operator.architectures.CNO(
            spatial_ndim=1,
            width=3,
            depth=2,
            activation=None,
            use_bias=False,
            coordinate_embedding=False,
            key=jr.key(71),
        )
        if architecture == "CNO"
        else phx.nn.operator.architectures.UNO(
            spatial_ndim=1,
            widths=(3, 5, 7),
            activation=None,
            use_bias=False,
            coordinate_embedding=False,
            key=jr.key(71),
        )
    )

    def evaluate(values: Any) -> Any:
        return model(
            _batch(
                values,
                axis,
                source_mask=source_mask,
                query_mask=query_mask,
                quadrature=quadrature,
            )
        )

    first = jnp.sin(2 * jnp.pi * nodes)
    second = jnp.cos(4 * jnp.pi * nodes)
    assert jnp.array_equal(evaluate(jnp.zeros_like(first)), jnp.zeros_like(first))
    assert jnp.allclose(
        evaluate(first + second),
        evaluate(first) + evaluate(second),
        atol=2e-12,
        rtol=2e-12,
    )
    assert jnp.allclose(
        evaluate(-2.5 * first), -2.5 * evaluate(first), atol=2e-12, rtol=2e-12
    )
    trained = trained_operator(
        model=model,
        dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
            parameter_dtype="float64", compute_dtype="float64", reduction_dtype="float64"
        ),
    )
    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.operator.training.load_trained_operator(destination)
    assert jnp.allclose(
        restored.execution_model(
            _batch(
                first,
                axis,
                source_mask=source_mask,
                query_mask=query_mask,
                quadrature=quadrature,
            )
        ),
        evaluate(first),
        atol=2e-12,
        rtol=2e-12,
    )


# DAG recipes now use positional dataclass items and shared references.
@pytest.mark.parametrize(
    ("architecture", "recipe_id", "artifact_id"),
    (
        (
            "CNO",
            "b7f29c9b6a93d2c453e6c8f54fbf8174afa0920dd3b7de600bd0b65f3a490f78",
            "ebc38d2f17ff06c5b2e22f2f9d3436b2f3a2ba62765aa1dddea1746ede742804",
        ),
        (
            "UNO",
            "266738ba9c6987fd0d464f26de65d0a319c9ba8c99782e1c0f5bfb6d3a8ebccd",
            "33da864ecbcaa3b703557a6c73e4cfdb56f844a5963964e6751b5f6b1cc42897",
        ),
    ),
    ids=("cno", "uno"),
)
def test_default_cno_recipe_and_artifact_preserve_merge_base_identity(
    architecture: str, recipe_id: str, artifact_id: str, tmp_path: Path
) -> None:
    model = (
        phx.nn.operator.architectures.CNO(spatial_ndim=1, width=4, depth=1)
        if architecture == "CNO"
        else phx.nn.operator.architectures.UNO(spatial_ndim=1, widths=(4, 6))
    )
    destination = phx.nn.operator.training.save_operator_artifact(
        tmp_path, trained_operator(model=model)
    )
    manifest = phx.nn.operator.training.load_operator_artifact_manifest(destination)
    recipe_bytes = json.dumps(
        manifest.execution_model_recipe, sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(recipe_bytes).hexdigest() == recipe_id
    assert manifest.artifact_id == artifact_id

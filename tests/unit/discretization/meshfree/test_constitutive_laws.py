# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from collections.abc import Iterator
from typing import Any, cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.discretization.meshfree._constitutive import (
    EdgeFeatureField,
    EdgeFrameFeatures,
    LipschitzEdgeFlux,
    MonotoneEdgeConductance,
)
from phydrax.discretization.meshfree._coverage import (
    EdgeCoverageStatus,
    EdgeFeatureCoverage,
)
from phydrax.nn.layers import Linear
from phydrax.nn.models import InputConvexNetwork, PartiallyInputConvexNetwork
from phydrax.nn.parameters import IdentityTransform, LowRankUpdate
from phydrax.units import LENGTH


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def test_scalar_vector_tensor_edge_reversal_and_cartesian_frame_identities() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-1.0, 1.0, 2.0]])
    pairs = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    scalar = np.asarray([1.0, 3.0, -2.0])
    vector = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [3.0, 1.0, -2.0]])
    tensor = np.asarray(
        [
            np.diag([1.0, 2.0, 3.0]),
            [[2.0, 0.4, 0.1], [0.4, 3.0, 0.2], [0.1, 0.2, 1.0]],
            np.diag([3.0, 1.0, 2.0]),
        ]
    )
    schema = (
        EdgeFeatureField("s", "scalar", dimension=LENGTH),
        EdgeFeatureField("v", "vector"),
        EdgeFeatureField("T", "symmetric-tensor"),
    )
    features = EdgeFrameFeatures(points, pairs, schema, (scalar, vector, tensor))
    reversed_features = features.reoriented()
    independently_reversed = EdgeFrameFeatures(
        points, pairs[:, ::-1], schema, (scalar, vector, tensor)
    )
    np.testing.assert_allclose(reversed_features.even, features.even, atol=1e-14)
    np.testing.assert_allclose(reversed_features.odd, -features.odd, atol=1e-14)
    np.testing.assert_allclose(
        reversed_features.even, independently_reversed.even, atol=1e-14
    )
    np.testing.assert_allclose(
        reversed_features.odd, independently_reversed.odd, atol=1e-14
    )
    # Include a reflection: scalar invariance is orthogonal, not just rotational.
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    transformed = features.transformed_frame(rotation)
    np.testing.assert_allclose(transformed.even, features.even, atol=1e-14)
    np.testing.assert_allclose(transformed.odd, features.odd, atol=1e-14)
    np.testing.assert_allclose(
        transformed.averages[1], features.averages[1] @ rotation.T, atol=1e-14
    )
    assert features.even_dimensions[0] == LENGTH
    with pytest.raises(ValueError, match="orthogonal"):
        features.transformed_frame(np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="symmetric"):
        EdgeFrameFeatures(points, pairs, (schema[2],), (np.triu(tensor),))


@pytest.mark.parametrize("convex_size", ["scalar", 1, (1,)])
def test_certified_convex_slope_is_odd_strongly_monotone_and_energy_consistent(
    convex_size: int | tuple[int, ...] | Literal["scalar"],
) -> None:
    potential = InputConvexNetwork(
        in_size=convex_size, width_size=3, depth=2, key=jr.key(3)
    )
    law = MonotoneEdgeConductance(potential, background_conductance=0.7)
    even = jnp.zeros((0,))
    odd = jnp.zeros((0,))
    grid = jnp.linspace(-3.0, 3.0, 31)
    values = jax.vmap(lambda value: law.edge_flux(value, even, odd))(grid)
    reverse = jax.vmap(lambda value: law.edge_flux(-value, even, odd))(grid)
    derivative = jax.vmap(jax.grad(lambda value: law.edge_flux(value, even, odd)))(grid)
    energy_derivative = jax.vmap(jax.grad(lambda value: law.energy(value, even)))(grid)
    np.testing.assert_allclose(values, -reverse, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(values, energy_derivative, rtol=1e-12, atol=1e-12)
    assert bool(jnp.all(derivative >= 0.7 - 1e-12))
    assert bool(jnp.all(grid * values >= 0.7 * grid**2 - 1e-12))
    # An asymmetric potential is an explicit regression for the incorrect
    # phi'(D)-phi'(0) construction: that construction need not be odd.
    slope = jax.grad(lambda value: law.conditioned_potential(value, even))
    centered = jax.vmap(lambda value: slope(value) - slope(jnp.asarray(0.0)))(grid)
    assert float(jnp.max(jnp.abs(centered + centered[::-1]))) > 1e-5


def test_conditioned_potential_uses_only_external_even_context() -> None:
    potential = PartiallyInputConvexNetwork(
        context_size=2, convex_size="scalar", width_size=3, depth=1, key=jr.key(5)
    )
    law = MonotoneEdgeConductance(potential, odd_size=1)
    even = jnp.asarray([0.3, -0.5])
    first = law.edge_flux(jnp.asarray(0.4), even, jnp.asarray([2.0]))
    second = law.edge_flux(jnp.asarray(0.4), even, jnp.asarray([-7.0]))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(
        first, -law.edge_flux(jnp.asarray(-0.4), even, jnp.asarray([-2.0]))
    )
    assert (
        float(
            jax.grad(lambda difference: law.edge_flux(difference, even, jnp.zeros((1,))))(
                0.4
            )
        )
        >= 1
    )


def test_monotone_law_refuses_uncertified_and_nonsmooth_potentials() -> None:
    with pytest.raises(TypeError, match="input-convex"):
        MonotoneEdgeConductance(
            cast(Any, Linear(in_size=1, out_size="scalar", rwf=False))
        )
    with pytest.raises(ValueError, match="softplus"):
        MonotoneEdgeConductance(InputConvexNetwork(in_size=1, activation="relu"))
    potential = InputConvexNetwork(in_size=1, width_size=2, depth=1)
    forged = eqx.tree_at(
        lambda model: model.state_layers[0].weight_transform,
        potential,
        IdentityTransform(),
    )
    with pytest.raises(ValueError, match="positive"):
        MonotoneEdgeConductance(forged)


def test_certified_lipschitz_model_reorients_state_and_odd_context_together() -> None:
    model = Linear(
        in_size=3, out_size="scalar", activation=jax.nn.tanh, rwf=False, key=jr.key(7)
    )
    model = eqx.tree_at(
        lambda layer: (layer.weight, layer.bias),
        model,
        (jnp.asarray([[0.2, -0.3, 0.4]]), jnp.asarray([0.7])),
    )
    law = LipschitzEdgeFlux(model, even_size=1, odd_size=1)
    even, odd = jnp.asarray([0.5]), jnp.asarray([0.8])
    difference = jnp.asarray(0.4)
    np.testing.assert_allclose(
        law.edge_flux(difference, even, odd),
        -law.edge_flux(-difference, even, -odd),
        atol=1e-14,
    )
    derivatives = jax.vmap(jax.grad(lambda value: law.perturbation(value, even, odd)))(
        jnp.linspace(-4.0, 4.0, 41)
    )
    assert bool(jnp.all(jnp.abs(derivatives) <= law.certified_lipschitz_bound() + 1e-14))
    np.testing.assert_allclose(
        law.certified_lipschitz_bound(), np.sqrt(0.2**2 + 0.3**2 + 0.4**2)
    )
    changed = eqx.tree_at(
        lambda constitutive: constitutive.model.weight,
        law,
        jnp.asarray([[0.6, -0.3, 0.4]]),
    )
    np.testing.assert_allclose(
        changed.certificate.bound, np.sqrt(0.6**2 + 0.3**2 + 0.4**2)
    )
    with pytest.raises(ValueError, match="global derivative bound"):
        LipschitzEdgeFlux(
            Linear(in_size=1, out_size="scalar", activation=jax.nn.relu, rwf=False)
        )
    low_rank = eqx.tree_at(
        lambda layer: layer.weight,
        model,
        LowRankUpdate(jnp.asarray([[0.2, -0.3, 0.4]]), rank=1),
    )
    with pytest.raises(TypeError):
        LipschitzEdgeFlux(low_rank, even_size=1, odd_size=1)


def test_quantile_box_and_joint_covariance_have_independent_refusal_evidence() -> None:
    samples = np.asarray([[-1.0, -1.0], [-0.5, -0.52], [0.5, 0.52], [1.0, 1.0]])
    coverage = EdgeFeatureCoverage.fit(
        samples, training_domain="correlated-training-only", feature_names=("x", "y")
    )
    assessed = eqx.filter_jit(coverage.assess)(
        jnp.asarray([[0.8, -0.8], [1.5, 0.0], [0.0, 0.0], [jnp.nan, 0.0]])
    )
    assert bool(assessed.marginal_covered[0]) and not bool(assessed.joint_covered[0])
    assert int(assessed.status[0]) == int(EdgeCoverageStatus.JOINT_OUTSIDE)
    assert int(assessed.status[1]) == int(EdgeCoverageStatus.MARGINAL_OUTSIDE)
    assert int(assessed.status[2]) == int(EdgeCoverageStatus.COVERED)
    assert int(assessed.status[3]) == int(EdgeCoverageStatus.NONFINITE)
    assert not bool(assessed.admitted)
    report = EdgeFeatureCoverage.fit(
        samples, training_domain="correlated-training-only", policy="report"
    )
    outside = report.assess(jnp.asarray([[0.8, -0.8]]))
    assert bool(outside.admitted) and not bool(outside.covered[0])
    with pytest.raises(ValueError, match="not positive definite"):
        EdgeFeatureCoverage.fit(
            np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            training_domain="collinear",
        )
    trimmed = EdgeFeatureCoverage.fit(
        np.linspace(-2.0, 2.0, 21)[:, None],
        training_domain="quantile-training",
        lower_quantile=0.1,
        upper_quantile=0.9,
    )
    assert int(trimmed.assess(jnp.asarray([[1.9]])).status[0]) == int(
        EdgeCoverageStatus.MARGINAL_OUTSIDE
    )

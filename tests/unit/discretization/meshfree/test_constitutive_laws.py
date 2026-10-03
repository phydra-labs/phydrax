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
    AbstractCoupledEdgeConstitutiveLaw,
    EdgeFeatureField,
    EdgeFrameFeatures,
    LipschitzCoupledEdgeFlux,
    LipschitzEdgeFlux,
    MonotoneCoupledEdgeFlux,
    MonotoneEdgeConductance,
    O3EdgeInvariants,
    O3EdgeNetwork,
)
from phydrax.discretization.meshfree._coverage import (
    EdgeCoverageStatus,
    EdgeFeatureCoverage,
)
from phydrax.nn.layers import Linear
from phydrax.nn.models import InputConvexNetwork, PartiallyInputConvexNetwork
from phydrax.nn.operator.representations import O3Representation
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


# A state with every polar and pseudo irrep type, including two vector channels,
# so that coupling across multiplicities and irrep types is observable.
_STATE = O3Representation(
    scalars=1, pseudoscalars=1, vectors=2, pseudovectors=1, tensors=1
)
_PROPER = np.asarray(
    [[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]], dtype=np.float64
)
_IMPROPER = -_PROPER


def _cloud(dimension: int = 3) -> EdgeFrameFeatures:
    rng = np.random.default_rng(17)
    points = rng.normal(size=(7, dimension))
    pairs = np.asarray(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0], [1, 4]],
        dtype=np.int32,
    )
    schema = (EdgeFeatureField("s", "scalar"), EdgeFeatureField("v", "vector"))
    values = (rng.normal(size=7), rng.normal(size=(7, dimension)))
    return EdgeFrameFeatures(points, pairs, schema, values)


def _states(seed: int, scale: float = 1.0) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(scale * rng.normal(size=(8, _STATE.packed_size)))


def _monotone_law(background: float = 0.6) -> MonotoneCoupledEdgeFlux:
    features = _cloud()
    invariants = O3EdgeInvariants(
        _STATE,
        quadratic_representation=O3Representation(
            scalars=2, pseudoscalars=1, vectors=2, pseudovectors=1, tensors=1
        ),
        linear_count=3,
        key=jr.key(0),
    )
    potential = PartiallyInputConvexNetwork(
        context_size=features.even.shape[1] + features.odd.shape[1],
        convex_size=invariants.size,
        width_size=6,
        depth=2,
        input_monotonicity="nondecreasing",
        key=jr.key(1),
    )
    return MonotoneCoupledEdgeFlux(
        invariants,
        potential,
        background_conductance=background,
        odd_size=features.odd.shape[1],
    )


def _lipschitz_law(background: float = 2.0) -> LipschitzCoupledEdgeFlux:
    features = _cloud()
    even, odd = features.even.shape[1], features.odd.shape[1]
    network = O3EdgeNetwork(
        _STATE,
        O3Representation(
            scalars=3, pseudoscalars=1, vectors=2, pseudovectors=2, tensors=1
        ),
        context_size=even + odd,
        key=jr.key(2),
    )
    return LipschitzCoupledEdgeFlux(
        network, even_size=even, odd_size=odd, background_conductance=background
    )


@pytest.mark.parametrize("family", ["monotone", "lipschitz"])
@pytest.mark.parametrize(
    "orthogonal", [_PROPER, _IMPROPER], ids=["rotation", "rotoreflection"]
)
def test_coupled_flux_is_exactly_reversal_odd_and_o3_covariant(
    family: str, orthogonal: np.ndarray
) -> None:
    law: AbstractCoupledEdgeConstitutiveLaw = (
        _monotone_law() if family == "monotone" else _lipschitz_law()
    )
    features = _cloud()
    states = _states(3)
    flux = law.flux(states, features)
    # Reversal parity holds bit-for-bit, against the feature owner's reorientation
    # and against an independently constructed reversed edge list.
    np.testing.assert_array_equal(law.flux(-states, features.reoriented()), -flux)
    rng = np.random.default_rng(17)
    points = rng.normal(size=(7, 3))
    values = (rng.normal(size=7), rng.normal(size=(7, 3)))
    swapped = EdgeFrameFeatures(
        points, np.asarray(features.pairs)[:, ::-1], features.schema, values
    )
    np.testing.assert_allclose(law.flux(-states, swapped), -flux, rtol=1e-13, atol=1e-13)
    q = jnp.asarray(orthogonal)
    assert abs(abs(float(np.linalg.det(orthogonal))) - 1) < 1e-12
    rotated = law.flux(
        _STATE.transform(states, q), features.transformed_frame(orthogonal)
    )
    np.testing.assert_allclose(rotated, _STATE.transform(flux, q), rtol=1e-11, atol=1e-11)


def test_monotone_coupled_flux_is_strongly_monotone_energy_gradient_and_coupled() -> None:
    law = _monotone_law(background=0.6)
    features = _cloud()
    tangents, even, odd = features.tangents, features.even, features.odd
    assert float(law.monotonicity_lower_bound()) == 0.6
    assert float(law.lipschitz_bound()) == float("inf")
    assert law.certificate.potential.input_monotonicity == "nondecreasing"
    np.testing.assert_allclose(
        law.flux(jnp.zeros((8, _STATE.packed_size)), features), 0.0, atol=1e-13
    )
    for seed in range(4):
        first = _states(10 + seed, scale=3.0)
        second = _states(20 + seed, scale=0.5)
        change = first - second
        gap = jnp.sum(
            (law.flux(first, features) - law.flux(second, features)) * change, axis=1
        )
        assert bool(jnp.all(gap >= 0.6 * jnp.sum(change * change, axis=1) - 1e-10))
    states = _states(4, scale=2.0)
    blocks = law.derivative(states, features)
    np.testing.assert_allclose(blocks, blocks.transpose(0, 2, 1), atol=1e-11)
    assert float(jnp.min(jnp.linalg.eigvalsh(blocks))) >= 0.6 - 1e-10
    # Genuine coupling across multiplicities: the first vector channel's flux
    # responds to the second vector channel on every edge (a reversal-even
    # v1.v2 interaction survives the odd symmetrization).
    assert float(jnp.min(jnp.max(jnp.abs(blocks[:, 2:5, 5:8]), axis=(1, 2)))) > 1e-3
    energy_gradient = jax.vmap(jax.grad(law.energy))(states, tangents, even, odd)
    np.testing.assert_allclose(
        energy_gradient, law.flux(states, features), rtol=1e-11, atol=1e-11
    )
    energy = jax.vmap(law.energy)(states, tangents, even, odd)
    assert bool(jnp.all(energy >= 0.3 * jnp.sum(states * states, axis=1) - 1e-10))
    linearized = law.linearize(states, features)
    direction = _states(5)
    np.testing.assert_allclose(
        linearized.pushforward(direction),
        jnp.einsum("eij,ej->ei", blocks, direction),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        linearized.pullback(direction),
        jnp.einsum("eji,ej->ei", blocks, direction),
        rtol=1e-11,
        atol=1e-11,
    )


def test_coupled_lipschitz_bound_covers_the_whole_output_and_tracks_weights() -> None:
    law = _lipschitz_law(background=2.0)
    features = _cloud()
    tangents, even, odd = features.tangents, features.even, features.odd
    bound = float(law.certified_lipschitz_bound())
    assert 0 < bound < float("inf")
    np.testing.assert_allclose(law.lipschitz_bound(), 2.0 + bound)
    np.testing.assert_allclose(law.monotonicity_lower_bound(), 2.0 - bound)
    perturbation = jax.vmap(law.perturbation)
    worst = 0.0
    for seed in range(12):
        first = _states(100 + seed, scale=4.0 / (seed + 1))
        second = first + _states(200 + seed, scale=10.0 ** (-seed % 4))
        change = jnp.linalg.norm(
            perturbation(first, tangents, even, odd)
            - perturbation(second, tangents, even, odd),
            axis=1,
        )
        worst = max(
            worst, float(jnp.max(change / jnp.linalg.norm(first - second, axis=1)))
        )
    assert worst <= bound
    jacobians = law.derivative(_states(7, scale=0.3), features) - 2.0 * jnp.eye(
        _STATE.packed_size
    )
    assert float(jnp.max(jnp.linalg.norm(jacobians, ord=2, axis=(1, 2)))) <= bound
    assert law.network.output_layer.weight is not None
    scaled = eqx.tree_at(
        lambda value: value.network.output_layer.weight,
        law,
        3.0 * law.network.output_layer.weight,
    )
    np.testing.assert_allclose(
        scaled.certified_lipschitz_bound(), 3.0 * bound, rtol=1e-12
    )


def test_coupled_laws_refuse_unsupported_and_conflicting_models() -> None:
    features = _cloud()
    context = features.even.shape[1] + features.odd.shape[1]
    invariants = _monotone_law().invariants

    def potential(**options: Any) -> PartiallyInputConvexNetwork:
        settings: dict[str, Any] = {
            "context_size": context,
            "convex_size": invariants.size,
            "width_size": 3,
            "depth": 1,
            "input_monotonicity": "nondecreasing",
            "key": jr.key(4),
        }
        settings.update(options)
        return PartiallyInputConvexNetwork(**settings)

    with pytest.raises(ValueError, match="nondecreasing in every invariant"):
        MonotoneCoupledEdgeFlux(invariants, potential(input_monotonicity="unconstrained"))
    with pytest.raises(ValueError, match="softplus"):
        MonotoneCoupledEdgeFlux(invariants, potential(activation="relu"))
    with pytest.raises(ValueError, match="invariant vector"):
        MonotoneCoupledEdgeFlux(invariants, potential(convex_size=invariants.size + 1))
    with pytest.raises(TypeError, match="input-convex"):
        MonotoneCoupledEdgeFlux(
            invariants, cast(Any, Linear(in_size=1, out_size="scalar", rwf=False))
        )
    forged = eqx.tree_at(
        lambda model: model.convex_input_layers[0].weight_transform,
        potential(),
        IdentityTransform(),
    )
    with pytest.raises(ValueError, match="positive convex-input weights"):
        MonotoneCoupledEdgeFlux(invariants, forged)
    external = eqx.tree_at(
        lambda value: value.quadratic.weight,
        invariants,
        None,
        is_leaf=lambda value: value is None,
    )
    with pytest.raises(ValueError, match="own its path weights"):
        MonotoneCoupledEdgeFlux(external, potential())
    law = MonotoneCoupledEdgeFlux(invariants, potential(), odd_size=features.odd.shape[1])
    with pytest.raises(ValueError, match="three-dimensional"):
        law.flux(jnp.zeros((8, _STATE.packed_size)), _cloud(dimension=2))
    with pytest.raises(ValueError, match="feature widths"):
        law.flux(
            jnp.zeros((8, _STATE.packed_size)),
            EdgeFrameFeatures(features.points, features.pairs),
        )
    with pytest.raises(ValueError, match="no legal"):
        O3EdgeInvariants(
            O3Representation(pseudovectors=1),
            quadratic_representation=O3Representation(scalars=1),
            linear_count=1,
            key=jr.key(5),
        )
    with pytest.raises(ValueError, match="polar scalar, vector, or tensor"):
        O3EdgeNetwork(_STATE, O3Representation(pseudovectors=2), key=jr.key(6))
    network = O3EdgeNetwork(
        _STATE, O3Representation(scalars=2, vectors=1), context_size=1, key=jr.key(7)
    )
    with pytest.raises(ValueError, match="even then odd"):
        LipschitzCoupledEdgeFlux(network, even_size=1, odd_size=1)
    with pytest.raises(TypeError, match="O3EdgeNetwork"):
        LipschitzCoupledEdgeFlux(cast(Any, invariants))

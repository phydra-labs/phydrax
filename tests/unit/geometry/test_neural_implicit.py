from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


# tanh(b + s x) + tanh(b - s x) = 2 tanh(b) - CURVATURE x^2 + O(x^4), so pairs of
# tanh units with a linear readout realize a smooth quadratic level set.
_SLOPE = 0.5
_OFFSET = 0.5
_CURVATURE = 2.0 * np.tanh(_OFFSET) * (1.0 - np.tanh(_OFFSET) ** 2) * _SLOPE**2
_BOUNDS = ((-1.5, -1.5), (1.5, 1.5))
_GeometryCapability = phx.geometry.GeometryCapability


def _set_layers(network, layers):
    for index, (weight, bias) in enumerate(layers):
        for select, value in (
            (lambda model, index=index: model.layers[index].weight, weight),
            (lambda model, index=index: model.layers[index].bias, bias),
        ):
            network = eqx.tree_at(select, network, jnp.asarray(value, dtype=jnp.float64))
    return network


def _set_weights(network, first_weight, first_bias, second_weight, second_bias):
    return _set_layers(
        network, ((first_weight, first_bias), (second_weight, second_bias))
    )


def _ellipse_network(*, split: float = 0.0, **options):
    """phi ~ x^2 / 4 + y^2 - 1/4 plus `split` times a bump across x = 0."""
    network = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        hidden_sizes=[6],
        activation=jax.nn.tanh,
        rwf=False,
        **options,
    )
    tanh = np.tanh(_OFFSET)
    return _set_weights(
        network,
        [
            [_SLOPE, 0.0],
            [-_SLOPE, 0.0],
            [0.0, _SLOPE],
            [0.0, -_SLOPE],
            [4.0, 0.0],
            [4.0, 0.0],
        ],
        [_OFFSET, _OFFSET, _OFFSET, _OFFSET, 1.0, -1.0],
        [[-0.25 / _CURVATURE] * 2 + [-1.0 / _CURVATURE] * 2 + [split, -split]],
        [2.5 * tanh / _CURVATURE - 0.25],
    )


def _region(network, **overrides):
    options = dict(
        interior_points=((0.7, 0.0), (-0.7, 0.0)),
        exterior_points=((0.0, 1.0), (1.4, 0.0)),
        sign_margin=0.05,
        evaluation_error=1.0e-12,
        gradient_margin=0.2,
        discovery_resolution=25,
        feature_id="ellipse",
    )
    options.update(overrides)
    return phx.geometry.NeuralImplicitRegion(network, _BOUNDS, **options)


@pytest.fixture(scope="module")
def region():
    return _region(_ellipse_network())


def _bias_id():
    return phx.geometry.ParameterId("ellipse", "layers[1].bias")


@pytest.mark.parametrize(
    "missing",
    [
        "interior_points",
        "exterior_points",
        "sign_margin",
        "evaluation_error",
        "discovery_resolution",
    ],
)
def test_construction_refuses_without_required_certificates(missing):
    with pytest.raises(ValueError, match=missing):
        _region(_ellipse_network(), **{missing: None})


def test_construction_refuses_unconstructible_lipschitz_bound_without_declaration():
    with pytest.raises(ValueError, match="Lipschitz"):
        _region(_ellipse_network(skip_connection=True))


def test_construction_refuses_failed_sign_margin():
    with pytest.raises(ValueError, match="interior_sign"):
        _region(_ellipse_network(), interior_points=((0.0, 0.6),))


def test_three_dimensional_region_certifies_ball_topology():
    network = phx.nn.models.MLP(
        in_size=3, out_size="scalar", hidden_sizes=[6], activation=jax.nn.tanh, rwf=False
    )
    first = np.concatenate((_SLOPE * np.eye(3), -_SLOPE * np.eye(3)))
    network = _set_weights(
        network,
        first,
        [_OFFSET] * 6,
        [[-1.0 / _CURVATURE] * 6],
        [6.0 * np.tanh(_OFFSET) / _CURVATURE - 0.36],
    )
    ball = phx.geometry.NeuralImplicitRegion(
        network,
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        interior_points=((0.0, 0.0, 0.0),),
        exterior_points=((0.9, 0.9, 0.9),),
        sign_margin=0.05,
        evaluation_error=1.0e-12,
        gradient_margin=0.5,
        discovery_resolution=9,
        feature_id="ball",
    )
    topology = ball.certificate.topology
    assert topology.betti_numbers == (1, 0, 0)
    assert topology.boundary_components == (("outer", 2),)
    assert _GeometryCapability.BOUNDARY_NORMAL in ball.compile().capabilities


def test_smooth_region_has_negative_inside_sign_with_sampled_evidence(region):
    geometry = region.compile()
    certificate = geometry.field_certificate

    assert region.certificate.topology.betti_numbers == (1, 0)
    assert certificate.topology_identity is None
    assert certificate.sign_reliability is phx.geometry.SignReliability.LOCAL
    assert certificate.zero_set_accuracy is phx.geometry.ZeroSetAccuracy.APPROXIMATE
    assert certificate.evaluation_error == 1.0e-12
    assert certificate.lipschitz_upper_bound > 0.0
    assert region.certificate.lipschitz_evidence is phx.CapabilityEvidenceKind.CONSTRUCTED
    assert bool(geometry.validity().accepted)

    points = jnp.asarray([[0.0, 0.0], [0.5, 0.2], [1.3, 0.0], [0.0, 0.8]])
    field = geometry.boundary_field(points)
    np.testing.assert_array_equal(np.asarray(field < 0.0), [True, True, False, False])
    np.testing.assert_array_equal(
        np.asarray(geometry.contains(points)), [True, True, False, False]
    )


def test_capabilities_are_only_the_evidenced_ones(region):
    geometry = region.compile()
    assert geometry.capabilities == frozenset(
        {
            _GeometryCapability.REGION_QUERY,
            _GeometryCapability.INTERIOR_SAMPLING,
            _GeometryCapability.BOUNDARY_NORMAL,
        }
    )
    normals = geometry.boundary_normal(jnp.asarray([[1.0, 0.0], [0.0, 0.5]]))
    np.testing.assert_allclose(np.asarray(normals), [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)
    with pytest.raises(NotImplementedError):
        geometry.measure
    with pytest.raises(NotImplementedError):
        geometry.closest_point(jnp.zeros((1, 2)))


def test_nonsmooth_network_does_not_advertise_normals():
    network = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        hidden_sizes=[4],
        activation=jax.nn.relu,
        rwf=False,
    )
    # |x| + |y| - 0.55: a diamond whose zero set avoids the lattice nodes.
    network = _set_weights(
        network,
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        [0.0, 0.0, 0.0, 0.0],
        [[1.0, 1.0, 1.0, 1.0]],
        [-0.55],
    )
    diamond = phx.geometry.NeuralImplicitRegion(
        network,
        ((-1.0, -1.0), (1.0, 1.0)),
        interior_points=((0.0, 0.0),),
        exterior_points=((0.9, 0.9),),
        sign_margin=0.05,
        evaluation_error=0.0,
        gradient_margin=0.5,
        discovery_resolution=17,
        feature_id="diamond",
    )
    geometry = diamond.compile()
    assert diamond.certificate.topology.betti_numbers == (1, 0)
    assert (
        geometry.field_certificate.regularity
        is phx.geometry.FieldRegularity.PIECEWISE_SMOOTH
    )
    assert _GeometryCapability.REGION_QUERY in geometry.capabilities
    assert _GeometryCapability.BOUNDARY_NORMAL not in geometry.capabilities
    with pytest.raises(NotImplementedError):
        geometry.boundary_normal(jnp.asarray([[0.55, 0.0]]))


def test_declared_lipschitz_bound_is_recorded_and_checked():
    declared = _region(_ellipse_network(), lipschitz_upper_bound=10.0)
    assert declared.certificate.lipschitz_evidence is phx.CapabilityEvidenceKind.DECLARED
    certificate = declared.compile().field_certificate
    assert certificate.lipschitz_upper_bound == 10.0
    # A declared bound checked only against sampled gradients certifies neither
    # the sign nor the topology of the region.
    assert certificate.topology_identity is None
    assert certificate.sign_reliability is not phx.geometry.SignReliability.RELIABLE
    with pytest.raises(ValueError, match="lipschitz_sampled_gradient"):
        _region(_ellipse_network(), lipschitz_upper_bound=0.1)


# A lattice cell of the 25-point discovery lattice over `_BOUNDS` spans
# [1.125, 1.25] per axis; the hidden component fills its central square.
_HIDDEN_LOW = 1.1575
_HIDDEN_HIGH = 1.2175
_HIDDEN_STEEPNESS = 400.0
_AND_GAIN = 40.0
_LINEAR_SCALE = 0.05
_HIDDEN_DEPTH = 2.5


def _hidden_component_network():
    """The ellipse field minus `_HIDDEN_DEPTH` on a square between lattice nodes.

    Layer one adds four steep tanh ridges bounding the square; layer two passes
    the ellipse field through a nearly linear tanh unit and fires a second unit
    only where both ridge pairs are active, so the square is a separate
    component of the region that no lattice node or sample point sees.
    """
    ellipse = _ellipse_network()
    steep = _HIDDEN_STEEPNESS
    network = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        hidden_sizes=[10, 2],
        activation=jax.nn.tanh,
        rwf=False,
    )
    half_gain = 0.5 * _AND_GAIN
    return _set_layers(
        network,
        (
            (
                np.concatenate(
                    (
                        np.asarray(ellipse.layers[0].weight),
                        [[steep, 0.0], [steep, 0.0], [0.0, steep], [0.0, steep]],
                    )
                ),
                np.concatenate(
                    (
                        np.asarray(ellipse.layers[0].bias),
                        -steep * np.asarray([_HIDDEN_LOW, _HIDDEN_HIGH] * 2),
                    )
                ),
            ),
            (
                [
                    [
                        *(_LINEAR_SCALE * np.asarray(ellipse.layers[1].weight[0])),
                        0,
                        0,
                        0,
                        0,
                    ],
                    [0.0] * 6 + [half_gain, -half_gain, half_gain, -half_gain],
                ],
                [
                    _LINEAR_SCALE * float(ellipse.layers[1].bias[0]),
                    -1.5 * _AND_GAIN,
                ],
            ),
            (
                [[1.0 / _LINEAR_SCALE, -0.5 * _HIDDEN_DEPTH]],
                [-0.5 * _HIDDEN_DEPTH],
            ),
        ),
    )


def test_zero_set_component_between_samples_is_not_certified():
    region = _region(_hidden_component_network())
    geometry = region.compile()
    hidden = 0.5 * (_HIDDEN_LOW + _HIDDEN_HIGH)

    # The true region has a second component the lattice never sees.
    np.testing.assert_array_equal(
        np.asarray(
            geometry.contains(
                jnp.asarray([[hidden, hidden], [hidden, 0.9], [0.9, hidden]])
            )
        ),
        [True, False, False],
    )
    assert region.certificate.topology.betti_numbers == (1, 0)
    certificate = geometry.field_certificate
    assert certificate.topology_identity is None
    assert certificate.sign_reliability is not phx.geometry.SignReliability.RELIABLE
    assert certificate.zero_set_accuracy is phx.geometry.ZeroSetAccuracy.APPROXIMATE


class _ShiftedField(phx.AbstractArrayModel):
    """A network evaluated at `x - shift` with a FIXED shift."""

    network: phx.nn.models.MLP
    shift: jax.Array = phx.fixed_field()
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, network, shift):
        self.network = network
        self.shift = jnp.asarray(shift, dtype=jnp.float64)
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return self.network(x - self.shift)

    def model_execution_contract(self):
        return self.network.model_execution_contract()


def test_fixed_field_leaves_are_fixed_data_not_design_parameters():
    shift = jnp.asarray([0.05, 0.0])
    network = _ShiftedField(_ellipse_network(), shift)
    region = _region(network, lipschitz_upper_bound=10.0)
    geometry = region.compile()

    names = {parameter.name for parameter in region.parameter_ids}
    assert names == {
        "network.layers[0].weight",
        "network.layers[0].bias",
        "network.layers[1].weight",
        "network.layers[1].bias",
    }
    assert geometry.schema.parameter_ids == region.parameter_ids
    with pytest.raises(KeyError):
        geometry.schema.index(phx.geometry.ParameterId("ellipse", "shift"))

    # The fixed shift still acts on the field and survives recertification.
    points = jnp.asarray([[0.0, 0.0], [0.5, 0.2], [1.2, 0.1]])
    np.testing.assert_allclose(
        np.asarray(geometry.boundary_field(points)),
        np.asarray(jax.vmap(network.network)(points - shift)),
        rtol=0.0,
        atol=1e-12,
    )
    bias = phx.geometry.ParameterId("ellipse", "network.layers[1].bias")
    base = geometry.state.values[geometry.schema.index(bias)]
    recertified = region.recertify(geometry.with_parameters({bias: base - 0.01}).state)
    np.testing.assert_array_equal(np.asarray(recertified.network.shift), shift)


def test_weights_update_through_the_design_state(region):
    geometry = region.compile()
    bias = _bias_id()
    assert bias in region.parameter_ids
    assert geometry.schema.spec(bias).trainable

    points = jnp.asarray([[0.0, 0.0], [0.5, 0.1]])
    base = geometry.state.values[geometry.schema.index(bias)]
    moved = geometry.with_parameters({bias: base + 0.02})
    np.testing.assert_allclose(
        np.asarray(moved.boundary_field(points)),
        np.asarray(geometry.boundary_field(points)) + 0.02,
        rtol=0.0,
        atol=1e-12,
    )
    gradient = jax.grad(
        lambda value: jnp.sum(
            geometry.with_parameters({bias: value}).boundary_field(points)
        )
    )(base)
    np.testing.assert_allclose(np.asarray(gradient), [2.0])

    evidence = moved.validity()
    assert bool(evidence.conditions_satisfied)
    assert not bool(evidence.resolved)
    assert not bool(evidence.accepted)


def test_small_weight_perturbation_keeps_the_sampled_topology(region):
    geometry = region.compile()
    bias = _bias_id()
    base = geometry.state.values[geometry.schema.index(bias)]
    trained = geometry.with_parameters({bias: base - 0.01})

    recertified = region.recertify(trained.state)
    assert recertified.certificate.topology == region.certificate.topology
    assert bool(recertified.compile().validity().accepted)


def test_topology_changing_update_is_rejected(region):
    geometry = region.compile()
    readout = phx.geometry.ParameterId("ellipse", "layers[1].weight")
    split = _ellipse_network(split=0.5).layers[1].weight
    trained = geometry.with_parameters({readout: split})

    with pytest.raises(ValueError, match="topology changed"):
        region.recertify(trained.state)


def test_vanishing_region_is_rejected(region):
    geometry = region.compile()
    bias = _bias_id()
    base = geometry.state.values[geometry.schema.index(bias)]
    with pytest.raises(ValueError, match="interior_sign"):
        region.recertify(geometry.with_parameters({bias: base + 2.0}).state)


def test_geometry_domain_contains_and_samples_the_region(region):
    geometry = region.compile()
    domain = phx.domain.GeometryDomain(geometry)
    points = domain.sample_interior(64, key=jr.key(0))
    assert points.shape == (64, 2)
    assert bool(jnp.all(geometry.contains(points)))
    assert bool(jnp.all(geometry.boundary_field(points) <= 0.0))
    np.testing.assert_array_equal(np.asarray(domain.bounds), np.asarray(_BOUNDS))


def test_exact_sdf_enclosure_requires_declared_field_bounds():
    exact = phx.geometry.exact_signed_distance_certificate(smooth=True)
    assert exact.lipschitz_upper_bound == 1.0
    assert exact.evaluation_error == 0.0
    with pytest.raises(ValueError, match="declared"):
        phx.geometry.ExactSDFEnclosureCertificate(replace(exact, evaluation_error=None))
    with pytest.raises(ValueError, match="non-negative"):
        replace(exact, lipschitz_upper_bound=-1.0)

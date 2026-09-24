import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax import AbstractConstructionCertificate
from phydrax.nn.models import (
    InputConvexCertificate,
    InputConvexNetwork,
    PartiallyInputConvexNetwork,
)


def test_input_convex_network_has_positive_semidefinite_hessians():
    model = InputConvexNetwork(
        in_size=3,
        width_size=12,
        depth=3,
        activation="softplus",
        key=jr.key(0),
    )
    points = jr.normal(jr.key(1), (8, 3))
    hessians = jax.jit(jax.vmap(model.hessian))(points)
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(hessians)

    assert jnp.min(eigenvalues) >= -1e-9
    assert jnp.all(jnp.isfinite(hessians))


def test_input_convex_gradient_is_monotone():
    model = InputConvexNetwork(in_size=2, width_size=10, depth=2, key=jr.key(2))
    first = jnp.asarray([-0.5, 0.8])
    second = jnp.asarray([0.7, -0.2])
    monotonicity = jnp.vdot(
        model.gradient(first) - model.gradient(second), first - second
    ).real

    assert monotonicity >= -1e-9


def test_positive_hidden_weights_survive_optimizer_style_raw_updates():
    model = InputConvexNetwork(in_size=2, width_size=8, depth=3, key=jr.key(3))
    updates = jax.tree.map(
        lambda leaf: -0.2 * jnp.ones_like(leaf) if eqx.is_array(leaf) else None,
        model,
    )
    updated = eqx.apply_updates(model, updates)

    for layer in updated.state_layers:
        effective = layer.weight_transform(layer.weight)
        assert jnp.all(effective > 0.0)
    assert (
        jnp.min(jnp.linalg.eigvalsh(updated.hessian(jnp.asarray([0.2, -0.4])))) >= -1e-9
    )


def test_partially_input_convex_network_is_convex_only_in_designated_input():
    model = PartiallyInputConvexNetwork(
        context_size=2,
        convex_size=3,
        width_size=11,
        depth=3,
        key=jr.key(5),
    )
    contexts = jr.normal(jr.key(6), (5, 2))
    convex_inputs = jr.normal(jr.key(7), (5, 3))
    hessians = jax.vmap(model.convex_hessian)(contexts, convex_inputs)
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(hessians)
    outputs = jax.jit(jax.vmap(model))((contexts, convex_inputs))

    assert outputs.shape == (5,)
    assert jnp.min(eigenvalues) >= -1e-9
    assert jnp.all(jnp.isfinite(hessians))


def test_input_convex_certificate_is_structural_construction_evidence():
    first = InputConvexNetwork(in_size=3, width_size=6, depth=2, key=jr.key(8))
    retrained = InputConvexNetwork(in_size=3, width_size=6, depth=2, key=jr.key(9))
    relu = InputConvexNetwork(
        in_size=3, width_size=6, depth=2, activation="relu", key=jr.key(8)
    )
    certificate = first.input_convex_certificate()

    assert isinstance(certificate, AbstractConstructionCertificate)
    assert certificate.capability_id == "input-convex"
    assert certificate.context_size is None
    assert certificate.convex_input_size == 3
    assert certificate.certificate_id == (
        retrained.input_convex_certificate().certificate_id
    )
    assert certificate.certificate_id != relu.input_convex_certificate().certificate_id


def test_partial_input_convex_certificate_names_the_convex_argument():
    model = PartiallyInputConvexNetwork(
        context_size=2, convex_size=3, width_size=5, depth=2, key=jr.key(10)
    )
    certificate = model.input_convex_certificate()
    joint = InputConvexNetwork(in_size=3, width_size=5, depth=2, key=jr.key(10))

    assert certificate.construction == "partially-input-convex-network"
    assert (certificate.context_size, certificate.convex_input_size) == (2, 3)
    assert certificate.certificate_id != joint.input_convex_certificate().certificate_id


def test_input_convex_certificate_refuses_unconstrained_hidden_couplings():
    model = InputConvexNetwork(in_size=2, width_size=4, depth=2, key=jr.key(11))
    tampered = eqx.tree_at(
        lambda value: value.state_layers[0].weight_transform,
        model,
        None,
        is_leaf=lambda value: value is None,
    )

    with pytest.raises(ValueError, match="positive hidden-state couplings"):
        tampered.input_convex_certificate()


def test_bound_input_convex_field_carries_certificate_until_transformed():
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    model = InputConvexNetwork(in_size=2, width_size=4, depth=2, key=jr.key(12))
    field = domain.Model("x")(model)
    attached = field.metadata["input_convex_certificate"]

    assert isinstance(attached, InputConvexCertificate)
    assert attached.certificate_id == model.input_convex_certificate().certificate_id
    assert "input_convex_certificate" not in (-field).metadata
    assert "input_convex_certificate" not in (field * field).metadata

    boundary = domain.component({"x": phx.domain.Boundary()})
    enforced = phx.enforcement.enforce_dirichlet(field, boundary, target=0.0)
    assert "input_convex_certificate" not in enforced.metadata

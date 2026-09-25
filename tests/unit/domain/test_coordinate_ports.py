#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.axes import AxisKey
from phydrax.domain import DatasetDomain, GeometryDomain, ScalarInterval, TimeInterval


def _square(label: str) -> GeometryDomain:
    return GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile(), label=label
    )


def test_domain_ports_follow_label_order_with_label_scoped_components():
    x_port, t_port = (_square("x") @ TimeInterval(0.0, 1.0)).value_ports()

    assert x_port.semantic_id == "x"
    assert x_port.event_shape == (2,)
    assert x_port.component_ids == ("x[0]", "x[1]")
    assert x_port.axis_keys == (AxisKey("domain-label:x", "0"),)
    assert x_port.representation == "domain-coordinates"
    assert x_port.variance == "neutral"
    assert x_port.dimensions is None
    assert (x_port.space_id, x_port.frame_id, x_port.normalization_id) == (
        None,
        None,
        None,
    )

    assert t_port.semantic_id == "t"
    assert t_port.event_shape == ()
    assert t_port.component_ids == ("t",)
    assert t_port.axis_keys == ()


def test_equal_coordinate_schemas_under_different_labels_have_distinct_ports():
    (x_port,) = _square("x").value_ports()
    (y_port,) = _square("y").value_ports()
    (relabeled,) = _square("x").relabel("y").value_ports()

    assert y_port.component_ids == ("y[0]", "y[1]")
    assert y_port.axis_keys == (AxisKey("domain-label:y", "0"),)
    assert x_port.port_id != y_port.port_id
    assert relabeled.port_id == y_port.port_id


def test_coordinate_ports_depend_on_declaration_not_support_values():
    (first,) = ScalarInterval(0.0, 1.0, label="s").value_ports()
    (second,) = ScalarInterval(-3.0, 5.0, label="s").value_ports()

    assert first.component_ids == ("s",)
    assert first.port_id == second.port_id


def test_structured_coordinates_have_no_dense_port():
    domain = DatasetDomain({"u": jnp.zeros((4, 2))}, label="data")

    with pytest.raises(ValueError, match="'data'"):
        domain.value_ports()


def _fitted_on(*ports):
    width = sum(len(port.component_ids) for port in ports)
    features = jnp.linspace(0.0, 1.0, 8 * width).reshape(8, width)
    return phx.ml.fit(
        phx.ml.linear.RidgeRecipe(alpha=1e-8),
        features,
        jnp.sum(features, axis=-1),
        feature_schema=phx.ml.FeatureSchema.from_ports(ports),
    )


def _identity(*ports):
    return phx.PortMapping(inputs=[(port.port_id, port.port_id) for port in ports])


def test_model_with_ports_binds_only_through_an_explicit_mapping_in_deps_order():
    domain = _square("x") @ TimeInterval(0.0, 1.0)
    x_port, t_port = domain.value_ports()
    fitted = _fitted_on(x_port, t_port)

    with pytest.raises(ValueError, match="requires an explicit port_mapping"):
        domain.Model("x", "t")(fitted.model)
    with pytest.raises(ValueError, match="never repacked"):
        domain.Model("t", "x", port_mapping=_identity(x_port, t_port))(fitted.model)
    with pytest.raises(ValueError, match="unknown owner input ports"):
        domain.Model("x", port_mapping=_identity(x_port, t_port))(fitted.model)
    with pytest.raises(ValueError, match="binds input ports only"):
        domain.Model(
            "x",
            "t",
            port_mapping=phx.PortMapping(
                inputs=_identity(x_port, t_port).inputs,
                outputs=[(t_port.port_id, t_port.port_id)],
            ),
        )

    field = domain.Model("x", "t", port_mapping=_identity(x_port, t_port))(fitted.model)
    evidence = field.port_binding
    assert evidence.inputs == (
        (x_port.port_id, x_port.port_id),
        (t_port.port_id, t_port.port_id),
    )
    assert evidence.outputs == tuple(
        (port.port_id, port.port_id) for port in fitted.model_ports().outputs
    )
    # Domains declare no units, so dimensions are recorded as unverified.
    assert not evidence.dimensions_verified
    assert ("input", x_port.port_id, "dimensions") in evidence.unverified
    assert ("input", x_port.port_id, "axes") not in evidence.unverified

    batch = domain.component().sample(
        phx.domain.PointSampling(4, layout=phx.domain.SampleLayout((("x", "t"),))),
        key=jax.random.key(1),
    )
    x = jnp.asarray(batch.points["x"].data)
    t = jnp.asarray(batch.points["t"].data).reshape(-1, 1)
    expected = jax.vmap(fitted.model)(jnp.concatenate((x, t), axis=-1))
    assert jnp.allclose(jnp.asarray(field(batch).data), expected)
    # A derived field no longer carries the bound model's value or its evidence.
    assert (field + 1.0).port_binding is None


def test_model_ports_must_match_the_mapped_domain_port():
    domain = _square("x") @ _square("y")
    x_port, y_port = domain.value_ports()
    fitted = _fitted_on(x_port)
    crossed = phx.PortMapping(inputs=[(x_port.port_id, y_port.port_id)])

    with pytest.raises(ValueError, match="semantic_id mismatch"):
        domain.Model("y", port_mapping=crossed)(fitted.model)


def test_models_without_ports_take_no_mapping():
    domain = _square("x")
    (x_port,) = domain.value_ports()
    network = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=4, depth=1, key=jax.random.key(0)
    )

    with pytest.raises(ValueError, match="declares no model ports"):
        domain.Model("x", port_mapping=_identity(x_port))(network)
    assert domain.Model("x")(network).port_binding is None

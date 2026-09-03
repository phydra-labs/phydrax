import jax.numpy as jnp

from phydrax.conditions import ArrayCodomain, FieldSpec, ProductFieldSpec
from phydrax.enforcement import finite_feature_linear_representation
from phydrax.linalg import ArraySpace


def test_explicit_linear_representation_round_trips_without_model_introspection():
    field_spec = ProductFieldSpec(
        (FieldSpec("u", ArrayCodomain.from_shape((2,), dtype=float)),)
    )
    space = ArraySpace((2,), dtype=float)
    representation = finite_feature_linear_representation(
        field_spec,
        space,
        space,
        lambda values: values["u"],
        lambda values, coefficients: {"u": coefficients},
        lambda coefficients: {"u": coefficients},
        lambda bound: None,
        support_ids=("fixed-feature-basis",),
    )
    values = {"u": jnp.asarray([1.0, -2.0])}
    coordinates = representation.extract(values)
    assert jnp.allclose(coordinates, values["u"])
    assert jnp.allclose(representation.synthesize(coordinates)["u"], values["u"])
    replaced = representation.replace(values, jnp.asarray([3.0, 4.0]))
    assert jnp.allclose(replaced["u"], jnp.asarray([3.0, 4.0]))
    assert representation.certificate.zero_preserving
    assert representation.certificate.round_trip_exact

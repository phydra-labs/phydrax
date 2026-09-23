import jax.numpy as jnp

from phydrax.conditions import ArrayCodomain, FieldSpec, ProductFieldSpec
from phydrax.enforcement import (
    finite_feature_linear_representation,
    ProductLinearRepresentation,
)
from phydrax.linalg import ArraySpace


def test_explicit_linear_representation_round_trips_without_model_introspection():
    field_spec = ProductFieldSpec(
        (FieldSpec("u", ArrayCodomain.from_shape((2,), dtype="float64")),)
    )
    space = ArraySpace((2,), dtype="float64")
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


def _representation(field):
    field_spec = ProductFieldSpec(
        (FieldSpec(field, ArrayCodomain.from_shape((1,), dtype="float64")),)
    )
    space = ArraySpace((1,), dtype="float64")
    return finite_feature_linear_representation(
        field_spec,
        space,
        space,
        lambda values: values[field],
        lambda values, coefficients: {field: coefficients},
        lambda coefficients: {field: coefficients},
        lambda bound: None,
        support_ids=(f"{field}-basis",),
    )


def test_replacement_preserves_unrepresented_fields_and_composes_disjoint_children():
    representation = ProductLinearRepresentation(
        (_representation("u"), _representation("v"))
    )
    values = {
        "u": jnp.asarray([1.0]),
        "aux": jnp.asarray([9.0]),
        "v": jnp.asarray([2.0]),
    }

    replaced = representation.replace(
        values,
        (jnp.asarray([3.0]), jnp.asarray([4.0])),
    )

    assert tuple(replaced) == ("u", "aux", "v")
    assert jnp.array_equal(replaced["u"], jnp.asarray([3.0]))
    assert jnp.array_equal(replaced["aux"], values["aux"])
    assert jnp.array_equal(replaced["v"], jnp.asarray([4.0]))

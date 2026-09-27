from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.equations import ChemicalComponentCatalog


def _catalog(**overrides: Any) -> Any:
    arguments = {
        "component_names": ("H2", "O2", "H2O"),
        "molar_masses": np.asarray((2.016, 31.998, 18.015)),
        "element_names": ("H", "O"),
        "element_composition": np.asarray(((2, 0, 2), (0, 2, 1)), dtype=np.int32),
    }
    arguments.update(overrides)
    charges = arguments.pop("charges", None)
    return ChemicalComponentCatalog(
        arguments["component_names"],
        arguments["molar_masses"],
        arguments["element_names"],
        arguments["element_composition"],
        charges=charges,
    )


def test_noninteger_charges_are_a_wrong_kind() -> None:
    with pytest.raises(TypeError):
        _catalog(charges=np.asarray((0.0, 0.0, 0.0)))


def test_noninteger_composition_is_a_wrong_kind() -> None:
    with pytest.raises(TypeError):
        _catalog(element_composition=np.asarray(((2.0, 0.0, 2.0), (0.0, 2.0, 1.0))))


def test_charge_shape_is_validated_before_charge_dtype() -> None:
    with pytest.raises(ValueError):
        _catalog(charges=np.asarray((0.0, 0.0)))


def test_integer_charges_are_stored_as_int32() -> None:
    catalog = _catalog(charges=np.asarray((0, 0, -1), dtype=np.int64))

    assert catalog.charges.dtype == np.int32
    np.testing.assert_array_equal(catalog.charges, (0, 0, -1))


_CATALOG_ID = "442b8c4f9b35c0d8c217ffd9d8829587b4e1504a3295887b1808b0243527dadf"


def test_valid_catalog_fields_identity_and_structure() -> None:
    catalog = _catalog(charges=np.asarray((0, 0, -1), dtype=np.int64))
    reference = _catalog()

    assert reference.catalog_id == _CATALOG_ID
    assert catalog.component_names == ("H2", "O2", "H2O")
    assert catalog.element_names == ("H", "O")
    assert (catalog.component_count, catalog.element_count) == (3, 2)
    assert catalog.provenance == "user-supplied"
    assert catalog.molar_masses.dtype == jnp.float64
    assert catalog.element_composition.dtype == jnp.int32
    assert catalog.charges.dtype == jnp.int32
    np.testing.assert_array_equal(catalog.element_composition, ((2, 0, 2), (0, 2, 1)))
    leaves, treedef = jax.tree_util.tree_flatten(reference)
    assert [leaf.shape for leaf in leaves] == [(3,), (2, 3), (3,)]
    assert treedef == jax.tree_util.tree_structure(_catalog())


def test_names_keep_surrounding_whitespace_and_elements_may_be_empty() -> None:
    padded = _catalog(component_names=(" H2", "O2", "H2O"))
    empty = _catalog(
        element_names=(), element_composition=np.zeros((0, 3), dtype=np.int64)
    )

    assert padded.component_names[0] == " H2"
    assert empty.element_count == 0
    assert empty.element_composition.shape == (0, 3)


def test_mass_conversion_follows_numpy_float64_conversion() -> None:
    parsed = _catalog(molar_masses=["2.0", "32.0", "18.0"])

    np.testing.assert_array_equal(parsed.molar_masses, (2.0, 32.0, 18.0))
    with pytest.raises(TypeError):
        _catalog(molar_masses=[2.0 + 1.0j, 32.0, 18.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"component_names": ()},
        {"component_names": ("H2", "", "H2O")},
        {"component_names": ("H2", "H2", "H2O")},
        {"element_names": ("H", "H")},
        {"element_names": ("H", "")},
        {"molar_masses": np.asarray((2.0, 32.0))},
        {"molar_masses": np.asarray((2.0, np.nan, 18.0))},
        {"molar_masses": np.asarray((2.0, np.inf, 18.0))},
        {"molar_masses": np.asarray((2.0, 0.0, 18.0))},
        {"molar_masses": np.asarray((2.0, -32.0, 18.0))},
        {"element_composition": np.asarray(((2, 0, 2),), dtype=np.int32)},
        {"element_composition": np.asarray(((2, 0, 2), (0, -2, 1)), dtype=np.int32)},
        {"charges": np.asarray((0, 0), dtype=np.int32)},
    ],
)
def test_invalid_values_raise_value_errors(overrides: Any) -> None:
    with pytest.raises(ValueError):
        _catalog(**overrides)


def test_empty_provenance_is_refused() -> None:
    with pytest.raises(ValueError):
        ChemicalComponentCatalog(
            ("H2",), np.asarray((2.016,)), ("H",), np.asarray(((2,),)), provenance=""
        )


def test_validation_order_reports_the_first_failing_contract() -> None:
    # Component names are checked before masses; masses before composition.
    with pytest.raises(ValueError):
        _catalog(
            component_names=("H2", "H2", "H2O"),
            element_composition=np.asarray(((2.0, 0.0, 2.0), (0.0, 2.0, 1.0))),
        )
    with pytest.raises(ValueError):
        _catalog(
            molar_masses=np.asarray((2.0, 32.0)),
            element_composition=np.asarray(((2.0, 0.0, 2.0), (0.0, 2.0, 1.0))),
        )
    # A composition shape failure precedes its dtype failure.
    with pytest.raises(ValueError):
        _catalog(element_composition=np.asarray(((2.0, 0.0, 2.0),)))
    # Composition dtype is checked before charges.
    with pytest.raises(TypeError):
        _catalog(
            element_composition=np.asarray(((2.0, 0.0, 2.0), (0.0, 2.0, 1.0))),
            charges=np.asarray((0, 0), dtype=np.int32),
        )

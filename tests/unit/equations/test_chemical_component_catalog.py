import numpy as np
import pytest

from phydrax.equations import ChemicalComponentCatalog


def _catalog(**overrides):
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


def test_noninteger_charges_are_a_wrong_kind():
    with pytest.raises(TypeError):
        _catalog(charges=np.asarray((0.0, 0.0, 0.0)))


def test_noninteger_composition_is_a_wrong_kind():
    with pytest.raises(TypeError):
        _catalog(element_composition=np.asarray(((2.0, 0.0, 2.0), (0.0, 2.0, 1.0))))


def test_charge_shape_is_validated_before_charge_dtype():
    with pytest.raises(ValueError):
        _catalog(charges=np.asarray((0.0, 0.0)))


def test_integer_charges_are_stored_as_int32():
    catalog = _catalog(charges=np.asarray((0, 0, -1), dtype=np.int64))

    assert catalog.charges.dtype == np.int32
    np.testing.assert_array_equal(catalog.charges, (0, 0, -1))

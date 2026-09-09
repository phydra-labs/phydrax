#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._properties import (
    ConcentrationTemperaturePropertyLaw,
    ConstantPropertyLaw,
    TabulatedPropertyLaw,
)


def _table(values=(1.0, 2.0, 4.0), *, source_mask=None):
    return TabulatedPropertyLaw(
        jnp.asarray((0.0, 0.5, 1.0)),
        jnp.asarray(values),
        value_bounds=(0.0, 5.0),
        source_mask=source_mask,
        quantity="diffusivity",
        coordinate="stoichiometry",
        value_unit="m2/s",
        coordinate_unit="1",
        source_id="test:property-source",
    )


def test_tabulated_property_reports_bounds_and_strict_support():
    law = _table()
    result = law.evaluate(jnp.asarray((-0.1, 0.25, 1.1)))
    np.testing.assert_array_equal(result.support, np.asarray((False, True, False)))
    np.testing.assert_allclose(result.values, np.asarray((0.0, 1.5, 0.0)))
    np.testing.assert_allclose(law.support_bounds, np.asarray((0.0, 1.0)))

    masked = _table(source_mask=(True, False, True))
    masked_result = masked(jnp.asarray((0.25, 0.75)))
    np.testing.assert_array_equal(masked_result.support, np.asarray((False, False)))
    np.testing.assert_allclose(masked_result.values, 0.0)


def test_property_value_bounds_fail_closed():
    with pytest.raises(Exception, match="outside declared value bounds"):
        _table(values=(1.0, 6.0, 4.0))
    with pytest.raises(ValueError, match="at least two active"):
        _table(source_mask=(True, False, False))


def test_property_jvp_tracks_dynamic_values_and_query():
    nodes = jnp.asarray((0.0, 0.5, 1.0))

    def evaluate(values, query):
        law = TabulatedPropertyLaw(
            nodes,
            values,
            value_bounds=(-jnp.inf, jnp.inf),
            quantity="open-circuit-voltage",
            coordinate="stoichiometry",
            value_unit="V",
            coordinate_unit="1",
            source_id="test:ocv",
        )
        return law(query).values

    value, tangent = jax.jvp(
        evaluate,
        (jnp.asarray((3.0, 4.0, 4.5)), jnp.asarray(0.25)),
        (jnp.asarray((1.0, 0.0, 0.0)), jnp.asarray(0.1)),
    )
    np.testing.assert_allclose(value, 3.5)
    np.testing.assert_allclose(tangent, 0.7)


def test_constant_and_tabulated_laws_have_static_jit_shapes():
    constant = ConstantPropertyLaw(
        jnp.asarray(2.0),
        jnp.asarray((250.0, 350.0)),
        value_bounds=(0.0, 3.0),
        quantity="heat-capacity",
        coordinate="temperature",
        value_unit="J/K",
        coordinate_unit="K",
        source_id="test:constant",
    )
    table = _table()
    queries = jnp.linspace(0.0, 1.0, 7)
    constant_result, table_result = jax.jit(lambda x: (constant(x), table(x)))(queries)
    assert constant_result.values.shape == (7,)
    assert constant_result.support.shape == (7,)
    assert table_result.values.shape == (7,)
    assert table_result.support.shape == (7,)
    np.testing.assert_allclose(constant.evaluate(300.0, derivative_order=1).values, 0.0)


def test_concentration_temperature_law_support_constant_special_case_and_jvp():
    concentration_nodes = jnp.asarray((500.0, 1000.0, 1500.0))
    temperature_nodes = jnp.asarray((280.0, 300.0, 320.0))
    concentration_scale = (concentration_nodes - 1000.0) / 1000.0
    temperature_scale = (temperature_nodes - 300.0) / 20.0
    values = 2.0 + concentration_scale[:, None] + 0.5 * temperature_scale[None, :]

    def evaluate(table, concentration, temperature):
        law = ConcentrationTemperaturePropertyLaw(
            concentration_nodes,
            temperature_nodes,
            table,
            value_bounds=(-jnp.inf, jnp.inf),
            quantity="electrolyte-diffusivity",
            value_unit="m2/s",
            source_id="test:bivariate-electrolyte",
        )
        return law(concentration, temperature).values

    result = ConcentrationTemperaturePropertyLaw(
        concentration_nodes,
        temperature_nodes,
        values,
        value_bounds=(0.0, 4.0),
        quantity="electrolyte-diffusivity",
        value_unit="m2/s",
        source_id="test:bivariate-electrolyte",
    ).evaluate(
        jnp.asarray((400.0, 750.0, 1250.0)),
        jnp.asarray((300.0, 290.0, 310.0)),
    )
    np.testing.assert_array_equal(result.support, (False, True, True))
    np.testing.assert_allclose(result.values, (0.0, 1.5, 2.5))

    value, tangent = jax.jvp(
        evaluate,
        (values, jnp.asarray(750.0), jnp.asarray(290.0)),
        (
            jnp.ones_like(values),
            jnp.asarray(10.0),
            jnp.asarray(2.0),
        ),
    )
    np.testing.assert_allclose(value, 1.5)
    np.testing.assert_allclose(tangent, 1.06)

    constant = ConcentrationTemperaturePropertyLaw(
        concentration_nodes,
        temperature_nodes,
        jnp.full((3, 3), 0.4),
        value_bounds=(0.0, 1.0),
        quantity="transference-number",
        value_unit="1",
        source_id="test:bivariate-constant",
    )
    constant_result = jax.jit(
        lambda concentration, temperature: constant.evaluate(concentration, temperature)
    )(
        jnp.asarray((600.0, 1400.0)),
        jnp.asarray((285.0, 315.0)),
    )
    np.testing.assert_allclose(constant_result.values, 0.4)
    np.testing.assert_array_equal(constant_result.support, True)


def test_concentration_temperature_law_requires_complete_bounded_support():
    with pytest.raises(ValueError, match="complete active cell"):
        ConcentrationTemperaturePropertyLaw(
            (500.0, 1000.0),
            (280.0, 320.0),
            ((1.0, 1.0), (1.0, 1.0)),
            source_mask=((True, False), (True, True)),
            quantity="electrolyte-conductivity",
            value_unit="S/m",
            source_id="test:no-complete-cell",
        )
    law = ConcentrationTemperaturePropertyLaw(
        (500.0, 1000.0, 1500.0),
        (280.0, 300.0, 320.0),
        jnp.ones((3, 3)),
        source_mask=(
            (True, True, False),
            (True, True, False),
            (False, False, True),
        ),
        quantity="electrolyte-conductivity",
        value_unit="S/m",
        source_id="test:masked-cell",
    )
    unsupported = law.evaluate(1250.0, 310.0)
    assert not bool(unsupported.support)
    np.testing.assert_allclose(unsupported.values, 0.0)

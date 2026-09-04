import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _context(differentiability="native-parameter", matter=0.3):
    scale = cosmology.CosmologyScaleContract(
        cosmology.CODE_COSMOLOGY_SCALE.length_unit,
        cosmology.CODE_COSMOLOGY_SCALE.mass_unit,
        cosmology.CODE_COSMOLOGY_SCALE.time_unit,
    )
    background = cosmology.FLRWBackground(1.0, matter, scale=scale)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="test-request",
        numerical_policy_id="test-policy",
        physics_policy_id="test-physics",
        scale_id=scale.scale_id,
        source_kind="native" if differentiability == "native-parameter" else "external",
        differentiation=differentiability,
    )
    return scale, background, provenance


def _power(
    background,
    provenance,
    values,
    *,
    left="cold_baryon",
    right="cold_baryon",
    stage="linear",
):
    return cosmology.MatterPowerTable(
        [0.5, 1.0],
        [1.0, 2.0, 4.0],
        values,
        cosmology.MatterPowerDescriptor(
            left,
            right,
            stage=stage,
        ),
        background.scale,
        provenance,
        background.realization,
    )


def test_expansion_growth_and_matter_power_products_preserve_realization():
    scale, background, provenance = _context()
    nodes = jnp.asarray([0.25, 0.5, 1.0])
    expansion = cosmology.ExpansionHistory(
        nodes,
        jnp.asarray([8.0, 4.0, 2.0]),
        scale,
        provenance,
        background.realization,
    )
    np.testing.assert_allclose(expansion.hubble(nodes), [8.0, 4.0, 2.0])

    values = jnp.asarray([[0.5, 1.0, 2.0], [1.0, 2.0, 4.0]])
    table = _power(background, provenance, values)
    np.testing.assert_allclose(table.evaluate([1.0, 3.0], 0.75), [0.75, 2.25])
    assert isinstance(table.power_unit, phx.units.UnitDefinition)
    assert table.power_unit.symbol == "code_length^3"
    assert table.descriptor.is_linear_cold_baryon_auto

    other = cosmology.FLRWBackground(1.0, 0.31, scale=scale)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="different physical"):
        jax.block_until_ready(
            background.realization.require_compatible(other.realization, jnp.asarray(1.0))
        )


def test_power_descriptor_allows_signed_cross_but_not_negative_auto():
    _, background, provenance = _context()
    cross = _power(
        background,
        provenance,
        [[1.0, -0.2, 0.5], [1.0, -0.1, 0.5]],
        left="cold_baryon",
        right="massive_neutrino_total",
    )
    assert cross.descriptor.left_field == "cold_baryon"
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="auto/cross"):
        invalid = _power(
            background,
            provenance,
            [[1.0, -0.2, 0.5], [1.0, 0.1, 0.5]],
        )
        jax.block_until_ready(invalid.power_values)


def test_differentiability_policies_are_enforced():
    _, native_background, native_provenance = _context("native-parameter")

    def native_value(amplitude):
        values = amplitude * jnp.asarray([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        return _power(native_background, native_provenance, values).evaluate(1.5, 0.75)

    assert jax.grad(native_value)(jnp.asarray(1.0)) != 0.0

    _, coordinate_background, coordinate_provenance = _context("coordinate-only")

    def stored_value(amplitude):
        values = amplitude * jnp.asarray([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        return _power(coordinate_background, coordinate_provenance, values).evaluate(
            1.5, 0.75
        )

    np.testing.assert_allclose(jax.grad(stored_value)(jnp.asarray(1.0)), 0.0)
    table = _power(
        coordinate_background,
        coordinate_provenance,
        [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]],
    )
    assert jax.grad(lambda k: table.evaluate(k, 0.75))(jnp.asarray(1.5)) != 0.0

    _, constant_background, constant_provenance = _context("constant")
    constant = _power(
        constant_background,
        constant_provenance,
        [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]],
    )
    np.testing.assert_allclose(
        jax.grad(lambda k: constant.evaluate(k, 0.75))(jnp.asarray(1.5)), 0.0
    )


def test_linear_transfer_and_neutrino_power_reconstruction():
    scale, background, provenance = _context()
    descriptor = cosmology.LinearTransferDescriptor(
        ("density/cold_baryon", "density/massive_neutrino_total"),
        gauge="synchronous",
        normalization="relative-to-curvature",
    )
    transfer = cosmology.LinearTransferTable(
        [0.5, 1.0],
        [1.0, 2.0, 4.0],
        jnp.asarray(
            [
                [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]],
                [[0.5, 1.0, 2.0], [1.0, 2.0, 4.0]],
            ]
        ),
        descriptor,
        scale,
        provenance,
        background.realization,
    )
    np.testing.assert_allclose(
        transfer.evaluate("density/cold_baryon", [1.0, 3.0], 0.75),
        [1.5, 4.5],
    )

    cb = _power(background, provenance, [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    nu = _power(
        background,
        provenance,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        left="massive_neutrino_total",
        right="massive_neutrino_total",
    )
    cross = _power(
        background,
        provenance,
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        left="cold_baryon",
        right="massive_neutrino_total",
    )
    total = cosmology.reconstruct_total_matter_power(cb, nu, cross, 0.8, 0.2)
    np.testing.assert_allclose(total.power_values, 3.24)
    assert total.descriptor.left_field == "total_matter"

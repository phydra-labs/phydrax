import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


def _chart(name="g2"):
    return phx.metrix.CoordinateChart(name, tuple(f"x{index}" for index in range(7)))


def test_octonion_bridge_cross_product_and_forms_share_one_convention():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    bridge = phx.metrix.OctonionG2Bridge(algebra, _chart())
    left = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    right = jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    embedded_product = bridge.product(
        bridge.embed_imaginary(left), bridge.embed_imaginary(right)
    )
    coefficients = bridge.associative_differential_form()(jnp.zeros((7,)))

    assert jnp.array_equal(
        bridge.cross(left, right), bridge.extract_imaginary(embedded_product)
    )
    assert jnp.array_equal(coefficients, bridge.coefficients)
    assert bridge.associative_tensor().shape == (7, 7, 7)


def test_canonical_flat_g2_structure_is_compatible_closed_and_ricci_flat():
    bridge = phx.metrix.OctonionG2Bridge(
        phx.metrix.algebra.OctonionAlgebraSpec(), _chart()
    )
    points = jnp.stack((jnp.zeros((7,)), jnp.linspace(-0.2, 0.3, 7)))
    report = phx.metrix.validate_local_g2_structure(
        bridge.local_structure(),
        points,
        require_ricci_flat=True,
    )
    phi = bridge.associative_differential_form()
    psi = bridge.coassociative_differential_form()
    volume = phx.metrix.wedge(phi, psi)(points)

    assert bool(report.valid)
    assert bool(report.algebraically_compatible)
    assert bool(report.torsion_free)
    assert bool(report.ricci_flat)
    assert report.maximum_metric_compatibility_residual < 1e-12
    assert report.maximum_volume_normalization_residual < 1e-12
    assert jnp.allclose(volume[..., 0], 7.0, atol=1e-12)


def test_g2_validation_separates_algebraic_and_torsion_failures():
    chart = _chart()
    bridge = phx.metrix.OctonionG2Bridge(phx.metrix.algebra.OctonionAlgebraSpec(), chart)
    perturbed = bridge.coefficients.at[0].add(0.1)
    incompatible_form = phx.metrix.DifferentialForm(
        lambda point: perturbed.astype(point.dtype),
        chart=chart,
        degree=3,
    )
    incompatible = phx.metrix.validate_local_g2_structure(
        phx.metrix.LocalG2Structure(bridge.metric, incompatible_form),
        jnp.zeros((7,)),
        require_torsion_free=False,
        raise_on_error=False,
    )
    varying_form = phx.metrix.DifferentialForm(
        lambda point: bridge.coefficients.at[0].set(
            bridge.coefficients[0] * (1.0 + point[6])
        ),
        chart=chart,
        degree=3,
    )
    torsion = phx.metrix.validate_local_g2_structure(
        phx.metrix.LocalG2Structure(bridge.metric, varying_form),
        jnp.zeros((7,)),
        require_torsion_free=True,
        raise_on_error=False,
    )

    assert not bool(incompatible.algebraically_compatible)
    assert not bool(incompatible.valid)
    assert bool(torsion.algebraically_compatible)
    assert not bool(torsion.closed)
    assert not bool(torsion.torsion_free)
    assert not bool(torsion.valid)


def test_g2_validation_is_jittable_when_runtime_errors_are_disabled():
    bridge = phx.metrix.OctonionG2Bridge(
        phx.metrix.algebra.OctonionAlgebraSpec(), _chart()
    )
    validate = eqx.filter_jit(
        lambda point: (
            phx.metrix.validate_local_g2_structure(
                bridge.local_structure(), point, raise_on_error=False
            ).maximum_metric_compatibility_residual
        )
    )

    assert validate(jnp.zeros((7,))) < 1e-12


def test_g2_geometry_rejects_invalid_charts_forms_and_lie_group_inference():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    chart = _chart()
    bridge = phx.metrix.OctonionG2Bridge(algebra, chart)

    with pytest.raises(ValueError, match="seven-dimensional"):
        phx.metrix.OctonionG2Bridge(
            algebra,
            phx.metrix.CoordinateChart("wrong", ("x", "y")),
        )
    with pytest.raises(ValueError, match="degree three"):
        phx.metrix.LocalG2Structure(
            bridge.metric,
            phx.metrix.DifferentialForm(
                lambda point: jnp.zeros((21,), dtype=point.dtype),
                chart=chart,
                degree=2,
            ),
        )
    with pytest.raises(ValueError, match="Lie-group"):
        phx.metrix.algebra.unit_algebra_state_geometry(algebra)

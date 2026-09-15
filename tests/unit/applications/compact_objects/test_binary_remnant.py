import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_uib2016_remnant_matches_reference_configurations():
    compact = phx.applications.compact_objects
    plan = compact.AlignedBinaryRemnantPlan()
    cases = (
        ((30.0, 30.0, 0.0, 0.0), (0.6863698793893069, 0.04841605714434129)),
        ((36.0, 24.0, 0.5, 0.2), (0.7979017752948777, 0.058315258273263684)),
        ((36.0, 24.0, -0.5, 0.2), (0.5589805903020263, 0.03880143913729326)),
    )
    for inputs, reference in cases:
        result = plan.evaluate(*inputs)
        np.testing.assert_allclose(
            result.final_dimensionless_spin, reference[0], rtol=0.0, atol=2.0e-14
        )
        np.testing.assert_allclose(
            result.radiated_energy_fraction, reference[1], rtol=0.0, atol=2.0e-14
        )
        np.testing.assert_allclose(
            result.final_mass,
            (inputs[0] + inputs[1]) * (1.0 - reference[1]),
            rtol=0.0,
            atol=2.0e-13,
        )
        assert result.successful
        assert bool(result.finite)
        assert bool(result.physically_valid)


def test_aligned_remnant_fit_is_jittable_and_has_interior_derivatives():
    compact = phx.applications.compact_objects
    plan = compact.AlignedBinaryRemnantPlan()
    result = eqx.filter_jit(plan.evaluate)(36.0, 24.0, 0.5, 0.2)
    derivative = jax.grad(
        lambda spin: plan.evaluate(36.0, 24.0, spin, 0.2).final_dimensionless_spin
    )(jnp.asarray(0.5))

    assert result.successful
    assert bool(result.derivative_valid)
    assert np.isfinite(float(derivative))
    assert float(derivative) > 0.0


def test_aligned_remnant_fit_fails_closed_outside_qualified_domain():
    compact = phx.applications.compact_objects
    plan = compact.AlignedBinaryRemnantPlan()
    outside_ratio = plan.evaluate(9.0, 1.0, 0.0, 0.0)
    outside_spin = plan.evaluate(2.0, 1.0, 0.81, 0.0)
    reversed_masses = plan.evaluate(1.0, 2.0, 0.0, 0.0)

    for result in (outside_ratio, outside_spin, reversed_masses):
        assert not result.successful
        assert int(result.status) == int(
            compact.AlignedBinaryRemnantStatus.OUTSIDE_CALIBRATION
        )
        assert bool(jnp.isnan(result.final_mass))

    nonfinite = plan.evaluate(jnp.nan, 1.0, 0.0, 0.0)
    nonphysical = plan.evaluate(-1.0, 1.0, 0.0, 0.0)
    assert not bool(nonfinite.finite)
    assert not bool(nonfinite.physically_valid)
    assert int(nonfinite.status) == int(
        compact.AlignedBinaryRemnantStatus.NONFINITE_INPUT
    )
    assert bool(nonphysical.finite)
    assert not bool(nonphysical.physically_valid)

    with pytest.raises(TypeError, match="real numeric scalars"):
        plan.evaluate(True, 1.0, 0.0, 0.0)
    with pytest.raises(TypeError, match="real numeric scalars"):
        plan.evaluate(1.0 + 0.0j, 1.0, 0.0, 0.0)

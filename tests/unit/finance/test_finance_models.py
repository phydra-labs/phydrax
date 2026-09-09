import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.finance.models._commodity import SchwartzTwoFactorModel
from phydrax.finance.models._dependence import CorrelationMatrix, MultiAssetLognormalModel
from phydrax.finance.models._diffusion import (
    HestonModel,
    LocalVolatilityModel,
    SABRModel,
)
from phydrax.finance.models._jump import (
    CGMYModel,
    KouJumpDiffusionModel,
    MertonJumpDiffusionModel,
    NormalInverseGaussianModel,
    VarianceGammaModel,
)
from phydrax.finance.models._rough import RoughBergomiModel


def test_local_volatility_interpolation_and_invalid_variance_fail_closed():
    model = LocalVolatilityModel(
        jnp.array([0.5, 1.0]),
        jnp.array([-0.2, 0.2]),
        jnp.array([[0.02, 0.04], [0.04, 0.08]]),
    )
    assert jnp.isclose(model.variance(0.75, 0.0), 0.045)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="local_variance"):
        LocalVolatilityModel(
            jnp.array([0.5, 1.0]),
            jnp.array([-0.2, 0.2]),
            jnp.array([[0.02, 0.04], [0.04, -0.01]]),
        )


def test_diffusion_and_rough_model_admissibility():
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="Heston"):
        HestonModel(1.0, 0.01, 1.0, -0.5, 0.01)
    sabr = SABRModel(0.2, 0.5, 0.4, -0.2)
    assert jnp.isfinite(sabr.implied_volatility(100.0, 100.0, 1.0))
    rough = RoughBergomiModel(
        0.1,
        1.5,
        -0.7,
        jnp.array([0.0, 1.0]),
        jnp.array([0.04, 0.05]),
    )
    weights = rough.kernel_weights(0.01, 8)
    assert jnp.all(weights > 0.0)
    assert jnp.all(jnp.diff(weights) < 0.0)


def test_jump_characteristic_exponents_normalize_and_moment_checks_fail_closed():
    models = (
        MertonJumpDiffusionModel(0.2, 0.5, -0.1, 0.2),
        KouJumpDiffusionModel(0.2, 0.5, 0.4, 3.0, 4.0),
        VarianceGammaModel(0.2, -0.1, 0.3),
        NormalInverseGaussianModel(5.0, -0.5, 0.2),
        CGMYModel(0.2, 5.0, 6.0, 0.5),
    )
    for model in models:
        assert jnp.isclose(model.characteristic_exponent(0.0), 0.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="upward_rate"):
        KouJumpDiffusionModel(0.2, 0.5, 0.4, 0.9, 4.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="first moment"):
        VarianceGammaModel(1.0, 1.0, 1.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="NIG tail"):
        NormalInverseGaussianModel(1.0, 0.5, 0.2)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="right_rate"):
        CGMYModel(0.2, 5.0, 0.9, 0.5)


def test_multiasset_and_commodity_covariances_are_financially_admissible():
    dependence = CorrelationMatrix(jnp.array([[1.0, 0.25], [0.25, 1.0]]))
    model = MultiAssetLognormalModel(jnp.array([0.2, 0.3]), dependence)
    expected = jnp.array([[0.04, 0.015], [0.015, 0.09]])
    assert jnp.allclose(model.covariance(), expected)
    commodity = SchwartzTwoFactorModel(1.2, 0.3, 0.15, -0.4, jnp.array([0.0, 0.0]))
    assert commodity.log_spot_variance(2.0) > 0.0
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="correlation"):
        CorrelationMatrix(jnp.array([[1.0, 1.1], [1.1, 1.0]]))

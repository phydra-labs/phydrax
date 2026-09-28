import jax.numpy as jnp
import numpy as np

from phydrax.optics.wave import ThinFilmInterferencePlan
from phydrax.rendering import (
    SpectralColorimetryPlan,
    SpectralIlluminant,
    thin_film_surface_colors,
    ThinFilmAppearancePlan,
    ThinFilmSurfaceColorStatus,
)


_VISIBLE = np.arange(380.0, 781.0, 5.0) * 1.0e-9


def _appearance() -> ThinFilmAppearancePlan:
    power = np.zeros_like(_VISIBLE)
    green = np.flatnonzero(
        np.isclose(_VISIBLE, 550.0e-9, rtol=0.0, atol=1.0e-15)
    )[0]
    power[green] = 1.0
    illuminant = SpectralIlluminant(
        _VISIBLE,
        power,
        illuminant_id="green-line-test",
    )
    return ThinFilmAppearancePlan(
        ThinFilmInterferencePlan(_VISIBLE, 1.0, 1.33, 1.0),
        SpectralColorimetryPlan(_VISIBLE, illuminant, exposure=4.0),
        two_sided=True,
    )


def test_surface_colors_preserve_support_rejection_and_thickness_color_shift() -> None:
    wavelength = 550.0e-9
    thickness = jnp.asarray(
        (wavelength / (2.0 * 1.33), wavelength / (4.0 * 1.33), 260.0e-9, -1.0e-9)
    )
    support = jnp.asarray((True, True, False, True))
    result = thin_film_surface_colors(
        _appearance(),
        thickness,
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        support,
    )

    np.testing.assert_array_equal(result.accepted, (True, True, False, False))
    np.testing.assert_array_equal(
        result.status,
        (
            ThinFilmSurfaceColorStatus.SUCCESS,
            ThinFilmSurfaceColorStatus.SUCCESS,
            ThinFilmSurfaceColorStatus.UNSUPPORTED,
            ThinFilmSurfaceColorStatus.APPEARANCE_REJECTED,
        ),
    )
    assert bool(jnp.all(jnp.isfinite(result.linear_srgb[:2])))
    assert bool(jnp.all(jnp.isfinite(result.encoded_srgb[:2])))
    assert bool(jnp.all(jnp.isnan(result.linear_srgb[2:])))
    assert bool(jnp.all(jnp.isnan(result.encoded_srgb[2:])))
    assert float(jnp.linalg.norm(result.linear_srgb[0])) < 1.0e-10
    assert float(jnp.linalg.norm(result.linear_srgb[1])) > 0.1

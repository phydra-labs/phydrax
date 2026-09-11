#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""JAX-native named special functions and integrals."""

from ._airy import airy, airye
from ._carlson import elliprc, elliprd, elliprf, elliprg, elliprj
from ._continuation import (
    iv_order_derivative,
    ive_order_derivative,
    jv_order_derivative,
    kv_order_derivative,
    kve_order_derivative,
    principal_log,
    principal_sqrt,
    yv_order_derivative,
)
from ._cylindrical_bessel import hankel1, hankel2, jv, yv
from ._dilog import dilog, spence
from ._faddeeva import dawsn, voigt_profile, wofz
from ._gegenbauer import gegenbauer_alpha_derivative, gegenbauer_c, gegenbauer_vander
from ._jacobi import ellipam, ellipj
from ._legendre import ellipe, ellipeinc, ellipk, ellipkinc, ellipkm1, ellippi, ellippiinc
from ._modified_bessel import iv, ive, kv, kve
from ._normal import (
    normal_cdf,
    normal_logcdf,
    normal_logpdf,
    normal_logsurvival,
    normal_pdf,
    normal_quantile,
    normal_survival,
)
from ._polylog import polylog
from ._solid_harmonic import solid_harmonic_irregular, solid_harmonic_regular
from ._spherical_harmonic import sph_harm_y, sph_harm_y_cart, sph_legendre_p
from ._zeta import hurwitz_zeta, zeta


__all__ = [
    "airy",
    "airye",
    "dawsn",
    "dilog",
    "ellipam",
    "ellipe",
    "ellipeinc",
    "ellipj",
    "ellipk",
    "ellipkinc",
    "ellipkm1",
    "ellippi",
    "ellippiinc",
    "elliprc",
    "elliprd",
    "elliprf",
    "elliprg",
    "elliprj",
    "gegenbauer_alpha_derivative",
    "gegenbauer_c",
    "gegenbauer_vander",
    "hankel1",
    "hankel2",
    "hurwitz_zeta",
    "iv",
    "iv_order_derivative",
    "ive",
    "ive_order_derivative",
    "jv",
    "jv_order_derivative",
    "kv",
    "kv_order_derivative",
    "kve",
    "kve_order_derivative",
    "normal_cdf",
    "normal_logcdf",
    "normal_logpdf",
    "normal_logsurvival",
    "normal_pdf",
    "normal_quantile",
    "normal_survival",
    "polylog",
    "principal_log",
    "principal_sqrt",
    "solid_harmonic_irregular",
    "solid_harmonic_regular",
    "spence",
    "sph_harm_y",
    "sph_harm_y_cart",
    "sph_legendre_p",
    "voigt_profile",
    "wofz",
    "yv",
    "yv_order_derivative",
    "zeta",
]

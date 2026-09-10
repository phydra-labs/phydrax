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
from ._faddeeva import dawsn, voigt_profile, wofz
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
from ._spherical_harmonic import sph_harm_y, sph_harm_y_cart, sph_legendre_p


__all__ = [
    "airy",
    "airye",
    "dawsn",
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
    "hankel1",
    "hankel2",
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
    "principal_log",
    "principal_sqrt",
    "sph_harm_y",
    "sph_harm_y_cart",
    "sph_legendre_p",
    "voigt_profile",
    "wofz",
    "yv",
    "yv_order_derivative",
]

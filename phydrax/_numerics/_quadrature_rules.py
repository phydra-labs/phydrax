#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


class QuadratureRuleData(NamedTuple):
    """Canonical quadrature data on a reference interval or cell."""

    nodes: Array
    weights: Array
    embedded_weights: Array | None
    degree: int | None


# Nonnegative Gauss--Kronrod abscissae and weights from the published
# QUADPACK QK15/QK21/QK31/QK41/QK51/QK61 rules. Storing only one symmetric
# half keeps the provenance data inspectable and avoids backend code reuse.
_GAUSS_KRONROD_POSITIVE: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    15: (
        (
            0.0,
            0.20778495500789848,
            0.4058451513773972,
            0.5860872354676911,
            0.7415311855993945,
            0.8648644233597691,
            0.9491079123427585,
            0.9914553711208126,
        ),
        (
            0.20948214108472782,
            0.20443294007529889,
            0.19035057806478542,
            0.1690047266392679,
            0.14065325971552592,
            0.10479001032225019,
            0.06309209262997856,
            0.022935322010529224,
        ),
    ),
    21: (
        (
            0.0,
            0.14887433898163122,
            0.2943928627014602,
            0.4333953941292472,
            0.5627571346686047,
            0.6794095682990244,
            0.7808177265864169,
            0.8650633666889845,
            0.9301574913557082,
            0.9739065285171717,
            0.9956571630258081,
        ),
        (
            0.1494455540029169,
            0.14773910490133849,
            0.14277593857706009,
            0.13470921731147334,
            0.12349197626206584,
            0.10938715880229764,
            0.0931254545836976,
            0.07503967481091996,
            0.054755896574351995,
            0.032558162307964725,
            0.011694638867371874,
        ),
    ),
    31: (
        (
            0.0,
            0.1011420669187175,
            0.20119409399743451,
            0.29918000715316884,
            0.3941513470775634,
            0.4850818636402397,
            0.5709721726085388,
            0.650996741297417,
            0.7244177313601701,
            0.790418501442466,
            0.8482065834104272,
            0.8972645323440819,
            0.937273392400706,
            0.9677390756791391,
            0.9879925180204854,
            0.9980022986933971,
        ),
        (
            0.10133000701479154,
            0.10076984552387559,
            0.09917359872179196,
            0.09664272698362368,
            0.09312659817082532,
            0.08856444305621176,
            0.08308050282313302,
            0.07684968075772038,
            0.06985412131872826,
            0.06200956780067064,
            0.05348152469092809,
            0.04458975132476488,
            0.03534636079137585,
            0.02546084732671532,
            0.015007947329316122,
            0.005377479872923349,
        ),
    ),
    41: (
        (
            0.0,
            0.07652652113349734,
            0.15260546524092267,
            0.22778585114164507,
            0.301627868114913,
            0.37370608871541955,
            0.4435931752387251,
            0.5108670019508271,
            0.5751404468197103,
            0.636053680726515,
            0.6932376563347514,
            0.7463319064601508,
            0.7950414288375512,
            0.8391169718222188,
            0.878276811252282,
            0.912234428251326,
            0.9408226338317548,
            0.9639719272779138,
            0.9815078774502503,
            0.9931285991850949,
            0.9988590315882777,
        ),
        (
            0.07660071191799965,
            0.07637786767208074,
            0.07570449768455667,
            0.07458287540049918,
            0.07303069033278667,
            0.07105442355344407,
            0.06864867292852161,
            0.06583459713361842,
            0.06265323755478117,
            0.05911140088063957,
            0.05519510534828599,
            0.05094457392372869,
            0.04643482186749767,
            0.041668873327973685,
            0.036600169758200796,
            0.0312873067770328,
            0.02588213360495116,
            0.020388373461266523,
            0.014626169256971253,
            0.008600269855642943,
            0.0030735837185205317,
        ),
    ),
    51: (
        (
            0.0,
            0.06154448300568508,
            0.1228646926107104,
            0.1837189394210489,
            0.24386688372098844,
            0.30308953893110785,
            0.36117230580938786,
            0.4178853821930377,
            0.473002731445715,
            0.5263252843347191,
            0.577662930241223,
            0.6268100990103174,
            0.6735663684734684,
            0.7177664068130843,
            0.7592592630373576,
            0.7978737979985001,
            0.833442628760834,
            0.8658470652932756,
            0.8949919978782753,
            0.9207471152817016,
            0.9429745712289743,
            0.9616149864258425,
            0.9766639214595175,
            0.9880357945340772,
            0.9955569697904981,
            0.9992621049926098,
        ),
        (
            0.061580818067832936,
            0.061471189871425316,
            0.061128509717053046,
            0.06053945537604586,
            0.05972034032417406,
            0.058689680022394206,
            0.057437116361567835,
            0.055950811220412316,
            0.05425112988854549,
            0.05236288580640747,
            0.05027767908071567,
            0.04798253713883671,
            0.04550291304992179,
            0.04287284502017005,
            0.04008382550403238,
            0.03711627148341554,
            0.034002130274329335,
            0.030792300167387487,
            0.02747531758785174,
            0.024009945606953215,
            0.020435371145882834,
            0.0168478177091283,
            0.013236229195571676,
            0.009473973386174152,
            0.005561932135356714,
            0.001987383892330316,
        ),
    ),
    61: (
        (
            0.0,
            0.0514718425553177,
            0.10280693796673702,
            0.15386991360858354,
            0.20452511668230988,
            0.25463692616788985,
            0.30407320227362505,
            0.3527047255308781,
            0.4004012548303944,
            0.44703376953808915,
            0.49248046786177857,
            0.5366241481420199,
            0.5793452358263617,
            0.6205261829892429,
            0.6600610641266269,
            0.6978504947933158,
            0.7337900624532268,
            0.7677774321048262,
            0.799727835821839,
            0.8295657623827684,
            0.8572052335460612,
            0.8825605357920527,
            0.9055733076999078,
            0.9262000474292743,
            0.94437444474856,
            0.9600218649683075,
            0.9731163225011262,
            0.9836681232797472,
            0.9916309968704046,
            0.9968934840746495,
            0.9994844100504906,
        ),
        (
            0.05149472942945157,
            0.05142612853745902,
            0.051221547849258774,
            0.05088179589874961,
            0.05040592140278235,
            0.04979568342707421,
            0.04905543455502978,
            0.04818586175708713,
            0.04718554656929915,
            0.04605923827100699,
            0.04481480013316266,
            0.04345253970135607,
            0.041969810215164244,
            0.040374538951535956,
            0.038678945624727595,
            0.03688236465182123,
            0.034979338028060025,
            0.03298144705748372,
            0.030907257562387762,
            0.02875404876504129,
            0.0265099548823331,
            0.0241911620780806,
            0.021828035821609193,
            0.019414141193942382,
            0.01692088918905327,
            0.014369729507045804,
            0.011823015253496341,
            0.009273279659517764,
            0.0066307039159312926,
            0.003890461127099884,
            0.0013890136986770077,
        ),
    ),
}


def _symmetric(values: np.ndarray) -> np.ndarray:
    return np.concatenate((-values[:0:-1], values))


def _symmetric_weights(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values[:0:-1], values))


@lru_cache(maxsize=None)
def gauss_legendre_data(order: int) -> QuadratureRuleData:
    """Return the order-``n`` Gauss--Legendre rule on ``[-1, 1]``."""
    order_ = int(order)
    if order_ < 1:
        raise ValueError("Gauss--Legendre order must be positive.")
    nodes, weights = np.polynomial.legendre.leggauss(order_)
    return QuadratureRuleData(
        jnp.asarray(nodes), jnp.asarray(weights), None, 2 * order_ - 1
    )


@lru_cache(maxsize=None)
def gauss_kronrod_data(order: int) -> QuadratureRuleData:
    """Return a published embedded Gauss--Kronrod rule on ``[-1, 1]``."""
    order_ = int(order)
    if order_ not in _GAUSS_KRONROD_POSITIVE:
        allowed = tuple(_GAUSS_KRONROD_POSITIVE)
        raise ValueError(f"Gauss--Kronrod order must be one of {allowed}.")
    positive_nodes, positive_weights = _GAUSS_KRONROD_POSITIVE[order_]
    nodes_np = _symmetric(np.asarray(positive_nodes, dtype=float))
    weights_np = _symmetric_weights(np.asarray(positive_weights, dtype=float))
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss((order_ - 1) // 2)
    embedded = np.zeros(order_, dtype=float)
    for node, weight in zip(gauss_nodes, gauss_weights, strict=True):
        index = int(np.argmin(np.abs(nodes_np - node)))
        if abs(float(nodes_np[index] - node)) > 5e-14:
            raise RuntimeError("Published Kronrod rule does not contain its Gauss nodes.")
        embedded[index] = weight
    return QuadratureRuleData(
        jnp.asarray(nodes_np),
        jnp.asarray(weights_np),
        jnp.asarray(embedded),
        None,
    )


@lru_cache(maxsize=None)
def clenshaw_curtis_data(order: int) -> QuadratureRuleData:
    """Return an endpoint-including Clenshaw--Curtis rule on ``[-1, 1]``."""
    order_ = int(order)
    if order_ < 1:
        raise ValueError("Clenshaw--Curtis order must be positive.")
    if order_ == 1:
        return QuadratureRuleData(
            jnp.asarray([0.0]),
            jnp.asarray([2.0]),
            None,
            1,
        )
    n = order_ - 1
    theta = np.pi * np.arange(order_, dtype=float) / float(n)
    nodes = np.cos(theta)
    weights = np.zeros(order_, dtype=float)
    interior = np.arange(1, n)
    values = np.ones(max(n - 1, 0), dtype=float)
    if n % 2 == 0:
        weights[0] = weights[-1] = 1.0 / (n * n - 1.0)
        for k in range(1, n // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
        values -= np.cos(n * theta[interior]) / (n * n - 1.0)
    else:
        weights[0] = weights[-1] = 1.0 / (n * n)
        for k in range(1, (n + 1) // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
    weights[interior] = 2.0 * values / float(n)
    nodes = nodes[::-1].copy()
    weights = weights[::-1].copy()
    return QuadratureRuleData(jnp.asarray(nodes), jnp.asarray(weights), None, order_ - 1)


@lru_cache(maxsize=None)
def tanh_sinh_data(order: int) -> QuadratureRuleData:
    """Return a finite double-exponential trapezoidal rule on ``[-1, 1]``."""
    order_ = int(order)
    if order_ < 3 or order_ % 2 == 0:
        raise ValueError("Tanh--sinh order must be an odd integer of at least three.")
    half = order_ // 2
    step = 3.5 / float(half)
    t = step * np.arange(-half, half + 1, dtype=float)
    sinh_t = np.sinh(t)
    argument = 0.5 * np.pi * sinh_t
    nodes = np.tanh(argument)
    weights = 0.5 * np.pi * np.cosh(t) / np.cosh(argument) ** 2 * step
    weights *= 2.0 / np.sum(weights)
    return QuadratureRuleData(jnp.asarray(nodes), jnp.asarray(weights), None, None)


__all__ = [
    "QuadratureRuleData",
    "clenshaw_curtis_data",
    "gauss_kronrod_data",
    "gauss_legendre_data",
    "tanh_sinh_data",
]

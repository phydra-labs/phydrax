# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent equations from primary nucleotide-model publications.

Ouldridge et al. arXiv:1009.4480 Appendix A (f1--f5); Sulc et al.
arXiv:1403.4180 Appendix A.2 (RNA geometry/interaction products); Snodin et al.
arXiv:1504.00821 Appendix A (screening and f6); Ratajczyk et al.
arXiv:2311.07709 II.2 (hybrid nonbonded terms use DNA functional forms).
No upstream software or parameter table is incorporated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def _acos(value):
    return jnp.arccos(jnp.clip(value, -1.0, 1.0))


@_acos.defjvp
def _acos_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    interior = jnp.abs(value) < 1.0
    denominator = jnp.sqrt(jnp.where(interior, 1 - value * value, 1.0))
    # Collinear frame directions have zero first-order physical angle work at
    # the smooth extrema of the admitted modulation products. No Hessian claim.
    return _acos(value), jnp.where(interior, -tangent / denominator, 0.0)


def _dot(a, b):
    return jnp.sum(a * b, axis=-1)


def _unit(vector):
    return vector / jnp.sqrt(jnp.sum(vector * vector, axis=-1, keepdims=True))


def radial_value(r, parameters, kind):
    """f1/f2: central shifted well joined C1 to compact quadratic tails.

    Parameters are [amplitude,r0,reference_cut,low,high,width]. Width is used
    only by the Morse branch. Smoothing coefficients are derived, not supplied.
    """
    amplitude, r0, rc, low, high, width = parameters

    def centre(x):
        if kind == "morse":
            exponential = jnp.exp(-width * (x - r0))
            reference = jnp.exp(-width * (rc - r0))
            return (1 - exponential) ** 2 - (
                1 - reference
            ) ** 2, 2 * width * exponential * (1 - exponential)
        return 0.5 * ((x - r0) ** 2 - (rc - r0) ** 2), x - r0

    vlo, dlo = centre(low)
    vhi, dhi = centre(high)
    cutlo, cuthi = low - 2 * vlo / dlo, high - 2 * vhi / dhi
    blo, bhi = dlo * dlo / (4 * vlo), dhi * dhi / (4 * vhi)
    # Bound inactive central evaluation to avoid overflow outside compact support.
    middle, _ = centre(jnp.where(r < low, low, jnp.where(r > high, high, r)))
    value = jnp.where(
        r < low,
        blo * (r - cutlo) ** 2,
        jnp.where(r > high, bhi * (r - cuthi) ** 2, middle),
    )
    return amplitude * jnp.where((r >= cutlo) & (r <= cuthi), value, 0.0)


def radial_support(parameters, kind):
    p = np.asarray(parameters, dtype=float)
    _, r0, rc, low, high, width = p

    def centre(x):
        if kind == "morse":
            e, ec = np.exp(-width * (x - r0)), np.exp(-width * (rc - r0))
            return (1 - e) ** 2 - (1 - ec) ** 2, 2 * width * e * (1 - e)
        return 0.5 * ((x - r0) ** 2 - (rc - r0) ** 2), x - r0

    vlo, dlo = centre(low)
    vhi, dhi = centre(high)
    if not (0 < low < r0 < high < rc and vlo < 0 and vhi < 0 and dlo < 0 and dhi > 0):
        raise ValueError(
            "Radial matching points must straddle a negative well with outward slopes."
        )
    cutlo, cuthi = low - 2 * vlo / dlo, high - 2 * vhi / dhi
    if cutlo < 0 or cutlo >= low or cuthi <= high:
        raise ValueError("Radial quadratic tails have invalid compact support.")
    return cutlo, cuthi


def excluded_value(r, parameters):
    """f3: repulsive LJ continued quadratically to zero, not WCA."""
    epsilon, sigma, join = parameters
    ratio = sigma / join
    value = 4 * (ratio**12 - ratio**6)
    derivative = 24 / join * (ratio**6 - 2 * ratio**12)
    cutoff = join - 2 * value / derivative
    curvature = derivative**2 / (4 * value)
    safe_r = jnp.where(r > join, join, r)
    x6 = (sigma / safe_r) ** 6
    central = 4 * (x6 * x6 - x6)
    return epsilon * jnp.where(
        r <= join, central, jnp.where(r < cutoff, curvature * (r - cutoff) ** 2, 0.0)
    )


def angular_value(theta, parameters):
    """f4, with [curvature, preferred angle, matching half-width]."""
    a, theta0, join = parameters
    delta = jnp.abs(theta - theta0)
    cutoff = 1 / (a * join)
    curvature = a * a * join * join / (1 - a * join * join)
    return jnp.where(
        delta <= join,
        1 - a * delta * delta,
        jnp.where(delta < cutoff, curvature * (delta - cutoff) ** 2, 0.0),
    )


def helicity_value(x, parameters):
    """f5 one-sided helicity window, parameters [curvature,negative join]."""
    a, join = parameters
    cutoff = 1 / (a * join)
    curvature = a * a * join * join / (1 - a * join * join)
    return jnp.where(
        x >= 0,
        1.0,
        jnp.where(
            x >= join,
            1 - a * x * x,
            jnp.where(x > cutoff, curvature * (x - cutoff) ** 2, 0.0),
        ),
    )


def screened_value(r, parameters):
    """DNA2 Eq.13--16; [energy*length prefactor, screening length]."""
    prefactor, screening_length = parameters
    join = 3 * screening_length
    value = jnp.exp(-3.0) / join
    derivative = -value * (1 / screening_length + 1 / join)
    cutoff = join - 2 * value / derivative
    curvature = derivative**2 / (4 * value)
    central = jnp.exp(-r / screening_length) / r
    return prefactor * jnp.where(
        r < join, central, jnp.where(r < cutoff, curvature * (r - cutoff) ** 2, 0.0)
    )


def interaction_energy(
    positions, pairs, profile, *, bonded, model, strengths, charge_scale, image_shift
):
    """Full published pair decomposition for one chemistry and neighbor class.

    Site slots: backbone, base/HB, stack3, stack5, coax, frame-origin,
    frame-a1, frame-a3. The last three carry the exact orientation differential
    through the same marker adjoint; they are not extra material/charge sites.
    RNA angle labels follow its primary Appendix A, DNA its Appendix A figure.
    """
    ia, ib = pairs[:, 0], pairs[:, 1]
    a, b = positions[ia], positions[ib] + image_shift[:, None, :]
    a1, a3 = _unit(a[:, 6] - a[:, 5]), _unit(a[:, 7] - a[:, 5])
    b1, b3 = _unit(b[:, 6] - b[:, 5]), _unit(b[:, 7] - b[:, 5])
    a2, b2 = jnp.cross(a3, a1), jnp.cross(b3, b1)

    def displacement(i, j):
        d = b[:, j] - a[:, i]
        squared = _dot(d, d)
        positive = squared > 0
        length = jnp.sqrt(jnp.where(positive, squared, 1.0))
        # Compact attractive support excludes coincident sites. Define the
        # otherwise irrelevant direction and its tangent without 0/0. Real
        # excluded-volume/backbone singularities retain their invalid energy.
        return jnp.where(positive, length, 0.0), d / length[:, None]

    rb, db = displacement(0, 0)
    rh, dh = displacement(1, 1)
    # Published DNA uses i -> j in the 3'->5' direction. The host compiler
    # reverses DNA bonded edges; RNA products use i -> j in the 5'->3' direction.
    rs, ds = displacement(2, 3)
    rx, dx = displacement(4, 4)
    theta1 = _acos(_dot(a1, b1) if model == "rna" else -_dot(a1, b1))
    theta2, theta3 = _acos(-_dot(b1, dh)), _acos(_dot(a1, dh))
    theta4 = _acos(_dot(a3, b3))
    theta7, theta8 = _acos(-_dot(b3, dh)), _acos(_dot(a3, dh))

    def f(theta, kind, name):
        return angular_value(theta, profile[kind]["angles"][name])

    def symmetric(theta, kind, name, period=jnp.pi):
        return f(theta, kind, name) + f(period - theta, kind, name)

    ev = profile["excluded"]
    result = excluded_value(rh, ev["base-base"])
    result += excluded_value(displacement(0, 1)[0], ev["back-base"])
    result += excluded_value(displacement(1, 0)[0], ev["back-base"])
    if bonded:
        eps, r0, extension = profile["backbone"]
        argument = 1 - ((rb - r0) / extension) ** 2
        result += -0.5 * eps * jnp.log(argument)
        stack = profile["stacking"]
        if model == "rna":
            theta5, theta6 = _acos(_dot(a1, ds)), _acos(-_dot(b3, ds))
            p3 = profile["p3"][0] * a1 + profile["p3"][1] * a2 + profile["p3"][2] * a3
            q5 = profile["p5"][0] * b1 + profile["p5"][1] * b2 + profile["p5"][2] * b3
            angular = f(theta5, "stacking", "5") * f(theta6, "stacking", "6")
            angular *= f(_acos(-_dot(p3, db)), "stacking", "9") * f(
                _acos(-_dot(q5, db)), "stacking", "10"
            )
            phi1, phi2 = _dot(db, a2), _dot(db, b2)
        else:
            angular = (
                f(theta4, "stacking", "4")
                * f(_acos(_dot(a3, ds)), "stacking", "5")
                * f(_acos(_dot(b3, ds)), "stacking", "6")
            )
            phi1, phi2 = _dot(jnp.cross(db, a1), a3), _dot(jnp.cross(db, b1), b3)
        angular *= helicity_value(phi1, stack["helicity"][0]) * helicity_value(
            phi2, stack["helicity"][1]
        )
        result += strengths[:, 0] * radial_value(rs, stack["radial"], "morse") * angular
        return result
    result += excluded_value(rb, ev["back-back"])
    hb = profile["hydrogen-bond"]
    angular = (
        f(theta1, "hydrogen-bond", "1")
        * f(theta2, "hydrogen-bond", "2")
        * f(theta3, "hydrogen-bond", "3")
    )
    angular *= (
        f(theta4, "hydrogen-bond", "4")
        * f(theta7, "hydrogen-bond", "7")
        * f(theta8, "hydrogen-bond", "8")
    )
    result += strengths[:, 1] * radial_value(rh, hb["radial"], "morse") * angular
    cross = profile["cross-stacking"]
    angular = (
        f(theta1, "cross-stacking", "1")
        * f(theta2, "cross-stacking", "2")
        * f(theta3, "cross-stacking", "3")
    )
    angular *= symmetric(theta7, "cross-stacking", "7") * symmetric(
        theta8, "cross-stacking", "8"
    )
    if model != "rna":
        angular *= symmetric(theta4, "cross-stacking", "4")
    result += radial_value(rh, cross["radial"], "harmonic") * angular
    coax = profile["coaxial-stacking"]
    theta5, theta6 = _acos(_dot(a3, dx)), _acos(-_dot(b3, dx))
    angular = (
        f(theta4, "coaxial-stacking", "4")
        * symmetric(theta5, "coaxial-stacking", "5")
        * symmetric(theta6, "coaxial-stacking", "6")
    )
    if model == "dna2":
        A, B = coax["f6"]
        angular *= (
            f(theta1, "coaxial-stacking", "1") + 0.5 * A * jnp.maximum(theta1 - B, 0) ** 2
        )
    else:
        angular *= symmetric(theta1, "coaxial-stacking", "1", 2 * jnp.pi)
        phi3, phi4 = _dot(dx, jnp.cross(db, a1)), _dot(dx, jnp.cross(db, b1))
        angular *= helicity_value(phi3, coax["helicity"][0]) * helicity_value(
            phi4, coax["helicity"][1]
        )
    result += radial_value(rx, coax["radial"], "harmonic") * angular
    if "screening" in profile:
        result += charge_scale * screened_value(rb, profile["screening"])
    return result

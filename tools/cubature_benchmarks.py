#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import math
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(callback, *, repeats: int = 5):
    started = time.perf_counter()
    first = callback()
    jax.block_until_ready(first)
    first_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    value = first
    for _ in range(repeats):
        value = callback()
        jax.block_until_ready(value)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats
    return float(value), first_ms, steady_ms


def _disk_records():
    domain = phx.domain.GeometryDomain(phx.geometry.Circle((0.0, 0.0), 1.0).compile())
    target = phx.integration.over(domain.component())
    legacy_plan = phx.integration.FixedQuadraturePlan(
        phx.integration.GaussLegendreRule(32)
    )
    native_rule = phx.integration.CubatureRule("disk", 6)
    native_plan = phx.integration.FixedQuadraturePlan(native_rule)
    legacy = phx.integration.materialize(target, legacy_plan)
    native = phx.integration.materialize(target, native_plan)
    legacy_value, legacy_first, legacy_steady = _timed(
        lambda: phx.integration.reduce(1.0, legacy).value.data
    )
    native_value, native_first, native_steady = _timed(
        lambda: phx.integration.reduce(1.0, native).value.data
    )
    return (
        {
            "case": "unit-disk-legacy-mask",
            "value": legacy_value,
            "absolute_error": abs(legacy_value - math.pi),
            "evaluations": int(legacy.batch.total_weight().data.size),
            "first_ms": legacy_first,
            "steady_ms": legacy_steady,
        },
        {
            "case": "unit-disk-native",
            "value": native_value,
            "absolute_error": abs(native_value - math.pi),
            "evaluations": native_rule.num_points,
            "first_ms": native_first,
            "steady_ms": native_steady,
        },
    )


def _sphere_records():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    )
    target = phx.integration.over(domain.component({"x": phx.domain.Boundary()}))
    legacy = phx.integration.materialize(
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )
    native_rule = phx.integration.CubatureRule("sphere", 3)
    native = phx.integration.materialize(
        target,
        phx.integration.FixedQuadraturePlan(native_rule),
    )

    def moments(realization):
        return jnp.stack(
            tuple(
                phx.integration.reduce(
                    domain.Function("x")(lambda x, axis=axis: x[axis] ** 2),
                    realization,
                ).value.data
                for axis in range(3)
            )
        )

    legacy_moments = moments(legacy)
    native_moments = moments(native)
    expected = 4.0 * math.pi / 3.0
    return (
        {
            "case": "unit-sphere-legacy-chart",
            "moments": [float(value) for value in legacy_moments],
            "maximum_error": float(jnp.max(jnp.abs(legacy_moments - expected))),
            "anisotropy": float(jnp.max(legacy_moments) - jnp.min(legacy_moments)),
            "evaluations": int(legacy.batch.weights.data.size),
        },
        {
            "case": "unit-sphere-native",
            "moments": [float(value) for value in native_moments],
            "maximum_error": float(jnp.max(jnp.abs(native_moments - expected))),
            "anisotropy": float(jnp.max(native_moments) - jnp.min(native_moments)),
            "evaluations": native_rule.num_points,
        },
    )


def _normal_records():
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    function = normal.Function("z")(lambda z: z**20)
    target = phx.integration.expectation(normal)
    expected = float(math.prod(range(1, 20, 2)))
    legacy = phx.integration.integrate(
        function,
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(40)),
    )
    native = phx.integration.integrate(
        function,
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussHermiteRule(11)),
    )
    return (
        {
            "case": "normal-moment-quantile-legendre",
            "value": float(legacy.value.data),
            "relative_error": abs(float(legacy.value.data) - expected) / expected,
            "evaluations": int(legacy.num_evaluations),
        },
        {
            "case": "normal-moment-gauss-hermite",
            "value": float(native.value.data),
            "relative_error": abs(float(native.value.data) - expected) / expected,
            "evaluations": int(native.num_evaluations),
        },
    )


def main() -> None:
    triangle = phx.integration.CubatureRule("triangle", 10)
    tetrahedron = phx.integration.CubatureRule("tetrahedron", 10)
    records = [*_disk_records(), *_sphere_records(), *_normal_records()]
    records.extend(
        (
            {
                "case": "triangle-degree-10",
                "family": triangle.family,
                "evaluations": triangle.num_points,
                "duffy_evaluations": 36,
            },
            {
                "case": "tetrahedron-degree-10",
                "family": tetrahedron.family,
                "evaluations": tetrahedron.num_points,
                "duffy_evaluations": 343,
            },
        )
    )
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import logical_array_bytes, synchronize
from phydrax.special._spherical_bessel import (
    _spherical_hankel1_sequence,
    _spherical_i_sequence,
    _spherical_j_sequence,
    _spherical_k_sequence,
    _spherical_sequence_derivative,
    _spherical_y_sequence,
)


def _checksum(tree: Any, /) -> float:
    return sum(
        float(jnp.sum(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _maximum_absolute_error(actual: Any, reference: Any, /) -> float:
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    reference_leaves, reference_structure = jax.tree_util.tree_flatten(reference)
    if actual_structure != reference_structure:
        raise ValueError("Benchmark result and reference must have the same structure.")
    return max(
        (
            float(jnp.max(jnp.abs(actual_leaf - reference_leaf)))
            for actual_leaf, reference_leaf in zip(
                actual_leaves, reference_leaves, strict=True
            )
        ),
        default=0.0,
    )


def _gegenbauer_references(
    degree: int, alpha: jax.Array, values: jax.Array, /
) -> tuple[jax.Array, jax.Array]:
    alpha, values = jnp.broadcast_arrays(alpha, values)
    polynomials = [jnp.ones_like(values)]
    derivatives = [jnp.zeros_like(values)]
    if degree == 0:
        return jnp.stack(polynomials, axis=-1), jnp.stack(derivatives, axis=-1)
    polynomials.append(2.0 * alpha * values)
    derivatives.append(2.0 * values)
    for current_degree in range(2, degree + 1):
        scale = float(current_degree)
        previous_scale = current_degree + alpha - 1.0
        trailing_scale = current_degree + 2.0 * alpha - 2.0
        polynomials.append(
            (
                2.0 * previous_scale * values * polynomials[-1]
                - trailing_scale * polynomials[-2]
            )
            / scale
        )
        derivatives.append(
            (
                2.0 * values * polynomials[-2]
                + 2.0 * previous_scale * values * derivatives[-1]
                - 2.0 * polynomials[-3]
                - trailing_scale * derivatives[-2]
            )
            / scale
        )
    return jnp.stack(polynomials, axis=-1), jnp.stack(derivatives, axis=-1)


def _polylog_series(order: int, values: jax.Array, /, *, terms: int = 96) -> jax.Array:
    index = jnp.arange(1, terms + 1, dtype=jnp.real(values).dtype)
    return jnp.sum(values[..., None] ** index / index**order, axis=-1)


def _spherical_bessel_references(
    maximum_order: int,
    ordinary_argument: jax.Array,
    modified_argument: jax.Array,
    /,
) -> dict[str, jax.Array]:
    ordinary = ordinary_argument
    j_values = [
        jnp.sin(ordinary) / ordinary,
        jnp.sin(ordinary) / ordinary**2 - jnp.cos(ordinary) / ordinary,
    ]
    y_values = [
        -jnp.cos(ordinary) / ordinary,
        -jnp.cos(ordinary) / ordinary**2 - jnp.sin(ordinary) / ordinary,
    ]
    for order in range(1, maximum_order):
        recurrence_scale = (2 * order + 1) / ordinary
        j_values.append(recurrence_scale * j_values[-1] - j_values[-2])
        y_values.append(recurrence_scale * y_values[-1] - y_values[-2])
    j_sequence = jnp.stack(j_values[: maximum_order + 1])
    y_sequence = jnp.stack(y_values[: maximum_order + 1])
    hankel_sequence = j_sequence + 1.0j * y_sequence

    modified = modified_argument
    i_values = [
        jnp.sinh(modified) / modified,
        (modified * jnp.cosh(modified) - jnp.sinh(modified)) / modified**2,
    ]
    k_scale = 0.5 * jnp.pi * jnp.exp(-modified) / modified
    k_values = [k_scale, k_scale * (1.0 + 1.0 / modified)]
    for order in range(1, maximum_order):
        recurrence_scale = (2 * order + 1) / modified
        i_values.append(i_values[-2] - recurrence_scale * i_values[-1])
        k_values.append(k_values[-2] + recurrence_scale * k_values[-1])
    i_sequence = jnp.stack(i_values[: maximum_order + 1])
    k_sequence = jnp.stack(k_values[: maximum_order + 1])
    return {
        "h1": hankel_sequence,
        "h1_scaled": jnp.exp(-1.0j * ordinary)[None, :] * hankel_sequence,
        "i": i_sequence,
        "i_scaled": jnp.exp(-modified)[None, :] * i_sequence,
        "j": j_sequence,
        "k": k_sequence,
        "k_scaled": jnp.exp(modified)[None, :] * k_sequence,
        "y": y_sequence,
    }


def _spherical_bessel_derivative_references(
    values: dict[str, jax.Array],
    ordinary_argument: jax.Array,
    modified_argument: jax.Array,
    /,
) -> dict[str, jax.Array]:
    def derivative(sequence, argument, first, sign):
        orders = jnp.arange(1, sequence.shape[0], dtype=argument.dtype)[:, None]
        tail = sign * sequence[:-1] - (orders + 1.0) / argument[None, :] * sequence[1:]
        return jnp.concatenate((first[None, :], tail), axis=0)

    j_derivative = derivative(values["j"], ordinary_argument, -values["j"][1], 1.0)
    y_derivative = derivative(values["y"], ordinary_argument, -values["y"][1], 1.0)
    hankel_derivative = derivative(values["h1"], ordinary_argument, -values["h1"][1], 1.0)
    i_derivative = derivative(values["i"], modified_argument, values["i"][1], 1.0)
    k_derivative = derivative(values["k"], modified_argument, -values["k"][1], -1.0)
    return {
        "h1": hankel_derivative,
        "h1_scaled": jnp.exp(-1.0j * ordinary_argument)[None, :]
        * (hankel_derivative - 1.0j * values["h1"]),
        "i": i_derivative,
        "i_scaled": jnp.exp(-modified_argument)[None, :] * (i_derivative - values["i"]),
        "j": j_derivative,
        "k": k_derivative,
        "k_scaled": jnp.exp(modified_argument)[None, :] * (k_derivative + values["k"]),
        "y": y_derivative,
    }


def _benchmark(
    operation: Callable[..., Any],
    arguments: tuple[Any, ...],
    /,
    *,
    repeats: int,
    reference: Any | None = None,
) -> dict[str, float | int | None]:
    lowered = jax.jit(operation).lower(*arguments)
    started = time.perf_counter()
    executable = lowered.compile()
    compile_ms = 1e3 * (time.perf_counter() - started)

    result = executable(*arguments)
    synchronize(result)
    started = time.perf_counter()
    for _ in range(repeats):
        result = executable(*arguments)
        synchronize(result)
    execution_ms = 1e3 * (time.perf_counter() - started) / repeats
    memory = executable.memory_analysis()
    metrics: dict[str, float | int | None] = {
        "compile_ms": compile_ms,
        "execution_mean_ms": execution_ms,
        "output_bytes": logical_array_bytes(result),
        "checksum": _checksum(result),
        "retained_bytes": (
            None if memory is None else int(memory.generated_code_size_in_bytes)
        ),
        "workspace_bytes": None if memory is None else int(memory.temp_size_in_bytes),
    }
    if reference is not None:
        metrics["reference_max_abs_error"] = _maximum_absolute_error(result, reference)
    return metrics


def run_benchmarks(*, batch_sizes: Sequence[int], repeats: int) -> dict[str, Any]:
    """Benchmark compiled values and first derivatives for every public family."""
    harmonic_degree = 8
    harmonic_order = 3
    gegenbauer_degree = 12
    results: dict[str, Any] = {
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "dtype": "float64/complex128",
            "repeats": repeats,
            "spherical_harmonic_degree": harmonic_degree,
            "spherical_harmonic_order": harmonic_order,
            "spherical_legendre_derivative_direction": "unit polar",
            "spherical_harmonic_derivative_direction": "unit polar plus unit azimuth",
            "gegenbauer_degree": gegenbauer_degree,
        },
        "batches": {},
    }
    for batch_size in batch_sizes:
        x = jnp.linspace(-8.0, 8.0, batch_size)
        positive = jnp.geomspace(0.1, 30.0, batch_size)
        unit = jnp.linspace(0.1, 0.9, batch_size)
        order = jnp.linspace(0.0, 20.0, batch_size)
        z = jax.lax.complex(x, jnp.full_like(x, 0.5))
        gegenbauer_x = jnp.linspace(-0.8, 0.8, batch_size)
        gegenbauer_alpha = jnp.full_like(gegenbauer_x, 1.25)
        gegenbauer_values, gegenbauer_derivatives = _gegenbauer_references(
            gegenbauer_degree, gegenbauer_alpha, gegenbauer_x
        )
        theta = jnp.linspace(0.1, jnp.pi - 0.1, batch_size)
        phi = jnp.linspace(-jnp.pi, jnp.pi, batch_size)
        directions = jnp.stack(
            (
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ),
            axis=-1,
        )
        direction_tangents = jnp.stack(
            (
                jnp.cos(theta) * jnp.cos(phi) - jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta) * jnp.sin(phi) + jnp.sin(theta) * jnp.cos(phi),
                -jnp.sin(theta),
            ),
            axis=-1,
        )
        solid_points = directions * (1.0 + unit)[..., None]
        zeta_order = jnp.linspace(1.5, 6.0, batch_size)
        hurwitz_parameter = jnp.linspace(0.75, 2.5, batch_size)
        analytic_argument = 0.45 * jnp.exp(0.4j * phi)
        polylog_order = jnp.linspace(0.5, 4.0, batch_size)
        operations = {
            "faddeeva_values": (
                lambda real, complex_: (
                    phx.special.dawsn(real),
                    phx.special.wofz(complex_),
                    phx.special.voigt_profile(real, 0.8, 0.2),
                ),
                (x, z),
            ),
            "faddeeva_derivatives": (
                lambda real, complex_: (
                    jax.jvp(
                        phx.special.dawsn,
                        (real,),
                        (jnp.ones_like(real),),
                    )[1],
                    jax.jvp(
                        phx.special.wofz,
                        (complex_,),
                        (jnp.ones_like(complex_),),
                    )[1],
                    jax.jvp(
                        lambda values: phx.special.voigt_profile(values, 0.8, 0.2),
                        (real,),
                        (jnp.ones_like(real),),
                    )[1],
                ),
                (x, z),
            ),
            "carlson_values": (
                lambda values: (
                    phx.special.elliprc(values, values + 0.5),
                    phx.special.elliprf(values, values + 0.5, values + 1.0),
                    phx.special.elliprd(values, values + 0.5, values + 1.0),
                    phx.special.elliprj(values, values + 0.5, values + 1.0, values + 0.8),
                    phx.special.elliprg(values, values + 0.5, values + 1.0),
                ),
                (unit,),
            ),
            "carlson_derivatives": (
                lambda values: jax.jvp(
                    lambda arguments: (
                        phx.special.elliprc(arguments, arguments + 0.5),
                        phx.special.elliprf(arguments, arguments + 0.5, arguments + 1.0),
                        phx.special.elliprd(arguments, arguments + 0.5, arguments + 1.0),
                        phx.special.elliprj(
                            arguments,
                            arguments + 0.5,
                            arguments + 1.0,
                            arguments + 0.8,
                        ),
                        phx.special.elliprg(arguments, arguments + 0.5, arguments + 1.0),
                    ),
                    (values,),
                    (jnp.ones_like(values),),
                )[1],
                (unit,),
            ),
            "legendre_values": (
                lambda values, amplitude: (
                    phx.special.ellipk(values),
                    phx.special.ellipkm1(1.0 - values),
                    phx.special.ellipe(values),
                    phx.special.ellipkinc(amplitude, values),
                    phx.special.ellipeinc(amplitude, values),
                    phx.special.ellippi(0.2, values),
                    phx.special.ellippiinc(0.2, amplitude, values),
                ),
                (unit, 0.5 * x),
            ),
            "legendre_derivatives": (
                lambda values: jax.jvp(
                    lambda parameters: (
                        phx.special.ellipk(parameters),
                        phx.special.ellipe(parameters),
                        phx.special.ellipkinc(0.5, parameters),
                        phx.special.ellipeinc(0.5, parameters),
                        phx.special.ellippi(0.2, parameters),
                        phx.special.ellippiinc(0.2, 0.5, parameters),
                    ),
                    (values,),
                    (jnp.ones_like(values),),
                )[1],
                (unit,),
            ),
            "jacobi_values": (
                lambda amplitude, values: (
                    phx.special.ellipj(amplitude, values),
                    phx.special.ellipam(amplitude, values),
                ),
                (x, unit),
            ),
            "jacobi_derivatives": (
                lambda amplitude, values: jax.jvp(
                    phx.special.ellipj,
                    (amplitude, values),
                    (jnp.ones_like(amplitude), jnp.ones_like(values)),
                )[1],
                (x, unit),
            ),
            "airy_values": (
                lambda values: (
                    phx.special.airy(values),
                    phx.special.airye(values),
                ),
                (x,),
            ),
            "airy_derivatives": (
                lambda values: (
                    jax.jvp(
                        phx.special.airy,
                        (values,),
                        (jnp.ones_like(values),),
                    )[1],
                    jax.jvp(
                        phx.special.airye,
                        (values,),
                        (jnp.ones_like(values),),
                    )[1],
                ),
                (x,),
            ),
            "modified_bessel_values": (
                lambda orders, arguments: (
                    phx.special.iv(orders, arguments),
                    phx.special.ive(orders, arguments),
                    phx.special.kv(orders, arguments),
                    phx.special.kve(orders, arguments),
                ),
                (order, positive),
            ),
            "modified_bessel_derivatives": (
                lambda orders, arguments: jax.jvp(
                    lambda values: (
                        phx.special.iv(orders, values),
                        phx.special.ive(orders, values),
                        phx.special.kv(orders, values),
                        phx.special.kve(orders, values),
                    ),
                    (arguments,),
                    (jnp.ones_like(arguments),),
                )[1],
                (order, positive),
            ),
            "cylindrical_bessel_values": (
                lambda orders, arguments: (
                    phx.special.jv(orders, arguments),
                    phx.special.yv(orders, arguments),
                    phx.special.hankel1(orders, arguments),
                    phx.special.hankel2(orders, arguments),
                ),
                (order, positive),
            ),
            "cylindrical_bessel_derivatives": (
                lambda orders, arguments: jax.jvp(
                    lambda values: (
                        phx.special.jv(orders, values),
                        phx.special.yv(orders, values),
                        phx.special.hankel1(orders, values),
                        phx.special.hankel2(orders, values),
                    ),
                    (arguments,),
                    (jnp.ones_like(arguments),),
                )[1],
                (order, positive),
            ),
            "spherical_legendre_values": (
                lambda polar: phx.special.sph_legendre_p(
                    harmonic_degree, harmonic_order, polar
                ),
                (theta,),
            ),
            "spherical_legendre_derivatives": (
                lambda polar: jax.jvp(
                    lambda values: phx.special.sph_legendre_p(
                        harmonic_degree, harmonic_order, values
                    ),
                    (polar,),
                    (jnp.ones_like(polar),),
                )[1],
                (theta,),
            ),
            "spherical_harmonic_angular_values": (
                lambda polar, azimuth: phx.special.sph_harm_y(
                    harmonic_degree, harmonic_order, polar, azimuth
                ),
                (theta, phi),
            ),
            "spherical_harmonic_angular_derivatives": (
                lambda polar, azimuth: jax.jvp(
                    lambda polar_, azimuth_: phx.special.sph_harm_y(
                        harmonic_degree, harmonic_order, polar_, azimuth_
                    ),
                    (polar, azimuth),
                    (jnp.ones_like(polar), jnp.ones_like(azimuth)),
                )[1],
                (theta, phi),
            ),
            "spherical_harmonic_cartesian_values": (
                lambda vectors: phx.special.sph_harm_y_cart(
                    harmonic_degree, harmonic_order, vectors
                ),
                (directions,),
            ),
            "spherical_harmonic_cartesian_derivatives": (
                lambda vectors, tangents: jax.jvp(
                    lambda values: phx.special.sph_harm_y_cart(
                        harmonic_degree, harmonic_order, values
                    ),
                    (vectors,),
                    (tangents,),
                )[1],
                (directions, direction_tangents),
            ),
            "solid_harmonic_values": (
                lambda vectors: (
                    phx.special.solid_harmonic_regular(6, 2, vectors),
                    phx.special.solid_harmonic_irregular(6, 2, vectors),
                ),
                (solid_points,),
            ),
            "solid_harmonic_derivatives": (
                lambda vectors, tangents: jax.jvp(
                    lambda values: (
                        phx.special.solid_harmonic_regular(6, 2, values),
                        phx.special.solid_harmonic_irregular(6, 2, values),
                    ),
                    (vectors,),
                    (tangents,),
                )[1],
                (solid_points, direction_tangents),
            ),
            "zeta_values": (
                lambda orders, parameters: (
                    phx.special.zeta(orders),
                    phx.special.hurwitz_zeta(orders, parameters),
                ),
                (zeta_order, hurwitz_parameter),
            ),
            "zeta_derivatives": (
                lambda orders, parameters: jax.jvp(
                    lambda order_, parameter_: (
                        phx.special.zeta(order_),
                        phx.special.hurwitz_zeta(order_, parameter_),
                    ),
                    (orders, parameters),
                    (jnp.ones_like(orders), jnp.ones_like(parameters)),
                )[1],
                (zeta_order, hurwitz_parameter),
            ),
            "dilog_polylog_values": (
                lambda orders, arguments: (
                    phx.special.dilog(arguments),
                    phx.special.spence(arguments),
                    phx.special.polylog(orders, arguments),
                ),
                (polylog_order, analytic_argument),
            ),
            "dilog_polylog_derivatives": (
                lambda orders, arguments: jax.jvp(
                    lambda order_, argument_: (
                        phx.special.dilog(argument_),
                        phx.special.polylog(order_, argument_),
                    ),
                    (orders, arguments),
                    (jnp.ones_like(orders), jnp.ones_like(arguments)),
                )[1],
                (polylog_order, analytic_argument),
            ),
            "spherical_bessel_values": (
                lambda real, complex_: (
                    _spherical_j_sequence(12, real),
                    _spherical_y_sequence(12, real),
                    _spherical_hankel1_sequence(12, complex_, scaled=True),
                    _spherical_i_sequence(12, real, scaled=True),
                    _spherical_k_sequence(12, real, scaled=True),
                ),
                (positive, z),
            ),
            "spherical_bessel_derivatives": (
                lambda real: (
                    _spherical_sequence_derivative(
                        _spherical_j_sequence(12, real), real, kind="j"
                    ),
                    _spherical_sequence_derivative(
                        _spherical_i_sequence(12, real, scaled=True),
                        real,
                        kind="i",
                        scaled=True,
                    ),
                    _spherical_sequence_derivative(
                        _spherical_k_sequence(12, real, scaled=True),
                        real,
                        kind="k",
                        scaled=True,
                    ),
                ),
                (positive,),
            ),
        }
        batch_results = {
            name: _benchmark(operation, arguments, repeats=repeats)
            for name, (operation, arguments) in operations.items()
        }
        batch_results["gegenbauer_values"] = _benchmark(
            lambda alpha, values: (
                phx.special.gegenbauer_c(gegenbauer_degree, alpha, values),
                phx.special.gegenbauer_vander(alpha, values, gegenbauer_degree),
            ),
            (gegenbauer_alpha, gegenbauer_x),
            repeats=repeats,
            reference=(
                gegenbauer_values[..., gegenbauer_degree],
                gegenbauer_values,
            ),
        )
        batch_results["gegenbauer_alpha_derivative"] = _benchmark(
            lambda alpha, values: phx.special.gegenbauer_alpha_derivative(
                gegenbauer_degree, alpha, values
            ),
            (gegenbauer_alpha, gegenbauer_x),
            repeats=repeats,
            reference=gegenbauer_derivatives[..., gegenbauer_degree],
        )
        results["batches"][str(batch_size)] = batch_results
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark JAX compilation and execution of phydrax.special."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 1_024, 65_536],
    )
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if min(*args.batch_sizes, args.repeats) <= 0:
        parser.error("batch sizes and repeats must be positive")
    print(
        json.dumps(
            run_benchmarks(batch_sizes=args.batch_sizes, repeats=args.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.equations._dark_radiation_moments import (
    CosmologicalMultigroupM1System,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=131072)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.cells <= 0
        or arguments.groups <= 0
        or arguments.repetitions <= 0
        or arguments.shards <= 0
    ):
        raise ValueError("Cell, group, repetition, and shard counts must be positive.")
    if arguments.shards != 1:
        raise ValueError(
            "this benchmark has no distributed execution path; --shards must be one"
        )

    edges = jnp.geomspace(1.0, 1.0e3, arguments.groups + 1)
    system = CosmologicalMultigroupM1System(
        edges,
        3,
        physical_light_speed=1.0,
        reduced_light_speed=0.2,
        beam_risk_limit=0.2,
    )
    coordinate = jnp.linspace(0.0, 2.0 * jnp.pi, arguments.cells)
    energy = (
        1.0
        + 0.1
        * jnp.sin(coordinate)[:, None]
        * jnp.linspace(0.5, 1.0, arguments.groups)[None, :]
    )
    flux = jnp.zeros((arguments.cells, arguments.groups, 3))
    flux = flux.at[..., 0].set(0.05 * energy)
    state = jnp.concatenate((energy[..., None], flux), axis=-1).reshape(
        (arguments.cells, -1)
    )
    hubble = jnp.full((arguments.cells,), 1.0e-3)
    volumes = jnp.ones((arguments.cells,))
    correction = jnp.zeros_like(state)
    correction = correction.at[:, 0].set(1.0e-8 * jnp.cos(coordinate))

    @eqx.filter_jit
    def action(values):
        redshift = system.group_redshift_flux(values, hubble)
        provisional = values + 1.0e-2 * redshift.conservative_rate
        realizability = system.enforce_realizability(provisional)
        reflux = system.reflux(
            realizability.accepted_state,
            correction,
            volumes,
            topology_id="m1-benchmark-amr",
        )
        return redshift, realizability, reflux

    start = time.perf_counter()
    first = action(state)
    jax.block_until_ready(first)
    compile_and_first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(arguments.repetitions):
        result = action(state)
    jax.block_until_ready(result)
    execution = (time.perf_counter() - start) / arguments.repetitions
    redshift, realizability, reflux = result
    if not bool(
        jnp.all(redshift.accepted)
        & jnp.all(realizability.evidence.accepted)
        & jnp.all(reflux.accepted)
    ):
        raise RuntimeError("Multigroup M1 benchmark produced refused states.")
    payload = {
        "packets": 0,
        "cells": arguments.cells,
        "groups": arguments.groups,
        "collectives": 0,
        "compile_and_first_ms": 1000.0 * compile_and_first,
        "execution_ms": 1000.0 * execution,
        "cell_groups_per_second": arguments.cells * arguments.groups / execution,
        "frequency_conservation_residual": float(
            jnp.max(jnp.abs(redshift.conservation_residual))
        ),
        "reflux_conservation_residual": float(
            jnp.max(jnp.abs(reflux.conservation_residual))
        ),
        "realizability_correction_residual": float(
            jnp.max(realizability.evidence.correction_norm)
        ),
        "physical_light_speed": system.physical_light_speed,
        "reduced_light_speed": system.reduced_light_speed,
        "physical_c_reduced_c_identity": system.physical_c_reduced_c_identity,
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

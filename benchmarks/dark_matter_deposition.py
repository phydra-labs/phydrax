#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax.applications.cosmology._energy_deposition import (
    CascadeKernelProduct,
    InjectionSpectrum,
)


def _problem(
    redshift_nodes: int,
    energy_nodes: int,
    species_count: int,
    channel_count: int,
    /,
) -> tuple[CascadeKernelProduct, InjectionSpectrum]:
    one_plus_redshift = np.geomspace(1.0e4, 1.0, redshift_nodes)
    scale_factors = 1.0 / one_plus_redshift
    energy_gev = np.geomspace(1.0e-6, 1.0e3, energy_nodes)
    species = tuple(f"species-{index}" for index in range(species_count))
    channels = tuple(f"channel-{index}" for index in range(channel_count))
    channel_weights = np.arange(1.0, channel_count + 1.0)
    channel_weights = 0.75 * channel_weights / np.sum(channel_weights)
    deposited = (
        channel_weights[None, :, None, None]
        * energy_gev[None, None, None, :]
        * np.ones((redshift_nodes, 1, species_count, 1))
    )
    borrowed = (
        0.05 * energy_gev[None, None, :] * np.ones((redshift_nodes, species_count, 1))
    )
    escaped = (
        0.30 * energy_gev[None, None, :] * np.ones((redshift_nodes, species_count, 1))
    )
    state_values = np.stack(
        (
            np.linspace(0.0, 1.0, redshift_nodes),
            2.7255 / scale_factors,
        ),
        axis=-1,
    )
    kernel = CascadeKernelProduct(
        scale_factors,
        energy_gev,
        state_values,
        deposited,
        escaped,
        borrowed,
        species,
        channels,
        ("electron_fraction", "cmb_temperature_k"),
    )
    key = jax.random.PRNGKey(20260915)
    values = jax.random.uniform(
        key,
        (redshift_nodes, species_count, energy_nodes),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float64,
    )
    injection = InjectionSpectrum(
        one_plus_redshift,
        energy_gev,
        values,
        species,
        source_id="dark-matter-deposition-benchmark",
    )
    return kernel, injection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redshift-nodes", type=int, default=256)
    parser.add_argument("--energy-nodes", type=int, default=512)
    parser.add_argument("--species", type=int, default=3)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.redshift_nodes < 2
        or arguments.energy_nodes < 2
        or arguments.species < 1
        or arguments.channels < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "redshift/energy nodes must be at least two, species/channels/repeats "
            "positive, and warmup non-negative."
        )
    jax.config.update("jax_enable_x64", True)
    kernel, injection = _problem(
        arguments.redshift_nodes,
        arguments.energy_nodes,
        arguments.species,
        arguments.channels,
    )

    def native_table_action(spectrum):
        ledger = kernel.apply(spectrum)
        return (
            ledger.deposited_energy_gev,
            ledger.escaped_energy_gev,
            ledger.borrowed_cmb_energy_gev,
            ledger.evidence.closure_residual_gev,
        )

    compiled_function = jax.jit(native_table_action)
    compiled, compilation = measure_lower_and_compile(
        lambda: compiled_function.lower(injection),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(injection),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    deposited, escaped, borrowed, closure = result
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    stablehlo = str(compiled_function.lower(injection).compiler_ir(dialect="stablehlo"))
    median_seconds = execution.median_seconds
    table_entries = (
        arguments.redshift_nodes
        * arguments.energy_nodes
        * arguments.species
        * arguments.channels
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "redshift_nodes": arguments.redshift_nodes,
            "energy_nodes": arguments.energy_nodes,
            "species": arguments.species,
            "channels": arguments.channels,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "native_table_action": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "execution": execution.to_milliseconds_dict(),
            "table_entries_per_second": table_entries / median_seconds,
            "logical_input_bytes": logical_array_bytes((kernel, injection)),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_memory_bytes": {
                "arguments": compiler.argument_bytes,
                "outputs": compiler.output_bytes,
                "temporary": compiler.temporary_bytes,
                "generated_code": compiler.generated_code_bytes,
                "estimated_device": compiler.estimated_device_memory_bytes,
            },
            "compiler_flops": compiler.flops,
            "compiler_bytes_accessed": compiler.bytes_accessed,
            "stablehlo_dot_general_count": stablehlo.count("stablehlo.dot_general"),
            "host_callback_present": "xla_python_cpu_callback" in stablehlo,
        },
        "energy_closure": {
            "maximum_absolute_residual_gev": float(
                jnp.max(jnp.abs(closure), initial=0.0)
            ),
            "total_deposited_energy_gev": float(jnp.sum(deposited)),
            "total_escaped_energy_gev": float(jnp.sum(escaped)),
            "total_borrowed_cmb_energy_gev": float(jnp.sum(borrowed)),
            "kernel_maximum_relative_residual": float(
                kernel.evidence.maximum_relative_closure_residual
            ),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consolidated compile and synchronized execution evidence for bioinformatics kernels."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax.bioinformatics import (
    genomics,
    omics,
    phylogenetics,
    rna,
    sequence,
    spatial,
    spectrometry,
    systems,
)
from tools.bioinformatics_common_qualification import (
    emit_report,
    fingerprint,
    METHOD_CLAIM_TAXONOMY,
)


def _compiler_report(compiled: Any, /) -> dict[str, Any]:
    unavailable: list[str] = []
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list):
            if len(cost) == 1 and isinstance(cost[0], Mapping):
                cost = cost[0]
            else:
                cost = None
                unavailable.append("cost analysis returned multiple computations")
    except (AttributeError, NotImplementedError, RuntimeError) as error:
        cost = None
        unavailable.append(f"cost analysis unavailable: {type(error).__name__}")
    try:
        memory = compiled.memory_analysis()
    except (AttributeError, NotImplementedError, RuntimeError) as error:
        memory = None
        unavailable.append(f"memory analysis unavailable: {type(error).__name__}")
    if not isinstance(cost, Mapping) or not any(
        key in cost for key in ("flops", "bytes accessed")
    ):
        cost = None
        unavailable.append("compiler returned no supported cost analysis")
    if memory is None:
        unavailable.append("compiler returned no memory analysis")
    reason = "; ".join(dict.fromkeys(unavailable)) or None
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-compiled-executable",
        unavailable_reason=reason,
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "source": evidence.source,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _all_finite(value: Any, /) -> bool:
    leaves = jax.tree.leaves(value)
    arrays = [np.asarray(leaf) for leaf in leaves if hasattr(leaf, "dtype")]
    return bool(arrays) and all(np.all(np.isfinite(array)) for array in arrays)


def _benchmark_kernel(
    name: str,
    domain: str,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    /,
    *,
    method: Mapping[str, Any],
    inputs: Mapping[str, Any],
    captured_inputs: Any,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    jitted = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    first_result, first_execution_seconds = measure_synchronized(
        lambda: compiled(*arguments)
    )
    result, distribution = measure_repeated(
        lambda: compiled(*arguments),
        warmup=warmup,
        repeats=repeats,
    )
    synchronize(first_result)
    synchronize(result)
    method_fingerprint = fingerprint(method)
    input_fingerprint = fingerprint(inputs)
    finite = _all_finite(result)
    return {
        "name": name,
        "domain": domain,
        "method": {"fingerprint": method_fingerprint, **dict(method)},
        "method_fingerprint": method_fingerprint,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": fingerprint(result) if finite else None,
        "logical_input_bytes": logical_array_bytes(arguments),
        "logical_captured_input_bytes": logical_array_bytes(captured_inputs),
        "logical_output_bytes": logical_array_bytes(result),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_synchronized_execution_seconds": first_execution_seconds,
        "steady_synchronized_execution": distribution.to_seconds_dict(),
        "compiler": _compiler_report(compiled),
        "finite_output": finite,
        "passed": finite,
    }


def _sequence_kernel():
    query = jnp.asarray((0, 1, 2, 3, 0, 1, 2, 3), dtype=jnp.int32)
    target = jnp.asarray((0, 1, 3, 3, 0, 2, 2, 3), dtype=jnp.int32)
    scoring = sequence.identity_substitution_table(
        ("A", "C", "G", "T"), match_score=2.0, mismatch_score=-1.0
    )
    penalties = sequence.AffineGapPenalties(-2.0, -0.5)
    plan = sequence.AlignmentExecutionPlan.full(8, 8, traceback_capacity=16)

    def kernel(left, right):
        return sequence.align_affine(left, right, scoring, penalties, plan).score

    return (
        "affine_alignment",
        "sequence_dp",
        kernel,
        (query, target),
        {
            "public_symbol": "phydrax.bioinformatics.sequence.align_affine",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "almost_everywhere",
            "domain_scope": "full_global_alignment_lattice",
        },
        {
            "query": query,
            "target": target,
            "scoring_table_id": scoring.table_id,
            "plan_id": plan.plan_id,
        },
        (scoring, penalties, plan),
    )


def _phylogenetics_kernel():
    topology = phylogenetics.tree_topology(jnp.asarray((2, 2, -1), dtype=jnp.int32))
    model = phylogenetics.jc69()
    tip_partials = jnp.asarray(
        (
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
            ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ((0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        )
    )
    partition = phylogenetics.LikelihoodPartition(jnp.asarray((True, True, True)), model)
    lengths = jnp.asarray((0.1, 0.2, 0.0))

    def kernel(branch_lengths):
        return phylogenetics.felsenstein_pruning(
            topology,
            tip_partials,
            branch_lengths,
            (partition,),
        ).log_likelihood

    return (
        "felsenstein_pruning",
        "phylogenetics",
        kernel,
        (lengths,),
        {
            "public_symbol": "phydrax.bioinformatics.phylogenetics.felsenstein_pruning",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "fixed_tree_finite_state_likelihood",
        },
        {
            "parent_indices": topology.parent_indices,
            "tip_partials": tip_partials,
            "branch_lengths": lengths,
            "rate_matrix": model.rate_matrix,
        },
        (topology, model, partition, tip_partials),
    )


def _genomics_kernel():
    state_space = genomics.enumerate_genotype_states(2, 2, 3)
    read_mask = jnp.asarray((True, True, True, True))
    log_likelihoods = jnp.log(
        jnp.asarray(((0.99, 0.01), (0.80, 0.20), (0.25, 0.75), (0.10, 0.90)))
    )

    def kernel(values):
        evidence = genomics.local_haplotype_evidence(values, read_mask)
        return genomics.genotype_likelihoods_from_reads(
            evidence, state_space
        ).log_likelihoods

    return (
        "diploid_genotype_likelihood",
        "genomics_variant",
        kernel,
        (log_likelihoods,),
        {
            "public_symbol": "phydrax.bioinformatics.genomics.genotype_likelihoods_from_reads",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "complete_diploid_genotype_state_space",
        },
        {
            "allele_log_likelihoods": log_likelihoods,
            "read_mask": read_mask,
            "genotype_states": state_space.states,
        },
        (state_space, read_mask),
    )


def _omics_kernel():
    counts = jnp.arange(1, 65, dtype=jnp.float32)
    means = jnp.linspace(1.5, 70.0, 64)
    dispersion = jnp.asarray(0.2)

    def kernel(observed, expected):
        return omics.negative_binomial_log_likelihood(observed, expected, dispersion)

    return (
        "negative_binomial_log_likelihood",
        "omics_statistics",
        kernel,
        (counts, means),
        {
            "public_symbol": "phydrax.bioinformatics.omics.negative_binomial_log_likelihood",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "NB2_count_likelihood",
        },
        {"counts": counts, "means": means, "dispersion": dispersion},
        (dispersion,),
    )


def _rna_kernel():
    model = rna.nussinov_energy_model(
        pair_energy=-0.8,
        wobble_energy=-0.3,
        unpaired_energy=0.1,
        minimum_hairpin_length=1,
    )
    codes = jnp.asarray((0, 3, 1, 2, 0, 3, 2, 1), dtype=jnp.int32)

    def kernel(sequence_codes):
        return rna.rna_log_partition(sequence_codes, model)

    return (
        "rna_partition",
        "structure_rna",
        kernel,
        (codes,),
        {
            "public_symbol": "phydrax.bioinformatics.rna.rna_log_partition",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "all_noncrossing_partial_matchings",
        },
        {
            "sequence_codes": codes,
            "model_id": model.model_id,
            "pair_energies": model.pair_energies,
        },
        (model,),
    )


def _spatial_kernel():
    coordinates = jnp.arange(16, dtype=jnp.float32)[:, None]
    plan = spatial.SpatialNeighborPlan("radius", capacity=2, radius=1.1, weight="binary")
    graph = spatial.build_spatial_neighbor_graph(coordinates, plan)
    values = jnp.linspace(-1.0, 1.0, 16)

    def kernel(observations):
        return graph.lag(observations)

    return (
        "spatial_neighbor_lag",
        "spatial_spectrometry",
        kernel,
        (values,),
        {
            "public_symbol": "phydrax.bioinformatics.spatial.SpatialNeighborGraph.lag",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "fixed_capacity_radius_graph",
        },
        {
            "coordinates": coordinates,
            "values": values,
            "neighbor_capacity": graph.capacity,
            "neighbor_indices": graph.indices,
            "neighbor_mask": graph.mask,
        },
        (graph,),
    )


def _spectrometry_kernel():
    mass_to_charge = jnp.linspace(100.0, 199.0, 100)
    intensity = jnp.linspace(1.0, 10.0, 100)
    spectrum = spectrometry.MassSpectrum(mass_to_charge, intensity)
    plan = spectrometry.MassBinningPlan(jnp.linspace(99.5, 199.5, 21))

    def kernel(signal):
        varied = eqx.tree_at(lambda candidate: candidate.intensity, spectrum, signal)
        return spectrometry.bin_mass_spectrum(varied, plan).intensity

    return (
        "mass_spectrum_binning",
        "spatial_spectrometry",
        kernel,
        (intensity,),
        {
            "public_symbol": "phydrax.bioinformatics.spectrometry.bin_mass_spectrum",
            "method_kind": "exact_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "fixed_edge_mass_bins",
        },
        {
            "mass_to_charge": mass_to_charge,
            "intensity": intensity,
            "edges": plan.edges,
            "mass_to_charge_unit": int(spectrum.units.mass_to_charge),
        },
        (spectrum, plan),
    )


def _systems_kernel():
    reaction = systems.KineticReaction(
        0,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        jnp.asarray((2.0, 3.0)),
        rate_law=systems.RateLawKind.MICHAELIS_MENTEN,
        rate_unit=systems.SUBSTANCE_FLUX,
        kinetic_id="quick-benchmark",
    )
    concentrations = jnp.asarray((4.0,))

    def kernel(values):
        return reaction.evaluate(values)

    return (
        "michaelis_menten_rate",
        "systems",
        kernel,
        (concentrations,),
        {
            "public_symbol": "phydrax.bioinformatics.systems.KineticReaction.evaluate",
            "method_kind": "approximate_model",
            "execution_kind": "floating_point_direct",
            "differentiation_kind": "exact_ad",
            "domain_scope": "quasi_steady_state_rate_law",
        },
        {
            "concentrations": concentrations,
            "parameters": reaction.parameters,
            "rate_scale": reaction.rate_scale,
        },
        (reaction,),
    )


def run_benchmarks(*, warmup: int = 1, repeats: int = 5) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive.")
    builders = (
        _sequence_kernel,
        _phylogenetics_kernel,
        _genomics_kernel,
        _omics_kernel,
        _rna_kernel,
        _spatial_kernel,
        _spectrometry_kernel,
        _systems_kernel,
    )
    kernels: dict[str, dict[str, Any]] = {}
    for builder in builders:
        (
            name,
            domain,
            function,
            arguments,
            method,
            inputs,
            captured_inputs,
        ) = builder()
        kernels[name] = _benchmark_kernel(
            name,
            domain,
            function,
            arguments,
            method=method,
            inputs=inputs,
            captured_inputs=captured_inputs,
            warmup=warmup,
            repeats=repeats,
        )
    environment = capture_environment().to_dict()
    return {
        "benchmark": "bioinformatics_quick",
        "environment": environment,
        "environment_fingerprint": environment["fingerprint"],
        "configuration": {
            "warmup_executions": warmup,
            "steady_repeats": repeats,
            "synchronization": "all reachable JAX arrays blocked until ready",
        },
        "input_fingerprint": fingerprint(
            {name: result["input_fingerprint"] for name, result in kernels.items()}
        ),
        "method_fingerprint": fingerprint(
            {name: result["method_fingerprint"] for name, result in kernels.items()}
        ),
        "method_claim_taxonomy": METHOD_CLAIM_TAXONOMY,
        "execution_boundaries": {
            "host_interchange": (
                "Public host constructors and capacity plans are prepared before timing."
            ),
            "jax_kernel": (
                "Only lowered, compiled public array kernels contribute execution "
                "samples."
            ),
        },
        "kernels": kernels,
        "passed": all(bool(result["passed"]) for result in kernels.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run quick lowering, compilation, compiler-analysis, and synchronized "
            "execution evidence for representative public bioinformatics kernels."
        )
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmarks(
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    return emit_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())

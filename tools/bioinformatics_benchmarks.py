#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark representative native bioinformatics kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _synchronize(value):
    return jax.block_until_ready(value)


def _measure(function, *arguments, repeats: int):
    started = perf_counter()
    compiled = function.lower(*arguments).compile()
    compile_seconds = perf_counter() - started
    value = _synchronize(compiled(*arguments))
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        value = _synchronize(compiled(*arguments))
        samples.append(perf_counter() - started)
    return value, float(compile_seconds), float(median(samples))


def _source_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_benchmarks(*, sequence_length: int = 64, repeats: int = 3) -> dict:
    length = int(sequence_length)
    repeat_count = int(repeats)
    if length <= 0:
        raise ValueError("sequence_length must be positive.")
    if repeat_count <= 0:
        raise ValueError("repeats must be positive.")

    query = jnp.arange(length, dtype=jnp.int32) % 4
    target = jnp.roll(query, 1)
    table = phx.bioinformatics.sequence.nucleotide_substitution_table(
        match_score=2.0,
        mismatch_score=-3.0,
    )
    penalties = phx.bioinformatics.sequence.AffineGapPenalties(-4.0, -1.0)
    alignment_plan = phx.bioinformatics.sequence.AlignmentExecutionPlan.full(
        length,
        length,
        traceback_capacity=2 * length,
    )
    alignment_function = eqx.filter_jit(
        lambda left, right: phx.bioinformatics.sequence.align_affine(
            left,
            right,
            table,
            penalties,
            alignment_plan,
        )
    )
    alignment, alignment_compile, alignment_execute = _measure(
        alignment_function,
        query,
        target,
        repeats=repeat_count,
    )

    pattern_count = max(4, length // 2)
    first_states = jnp.arange(pattern_count, dtype=jnp.int32) % 4
    second_states = (first_states + 1) % 4
    partials = jnp.stack(
        (
            jax.nn.one_hot(first_states, 4),
            jax.nn.one_hot(second_states, 4),
        ),
        axis=1,
    )
    topology = phx.bioinformatics.phylogenetics.tree_topology(
        jnp.asarray((2, 2, -1), dtype=jnp.int32)
    )
    substitution = phx.bioinformatics.phylogenetics.jc69(dtype=jnp.float64)
    partition = phx.bioinformatics.phylogenetics.LikelihoodPartition(
        jnp.ones((pattern_count,), dtype=bool),
        substitution,
    )
    branch_lengths = jnp.asarray((0.2, 0.35, 0.0), dtype=jnp.float64)
    phylogenetic_function = eqx.filter_jit(
        lambda values, lengths: phx.bioinformatics.phylogenetics.felsenstein_pruning(
            topology,
            values,
            lengths,
            (partition,),
        )
    )
    phylogeny, phylogeny_compile, phylogeny_execute = _measure(
        phylogenetic_function,
        partials,
        branch_lengths,
        repeats=repeat_count,
    )

    counts = jnp.arange(length, dtype=jnp.float64) % 17
    means = 1.0 + jnp.arange(length, dtype=jnp.float64) % 11
    dispersion = jnp.asarray(0.2, dtype=jnp.float64)
    count_function = jax.jit(phx.bioinformatics.omics.negative_binomial_log_probability)
    count_log_probability, count_compile, count_execute = _measure(
        count_function,
        counts,
        means,
        dispersion,
        repeats=repeat_count,
    )

    records = {
        "alignment": {
            "compile_seconds": alignment_compile,
            "execution_seconds": alignment_execute,
            "length": length,
            "score": float(alignment.score),
            "valid": bool(alignment.valid),
            "method_contract_id": alignment.method_contract.contract_id,
        },
        "phylogenetics": {
            "compile_seconds": phylogeny_compile,
            "execution_seconds": phylogeny_execute,
            "pattern_count": pattern_count,
            "log_likelihood": float(phylogeny.log_likelihood),
            "valid": bool(phylogeny.valid),
            "method_contract_id": phylogeny.method_contract.contract_id,
        },
        "negative_binomial": {
            "compile_seconds": count_compile,
            "execution_seconds": count_execute,
            "element_count": length,
            "finite": bool(jnp.all(jnp.isfinite(count_log_probability))),
        },
    }
    passed = all(
        (
            records["alignment"]["valid"],
            records["phylogenetics"]["valid"],
            records["negative_binomial"]["finite"],
        )
    )
    return {
        "environment": {
            "jax_version": jax.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "input": {
            "repeats": repeat_count,
            "sequence_length": length,
        },
        "records": records,
        "source_sha256": _source_fingerprint(),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_benchmarks(
        sequence_length=arguments.sequence_length,
        repeats=arguments.repeats,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

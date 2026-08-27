#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._trainable import partition_trainable


def _state_bytes(state) -> int:
    return sum(
        int(entry.value.size * entry.value.dtype.itemsize) for entry in state.entries
    )


def _complex_trainable_leaves(value) -> int:
    trainable, _ = partition_trainable(value)
    return sum(
        int(jnp.iscomplexobj(leaf))
        for leaf in jax.tree.leaves(trainable)
        if isinstance(leaf, jax.Array)
    )


def _constrained_potential():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(4)
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        (
            phx.equations.HolomorphicPointFunctional.value(-1.0),
            phx.equations.HolomorphicPointFunctional.value(1.0),
        ),
    ).prepare()
    coefficient_map = operator.affine_map(jnp.asarray([0.2, -0.1]))
    return phx.equations.ConstrainedHolomorphicPotential(
        coefficient_map,
        initial_free_coordinates=jnp.linspace(-0.2, 0.3, coefficient_map.nullity),
    )


def run_complex_parameter_interchange_benchmarks() -> dict[str, Any]:
    model = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(16, 12),
        linear_ranks=(None, 6, 2),
        key=jr.key(0),
    )
    model_target = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(16, 12),
        linear_ranks=(None, 6, 2),
        key=jr.key(1),
    )
    coordinate = jnp.asarray([0.2 + 0.1j, -0.3 + 0.25j])
    started = time.perf_counter()
    model_state = phx.export.export_complex_parameters(model)
    export_seconds = time.perf_counter() - started
    started = time.perf_counter()
    restored_model = phx.export.import_complex_parameters(model_target, model_state)
    import_seconds = time.perf_counter() - started
    model_error = float(jnp.max(jnp.abs(model(coordinate) - restored_model(coordinate))))

    polynomial = phx.equations.HolomorphicPolynomialPotential(
        2,
        5,
        initial_scale=0.3,
        key=jr.key(2),
    )
    polynomial_target = phx.equations.HolomorphicPolynomialPotential(2, 5)
    polynomial_state = phx.export.export_complex_parameters(polynomial)
    restored_polynomial = phx.export.import_complex_parameters(
        polynomial_target,
        polynomial_state,
    )
    polynomial_error = float(
        jnp.max(jnp.abs(polynomial(0.15 - 0.2j) - restored_polynomial(0.15 - 0.2j)))
    )

    constrained = _constrained_potential()
    constrained_target = phx.equations.ConstrainedHolomorphicPotential(
        constrained.coefficient_map
    )
    constrained_state = phx.export.export_complex_parameters(constrained)
    restored_constrained = phx.export.import_complex_parameters(
        constrained_target,
        constrained_state,
    )
    constrained_error = float(
        jnp.linalg.norm(
            restored_constrained.coefficient_vector - constrained.coefficient_vector
        )
    )
    constrained_residual = float(
        jnp.linalg.norm(restored_constrained.constraint_residual())
    )

    poles = phx.equations.PoleSet(jnp.asarray([2.0 + 0.1j]), (2,))
    meromorphic_frame = phx.equations.MeromorphicLinearFrame(2, poles)
    meromorphic_operator = phx.equations.HolomorphicConstraintOperatorPlan(
        meromorphic_frame,
        (
            phx.equations.HolomorphicPointFunctional.value(-0.5),
            phx.equations.HolomorphicPointFunctional.value(0.5),
        ),
    ).prepare()
    meromorphic_map = meromorphic_operator.affine_map(jnp.asarray([0.0, 0.1]))
    meromorphic = phx.equations.ConstrainedMeromorphicPotential(
        meromorphic_map,
        initial_free_coordinates=jnp.linspace(-0.1, 0.2, meromorphic_map.nullity),
    )
    meromorphic_state = phx.export.export_complex_parameters(meromorphic)
    restored_meromorphic = phx.export.import_complex_parameters(
        phx.equations.ConstrainedMeromorphicPotential(meromorphic_map),
        meromorphic_state,
    )
    meromorphic_error = float(
        jnp.linalg.norm(
            restored_meromorphic.coefficient_vector - meromorphic.coefficient_vector
        )
    )

    state_bytes = sum(
        _state_bytes(state)
        for state in (
            model_state,
            polynomial_state,
            constrained_state,
            meromorphic_state,
        )
    )
    internal_complex_leaves = sum(
        _complex_trainable_leaves(value)
        for value in (
            restored_model,
            restored_polynomial,
            restored_constrained,
            restored_meromorphic,
        )
    )
    passed = bool(
        model_error == 0.0
        and polynomial_error == 0.0
        and constrained_error < 1e-11
        and constrained_residual < 1e-11
        and meromorphic_error < 1e-11
        and internal_complex_leaves == 0
    )
    payload = {
        "kind": "complex-parameter-interchange-benchmark",
        "passed": passed,
        "model_error": model_error,
        "polynomial_error": polynomial_error,
        "constrained_coefficient_error": constrained_error,
        "constrained_residual": constrained_residual,
        "meromorphic_coefficient_error": meromorphic_error,
        "internal_complex_trainable_leaves": internal_complex_leaves,
        "state_payload_bytes": state_bytes,
        "export_seconds": export_seconds,
        "import_seconds": import_seconds,
        "model_state_id": model_state.state_id,
    }
    payload["benchmark_id"] = canonical_fingerprint(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact mathematical complex parameter interchange."
    )
    parser.add_argument("--output", type=str, default=None)
    arguments = parser.parse_args()
    payload = run_complex_parameter_interchange_benchmarks()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()

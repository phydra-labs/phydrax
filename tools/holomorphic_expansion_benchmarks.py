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


def run_holomorphic_expansion_benchmarks() -> dict[str, Any]:
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(5)
    functionals = (
        phx.equations.HolomorphicPointFunctional.value(-1.0),
        phx.equations.HolomorphicPointFunctional.value(1.0),
        phx.equations.HolomorphicPointFunctional.normal_derivative(
            0.0,
            (1.0, 0.0),
        ),
    )
    started = time.perf_counter()
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        functionals,
    ).prepare()
    preparation_seconds = time.perf_counter() - started
    targets = jnp.stack(
        tuple(
            jnp.asarray([0.1 * index, -0.05 * index, 0.02 * index]) for index in range(32)
        )
    )
    started = time.perf_counter()
    coefficients = operator.minimum_norm_coefficients(targets)
    jax.block_until_ready(coefficients)
    batched_lift_seconds = time.perf_counter() - started
    target_residual = float(jnp.linalg.norm(operator.target_residual(targets)))

    projector_frame = phx.equations.HolomorphicPolynomialFrame.one_variable(1)
    projector_operator = phx.equations.HolomorphicConstraintOperatorPlan(
        projector_frame,
        functionals[:2],
    ).prepare()
    provider = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=1,
        hidden_sizes=(8,),
        key=jr.key(0),
    )
    projected = phx.equations.HolomorphicConstraintProjector(projector_operator).project(
        provider, jnp.asarray([0.4, -0.2])
    )
    projection_residual = float(
        jnp.max(
            jnp.abs(
                jnp.asarray(
                    [
                        jnp.real(projected(-1.0)[0]) - 0.4,
                        jnp.real(projected(1.0)[0]) + 0.2,
                    ]
                )
            )
        )
    )
    conditional_map = projector_operator.affine_map(jnp.asarray([0.4, -0.2]))
    conditional_trunk = phx.nn.operator.architectures.HolomorphicBasisTrunk(
        projector_frame,
        coefficient_map=conditional_map,
    )
    conditional_branch = phx.nn.models.MLP(
        in_size=3,
        out_size=conditional_trunk.latent_size,
        width_size=5,
        depth=1,
        key=jr.key(4),
    )
    conditional_deep = phx.nn.operator.architectures.DeepONet(
        branch=conditional_branch,
        trunk=conditional_trunk,
        coord_dim=2,
        latent_size=conditional_trunk.latent_size,
        use_bias=False,
    )
    conditional = phx.nn.operator.architectures.ConditionalHolomorphicDeepONet(
        conditional_deep
    )
    conditional_values = conditional(
        (
            jnp.asarray([0.1, -0.2, 0.3]),
            jnp.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        )
    )
    conditional_boundary_residual = float(
        jnp.max(jnp.abs(jnp.real(conditional_values) - jnp.asarray([0.4, -0.2])))
    )

    physical_frame = phx.equations.HolomorphicPolynomialFrame.one_variable(2, 2)
    physical_coefficients = jnp.linspace(
        -0.3,
        0.4,
        physical_frame.real_coefficient_count,
    )
    physical_coordinate = 0.2 - 0.1j
    physical_functional = phx.equations.plane_elasticity_stress_functional(
        physical_coordinate,
        "xx",
    )
    physical_value = (
        physical_functional.assemble_row(physical_frame) @ physical_coefficients
    )
    first_derivative = (
        physical_frame.basis_derivative(physical_coordinate, (1,)) @ physical_coefficients
    )
    second_derivative = (
        physical_frame.basis_derivative(physical_coordinate, (2,)) @ physical_coefficients
    )
    physical_expected = jnp.real(
        2.0 * first_derivative[0]
        - jnp.conj(physical_coordinate) * second_derivative[0]
        - first_derivative[1]
    )
    physical_functional_error = float(jnp.abs(physical_value - physical_expected))

    trace_plan = phx.equations.DiskHolomorphicTracePlan(4)
    cosine = jnp.asarray([0.2, 0.4, -0.1, 0.05, 0.03])
    sine = jnp.asarray([0.0, -0.2, 0.15, 0.04, -0.02])
    trace = trace_plan.lift(cosine, sine)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 257, endpoint=False)
    boundary = jnp.exp(1j * angles)
    expected = cosine[0] + sum(
        cosine[mode] * jnp.cos(mode * angles) + sine[mode] * jnp.sin(mode * angles)
        for mode in range(1, 5)
    )
    trace_error = float(
        jnp.max(
            jnp.abs(
                jax.vmap(lambda point: jnp.real(trace(point)[0]))(boundary) - expected
            )
        )
    )

    indices = phx.equations.HolomorphicMultiIndexSet.total_degree(2, 2)
    multivariable = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=1,
        hidden_sizes=(6,),
        key=jr.key(1),
    )
    coordinate = jnp.asarray([0.15 + 0.1j, -0.2 + 0.25j])
    started = time.perf_counter()
    multijet = multivariable.multi_jet(coordinate, indices)
    jax.block_until_ready(multijet.derivative((1, 1)))
    multijet_seconds = time.perf_counter() - started
    hessian = jax.jacfwd(
        jax.jacfwd(multivariable, holomorphic=True),
        holomorphic=True,
    )(coordinate)
    multijet_error = float(
        jnp.max(jnp.abs(multijet.derivative((1, 1)) - hessian[:, 0, 1]))
    )
    pluriharmonic = phx.equations.PluriharmonicPotential(multivariable)
    real_point = jnp.asarray([0.15, -0.2, 0.1, 0.25])
    laplacian_residual = float(jnp.abs(pluriharmonic.laplacian(real_point)))

    poles = phx.equations.PoleSet(jnp.asarray([2.0 + 0.1j]), (1,))
    meromorphic_frame = phx.equations.MeromorphicLinearFrame(2, poles)
    meromorphic_operator = phx.equations.HolomorphicConstraintOperatorPlan(
        meromorphic_frame,
        functionals[:2],
    ).prepare()
    meromorphic = phx.equations.ConstrainedMeromorphicPotential(
        meromorphic_operator.affine_map(jnp.zeros((2,)))
    )
    domain_certificate = meromorphic.certify_on_disk(
        center=0.0j,
        radius=1.0,
        required_clearance=0.2,
    )

    passed = bool(
        target_residual < 1e-10
        and projection_residual < 1e-10
        and trace_error < 1e-10
        and conditional_boundary_residual < 1e-10
        and physical_functional_error < 1e-10
        and multijet_error < 1e-9
        and laplacian_residual < 1e-9
        and bool(domain_certificate.certificate_id)
    )
    payload = {
        "kind": "holomorphic-expansion-benchmark",
        "passed": passed,
        "constraint_operator": {
            "rank": operator.evidence.rank,
            "nullity": operator.evidence.nullity,
            "target_residual": target_residual,
            "preparation_seconds": preparation_seconds,
            "batched_lift_seconds": batched_lift_seconds,
        },
        "projection_residual": projection_residual,
        "conditional_boundary_residual": conditional_boundary_residual,
        "physical_functional_error": physical_functional_error,
        "continuous_trace_error": trace_error,
        "continuous_trace_kind": trace.trace_certificate().evidence_kind,
        "multijet_error": multijet_error,
        "multijet_seconds": multijet_seconds,
        "pluriharmonic_laplacian_residual": laplacian_residual,
        "domain_holomorphic_certificate": domain_certificate.certificate_id,
    }
    payload["benchmark_id"] = canonical_fingerprint(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark generalized holomorphic constraints and geometry."
    )
    parser.add_argument("--output", type=str, default=None)
    arguments = parser.parse_args()
    payload = run_holomorphic_expansion_benchmarks()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()

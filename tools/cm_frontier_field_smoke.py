#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tiny end-to-end smoke for unreleased fermionic frontier candidates."""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications import (
    diagrammatic_field as df,
    functional_rg as frg,
    nonequilibrium_field as nef,
    sign_problem as sp,
)
from phydrax.operators.quantum._two_particle_green import (
    fermionic_crossing_evidence,
    FermionicTwoParticleChannelConvention,
    MatsubaraTwoParticleGreenFunction,
)


def _case(name, passed, **evidence):
    return {"case": name, "passed": bool(passed), **evidence}


def _vertex():
    labels = jnp.arange(-1, 2, dtype=jnp.int32)
    convention = FermionicTwoParticleChannelConvention("particle-hole-direct")
    route = convention.route(
        labels[:, None, None], labels[None, :, None], labels[None, None, :]
    )
    values = (route[..., 0] - route[..., 2]) * (route[..., 1] - route[..., 3])
    return MatsubaraTwoParticleGreenFunction(
        convention,
        4.0,
        labels,
        labels,
        labels,
        values[..., None, None, None, None].astype(complex),
        fermion_label_minimum=-1,
        fermion_label_count=3,
    )


def _diagram(order):
    field = df.FieldSpec(f"smoke-psi-{order}", statistics="fermion")
    propagator = df.PropagatorSpec(field, mass=1.0)
    rule = df.VertexRule(
        f"smoke-bilinear-{order}",
        (field, field),
        1.0,
        perturbative_order=order,
    )
    return df.DiagramGraph(
        (df.VertexInsertion("v", rule),),
        (df.PropagatorLine("loop", propagator, "v", "v", df.MomentumRoute([0.0])),),
    )


def run_smoke():
    cases = []
    vertex = _vertex()
    crossing = fermionic_crossing_evidence(vertex)
    cases.append(
        _case(
            "two-particle-routing-crossing",
            crossing.satisfied,
            coverage=float(crossing.coverage_fraction),
            maximum_residual=float(
                jnp.max(
                    jnp.asarray(
                        (
                            crossing.annihilation_exchange_residual,
                            crossing.creation_exchange_residual,
                            crossing.simultaneous_exchange_residual,
                        )
                    )
                )
            ),
        )
    )

    pi = np.pi
    patch = frg.FermiSurfacePatchRGPlan(
        jnp.asarray(((0.0, 0.0), (pi, 0.0), (0.0, pi), (pi, pi))),
        jnp.diag(jnp.asarray((2.0 * pi, 2.0 * pi))),
        jnp.asarray(((1.0, 0.2), (-1.0, 0.2), (0.2, 1.0), (0.2, -1.0))),
        jnp.ones(4),
        jnp.asarray((-1.0, -0.5, 0.5, 1.0)),
        jnp.ones(4) / 4.0,
        beta=4.0,
        matsubara_count=2,
    ).prepare()
    flow = patch.evaluate(patch.vertex(jnp.full((4, 4, 4), 0.1)), 0.5)
    cases.append(
        _case(
            "fermion-patch-frg",
            flow.evidence.admissible,
            routing_residual=float(flow.evidence.momentum_routing_residual),
            crossing_residual=float(flow.evidence.beta_outgoing_crossing_residual),
        )
    )

    parquet = (
        df.LatticeParquetPlan(tolerance=1e-12)
        .prepare(vertex, jnp.zeros((3,) + vertex.values.shape[:3]))
        .iterate()
    )
    cases.append(
        _case(
            "lattice-parquet",
            parquet.evidence.successful,
            residual=float(parquet.evidence.fixed_point_residual),
            parquet_identity=float(parquet.evidence.parquet_identity_residual),
        )
    )

    signs = jnp.asarray((1, -1), dtype=jnp.int32)

    def up_kernel(source, target, delta):
        return jnp.where(source == target, jnp.where(delta >= 0.0, 0.75, -0.25), 0.1)

    def down_kernel(source, target, delta):
        return -signs[source] * signs[target] * jnp.conj(up_kernel(source, target, delta))

    ctint = (
        df.SignFreeCTINTPlan(2.0, 0.5, signs, steps=32, maximum_order=2)
        .prepare(up_kernel, down_kernel, kernel_id="smoke-half-filled-two-site")
        .run(jr.key(1))
    )
    cases.append(
        _case(
            "sign-free-ct-int-control",
            ctint.evidence.successful,
            average_sign=float(ctint.evidence.average_sign),
            symmetry_residual=float(ctint.evidence.sign_free_symmetry_residual),
        )
    )

    diagram_mc = (
        df.LowOrderFermionDiagramMonteCarloPlan(
            steps=64, maximum_order=2, maximum_diagrams=2
        )
        .prepare(
            (_diagram(1), _diagram(2)),
            jnp.asarray((1.0 + 0.0j, -0.5 + 0.0j)),
            jnp.asarray(((0.5, 0.5), (0.5, 0.5))),
        )
        .run(jr.key(2))
    )
    cases.append(
        _case(
            "low-order-diagram-monte-carlo",
            diagram_mc.evidence.successful,
            detailed_balance=float(diagram_mc.evidence.detailed_balance_residual),
            order_ess=float(diagram_mc.evidence.order_effective_sample_size),
            average_sign=float(diagram_mc.evidence.average_sign),
        )
    )

    grid = nef.ClosedTimePathPlan(jnp.linspace(0.0, 0.2, 3)).prepare()
    free = nef.fermionic_keldysh_from_propagators(
        grid,
        jnp.broadcast_to(jnp.eye(1, dtype=complex), (3, 1, 1)),
        jnp.asarray([[0.5]]),
        source_id="smoke-stationary-level",
    )
    second_born = (
        nef.FermionicSecondBornPlan(
            grid,
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            maximum_iterations=4,
            tolerance=1e-10,
            conservation_tolerance=1e-10,
        )
        .prepare(free)
        .solve()
    )
    cases.append(
        _case(
            "fermionic-keldysh-second-born",
            second_born.evidence.successful,
            car_residual=float(second_born.evidence.car_residual),
            causality_residual=float(second_born.evidence.causality_residual),
            schwinger_dyson_residual=float(second_born.evidence.schwinger_dyson_residual),
        )
    )

    cancellation = jnp.tile(jnp.asarray((1.0 + 0.0j, -1.0 + 0.0j)), (1, 16))
    study = sp.evaluate_controlled_sign_study(
        sp.ControlledSignStudyPlan(
            minimum_draws=16,
            minimum_average_phase=0.1,
            minimum_effective_sample_size=2.0,
        ),
        cancellation,
        jnp.ones(cancellation.shape + (1,)),
        study_id="smoke-cancellation",
    )
    cases.append(
        _case(
            "controlled-sign-abstention",
            study.abstained
            & (study.status == int(sp.ControlledSignStudyStatus.PHASE_CANCELLATION)),
            average_sign=float(study.average_sign),
            effective_sample_size=float(study.effective_sample_size),
            status=int(study.status),
        )
    )
    return {
        "claim": "candidate smoke only; not qualification, production, or a generic sign cure",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def main():
    payload = run_smoke()
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

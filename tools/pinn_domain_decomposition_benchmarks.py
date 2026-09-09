#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax._trainable import partition_trainable


@dataclasses.dataclass(frozen=True, slots=True)
class PINNDomainDecompositionBenchmark:
    method: str
    scenario: str
    iterations: int
    patches: int
    parameters: int
    initial_loss: float
    final_loss: float
    relative_l2: float
    interface_defect: float
    elapsed_seconds: float


def _parameter_count(tree) -> int:
    trainable, _ = partition_trainable(tree)
    return sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if eqx.is_inexact_array(leaf)
    )


def _fixed_source(component, count: int, *, key):
    layout = phx.domain.SampleLayout((("x",),))
    batch = component.sample(
        phx.domain.PointSampling(count, layout=layout),
        key=key,
    )
    return phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )


def _poisson_terms(domain, field_name: str, *, points: int):
    interior = domain.component()
    boundary = domain.component({"x": phx.domain.Boundary()})
    forcing = domain.Function("x")(lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x[0]))
    target = domain.Function("x")(lambda x: jnp.sin(jnp.pi * x[0]))
    residual = phx.conditions.Residual(
        field_name,
        interior,
        lambda field: phx.operators.laplacian(field, var="x") + forcing,
    )
    boundary_condition = phx.conditions.Dirichlet(
        field_name,
        boundary,
        target=target,
    )
    return (
        phx.terms.ResidualPenalty(
            residual,
            _fixed_source(interior, points, key=jr.key(100)),
        ),
        phx.terms.ResidualPenalty(
            boundary_condition,
            _fixed_source(boundary, max(points // 4, 2), key=jr.key(101)),
            scale=10.0,
        ),
    )


def _relative_l2(field, domain, *, points: int = 512) -> float:
    coordinates = jnp.linspace(0.0, 1.0, points)
    batch = domain.component().points({"x": coordinates[:, None]})
    prediction = jnp.asarray(field(batch).data)
    reference = jnp.sin(jnp.pi * coordinates)
    error = jnp.linalg.norm(prediction - reference) / jnp.linalg.norm(reference)
    return float(jax.device_get(error))


def _run_global(
    domain,
    *,
    iterations: int,
    points: int,
    width: int,
    seed: int,
) -> PINNDomainDecompositionBenchmark:
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        width_size=width,
        depth=2,
        activation=jnp.tanh,
        rwf=False,
        key=jr.key(seed),
    )
    field = domain.Model("x")(model)
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=_poisson_terms(domain, "u", points=points),
    )
    initial = float(solver.loss(key=jr.key(seed + 1)))
    started = time.perf_counter()
    trained = solver.solve(
        num_iter=iterations,
        optim=optax.adam(1.0e-3),
        seed=seed,
        keep_best=False,
        log_every=0,
    )
    final = trained.loss(key=jr.key(seed + 2))
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    return PINNDomainDecompositionBenchmark(
        method="global-pinn",
        scenario="poisson-1d",
        iterations=iterations,
        patches=1,
        parameters=_parameter_count(trained.functions),
        initial_loss=initial,
        final_loss=float(final),
        relative_l2=_relative_l2(trained.functions["u"], domain),
        interface_defect=0.0,
        elapsed_seconds=elapsed,
    )


def _run_partition_of_unity(
    domain,
    *,
    iterations: int,
    points: int,
    width: int,
    patches: int,
    seed: int,
    training: str,
) -> PINNDomainDecompositionBenchmark:
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        patches,
        overlap_fraction=0.2,
    )
    keys = jr.split(jr.key(seed), patches)
    local_fields = {}
    for patch, key in zip(cover.patches, keys, strict=True):
        model = phx.nn.models.MLP(
            in_size=1,
            out_size="scalar",
            width_size=width,
            depth=2,
            activation=jnp.tanh,
            rwf=False,
            key=key,
        )
        local_fields[patch.patch_id] = patch.domain.Model("x")(model)
    family = phx.domain.LocalFieldFamily("u", cover, local_fields)
    problem = phx.solver.FunctionalDecompositionProblem.partition_of_unity(
        family,
        _poisson_terms(domain, "u", points=points),
    )
    if training == "joint":
        strategy = phx.solver.JointDecompositionTraining(iterations)
        method = "partition-of-unity-joint"
    elif training == "colored-block":
        strategy = phx.solver.BlockDecompositionTraining(
            iterations,
            1,
            sweep="colored",
        )
        method = "partition-of-unity-colored-block"
    else:
        raise ValueError("Unknown partition-of-unity training route.")
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            strategy,
            required_window_regularity=2,
        ),
    )
    initial = float(prepared.solver.loss(key=jr.key(seed + 1)))
    started = time.perf_counter()
    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.adam(1.0e-3),
        seed=seed,
        keep_best=False,
        log_every=0,
    )
    jax.block_until_ready(result.evidence.training_loss)
    elapsed = time.perf_counter() - started
    if result.global_field is None:
        raise RuntimeError("Partition-of-unity benchmark did not return a global field.")
    return PINNDomainDecompositionBenchmark(
        method=method,
        scenario="poisson-1d",
        iterations=iterations,
        patches=patches,
        parameters=_parameter_count(result.solver.functions),
        initial_loss=initial,
        final_loss=float(result.evidence.training_loss),
        relative_l2=_relative_l2(result.global_field, domain),
        elapsed_seconds=elapsed,
        interface_defect=0.0,
    )


def _run_schwarz(
    domain,
    *,
    iterations: int,
    points: int,
    seed: int,
) -> PINNDomainDecompositionBenchmark:
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 2)
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            cover.patch_ids[0]: cover.patches[0].domain.Parameter(1.0),
            cover.patch_ids[1]: cover.patches[1].domain.Parameter(-1.0),
        },
    )
    pairing = cover.pairings[0]
    condition = phx.conditions.SubdomainValueJump(
        family.ref(pairing.left_patch_id),
        family.ref(pairing.right_patch_id),
        pairing,
    )
    batch = pairing.component.sample(
        phx.domain.PointSampling(max(points // 4, 4)),
        key=jr.key(seed + 20),
    )
    source = phx.integration.fixed(
        phx.integration.from_samples(
            phx.integration.mean_over(pairing.component),
            batch,
        )
    )
    term = phx.solver.ScopedFunctionalTerm(
        phx.terms.ResidualPenalty(condition, source),
        phx.solver.PairScope(pairing.pairing_id),
    )
    problem = phx.solver.FunctionalDecompositionProblem.broken(family, term)
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(
                iterations,
                1,
                sweep="jacobi",
                relaxation=0.75,
            ),
            trace_points=max(points // 4, 4),
        ),
    )
    initial = float(prepared.solver.loss(key=jr.key(seed + 21)))
    started = time.perf_counter()
    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.sgd(0.1),
        seed=seed,
        jit=True,
    )
    elapsed = time.perf_counter() - started
    values = jnp.stack(
        tuple(jnp.asarray(field.func.value) for field in result.family.fields)
    )
    trace_state = result.state.trace_state
    if trace_state is None:
        raise RuntimeError("Schwarz benchmark did not retain trace state.")
    return PINNDomainDecompositionBenchmark(
        method="schwarz-jacobi",
        scenario="interface-equilibration",
        iterations=iterations,
        patches=2,
        parameters=_parameter_count(result.solver.functions),
        initial_loss=initial,
        final_loss=float(result.evidence.training_loss),
        relative_l2=float(jnp.linalg.norm(values) / jnp.sqrt(2.0)),
        interface_defect=float(trace_state.maximum_defect),
        elapsed_seconds=elapsed,
    )


def run_benchmarks(
    *,
    iterations: int = 20,
    points: int = 64,
    width: int = 16,
    patches: int = 2,
    seed: int = 0,
) -> tuple[PINNDomainDecompositionBenchmark, ...]:
    if iterations <= 0 or points < 4 or width <= 0 or patches < 2:
        raise ValueError("Benchmark work sizes are invalid.")
    domain = phx.domain.Interval1d(0.0, 1.0)
    return (
        _run_global(
            domain,
            iterations=iterations,
            points=points,
            width=width,
            seed=seed,
        ),
        _run_partition_of_unity(
            domain,
            iterations=iterations,
            points=points,
            width=max(width // patches, 2),
            patches=patches,
            seed=seed,
            training="joint",
        ),
        _run_partition_of_unity(
            domain,
            iterations=iterations,
            points=points,
            width=max(width // patches, 2),
            patches=patches,
            seed=seed,
            training="colored-block",
        ),
        _run_schwarz(
            domain,
            iterations=iterations,
            points=points,
            seed=seed,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark global, joint/block POU, and trace-state Schwarz routes."
        )
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--patches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    results = run_benchmarks(
        iterations=arguments.iterations,
        points=arguments.points,
        width=arguments.width,
        patches=arguments.patches,
        seed=arguments.seed,
    )
    payload = [dataclasses.asdict(result) for result in results]
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

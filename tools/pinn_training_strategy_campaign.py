#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax.solver._functional_objective import evaluate_prepared_objective


_STRATEGIES = ("baseline", "grad_norm", "ntk_trace", "pseudo", "combined")
_OPTIMIZERS = ("adam", "kfac", "soap")


def _problem(*, seed: int, samples: int, width: int, depth: int):
    domain = phx.domain.Interval1d(-1.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(width,) * depth,
        rwf=False,
        key=jr.key(seed),
    )
    field = domain.Model("x")(model)

    @domain.Function("x")
    def exact(x):
        return jnp.sin(jnp.pi * x[0])

    interior = domain.component()
    equation = phx.conditions.Residual(
        "u",
        interior,
        lambda value: phx.operators.laplacian(value, var="x")
        + jnp.pi**2 * exact,
        label="equation",
    )
    equation_term = phx.terms.ResidualPenalty(
        equation,
        phx.integration.per_step(
            phx.integration.mean_over(interior),
            phx.domain.PointSampling(
                samples,
                layout=phx.domain.SampleLayout((("x",),)),
                design="sobol_scrambled",
            ),
        ),
        label="equation",
    )
    boundary_component = domain.component({"x": phx.domain.Boundary()})
    boundary = phx.conditions.Residual(
        "u", boundary_component, lambda value: value, label="boundary"
    )
    boundary_batch = boundary_component.sample(
        phx.domain.PointSampling(
            max(4, samples // 8),
            layout=phx.domain.SampleLayout((("x",),)),
        ),
        key=jr.key(seed + 1),
    )
    boundary_term = phx.terms.ResidualPenalty(
        boundary,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(boundary_component), boundary_batch
            )
        ),
        scale=10.0,
        label="boundary",
    )
    evaluation_batch = interior.sample(
        phx.domain.PointSampling(
            257,
            layout=phx.domain.SampleLayout((("x",),)),
            design="sobol",
        ),
        key=jr.key(seed + 2),
    )
    evaluation_condition = phx.conditions.Observation(
        "u", interior, exact, label="solution"
    )
    evaluation_term = phx.terms.ObservationPenalty(
        evaluation_condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(interior), evaluation_batch
            )
        ),
        label="solution",
    )
    return phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(equation_term, boundary_term),
        evaluation_terms=(evaluation_term,),
    )


def _training_plan(strategy: str):
    balance = None
    pseudo = ()
    diagnostics = phx.solver.FunctionalDiagnosticsPolicy(
        every=10,
        gradient_alignment=True,
        ntk=False,
    )
    blocks = (
        phx.terms.ResidualBlockRef(0),
        phx.terms.ResidualBlockRef(1),
    )
    if strategy in ("grad_norm", "combined"):
        balance = phx.solver.FunctionalTermBalancePolicy(
            blocks,
            method="gradient_norm",
            start=2,
            every=10,
        )
    elif strategy == "ntk_trace":
        balance = phx.solver.FunctionalTermBalancePolicy(
            blocks,
            method="ntk_trace",
            start=2,
            every=10,
            ntk_probes=4,
            maximum_relative_standard_error=1.0,
        )
    if strategy in ("pseudo", "combined"):
        pseudo = (
            phx.solver.PseudoTransientPolicy(
                0,
                phx.solver.ResidualRelaxationMap("u", lambda value: value),
                adaptation=phx.solver.PseudoTransientAdaptation(
                    start=2,
                    every=10,
                ),
            ),
        )
    return phx.solver.FunctionalTrainingPlan(
        pseudo_transient=pseudo,
        term_balance=balance,
        diagnostics=diagnostics,
    )


def _optimizer(name: str):
    if name == "adam":
        return optax.adam(1e-3)
    if name == "kfac":
        return phx.optim.kfac(damping=1e-2, factor_update_period=10)
    if name == "soap":
        return phx.optim.soap(
            learning_rate=1e-3,
            b1=0.9,
            b2=0.999,
            precondition_frequency=2,
            max_preconditioner_size=10_000,
        )
    raise ValueError(f"Unknown optimizer {name!r}.")


def run(args):
    samples = 16 if args.smoke else args.samples
    width = 8 if args.smoke else args.width
    depth = 1 if args.smoke else args.depth
    steps = 2 if args.smoke else args.steps
    solver = _problem(seed=args.seed, samples=samples, width=width, depth=depth)
    prepared_evaluation = solver.objective.prepare_evaluation(
        key=jr.key(args.seed + 10),
        iteration=0,
    )
    initial_evaluation = float(
        evaluate_prepared_objective(prepared_evaluation, solver.functions).total
    )
    started = time.perf_counter()
    trained = solver.solve(
        num_iter=steps,
        optim=_optimizer(args.optimizer),
        keep_best=False,
        log_every=0,
        seed=args.seed,
        training=_training_plan(args.strategy),
    )
    jax.block_until_ready(trained.functions)
    elapsed = time.perf_counter() - started
    prepared_evaluation = trained.objective.prepare_evaluation(
        key=jr.key(args.seed + 10), iteration=steps
    )
    final_evaluation = float(
        evaluate_prepared_objective(prepared_evaluation, trained.functions).total
    )
    ntk = phx.solver.prepare_functional_ntk(
        trained,
        key=jr.key(args.seed + 20),
    )
    ntk_diagnostics = ntk.diagnostics(
        policy=phx.nn.neural_tangent.NTKDiagnosticsPolicy(
            dense_max_dimension=512,
            eigenvalue_count=4,
        ),
        key=jr.key(args.seed + 21),
    )
    alignment = float(
        trained.training_diagnostics.get(
            "gradient_alignment/intra",
            jnp.asarray(jnp.nan),
        )
    )
    return {
        "strategy": args.strategy,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "steps": steps,
        "samples": samples,
        "width": width,
        "depth": depth,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "training_seconds": elapsed,
        "ntk_trace": float(ntk_diagnostics.trace),
        "ntk_stable_rank": float(ntk_diagnostics.stable_rank),
        "ntk_effective_rank": float(ntk_diagnostics.effective_rank),
        "gradient_alignment_intra": (
            alignment if math.isfinite(alignment) else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=_STRATEGIES, default="baseline")
    parser.add_argument("--optimizer", choices=_OPTIMIZERS, default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()

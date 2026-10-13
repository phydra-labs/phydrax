"""Compiler-memory scaling of private open-loop game stationarity (small defaults)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import phydrax as phx
from phydrax.control.games._open_loop_kkt import (
    _kkt_operator,
    _KKTArguments,
    NonlinearOpenLoopGameProblem,
    plan_open_loop_game_kkt,
)

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


@runtime_checkable
class _Lowerable(Protocol):
    def __call__(
        self, state: tuple[Array, Array, Array], arguments: _KKTArguments
    ) -> tuple[Array, Array, Array]: ...
    def lower(
        self, state: tuple[Array, Array, Array], arguments: _KKTArguments
    ) -> Any: ...


def make_problem(horizon: int, players: int) -> NonlinearOpenLoopGameProblem:
    partition = phx.control.games.PlayerControlPartition(
        tuple(f"p{i}" for i in range(players)), (1,) * players
    )

    def transition(context: Any, state: Array, control: Array, args: Any) -> Array:
        del context, args
        return state + 0.01 * jnp.sum(control) + 0.001 * state**2

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((players,), roles="control"),
        system_id="memory-game-dynamics",
    )

    def cost_for(player: int) -> Any:
        def cost(context: Any, state: Array, control: Array, args: Any) -> Array:
            del context, args
            return control[player] ** 2 + 0.2 * state[0] ** 2

        return cost

    def terminal(time: Array, state: Array, args: Any) -> Array:
        del time, args
        return state[0] ** 2

    def constraint_for(player: int) -> Any:
        def residual(time: Array, state: Array, control: Array, args: Any) -> Array:
            del time, args
            return control[player] + 0.01 * state[0] ** 2 - 0.2

        name = f"p{player}"
        return phx.control.games.GameConstraintBlock(
            phx.control.BoundedPathConstraint(
                residual, lower=-jnp.inf, upper=0.0, constraint_id=f"memory-game:{name}"
            ),
            scope=phx.control.games.GameConstraintScope.PLAYER_LOCAL,
            participants=(name,),
            owner=name,
            site=phx.control.games.GameConstraintSite.PATH,
            equality=False,
            residual_shape=(),
            time_dependent=False,
            state_dependent=True,
            control_dependencies=(name,),
        )

    constraints = phx.control.games.OpenLoopGameConstraints(
        partition, tuple(constraint_for(i) for i in range(players))
    )
    return NonlinearOpenLoopGameProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.arange(horizon + 1, dtype=jnp.float64), time_id="memory-game-time"
        ),
        jnp.zeros((1,), dtype=jnp.float64),
        partition,
        stage_costs=tuple(cost_for(i) for i in range(players)),
        terminal_costs=(terminal,) * players,
        constraints=constraints,
        problem_id="memory-game",
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    parser.add_argument("--baseline-record", type=Path)
    options = parser.parse_args()
    cases: list[dict[str, Any]] = []
    for horizon in options.horizons:
        problem, host_peak = host_allocation_peak(
            lambda: make_problem(horizon, options.players)
        )
        plan = plan_open_loop_game_kkt(problem)
        arguments = _KKTArguments(plan, problem, None)
        state = (
            jnp.full((horizon * options.players,), 0.1, dtype=jnp.float64),
            jnp.zeros((max(plan.num_equalities, 1),), dtype=jnp.float64),
            jnp.zeros((max(plan.num_inequalities, 1),), dtype=jnp.float64),
        )
        entry = eqx.filter_jit(_kkt_operator)
        if not isinstance(entry, _Lowerable):
            raise TypeError("Game memory benchmark requires a lowering entry.")
        executable, timing = measure_lower_and_compile(
            lambda: entry.lower(state, arguments), lambda lowered: lowered.compile()
        )
        result, warm = measure_repeated(
            lambda: executable(state, arguments), warmup=1, repeats=2
        )
        evidence = compiler_evidence(
            executable.compiled.cost_analysis(),
            executable.compiled.memory_analysis(),
            source="game-kkt-stationarity",
        )
        cases.append(
            {
                "capacity": horizon,
                "players": options.players,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "compiler": asdict(evidence),
                "host_peak_bytes": host_peak,
                "retained_bytes": logical_array_bytes(result),
                "finite": bool(jnp.all(jnp.isfinite(result[0]))),
            }
        )
        jax.clear_caches()
    record = {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), ("cases", "environment")
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "observed_peak_rss_bytes": process_peak_rss_bytes(),
        "compiler_scaling_exponent": fitted_scaling_exponent(
            options.horizons,
            [max(case["compiler"]["temporary_bytes"], 1) for case in cases],
        ),
    }
    if options.baseline_record is not None:
        baseline_record = json.loads(options.baseline_record.read_text())
        record["gate"] = evaluate_memory_gate(
            options.horizons,
            [case["compiler"]["temporary_bytes"] for case in baseline_record["cases"]],
            [case["compiler"]["temporary_bytes"] for case in cases],
            MemoryGatePolicy(),
        ).to_dict()
    write_json_atomic(options.output, record)


if __name__ == "__main__":
    main()

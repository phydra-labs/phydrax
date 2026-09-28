#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Structured two-phase VOF step: lowering, compilation, warmed step, status.

A periodic unit box holds a centered liquid drop (densities 1000/1, surface
tension 1) initialized with area-accurate volume fractions.  Around the base
case (64 cells per axis, radius 1/4, viscosities 0.1/0.001) one sweep varies
the cells per axis, one the drop radius (interface cell count), and one the
liquid/gas viscosities.  The step size is half the Brackbill capillary limit
``sqrt(rho_mean h^3 / (2 pi sigma))``.  The warmed time is the step from rest;
a short trajectory then records how many consecutive steps are accepted.

The driver uses only the plan, material, state, and step API shared by the
revisions it compares (no gravity or wall selection), so the same script
measures a source snapshot through ``PYTHONPATH``.  The candidate evidence of
the last attempted step is serialized field by field as reported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    BenchmarkIdentity,
    capture_benchmark_identity,
    capture_environment,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    validate_benchmark_record,
)


LIQUID_DENSITY = 1000.0
GAS_DENSITY = 1.0
SURFACE_TENSION = 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class DropCase:
    cells: int
    radius: float
    liquid_viscosity: float
    gas_viscosity: float


BASE = DropCase(cells=64, radius=0.25, liquid_viscosity=0.1, gas_viscosity=0.001)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRIVER_PATH = Path(__file__).resolve()
_EVIDENCE_FIELDS = tuple(
    field.name
    for field in dataclasses.fields(phx.applications.two_phase_flow.TwoPhaseStepEvidence)
)


def _benchmark_identity() -> BenchmarkIdentity:
    return capture_benchmark_identity(
        _PROJECT_ROOT,
        _DRIVER_PATH,
        _EVIDENCE_FIELDS,
    )


def _viscosity_pair(value: str) -> tuple[float, float]:
    liquid, gas = (float(entry) for entry in value.split(","))
    return liquid, gas


def _drop_case(value: str) -> DropCase:
    entries = value.split(",")
    if len(entries) != 4:
        raise argparse.ArgumentTypeError(
            "case must be cells,radius,liquid_viscosity,gas_viscosity"
        )
    return DropCase(
        cells=int(entries[0]),
        radius=float(entries[1]),
        liquid_viscosity=float(entries[2]),
        gas_viscosity=float(entries[3]),
    )


def _cases(
    cells: tuple[int, ...],
    radii: tuple[float, ...],
    viscosities: tuple[tuple[float, float], ...],
) -> tuple[DropCase, ...]:
    sweeps = (
        *(dataclasses.replace(BASE, cells=count) for count in cells),
        *(dataclasses.replace(BASE, radius=radius) for radius in radii),
        *(
            dataclasses.replace(BASE, liquid_viscosity=liquid, gas_viscosity=gas)
            for liquid, gas in viscosities
        ),
    )
    return tuple(dict.fromkeys(sweeps))


def _drop_fraction(cells: int, radius: float, samples: int = 16) -> np.ndarray:
    """Liquid fraction of the centered drop from ``samples²`` points per cell."""

    offsets = (np.arange(samples) + 0.5) / (samples * cells)
    x = (np.arange(cells)[:, None] / cells + offsets).reshape(-1)
    inside = (x[:, None] - 0.5) ** 2 + (x[None, :] - 0.5) ** 2 < radius**2
    return inside.reshape(cells, samples, cells, samples).mean(axis=(1, 3))


def _prepare(
    case: DropCase, alpha: np.ndarray, maximum_iterations: int, /
) -> tuple[Any, Any]:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(case.cells, periodic=True),
            phx.discretization.UniformCellAxisSpec(case.cells, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=LIQUID_DENSITY,
        gas_density=GAS_DENSITY,
        liquid_viscosity=case.liquid_viscosity,
        gas_viscosity=case.gas_viscosity,
        surface_tension=SURFACE_TENSION,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization, material, maximum_iterations=maximum_iterations
    ).prepare()
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    return method, method.initial_continuation(
        two_phase.initial_state(jnp.asarray(alpha))
    )


def _evidence_record(evidence: Any, /) -> dict[str, Any]:
    return {
        field.name: np.asarray(getattr(evidence, field.name)).item()
        for field in dataclasses.fields(evidence)
    }


def _array_types(value: Any, /) -> list[tuple[tuple[int, ...], str]]:
    return [
        (leaf.shape, str(leaf.dtype))
        for leaf in jax.tree.leaves(eqx.filter(value, eqx.is_array))
    ]


def _measure(
    case: DropCase, *, steps: int, warmup: int, repeats: int, maximum_iterations: int
) -> dict[str, Any]:
    alpha = _drop_fraction(case.cells, case.radius)
    (method, initial), preparation_seconds = measure_synchronized(
        lambda: _prepare(case, alpha, maximum_iterations)
    )
    width = 1.0 / case.cells
    step_size = 0.5 * float(
        np.sqrt(
            0.5
            * (LIQUID_DENSITY + GAS_DENSITY)
            * width**3
            / (2.0 * np.pi * SURFACE_TENSION)
        )
    )
    dt = jnp.asarray(step_size)
    first_index = jnp.asarray(0, dtype=jnp.int32)
    first_time = jnp.asarray(0.0)
    step = eqx.filter_jit(method.step)
    compiled, compilation = measure_lower_and_compile(
        # ty: ignore[unresolved-attribute]
        lambda: step.lower(first_index, first_time, initial, dt, None),
        lambda lowered: lowered.compile(),
    )
    first, execution = measure_repeated(
        lambda: compiled(first_index, first_time, initial, dt, None),
        warmup=warmup,
        repeats=repeats,
    )
    # The trajectory uses the traced step: it retraces once if the accepted
    # continuation's array types differ from the initial ones, which the AOT
    # executable would refuse.
    continuation = initial
    accepted = 0
    iterations: list[int] = []
    last = first
    for index in range(steps):
        if index > 0:
            last = step(
                jnp.asarray(index, dtype=jnp.int32),
                jnp.asarray(index * step_size),
                continuation,
                dt,
                None,
            )
        iterations.append(int(last.iterations))
        if not bool(last.successful):
            break
        continuation = last.accepted_state
        accepted += 1
    return {
        "case": dataclasses.asdict(case),
        "cells": case.cells**2,
        "mixed_cells": int(np.count_nonzero((alpha > 0.0) & (alpha < 1.0))),
        "cells_per_radius": case.radius * case.cells,
        "step_size": step_size,
        "preparation_seconds": preparation_seconds,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "first_step_successful": bool(first.successful),
        "continuation_types_stable": _array_types(first.accepted_state)
        == _array_types(initial),
        "steps_requested": steps,
        "steps_accepted": accepted,
        "iterations": iterations,
        "last_step_evidence": _evidence_record(last.candidate_state.evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--radii", type=float, nargs="+", default=(0.125, 0.25, 0.375))
    parser.add_argument(
        "--viscosities",
        type=_viscosity_pair,
        nargs="+",
        default=((0.0, 0.0), (0.01, 0.01), (0.1, 0.001)),
        help="liquid,gas viscosity pairs",
    )
    parser.add_argument(
        "--case",
        type=_drop_case,
        help="run one cells,radius,liquid_viscosity,gas_viscosity case",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-iterations", type=int, default=4000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-stored",
        type=Path,
        help="validate a stored record against current source, driver, and evidence",
    )
    args = parser.parse_args()
    identity = _benchmark_identity()
    if args.validate_stored is not None:
        stored = json.loads(args.validate_stored.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            raise ValueError("Stored benchmark record must be a JSON object.")
        validate_benchmark_record(stored, identity)
        return
    cases = (
        (args.case,)
        if args.case is not None
        else _cases(tuple(args.cells), tuple(args.radii), tuple(args.viscosities))
    )
    if (
        any(
            case.cells < 8
            or not 0.0 < case.radius < 0.5
            or min(case.liquid_viscosity, case.gas_viscosity) < 0.0
            for case in cases
        )
        or args.steps <= 0
        or args.warmup < 0
        or args.repeats <= 0
        or args.maximum_iterations <= 0
    ):
        raise ValueError(
            "cells >= 8, radii in (0, 1/2), nonnegative viscosities, and positive "
            "steps/repeats/iterations with nonnegative warmup are required"
        )
    payload = {
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "configuration": {
            "liquid_density": LIQUID_DENSITY,
            "gas_density": GAS_DENSITY,
            "surface_tension": SURFACE_TENSION,
            "base": dataclasses.asdict(BASE),
            "selection": "single-case" if args.case is not None else "sweeps",
            "steps": args.steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "maximum_iterations": args.maximum_iterations,
        },
        "cases": [
            _measure(
                case,
                steps=args.steps,
                warmup=args.warmup,
                repeats=args.repeats,
                maximum_iterations=args.maximum_iterations,
            )
            for case in cases
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

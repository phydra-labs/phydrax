#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated time-independent averaged-FEL benchmark.

A seeded helical amplifier runs ``--slices`` independent slices through a
``--periods``-period undulator in both transverse models. Each case is lowered,
compiled, and executed separately; the beamlet count (macroparticles per
slice) is the controlling capacity of the scaling campaign.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import equinox as eqx
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

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import fel, InsertionDeviceField
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import AngularSpectrumPlan, PlaneFieldSpace


_MODELS = ("one-dimensional", "angular-spectrum")
_PERIOD = 0.03
_GAMMA = 1.0e4
_BEAM_SIZE = 30.0e-6
_BETA = 9.0


def _compiler_record(compiled: Any) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
    }


def _plan(
    model: str,
    beamlets: int,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> fel.FELPlan:
    light = float(scale.speed_of_light)
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    deflection = 3.5 / math.sqrt(2.0)
    natural = (deflection * 2.0 * math.pi / _PERIOD) ** 2 / (2.0 * _GAMMA**2)
    device = InsertionDeviceField(
        deflection * 2.0 * math.pi * mass * light / (charge * _PERIOD),
        _PERIOD,
        arguments.periods,
        polarization="helical",
        center=0.0,
        aperture=(1.0e-2, 1.0e-2),
    )
    segment = fel.FELUndulatorSegment(
        device,
        smooth_focusing_gradient=(1.0 / _BETA**2 - natural)
        * _GAMMA
        * mass
        * light
        / charge,
    )
    lattice = fel.FELUndulatorLattice(scale, (segment,), step_length=arguments.step)
    loading = fel.FELLoading(beamlets, 4, shot_noise="fawley")
    seed = fel.FELSeed(1.0e3, waist=40.0e-6)
    wavelength = lattice.resonant_wavelength(_GAMMA)
    if model == "one-dimensional":
        return fel.FELPlan(
            lattice,
            wavelength,
            loading=loading,
            seed=seed,
            slice_batch=arguments.slice_batch,
        )
    half = 8.0 * _BEAM_SIZE
    grid = TensorGridPlan(
        (UniformAxisSpec(arguments.grid), UniformAxisSpec(arguments.grid)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[-half, -half], [half, half]]))
    return fel.FELPlan(
        lattice,
        wavelength,
        loading=loading,
        transverse="angular-spectrum",
        field_space=PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window"),
        propagation=AngularSpectrumPlan(arguments.grid // 4),
        seed=seed,
        slice_batch=arguments.slice_batch,
    )


def _case(
    model: str,
    beamlets: int,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> dict[str, object]:
    plan = _plan(model, beamlets, arguments, scale)
    count = arguments.slices
    emittance = _BEAM_SIZE**2 / _BETA * _GAMMA
    slices = fel.FELBeamSlices(
        np.arange(count, dtype=np.float64) * plan.wavelength,
        np.full(count, 3000.0),
        np.full(count, _GAMMA),
        np.full(count, 1.0e-4),
        np.full((count, 2), emittance),
        np.full((count, 2), _BETA),
        np.zeros((count, 2)),
    )
    key = jax.random.key(0)
    dynamic, static = eqx.partition((plan, slices), eqx.is_array)

    def evaluate(leaves: Any, key_: jax.Array) -> Any:
        plan_, slices_ = eqx.combine(leaves, static)
        result = plan_.solve(slices_, key_)
        return result.power, result.ledger.relative_defect, result.evidence.status

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic, key), lambda lowered: lowered.compile()
    )
    (power, defect, status), execution = measure_repeated(
        lambda: compiled(dynamic, key), warmup=arguments.warmup, repeats=arguments.repeats
    )
    return {
        "model": model,
        "beamlets": beamlets,
        "particles_per_slice": plan.loading.particle_count,
        "steps": int(plan.step_lengths.shape[0]),
        "identity": plan.plan_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "compiler": _compiler_record(compiled),
        "execution": execution.to_seconds_dict(),
        "memory": {
            "input_logical_bytes": logical_array_bytes(dynamic),
            "output_logical_bytes": logical_array_bytes(power),
        },
        "physics": {
            "finite": bool(np.all(np.isfinite(np.asarray(power)))),
            "status": [int(value) for value in np.asarray(status)],
            "maximum_relative_ledger_defect": float(np.max(np.asarray(defect))),
            "exit_power_W": float(np.max(np.asarray(power)[:, -1, 0])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beamlets", type=str, default="256,1024")
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--step", type=float, default=0.06)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--slices", type=int, default=2)
    parser.add_argument("--slice-batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    beamlet_counts = tuple(int(value) for value in arguments.beamlets.split(","))
    if (
        min(beamlet_counts) < 1
        or arguments.periods < 1
        or arguments.step <= 0.0
        or arguments.grid < 8
        or min(arguments.slices, arguments.slice_batch) < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "positive beamlets, periods, step, slices and slice batch, grid >= 8, "
            "nonnegative warmup and positive repeats are required"
        )
    scale = ElectromagneticScaleContract.si()
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "periods": arguments.periods,
            "step": arguments.step,
            "grid": arguments.grid,
            "slices": arguments.slices,
            "slice_batch": arguments.slice_batch,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": [
            _case(model, count, arguments, scale)
            for model in _MODELS
            for count in beamlet_counts
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

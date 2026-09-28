#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated time-dependent averaged-FEL benchmark.

A SASE helical amplifier with ``--slices`` coupled slices (one wavelength
apart, open window with head padding covering the slippage) runs through a
``--periods``-period undulator for both slippage routes and both transverse
models. Each case is lowered, compiled, and executed separately; the slice
count (radiation window size) is the controlling capacity of the scaling
campaign.
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
_ROUTES = ("commensurate", "spectral")
_PERIOD = 0.03
_GAMMA = 1.0e4
_BEAM_SIZE = 30.0e-6
_DEFLECTION = 3.5 / math.sqrt(2.0)
_BETA = math.sqrt(2.0) * _GAMMA / (_DEFLECTION * 2.0 * math.pi / _PERIOD)


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
    route: str,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> fel.FELTimeDependentPlan:
    light = float(scale.speed_of_light)
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    device = InsertionDeviceField(
        _DEFLECTION * 2.0 * math.pi * mass * light / (charge * _PERIOD),
        _PERIOD,
        arguments.periods,
        polarization="helical",
        center=0.0,
        aperture=(1.0e-2, 1.0e-2),
    )
    # One-period steps slip one wavelength: one slot per step (commensurate).
    lattice = fel.FELUndulatorLattice(
        scale, (fel.FELUndulatorSegment(device),), step_length=_PERIOD
    )
    wavelength = lattice.resonant_wavelength(_GAMMA)
    loading = fel.FELLoading(arguments.beamlets, 4, shot_noise="fawley")
    if model == "one-dimensional":
        core = fel.FELPlan(
            lattice, wavelength, loading=loading, slice_batch=arguments.slice_batch
        )
    else:
        half = 8.0 * _BEAM_SIZE
        grid = TensorGridPlan(
            (UniformAxisSpec(arguments.grid), UniformAxisSpec(arguments.grid)),
            axis_names=("x", "y"),
        ).prepare(jnp.asarray([[-half, -half], [half, half]]))
        core = fel.FELPlan(
            lattice,
            wavelength,
            loading=loading,
            transverse="angular-spectrum",
            field_space=PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window"),
            propagation=AngularSpectrumPlan(arguments.grid // 4),
            slice_batch=arguments.slice_batch,
        )
    return fel.FELTimeDependentPlan(
        core,
        slippage=route,  # ty: ignore[invalid-argument-type]
        boundary="open",
        head_padding=arguments.periods,
    )


def _case(
    model: str,
    route: str,
    count: int,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> dict[str, object]:
    plan = _plan(model, route, arguments, scale)
    emittance = _BEAM_SIZE**2 / _BETA * _GAMMA
    slices = fel.FELBeamSlices(
        np.arange(count, dtype=np.float64) * plan.core.wavelength,
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
        return (
            result.pulse_energy,
            result.spectrum.spike_count,
            result.ledger.relative_defect,
            result.evidence.status,
        )

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic, key), lambda lowered: lowered.compile()
    )
    (energy, spikes, defect, status), execution = measure_repeated(
        lambda: compiled(dynamic, key), warmup=arguments.warmup, repeats=arguments.repeats
    )
    return {
        "model": model,
        "route": route,
        "slices": count,
        "window_slots": count + plan.head_padding,
        "particles_per_slice": plan.core.loading.particle_count,
        "steps": int(plan.core.step_lengths.shape[0]),
        "identity": plan.plan_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "compiler": _compiler_record(compiled),
        "execution": execution.to_seconds_dict(),
        "memory": {
            "input_logical_bytes": logical_array_bytes(dynamic),
            "output_logical_bytes": logical_array_bytes(energy),
        },
        "physics": {
            "finite": bool(np.all(np.isfinite(np.asarray(energy)))),
            "status": int(status),
            "relative_ledger_defect": float(defect),
            "exit_pulse_energy_J": float(np.asarray(energy)[-1, 0]),
            "spikes": int(np.asarray(spikes)[0]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", type=str, default="128,512")
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--beamlets", type=int, default=4)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--slice-batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    slice_counts = tuple(int(value) for value in arguments.slices.split(","))
    if (
        min(slice_counts) < 2
        or arguments.periods < 1
        or arguments.beamlets < 1
        or arguments.grid < 8
        or arguments.slice_batch < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "at least two slices, positive periods, beamlets and slice batch, "
            "grid >= 8, nonnegative warmup and positive repeats are required"
        )
    scale = ElectromagneticScaleContract.si()
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "periods": arguments.periods,
            "beamlets": arguments.beamlets,
            "grid": arguments.grid,
            "slice_batch": arguments.slice_batch,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": [
            _case(model, route, count, arguments, scale)
            for model in _MODELS
            for route in _ROUTES
            for count in slice_counts
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

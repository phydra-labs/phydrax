"""Phase-separated strong-field QED benchmark.

Phases: host table preparation (both processes), a nonlinear Compton step of
``--particles`` leptons, a Breit–Wheeler step of ``--particles`` photons, and a
full PIC cascade step (rotating electric field, reduced 1-D grid) whose lepton
capacity is ``--capacity`` and photon capacity ``4 × capacity``, for the
``--polarization`` model (`QEDPolarizationModel`; polarized models also prepare
the polarized tables). Each compiled phase reports lowering, compilation, and
warmed execution separately.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, assert_never, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from _runtime import (
    capture_environment,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax.discretization import pic
from phydrax.units import CHARGE, UnitDefinition


# Code units c = ε₀ = 1, ω₀ = 1 for a 1 μm laser: a_S = 4.1e5, q = m = ħ a_S.
_SCHWINGER = 410000
_HBAR = 4 * Fraction(math.pi) * Fraction(1000, 137036) / _SCHWINGER**2
_MASS = float(_HBAR) * _SCHWINGER


def _scale() -> phx.ElectromagneticScaleContract:
    return phx.ElectromagneticScaleContract.code_units(
        pic.PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=_HBAR,
        boltzmann_constant=1,
        elementary_charge=Fraction(_MASS),
        electron_mass=Fraction(_MASS),
        vacuum_permittivity=1,
        constant_set_id="qed-cascade-benchmark",
    )


class _RotatingField(StrictModule):
    amplitude: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return f"rotating-electric-{self.amplitude!r}"

    def external_fields(self, positions: Any, times: Any, /) -> pic.ExternalFieldSample:
        del positions
        electric = self.amplitude * jnp.stack(
            (jnp.cos(times), jnp.sin(times), jnp.zeros_like(times)), axis=-1
        )
        return pic.ExternalFieldSample(
            electric, jnp.zeros_like(electric), jnp.ones(times.shape, dtype=bool)
        )


def _species(capacity: int, sign: float, name: str, offset: int) -> pic.PICSpeciesPlan:
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + capacity), jnp.ones((capacity,)), ambient_dimension=1
    ).prepare()
    return pic.PICSpeciesPlan(
        phx.discretization.ParticlePopulationPlan(support),
        pic.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


def _phase(
    function: Any, arguments: tuple[Any, ...], warmup: int, repeats: int
) -> dict[str, Any]:
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(*arguments), lambda lowered: lowered.compile()
    )
    _, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return {
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--polarization",
        choices=get_args(pic.QEDPolarizationModel),
        default="unpolarized",
    )
    args = parser.parse_args()
    polarization: pic.QEDPolarizationModel = args.polarization
    match polarization:
        case "unpolarized":
            photon, spin = False, False
        case "photon-polarized":
            photon, spin = True, False
        case "spin-and-photon-polarized":
            photon, spin = True, True
        case _:
            assert_never(polarization)
    if args.particles <= 0 or args.capacity <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("positive sizes/repeats and nonnegative warmup are required")

    (compton_table, pair_table, polarized), table_seconds = measure_host(
        lambda: (
            pic.QEDTable("nonlinear-compton", maximum_chi=100.0),
            pic.QEDTable("nonlinear-breit-wheeler", maximum_chi=100.0),
            {
                (process, sign): pic.QEDTable(
                    process, maximum_chi=100.0, polarization=sign
                )
                for process, needed in (
                    ("nonlinear-compton", spin),
                    ("nonlinear-breit-wheeler", photon),
                )
                if needed
                for sign in ("positive", "negative")
            },
        )
    )
    scale = _scale()
    compton = pic.NonlinearComptonPlan(
        "improved-lcfa",
        scale,
        -_MASS,
        _MASS,
        compton_table,
        maximum_chi=100.0,
        minimum_gamma=1.0,
        polarization=polarization,
        spin_tables=(
            (
                polarized["nonlinear-compton", "positive"],
                polarized["nonlinear-compton", "negative"],
            )
            if spin
            else None
        ),
    )
    pairs = pic.NonlinearBreitWheelerPlan(
        scale,
        _MASS,
        _MASS,
        pair_table,
        maximum_chi=100.0,
        polarization=polarization,
        polarized_tables=(
            (
                polarized["nonlinear-breit-wheeler", "positive"],
                polarized["nonlinear-breit-wheeler", "negative"],
            )
            if photon
            else None
        ),
    )
    count = args.particles
    keys = jr.split(jr.key(0), 4)
    gamma = jr.uniform(keys[0], (count,), dtype=jnp.float64, minval=500.0, maxval=2000.0)
    direction = jr.normal(keys[1], (count, 3), dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    proper = direction * jnp.sqrt(gamma**2 - 1.0)[:, None]
    electric = jnp.broadcast_to(jnp.asarray([600.0, 0.0, 0.0]), (count, 3))
    magnetic = jnp.zeros((count, 3))
    active = jnp.ones((count,), dtype=bool)
    high = jnp.zeros((count,), dtype=jnp.uint32)
    low = jnp.arange(count, dtype=jnp.uint32)
    tau = jnp.full((count,), 1.0e6 * compton.compton_time)

    def emit(key: Any) -> Any:
        result = compton.apply(
            proper,
            electric,
            magnetic,
            0.01,
            active,
            compton.initial_optical_depth(key, high, low),
            compton.uniforms(key, high, low),
            variation_time=tau,
            spin=0.5 * direction if spin else None,
        )
        return result.proper_velocity, jnp.sum(result.photon_energy), result.flags

    momentum = proper * _MASS
    # Photons linearly polarized along a direction perpendicular to their momentum.
    axis = jnp.cross(direction, jnp.asarray([0.0, 0.0, 1.0]))
    axis = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
    stokes = jnp.broadcast_to(jnp.asarray([0.5, 0.0]), (count, 2))

    def decay(key: Any) -> Any:
        result = pairs.apply(
            momentum,
            electric,
            magnetic,
            0.01,
            active,
            pairs.initial_optical_depth(key, high, low),
            pairs.uniforms(key, high, low),
            stokes=stokes if photon else None,
            polarization_axis=axis if photon else None,
        )
        return jnp.sum(result.decayed), result.electron_momentum

    capacity = args.capacity
    photons = pic.QEDPhotonSpeciesPlan(
        4 * capacity,
        1,
        escape_lower=(-1.0e9,),
        escape_upper=(1.0e9,),
        energy_edges=tuple(_MASS * 10.0 ** (0.5 * k) for k in range(11)),
    )
    process = pic.QEDCascadeProcess(
        compton,
        photons,
        emitters=(0, 1),
        breit_wheeler=pairs,
        electron=0,
        positron=1,
        gather_species=0,
        minimum_photon_energy=2.0 * _MASS,
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(64, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [100.0]]))
    run = phx.solver.ElectromagneticPICPlan(
        phx.solver.ReducedMaxwellPICFieldSolver(
            phx.solver.CompatibleMaxwell1DPlan(grid), pic.ReducedPICTransferPlan(grid)
        ),
        species=(
            _species(capacity, -1.0, "electrons", 0),
            _species(capacity, 1.0, "positrons", 10 * capacity),
        ),
        processes=(process,),
        ownership="subgrid-reaction",
        external_fields=(_RotatingField(1000.0),),
        key=jr.key(1),
    )
    seeds = capacity // 2
    position = jnp.zeros((capacity, 1)).at[:seeds, 0].set(jnp.linspace(10.0, 90.0, seeds))
    live = jnp.arange(capacity) < seeds
    mass = jnp.where(live, 1.0e-9 * _MASS, 0.0)
    state = run.initialize(
        (position, position),
        (jnp.zeros((capacity, 3)),) * 2,
        0.01,
        active_masks=(live, live),
        masses=(mass, mass),
    )

    @eqx.filter_jit
    def cascade_step(plan: Any, state: Any) -> Any:
        return plan.step_detailed(state, 0.01).accepted_state

    step_function = jax.jit(lambda state: cascade_step(run, state))
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "particles": count,
            "capacity": capacity,
            "photon_capacity": 4 * capacity,
            "polarization": polarization,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "tables": {
            "preparation_seconds": table_seconds,
            "compton": {
                "identity": compton_table.table_id,
                "rows": compton_table.row_count,
                "rate_interpolation_error": compton_table.rate_interpolation_error,
                "cdf_error": compton_table.cdf_row_error + compton_table.cdf_node_error,
            },
            "breit_wheeler": {
                "identity": pair_table.table_id,
                "rows": pair_table.row_count,
                "rate_interpolation_error": pair_table.rate_interpolation_error,
                "cdf_error": pair_table.cdf_row_error + pair_table.cdf_node_error,
            },
        },
        "nonlinear_compton": _phase(jax.jit(emit), (keys[2],), args.warmup, args.repeats),
        "breit_wheeler": _phase(jax.jit(decay), (keys[3],), args.warmup, args.repeats),
        "cascade_step": _phase(step_function, (state,), args.warmup, args.repeats),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

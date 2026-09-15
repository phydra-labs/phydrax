#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _runtime import capture_environment, measure_repeated

import phydrax as phx
from phydrax.applications import functional_rg as frg
from phydrax.discretization import dlr
from phydrax.nn.quantum import variable_sector as variable_amplitudes
from phydrax.operators.quantum import variable_sector


def _algebraic_closure() -> dict:
    order = phx.operators.quantum.FermionModeOrder(("a", "b", "c"))
    car = phx.operators.quantum.car_evidence(order)
    fibonacci = phx.tensor_network.fibonacci_fusion_category()
    squeezed = 0.5 * jnp.asarray(
        [
            [jnp.cosh(1.0), 0.0, jnp.sinh(1.0), 0.0],
            [0.0, jnp.cosh(1.0), 0.0, -jnp.sinh(1.0)],
            [jnp.sinh(1.0), 0.0, jnp.cosh(1.0), 0.0],
            [0.0, -jnp.sinh(1.0), 0.0, jnp.cosh(1.0)],
        ]
    )
    gaussian = (
        phx.metrix.GaussianEntanglementPlan(
            2,
            (0, 1),
            transposed_modes=(1,),
        )
        .prepare()
        .evaluate(phx.metrix.BosonicGaussianState(jnp.zeros(4), squeezed))
    )
    return {
        "car_maximum_residual": float(
            jnp.max(
                jnp.stack(
                    (
                        car.annihilation_residual,
                        car.mixed_residual,
                        car.permutation_unitarity_residual,
                    )
                )
            )
        ),
        "car_valid": bool(car.valid),
        "fibonacci_pentagon_residual": float(fibonacci.coherence.pentagon_residual),
        "fibonacci_hexagon_residual": float(fibonacci.coherence.hexagon_residual),
        "fibonacci_coherent": bool(fibonacci.coherence.coherent),
        "gaussian_logarithmic_negativity": float(gaussian.logarithmic_negativity),
        "gaussian_valid": bool(gaussian.valid),
    }


def _thermal_closure(repeats: int) -> dict:
    basis, timing = measure_repeated(
        lambda: dlr.generate_dlr_basis(
            6.0,
            5.0,
            policy=dlr.DLRBasisPolicy(
                tolerance=2e-6,
                maximum_rank=24,
                candidate_count=56,
            ),
        ),
        warmup=0,
        repeats=repeats,
    )
    poles = jnp.asarray([-1.2, 0.4, 1.8])
    residues = jnp.asarray([0.25, 0.5, 0.25])
    green = phx.operators.quantum.dlr_from_poles(
        basis,
        poles,
        residues,
        tolerance=2e-5,
    )
    labels = jnp.arange(-10, 11)
    exact = jnp.sum(
        residues[None, :]
        / (
            1j
            * dlr.matsubara_frequencies(
                labels,
                beta=basis.beta,
                statistics="fermionic",
            )[:, None]
            - poles[None, :]
        ),
        axis=1,
    )
    computed = phx.operators.quantum.evaluate_dlr_matsubara(green, labels)
    return {
        "rank": int(basis.rank),
        "requested_tolerance": basis.evidence.requested_tolerance,
        "achieved_tolerance": float(basis.evidence.achieved_tolerance),
        "matsubara_maximum_error": float(jnp.max(jnp.abs(computed - exact))),
        "generation": timing.to_milliseconds_dict(),
    }


def _integration_closure(repeats: int) -> dict:
    plan = phx.integration.VegasPlan(
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        bins=8,
        adaptation_iterations=2,
        adaptation_samples=256,
        production_iterations=4,
        production_samples=512,
    )
    result, timing = measure_repeated(
        lambda: phx.integration.vegas_integrate(
            lambda points: points[:, 0] ** 2,
            plan,
            jr.key(1),
            jr.key(2),
        ),
        warmup=0,
        repeats=repeats,
    )
    measure = phx.integration.complex_weight_measure(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        jnp.asarray([1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 0.0j]),
        source_id="closure-three-point",
    )
    prepared = phx.integration.prepare_phase_quenched_reweighting(
        measure,
        phx.integration.PhaseQuenchedReweightingPlan(
            minimum_average_phase=0.1,
            minimum_effective_sample_size=2.0,
            maximum_samples=8,
        ),
    )
    reweighted = phx.integration.phase_quenched_reweight(
        prepared,
        jnp.asarray([0.0, 2.0, 1.0]),
    )
    return {
        "vegas_value": float(result.value),
        "vegas_absolute_error": float(jnp.abs(result.value - 1.0 / 3.0)),
        "vegas_standard_error": float(result.standard_error),
        "vegas_status": int(result.status),
        "reweighting_value_real": float(jnp.real(reweighted.value)),
        "reweighting_value_imag": float(jnp.imag(reweighted.value)),
        "reweighting_status": int(reweighted.status),
        "execution": timing.to_milliseconds_dict(),
    }


def _functional_scattering_closure(repeats: int) -> dict:
    regulator = frg.Regulator.optimized()
    quadrature = frg.ThresholdQuadraturePlan(
        3.0,
        quadrature_order=16,
        momentum_upper=8.0,
    )
    flow = frg.PolynomialONFlowPlan(
        1,
        3.0,
        regulator,
        quadrature,
        coupling_count=2,
    )
    fixed, timing = measure_repeated(
        lambda: frg.FixedPointSearchPlan(
            2,
            maximum_iterations=40,
            absolute_tolerance=1e-9,
            relative_tolerance=1e-8,
        ).search(flow, jnp.asarray([-0.1, 5.0])),
        warmup=0,
        repeats=repeats,
    )
    electron = phx.applications.relativistic_scattering.Particle(
        "electron",
        mass=0.511,
        charge=-1.0,
        spin_twice=1,
        antiparticle="positron",
        statistics="fermion",
    )
    shell = phx.applications.relativistic_scattering.MassShell(electron)
    momentum = shell.from_spatial(jnp.asarray([0.3, -0.2, 0.7]))
    boost = phx.applications.relativistic_scattering.LorentzFrame.boost(
        jnp.asarray([0.2, -0.1, 0.05])
    )
    transformed = boost.apply(momentum)
    invariant_error = jnp.abs(
        phx.applications.relativistic_scattering.minkowski_dot(
            transformed.value,
            transformed.value,
        )
        - phx.applications.relativistic_scattering.minkowski_dot(
            momentum.value,
            momentum.value,
        )
    )
    return {
        "fixed_point_converged": bool(fixed.converged),
        "fixed_point_residual": float(fixed.residual_norm),
        "critical_exponents": [float(value) for value in fixed.critical_exponents],
        "lorentz_invariant_error": float(invariant_error),
        "fixed_point_execution": timing.to_milliseconds_dict(),
    }


def _variable_sector_closure() -> dict:
    space = variable_sector.VariableSectorSpace(4, 1, 2)
    configuration = variable_sector.VariableParticleConfiguration(
        jnp.asarray([[1.0], [2.0], [100.0], [-20.0]]),
        jnp.asarray([True, True, False, False]),
        jnp.asarray([0, 0, 1, 1]),
    )
    measure = variable_sector.VariableSectorMeasure(space)
    amplitude = variable_amplitudes.BosonicJastrowAmplitude(
        space,
        precisions=jnp.asarray([[1.0], [2.0]]),
        pair_cusp=jnp.asarray([[0.3, 0.2], [0.2, 0.1]]),
        pair_range=0.5,
    )(configuration)
    return {
        "sector_factor": float(measure.sector_factor(configuration)),
        "log_amplitude": float(amplitude.log_abs),
        "amplitude_valid": bool(amplitude.valid),
    }


def _json_default(value):
    if isinstance(value, jax.Array):
        host = np.asarray(jax.device_get(value))
        return host.item() if host.shape == () else host.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "algebraic": _algebraic_closure(),
        "thermal": _thermal_closure(arguments.repeats),
        "integration": _integration_closure(arguments.repeats),
        "functional_scattering": _functional_scattering_closure(arguments.repeats),
        "variable_sector": _variable_sector_closure(),
    }
    encoded = json.dumps(payload, indent=2, default=_json_default)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

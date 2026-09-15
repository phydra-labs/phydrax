#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.atomistic import AtomicStructure, AtomisticScaleContract
from phydrax.discretization import PeriodicCell
from phydrax.nn.quantum._periodic_ferminet import PeriodicFermiNet
from phydrax.operators.quantum._electronic import ElectronicKineticPolicy
from phydrax.operators.quantum._electronic_advanced import ElectronicVMCResourcePlan
from phydrax.operators.quantum._periodic_electronic import (
    PeriodicElectronicCoulombHamiltonian,
    PeriodicElectronicEwaldPolicy,
)
from phydrax.units import BOHR, HARTREE


CASES = (
    (1, 1, 0, 0),
    (2, 1, 1, 1),
    (4, 2, 1, 1),
)


def _case(electrons: int, determinants: int, real_radius: int, reciprocal_radius: int):
    vectors = jnp.asarray(
        [[5.0, 0.0, 0.0], [0.7, 4.6, 0.0], [0.3, 0.4, 5.3]],
        dtype=jnp.float64,
    )
    cell = PeriodicCell(vectors)
    structure = AtomicStructure(
        jnp.asarray([electrons], dtype=jnp.int32),
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.ones((1,), dtype=jnp.float64),
        AtomisticScaleContract(BOHR, HARTREE),
        cell=vectors,
        periodic_axes=jnp.ones((3,), dtype=bool),
        name=f"periodic-Z{electrons}",
    )
    resource = ElectronicVMCResourcePlan(
        electrons,
        determinant_count=determinants,
        spatial_dimension=3,
        maximum_pair_elements=max(3 * electrons * electrons, 1),
        maximum_determinant_work=max(determinants * electrons**3, 1),
    )
    modes = jnp.stack(
        (
            jnp.arange(electrons, dtype=jnp.int32),
            jnp.zeros((electrons,), dtype=jnp.int32),
            jnp.zeros((electrons,), dtype=jnp.int32),
        ),
        axis=1,
    )
    orbital_coefficients = jnp.broadcast_to(
        jnp.eye(electrons, dtype=jnp.float64),
        (determinants, electrons, electrons),
    )
    twist = jnp.asarray([0.17, -0.11, 0.07], dtype=jnp.float64)
    model = PeriodicFermiNet(
        cell,
        modes,
        orbital_coefficients,
        jnp.ones((determinants,), dtype=jnp.float64) / determinants,
        twist=twist,
        pair_jastrow_strength=0.05,
        resource_plan=resource,
    )
    particle_count = electrons + 1
    real_image_count = (2 * real_radius + 1) ** 3
    reciprocal_mode_count = (2 * reciprocal_radius + 1) ** 3 - 1
    ewald = PeriodicElectronicEwaldPolicy(
        real_image_radius=real_radius,
        reciprocal_radius=reciprocal_radius,
        screening=0.75,
        maximum_real_pair_terms=particle_count**2 * real_image_count,
        maximum_reciprocal_structure_terms=max(particle_count * reciprocal_mode_count, 1),
    )
    operator = PeriodicElectronicCoulombHamiltonian(
        structure,
        electrons,
        cell,
        twist=twist,
        ewald=ewald,
        kinetic=ElectronicKineticPolicy(
            trace_method="chunked-exact", coordinate_chunk_size=3
        ),
        resource_plan=resource,
    )
    index = jnp.arange(electrons, dtype=jnp.float64)
    fractional = jnp.stack(
        (
            (index + 1.0) / (electrons + 2.0),
            jnp.mod(0.19 + 0.23 * index, 1.0),
            jnp.mod(0.31 + 0.17 * index, 1.0),
        ),
        axis=1,
    )
    base = cell.cartesian(fractional)
    offsets = jnp.asarray([0.0, 0.003, -0.002, 0.005], dtype=jnp.float64)
    walkers = base[None, :, :] + offsets[:, None, None]
    return model, operator, walkers


def _record(
    electrons: int,
    determinants: int,
    real_radius: int,
    reciprocal_radius: int,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model, operator, walkers = _case(
        electrons, determinants, real_radius, reciprocal_radius
    )

    @eqx.filter_jit
    def evaluate(model_, operator_, walkers_):
        return operator_.estimate(model_, walkers_), operator_.potential(walkers_)

    inputs = (model, operator, walkers)
    compiled, compilation = measure_lower_and_compile(
        lambda: evaluate.lower(*inputs), lambda lowered: lowered.compile()
    )
    (estimate, potential), execution = measure_repeated(
        lambda: compiled(*inputs), warmup=warmup, repeats=repeats
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    shifted = walkers + operator.cell.vectors[0]
    shifted_estimate = evaluate(model, operator, shifted)[0]
    periodic_residual = float(
        jnp.max(jnp.abs(shifted_estimate.value - estimate.value), initial=0.0)
    )
    decomposition_residual = float(
        jnp.max(
            jnp.abs(
                potential.total
                - potential.electron_electron
                - potential.electron_nucleus
                - potential.nucleus_nucleus
            ),
            initial=0.0,
        )
    )
    evidence = operator.resource_evidence
    return {
        "axes": {
            "electron_count": electrons,
            "determinant_count": determinants,
            "walker_count": int(walkers.shape[0]),
            "coordinate_count": 3 * electrons,
            "real_image_radius": real_radius,
            "reciprocal_radius": reciprocal_radius,
        },
        "identity": {
            "operator_id": operator.operator_id,
            "method_id": operator.method_id,
            "network_id": model.network_id,
            "cell_id": operator.cell.cell_id,
            "boundary_id": operator.boundary_id,
            "ewald_policy_id": operator.ewald.policy_id,
            "input_fingerprint": array_tree_fingerprint(walkers),
        },
        "timing": {
            "lower_seconds": compilation.lowering_seconds,
            "compile_seconds": compilation.compilation_seconds,
            "steady": execution.to_milliseconds_dict(),
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(inputs),
            "logical_output_bytes": logical_array_bytes((estimate, potential)),
            "compiler_argument_bytes": compiler.argument_bytes,
            "compiler_output_bytes": compiler.output_bytes,
            "compiler_temporary_bytes": compiler.temporary_bytes,
            "compiler_generated_code_bytes": compiler.generated_code_bytes,
            "measured_peak_host_bytes": None,
            "measured_peak_host_unavailable_reason": "no portable process-local peak sampler in benchmark runtime",
            "measured_peak_device_bytes": None,
            "measured_peak_device_unavailable_reason": (
                "backend exposes compiler estimates, not synchronized peak allocation"
            ),
        },
        "work": {
            "pair_feature_elements": evidence.pair_feature_elements,
            "electron_pair_count": evidence.electron_pair_count,
            "determinant_cubic_work": evidence.determinant_cubic_work,
            "kinetic_hessian_vector_products": evidence.kinetic_hessian_vector_products,
            "kinetic_coordinate_chunk_size": evidence.kinetic_coordinate_chunk_size,
            "ewald_real_image_count": evidence.ewald_real_image_count,
            "ewald_reciprocal_mode_count": evidence.ewald_reciprocal_mode_count,
            "ewald_real_pair_terms": evidence.ewald_real_pair_terms,
            "ewald_reciprocal_structure_terms": evidence.ewald_reciprocal_structure_terms,
            "primitive_work_per_configuration": evidence.primitive_work_per_configuration,
            "compiler_flops": compiler.flops,
            "compiler_bytes_accessed": compiler.bytes_accessed,
        },
        "scientific_residuals": {
            "integer_cell_translation_local_energy": periodic_residual,
            "local_potential_decomposition": decomposition_residual,
        },
        "execution": {
            "successful_walkers": int(jnp.sum(estimate.successful)),
            "unsuccessful_walkers": int(jnp.sum(~estimate.successful)),
            "status_codes": np.asarray(estimate.status).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive.")
    payload = {
        "benchmark": "fixed-cell-periodic-electronic-vmc-local-energy",
        "environment": capture_environment().to_dict(),
        "cases": [
            _record(*case, warmup=args.warmup, repeats=args.repeats) for case in CASES
        ],
        "nonclaims": [
            "benchmark success is not qualification or release evidence",
            "finite Ewald shells are not continuum-converged electrostatics",
            "no gradient is claimed across wrapping or minimum-image selection",
            "measured cases do not imply an unrestricted electron-count envelope",
        ],
    }
    payload["artifact_bytes_excluding_this_field"] = len(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

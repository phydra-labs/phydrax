#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tiny regulated 2D twisted-SYM algebra, RHMC, Ward, and phase evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.supersymmetric_lattice import (
    assess_twisted_fermion_algebra,
    assess_twisted_n2_chain,
    PfaffianControlPlan,
    prepare_twisted_n2_rhmc,
    sample_twisted_n2_rhmc,
    supersymmetric_lattice_candidate_profiles,
    TwistedN2SYMPlan,
    TwistedSYMCoordinateLayout,
    WardIdentityPlan,
)
from phydrax.sampling import RHMCResourcePolicy


jax.config.update("jax_enable_x64", True)


def run_qualification() -> dict[str, object]:
    theory = TwistedN2SYMPlan(
        (1, 1),
        matrix_rank=1,
        coupling=1.0,
        fermion_mass=1.0,
        coordinate_bound=0.25,
        temporal_axis=0,
        fermion_boundary_phase=-1.0,
    )
    layout = TwistedSYMCoordinateLayout(theory.prepare_bosonic())
    coordinates = jnp.zeros(layout.coordinate_shape, dtype=jnp.float64)
    prepared = prepare_twisted_n2_rhmc(
        theory,
        coordinates,
        step_size=1e-4,
        trajectory_steps=1,
        rational_poles=4,
        rational_verification_points=1025,
        bosonic_substeps=1,
        resources=RHMCResourcePolicy(
            maximum_terms=8,
            maximum_force_evaluations=64,
            maximum_retained_bytes=20_000_000,
            maximum_output_bytes=20_000_000,
            maximum_draws=2,
        ),
    )
    coordinate_evidence = layout.evidence(coordinates)
    algebra = assess_twisted_fermion_algebra(
        prepared.kahler_dirac,
        regulator_mass=theory.fermion_mass,
        structural_lower_bound=theory.normal_spectral_lower,
        structural_upper_bound=theory.normal_spectral_upper,
        maximum_dense_elements=prepared.kahler_dirac.source.size**2,
        tolerance=1e-10,
    )
    run = sample_twisted_n2_rhmc(
        prepared,
        coordinates,
        jax.random.key(481),
        num_draws=1,
    )
    chain = assess_twisted_n2_chain(
        prepared,
        run,
        WardIdentityPlan(
            float(run.samples.bosonic_action[0]),
            absolute_tolerance=1e-12,
            maximum_samples=1,
        ),
        PfaffianControlPlan(
            maximum_dimension=4,
            minimum_magnitude=0.0,
        ),
        minimum_phase_effective_samples=1.0,
        maximum_dense_elements=16,
    )
    action_rational = prepared.pseudofermion.action_approximation
    refresh_rational = prepared.pseudofermion.refresh_approximation
    successful = bool(
        coordinate_evidence.finite
        and algebra.accepted
        and run.evidence.successful
        and chain.phase_overlap_sufficient
    )
    return {
        "kind": "supersymmetric-lattice-candidate-qualification",
        "profiles": [
            profile.to_record() for profile in supersymmetric_lattice_candidate_profiles()
        ],
        "case": {
            "theory": "regulated-two-dimensional-twisted-n2-u1",
            "theory_plan_id": theory.plan_id,
            "prepared_id": prepared.prepared_id,
            "kernel_id": prepared.kernel.kernel_id,
            "coordinate_layout_id": layout.layout_id,
            "kahler_dirac_id": prepared.kahler_dirac.operator_id,
            "regulated_dirac_id": prepared.regulated_dirac.operator_id,
            "phase_quenched_power": prepared.phase_quenched_power,
        },
        "raw": {
            "initial_coordinates": np.asarray(coordinates).tolist(),
            "retained_configurations": np.asarray(run.samples.configurations).tolist(),
            "accepted": np.asarray(run.samples.accepted).tolist(),
            "acceptance_probability": np.asarray(
                run.samples.acceptance_probability
            ).tolist(),
            "energy_error": np.asarray(run.samples.energy_error).tolist(),
            "pfaffian_phase": np.asarray(chain.pfaffian_phases).tolist(),
            "average_phase_real": float(jnp.real(chain.average_phase)),
            "average_phase_imag": float(jnp.imag(chain.average_phase)),
        },
        "criteria": {
            "coordinate_roundtrip_residual": float(
                coordinate_evidence.roundtrip_residual
            ),
            "fermion_antisymmetry_residual": float(algebra.antisymmetry_residual),
            "fermion_adjoint_residual": float(algebra.adjoint_residual),
            "normal_minimum_eigenvalue": float(algebra.normal_minimum_eigenvalue),
            "normal_maximum_eigenvalue": float(algebra.normal_maximum_eigenvalue),
            "action_rational_relative_error": float(
                action_rational.maximum_relative_error
            ),
            "refresh_rational_relative_error": float(
                refresh_rational.maximum_relative_error
            ),
            "acceptance_rate": float(run.evidence.acceptance_rate),
            "maximum_energy_error": float(run.evidence.maximum_energy_error),
            "phase_effective_samples": float(chain.phase_effective_samples),
            "ward_residual": float(chain.ward_pfaffian.ward_residual),
        },
        "successful": successful,
        "claim": "finite-regulated-phase-quenched-reference-no-continuum-supersymmetry-claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(run_qualification(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

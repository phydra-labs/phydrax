#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A real bounded Bose-Hubbard projector, physical guide, and durable continuation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax import StrictModule
from phydrax.operators.quantum import LogAmplitude
from phydrax.operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice_columns,
    QuantumAddress,
    QuantumColumnResourcePolicy,
    QuantumConfigurationDomain,
    QuantumGuide,
    QuantumLatticeColumnOperator,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.solver import (
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    PreparedProjectorMonteCarlo,
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    read_projector_monte_carlo_checkpoint,
    solve_projector_monte_carlo,
    write_projector_monte_carlo_checkpoint,
    write_projector_monte_carlo_result,
)
from phydrax.typing import as_array, Float64, Scalar
from phydrax.units import derived_unit, HARTREE, ONE
from phydrax.uq import CorrelatedRatioPolicy, CorrelatedRatioResult


class ConstantGuide(StrictModule):
    __strict_contract__ = True

    logarithm: Float64[Scalar]

    def __init__(self, scale: float, /) -> None:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("The declared constant guide must be finite and positive.")
        self.logarithm = as_array(np.log(scale), Float64[Scalar], "logarithm")

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        del address
        return LogAmplitude(self.logarithm, jnp.asarray(1.0, dtype=jnp.complex128))


def make_problem() -> ProjectorMonteCarloProblem:
    spaces = tuple(LocalSpacePlan.boson(site, 3) for site in ("left", "right"))
    annihilation = np.asarray(
        ((0, 1, 0), (0, 0, np.sqrt(2.0)), (0, 0, 0)), dtype=np.complex128
    )
    create = tuple(
        LocalOperatorPlan(space, "create", annihilation.T, (1,)) for space in spaces
    )
    destroy = tuple(
        LocalOperatorPlan(space, "annihilate", annihilation, (-1,)) for space in spaces
    )
    pairs = tuple(
        LocalOperatorPlan(
            space,
            "pair-number",
            np.diag(np.asarray((0, 0, 1), dtype=np.complex128)),
            (0,),
        )
        for space in spaces
    )
    specification = QuantumLatticeSpecification(
        spaces,
        (
            QuantumLatticeTerm(
                (create[0], destroy[1]),
                coefficient=-1.0,
                add_adjoint=True,
                label="hopping",
            ),
            QuantumLatticeTerm((pairs[0],), coefficient=2.0, label="left-interaction"),
            QuantumLatticeTerm((pairs[1],), coefficient=2.0, label="right-interaction"),
        ),
    )
    domain = QuantumConfigurationDomain(specification, species_ids=("boson", "boson"))
    resources = QuantumColumnResourcePolicy(
        maximum_monomials=8,
        maximum_factors_per_monomial=2,
        maximum_transition_table_bytes=100_000,
        maximum_raw_routes=16,
        maximum_column_targets=8,
        maximum_workspace_bytes=1_000_000,
    )
    hamiltonian = QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(specification, resources), domain
    )
    number = LocalOperatorPlan(
        spaces[0], "left-number", np.diag(np.arange(3, dtype=np.float64)), (0,)
    )
    number_specification = QuantumLatticeSpecification(
        spaces, (QuantumLatticeTerm((number,), label="left-number-observable"),)
    )
    observable = QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(number_specification, resources), domain
    )
    initial = domain.address(np.asarray((1, 1), dtype=np.int32))
    guide = QuantumGuide(
        domain,
        ConstantGuide(3.0),
        provider_id="constant-positive-guide",
        mapping_id="packed-address-to-constant-magnitude",
        globally_positive=True,
    )
    return ProjectorMonteCarloProblem(
        hamiltonian,
        initial.key_words[None, :],
        np.asarray((1.0,), dtype=np.complex128),
        dt=0.02,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id="finite-two-site-two-boson-control",
        guide=guide,
        observables=(observable,),
        observable_units=(ONE,),
    )


def make_prepared(history_capacity: int = 256) -> PreparedProjectorMonteCarlo:
    plan = ProjectorMonteCarloPlan(
        replicas=2,
        support_capacity=3,
        group_capacity=8,
        event_capacity=2,
        attempt_capacity=64,
        source_capacity=1,
        history_capacity=history_capacity,
        maximum_retained_bytes=20_000_000,
        maximum_workspace_bytes=20_000_000,
        spawn_policy="semistochastic",
        compression="threshold",
        controller="double-log",
        target_population=8.0,
        initial_shift=0.0,
        damping=0.08,
        restoring=0.0016,
    )
    return prepare_projector_monte_carlo(make_problem(), plan)


def estimate_record(result: CorrelatedRatioResult) -> dict[str, object]:
    point = np.asarray(result.exploratory_value)
    value = None
    if np.isfinite(point):
        value = [float(np.real(point)), float(np.imag(point))]
    return {
        "exploratory_value": value,
        "statistically_valid": bool(result.statistically_valid),
        "status": int(result.status),
        "standard_error": float(result.standard_error)
        if np.isfinite(result.standard_error)
        else None,
        "systematic_bias": "not-certified-absent",
    }


def main() -> None:
    prepared = make_prepared()
    initial = initialize_projector_monte_carlo(prepared, jax.random.key(17))
    first = solve_projector_monte_carlo(prepared, initial, steps=64)
    if int(first.status) != 0:
        raise RuntimeError(
            f"Projector propagation refused with status {int(first.status)}."
        )
    with TemporaryDirectory(prefix="phydrax-projector-") as directory:
        checkpoint = write_projector_monte_carlo_checkpoint(
            Path(directory) / "accepted.phx", prepared, first.state
        )
        restored = read_projector_monte_carlo_checkpoint(checkpoint, prepared, initial)
        result = solve_projector_monte_carlo(prepared, restored, steps=64)
        uninterrupted = solve_projector_monte_carlo(prepared, initial, steps=128)
        if int(result.status) != 0 or int(uninterrupted.status) != 0:
            raise RuntimeError("The complete projector continuation did not finish.")
        for resumed, full in zip(
            jax.tree.leaves(result.state),
            jax.tree.leaves(uninterrupted.state),
            strict=True,
        ):
            if not isinstance(resumed, Array) or not isinstance(full, Array):
                raise TypeError("Committed state payloads must be native arrays.")
            left = (
                jax.random.key_data(resumed)
                if jax.dtypes.issubdtype(resumed.dtype, jax.dtypes.prng_key)
                else resumed
            )
            right = (
                jax.random.key_data(full)
                if jax.dtypes.issubdtype(full.dtype, jax.dtypes.prng_key)
                else full
            )
            if not np.array_equal(left, right):
                raise RuntimeError(
                    "Checkpoint continuation changed a committed numerical record."
                )
        analysis = analyze_projector_monte_carlo(
            prepared,
            result,
            policy=ProjectorEstimatorPolicy(
                ratio_policy=CorrelatedRatioPolicy(max_lag=32),
                burn_in=32,
                history_depths=(0, 4, 16),
            ),
        )
        archive = write_projector_monte_carlo_result(
            Path(directory) / "raw-and-statistical-result.phx",
            prepared,
            result,
            run_id="example-projector-run",
            analysis=analysis,
        )
        print(
            json.dumps(
                {
                    "completed_steps": int(result.state.step),
                    "projected_energy": estimate_record(analysis.projected),
                    "replica_energy": estimate_record(analysis.replicas[0]),
                    "left_occupation": estimate_record(analysis.replicas[1]),
                    "independent_ground_energy": float(1.0 - np.sqrt(5.0)),
                    "checkpoint_replay": "bitwise-on-this-backend",
                    "archive_id": archive.archive_id,
                    "claim": "bounded-finite-workflow-not-a-release-or-sign-cure",
                },
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Locked finite controls, not a stationary-estimator or sign-cure certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from benchmarks._runtime import capture_benchmark_identity, capture_environment
from phydrax._strict import StrictModule
from phydrax.operators.quantum import FermionModeOrder, LogAmplitude
from phydrax.operators.quantum.lattice._address import (
    QuantumAddress,
    QuantumConfigurationDomain,
)
from phydrax.operators.quantum.lattice._column import QuantumLatticeColumnOperator
from phydrax.operators.quantum.lattice._column_compile import (
    prepare_quantum_lattice_columns,
    QuantumColumnResourcePolicy,
)
from phydrax.operators.quantum.lattice._guide import QuantumGuide
from phydrax.operators.quantum.lattice._model import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.operators.quantum.lattice._qualification import (
    quantum_lattice_candidate_profiles,
)
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from phydrax.solver._projector_monte_carlo import (
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
)
from phydrax.solver._projector_monte_carlo_contracts import (
    CompressionPolicy,
    ControllerPolicy,
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStatus,
    SpawnPolicy,
)
from phydrax.solver._projector_monte_carlo_estimators import (
    analyze_projector_monte_carlo,
    ProjectorEstimatorPolicy,
)
from phydrax.solver._projector_monte_carlo_lifecycle import (
    read_projector_monte_carlo_checkpoint,
    transport_projector_monte_carlo_resources,
    write_projector_monte_carlo_checkpoint,
)
from phydrax.units import derived_unit, HARTREE


_SEEDS = (17, 29, 43)
_POPULATIONS = (4.0, 8.0, 16.0)
_STEPS = 128
_DEPTHS = (0, 4, 16)
_DTS = (0.02, 0.01)
_TOLERANCE = 2e-11
_ENERGY_TARGET = 0.25


@dataclass(frozen=True, slots=True)
class _FiniteControl:
    name: str
    specification: QuantumLatticeSpecification
    coordinates: npt.NDArray[np.int32]
    matrix: npt.NDArray[np.complex128]
    ground_energy: float


def _boson_control() -> _FiniteControl:
    spaces = tuple(LocalSpacePlan.boson(site, 3) for site in ("left", "right"))
    destroy_matrix = np.asarray(
        ((0, 1, 0), (0, 0, np.sqrt(2)), (0, 0, 0)), dtype=np.complex128
    )
    create = tuple(
        LocalOperatorPlan(space, "create", destroy_matrix.T, (1,)) for space in spaces
    )
    destroy = tuple(
        LocalOperatorPlan(space, "destroy", destroy_matrix, (-1,)) for space in spaces
    )
    pair = tuple(
        LocalOperatorPlan(
            space, "pair", np.diag(np.asarray((0, 0, 1), dtype=np.complex128)), (0,)
        )
        for space in spaces
    )
    terms = (
        QuantumLatticeTerm(
            (create[0], destroy[1]), coefficient=-1.0, add_adjoint=True, label="hop"
        ),
        QuantumLatticeTerm((pair[0],), coefficient=2.0, label="left-U"),
        QuantumLatticeTerm((pair[1],), coefficient=2.0, label="right-U"),
    )
    # Independently derived in |2,0>, |1,1>, |0,2>; no package dense lowering.
    matrix = np.asarray(
        ((2, -np.sqrt(2), 0), (-np.sqrt(2), 0, -np.sqrt(2)), (0, -np.sqrt(2), 2)),
        dtype=np.complex128,
    )
    return _FiniteControl(
        "two-boson-two-site",
        QuantumLatticeSpecification(spaces, terms),
        np.asarray(((2, 0), (1, 1), (0, 2)), dtype=np.int32),
        matrix,
        1 - np.sqrt(5),
    )


def _fermion_control() -> _FiniteControl:
    order = FermionModeOrder(("a", "b", "c"))
    spaces = tuple(LocalSpacePlan.fermion(site, site) for site in order.labels)
    matrix = np.asarray(((0, 0), (1, 0)), dtype=np.complex128)
    create = LocalOperatorPlan(spaces[0], "create", matrix, (1,))
    destroy = LocalOperatorPlan(spaces[2], "destroy", matrix.T, (-1,))
    term = QuantumLatticeTerm(
        (create, destroy), coefficient=0.75, add_adjoint=True, label="exterior-hop"
    )
    hand = np.asarray(((0, 0, -0.75), (0, 0, 0), (-0.75, 0, 0)), dtype=np.complex128)
    return _FiniteControl(
        "fermion-exterior",
        QuantumLatticeSpecification(spaces, (term,), fermion_mode_order=order),
        np.asarray(((1, 1, 0), (1, 0, 1), (0, 1, 1)), dtype=np.int32),
        hand,
        -0.75,
    )


def _flux_control() -> _FiniteControl:
    spaces = tuple(LocalSpacePlan.boson(site, 2) for site in ("a", "b", "c"))
    matrix = np.asarray(((0, 0), (1, 0)), dtype=np.complex128)
    create = tuple(LocalOperatorPlan(space, "create", matrix, (1,)) for space in spaces)
    destroy = tuple(
        LocalOperatorPlan(space, "destroy", matrix.T, (-1,)) for space in spaces
    )
    forward = -np.exp(0.4j)
    terms = tuple(
        QuantumLatticeTerm(
            (create[target], destroy[source]),
            coefficient=forward,
            add_adjoint=True,
            label=f"flux-{source}-{target}",
        )
        for source, target in ((0, 1), (1, 2), (2, 0))
    )
    hand = np.asarray(
        (
            (0, forward.conjugate(), forward),
            (forward, 0, forward.conjugate()),
            (forward.conjugate(), forward, 0),
        ),
        dtype=np.complex128,
    )
    return _FiniteControl(
        "complex-flux",
        QuantumLatticeSpecification(spaces, terms),
        np.eye(3, dtype=np.int32),
        hand,
        float(
            np.min(
                -2
                * np.cos(
                    0.4 - np.asarray((0, 2 * np.pi / 3, 4 * np.pi / 3), dtype=np.float64)
                )
            )
        ),
    )


def _column_resources() -> QuantumColumnResourcePolicy:
    return QuantumColumnResourcePolicy(
        maximum_monomials=32,
        maximum_factors_per_monomial=4,
        maximum_transition_table_bytes=1_000_000,
        maximum_raw_routes=256,
        maximum_column_targets=256,
        maximum_workspace_bytes=2_000_000,
    )


def _operator(specification: QuantumLatticeSpecification) -> QuantumLatticeColumnOperator:
    domain = QuantumConfigurationDomain(
        specification, species_ids=("particle",) * len(specification.spaces)
    )
    return QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(specification, _column_resources()), domain
    )


def _control_keys(
    operator: QuantumLatticeColumnOperator, control: _FiniteControl
) -> Array:
    return jnp.stack(
        tuple(
            operator.domain.address(coordinate).key_words
            for coordinate in control.coordinates
        )
    )


class _PositiveSnapshot(StrictModule):
    domain: QuantumConfigurationDomain
    slope: Array

    def __init__(self, domain: QuantumConfigurationDomain) -> None:
        self.domain = domain
        self.slope = jnp.asarray(0.3, dtype=jnp.float64)

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        coordinate = self.domain.decode(address.key_words)
        return LogAmplitude(
            self.slope * coordinate[0].astype(jnp.float64),
            jnp.asarray(1 + 0j, dtype=jnp.complex128),
        )


def _prepare_boson(
    *,
    seed: int = 17,
    support_capacity: int = 3,
    event_capacity: int = 4,
    group_capacity: int = 3,
    attempt_capacity: int = 128,
    source_capacity: int = 3,
    replicas: int = 2,
    history_capacity: int = 128,
    target_population: float = 8.0,
    spawn_policy: SpawnPolicy = "semistochastic",
    compression: CompressionPolicy = "threshold",
    controller: ControllerPolicy = "double-log",
    dt: float = 0.02,
    guided: bool = False,
) -> tuple[ProjectorMonteCarloProblem, PreparedProjectorMonteCarlo, Array]:
    control = _boson_control()
    operator = _operator(control.specification)
    keys = _control_keys(operator, control)
    guide = (
        QuantumGuide(
            operator.domain,
            _PositiveSnapshot(operator.domain),
            provider_id="finite-positive-snapshot",
            mapping_id="decoded-first-local-occupation",
            globally_positive=True,
        )
        if guided
        else None
    )
    # A nonexact, one-configuration trial prevents tautological E0 projection.
    problem = ProjectorMonteCarloProblem(
        operator,
        keys[1:2],
        jnp.asarray((target_population,), dtype=jnp.complex128),
        dt=dt,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id="analytic-two-boson-two-site",
        trial_keys=keys[1:2],
        trial_coefficients=jnp.asarray((1 + 0j,), dtype=jnp.complex128),
        guide=guide,
    )
    plan = ProjectorMonteCarloPlan(
        replicas=replicas,
        support_capacity=support_capacity,
        group_capacity=group_capacity,
        event_capacity=event_capacity,
        attempt_capacity=attempt_capacity,
        source_capacity=source_capacity,
        history_capacity=history_capacity,
        maximum_retained_bytes=64_000_000,
        maximum_workspace_bytes=64_000_000,
        spawn_policy=spawn_policy,
        compression=compression,
        controller=controller,
        target_population=target_population,
        relative_threshold=1.0,
        initial_shift=0.0,
    )
    return problem, prepare_projector_monte_carlo(problem, plan), jax.random.key(seed)


def _json(value: Any) -> Any:
    """Keep unavailable numerical evidence null, never fabricated finite zeros."""
    if isinstance(value, Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    ):
        return _json(jax.random.key_data(value))
    if isinstance(value, (Array, np.ndarray)):
        return _json(np.asarray(value).tolist())
    if isinstance(value, complex):
        return {"real": _json(value.real), "imag": _json(value.imag)}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    return value


def _physical_vector(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    control: _FiniteControl,
) -> npt.NDArray[np.complex128]:
    keys = np.asarray(_control_keys(prepared.original_operator, control))
    output = np.zeros((prepared.plan.replicas, len(keys)), dtype=np.complex128)
    for replica in range(prepared.plan.replicas):
        for key, coefficient, active in zip(
            np.asarray(state.support_keys[replica]),
            np.asarray(state.coefficients[replica]),
            np.asarray(state.active[replica]),
            strict=True,
        ):
            if active:
                index = np.flatnonzero(np.all(keys == key, axis=1))
                if len(index) != 1:
                    raise ValueError(
                        "Finite control escaped its conserved particle sector."
                    )
                if prepared.guide is not None:
                    logarithm, valid = prepared.guide.log_value(
                        prepared.original_operator.domain.from_key(
                            jnp.asarray(key, dtype=jnp.uint32)
                        )
                    )
                    if not bool(valid):
                        raise ValueError("Invalid encountered guide.")
                    coefficient *= np.exp(-float(logarithm))
                output[replica, index[0]] = coefficient
    return output


def _state_equal(left: ProjectorMonteCarloState, right: ProjectorMonteCarloState) -> bool:
    return _json(left) == _json(right)


def _column_reference(control: _FiniteControl) -> dict[str, Any]:
    operator = _operator(control.specification)
    keys = np.asarray(_control_keys(operator, control))
    observed = np.zeros_like(control.matrix)
    raw = np.zeros_like(control.matrix)
    proposal_probability_error = 0.0
    # These controls have at most one admissible local transition per factor.
    # The independent proposal law is uniform over physical expanded terms.
    expected_route_probability = 1 / sum(
        2 if term.add_adjoint else 1 for term in control.specification.terms
    )
    for source, coordinate in enumerate(control.coordinates):
        address = operator.domain.address(coordinate)
        column = operator.outgoing_column(address)
        for target_key, value, valid in zip(
            np.asarray(column.target_keys),
            np.asarray(column.matrix_elements),
            np.asarray(column.valid),
            strict=True,
        ):
            if valid:
                target = np.flatnonzero(np.all(keys == target_key, axis=1))
                if len(target) != 1:
                    raise ValueError(
                        "Column escaped independently declared finite sector."
                    )
                observed[target[0], source] += value
        for route_index in range(operator.raw_route_bound):
            route = operator.raw_route(address, jnp.asarray(route_index, dtype=jnp.int32))
            if bool(route.valid):
                target = np.flatnonzero(
                    np.all(keys == np.asarray(route.target_key), axis=1)
                )
                raw[target[0], source] += complex(route.matrix_element)
                proposal_probability_error = max(
                    proposal_probability_error,
                    abs(float(route.p_raw) - expected_route_probability),
                )
    column_error = float(np.max(np.abs(observed - control.matrix)))
    raw_error = float(np.max(np.abs(raw - control.matrix)))
    energy_error = abs(
        float(np.linalg.eigvalsh(control.matrix)[0]) - control.ground_energy
    )
    return {
        "case_id": control.name,
        "column_error": column_error,
        "raw_route_error": raw_error,
        "proposal_probability_error": proposal_probability_error,
        "analytic_energy_error": energy_error,
        "raw_route_bound": operator.raw_route_bound,
        "matrix": control.matrix,
        "ground_energy": control.ground_energy,
        "passed": max(column_error, raw_error, energy_error, proposal_probability_error)
        <= _TOLERANCE,
    }


def _address_control() -> dict[str, Any]:
    spaces = tuple(LocalSpacePlan.boson(f"mode-{index}", 2) for index in range(100))
    occupation = LocalOperatorPlan(
        spaces[99], "occupation", np.diag(np.asarray((0, 1), dtype=np.complex128)), (0,)
    )
    operator = _operator(
        QuantumLatticeSpecification(
            spaces,
            (QuantumLatticeTerm((occupation,), coefficient=2.0, label="tail-diagonal"),),
        )
    )
    coordinate = np.zeros(100, dtype=np.int32)
    coordinate[99] = 1
    address = operator.domain.address(coordinate)
    column = operator.outgoing_column(address)
    decoded = np.asarray(operator.domain.decode(address.key_words))
    return {
        "case_id": "100-mode-rank-free-query",
        "mode_count": 100,
        "word_count": operator.domain.codec.word_count,
        "key": address.key_words,
        "roundtrip": bool(np.array_equal(decoded, coordinate)),
        "diagonal": column.diagonal,
        "passed": bool(np.array_equal(decoded, coordinate))
        and abs(complex(column.diagonal) - 2) <= _TOLERANCE
        and bool(column.successful),
        "scope": "bounded address and column query only; not a solution of the 100-mode model",
    }


def _replay_control() -> dict[str, Any]:
    _, prepared, key = _prepare_boson(history_capacity=16)
    initial = initialize_projector_monte_carlo(prepared, key)
    uninterrupted = solve_projector_monte_carlo(prepared, initial, steps=8)
    repeated = solve_projector_monte_carlo(prepared, initial, steps=8)
    prefix = solve_projector_monte_carlo(prepared, initial, steps=4)
    with TemporaryDirectory(prefix="phydrax-projector-") as directory:
        path = Path(directory) / "checkpoint.npz"
        write_projector_monte_carlo_checkpoint(path, prepared, prefix.state)
        restored = read_projector_monte_carlo_checkpoint(path, prepared, initial)
        resumed = solve_projector_monte_carlo(prepared, restored, steps=4)
    _, small, _ = _prepare_boson(
        support_capacity=1, group_capacity=1, history_capacity=16
    )
    small_initial = initialize_projector_monte_carlo(small, key)
    refused = step_projector_monte_carlo(small, small_initial)
    enlarged, relation = transport_projector_monte_carlo_resources(
        small, prepared, refused.state
    )
    replay = step_projector_monte_carlo(prepared, enlarged)
    direct = step_projector_monte_carlo(prepared, initial)
    unchanged = _state_equal(refused.state, small_initial)
    comparisons = {
        "repeat": _state_equal(uninterrupted.state, repeated.state),
        "checkpoint": _state_equal(uninterrupted.state, resumed.state),
        "refusal_unchanged": unchanged,
        "resource_replay": _state_equal(replay.state, direct.state),
    }
    return {
        "case_id": "same-draw-lifecycle",
        "comparisons": comparisons,
        "refusal_status": refused.status,
        "refusal_evidence": refused.evidence,
        "restart_relation": {
            "relation_id": relation.relation_id,
            "source_topology_id": relation.source_topology_id,
            "target_topology_id": relation.target_topology_id,
            "classification": relation.classification,
        },
        "direct_status": direct.status,
        "passed": all(comparisons.values())
        and int(refused.status)
        in (
            int(ProjectorMonteCarloStatus.INTERMEDIATE_GROUP_OVERFLOW),
            int(ProjectorMonteCarloStatus.FINAL_SUPPORT_OVERFLOW),
        )
        and int(uninterrupted.status) == 0
        and int(replay.status) == 0,
    }


def _exact_control(*, guided: bool) -> dict[str, Any]:
    control = _boson_control()
    _, prepared, key = _prepare_boson(
        spawn_policy="exact",
        compression="none",
        controller="fixed-shift",
        guided=guided,
        history_capacity=16,
    )
    state = initialize_projector_monte_carlo(prepared, key)
    expected = _physical_vector(prepared, state, control)
    result = solve_projector_monte_carlo(prepared, state, steps=12)
    for _ in range(12):
        expected = (
            expected
            @ (np.eye(3, dtype=np.complex128) - prepared.problem.dt * control.matrix).T
        )
    actual = _physical_vector(prepared, result.state, control)
    vector_error = float(np.max(np.abs(actual - expected)))
    pair_denominator = np.vdot(actual[1], actual[0])
    pair_numerator = np.vdot(actual[1], control.matrix @ actual[0])
    h = result.state.history
    metric_error = max(
        abs(complex(h.pair_denominators[0, 11]) - pair_denominator),
        abs(complex(h.pair_numerators[0, 11, 0]) - pair_numerator),
    )
    projected = (control.matrix @ actual[0])[1] / actual[0, 1]
    projected_error = abs(
        complex(h.projected_numerator[0, 11] / h.projected_denominator[0, 11]) - projected
    )
    return {
        "case_id": "guided-physical-control" if guided else "independent-Euler-control",
        "vector_error": vector_error,
        "physical_metric_error": metric_error,
        "projected_error": projected_error,
        "physical_energy": pair_numerator / pair_denominator,
        "ground_energy_error": abs(
            pair_numerator / pair_denominator - control.ground_energy
        ),
        "status": result.status,
        "passed": int(result.status) == 0
        and max(vector_error, metric_error, projected_error) <= _TOLERANCE,
    }


def _stochastic_case(
    seed: int, population: float, dt: float, spawn: SpawnPolicy
) -> dict[str, Any]:
    control = _boson_control()
    _, prepared, key = _prepare_boson(
        seed=seed,
        target_population=population,
        dt=dt,
        spawn_policy=spawn,
        history_capacity=_STEPS,
    )
    initial = initialize_projector_monte_carlo(prepared, key)
    result = solve_projector_monte_carlo(prepared, initial, steps=_STEPS)
    count = int(result.state.history.count)
    analysis = analyze_projector_monte_carlo(
        prepared,
        result,
        policy=ProjectorEstimatorPolicy(
            burn_in=min(32, count),
            history_depths=_DEPTHS,
            reference_energy=control.ground_energy,
        ),
    )
    exploratory = complex(analysis.projected.exploratory_value)
    error = abs(exploratory - control.ground_energy) if np.isfinite(exploratory) else None
    standard_error = float(analysis.projected.standard_error)
    covered = (
        abs(exploratory - control.ground_energy) <= 1.96 * standard_error
        if bool(analysis.projected.statistically_valid) and math.isfinite(standard_error)
        else None
    )
    h = result.state.history
    return {
        "case_id": f"{spawn}-population-{population:g}-dt-{dt:g}-seed-{seed}",
        "seed": seed,
        "target_population": population,
        "dt": dt,
        "spawn_policy": spawn,
        "physical_raw_route_bound": prepared.original_operator.raw_route_bound,
        "accepted_steps": result.accepted_steps,
        "status": result.status,
        "last_evidence": result.evidence,
        "state": result.state,
        "raw_applied_shifts": h.applied_shifts[:, :count],
        "raw_projected_numerator": h.projected_numerator[:, :count],
        "raw_projected_denominator": h.projected_denominator[:, :count],
        "raw_pair_numerators": h.pair_numerators[:, :count, :],
        "raw_pair_denominators": h.pair_denominators[:, :count],
        "analysis": analysis,
        "independent_energy_error": error,
        "statistical_coverage": covered,
        "statistical_status": "qualified"
        if bool(analysis.projected.statistically_valid)
        else "unresolved",
        "systematic_status": "finite-population-finite-history-not-certified-unbiased",
        "matched_error_target": _ENERGY_TARGET,
        "matches_error_target": error is not None and error <= _ENERGY_TARGET,
        "source_exercise": {
            "exact": result.evidence.exact_sources,
            "sampled": result.evidence.sampled_sources,
        },
    }


def run(output: Path) -> None:
    finite_controls = (_boson_control(), _fermion_control(), _flux_control())
    controls = [_column_reference(control) for control in finite_controls]
    controls.extend(
        (
            _address_control(),
            _exact_control(guided=False),
            _exact_control(guided=True),
            _replay_control(),
        )
    )
    cases = [
        _stochastic_case(seed, population, dt, spawn)
        for spawn in ("sampled", "semistochastic")
        for dt in _DTS
        for population in _POPULATIONS
        for seed in _SEEDS
    ]
    reference_artifact = [
        {
            "name": control.name,
            "coordinates": control.coordinates,
            "matrix": control.matrix,
            "ground_energy": control.ground_energy,
        }
        for control in finite_controls
    ]
    artifact = json.dumps(
        _json(reference_artifact), sort_keys=True, allow_nan=False
    ).encode()
    manifest = ReferenceArtifactManifest(
        "independent-hand-finite-controls",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(artifact).hexdigest(),
        size_bytes=len(artifact),
        license_id="PHYDRA-proprietary",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="internal-finite-mathematical-reference",
        nondimensionalization={"hartree": 1.0},
        uncertainty=None,
        lineage_ids=("hand-boson-exterior-flux-equations",),
    )
    calibration = ScientificCase(
        "finite-invariants",
        "independent-hand-equations",
        "finite-controls",
        "bounded-column-and-Euler",
        "finite-reference-preparation",
        "deterministic-controls",
        (manifest.manifest_id,),
    )
    locked = tuple(
        ScientificCase(
            case["case_id"],
            f"root-seed-{case['seed']}",
            "two-boson-two-site",
            f"{case['spawn_policy']}-dt-{case['dt']}-population-{case['target_population']}",
            "locked-stochastic-preparation",
            "locked-stochastic-draws",
            (manifest.manifest_id,),
        )
        for case in cases
    )
    campaign = ScientificCampaign(
        (calibration, *locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", tuple(case.case_id for case in locked)),
        ),
        criteria_ids=("finite-invariant-tolerance", "retain-all-locked-draws"),
    )
    exercised = {
        role: any(
            int(np.sum(case["source_exercise"][role])) > 0
            for case in cases
            if case["spawn_policy"] == "semistochastic"
        )
        for role in ("exact", "sampled")
    }
    payload = {
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parents[1],
            Path(__file__),
            ("controls", "cases", "finite_workflow_qualified"),
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "campaign": campaign.to_record(),
        "reference": manifest.to_record(),
        "candidate_profiles": [
            profile
            for profile in quantum_lattice_candidate_profiles()
            if "projector" in profile.capability
        ],
        "locked_policy": {
            "seeds": _SEEDS,
            "populations": _POPULATIONS,
            "steps": _STEPS,
            "history_depths": _DEPTHS,
            "dt": _DTS,
            "burn_in": 32,
            "invariant_tolerance": _TOLERANCE,
            "energy_target": _ENERGY_TARGET,
            "no_resampling": True,
        },
        "controls": controls,
        "cases": cases,
        "semistochastic_sources_exercised": exercised,
        "finite_workflow_qualified": all(control["passed"] for control in controls)
        and all(exercised.values())
        and all(int(case["status"]) == 0 for case in cases),
        "claim_scope": "Declared finite workflow/invariants only. Statistical and systematic failures remain in cases; no release, generic sign cure, or all-ratios-unbiased claim.",
        "unavailable_numeric_evidence": "Nonfinite or statistically refused qualified values are null; native statuses and exploratory values retained.",
    }
    payload["reference_artifact_utf8"] = artifact.decode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "finite_workflow_qualified": payload["finite_workflow_qualified"],
                "locked_cases": len(cases),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

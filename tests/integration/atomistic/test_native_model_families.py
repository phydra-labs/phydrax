"""End-to-end native model-family workflows: MACE, MACELES, AIMNet2 and UMA.

Every model is a random native model or a native reconstruction of a random
source-layout fixture. These tests establish native consumer behavior only:
conservative E/F/S against independent finite-difference, translation and
strain controls; canonical training and restart continuation; predicted-charge
and latent electrostatics composed through native owners; prepared-form
staleness; and declared derivative/property refusals. They make no trained
accuracy, pretrained-weight or source-provider-equivalence claim.

Source-family prerequisites outside this file stay explicit: pretrained AIMNet2
ensemble members, gated UMA weights and terms, and MACE ASL-bearing checkpoints
are not exercised here and are not replaced by these random fixtures.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._differentiation import (
    ComponentAuthority,
    DERIVATIVE_UNSUPPORTED,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_model_revision,
    AtomisticBatch,
    AtomisticCellListImageSearch,
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticGraphExecutionPlan,
    AtomisticGraphInputs,
    AtomisticGraphLayout,
    AtomisticNeighborhoodRole,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticPotentialProgram,
    AtomisticSystemPlan,
    AtomisticTrainingPolicy,
    energy_and_forces,
    fit_atomistic_potential,
    LearnedGraphPotentialTerm,
    ParticleMeshEwaldPotential,
    read_atomistic_model_artifact,
    read_atomistic_training_restart,
    write_atomistic_model_artifact,
    write_atomistic_training_restart,
)
from phydrax.atomistic._hessian import (
    ATOMISTIC_HESSIAN_REQUEST,
    prepare_atomistic_hessian,
)
from phydrax.atomistic._latent_electrostatics import LatentEwaldEvaluation
from phydrax.discretization import ParticleImageCapacity, PeriodicCell
from phydrax.nn.atomistic import (
    prepare_mace_potential,
    StaleUMAMergeError,
    UMAPotential,
    UnsupportedMACEPreparedRoute,
)
from phydrax.units import DALTON
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_long_range_reference import central_difference
from tests._support.atomistic_model_families import (
    aimnet2_model,
    CHARGE_STATES,
    energy_oracle,
    FAMILIES,
    family_scenario,
    finite_execution,
    MACE_CUTOFF,
    mace_model,
    NEUTRAL,
    periodic_execution,
    SCALE,
    training_problem,
    uma_binding,
    uma_model,
    uma_property_model,
    UNITS,
    water_batch,
    water_box,
    water_molecules,
)
from tests._support.mace_deployment import strained_water_problem, tiny_mace


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
REPOSITORY = Path(__file__).resolve().parents[3]
DIMER = water_molecules(2, seed=1)
# Two waters whose O-O separation lies inside the 2.5 A UMA fixture cutoff:
# every O has three eligible routes, so a one-neighbor cap, whose FAIRChem
# degeneracy band keeps the two nearest, must drop the other O. In DIMER each
# receiver has only its two intramolecular routes and no cap can remove one.
CONTACT = water_molecules(2, seed=1, spacing=2.2)
CELL, BOX = water_box(1, seed=2)
ZERO = jnp.zeros((3, 3), dtype=jnp.float64)
STEP = 1.0e-5
AE = RegularityPolicy(allow_almost_everywhere=True)
MIXED = DifferentiationRequest(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER),
    order=3,
    authority=ComponentAuthority.MODEL,
)
# The physical (total charge, multiplicity) each family is evaluated in: an
# unconditioned or single-channel family only represents the neutral singlet.
FINITE_STATE = {
    "mace": NEUTRAL,
    "maceles": NEUTRAL,
    "aimnet2": NEUTRAL,
    "aimnet2-nse": (1, 2),
    "uma": (1, 2),
}
SOURCE_PACKAGES = ("torch", "aimnet", "fairchem", "e3nn", "mace")


def _close(observed: Any, expected: Any, relative: float) -> None:
    observed_ = np.asarray(observed, dtype=np.float64)
    expected_ = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected_))))
    np.testing.assert_allclose(observed_, expected_, rtol=0.0, atol=relative * scale)


def _finite(
    name: str, *, seed: int = 3, model: Any = None, positions: np.ndarray = DIMER
) -> tuple[Any, ...]:
    batch = water_batch(positions[None])
    scenario = family_scenario(
        name, periodic=False, atom_capacity=batch.atom_capacity, seed=seed, model=model
    )
    execution = finite_execution(batch.atom_capacity)
    conditioning = scenario.conditioning(batch, (FINITE_STATE[name],))
    return scenario, batch, execution, conditioning


def _periodic(name: str) -> tuple[Any, ...]:
    batch = water_batch(BOX[None], cells=CELL[None])
    scenario = family_scenario(
        name, periodic=True, atom_capacity=batch.atom_capacity, cell=CELL
    )
    return scenario, batch, periodic_execution(), scenario.conditioning(batch, (NEUTRAL,))


def _inputs(model: Any, batch: AtomisticBatch, execution: Any, conditioning: Any) -> Any:
    cutoff = float(model.configuration.cutoff)
    payload = AtomisticGraphLayout.for_batch(batch, execution, cutoff=cutoff).prepare(
        batch
    )
    graph = payload.graph(
        cutoff=cutoff,
        selection=model.requirements.neighborhood(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        ).selection,
    )
    return AtomisticGraphInputs(
        graph,
        batch.scale,
        species_kind=model.capabilities.species_kind,
        conditioning=conditioning,
    )


def _leaves(model: Any) -> list[np.ndarray]:
    return [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    ]


# ---------------------------------------------------------------- E/F/S


@pytest.mark.parametrize("name", FAMILIES)
def test_finite_forces_are_the_negative_energy_gradient_and_translation_invariant(
    name: str,
) -> None:
    scenario, batch, execution, conditioning = _finite(name)
    prediction = energy_and_forces(
        scenario.model, batch, execution, conditioning=conditioning
    )
    assert bool(prediction.valid[0])
    forces = np.asarray(prediction.forces[0])
    assert float(np.max(np.abs(forces))) > 1e-6
    energy = energy_oracle(scenario.model, batch, execution, conditioning)
    reference = float(prediction.energy[0])
    np.testing.assert_allclose(
        float(energy(jnp.asarray(batch.positions), ZERO)),
        reference,
        rtol=1e-12,
        atol=1e-10,
    )
    numerical = central_difference(
        lambda point: float(energy(jnp.asarray(point)[None], ZERO)), DIMER, STEP
    )
    _close(forces, -numerical, 1e-6)
    shifted = jnp.asarray(batch.positions) + jnp.asarray([0.37, -0.21, 0.53])
    np.testing.assert_allclose(
        float(energy(shifted, ZERO)), reference, rtol=1e-11, atol=1e-10
    )
    _close(
        forces.sum(axis=0), np.zeros(3), 1e-9 * max(1.0, float(np.max(np.abs(forces))))
    )


@pytest.mark.parametrize("name", ("mace", "maceles", "aimnet2", "uma"))
def test_periodic_stress_is_the_volume_normalized_strain_derivative(name: str) -> None:
    scenario, batch, execution, conditioning = _periodic(name)
    prediction = energy_and_forces(
        scenario.model, batch, execution, compute_stress=True, conditioning=conditioning
    )
    assert bool(prediction.valid[0]) and bool(prediction.stress_available[0])
    assert prediction.stress is not None
    stress = np.asarray(prediction.stress[0])
    np.testing.assert_allclose(stress, stress.T, atol=1e-10)
    forces = np.asarray(prediction.forces[0])
    energy = energy_oracle(scenario.model, batch, execution, conditioning)
    positions = jnp.asarray(batch.positions)
    np.testing.assert_allclose(
        energy(positions, ZERO), float(prediction.energy[0]), rtol=1e-12, atol=1e-10
    )
    numerical = central_difference(
        lambda point: float(energy(jnp.asarray(point)[None], ZERO)), BOX, STEP
    )
    _close(forces, -numerical, 1e-6)
    _close(
        forces.sum(axis=0), np.zeros(3), 1e-9 * max(1.0, float(np.max(np.abs(forces))))
    )
    volume = abs(float(np.linalg.det(CELL)))
    for a in range(3):
        for b in range(a, 3):
            direction = np.zeros((3, 3))
            direction[a, b] = direction[b, a] = 1.0
            difference = (
                float(energy(positions, jnp.asarray(STEP * direction)))
                - float(energy(positions, jnp.asarray(-STEP * direction)))
            ) / (2.0 * STEP)
            expected = volume * stress[a, b] * (1.0 if a == b else 2.0)
            np.testing.assert_allclose(
                difference, expected, rtol=1e-6, atol=1e-6 * max(1.0, abs(expected))
            )


# ------------------------------------------------------------- training


def _training_case(name: str) -> tuple[Any, Any]:
    """A problem and a factory of identical fresh students of one family."""
    if name == "mace":
        return strained_water_problem(), lambda: tiny_mace(seed=4)
    positions = np.stack([water_molecules(2, seed=10 + case) for case in range(3)])
    batch = water_batch(positions, prefix=f"{name}-train")
    execution = finite_execution(batch.atom_capacity)
    teacher = family_scenario(
        name, periodic=False, atom_capacity=batch.atom_capacity, seed=5
    )
    problem = training_problem(teacher, batch, execution, CHARGE_STATES, prefix=name)
    return (
        problem,
        lambda: (
            family_scenario(
                name, periodic=False, atom_capacity=batch.atom_capacity, seed=3
            ).model
        ),
    )


@pytest.mark.parametrize("name", ("mace", "aimnet2-nse", "uma"))
def test_training_step_continues_through_the_canonical_restart(
    name: str, tmp_path: Path
) -> None:
    problem, student = _training_case(name)
    saved = AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1.0e-3)
    total = AtomisticTrainingPolicy(maximum_steps=3, learning_rate=1.0e-3)
    initial = atomistic_model_revision(student()).revision_id
    partial = fit_atomistic_potential(student(), problem, saved, key=jr.key(5))
    assert bool(partial.successful)
    values = np.asarray(partial.training_values)
    assert values.shape == (2,) and np.all(np.isfinite(values)) and values[0] > 0.0
    trained = atomistic_model_revision(partial.potential).revision_id
    assert trained != initial

    directory = write_atomistic_training_restart(
        tmp_path / "training-restart", partial, problem, saved
    )
    artifact, restored = read_atomistic_training_restart(directory, problem, saved)
    assert type(artifact.model) is type(partial.potential)
    assert artifact.manifest.numeric_revision.revision_id == trained
    continued = fit_atomistic_potential(
        artifact.model, problem, total, key=jr.key(5), continuation=restored
    )
    uninterrupted = fit_atomistic_potential(student(), problem, total, key=jr.key(5))
    assert bool(continued.successful) and bool(uninterrupted.successful)
    # Bitwise replay needs ordered accumulation; agreement is asserted numerically.
    np.testing.assert_allclose(
        np.asarray(continued.training_values),
        np.asarray(uninterrupted.training_values),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(continued.training_objectives),
        np.asarray(uninterrupted.training_objectives),
        rtol=1e-9,
        atol=1e-12,
    )
    for observed, expected in zip(
        _leaves(continued.potential), _leaves(uninterrupted.potential), strict=True
    ):
        np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-12)


def test_mace_hvp_training_runs_only_under_explicit_almost_everywhere_admission(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")
    positions = np.stack([water_molecules(2, seed=20 + case) for case in range(3)])
    batch = water_batch(positions, prefix="hvp")
    execution = finite_execution(batch.atom_capacity)
    probe = phx.atomistic.AtomisticHessianProbe("rademacher", count=1, seed=7)
    labeled = phx.atomistic.label_atomistic_structures(
        phx.atomistic.AtomisticTeacher(mace_model(1), name="random-teacher"),
        phx.atomistic.atomistic_training_loader(
            phx.atomistic.atomistic_training_source(batch, mass_unit=DALTON),
            shuffle=False,
        ),
        phx.atomistic.AtomisticTrainingFields(),
        phx.atomistic.AtomisticLabelRequest(hvp=probe),
        graph_execution=execution,
        cutoff=MACE_CUTOFF,
        store=phx.atomistic.AtomisticStructureStoreWriter(
            tmp_path / "teacher-labels",
            layout=phx.atomistic.AtomisticStoreLayout(
                records_per_shard=2, target_chunk_bytes=1 << 16
            ),
            limits=phx.atomistic.AtomisticStoreLimits(
                max_commit_bytes=1 << 20,
                max_member_bytes=1 << 24,
                max_shard_bytes=1 << 26,
                max_shard_members=1 << 14,
                max_chunk_bytes=1 << 20,
                max_index_bytes=1 << 22,
                max_decoded_bytes=1 << 26,
                max_read_bytes=1 << 26,
                max_records=1024,
            ),
        ),
    )

    def problem(policy: RegularityPolicy | None) -> Any:
        return phx.atomistic.AtomisticTrainingProblem(
            phx.atomistic.atomistic_training_loader(labeled, shuffle=False),
            execution,
            (
                phx.atomistic.AtomisticEnergyObjective(
                    "teacher_energy", name="energy", scale=1.0
                ),
                phx.atomistic.AtomisticForceObjective(
                    "teacher_forces", name="forces", scale=1.0
                ),
                phx.atomistic.AtomisticHessianVectorObjective(
                    probe, policy=policy, scale=1.0
                ),
            ),
            scale=SCALE,
            cutoff=MACE_CUTOFF,
        )

    policy = AtomisticTrainingPolicy(maximum_steps=1, learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="almost-everywhere"):
        fit_atomistic_potential(mace_model(3), problem(None), policy, key=jr.key(2))
    result = fit_atomistic_potential(mace_model(3), problem(AE), policy, key=jr.key(2))
    assert bool(result.successful)
    objectives = np.asarray(result.training_objectives)
    assert objectives.shape == (1, 3) and np.all(np.isfinite(objectives))
    assert objectives[0, 2] > 0.0


# ------------------------------------------------- composed electrostatics


PME_CELL = np.asarray([[3.6, 0.0, 0.0], [0.3, 3.7, 0.0], [0.2, 0.1, 3.8]])
PME_NUMBERS = np.asarray([8, 1, 1, 6])
PME_POSITIONS = (
    np.asarray(
        [[0.12, 0.2, 0.31], [0.38, 0.27, 0.33], [0.55, 0.61, 0.18], [0.81, 0.74, 0.66]]
    )
    @ PME_CELL
)
PME_EXECUTION = AtomisticGraphExecutionPlan(
    256,
    backend="particle",
    image_capacity=ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=8192,
        maximum_degree=256,
        maximum_images=343,
    ),
    search_cell=PME_CELL,
    search_margin=0.05,
)
PME_RESOURCES = {
    "search": AtomisticCellListImageSearch(maximum_candidate_slots=1 << 16),
    "skin": 0.3,
    "search_capacities": (search_capacity(4096, particles_per_cell=32, images=343),),
    "role_capacities": (role_capacity(4096, 256),),
}


def _pme(
    charge_field: AtomisticFieldReference | None = None,
) -> ParticleMeshEwaldPotential:
    # The minimum-image real-space cutoff stays below half the cell width.
    # AIMNet2 equilibrates its charges to the requested total only up to the
    # source NSE epsilon (`CHARGE_RESIDUAL`), so the cell carries that residual
    # net charge and PME declares the uniform neutralizing background.
    return ParticleMeshEwaldPotential(
        2.2,
        1.6,
        (16, 16, 16),
        spline_degree=4,
        neutrality="uniform-background",
        charge_field=charge_field,
    )


def _pme_system(charges: Any) -> Any:
    return AtomisticSystemPlan(
        np.arange(len(PME_NUMBERS), dtype=np.int64),
        PME_NUMBERS,
        np.ones(len(PME_NUMBERS)),
        UNITS,
        charges=np.asarray(charges, dtype=np.float64),
        cell=PeriodicCell(PME_CELL),
    ).prepare()


def test_predicted_charges_drive_the_connected_pme_energy_and_forces() -> None:
    model = aimnet2_model(3)
    batch = AtomisticBatch.from_structure(
        AtomicStructure(
            PME_NUMBERS,
            PME_POSITIONS,
            np.ones(len(PME_NUMBERS)),
            SCALE,
            cell=PME_CELL,
            periodic_axes=np.asarray([True, True, True]),
        )
    )

    def conditioning(capacity: int) -> AtomisticConditioning:
        return AtomisticConditioning(
            model.conditioning_spec,
            {"charge": np.asarray([0.0])},
            case_count=1,
            atom_capacity=capacity,
        )

    outputs = model.evaluate_graph(
        _inputs(model, batch, PME_EXECUTION, conditioning(batch.atom_capacity)),
        AtomisticObservableRequest(
            (
                AtomisticObservable.ENERGY,
                AtomisticObservable.CHARGES,
                AtomisticObservable.CHARGE_RESIDUAL,
            )
        ),
    )
    predicted = outputs.observable(AtomisticObservable.CHARGES)
    charges = np.asarray(predicted.values)
    assert float(np.max(np.abs(charges))) > 1e-6
    # The neutral request leaves exactly the published residual as net charge,
    # beyond any neutrality admission threshold: the background is required.
    residual = np.asarray(outputs.observable(AtomisticObservable.CHARGE_RESIDUAL).values)
    np.testing.assert_allclose(charges.sum() + residual.ravel()[0], 0.0, atol=1e-14)
    assert abs(float(charges.sum())) > _pme().charge_tolerance
    field = AtomisticFieldReference(
        "aimnet2",
        AtomisticObservable.CHARGES,
        predicted.port,
        level=AtomisticConditionLevel.ATOM,
        unit=UNITS.charge_unit,
    )
    system = _pme_system(np.zeros(len(PME_NUMBERS)))
    connected = prepare_program(
        AtomisticPotentialProgram(
            [
                LearnedGraphPotentialTerm(
                    model,
                    name="aimnet2",
                    allow_periodic=True,
                    conditioning=conditioning(system.capacity),
                ),
                _pme(field),
            ]
        ),
        system,
        graph_execution=PME_EXECUTION,
        **PME_RESOURCES,
    )
    evaluation = connected.evaluate(
        jnp.asarray(PME_POSITIONS), bundle_state(connected, PME_POSITIONS)
    )
    assert bool(evaluation.successful)

    alone = energy_and_forces(
        model, batch, PME_EXECUTION, conditioning=conditioning(batch.atom_capacity)
    )
    fixed = prepare_program(
        AtomisticPotentialProgram([_pme()]), _pme_system(charges), **PME_RESOURCES
    )
    held = fixed.evaluate(jnp.asarray(PME_POSITIONS), bundle_state(fixed, PME_POSITIONS))
    assert bool(alone.valid[0]) and bool(held.successful)
    terms = np.asarray(evaluation.term_energies)
    np.testing.assert_allclose(terms[0], float(alone.energy[0]), rtol=1e-10)
    # The PME term reads exactly the model's predicted charges, not system charges.
    np.testing.assert_allclose(terms[1], float(held.energy), rtol=1e-10, atol=1e-12)
    # dq/dx enters the forces: they are not the fixed-charge composition.
    response = np.asarray(evaluation.forces) - (
        np.asarray(alone.forces[0]) + np.asarray(held.forces)
    )
    assert float(np.max(np.abs(response))) > 1e-6

    def total(point: np.ndarray) -> float:
        energy, auxiliary = connected.energy(
            jnp.asarray(point), bundle_state(connected, point)
        )
        assert bool(auxiliary.successful), "A PME energy probe was not successful."
        return float(energy)

    for atom, axis in ((0, 0), (2, 1), (3, 2)):
        shift = np.zeros_like(PME_POSITIONS)
        shift[atom, axis] = STEP
        difference = (total(PME_POSITIONS + shift) - total(PME_POSITIONS - shift)) / (
            2.0 * STEP
        )
        np.testing.assert_allclose(
            float(evaluation.forces[atom, axis]), -difference, rtol=1e-6, atol=1e-7
        )


@pytest.mark.parametrize("periodic", [False, True], ids=["finite", "periodic"])
def test_maceles_latent_energy_is_global_and_moves_conservative_forces(
    periodic: bool,
) -> None:
    scenario, batch, execution, conditioning = (
        _periodic("maceles") if periodic else _finite("maceles")
    )
    model = scenario.model
    prediction = energy_and_forces(model, batch, execution, compute_stress=periodic)
    local = energy_and_forces(model.mace, batch, execution, compute_stress=periodic)
    assert bool(prediction.valid[0]) and bool(local.valid[0])
    outputs = model.evaluate_graph(
        _inputs(model, batch, execution, conditioning),
        AtomisticObservableRequest(
            (
                AtomisticObservable.ENERGY,
                AtomisticObservable.ATOM_ENERGY,
                AtomisticObservable.GLOBAL_ENERGY_COMPONENT,
                AtomisticObservable.LATENT_CHARGES,
            )
        ),
    )
    ledger = outputs.observable(AtomisticObservable.GLOBAL_ENERGY_COMPONENT)
    latent = float(np.ravel(np.asarray(ledger.values))[0])
    assert abs(latent) > 1e-8
    np.testing.assert_allclose(
        float(prediction.energy[0]),
        float(local.energy[0]) + latent,
        rtol=1e-12,
        atol=1e-10,
    )
    # Atomic shares exist only for the local MACE energy, never for the latent sum.
    assert not outputs.atom_energy_complete
    np.testing.assert_allclose(
        float(jnp.sum(outputs.atom_energy[0])),
        float(local.energy[0]),
        rtol=1e-12,
        atol=1e-10,
    )
    evidence = outputs.evidence[0]
    assert isinstance(evidence, LatentEwaldEvaluation)
    assert bool(evidence.periodic[0]) == periodic
    if periodic:
        assert bool(evidence.mode_coverage_successful[0])
    assert float(np.max(np.abs(np.asarray(prediction.forces - local.forces)))) > 1e-6
    # Learned latent channels are not physical partial charges.
    with pytest.raises(ValueError, match="unavailable"):
        outputs.observable(AtomisticObservable.CHARGES)
    with pytest.raises(UnsupportedMACEPreparedRoute):
        prepare_mace_potential(model)


# ------------------------------------------------- prepared forms, UMA rules


def test_uma_merged_realization_is_stale_after_training_and_reprepares() -> None:
    problem, student = _training_case("uma")
    raw = student()
    scenario, batch, execution, conditioning = _finite("uma", model=raw)
    neutral = scenario.conditioning(batch, (NEUTRAL,))
    binding = uma_binding(raw, batch, periodic=False)
    merged = raw.merged(binding)
    merged.validate()
    exact = energy_and_forces(raw, batch, execution, conditioning=neutral)
    bound = energy_and_forces(merged, batch, execution, conditioning=neutral)
    assert bool(exact.valid[0]) and bool(bound.valid[0])
    np.testing.assert_allclose(bound.energy, exact.energy, rtol=1e-11, atol=1e-10)
    # Conditions outside the merged binding are not silently evaluated.
    other = merged.evaluate_graph(_inputs(merged, batch, execution, conditioning))
    assert not bool(other.successful[0])

    result = fit_atomistic_potential(
        raw,
        problem,
        AtomisticTrainingPolicy(maximum_steps=1, learning_rate=1.0e-2),
        key=jr.key(1),
    )
    assert bool(result.successful)
    trained = result.potential
    assert isinstance(trained, UMAPotential)
    merged.require_current(raw)
    with pytest.raises(StaleUMAMergeError):
        merged.require_current(trained)
    refreshed = trained.merged(binding)
    assert refreshed.prepared_id != merged.prepared_id
    assert refreshed.source_revision_id == atomistic_model_revision(trained).revision_id
    current = energy_and_forces(trained, batch, execution, conditioning=neutral)
    rebound = energy_and_forces(refreshed, batch, execution, conditioning=neutral)
    np.testing.assert_allclose(rebound.energy, current.energy, rtol=1e-11, atol=1e-10)
    _close(rebound.forces, current.forces, 1e-10)
    assert abs(float(current.energy[0] - bound.energy[0])) > 1e-10


def test_uma_nearest_neighbor_cap_is_part_of_the_scientific_model() -> None:
    capped = uma_model(3, ff_type="spectral", max_neighbors=1)
    full = uma_model(3, ff_type="spectral")

    def selection(model: Any) -> Any:
        return model.requirements.neighborhood(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        ).selection

    assert selection(capped).policy_id != selection(full).policy_id
    results = []
    for model in (capped, full):
        scenario, batch, execution, conditioning = _finite(
            "uma", model=model, positions=CONTACT
        )
        evidence = _inputs(model, batch, execution, conditioning).graph.selection_evidence
        assert bool(evidence.successful[0])
        results.append(
            (
                energy_and_forces(model, batch, execution, conditioning=conditioning),
                energy_oracle(model, batch, execution, conditioning),
                evidence,
            )
        )
    (short, energy, cut), (complete, _, kept) = results
    # The same eligible routes: the full model keeps all, the cap removes some.
    assert int(cut.eligible_routes[0]) == int(kept.eligible_routes[0])
    assert int(kept.capped_receivers[0]) == 0
    assert int(kept.selected_routes[0]) == int(kept.eligible_routes[0])
    assert int(cut.capped_receivers[0]) > 0
    assert int(cut.selected_routes[0]) < int(cut.eligible_routes[0])
    assert bool(short.valid[0]) and bool(complete.valid[0])
    assert abs(float(short.energy[0] - complete.energy[0])) > 1e-8
    # Selection is fixed at the geometry; derivatives never pass through it.
    numerical = central_difference(
        lambda point: float(energy(jnp.asarray(point)[None], ZERO)), CONTACT, STEP
    )
    _close(np.asarray(short.forces[0]), -numerical, 1e-6)


def test_property_only_uma_reports_without_energy_and_refuses_conservative_work() -> None:
    model = uma_property_model()
    batch = water_batch(DIMER[None])
    execution = finite_execution(batch.atom_capacity)
    conditioning = model.schema.conditioning(
        ["omol"], atom_capacity=batch.atom_capacity, charges=[0], spins=[1]
    )
    outputs = model.evaluate_graph(_inputs(model, batch, execution, conditioning))
    assert not outputs.energy_available
    assert not outputs.atom_energy_complete
    with pytest.raises(ValueError):
        _ = outputs.energy
    reported = outputs.observable(AtomisticObservable.REPORTED_ENERGY)
    direct = outputs.observable(AtomisticObservable.DIRECT_FORCES)
    assert bool(reported.available[0])
    assert np.all(np.isfinite(np.asarray(direct.values)))
    with pytest.raises(ValueError):
        model.evaluate_graph(
            _inputs(model, batch, execution, conditioning),
            AtomisticObservableRequest((AtomisticObservable.ENERGY,)),
        )
    with pytest.raises(ValueError):
        energy_and_forces(model, batch, execution, conditioning=conditioning)
    with pytest.raises(ValueError, match=DERIVATIVE_UNSUPPORTED):
        prepare_atomistic_hessian(model, batch, execution, conditioning=conditioning)
    problem, _ = _training_case("uma")
    with pytest.raises(ValueError):
        fit_atomistic_potential(
            model, problem, AtomisticTrainingPolicy(maximum_steps=1), key=jr.key(0)
        )


def test_mace_hessian_preserves_classical_symmetry_and_mixed_order_policy() -> None:
    scenario, batch, execution, _ = _finite("mace")
    model = scenario.model
    operator = prepare_atomistic_hessian(model, batch, execution).operator
    assert operator.properties.self_adjoint
    rng = np.random.default_rng(31)
    left = jnp.asarray(rng.normal(size=batch.positions.shape))
    right = jnp.asarray(rng.normal(size=batch.positions.shape))
    np.testing.assert_allclose(
        float(jnp.vdot(left, operator.mv(right))),
        float(jnp.vdot(right, operator.mv(left))),
        rtol=1e-9,
    )
    # A rigid translation is a null direction of the Hessian at any geometry.
    translation = jnp.broadcast_to(jnp.asarray([0.3, -0.5, 0.8]), batch.positions.shape)
    scale = float(jnp.max(jnp.abs(operator.mv(right))))
    _close(
        operator.mv(translation), np.zeros(batch.positions.shape), 1e-9 * max(1.0, scale)
    )
    mixed = AtomisticObservableRequest(
        (AtomisticObservable.HESSIAN_ACTION,),
        derivatives=(ATOMISTIC_HESSIAN_REQUEST, MIXED),
    )
    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        prepare_atomistic_hessian(model, batch, execution, request=mixed)


# ------------------------------------------- provider-free artifact restore


_FRESH_EVALUATION = """
import json
import sys

import numpy as np

import phydrax as phx

phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
path, family, revision, structure = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
model = phx.atomistic.read_atomistic_model_artifact(path, numeric_revision_id=revision).model
units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
atoms = len(structure["numbers"])
batch = phx.atomistic.AtomisticBatch(
    np.asarray([structure["numbers"]], dtype=np.int32),
    np.asarray([structure["positions"]], dtype=np.float64),
    np.asarray([structure["masses"]], dtype=np.float64),
    units.scale,
)
execution = phx.atomistic.AtomisticGraphExecutionPlan(atoms - 1, maximum_dense_atoms=atoms)
if family.startswith("uma"):
    conditioning = model.schema.conditioning(
        ["omol"], atom_capacity=atoms, charges=[0], spins=[1]
    )
elif family == "aimnet2":
    conditioning = phx.atomistic.AtomisticConditioning(
        model.conditioning_spec,
        {"charge": np.asarray([0.0])},
        case_count=1,
        atom_capacity=atoms,
    )
else:
    conditioning = None
prediction = phx.atomistic.energy_and_forces(model, batch, execution, conditioning=conditioning)
print(json.dumps({
    "type": type(model).__name__,
    "valid": bool(np.all(np.asarray(prediction.valid))),
    "energy": np.asarray(prediction.energy).tolist(),
    "forces": np.asarray(prediction.forces).tolist(),
    "foreign": sorted(name for name in %r if name in sys.modules),
}))
""" % (SOURCE_PACKAGES,)


@pytest.mark.parametrize("name", ("aimnet2", "maceles", "uma", "uma-merged"))
def test_restored_artifact_evaluates_in_a_fresh_process_without_source_packages(
    name: str, tmp_path: Path
) -> None:
    scenario, batch, execution, conditioning = _finite(
        "uma" if name == "uma-merged" else name
    )
    if name.startswith("uma"):
        conditioning = scenario.conditioning(batch, (NEUTRAL,))
    model = scenario.model
    if name == "uma-merged":
        model = model.merged(uma_binding(model, batch, periodic=False))
    expected = energy_and_forces(model, batch, execution, conditioning=conditioning)
    assert bool(expected.valid[0])
    path = tmp_path / name
    manifest = write_atomistic_model_artifact(path, model, licenses=("test-fixture",))
    restored = read_atomistic_model_artifact(
        path, numeric_revision_id=manifest.numeric_revision.revision_id
    )
    assert type(restored.model) is type(model)
    structure = {
        "numbers": np.asarray(batch.atomic_numbers)[0].tolist(),
        "positions": DIMER.tolist(),
        "masses": np.asarray(batch.masses)[0].tolist(),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_FRESH_EVALUATION),
            str(path),
            name,
            manifest.numeric_revision.revision_id,
            json.dumps(structure),
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    fresh = json.loads(completed.stdout.strip().splitlines()[-1])
    assert fresh["type"] == type(model).__name__
    assert fresh["valid"] and fresh["foreign"] == []
    np.testing.assert_allclose(fresh["energy"], np.asarray(expected.energy), rtol=1e-10)
    _close(fresh["forces"], np.asarray(expected.forces), 1e-9)

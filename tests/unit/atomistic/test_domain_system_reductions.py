"""Domain-runtime system reductions: AIMNet2 per-pass charge equilibration.

AIMNet2 normalizes every pass's predicted charges per system and channel, so
an owner-local evaluation is correct only if the executor reduces the owners'
partial sums into the replicated system context and returns its transpose
exactly once. The reference is the same native AIMNet2 model evaluated on the
single-device periodic graph (`energy_and_forces` / `evaluate_graph`).
Executor-level stage tests live in ``test_layered_system_reductions.py``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    energy_and_forces,
)
from phydrax.atomistic._conditioning import (
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from phydrax.atomistic._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._domain_terms import AtomisticDomainEwaldTerm
from phydrax.atomistic._electrostatics import (
    ChargeNeutralityPolicy,
    ParticleMeshEwaldPotential,
    PeriodicCoulombEvidence,
)
from phydrax.atomistic._neighborhood_bundle import AtomisticCellListImageSearch
from phydrax.atomistic._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
)
from phydrax.atomistic._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
)
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.discretization import ParticleImageCapacity
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.nn.atomistic._aimnet2 import _CHARGE_PORT
from phydrax.nn.atomistic._aimnet2_source import (
    aimnet2_potential_from_source,
    externalize_source_long_range,
)
from phydrax.sparse._streamed import StreamedRelationPlan
from tests._support.aimnet2_source import RC_S, source_inventory
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_domain import domain_plan, SCALE


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
CELL = np.asarray([[3.6, 0.0, 0.0], [0.3, 3.7, 0.0], [0.2, 0.1, 3.8]])
NUMBERS = np.asarray([8, 1, 1, 6])
FRACTIONAL = np.asarray(
    [[0.12, 0.2, 0.31], [0.38, 0.27, 0.33], [0.55, 0.61, 0.18], [0.81, 0.74, 0.66]]
)
EXECUTION = AtomisticGraphExecutionPlan(
    256,
    backend="particle",
    image_capacity=ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=8192,
        maximum_degree=256,
        maximum_images=343,
    ),
    search_cell=CELL,
    search_margin=0.05,
)


def _model() -> Any:
    declaration, tensors = source_inventory(3)
    core, state, _ = externalize_source_long_range(declaration, tensors)
    return aimnet2_potential_from_source(
        SCALE, core, state, neighbor_cutoff=RC_S
    ).potential


def _batch(positions: np.ndarray) -> AtomisticBatch:
    return AtomisticBatch.from_structure(
        AtomicStructure(
            NUMBERS,
            positions,
            np.ones(len(NUMBERS)),
            SCALE,
            cell=CELL,
            periodic_axes=np.asarray([True, True, True]),
        )
    )


def _conditioning(model: Any, charge: float, atom_capacity: int) -> AtomisticConditioning:
    return AtomisticConditioning(
        model.conditioning_spec,
        {"charge": np.asarray([charge], dtype=np.float64)},
        case_count=1,
        atom_capacity=atom_capacity,
    )


def _domain(model: Any, owners: int, charge: float, positions: np.ndarray) -> Any:
    plan = domain_plan(
        model,
        owners=owners,
        cell=CELL,
        edge_capacity=512,
        alias_capacity=1024,
        halo_capacity=8,
        message_capacity_bytes=1 << 24,
        stream=StreamedRelationPlan(),
    )
    runtime = prepare_atomistic_domain(plan, model)
    mask = jnp.ones(NUMBERS.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(NUMBERS), mask))
    state = runtime.ingress_global(
        positions,
        species,
        rng_key=jax.random.key(0),
        conditioning=_conditioning(model, charge, plan.ownership.local_capacity),
    )
    return runtime, state


@pytest.mark.parametrize("charge", [0.0, 1.0], ids=["neutral", "cation"])
@pytest.mark.parametrize("owners", [1, 2, 3])
def test_aimnet2_domain_matches_single_device_energy_forces_stress(
    owners: int, charge: float
) -> None:
    model = _model()
    positions = FRACTIONAL @ CELL
    runtime, state = _domain(model, owners, charge, positions)
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    batch = _batch(positions)
    reference = energy_and_forces(
        model,
        batch,
        EXECUTION,
        compute_stress=True,
        conditioning=_conditioning(model, charge, batch.atom_capacity),
    )
    assert bool(reference.valid[0])
    np.testing.assert_allclose(
        float(evaluation.energy), float(reference.energy[0]), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)),
        np.asarray(reference.forces[0, : len(NUMBERS)]),
        atol=1e-9,
    )
    reference_stress = reference.stress
    assert reference_stress is not None  # compute_stress=True on a periodic cell
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), np.asarray(reference_stress[0]), atol=1e-9
    )
    assert len(jax.tree.leaves(evaluation.system_contexts)) > 0


def test_aimnet2_empty_owner_keeps_the_system_context() -> None:
    """An owner without atoms contributes zero partials, never a skewed mean."""
    model = _model()
    fractional = FRACTIONAL.copy()
    fractional[:, 0] = 0.1 + 0.5 * fractional[:, 0]
    positions = fractional @ CELL
    runtime, state = _domain(model, 4, 1.0, positions)
    loads = np.asarray(state.layout.active).reshape((4, -1)).sum(axis=1)
    assert np.any(loads == 0)
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    batch = _batch(positions)
    reference = energy_and_forces(
        model,
        batch,
        EXECUTION,
        conditioning=_conditioning(model, 1.0, batch.atom_capacity),
    )
    np.testing.assert_allclose(
        float(evaluation.energy), float(reference.energy[0]), rtol=1e-10
    )


def test_aimnet2_parameter_partials_are_exactly_once_for_any_owner_count() -> None:
    model = _model()
    positions = FRACTIONAL @ CELL
    batch = _batch(positions)
    lanes = partition_parameters(model)
    conditioning = _conditioning(model, 1.0, batch.atom_capacity)

    def energy(lane: Any) -> jax.Array:
        candidate = combine_parameters(lane, lanes[1], lanes[2])
        return candidate.energy(batch, EXECUTION, conditioning=conditioning)[0]

    expected = jax.grad(energy)(lanes[0])
    for owners in (1, 2, 3):
        runtime, state = _domain(model, owners, 1.0, positions)
        partials = runtime.owner_parameter_partials(state, energy_cotangent=1.0)
        gradient = jax.tree.map(runtime.plan.reduction.ordered_sum, partials)
        for observed, oracle in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(observed), np.asarray(oracle), rtol=1e-8, atol=1e-10
            )


def test_domain_refuses_conditioning_not_bound_to_owner_rows() -> None:
    model = _model()
    positions = FRACTIONAL @ CELL
    runtime, _ = _domain(model, 2, 0.0, positions)
    mask = jnp.ones(NUMBERS.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(NUMBERS), mask))
    with pytest.raises(ValueError, match="atom_capacity"):
        runtime.ingress_global(
            positions,
            species,
            rng_key=jax.random.key(0),
            conditioning=_conditioning(model, 0.0, len(NUMBERS)),
        )


def _charge_field_case(
    owners: int, charge: float, neutrality: ChargeNeutralityPolicy = "uniform-background"
) -> tuple[Any, ...]:
    """AIMNet2 whose predicted CHARGES feed a PME term, domain and program."""
    model = _model()
    positions = FRACTIONAL @ CELL
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = AtomisticSystemPlan(
        np.arange(len(NUMBERS), dtype=np.int64),
        NUMBERS,
        np.ones(len(NUMBERS)),
        units,
        charges=np.zeros(len(NUMBERS)),
        cell=PeriodicCell(CELL),
    ).prepare()
    field = AtomisticFieldReference(
        "aimnet2",
        AtomisticObservable.CHARGES,
        _CHARGE_PORT,
        level=AtomisticConditionLevel.ATOM,
        unit=units.charge_unit,
    )
    # The minimum-image real-space cutoff stays below half the cell width.
    # AIMNet2 equilibrates its predicted charges to the requested total charge
    # only up to the source ``ops.nse`` epsilon (`CHARGE_RESIDUAL`), and the
    # cation is charged by request: the cell is net charged, so PME declares
    # the uniform neutralizing background unless a test requires neutrality.
    pme = ParticleMeshEwaldPotential(
        2.2,
        1.6,
        (16, 16, 16),
        spline_degree=4,
        neutrality=neutrality,
        charge_field=field,
    )
    prepared_pme = pme.prepare(system)
    term = AtomisticDomainEwaldTerm(
        prepared_pme, AtomisticReciprocalDistributionPlan(prepared_pme, reuse_reach=0.3)
    )
    (requirement,) = pme.requirements.neighborhoods
    pme_role = AtomisticDomainRolePlan(
        requirement,
        PeriodicCell(CELL).image_stencil(
            requirement.bounded_cutoff() + 0.3, maximum_image_count=4096
        ),
        streaming=StreamedRelationPlan(),
        skin=0.3,
        edge_capacity=512,
        alias_capacity=1024,
        halo_capacity=8,
        term_key=pme.name,
    )
    base = domain_plan(
        model,
        owners=owners,
        cell=CELL,
        edge_capacity=512,
        alias_capacity=1024,
        halo_capacity=8,
        message_capacity_bytes=1 << 24,
        stream=StreamedRelationPlan(),
    )
    plan = AtomisticDomainPlan(
        base.ownership,
        base.partition,
        (*base.roles, pme_role),
        migration_capacity=8,
        message_capacity_bytes=1 << 24,
        reduction=base.reduction,
        terms=(term,),
    )
    runtime = prepare_atomistic_domain(plan, model)
    mask = jnp.ones(NUMBERS.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(NUMBERS), mask))
    state = runtime.ingress_global(
        positions,
        species,
        rng_key=jax.random.key(0),
        conditioning=_conditioning(model, charge, plan.ownership.local_capacity),
    )
    program = AtomisticPotentialProgram(
        [
            LearnedGraphPotentialTerm(
                model,
                name="aimnet2",
                allow_periodic=True,
                conditioning=_conditioning(model, charge, system.capacity),
            ),
            pme,
        ]
    )
    reference = prepare_program(
        program,
        system,
        graph_execution=EXECUTION,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 16),
        skin=0.3,
        search_capacities=(search_capacity(4096, particles_per_cell=32, images=343),),
        role_capacities=(role_capacity(4096, 256),),
    )
    reference_state = bundle_state(reference, positions)
    return runtime, state, reference, reference_state, positions


@pytest.mark.parametrize("charge", [0.0, 1.0], ids=["neutral", "cation"])
@pytest.mark.parametrize("owners", [1, 2])
def test_predicted_charges_feed_distributed_pme_like_the_single_device_program(
    owners: int, charge: float
) -> None:
    """Field-dependent PME: charge cotangents return through AIMNet2 exactly once."""
    runtime, state, reference, reference_state, positions = _charge_field_case(
        owners, charge
    )
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    assert not bool(evaluation.atom_energy_complete)
    expected = reference.evaluate(
        jnp.asarray(positions), reference_state, compute_stress=True
    )
    assert bool(expected.successful)
    np.testing.assert_allclose(
        float(evaluation.energy), float(expected.energy), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.term_energies),
        np.asarray(expected.term_energies)[1:],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)),
        np.asarray(expected.forces),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), np.asarray(expected.stress), atol=1e-9
    )


@pytest.mark.parametrize("charge", [0.0, 1.0], ids=["neutral", "cation"])
def test_require_neutral_pme_refuses_the_predicted_net_charge(charge: float) -> None:
    """AIMNet2 meets the requested charge only up to its source epsilon; under
    ``require-neutral`` PME the domain and the program both refuse that net
    charge as a scientific (not capacity or binding) failure."""
    runtime, state, reference, reference_state, positions = _charge_field_case(
        2, charge, neutrality="require-neutral"
    )
    evaluation = runtime.evaluate(state)
    status = evaluation.status
    assert not bool(evaluation.successful)
    assert bool(evaluation.certificate.certified)
    assert bool(status.identity_bound) and bool(status.relation_successful)
    assert np.all(np.asarray(status.term_capacity_successful))
    assert not np.any(np.asarray(status.term_scientific_successful))
    expected = reference.evaluate(
        jnp.asarray(positions), reference_state, compute_stress=True
    )
    assert not bool(expected.successful)
    (evidence,) = [
        item
        for record in expected.term_evidence
        for item in record
        if isinstance(item, PeriodicCoulombEvidence)
    ]
    assert bool(evidence.charges_available) and bool(evidence.charges_successful)
    assert bool(evidence.geometry_successful) and bool(evidence.reciprocal_successful)
    assert not bool(evidence.neutral)
    assert abs(float(evidence.total_charge)) > 0.0


def test_field_dependent_terms_need_the_producing_model() -> None:
    runtime, _, _, _, _ = _charge_field_case(1, 0.0)
    plan = runtime.plan
    with pytest.raises(ValueError, match="predicted"):
        prepare_atomistic_domain(
            AtomisticDomainPlan(
                plan.ownership,
                plan.partition,
                (plan.roles[1],),
                migration_capacity=8,
                message_capacity_bytes=1 << 24,
                terms=plan.terms,
            )
        )


def test_predicted_charge_pme_parameter_gradient_matches_single_device() -> None:
    """The field term's parameter dependence (through predicted charges) is in
    the owners' exactly-once partials: their sum is ``dE/dθ`` of the program."""
    runtime, state, reference, reference_state, positions = _charge_field_case(2, 1.0)
    lanes = partition_parameters(runtime.model)
    (index,) = [
        position
        for position, term in enumerate(reference.terms)
        if term.name == "aimnet2"
    ]

    def energy(lane: Any) -> jax.Array:
        candidate = combine_parameters(lane, lanes[1], lanes[2])
        rebound = eqx.tree_at(
            lambda program: program.terms[index].plan.potential, reference, candidate
        )
        return rebound.energy(jnp.asarray(positions), reference_state)[0]

    expected = jax.grad(energy)(lanes[0])
    partials = runtime.owner_parameter_partials(state, energy_cotangent=1.0)
    gradient = jax.tree.map(runtime.plan.reduction.ordered_sum, partials)
    for observed, oracle in zip(
        jax.tree.leaves(gradient), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(observed), np.asarray(oracle), rtol=1e-8, atol=1e-10
        )
    # Energy-only loss 0.5 (E - (E - 1))^2: its gradient is exactly dE/dθ.
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    loss = runtime.loss_gradient(
        state,
        evaluation.energy - 1.0,
        jnp.zeros_like(state.positions),
        energy_weight=0.5,
        force_weight=0.0,
    )
    assert bool(loss.successful)
    for observed, oracle in zip(
        jax.tree.leaves(loss.parameter_gradient), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(observed), np.asarray(oracle), rtol=1e-8, atol=1e-10
        )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed dynamics over ``AtomisticDomainState`` against single-device P5.

The same random native MACE model drives an independent single-device P5
``PreparedAtomisticDynamics`` (learned program over its own neighborhood
bundle) and the distributed domain runtime.  Trajectories are compared in
unwrapped logical coordinates after enough steps that the domain certificate
expires and every continuation leaf (including constraint rows) migrates.
Lane-reference ownership is the functional oracle; ``devices`` cases run the
device-collective route when enough local devices are visible.  Two
water-like rigid molecules straddle the owner boundary, so every constraint
component spans owners.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group, ExecutionRuntime
from phydrax._fingerprint import canonical_fingerprint
from phydrax.atomistic._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
)
from phydrax.atomistic._cell_dynamics import MTKBarostatPlan, MTKNPHPlan, MTKNPTPlan
from phydrax.atomistic._constraints import DistanceConstraintPlan
from phydrax.atomistic._domain_dynamics import (
    AtomisticDomainConstraintPlan,
    AtomisticDomainConstraintRows,
    AtomisticDomainDynamicsPlan,
    AtomisticDomainEnsemblePlan,
    domain_constraint_rows,
    export_domain_constraints,
    GLE_MEMORY_PAYLOAD,
    import_domain_constraints,
    owner_inner_product,
    owner_masked_sum,
    PreparedAtomisticDomainDynamics,
    rebuild_domain_constraints,
)
from phydrax.atomistic._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionPolicy,
    AtomisticDomainRolePlan,
)
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticStepRejectionReason,
    VelocityVerletPlan,
)
from phydrax.atomistic._graph import AtomisticGraphExecutionPlan
from phydrax.atomistic._neighborhood_bundle import AtomisticNeighborhoodRole
from phydrax.atomistic._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
)
from phydrax.atomistic._stochastic_runtime import GeneralizedLangevinPlan
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._thermal import (
    BAOABLangevinPlan,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
    StochasticVelocityRescalingPlan,
)
from phydrax.atomistic._thermodynamic import (
    AtomisticEnsembleKind,
    AtomisticMomentumRestriction,
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticThermodynamicStatePlan,
    AtomisticVolumeCoordinateConvention,
    PreparedThermodynamicStateTable,
)
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import DistributedOwnershipPlan
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.sparse._streamed import StreamedRelationPlan
from tests._support.atomistic_bundles import prepare_program
from tests._support.atomistic_dynamics_fixtures import (
    ROLE_CAPACITIES,
    SEARCH,
    SEARCH_CAPACITIES,
)


UNITS = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
CUTOFF = 3.0
SKIN = 0.3
CELL = np.array([[4.6, 0.0, 0.0], [0.9, 4.4, 0.0], [0.5, -0.6, 4.8]])
# Two rigid water-like molecules; every O-H bond crosses fractional x = 1/2.
POSITIONS = np.array(
    [
        [2.0, 0.3, 0.1],
        [2.96, 0.3, 0.1],
        [1.76, 1.23, 0.1],
        [2.5, 2.4, 2.6],
        [3.46, 2.4, 2.6],
        [2.26, 3.33, 2.6],
    ]
)
NUMBERS = np.array([8, 1, 1, 8, 1, 1], dtype=np.int32)
MASSES = np.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008])
PAIRS = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]], dtype=np.int64)
TOLERANCE = 1.0e-8
TEMPERATURE = 300.0

# Jacobi SHAKE/RATTLE sweeps at tolerance 1e-10. The rigid water triangles'
# mass-weighted constraint coupling contracts by rho ~= 0.774 per sweep.
# Velocity Verlet enters each projection O(dt^2) off the manifold, within 64
# sweeps. BAOAB's O-step noise is not tangent to the manifold: the declared
# NPT trajectory's first step enters its position projection with a 7e-3
# A^2/fs velocity defect needing ~71 sweeps, so 64 sweeps refuse on both
# routes. 96 sweeps (rho^96 ~= 2e-11) admit defects ~670 times larger.
CONSTRAINT_ITERATIONS = 64
LANGEVIN_CONSTRAINT_ITERATIONS = 96


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def _model(key: int = 0) -> MACEPotential:
    architecture = MACEArchitecture(
        species=(1, 8),
        cutoff=CUTOFF,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=6.0,
    )
    return MACEPotential(
        UNITS.scale,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jax.random.key(key),
    )


def _velocities(scale: float = 0.03) -> np.ndarray:
    return scale * np.asarray(jax.random.normal(jax.random.key(1), POSITIONS.shape))


def _distances() -> np.ndarray:
    return np.linalg.norm(POSITIONS[PAIRS[:, 0]] - POSITIONS[PAIRS[:, 1]], axis=1)


def _rigid_velocities() -> np.ndarray:
    """Velocities tangent to every bond (rigid rotations plus translations)."""
    velocity = _velocities()
    for start in (0, 3):
        block = slice(start, start + 3)
        center = np.average(POSITIONS[block], axis=0, weights=MASSES[block])
        omega = np.array([0.0, 0.0, 0.02 if start == 0 else -0.015])
        velocity[block] = velocity[start] + np.cross(omega, POSITIONS[block] - center)
    return velocity


# --------------------------------------------------------------- single device


MOLECULES = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
# One fixed hydrogen (not constrained in the fixed-atom cases).
MOBILE = np.array([True, True, True, True, True, False])


def _reference(
    model: MACEPotential,
    integrator: Any,
    *,
    ensemble: AtomisticEnsembleKind,
    temperature: float | None = None,
    pressure: float | None = None,
    constrained: bool = False,
    restriction: AtomisticMomentumRestriction = "none",
    convention: AtomisticVolumeCoordinateConvention = "atomic",
    fixed: bool = False,
    constraint_iterations: int = CONSTRAINT_ITERATIONS,
) -> tuple[Any, Any]:
    topology = (
        MolecularTopologyPlan(
            constraints=PAIRS.tolist(), constraint_distances=_distances().tolist()
        )
        if constrained
        else None
    )
    system = AtomisticSystemPlan(
        np.arange(POSITIONS.shape[0], dtype=np.int64),
        NUMBERS,
        MASSES,
        UNITS,
        mobile_mask=MOBILE if fixed else None,
        molecule_ids=MOLECULES,
        topology=topology,
        cell=PeriodicCell(jnp.asarray(CELL)),
    ).prepare()
    program = AtomisticPotentialProgram(
        [LearnedGraphPotentialTerm(model, allow_periodic=True)]
    )
    potential = prepare_program(
        program,
        system,
        graph_execution=AtomisticGraphExecutionPlan(
            64,
            backend="particle",
            streamed=StreamedRelationPlan(accumulation="deterministic"),
        ),
        search=SEARCH,
        skin=0.4,
        search_capacities=SEARCH_CAPACITIES,
        role_capacities=ROLE_CAPACITIES,
    )
    constraints = (
        DistanceConstraintPlan(
            maximum_iterations=constraint_iterations, tolerance=1.0e-10
        ).prepare(system)
        if constrained
        else None
    )
    dynamics = AtomisticDynamicsPlan(
        system, potential, integrator, constraints=constraints
    ).prepare()
    measure = AtomisticPhaseSpaceMeasurePlan(
        system,
        volume_coordinate_convention=convention,
        momentum_restriction=restriction,
    )
    table = PreparedThermodynamicStateTable(
        dynamics,
        (
            AtomisticThermodynamicStatePlan(
                measure,
                ensemble=ensemble,
                temperature=temperature,
                pressure=pressure,
            ),
        ),
    )
    return dynamics, table


# ------------------------------------------------------------------ domain


def _ownership(owners: int, *, devices: bool) -> DistributedOwnershipPlan:
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
    )
    if jax.process_count() != 1:
        pytest.skip("These global-host-ingress cases require a single JAX process.")
    runtime = ExecutionRuntime.current()
    parent = runtime.root_group.spec
    if devices:
        if parent.device_count < owners:
            pytest.skip(f"Requires at least {owners} visible devices.")
        keys = parent.device_keys[:owners]
        group = bind_execution_group(
            ExecutionGroupSpec(
                canonical_fingerprint(
                    {
                        "parent_group_id": parent.group_id,
                        "group_index": 0,
                        "device_keys": [list(key) for key in keys],
                    }
                ),
                parent.process_indices,
                keys,
                mesh_axes=(("device", owners),),
                parent_group_id=parent.group_id,
                process_host_ids=parent.process_host_ids,
            )
        )
        return DistributedOwnershipPlan(address, group, 8)
    group = runtime.child_groups(parent.device_count)[0]
    return DistributedOwnershipPlan(address, group, 8, owner_lanes=owners)


def _domain_plan(
    model: MACEPotential,
    owners: int,
    *,
    devices: bool = False,
    dynamic_cell: bool = False,
    migration_capacity: int = 8,
) -> AtomisticDomainPlan:
    periodic = PeriodicCell(jnp.asarray(CELL))
    requirement = model.requirements.neighborhood(
        AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
    )
    role = AtomisticDomainRolePlan(
        requirement,
        periodic.image_stencil(
            CUTOFF + SKIN + (0.6 if dynamic_cell else 0.0), maximum_image_count=4096
        ),
        streaming=StreamedRelationPlan(
            receiver_tile=4,
            edge_tile=32,
            channel_capacity=model.configuration.message_payload_width(),
            accumulation="deterministic",
        ),
        skin=SKIN,
        edge_capacity=1024,
        alias_capacity=512,
        halo_capacity=8,
    )
    return AtomisticDomainPlan(
        _ownership(owners, devices=devices),
        FractionalOwnerPartition(periodic, (owners, 1, 1)),
        (role,),
        migration_capacity=migration_capacity,
        message_capacity_bytes=1 << 20,
        reduction=AtomisticDomainReductionPolicy("deterministic"),
        dynamic_cell=dynamic_cell,
    )


def _constraint_plan(
    *,
    halo_capacity: int = 8,
    directory_capacity: int = 16,
    iterations: int = CONSTRAINT_ITERATIONS,
) -> AtomisticDomainConstraintPlan:
    return AtomisticDomainConstraintPlan(
        DistanceConstraintPlan(maximum_iterations=iterations, tolerance=1.0e-10),
        constraint_count=PAIRS.shape[0],
        row_capacity=PAIRS.shape[0],
        halo_capacity=halo_capacity,
        directory_capacity=directory_capacity,
        migration_capacity=PAIRS.shape[0],
        component_row_capacity=3,
    )


def _domain(
    model: MACEPotential,
    integrator: Any,
    owners: int,
    *,
    devices: bool = False,
    ensemble: AtomisticEnsembleKind,
    temperature: float | None = None,
    pressure: float | None = None,
    constrained: bool = False,
    dynamic_cell: bool = False,
    restriction: AtomisticMomentumRestriction = "none",
    convention: AtomisticVolumeCoordinateConvention = "atomic",
    fixed: bool = False,
    barostat: IsotropicMonteCarloBarostatPlan | None = None,
    velocities: np.ndarray | None = None,
    constraint_iterations: int = CONSTRAINT_ITERATIONS,
) -> tuple[PreparedAtomisticDomainDynamics, Any]:
    plan = _domain_plan(model, owners, devices=devices, dynamic_cell=dynamic_cell)
    runtime = prepare_atomistic_domain(plan, model)
    constraints = (
        _constraint_plan(iterations=constraint_iterations) if constrained else None
    )
    dynamics_plan = AtomisticDomainDynamicsPlan(
        plan,
        integrator,
        AtomisticDomainEnsemblePlan(
            UNITS,
            ensemble=ensemble,
            atom_count=POSITIONS.shape[0],
            temperature=temperature,
            pressure=pressure,
            mobile_count=int(MOBILE.sum()) if fixed else None,
            momentum_restriction=restriction,
            volume_coordinate_convention=convention,
            molecule_count=2 if convention == "molecular-center" else None,
        ),
        constraints=constraints,
        barostat=barostat,
    )
    dynamics = dynamics_plan.prepare(runtime)
    mask = jnp.ones(NUMBERS.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(NUMBERS), mask))
    # Routing and continuation fixtures start on the declared velocity manifold;
    # unconstrained random velocities are a separate projection-convergence case.
    velocity = (
        (_rigid_velocities() if constrained else _velocities())
        if velocities is None
        else velocities
    )
    rows = (
        None
        if constraints is None
        else domain_constraint_rows(constraints, plan.ownership, PAIRS, _distances())
    )
    state = runtime.ingress_global(
        POSITIONS,
        species,
        rng_key=jax.random.key(2),
        momenta=MASSES[:, None] * velocity,
        masses=MASSES,
        cell_vectors=CELL,
        stable_ids=np.arange(POSITIONS.shape[0], dtype=np.int64),
        atom_payload=dynamics_plan.atom_payload(
            mobile=MOBILE if fixed else None,
            molecules=MOLECULES if convention == "molecular-center" else None,
        ),
        constraints=rows,
    )
    return dynamics, dynamics.initialize(state)


@eqx.filter_jit
def _domain_step(dynamics: PreparedAtomisticDomainDynamics, state: Any) -> Any:
    return dynamics.step_detailed(state)


@eqx.filter_jit
def _domain_volume_move(dynamics: PreparedAtomisticDomainDynamics, state: Any) -> Any:
    return dynamics.monte_carlo_volume_move(state)


@eqx.filter_jit
def _reference_step(dynamics: Any, state: Any, table: Any) -> Any:
    return dynamics.step_detailed(state, table)


@eqx.filter_jit
def _reference_volume_move(dynamics: Any, state: Any, table: Any, plan: Any) -> Any:
    return apply_isotropic_monte_carlo_barostat(dynamics, state, table, plan)


def _run_pair(
    reference: tuple[Any, Any],
    domain: tuple[PreparedAtomisticDomainDynamics, Any],
    steps: int,
    *,
    velocities: np.ndarray | None = None,
    barostat: IsotropicMonteCarloBarostatPlan | None = None,
) -> tuple[Any, Any, int]:
    dynamics, table = reference
    velocity = _velocities() if velocities is None else velocities
    single = dynamics.initialize_state(
        jnp.asarray(POSITIONS),
        table,
        momentum=MASSES[:, None] * velocity,
        key=jax.random.key(2),
    )
    distributed_dynamics, distributed = domain
    epochs_before = int(distributed.layout.owner_epochs[0])
    for _ in range(steps):
        reference_step = _reference_step(dynamics, single, table)
        assert bool(reference_step.successful), int(reference_step.rejection_reasons)
        single = reference_step.accepted_state
        evaluation = _domain_step(distributed_dynamics, distributed)
        assert bool(evaluation.successful), int(evaluation.rejection_reasons)
        distributed = evaluation.accepted_state
        if barostat is not None:
            reference_move = _reference_volume_move(dynamics, single, table, barostat)
            assert bool(reference_move.successful), int(reference_move.rejection_reasons)
            move = _domain_volume_move(distributed_dynamics, distributed)
            assert bool(move.successful), int(move.rejection_reasons)
            assert bool(move.accepted) == bool(reference_move.accepted)
            np.testing.assert_allclose(
                float(move.log_acceptance_probability),
                float(reference_move.log_acceptance_probability),
                atol=TOLERANCE,
            )
            single = reference_move.accepted_state
            distributed = move.accepted_state
    migrations = int(distributed.layout.owner_epochs[0]) - epochs_before
    return single, distributed, migrations


_CONTROLLER_FIELDS = (
    "particle_chain",
    "cell_chain",
    "cell",
    "move_counter",
    "accepted_moves",
    "cell_epoch",
    "rescale_events",
)


def _assert_parity(single: Any, distributed: Any) -> None:
    layout = distributed.layout
    unwrapped = (
        single.kinematics.positions
        + single.kinematics.image_counts.astype(jnp.float64) @ single.cell_vectors
    )
    np.testing.assert_allclose(
        np.asarray(layout.collect(distributed.unwrapped_positions())),
        np.asarray(unwrapped),
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(layout.collect(distributed.momenta)),
        np.asarray(single.kinematics.momenta),
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(distributed.cell_vectors),
        np.asarray(single.cell_vectors),
        atol=TOLERANCE,
    )
    ledger, reference = distributed.ledger, single.energy
    for name in (
        "kinetic_energy",
        "potential_energy",
        "extended_energy",
        "thermostat_heat",
        "reservoir_heat",
        "barostat_work",
        "constraint_work",
    ):
        np.testing.assert_allclose(
            float(getattr(ledger, name)), float(getattr(reference, name)), atol=TOLERANCE
        )
    assert int(ledger.accepted_steps) == int(reference.accepted_steps)
    assert int(distributed.step_index) == int(single.step_index)
    for name in _CONTROLLER_FIELDS:
        for left, right in zip(
            jax.tree.leaves(getattr(distributed.controllers, name)),
            jax.tree.leaves(getattr(single.controllers, name)),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(left), np.asarray(right), atol=TOLERANCE
            )
    if single.controllers.gle is not None:
        assert distributed.controllers.gle is None
        np.testing.assert_allclose(
            np.asarray(layout.collect(distributed.atom_payload[GLE_MEMORY_PAYLOAD])),
            np.asarray(single.controllers.gle.memory),
            atol=TOLERANCE,
        )


# -------------------------------------------------------------------- tests


@pytest.mark.parametrize(
    ("owners", "devices"),
    [(2, False), (3, False), (2, True)],
    ids=["lanes-2", "lanes-3", "devices-2"],
)
def test_nve_trajectory_matches_single_device_through_migration(
    owners: int, devices: bool
) -> None:
    model = _model()
    integrator = VelocityVerletPlan(0.5)
    single, distributed, migrations = _run_pair(
        _reference(model, integrator, ensemble="nve"),
        _domain(model, integrator, owners, devices=devices, ensemble="nve"),
        30,
    )
    assert migrations >= 1
    _assert_parity(single, distributed)


def _gle() -> GeneralizedLangevinPlan:
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return GeneralizedLangevinPlan(
        0.5, 0.9 * rotation, np.sqrt(1.0 - 0.81) * np.eye(2), realization_id=7
    )


@pytest.mark.parametrize(
    "integrator",
    [
        BAOABLangevinPlan(0.5, 0.05, realization_id=3),
        NoseHooverChainPlan(0.5, NoseHooverChainParameters(20.0, chain_length=3)),
        StochasticVelocityRescalingPlan(0.5, 20.0, realization_id=5),
        _gle(),
    ],
    ids=["baoab", "nose-hoover-chain", "csvr", "gle"],
)
@pytest.mark.parametrize("owners", [2, 3], ids=["two-owners", "three-owners"])
def test_thermostats_match_single_device_rng_lineage(
    integrator: Any, owners: int
) -> None:
    model = _model()
    # Deliberate initial drift exercises owner changes even for strongly damped
    # GLE dynamics; migration must not depend on one random thermal realization.
    velocity = _velocities()
    velocity[:, 0] += 0.2
    single, distributed, migrations = _run_pair(
        _reference(model, integrator, ensemble="nvt", temperature=TEMPERATURE),
        _domain(
            model,
            integrator,
            owners,
            ensemble="nvt",
            temperature=TEMPERATURE,
            velocities=velocity,
        ),
        30,
        velocities=velocity,
    )
    assert migrations >= 1
    _assert_parity(single, distributed)


def test_fixed_atoms_and_center_of_mass_restriction_match_single_device() -> None:
    model = _model()
    fixed = VelocityVerletPlan(0.5)
    single, distributed, _ = _run_pair(
        _reference(model, fixed, ensemble="nve", fixed=True),
        _domain(model, fixed, 2, ensemble="nve", fixed=True),
        20,
    )
    _assert_parity(single, distributed)
    layout = distributed.layout
    np.testing.assert_array_equal(
        np.asarray(layout.collect(distributed.momenta))[~MOBILE], 0.0
    )
    assert int(distributed.ledger.accepted_steps) == 20
    chain = NoseHooverChainPlan(0.5, NoseHooverChainParameters(20.0))
    single, distributed, _ = _run_pair(
        _reference(
            model,
            chain,
            ensemble="nvt",
            temperature=TEMPERATURE,
            restriction="center-of-mass",
        ),
        _domain(
            model,
            chain,
            3,
            ensemble="nvt",
            temperature=TEMPERATURE,
            restriction="center-of-mass",
        ),
        20,
    )
    _assert_parity(single, distributed)
    np.testing.assert_allclose(
        np.asarray(distributed.layout.collect(distributed.momenta)).sum(axis=0),
        0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("convention", "constrained", "owners"),
    [("atomic", False, 2), ("atomic", False, 3), ("molecular-center", True, 2)],
    ids=["atomic-two-owners", "atomic-three-owners", "molecular-center-constrained"],
)
def test_monte_carlo_volume_moves_match_single_device(
    convention: AtomisticVolumeCoordinateConvention, constrained: bool, owners: int
) -> None:
    model = _model()
    integrator = BAOABLangevinPlan(0.5, 0.05, realization_id=2)
    barostat = IsotropicMonteCarloBarostatPlan(0.05, realization_id=4)
    velocity = _rigid_velocities() if constrained else None
    single, distributed, _ = _run_pair(
        _reference(
            model,
            integrator,
            ensemble="npt",
            temperature=TEMPERATURE,
            pressure=1.0e-3,
            constrained=constrained,
            convention=convention,
            constraint_iterations=LANGEVIN_CONSTRAINT_ITERATIONS,
        ),
        _domain(
            model,
            integrator,
            owners,
            ensemble="npt",
            temperature=TEMPERATURE,
            pressure=1.0e-3,
            constrained=constrained,
            dynamic_cell=True,
            convention=convention,
            barostat=barostat,
            velocities=velocity,
            constraint_iterations=LANGEVIN_CONSTRAINT_ITERATIONS,
        ),
        12,
        velocities=velocity,
        barostat=barostat,
    )
    assert int(distributed.controllers.move_counter) == 12
    _assert_parity(single, distributed)


def test_constrained_langevin_step_refuses_an_exhausted_projection_budget() -> None:
    """At 64 sweeps the constrained NPT trajectory's first BAOAB step leaves its
    RATTLE residual above tolerance on both routes; each refuses it as a
    constraint failure only and keeps the accepted state."""
    model = _model()
    integrator = BAOABLangevinPlan(0.5, 0.05, realization_id=2)
    velocity = _rigid_velocities()
    dynamics, table = _reference(
        model,
        integrator,
        ensemble="npt",
        temperature=TEMPERATURE,
        pressure=1.0e-3,
        constrained=True,
        convention="molecular-center",
    )
    distributed_dynamics, distributed = _domain(
        model,
        integrator,
        2,
        ensemble="npt",
        temperature=TEMPERATURE,
        pressure=1.0e-3,
        constrained=True,
        dynamic_cell=True,
        convention="molecular-center",
        barostat=IsotropicMonteCarloBarostatPlan(0.05, realization_id=4),
        velocities=velocity,
    )
    single = dynamics.initialize_state(
        jnp.asarray(POSITIONS),
        table,
        momentum=MASSES[:, None] * velocity,
        key=jax.random.key(2),
    )
    reference = _reference_step(dynamics, single, table)
    evaluation = _domain_step(distributed_dynamics, distributed)
    constraint = int(AtomisticStepRejectionReason.CONSTRAINT)
    assert not bool(reference.successful)
    assert int(reference.rejection_reasons) == constraint
    assert not bool(evaluation.successful)
    assert int(evaluation.rejection_reasons) == constraint
    residual = float(reference.candidate_state.constraint_velocity_residual)
    assert residual > 1.0e-10
    np.testing.assert_allclose(
        float(evaluation.candidate_state.constraints.velocity_residual),
        residual,
        rtol=1.0e-4,
    )
    assert int(reference.accepted_state.step_index) == 0
    np.testing.assert_array_equal(
        np.asarray(reference.accepted_state.kinematics.momenta),
        np.asarray(single.kinematics.momenta),
    )
    assert int(evaluation.accepted_state.step_index) == 0
    np.testing.assert_array_equal(
        np.asarray(evaluation.accepted_state.momenta), np.asarray(distributed.momenta)
    )


@pytest.mark.parametrize(
    ("integrator", "ensemble", "temperature"),
    [
        (
            MTKNPTPlan(
                0.5,
                MTKBarostatPlan(100.0),
                NoseHooverChainParameters(20.0),
                NoseHooverChainParameters(100.0),
            ),
            "npt",
            TEMPERATURE,
        ),
        (
            MTKNPHPlan(0.5, MTKBarostatPlan(100.0, inertia_temperature=TEMPERATURE)),
            "nph",
            None,
        ),
    ],
    ids=["mtk-npt", "mtk-nph"],
)
def test_mtk_cell_dynamics_matches_single_device(
    integrator: Any, ensemble: AtomisticEnsembleKind, temperature: float | None
) -> None:
    model = _model()
    single, distributed, _ = _run_pair(
        _reference(
            model, integrator, ensemble=ensemble, temperature=temperature, pressure=1.0e-3
        ),
        _domain(
            model,
            integrator,
            2,
            ensemble=ensemble,
            temperature=temperature,
            pressure=1.0e-3,
            dynamic_cell=True,
        ),
        15,
    )
    assert int(distributed.controllers.cell_epoch) == 15
    _assert_parity(single, distributed)


@pytest.mark.parametrize(
    ("owners", "devices"),
    [(2, False), (3, False), (2, True)],
    ids=["lanes-2", "lanes-3", "devices-2"],
)
def test_constraints_across_owners_hold_global_tolerance_and_match_p5(
    owners: int, devices: bool
) -> None:
    model = _model()
    integrator = VelocityVerletPlan(0.5)
    velocity = _rigid_velocities()
    single, distributed, migrations = _run_pair(
        _reference(model, integrator, ensemble="nve", constrained=True),
        _domain(
            model,
            integrator,
            owners,
            devices=devices,
            ensemble="nve",
            constrained=True,
            velocities=velocity,
        ),
        30,
        velocities=velocity,
    )
    assert migrations >= 1
    _assert_parity(single, distributed)
    constraints = distributed.constraints
    assert bool(constraints.successful)
    assert float(constraints.position_residual) <= 1.0e-10
    assert float(constraints.velocity_residual) <= 1.0e-10
    positions = np.asarray(distributed.layout.collect(distributed.unwrapped_positions()))
    np.testing.assert_allclose(
        np.linalg.norm(positions[PAIRS[:, 0]] - positions[PAIRS[:, 1]], axis=1),
        _distances(),
        atol=1.0e-9,
    )
    rows = distributed.constraints.rows
    lagrange = np.asarray(rows.collect(constraints.lagrange))
    np.testing.assert_allclose(
        lagrange, np.asarray(single.constraint_lagrange), atol=1.0e-8
    )


def _owner_of(layout: Any) -> dict[int, int]:
    ids = np.asarray(layout.stable_ids)
    active = np.asarray(layout.active)
    capacity = layout.plan.local_capacity
    return {int(ids[row]): row // capacity for row in np.flatnonzero(active)}


def test_constraint_rows_live_on_anchor_owner_and_migrate_exactly() -> None:
    model = _model()
    dynamics, state = _domain(
        model, VelocityVerletPlan(0.5), 2, ensemble="nve", constrained=True
    )
    constraints = state.constraints
    assert constraints is not None
    atom_owner = _owner_of(state.layout)
    row_owner = _owner_of(constraints.rows)
    atoms = np.asarray(constraints.atoms)
    components = np.asarray(constraints.components)
    row_ids = np.asarray(constraints.rows.stable_ids)
    active = np.asarray(constraints.rows.active)
    record = {}
    for slot in np.flatnonzero(active):
        assert row_owner[int(row_ids[slot])] == atom_owner[int(components[slot])]
        record[int(row_ids[slot])] = (tuple(atoms[slot]), int(components[slot]))
    assert record == {
        index: (tuple(PAIRS[index]), 0 if index < 3 else 3) for index in range(6)
    }
    # Shift every atom by half a cell along a so that ownership swaps owners.
    shifted = runtime_shift(dynamics, state, 0.5)
    transition = dynamics.runtime.rebuild(shifted)
    assert bool(transition.committed)
    moved = transition.state.constraints
    assert moved is not None
    new_atom_owner = _owner_of(transition.state.layout)
    assert new_atom_owner != atom_owner
    new_row_owner = _owner_of(moved.rows)
    new_atoms = np.asarray(moved.atoms)
    new_ids = np.asarray(moved.rows.stable_ids)
    after = {}
    for slot in np.flatnonzero(np.asarray(moved.rows.active)):
        assert (
            new_row_owner[int(new_ids[slot])]
            == new_atom_owner[int(moved.components[slot])]
        )
        after[int(new_ids[slot])] = (tuple(new_atoms[slot]), int(moved.components[slot]))
    assert after == record
    assert int(moved.topology_epoch) == int(constraints.topology_epoch) + 1
    np.testing.assert_array_equal(
        np.sort(np.asarray(moved.rows.collect(moved.targets))),
        np.sort(np.asarray(constraints.rows.collect(constraints.targets))),
    )
    # Replicated controllers and the ledger migrate bitwise.
    for left, right in zip(
        jax.tree.leaves((transition.state.controllers, transition.state.ledger)),
        jax.tree.leaves((state.controllers, state.ledger)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def runtime_shift(
    dynamics: PreparedAtomisticDomainDynamics, state: Any, fraction: float
) -> Any:
    shift = fraction * jnp.asarray(CELL[0])
    positions = jnp.where(state.layout.active[:, None], state.positions + shift, 0.0)
    return dynamics.runtime.with_dynamics(
        state, positions, state.momenta, step_index=state.step_index
    )


def test_export_import_roundtrip_reproduces_constraint_rows() -> None:
    model = _model()
    _, state = _domain(
        model, VelocityVerletPlan(0.5), 3, ensemble="nve", constrained=True
    )
    exported = export_domain_constraints(state.constraints)
    transition = import_domain_constraints(exported, state.layout)
    assert bool(transition.committed)
    restored = transition.state
    for name in ("atoms", "components", "targets", "lagrange"):
        np.testing.assert_array_equal(
            np.asarray(state.constraints.rows.collect(getattr(state.constraints, name))),
            np.asarray(restored.rows.collect(getattr(restored, name))),
        )
    np.testing.assert_array_equal(
        np.asarray(restored.halo.route_columns),
        np.asarray(state.constraints.halo.route_columns),
    )


def _rows_with(
    rows: AtomisticDomainConstraintRows, **changes: Any
) -> AtomisticDomainConstraintRows:
    fields = {
        name: getattr(rows, name)
        for name in (
            "row_ids",
            "atoms",
            "components",
            "targets",
            "lagrange",
            "active",
            "position_residual",
            "velocity_residual",
            "topology_epoch",
            "row_capacity",
            "row_count",
            "halo_capacity",
            "directory_capacity",
            "migration_capacity",
            "plan_id",
        )
    }
    fields.update(changes)
    return AtomisticDomainConstraintRows(**fields)


def test_constraint_capacity_refusal_is_distinct_from_scientific_failure() -> None:
    model = _model()
    _, state = _domain(
        model, VelocityVerletPlan(0.5), 2, ensemble="nve", constrained=True
    )
    exported = export_domain_constraints(state.constraints)
    capacity = int(AtomisticStepRejectionReason.ROUTE_CAPACITY)
    scientific = int(AtomisticStepRejectionReason.CONSTRAINT)
    # Both components are anchored on owner 0 and each needs one remote H.
    halo = import_domain_constraints(_rows_with(exported, halo_capacity=1), state.layout)
    assert not bool(halo.committed)
    assert int(halo.evidence.reasons) & capacity
    assert not int(halo.evidence.reasons) & scientific
    directory = import_domain_constraints(
        _rows_with(exported, directory_capacity=1), state.layout
    )
    assert not bool(directory.committed)
    assert int(directory.evidence.reasons) & capacity
    assert not int(directory.evidence.reasons) & scientific
    assert int(directory.evidence.maximum_directory_load) > 1
    # A forged component anchor splits one component over two ids: scientific.
    components = exported.components.copy()
    components[int(np.flatnonzero(exported.row_ids == 1)[0])] = 1
    split = import_domain_constraints(
        _rows_with(exported, components=components), state.layout
    )
    assert not bool(split.committed)
    assert int(split.evidence.reasons) & scientific
    assert not int(split.evidence.reasons) & capacity
    assert int(split.evidence.component_conflicts) + int(split.evidence.split_members) > 0


def test_rebuild_preserves_constraint_capacity_refusal_and_rolls_back() -> None:
    model = _model()
    plan = _domain_plan(model, 2)
    runtime = prepare_atomistic_domain(plan, model)
    constraints = _constraint_plan(halo_capacity=2)
    rows = domain_constraint_rows(constraints, plan.ownership, PAIRS, _distances())
    numbers = jnp.asarray(NUMBERS)
    species = model.species_indices(numbers, jnp.ones(numbers.shape, dtype=jnp.bool_))
    state = runtime.ingress_global(
        POSITIONS,
        species,
        rng_key=jax.random.key(2),
        masses=MASSES,
        cell_vectors=CELL,
        stable_ids=np.arange(numbers.size, dtype=np.int64),
        constraints=rows,
    )
    # Two remote members fit initially. Moving atom 2 across the owner
    # boundary requires a third member without changing the halo capacity.
    moving = state.layout.active & (state.layout.stable_ids == 2)
    positions = jnp.where(
        moving[:, None], state.positions + 0.5 * jnp.asarray(CELL[0]), state.positions
    )
    candidate = runtime.with_dynamics(
        state, positions, state.momenta, step_index=state.step_index
    )
    transition = runtime.rebuild(candidate)
    assert not bool(transition.committed)
    assert not bool(transition.constraints_committed)
    assert bool(transition.migration.committed)
    assert bool(transition.cell_admissible)
    assert all(bool(item.successful) for item in transition.topologies)
    assert int(transition.reasons) == int(AtomisticStepRejectionReason.ROUTE_CAPACITY)
    assert jax.tree.structure(transition.state) == jax.tree.structure(candidate)
    for actual, expected in zip(
        jax.tree.leaves(transition.state), jax.tree.leaves(candidate), strict=True
    ):
        assert actual.dtype == expected.dtype
        if jax.dtypes.issubdtype(expected.dtype, jax.dtypes.prng_key):
            actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_rebuild_hook_is_traceable_and_structure_preserving() -> None:
    model = _model()
    _, state = _domain(
        model, VelocityVerletPlan(0.5), 2, ensemble="nve", constrained=True
    )
    transition = eqx.filter_jit(rebuild_domain_constraints)(
        state.constraints, state.layout
    )
    assert bool(transition.committed)
    assert jax.tree.structure(transition.state) == jax.tree.structure(state.constraints)


def test_reductions_with_empty_owners() -> None:
    model = _model()
    # Every atom has fractional x in (0.25, 0.75): owners 0 and 3 are empty.
    dynamics, state = _domain(
        model, VelocityVerletPlan(0.5), 4, ensemble="nve", constrained=True
    )
    counts = np.bincount(
        np.fromiter(_owner_of(state.layout).values(), dtype=np.int64), minlength=4
    )
    assert counts[0] == 0 and counts[3] == 0
    reductions = dynamics.reductions(state)
    layout = state.layout
    momenta = np.asarray(layout.collect(state.momenta))
    masses = np.asarray(layout.collect(state.masses))
    kinetic = 0.5 * UNITS.kinetic_to_energy * np.sum(momenta**2 / masses[:, None])
    np.testing.assert_allclose(float(reductions.kinetic_energy), kinetic, rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(reductions.twice_kinetic_tensor),
        UNITS.kinetic_to_energy
        * np.einsum("ni,nj,n->ij", momenta, momenta, 1.0 / masses),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(reductions.linear_momentum), momenta.sum(axis=0), atol=1e-14
    )
    assert int(reductions.atom_count) == 6
    assert int(reductions.constraint_count) == 6
    assert int(reductions.degrees_of_freedom) == 3 * 6 - 6
    diagnostics = dynamics.diagnostics(state)
    np.testing.assert_allclose(
        float(diagnostics.temperature),
        2.0 * kinetic / (12 * UNITS.boltzmann_constant),
        rtol=1e-12,
    )
    ownership = layout.plan
    axis = ownership.axis_name
    reduction = dynamics.plan.domain.reduction
    row = jax.sharding.PartitionSpec(axis)
    point = jax.sharding.PartitionSpec(axis, None)
    replicated = jax.sharding.PartitionSpec()
    inner, total = ownership.map(
        lambda p, m, a: (
            owner_inner_product(p, p, a, reduction=reduction, axis_name=axis),
            owner_masked_sum(m, a, reduction=reduction, axis_name=axis),
        ),
        (point, row, row),
        (replicated, replicated),
    )(state.momenta, state.masses, layout.active)
    np.testing.assert_allclose(float(inner), float(np.sum(momenta**2)), rtol=1e-13)
    np.testing.assert_allclose(float(total), float(MASSES.sum()), rtol=1e-14)


def test_initialize_installs_typed_controllers_and_ledger() -> None:
    model = _model()
    chain = NoseHooverChainPlan(0.5, NoseHooverChainParameters(20.0, chain_length=3))
    dynamics, state = _domain(model, chain, 2, ensemble="nvt", temperature=TEMPERATURE)
    controllers = state.controllers
    assert controllers is not None and state.ledger is not None
    assert controllers.particle_chain is not None
    assert controllers.particle_chain.velocities.shape == (3,)
    assert controllers.cell is None and controllers.gle is None
    assert int(controllers.move_counter) == 0
    assert int(state.ledger.accepted_steps) == 0
    reductions = dynamics.reductions(state)
    np.testing.assert_allclose(
        float(state.ledger.kinetic_energy), float(reductions.kinetic_energy), rtol=1e-14
    )
    np.testing.assert_allclose(
        float(state.ledger.potential_energy), float(state.force.energy), rtol=1e-14
    )
    assert bool(dynamics.force_cache_valid(state))
    # A second initialization of an already initialized state is refused.
    with pytest.raises(ValueError, match="fresh ingress"):
        dynamics.initialize(state)
    # Later transitions keep the installed structure.
    stepped = _domain_step(dynamics, state).accepted_state
    assert jax.tree.structure(stepped) == jax.tree.structure(state)


def test_volume_move_without_strain_response_is_not_poisoned_by_stress() -> None:
    model = _model()
    integrator = BAOABLangevinPlan(0.5, 0.05)
    barostat = IsotropicMonteCarloBarostatPlan(0.05, realization_id=1)
    dynamics, state = _domain(
        model,
        integrator,
        2,
        ensemble="npt",
        temperature=TEMPERATURE,
        pressure=1.0e-3,
        dynamic_cell=True,
        barostat=barostat,
    )
    assert not dynamics.plan.strain_response
    for _ in range(3):
        move = _domain_volume_move(dynamics, state)
        assert bool(move.successful), int(move.rejection_reasons)
        assert not int(move.rejection_reasons) & int(
            AtomisticStepRejectionReason.NONFINITE
        )
        assert bool(np.all(np.isnan(np.asarray(move.candidate_state.force.stress))))
        state = move.accepted_state
    assert int(state.controllers.move_counter) == 3


def test_stale_force_rejects_and_returns_the_accepted_state_bitwise() -> None:
    model = _model()
    dynamics, state = _domain(model, VelocityVerletPlan(0.5), 2, ensemble="nve")
    stale = eqx.tree_at(lambda value: value.force.step, state, state.force.step - 1)
    evaluation = _domain_step(dynamics, stale)
    assert not bool(evaluation.successful)
    assert int(evaluation.rejection_reasons) & int(
        AtomisticStepRejectionReason.STALE_FORCE
    )
    for left, right in zip(
        jax.tree.leaves(evaluation.accepted_state), jax.tree.leaves(stale), strict=True
    ):
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_unsupported_combinations_refuse_before_stepping() -> None:
    model = _model()
    plan = _domain_plan(model, 2)
    nve = AtomisticDomainEnsemblePlan(UNITS, ensemble="nve", atom_count=6)
    nvt = AtomisticDomainEnsemblePlan(
        UNITS, ensemble="nvt", atom_count=6, temperature=TEMPERATURE
    )
    # GLE runs distributed; its per-atom memory must be declared at ingress.
    runtime = prepare_atomistic_domain(plan, model)
    gle = AtomisticDomainDynamicsPlan(plan, _gle(), nvt).prepare(runtime)
    mask = jnp.ones(NUMBERS.shape, dtype=jnp.bool_)
    bare = runtime.ingress_global(
        POSITIONS,
        np.asarray(model.species_indices(jnp.asarray(NUMBERS), mask)),
        rng_key=jax.random.key(2),
        masses=MASSES,
        cell_vectors=CELL,
    )
    with pytest.raises(ValueError, match="atom_payload"):
        gle.initialize(bare)
    # P5 _barostat._admit: a non-canonical particle method, a cell-owning
    # integrator and atomic scaling of constrained molecules are refused.
    dynamic_plan = _domain_plan(model, 2, dynamic_cell=True)
    npt_atomic = AtomisticDomainEnsemblePlan(
        UNITS, ensemble="npt", atom_count=6, temperature=TEMPERATURE, pressure=0.0
    )
    barostat = IsotropicMonteCarloBarostatPlan(0.05)
    with pytest.raises(ValueError, match="canonical"):
        AtomisticDomainDynamicsPlan(
            dynamic_plan,
            VelocityVerletPlan(0.5),
            AtomisticDomainEnsemblePlan(UNITS, ensemble="nve", atom_count=6),
            barostat=barostat,
        )
    with pytest.raises(ValueError, match="Atomic volume scaling"):
        AtomisticDomainDynamicsPlan(
            dynamic_plan,
            BAOABLangevinPlan(0.5, 0.05),
            npt_atomic,
            constraints=_constraint_plan(),
            barostat=barostat,
        )
    with pytest.raises(ValueError, match="dynamic_cell"):
        AtomisticDomainDynamicsPlan(
            plan, BAOABLangevinPlan(0.5, 0.05), npt_atomic, barostat=barostat
        )
    # P5 AtomisticPhaseSpaceMeasurePlan: fixed atoms forbid a center-of-mass
    # restriction.
    with pytest.raises(ValueError, match="center-of-mass"):
        AtomisticDomainEnsemblePlan(
            UNITS,
            ensemble="nve",
            atom_count=6,
            mobile_count=5,
            momentum_restriction="center-of-mass",
        )
    with pytest.raises(ValueError):
        AtomisticDomainDynamicsPlan(plan, VelocityVerletPlan(0.5), nvt)
    with pytest.raises(ValueError):
        AtomisticDomainDynamicsPlan(
            plan,
            BAOABLangevinPlan(0.5, 0.05),
            AtomisticDomainEnsemblePlan(
                UNITS,
                ensemble="nvt",
                atom_count=6,
                temperature=TEMPERATURE,
                momentum_restriction="center-of-mass",
            ),
        )
    npt = AtomisticDomainEnsemblePlan(
        UNITS, ensemble="npt", atom_count=6, temperature=TEMPERATURE, pressure=0.0
    )
    mtk = MTKNPTPlan(
        0.5,
        MTKBarostatPlan(100.0),
        NoseHooverChainParameters(20.0),
        NoseHooverChainParameters(100.0),
    )
    with pytest.raises(ValueError, match="dynamic_cell"):
        AtomisticDomainDynamicsPlan(plan, mtk, npt)
    dynamic = _domain_plan(model, 2, dynamic_cell=True)
    with pytest.raises(ValueError, match="Constrained MTK"):
        AtomisticDomainDynamicsPlan(dynamic, mtk, npt, constraints=_constraint_plan())
    with pytest.raises(ValueError, match="degrees of freedom"):
        AtomisticDomainDynamicsPlan(
            plan,
            VelocityVerletPlan(0.5),
            AtomisticDomainEnsemblePlan(UNITS, ensemble="nve", atom_count=2),
            constraints=_constraint_plan(),
        )
    with pytest.raises(ValueError, match="intensive"):
        AtomisticDomainEnsemblePlan(UNITS, ensemble="nvt", atom_count=6)
    # A constraint-free state cannot enter a constrained plan (and vice versa).
    runtime = prepare_atomistic_domain(plan, model)
    constrained = AtomisticDomainDynamicsPlan(
        plan, VelocityVerletPlan(0.5), nve, constraints=_constraint_plan()
    ).prepare(runtime)
    mask = jnp.ones(NUMBERS.shape, dtype=jnp.bool_)
    state = runtime.ingress_global(
        POSITIONS,
        np.asarray(model.species_indices(jnp.asarray(NUMBERS), mask)),
        rng_key=jax.random.key(2),
        masses=MASSES,
        cell_vectors=CELL,
    )
    with pytest.raises(ValueError, match="constraint rows"):
        constrained.initialize(state)


def test_scientific_identity_is_layout_independent_and_binds_the_physics() -> None:
    """Owner layout, migration and constraint capacities change only the
    layout-bound identities; model, method, ensemble and constraint
    definition each change the cached scientific identity."""
    model = _model()
    nve = AtomisticDomainEnsemblePlan(UNITS, ensemble="nve", atom_count=6)
    nvt = AtomisticDomainEnsemblePlan(
        UNITS, ensemble="nvt", atom_count=6, temperature=TEMPERATURE
    )

    def prepared(
        owners: int,
        integrator: Any,
        ensemble: AtomisticDomainEnsemblePlan,
        *,
        potential: MACEPotential = model,
        constraints: AtomisticDomainConstraintPlan | None = None,
        migration_capacity: int = 8,
    ) -> PreparedAtomisticDomainDynamics:
        plan = _domain_plan(potential, owners, migration_capacity=migration_capacity)
        return AtomisticDomainDynamicsPlan(
            plan, integrator, ensemble, constraints=constraints
        ).prepare(prepare_atomistic_domain(plan, potential))

    two = prepared(2, BAOABLangevinPlan(0.5, 0.05), nvt)
    three = prepared(3, BAOABLangevinPlan(0.5, 0.05), nvt, migration_capacity=16)
    assert two.prepared_id != three.prepared_id
    assert two.runtime.runtime_identity.fingerprint() != (
        three.runtime.runtime_identity.fingerprint()
    )
    assert two.scientific_id == three.scientific_id
    assert two.runtime.scientific_id == three.runtime.scientific_id
    assert two.runtime.hamiltonian_revision_id == three.runtime.hamiltonian_revision_id
    # Constraint capacities size buffers; the method and the global row count
    # are physics.
    small = prepared(2, VelocityVerletPlan(0.5), nve, constraints=_constraint_plan())
    large = prepared(
        3,
        VelocityVerletPlan(0.5),
        nve,
        constraints=_constraint_plan(halo_capacity=16, directory_capacity=32),
    )
    assert small.plan.constraints is not None and large.plan.constraints is not None
    assert small.plan.constraints.plan_id != large.plan.constraints.plan_id
    assert small.scientific_id == large.scientific_id
    assert prepared(2, VelocityVerletPlan(0.5), nve).scientific_id != small.scientific_id
    tighter = AtomisticDomainConstraintPlan(
        DistanceConstraintPlan(maximum_iterations=64, tolerance=1.0e-12),
        constraint_count=PAIRS.shape[0],
        row_capacity=PAIRS.shape[0],
        halo_capacity=8,
        directory_capacity=16,
        migration_capacity=PAIRS.shape[0],
        component_row_capacity=3,
    )
    assert (
        prepared(2, VelocityVerletPlan(0.5), nve, constraints=tighter).scientific_id
        != small.scientific_id
    )
    # Integrator, thermodynamic state and model weights are physics.
    assert prepared(2, BAOABLangevinPlan(0.25, 0.05), nvt).scientific_id != (
        two.scientific_id
    )
    warmer = AtomisticDomainEnsemblePlan(
        UNITS, ensemble="nvt", atom_count=6, temperature=TEMPERATURE + 50.0
    )
    hotter = prepared(2, BAOABLangevinPlan(0.5, 0.05), warmer)
    assert hotter.scientific_id != two.scientific_id
    # The domain physics is shared; the method's thermodynamic state is not.
    assert hotter.runtime.scientific_id == two.runtime.scientific_id
    assert hotter.runtime.hamiltonian_revision_id == two.runtime.hamiltonian_revision_id
    other = prepared(2, BAOABLangevinPlan(0.5, 0.05), nvt, potential=_model(1))
    assert other.runtime.model_revision_id != two.runtime.model_revision_id
    assert other.runtime.scientific_id != two.runtime.scientific_id
    assert other.runtime.hamiltonian_revision_id != two.runtime.hamiltonian_revision_id
    assert other.scientific_id != two.scientific_id
    # Numeric-only changes of one layout keep one treedef (one executable): the
    # identities are dynamic words, never static content-bearing strings.
    assert jax.tree.structure(hotter) == jax.tree.structure(two)
    assert jax.tree.structure(other) == jax.tree.structure(two)

from itertools import product
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._thermal import NoseHooverChainParameters, NoseHooverChainPlan
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_force_field,
    prepare_program,
    role_capacity,
    search_capacity,
)


# Finite three-atom fixtures: explicit nonperiodic search frame, the three
# pairs (two per receiver) of the former dense neighborhood and the complete
# three-pair budget of cutoff-free pair terms.
_RESOURCES: dict[str, Any] = {
    "search": phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
    "skin": 0.5,
    "search_capacities": (search_capacity(16, particles_per_cell=3, images=27),),
    "role_capacities": (role_capacity(3, 2),),
    "maximum_pairs": 3,
    "frame": phx.discretization.PeriodicCell(
        20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
    ),
}


def _system(*, coordinate_map: Any = None, cell: Any = None) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    plan = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0],
        # ty: ignore[invalid-argument-type]
        charges=[0.4, -0.2, -0.2],
        coordinate_map=coordinate_map,
        cell=cell,
    )
    return plan.prepare(), units


def _runtime() -> Any:
    system, _ = _system()
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
        ),
        system,
        **_RESOURCES,
    )
    # Canonical rows at two temperatures need a canonical NVT method.
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        NoseHooverChainPlan(1e-3, NoseHooverChainParameters(0.5)),
    ).prepare()
    measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system)
    thermodynamic = phx.atomistic.PreparedThermodynamicStateTable(
        dynamics,
        (
            phx.atomistic.AtomisticThermodynamicStatePlan(
                measure, ensemble="nvt", temperature=1.0, state_id="ecosystem-low"
            ),
            phx.atomistic.AtomisticThermodynamicStatePlan(
                measure, ensemble="nvt", temperature=2.0, state_id="ecosystem-high"
            ),
        ),
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        state_index=0,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
    return system, potential, dynamics, thermodynamic, state


def test_ecosystem_scenario_1() -> None:
    physical_ids = np.asarray([10, 20, 30])
    sites = phx.atomistic.AtomisticInteractionSitePlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30, 40],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1, 0],
        # ty: ignore[invalid-argument-type]
        [0, 0, 0, 1],
        # ty: ignore[invalid-argument-type]
        [0.4, -0.2, -0.2, 0.1],
        # ty: ignore[invalid-argument-type]
        physical_mask=[True, True, True, False],
    )
    rule = phx.atomistic.VirtualSiteRule(
        phx.atomistic.VirtualSiteKind.LOCAL_FRAME,
        40,
        physical_ids,
        # ty: ignore[invalid-argument-type]
        [0.2, 0.1, 0.0],
    )
    mapping = phx.atomistic.AtomisticCoordinateMapPlan(
        physical_ids,
        sites,
        # ty: ignore[invalid-argument-type]
        [0, 1, 2, -1],
        virtual_rules=(rule,),
    )
    system, _ = _system(coordinate_map=mapping)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    realized = system.coordinate_map.realize(positions)
    assert bool(realized.successful)
    np.testing.assert_allclose(realized.positions[3], [0.2, 0.1, 0.0], atol=1e-12)
    system, units = _system()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    terms = [
        # ty: ignore[invalid-argument-type]
        phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5),
        # ty: ignore[invalid-argument-type]
        phx.atomistic.BuckinghamPotential([[1.0]], [[2.0]], [[0.1]], 2.5),
    ]
    program = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(terms), system, **_RESOURCES
    )
    evaluation = program.evaluate(positions, bundle_state(program, positions))
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    provenance = phx.atomistic.AtomisticForceFieldProvenance(
        "native", ("source",), "custom", "explicit"
    )
    bundle = prepare_force_field(
        phx.atomistic.AtomisticForceFieldPlan(
            system.plan,
            phx.atomistic.AtomisticPotentialProgram(
                (*terms, phx.atomistic.DirectCoulombPotential())
            ),
            phx.atomistic.AtomisticNonbondedPolicy(2.5, electrostatics="direct"),
            provenance,
        ),
        **_RESOURCES,
    )
    assert dict(bundle.preparation.resource_counts)["interaction_sites"] == 3
    assert bundle.system.plan.units.unit_system_id == units.unit_system_id
    system, _, dynamics, thermodynamic, state = _runtime()
    cv = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    program = phx.atomistic.sampling.CollectiveVariableProgram((cv,))
    geometry = phx.atomistic.sampling.CollectiveVariableGeometry(
        state.kinematics.positions
    )
    evaluation = program.evaluate(geometry)
    np.testing.assert_allclose(evaluation.values, [1.2])
    assert bool(evaluation.successful)
    bias = phx.atomistic.sampling.AtomisticBiasTerm(
        program,
        phx.atomistic.sampling.HarmonicBiasForm(np.asarray([1.0]), 2.0),
        name="distance-umbrella",
        force_group=1,
    ).prepare(system)
    bias_energy, _, bias_valid = bias.bias(geometry, bias.initial_term_state(jnp.float64))
    assert bool(bias_valid)
    np.testing.assert_allclose(bias_energy, 0.04, atol=1e-12)
    replica_plan = phx.atomistic.sampling.AtomisticMultistatePlan(
        thermodynamic,
        # ty: ignore[invalid-argument-type]
        [10, 20],
        qualification=phx.atomistic.sampling.AtomisticCanonicalSamplingQualification(
            dynamics,
            thermodynamic,
            "ecosystem-synthetic-kernel-qualification",
            sampling_exact=False,
            sampling_bias_bound=1.0,
        ),
        exchange=phx.atomistic.sampling.AtomisticReplicaExchangePlan(1),
        run_id="ecosystem-exchange",
    ).prepare(dynamics)
    replica = replica_plan.initialize(
        (
            state,
            dynamics.initialize_state(
                state.kinematics.positions,
                thermodynamic,
                state_index=1,
                momentum=state.kinematics.momenta,
                key=jax.random.key(3),
            ),
        ),
        # ty: ignore[invalid-argument-type]
        [0, 1],
        jax.random.key(4),
    )
    exchanged = replica_plan.iterate(replica)
    assert bool(exchanged.successful)


def test_frame_xyz_h5md_and_rerun_roundtrip(tmp_path: Path) -> None:
    system, potential, dynamics, _, state = _runtime()
    reporter = phx.atomistic.AtomisticReporterPlan(
        phx.atomistic.interchange.ExtendedXYZTrajectoryPlan(tmp_path / "trajectory.xyz")
    )
    frame = reporter.frame(dynamics, state)
    with reporter.sink.open(append=False) as writer:
        writer.write(frame)
    # ty: ignore[missing-argument]
    with reporter.sink.open() as reader:
        # ty: ignore[invalid-argument-type]
        observed = tuple(reader)
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0].positions, frame.positions)
    h5 = phx.atomistic.interchange.H5MDTrajectoryPlan(tmp_path / "trajectory.h5")
    with h5.open(append=False) as writer:
        writer.write(frame)
    with h5.open() as reader:
        h5_frames = tuple(reader)
    np.testing.assert_allclose(h5_frames[0].positions, frame.positions)
    source = phx.atomistic.InMemoryTrajectorySourcePlan((frame,))
    rerun = phx.atomistic.AtomisticRerunPlan(source, potential).run()
    assert bool(rerun.successful)
    np.testing.assert_allclose(
        rerun.evaluations[0][0].energy, state.force.potential_energy
    )


def test_ecosystem_scenario_2() -> None:
    system, potential, _, _, state = _runtime()
    other = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.21], [1.0], 2.5)]
        ),
        system,
        **_RESOURCES,
    )
    committee = phx.atomistic.CommitteeAtomisticPotential(
        (potential, other),
        phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0),
    )
    # Members with the same prepared resource bundle share one accepted state.
    assert len(committee.bundles) == 1 and committee.member_bundles == (0, 0)
    evidence = committee.evaluate(state.kinematics.positions, (state.neighborhoods,))
    assert bool(evidence.successful)
    multipoles = phx.atomistic.PermanentMultipoleSiteData(
        # ty: ignore[invalid-argument-type]
        [0.4, -0.2, -0.2],
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3, 3)),
        # ty: ignore[invalid-argument-type]
        [0.1, 0.1, 0.1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
    )
    polarization = phx.atomistic.evaluate_prepared_polarization(
        phx.atomistic.PolarizationPlan(
            coulomb_constant=system.plan.units.coulomb_constant,
            maximum_iterations=200,
            tolerance=1e-6,
        ).prepare(multipoles),
        state.kinematics.positions,
    )
    assert bool(polarization.successful)
    assert polarization.evidence.mode == "implicit-stationary"
    normal = phx.atomistic.RingPolymerNormalModePlan(2, 1.0)
    q = jnp.stack((state.kinematics.positions, state.kinematics.positions))
    p = jnp.zeros_like(q)
    propagated_q, propagated_p = normal.propagate(q, p, system.plan.masses, 0.01)
    assert propagated_q.shape == q.shape
    assert propagated_p.shape == p.shape
    # Test-local labeled reference, not a product route: the single-device
    # classical atom energies split by a declared fractional owner grid of the
    # frame own every atom energy exactly once (the distributed product route
    # is the layered domain runtime, `test_domain_execution.py`).
    frame = _RESOURCES["frame"]
    partition = FractionalOwnerPartition(frame, (2, 1, 1))
    owners = np.asarray(partition.owners(frame.fractional(state.kinematics.positions)))
    atom_energy = np.asarray(state.force.atom_energy).reshape((-1,))[: owners.size]
    shares = np.bincount(owners, weights=atom_energy, minlength=2)
    assert shares.shape == (2,)
    np.testing.assert_allclose(
        np.sum(shares), np.asarray(state.force.potential_energy).sum(), atol=1.0e-12
    )
    physical_ids = np.asarray([10, 20, 30])
    sites = phx.atomistic.AtomisticInteractionSitePlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30, 40],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1, 0],
        # ty: ignore[invalid-argument-type]
        [0, 0, 0, 1],
        # ty: ignore[invalid-argument-type]
        [0.4, -0.2, -0.2, 0.1],
        # ty: ignore[invalid-argument-type]
        physical_mask=[True, True, True, False],
    )
    rule = phx.atomistic.VirtualSiteRule(
        phx.atomistic.VirtualSiteKind.LOCAL_FRAME,
        40,
        physical_ids,
        # ty: ignore[invalid-argument-type]
        [0.2, 0.1, 0.0],
    )
    mapping = phx.atomistic.AtomisticCoordinateMapPlan(
        physical_ids,
        sites,
        # ty: ignore[invalid-argument-type]
        [0, 1, 2, -1],
        virtual_rules=(rule,),
    )
    system, _ = _system(coordinate_map=mapping)
    cv = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE,
        [0, 3],
        domain=phx.atomistic.AtomisticSiteDomain.INTERACTION_SITES,
    ).prepare(system)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    geometry = phx.atomistic.sampling.CollectiveVariableGeometry
    # Interaction sites are realized by the system coordinate map.
    np.testing.assert_allclose(
        cv.evaluate(geometry(positions)).value, np.hypot(0.2, 0.1), atol=1e-12
    )
    with pytest.raises(ValueError, match="fractions and cell vectors"):
        cv.evaluate(geometry(positions, cell_vectors=10.0 * jnp.eye(3)))


def test_committee_members_bind_independently_valid_resources() -> None:
    system, potential, _, _, state = _runtime()
    shorter = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.0)]
        ),
        system,
        **_RESOURCES,
    )
    policy = phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0)
    committee = phx.atomistic.CommitteeAtomisticPotential(
        (potential, shorter, potential), policy
    )
    assert len(committee.bundles) == 2 and committee.member_bundles == (0, 1, 0)
    positions = state.kinematics.positions
    resources = committee.resources(positions)
    assert resources.committee is committee and resources.capacity_growths == 0
    assert not resources.exhausted
    evaluation = committee.evaluate(positions, resources.neighborhoods)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        evaluation.member_energies[1],
        shorter.evaluate(positions, resources.neighborhoods[1]).energy,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        evaluation.member_energies[0],
        potential.evaluate(positions, resources.neighborhoods[0]).energy,
        rtol=1e-12,
    )
    moved = positions.at[1, 0].add(0.05)
    refreshed = committee.resources(moved, resources.neighborhoods)
    assert bool(committee.evaluate(moved, refreshed.neighborhoods).successful)
    with pytest.raises(ValueError, match="one state per committee bundle"):
        committee.evaluate(positions, resources.neighborhoods[:1])
    blend = phx.atomistic.ConservativeUncertaintyBlend(committee, shorter)
    energy, forces, uncertainty, weight = blend.evaluate(
        positions, resources.neighborhoods, resources.neighborhoods[1]
    )
    assert np.isfinite(float(energy)) and bool(uncertainty.successful)
    assert forces.shape == positions.shape and 0.0 <= float(weight) <= 1.0


def test_committee_resources_recover_capacity_on_declared_ladders() -> None:
    system, potential, _, _, state = _runtime()
    grown = {
        **_RESOURCES,
        "role_capacities": (role_capacity(1, 1), role_capacity(3, 2)),
    }
    small = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.0)]
        ),
        system,
        **grown,
    )
    committee = phx.atomistic.CommitteeAtomisticPotential(
        (potential, small), phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0)
    )
    resources = committee.resources(state.kinematics.positions)
    assert resources.capacity_growths >= 1 and not resources.exhausted
    assert resources.committee.committee_id != committee.committee_id
    assert bool(
        resources.committee.evaluate(
            state.kinematics.positions, resources.neighborhoods
        ).successful
    )


def test_rerun_frames_bind_independently_valid_recovered_resources(
    tmp_path: Path,
) -> None:
    system, _, dynamics, _, state = _runtime()
    reporter = phx.atomistic.AtomisticReporterPlan(
        phx.atomistic.interchange.ExtendedXYZTrajectoryPlan(tmp_path / "frames.xyz")
    )
    first = reporter.frame(dynamics, state)
    # The second frame moves beyond the skin: its resources are rebuilt.
    moved = eqx.tree_at(
        lambda frame: frame.positions, first, first.positions.at[1, 0].add(0.6)
    )
    small = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
        ),
        system,
        **{**_RESOURCES, "role_capacities": (role_capacity(1, 1), role_capacity(3, 2))},
    )
    result = phx.atomistic.AtomisticRerunPlan(
        phx.atomistic.InMemoryTrajectorySourcePlan((first, moved)), small
    ).run()
    assert bool(result.successful)
    assert result.capacity_growths >= 1 and not result.capacity_exhausted
    for frame, row in zip((first, moved), result.evaluations, strict=True):
        expected = phx.atomistic.LennardJonesPotential(
            np.asarray([0.2]), np.asarray([1.0]), 2.5
        )
        reference = prepare_program(
            phx.atomistic.AtomisticPotentialProgram([expected]),
            system,
            **_RESOURCES,
        )
        np.testing.assert_allclose(
            row[0].energy,
            reference.evaluate(
                frame.positions, bundle_state(reference, frame.positions)
            ).energy,
            rtol=1e-12,
        )
    exhausted = phx.atomistic.AtomisticRerunPlan(
        phx.atomistic.InMemoryTrajectorySourcePlan((first,)),
        prepare_program(
            phx.atomistic.AtomisticPotentialProgram(
                # ty: ignore[invalid-argument-type]
                [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
            ),
            system,
            **{**_RESOURCES, "role_capacities": (role_capacity(1, 1),)},
        ),
    ).run()
    assert exhausted.capacity_exhausted and not bool(exhausted.successful)


def _virtual_site_reference(
    positions: np.ndarray,
    vectors: np.ndarray,
    kind: phx.atomistic.VirtualSiteKind,
) -> np.ndarray:
    """Independent bounded lattice search for the small skew-cell fixture."""
    shifts = np.asarray(
        tuple(product(range(-2, 3), repeat=vectors.shape[0])), dtype=np.int32
    )
    candidates = (positions[1:] - positions[0])[:, None, :] - (shifts @ vectors)[
        None, :, :
    ]
    selected = np.argmin(np.sum(candidates * candidates, axis=-1), axis=-1)
    first, second = candidates[np.arange(2), selected]
    match kind:
        case phx.atomistic.VirtualSiteKind.WEIGHTED:
            return positions[0] + 0.75 * first
        case phx.atomistic.VirtualSiteKind.LOCAL_FRAME:
            x_axis = first / np.linalg.norm(first)
            normal = np.cross(first, second)
            z_axis = normal / np.linalg.norm(normal)
            y_axis = np.cross(z_axis, x_axis)
            return positions[0] + 0.2 * x_axis + 0.1 * y_axis + 0.05 * z_axis
        case _:
            raise ValueError("The reference requires an explicit virtual-site kind.")


@pytest.mark.parametrize("rank", [2, 3], ids=["embedded-lattice", "full-lattice"])
@pytest.mark.parametrize(
    "kind",
    [phx.atomistic.VirtualSiteKind.WEIGHTED, phx.atomistic.VirtualSiteKind.LOCAL_FRAME],
    ids=["weighted", "local-frame"],
)
def test_dynamic_virtual_sites_preserve_cartesian_geometry_and_derivatives(
    rank: int, kind: phx.atomistic.VirtualSiteKind
) -> None:
    vectors = np.asarray(
        [[3.0, 0.0, 0.0], [1.2, 2.5, 0.0], [0.0, 0.0, 4.0]],
        dtype=np.float64,
    )[:rank]
    positions = np.asarray(
        [[0.1, 0.2, 0.3], [2.158, 1.425, 0.9], [0.3, 0.3, 1.4]],
        dtype=np.float64,
    )
    cell = phx.discretization.PeriodicCell(vectors)
    physical_ids = np.asarray([10, 20, 30], dtype=np.int64)
    sites = phx.atomistic.AtomisticInteractionSitePlan(
        np.asarray([10, 20, 30, 40], dtype=np.int64),
        np.asarray([1, 1, 1, 0], dtype=np.int32),
        np.asarray([0, 0, 0, 1], dtype=np.int32),
        np.zeros((4,), dtype=np.float64),
        physical_mask=np.asarray([True, True, True, False], dtype=np.bool_),
    )
    weighted = kind is phx.atomistic.VirtualSiteKind.WEIGHTED
    rule = phx.atomistic.VirtualSiteRule(
        kind,
        40,
        physical_ids[:2] if weighted else physical_ids,
        np.asarray([0.25, 0.75] if weighted else [0.2, 0.1, 0.05], dtype=np.float64),
    )
    plan = phx.atomistic.AtomisticCoordinateMapPlan(
        physical_ids,
        sites,
        np.asarray([0, 1, 2, -1], dtype=np.int32),
        virtual_rules=(rule,),
    )
    system, _ = _system(coordinate_map=plan, cell=cell)
    mapping = system.coordinate_map
    fractional = cell.fractional(jnp.asarray(positions))
    realized = mapping.realize(
        positions, cell=cell, fractional_positions=fractional, cell_vectors=cell.vectors
    )
    assert bool(realized.successful)
    np.testing.assert_allclose(
        realized.positions[3],
        _virtual_site_reference(positions, vectors, kind),
        rtol=0.0,
        atol=1e-12,
    )
    force = np.asarray([0.7, -0.4, 0.2], dtype=np.float64)
    pulled = mapping.force_pullback(
        positions,
        jnp.zeros((4, 3), dtype=jnp.float64).at[3].set(jnp.asarray(force)),
        cell=cell,
        fractional_positions=fractional,
        cell_vectors=cell.vectors,
    )
    delta = 1e-6
    expected_force = np.zeros_like(positions)
    for atom, axis in product(range(3), range(3)):
        plus, minus = positions.copy(), positions.copy()
        plus[atom, axis] += delta
        minus[atom, axis] -= delta
        expected_force[atom, axis] = (
            (
                _virtual_site_reference(plus, vectors, kind)
                - _virtual_site_reference(minus, vectors, kind)
            )
            @ force
            / (2.0 * delta)
        )
    np.testing.assert_allclose(pulled, expected_force, rtol=0.0, atol=1e-8)

    direction = np.arange(rank * 3, dtype=np.float64).reshape((rank, 3)) * 0.03 - 0.07

    def cell_response(value: jax.Array) -> jax.Array:
        return mapping.realize(
            positions, cell=cell, fractional_positions=fractional, cell_vectors=value
        ).positions[3]

    _, tangent = jax.jvp(cell_response, (cell.vectors,), (jnp.asarray(direction),))
    expected_tangent = (
        _virtual_site_reference(positions, vectors + delta * direction, kind)
        - _virtual_site_reference(positions, vectors - delta * direction, kind)
    ) / (2.0 * delta)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=0.0, atol=1e-8)
    invalid_chart = mapping.realize(
        positions,
        cell=cell,
        fractional_positions=fractional.at[1, 0].set(jnp.nan),
        cell_vectors=cell.vectors,
    )
    assert not bool(invalid_chart.successful)

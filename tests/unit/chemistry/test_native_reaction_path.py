#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device-resident grouped NEB on a native atomistic surface.

Independent control: a mobile Lennard-Jones atom crossing the window of four
fixed atoms.  The window centre is a first-order saddle by symmetry (one
negative transverse curvature, verified here only through HVP/inertia
evidence), and the endpoint minima are found by host bisection of the exact
one-dimensional force along the crossing axis.
"""

import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodSource,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
)
from phydrax.atomistic._potential_program import PreparedAtomisticPotentialProgram
from phydrax.atomistic._relaxation import (
    AtomisticHamiltonianTransitionKind,
    AtomisticRelaxationStatus,
    AtomisticSensitivityStatus,
    AtomisticTrialOutcome,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.chemistry.reaction import (
    AtomisticNudgedElasticBandPlan,
    nudged_elastic_band_forces,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.optim import FIRE
from phydrax.typing import ConvertibleToArray


_BOX = 8.0
_CENTER = 4.0
_FIXED = np.asarray(
    [
        [_CENTER - 1.0, _CENTER, _CENTER],
        [_CENTER + 1.0, _CENTER, _CENTER],
        [_CENTER, _CENTER, _CENTER - 1.1],
        [_CENTER, _CENTER, _CENTER + 1.1],
    ]
)


def _lj_derivative(distance: float) -> float:
    return 4.0 * (-12.0 * distance**-13 + 6.0 * distance**-7)


def _crossing_force(height: float) -> float:
    first = np.sqrt(1.0 + height * height)
    second = np.sqrt(1.21 + height * height)
    return (
        2.0 * _lj_derivative(first) * height / first
        + 2.0 * _lj_derivative(second) * height / second
    )


def _minimum_height() -> float:
    low, high = 0.05, 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if np.sign(_crossing_force(middle)) == np.sign(_crossing_force(low)):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _surface() -> tuple[
    phx.atomistic.AtomisticUnitSystem,
    PreparedAtomisticSystem,
    phx.chemistry.AtomisticPotentialEnergySurface,
]:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    ids = np.arange(1, 6)
    system = phx.atomistic.AtomisticSystemPlan(
        ids,
        np.ones((5,), dtype=np.int32),
        np.ones((5,)),
        units,
        atom_type_ids=np.zeros((5,), dtype=np.int32),
        mobile_mask=np.asarray([False, False, False, False, True]),
        cell=PeriodicCell(_BOX * np.eye(3)),
    ).prepare()
    program = _program(system, epsilon=1.0)
    return units, system, phx.chemistry.AtomisticPotentialEnergySurface(system, program)


def _program(
    system: PreparedAtomisticSystem, *, epsilon: float
) -> PreparedAtomisticPotentialProgram:
    plan = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([epsilon]), np.asarray([1.0]), 2.5
            )
        ]
    )
    (requirement,) = plan.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticCellListImageSearch(
            maximum_candidate_slots=10**5, deformation_margin=1.0
        ),
        ParticleImageCapacityLadder(
            (
                ParticleImageCapacity(
                    maximum_particles_per_cell=16,
                    maximum_edges=256,
                    maximum_degree=64,
                    maximum_images=512,
                    maximum_case_edges=256,
                ),
            )
        ),
        skin=0.4,
    )
    ladder = AtomisticRoleCapacityLadder(
        requirement, (AtomisticRoleCapacity(maximum_routes=256, maximum_degree=64),)
    )
    return plan.prepare(
        system,
        neighborhood_bundle=AtomisticNeighborhoodBundlePlan(
            (requirement,), (source,), (ladder,)
        ),
    )


def _structure(
    units: phx.atomistic.AtomisticUnitSystem, mobile: ConvertibleToArray
) -> phx.atomistic.AtomicStructure:
    positions = np.concatenate((_FIXED, np.asarray(mobile, dtype=np.float64)[None]))
    return phx.atomistic.AtomicStructure(
        np.ones((5,), dtype=np.int32),
        positions,
        np.ones((5,)),
        units.scale,
        particle_ids=np.arange(1, 6),
        cell=_BOX * np.eye(3),
        periodic_axes=np.asarray([True, True, True]),
    )


def test_grouped_device_neb_converges_to_the_window_saddle_band_atomically() -> None:
    units, system, surface = _surface()
    height = _minimum_height()
    reactant = _structure(units, [_CENTER, _CENTER + height, _CENTER])
    product = _structure(units, [_CENTER, _CENTER - height, _CENTER])
    count = 7
    fractions = np.linspace(0.0, 1.0, count)
    band = (
        np.asarray(reactant.positions)[None]
        + fractions[:, None, None]
        * (np.asarray(product.positions) - np.asarray(reactant.positions))[None]
    )
    bowed = band.copy()
    bowed[:, 4, 0] += 0.15 * np.sin(np.pi * fractions)
    broken = bowed.copy()
    broken[3, 4] = _FIXED[0]
    plan = AtomisticNudgedElasticBandPlan(
        surface,
        image_count=count,
        spring_constant=1.0,
        force_tolerance=1.0e-6,
        maximum_evaluations=4000,
        climbing_start=0,
        group_count=2,
        method=FIRE(maximum_step=0.05),
    )
    runtime, state, certificates = plan.prepare().init(
        [(reactant, product), (reactant, product)],
        initial_images=np.stack((bowed, broken)),
    )
    assert len(certificates) == 2
    runtime, final = runtime.solve(state, maximum_rounds=4000)
    status = np.asarray(final.status)
    assert int(status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    # The broken band fails as a whole; its accepted band is untouched.
    assert int(status[1]) == int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE)
    assert int(final.trial_outcome[1]) == int(AtomisticTrialOutcome.SCIENTIFIC)
    np.testing.assert_array_equal(np.asarray(final.images[1]), broken)

    path = runtime.path_result(final, 0)
    assert bool(path.successful)
    assert path.climbing_active
    assert path.climbing_image == count // 2
    np.testing.assert_allclose(
        path.transition_state_positions[4], [_CENTER, _CENTER, _CENTER], atol=1e-5
    )
    # Fixed endpoints and fixed atoms never move.
    np.testing.assert_array_equal(
        np.asarray(path.images[0]), np.asarray(reactant.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(path.images[-1]), np.asarray(product.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(path.images[:, :4]), np.broadcast_to(_FIXED, (count, 4, 3))
    )
    # One accepted iterate: host re-evaluation reproduces energies and forces.
    for image in range(count):
        value = surface.evaluate(np.asarray(path.images[image]))
        np.testing.assert_allclose(value.energy, path.energies[image], atol=1e-10)
        if 0 < image < count - 1:
            np.testing.assert_allclose(value.forces, path.forces[image], atol=1e-9)
    mask = np.asarray([False, False, False, False, True])
    projection = nudged_elastic_band_forces(
        path.images,
        path.energies,
        path.forces,
        mask,
        spring_constant=plan.spring_constant,
        climbing=True,
    )
    np.testing.assert_allclose(projection.forces, path.neb_forces, atol=1e-12)
    assert float(np.max(np.asarray(path.image_residual))) <= 1.0e-6

    inertia = runtime.transition_state_inertia(final, 0)
    assert int(inertia.status) == int(AtomisticSensitivityStatus.ADMITTED)
    assert int(inertia.negative_count) == 1
    refused = runtime.transition_state_inertia(final, 1)
    assert int(refused.status) == int(AtomisticSensitivityStatus.NOT_CONVERGED)

    # Model update: the whole band continues under a new scientific Hamiltonian.
    with pytest.raises(ValueError, match="same scientific identity"):
        runtime.transition_hamiltonian(runtime.program, final)
    stronger = _program(system, epsilon=1.25)
    moved = runtime.transition_hamiltonian(stronger, final)
    assert moved.record.successful
    assert moved.record.optimizer_policy == "reset-history"
    assert moved.record.kind is AtomisticHamiltonianTransitionKind.SCIENTIFIC_MODEL
    assert moved.record.scientific_id != moved.record.previous_scientific_id
    assert moved.runtime.prepared_id != runtime.prepared_id
    assert moved.state.hamiltonian_lineage == (moved.record.transition_id,)
    np.testing.assert_array_equal(np.asarray(moved.record.reprimed), [True, False])
    np.testing.assert_array_equal(
        np.asarray(moved.state.images), np.asarray(final.images)
    )
    np.testing.assert_array_equal(
        np.asarray(moved.state.evaluations), np.asarray(final.evaluations)
    )
    # Lennard-Jones energies scale linearly with epsilon at the kept band.
    np.testing.assert_allclose(
        moved.state.energies[0], 1.25 * final.energies[0], rtol=1e-12
    )
    np.testing.assert_allclose(
        moved.state.endpoint_energies[0], 1.25 * final.endpoint_energies[0], rtol=1e-12
    )
    # The failed lane is carried unchanged.
    assert int(moved.state.status[1]) == int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE)
    np.testing.assert_array_equal(
        np.asarray(moved.state.energies[1]), np.asarray(final.energies[1])
    )
    _, pending, _ = moved.runtime.init(
        [(reactant, product), (reactant, product)],
        initial_images=np.stack((bowed, broken)),
    )
    assert int(pending.status[0]) == int(AtomisticRelaxationStatus.ITERATING)
    refused = runtime.step(pending)
    assert int(refused.status[0]) == int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE)
    assert int(refused.trial_outcome[0]) == int(AtomisticTrialOutcome.SCIENTIFIC)
    np.testing.assert_array_equal(refused.images, pending.images)
    np.testing.assert_array_equal(refused.evaluations, pending.evaluations)
    updated, continued = moved.runtime.solve(moved.state, maximum_rounds=4000)
    assert int(continued.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    np.testing.assert_allclose(
        updated.path_result(continued, 0).transition_state_positions[4],
        [_CENTER, _CENTER, _CENTER],
        atol=1e-5,
    )
    assert int(continued.evaluations[0]) >= int(final.evaluations[0])

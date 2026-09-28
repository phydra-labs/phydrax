#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _manifest() -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "charged-material",
        checksum_algorithm="sha256",
        checksum="f" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:charged",),
    )


def _plan(
    *,
    stopping: Any = 1000.0,
    box_length: Any = 2.0,
    bremsstrahlung: Any = 0.0,
    step_bank_capacity: Any = None,
    photon_stack: Any = None,
) -> Any:
    materials = phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2000.0)),
        jnp.full((1, 2), stopping),
        jnp.zeros((1, 2)),
        jnp.full((1, 2), bremsstrahlung),
        ("material",),
        _manifest(),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 1.0, box_length)),
        jnp.zeros((1, 1, 2), dtype=jnp.int32),
        material_count=1,
    )
    return phx.solver.ChargedParticleTransportPlan(
        geometry,
        materials,
        maximum_steps=256,
        maximum_step_length=0.05,
        maximum_fractional_energy_loss=0.05,
        cutoff_energy_ev=10.0,
        step_bank_capacity=step_bank_capacity,
        photon_stack=photon_stack,
    )


_ELECTRON = int(phx.equations.ChargedRadiationParticleKind.ELECTRON)
_ORIGIN = jnp.asarray(((0.5, 0.5, 0.1),))
_AXIS = jnp.asarray(((0.0, 0.0, 1.0),))


def test_charged_particle_transport_scenario_1() -> None:
    plan = _plan()
    result = plan.simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((1000.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
        jr.key(3),
    )

    assert bool(result.all_successful)
    np.testing.assert_allclose(result.deposited_energy, 1000.0, atol=1e-8)
    np.testing.assert_allclose(result.path_length, 0.99, atol=2e-3)
    np.testing.assert_allclose(result.maximum_kinetic_ledger_residual, 0.0, atol=1e-9)
    escaping = _plan(stopping=1.0, box_length=0.5).simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((1000.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
        jr.key(4),
    )
    annihilating = _plan(stopping=1000.0).simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((20.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.POSITRON),)),
        jr.key(5),
    )

    assert bool(escaping.all_successful)
    assert escaping.escaped_energy[0] > 999.0
    assert bool(annihilating.all_successful)
    np.testing.assert_allclose(
        annihilating.annihilation_photon_energy, 2.0 * 510998.95069
    )
    np.testing.assert_allclose(
        annihilating.deposited_energy + annihilating.escaped_energy,
        20.0,
        atol=1e-8,
    )


def test_charged_step_bank_records_every_supported_step() -> None:
    result = _plan(step_bank_capacity=256).simulate(
        _ORIGIN, _AXIS, jnp.asarray((1000.0,)), jnp.asarray((_ELECTRON,)), jr.key(6)
    )
    bank = result.step_bank

    assert bank is not None
    assert bool(result.all_successful)
    steps = int(result.step_count[0])
    assert int(bank.recorded_count[0]) == steps
    assert bool(bank.complete[0])
    active = np.asarray(bank.active[0])
    assert active[:steps].all() and not active[steps:].any()
    starts = np.asarray(bank.start_positions[0, :steps])
    ends = np.asarray(bank.end_positions[0, :steps])
    np.testing.assert_allclose(ends[:-1], starts[1:], atol=1e-12)
    np.testing.assert_allclose(np.asarray(bank.deposited_energy[0]).sum(), 1000.0)
    rest = 510998.95069
    expected_beta = np.sqrt(1000.0 * (1000.0 + 2.0 * rest)) / (1000.0 + rest)
    assert float(bank.end_beta[0, steps - 1]) < float(bank.start_beta[0, 0])
    np.testing.assert_allclose(float(bank.start_beta[0, 0]), expected_beta, rtol=1e-9)

    partial = _plan(step_bank_capacity=4).simulate(
        _ORIGIN, _AXIS, jnp.asarray((1000.0,)), jnp.asarray((_ELECTRON,)), jr.key(6)
    )
    partial_bank = partial.step_bank
    assert partial_bank is not None
    assert bool(partial.all_successful)
    assert int(partial_bank.recorded_count[0]) == 4
    assert int(partial_bank.unrecorded_count[0]) == steps - 4
    assert not bool(partial_bank.complete[0])


def test_charged_photon_stack_makes_bremsstrahlung_real_secondaries() -> None:
    count = 64
    plan = _plan(
        stopping=1.0,
        bremsstrahlung=5.0,
        photon_stack=phx.solver.SecondaryStackSpec(64, minimum_energy=50.0),
    )
    result = plan.simulate(
        jnp.broadcast_to(_ORIGIN, (count, 3)),
        jnp.broadcast_to(_AXIS, (count, 3)),
        jnp.full((count,), 1000.0),
        jnp.full((count,), _ELECTRON, dtype=jnp.int32),
        jr.key(8),
    )
    stack = result.secondary_photons

    assert stack is not None
    assert bool(result.all_successful)
    assert int(jnp.sum(stack.count)) > 0
    energies = np.asarray(stack.energies)
    active = np.asarray(stack.active)
    assert np.all(energies[active] >= 50.0)
    np.testing.assert_allclose(
        np.where(active, energies, 0.0).sum(axis=1), np.asarray(stack.energy)
    )
    np.testing.assert_array_equal(stack.energy, result.bremsstrahlung_energy)
    closed = (
        np.asarray(result.deposited_energy)
        + np.asarray(result.escaped_energy)
        + np.asarray(result.bremsstrahlung_energy)
    )
    np.testing.assert_allclose(closed, 1000.0, atol=1e-9)
    directions = np.asarray(stack.directions)[active]
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    assert np.all(np.asarray(stack.material_index)[active] == 0)
    creation = np.asarray(stack.creation_index)[active]
    assert np.all(creation >= 0)
    # Each history's photons are created at distinct steps in increasing order.
    for row in range(count):
        row_creation = np.asarray(stack.creation_index[row])[active[row]]
        assert np.all(np.diff(row_creation) > 0)


def test_charged_history_outcome_follows_identity_not_slot() -> None:
    plan = _plan(stopping=1.0, bremsstrahlung=5.0)
    identities = (
        jnp.zeros((3,), dtype=jnp.uint32),
        jnp.asarray((5, 6, 7), dtype=jnp.uint32),
    )
    together = plan.simulate(
        jnp.broadcast_to(_ORIGIN, (3, 3)),
        jnp.broadcast_to(_AXIS, (3, 3)),
        jnp.full((3,), 1000.0),
        jnp.full((3,), _ELECTRON, dtype=jnp.int32),
        jr.key(9),
        identities=identities,
    )
    alone = plan.simulate(
        _ORIGIN,
        _AXIS,
        jnp.asarray((1000.0,)),
        jnp.asarray((_ELECTRON,)),
        jr.key(9),
        identities=(identities[0][1:2], identities[1][1:2]),
    )

    np.testing.assert_array_equal(
        alone.bremsstrahlung_energy, together.bremsstrahlung_energy[1:2]
    )
    np.testing.assert_array_equal(
        alone.terminal_position, together.terminal_position[1:2]
    )
    np.testing.assert_array_equal(alone.step_count, together.step_count[1:2])

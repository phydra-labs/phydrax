#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.nucleic_rigid import parameter_data
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.coarse import (
    nucleotide_reference_sites,
    NucleotideModelPlan,
    NucleotideParameterArtifact,
)
from phydrax.applications.protein_folding.hybrid import (
    HybridCrossInteractionPlan,
    PreparedHybridModel,
)
from phydrax.atomistic import AtomisticSystemPlan, AtomisticUnitSystem, ElasticNetworkPlan
from phydrax.atomistic._sites import (
    AtomisticCoordinateMapPlan,
    AtomisticInteractionSitePlan,
    VirtualSiteKind,
    VirtualSiteRule,
)
from phydrax.discretization.particle._rigid_body import _quaternion_retract
from phydrax.qualification import ReferenceArtifactManifest


def _source(payload: bytes, name: str, *, export: bool = True):
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Author-Owned-Numerical-Fixture",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=export,
        export_classification="numerical-fixture",
        nondimensionalization={"length": 1.0, "energy": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic-model-not-biological-calibration",),
    )


def _fixture(
    *,
    units=None,
    fixed_protein=True,
    fixed_body=False,
    order=(0, 1),
    virtual=False,
    cross_kwargs=None,
    nucleotide_export=True,
):
    units = AtomisticUnitSystem.reduced() if units is None else units
    order = np.asarray(order)
    ids = np.asarray([7, 11], dtype=np.int64)[order]
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])[order]
    coordinate_map = None
    if virtual:
        sites = AtomisticInteractionSitePlan(
            [7, 11, 90],
            [0, 0, 0],
            [0, 0, 0],
            [0.0, 0.0, 0.0],
            element_mask=[False] * 3,
            physical_mask=[True, True, False],
        )
        lookup = {int(key): slot for slot, key in enumerate(ids)}
        coordinate_map = AtomisticCoordinateMapPlan(
            ids,
            sites,
            [lookup[7], lookup[11], -1],
            virtual_rules=(
                VirtualSiteRule(VirtualSiteKind.WEIGHTED, 90, [7, 11], [0.5, 0.5]),
            ),
        )
    system = AtomisticSystemPlan(
        ids,
        np.zeros(2, dtype=np.int32),
        np.asarray([2.0, 3.0])[order],
        units,
        element_mask=[False, False],
        mobile_mask=np.asarray([not fixed_protein, True])[order],
        coordinate_map=coordinate_map,
    ).prepare()
    reference_source = _source(reference.tobytes(), "protein-reference")
    network = ElasticNetworkPlan(1.5, 4.0, 1).prepare(
        system, reference, reference_id=reference_source.manifest_id
    )
    parameter_record = parameter_data()
    payload = json.dumps(parameter_record, sort_keys=True).encode()
    artifact = NucleotideParameterArtifact(
        _source(payload, "nucleotide-parameters", export=nucleotide_export),
        payload,
        units,
    )
    construct = NucleicAcidConstruct(("dna",), ("AA",), ("DNA",), (False,))
    nucleotide = NucleotideModelPlan(
        construct,
        np.asarray([7, 11], dtype=np.int64),
        np.arange(7, 23, dtype=np.int64).reshape(2, 8),
        nucleotide_reference_sites(construct, artifact),
        np.asarray([4.0, 5.0]),
        np.tile(np.eye(3), (2, 1, 1)),
        artifact,
        fixed_mask=[fixed_body, False],
    ).prepare()
    kwargs = (
        {
            "steric_energy": 0.2,
            "steric_radius": 2.5,
            "linker_stiffness": 1.3,
            "linker_length": 1.0,
            "electrostatic_prefactor": -0.1,
            "screening": 0.2,
        }
        if cross_kwargs is None
        else cross_kwargs
    )
    cross_payload = json.dumps(kwargs, sort_keys=True).encode()
    cross = HybridCrossInteractionPlan(
        np.asarray([[90 if virtual else 11, 8]], dtype=np.int64),
        units,
        _source(cross_payload, "cross-parameters"),
        **kwargs,
    )
    model = PreparedHybridModel(network, nucleotide, cross, reference_source)
    spacing = parameter_record["profiles"]["DNA"]["backbone"][1]
    rigid = nucleotide.bodies.kinematics(
        jnp.asarray([[2.0, 1.0, 0.0], [2.0 + spacing, 1.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    positions = reference + np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])[order]
    return model, model.initialize(positions, np.zeros_like(positions), rigid)


def test_mixed_forces_obey_virtual_work_and_keep_fixed_reactions():
    model, state = _fixture(fixed_body=True, virtual=True)
    evaluation = jax.jit(lambda value: model.evaluate(value))(state)
    assert bool(evaluation.successful)
    assert np.linalg.norm(evaluation.protein_reaction_forces) > 0.0
    assert np.linalg.norm(evaluation.nucleotide_reaction_load.torque) > 0.0
    stepped = model.step(state, 1e-3)
    np.testing.assert_array_equal(
        stepped.state.nucleotide.position[0], state.nucleotide.position[0]
    )
    np.testing.assert_array_equal(
        stepped.state.nucleotide.orientation[0], state.nucleotide.orientation[0]
    )
    np.testing.assert_allclose(
        evaluation.protein_forces,
        evaluation.protein_mobile_forces + evaluation.protein_reaction_forces,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        evaluation.nucleotide_load.force,
        evaluation.nucleotide_mobile_load.force
        + evaluation.nucleotide_reaction_load.force,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        jnp.sum(evaluation.protein_forces, axis=0)
        + jnp.sum(evaluation.nucleotide_load.force, axis=0),
        0.0,
        atol=1e-12,
    )
    torque = jnp.sum(
        jnp.cross(state.protein.positions, evaluation.protein_forces), axis=0
    ) + jnp.sum(
        jnp.cross(state.nucleotide.position, evaluation.nucleotide_load.force)
        + evaluation.nucleotide_load.torque,
        axis=0,
    )
    np.testing.assert_allclose(torque, 0.0, atol=1e-12)
    gradient = jax.jit(jax.grad(lambda x: model.energy(x, state.nucleotide)))(
        state.protein.positions
    )
    np.testing.assert_allclose(gradient, -evaluation.protein_forces, atol=1e-12)
    dx = jnp.asarray([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
    dr = jnp.asarray([[0.1, 0.2, 0.3], [0.2, -0.1, 0.0]])
    rotation = jnp.asarray([[0.2, -0.3, 0.1], [0.1, 0.2, 0.3]])

    def displaced_energy(amount):
        rigid = eqx.tree_at(
            lambda x: (x.position, x.orientation),
            state.nucleotide,
            (
                state.nucleotide.position + amount * dr,
                _quaternion_retract(state.nucleotide.orientation, amount * rotation),
            ),
        )
        return model.energy(state.protein.positions + amount * dx, rigid)

    derivative = (displaced_energy(1e-5) - displaced_energy(-1e-5)) / 2e-5
    work = (
        jnp.sum(evaluation.protein_forces * dx)
        + jnp.sum(evaluation.nucleotide_load.force * dr)
        + jnp.sum(evaluation.nucleotide_load.torque * rotation)
    )
    np.testing.assert_allclose(derivative, -work, rtol=1e-7, atol=1e-9)
    negated = eqx.tree_at(
        lambda x: x.nucleotide.orientation, state, -state.nucleotide.orientation
    )
    np.testing.assert_allclose(
        model.evaluate(negated).nucleotide_load.torque,
        evaluation.nucleotide_load.torque,
        atol=1e-13,
    )


def test_disjoint_identity_and_site_binding_survive_protein_order():
    model, state = _fixture()
    reordered, reordered_state = _fixture(order=(1, 0))
    identities = [record[2] for record in model.support_map.records]
    assert len(set(identities)) == len(identities)
    assert model.support_map.records == reordered.support_map.records
    for kind, source, target in model.support_map.records:
        assert model.support_map.source(model.support_map.global_id(kind, source)) == (
            kind,
            source,
        )
    original = model.evaluate(state)
    other = reordered.evaluate(reordered_state)
    np.testing.assert_allclose(original.energy, other.energy, atol=1e-13)
    np.testing.assert_allclose(
        original.protein_forces, other.protein_forces[::-1], atol=1e-13
    )
    np.testing.assert_allclose(
        original.nucleotide_load.torque, other.nucleotide_load.torque, atol=1e-13
    )
    bad_cross = HybridCrossInteractionPlan(
        [[999, 8]], model.cross.units, model.cross.parameter_source, linker_stiffness=1.0
    )
    with pytest.raises(ValueError, match="active stable sites"):
        PreparedHybridModel(
            model.protein_network,
            model.nucleotide_model,
            bad_cross,
            model.protein_reference,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        HybridCrossInteractionPlan(
            [[11, 8], [11, 8]], model.cross.units, model.cross.parameter_source
        )
    frame_id = int(model.nucleotide_model.marker_map.markers.plan.marker_ids[5])
    frame_cross = HybridCrossInteractionPlan(
        [[11, frame_id]],
        model.cross.units,
        model.cross.parameter_source,
        linker_stiffness=1.0,
    )
    with pytest.raises(ValueError, match="differential frame"):
        PreparedHybridModel(
            model.protein_network,
            model.nucleotide_model,
            frame_cross,
            model.protein_reference,
        )


def test_linker_force_and_reference_linear_response():
    model, state = _fixture(cross_kwargs={"linker_stiffness": 2.0, "linker_length": 1.0})
    evaluation = model.evaluate(state)
    site = model.nucleotide_model.site_positions(state.nucleotide)[1]
    delta = state.protein.positions[1] - site
    distance = jnp.sqrt(jnp.sum(delta**2))
    expected = -2.0 * (distance - 1.0) * delta / distance
    native_protein = model.protein_network.evaluate(state.protein.positions)
    np.testing.assert_allclose(
        evaluation.protein_forces[1] - native_protein.forces[1], expected, atol=1e-12
    )
    np.testing.assert_allclose(
        evaluation.components[3], (distance - 1.0) ** 2, atol=1e-12
    )
    offset = site - state.nucleotide.position[0]
    native_torque = model.nucleotide_model.evaluate(state.nucleotide).loads.load.torque[0]
    np.testing.assert_allclose(
        evaluation.nucleotide_load.torque[0] - native_torque,
        jnp.cross(offset, -expected),
        atol=1e-12,
    )
    response = jax.grad(
        lambda shift: model.protein_network.evaluate(
            state.protein.positions.at[1, 0].add(shift)
        ).forces[1, 0]
    )(0.0)
    np.testing.assert_allclose(response, -4.0, atol=1e-12)


@pytest.mark.parametrize("physical_units", [False, True])
def test_split_drift_uses_shared_old_force_and_correct_units(physical_units):
    units = (
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
        if physical_units
        else AtomisticUnitSystem.reduced()
    )
    model, state = _fixture(units=units)
    old = model.evaluate(state)
    dt = 1e-3
    result = jax.jit(lambda value, step: model.step(value, step))(state, dt)
    assert bool(result.successful)
    scale = units.force_to_momentum_rate
    system = model.protein_network.system
    bodies = model.nucleotide_model.bodies
    expected_protein = (
        state.protein.positions
        + 0.5 * dt**2 * scale * old.protein_mobile_forces * system.inverse_masses[:, None]
    )
    expected_rigid = (
        state.nucleotide.position
        + 0.5
        * dt**2
        * scale
        * old.nucleotide_mobile_load.force
        * bodies.inverse_masses[:, None]
    )
    np.testing.assert_allclose(
        result.state.protein.positions, expected_protein, atol=1e-13
    )
    np.testing.assert_allclose(
        result.state.nucleotide.position, expected_rigid, atol=1e-13
    )
    expected_orientation = _quaternion_retract(
        state.nucleotide.orientation,
        0.5 * dt**2 * scale * old.nucleotide_mobile_load.torque,
    )
    np.testing.assert_allclose(
        result.state.nucleotide.orientation, expected_orientation, atol=1e-13
    )
    np.testing.assert_array_equal(result.state.protein.momenta[0], 0.0)
    assert not bool(model.step(state, 0.0).successful)


def test_kdk_energy_error_decreases_with_step_size():
    model, initial = _fixture(fixed_protein=False)
    initial_energy = model.evaluate(initial).energy + model.kinetic_energy(initial)

    def evolve(dt, steps):
        def body(state, _):
            result = model.step(state, dt)
            return result.state, (result.total_energy, result.successful)

        return jax.jit(lambda state: jax.lax.scan(body, state, None, length=steps))(
            initial
        )

    _, (coarse, coarse_success) = evolve(0.02, 10)
    _, (fine, fine_success) = evolve(0.01, 20)
    assert bool(jnp.all(coarse_success) & jnp.all(fine_success))
    coarse_error = jnp.max(jnp.abs(coarse - initial_energy))
    fine_error = jnp.max(jnp.abs(fine - initial_energy))
    assert float(fine_error) < 0.4 * float(coarse_error)


def test_padding_never_gains_material_or_cross_interactions():
    model, initial = _fixture()
    units = model.cross.units
    system = AtomisticSystemPlan(
        [7, 11, 99],
        [0, 0, 0],
        [2.0, 3.0, 1.0],
        units,
        element_mask=[False] * 3,
        active_mask=[True, True, False],
        mobile_mask=[False, True, False],
    ).prepare()
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    reference_source = _source(reference.tobytes(), "padded-reference")
    network = ElasticNetworkPlan(1.5, 4.0, 1).prepare(
        system, reference, reference_id=reference_source.manifest_id
    )
    padded = PreparedHybridModel(
        network, model.nucleotide_model, model.cross, reference_source
    )
    positions = np.concatenate((np.asarray(initial.protein.positions), reference[2:]))
    state = padded.initialize(positions, np.zeros_like(positions), initial.nucleotide)
    result = padded.evaluate(state)
    assert bool(result.successful)
    np.testing.assert_allclose(result.energy, model.evaluate(initial).energy, atol=1e-13)
    np.testing.assert_array_equal(result.protein_forces[2], 0.0)
    bad_cross = HybridCrossInteractionPlan(
        [[99, 8]], units, model.cross.parameter_source, linker_stiffness=1.0
    )
    with pytest.raises(ValueError, match="active stable sites"):
        PreparedHybridModel(network, model.nucleotide_model, bad_cross, reference_source)


def test_incompatible_scales_reference_rights_and_singular_sites_refuse():
    model, state = _fixture()
    different = HybridCrossInteractionPlan(
        [[11, 8]],
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        model.cross.parameter_source,
    )
    with pytest.raises(ValueError, match="same exact unit"):
        PreparedHybridModel(
            model.protein_network,
            model.nucleotide_model,
            different,
            model.protein_reference,
        )
    restricted = _source(b"restricted-parameters", "restricted", export=False)
    cross = HybridCrossInteractionPlan([[11, 8]], model.cross.units, restricted)
    with pytest.raises(PermissionError):
        PreparedHybridModel(
            model.protein_network,
            model.nucleotide_model,
            cross,
            model.protein_reference,
            export=True,
        )
    inherited, _ = _fixture(nucleotide_export=False)
    with pytest.raises(PermissionError):
        PreparedHybridModel(
            inherited.protein_network,
            inherited.nucleotide_model,
            inherited.cross,
            inherited.protein_reference,
            export=True,
        )
    coincident = eqx.tree_at(
        lambda value: value.protein.positions,
        state,
        state.protein.positions.at[1].set(
            model.nucleotide_model.site_positions(state.nucleotide)[1]
        ),
    )
    assert not bool(model.evaluate(coincident).successful)
    assert not bool(model.step(coincident, 1e-3).successful)
    assert not bool(
        jnp.isfinite(model.energy(coincident.protein.positions, coincident.nucleotide))
    )

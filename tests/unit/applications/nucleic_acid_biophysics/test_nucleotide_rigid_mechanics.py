# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independently authored fixtures qualify equations, not physical calibration."""

import hashlib
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.nucleic_rigid import parameter_artifact, parameter_data
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.coarse import (
    nucleotide_reference_sites,
    NucleotideModelPlan,
    NucleotideParameterArtifact,
)
from phydrax.applications.nucleic_acid_biophysics.coarse._published import (
    angular_value,
    excluded_value,
    helicity_value,
    radial_support,
    radial_value,
    screened_value,
)
from phydrax.discretization._lagrangian_marker import LagrangianMarkerSetPlan
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._rigid_body import (
    _quaternion_retract,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodySetPlan,
)
from phydrax.discretization.particle._rigid_marker import RigidMarkerMapPlan
from phydrax.discretization.particle._rigid_thermal import (
    PreparedRigidHeatBath,
    rigid_periodic_presentation,
)
from phydrax.qualification._reference import ReferenceArtifactManifest


def _artifact_data(data):
    payload = json.dumps(data, sort_keys=True).encode()
    manifest = ReferenceArtifactManifest(
        "independent-equation-regression",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"reduced-energy": 1.0},
        uncertainty=None,
        lineage_ids=("independent-equation-fixture",),
    )
    return NucleotideParameterArtifact(manifest, payload, parameter_artifact().units)


def _model(
    *, fixed=False, periodic=False, backbone=False, family="average-dna", data=None
):
    if backbone:
        construct = NucleicAcidConstruct(("s",), ("AT",), ("DNA",), (False,))
    elif family == "rna":
        construct = NucleicAcidConstruct(
            ("a", "b"), ("A", "U"), ("RNA", "RNA"), (False, False)
        )
    elif family == "dna-rna-hybrid":
        construct = NucleicAcidConstruct(
            ("a", "b"), ("A", "U"), ("DNA", "RNA"), (False, False)
        )
    else:
        construct = NucleicAcidConstruct(
            ("a", "b"), ("A", "T"), ("DNA", "DNA"), (False, False)
        )
    artifact = parameter_artifact(family) if data is None else _artifact_data(data)
    reference = nucleotide_reference_sites(construct, artifact)
    model = NucleotideModelPlan(
        construct,
        np.array([101, 909]),
        13 + np.arange(16).reshape(2, 8) * 7,
        reference,
        np.array([2.0, 3.0]),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        artifact,
        fixed_mask=np.array([False, fixed]),
        cell=PeriodicCell(np.eye(3) * 10) if periodic else None,
    ).prepare()
    state = model.bodies.kinematics(
        jnp.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.1]]),
        jnp.zeros((2, 3)),
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, np.cos(0.2), np.sin(0.2)]]),
        jnp.zeros((2, 3)),
    )
    return model, state


def test_energy_gradient_wrench_virtual_work_and_reaction_balance():
    model, state = _model(fixed=True)
    evaluation = eqx.filter_jit(model.evaluate)(state)
    assert evaluation.successful
    load = evaluation.loads.load
    np.testing.assert_allclose(jnp.sum(load.force, axis=0), 0.0, atol=2e-12)
    np.testing.assert_allclose(
        jnp.sum(load.torque + jnp.cross(state.position, load.force), axis=0),
        0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(evaluation.loads.mobile_load.force[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        evaluation.loads.reaction_load.force[1], load.force[1], atol=1e-14
    )
    assert np.linalg.norm(load.torque) > 1e-3
    translation = jnp.array([[0.2, -0.1, 0.3], [-0.1, 0.2, 0.1]])
    rotation = jnp.array([[0.1, 0.3, -0.2], [-0.3, 0.1, 0.2]])

    def energy_at(t):
        moved = RigidBodyKinematics(
            state.position + t * translation,
            state.velocity,
            _quaternion_retract(state.orientation, t * rotation),
            state.angular_velocity,
        )
        return model.energy(moved)

    numerical = (energy_at(1e-5) - energy_at(-1e-5)) / 2e-5
    work = -jnp.sum(translation * load.force + rotation * load.torque)
    np.testing.assert_allclose(numerical, work, rtol=2e-6, atol=2e-8)
    result = model.step(state, 0.0, 0.001)
    np.testing.assert_allclose(result.kinematics.position[1], state.position[1])
    np.testing.assert_allclose(result.kinematics.orientation[1], state.orientation[1])
    assert np.linalg.norm(result.kinematics.angular_velocity[0]) > 0


def test_quaternion_sign_and_proper_rotation_covariance():
    model, state = _model()
    reference = model.evaluate(state)
    negative = RigidBodyKinematics(
        state.position, state.velocity, -state.orientation, state.angular_velocity
    )
    np.testing.assert_allclose(
        model.evaluate(negative).loads.load.torque,
        reference.loads.load.torque,
        atol=1e-12,
    )
    vector = jnp.broadcast_to(jnp.array([0.3, -0.4, 0.1]), (2, 3))
    global_q = _quaternion_retract(jnp.array([[1.0, 0.0, 0.0, 0.0]]), vector[:1])
    matrix = quaternion_rotation_matrix(global_q)[0]
    rotated = RigidBodyKinematics(
        state.position @ matrix.T + jnp.array([3.0, -2.0, 1.0]),
        state.velocity @ matrix.T,
        _quaternion_retract(state.orientation, vector),
        state.angular_velocity @ matrix.T,
    )
    transformed = model.evaluate(rotated)
    np.testing.assert_allclose(transformed.energy, reference.energy, atol=2e-12)
    np.testing.assert_allclose(
        transformed.loads.load.force, reference.loads.load.force @ matrix.T, atol=2e-11
    )
    np.testing.assert_allclose(
        transformed.loads.load.torque, reference.loads.load.torque @ matrix.T, atol=2e-11
    )


def test_point_force_binding_is_order_invariant_not_quadrature_weighted():
    model, state = _model(fixed=True)
    marker = model.marker_map.markers
    weighted = LagrangianMarkerSetPlan(
        marker.plan.marker_ids, marker.reference_position, jnp.arange(1.0, 17.0)
    ).prepare()
    mapping = RigidMarkerMapPlan(
        weighted, model.bodies, model.marker_map.marker_owner
    ).prepare()
    forces = jnp.arange(48.0, dtype=jnp.float64).reshape(16, 3) / 10
    order = np.arange(16)[::-1]
    ids = np.asarray(marker.plan.marker_ids)[order]
    owners = np.asarray(model.bodies.particles.particle_ids)[
        np.asarray(mapping.marker_owner)[order]
    ]
    bound = mapping.bind_site_forces(ids, owners)
    actual = eqx.filter_jit(bound.evaluate)(state, forces[order])
    expected = model.marker_map.site_force_load(state, forces)
    np.testing.assert_allclose(actual.load.force, expected.load.force, atol=1e-12)
    np.testing.assert_allclose(actual.load.torque, expected.load.torque, atol=1e-12)
    with pytest.raises(ValueError):
        mapping.bind_site_forces(ids, owners + 1)
    with pytest.raises(ValueError):
        mapping.bind_site_forces(ids[:-1], owners[:-1])


def test_all_fixed_marker_map_retains_full_reactions():
    model, state = _model()
    fixed = RigidBodySetPlan(
        np.zeros(2, dtype=int),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        fixed_mask=np.ones(2, dtype=bool),
    ).prepare(model.bodies.particles)
    mapping = RigidMarkerMapPlan(
        model.marker_map.markers, fixed, model.marker_map.marker_owner
    ).prepare()
    result = eqx.filter_jit(mapping.site_force_load)(state, jnp.ones((16, 3)))
    np.testing.assert_allclose(result.load.force, 8.0)
    np.testing.assert_allclose(result.mobile_load.force, 0.0)
    np.testing.assert_allclose(result.reaction_load.force, result.load.force)


def test_energy_drift_decreases_with_kdk_timestep():
    model, state = _model()
    initial = model.energy(state) + model.kinetic_energy(state)

    def rollout(dt, count):
        def body(q, i):
            step = model.step(q, i * dt, dt)
            return step.kinematics, model.energy(step.kinematics) + model.kinetic_energy(
                step.kinematics
            )

        return jax.lax.scan(body, state, jnp.arange(count))[1]

    coarse = eqx.filter_jit(lambda: rollout(0.01, 40))()
    fine = eqx.filter_jit(lambda: rollout(0.005, 80))()
    coarse_error = np.max(np.abs(np.asarray(coarse - initial)))
    fine_error = np.max(np.abs(np.asarray(fine - initial)))
    assert fine_error < 0.4 * coarse_error
    assert fine_error < 1e-3


def test_periodic_com_face_crossing_preserves_sites_and_pair_energy():
    model, state = _model(periodic=True)
    baseline = model.evaluate(state)
    shifted = RigidBodyKinematics(
        state.position.at[1, 0].add(10.0),
        state.velocity,
        state.orientation,
        state.angular_velocity,
    )
    np.testing.assert_allclose(
        model.evaluate(shifted).loads.load.force, baseline.loads.load.force, atol=2e-11
    )
    np.testing.assert_allclose(model.energy(shifted), baseline.energy, atol=1e-12)
    moving = RigidBodyKinematics(
        state.position + jnp.array([9.9, 0.0, 0.0]),
        jnp.ones((2, 3)) * jnp.array([2.0, 0.0, 0.0]),
        state.orientation,
        state.angular_velocity,
    )
    stepped = model.step(moving, 0.0, 0.1).kinematics
    presentation = rigid_periodic_presentation(model.cell, stepped)
    translation = presentation.images @ model.cell.vectors
    np.testing.assert_allclose(
        presentation.position + translation, stepped.position, atol=1e-12
    )
    assert int(presentation.images[0, 0]) == 1
    before = model.site_positions(stepped).reshape(2, 8, 3)
    wrapped = before - translation[:, None, :]
    np.testing.assert_allclose(
        wrapped[:, 1] - wrapped[:, 0], before[:, 1] - before[:, 0], atol=1e-12
    )


def test_fene_domain_and_published_geometry_are_not_silently_repaired():
    model, state = _model(backbone=True)
    assert model.evaluate(state).successful
    invalid = RigidBodyKinematics(
        state.position.at[1, 0].set(3.0),
        state.velocity,
        state.orientation,
        state.angular_velocity,
    )
    assert not model.evaluate(invalid).successful
    artifact = parameter_artifact()
    construct = NucleicAcidConstruct(("s",), ("AT",), ("DNA",), (False,))
    geometry = nucleotide_reference_sites(construct, artifact)
    geometry[0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="geometry"):
        NucleotideModelPlan(
            construct,
            np.array([101, 909]),
            np.arange(16).reshape(2, 8),
            geometry,
            np.ones(2),
            np.broadcast_to(np.eye(3), (2, 3, 3)),
            artifact,
        ).prepare()


def test_anisotropic_heat_bath_has_fluctuation_dissipation_covariance():
    particles = ParticleSetPlan(
        np.array([10, 20]), np.array([2.0, 3.0]), ambient_dimension=3
    ).prepare()
    inertia = np.array([np.diag([1.0, 2.0, 4.0]), np.diag([2.0, 3.0, 5.0])])
    bodies = RigidBodySetPlan(
        np.zeros(2, dtype=int), inertia, fixed_mask=np.array([False, True])
    ).prepare(particles)
    state = bodies.kinematics(
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        jnp.array([[np.cos(0.3), 0.0, np.sin(0.3), 0.0], [1.0, 0.0, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
    )
    bath = PreparedRigidHeatBath(bodies, 1.7, 2.0, 3.0)
    samples = eqx.filter_jit(jax.vmap(lambda key: bath.apply(state, 0.4, key)))(
        jax.random.split(jax.random.key(5), 12000)
    )
    np.testing.assert_allclose(
        np.var(samples.velocity[:, 0], axis=0),
        1.7 / 2 * (1 - np.exp(-2 * 2 * 0.4)),
        rtol=0.05,
    )
    matrix = np.asarray(quaternion_rotation_matrix(state.orientation)[0])
    expected = (
        1.7 * (1 - np.exp(-2 * 3 * 0.4)) * matrix @ np.diag([1.0, 0.5, 0.25]) @ matrix.T
    )
    np.testing.assert_allclose(
        np.cov(np.asarray(samples.angular_velocity[:, 0]).T),
        expected,
        rtol=0.09,
        atol=0.03,
    )
    np.testing.assert_allclose(samples.velocity[:, 1], 0.0)
    np.testing.assert_allclose(samples.angular_velocity[:, 1], 0.0)
    np.testing.assert_allclose(
        bath.apply(state, 0.0, jax.random.key(6)).velocity, state.velocity
    )


def test_piecewise_published_windows_match_value_and_force_at_joins():
    for kind, p in (
        ("morse", [1.0, 1.0, 2.0, 0.7, 1.7, 2.0]),
        ("harmonic", [1.0, 1.0, 1.8, 0.25, 1.6, 1.0]),
    ):
        low, high = radial_support(p, kind)
        fn = lambda r: radial_value(r, jnp.array(p), kind)
        for join in (low, p[3], p[4], high):
            np.testing.assert_allclose(fn(join - 1e-7), fn(join + 1e-7), atol=3e-6)
            np.testing.assert_allclose(
                jax.grad(fn)(join - 1e-7), jax.grad(fn)(join + 1e-7), atol=5e-5
            )
            # At support edges E is C1, not C2: centered derivative error is O(h).
            np.testing.assert_allclose(
                jax.grad(fn)(join), (fn(join + 1e-8) - fn(join - 1e-8)) / 2e-8, atol=3e-6
            )
        assert float(fn(high + 0.1)) == 0.0
    window = lambda theta: angular_value(theta, jnp.array([1.3, 0.4, 0.6]))
    for join in (1.0, 0.4 + 1 / (1.3 * 0.6)):
        np.testing.assert_allclose(
            jax.grad(window)(join - 1e-7), jax.grad(window)(join + 1e-7), atol=5e-6
        )
    h = lambda x: helicity_value(x, jnp.array([1.0, -0.5]))
    np.testing.assert_allclose(jax.grad(h)(-1e-7), jax.grad(h)(1e-7), atol=1e-6)
    assert float(h(0.3)) == 1.0 and float(h(-3.0)) == 0.0
    ev = lambda r: excluded_value(r, jnp.array([0.1, 0.3, 0.28]))
    np.testing.assert_allclose(
        jax.grad(ev)(0.28 - 1e-9), jax.grad(ev)(0.28 + 1e-9), atol=2e-5
    )
    np.testing.assert_allclose(
        jax.grad(ev)(0.28), (ev(0.28 + 1e-7) - ev(0.28 - 1e-7)) / 2e-7, atol=1e-3
    )
    dh = lambda r: screened_value(r, jnp.array([0.1, 0.2]))
    np.testing.assert_allclose(
        jax.grad(dh)(0.6 - 1e-8), jax.grad(dh)(0.6 + 1e-8), atol=2e-7
    )
    assert float(dh(0.91)) == 0.0


def test_model_variants_execute_distinct_geometry_conditions_and_hybrid_strengths():
    dna, state = _model()
    dna2, q2 = _model(family="groove-salt-dna")
    rna, qr = _model(family="rna")
    hybrid, qh = _model(family="dna-rna-hybrid")
    for model, q in ((dna, state), (dna2, q2), (rna, qr), (hybrid, qh)):
        assert eqx.filter_jit(model.evaluate)(q).successful
    assert not np.allclose(dna.site_positions(state), dna2.site_positions(q2))
    assert not np.allclose(dna.site_positions(state), rna.site_positions(qr))
    data = parameter_data("groove-salt-dna")
    data["salt_concentration"] = 0.04
    low_salt, ql = _model(family="groove-salt-dna", data=data)
    assert float(low_salt.energy(ql)) > float(dna2.energy(q2))
    data = parameter_data("dna-rna-hybrid")
    data["sequence_strengths"]["HYBRID"]["hydrogen-bond"][0][3] *= 2
    stronger, qs = _model(family="dna-rna-hybrid", data=data)
    assert float(stronger.energy(qs)) < float(hybrid.energy(qh))
    data["profiles"].pop("HYBRID")
    with pytest.raises(ValueError):
        _artifact_data(data)


def test_parameter_payload_tampering_and_false_family_geometry_are_refused():
    artifact = parameter_artifact()
    with pytest.raises(ValueError, match="source manifest"):
        NucleotideParameterArtifact(
            artifact.manifest,
            artifact.raw_payload.replace(b"1.0", b"2.0", 1),
            artifact.units,
        )
    data = parameter_data()
    data["geometry"]["DNA"]["backbone"][1] = 0.1
    with pytest.raises(ValueError, match="collinear"):
        _artifact_data(data)
    data = parameter_data("rna")
    data["geometry"]["RNA"]["stack5"] = data["geometry"]["RNA"]["stack3"]
    with pytest.raises(ValueError, match="distinct"):
        _artifact_data(data)


def test_directed_sequence_stacking_keeps_five_to_three_table_order():
    artifact = parameter_artifact("sequence-dna")
    energies = []
    for sequence in ("AT", "TA"):
        construct = NucleicAcidConstruct(("s",), (sequence,), ("DNA",), (False,))
        model = NucleotideModelPlan(
            construct,
            np.array([41, 13]),
            np.arange(16).reshape(2, 8),
            nucleotide_reference_sites(construct, artifact),
            np.ones(2),
            np.broadcast_to(np.eye(3), (2, 3, 3)),
            artifact,
        ).prepare()
        state = model.bodies.kinematics(
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.5]]),
            jnp.zeros((2, 3)),
            jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            jnp.zeros((2, 3)),
        )
        energies.append(eqx.filter_jit(model.energy)(state))
    expected = (
        0.5
        * 1.1
        * radial_value(
            jnp.asarray(1.5), jnp.array([0.2, 1.0, 2.0, 0.7, 1.7, 2.0]), "morse"
        )
    )
    np.testing.assert_allclose(energies[0] - energies[1], expected, atol=1e-12)


def test_inactive_marker_force_nan_is_not_material_and_never_enters_loads():
    particles = ParticleSetPlan(
        np.array([41, 13, 7]),
        np.ones(3),
        active_mask=np.array([True, True, False]),
        ambient_dimension=3,
    ).prepare()
    bodies = RigidBodySetPlan(
        np.zeros(3, dtype=int),
        np.broadcast_to(np.eye(3), (3, 3, 3)),
        fixed_mask=np.array([False, True, False]),
    ).prepare(particles)
    markers = LagrangianMarkerSetPlan(
        np.array([101, 103, 105, 107]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        np.ones(4),
        active_mask=np.array([True, True, True, False]),
    ).prepare()
    mapping = RigidMarkerMapPlan(markers, bodies, np.array([0, 1, 1, 0])).prepare()
    state = bodies.kinematics(
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3)),
        jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (3, 1)),
        jnp.zeros((3, 3)),
    )
    result = eqx.filter_jit(mapping.site_force_load)(
        state,
        jnp.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [2.0, 3.0, 1.0],
                [jnp.nan, jnp.nan, jnp.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        result.load.force, jnp.array([[1.0, 2.0, 3.0], [5.0, 5.0, 2.0], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_allclose(result.reaction_load.force[1], jnp.array([5.0, 5.0, 2.0]))
    np.testing.assert_allclose(result.mobile_load.force[1:], 0.0)


def test_coincident_inactive_coaxial_sites_preserve_finite_repulsive_force():
    model, state = _model()
    state = RigidBodyKinematics(
        jnp.array([[0.0, 0.0, 0.0], [0.16, 0.0, 0.0]]),
        state.velocity,
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        state.angular_velocity,
    )
    result = eqx.filter_jit(model.evaluate)(state)
    assert result.successful
    displacement = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def energy_shift(amount):
        return model.energy(
            RigidBodyKinematics(
                state.position + amount * displacement,
                state.velocity,
                state.orientation,
                state.angular_velocity,
            )
        )

    derivative = (energy_shift(1e-8) - energy_shift(-1e-8)) / 2e-8
    np.testing.assert_allclose(-result.loads.load.force[1, 0], derivative, rtol=1e-7)
    np.testing.assert_allclose(jnp.sum(result.loads.load.force, axis=0), 0.0, atol=1e-4)

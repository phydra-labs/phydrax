#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


s = phx.solver
q = phx.operators.quantum


def _two_mode_device(*, edge_mask=None):
    topology = phx.graph.GraphIR(
        n_node=jnp.asarray([2]),
        n_edge=jnp.asarray([1]),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([1]),
        edge_mask=edge_mask,
    )
    basis = q.OscillatorBasis(5)
    reduction = q.ModeReductionPolicy(2)
    placements = (
        s.CircuitModePlacement("a", "harmonic", basis, 0, reduction),
        s.CircuitModePlacement("b", "harmonic", basis, 0, reduction),
    )
    interaction = s.CircuitInteraction((0, 1), ("phase", "phase"), 0)
    port = s.CircuitDrivePort(0, "charge", 0, port_id="drive-a")
    spec = s.CircuitQEDDeviceSpec(
        topology,
        placements,
        (interaction,),
        drive_ports=(port,),
    )
    parameters = s.CircuitQEDDeviceParameters(
        (q.HarmonicModeParameters(3.0),),
        interaction_strengths=jnp.asarray([0.2]),
        drive_scales=jnp.asarray([0.7]),
    )
    return spec, parameters


def test_device_compiler_reuses_shared_parameters_and_matches_dense_algebra():
    spec, parameters = _two_mode_device()
    prepared = s.prepare_circuit_qed_device(spec, parameters)
    dense = s.materialize_local_hamiltonian(prepared.drift)
    local_energy = jnp.diag(jnp.asarray([0.0, 3.0], dtype=jnp.complex128))
    phase = prepared.reductions[0].operator("phase").matrix
    expected = (
        jnp.kron(local_energy, jnp.eye(2))
        + jnp.kron(jnp.eye(2), local_energy)
        + 0.2 * jnp.kron(phase, phase)
    )

    assert bool(prepared.diagnostics.valid)
    assert prepared.plan.layout.local_dimensions == (2, 2)
    assert prepared.plan.required_mode_parameters == 1
    assert len(prepared.drift.terms) == 3
    assert len(prepared.drive_terms) == 1
    assert jnp.allclose(dense, expected)


def test_device_refresh_preserves_plan_and_differentiates_shared_mode_block():
    spec, parameters = _two_mode_device()
    prepared = s.prepare_circuit_qed_device(spec, parameters)

    def objective(rate):
        refreshed = s.refresh_circuit_qed_device(
            prepared,
            s.CircuitQEDDeviceParameters(
                (q.HarmonicModeParameters(rate),),
                interaction_strengths=jnp.asarray([0.2]),
                drive_scales=jnp.asarray([0.7]),
            ),
        )
        dense = s.materialize_local_hamiltonian(refreshed.drift)
        return jnp.real(jnp.trace(dense))

    derivative = jax.grad(objective)(jnp.asarray(3.0))
    refreshed = s.refresh_circuit_qed_device(
        prepared,
        s.CircuitQEDDeviceParameters(
            (q.HarmonicModeParameters(3.2),),
            interaction_strengths=jnp.asarray([0.2]),
            drive_scales=jnp.asarray([0.7]),
        ),
    )

    assert jnp.isfinite(derivative)
    assert derivative > 0.0
    assert int(refreshed.numeric_version) == 1
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert jnp.allclose(
        refreshed.reductions[0].energies, refreshed.reductions[1].energies
    )


def test_device_compiler_respects_inactive_edges_and_dense_resource_limits():
    spec, parameters = _two_mode_device(edge_mask=jnp.asarray([False]))
    plan = s.plan_circuit_qed_device(
        spec,
        s.CircuitQEDDevicePolicy(maximum_dense_entries=1),
    )
    prepared = s.prepare_circuit_qed_device(spec, parameters, plan)

    assert not plan.cost.dense_admissible
    assert len(prepared.drift.terms) == 2
    assert bool(prepared.diagnostics.valid)
    with pytest.raises(ValueError, match="maximum_hilbert_dimension"):
        s.plan_circuit_qed_device(
            spec,
            s.CircuitQEDDevicePolicy(maximum_hilbert_dimension=2),
        )


def test_device_compiler_rejects_topology_parameter_and_operator_mismatches():
    spec, _ = _two_mode_device()
    plan = s.plan_circuit_qed_device(spec)
    with pytest.raises(ValueError, match="mode_parameters count"):
        s.prepare_circuit_qed_device(
            spec,
            s.CircuitQEDDeviceParameters(
                (q.HarmonicModeParameters(3.0), q.HarmonicModeParameters(4.0)),
                interaction_strengths=jnp.asarray([0.2]),
                drive_scales=jnp.asarray([0.7]),
            ),
            plan,
        )

    bad_placement = s.CircuitModePlacement(
        "q",
        "transmon",
        q.ChargeBasis(3),
        0,
        q.ModeReductionPolicy(2),
    )
    bad_spec = s.CircuitQEDDeviceSpec(
        phx.graph.GraphIR(n_node=jnp.asarray([1]), n_edge=jnp.asarray([0])),
        (bad_placement,),
        (),
    )
    with pytest.raises(TypeError, match="do not match"):
        s.prepare_circuit_qed_device(
            bad_spec,
            s.CircuitQEDDeviceParameters((q.HarmonicModeParameters(3.0),)),
        )

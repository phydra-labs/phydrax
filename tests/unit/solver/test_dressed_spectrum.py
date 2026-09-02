#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


s = phx.solver
q = phx.operators.quantum


def _device(*, shared=False, coupling=0.0):
    topology = phx.graph.GraphIR(
        n_node=jnp.asarray([2]),
        n_edge=jnp.asarray([1]),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([1]),
    )
    basis = q.OscillatorBasis(4)
    policy = q.ModeReductionPolicy(2)
    placements = (
        s.CircuitModePlacement("a", "harmonic", basis, 0, policy),
        s.CircuitModePlacement("b", "harmonic", basis, 0 if shared else 1, policy),
    )
    interaction = s.CircuitInteraction((0, 1), ("phase", "phase"), 0)
    spec = s.CircuitQEDDeviceSpec(topology, placements, (interaction,))
    parameters = s.CircuitQEDDeviceParameters(
        (
            (q.HarmonicModeParameters(3.0),)
            if shared
            else (q.HarmonicModeParameters(3.0), q.HarmonicModeParameters(4.0))
        ),
        interaction_strengths=jnp.asarray([coupling]),
    )
    return s.prepare_circuit_qed_device(spec, parameters), spec


def _labels():
    return tuple(
        s.DressedStateLabel(levels) for levels in ((0, 0), (0, 1), (1, 0), (1, 1))
    )


def test_dressed_spectrum_maps_product_labels_one_to_one_and_static_zz_is_zero():
    device, _ = _device(coupling=0.0)
    dressed = s.prepare_dressed_spectrum(device, labels=_labels())
    zeta = (
        dressed.energy((1, 1))
        - dressed.energy((1, 0))
        - dressed.energy((0, 1))
        + dressed.energy((0, 0))
    )

    assert bool(dressed.diagnostics.valid)
    assert jnp.allclose(dressed.energies, jnp.asarray([0.0, 4.0, 3.0, 7.0]))
    assert jnp.allclose(zeta, 0.0, atol=1e-12)
    assert dressed.subspace.logical_dimension == 4
    assert len(set(map(int, dressed.plan.tracking_plan.clusters[0]))) == len(
        dressed.plan.tracking_plan.clusters[0]
    )


def test_dressed_spectrum_refresh_preserves_labels_and_builds_selected_subspace():
    device, spec = _device(coupling=0.0)
    dressed = s.prepare_dressed_spectrum(device, labels=_labels())
    refreshed_device = s.refresh_circuit_qed_device(
        device,
        s.CircuitQEDDeviceParameters(
            (q.HarmonicModeParameters(3.0), q.HarmonicModeParameters(4.0)),
            interaction_strengths=jnp.asarray([0.05]),
        ),
    )
    refreshed = s.refresh_dressed_spectrum(dressed, refreshed_device)
    logical = s.dressed_quantum_subspace(
        refreshed,
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )

    assert spec.spec_id == refreshed_device.plan.spec.spec_id
    assert bool(refreshed.diagnostics.valid)
    assert int(refreshed.numeric_version) == 1
    assert refreshed.plan.plan_id == dressed.plan.plan_id
    assert logical.logical_dimension == 4


def test_dressed_spectrum_reports_a_split_reference_degeneracy():
    device, _ = _device(shared=True, coupling=0.0)
    dressed = s.prepare_dressed_spectrum(device, labels=_labels())
    split_device = s.refresh_circuit_qed_device(
        device,
        s.CircuitQEDDeviceParameters(
            (q.HarmonicModeParameters(3.0),),
            interaction_strengths=jnp.asarray([0.2]),
        ),
    )
    split = s.refresh_dressed_spectrum(dressed, split_device)

    assert not bool(split.diagnostics.valid)


def test_dressed_spectrum_enforces_dense_resources_and_label_bounds():
    device, _ = _device()
    with pytest.raises(ValueError, match="dense resource"):
        s.plan_dressed_spectrum(
            device,
            _labels(),
            s.DressedSpectrumPolicy(maximum_dense_entries=1),
        )
    with pytest.raises(ValueError, match="out-of-range"):
        s.plan_dressed_spectrum(device, ((0, 2),))

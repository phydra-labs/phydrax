#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _rotation_map(angle):
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.asarray([[cosine, sine], [-sine, cosine]])


def test_ring_map_optics_tracking_and_wake_are_bounded():
    accelerator = phx.applications.accelerator
    convention = accelerator.AcceleratorConvention()
    matrix = jnp.zeros((6, 6))
    matrix = matrix.at[:2, :2].set(_rotation_map(0.2))
    matrix = matrix.at[2:4, 2:4].set(_rotation_map(0.3))
    matrix = matrix.at[4:6, 4:6].set(_rotation_map(0.1))
    transfer = accelerator.SymplecticMapPlan(
        matrix,
        jnp.zeros(6),
        convention,
        element_id="one-turn",
    )
    optics = accelerator.linear_ring_optics(transfer)
    assert bool(optics.stable)
    assert jnp.isclose(optics.horizontal_tune, 0.2 / (2.0 * jnp.pi))

    bunch = accelerator.AcceleratorBunch(
        jnp.asarray(
            [[1.0e-3, 0.0, 0.0, 0.0, -0.5, 0.0], [-1.0e-3, 0.0, 0.0, 0.0, 0.5, 0.0]]
        ),
        jnp.ones(2),
        jnp.asarray([1, 2]),
        reference_rest_energy=1.0,
        reference_momentum=2.0,
        reference_charge=1.0,
        convention=convention,
        bunch_id="ring-test",
    )
    tracked = accelerator.track_ring(
        accelerator.RingTrackingPlan(
            transfer,
            16,
            horizontal_aperture=0.01,
            vertical_aperture=0.01,
        ),
        bunch,
    )
    assert bool(tracked.finite)
    assert jnp.all(tracked.bunch.active)

    wake = accelerator.apply_longitudinal_wake(
        accelerator.LongitudinalWakePlan(
            jnp.asarray([-1.0, 0.0, 1.0]),
            jnp.asarray([0.1, 0.05]),
            kick_scale=0.01,
        ),
        bunch,
    )
    assert bool(wake.finite)
    assert jnp.any(wake.kicks != 0.0)


def test_qcd_transport_and_flow_observables_preserve_physics_contracts():
    qcd = phx.applications.lattice_field
    table = qcd.QCDTransportTable(
        jnp.asarray([0.15, 0.25]),
        jnp.asarray([0.0, 0.1]),
        jnp.asarray([[0.1, 0.11], [0.12, 0.13]]),
        jnp.asarray([[0.02, 0.03], [0.04, 0.05]]),
        jnp.asarray([[0.5, 0.45], [0.4, 0.35]]),
        source_kind=qcd.QCDTransportSourceKind.PHENOMENOLOGICAL,
        source_id="transport-test",
    )
    evaluated = qcd.evaluate_qcd_transport(table, 0.2, 0.05)
    assert bool(evaluated.valid)
    assert jnp.isclose(evaluated.shear_viscosity_over_entropy, 0.115)

    heavy_ion = phx.applications.heavy_ion
    flow = heavy_ion.compute_flow_observables(
        heavy_ion.FlowObservablePlan((2, 3)),
        jnp.asarray([[0.0, jnp.pi / 2.0, jnp.pi, 3.0 * jnp.pi / 2.0]]),
        jnp.ones((1, 4)),
        jnp.ones((1, 4), dtype="bool"),
    )
    assert bool(flow.valid[0])
    assert jnp.isclose(flow.flow_magnitudes[0, 2 - 2], 0.0, atol=1.0e-12)

    conservation = heavy_ion.audit_hydrodynamic_conservation(
        jnp.asarray([[10.0, 1.0, 0.0, 0.0]]),
        jnp.asarray([[10.0, 1.0, 0.0, 0.0]]),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        maximum_relative_residual=1.0e-12,
        source_id="hydro-test",
    )
    assert bool(conservation.valid[0])

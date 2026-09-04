#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._integration_guardrails import (
    CoreAbstractionRegistry,
    reject_external_runtime,
)


def _grid(shape=(3, 3, 3), *, periodic=False):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for count in shape
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.asarray([[0.0] * len(shape), [1.0] * len(shape)]))


def test_maxwell_db_state_boundaries_observers_and_cpml_are_composable():
    bridge = phx.discretization.StructuredCochainBridge(_grid())
    n0, n1, n2, _ = bridge.cochain.cell_counts
    constitutive = phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(
        permittivity=1.0 + 0.1 * jnp.arange(n1) / n1,
        permeability=1.0 + 0.1 * jnp.arange(n2) / n2,
    )
    probe = phx.solver.maxwell.FieldProbePlan("electric", jnp.asarray([0, 1, 2]))
    dft = phx.solver.maxwell.DFTObserverPlan(probe, jnp.asarray([1.0, 2.0]))
    pml = phx.solver.maxwell.MaxwellCPMLPlan(1)
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=constitutive,
        boundaries=(phx.solver.maxwell.MaxwellBoundaryPlan("pec"),),
        observers=(probe, dft),
        pml=pml,
    ).prepare()
    electric = jnp.sin(jnp.arange(n1, dtype=float) / 9.0)
    displacement = runtime.constitutive.electric_displacement(electric, None)
    magnetic = bridge.exterior_derivative(1, electric)
    unconstrained = runtime.pack(displacement, magnetic)
    charge = bridge.codifferential(
        1,
        unconstrained.primary.electric_displacement,
    )
    state = runtime.pack(
        unconstrained.primary.electric_displacement,
        unconstrained.primary.magnetic_flux,
        charge,
        boundary_state=unconstrained.auxiliary.boundary,
        observations=unconstrained.observations,
    )
    stepped = runtime.leapfrog_step(0.0, state, 0.05 * runtime.stable_dt)
    report = runtime.diagnostics(0.05 * runtime.stable_dt, stepped)

    assert state.primary.electric_displacement.shape == (n1,)
    assert state.primary.magnetic_flux.shape == (n2,)
    assert state.primary.charge.shape == (n0,)
    assert len(runtime.observe(stepped)) == 2
    assert jnp.isfinite(report.energy)
    assert report.electric_constraint_linf < 1e-8
    assert report.magnetic_constraint_linf < 1e-8
    assert report.pml_dissipation >= 0.0


def test_periodic_and_bloch_cochain_derivatives_preserve_chain_identity():
    bridge = phx.discretization.StructuredCochainBridge(_grid(periodic=True))
    values = jnp.sin(jnp.arange(bridge.cochain.cell_counts[0], dtype=float))
    np.testing.assert_allclose(
        bridge.exterior_derivative(1, bridge.exterior_derivative(0, values)),
        0.0,
        atol=1e-15,
        rtol=0.0,
    )
    bloch = phx.solver.maxwell.BlochCochainCalculus(bridge, jnp.asarray([0.2, -0.1, 0.3]))
    np.testing.assert_allclose(bloch.chain_residual(0, values), 0.0, atol=2e-12)
    np.testing.assert_allclose(jnp.abs(bloch.phases), 1.0, atol=2e-12)


def test_material_families_are_fail_closed_and_differentiable():
    bridge = phx.discretization.StructuredCochainBridge(_grid((2, 2, 2)))
    layout = phx.solver.maxwell.MaxwellCochainLayout(bridge)
    n1 = bridge.cochain.cell_counts[1]
    n2 = bridge.cochain.cell_counts[2]
    conductive = phx.solver.maxwell.ConductiveMaxwellConstitutivePlan(
        electric_conductivity=0.2,
        magnetic_conductivity=0.1,
    ).prepare(bridge.cochain, layout)
    electric = jnp.ones((n1,))
    magnetic = jnp.ones((n2,))
    assert (
        conductive.dissipated_power(
            electric,
            magnetic,
            None,
            bridge.cochain.hodge_stars[1],
            bridge.cochain.hodge_stars[2],
        )
        > 0.0
    )

    dispersive = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        jnp.asarray([1.0]), jnp.asarray([0.1]), jnp.asarray([0.5])
    ).prepare(bridge.cochain, layout)
    material_state = dispersive.initialize_state()
    updated = dispersive.advance_state(
        jnp.asarray(0.0),
        material_state,
        jnp.ones((n1,)),
        jnp.zeros((n2,)),
        jnp.asarray(1e-3),
        None,
    )
    assert jnp.linalg.norm(updated.velocity) > 0.0

    nonlinear = phx.solver.maxwell.KerrPockelsMaxwellConstitutivePlan(
        pockels=0.01,
        kerr=0.02,
        field_bound=2.0,
    ).prepare(bridge.cochain, layout)
    field = jnp.linspace(-0.3, 0.3, n1)
    displacement = nonlinear.electric_displacement(field, None)
    np.testing.assert_allclose(
        nonlinear.electric_field(displacement, None), field, rtol=1e-9, atol=1e-9
    )

    active = phx.solver.maxwell.ActiveGainMaxwellConstitutivePlan(
        0.1, saturation_intensity=1.0
    ).prepare(bridge.cochain, layout)
    assert (
        active.dissipated_power(
            electric,
            magnetic,
            None,
            bridge.cochain.hodge_stars[1],
            bridge.cochain.hodge_stars[2],
        )
        < 0.0
    )


def test_frequency_modes_adjoints_and_reversible_execution():
    bridge = phx.discretization.StructuredCochainBridge(_grid((2, 2, 2)))
    runtime = phx.solver.CompatibleMaxwellPlan(bridge).prepare()
    state = runtime.initialize()
    reversible = phx.solver.maxwell.MaxwellReversibleAdjointPlan(runtime, 2)
    final = reversible.evolve(state, 0.0, 0.05 * runtime.stable_dt)
    diagnostics = reversible.reconstruction_diagnostics(
        state, 0.0, 0.05 * runtime.stable_dt
    )
    assert diagnostics.passed
    assert isinstance(final, phx.solver.CompatibleMaxwellState)

    operator = phx.solver.maxwell.FrequencyMaxwellOperator(
        bridge.cochain, runtime.layout, runtime.constitutive, 0.5
    )
    field = jnp.ones((operator.size,), dtype=complex)
    assert operator.mv(field).shape == field.shape
    if operator.size <= 256:
        modes = operator.eigensystem(min(2, operator.size))
        assert jnp.all(modes.residuals < 1e-7)
    identity = jnp.eye(2, dtype=complex)
    transverse = phx.solver.maxwell.FixedFrequencyGuidedModePlan(
        -jnp.diag(jnp.asarray([4.0, 1.0], dtype=complex)),
        jnp.zeros((2, 2), dtype=complex),
        identity,
        1,
        angular_frequency=1.0,
        right_electric_trace_coefficients=(identity,),
        right_magnetic_trace_coefficients=(identity,),
        left_electric_trace_coefficients=(identity,),
        left_magnetic_trace_coefficients=(identity,),
        divergence_coefficients=(jnp.zeros((1, 2), dtype=complex),),
        power_pairing=identity,
        target_propagation_constant=2.0,
    ).solve()
    np.testing.assert_allclose(transverse.propagation_constants, 2.0)
    assert jnp.all(transverse.polynomial_residuals < 1e-12)
    assert int(transverse.status) == 0

    report = phx.solver.maxwell.audit_directional_derivative(
        lambda value: jnp.sum(value**2),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.5, -0.25]),
    )
    assert report.passed


def test_tetrahedral_whitney_hodge_is_positive_and_oriented():
    vertices = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetrahedra = jnp.asarray([[0, 1, 2, 3]])
    hodge = phx.solver.maxwell.tetrahedral_maxwell_hodge(vertices, tetrahedra)
    assert hodge.quality.passed
    assert hodge.quality.minimum_volume > 0.0
    edge_values = jnp.arange(
        hodge.cochain.cell_counts[1],
        dtype=hodge.electric_mass.dtype,
    )
    face_values = jnp.arange(
        hodge.cochain.cell_counts[2],
        dtype=hodge.magnetic_mass.dtype,
    )
    np.testing.assert_allclose(
        hodge.cochain.apply_hodge(1, edge_values),
        hodge.electric_mass @ edge_values,
    )
    np.testing.assert_allclose(
        hodge.cochain.apply_hodge(2, face_values),
        hodge.magnetic_mass @ face_values,
    )
    expected_codifferential = jnp.linalg.solve(
        hodge.electric_mass,
        hodge.cochain.topology.incidences[1]
        .exterior_derivative()
        .transpose_mv(hodge.magnetic_mass @ face_values),
    )
    np.testing.assert_allclose(
        hodge.cochain.codifferential(2, face_values),
        expected_codifferential,
        rtol=1e-10,
        atol=1e-10,
    )
    assert (
        jnp.linalg.norm(hodge.electric_mass - jnp.diag(jnp.diag(hodge.electric_mass)))
        > 0.0
    )
    np.testing.assert_allclose(
        hodge.cochain.exterior_derivative(
            1, hodge.cochain.exterior_derivative(0, jnp.arange(4.0))
        ),
        0.0,
        atol=0.0,
    )


def test_point_cloud_calculus_reproduces_polynomials_and_diffuses_energy():
    axis = jnp.linspace(-1.0, 1.0, 5)
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    points = jnp.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    boundary = jnp.isclose(jnp.abs(points[:, 0]), 1.0) | jnp.isclose(
        jnp.abs(points[:, 1]), 1.0
    )
    normals = jnp.zeros_like(points)
    normals = normals.at[:, 0].set(
        jnp.where(jnp.isclose(jnp.abs(points[:, 0]), 1.0), points[:, 0], 0.0)
    )
    normals = normals.at[:, 1].set(
        jnp.where(jnp.isclose(jnp.abs(points[:, 1]), 1.0), points[:, 1], 0.0)
    )
    diagonal = jnp.linalg.norm(normals, axis=1)
    normals = normals / jnp.where(diagonal > 0.0, diagonal, 1.0)[:, None]
    discretization = phx.discretization.PointCloudPlan(
        points,
        jnp.ones((points.shape[0],)) / points.shape[0],
        boundary_mask=boundary,
        boundary_normals=normals,
        degree=2,
        neighbor_count=12,
    ).prepare()
    values = points[:, 0] ** 2 + 3.0 * points[:, 1]
    np.testing.assert_allclose(
        discretization.partial_derivative(values, axis=0),
        2.0 * points[:, 0],
        atol=2e-8,
    )
    np.testing.assert_allclose(
        discretization.partial_derivative(values, axis=1),
        3.0,
        atol=2e-8,
    )
    diffusion = phx.discretization.DissipativePointDiffusion(discretization)
    assert diffusion.energy_rate(values) <= 1e-10


def test_high_order_teno_filter_lowering_and_neutral_guardrails():
    values = jnp.linspace(-1.0, 1.0, 32) ** 5
    for order in (6, 8):
        teno = phx.discretization.HighResolutionReconstructionPlan("teno", order=order)
        left, right = teno.reconstruct(values)
        assert left.shape == values.shape
        assert right.shape == values.shape
        assert teno.qualification.passed
    vector_values = jnp.stack((values, values**2), axis=1)

    def identity_eigensystem(left, right, args):
        del right, args
        identity = jnp.broadcast_to(
            jnp.eye(left.shape[1], dtype=left.dtype),
            (left.shape[0], left.shape[1], left.shape[1]),
        )
        return identity, identity, jnp.ones_like(left)

    characteristic = phx.discretization.CharacteristicReconstructionPlan(
        phx.discretization.HighResolutionReconstructionPlan("teno", order=8),
        phx.discretization.CharacteristicSystem(
            identity_eigensystem,
            system_id="identity-eigensystem",
        ),
    )
    characteristic_left, characteristic_right, _ = characteristic.reconstruct(
        vector_values
    )
    component_left, component_right = characteristic.reconstruction.reconstruct(
        vector_values
    )
    np.testing.assert_allclose(characteristic_left, component_left)
    np.testing.assert_allclose(characteristic_right, component_right)
    assert phx.discretization.reconstruction_ghost_width(characteristic) == 4
    filtered = phx.discretization.ExplicitStabilizationPlan(0.1).apply(
        values,
        measure=jnp.ones_like(values),
    )
    np.testing.assert_allclose(jnp.mean(filtered), jnp.mean(values), atol=2e-12)

    buffers = (
        phx.discretization.LoweredBufferSpec("x", (3,), float),
        phx.discretization.LoweredBufferSpec("y", (3,), float),
    )
    kernel = phx.discretization.LoweredKernel(
        "double",
        ("x",),
        ("y",),
        lambda state: {"y": 2.0 * state["x"]},
        lambda state: {"y": 2.0 * state["x"]},
        implementation_id="double-v1",
    )
    parity = phx.discretization.compare_lowered_backends(
        phx.discretization.LoweredOperatorProgram(buffers, (kernel,)),
        {"x": np.arange(3.0), "y": np.zeros(3)},
    )
    assert parity.passed

    neutral = phx.export.NeutralPointCloudSchema(
        jnp.asarray([[0.0], [1.0]]),
        jnp.ones((2,)),
        jnp.zeros((2,), dtype=int),
    )
    assert phx.export.NeutralAdapterBoundary().export(neutral)["kind"] == "point_cloud"
    with pytest.raises(ValueError, match="forbidden"):
        reject_external_runtime("fdtdx")
    assert CoreAbstractionRegistry().owner("cochain").endswith("CochainDiscretization")

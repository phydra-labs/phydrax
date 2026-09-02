#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._sharp_measures import exact_sharp_geometry


def test_fixed_population_flip_workflow_is_jittable_and_transactional():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(mac).prepare()
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac, boundaries=boundaries, tolerance=1.0e-7
    )
    position = jnp.asarray([[0.25, 0.25], [0.40, 0.25], [0.25, 0.40], [0.40, 0.40]])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=2
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(particles)
    compiled = phx.equations.compile_flip_problem(
        phx.equations.FLIPProblemIR("workflow", 1.0, jnp.asarray([0.0, -0.1])),
        transfer,
        projection,
        phx.discretization.flip.FLIPMethodPlan(0.05, liquid_fraction_threshold=0.01),
    )
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    apply = jax.jit(lambda value, dt: compiled.step_detailed(value, dt))
    result = apply(state, jnp.asarray(1.0e-4))
    assert result.successful
    assert result.diagnostics.mass_balance_defect < 1.0e-12
    assert result.diagnostics.momentum_balance_defect < 1.0e-12
    assert result.diagnostics.divergence_norm < 1.0e-6
    assert result.accepted_state.particles.position.shape == position.shape


def _qualified_flip_problem():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    cell_fraction = jnp.ones(finite_volume.cell_shape).at[0, :].set(0.0)
    x_aperture = (
        jnp.ones(finite_volume.face_layouts[0].shape).at[0, :].set(0.0).at[1, :].set(0.0)
    )
    y_aperture = jnp.ones(finite_volume.face_layouts[1].shape).at[0, :].set(0.0)
    geometry = exact_sharp_geometry(
        finite_volume.cell_volumes * cell_fraction,
        finite_volume.cell_volumes,
        (
            finite_volume.face_measures[0] * x_aperture,
            finite_volume.face_measures[1] * y_aperture,
        ),
        finite_volume.face_measures,
        source_id="static-flip-plane",
        source_fidelity="exact-polytope",
        measure_evidence_id="aligned-plane-cell-clipping",
        support_id=finite_volume.support.support_id,
        cell_field_id=finite_volume.cell_space.field_space_id,
        face_field_ids=tuple(space.field_space_id for space in finite_volume.face_spaces),
        operator_id=operators.prepared_id,
        pairing_id=canonical_fingerprint(
            {
                "pressure": operators.pressure_space.space_id,
                "velocity": operators.velocity_space.space_id,
            }
        ),
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=2
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(operators).prepare(
        particles
    )
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        operators, boundaries=boundaries, tolerance=1.0e-7
    )

    def plane(points, time, args):
        del time, args
        return points[..., 0] - 0.125

    def wall(points, time, args):
        del time, args
        return jnp.zeros_like(points)

    collision = phx.discretization.flip.FLIPSolidBoundaryPlan(
        plane,
        wall,
        no_slip=False,
        field_id="static-flip-plane",
    )
    compiled = phx.equations.compile_flip_problem(
        phx.equations.FLIPProblemIR(
            "qualified-workflow",
            1.0,
            jnp.asarray([0.0, 0.0]),
            solid_geometry_source_id="static-flip-plane",
        ),
        transfer,
        projection,
        phx.discretization.flip.FLIPMethodPlan(0.05, liquid_fraction_threshold=0.01),
        geometry=geometry,
        solid_boundary=collision,
    )
    return finite_volume, geometry, transfer, compiled


def test_qualified_flip_transfer_normalizes_only_over_open_support():
    _, geometry, transfer, _ = _qualified_flip_problem()
    position = jnp.asarray([[0.20, 0.25], [0.30, 0.25], [0.20, 0.40], [0.30, 0.40]])
    routes = transfer.build(position)
    result = transfer.particle_to_grid(
        routes, jnp.zeros_like(position), 1.0, geometry=geometry
    )

    assert result.successful
    assert result.geometry_id == geometry.realization_id
    assert jnp.all(result.liquid_fraction[0, :] == 0.0)
    assert jnp.all(~result.face_support[0][0, :])
    assert jnp.isclose(jnp.sum(result.particle_volume_content), 4.0)


def test_qualified_flip_geometry_failure_rolls_back_whole_state():
    _, geometry, _, compiled = _qualified_flip_problem()
    position = jnp.asarray([[0.20, 0.25], [0.30, 0.25], [0.20, 0.40], [0.30, 0.40]])
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    failed = eqx.tree_at(
        lambda value: value.geometry.evidence.accepted,
        compiled,
        jnp.asarray(False),
    )
    result = failed.step_detailed(state, jnp.asarray(1.0e-4))

    assert not result.successful
    assert not result.diagnostics.geometry_accepted
    assert result.accepted_state.geometry_id == geometry.realization_id
    assert result.accepted_state.time == state.time
    assert result.accepted_state.accepted_step == state.accepted_step
    assert jnp.array_equal(
        result.accepted_state.particles.position, state.particles.position
    )

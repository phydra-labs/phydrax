#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native unstructured Euler integration of the driven auxiliary banks."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.discretization.finite_volume import (
    advance_driven_euler_banks,
    driven_euler_cell_keys,
    DrivenEulerSource,
    DrivenEulerSourceArgs,
    PreparedUnstructuredFiniteVolumeDynamics,
)


def test_driven_euler_native_step_ledger_and_rejected_rollback() -> None:
    system = phx.equations.EulerSystem(2)
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        component_names=system.component_names,
    ).prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    source = DrivenEulerSource(1.3)
    problem = phx.equations.ConservationProblemIR(
        "driven-native",
        "state",
        system,
        boundaries,
        source=source,
        source_id="driven-native-source",
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    assert isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(
            cfl=0.4, maximum_retries=0, reduction_factor=0.5
        ),
    )
    state = system.primitive_to_conserved(jnp.asarray(((1.0, 0.1, -0.05, 1.0),)))
    initial = runtime.initialize_state(state, 0.0, 0.001)
    keys = driven_euler_cell_keys(
        jr.key(12), jnp.asarray([9], dtype=jnp.uint32), jnp.asarray([7], dtype=jnp.uint32)
    )
    banks = DrivenEulerSourceArgs(
        jnp.asarray([[0.4, 0.3, 0.2]]),
        jnp.asarray([0.25]),
        keys,
        jnp.asarray([0.0]),
        jnp.asarray([True]),
    )
    result, updated = runtime.advance_with_auxiliary(
        initial, banks, advance_driven_euler_banks, banks
    )
    assert bool(result.accepted)
    dt = float(result.accepted_step_size)
    # A single uniform cell has zero net extrapolation-boundary flux. Independently
    # reconstruct the three SSPRK3 source rates and their content quadrature.
    initial_value = np.asarray(state[0])
    phase = 2 * np.pi * float(jr.uniform(keys[0], (), dtype=state.dtype))

    def rate(t: float, value: np.ndarray) -> np.ndarray:
        acceleration = 0.4 * (0.5 + 0.5 * np.sin(1.3 * (0.25 + t) + phase))
        heat = 0.3 * np.exp(-0.2 * t)
        return np.asarray(
            (0.0, value[0] * acceleration, 0.0, value[1] * acceleration + value[0] * heat)
        )

    rate0 = rate(0.0, initial_value)
    first = initial_value + dt * rate0
    rate1 = rate(dt, first)
    second = 0.75 * initial_value + 0.25 * (first + dt * rate1)
    rate2 = rate(0.5 * dt, second)
    expected = dt * (rate0 / 6 + rate1 / 6 + 2 * rate2 / 3)
    volume = np.asarray(initial.content_state.effective_cell_volumes)[0]
    ledger = result.accepted_flux_integrals
    change = (
        result.runtime_state.content_state.conservative_content
        - initial.content_state.conservative_content
    )
    np.testing.assert_allclose(
        ledger.source_integral[0], volume * expected, rtol=3e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        ledger.scatter_content_integral(), change, rtol=3e-5, atol=2e-7
    )
    np.testing.assert_allclose(change[0], volume * expected, rtol=3e-5, atol=2e-7)
    assert float(updated.material[0, 1]) < float(banks.material[0, 1])
    np.testing.assert_allclose(updated.history, banks.history + dt)
    np.testing.assert_allclose(updated.clock, banks.clock + dt)
    assert not np.array_equal(jr.key_data(updated.keys), jr.key_data(banks.keys))

    cooling = banks._replace(material=banks.material.at[0, 1].set(-1e8))
    rejected, preserved = runtime.advance_with_auxiliary(
        initial, cooling, advance_driven_euler_banks, cooling
    )
    assert not bool(rejected.accepted)
    from phydrax.solver._finite_volume_runtime import FiniteVolumeRunStatus

    assert float(rejected.accepted_step_size) == 0.0
    assert int(rejected.runtime_state.last_status) == int(
        FiniteVolumeRunStatus.RETRY_LIMIT_REACHED
    )
    assert int(rejected.runtime_state.accepted_step) == int(initial.accepted_step)
    np.testing.assert_array_equal(rejected.runtime_state.step_size, initial.step_size)
    np.testing.assert_array_equal(
        rejected.runtime_state.content_state.conservative_content,
        initial.content_state.conservative_content,
    )
    for old, new in zip(cooling, preserved, strict=True):
        if old is cooling.keys:
            old, new = jr.key_data(old), jr.key_data(new)
        np.testing.assert_array_equal(old, new)


def test_driven_euler_overset_hole_and_mixed_precision_atomic_steps() -> None:
    """Real source-cell selection excludes a hole without truncating stored banks."""
    system = phx.equations.EulerSystem(2)
    vertices = np.asarray(
        ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)),
        dtype=np.float64,
    )
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(
            ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)), dtype=np.int32
        ),
        component_names=system.component_names,
    ).prepare()
    faces = np.flatnonzero(
        (np.asarray(discretization.owner_cells) == 1)
        & (np.asarray(discretization.neighbor_cells) == 2)
    )
    face = int(faces[0])
    points = np.asarray(discretization.face_quadrature_points)[face : face + 1]
    normal = -np.asarray(discretization.area_vectors)[face] / float(
        np.asarray(discretization.face_measures)[face]
    )
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((2,), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1,), dtype=np.int32),
        np.asarray((1.0,)),
        hole_mask=np.asarray((True, False, False)),
        epoch_id="driven-hole",
        receptor_face_ids=np.asarray((face,), dtype=np.int32),
        receptor_face_points=points,
        receptor_face_normals=np.broadcast_to(normal, points.shape),
        receptor_face_measures=np.asarray(discretization.face_quadrature_weights)[
            face : face + 1
        ],
        receptor_face_cells=np.asarray((2,), dtype=np.int32),
        receptor_face_orientations=np.asarray((-1,), dtype=np.int8),
        receptor_route_faces=np.asarray((0,), dtype=np.int32),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    precision = phx.discretization.FiniteVolumePrecisionPolicy(
        "float32", flux_dtype="float32", reduction_dtype="float64"
    )
    problem = phx.equations.ConservationProblemIR(
        "driven-hole",
        "state",
        system,
        boundaries,
        source=DrivenEulerSource(1.3),
        source_id="driven-hole-source",
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        precision=precision,
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(overset=overset),
    ).dynamics
    assert isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
    np.testing.assert_array_equal(dynamics.source_cell_indices, np.asarray((1, 2)))
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=0),
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.1, 0.0, 1.0), dtype=jnp.float32), (3, 4)
    )
    state = runtime.initialize_state(system.primitive_to_conserved(primitive), 0.0, 1e-4)
    keys = driven_euler_cell_keys(
        jr.key(15),
        jnp.asarray((0, 1, 2), dtype=jnp.uint32),
        jnp.asarray((9, 9, 9), dtype=jnp.uint32),
    )
    banks = DrivenEulerSourceArgs(
        jnp.asarray(((9, 9, 9), (0.2, 0.3, 0.1), (0.4, 0.5, 0.2)), dtype=jnp.float32),
        jnp.asarray((0.1, 0.2, 0.3), dtype=jnp.float32),
        keys,
        jnp.asarray((0.0, 0.0, 0.0), dtype=jnp.float32),
        jnp.asarray((False, True, True)),
    )
    accepted, updated = runtime.advance_with_auxiliary(
        state,
        banks,
        advance_driven_euler_banks,
        args=banks.select_rows(dynamics.source_cell_indices),
    )
    assert bool(accepted.accepted)
    assert accepted.accepted_step_size.dtype == jnp.float64
    assert (
        updated.material.dtype
        == updated.history.dtype
        == updated.clock.dtype
        == jnp.float32
    )
    np.testing.assert_array_equal(updated.material[0], banks.material[0])
    np.testing.assert_array_equal(
        jr.key_data(updated.keys[0]), jr.key_data(banks.keys[0])
    )
    assert not np.array_equal(
        np.asarray(accepted.runtime_state.content_state.conservative_content[1:]),
        np.asarray(state.content_state.conservative_content[1:]),
    )
    cooling = banks._replace(material=banks.material.at[1:, 1].set(-1e9))
    rejected, retained = runtime.advance_with_auxiliary(
        state,
        banks,
        advance_driven_euler_banks,
        args=cooling.select_rows(dynamics.source_cell_indices),
    )
    assert not bool(rejected.accepted)
    assert float(rejected.accepted_step_size) == 0.0
    np.testing.assert_array_equal(
        rejected.runtime_state.content_state.conservative_content,
        state.content_state.conservative_content,
    )
    for old, new in zip(banks, retained, strict=True):
        if old is banks.keys:
            old, new = jr.key_data(old), jr.key_data(new)
        np.testing.assert_array_equal(old, new)

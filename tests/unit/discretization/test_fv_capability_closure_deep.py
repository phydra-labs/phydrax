#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume import (
    ConservativeMultiblockFluxResult,
    ConservativeMultiblockInterfacePlan,
    EntropyConservativeEulerFluxPlan,
    EntropyStableFluxPlan,
    evaluate_content_form_entropy_diagnostics,
    FiniteVolumeMultiblockRuntimePlan,
    FiniteVolumeStageEpochTransition,
    FluxPositivityPlan,
    lower_static_unstructured_stage_metrics,
    MappedFiniteVolumePlan,
    MappedPeriodicSeamPlan,
    UnstructuredConservativeRemapPlan,
    UnstructuredWENOZReconstructionPlan,
)
from phydrax.equations._entropy_pair import ideal_gas_euler_entropy_pair
from phydrax.solver._unstructured_stage_runtime import (
    PreparedUnstructuredSSPRK3Runtime,
    UnstructuredSSPRK3EpochStageResult,
)


def _grid(shape: Any, *, periodic: Any = None) -> Any:
    periodic_ = (False,) * len(shape) if periodic is None else periodic
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic_[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def _quad_mesh() -> Any:
    vertices = np.asarray(
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
        dtype="float64",
    )
    quads = np.asarray(((0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=quads
    ).prepare()


def _large_quad_mesh() -> Any:
    vertices = np.asarray([(x, y) for y in range(4) for x in range(4)], dtype="float64")
    quads = np.asarray(
        [
            (
                y * 4 + x,
                y * 4 + x + 1,
                (y + 1) * 4 + x + 1,
                (y + 1) * 4 + x,
            )
            for y in range(3)
            for x in range(3)
        ]
    )
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=quads
    ).prepare()


def test_fv_capability_closure_deep_scenario_1() -> None:
    reference = phx.discretization.FiniteVolumePlan(
        _grid((4, 3), periodic=(True, False)),
        component_names=("density", "momentum_x", "momentum_y", "energy"),
    ).prepare()
    seam_plan = MappedPeriodicSeamPlan(
        0, jnp.eye(2), jnp.asarray((1.0, 0.0)), tolerance=1e-10
    )
    mapped = MappedFiniteVolumePlan(
        reference,
        lambda point: point,
        mapping_id="translated-periodic-x",
        periodic_seams=(seam_plan,),
    ).prepare()
    seam = mapped.periodic_seams[0]
    state = jnp.asarray((1.0, 2.0, -3.0, 10.0))
    np.testing.assert_allclose(seam.transform_conserved(state), state)
    np.testing.assert_allclose(
        seam.image(jnp.asarray((0.0, 0.25))), jnp.asarray((1.0, 0.25))
    )
    with pytest.raises(ValueError, match="orthogonal"):
        MappedPeriodicSeamPlan(
            0,
            jnp.asarray(((2.0, 0.0), (0.0, 1.0))),
            jnp.zeros((2,)),
        )
    system = phx.equations.ShallowWaterSystem()
    left = phx.discretization.FiniteVolumePlan(
        _grid((2,)), component_names=system.component_names
    ).prepare()
    right = phx.discretization.FiniteVolumePlan(
        _grid((2,)), component_names=system.component_names
    ).prepare()
    interface = ConservativeMultiblockInterfacePlan(
        left,
        right,
        0,
        0,
        phx.discretization.InterfaceOrientation(0),
        phx.discretization.RusanovFluxPlan(),
    )
    runtime = FiniteVolumeMultiblockRuntimePlan(
        (object(), object()), (interface,), FluxPositivityPlan(8)
    )
    base = (jnp.asarray(((1.0, 0.0), (1.0, 0.0))),) * 2
    high = (jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),) * 2
    invalid_fallback = (jnp.asarray(((-0.5, 0.0), (1.0, 0.0))),) * 2
    flux = ConservativeMultiblockFluxResult(
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.asarray(0.0),
    )
    result = runtime.limit_stage(system, base, high, invalid_fallback, (flux,), (flux,))
    assert not bool(result.accepted)
    np.testing.assert_array_equal(result.states[0], base[0])
    np.testing.assert_array_equal(result.states[1], base[1])
    np.testing.assert_array_equal(result.interface_integrals[0], jnp.zeros((2,)))
    np.testing.assert_array_equal(result.conservation_defect, jnp.zeros((2,)))
    mesh = _quad_mesh()
    reconstruction = phx.discretization.CellPolynomialReconstructionPlan(
        1, oversampling=0
    ).prepare(mesh)
    metrics = lower_static_unstructured_stage_metrics(mesh)
    moved_centers = metrics.cell_centers @ jnp.asarray(((1.0, 0.2), (0.0, 1.0))).T
    moved = eqx.tree_at(lambda item: item.cell_centers, metrics, moved_centers)
    state = (2.0 * moved_centers[:, 0] - 0.5 * moved_centers[:, 1])[:, None]
    coefficients, lengths = reconstruction.stage_coefficients(state, moved)
    routes = jnp.arange(mesh.cell_count, dtype=jnp.int32)
    query = moved_centers[:, None, :] + jnp.asarray((0.02, -0.01))
    reconstructed = reconstruction.evaluate_stage_coefficients(
        state, coefficients, lengths, moved, routes, query
    )[..., 0]
    expected = 2.0 * query[..., 0] - 0.5 * query[..., 1]
    np.testing.assert_allclose(reconstructed, expected, atol=2e-6)

    offsets = jnp.arange(mesh.cell_count + 1, dtype=jnp.int32)
    indices = jnp.arange(mesh.cell_count, dtype=jnp.int32)
    remap = UnstructuredConservativeRemapPlan(
        mesh,
        mesh,
        offsets,
        indices,
        mesh.cell_volumes,
        method="identity-common-refinement",
        provenance="test",
    )
    tangent = jnp.arange(mesh.cell_count, dtype=mesh.cell_volumes.dtype)
    primal = jnp.arange(mesh.cell_count, dtype=mesh.cell_volumes.dtype)
    _, image_tangent = jax.jvp(
        lambda value: remap.apply_fixed_combinatorics(
            value, mesh.cell_volumes, mesh.cell_volumes, mesh.cell_volumes
        ),
        (primal,),
        (tangent,),
    )
    np.testing.assert_allclose(image_tangent, tangent)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="coverage"):
        remap.apply_fixed_combinatorics(
            primal, 0.5 * mesh.cell_volumes, mesh.cell_volumes, mesh.cell_volumes
        ).block_until_ready()
    mesh = _large_quad_mesh()
    polynomial = phx.discretization.CellPolynomialReconstructionPlan(
        2, oversampling=0
    ).prepare(mesh)
    weno = UnstructuredWENOZReconstructionPlan(2, oversampling=0).prepare(mesh)
    metrics = lower_static_unstructured_stage_metrics(mesh)
    translated = eqx.tree_at(
        lambda item: item.cell_centers,
        metrics,
        metrics.cell_centers + jnp.asarray((0.3, -0.2)),
    )
    state = jnp.ones((mesh.cell_count, 1))
    coefficients, lengths = polynomial.stage_coefficients(state, translated)
    routes = jnp.arange(mesh.cell_count, dtype=jnp.int32)
    values = polynomial.evaluate_stage_coefficients(
        state,
        coefficients,
        lengths,
        translated,
        routes,
        translated.cell_centers[:, None, :],
    )
    np.testing.assert_allclose(values, 1.0)
    weno.optimal.stage_coefficients(state, translated)

    deformed = eqx.tree_at(
        lambda item: item.cell_centers,
        metrics,
        metrics.cell_centers.at[0, 0].add(0.1),
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="rigid translation"):
        polynomial.stage_coefficients(state, deformed)[0].block_until_ready()


def test_fv_capability_closure_deep_scenario_2() -> None:
    mesh = _quad_mesh()
    offsets = jnp.arange(mesh.cell_count + 1, dtype=jnp.int32)
    indices = jnp.arange(mesh.cell_count, dtype=jnp.int32)
    incomplete = UnstructuredConservativeRemapPlan(
        mesh,
        mesh,
        offsets,
        indices,
        0.5 * mesh.cell_volumes,
        method="incomplete",
        provenance="test",
        require_complete=False,
    )
    with pytest.raises(ValueError, match="complete remap coverage"):
        FiniteVolumeStageEpochTransition(
            "source", object(), "target", incomplete, 1, "event"
        )
    system = phx.equations.EulerSystem(2)
    pair = ideal_gas_euler_entropy_pair(system)
    central = EntropyConservativeEulerFluxPlan()
    stable = EntropyStableFluxPlan(central, pair)
    left = system.primitive_to_conserved(jnp.asarray((1.0, 0.2, -0.1, 1.0)))
    right = system.primitive_to_conserved(jnp.asarray((0.9, 0.1, 0.05, 0.95)))
    normal = jnp.asarray((3.0, 4.0)) / 5.0
    flux = stable.normal_face_flux(system, left, right, normal)
    residual = pair.normal_interface_entropy_residual(
        left, right, flux.normal_flux, normal
    )
    assert jnp.isfinite(residual)
    assert stable.entropy_dissipation(left, right, flux.max_speed) <= 0.0

    state = jnp.stack((left, right))
    zeros = jnp.zeros_like(state)
    diagnostics = evaluate_content_form_entropy_diagnostics(
        pair,
        state,
        jnp.ones((2,)),
        jnp.zeros((2,)),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    )
    np.testing.assert_allclose(diagnostics.semidiscrete_entropy_rate, 0.0)
    np.testing.assert_allclose(diagnostics.shear_entropy_production, 0.0)


def test_segmented_ssprk_epoch_transfer_and_stage_failure_rollback() -> None:
    mesh = _quad_mesh()
    offsets = jnp.arange(mesh.cell_count + 1, dtype=jnp.int32)
    indices = jnp.arange(mesh.cell_count, dtype=jnp.int32)
    remap = UnstructuredConservativeRemapPlan(
        mesh,
        mesh,
        offsets,
        indices,
        mesh.cell_volumes,
        method="identity",
        provenance="test",
    )
    transition = FiniteVolumeStageEpochTransition(
        "source", "successor", "successor", remap, 1, "event"
    )

    def accept(
        stage: Any, dynamics: Any, step_start: Any, current: Any, dt: Any, args: Any
    ) -> Any:
        del stage, dynamics, step_start, dt, args
        return UnstructuredSSPRK3EpochStageResult(
            current + 1.0, current + 1.0, jnp.asarray(True)
        )

    runtime = PreparedUnstructuredSSPRK3Runtime(
        "source", accept, dynamics_id="source", executor_id="test"
    )
    initial = jnp.zeros((mesh.cell_count, 1))
    accepted = runtime.step(initial, 0.1, stage_transitions=(transition, None))
    assert bool(accepted.accepted)
    assert accepted.final_dynamics == "successor"
    np.testing.assert_allclose(accepted.content, 3.0)

    def reject_second(
        stage: Any, dynamics: Any, step_start: Any, current: Any, dt: Any, args: Any
    ) -> Any:
        del dynamics, step_start, dt, args
        return UnstructuredSSPRK3EpochStageResult(
            current + 1.0,
            current + 1.0,
            jnp.asarray(stage != 2),
        )

    failing = PreparedUnstructuredSSPRK3Runtime(
        "source",
        reject_second,
        dynamics_id="source",
        executor_id="reject-second",
    ).step(initial, 0.1, stage_transitions=(transition, None))
    assert not bool(failing.accepted)
    np.testing.assert_array_equal(failing.content, initial)
    assert int(failing.failed_stage) == 2

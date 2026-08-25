#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _grid(shape, *, periodic=None):
    periodic = (False,) * len(shape) if periodic is None else periodic
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xy"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def _scalar_system(dimension):
    velocity = (0.7, -0.2)
    return phx.equations.ScalarConservationSystem(
        dimension,
        lambda state, axis, args: velocity[axis] * state,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], abs(velocity[axis])),
        system_id=f"mapped-transport-{dimension}",
    )


def test_mapped_identity_preserves_cartesian_measures_and_constant_free_stream():
    reference = phx.discretization.FiniteVolumePlan(_grid((6, 5))).prepare()
    mapped = phx.discretization.MappedFiniteVolumePlan(
        reference, lambda point: point, mapping_id="identity"
    ).prepare()
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    system = _scalar_system(2)
    problem = phx.equations.ConservationProblemIR(
        "mapped-transport",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x", "y"), (pair, pair)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, mapped, method)

    np.testing.assert_allclose(mapped.cell_volumes, reference.cell_volumes, rtol=1e-12)
    residual = compiled(jnp.asarray(0.0), jnp.ones(mapped.state_shape))
    np.testing.assert_allclose(residual, 0.0, atol=2e-12)


def test_warped_mapped_geometry_preserves_constant_flux_divergence():
    reference = phx.discretization.FiniteVolumePlan(_grid((8, 7))).prepare()
    mapped = phx.discretization.MappedFiniteVolumePlan(
        reference,
        lambda point: jnp.stack(
            (point[0] + 0.1 * point[0] * point[1], point[1] + 0.05 * point[0])
        ),
        mapping_id="bilinear-warp",
    ).prepare()
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "warped-transport",
        "state",
        _scalar_system(2),
        phx.discretization.FiniteVolumeBoundarySet(("x", "y"), (pair, pair)),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        mapped,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
    )

    residual = compiled(jnp.asarray(0.0), jnp.ones(mapped.state_shape))
    np.testing.assert_allclose(residual, 0.0, atol=2e-11)
    assert jnp.all(mapped.cell_volumes > 0.0)


def test_conforming_multiblock_interface_uses_one_conservative_flux():
    left = phx.discretization.FiniteVolumePlan(_grid((6,))).prepare()
    right = phx.discretization.FiniteVolumePlan(_grid((5,))).prepare()
    plan = phx.discretization.ConservativeMultiblockInterfacePlan(
        left,
        right,
        0,
        0,
        phx.discretization.InterfaceOrientation(0),
        phx.discretization.RusanovFluxPlan(),
    )
    result = plan.flux(
        _scalar_system(1),
        jnp.ones(left.state_shape),
        2.0 * jnp.ones(right.state_shape),
    )

    np.testing.assert_allclose(result.conservation_defect, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        result.left_integrated_flux + result.right_integrated_flux, 0.0, atol=1e-13
    )


def test_nested_multiblock_interface_sums_fine_fluxes_to_coarse_faces():
    left = phx.discretization.FiniteVolumePlan(_grid((4, 3))).prepare()
    right = phx.discretization.FiniteVolumePlan(_grid((4, 6))).prepare()
    plan = phx.discretization.ConservativeMultiblockInterfacePlan(
        left,
        right,
        0,
        0,
        phx.discretization.InterfaceOrientation(1),
        phx.discretization.RusanovFluxPlan(),
    )
    result = plan.flux(
        _scalar_system(2),
        jnp.ones(left.state_shape),
        2.0 * jnp.ones(right.state_shape),
    )

    assert result.left_integrated_flux.shape == (3, 1)
    assert result.right_integrated_flux.shape == (6, 1)
    np.testing.assert_allclose(result.conservation_defect, 0.0, atol=1e-13)


def test_integrated_flux_register_applies_oriented_reflux_correction():
    register = phx.discretization.FluxRegister(
        jnp.asarray([[2.0], [3.0]]),
        jnp.asarray([[5.0], [1.0]]),
        jnp.asarray([True, False]),
        accumulated_time=0.1,
        orientation=-1,
        refinement_ratio=2,
    )
    state = jnp.asarray([[10.0], [20.0]])
    corrected = register.apply(state, jnp.asarray([0.5, 0.5]))

    np.testing.assert_allclose(register.mismatch(), [[-3.0], [0.0]])
    np.testing.assert_allclose(corrected, [[4.0], [20.0]])
    assert register.refinement_ratio == 2


def test_amr_synchronization_refluxes_then_restricts_covered_cells():
    plan = phx.discretization.ConservativeAMRSynchronizationPlan(1, 2)
    result = plan.advance(
        0.0,
        jnp.asarray([[1.0], [2.0]]),
        jnp.asarray([[1.0], [1.0], [3.0], [3.0]]),
        0.1,
        lambda time, state, dt, args: state,
        lambda time, state, dt, args: state,
        lambda state, args: jnp.zeros_like(state),
        lambda state, args: jnp.zeros_like(state),
        lambda flux: phx.discretization.ConservativeBlockTransfer(1, 2).restrict(flux),
        jnp.asarray([True, False]),
        jnp.asarray([True, True]),
        jnp.asarray([0.5, 0.5]),
    )

    np.testing.assert_allclose(result.restricted_fine_state, [[1.0], [3.0]])
    np.testing.assert_allclose(result.coarse_state, [[1.0], [3.0]])
    np.testing.assert_allclose(result.conservation_defect, [0.0])


def test_amr_register_consumes_accepted_flux_integrals_once():
    def accepted_ledger(
        flux_integral, owner_cells, cell_count, level, start_time, end_time, step
    ):
        block = phx.discretization.FiniteVolumeAcceptedFluxIntegralBlock(
            jnp.asarray(flux_integral),
            jnp.asarray(owner_cells, dtype=jnp.int32),
            jnp.full((len(owner_cells),), -1, dtype=jnp.int32),
            jnp.ones((len(owner_cells),), dtype=bool),
            "mapped-amr:x-interface",
            "mapped-amr-interface",
        )
        versions = (jnp.asarray(0),) * 3
        return phx.discretization.FiniteVolumeAcceptedFluxIntegralLedger(
            (block,),
            jnp.zeros((cell_count, 1)),
            jnp.ones((cell_count,), dtype=bool),
            geometry_family_id=f"mapped-amr:{level}:geometry-family",
            geometry_layout_id=f"mapped-amr:{level}",
            stage_geometry_versions=versions,
            start_geometry_version=jnp.asarray(0),
            end_geometry_version=jnp.asarray(0),
            evidence_policy_id="mapped-amr:accepted-integrals",
            stage_evidence_versions=versions,
            start_evidence_version=jnp.asarray(0),
            end_evidence_version=jnp.asarray(0),
            start_topology_epoch_id=f"mapped-amr:{level}:topology",
            end_topology_epoch_id=f"mapped-amr:{level}:topology",
            start_time=jnp.asarray(start_time),
            end_time=jnp.asarray(end_time),
            accepted_step=jnp.asarray(step, dtype=jnp.int32),
        )

    coarse_ledger = accepted_ledger([[0.4], [0.8]], (0, 1), 2, "coarse", 0.0, 0.2, 1)
    fine_ledgers = (
        accepted_ledger([[0.05], [0.2]], (0, 2), 4, "fine", 0.0, 0.1, 1),
        accepted_ledger([[0.15], [0.4]], (0, 2), 4, "fine", 0.1, 0.2, 2),
    )
    assert coarse_ledger.units == fine_ledgers[0].units == "content"
    assert coarse_ledger.blocks[0].units == fine_ledgers[0].blocks[0].units == "content"
    assert coarse_ledger.blocks[0].route_id != fine_ledgers[0].blocks[0].route_id
    assert coarse_ledger.ledger_id != fine_ledgers[0].ledger_id
    assert fine_ledgers[0].ledger_id == fine_ledgers[1].ledger_id

    coarse = SimpleNamespace(
        accepted=True,
        accepted_step_size=jnp.asarray(0.2),
        accepted_flux_integrals=coarse_ledger,
    )
    fine = tuple(
        SimpleNamespace(
            accepted=True,
            accepted_step_size=jnp.asarray(0.1),
            accepted_flux_integrals=ledger,
        )
        for ledger in fine_ledgers
    )
    register = phx.discretization.flux_register_from_accepted_steps(
        coarse,
        fine,
        0,
        lambda value: value,
        jnp.asarray([True, True]),
    )

    np.testing.assert_allclose(register.coarse_flux, [[0.4], [0.8]])
    np.testing.assert_allclose(register.fine_flux, [[0.2], [0.6]])
    np.testing.assert_allclose(register.mismatch(), [[-0.2], [-0.2]])

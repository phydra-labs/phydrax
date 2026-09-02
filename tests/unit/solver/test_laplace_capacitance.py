import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.solver._laplace_capacitance as capacitance_solver


_TETRA_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _prepared_two_conductors():
    base = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    vertices = jnp.concatenate((base, base + jnp.asarray([3.0, 0.0, 0.0])))
    faces = jnp.concatenate((_TETRA_FACES, _TETRA_FACES + 4))
    policy = phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        near_ratio=1.0,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        target_block_size=3,
        source_block_size=3,
    )
    return phx.operators.prepare_laplace_single_layer_dp0_3d(
        phx.geometry.MeshRegion(vertices, faces), policy=policy
    )


def _selections(prepared):
    left = phx.discretization.EntitySelection(
        prepared.surface_entities,
        jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
    )
    right = phx.discretization.EntitySelection(
        prepared.surface_entities,
        jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    )
    return left, right


def test_capacitance_solver_preserves_names_units_and_existing_potentials(monkeypatch):
    prepared = _prepared_two_conductors()
    left, right = _selections(prepared)
    preparation_calls = []
    prepare_linear = capacitance_solver.prepare_linear

    def counted_prepare_linear(problem, policy):
        preparation_calls.append((problem, policy))
        return prepare_linear(problem, policy)

    monkeypatch.setattr(capacitance_solver, "prepare_linear", counted_prepare_linear)
    plan = capacitance_solver.LaplaceCapacitancePlan3D(
        phx.operators.BoundaryMeshEpoch(prepared._binding.mesh),
        prepared,
        {"right": right, "left": left},
    )
    result = plan.prepare().solve(permittivity=2.0)

    assert result.conductor_names == ("left", "right")
    assert result.layer_density.shape == (8, 2)
    assert result.capacitance.shape == (2, 2)
    assert len(result.linear_results) == 2
    assert len(preparation_calls) == 1
    assert len(result.potentials) == 2
    assert bool(result.valid)
    assert jnp.allclose(result.surface_charge_density, 2.0 * result.layer_density)
    assert jnp.allclose(
        result.potentials[0].density,
        jnp.repeat(result.layer_density[:, 0], prepared.panelization.nodes_per_panel),
    )
    assert result.capacitance_reciprocity_defect < 1.0e-5
    assert jnp.all(jnp.linalg.eigvalsh(result.capacitance) > 0.0)

    values, report = phx.operators.evaluate_laplace_layer_3d(
        result.potentials[0],
        jnp.asarray([[6.0, 0.0, 0.0]]),
        target_side="exterior",
    )
    assert bool(report.pde_membership_valid)
    assert jnp.all(jnp.isfinite(values))


def test_capacitance_plan_rejects_invalid_partitions_and_supports_mathematical_ad():
    prepared = _prepared_two_conductors()
    epoch = phx.operators.BoundaryMeshEpoch(prepared._binding.mesh)
    left, right = _selections(prepared)
    incomplete = phx.discretization.EntitySelection(
        prepared.surface_entities,
        jnp.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
    )
    with pytest.raises(ValueError, match="disjointly cover"):
        capacitance_solver.LaplaceCapacitancePlan3D(
            epoch,
            prepared,
            {"left": incomplete, "right": right},
        )
    plan = capacitance_solver.LaplaceCapacitancePlan3D(
        epoch,
        prepared,
        {"left": left, "right": right},
    )
    with pytest.raises(Exception, match="permittivity must be finite and positive"):
        plan.prepare().solve(permittivity=0.0)

    differentiable = phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
    )
    prepared_capacitance = capacitance_solver.LaplaceCapacitancePlan3D(
        epoch,
        prepared,
        {"left": left, "right": right},
        linear_policy=differentiable,
    ).prepare()
    assert prepared_capacitance.sensitivity.permittivity_differentiable


def test_capacitance_fixed_epoch_coordinate_jvp_and_stable_dual_preconditioner():
    galerkin = _prepared_two_conductors()
    epoch = phx.operators.BoundaryMeshEpoch(galerkin._binding.mesh)
    left, right = _selections(galerkin)
    prepared = capacitance_solver.LaplaceCapacitancePlan3D(
        epoch,
        galerkin,
        {"left": left, "right": right},
    ).prepare()
    result = prepared.solve()
    face_count = galerkin.face_count
    derivative = capacitance_solver.differentiate_laplace_capacitance_coordinates_3d(
        prepared,
        result,
        jnp.zeros((1, face_count, face_count)),
        jnp.zeros((1, face_count)),
    )
    assert bool(derivative.valid)
    assert jnp.allclose(derivative.capacitance_tangent, 0.0)

    preconditioner = capacitance_solver.prepare_laplace_stable_dual_calderon_3d(
        galerkin,
        jnp.eye(face_count),
        jnp.eye(face_count),
        shape_regularity_margin=0.1,
    )
    assert preconditioner.dual_mass_condition_number == 1.0
    assert jnp.all(jnp.isfinite(preconditioner.apply(jnp.ones((face_count,)))))

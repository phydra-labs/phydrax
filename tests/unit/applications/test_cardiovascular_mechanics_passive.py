#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.cardiovascular.anatomy._microstructure import (
    CardiacMaterialFrame,
)
from phydrax.applications.cardiovascular.anatomy._surfaces import ChamberSurfacePlan
from phydrax.applications.cardiovascular.mechanics._chambers import (
    ChamberVolumePlan,
    FollowerPressurePlan,
    MechanicsChamber,
)
from phydrax.applications.cardiovascular.mechanics._guccione import (
    Guccione1991Energy,
    Guccione1991Parameters,
    guccione_1991_reference_energy,
)
from phydrax.applications.cardiovascular.mechanics._holzapfel_ogden import (
    HolzapfelOgden2009Parameters,
    HolzapfelOgden2009TensionOnlyEnergy,
)
from phydrax.applications.cardiovascular.mechanics._materials import (
    cardiac_passive_functional,
    ExactIncompressibleCardiacMaterial,
    FiniteBulkCardiacMaterial,
)
from phydrax.applications.cardiovascular.mechanics._supports import (
    BasalSupport,
    EpicardialSupport,
    PericardialSupport,
    VascularSupport,
)
from phydrax.applications.cardiovascular.mechanics._unloading import (
    ForwardContinuationResult,
    read_unloaded_reference_checkpoint,
    recover_unloaded_reference,
    UnloadedReferenceRecoveryPlan,
    write_unloaded_reference_checkpoint,
)
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def _anatomy_frame() -> CardiacMaterialFrame:
    return CardiacMaterialFrame(
        jnp.asarray(((1.0, 0.0, 0.0),)),
        jnp.asarray(((0.0, 1.0, 0.0),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((True,)),
        frame_id="unit-material-frame",
    )


def _mixed_hexahedral_mesh() -> CellMesh:
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    )
    block = CellBlock(
        "myocardium",
        "hexahedron",
        jnp.asarray(((0, 1, 2, 3, 4, 5, 6, 7),), dtype=jnp.int32),
    )
    return CellMesh(coordinates, (block,))


def _energies():
    frame = _anatomy_frame()
    guccione = Guccione1991Energy(
        Guccione1991Parameters(0.9, 8.0, 2.0, 4.0),
        frame,
        cell_index=0,
    )
    holzapfel = HolzapfelOgden2009TensionOnlyEnergy(
        HolzapfelOgden2009Parameters(0.12, 5.0, 1.8, 8.0, 0.7, 6.0, 0.3, 4.0),
        frame,
        cell_index=0,
    )
    return guccione, holzapfel


def _tetra_surface():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    triangles = jnp.asarray(
        ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
        dtype=jnp.int32,
    )
    return ChamberSurfacePlan("left-ventricle", coordinates, triangles).prepare()


def test_guccione_1991_exact_component_convention_and_anatomy_frame() -> None:
    frame = _anatomy_frame()
    parameters = Guccione1991Parameters(2.0, 3.0, 5.0, 7.0)
    energy = Guccione1991Energy(parameters, frame, cell_index=0)
    deformation = jnp.asarray(((1.1, 0.1, 0.0), (0.0, 0.9, 0.04), (0.0, 0.0, 1.03)))
    strain = 0.5 * (deformation.T @ deformation - jnp.eye(3))
    quadratic = (
        3.0 * strain[0, 0] ** 2
        + 5.0 * (strain[1, 1] ** 2 + strain[2, 2] ** 2 + 2.0 * strain[1, 2] ** 2)
        + 14.0 * (strain[0, 1] ** 2 + strain[0, 2] ** 2)
    )
    expected = jnp.expm1(quadratic)
    assert energy.frame_id == frame.frame_id
    assert energy.frame_cell_index == 0
    assert jnp.allclose(energy(deformation), expected)
    assert jnp.allclose(
        energy(deformation),
        guccione_1991_reference_energy(
            deformation,
            parameters,
            frame.matrix[0],
        ),
    )


def test_holzapfel_ogden_2009_tension_only_convention() -> None:
    frame = _anatomy_frame()
    full = HolzapfelOgden2009TensionOnlyEnergy(
        HolzapfelOgden2009Parameters(0.2, 3.0, 2.0, 7.0, 1.0, 6.0, 0.4, 5.0),
        frame,
        cell_index=0,
    )
    without_fiber = HolzapfelOgden2009TensionOnlyEnergy(
        HolzapfelOgden2009Parameters(0.2, 3.0, 0.0, 7.0, 1.0, 6.0, 0.4, 5.0),
        frame,
        cell_index=0,
    )
    identity = jnp.eye(3)
    fiber_compression = jnp.diag(jnp.asarray((0.9, 1.0 / 0.9, 1.0)))
    fiber_extension = jnp.diag(jnp.asarray((1.1, 1.0 / 1.1, 1.0)))
    assert jnp.allclose(full(identity), 0.0)
    assert jnp.allclose(full(fiber_compression), without_fiber(fiber_compression))
    assert full(fiber_extension) > without_fiber(fiber_extension)


@pytest.mark.parametrize("energy_index", (0, 1))
def test_finite_bulk_objectivity_energy_stress_and_tangent(energy_index: int) -> None:
    energy = _energies()[energy_index]
    material = energy.finite_bulk(80.0)
    deformation = jnp.asarray(((1.08, 0.06, 0.01), (0.02, 0.96, 0.04), (0.0, 0.01, 1.01)))
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    response = material.evaluate(deformation)
    rotated = material.evaluate(rotation @ deformation)
    energy_gradient = jax.grad(material.reference_energy_density)(deformation)
    stress_tangent = jax.jacfwd(
        lambda value: jax.grad(material.reference_energy_density)(value)
    )(deformation)
    assert isinstance(material, FiniteBulkCardiacMaterial)
    assert bool(response.admissible)
    assert jnp.allclose(
        rotated.reference_energy_density, response.reference_energy_density
    )
    assert jnp.allclose(rotated.first_piola, rotation @ response.first_piola, atol=2.0e-5)
    assert jnp.allclose(response.first_piola, energy_gradient, atol=2.0e-5)
    assert jnp.allclose(response.tangent, stress_tangent, atol=2.0e-5)


@pytest.mark.parametrize("energy_index", (0, 1))
def test_exact_mixed_static_adapter_preserves_energy(energy_index: int) -> None:
    energy = _energies()[energy_index]
    exact = energy.exact_incompressible()
    deformation = jnp.asarray(((1.1, 0.04, 0.0), (0.0, 1.0 / 1.1, 0.02), (0.0, 0.0, 1.0)))
    assert jnp.allclose(exact.law.isochoric_value(deformation), energy(deformation))
    assert bool(exact.evaluate(deformation, 0.7).evidence.valid)


def test_exact_mixed_route_block_derivatives_and_lbb_evidence() -> None:
    energy, _ = _energies()
    exact = energy.exact_incompressible()
    deformation = jnp.asarray(((1.1, 0.03, 0.0), (0.0, 1.0 / 1.1, 0.02), (0.0, 0.0, 1.0)))
    pressure = jnp.asarray(1.7)
    response = exact.evaluate(deformation, pressure)
    blocks = exact.block_tangent(deformation, pressure)
    direct_pressure = jax.jacfwd(
        lambda value: exact.evaluate(deformation, value).first_piola
    )(pressure)
    direct_constraint = jax.jacfwd(
        lambda value: exact.evaluate(value, pressure).constraint_residual
    )(deformation)
    assert isinstance(exact, ExactIncompressibleCardiacMaterial)
    assert exact.law.formulation == "exact"
    assert bool(response.evidence.valid)
    assert jnp.allclose(response.constraint_residual, jnp.linalg.det(deformation) - 1.0)
    assert jnp.allclose(blocks.deformation_pressure, direct_pressure)
    assert jnp.allclose(blocks.pressure_deformation, direct_constraint)
    assert jnp.allclose(blocks.deformation_pressure, blocks.pressure_deformation)

    gauge = PressureGaugePolicy("mean-zero")
    mixed_plan = MixedFiniteElementConstraintPlan(
        _mixed_hexahedral_mesh(),
        gauge,
        displacement_field="u",
        pressure_field="p",
        plan_id="unit-exact-mixed-q2-q1",
    )
    qualified = exact.prepare_qualified(
        mixed_plan,
        form_id="unit-cardiac-exact-mixed",
    )
    evidence = qualified.qualification
    prepared = qualified.prepared
    evaluated = prepared.evaluate(prepared.problem.state_space.zeros())
    assert evidence.gauge_mode == "mean-zero"
    assert evidence.pair_names == ("q2-q1",)
    assert evidence.residual_finite
    assert evidence.gauge_valid
    assert evidence.stable_pair
    assert evidence.assembled_inf_sup_stable
    assert evidence.locking_safe
    assert evidence.valid
    assert prepared.spaces.displacement_degree == 2
    assert prepared.spaces.pressure_degree == 1
    assert prepared.inf_sup.adjoint_defect < 1.0e-12
    assert prepared.inf_sup.inf_sup_constant > 0.0
    assert bool(evaluated.valid)


def test_finite_bulk_variational_functional_retains_material_identity() -> None:
    material = _energies()[0].finite_bulk(75.0, material_id="passive-material")
    functional = cardiac_passive_functional(
        "u",
        material,
        region="myocardium",
        functional_id="passive-functional",
    )
    assert functional.identifier == "passive-functional"
    assert functional.variable_fields == ("u",)
    assert functional.terms[0].region == "myocardium"


@pytest.mark.parametrize(
    "support",
    (
        BasalSupport((0.0, 0.0, 1.0), 3.0, 2.0, support_id="base"),
        VascularSupport((1.0, 0.0, 0.0), 3.0, 2.0, support_id="vessel"),
        EpicardialSupport((0.0, 1.0, 0.0), 3.0, 2.0, support_id="epi"),
        PericardialSupport((0.0, 1.0, 0.0), 3.0, 2.0, support_id="peri"),
    ),
)
def test_named_support_energy_traction_and_tangent(support) -> None:
    displacement = jnp.asarray((0.12, -0.07, 0.03))
    response = support.evaluate(displacement)
    gradient = jax.grad(support.energy_density)(displacement)
    tangent = jax.jacfwd(lambda value: support.evaluate(value).restoring_traction)(
        displacement
    )
    assert bool(response.valid)
    assert jnp.allclose(response.energy_gradient, gradient)
    assert jnp.allclose(response.restoring_traction, -gradient)
    assert jnp.allclose(response.traction_tangent, tangent)


def test_support_zero_stiffness_is_exact_traction_free_limit() -> None:
    displacement = jnp.asarray((1.0, -2.0, 3.0))
    support = PericardialSupport((0.0, 0.0, 1.0), 0.0, 0.0)
    response = support.evaluate(displacement)
    assert jnp.allclose(response.energy_density, 0.0)
    assert jnp.allclose(response.restoring_traction, jnp.zeros((3,)))
    assert jnp.allclose(response.traction_tangent, jnp.zeros((3, 3)))


def test_oriented_volume_derivative_follower_work_and_volume_rate() -> None:
    surface = _tetra_surface()
    coordinates = surface.reference_coordinates
    volume_plan = ChamberVolumePlan(surface)
    pressure_plan = FollowerPressurePlan(volume_plan, load_id="lv-pressure")
    chamber = MechanicsChamber("lv", volume_plan, pressure_load_id="lv-pressure")
    pressure = jnp.asarray(2.5)
    direction = jnp.asarray(
        ((0.03, 0.01, 0.02), (-0.01, 0.02, 0.0), (0.0, -0.02, 0.01), (0.02, 0.0, 0.04))
    )
    volume = volume_plan.evaluate(coordinates)
    automatic_gradient = jax.grad(volume_plan.volume)(coordinates)
    pressure_response = pressure_plan.evaluate(coordinates, pressure)
    potential_gradient = jax.grad(
        lambda value: pressure_plan.evaluate(value, pressure).pressure_potential
    )(coordinates)
    epsilon = 1.0e-3
    directional_difference = (
        volume_plan.volume(coordinates + epsilon * direction)
        - volume_plan.volume(coordinates - epsilon * direction)
    ) / (2.0 * epsilon)
    virtual_work = pressure_plan.virtual_work(coordinates, pressure, direction)
    assert bool(volume.valid)
    assert jnp.allclose(volume.volume, 1.0 / 6.0)
    assert jnp.allclose(volume.volume_gradient, automatic_gradient, atol=2.0e-6)
    assert jnp.allclose(pressure_response.nodal_force, -potential_gradient, atol=2.0e-6)
    assert jnp.allclose(
        virtual_work,
        pressure * directional_difference,
        rtol=2.0e-4,
        atol=2.0e-6,
    )
    assert jnp.allclose(
        chamber.volume_rate(coordinates, direction), directional_difference, rtol=2.0e-4
    )
    expanded = 1.1 * coordinates
    assert jnp.allclose(
        pressure_plan.work_between(coordinates, expanded, pressure),
        pressure * (volume_plan.volume(expanded) - volume.volume),
    )
    generic_load = pressure_plan.solid_mechanics_load(pressure)
    assert generic_load.semantics.orientation_id == volume_plan.orientation_id


def test_continuation_unloaded_reference_recovery_and_checkpoint(tmp_path) -> None:
    unloaded = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    load_displacement = jnp.asarray(
        ((0.02, -0.01, 0.03), (0.04, 0.0, 0.01), (0.0, 0.03, -0.02), (-0.01, 0.02, 0.05))
    )
    loaded = unloaded + load_displacement

    def forward_path(reference, load_factors, args):
        del args
        coordinates = (
            reference[None, ...] + load_factors[:, None, None] * load_displacement
        )
        return ForwardContinuationResult(
            coordinates,
            jnp.zeros_like(load_factors),
            jnp.ones_like(load_factors, dtype=bool),
        )

    plan = UnloadedReferenceRecoveryPlan(
        jnp.linspace(0.0, 1.0, 6),
        residual_tolerance=2.0e-6,
        equilibrium_tolerance=1.0e-10,
        maximum_steps=20,
        plan_id="unloading-unit",
    )
    prepared = plan.prepare(loaded, forward_path)
    result = recover_unloaded_reference(prepared, loaded)
    assert bool(result.successful)
    assert jnp.allclose(result.reference_coordinates, unloaded, atol=2.0e-6)
    assert result.state.continuation_coordinates.shape == (6, 4, 3)
    assert jnp.all(result.state.stage_successful)
    assert jnp.allclose(result.state.equilibrium_residual_norm, 0.0)
    assert result.evidence.zero_load_consistent
    assert result.evidence.target_matched

    path = tmp_path / "unloaded-reference.phx"
    write_unloaded_reference_checkpoint(path, result.state)
    restored = read_unloaded_reference_checkpoint(path, prepared)
    assert restored.prepared_id == result.state.prepared_id
    assert restored.state_id == result.state.state_id
    assert jnp.array_equal(restored.reference_coordinates, result.reference_coordinates)

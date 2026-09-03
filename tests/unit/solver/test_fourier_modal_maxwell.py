from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _harmonics():
    return LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )


def _boundary_policy() -> fm.BoundaryCascadePolicy:
    return fm.BoundaryCascadePolicy(
        doublings=6,
        initializer_order=7,
        paired_error=True,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-10,
    )


def test_fresnel_interface_complex_amplitudes_and_power() -> None:
    harmonics = _harmonics()
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    dielectric = fm.FrequencyMaxwellMaterial(4.0, material_id="dielectric")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (),
        fm.HomogeneousMaxwellPort(dielectric, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem)
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "te",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    assert int(result.status) == int(fm.FourierModalSolveStatus.SUCCESS)
    assert float(result.left_outgoing_power[0]) == pytest.approx(
        1.0 / 9.0, rel=2e-6, abs=2e-7
    )
    assert float(result.right_outgoing_power[0]) == pytest.approx(
        8.0 / 9.0, rel=2e-6, abs=2e-7
    )


def test_lossless_film_conserves_power_and_reconstructs_fields() -> None:
    harmonics = _harmonics()
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    film_material = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    layer = fm.FourierModalLayer(
        film_material,
        0.2,
        fm.DirectFourierFactorizationPlan(),
        layer_id="film",
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (layer,),
        fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(
        problem,
        fm.FourierModalSolvePolicy(boundary=_boundary_policy()),
    )
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "tm",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    np.testing.assert_allclose(
        np.asarray(result.net_port_power_into_stack),
        0.0,
        rtol=1e-7,
        atol=1e-9,
    )
    field = fm.fields_in_layer(prepared, result, 0, 0.1)
    assert field.electric_field.shape == harmonics.sample_shape + (3, 1)
    assert field.magnetic_field.shape == harmonics.sample_shape + (3, 1)
    assert bool(jnp.all(jnp.isfinite(field.electric_field)))
    farfield = fm.diffraction_order_far_field(prepared, result)
    assert farfield.power.shape == (1, 2, 1)
    assert bool(jnp.all(farfield.propagating))
    aperture_plan = fm.FiniteApertureFarFieldPlan(
        jnp.asarray(((0.0, 0.0, 1.0),)),
        fm.RectangularFiniteAperture((4.0, 3.0)),
        2,
    )
    aperture_field = fm.finite_aperture_far_field(prepared, result, aperture_plan)
    assert aperture_field.electric_amplitudes.shape == (2, 3, 1)
    assert aperture_field.finite
    np.testing.assert_array_equal(aperture_field.active, (True, False))
    np.testing.assert_array_equal(
        aperture_field.power_density[1],
        jnp.zeros_like(aperture_field.power_density[1]),
    )


def test_full_tensor_layer_operator_matches_finite_contract() -> None:
    harmonics = _harmonics()
    epsilon = jnp.asarray(
        (
            (2.0 + 0.0j, 0.0, 0.2),
            (0.0, 2.4 + 0.0j, 0.0),
            (0.2, 0.0, 2.8 + 0.0j),
        )
    )
    material = fm.FrequencyMaxwellMaterial(epsilon, material_id="anisotropic")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.2, 0.0)),
    )
    assert operator.matrix.shape == (4, 4)
    assert bool(operator.diagnostics.finite)
    assert float(operator.diagnostics.constitutive_residual) < 1e-10


def test_bianisotropic_zero_coupling_parity_and_chiral_operator_are_finite() -> None:
    harmonics = _harmonics()
    factorization = fm.DirectFourierFactorizationPlan()
    baseline = fm.prepare_layer_operator(
        fm.prepare_fourier_material(
            fm.FrequencyMaxwellMaterial(2.0, material_id="baseline"),
            harmonics,
            factorization,
        ),
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.1, 0.0)),
    )
    zeros = jnp.zeros((3, 3))
    explicit = fm.prepare_layer_operator(
        fm.prepare_fourier_material(
            fm.FrequencyMaxwellMaterial(
                2.0,
                magnetoelectric_xi=zeros,
                magnetoelectric_zeta=zeros,
                material_id="explicit-zero",
            ),
            harmonics,
            factorization,
        ),
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.1, 0.0)),
    )
    np.testing.assert_allclose(explicit.matrix, baseline.matrix, atol=1e-12)
    coupling = 0.02j * jnp.eye(3)
    chiral = fm.prepare_layer_operator(
        fm.prepare_fourier_material(
            fm.FrequencyMaxwellMaterial(
                2.0,
                magnetoelectric_xi=coupling,
                magnetoelectric_zeta=-coupling.T,
                material_id="chiral",
            ),
            harmonics,
            factorization,
        ),
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.1, 0.0)),
    )
    assert bool(chiral.diagnostics.finite)
    assert float(chiral.diagnostics.reciprocity_residual) < 1e-10


def test_constant_continuous_layer_and_zero_to_pml_reduce_to_existing_paths() -> None:
    harmonics = _harmonics()
    material = fm.FrequencyMaxwellMaterial(2.0, material_id="continuous-constant")
    factorization = fm.DirectFourierFactorizationPlan()
    port = fm.HomogeneousMaxwellPort(material, port_id="continuous-port")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        port,
        (),
        port,
    )
    continuous = fm.ContinuousFourierModalLayer(
        lambda coordinate: material,
        0.15,
        factorization,
        fm.ContinuousZIntegrationPolicy(
            absolute_tolerance=1.0e-9,
            relative_tolerance=1.0e-7,
            maximum_segments=4,
        ),
        layer_id="constant-profile",
    )
    prepared_continuous = fm.prepare_continuous_fourier_modal_layer(
        problem,
        continuous,
        _boundary_policy(),
    )
    operator = fm.prepare_layer_operator(
        fm.prepare_fourier_material(material, harmonics, factorization),
        harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
    )
    constant_boundary = fm.prepare_layer_boundary(
        operator, continuous.thickness, _boundary_policy()
    )

    assert prepared_continuous.successful
    assert int(jnp.sum(prepared_continuous.segment_active)) == 1
    for actual, expected in zip(
        (
            prepared_continuous.boundary.a,
            prepared_continuous.boundary.b,
            prepared_continuous.boundary.c,
            prepared_continuous.boundary.d,
        ),
        (
            constant_boundary.a,
            constant_boundary.b,
            constant_boundary.c,
            constant_boundary.d,
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-8)

    zero_pml = fm.LateralTransformationOpticsPMLPlan(
        jnp.ones(harmonics.sample_shape + (3,), dtype=jnp.complex128),
        jnp.zeros(harmonics.sample_shape, dtype=bool),
        pml_id="zero-pml",
    )
    transformed = fm.transform_fourier_modal_material(material, harmonics, zero_pml)
    expected = jnp.broadcast_to(
        2.0 * jnp.eye(3, dtype=jnp.complex128),
        harmonics.sample_shape + (3, 3),
    )
    assert transformed.evidence.successful
    np.testing.assert_allclose(transformed.material.permittivity, expected)
    np.testing.assert_allclose(
        transformed.material.magnetoelectric_xi,
        jnp.zeros_like(expected),
    )
    assert transformed.material.material_role == "artificial_pml"
    assert transformed.material.origin_evidence_id == zero_pml.pml_id


def test_continuous_profile_samples_are_never_aliased_by_material_id() -> None:
    harmonics = _harmonics()
    port_material = fm.FrequencyMaxwellMaterial(
        1.0, material_id="continuous-sampling-port"
    )
    port = fm.HomogeneousMaxwellPort(port_material, port_id="port")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.zeros((2,)),
        port,
        (),
        port,
    )
    evaluated_coordinates = []

    def profile(coordinate):
        evaluated_coordinates.append(float(coordinate))
        return fm.FrequencyMaxwellMaterial(
            2.0 + 0.1 * coordinate,
            material_id="one-continuous-law",
        )

    layer = fm.ContinuousFourierModalLayer(
        profile,
        0.2,
        fm.DirectFourierFactorizationPlan(),
        fm.ContinuousZIntegrationPolicy(maximum_segments=2),
        layer_id="graded",
    )
    fm.prepare_continuous_fourier_modal_layer(problem, layer, _boundary_policy())
    assert len(evaluated_coordinates) >= 3
    assert len(set(evaluated_coordinates)) >= 3


def test_zero_thickness_boundary_is_identity() -> None:
    harmonics = _harmonics()
    material = fm.FrequencyMaxwellMaterial(2.0, material_id="uniform")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.0, 0.0)),
    )
    relation = fm.prepare_layer_boundary(operator, 0.0, _boundary_policy())
    np.testing.assert_allclose(np.asarray(relation.a), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.asarray(relation.d), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.asarray(relation.b), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(relation.c), 0.0, atol=1e-12)


def test_resource_plan_accounts_for_retained_layer_and_global_operators() -> None:
    harmonics = _harmonics()
    material = fm.FrequencyMaxwellMaterial(2.0, material_id="resource-material")
    layer = fm.FourierModalLayer(
        material,
        0.1,
        fm.DirectFourierFactorizationPlan(),
        layer_id="resource-layer",
    )
    port = fm.HomogeneousMaxwellPort(material, port_id="exterior")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        port,
        (layer,),
        port,
    )
    itemsize = np.dtype(harmonics.plan.precision.coefficient_dtype).itemsize
    old_layer_only_estimate = 50 * harmonics.harmonic_count**2 * itemsize
    constrained = fm.FourierModalSolvePolicy(
        resources=fm.FourierModalResourcePolicy(
            preparation_bytes=old_layer_only_estimate,
        )
    )
    with pytest.raises(ValueError, match="preparation byte budget"):
        fm.plan_fourier_modal_maxwell(problem, constrained)


def test_boundary_composition_accumulates_initializer_and_paired_errors() -> None:
    identity = jnp.eye(1, dtype=jnp.complex128)
    zero = jnp.zeros_like(identity)
    relation = fm.BoundaryRelation(
        identity,
        zero,
        zero,
        identity,
        fm.BoundaryRelationDiagnostics(
            jnp.asarray(0.0),
            jnp.asarray(0.25),
            jnp.asarray(0.125),
            jnp.asarray(True),
            jnp.asarray(True),
        ),
    )
    doubled = fm.compose_boundary_relations(relation, relation)
    assert float(doubled.diagnostics.initializer_remainder) == pytest.approx(0.5)
    assert float(doubled.diagnostics.paired_error) == pytest.approx(0.25)

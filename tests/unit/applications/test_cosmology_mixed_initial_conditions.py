import hashlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
import phydrax.ein as ein
from phydrax.applications.cosmology._mixed_initial_conditions import (
    component_transfer_payload_bytes,
    ComponentTransferMatrixProduct,
    imported_complex_field_payload_bytes,
    ImportedComplexFieldValidationPlan,
    MixedInitialConditionPlan,
    PrimordialModeRealization,
    SolitonSeedPlan,
    VortexSeedPlan,
    WavePhaseSeedPlan,
)
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.stochastic import GaussianCoefficientRealization
from phydrax.units import ONE


cosmology = phx.applications.cosmology


def _space(count=6, dimension=3):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=tuple("xyz"[:dimension]),
        field_name="initial-condition",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(dimension))
    )


def _reference_binding(
    payload,
    artifact_kind,
    *,
    commercial_use_permitted=True,
):
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = ReferenceArtifactManifest(
        f"{artifact_kind}-fixture",
        checksum_algorithm="sha256",
        checksum=checksum,
        size_bytes=len(payload),
        license_id="internal-test",
        commercial_use_permitted=commercial_use_permitted,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"code": 1.0},
        uncertainty={"fixture": 0.0},
        lineage_ids=("fixture-lineage",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind=artifact_kind,
        content_digest=checksum,
        producer="test-transfer-provider",
        producer_version="current",
        build_id="fixture",
        license_id=manifest.license_id,
        resource_id="in-memory",
        status="complete",
        parent_artifact_ids=(manifest.manifest_id, *manifest.lineage_ids),
    )
    return artifact, manifest


def _transfer(space, *, component_units=None, gauge="synchronous"):
    scale = cosmology.CODE_COSMOLOGY_SCALE
    background = cosmology.FLRWBackground(1.0, 1.0, scale=scale)
    matrix = jnp.asarray([[2.0e-4, 1.0e-4], [-0.5e-4, 3.0e-4]])
    values = jnp.broadcast_to(matrix[:, :, None, None], (2, 2, 2, 2))
    payload = component_transfer_payload_bytes([0.1, 1.0], [0.1, 100.0], values)
    artifact, manifest = _reference_binding(payload, "component-transfer-matrix")
    provenance = cosmology.CosmologyProductProvenance(
        producer=artifact.producer,
        producer_version=artifact.producer_version,
        model_form_id=background.model_form_id,
        request_id="mixed-ic-transfer",
        numerical_policy_id="fixture-linear-interpolation",
        physics_policy_id="correlated-linear-components",
        scale_id=scale.scale_id,
        source_kind="external",
        differentiation="constant",
        parent_product_ids=(artifact.artifact_id, manifest.manifest_id),
    )
    product = ComponentTransferMatrixProduct(
        [0.1, 1.0],
        [0.1, 100.0],
        values,
        components=("cold_baryon", "wave_dark_matter"),
        primordial_components=("adiabatic", "isocurvature"),
        scale=scale,
        provenance=provenance,
        realization=background.realization,
        artifact=artifact,
        manifest=manifest,
        commercial_use=True,
        gauge=gauge,
        component_units=component_units,
        spatial_dimension=len(space.axes),
    )
    return background, product, matrix


def _primordial(space, components=("adiabatic", "isocurvature"), *, key=0):
    mode_ids = PrimordialModeRealization.required_mode_ids(space, components)
    gaussian = GaussianCoefficientRealization.sample(
        jr.key(key), mode_ids, coupling_id="mixed-ic-resolution-coupling"
    )
    return PrimordialModeRealization.from_gaussian_modes(space, gaussian, components)


def _prepared_wave(count=8):
    space = _space(count, dimension=2)
    background = cosmology.FLRWBackground(1.0, 1.0)
    prepared = WaveDarkMatterPlan(
        1.0,
        (0.1, 0.1001),
        gravitational_constant=0.1,
        reduced_planck_constant=1.0,
        step_policy=WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
        ),
    ).prepare(space, background)
    return space, prepared


def test_correlated_component_modes_preserve_exact_auto_cross_spectra_and_identity():
    space = _space(6, dimension=3)
    _, product, matrix = _transfer(space)
    primordial = _primordial(space)

    realized = product.realize(primordial, 0.1)
    expected = np.asarray(ein.contract("cp,dp->cd", matrix, matrix))

    np.testing.assert_allclose(
        product.covariance_values[:, :, 0, 0], expected, rtol=1e-14, atol=0.0
    )
    active = np.asarray(primordial.active_mode_mask)
    for left in range(2):
        for right in range(2):
            np.testing.assert_allclose(
                np.asarray(realized.covariance[..., left, right])[active],
                expected[left, right],
                rtol=1e-13,
                atol=0.0,
            )
    assert bool(realized.successful)
    assert realized.source_realization_id == primordial.realization_id
    assert realized.coupling_id == primordial.coupling_id
    assert realized.provenance_id == product.provenance.provenance_id
    assert realized.artifact_id == product.artifact.artifact_id
    assert realized.manifest_id == product.manifest.manifest_id
    assert realized.requested_use_id == product.requested_use_id
    assert realized.gauge == product.gauge


def test_transfer_rights_denial_manifest_substitution_and_native_generation():
    space = _space(4, dimension=2)
    background, product, matrix = _transfer(space)
    values = jnp.broadcast_to(matrix[:, :, None, None], (2, 2, 2, 2))
    payload = component_transfer_payload_bytes([0.1, 1.0], [0.1, 100.0], values)
    denied_artifact, denied_manifest = _reference_binding(
        payload,
        "component-transfer-matrix",
        commercial_use_permitted=False,
    )
    denied_provenance = cosmology.CosmologyProductProvenance(
        producer=denied_artifact.producer,
        producer_version=denied_artifact.producer_version,
        model_form_id=background.model_form_id,
        request_id="denied-transfer",
        numerical_policy_id="fixture",
        physics_policy_id="fixture",
        scale_id=background.scale.scale_id,
        source_kind="external",
        differentiation="constant",
        parent_product_ids=(
            denied_artifact.artifact_id,
            denied_manifest.manifest_id,
        ),
    )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        ComponentTransferMatrixProduct(
            [0.1, 1.0],
            [0.1, 100.0],
            values,
            components=product.components,
            primordial_components=product.primordial_components,
            scale=background.scale,
            provenance=denied_provenance,
            realization=background.realization,
            artifact=denied_artifact,
            manifest=denied_manifest,
            commercial_use=True,
            gauge=product.gauge,
            spatial_dimension=2,
        )

    _, substitute_manifest = _reference_binding(
        payload + b"substitution", "component-transfer-matrix"
    )
    substituted_provenance = cosmology.CosmologyProductProvenance(
        producer=product.artifact.producer,
        producer_version=product.artifact.producer_version,
        model_form_id=background.model_form_id,
        request_id="substituted-transfer",
        numerical_policy_id="fixture",
        physics_policy_id="fixture",
        scale_id=background.scale.scale_id,
        source_kind="external",
        differentiation="constant",
        parent_product_ids=(
            product.artifact.artifact_id,
            substitute_manifest.manifest_id,
        ),
    )
    with pytest.raises(ValueError, match="digest/license/lineage"):
        ComponentTransferMatrixProduct(
            [0.1, 1.0],
            [0.1, 100.0],
            values,
            components=product.components,
            primordial_components=product.primordial_components,
            scale=background.scale,
            provenance=substituted_provenance,
            realization=background.realization,
            artifact=product.artifact,
            manifest=substitute_manifest,
            gauge=product.gauge,
            spatial_dimension=2,
        )

    native_provenance = cosmology.CosmologyProductProvenance(
        producer="phydrax-native",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="native-transfer",
        numerical_policy_id="native",
        physics_policy_id="native",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiation="constant",
    )
    native = ComponentTransferMatrixProduct(
        [0.1, 1.0],
        [0.1, 100.0],
        values,
        components=product.components,
        primordial_components=product.primordial_components,
        scale=background.scale,
        provenance=native_provenance,
        realization=background.realization,
        artifact=None,
        gauge=product.gauge,
        spatial_dimension=2,
    )
    assert native.artifact is None
    assert native.manifest is None
    assert native.requested_use_id == "native-generated"


def test_primordial_mode_ids_couple_resolutions_deterministically():
    fine_space = _space(8, dimension=2)
    coarse_space = _space(4, dimension=2)
    fine = _primordial(fine_space, key=17)

    coarse = fine.at_resolution(coarse_space)
    selected = fine.gaussian.select(coarse.mode_ids)

    assert coarse.coupling_id == fine.coupling_id
    assert set(coarse.mode_ids).issubset(fine.mode_ids)
    np.testing.assert_array_equal(coarse.gaussian.coefficients, selected.coefficients)
    np.testing.assert_array_equal(coarse.gaussian.mode_ids, coarse.mode_ids)
    assert bool(coarse.successful)


def test_mixed_plan_rejects_gauge_and_unit_mismatch():
    space = _space(4, dimension=3)
    _, product, _ = _transfer(space)
    capacity = int(np.prod(space.physical_shape))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.full((capacity,), 1.0 / capacity),
        ambient_dimension=3,
    ).prepare()
    lpt = cosmology.LagrangianPerturbationInitialConditionPlan(
        particles, space.physical_shape, (1.0, 1.0, 1.0)
    )

    with pytest.raises(ValueError, match="gauges disagree"):
        MixedInitialConditionPlan(product, lpt, gauge="newtonian")
    with pytest.raises(ValueError, match="qualified only for cold_baryon"):
        MixedInitialConditionPlan(
            product,
            lpt,
            particle_component="wave_dark_matter",
        )

    with pytest.raises(ValueError, match="units and cosmology scale disagree"):
        _transfer(space, component_units=(ONE, ONE))


def test_particle_projection_uses_lpt_and_closes_modes_and_mass():
    space = _space(4, dimension=3)
    background, product, _ = _transfer(space)
    primordial = _primordial(space, key=5)
    capacity = int(np.prod(space.physical_shape))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.full((capacity,), 2.0 / capacity),
        ambient_dimension=3,
    ).prepare()
    lpt = cosmology.LagrangianPerturbationInitialConditionPlan(
        particles,
        space.physical_shape,
        (1.0, 1.0, 1.0),
        order=2,
    )
    growth = cosmology.FLRWGrowthPlan(jnp.geomspace(0.01, 1.0, 24)).solve(background)
    plan = MixedInitialConditionPlan(
        product,
        lpt,
        particle_target_mass=2.0,
        mode_relative_tolerance=1.0e-9,
    )

    modes = plan.realize_modes(primordial, 0.1)
    projected = plan.project_particles(modes, background, growth)

    assert bool(projected.successful)
    np.testing.assert_allclose(projected.particle_mass, 2.0, rtol=1e-14)
    assert projected.mass_relative_residual < 1e-14
    assert projected.mode_relative_residual < 1e-10
    np.testing.assert_allclose(
        projected.recovered_modes,
        projected.target_modes,
        rtol=1e-10,
        atol=1e-12,
    )


def test_compatible_current_reconstructs_mean_zero_phase_and_wave_mass():
    space, prepared = _prepared_wave()
    x = space.axes[0].nodes[:, None]
    phase = 0.02 * jnp.sin(2.0 * jnp.pi * x)
    density = jnp.ones(space.physical_shape) * 2.0
    velocity_x = 0.04 * jnp.pi * jnp.cos(2.0 * jnp.pi * x) / 0.1**2
    current = jnp.stack(
        (
            jnp.broadcast_to(density * velocity_x, space.physical_shape),
            jnp.zeros(space.physical_shape),
        ),
        axis=-1,
    )
    plan = WavePhaseSeedPlan(prepared)

    result = plan.realize(
        density,
        current,
        0.1,
        density_unit=plan.density_unit,
        current_unit=plan.current_unit,
        current_convention=plan.current_convention,
    )

    assert bool(result.successful)
    assert result.evidence.current_relative_residual < 1e-10
    assert result.evidence.curl_relative_residual < 1e-10
    assert result.evidence.phase_gauge_absolute < 1e-12
    assert result.evidence.mass_relative_residual < 1e-12
    np.testing.assert_allclose(result.evidence.mass, 2.0, rtol=1e-12)
    np.testing.assert_allclose(
        jnp.angle(result.state.psi),
        jnp.broadcast_to(phase, space.physical_shape),
        rtol=1e-10,
        atol=1e-10,
    )


def test_phase_seed_rejects_curl_circulation_nodes_and_unit_mismatch():
    space, prepared = _prepared_wave()
    plan = WavePhaseSeedPlan(prepared)
    y = space.axes[1].nodes[None, :]
    density = jnp.ones(space.physical_shape)
    rotational = jnp.stack(
        (
            jnp.broadcast_to(jnp.sin(2.0 * jnp.pi * y), space.physical_shape),
            jnp.zeros(space.physical_shape),
        ),
        axis=-1,
    )

    curled = plan.realize(
        density,
        rotational,
        0.1,
        density_unit=plan.density_unit,
        current_unit=plan.current_unit,
        current_convention=plan.current_convention,
    )
    assert not bool(curled.successful)
    assert int(curled.evidence.status) == 3
    assert curled.evidence.curl_relative_residual > 0.1

    circulation = jnp.broadcast_to(jnp.asarray((1.0, 0.0)), space.physical_shape + (2,))
    harmonic = plan.realize(
        density,
        circulation,
        0.1,
        density_unit=plan.density_unit,
        current_unit=plan.current_unit,
        current_convention=plan.current_convention,
    )
    assert not bool(harmonic.successful)
    assert int(harmonic.evidence.status) == 4

    with_node = density.at[0, 0].set(0.0)
    node = plan.realize(
        with_node,
        jnp.zeros_like(rotational),
        0.1,
        density_unit=plan.density_unit,
        current_unit=plan.current_unit,
        current_convention=plan.current_convention,
    )
    assert not bool(node.successful)
    assert int(node.evidence.status) == 2

    with pytest.raises(ValueError, match="density units disagree"):
        plan.realize(
            density,
            jnp.zeros_like(rotational),
            0.1,
            density_unit=ONE,
            current_unit=plan.current_unit,
            current_convention=plan.current_convention,
        )


def test_soliton_normalization_and_integer_vortex_winding_are_distinct():
    _, prepared = _prepared_wave()
    soliton = SolitonSeedPlan(prepared, (0.5, 0.5), 0.12, 3.0).realize(0.1)
    vortex_plan = VortexSeedPlan(prepared, (0.5, 0.5), 0.08, 2.0, 1)
    vortex = vortex_plan.realize(0.1)

    assert soliton.seed_kind == "soliton"
    assert vortex.seed_kind == "vortex"
    assert bool(soliton.successful)
    assert bool(vortex.successful)
    np.testing.assert_allclose(soliton.evidence.wave_mass, 3.0, rtol=1e-12)
    np.testing.assert_allclose(vortex.evidence.wave_mass, 2.0, rtol=1e-12)
    np.testing.assert_allclose(vortex.evidence.measured_winding, 1.0, atol=1e-12)
    np.testing.assert_allclose(
        vortex.evidence.measured_antivortex_winding, -1.0, atol=1e-12
    )
    np.testing.assert_allclose(vortex.evidence.net_winding, 0.0, atol=1e-12)
    assert bool(vortex.evidence.node_present)
    assert not np.array_equal(soliton.state.psi, vortex.state.psi)

    with pytest.raises(TypeError, match="must be an integer"):
        VortexSeedPlan(prepared, (0.5, 0.5), 0.08, 2.0, 1.5)


def test_imported_complex_field_validation_binds_rights_payload_and_result_identity():
    space, prepared = _prepared_wave()
    psi = jnp.ones(space.physical_shape, dtype=jnp.complex128) * jnp.sqrt(2.0)
    payload = imported_complex_field_payload_bytes(psi, 0.1, 2.0)
    artifact, manifest = _reference_binding(payload, "external-complex-wave-field")
    plan = ImportedComplexFieldValidationPlan(
        prepared,
        artifact,
        manifest,
        expected_artifact_kind="external-complex-wave-field",
        export=True,
    )

    result = plan.validate(
        psi,
        0.1,
        2.0,
        wavefunction_unit=plan.wavefunction_unit,
        coordinate_convention=plan.coordinate_convention,
        normalization=plan.normalization,
    )

    assert bool(result.successful)
    assert result.artifact_id == artifact.artifact_id
    assert result.manifest_id == manifest.manifest_id
    assert result.requested_use_id == plan.requested_use_id
    assert result.provider == artifact.producer
    assert result.provider_version == artifact.producer_version
    assert result.validation_id != plan.plan_id
    np.testing.assert_allclose(result.evidence.wave_mass, 2.0, rtol=1e-12)

    with pytest.raises(ValueError, match="checksum mismatch"):
        plan.validate(
            psi.at[0, 0].set(3.0 + 0.0j),
            0.1,
            2.0,
            wavefunction_unit=plan.wavefunction_unit,
            coordinate_convention=plan.coordinate_convention,
            normalization=plan.normalization,
        )
    with pytest.raises(PermissionError, match="training-use-not-permitted"):
        ImportedComplexFieldValidationPlan(
            prepared,
            artifact,
            manifest,
            expected_artifact_kind="external-complex-wave-field",
            training_use=True,
        )
    with pytest.raises(ValueError, match="artifact kind disagrees"):
        ImportedComplexFieldValidationPlan(
            prepared,
            artifact,
            manifest,
            expected_artifact_kind="wrong-kind",
        )

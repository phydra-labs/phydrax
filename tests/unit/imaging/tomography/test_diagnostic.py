import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


_M2_PER_KG = phx.units.derived_unit(
    "m2/kg", ((phx.units.METER, 2), (phx.units.KILOGRAM, -1))
)


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-diagnostic-photon-table",
        checksum_algorithm="sha256",
        checksum="1" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"coefficient": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic-diagnostic-photon",),
    )


def _provenance():
    return phx.nuclear.NuclearDataProvenance(
        _manifest(),
        "https://example.invalid/synthetic-diagnostic-photon",
        "synthetic-diagnostic-photon",
        "test-release",
        "test-evaluation",
    )


def _energy_grid():
    return phx.equations.PhotonEnergyGrid(np.asarray((40.0, 80.0)) * 1.602176634e-16)


def _support():
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="ct-scanner",
    )
    rays = phx.measurement.RaySampleSupport(
        np.asarray(((-1.0, 0.5, 0.5),)),
        np.asarray(((1.0, 0.0, 0.0),)),
        ("ray-0",),
        contract,
        far=np.asarray((5.0,)),
    )
    return phx.imaging.tomography.ProjectionSupport(rays, (1,), ("view-0",))


def _protocol(grid, *, exposure_basis="absolute", dark=5.0, gain=3.0):
    tomography = phx.imaging.tomography
    return tomography.CTAcquisitionProtocol(
        (
            tomography.CTViewAcquisition(
                "view-0",
                tomography.TubeSpectrum(grid, np.asarray((100.0, 200.0)), exposure_basis),
                tomography.FilterStack(grid, np.asarray((1.0, 0.5))),
                tomography.BowtieTransmission(grid, np.asarray((1.0, 0.5))),
                tomography.AECSetting(2.0, exposure_basis),
                tomography.DetectorResponse(
                    grid,
                    np.asarray((1.0, 2.0)),
                    dark_signal=dark,
                    gain=gain,
                ),
            ),
        )
    )


def _coefficients(grid):
    return phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        grid,
        ("material-a", "material-b"),
        np.asarray(((1.0, 2.0), (3.0, 4.0))),
        _M2_PER_KG,
        _provenance(),
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )


def test_material_basis_projection_preserves_declared_order():
    support = _support()
    transform = phx.imaging.tomography.VoxelXRayTransformPlan(
        support,
        (2, 1, 1),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        support.rays.coordinate_contract,
    )
    plan = phx.imaging.tomography.MaterialBasisProjectionPlan(
        transform, ("material-a", "material-b")
    )
    density = jnp.asarray((0.1, 0.2)).reshape((2, 1, 1))
    fractions = jnp.asarray(((1.0, 0.0), (0.0, 1.0))).reshape((2, 1, 1, 2))
    np.testing.assert_allclose(plan.project(density, fractions), ((0.1, 0.2),))
    with pytest.raises(ValueError, match="nonnegative"):
        plan.project(density, fractions.at[0, 0, 0, 0].set(-1.0))
    with pytest.raises(ValueError, match="sum to one"):
        plan.project(density, 0.5 * fractions)


def test_two_material_two_energy_signal_and_detector_semantics():
    grid = _energy_grid()
    plan = phx.imaging.tomography.PolychromaticDetectorPlan(
        _support(), _protocol(grid), _coefficients(grid)
    )
    scatter = 7.0
    result = plan.evaluate(
        jnp.asarray(((0.1, 0.2),)),
        scatter_signal=scatter,
        scatter_label=phx.imaging.tomography.ScatterLabel(
            "synthetic-scatter", "absolute"
        ),
    )
    primary = 200.0 * np.exp(-0.7) + 200.0 * np.exp(-1.0)
    expected = 5.0 + 3.0 * (primary + scatter)
    np.testing.assert_allclose(result.transmitted_signal, (primary,), rtol=2.0e-6)
    np.testing.assert_allclose(result.scatter_signal, (scatter,))
    np.testing.assert_allclose(result.expected_signal, (expected,), rtol=2.0e-6)
    assert bool(result.successful)


def test_energy_and_exposure_bases_fail_closed():
    grid = _energy_grid()
    other_grid = phx.equations.PhotonEnergyGrid(
        np.asarray((41.0, 81.0)) * 1.602176634e-16
    )
    tomography = phx.imaging.tomography
    with pytest.raises(ValueError, match="share one photon-energy grid"):
        tomography.CTViewAcquisition(
            "view-0",
            tomography.TubeSpectrum(grid, np.ones(2), "absolute"),
            tomography.FilterStack(other_grid, np.ones(2)),
            tomography.BowtieTransmission(grid, np.ones(2)),
            tomography.AECSetting(1.0, "absolute"),
            tomography.DetectorResponse(grid, np.ones(2)),
        )
    narrow_grid = phx.equations.PhotonEnergyGrid(
        np.asarray((50.0, 70.0)) * 1.602176634e-16
    )
    with pytest.raises(ValueError, match="does not support the protocol grid"):
        tomography.PolychromaticDetectorPlan(
            _support(), _protocol(grid), _coefficients(narrow_grid)
        )
    with pytest.raises(ValueError, match="cannot be mixed"):
        tomography.CTViewAcquisition(
            "view-0",
            tomography.TubeSpectrum(grid, np.ones(2), "relative"),
            tomography.FilterStack(grid, np.ones(2)),
            tomography.BowtieTransmission(grid, np.ones(2)),
            tomography.AECSetting(1.0, "absolute"),
            tomography.DetectorResponse(grid, np.ones(2)),
        )


def test_scatter_requires_a_label_and_forward_model_is_differentiable():
    grid = _energy_grid()
    plan = phx.imaging.tomography.PolychromaticDetectorPlan(
        _support(), _protocol(grid), _coefficients(grid)
    )
    with pytest.raises(TypeError, match="requires a ScatterLabel"):
        plan.evaluate(jnp.asarray(((0.1, 0.2),)), scatter_signal=1.0)
    with pytest.raises(ValueError, match="cannot be mixed"):
        plan.evaluate(
            jnp.asarray(((0.1, 0.2),)),
            scatter_signal=1.0,
            scatter_label=phx.imaging.tomography.ScatterLabel(
                "relative-scatter", "relative"
            ),
        )

    gradient = jax.grad(
        lambda areal_mass: jnp.sum(plan.evaluate(areal_mass).expected_signal)
    )(jnp.asarray(((0.1, 0.2),)))
    assert gradient.shape == (1, 2)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(gradient < 0.0))

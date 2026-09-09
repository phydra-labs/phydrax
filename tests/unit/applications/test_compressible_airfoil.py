import jax.numpy as jnp
import numpy as np

from phydrax.applications.compressible_flow import (
    AirfoilOGridPlan,
    AirfoilSectionPlan,
    RAE2822CasePlan,
    TransonicFixedLiftPlan,
)
from phydrax.qualification import ReferenceArtifactManifest


def _ellipse_section():
    angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    coordinates = np.stack((0.5 * np.cos(angle), 0.1 * np.sin(angle)), axis=-1)
    return AirfoilSectionPlan(coordinates)


def test_airfoil_section_distance_and_o_grid_are_finite():
    section = _ellipse_section()
    distance = section.closest_distance(jnp.asarray(((0.0, 0.0), (1.0, 0.0))))
    np.testing.assert_allclose(distance, (0.1, 0.5), atol=2.0e-3)

    prepared = AirfoilOGridPlan(section, 32, 8, 2.0).prepare(
        ("density", "momentum_x", "momentum_y", "total_energy")
    )
    assert bool(prepared.successful)
    assert prepared.discretization.cell_shape == (32, 8)
    assert jnp.all(prepared.wall_distance > 0.0)
    assert prepared.minimum_cell_volume > 0.0


def test_fixed_lift_uses_declared_bracket_and_residual():
    plan = TransonicFixedLiftPlan(0.5, (0.0, 0.5), tolerance=1.0e-10)
    result = plan.solve(lambda angle, args: 0.1 + 2.0 * angle)

    assert bool(result.successful)
    np.testing.assert_allclose(result.angle_of_attack, 0.2, atol=1.0e-9)
    np.testing.assert_allclose(result.lift_coefficient, 0.5, atol=1.0e-9)
    assert abs(float(result.residual)) <= 1.0e-10


def test_rae_case_binds_reference_rights_and_exact_conditions():
    manifest = ReferenceArtifactManifest(
        "rae2822-reference",
        checksum_algorithm="sha256",
        checksum="0" * 64,
        size_bytes=1,
        license_id="reference-only-test",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"chord": 1.0},
        uncertainty=None,
        lineage_ids=("rae2822",),
    )
    case = RAE2822CasePlan(
        manifest,
        mach=0.729,
        reynolds_number=6.5e6,
        target_lift_coefficient=0.724,
        reference_temperature=288.15,
    )
    accepted = case.qualify_lift(0.7245, tolerance=1.0e-3)
    rejected = case.qualify_lift(0.73, tolerance=1.0e-3)

    assert bool(accepted.successful)
    assert not bool(rejected.successful)
    assert accepted.case_id == case.case_id

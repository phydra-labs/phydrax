from fractions import Fraction
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.geometry.brep._curve_approximation_bounds import (
    certify_coupled_approximation,
    propose_intersection_approximation,
)
from phydrax.geometry.brep._curve_approximation_topology import (
    certify_branch_approximation_topology,
    certify_branch_trim_separation,
)
from phydrax.geometry.brep._intersection import (
    exact_plane_cylinder_conic_section,
    intersect_surface_regions,
)
from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._constructors import brep_cylinder
from phydrax.interchange._cad import CadCurveFitPolicy, CadExportPolicy
from phydrax.interchange._step import write_step
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    CylinderPatch,
    LineCurve,
    PlanePatch,
)


def test_coupled_curve_bounds_cover_dense_independent_samples_and_exact_roots() -> None:
    plane_patch = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.5), (0.0, 1.0, 0.25))
    plane = SurfaceRegion(
        plane_patch,
        np.asarray(((-2.0, -2.0), (2.0, 2.0)), dtype=np.float64),
    )
    cylinder_patch = CylinderPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    cylinder = SurfaceRegion(
        cylinder_patch,
        np.asarray(((0.0, -1.0), (math.pi / 2.0, 1.0)), dtype=np.float64),
    )
    result = intersect_surface_regions(plane, cylinder)
    assert result.complete and len(result.curves) == 1
    source = result.curves[0]
    circle = CircleCurve(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        1.0,
        center_exact=(Fraction(0), Fraction(0), Fraction(0)),
        first_axis_exact=(Fraction(1), Fraction(0), Fraction(0)),
        second_axis_exact=(Fraction(0), Fraction(1), Fraction(0)),
        radius_exact=Fraction(1),
    )
    exact_section = exact_plane_cylinder_conic_section(
        plane,
        cylinder,
        (
            ConicParameterRoot(circle, Fraction(1), Fraction(0)),
            ConicParameterRoot(circle, Fraction(0), Fraction(1)),
        ),
    )
    assert exact_section is not None
    exact_roots = np.asarray(
        tuple(
            tuple(float(value) for value in root.exact_point())
            for root in exact_section.endpoint_roots
        ),
        dtype=np.float64,
    )
    source_ends = np.asarray(
        (
            source.evaluate_coupled(jnp.asarray(0.0, dtype=jnp.float64)).point,
            source.evaluate_coupled(
                jnp.asarray(float(source.num_charts), dtype=jnp.float64)
            ).point,
        ),
        dtype=np.float64,
    )
    if np.sum(np.linalg.norm(source_ends - exact_roots, axis=1)) > np.sum(
        np.linalg.norm(source_ends - exact_roots[::-1], axis=1)
    ):
        exact_roots = exact_roots[::-1]
    fitted = propose_intersection_approximation(
        source, endpoint_points=(exact_roots[0], exact_roots[1])
    )
    evidence = certify_coupled_approximation(
        source,
        *fitted,
        distance_tolerance=2.0e-2,
        parameter_tolerance=2.0e-2,
    )
    topology = certify_branch_approximation_topology(source, *fitted, evidence)
    trim_evidence = certify_branch_trim_separation(
        topology,
        LineCurve((10.0, 10.0), (1.0, 0.0)),
        "first",
        first=0.0,
        last=1.0,
    )

    parameters = np.linspace(0.0, float(source.num_charts), 1025, dtype=np.float64)
    source_values = np.asarray(source.evaluate_coupled(jnp.asarray(parameters)).point)
    fitted_values = np.asarray(fitted[0].evaluate(jnp.asarray(parameters)))
    dense_error = np.max(np.linalg.norm(source_values - fitted_values, axis=1))
    pairwise = np.linalg.norm(
        source_values[:, None, :] - fitted_values[None, :, :], axis=2
    )
    dense_hausdorff = max(
        float(np.max(np.min(pairwise, axis=1))),
        float(np.max(np.min(pairwise, axis=0))),
    )
    first_exact_uv = np.asarray(source.p_curve("first").evaluate(jnp.asarray(parameters)))
    second_exact_uv = np.asarray(source.p_curve("second").evaluate(jnp.asarray(parameters)))
    first_fit_uv = np.asarray(fitted[1].evaluate(jnp.asarray(parameters)))
    second_fit_uv = np.asarray(fitted[2].evaluate(jnp.asarray(parameters)))
    first_lift_error = np.max(
        np.linalg.norm(plane_patch.evaluate(jnp.asarray(first_fit_uv)) - fitted_values, axis=1)
    )
    second_lift_error = np.max(
        np.linalg.norm(cylinder_patch.evaluate(jnp.asarray(second_fit_uv)) - fitted_values, axis=1)
    )
    parameter_error = max(
        np.max(np.linalg.norm(first_exact_uv - first_fit_uv, axis=1)),
        np.max(np.linalg.norm(second_exact_uv - second_fit_uv, axis=1)),
    )
    fit_roots = np.asarray(
        (
            fitted[0].evaluate(jnp.asarray(0.0, dtype=jnp.float64)),
            fitted[0].evaluate(jnp.asarray(float(source.num_charts), dtype=jnp.float64)),
        )
    )

    assert dense_error <= evidence.distance_bound
    assert dense_hausdorff <= evidence.distance_bound
    assert parameter_error <= evidence.parameter_bound
    assert first_lift_error <= evidence.first_correspondence_bound
    assert second_lift_error <= evidence.second_correspondence_bound
    assert evidence.distance_bound <= 2.0e-2
    assert evidence.parameter_bound <= 2.0e-2
    assert topology.separated_pairs > 0
    assert trim_evidence.separation_lower_bound > 0.0
    np.testing.assert_allclose(fit_roots, exact_roots, atol=1.0e-14)


def test_step_export_uses_certified_intersection_approximation(
    tmp_path: Path,
) -> None:
    contract = SpatialCoordinateContract.si()
    first = brep_cylinder(
        1.0,
        4.0,
        base_center=(0.0, 0.0, -2.0),
        coordinate_contract=contract,
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(-2.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=contract,
    )
    model = boolean_brep(first, second, "intersection").model
    destination = tmp_path / "certified-intersection.step"
    policy = CadExportPolicy(CadCurveFitPolicy(tolerance=0.05))

    exported = write_step(model, destination, policy=policy)

    assert destination.is_file()
    assert exported.report.valid
    assert any(
        loss.path.startswith("edge:") and loss.path.endswith(".intersection-curve")
        and loss.changes_interpretation
        for loss in exported.report.losses
    )

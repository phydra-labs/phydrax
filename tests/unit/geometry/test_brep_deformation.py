"""Polynomial deformation, with closed-form oracles independent of composition."""

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry.brep._constructors import brep_box, BRepTessellationPolicy
from phydrax.geometry.brep._deformation import (
    _prepare_composition,
    certify_deformation,
    deform_brep,
    deformation_feasibility_scale,
    identity_lattice,
    PolynomialDeformation,
    TaperedCylinderPatch,
)
from phydrax.geometry.brep._differentiable import FixedTopologyBRepSource
from phydrax.geometry.brep._linked_parameters import validate_linked_geometry
from phydrax.geometry.brep._parameter_range import CertifiedParameterRange
from phydrax.geometry.brep._patches import (
    BernsteinSurfacePiece,
    BSplineSurfacePatch,
    CylinderPatch,
)
from phydrax.geometry.design._schema import ParameterCollector, ParameterId
from phydrax.units import METER
from tests._support.cad_models import native_solid


# Empirical `test_tapered_cylinder_is_exact_cone_and_inverse`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_TAPERED_CYLINDER_INVERSE_POINT_RTOL = 2e-15
# Empirical `test_interior_box_volume_and_control_derivative_closed_form`: matched max |delta|=1.11022e-16; joint allowance fraction=0.0555112, margin=94.4489%.
_BOX_DEFORMATION_SURFACE_POINT_ATOL = 2e-15
# Empirical `test_interior_box_volume_and_control_derivative_closed_form`: matched max |delta|=1.38778e-17; joint allowance fraction=9.15635e-05, margin=99.9908%.
_BOX_DEFORMATION_VOLUME_RTOL = 2e-12
# Empirical `test_interior_box_volume_and_control_derivative_closed_form`: matched max |delta|=1.38778e-17; joint allowance fraction=9.15635e-05, margin=99.9908%.
_BOX_DEFORMATION_VOLUME_ATOL = 2e-14
# Empirical `test_interior_box_volume_and_control_derivative_closed_form`: matched max |delta|=1.56125e-17; joint allowance fraction=1.23655e-05, margin=99.9988%.
_BOX_CONTROL_VOLUME_DERIVATIVE_RTOL = 2e-11
# Empirical `test_interior_box_volume_and_control_derivative_closed_form`: matched max |delta|=6.33174e-15; joint allowance fraction=0.000232283, margin=99.9768%.
_BOX_CONTROL_FINITE_DIFFERENCE_RTOL = 2e-9
# Empirical `test_tapered_cylinder_is_exact_cone_and_inverse`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_TAPERED_CYLINDER_SURFACE_POINT_RTOL = 2e-15
# Empirical `test_rational_patch_composition_degree_and_weighted_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_RATIONAL_COMPOSITION_POINT_RTOL = 2e-15
# Empirical `test_rational_patch_composition_degree_and_weighted_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_RATIONAL_COMPOSITION_POINT_ATOL = 2e-15


_UNIT = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float64)
_INNER = np.asarray(((0.125, 0.25, 0.25), (0.375, 0.75, 0.75)), dtype=np.float64)


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(200_000_000, bytes_upper=2_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=200_000_000, maximum_memory_bytes=2_000_000_000
        )
        with budget.activate():
            yield budget


def _lattice(displacement: float) -> PolynomialDeformation:
    controls = identity_lattice(_UNIT, (2, 2, 2)).at[1, 1, 1, 0].add(displacement)
    return PolynomialDeformation(_UNIT, controls)


def test_fixed_trilinear_lattice_is_identity() -> None:
    value = PolynomialDeformation(_UNIT, identity_lattice(_UNIT, (1, 1, 1)))
    with _budget() as budget:
        evidence = certify_deformation(value, maximum_cells=8, coordinate_budget=budget)
    assert evidence.status == "certified"
    assert evidence.cells[0].determinant_range == (1.0, 1.0)
    assert np.prod(_UNIT[1] - _UNIT[0]) == 1.0
    points = jnp.asarray(((0.125, 0.25, 0.75), (1.25, 0.5, 0.5)), dtype=jnp.float64)
    np.testing.assert_array_equal(value.evaluate(points), points)
    controls = identity_lattice(_UNIT, (1, 1, 1)).at[0, 0, 0, 0].add(0.125)
    with pytest.raises(ValueError, match="boundary_controls"):
        PolynomialDeformation(_UNIT, controls)


def test_interior_box_volume_and_control_derivative_closed_form() -> None:
    # F = (x + delta*b(x)*b(y)*b(z), y, z), b(t)=2t(1-t).
    # det J = 1 + delta*b'(x)*b(y)*b(z). The independent rational
    # integral is V = 1/16 + delta*(1/4)*(11/48)^2.
    coefficient = Fraction(1, 4) * Fraction(11, 48) ** 2
    delta = 0.25
    expected = Fraction(1, 16) + Fraction(delta) * coefficient
    model = brep_box(
        _INNER[0],
        _INNER[1],
        coordinate_contract=SpatialCoordinateContract(METER),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    value = _lattice(delta)
    original_id = model.model_id, model.require_geometry().geometry_id
    with _budget() as budget:
        result = deform_brep(
            model,
            value,
            maximum_cells=32,
            coordinate_budget=budget,
            feature_id="lattice-volume",
        )
    assert result.status == "certified", result.reason
    parameter_map = result.parameter_map
    assert parameter_map is not None
    collector = ParameterCollector(METER)
    bound = parameter_map.compile(collector, model)
    schema, state = collector.finish()
    assert schema.parameter_ids == (ParameterId("lattice-volume", "control_points"),)
    assert state.values[0].shape == (1, 3)
    realization = validate_linked_geometry(bound.realize(state), model)
    assert all(isinstance(patch, BernsteinSurfacePiece) for patch in realization.patches)
    assert all(
        patch.controls.shape == (7, 7, 4)
        for patch in realization.patches
        if isinstance(patch, BernsteinSurfacePiece)
    )
    assert realization.geometry.pcurves == model.require_geometry().pcurves
    assert (model.model_id, realization.geometry.geometry_id) == original_id
    for source, target, domain in zip(
        model.patches,
        realization.patches,
        np.asarray(model.parameter_bounds),
        strict=True,
    ):
        parameter = jnp.asarray(np.mean(domain, axis=0), dtype=jnp.float64)
        np.testing.assert_allclose(
            target.evaluate(parameter),
            value.evaluate(source.evaluate(parameter)),
            rtol=0,
            atol=_BOX_DEFORMATION_SURFACE_POINT_ATOL,
        )
    compiled = FixedTopologyBRepSource(model, parameter_map=parameter_map).compile(
        coordinate_unit=METER
    )
    index = compiled.schema.index(ParameterId("lattice-volume", "control_points"))

    def volume(displacement: Array) -> Array:
        point = (
            jnp.asarray([[0.5, 0.5, 0.5]], dtype=jnp.float64).at[0, 0].add(displacement)
        )
        return compiled.kernel.measure(compiled.state.replace_at(index, point))

    primal = float(volume(jnp.asarray(delta, dtype=jnp.float64)))
    assert primal == pytest.approx(
        float(expected),
        rel=_BOX_DEFORMATION_VOLUME_RTOL,
        abs=_BOX_DEFORMATION_VOLUME_ATOL,
    )
    automatic = float(jax.grad(volume)(jnp.asarray(delta, dtype=jnp.float64)))
    step = 2.0**-12
    finite_difference = (
        float(volume(jnp.asarray(delta + step, dtype=jnp.float64)))
        - float(volume(jnp.asarray(delta - step, dtype=jnp.float64)))
    ) / (2 * step)
    assert automatic == pytest.approx(
        float(coefficient), rel=_BOX_CONTROL_VOLUME_DERIVATIVE_RTOL
    )
    assert automatic == pytest.approx(
        finite_difference, rel=_BOX_CONTROL_FINITE_DIFFERENCE_RTOL
    )


def test_fold_reports_negative_cell_and_exhaustion_reports_unresolved() -> None:
    with _budget() as budget:
        folded = certify_deformation(
            _lattice(3.0), maximum_cells=128, coordinate_budget=budget
        )
        unresolved = certify_deformation(
            _lattice(3.0), maximum_cells=1, coordinate_budget=budget
        )
    assert folded.status == "negative"
    assert any(
        cell.status == "negative" and cell.determinant_range[1] < 0
        for cell in folded.cells
    )
    assert unresolved.status == "unresolved"
    assert (
        unresolved.cells[0].determinant_range[0]
        < 0
        < unresolved.cells[0].determinant_range[1]
    )


def test_tapered_cylinder_is_exact_cone_and_inverse() -> None:
    model = native_solid("cylinder", catalog="geometry")
    box = np.asarray(((-2.0, -2.0, -1.0), (2.0, 2.0, 4.0)), dtype=np.float64)
    value = PolynomialDeformation.taper(box, axis=2, rate=0.25)
    with _budget() as budget:
        result = deform_brep(
            model,
            value,
            maximum_cells=16,
            coordinate_budget=budget,
            feature_id="taper-cylinder",
        )
    assert result.status == "certified", result.reason
    parameter_map = result.parameter_map
    assert parameter_map is not None
    pairs = [
        (source, target)
        for source, target in zip(model.patches, parameter_map.patches, strict=True)
        if isinstance(source, CylinderPatch)
    ]
    assert pairs
    parameters = jnp.asarray((0.75, 1.25), dtype=jnp.float64)
    for source, target in pairs:
        assert isinstance(target, TaperedCylinderPatch)
        assert float(target.radial_slope) == pytest.approx(float(source.radius) / 4)
        angle, height = 0.75, 1.25
        expected = (
            np.asarray(source.origin)
            + height * np.asarray(source.axis)
            + float(source.radius)
            * (1.0 + height / 4)
            * (
                np.cos(angle) * np.asarray(source.first_axis)
                + np.sin(angle) * np.asarray(source.second_axis)
            )
        )
        np.testing.assert_allclose(
            target.evaluate(parameters),
            expected,
            rtol=_TAPERED_CYLINDER_SURFACE_POINT_RTOL,
        )
        np.testing.assert_allclose(
            target.evaluate(parameters),
            value.evaluate(source.evaluate(parameters)),
            rtol=_TAPERED_CYLINDER_SURFACE_POINT_RTOL,
        )
    point = jnp.asarray((0.5, -0.25, 1.5), dtype=jnp.float64)
    np.testing.assert_allclose(
        value.inverse(value.evaluate(point), certificate=result.certificate),
        point,
        rtol=_TAPERED_CYLINDER_INVERSE_POINT_RTOL,
    )
    different = PolynomialDeformation.shear(box, axis=0, driver=2, rate=0.25)
    with pytest.raises(ValueError, match="mismatched"):
        different.inverse(point, certificate=result.certificate)


@pytest.mark.parametrize("kind,degree", [("taper", 2), ("shear", 1)])
def test_rational_patch_composition_degree_and_weighted_oracle(
    kind: str, degree: int
) -> None:
    box = np.asarray(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)), dtype=np.float64)
    controls = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.5)), ((1.0, 0.0, 0.25), (1.0, 1.0, 1.0))),
        dtype=np.float64,
    )
    weights = np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64)
    knots = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    patch = BSplineSurfacePatch(controls, weights, knots, knots, 1, 1)
    value = (
        PolynomialDeformation.taper(box, axis=2, rate=0.125)
        if kind == "taper"
        else PolynomialDeformation.shear(box, axis=0, driver=2, rate=0.125)
    )
    with _budget() as budget:
        composition = _prepare_composition(
            value, patch, np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64), budget
        )
    assert composition.template.controls.shape == (degree + 1, degree + 1, 4)
    for u, v in ((0.0, 0.0), (0.25, 0.5), (0.75, 0.25), (1.0, 1.0)):
        basis = np.outer((1 - u, u), (1 - v, v)) * weights
        point = np.sum(basis[..., None] * controls, axis=(0, 1)) / np.sum(basis)
        expected = point.copy()
        if kind == "taper":
            expected[:2] *= 1.0 + point[2] / 8
        else:
            expected[0] += point[2] / 8
        np.testing.assert_allclose(
            composition.template.evaluate(jnp.asarray((u, v), dtype=jnp.float64)),
            expected,
            rtol=_RATIONAL_COMPOSITION_POINT_RTOL,
            atol=_RATIONAL_COMPOSITION_POINT_ATOL,
        )
    assert np.all(
        np.asarray(composition.template.controls_lower)
        <= np.asarray(composition.template.controls)
    )
    assert np.all(
        np.asarray(composition.template.controls)
        <= np.asarray(composition.template.controls_upper)
    )


def test_crossing_patch_refuses_and_outside_patch_remains_identity() -> None:
    crossing = brep_box(
        (-0.25, 0.25, 0.25),
        (0.25, 0.75, 0.75),
        coordinate_contract=SpatialCoordinateContract(METER),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    outside = brep_box(
        (2.0, 2.0, 2.0),
        (3.0, 3.0, 3.0),
        coordinate_contract=SpatialCoordinateContract(METER),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    with _budget() as budget:
        refused = deform_brep(
            crossing,
            _lattice(0.25),
            maximum_cells=16,
            coordinate_budget=budget,
            feature_id="crossing",
        )
        accepted = deform_brep(
            outside,
            _lattice(0.25),
            maximum_cells=16,
            coordinate_budget=budget,
            feature_id="outside",
        )
    assert refused.status == "unsupported"
    assert "crossing" in refused.reason
    assert accepted.status == "certified"
    assert accepted.parameter_map is not None
    assert accepted.parameter_map.patches == outside.patches


def test_feasible_scale_certifies_interval_not_endpoint_samples() -> None:
    # delta=3 has first singularity at scale 2/3: min det=1-3*scale/2.
    with _budget() as budget:
        interval = deformation_feasibility_scale(
            _lattice(3.0),
            maximum_scale=1.0,
            refinements=6,
            maximum_cells=64,
            coordinate_budget=budget,
        )
    assert isinstance(interval, CertifiedParameterRange)
    lower, upper = interval.interval
    assert lower == 0.0
    assert upper is not None and 0 < upper < Fraction(2, 3)
    assert interval.end_status == ("attained_exactly", "certified_inner_bound")
    assert interval.constraints == ("parameter_domain", "det_j_sign")
    assert any("searched scale cap is 1.0" in entry for entry in interval.evidence)
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with pytest.raises(CoordinateEnclosureResourceError):
            certify_deformation(
                _lattice(0.25), maximum_cells=16, coordinate_budget=budget
            )


def test_named_nonpositive_scale_refuses() -> None:
    value = PolynomialDeformation.taper(_UNIT, axis=2, rate=-1.0)
    with _budget() as budget:
        certificate = certify_deformation(
            value, maximum_cells=8, coordinate_budget=budget
        )
    assert certificate.status == "unsupported"


def test_shared_parameter_range_on_success_and_fold_refusal() -> None:
    model = brep_box(
        _INNER[0],
        _INNER[1],
        coordinate_contract=SpatialCoordinateContract(METER),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    with _budget() as budget:
        successful = deform_brep(
            model,
            _lattice(0.25),
            maximum_cells=32,
            coordinate_budget=budget,
            feature_id="range-success",
            include_parameter_range=True,
            range_refinements=3,
        )
        refused = deform_brep(
            model,
            _lattice(3.0),
            maximum_cells=64,
            coordinate_budget=budget,
            feature_id="range-fold",
            range_refinements=3,
        )
    assert successful.status == "certified"
    assert successful.parameter_range is not None
    assert successful.parameter_range.interval == (0.0, 1.0)
    assert successful.parameter_range.end_status == (
        "attained_exactly",
        "certified_inner_bound",
    )
    assert refused.status == "negative"
    assert refused.parameter_range is not None
    lower, upper = refused.parameter_range.interval
    assert lower == 0.0
    assert upper is not None and 0 < upper < 2 / 3
    assert refused.parameter_range.constraints == ("parameter_domain", "det_j_sign")


def test_scale_cell_bounds_are_physical_determinants_on_nonunit_box() -> None:
    box = np.asarray(((-1.0, -2.0, -3.0), (1.0, 2.0, 5.0)), dtype=np.float64)
    controls = identity_lattice(box, (2, 2, 2)).at[1, 1, 1, 0].add(0.5)
    value = PolynomialDeformation(box, controls)
    with _budget() as budget:
        interval = deformation_feasibility_scale(
            value,
            maximum_scale=1.0,
            refinements=1,
            maximum_cells=1,
            coordinate_budget=budget,
        )
    assert interval.interval == (0.0, 1.0)
    # det J = 1 + (0.5/2)*s*b'(x)*b(y)*b(z), b(t)=2t(1-t).
    # Root Bernstein controls span [1-1/2,1+1/2], independent of the
    # physical reference volume 2*4*8=64.
    assert "Accepted physical determinant bounds: (0.5, 1.5)." in interval.evidence


def test_named_form_axes_refuse_boolean_indices() -> None:
    with pytest.raises(TypeError, match="PolynomialDeformation.axis"):
        PolynomialDeformation.taper(_UNIT, axis=True, rate=0.25)
    with pytest.raises(TypeError, match="PolynomialDeformation.driver"):
        PolynomialDeformation.shear(_UNIT, axis=0, driver=False, rate=0.25)


def test_refusal_reasons_match_certified_cell_outcomes() -> None:
    with _budget() as budget:
        negative = certify_deformation(
            _lattice(3.0), maximum_cells=64, coordinate_budget=budget
        )
        unresolved = certify_deformation(
            _lattice(3.0), maximum_cells=1, coordinate_budget=budget
        )
    assert negative.status == "negative"
    assert negative.reason == "negative determinant in a certified cell"
    assert unresolved.status == "unresolved"
    assert (
        unresolved.reason == "determinant sign unresolved within the declared cell budget"
    )


def test_named_scale_producer_returns_shared_unknown_range() -> None:
    value = PolynomialDeformation.taper(_UNIT, axis=2, rate=0.25)
    with _budget() as budget:
        result = deformation_feasibility_scale(
            value,
            maximum_scale=1.0,
            refinements=1,
            maximum_cells=1,
            coordinate_budget=budget,
        )
    assert isinstance(result, CertifiedParameterRange)
    assert result.interval == (None, None)
    assert result.end_status == ("unknown", "unknown")

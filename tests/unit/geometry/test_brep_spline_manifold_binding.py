from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization.iga._brep_surface_binding import (
    _at_interior_knot,
    BRepSplineManifoldBinding,
)
from phydrax.discretization.iga._manifold import ParametricBasisPayload
from phydrax.geometry.brep._constructors import (
    brep_planar_face,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import BSplineSurfacePatch


def _model(height: float = 0.25) -> BRepModel:
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="spline-manifold-binding",
    )
    controls = np.array(
        [
            [[u / 2, v / 2, height if (u, v) == (1, 1) else 0.0] for v in range(3)]
            for u in range(3)
        ],
        dtype=np.float64,
    )
    weights = np.ones((3, 3))
    weights[1, 1] = 2.0
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    patch = BSplineSurfacePatch(controls, weights, knots, knots, 2, 2)
    return BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def test_rational_spline_second_jet_reaches_manifold_consumer() -> None:
    model = _model()
    binding = BRepSplineManifoldBinding(model, 0, full_patch_domain=True)
    geometry = binding.evaluate(np.array([[0.5, 0.5]]))
    assert binding.face_id == model.face_ids[0]
    assert binding.geometry_id == model.geometry.geometry_id
    assert geometry.source_id == binding.binding_id
    np.testing.assert_allclose(geometry.frame.origin[0], [0.5, 0.5, 0.1], atol=1e-12)
    # z = (0.5 A(u) A(v))/(1 + A(u) A(v)), A(t)=2t(1-t).
    np.testing.assert_allclose(
        geometry.mapping_hessian[0, 2], [[-0.64, 0.0], [0.0, -0.64]], atol=1e-12
    )
    covariant = geometry.covariant_hessian(np.array([[1.0, 0.0]]), np.zeros((1, 2, 2)))
    np.testing.assert_allclose(
        covariant, -np.asarray(geometry.christoffel[:, 0]), atol=1e-12
    )


def test_binding_identity_tracks_source_revision_and_rejects_wrong_inputs() -> None:
    model = _model()
    binding = BRepSplineManifoldBinding(model, 0, full_patch_domain=True)
    changed = BRepSplineManifoldBinding(_model(0.125), 0, full_patch_domain=True)
    assert binding.face_id != changed.face_id
    assert binding.binding_id != changed.binding_id
    assert (
        binding.binding_id
        != BRepSplineManifoldBinding(
            model, 0, full_patch_domain=True, rank_tolerance=1e-8
        ).binding_id
    )
    with pytest.raises(ValueError, match="rank_tolerance"):
        BRepSplineManifoldBinding(model, 0, full_patch_domain=True, rank_tolerance=0.0)
    with pytest.raises(IndexError):
        BRepSplineManifoldBinding(model, 1, full_patch_domain=True)
    with pytest.raises(TypeError):
        BRepSplineManifoldBinding(model, True, full_patch_domain=True)
    with pytest.raises(ValueError, match="full_patch_domain"):
        BRepSplineManifoldBinding(model, 0, full_patch_domain=False)
    with pytest.raises(ValueError, match="closed unit chart"):
        binding.evaluate(np.array([-0.01, 0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="trailing dimension"):
        binding.evaluate(np.array([0.5], dtype=np.float64))


def test_bound_basis_realization_consumes_second_geometry_jet() -> None:
    binding = BRepSplineManifoldBinding(_model(), 0, full_patch_domain=True)
    payload = ParametricBasisPayload(
        np.array([[0.3]]),
        np.array([[[1.0, 0.0]]]),
        np.zeros((1, 1, 2, 2)),
        provider_revision="linear-u",
    )
    realization = binding.realize_basis(np.array([[0.3, 0.5]]), payload)
    correction = np.asarray(realization.geometry.christoffel[:, 0])
    assert np.max(np.abs(correction)) > 1e-4
    np.testing.assert_allclose(realization.covariant_hessians[:, 0], -correction)
    assert realization.geometry.source_id == binding.binding_id


def test_interior_knot_is_detected_by_exact_reference_pullback() -> None:
    lower, upper, knot = Fraction(1, 10), Fraction(9, 10), Fraction(3, 10)
    assert _at_interior_knot(np.array([0.25]), lower, upper, (knot,))
    assert not _at_interior_knot(np.array([0.2]), lower, upper, (knot,))


def test_knot_admission_refuses_affine_execution_straddling_exact_knot() -> None:
    lower, upper = Fraction(1, 10), Fraction(9, 10)
    numeric_value = float(lower) + 0.25 * (float(upper) - float(lower))
    knot = Fraction.from_float(numeric_value)
    assert float((knot - lower) / (upper - lower)) != 0.25
    assert _at_interior_knot(np.array([0.25]), lower, upper, (knot,))

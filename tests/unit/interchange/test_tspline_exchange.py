from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax._interpolation import TMesh2D
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import (
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
    brep_planar_face,
)
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import BSplineSurfacePatch
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.interchange._cad import CadImportPolicy
from phydrax.interchange._step import read_step, write_step


_CONTRACT = SpatialCoordinateContract.si()
_POLICY = CadImportPolicy(_CONTRACT)


def _model() -> BRepModel:
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            ),
        ),
        coordinate_contract=_CONTRACT,
    )
    mesh = TMesh2D.tensor_product((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    exact_controls = tuple(
        (*anchor.local_knots.greville_parameter, Fraction(0)) for anchor in mesh.anchors
    )
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in exact_controls]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_controls,
        exact_weights=tuple(Fraction(1) for _ in exact_controls),
    )
    return BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=_CONTRACT,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def test_single_piece_t_spline_step_exchange_reports_structure_loss(tmp_path: Path) -> None:
    model = _model()
    archive_path = tmp_path / "native-tspline.brepz"
    save_brep_archive(model, archive_path)
    restored_native = load_brep_archive(archive_path)
    assert restored_native.model_id == model.model_id
    assert isinstance(restored_native.patches[0], TSplineSurfacePatch)
    destination = tmp_path / "native-tspline.step"

    exported = write_step(model, destination)
    restored = read_step(destination, _POLICY, trusted_root=tmp_path)

    assert exported.report.valid and restored.report.valid
    assert any(
        loss.path == "face:0.tspline-refinement"
        and loss.changes_interpretation
        for loss in exported.report.losses
    )
    patch = restored.model.patches[0]
    assert isinstance(patch, BSplineSurfacePatch)
    parameters = jnp.asarray(((0.15, 0.25), (0.5, 0.5), (0.85, 0.75)))
    expected = np.asarray(model.patches[0].evaluate(parameters), dtype=np.float64)
    actual = np.asarray(patch.evaluate(parameters), dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=2.0e-12)

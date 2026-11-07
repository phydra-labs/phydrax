from fractions import Fraction as F
from inspect import signature
from math import pi, tau

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry._surface_affine_cell_proof import (
    certify_affine_surface_cell,
    certify_periodic_affine_surface_cell,
)
from phydrax.geometry.brep._patches import CylinderPatch, PlanePatch, SpherePatch
from phydrax.geometry.brep._trim_helpers import PeriodOffset


O = (0, 0, 0)
X = (1, 0, 0)
Y = (0, 1, 0)
Z = (0, 0, 1)
TRIANGLE = ((F(0), F(0)), (F(1), F(0)), (F(0), F(1)))


def test_proof_binds_exact_triangle_and_current_source() -> None:
    plane = PlanePatch(O, X, Y)
    proof = certify_affine_surface_cell(plane, TRIANGLE)
    assert proof is not None and proof.determinant == 1
    assert proof.replay(plane, TRIANGLE)
    assert not proof.replay(PlanePatch(O, X, Y), TRIANGLE)
    assert not proof.replay(plane, (TRIANGLE[0], TRIANGLE[2], TRIANGLE[1]))
    assert certify_affine_surface_cell(PlanePatch(O, X, X), TRIANGLE) is None
    assert (
        certify_affine_surface_cell(plane, (TRIANGLE[0], TRIANGLE[0], TRIANGLE[2]))
        is None
    )
    invalid = signature(certify_affine_surface_cell).bind(
        plane, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    assert certify_affine_surface_cell(*invalid.args, **invalid.kwargs) is None


def test_nonorthogonal_source_frame_is_not_normalized() -> None:
    patch = CylinderPatch(O, (2, 0, 0), (1, 3, 0), (1, 0, 4), 2)
    assert certify_affine_surface_cell(patch, TRIANGLE) is not None
    value = patch.evaluate(jnp.asarray((0.0, 1.0)))
    np.testing.assert_allclose(np.asarray(value), (5, 0, 4), rtol=1e-6)


def test_native_sheet_refuses_decks_and_poles_without_authority() -> None:
    cylinder = CylinderPatch(O, X, Y, Z, 2)
    assert certify_affine_surface_cell(cylinder, TRIANGLE) is not None
    wide = ((F(0), F(0)), (F.from_float(tau) + F(1, 100), F(0)), (F(0), F(1)))
    assert certify_affine_surface_cell(cylinder, wide) is None
    sphere = SpherePatch(O, X, Y, Z, 2)
    assert certify_affine_surface_cell(sphere, TRIANGLE) is not None
    beyond_pole = ((F(0), F(0)), (F(1), F(0)), (F(0), F.from_float(pi / 2) + F(1, 100)))
    assert certify_affine_surface_cell(sphere, beyond_pole) is None


def test_symbolic_period_cell_is_proved_without_float_period_snapping() -> None:
    cylinder = CylinderPatch(O, X, Y, Z, 2)
    turn = PeriodOffset(F(0), F(1))
    corners = ((turn, F(0)), (turn - F(1), F(0)), (turn, F(1)))
    proof = certify_periodic_affine_surface_cell(cylinder, corners)
    assert proof is not None and proof.orientation == -1
    assert proof.replay(cylinder, corners)
    degenerate = (corners[0], corners[1], corners[1])
    assert certify_periodic_affine_surface_cell(cylinder, degenerate) is None


def test_cylinder_chart_jvp_matches_exact_composition() -> None:
    patch = CylinderPatch(O, X, Y, Z, 2)
    proof = certify_affine_surface_cell(patch, TRIANGLE)
    assert proof is not None
    point = jnp.asarray((0.25, 0.35))
    direction = jnp.asarray((0.2, -0.4))
    value, derivative = jax.jvp(patch.evaluate, (point,), (direction,))
    u, v = np.asarray(point)
    du, dv = np.asarray(direction)
    np.testing.assert_allclose(
        np.asarray(value), (2 * np.cos(u), 2 * np.sin(u), v), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(derivative), (-2 * np.sin(u) * du, 2 * np.cos(u) * du, dv), rtol=1e-6
    )
    differential = np.asarray(jax.jacfwd(patch.evaluate)(point))
    np.testing.assert_allclose(differential.T @ differential, np.diag((4, 1)), atol=1e-6)

"""Consumer-visible typing for the source-bound T-spline IGA path."""

from fractions import Fraction
from typing import assert_type

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import iga
from phydrax.geometry.brep import TSplineSurfacePatch


def tspline_plan(
    mesh: iga.TMesh2D,
    control_points: ArrayLike,
    raw_weights: ArrayLike,
    query_points: ArrayLike,
) -> None:
    anchor_ids = tuple(anchor.anchor_id for anchor in mesh.anchors)
    source = iga.TSplineGeometryState(mesh, control_points, raw_weights, anchor_ids)
    carrier: TSplineSurfacePatch = source.surface_patch
    assert_type(carrier.basis_certificate, iga.ASTSCertificate)
    assert_type(carrier.is_c1_along(0, 0.5, 0.1, 0.9), bool)
    geometry = iga.TSplineGeometryState(carrier)
    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    field = iga.TSplineH1FieldSpec(
        "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
    )
    plan = iga.TSplineIsogeometricPlan(
        geometry,
        field,
        trim=trim,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(4),
    )
    prepared = plan.prepare(numeric_version="typing-fixture")
    route = prepared.prepare_query_route(query_points)
    assert_type(route, iga.TSplineQueryRoute)
    geometry_jet = prepared.evaluate_geometry_route(route, maximum_order=2)
    assert_type(geometry_jet, iga.TSplineRationalJet)
    coefficients = jnp.ones((mesh.coefficient_count,))
    field_values = prepared.evaluate_field_route("u", route, coefficients)
    assert_type(field_values, jax.Array)

    refinement = plan.refine((mesh.cells[0].cell_id,))
    assert_type(refinement, iga.TSplineIsogeometricRefinement)
    prolonged_values, prolonged_weights = refinement.prolong_field("u", coefficients)
    assert_type(prolonged_values, jax.Array)
    assert_type(prolonged_weights, jax.Array | None)
    exact_coefficients: tuple[tuple[Fraction, ...], ...] = tuple(
        (Fraction(1),) for _ in mesh.anchors
    )
    exact_values, exact_weights = refinement.prolong_exact_field("u", exact_coefficients)
    assert_type(exact_values, tuple[tuple[Fraction, ...], ...])
    assert_type(exact_weights, tuple[Fraction, ...] | None)

    runtime = prepared.default_runtime
    runtime_bank = runtime.realize_parameter_bank(
        control_points,
        raw_weights,
        source_authority_id=runtime.source_authority_id,
    )
    assert_type(runtime_bank, tuple[jax.Array, jax.Array])


def cad_carrier_constructor(
    mesh: iga.TMesh2D,
    control_points: ArrayLike,
    raw_weights: ArrayLike,
) -> None:
    carrier = phx.geometry.brep.TSplineSurfacePatch(
        mesh,
        control_points,
        raw_weights,
    )
    assert_type(carrier, TSplineSurfacePatch)

"""Projection reconstruction requires a compiled discovery geometry."""

from typing import assert_type

from phydrax.geometry import CompiledGeometry, ImplicitPointProjectionPlan


def projection_reconstruction(
    plan: ImplicitPointProjectionPlan, geometry: CompiledGeometry
) -> None:
    assert_type(
        ImplicitPointProjectionPlan.from_stored_fields(plan, geometry=geometry),
        ImplicitPointProjectionPlan,
    )
    ImplicitPointProjectionPlan.from_stored_fields(plan)  # ty: ignore[missing-argument]
    ImplicitPointProjectionPlan.from_stored_fields(plan, geometry=plan)  # ty: ignore[invalid-argument-type]

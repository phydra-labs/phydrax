"""Projected form kernels carry explicit scientific output types."""

from jax import Array
from typing_extensions import assert_type

from phydrax.exterior import FormType
from phydrax.kernels import (
    AbstractPositiveDefiniteKernel,
    ProjectedDifferentialFormKernel,
)


def projected_form_kernel(
    scalar: AbstractPositiveDefiniteKernel, projector_point: Array
) -> None:
    def projector(point: Array) -> Array:
        return point[:, None] * point[None, :]

    kernel = ProjectedDifferentialFormKernel(
        scalar,
        projector,
        FormType(2, 1, ambient_dimension=3),
        projector_id="rank-one",
        projector_derivative_order=2,
    )
    assert_type(kernel.form_type, FormType)
    assert_type(kernel.block(projector_point, projector_point), Array)
    assert_type(kernel.output_dimension, int)
    ProjectedDifferentialFormKernel(
        scalar,
        projector,
        3,  # ty: ignore[invalid-argument-type]
        projector_id="rank-one",
    )

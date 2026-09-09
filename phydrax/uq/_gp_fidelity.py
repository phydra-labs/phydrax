#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.kernels import AbstractPositiveDefiniteKernel

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..fidelity import FidelityDataset, FidelityPath
from ._gp_multioutput import (
    AbstractMultiOutputKernel,
    MultiOutputDesign,
    MultiOutputGaussianProcessCondition,
    MultiOutputGaussianProcessDiscrepancy,
    MultiOutputGaussianProcessLikelihoodState,
)


class AutoregressiveFidelityKernel(AbstractMultiOutputKernel):
    """Positive-definite autoregressive covariance along one fidelity path."""

    path: FidelityPath
    spatial_kernels: tuple[AbstractPositiveDefiniteKernel, ...]
    transfer_coefficients: Array

    def __init__(
        self,
        path: FidelityPath,
        spatial_kernels: Sequence[AbstractPositiveDefiniteKernel],
        /,
        *,
        transfer_coefficients: ArrayLike,
    ):
        if not isinstance(path, FidelityPath):
            raise TypeError("path must be a FidelityPath.")
        kernels = tuple(spatial_kernels)
        if len(kernels) != path.num_levels or not all(
            isinstance(kernel, AbstractPositiveDefiniteKernel) for kernel in kernels
        ):
            raise ValueError("Provide one positive-definite spatial kernel per level.")
        coefficients = jnp.asarray(transfer_coefficients, dtype=float)
        if coefficients.shape != (path.num_levels - 1,):
            raise ValueError(
                "transfer_coefficients must contain one value per fidelity relation."
            )
        self.path = path
        self.spatial_kernels = kernels
        self.transfer_coefficients = eqx.error_if(
            coefficients,
            jnp.any(~jnp.isfinite(coefficients)),
            "Fidelity transfer coefficients must be finite.",
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.path.level_ids

    @property
    def mixing_matrix(self) -> Array:
        count = self.path.num_levels
        rows = []
        for output in range(count):
            values = []
            for latent in range(count):
                if latent > output:
                    values.append(
                        jnp.asarray(0.0, dtype=self.transfer_coefficients.dtype)
                    )
                elif latent == output:
                    values.append(
                        jnp.asarray(1.0, dtype=self.transfer_coefficients.dtype)
                    )
                else:
                    values.append(jnp.prod(self.transfer_coefficients[latent:output]))
            rows.append(jnp.stack(values))
        return jnp.stack(rows)

    def matrix(
        self,
        left: MultiOutputDesign,
        right: MultiOutputDesign,
        /,
    ) -> Array:
        self._validate_design_pair(left, right)
        mixing = self.mixing_matrix
        result = jnp.zeros(
            (left.num_observations, right.num_observations),
            dtype=jnp.result_type(left.points, right.points, self.transfer_coefficients),
        )
        for latent, kernel in enumerate(self.spatial_kernels):
            left_scale = mixing[left.output_index, latent]
            right_scale = mixing[right.output_index, latent]
            result = result + kernel.matrix(left.points, right.points) * (
                left_scale[:, None] * right_scale[None, :]
            )
        return result

    def diagonal(self, design: MultiOutputDesign, /) -> Array:
        self._validate_design(design)
        mixing = self.mixing_matrix
        result = jnp.zeros(
            (design.num_observations,),
            dtype=jnp.result_type(design.points, self.transfer_coefficients),
        )
        for latent, kernel in enumerate(self.spatial_kernels):
            scale = mixing[design.output_index, latent]
            result = result + kernel.diagonal(design.points) * scale * scale
        return result

    @property
    def max_derivative_order(self) -> int | None:
        finite = tuple(
            order
            for kernel in self.spatial_kernels
            if (order := kernel.max_derivative_order) is not None
        )
        return None if not finite else min(finite)

    @property
    def kernel_id(self) -> str:
        kernels = ",".join(kernel.kernel_id for kernel in self.spatial_kernels)
        return f"AutoregressiveFidelityKernel[{self.path.path_id};{kernels}]"

    def _validate_design(self, design: MultiOutputDesign, /) -> None:
        if not isinstance(design, MultiOutputDesign):
            raise TypeError("Expected a MultiOutputDesign.")
        if design.output_names != self.output_names:
            raise ValueError("Fidelity design levels do not match the kernel path.")

    def _validate_design_pair(
        self,
        left: MultiOutputDesign,
        right: MultiOutputDesign,
        /,
    ) -> None:
        self._validate_design(left)
        self._validate_design(right)
        if left.points.shape[1] != right.points.shape[1]:
            raise ValueError("Fidelity designs require equal coordinate dimensions.")


class FidelityGaussianProcess(StrictModule):
    """Target-aware exact GP over sparse heterogeneous fidelity observations."""

    path: FidelityPath
    discrepancy: MultiOutputGaussianProcessDiscrepancy
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: FidelityPath,
        dataset: FidelityDataset,
        /,
    ):
        if not isinstance(path, FidelityPath):
            raise TypeError("path must be a FidelityPath.")
        if not isinstance(dataset, FidelityDataset):
            raise TypeError("dataset must be a FidelityDataset.")
        if (
            dataset.hierarchy.hierarchy_id != path.hierarchy_id
            or dataset.hierarchy.fingerprint != path.hierarchy_fingerprint
        ):
            raise ValueError("Fidelity dataset and path belong to different hierarchies.")
        design, observations = fidelity_design(dataset, path)
        self.path = path
        self.discrepancy = MultiOutputGaussianProcessDiscrepancy(
            design,
            observations,
        )
        self.dataset_id = dataset.dataset_id

    @property
    def design(self) -> MultiOutputDesign:
        return self.discrepancy.design

    @property
    def observations(self) -> Array:
        return self.discrepancy.observations

    def log_marginal_likelihood(
        self,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
    ) -> Array:
        self.validate_state(state)
        return self.discrepancy.log_marginal_likelihood(
            jnp.zeros_like(self.observations),
            state=state,
        )

    def condition_level(
        self,
        points: ArrayLike,
        level_id: str,
        /,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
    ) -> MultiOutputGaussianProcessCondition:
        self.validate_state(state)
        level = str(level_id)
        if level not in self.path.level_ids:
            raise ValueError(f"Fidelity level {level!r} is not on the GP path.")
        point_array = _as_points(points)
        level_index = self.path.level_ids.index(level)
        query = MultiOutputDesign(
            point_array,
            jnp.full((point_array.shape[0],), level_index, dtype=jnp.int32),
            output_names=self.path.level_ids,
        )
        return self.discrepancy.condition(
            jnp.zeros_like(self.observations),
            query,
            state=state,
        )

    def condition_target(
        self,
        points: ArrayLike,
        /,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
    ) -> MultiOutputGaussianProcessCondition:
        return self.condition_level(
            points,
            self.path.target.level_id,
            state=state,
        )

    def condition_target_result(
        self,
        points: ArrayLike,
        /,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
    ) -> FidelityGaussianProcessResult:
        return FidelityGaussianProcessResult(
            state=state,
            condition=self.condition_target(points, state=state),
            path_id=self.path.path_id,
            dataset_id=self.dataset_id,
            target_level_id=self.path.target.level_id,
        )

    def validate_state(
        self,
        state: MultiOutputGaussianProcessLikelihoodState,
        /,
    ) -> None:
        if not isinstance(state, MultiOutputGaussianProcessLikelihoodState):
            raise TypeError("state must be a MultiOutputGaussianProcessLikelihoodState.")
        if state.kernel.output_names != self.path.level_ids:
            raise ValueError("GP state fidelity levels do not match the model path.")


class FidelityGaussianProcessResult(StrictModule, NonTrainableState):
    """Portable target-level GP state and conditioned prediction."""

    state: MultiOutputGaussianProcessLikelihoodState
    condition: MultiOutputGaussianProcessCondition
    path_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    target_level_id: str = eqx.field(static=True)


def fidelity_design(
    dataset: FidelityDataset,
    path: FidelityPath,
    /,
) -> tuple[MultiOutputDesign, Array]:
    """Flatten active scalar fidelity observations into a heterotopic GP design."""

    if not isinstance(dataset, FidelityDataset) or not isinstance(path, FidelityPath):
        raise TypeError(
            "dataset and path must be FidelityDataset and FidelityPath values."
        )
    if (
        dataset.hierarchy.hierarchy_id != path.hierarchy_id
        or dataset.hierarchy.fingerprint != path.hierarchy_fingerprint
    ):
        raise ValueError("Fidelity dataset and path belong to different hierarchies.")
    cases = {case.case_id: (index, case) for index, case in enumerate(dataset.cases)}
    rows: list[Array] = []
    outputs: list[int] = []
    sources: list[int] = []
    observations: list[Array] = []
    for evaluation in dataset.evaluations:
        if evaluation.level_id not in path.level_ids or not bool(
            np.asarray(evaluation.valid)
        ):
            continue
        case_index, case = cases[evaluation.case_id]
        input_leaves = tuple(jax.tree_util.tree_leaves(case.inputs))
        if len(input_leaves) != 1:
            raise ValueError(
                "Fidelity GP cases require exactly one finite array input leaf."
            )
        point = jnp.ravel(jnp.asarray(input_leaves[0], dtype=float))
        observable_leaves = tuple(jax.tree_util.tree_leaves(evaluation.observable))
        if len(observable_leaves) != 1 or jnp.asarray(observable_leaves[0]).size != 1:
            raise ValueError("Fidelity GP observations must be scalar.")
        rows.append(point)
        outputs.append(path.level_ids.index(evaluation.level_id))
        sources.append(case_index)
        observations.append(jnp.ravel(jnp.asarray(observable_leaves[0], dtype=float))[0])
    if not rows:
        raise ValueError("Fidelity GP requires at least one active path observation.")
    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise ValueError("Fidelity GP case inputs must share one coordinate shape.")
    design = MultiOutputDesign(
        jnp.stack(rows),
        jnp.asarray(outputs, dtype=jnp.int32),
        output_names=path.level_ids,
        source_index=jnp.asarray(sources, dtype=jnp.int32),
    )
    return design, jnp.stack(observations)


def _as_points(value: ArrayLike, /) -> Array:
    points = jnp.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("Fidelity GP points must have shape (point, coordinate).")
    if not bool(jnp.all(jnp.isfinite(points))):
        raise ValueError("Fidelity GP points must be finite.")
    return points


__all__ = [
    "AutoregressiveFidelityKernel",
    "FidelityGaussianProcess",
    "FidelityGaussianProcessResult",
    "fidelity_design",
]

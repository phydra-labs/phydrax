#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import ChartTransition, CoordinateChart
from ._complex import AlmostComplexStructure, holomorphicity_residual


class CoordinateAtlas(StrictModule):
    """A fixed chart collection and explicit directed transition graph."""

    charts: tuple[CoordinateChart, ...]
    transitions: tuple[ChartTransition, ...]
    transition_indices: tuple[tuple[int, int], ...]

    def __init__(
        self,
        charts: Sequence[CoordinateChart],
        transitions: Sequence[ChartTransition],
        /,
    ):
        charts_ = tuple(charts)
        if not charts_ or any(
            not isinstance(chart, CoordinateChart) for chart in charts_
        ):
            raise TypeError("CoordinateAtlas requires CoordinateChart objects.")
        identities = tuple((chart.name, chart.coordinates) for chart in charts_)
        if len(set(identities)) != len(identities):
            raise ValueError("CoordinateAtlas chart identities must be unique.")
        transitions_ = tuple(transitions)
        indices = []
        for transition in transitions_:
            if not isinstance(transition, ChartTransition):
                raise TypeError("Atlas transitions must be ChartTransition objects.")
            source = self._chart_index(charts_, transition.source)
            target = self._chart_index(charts_, transition.target)
            if (source, target) in indices:
                raise ValueError(
                    "CoordinateAtlas transitions must be unique by direction."
                )
            indices.append((source, target))
        self.charts = charts_
        self.transitions = transitions_
        self.transition_indices = tuple(indices)

    @staticmethod
    def _chart_index(
        charts: tuple[CoordinateChart, ...],
        chart: CoordinateChart,
        /,
    ) -> int:
        matches = tuple(
            index
            for index, candidate in enumerate(charts)
            if candidate.compatible_with(chart)
        )
        if len(matches) != 1:
            raise ValueError("Atlas transition chart is not uniquely owned by the atlas.")
        return matches[0]

    def transition(self, source: int, target: int, /) -> ChartTransition:
        source_ = int(source)
        target_ = int(target)
        if source_ == target_:
            return ChartTransition.identity(self.charts[source_])
        for indices, transition in zip(
            self.transition_indices, self.transitions, strict=True
        ):
            if indices == (source_, target_):
                return transition
        path = self.transition_path(source_, target_)
        result = path[0]
        for transition in path[1:]:
            result = result.compose(transition)
        return result

    def transition_path(
        self,
        source: int,
        target: int,
        /,
    ) -> tuple[ChartTransition, ...]:
        source_ = int(source)
        target_ = int(target)
        if not (0 <= source_ < len(self.charts) and 0 <= target_ < len(self.charts)):
            raise ValueError("Atlas chart index is out of range.")
        if source_ == target_:
            return (ChartTransition.identity(self.charts[source_]),)
        frontier: list[tuple[int, tuple[ChartTransition, ...]]] = [(source_, ())]
        visited = {source_}
        while frontier:
            node, path = frontier.pop(0)
            for indices, transition in zip(
                self.transition_indices, self.transitions, strict=True
            ):
                if indices[0] != node or indices[1] in visited:
                    continue
                next_path = path + (transition,)
                if indices[1] == target_:
                    return next_path
                visited.add(indices[1])
                frontier.append((indices[1], next_path))
        raise ValueError(
            f"No atlas transition path connects chart {source_} to {target_}."
        )


class AtlasValidationReport(StrictModule):
    valid: Array
    finite: Array
    maximum_inverse_residual: Array
    maximum_jacobian_inverse_residual: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        finite: ArrayLike,
        maximum_inverse_residual: ArrayLike,
        maximum_jacobian_inverse_residual: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.maximum_inverse_residual = jnp.asarray(maximum_inverse_residual)
        self.maximum_jacobian_inverse_residual = jnp.asarray(
            maximum_jacobian_inverse_residual
        )


def validate_coordinate_atlas(
    atlas: CoordinateAtlas,
    transition_points: Sequence[ArrayLike],
    /,
    *,
    inverse_tolerance: float = 1e-8,
    jacobian_tolerance: float = 1e-8,
    raise_on_error: bool = True,
) -> AtlasValidationReport:
    """Validate supplied inverse transitions at representative overlap points."""
    if not isinstance(atlas, CoordinateAtlas):
        raise TypeError("atlas must be a CoordinateAtlas.")
    points = tuple(transition_points)
    if len(points) != len(atlas.transitions):
        raise ValueError("One overlap-point design is required per atlas transition.")
    inverse_residuals = []
    jacobian_residuals = []
    finite_values = []
    for transition, values in zip(atlas.transitions, points, strict=True):
        if transition.inverse_function is None:
            raise ValueError("Atlas validation requires explicit transition inverses.")
        source = jnp.asarray(values)
        mapped = transition(source)
        reconstructed = transition.inverse(mapped)
        inverse_residuals.append(jnp.max(jnp.abs(reconstructed - source)))
        jacobian = transition.jacobian(source)
        inverse_jacobian = transition.inverse_jacobian(mapped)
        identity = jnp.eye(transition.source.dimension, dtype=jacobian.dtype)
        jacobian_residuals.append(
            jnp.max(jnp.abs(inverse_jacobian @ jacobian - identity))
        )
        finite_values.append(
            jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(mapped))
            & jnp.all(jnp.isfinite(jacobian))
        )
    maximum_inverse = jnp.max(jnp.stack(inverse_residuals), initial=0.0)
    maximum_jacobian = jnp.max(jnp.stack(jacobian_residuals), initial=0.0)
    finite = jnp.all(jnp.stack(finite_values))
    valid = (
        finite
        & (maximum_inverse <= inverse_tolerance)
        & (maximum_jacobian <= jacobian_tolerance)
    )
    report = AtlasValidationReport(
        valid=valid,
        finite=finite,
        maximum_inverse_residual=maximum_inverse,
        maximum_jacobian_inverse_residual=maximum_jacobian,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Coordinate-atlas validation failed: "
            f"inverse_residual={float(jax.device_get(maximum_inverse))}, "
            f"jacobian_residual={float(jax.device_get(maximum_jacobian))}."
        )
    return report


class PatchwiseScalarField(StrictModule):
    """Scalar local representatives over every chart in one atlas."""

    atlas: CoordinateAtlas
    local_fields: tuple[Callable[[Array], Array], ...]

    def __init__(
        self,
        atlas: CoordinateAtlas,
        local_fields: Sequence[Callable[[Array], Array]],
        /,
    ):
        if not isinstance(atlas, CoordinateAtlas):
            raise TypeError("PatchwiseScalarField requires a CoordinateAtlas.")
        fields = tuple(local_fields)
        if len(fields) != len(atlas.charts) or any(
            not callable(field) for field in fields
        ):
            raise ValueError(
                "One callable scalar field is required for every atlas chart."
            )
        self.atlas = atlas
        self.local_fields = fields

    def transition_residual(
        self,
        source: int,
        target: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        transition = self.atlas.transition(source, target)
        source_value = jnp.asarray(self.local_fields[int(source)](coordinates))
        target_value = jnp.asarray(
            self.local_fields[int(target)](transition(coordinates))
        )
        return jnp.max(jnp.abs(source_value - target_value))


class ComplexAtlasStructure(StrictModule):
    """Almost-complex local structures over a fixed coordinate atlas."""

    atlas: CoordinateAtlas
    local_structures: tuple[AlmostComplexStructure, ...]

    def __init__(
        self,
        atlas: CoordinateAtlas,
        local_structures: Sequence[AlmostComplexStructure],
        /,
    ):
        if not isinstance(atlas, CoordinateAtlas):
            raise TypeError("ComplexAtlasStructure requires a CoordinateAtlas.")
        structures = tuple(local_structures)
        if len(structures) != len(atlas.charts):
            raise ValueError("One almost-complex structure is required per atlas chart.")
        for chart, structure in zip(atlas.charts, structures, strict=True):
            if not isinstance(structure, AlmostComplexStructure):
                raise TypeError(
                    "Atlas complex structures must be AlmostComplexStructure objects."
                )
            if not chart.compatible_with(structure.chart):
                raise ValueError("Atlas chart and local complex structure must match.")
        self.atlas = atlas
        self.local_structures = structures

    def transition_residual(
        self,
        source: int,
        target: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        transition = self.atlas.transition(source, target)
        return holomorphicity_residual(
            transition,
            self.local_structures[int(source)],
            self.local_structures[int(target)],
            coordinates,
        )


__all__ = [
    "AtlasValidationReport",
    "ComplexAtlasStructure",
    "CoordinateAtlas",
    "PatchwiseScalarField",
    "validate_coordinate_atlas",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._axis import TensorGridPlan, UniformCellAxisSpec
from ...discretization.finite_volume._mapped import (
    MappedFiniteVolumeDiscretization,
    MappedFiniteVolumePlan,
    MappedPeriodicSeamPlan,
)
from ...discretization.finite_volume._structured import FiniteVolumePlan
from ...nonlinear import Bisection, NonlinearTermination, scalar_root, ScalarRootProblem
from ...qualification._reference import ReferenceArtifactManifest


def _static_airfoil_o_grid_map(
    point: Array,
    /,
    *,
    coordinates: tuple[tuple[float, float], ...],
    center: tuple[float, float],
    radius: float,
) -> Array:
    reference = jnp.asarray(point)
    parameter = jnp.mod(reference[0], 1.0)
    radial = reference[1]
    points = jnp.asarray(coordinates, dtype=reference.dtype)
    count = len(coordinates)
    scaled = parameter * count
    index = jnp.floor(scaled).astype(jnp.int32) % count
    fraction = scaled - jnp.floor(scaled)
    inner = points[index] + fraction * (points[(index + 1) % count] - points[index])
    angle = -2.0 * jnp.pi * parameter
    outer = jnp.asarray(center, dtype=reference.dtype) + radius * jnp.stack(
        (jnp.cos(angle), jnp.sin(angle))
    )
    return (1.0 - radial) * inner + radial * outer


class AirfoilSectionPlan(StrictModule, NonTrainableState):
    """Closed, clockwise piecewise-linear airfoil section."""

    coordinates: Array
    source_manifest: ReferenceArtifactManifest | None
    chord: float = eqx.field(static=True)
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        /,
        *,
        source_manifest: ReferenceArtifactManifest | None = None,
    ):
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 8:
            raise ValueError("Airfoil section requires at least eight planar points.")
        if np.allclose(points[0], points[-1], rtol=0.0, atol=1.0e-14):
            points = points[:-1]
        segments = np.roll(points, -1, axis=0) - points
        signed_area = 0.5 * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
        chord = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        if (
            np.any(~np.isfinite(points))
            or np.any(np.linalg.norm(segments, axis=-1) <= 1.0e-14)
            or not np.isfinite(signed_area)
            or abs(signed_area) <= 1.0e-14
            or not np.isfinite(chord)
            or chord <= 0.0
            or (
                source_manifest is not None
                and not isinstance(source_manifest, ReferenceArtifactManifest)
            )
        ):
            raise ValueError("Airfoil coordinates or source manifest are invalid.")
        if signed_area > 0.0:
            points = np.concatenate((points[:1], points[:0:-1]), axis=0)
        value = jnp.asarray(points)
        self.coordinates = value
        self.source_manifest = source_manifest
        self.chord = chord
        self.section_id = canonical_fingerprint(
            {
                "kind": "airfoil-section",
                "coordinates": array_tree_fingerprint(value),
                "source_manifest": None
                if source_manifest is None
                else source_manifest.manifest_id,
            }
        )

    def surface(self, parameter: ArrayLike, /) -> Array:
        value = jnp.mod(jnp.asarray(parameter), 1.0)
        count = self.coordinates.shape[0]
        scaled = value * count
        index = jnp.floor(scaled).astype(jnp.int32) % count
        fraction = scaled - jnp.floor(scaled)
        start = self.coordinates[index]
        end = self.coordinates[(index + 1) % count]
        return start + fraction[..., None] * (end - start)

    def closest_distance(self, points: ArrayLike, /) -> Array:
        value = jnp.asarray(points)
        if value.shape[-1] != 2:
            raise ValueError("Airfoil closest points must be planar.")
        start = self.coordinates
        segment = jnp.roll(start, -1, axis=0) - start
        displacement = value[..., None, :] - start
        parameter = jnp.sum(displacement * segment, axis=-1) / jnp.sum(
            segment * segment, axis=-1
        )
        projection = start + jnp.clip(parameter, 0.0, 1.0)[..., None] * segment
        difference = value[..., None, :] - projection
        return jnp.min(jnp.sqrt(jnp.sum(difference * difference, axis=-1)), axis=-1)


class PreparedAirfoilOGrid(StrictModule, NonTrainableState):
    discretization: MappedFiniteVolumeDiscretization
    wall_distance: Array
    minimum_cell_volume: Array
    finite: Array
    successful: Array
    grid_id: str = eqx.field(static=True)


class AirfoilOGridPlan(StrictModule, NonTrainableState):
    """Static O-grid map from one airfoil loop to a circular farfield."""

    section: AirfoilSectionPlan
    center: Array
    circumferential_cells: int = eqx.field(static=True)
    radial_cells: int = eqx.field(static=True)
    farfield_radius: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        section: AirfoilSectionPlan,
        circumferential_cells: int,
        radial_cells: int,
        farfield_radius: float,
        /,
        *,
        center: ArrayLike | None = None,
    ):
        circumferential = int(circumferential_cells)
        radial = int(radial_cells)
        radius = float(farfield_radius)
        center_ = (
            jnp.mean(section.coordinates, axis=0)
            if center is None
            else jnp.asarray(center)
        )
        maximum_radius = float(
            np.max(
                np.linalg.norm(
                    np.asarray(section.coordinates) - np.asarray(center_), axis=-1
                )
            )
        )
        if (
            not isinstance(section, AirfoilSectionPlan)
            or circumferential < section.coordinates.shape[0]
            or radial < 2
            or not np.isfinite(radius)
            or radius <= 1.5 * maximum_radius
            or center_.shape != (2,)
        ):
            raise ValueError("Airfoil O-grid topology or farfield is invalid.")
        self.section = section
        self.circumferential_cells = circumferential
        self.radial_cells = radial
        self.farfield_radius = radius
        self.center = center_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "airfoil-o-grid",
                "section": section.section_id,
                "circumferential_cells": circumferential,
                "radial_cells": radial,
                "farfield_radius": radius,
                "center": array_tree_fingerprint(center_),
            }
        )

    def coordinate_map(self, point: Array, /) -> Array:
        reference = jnp.asarray(point)
        parameter = reference[0]
        radial = reference[1]
        inner = self.section.surface(parameter)
        angle = -2.0 * jnp.pi * parameter
        outer = self.center + self.farfield_radius * jnp.stack(
            (jnp.cos(angle), jnp.sin(angle))
        )
        return (1.0 - radial) * inner + radial * outer

    def prepare(self, component_names: Sequence[str], /) -> PreparedAirfoilOGrid:
        names = tuple(str(name) for name in component_names)
        if not names or any(not name for name in names):
            raise ValueError("Airfoil O-grid component names are required.")
        grid = TensorGridPlan(
            (
                UniformCellAxisSpec(self.circumferential_cells, periodic=True),
                UniformCellAxisSpec(self.radial_cells),
            ),
            axis_names=("surface", "normal"),
        ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
        reference = FiniteVolumePlan(grid, component_names=names).prepare()
        seam = MappedPeriodicSeamPlan(0, jnp.eye(2), jnp.zeros((2,)))
        coordinate_map = partial(
            _static_airfoil_o_grid_map,
            coordinates=tuple(
                tuple(float(component) for component in point)
                for point in np.asarray(self.section.coordinates)
            ),
            center=tuple(float(component) for component in np.asarray(self.center)),
            radius=self.farfield_radius,
        )
        discretization = MappedFiniteVolumePlan(
            reference,
            coordinate_map,
            mapping_id=self.plan_id,
            periodic_seams=(seam,),
        ).prepare()
        wall_distance = self.section.closest_distance(discretization.cell_centers)
        minimum_volume = jnp.min(discretization.cell_volumes)
        finite = (
            jnp.all(jnp.isfinite(discretization.cell_centers))
            & jnp.all(jnp.isfinite(discretization.cell_volumes))
            & jnp.all(jnp.isfinite(wall_distance))
        )
        successful = finite & (minimum_volume > 0.0) & jnp.all(wall_distance > 0.0)
        return PreparedAirfoilOGrid(
            discretization,
            wall_distance,
            minimum_volume,
            finite,
            successful,
            canonical_fingerprint(
                {
                    "kind": "prepared-airfoil-o-grid",
                    "plan": self.plan_id,
                    "geometry": discretization.prepared_id,
                }
            ),
        )


class RAE2822QualificationEvidence(StrictModule):
    lift_coefficient: Array
    target_lift_coefficient: Array
    lift_residual: Array
    tolerance: Array
    finite: Array
    successful: Array
    case_id: str = eqx.field(static=True)


class RAE2822CasePlan(StrictModule, NonTrainableState):
    """Exact-artifact transonic RAE2822 fixed-lift case contract."""

    reference_manifest: ReferenceArtifactManifest
    mach: float = eqx.field(static=True)
    reynolds_number: float = eqx.field(static=True)
    target_lift_coefficient: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_manifest: ReferenceArtifactManifest,
        /,
        *,
        mach: float,
        reynolds_number: float,
        target_lift_coefficient: float,
        reference_temperature: float,
    ):
        values = tuple(
            float(value)
            for value in (
                mach,
                reynolds_number,
                target_lift_coefficient,
                reference_temperature,
            )
        )
        if (
            not isinstance(reference_manifest, ReferenceArtifactManifest)
            or not reference_manifest.commercial_use_permitted
            or any(not np.isfinite(value) for value in values)
            or not 0.6 < values[0] < 1.0
            or values[1] <= 0.0
            or values[3] <= 0.0
        ):
            raise ValueError("RAE2822 case conditions or reference rights are invalid.")
        self.reference_manifest = reference_manifest
        self.mach = values[0]
        self.reynolds_number = values[1]
        self.target_lift_coefficient = values[2]
        self.reference_temperature = values[3]
        self.case_id = canonical_fingerprint(
            {
                "kind": "rae2822-fixed-lift-case",
                "reference": reference_manifest.manifest_id,
                "mach": values[0],
                "reynolds_number": values[1],
                "target_lift_coefficient": values[2],
                "reference_temperature": values[3],
            }
        )

    def qualify_lift(
        self, lift_coefficient: ArrayLike, /, *, tolerance: float
    ) -> RAE2822QualificationEvidence:
        lift = jnp.asarray(lift_coefficient)
        tolerance_ = jnp.asarray(tolerance, dtype=lift.dtype)
        residual = lift - self.target_lift_coefficient
        finite = jnp.isfinite(lift) & jnp.isfinite(tolerance_) & (tolerance_ > 0.0)
        return RAE2822QualificationEvidence(
            lift,
            jnp.asarray(self.target_lift_coefficient, dtype=lift.dtype),
            residual,
            tolerance_,
            finite,
            finite & (jnp.abs(residual) <= tolerance_),
            self.case_id,
        )


class TransonicFixedLiftResult(StrictModule):
    angle_of_attack: Array
    lift_coefficient: Array
    residual: Array
    lower_angle: Array
    upper_angle: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class TransonicFixedLiftPlan(StrictModule, NonTrainableState):
    """Bracketed fixed-lift solve around one deterministic flow evaluator."""

    target_lift_coefficient: float = eqx.field(static=True)
    lower_angle: float = eqx.field(static=True)
    upper_angle: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_lift_coefficient: float,
        angle_bracket: tuple[float, float],
        /,
        *,
        tolerance: float = 1.0e-8,
        maximum_steps: int = 64,
    ):
        target = float(target_lift_coefficient)
        lower, upper = (float(value) for value in angle_bracket)
        tolerance_ = float(tolerance)
        steps = int(maximum_steps)
        if (
            any(not np.isfinite(value) for value in (target, lower, upper, tolerance_))
            or lower >= upper
            or tolerance_ <= 0.0
            or steps <= 0
        ):
            raise ValueError(
                "Fixed-lift target, bracket, or solver controls are invalid."
            )
        self.target_lift_coefficient = target
        self.lower_angle = lower
        self.upper_angle = upper
        self.tolerance = tolerance_
        self.maximum_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transonic-fixed-lift",
                "target": target,
                "bracket": (lower, upper),
                "tolerance": tolerance_,
                "maximum_steps": steps,
            }
        )

    def solve(
        self,
        lift_evaluator: Callable[[Array, Any], ArrayLike],
        args: Any = None,
        /,
    ) -> TransonicFixedLiftResult:
        if not callable(lift_evaluator):
            raise TypeError("lift_evaluator must be callable.")
        problem = ScalarRootProblem(
            lambda angle, runtime_args: (
                jnp.asarray(lift_evaluator(angle, runtime_args))
                - self.target_lift_coefficient
            ),
            bracket=(self.lower_angle, self.upper_angle),
            problem_id=f"fixed-lift:{self.plan_id}",
        )
        result = scalar_root(
            problem,
            method=Bisection(),
            termination=NonlinearTermination(
                absolute_residual=self.tolerance,
                relative_residual=0.0,
                maximum_steps=self.maximum_steps,
                maximum_evaluations=2 * self.maximum_steps + 4,
                maximum_linear_iterations=1,
            ),
            args=args,
        )
        lift = jnp.asarray(lift_evaluator(result.root, args))
        return TransonicFixedLiftResult(
            result.root,
            lift,
            result.value,
            result.lower,
            result.upper,
            result.successful,
            self.plan_id,
        )


__all__ = [
    "AirfoilOGridPlan",
    "AirfoilSectionPlan",
    "PreparedAirfoilOGrid",
    "RAE2822CasePlan",
    "RAE2822QualificationEvidence",
    "TransonicFixedLiftPlan",
    "TransonicFixedLiftResult",
]

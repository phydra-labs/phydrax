#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._geospatial import GeospatialContract, GeospatialTransform
from ._resource import read_bounded_resource, ResourceLimits, ResourceManifest


class GeodeticDependencyError(RuntimeError):
    """The explicitly optional PROJ transformation runtime is unavailable."""


class CoordinateTransformPlan(StrictModule, NonTrainableState):
    """One exact PROJ pipeline with pinned local runtime resources.

    The plan never asks PROJ to choose an operation from CRS identifiers. The
    caller supplies the complete pipeline, a bounded source-domain envelope,
    and every required local grid name. PROJ network access is disabled before
    construction and remains disabled after execution.
    """

    source: GeospatialContract
    target: GeospatialContract
    pipeline: str = eqx.field(static=True)
    source_bounds: tuple[tuple[float, float], ...] | None = eqx.field(static=True)
    required_grid_names: tuple[str, ...] = eqx.field(static=True)
    maximum_accuracy_m: float | None = eqx.field(static=True)
    expected_pyproj_version: str = eqx.field(static=True)
    expected_proj_version: str = eqx.field(static=True)
    expected_resource_sha256: tuple[tuple[str, str], ...] = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: GeospatialContract,
        target: GeospatialContract,
        pipeline: str,
        /,
        *,
        source_bounds: Sequence[tuple[float, float]] | None = None,
        required_grid_names: Sequence[str] = (),
        expected_pyproj_version: str,
        expected_proj_version: str,
        expected_resource_sha256: Mapping[str, str],
        maximum_accuracy_m: float | None = None,
        maximum_resource_bytes: int = 1 << 30,
    ):
        if not isinstance(source, GeospatialContract) or not isinstance(
            target, GeospatialContract
        ):
            raise TypeError(
                "Coordinate transformation endpoints must be geospatial contracts."
            )
        pipeline_ = str(pipeline).strip()
        if not pipeline_ or "+proj=pipeline" not in pipeline_:
            raise ValueError("An explicit complete PROJ pipeline is required.")
        bounds = None if source_bounds is None else tuple(source_bounds)
        if bounds is not None:
            if len(bounds) not in (2, 3, 4):
                raise ValueError(
                    "Source bounds must describe two, three, or four coordinates."
                )
            normalized: list[tuple[float, float]] = []
            for lower, upper in bounds:
                lower_, upper_ = float(lower), float(upper)
                if not np.isfinite(lower_) or not np.isfinite(upper_) or lower_ > upper_:
                    raise ValueError("Source bounds must be finite ordered intervals.")
                normalized.append((lower_, upper_))
            bounds = tuple(normalized)
        grids = tuple(str(name).strip() for name in required_grid_names)
        if any(not name or Path(name).name != name for name in grids):
            raise ValueError("Required grids must be local base names without traversal.")
        if len(set(grids)) != len(grids):
            raise ValueError("Required grid names must be unique.")
        referenced_grids: list[str] = []
        for encoded in re.findall(r"\+(?:grids|nadgrids|geoidgrids)=([^\s]+)", pipeline_):
            for name in encoded.split(","):
                if name == "null":
                    continue
                if name.startswith("@") or Path(name).name != name:
                    raise ValueError(
                        "PROJ grid references must be required local base names."
                    )
                referenced_grids.append(name)
        if set(referenced_grids) != set(grids):
            raise ValueError(
                "Declared PROJ grid resources must exactly match pipeline references."
            )
        if "proj.db" in grids:
            raise ValueError("proj.db is pinned implicitly and must not be a grid name.")
        if not isinstance(expected_resource_sha256, Mapping):
            raise TypeError("Expected PROJ resource SHA-256 values must be a mapping.")
        pyproj_version = str(expected_pyproj_version).strip()
        proj_version = str(expected_proj_version).strip()
        expected_resources = {
            str(name): str(digest).lower()
            for name, digest in expected_resource_sha256.items()
        }
        expected_names = {"proj.db", *grids}
        if (
            not pyproj_version
            or not proj_version
            or set(expected_resources) != expected_names
            or any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in expected_resources.values()
            )
        ):
            raise ValueError(
                "Expected pyproj/PROJ versions and exact resource SHA-256 values are required."
            )
        accuracy = None if maximum_accuracy_m is None else float(maximum_accuracy_m)
        if accuracy is not None and (not np.isfinite(accuracy) or accuracy <= 0):
            raise ValueError("Maximum transformation accuracy must be positive or None.")
        resource_bytes = int(maximum_resource_bytes)
        if resource_bytes <= 0:
            raise ValueError("Transformation resource budget must be positive.")
        self.source, self.target = source, target
        self.pipeline = pipeline_
        self.source_bounds = bounds
        self.required_grid_names = grids
        self.maximum_accuracy_m = accuracy
        self.expected_pyproj_version = pyproj_version
        self.expected_proj_version = proj_version
        self.expected_resource_sha256 = tuple(sorted(expected_resources.items()))
        self.maximum_resource_bytes = resource_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coordinate-transform-plan",
                "source": source.coordinate_id,
                "target": target.coordinate_id,
                "pipeline": pipeline_,
                "source_bounds": bounds,
                "required_grid_names": grids,
                "maximum_accuracy_m": accuracy,
                "expected_pyproj_version": pyproj_version,
                "expected_proj_version": proj_version,
                "expected_resource_sha256": expected_resources,
                "maximum_resource_bytes": resource_bytes,
            }
        )

    def execute(self, coordinates: ArrayLike) -> CoordinateTransformResult:
        return execute_coordinate_transform(self, coordinates)


class CoordinateTransformResult(StrictModule, NonTrainableState):
    coordinates: Array
    transform: GeospatialTransform
    source_shape: tuple[int, ...] = eqx.field(static=True)
    pyproj_version: str = eqx.field(static=True)
    proj_version: str = eqx.field(static=True)
    reported_accuracy_m: float | None = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        transform: GeospatialTransform,
        /,
        *,
        source_shape: tuple[int, ...],
        pyproj_version: str,
        proj_version: str,
        reported_accuracy_m: float | None,
    ):
        values = np.asarray(coordinates, dtype=float)
        if (
            values.shape != source_shape
            or values.ndim < 2
            or values.shape[-1] not in (2, 3, 4)
        ):
            raise ValueError(
                "Transformed coordinates must preserve source shape and arity."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("Coordinate transformation produced non-finite values.")
        self.coordinates = jnp.asarray(values)
        self.transform = transform
        self.source_shape = tuple(int(value) for value in source_shape)
        self.pyproj_version = str(pyproj_version)
        self.proj_version = str(proj_version)
        self.reported_accuracy_m = reported_accuracy_m
        self.result_id = canonical_fingerprint(
            {
                "kind": "coordinate-transform-result",
                "transform": transform.transform_id,
                "shape": self.source_shape,
                "values": values,
            }
        )


def _runtime() -> tuple[Any, str, str, Path]:
    try:
        pyproj = cast(Any, import_module("pyproj"))
    except ImportError as error:
        raise GeodeticDependencyError(
            "Pinned coordinate transformation requires the optional pyproj runtime."
        ) from error
    pyproj.network.set_network_enabled(active=False)
    if pyproj.network.is_network_enabled():
        raise RuntimeError("PROJ network access could not be disabled.")
    data_directory = Path(pyproj.datadir.get_data_dir()).resolve()
    return pyproj, pyproj.__version__, pyproj.proj_version_str, data_directory


def _resources(
    directory: Path,
    grid_names: tuple[str, ...],
    maximum_bytes: int,
    expected_sha256: Mapping[str, str],
) -> tuple[ResourceManifest, ...]:
    names = ("proj.db", *grid_names)
    manifests: list[ResourceManifest] = []
    remaining = maximum_bytes
    for name in names:
        resource = read_bounded_resource(
            name,
            trusted_root=directory,
            limits=ResourceLimits(remaining, 1, 1, 1, 1),
        )
        manifests.append(resource.manifest)
        if resource.manifest.content_sha256 != expected_sha256[name]:
            raise ValueError(f"PROJ resource {name!r} does not match its pinned SHA-256.")
        remaining -= resource.manifest.size_bytes
        if remaining < 0:
            raise ValueError("PROJ resources exceed the declared byte budget.")
    return tuple(manifests)


def execute_coordinate_transform(
    plan: CoordinateTransformPlan, coordinates: ArrayLike, /
) -> CoordinateTransformResult:
    if not isinstance(plan, CoordinateTransformPlan):
        raise TypeError("execute_coordinate_transform requires CoordinateTransformPlan.")
    values = np.asarray(coordinates, dtype=float)
    if values.ndim < 2 or values.shape[-1] not in (2, 3, 4):
        raise ValueError("Coordinates must have trailing arity two, three, or four.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Source coordinates must be finite.")
    if plan.source_bounds is not None:
        if len(plan.source_bounds) != values.shape[-1]:
            raise ValueError("Source bounds and coordinate arity disagree.")
        for axis, (lower, upper) in enumerate(plan.source_bounds):
            if np.any((values[..., axis] < lower) | (values[..., axis] > upper)):
                raise ValueError(
                    "A source coordinate lies outside the declared operation domain."
                )

    pyproj, pyproj_version, proj_version, directory = _runtime()
    if (
        pyproj_version != plan.expected_pyproj_version
        or proj_version != plan.expected_proj_version
    ):
        raise ValueError("Observed pyproj/PROJ versions do not match the transform plan.")
    resources = _resources(
        directory,
        plan.required_grid_names,
        plan.maximum_resource_bytes,
        dict(plan.expected_resource_sha256),
    )
    transformer = pyproj.Transformer.from_pipeline(plan.pipeline)
    reported = float(transformer.accuracy)
    accuracy = None if reported < 0 else reported
    if plan.maximum_accuracy_m is not None:
        if accuracy is None or accuracy > plan.maximum_accuracy_m:
            raise ValueError(
                "PROJ operation does not meet the declared accuracy requirement."
            )
    flat = values.reshape((-1, values.shape[-1]))
    transformed = transformer.transform(
        *tuple(flat[:, axis] for axis in range(flat.shape[1])), errcheck=True
    )
    output = np.stack(transformed, axis=-1).reshape(values.shape)
    if np.any(~np.isfinite(output)):
        raise ValueError("PROJ transformation produced non-finite coordinates.")
    record = GeospatialTransform(
        "proj-pipeline",
        plan.source.coordinate_id,
        plan.target.coordinate_id,
        parameters={
            "pipeline": plan.pipeline,
            "definition": transformer.definition,
            "description": transformer.description,
            "pyproj_version": pyproj_version,
            "proj_version": proj_version,
            "network_enabled": False,
            "reported_accuracy_m": accuracy,
            "source_bounds": plan.source_bounds,
        },
        resources=resources,
    )
    return CoordinateTransformResult(
        output,
        record,
        source_shape=values.shape,
        pyproj_version=pyproj_version,
        proj_version=proj_version,
        reported_accuracy_m=accuracy,
    )


__all__ = [
    "CoordinateTransformPlan",
    "CoordinateTransformResult",
    "GeodeticDependencyError",
    "execute_coordinate_transform",
]

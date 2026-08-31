#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import sqrt
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._force_density_topology import ForceDensityStructure


def _positions(structure: ForceDensityStructure, value: ArrayLike, /) -> Array:
    positions = jnp.asarray(value)
    expected = (structure.node_count, structure.dimension)
    if positions.shape != expected:
        raise ValueError(f"positions must have shape {expected}; got {positions.shape}.")
    if not jnp.issubdtype(positions.dtype, jnp.inexact) or jnp.iscomplexobj(positions):
        raise TypeError("positions must be a real inexact array.")
    return positions


def _real_array(name: str, value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    return array


class AbstractForceDensityLoadModel(StrictModule):
    """Map physical positions and one parameter PyTree to nodal loads."""

    @property
    @abc.abstractmethod
    def load_model_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def depends_on_positions(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: Any,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def valid(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: Any,
        /,
    ) -> Array:
        raise NotImplementedError


class FixedNodalLoadModel(AbstractForceDensityLoadModel):
    """Fixed global nodal loads supplied as the load parameters."""

    def __init__(self):
        pass

    @property
    def load_model_id(self) -> str:
        return "fixed-nodal-loads"

    @property
    def depends_on_positions(self) -> bool:
        return False

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        _positions(structure, positions)
        loads = _real_array(
            "nodal load parameters",
            parameters,
            (structure.node_count, structure.dimension),
        )
        loads = eqx.error_if(
            loads,
            jnp.any(structure.node_valid[:, None] & ~jnp.isfinite(loads)),
            "Active nodal loads must be finite.",
        )
        return jnp.where(structure.node_valid[:, None], loads, 0.0)

    def valid(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        loads = self.nodal_loads(structure, positions, parameters)
        return jnp.all(jnp.isfinite(loads))


class EdgeLineLoadModel(AbstractForceDensityLoadModel, NonTrainableState):
    """Global vector line loads integrated over current or reference members."""

    measure: Literal["current", "reference"] = eqx.field(static=True)
    reference_lengths: Array | None
    _load_model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        measure: Literal["current", "reference"] = "current",
        reference_lengths: ArrayLike | None = None,
    ):
        if measure not in ("current", "reference"):
            raise ValueError("measure must be 'current' or 'reference'.")
        if measure == "current" and reference_lengths is not None:
            raise ValueError("Current-measure edge loads do not take reference_lengths.")
        reference = None
        if measure == "reference":
            if reference_lengths is None:
                raise ValueError(
                    "Reference-measure edge loads require reference_lengths."
                )
            reference = jnp.asarray(reference_lengths)
            if reference.ndim != 1:
                raise ValueError("reference_lengths must be rank-1.")
            if not jnp.issubdtype(reference.dtype, jnp.inexact) or jnp.iscomplexobj(
                reference
            ):
                raise TypeError("reference_lengths must be a real inexact array.")
            if bool(jnp.any(~jnp.isfinite(reference) | (reference <= 0.0))):
                raise ValueError("reference_lengths must be finite and positive.")
        self.measure = measure
        self.reference_lengths = reference
        self._load_model_id = canonical_fingerprint(
            {
                "kind": "force-density-edge-line-load",
                "measure": measure,
                "reference": (
                    None if reference is None else array_tree_fingerprint(reference)
                ),
            }
        )

    @property
    def load_model_id(self) -> str:
        return self._load_model_id

    @property
    def depends_on_positions(self) -> bool:
        return self.measure == "current"

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        xyz = _positions(structure, positions)
        line_loads = _real_array(
            "edge line-load parameters",
            parameters,
            (structure.member_count, structure.dimension),
        )
        line_loads = eqx.error_if(
            line_loads,
            jnp.any(structure.member_valid[:, None] & ~jnp.isfinite(line_loads)),
            "Active edge line loads must be finite.",
        )
        if self.measure == "current":
            vectors = xyz[structure.receivers] - xyz[structure.senders]
            lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        else:
            if self.reference_lengths is None:
                raise RuntimeError("Reference edge-load lengths are unavailable.")
            if self.reference_lengths.shape != (structure.member_count,):
                raise ValueError(
                    "reference_lengths must match the force-density member count."
                )
            lengths = self.reference_lengths
        total = jnp.where(
            structure.member_valid[:, None],
            line_loads * lengths[:, None],
            0.0,
        )
        nodal = jnp.zeros((structure.node_count, structure.dimension), dtype=total.dtype)
        nodal = nodal.at[structure.senders].add(0.5 * total)
        return nodal.at[structure.receivers].add(0.5 * total)

    def valid(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        xyz = _positions(structure, positions)
        loads = _real_array(
            "edge line-load parameters",
            parameters,
            (structure.member_count, structure.dimension),
        )
        finite = jnp.all((~structure.member_valid[:, None]) | jnp.isfinite(loads))
        if self.measure == "reference":
            return finite
        vectors = xyz[structure.receivers] - xyz[structure.senders]
        squared_lengths = jnp.sum(vectors * vectors, axis=-1)
        return finite & jnp.all((~structure.member_valid) | (squared_lengths > 0.0))


class SurfacePressureLoadModel(AbstractForceDensityLoadModel):
    """Follower pressure integrated on oriented T3 and bilinear Q4 cells."""

    def __init__(self):
        pass

    @property
    def load_model_id(self) -> str:
        return "follower-surface-pressure-t3-q4"

    @property
    def depends_on_positions(self) -> bool:
        return True

    @staticmethod
    def _triangle_forces(points: Array, pressure: Array) -> Array:
        area_vector = 0.5 * jnp.cross(
            points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        )
        force = pressure[:, None] * area_vector / 3.0
        return jnp.concatenate(
            (jnp.repeat(force[:, None, :], 3, axis=1), jnp.zeros_like(force[:, None, :])),
            axis=1,
        )

    @staticmethod
    def _quadrilateral_forces(points: Array, pressure: Array) -> Array:
        coordinate = 1.0 / sqrt(3.0)
        quadrature = jnp.asarray(
            (
                (-coordinate, -coordinate),
                (coordinate, -coordinate),
                (coordinate, coordinate),
                (-coordinate, coordinate),
            ),
            dtype=points.dtype,
        )
        node_coordinates = jnp.asarray(
            ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
            dtype=points.dtype,
        )
        xi = quadrature[:, 0, None]
        eta = quadrature[:, 1, None]
        node_xi = node_coordinates[None, :, 0]
        node_eta = node_coordinates[None, :, 1]
        basis = 0.25 * (1.0 + xi * node_xi) * (1.0 + eta * node_eta)
        derivative_xi = 0.25 * node_xi * (1.0 + eta * node_eta)
        derivative_eta = 0.25 * node_eta * (1.0 + xi * node_xi)
        tangent_xi = oe.contract("qi,cid->cqd", derivative_xi, points)
        tangent_eta = oe.contract("qi,cid->cqd", derivative_eta, points)
        area_vectors = jnp.cross(tangent_xi, tangent_eta)
        return pressure[:, None, None] * oe.contract("qi,cqd->cid", basis, area_vectors)

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        if structure.dimension != 3:
            raise ValueError("Surface pressure requires three-dimensional positions.")
        connectivity = structure.surface_connectivity
        if connectivity is None:
            raise ValueError("Surface pressure requires polygonal surface connectivity.")
        xyz = _positions(structure, positions)
        pressure = _real_array(
            "surface pressure parameters", parameters, (connectivity.cell_count,)
        )
        pressure = eqx.error_if(
            pressure,
            jnp.any(~jnp.isfinite(pressure)),
            "Surface pressures must be finite.",
        )
        indices = jnp.where(connectivity.cell_vertex_valid, connectivity.cell_vertices, 0)
        points = xyz[indices]
        triangle = self._triangle_forces(points, pressure)
        quadrilateral = self._quadrilateral_forces(points, pressure)
        forces = jnp.where(
            (connectivity.cell_kinds == 3)[:, None, None],
            triangle,
            quadrilateral,
        )
        forces = jnp.where(connectivity.cell_vertex_valid[:, :, None], forces, 0.0)
        nodal = jnp.zeros((structure.node_count, 3), dtype=forces.dtype)
        return nodal.at[indices.reshape((-1,))].add(forces.reshape((-1, 3)))

    def valid(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        connectivity = structure.surface_connectivity
        if structure.dimension != 3 or connectivity is None:
            return jnp.asarray(False)
        xyz = _positions(structure, positions)
        pressure = _real_array(
            "surface pressure parameters", parameters, (connectivity.cell_count,)
        )
        indices = jnp.where(connectivity.cell_vertex_valid, connectivity.cell_vertices, 0)
        points = xyz[indices]
        unit_pressure = jnp.ones_like(pressure)
        triangle = self._triangle_forces(points, unit_pressure)
        quadrilateral = self._quadrilateral_forces(points, unit_pressure)
        unit_forces = jnp.where(
            (connectivity.cell_kinds == 3)[:, None, None],
            triangle,
            quadrilateral,
        )
        unit_forces = jnp.where(
            connectivity.cell_vertex_valid[:, :, None], unit_forces, 0.0
        )
        resultants = jnp.sum(unit_forces, axis=1)
        area_squared = jnp.sum(resultants * resultants, axis=-1)
        return jnp.all(jnp.isfinite(pressure)) & jnp.all(area_squared > 0.0)


class CompositeForceDensityLoadModel(AbstractForceDensityLoadModel):
    """Static sum of independently parameterized force-density load models."""

    models: tuple[AbstractForceDensityLoadModel, ...]
    _load_model_id: str = eqx.field(static=True)
    _depends_on_positions: bool = eqx.field(static=True)

    def __init__(self, models: Sequence[AbstractForceDensityLoadModel], /):
        resolved = tuple(models)
        if not resolved or any(
            not isinstance(model, AbstractForceDensityLoadModel) for model in resolved
        ):
            raise TypeError(
                "models must contain at least one AbstractForceDensityLoadModel."
            )
        self.models = resolved
        self._load_model_id = canonical_fingerprint(
            {
                "kind": "composite-force-density-load",
                "children": [model.load_model_id for model in resolved],
            }
        )
        self._depends_on_positions = any(model.depends_on_positions for model in resolved)

    @property
    def load_model_id(self) -> str:
        return self._load_model_id

    @property
    def depends_on_positions(self) -> bool:
        return self._depends_on_positions

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: Any,
        /,
    ) -> Array:
        if not isinstance(parameters, tuple) or len(parameters) != len(self.models):
            raise TypeError(
                "Composite load parameters must be one tuple matching the models."
            )
        loads = jnp.zeros(
            (structure.node_count, structure.dimension),
            dtype=jnp.asarray(positions).dtype,
        )
        for model, child_parameters in zip(self.models, parameters, strict=True):
            loads = loads + model.nodal_loads(structure, positions, child_parameters)
        return loads

    def valid(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: Any,
        /,
    ) -> Array:
        if not isinstance(parameters, tuple) or len(parameters) != len(self.models):
            raise TypeError(
                "Composite load parameters must be one tuple matching the models."
            )
        valid = jnp.asarray(True)
        for model, child_parameters in zip(self.models, parameters, strict=True):
            valid = valid & model.valid(structure, positions, child_parameters)
        return valid


__all__ = [
    "AbstractForceDensityLoadModel",
    "CompositeForceDensityLoadModel",
    "EdgeLineLoadModel",
    "FixedNodalLoadModel",
    "SurfacePressureLoadModel",
]

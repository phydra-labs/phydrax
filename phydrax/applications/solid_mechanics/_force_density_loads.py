#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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


def _surface_points(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
) -> tuple[Any, Array, Array]:
    if structure.dimension != 3:
        raise ValueError("Surface loads require three-dimensional positions.")
    connectivity = structure.surface_connectivity
    if connectivity is None:
        raise ValueError("Surface loads require polygonal surface connectivity.")
    xyz = _positions(structure, positions)
    indices = jnp.where(connectivity.cell_vertex_valid, connectivity.cell_vertices, 0)
    return connectivity, indices, xyz[indices]


def _quadrature_data(points: Array, /) -> tuple[Array, Array, Array]:
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
    return basis, tangent_xi, tangent_eta


def _triangle_area_vectors(points: Array, /) -> Array:
    return 0.5 * jnp.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])


def _surface_regularity(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
) -> tuple[Array, Array]:
    connectivity, _, points = _surface_points(structure, positions)
    triangle_area = _triangle_area_vectors(points)
    triangle_magnitude = jnp.sqrt(jnp.sum(triangle_area * triangle_area, axis=-1))
    _, tangent_xi, tangent_eta = _quadrature_data(points)
    quadrature_area = jnp.cross(tangent_xi, tangent_eta)
    quadrature_magnitude = jnp.sqrt(jnp.sum(quadrature_area * quadrature_area, axis=-1))
    resultant = jnp.sum(quadrature_area, axis=1)
    resultant_magnitude = jnp.sqrt(jnp.sum(resultant * resultant, axis=-1))
    orientation = oe.contract("cqd,cd->cq", quadrature_area, resultant)
    scale = jnp.maximum(
        quadrature_magnitude * resultant_magnitude[:, None],
        jnp.finfo(points.dtype).tiny,
    )
    orientation_cosine = orientation / scale
    quad_margin = jnp.minimum(
        jnp.min(quadrature_magnitude, axis=1),
        jnp.min(orientation_cosine, axis=1),
    )
    margin = jnp.where(connectivity.cell_kinds == 3, triangle_magnitude, quad_margin)
    valid = jnp.all(jnp.isfinite(margin)) & jnp.all(margin > 0.0)
    return valid, jnp.min(margin)


def _pressure_forces(points: Array, cell_kinds: Array, pressure: Array, /) -> Array:
    triangle_area = _triangle_area_vectors(points)
    triangle_force = pressure[:, None] * triangle_area / 3.0
    triangle = jnp.concatenate(
        (
            jnp.repeat(triangle_force[:, None, :], 3, axis=1),
            jnp.zeros_like(triangle_force[:, None, :]),
        ),
        axis=1,
    )
    basis, tangent_xi, tangent_eta = _quadrature_data(points)
    quadrature_area = jnp.cross(tangent_xi, tangent_eta)
    quadrilateral = pressure[:, None, None] * oe.contract(
        "qi,cqd->cid", basis, quadrature_area
    )
    return jnp.where((cell_kinds == 3)[:, None, None], triangle, quadrilateral)


def _traction_forces(points: Array, cell_kinds: Array, traction: Array, /) -> Array:
    triangle_area = _triangle_area_vectors(points)
    triangle_magnitude = jnp.sqrt(jnp.sum(triangle_area * triangle_area, axis=-1))
    triangle_force = traction * triangle_magnitude[:, None] / 3.0
    triangle = jnp.concatenate(
        (
            jnp.repeat(triangle_force[:, None, :], 3, axis=1),
            jnp.zeros_like(triangle_force[:, None, :]),
        ),
        axis=1,
    )
    basis, tangent_xi, tangent_eta = _quadrature_data(points)
    quadrature_area = jnp.cross(tangent_xi, tangent_eta)
    magnitudes = jnp.sqrt(jnp.sum(quadrature_area * quadrature_area, axis=-1))
    weights = oe.contract("qi,cq->ci", basis, magnitudes)
    quadrilateral = weights[:, :, None] * traction[:, None, :]
    return jnp.where((cell_kinds == 3)[:, None, None], triangle, quadrilateral)


def _scatter_surface_forces(
    structure: ForceDensityStructure,
    indices: Array,
    valid_slots: Array,
    forces: Array,
    /,
) -> Array:
    masked = jnp.where(valid_slots[:, :, None], forces, 0.0)
    nodal = jnp.zeros((structure.node_count, 3), dtype=forces.dtype)
    return nodal.at[indices.reshape((-1,))].add(masked.reshape((-1, 3)))


def enclosed_surface_volume(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
) -> Array:
    """Return signed volume enclosed by one oriented closed T3/Q4 surface."""
    connectivity, indices, points = _surface_points(structure, positions)
    first = points[:, 0]
    second = points[:, 1]
    third = points[:, 2]
    first_volume = oe.contract("cd,cd->c", first, jnp.cross(second, third)) / 6.0
    fourth = points[:, 3]
    second_volume = oe.contract("cd,cd->c", first, jnp.cross(third, fourth)) / 6.0
    cell_volume = jnp.where(
        connectivity.cell_kinds == 3, first_volume, first_volume + second_volume
    )
    del indices
    return jnp.sum(cell_volume)


class ForceDensityLoadState(StrictModule):
    """Aggregated nodal loads with named component and geometry evidence."""

    total: Array
    components: tuple[Array, ...]
    component_ids: tuple[str, ...] = eqx.field(static=True)
    valid: Array
    minimum_regularity: Array


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

    def _lengths(self, structure: ForceDensityStructure, positions: Array, /) -> Array:
        if self.measure == "current":
            vectors = positions[structure.receivers] - positions[structure.senders]
            return jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        if self.reference_lengths is None:
            raise RuntimeError("Reference edge-load lengths are unavailable.")
        if self.reference_lengths.shape != (structure.member_count,):
            raise ValueError("reference_lengths must match the member count.")
        return self.reference_lengths

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
        total = jnp.where(
            structure.member_valid[:, None],
            line_loads * self._lengths(structure, xyz)[:, None],
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
        lengths = self._lengths(structure, xyz)
        return finite & jnp.all((~structure.member_valid) | (lengths > 0.0))


class ReferenceMemberSelfWeightModel(AbstractForceDensityLoadModel, NonTrainableState):
    """Member self-weight from reference line mass and one gravity vector."""

    reference_lengths: Array
    gravity: Array
    _load_model_id: str = eqx.field(static=True)

    def __init__(self, reference_lengths: ArrayLike, gravity: ArrayLike, /):
        lengths = jnp.asarray(reference_lengths)
        gravity_ = jnp.asarray(gravity)
        if lengths.ndim != 1 or gravity_.ndim != 1:
            raise ValueError("reference_lengths and gravity must be rank-1.")
        if not jnp.issubdtype(lengths.dtype, jnp.inexact) or jnp.iscomplexobj(lengths):
            raise TypeError("reference_lengths must be real inexact values.")
        if gravity_.dtype != lengths.dtype or jnp.iscomplexobj(gravity_):
            raise TypeError("gravity must share the real reference-length dtype.")
        if bool(jnp.any(~jnp.isfinite(lengths) | (lengths <= 0.0))):
            raise ValueError("reference_lengths must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(gravity_))):
            raise ValueError("gravity must be finite.")
        self.reference_lengths = lengths
        self.gravity = gravity_
        self._load_model_id = canonical_fingerprint(
            {
                "kind": "reference-member-self-weight",
                "reference": array_tree_fingerprint(lengths),
                "gravity": array_tree_fingerprint(gravity_),
            }
        )

    @property
    def load_model_id(self) -> str:
        return self._load_model_id

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
        if self.reference_lengths.shape != (structure.member_count,):
            raise ValueError("reference_lengths must match the member count.")
        if self.gravity.shape != (structure.dimension,):
            raise ValueError("gravity must match the spatial dimension.")
        line_mass = _real_array(
            "member line-mass parameters", parameters, (structure.member_count,)
        )
        line_mass = eqx.error_if(
            line_mass,
            jnp.any(
                structure.member_valid & (~jnp.isfinite(line_mass) | (line_mass < 0.0))
            ),
            "Active member line masses must be finite and nonnegative.",
        )
        total = jnp.where(
            structure.member_valid[:, None],
            line_mass[:, None] * self.reference_lengths[:, None] * self.gravity,
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
        loads = self.nodal_loads(structure, positions, parameters)
        return jnp.all(jnp.isfinite(loads))


class SurfacePressureLoadModel(AbstractForceDensityLoadModel):
    """Follower pressure integrated on regular oriented T3 and Q4 cells."""

    def __init__(self):
        pass

    @property
    def load_model_id(self) -> str:
        return "follower-surface-pressure-t3-q4"

    @property
    def depends_on_positions(self) -> bool:
        return True

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        connectivity, indices, points = _surface_points(structure, positions)
        if connectivity.cell_vertices.shape[1] != 4 or not np.all(
            np.isin(np.asarray(connectivity.cell_kinds), (3, 4))
        ):
            raise ValueError(
                "Surface pressure supports only triangular and quadrilateral cells."
            )
        pressure = _real_array(
            "surface pressure parameters", parameters, (connectivity.cell_count,)
        )
        pressure = eqx.error_if(
            pressure,
            jnp.any(~jnp.isfinite(pressure)),
            "Surface pressures must be finite.",
        )
        forces = _pressure_forces(points, connectivity.cell_kinds, pressure)
        return _scatter_surface_forces(
            structure, indices, connectivity.cell_vertex_valid, forces
        )

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
        if connectivity.cell_vertices.shape[1] != 4 or not np.all(
            np.isin(np.asarray(connectivity.cell_kinds), (3, 4))
        ):
            return jnp.asarray(False)
        pressure = _real_array(
            "surface pressure parameters", parameters, (connectivity.cell_count,)
        )
        regular, _ = _surface_regularity(structure, positions)
        return jnp.all(jnp.isfinite(pressure)) & regular


class SurfaceTractionLoadModel(AbstractForceDensityLoadModel, NonTrainableState):
    """Global vector traction integrated over current or reference cell area."""

    measure: Literal["current", "reference"] = eqx.field(static=True)
    reference_positions: Array | None
    _load_model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        measure: Literal["current", "reference"] = "current",
        reference_positions: ArrayLike | None = None,
    ):
        if measure not in ("current", "reference"):
            raise ValueError("measure must be 'current' or 'reference'.")
        if measure == "reference" and reference_positions is None:
            raise ValueError("Reference surface traction requires reference_positions.")
        if measure == "current" and reference_positions is not None:
            raise ValueError(
                "Current surface traction does not take reference_positions."
            )
        reference = (
            None if reference_positions is None else jnp.asarray(reference_positions)
        )
        if reference is not None and (
            reference.ndim != 2
            or not jnp.issubdtype(reference.dtype, jnp.inexact)
            or jnp.iscomplexobj(reference)
        ):
            raise TypeError("reference_positions must be one real rank-2 array.")
        self.measure = measure
        self.reference_positions = reference
        self._load_model_id = canonical_fingerprint(
            {
                "kind": "surface-traction-t3-q4",
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

    def _geometry(self, positions: ArrayLike, /) -> ArrayLike:
        if self.measure == "current":
            return positions
        if self.reference_positions is None:
            raise RuntimeError("Reference surface geometry is unavailable.")
        return self.reference_positions

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        connectivity, indices, points = _surface_points(
            structure, self._geometry(positions)
        )
        traction = _real_array(
            "surface traction parameters", parameters, (connectivity.cell_count, 3)
        )
        traction = eqx.error_if(
            traction,
            jnp.any(~jnp.isfinite(traction)),
            "Surface tractions must be finite.",
        )
        forces = _traction_forces(points, connectivity.cell_kinds, traction)
        return _scatter_surface_forces(
            structure, indices, connectivity.cell_vertex_valid, forces
        )

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
        traction = _real_array(
            "surface traction parameters", parameters, (connectivity.cell_count, 3)
        )
        regular, _ = _surface_regularity(structure, self._geometry(positions))
        return jnp.all(jnp.isfinite(traction)) & regular


class PneumaticPressureLoadModel(AbstractForceDensityLoadModel, NonTrainableState):
    """Closed-surface follower pressure with fixed or ideal-gas volume law."""

    law: Literal["fixed", "ideal-gas"] = eqx.field(static=True)
    reference_volume: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    _load_model_id: str = eqx.field(static=True)

    def __init__(
        self,
        law: Literal["fixed", "ideal-gas"] = "fixed",
        /,
        *,
        reference_volume: float = 1.0,
        exponent: float = 1.0,
    ):
        if law not in ("fixed", "ideal-gas"):
            raise ValueError("law must be 'fixed' or 'ideal-gas'.")
        volume = float(reference_volume)
        exponent_ = float(exponent)
        if not isfinite(volume) or volume == 0.0:
            raise ValueError("reference_volume must be finite and nonzero.")
        if not isfinite(exponent_) or exponent_ <= 0.0:
            raise ValueError("exponent must be finite and positive.")
        self.law = law
        self.reference_volume = volume
        self.exponent = exponent_
        self._load_model_id = canonical_fingerprint(
            {
                "kind": "pneumatic-pressure-t3-q4",
                "law": law,
                "reference_volume": volume,
                "exponent": exponent_,
            }
        )

    @property
    def load_model_id(self) -> str:
        return self._load_model_id

    @property
    def depends_on_positions(self) -> bool:
        return True

    def pressure(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameter: ArrayLike,
        /,
    ) -> Array:
        reference_pressure = _real_array("pneumatic pressure parameter", parameter, ())
        if self.law == "fixed":
            return reference_pressure
        volume = enclosed_surface_volume(structure, positions)
        ratio = jnp.asarray(self.reference_volume, dtype=volume.dtype) / volume
        return reference_pressure * ratio**self.exponent

    def nodal_loads(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> Array:
        connectivity, indices, points = _surface_points(structure, positions)
        pressure = self.pressure(structure, positions, parameters)
        pressure = eqx.error_if(
            pressure,
            jnp.any(connectivity.boundary_edges),
            "Pneumatic pressure requires a closed polygonal surface.",
        )
        forces = _pressure_forces(
            points,
            connectivity.cell_kinds,
            jnp.full((connectivity.cell_count,), pressure, dtype=points.dtype),
        )
        return _scatter_surface_forces(
            structure, indices, connectivity.cell_vertex_valid, forces
        )

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
        closed = ~jnp.any(connectivity.boundary_edges)
        regular, _ = _surface_regularity(structure, positions)
        volume = enclosed_surface_volume(structure, positions)
        reference = jnp.asarray(self.reference_volume, dtype=volume.dtype)
        pressure = self.pressure(structure, positions, parameters)
        return (
            closed
            & regular
            & jnp.isfinite(volume)
            & (jnp.abs(volume) > 0.0)
            & (volume * reference > 0.0)
            & jnp.isfinite(pressure)
        )


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
        return evaluate_force_density_load(self, structure, positions, parameters).total

    def valid(
        self,
        structure: ForceDensityStructure,
        positions: ArrayLike,
        parameters: Any,
        /,
    ) -> Array:
        return evaluate_force_density_load(self, structure, positions, parameters).valid


def _component_regularity(
    model: AbstractForceDensityLoadModel,
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
) -> Array:
    if isinstance(model, SurfaceTractionLoadModel):
        if model.measure == "reference":
            if model.reference_positions is None:
                return jnp.asarray(-jnp.inf)
            positions = model.reference_positions
        return _surface_regularity(structure, positions)[1]
    if isinstance(model, (SurfacePressureLoadModel, PneumaticPressureLoadModel)):
        return _surface_regularity(structure, positions)[1]
    return jnp.asarray(jnp.inf, dtype=jnp.asarray(positions).dtype)


def evaluate_force_density_load(
    model: AbstractForceDensityLoadModel,
    structure: ForceDensityStructure,
    positions: ArrayLike,
    parameters: Any,
    /,
) -> ForceDensityLoadState:
    """Evaluate aggregate load and retain every physical component."""
    if isinstance(model, CompositeForceDensityLoadModel):
        if not isinstance(parameters, tuple) or len(parameters) != len(model.models):
            raise TypeError(
                "Composite load parameters must be one tuple matching the models."
            )
        children = tuple(
            evaluate_force_density_load(child, structure, positions, child_parameters)
            for child, child_parameters in zip(model.models, parameters, strict=True)
        )
        components = tuple(
            component for child in children for component in child.components
        )
        identifiers = tuple(
            identifier for child in children for identifier in child.component_ids
        )
        total = sum(
            (child.total for child in children),
            jnp.zeros(
                (structure.node_count, structure.dimension),
                dtype=jnp.asarray(positions).dtype,
            ),
        )
        valid = jnp.asarray(True)
        regularity = jnp.asarray(jnp.inf, dtype=total.dtype)
        for child in children:
            valid = valid & child.valid
            regularity = jnp.minimum(regularity, child.minimum_regularity)
        return ForceDensityLoadState(total, components, identifiers, valid, regularity)
    loads = model.nodal_loads(structure, positions, parameters)
    valid = model.valid(structure, positions, parameters)
    regularity = _component_regularity(model, structure, positions)
    return ForceDensityLoadState(
        loads,
        (loads,),
        (model.load_model_id,),
        valid,
        regularity,
    )


__all__ = [
    "AbstractForceDensityLoadModel",
    "CompositeForceDensityLoadModel",
    "EdgeLineLoadModel",
    "FixedNodalLoadModel",
    "ForceDensityLoadState",
    "PneumaticPressureLoadModel",
    "ReferenceMemberSelfWeightModel",
    "SurfacePressureLoadModel",
    "SurfaceTractionLoadModel",
    "enclosed_surface_volume",
    "evaluate_force_density_load",
]

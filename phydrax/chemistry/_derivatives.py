#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic and finite-difference molecular Hessian workflows."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomicStructure, AtomisticSystemPlan, AtomisticUnitSystem
from ..execution import HostTaskExecutor, InlineTaskExecutor
from ._optimization import _require_structure_matches_system
from ._surface import AbstractPreparedPotentialEnergySurface
from ._units import hessian_unit


class MolecularHessianResult(StrictModule, NonTrainableState):
    raw_hessian: Array
    hessian: Array
    antisymmetry_residual: Array
    evaluation_count: Array
    successful: Array
    units: AtomisticUnitSystem
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_hessian,
        hessian,
        antisymmetry_residual,
        evaluation_count: int,
        successful,
        units: AtomisticUnitSystem,
        source_result_ids: tuple[str, ...],
        plan_id: str,
        /,
    ):
        raw = jnp.asarray(raw_hessian)
        symmetric = jnp.asarray(hessian, dtype=raw.dtype)
        if raw.ndim != 4 or raw.shape != symmetric.shape or raw.shape[1::2] != (3, 3):
            raise ValueError("Molecular Hessians must have shape (atom,3,atom,3).")
        residual = jnp.asarray(antisymmetry_residual, dtype=raw.dtype).reshape(())
        count = jnp.asarray(evaluation_count, dtype=jnp.int32).reshape(())
        successful_ = jnp.asarray(successful, dtype=bool).reshape(())
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        sources = tuple(str(value).strip() for value in source_result_ids)
        if not sources or any(not value for value in sources):
            raise ValueError("Hessian source result IDs must be non-empty.")
        self.raw_hessian = raw
        self.hessian = symmetric
        self.antisymmetry_residual = residual
        self.evaluation_count = count
        self.successful = successful_
        self.units = units
        self.source_result_ids = sources
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "molecular-hessian-result",
                "plan": self.plan_id,
                "units": hessian_unit(units).unit_id,
                "sources": list(sources),
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "raw_hessian": np.asarray(raw),
                        "hessian": np.asarray(symmetric),
                        "antisymmetry_residual": np.asarray(residual),
                        "evaluation_count": np.asarray(count),
                    }
                ),
            }
        )


class MolecularHessianPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    displacement: float = eqx.field(static=True)
    antisymmetry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        /,
        *,
        displacement: float = 1.0e-3,
        antisymmetry_tolerance: float = 1.0e-5,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if not isinstance(surface, AbstractPreparedPotentialEnergySurface):
            raise TypeError("surface must be a prepared potential-energy surface.")
        if surface.system_id != system.system_id:
            raise ValueError("Hessian surface belongs to another system.")
        if surface.units.unit_system_id != system.units.unit_system_id:
            raise ValueError("Hessian surface and system units differ.")
        if not surface.capabilities.forces:
            raise ValueError("Molecular Hessians require surface forces.")
        step = float(displacement)
        tolerance = float(antisymmetry_tolerance)
        if any(not isfinite(value) or value <= 0.0 for value in (step, tolerance)):
            raise ValueError("Hessian displacement and tolerance must be positive finite.")
        self.system = system
        self.surface = surface
        self.displacement = step
        self.antisymmetry_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-hessian-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "displacement": step,
                "antisymmetry_tolerance": tolerance,
                "length_unit": system.units.scale.length_unit.unit_id,
            }
        )

    def evaluate(
        self,
        structure: AtomicStructure,
        /,
        *,
        executor: HostTaskExecutor | None = None,
    ) -> MolecularHessianResult:
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be AtomicStructure.")
        _require_structure_matches_system(structure, self.system)
        positions = np.asarray(structure.positions, dtype=np.dtype(self.system.coordinate_dtype))
        active = np.asarray(self.system.active_mask, dtype=bool)
        active_indices = np.flatnonzero(active)
        cell = None if structure.cell is None else np.asarray(structure.cell)
        if self.surface.capabilities.hessian:
            evaluation = self.surface.evaluate(positions, cell)
            if evaluation.hessian is None:
                raise ValueError("Hessian-capable surface omitted its Hessian.")
            raw = np.array(evaluation.hessian, copy=True)
            sources = (evaluation.source_result_id,)
            count = 1
            all_successful = bool(evaluation.successful)
        else:
            coordinates = tuple(
                (int(atom), component)
                for atom in active_indices
                for component in range(3)
            )
            owned_executor = executor is None
            selected: HostTaskExecutor = InlineTaskExecutor() if executor is None else executor

            def displaced(atom: int, component: int, direction: int):
                candidate = positions.copy()
                candidate[atom, component] += direction * self.displacement
                return self.surface.evaluate(candidate, cell)

            handles = []
            for atom, component in coordinates:
                for direction in (-1, 1):
                    task_id = f"{self.plan_id}:{atom}:{component}:{direction:+d}"
                    handles.append(
                        (
                            atom,
                            component,
                            direction,
                            selected.submit(
                                task_id,
                                displaced,
                                atom,
                                component,
                                direction,
                                byte_count=positions.nbytes,
                            ),
                        )
                    )
            values = tuple(
                (atom, component, direction, handle.result())
                for atom, component, direction, handle in handles
            )
            if owned_executor:
                selected.close()
            by_coordinate = {
                (atom, component, direction): value
                for atom, component, direction, value in values
            }
            raw = np.zeros((active.size, 3, active.size, 3), dtype=positions.dtype)
            sources_list: list[str] = []
            all_successful = True
            for atom, component in coordinates:
                minus = by_coordinate[(atom, component, -1)]
                plus = by_coordinate[(atom, component, 1)]
                raw[:, :, atom, component] = -(
                    np.asarray(plus.forces) - np.asarray(minus.forces)
                ) / (2.0 * self.displacement)
                sources_list.extend((minus.source_result_id, plus.source_result_id))
                all_successful = (
                    all_successful and bool(minus.successful) and bool(plus.successful)
                )
            sources = tuple(sources_list)
            count = len(values)
        transpose = np.transpose(raw, (2, 3, 0, 1))
        scale = max(float(np.max(np.abs(raw), initial=0.0)), 1.0)
        residual = float(np.max(np.abs(raw - transpose), initial=0.0) / scale)
        symmetric = 0.5 * (raw + transpose)
        inactive = ~active
        symmetric[inactive, :, :, :] = 0.0
        symmetric[:, :, inactive, :] = 0.0
        raw[inactive, :, :, :] = 0.0
        raw[:, :, inactive, :] = 0.0
        successful = (
            all_successful
            and np.all(np.isfinite(symmetric))
            and residual <= self.antisymmetry_tolerance
        )
        return MolecularHessianResult(
            raw,
            symmetric,
            residual,
            count,
            successful,
            self.system.units,
            sources,
            self.plan_id,
        )


__all__ = ["MolecularHessianPlan", "MolecularHessianResult"]

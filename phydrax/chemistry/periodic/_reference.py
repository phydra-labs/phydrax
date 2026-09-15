#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral periodic electronic tasks and reference results."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition


class PeriodicElectronicTaskPlan(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    properties: tuple[str, ...] = eqx.field(static=True)
    reference_definition_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        properties: Sequence[str],
        reference_definition_id: str,
        /,
    ):
        method_ = str(method).strip().lower()
        properties_ = tuple(sorted(str(value).strip().lower() for value in properties))
        definition = str(reference_definition_id).strip()
        allowed = {
            "energy",
            "forces",
            "stress",
            "bands",
            "density",
            "polarization",
        }
        if (
            not method_
            or not properties_
            or len(set(properties_)) != len(properties_)
            or any(value not in allowed for value in properties_)
            or not definition
        ):
            raise ValueError(
                "Periodic method, properties, or reference definition is invalid."
            )
        self.method = method_
        self.properties = properties_
        self.reference_definition_id = definition
        self.task_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-task",
                "method": method_,
                "properties": properties_,
                "reference_definition": definition,
            }
        )


class PeriodicElectronicReferenceResult(StrictModule, NonTrainableState):
    energy: Array
    forces: Array | None
    stress: Array | None
    band_energies: Array | None
    density_matrices: Array | None
    polarization: Array | None
    residual: Array
    successful: Array
    energy_unit: UnitDefinition
    provider_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        residual: ArrayLike,
        successful: ArrayLike,
        energy_unit: UnitDefinition,
        provider_id: str,
        task_id: str,
        /,
        *,
        forces: ArrayLike | None = None,
        stress: ArrayLike | None = None,
        band_energies: ArrayLike | None = None,
        density_matrices: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        force = None if forces is None else jnp.asarray(forces, dtype=energy_.dtype)
        stress_ = None if stress is None else jnp.asarray(stress, dtype=energy_.dtype)
        bands = None if band_energies is None else jnp.asarray(band_energies)
        density = None if density_matrices is None else jnp.asarray(density_matrices)
        polarization_ = (
            None
            if polarization is None
            else jnp.asarray(polarization, dtype=energy_.real.dtype)
        )
        provider = str(provider_id).strip()
        task = str(task_id).strip()
        if force is not None and (force.ndim != 2 or force.shape[1] != 3):
            raise ValueError("Periodic forces must have shape (atom, 3).")
        if stress_ is not None and stress_.shape != (3, 3):
            raise ValueError("Periodic stress must have shape (3, 3).")
        if bands is not None and bands.ndim not in (2, 3):
            raise ValueError("Periodic bands must have shape (k,band) or (spin,k,band).")
        if density is not None and density.ndim not in (3, 4):
            raise ValueError(
                "Periodic densities must carry k and AO axes, optionally spin."
            )
        if polarization_ is not None and polarization_.shape != (3,):
            raise ValueError("Periodic polarization must have shape (3,).")
        if not isinstance(energy_unit, UnitDefinition) or not provider or not task:
            raise ValueError("Periodic result units and identities are invalid.")
        residual_ = jnp.asarray(residual, dtype=energy_.real.dtype).reshape(())
        finite = (
            jnp.isfinite(energy_)
            & jnp.isfinite(residual_)
            & (force is None or jnp.all(jnp.isfinite(force)))
            & (stress_ is None or jnp.all(jnp.isfinite(stress_)))
            & (bands is None or jnp.all(jnp.isfinite(bands)))
            & (density is None or jnp.all(jnp.isfinite(density)))
            & (polarization_ is None or jnp.all(jnp.isfinite(polarization_)))
        )
        self.energy = energy_
        self.forces = force
        self.stress = stress_
        self.band_energies = bands
        self.density_matrices = density
        self.polarization = polarization_
        self.residual = residual_
        self.successful = jnp.asarray(successful, dtype=bool).reshape(()) & finite
        self.energy_unit = energy_unit
        self.provider_id = provider
        self.task_id = task
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-reference-result",
                "provider": provider,
                "task": task,
                "energy_unit": energy_unit.unit_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "forces": None if force is None else np.asarray(force),
                        "stress": None if stress_ is None else np.asarray(stress_),
                        "bands": None if bands is None else np.asarray(bands),
                        "density": None if density is None else np.asarray(density),
                        "polarization": None
                        if polarization_ is None
                        else np.asarray(polarization_),
                        "residual": np.asarray(residual_),
                    }
                ),
            }
        )


class AbstractPeriodicReferenceProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]
    reference_definition_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        task: PeriodicElectronicTaskPlan,
        positions: ArrayLike,
        cell_vectors: ArrayLike,
        /,
    ) -> PeriodicElectronicReferenceResult:
        raise NotImplementedError


PeriodicReferenceEvaluator = Callable[
    [PeriodicElectronicTaskPlan, ArrayLike, ArrayLike],
    PeriodicElectronicReferenceResult,
]


class CallablePeriodicReferenceProvider(AbstractPeriodicReferenceProvider):
    evaluator: PeriodicReferenceEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    reference_definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: PeriodicReferenceEvaluator,
        provider_id: str,
        reference_definition_id: str,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        definition = str(reference_definition_id).strip()
        if not provider or not definition:
            raise ValueError("Periodic provider identities must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider
        self.reference_definition_id = definition

    def evaluate(
        self,
        task: PeriodicElectronicTaskPlan,
        positions: ArrayLike,
        cell_vectors: ArrayLike,
        /,
    ) -> PeriodicElectronicReferenceResult:
        if task.reference_definition_id != self.reference_definition_id:
            raise ValueError("Periodic task requests another provider definition.")
        result = self.evaluator(task, positions, cell_vectors)
        if not isinstance(result, PeriodicElectronicReferenceResult):
            raise TypeError("Periodic provider returned the wrong result type.")
        if result.provider_id != self.provider_id or result.task_id != task.task_id:
            raise ValueError("Periodic provider changed bound identities.")
        supplied = {
            "energy": True,
            "forces": result.forces is not None,
            "stress": result.stress is not None,
            "bands": result.band_energies is not None,
            "density": result.density_matrices is not None,
            "polarization": result.polarization is not None,
        }
        if any(not supplied[property_] for property_ in task.properties):
            raise ValueError("Periodic provider omitted a requested property.")
        return result


__all__ = [
    "AbstractPeriodicReferenceProvider",
    "CallablePeriodicReferenceProvider",
    "PeriodicElectronicReferenceResult",
    "PeriodicElectronicTaskPlan",
]

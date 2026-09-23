#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit electronic-provider capability and preparation boundary."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._context import ElectronicEvaluationContext
from ._model import ElectronicMethodFamily, ElectronicReferenceKind
from ._result import (
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicEvaluation,
    ElectronicGroundStatePropertyEvaluation,
    ElectronicPeriodicEvaluation,
)
from ._task import (
    ElectronicProperty,
    ElectronicTaskKind,
    ExcitedManifoldTaskPlan,
    LinearResponseTaskPlan,
)


if TYPE_CHECKING:
    from ._calculation import ElectronicCalculationPlan


ElectronicExecutionKind = Literal["host", "device"]
ElectronicConcurrencyKind = Literal["serial", "thread-safe", "process-isolated"]


class ElectronicProviderUnavailableError(RuntimeError):
    """An explicitly selected provider is unavailable."""


class ElectronicCapabilityError(ValueError):
    """An available provider cannot satisfy the requested calculation."""


def _enum_tuple(values: Sequence, enum_type: type, name: str, /) -> tuple:
    normalized = tuple(sorted(set(values), key=lambda value: value.value))
    if not normalized or any(not isinstance(value, enum_type) for value in normalized):
        raise TypeError(f"{name} must contain typed non-empty values.")
    return normalized


class ElectronicTheoryCapabilities(StrictModule, NonTrainableState):
    families: tuple[ElectronicMethodFamily, ...] = eqx.field(static=True)
    references: tuple[ElectronicReferenceKind, ...] = eqx.field(static=True)
    total_charge: bool = eqx.field(static=True)
    spin_multiplicity: bool = eqx.field(static=True)
    spinor: bool = eqx.field(static=True)
    spatial_symmetry: bool = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        families: Sequence[ElectronicMethodFamily],
        references: Sequence[ElectronicReferenceKind],
        /,
        *,
        total_charge: bool = True,
        spin_multiplicity: bool = True,
        spinor: bool = False,
        spatial_symmetry: bool = False,
    ):
        families_ = _enum_tuple(families, ElectronicMethodFamily, "method families")
        references_ = _enum_tuple(references, ElectronicReferenceKind, "reference kinds")
        self.families = families_
        self.references = references_
        self.total_charge = bool(total_charge)
        self.spin_multiplicity = bool(spin_multiplicity)
        self.spinor = bool(spinor)
        self.spatial_symmetry = bool(spatial_symmetry)
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-theory-capabilities",
                "families": [value.value for value in families_],
                "references": [value.value for value in references_],
                "total_charge": self.total_charge,
                "spin_multiplicity": self.spin_multiplicity,
                "spinor": self.spinor,
                "spatial_symmetry": self.spatial_symmetry,
            }
        )


class ElectronicGeometryCapabilities(StrictModule, NonTrainableState):
    finite: bool = eqx.field(static=True)
    periodic_ranks: tuple[int, ...] = eqx.field(static=True)
    variable_cell: bool = eqx.field(static=True)
    low_dimensional_coulomb: bool = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        finite: bool = True,
        periodic_ranks: Sequence[int] = (),
        variable_cell: bool = False,
        low_dimensional_coulomb: bool = False,
    ):
        ranks = tuple(sorted(set(int(value) for value in periodic_ranks)))
        if any(value not in (1, 2, 3) for value in ranks):
            raise ValueError("Periodic geometry ranks must be one, two, or three.")
        if variable_cell and not ranks:
            raise ValueError("Variable-cell support requires periodic geometry.")
        if low_dimensional_coulomb and not any(value in (1, 2) for value in ranks):
            raise ValueError(
                "Low-dimensional Coulomb support requires rank-one or rank-two periodicity."
            )
        self.finite = bool(finite)
        self.periodic_ranks = ranks
        self.variable_cell = bool(variable_cell)
        self.low_dimensional_coulomb = bool(low_dimensional_coulomb)
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-geometry-capabilities",
                "finite": self.finite,
                "periodic_ranks": list(ranks),
                "variable_cell": self.variable_cell,
                "low_dimensional_coulomb": self.low_dimensional_coulomb,
            }
        )


class ElectronicObservableCapabilities(StrictModule, NonTrainableState):
    tasks: tuple[ElectronicTaskKind, ...] = eqx.field(static=True)
    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    derivative_orders: tuple[int, ...] = eqx.field(static=True)
    gauges: tuple[str, ...] = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        tasks: Sequence[ElectronicTaskKind],
        properties: Sequence[ElectronicProperty],
        /,
        *,
        derivative_orders: Sequence[int] = (),
        gauges: Sequence[str] = ("length",),
    ):
        tasks_ = _enum_tuple(tasks, ElectronicTaskKind, "electronic tasks")
        properties_ = _enum_tuple(properties, ElectronicProperty, "electronic properties")
        orders = tuple(sorted(set(int(value) for value in derivative_orders)))
        if any(value not in (0, 1, 2, 3) for value in orders):
            raise ValueError(
                "Derivative orders must lie in the bounded range zero through three."
            )
        if (
            ElectronicProperty.HESSIAN in properties_
            and ElectronicProperty.FORCES not in properties_
        ):
            raise ValueError("Hessian provider capability requires forces.")
        gauges_ = tuple(sorted(set(str(value).strip() for value in gauges)))
        if any(not value for value in gauges_):
            raise ValueError("Gauge identifiers must be non-empty.")
        self.tasks = tasks_
        self.properties = properties_
        self.derivative_orders = orders
        self.gauges = gauges_
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-observable-capabilities",
                "tasks": [value.value for value in tasks_],
                "properties": [value.value for value in properties_],
                "derivative_orders": list(orders),
                "gauges": list(gauges_),
            }
        )


class ElectronicEmbeddingCapabilities(StrictModule, NonTrainableState):
    point_charges: bool = eqx.field(static=True)
    point_charge_forces: bool = eqx.field(static=True)
    permanent_multipoles: bool = eqx.field(static=True)
    polarizable: bool = eqx.field(static=True)
    stress: bool = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        point_charges: bool = False,
        point_charge_forces: bool = False,
        permanent_multipoles: bool = False,
        polarizable: bool = False,
        stress: bool = False,
    ):
        if point_charge_forces and not point_charges:
            raise ValueError("Point-charge forces require point-charge embedding.")
        if polarizable and not (point_charges or permanent_multipoles):
            raise ValueError(
                "Polarizable embedding requires an electrostatic source model."
            )
        self.point_charges = bool(point_charges)
        self.point_charge_forces = bool(point_charge_forces)
        self.permanent_multipoles = bool(permanent_multipoles)
        self.polarizable = bool(polarizable)
        self.stress = bool(stress)
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-embedding-capabilities",
                "point_charges": self.point_charges,
                "point_charge_forces": self.point_charge_forces,
                "permanent_multipoles": self.permanent_multipoles,
                "polarizable": self.polarizable,
                "stress": self.stress,
            }
        )


class ElectronicExecutionCapabilities(StrictModule, NonTrainableState):
    conservative_forces: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    batching: bool = eqx.field(static=True)
    checkpointing: bool = eqx.field(static=True)
    execution: ElectronicExecutionKind = eqx.field(static=True)
    concurrency: ElectronicConcurrencyKind = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        conservative_forces: bool = True,
        differentiable: bool = False,
        batching: bool = False,
        checkpointing: bool = False,
        execution: ElectronicExecutionKind = "host",
        concurrency: ElectronicConcurrencyKind = "serial",
    ):
        if execution not in ("host", "device"):
            raise ValueError("execution must be host or device.")
        if concurrency not in ("serial", "thread-safe", "process-isolated"):
            raise ValueError("Unknown electronic provider concurrency model.")
        self.conservative_forces = bool(conservative_forces)
        self.differentiable = bool(differentiable)
        self.batching = bool(batching)
        self.checkpointing = bool(checkpointing)
        self.execution = execution
        self.concurrency = concurrency
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-execution-capabilities",
                "conservative_forces": self.conservative_forces,
                "differentiable": self.differentiable,
                "batching": self.batching,
                "checkpointing": self.checkpointing,
                "execution": execution,
                "concurrency": concurrency,
            }
        )


class ElectronicProviderCapabilities(StrictModule, NonTrainableState):
    """Nested immutable support of one electronic provider."""

    theory: ElectronicTheoryCapabilities
    geometry: ElectronicGeometryCapabilities
    observables: ElectronicObservableCapabilities
    embedding: ElectronicEmbeddingCapabilities
    execution: ElectronicExecutionCapabilities
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        theory: ElectronicTheoryCapabilities,
        geometry: ElectronicGeometryCapabilities,
        observables: ElectronicObservableCapabilities,
        /,
        *,
        embedding: ElectronicEmbeddingCapabilities | None = None,
        execution: ElectronicExecutionCapabilities | None = None,
    ):
        if not isinstance(theory, ElectronicTheoryCapabilities):
            raise TypeError("theory must be ElectronicTheoryCapabilities.")
        if not isinstance(geometry, ElectronicGeometryCapabilities):
            raise TypeError("geometry must be ElectronicGeometryCapabilities.")
        if not isinstance(observables, ElectronicObservableCapabilities):
            raise TypeError("observables must be ElectronicObservableCapabilities.")
        embedding_ = ElectronicEmbeddingCapabilities() if embedding is None else embedding
        execution_ = ElectronicExecutionCapabilities() if execution is None else execution
        if not isinstance(embedding_, ElectronicEmbeddingCapabilities):
            raise TypeError("embedding must be ElectronicEmbeddingCapabilities or None.")
        if not isinstance(execution_, ElectronicExecutionCapabilities):
            raise TypeError("execution must be ElectronicExecutionCapabilities or None.")
        if any(
            (
                embedding_.point_charges,
                embedding_.point_charge_forces,
                embedding_.permanent_multipoles,
                embedding_.polarizable,
                embedding_.stress,
            )
        ):
            raise ValueError(
                "Electronic providers cannot advertise embedding contexts until a prepared implementation supplies evaluate_context."
            )
        self.theory = theory
        self.geometry = geometry
        self.observables = observables
        self.embedding = embedding_
        self.execution = execution_
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-provider-capabilities",
                "theory": theory.capabilities_id,
                "geometry": geometry.capabilities_id,
                "observables": observables.capabilities_id,
                "embedding": embedding_.capabilities_id,
                "execution": execution_.capabilities_id,
            }
        )

    @classmethod
    def molecular_ground_state(
        cls,
        properties: Sequence[ElectronicProperty],
        references: Sequence[ElectronicReferenceKind],
        /,
        *,
        families: Sequence[ElectronicMethodFamily] = (
            ElectronicMethodFamily.HARTREE_FOCK,
            ElectronicMethodFamily.KOHN_SHAM_DFT,
            ElectronicMethodFamily.CUSTOM_EXTERNAL,
        ),
        total_charge: bool = True,
        spin_multiplicity: bool = True,
        point_charges: bool = False,
        point_charge_forces: bool = False,
        conservative_forces: bool = True,
        differentiable: bool = False,
        batching: bool = False,
        execution: ElectronicExecutionKind = "host",
        concurrency: ElectronicConcurrencyKind = "serial",
    ) -> ElectronicProviderCapabilities:
        derivative_orders = tuple(
            sorted(
                {
                    order
                    for property_, order in (
                        (ElectronicProperty.ENERGY, 0),
                        (ElectronicProperty.FORCES, 1),
                        (ElectronicProperty.HESSIAN, 2),
                    )
                    if property_ in properties
                }
            )
        )
        return cls(
            ElectronicTheoryCapabilities(
                families,
                references,
                total_charge=total_charge,
                spin_multiplicity=spin_multiplicity,
            ),
            ElectronicGeometryCapabilities(finite=True),
            ElectronicObservableCapabilities(
                (ElectronicTaskKind.GROUND_STATE,),
                properties,
                derivative_orders=derivative_orders,
            ),
            embedding=ElectronicEmbeddingCapabilities(
                point_charges=point_charges,
                point_charge_forces=point_charge_forces,
            ),
            execution=ElectronicExecutionCapabilities(
                conservative_forces=conservative_forces,
                differentiable=differentiable,
                batching=batching,
                execution=execution,
                concurrency=concurrency,
            ),
        )

    def require(self, calculation: ElectronicCalculationPlan, /) -> None:
        from ._calculation import ElectronicCalculationPlan

        if not isinstance(calculation, ElectronicCalculationPlan):
            raise TypeError("calculation must be ElectronicCalculationPlan.")
        cell = calculation.system.cell
        periodic_rank = (
            0 if cell is None else sum(bool(value) for value in cell.periodic_axes)
        )
        if periodic_rank and periodic_rank not in self.geometry.periodic_ranks:
            raise ElectronicCapabilityError(
                f"Provider does not support periodic rank {periodic_rank}."
            )
        if not periodic_rank and not self.geometry.finite:
            raise ElectronicCapabilityError("Provider does not support finite geometry.")
        if periodic_rank in (1, 2) and not self.geometry.low_dimensional_coulomb:
            raise ElectronicCapabilityError(
                "Provider lacks low-dimensional Coulomb support."
            )
        method = calculation.model_chemistry.method
        if method.family not in self.theory.families:
            raise ElectronicCapabilityError(
                f"Provider does not support method family {method.family.value!r}."
            )
        if method.reference not in self.theory.references:
            raise ElectronicCapabilityError(
                f"Provider does not support reference kind {method.reference.value!r}."
            )
        if (
            method.reference is ElectronicReferenceKind.NONCOLLINEAR
            and not self.theory.spinor
        ):
            raise ElectronicCapabilityError(
                "Noncollinear references require provider spinor support."
            )
        if calculation.task.task_kind not in self.observables.tasks:
            raise ElectronicCapabilityError(
                f"Provider does not support task kind {calculation.task.task_kind.value!r}."
            )
        missing = tuple(
            property_
            for property_ in calculation.task.properties
            if property_ not in self.observables.properties
        )
        if missing:
            names = ", ".join(value.value for value in missing)
            raise ElectronicCapabilityError(
                f"Provider lacks requested properties: {names}."
            )
        required_orders = {
            order
            for property_, order in (
                (ElectronicProperty.ENERGY, 0),
                (ElectronicProperty.FORCES, 1),
                (ElectronicProperty.HESSIAN, 2),
            )
            if property_ in calculation.task.properties
        }
        if not required_orders.issubset(self.observables.derivative_orders):
            raise ElectronicCapabilityError(
                "Provider lacks the derivative orders required by the task."
            )
        if (
            ElectronicProperty.STRESS in calculation.task.properties
            and not self.geometry.variable_cell
        ):
            raise ElectronicCapabilityError(
                "Stress calculations require variable-cell provider support."
            )
        if isinstance(calculation.task, LinearResponseTaskPlan):
            if calculation.task.gauge not in self.observables.gauges:
                raise ElectronicCapabilityError(
                    "Provider does not support the requested response gauge."
                )
            if calculation.task.response_order not in self.observables.derivative_orders:
                raise ElectronicCapabilityError(
                    "Provider does not support the requested response order."
                )
        if (
            isinstance(calculation.task, ExcitedManifoldTaskPlan)
            and calculation.task.symmetry_sector is not None
            and not self.theory.spatial_symmetry
        ):
            raise ElectronicCapabilityError(
                "Provider does not support spatial-symmetry sectors."
            )
        state = calculation.state
        if state.total_charge != 0 and not self.theory.total_charge:
            raise ElectronicCapabilityError("Provider does not support charged systems.")
        if state.spin_multiplicity != 1 and not self.theory.spin_multiplicity:
            raise ElectronicCapabilityError(
                "Provider does not support open-shell systems."
            )


class AbstractPreparedElectronicCalculation(StrictModule, NonTrainableState):
    calculation: eqx.AbstractVar[ElectronicCalculationPlan]
    capabilities: eqx.AbstractVar[ElectronicProviderCapabilities]
    provider_id: eqx.AbstractVar[str]
    prepared_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        raise NotImplementedError

    def evaluate_context(
        self, context: ElectronicEvaluationContext, /
    ) -> ElectronicEvaluation:
        if not isinstance(context, ElectronicEvaluationContext):
            raise TypeError("context must be ElectronicEvaluationContext.")
        expected_shape = (
            self.calculation.system.particle_ids.shape[0],
            3,
        )
        if context.positions.shape != expected_shape:
            raise ValueError(f"Context positions must have shape {expected_shape}.")
        if context.embedding is not None:
            if (
                context.embedding.units.unit_system_id
                != self.calculation.system.units.unit_system_id
            ):
                raise ValueError(
                    "Embedding and electronic calculation unit systems differ."
                )
            raise ElectronicCapabilityError(
                "Provider does not implement point-charge embedding contexts."
            )
        if context.multipole_embedding is not None:
            if (
                context.multipole_embedding.units.unit_system_id
                != self.calculation.system.units.unit_system_id
            ):
                raise ValueError(
                    "Multipole embedding and electronic calculation units differ."
                )
            raise ElectronicCapabilityError(
                "Provider does not implement permanent-multipole contexts."
            )
        if context.polarizable_embedding is not None:
            if (
                context.polarizable_embedding.permanent.units.unit_system_id
                != self.calculation.system.units.unit_system_id
            ):
                raise ValueError(
                    "Polarizable embedding and electronic calculation units differ."
                )
            raise ElectronicCapabilityError(
                "Provider does not implement polarizable embedding contexts."
            )
        if context.external_field is not None:
            if (
                context.external_field.units.unit_system_id
                != self.calculation.system.units.unit_system_id
            ):
                raise ValueError(
                    "External field and electronic calculation unit systems differ."
                )
            raise ElectronicCapabilityError(
                "Provider does not implement external-field contexts."
            )
        if context.initial_guess is not None:
            raise ElectronicCapabilityError(
                "Provider does not implement explicit electronic-guess contexts."
            )
        return self.evaluate(context.positions, context.cell_vectors)


class AbstractElectronicProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    capabilities: eqx.AbstractVar[ElectronicProviderCapabilities]

    @abc.abstractmethod
    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> AbstractPreparedElectronicCalculation:
        raise NotImplementedError


ElectronicEvaluator = Callable[
    ["ElectronicCalculationPlan", ArrayLike, ArrayLike | None], ElectronicEvaluation
]


class CallablePreparedElectronicCalculation(AbstractPreparedElectronicCalculation):
    calculation: ElectronicCalculationPlan
    capabilities: ElectronicProviderCapabilities
    evaluator: ElectronicEvaluator
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        capabilities: ElectronicProviderCapabilities,
        evaluator: ElectronicEvaluator,
        provider_id: str,
        /,
    ):
        from ._calculation import ElectronicCalculationPlan

        if not isinstance(calculation, ElectronicCalculationPlan):
            raise TypeError("calculation must be ElectronicCalculationPlan.")
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        identifier = str(provider_id).strip()
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        capabilities.require(calculation)
        embedding = capabilities.embedding
        if any(
            (
                embedding.point_charges,
                embedding.point_charge_forces,
                embedding.permanent_multipoles,
                embedding.polarizable,
                embedding.stress,
            )
        ):
            raise ElectronicCapabilityError(
                "Callable prepared calculations cannot advertise context capabilities they do not implement."
            )
        self.calculation = calculation
        self.capabilities = capabilities
        self.evaluator = evaluator
        self.provider_id = identifier
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-callable-electronic-calculation",
                "calculation": calculation.calculation_id,
                "capabilities": capabilities.capabilities_id,
                "provider": identifier,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        from ._calculation import electronic_geometry_id

        expected_geometry = electronic_geometry_id(
            self.calculation.system, positions, cell_vectors
        )
        result = self.evaluator(self.calculation, positions, cell_vectors)
        if not isinstance(
            result,
            (
                ElectronicEnergyEvaluation,
                ElectronicEnergyForceEvaluation,
                ElectronicEnergyForceHessianEvaluation,
                ElectronicGroundStatePropertyEvaluation,
                ElectronicPeriodicEvaluation,
            ),
        ):
            raise TypeError("Electronic evaluator returned an unsupported result type.")
        header = result.header
        expected = {
            "provider_id": self.provider_id,
            "system_id": self.calculation.system.system_id,
            "geometry_id": expected_geometry,
            "state_id": self.calculation.state.prepared_id,
            "model_chemistry_id": self.calculation.model_chemistry.model_chemistry_id,
            "task_id": self.calculation.task.task_id,
        }
        if any(getattr(header, name) != value for name, value in expected.items()):
            raise ValueError(
                "Electronic evaluator changed a prepared calculation identity."
            )
        if (
            header.units.unit_system_id != self.calculation.system.units.unit_system_id
            or not np.array_equal(
                np.asarray(header.stable_particle_ids),
                np.asarray(self.calculation.system.particle_ids),
            )
            or not np.array_equal(
                np.asarray(header.active_mask),
                np.asarray(self.calculation.system.active_mask),
            )
        ):
            raise ValueError(
                "Electronic evaluator changed system units or particle support."
            )
        return result


class CallableElectronicProvider(AbstractElectronicProvider):
    evaluator: ElectronicEvaluator
    provider_id: str = eqx.field(static=True)
    capabilities: ElectronicProviderCapabilities

    def __init__(
        self,
        evaluator: ElectronicEvaluator,
        provider_id: str,
        capabilities: ElectronicProviderCapabilities,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        identifier = str(provider_id).strip()
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        self.evaluator = evaluator
        self.provider_id = identifier
        self.capabilities = capabilities

    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> CallablePreparedElectronicCalculation:
        from ._calculation import ElectronicCalculationPlan

        if not isinstance(calculation, ElectronicCalculationPlan):
            raise TypeError("calculation must be ElectronicCalculationPlan.")
        self.capabilities.require(calculation)
        return CallablePreparedElectronicCalculation(
            calculation,
            self.capabilities,
            self.evaluator,
            self.provider_id,
        )


__all__ = [
    "AbstractElectronicProvider",
    "AbstractPreparedElectronicCalculation",
    "CallableElectronicProvider",
    "CallablePreparedElectronicCalculation",
    "ElectronicCapabilityError",
    "ElectronicConcurrencyKind",
    "ElectronicEmbeddingCapabilities",
    "ElectronicExecutionCapabilities",
    "ElectronicExecutionKind",
    "ElectronicGeometryCapabilities",
    "ElectronicObservableCapabilities",
    "ElectronicProviderCapabilities",
    "ElectronicProviderUnavailableError",
    "ElectronicTheoryCapabilities",
]

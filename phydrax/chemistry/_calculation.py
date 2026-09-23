#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral electronic calculation planning and result construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticPrecisionPolicy, AtomisticSystemPlan
from ..units import ONE
from ._model import ElectronicModelChemistryPlan, ElectronicReferenceKind
from ._numerical import ElectronicNumericalPlan
from ._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicEvaluation,
    ElectronicEvaluationHeader,
    ElectronicGroundStatePropertyEvaluation,
    ElectronicPeriodicEvaluation,
    ElectronicWorkEvidence,
)
from ._state import (
    ElectronicSectorPlan,
    MolecularElectronicSectorPlan,
    PeriodicElectronicSectorPlan,
    PreparedElectronicSector,
    PreparedMolecularElectronicSector,
    PreparedPeriodicElectronicSector,
)
from ._task import (
    BandStructureTaskPlan,
    CorrelationTaskPlan,
    ElectronicProperty,
    ElectronicTaskPlan,
    GroundStateTaskPlan,
)
from ._units import (
    dipole_unit,
    hessian_unit,
    polarization_density_unit,
    stress_unit,
)


if TYPE_CHECKING:
    from ._provider import (
        AbstractElectronicProvider,
        AbstractPreparedElectronicCalculation,
    )


class ElectronicCalculationPlan(StrictModule, NonTrainableState):
    """One system, electronic sector, physical model, task, and numerical plan."""

    system: AtomisticSystemPlan
    state: PreparedElectronicSector
    model_chemistry: ElectronicModelChemistryPlan
    task: ElectronicTaskPlan
    numerical: ElectronicNumericalPlan
    precision: AtomisticPrecisionPolicy
    calculation_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        state: ElectronicSectorPlan | PreparedElectronicSector,
        model_chemistry: ElectronicModelChemistryPlan,
        task: ElectronicTaskPlan,
        /,
        *,
        numerical: ElectronicNumericalPlan | None = None,
        precision: AtomisticPrecisionPolicy | None = None,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if isinstance(
            state, (MolecularElectronicSectorPlan, PeriodicElectronicSectorPlan)
        ):
            prepared_state = state.prepare(system)
        elif isinstance(
            state, (PreparedMolecularElectronicSector, PreparedPeriodicElectronicSector)
        ):
            prepared_state = state
        else:
            raise TypeError(
                "state must be a molecular or periodic electronic sector plan, or its prepared form."
            )
        if prepared_state.system_id != system.system_id:
            raise ValueError("Prepared electronic sector belongs to another system.")
        if not isinstance(model_chemistry, ElectronicModelChemistryPlan):
            raise TypeError("model_chemistry must be ElectronicModelChemistryPlan.")
        if not isinstance(task, ElectronicTaskPlan):
            raise TypeError("task must be a typed electronic task plan.")
        reference = model_chemistry.method.reference
        if (
            reference is ElectronicReferenceKind.RESTRICTED
            and prepared_state.alpha_electron_count != prepared_state.beta_electron_count
        ):
            raise ValueError(
                "Restricted electronic references require equal alpha and beta populations."
            )
        periodic_task = isinstance(task, BandStructureTaskPlan) or any(
            property_
            in (
                ElectronicProperty.STRESS,
                ElectronicProperty.BAND_ENERGIES,
                ElectronicProperty.DENSITY_MATRIX,
                ElectronicProperty.POLARIZATION,
            )
            for property_ in task.properties
        )
        if periodic_task and (
            system.cell is None
            or not any(system.cell.periodic_axes)
            or not isinstance(prepared_state, PreparedPeriodicElectronicSector)
        ):
            raise ValueError(
                "Periodic electronic tasks require a periodic system and periodic sector."
            )
        numerical_ = ElectronicNumericalPlan() if numerical is None else numerical
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(numerical_, ElectronicNumericalPlan):
            raise TypeError("numerical must be ElectronicNumericalPlan or None.")
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be AtomisticPrecisionPolicy or None.")
        self.system = system
        self.state = prepared_state
        self.model_chemistry = model_chemistry
        self.task = task
        self.numerical = numerical_
        self.precision = precision_
        self.calculation_id = canonical_fingerprint(
            {
                "kind": "electronic-calculation-plan",
                "system": system.system_id,
                "state": prepared_state.prepared_id,
                "model_chemistry": model_chemistry.model_chemistry_id,
                "task": task.task_id,
                "numerical": numerical_.numerical_id,
                "precision": precision_.policy_id,
            }
        )

    def prepare(
        self, provider: AbstractElectronicProvider, /
    ) -> AbstractPreparedElectronicCalculation:
        from ._provider import AbstractElectronicProvider

        if not isinstance(provider, AbstractElectronicProvider):
            raise TypeError("provider must implement AbstractElectronicProvider.")
        return provider.prepare(self)


def electronic_geometry_id(
    system: AtomisticSystemPlan,
    positions: ArrayLike,
    cell_vectors: ArrayLike | None = None,
    /,
) -> str:
    if not isinstance(system, AtomisticSystemPlan):
        raise TypeError("system must be AtomisticSystemPlan.")
    coordinates = np.asarray(positions, dtype=np.dtype(system.coordinate_dtype))
    expected = (system.particle_ids.shape[0], 3)
    if coordinates.shape != expected:
        raise ValueError(f"positions must have shape {expected}.")
    active = np.asarray(system.active_mask, dtype=np.bool_)
    if np.any(~np.isfinite(coordinates[active])):
        raise ValueError("Active electronic coordinates must be finite.")
    vectors = (
        None
        if cell_vectors is None
        else np.asarray(cell_vectors, dtype=coordinates.dtype)
    )
    if vectors is not None and (vectors.shape != (3, 3) or np.any(~np.isfinite(vectors))):
        raise ValueError("cell_vectors must be a finite (3, 3) array when provided.")
    return canonical_fingerprint(
        {
            "kind": "electronic-geometry",
            "system": system.system_id,
            "arrays": array_tree_fingerprint(
                {
                    "positions": coordinates,
                    "cell_vectors": vectors,
                    "particle_ids": np.asarray(system.particle_ids),
                    "active_mask": active,
                }
            ),
        }
    )


def make_electronic_evaluation(
    calculation: ElectronicCalculationPlan,
    provider_id: str,
    positions: ArrayLike,
    energy: ArrayLike,
    /,
    *,
    forces: ArrayLike | None = None,
    hessian: ArrayLike | None = None,
    dipole: ArrayLike | None = None,
    stress: ArrayLike | None = None,
    band_energies: ArrayLike | None = None,
    density_matrices: ArrayLike | None = None,
    polarization: ArrayLike | None = None,
    cell_vectors: ArrayLike | None = None,
    convergence: ElectronicConvergenceEvidence | None = None,
    work: ElectronicWorkEvidence | None = None,
    status: ElectronicCalculationStatus = ElectronicCalculationStatus.SUCCESS,
    source_unit_ids: tuple[tuple[str, str], ...] | None = None,
    artifact_ids: tuple[str, ...] = (),
) -> ElectronicEvaluation:
    """Validate one provider value and construct the exact requested result variant."""

    if not isinstance(calculation, ElectronicCalculationPlan):
        raise TypeError("calculation must be ElectronicCalculationPlan.")
    if not isinstance(
        calculation.task,
        (GroundStateTaskPlan, CorrelationTaskPlan, BandStructureTaskPlan),
    ):
        raise TypeError(
            "make_electronic_evaluation only constructs ground-state, correlation, or band-structure evaluations."
        )
    provider = str(provider_id).strip()
    if not provider:
        raise ValueError("provider_id must be non-empty.")
    status_ = ElectronicCalculationStatus(int(np.asarray(status)))
    convergence_ = (
        ElectronicConvergenceEvidence(
            False,
            energy_residual=jnp.nan,
            density_residual=jnp.nan,
            message="not-reported",
        )
        if convergence is None
        else convergence
    )
    if not isinstance(convergence_, ElectronicConvergenceEvidence):
        raise TypeError("convergence must be ElectronicConvergenceEvidence or None.")
    work_ = (
        ElectronicWorkEvidence(
            energy_evaluations=1,
            force_evaluations=int(forces is not None),
            hessian_evaluations=int(hessian is not None),
            property_evaluations=int(dipole is not None),
        )
        if work is None
        else work
    )
    if not isinstance(work_, ElectronicWorkEvidence):
        raise TypeError("work must be ElectronicWorkEvidence or None.")
    if int(work_.energy_evaluations) < 1:
        raise ValueError("Electronic evaluations must report at least one energy call.")
    if forces is not None and int(work_.force_evaluations) < 1:
        raise ValueError("Force results must report force-evaluation work.")
    if hessian is not None and (
        int(work_.hessian_evaluations) < 1 and int(work_.force_evaluations) < 1
    ):
        raise ValueError("Hessian results must report Hessian or force-difference work.")
    if dipole is not None and int(work_.property_evaluations) < 1:
        raise ValueError("Dipole results must report property-evaluation work.")
    units = calculation.system.units
    if source_unit_ids is None:
        source_values = [
            ("energy", units.scale.energy_unit.unit_id),
            ("length", units.scale.length_unit.unit_id),
        ]
        if forces is not None:
            source_values.append(("forces", units.scale.force_unit.unit_id))
        if hessian is not None:
            source_values.append(("hessian", hessian_unit(units).unit_id))
        if dipole is not None:
            source_values.append(("dipole", dipole_unit(units).unit_id))
        if stress is not None:
            source_values.append(("stress", stress_unit(units).unit_id))
        if band_energies is not None:
            source_values.append(("band_energies", units.scale.energy_unit.unit_id))
        if density_matrices is not None:
            source_values.append(("density_matrices", ONE.unit_id))
        if polarization is not None:
            source_values.append(
                ("polarization", polarization_density_unit(units).unit_id)
            )
        sources = tuple(source_values)
    else:
        sources = source_unit_ids
    source_map = dict(sources)
    required_source_fields = {"energy", "length"}
    for name, supplied in (
        ("forces", forces),
        ("hessian", hessian),
        ("dipole", dipole),
        ("stress", stress),
        ("band_energies", band_energies),
        ("density_matrices", density_matrices),
        ("polarization", polarization),
    ):
        if supplied is not None:
            required_source_fields.add(name)
    missing_source_fields = tuple(sorted(required_source_fields - source_map.keys()))
    if missing_source_fields:
        raise ValueError(
            "source_unit_ids omit supplied result fields: "
            + ", ".join(missing_source_fields)
            + "."
        )
    header = ElectronicEvaluationHeader(
        calculation.system.particle_ids,
        calculation.system.active_mask,
        units,
        convergence_,
        work_,
        status_,
        system_id=calculation.system.system_id,
        geometry_id=electronic_geometry_id(calculation.system, positions, cell_vectors),
        state_id=calculation.state.prepared_id,
        model_chemistry_id=calculation.model_chemistry.model_chemistry_id,
        task_id=calculation.task.task_id,
        provider_id=provider,
        source_unit_ids=sources,
        artifact_ids=artifact_ids,
    )
    supported_properties = {
        ElectronicProperty.ENERGY,
        ElectronicProperty.FORCES,
        ElectronicProperty.HESSIAN,
        ElectronicProperty.DIPOLE,
        ElectronicProperty.STRESS,
        ElectronicProperty.BAND_ENERGIES,
        ElectronicProperty.DENSITY_MATRIX,
        ElectronicProperty.POLARIZATION,
    }
    unsupported = tuple(
        value.value
        for value in calculation.task.properties
        if value not in supported_properties
    )
    if unsupported:
        raise ValueError(
            "Electronic result builder cannot represent requested properties: "
            + ", ".join(unsupported)
            + "."
        )
    if calculation.system.cell is None and cell_vectors is not None:
        raise ValueError("Finite electronic calculations do not accept cell_vectors.")
    task = calculation.task
    periodic_properties = {
        ElectronicProperty.STRESS: stress,
        ElectronicProperty.BAND_ENERGIES: band_energies,
        ElectronicProperty.DENSITY_MATRIX: density_matrices,
        ElectronicProperty.POLARIZATION: polarization,
    }
    periodic_requested = any(
        property_ in calculation.task.properties for property_ in periodic_properties
    )
    periodic_supplied = any(value is not None for value in periodic_properties.values())
    if periodic_requested or periodic_supplied:
        if calculation.system.cell is None:
            raise ValueError("Periodic electronic results require a periodic system.")
        if cell_vectors is None:
            raise ValueError("Periodic electronic results require cell_vectors.")
        if calculation.task.requires(
            ElectronicProperty.HESSIAN
        ) or calculation.task.requires(ElectronicProperty.DIPOLE):
            raise ValueError(
                "Periodic stress, band, density, or polarization results cannot "
                "be combined with molecular Hessian or dipole variants."
            )
        missing = tuple(
            property_.value
            for property_, value in periodic_properties.items()
            if calculation.task.requires(property_) and value is None
        )
        if missing:
            raise ValueError(
                "Periodic provider omitted requested properties: "
                + ", ".join(missing)
                + "."
            )
        if calculation.task.requires(ElectronicProperty.FORCES) and forces is None:
            raise ValueError("Requested periodic force result is incomplete.")
        return ElectronicPeriodicEvaluation(
            header,
            energy,
            forces=forces,
            stress=stress,
            band_energies=band_energies,
            density_matrices=density_matrices,
            polarization=polarization,
        )
    if task.requires(ElectronicProperty.HESSIAN):
        if forces is None or hessian is None:
            raise ValueError("Requested force/Hessian result is incomplete.")
        return ElectronicEnergyForceHessianEvaluation(header, energy, forces, hessian)
    if task.requires(ElectronicProperty.DIPOLE):
        if forces is None or dipole is None:
            raise ValueError("Requested force/dipole result is incomplete.")
        return ElectronicGroundStatePropertyEvaluation(header, energy, forces, dipole)
    if task.requires(ElectronicProperty.FORCES):
        if forces is None:
            raise ValueError("Requested force result is incomplete.")
        return ElectronicEnergyForceEvaluation(header, energy, forces)
    return ElectronicEnergyEvaluation(header, energy)


__all__ = [
    "ElectronicCalculationPlan",
    "electronic_geometry_id",
    "make_electronic_evaluation",
]

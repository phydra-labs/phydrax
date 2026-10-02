#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from ...discretization.fem import (
    FiniteElementHPTransaction,
    FiniteElementTopologyTransfer,
)
from ...linalg import ArraySpace
from ...meshing import (
    BisectionCompatibility,
    BisectionHierarchy,
    certify_cell_mesh,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from ...solver import (
    FiniteElementAcceptedState,
    FiniteElementHPTopologyResult,
    FiniteElementTopologyTransaction,
)
from ...typing import checked
from ._binary import (
    AllenCahnAcceptedState,
    CahnHilliardAcceptedState,
    PreparedAllenCahnFEM,
    PreparedCahnHilliardFEM,
)


class PhaseFieldAdaptivityEvidence(StrictModule):
    indicator: Array
    marked: Array
    source_mass: Array
    target_mass: Array
    mass_defect: Array
    source_energy: Array
    target_energy: Array
    energy_defect: Array
    resolution_passed: Array
    finite: Array
    successful: Array


class PhaseFieldAdaptiveEpoch(StrictModule, NonTrainableState):
    """One accepted phase-field epoch; ``hierarchy`` continues its bisection labels."""

    method: PreparedAllenCahnFEM | PreparedCahnHilliardFEM
    state: AllenCahnAcceptedState | CahnHilliardAcceptedState
    hierarchy: BisectionHierarchy | None
    epoch_index: int = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: PreparedAllenCahnFEM | PreparedCahnHilliardFEM,
        state: AllenCahnAcceptedState | CahnHilliardAcceptedState,
        epoch_index: int = 0,
        /,
        *,
        hierarchy: BisectionHierarchy | None = None,
    ) -> None:
        index = int(epoch_index)
        if not isinstance(method, (PreparedAllenCahnFEM, PreparedCahnHilliardFEM)):
            raise TypeError("Adaptive phase-field method has an invalid type.")
        if not isinstance(state, (AllenCahnAcceptedState, CahnHilliardAcceptedState)):
            raise TypeError("Adaptive phase-field state has an invalid type.")
        if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
            raise TypeError("hierarchy must be BisectionHierarchy or None.")
        if index < 0:
            raise ValueError("Adaptive phase-field epoch index must be nonnegative.")
        self.method = method
        self.state = state
        self.hierarchy = hierarchy
        self.epoch_index = index
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "phase-field-adaptive-epoch",
                "method": method.method_id,
                "epoch_index": index,
                "hierarchy": None if hierarchy is None else hierarchy.hierarchy_id,
            }
        )


def _integration_weights(discretization: Any, field_index: int, /) -> Array:
    space = discretization.field_spaces[field_index].vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError("Adaptive phase-field integration weights require ArraySpace.")
    result = jnp.zeros(space.shape, dtype=space.dtype)
    for block_index, geometry in enumerate(discretization.block_geometries[field_index]):
        dofs = discretization.dof_maps[field_index].cell_dofs[block_index]
        orientation = discretization.dof_maps[field_index].orientations[block_index]
        local = ein.contract(
            "cq,qi->ci", geometry.physical_weights, geometry.basis_values
        )
        result = result.at[dofs].add(local * orientation)
    return result


class PhaseFieldAdaptationResult(StrictModule, NonTrainableState):
    accepted: PhaseFieldAdaptiveEpoch
    candidate: PhaseFieldAdaptiveEpoch
    adaptation: MeshAdaptationResult | None
    transfer: FiniteElementTopologyTransfer | None
    evidence: PhaseFieldAdaptivityEvidence
    committed: Array


class PhaseFieldAdaptivityPlan(StrictModule, NonTrainableState):
    """Gradient-marked native bisection with transfer and acceptance evidence.

    ``coordinate_contract`` certifies the phase-field mesh for adaptation;
    ``compatibility`` decides incompatible initial bisection labels.
    """

    gradient_threshold: float = eqx.field(static=True)
    mass_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract
    compatibility: BisectionCompatibility = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        gradient_threshold: float,
        coordinate_contract: SpatialCoordinateContract,
        mass_tolerance: float = 1.0e-10,
        energy_tolerance: float = 1.0e-6,
        compatibility: BisectionCompatibility = BisectionCompatibility.REJECT,
    ) -> None:
        values = tuple(
            float(value)
            for value in (gradient_threshold, mass_tolerance, energy_tolerance)
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Phase-field adaptivity thresholds must be nonnegative.")
        if not isinstance(compatibility, BisectionCompatibility):
            raise TypeError("compatibility must be BisectionCompatibility.")
        self.gradient_threshold, self.mass_tolerance, self.energy_tolerance = values
        self.coordinate_contract = coordinate_contract
        self.compatibility = compatibility
        self.plan_id = canonical_fingerprint(
            {
                "kind": "phase-field-adaptivity-plan",
                "gradient_threshold": values[0],
                "mass_tolerance": values[1],
                "energy_tolerance": values[2],
                "coordinate_contract": coordinate_contract.spatial_id,
                "compatibility": compatibility.value,
            }
        )

    def indicator(self, epoch: PhaseFieldAdaptiveEpoch, /) -> Array:
        method = epoch.method
        state = epoch.state
        if isinstance(method, PreparedAllenCahnFEM) and isinstance(
            state, AllenCahnAcceptedState
        ):
            field = state.phase
            field_index = method.field_index
        elif isinstance(method, PreparedCahnHilliardFEM) and isinstance(
            state, CahnHilliardAcceptedState
        ):
            field = state.concentration
            field_index = method.concentration_index
        else:
            raise TypeError("Adaptive binary epoch method and state are incompatible.")
        values = []
        for block_index, geometry in enumerate(
            method.discretization.block_geometries[field_index]
        ):
            dofs = method.discretization.dof_maps[field_index].cell_dofs[block_index]
            orientation = method.discretization.dof_maps[field_index].orientations[
                block_index
            ]
            local = field[dofs] * orientation
            gradient = ein.contract("cqid,ci->cqd", geometry.physical_gradients, local)
            values.append(jnp.max(jnp.sqrt(jnp.sum(gradient**2, axis=-1)), axis=1))
        return jnp.concatenate(tuple(values))

    def refine(
        self,
        epoch: PhaseFieldAdaptiveEpoch,
        marked_cell_ids: ArrayLike | None = None,
        /,
    ) -> PhaseFieldAdaptationResult:
        method = epoch.method
        state = epoch.state
        mesh = method.discretization.mesh
        if any(block.cell_kind != "triangle" for block in mesh.blocks):
            raise ValueError("Local phase-field refinement requires T3 blocks.")
        indicator = self.indicator(epoch)
        cell_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        marked = (
            np.asarray(marked_cell_ids, dtype=np.int64)
            if marked_cell_ids is not None
            else cell_ids[np.asarray(indicator) > self.gradient_threshold]
        )
        adaptation = None
        if marked.size:
            adaptation = execute_mesh_adaptation(
                prepare_mesh_adaptation(
                    certify_cell_mesh(mesh, self.coordinate_contract),
                    MarkedMeshAdaptation(marked, hierarchy=epoch.hierarchy),
                    policy=MeshAdaptationPolicy(
                        MeshAdaptationRoute.NATIVE_BISECTION,
                        compatibility=self.compatibility,
                    ),
                )
            )
        if adaptation is None or adaptation.transition is None:
            evidence = PhaseFieldAdaptivityEvidence(
                indicator,
                jnp.zeros(indicator.shape, dtype=jnp.bool_),
                state.mass,
                state.mass,
                jnp.asarray(0.0),
                state.energy,
                state.energy,
                jnp.asarray(0.0),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
            )
            return PhaseFieldAdaptationResult(
                epoch, epoch, adaptation, None, evidence, jnp.asarray(False)
            )
        target_mesh = adaptation.target.mesh
        transfer = adaptation.transfer
        if transfer is None:
            raise RuntimeError(
                "Native bisection adaptation did not provide its required transfer."
            )
        if not isinstance(adaptation.hierarchy, BisectionHierarchy):
            raise RuntimeError(
                "Native bisection adaptation did not provide a bisection hierarchy."
            )
        element = lagrange_element("triangle", 1)
        if isinstance(method, PreparedAllenCahnFEM):
            if not isinstance(state, AllenCahnAcceptedState):
                raise TypeError("Allen-Cahn adaptive epoch state is incompatible.")
            discretization = FiniteElementPlan(
                target_mesh,
                FiniteElementFieldSpec(method.field_name, element),
            ).prepare()
            successor_noise = (
                None if method.noise is None else method.noise.transfer(transfer)
            )
            candidate_method = method.plan.prepare(
                discretization, method.field_name, noise=successor_noise
            )
            phase = transfer.apply(state.phase)
            candidate_state = candidate_method.initialize(phase)
        elif isinstance(method, PreparedCahnHilliardFEM):
            if not isinstance(state, CahnHilliardAcceptedState):
                raise TypeError("Cahn-Hilliard adaptive epoch state is incompatible.")
            discretization = FiniteElementPlan(
                target_mesh,
                (
                    FiniteElementFieldSpec(method.concentration_field, element),
                    FiniteElementFieldSpec(method.chemical_field, element),
                ),
            ).prepare()
            successor_noise = (
                None
                if method.noise is None
                else method.noise.transfer(
                    transfer,
                    conservation_weights=_integration_weights(discretization, 0),
                )
            )
            candidate_method = method.plan.prepare(
                discretization,
                method.concentration_field,
                method.chemical_field,
                noise=successor_noise,
            )
            concentration = transfer.apply(state.concentration)
            chemical = transfer.apply(state.chemical_potential)
            initialized = candidate_method.initialize(
                concentration, chemical_potential=chemical
            )
            candidate_state = CahnHilliardAcceptedState(
                initialized.concentration,
                initialized.chemical_potential,
                state.reference_mass,
                state.cumulative_mass_source,
                initialized.mass,
                initialized.energy,
            )
        else:
            raise TypeError("Unsupported adaptive phase-field method.")
        candidate_epoch = PhaseFieldAdaptiveEpoch(
            candidate_method,
            candidate_state,
            epoch.epoch_index + 1,
            hierarchy=adaptation.hierarchy,
        )
        mass_defect = jnp.abs(candidate_state.mass - state.mass)
        energy_defect = candidate_state.energy - state.energy
        finite = jnp.isfinite(mass_defect) & jnp.isfinite(energy_defect)
        mass_required = isinstance(method, PreparedCahnHilliardFEM)
        successful = (
            finite
            & candidate_method.resolution.passed
            & ((not mass_required) | (mass_defect <= self.mass_tolerance))
            & (energy_defect <= self.energy_tolerance)
        )
        selected = candidate_epoch if bool(successful) else epoch
        evidence = PhaseFieldAdaptivityEvidence(
            indicator,
            jnp.isin(jnp.asarray(cell_ids), jnp.asarray(marked)),
            state.mass,
            candidate_state.mass,
            mass_defect,
            state.energy,
            candidate_state.energy,
            energy_defect,
            jnp.asarray(candidate_method.resolution.passed),
            finite,
            successful,
        )
        return PhaseFieldAdaptationResult(
            selected,
            candidate_epoch,
            adaptation,
            transfer,
            evidence,
            successful,
        )


class PhaseFieldHPTransactionPlan(StrictModule, NonTrainableState):
    """Phase-field specialization of the native atomic hp transaction."""

    transaction: FiniteElementTopologyTransaction
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, mass_tolerance: float = 1.0e-10) -> None:
        tolerance = float(mass_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("hp phase-field mass tolerance must be nonnegative.")

        def certify(
            epoch: Any, fields: Any, materials: Any, transaction: Any, args: Any
        ) -> Any:
            del epoch, materials, args
            finite = jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in fields))
            )
            return finite & (jnp.abs(transaction.conservation_error) <= tolerance)

        self.transaction = FiniteElementTopologyTransaction(
            certify,
            transaction_id="phase-field-hp-topology-transaction",
        )
        self.plan_id = canonical_fingerprint(
            {"kind": "phase-field-hp-transaction-plan", "tolerance": tolerance}
        )

    def promote(
        self,
        accepted: FiniteElementAcceptedState,
        transaction: FiniteElementHPTransaction,
        args: object = None,
        /,
    ) -> FiniteElementHPTopologyResult:
        return self.transaction.execute_hp(accepted, transaction, args)


__all__ = [
    "PhaseFieldAdaptationResult",
    "PhaseFieldAdaptivityEvidence",
    "PhaseFieldAdaptivityPlan",
    "PhaseFieldAdaptiveEpoch",
    "PhaseFieldHPTransactionPlan",
]

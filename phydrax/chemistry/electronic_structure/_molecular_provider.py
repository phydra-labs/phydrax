#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed, non-differentiable RHF energy/analytic-force and PES integration.

Construct ``MolecularRHFProvider(plan, model.model_chemistry_id)`` and pass
``numerical=provider.numerical`` to ``ElectronicCalculationPlan``. The numerical
identity binds the actual orbital basis, integral source (including auxiliary
basis, conditioning and screening), and SCF policies, rather than a provider
label. Prepare an energy/force calculation and wrap it in
``ElectronicPotentialEnergySurface`` for molecular geometry optimization.

Only finite, all-electron, real-FP64, integer-occupation restricted HF is
admitted. Native exact integrals and cuEST density fitting supply their own
same-definition contracted analytic gradients; fixed factors and screened
native direct J/K do not. No arbitrary AD, batching or analytic Hessian is
advertised. Backend exceptions propagate without fabricating a result.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...typing import checked
from ...units import BOHR, conversion_factor, derived_unit, HARTREE
from .._calculation import ElectronicCalculationPlan, make_electronic_evaluation
from .._method import HartreeFockMethodPlan
from .._model import ElectronicMethodFamily, ElectronicReferenceKind
from .._numerical import (
    ElectronicDerivativeRoute,
    ElectronicNumericalPlan,
    IntegralRepresentationKind,
)
from .._provider import (
    AbstractElectronicProvider,
    AbstractPreparedElectronicCalculation,
    ElectronicCapabilityError,
    ElectronicProviderCapabilities,
)
from .._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEvaluation,
    ElectronicWorkEvidence,
)
from .._task import ElectronicProperty
from ._mean_field import ElectronicOccupationKind, InitialGuessKind
from ._molecular_hf import MolecularHartreeFockPlan


_ATOMIC_FORCE_UNIT = derived_unit("hartree/bohr", ((HARTREE, 1), (BOHR, -1)))


class PreparedMolecularRHFCalculation(AbstractPreparedElectronicCalculation):
    """A prepared RHF calculation accepting dynamic positions in system units."""

    calculation: ElectronicCalculationPlan
    plan: MolecularHartreeFockPlan
    capabilities: ElectronicProviderCapabilities
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        provider: MolecularRHFProvider,
        /,
    ) -> None:
        provider._require(calculation)
        self.calculation = calculation
        self.plan = provider.plan
        self.capabilities = provider.capabilities
        self.provider_id = provider.provider_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-molecular-rhf-calculation",
                "calculation": calculation.calculation_id,
                "plan": provider.plan.plan_id,
                "provider": provider.provider_id,
            }
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> ElectronicEvaluation:
        if cell_vectors is not None:
            raise ElectronicCapabilityError(
                "Molecular RHF does not accept periodic cell vectors."
            )
        host = np.asarray(positions)
        expected = (self.calculation.system.particle_ids.size, 3)
        if host.shape != expected or host.dtype.kind not in "fiu":
            raise ValueError(f"positions must be a real array of shape {expected}.")
        if not np.all(np.isfinite(host)):
            raise ValueError("Molecular RHF positions must be finite.")
        coordinate = jnp.asarray(host, dtype=jnp.float64)
        if coordinate.dtype != np.dtype("float64"):
            raise ElectronicCapabilityError("Molecular RHF requires real FP64 execution.")
        units = self.calculation.system.units
        length_to_bohr = float(conversion_factor(units.scale.length_unit, BOHR))
        energy_from_hartree = float(conversion_factor(HARTREE, units.scale.energy_unit))
        coordinate_bohr = coordinate * length_to_bohr
        state, geometry = self.plan._solve_with_geometry(coordinate_bohr)
        definition = self.plan.prepared_integrals.definition_id
        if state.integral_definition_id != definition:
            raise ValueError("Molecular RHF state has a different integral definition.")
        evidence = state.evidence
        forces = None
        force_requested = self.calculation.task.requires(ElectronicProperty.FORCES)
        successful = bool(evidence.successful)
        if force_requested:
            gradient = self.plan._gradient_at_geometry(coordinate_bohr, state, geometry)
            forces = gradient.forces * (energy_from_hartree * length_to_bohr)
            successful = successful and bool(gradient.successful)
        energy = state.total_energy * energy_from_hartree
        finite = bool(np.isfinite(np.asarray(energy))) and (
            forces is None or bool(np.all(np.isfinite(np.asarray(forces))))
        )
        if not finite:
            status = ElectronicCalculationStatus.NONFINITE_RESULT
            message = "molecular-rhf-nonfinite-result"
        elif not successful:
            status = ElectronicCalculationStatus.ELECTRONIC_NOT_CONVERGED
            message = (
                "molecular-rhf-not-stationary"
                if force_requested
                else "molecular-rhf-not-converged"
            )
        else:
            status = ElectronicCalculationStatus.SUCCESS
            message = (
                "molecular-rhf-stationary-analytic-forces"
                if force_requested
                else "molecular-rhf-converged"
            )
        return make_electronic_evaluation(
            self.calculation,
            self.provider_id,
            coordinate,
            energy,
            forces=forces,
            convergence=ElectronicConvergenceEvidence(
                successful and finite,
                iterations=evidence.iterations,
                energy_residual=evidence.energy_residual,
                density_residual=evidence.density_residual,
                message=message,
            ),
            work=ElectronicWorkEvidence(
                energy_evaluations=1,
                force_evaluations=int(force_requested),
            ),
            status=status,
            source_unit_ids=(
                ("energy", HARTREE.unit_id),
                ("length", BOHR.unit_id),
                *((("forces", _ATOMIC_FORCE_UNIT.unit_id),) if force_requested else ()),
            ),
            # Named binding survives archives and prevents lifecycle relabeling
            # under another numerical definition, even with the same model.
            artifact_ids=(
                "molecular-rhf-calculation:" + self.calculation.calculation_id,
                state.state_id,
                definition,
                self.plan.plan_id,
                self.calculation.numerical.numerical_id,
            ),
        )


class MolecularRHFProvider(AbstractElectronicProvider):
    """Bind a molecular SCF plan to explicit model and numerical admission.

    ``numerical`` is the required calculation numerical plan. In particular,
    generic density-fitting/provider-native aliases cannot substitute for the
    actual auxiliary basis, metric policy, screening and resource definition.
    """

    plan: MolecularHartreeFockPlan
    numerical: ElectronicNumericalPlan
    capabilities: ElectronicProviderCapabilities
    model_chemistry_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: MolecularHartreeFockPlan, model_chemistry_id: str, /
    ) -> None:
        identifier = str(model_chemistry_id).strip()
        if not identifier:
            raise ValueError("model_chemistry_id must be non-empty.")
        if plan.reference is not ElectronicReferenceKind.RESTRICTED:
            raise ElectronicCapabilityError(
                "MolecularRHFProvider supports restricted HF only."
            )
        if plan.occupations.kind is not ElectronicOccupationKind.INTEGER:
            raise ElectronicCapabilityError("Molecular RHF requires integer occupations.")
        if plan.guess.kind in (InitialGuessKind.EXPLICIT, InitialGuessKind.PROJECTED):
            raise ElectronicCapabilityError(
                "MolecularRHFProvider does not accept explicit initial-guess contexts."
            )
        system = plan.system
        if system.cell is not None:
            raise ElectronicCapabilityError(
                "Molecular RHF requires a nonperiodic system without a cell."
            )
        if (
            not np.all(np.asarray(system.active_mask))
            or not np.all(np.asarray(system.element_mask))
            or np.any(np.asarray(system.atomic_numbers) <= 0)
        ):
            raise ElectronicCapabilityError(
                "Molecular RHF requires active all-electron nuclei only."
            )
        if system.coordinate_dtype != "float64" or any(
            np.asarray(value).dtype != np.dtype("float64")
            for value in (
                plan.basis.exponents,
                plan.basis.normalized_coefficients,
                plan.basis.transformation,
            )
        ):
            raise ElectronicCapabilityError(
                "Molecular RHF requires real FP64 coordinates and basis data."
            )
        source = plan.integrals.source
        if source is None:
            representation = IntegralRepresentationKind.DENSE
            execution = "host"
        else:
            from ...operators.quantum.gaussian import CuestIntegralPlan

            if not isinstance(source, CuestIntegralPlan):
                raise ElectronicCapabilityError(
                    "MolecularRHFProvider analytic forces require native exact or "
                    "cuEST integrals; fixed factors and native direct J/K are not admitted."
                )
            representation = IntegralRepresentationKind.GAUSSIAN_DENSITY_FITTING
            execution = "device"
        numerical = ElectronicNumericalPlan(
            integrals=representation,
            derivative_route=ElectronicDerivativeRoute.ANALYTIC_LAGRANGIAN,
            component_plan_ids=(plan.plan_id, plan.integrals.definition_id(plan.basis)),
        )
        capabilities = ElectronicProviderCapabilities.molecular_ground_state(
            (ElectronicProperty.ENERGY, ElectronicProperty.FORCES),
            (ElectronicReferenceKind.RESTRICTED,),
            families=(ElectronicMethodFamily.HARTREE_FOCK,),
            spin_multiplicity=False,
            conservative_forces=True,
            differentiable=False,
            batching=False,
            execution=execution,
            concurrency="serial",
        )
        self.plan = plan
        self.numerical = numerical
        self.capabilities = capabilities
        self.model_chemistry_id = identifier
        self.provider_id = canonical_fingerprint(
            {
                "kind": "molecular-rhf-provider",
                "plan": plan.plan_id,
                "model_chemistry": identifier,
                "numerical": numerical.numerical_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def _require(self, calculation: ElectronicCalculationPlan, /) -> None:
        self.capabilities.require(calculation)
        if calculation.system.system_id != self.plan.system.system_id:
            raise ElectronicCapabilityError(
                "Molecular RHF plan belongs to another system."
            )
        if calculation.state.prepared_id != self.plan.sector.prepared_id:
            raise ElectronicCapabilityError(
                "Molecular RHF plan belongs to another electronic sector."
            )
        model = calculation.model_chemistry
        if model.model_chemistry_id != self.model_chemistry_id:
            raise ElectronicCapabilityError(
                "Molecular RHF provider is bound to another model chemistry."
            )
        if not isinstance(model.method, HartreeFockMethodPlan):
            raise ElectronicCapabilityError(
                "Molecular RHF requires the typed Hartree--Fock method."
            )
        if (
            model.environment.kind != "vacuum"
            or model.environment.parameter_ids
            or model.correction_ids
            or model.relativistic_id is not None
            or model.model_artifact is not None
        ):
            raise ElectronicCapabilityError(
                "Molecular RHF supports uncorrected nonrelativistic all-electron vacuum HF only."
            )
        basis = model.basis
        if basis is None or basis.source_id != self.plan.basis.plan.source_id:
            raise ElectronicCapabilityError(
                "Model basis source differs from the actual orbital basis."
            )
        conventions = {
            "spherical" if shell.representation.value == "real-spherical" else "cartesian"
            for shell in self.plan.basis.plan.shells
        }
        if basis.convention != "provider-native" and conventions != {basis.convention}:
            raise ElectronicCapabilityError(
                "Model basis convention differs from the actual orbital basis."
            )
        artifact = self.plan.basis.plan.source_artifact_id
        if (None if basis.artifact is None else basis.artifact.manifest_id) != artifact:
            raise ElectronicCapabilityError(
                "Model basis artifact differs from the actual orbital basis."
            )
        if calculation.numerical.numerical_id != self.numerical.numerical_id:
            raise ElectronicCapabilityError(
                "Molecular RHF numerical definition differs; use numerical=provider.numerical."
            )
        precision = calculation.precision
        if any(
            value != "float64"
            for value in (
                precision.coordinate_dtype,
                precision.compute_dtype,
                precision.reduction_dtype,
                precision.output_dtype,
            )
        ):
            raise ElectronicCapabilityError(
                "Molecular RHF requires an entirely real FP64 precision policy."
            )

    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> PreparedMolecularRHFCalculation:
        return PreparedMolecularRHFCalculation(calculation, self)


__all__ = ["MolecularRHFProvider", "PreparedMolecularRHFCalculation"]

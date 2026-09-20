#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provenance-retaining postprocessors for supplied diagonal GW and BSE data."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import Bisection, NonlinearTermination, scalar_root, ScalarRootProblem
from ...units import ENERGY, UnitDefinition
from ..excited import ElectronicManifoldResult, RandomPhaseApproximationPlan
from ..excited._tda import ExcitedStateManifoldPlan, TammDancoffPlan
from ._source import PeriodicProvenanceManifest


DiagonalSelfEnergy = Callable[[int, Array], Array]
BSEApproximation = Literal["tda", "full"]


class GWQuasiparticleEvidence(StrictModule, NonTrainableState):
    root_residuals: Array
    successful_roots: Array
    successful: Array
    residual_tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_residuals: ArrayLike,
        successful_roots: ArrayLike,
        residual_tolerance: float,
        /,
    ):
        residuals = jnp.asarray(root_residuals)
        roots = jnp.asarray(successful_roots, dtype=jnp.bool_)
        tolerance = float(residual_tolerance)
        if (
            residuals.ndim != 1
            or roots.shape != residuals.shape
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("GW root evidence arrays or tolerance are invalid.")
        successful = (
            jnp.all(roots)
            & jnp.all(jnp.isfinite(residuals))
            & jnp.all(residuals >= 0.0)
            & jnp.all(residuals <= tolerance)
        )
        self.root_residuals = residuals
        self.successful_roots = roots
        self.successful = successful
        self.residual_tolerance = tolerance
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "gw-quasiparticle-evidence",
                "residual_tolerance": tolerance.hex(),
                "successful": bool(successful),
                "arrays": array_tree_fingerprint(
                    {"residuals": np.asarray(residuals), "roots": np.asarray(roots)}
                ),
            }
        )


class GWQuasiparticleResult(StrictModule, NonTrainableState):
    mean_field_energies: Array
    quasiparticle_energies: Array
    renormalization_factors: Array
    evidence: GWQuasiparticleEvidence
    successful: Array
    energy_unit: UnitDefinition
    provider_id: str = eqx.field(static=True)
    self_energy_definition_id: str = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_field_energies: ArrayLike,
        quasiparticle_energies: ArrayLike,
        renormalization_factors: ArrayLike,
        evidence: GWQuasiparticleEvidence,
        energy_unit: UnitDefinition,
        /,
        *,
        provider_id: str,
        self_energy_definition_id: str,
        source_manifest_id: str,
        plan_id: str,
    ):
        mean_field = jnp.asarray(mean_field_energies)
        quasiparticle = jnp.asarray(quasiparticle_energies, dtype=mean_field.dtype)
        factors = jnp.asarray(renormalization_factors, dtype=mean_field.real.dtype)
        if (
            mean_field.ndim != 1
            or quasiparticle.shape != mean_field.shape
            or factors.shape != mean_field.shape
        ):
            raise ValueError("GW quasiparticle arrays do not align.")
        if not isinstance(evidence, GWQuasiparticleEvidence):
            raise TypeError("evidence must be GWQuasiparticleEvidence.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("GW energy_unit must have energy dimension.")
        identifiers = tuple(
            str(value).strip()
            for value in (
                provider_id,
                self_energy_definition_id,
                source_manifest_id,
                plan_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError(
                "GW provider, kernel, source, and plan identities are required."
            )
        finite = (
            jnp.all(jnp.isfinite(mean_field))
            & jnp.all(jnp.isfinite(quasiparticle))
            & jnp.all(jnp.isfinite(factors))
        )
        self.mean_field_energies = mean_field
        self.quasiparticle_energies = quasiparticle
        self.renormalization_factors = factors
        self.evidence = evidence
        self.successful = evidence.successful & finite
        self.energy_unit = energy_unit
        (
            self.provider_id,
            self.self_energy_definition_id,
            self.source_manifest_id,
            self.plan_id,
        ) = identifiers
        self.result_id = canonical_fingerprint(
            {
                "kind": "supplied-diagonal-gw-result",
                "provider": self.provider_id,
                "self_energy_definition": self.self_energy_definition_id,
                "source_manifest": self.source_manifest_id,
                "plan": self.plan_id,
                "energy_unit": energy_unit.unit_id,
                "evidence": evidence.evidence_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "mean_field": np.asarray(mean_field),
                        "quasiparticle": np.asarray(quasiparticle),
                        "renormalization": np.asarray(factors),
                    }
                ),
            }
        )


class DiagonalGWPlan(StrictModule, NonTrainableState):
    """Solve quasiparticle roots from a caller-supplied diagonal self-energy kernel."""

    mean_field_energies: Array
    mean_field_xc_expectations: Array
    brackets: Array
    self_energy: DiagonalSelfEnergy = eqx.field(static=True)
    self_energy_definition_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    source_manifest: PeriodicProvenanceManifest
    residual_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_field_energies: ArrayLike,
        mean_field_xc_expectations: ArrayLike,
        brackets: ArrayLike,
        self_energy: DiagonalSelfEnergy,
        self_energy_definition_id: str,
        provider_id: str,
        source_manifest: PeriodicProvenanceManifest,
        energy_unit: UnitDefinition,
        /,
        *,
        residual_tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
    ):
        energies = jnp.asarray(mean_field_energies)
        xc = jnp.asarray(mean_field_xc_expectations, dtype=energies.dtype)
        brackets_ = jnp.asarray(brackets, dtype=energies.real.dtype)
        definition = str(self_energy_definition_id).strip()
        provider = str(provider_id).strip()
        tolerance = float(residual_tolerance)
        maximum = int(maximum_iterations)
        if (
            energies.ndim != 1
            or energies.size == 0
            or xc.shape != energies.shape
            or brackets_.shape != (energies.size, 2)
            or not bool(jnp.all(jnp.isfinite(energies)))
            or not bool(jnp.all(jnp.isfinite(xc)))
            or not bool(jnp.all(jnp.isfinite(brackets_)))
            or bool(jnp.any(brackets_[:, 1] <= brackets_[:, 0]))
            or not callable(self_energy)
            or not definition
            or not provider
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or maximum <= 0
        ):
            raise ValueError(
                "Supplied GW energies, brackets, provider kernel, or solve policy is invalid."
            )
        if not isinstance(source_manifest, PeriodicProvenanceManifest):
            raise TypeError("GW source_manifest must be PeriodicProvenanceManifest.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("GW energy_unit must have energy dimension.")
        self.mean_field_energies = energies
        self.mean_field_xc_expectations = xc
        self.brackets = brackets_
        self.self_energy = self_energy
        self.self_energy_definition_id = definition
        self.provider_id = provider
        self.source_manifest = source_manifest
        self.residual_tolerance = tolerance
        self.maximum_iterations = maximum
        self.energy_unit = energy_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "supplied-diagonal-gw-plan",
                "provider": provider,
                "self_energy_definition": definition,
                "source_manifest": source_manifest.manifest_id,
                "residual_tolerance": tolerance.hex(),
                "maximum_iterations": maximum,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "mean_field": np.asarray(energies),
                        "xc": np.asarray(xc),
                        "brackets": np.asarray(brackets_),
                    }
                ),
            }
        )

    def evaluate(self, /) -> GWQuasiparticleResult:
        values = []
        factors = []
        residuals = []
        successful = []
        for index in range(self.mean_field_energies.size):
            state_index = index

            def equation(energy, _, state_index=state_index):
                return (
                    energy
                    - self.mean_field_energies[state_index]
                    - jnp.real(self.self_energy(state_index, energy))
                    + self.mean_field_xc_expectations[state_index]
                )

            result = scalar_root(
                ScalarRootProblem(
                    equation,
                    bracket=(
                        self.brackets[state_index, 0],
                        self.brackets[state_index, 1],
                    ),
                    problem_id=f"supplied-gw:{self.plan_id}:{state_index}",
                ),
                method=Bisection(),
                termination=NonlinearTermination(
                    absolute_residual=self.residual_tolerance,
                    relative_residual=0.0,
                    maximum_steps=self.maximum_iterations,
                    maximum_evaluations=2 * self.maximum_iterations + 4,
                    maximum_linear_iterations=1,
                ),
            )
            derivative = jax.grad(
                lambda energy, state_index=state_index: jnp.real(
                    self.self_energy(state_index, energy)
                )
            )(result.root)
            values.append(result.root)
            factors.append(1.0 / (1.0 - derivative))
            residuals.append(jnp.abs(equation(result.root, None)))
            successful.append(result.successful)
        residual_array = jnp.stack(tuple(residuals))
        evidence = GWQuasiparticleEvidence(
            residual_array,
            jnp.stack(tuple(successful)),
            self.residual_tolerance,
        )
        return GWQuasiparticleResult(
            self.mean_field_energies,
            jnp.stack(tuple(values)),
            jnp.stack(tuple(factors)),
            evidence,
            self.energy_unit,
            provider_id=self.provider_id,
            self_energy_definition_id=self.self_energy_definition_id,
            source_manifest_id=self.source_manifest.manifest_id,
            plan_id=self.plan_id,
        )


class BSEPostprocessEvidence(StrictModule, NonTrainableState):
    eigenpair_residuals: Array
    maximum_eigenpair_residual: Array
    successful: Array
    residual_tolerance: float = eqx.field(static=True)
    transition_order_id: str = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        eigenpair_residuals: ArrayLike,
        successful: ArrayLike,
        transition_order_id: str,
        source_manifest_id: str,
        provider_id: str,
        residual_tolerance: float,
        /,
    ):
        residuals = jnp.asarray(eigenpair_residuals).reshape((-1,))
        tolerance = float(residual_tolerance)
        identifiers = tuple(
            str(value).strip()
            for value in (transition_order_id, source_manifest_id, provider_id)
        )
        if (
            residuals.size == 0
            or any(not value for value in identifiers)
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "BSE evidence residuals, tolerance, and provenance are required."
            )
        maximum = jnp.max(jnp.abs(residuals), initial=0.0)
        admitted = (
            jnp.asarray(successful, dtype=jnp.bool_).reshape(())
            & jnp.all(jnp.isfinite(residuals))
            & jnp.all(residuals >= 0.0)
            & (maximum <= tolerance)
        )
        self.eigenpair_residuals = residuals
        self.maximum_eigenpair_residual = maximum
        self.successful = admitted
        self.residual_tolerance = tolerance
        (
            self.transition_order_id,
            self.source_manifest_id,
            self.provider_id,
        ) = identifiers
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "supplied-bse-evidence",
                "transition_order": self.transition_order_id,
                "source_manifest": self.source_manifest_id,
                "provider": self.provider_id,
                "residual_tolerance": tolerance.hex(),
                "successful": bool(admitted),
                "residuals": array_tree_fingerprint(np.asarray(residuals)),
            }
        )


class BSEPostprocessResult(StrictModule, NonTrainableState):
    manifold: ElectronicManifoldResult
    evidence: BSEPostprocessEvidence
    successful: Array
    approximation: BSEApproximation = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifold: ElectronicManifoldResult,
        evidence: BSEPostprocessEvidence,
        approximation: BSEApproximation,
        plan_id: str,
        /,
    ):
        if not isinstance(manifold, ElectronicManifoldResult):
            raise TypeError("manifold must be ElectronicManifoldResult.")
        if not isinstance(evidence, BSEPostprocessEvidence):
            raise TypeError("evidence must be BSEPostprocessEvidence.")
        if approximation not in ("tda", "full"):
            raise ValueError("BSE approximation must be tda or full.")
        plan = str(plan_id).strip()
        if not plan:
            raise ValueError("BSE plan_id must be non-empty.")
        self.manifold = manifold
        self.evidence = evidence
        self.successful = manifold.successful & evidence.successful
        self.approximation = approximation
        self.plan_id = plan
        self.result_id = canonical_fingerprint(
            {
                "kind": "supplied-bse-postprocess-result",
                "approximation": approximation,
                "plan": plan,
                "manifold": manifold.result_id,
                "evidence": evidence.evidence_id,
                "successful": bool(self.successful),
            }
        )


class BetheSalpeterPlan(StrictModule, NonTrainableState):
    """Bounded eigensystem postprocessing for a provider-supplied transition kernel."""

    transition_energies: Array
    screened_direct: Array
    bare_exchange: Array
    coupling: Array
    transition_dipoles: Array
    ground_state_energy: Array
    transition_ids: tuple[str, ...] = eqx.field(static=True)
    transition_order_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    source_manifest: PeriodicProvenanceManifest
    maximum_transitions: int = eqx.field(static=True)
    eigenpair_tolerance: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    transition_dipole_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_ids: tuple[str, ...],
        transition_energies: ArrayLike,
        screened_direct: ArrayLike,
        bare_exchange: ArrayLike,
        transition_dipoles: ArrayLike,
        ground_state_energy: ArrayLike,
        energy_unit: UnitDefinition,
        transition_dipole_unit: UnitDefinition,
        provider_id: str,
        source_manifest: PeriodicProvenanceManifest,
        /,
        *,
        coupling: ArrayLike | None = None,
        maximum_transitions: int = 4096,
        eigenpair_tolerance: float = 1.0e-9,
    ):
        identifiers = tuple(str(value).strip() for value in transition_ids)
        transition_input = jnp.asarray(transition_energies)
        transitions = jnp.real(transition_input)
        direct_input = jnp.asarray(screened_direct)
        exchange_input = jnp.asarray(bare_exchange)
        coupling_input = (
            jnp.zeros_like(direct_input) if coupling is None else jnp.asarray(coupling)
        )
        kernel_dtype = jnp.result_type(
            transitions.dtype,
            direct_input.dtype,
            exchange_input.dtype,
            coupling_input.dtype,
        )
        direct = direct_input.astype(kernel_dtype)
        exchange = exchange_input.astype(kernel_dtype)
        coupling_ = coupling_input.astype(kernel_dtype)
        dipoles = jnp.asarray(transition_dipoles).astype(
            jnp.result_type(kernel_dtype, jnp.asarray(transition_dipoles).dtype)
        )
        count = transitions.size
        provider = str(provider_id).strip()
        capacity = int(maximum_transitions)
        eigenpair_tolerance_ = float(eigenpair_tolerance)
        ground = jnp.asarray(ground_state_energy, dtype=transitions.dtype).reshape(())
        if (
            not identifiers
            or len(identifiers) != count
            or any(not value for value in identifiers)
            or len(set(identifiers)) != count
            or count > capacity
            or capacity <= 0
            or not isfinite(eigenpair_tolerance_)
            or eigenpair_tolerance_ <= 0.0
            or transitions.shape != (count,)
            or direct.shape != (count, count)
            or exchange.shape != direct.shape
            or coupling_.shape != direct.shape
            or dipoles.shape != (count, 3)
            or bool(jnp.any(jnp.abs(jnp.imag(transition_input)) > 1.0e-12))
            or bool(jnp.any(transitions <= 0.0))
            or not np.allclose(
                np.asarray(direct),
                np.asarray(jnp.conj(direct.T)),
                rtol=0.0,
                atol=1.0e-10,
            )
            or not np.allclose(
                np.asarray(exchange),
                np.asarray(jnp.conj(exchange.T)),
                rtol=0.0,
                atol=1.0e-10,
            )
            or not np.allclose(
                np.asarray(coupling_),
                np.asarray(coupling_.T),
                rtol=0.0,
                atol=1.0e-10,
            )
            or not bool(jnp.all(jnp.isfinite(direct)))
            or not bool(jnp.all(jnp.isfinite(exchange)))
            or not bool(jnp.all(jnp.isfinite(coupling_)))
            or not bool(jnp.all(jnp.isfinite(dipoles)))
            or not bool(jnp.isfinite(ground))
            or not provider
        ):
            raise ValueError(
                "Supplied BSE transitions, ordered identities, kernels, or bounds are invalid."
            )
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("BSE energy_unit must have energy dimension.")
        if not isinstance(transition_dipole_unit, UnitDefinition):
            raise TypeError("transition_dipole_unit must be UnitDefinition.")
        if not isinstance(source_manifest, PeriodicProvenanceManifest):
            raise TypeError("BSE source_manifest must be PeriodicProvenanceManifest.")
        self.transition_energies = transitions
        self.screened_direct = direct
        self.bare_exchange = exchange
        self.coupling = coupling_
        self.transition_dipoles = dipoles
        self.ground_state_energy = ground
        self.transition_ids = identifiers
        self.provider_id = provider
        self.source_manifest = source_manifest
        self.maximum_transitions = capacity
        self.eigenpair_tolerance = eigenpair_tolerance_
        self.energy_unit = energy_unit
        self.transition_dipole_unit = transition_dipole_unit
        self.transition_order_id = canonical_fingerprint(
            {
                "kind": "bse-transition-order",
                "transition_ids": list(identifiers),
            }
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "supplied-bethe-salpeter-plan",
                "provider": provider,
                "source_manifest": source_manifest.manifest_id,
                "transition_order": self.transition_order_id,
                "maximum_transitions": capacity,
                "eigenpair_tolerance": eigenpair_tolerance_.hex(),
                "energy_unit": energy_unit.unit_id,
                "transition_dipole_unit": transition_dipole_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "transitions": np.asarray(transitions),
                        "direct": np.asarray(direct),
                        "exchange": np.asarray(exchange),
                        "coupling": np.asarray(coupling_),
                        "dipoles": np.asarray(dipoles),
                        "ground_state_energy": np.asarray(ground),
                    }
                ),
            }
        )

    @property
    def resonant_matrix(self) -> Array:
        return (
            jnp.diag(self.transition_energies) + self.bare_exchange - self.screened_direct
        )

    def _result(
        self,
        manifold: ElectronicManifoldResult,
        approximation: BSEApproximation,
        /,
    ) -> BSEPostprocessResult:
        evidence = BSEPostprocessEvidence(
            manifold.residuals,
            manifold.successful,
            self.transition_order_id,
            self.source_manifest.manifest_id,
            self.provider_id,
            self.eigenpair_tolerance,
        )
        return BSEPostprocessResult(manifold, evidence, approximation, self.plan_id)

    def tda(self, root_count: int, /) -> BSEPostprocessResult:
        manifold = TammDancoffPlan(
            ExcitedStateManifoldPlan(root_count, spin_sector="singlet"),
            self.resonant_matrix,
            self.transition_dipoles,
            self.ground_state_energy,
            self.energy_unit,
            self.transition_dipole_unit,
        ).solve()
        return self._result(manifold, "tda")

    def full(self, root_count: int, /) -> BSEPostprocessResult:
        manifold = RandomPhaseApproximationPlan(
            self.resonant_matrix,
            self.coupling,
            self.transition_dipoles,
            self.ground_state_energy,
            root_count,
            "bse-supplied-kernel",
            self.energy_unit,
            self.transition_dipole_unit,
            spin_sector="singlet",
        ).solve()
        return self._result(manifold, "full")


__all__ = [
    "BSEApproximation",
    "BSEPostprocessEvidence",
    "BSEPostprocessResult",
    "BetheSalpeterPlan",
    "DiagonalGWPlan",
    "DiagonalSelfEnergy",
    "GWQuasiparticleEvidence",
    "GWQuasiparticleResult",
]

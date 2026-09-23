#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""ADC, EOM, and active-space manifold provider contracts and adapters."""

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import (
    BOHR,
    conversion_factor,
    derived_unit,
    ELEMENTARY_CHARGE,
    HARTREE,
    UnitDefinition,
)
from ..electronic_structure.correlation import CASCIResult
from ._manifold import ElectronicManifoldResult
from ._representation import BiorthogonalStateRepresentation, CIStateRepresentation


class CorrelatedManifoldPlan(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    sector: str = eqx.field(static=True)
    root_count: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        sector: str,
        root_count: int,
        /,
        *,
        residual_tolerance: float = 1.0e-8,
    ):
        method_ = str(method).strip().lower()
        sector_ = str(sector).strip().lower()
        roots = int(root_count)
        tolerance = float(residual_tolerance)
        if method_ not in ("adc(2)", "adc(2)-x", "adc(3)", "eom-ccsd", "casci", "casscf"):
            raise ValueError("Unknown correlated excited-state method.")
        if (
            sector_ not in ("ee", "ip", "ea", "sf")
            or roots <= 0
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Correlated manifold sector, roots, or tolerance is invalid."
            )
        self.method = method_
        self.sector = sector_
        self.root_count = roots
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "correlated-manifold-plan",
                "method": method_,
                "sector": sector_,
                "root_count": roots,
                "residual_tolerance": tolerance,
            }
        )


class AbstractCorrelatedManifoldProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(self, plan: CorrelatedManifoldPlan, /) -> ElectronicManifoldResult:
        raise NotImplementedError


CorrelatedManifoldEvaluator = Callable[[CorrelatedManifoldPlan], ElectronicManifoldResult]


class CallableCorrelatedManifoldProvider(AbstractCorrelatedManifoldProvider):
    evaluator: CorrelatedManifoldEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: CorrelatedManifoldEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(self, plan: CorrelatedManifoldPlan, /) -> ElectronicManifoldResult:
        if not isinstance(plan, CorrelatedManifoldPlan):
            raise TypeError("plan must be CorrelatedManifoldPlan.")
        result = self.evaluator(plan)
        if not isinstance(result, ElectronicManifoldResult):
            raise TypeError(
                "Correlated manifold provider returned the wrong result type."
            )
        if (
            result.provider_id != self.provider_id
            or result.request_id != plan.plan_id
            or result.method != plan.method
            or result.spin_sector != plan.sector
            or result.excitation_energies.shape != (plan.root_count,)
        ):
            raise ValueError(
                "Correlated manifold result does not match provider, plan, method, sector, or root count."
            )
        return result


def casci_manifold(
    plan: CorrelatedManifoldPlan,
    result: CASCIResult,
    transition_dipoles: ArrayLike,
    energy_unit: UnitDefinition,
    transition_dipole_unit: UnitDefinition,
    /,
) -> ElectronicManifoldResult:
    if plan.method not in ("casci", "casscf") or plan.sector != "ee":
        raise ValueError("CASCI manifold adapter requires an EE CAS plan.")
    if not isinstance(result, CASCIResult) or not bool(result.successful):
        raise ValueError("CASCI manifold adapter requires a successful CAS result.")
    if result.energies.size < plan.root_count + 1:
        raise ValueError(
            "CAS result does not contain ground plus requested excited roots."
        )
    excitation = result.energies[1 : plan.root_count + 1] - result.energies[0]
    absolute = result.energies[1 : plan.root_count + 1]
    coefficients = result.coefficients[:, 1 : plan.root_count + 1]
    representation = CIStateRepresentation(coefficients, result.determinant_bits)
    dipoles = jnp.asarray(transition_dipoles, dtype=excitation.dtype)
    if dipoles.shape != (plan.root_count, 3):
        raise ValueError(
            "CAS transition dipoles must align with requested excited roots."
        )
    atomic_dipole = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
    excitation_atomic = excitation * float(conversion_factor(energy_unit, HARTREE))
    dipoles_atomic = dipoles * float(
        conversion_factor(transition_dipole_unit, atomic_dipole)
    )
    oscillator = (
        (2.0 / 3.0) * excitation_atomic * jnp.sum(jnp.abs(dipoles_atomic) ** 2, axis=1)
    )
    residuals = result.hamiltonian_residuals[1 : plan.root_count + 1]
    successful = jnp.all(residuals <= plan.residual_tolerance) & jnp.all(excitation > 0.0)
    return ElectronicManifoldResult(
        excitation,
        absolute,
        representation,
        dipoles,
        oscillator,
        residuals,
        successful,
        plan.method,
        plan.sector,
        tuple((index,) for index in range(plan.root_count)),
        energy_unit,
        transition_dipole_unit,
        provider_id="native-casci",
        request_id=plan.plan_id,
        state_space_id=plan.plan_id,
    )


def biorthogonal_manifold(
    plan: CorrelatedManifoldPlan,
    ground_state_energy: ArrayLike,
    excitation_energies: ArrayLike,
    right_amplitudes: ArrayLike,
    left_amplitudes: ArrayLike,
    transition_dipoles: ArrayLike,
    residuals: ArrayLike,
    energy_unit: UnitDefinition,
    transition_dipole_unit: UnitDefinition,
    /,
) -> ElectronicManifoldResult:
    excitation = jnp.asarray(excitation_energies)
    if excitation.shape != (plan.root_count,):
        raise ValueError("Biorthogonal excitation energies do not match requested roots.")
    representation = BiorthogonalStateRepresentation(
        right_amplitudes, left_amplitudes, plan.method
    )
    dipoles = jnp.asarray(transition_dipoles, dtype=excitation.dtype)
    residual = jnp.asarray(residuals, dtype=excitation.real.dtype)
    atomic_dipole = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
    excitation_atomic = excitation * float(conversion_factor(energy_unit, HARTREE))
    dipoles_atomic = dipoles * float(
        conversion_factor(transition_dipole_unit, atomic_dipole)
    )
    oscillator = (
        (2.0 / 3.0) * excitation_atomic * jnp.sum(jnp.abs(dipoles_atomic) ** 2, axis=1)
    )
    successful = (
        residual.shape == excitation.shape
        and dipoles.shape == (plan.root_count, 3)
        and bool(jnp.all(residual <= plan.residual_tolerance))
        and bool(representation.biorthogonality_residual <= plan.residual_tolerance)
    )
    return ElectronicManifoldResult(
        excitation,
        jnp.asarray(ground_state_energy, dtype=excitation.dtype) + excitation,
        representation,
        dipoles,
        oscillator,
        residual,
        successful,
        plan.method,
        plan.sector,
        tuple((index,) for index in range(plan.root_count)),
        energy_unit,
        transition_dipole_unit,
        provider_id="native-correlated",
        request_id=plan.plan_id,
        state_space_id=plan.plan_id,
    )


__all__ = [
    "AbstractCorrelatedManifoldProvider",
    "CallableCorrelatedManifoldProvider",
    "CorrelatedManifoldPlan",
    "biorthogonal_manifold",
    "casci_manifold",
]

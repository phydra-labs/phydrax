#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded candidate-only contracts for lattice-material frontier workflows."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)
from ...units import UnitDefinition


def _identifier(value: str, name: str) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return result


class LatticeFrontierCapacity(StrictModule, NonTrainableState):
    """Conservative code admission policy, never a release envelope."""

    maximum_atoms: int = eqx.field(static=True)
    maximum_qpoints: int = eqx.field(static=True)
    maximum_branches: int = eqx.field(static=True)
    maximum_routes: int = eqx.field(static=True)
    maximum_channels: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_atoms: int,
        maximum_qpoints: int,
        maximum_branches: int,
        maximum_routes: int,
        maximum_channels: int,
        maximum_bytes: int,
    ):
        values = tuple(
            (
                maximum_atoms,
                maximum_qpoints,
                maximum_branches,
                maximum_routes,
                maximum_channels,
                maximum_bytes,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError(
                "Frontier capacities must be positive conservative code limits."
            )
        (
            self.maximum_atoms,
            self.maximum_qpoints,
            self.maximum_branches,
            self.maximum_routes,
            self.maximum_channels,
            self.maximum_bytes,
        ) = values
        self.capacity_id = canonical_fingerprint(
            {"kind": "lattice-frontier-capacity", "values": list(values)}
        )

    def admit(
        self,
        *,
        atoms: int,
        qpoints: int,
        branches: int,
        routes: int,
        channels: int,
        bytes_required: int,
    ) -> None:
        requested = (
            int(atoms),
            int(qpoints),
            int(branches),
            int(routes),
            int(channels),
            int(bytes_required),
        )
        limits = (
            self.maximum_atoms,
            self.maximum_qpoints,
            self.maximum_branches,
            self.maximum_routes,
            self.maximum_channels,
            self.maximum_bytes,
        )
        if any(value < 0 for value in requested) or any(
            value > limit for value, limit in zip(requested, limits, strict=True)
        ):
            raise ValueError(
                "Candidate lattice workflow exceeds its conservative code capacity."
            )


class LatticeFrontierContract(StrictModule, NonTrainableState):
    capability: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    required_evidence: tuple[str, ...] = eqx.field(static=True)
    excluded_claims: tuple[str, ...] = eqx.field(static=True)
    capacity: LatticeFrontierCapacity
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        capability: str,
        method: str,
        approximation: str,
        required_evidence: tuple[str, ...],
        excluded_claims: tuple[str, ...],
        capacity: LatticeFrontierCapacity,
        /,
    ):
        if not isinstance(capacity, LatticeFrontierCapacity):
            raise TypeError("capacity must be LatticeFrontierCapacity.")
        self.capability = _identifier(capability, "capability")
        self.method = _identifier(method, "method")
        self.approximation = _identifier(approximation, "approximation")
        self.required_evidence = tuple(
            _identifier(value, "required evidence") for value in required_evidence
        )
        self.excluded_claims = tuple(
            _identifier(value, "excluded claim") for value in excluded_claims
        )
        if (
            not self.required_evidence
            or not self.excluded_claims
            or len(set(self.required_evidence)) != len(self.required_evidence)
        ):
            raise ValueError(
                "Frontier contract requires unique evidence and explicit nonclaims."
            )
        self.capacity = capacity
        self.contract_id = canonical_fingerprint(
            {
                "kind": "lattice-frontier-contract",
                "capability": self.capability,
                "method": self.method,
                "approximation": self.approximation,
                "required_evidence": list(self.required_evidence),
                "excluded_claims": list(self.excluded_claims),
            }
        )

    def support_tuple(
        self, coordinates: Mapping[str, str | int | bool], /
    ) -> SupportTuple:
        values = dict(coordinates)
        values.update(
            method=self.method,
            approximation=self.approximation,
            contract_id=self.contract_id,
        )
        return SupportTuple(self.capability, values)


class ElectronPhononCoupling(StrictModule, NonTrainableState):
    """Provider-normalized g[m,n,nu](k,q), not an electronic scattering solver."""

    vertices: Array
    energy_unit: UnitDefinition
    electron_model_id: str = eqx.field(static=True)
    phonon_plan_id: str = eqx.field(static=True)
    electron_mesh_id: str = eqx.field(static=True)
    phonon_mesh_id: str = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)
    provider_artifact_id: str = eqx.field(static=True)
    hermitian_reverse_residual: Array
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        energy_unit: UnitDefinition,
        /,
        *,
        electron_model_id: str,
        phonon_plan_id: str,
        electron_mesh_id: str,
        phonon_mesh_id: str,
        gauge_id: str,
        provider_artifact_id: str,
        hermitian_reverse_residual: float,
        capacity: LatticeFrontierCapacity,
    ):
        value = np.asarray(vertices)
        if value.ndim != 5 or np.any(~np.isfinite(value)):
            raise ValueError(
                "Electron-phonon vertices must be finite (k,q,m,n,mode) values."
            )
        capacity.admit(
            atoms=1,
            qpoints=value.shape[1],
            branches=value.shape[-1],
            routes=1,
            channels=value.size,
            bytes_required=value.nbytes,
        )
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        residual = float(hermitian_reverse_residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("Hermitian reverse residual must be finite and nonnegative.")
        self.vertices = jnp.asarray(value)
        self.energy_unit = energy_unit
        self.electron_model_id = _identifier(electron_model_id, "electron_model_id")
        self.phonon_plan_id = _identifier(phonon_plan_id, "phonon_plan_id")
        self.electron_mesh_id = _identifier(electron_mesh_id, "electron_mesh_id")
        self.phonon_mesh_id = _identifier(phonon_mesh_id, "phonon_mesh_id")
        self.gauge_id = _identifier(gauge_id, "gauge_id")
        self.provider_artifact_id = _identifier(
            provider_artifact_id, "provider_artifact_id"
        )
        self.hermitian_reverse_residual = jnp.asarray(residual)
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "electron-phonon-coupling",
                "electron": self.electron_model_id,
                "phonon": self.phonon_plan_id,
                "electron_mesh": self.electron_mesh_id,
                "phonon_mesh": self.phonon_mesh_id,
                "gauge": self.gauge_id,
                "provider": self.provider_artifact_id,
                "unit": energy_unit.unit_id,
                "vertices": array_tree_fingerprint(value),
            }
        )


class SpinPhononCoupling(StrictModule, NonTrainableState):
    """Provider-normalized spin-term derivatives in one declared normal coordinate."""

    derivatives: Array
    derivative_unit: UnitDefinition
    spin_model_id: str = eqx.field(static=True)
    spin_term_ids: tuple[str, ...] = eqx.field(static=True)
    phonon_plan_id: str = eqx.field(static=True)
    coordinate_convention_id: str = eqx.field(static=True)
    provider_artifact_id: str = eqx.field(static=True)
    reality_residual: Array
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        derivatives: ArrayLike,
        derivative_unit: UnitDefinition,
        /,
        *,
        spin_model_id: str,
        spin_term_ids: tuple[str, ...],
        phonon_plan_id: str,
        coordinate_convention_id: str,
        provider_artifact_id: str,
        reality_residual: float,
        capacity: LatticeFrontierCapacity,
    ):
        value = np.asarray(derivatives)
        if value.ndim < 2 or np.any(~np.isfinite(value)):
            raise ValueError(
                "Spin-phonon derivatives must be finite with term and mode axes."
            )
        capacity.admit(
            atoms=1,
            qpoints=1,
            branches=value.shape[-1],
            routes=value.shape[0],
            channels=value.size,
            bytes_required=value.nbytes,
        )
        if not isinstance(derivative_unit, UnitDefinition):
            raise TypeError("derivative_unit must be UnitDefinition.")
        terms = tuple(_identifier(item, "spin term ID") for item in spin_term_ids)
        if len(terms) != value.shape[0] or len(set(terms)) != len(terms):
            raise ValueError(
                "spin_term_ids must uniquely align with the derivative term axis."
            )
        residual = float(reality_residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("reality_residual must be finite and nonnegative.")
        self.derivatives = jnp.asarray(value)
        self.derivative_unit = derivative_unit
        self.spin_model_id = _identifier(spin_model_id, "spin_model_id")
        self.spin_term_ids = terms
        self.phonon_plan_id = _identifier(phonon_plan_id, "phonon_plan_id")
        self.coordinate_convention_id = _identifier(
            coordinate_convention_id, "coordinate_convention_id"
        )
        self.provider_artifact_id = _identifier(
            provider_artifact_id, "provider_artifact_id"
        )
        self.reality_residual = jnp.asarray(residual)
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "spin-phonon-coupling",
                "spin": self.spin_model_id,
                "terms": list(terms),
                "phonon": self.phonon_plan_id,
                "coordinate": self.coordinate_convention_id,
                "provider": self.provider_artifact_id,
                "unit": derivative_unit.unit_id,
                "derivatives": array_tree_fingerprint(value),
            }
        )


def lattice_frontier_contracts() -> tuple[LatticeFrontierContract, ...]:
    small = LatticeFrontierCapacity(
        maximum_atoms=64,
        maximum_qpoints=4096,
        maximum_branches=192,
        maximum_routes=2_000_000,
        maximum_channels=50_000_000,
        maximum_bytes=2_147_483_648,
    )
    specs = (
        (
            "chemistry.lattice.ifc3.native",
            "mixed-central-finite-displacement",
            "native-ifc3",
            (
                "sixfold-permutation",
                "translational-rotational-residual",
                "h-refinement",
                "force-noise",
            ),
            ("no-production-accuracy", "no-self-consistent-phonons"),
        ),
        (
            "chemistry.lattice.transport.iterative-bte",
            "projected-linearized-collision-solve",
            "iterative-bte",
            (
                "energy-nullspace",
                "normal-umklapp-decomposition",
                "linear-residual",
                "mesh-convergence",
            ),
            ("no-hydrodynamic-limit", "no-rta-equivalence"),
        ),
        (
            "atomistic.transport.green-kubo",
            "heat-current-autocorrelation",
            "green-kubo",
            (
                "energy-partition-gauge",
                "stationarity",
                "correlated-sample-ess",
                "window-tail-convergence",
            ),
            ("no-bte-equivalence", "no-experimental-conductivity"),
        ),
        (
            "chemistry.lattice.coupling.electron-phonon",
            "provider-normalized-mode-vertex",
            "electron-phonon",
            ("electron-gauge", "phonon-gauge", "momentum-map", "hermitian-reverse"),
            ("no-electronic-scattering-solver", "no-superconducting-prediction"),
        ),
        (
            "chemistry.lattice.coupling.spin-phonon",
            "provider-normalized-spin-derivative",
            "spin-phonon",
            (
                "spin-term-identity",
                "normal-coordinate-convention",
                "reality-symmetry",
                "provider-provenance",
            ),
            ("no-spin-dynamics", "no-magnetic-phase-prediction"),
        ),
        (
            "chemistry.lattice.defects",
            "primitive-supercell-projection",
            "defect-phonons",
            (
                "stable-image-map",
                "spectral-weight-sum-rule",
                "mass-ifc-perturbation",
                "localization-normalization",
            ),
            ("no-defect-scattering-production", "no-branch-identity"),
        ),
        (
            "chemistry.lattice.interfaces",
            "principal-layer-harmonic-transport",
            "harmonic-interface",
            (
                "lead-closure",
                "spectral-positivity",
                "energy-current-balance",
                "frequency-refinement",
            ),
            ("no-anharmonic-interface", "no-generic-negf"),
        ),
    )
    return tuple(LatticeFrontierContract(*spec, small) for spec in specs)


def candidate_lattice_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        contract.support_tuple(
            {"resource_accounting": "caller-measured-input-not-support-claim"}
        )
        for contract in lattice_frontier_contracts()
    )


def candidate_lattice_profiles() -> tuple[CapabilityProfile, ...]:
    """Return unreleased profiles over maturity-neutral frontier support."""
    contracts = lattice_frontier_contracts()
    supports = candidate_lattice_support_tuples()
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=(
                *contract.required_evidence,
                "resource-envelope",
                "lifecycle-restore",
                "documentation-nonclaims",
            ),
            released=False,
        )
        for contract, support in zip(contracts, supports, strict=True)
    )


def candidate_lattice_campaigns() -> tuple[ScientificCampaign, ...]:
    """Return one disjoint calibration/locked campaign per implemented frontier."""
    campaigns = []
    for contract in lattice_frontier_contracts():
        slug = contract.capability.replace(".", "-")
        calibration = ScientificCase(
            f"{slug}-calibration",
            f"{slug}-calibration-unit",
            contract.capability,
            "bounded-analytic-control",
            f"{slug}-calibration-preparation",
            f"{slug}-calibration-batch",
            (f"source:{slug}:analytic",),
        )
        locked = ScientificCase(
            f"{slug}-locked",
            f"{slug}-locked-unit",
            contract.capability,
            "independent-locked-control",
            f"{slug}-locked-preparation",
            f"{slug}-locked-batch",
            (f"source:{slug}:independent",),
        )
        campaigns.append(
            ScientificCampaign(
                (calibration, locked),
                (
                    CampaignRole("calibration", (calibration.case_id,)),
                    CampaignRole("locked_evaluation", (locked.case_id,)),
                ),
                criteria_ids=contract.required_evidence,
            )
        )
    return tuple(campaigns)


__all__ = [
    "candidate_lattice_campaigns",
    "candidate_lattice_profiles",
    "ElectronPhononCoupling",
    "LatticeFrontierCapacity",
    "LatticeFrontierContract",
    "SpinPhononCoupling",
    "candidate_lattice_support_tuples",
    "lattice_frontier_contracts",
]

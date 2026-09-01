#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

import numpy as np

from ..artifacts import ArtifactManifest
from ._core import _fingerprint


class ProfileName(StrEnum):
    """Named ROM capabilities; deliberately no umbrella ROM capability exists."""

    LINEAR_COERCIVE_RB = "linear-coercive-rb"
    LINEAR_POD = "linear-pod"
    PARAMETRIC_CERTIFIED = "parametric-certified"
    TRANSIENT_INDICATOR = "transient-indicator"
    NONLINEAR_INDICATOR = "nonlinear-indicator"
    DEIM_HYPERREDUCTION = "deim-hyperreduction"
    GNAT_HYPERREDUCTION = "gnat-hyperreduction"
    ECSW_HYPERREDUCTION = "ecsw-hyperreduction"
    LOCAL_REGISTERED_BASES = "local-registered-bases"
    OPERATOR_INFERENCE_OOD = "operator-inference-ood"
    NEURAL_EMPIRICAL_OOD = "neural-empirical-ood"
    MULTIFIDELITY_CONTROL_VARIATE = "multifidelity-control-variate"
    MULTIFIDELITY_MLMC = "multifidelity-mlmc"


class CertificateKind(StrEnum):
    NONE = "none"
    INDICATOR = "indicator"
    ERROR_BOUND = "error-bound"


@dataclass(frozen=True, slots=True)
class ErrorBoundContract:
    """Identity of externally justified constants that turn a residual into a bound."""

    residual_dual_norm_id: str
    error_norm_id: str
    lower_bound_source_id: str
    validity_id: str
    qoi_bound: bool = False
    contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        residual = self.residual_dual_norm_id.strip()
        error = self.error_norm_id.strip()
        source = self.lower_bound_source_id.strip()
        validity = self.validity_id.strip()
        if not residual or not error or not source or not validity:
            raise ValueError("Certified error-bound contract IDs must be non-empty.")
        object.__setattr__(self, "residual_dual_norm_id", residual)
        object.__setattr__(self, "error_norm_id", error)
        object.__setattr__(self, "lower_bound_source_id", source)
        object.__setattr__(self, "validity_id", validity)
        object.__setattr__(
            self,
            "contract_id",
            _fingerprint(
                {
                    "kind": "rom-error-bound-contract",
                    "residual_dual_norm_id": residual,
                    "error_norm_id": error,
                    "lower_bound_source_id": source,
                    "validity_id": validity,
                    "qoi_bound": bool(self.qoi_bound),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class LinearCoerciveRBProfile:
    maximum_basis_size: int
    greedy_tolerance: float
    bound_contract: ErrorBoundContract
    include_primal_dual_qoi: bool = True
    profile_name: ProfileName = field(default=ProfileName.LINEAR_COERCIVE_RB, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.maximum_basis_size)
        _nonnegative_finite(self.greedy_tolerance, "greedy_tolerance")
        if not isinstance(self.bound_contract, ErrorBoundContract):
            raise TypeError("Linear coercive RB requires an ErrorBoundContract.")
        if self.include_primal_dual_qoi and not self.bound_contract.qoi_bound:
            raise ValueError("Primal-dual QoI certification requires qoi_bound=True.")


@dataclass(frozen=True, slots=True)
class LinearPODProfile:
    basis_size: int
    retained_energy: float = 1.0
    profile_name: ProfileName = field(default=ProfileName.LINEAR_POD, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.basis_size)
        retained = float(self.retained_energy)
        if not np.isfinite(retained) or retained <= 0.0 or retained > 1.0:
            raise ValueError("retained_energy must be in (0, 1].")
        object.__setattr__(self, "retained_energy", retained)


@dataclass(frozen=True, slots=True)
class ParametricCertifiedProfile:
    maximum_basis_size: int
    greedy_tolerance: float
    bound_contract: ErrorBoundContract
    include_primal_dual_qoi: bool = True
    profile_name: ProfileName = field(
        default=ProfileName.PARAMETRIC_CERTIFIED, init=False
    )

    def __post_init__(self) -> None:
        _positive_rank(self.maximum_basis_size)
        _nonnegative_finite(self.greedy_tolerance, "greedy_tolerance")
        if not isinstance(self.bound_contract, ErrorBoundContract):
            raise TypeError("Parametric certification requires an ErrorBoundContract.")
        if self.include_primal_dual_qoi and not self.bound_contract.qoi_bound:
            raise ValueError("Primal-dual QoI certification requires qoi_bound=True.")


@dataclass(frozen=True, slots=True)
class TransientIndicatorProfile:
    basis_size: int
    indicator_tolerance: float
    bound_contract: ErrorBoundContract | None = None
    profile_name: ProfileName = field(default=ProfileName.TRANSIENT_INDICATOR, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.basis_size)
        _positive_finite(self.indicator_tolerance, "indicator_tolerance")
        _optional_bound(self.bound_contract)


@dataclass(frozen=True, slots=True)
class NonlinearIndicatorProfile:
    basis_size: int
    indicator_tolerance: float
    bound_contract: ErrorBoundContract | None = None
    profile_name: ProfileName = field(default=ProfileName.NONLINEAR_INDICATOR, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.basis_size)
        _positive_finite(self.indicator_tolerance, "indicator_tolerance")
        _optional_bound(self.bound_contract)


@dataclass(frozen=True, slots=True)
class DEIMHyperreductionProfile:
    collateral_basis_size: int
    defect_tolerance: float
    profile_name: ProfileName = field(default=ProfileName.DEIM_HYPERREDUCTION, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.collateral_basis_size)
        _nonnegative_finite(self.defect_tolerance, "defect_tolerance")


@dataclass(frozen=True, slots=True)
class GNATHyperreductionProfile:
    residual_basis_size: int
    sample_size: int
    defect_tolerance: float
    profile_name: ProfileName = field(default=ProfileName.GNAT_HYPERREDUCTION, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.residual_basis_size)
        _positive_rank(self.sample_size)
        if self.sample_size < self.residual_basis_size:
            raise ValueError("GNAT sample_size must cover the residual basis.")
        _nonnegative_finite(self.defect_tolerance, "defect_tolerance")


@dataclass(frozen=True, slots=True)
class ECSWHyperreductionProfile:
    maximum_elements: int
    defect_tolerance: float
    nnls_tolerance: float = 1e-12
    profile_name: ProfileName = field(default=ProfileName.ECSW_HYPERREDUCTION, init=False)

    def __post_init__(self) -> None:
        _positive_rank(self.maximum_elements)
        _nonnegative_finite(self.defect_tolerance, "defect_tolerance")
        _positive_finite(self.nnls_tolerance, "nnls_tolerance")


@dataclass(frozen=True, slots=True)
class LocalRegisteredBasesProfile:
    basis_size: int
    number_of_bases: int
    profile_name: ProfileName = field(
        default=ProfileName.LOCAL_REGISTERED_BASES, init=False
    )

    def __post_init__(self) -> None:
        _positive_rank(self.basis_size)
        _positive_rank(self.number_of_bases)


@dataclass(frozen=True, slots=True)
class OperatorInferenceOODProfile:
    basis_size: int
    ood_threshold: float
    ridge: float = 0.0
    continuous_time: bool = True
    profile_name: ProfileName = field(
        default=ProfileName.OPERATOR_INFERENCE_OOD, init=False
    )

    def __post_init__(self) -> None:
        _positive_rank(self.basis_size)
        _positive_finite(self.ood_threshold, "ood_threshold")
        _nonnegative_finite(self.ridge, "ridge")


@dataclass(frozen=True, slots=True)
class NeuralEmpiricalOODProfile:
    model_manifest: ArtifactManifest
    ood_threshold: float
    state_size: int
    profile_name: ProfileName = field(
        default=ProfileName.NEURAL_EMPIRICAL_OOD, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_manifest, ArtifactManifest):
            raise TypeError("Neural empirical ROMs require an ArtifactManifest.")
        _positive_finite(self.ood_threshold, "ood_threshold")
        _positive_rank(self.state_size)


@dataclass(frozen=True, slots=True)
class MultifidelityControlVariateProfile:
    minimum_paired_cases: int = 3
    profile_name: ProfileName = field(
        default=ProfileName.MULTIFIDELITY_CONTROL_VARIATE, init=False
    )

    def __post_init__(self) -> None:
        if int(self.minimum_paired_cases) < 2:
            raise ValueError("Control-variate training requires at least two pairs.")


@dataclass(frozen=True, slots=True)
class MultifidelityMLMCProfile:
    minimum_levels: int = 2
    profile_name: ProfileName = field(default=ProfileName.MULTIFIDELITY_MLMC, init=False)

    def __post_init__(self) -> None:
        if int(self.minimum_levels) < 2:
            raise ValueError("MLMC training requires at least two fidelity levels.")


ROMProfile: TypeAlias = (
    LinearCoerciveRBProfile
    | LinearPODProfile
    | ParametricCertifiedProfile
    | TransientIndicatorProfile
    | NonlinearIndicatorProfile
    | DEIMHyperreductionProfile
    | GNATHyperreductionProfile
    | ECSWHyperreductionProfile
    | LocalRegisteredBasesProfile
    | OperatorInferenceOODProfile
    | NeuralEmpiricalOODProfile
    | MultifidelityControlVariateProfile
    | MultifidelityMLMCProfile
)


def profile_descriptor(profile: ROMProfile, /) -> dict[str, object]:
    """Return a portable, canonical description of one named profile request."""
    common: dict[str, object] = {"profile_name": profile.profile_name.value}
    if isinstance(profile, (LinearCoerciveRBProfile, ParametricCertifiedProfile)):
        return {
            **common,
            "maximum_basis_size": profile.maximum_basis_size,
            "greedy_tolerance": profile.greedy_tolerance,
            "include_primal_dual_qoi": profile.include_primal_dual_qoi,
            "bound_contract": _bound_descriptor(profile.bound_contract),
        }
    if isinstance(profile, LinearPODProfile):
        return {
            **common,
            "basis_size": profile.basis_size,
            "retained_energy": profile.retained_energy,
        }
    if isinstance(profile, (TransientIndicatorProfile, NonlinearIndicatorProfile)):
        return {
            **common,
            "basis_size": profile.basis_size,
            "indicator_tolerance": profile.indicator_tolerance,
            "bound_contract": (
                None
                if profile.bound_contract is None
                else _bound_descriptor(profile.bound_contract)
            ),
        }
    if isinstance(profile, DEIMHyperreductionProfile):
        return {
            **common,
            "collateral_basis_size": profile.collateral_basis_size,
            "defect_tolerance": profile.defect_tolerance,
        }
    if isinstance(profile, GNATHyperreductionProfile):
        return {
            **common,
            "residual_basis_size": profile.residual_basis_size,
            "sample_size": profile.sample_size,
            "defect_tolerance": profile.defect_tolerance,
        }
    if isinstance(profile, ECSWHyperreductionProfile):
        return {
            **common,
            "maximum_elements": profile.maximum_elements,
            "defect_tolerance": profile.defect_tolerance,
            "nnls_tolerance": profile.nnls_tolerance,
        }
    if isinstance(profile, LocalRegisteredBasesProfile):
        return {
            **common,
            "basis_size": profile.basis_size,
            "number_of_bases": profile.number_of_bases,
        }
    if isinstance(profile, OperatorInferenceOODProfile):
        return {
            **common,
            "basis_size": profile.basis_size,
            "ood_threshold": profile.ood_threshold,
            "ridge": profile.ridge,
            "continuous_time": profile.continuous_time,
        }
    if isinstance(profile, NeuralEmpiricalOODProfile):
        return {
            **common,
            "model_artifact_id": profile.model_manifest.artifact_id,
            "model_manifest_id": profile.model_manifest.manifest_id,
            "model_sha256": profile.model_manifest.sha256,
            "ood_threshold": profile.ood_threshold,
            "state_size": profile.state_size,
        }
    if isinstance(profile, MultifidelityControlVariateProfile):
        return {**common, "minimum_paired_cases": profile.minimum_paired_cases}
    if isinstance(profile, MultifidelityMLMCProfile):
        return {**common, "minimum_levels": profile.minimum_levels}
    raise TypeError("Unknown ROM profile request.")


def _bound_descriptor(contract: ErrorBoundContract) -> dict[str, object]:
    return {
        "residual_dual_norm_id": contract.residual_dual_norm_id,
        "error_norm_id": contract.error_norm_id,
        "lower_bound_source_id": contract.lower_bound_source_id,
        "validity_id": contract.validity_id,
        "qoi_bound": contract.qoi_bound,
        "contract_id": contract.contract_id,
    }


def _positive_rank(value: int) -> None:
    if int(value) <= 0:
        raise ValueError("ROM basis and sample sizes must be positive.")


def _positive_finite(value: float, name: str) -> None:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def _nonnegative_finite(value: float, name: str) -> None:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _optional_bound(value: ErrorBoundContract | None) -> None:
    if value is not None and not isinstance(value, ErrorBoundContract):
        raise TypeError("bound_contract must be ErrorBoundContract or None.")


__all__ = [
    "CertificateKind",
    "DEIMHyperreductionProfile",
    "ECSWHyperreductionProfile",
    "ErrorBoundContract",
    "GNATHyperreductionProfile",
    "LinearCoerciveRBProfile",
    "LinearPODProfile",
    "LocalRegisteredBasesProfile",
    "MultifidelityControlVariateProfile",
    "MultifidelityMLMCProfile",
    "NeuralEmpiricalOODProfile",
    "NonlinearIndicatorProfile",
    "OperatorInferenceOODProfile",
    "ParametricCertifiedProfile",
    "ProfileName",
    "ROMProfile",
    "TransientIndicatorProfile",
    "profile_descriptor",
]

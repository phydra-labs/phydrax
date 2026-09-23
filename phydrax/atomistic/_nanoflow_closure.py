#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._admissibility import AdmissibilityHeader
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._nanoflow_observables import (
    DiffusionTensorFitResult,
    DrivenSlipFitResult,
    WallForceCorrelationResult,
)


class NanoflowClosureKind(str, Enum):
    DIFFUSION_TENSOR = "diffusion-tensor"
    SLIP_LENGTH = "slip-length"
    WALL_FRICTION = "wall-friction"


class NanoflowClosureSupport(StrictModule, NonTrainableState):
    temperature_interval: tuple[float, float] = eqx.field(static=True)
    confinement_interval: tuple[float, float] = eqx.field(static=True)
    maximum_driving_magnitude: float = eqx.field(static=True)
    composition_id: str = eqx.field(static=True)
    lower_wall_id: str = eqx.field(static=True)
    upper_wall_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        temperature_interval: tuple[float, float],
        confinement_interval: tuple[float, float],
        maximum_driving_magnitude: float,
        composition_id: str,
        lower_wall_id: str,
        upper_wall_id: str,
    ) -> None:
        temperature = tuple(float(value) for value in temperature_interval)
        confinement = tuple(float(value) for value in confinement_interval)
        driving = float(maximum_driving_magnitude)
        identities = tuple(
            str(value) for value in (composition_id, lower_wall_id, upper_wall_id)
        )
        if (
            len(temperature) != 2
            or len(confinement) != 2
            or any(
                not np.isfinite(value) for value in (*temperature, *confinement, driving)
            )
            or not 0.0 < temperature[0] <= temperature[1]
            or not 0.0 < confinement[0] <= confinement[1]
            or driving < 0.0
            or any(not value for value in identities)
            or identities[1] == identities[2]
        ):
            raise ValueError("Nanoflow closure support is invalid.")
        self.temperature_interval = temperature
        self.confinement_interval = confinement
        self.maximum_driving_magnitude = driving
        self.composition_id, self.lower_wall_id, self.upper_wall_id = identities
        self.support_id = canonical_fingerprint(
            {
                "kind": "nanoflow-closure-support",
                "temperature": temperature,
                "confinement": confinement,
                "maximum_driving_magnitude": driving,
                "composition": identities[0],
                "lower_wall": identities[1],
                "upper_wall": identities[2],
            }
        )


class AtomisticNanoflowClosureArtifact(StrictModule, NonTrainableState):
    value: Array
    covariance: Array
    evidence: AdmissibilityHeader
    support: NanoflowClosureSupport
    kind: NanoflowClosureKind = eqx.field(static=True)
    value_unit_id: str = eqx.field(static=True)
    covariance_unit_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    force_field_id: str = eqx.field(static=True)
    rollout_id: str = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: NanoflowClosureKind,
        value: ArrayLike,
        covariance: ArrayLike,
        evidence: AdmissibilityHeader,
        support: NanoflowClosureSupport,
        /,
        *,
        value_unit_id: str,
        covariance_unit_id: str,
        system_id: str,
        force_field_id: str,
        rollout_id: str,
        observer_id: str,
    ) -> None:
        value_ = np.asarray(value)
        covariance_ = np.asarray(covariance)
        identities = tuple(
            str(item)
            for item in (
                value_unit_id,
                covariance_unit_id,
                system_id,
                force_field_id,
                rollout_id,
                observer_id,
            )
        )
        if (
            not isinstance(kind, NanoflowClosureKind)
            or not isinstance(evidence, AdmissibilityHeader)
            or not isinstance(support, NanoflowClosureSupport)
            or value_.size == 0
            or np.any(~np.isfinite(value_))
            or np.any(~np.isfinite(covariance_))
            or covariance_.shape != (value_.size, value_.size)
            or any(not item for item in identities)
            or not bool(np.asarray(evidence.globally_eligible))
        ):
            raise ValueError(
                "Only admitted finite nanoflow closures can become artifacts."
            )
        symmetry_scale = max(1.0, float(np.max(np.abs(covariance_))))
        symmetry_tolerance = 128.0 * np.finfo(covariance_.dtype).eps * symmetry_scale
        if not np.allclose(covariance_, covariance_.T, rtol=0.0, atol=symmetry_tolerance):
            raise ValueError("Nanoflow closure covariance must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(covariance_)
        if float(np.min(eigenvalues)) < -symmetry_tolerance:
            raise ValueError("Nanoflow closure covariance must be positive semidefinite.")

        self.value = jnp.asarray(value_)
        self.covariance = jnp.asarray(covariance_)
        self.evidence = evidence
        self.support = support
        self.kind = kind
        (
            self.value_unit_id,
            self.covariance_unit_id,
            self.system_id,
            self.force_field_id,
            self.rollout_id,
            self.observer_id,
        ) = identities
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "atomistic-nanoflow-closure-artifact",
                "closure_kind": kind.value,
                "value": array_tree_fingerprint(value_),
                "covariance": array_tree_fingerprint(covariance_),
                "evidence": evidence.evidence_id,
                "support": support.support_id,
                "units": identities[:2],
                "system": identities[2],
                "force_field": identities[3],
                "rollout": identities[4],
                "observer": identities[5],
            }
        )


def diffusion_closure_artifact(
    fit: DiffusionTensorFitResult,
    support: NanoflowClosureSupport,
    /,
    *,
    value_unit_id: str,
    covariance_unit_id: str,
    system_id: str,
    force_field_id: str,
    rollout_id: str,
    observer_id: str,
) -> AtomisticNanoflowClosureArtifact:
    if not isinstance(fit, DiffusionTensorFitResult):
        raise TypeError("fit must be DiffusionTensorFitResult.")
    if (
        fit.support_id != support.support_id
        or fit.system_id != system_id
        or fit.force_field_id != force_field_id
        or fit.rollout_id != rollout_id
        or observer_id != fit.source_observer_id
    ):
        raise ValueError("Diffusion fit provenance does not match the closure support.")
    return AtomisticNanoflowClosureArtifact(
        NanoflowClosureKind.DIFFUSION_TENSOR,
        fit.diffusion_tensor,
        fit.covariance,
        fit.header,
        support,
        value_unit_id=value_unit_id,
        covariance_unit_id=covariance_unit_id,
        system_id=system_id,
        force_field_id=force_field_id,
        rollout_id=rollout_id,
        observer_id=observer_id,
    )


def slip_closure_artifact(
    fit: DrivenSlipFitResult,
    support: NanoflowClosureSupport,
    /,
    *,
    value_unit_id: str,
    covariance_unit_id: str,
    system_id: str,
    force_field_id: str,
    rollout_id: str,
    observer_id: str,
) -> AtomisticNanoflowClosureArtifact:
    if not isinstance(fit, DrivenSlipFitResult):
        raise TypeError("fit must be DrivenSlipFitResult.")
    if (
        fit.support_id != support.support_id
        or fit.system_id != system_id
        or fit.force_field_id != force_field_id
        or fit.rollout_id != rollout_id
        or observer_id != fit.source_observer_id
    ):
        raise ValueError("Slip fit provenance does not match the closure support.")
    return AtomisticNanoflowClosureArtifact(
        NanoflowClosureKind.SLIP_LENGTH,
        fit.slip_lengths,
        fit.covariance,
        fit.header,
        support,
        value_unit_id=value_unit_id,
        covariance_unit_id=covariance_unit_id,
        system_id=system_id,
        force_field_id=force_field_id,
        rollout_id=rollout_id,
        observer_id=observer_id,
    )


def wall_friction_closure_artifact(
    correlation: WallForceCorrelationResult,
    support: NanoflowClosureSupport,
    /,
    *,
    value_unit_id: str,
    covariance_unit_id: str,
    system_id: str,
    force_field_id: str,
    rollout_id: str,
    observer_id: str,
) -> AtomisticNanoflowClosureArtifact:
    if not isinstance(correlation, WallForceCorrelationResult):
        raise TypeError("correlation must be WallForceCorrelationResult.")
    if (
        correlation.support_id != support.support_id
        or correlation.system_id != system_id
        or correlation.force_field_id != force_field_id
        or correlation.rollout_id != rollout_id
        or observer_id != correlation.force_source_id
    ):
        raise ValueError(
            "Wall-force correlation provenance does not match the closure support."
        )
    return AtomisticNanoflowClosureArtifact(
        NanoflowClosureKind.WALL_FRICTION,
        jnp.asarray((correlation.friction_coefficient,)),
        correlation.covariance,
        correlation.header,
        support,
        value_unit_id=value_unit_id,
        covariance_unit_id=covariance_unit_id,
        system_id=system_id,
        force_field_id=force_field_id,
        rollout_id=rollout_id,
        observer_id=observer_id,
    )


__all__ = [
    "AtomisticNanoflowClosureArtifact",
    "NanoflowClosureKind",
    "NanoflowClosureSupport",
    "diffusion_closure_artifact",
    "slip_closure_artifact",
    "wall_friction_closure_artifact",
]

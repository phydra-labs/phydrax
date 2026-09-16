#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed approximation, observable, status, and provenance for particle spectra."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


class SpectrumStatus(IntEnum):
    SUCCESS = 0
    NUMERICAL_NONCONVERGENCE = 1
    NONFINITE_OUTPUT = 2
    NO_ROOT = 3
    NO_ELECTROWEAK_MINIMUM = 4
    RUNNING_TACHYON = 5
    POLE_TACHYON = 6
    NONPERTURBATIVE = 7
    INVALID_INPUT = 8
    PROVIDER_UNAVAILABLE = 9
    PROVIDER_FAILED = 10
    APPROXIMATION_WARNING = 11


class SpectrumApproximationProfile(StrictModule):
    """All perturbative, scheme, threshold, scale, and provider semantics."""

    model_id: str = eqx.field(static=True)
    renormalization_scheme: str = eqx.field(static=True)
    rge_loop_order: int = eqx.field(static=True)
    threshold_loop_order: int = eqx.field(static=True)
    pole_mass_loop_order: int = eqx.field(static=True)
    matching_scale_rule: str = eqx.field(static=True)
    electroweak_scale_rule: str = eqx.field(static=True)
    correction_ids: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        model_id: str,
        renormalization_scheme: str,
        rge_loop_order: int,
        threshold_loop_order: int,
        pole_mass_loop_order: int,
        matching_scale_rule: str,
        electroweak_scale_rule: str,
        correction_ids: Sequence[str],
        source_ids: Sequence[str],
    ):
        strings = tuple(
            str(value).strip()
            for value in (
                model_id,
                renormalization_scheme,
                matching_scale_rule,
                electroweak_scale_rule,
            )
        )
        orders = tuple(
            int(value)
            for value in (rge_loop_order, threshold_loop_order, pole_mass_loop_order)
        )
        corrections = tuple(sorted(str(value).strip() for value in correction_ids))
        sources = tuple(sorted(str(value).strip() for value in source_ids))
        if any(not value for value in strings) or any(value < 0 for value in orders):
            raise ValueError("Spectrum approximation strings/orders are invalid.")
        if any(not value for value in corrections + sources):
            raise ValueError("Spectrum correction/source identities must be non-empty.")
        if len(set(corrections)) != len(corrections) or len(set(sources)) != len(sources):
            raise ValueError("Spectrum correction/source identities must be unique.")
        self.model_id = strings[0]
        self.renormalization_scheme = strings[1]
        self.rge_loop_order = orders[0]
        self.threshold_loop_order = orders[1]
        self.pole_mass_loop_order = orders[2]
        self.matching_scale_rule = strings[2]
        self.electroweak_scale_rule = strings[3]
        self.correction_ids = corrections
        self.source_ids = sources
        self.profile_id = canonical_fingerprint(
            {
                "kind": "particle-spectrum-approximation-profile",
                "model_id": strings[0],
                "renormalization_scheme": strings[1],
                "orders": orders,
                "matching_scale_rule": strings[2],
                "electroweak_scale_rule": strings[3],
                "correction_ids": corrections,
                "source_ids": sources,
            }
        )


class SpectrumObservableTable(StrictModule):
    labels: tuple[str, ...] = eqx.field(static=True)
    kinds: tuple[str, ...] = eqx.field(static=True)
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    values: Array
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: Sequence[str],
        kinds: Sequence[str],
        unit_ids: Sequence[str],
        values: ArrayLike,
        /,
    ):
        label_values = tuple(str(value).strip() for value in labels)
        kind_values = tuple(str(value).strip() for value in kinds)
        units = tuple(str(value).strip() for value in unit_ids)
        array = jnp.asarray(values)
        if (
            not label_values
            or len(label_values) != len(kind_values)
            or len(label_values) != len(units)
            or array.shape != (len(label_values),)
        ):
            raise ValueError("Spectrum observable metadata and values must align.")
        if any(not value for value in label_values + kind_values + units):
            raise ValueError("Spectrum observable metadata must be non-empty.")
        if len(set(zip(label_values, kind_values, strict=True))) != len(label_values):
            raise ValueError("Spectrum observable label/kind pairs must be unique.")
        self.labels = label_values
        self.kinds = kind_values
        self.unit_ids = units
        self.values = array
        self.table_id = canonical_fingerprint(
            {
                "kind": "particle-spectrum-observable-table",
                "labels": label_values,
                "kinds": kind_values,
                "unit_ids": units,
                "values": array_tree_fingerprint(array),
            }
        )

    def value(self, label: str, /, *, kind: str) -> Array:
        key = (str(label), str(kind))
        pairs = tuple(zip(self.labels, self.kinds, strict=True))
        if key not in pairs:
            raise ValueError("Requested spectrum observable is absent.")
        return self.values[pairs.index(key)]


class SpectrumDiagnostics(StrictModule):
    numerical_status: Array
    provider_available: Array
    finite: Array
    root_found: Array
    electroweak_minimum: Array
    perturbative: Array
    running_tachyons: Array
    pole_tachyons: Array
    physical_admissible: Array
    approximation_warning: Array
    residual_norm: Array
    warning_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        numerical_status: int | SpectrumStatus,
        /,
        *,
        provider_available: ArrayLike,
        finite: ArrayLike,
        root_found: ArrayLike,
        electroweak_minimum: ArrayLike,
        perturbative: ArrayLike,
        running_tachyons: ArrayLike,
        pole_tachyons: ArrayLike,
        approximation_warning: ArrayLike,
        residual_norm: ArrayLike,
        warning_ids: Sequence[str] = (),
    ):
        status = SpectrumStatus(int(numerical_status))
        warnings = tuple(sorted(str(value).strip() for value in warning_ids))
        if any(not value for value in warnings) or len(set(warnings)) != len(warnings):
            raise ValueError("Spectrum warning IDs must be unique and non-empty.")
        provider = jnp.asarray(provider_available, dtype=bool).reshape(())
        finite_value = jnp.asarray(finite, dtype=bool).reshape(())
        root = jnp.asarray(root_found, dtype=bool).reshape(())
        minimum = jnp.asarray(electroweak_minimum, dtype=bool).reshape(())
        perturbative_value = jnp.asarray(perturbative, dtype=bool).reshape(())
        running = jnp.asarray(running_tachyons, dtype=bool).reshape(())
        pole = jnp.asarray(pole_tachyons, dtype=bool).reshape(())
        warning = jnp.asarray(approximation_warning, dtype=bool).reshape(())
        residual = jnp.asarray(residual_norm).reshape(())
        physical = (
            provider
            & finite_value
            & root
            & minimum
            & perturbative_value
            & ~running
            & ~pole
        )
        self.numerical_status = jnp.asarray(int(status), dtype=jnp.int32)
        self.provider_available = provider
        self.finite = finite_value
        self.root_found = root
        self.electroweak_minimum = minimum
        self.perturbative = perturbative_value
        self.running_tachyons = running
        self.pole_tachyons = pole
        self.physical_admissible = physical
        self.approximation_warning = warning
        self.residual_norm = residual
        self.warning_ids = warnings

    @property
    def successful(self) -> Array:
        return (
            (self.numerical_status == int(SpectrumStatus.SUCCESS))
            & self.physical_admissible
            & ~self.approximation_warning
        )


class SpectrumCalculationResult(StrictModule):
    observables: SpectrumObservableTable
    running_scales: Array
    running_parameters: Array
    diagnostics: SpectrumDiagnostics
    approximation: SpectrumApproximationProfile = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    input_artifact_id: str = eqx.field(static=True)
    output_artifact_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        observables: SpectrumObservableTable,
        running_scales: ArrayLike,
        running_parameters: ArrayLike,
        diagnostics: SpectrumDiagnostics,
        approximation: SpectrumApproximationProfile,
        /,
        *,
        provider_id: str,
        input_artifact_id: str,
        output_artifact_id: str,
    ):
        if not isinstance(observables, SpectrumObservableTable):
            raise TypeError("observables must be SpectrumObservableTable.")
        if not isinstance(diagnostics, SpectrumDiagnostics):
            raise TypeError("diagnostics must be SpectrumDiagnostics.")
        if not isinstance(approximation, SpectrumApproximationProfile):
            raise TypeError("approximation must be SpectrumApproximationProfile.")
        scales = jnp.asarray(running_scales)
        parameters = jnp.asarray(running_parameters)
        if scales.ndim != 1 or scales.size < 1 or parameters.shape[:1] != scales.shape:
            raise ValueError("Running scales and parameter trajectory must align.")
        if parameters.ndim != 2:
            raise ValueError("running_parameters must have shape (scales, parameters).")
        identities = tuple(
            str(value).strip()
            for value in (provider_id, input_artifact_id, output_artifact_id)
        )
        if any(not value for value in identities):
            raise ValueError("Spectrum provider and artifact identities are required.")
        self.observables = observables
        self.running_scales = scales
        self.running_parameters = parameters
        self.diagnostics = diagnostics
        self.approximation = approximation
        self.provider_id, self.input_artifact_id, self.output_artifact_id = identities
        self.result_id = canonical_fingerprint(
            {
                "kind": "particle-spectrum-calculation-result",
                "observables": observables.table_id,
                "running": array_tree_fingerprint((scales, parameters)),
                "approximation": approximation.profile_id,
                "provider": identities[0],
                "input_artifact": identities[1],
                "output_artifact": identities[2],
                "status": int(diagnostics.numerical_status),
                "warnings": diagnostics.warning_ids,
            }
        )
        self.claim = "finite-spectrum-result-with-explicit-approximation-provider-and-physical-status"


__all__ = [
    "SpectrumApproximationProfile",
    "SpectrumCalculationResult",
    "SpectrumDiagnostics",
    "SpectrumObservableTable",
    "SpectrumStatus",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact provider boundaries for Raman optical activity and periodic spectra."""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


class RamanOpticalActivityResult(StrictModule, NonTrainableState):
    wavenumbers: Array
    electric_polarizability_derivatives: Array
    electric_magnetic_derivatives: Array
    electric_quadrupole_derivatives: Array
    right_circular_intensities: Array
    left_circular_intensities: Array
    circular_intensity_differences: Array
    dissymmetry_factors: Array
    successful: Array
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavenumbers: ArrayLike,
        electric_polarizability_derivatives: ArrayLike,
        electric_magnetic_derivatives: ArrayLike,
        electric_quadrupole_derivatives: ArrayLike,
        right_circular_intensities: ArrayLike,
        left_circular_intensities: ArrayLike,
        successful: ArrayLike,
        provider_id: str,
        /,
    ):
        waves = jnp.asarray(wavenumbers)
        electric = jnp.asarray(electric_polarizability_derivatives, dtype=waves.dtype)
        magnetic = jnp.asarray(electric_magnetic_derivatives, dtype=waves.dtype)
        quadrupole = jnp.asarray(electric_quadrupole_derivatives, dtype=waves.dtype)
        right = jnp.asarray(right_circular_intensities, dtype=waves.dtype)
        left = jnp.asarray(left_circular_intensities, dtype=waves.dtype)
        modes = int(waves.size)
        provider = str(provider_id).strip()
        if (
            electric.shape != (modes, 3, 3)
            or magnetic.shape != (modes, 3, 3)
            or quadrupole.shape != (modes, 3, 3, 3)
            or right.shape != (modes,)
            or left.shape != (modes,)
            or not provider
        ):
            raise ValueError(
                "ROA tensors, intensities, or provider identity do not align."
            )
        difference = right - left
        total = right + left
        dissymmetry = jnp.where(total > 0.0, 2.0 * difference / total, 0.0)
        valid = (
            jnp.asarray(successful, dtype=bool)
            & jnp.all(right >= 0.0)
            & jnp.all(left >= 0.0)
            & jnp.all(jnp.isfinite(dissymmetry))
        )
        self.wavenumbers = waves
        self.electric_polarizability_derivatives = electric
        self.electric_magnetic_derivatives = magnetic
        self.electric_quadrupole_derivatives = quadrupole
        self.right_circular_intensities = right
        self.left_circular_intensities = left
        self.circular_intensity_differences = difference
        self.dissymmetry_factors = dissymmetry
        self.successful = valid
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "raman-optical-activity-result",
                "provider": provider,
                "successful": bool(valid),
                "arrays": array_tree_fingerprint(
                    {
                        "wavenumbers": np.asarray(waves),
                        "electric": np.asarray(electric),
                        "magnetic": np.asarray(magnetic),
                        "quadrupole": np.asarray(quadrupole),
                        "right": np.asarray(right),
                        "left": np.asarray(left),
                    }
                ),
            }
        )


class AbstractRamanOpticalActivityProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(self, positions: ArrayLike, /) -> RamanOpticalActivityResult:
        raise NotImplementedError


ROAEvaluator = Callable[[ArrayLike], RamanOpticalActivityResult]


class CallableRamanOpticalActivityProvider(AbstractRamanOpticalActivityProvider):
    evaluator: ROAEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: ROAEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(self, positions: ArrayLike, /) -> RamanOpticalActivityResult:
        result = self.evaluator(positions)
        if (
            not isinstance(result, RamanOpticalActivityResult)
            or result.provider_id != self.provider_id
        ):
            raise ValueError("ROA provider changed result type or identity.")
        return result


class PeriodicSpectroscopyResult(StrictModule, NonTrainableState):
    qpoints: Array
    frequencies: Array
    infrared_oscillator_strengths: Array
    raman_tensors: Array
    born_effective_charges: Array
    dielectric_tensor: Array
    acoustic_sum_rule_residual: Array
    successful: Array
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        qpoints: ArrayLike,
        frequencies: ArrayLike,
        infrared_oscillator_strengths: ArrayLike,
        raman_tensors: ArrayLike,
        born_effective_charges: ArrayLike,
        dielectric_tensor: ArrayLike,
        acoustic_sum_rule_residual: ArrayLike,
        successful: ArrayLike,
        provider_id: str,
        /,
    ):
        qpoints_ = jnp.asarray(qpoints)
        frequencies_ = jnp.asarray(frequencies, dtype=qpoints_.dtype)
        infrared = jnp.asarray(infrared_oscillator_strengths, dtype=qpoints_.dtype)
        raman = jnp.asarray(raman_tensors, dtype=qpoints_.dtype)
        born = jnp.asarray(born_effective_charges, dtype=qpoints_.dtype)
        dielectric = jnp.asarray(dielectric_tensor, dtype=qpoints_.dtype)
        provider = str(provider_id).strip()
        qcount = (
            qpoints_.shape[0] if qpoints_.ndim == 2 and qpoints_.shape[1] == 3 else -1
        )
        branches = (
            frequencies_.shape[1]
            if frequencies_.ndim == 2 and frequencies_.shape[0] == qcount
            else -1
        )
        if (
            infrared.shape != (qcount, branches, 3)
            or raman.shape != (qcount, branches, 3, 3)
            or born.ndim != 3
            or born.shape[1:] != (3, 3)
            or dielectric.shape != (3, 3)
            or not provider
        ):
            raise ValueError(
                "Periodic spectral arrays or provider identity do not align."
            )
        residual = jnp.asarray(acoustic_sum_rule_residual, dtype=qpoints_.dtype).reshape(
            ()
        )
        valid = (
            jnp.asarray(successful, dtype=bool)
            & jnp.all(jnp.isfinite(frequencies_))
            & jnp.all(jnp.isfinite(infrared))
            & jnp.all(jnp.isfinite(raman))
            & jnp.all(jnp.isfinite(born))
            & jnp.all(jnp.isfinite(dielectric))
            & jnp.isfinite(residual)
        )
        self.qpoints = qpoints_
        self.frequencies = frequencies_
        self.infrared_oscillator_strengths = infrared
        self.raman_tensors = raman
        self.born_effective_charges = born
        self.dielectric_tensor = dielectric
        self.acoustic_sum_rule_residual = residual
        self.successful = valid
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-spectroscopy-result",
                "provider": provider,
                "successful": bool(valid),
                "arrays": array_tree_fingerprint(
                    {
                        "qpoints": np.asarray(qpoints_),
                        "frequencies": np.asarray(frequencies_),
                        "infrared": np.asarray(infrared),
                        "raman": np.asarray(raman),
                        "born": np.asarray(born),
                        "dielectric": np.asarray(dielectric),
                        "asr_residual": np.asarray(residual),
                    }
                ),
            }
        )


class AbstractPeriodicSpectroscopyProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        cell: ArrayLike,
        qpoints: ArrayLike,
        /,
    ) -> PeriodicSpectroscopyResult:
        raise NotImplementedError


PeriodicSpectroscopyEvaluator = Callable[
    [ArrayLike, ArrayLike, ArrayLike], PeriodicSpectroscopyResult
]


class CallablePeriodicSpectroscopyProvider(AbstractPeriodicSpectroscopyProvider):
    evaluator: PeriodicSpectroscopyEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: PeriodicSpectroscopyEvaluator,
        provider_id: str,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(
        self,
        positions: ArrayLike,
        cell: ArrayLike,
        qpoints: ArrayLike,
        /,
    ) -> PeriodicSpectroscopyResult:
        result = self.evaluator(positions, cell, qpoints)
        if (
            not isinstance(result, PeriodicSpectroscopyResult)
            or result.provider_id != self.provider_id
        ):
            raise ValueError(
                "Periodic spectroscopy provider changed result type or identity."
            )
        return result


__all__ = [
    "AbstractPeriodicSpectroscopyProvider",
    "AbstractRamanOpticalActivityProvider",
    "CallablePeriodicSpectroscopyProvider",
    "CallableRamanOpticalActivityProvider",
    "PeriodicSpectroscopyResult",
    "RamanOpticalActivityResult",
]

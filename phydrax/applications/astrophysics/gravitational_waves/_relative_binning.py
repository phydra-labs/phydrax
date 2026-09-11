#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._approximation import (
    LikelihoodApproximationPolicy,
    QualifiedGravitationalWaveLikelihood,
    qualify_likelihood,
)
from ._data import DetectorNetworkData
from ._detector import DetectorResponsePlan
from ._likelihood import (
    AbstractGravitationalWaveLikelihood,
    GravitationalWaveLikelihoodEvaluation,
    GravitationalWaveLikelihoodPlan,
)
from ._status import GravitationalWaveStatus
from ._waveform import AbstractFrequencyDomainWaveform


class _RelativeBinningLikelihood(AbstractGravitationalWaveLikelihood):
    base: GravitationalWaveLikelihoodPlan
    network: DetectorNetworkData
    response: DetectorResponsePlan
    waveform: AbstractFrequencyDomainWaveform
    edge_frequency: Array
    fiducial_edge_signal: Array
    a0: Array
    a1: Array
    b0: Array
    b1: Array
    b2: Array
    bin_width: Array
    parameterization_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: GravitationalWaveLikelihoodPlan,
        fiducial_parameters: PyTree[Any],
        /,
        *,
        num_bins: int,
    ):
        if not isinstance(base, GravitationalWaveLikelihoodPlan):
            raise TypeError("base must be GravitationalWaveLikelihoodPlan.")
        if not base.waveform.capabilities.arbitrary_frequencies:
            raise ValueError(
                "Relative binning requires arbitrary-frequency waveform evaluation."
            )
        if base.calibration_fn is not None:
            raise ValueError("Relative binning does not support calibration callbacks.")
        bins = int(num_bins)
        active_union = np.any(np.asarray(base.network.active), axis=0)
        active_indices = np.flatnonzero(active_union)
        if bins < 2 or active_indices.size < bins + 1:
            raise ValueError("Relative bin count must fit the active frequency support.")
        edges = np.linspace(
            active_indices[0], active_indices[-1], bins + 1, dtype=np.int32
        )
        if np.unique(edges).size != edges.size or np.any(np.diff(edges) <= 0):
            raise ValueError("Relative-bin edges must be unique and increasing.")
        edge_frequency = base.network.frequency[jnp.asarray(edges)]
        fiducial_full, _, _ = base.detector_signal(fiducial_parameters)
        fiducial_edges, _, _ = base.detector_signal(
            fiducial_parameters, frequency=edge_frequency
        )
        if bool(jnp.any(~jnp.isfinite(fiducial_edges))) or bool(
            jnp.any(
                jnp.abs(fiducial_edges) <= jnp.finfo(jnp.real(fiducial_edges).dtype).tiny
            )
        ):
            raise ValueError("Relative-binning fiducial waveform vanishes at a bin edge.")
        inverse_variance = base.network.inverse_variance
        a0_rows = []
        a1_rows = []
        b0_rows = []
        b1_rows = []
        b2_rows = []
        widths = []
        for bin_index, (start, stop) in enumerate(itertools.pairwise(edges)):
            end = int(stop) + (1 if bin_index == bins - 1 else 0)
            indices = jnp.arange(int(start), end)
            frequency = base.network.frequency[indices]
            center = 0.5 * (
                base.network.frequency[int(start)] + base.network.frequency[int(stop)]
            )
            offset = frequency - center
            h0 = fiducial_full[:, indices]
            inverse = inverse_variance[:, indices]
            linear = 2.0 * jnp.conj(base.network.strain[:, indices]) * h0 * inverse
            quadratic = 2.0 * jnp.abs(h0) ** 2 * inverse
            a0_rows.append(jnp.sum(linear, axis=-1))
            a1_rows.append(jnp.sum(linear * offset[None, :], axis=-1))
            b0_rows.append(jnp.sum(quadratic, axis=-1))
            b1_rows.append(jnp.sum(quadratic * offset[None, :], axis=-1))
            b2_rows.append(jnp.sum(quadratic * offset[None, :] ** 2, axis=-1))
            widths.append(
                base.network.frequency[int(stop)] - base.network.frequency[int(start)]
            )
        self.base = base
        self.network = base.network
        self.response = base.response
        self.waveform = base.waveform
        self.parameterization_id = base.parameterization_id
        self.edge_frequency = edge_frequency
        self.fiducial_edge_signal = fiducial_edges
        self.a0 = jnp.stack(tuple(a0_rows), axis=-1)
        self.a1 = jnp.stack(tuple(a1_rows), axis=-1)
        self.b0 = jnp.stack(tuple(b0_rows), axis=-1)
        self.b1 = jnp.stack(tuple(b1_rows), axis=-1)
        self.b2 = jnp.stack(tuple(b2_rows), axis=-1)
        self.bin_width = jnp.asarray(widths)
        self.approximation_id = "relative-binning:" + canonical_fingerprint(
            {
                "base": base.likelihood_id,
                "bins": bins,
                "edges": array_tree_fingerprint(edge_frequency)["sha256"],
                "fiducial": array_tree_fingerprint(fiducial_parameters)["sha256"],
            }
        )
        self.likelihood_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-relative-binning",
                "base": base.likelihood_id,
                "bins": bins,
                "fiducial": array_tree_fingerprint(fiducial_parameters),
                "content": array_tree_fingerprint(
                    {
                        "edges": edge_frequency,
                        "fiducial": fiducial_edges,
                        "a0": self.a0,
                        "a1": self.a1,
                        "b0": self.b0,
                        "b1": self.b1,
                        "b2": self.b2,
                    }
                )["sha256"],
            }
        )

    def detector_signal(
        self, parameters: PyTree[Any], /, *, frequency: Array | None = None
    ):
        return self.base.detector_signal(parameters, frequency=frequency)

    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> GravitationalWaveLikelihoodEvaluation:
        edge_signal, response, polarizations = self.base.detector_signal(
            parameters, frequency=self.edge_frequency
        )
        ratio_edges = edge_signal / self.fiducial_edge_signal
        r0 = 0.5 * (ratio_edges[:, 1:] + ratio_edges[:, :-1])
        r1 = (ratio_edges[:, 1:] - ratio_edges[:, :-1]) / self.bin_width[None, :]
        complex_inner = jnp.sum(self.a0 * r0 + self.a1 * r1, axis=-1)
        signal_norm = jnp.sum(
            self.b0 * jnp.abs(r0) ** 2
            + 2.0 * self.b1 * jnp.real(jnp.conj(r0) * r1)
            + self.b2 * jnp.abs(r1) ** 2,
            axis=-1,
        )
        ratio = jnp.real(complex_inner) - 0.5 * signal_norm
        noise = self.network.noise_log_probability_by_detector
        log_probability = noise + ratio
        data_norm = self.network.data_norm_by_detector
        valid = (
            response.valid
            & polarizations.valid
            & jnp.all(jnp.isfinite(log_probability))
            & jnp.all(signal_norm >= 0.0)
        )
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            int(GravitationalWaveStatus.APPROXIMATION_FAILURE),
        ).astype(jnp.int32)
        matched_filter = complex_inner / jnp.sqrt(
            jnp.where(signal_norm > 0.0, signal_norm, 1.0)
        )
        return GravitationalWaveLikelihoodEvaluation(
            edge_signal,
            complex_inner,
            signal_norm,
            data_norm,
            log_probability,
            noise,
            ratio,
            matched_filter,
            signal_norm,
            valid,
            status,
            self.likelihood_id,
            self.approximation_id,
        )


def prepare_relative_binning_likelihood(
    exact: GravitationalWaveLikelihoodPlan,
    fiducial_parameters: PyTree[Any],
    validation_parameters: Sequence[PyTree[Any]],
    validation_ids: Sequence[str],
    /,
    *,
    num_bins: int,
    policy: LikelihoodApproximationPolicy | None = None,
) -> QualifiedGravitationalWaveLikelihood:
    candidate = _RelativeBinningLikelihood(
        exact,
        fiducial_parameters,
        num_bins=num_bins,
    )
    return qualify_likelihood(
        exact,
        candidate,
        validation_parameters,
        validation_ids,
        policy=policy,
    )


__all__ = ["prepare_relative_binning_likelihood"]

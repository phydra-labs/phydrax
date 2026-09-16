#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...atomistic import (
    DebyeScatteringResult,
    PartialStructureFactorResult,
)
from ...discretization.spectral import TensorSpectralDiscretization
from ...observation import CoordinateLayout, TheoryVector
from ..polymer_field_theory import SCFTEvaluation
from ._prism import PRISMResult


def _product_id(kind: str, payload: dict, /) -> str:
    return canonical_fingerprint({"kind": kind, **payload})


def debye_theory_vector(
    result: DebyeScatteringResult,
    /,
    *,
    wave_number_unit_id: str,
    intensity_normalization_id: str,
) -> TheoryVector:
    if not isinstance(result, DebyeScatteringResult):
        raise TypeError("result must be DebyeScatteringResult.")
    if not bool(np.asarray(result.successful)):
        raise ValueError("A failed Debye result cannot become an observation product.")
    labels = tuple(f"debye:q[{index}]" for index in range(result.values.size))
    product = _product_id(
        "polymer-debye-theory-vector",
        {
            "plan": result.plan_id,
            "wave_numbers": array_tree_fingerprint(np.asarray(result.wave_numbers)),
            "wave_number_unit": str(wave_number_unit_id),
            "normalization": str(intensity_normalization_id),
        },
    )
    return TheoryVector(result.values.reshape((-1,)), CoordinateLayout(labels), product)


def partial_structure_theory_vector(
    result: PartialStructureFactorResult,
    site_ids: tuple[str, ...],
    /,
    *,
    wave_vector_unit_id: str,
    normalization_id: str,
) -> TheoryVector:
    if not isinstance(result, PartialStructureFactorResult):
        raise TypeError("result must be PartialStructureFactorResult.")
    if not bool(np.asarray(result.successful)):
        raise ValueError(
            "A failed partial structure factor cannot become a theory vector."
        )
    count = result.values.shape[-1]
    if len(site_ids) != count or len(set(site_ids)) != count:
        raise ValueError("site_ids must identify every partial structure-factor channel.")
    labels = tuple(
        f"partial-s:k[{wave_index}]:{left}:{right}"
        for wave_index in range(result.values.shape[0])
        for left in site_ids
        for right in site_ids
    )
    product = _product_id(
        "polymer-partial-structure-theory-vector",
        {
            "plan": result.plan_id,
            "wave_vectors": array_tree_fingerprint(np.asarray(result.wave_vectors)),
            "site_ids": list(site_ids),
            "wave_vector_unit": str(wave_vector_unit_id),
            "normalization": str(normalization_id),
        },
    )
    return TheoryVector(result.values.reshape((-1,)), CoordinateLayout(labels), product)


def prism_structure_theory_vector(
    result: PRISMResult,
    site_ids: tuple[str, ...],
    wave_numbers: ArrayLike,
    /,
    *,
    wave_number_unit_id: str,
    normalization_id: str,
) -> TheoryVector:
    if not isinstance(result, PRISMResult):
        raise TypeError("result must be PRISMResult.")
    if not bool(np.asarray(result.successful)):
        raise ValueError("A failed PRISM result cannot become an observation product.")
    structure = result.evaluation.oz.structure_factor
    wave = jnp.asarray(wave_numbers)
    count = structure.shape[-1]
    if structure.shape[0] != wave.size or len(site_ids) != count:
        raise ValueError("PRISM structure factors, wave numbers, and site IDs differ.")
    labels = tuple(
        f"prism-s:k[{wave_index}]:{left}:{right}"
        for wave_index in range(wave.size)
        for left in site_ids
        for right in site_ids
    )
    product = _product_id(
        "prism-structure-theory-vector",
        {
            "prepared": result.prepared_id,
            "wave_numbers": array_tree_fingerprint(np.asarray(wave)),
            "site_ids": list(site_ids),
            "wave_number_unit": str(wave_number_unit_id),
            "normalization": str(normalization_id),
        },
    )
    return TheoryVector(structure.reshape((-1,)), CoordinateLayout(labels), product)


def scft_density_scattering_theory_vector(
    evaluation: SCFTEvaluation,
    spectral: TensorSpectralDiscretization,
    contrasts: ArrayLike,
    /,
    *,
    reciprocal_unit_id: str,
    normalization_id: str,
) -> TheoryVector:
    if not isinstance(evaluation, SCFTEvaluation):
        raise TypeError("evaluation must be SCFTEvaluation.")
    if not isinstance(spectral, TensorSpectralDiscretization):
        raise TypeError("spectral must be TensorSpectralDiscretization.")
    if not bool(np.asarray(evaluation.successful)):
        raise ValueError("A failed SCFT evaluation cannot become an observation product.")
    contrast = jnp.asarray(contrasts, dtype=evaluation.densities.dtype)
    if contrast.shape != (evaluation.densities.shape[-1],):
        raise ValueError("contrasts must provide one value per SCFT species.")
    contrast_density = jnp.sum(evaluation.densities * contrast, axis=-1)
    modal = spectral.project(contrast_density)
    intensity = jnp.real(modal * jnp.conj(modal)).reshape((-1,))
    labels = tuple(f"scft-density-mode[{index}]" for index in range(intensity.size))
    product = _product_id(
        "scft-density-scattering-theory-vector",
        {
            "prepared": evaluation.prepared_id,
            "spectral": spectral.prepared_id,
            "contrasts": array_tree_fingerprint(np.asarray(contrast)),
            "reciprocal_unit": str(reciprocal_unit_id),
            "normalization": str(normalization_id),
        },
    )
    return TheoryVector(intensity, CoordinateLayout(labels), product)


__all__ = [
    "debye_theory_vector",
    "partial_structure_theory_vector",
    "prism_structure_theory_vector",
    "scft_density_scattering_theory_vector",
]

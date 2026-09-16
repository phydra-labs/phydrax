#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


PlateauModulusConvention: TypeAlias = Literal["affine", "tube-four-fifths"]


class EntanglementEstimatorPlan(StrictModule, NonTrainableState):
    monomer_number_density: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    block_count: int = eqx.field(static=True)
    minimum_frames: int = eqx.field(static=True)
    maximum_relative_standard_error: float = eqx.field(static=True)
    plateau_convention: PlateauModulusConvention = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        monomer_number_density: float,
        temperature: float,
        /,
        *,
        boltzmann_constant: float = 1.0,
        block_count: int = 4,
        minimum_frames: int = 8,
        maximum_relative_standard_error: float = 0.25,
        plateau_convention: PlateauModulusConvention = "tube-four-fifths",
    ):
        density = float(monomer_number_density)
        thermal = float(temperature)
        boltzmann = float(boltzmann_constant)
        blocks = int(block_count)
        minimum = int(minimum_frames)
        relative = float(maximum_relative_standard_error)
        if (
            not math.isfinite(density)
            or density <= 0.0
            or not math.isfinite(thermal)
            or thermal <= 0.0
            or not math.isfinite(boltzmann)
            or boltzmann <= 0.0
            or blocks < 2
            or minimum < blocks
            or not math.isfinite(relative)
            or relative <= 0.0
            or plateau_convention not in ("affine", "tube-four-fifths")
        ):
            raise ValueError("Entanglement estimator controls are invalid.")
        self.monomer_number_density = density
        self.temperature = thermal
        self.boltzmann_constant = boltzmann
        self.block_count = blocks
        self.minimum_frames = minimum
        self.maximum_relative_standard_error = relative
        self.plateau_convention = plateau_convention
        self.plan_id = canonical_fingerprint(
            {
                "kind": "entanglement-estimator-plan",
                "monomer_number_density": density,
                "temperature": thermal,
                "boltzmann_constant": boltzmann,
                "block_count": blocks,
                "minimum_frames": minimum,
                "maximum_relative_standard_error": relative,
                "plateau_convention": plateau_convention,
            }
        )


class EntanglementEstimatorResult(StrictModule):
    primitive_path_step_length: Array
    coil_entanglement_length: Array
    kink_entanglement_length: Array
    plateau_modulus: Array
    block_coil_entanglement_length: Array
    coil_standard_error: Array
    coil_relative_standard_error: Array
    frame_count: Array
    chain_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def estimate_entanglement(
    plan: EntanglementEstimatorPlan,
    bond_counts: ArrayLike,
    end_to_end_squared: ArrayLike,
    primitive_contour_lengths: ArrayLike,
    /,
    *,
    kink_counts: ArrayLike | None = None,
) -> EntanglementEstimatorResult:
    if not isinstance(plan, EntanglementEstimatorPlan):
        raise TypeError("plan must be EntanglementEstimatorPlan.")
    bonds = jnp.asarray(bond_counts)
    ree_squared = jnp.asarray(end_to_end_squared)
    contour = jnp.asarray(primitive_contour_lengths, dtype=ree_squared.dtype)
    if ree_squared.ndim == 1:
        ree_squared = ree_squared[None, :]
    if contour.ndim == 1:
        contour = contour[None, :]
    if (
        bonds.ndim != 1
        or ree_squared.shape != contour.shape
        or ree_squared.shape[1] != bonds.size
    ):
        raise ValueError("Entanglement estimator arrays are misaligned.")
    frames, chains = ree_squared.shape
    mean_bonds = jnp.mean(bonds.astype(ree_squared.dtype))
    mean_ree = jnp.mean(ree_squared)
    mean_contour = jnp.mean(contour)
    step_length = mean_ree / mean_contour
    coil = mean_bonds * mean_ree / (mean_contour * mean_contour)
    if kink_counts is None:
        kink = jnp.asarray(jnp.nan, dtype=ree_squared.dtype)
        kink_valid = jnp.asarray(True)
    else:
        kinks = jnp.asarray(kink_counts, dtype=ree_squared.dtype)
        if kinks.ndim == 1:
            kinks = kinks[None, :]
        if kinks.shape != ree_squared.shape:
            raise ValueError("kink_counts must match frame and chain axes.")
        mean_kinks = jnp.mean(kinks)
        kink = mean_bonds / mean_kinks
        kink_valid = jnp.isfinite(kink) & (mean_kinks > 0.0)
    prefactor = 1.0 if plan.plateau_convention == "affine" else 4.0 / 5.0
    plateau = (
        prefactor
        * plan.monomer_number_density
        * plan.boltzmann_constant
        * plan.temperature
        / coil
    )
    block_size = frames // plan.block_count
    blocks = []
    for block_index in range(plan.block_count):
        start = block_index * block_size
        stop = (
            frames
            if block_index == plan.block_count - 1
            else (block_index + 1) * block_size
        )
        block_ree = jnp.mean(ree_squared[start:stop])
        block_contour = jnp.mean(contour[start:stop])
        blocks.append(mean_bonds * block_ree / (block_contour * block_contour))
    block_values = jnp.stack(blocks)
    standard_error = jnp.std(block_values, ddof=1) / jnp.sqrt(plan.block_count)
    relative_error = standard_error / jnp.maximum(
        jnp.abs(coil), jnp.finfo(ree_squared.dtype).tiny
    )
    finite = (
        jnp.all(bonds > 0)
        & jnp.all(jnp.isfinite(ree_squared))
        & jnp.all(jnp.isfinite(contour))
        & jnp.all(ree_squared > 0.0)
        & jnp.all(contour > 0.0)
        & jnp.isfinite(step_length)
        & jnp.isfinite(coil)
        & jnp.isfinite(plateau)
        & jnp.isfinite(standard_error)
        & kink_valid
    )
    successful = (
        finite
        & (frames >= plan.minimum_frames)
        & (block_size > 0)
        & (coil > 0.0)
        & (relative_error <= plan.maximum_relative_standard_error)
    )
    return EntanglementEstimatorResult(
        step_length,
        coil,
        kink,
        plateau,
        block_values,
        standard_error,
        relative_error,
        jnp.asarray(frames, dtype=jnp.int32),
        jnp.asarray(chains, dtype=jnp.int32),
        finite,
        successful,
        plan.plan_id,
    )


class MultiLengthKinkEstimatorResult(StrictModule):
    entanglement_length: Array
    slope: Array
    intercept: Array
    r_squared: Array
    successful: Array


def multi_length_kink_entanglement(
    chain_lengths: ArrayLike,
    mean_kink_counts: ArrayLike,
    /,
    *,
    minimum_r_squared: float = 0.95,
) -> MultiLengthKinkEstimatorResult:
    length = jnp.asarray(chain_lengths, dtype=float)
    kinks = jnp.asarray(mean_kink_counts, dtype=length.dtype)
    threshold = float(minimum_r_squared)
    if (
        length.ndim != 1
        or kinks.shape != length.shape
        or length.size < 3
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("Multi-length kink estimator inputs are invalid.")
    centered = length - jnp.mean(length)
    centered_kinks = kinks - jnp.mean(kinks)
    denominator = jnp.sum(centered * centered)
    slope = jnp.sum(centered * centered_kinks) / denominator
    intercept = jnp.mean(kinks) - slope * jnp.mean(length)
    residual = kinks - (slope * length + intercept)
    total = jnp.sum(centered_kinks * centered_kinks)
    r_squared = 1.0 - jnp.sum(residual * residual) / jnp.maximum(
        total, jnp.finfo(length.dtype).tiny
    )
    entanglement = 1.0 / slope
    successful = (
        jnp.all(jnp.isfinite(length))
        & jnp.all(jnp.isfinite(kinks))
        & jnp.all(jnp.diff(length) > 0.0)
        & (denominator > 0.0)
        & (slope > 0.0)
        & jnp.isfinite(entanglement)
        & (r_squared >= threshold)
    )
    return MultiLengthKinkEstimatorResult(
        entanglement, slope, intercept, r_squared, successful
    )


__all__ = [
    "EntanglementEstimatorPlan",
    "EntanglementEstimatorResult",
    "MultiLengthKinkEstimatorResult",
    "PlateauModulusConvention",
    "estimate_entanglement",
    "multi_length_kink_entanglement",
]

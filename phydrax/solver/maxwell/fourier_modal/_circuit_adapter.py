#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....circuit import MatrixScatteringComponent, ModalWaveReference, WavePort
from ._runtime import PreparedFourierModalMaxwell


def _mode_indices(
    available: tuple[str, ...],
    selected: Sequence[str | int],
    /,
    *,
    side: str,
) -> tuple[int, ...]:
    indices: list[int] = []
    for value in selected:
        index = int(value) if isinstance(value, int) else available.index(str(value))
        if index < 0 or index >= len(available):
            raise IndexError(f"{side} modal selection lies outside the prepared basis.")
        indices.append(index)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"{side} modal selection must be non-empty and unique.")
    return tuple(indices)


def _validate_eligible(
    prepared_modes, indices: tuple[int, ...], side: str, tolerance: float
) -> None:
    index = jnp.asarray(indices)
    if not bool(jnp.all(prepared_modes.propagating[index])):
        raise ValueError(f"{side} circuit modes must all be propagating.")
    if bool(jnp.any(prepared_modes.grazing[index])):
        raise ValueError(f"{side} circuit modes must be nongrazing.")
    if not bool(
        jnp.all(jnp.abs(jnp.abs(prepared_modes.flux_weights[index]) - 1.0) <= tolerance)
    ):
        raise ValueError(f"{side} circuit modes must have certified unit flux magnitude.")


def fourier_modal_scattering_component(
    prepared: PreparedFourierModalMaxwell,
    /,
    *,
    left_modes: Sequence[str | int],
    right_modes: Sequence[str | int],
    flux_tolerance: float = 1e-6,
    component_id: str | None = None,
) -> MatrixScatteringComponent:
    """Adapt selected far-field modes with canonical ``[[s21,s22],[s11,s12]]`` ordering."""
    if not isinstance(prepared, PreparedFourierModalMaxwell):
        raise TypeError("prepared must be PreparedFourierModalMaxwell.")
    if flux_tolerance < 0.0:
        raise ValueError("flux_tolerance must be non-negative.")
    scattering = prepared.scattering
    if not bool(scattering.diagnostics.finite):
        raise ValueError(
            "Fourier-modal scattering must be finite before circuit adaptation."
        )
    if not bool(scattering.diagnostics.power_normalized):
        raise ValueError("Fourier-modal scattering lacks certified power normalization.")
    left = _mode_indices(scattering.left_modes.mode_ids, left_modes, side="left")
    right = _mode_indices(scattering.right_modes.mode_ids, right_modes, side="right")
    _validate_eligible(scattering.left_modes, left, "left", flux_tolerance)
    _validate_eligible(scattering.right_modes, right, "right", flux_tolerance)
    li = jnp.asarray(left)
    ri = jnp.asarray(right)
    top_left = scattering.s21.matrix[li[:, None], li[None, :]]
    top_right = scattering.s22.matrix[li[:, None], ri[None, :]]
    bottom_left = scattering.s11.matrix[ri[:, None], li[None, :]]
    bottom_right = scattering.s12.matrix[ri[:, None], ri[None, :]]
    matrix = jnp.concatenate(
        (
            jnp.concatenate((top_left, top_right), axis=-1),
            jnp.concatenate((bottom_left, bottom_right), axis=-1),
        ),
        axis=-2,
    )
    problem = prepared.problem
    basis_id = canonical_fingerprint(
        {
            "kind": "fourier-modal-unit-flux-basis",
            "problem": problem.problem_id,
            "bloch_wavevector": array_tree_fingerprint(problem.bloch_wavevector),
            "angular_frequency": array_tree_fingerprint(problem.angular_frequency),
            "superstrate_material": {
                "id": problem.superstrate.material.material_id,
                "arrays": array_tree_fingerprint(problem.superstrate.material),
            },
            "substrate_material": {
                "id": problem.substrate.material.material_id,
                "arrays": array_tree_fingerprint(problem.substrate.material),
            },
            "left_basis_arrays": array_tree_fingerprint(scattering.left_modes),
            "right_basis_arrays": array_tree_fingerprint(scattering.right_modes),
            "left_modes": list(scattering.left_modes.mode_ids),
            "right_modes": list(scattering.right_modes.mode_ids),
        }
    )
    ports = []
    for side, indices, modes, reference_plane in (
        ("left", left, scattering.left_modes, problem.superstrate.reference_plane),
        ("right", right, scattering.right_modes, problem.substrate.reference_plane),
    ):
        mode_ids = tuple(modes.mode_ids[index] for index in indices)
        references = tuple(
            ModalWaveReference(
                basis_id,
                mode_id,
                polarization=mode_id.rsplit(":", 1)[-1],
                normalization="unit-flux",
                orientation="into-component",
                reference_plane=reference_plane,
            )
            for mode_id in mode_ids
        )
        ports.append(WavePort(side, references, coordinate_ids=mode_ids))
    identifier = (
        f"{problem.problem_id}/far-field-circuit"
        if component_id is None
        else str(component_id)
    )
    if not identifier:
        raise ValueError("component_id must be non-empty.")
    return MatrixScatteringComponent(
        matrix,
        ports,
        numeric_version=prepared.refresh_count,
        component_id=identifier,
    )


__all__ = ["fourier_modal_scattering_component"]

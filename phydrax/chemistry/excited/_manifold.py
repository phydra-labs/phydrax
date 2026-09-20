#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Method-neutral excited manifolds with representation-specific amplitudes."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition
from ._representation import ExcitedStateRepresentation


class ElectronicManifoldResult(StrictModule, NonTrainableState):
    excitation_energies: Array
    absolute_energies: Array
    representation: ExcitedStateRepresentation
    electric_transition_dipoles: Array
    magnetic_transition_dipoles: Array | None
    oscillator_strengths: Array
    rotatory_strengths: Array | None
    residuals: Array
    successful: Array
    method: str = eqx.field(static=True)
    spin_sector: str = eqx.field(static=True)
    symmetry_sector: str | None = eqx.field(static=True)
    clusters: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    energy_unit: UnitDefinition
    electric_dipole_unit: UnitDefinition
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        excitation_energies: ArrayLike,
        absolute_energies: ArrayLike,
        representation: ExcitedStateRepresentation,
        electric_transition_dipoles: ArrayLike,
        oscillator_strengths: ArrayLike,
        residuals: ArrayLike,
        successful: ArrayLike,
        method: str,
        spin_sector: str,
        clusters: tuple[tuple[int, ...], ...],
        energy_unit: UnitDefinition,
        electric_dipole_unit: UnitDefinition,
        /,
        *,
        symmetry_sector: str | None = None,
        magnetic_transition_dipoles: ArrayLike | None = None,
        rotatory_strengths: ArrayLike | None = None,
    ):
        excitation = jnp.asarray(excitation_energies)
        absolute = jnp.asarray(absolute_energies, dtype=excitation.dtype)
        electric_input = jnp.asarray(electric_transition_dipoles)
        property_dtype = jnp.result_type(excitation.dtype, electric_input.dtype)
        electric = electric_input.astype(property_dtype)
        oscillator = jnp.asarray(oscillator_strengths, dtype=excitation.real.dtype)
        residual = jnp.asarray(residuals, dtype=excitation.real.dtype)
        roots = excitation.size
        magnetic = (
            None
            if magnetic_transition_dipoles is None
            else jnp.asarray(magnetic_transition_dipoles, dtype=property_dtype)
        )
        rotatory = (
            None
            if rotatory_strengths is None
            else jnp.asarray(rotatory_strengths, dtype=excitation.real.dtype)
        )
        method_ = str(method).strip()
        spin = str(spin_sector).strip()
        symmetry = None if symmetry_sector is None else str(symmetry_sector).strip()
        if (
            absolute.shape != (roots,)
            or electric.shape != (roots, 3)
            or oscillator.shape != (roots,)
            or residual.shape != (roots,)
            or magnetic is not None
            and magnetic.shape != (roots, 3)
            or rotatory is not None
            and rotatory.shape != (roots,)
            or not method_
            or not spin
            or symmetry_sector is not None
            and not symmetry
        ):
            raise ValueError(
                "Excited-manifold energies, properties, or identities do not align."
            )
        if sum(len(cluster) for cluster in clusters) != roots or tuple(
            value for cluster in clusters for value in cluster
        ) != tuple(range(roots)):
            raise ValueError("Excited-state clusters must partition roots in order.")
        self.excitation_energies = excitation
        self.absolute_energies = absolute
        self.representation = representation
        self.electric_transition_dipoles = electric
        self.magnetic_transition_dipoles = magnetic
        self.oscillator_strengths = oscillator
        self.rotatory_strengths = rotatory
        self.residuals = residual
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.method = method_
        self.spin_sector = spin
        self.symmetry_sector = symmetry
        self.clusters = clusters
        self.energy_unit = energy_unit
        self.electric_dipole_unit = electric_dipole_unit
        self.result_id = canonical_fingerprint(
            {
                "kind": "electronic-manifold-result",
                "method": method_,
                "spin_sector": spin,
                "symmetry_sector": symmetry,
                "representation": representation.representation_id,
                "clusters": [list(value) for value in clusters],
                "energy_unit": energy_unit.unit_id,
                "electric_dipole_unit": electric_dipole_unit.unit_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "excitation_energies": np.asarray(excitation),
                        "absolute_energies": np.asarray(absolute),
                        "electric_transition_dipoles": np.asarray(electric),
                        "magnetic_transition_dipoles": None
                        if magnetic is None
                        else np.asarray(magnetic),
                        "oscillator_strengths": np.asarray(oscillator),
                        "rotatory_strengths": None
                        if rotatory is None
                        else np.asarray(rotatory),
                        "residuals": np.asarray(residual),
                    }
                ),
            }
        )


__all__ = ["ElectronicManifoldResult"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry, embedding, field, and guess state for electronic evaluations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticUnitSystem, PermanentMultipoleSiteData


class ElectrostaticEmbeddingState(StrictModule, NonTrainableState):
    """Fixed-capacity external point charges in native atomistic units."""

    point_ids: Array
    positions: Array
    charges: Array
    active_mask: Array
    units: AtomisticUnitSystem
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        point_ids: ArrayLike,
        positions: ArrayLike,
        charges: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ):
        ids = jnp.asarray(point_ids, dtype=jnp.int64)
        coordinate = jnp.asarray(positions)
        charge = jnp.asarray(charges, dtype=coordinate.dtype)
        if (
            ids.ndim != 1
            or coordinate.shape != (ids.size, 3)
            or charge.shape != ids.shape
        ):
            raise ValueError(
                "Embedding IDs, positions, and charges must have shapes (P,), (P,3), and (P,)."
            )
        active = (
            jnp.ones(ids.shape, dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != ids.shape:
            raise ValueError("Embedding active_mask must align with point IDs.")
        ids_host = np.asarray(ids)
        active_host = np.asarray(active)
        if len(set(int(value) for value in ids_host[active_host])) != int(
            np.count_nonzero(active_host)
        ):
            raise ValueError("Active embedding point IDs must be unique.")
        if np.any(~np.isfinite(np.asarray(coordinate)[active_host])) or np.any(
            ~np.isfinite(np.asarray(charge)[active_host])
        ):
            raise ValueError("Active embedding positions and charges must be finite.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        self.point_ids = ids
        self.positions = coordinate
        self.charges = charge
        self.active_mask = active
        self.units = units
        self.embedding_id = canonical_fingerprint(
            {
                "kind": "electrostatic-embedding-state",
                "units": units.unit_system_id,
                "arrays": array_tree_fingerprint(
                    {
                        "point_ids": ids_host,
                        "positions": np.asarray(coordinate),
                        "charges": np.asarray(charge),
                        "active_mask": active_host,
                    }
                ),
            }
        )


class ExternalFieldState(StrictModule, NonTrainableState):
    """Uniform electric and magnetic fields with an exact frequency-domain gauge."""

    electric: Array
    magnetic: Array
    angular_frequency: Array
    damping: Array
    gauge: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        magnetic: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_frequency: ArrayLike = 0.0,
        damping: ArrayLike = 0.0,
        gauge: str = "length",
    ):
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic, dtype=electric_.dtype)
        if electric_.shape != (3,) or magnetic_.shape != (3,):
            raise ValueError(
                "External electric and magnetic fields must have shape (3,)."
            )
        frequency = jnp.asarray(angular_frequency, dtype=electric_.dtype).reshape(())
        damping_ = jnp.asarray(damping, dtype=electric_.dtype).reshape(())
        values = np.asarray(
            jnp.concatenate((electric_, magnetic_, frequency[None], damping_[None]))
        )
        gauge_ = str(gauge).strip()
        if (
            np.any(~np.isfinite(values))
            or float(frequency) < 0.0
            or float(damping_) < 0.0
            or not gauge_
        ):
            raise ValueError(
                "External fields, frequency, damping, and gauge must be valid."
            )
        self.electric = electric_
        self.magnetic = magnetic_
        self.angular_frequency = frequency
        self.damping = damping_
        self.gauge = gauge_
        self.units = units
        self.field_id = canonical_fingerprint(
            {
                "kind": "external-field-state",
                "units": units.unit_system_id,
                "gauge": gauge_,
                "arrays": array_tree_fingerprint(
                    {
                        "electric": np.asarray(electric_),
                        "magnetic": np.asarray(magnetic_),
                        "angular_frequency": np.asarray(frequency),
                        "damping": np.asarray(damping_),
                    }
                ),
            }
        )


class ElectronicInitialGuessState(StrictModule, NonTrainableState):
    """A basis- and sector-bound density/orbital guess."""

    density: Array
    coefficients: Array | None
    occupations: Array | None
    kind: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)
    guess_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        /,
        *,
        kind: str,
        basis_id: str,
        sector_id: str,
        coefficients: ArrayLike | None = None,
        occupations: ArrayLike | None = None,
    ):
        density_ = jnp.asarray(density)
        if density_.ndim not in (2, 3) or density_.shape[-1] != density_.shape[-2]:
            raise ValueError("Guess density must contain one or more square matrices.")
        coefficients_ = (
            None
            if coefficients is None
            else jnp.asarray(coefficients, dtype=density_.dtype)
        )
        occupations_ = (
            None
            if occupations is None
            else jnp.asarray(occupations, dtype=density_.real.dtype)
        )
        if coefficients_ is not None and (
            coefficients_.ndim not in (2, 3)
            or coefficients_.shape[-2] != density_.shape[-1]
        ):
            raise ValueError("Guess coefficients do not align with the AO density.")
        if occupations_ is not None and coefficients_ is None:
            raise ValueError("Guess occupations require orbital coefficients.")
        kind_ = str(kind).strip()
        basis = str(basis_id).strip()
        sector = str(sector_id).strip()
        arrays = {
            "density": np.asarray(density_),
            "coefficients": (
                None if coefficients_ is None else np.asarray(coefficients_)
            ),
            "occupations": (None if occupations_ is None else np.asarray(occupations_)),
        }
        if (
            not kind_
            or not basis
            or not sector
            or any(
                value is not None and np.any(~np.isfinite(value))
                for value in arrays.values()
            )
        ):
            raise ValueError("Electronic guess values and identities must be finite.")
        self.density = density_
        self.coefficients = coefficients_
        self.occupations = occupations_
        self.kind = kind_
        self.basis_id = basis
        self.sector_id = sector
        self.guess_id = canonical_fingerprint(
            {
                "kind": "electronic-initial-guess",
                "guess_kind": kind_,
                "basis": basis,
                "sector": sector,
                "arrays": array_tree_fingerprint(arrays),
            }
        )


class PermanentMultipoleEmbeddingState(StrictModule, NonTrainableState):
    """External permanent multipoles on stable fixed-capacity sites."""

    point_ids: Array
    positions: Array
    multipoles: PermanentMultipoleSiteData
    active_mask: Array
    units: AtomisticUnitSystem
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        point_ids: ArrayLike,
        positions: ArrayLike,
        multipoles: PermanentMultipoleSiteData,
        units: AtomisticUnitSystem,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ):
        ids = jnp.asarray(point_ids, dtype=jnp.int64)
        positions_ = jnp.asarray(positions)
        if not isinstance(multipoles, PermanentMultipoleSiteData):
            raise TypeError("multipoles must be PermanentMultipoleSiteData.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if ids.ndim != 1 or positions_.shape != (ids.size, 3):
            raise ValueError(
                "Multipole IDs and positions must have shapes (P,) and (P, 3)."
            )
        if multipoles.site_capacity != ids.size:
            raise ValueError("Multipole capacity must match embedding site IDs.")
        active = (
            jnp.ones(ids.shape, dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != ids.shape:
            raise ValueError("Multipole active_mask must align with site IDs.")
        active_host = np.asarray(active)
        if len(set(int(value) for value in np.asarray(ids)[active_host])) != int(
            np.count_nonzero(active_host)
        ) or np.any(~np.isfinite(np.asarray(positions_)[active_host])):
            raise ValueError(
                "Active multipole sites require unique IDs and finite positions."
            )
        self.point_ids = ids
        self.positions = positions_
        self.multipoles = multipoles
        self.active_mask = active
        self.units = units
        self.embedding_id = canonical_fingerprint(
            {
                "kind": "permanent-multipole-embedding",
                "multipoles": multipoles.multipole_id,
                "units": units.unit_system_id,
                "arrays": array_tree_fingerprint(
                    {
                        "ids": np.asarray(ids),
                        "positions": np.asarray(positions_),
                        "active": active_host,
                    }
                ),
            }
        )


class PolarizableEmbeddingState(StrictModule, NonTrainableState):
    """Permanent embedding plus the accepted induced-dipole root."""

    permanent: PermanentMultipoleEmbeddingState
    induced_dipoles: Array
    residual: Array
    solver_state_id: str = eqx.field(static=True)
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        permanent: PermanentMultipoleEmbeddingState,
        induced_dipoles: ArrayLike,
        residual: ArrayLike,
        solver_state_id: str,
        /,
    ):
        if not isinstance(permanent, PermanentMultipoleEmbeddingState):
            raise TypeError("permanent must be PermanentMultipoleEmbeddingState.")
        induced = jnp.asarray(induced_dipoles, dtype=permanent.positions.dtype)
        residual_ = jnp.asarray(residual, dtype=induced.real.dtype).reshape(())
        solver = str(solver_state_id).strip()
        if (
            induced.shape != permanent.positions.shape
            or np.any(~np.isfinite(np.asarray(induced)))
            or not np.isfinite(float(residual_))
            or float(residual_) < 0.0
            or not solver
        ):
            raise ValueError("Polarizable embedding state is invalid.")
        self.permanent = permanent
        self.induced_dipoles = induced
        self.residual = residual_
        self.solver_state_id = solver
        self.embedding_id = canonical_fingerprint(
            {
                "kind": "polarizable-embedding-state",
                "permanent": permanent.embedding_id,
                "solver_state": solver,
                "arrays": array_tree_fingerprint(
                    {
                        "induced_dipoles": np.asarray(induced),
                        "residual": np.asarray(residual_),
                    }
                ),
            }
        )


class ElectronicEvaluationContext(StrictModule, NonTrainableState):
    """All dynamic state supplied to one prepared electronic calculation."""

    positions: Array
    cell_vectors: Array | None
    embedding: ElectrostaticEmbeddingState | None
    multipole_embedding: PermanentMultipoleEmbeddingState | None
    polarizable_embedding: PolarizableEmbeddingState | None
    external_field: ExternalFieldState | None
    initial_guess: ElectronicInitialGuessState | None
    time: Array
    topology_epoch_id: str = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        embedding: ElectrostaticEmbeddingState | None = None,
        multipole_embedding: PermanentMultipoleEmbeddingState | None = None,
        polarizable_embedding: PolarizableEmbeddingState | None = None,
        external_field: ExternalFieldState | None = None,
        initial_guess: ElectronicInitialGuessState | None = None,
        time: ArrayLike = 0.0,
        topology_epoch_id: str = "fixed",
    ):
        coordinate = jnp.asarray(positions)
        if coordinate.ndim != 2 or coordinate.shape[1] != 3:
            raise ValueError("Electronic positions must have shape (atom_capacity, 3).")
        cell = (
            None
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=coordinate.dtype)
        )
        if cell is not None and cell.shape != (3, 3):
            raise ValueError("Electronic cell_vectors must have shape (3, 3).")
        if embedding is not None and not isinstance(
            embedding, ElectrostaticEmbeddingState
        ):
            raise TypeError("embedding must be ElectrostaticEmbeddingState or None.")
        if multipole_embedding is not None and not isinstance(
            multipole_embedding, PermanentMultipoleEmbeddingState
        ):
            raise TypeError(
                "multipole_embedding must be PermanentMultipoleEmbeddingState or None."
            )
        if polarizable_embedding is not None and not isinstance(
            polarizable_embedding, PolarizableEmbeddingState
        ):
            raise TypeError(
                "polarizable_embedding must be PolarizableEmbeddingState or None."
            )
        if external_field is not None and not isinstance(
            external_field, ExternalFieldState
        ):
            raise TypeError("external_field must be ExternalFieldState or None.")
        if initial_guess is not None and not isinstance(
            initial_guess, ElectronicInitialGuessState
        ):
            raise TypeError("initial_guess must be ElectronicInitialGuessState or None.")
        time_ = jnp.asarray(time, dtype=coordinate.dtype).reshape(())
        topology = str(topology_epoch_id).strip()
        if (
            np.any(~np.isfinite(np.asarray(coordinate)))
            or (cell is not None and np.any(~np.isfinite(np.asarray(cell))))
            or not np.isfinite(float(time_))
            or not topology
        ):
            raise ValueError(
                "Electronic geometry, time, and topology epoch must be valid."
            )
        arrays = {
            "positions": np.asarray(coordinate),
            "cell_vectors": None if cell is None else np.asarray(cell),
            "time": np.asarray(time_),
        }
        self.positions = coordinate
        self.cell_vectors = cell
        self.embedding = embedding
        self.multipole_embedding = multipole_embedding
        self.polarizable_embedding = polarizable_embedding
        self.external_field = external_field
        self.initial_guess = initial_guess
        self.time = time_
        self.topology_epoch_id = topology
        self.context_id = canonical_fingerprint(
            {
                "kind": "electronic-evaluation-context",
                "arrays": array_tree_fingerprint(arrays),
                "embedding": None if embedding is None else embedding.embedding_id,
                "multipole_embedding": (
                    None
                    if multipole_embedding is None
                    else multipole_embedding.embedding_id
                ),
                "polarizable_embedding": (
                    None
                    if polarizable_embedding is None
                    else polarizable_embedding.embedding_id
                ),
                "external_field": (
                    None if external_field is None else external_field.field_id
                ),
                "initial_guess": (
                    None if initial_guess is None else initial_guess.guess_id
                ),
                "topology_epoch": topology,
            }
        )


__all__ = [
    "ElectronicEvaluationContext",
    "ElectronicInitialGuessState",
    "ElectrostaticEmbeddingState",
    "ExternalFieldState",
    "PermanentMultipoleEmbeddingState",
    "PolarizableEmbeddingState",
]

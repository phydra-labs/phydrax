#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native AIMNet2 ``DispParam`` coefficients and ``D3TS`` pair dispersion.

This is the source-specific dispersion of AIMNetcentral 0.2.0
(``aimnet/modules/lr.py``, ``DispParam`` and ``D3TS``), not conventional
coordination-dependent DFT-D3. Atomic coefficients are a species reference
``(C6, alpha)`` multiplied elementwise by ``exp(clamp(raw, -4, 4))`` of a
network output. Pairs use the Tkatchenko-Scheffler combination rule

``C6_ij = 2 C6_i C6_j / max(C6_i alpha_j / alpha_i + C6_j alpha_i / alpha_j, 1e-4)``

and a Becke-Johnson-like rational damping with ``r0 = a1 sqrt(3 r4r2_i r4r2_j) + a2``
in bohr:

``e_ij = C6_ij (s6 / (d^6 + r0^6) + s8 3 r4r2_i r4r2_j / (d^8 + r0^8))``.

The energy of one directed route is ``-Hartree/2 e_ij`` in eV, so a full
directed pair list counts every unordered pair (and every periodic image route)
exactly once in total. The two clamps are declared model nonlinearities: their
kinks (``raw = +-4`` and the denominator floor) keep the source's one-sided
torch ``clamp`` derivative convention (the value passes its gradient at and
inside a bound, zero outside). Distances are angstrom; ``C6`` uses
hartree-bohr^6; only polarizability ratios enter the energy, so ``alpha`` is in
the unit of the admitted reference table. There is no smooth cutoff: the source
sums every route of its dispersion neighbor list, so ``pair_cutoff`` is a hard,
explicitly declared truncation radius that enters identity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import array_tree_fingerprint
from .._identity import IdentityWords
from .._model import register_artifact_value
from .._strict import StrictModule
from .._trainable import NonTrainableState


# AIMNetcentral 0.2.0 ``aimnet/constants.py``.
SOURCE_BOHR_ANGSTROM = 0.5291772105638411
SOURCE_HARTREE_ELECTRONVOLT = 27.211386024367243
# ``DispParam`` bounds the log multiplier and ``D3TS`` floors the TS denominator.
SOURCE_LOG_MULTIPLIER_BOUND = 4.0
SOURCE_COMBINATION_DENOMINATOR_FLOOR = 1.0e-4


def source_clamp(value: Array, lower: float | None, upper: float | None, /) -> Array:
    """``torch.clamp`` with its one-sided derivative convention.

    The gradient passes where ``lower <= value <= upper`` (bounds inclusive) and
    is zero outside, unlike ``jnp.clip`` which splits a tie.
    """
    inside = jnp.ones(value.shape, dtype=jnp.bool_)
    result = value
    if lower is not None:
        bound = jnp.asarray(lower, dtype=value.dtype)
        inside = inside & (value >= bound)
        result = jnp.where(value < bound, bound, result)
    if upper is not None:
        bound = jnp.asarray(upper, dtype=value.dtype)
        inside = inside & (value <= bound)
        result = jnp.where(value > bound, bound, result)
    return jnp.where(inside, value, result)


def _species(values: Sequence[int], name: str, /) -> tuple[int, ...]:
    species = tuple(int(value) for value in values)
    if any(isinstance(value, bool) for value in values):
        raise TypeError(f"{name} must contain integers.")
    if not species or any(value <= 0 for value in species):
        raise ValueError(f"{name} must name positive atomic numbers.")
    if len(set(species)) != len(species) or tuple(sorted(species)) != species:
        raise ValueError(f"{name} must be strictly increasing.")
    return species


class TSDispersionReference(StrictModule, NonTrainableState):
    """Source ``DispParam``: species reference ``(C6, alpha)`` and bounded multipliers.

    ``table[Z] = (C6_Z, alpha_Z)``; row zero is the declared padding dummy
    ``(0, 1)``. Every implemented species must have finite positive reference
    values (coverage is checked here, never inferred at evaluation). The
    predicted atomic coefficients are ``table[Z] * exp(clamp(raw, -4, 4))``.

    PyTree layout: the table shape is the layout. The implemented species are
    the dynamic row mask ``implemented_mask`` (``bool[S]``, decoded by the host
    property ``implemented_species``) and ``reference_identity`` is the exact
    former ``reference_id`` digest (decoded by the host property).
    """

    table: Array
    implemented_mask: Array
    reference_identity: IdentityWords

    def __init__(
        self,
        table: ArrayLike,
        /,
        *,
        implemented_species: Sequence[int],
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        reference = np.asarray(table)
        if reference.dtype.kind != "f":
            raise TypeError("The dispersion reference table must be floating point.")
        if reference.ndim != 2 or reference.shape[1] != 2 or reference.shape[0] < 2:
            raise ValueError("The dispersion reference table must have shape (S, 2).")
        if not np.all(np.isfinite(reference)):
            raise ValueError("The dispersion reference table must be finite.")
        if not np.array_equal(reference[0], np.asarray([0.0, 1.0], reference.dtype)):
            raise ValueError(
                "Dispersion reference row 0 must be the source padding dummy (0, 1)."
            )
        species = _species(implemented_species, "implemented_species")
        if species[-1] >= reference.shape[0]:
            raise ValueError("An implemented species lies outside the reference table.")
        covered = reference[np.asarray(species)]
        if np.any(covered <= 0.0):
            raise ValueError(
                "Every implemented species needs positive reference C6 and polarizability."
            )
        values = jnp.asarray(reference, dtype=dtype)
        mask = np.zeros((reference.shape[0],), dtype=np.bool_)
        mask[np.asarray(species)] = True
        self.table = values
        self.implemented_mask = jnp.asarray(mask)
        executed = np.asarray(jax.device_get(values))
        self.reference_identity = IdentityWords.of(
            {
                "kind": "aimnet2-dispersion-reference",
                "table": array_tree_fingerprint(executed),
                "dtype": executed.dtype.name,
                "implemented_species": species,
                "log_multiplier_bound": SOURCE_LOG_MULTIPLIER_BOUND,
            }
        )

    @property
    def implemented_species(self) -> tuple[int, ...]:
        """Host decoding of the implemented atomic numbers (increasing)."""
        return tuple(
            int(value) for value in np.flatnonzero(np.asarray(self.implemented_mask))
        )

    @property
    def reference_id(self) -> str:
        """Host decoding of the canonical dispersion-reference digest."""
        return self.reference_identity.fingerprint()

    def rebuilt(self) -> TSDispersionReference:
        """The constructor's value from this reference's own stored leaves."""
        return TSDispersionReference(
            np.asarray(jax.device_get(self.table)),
            implemented_species=self.implemented_species,
            dtype=self.table.dtype,
        )

    @property
    def species_capacity(self) -> int:
        return int(self.table.shape[0])

    def coefficients(self, raw: Array, species: Array, /) -> Array:
        """Atomic ``(C6, alpha)`` of rows ``species`` from raw ``(..., 2)`` outputs."""
        index = jnp.clip(species, 0, self.species_capacity - 1)
        bounded = source_clamp(
            raw, -SOURCE_LOG_MULTIPLIER_BOUND, SOURCE_LOG_MULTIPLIER_BOUND
        )
        return self.table[index].astype(raw.dtype) * jnp.exp(bounded)


class TSDispersionPotential(StrictModule, NonTrainableState):
    """Source ``D3TS`` pair energy over an explicitly counted directed route set.

    ``pair_cutoff`` (angstrom) is the hard route radius: the source sums every
    route of its dispersion neighbor list without a switching function, so the
    energy jumps where a route crosses it. Every directed route (including
    nonzero periodic self images) contributes ``-Hartree/2 e_ij`` once;
    ``route_energies`` accepts an explicit integer route multiplicity for
    half-stored lists. The coordinate, cell and coefficient derivatives are
    the derivatives of this closed form, so a model that predicts the
    coefficients from geometry gets the connected response by composition.

    ``r4r2`` is a caller-supplied buffer indexed by atomic number; no source
    element table is bundled or reconstructed implicitly. Source conversion
    carries the exact buffer from the admitted source inventory.

    PyTree layout: ``pair_cutoff`` is the declared route radius (it defines the
    model's dispersion neighborhood) and stays static. ``s6``, ``s8``, ``a1``
    and ``a2`` are dynamic float64 scalars and ``potential_identity`` is the
    exact former ``potential_id`` digest (decoded by the host property).
    """

    r4r2: Array
    s6: Array
    s8: Array
    a1: Array
    a2: Array
    pair_cutoff: float = eqx.field(static=True)
    potential_identity: IdentityWords

    def __init__(
        self,
        *,
        a1: float,
        a2: float,
        s8: float,
        s6: float = 1.0,
        pair_cutoff: float,
        r4r2: ArrayLike,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        constants = {"s6": s6, "s8": s8, "a1": a1, "a2": a2}
        parsed = {}
        for name, value in constants.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
            parsed[name] = float(value)
        cutoff = float(pair_cutoff)
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("pair_cutoff must be finite and positive.")
        table = np.asarray(r4r2)
        if table.dtype.kind != "f" or table.ndim != 1 or table.shape[0] < 2:
            raise ValueError("r4r2 must be a floating table indexed by atomic number.")
        if not np.all(np.isfinite(table)) or np.any(table < 0.0):
            raise ValueError("r4r2 must be finite and nonnegative.")
        self.r4r2 = jnp.asarray(table, dtype=dtype)
        self.s6 = jnp.asarray(parsed["s6"], dtype=jnp.float64)
        self.s8 = jnp.asarray(parsed["s8"], dtype=jnp.float64)
        self.a1 = jnp.asarray(parsed["a1"], dtype=jnp.float64)
        self.a2 = jnp.asarray(parsed["a2"], dtype=jnp.float64)
        self.pair_cutoff = cutoff
        executed = np.asarray(jax.device_get(self.r4r2))
        self.potential_identity = IdentityWords.of(
            {
                "kind": "aimnet2-ts-dispersion",
                **parsed,
                "pair_cutoff": cutoff,
                "r4r2": array_tree_fingerprint(executed),
                "r4r2_dtype": executed.dtype.name,
                "combination_floor": SOURCE_COMBINATION_DENOMINATOR_FLOOR,
                "bohr": SOURCE_BOHR_ANGSTROM,
                "hartree": SOURCE_HARTREE_ELECTRONVOLT,
            }
        )

    @property
    def potential_id(self) -> str:
        """Host decoding of the canonical TS dispersion digest."""
        return self.potential_identity.fingerprint()

    def rebuilt(self) -> TSDispersionPotential:
        """The constructor's value from this term's own stored leaves."""
        return TSDispersionPotential(
            a1=float(self.a1),
            a2=float(self.a2),
            s8=float(self.s8),
            s6=float(self.s6),
            pair_cutoff=self.pair_cutoff,
            r4r2=np.asarray(jax.device_get(self.r4r2)),
            dtype=self.r4r2.dtype,
        )

    def require_coverage(self, species: Sequence[int], /) -> None:
        """Refuse species without a positive ``r4r2`` reference."""
        table = np.asarray(self.r4r2)
        for value in species:
            if not 0 < int(value) < table.size or table[int(value)] <= 0.0:
                raise ValueError(f"Species {value} has no r4r2 dispersion reference.")

    def pair_energy(
        self,
        receiver: tuple[Array, Array],
        sender: tuple[Array, Array],
        distance: Array,
        /,
    ) -> Array:
        """Energy (eV) of directed routes; ``receiver``/``sender`` are ``(C6 alpha, Z)``.

        Coefficient arrays end in ``(C6, alpha)``; ``distance`` is angstrom.
        """
        coefficients_i, species_i = receiver
        coefficients_j, species_j = sender
        c6_i, alpha_i = coefficients_i[..., 0], coefficients_i[..., 1]
        c6_j, alpha_j = coefficients_j[..., 0], coefficients_j[..., 1]
        denominator = source_clamp(
            c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j,
            SOURCE_COMBINATION_DENOMINATOR_FLOOR,
            None,
        )
        c6 = 2.0 * c6_i * c6_j / denominator
        size = self.r4r2.shape[0]
        rr = (
            3.0
            * self.r4r2[jnp.clip(species_i, 0, size - 1)]
            * self.r4r2[jnp.clip(species_j, 0, size - 1)]
        ).astype(c6.dtype)
        r0 = self.a1.astype(c6.dtype) * jnp.sqrt(rr) + self.a2.astype(c6.dtype)
        bohr = distance.astype(c6.dtype) / SOURCE_BOHR_ANGSTROM
        damped = self.s6.astype(c6.dtype) / (bohr**6 + r0**6) + self.s8.astype(
            c6.dtype
        ) * rr / (bohr**8 + r0**8)
        return -0.5 * SOURCE_HARTREE_ELECTRONVOLT * c6 * damped

    def route_energies(
        self,
        coefficients: Array,
        species: Array,
        receivers: Array,
        senders: Array,
        distance: Array,
        active: Array,
        /,
        *,
        multiplicity: Array | None = None,
    ) -> Array:
        """Per-route energies, zero on inactive routes and beyond ``pair_cutoff``.

        ``multiplicity`` (nonnegative integers) counts how many directed routes
        one stored route represents; it defaults to one for full directed lists.
        """
        counted = active & (distance < jnp.asarray(self.pair_cutoff, distance.dtype))
        safe = jnp.where(counted, distance, jnp.ones((), dtype=distance.dtype))
        energy = self.pair_energy(
            (coefficients[receivers], species[receivers]),
            (coefficients[senders], species[senders]),
            safe,
        )
        if multiplicity is not None:
            energy = energy * jnp.asarray(multiplicity).astype(energy.dtype)
        return jnp.where(counted, energy, jnp.zeros((), dtype=energy.dtype))


register_artifact_value("phydrax.atomistic:TSDispersionReference", TSDispersionReference)
register_artifact_value("phydrax.atomistic:TSDispersionPotential", TSDispersionPotential)


__all__ = [
    "SOURCE_BOHR_ANGSTROM",
    "SOURCE_COMBINATION_DENOMINATOR_FLOOR",
    "SOURCE_HARTREE_ELECTRONVOLT",
    "SOURCE_LOG_MULTIPLIER_BOUND",
    "source_clamp",
    "TSDispersionPotential",
    "TSDispersionReference",
]

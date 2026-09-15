#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Angular-momentum labels, electric-dipole rules, and frame rotations.

All angular momenta and magnetic projections are represented by doubled
integers.  This makes half-integer labels exact and gives every public
selection-rule decision host-static semantics.  Spherical vector components
are ordered ``q=(-1, 0, +1)`` and rotations are active ZYZ rotations.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...tensor_network._su2 import su2_clebsch_gordan, su2_fusion, su2_wigner_6j


def _doubled(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a doubled integer angular momentum.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _projection(value: int, angular_momentum: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a doubled integer projection.")
    result = int(value)
    if abs(result) > angular_momentum or (result + angular_momentum) % 2:
        raise ValueError(f"{name} is not a magnetic projection of its manifold.")
    return result


def wigner_3j(
    twice_j1: int,
    twice_j2: int,
    twice_j3: int,
    twice_m1: int,
    twice_m2: int,
    twice_m3: int,
    /,
) -> float:
    """Condon--Shortley Wigner 3j symbol for doubled integer arguments."""

    j1 = _doubled(twice_j1, "twice_j1")
    j2 = _doubled(twice_j2, "twice_j2")
    j3 = _doubled(twice_j3, "twice_j3")
    m1 = _projection(twice_m1, j1, "twice_m1")
    m2 = _projection(twice_m2, j2, "twice_m2")
    m3 = _projection(twice_m3, j3, "twice_m3")
    if m1 + m2 + m3 != 0 or j3 not in su2_fusion(j1, j2):
        return 0.0
    table = su2_clebsch_gordan(j1, j2, j3)
    coefficient = float(table[(m1 + j1) // 2, (m2 + j2) // 2, (-m3 + j3) // 2])
    phase = -1.0 if ((j1 - j2 - m3) // 2) % 2 else 1.0
    return phase * coefficient / math.sqrt(j3 + 1.0)


def wigner_6j(
    twice_j1: int,
    twice_j2: int,
    twice_j3: int,
    twice_j4: int,
    twice_j5: int,
    twice_j6: int,
    /,
) -> float:
    """Racah Wigner 6j symbol for doubled integer arguments."""

    values = tuple(
        _doubled(value, name)
        for value, name in zip(
            (twice_j1, twice_j2, twice_j3, twice_j4, twice_j5, twice_j6),
            (
                "twice_j1",
                "twice_j2",
                "twice_j3",
                "twice_j4",
                "twice_j5",
                "twice_j6",
            ),
            strict=True,
        )
    )
    return su2_wigner_6j(*values)


class AtomicManifold(StrictModule, NonTrainableState):
    """One hyperfine manifold ``|(J I) F mF>`` with a common bare frequency."""

    angular_frequency: Array
    label: str = eqx.field(static=True)
    twice_electronic_j: int = eqx.field(static=True)
    twice_nuclear_i: int = eqx.field(static=True)
    twice_total_f: int = eqx.field(static=True)
    parity: int = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        /,
        *,
        twice_electronic_j: int,
        twice_nuclear_i: int,
        twice_total_f: int,
        parity: int,
        angular_frequency: float,
    ):
        label_ = str(label)
        if not label_:
            raise ValueError("Atomic manifold labels must be nonempty.")
        electronic = _doubled(twice_electronic_j, "twice_electronic_j")
        nuclear = _doubled(twice_nuclear_i, "twice_nuclear_i")
        total = _doubled(twice_total_f, "twice_total_f")
        if total not in su2_fusion(electronic, nuclear):
            raise ValueError("twice_total_f is forbidden by electronic/nuclear fusion.")
        if (
            isinstance(parity, bool)
            or not isinstance(parity, (int, np.integer))
            or int(parity) not in (-1, 1)
        ):
            raise ValueError("Atomic parity must be exactly -1 or +1.")
        parity_ = int(parity)
        frequency = float(angular_frequency)
        if not math.isfinite(frequency):
            raise ValueError("angular_frequency must be finite.")
        manifold_id = canonical_fingerprint(
            {
                "kind": "atomic-angular-momentum-manifold",
                "label": label_,
                "twice_electronic_j": electronic,
                "twice_nuclear_i": nuclear,
                "twice_total_f": total,
                "parity": parity_,
                "angular_frequency": frequency,
            }
        )
        self.angular_frequency = jnp.asarray(frequency)
        self.label = label_
        self.twice_electronic_j = electronic
        self.twice_nuclear_i = nuclear
        self.twice_total_f = total
        self.parity = parity_
        self.manifold_id = manifold_id

    @property
    def dimension(self) -> int:
        return self.twice_total_f + 1

    @property
    def magnetic_projections(self) -> tuple[int, ...]:
        return tuple(range(-self.twice_total_f, self.twice_total_f + 1, 2))

    def validate_projection(self, twice_m: int, /) -> int:
        return _projection(twice_m, self.twice_total_f, "twice_m")


def electric_dipole_allowed(
    bra: AtomicManifold,
    ket: AtomicManifold,
    /,
    *,
    twice_m_bra: int | None = None,
    twice_m_ket: int | None = None,
    q: int | None = None,
) -> bool:
    """Apply parity, Delta-J/F, nuclear-spin, and optional Delta-m rules."""

    if not isinstance(bra, AtomicManifold) or not isinstance(ket, AtomicManifold):
        raise TypeError("bra and ket must be AtomicManifold instances.")
    electronic = (
        bra.twice_nuclear_i == ket.twice_nuclear_i
        and bra.parity == -ket.parity
        and 2 in su2_fusion(bra.twice_electronic_j, ket.twice_electronic_j)
        and not (bra.twice_electronic_j == ket.twice_electronic_j == 0)
    )
    hyperfine = 2 in su2_fusion(bra.twice_total_f, ket.twice_total_f) and not (
        bra.twice_total_f == ket.twice_total_f == 0
    )
    if not electronic or not hyperfine:
        return False
    supplied = (twice_m_bra is not None, twice_m_ket is not None, q is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("Magnetic selection requires twice_m_bra, twice_m_ket, and q.")
    if all(supplied):
        magnetic_bra = bra.validate_projection(twice_m_bra)
        magnetic_ket = ket.validate_projection(twice_m_ket)
        if (
            isinstance(q, bool)
            or not isinstance(q, (int, np.integer))
            or int(q) not in (-1, 0, 1)
        ):
            raise ValueError("Dipole q must be -1, 0, or +1.")
        return magnetic_bra == magnetic_ket + 2 * int(q)
    return True


def hyperfine_dipole_coefficient(
    bra: AtomicManifold,
    ket: AtomicManifold,
    twice_m_bra: int,
    twice_m_ket: int,
    q: int,
    /,
) -> float:
    """Return ``<bra|d_q|ket>/<J_bra||d||J_ket>`` in a fixed phase gauge."""

    if not electric_dipole_allowed(
        bra,
        ket,
        twice_m_bra=twice_m_bra,
        twice_m_ket=twice_m_ket,
        q=q,
    ):
        return 0.0
    magnetic_bra = bra.validate_projection(twice_m_bra)
    magnetic_ket = ket.validate_projection(twice_m_ket)
    hyperfine_phase_exponent = (
        bra.twice_electronic_j + bra.twice_nuclear_i + ket.twice_total_f + 2
    ) // 2
    reduced_phase = -1.0 if hyperfine_phase_exponent % 2 else 1.0
    recoupling = wigner_6j(
        bra.twice_electronic_j,
        bra.twice_total_f,
        bra.twice_nuclear_i,
        ket.twice_total_f,
        ket.twice_electronic_j,
        2,
    )
    reduced = (
        reduced_phase
        * math.sqrt((bra.twice_total_f + 1) * (ket.twice_total_f + 1))
        * recoupling
    )
    magnetic_phase = -1.0 if ((bra.twice_total_f - magnetic_bra) // 2) % 2 else 1.0
    angular = wigner_3j(
        bra.twice_total_f,
        2,
        ket.twice_total_f,
        -magnetic_bra,
        2 * int(q),
        magnetic_ket,
    )
    return magnetic_phase * angular * reduced


def cartesian_rotation_zyz(euler_angles: ArrayLike, /) -> Array:
    """Return the active Cartesian ``Rz(alpha) Ry(beta) Rz(gamma)`` matrix."""

    angles = jnp.asarray(euler_angles)
    if angles.shape != (3,):
        raise ValueError("euler_angles must have shape (3,).")
    alpha, beta, gamma = angles
    zero = jnp.zeros((), dtype=angles.dtype)
    one = jnp.ones((), dtype=angles.dtype)
    rz_alpha = jnp.stack(
        (
            jnp.stack((jnp.cos(alpha), -jnp.sin(alpha), zero)),
            jnp.stack((jnp.sin(alpha), jnp.cos(alpha), zero)),
            jnp.stack((zero, zero, one)),
        )
    )
    ry_beta = jnp.stack(
        (
            jnp.stack((jnp.cos(beta), zero, jnp.sin(beta))),
            jnp.stack((zero, one, zero)),
            jnp.stack((-jnp.sin(beta), zero, jnp.cos(beta))),
        )
    )
    rz_gamma = jnp.stack(
        (
            jnp.stack((jnp.cos(gamma), -jnp.sin(gamma), zero)),
            jnp.stack((jnp.sin(gamma), jnp.cos(gamma), zero)),
            jnp.stack((zero, zero, one)),
        )
    )
    return rz_alpha @ ry_beta @ rz_gamma


def spherical_rotation_matrix(euler_angles: ArrayLike, /) -> Array:
    """Return the spin-one rotation in component order ``(-1, 0, +1)``."""

    angles = jnp.asarray(euler_angles)
    real_dtype = angles.dtype
    complex_dtype = jnp.complex64 if real_dtype.itemsize <= 4 else jnp.complex128
    root_two = jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype))
    cartesian_to_spherical = (
        jnp.asarray(
            (
                (1.0, -1.0j, 0.0),
                (0.0, 0.0, math.sqrt(2.0)),
                (-1.0, -1.0j, 0.0),
            ),
            dtype=complex_dtype,
        )
        / root_two
    )
    rotation = cartesian_rotation_zyz(angles).astype(complex_dtype)
    return cartesian_to_spherical @ rotation @ jnp.conj(cartesian_to_spherical.T)


def rotate_spherical_vector(
    components: ArrayLike,
    euler_angles: ArrayLike,
    /,
) -> Array:
    """Actively rotate spherical vector components in ``(-1, 0, +1)`` order."""

    values = jnp.asarray(components)
    if values.shape != (3,):
        raise ValueError("Spherical components must have shape (3,).")
    rotation = spherical_rotation_matrix(euler_angles).astype(
        jnp.result_type(values.dtype, 1j)
    )
    return ein.contract("ab,b->a", rotation, values, backend="jax")


__all__ = [
    "AtomicManifold",
    "cartesian_rotation_zyz",
    "electric_dipole_allowed",
    "hyperfine_dipole_coefficient",
    "rotate_spherical_vector",
    "spherical_rotation_matrix",
    "wigner_3j",
    "wigner_6j",
]

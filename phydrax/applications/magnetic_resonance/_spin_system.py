#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-spin records and local-Hamiltonian assembly."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import MaterializationPolicy
from ...operators.quantum._register import HilbertRegisterLayout
from ...solver._local_hamiltonian import (
    LocalHamiltonian,
    LocalHamiltonianTerm,
    materialize_local_hamiltonian,
)
from ._conventions import (
    HBAR_J_S,
    MagneticResonanceConvention,
    MagneticResonanceResourcePolicy,
    MU0_OVER_4PI_N_A2,
    TWO_PI,
)
from ._orientation import SingleCrystalOrientation


ParticleKind: TypeAlias = Literal["nucleus", "electron", "positive-muon"]


def _identifier(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


def _finite_vector(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=float)
    if host.shape != (3,) or np.any(~np.isfinite(host)):
        raise ValueError(f"{name} must be finite with shape (3,).")
    return jnp.asarray(host)


def _finite_tensor(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=float)
    if host.shape != (3, 3) or np.any(~np.isfinite(host)):
        raise ValueError(f"{name} must be finite with shape (3, 3).")
    return jnp.asarray(host)


def _validate_spin(spin: float, /) -> tuple[float, int]:
    value = float(spin)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("spin must be a positive integer or half-integer.")
    twice = round(2.0 * value)
    if abs(2.0 * value - twice) > 1.0e-12:
        raise ValueError("spin must be a positive integer or half-integer.")
    return value, int(twice + 1)


class ResonanceIsotope(StrictModule):
    """Signed gyromagnetic-ratio and spin record for one particle species."""

    isotope_id: str = eqx.field(static=True)
    particle_kind: ParticleKind = eqx.field(static=True)
    spin: float = eqx.field(static=True)
    gyromagnetic_ratio_rad_s_t: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        isotope_id: str,
        particle_kind: ParticleKind,
        spin: float,
        gyromagnetic_ratio_rad_s_t: float,
        /,
    ):
        identifier = _identifier(isotope_id, "isotope_id")
        if particle_kind not in ("nucleus", "electron", "positive-muon"):
            raise ValueError("particle_kind must be nucleus, electron, or positive-muon.")
        spin_, dimension = _validate_spin(spin)
        gamma = float(gyromagnetic_ratio_rad_s_t)
        if not math.isfinite(gamma) or gamma == 0.0:
            raise ValueError("gyromagnetic_ratio_rad_s_t must be finite and nonzero.")
        self.isotope_id = identifier
        self.particle_kind = particle_kind
        self.spin = spin_
        self.gyromagnetic_ratio_rad_s_t = gamma
        self.dimension = dimension


HYDROGEN_1 = ResonanceIsotope("1H", "nucleus", 0.5, 267.522_187_44e6)
CARBON_13 = ResonanceIsotope("13C", "nucleus", 0.5, 67.282_840e6)
NITROGEN_14 = ResonanceIsotope("14N", "nucleus", 1.0, 19.331e6)
ELECTRON = ResonanceIsotope("electron", "electron", 0.5, -1.760_859_630_23e11)
POSITIVE_MUON = ResonanceIsotope("mu+", "positive-muon", 0.5, 851.615e6)


class SpinOperators(StrictModule):
    identity: Array
    x: Array
    y: Array
    z: Array


def spin_operators(spin: float, /) -> SpinOperators:
    """Return dimensionless Cartesian spin matrices in ascending-m order."""

    value, dimension = _validate_spin(spin)
    magnetic = np.arange(-value, value + 1.0, 1.0, dtype=float)
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for column, m_value in enumerate(magnetic[:-1]):
        raising[column + 1, column] = math.sqrt(
            value * (value + 1.0) - m_value * (m_value + 1.0)
        )
    lowering = raising.T
    return SpinOperators(
        jnp.eye(dimension, dtype=jnp.complex128),
        jnp.asarray(0.5 * (raising + lowering)),
        jnp.asarray((raising - lowering) / (2.0j)),
        jnp.asarray(np.diag(magnetic).astype(np.complex128)),
    )


class SpinSite(StrictModule):
    """One ordered spin site with molecular-frame magnetic tensors."""

    isotope: ResonanceIsotope
    position_m: Array
    zeeman_tensor: Array
    chemical_shift_ppm: Array
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_id: str,
        isotope: ResonanceIsotope,
        /,
        *,
        position_m: ArrayLike = (0.0, 0.0, 0.0),
        zeeman_tensor: ArrayLike = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        chemical_shift_ppm: ArrayLike = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    ):
        if not isinstance(isotope, ResonanceIsotope):
            raise TypeError("isotope must be a ResonanceIsotope.")
        self.isotope = isotope
        self.position_m = _finite_vector(position_m, "position_m")
        self.zeeman_tensor = _finite_tensor(zeeman_tensor, "zeeman_tensor")
        self.chemical_shift_ppm = _finite_tensor(chemical_shift_ppm, "chemical_shift_ppm")
        self.site_id = _identifier(site_id, "site_id")


class ScalarJCoupling(StrictModule):
    site_a: str = eqx.field(static=True)
    site_b: str = eqx.field(static=True)
    coupling_hz: float = eqx.field(static=True)

    def __init__(self, site_a: str, site_b: str, coupling_hz: float, /):
        a = _identifier(site_a, "site_a")
        b = _identifier(site_b, "site_b")
        value = float(coupling_hz)
        if a == b:
            raise ValueError("A J coupling requires two distinct sites.")
        if not math.isfinite(value):
            raise ValueError("coupling_hz must be finite.")
        self.site_a, self.site_b, self.coupling_hz = a, b, value


class DipolarCoupling(StrictModule):
    site_a: str = eqx.field(static=True)
    site_b: str = eqx.field(static=True)

    def __init__(self, site_a: str, site_b: str, /):
        a = _identifier(site_a, "site_a")
        b = _identifier(site_b, "site_b")
        if a == b:
            raise ValueError("A dipolar coupling requires two distinct sites.")
        self.site_a, self.site_b = a, b


class HyperfineCoupling(StrictModule):
    """Full molecular-frame Cartesian coefficient tensor in cycles/s."""

    tensor_hz: Array
    site_a: str = eqx.field(static=True)
    site_b: str = eqx.field(static=True)

    def __init__(
        self,
        site_a: str,
        site_b: str,
        tensor_hz: ArrayLike,
        /,
    ):
        a = _identifier(site_a, "site_a")
        b = _identifier(site_b, "site_b")
        if a == b:
            raise ValueError("A hyperfine coupling requires two distinct sites.")
        self.tensor_hz = _finite_tensor(tensor_hz, "tensor_hz")
        self.site_a, self.site_b = a, b


class QuadrupolarInteraction(StrictModule):
    """Symmetric traceless molecular-frame Cartesian coefficient in cycles/s."""

    tensor_hz: Array
    site_id: str = eqx.field(static=True)

    def __init__(self, site_id: str, tensor_hz: ArrayLike, /):
        tensor = np.asarray(tensor_hz, dtype=float)
        if tensor.shape != (3, 3) or np.any(~np.isfinite(tensor)):
            raise ValueError("tensor_hz must be finite with shape (3, 3).")
        if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("Quadrupolar tensor_hz must be symmetric.")
        if not math.isclose(float(np.trace(tensor)), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Quadrupolar tensor_hz must be traceless.")
        self.tensor_hz = jnp.asarray(tensor)
        self.site_id = _identifier(site_id, "site_id")


SpinInteraction: TypeAlias = (
    ScalarJCoupling | DipolarCoupling | HyperfineCoupling | QuadrupolarInteraction
)


class MagneticResonanceSpinSystem(StrictModule):
    """Ordered finite spins, static field, interactions, and single orientation."""

    sites: tuple[SpinSite, ...]
    interactions: tuple[SpinInteraction, ...]
    static_field_t: Array
    orientation: SingleCrystalOrientation
    resource_policy: MagneticResonanceResourcePolicy
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        sites: Sequence[SpinSite],
        static_field_t: ArrayLike,
        /,
        *,
        interactions: Sequence[SpinInteraction] = (),
        orientation: SingleCrystalOrientation | None = None,
        resource_policy: MagneticResonanceResourcePolicy | None = None,
        system_id: str = "finite-spin-system",
    ):
        selected = tuple(sites)
        if not selected or not all(isinstance(site, SpinSite) for site in selected):
            raise ValueError("sites must contain at least one SpinSite.")
        ids = tuple(site.site_id for site in selected)
        if len(set(ids)) != len(ids):
            raise ValueError("Spin-site IDs must be unique.")
        chosen_interactions = tuple(interactions)
        if not all(
            isinstance(
                interaction,
                (
                    ScalarJCoupling,
                    DipolarCoupling,
                    HyperfineCoupling,
                    QuadrupolarInteraction,
                ),
            )
            for interaction in chosen_interactions
        ):
            raise TypeError("interactions contain an unsupported interaction record.")
        for interaction in chosen_interactions:
            referenced = (
                (interaction.site_id,)
                if isinstance(interaction, QuadrupolarInteraction)
                else (interaction.site_a, interaction.site_b)
            )
            if any(site_id not in ids for site_id in referenced):
                raise ValueError("Every interaction must reference known spin-site IDs.")
        orientation_ = SingleCrystalOrientation() if orientation is None else orientation
        policy = (
            MagneticResonanceResourcePolicy()
            if resource_policy is None
            else resource_policy
        )
        if not isinstance(orientation_, SingleCrystalOrientation):
            raise TypeError("orientation must be a SingleCrystalOrientation.")
        if not isinstance(policy, MagneticResonanceResourcePolicy):
            raise TypeError("resource_policy must be a MagneticResonanceResourcePolicy.")
        self.sites = selected
        self.interactions = chosen_interactions
        self.static_field_t = _finite_vector(static_field_t, "static_field_t")
        self.orientation = orientation_
        self.resource_policy = policy
        self.system_id = _identifier(system_id, "system_id")


class SpinAssemblyEvidence(StrictModule):
    hilbert_dimension: int = eqx.field(static=True)
    density_elements: int = eqx.field(static=True)
    hermiticity_residual: Array
    orientation_orthogonality_residual: Array
    local_terms_valid: Array
    finite: Array
    valid: Array


class PreparedMagneticResonanceSystem(StrictModule):
    convention: MagneticResonanceConvention
    system: MagneticResonanceSpinSystem
    layout: HilbertRegisterLayout
    hamiltonian: LocalHamiltonian
    drive_terms: tuple[LocalHamiltonianTerm, ...]
    dense_hamiltonian_rad_s: Array
    evidence: SpinAssemblyEvidence


def _site_index(system: MagneticResonanceSpinSystem, site_id: str, /) -> int:
    ids = tuple(site.site_id for site in system.sites)
    if site_id not in ids:
        raise ValueError(f"Interaction references unknown spin site {site_id!r}.")
    return ids.index(site_id)


def _pair_generator(left: SpinOperators, right: SpinOperators, tensor: Array, /) -> Array:
    components_left = (left.x, left.y, left.z)
    components_right = (right.x, right.y, right.z)
    result = jnp.zeros(
        (left.x.shape[0] * right.x.shape[0],) * 2,
        dtype=jnp.complex128,
    )
    for row in range(3):
        for column in range(3):
            result = result + tensor[row, column] * jnp.kron(
                components_left[row], components_right[column]
            )
    return result


def _single_generator(operators: SpinOperators, coefficients: Array, /) -> Array:
    return (
        coefficients[0] * operators.x
        + coefficients[1] * operators.y
        + coefficients[2] * operators.z
    )


def prepare_spin_system(
    system: MagneticResonanceSpinSystem,
    /,
) -> PreparedMagneticResonanceSystem:
    """Assemble exact static and lab-field drive terms under D and D² guards."""

    if not isinstance(system, MagneticResonanceSpinSystem):
        raise TypeError("system must be a MagneticResonanceSpinSystem.")
    layout = HilbertRegisterLayout(
        tuple(site.site_id for site in system.sites),
        tuple(site.isotope.dimension for site in system.sites),
    )
    system.resource_policy.guard(layout.dimension)
    operators = tuple(spin_operators(site.isotope.spin) for site in system.sites)
    terms: list[LocalHamiltonianTerm] = []
    drives: list[LocalHamiltonianTerm] = []
    field = system.static_field_t
    orientation = system.orientation

    for site, spin in zip(system.sites, operators, strict=True):
        gamma = site.isotope.gyromagnetic_ratio_rad_s_t
        zeeman = orientation.tensor_to_lab(site.zeeman_tensor)
        zeeman_coefficients = -gamma * (field @ zeeman)
        terms.append(
            LocalHamiltonianTerm(
                _single_generator(spin, zeeman_coefficients),
                (site.site_id,),
                term_id=f"{system.system_id}:zeeman:{site.site_id}",
            )
        )
        shift = orientation.tensor_to_lab(site.chemical_shift_ppm) * 1.0e-6
        shift_coefficients = -gamma * (field @ shift)
        if bool(np.any(np.asarray(site.chemical_shift_ppm) != 0.0)):
            terms.append(
                LocalHamiltonianTerm(
                    _single_generator(spin, shift_coefficients),
                    (site.site_id,),
                    term_id=f"{system.system_id}:chemical-shift:{site.site_id}",
                )
            )
        for axis in range(3):
            coefficients = -gamma * zeeman[axis, :]
            drives.append(
                LocalHamiltonianTerm(
                    _single_generator(spin, coefficients),
                    (site.site_id,),
                    term_id=f"{system.system_id}:lab-field:{site.site_id}:{axis}",
                )
            )

    for interaction_index, interaction in enumerate(system.interactions):
        if isinstance(interaction, QuadrupolarInteraction):
            index = _site_index(system, interaction.site_id)
            site = system.sites[index]
            if site.isotope.spin <= 0.5:
                raise ValueError(
                    f"Quadrupolar interaction requires I >= 1 for site {site.site_id!r}."
                )
            tensor = TWO_PI * orientation.tensor_to_lab(interaction.tensor_hz)
            components = (operators[index].x, operators[index].y, operators[index].z)
            generator = jnp.zeros_like(operators[index].x)
            for row in range(3):
                for column in range(3):
                    generator = generator + 0.5 * tensor[row, column] * (
                        components[row] @ components[column]
                        + components[column] @ components[row]
                    )
            terms.append(
                LocalHamiltonianTerm(
                    generator,
                    (site.site_id,),
                    term_id=f"{system.system_id}:quadrupole:{interaction_index}",
                )
            )
            continue

        left_index = _site_index(system, interaction.site_a)
        right_index = _site_index(system, interaction.site_b)
        left = system.sites[left_index]
        right = system.sites[right_index]
        if isinstance(interaction, ScalarJCoupling):
            tensor = TWO_PI * interaction.coupling_hz * jnp.eye(3)
            kind = "j"
        elif isinstance(interaction, HyperfineCoupling):
            tensor = TWO_PI * orientation.tensor_to_lab(interaction.tensor_hz)
            kind = "hyperfine"
        else:
            displacement = orientation.vector_to_lab(right.position_m - left.position_m)
            distance = float(np.linalg.norm(np.asarray(displacement)))
            if not math.isfinite(distance) or distance <= 0.0:
                raise ValueError(
                    "Dipolar-coupled sites must have distinct finite positions."
                )
            direction = displacement / distance
            prefactor = (
                MU0_OVER_4PI_N_A2
                * HBAR_J_S
                * left.isotope.gyromagnetic_ratio_rad_s_t
                * right.isotope.gyromagnetic_ratio_rad_s_t
                / distance**3
            )
            tensor = prefactor * (jnp.eye(3) - 3.0 * jnp.outer(direction, direction))
            kind = "dipolar"
        generator = _pair_generator(operators[left_index], operators[right_index], tensor)
        terms.append(
            LocalHamiltonianTerm(
                generator,
                (left.site_id, right.site_id),
                term_id=f"{system.system_id}:{kind}:{interaction_index}",
            )
        )

    hamiltonian = LocalHamiltonian(
        layout,
        tuple(terms),
        hamiltonian_id=f"{system.system_id}:static",
    )
    dense = materialize_local_hamiltonian(
        hamiltonian,
        policy=MaterializationPolicy(
            max_entries=system.resource_policy.maximum_density_elements,
            max_bytes=16 * system.resource_policy.maximum_density_elements,
        ),
    )
    residual = jnp.max(jnp.abs(dense - jnp.conj(dense.T)))
    finite = jnp.all(jnp.isfinite(dense))
    local_valid = hamiltonian.valid & jnp.all(
        jnp.stack(tuple(term.valid for term in drives))
    )
    evidence = SpinAssemblyEvidence(
        layout.dimension,
        layout.dimension * layout.dimension,
        residual,
        orientation.orthogonality_residual,
        local_valid,
        finite,
        local_valid & finite & (residual <= 1.0e-9),
    )
    return PreparedMagneticResonanceSystem(
        MagneticResonanceConvention(),
        system,
        layout,
        hamiltonian,
        tuple(drives),
        dense,
        evidence,
    )


__all__ = [
    "CARBON_13",
    "ELECTRON",
    "HYDROGEN_1",
    "NITROGEN_14",
    "POSITIVE_MUON",
    "DipolarCoupling",
    "HyperfineCoupling",
    "MagneticResonanceSpinSystem",
    "ParticleKind",
    "PreparedMagneticResonanceSystem",
    "QuadrupolarInteraction",
    "ResonanceIsotope",
    "ScalarJCoupling",
    "SpinAssemblyEvidence",
    "SpinInteraction",
    "SpinOperators",
    "SpinSite",
    "prepare_spin_system",
    "spin_operators",
]

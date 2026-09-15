#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared molecular mean-field convergence, occupation, guess, and state contracts."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._model import ElectronicReferenceKind


class SCFAccelerationKind(StrEnum):
    DAMPING = "damping"
    DIIS = "diis"
    ENERGY_DIIS = "energy-diis"
    AUGMENTED_DIIS = "augmented-diis"
    CIAH = "ciah"


class ElectronicOccupationKind(StrEnum):
    INTEGER = "integer"
    FERMI_DIRAC = "fermi-dirac"
    EXPLICIT = "explicit"
    MAXIMUM_OVERLAP = "maximum-overlap"


class InitialGuessKind(StrEnum):
    CORE = "core"
    ZERO = "zero"
    EXPLICIT = "explicit"
    PROJECTED = "projected"
    SAD = "sad"
    SAP = "sap"
    HUCKEL = "huckel"


class SCFConvergencePlan(StrictModule, NonTrainableState):
    energy_tolerance: float = eqx.field(static=True)
    density_tolerance: float = eqx.field(static=True)
    commutator_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        energy_tolerance: float = 1.0e-10,
        density_tolerance: float = 1.0e-8,
        commutator_tolerance: float = 1.0e-8,
        maximum_iterations: int = 128,
    ):
        values = tuple(
            float(value)
            for value in (
                energy_tolerance,
                density_tolerance,
                commutator_tolerance,
            )
        )
        iterations = int(maximum_iterations)
        if (
            any(not isfinite(value) or value <= 0.0 for value in values)
            or iterations <= 0
        ):
            raise ValueError(
                "SCF convergence tolerances and iteration limit must be positive."
            )
        self.energy_tolerance, self.density_tolerance, self.commutator_tolerance = values
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scf-convergence-plan",
                "energy_tolerance": values[0],
                "density_tolerance": values[1],
                "commutator_tolerance": values[2],
                "maximum_iterations": iterations,
            }
        )


class SCFAccelerationPlan(StrictModule, NonTrainableState):
    schedule: tuple[SCFAccelerationKind, ...] = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    diis_start: int = eqx.field(static=True)
    diis_space: int = eqx.field(static=True)
    level_shift: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: tuple[SCFAccelerationKind, ...] = (
            SCFAccelerationKind.DAMPING,
            SCFAccelerationKind.DIIS,
        ),
        /,
        *,
        damping: float = 0.2,
        diis_start: int = 2,
        diis_space: int = 8,
        level_shift: float = 0.0,
    ):
        schedule_ = tuple(schedule)
        if not schedule_ or any(
            not isinstance(value, SCFAccelerationKind) for value in schedule_
        ):
            raise TypeError("SCF acceleration schedule must contain typed phases.")
        unsupported = tuple(
            value
            for value in schedule_
            if value
            not in (
                SCFAccelerationKind.DAMPING,
                SCFAccelerationKind.DIIS,
            )
        )
        if unsupported:
            names = ", ".join(value.value for value in unsupported)
            raise NotImplementedError(
                f"Native molecular SCF does not implement acceleration phases: {names}."
            )
        damping_ = float(damping)
        start = int(diis_start)
        space = int(diis_space)
        shift = float(level_shift)
        if (
            not isfinite(damping_)
            or not 0.0 <= damping_ < 1.0
            or start < 0
            or space < 2
            or not isfinite(shift)
            or shift < 0.0
        ):
            raise ValueError("SCF acceleration parameters are invalid.")
        self.schedule = schedule_
        self.damping = damping_
        self.diis_start = start
        self.diis_space = space
        self.level_shift = shift
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scf-acceleration-plan",
                "schedule": [value.value for value in schedule_],
                "damping": damping_,
                "diis_start": start,
                "diis_space": space,
                "level_shift": shift,
            }
        )


class ElectronicOccupationPlan(StrictModule, NonTrainableState):
    kind: ElectronicOccupationKind = eqx.field(static=True)
    smearing_energy: float = eqx.field(static=True)
    explicit_occupations: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ElectronicOccupationKind = ElectronicOccupationKind.INTEGER,
        /,
        *,
        smearing_energy: float = 0.0,
        explicit_occupations: ArrayLike | None = None,
    ):
        if not isinstance(kind, ElectronicOccupationKind):
            raise TypeError("kind must be ElectronicOccupationKind.")
        smearing = float(smearing_energy)
        explicit = (
            None
            if explicit_occupations is None
            else jnp.asarray(explicit_occupations, dtype=float).reshape((-1,))
        )
        if not isfinite(smearing) or smearing < 0.0:
            raise ValueError("smearing_energy must be finite and non-negative.")
        if kind is ElectronicOccupationKind.FERMI_DIRAC and smearing <= 0.0:
            raise ValueError("Fermi-Dirac occupations require positive smearing_energy.")
        if kind is ElectronicOccupationKind.EXPLICIT and explicit is None:
            raise ValueError("Explicit occupations require explicit_occupations.")
        if kind is not ElectronicOccupationKind.EXPLICIT and explicit is not None:
            raise ValueError(
                "explicit_occupations belong only to explicit occupation plans."
            )
        if explicit is not None and (
            np.any(~np.isfinite(np.asarray(explicit)))
            or np.any(np.asarray(explicit) < 0.0)
        ):
            raise ValueError("Explicit occupations must be finite and non-negative.")
        self.kind = kind
        self.smearing_energy = smearing
        self.explicit_occupations = explicit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electronic-occupation-plan",
                "occupation_kind": kind.value,
                "smearing_energy": smearing,
                "explicit": None
                if explicit is None
                else array_tree_fingerprint(np.asarray(explicit)),
            }
        )


class InitialGuessPlan(StrictModule, NonTrainableState):
    kind: InitialGuessKind = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: InitialGuessKind = InitialGuessKind.CORE,
        /,
        *,
        source_id: str | None = None,
    ):
        if not isinstance(kind, InitialGuessKind):
            raise TypeError("kind must be InitialGuessKind.")
        source = None if source_id is None else str(source_id).strip()
        if kind in (InitialGuessKind.EXPLICIT, InitialGuessKind.PROJECTED) and not source:
            raise ValueError("Explicit/projected guesses require a source_id.")
        if source_id is not None and not source:
            raise ValueError("source_id must be non-empty when provided.")
        self.kind = kind
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {"kind": "initial-guess-plan", "guess_kind": kind.value, "source": source}
        )


class SCFStabilityPlan(StrictModule, NonTrainableState):
    require_internal: bool = eqx.field(static=True)
    require_external: bool = eqx.field(static=True)
    eigenvalue_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        require_internal: bool = False,
        require_external: bool = False,
        eigenvalue_tolerance: float = 1.0e-7,
    ):
        tolerance = float(eigenvalue_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Stability eigenvalue tolerance must be non-negative.")
        self.require_internal = bool(require_internal)
        self.require_external = bool(require_external)
        self.eigenvalue_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scf-stability-plan",
                "require_internal": self.require_internal,
                "require_external": self.require_external,
                "eigenvalue_tolerance": tolerance,
            }
        )


class SCFStabilityResult(StrictModule):
    internal_eigenvalues: Array
    external_eigenvalues: Array
    internal_stable: Array
    external_stable: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class SCFConvergenceEvidence(StrictModule):
    energy_residual: Array
    density_residual: Array
    commutator_residual: Array
    electron_count_residual: Array
    spin_residual: Array
    iterations: Array
    converged: Array
    stable: Array
    finite: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged & self.stable & self.finite


class RestrictedMeanFieldState(StrictModule, NonTrainableState):
    density: Array
    coefficients: Array
    orbital_energies: Array
    occupations: Array
    fock: Array
    overlap: Array
    electronic_energy: Array
    total_energy: Array
    entropy: Array
    free_energy: Array
    evidence: SCFConvergenceEvidence
    reference: ElectronicReferenceKind = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        coefficients: ArrayLike,
        orbital_energies: ArrayLike,
        occupations: ArrayLike,
        fock: ArrayLike,
        overlap: ArrayLike,
        electronic_energy: ArrayLike,
        total_energy: ArrayLike,
        entropy: ArrayLike,
        free_energy: ArrayLike,
        evidence: SCFConvergenceEvidence,
        /,
        *,
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
    ):
        density_ = jnp.asarray(density)
        coefficients_ = jnp.asarray(coefficients, dtype=density_.dtype)
        energies = jnp.asarray(orbital_energies, dtype=density_.real.dtype)
        occupations_ = jnp.asarray(occupations, dtype=energies.dtype)
        fock_ = jnp.asarray(fock, dtype=density_.dtype)
        overlap_ = jnp.asarray(overlap, dtype=density_.dtype)
        count = density_.shape[0] if density_.ndim == 2 else -1
        orbital_count = (
            coefficients_.shape[1]
            if coefficients_.ndim == 2 and coefficients_.shape[0] == count
            else -1
        )
        if (
            density_.shape != (count, count)
            or orbital_count <= 0
            or energies.shape != (orbital_count,)
            or occupations_.shape != (orbital_count,)
            or fock_.shape != density_.shape
            or overlap_.shape != density_.shape
        ):
            raise ValueError(
                "Restricted mean-field arrays do not share one AO/orbital layout."
            )
        if not isinstance(evidence, SCFConvergenceEvidence):
            raise TypeError("evidence must be SCFConvergenceEvidence.")
        if reference not in (
            ElectronicReferenceKind.RESTRICTED,
            ElectronicReferenceKind.RESTRICTED_OPEN_SHELL,
        ):
            raise ValueError(
                "Restricted state requires restricted or restricted-open-shell reference."
            )
        self.density = density_
        self.coefficients = coefficients_
        self.orbital_energies = energies
        self.occupations = occupations_
        self.fock = fock_
        self.overlap = overlap_
        self.electronic_energy = jnp.asarray(
            electronic_energy, dtype=energies.dtype
        ).reshape(())
        self.total_energy = jnp.asarray(total_energy, dtype=energies.dtype).reshape(())
        self.entropy = jnp.asarray(entropy, dtype=energies.dtype).reshape(())
        self.free_energy = jnp.asarray(free_energy, dtype=energies.dtype).reshape(())
        self.evidence = evidence
        self.reference = reference
        self.state_id = canonical_fingerprint(
            {
                "kind": "restricted-mean-field-state",
                "reference": reference.value,
                "evidence": evidence.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "density": np.asarray(density_),
                        "coefficients": np.asarray(coefficients_),
                        "orbital_energies": np.asarray(energies),
                        "occupations": np.asarray(occupations_),
                        "fock": np.asarray(fock_),
                        "overlap": np.asarray(overlap_),
                        "electronic_energy": np.asarray(self.electronic_energy),
                        "total_energy": np.asarray(self.total_energy),
                        "entropy": np.asarray(self.entropy),
                        "free_energy": np.asarray(self.free_energy),
                    }
                ),
            }
        )


class UnrestrictedMeanFieldState(StrictModule, NonTrainableState):
    alpha_density: Array
    beta_density: Array
    alpha_coefficients: Array
    beta_coefficients: Array
    alpha_orbital_energies: Array
    beta_orbital_energies: Array
    alpha_occupations: Array
    beta_occupations: Array
    alpha_fock: Array
    beta_fock: Array
    overlap: Array
    electronic_energy: Array
    total_energy: Array
    entropy: Array
    free_energy: Array
    evidence: SCFConvergenceEvidence
    reference: ElectronicReferenceKind = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha_density: ArrayLike,
        beta_density: ArrayLike,
        alpha_coefficients: ArrayLike,
        beta_coefficients: ArrayLike,
        alpha_orbital_energies: ArrayLike,
        beta_orbital_energies: ArrayLike,
        alpha_occupations: ArrayLike,
        beta_occupations: ArrayLike,
        alpha_fock: ArrayLike,
        beta_fock: ArrayLike,
        overlap: ArrayLike,
        electronic_energy: ArrayLike,
        total_energy: ArrayLike,
        entropy: ArrayLike,
        free_energy: ArrayLike,
        evidence: SCFConvergenceEvidence,
        /,
        *,
        reference: ElectronicReferenceKind = ElectronicReferenceKind.UNRESTRICTED,
    ):
        alpha = jnp.asarray(alpha_density)
        beta = jnp.asarray(beta_density, dtype=alpha.dtype)
        count = alpha.shape[0] if alpha.ndim == 2 else -1
        alpha_coefficients_ = jnp.asarray(alpha_coefficients, dtype=alpha.dtype)
        beta_coefficients_ = jnp.asarray(beta_coefficients, dtype=alpha.dtype)
        alpha_fock_ = jnp.asarray(alpha_fock, dtype=alpha.dtype)
        beta_fock_ = jnp.asarray(beta_fock, dtype=alpha.dtype)
        overlap_ = jnp.asarray(overlap, dtype=alpha.dtype)
        energies = tuple(
            jnp.asarray(value, dtype=alpha.real.dtype)
            for value in (alpha_orbital_energies, beta_orbital_energies)
        )
        occupations = tuple(
            jnp.asarray(value, dtype=alpha.real.dtype)
            for value in (alpha_occupations, beta_occupations)
        )
        alpha_orbitals = (
            alpha_coefficients_.shape[1]
            if alpha_coefficients_.ndim == 2 and alpha_coefficients_.shape[0] == count
            else -1
        )
        beta_orbitals = (
            beta_coefficients_.shape[1]
            if beta_coefficients_.ndim == 2 and beta_coefficients_.shape[0] == count
            else -1
        )
        if (
            alpha.shape != (count, count)
            or beta.shape != alpha.shape
            or alpha_fock_.shape != alpha.shape
            or beta_fock_.shape != alpha.shape
            or overlap_.shape != alpha.shape
            or energies[0].shape != (alpha_orbitals,)
            or energies[1].shape != (beta_orbitals,)
            or occupations[0].shape != (alpha_orbitals,)
            or occupations[1].shape != (beta_orbitals,)
        ):
            raise ValueError("Unrestricted mean-field arrays do not share one layout.")
        if not isinstance(evidence, SCFConvergenceEvidence):
            raise TypeError("evidence must be SCFConvergenceEvidence.")
        if reference not in (
            ElectronicReferenceKind.UNRESTRICTED,
            ElectronicReferenceKind.RESTRICTED_OPEN_SHELL,
        ):
            raise ValueError("Unrestricted state has an incompatible reference.")
        self.alpha_density, self.beta_density = alpha, beta
        self.alpha_coefficients = alpha_coefficients_
        self.beta_coefficients = beta_coefficients_
        self.alpha_fock = alpha_fock_
        self.beta_fock = beta_fock_
        self.overlap = overlap_
        self.alpha_orbital_energies, self.beta_orbital_energies = energies
        self.alpha_occupations, self.beta_occupations = occupations
        self.electronic_energy = jnp.asarray(
            electronic_energy, dtype=alpha.real.dtype
        ).reshape(())
        self.total_energy = jnp.asarray(total_energy, dtype=alpha.real.dtype).reshape(())
        self.entropy = jnp.asarray(entropy, dtype=alpha.real.dtype).reshape(())
        self.free_energy = jnp.asarray(free_energy, dtype=alpha.real.dtype).reshape(())
        self.evidence = evidence
        self.reference = reference
        self.state_id = canonical_fingerprint(
            {
                "kind": "unrestricted-mean-field-state",
                "reference": reference.value,
                "evidence": evidence.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "alpha_density": np.asarray(alpha),
                        "beta_density": np.asarray(beta),
                        "alpha_coefficients": np.asarray(self.alpha_coefficients),
                        "beta_coefficients": np.asarray(self.beta_coefficients),
                        "alpha_energies": np.asarray(self.alpha_orbital_energies),
                        "beta_energies": np.asarray(self.beta_orbital_energies),
                        "alpha_occupations": np.asarray(self.alpha_occupations),
                        "beta_occupations": np.asarray(self.beta_occupations),
                        "electronic_energy": np.asarray(self.electronic_energy),
                        "total_energy": np.asarray(self.total_energy),
                    }
                ),
            }
        )


class GeneralizedMeanFieldState(StrictModule, NonTrainableState):
    density: Array
    coefficients: Array
    orbital_energies: Array
    occupations: Array
    fock: Array
    overlap: Array
    electronic_energy: Array
    total_energy: Array
    evidence: SCFConvergenceEvidence
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        coefficients: ArrayLike,
        orbital_energies: ArrayLike,
        occupations: ArrayLike,
        fock: ArrayLike,
        overlap: ArrayLike,
        electronic_energy: ArrayLike,
        total_energy: ArrayLike,
        evidence: SCFConvergenceEvidence,
        /,
    ):
        density_ = jnp.asarray(density)
        count = density_.shape[0] if density_.ndim == 2 else -1
        coefficients_ = jnp.asarray(coefficients, dtype=density_.dtype)
        energies = jnp.asarray(orbital_energies, dtype=density_.real.dtype)
        occupations_ = jnp.asarray(occupations, dtype=energies.dtype)
        fock_ = jnp.asarray(fock, dtype=density_.dtype)
        overlap_ = jnp.asarray(overlap, dtype=density_.dtype)
        orbital_count = (
            coefficients_.shape[1]
            if coefficients_.ndim == 2 and coefficients_.shape[0] == count
            else -1
        )
        if (
            density_.shape != (count, count)
            or orbital_count <= 0
            or fock_.shape != density_.shape
            or overlap_.shape != density_.shape
            or energies.shape != (orbital_count,)
            or occupations_.shape != (orbital_count,)
        ):
            raise ValueError(
                "Generalized mean-field arrays do not share one spinor layout."
            )
        self.density = density_
        self.coefficients = coefficients_
        self.orbital_energies = energies
        self.occupations = occupations_
        self.fock = fock_
        self.overlap = overlap_
        self.electronic_energy = jnp.asarray(
            electronic_energy, dtype=energies.dtype
        ).reshape(())
        self.total_energy = jnp.asarray(total_energy, dtype=energies.dtype).reshape(())
        self.evidence = evidence
        self.state_id = canonical_fingerprint(
            {
                "kind": "generalized-mean-field-state",
                "evidence": evidence.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "density": np.asarray(density_),
                        "coefficients": np.asarray(coefficients_),
                        "orbital_energies": np.asarray(energies),
                        "occupations": np.asarray(occupations_),
                        "electronic_energy": np.asarray(self.electronic_energy),
                        "total_energy": np.asarray(self.total_energy),
                    }
                ),
            }
        )


MeanFieldState = (
    RestrictedMeanFieldState | UnrestrictedMeanFieldState | GeneralizedMeanFieldState
)


__all__ = [
    "ElectronicOccupationKind",
    "ElectronicOccupationPlan",
    "GeneralizedMeanFieldState",
    "InitialGuessKind",
    "InitialGuessPlan",
    "SCFStabilityPlan",
    "MeanFieldState",
    "RestrictedMeanFieldState",
    "SCFAccelerationKind",
    "SCFAccelerationPlan",
    "SCFConvergenceEvidence",
    "SCFConvergencePlan",
    "SCFStabilityResult",
    "UnrestrictedMeanFieldState",
]

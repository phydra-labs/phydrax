#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Convention-complete primitives for finite-dimensional electronic transport.

This module owns only algebra shared by distinct transport methods.  It does
not introduce a device model, an energy integration, a collision model, or a
second NEGF hierarchy.  Semiconductor scalar-chain Landauer, coherent AC,
finite-lead transient, and optical-phonon SCBA remain separate application
profiles.  Periodic Kubo and Boltzmann response remain chemistry-owned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    HermitianSpectrum,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


_ELECTRON_CHARGE_SI = -1.602176634e-19


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _square_matrix(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be one square matrix.")
    return result


def _positive_tolerance(value: float, /) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    return tolerance


class ElectronicTransportConvention(StrictModule, NonTrainableState):
    """One fixed electron/Fourier/current convention.

    Bloch matrices use ``exp(+i k·R)`` and retarded functions use the
    ``exp(-i omega t)`` time transform.  Contact particle, charge, energy and
    heat currents are positive *into* the finite region.  The electron charge
    is signed, so a positive electron-particle current has negative
    conventional charge current.  Heat current at contact ``p`` is
    ``(E-mu[p])`` times its particle-current kernel.
    """

    electron_charge_coulomb: float = eqx.field(static=True)
    bloch_phase_sign: int = eqx.field(static=True)
    retarded_time_sign: int = eqx.field(static=True)
    current_orientation: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self):
        self.electron_charge_coulomb = _ELECTRON_CHARGE_SI
        self.bloch_phase_sign = 1
        self.retarded_time_sign = -1
        self.current_orientation = "positive-into-finite-region"
        self.convention_id = canonical_fingerprint(
            {
                "kind": "electronic-transport-convention",
                "electron_charge_coulomb": _ELECTRON_CHARGE_SI,
                "bloch_phase_sign": 1,
                "retarded_time_sign": -1,
                "current_orientation": self.current_orientation,
            }
        )

    def charge_current(self, particle_current: ArrayLike, /) -> Array:
        """Convert an electron-particle current into conventional charge current."""

        return self.electron_charge_coulomb * jnp.asarray(particle_current)

    def heat_current(
        self,
        particle_current: ArrayLike,
        energy_joule: ArrayLike,
        chemical_potential_joule: ArrayLike,
        /,
    ) -> Array:
        """Convert a particle-current kernel to contact-referenced heat current."""

        return (
            jnp.asarray(energy_joule) - jnp.asarray(chemical_potential_joule)
        ) * jnp.asarray(particle_current)


ELECTRONIC_TRANSPORT_CONVENTION = ElectronicTransportConvention()


class RetardedEmbeddingEvidence(StrictModule, NonTrainableState):
    """Causality and positive-semidefinite-broadening evidence."""

    surface_spectral_minimum_eigenvalue: Array
    broadening_minimum_eigenvalue: Array
    surface_hermiticity_residual: Array
    broadening_hermiticity_residual: Array
    finite: Array
    causal: Array
    tolerance: float = eqx.field(static=True)


class RetardedEmbedding(StrictModule, NonTrainableState):
    """One retarded Schur-complement self-energy at one complex energy."""

    energy_joule: Array
    self_energy: Array
    broadening: Array
    surface_green: Array
    coupling_h_device_lead: Array
    coupling_s_device_lead: Array
    evidence: RetardedEmbeddingEvidence


def retarded_embedding(
    energy_joule: ArrayLike,
    surface_green: ArrayLike,
    coupling_h_device_lead: ArrayLike,
    coupling_s_device_lead: ArrayLike | None = None,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> RetardedEmbedding:
    """Form ``Sigma=(z S_dl-H_dl) g (z S_ld-H_ld)`` at the same ``z``.

    In a nonorthogonal partition the reverse analytic block contains the same
    retarded energy ``z``.  Conjugating ``z`` would violate analyticity.  The
    function therefore constructs the reverse block explicitly rather than
    taking the adjoint of the completed forward block.
    """

    tolerance_ = _positive_tolerance(tolerance)
    energy = jnp.asarray(energy_joule)
    if energy.shape != ():
        raise ValueError("energy_joule must be one scalar complex energy.")
    surface = _square_matrix(surface_green, "surface_green")
    coupling_h = jnp.asarray(coupling_h_device_lead)
    if coupling_h.ndim != 2 or coupling_h.shape[1] != surface.shape[0]:
        raise ValueError(
            "coupling_h_device_lead must have shape (device, lead) matching surface_green."
        )
    coupling_s = (
        jnp.zeros_like(coupling_h)
        if coupling_s_device_lead is None
        else jnp.asarray(coupling_s_device_lead, dtype=coupling_h.dtype)
    )
    if coupling_s.shape != coupling_h.shape:
        raise ValueError("Hamiltonian and overlap contact couplings must align.")

    forward = energy * coupling_s - coupling_h
    reverse = energy * _adjoint(coupling_s) - _adjoint(coupling_h)
    self_energy = contract("ai,ij,jb->ab", forward, surface, reverse, backend="jax")
    broadening = 1j * (self_energy - _adjoint(self_energy))
    surface_spectral = 1j * (surface - _adjoint(surface))
    surface_spectrum = HermitianSpectrum(surface_spectral, tolerance=tolerance_)
    broadening_spectrum = HermitianSpectrum(broadening, tolerance=tolerance_)
    surface_scale = jnp.maximum(
        jnp.max(jnp.abs(surface_spectrum.eigenvalues), initial=0.0),
        jnp.finfo(surface_spectrum.eigenvalues.dtype).tiny,
    )
    broadening_scale = jnp.maximum(
        jnp.max(jnp.abs(broadening_spectrum.eigenvalues), initial=0.0),
        jnp.finfo(broadening_spectrum.eigenvalues.dtype).tiny,
    )
    finite = (
        jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(surface))
        & jnp.all(jnp.isfinite(coupling_h))
        & jnp.all(jnp.isfinite(coupling_s))
        & jnp.all(jnp.isfinite(self_energy))
    )
    causal = (
        (jnp.imag(energy) >= 0.0)
        & surface_spectrum.valid
        & broadening_spectrum.valid
        & (surface_spectrum.minimum_eigenvalue >= -tolerance_ * surface_scale)
        & (broadening_spectrum.minimum_eigenvalue >= -tolerance_ * broadening_scale)
    )
    evidence = RetardedEmbeddingEvidence(
        surface_spectrum.minimum_eigenvalue,
        broadening_spectrum.minimum_eigenvalue,
        surface_spectrum.hermiticity_residual,
        broadening_spectrum.hermiticity_residual,
        finite,
        finite & causal,
        tolerance_,
    )
    return RetardedEmbedding(
        energy,
        self_energy,
        broadening,
        surface,
        coupling_h,
        coupling_s,
        evidence,
    )


class RetardedOpenSystemPoint(StrictModule, NonTrainableState):
    """Dense finite-region retarded solution at one energy, not a NEGF solver."""

    energy_joule: Array
    green: Array
    spectral: Array
    contact_spectral: Array
    contact_broadenings: Array
    numerical_broadening: Array
    linear_residual: Array
    spectral_identity_residual: Array
    causal: Array
    successful: Array
    numerical_broadening_is_physical: bool = eqx.field(static=True)


def retarded_open_system_point(
    energy_joule: ArrayLike,
    hamiltonian: ArrayLike,
    overlap: ArrayLike,
    embeddings: Sequence[RetardedEmbedding],
    /,
    *,
    tolerance: float = 1.0e-10,
) -> RetardedOpenSystemPoint:
    """Solve one finite dense retarded resolvent using native linear algebra.

    ``Im(z)`` is tracked as a numerical broadening matrix and is never treated
    as a reservoir, collision rate, relaxation time, or finite ballistic DC
    conductivity.  With nonzero ``Im(z)``, contact currents need not conserve
    because this explicit numerical sink remains in the spectral identity.
    """

    tolerance_ = _positive_tolerance(tolerance)
    energy = jnp.asarray(energy_joule)
    if energy.shape != ():
        raise ValueError("energy_joule must be one scalar complex energy.")
    hamiltonian_ = _square_matrix(hamiltonian, "hamiltonian")
    overlap_ = _square_matrix(overlap, "overlap")
    if overlap_.shape != hamiltonian_.shape:
        raise ValueError("hamiltonian and overlap must have identical shapes.")
    contacts = tuple(embeddings)
    if not contacts or any(
        not isinstance(value, RetardedEmbedding) for value in contacts
    ):
        raise TypeError("embeddings must contain at least one RetardedEmbedding.")
    if any(value.self_energy.shape != hamiltonian_.shape for value in contacts):
        raise ValueError("Every embedding must act on the finite-region basis.")
    if any(not bool(jnp.all(value.energy_joule == energy)) for value in contacts):
        raise ValueError("Every embedding must be evaluated at this same energy.")

    self_energy = jnp.sum(
        jnp.stack(tuple(value.self_energy for value in contacts)), axis=0
    )
    contact_broadenings = jnp.stack(tuple(value.broadening for value in contacts))
    matrix = energy * overlap_ - hamiltonian_ - self_energy
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    solved = solve(
        LinearSystem(
            DenseLinearOperator(matrix),
            problem_id="retarded-open-system-point",
        ),
        identity,
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        ),
    )
    green = solved.value
    spectral = 1j * (green - _adjoint(green))
    contact_spectral = contract(
        "ij,pjk,lk->pil",
        green,
        contact_broadenings,
        jnp.conj(green),
        backend="jax",
    )
    numerical_broadening = 2.0 * jnp.imag(energy) * overlap_
    total_broadening = jnp.sum(contact_broadenings, axis=0) + numerical_broadening
    reconstructed = contract(
        "ij,jk,lk->il", green, total_broadening, jnp.conj(green), backend="jax"
    )
    linear_residual = jnp.max(jnp.abs(matrix @ green - identity), initial=0.0)
    spectral_scale = jnp.maximum(
        jnp.max(jnp.abs(spectral), initial=0.0),
        jnp.finfo(spectral.real.dtype).tiny,
    )
    spectral_residual = (
        jnp.max(jnp.abs(spectral - reconstructed), initial=0.0) / spectral_scale
    )
    spectral_spectrum = HermitianSpectrum(spectral, tolerance=tolerance_)
    spectral_eigenvalue_scale = jnp.maximum(
        jnp.max(jnp.abs(spectral_spectrum.eigenvalues), initial=0.0),
        jnp.finfo(spectral_spectrum.eigenvalues.dtype).tiny,
    )
    finite = (
        jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(green))
        & jnp.all(jnp.isfinite(spectral))
    )
    causal = (
        (jnp.imag(energy) >= 0.0)
        & jnp.all(jnp.stack(tuple(value.evidence.causal for value in contacts)))
        & spectral_spectrum.valid
        & (
            spectral_spectrum.minimum_eigenvalue
            >= -tolerance_ * spectral_eigenvalue_scale
        )
    )
    return RetardedOpenSystemPoint(
        energy,
        green,
        spectral,
        contact_spectral,
        contact_broadenings,
        numerical_broadening,
        linear_residual,
        spectral_residual,
        causal,
        finite
        & causal
        & solved.successful
        & (linear_residual <= tolerance_)
        & (spectral_residual <= 10.0 * tolerance_),
        False,
    )


def elastic_transmission(
    green: ArrayLike,
    left_broadening: ArrayLike,
    right_broadening: ArrayLike,
    /,
) -> Array:
    """Return the coherent elastic Caroli transmission at one energy."""

    green_ = _square_matrix(green, "green")
    left = _square_matrix(left_broadening, "left_broadening")
    right = _square_matrix(right_broadening, "right_broadening")
    if left.shape != green_.shape or right.shape != green_.shape:
        raise ValueError("Green function and contact broadenings must align.")
    product = contract(
        "ij,jk,kl,il->",
        left,
        green_,
        right,
        jnp.conj(green_),
        backend="jax",
    )
    return jnp.real(product)


class ElectronicCurrentKernels(StrictModule, NonTrainableState):
    """Contact kernels before the caller's energy measure and Planck prefactor."""

    electron_correlation: Array
    particle_current_into_region: Array
    charge_current_into_region: Array
    energy_current_into_region: Array
    heat_current_into_region: Array
    particle_continuity_residual: Array
    energy_continuity_residual: Array
    finite: Array
    convention_id: str = eqx.field(static=True)


def electronic_contact_current_kernels(
    point: RetardedOpenSystemPoint,
    occupations: ArrayLike,
    chemical_potentials_joule: ArrayLike,
    /,
    *,
    convention: ElectronicTransportConvention = ELECTRONIC_TRANSPORT_CONVENTION,
) -> ElectronicCurrentKernels:
    """Evaluate elastic contact particle/charge/energy/heat current kernels.

    The electron correlation is ``G^n=sum_p f_p G Gamma_p G†``.  Contact
    particle current is positive into the finite region:
    ``Tr[Gamma_p (f_p A-G^n)]``.  The charge sign follows the signed electron
    charge.  These are energy-resolved kernels; this function does not choose
    an integration measure or claim a Landauer/Meir--Wingreen solver.
    """

    if not isinstance(point, RetardedOpenSystemPoint):
        raise TypeError("point must be RetardedOpenSystemPoint.")
    if not isinstance(convention, ElectronicTransportConvention):
        raise TypeError("convention must be ElectronicTransportConvention.")
    occupation = jnp.asarray(occupations)
    chemical = jnp.asarray(chemical_potentials_joule)
    contacts = point.contact_broadenings.shape[0]
    if occupation.shape != (contacts,) or chemical.shape != (contacts,):
        raise ValueError(
            "Occupations and chemical potentials require one scalar per contact."
        )
    correlation = jnp.sum(occupation[:, None, None] * point.contact_spectral, axis=0)
    particle = jnp.real(
        contract(
            "pij,pji->p",
            point.contact_broadenings,
            occupation[:, None, None] * point.spectral[None, :, :]
            - correlation[None, :, :],
            backend="jax",
        )
    )
    energy = jnp.real(point.energy_joule) * particle
    heat = (jnp.real(point.energy_joule) - chemical) * particle
    charge = convention.charge_current(particle)
    finite = (
        point.successful
        & jnp.all(jnp.isfinite(occupation))
        & jnp.all((occupation >= 0.0) & (occupation <= 1.0))
        & jnp.all(jnp.isfinite(chemical))
        & jnp.all(jnp.isfinite(particle))
    )
    return ElectronicCurrentKernels(
        correlation,
        particle,
        charge,
        energy,
        heat,
        jnp.abs(jnp.sum(particle)),
        jnp.abs(jnp.sum(energy)),
        finite,
        convention.convention_id,
    )


class ElasticDisorderEnsemblePlan(StrictModule, NonTrainableState):
    """A fixed, explicit probability measure over named elastic realizations."""

    realization_ids: tuple[str, ...] = eqx.field(static=True)
    probabilities: Array
    ensemble_id: str = eqx.field(static=True)

    def __init__(self, realization_ids: Sequence[str], probabilities: ArrayLike, /):
        identifiers = tuple(str(value).strip() for value in realization_ids)
        weights = np.asarray(probabilities, dtype=np.float64)
        if (
            len(identifiers) < 2
            or any(not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or weights.shape != (len(identifiers),)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
        ):
            raise ValueError(
                "Elastic disorder requires at least two unique named realizations "
                "and positive probabilities summing to one."
            )
        self.realization_ids = identifiers
        self.probabilities = jnp.asarray(weights)
        self.ensemble_id = canonical_fingerprint(
            {
                "kind": "explicit-elastic-disorder-ensemble",
                "realization_ids": identifiers,
                "probabilities": weights.tolist(),
            }
        )


class ElasticDisorderEnsembleResult(StrictModule, NonTrainableState):
    """Raw realization values and fixed-measure population statistics."""

    realization_values: Array
    realization_successful: Array
    mean: Array
    flattened_covariance: Array
    effective_sample_size: Array
    successful: Array
    ensemble_id: str = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)


def evaluate_elastic_disorder_ensemble(
    plan: ElasticDisorderEnsemblePlan,
    realization_values: ArrayLike,
    realization_successful: ArrayLike,
    /,
) -> ElasticDisorderEnsembleResult:
    """Evaluate the declared ensemble without dropping or renormalizing failures."""

    if not isinstance(plan, ElasticDisorderEnsemblePlan):
        raise TypeError("plan must be ElasticDisorderEnsemblePlan.")
    values = jnp.asarray(realization_values)
    successful = jnp.asarray(realization_successful, dtype=jnp.bool_)
    count = len(plan.realization_ids)
    if values.ndim < 1 or values.shape[0] != count or successful.shape != (count,):
        raise ValueError(
            "Values and success flags require one leading entry per realization."
        )
    flat = values.reshape((count, -1))
    mean_flat = contract("r,rf->f", plan.probabilities, flat, backend="jax")
    centered = flat - mean_flat[None, :]
    covariance = contract(
        "r,rf,rg->fg", plan.probabilities, centered, jnp.conj(centered), backend="jax"
    )
    finite = jnp.all(jnp.isfinite(values))
    return ElasticDisorderEnsembleResult(
        values,
        successful,
        mean_flat.reshape(values.shape[1:]),
        covariance,
        1.0 / jnp.sum(plan.probabilities**2),
        finite & jnp.all(successful),
        plan.ensemble_id,
        tuple(values.shape[1:]),
    )


class FermionicKeldyshTransportState(StrictModule, NonTrainableState):
    """Fermionic two-time transport data, distinct from scalar-field Keldysh data."""

    lesser: Array
    greater: Array
    retarded: Array
    advanced: Array
    time_nodes: Array
    car_residual: Array
    retarded_causality_residual: Array
    advanced_causality_residual: Array
    advanced_adjoint_residual: Array
    particle_continuity_residual: Array
    finite: Array
    valid: Array
    mode_order_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    statistics: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)

    def __init__(
        self,
        lesser: ArrayLike,
        greater: ArrayLike,
        retarded: ArrayLike,
        advanced: ArrayLike,
        time_nodes: ArrayLike,
        /,
        *,
        mode_order_id: str,
        source_id: str,
        particle_continuity_residual: ArrayLike,
        tolerance: float = 1.0e-10,
    ):
        tolerance_ = _positive_tolerance(tolerance)
        lesser_ = jnp.asarray(lesser)
        greater_ = jnp.asarray(greater, dtype=lesser_.dtype)
        retarded_ = jnp.asarray(retarded, dtype=lesser_.dtype)
        advanced_ = jnp.asarray(advanced, dtype=lesser_.dtype)
        times = jnp.asarray(time_nodes)
        identifier = str(mode_order_id).strip()
        source = str(source_id).strip()
        if (
            lesser_.ndim != 4
            or lesser_.shape[0] != lesser_.shape[1]
            or lesser_.shape[2] != lesser_.shape[3]
            or greater_.shape != lesser_.shape
            or retarded_.shape != lesser_.shape
            or advanced_.shape != lesser_.shape
            or times.shape != (lesser_.shape[0],)
            or not identifier
            or not source
        ):
            raise ValueError(
                "Fermionic two-time functions require aligned (time,time,mode,mode) arrays and identities."
            )
        continuity = jnp.asarray(particle_continuity_residual)
        if continuity.shape != ():
            raise ValueError("particle_continuity_residual must be scalar.")
        difference = (greater_ - lesser_) - (retarded_ - advanced_)
        car = jnp.max(jnp.abs(difference), initial=0.0)
        earlier = times[:, None] < times[None, :]
        later = times[:, None] > times[None, :]
        retarded_residual = jnp.max(
            jnp.where(earlier[..., None, None], jnp.abs(retarded_), 0.0), initial=0.0
        )
        advanced_residual = jnp.max(
            jnp.where(later[..., None, None], jnp.abs(advanced_), 0.0), initial=0.0
        )
        advanced_adjoint = jnp.max(
            jnp.abs(advanced_ - jnp.swapaxes(_adjoint(retarded_), 0, 1)),
            initial=0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(times))
            & jnp.all(jnp.diff(times) > 0.0)
            & jnp.all(jnp.isfinite(lesser_))
            & jnp.all(jnp.isfinite(greater_))
            & jnp.all(jnp.isfinite(retarded_))
            & jnp.all(jnp.isfinite(advanced_))
            & jnp.isfinite(continuity)
            & (continuity >= 0.0)
        )
        self.lesser = lesser_
        self.greater = greater_
        self.retarded = retarded_
        self.advanced = advanced_
        self.time_nodes = times
        self.car_residual = car
        self.retarded_causality_residual = retarded_residual
        self.advanced_causality_residual = advanced_residual
        self.advanced_adjoint_residual = advanced_adjoint
        self.particle_continuity_residual = continuity
        self.finite = finite
        self.valid = (
            finite
            & (car <= tolerance_)
            & (retarded_residual <= tolerance_)
            & (advanced_residual <= tolerance_)
            & (advanced_adjoint <= tolerance_)
            & (continuity <= tolerance_)
        )
        self.mode_order_id = identifier
        self.source_id = source
        self.statistics = "fermionic"
        self.convention = "Ggreater-Glesser=GR-GA; GA(t,tprime)=GR(tprime,t)^dagger; retarded vanishes for t<tprime"


@runtime_checkable
class FermionicNonequilibriumTransportAdapter(Protocol):
    """Consumer-side contract for a fermionic two-time transport provider."""

    mode_order_id: str

    def evaluate_fermionic_transport(self, /) -> FermionicKeldyshTransportState:
        """Return fixed-mode-order data with CAR, causality, and conservation evidence."""


__all__ = [
    "ELECTRONIC_TRANSPORT_CONVENTION",
    "ElectronicCurrentKernels",
    "ElectronicTransportConvention",
    "ElasticDisorderEnsemblePlan",
    "ElasticDisorderEnsembleResult",
    "FermionicKeldyshTransportState",
    "FermionicNonequilibriumTransportAdapter",
    "RetardedEmbedding",
    "RetardedEmbeddingEvidence",
    "RetardedOpenSystemPoint",
    "elastic_transmission",
    "electronic_contact_current_kernels",
    "evaluate_elastic_disorder_ensemble",
    "retarded_embedding",
    "retarded_open_system_point",
]

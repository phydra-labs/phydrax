#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._fermionic_fock import FermionModeOrder


class NambuConvention(StrictModule):
    """Canonical class-D coordinate ``(c_k, c†_-k)`` in one fermion order."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    particle_labels: tuple[str, ...] = eqx.field(static=True)
    hole_labels: tuple[str, ...] = eqx.field(static=True)
    particle_hole_matrix: Array
    convention_id: str = eqx.field(static=True)

    def __init__(self, mode_order: FermionModeOrder, /):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        count = mode_order.mode_count
        zero = jnp.zeros((count, count), dtype=jnp.complex128)
        identity = jnp.eye(count, dtype=jnp.complex128)
        self.mode_order = mode_order
        self.particle_labels = mode_order.labels
        self.hole_labels = tuple(f"hole({label})" for label in mode_order.labels)
        self.particle_hole_matrix = jnp.concatenate(
            (
                jnp.concatenate((zero, identity), axis=1),
                jnp.concatenate((identity, zero), axis=1),
            ),
            axis=0,
        )
        self.convention_id = canonical_fingerprint(
            {"kind": "fermionic-nambu-convention", "mode_order": mode_order.order_id}
        )


class FermionicPairingPlan(StrictModule):
    """Reciprocal support for the exact antisymmetry Delta(k)=-Deltaᵀ(-k)."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    minus_k_indices: Array
    k_weights: Array
    mesh_id: str = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_order: FermionModeOrder,
        minus_k_indices: ArrayLike,
        k_weights: ArrayLike,
        /,
        *,
        mesh_id: str,
        energy_unit: str,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        minus = np.asarray(minus_k_indices)
        weights = np.asarray(k_weights)
        if (
            minus.ndim != 1
            or minus.size == 0
            or not np.issubdtype(minus.dtype, np.integer)
        ):
            raise TypeError("minus_k_indices must be a nonempty integer vector.")
        if (
            weights.shape != minus.shape
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "k_weights must be finite, non-negative, and match the mesh."
            )
        if not np.isclose(float(np.sum(weights)), 1.0, atol=tolerance):
            raise ValueError("BdG k_weights must sum to one.")
        minus = minus.astype(np.int32, copy=False)
        if np.any(minus < 0) or np.any(minus >= minus.size):
            raise ValueError("minus_k_indices contain an out-of-range point.")
        if not np.array_equal(minus[minus], np.arange(minus.size)):
            raise ValueError("The k→-k map must be an involution.")
        mesh = str(mesh_id).strip()
        unit = str(energy_unit).strip()
        tolerance_ = float(tolerance)
        if not mesh or not unit:
            raise ValueError("mesh_id and energy_unit must be explicit.")
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Pairing tolerance must be finite and positive.")
        self.mode_order = mode_order
        self.minus_k_indices = jnp.asarray(minus)
        self.k_weights = jnp.asarray(weights)
        self.mesh_id = mesh
        self.energy_unit = unit
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fermionic-pairing-plan",
                "mode_order": mode_order.order_id,
                "mesh": mesh,
                "energy_unit": unit,
                "minus_k_indices": minus.tolist(),
                "k_weights": weights.tolist(),
                "tolerance": tolerance_,
            }
        )

    @property
    def k_count(self) -> int:
        return self.minus_k_indices.size


class FermionicBdGPlan(StrictModule):
    convention: NambuConvention
    pairing: FermionicPairingPlan
    maximum_mode_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        convention: NambuConvention,
        pairing: FermionicPairingPlan,
        /,
        *,
        maximum_mode_count: int,
    ):
        if not isinstance(convention, NambuConvention) or not isinstance(
            pairing, FermionicPairingPlan
        ):
            raise TypeError("BdG plan requires NambuConvention and FermionicPairingPlan.")
        if convention.mode_order.order_id != pairing.mode_order.order_id:
            raise ValueError("Nambu and pairing plans use different fermion mode orders.")
        maximum = int(maximum_mode_count)
        if maximum < 1 or convention.mode_order.mode_count > maximum:
            raise ValueError(
                "BdG particle mode count exceeds the explicit resource policy."
            )
        self.convention = convention
        self.pairing = pairing
        self.maximum_mode_count = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fermionic-bdg-plan",
                "convention": convention.convention_id,
                "pairing": pairing.plan_id,
                "maximum_mode_count": maximum,
            }
        )


class PreparedFermionicBdG(StrictModule):
    plan: FermionicBdGPlan
    normal_hamiltonians: Array
    pairing_matrices: Array
    chemical_potential: Array
    double_counting_constant: Array
    normal_family_id: str = eqx.field(static=True)
    pairing_family_id: str = eqx.field(static=True)
    numeric_revision: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class BdGSpectrumResult(StrictModule):
    hamiltonians: Array
    eigenvalues: Array
    eigenvectors: Array
    particle_hole_residual: Array
    spectral_pairing_residual: Array
    hermiticity_residual: Array
    minimum_direct_gap: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)


class PairingObservableResult(StrictModule):
    normal_density: Array
    anomalous_density: Array
    particle_number: Array
    entropy: Array
    quasiparticle_grand_potential: Array
    nambu_half_trace_correction: Array
    double_counting_constant: Array
    grand_potential: Array
    helmholtz_free_energy: Array
    minimum_direct_gap: Array
    density_particle_hole_residual: Array
    spectrum: BdGSpectrumResult


def prepare_fermionic_bdg(
    plan: FermionicBdGPlan,
    normal_hamiltonians: ArrayLike,
    pairing_matrices: ArrayLike,
    /,
    *,
    chemical_potential: ArrayLike,
    double_counting_constant: ArrayLike = 0.0,
    normal_family_id: str,
    pairing_family_id: str,
    numeric_revision: str = "0",
) -> PreparedFermionicBdG:
    """Bind periodic-family evaluations; this function never recomputes phases."""

    if not isinstance(plan, FermionicBdGPlan):
        raise TypeError("plan must be FermionicBdGPlan.")
    normal = np.asarray(normal_hamiltonians)
    pairing = np.asarray(pairing_matrices)
    count = plan.convention.mode_order.mode_count
    expected = (plan.pairing.k_count, count, count)
    if normal.shape != expected or pairing.shape != expected:
        raise ValueError(f"Normal and pairing blocks must both have shape {expected}.")
    mu = np.asarray(chemical_potential)
    constant = np.asarray(double_counting_constant)
    if mu.shape != () or constant.shape != ():
        raise ValueError(
            "chemical_potential and double_counting_constant must be scalar."
        )
    if (
        np.any(~np.isfinite(normal))
        or np.any(~np.isfinite(pairing))
        or not np.isfinite(mu)
        or not np.isfinite(constant)
    ):
        raise ValueError("BdG numeric values must be finite.")
    tolerance = plan.pairing.tolerance
    scale = max(float(np.max(np.abs(normal), initial=0.0)), 1.0)
    if (
        np.max(np.abs(normal - np.swapaxes(np.conj(normal), -1, -2)), initial=0.0)
        > tolerance * scale
    ):
        raise ValueError("Every normal Bloch Hamiltonian must be Hermitian.")
    minus = np.asarray(plan.pairing.minus_k_indices)
    antisymmetry = pairing + np.swapaxes(pairing[minus], -1, -2)
    pair_scale = max(float(np.max(np.abs(pairing), initial=0.0)), 1.0)
    if np.max(np.abs(antisymmetry), initial=0.0) > tolerance * pair_scale:
        raise ValueError("Pair antisymmetry requires Delta(k)=-Delta^T(-k).")
    normal_id = str(normal_family_id).strip()
    pairing_id = str(pairing_family_id).strip()
    revision = str(numeric_revision).strip()
    if not normal_id or not pairing_id or not revision:
        raise ValueError("BdG family IDs and numeric_revision must be non-empty.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-fermionic-bdg",
            "plan": plan.plan_id,
            "normal_family": normal_id,
            "pairing_family": pairing_id,
            "revision": revision,
            "arrays": array_tree_fingerprint(
                {
                    "normal": normal,
                    "pairing": pairing,
                    "mu": mu,
                    "double_counting": constant,
                }
            ),
        }
    )
    return PreparedFermionicBdG(
        plan,
        jnp.asarray(normal),
        jnp.asarray(pairing),
        jnp.asarray(mu),
        jnp.asarray(constant),
        normal_id,
        pairing_id,
        revision,
        prepared_id,
    )


def refresh_fermionic_bdg(
    prepared: PreparedFermionicBdG,
    /,
    *,
    normal_hamiltonians: ArrayLike | None = None,
    pairing_matrices: ArrayLike | None = None,
    chemical_potential: ArrayLike | None = None,
    double_counting_constant: ArrayLike | None = None,
    numeric_revision: str,
) -> PreparedFermionicBdG:
    if not isinstance(prepared, PreparedFermionicBdG):
        raise TypeError("prepared must be PreparedFermionicBdG.")
    return prepare_fermionic_bdg(
        prepared.plan,
        prepared.normal_hamiltonians
        if normal_hamiltonians is None
        else normal_hamiltonians,
        prepared.pairing_matrices if pairing_matrices is None else pairing_matrices,
        chemical_potential=prepared.chemical_potential
        if chemical_potential is None
        else chemical_potential,
        double_counting_constant=(
            prepared.double_counting_constant
            if double_counting_constant is None
            else double_counting_constant
        ),
        normal_family_id=prepared.normal_family_id,
        pairing_family_id=prepared.pairing_family_id,
        numeric_revision=numeric_revision,
    )


def _bdg_hamiltonians(
    normal: Array,
    pairing: Array,
    minus_k_indices: Array,
    chemical_potential: Array,
    /,
) -> Array:
    count = normal.shape[-1]
    identity = jnp.eye(count, dtype=jnp.result_type(normal, pairing))
    particle = normal - chemical_potential * identity
    hole = -jnp.swapaxes(normal[minus_k_indices], -1, -2) + chemical_potential * identity
    upper = jnp.concatenate((particle, pairing), axis=-1)
    lower = jnp.concatenate((jnp.conj(jnp.swapaxes(pairing, -1, -2)), hole), axis=-1)
    return jnp.concatenate((upper, lower), axis=-2)


def evaluate_bdg_thermal_kernel(
    normal_hamiltonians: ArrayLike,
    pairing_matrices: ArrayLike,
    minus_k_indices: ArrayLike,
    chemical_potential: ArrayLike,
    temperature_energy: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """JAX numeric BdG kernel for fixed-support nonlinear consumers.

    Structural Hermiticity, antisymmetry, k involution, and units are admitted by
    :func:`prepare_fermionic_bdg`; this kernel deliberately performs no host-side
    branch or support decisions.
    """

    normal = jnp.asarray(normal_hamiltonians)
    pairing = jnp.asarray(pairing_matrices)
    minus = jnp.asarray(minus_k_indices, dtype=jnp.int32)
    chemical = jnp.asarray(chemical_potential)
    thermal = jnp.asarray(temperature_energy)
    matrices = _bdg_hamiltonians(normal, pairing, minus, chemical)
    properties = OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "construction"}
    )
    solved = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                0.5 * (matrices + jnp.conj(jnp.swapaxes(matrices, -1, -2))),
                properties=properties,
            )
        ),
        policy=EigenSolvePolicy(
            DenseEigh(),
            count=matrices.shape[-1],
            which="smallest-algebraic",
        ),
    )
    occupations = 0.5 * (1.0 - jnp.tanh(solved.eigenvalues / (2.0 * thermal)))
    density = ein.contract(
        "kic,kc,kjc->kij",
        solved.eigenvectors,
        occupations,
        jnp.conj(solved.eigenvectors),
    )
    return matrices, solved.eigenvalues, solved.eigenvectors, density


def _hermitian_eigensystem(matrix: Array, /) -> tuple[Array, Array]:
    solved = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                0.5 * (matrix + jnp.conj(matrix.T)),
                properties=OperatorProperties(
                    self_adjoint=True, evidence={"self_adjoint": "construction"}
                ),
            )
        ),
        policy=EigenSolvePolicy(
            DenseEigh(), count=matrix.shape[0], which="smallest-algebraic"
        ),
    )
    if not bool(solved.successful):
        raise RuntimeError("Fermionic BdG Hermitian eigensolve failed.")
    return solved.eigenvalues, solved.eigenvectors


def evaluate_bdg_spectrum(prepared: PreparedFermionicBdG, /) -> BdGSpectrumResult:
    if not isinstance(prepared, PreparedFermionicBdG):
        raise TypeError("prepared must be PreparedFermionicBdG.")
    hamiltonians = _bdg_hamiltonians(
        prepared.normal_hamiltonians,
        prepared.pairing_matrices,
        prepared.plan.pairing.minus_k_indices,
        prepared.chemical_potential,
    )
    eigensystems = tuple(_hermitian_eigensystem(matrix) for matrix in hamiltonians)
    eigenvalues = jnp.stack(tuple(value for value, _ in eigensystems))
    eigenvectors = jnp.stack(tuple(value for _, value in eigensystems))
    tau_x = prepared.plan.convention.particle_hole_matrix
    minus = prepared.plan.pairing.minus_k_indices
    phs = hamiltonians + ein.contract(
        "ab,kbc,cd->kad", tau_x, jnp.conj(hamiltonians[minus]), tau_x
    )
    phs_residual = jnp.max(jnp.abs(phs), initial=0.0)
    paired_spectrum = eigenvalues + eigenvalues[minus, ::-1]
    pairing_residual = jnp.max(jnp.abs(paired_spectrum), initial=0.0)
    hermitian_residual = jnp.max(
        jnp.abs(hamiltonians - jnp.conj(jnp.swapaxes(hamiltonians, -1, -2))),
        initial=0.0,
    )
    gap = jnp.min(jnp.abs(eigenvalues))
    scale = jnp.maximum(jnp.max(jnp.abs(hamiltonians)), 1.0)
    tolerance = prepared.plan.pairing.tolerance * scale
    successful = (
        jnp.all(jnp.isfinite(hamiltonians))
        & jnp.all(jnp.isfinite(eigenvalues))
        & (phs_residual <= tolerance)
        & (pairing_residual <= tolerance)
        & (hermitian_residual <= tolerance)
    )
    return BdGSpectrumResult(
        hamiltonians,
        eigenvalues,
        eigenvectors,
        phs_residual,
        pairing_residual,
        hermitian_residual,
        gap,
        successful,
        prepared.prepared_id,
        prepared.plan.pairing.mesh_id,
    )


def evaluate_pairing_observables(
    prepared: PreparedFermionicBdG,
    /,
    *,
    temperature_energy: float,
) -> PairingObservableResult:
    """Evaluate finite-temperature densities with every Nambu half explicit."""

    thermal = float(temperature_energy)
    if not isfinite(thermal) or thermal <= 0.0:
        raise ValueError("temperature_energy=k_B T must be finite and positive.")
    spectrum = evaluate_bdg_spectrum(prepared)
    occupations = 0.5 * (1.0 - jnp.tanh(spectrum.eigenvalues / (2.0 * thermal)))
    density = ein.contract(
        "kic,kc,kjc->kij",
        spectrum.eigenvectors,
        occupations,
        jnp.conj(spectrum.eigenvectors),
    )
    count = prepared.plan.convention.mode_order.mode_count
    normal_density = density[:, :count, :count]
    anomalous_density = density[:, :count, count:]
    weights = prepared.plan.pairing.k_weights
    particle_number = jnp.real(
        jnp.sum(weights * jnp.trace(normal_density, axis1=-2, axis2=-1))
    )
    clipped = jnp.clip(occupations, jnp.finfo(occupations.dtype).tiny, 1.0)
    complements = jnp.clip(1.0 - occupations, jnp.finfo(occupations.dtype).tiny, 1.0)
    entropy = -0.5 * jnp.sum(
        weights[:, None]
        * (occupations * jnp.log(clipped) + (1.0 - occupations) * jnp.log(complements))
    )
    quasiparticle = (
        -0.5
        * thermal
        * jnp.sum(weights[:, None] * jnp.logaddexp(0.0, -spectrum.eigenvalues / thermal))
    )
    identity = jnp.eye(count, dtype=prepared.normal_hamiltonians.dtype)
    xi = prepared.normal_hamiltonians - prepared.chemical_potential * identity
    half_trace = 0.5 * jnp.real(jnp.sum(weights * jnp.trace(xi, axis1=-2, axis2=-1)))
    grand = quasiparticle + half_trace + prepared.double_counting_constant
    helmholtz = grand + prepared.chemical_potential * particle_number
    lower = density[:, count:, count:]
    density_phs = jnp.max(
        jnp.abs(
            lower
            - (
                identity[None, :, :]
                - jnp.conj(normal_density[prepared.plan.pairing.minus_k_indices])
            )
        ),
        initial=0.0,
    )
    return PairingObservableResult(
        normal_density,
        anomalous_density,
        particle_number,
        entropy,
        quasiparticle,
        half_trace,
        prepared.double_counting_constant,
        grand,
        helmholtz,
        spectrum.minimum_direct_gap,
        density_phs,
        spectrum,
    )


__all__ = [
    "BdGSpectrumResult",
    "FermionicBdGPlan",
    "FermionicPairingPlan",
    "NambuConvention",
    "PairingObservableResult",
    "PreparedFermionicBdG",
    "evaluate_bdg_spectrum",
    "evaluate_bdg_thermal_kernel",
    "evaluate_pairing_observables",
    "prepare_fermionic_bdg",
    "refresh_fermionic_bdg",
]

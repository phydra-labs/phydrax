#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic response-eigenproblem derivatives and excited-state provider boundary."""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._manifold import ElectronicManifoldResult
from ._representation import RPAStateRepresentation, TDAStateRepresentation


class ExcitedStateDerivativeResult(StrictModule, NonTrainableState):
    energy_gradients: Array
    derivative_couplings: Array | None
    energy_weighted_couplings: Array | None
    residuals: Array
    successful: Array
    route: str = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_gradients: ArrayLike,
        residuals: ArrayLike,
        successful: ArrayLike,
        route: str,
        manifold_id: str,
        /,
        *,
        derivative_couplings: ArrayLike | None = None,
        energy_weighted_couplings: ArrayLike | None = None,
    ):
        gradients = jnp.asarray(energy_gradients)
        residual = jnp.asarray(residuals, dtype=gradients.real.dtype)
        couplings = (
            None
            if derivative_couplings is None
            else jnp.asarray(derivative_couplings, dtype=gradients.dtype)
        )
        weighted = (
            None
            if energy_weighted_couplings is None
            else jnp.asarray(energy_weighted_couplings, dtype=gradients.dtype)
        )
        route_ = str(route).strip()
        manifold = str(manifold_id).strip()
        if (
            gradients.ndim < 2
            or residual.shape != (gradients.shape[0],)
            or not route_
            or not manifold
        ):
            raise ValueError("Excited gradients, residuals, or identities are invalid.")
        if (couplings is None) != (weighted is None):
            raise ValueError(
                "Derivative and energy-weighted couplings must be supplied together."
            )
        self.energy_gradients = gradients
        self.derivative_couplings = couplings
        self.energy_weighted_couplings = weighted
        self.residuals = residual
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.route = route_
        self.manifold_id = manifold
        self.result_id = canonical_fingerprint(
            {
                "kind": "excited-state-derivative-result",
                "route": route_,
                "manifold": manifold,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy_gradients": np.asarray(gradients),
                        "derivative_couplings": None
                        if couplings is None
                        else np.asarray(couplings),
                        "energy_weighted_couplings": None
                        if weighted is None
                        else np.asarray(weighted),
                        "residuals": np.asarray(residual),
                    }
                ),
            }
        )


class AbstractExcitedDerivativeProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(
        self, manifold: ElectronicManifoldResult, /
    ) -> ExcitedStateDerivativeResult:
        raise NotImplementedError


ExcitedDerivativeEvaluator = Callable[
    [ElectronicManifoldResult], ExcitedStateDerivativeResult
]


class CallableExcitedDerivativeProvider(AbstractExcitedDerivativeProvider):
    evaluator: ExcitedDerivativeEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: ExcitedDerivativeEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(
        self, manifold: ElectronicManifoldResult, /
    ) -> ExcitedStateDerivativeResult:
        result = self.evaluator(manifold)
        if (
            not isinstance(result, ExcitedStateDerivativeResult)
            or result.manifold_id != manifold.result_id
        ):
            raise ValueError(
                "Excited derivative provider changed result type or manifold identity."
            )
        return result


def tda_eigenvalue_derivatives(
    representation: TDAStateRepresentation,
    response_derivatives: ArrayLike,
    /,
) -> Array:
    if not isinstance(representation, TDAStateRepresentation):
        raise TypeError("representation must be TDAStateRepresentation.")
    derivative = jnp.asarray(response_derivatives, dtype=representation.amplitudes.dtype)
    dimension, roots = representation.amplitudes.shape
    if derivative.shape[:2] != (dimension, dimension):
        raise ValueError("TDA response derivatives must start with response matrix axes.")
    return jnp.real(
        contract(
            "ir,ij...,jr->r...",
            jnp.conj(representation.amplitudes),
            derivative,
            representation.amplitudes,
        )
    )


def rpa_eigenvalue_derivatives(
    representation: RPAStateRepresentation,
    response_derivatives: ArrayLike,
    /,
) -> Array:
    if not isinstance(representation, RPAStateRepresentation):
        raise TypeError("representation must be RPAStateRepresentation.")
    right = jnp.concatenate(
        (representation.x_amplitudes, representation.y_amplitudes), axis=0
    )
    derivative = jnp.asarray(response_derivatives, dtype=right.dtype)
    if derivative.shape[:2] != (right.shape[0], right.shape[0]):
        raise ValueError("RPA response derivatives must start with full response axes.")
    return jnp.real(
        contract(
            "ir,ij...,jr->r...",
            jnp.conj(representation.left_amplitudes),
            derivative,
            right,
        )
    )


def tda_derivative_couplings(
    representation: TDAStateRepresentation,
    excitation_energies: ArrayLike,
    response_derivatives: ArrayLike,
    /,
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> tuple[Array, Array]:
    """Return derivative and energy-weighted couplings for a TDA eigenproblem."""
    if not isinstance(representation, TDAStateRepresentation):
        raise TypeError("representation must be TDAStateRepresentation.")
    amplitudes = representation.amplitudes
    energy = jnp.asarray(excitation_energies, dtype=amplitudes.real.dtype)
    derivative = jnp.asarray(response_derivatives, dtype=amplitudes.dtype)
    if energy.shape != (amplitudes.shape[1],) or derivative.shape[:2] != (
        amplitudes.shape[0],
        amplitudes.shape[0],
    ):
        raise ValueError("TDA energies and response derivatives do not align.")
    weighted = contract(
        "ir,ij...,js->rs...",
        jnp.conj(amplitudes),
        derivative,
        amplitudes,
    )
    gaps = energy[None, :] - energy[:, None]
    extra_axes = (None,) * (weighted.ndim - 2)
    safe = jnp.abs(gaps) > float(degeneracy_tolerance)
    couplings = jnp.where(
        safe[(...,) + extra_axes],
        weighted / jnp.where(safe, gaps, 1.0)[(...,) + extra_axes],
        0.0,
    )
    return couplings, weighted


def rpa_derivative_couplings(
    representation: RPAStateRepresentation,
    excitation_energies: ArrayLike,
    response_derivatives: ArrayLike,
    /,
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> tuple[Array, Array]:
    """Return biorthogonal derivative couplings for a full RPA eigenproblem."""
    if not isinstance(representation, RPAStateRepresentation):
        raise TypeError("representation must be RPAStateRepresentation.")
    right = jnp.concatenate(
        (representation.x_amplitudes, representation.y_amplitudes), axis=0
    )
    energy = jnp.asarray(excitation_energies, dtype=right.real.dtype)
    derivative = jnp.asarray(response_derivatives, dtype=right.dtype)
    if energy.shape != (right.shape[1],) or derivative.shape[:2] != (
        right.shape[0],
        right.shape[0],
    ):
        raise ValueError("RPA energies and response derivatives do not align.")
    weighted = contract(
        "ir,ij...,js->rs...",
        jnp.conj(representation.left_amplitudes),
        derivative,
        right,
    )
    gaps = energy[None, :] - energy[:, None]
    extra_axes = (None,) * (weighted.ndim - 2)
    safe = jnp.abs(gaps) > float(degeneracy_tolerance)
    couplings = jnp.where(
        safe[(...,) + extra_axes],
        weighted / jnp.where(safe, gaps, 1.0)[(...,) + extra_axes],
        0.0,
    )
    return couplings, weighted


class TDAPropertyDerivativeResult(StrictModule, NonTrainableState):
    excitation_energy_derivatives: Array
    transition_dipole_derivatives: Array
    oscillator_strength_derivatives: Array
    eigenvector_derivative_residuals: Array
    successful: Array
    representation_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        excitation_energy_derivatives,
        transition_dipole_derivatives,
        oscillator_strength_derivatives,
        eigenvector_derivative_residuals,
        successful,
        representation_id,
        /,
    ):
        energy = jnp.asarray(excitation_energy_derivatives)
        dipole = jnp.asarray(transition_dipole_derivatives, dtype=energy.dtype)
        oscillator = jnp.asarray(oscillator_strength_derivatives, dtype=energy.real.dtype)
        residual = jnp.asarray(eigenvector_derivative_residuals, dtype=energy.real.dtype)
        if (
            energy.ndim < 2
            or dipole.shape != (energy.shape[0], 3) + energy.shape[1:]
            or oscillator.shape != energy.shape
            or residual.shape != energy.shape
        ):
            raise ValueError(
                "TDA property derivatives must align by root and coordinate."
            )
        self.excitation_energy_derivatives = energy
        self.transition_dipole_derivatives = dipole
        self.oscillator_strength_derivatives = oscillator
        self.eigenvector_derivative_residuals = residual
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.representation_id = str(representation_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "tda-property-derivative-result",
                "representation": self.representation_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy),
                        "dipole": np.asarray(dipole),
                        "oscillator": np.asarray(oscillator),
                        "residual": np.asarray(residual),
                    }
                ),
            }
        )


def tda_property_derivatives(
    response_matrix: ArrayLike,
    representation: TDAStateRepresentation,
    excitation_energies: ArrayLike,
    response_derivatives: ArrayLike,
    basis_transition_dipoles: ArrayLike,
    basis_transition_dipole_derivatives: ArrayLike,
    /,
    *,
    degeneracy_tolerance: float = 1.0e-8,
    residual_tolerance: float = 1.0e-7,
    energy_scale_to_hartree: float = 1.0,
    dipole_scale_to_atomic: float = 1.0,
) -> TDAPropertyDerivativeResult:
    if not isinstance(representation, TDAStateRepresentation):
        raise TypeError("representation must be TDAStateRepresentation.")
    matrix = jnp.asarray(response_matrix)
    amplitudes = representation.amplitudes
    energies = jnp.asarray(excitation_energies, dtype=amplitudes.real.dtype)
    derivative = jnp.asarray(response_derivatives, dtype=amplitudes.dtype)
    dipoles = jnp.asarray(basis_transition_dipoles, dtype=amplitudes.dtype)
    dipole_derivative = jnp.asarray(
        basis_transition_dipole_derivatives, dtype=amplitudes.dtype
    )
    dimension, roots = amplitudes.shape
    coordinate_shape = derivative.shape[2:]
    if (
        matrix.shape != (dimension, dimension)
        or energies.shape != (roots,)
        or derivative.shape[:2] != matrix.shape
        or dipoles.shape != (dimension, 3)
        or dipole_derivative.shape != (dimension, 3) + coordinate_shape
        or not coordinate_shape
    ):
        raise ValueError("TDA response/property derivative arrays do not align.")
    spectrum = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                0.5 * (matrix + jnp.conj(matrix.T)),
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
            )
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=dimension, which="smallest-algebraic"),
    )
    full_values = spectrum.eigenvalues
    full_vectors = spectrum.eigenvectors
    coordinate_count = int(np.prod(coordinate_shape))
    derivatives_flat = derivative.reshape((dimension, dimension, coordinate_count))
    dipole_derivatives_flat = dipole_derivative.reshape((dimension, 3, coordinate_count))
    energy_gradients = []
    transition_gradients = []
    oscillator_gradients = []
    residuals = []
    for root in range(roots):
        vector = amplitudes[:, root]
        transition = jnp.conj(vector) @ dipoles
        root_energy_gradients = []
        root_transition_gradients = []
        root_oscillator_gradients = []
        root_residuals = []
        for coordinate in range(coordinate_count):
            matrix_derivative = derivatives_flat[:, :, coordinate]
            energy_derivative = jnp.real(jnp.conj(vector) @ matrix_derivative @ vector)
            couplings = jnp.conj(full_vectors.T) @ matrix_derivative @ vector
            denominators = energies[root] - full_values
            safe = jnp.abs(denominators) > float(degeneracy_tolerance)
            coefficients = jnp.where(
                safe, couplings / jnp.where(safe, denominators, 1.0), 0.0
            )
            vector_derivative = full_vectors @ coefficients
            transition_derivative = (
                jnp.conj(vector_derivative) @ dipoles
                + jnp.conj(vector) @ dipole_derivatives_flat[:, :, coordinate]
            )
            energy_atomic = energies[root] * energy_scale_to_hartree
            energy_derivative_atomic = energy_derivative * energy_scale_to_hartree
            transition_atomic = transition * dipole_scale_to_atomic
            transition_derivative_atomic = transition_derivative * dipole_scale_to_atomic
            oscillator_derivative = (2.0 / 3.0) * (
                energy_derivative_atomic * jnp.sum(jnp.abs(transition_atomic) ** 2)
                + 2.0
                * energy_atomic
                * jnp.real(
                    jnp.vdot(
                        transition_atomic,
                        transition_derivative_atomic,
                    )
                )
            )
            residual = jnp.linalg.norm(
                (matrix - energies[root] * jnp.eye(dimension)) @ vector_derivative
                + (matrix_derivative - energy_derivative * jnp.eye(dimension)) @ vector
            )
            root_energy_gradients.append(energy_derivative)
            root_transition_gradients.append(transition_derivative)
            root_oscillator_gradients.append(oscillator_derivative)
            root_residuals.append(residual)
        energy_gradients.append(jnp.stack(tuple(root_energy_gradients)))
        transition_gradients.append(jnp.stack(tuple(root_transition_gradients), axis=1))
        oscillator_gradients.append(jnp.stack(tuple(root_oscillator_gradients)))
        residuals.append(jnp.stack(tuple(root_residuals)))
    energy_result = jnp.stack(tuple(energy_gradients)).reshape(
        (roots,) + coordinate_shape
    )
    transition_result = jnp.stack(tuple(transition_gradients)).reshape(
        (roots, 3) + coordinate_shape
    )
    oscillator_result = jnp.stack(tuple(oscillator_gradients)).reshape(
        (roots,) + coordinate_shape
    )
    residual_result = jnp.stack(tuple(residuals)).reshape((roots,) + coordinate_shape)
    successful = (
        spectrum.successful
        & jnp.all(residual_result <= float(residual_tolerance))
        & jnp.all(jnp.isfinite(oscillator_result))
    )
    return TDAPropertyDerivativeResult(
        energy_result,
        transition_result,
        oscillator_result,
        residual_result,
        successful,
        representation.representation_id,
    )


__all__ = [
    "AbstractExcitedDerivativeProvider",
    "CallableExcitedDerivativeProvider",
    "ExcitedStateDerivativeResult",
    "TDAPropertyDerivativeResult",
    "rpa_derivative_couplings",
    "rpa_eigenvalue_derivatives",
    "tda_derivative_couplings",
    "tda_eigenvalue_derivatives",
    "tda_property_derivatives",
]

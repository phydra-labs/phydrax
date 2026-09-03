#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, sqrt
from numbers import Integral

import equinox as eqx
import jax.core as jax_core
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import HermitianSpectrum
from ._fock import BosonicFockSpace
from ._mode_reduction import ModeReductionProblem, NamedModeOperator


def _real_scalar(
    value: ArrayLike,
    name: str,
    /,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Array:
    if positive and nonnegative:
        raise ValueError("A scalar constraint cannot be both positive and nonnegative.")
    result = jnp.asarray(value)
    if result.shape != () or jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be one real scalar.")
    invalid = (
        ~jnp.isfinite(result) | (positive & (result <= 0)) | (nonnegative & (result < 0))
    )
    suffix = (
        " and positive." if positive else " and non-negative." if nonnegative else "."
    )
    message = f"{name} must be finite{suffix}"
    if isinstance(result, jax_core.Tracer):
        return eqx.error_if(result, invalid, message)
    if bool(invalid):
        raise ValueError(message)
    return result


class ChargeBasis(StrictModule):
    """Finite integer-charge basis for periodic superconducting modes."""

    charges: Array
    identity: Array
    phase_raising: Array
    charge: Array
    cosine: Array
    sine: Array
    cutoff: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(self, cutoff: int, /):
        if isinstance(cutoff, bool) or not isinstance(cutoff, Integral):
            raise TypeError("cutoff must be a positive integer.")
        cutoff_ = int(cutoff)
        if cutoff_ <= 0:
            raise ValueError("cutoff must be positive.")
        dimension = 2 * cutoff_ + 1
        charges = jnp.arange(-cutoff_, cutoff_ + 1, dtype=jnp.float64)
        identity = jnp.eye(dimension, dtype=jnp.complex128)
        raising = jnp.diag(jnp.ones((dimension - 1,), dtype=jnp.complex128), -1)
        charge = jnp.diag(charges).astype(jnp.complex128)
        cosine = 0.5 * (raising + jnp.conj(raising.T))
        sine = (raising - jnp.conj(raising.T)) / (2j)
        self.charges = charges
        self.identity = identity
        self.phase_raising = raising
        self.charge = charge
        self.cosine = cosine
        self.sine = sine
        self.cutoff = cutoff_
        self.dimension = dimension
        self.basis_id = canonical_fingerprint({"kind": "charge-basis", "cutoff": cutoff_})


class OscillatorBasis(StrictModule):
    """Fixed-reference oscillator basis with canonical phase and charge matrices."""

    identity: Array
    lowering: Array
    raising: Array
    number: Array
    phase: Array
    charge: Array
    cosine: Array
    sine: Array
    dimension: int = eqx.field(static=True)
    phase_scale: float = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, phase_scale: float = 1.0):
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be an integer greater than one.")
        dimension_ = int(dimension)
        if dimension_ < 2:
            raise ValueError("dimension must be greater than one.")
        scale = float(phase_scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("phase_scale must be finite and positive.")
        fock = BosonicFockSpace((dimension_,))
        lowering = fock.annihilation_matrix(0).astype(jnp.complex128)
        raising = jnp.conj(lowering.T)
        identity = jnp.eye(dimension_, dtype=jnp.complex128)
        number = fock.number_matrix(0).astype(jnp.complex128)
        position = (lowering + raising) / sqrt(2.0)
        momentum = 1j * (raising - lowering) / sqrt(2.0)
        phase = scale * position
        charge = momentum / scale
        phase_spectrum = HermitianSpectrum(phase)
        cosine = (
            phase_spectrum.eigenvectors * jnp.cos(phase_spectrum.eigenvalues)[None, :]
        ) @ jnp.conj(phase_spectrum.eigenvectors.T)
        sine = (
            phase_spectrum.eigenvectors * jnp.sin(phase_spectrum.eigenvalues)[None, :]
        ) @ jnp.conj(phase_spectrum.eigenvectors.T)
        self.identity = identity
        self.lowering = lowering
        self.raising = raising
        self.number = number
        self.phase = phase
        self.charge = charge
        self.cosine = 0.5 * (cosine + jnp.conj(cosine.T))
        self.sine = 0.5 * (sine + jnp.conj(sine.T))
        self.dimension = dimension_
        self.phase_scale = scale
        self.basis_id = canonical_fingerprint(
            {
                "kind": "oscillator-basis",
                "dimension": dimension_,
                "phase_scale": scale,
            }
        )


class TransmonParameters(StrictModule):
    """Two-junction transmon parameters in one explicit Hamiltonian convention."""

    charging_rate: Array
    left_junction_rate: Array
    right_junction_rate: Array
    offset_charge: Array
    external_phase: Array

    def __init__(
        self,
        charging_rate: ArrayLike,
        left_junction_rate: ArrayLike,
        right_junction_rate: ArrayLike,
        /,
        *,
        offset_charge: ArrayLike = 0.0,
        external_phase: ArrayLike = 0.0,
    ):
        self.charging_rate = _real_scalar(charging_rate, "charging_rate", positive=True)
        self.left_junction_rate = _real_scalar(
            left_junction_rate, "left_junction_rate", nonnegative=True
        )
        self.right_junction_rate = _real_scalar(
            right_junction_rate, "right_junction_rate", nonnegative=True
        )
        self.offset_charge = _real_scalar(offset_charge, "offset_charge")
        self.external_phase = _real_scalar(external_phase, "external_phase")


class FluxoniumParameters(StrictModule):
    """Fluxonium rates and reduced external phase."""

    charging_rate: Array
    inductive_rate: Array
    josephson_rate: Array
    external_phase: Array

    def __init__(
        self,
        charging_rate: ArrayLike,
        inductive_rate: ArrayLike,
        josephson_rate: ArrayLike,
        /,
        *,
        external_phase: ArrayLike = 0.0,
    ):
        self.charging_rate = _real_scalar(charging_rate, "charging_rate", positive=True)
        self.inductive_rate = _real_scalar(
            inductive_rate, "inductive_rate", positive=True
        )
        self.josephson_rate = _real_scalar(
            josephson_rate, "josephson_rate", nonnegative=True
        )
        self.external_phase = _real_scalar(external_phase, "external_phase")


class HarmonicModeParameters(StrictModule):
    """Harmonic-mode angular rate in the selected Hamiltonian units."""

    angular_rate: Array

    def __init__(self, angular_rate: ArrayLike, /):
        self.angular_rate = _real_scalar(angular_rate, "angular_rate", positive=True)


def _named(
    model: str,
    basis_id: str,
    name: str,
    matrix: Array,
    /,
    *,
    hermitian: bool,
) -> NamedModeOperator:
    return NamedModeOperator(
        name,
        matrix,
        hermitian=hermitian,
        operator_id=canonical_fingerprint(
            {
                "kind": "circuit-mode-operator",
                "model": model,
                "basis": basis_id,
                "name": name,
            }
        ),
    )


def transmon_mode_problem(
    parameters: TransmonParameters,
    basis: ChargeBasis,
    /,
    *,
    hbar: ArrayLike = 1.0,
    problem_id: str | None = None,
) -> ModeReductionProblem:
    """Construct a two-junction transmon in a finite integer-charge basis."""

    if not isinstance(parameters, TransmonParameters):
        raise TypeError("parameters must be TransmonParameters.")
    if not isinstance(basis, ChargeBasis):
        raise TypeError("basis must be a ChargeBasis.")
    shifted_charge = basis.charge - parameters.offset_charge * basis.identity
    charging = 4.0 * parameters.charging_rate * (shifted_charge @ shifted_charge)
    hopping = -0.5 * (
        parameters.left_junction_rate * jnp.exp(0.5j * parameters.external_phase)
        + parameters.right_junction_rate * jnp.exp(-0.5j * parameters.external_phase)
    )
    josephson = hopping * basis.phase_raising + jnp.conj(hopping) * jnp.conj(
        basis.phase_raising.T
    )
    hamiltonian = charging + josephson
    identifier = f"transmon:{basis.basis_id}" if problem_id is None else str(problem_id)
    return ModeReductionProblem(
        0.5 * (hamiltonian + jnp.conj(hamiltonian.T)),
        (
            _named("transmon", basis.basis_id, "charge", basis.charge, hermitian=True),
            _named("transmon", basis.basis_id, "cos_phase", basis.cosine, hermitian=True),
            _named("transmon", basis.basis_id, "sin_phase", basis.sine, hermitian=True),
            _named(
                "transmon",
                basis.basis_id,
                "phase_raising",
                basis.phase_raising,
                hermitian=False,
            ),
        ),
        hbar=hbar,
        problem_id=identifier,
    )


def fluxonium_mode_problem(
    parameters: FluxoniumParameters,
    basis: OscillatorBasis,
    /,
    *,
    hbar: ArrayLike = 1.0,
    problem_id: str | None = None,
) -> ModeReductionProblem:
    """Construct fluxonium in a fixed-reference oscillator basis."""

    if not isinstance(parameters, FluxoniumParameters):
        raise TypeError("parameters must be FluxoniumParameters.")
    if not isinstance(basis, OscillatorBasis):
        raise TypeError("basis must be an OscillatorBasis.")
    shifted_phase = basis.phase - parameters.external_phase * basis.identity
    hamiltonian = (
        4.0 * parameters.charging_rate * (basis.charge @ basis.charge)
        + 0.5 * parameters.inductive_rate * (shifted_phase @ shifted_phase)
        - parameters.josephson_rate * basis.cosine
    )
    identifier = f"fluxonium:{basis.basis_id}" if problem_id is None else str(problem_id)
    return ModeReductionProblem(
        0.5 * (hamiltonian + jnp.conj(hamiltonian.T)),
        (
            _named("fluxonium", basis.basis_id, "charge", basis.charge, hermitian=True),
            _named("fluxonium", basis.basis_id, "phase", basis.phase, hermitian=True),
            _named(
                "fluxonium", basis.basis_id, "cos_phase", basis.cosine, hermitian=True
            ),
            _named("fluxonium", basis.basis_id, "sin_phase", basis.sine, hermitian=True),
            _named(
                "fluxonium", basis.basis_id, "lowering", basis.lowering, hermitian=False
            ),
        ),
        hbar=hbar,
        problem_id=identifier,
    )


def harmonic_mode_problem(
    parameters: HarmonicModeParameters,
    basis: OscillatorBasis,
    /,
    *,
    hbar: ArrayLike = 1.0,
    problem_id: str | None = None,
) -> ModeReductionProblem:
    """Construct one harmonic mode with explicit quadrature scales."""

    if not isinstance(parameters, HarmonicModeParameters):
        raise TypeError("parameters must be HarmonicModeParameters.")
    if not isinstance(basis, OscillatorBasis):
        raise TypeError("basis must be an OscillatorBasis.")
    hamiltonian = parameters.angular_rate * (basis.number + 0.5 * basis.identity)
    identifier = f"harmonic:{basis.basis_id}" if problem_id is None else str(problem_id)
    return ModeReductionProblem(
        hamiltonian,
        (
            _named("harmonic", basis.basis_id, "charge", basis.charge, hermitian=True),
            _named("harmonic", basis.basis_id, "phase", basis.phase, hermitian=True),
            _named(
                "harmonic", basis.basis_id, "lowering", basis.lowering, hermitian=False
            ),
            _named("harmonic", basis.basis_id, "number", basis.number, hermitian=True),
        ),
        hbar=hbar,
        problem_id=identifier,
    )


__all__ = [
    "ChargeBasis",
    "FluxoniumParameters",
    "HarmonicModeParameters",
    "OscillatorBasis",
    "TransmonParameters",
    "fluxonium_mode_problem",
    "harmonic_mode_problem",
    "transmon_mode_problem",
]

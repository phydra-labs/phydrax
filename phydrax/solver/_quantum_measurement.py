#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical finite POVMs, instruments, branches, and addressed measurements."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..tensor_network import LocallyPurifiedDensity, MatrixProductState
from ._quantum_program import DenseQuantumProgramResult, PreparedDenseQuantumProgram


_MEASUREMENT_SHOT_ADDRESS = SampleAddress(
    "quantum", "povm-shot", target="outcome", role="shot"
)


class QuantumPOVM(StrictModule):
    """One fixed-capacity POVM with separately reported positivity/completeness."""

    effects: Array
    hermiticity_residuals: Array
    minimum_eigenvalues: Array
    completeness_residual: Array
    finite: Array
    positive_semidefinite: Array
    complete: Array
    valid: Array
    outcome_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    povm_id: str = eqx.field(static=True)

    def __init__(self, effects: ArrayLike, /, *, tolerance: float = 1e-8):
        values = jnp.asarray(effects)
        tolerance_ = float(tolerance)
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] != values.shape[2]:
            raise ValueError(
                "POVM effects require shape (outcomes, dimension, dimension)."
            )
        if not jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("POVM effects must use complex floating coordinates.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("POVM tolerance must be finite and nonnegative.")
        adjoints = jnp.swapaxes(jnp.conj(values), -1, -2)
        hermiticity = jnp.max(jnp.abs(values - adjoints), axis=(-2, -1))
        eigenvalues = jnp.linalg.eigvalsh(0.5 * (values + adjoints))
        minimum = jnp.min(eigenvalues, axis=-1)
        identity = jnp.eye(values.shape[-1], dtype=values.dtype)
        completeness = jnp.max(jnp.abs(jnp.sum(values, axis=0) - identity))
        finite = jnp.all(jnp.isfinite(values))
        positive = jnp.all(hermiticity <= tolerance_) & jnp.all(minimum >= -tolerance_)
        complete = jnp.isfinite(completeness) & (completeness <= tolerance_)
        self.effects = values
        self.hermiticity_residuals = hermiticity
        self.minimum_eigenvalues = minimum
        self.completeness_residual = completeness
        self.finite = finite
        self.positive_semidefinite = positive
        self.complete = complete
        self.valid = finite & positive & complete
        self.outcome_count = int(values.shape[0])
        self.dimension = int(values.shape[-1])
        self.tolerance = tolerance_
        self.povm_id = canonical_fingerprint(
            {
                "kind": "quantum-povm",
                "shape": values.shape,
                "dtype": str(values.dtype),
                "tolerance": tolerance_,
            }
        )


class QuantumInstrument(StrictModule):
    """Finite-outcome instrument with a fixed Kraus capacity and dynamic mask."""

    kraus: Array
    kraus_mask: Array
    effects: Array
    effect_hermiticity_residuals: Array
    minimum_effect_eigenvalues: Array
    completeness_residual: Array
    finite: Array
    completely_positive_by_construction: Array
    trace_preserving: Array
    valid: Array
    outcome_count: int = eqx.field(static=True)
    kraus_capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    instrument_id: str = eqx.field(static=True)

    def __init__(
        self,
        kraus: ArrayLike,
        kraus_mask: ArrayLike,
        /,
        *,
        tolerance: float = 1e-8,
    ):
        operators = jnp.asarray(kraus)
        mask = jnp.asarray(kraus_mask, dtype=bool)
        tolerance_ = float(tolerance)
        if (
            operators.ndim != 4
            or operators.shape[0] < 1
            or operators.shape[1] < 1
            or operators.shape[2] != operators.shape[3]
        ):
            raise ValueError(
                "Instrument Kraus operators require shape "
                "(outcomes, capacity, dimension, dimension)."
            )
        if mask.shape != operators.shape[:2]:
            raise ValueError("kraus_mask must have shape (outcomes, capacity).")
        if not jnp.issubdtype(operators.dtype, jnp.complexfloating):
            raise TypeError("Instrument Kraus operators must use complex coordinates.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Instrument tolerance must be finite and nonnegative.")
        if not bool(jax.device_get(jnp.all(jnp.any(mask, axis=1)))):
            raise ValueError(
                "Every instrument outcome requires an active Kraus operator."
            )
        weights = mask.astype(operators.real.dtype)
        effects = ein.contract(
            "ok,okai,okaj->oij", weights, jnp.conj(operators), operators
        )
        adjoints = jnp.swapaxes(jnp.conj(effects), -1, -2)
        hermiticity = jnp.max(jnp.abs(effects - adjoints), axis=(-2, -1))
        minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (effects + adjoints)), axis=-1)
        identity = jnp.eye(operators.shape[-1], dtype=operators.dtype)
        completeness = jnp.max(jnp.abs(jnp.sum(effects, axis=0) - identity))
        active_finite = jnp.all(
            jnp.where(mask[..., None, None], jnp.isfinite(operators), True)
        )
        cp = active_finite
        tp = jnp.isfinite(completeness) & (completeness <= tolerance_)
        effects_valid = jnp.all(hermiticity <= tolerance_) & jnp.all(
            minimum >= -tolerance_
        )
        self.kraus = operators
        self.kraus_mask = mask
        self.effects = effects
        self.effect_hermiticity_residuals = hermiticity
        self.minimum_effect_eigenvalues = minimum
        self.completeness_residual = completeness
        self.finite = active_finite
        self.completely_positive_by_construction = cp
        self.trace_preserving = tp
        self.valid = active_finite & cp & tp & effects_valid
        self.outcome_count = int(operators.shape[0])
        self.kraus_capacity = int(operators.shape[1])
        self.dimension = int(operators.shape[-1])
        self.tolerance = tolerance_
        self.instrument_id = canonical_fingerprint(
            {
                "kind": "quantum-instrument",
                "shape": operators.shape,
                "dtype": str(operators.dtype),
                "mask_shape": mask.shape,
                "tolerance": tolerance_,
            }
        )

    def povm(self) -> QuantumPOVM:
        return QuantumPOVM(self.effects, tolerance=self.tolerance)


class DenseInstrumentBranchResult(StrictModule):
    """Fixed-capacity dense conditional branches and explicit normalization evidence."""

    unnormalized_densities: Array
    conditional_densities: Array
    probabilities: Array
    normalization_denominators: Array
    normalization_applied: Array
    zero_probability: Array
    branch_finite: Array
    probability_sum_residual: Array
    negative_probability_residual: Array
    valid: Array
    instrument_id: str = eqx.field(static=True)


class MPSInstrumentBranchResult(StrictModule):
    """Unnormalized pure MPS branches for one-Kraus-per-outcome instruments."""

    branch_states: tuple[MatrixProductState, ...]
    probabilities: Array
    zero_probability: Array
    branch_finite: Array
    valid: Array
    site: int = eqx.field(static=True)
    instrument_id: str = eqx.field(static=True)


class LPDOInstrumentBranchResult(StrictModule):
    """Unnormalized CP/PSD LPDO branches with purification truncation evidence."""

    branch_states: tuple[LocallyPurifiedDensity, ...]
    probabilities: Array
    zero_probability: Array
    branch_finite: Array
    purification_discarded_weights: Array
    completely_positive_by_construction: Array
    positive_semidefinite_by_construction: Array
    valid: Array
    site: int = eqx.field(static=True)
    instrument_id: str = eqx.field(static=True)


def _as_density(state: ArrayLike, dimension: int, /) -> Array:
    value = jnp.asarray(state)
    if value.shape == (dimension,):
        return value[:, None] * jnp.conj(value[None, :])
    if value.shape == (dimension, dimension):
        return value
    raise ValueError("Instrument state must be a vector (d,) or density matrix (d,d).")


def apply_dense_quantum_instrument(
    instrument: QuantumInstrument,
    state: ArrayLike,
    /,
    *,
    zero_probability_tolerance: float = 1e-12,
) -> DenseInstrumentBranchResult:
    """Enumerate every outcome; conditional normalization is explicit in evidence."""
    if not isinstance(instrument, QuantumInstrument):
        raise TypeError("instrument must be QuantumInstrument.")
    threshold = float(zero_probability_tolerance)
    if not isfinite(threshold) or threshold < 0.0:
        raise ValueError("zero_probability_tolerance must be finite and nonnegative.")
    density = _as_density(state, instrument.dimension)
    weights = instrument.kraus_mask.astype(instrument.kraus.real.dtype)
    branches = ein.contract(
        "ok,okai,ij,okbj->oab",
        weights,
        instrument.kraus,
        density,
        jnp.conj(instrument.kraus),
    )
    probabilities = jnp.real(jnp.trace(branches, axis1=-2, axis2=-1))
    zero = probabilities <= threshold
    denominators = jnp.where(zero, jnp.ones_like(probabilities), probabilities)
    conditional = branches / denominators[:, None, None]
    branch_finite = jnp.all(jnp.isfinite(branches), axis=(-2, -1)) & jnp.all(
        jnp.isfinite(conditional), axis=(-2, -1)
    )
    sum_residual = jnp.abs(jnp.sum(probabilities) - jnp.real(jnp.trace(density)))
    negative = jnp.maximum(-jnp.min(probabilities), 0.0)
    valid = (
        instrument.valid
        & jnp.all(jnp.isfinite(density))
        & jnp.all(branch_finite)
        & (negative <= instrument.tolerance)
        & (sum_residual <= instrument.tolerance)
    )
    return DenseInstrumentBranchResult(
        branches,
        conditional,
        probabilities,
        denominators,
        ~zero,
        zero,
        branch_finite,
        sum_residual,
        negative,
        valid,
        instrument.instrument_id,
    )


def apply_mps_quantum_instrument(
    instrument: QuantumInstrument,
    state: MatrixProductState,
    site: int,
    /,
    *,
    zero_probability_tolerance: float = 1e-12,
) -> MPSInstrumentBranchResult:
    """Apply pure branches without densification; mixed outcomes are rejected."""
    if not isinstance(instrument, QuantumInstrument) or not isinstance(
        state, MatrixProductState
    ):
        raise TypeError("instrument/state types are invalid.")
    site_ = int(site)
    threshold = float(zero_probability_tolerance)
    if not 0 <= site_ < state.site_count:
        raise ValueError("Instrument site is outside the MPS.")
    if state.physical_dimensions[site_] != instrument.dimension:
        raise ValueError("Instrument and MPS physical dimensions differ.")
    mask = jax.device_get(instrument.kraus_mask)
    active_indices: list[int] = []
    for outcome in range(instrument.outcome_count):
        active = [
            index
            for index in range(instrument.kraus_capacity)
            if bool(mask[outcome, index])
        ]
        if len(active) != 1:
            raise ValueError(
                "Pure MPS branches require exactly one active Kraus operator per "
                "outcome; multi-Kraus mixed outcomes require LPDO execution."
            )
        active_indices.append(active[0])
    branches: list[MatrixProductState] = []
    probabilities: list[Array] = []
    finite: list[Array] = []
    for outcome, kraus_index in enumerate(active_indices):
        tensors = list(state.tensors)
        tensors[site_] = ein.contract(
            "oi,lir->lor", instrument.kraus[outcome, kraus_index], tensors[site_]
        )
        branch = MatrixProductState(tuple(tensors), precision=state.precision)
        probability = jnp.real(branch.inner(branch))
        branches.append(branch)
        probabilities.append(probability)
        finite.append(
            jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in branch.tensors])
            )
        )
    probabilities_ = jnp.stack(probabilities)
    finite_ = jnp.stack(finite)
    zero = probabilities_ <= threshold
    valid = (
        instrument.valid
        & jnp.all(finite_)
        & jnp.all(probabilities_ >= -instrument.tolerance)
        & (
            jnp.abs(jnp.sum(probabilities_) - jnp.real(state.inner(state)))
            <= instrument.tolerance
        )
    )
    return MPSInstrumentBranchResult(
        tuple(branches),
        probabilities_,
        zero,
        finite_,
        valid,
        site_,
        instrument.instrument_id,
    )


def apply_lpdo_quantum_instrument(
    instrument: QuantumInstrument,
    state: LocallyPurifiedDensity,
    site: int,
    /,
    *,
    maximum_purification_dimension: int,
    zero_probability_tolerance: float = 1e-12,
) -> LPDOInstrumentBranchResult:
    """Apply each CP outcome map as an unnormalized LPDO branch."""
    from ._purified_lindblad import apply_local_kraus_channel, LocalKrausChannel

    if not isinstance(instrument, QuantumInstrument) or not isinstance(
        state, LocallyPurifiedDensity
    ):
        raise TypeError("instrument/state types are invalid.")
    site_ = int(site)
    threshold = float(zero_probability_tolerance)
    capacity = int(maximum_purification_dimension)
    if capacity < 1:
        raise ValueError("maximum_purification_dimension must be positive.")
    if not 0 <= site_ < state.site_count:
        raise ValueError("Instrument site is outside the LPDO.")
    if state.physical_dimensions[site_] != instrument.dimension:
        raise ValueError("Instrument and LPDO physical dimensions differ.")
    branches: list[LocallyPurifiedDensity] = []
    probabilities: list[Array] = []
    finite: list[Array] = []
    discarded: list[Array] = []
    for outcome in range(instrument.outcome_count):
        masked = instrument.kraus[outcome] * instrument.kraus_mask[outcome, :, None, None]
        branch, evidence = apply_local_kraus_channel(
            state,
            LocalKrausChannel(
                site_,
                masked,
                channel_id=canonical_fingerprint(
                    {
                        "kind": "instrument-outcome-channel",
                        "instrument": instrument.instrument_id,
                        "outcome": outcome,
                    }
                ),
            ),
            maximum_purification_dimension=capacity,
        )
        probability = branch.raw_trace()
        branches.append(branch)
        probabilities.append(probability)
        finite.append(
            jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in branch.tensors])
            )
        )
        discarded.append(evidence.truncation.discarded_weight)
    probabilities_ = jnp.stack(probabilities)
    finite_ = jnp.stack(finite)
    discarded_ = jnp.stack(discarded)
    zero = probabilities_ <= threshold
    cp = jnp.full((instrument.outcome_count,), True)
    psd = jnp.full((instrument.outcome_count,), True)
    valid = (
        instrument.valid
        & jnp.all(finite_)
        & jnp.all(jnp.isfinite(discarded_))
        & jnp.all(probabilities_ >= -instrument.tolerance)
        & (jnp.abs(jnp.sum(probabilities_) - state.raw_trace()) <= instrument.tolerance)
    )
    return LPDOInstrumentBranchResult(
        tuple(branches),
        probabilities_,
        zero,
        finite_,
        discarded_,
        cp,
        psd,
        valid,
        site_,
        instrument.instrument_id,
    )


class QuantumMeasurementResult(StrictModule):
    probabilities: Array
    counts: Array
    sampled_outcomes: Array
    probability_sum_residual: Array
    negative_probability_residual: Array
    sampling_log_normalizer: Array
    valid: Array
    root_key: Array
    first_shot_address: int = eqx.field(static=True)
    povm_id: str = eqx.field(static=True)


def measure_dense_quantum_program(
    prepared: PreparedDenseQuantumProgram,
    result: DenseQuantumProgramResult,
    povm: QuantumPOVM,
    /,
    *,
    shots: int = 0,
    first_shot_address: int = 0,
    key: Key[Array, ""] = DOC_KEY0,
) -> QuantumMeasurementResult:
    """Evaluate exact probabilities and address every shot independently."""
    if (
        not isinstance(prepared, PreparedDenseQuantumProgram)
        or not isinstance(result, DenseQuantumProgramResult)
        or not isinstance(povm, QuantumPOVM)
    ):
        raise TypeError("prepared/result/povm types are invalid.")
    shot_count = int(shots)
    first = int(first_shot_address)
    if shot_count < 0 or first < 0:
        raise ValueError("shots and first_shot_address must be nonnegative.")
    if povm.dimension != prepared.plan.cost.total_dimension:
        raise ValueError("POVM dimension does not match the prepared Hilbert layout.")
    state = result.final_state
    if state.ndim != (1 if prepared.plan.state_kind == "state-vector" else 2):
        raise ValueError("POVM measurement requires one unbatched state.")
    if prepared.plan.state_kind == "state-vector":
        probabilities = jnp.real(
            jax.vmap(lambda effect: jnp.vdot(state, effect @ state))(povm.effects)
        )
    else:
        probabilities = jnp.real(
            jax.vmap(lambda effect: jnp.trace(effect @ state))(povm.effects)
        )
    sum_residual = jnp.abs(jnp.sum(probabilities) - 1.0)
    negative = jnp.maximum(-jnp.min(probabilities), 0.0)
    logits = jnp.where(probabilities > 0.0, jnp.log(probabilities), -jnp.inf)
    log_normalizer = jax.scipy.special.logsumexp(logits)
    if shot_count:
        addresses = jnp.arange(first, first + shot_count, dtype=jnp.uint32)
        keys = jax.vmap(
            lambda address: derive_key(key, _MEASUREMENT_SHOT_ADDRESS, address, 0)
        )(addresses)
        outcomes = jax.vmap(lambda shot_key: jr.categorical(shot_key, logits))(keys)
        counts = jnp.bincount(outcomes, length=povm.outcome_count)
    else:
        outcomes = jnp.empty((0,), dtype=jnp.int32)
        counts = jnp.zeros((povm.outcome_count,), dtype=jnp.int32)
    valid = (
        povm.valid
        & result.diagnostics.successful
        & jnp.all(jnp.isfinite(probabilities))
        & (negative <= prepared.plan.policy.positivity_tolerance)
        & (sum_residual <= prepared.plan.policy.trace_tolerance)
        & jnp.all(probabilities >= 0.0)
        & jnp.isfinite(log_normalizer)
        & (jnp.abs(jnp.expm1(log_normalizer)) <= prepared.plan.policy.trace_tolerance)
    )
    return QuantumMeasurementResult(
        probabilities,
        counts,
        outcomes,
        sum_residual,
        negative,
        log_normalizer,
        valid,
        jnp.asarray(key),
        first,
        povm.povm_id,
    )


__all__ = [
    "DenseInstrumentBranchResult",
    "LPDOInstrumentBranchResult",
    "MPSInstrumentBranchResult",
    "QuantumInstrument",
    "QuantumMeasurementResult",
    "QuantumPOVM",
    "apply_dense_quantum_instrument",
    "apply_lpdo_quantum_instrument",
    "apply_mps_quantum_instrument",
    "measure_dense_quantum_program",
]

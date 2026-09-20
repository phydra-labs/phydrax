#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded atomic Hamiltonian and open-channel compilation.

A plan contains immutable spectroscopy and resource policy.  Preparation is a
host operation that enumerates the finite Zeeman basis and channel support once.
The prepared arrays then feed Phydrax's dense Lindblad, finite CPTP, and quantum
jump runtimes without changing rate conventions: ``jumps`` are dimensionless,
``rates`` carry inverse-time units, and dense collapse operators multiply by
``sqrt(rate)`` exactly once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._temporal import TemporalMesh
from ...solver._finite_cptp import FiniteLindbladChannelPlan
from ...solver._lindblad import LindbladProblem
from ...solver._quantum_jump import QuantumJumpProblem, StateVectorOperator
from ._angular_momentum import (
    AtomicManifold,
    electric_dipole_allowed,
    hyperfine_dipole_coefficient,
    rotate_spherical_vector,
)


def _identifier(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


def _finite_scalar(value: complex | float, name: str, /) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_rate(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


class CoherentDrive(StrictModule, NonTrainableState):
    """Static rotating-frame drive between two named atomic manifolds."""

    rabi_frequency: Array
    polarization: Array
    frame_euler_angles: Array
    drive_id: str = eqx.field(static=True)
    lower_label: str = eqx.field(static=True)
    upper_label: str = eqx.field(static=True)

    def __init__(
        self,
        lower_label: str,
        upper_label: str,
        rabi_frequency: complex,
        polarization: ArrayLike,
        /,
        *,
        frame_euler_angles: ArrayLike = (0.0, 0.0, 0.0),
    ):
        lower = _identifier(lower_label, "lower_label")
        upper = _identifier(upper_label, "upper_label")
        if lower == upper:
            raise ValueError("A coherent dipole drive requires distinct manifolds.")
        frequency = _finite_scalar(rabi_frequency, "rabi_frequency")
        polarization_ = np.asarray(polarization, dtype=np.complex128)
        angles = np.asarray(frame_euler_angles, dtype=np.float64)
        if polarization_.shape != (3,) or np.any(~np.isfinite(polarization_)):
            raise ValueError("polarization must be finite with shape (3,).")
        if angles.shape != (3,) or np.any(~np.isfinite(angles)):
            raise ValueError("frame_euler_angles must be finite with shape (3,).")
        norm = float(np.linalg.norm(polarization_))
        if norm == 0.0:
            raise ValueError("polarization must have nonzero norm.")
        normalized = polarization_ / norm
        canonical_id = canonical_fingerprint(
            {
                "kind": "atomic-coherent-drive",
                "lower": lower,
                "upper": upper,
                "rabi_frequency": (frequency.real, frequency.imag),
                "polarization": array_tree_fingerprint(normalized),
                "frame_euler_angles": array_tree_fingerprint(angles),
            }
        )
        self.rabi_frequency = jnp.asarray(frequency)
        self.polarization = jnp.asarray(normalized)
        self.frame_euler_angles = jnp.asarray(angles)
        self.drive_id = canonical_id
        self.lower_label = lower
        self.upper_label = upper

    def quantization_frame_polarization(self, /) -> Array:
        return rotate_spherical_vector(self.polarization, self.frame_euler_angles)


class RadiativeTransition(StrictModule, NonTrainableState):
    """Spontaneous line with total rate per source Zeeman sublevel."""

    rate: Array
    upper_label: str = eqx.field(static=True)
    lower_label: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        upper_label: str,
        lower_label: str,
        rate: float,
        /,
    ):
        upper = _identifier(upper_label, "upper_label")
        lower = _identifier(lower_label, "lower_label")
        if upper == lower:
            raise ValueError("A radiative transition requires distinct manifolds.")
        rate_ = _nonnegative_rate(rate, "rate")
        canonical_id = canonical_fingerprint(
            {
                "kind": "atomic-radiative-transition",
                "upper": upper,
                "lower": lower,
                "rate": rate_,
            }
        )
        self.rate = jnp.asarray(rate_)
        self.upper_label = upper
        self.lower_label = lower
        self.transition_id = canonical_id


class LeakageTransition(StrictModule, NonTrainableState):
    """Incoherent transfer to an explicit modeled target manifold."""

    rate: Array
    source_label: str = eqx.field(static=True)
    target_label: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_label: str,
        target_label: str,
        rate: float,
        /,
    ):
        source = _identifier(source_label, "source_label")
        target = _identifier(target_label, "target_label")
        if source == target:
            raise ValueError("Leakage source and target manifolds must differ.")
        rate_ = _nonnegative_rate(rate, "rate")
        canonical_id = canonical_fingerprint(
            {
                "kind": "atomic-leakage-transition",
                "source": source,
                "target": target,
                "rate": rate_,
            }
        )
        self.rate = jnp.asarray(rate_)
        self.source_label = source
        self.target_label = target
        self.transition_id = canonical_id


class AtomicCompilationEvidence(StrictModule, NonTrainableState):
    """Measured preparation residuals and fixed-resource accounting."""

    hamiltonian_hermiticity_residual: Array
    branching_normalization_residual: Array
    trace_preservation_residual: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    active_channel_count: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    estimated_materialization_bytes: int = eqx.field(static=True)
    selection_rules_enforced: bool = eqx.field(static=True)
    rate_convention: str = eqx.field(static=True)
    status: str = eqx.field(static=True)
    failure_reason: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class AtomicQuantumPlan(StrictModule, NonTrainableState):
    """Immutable finite atomic-sector plan with explicit dense resource guards."""

    manifolds: tuple[AtomicManifold, ...]
    drives: tuple[CoherentDrive, ...]
    radiative_transitions: tuple[RadiativeTransition, ...]
    leakage_transitions: tuple[LeakageTransition, ...]
    maximum_dimension: int = eqx.field(static=True)
    maximum_liouville_elements: int = eqx.field(static=True)
    maximum_channels: int = eqx.field(static=True)
    maximum_materialization_bytes: int = eqx.field(static=True)
    estimated_materialization_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifolds: Sequence[AtomicManifold],
        /,
        *,
        drives: Sequence[CoherentDrive] = (),
        radiative_transitions: Sequence[RadiativeTransition] = (),
        leakage_transitions: Sequence[LeakageTransition] = (),
        maximum_dimension: int = 64,
        maximum_liouville_elements: int = 16_777_216,
        maximum_channels: int = 512,
        maximum_materialization_bytes: int = 134_217_728,
    ):
        manifolds_ = tuple(manifolds)
        drives_ = tuple(drives)
        radiative_ = tuple(radiative_transitions)
        leakage_ = tuple(leakage_transitions)
        if not manifolds_ or any(
            not isinstance(value, AtomicManifold) for value in manifolds_
        ):
            raise TypeError("manifolds must be a nonempty sequence of AtomicManifold.")
        if any(not isinstance(value, CoherentDrive) for value in drives_):
            raise TypeError("drives must contain only CoherentDrive values.")
        if any(not isinstance(value, RadiativeTransition) for value in radiative_):
            raise TypeError(
                "radiative_transitions must contain only RadiativeTransition values."
            )
        if any(not isinstance(value, LeakageTransition) for value in leakage_):
            raise TypeError(
                "leakage_transitions must contain only LeakageTransition values."
            )
        labels = tuple(manifold.label for manifold in manifolds_)
        if len(set(labels)) != len(labels):
            raise ValueError("Atomic manifold labels must be unique within a plan.")
        drive_ids = tuple(drive.drive_id for drive in drives_)
        transition_ids = tuple(value.transition_id for value in (*radiative_, *leakage_))
        if len(set(drive_ids)) != len(drive_ids):
            raise ValueError("Atomic coherent drives must be unique within a plan.")
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("Atomic open transitions must be unique within a plan.")
        limits = (
            maximum_dimension,
            maximum_liouville_elements,
            maximum_channels,
            maximum_materialization_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in limits
        ):
            raise TypeError("Atomic resource limits must be integers.")
        dimension_limit, liouville_limit, channel_limit, byte_limit = map(int, limits)
        if min(dimension_limit, liouville_limit, channel_limit, byte_limit) < 1:
            raise ValueError("Atomic resource limits must be positive.")
        dimension = sum(manifold.dimension for manifold in manifolds_)
        if dimension > dimension_limit or dimension**4 > liouville_limit:
            raise ValueError("Atomic basis exceeds declared dense resource bounds.")
        estimated_bytes = (1 + channel_limit) * dimension * dimension * np.dtype(
            np.complex128
        ).itemsize + channel_limit * (
            np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize
        )
        if estimated_bytes > byte_limit:
            raise ValueError("Atomic fixed arrays exceed maximum_materialization_bytes.")
        plan_id = canonical_fingerprint(
            {
                "kind": "finite-atomic-quantum-plan",
                "manifolds": tuple(manifold.manifold_id for manifold in manifolds_),
                "drives": tuple(drive.drive_id for drive in drives_),
                "radiative": tuple(value.transition_id for value in radiative_),
                "leakage": tuple(value.transition_id for value in leakage_),
                "maximum_dimension": dimension_limit,
                "maximum_liouville_elements": liouville_limit,
                "maximum_channels": channel_limit,
                "maximum_materialization_bytes": byte_limit,
            }
        )
        self.manifolds = manifolds_
        self.drives = drives_
        self.radiative_transitions = radiative_
        self.leakage_transitions = leakage_
        self.maximum_dimension = dimension_limit
        self.maximum_liouville_elements = liouville_limit
        self.maximum_channels = channel_limit
        self.maximum_materialization_bytes = byte_limit
        self.estimated_materialization_bytes = estimated_bytes
        self.plan_id = plan_id

    def prepare(self, /) -> "PreparedAtomicQuantumSystem":
        return PreparedAtomicQuantumSystem(self)


class PreparedAtomicQuantumSystem(StrictModule, NonTrainableState):
    """Fixed-basis Hamiltonian and padded open-system channel arrays."""

    plan: AtomicQuantumPlan
    hamiltonian: Array
    jumps: Array
    rates: Array
    active_channels: Array
    evidence: AtomicCompilationEvidence
    basis_labels: tuple[tuple[str, int], ...] = eqx.field(static=True)
    channel_ids: tuple[str, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: AtomicQuantumPlan, /):
        if not isinstance(plan, AtomicQuantumPlan):
            raise TypeError("plan must be an AtomicQuantumPlan.")
        manifold_by_label = {manifold.label: manifold for manifold in plan.manifolds}
        basis_labels = tuple(
            (manifold.label, twice_m)
            for manifold in plan.manifolds
            for twice_m in manifold.magnetic_projections
        )
        basis_index = {label: index for index, label in enumerate(basis_labels)}
        dimension = len(basis_labels)
        hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
        for manifold in plan.manifolds:
            frequency = float(manifold.angular_frequency)
            for twice_m in manifold.magnetic_projections:
                index = basis_index[(manifold.label, twice_m)]
                hamiltonian[index, index] = frequency

        for drive in plan.drives:
            if (
                drive.lower_label not in manifold_by_label
                or drive.upper_label not in manifold_by_label
            ):
                raise ValueError("Coherent drive references an unknown atomic manifold.")
            lower = manifold_by_label[drive.lower_label]
            upper = manifold_by_label[drive.upper_label]
            if not electric_dipole_allowed(upper, lower):
                raise ValueError(
                    "Coherent drive violates electric-dipole selection rules."
                )
            polarization = np.asarray(drive.quantization_frame_polarization())
            rabi_frequency = complex(drive.rabi_frequency)
            for lower_m in lower.magnetic_projections:
                for upper_m in upper.magnetic_projections:
                    delta = upper_m - lower_m
                    if delta not in (-2, 0, 2):
                        continue
                    q = delta // 2
                    coefficient = hyperfine_dipole_coefficient(
                        upper, lower, upper_m, lower_m, q
                    )
                    coupling = 0.5 * rabi_frequency * polarization[q + 1] * coefficient
                    upper_index = basis_index[(upper.label, upper_m)]
                    lower_index = basis_index[(lower.label, lower_m)]
                    hamiltonian[upper_index, lower_index] += coupling
                    hamiltonian[lower_index, upper_index] += coupling.conjugate()

        estimated_bytes = plan.estimated_materialization_bytes
        jumps = np.zeros(
            (plan.maximum_channels, dimension, dimension), dtype=np.complex128
        )
        rates = np.zeros((plan.maximum_channels,), dtype=np.float64)
        active = np.zeros((plan.maximum_channels,), dtype=np.bool_)
        channel_ids: list[str] = []
        active_count = 0
        maximum_branching_residual = 0.0

        for transition in plan.radiative_transitions:
            if (
                transition.upper_label not in manifold_by_label
                or transition.lower_label not in manifold_by_label
            ):
                raise ValueError("Radiative transition references an unknown manifold.")
            upper = manifold_by_label[transition.upper_label]
            lower = manifold_by_label[transition.lower_label]
            if not electric_dipole_allowed(lower, upper):
                raise ValueError(
                    "Radiative transition violates electric-dipole selection rules."
                )
            total_rate = float(transition.rate)
            for upper_m in upper.magnetic_projections:
                branches: list[tuple[int, float]] = []
                for lower_m in lower.magnetic_projections:
                    delta = lower_m - upper_m
                    if delta not in (-2, 0, 2):
                        continue
                    q = delta // 2
                    coefficient = hyperfine_dipole_coefficient(
                        lower, upper, lower_m, upper_m, q
                    )
                    strength = coefficient * coefficient
                    if strength > 0.0:
                        branches.append((lower_m, strength))
                normalization = math.fsum(strength for _, strength in branches)
                if normalization <= 0.0:
                    raise ValueError(
                        "Allowed radiative line has no nonzero Zeeman branches."
                    )
                accumulated = 0.0
                for lower_m, strength in branches:
                    if active_count >= plan.maximum_channels:
                        raise ValueError("Atomic channels exceed maximum_channels.")
                    lower_index = basis_index[(lower.label, lower_m)]
                    upper_index = basis_index[(upper.label, upper_m)]
                    jumps[active_count, lower_index, upper_index] = 1.0
                    branch_rate = total_rate * strength / normalization
                    rates[active_count] = branch_rate
                    active[active_count] = True
                    channel_ids.append(
                        f"{transition.transition_id}:m{upper_m}:to:m{lower_m}"
                    )
                    active_count += 1
                    accumulated += branch_rate
                maximum_branching_residual = max(
                    maximum_branching_residual, abs(accumulated - total_rate)
                )

        for transition in plan.leakage_transitions:
            if (
                transition.source_label not in manifold_by_label
                or transition.target_label not in manifold_by_label
            ):
                raise ValueError("Leakage transition references an unknown manifold.")
            source = manifold_by_label[transition.source_label]
            target = manifold_by_label[transition.target_label]
            total_rate = float(transition.rate)
            branch_rate = total_rate / target.dimension
            for source_m in source.magnetic_projections:
                accumulated = 0.0
                for target_m in target.magnetic_projections:
                    if active_count >= plan.maximum_channels:
                        raise ValueError("Atomic channels exceed maximum_channels.")
                    target_index = basis_index[(target.label, target_m)]
                    source_index = basis_index[(source.label, source_m)]
                    jumps[active_count, target_index, source_index] = 1.0
                    rates[active_count] = branch_rate
                    active[active_count] = True
                    channel_ids.append(
                        f"{transition.transition_id}:m{source_m}:to:m{target_m}"
                    )
                    active_count += 1
                    accumulated += branch_rate
                maximum_branching_residual = max(
                    maximum_branching_residual, abs(accumulated - total_rate)
                )

        padded_ids = tuple(channel_ids) + tuple(
            f"inactive:{index}" for index in range(active_count, plan.maximum_channels)
        )
        hermiticity_residual = float(
            np.max(np.abs(hamiltonian - hamiltonian.conjugate().T))
        )
        identity = np.eye(dimension, dtype=np.complex128)
        dual_identity = 1j * (hamiltonian @ identity - identity @ hamiltonian)
        for operator, rate, enabled in zip(jumps, rates, active, strict=True):
            if enabled:
                square = operator.conjugate().T @ operator
                dual_identity += rate * (
                    operator.conjugate().T @ identity @ operator
                    - 0.5 * (square @ identity + identity @ square)
                )
        trace_residual = float(np.max(np.abs(dual_identity)))
        valid = (
            np.all(np.isfinite(hamiltonian))
            and np.all(np.isfinite(jumps))
            and np.all(np.isfinite(rates))
            and np.all(rates >= 0.0)
            and hermiticity_residual <= 1e-12
            and maximum_branching_residual <= 1e-12
            and trace_residual <= 1e-12
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomic-quantum-system",
                "plan": plan.plan_id,
                "hamiltonian": array_tree_fingerprint(hamiltonian),
                "jumps": array_tree_fingerprint(jumps),
                "rates": array_tree_fingerprint(rates),
                "active_channels": array_tree_fingerprint(active),
                "basis_labels": basis_labels,
                "channel_ids": padded_ids,
            }
        )
        evidence = AtomicCompilationEvidence(
            hamiltonian_hermiticity_residual=jnp.asarray(hermiticity_residual),
            branching_normalization_residual=jnp.asarray(maximum_branching_residual),
            trace_preservation_residual=jnp.asarray(trace_residual),
            valid=jnp.asarray(valid),
            dimension=dimension,
            active_channel_count=active_count,
            channel_capacity=plan.maximum_channels,
            estimated_materialization_bytes=estimated_bytes,
            selection_rules_enforced=True,
            rate_convention="dimensionless-jumps-with-separate-inverse-time-rates",
            status="complete" if valid else "failed",
            failure_reason=(
                "none"
                if valid
                else "atomic-operator-or-open-channel-certification-failed"
            ),
            plan_id=plan.plan_id,
        )
        self.plan = plan
        self.hamiltonian = jnp.asarray(hamiltonian)
        self.jumps = jnp.asarray(jumps)
        self.rates = jnp.asarray(rates)
        self.active_channels = jnp.asarray(active)
        self.evidence = evidence
        self.basis_labels = basis_labels
        self.channel_ids = padded_ids
        self.dimension = dimension
        self.prepared_id = prepared_id

    @property
    def collapse_operators(self) -> Array:
        rates = jnp.where(self.active_channels, self.rates, 0.0)
        return self.jumps * jnp.sqrt(rates)[:, None, None]

    def basis_index(self, manifold_label: str, twice_m: int, /) -> int:
        if isinstance(twice_m, bool) or not isinstance(twice_m, (int, np.integer)):
            raise TypeError("twice_m must be a doubled integer projection.")
        label = (_identifier(manifold_label, "manifold_label"), int(twice_m))
        if label not in self.basis_labels:
            raise ValueError("Atomic basis label is outside prepared support.")
        return self.basis_labels.index(label)

    def basis_state(self, manifold_label: str, twice_m: int, /) -> Array:
        index = self.basis_index(manifold_label, twice_m)
        return (
            jnp.zeros((self.dimension,), dtype=self.hamiltonian.dtype).at[index].set(1.0)
        )

    def manifold_projector(self, manifold_label: str, /) -> Array:
        label = _identifier(manifold_label, "manifold_label")
        manifold_labels = tuple(value.label for value in self.plan.manifolds)
        if label not in manifold_labels:
            raise ValueError("Unknown atomic manifold label.")
        diagonal = jnp.asarray(
            tuple(float(basis_label == label) for basis_label, _ in self.basis_labels),
            dtype=self.hamiltonian.real.dtype,
        )
        return jnp.diag(diagonal).astype(self.hamiltonian.dtype)

    def density_expectation(self, density: ArrayLike, observable: ArrayLike, /) -> Array:
        density_ = jnp.asarray(density)
        observable_ = jnp.asarray(observable)
        if density_.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("density has the wrong trailing atomic dimension.")
        if observable_.shape != (self.dimension, self.dimension):
            raise ValueError("observable has the wrong atomic dimension.")
        return jnp.real(
            ein.contract("...ij,ji->...", density_, observable_, backend="jax")
        )

    def trajectory_operator(
        self, observable: ArrayLike, /, *, operator_id: str
    ) -> StateVectorOperator:
        value = jnp.asarray(observable)
        if value.shape != (self.dimension, self.dimension):
            raise ValueError("observable has the wrong atomic dimension.")
        return StateVectorOperator.from_matrix(
            value, operator_id=_identifier(operator_id, "operator_id")
        )

    def density_problem(self, initial_density: ArrayLike, /) -> LindbladProblem:
        return LindbladProblem(
            self.hamiltonian,
            self.collapse_operators,
            initial_density,
            problem_id=self.prepared_id,
        )

    def finite_plan(
        self,
        slicing: TemporalMesh,
        /,
        *,
        tolerance: float = 1e-8,
    ) -> FiniteLindbladChannelPlan:
        if not isinstance(slicing, TemporalMesh):
            raise TypeError("slicing must be a TemporalMesh.")
        if not bool(jnp.all(slicing.active_intervals)):
            raise ValueError("Atomic finite evolution requires all intervals active.")
        return FiniteLindbladChannelPlan(
            self.hamiltonian,
            self.jumps,
            self.rates,
            slicing,
            active_jumps=self.active_channels,
            evaluation="left",
            tolerance=tolerance,
            plan_id=self.prepared_id,
        )

    def jump_problem(self, initial_state: ArrayLike, /) -> QuantumJumpProblem:
        state = jnp.asarray(initial_state)
        if state.shape != (self.dimension,):
            raise ValueError("initial_state has the wrong atomic dimension.")
        hamiltonian = StateVectorOperator.from_matrix(
            self.hamiltonian, operator_id=f"{self.prepared_id}:hamiltonian"
        )
        collapse = tuple(
            StateVectorOperator.from_matrix(operator, operator_id=channel_id)
            for operator, channel_id in zip(
                self.collapse_operators, self.channel_ids, strict=True
            )
        )
        return QuantumJumpProblem(
            hamiltonian,
            collapse,
            state,
            problem_id=self.prepared_id,
        )


__all__ = [
    "AtomicCompilationEvidence",
    "AtomicQuantumPlan",
    "CoherentDrive",
    "LeakageTransition",
    "PreparedAtomicQuantumSystem",
    "RadiativeTransition",
]

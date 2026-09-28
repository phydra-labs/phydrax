#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""State, kernel, evidence and resource contracts shared by threshold-dynamics routes.

Threshold dynamics (Merriman–Bence–Osher; Esedoglu–Otto, CPAM 68, 2015) advances a
hard label field ``l(x)`` by one heat-kernel convolution and one pointwise
assignment per step. For labels ``i, j`` with tension ``sigma_ij`` and mobility
``mu_ij`` over a physical step ``dt`` the reduced kernel time is
``tau_ij = mu_ij * sigma_ij * dt`` (length squared), and the pair kernel is

``K_ij = a_ij G(tau_alpha) + b_ij G(tau_beta)``

with heat kernels ``G(tau)`` (``exp(-tau |k|^2)`` in Fourier space). The
coefficients solve ``a sqrt(tau_alpha) + b sqrt(tau_beta) = sigma_ij`` (tension)
and ``a / sqrt(tau_alpha) + b / sqrt(tau_beta) = sigma_ij / tau_ij`` (mobility),
the two-kernel construction of Salvador and Esedoglu (J. Sci. Comput. 79, 2019).
The ``"single-gaussian"`` form requires one shared ``tau_ij`` and reduces to the
Esedoglu–Otto scheme ``K_ij = sigma_ij G(tau) / sqrt(tau)``.

With ``psi_i = sum_j K_ij * u_j`` the discrete Lyapunov energy is
``E(u) = (sqrt(pi) / 2) sum_x m(x) psi_{l(x)}(x)``, which converges to
``sum_{i<j} sigma_ij |Gamma_ij|``. It is non-increasing for every time step when
both coefficient matrices are conditionally negative semidefinite; only then is
``dissipation_admitted`` true and an energy increase a failure.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_integer, unique_identifiers
from ..combinatorial import CapacitatedAuctionEvidence
from ..interfacial_transport import (
    InterfaceMobilityEvidence,
    InterfaceMobilityMatrix,
    InterfaceTensionEvidence,
    InterfaceTensionMatrix,
)
from ..interfacial_transport._tension import _conditional_spectrum
from ..typing import (
    as_array,
    Bool,
    Dim,
    Float,
    Identifier,
    Identifiers,
    Int32,
    parse,
    Scalar,
    Scope,
    VariadicDim,
)
from ._status import _committed, ThresholdDynamicsStatus


ThresholdKernelForm: TypeAlias = Literal["single-gaussian", "two-gaussian"]

# Roundoff admission for kernel consistency and energy comparisons.
_RELATIVE_ROUNDOFF = 256.0


class _SiteDims(VariadicDim):
    """Site layout of one label field (grid axes, active sites or mesh vertices)."""


class _LabelDim(Dim, minimum=2):
    """Declared labels of one threshold-dynamics system."""


class LabelFieldState(StrictModule, NonTrainableState):
    """Hard label field bound to one prepared label and site interpretation.

    ``labels`` holds indices into the stable, ordered ``label_ids``.
    ``active_labels`` marks labels that still own sites; every owning label must
    be active and an inactive label never re-nucleates. ``epoch`` increments
    whenever a step extinguishes at least one label, so consumers can detect
    topology boundaries (no derivative is claimed across them).

    The route, prepared, and site identifiers prevent equal-shaped states from
    being reinterpreted by a different numerical or scientific preparation.
    """

    __strict_contract__ = True

    labels: Int32[_SiteDims]
    active_labels: Bool[_LabelDim]
    epoch: Int32[Scalar]
    time: Float[Scalar]
    label_ids: Identifiers[_LabelDim] = eqx.field(static=True)
    route_id: Identifier = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)
    site_id: Identifier = eqx.field(static=True)
    binding_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        labels: Array,
        active_labels: Array,
        epoch: Array,
        time: Array,
        /,
        *,
        label_ids: Sequence[str],
        route_id: str,
        prepared_id: str,
        site_id: str,
    ) -> None:
        ids = unique_identifiers(label_ids, "label_ids")
        scope = Scope()
        parse(ids, Identifiers[_LabelDim], "label_ids", scope=scope)
        labels_ = as_array(labels, Int32[_SiteDims], "labels", scope=scope)
        active = as_array(active_labels, Bool[_LabelDim], "active_labels", scope=scope)
        epoch_ = as_array(epoch, Int32[Scalar], "epoch", casting="no")
        time_ = as_array(time, Float[Scalar], "time")
        host_labels = np.asarray(labels_)
        host_active = np.asarray(active)
        if np.any(host_labels < 0) or np.any(host_labels >= len(ids)):
            raise ValueError(f"labels must index the {len(ids)} declared label_ids.")
        if np.any(~host_active[host_labels.reshape(-1)]):
            raise ValueError("Every label that owns sites must be active.")
        route = canonical_identifier(route_id, "route_id")
        prepared = canonical_identifier(prepared_id, "prepared_id")
        sites = canonical_identifier(site_id, "site_id")
        self.labels = labels_
        self.active_labels = active
        self.epoch = epoch_
        self.time = time_
        self.label_ids = ids
        self.route_id = route
        self.prepared_id = prepared
        self.site_id = sites
        self.binding_id = canonical_fingerprint(
            {
                "kind": "threshold-label-field-binding",
                "label_ids": list(ids),
                "route": route,
                "prepared": prepared,
                "sites": sites,
            }
        )

    @property
    def label_count(self) -> int:
        return len(self.label_ids)

    def label_counts(self) -> Array:
        """Number of sites owned by every declared label."""
        return jnp.bincount(self.labels.reshape(-1), length=self.label_count).astype(
            jnp.int32
        )


class HeatActionEvidence(StrictModule, NonTrainableState):
    """Route-native accuracy, work, resource, and provenance evidence.

    Exact routes report zero iterative work. Approximate mesh routes retain the
    native matrix-function status, convergence and derivative validity, bounded
    iteration/work counts, retained/workspace bytes, and operator/preparation
    identity. Step aggregation keeps the worst accuracy/status and sums action
    work without discarding the route diagnostics.
    """

    successful: Array
    error_estimate: Array
    native_status: Array
    converged: Array
    derivative_valid: Array
    iterations: Array
    setup_matvec_count: Array
    action_matvec_count: Array
    transpose_matvec_count: Array
    breakdown_status: Array
    numeric_version: Array
    method: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    retained_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    operator_id: str | None = eqx.field(static=True)
    prepared_id: str | None = eqx.field(static=True)


class AbstractThresholdHeatKernel(StrictModule):
    """Dense-label heat-kernel route over a fixed site set.

    ``combine(fields, times, coefficients)`` returns the sum of heat actions for
    label fields of shape ``site_shape + (L,)``. ``coefficients`` is either
    ``(K, L, L)`` or a structured ``(K,)`` value shared by every off-diagonal
    label pair; the latter must never be expanded to ``L x L``. ``G(tau)`` is the
    heat semigroup ``exp(tau * Laplacian)`` of the route's discrete space. The
    operator must be self-adjoint and positive semidefinite in the
    ``site_measures`` inner product so the Esedoglu–Otto dissipation argument
    applies.
    """

    site_shape: eqx.AbstractVar[tuple[int, ...]]
    resolution_length: eqx.AbstractVar[float]
    equal_site_measure: eqx.AbstractVar[bool]
    route_id: eqx.AbstractVar[str]
    site_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def site_measures(self) -> Array:
        """Measure of every site (area/volume weight), shape ``site_shape``."""

    @abc.abstractmethod
    def combine(
        self, fields: Array, times: Array, coefficients: Array, /
    ) -> tuple[Array, HeatActionEvidence]:
        """Combine dense or structured-off-diagonal label coefficients."""

    @abc.abstractmethod
    def smooth(self, fields: Array, time: Array, /) -> tuple[Array, HeatActionEvidence]:
        """Apply one heat semigroup to every label column."""


class ThresholdKernelDecomposition(StrictModule, NonTrainableState):
    """Two-Gaussian kernel coefficients derived from tension, mobility and ``dt``.

    ``times = (tau_alpha, tau_beta)``. ``coefficients`` has shape ``(2, L, L)`` for
    pairwise inputs and ``(2,)`` (one value shared by every distinct pair) when
    tension and mobility are both uniform. ``consistency_defect`` is the relative
    spread of ``tau_ij``, which must vanish (``consistent``) for the single-Gaussian
    form.
    """

    times: Array
    coefficients: Array
    pair_times: Array
    consistency_defect: Array
    consistent: Array
    tension: InterfaceTensionEvidence
    mobility: InterfaceMobilityEvidence
    kernel_negative_semidefinite: Array
    admissible: Array
    dissipation_admitted: Array
    form: ThresholdKernelForm = eqx.field(static=True)
    uniform: bool = eqx.field(static=True)

    def pair_coefficients(self, first: Array, second: Array, /) -> Array:
        """Kernel coefficients ``(a, b)`` for label-index arrays, stacked first."""
        if self.uniform:
            distinct = (first != second).astype(self.coefficients.dtype)
            return self.coefficients.reshape((2,) + (1,) * distinct.ndim) * distinct
        return self.coefficients[:, first, second]


def _pair_minimum(values: Array, uniform: bool, /) -> Array:
    if uniform:
        return values
    off_diagonal = ~jnp.eye(values.shape[0], dtype=jnp.bool_)
    return jnp.min(jnp.where(off_diagonal, values, jnp.inf))


def _pair_maximum(values: Array, uniform: bool, /) -> Array:
    if uniform:
        return values
    off_diagonal = ~jnp.eye(values.shape[0], dtype=jnp.bool_)
    return jnp.max(jnp.where(off_diagonal, values, -jnp.inf))


def decompose_threshold_kernel(
    tension: InterfaceTensionMatrix,
    mobility: InterfaceMobilityMatrix,
    time_step: Array,
    form: ThresholdKernelForm,
    /,
) -> ThresholdKernelDecomposition:
    """Kernel times and coefficients for one step (traceable in the values)."""
    uniform = tension.structure == "uniform" and mobility.structure == "uniform"
    # Mixed structured/pairwise inputs broadcast scalars directly; do not
    # materialize an extra dense copy of the structured matrix.
    sigma = tension.values
    mu = mobility.values
    pair_times = mu * sigma * time_step
    tau_min = _pair_minimum(pair_times, uniform)
    tau_max = _pair_maximum(pair_times, uniform)
    eps = jnp.finfo(pair_times.dtype).eps
    defect = (tau_max - tau_min) / tau_max
    match form:
        case "single-gaussian":
            tau_alpha = tau_max
            tau_beta = tau_max
            a = sigma / jnp.sqrt(tau_alpha)
            b = jnp.zeros_like(a)
            consistent = defect <= _RELATIVE_ROUNDOFF * eps
        case "two-gaussian":
            # Nonnegative coefficients follow from tau_alpha <= tau_ij <= tau_beta;
            # the floor 2 tau_alpha keeps the pair well posed when every tau_ij agrees.
            tau_alpha = tau_min
            tau_beta = jnp.maximum(tau_max, 2.0 * tau_min)
            safe = jnp.where(pair_times > 0.0, pair_times, 1.0)
            spread = tau_beta - tau_alpha
            a = jnp.sqrt(tau_alpha) * sigma * (tau_beta / safe - 1.0) / spread
            b = jnp.sqrt(tau_beta) * sigma * (1.0 - tau_alpha / safe) / spread
            consistent = jnp.asarray(True)
        case _:
            assert_never(form)
    if not uniform:
        diagonal = jnp.eye(pair_times.shape[0], dtype=jnp.bool_)
        a = jnp.where(diagonal, 0.0, a)
        b = jnp.where(diagonal, 0.0, b)
    tension_evidence = tension.admissibility()
    mobility_evidence = mobility.admissibility()
    if uniform:
        kernel_cnsd = (a >= 0.0) & (b >= 0.0)
    else:
        kernel_cnsd = _conditional_spectrum(a)[1] & _conditional_spectrum(b)[1]
    admissible = (
        tension_evidence.admissible
        & (tension_evidence.minimum_pair_value > 0.0)
        & mobility_evidence.admissible
        & jnp.isfinite(time_step)
        & (time_step > 0.0)
        & consistent
        & jnp.all(jnp.isfinite(a))
        & jnp.all(jnp.isfinite(b))
    )
    return ThresholdKernelDecomposition(
        times=jnp.stack((tau_alpha, tau_beta)),
        coefficients=jnp.stack((a, b)),
        pair_times=pair_times,
        consistency_defect=defect,
        consistent=consistent,
        tension=tension_evidence,
        mobility=mobility_evidence,
        kernel_negative_semidefinite=kernel_cnsd,
        admissible=admissible,
        dissipation_admitted=admissible & kernel_cnsd,
        form=form,
        uniform=uniform,
    )


class SparseCandidateEvidence(StrictModule, NonTrainableState):
    """Capacity, kernel and work evidence of the sparse candidate-label route.

    For one potential evaluation, ``required_candidates`` is the largest number
    of distinct labels in a brick halo, ``candidate_overflow`` records capacity
    failure, and ``overflowed_sites`` counts affected sites. Step evidence
    aggregates current and candidate records by maximum/OR, so every overflow
    that causes rollback remains visible even when the other state uses fewer
    labels. ``truncated_kernel_mass`` is the worst heat-kernel mass outside the
    box stencil and ``kernel_symbol_minimum`` the smallest eigenvalue factor
    (nonnegative certifies a positive semidefinite kernel). ``stencil_routes``
    counts site-by-stencil-offset gathers and is summed across a complete step.
    """

    required_candidates: Array
    candidate_overflow: Array
    overflowed_sites: Array
    active_sites: Array
    active_label_count: Array
    stencil_routes: Array
    truncated_kernel_mass: Array
    kernel_symbol_minimum: Array
    candidate_capacity: int = eqx.field(static=True)


class VolumeConstraintEvidence(StrictModule, NonTrainableState):
    """Capacitated-assignment evidence for one volume-constrained step.

    ``prices`` are the auction's numerical dual variables in kernel-potential
    units (the gauge is free for exact counts); they are not physical pressures.
    """

    auction_status: Array
    auction_success: Array
    prices: Array
    lower_counts: Array
    upper_counts: Array
    feasible_before: Array
    auction: CapacitatedAuctionEvidence


class ThresholdDynamicsEvidence(StrictModule, NonTrainableState):
    """Consumer-visible evidence of one threshold-dynamics step."""

    status: Array
    committed: Array
    energy_before: Array
    energy_after: Array
    energy_tolerance: Array
    energy_decreased: Array
    dissipation_admitted: Array
    parameters_admissible: Array
    resolution_ratio: Array
    kernel_times: Array
    label_counts: Array
    extinct_labels: Array
    changed_sites: Array
    heat: HeatActionEvidence
    volume: VolumeConstraintEvidence | None
    sparse: SparseCandidateEvidence | None

    @property
    def success(self) -> Array:
        return self.status == int(ThresholdDynamicsStatus.SUCCESS)


class ThresholdDynamicsStepResult(StrictModule):
    """Committed (or rolled-back) state and its step evidence."""

    state: LabelFieldState
    evidence: ThresholdDynamicsEvidence

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def committed(self) -> Array:
        return _committed(self.evidence.status)


class ThresholdDynamicsRunResult(StrictModule):
    """Final state and per-step evidence stacked along a leading step axis.

    A step that does not commit halts the run: later steps are skipped, keep the
    halting state, and report the halting status with ``committed = False``.
    """

    state: LabelFieldState
    initial_energy: Array
    evidence: ThresholdDynamicsEvidence

    @property
    def committed_steps(self) -> Array:
        return jnp.sum(self.evidence.committed.astype(jnp.int32))

    @property
    def status(self) -> Array:
        """Worst status over the run."""
        return jnp.max(self.evidence.status)


class ThresholdDynamicsResourcePolicy(StrictModule):
    """Admission bound on working arrays and nonuniform kernel coefficients.

    Routes hold about eight ``site x slot`` arrays of the working dtype at once
    (indicators, stencil accumulators, kernel potentials of the current and
    candidate states, and complex modal work). Nonuniform decompositions also
    retain two coefficient matrices and their pair-time matrix. Both costs are
    admitted before those coefficient arrays are formed.
    """

    maximum_working_bytes: int = eqx.field(static=True)

    def __init__(self, *, maximum_working_bytes: int = 4 * 1024**3) -> None:
        self.maximum_working_bytes = positive_integer(
            maximum_working_bytes, "maximum_working_bytes"
        )

    def coefficient_bytes(
        self, label_count: int, itemsize: int, /, *, uniform: bool
    ) -> int:
        """Retained pair-time plus two coefficient arrays."""
        count = positive_integer(label_count, "label_count")
        size = positive_integer(itemsize, "itemsize")
        estimate = 0 if uniform else 3 * count * count * size
        self._refuse(estimate, "nonuniform threshold-kernel coefficients")
        return estimate

    def admit(
        self,
        sites: int,
        width: int,
        itemsize: int,
        /,
        *,
        coefficient_bytes: int = 0,
    ) -> int:
        """Refuse total route work and coefficient storage above the bound."""
        coefficient_storage = (
            coefficient_bytes
            if isinstance(coefficient_bytes, int) and coefficient_bytes >= 0
            else -1
        )
        if coefficient_storage < 0:
            raise ValueError("coefficient_bytes must be a nonnegative integer.")
        estimate = 8 * sites * width * itemsize + coefficient_storage
        self._refuse(
            estimate,
            f"{sites} sites x {width} label slots and coefficient storage",
        )
        return estimate

    def _refuse(self, estimate: int, resource: str, /) -> None:
        if estimate > self.maximum_working_bytes:
            raise MemoryError(
                f"Threshold dynamics needs about {estimate} bytes for {resource}, "
                f"above maximum_working_bytes={self.maximum_working_bytes}; use the "
                "structured uniform or sparse candidate-label route, increase the "
                "resource bound, or reduce the problem."
            )


class ThresholdPotentials(NamedTuple):
    """Kernel potentials of one label state over per-site candidate-label slots.

    ``values[n, c]`` is ``psi`` of label ``keys[n, c]`` at site ``n`` (``inf`` on
    invalid slots); ``own[n]`` is ``psi`` of the site's current label.
    ``kernel_admitted`` certifies that the route's discrete kernel is positive
    semidefinite in the site measure (exact routes: always).
    """

    values: Array
    keys: Array
    valid: Array
    own: Array
    heat: HeatActionEvidence
    sparse: SparseCandidateEvidence | None
    kernel_admitted: Array


def _energy(own: Array, measures: Array, /) -> Array:
    """``(sqrt(pi) / 2) sum_x m(x) psi_{l(x)}(x)`` from own-label potentials."""
    return 0.5 * jnp.sqrt(jnp.pi) * jnp.sum(measures * own)


def _energy_tolerance(before: Array, after: Array, /) -> Array:
    eps = jnp.finfo(before.dtype).eps
    return _RELATIVE_ROUNDOFF * eps * (jnp.abs(before) + jnp.abs(after))


__all__ = [
    "AbstractThresholdHeatKernel",
    "decompose_threshold_kernel",
    "HeatActionEvidence",
    "LabelFieldState",
    "SparseCandidateEvidence",
    "ThresholdDynamicsEvidence",
    "ThresholdDynamicsResourcePolicy",
    "ThresholdDynamicsRunResult",
    "ThresholdDynamicsStepResult",
    "ThresholdKernelDecomposition",
    "ThresholdKernelForm",
    "ThresholdPotentials",
    "VolumeConstraintEvidence",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive biasing force: conditional mean-force estimator and force-only port.

For CVs ``xi`` with Jacobian ``J_xi`` and (optionally) holonomic constraints
``sigma`` with Jacobian ``J_sigma`` on the closure of the CV support, the dual
fields ``B = J^T (J J^T)^{-1} E`` with ``J = [J_xi; J_sigma]`` and ``E`` the CV
block of the identity satisfy ``J_xi B = I`` and ``J_sigma B = 0``.  For any
configurational density ``rho`` restricted to the CV level set,

    dA/dxi_a = E[ grad U . b_a - kT div b_a - kT b_a . grad log w | xi ],

where ``w`` is the density of the declared base measure relative to Lebesgue
measure on the ambient (constraint-restricted) coordinates.  Two base measures
are admitted: ``CONFIGURATIONAL`` (``w = 1``, no constraints in the closure)
and ``CONSTRAINED_DYNAMICS``.  The latter is the configurational marginal of
the canonical measure on the constrained phase space that RATTLE dynamics with
a constraint-projected thermostat samples: integrating the momenta tangent to
the constraint manifold gives
``delta(sigma) sqrt(det G_sigma) exp(-U/kT) dq`` with
``G_sigma = J_sigma M^-1 J_sigma^T``, so ``w = det(G_sigma)^{1/2}`` and the
Fixman term is ``- (kT/2) b_a . grad log det G_sigma``.  The divergence is the
trace of the dual-field Jacobian on the closure coordinates only; no Hessian
of ``U`` is formed.  For a pair distance in three dimensions the estimator
reduces to ``-(F_j - F_i) . u / 2 - 2 kT / r``.

Bins are half open; out-of-range samples are neither recorded nor biased.
The force-only realization applies ``ramp * clip(mean) . grad xi`` through a
typed nonconservative force port; it has no energy.  The integrated
realization is a separate conservative term built from a frozen profile.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import StrEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint
from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    HermitianSpectrum,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    LinearSolveProvenance,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    prepare,
    prepare_linearization,
    PreparedLinearization,
    solve_many,
)
from ...typing import Bool, checked, Dim, Int32, Scalar
from .._constraints import PreparedDistanceConstraints
from .._force_port import (
    AbstractAtomisticForcePort,
    AtomisticForcePortEvaluation,
    AtomisticForcePortGeometry,
)
from .._system import PreparedAtomisticSystem
from .._term_state import AbstractAtomisticTermState
from ._collective_variable import AbstractCollectiveVariableProgram
from ._cv_geometry import CollectiveVariableGeometry, reduce_status


class ABFBaseMeasure(StrEnum):
    CONFIGURATIONAL = "configurational-lebesgue"
    CONSTRAINED_DYNAMICS = "rattle-constrained-fixman"

    @property
    def density(self) -> str:
        """Explicit configurational density the estimator conditions on."""
        if self is ABFBaseMeasure.CONSTRAINED_DYNAMICS:
            return "delta(sigma) sqrt(det(J_sigma M^-1 J_sigma^T)) dq"
        return "dq"


class ABFSampleStatus(StrEnum):
    """Why a mean-force sample was not recorded."""

    OUT_OF_RANGE = "out-of-range"
    IRREGULAR = "irregular"


class AtomisticABFPlan(StrictModule, NonTrainableState):
    """Scientific declaration of one ABF estimator.

    ``minimum``/``maximum``/``bins`` define half-open bins per CV.  A periodic
    CV is first mapped to the branch ``[minimum, minimum + period)``.
    ``temperature`` must equal the thermostat temperature of the sampled
    row; ``min_samples``/``full_samples`` the per-bin ramp; ``max_force`` the
    optional cap on every applied ``dA/dxi`` component.

    PyTree: ``bins`` (shape), ``measure`` (selector) and the declared
    ``maximum_closure_atoms`` bound are static; ``temperature``,
    ``min_samples``, ``full_samples`` and ``max_force`` are dynamic scalars
    (the presence of ``max_force`` selects capping) and the exact plan digest
    is the dynamic ``abf_identity`` (``abf_id`` decodes it on the host).
    """

    variables: AbstractCollectiveVariableProgram
    minimum: Array
    maximum: Array
    bins: tuple[int, ...] = eqx.field(static=True)
    temperature: Array
    min_samples: Array
    full_samples: Array
    max_force: Array | None
    measure: ABFBaseMeasure = eqx.field(static=True)
    maximum_closure_atoms: int = eqx.field(static=True)
    abf_identity: IdentityWords

    def __init__(
        self,
        variables: AbstractCollectiveVariableProgram,
        /,
        *,
        minimum: ArrayLike,
        maximum: ArrayLike,
        bins: int | tuple[int, ...],
        temperature: float,
        min_samples: int = 200,
        full_samples: int | None = None,
        max_force: float | None = None,
        measure: ABFBaseMeasure = ABFBaseMeasure.CONFIGURATIONAL,
        maximum_closure_atoms: int = 64,
    ) -> None:
        if not isinstance(variables, AbstractCollectiveVariableProgram):
            raise TypeError("ABF requires a collective-variable program.")
        if variables.requirements:
            raise ValueError(
                "ABF CVs are evaluated at force-port boundaries, where neighborhood "
                "bundle views are not bound; use explicit-index CVs."
            )
        dimension = variables.output_size
        low = np.asarray(minimum, dtype=np.float64).reshape((-1,))
        high = np.asarray(maximum, dtype=np.float64).reshape((-1,))
        count = (
            (int(bins),) * dimension
            if isinstance(bins, int)
            else tuple(int(b) for b in bins)
        )
        if (
            low.shape != (dimension,)
            or high.shape != (dimension,)
            or len(count) != dimension
            or any(b < 1 for b in count)
            or np.any(~np.isfinite(low))
            or np.any(~np.isfinite(high))
            or np.any(high <= low)
        ):
            raise ValueError(
                "ABF bounds and bins must align with the CVs and be ordered."
            )
        for index, metric in enumerate(variables.metrics):
            if metric.period is not None and high[index] - low[index] > metric.period:
                raise ValueError("A periodic ABF range cannot exceed its period.")
        kelvin = float(temperature)
        samples = int(min_samples)
        ramp_end = 2 * samples if full_samples is None else int(full_samples)
        if (
            not math.isfinite(kelvin)
            or kelvin <= 0.0
            or samples < 0
            or ramp_end < samples
        ):
            raise ValueError(
                "ABF temperature must be positive and full_samples >= min_samples >= 0."
            )
        cap = None if max_force is None else float(max_force)
        if cap is not None and (not math.isfinite(cap) or cap <= 0.0):
            raise ValueError("max_force must be finite and positive.")
        if not isinstance(measure, ABFBaseMeasure):
            raise TypeError("measure must be an ABFBaseMeasure.")
        self.variables = variables
        self.minimum = jnp.asarray(low)
        self.maximum = jnp.asarray(high)
        self.bins = count
        self.temperature = jnp.asarray(kelvin, dtype=jnp.float64)
        self.min_samples = jnp.asarray(samples, dtype=jnp.int32)
        self.full_samples = jnp.asarray(ramp_end, dtype=jnp.int32)
        self.max_force = None if cap is None else jnp.asarray(cap, dtype=jnp.float64)
        self.measure = measure
        self.maximum_closure_atoms = int(maximum_closure_atoms)
        identity: dict[str, object] = {
            "kind": "abf-plan",
            "variables": variables.program_id,
            "bounds": array_tree_fingerprint({"minimum": low, "maximum": high}),
            "bins": list(count),
            "temperature": kelvin.hex(),
            "min_samples": samples,
            "full_samples": ramp_end,
            "max_force": None if cap is None else cap.hex(),
            "measure": measure.value,
        }
        if measure is ABFBaseMeasure.CONSTRAINED_DYNAMICS:
            # Accumulated constrained statistics mean the conditional mean force
            # of exactly this density; a history of another density never
            # resumes under this plan.
            identity["configurational_density"] = measure.density
        self.abf_identity = IdentityWords.of(identity)

    @property
    def abf_id(self) -> str:
        """Host decoding of the canonical ABF-plan digest."""
        return self.abf_identity.fingerprint()

    @property
    def bin_count(self) -> int:
        return int(np.prod(self.bins))

    @property
    def widths(self) -> Array:
        return (self.maximum - self.minimum) / jnp.asarray(
            self.bins, dtype=self.minimum.dtype
        )

    def centers(self) -> tuple[Array, ...]:
        widths = self.widths
        return tuple(
            self.minimum[axis] + (jnp.arange(count) + 0.5) * widths[axis]
            for axis, count in enumerate(self.bins)
        )

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        constraints: PreparedDistanceConstraints | None = None,
    ) -> "PreparedABF":
        return PreparedABF(self, system, constraints)


class AtomisticABFState(AbstractAtomisticTermState):
    """Per-bin sample counts and summed instantaneous ``dA/dxi`` samples.

    ``owner_identity`` is the owning plan's exact ``abf_identity`` (dynamic
    words); it is not estimator content.
    """

    __strict_contract__ = True

    counts: Array
    force_sums: Array
    out_of_range: Array
    irregular: Array
    revision: Int32[Scalar]
    successful: Bool[Scalar]
    owner_identity: IdentityWords


def _abf_state_content(state: AtomisticABFState, /) -> AtomisticABFState:
    """Host arrays of an ABF state without its owner identity words.

    Keeps the original attribute paths, so content fingerprints of the bins
    and counters are unchanged by the identity representation.
    """

    arrays = jax.tree.map(np.asarray, state)
    keep = eqx.tree_at(
        lambda value: value.owner_identity.words,
        jax.tree.map(lambda _: True, arrays),
        False,
    )
    return eqx.filter(arrays, keep)


class ABFRhsDim(Dim):
    pass


@final
class ABFLinearSolveEvidence(StrictModule, NonTrainableState):
    """Native evidence with one status per RHS and no duplicate solution values."""

    __strict_contract__ = True

    status: Int32[ABFRhsDim]
    derivative_valid: Bool[ABFRhsDim]
    diagnostics: LinearSolveDiagnostics
    provenance: LinearSolveProvenance

    @checked
    def __init__(self, result: LinearSolveResult) -> None:
        self.status = jnp.atleast_1d(result.status)
        self.derivative_valid = jnp.atleast_1d(result.derivative_valid)
        self.diagnostics = result.diagnostics
        self.provenance = result.provenance

    @property
    def successful(self) -> Array:
        return jnp.all(
            (self.status == int(LinearSolveStatus.SUCCESS)) & self.derivative_valid
        )


class ABFSample(StrictModule):
    """One instantaneous estimator sample and its admission evidence."""

    values: Array
    bin_index: Array
    in_range: Array
    regular: Array
    mean_force: Array
    gram_condition: Array
    status: Array
    dual_solve: ABFLinearSolveEvidence
    fixman_solve: ABFLinearSolveEvidence | None


class PreparedABF(StrictModule, NonTrainableState):
    """ABF estimator bound to a system, its constraint closure and units.

    ``closure`` holds the CV support plus every atom of a constraint component
    touching it; ``constraint_pairs``/``constraint_distances`` (local closure
    indices) are the constraints of that closure.  ``thermal_energy`` is a
    dynamic scalar and the exact prepared digest is the dynamic
    ``prepared_identity`` (``prepared_id`` decodes it on the host).
    """

    plan: AtomisticABFPlan
    system: PreparedAtomisticSystem
    closure: Array
    constraint_pairs: Array
    constraint_distances: Array
    closure_masses: Array
    thermal_energy: Array
    prepared_identity: IdentityWords
    bin_edges: tuple[Array, ...]

    def __init__(
        self,
        plan: AtomisticABFPlan,
        system: PreparedAtomisticSystem,
        constraints: PreparedDistanceConstraints | None,
        /,
    ) -> None:
        if plan.variables.system_id != system.prepared_id:
            raise ValueError("ABF variables belong to another atomistic system.")
        support = set(int(index) for index in plan.variables.support)
        pairs = np.zeros((0, 2), dtype=np.int64)
        distances = np.zeros((0,), dtype=np.float64)
        if constraints is not None:
            if constraints.system.prepared_id != system.prepared_id:
                raise ValueError("Constraints belong to another atomistic system.")
            all_pairs = np.asarray(system.topology.constraint_indices, dtype=np.int64)
            all_distances = np.asarray(
                system.topology.constraint_distances, dtype=np.float64
            )
            grown = True
            while grown:
                grown = False
                for a, b in all_pairs:
                    if (int(a) in support) != (int(b) in support):
                        support.update((int(a), int(b)))
                        grown = True
            keep = np.asarray(
                [int(a) in support and int(b) in support for a, b in all_pairs],
                dtype=np.bool_,
            ).reshape((-1,))
            pairs = all_pairs[keep] if all_pairs.size else pairs
            distances = all_distances[keep] if all_distances.size else distances
        closure = np.asarray(sorted(support), dtype=np.int64)
        if closure.size == 0 or closure.size > plan.maximum_closure_atoms:
            raise ValueError(
                f"The ABF closure has {closure.size} atoms; the declared bound is "
                f"{plan.maximum_closure_atoms}."
            )
        constrained = pairs.shape[0] > 0
        if constrained != (plan.measure is ABFBaseMeasure.CONSTRAINED_DYNAMICS):
            raise ValueError(
                "The declared ABF base measure must match the dynamics: constraints "
                "touching the CV closure require CONSTRAINED_DYNAMICS, none require "
                "CONFIGURATIONAL."
            )
        local = {int(slot): position for position, slot in enumerate(closure.tolist())}
        local_pairs = np.asarray(
            [[local[int(a)], local[int(b)]] for a, b in pairs], dtype=np.int32
        ).reshape((-1, 2))
        dtype = np.dtype(system.plan.coordinate_dtype)
        limit = np.finfo(dtype)
        edges = []
        for low, high, count in zip(
            np.asarray(plan.minimum), np.asarray(plan.maximum), plan.bins, strict=True
        ):
            width = (float(high) - float(low)) / count
            if low < limit.min or high > limit.max or not math.isfinite(width):
                raise ValueError("ABF bin bounds exceed the system coordinate precision.")
            edge = np.linspace(low, high, count + 1, dtype=dtype)
            if np.any(np.diff(edge) <= 0.0):
                raise ValueError(
                    "ABF bin edges must be distinct at the system coordinate precision."
                )
            edges.append(jnp.asarray(edge))
        self.plan = plan
        self.system = system
        self.closure = jnp.asarray(closure, dtype=jnp.int32)
        self.constraint_pairs = jnp.asarray(local_pairs)
        self.constraint_distances = jnp.asarray(distances)
        self.closure_masses = jnp.asarray(np.asarray(system.plan.masses)[closure])
        self.thermal_energy = jnp.asarray(
            float(system.plan.units.boltzmann_constant)
            * float(np.asarray(plan.temperature)),
            dtype=jnp.float64,
        )
        self.bin_edges = tuple(edges)
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-abf",
                "plan": plan.abf_id,
                "system": system.prepared_id,
                "closure": closure.tolist(),
                "constraints": local_pairs.tolist(),
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-ABF digest."""
        return self.prepared_identity.fingerprint()

    def initialize(self, dtype: DTypeLike = jnp.float64) -> AtomisticABFState:
        return AtomisticABFState(
            counts=jnp.zeros((self.plan.bin_count,), dtype=jnp.int32),
            force_sums=jnp.zeros(
                (self.plan.bin_count, self.plan.variables.output_size), dtype=dtype
            ),
            out_of_range=jnp.zeros((), dtype=jnp.int32),
            irregular=jnp.zeros((), dtype=jnp.int32),
            revision=jnp.zeros((), dtype=jnp.int32),
            successful=jnp.asarray(True),
            owner_identity=self.plan.abf_identity,
        )

    # ------------------------------------------------------------- geometry

    def _branch(self, values: Array, /) -> Array:
        mapped = []
        for axis, metric in enumerate(self.plan.variables.metrics):
            value = values[axis]
            if metric.period is not None:
                low = self.plan.minimum[axis].astype(value.dtype)
                value = low + jnp.mod(value - low, metric.period)
            mapped.append(value)
        return jnp.stack(mapped)

    def bin_index(self, values: ArrayLike, /) -> tuple[Array, Array]:
        """Flat half-open bin and whether every component is inside its range."""

        branch = self._branch(jnp.asarray(values))
        low = self.plan.minimum.astype(branch.dtype)
        high = self.plan.maximum.astype(branch.dtype)
        inside = jnp.all((branch >= low) & (branch < high))
        # Declared CV axes may have different bin counts. Search their prepared
        # edges: floating division can misclassify an exact interior boundary.
        raw = jnp.stack(
            tuple(
                jnp.searchsorted(edge.astype(branch.dtype), branch[axis], side="right")
                - 1
                for axis, edge in enumerate(self.bin_edges)
            )
        ).astype(jnp.int32)
        sizes = jnp.asarray(self.plan.bins, dtype=jnp.int32)
        clipped = jnp.clip(raw, 0, sizes - 1)
        strides = np.cumprod((1,) + self.plan.bins[::-1][:-1])[::-1]
        flat = jnp.sum(clipped * jnp.asarray(strides, dtype=jnp.int32))
        return flat, inside

    def _closure_geometry(
        self, geometry: CollectiveVariableGeometry, /
    ) -> tuple[Array, Callable[[Array], Array]]:
        """Closure coordinates ``q (3s,)`` and the CV map ``q -> values``."""

        base_unwrapped = jax.lax.stop_gradient(geometry.unwrapped_positions)
        base_positions = jax.lax.stop_gradient(geometry.positions)
        closure = self.closure
        q0 = base_unwrapped[closure].reshape((-1,))

        def values(q: Array) -> Array:
            coordinates = q.reshape((-1, 3))
            shift = coordinates - base_unwrapped[closure]
            moved = CollectiveVariableGeometry(
                base_positions.at[closure].add(shift),
                unwrapped_positions=base_unwrapped.at[closure].set(coordinates),
                cell=geometry.cell,
                cell_vectors=geometry.cell_vectors,
            )
            return self.plan.variables.evaluate(moved).values

        return q0, values

    def _constraint_values(self, q: Array, /) -> Array:
        coordinates = q.reshape((-1, 3))
        left, right = self.constraint_pairs[:, 0], self.constraint_pairs[:, 1]
        bond = coordinates[left] - coordinates[right]
        return (
            jnp.sum(bond * bond, axis=-1) - self.constraint_distances.astype(q.dtype) ** 2
        )

    def _jacobian(self, q: Array, values: Callable[[Array], Array], /) -> Array:
        cv = jax.jacrev(values)(q)
        if self.constraint_pairs.shape[0] == 0:
            return cv
        return jnp.concatenate((cv, jax.jacrev(self._constraint_values)(q)), axis=0)

    def _dual_fields(
        self, q: Array, values: Callable[[Array], Array], /
    ) -> tuple[Array, tuple[Array, ABFLinearSolveEvidence]]:
        """Complete local coarea covectors, their Gram condition and solve evidence."""
        jacobian = self._jacobian(q, values)
        gram = jacobian @ jacobian.T
        prepared = prepare(
            LinearSystem(DenseLinearOperator(gram)),
            LinearSolvePolicy(DenseLU()),
        )
        # Solve for the covectors themselves, not an explicit inverse of the Gram.
        result = solve_many(prepared, jacobian)
        fields = jnp.asarray(result.value)[: self.plan.variables.output_size].T
        condition = HermitianSpectrum(jax.lax.stop_gradient(gram)).condition_number
        return fields, (condition, ABFLinearSolveEvidence(result))

    @staticmethod
    def _divergence(linearization: PreparedLinearization, q: Array) -> Array:
        """Exact diagonal JVP contraction with one coordinate-sized work buffer."""
        output = jnp.asarray(linearization.primal)

        def diagonal(index: Array, accumulated: Array) -> Array:
            direction = jnp.zeros_like(q).at[index].set(1.0)
            tangent = jnp.asarray(linearization.jvp(direction))
            return accumulated + tangent[index]

        return jax.lax.fori_loop(
            0, q.size, diagonal, jnp.zeros((output.shape[1],), dtype=q.dtype)
        )

    def _fixman(
        self, q: Array, fields: Array
    ) -> tuple[Array, Array, ABFLinearSolveEvidence]:
        """Directional log-determinant actions with one prepared metric solve."""
        inverse_masses = jnp.repeat(1.0 / self.closure_masses.astype(q.dtype), 3)

        def metric(coordinates: Array) -> Array:
            rows = jax.jacrev(self._constraint_values)(coordinates)
            return (rows * inverse_masses[None, :]) @ rows.T

        linearization = prepare_linearization(metric, q)
        gram = jnp.asarray(linearization.primal)
        prepared = prepare(
            LinearSystem(DenseLinearOperator(gram)),
            LinearSolvePolicy(DenseLU()),
        )
        directions = jax.vmap(linearization.jvp, in_axes=1, out_axes=2)(fields)
        result = solve_many(prepared, directions.reshape((gram.shape[0], -1)))
        actions = jnp.asarray(result.value).reshape(directions.shape)
        correction = jnp.trace(actions, axis1=0, axis2=1)
        condition = HermitianSpectrum(jax.lax.stop_gradient(gram)).condition_number
        return correction, condition, ABFLinearSolveEvidence(result)

    # -------------------------------------------------------------- sampling

    def sample(
        self,
        geometry: CollectiveVariableGeometry,
        physical_forces: ArrayLike,
        /,
        *,
        maximum_condition: float = 1.0e10,
    ) -> ABFSample:
        """Instantaneous ``dA/dxi`` sample at one committed configuration.

        ``physical_forces`` must be the forces of the physical Hamiltonian only
        (an explicitly requested physical force-group evaluation), never total
        forces that include any bias.
        """

        evaluation = self.plan.variables.evaluate(geometry)
        flat, inside = self.bin_index(evaluation.values)
        q0, values = self._closure_geometry(geometry)
        linearization = prepare_linearization(
            lambda q: self._dual_fields(q, values), q0, has_aux=True
        )
        fields = jnp.asarray(linearization.primal)
        condition, dual_evidence = linearization.auxiliary
        kt = self.thermal_energy.astype(q0.dtype)
        divergence = self._divergence(linearization, q0)
        force = jnp.asarray(physical_forces, dtype=q0.dtype)[self.closure].reshape((-1,))
        mean_force = -(force @ fields) - kt * divergence
        fixman_evidence = None
        solve_successful = dual_evidence.successful
        if self.constraint_pairs.shape[0]:
            fixman, metric_condition, fixman_evidence = self._fixman(q0, fields)
            # w = det(G_sigma)^{1/2}:  -kT b . grad log w = -(kT/2) b . grad log det G.
            mean_force = mean_force - 0.5 * kt * fixman
            condition = jnp.maximum(condition, metric_condition)
            solve_successful = solve_successful & fixman_evidence.successful
        regular = (
            evaluation.successful
            & evaluation.regular
            & solve_successful
            & (condition < maximum_condition)
            & jnp.all(jnp.isfinite(mean_force))
        )
        status = reduce_status(evaluation.status)
        return ABFSample(
            values=evaluation.values,
            bin_index=flat,
            in_range=inside,
            regular=regular,
            mean_force=jnp.where(regular, mean_force, jnp.nan),
            gram_condition=condition,
            status=jnp.asarray(status, dtype=jnp.int32),
            dual_solve=dual_evidence,
            fixman_solve=fixman_evidence,
        )

    def ramp(self, counts: Array, /) -> Array:
        minimum = self.plan.min_samples.astype(jnp.float64)
        full = self.plan.full_samples.astype(jnp.float64)
        span = full - minimum
        value = counts.astype(jnp.float64)
        degenerate = span <= 0.0
        fraction = jnp.where(
            degenerate,
            (value >= full).astype(value.dtype),
            jnp.clip((value - minimum) / jnp.where(degenerate, 1.0, span), 0.0, 1.0),
        )
        return jnp.where(counts > 0, fraction, 0.0)

    def accumulate(
        self, state: AtomisticABFState, sample: ABFSample, /
    ) -> AtomisticABFState:
        """Record one committed sample; the revision advances only when the
        applied force field of the sampled bin changes.  A state owned by
        another ABF plan records nothing and is marked unsuccessful."""

        if state.counts.shape != (self.plan.bin_count,):
            raise ValueError("ABF state does not match the plan's bin layout.")
        bound = state.owner_identity.equals(self.plan.abf_identity)
        record = sample.in_range & sample.regular & bound
        index = sample.bin_index
        counts = state.counts.at[index].add(record.astype(jnp.int32))
        sums = state.force_sums.at[index].add(
            jnp.where(record, sample.mean_force.astype(state.force_sums.dtype), 0.0)
        )
        changed = record & (self.ramp(counts[index]) > 0.0)
        return AtomisticABFState(
            counts=counts,
            force_sums=sums,
            out_of_range=state.out_of_range + (~sample.in_range).astype(jnp.int32),
            irregular=state.irregular
            + (sample.in_range & ~sample.regular).astype(jnp.int32),
            revision=state.revision + changed.astype(jnp.int32),
            successful=state.successful & bound,
            owner_identity=state.owner_identity,
        )

    def mean_force(self, state: AtomisticABFState, /) -> Array:
        """Per-bin ``dA/dxi`` estimate; NaN where a bin has no sample."""

        counts = state.counts[:, None].astype(state.force_sums.dtype)
        return jnp.where(counts > 0, state.force_sums / jnp.maximum(counts, 1.0), jnp.nan)

    def applied_gradient(self, state: AtomisticABFState, /) -> Array:
        """Ramped, capped ``dA/dxi`` actually applied, zero where nothing applies."""

        counts = state.counts[:, None].astype(state.force_sums.dtype)
        estimate = jnp.where(counts > 0, state.force_sums / jnp.maximum(counts, 1.0), 0.0)
        if self.plan.max_force is not None:
            cap = self.plan.max_force.astype(estimate.dtype)
            estimate = jnp.clip(estimate, -cap, cap)
        return estimate * self.ramp(state.counts)[:, None].astype(estimate.dtype)


class AtomisticABFForcePort(AbstractAtomisticForcePort):
    """Force-only ABF realization: ``F = sum_a g_a grad xi_a`` in the current bin.

    ``g`` is the ramped, capped mean-force estimate (``V_bias = -A`` locally,
    so the bias force is ``+dA/dxi grad xi``).  Out of range nothing applies.
    """

    abf: PreparedABF
    name: str = eqx.field(static=True)
    owner_identity: IdentityWords
    system_identity: IdentityWords

    def __init__(self, abf: PreparedABF, /, *, name: str) -> None:
        label = str(name).strip()
        if not label:
            raise ValueError("A force port requires a non-empty name.")
        self.abf = abf
        self.name = label
        self.owner_identity = abf.plan.abf_identity
        self.system_identity = abf.system.prepared_identity

    @property
    def port_id(self) -> str:
        """Host: the port's exact declared identity (slot and prepared ABF)."""
        return f"abf-force-port:{self.name}:{self.abf.prepared_id}"

    def initial_term_state(self, dtype: DTypeLike, /) -> AtomisticABFState:
        return self.abf.initialize(dtype)

    @staticmethod
    def _pulled_back(
        linearization: PreparedLinearization, gradient: Array, q: Array, /
    ) -> Array:
        """``gradient @ dxi/dq`` without the CV-by-coordinate Jacobian.

        Matching CV and coordinate dtypes use one pullback. Another CV dtype pulls
        back exact output-basis rows one at a time and accumulates them in the
        coordinate dtype, as the dense contraction did; the cotangent is never
        narrowed.
        """
        output = jnp.asarray(linearization.primal)
        if output.dtype == q.dtype:
            return jnp.asarray(linearization.vjp(gradient))

        def accumulate(index: Array, total: Array) -> Array:
            basis = jnp.zeros_like(output).at[index].set(1)
            return total + gradient[index] * jnp.asarray(linearization.vjp(basis))

        return jax.lax.fori_loop(0, output.size, accumulate, jnp.zeros_like(q))

    def forces(
        self,
        geometry: AtomisticForcePortGeometry,
        state: AbstractAtomisticTermState,
        /,
    ) -> AtomisticForcePortEvaluation:
        if not isinstance(state, AtomisticABFState):
            raise TypeError("The ABF force port requires its own AtomisticABFState.")
        system = self.abf.system
        cv_geometry = CollectiveVariableGeometry(
            geometry.positions,
            unwrapped_positions=geometry.unwrapped_positions,
            cell=system.cell,
            cell_vectors=geometry.cell_vectors,
        )
        evaluation = self.abf.plan.variables.evaluate(cv_geometry)
        flat, inside = self.abf.bin_index(evaluation.values)
        q0, values = self.abf._closure_geometry(cv_geometry)
        linearization = prepare_linearization(values, q0)
        gradient = self.abf.applied_gradient(state)[flat].astype(q0.dtype)
        local = self._pulled_back(linearization, gradient, q0).reshape((-1, 3))
        regular = evaluation.successful & evaluation.regular
        bound = state.owner_identity.equals(self.owner_identity)
        apply = inside & regular & state.successful & bound
        forces = (
            jnp.zeros_like(geometry.positions)
            .at[self.abf.closure]
            .add(jnp.where(apply, local, 0.0))
        )
        status = reduce_status(evaluation.status)
        return AtomisticForcePortEvaluation(
            forces=forces,
            successful=state.successful & bound & (~inside | regular),
            status=jnp.asarray(status, dtype=jnp.int32),
        )


__all__ = [
    "ABFBaseMeasure",
    "ABFLinearSolveEvidence",
    "ABFSample",
    "AtomisticABFForcePort",
    "AtomisticABFPlan",
    "AtomisticABFState",
    "PreparedABF",
]

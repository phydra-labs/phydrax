#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative collective-variable biases as terms of the scalar program.

:class:`AtomisticBiasTerm` is an ordinary energy term: it evaluates its CV
program on the program's current (strain-mapped) geometry and adds one
conservative scalar.  Forces, constraint projection, stress/strain response,
Monte Carlo cell trials and reduced-potential cross-evaluation therefore all
see the same bias definition.  Controls (umbrella centers, moving-restraint
centers, history owner keys) and adaptive kernel histories are read-only term
state bound under the term's declared slot ``name`` and validated by its exact
``owner_identity`` words; nothing here advances them.
"""

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint
from ..._identity import IdentityWords
from ..._interpolation import (
    apply_gather_stencil,
    cubic_hermite_interpolate,
    rectilinear_stencil,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import HermitianSpectrum
from ...typing import Bool, Int32, Scalar
from .._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from .._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from .._system import PreparedAtomisticSystem
from .._term_state import AbstractAtomisticTermState, AtomisticTermStateBindings
from .._thermodynamic import AtomisticBiasRowBinding
from ._adaptive_profile import AdaptiveProfileEstimator, AdaptiveProfileResult
from ._bias_history import (
    AtomisticBiasHistoryPlan,
    AtomisticBiasHistoryState,
    bias_history_digest,
    BiasHistoryOwnership,
)
from ._collective_variable import (
    AbstractCollectiveVariableProgram,
    CollectiveVariableMetric,
    CollectiveVariableProgramEvaluation,
)
from ._cv_geometry import CollectiveVariableGeometry


class BiasFormEvaluation(StrictModule):
    """One form energy with its admission evidence.

    ``status`` is a :class:`CollectiveVariableStatus` word of the form's own
    geometric operations (zero for forms without any);
    ``minimum_relative_gap`` is the smallest relative alignment eigengap of
    an aligned configuration kernel, ``None`` for forms that align nothing.
    """

    energy: Array
    successful: Array
    status: Array
    minimum_relative_gap: Array | None

    @classmethod
    def without_geometry(
        cls, energy: Array, successful: ArrayLike, /
    ) -> BiasFormEvaluation:
        """Evaluation of a form that performs no geometric operation of its own."""
        return cls(
            energy=energy,
            successful=jnp.asarray(successful),
            status=jnp.zeros((), dtype=jnp.int32),
            minimum_relative_gap=None,
        )


class AbstractBiasForm(StrictModule, NonTrainableState):
    """Scalar bias as a function of CV values, controls and a read-only history.

    ``control_size`` scalars (e.g. umbrella centers) are dynamic term state so
    window rows and moving restraints change them without re-preparation.
    ``history`` declares an adaptive kernel history (``None`` for static forms).
    ``form_identity`` is the exact canonical form digest as dynamic words
    (``form_id`` decodes it on the host); numerical parameters are dynamic.
    """

    form_identity: eqx.AbstractVar[IdentityWords]
    dimension: eqx.AbstractVar[int]
    control_size: eqx.AbstractVar[int]
    history: eqx.AbstractVar[AtomisticBiasHistoryPlan | None]

    @property
    def form_id(self) -> str:
        """Host decoding of the canonical bias-form digest."""
        return self.form_identity.fingerprint()

    def initial_controls(self) -> Array:
        return jnp.zeros((0,), dtype=jnp.float64)

    @abc.abstractmethod
    def evaluate(
        self,
        values: Array,
        metrics: tuple[CollectiveVariableMetric, ...],
        controls: Array,
        history: AtomisticBiasHistoryState | None,
        owner_key: Array,
        /,
    ) -> BiasFormEvaluation:
        """The form energy at one CV vector with its admission evidence."""
        raise NotImplementedError


def _vector(value: ArrayLike, dimension: int, name: str, /) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape((-1,))
    if array.size == 1:
        array = np.full((dimension,), float(array[0]))
    if array.shape != (dimension,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or one value per CV.")
    return array


class HarmonicBiasForm(AbstractBiasForm):
    """Umbrella ``(1/2) d^T K d`` of the metric displacement ``d = x (-) c``.

    ``stiffness`` is scalar, diagonal ``(d,)`` or a full symmetric ``(d, d)``
    matrix whose positive semidefiniteness is established by the native
    Hermitian spectrum (``stiffness_minimum_eigenvalue``/``stiffness_condition``
    are retained evidence, dynamic float64 scalars).  The center is a control;
    on a periodic metric the nearest-image displacement keeps the source seam
    behaviour (force jump on the seam ``|d| = P/2``), reported by the
    evaluation margin.
    """

    center: Array
    stiffness: Array
    dimension: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    history: AtomisticBiasHistoryPlan | None
    stiffness_minimum_eigenvalue: Array
    stiffness_condition: Array
    form_identity: IdentityWords

    def __init__(self, center: ArrayLike, stiffness: ArrayLike, /) -> None:
        center_ = np.asarray(center, dtype=np.float64).reshape((-1,))
        dimension = center_.size
        if dimension == 0 or np.any(~np.isfinite(center_)):
            raise ValueError("Umbrella centers must be finite.")
        raw = np.asarray(stiffness, dtype=np.float64)
        if raw.ndim == 2:
            matrix = raw
        else:
            matrix = np.diag(_vector(raw, dimension, "stiffness"))
        if matrix.shape != (dimension, dimension) or np.any(~np.isfinite(matrix)):
            raise ValueError("Umbrella stiffness must be scalar, (d,) or (d, d).")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=0.0):
            raise ValueError("A full umbrella stiffness must be symmetric.")
        spectrum = HermitianSpectrum(jnp.asarray(matrix))
        minimum = float(np.asarray(spectrum.minimum_eigenvalue))
        scale = max(float(np.max(np.abs(np.asarray(spectrum.eigenvalues)))), 1.0e-300)
        if minimum < -1.0e-12 * scale or not bool(np.asarray(spectrum.valid)):
            raise ValueError("Umbrella stiffness must be positive semidefinite.")
        self.center = jnp.asarray(center_)
        self.stiffness = jnp.asarray(matrix)
        self.dimension = dimension
        self.control_size = dimension
        self.history = None
        self.stiffness_minimum_eigenvalue = jnp.asarray(minimum, dtype=jnp.float64)
        self.stiffness_condition = jnp.asarray(
            float(np.asarray(spectrum.condition_number)), dtype=jnp.float64
        )
        self.form_identity = IdentityWords.of(
            {
                "kind": "harmonic-bias",
                "center": array_tree_fingerprint(center_),
                "stiffness": array_tree_fingerprint(matrix),
            }
        )

    def initial_controls(self) -> Array:
        return self.center

    def evaluate(
        self,
        values: Array,
        metrics: tuple[CollectiveVariableMetric, ...],
        controls: Array,
        history: AtomisticBiasHistoryState | None,
        owner_key: Array,
        /,
    ) -> BiasFormEvaluation:
        del history, owner_key
        displacement = jnp.stack(
            tuple(
                metric.difference(values[index], controls[index])
                for index, metric in enumerate(metrics)
            )
        )
        stiffness = self.stiffness.astype(displacement.dtype)
        energy = 0.5 * displacement @ stiffness @ displacement
        return BiasFormEvaluation.without_geometry(energy, True)


class WallSide(StrEnum):
    LOWER = "lower"
    UPPER = "upper"


def _branch_values(
    values: Array,
    metrics: tuple[CollectiveVariableMetric, ...],
    branch_start: Array | None,
    /,
) -> Array:
    mapped = []
    for index, metric in enumerate(metrics):
        value = values[index]
        if metric.period is not None:
            if branch_start is None:
                raise ValueError("A periodic wall requires an explicit branch interval.")
            start = branch_start[index].astype(value.dtype)
            value = start + jnp.mod(value - start, metric.period)
        mapped.append(value)
    return jnp.stack(mapped)


def _branch_start(branch_start: ArrayLike | None, dimension: int, /) -> Array | None:
    if branch_start is None:
        return None
    return jnp.asarray(_vector(branch_start, dimension, "branch_start"))


class FlatBottomBiasForm(AbstractBiasForm):
    """``(k/p) sum (dist(x, [lower, upper]))^p`` with ``p >= 2`` (C^{p-1}).

    Periodic CVs are first mapped to the declared branch
    ``[branch_start, branch_start + period)``; without one they are refused.
    The real exponent ``p`` is a physical parameter, a dynamic float64 scalar.
    """

    lower: Array
    upper: Array
    stiffness: Array
    branch_start: Array | None
    exponent: Array
    dimension: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    history: AtomisticBiasHistoryPlan | None
    form_identity: IdentityWords

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        stiffness: ArrayLike,
        /,
        *,
        exponent: float = 2.0,
        branch_start: ArrayLike | None = None,
    ) -> None:
        low = np.asarray(lower, dtype=np.float64).reshape((-1,))
        dimension = low.size
        high = _vector(upper, dimension, "upper")
        k = _vector(stiffness, dimension, "stiffness")
        power = float(exponent)
        if dimension == 0 or np.any(high < low) or np.any(k <= 0.0) or not power >= 2.0:
            raise ValueError("Flat-bottom bounds, positive stiffness and exponent >= 2.")
        self.lower = jnp.asarray(low)
        self.upper = jnp.asarray(high)
        self.stiffness = jnp.asarray(k)
        self.branch_start = _branch_start(branch_start, dimension)
        self.exponent = jnp.asarray(power, dtype=jnp.float64)
        self.dimension = dimension
        self.control_size = 0
        self.history = None
        self.form_identity = IdentityWords.of(
            {
                "kind": "flat-bottom-bias",
                "arrays": array_tree_fingerprint({"lower": low, "upper": high, "k": k}),
                "exponent": power.hex(),
                "branch": None
                if branch_start is None
                else np.asarray(self.branch_start).tolist(),
            }
        )

    def evaluate(
        self,
        values: Array,
        metrics: tuple[CollectiveVariableMetric, ...],
        controls: Array,
        history: AtomisticBiasHistoryState | None,
        owner_key: Array,
        /,
    ) -> BiasFormEvaluation:
        del controls, history, owner_key
        branch = _branch_values(values, metrics, self.branch_start)
        below = jnp.maximum(self.lower.astype(branch.dtype) - branch, 0.0)
        above = jnp.maximum(branch - self.upper.astype(branch.dtype), 0.0)
        excess = below + above
        exponent = self.exponent.astype(branch.dtype)
        energy = jnp.sum(self.stiffness * excess**exponent) / exponent
        return BiasFormEvaluation.without_geometry(energy, True)


class WallBiasForm(AbstractBiasForm):
    """One-sided wall ``(k/p) max(0, +-(x - bound))^p`` on a declared branch.

    ``side`` is a static selector; the real exponent ``p`` is a dynamic
    float64 scalar.
    """

    bound: Array
    stiffness: Array
    branch_start: Array | None
    side: WallSide = eqx.field(static=True)
    exponent: Array
    dimension: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    history: AtomisticBiasHistoryPlan | None
    form_identity: IdentityWords

    def __init__(
        self,
        bound: ArrayLike,
        stiffness: ArrayLike,
        side: WallSide,
        /,
        *,
        exponent: float = 2.0,
        branch_start: ArrayLike | None = None,
    ) -> None:
        limit = np.asarray(bound, dtype=np.float64).reshape((-1,))
        dimension = limit.size
        k = _vector(stiffness, dimension, "stiffness")
        power = float(exponent)
        if not isinstance(side, WallSide):
            raise TypeError("side must be WallSide.")
        if (
            dimension == 0
            or np.any(~np.isfinite(limit))
            or np.any(k <= 0.0)
            or not power >= 2.0
        ):
            raise ValueError("Wall bound, positive stiffness and exponent >= 2.")
        self.bound = jnp.asarray(limit)
        self.stiffness = jnp.asarray(k)
        self.branch_start = _branch_start(branch_start, dimension)
        self.side = side
        self.exponent = jnp.asarray(power, dtype=jnp.float64)
        self.dimension = dimension
        self.control_size = 0
        self.history = None
        self.form_identity = IdentityWords.of(
            {
                "kind": "wall-bias",
                "side": side.value,
                "arrays": array_tree_fingerprint({"bound": limit, "k": k}),
                "exponent": power.hex(),
                "branch": None
                if branch_start is None
                else np.asarray(self.branch_start).tolist(),
            }
        )

    def evaluate(
        self,
        values: Array,
        metrics: tuple[CollectiveVariableMetric, ...],
        controls: Array,
        history: AtomisticBiasHistoryState | None,
        owner_key: Array,
        /,
    ) -> BiasFormEvaluation:
        del controls, history, owner_key
        branch = _branch_values(values, metrics, self.branch_start)
        offset = branch - self.bound.astype(branch.dtype)
        excess = jnp.maximum(offset if self.side is WallSide.UPPER else -offset, 0.0)
        exponent = self.exponent.astype(branch.dtype)
        energy = jnp.sum(self.stiffness * excess**exponent) / exponent
        return BiasFormEvaluation.without_geometry(energy, True)


class HillHistoryBiasForm(AbstractBiasForm):
    """Adaptive kernel history (metadynamics, well-tempered, RMSD history)."""

    dimension: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    history: AtomisticBiasHistoryPlan | None
    form_identity: IdentityWords

    def __init__(self, history: AtomisticBiasHistoryPlan, /) -> None:
        if not isinstance(history, AtomisticBiasHistoryPlan):
            raise TypeError("history must be AtomisticBiasHistoryPlan.")
        self.dimension = history.dimension
        self.control_size = 0
        self.history = history
        self.form_identity = IdentityWords.of(
            {"kind": "hill-history-bias", "history": history.scientific_id}
        )

    def evaluate(
        self,
        values: Array,
        metrics: tuple[CollectiveVariableMetric, ...],
        controls: Array,
        history: AtomisticBiasHistoryState | None,
        owner_key: Array,
        /,
    ) -> BiasFormEvaluation:
        """History energy; an aligned kernel's single superposition per active
        kernel supplies its regularity, status and eigengap evidence."""
        del controls
        plan = self.history
        if plan is None or history is None:
            raise ValueError("A hill-history bias requires its bound history state.")
        evaluation = plan.evaluate(values, metrics, history, owner_key)
        return BiasFormEvaluation(
            energy=evaluation.energy,
            successful=evaluation.successful,
            status=evaluation.status,
            minimum_relative_gap=evaluation.minimum_relative_gap,
        )


class IntegratedProfileBiasForm(AbstractBiasForm):
    """Conservative ``V_bias = -A`` of a frozen, integrated ABF profile.

    One CV uses cubic Hermite interpolation of ``-A`` with the measured mean
    force as node slopes; several CVs use native tensor-product multilinear
    interpolation and require full-grid support.  A periodic CV must close
    its period (``profile.periodic_axes``) and be supported on every bin:
    queries are reduced to the period and the interval between the last and
    first node is interpolated through the seam, so energy and force are
    continuous across the representation seam.  A periodic CV whose profile
    covers only part of the period, or has unsupported bins, is refused.  On
    a nonperiodic axis the energy is held constant outside the node range
    (no bias force there).  An outer node with a nonzero slope therefore has
    unequal one-sided forces; there is no unique derivative at that kink.
    This is conservative only because the profile is frozen.  ``periods`` are
    the static per-axis periods (``None`` on nonperiodic axes) and must equal
    the evaluating CV metrics.  The source
    profile digest is the dynamic ``profile_identity`` (``profile_id`` on the
    host).
    """

    nodes: tuple[Array, ...]
    values: Array
    slopes: Array | None
    periods: tuple[float | None, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    history: AtomisticBiasHistoryPlan | None
    profile_identity: IdentityWords
    form_identity: IdentityWords

    def __init__(self, profile: AdaptiveProfileResult, /) -> None:
        if (
            profile.estimator is not AdaptiveProfileEstimator.ABF
            or profile.gradient is None
        ):
            raise ValueError("The integrated realization consumes an ABF profile.")
        if not profile.formula_admitted:
            raise ValueError(
                f"The ABF profile is not admitted for integration: {profile.refusals}."
            )
        support = np.asarray(profile.support)
        free_energy = np.asarray(profile.free_energy)
        gradient = np.asarray(profile.gradient)
        dimension = len(profile.nodes)
        if len(profile.metrics) != dimension or len(profile.periodic_axes) != dimension:
            raise ValueError("The ABF profile metrics must align with its node axes.")
        periods: list[float | None] = []
        for axis, metric in enumerate(profile.metrics):
            if metric.period is None:
                periods.append(None)
            elif profile.periodic_axes[axis]:
                periods.append(float(metric.period))
            else:
                raise ValueError(
                    "A periodic CV profile must close its period (ABF range equal to "
                    "the period); a partial periodic range has no seam-continuous "
                    "realization."
                )
        if any(period is not None for period in periods) and not np.all(support):
            raise ValueError(
                "A periodic integrated realization requires support on every bin of "
                "the period; sample the unsupported bins first."
            )
        if dimension == 1:
            node = np.asarray(profile.nodes[0])
            period = periods[0]
            if period is not None:
                # Close the period with the first node one period on: the
                # wrapping interval is interpolated with the same Hermite data.
                nodes = (np.append(node, node[0] + period),)
                values = -np.append(free_energy, free_energy[0])
                slopes = -np.append(gradient[:, 0], gradient[0, 0])
            else:
                kept = np.flatnonzero(support)
                if kept.size < 2:
                    raise ValueError(
                        "A 1-D integrated profile needs two supported nodes."
                    )
                nodes = (node[kept],)
                values = -free_energy[kept]
                slopes = -gradient[kept, 0]
        else:
            if not np.all(support):
                raise ValueError(
                    "A multi-CV integrated realization requires full-grid support; "
                    "sample the unsupported bins first."
                )
            nodes = tuple(np.asarray(node) for node in profile.nodes)
            values = -free_energy
            slopes = None
        self.nodes = tuple(jnp.asarray(node) for node in nodes)
        self.values = jnp.asarray(values)
        self.slopes = None if slopes is None else jnp.asarray(slopes)
        self.periods = tuple(periods)
        self.dimension = dimension
        self.control_size = 0
        self.history = None
        self.profile_identity = IdentityWords.from_fingerprint(profile.profile_id)
        self.form_identity = IdentityWords.of(
            {"kind": "integrated-abf-bias", "profile": profile.profile_id}
        )

    @property
    def profile_id(self) -> str:
        """Host decoding of the integrated ABF profile digest."""
        return self.profile_identity.fingerprint()

    def evaluate(
        self,
        values: Array,
        metrics: tuple[CollectiveVariableMetric, ...],
        controls: Array,
        history: AtomisticBiasHistoryState | None,
        owner_key: Array,
        /,
    ) -> BiasFormEvaluation:
        del controls, history, owner_key
        if len(metrics) != self.dimension or any(
            metric.period != period
            for metric, period in zip(metrics, self.periods, strict=True)
        ):
            raise ValueError("The evaluating CV metrics must have the profile's periods.")
        point = jnp.asarray(values)
        dtype = point.dtype
        finite = jnp.all(jnp.isfinite(point))
        # A failed CV value is refused by status; interpolation sees a node.
        origin = jnp.stack(tuple(node[0] for node in self.nodes)).astype(dtype)
        point = jnp.where(finite, point, origin)
        if self.slopes is not None:
            node = self.nodes[0].astype(dtype)
            period = self.periods[0]
            query = (
                jnp.clip(point[0], node[0], node[-1])
                if period is None
                else node[0] + jnp.mod(point[0] - node[0], period)
            )
            result = cubic_hermite_interpolate(
                node,
                self.values.astype(dtype),
                query,
                slopes=self.slopes.astype(dtype),
            )
            energy = jnp.asarray(result.values).reshape(())
            supported = jnp.all(result.support)
        else:
            stencil = rectilinear_stencil(
                tuple(node.astype(dtype) for node in self.nodes),
                point,
                boundary=tuple(
                    "clamp" if period is None else "periodic" for period in self.periods
                ),
                periods=self.periods,
            )
            result = apply_gather_stencil(
                self.values.astype(dtype).reshape((-1,)), stencil
            )
            energy = jnp.asarray(result.values).reshape(())
            supported = jnp.all(result.support)
        successful = finite & supported
        return BiasFormEvaluation.without_geometry(
            jnp.where(successful, energy, jnp.asarray(jnp.nan, dtype=dtype)), successful
        )


class AtomisticBiasTermState(AbstractAtomisticTermState):
    """Read-only state of one bias term.

    ``controls`` (umbrella/moving centers), ``history`` view, ``owner_key``
    (history owner evaluated) and ``coupling`` (scalar multiplying the form
    energy; ``0`` realizes an unbiased analysis row of the same term).
    ``owner_identity`` is the owning term's exact ``term_identity`` (dynamic
    words); it replaces the former static ``term_id`` field.
    """

    __strict_contract__ = True

    controls: Array
    history: AtomisticBiasHistoryState | None
    owner_key: Array
    coupling: Array
    revision: Int32[Scalar]
    successful: Bool[Scalar]
    owner_identity: IdentityWords


class AtomisticBiasEvidence(StrictModule):
    """CV values, status words and branch margins behind one bias energy.

    ``regular`` is the CV program's regularity; ``form_status``,
    ``form_successful`` and ``alignment_relative_gap`` are the form's own
    admission evidence (:class:`BiasFormEvaluation`), e.g. the alignment
    status and smallest eigengap of an aligned configuration-kernel history's
    active kernels (``alignment_relative_gap`` is ``None`` for forms that
    align nothing).
    """

    values: Array
    cv_status: Array
    branch_margins: Array
    regular: Array
    form_status: Array
    form_successful: Array
    alignment_relative_gap: Array | None


class AtomisticBiasTerm(AbstractAtomisticEnergyTerm):
    """Conservative CV bias term of the scalar program.

    ``variables`` is a CV program prepared on the program's system; ``form``
    the bias form.  ``center_rate`` declares a moving restraint whose center
    control advances at committed boundaries (the protocol owner books the
    resulting work); it belongs only to a harmonic form.  ``name`` is the
    static program/state slot; the exact term digest is the dynamic
    ``term_identity`` (``term_id`` decodes it on the host).
    """

    variables: AbstractCollectiveVariableProgram
    form: AbstractBiasForm
    center_rate: Array | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        variables: AbstractCollectiveVariableProgram,
        form: AbstractBiasForm,
        /,
        *,
        name: str,
        force_group: int,
        center_rate: ArrayLike | None = None,
    ) -> None:
        if not isinstance(variables, AbstractCollectiveVariableProgram):
            raise TypeError("variables must be a prepared collective-variable program.")
        if not isinstance(form, AbstractBiasForm):
            raise TypeError("form must be an AbstractBiasForm.")
        if form.dimension != variables.output_size:
            raise ValueError("The bias form dimension must match the CV program.")
        label = str(name).strip()
        if not label:
            raise ValueError("A bias term requires a name.")
        rate = None
        if center_rate is not None:
            if not isinstance(form, HarmonicBiasForm):
                raise ValueError("Only a harmonic restraint has a moving center.")
            rate = jnp.asarray(_vector(center_rate, form.dimension, "center_rate"))
        self.variables = variables
        self.form = form
        self.center_rate = rate
        self.name = label
        self.force_group = int(force_group)
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=False,
        )
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=variables.requirements,
            state_required=self.stateful,
        )
        self.term_identity = IdentityWords.of(
            {
                "kind": "atomistic-bias-term",
                "name": label,
                "force_group": self.force_group,
                "variables": variables.program_id,
                "form": form.form_id,
                "center_rate": None if rate is None else np.asarray(rate).tolist(),
            }
        )

    @property
    def stateful(self) -> bool:
        return self.form.control_size > 0 or self.form.history is not None

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedAtomisticBiasTerm":
        if self.variables.system_id != system.prepared_id:
            raise ValueError("The bias CVs belong to another prepared system.")
        return PreparedAtomisticBiasTerm(self, system)


class PreparedAtomisticBiasTerm(AbstractPreparedAtomisticEnergyTerm):
    """Bias term bound to one prepared system.

    ``term_identity`` copies the plan's words; ``prepared_identity`` is the
    exact prepared-term digest (term, history capacity identity, system).
    Bound states are validated by slot (``name``), static layout, and in
    traced code by ``owner_identity.equals(term_identity)`` folded into the
    evaluation status.
    """

    plan: AtomisticBiasTerm
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: AtomisticBiasTerm, system: PreparedAtomisticSystem, /
    ) -> None:
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        history = plan.form.history
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-bias-term",
                "term": plan.term_id,
                "history_capacity": None if history is None else history.history_id,
                "system": system.prepared_id,
            }
        )

    def initial_term_state(self, dtype: DTypeLike, /) -> AtomisticBiasTermState | None:
        if not self.plan.stateful:
            return None
        history = self.plan.form.history
        return AtomisticBiasTermState(
            controls=jnp.asarray(self.plan.form.initial_controls(), dtype=dtype),
            history=None if history is None else history.initialize(dtype),
            owner_key=jnp.zeros((), dtype=jnp.int32),
            coupling=jnp.ones((), dtype=dtype),
            revision=jnp.zeros((), dtype=jnp.int32),
            successful=jnp.asarray(True),
            owner_identity=self.term_identity,
        )

    def require_state(
        self, state: AbstractAtomisticTermState, /
    ) -> AtomisticBiasTermState:
        """Validate the static layout of a bound state.

        The exact owner/history identities are dynamic words: :meth:`bias`
        folds ``owner_identity.equals(term_identity)`` into its status and a
        history of another plan evaluates to a non-finite (failed) energy.
        """
        history = self.plan.form.history
        if not isinstance(state, AtomisticBiasTermState):
            raise TypeError("The bias term requires its own AtomisticBiasTermState.")
        if (history is None) != (state.history is None) or (
            history is not None
            and state.history is not None
            and state.history.centers.shape != (history.capacity, history.dimension)
        ):
            raise ValueError("The bound history does not match the history layout.")
        if state.controls.shape != (self.plan.form.control_size,):
            raise ValueError("Bias controls do not match the form.")
        return state

    def _evaluate(
        self,
        geometry: CollectiveVariableGeometry,
        state: AbstractAtomisticTermState | None,
        /,
    ) -> tuple[Array, CollectiveVariableProgramEvaluation, BiasFormEvaluation, Array]:
        evaluation = self.plan.variables.evaluate(geometry)
        form = self.plan.form
        if self.plan.stateful:
            if state is None:
                raise ValueError("A stateful bias requires its term state.")
            bound = self.require_state(state)
            controls, history, owner = bound.controls, bound.history, bound.owner_key
            coupling = bound.coupling
            state_ok = bound.successful & bound.owner_identity.equals(self.term_identity)
        else:
            controls = form.initial_controls()
            history, owner = None, jnp.zeros((), dtype=jnp.int32)
            coupling = jnp.ones((), dtype=evaluation.values.dtype)
            state_ok = jnp.asarray(True)
        form_evaluation = form.evaluate(
            evaluation.values,
            self.plan.variables.metrics,
            controls.astype(evaluation.values.dtype),
            history,
            owner,
        )
        energy = coupling.astype(form_evaluation.energy.dtype) * form_evaluation.energy
        successful = (
            evaluation.successful
            & evaluation.regular
            & form_evaluation.successful
            & state_ok
            & jnp.isfinite(energy)
        )
        return energy, evaluation, form_evaluation, successful

    def bias(
        self,
        geometry: CollectiveVariableGeometry,
        state: AbstractAtomisticTermState | None,
        /,
    ) -> tuple[Array, CollectiveVariableProgramEvaluation, Array]:
        """``(energy, CV evaluation, successful)`` at one geometry and state.

        A stateful bias validates ``state`` with :meth:`require_state`.
        ``successful`` also requires the form's own admission (for example
        the alignment regularity of every active aligned history kernel).
        """
        energy, evaluation, _, successful = self._evaluate(geometry, state)
        return energy, evaluation, successful

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        state = (
            self.require_state(context.term_states[self.name])
            if self.plan.stateful
            else None
        )
        energy, evaluation, form_evaluation, successful = self._evaluate(
            CollectiveVariableGeometry.from_context(context), state
        )
        gap = form_evaluation.minimum_relative_gap
        return AtomisticTermEvaluation(
            jnp.where(successful, energy, jnp.nan),
            jnp.zeros((self.system.capacity,), dtype=energy.dtype),
            successful,
            atom_energy_complete=False,
            evidence=(
                AtomisticBiasEvidence(
                    values=jax.lax.stop_gradient(evaluation.values),
                    cv_status=evaluation.status,
                    branch_margins=jax.lax.stop_gradient(evaluation.branch_margins),
                    regular=evaluation.regular,
                    form_status=form_evaluation.status,
                    form_successful=form_evaluation.successful,
                    alignment_relative_gap=None
                    if gap is None
                    else jax.lax.stop_gradient(gap),
                ),
            ),
        )


def with_bias_controls(
    state: AtomisticBiasTermState,
    controls: ArrayLike,
    /,
    *,
    owner_key: ArrayLike | None = None,
    coupling: ArrayLike | None = None,
) -> AtomisticBiasTermState:
    """Replace controls/owner/coupling at a committed boundary; the revision advances."""

    value = jnp.asarray(controls, dtype=state.controls.dtype)
    if value.shape != state.controls.shape:
        raise ValueError("Replacement controls must keep the control layout.")
    return AtomisticBiasTermState(
        controls=value,
        history=state.history,
        owner_key=state.owner_key
        if owner_key is None
        else jnp.asarray(owner_key, dtype=jnp.int32).reshape(()),
        coupling=state.coupling
        if coupling is None
        else jnp.asarray(coupling, dtype=state.coupling.dtype).reshape(()),
        revision=state.revision + 1,
        successful=state.successful,
        owner_identity=state.owner_identity,
    )


def with_bias_history(
    state: AtomisticBiasTermState, history: AtomisticBiasHistoryState, /
) -> AtomisticBiasTermState:
    """Bind a changed history at a committed boundary; the revision advances.

    The replacement keeps the history layout (static check); its plan
    identity travels in its own words and is checked where it is evaluated.
    """

    if state.history is None or history.centers.shape != state.history.centers.shape:
        raise ValueError("The replacement history does not match the history layout.")
    return AtomisticBiasTermState(
        controls=state.controls,
        history=history,
        owner_key=state.owner_key,
        coupling=state.coupling,
        revision=state.revision + 1,
        successful=state.successful,
        owner_identity=state.owner_identity,
    )


def authenticate_frozen_bias_histories(
    row: AtomisticBiasRowBinding | None,
    term_states: AtomisticTermStateBindings,
    /,
) -> tuple[str, ...]:
    """Check every ``"frozen:<digest>"`` declaration of a row against bound content.

    Host transaction over one unbatched state: each declared history must be
    bound under the row's term slot and its :func:`bias_history_digest` must
    equal the declared digest.  A matching label alone is never accepted.
    Returns the verified digests in row term order.
    """

    if row is None:
        return ()
    digests: list[str] = []
    for slot, declaration in zip(row.term_slots, row.histories, strict=True):
        if declaration is None or declaration == "adaptive":
            continue
        state = term_states[slot]
        if not isinstance(state, AtomisticBiasTermState) or state.history is None:
            raise ValueError(
                f"Bias term {slot!r} declares a frozen history but binds none."
            )
        if state.history.heights.ndim != 1:
            raise ValueError("Frozen-history authentication reads one unbatched state.")
        digest = bias_history_digest(state.history)
        if declaration != f"frozen:{digest}":
            raise ValueError(
                f"Bias term {slot!r}: the bound history content does not "
                "authenticate its frozen declaration."
            )
        digests.append(digest)
    return tuple(digests)


def bind_bias_row(
    terms: tuple[PreparedAtomisticBiasTerm, ...],
    row: AtomisticBiasRowBinding | None,
    term_states: AtomisticTermStateBindings,
    /,
) -> AtomisticTermStateBindings:
    """Unbatched term states realizing one *fixed* thermodynamic bias row.

    Host transaction for cross-evaluation owners outside the multistate
    runtime: the row's controls and couplings replace the bound ones, every
    state- or walker-owned history evaluates the row's declared owner key
    (every history of a fixed row is frozen), and every frozen history
    declaration is authenticated against the bound content.  An
    ``"adaptive"`` row is not a fixed Hamiltonian and is refused.
    """

    if row is None:
        return term_states
    if any(declaration == "adaptive" for declaration in row.histories):
        raise ValueError(
            "An adaptive-history row is not a fixed Hamiltonian; freeze it first."
        )
    authenticate_frozen_bias_histories(row, term_states)
    by_slot = {term.name: term for term in terms}
    values = dict(zip(term_states.slots, term_states.values, strict=True))
    for slot, controls, owner, coupling in zip(
        row.term_slots, row.controls, row.owner_keys, row.couplings, strict=True
    ):
        term = by_slot.get(slot)
        if term is None or slot not in values:
            raise ValueError(f"Bias row term {slot!r} is not a bound bias term.")
        current = term.require_state(values[slot])
        if current.owner_identity.fingerprint() != term.term_id:
            raise ValueError(f"Bias row term {slot!r} binds another term's state.")
        history = term.plan.form.history
        values[slot] = with_bias_controls(
            current,
            current.controls if controls is None else controls,
            owner_key=owner
            if history is not None
            and history.ownership is not BiasHistoryOwnership.SHARED
            else None,
            coupling=coupling,
        )
    return AtomisticTermStateBindings(values)


__all__ = [
    "AbstractBiasForm",
    "AtomisticBiasEvidence",
    "AtomisticBiasTerm",
    "AtomisticBiasTermState",
    "BiasFormEvaluation",
    "FlatBottomBiasForm",
    "HarmonicBiasForm",
    "HillHistoryBiasForm",
    "IntegratedProfileBiasForm",
    "PreparedAtomisticBiasTerm",
    "WallBiasForm",
    "WallSide",
    "authenticate_frozen_bias_histories",
    "bind_bias_row",
    "with_bias_controls",
    "with_bias_history",
]

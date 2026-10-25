#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native device-resident relaxation of prepared atomistic scalar programs.

An `AtomisticRelaxationPlan` binds one prepared scalar Hamiltonian, one
coordinate/cell chart, one native force-driven optimizer (`FIRE`, `FIRE2` or
`TrustScaledLBFGS`) and one convergence policy.  A batch of ``G`` independent
relaxations of the same program executes as ``G`` optimizer groups of one
`OptimizerGroupLayout`; every group owns its chart reference, its accepted
case-one neighborhood bundle state and its accepted evaluation.

Transaction of one round (`PreparedAtomisticRelaxation.step`)
------------------------------------------------------------
1. The pending optimizer proposal of every iterating group is realized by its
   chart into *unwrapped* Cartesian positions and cell vectors.  Integer image
   counts are formed outside differentiation; positions handed to the
   resource owner are the wrapped representation of the same point.
2. The candidate resource state is refreshed purely from the group's accepted
   bundle state (`PreparedAtomisticNeighborhoodBundle.refresh`); accepted
   resources are never modified by a trial.
3. One reverse pass differentiates the scalar objective (energy, plus ``P V``
   for a variable cell) through the chart: the Cartesian covectors
   ``dE/dr`` (fixed cell) and ``dE/dH`` (fixed Cartesian positions) come from
   the same pass as the chart gradient, which is their native pullback.
4. A capacity-only resource failure is *not* consumed: the optimizer group is
   left untouched, the group is ``CAPACITY_PENDING`` and the host lifecycle
   (`grow`) selects larger declared capacities, rebases every accepted
   resource state and replays the identical pending proposal.  Scientific,
   numerical and cell failures are consumed as nonfinite evaluations, so the
   optimizer returns to its accepted iterate, and the group terminates with an
   explicit failure status.
5. Candidate resources are committed (`select_cases`) exactly for groups whose
   evaluation the optimizer accepted; every reported energy, force, stress and
   residual belongs to that accepted iterate.

Residuals
---------
The *projected* residual is the largest optimizer-site norm of the chart
gradient (chart units).  The *full* residual is the largest Cartesian force
norm over mobile atoms and, for a variable cell, the largest component of
``sigma + P I`` (pressure units).  A projected minimum of a constrained or
symmetry chart is not a full minimum unless the full residual is small too;
`AtomisticRelaxationConvergence.require_full` selects which one terminates.

Charts
------
`AtomisticCartesianChart` (fixed cell) moves unwrapped Cartesian coordinates
through optional sparse per-atom projectors or translation-gauge removal.
`AtomisticDeformationChart` adds a symmetric strain subspace realized by an
affine (``F = I + c``) or logarithmic (``F = exp(c)``, native dense matrix
exponential) deformation; the generalized cell gradient is the chart pullback
of the same enthalpy, never ``-V sigma``.  `AtomisticSymmetryChart` moves an
atomic or rigid-molecular crystal chart by its genuine retraction from a fixed
anchor; its projected residual is the `SymmetryTangentOperator` adjoint action.

Holonomic topology constraints are refused: no constrained-manifold
retraction exists for them here.
"""

from __future__ import annotations

import abc
import math
from enum import IntEnum, StrEnum
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PeriodicCell
from ..ein import contract
from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    determinant_small_linear,
    estimate_spectral_bounds,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    OperatorProperties,
    PCG,
    RHSLayout,
    SmallLinearSolvePlan,
    solve as solve_linear,
    TolerancePolicy,
)
from ..linalg._dense_svd import economy_svd
from ..linalg._matrix_functions import matrix_exponential_action, MatrixFunctionPolicy
from ..linalg.eigen import Eigenproblem, eigensolve, EigenSolvePolicy
from ..optim._bounds import TrustScaledLBFGS, TrustScaledLBFGSState
from ..optim._fire import FIRE, FIRE2, FIREState
from ..optim._group_layout import OptimizerGroupLayout
from ..optim._iterative._types import OptimizationStatus
from ..typing import checked, ConvertibleToArray
from ._cell_dynamics import bind_atomistic_bundle_geometry
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodGeometry,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential_program import (
    AbstractPreparedAtomisticHamiltonian,
    AtomisticHamiltonianEvaluation,
)
from ._system import PreparedAtomisticSystem
from .crystal._asu import AsymmetricUnitState, expand_asymmetric_unit
from .crystal._symmetry_basis import (
    atomic_tangent,
    AtomicSymmetricState,
    AtomicSymmetryPlan,
    expand_symmetric_atoms,
    retract_atomic,
    RigidSymmetryLayout,
    symmetry_residual,
    symmetry_residuals,
    SymmetryResidual,
)


_DETERMINANT = SmallLinearSolvePlan(3)
_EXPONENTIAL = MatrixFunctionPolicy("dense")
_IDENTITY_COLUMNS = RHSLayout((3,))
_SUCCESS = int(OptimizationStatus.SUCCESS)

RelaxationMethod = FIRE | FIRE2 | TrustScaledLBFGS
_RelaxationOptimizerState = FIREState | TrustScaledLBFGSState


def _step_relaxation_method(
    method: RelaxationMethod,
    state: _RelaxationOptimizerState,
    value: ArrayLike,
    gradient: PyTree[ArrayLike],
    /,
    *,
    converged: ArrayLike | None = None,
    active: ArrayLike | None = None,
) -> _RelaxationOptimizerState:
    """One native optimizer step of ``method`` on its own state type.

    FIRE and FIRE2 advance `FIREState`; `TrustScaledLBFGS` advances
    `TrustScaledLBFGSState`.  A state of the other family is refused.
    """
    if isinstance(method, TrustScaledLBFGS):
        if isinstance(state, TrustScaledLBFGSState):
            return method.step(state, value, gradient, converged=converged, active=active)
    elif isinstance(state, FIREState):
        return method.step(state, value, gradient, converged=converged, active=active)
    raise TypeError(
        f"{type(state).__name__} is not the optimizer state of {type(method).__name__}."
    )


# -- statuses ------------------------------------------------------------------


class AtomisticRelaxationStatus(IntEnum):
    """Per-group outcome of a relaxation or path group.

    ``CAPACITY_PENDING`` is the only nonterminal non-iterating status: the
    pending proposal was not consumed and is replayed after `grow`.
    """

    ITERATING = 0
    CONVERGED = 1
    EVALUATION_LIMIT = 2
    CAPACITY_PENDING = 3
    SCIENTIFIC_FAILURE = 4
    CELL_FAILURE = 5
    RESOURCE_EXHAUSTED = 6
    INACTIVE = 7


class AtomisticTrialOutcome(IntEnum):
    """What happened to the latest trial of one group."""

    NONE = 0
    ACCEPTED = 1
    ROLLED_BACK = 2
    CAPACITY = 3
    SCIENTIFIC = 4
    CELL = 5


class AtomisticCellChartKind(StrEnum):
    """Deformation realization of a variable-cell chart."""

    AFFINE = "affine"
    LOGARITHMIC = "logarithmic"


class AtomisticSensitivityStatus(IntEnum):
    """Admission outcome of a stationary-point derivative or inertia claim."""

    ADMITTED = 0
    NOT_CONVERGED = 1
    NOT_STATIONARY = 2
    GAUGE_NOT_NULL = 3
    WRONG_INERTIA = 4
    EIGEN_FAILED = 5
    SOLVE_FAILED = 6


# -- small helpers --------------------------------------------------------------


def _tree_take(tree: PyTree[Any], index: int, /) -> PyTree[Any]:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _tree_stack(trees: list[PyTree[Any]], /) -> PyTree[Any]:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


def _tree_where(mask: Array, new: PyTree[Any], old: PyTree[Any], /) -> PyTree[Any]:
    def select(left: Array, right: Array) -> Array:
        shape = mask.shape + (1,) * (jnp.ndim(left) - mask.ndim)
        return jnp.where(jnp.reshape(mask, shape), left, right)

    return jax.tree.map(select, new, old)


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _nonnegative(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return number


def _mobile(system: PreparedAtomisticSystem, /) -> np.ndarray:
    """Host mask of active mobile atoms (preparation only, never traced)."""
    return np.asarray(system.active_mask, dtype=np.bool_) & np.asarray(
        system.mobile_mask, dtype=np.bool_
    )


def _mobile_device(system: PreparedAtomisticSystem, /) -> Array:
    return jnp.asarray(system.active_mask, dtype=jnp.bool_) & jnp.asarray(
        system.mobile_mask, dtype=jnp.bool_
    )


def _tree_finite(tree: PyTree[Array], /) -> Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(tree):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def _signed_volume(vectors: Array, /) -> Array:
    return determinant_small_linear(_DETERMINANT, vectors)


def _identity_rows(values: tuple[str, ...], /) -> IdentityWords:
    """Exact canonical digests as stacked ``(L, 8)`` identity words (host).

    The row count is part of the array shape, so it specializes compilation;
    the digests themselves are dynamic leaves.
    """
    if not values:
        return IdentityWords(jnp.zeros((0, 8), dtype=jnp.uint32))
    return IdentityWords(
        jnp.stack([IdentityWords.from_fingerprint(value).words for value in values])
    )


def _identity_strings(words: IdentityWords, /) -> tuple[str, ...]:
    """Host decoding of stacked ``(L, 8)`` identity words to their exact digests."""
    return tuple(
        IdentityWords(words.words[row]).fingerprint()
        for row in range(int(words.words.shape[0]))
    )


# -- strain subspaces and coordinate constraints ---------------------------------


_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _voigt_generator(index: int, /) -> np.ndarray:
    i, j = _VOIGT[index]
    generator = np.zeros((3, 3), dtype=np.float64)
    if i == j:
        generator[i, i] = 1.0
    else:
        generator[i, j] = generator[j, i] = 1.0 / math.sqrt(2.0)
    return generator


@final
class AtomisticStrainSubspace(StrictModule, NonTrainableState):
    """Prepared symmetric strain directions of a variable-cell chart.

    ``generators`` are Frobenius-orthonormal symmetric ``3 x 3`` matrices
    ``B_k``; the chart strain is ``c = sum_k theta_k B_k / cell_factor``.  Being
    symmetric they exclude the rigid-rotation gauge of the cell.  A traceless
    subspace preserves volume exactly only under the logarithmic chart
    (``det exp(c) = exp(tr c)``), which `AtomisticDeformationChart` enforces.

    PyTree: ``generators`` (exact host ``float64``) are a dynamic leaf and the
    canonical subspace digest is dynamic `IdentityWords` (``subspace_identity``;
    ``subspace_id`` decodes it on the host).  The declared ``name``, the
    dimension (array shape) and the ``traceless`` method selector are layout.
    """

    generators: np.ndarray
    subspace_identity: IdentityWords
    name: str = eqx.field(static=True)
    traceless: bool = eqx.field(static=True)

    def __init__(self, generators: ConvertibleToArray, /, *, name: str) -> None:
        values = np.asarray(generators, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[1:] != (3, 3)
            or not 1 <= values.shape[0] <= 6
        ):
            raise ValueError("Strain generators must have shape (K, 3, 3), 1 <= K <= 6.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Strain generators must be finite.")
        if not np.allclose(values, np.swapaxes(values, 1, 2), atol=1e-12):
            raise ValueError("Strain generators must be symmetric.")
        gram = np.einsum("kij,lij->kl", values, values)
        if not np.allclose(gram, np.eye(values.shape[0]), atol=1e-10):
            raise ValueError("Strain generators must be Frobenius-orthonormal.")
        label = str(name).strip()
        if not label:
            raise ValueError("name must be non-empty.")
        self.generators = values
        self.name = label
        self.traceless = bool(np.allclose(np.trace(values, axis1=1, axis2=2), 0.0))
        self.subspace_identity = IdentityWords.of(
            {
                "kind": "atomistic-strain-subspace",
                "name": label,
                "generators": np.round(values, 15).tolist(),
            }
        )

    @property
    def subspace_id(self) -> str:
        return self.subspace_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free layout: declared name, dimension and the traceless selector."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-strain-subspace-layout",
                "name": self.name,
                "dimension": self.dimension,
                "traceless": self.traceless,
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.generators.shape[0])

    @classmethod
    def symmetric(cls) -> AtomisticStrainSubspace:
        """All six symmetric strains (Voigt order ``xx, yy, zz, yz, xz, xy``)."""
        return cls(np.stack([_voigt_generator(k) for k in range(6)]), name="symmetric")

    @classmethod
    def hydrostatic(cls) -> AtomisticStrainSubspace:
        """Isotropic dilation only."""
        return cls((np.eye(3) / math.sqrt(3.0))[None], name="hydrostatic")

    @classmethod
    def fixed_volume(cls) -> AtomisticStrainSubspace:
        """The five traceless symmetric strains."""
        first = np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0)
        second = np.diag([1.0, 1.0, -2.0]) / math.sqrt(6.0)
        shear = [_voigt_generator(k) for k in (3, 4, 5)]
        return cls(np.stack([first, second, *shear]), name="fixed-volume")

    @classmethod
    def selected(cls, components: ConvertibleToArray, /) -> AtomisticStrainSubspace:
        """The Voigt components selected by a length-six boolean mask."""
        mask = np.asarray(components, dtype=np.bool_)
        if mask.shape != (6,) or not np.any(mask):
            raise ValueError("components must be a non-empty length-six boolean mask.")
        chosen = [int(k) for k in np.flatnonzero(mask)]
        return cls(
            np.stack([_voigt_generator(k) for k in chosen]),
            name="selected-" + "".join("1" if value else "0" for value in mask),
        )


@final
class AtomisticCoordinateConstraint(StrictModule, NonTrainableState):
    """Sparse linear tangent constraints of atomic displacements.

    ``projectors`` are per-atom orthogonal projectors ``P_a`` (``(N, 3, 3)``,
    prepared once): a displacement ``d_a`` is realized as ``P_a d_a`` and the
    chart gradient is ``P_a g_a``, a block-diagonal action with no dense
    atom-by-parameter basis.  ``remove_translation`` realizes ``d - mean(d)``
    over mobile atoms, which removes the uniform-translation gauge of the
    chart.  The two are not combined: a mean shift does not respect general
    projector constraints.

    PyTree: ``projectors`` (exact host ``float64``) are a dynamic leaf and the
    canonical constraint digest is dynamic `IdentityWords`
    (``constraint_identity``; ``constraint_id`` decodes it on the host).
    Projector presence/shape and ``remove_translation`` are layout.
    """

    projectors: np.ndarray | None
    constraint_identity: IdentityWords
    remove_translation: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        projectors: ConvertibleToArray | None = None,
        remove_translation: bool = False,
    ) -> None:
        if not isinstance(remove_translation, bool):
            raise TypeError("remove_translation must be a bool.")
        values = None
        if projectors is not None:
            values = np.asarray(projectors, dtype=np.float64)
            if values.ndim != 3 or values.shape[1:] != (3, 3):
                raise ValueError("projectors must have shape (N, 3, 3).")
            if not np.all(np.isfinite(values)):
                raise ValueError("projectors must be finite.")
            if not np.allclose(values, np.swapaxes(values, 1, 2), atol=1e-12):
                raise ValueError("projectors must be symmetric.")
            if not np.allclose(values @ values, values, atol=1e-10):
                raise ValueError("projectors must be idempotent.")
        if values is not None and remove_translation:
            raise ValueError(
                "Translation removal and per-atom projectors are separate contracts."
            )
        if values is None and not remove_translation:
            raise ValueError("A coordinate constraint must constrain something.")
        self.projectors = values
        self.remove_translation = remove_translation
        self.constraint_identity = IdentityWords.of(
            {
                "kind": "atomistic-coordinate-constraint",
                "projectors": None if values is None else np.round(values, 15).tolist(),
                "remove_translation": remove_translation,
            }
        )

    @property
    def constraint_id(self) -> str:
        return self.constraint_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free layout: projector extent and the translation selector."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-coordinate-constraint-layout",
                "projectors": None
                if self.projectors is None
                else list(self.projectors.shape),
                "remove_translation": self.remove_translation,
            }
        )

    @classmethod
    def fixed_components(
        cls, fixed: ConvertibleToArray, /
    ) -> AtomisticCoordinateConstraint:
        """Hold the selected Cartesian components ``fixed[(N, 3)]``."""
        mask = np.asarray(fixed, dtype=np.bool_)
        if mask.ndim != 2 or mask.shape[1] != 3:
            raise ValueError("fixed must be a boolean (N, 3) mask.")
        projectors = np.zeros((mask.shape[0], 3, 3), dtype=np.float64)
        for axis in range(3):
            projectors[:, axis, axis] = (~mask[:, axis]).astype(np.float64)
        return cls(projectors=projectors)

    def admit(self, system: PreparedAtomisticSystem, /) -> None:
        if self.projectors is not None and self.projectors.shape[0] != system.capacity:
            raise ValueError("Constraint projectors must cover the system capacity.")

    def apply(self, displacement: Array, mobile: Array, /) -> Array:
        moved = jnp.where(mobile[:, None], displacement, 0.0)
        if self.projectors is not None:
            moved = contract(
                "aij,aj->ai", jnp.asarray(self.projectors, dtype=moved.dtype), moved
            )
        if self.remove_translation:
            weight = mobile.astype(moved.dtype)
            mean = jnp.sum(weight[:, None] * moved, axis=0) / jnp.maximum(
                jnp.sum(weight), 1.0
            )
            moved = jnp.where(mobile[:, None], moved - mean, 0.0)
        return moved


def _constrained(
    constraint: AtomisticCoordinateConstraint | None,
    displacement: Array,
    mobile: Array,
    /,
) -> Array:
    if constraint is None:
        return jnp.where(mobile[:, None], displacement, 0.0)
    return constraint.apply(displacement, mobile)


# -- charts -------------------------------------------------------------------------


class AbstractAtomisticRelaxationChart(StrictModule, NonTrainableState):
    """Smooth map from optimizer parameters to unwrapped positions and cell rows.

    ``realize(parameters, reference)`` is differentiated by the runtime: the
    chart gradient is the reverse-mode pullback of the Cartesian covectors, so
    every chart gets its exact generalized gradient from one evaluation.
    Parameters and references are single-group PyTrees; the runtime stacks a
    leading group axis.  ``site_masks`` gives, per parameter leaf, the
    optimizer-site activity whose shape is a leading prefix of the leaf shape.

    The canonical chart digest is dynamic `IdentityWords` (``chart_identity``;
    the base ``chart_id`` decodes it on the host); ``layout_id`` is the
    content-free chart layout (kind, selectors and extents).
    """

    chart_identity: eqx.AbstractVar[IdentityWords]

    @property
    def chart_id(self) -> str:
        return self.chart_identity.fingerprint()

    @property
    @abc.abstractmethod
    def layout_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def variable_cell(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def admit(self, system: PreparedAtomisticSystem, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def site_masks(self, system: PreparedAtomisticSystem, /) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def start(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        positions: Array | None,
        cell_vectors: Array | None,
        chart_state: Any,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def realize(
        self, parameters: dict[str, Array], reference: dict[str, Any], /
    ) -> tuple[tuple[Array, Array | None], Array]:
        raise NotImplementedError


def _cartesian_reference(
    system: PreparedAtomisticSystem,
    positions: Array | None,
    cell_vectors: Array | None,
    /,
    *,
    variable_cell: bool,
) -> dict[str, Any]:
    dtype = jnp.dtype(system.plan.coordinate_dtype)
    if positions is None:
        raise ValueError("Cartesian charts require starting positions.")
    value = jnp.asarray(positions, dtype=dtype)
    if value.shape != (system.capacity, 3):
        raise ValueError(f"positions must have shape {(system.capacity, 3)}.")
    active = jnp.asarray(system.active_mask, dtype=jnp.bool_)
    reference: dict[str, Any] = {
        # Inactive capacity slots carry no coordinates; zero keeps them finite.
        "positions": jnp.where(active[:, None], value, 0.0),
        "mobile": jnp.asarray(_mobile(system)),
    }
    cell = system.cell
    if cell is None:
        if cell_vectors is not None:
            raise ValueError("A finite system takes no cell vectors.")
        if variable_cell:
            raise ValueError("A variable-cell chart requires a periodic system.")
        return reference
    vectors = (
        cell.vectors.astype(dtype)
        if cell_vectors is None
        else jnp.asarray(cell_vectors, dtype=dtype)
    )
    if vectors.shape != cell.vectors.shape:
        raise ValueError(f"cell_vectors must have shape {cell.vectors.shape}.")
    reference["cell"] = vectors
    return reference


@final
class AtomisticCartesianChart(AbstractAtomisticRelaxationChart):
    """Fixed-cell chart over unwrapped Cartesian displacements of mobile atoms.

    Parameters are ``{"atoms": d (N, 3)}`` with ``r = r0 + C d``; ``C`` is the
    optional `AtomisticCoordinateConstraint`.  Immobile and inactive atoms are
    inactive optimizer sites and never move.  Wrapping never touches ``d``, so
    optimizer history is unaffected by boundary crossings.

    PyTree: the chart digest is dynamic ``chart_identity``; constraint
    presence and layout are static.
    """

    constraint: AtomisticCoordinateConstraint | None
    chart_identity: IdentityWords

    def __init__(
        self, *, constraint: AtomisticCoordinateConstraint | None = None
    ) -> None:
        if constraint is not None and not isinstance(
            constraint, AtomisticCoordinateConstraint
        ):
            raise TypeError(
                "constraint must be an AtomisticCoordinateConstraint or None."
            )
        self.constraint = constraint
        self.chart_identity = IdentityWords.of(
            {
                "kind": "atomistic-cartesian-chart",
                "constraint": None if constraint is None else constraint.constraint_id,
            }
        )

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "atomistic-cartesian-chart-layout",
                "constraint": None
                if self.constraint is None
                else self.constraint.layout_id,
            }
        )

    @property
    def variable_cell(self) -> bool:
        return False

    def admit(self, system: PreparedAtomisticSystem, /) -> None:
        if self.constraint is not None:
            self.constraint.admit(system)

    def site_masks(self, system: PreparedAtomisticSystem, /) -> dict[str, np.ndarray]:
        return {"atoms": _mobile(system)}

    def start(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        positions: Array | None,
        cell_vectors: Array | None,
        chart_state: Any,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        if chart_state is not None:
            raise ValueError("A Cartesian chart takes no chart state.")
        reference = _cartesian_reference(
            system, positions, cell_vectors, variable_cell=False
        )
        return {"atoms": jnp.zeros_like(reference["positions"])}, reference

    def realize(
        self, parameters: dict[str, Array], reference: dict[str, Any], /
    ) -> tuple[tuple[Array, Array | None], Array]:
        moved = _constrained(self.constraint, parameters["atoms"], reference["mobile"])
        positions = reference["positions"] + moved
        return (positions, reference.get("cell")), jnp.asarray(True)


@final
class AtomisticDeformationChart(AbstractAtomisticRelaxationChart):
    """Variable-cell chart: reference-frame displacements plus a strain subspace.

    Parameters are ``{"atoms": d (N, 3), "strain": theta (K,)}``.  With
    ``c = sum_k theta_k B_k / cell_factor`` the deformation is ``F = I + c``
    (affine) or ``F = exp(c)`` (logarithmic, native dense matrix exponential);
    positions are ``(r0 + C d) F^T`` and cell rows ``H0 F^T``.  The cell
    gradient is the reverse-mode pullback of the enthalpy through ``F``, the
    exact generalized force of this chart at every point.  ``cell_factor``
    scales the strain coordinates against atomic ones (``None`` uses the
    number of mobile atoms of the bound system).  A fully periodic rank-three
    cell is required.

    PyTree: ``cell_factor`` is a dynamic exact-``float64`` leaf (its absence
    is a static selector) and the chart digest is dynamic ``chart_identity``;
    the deformation kind, strain/constraint layouts are static.
    """

    kind: AtomisticCellChartKind = eqx.field(static=True)
    strain: AtomisticStrainSubspace
    constraint: AtomisticCoordinateConstraint | None
    cell_factor: np.ndarray | None
    chart_identity: IdentityWords

    def __init__(
        self,
        kind: AtomisticCellChartKind | str = AtomisticCellChartKind.LOGARITHMIC,
        /,
        *,
        strain: AtomisticStrainSubspace | None = None,
        constraint: AtomisticCoordinateConstraint | None = None,
        cell_factor: float | None = None,
    ) -> None:
        kind_ = AtomisticCellChartKind(kind)
        strain_ = AtomisticStrainSubspace.symmetric() if strain is None else strain
        if not isinstance(strain_, AtomisticStrainSubspace):
            raise TypeError("strain must be an AtomisticStrainSubspace or None.")
        if strain_.traceless and kind_ is AtomisticCellChartKind.AFFINE:
            raise ValueError(
                "A traceless (fixed-volume) subspace preserves volume only under "
                "the logarithmic chart."
            )
        if constraint is not None and not isinstance(
            constraint, AtomisticCoordinateConstraint
        ):
            raise TypeError(
                "constraint must be an AtomisticCoordinateConstraint or None."
            )
        factor = None if cell_factor is None else _positive(cell_factor, "cell_factor")
        self.kind = kind_
        self.strain = strain_
        self.constraint = constraint
        self.cell_factor = (
            None if factor is None else np.asarray(factor, dtype=np.float64)
        )
        self.chart_identity = IdentityWords.of(
            {
                "kind": "atomistic-deformation-chart",
                "deformation": kind_.value,
                "strain": strain_.subspace_id,
                "constraint": None if constraint is None else constraint.constraint_id,
                "cell_factor": factor,
            }
        )

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "atomistic-deformation-chart-layout",
                "deformation": self.kind.value,
                "strain": self.strain.layout_id,
                "constraint": None
                if self.constraint is None
                else self.constraint.layout_id,
                "cell_factor": self.cell_factor is not None,
            }
        )

    @property
    def variable_cell(self) -> bool:
        return True

    def admit(self, system: PreparedAtomisticSystem, /) -> None:
        cell = system.cell
        if cell is None or cell.rank != 3 or not cell.fully_periodic:
            raise ValueError("A variable-cell chart requires a fully periodic 3D cell.")
        if self.constraint is not None:
            self.constraint.admit(system)

    def site_masks(self, system: PreparedAtomisticSystem, /) -> dict[str, np.ndarray]:
        return {"atoms": _mobile(system), "strain": np.asarray(True)}

    def start(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        positions: Array | None,
        cell_vectors: Array | None,
        chart_state: Any,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        if chart_state is not None:
            raise ValueError("A deformation chart takes no chart state.")
        reference = _cartesian_reference(
            system, positions, cell_vectors, variable_cell=True
        )
        factor = (
            float(max(int(np.count_nonzero(_mobile(system))), 1))
            if self.cell_factor is None
            else self.cell_factor
        )
        dtype = reference["positions"].dtype
        reference["cell_factor"] = jnp.asarray(factor, dtype=dtype)
        reference["orientation"] = jnp.sign(_signed_volume(reference["cell"]))
        parameters = {
            "atoms": jnp.zeros_like(reference["positions"]),
            "strain": jnp.zeros((self.strain.dimension,), dtype=dtype),
        }
        return parameters, reference

    def deformation(self, strain: Array, cell_factor: Array, /) -> tuple[Array, Array]:
        """Return ``F`` and its realization status for chart strain coordinates."""
        generators = jnp.asarray(self.strain.generators, dtype=strain.dtype)
        logarithm = contract("k,kij->ij", strain / cell_factor, generators)
        identity = jnp.eye(3, dtype=strain.dtype)
        if self.kind is AtomisticCellChartKind.AFFINE:
            gradient = identity + logarithm
            return gradient, _signed_volume(gradient) > 0.0
        result = matrix_exponential_action(
            DenseLinearOperator(logarithm[None]),
            identity[None],
            1.0,
            policy=_EXPONENTIAL,
            rhs_layout=_IDENTITY_COLUMNS,
        )
        value = jnp.asarray(result.value)[0]
        return value, jnp.all(result.successful)

    def realize(
        self, parameters: dict[str, Array], reference: dict[str, Any], /
    ) -> tuple[tuple[Array, Array | None], Array]:
        moved = _constrained(self.constraint, parameters["atoms"], reference["mobile"])
        gradient, realized = self.deformation(
            parameters["strain"], reference["cell_factor"]
        )
        positions = contract("ni,ji->nj", reference["positions"] + moved, gradient)
        vectors = contract("ki,ji->kj", reference["cell"], gradient)
        return (positions, vectors), realized & jnp.all(jnp.isfinite(gradient))


SymmetryLayout = AtomicSymmetryPlan | RigidSymmetryLayout


def _symmetry_slots(layout: SymmetryLayout, /) -> np.ndarray:
    if isinstance(layout, AtomicSymmetryPlan):
        starts = layout.representative_start
        widths = layout.representative_dimension
    else:
        starts = layout.site_start
        widths = tuple(
            center + orientation
            for center, orientation in zip(
                layout.center_dimension, layout.orientation_dimension, strict=True
            )
        )
    width = max(max(widths, default=0), 1)
    slots = np.full((len(starts), width), -1, dtype=np.int64)
    for row, (start, count) in enumerate(zip(starts, widths, strict=True)):
        slots[row, :count] = np.arange(start, start + count)
    return slots


def _expanded_atomic_numbers(layout: SymmetryLayout, /) -> np.ndarray:
    return np.asarray(jax.device_get(layout.unit.atomic_numbers), dtype=np.int64)


@final
class AtomisticSymmetryChart(AbstractAtomisticRelaxationChart):
    """Crystal symmetry chart moved by its genuine retraction from an anchor.

    ``layout`` is an `AtomicSymmetryPlan` (representative atoms along exact
    site directions, affine) or a `RigidSymmetryLayout` (rigid molecules,
    quaternion orientation retraction and polar frame rotation under strain).
    ``system_atoms[a]`` is the explicit system slot of expanded atom ``a``;
    elements must agree and every active system atom must be mapped exactly
    once, so the chart is bound by its action contract, not by array shapes.

    Parameters are ``{"sites": (S, W)}`` per symmetry site (padded slots stay
    inactive by construction) plus ``{"strain": (K,)}`` metric coordinates when
    ``vary_cell``.  The point at ``xi`` is ``retract(anchor, xi)``; its chart
    gradient is the exact pullback, and at the anchor it equals the
    `SymmetryTangentOperator` adjoint action.  The bound system cell should be
    the chart's lower-triangular Cholesky frame of the anchor.

    PyTree: the system binding ``atom_slots`` (``int32`` per expanded atom) and
    the chart digest ``chart_identity`` are dynamic; ``slot_table`` is derived
    only from the symmetry layout's static site extents and stays static
    (layout), as do ``vary_cell`` and the expanded-atom count (shape).
    """

    layout: AtomicSymmetryPlan | RigidSymmetryLayout
    atom_slots: Array
    chart_identity: IdentityWords
    slot_table: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    vary_cell: bool = eqx.field(static=True)

    def __init__(
        self,
        layout: AtomicSymmetryPlan | RigidSymmetryLayout,
        system_atoms: ArrayLike,
        /,
        *,
        vary_cell: bool = True,
    ) -> None:
        if not isinstance(layout, (AtomicSymmetryPlan, RigidSymmetryLayout)):
            raise TypeError(
                "layout must be an AtomicSymmetryPlan or RigidSymmetryLayout."
            )
        if not isinstance(vary_cell, bool):
            raise TypeError("vary_cell must be a bool.")
        atoms = np.asarray(system_atoms)
        count = layout.unit.atom_count
        if not np.issubdtype(atoms.dtype, np.integer) or atoms.shape != (count,):
            raise ValueError(f"system_atoms must be {count} integer system slots.")
        if np.unique(atoms).shape[0] != count or np.any(atoms < 0):
            raise ValueError("system_atoms must be distinct non-negative slots.")
        self.layout = layout
        self.atom_slots = jnp.asarray(atoms, dtype=jnp.int32)
        self.slot_table = tuple(
            tuple(int(value) for value in row) for row in _symmetry_slots(layout)
        )
        self.vary_cell = vary_cell
        self.chart_identity = IdentityWords.of(
            {
                "kind": "atomistic-symmetry-chart",
                "layout": layout.layout_id,
                "system_atoms": [int(value) for value in atoms],
                "vary_cell": vary_cell,
            }
        )

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "atomistic-symmetry-chart-layout",
                "layout": self.layout.layout_id,
                "atoms": int(self.atom_slots.shape[0]),
                "vary_cell": self.vary_cell,
            }
        )

    @property
    def system_atoms(self) -> Array:
        """System slot of every expanded atom (dynamic ``int32``)."""
        return self.atom_slots

    @property
    def site_slots(self) -> np.ndarray:
        """Static ``(S, W)`` flat chart index of every site slot, ``-1`` padded."""
        return np.asarray(self.slot_table, dtype=np.int64)

    @property
    def variable_cell(self) -> bool:
        return self.vary_cell

    @property
    def parameter_count(self) -> int:
        return int(self.layout.parameter_count)

    @property
    def strain_start(self) -> int:
        return int(self.layout.strain_start)

    @property
    def atomic(self) -> bool:
        return isinstance(self.layout, AtomicSymmetryPlan)

    def admit(self, system: PreparedAtomisticSystem, /) -> None:
        cell = system.cell
        if cell is None or cell.rank != 3 or not cell.fully_periodic:
            raise ValueError("A symmetry chart requires a fully periodic 3D cell.")
        atoms = np.asarray(self.atom_slots, dtype=np.int64)
        if int(np.max(atoms)) >= system.capacity:
            raise ValueError("system_atoms exceed the system capacity.")
        active = np.flatnonzero(np.asarray(system.active_mask, dtype=np.bool_))
        if not np.array_equal(np.sort(atoms), active):
            raise ValueError(
                "system_atoms must map every active system atom exactly once."
            )
        numbers = np.asarray(system.plan.atomic_numbers)[atoms]
        if not np.array_equal(numbers, _expanded_atomic_numbers(self.layout)):
            raise ValueError(
                "Expanded symmetry atoms and system atoms differ in element."
            )
        mobile = _mobile(system)
        if not np.all(mobile[atoms]):
            raise ValueError(
                "A symmetry chart moves every mapped atom; all must be mobile."
            )

    def site_masks(self, system: PreparedAtomisticSystem, /) -> dict[str, np.ndarray]:
        masks = {"sites": np.any(self.site_slots >= 0, axis=1)}
        if self.vary_cell:
            masks["strain"] = np.asarray(True)
        return masks

    def start(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        positions: Array | None,
        cell_vectors: Array | None,
        chart_state: Any,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        if cell_vectors is not None:
            raise ValueError("A symmetry chart realizes its own cell vectors.")
        expected = AtomicSymmetricState if self.atomic else AsymmetricUnitState
        if not isinstance(chart_state, expected):
            raise TypeError(f"chart_state must be an {expected.__name__}.")
        dtype = jnp.dtype(system.plan.coordinate_dtype)
        padding = (
            jnp.zeros((system.capacity, 3), dtype=dtype)
            if positions is None
            else jnp.where(
                jnp.asarray(system.active_mask)[:, None],
                0.0,
                jnp.nan_to_num(jnp.asarray(positions, dtype=dtype)),
            )
        )
        parameters = {
            "sites": jnp.zeros(self.site_slots.shape, dtype=dtype),
        }
        if self.vary_cell:
            parameters["strain"] = jnp.zeros(
                (self.parameter_count - self.strain_start,), dtype=dtype
            )
        return parameters, {"anchor": chart_state, "padding": padding}

    def coordinates(self, parameters: dict[str, Array], /) -> Array:
        """Scatter site/strain parameters (or gradients) to the flat chart vector."""
        sites = parameters["sites"]
        rows, columns = np.nonzero(self.site_slots >= 0)
        flat = jnp.zeros((self.parameter_count,), dtype=sites.dtype)
        flat = flat.at[self.site_slots[rows, columns]].set(sites[rows, columns])
        if self.vary_cell:
            flat = flat.at[self.strain_start :].set(parameters["strain"])
        return flat

    def state(self, parameters: dict[str, Array], reference: dict[str, Any], /) -> Any:
        """Chart state ``retract(anchor, xi)`` at ``parameters``."""
        step = self.coordinates(parameters)
        anchor = reference["anchor"]
        if isinstance(self.layout, AtomicSymmetryPlan):
            return retract_atomic(self.layout, anchor, step)
        return self.layout.retract(anchor, step)

    def realize(
        self, parameters: dict[str, Array], reference: dict[str, Any], /
    ) -> tuple[tuple[Array, Array | None], Array]:
        state = self.state(parameters, reference)
        if isinstance(self.layout, AtomicSymmetryPlan):
            expansion = expand_symmetric_atoms(self.layout, state)
            realized = jnp.all(expansion.cell.successful)
        else:
            expansion = expand_asymmetric_unit(self.layout.unit, state)
            realized = jnp.all(expansion.cell.successful) & jnp.all(
                expansion.offset_solve_successful
            )
        positions = reference["padding"].at[self.system_atoms].set(expansion.positions)
        return (positions, expansion.cell.vectors), realized


# -- convergence and plan ----------------------------------------------------------


@final
class AtomisticRelaxationConvergence(StrictModule, NonTrainableState):
    """Projected/full residual thresholds and the evaluation budget.

    ``projected_tolerance`` bounds the largest optimizer-site norm of the chart
    gradient (chart units).  ``full_force_tolerance`` bounds the largest
    Cartesian force norm over mobile atoms and ``full_stress_tolerance`` the
    largest component of ``sigma + P I`` (variable cells only).  With
    ``require_full`` a group converges only when both residual families pass;
    otherwise the full residual is reported but does not terminate.
    ``maximum_evaluations`` counts consumed (not capacity-replayed) trials.

    PyTree: the tolerances and the evaluation budget are dynamic 0-d leaves
    holding the exact host values (``float64``/``int64``); which optional
    tolerances are declared and ``require_full`` are static selectors.  The
    canonical digest is dynamic ``convergence_identity`` (host
    ``convergence_id``); ``layout_id`` is the content-free selector layout.
    """

    projected_tolerance: np.ndarray
    full_force_tolerance: np.ndarray | None
    full_stress_tolerance: np.ndarray | None
    maximum_evaluations: np.ndarray
    convergence_identity: IdentityWords
    require_full: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        projected_tolerance: float,
        maximum_evaluations: int,
        full_force_tolerance: float | None = None,
        full_stress_tolerance: float | None = None,
        require_full: bool = False,
    ) -> None:
        projected = _positive(projected_tolerance, "projected_tolerance")
        force = (
            None
            if full_force_tolerance is None
            else _positive(full_force_tolerance, "full_force_tolerance")
        )
        stress = (
            None
            if full_stress_tolerance is None
            else _positive(full_stress_tolerance, "full_stress_tolerance")
        )
        if not isinstance(require_full, bool):
            raise TypeError("require_full must be a bool.")
        if require_full and force is None:
            raise ValueError("require_full needs full_force_tolerance.")
        if isinstance(maximum_evaluations, bool) or int(maximum_evaluations) < 1:
            raise ValueError("maximum_evaluations must be a positive int.")
        evaluations = int(maximum_evaluations)
        self.projected_tolerance = np.asarray(projected, dtype=np.float64)
        self.full_force_tolerance = (
            None if force is None else np.asarray(force, dtype=np.float64)
        )
        self.full_stress_tolerance = (
            None if stress is None else np.asarray(stress, dtype=np.float64)
        )
        self.require_full = require_full
        self.maximum_evaluations = np.asarray(evaluations, dtype=np.int64)
        self.convergence_identity = IdentityWords.of(
            {
                "kind": "atomistic-relaxation-convergence",
                "projected": projected,
                "force": force,
                "stress": stress,
                "require_full": require_full,
                "maximum_evaluations": evaluations,
            }
        )

    @property
    def convergence_id(self) -> str:
        return self.convergence_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "atomistic-relaxation-convergence-layout",
                "force": self.full_force_tolerance is not None,
                "stress": self.full_stress_tolerance is not None,
                "require_full": self.require_full,
            }
        )


def _method_id(method: RelaxationMethod, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-relaxation-method",
            "method": method.method_id,
            "controls": {
                name: getattr(method, name)
                for name in sorted(vars(method))
                if isinstance(getattr(method, name), (int, float, bool, str))
            },
        }
    )


@final
class AtomisticRelaxationPlan(StrictModule, NonTrainableState):
    """Prepared scalar program, chart, native optimizer and convergence policy.

    ``target_pressure`` (program energy per cubed length unit) adds ``P |det H|``
    to the minimized objective and requires a variable-cell chart; a tensor
    target stress is not a scalar enthalpy and is not accepted.  FIRE with a
    downhill check compares enthalpies of accepted iterates; FIRE2 and
    `TrustScaledLBFGS` are force-driven and never claim monotone decrease.
    ``group_count`` independent relaxations of the same program share one
    device-resident layout.

    PyTree: ``target_pressure`` is a dynamic exact-``float64`` 0-d leaf and
    the plan digest is dynamic ``plan_identity`` (host ``plan_id``);
    ``group_count`` is layout.  ``layout_id`` is the content-free plan layout
    (program, chart and convergence layouts, optimizer method, groups).
    """

    program: AbstractPreparedAtomisticHamiltonian
    chart: AbstractAtomisticRelaxationChart
    method: FIRE | FIRE2 | TrustScaledLBFGS
    convergence: AtomisticRelaxationConvergence
    target_pressure: np.ndarray
    plan_identity: IdentityWords
    group_count: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        program: AbstractPreparedAtomisticHamiltonian,
        chart: AbstractAtomisticRelaxationChart,
        /,
        *,
        method: FIRE | FIRE2 | TrustScaledLBFGS,
        convergence: AtomisticRelaxationConvergence,
        target_pressure: float = 0.0,
        group_count: int = 1,
    ) -> None:
        system = program.system
        if system.topology.constraint_count > 0:
            raise ValueError(
                "Holonomic topology constraints have no relaxation chart retraction; "
                "relaxation refuses constrained systems."
            )
        chart.admit(system)
        pressure = float(target_pressure)
        if not math.isfinite(pressure):
            raise ValueError("target_pressure must be finite.")
        if pressure != 0.0 and not chart.variable_cell:
            raise ValueError("A target pressure requires a variable-cell chart.")
        if convergence.full_stress_tolerance is not None and not chart.variable_cell:
            raise ValueError("full_stress_tolerance applies only to variable cells.")
        if isinstance(group_count, bool) or int(group_count) < 1:
            raise ValueError("group_count must be a positive int.")
        if not np.any(_mobile(system)):
            raise ValueError("Relaxation requires at least one mobile active atom.")
        self.program = program
        self.chart = chart
        self.method = method
        self.convergence = convergence
        self.target_pressure = np.asarray(pressure, dtype=np.float64)
        self.group_count = int(group_count)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-relaxation-plan",
                "program": program.prepared_id,
                "chart": chart.chart_id,
                "method": _method_id(method),
                "convergence": convergence.convergence_id,
                "target_pressure": pressure,
                "group_count": self.group_count,
            }
        )

    @property
    def plan_id(self) -> str:
        return self.plan_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "atomistic-relaxation-plan-layout",
                "program": self.program.layout_id,
                "chart": self.chart.layout_id,
                "method": f"{type(self.method).__qualname__}:{self.method.method_id}",
                "convergence": self.convergence.layout_id,
                "group_count": self.group_count,
            }
        )

    def prepare(self) -> PreparedAtomisticRelaxation:
        return PreparedAtomisticRelaxation(self, self.program)


# -- state ------------------------------------------------------------------------


@final
class AtomisticRelaxationState(StrictModule):
    """Complete group-batched relaxation state (leading axis ``G``).

    ``optimizer.parameters`` is the pending proposal and
    ``optimizer.accepted_parameters`` the accepted iterate; ``resources`` is
    the accepted case-one bundle state of every group and
    ``candidate_resources`` the most recent candidate (capacity evidence for
    `grow`).  Every ``accepted_*``/residual field belongs to the accepted
    iterate.  ``cell_vectors``, ``cell_covector`` (``d(E + P V)/dH`` at fixed
    unwrapped positions), ``stress`` (tension-positive Cauchy stress of ``E``)
    and ``volume`` are ``None`` without a cell.  Counters: ``evaluations`` and
    ``iterations`` are run totals, the optimizer counters plus the
    ``evaluation_base``/``iteration_base`` carried across Hamiltonian
    transitions (a fresh optimizer restarts its own counters);
    ``capacity_replays`` counts capacity-refused trials, ``rejected_trials``
    consumed but rejected trials and ``reprimes`` budget-free evidence
    re-evaluations after a resource rebinding or Hamiltonian transition.
    ``hamiltonian_lineage`` lists the ``transition_id`` of every accepted
    `transition_hamiltonian`.

    PyTree: the plan and resource-binding digests are dynamic
    ``plan_identity``/``binding_identity`` and the lineage is the stacked
    ``(L, 8)`` ``lineage_identities`` (host properties ``plan_id``,
    ``binding_id``, ``hamiltonian_lineage`` decode the exact strings).  Compiled
    `step` compares the words and fails unbound groups explicitly; the lineage
    depth ``L`` is an array shape, so each Hamiltonian transition depth
    compiles once (the exact transition digest commits to the full previous
    lineage, which therefore cannot be folded into a fixed-shape head).
    """

    optimizer: FIREState | TrustScaledLBFGSState
    reference: PyTree[Any]
    resources: AtomisticNeighborhoodBundleState
    candidate_resources: AtomisticNeighborhoodBundleState
    positions: Array
    cell_vectors: Array | None
    energy: Array
    objective: Array
    forces: Array
    cell_covector: Array | None
    stress: Array | None
    volume: Array | None
    projected_residual: Array
    full_force_residual: Array
    full_stress_residual: Array
    projected_converged: Array
    full_converged: Array
    status: Array
    trial_outcome: Array
    active: Array
    capacity_replays: Array
    rejected_trials: Array
    reprimes: Array
    evaluation_base: Array
    iteration_base: Array
    plan_identity: IdentityWords
    binding_identity: IdentityWords
    lineage_identities: IdentityWords

    @property
    def plan_id(self) -> str:
        return self.plan_identity.fingerprint()

    @property
    def binding_id(self) -> str:
        return self.binding_identity.fingerprint()

    @property
    def hamiltonian_lineage(self) -> tuple[str, ...]:
        return _identity_strings(self.lineage_identities)

    @property
    def terminal(self) -> Array:
        status = self.status
        return (status != int(AtomisticRelaxationStatus.ITERATING)) & (
            status != int(AtomisticRelaxationStatus.CAPACITY_PENDING)
        )

    @property
    def evaluations(self) -> Array:
        """Consumed evaluations of the whole run, across Hamiltonian transitions."""
        return self.optimizer.evaluations + self.evaluation_base

    @property
    def iterations(self) -> Array:
        """Optimizer steps of the whole run, across Hamiltonian transitions."""
        return self.optimizer.iterations + self.iteration_base


class _Trial(NamedTuple):
    candidate: AtomisticNeighborhoodBundleState
    positions: Array
    cell_vectors: Array | None
    objective: Array
    energy: Array
    gradient: dict[str, Array]
    forces: Array
    cell_covector: Array | None
    stress: Array | None
    volume: Array | None
    full_force: Array
    full_stress: Array
    capacity: Array
    scientific: Array
    cell_failure: Array


def _wrapped(
    system: PreparedAtomisticSystem, positions: Array, vectors: Array | None, /
) -> tuple[Array, Array | None]:
    """Wrapped positions and integer image counts (outside differentiation)."""
    cell = system.cell
    if cell is None or vectors is None:
        return positions, None
    wrapped, images = cell.wrap_with_vectors(
        jax.lax.stop_gradient(positions), jax.lax.stop_gradient(vectors)
    )
    del wrapped
    return positions - contract("ni,ij->nj", images.astype(positions.dtype), vectors), (
        jax.lax.stop_gradient(images)
    )


def _cell_admissible(
    cell: PeriodicCell | None, vectors: Array | None, orientation: Array | None, /
) -> Array:
    if cell is None or vectors is None:
        return jnp.asarray(True)
    _, inverse_ok = cell.inverse_for_vectors_with_status(vectors)
    admissible = jnp.all(inverse_ok) & jnp.all(jnp.isfinite(vectors))
    if orientation is not None:
        admissible = admissible & (_signed_volume(vectors) * orientation > 0.0)
    return admissible


def trial_geometry(
    program: AbstractPreparedAtomisticHamiltonian,
    bundle: PreparedAtomisticNeighborhoodBundle,
    positions: Array,
    cell_vectors: Array | None,
    /,
) -> tuple[AtomisticNeighborhoodGeometry, Array, Array | None]:
    """Bind unwrapped ``positions`` as wrapped case-one bundle geometry.

    Returns the geometry, the wrapped positions (differentiable in
    ``positions``/``cell_vectors``) and the fixed integer image counts.
    """
    wrapped, images = _wrapped(program.system, positions, cell_vectors)
    geometry = bind_atomistic_bundle_geometry(
        bundle,
        program.system,
        jax.lax.stop_gradient(wrapped),
        cell_vectors=None
        if cell_vectors is None
        else jax.lax.stop_gradient(cell_vectors),
        image_counts=images,
    )
    return geometry, wrapped, images


def program_energy(
    program: AbstractPreparedAtomisticHamiltonian,
    resources: AtomisticNeighborhoodBundleState,
    positions: Array,
    cell_vectors: Array | None,
    images: Array | None,
    /,
) -> tuple[Array, Array]:
    """Program energy at unwrapped ``positions`` bound to fixed image counts.

    Differentiable in ``positions`` (cell fixed) and ``cell_vectors`` (fixed
    unwrapped positions).  Returns ``(energy, program_successful)``.
    """
    if cell_vectors is None or images is None:
        energy, auxiliary = program.energy(positions, resources)
    else:
        wrapped = positions - contract(
            "ni,ij->nj", images.astype(positions.dtype), cell_vectors
        )
        energy, auxiliary = program.energy(
            wrapped, resources, cell_vectors=cell_vectors, image_counts=images
        )
    return energy, auxiliary.successful


class AtomisticCartesianTrial(NamedTuple):
    """Fixed-cell Cartesian trial evaluation against accepted resources."""

    candidate: AtomisticNeighborhoodBundleState
    energy: Array
    forces: Array
    capacity: Array
    scientific: Array


def evaluate_cartesian_trial(
    program: AbstractPreparedAtomisticHamiltonian,
    bundle: PreparedAtomisticNeighborhoodBundle,
    resources: AtomisticNeighborhoodBundleState,
    positions: Array,
    cell_vectors: Array | None,
    /,
) -> AtomisticCartesianTrial:
    """Refresh candidate resources and evaluate energy and forces at one point.

    Pure and traceable; vmap it over images or groups with stacked accepted
    resources.  ``capacity`` marks a capacity-only resource failure (the trial
    must be replayed, not consumed); ``scientific`` any other failure.
    """
    geometry, _, images = trial_geometry(program, bundle, positions, cell_vectors)
    candidate = bundle.refresh(resources, geometry)
    (energy, successful), gradient = jax.value_and_grad(
        lambda value: program_energy(program, candidate, value, cell_vectors, images),
        has_aux=True,
    )(positions)
    active = jnp.asarray(program.system.active_mask)[:, None]
    forces = jnp.where(active, -gradient, 0.0)
    capacity = candidate.capacity_only[0]
    finite = jnp.isfinite(energy) & jnp.all(jnp.isfinite(forces))
    scientific = ~capacity & (~candidate.successful[0] | ~successful | ~finite)
    return AtomisticCartesianTrial(candidate, energy, forces, capacity, scientific)


def replay_capacity(
    bundle: PreparedAtomisticNeighborhoodBundle,
    accepted: AtomisticNeighborhoodBundleState,
    geometry: AtomisticNeighborhoodGeometry,
    /,
) -> tuple[PreparedAtomisticNeighborhoodBundle, bool]:
    """Host lifecycle: grow declared capacities until ``geometry`` fits.

    Replays the trial from ``accepted`` (rebased) at every level.  Returns the
    final bundle and whether a declared ladder was exhausted.
    """
    candidate = bundle.refresh(bundle.rebase(accepted), geometry)
    while bool(np.asarray(candidate.capacity_only)[0]):
        try:
            bundle = bundle.grow_capacities(candidate)
        except RuntimeError:
            return bundle, True
        candidate = bundle.refresh(bundle.rebase(accepted), geometry)
    return bundle, False


def initialize_capacity(
    bundle: PreparedAtomisticNeighborhoodBundle,
    geometry: AtomisticNeighborhoodGeometry,
    /,
) -> tuple[PreparedAtomisticNeighborhoodBundle, bool]:
    """Host lifecycle: grow declared capacities until a fresh epoch at ``geometry`` fits.

    Returns the final bundle and whether a declared ladder was exhausted.
    """
    state = bundle.initialize(geometry)
    while bool(np.asarray(state.capacity_only)[0]):
        try:
            bundle = bundle.grow_capacities(state)
        except RuntimeError:
            return bundle, True
        state = bundle.initialize(geometry)
    return bundle, False


class AtomisticHamiltonianTransitionKind(StrEnum):
    """Why a runtime moves to another prepared Hamiltonian.

    ``SCIENTIFIC_MODEL``: the canonical ``scientific_id`` changed (model,
    parameters, terms, coefficients).  ``NUMERICAL_REALIZATION``: the same
    scientific Hamiltonian realized differently (prepared term realizations or
    graph execution changed), recorded as its own lineage step and never
    presented as a scientific change.
    """

    SCIENTIFIC_MODEL = "scientific-model"
    NUMERICAL_REALIZATION = "numerical-realization"


def hamiltonian_transition_kind(
    previous: AbstractPreparedAtomisticHamiltonian,
    program: AbstractPreparedAtomisticHamiltonian,
    /,
) -> AtomisticHamiltonianTransitionKind:
    """Classify a Hamiltonian change by the canonical ``scientific_id``.

    Refuses a program for another prepared system and a pure resource
    rebinding (same scientific identity and the same prepared term
    realizations), which is ``grow``/``with_neighborhood_bundle``, not a
    transition.
    """
    if not isinstance(program, AbstractPreparedAtomisticHamiltonian):
        raise TypeError("program must be a prepared atomistic Hamiltonian.")
    if program.system.prepared_id != previous.system.prepared_id:
        raise ValueError("The new Hamiltonian binds another prepared system.")
    if program.scientific_id != previous.scientific_id:
        return AtomisticHamiltonianTransitionKind.SCIENTIFIC_MODEL
    if tuple(term.prepared_id for term in program.terms) != tuple(
        term.prepared_id for term in previous.terms
    ):
        return AtomisticHamiltonianTransitionKind.NUMERICAL_REALIZATION
    raise ValueError(
        "The program has the same scientific identity and numerical realization; "
        "a resource-only rebinding is grow(), not a Hamiltonian transition."
    )


@final
class AtomisticHamiltonianTransition(StrictModule, NonTrainableState):
    """Lineage record of one Hamiltonian update of a relaxation or path runtime.

    ``kind`` says whether the canonical scientific identity changed or only
    its numerical realization.  ``successful`` false means the update was
    refused for ``reason`` and the returned runtime/state are the previous
    ones, unchanged.  On success the accepted geometry is retained, every
    revisable group with an accepted iterate (``reprimed``) was re-evaluated
    under the new Hamiltonian without consuming budget, and the optimizer
    history was discarded (``optimizer_policy``): L-BFGS secant pairs and FIRE
    velocity, time step, mixing and positive-step count are never reused.
    """

    reprimed: Array
    successful: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    kind: AtomisticHamiltonianTransitionKind = eqx.field(static=True)
    previous_scientific_id: str = eqx.field(static=True)
    scientific_id: str = eqx.field(static=True)
    previous_binding_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    previous_plan_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    optimizer_policy: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        reprimed: ArrayLike,
        /,
        *,
        successful: bool,
        reason: str,
        kind: AtomisticHamiltonianTransitionKind,
        previous_scientific_id: str,
        scientific_id: str,
        previous_binding_id: str,
        binding_id: str,
        previous_plan_id: str,
        plan_id: str,
        lineage: tuple[str, ...],
    ) -> None:
        mask = jnp.asarray(reprimed, dtype=jnp.bool_)
        if successful == bool(reason):
            raise ValueError(
                "A refused transition carries a reason; a successful one none."
            )
        self.reprimed = mask
        self.successful = bool(successful)
        self.reason = str(reason)
        self.kind = AtomisticHamiltonianTransitionKind(kind)
        self.previous_scientific_id = previous_scientific_id
        self.scientific_id = scientific_id
        self.previous_binding_id = previous_binding_id
        self.binding_id = binding_id
        self.previous_plan_id = previous_plan_id
        self.plan_id = plan_id
        self.optimizer_policy = "reset-history"
        self.transition_id = canonical_fingerprint(
            {
                "kind": "atomistic-hamiltonian-transition",
                "successful": self.successful,
                "transition_kind": self.kind.value,
                "reason": self.reason,
                "previous_scientific": previous_scientific_id,
                "scientific": scientific_id,
                "previous_binding": previous_binding_id,
                "binding": binding_id,
                "previous_plan": previous_plan_id,
                "plan": plan_id,
                "optimizer_policy": self.optimizer_policy,
                "reprimed": np.asarray(jax.device_get(mask)).tolist(),
                "lineage": list(lineage),
            }
        )


class AtomisticHamiltonianTransitionResult(NamedTuple):
    """Runtime, state and lineage record returned by ``transition_hamiltonian``."""

    runtime: PreparedAtomisticRelaxation
    state: AtomisticRelaxationState
    record: AtomisticHamiltonianTransition


class _Assessment(NamedTuple):
    valid: Array
    projected: Array
    projected_ok: Array
    full_ok: Array
    converged: Array
    value: Array
    gradient: dict[str, Array]


_REVISABLE = (
    int(AtomisticRelaxationStatus.ITERATING),
    int(AtomisticRelaxationStatus.CONVERGED),
    int(AtomisticRelaxationStatus.CAPACITY_PENDING),
)


@final
class PreparedAtomisticRelaxation(StrictModule, NonTrainableState):
    """A relaxation plan bound to one program resource binding and its layout.

    `init`, `step` and `run` are pure and device-resident; `grow` and `solve`
    are host lifecycle transactions.  After `grow` the returned runtime holds
    the rebound program (new ``prepared_id``) and the state its rebased
    resources and reprimed accepted evidence.

    PyTree: the binding digest is dynamic ``prepared_identity`` (host
    ``prepared_id``); ``layout_id`` is the content-free executable layout
    (plan layout and the bundle layout).
    """

    plan: AtomisticRelaxationPlan
    program: AbstractPreparedAtomisticHamiltonian
    bundle: PreparedAtomisticNeighborhoodBundle
    layout: OptimizerGroupLayout
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: AtomisticRelaxationPlan,
        program: AbstractPreparedAtomisticHamiltonian,
        /,
    ) -> None:
        if not isinstance(plan, AtomisticRelaxationPlan):
            raise TypeError("plan must be an AtomisticRelaxationPlan.")
        if program.system.prepared_id != plan.program.system.prepared_id:
            raise ValueError("The rebound program belongs to another prepared system.")
        groups = plan.group_count
        masks = plan.chart.site_masks(program.system)
        index = {
            name: np.where(
                np.broadcast_to(mask, (groups, *np.shape(mask))),
                np.arange(groups).reshape((groups,) + (1,) * np.ndim(mask)),
                -1,
            ).astype(np.int32)
            for name, mask in masks.items()
        }
        self.plan = plan
        self.program = program
        self.bundle = program.neighborhood_bundle
        self.layout = OptimizerGroupLayout(index, group_count=groups)
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-atomistic-relaxation",
                "plan": plan.plan_id,
                "program_binding": program.prepared_id,
                "bundle": self.bundle.prepared_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        return self.prepared_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "prepared-atomistic-relaxation-layout",
                "plan": self.plan.layout_id,
                "bundle": self.bundle.layout_id,
            }
        )

    @property
    def system(self) -> PreparedAtomisticSystem:
        return self.program.system

    # -- core evaluation ----------------------------------------------------------

    def _objective(
        self,
        program: AbstractPreparedAtomisticHamiltonian,
        resources: AtomisticNeighborhoodBundleState,
        positions: Array,
        vectors: Array | None,
        images: Array | None,
        pressure: Array,
        /,
    ) -> tuple[Array, tuple[Array, Array, Array | None]]:
        energy, successful = program_energy(
            program, resources, positions, vectors, images
        )
        volume = None
        objective = energy
        if self.plan.chart.variable_cell and vectors is not None:
            volume = jnp.abs(_signed_volume(vectors))
            objective = energy + pressure * volume
        return objective, (energy, successful, volume)

    def _trial(
        self,
        parameters: dict[str, Array],
        reference: dict[str, Any],
        resources: AtomisticNeighborhoodBundleState,
        /,
    ) -> _Trial:
        system = self.system
        chart = self.plan.chart
        (point, vjp_chart, realized) = jax.vjp(
            lambda value: chart.realize(value, reference), parameters, has_aux=True
        )
        positions, vectors = point
        geometry, _, images = trial_geometry(
            self.program, self.bundle, positions, vectors
        )
        candidate = self.bundle.refresh(resources, geometry)
        pressure = jnp.asarray(self.plan.target_pressure, dtype=positions.dtype)
        objective, vjp_cartesian, (energy, successful, volume) = jax.vjp(
            lambda r, h: self._objective(self.program, candidate, r, h, images, pressure),
            positions,
            vectors,
            has_aux=True,
        )
        gradient_positions, gradient_cell = vjp_cartesian(jnp.ones_like(objective))
        (gradient,) = vjp_chart((gradient_positions, gradient_cell))
        mobile = _mobile_device(system)
        active = jnp.asarray(system.active_mask)[:, None]
        forces = jnp.where(active, -gradient_positions, 0.0)
        full_force = jnp.max(jnp.where(mobile, jnp.linalg.norm(forces, axis=-1), 0.0))
        stress = None
        full_stress = jnp.zeros((), dtype=positions.dtype)
        if chart.variable_cell and vectors is not None and volume is not None:
            strain = contract("ni,nj->ij", gradient_positions, positions) + contract(
                "ki,kj->ij", gradient_cell, vectors
            )
            symmetric = 0.5 * (strain + strain.T)
            full_stress = jnp.max(jnp.abs(symmetric)) / volume
            stress = (symmetric - pressure * volume * jnp.eye(3, dtype=strain.dtype)) / (
                volume
            )
        orientation = reference.get("orientation")
        cell_ok = realized & _cell_admissible(
            system.cell if chart.variable_cell else None, vectors, orientation
        )
        finite = (
            jnp.isfinite(objective)
            & _tree_finite(gradient)
            & jnp.all(jnp.isfinite(forces))
        )
        capacity = cell_ok & candidate.capacity_only[0]
        scientific = (
            cell_ok & ~capacity & (~candidate.successful[0] | ~successful | ~finite)
        )
        return _Trial(
            candidate,
            positions,
            vectors,
            objective,
            energy,
            gradient,
            forces,
            gradient_cell if chart.variable_cell else None,
            stress,
            volume,
            full_force,
            full_stress,
            capacity,
            scientific,
            ~cell_ok,
        )

    def _assess(self, trial: _Trial, /) -> _Assessment:
        """Validity, residual convergence and the optimizer feed of a trial."""
        convergence = self.plan.convergence
        valid = ~trial.scientific & ~trial.cell_failure & ~trial.capacity
        projected = self.layout.site_norm_max(trial.gradient)
        projected_ok = projected <= convergence.projected_tolerance
        full_ok = (
            trial.full_force <= convergence.full_force_tolerance
            if convergence.full_force_tolerance is not None
            else jnp.zeros_like(projected_ok)
        )
        if convergence.full_stress_tolerance is not None:
            full_ok = full_ok & (trial.full_stress <= convergence.full_stress_tolerance)
        converged = valid & projected_ok & (full_ok if convergence.require_full else True)
        nan = jnp.asarray(jnp.nan, dtype=trial.objective.dtype)
        gradient = _tree_where(
            valid,
            trial.gradient,
            jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), trial.gradient),
        )
        return _Assessment(
            valid,
            projected,
            projected_ok,
            full_ok,
            converged,
            jnp.where(valid, trial.objective, nan),
            gradient,
        )

    # -- lifecycle ---------------------------------------------------------------

    def init(
        self,
        *,
        positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        chart_states: Any = None,
        active: ArrayLike | None = None,
        hamiltonian_lineage: tuple[str, ...] = (),
    ) -> AtomisticRelaxationState:
        """Start every group at rest with resources initialized at its start point.

        ``positions`` ``(G, N, 3)`` (unwrapped Cartesian) and optional
        ``cell_vectors`` ``(G, r, 3)`` start Cartesian/deformation charts;
        ``chart_states`` (stacked over ``G``) start a symmetry chart.
        ``active`` ``(G,)`` marks lanes in use; inactive lanes never evaluate.
        ``hamiltonian_lineage`` is the Hamiltonian history of the bank the new
        groups join (transition ids of earlier `transition_hamiltonian` calls);
        it describes the Hamiltonian, not these groups, whose counters start at
        zero.  Its entries are exact canonical ``transition_id`` digests.
        """
        lineage = _identity_rows(tuple(hamiltonian_lineage))
        groups = self.plan.group_count
        system = self.system
        starts = []
        for group in range(groups):
            starts.append(
                self.plan.chart.start(
                    system,
                    positions=None
                    if positions is None
                    else jnp.asarray(positions)[group],
                    cell_vectors=(
                        None if cell_vectors is None else jnp.asarray(cell_vectors)[group]
                    ),
                    chart_state=(
                        None if chart_states is None else _tree_take(chart_states, group)
                    ),
                )
            )
        parameters = _tree_stack([value[0] for value in starts])
        reference = _tree_stack([value[1] for value in starts])
        lanes = (
            jnp.ones((groups,), dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        if lanes.shape != (groups,):
            raise ValueError("active must have shape (group_count,).")

        def first(parameters_: dict[str, Array], reference_: dict[str, Any]) -> Any:
            (positions_, vectors_), _ = self.plan.chart.realize(parameters_, reference_)
            geometry, _, _ = trial_geometry(
                self.program, self.bundle, positions_, vectors_
            )
            return self.bundle.initialize(geometry), positions_, vectors_

        resources, start_positions, start_vectors = jax.vmap(first)(parameters, reference)
        dtype = start_positions.dtype
        optimizer = self.plan.method.init(
            parameters, layout=self.layout, value_dtype=dtype
        )
        nan = jnp.full((groups,), jnp.nan, dtype=dtype)
        no = jnp.zeros((groups,), dtype=jnp.bool_)
        none = jnp.zeros((groups,), dtype=jnp.int32)
        variable = self.plan.chart.variable_cell
        status = jnp.where(
            lanes,
            int(AtomisticRelaxationStatus.ITERATING),
            int(AtomisticRelaxationStatus.INACTIVE),
        ).astype(jnp.int32)
        return AtomisticRelaxationState(
            optimizer=optimizer,
            reference=reference,
            resources=resources,
            candidate_resources=resources,
            positions=start_positions,
            cell_vectors=start_vectors,
            energy=nan,
            objective=nan,
            forces=jnp.full_like(start_positions, jnp.nan),
            cell_covector=(
                jnp.full((groups, 3, 3), jnp.nan, dtype=dtype) if variable else None
            ),
            stress=jnp.full((groups, 3, 3), jnp.nan, dtype=dtype) if variable else None,
            volume=nan if variable else None,
            projected_residual=nan,
            full_force_residual=nan,
            full_stress_residual=nan,
            projected_converged=no,
            full_converged=no,
            status=status,
            trial_outcome=none,
            active=lanes,
            capacity_replays=none,
            rejected_trials=none,
            reprimes=none,
            evaluation_base=none,
            iteration_base=none,
            plan_identity=self.plan.plan_identity,
            binding_identity=self.prepared_identity,
            lineage_identities=lineage,
        )

    def _require_state(self, state: AtomisticRelaxationState, /) -> None:
        """Host transactions: exact refusal of a state of another plan or binding."""
        if not isinstance(state, AtomisticRelaxationState):
            raise TypeError("state must be an AtomisticRelaxationState.")
        if state.plan_id != self.plan.plan_id:
            raise ValueError("Relaxation state belongs to another plan.")
        if state.binding_id != self.prepared_id:
            raise ValueError(
                "Relaxation state belongs to another resource binding; use grow()."
            )

    def _bound(self, state: AtomisticRelaxationState, /) -> Array:
        """Traced: whether ``state`` belongs to this plan and resource binding."""
        if not isinstance(state, AtomisticRelaxationState):
            raise TypeError("state must be an AtomisticRelaxationState.")
        return state.plan_identity.equals(self.plan.plan_identity) & (
            state.binding_identity.equals(self.prepared_identity)
        )

    def step(self, state: AtomisticRelaxationState, /) -> AtomisticRelaxationState:
        """Evaluate every pending proposal once and commit accepted iterates.

        Iterating groups of a state bound to another plan or resource binding
        are not evaluated: they terminate with ``SCIENTIFIC_FAILURE`` (trial
        outcome ``SCIENTIFIC``) without consuming budget.
        """
        convergence = self.plan.convergence
        iterating = state.status == int(AtomisticRelaxationStatus.ITERATING)
        bound = self._bound(state)
        unbound = iterating & ~bound
        pending = iterating & bound
        trial = jax.vmap(self._trial)(
            state.optimizer.parameters, state.reference, state.resources
        )
        consume = pending & ~trial.capacity
        assessment = self._assess(trial)
        valid = assessment.valid
        projected = assessment.projected
        projected_ok = assessment.projected_ok
        full_ok = assessment.full_ok
        converged = assessment.converged
        value = assessment.value
        gradient = assessment.gradient
        optimizer = _step_relaxation_method(
            self.plan.method,
            state.optimizer,
            value,
            gradient,
            converged=converged,
            active=consume,
        )
        accept = consume & valid & optimizer.evaluation_accepted
        resources = jax.vmap(self.bundle.select_cases)(
            accept[:, None], trial.candidate, state.resources
        )

        def keep(new: Array, old: Array) -> Array:
            return _tree_where(accept, new, old)

        def keep_optional(new: Array | None, old: Array | None) -> Array | None:
            if new is None or old is None:
                return old
            return _tree_where(accept, new, old)

        budget = (
            optimizer.evaluations + state.evaluation_base
            >= convergence.maximum_evaluations
        )
        status = state.status
        status = jnp.where(
            unbound,
            int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE),
            jnp.where(
                pending & trial.capacity,
                int(AtomisticRelaxationStatus.CAPACITY_PENDING),
                jnp.where(
                    consume & trial.cell_failure,
                    int(AtomisticRelaxationStatus.CELL_FAILURE),
                    jnp.where(
                        consume & trial.scientific,
                        int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE),
                        jnp.where(
                            consume & (optimizer.status == _SUCCESS),
                            int(AtomisticRelaxationStatus.CONVERGED),
                            jnp.where(
                                consume & budget,
                                int(AtomisticRelaxationStatus.EVALUATION_LIMIT),
                                status,
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        outcome = jnp.where(
            unbound,
            int(AtomisticTrialOutcome.SCIENTIFIC),
            jnp.where(
                accept,
                int(AtomisticTrialOutcome.ACCEPTED),
                jnp.where(
                    pending & trial.capacity,
                    int(AtomisticTrialOutcome.CAPACITY),
                    jnp.where(
                        consume & trial.cell_failure,
                        int(AtomisticTrialOutcome.CELL),
                        jnp.where(
                            consume & trial.scientific,
                            int(AtomisticTrialOutcome.SCIENTIFIC),
                            jnp.where(
                                consume,
                                int(AtomisticTrialOutcome.ROLLED_BACK),
                                int(AtomisticTrialOutcome.NONE),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return AtomisticRelaxationState(
            optimizer=optimizer,
            reference=state.reference,
            resources=resources,
            candidate_resources=trial.candidate,
            positions=keep(trial.positions, state.positions),
            cell_vectors=keep_optional(trial.cell_vectors, state.cell_vectors),
            energy=keep(trial.energy, state.energy),
            objective=keep(trial.objective, state.objective),
            forces=keep(trial.forces, state.forces),
            cell_covector=keep_optional(trial.cell_covector, state.cell_covector),
            stress=keep_optional(trial.stress, state.stress),
            volume=keep_optional(trial.volume, state.volume),
            projected_residual=keep(projected, state.projected_residual),
            full_force_residual=keep(trial.full_force, state.full_force_residual),
            full_stress_residual=keep(trial.full_stress, state.full_stress_residual),
            projected_converged=keep(projected_ok, state.projected_converged),
            full_converged=keep(full_ok, state.full_converged),
            status=status,
            trial_outcome=jnp.where(iterating, outcome, state.trial_outcome).astype(
                jnp.int32
            ),
            active=state.active,
            capacity_replays=state.capacity_replays
            + (pending & trial.capacity).astype(jnp.int32),
            rejected_trials=state.rejected_trials + (consume & ~accept).astype(jnp.int32),
            reprimes=state.reprimes,
            evaluation_base=state.evaluation_base,
            iteration_base=state.iteration_base,
            plan_identity=state.plan_identity,
            binding_identity=state.binding_identity,
            lineage_identities=state.lineage_identities,
        )

    def run(
        self, state: AtomisticRelaxationState, /, *, maximum_rounds: int
    ) -> AtomisticRelaxationState:
        """Step on device until no group iterates or ``maximum_rounds`` pass."""
        rounds = int(maximum_rounds)
        if rounds < 1:
            raise ValueError("maximum_rounds must be positive.")
        if not isinstance(state, AtomisticRelaxationState):
            raise TypeError("state must be an AtomisticRelaxationState.")

        def condition(carry: tuple[Array, AtomisticRelaxationState]) -> Array:
            count, current = carry
            return (count < rounds) & jnp.any(
                current.status == int(AtomisticRelaxationStatus.ITERATING)
            )

        def body(
            carry: tuple[Array, AtomisticRelaxationState],
        ) -> tuple[Array, AtomisticRelaxationState]:
            count, current = carry
            return count + 1, self.step(current)

        _, final = jax.lax.while_loop(
            condition, body, (jnp.asarray(0, dtype=jnp.int32), state)
        )
        return final

    def grow(
        self, state: AtomisticRelaxationState, /
    ) -> tuple[PreparedAtomisticRelaxation, AtomisticRelaxationState]:
        """Host transaction: grow resources for capacity-pending groups and replay.

        Every capacity-pending group's pending proposal is replayed from its
        accepted resources at successive declared capacity levels until it fits
        (an exhausted ladder terminates that group with ``RESOURCE_EXHAUSTED``).
        The program is rebound to the final bundle, every accepted resource
        state is rebased, and the accepted evidence of every group with an
        accepted iterate is reprimed under the new binding without consuming
        budget.  Pending groups resume ``ITERATING`` with their unchanged
        proposal, which the next `step` evaluates.
        """
        self._require_state(state)
        status = np.asarray(state.status)
        pending = np.flatnonzero(
            status == int(AtomisticRelaxationStatus.CAPACITY_PENDING)
        )
        if pending.size == 0:
            return self, state
        bundle = self.bundle
        exhausted = np.zeros(status.shape, dtype=np.bool_)
        for group in pending:
            parameters = _tree_take(state.optimizer.parameters, int(group))
            reference = _tree_take(state.reference, int(group))
            (positions, vectors), _ = self.plan.chart.realize(parameters, reference)
            geometry, _, _ = trial_geometry(self.program, bundle, positions, vectors)
            bundle, ran_out = replay_capacity(
                bundle, _tree_take(state.resources, int(group)), geometry
            )
            exhausted[group] = ran_out
        program = self.program.with_neighborhood_bundle(bundle)
        runtime = PreparedAtomisticRelaxation(self.plan, program)
        groups = self.plan.group_count
        rebased = _tree_stack(
            [bundle.rebase(_tree_take(state.resources, group)) for group in range(groups)]
        )
        reprime = jax.vmap(runtime._trial)(
            state.optimizer.accepted_parameters, state.reference, rebased
        )
        has_accepted = state.optimizer.has_accepted
        accepted_ok = (
            has_accepted & ~reprime.capacity & ~reprime.scientific & ~reprime.cell_failure
        )

        def refresh(new: Array, old: Array) -> Array:
            return _tree_where(accepted_ok, new, old)

        def refresh_optional(new: Array | None, old: Array | None) -> Array | None:
            if new is None or old is None:
                return old
            return _tree_where(accepted_ok, new, old)

        pending_mask = jnp.asarray(
            status == int(AtomisticRelaxationStatus.CAPACITY_PENDING)
        )
        new_status = jnp.where(
            jnp.asarray(exhausted),
            int(AtomisticRelaxationStatus.RESOURCE_EXHAUSTED),
            jnp.where(
                pending_mask, int(AtomisticRelaxationStatus.ITERATING), state.status
            ),
        ).astype(jnp.int32)
        return runtime, AtomisticRelaxationState(
            optimizer=state.optimizer,
            reference=state.reference,
            resources=rebased,
            candidate_resources=rebased,
            positions=state.positions,
            cell_vectors=state.cell_vectors,
            energy=refresh(reprime.energy, state.energy),
            objective=refresh(reprime.objective, state.objective),
            forces=refresh(reprime.forces, state.forces),
            cell_covector=refresh_optional(reprime.cell_covector, state.cell_covector),
            stress=refresh_optional(reprime.stress, state.stress),
            volume=refresh_optional(reprime.volume, state.volume),
            projected_residual=refresh(
                runtime.layout.site_norm_max(reprime.gradient), state.projected_residual
            ),
            full_force_residual=refresh(reprime.full_force, state.full_force_residual),
            full_stress_residual=refresh(reprime.full_stress, state.full_stress_residual),
            projected_converged=state.projected_converged,
            full_converged=state.full_converged,
            status=new_status,
            trial_outcome=state.trial_outcome,
            active=state.active,
            capacity_replays=state.capacity_replays,
            rejected_trials=state.rejected_trials,
            reprimes=state.reprimes + accepted_ok.astype(jnp.int32),
            evaluation_base=state.evaluation_base,
            iteration_base=state.iteration_base,
            plan_identity=state.plan_identity,
            binding_identity=runtime.prepared_identity,
            lineage_identities=state.lineage_identities,
        )

    def solve(
        self,
        state: AtomisticRelaxationState,
        /,
        *,
        maximum_rounds: int,
        maximum_growths: int = 8,
    ) -> tuple[PreparedAtomisticRelaxation, AtomisticRelaxationState]:
        """Host driver: compiled `run` segments separated by `grow` transactions.

        Every runtime of one layout (any material, parameters or resource
        binding) shares the single compiled `run` entrypoint.
        """
        growths = int(maximum_growths)
        if growths < 0:
            raise ValueError("maximum_growths must be non-negative.")
        self._require_state(state)
        runtime = self
        for _ in range(growths + 1):
            state = _RUN_ROUNDS(runtime, state, int(maximum_rounds))
            pending = np.asarray(state.status) == int(
                AtomisticRelaxationStatus.CAPACITY_PENDING
            )
            if not np.any(pending):
                break
            runtime, state = runtime.grow(state)
        return runtime, state

    def transition_hamiltonian(
        self,
        program: AbstractPreparedAtomisticHamiltonian,
        state: AtomisticRelaxationState,
        /,
    ) -> AtomisticHamiltonianTransitionResult:
        """Host transaction: continue every group under a new prepared Hamiltonian.

        ``program`` must bind the same prepared system and differ either in the
        canonical ``scientific_id`` (a model update) or in its numerical
        realization at the same scientific identity; the record's ``kind``
        says which (`hamiltonian_transition_kind`).  A resource-only rebinding
        is `grow`.  Its neighborhood bundle may be any preparation of its plan.
        Groups that are iterating, converged or capacity-pending are revisable:
        their accepted geometry is kept, resources are initialized (and grown)
        at it under the new program, the accepted iterate is re-evaluated
        without consuming budget and fed as the first evaluation of a freshly
        initialized optimizer, and their status is decided anew.  Pending
        proposals of the old Hamiltonian are discarded.  Terminal and inactive
        groups keep geometry, evidence and status (their resources are
        re-expressed under the new bundle).  Evaluation and iteration totals
        continue across the transition.  If the plan refuses the program, a
        declared capacity ladder cannot hold a revisable group, or any
        reprimed iterate fails, the previous runtime and state are returned
        unchanged with a refused record.
        """
        self._require_state(state)
        kind = hamiltonian_transition_kind(self.program, program)
        previous_id = self.program.scientific_id
        scientific_id = program.scientific_id
        groups = self.plan.group_count
        status = np.asarray(state.status)
        revisable = np.asarray(state.active, dtype=np.bool_) & np.isin(status, _REVISABLE)
        reprimed = revisable & np.asarray(state.optimizer.has_accepted, dtype=np.bool_)

        def refuse(reason: str) -> AtomisticHamiltonianTransitionResult:
            return AtomisticHamiltonianTransitionResult(
                self,
                state,
                AtomisticHamiltonianTransition(
                    np.zeros((groups,), dtype=np.bool_),
                    successful=False,
                    reason=reason,
                    kind=kind,
                    previous_scientific_id=previous_id,
                    scientific_id=scientific_id,
                    previous_binding_id=self.prepared_id,
                    binding_id=self.prepared_id,
                    previous_plan_id=self.plan.plan_id,
                    plan_id=self.plan.plan_id,
                    lineage=state.hamiltonian_lineage,
                ),
            )

        try:
            plan = AtomisticRelaxationPlan(
                program,
                self.plan.chart,
                method=self.plan.method,
                convergence=self.plan.convergence,
                target_pressure=float(self.plan.target_pressure),
                group_count=groups,
            )
        except ValueError as error:
            return refuse(f"The relaxation plan refuses the new Hamiltonian: {error}")
        accepted = state.optimizer.accepted_parameters
        points = [
            self.plan.chart.realize(
                _tree_take(accepted, group), _tree_take(state.reference, group)
            )[0]
            for group in range(groups)
        ]
        bundle = program.neighborhood_bundle
        for group in np.flatnonzero(revisable):
            geometry, _, _ = trial_geometry(program, bundle, *points[group])
            bundle, exhausted = initialize_capacity(bundle, geometry)
            if exhausted:
                return refuse(
                    f"The declared capacities cannot hold group {int(group)} under "
                    "the new Hamiltonian."
                )
        if bundle.prepared_id != program.neighborhood_bundle.prepared_id:
            program = program.with_neighborhood_bundle(bundle)
        runtime = PreparedAtomisticRelaxation(plan, program)
        resources = _tree_stack(
            [
                bundle.initialize(trial_geometry(program, bundle, *points[group])[0])
                for group in range(groups)
            ]
        )
        reprime = jax.vmap(runtime._trial)(accepted, state.reference, resources)
        broken = reprimed & np.asarray(
            reprime.capacity | reprime.scientific | reprime.cell_failure
        )
        if np.any(broken):
            return refuse(
                "Accepted iterates of groups "
                f"{np.flatnonzero(broken).tolist()} fail under the new Hamiltonian."
            )
        assessment = runtime._assess(reprime)
        mask = jnp.asarray(reprimed)
        dtype = jax.tree.leaves(accepted)[0].dtype
        optimizer = plan.method.init(accepted, layout=runtime.layout, value_dtype=dtype)
        optimizer = _step_relaxation_method(
            plan.method,
            optimizer,
            assessment.value,
            assessment.gradient,
            converged=mask & assessment.converged,
            active=mask,
        )
        # The reprime is not consumed work: totals continue unchanged.
        evaluation_base = state.evaluations - optimizer.evaluations
        iteration_base = state.iterations
        lanes = jnp.asarray(revisable)
        budget = (
            optimizer.evaluations + evaluation_base
            >= plan.convergence.maximum_evaluations
        )
        new_status = jnp.where(
            lanes,
            jnp.where(
                mask & (optimizer.status == _SUCCESS),
                int(AtomisticRelaxationStatus.CONVERGED),
                jnp.where(
                    budget,
                    int(AtomisticRelaxationStatus.EVALUATION_LIMIT),
                    int(AtomisticRelaxationStatus.ITERATING),
                ),
            ),
            state.status,
        ).astype(jnp.int32)

        def keep(new: Array, old: Array) -> Array:
            return _tree_where(mask, new, old)

        def keep_optional(new: Array | None, old: Array | None) -> Array | None:
            if new is None or old is None:
                return old
            return _tree_where(mask, new, old)

        record = AtomisticHamiltonianTransition(
            reprimed,
            successful=True,
            reason="",
            kind=kind,
            previous_scientific_id=previous_id,
            scientific_id=scientific_id,
            previous_binding_id=self.prepared_id,
            binding_id=runtime.prepared_id,
            previous_plan_id=self.plan.plan_id,
            plan_id=plan.plan_id,
            lineage=state.hamiltonian_lineage,
        )
        continued = AtomisticRelaxationState(
            optimizer=optimizer,
            reference=state.reference,
            resources=resources,
            candidate_resources=resources,
            positions=keep(reprime.positions, state.positions),
            cell_vectors=keep_optional(reprime.cell_vectors, state.cell_vectors),
            energy=keep(reprime.energy, state.energy),
            objective=keep(reprime.objective, state.objective),
            forces=keep(reprime.forces, state.forces),
            cell_covector=keep_optional(reprime.cell_covector, state.cell_covector),
            stress=keep_optional(reprime.stress, state.stress),
            volume=keep_optional(reprime.volume, state.volume),
            projected_residual=keep(assessment.projected, state.projected_residual),
            full_force_residual=keep(reprime.full_force, state.full_force_residual),
            full_stress_residual=keep(reprime.full_stress, state.full_stress_residual),
            projected_converged=keep(assessment.projected_ok, state.projected_converged),
            full_converged=keep(assessment.full_ok, state.full_converged),
            status=new_status,
            trial_outcome=jnp.where(
                lanes, int(AtomisticTrialOutcome.NONE), state.trial_outcome
            ).astype(jnp.int32),
            active=state.active,
            capacity_replays=state.capacity_replays,
            rejected_trials=state.rejected_trials,
            reprimes=state.reprimes + mask.astype(jnp.int32),
            evaluation_base=evaluation_base.astype(jnp.int32),
            iteration_base=iteration_base.astype(jnp.int32),
            plan_identity=plan.plan_identity,
            binding_identity=runtime.prepared_identity,
            lineage_identities=_identity_rows(
                (*state.hamiltonian_lineage, record.transition_id)
            ),
        )
        return AtomisticHamiltonianTransitionResult(runtime, continued, record)

    # -- accepted observations ------------------------------------------------------

    def accepted_evaluation(
        self, state: AtomisticRelaxationState, group: int, /
    ) -> AtomisticHamiltonianEvaluation:
        """Full program evaluation at a group's accepted iterate and resources.

        Carries term evidence, typed producer fields, atom-energy completeness
        and, for variable cells, the program's own strain derivative/stress.
        """
        self._require_state(state)
        index = int(group)
        if not bool(np.asarray(state.optimizer.has_accepted)[index]):
            raise ValueError("The group has no accepted iterate.")
        resources = _tree_take(state.resources, index)
        # Resources bind the accepted numerical realization, not a fresh chart
        # evaluation that can round differently outside its compiled step.
        positions = state.positions[index]
        vectors = None if state.cell_vectors is None else state.cell_vectors[index]
        wrapped, images = _wrapped(self.system, positions, vectors)
        if vectors is None:
            return self.program.evaluate(wrapped, resources)
        return self.program.evaluate(
            wrapped,
            resources,
            compute_stress=self.plan.chart.variable_cell,
            cell_vectors=vectors,
            image_counts=images,
        )

    def symmetry_evidence(
        self, state: AtomisticRelaxationState, group: int, /
    ) -> AtomisticSymmetryEvidence:
        """Tangent-operator projected residual and symmetry deviation of a group.

        ``residual.projected`` is ``T^T (f, -dE/dH)`` of the actual
        `SymmetryTangentOperator` at the accepted chart state; ``chart_force``
        is the negative chart gradient the optimizer consumed (equal to it for
        the atomic chart and at the anchor of the rigid chart).
        ``deviation`` is the per-atom distance from exact symmetric images.
        """
        self._require_state(state)
        chart = self.plan.chart
        if not isinstance(chart, AtomisticSymmetryChart):
            raise TypeError("symmetry_evidence requires an AtomisticSymmetryChart.")
        index = int(group)
        parameters = _tree_take(state.optimizer.accepted_parameters, index)
        reference = _tree_take(state.reference, index)
        chart_state = chart.state(parameters, reference)
        if isinstance(chart.layout, AtomicSymmetryPlan):
            operator, expansion = atomic_tangent(chart.layout, chart_state)
            atomic_plan = chart.layout
        else:
            operator, expansion = chart.layout.tangent(chart_state)
            atomic_plan = chart.layout.atomic
        atoms = jnp.asarray(chart.system_atoms)
        forces = state.forces[index][atoms]
        covector = (
            jnp.zeros((3, 3), dtype=forces.dtype)
            if state.cell_covector is None
            else state.cell_covector[index]
        )
        residual = symmetry_residuals(operator, forces, -covector)
        gradient = _tree_take(state.optimizer.accepted_gradient, index)
        return AtomisticSymmetryEvidence(
            residual,
            -chart.coordinates(gradient),
            symmetry_residual(atomic_plan, expansion.positions, expansion.cell.vectors),
        )

    # -- stationary sensitivity -------------------------------------------------------

    def sensitivity(
        self,
        state: AtomisticRelaxationState,
        group: int,
        /,
        *,
        program_tangent: AbstractPreparedAtomisticHamiltonian | None = None,
        pressure_tangent: float = 0.0,
        gauge: AtomisticGaugePolicy | None = None,
        policy: AtomisticStationaryPolicy | None = None,
    ) -> AtomisticRelaxationSensitivity:
        """Implicit tangent of a converged minimum along a program/pressure change.

        Admission requires status ``CONVERGED``, a stationary chart gradient on
        the free non-gauge space, declared gauge directions that are numerical
        null directions of the Hessian, and positive curvature (native
        eigensolve on the gauge-deflated Hessian action).  The tangent solves
        ``A x = -Q d_theta g`` with native PCG, where ``A = Q H Q + s (I - Q)``.
        ``objective`` is the envelope derivative ``d f*/d theta``.  Refusals
        return NaN values with their status.
        """
        self._require_state(state)
        gauge_ = AtomisticGaugePolicy() if gauge is None else gauge
        policy_ = AtomisticStationaryPolicy() if policy is None else policy
        index = int(group)
        chart = self.plan.chart
        if (gauge_.translations or gauge_.rotations) and isinstance(
            chart, AtomisticSymmetryChart
        ):
            raise ValueError("Symmetry charts declare no Cartesian gauge directions.")
        if gauge_.rotations and (
            self.system.cell is not None or not isinstance(chart, AtomisticCartesianChart)
        ):
            raise ValueError("A rotation gauge exists only for finite Cartesian charts.")
        parameters = _tree_take(state.optimizer.accepted_parameters, index)
        reference = _tree_take(state.reference, index)
        resources = _tree_take(state.resources, index)
        (positions, vectors), _ = chart.realize(parameters, reference)
        _, _, images = trial_geometry(self.program, self.bundle, positions, vectors)
        flat, unravel = ravel_pytree(parameters)
        dynamic, static = eqx.partition(self.program, eqx.is_inexact_array)
        pressure = jnp.asarray(self.plan.target_pressure, dtype=flat.dtype)

        def objective(point: Array, dynamic_: Any, pressure_: Array) -> Array:
            program = eqx.combine(dynamic_, static)
            (r, h), _ = chart.realize(unravel(point), reference)
            value, _ = self._objective(program, resources, r, h, images, pressure_)
            return value

        masks = chart.site_masks(self.system)
        if isinstance(chart, AtomisticSymmetryChart):
            masks = {**masks, "sites": chart.site_slots >= 0}
        constraint = (
            chart.constraint
            if isinstance(chart, (AtomisticCartesianChart, AtomisticDeformationChart))
            else None
        )
        projectors = (
            None
            if constraint is None or constraint.projectors is None
            else jnp.asarray(constraint.projectors, dtype=flat.dtype)
        )

        def free(vector: Array) -> Array:
            tree = unravel(vector)
            kept = {}
            for name, leaf in tree.items():
                mask = np.asarray(masks[name])
                selected = jnp.where(
                    jnp.asarray(mask).reshape(
                        mask.shape + (1,) * (leaf.ndim - mask.ndim)
                    ),
                    leaf,
                    0.0,
                )
                if name == "atoms" and projectors is not None:
                    selected = contract("aij,aj->ai", projectors, selected)
                kept[name] = selected
            return ravel_pytree(kept)[0]

        basis = _gauge_basis(gauge_, parameters, positions, self.system, flat, free)
        evidence, hvp, project = _stationary_evidence(
            lambda point: objective(point, dynamic, pressure),
            flat,
            free,
            basis,
            policy_,
            minimum=True,
            converged=bool(
                np.asarray(state.status)[index]
                == int(AtomisticRelaxationStatus.CONVERGED)
            ),
        )
        nan_tree = jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), parameters)
        if program_tangent is None:
            tangent_dynamic = jax.tree.map(jnp.zeros_like, dynamic)
        else:
            tangent_dynamic, _ = eqx.partition(program_tangent, eqx.is_inexact_array)
            if jax.tree.structure(tangent_dynamic) != jax.tree.structure(dynamic):
                raise ValueError("program_tangent must share the program structure.")
        pressure_direction = jnp.asarray(float(pressure_tangent), dtype=flat.dtype)
        if pressure_direction != 0.0 and not chart.variable_cell:
            raise ValueError("A pressure tangent requires a variable-cell chart.")

        def gradient_in_theta(dynamic_: Any, pressure_: Array) -> Array:
            return jax.grad(objective)(flat, dynamic_, pressure_)

        _, mixed = jax.jvp(
            gradient_in_theta,
            (dynamic, pressure),
            (tangent_dynamic, pressure_direction),
        )
        _, envelope = jax.jvp(
            lambda dynamic_, pressure_: objective(flat, dynamic_, pressure_),
            (dynamic, pressure),
            (tangent_dynamic, pressure_direction),
        )
        status = int(evidence.status)
        if status != int(AtomisticSensitivityStatus.ADMITTED) or hvp is None:
            return AtomisticRelaxationSensitivity(
                nan_tree,
                jnp.full_like(positions, jnp.nan),
                None if vectors is None else jnp.full_like(vectors, jnp.nan),
                jnp.asarray(jnp.nan, dtype=flat.dtype),
                evidence,
            )
        shift = evidence.shift
        size = flat.shape[0]
        space = ArraySpace((size,), dtype=flat.dtype)

        def action(vector: Array) -> Array:
            projected = project(vector)
            return project(hvp(projected)) + shift * (vector - projected)

        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            transpose_action=action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"self_adjoint": "asserted", "positive_definite": "asserted"},
            ),
            operator_id=f"{self.prepared_id}:deflated-relaxation-hessian",
        )
        result = solve_linear(
            LinearSystem(operator),
            -project(mixed),
            policy=LinearSolvePolicy(
                PCG(),
                tolerance=TolerancePolicy(
                    relative=policy_.solve_relative_tolerance,
                    absolute=policy_.solve_absolute_tolerance,
                    max_steps=policy_.solve_maximum_steps,
                ),
            ),
        )
        solve_status = int(np.asarray(result.status).reshape(-1)[0])
        if solve_status != int(LinearSolveStatus.SUCCESS):
            failed = AtomisticStationaryEvidence(
                evidence.stationarity,
                evidence.gauge_residual,
                evidence.curvatures,
                evidence.negative_count,
                evidence.shift,
                jnp.asarray(solve_status, dtype=jnp.int32),
                jnp.asarray(
                    int(AtomisticSensitivityStatus.SOLVE_FAILED), dtype=jnp.int32
                ),
            )
            return AtomisticRelaxationSensitivity(
                nan_tree,
                jnp.full_like(positions, jnp.nan),
                None if vectors is None else jnp.full_like(vectors, jnp.nan),
                jnp.asarray(jnp.nan, dtype=flat.dtype),
                failed,
            )
        direction = project(jnp.asarray(result.value))
        _, (moved, cell_moved) = jax.jvp(
            lambda point: chart.realize(unravel(point), reference)[0],
            (flat,),
            (direction,),
        )
        admitted = AtomisticStationaryEvidence(
            evidence.stationarity,
            evidence.gauge_residual,
            evidence.curvatures,
            evidence.negative_count,
            evidence.shift,
            jnp.asarray(solve_status, dtype=jnp.int32),
            evidence.status,
        )
        return AtomisticRelaxationSensitivity(
            unravel(direction), moved, cell_moved, envelope, admitted
        )


def _run_rounds(
    runtime: PreparedAtomisticRelaxation, state: AtomisticRelaxationState, rounds: int, /
) -> AtomisticRelaxationState:
    return runtime.run(state, maximum_rounds=rounds)


_RUN_ROUNDS = eqx.filter_jit(_run_rounds)


@final
class AtomisticSymmetryEvidence(StrictModule):
    """Tangent-operator residuals, consumed chart force and symmetry deviation."""

    residual: SymmetryResidual
    chart_force: Array
    deviation: Array


# -- stationary-point admission -------------------------------------------------------


@final
class AtomisticGaugePolicy(StrictModule, NonTrainableState):
    """Declared continuous gauge directions of a stationary point.

    ``translations`` declares uniform translations of mobile atoms (periodic
    or finite); ``rotations`` rigid rotations about the mobile center (finite
    Cartesian charts only).  Declared directions are verified as numerical
    Hessian null directions before any claim; linearly dependent directions
    (for example rotation about a linear molecule's axis) are dropped.
    """

    translations: bool = eqx.field(static=True)
    rotations: bool = eqx.field(static=True)

    def __init__(self, *, translations: bool = False, rotations: bool = False) -> None:
        if not isinstance(translations, bool) or not isinstance(rotations, bool):
            raise TypeError("Gauge flags must be bools.")
        self.translations = translations
        self.rotations = rotations


@final
class AtomisticStationaryPolicy(StrictModule, NonTrainableState):
    """Tolerances of stationary-point admission and the tangent solve."""

    stationarity_tolerance: float = eqx.field(static=True)
    gauge_tolerance: float = eqx.field(static=True)
    curvature_tolerance: float = eqx.field(static=True)
    eigen_steps: int = eqx.field(static=True)
    solve_relative_tolerance: float = eqx.field(static=True)
    solve_absolute_tolerance: float = eqx.field(static=True)
    solve_maximum_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        stationarity_tolerance: float = 1.0e-6,
        gauge_tolerance: float = 1.0e-6,
        curvature_tolerance: float = 1.0e-8,
        eigen_steps: int = 200,
        solve_relative_tolerance: float = 1.0e-10,
        solve_absolute_tolerance: float = 1.0e-12,
        solve_maximum_steps: int = 1000,
    ) -> None:
        self.stationarity_tolerance = _positive(
            stationarity_tolerance, "stationarity_tolerance"
        )
        self.gauge_tolerance = _positive(gauge_tolerance, "gauge_tolerance")
        self.curvature_tolerance = _nonnegative(
            curvature_tolerance, "curvature_tolerance"
        )
        self.solve_relative_tolerance = _positive(
            solve_relative_tolerance, "solve_relative_tolerance"
        )
        self.solve_absolute_tolerance = _positive(
            solve_absolute_tolerance, "solve_absolute_tolerance"
        )
        if int(eigen_steps) < 1 or int(solve_maximum_steps) < 1:
            raise ValueError("Iteration budgets must be positive.")
        self.eigen_steps = int(eigen_steps)
        self.solve_maximum_steps = int(solve_maximum_steps)


@final
class AtomisticStationaryEvidence(StrictModule):
    """Stationarity, gauge, inertia and solve evidence of one stationary claim.

    ``curvatures`` are the lowest eigenvalues of the free, gauge-deflated
    Hessian action (ascending); ``negative_count`` counts those below
    ``-curvature_tolerance``.  ``shift`` is the positive value assigned to the
    deflated directions.  ``status`` is an `AtomisticSensitivityStatus`.
    """

    stationarity: Array
    gauge_residual: Array
    curvatures: Array
    negative_count: Array
    shift: Array
    solve_status: Array
    status: Array

    @property
    def admitted(self) -> bool:
        return int(self.status) == int(AtomisticSensitivityStatus.ADMITTED)


@final
class AtomisticRelaxationSensitivity(StrictModule):
    """Implicit tangent of a relaxed minimum and its admission evidence."""

    parameters: PyTree[Array]
    positions: Array
    cell_vectors: Array | None
    objective: Array
    evidence: AtomisticStationaryEvidence

    @property
    def admitted(self) -> bool:
        return self.evidence.admitted


def _orthonormal_columns(matrix: Array, /) -> Array:
    """Orthonormal basis of the numerically independent columns of ``matrix``."""
    if matrix.shape[1] == 0:
        return matrix
    left, singular, _ = economy_svd(matrix)
    values = np.asarray(singular)
    keep = values > 1.0e-8 * max(float(np.max(values, initial=0.0)), 1.0)
    return left[:, jnp.asarray(np.flatnonzero(keep))]


def _gauge_basis(
    gauge: AtomisticGaugePolicy,
    parameters: dict[str, Array],
    positions: Array,
    system: PreparedAtomisticSystem,
    flat: Array,
    free: Any,
    /,
) -> Array:
    """Orthonormal free parameter-space gauge columns ``(n, k)``, ``k <= 6``."""
    if not (gauge.translations or gauge.rotations):
        return jnp.zeros((flat.shape[0], 0), dtype=flat.dtype)
    weight = _mobile_device(system).astype(flat.dtype)[:, None]
    columns = []
    zero = jax.tree.map(jnp.zeros_like, parameters)
    if gauge.translations:
        for axis in range(3):
            direction = jnp.zeros_like(positions).at[:, axis].set(1.0) * weight
            columns.append(free(ravel_pytree({**zero, "atoms": direction})[0]))
    if gauge.rotations:
        center = jnp.sum(weight * positions, axis=0) / jnp.maximum(jnp.sum(weight), 1.0)
        relative = (positions - center) * weight
        for axis in range(3):
            omega = jnp.zeros((3,), dtype=flat.dtype).at[axis].set(1.0)
            direction = jnp.cross(omega, relative) * weight
            columns.append(free(ravel_pytree({**zero, "atoms": direction})[0]))
    return _orthonormal_columns(jnp.stack(columns, axis=1))


def _stationary_evidence(
    objective: Any,
    point: Array,
    free: Any,
    basis: Array,
    policy: AtomisticStationaryPolicy,
    /,
    *,
    minimum: bool,
    converged: bool,
) -> tuple[AtomisticStationaryEvidence, Any, Any]:
    """Admission evidence for a minimum (``minimum``) or first-order saddle.

    ``free`` is the orthogonal projector onto free coordinates and ``basis``
    orthonormal gauge columns inside that free space.
    """
    dtype = point.dtype
    size = point.shape[0]

    def project(vector: Array) -> Array:
        masked = free(vector)
        if basis.shape[1] == 0:
            return masked
        return masked - basis @ (basis.T @ masked)

    gradient, hvp = jax.linearize(jax.grad(objective), point)
    stationarity = jnp.linalg.norm(project(gradient))
    gauge_residual = (
        jnp.max(jnp.linalg.norm(jax.vmap(hvp, in_axes=1, out_axes=1)(basis), axis=0))
        if basis.shape[1]
        else jnp.zeros((), dtype=dtype)
    )
    count = 1 if minimum else 2
    nan = jnp.full((count,), jnp.nan, dtype=dtype)

    def refuse(reason: AtomisticSensitivityStatus) -> AtomisticStationaryEvidence:
        return AtomisticStationaryEvidence(
            stationarity,
            gauge_residual,
            nan,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(int(reason), dtype=jnp.int32),
        )

    if not converged:
        return refuse(AtomisticSensitivityStatus.NOT_CONVERGED), None, project
    if not float(stationarity) <= policy.stationarity_tolerance:
        return refuse(AtomisticSensitivityStatus.NOT_STATIONARY), None, project
    if not float(gauge_residual) <= policy.gauge_tolerance:
        return refuse(AtomisticSensitivityStatus.GAUGE_NOT_NULL), None, project
    space = ArraySpace((size,), dtype=dtype)
    properties = OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "asserted"}
    )
    projected_hessian = FunctionLinearOperator(
        lambda vector: project(hvp(project(vector))),
        source=space,
        target=space,
        properties=properties,
        operator_id="deflated-stationary-hessian",
    )
    bounds = estimate_spectral_bounds(projected_hessian, max_dimension=min(size, 64))
    shift = 1.0 + 2.0 * jnp.maximum(jnp.abs(bounds.upper), jnp.abs(bounds.lower))

    def deflated(vector: Array) -> Array:
        projected = project(vector)
        return project(hvp(projected)) + shift * (vector - projected)

    eigen = eigensolve(
        Eigenproblem(
            FunctionLinearOperator(
                deflated,
                source=space,
                target=space,
                properties=properties,
                operator_id="shifted-stationary-hessian",
            )
        ),
        policy=EigenSolvePolicy(
            count=min(count, size),
            which="smallest-algebraic",
            max_steps=policy.eigen_steps,
        ),
    )
    if not bool(np.all(np.asarray(eigen.successful))):
        return refuse(AtomisticSensitivityStatus.EIGEN_FAILED), None, project
    values = jnp.sort(jnp.asarray(eigen.eigenvalues).reshape((-1,)))
    tolerance = policy.curvature_tolerance
    negative = jnp.sum(values < -tolerance).astype(jnp.int32)
    if minimum:
        correct = bool(values[0] > tolerance)
    else:
        correct = bool(values[0] < -tolerance) and (
            values.shape[0] < 2 or bool(values[1] > tolerance)
        )
    status = (
        AtomisticSensitivityStatus.ADMITTED
        if correct
        else AtomisticSensitivityStatus.WRONG_INERTIA
    )
    evidence = AtomisticStationaryEvidence(
        stationarity,
        gauge_residual,
        values,
        negative,
        jnp.asarray(shift, dtype=dtype),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    return evidence, (hvp if correct else None), project


def atomistic_stationary_inertia(
    objective: Any,
    point: ArrayLike,
    /,
    *,
    free: ArrayLike,
    gauge_basis: ArrayLike | None = None,
    policy: AtomisticStationaryPolicy | None = None,
    saddle: bool = False,
    converged: bool = True,
) -> AtomisticStationaryEvidence:
    """HVP/inertia evidence of a flat stationary point on the non-gauge space.

    ``objective`` maps a flat parameter vector to a scalar; ``free`` masks the
    free coordinates and ``gauge_basis`` ``(n, k)`` declares gauge columns
    (orthonormalized here).  A minimum needs positive lowest curvature; a
    first-order saddle exactly one negative and the next positive.
    """
    flat = jnp.asarray(point)
    if flat.ndim != 1:
        raise ValueError("point must be a flat vector.")
    mask = jnp.asarray(free, dtype=flat.dtype)
    if mask.shape != flat.shape:
        raise ValueError("free must match point.")
    basis = (
        jnp.zeros((flat.shape[0], 0), dtype=flat.dtype)
        if gauge_basis is None
        else jnp.asarray(gauge_basis, dtype=flat.dtype)
    )
    if basis.ndim != 2 or basis.shape[0] != flat.shape[0] or basis.shape[1] > 6:
        raise ValueError("gauge_basis must have shape (n, k) with k <= 6.")
    basis = _orthonormal_columns(basis * mask[:, None])
    evidence, _, _ = _stationary_evidence(
        objective,
        flat,
        lambda vector: mask * vector,
        basis,
        AtomisticStationaryPolicy() if policy is None else policy,
        minimum=not saddle,
        converged=converged,
    )
    return evidence


__all__ = [
    "AbstractAtomisticRelaxationChart",
    "atomistic_stationary_inertia",
    "AtomisticCartesianChart",
    "AtomisticCartesianTrial",
    "AtomisticCellChartKind",
    "AtomisticCoordinateConstraint",
    "AtomisticDeformationChart",
    "AtomisticGaugePolicy",
    "AtomisticHamiltonianTransition",
    "AtomisticHamiltonianTransitionKind",
    "AtomisticHamiltonianTransitionResult",
    "AtomisticRelaxationConvergence",
    "AtomisticRelaxationPlan",
    "AtomisticRelaxationSensitivity",
    "AtomisticRelaxationState",
    "AtomisticRelaxationStatus",
    "AtomisticSensitivityStatus",
    "AtomisticStationaryEvidence",
    "AtomisticStationaryPolicy",
    "AtomisticStrainSubspace",
    "AtomisticSymmetryChart",
    "AtomisticSymmetryEvidence",
    "AtomisticTrialOutcome",
    "evaluate_cartesian_trial",
    "hamiltonian_transition_kind",
    "initialize_capacity",
    "PreparedAtomisticRelaxation",
    "program_energy",
    "replay_capacity",
    "trial_geometry",
]

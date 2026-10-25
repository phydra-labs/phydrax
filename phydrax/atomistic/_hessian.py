#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coordinate Hessian actions of native scalar atomistic potentials.

This module owns only the atomistic domain binding. Differentiation is the
existing pure boundary ``atomistic_energy_derivatives`` (conservative forces
``F = -grad E``), linearized once through ``phydrax.linalg.prepare_linearization``;
the Hessian action is ``H v = -DF[v]``. Repeated forward, transpose and adjoint
actions reuse that one prepared linearization. Admission is the model's declared
``DerivativeContract`` evaluated by the canonical ``require`` under an explicit
``RegularityPolicy``; regularity is never inferred from shapes or finite values.

Declared non-smooth loci are named regularity conditions. The binding resolves
the loci it can certify at the actual geometry and cell from the bound graph:

* ``ATOMISTIC_NONCOINCIDENT_SUPPORT`` (a regularity ``support``): every candidate
  pair has a positive distance;
* ``atomistic_edge_distance_locus(r)``: no candidate pair lies exactly at the
  join radius ``r``. A join only matters for derivative orders above the declared
  continuity, so it is checked only for admissions below ``SMOOTH``;
* ``"singular-part-ignored"``: resolved exactly when every declared locus of the
  admission is resolved at the point (the singular part lives on those loci).

A resolved violation invalidates the affected case: its values are NaN and its
derivatives are poisoned. Conditions this binding cannot certify (for example a
model-internal activation or selection-rank locus) stay explicit in
``AtomisticHessianOperator.unresolved_conditions``; they are not claimed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._admissibility import (
    AdmissibilityReason,
    guard_derivative_validity,
    refuse_derivative_dependencies,
)
from .._differentiation import (
    ComponentAuthority,
    DERIVATIVE_UNSUPPORTED,
    DerivativeAdmission,
    DerivativeContract,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
    RegularityPolicy,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model._ports import ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState, partition_parameters
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    JacobianLinearOperator,
    LinearCapabilityError,
    LinearizationPolicy,
    MaterializationPolicy,
    OperatorCapabilities,
    OperatorProperties,
    prepare_linearization,
)
from ..typing import checked, PRNGKey
from ..units import derived_unit, UnitDefinition
from ._conditioning import (
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
)
from ._execution_binding import atomistic_executable_signature
from ._graph import (
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
)
from ._neighborhood_bundle import AtomisticNeighborhoodRole
from ._potential import (
    AbstractAtomisticModel,
    AbstractAtomisticPotential,
    AbstractPreparedAtomisticModel,
)
from ._prediction import atomistic_energy_derivatives, AtomisticProvenance
from ._types import AtomisticBatch


ATOMISTIC_NONCOINCIDENT_SUPPORT = "atomistic-support:noncoincident-positions"
_LOCUS_PREFIX = "atomistic-nonsmooth-locus:"
_EDGE_DISTANCE_PREFIX = f"{_LOCUS_PREFIX}edge-distance="
_SINGULAR_PART = "singular-part-ignored"

ATOMISTIC_HESSIAN_REQUEST = DifferentiationRequest(
    (DerivativeSurface.INPUT,), order=2, authority=ComponentAuthority.MODEL
)
"""The coordinate Hessian itself: second order in atomic positions."""

_DEFAULT_REQUEST = AtomisticObservableRequest(
    (AtomisticObservable.HESSIAN_ACTION,), derivatives=(ATOMISTIC_HESSIAN_REQUEST,)
)
_SUPPORTED_OBSERVABLES = (
    AtomisticObservable.ENERGY,
    AtomisticObservable.FORCES,
    AtomisticObservable.HESSIAN_ACTION,
    AtomisticObservable.HESSIAN_BLOCKS,
)
_RESPONSE_SURFACES = frozenset(
    {DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER}
)
_IDENTITY_COMPONENTS = (
    "model-definition",
    "model-revision",
    "derivative-admission",
    "batch",
    "graph",
    "geometry",
    "cell",
    "conditioning",
    "key",
    "image-counts",
    "resource-epochs",
)


def atomistic_edge_distance_locus(radius: float, /) -> str:
    """Name the non-smooth locus where a candidate pair distance equals ``radius``.

    Models declare it as a ``DerivativeRegularity`` condition for each cutoff,
    envelope or switching join whose continuity is below the regularity's
    smooth pieces. ``radius`` is in the model's length unit.
    """
    value = float(radius)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "A non-smooth edge-distance locus needs a finite positive radius."
        )
    return f"{_EDGE_DISTANCE_PREFIX}{value!r}"


def _edge_locus_radius(condition: str, /) -> float | None:
    if not condition.startswith(_EDGE_DISTANCE_PREFIX):
        return None
    return float(condition.removeprefix(_EDGE_DISTANCE_PREFIX))


def _definition(
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel, /
) -> AbstractAtomisticModel:
    if isinstance(potential, AbstractPreparedAtomisticModel):
        return potential.model
    if isinstance(potential, AbstractAtomisticModel):
        return potential
    raise TypeError("potential must implement a native atomistic model contract.")


def _hessian_admissions(
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
    request: AtomisticObservableRequest,
    policy: RegularityPolicy,
    /,
) -> tuple[DerivativeContract, tuple[DerivativeAdmission, ...]]:
    """Refuse every request that is not a scalar-potential coordinate response."""
    definition = _definition(potential)
    name = type(potential).__name__
    if not isinstance(definition, AbstractAtomisticPotential):
        raise ValueError(
            f"{DERIVATIVE_UNSUPPORTED}: {name} is a property-only model; its reports "
            "have no scalar potential energy and therefore no Hessian."
        )
    if not potential.capabilities.conservative_energy:
        raise ValueError(
            f"{DERIVATIVE_UNSUPPORTED}: {name} does not define a conservative scalar "
            "potential; its forces are not the gradient of an energy."
        )
    request.require_supported(_SUPPORTED_OBSERVABLES, "AtomisticHessianOperator")
    if (
        AtomisticObservable.HESSIAN_ACTION not in request.observables
        and AtomisticObservable.HESSIAN_BLOCKS not in request.observables
    ):
        raise ValueError("A Hessian binding requests Hessian actions or blocks.")
    if not any(
        value.surfaces == (DerivativeSurface.INPUT,) and value.order == 2
        for value in request.derivatives
    ):
        raise ValueError(
            "The request must contain the coordinate Hessian derivative request "
            "DifferentiationRequest((DerivativeSurface.INPUT,), order=2)."
        )
    for value in request.derivatives:
        surfaces = frozenset(value.surfaces)
        if DerivativeSurface.INPUT not in surfaces or not surfaces <= _RESPONSE_SURFACES:
            names = tuple(surface.value for surface in value.surfaces)
            raise ValueError(
                f"{DERIVATIVE_UNSUPPORTED}: derivatives on {names!r} have no admitted "
                "Hessian route; only coordinate (INPUT) Hessian responses and their "
                "MODEL_PARAMETER derivatives are implemented (strain, cell and "
                "conditioning responses need their own admission)."
            )
        if value.order < 2:
            raise ValueError("A Hessian derivative request has order at least two.")
        if DerivativeSurface.MODEL_PARAMETER in surfaces and value.order < 3:
            raise ValueError(
                "A parameter derivative of a Hessian action is at least order three."
            )
    contract = potential.derivative_contract
    if contract is None:
        raise ValueError(
            f"{DERIVATIVE_UNSUPPORTED}: {name} declares no derivative contract; "
            "Hessian regularity is never inferred from shapes or finite values."
        )
    if not isinstance(contract, DerivativeContract):
        raise TypeError("derivative_contract must be a DerivativeContract or None.")
    return contract, tuple(
        contract.require(value, policy=policy) for value in request.derivatives
    )


def _request(request: AtomisticObservableRequest | None, /) -> AtomisticObservableRequest:
    if request is None:
        return _DEFAULT_REQUEST
    if not isinstance(request, AtomisticObservableRequest):
        raise TypeError("request must be an AtomisticObservableRequest or None.")
    return request


def _policy(policy: RegularityPolicy | None, /) -> RegularityPolicy:
    if policy is None:
        return RegularityPolicy()
    if not isinstance(policy, RegularityPolicy):
        raise TypeError("policy must be a RegularityPolicy or None.")
    return policy


@final
class _AtomisticForceMap(StrictModule):
    """Conservative forces of one bound potential and geometry, as a PyTree callable.

    Every numerical capture (model leaves, cell, conditioning, key, image counts)
    is a dynamic leaf, so closure conversion hoists them and an enclosing
    parameter derivative reaches them.
    """

    __strict_contract__ = True

    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel
    payload: AtomisticGraphPayload
    cell_vectors: Array | None
    conditioning: AtomisticConditioning | None
    key: PRNGKey | None
    image_counts: Array | None

    def __call__(self, positions: Array, /) -> tuple[Array, tuple[Array, Array, Array]]:
        derivatives = atomistic_energy_derivatives(
            self.potential,
            self.payload,
            positions,
            cell_vectors=self.cell_vectors,
            conditioning=self.conditioning,
            key=self.key,
            image_counts=self.image_counts,
        )
        if derivatives.forces is None:
            raise RuntimeError("The conservative force derivative was not returned.")
        return derivatives.forces, (
            derivatives.energy,
            derivatives.successful,
            derivatives.overflow,
        )


def _case_minimum(values: Array, cases: Array, active: Array, count: int, /) -> Array:
    infinity = jnp.asarray(jnp.inf, dtype=values.dtype)
    return jnp.full((count,), infinity).at[cases].min(jnp.where(active, values, infinity))


def _domain_evidence(
    contract: DerivativeContract,
    admissions: tuple[DerivativeAdmission, ...],
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
    payload: AtomisticGraphPayload,
    positions: Array,
    cell_vectors: Array | None,
    image_counts: Array | None,
    successful: Array,
    overflow: Array,
    /,
) -> tuple[tuple[DerivativeAdmission, ...], tuple[str, ...]]:
    """Attach per-case point-domain evidence of each admission at the bound state."""
    cases = payload.case_count
    failure = jnp.where(
        overflow,
        int(AdmissibilityReason.CAPACITY_INSUFFICIENT),
        jnp.where(successful, 0, int(AdmissibilityReason.NONFINITE)),
    ).astype(jnp.int32)
    outside = int(AdmissibilityReason.OUTSIDE_SUPPORT)
    regularity = contract.regularity
    support = None if regularity is None else regularity.support
    unresolved: set[str] = set()
    geometry: list[tuple[Array, Array, Array]] = []

    def edges() -> tuple[Array, Array, Array]:
        # Evidence only: the bound geometry is inspected, never differentiated.
        if not geometry:
            graph = payload.graph(
                cutoff=potential.configuration.cutoff,
                positions=jax.lax.stop_gradient(positions),
                cell_vectors=(
                    None if cell_vectors is None else jax.lax.stop_gradient(cell_vectors)
                ),
                image_counts=image_counts,
                selection=potential.requirements.neighborhood(
                    AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
                ).selection,
            )
            geometry.append(
                (
                    jnp.asarray(graph.graph.edges["distance"])[:, 0],
                    graph.candidate_mask,
                    graph.topology.edge_cases,
                )
            )
        return geometry[0]

    support_status = jnp.zeros((cases,), dtype=jnp.int32)
    if support == ATOMISTIC_NONCOINCIDENT_SUPPORT:
        distance, active, edge_cases = edges()
        nearest = _case_minimum(distance, edge_cases, active, cases)
        support_status = jnp.where(nearest > 0.0, 0, outside).astype(jnp.int32)
    elif support is not None:
        unresolved.add(support)
    evidenced = []
    for admission in admissions:
        status = failure | support_status
        smooth = all(level is GradientLevel.SMOOTH for level in admission.levels)
        loci_resolved = True
        locus_count = 0
        for condition in admission.conditions:
            if condition == _SINGULAR_PART:
                continue
            radius = _edge_locus_radius(condition)
            is_locus = condition.startswith(_LOCUS_PREFIX)
            locus_count += int(is_locus)
            if is_locus and smooth:
                # A join of order above this admission's continuity is harmless.
                continue
            if radius is None:
                unresolved.add(condition)
                loci_resolved = loci_resolved and not is_locus
                continue
            if radius > payload.topology.search_radius:
                # Pairs at the join radius may lie outside the candidate set.
                unresolved.add(condition)
                loci_resolved = False
                continue
            distance, active, edge_cases = edges()
            margin = _case_minimum(
                jnp.abs(distance - jnp.asarray(radius, dtype=distance.dtype)),
                edge_cases,
                active,
                cases,
            )
            status = status | jnp.where(margin > 0.0, 0, outside).astype(jnp.int32)
        if _SINGULAR_PART in admission.conditions and (
            not loci_resolved or locus_count == 0
        ):
            unresolved.add(_SINGULAR_PART)
        count = len(admission.request.surfaces)
        evidenced.append(
            admission.with_runtime_evidence(
                jnp.broadcast_to((status == 0)[:, None], (cases, count)),
                jnp.broadcast_to(status[:, None], (cases, count)),
            )
        )
    return tuple(evidenced), tuple(sorted(unresolved))


def _inexact_leaves(tree: object, /) -> tuple[Array, ...]:
    return tuple(leaf for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf))


@final
class AtomisticHessianBlockSelection(StrictModule, NonTrainableState):
    """Explicit atom-slot rows and columns of bounded per-case dense Hessian blocks.

    Slots index the case-local atom capacity and apply to every case. Blocks are
    case-major ``(case, output_atom, 3, input_atom, 3)``: cases are independent
    systems, so no cross-case zeros are ever formed.
    """

    output_atoms: tuple[int, ...] = eqx.field(static=True)
    input_atoms: tuple[int, ...] = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self, output_atoms: Sequence[int], input_atoms: Sequence[int], /
    ) -> None:
        rows = tuple(output_atoms)
        columns = tuple(input_atoms)
        for name, values in (("output_atoms", rows), ("input_atoms", columns)):
            if not values:
                raise ValueError(f"{name} must select at least one atom slot.")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise TypeError(f"{name} must contain integer atom slots.")
            if any(int(value) < 0 for value in values):
                raise ValueError(f"{name} must contain nonnegative atom slots.")
            if len(set(int(value) for value in values)) != len(values):
                raise ValueError(f"{name} must not repeat atom slots.")
        rows_ = tuple(int(value) for value in rows)
        columns_ = tuple(int(value) for value in columns)
        self.output_atoms = rows_
        self.input_atoms = columns_
        self.selection_id = canonical_fingerprint(
            {
                "kind": "atomistic-hessian-block-selection",
                "output_atoms": rows_,
                "input_atoms": columns_,
            }
        )


@final
class AtomisticHessianOperator(AbstractLinearOperator):
    """Matrix-free coordinate Hessian ``H = -DF`` of one bound scalar potential.

    Built by ``bind_atomistic_hessian`` (pure) or ``prepare_atomistic_hessian``
    (host identity boundary). Source vectors are coordinate tangents shaped like
    the batch positions; target vectors are energy-gradient covectors in the
    potential's output dtype. Every action reuses the one prepared force
    linearization. Derivatives through unadmitted inputs raise
    ``DERIVATIVE_UNSUPPORTED`` at trace time. Dense full materialization is
    refused; use ``blocks`` with an explicit bounded selection.
    """

    force_jacobian: JacobianLinearOperator
    admissions: tuple[DerivativeAdmission, ...]
    atom_mask: Array
    action_valid: Array
    response_valid: Array
    refused: tuple[Array, ...]
    admitted: tuple[Array, ...]
    hessian_unit: UnitDefinition
    observables: tuple[AtomisticObservable, ...] = eqx.field(static=True)
    unresolved_conditions: tuple[str, ...] = eqx.field(static=True)
    refusal_message: str = eqx.field(static=True)

    def __init__(
        self,
        force_jacobian: JacobianLinearOperator,
        /,
        *,
        admissions: tuple[DerivativeAdmission, ...],
        unresolved_conditions: tuple[str, ...],
        observables: tuple[AtomisticObservable, ...],
        atom_mask: Array,
        refused: tuple[Array, ...],
        admitted: tuple[Array, ...],
        refusal_message: str,
        hessian_unit: UnitDefinition,
        operator_id: str,
    ) -> None:
        source = force_jacobian.source
        target = force_jacobian.target
        if not isinstance(source, ArraySpace) or not isinstance(target, ArraySpace):
            raise TypeError("Atomistic Hessian spaces must be array spaces.")
        if source.shape != target.shape or len(source.shape) != 3:
            raise ValueError("Atomistic Hessian spaces must be (case, atom, 3) arrays.")
        if not admissions:
            raise ValueError("An atomistic Hessian binding needs its admissions.")
        base = next(
            (
                value
                for value in admissions
                if value.request.surfaces == (DerivativeSurface.INPUT,)
                and value.request.order == 2
            ),
            None,
        )
        if base is None or base.runtime_valid is None:
            raise ValueError(
                "The coordinate Hessian admission and its evidence are required."
            )
        responses = tuple(value.runtime_valid for value in admissions)
        if any(value is None for value in responses):
            raise ValueError("Every Hessian admission must carry point-domain evidence.")
        successful = force_jacobian.linearization.auxiliary[1]
        action_valid = successful & jnp.all(base.runtime_valid, axis=1)
        response_valid = action_valid
        for value in admissions:
            if value.runtime_valid is not None:
                response_valid = response_valid & jnp.all(value.runtime_valid, axis=1)
        # Schwarz symmetry is claimed only for a classical (SMOOTH) second
        # derivative; an almost-everywhere Hessian makes no structural claim.
        symmetric = source.compatible(target) and all(
            level is GradientLevel.SMOOTH for level in base.levels
        )
        properties = (
            OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            )
            if symmetric
            else OperatorProperties()
        )
        self.source = source
        self.target = target
        self.properties = properties
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.force_jacobian = force_jacobian
        self.admissions = admissions
        self.atom_mask = atom_mask
        self.action_valid = action_valid
        self.response_valid = response_valid
        self.refused = refused
        self.admitted = admitted
        self.hessian_unit = hessian_unit
        self.observables = observables
        self.unresolved_conditions = unresolved_conditions
        self.refusal_message = refusal_message
        self.batch_shape = ()
        self.operator_id = operator_id

    @property
    def positions(self) -> Array:
        """The coordinates of the one prepared linearization point."""
        return self.force_jacobian.linearization.point

    @property
    def forces(self) -> Array:
        """Conservative forces ``-grad E`` at the bound point."""
        return self.force_jacobian.linearization.primal

    @property
    def energy(self) -> Array:
        return self.force_jacobian.linearization.auxiliary[0]

    @property
    def successful(self) -> Array:
        """Per-case energy/force evaluation success, before domain evidence."""
        return self.force_jacobian.linearization.auxiliary[1]

    @property
    def overflow(self) -> Array:
        """Per-case graph capacity failure, distinct from scientific failure."""
        return self.force_jacobian.linearization.auxiliary[2]

    def admission(self, request: DifferentiationRequest, /) -> DerivativeAdmission:
        """Return the evidenced admission of one requested derivative."""
        for value in self.admissions:
            if value.request == request:
                return value
        raise ValueError("The derivative was not requested for this Hessian binding.")

    def _finish(self, value: Array, /) -> Array:
        if self.refused:
            value = refuse_derivative_dependencies(
                value, self.refused, message=self.refusal_message
            )
        if self.admitted:
            value = guard_derivative_validity(
                value,
                self.response_valid[:, None, None],
                dependencies=self.admitted,
                message="The atomistic Hessian response is invalid at this point.",
            )
        scale = jnp.where(
            self.action_valid,
            jnp.ones((), dtype=value.dtype),
            jnp.asarray(jnp.nan, dtype=value.dtype),
        )
        return value * scale[:, None, None]

    def mv(self, vector: Array, /) -> Array:
        """``H v = -DF[v]`` on the prepared force linearization."""
        return self._finish(-self.force_jacobian.mv(vector))

    def transpose_mv(self, vector: Array, /) -> Array:
        """``H^T w = -DF^T[w]`` through the prepared linearization's transpose."""
        return self._finish(-self.force_jacobian.transpose_mv(vector))

    def adjoint_mv(self, vector: Array, /) -> Array:
        return self._finish(-self.force_jacobian.adjoint_mv(vector))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "A full atomistic Hessian is never materialized; request bounded per-case "
            "blocks through AtomisticHessianOperator.blocks."
        )

    def blocks(
        self,
        selection: AtomisticHessianBlockSelection,
        policy: MaterializationPolicy,
        /,
    ) -> AtomisticObservableValue:
        """Materialize selected per-case dense blocks under an explicit bound.

        One shared block action applies ``3 * len(input_atoms)`` coordinate
        tangents to every case at once (cases are independent systems); only the
        selected output rows are retained. Entries and workspace are admitted by
        ``policy`` before execution. A case is available only when its Hessian
        action is valid and every selected slot holds an active atom; otherwise
        its block is NaN.
        """
        if AtomisticObservable.HESSIAN_BLOCKS not in self.observables:
            raise ValueError("Hessian blocks were not requested for this binding.")
        if not isinstance(selection, AtomisticHessianBlockSelection):
            raise TypeError("selection must be an AtomisticHessianBlockSelection.")
        if not isinstance(policy, MaterializationPolicy):
            raise TypeError("policy must be a MaterializationPolicy.")
        source, target = self.source, self.target
        if not isinstance(source, ArraySpace) or not isinstance(target, ArraySpace):
            raise TypeError("Atomistic Hessian spaces must be array spaces.")
        cases, atoms, _ = source.shape
        rows = np.asarray(selection.output_atoms, dtype=np.int64)
        columns = np.asarray(selection.input_atoms, dtype=np.int64)
        if np.any(rows >= atoms) or np.any(columns >= atoms):
            raise ValueError("Selected atom slots exceed the case atom capacity.")
        tangent_count = 3 * columns.size
        policy.require(cases * rows.size * 3 * tangent_count, target.dtype)
        policy.require_workspace(
            tangent_count
            * (source.size * source.dtype.itemsize + target.size * target.dtype.itemsize)
        )
        basis = np.zeros((atoms, 3, columns.size, 3), dtype=source.dtype)
        for index, atom in enumerate(columns):
            basis[atom, :, index, :] = np.eye(3, dtype=source.dtype)
        coordinates = jnp.broadcast_to(
            jnp.asarray(basis.reshape((1, atoms * 3, tangent_count))),
            (cases, atoms * 3, tangent_count),
        ).reshape((source.size, tangent_count))
        response = self.mv_block(coordinates).reshape((cases, atoms, 3, columns.size, 3))
        values = response[:, rows]
        selected = jnp.concatenate(
            (self.atom_mask[:, rows], self.atom_mask[:, columns]), axis=1
        )
        available = self.action_valid & jnp.all(selected, axis=1)
        values = jnp.where(
            available[:, None, None, None, None],
            values,
            jnp.asarray(jnp.nan, dtype=values.dtype),
        )
        axes = ("x", "y", "z")
        port = ValuePort(
            "atomistic.hessian-block",
            event_shape=(rows.size, 3, columns.size, 3),
            component_ids=tuple(
                f"out{row}.{left}:in{column}.{right}"
                for row in rows
                for left in axes
                for column in columns
                for right in axes
            ),
            representation="second-coordinate-derivative-of-case-energy",
            dimensions=(self.hessian_unit.dimension,) * (rows.size * 9 * columns.size),
            frame_id="input-cartesian",
            normalization_id="identity",
        )
        return AtomisticObservableValue(
            AtomisticObservable.HESSIAN_BLOCKS,
            port,
            values,
            available,
            level=AtomisticConditionLevel.CASE,
            case_count=cases,
            atom_capacity=atoms,
            unit=self.hessian_unit,
        )


@checked
def bind_atomistic_hessian(
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
    payload: AtomisticGraphPayload,
    positions: ArrayLike,
    /,
    *,
    request: AtomisticObservableRequest | None = None,
    policy: RegularityPolicy | None = None,
    cell_vectors: ArrayLike | None = None,
    conditioning: AtomisticConditioning | None = None,
    key: PRNGKey | None = None,
    image_counts: ArrayLike | None = None,
    linearization: LinearizationPolicy | None = None,
) -> AtomisticHessianOperator:
    """Bind a coordinate Hessian on a layout-bound graph payload; pure and traceable.

    This is the numerical binding for use inside ``jit``/``grad``, including
    HVP-matching parameter derivatives: it performs no host synchronization and
    never hashes numerical values. Admission is static. The derivative of an
    action through model parameters requires an admitted request containing
    ``DerivativeSurface.MODEL_PARAMETER`` (mixed order three for a coordinate HVP
    loss); through positions it requires an admitted coordinate order of at least
    three. Every other enclosing derivative (cell, conditioning, model state, or
    unrequested surfaces) raises ``DERIVATIVE_UNSUPPORTED`` at trace time.
    """
    request_ = _request(request)
    policy_ = _policy(policy)
    contract, admissions = _hessian_admissions(potential, request_, policy_)
    definition = _definition(potential)
    precision = potential.precision
    coordinate_dtype = np.dtype(precision.coordinate_dtype)
    coordinate = jnp.asarray(positions, dtype=coordinate_dtype)
    shape = payload.positions.shape
    if coordinate.shape != shape:
        raise ValueError("positions must have the payload shape (case, atom, 3).")
    vectors = (
        None
        if cell_vectors is None
        else jnp.asarray(cell_vectors, dtype=coordinate_dtype)
    )
    counts = None if image_counts is None else jnp.asarray(image_counts)
    source = ArraySpace(shape, dtype=coordinate_dtype)
    target = ArraySpace(shape, dtype=np.dtype(precision.output_dtype))
    # This is an executable label, not the host point/revision binding.
    # The latter remains complete in atomistic_hessian_identity; never decode
    # dynamic prepared identity words while constructing a traced linearization.
    executable = atomistic_executable_signature(
        potential,
        layout_id=canonical_fingerprint(
            {
                "kind": "atomistic-hessian-model-layout",
                "architecture": definition.architecture_id,
                "method": definition.method_id,
                "contract": contract.contract_id,
            }
        ),
    )
    method = "store" if linearization is None else linearization.rematerialization
    linearization_id = canonical_fingerprint(
        {
            "kind": "atomistic-hessian-force-linearization",
            "architecture": definition.architecture_id,
            "method": definition.method_id,
            "model_executable": executable.signature_id,
            "precision": precision.policy_id,
            "layout": payload.layout_id,
            "graph_schema": payload.topology.topology_id,
            "execution": payload.execution.plan_id,
            "request": request_.request_id,
            "rematerialization": method,
        }
    )
    prepared = prepare_linearization(
        _AtomisticForceMap(
            potential,
            payload,
            vectors,
            conditioning,
            key,
            counts,
        ),
        coordinate,
        source=source,
        target=target,
        has_aux=True,
        policy=linearization,
        linearization_id=linearization_id,
    )
    _, successful, overflow = prepared.auxiliary
    resolved_cell = payload.cells if vectors is None else vectors
    evidenced, unresolved = _domain_evidence(
        contract,
        admissions,
        potential,
        payload,
        coordinate,
        vectors,
        counts,
        successful,
        overflow,
    )
    coordinate_response = any(
        DerivativeSurface.INPUT in value.request.surfaces and value.request.order >= 3
        for value in evidenced
    )
    parameter_response = any(
        DerivativeSurface.MODEL_PARAMETER in value.request.surfaces for value in evidenced
    )
    parameters, model_state, _ = partition_parameters(potential)
    refused: list[Array] = list(_inexact_leaves(model_state))
    admitted: list[Array] = []
    refused_names = ["model state"]
    for selected, leaves, label in (
        (coordinate_response, (coordinate,), "coordinates beyond second order"),
        (parameter_response, _inexact_leaves(parameters), "model parameters"),
    ):
        if selected:
            admitted.extend(leaf for leaf in leaves if eqx.is_inexact_array(leaf))
        else:
            refused.extend(leaf for leaf in leaves if eqx.is_inexact_array(leaf))
            refused_names.append(label)
    refused.extend(_inexact_leaves(resolved_cell))
    refused.extend(_inexact_leaves(conditioning))
    refused_names.extend(("cell", "conditioning"))
    scale = payload.scale
    hessian_unit = derived_unit(
        f"{scale.energy_unit.symbol}/{scale.length_unit.symbol}^2",
        ((scale.energy_unit, 1), (scale.length_unit, -2)),
    )
    return AtomisticHessianOperator(
        JacobianLinearOperator(prepared, operator_id=f"{linearization_id}:forces"),
        admissions=evidenced,
        unresolved_conditions=unresolved,
        observables=request_.observables,
        atom_mask=payload.atom_mask,
        refused=tuple(refused),
        admitted=tuple(admitted),
        refusal_message=(
            "this atomistic Hessian binding admits no derivative of its actions "
            f"through {', '.join(refused_names)}; request the corresponding "
            "higher-order DifferentiationRequest to admit it."
        ),
        hessian_unit=hessian_unit,
        operator_id=canonical_fingerprint(
            {
                "kind": "atomistic-hessian-operator",
                "linearization": linearization_id,
                "contract": contract.contract_id,
                "policy": (
                    policy_.allow_almost_everywhere,
                    policy_.allow_undeclared,
                ),
                "source": source.space_id,
                "target": target.space_id,
            }
        ),
    )


@final
class AtomisticHessianIdentity(StrictModule, NonTrainableState):
    """Host identity of every input a prepared Hessian binding depends on.

    Components: model definition and numeric revision, derivative admission,
    batch, graph content/layout/execution/topology epoch, actual coordinates,
    cell, conditioning payload, random key, image counts and external resource
    epochs.
    """

    components: tuple[tuple[str, str], ...] = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(self, components: Sequence[tuple[str, str]], /) -> None:
        values = tuple((str(name), str(value)) for name, value in components)
        if tuple(name for name, _ in values) != _IDENTITY_COMPONENTS:
            raise ValueError("Atomistic Hessian identity components are incomplete.")
        self.components = values
        self.identity_id = canonical_fingerprint(
            {"kind": "atomistic-hessian-identity", "components": values}
        )

    def stale_components(self, current: AtomisticHessianIdentity, /) -> tuple[str, ...]:
        """Names of the components whose identity differs from ``current``."""
        if not isinstance(current, AtomisticHessianIdentity):
            raise TypeError("current must be an AtomisticHessianIdentity.")
        return tuple(
            name
            for (name, value), (_, other) in zip(
                self.components, current.components, strict=True
            )
            if value != other
        )


def _host_payload(
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    payload: AtomisticGraphPayload | None,
    /,
) -> AtomisticGraphPayload:
    """The batch's one-batch layout payload, or a supplied payload bound to it."""
    bound = (
        AtomisticGraphLayout.for_batch(
            batch, execution, cutoff=potential.configuration.cutoff
        ).prepare(batch)
        if payload is None
        else payload.admit(batch)
    )
    if bound.execution.plan_id != execution.plan_id:
        raise ValueError("The graph payload was prepared under another execution plan.")
    return bound


def _host_array_identity(kind: str, value: ArrayLike | None, /) -> str:
    if value is None:
        return "absent"
    return canonical_fingerprint(
        {"kind": kind, "values": array_tree_fingerprint(np.asarray(value))}
    )


def atomistic_hessian_identity(
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    positions: ArrayLike | None = None,
    payload: AtomisticGraphPayload | None = None,
    request: AtomisticObservableRequest | None = None,
    policy: RegularityPolicy | None = None,
    cell_vectors: ArrayLike | None = None,
    conditioning: AtomisticConditioning | None = None,
    key: PRNGKey | None = None,
    image_counts: ArrayLike | None = None,
    resource_epochs: ArrayLike | None = None,
) -> AtomisticHessianIdentity:
    """Host boundary: content-address the scientific inputs of one binding.

    Every numerical input must be concrete; call outside ``jit``/``grad``.
    ``resource_epochs`` are integer epochs of external resources (for example a
    neighborhood bundle) the binding was prepared against.
    """
    request_ = _request(request)
    policy_ = _policy(policy)
    contract, admissions = _hessian_admissions(potential, request_, policy_)
    payload_ = _host_payload(potential, batch, execution, payload)
    provenance = AtomisticProvenance(
        potential,
        batch,
        execution,
        payload=payload_,
        conditioning=conditioning,
        key=key,
        image_counts=image_counts,
    )
    dtype = np.dtype(potential.precision.coordinate_dtype)
    coordinate = np.asarray(
        batch.positions if positions is None else positions, dtype=dtype
    )
    if coordinate.shape != batch.positions.shape:
        raise ValueError("positions must have the batch shape (case, atom, 3).")
    coordinate = np.where(np.asarray(batch.atom_mask)[:, :, None], coordinate, 0.0)
    cell = batch.cells if cell_vectors is None else cell_vectors
    epochs = None
    if resource_epochs is not None:
        epochs = np.asarray(resource_epochs)
        if not np.issubdtype(epochs.dtype, np.integer):
            raise TypeError("resource_epochs must be integer resource epochs.")
    realization = provenance.realization_id
    values = {
        "model-definition": canonical_fingerprint(
            {
                "kind": "atomistic-hessian-model-definition",
                "class": f"{type(potential).__module__}.{type(potential).__qualname__}",
                "architecture": provenance.architecture_id,
                "method": provenance.method_id,
                "precision": provenance.precision_id,
                "realization": realization,
            }
        ),
        "model-revision": provenance.potential_revision_id,
        "derivative-admission": canonical_fingerprint(
            {
                "kind": "atomistic-hessian-admission",
                "contract": contract.contract_id,
                "request": request_.request_id,
                "policy": (
                    policy_.allow_almost_everywhere,
                    policy_.allow_undeclared,
                ),
                "admissions": tuple(
                    (
                        tuple(level.value for level in value.levels),
                        value.conditions,
                    )
                    for value in admissions
                ),
            }
        ),
        "batch": batch.batch_id,
        "graph": canonical_fingerprint(
            {
                "kind": "atomistic-hessian-graph",
                "content": provenance.graph_topology_id,
                "layout": provenance.graph_layout_id,
                "execution": execution.plan_id,
                "epoch": array_tree_fingerprint(np.asarray(payload_.topology.epoch)),
            }
        ),
        "geometry": _host_array_identity("atomistic-hessian-geometry", coordinate),
        "cell": _host_array_identity(
            "atomistic-hessian-cell",
            None if cell is None else np.asarray(cell, dtype=dtype),
        ),
        "conditioning": provenance.conditioning_id or "absent",
        "key": provenance.random_realization_id or "absent",
        "image-counts": provenance.image_counts_id or "absent",
        "resource-epochs": _host_array_identity("atomistic-resource-epochs", epochs),
    }
    return AtomisticHessianIdentity(
        tuple((name, values[name]) for name in _IDENTITY_COMPONENTS)
    )


@final
class PreparedAtomisticHessian(StrictModule, NonTrainableState):
    """A reusable Hessian action bound to its exact scientific identity.

    Apply ``operator`` repeatedly at the prepared state. Before reuse at a later
    state, ``require_current`` recomputes the identity and refuses any change of
    model definition/revision, admission, batch, graph content/epoch, geometry,
    cell, conditioning, key, image counts or resource epochs.
    """

    operator: AtomisticHessianOperator
    identity: AtomisticHessianIdentity
    request: AtomisticObservableRequest
    policy: RegularityPolicy

    def __init__(
        self,
        operator: AtomisticHessianOperator,
        identity: AtomisticHessianIdentity,
        request: AtomisticObservableRequest,
        policy: RegularityPolicy,
        /,
    ) -> None:
        if not isinstance(operator, AtomisticHessianOperator):
            raise TypeError("operator must be an AtomisticHessianOperator.")
        if not isinstance(identity, AtomisticHessianIdentity):
            raise TypeError("identity must be an AtomisticHessianIdentity.")
        self.operator = operator
        self.identity = identity
        self.request = request
        self.policy = policy

    def require_current(
        self,
        potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        positions: ArrayLike | None = None,
        payload: AtomisticGraphPayload | None = None,
        cell_vectors: ArrayLike | None = None,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        image_counts: ArrayLike | None = None,
        resource_epochs: ArrayLike | None = None,
    ) -> AtomisticHessianOperator:
        """Return the prepared operator only if every bound input is unchanged."""
        current = atomistic_hessian_identity(
            potential,
            batch,
            execution,
            positions=positions,
            payload=payload,
            request=self.request,
            policy=self.policy,
            cell_vectors=cell_vectors,
            conditioning=conditioning,
            key=key,
            image_counts=image_counts,
            resource_epochs=resource_epochs,
        )
        stale = self.identity.stale_components(current)
        if stale:
            raise ValueError(
                f"Stale atomistic Hessian binding: {', '.join(stale)} changed; "
                "prepare the Hessian again for the current state."
            )
        return self.operator


def prepare_atomistic_hessian(
    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    positions: ArrayLike | None = None,
    payload: AtomisticGraphPayload | None = None,
    request: AtomisticObservableRequest | None = None,
    policy: RegularityPolicy | None = None,
    cell_vectors: ArrayLike | None = None,
    conditioning: AtomisticConditioning | None = None,
    key: PRNGKey | None = None,
    image_counts: ArrayLike | None = None,
    resource_epochs: ArrayLike | None = None,
    linearization: LinearizationPolicy | None = None,
) -> PreparedAtomisticHessian:
    """Host boundary: prepare one reusable, identity-bound Hessian action.

    Graph capacity failure is refused here (grow the resource through its owner
    and prepare again); scientific per-case failure and point-domain violations
    remain per-case evidence on the operator.
    """
    request_ = _request(request)
    policy_ = _policy(policy)
    # Refuse unsupported models and requests before any host graph work.
    _hessian_admissions(potential, request_, policy_)
    payload_ = _host_payload(potential, batch, execution, payload)
    coordinate = batch.positions if positions is None else positions
    identity = atomistic_hessian_identity(
        potential,
        batch,
        execution,
        positions=coordinate,
        payload=payload_,
        request=request_,
        policy=policy_,
        cell_vectors=cell_vectors,
        conditioning=conditioning,
        key=key,
        image_counts=image_counts,
        resource_epochs=resource_epochs,
    )
    operator = bind_atomistic_hessian(
        potential,
        payload_,
        jnp.asarray(coordinate),
        request=request_,
        policy=policy_,
        cell_vectors=cell_vectors,
        conditioning=conditioning,
        key=key,
        image_counts=image_counts,
        linearization=linearization,
    )
    overflow = np.asarray(jax.device_get(operator.overflow))
    if np.any(overflow):
        cases = tuple(int(value) for value in np.flatnonzero(overflow))
        raise ValueError(
            f"Graph capacity failed for cases {cases}; the Hessian of a truncated "
            "graph is not the model's Hessian. Grow the neighbor/topology capacity "
            "through its resource owner and prepare again."
        )
    return PreparedAtomisticHessian(operator, identity, request_, policy_)


__all__ = [
    "ATOMISTIC_HESSIAN_REQUEST",
    "ATOMISTIC_NONCOINCIDENT_SUPPORT",
    "AtomisticHessianBlockSelection",
    "AtomisticHessianIdentity",
    "AtomisticHessianOperator",
    "PreparedAtomisticHessian",
    "atomistic_edge_distance_locus",
    "atomistic_hessian_identity",
    "bind_atomistic_hessian",
    "prepare_atomistic_hessian",
]

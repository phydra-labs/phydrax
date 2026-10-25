#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomistic training objectives as explicit numerator/support reductions.

Every objective returns an additive numerator and its support over one payload
(`AtomisticObjectiveTerm`); the shared training kernel merges numerators and
supports across accumulation microbatches (and a distribution reduction across
ranks) before normalizing, so unequal batch or rank means are never averaged.
Padding cases carry zero weight and are inert. A failed model or resource
evaluation of any valid case poisons the numerator (and forces positive
support), so the whole update is rejected, not just a conveniently omitted loss
component.

Residual forms (`AtomisticResidualLoss`), physical scales, reductions
(structure-, atom- or component-balanced), objective-weight schedules
(`AtomisticObjectiveSchedule`) and the label/loss/accumulation precisions
(`AtomisticTrainingPrecision`) are declared, never inferred. Reference energies
(`fit_atomistic_reference_energies`) use the native rank-revealing
pseudoinverse and retain rank, singular-value and residual evidence.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, assert_never, final, Literal, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from .._differentiation import DerivativeRoute, ObjectiveKind, RegularityPolicy
from .._fingerprint import canonical_fingerprint
from .._jax_config import x64_enabled
from .._model._ports import ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import RankPolicy
from ..linalg._dense_pseudoinverse import apply_pseudoinverse, factor_pseudoinverse
from ..typing import ConvertibleToArray, parse
from ..units import derived_unit, UnitDefinition
from ._conditioning import (
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from ._graph import AtomisticGraphPayload
from ._hessian import AtomisticHessianOperator, bind_atomistic_hessian
from ._model_inputs import AtomisticGraphInputs
from ._neighborhood_bundle import AtomisticNeighborhoodRole
from ._potential import (
    AbstractAtomisticModel,
    AbstractAtomisticPotential,
    AtomisticStressConvention,
)
from ._prediction import atomistic_energy_derivatives, AtomisticEnergyDerivatives
from ._training_data import AtomisticFieldBinding, AtomisticStructureLoader
from ._types import AtomisticScaleContract


AtomisticResidualKind = Literal["squared", "absolute", "huber", "euclidean"]
AtomisticScheduleKind = Literal["constant", "linear", "cosine", "piecewise"]
AtomisticEnergyReduction = Literal[
    "per-atom-structure", "total-structure", "per-atom-atom-weighted"
]
AtomisticForceReduction = Literal["component", "atom", "structure"]
AtomisticTensorReduction = Literal["component", "structure"]
AtomisticLabelLevel = Literal["structure", "atom"]
_RealDType = Literal["float32", "float64"]

_T = TypeVar("_T")


def _name(value: str, label: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string.")
    return value


def _finite_positive(value: float | None, label: str, /) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{label} must be a real number or None.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive.")
    return resolved


def _real_dtype(value: Any, label: str, /) -> str:
    name = parse(np.dtype(value).name, _RealDType, label)
    if name == "float64" and not x64_enabled():
        raise ValueError(
            f"{label}=float64 needs JAX x64; enable it or request float32 explicitly."
        )
    return name


# ----------------------------------------------------------------- precision


@final
class AtomisticTrainingPrecision(StrictModule, NonTrainableState):
    """Explicit loss and accumulation precisions, separate from label storage.

    Labels keep their stored (binding) dtype through ingest; predictions keep
    the model's declared output precision. Both are cast exactly once into
    ``loss_dtype`` where residuals form; ``accumulation_dtype`` is the kernel's
    numerator/support/gradient accumulation precision. A lower loss dtype than
    a label dtype is an explicit numerical choice recorded in this identity,
    never an automatic cast to coordinate precision.
    """

    loss_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)

    def __init__(
        self, *, loss_dtype: Any = "float64", accumulation_dtype: Any = "float64"
    ) -> None:
        loss = _real_dtype(loss_dtype, "loss_dtype")
        accumulation = _real_dtype(accumulation_dtype, "accumulation_dtype")
        self.loss_dtype = loss
        self.accumulation_dtype = accumulation
        self.precision_id = canonical_fingerprint(
            {
                "kind": "atomistic-training-precision",
                "loss": loss,
                "accumulation": accumulation,
            }
        )


# ------------------------------------------------------------- residual losses


@final
class AtomisticResidualLoss(StrictModule, NonTrainableState):
    """Pointwise residual penalty ``rho`` of a scaled residual ``r``.

    ``squared``: ``r**2``. ``absolute``: ``|r|``. ``huber``: ``r**2 / 2`` for
    ``|r| <= delta`` and ``delta * (|r| - delta / 2)`` otherwise (the source
    Huber convention). ``euclidean``: the Euclidean norm of a vector residual
    (one value per atom or row, for example the per-atom force-error norm);
    it is defined only on vector quantities and has no component reduction.
    """

    kind: AtomisticResidualKind = eqx.field(static=True)
    delta: float | None = eqx.field(static=True)
    loss_id: str = eqx.field(static=True)

    def __init__(
        self, kind: AtomisticResidualKind = "squared", /, *, delta: float | None = None
    ) -> None:
        kind_ = parse(kind, AtomisticResidualKind, "kind")
        delta_ = _finite_positive(delta, "delta")
        if (kind_ == "huber") != (delta_ is not None):
            raise ValueError("Exactly the huber residual loss declares delta.")
        self.kind = kind_
        self.delta = delta_
        self.loss_id = canonical_fingerprint(
            {"kind": "atomistic-residual-loss", "form": kind_, "delta": delta_}
        )

    @property
    def vector_only(self) -> bool:
        return self.kind == "euclidean"

    def pointwise(self, residual: Array, /) -> Array:
        """Elementwise penalty; refused for the vector-only Euclidean form."""
        match self.kind:
            case "squared":
                return residual * residual
            case "absolute":
                return jnp.abs(residual)
            case "huber":
                if self.delta is None:
                    raise RuntimeError("Huber loss lacks its delta.")
                delta = jnp.asarray(self.delta, dtype=residual.dtype)
                magnitude = jnp.abs(residual)
                return jnp.where(
                    magnitude <= delta,
                    0.5 * residual * residual,
                    delta * (magnitude - 0.5 * delta),
                )
            case "euclidean":
                raise ValueError(
                    "The euclidean residual has no componentwise penalty; use an atom "
                    "or structure reduction of a vector quantity."
                )
            case _:
                assert_never(self.kind)

    def vector(self, residual: Array, /) -> Array:
        """Penalty of each trailing-axis vector: sum of components or its norm."""
        if self.kind != "euclidean":
            return jnp.sum(self.pointwise(residual), axis=-1)
        squared = jnp.sum(residual * residual, axis=-1)
        tiny = jnp.asarray(jnp.finfo(residual.dtype).tiny, dtype=residual.dtype)
        positive = squared > 0.0
        return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, tiny)), 0.0)


# ------------------------------------------------------------------ schedules


@final
class AtomisticObjectiveSchedule(StrictModule, NonTrainableState):
    """Objective weight as a function of the committed-update count.

    The argument is the kernel's accepted cursor, so a weight is constant over
    an accumulation window and never advances on rejected or skipped attempts.
    ``linear`` and ``cosine`` move from ``start`` to ``end`` over ``steps``
    committed updates and then hold ``end``; ``piecewise`` holds ``values[k]``
    on ``boundaries[k-1] <= step < boundaries[k]``. Weights are nonnegative;
    signed terms belong to a declared objective, never to a schedule.
    """

    kind: AtomisticScheduleKind = eqx.field(static=True)
    start: float = eqx.field(static=True)
    end: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    boundaries: tuple[int, ...] = eqx.field(static=True)
    values: tuple[float, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: AtomisticScheduleKind,
        /,
        *,
        start: float = 1.0,
        end: float | None = None,
        steps: int = 0,
        boundaries: Sequence[int] = (),
        values: Sequence[float] = (),
    ) -> None:
        kind_ = parse(kind, AtomisticScheduleKind, "kind")
        start_ = float(start)
        end_ = start_ if end is None else float(end)
        steps_ = int(steps)
        bounds = tuple(int(value) for value in boundaries)
        levels = tuple(float(value) for value in values)
        for value in (start_, end_, *levels):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Objective schedule weights are finite and nonnegative.")
        match kind_:
            case "constant":
                if end is not None or steps_ or bounds or levels:
                    raise ValueError("A constant schedule takes only start.")
            case "linear" | "cosine":
                if steps_ <= 0 or bounds or levels:
                    raise ValueError(f"A {kind_} schedule needs positive steps only.")
            case "piecewise":
                if not levels or len(levels) != len(bounds) + 1:
                    raise ValueError(
                        "A piecewise schedule needs one more value than boundaries."
                    )
                if any(value <= 0 for value in bounds) or any(
                    right <= left for left, right in zip(bounds, bounds[1:])
                ):
                    raise ValueError(
                        "Piecewise boundaries are strictly increasing positive steps."
                    )
                start_, end_ = levels[0], levels[-1]
            case _:
                assert_never(kind_)
        self.kind = kind_
        self.start = start_
        self.end = end_
        self.steps = steps_
        self.boundaries = bounds
        self.values = levels
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "atomistic-objective-schedule",
                "form": kind_,
                "start": start_,
                "end": end_,
                "steps": steps_,
                "boundaries": bounds,
                "values": levels,
            }
        )

    @classmethod
    def constant(cls, value: float = 1.0, /) -> AtomisticObjectiveSchedule:
        return cls("constant", start=value)

    @classmethod
    def linear(
        cls, start: float, end: float, /, *, steps: int
    ) -> AtomisticObjectiveSchedule:
        return cls("linear", start=start, end=end, steps=steps)

    @classmethod
    def cosine(
        cls, start: float, end: float, /, *, steps: int
    ) -> AtomisticObjectiveSchedule:
        return cls("cosine", start=start, end=end, steps=steps)

    @classmethod
    def piecewise(
        cls, boundaries: Sequence[int], values: Sequence[float], /
    ) -> AtomisticObjectiveSchedule:
        return cls("piecewise", boundaries=boundaries, values=values)

    @property
    def minimum(self) -> float:
        """Smallest weight the schedule takes at any committed update."""
        if self.kind == "piecewise":
            return min(self.values)
        return min(self.start, self.end)

    def __call__(self, step: Array, /, *, dtype: Any) -> Array:
        count = jnp.asarray(step).astype(dtype)
        match self.kind:
            case "constant":
                return jnp.asarray(self.start, dtype=dtype)
            case "linear":
                fraction = jnp.clip(count / self.steps, 0.0, 1.0)
                return jnp.asarray(self.start, dtype) + (self.end - self.start) * fraction
            case "cosine":
                fraction = jnp.clip(count / self.steps, 0.0, 1.0)
                return jnp.asarray(self.end, dtype) + (self.start - self.end) * 0.5 * (
                    1.0 + jnp.cos(jnp.pi * fraction)
                )
            case "piecewise":
                index = jnp.searchsorted(
                    jnp.asarray(
                        self.boundaries, dtype=jnp.int64 if x64_enabled() else jnp.int32
                    ),
                    jnp.asarray(step),
                    side="right",
                )
                return jnp.asarray(self.values, dtype=dtype)[index]
            case _:
                assert_never(self.kind)


# --------------------------------------------------------------------- batches


@final
class AtomisticTrainingBatch(StrictModule):
    """One host-prepared payload: its layout-bound graph payload and labels.

    ``graph`` is the ``AtomisticGraphPayload`` of the bucket's
    ``AtomisticGraphLayout``: every batch of one layout shares one static tree
    definition, so repeated payloads reuse one compiled step; material case and
    particle identities are its dynamic leaves. Cases follow the loader
    payload. ``case_valid`` is false on padding cases, which repeat a real
    record and carry zero ``case_weight``. Structure-level labels have shape
    ``(case, *trailing)`` and atom-level labels ``(case, atom, *trailing)``;
    ``available`` is the stored availability (nonfinite available values are
    retained, never treated as missing). ``stress_cells`` marks cases whose
    periodic axes and finite nonsingular cell admit a cell-volume-normalized
    stress. ``draw_index``/``case_epoch`` are the loader's semantic draw
    identities.
    """

    graph: AtomisticGraphPayload
    conditioning: AtomisticConditioning | None
    case_valid: Array
    case_weight: Array
    draw_index: Array
    case_epoch: Array
    stress_cells: Array
    labels: tuple[Array, ...]
    available: tuple[Array, ...]
    label_slots: tuple[str, ...] = eqx.field(static=True)
    label_levels: tuple[AtomisticLabelLevel, ...] = eqx.field(static=True)

    @property
    def case_count(self) -> int:
        return self.graph.case_count

    @property
    def atom_capacity(self) -> int:
        return self.graph.atom_capacity

    @property
    def atom_mask(self) -> Array:
        return self.graph.atom_mask

    @property
    def atom_cases(self) -> Array:
        """Flat case-major owning case of every atom row."""
        return self.graph.atom_cases

    @property
    def atom_counts(self) -> Array:
        return jnp.sum(self.graph.atom_mask, axis=1, dtype=jnp.int32)

    @property
    def species(self) -> Array:
        return self.graph.atom_type_ids

    def label(self, slot: str, /) -> tuple[Array, Array]:
        if slot not in self.label_slots:
            raise KeyError(f"The training batch carries no label slot {slot!r}.")
        index = self.label_slots.index(slot)
        return self.labels[index], self.available[index]

    def level(self, slot: str, /) -> AtomisticLabelLevel:
        return self.label_levels[self.label_slots.index(slot)]


@final
class AtomisticHostLabels(StrictModule, NonTrainableState):
    """Host label view of one payload, used for fitted normalization moments."""

    case_valid: np.ndarray
    atom_mask: np.ndarray
    species: np.ndarray
    values: Mapping[str, np.ndarray]
    available: Mapping[str, np.ndarray]

    def label(self, slot: str, /) -> tuple[np.ndarray, np.ndarray]:
        return self.values[slot], self.available[slot]

    @property
    def atom_counts(self) -> np.ndarray:
        return np.sum(self.atom_mask, axis=1)


# ---------------------------------------------------------------- objective terms


@final
class AtomisticObjectiveTerm(StrictModule):
    """Additive numerator, support and failure evidence of one objective.

    Failures count valid cases; ``capacity_failures`` are graph/resource
    capacity overflows, distinct from ``scientific_failures`` (unsuccessful or
    nonfinite model evaluations).
    """

    numerator: Array
    support: Array
    capacity_failures: Array
    scientific_failures: Array


def _term(
    numerator: Array,
    support: Array,
    capacity: Array,
    scientific: Array,
    /,
) -> AtomisticObjectiveTerm:
    """Poison the update on any failed valid case; zero support stays inert."""
    capacity_count = jnp.sum(capacity, dtype=jnp.int32)
    scientific_count = jnp.sum(scientific & ~capacity, dtype=jnp.int32)
    failed = (capacity_count + scientific_count) > 0
    nan = jnp.asarray(jnp.nan, dtype=numerator.dtype)
    poison = jnp.where(failed, nan, jnp.ones((), dtype=numerator.dtype))
    return AtomisticObjectiveTerm(
        numerator=numerator * poison,
        support=jnp.where(failed, jnp.maximum(support, 1.0), support).astype(
            numerator.dtype
        ),
        capacity_failures=capacity_count,
        scientific_failures=scientific_count,
    )


# ------------------------------------------------------------------ reductions


class AbstractAtomisticObjectiveReduction(StrictModule, NonTrainableState):
    """Device-side placement of objective terms and parameters.

    ``bind_parameters`` receives the PARAMETER lane where it enters every
    objective evaluation; ``reduce`` receives each complete local term
    (weighted numerator, support and failure counts) before it leaves the
    objective function, in training and validation alike, so every diagnostic
    is the reduced term; ``map_step`` wraps every compiled kernel step and
    validation evaluation (the place to bind a data axis). The reduction
    identity enters the training continuation identity.
    """

    reduction_id: eqx.AbstractVar[str]

    def bind_parameters(self, parameters: PyTree[Any], /) -> PyTree[Any]:
        return parameters

    def reduce(self, term: AtomisticObjectiveTerm, /) -> AtomisticObjectiveTerm:
        return term

    def map_step(self, step: Callable[..., _T], /) -> Callable[..., _T]:
        return step

    @property
    def local_shards(self) -> int:
        """Data shards of every loader payload that this process places."""
        return 1

    def assemble(self, batches: tuple[AtomisticTrainingBatch, ...], /) -> Any:
        """Place one prepared batch per local data shard as the step argument."""
        if len(batches) != 1:
            raise ValueError("The local objective reduction places exactly one shard.")
        return batches[0]


@final
class AtomisticLocalObjectiveReduction(AbstractAtomisticObjectiveReduction):
    """Single-placement identity reduction."""

    reduction_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.reduction_id = canonical_fingerprint(
            {"kind": "atomistic-local-objective-reduction"}
        )


# ------------------------------------------------------------- model evaluation


def atomistic_training_derivatives(
    model: AbstractAtomisticModel,
    batch: AtomisticTrainingBatch,
    /,
    *,
    forces: bool,
    stress_cases: Array | None = None,
) -> AtomisticEnergyDerivatives:
    """Energy (and requested force/stress) derivatives on the batch graph payload."""
    return atomistic_energy_derivatives(
        model,
        batch.graph,
        batch.graph.positions,
        compute_forces=forces,
        compute_stress=stress_cases is not None,
        stress_case_mask=stress_cases,
        conditioning=batch.conditioning,
    )


def atomistic_training_hessian(
    model: AbstractAtomisticModel,
    batch: AtomisticTrainingBatch,
    /,
    *,
    request: AtomisticObservableRequest,
    policy: RegularityPolicy | None = None,
) -> AtomisticHessianOperator:
    """Coordinate Hessian ``H = -DF`` of ``model`` on the batch graph payload (pure).

    Admission is the model's declared derivative contract for ``request``; a
    parameter derivative of an action needs its mixed-order request.
    """
    return bind_atomistic_hessian(
        model,
        batch.graph,
        batch.graph.positions,
        request=request,
        policy=policy,
        conditioning=batch.conditioning,
    )


def atomistic_training_outputs(
    model: AbstractAtomisticModel,
    batch: AtomisticTrainingBatch,
    request: AtomisticObservableRequest,
    /,
) -> tuple[AtomisticModelOutputs, Array, Array]:
    """Typed model outputs on the batch graph payload, with case success/capacity."""
    payload = batch.graph
    graph = payload.graph(
        cutoff=float(model.configuration.cutoff),
        positions=jnp.asarray(payload.positions, dtype=model.precision.coordinate_dtype),
        cell_vectors=payload.cells,
        selection=model.requirements.neighborhood(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        ).selection,
    )
    outputs = model.evaluate_graph(
        AtomisticGraphInputs(
            graph,
            payload.scale,
            species_kind=model.capabilities.species_kind,
            conditioning=batch.conditioning,
            key=None,
        ),
        request,
    )
    return outputs, outputs.successful & graph.valid, graph.overflow


def _masked(value: Array, mask: Array, /) -> Array:
    return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))


def case_atom_sum(values: Array, /) -> Array:
    """Per-case sum over the atom axis of ``(case, atom, ...)`` values."""
    return jnp.sum(values, axis=1)


# ------------------------------------------------------------------ objectives


class AbstractAtomisticObjective(StrictModule, NonTrainableState):
    """One genuine training objective: an additive numerator and its support.

    ``scale`` divides residuals in physical label units; ``None`` fits it from
    the training source (``fitted_scale``) before optimization and binds it in
    the training normalization identity. ``schedule`` multiplies the numerator
    by the committed-update weight. ``term`` is pure and device-side; ``admit``
    validates the model and the loader bindings of every label slot on the host.
    Objectives whose evaluation needs a trainable auxiliary component (for
    example an embedding projector) declare ``requires_component``; the
    component is supplied to the trainer, never held by the objective.
    """

    name: eqx.AbstractVar[str]
    scale: eqx.AbstractVar[float | None]
    schedule: eqx.AbstractVar[AtomisticObjectiveSchedule]
    objective_id: eqx.AbstractVar[str]

    @property
    def kind(self) -> ObjectiveKind:
        return ObjectiveKind.DATA_FIT

    @property
    def route(self) -> DerivativeRoute:
        return DerivativeRoute.DIRECT

    @property
    def requires_component(self) -> bool:
        return False

    @property
    def admissible_cell_slots(self) -> tuple[str, ...]:
        """Label slots that are only defined on stress-admissible periodic cells."""
        return ()

    def admit_component(self, component: Any, /) -> None:
        if component is not None:
            raise ValueError(f"Objective {self.name!r} takes no trainable component.")

    @property
    @abstractmethod
    def label_slots(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        """Host ``(count, sum, sum_of_squares)`` of the scaled label quantity."""
        raise NotImplementedError

    def fitted_scale(self, moments: np.ndarray, floor: float, /) -> float:
        """Root-mean-square training label magnitude (refused if degenerate).

        A fit without available training labels, with nonfinite labels, or below
        ``floor`` is refused: declare the objective's scale explicitly instead of
        silently inflating or deflating it.
        """
        count, _, squares = (float(value) for value in moments)
        _require_fittable(self.name, count, squares)
        return _require_scale(self.name, math.sqrt(squares / count), floor)

    @abstractmethod
    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        raise NotImplementedError


def _binding(
    bindings: Mapping[str, AtomisticFieldBinding], slot: str, owner: str, /
) -> AtomisticFieldBinding:
    if slot not in bindings:
        raise ValueError(f"{owner} needs loader slot {slot!r}, which is not bound.")
    return bindings[slot]


def _require_binding(
    binding: AtomisticFieldBinding,
    owner: str,
    /,
    *,
    level: Literal["structure", "atom"],
    shape: tuple[int, ...],
    unit: UnitDefinition | None,
    conventions: tuple[str, ...] | None = None,
) -> None:
    if binding.level_kind != level:
        raise ValueError(f"{owner} label {binding.slot!r} must be {level}-level.")
    if binding.value_kind != "continuous" or binding.role != "label":
        raise ValueError(f"{owner} label {binding.slot!r} must be a continuous label.")
    if tuple(axis.size for axis in binding.axes) != shape:
        raise ValueError(
            f"{owner} label {binding.slot!r} must have trailing shape {shape}."
        )
    if unit is not None and (
        binding.unit is None or binding.unit.unit_id != unit.unit_id
    ):
        raise ValueError(
            f"{owner} label {binding.slot!r} must be bound in {unit.symbol}; convert it "
            "explicitly with the binding's convert_from."
        )
    if conventions is not None and binding.convention not in conventions:
        raise ValueError(
            f"{owner} label {binding.slot!r} convention {binding.convention!r} is not "
            f"one of {conventions!r}."
        )


def _objective_identity(kind: str, /, **fields: Any) -> str:
    return canonical_fingerprint({"kind": kind, **fields})


def _schedule(value: AtomisticObjectiveSchedule | None, /) -> AtomisticObjectiveSchedule:
    if value is None:
        return AtomisticObjectiveSchedule.constant()
    if not isinstance(value, AtomisticObjectiveSchedule):
        raise TypeError("schedule must be an AtomisticObjectiveSchedule or None.")
    return value


def _loss(value: AtomisticResidualLoss | None, /) -> AtomisticResidualLoss:
    if value is None:
        return AtomisticResidualLoss()
    if not isinstance(value, AtomisticResidualLoss):
        raise TypeError("loss must be an AtomisticResidualLoss or None.")
    return value


def _require_fittable(name: str, count: float, squares: float, /) -> None:
    if count <= 0.0:
        raise ValueError(
            f"Objective {name!r} has no available training labels to fit its scale; "
            "declare the scale explicitly."
        )
    if not math.isfinite(squares):
        raise ValueError(
            f"Objective {name!r} has nonfinite training labels; its scale cannot be "
            "fitted."
        )


def _require_scale(name: str, value: float, floor: float, /) -> float:
    if value < floor:
        raise ValueError(
            f"Objective {name!r} fitted a degenerate scale {value!r} below the floor "
            f"{floor!r}; declare the scale explicitly."
        )
    return value


def _moments(values: np.ndarray, /) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64).reshape((-1,))
    return np.asarray(
        [selected.size, np.sum(selected), np.sum(selected * selected)],
        dtype=np.float64,
    )


def _require_model_scale(
    model: AbstractAtomisticModel, scale: AtomisticScaleContract, owner: str, /
) -> None:
    if model.scale.scale_id != scale.scale_id:
        raise ValueError(f"{owner}: the model and training data scales differ.")


def _require_energy_model(model: AbstractAtomisticModel, owner: str, /) -> None:
    if not isinstance(model, AbstractAtomisticPotential) or not (
        model.capabilities.conservative_energy
    ):
        raise ValueError(
            f"{owner} needs a conservative scalar-energy potential; property-only "
            "models have no energy."
        )


@final
class AtomisticReferenceEnergyFit(StrictModule, NonTrainableState):
    """Per-species reference energies fitted by native rank-revealing least squares.

    ``energies[k]`` is the energy of species code ``species[k]`` in the scale's
    energy unit. The fit minimizes ``sum_s (E_s - sum_k n_sk e_k)**2`` over the
    admitted draws of one source, split and head through the native economy-SVD
    pseudoinverse of the composition Gram matrix; ``rank`` and
    ``gram_singular_values`` are its rank evidence (a rank-deficient composition
    returns the minimum-norm solution, never an invented unique one), and
    ``residual_rms`` the per-structure root-mean-square residual.
    ``convention`` names the scientific energy convention the fit refers to.
    """

    energies: Array
    gram_singular_values: Array
    energy_unit: UnitDefinition
    species: tuple[int, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    residual_rms: float = eqx.field(static=True)
    record_count: int = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    split: str = eqx.field(static=True)
    head: str | None = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    fit_id: str = eqx.field(static=True)

    def composition_energy(
        self,
        species: Array,
        atom_mask: Array,
        /,
        *,
        atom_sum: Callable[[Array], Array] | None = None,
    ) -> tuple[Array, Array]:
        """Reference energy of each case and whether every atom's species is fitted.

        ``atom_sum`` reduces over the atom axis (per-case sum by default).
        """
        reduce = case_atom_sum if atom_sum is None else atom_sum
        table = jnp.asarray(self.species, dtype=species.dtype)
        index = jnp.clip(jnp.searchsorted(table, species), 0, table.size - 1)
        known = table[index] == species
        energies = self.energies[index]
        total = reduce(jnp.where(atom_mask, energies, 0.0))
        unknown = reduce((atom_mask & ~known).astype(jnp.int32))
        return total, unknown == 0


def fit_atomistic_reference_energies(
    loader: AtomisticStructureLoader,
    /,
    *,
    species_slot: str,
    energy_slot: str,
    species: Sequence[int],
    rank_policy: RankPolicy | None = None,
    split: str = "training",
    head: str | None = None,
    convention: str = "total-energy",
) -> AtomisticReferenceEnergyFit:
    """Fit per-species reference energies over one full epoch of ``loader``.

    Only valid draws with an available finite energy enter; every atom species
    must be declared in ``species``. The composition Gram matrix and right-hand
    side are accumulated in float64 on the host (bounded by the species count),
    then solved with the native pseudoinverse under ``rank_policy`` (relative
    cutoff on the Gram spectrum, which squares the composition condition
    number).
    """
    if not isinstance(loader, AtomisticStructureLoader):
        raise TypeError("loader must be an AtomisticStructureLoader.")
    table = tuple(sorted({int(value) for value in species}))
    if not table or len(table) != len(tuple(species)):
        raise ValueError("species must list distinct species codes.")
    policy = RankPolicy(relative_cutoff=1e-12) if rank_policy is None else rank_policy
    if not isinstance(policy, RankPolicy):
        raise TypeError("rank_policy must be a RankPolicy or None.")
    schema = loader.mixture.schema
    codes = np.asarray(table, dtype=np.int64)
    gram = np.zeros((codes.size, codes.size), dtype=np.float64)
    rhs = np.zeros((codes.size,), dtype=np.float64)
    target_squares = 0.0
    records = 0
    with loader.epoch(loader.start(0)) as epoch:
        for item in epoch:
            payload = item.payload
            layout = payload.layout
            cases = layout.case_capacity * layout.local_shards
            atom = payload.level(schema.atom_level)
            atoms = np.asarray(atom.row_mask).size // cases
            mask = np.asarray(atom.row_mask, dtype=np.bool_).reshape((cases, atoms))
            kinds = (
                np.asarray(payload.value(species_slot))
                .astype(np.int64)
                .reshape((cases, atoms))
            )
            energy = np.asarray(payload.value(energy_slot), dtype=np.float64).reshape(
                (cases,)
            )
            active = np.asarray(payload.case_valid, dtype=np.bool_) & np.asarray(
                payload.availability(energy_slot), dtype=np.bool_
            ).reshape((cases,))
            active &= np.isfinite(energy)
            index = np.searchsorted(codes, kinds)
            known = (index < codes.size) & (
                codes[np.minimum(index, codes.size - 1)] == kinds
            )
            if np.any(active[:, None] & mask & ~known):
                raise ValueError(
                    "A training structure contains a species absent from the declared "
                    "reference-energy species."
                )
            counts = np.zeros((cases, codes.size), dtype=np.float64)
            rows, columns = np.nonzero(active[:, None] & mask)
            np.add.at(counts, (rows, index[rows, columns]), 1.0)
            selected = counts[active]
            targets = energy[active]
            gram += selected.T @ selected
            rhs += selected.T @ targets
            target_squares += float(targets @ targets)
            records += int(np.count_nonzero(active))
    if records == 0:
        raise ValueError("No valid structure carries an available energy label.")
    factors = factor_pseudoinverse(jnp.asarray(gram), policy, hermitian=True)
    solution = np.asarray(apply_pseudoinverse(factors, jnp.asarray(rhs)))
    residual_squares = max(
        target_squares - 2.0 * float(solution @ rhs) + float(solution @ gram @ solution),
        0.0,
    )
    rank = int(np.asarray(factors.rank))
    energy_binding = next(
        binding for binding in loader.bindings if binding.slot == energy_slot
    )
    if energy_binding.unit is None:
        raise ValueError("The energy slot must be a continuous energy label.")
    singular = np.asarray(factors.singular_values)
    fit_id = canonical_fingerprint(
        {
            "kind": "atomistic-reference-energy-fit",
            "loader": loader.semantics_id,
            "species": table,
            "species_slot": species_slot,
            "energy_slot": energy_slot,
            "energies": [float(value).hex() for value in solution],
            "rank": rank,
            "rank_policy": [policy.relative_cutoff, policy.absolute_cutoff],
            "split": split,
            "head": head,
            "convention": convention,
            "unit": energy_binding.unit.unit_id,
        }
    )
    if rank < codes.size and policy.require_full_rank:
        raise ValueError(
            f"The reference-energy composition has rank {rank} < {codes.size} species."
        )
    return AtomisticReferenceEnergyFit(
        energies=jnp.asarray(solution),
        gram_singular_values=jnp.asarray(singular),
        energy_unit=energy_binding.unit,
        species=table,
        rank=rank,
        residual_rms=math.sqrt(residual_squares / records),
        record_count=records,
        source_ids=loader.mixture.source_ids,
        split=_name(split, "split"),
        head=None if head is None else _name(head, "head"),
        convention=_name(convention, "convention"),
        fit_id=fit_id,
    )


@final
class AtomisticEnergyObjective(AbstractAtomisticObjective):
    """Total-energy matching under an explicit structure or atom balance.

    ``per-atom-structure`` (the structure-balanced per-atom energy policy):
    ``sum_s w_s rho((E_s - Y_s) / (N_s c))`` with support ``sum_s w_s``.
    ``total-structure``: residual ``(E_s - Y_s) / c`` with the same support.
    ``per-atom-atom-weighted``: per-atom residual weighted by ``N_s`` with support
    ``sum_s w_s N_s``. ``reference`` declares that the model predicts energy
    relative to a fitted per-species reference, so the label becomes
    ``Y_s - sum_i e(z_i)``; an undeclared species fails the case.
    """

    name: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    reduction: AtomisticEnergyReduction = eqx.field(static=True)
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    reference: AtomisticReferenceEnergyFit | None
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        slot: str = "energy",
        /,
        *,
        name: str = "energy",
        reduction: AtomisticEnergyReduction = "per-atom-structure",
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
        reference: AtomisticReferenceEnergyFit | None = None,
    ) -> None:
        loss_ = _loss(loss)
        if loss_.vector_only:
            raise ValueError("Energy residuals are scalar; euclidean is not admitted.")
        if reference is not None and not isinstance(
            reference, AtomisticReferenceEnergyFit
        ):
            raise TypeError("reference must be an AtomisticReferenceEnergyFit or None.")
        self.name = _name(name, "name")
        self.slot = _name(slot, "slot")
        self.reduction = parse(reduction, AtomisticEnergyReduction, "reduction")
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.reference = reference
        self.objective_id = _objective_identity(
            "atomistic-energy-objective",
            name=self.name,
            slot=self.slot,
            reduction=self.reduction,
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
            reference=None if reference is None else reference.fit_id,
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Energy objective {self.name!r}"
        _require_energy_model(model, owner)
        _require_model_scale(model, scale, owner)
        _require_binding(
            _binding(bindings, self.slot, owner),
            owner,
            level="structure",
            shape=(),
            unit=scale.energy_unit,
        )
        if (
            self.reference is not None
            and self.reference.energy_unit.unit_id != scale.energy_unit.unit_id
        ):
            raise ValueError(f"{owner}: the reference fit uses another energy unit.")

    def _targets(
        self, labels: np.ndarray, species: np.ndarray, mask: np.ndarray, /
    ) -> np.ndarray:
        if self.reference is None:
            return labels
        table = np.asarray(self.reference.species, dtype=np.int64)
        index = np.clip(np.searchsorted(table, species), 0, table.size - 1)
        energies = np.asarray(self.reference.energies, dtype=np.float64)[index]
        return labels - np.sum(np.where(mask, energies, 0.0), axis=1)

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        active = labels.case_valid & available
        targets = self._targets(
            np.asarray(values, dtype=np.float64), labels.species, labels.atom_mask
        )
        if self.reduction != "total-structure":
            targets = targets / labels.atom_counts
        return _moments(targets[active])

    def fitted_scale(self, moments: np.ndarray, floor: float, /) -> float:
        """Standard deviation of the scaled training energy (refused if degenerate)."""
        count, total, squares = (float(value) for value in moments)
        _require_fittable(self.name, count, squares)
        mean = total / count
        return _require_scale(
            self.name, math.sqrt(max(squares / count - mean * mean, 0.0)), floor
        )

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        derivatives = atomistic_training_derivatives(model, batch, forces=False)
        values, available = batch.label(self.slot)
        return energy_objective_term(
            self,
            derivatives.energy,
            derivatives.successful,
            derivatives.overflow,
            labels=values,
            available=available,
            case_valid=batch.case_valid,
            case_weight=batch.case_weight,
            species=batch.species,
            atom_mask=batch.atom_mask,
            scale=scale,
            dtype=dtype,
        )


def energy_objective_term(
    objective: AtomisticEnergyObjective,
    energy: Array,
    successful: Array,
    overflow: Array,
    /,
    *,
    labels: Array,
    available: Array,
    case_valid: Array,
    case_weight: Array,
    species: Array,
    atom_mask: Array,
    scale: float,
    dtype: str,
    atom_sum: Callable[[Array], Array] = case_atom_sum,
) -> AtomisticObjectiveTerm:
    """Energy numerator/support from case energies ``(case,)``.

    Atom counts and the reference composition energy reduce over atoms through
    ``atom_sum``; case sums are ordinary sums.
    """
    target = labels.astype(dtype)
    known = jnp.ones(case_valid.shape, dtype=jnp.bool_)
    if objective.reference is not None:
        offset, known = objective.reference.composition_energy(
            species, atom_mask, atom_sum=atom_sum
        )
        target = target - offset.astype(dtype)
    counts = atom_sum(atom_mask.astype(dtype))
    residual = energy.astype(dtype) - target
    if objective.reduction != "total-structure":
        residual = residual / jnp.where(counts > 0.0, counts, 1.0)
    weight = jnp.where(case_valid & available, case_weight.astype(dtype), 0.0)
    penalty = _masked(objective.loss.pointwise(residual / scale), weight > 0.0)
    if objective.reduction == "per-atom-atom-weighted":
        weight = weight * counts
    numerator = jnp.sum(weight * penalty, dtype=dtype)
    support = jnp.sum(weight, dtype=dtype)
    return _term(
        numerator,
        support,
        case_valid & overflow,
        case_valid & (~successful | ~known),
    )


@final
class AtomisticForceObjective(AbstractAtomisticObjective):
    """Conservative force matching under an explicit balance.

    ``component``: every available Cartesian component has unit support.
    ``atom``: each available atom contributes ``rho_vec(r_i)`` (the sum of
    componentwise penalties, or the Euclidean norm) with unit support.
    ``structure`` (graph-balanced): each structure contributes the mean of its
    available atoms' ``rho_vec`` with unit support. Residuals are divided by
    ``scale`` in force units.
    """

    name: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    reduction: AtomisticForceReduction = eqx.field(static=True)
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        slot: str = "forces",
        /,
        *,
        name: str = "forces",
        reduction: AtomisticForceReduction = "component",
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        loss_ = _loss(loss)
        reduction_ = parse(reduction, AtomisticForceReduction, "reduction")
        if loss_.vector_only and reduction_ == "component":
            raise ValueError("A euclidean force residual has no component reduction.")
        self.name = _name(name, "name")
        self.slot = _name(slot, "slot")
        self.reduction = reduction_
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-force-objective",
            name=self.name,
            slot=self.slot,
            reduction=reduction_,
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Force objective {self.name!r}"
        _require_energy_model(model, owner)
        _require_model_scale(model, scale, owner)
        _require_binding(
            _binding(bindings, self.slot, owner),
            owner,
            level="atom",
            shape=(3,),
            unit=scale.force_unit,
        )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        active = labels.case_valid[:, None] & available & labels.atom_mask
        return _moments(np.asarray(values)[active])

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        derivatives = atomistic_training_derivatives(model, batch, forces=True)
        if derivatives.forces is None:
            raise RuntimeError("Requested conservative forces were not returned.")
        values, available = batch.label(self.slot)
        return force_objective_term(
            self,
            derivatives.forces,
            derivatives.successful,
            derivatives.overflow,
            labels=values,
            available=available,
            case_valid=batch.case_valid,
            case_weight=batch.case_weight,
            atom_mask=batch.atom_mask,
            scale=scale,
            dtype=dtype,
        )


def force_objective_term(
    objective: AtomisticForceObjective,
    forces: Array,
    successful: Array,
    overflow: Array,
    /,
    *,
    labels: Array,
    available: Array,
    case_valid: Array,
    case_weight: Array,
    atom_mask: Array,
    scale: float,
    dtype: str,
    atom_sum: Callable[[Array], Array] = case_atom_sum,
) -> AtomisticObjectiveTerm:
    """Force numerator/support from predictions ``(case, atom, 3)``.

    Every reduction over the atom axis goes through ``atom_sum`` (per-case sum
    over axis 1 by default; an owner-axis sum for domain-decomposed atoms).
    """
    atoms = case_valid[:, None] & available & atom_mask
    weight = jnp.where(atoms, case_weight.astype(dtype)[:, None], 0.0)
    residual = _masked(
        (forces.astype(dtype) - labels.astype(dtype)) / scale, atoms[:, :, None]
    )
    match objective.reduction:
        case "component":
            penalty = jnp.sum(objective.loss.pointwise(residual), axis=-1)
            numerator = jnp.sum(atom_sum(weight * penalty), dtype=dtype)
            support = 3.0 * jnp.sum(atom_sum(weight), dtype=dtype)
        case "atom":
            numerator = jnp.sum(
                atom_sum(weight * objective.loss.vector(residual)), dtype=dtype
            )
            support = jnp.sum(atom_sum(weight), dtype=dtype)
        case "structure":
            counts = atom_sum(atoms.astype(dtype))
            present = counts > 0.0
            per_atom = jnp.where(atoms, objective.loss.vector(residual), 0.0)
            mean = atom_sum(per_atom) / jnp.where(present, counts, 1.0)
            structure_weight = jnp.where(
                case_valid & present, case_weight.astype(dtype), 0.0
            )
            numerator = jnp.sum(structure_weight * mean, dtype=dtype)
            support = jnp.sum(structure_weight, dtype=dtype)
        case _:
            assert_never(objective.reduction)
    return _term(numerator, support, case_valid & overflow, case_valid & ~successful)


_STRESS_CONVENTIONS = (
    AtomisticStressConvention.CAUCHY_TENSION_POSITIVE.value,
    AtomisticStressConvention.CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME.value,
)


def atomistic_stress_unit(scale: AtomisticScaleContract, /) -> UnitDefinition:
    """Energy per cubic length of a cell-volume-normalized stress label."""
    return derived_unit(
        f"{scale.energy_unit.symbol}/{scale.length_unit.symbol}^3",
        ((scale.energy_unit, 1), (scale.length_unit, -3)),
    )


@final
class AtomisticStressObjective(AbstractAtomisticObjective):
    """Tensile-positive cell-volume-normalized stress matching.

    Labels are ``(1/V) dE/d strain`` (``AtomisticStressConvention``) in energy per
    cubic length, available only on cases with periodic axes and a finite
    nonsingular cell. ``components`` is an explicit ``(3, 3)`` mask of matched
    components. ``component``: unit support per available masked component;
    ``structure``: each case contributes the mean over its masked components.
    """

    name: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    reduction: AtomisticTensorReduction = eqx.field(static=True)
    components: tuple[tuple[bool, bool, bool], ...] = eqx.field(static=True)
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        slot: str = "stress",
        /,
        *,
        name: str = "stress",
        reduction: AtomisticTensorReduction = "component",
        components: ConvertibleToArray | None = None,
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        loss_ = _loss(loss)
        if loss_.vector_only:
            raise ValueError("Stress components use componentwise residual forms.")
        mask = (
            np.ones((3, 3), dtype=np.bool_)
            if components is None
            else np.asarray(components, dtype=np.bool_)
        )
        if mask.shape != (3, 3) or not np.any(mask):
            raise ValueError("components must be a nonempty (3, 3) boolean mask.")
        self.name = _name(name, "name")
        self.slot = _name(slot, "slot")
        self.reduction = parse(reduction, AtomisticTensorReduction, "reduction")
        self.components = tuple(
            (bool(row[0]), bool(row[1]), bool(row[2])) for row in mask
        )
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-stress-objective",
            name=self.name,
            slot=self.slot,
            reduction=self.reduction,
            components=self.components,
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    @property
    def admissible_cell_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Stress objective {self.name!r}"
        _require_energy_model(model, owner)
        _require_model_scale(model, scale, owner)
        if not model.capabilities.cell_derivative:
            raise ValueError(
                f"{owner} needs a potential with cell-derivative capability."
            )
        _require_binding(
            _binding(bindings, self.slot, owner),
            owner,
            level="structure",
            shape=(3, 3),
            unit=atomistic_stress_unit(scale),
            conventions=_STRESS_CONVENTIONS,
        )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        active = labels.case_valid & available
        mask = np.asarray(self.components, dtype=np.bool_)
        return _moments(np.asarray(values)[active][:, mask])

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        values, available = batch.label(self.slot)
        requested = available & batch.stress_cells
        derivatives = atomistic_training_derivatives(
            model, batch, forces=False, stress_cases=requested
        )
        if derivatives.stress is None:
            raise RuntimeError("Requested stress was not returned.")
        return stress_objective_term(
            self,
            derivatives.stress,
            derivatives.successful,
            derivatives.overflow,
            labels=values,
            available=available,
            requested=requested,
            case_valid=batch.case_valid,
            case_weight=batch.case_weight,
            scale=scale,
            dtype=dtype,
        )


def stress_objective_term(
    objective: AtomisticStressObjective,
    stress: Array,
    successful: Array,
    overflow: Array,
    /,
    *,
    labels: Array,
    available: Array,
    requested: Array,
    case_valid: Array,
    case_weight: Array,
    scale: float,
    dtype: str,
) -> AtomisticObjectiveTerm:
    """Stress numerator/support from predictions; ``requested`` is the
    available-and-admissible-cell mask. Every quantity is per case."""
    mask = jnp.asarray(objective.components, dtype=jnp.bool_)
    selected = requested & case_valid
    weight = jnp.where(selected, case_weight.astype(dtype), 0.0)
    residual = _masked(
        (stress.astype(dtype) - labels.astype(dtype)) / scale,
        selected[:, None, None] & mask[None],
    )
    penalty = jnp.sum(
        jnp.where(mask[None], objective.loss.pointwise(residual), 0.0), axis=(1, 2)
    )
    components = jnp.sum(mask).astype(dtype)
    match objective.reduction:
        case "component":
            numerator = jnp.sum(weight * penalty, dtype=dtype)
            support = components * jnp.sum(weight, dtype=dtype)
        case "structure":
            numerator = jnp.sum(weight * penalty / components, dtype=dtype)
            support = jnp.sum(weight, dtype=dtype)
        case _:
            assert_never(objective.reduction)
    return _term(
        numerator,
        support,
        case_valid & overflow,
        case_valid & (~successful | (available & ~requested)),
    )


@final
class AtomisticPropertyObjective(AbstractAtomisticObjective):
    """Matching of one additional typed model observable against a label slot.

    The model is evaluated with a request for ``observable`` and the value whose
    ``ValuePort`` equals ``port`` is matched componentwise to the label, which
    has the port's level (case or atom) and event shape. ``unit`` is the label
    unit and must equal the observable's declared unit. ``component`` reduction
    gives unit support per available component; ``structure`` averages each
    row's components (atom-level values are averaged over a case's available
    atoms first).
    """

    name: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    observable: AtomisticObservable = eqx.field(static=True)
    port: ValuePort
    level: AtomisticConditionLevel = eqx.field(static=True)
    unit: UnitDefinition | None
    reduction: AtomisticTensorReduction = eqx.field(static=True)
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        observable: AtomisticObservable,
        port: ValuePort,
        slot: str,
        /,
        *,
        level: AtomisticConditionLevel,
        unit: UnitDefinition | None,
        name: str | None = None,
        reduction: AtomisticTensorReduction = "component",
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        observable_ = parse(observable, AtomisticObservable, "observable")
        if observable_ in (
            AtomisticObservable.ENERGY,
            AtomisticObservable.FORCES,
            AtomisticObservable.STRESS,
            AtomisticObservable.ATOM_ENERGY,
            AtomisticObservable.HESSIAN_ACTION,
            AtomisticObservable.HESSIAN_BLOCKS,
        ):
            raise ValueError(
                f"{observable_.value!r} has its own conservative objective, not a "
                "property objective."
            )
        if not isinstance(port, ValuePort):
            raise TypeError("port must be a ValuePort.")
        loss_ = _loss(loss)
        if loss_.vector_only:
            raise ValueError("Property components use componentwise residual forms.")
        self.name = _name(observable_.value if name is None else name, "name")
        self.slot = _name(slot, "slot")
        self.observable = observable_
        self.port = port
        self.level = parse(level, AtomisticConditionLevel, "level")
        self.unit = unit
        self.reduction = parse(reduction, AtomisticTensorReduction, "reduction")
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-property-objective",
            name=self.name,
            slot=self.slot,
            observable=observable_.value,
            port=port.port_id,
            level=self.level.value,
            unit=None if unit is None else unit.unit_id,
            reduction=self.reduction,
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Property objective {self.name!r}"
        _require_model_scale(model, scale, owner)
        binding = _binding(bindings, self.slot, owner)
        _require_binding(
            binding,
            owner,
            level="structure" if self.level is AtomisticConditionLevel.CASE else "atom",
            shape=self.port.event_shape,
            unit=self.unit,
        )
        if (binding.unit is None) != (self.unit is None):
            raise ValueError(
                f"{owner}: label and observable units must both be declared."
            )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        if self.level is AtomisticConditionLevel.CASE:
            active = labels.case_valid & available
        else:
            active = labels.case_valid[:, None] & available & labels.atom_mask
        return _moments(np.asarray(values)[active])

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        outputs, successful, overflow = atomistic_training_outputs(
            model, batch, AtomisticObservableRequest((self.observable,))
        )
        value = outputs.observable(self.observable, port=self.port)
        if value.level is not self.level:
            raise ValueError(f"Property {self.name!r} has another observable level.")
        if (value.unit is None) != (self.unit is None) or (
            value.unit is not None
            and self.unit is not None
            and value.unit.unit_id != self.unit.unit_id
        ):
            raise ValueError(f"Property {self.name!r} label and model units differ.")
        labels, available = batch.label(self.slot)
        event = self.port.event_shape
        axes = tuple(range(-len(event), 0)) if event else ()
        if self.level is AtomisticConditionLevel.CASE:
            rows = batch.case_valid & available & value.available
            prediction = value.values
            row_weight = jnp.where(rows, batch.case_weight.astype(dtype), 0.0)
            row_case = jnp.arange(batch.case_count)
        else:
            rows = (
                (batch.case_valid & value.available)[:, None]
                & available
                & batch.atom_mask
            ).reshape((-1,))
            prediction = value.values
            labels = labels.reshape((-1, *event))
            row_case = batch.atom_cases
            row_weight = jnp.where(rows, batch.case_weight.astype(dtype)[row_case], 0.0)
        residual = _masked(
            (prediction.astype(dtype) - labels.astype(dtype)) / scale,
            rows.reshape(rows.shape + (1,) * len(event)),
        )
        penalty = (
            jnp.sum(self.loss.pointwise(residual), axis=axes)
            if axes
            else self.loss.pointwise(residual)
        )
        size = float(math.prod(event)) if event else 1.0
        match self.reduction:
            case "component":
                numerator = jnp.sum(row_weight * penalty, dtype=dtype)
                support = size * jnp.sum(row_weight, dtype=dtype)
            case "structure":
                if self.level is AtomisticConditionLevel.CASE:
                    numerator = jnp.sum(row_weight * penalty / size, dtype=dtype)
                    support = jnp.sum(row_weight, dtype=dtype)
                else:
                    counts = (
                        jnp.zeros((batch.case_count,), dtype)
                        .at[row_case]
                        .add(rows.astype(dtype))
                    )
                    sums = (
                        jnp.zeros((batch.case_count,), dtype)
                        .at[row_case]
                        .add(jnp.where(rows, penalty / size, 0.0))
                    )
                    present = counts > 0.0
                    case_weight = jnp.where(
                        batch.case_valid & present, batch.case_weight.astype(dtype), 0.0
                    )
                    numerator = jnp.sum(
                        case_weight * sums / jnp.where(present, counts, 1.0), dtype=dtype
                    )
                    support = jnp.sum(case_weight, dtype=dtype)
            case _:
                assert_never(self.reduction)
        valid = batch.case_valid
        return _term(numerator, support, valid & overflow, valid & ~successful)


@final
class AtomisticAtomEnergyObjective(AbstractAtomisticObjective):
    """Decomposition-bound regularizer matching scoped atomic energies.

    Atomic energies are not a universal observable: they belong to one
    decomposition. ``decomposition`` must equal the label binding's convention,
    and ``complete`` declares whether that decomposition sums to the total
    energy (no case-level global correction); a model whose
    ``atom_energy_complete`` differs is refused at trace time instead of
    comparing incompatible gauges. Each available atom has unit support.
    """

    name: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    decomposition: str = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        slot: str,
        /,
        *,
        decomposition: str,
        complete: bool,
        name: str = "atom-energy",
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        loss_ = _loss(loss)
        if loss_.vector_only:
            raise ValueError("Atomic energies are scalars; euclidean is not admitted.")
        if not isinstance(complete, bool):
            raise TypeError("complete must be a boolean.")
        self.name = _name(name, "name")
        self.slot = _name(slot, "slot")
        self.decomposition = _name(decomposition, "decomposition")
        self.complete = complete
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-atom-energy-objective",
            name=self.name,
            slot=self.slot,
            decomposition=self.decomposition,
            complete=complete,
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Atom-energy objective {self.name!r}"
        _require_energy_model(model, owner)
        _require_model_scale(model, scale, owner)
        _require_binding(
            _binding(bindings, self.slot, owner),
            owner,
            level="atom",
            shape=(),
            unit=scale.energy_unit,
            conventions=(self.decomposition,),
        )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        active = labels.case_valid[:, None] & available & labels.atom_mask
        return _moments(np.asarray(values)[active])

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        outputs, successful, overflow = atomistic_training_outputs(
            model, batch, AtomisticObservableRequest((AtomisticObservable.ATOM_ENERGY,))
        )
        if outputs.atom_energy_complete != self.complete:
            raise ValueError(
                f"Atom-energy objective {self.name!r} declares complete={self.complete}, "
                f"but the model decomposition has atom_energy_complete="
                f"{outputs.atom_energy_complete}; the gauges are incompatible."
            )
        values, available = batch.label(self.slot)
        atoms = batch.case_valid[:, None] & available & batch.atom_mask
        weight = jnp.where(atoms, batch.case_weight.astype(dtype)[:, None], 0.0)
        residual = _masked(
            (outputs.atom_energy.astype(dtype) - values.astype(dtype)) / scale, atoms
        )
        numerator = jnp.sum(weight * self.loss.pointwise(residual), dtype=dtype)
        support = jnp.sum(weight, dtype=dtype)
        valid = batch.case_valid
        return _term(numerator, support, valid & overflow, valid & ~successful)


__all__ = [
    "AbstractAtomisticObjective",
    "AbstractAtomisticObjectiveReduction",
    "AtomisticAtomEnergyObjective",
    "AtomisticEnergyObjective",
    "AtomisticEnergyReduction",
    "AtomisticForceObjective",
    "AtomisticForceReduction",
    "AtomisticHostLabels",
    "AtomisticLabelLevel",
    "AtomisticLocalObjectiveReduction",
    "AtomisticObjectiveSchedule",
    "AtomisticObjectiveTerm",
    "AtomisticPropertyObjective",
    "AtomisticReferenceEnergyFit",
    "AtomisticResidualKind",
    "AtomisticResidualLoss",
    "AtomisticScheduleKind",
    "AtomisticStressObjective",
    "AtomisticTensorReduction",
    "AtomisticTrainingBatch",
    "AtomisticTrainingPrecision",
    "atomistic_stress_unit",
    "atomistic_training_derivatives",
    "atomistic_training_hessian",
    "atomistic_training_outputs",
    "case_atom_sum",
    "energy_objective_term",
    "fit_atomistic_reference_energies",
    "force_objective_term",
    "stress_objective_term",
]

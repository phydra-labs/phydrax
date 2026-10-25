#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""UMA task/charge/spin conditioning, mixture-of-linear-experts routing and balance.

Conditions (pinned fairchem ``bb0a062``):

- Total charge: integer elementary charges on the shared ``TOTAL_CHARGE_PORT``,
  admitted in ``[-100, 100]`` as a *categorical* code (the source embeds it by
  table lookup, sinusoid or linear map of the integer; it never accepts a
  fractional charge).
- Spin: the source's own integer *spin index* on `UMA_SPIN_INDEX_PORT`, in
  ``[0, 100]``. Index ``0`` is the source's null token (its sinusoidal
  embedding is zeroed, its linear embedding reads ``-100``); a positive index
  ``k`` is the spin multiplicity ``k = 2S + 1`` used for tasks trained with
  spin. Index ``0`` is therefore *not* a multiplicity of zero, and this port
  is distinct from the physical ``SPIN_MULTIPLICITY_PORT``.
- Task: the dataset head selected by an integer code on `UMA_TASK_PORT`; codes
  are positions in the sorted source ``dataset_mapping`` keys.

The source prediction unit's defaults (charge ``0``; spin ``1`` for ``omol``,
``0`` otherwise) are applied only by `UMAConditionSchema.conditioning` and
are recorded as explicit default use.

System embedding: ``silu(mix_csd([charge, spin, dataset]))``. Mixture routing:
``silu(L3(silu(L2(silu(L1([composition, csd]))))))`` normalized by the declared
coefficient rule; the composition is the per-case mean of a species embedding
(`UMACompositionPolicy` ``"mean"``) or the source UMA-S-1.2 rule that includes
the zero-initialized reduction seed, ``sum / (n + 1)``
(``"sum-over-count-plus-one"``). These are different scientific definitions
with different identities. Channel balancing shifts declared scalar channels
so their per-case sum equals ``charge`` (charge channels) or ``spin - 1``
(spin channels), dividing the residual equally over the case's atoms.

Every per-case quantity is a sum over the case's atoms followed by a local
rule: composition and routing read species counts, balancing reads channel
sums and atom counts. Single-graph evaluation forms those sums over its
case-major slabs (`case_reduce`); domain execution forms owner partials over
rows that name their case (`case_sums`), combines them once across owners and
applies the same rule (`channel_balance_correction`, `apply_channel_balance`,
`UMAMoLERouter.coefficients`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._model._fixed_binding import require_identical
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import ExplicitFreeze, NonTrainableState, ParameterOwner
from ..._validation import integer, positive_integer
from ...atomistic._conditioning import (
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionKind,
    AtomisticConditionLevel,
    TOTAL_CHARGE_PORT,
)
from ...typing import Dim, Identifier, Identifiers, parse
from ...units import DIMENSIONLESS, ELEMENTARY_CHARGE
from ._uma_layers import UMALinear


UMAChargeSpinEmbeddingKind: TypeAlias = Literal["pos_emb", "lin_emb", "rand_emb"]
UMACompositionPolicy: TypeAlias = Literal["mean", "sum-over-count-plus-one"]
UMACoefficientNorm: TypeAlias = Literal["softmax-plus-0.005", "pnorm-batch-count"]

UMA_CHARGE_RANGE = (-100, 100)
UMA_SPIN_RANGE = (0, 100)
UMA_CHARGE_FIELD = "total_charge"
UMA_SPIN_FIELD = "uma_spin_index"
UMA_TASK_FIELD = "uma_task"
# Source prediction-unit defaults (api/inference.py).
_DEFAULT_CHARGE = 0
_DEFAULT_SPIN = 0
_DEFAULT_SPIN_BY_TASK = {"omol": 1}

UMA_SPIN_INDEX_PORT = ValuePort(
    "uma.spin-index",
    event_shape=(),
    component_ids=("spin_index",),
    representation="uma-source-spin-index-null-zero-else-multiplicity",
    dimensions=(DIMENSIONLESS,),
    normalization_id="identity",
)
UMA_TASK_PORT = ValuePort(
    "uma.task",
    event_shape=(),
    component_ids=("task",),
    representation="uma-sorted-dataset-mapping-key-code",
    dimensions=(DIMENSIONLESS,),
    normalization_id="identity",
)


class UMATaskDim(Dim):
    pass


@final
class UMAConditionSchema(StrictModule, NonTrainableState):
    """Declared UMA condition fields and task codes of one source model."""

    __strict_contract__ = True

    spec: AtomisticConditioningSpec
    tasks: Identifiers[UMATaskDim] = eqx.field(static=True)

    def __init__(self, tasks: Sequence[str], /) -> None:
        names = tuple(sorted(str(task) for task in tasks))
        if not names or len(set(names)) != len(names):
            raise ValueError("UMA tasks must be a nonempty set of names.")
        for name in names:
            if not name or name != name.strip():
                raise ValueError("UMA task names must be nonempty identifiers.")
        self.tasks = names
        self.spec = AtomisticConditioningSpec(
            (
                AtomisticConditionField(
                    UMA_CHARGE_FIELD,
                    TOTAL_CHARGE_PORT,
                    level=AtomisticConditionLevel.CASE,
                    kind=AtomisticConditionKind.CATEGORICAL,
                    dtype=jnp.int64,
                    minimum=UMA_CHARGE_RANGE[0],
                    maximum=UMA_CHARGE_RANGE[1],
                    unit=ELEMENTARY_CHARGE,
                ),
                AtomisticConditionField(
                    UMA_SPIN_FIELD,
                    UMA_SPIN_INDEX_PORT,
                    level=AtomisticConditionLevel.CASE,
                    kind=AtomisticConditionKind.CATEGORICAL,
                    dtype=jnp.int64,
                    minimum=UMA_SPIN_RANGE[0],
                    maximum=UMA_SPIN_RANGE[1],
                ),
                AtomisticConditionField(
                    UMA_TASK_FIELD,
                    UMA_TASK_PORT,
                    level=AtomisticConditionLevel.CASE,
                    kind=AtomisticConditionKind.CATEGORICAL,
                    dtype=jnp.int64,
                    categories=tuple(range(len(names))),
                ),
            )
        )

    def code(self, task: str, /) -> int:
        if task not in self.tasks:
            raise ValueError(f"UMA task {task!r} is not one of {self.tasks!r}.")
        return self.tasks.index(task)

    def conditioning(
        self,
        tasks: Sequence[str],
        /,
        *,
        atom_capacity: int,
        charges: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
    ) -> AtomisticConditioning:
        """Conditions of ``len(tasks)`` cases with the source prediction defaults.

        A missing ``charges`` uses the source default ``0``. A missing ``spins``
        uses the source default of the (single) task: ``1`` for ``omol``,
        ``0`` (null) otherwise; mixed-task batches must state spins
        explicitly. Default use is recorded in ``defaults_used``.
        """
        codes = np.asarray([self.code(task) for task in tasks], dtype=np.int64)
        if codes.size == 0:
            raise ValueError("UMA conditioning needs at least one case.")
        values: dict[str, np.ndarray] = {UMA_TASK_FIELD: codes}
        defaults: dict[str, int] = {}
        if charges is None:
            defaults[UMA_CHARGE_FIELD] = _DEFAULT_CHARGE
        else:
            values[UMA_CHARGE_FIELD] = _integer_values(charges, codes.size, "charges")
        if spins is None:
            distinct = set(tasks)
            if len(distinct) != 1:
                raise ValueError(
                    "Mixed-task UMA batches must state each case's spin index; the "
                    "source default depends on the task."
                )
            defaults[UMA_SPIN_FIELD] = _DEFAULT_SPIN_BY_TASK.get(
                next(iter(distinct)), _DEFAULT_SPIN
            )
        else:
            values[UMA_SPIN_FIELD] = _integer_values(spins, codes.size, "spins")
        return AtomisticConditioning(
            self.spec,
            values,
            case_count=int(codes.size),
            atom_capacity=atom_capacity,
            defaults=defaults or None,
        )


def _integer_values(values: Sequence[int], count: int, name: str, /) -> np.ndarray:
    resolved = [integer(value, name) for value in values]
    if len(resolved) != count:
        raise ValueError(f"{name} must give one integer per case.")
    return np.asarray(resolved, dtype=np.int64)


@final
class UMAFrozen(StrictModule, ExplicitFreeze):
    """A source component whose parameters the source keeps without gradients.

    The source sets ``requires_grad=False`` for charge/spin embeddings unless
    ``cs_emb_grad`` and for dataset embeddings unless ``dataset_emb_grad``; the
    native role is explicitly FIXED, so optimizers never update them and the
    role audit recognizes the deliberate freezing of parameter-bearing components.
    """

    value: StrictModule

    def __init__(self, value: StrictModule, /) -> None:
        self.value = value


@final
class UMAChargeSpinEmbedding(StrictModule, ParameterOwner):
    """Source ``ChgSpinEmbedding`` of total charge or spin index ``(B,) -> (B, C)``."""

    weight: Array
    bias: Array | None
    kind: UMAChargeSpinEmbeddingKind = eqx.field(static=True)
    target: Literal["charge", "spin"] = eqx.field(static=True)

    def __init__(
        self,
        kind: UMAChargeSpinEmbeddingKind,
        target: Literal["charge", "spin"],
        weight: ArrayLike,
        /,
        *,
        bias: ArrayLike | None = None,
    ) -> None:
        kind_ = parse(kind, UMAChargeSpinEmbeddingKind, "embedding kind")
        target_ = parse(target, Literal["charge", "spin"], "embedding target")
        weight_ = jnp.asarray(weight)
        rows = 201 if target_ == "charge" else 101
        match kind_:
            case "pos_emb":
                valid = weight_.ndim == 1 and bias is None
            case "lin_emb":
                valid = weight_.ndim == 2 and weight_.shape[1] == 1 and bias is not None
            case "rand_emb":
                valid = weight_.ndim == 2 and weight_.shape[0] == rows and bias is None
        if not valid or weight_.dtype not in (jnp.float32, jnp.float64):
            raise ValueError(f"{kind_} {target_} embedding parameters are malformed.")
        bias_ = None if bias is None else jnp.asarray(bias, dtype=weight_.dtype)
        if bias_ is not None and bias_.shape != (weight_.shape[0],):
            raise ValueError("lin_emb bias must match its output width.")
        self.weight = weight_
        self.bias = bias_
        self.kind = kind_
        self.target = target_

    @property
    def width(self) -> int:
        return (
            2 * int(self.weight.shape[0])
            if self.kind == "pos_emb"
            else int(
                self.weight.shape[0] if self.kind == "lin_emb" else self.weight.shape[1]
            )
        )

    def __call__(self, values: Array, /) -> Array:
        dtype = self.weight.dtype
        match self.kind:
            case "pos_emb":
                projected = (
                    values.astype(dtype)[:, None] * self.weight[None, :] * (2.0 * math.pi)
                )
                embedding = jnp.concatenate(
                    [jnp.sin(projected), jnp.cos(projected)], axis=-1
                )
                if self.target == "spin":
                    embedding = jnp.where((values == 0)[:, None], 0.0, embedding)
                return embedding
            case "lin_emb":
                assert self.bias is not None
                scalar = values
                if self.target == "spin":
                    scalar = jnp.where(values == 0, -100, values)
                return scalar.astype(dtype)[:, None] * self.weight[:, 0] + self.bias
            case "rand_emb":
                offset = 100 if self.target == "charge" else 0
                rows = self.weight.shape[0]
                index = jnp.clip(values.astype(jnp.int32) + offset, 0, rows - 1)
                return self.weight[index]


@final
class UMADatasetEmbedding(StrictModule, ParameterOwner):
    """Source ``DatasetEmbedding``: one ``(1, C)`` table per ``dataset_mapping`` key.

    ``table`` stacks those rows in source key order; a task code uses the row
    of the key its mapped value names. Rows of keys that no value names are
    retained source parameters that no task reaches.
    """

    table: Array
    rows: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, table: ArrayLike, rows: Sequence[int], /) -> None:
        table_ = jnp.asarray(table)
        rows_ = tuple(int(row) for row in rows)
        if table_.ndim != 2 or any(row < 0 or row >= table_.shape[0] for row in rows_):
            raise ValueError("Dataset embedding rows are inconsistent.")
        self.table = table_
        self.rows = rows_

    def __call__(self, codes: Array, /) -> Array:
        rows = jnp.asarray(self.rows, dtype=jnp.int32)
        return self.table[rows[jnp.clip(codes.astype(jnp.int32), 0, len(self.rows) - 1)]]


def _component[Component: StrictModule](
    value: Component | UMAFrozen, kind: type[Component], /
) -> Component:
    """The (possibly frozen) source component, validated against its declared class."""
    inner = value.value if isinstance(value, UMAFrozen) else value
    if not isinstance(inner, kind):
        raise TypeError(f"The UMA system embedding expects a {kind.__name__} component.")
    return inner


@final
class UMASystemEmbedding(StrictModule, ParameterOwner):
    """Source ``csd_embedding``: ``silu(mix_csd(cat(charge, spin[, dataset])))``."""

    charge: UMAChargeSpinEmbedding | UMAFrozen
    spin: UMAChargeSpinEmbedding | UMAFrozen
    dataset: UMADatasetEmbedding | UMAFrozen | None
    mix: UMALinear

    def __init__(
        self,
        charge: UMAChargeSpinEmbedding | UMAFrozen,
        spin: UMAChargeSpinEmbedding | UMAFrozen,
        dataset: UMADatasetEmbedding | UMAFrozen | None,
        mix: UMALinear,
        /,
    ) -> None:
        widths = [
            _component(charge, UMAChargeSpinEmbedding).width,
            _component(spin, UMAChargeSpinEmbedding).width,
        ]
        if dataset is not None:
            widths.append(int(_component(dataset, UMADatasetEmbedding).table.shape[1]))
        if mix.in_features != sum(widths) or mix.bias is None:
            raise ValueError(
                "mix_csd input width must equal the concatenated embeddings."
            )
        self.charge = charge
        self.spin = spin
        self.dataset = dataset
        self.mix = mix

    def __call__(self, charge: Array, spin: Array, task: Array, /) -> Array:
        parts = [
            _component(self.charge, UMAChargeSpinEmbedding)(charge),
            _component(self.spin, UMAChargeSpinEmbedding)(spin),
        ]
        if self.dataset is not None:
            parts.append(_component(self.dataset, UMADatasetEmbedding)(task))
        return jax.nn.silu(self.mix(jnp.concatenate(parts, axis=-1)))


def case_reduce(
    values: Array, mask: Array, case_count: int, atom_capacity: int, /
) -> Array:
    """Per-case sums of masked flat case-major atom rows ``(N, ...) -> (B, ...)``."""
    zero = jnp.zeros((), dtype=values.dtype)
    shaped = jnp.where(
        mask.reshape(mask.shape + (1,) * (values.ndim - 1)), values, zero
    ).reshape((case_count, atom_capacity) + values.shape[1:])
    return jnp.sum(shaped, axis=1)


def case_atom_counts(mask: Array, case_count: int, atom_capacity: int, /) -> Array:
    return jnp.sum(mask.reshape(case_count, atom_capacity), axis=1).astype(jnp.int32)


def case_sums(values: Array, mask: Array, atom_cases: Array, case_count: int, /) -> Array:
    """Per-case sums ``(N, ...) -> (B, ...)`` of masked rows that name their case.

    Domain owner rows are not case-major slabs: row ``i`` belongs to case
    ``atom_cases[i]``. Each case selects its active rows through a masked
    ``(N, B)`` indicator and sums them along the row axis (no duplicate-index
    scatter-add); integer values stay exact.
    """
    flat = values.reshape((values.shape[0], -1))
    cases = jnp.arange(case_count, dtype=atom_cases.dtype)
    select = (atom_cases[:, None] == cases[None, :]) & mask[:, None]
    zero = jnp.zeros((), dtype=values.dtype)
    sums = jnp.sum(jnp.where(select[:, :, None], flat[:, None, :], zero), axis=0)
    return sums.reshape((case_count,) + values.shape[1:])


@final
class UMAMoLERouter(StrictModule, ParameterOwner):
    """Source eSCN-MoE routing to ``(B, E)`` expert coefficients.

    ``composition`` is the optional ``(max_elements, C)`` species table; the
    routing MLP is three biased linears each followed by SiLU.
    """

    linears: tuple[UMALinear, UMALinear, UMALinear]
    composition: Array | None
    composition_policy: UMACompositionPolicy = eqx.field(static=True)
    normalization: UMACoefficientNorm = eqx.field(static=True)

    def __init__(
        self,
        linears: Sequence[UMALinear],
        /,
        *,
        composition: ArrayLike | None,
        composition_policy: UMACompositionPolicy,
        normalization: UMACoefficientNorm,
    ) -> None:
        linears_ = tuple(linears)
        if len(linears_) != 3 or any(linear.bias is None for linear in linears_):
            raise ValueError("The MoLE routing MLP has three biased linears.")
        for previous, current in zip(linears_[:-1], linears_[1:]):
            if current.in_features != previous.out_features:
                raise ValueError("MoLE routing widths are inconsistent.")
        composition_ = None if composition is None else jnp.asarray(composition)
        if composition_ is not None and composition_.ndim != 2:
            raise ValueError("The MoLE composition embedding must be a species table.")
        first, second, third = linears_
        self.linears = (first, second, third)
        self.composition = composition_
        self.composition_policy = parse(
            composition_policy, UMACompositionPolicy, "composition policy"
        )
        self.normalization = parse(normalization, UMACoefficientNorm, "coefficient rule")

    @property
    def expert_count(self) -> int:
        return self.linears[-1].out_features

    def composition_embedding(self, counts: Array, /) -> Array:
        """Per-case composition embeddings from species counts ``(B, Z)``."""
        assert self.composition is not None
        dtype = self.composition.dtype
        total = jnp.sum(counts, axis=-1, keepdims=True).astype(dtype)
        summed = counts.astype(dtype) @ self.composition
        match self.composition_policy:
            case "mean":
                return jnp.where(total > 0, summed / jnp.maximum(total, 1.0), 0.0)
            case "sum-over-count-plus-one":
                return summed / (total + 1.0)

    def coefficients(
        self, system: Array, counts: Array | None, active_cases: Array, /
    ) -> Array:
        """Expert coefficients ``(B, E)`` of system embeddings and species counts."""
        parts = []
        if self.composition is not None:
            if counts is None:
                raise ValueError("Composition routing needs species counts.")
            parts.append(self.composition_embedding(counts))
        parts.append(system)
        hidden = jnp.concatenate(parts, axis=-1)
        for linear in self.linears:
            hidden = jax.nn.silu(linear(hidden))
        match self.normalization:
            case "softmax-plus-0.005":
                return jax.nn.softmax(hidden, axis=-1) + 0.005
            case "pnorm-batch-count":
                # Source _pnorm: normalize(|x| + 2 / x.shape[0], p=1); the offset
                # uses the number of systems in the evaluated batch.
                count = jnp.maximum(jnp.sum(active_cases), 1).astype(hidden.dtype)
                shifted = jnp.abs(hidden) + 2.0 / count
                total = jnp.sum(jnp.abs(shifted), axis=-1, keepdims=True)
                return shifted / jnp.maximum(total, 1.0e-12)


def species_counts(
    species: Array, mask: Array, case_count: int, atom_capacity: int, elements: int, /
) -> Array:
    """Per-case species counts ``(B, elements)`` of active atoms."""
    one_hot = jax.nn.one_hot(species, elements, dtype=jnp.int32)
    return case_reduce(one_hot, mask, case_count, atom_capacity)


def balance_channels(
    values: Array,
    target: Array,
    mask: Array,
    case_count: int,
    atom_capacity: int,
    channels: tuple[int, int],
    offset: float,
    /,
) -> Array:
    """Source ``balance_channels_batched`` on scalar channels ``start:end``.

    Each case's active atoms receive ``-(sum - (target - offset)) / n``.
    """
    start, end = channels
    if end <= start:
        return values
    sums = case_reduce(values[:, 0, start:end], mask, case_count, atom_capacity)
    counts = case_atom_counts(mask, case_count, atom_capacity)
    correction = channel_balance_correction(sums, counts, target, offset)
    return apply_channel_balance(
        values, jnp.repeat(correction, atom_capacity, axis=0), mask, channels
    )


def channel_balance_correction(
    sums: Array, counts: Array, target: Array, offset: float, /
) -> Array:
    """Per-case shift ``(sum - (target - offset)) / n`` of balanced channels ``(B, k)``.

    ``sums`` are the case's complete channel sums and ``counts`` its active
    atom counts; a case without atoms divides by one and shifts no atom.
    """
    goal = (target.astype(sums.dtype) - offset)[:, None]
    return (sums - goal) / jnp.maximum(counts, 1)[:, None].astype(sums.dtype)


def apply_channel_balance(
    values: Array, correction: Array, mask: Array, channels: tuple[int, int], /
) -> Array:
    """Subtract each active row's case shift ``(N, k)`` from scalar channels ``start:end``."""
    start, end = channels
    selected = values[:, 0, start:end]
    updated = jnp.where(mask[:, None], selected - correction, selected)
    return values.at[:, 0, start:end].set(updated)


@final
class UMAMoLEBinding(StrictModule, NonTrainableState):
    """Exact composition and conditions one merged MoLE realization is bound to.

    Under the ``"mean"`` composition policy the routing depends only on the
    reduced composition, which is stored divided by the gcd of its counts and
    compared exactly (``counts * sum(bound) == bound * sum(counts)``). Under
    ``"sum-over-count-plus-one"`` it depends on the absolute counts, which must
    be equal. ``batch_count`` is the batch size entering the ``pnorm`` offset.
    """

    __strict_contract__ = True

    composition: tuple[tuple[int, int], ...] = eqx.field(static=True)
    total_charge: int = eqx.field(static=True)
    spin_index: int = eqx.field(static=True)
    task: int = eqx.field(static=True)
    composition_policy: UMACompositionPolicy = eqx.field(static=True)
    batch_count: int = eqx.field(static=True)
    binding_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        composition: Mapping[int, int],
        /,
        *,
        total_charge: int,
        spin_index: int,
        task: int,
        composition_policy: UMACompositionPolicy,
        batch_count: int = 1,
    ) -> None:
        items = sorted(
            (integer(key, "element"), integer(value, "count"))
            for key, value in composition.items()
        )
        if not items or any(element < 0 or count <= 0 for element, count in items):
            raise ValueError("A MoLE binding needs positive species counts.")
        policy = parse(composition_policy, UMACompositionPolicy, "composition policy")
        if policy == "mean":
            divisor = math.gcd(*(count for _, count in items))
            items = [(element, count // divisor) for element, count in items]
        charge = integer(total_charge, "total_charge")
        spin = integer(spin_index, "spin_index")
        if not UMA_CHARGE_RANGE[0] <= charge <= UMA_CHARGE_RANGE[1] or not (
            UMA_SPIN_RANGE[0] <= spin <= UMA_SPIN_RANGE[1]
        ):
            raise ValueError("MoLE binding conditions are outside the source ranges.")
        self.composition = tuple(items)
        self.total_charge = charge
        self.spin_index = spin
        self.task = integer(task, "task")
        self.composition_policy = policy
        self.batch_count = positive_integer(batch_count, "batch_count")
        self.binding_id = canonical_fingerprint(
            {
                "kind": "uma-mole-binding",
                "composition": [list(item) for item in self.composition],
                "total_charge": charge,
                "spin_index": spin,
                "task": self.task,
                "composition_policy": policy,
                "batch_count": self.batch_count,
            }
        )

    def validate(self) -> None:
        """Replay the constructor on these fields; refuse a non-canonical binding.

        A restored binding bypasses the constructor, so its normalization
        (sorted unique positive counts, gcd-reduced under ``"mean"``), source
        condition ranges, batch count and ``binding_id`` are re-derived and
        must reproduce every recorded field exactly.
        """
        replay = UMAMoLEBinding(
            dict(self.composition),
            total_charge=self.total_charge,
            spin_index=self.spin_index,
            task=self.task,
            composition_policy=self.composition_policy,
            batch_count=self.batch_count,
        )
        require_identical(self, replay, "The MoLE binding", ValueError)

    def counts(self, elements: int, /) -> np.ndarray:
        vector = np.zeros((elements,), dtype=np.int64)
        for element, count in self.composition:
            if element >= elements:
                raise ValueError("A bound element exceeds the model's element table.")
            vector[element] = count
        return vector

    def matches(self, counts: Array, charge: Array, spin: Array, task: Array, /) -> Array:
        """Per-case exact agreement of species counts ``(B, Z)`` and conditions."""
        bound = jnp.asarray(self.counts(counts.shape[-1]), dtype=counts.dtype)
        if self.composition_policy == "mean":
            total = jnp.sum(counts, axis=-1, keepdims=True)
            composition = jnp.all(counts * jnp.sum(bound) == bound * total, axis=-1)
        else:
            composition = jnp.all(counts == bound, axis=-1)
        return (
            composition
            & (charge == self.total_charge)
            & (spin == self.spin_index)
            & (task == self.task)
        )


for _type in (
    UMAConditionSchema,
    UMAFrozen,
    UMAChargeSpinEmbedding,
    UMADatasetEmbedding,
    UMASystemEmbedding,
    UMAMoLERouter,
    UMAMoLEBinding,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_type.__name__}", _type)


__all__ = [
    "apply_channel_balance",
    "balance_channels",
    "case_atom_counts",
    "case_reduce",
    "case_sums",
    "channel_balance_correction",
    "species_counts",
    "UMA_CHARGE_FIELD",
    "UMA_CHARGE_RANGE",
    "UMA_SPIN_FIELD",
    "UMA_SPIN_INDEX_PORT",
    "UMA_SPIN_RANGE",
    "UMA_TASK_FIELD",
    "UMA_TASK_PORT",
    "UMAChargeSpinEmbedding",
    "UMAChargeSpinEmbeddingKind",
    "UMACoefficientNorm",
    "UMACompositionPolicy",
    "UMAConditionSchema",
    "UMADatasetEmbedding",
    "UMAFrozen",
    "UMAMoLEBinding",
    "UMAMoLERouter",
    "UMASystemEmbedding",
]

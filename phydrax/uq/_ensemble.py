#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .._frozendict import frozendict
from .._model import FrozenModel
from .._strict import StrictModule
from ..nn._base import _AbstractBaseModel
from ..nn._keys import EvalKey, split_eval_key
from ..nn.operator.data import OperatorBatch
from ..nn.operator.protocols import OperatorModel
from ._operator import operator_predictive_from_samples, OperatorPredictiveField
from ._predictive import _sample_validity, PredictiveField, SampleAxis


class HomogeneousFunctionEnsemble(StrictModule):
    """One member-axis-stacked PyTree constructed or validated as homogeneous."""

    model: Any
    num_members: int
    source_dim: str

    def __init__(
        self,
        model: Any,
        num_members: int,
        /,
        *,
        source_dim: str = "__phydra_uq_epistemic",
    ):
        count = int(num_members)
        if count <= 0:
            raise ValueError("num_members must be positive.")
        if not isinstance(source_dim, str) or not source_dim:
            raise ValueError("source_dim must be a non-empty string.")
        for leaf in jax.tree_util.tree_leaves(model):
            if eqx.is_array(leaf) and (leaf.ndim == 0 or int(leaf.shape[0]) != count):
                raise ValueError(
                    "Every homogeneous ensemble array leaf must have a leading "
                    f"member axis of size {count}."
                )
        self.model = model
        self.num_members = count
        self.source_dim = source_dim

    @classmethod
    def from_factory(
        cls,
        factory: Callable[[Any], Any],
        /,
        *,
        num_members: int,
        key,
        source_dim: str = "__phydra_uq_epistemic",
    ) -> "HomogeneousFunctionEnsemble":
        count = int(num_members)
        if count <= 0:
            raise ValueError("num_members must be positive.")
        member_keys = jr.split(key, count)
        model = eqx.filter_vmap(factory)(member_keys)
        return cls(model, count, source_dim=source_dim)

    @classmethod
    def from_members(
        cls,
        members: Sequence[Any],
        /,
        *,
        source_dim: str = "__phydra_uq_epistemic",
    ) -> "HomogeneousFunctionEnsemble":
        members_tuple = tuple(members)
        if not members_tuple:
            raise ValueError("members must be non-empty.")
        model = _stack_homogeneous_members(members_tuple)
        return cls(model, len(members_tuple), source_dim=source_dim)

    from_solvers = from_members

    def predict(
        self,
        points: Any,
        /,
        *,
        key,
        variable: str | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> PredictiveField:
        member_keys = jr.split(key, self.num_members)
        template_member = _take_member(self.model, 0)
        template = _evaluate_field(
            template_member, points, variable=variable, key=member_keys[0], **kwargs
        )

        def evaluate(member, member_key):
            return _evaluate_field(
                member,
                points,
                variable=variable,
                key=member_key,
                **kwargs,
            ).data

        data = eqx.filter_vmap(
            evaluate,
            in_axes=(eqx.if_array(0), 0),
        )(self.model, member_keys)
        return _predictive_from_member_data(
            data,
            template,
            self.source_dim,
            valid_policy=valid_policy,
            owner="Homogeneous ensemble prediction",
        )

    def predict_operator(
        self,
        batch: OperatorBatch,
        /,
        *,
        key,
        field_name: str,
        query_name: str,
        input_sample_axes: Sequence[str] = (),
        valid_policy: Literal["record", "raise"] = "record",
    ) -> OperatorPredictiveField:
        """Evaluate homogeneous members on one operator source/query batch."""
        if not isinstance(batch, OperatorBatch):
            raise TypeError("batch must be an OperatorBatch.")
        template_member = _take_member(self.model, 0)
        if not isinstance(template_member, OperatorModel):
            raise TypeError(
                "Homogeneous operator ensembles require operator-protocol members."
            )
        member_keys = jr.split(key, self.num_members)
        template_prediction = template_member.evaluate(batch, key=member_keys[0])
        template_field = template_prediction.field(field_name)
        if template_field.query_name != query_name:
            raise ValueError(
                f"Output field {field_name!r} is bound to query "
                f"{template_field.query_name!r}, not {query_name!r}."
            )

        def evaluate(member, member_key):
            return member.evaluate(batch, key=member_key).field(field_name).values

        data = eqx.filter_vmap(
            evaluate,
            in_axes=(eqx.if_array(0), 0),
        )(self.model, member_keys)
        return operator_predictive_from_samples(
            data,
            batch,
            template_field.spec,
            sample_axes=(SampleAxis(self.source_dim, "epistemic"),),
            field_name=field_name,
            query_name=query_name,
            input_sample_axes=input_sample_axes,
            valid_policy=valid_policy,
        )

    def predict_many(
        self,
        variables: Sequence[str],
        points: Any,
        /,
        *,
        key,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> frozendict[str, PredictiveField]:
        selected = tuple(str(variable) for variable in variables)
        if not selected:
            raise ValueError("variables must be non-empty.")
        return frozendict(
            {
                variable: self.predict(
                    points,
                    key=key,
                    variable=variable,
                    valid_policy=valid_policy,
                    **kwargs,
                )
                for variable in selected
            }
        )


class HeterogeneousFunctionEnsemble(StrictModule):
    """Tuple fallback for members with different PyTrees or static configuration."""

    members: tuple[Any, ...]
    source_dim: str

    def __init__(
        self,
        members: Sequence[Any],
        /,
        *,
        source_dim: str = "__phydra_uq_epistemic",
    ):
        values = tuple(members)
        if not values:
            raise ValueError("members must be non-empty.")
        if not isinstance(source_dim, str) or not source_dim:
            raise ValueError("source_dim must be a non-empty string.")
        self.members = values
        self.source_dim = source_dim

    @property
    def num_members(self) -> int:
        return len(self.members)

    def predict(
        self,
        points: Any,
        /,
        *,
        key,
        variable: str | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> PredictiveField:
        fields = tuple(
            _evaluate_field(
                member,
                points,
                variable=variable,
                key=member_key,
                **kwargs,
            )
            for member, member_key in zip(
                self.members, jr.split(key, self.num_members), strict=True
            )
        )
        first = fields[0]
        for field in fields[1:]:
            if field.dims != first.dims or field.data.shape != first.data.shape:
                raise ValueError(
                    "Heterogeneous ensemble predictions must have aligned field structure."
                )
        data = jnp.stack(tuple(jnp.asarray(field.data) for field in fields), axis=0)
        return _predictive_from_member_data(
            data,
            first,
            self.source_dim,
            valid_policy=valid_policy,
            owner="Heterogeneous ensemble prediction",
        )

    def predict_operator(
        self,
        batch: OperatorBatch,
        /,
        *,
        key,
        field_name: str,
        query_name: str,
        input_sample_axes: Sequence[str] = (),
        valid_policy: Literal["record", "raise"] = "record",
    ) -> OperatorPredictiveField:
        """Evaluate heterogeneous operator members on one aligned batch."""
        if not isinstance(batch, OperatorBatch):
            raise TypeError("batch must be an OperatorBatch.")
        if any(not isinstance(member, OperatorModel) for member in self.members):
            raise TypeError(
                "Heterogeneous operator ensembles require operator-protocol members."
            )
        operator_members = tuple(self.members)
        first = operator_members[0]
        first_prediction = first.evaluate(
            batch,
            key=jr.fold_in(key, self.num_members),
        )
        first_field = first_prediction.field(field_name)
        if first_field.query_name != query_name:
            raise ValueError(
                f"Output field {field_name!r} is bound to query "
                f"{first_field.query_name!r}, not {query_name!r}."
            )
        first_spec = first_field.spec
        predictions = tuple(
            member.evaluate(batch, key=member_key).field(field_name)
            for member, member_key in zip(
                operator_members,
                jr.split(key, self.num_members),
                strict=True,
            )
        )
        for prediction in predictions:
            if prediction.query_name != query_name:
                raise ValueError(
                    f"Output field {field_name!r} has inconsistent query bindings."
                )
            spec = prediction.spec
            if (
                spec.channels != first_spec.channels
                or spec.component_names != first_spec.component_names
                or prediction.values.shape != predictions[0].values.shape
            ):
                raise ValueError(
                    "Heterogeneous operator predictions must have aligned output "
                    "specifications and shapes."
                )
        data = jnp.stack(
            tuple(jnp.asarray(prediction.values) for prediction in predictions),
            axis=0,
        )
        return operator_predictive_from_samples(
            data,
            batch,
            first_spec,
            sample_axes=(SampleAxis(self.source_dim, "epistemic"),),
            field_name=field_name,
            query_name=query_name,
            input_sample_axes=input_sample_axes,
            valid_policy=valid_policy,
        )

    def predict_many(
        self,
        variables: Sequence[str],
        points: Any,
        /,
        *,
        key,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> frozendict[str, PredictiveField]:
        selected = tuple(str(variable) for variable in variables)
        if not selected:
            raise ValueError("variables must be non-empty.")
        return frozendict(
            {
                variable: self.predict(
                    points,
                    key=key,
                    variable=variable,
                    valid_policy=valid_policy,
                    **kwargs,
                )
                for variable in selected
            }
        )




class RandomizedPriorModel(_AbstractBaseModel):
    """Trainable model plus an independently initialized, structurally frozen prior."""

    learned: _AbstractBaseModel
    prior: FrozenModel
    beta: float
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        learned: _AbstractBaseModel,
        prior: _AbstractBaseModel,
        /,
        *,
        beta: float = 1.0,
    ):
        if learned.in_size != prior.in_size or learned.out_size != prior.out_size:
            raise ValueError(
                "Learned and prior models must have matching input/output sizes."
            )
        coefficient = float(beta)
        if not jnp.isfinite(coefficient):
            raise ValueError("beta must be finite.")
        self.learned = learned
        self.prior = FrozenModel(prior)
        self.beta = coefficient
        self.in_size = learned.in_size
        self.out_size = learned.out_size

    def __call__(self, x, /, *, key: EvalKey = None):
        learned_key, prior_key = split_eval_key(key, 2)
        return self.learned(x, key=learned_key) + self.beta * self.prior(x, key=prior_key)


def randomized_prior_ensemble(
    factory: Callable[[Any], _AbstractBaseModel],
    /,
    *,
    num_members: int,
    key,
    beta: float = 1.0,
    homogeneous: bool = True,
    source_dim: str = "__phydra_uq_epistemic",
) -> HomogeneousFunctionEnsemble | HeterogeneousFunctionEnsemble:
    """Initialize independent learned/prior pairs from distinct keys."""
    count = int(num_members)
    if count <= 0:
        raise ValueError("num_members must be positive.")
    members = []
    for member_key in jr.split(key, count):
        learned_key, prior_key = jr.split(member_key, 2)
        members.append(
            RandomizedPriorModel(factory(learned_key), factory(prior_key), beta=beta)
        )
    if homogeneous:
        return HomogeneousFunctionEnsemble.from_members(members, source_dim=source_dim)
    return HeterogeneousFunctionEnsemble(members, source_dim=source_dim)


class EnsembleMemberDiagnostics(StrictModule):
    """Deterministic identity, duration, and solver diagnostics for one fitted member."""

    member_index: int
    seed: int
    duration_seconds: float
    training_diagnostics: frozendict[str, Any]

    def __init__(
        self,
        *,
        member_index: int,
        seed: int,
        duration_seconds: float,
        training_diagnostics: Mapping[str, Any] | None = None,
    ):
        self.member_index = int(member_index)
        self.seed = int(seed)
        self.duration_seconds = float(duration_seconds)
        self.training_diagnostics = frozendict(training_diagnostics or {})


class EnsembleFitResult(StrictModule):
    """A fitted ensemble together with memberwise training diagnostics."""

    ensemble: HomogeneousFunctionEnsemble | HeterogeneousFunctionEnsemble
    members: tuple[EnsembleMemberDiagnostics, ...]

    def __init__(
        self,
        ensemble: HomogeneousFunctionEnsemble | HeterogeneousFunctionEnsemble,
        members: Sequence[EnsembleMemberDiagnostics],
        /,
    ):
        diagnostics = tuple(members)
        if len(diagnostics) != ensemble.num_members:
            raise ValueError("Member diagnostics must align with the fitted ensemble.")
        self.ensemble = ensemble
        self.members = diagnostics

    @property
    def total_duration_seconds(self) -> float:
        return sum(member.duration_seconds for member in self.members)


class EnsembleFitError(RuntimeError):
    """Member-indexed ensemble training failure with completed diagnostics."""

    def __init__(
        self,
        *,
        member_index: int,
        seed: int,
        duration_seconds: float,
        completed: Sequence[EnsembleMemberDiagnostics],
    ):
        super().__init__(
            f"Ensemble member {member_index} failed during fitting with seed {seed}."
        )
        self.member_index = int(member_index)
        self.seed = int(seed)
        self.duration_seconds = float(duration_seconds)
        self.completed = tuple(completed)


def fit_ensemble(
    factory: Callable[[Any], Any],
    /,
    *,
    num_members: int,
    key,
    solve_kwargs: Mapping[str, Any] | None = None,
    homogeneous: bool = True,
    source_dim: str = "__phydra_uq_epistemic",
    return_diagnostics: bool = False,
) -> HomogeneousFunctionEnsemble | HeterogeneousFunctionEnsemble | EnsembleFitResult:
    """Fit independent solver members, optionally retaining memberwise diagnostics."""
    count = int(num_members)
    if count <= 0:
        raise ValueError("num_members must be positive.")
    kwargs = dict(solve_kwargs or {})
    members = []
    diagnostics: list[EnsembleMemberDiagnostics] = []
    for member_index, member_key in enumerate(jr.split(key, count)):
        init_key, seed_key = jr.split(member_key, 2)
        member_kwargs = dict(kwargs)
        member_kwargs.setdefault(
            "seed",
            int(jr.randint(seed_key, (), 0, jnp.iinfo(jnp.int32).max)),
        )
        seed = int(member_kwargs["seed"])
        started = time.perf_counter()
        try:
            solver = factory(init_key)
            solve = getattr(solver, "solve", None)
            if not callable(solve):
                raise TypeError(
                    "fit_ensemble factory must return an object with solve()."
                )
            member = solve(**member_kwargs)
            if return_diagnostics:
                jax.block_until_ready(member)
        except Exception as exc:
            raise EnsembleFitError(
                member_index=member_index,
                seed=seed,
                duration_seconds=time.perf_counter() - started,
                completed=diagnostics,
            ) from exc
        members.append(member)
        if return_diagnostics:
            training_diagnostics = getattr(member, "training_diagnostics", {})
            if not isinstance(training_diagnostics, Mapping):
                raise TypeError("Member training_diagnostics must be a mapping.")
            diagnostics.append(
                EnsembleMemberDiagnostics(
                    member_index=member_index,
                    seed=seed,
                    duration_seconds=time.perf_counter() - started,
                    training_diagnostics=training_diagnostics,
                )
            )
    if homogeneous:
        ensemble = HomogeneousFunctionEnsemble.from_members(
            members, source_dim=source_dim
        )
    else:
        ensemble = HeterogeneousFunctionEnsemble(members, source_dim=source_dim)
    if return_diagnostics:
        return EnsembleFitResult(ensemble, diagnostics)
    return ensemble


def _evaluate_field(
    member: Any,
    points: Any,
    /,
    *,
    variable: str | None,
    key,
    **kwargs: Any,
) -> cx.Field:
    ansatz = getattr(member, "ansatz_functions", None)
    if callable(ansatz):
        functions = ansatz()
    elif isinstance(member, Mapping):
        functions = member
    else:
        functions = None
    if functions is not None:
        if variable is None:
            if len(functions) != 1:
                raise ValueError("variable is required for a multi-field ensemble.")
            variable = next(iter(functions))
        if variable not in functions:
            raise KeyError(f"Unknown ensemble field {variable!r}.")
        value = functions[variable](points, key=key, **kwargs)
    elif callable(member):
        if variable is not None:
            raise ValueError(
                "variable is only valid for solver or field-mapping members."
            )
        value = member(points, key=key, **kwargs)
    else:
        raise TypeError("Ensemble members must be callable, field mappings, or solvers.")
    if isinstance(value, cx.Field):
        return value
    array = jnp.asarray(value)
    return cx.Field(array, dims=(None,) * array.ndim)


def _predictive_from_member_data(
    data: Any,
    template: cx.Field,
    source_dim: str,
    *,
    valid_policy: Literal["record", "raise"],
    owner: str,
) -> PredictiveField:
    sample_data = jnp.asarray(data)
    samples = cx.Field(sample_data, dims=(source_dim, *template.dims))
    valid = _sample_validity(
        sample_data,
        sample_dim=source_dim,
        valid_policy=valid_policy,
        owner=owner,
    )
    return PredictiveField(
        samples,
        (SampleAxis(source_dim, "epistemic"),),
        valid=valid,
    )


def _take_member(tree: Any, index: int) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf[index] if eqx.is_array(leaf) else leaf,
        tree,
    )


def _stack_homogeneous_members(members: tuple[Any, ...]) -> Any:
    dynamic_static = tuple(eqx.partition(member, eqx.is_array) for member in members)
    dynamic = tuple(item[0] for item in dynamic_static)
    static = tuple(item[1] for item in dynamic_static)
    reference_structure = jax.tree_util.tree_structure(dynamic[0])
    for item in dynamic[1:]:
        if jax.tree_util.tree_structure(item) != reference_structure:
            raise ValueError(
                "Homogeneous ensemble members have different PyTree structures."
            )
    for item in static[1:]:
        equal = eqx.tree_equal(static[0], item)
        if not bool(equal):
            raise ValueError("Homogeneous ensemble members have different static leaves.")

    def stack(*leaves):
        if leaves[0] is None:
            return None
        return jnp.stack(leaves, axis=0)

    stacked = jax.tree_util.tree_map(
        stack,
        *dynamic,
        is_leaf=lambda leaf: leaf is None,
    )
    return eqx.combine(stacked, static[0])


__all__ = [
    "EnsembleFitError",
    "EnsembleFitResult",
    "EnsembleMemberDiagnostics",
    "FrozenModel",
    "HeterogeneousFunctionEnsemble",
    "HomogeneousFunctionEnsemble",
    "RandomizedPriorModel",
    "fit_ensemble",
    "randomized_prior_ensemble",
]

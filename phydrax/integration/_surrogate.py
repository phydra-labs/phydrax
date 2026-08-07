#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction, ProbabilityDomain

from .._strict import StrictModule
from ..stochastic._hierarchy import StochasticHierarchy, StochasticLevelSpec
from ._api import integrate
from ._multilevel import MultilevelSampleBatch
from ._plans import SparseGridPlan
from ._targets import MultilevelTarget, over


if TYPE_CHECKING:
    from ..operators.interpolation._smolyak import SmolyakInterpolant


SmolyakInputSampler: TypeAlias = Callable[[Array, Key[Array, ""]], tuple[Array, ...]]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _smolyak_interpolant(surrogate: DomainFunction, /) -> SmolyakInterpolant:
    from ..operators.interpolation._smolyak import SmolyakInterpolant

    if not isinstance(surrogate, DomainFunction) or not isinstance(
        surrogate.func, SmolyakInterpolant
    ):
        raise TypeError(
            "surrogate must be a DomainFunction returned by interpolate_smolyak."
        )
    return surrogate.func


def _block_arrays(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        if eqx.is_array(leaf):
            jax.block_until_ready(leaf)


def _finite_samples(values: Array, /) -> Array:
    axes = tuple(range(1, values.ndim))
    finite = jnp.isfinite(values)
    return jnp.all(finite, axis=axes) if axes else finite


def _evaluate_model(model: Any, coordinates: tuple[Array, ...], /) -> Array:
    function = model.func if isinstance(model, DomainFunction) else model
    if not callable(function):
        raise TypeError("A surrogate-hierarchy model must be callable.")
    return jnp.asarray(jax.vmap(function)(*coordinates))


class SmolyakProbabilityInputSampler(StrictModule):
    """Prefix-addressed independent sampling for Smolyak probability axes."""

    factors: tuple[ProbabilityDomain, ...]
    axis_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        factors: Sequence[ProbabilityDomain],
        /,
        *,
        axis_labels: Sequence[str],
    ):
        values = tuple(factors)
        labels = tuple(str(label) for label in axis_labels)
        if not values or any(
            not isinstance(value, ProbabilityDomain) for value in values
        ):
            raise TypeError(
                "SmolyakProbabilityInputSampler requires only ProbabilityDomain factors."
            )
        if len(labels) != len(values) or any(not label for label in labels):
            raise ValueError("axis_labels must align with the probability factors.")
        self.factors = values
        self.axis_labels = labels

    @classmethod
    def from_surrogate(
        cls,
        surrogate: DomainFunction,
        /,
    ) -> "SmolyakProbabilityInputSampler":
        interpolant = _smolyak_interpolant(surrogate)
        if any(
            not isinstance(factor, ProbabilityDomain) for factor in interpolant.factors
        ):
            raise TypeError(
                "Automatic surrogate input sampling requires every interpolation axis "
                "to be a ProbabilityDomain. Supply input_sampler explicitly otherwise."
            )
        return cls(
            tuple(interpolant.factors),
            axis_labels=interpolant.axis_labels,
        )

    def __call__(
        self,
        sample_indices: Array,
        root_key: Key[Array, ""],
        /,
    ) -> tuple[Array, ...]:
        indices = jnp.asarray(sample_indices, dtype=jnp.uint32).reshape((-1,))
        path_keys = jax.vmap(lambda index: jr.fold_in(root_key, index))(indices)
        coordinates: list[Array] = []
        for axis, factor in enumerate(self.factors):
            axis_keys = jax.vmap(lambda key: jr.fold_in(key, axis))(path_keys)
            values = jax.vmap(
                lambda key: factor.distribution.sample(key, sample_shape=())
            )(axis_keys)
            values = jnp.asarray(values)
            if values.shape != indices.shape:
                raise ValueError(
                    "Automatic Smolyak input sampling requires scalar distributions."
                )
            coordinates.append(values)
        return tuple(coordinates)


class SmolyakSurrogateHierarchyAdapter(StrictModule):
    """Two-level MLMC adapter for a deterministic Smolyak control surrogate.

    Level zero estimates the surrogate expectation, or reuses an externally computed
    deterministic expectation. Level one evaluates fine and surrogate models at exactly
    the same uncertain inputs. Level namespaces are independent, while global sample
    indices remain prefix-stable under changed batch sizes.
    """

    surrogate: DomainFunction
    fine_model: Any
    input_sampler: SmolyakInputSampler
    hierarchy: StochasticHierarchy
    surrogate_expectation: Array | None
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        surrogate: DomainFunction,
        fine_model: Any,
        /,
        *,
        problem_id: str,
        observable_id: str,
        hierarchy_id: str,
        sampler_id: str,
        fine_solver_id: str,
        fine_approximation_id: str,
        input_sampler: SmolyakInputSampler | None = None,
        surrogate_expectation: ArrayLike | None = None,
    ):
        interpolant = _smolyak_interpolant(surrogate)
        if not callable(fine_model):
            raise TypeError("fine_model must be callable.")
        sampler = (
            SmolyakProbabilityInputSampler.from_surrogate(surrogate)
            if input_sampler is None
            else input_sampler
        )
        if not callable(sampler):
            raise TypeError("input_sampler must be callable.")
        problem = _identifier(problem_id, "problem_id")
        observable = _identifier(observable_id, "observable_id")
        hierarchy_name = _identifier(hierarchy_id, "hierarchy_id")
        sampler_name = _identifier(sampler_id, "sampler_id")
        fine_solver = _identifier(fine_solver_id, "fine_solver_id")
        fine_approximation = _identifier(
            fine_approximation_id,
            "fine_approximation_id",
        )
        output_shape = interpolant.output_shape or (1,)
        coarse_id = f"{hierarchy_name}:smolyak"
        fine_id = f"{hierarchy_name}:fine"
        coarse = StochasticLevelSpec(
            coarse_id,
            0,
            refinement_axes=("surrogate",),
            resolutions=(2.0,),
            state_shape=output_shape,
            problem_id=problem,
            observable_id=observable,
            solver_id="smolyak-interpolation",
            approximation_id=f"smolyak-level-{interpolant.level}",
            noise_coupling="shared",
            metadata={
                "surrogate_family": "smolyak",
                "surrogate_level": str(interpolant.level),
            },
        )
        fine = StochasticLevelSpec(
            fine_id,
            1,
            refinement_axes=("surrogate",),
            resolutions=(1.0,),
            state_shape=output_shape,
            problem_id=problem,
            observable_id=observable,
            solver_id=fine_solver,
            approximation_id=fine_approximation,
            parent_level_id=coarse_id,
            noise_coupling="shared",
            metadata={"surrogate_family": "fine-model"},
        )
        expectation = (
            None if surrogate_expectation is None else jnp.asarray(surrogate_expectation)
        )
        if expectation is not None and expectation.shape != interpolant.output_shape:
            raise ValueError(
                "surrogate_expectation shape must equal the Smolyak output shape."
            )
        self.surrogate = surrogate
        self.fine_model = fine_model
        self.input_sampler = sampler
        self.hierarchy = StochasticHierarchy(
            (coarse, fine),
            hierarchy_id=hierarchy_name,
        )
        self.surrogate_expectation = expectation
        self.sampler_id = sampler_name

    @property
    def target(self) -> MultilevelTarget:
        return MultilevelTarget(
            self.hierarchy,
            self.sample,
            sampler_id=self.sampler_id,
        )

    @staticmethod
    def observable(samples: Any, level: StochasticLevelSpec, /) -> Any:
        del level
        return samples

    def _coordinates(
        self,
        level_index: int,
        sample_indices: Array,
        root_key: Key[Array, ""],
        /,
    ) -> tuple[Array, ...]:
        level_key = jr.fold_in(root_key, level_index)
        coordinates = tuple(self.input_sampler(sample_indices, level_key))
        dimension = _smolyak_interpolant(self.surrogate).axis_labels
        if len(coordinates) != len(dimension):
            raise ValueError("input_sampler returned the wrong number of coordinates.")
        expected = (int(sample_indices.size),)
        if any(jnp.asarray(value).shape != expected for value in coordinates):
            raise ValueError(
                "Every sampled surrogate coordinate must have the leading sample shape."
            )
        return tuple(jnp.asarray(value) for value in coordinates)

    def sample(
        self,
        level_index: int,
        sample_indices: Array,
        root_key: Key[Array, ""],
        /,
    ) -> MultilevelSampleBatch:
        level = int(level_index)
        if level not in (0, 1):
            raise ValueError("The Smolyak surrogate hierarchy has exactly two levels.")
        indices = jnp.asarray(sample_indices, dtype=jnp.int64).reshape((-1,))
        if indices.size == 0:
            raise ValueError("Surrogate hierarchy sampling requires non-empty indices.")
        count = int(indices.size)
        started = perf_counter()
        if level == 0 and self.surrogate_expectation is not None:
            values = jnp.broadcast_to(
                self.surrogate_expectation,
                (count,) + self.surrogate_expectation.shape,
            )
            _block_arrays(values)
            elapsed = perf_counter() - started
            cost = max(elapsed / count, jnp.finfo(float).tiny)
            return MultilevelSampleBatch(
                values,
                None,
                indices,
                cost,
                level_index=0,
                provenance=self.sampler_id,
            )
        coordinates = self._coordinates(level, indices, root_key)
        coarse = _evaluate_model(self.surrogate, coordinates)
        if coarse.shape[:1] != (count,):
            raise ValueError("The Smolyak surrogate must preserve the sample axis.")
        if level == 0:
            _block_arrays(coarse)
            elapsed = perf_counter() - started
            cost = max(elapsed / count, jnp.finfo(float).tiny)
            return MultilevelSampleBatch(
                coarse,
                None,
                indices,
                cost,
                level_index=0,
                fine_valid=_finite_samples(coarse),
                provenance=self.sampler_id,
            )
        fine = _evaluate_model(self.fine_model, coordinates)
        if fine.shape != coarse.shape:
            raise ValueError(
                "Fine and Smolyak models must return identical sample/output shapes."
            )
        _block_arrays((fine, coarse))
        elapsed = perf_counter() - started
        cost = max(elapsed / count, jnp.finfo(float).tiny)
        return MultilevelSampleBatch(
            fine,
            coarse,
            indices,
            cost,
            level_index=1,
            fine_valid=_finite_samples(fine),
            coarse_valid=_finite_samples(coarse),
            pair_ids=indices,
            provenance=self.sampler_id,
        )


def smolyak_surrogate_expectation(
    surrogate: DomainFunction,
    /,
    *,
    quadrature_level: int | None = None,
):
    """Integrate a Smolyak surrogate over its declared physical/probability axes."""

    interpolant = _smolyak_interpolant(surrogate)
    level = (
        max(interpolant.level + 1, 2)
        if quadrature_level is None
        else int(quadrature_level)
    )
    if level < 1:
        raise ValueError("quadrature_level must be positive.")
    factors = tuple(surrogate.domain.factor(label) for label in surrogate.deps)
    dependency_domain = factors[0]
    for factor in factors[1:]:
        dependency_domain = dependency_domain.join(factor)
    function = DomainFunction(
        domain=dependency_domain,
        deps=surrogate.deps,
        func=surrogate.func,
        metadata=surrogate.metadata,
    )
    rules = tuple(
        "gauss-hermite" if rule == "gauss-hermite" else "clenshaw-curtis"
        for rule in interpolant.axis_rules
    )
    return integrate(
        function,
        over(dependency_domain.component()),
        SparseGridPlan(
            len(factors),
            level,
            anisotropy=interpolant.anisotropy,
            axis_rules=rules,
        ),
    )


__all__ = [
    "SmolyakInputSampler",
    "SmolyakProbabilityInputSampler",
    "smolyak_surrogate_expectation",
    "SmolyakSurrogateHierarchyAdapter",
]

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import abc
from collections.abc import Callable, Iterator, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Key

from .._doc import DOC_KEY0
from .._sampling import get_sampler_host, seed_from_key
from ._coordinate import CoordinateSpec
from ._domain import JointFactor
from ._factor_component import FactorComponent
from ._measure import BaseMeasure, ExactMass
from ._selection import Boundary, Fixed, FixedEnd, FixedStart, Interior, Selection


class AbstractScalarDomain(JointFactor):
    @property
    @abc.abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.label,)

    @property
    def coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        return (CoordinateSpec((), kind="scalar", differentiable=True),)

    def bind_component(
        self,
        selections: Mapping[str, Selection],
        /,
    ) -> FactorComponent:
        if tuple(selections) != self.labels:
            raise ValueError(
                f"Scalar factor {self.labels} requires exactly one ordered selection."
            )
        selection = selections[self.label]
        return FactorComponent(
            factor=self,
            selections=selections,
            measure=self._component_base_measure(selection),
        )

    @abc.abstractmethod
    def _component_base_measure(self, selection: Selection, /) -> BaseMeasure:
        raise NotImplementedError

    def _replace_labels(
        self,
        labels: tuple[str, ...],
        /,
    ) -> "AbstractScalarDomain":
        return eqx.tree_at(lambda factor: factor._label, self, labels[0])

    @property
    @abc.abstractmethod
    def measure(self) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def fixed(self, which: str, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self,
        num_points: int,
        *,
        sampler: str,
        key: Key[Array, ""],
    ) -> Array:
        raise NotImplementedError


class ScalarInterval(AbstractScalarDomain):
    r"""A closed finite 1D scalar interval domain.

    Represents the domain

    $$
    \Omega_{\ell} = [a, b],
    $$

    with label given by `label`. The measure is the interval length $|\Omega_{\ell}|=b-a$.

    Sampling draws points $x_i\in[a,b]$ using the chosen sampler (uniform, Latin
    hypercube, Sobol, etc.), optionally filtered by a predicate `where(x)`.
    """

    start: Array
    end: Array
    _label: str

    def __init__(
        self,
        start: float,
        end: float,
        *,
        label: str = "t",
    ):
        start_arr = jnp.asarray(start, dtype=float).reshape(())
        end_arr = jnp.asarray(end, dtype=float).reshape(())
        if bool(start_arr >= end_arr):
            raise ValueError("`start` must be less than `end`.")

        self.start = start_arr
        self.end = end_arr
        self._label = str(label)

    @property
    def label(self) -> str:
        return self._label

    @property
    def bounds(self) -> Iterator[Array]:
        r"""Return an iterator over $(a, b)$."""
        return iter((self.start, self.end))

    @property
    def extent(self) -> Array:
        r"""Return the interval length $b-a$."""
        return self.end - self.start

    @property
    def measure(self) -> Array:
        r"""Return the measure $|\Omega_{\ell}|$ (equal to `extent`)."""
        return self.extent

    def _component_base_measure(self, selection: Selection, /) -> BaseMeasure:
        if isinstance(selection, Interior):
            return BaseMeasure("lebesgue", ExactMass(self.measure))
        if isinstance(selection, Boundary):
            if selection.tags is not None or selection.entity_ids is not None:
                raise ValueError("Scalar boundaries do not support tags or entity IDs.")
            return BaseMeasure("counting", ExactMass(2.0))
        if isinstance(selection, (FixedStart, FixedEnd)):
            return BaseMeasure("dirac", ExactMass(1.0), normalized=True)
        if isinstance(selection, Fixed):
            if selection.value.ndim != 0:
                raise ValueError("A scalar Fixed selection requires a scalar value.")
            if not bool(jnp.asarray(self._contains(selection.value))):
                raise ValueError(
                    f"Fixed value {selection.value} lies outside factor {self.labels}."
                )
            return BaseMeasure("dirac", ExactMass(1.0), normalized=True)
        raise TypeError(f"Unsupported scalar selection {type(selection).__name__}.")

    def fixed(self, which: str, /) -> Array:
        if which == "start":
            return self.start
        if which == "end":
            return self.end
        raise ValueError("fixed(which) must be 'start' or 'end'.")

    def _same_factor_support(self, other: object, /) -> bool:
        if not isinstance(other, ScalarInterval):
            return False
        start_eq = np.isclose(
            np.asarray(self.start),
            np.asarray(other.start),
            rtol=1e-6,
            atol=1e-8,
        )
        end_eq = np.isclose(
            np.asarray(self.end),
            np.asarray(other.end),
            rtol=1e-6,
            atol=1e-8,
        )
        return bool(start_eq) and bool(end_eq)

    def sample(
        self,
        num_points: int,
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        def _sample_host(num_points, sampler, where, key):
            rng = np.random.default_rng(seed_from_key(key))
            sampler_fn = get_sampler_host(sampler, dim=1, seed=rng)
            samples = np.empty((0, 1), dtype=float)

            while samples.shape[0] < num_points:
                remaining_points = num_points - samples.shape[0]
                samples_ = sampler_fn(int(remaining_points))

                if where:
                    samples_scaled = self.start + jnp.asarray(samples_, dtype=float) * (
                        self.end - self.start
                    )
                    mask = np.asarray(
                        jax.vmap(where)(samples_scaled), dtype=bool
                    ).reshape((-1,))
                    samples_ = samples_[mask]
                    samples = np.vstack((samples, samples_))
                else:
                    samples = np.vstack((samples, samples_))

            return samples[:num_points].reshape((-1,))

        zeros = jnp.zeros((num_points,), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

        sampled = eqx.filter_pure_callback(
            _sample_host,
            num_points,
            sampler,
            where,
            key,
            result_shape_dtypes=shape_dtype,
        )

        sampled = sampled * (self.end - self.start) + self.start
        return sampled

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        return (self.start <= points) & (points <= self.end)

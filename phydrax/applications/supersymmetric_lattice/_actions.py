#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._fields import (
    ComplexifiedPFormField,
    invert_gauge_transform,
    PFormLatticePlan,
    transform_p_form,
)


def _mul(left: Array, right: Array, /) -> Array:
    return ein.contract("...ij,...jk->...ik", left, right)


def _trace(value: Array, /) -> Array:
    return ein.contract("...ii->...", value)


def _shift(value: Array, axis: int, amount: int, /) -> Array:
    return jnp.roll(value, -amount, axis=axis)


class TwistedSYMPlan(StrictModule):
    """Immutable finite-lattice resource and coupling plan for twisted Yang--Mills."""

    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    matrix_rank: int = eqx.field(static=True)
    coupling: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    maximum_field_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice_shape: Sequence[int],
        /,
        *,
        matrix_rank: int,
        coupling: float,
        lattice_spacing: float = 1.0,
        maximum_field_elements: int = 10_000_000,
    ):
        shape = tuple(int(size) for size in lattice_shape)
        rank = int(matrix_rank)
        coupling_ = float(coupling)
        spacing = float(lattice_spacing)
        maximum = int(maximum_field_elements)
        if not shape or any(size < 1 for size in shape):
            raise ValueError("lattice_shape must contain positive extents.")
        if rank < 1 or coupling_ <= 0.0 or spacing <= 0.0:
            raise ValueError(
                "matrix_rank, coupling, and lattice_spacing must be positive."
            )
        if not np.isfinite(coupling_) or not np.isfinite(spacing):
            raise ValueError("Twisted-SYM scalar parameters must be finite.")
        required = 2 * prod(shape) * len(shape) * rank * rank
        if maximum < 1 or required > maximum:
            raise ValueError(
                f"Twisted-SYM links require {required} scalar elements; capacity is "
                f"{maximum}."
            )
        self.lattice_shape = shape
        self.matrix_rank = rank
        self.coupling = coupling_
        self.lattice_spacing = spacing
        self.maximum_field_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-complexified-twisted-sym-plan",
                "lattice_shape": shape,
                "matrix_rank": rank,
                "coupling": coupling_,
                "lattice_spacing": spacing,
                "maximum_field_elements": maximum,
            }
        )


class TwistedSYMConfiguration(StrictModule):
    """Independent forward and reverse complexified gauge links."""

    links: ComplexifiedPFormField
    reverse_links: ComplexifiedPFormField

    def __init__(
        self,
        links: ComplexifiedPFormField,
        reverse_links: ComplexifiedPFormField,
        /,
    ):
        if not isinstance(links, ComplexifiedPFormField) or not isinstance(
            reverse_links, ComplexifiedPFormField
        ):
            raise TypeError("links and reverse_links must be complexified p-form fields.")
        if links.plan.degree != 1 or reverse_links.plan.degree != 1:
            raise ValueError("Twisted-SYM gauge links must be one-forms.")
        if links.orientation != "forward" or reverse_links.orientation != "reverse":
            raise ValueError("Twisted-SYM links require forward/reverse orientations.")
        if (
            links.plan.plan_id != reverse_links.plan.plan_id
            or links.matrix_rank != reverse_links.matrix_rank
        ):
            raise ValueError("Forward and reverse links must share one field space.")
        self.links = links
        self.reverse_links = reverse_links


class PreparedTwistedSYMAction(StrictModule):
    """Prepared Q-exact bosonic twisted-SYM reference on one periodic lattice."""

    plan: TwistedSYMPlan = eqx.field(static=True)
    link_plan: PFormLatticePlan
    curvature_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    prefactor: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        plan: TwistedSYMPlan,
        link_plan: PFormLatticePlan,
        curvature_pairs: tuple[tuple[int, int], ...],
        prefactor: Array,
        /,
    ):
        self.plan = plan
        self.link_plan = link_plan
        self.curvature_pairs = curvature_pairs
        self.prefactor = prefactor
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-complexified-twisted-sym",
                "plan": plan.plan_id,
                "link_plan": link_plan.plan_id,
                "curvature_pairs": curvature_pairs,
                "convention": "forward-reverse-q-exact-bosonic",
            }
        )
        self.claim = "finite-regulated-reference-only"

    def configuration(
        self, links: ArrayLike, reverse_links: ArrayLike, /
    ) -> TwistedSYMConfiguration:
        return TwistedSYMConfiguration(
            ComplexifiedPFormField(self.link_plan, links, orientation="forward"),
            ComplexifiedPFormField(self.link_plan, reverse_links, orientation="reverse"),
        )

    def field_strengths(
        self, configuration: TwistedSYMConfiguration, /
    ) -> tuple[Array, Array]:
        self._validate(configuration)
        forward = configuration.links.values
        reverse = configuration.reverse_links.values
        strengths = []
        reverse_strengths = []
        for first, second in self.curvature_pairs:
            first_link = forward[..., first, :, :]
            second_link = forward[..., second, :, :]
            forward_curvature = _mul(first_link, _shift(second_link, first, 1)) - _mul(
                second_link, _shift(first_link, second, 1)
            )
            first_reverse = reverse[..., first, :, :]
            second_reverse = reverse[..., second, :, :]
            reverse_curvature = _mul(
                _shift(second_reverse, first, 1), first_reverse
            ) - _mul(_shift(first_reverse, second, 1), second_reverse)
            strengths.append(forward_curvature)
            reverse_strengths.append(reverse_curvature)
        shape = self.plan.lattice_shape + (
            0,
            self.plan.matrix_rank,
            self.plan.matrix_rank,
        )
        forward_result = (
            jnp.stack(strengths, axis=-3)
            if strengths
            else jnp.zeros(shape, dtype=forward.dtype)
        )
        reverse_result = (
            jnp.stack(reverse_strengths, axis=-3)
            if reverse_strengths
            else jnp.zeros(shape, dtype=reverse.dtype)
        )
        return forward_result, reverse_result

    def gauge_divergence(self, configuration: TwistedSYMConfiguration, /) -> Array:
        self._validate(configuration)
        forward = configuration.links.values
        reverse = configuration.reverse_links.values
        divergence = jnp.zeros(
            self.plan.lattice_shape + (self.plan.matrix_rank, self.plan.matrix_rank),
            dtype=forward.dtype,
        )
        for axis in range(len(self.plan.lattice_shape)):
            link = forward[..., axis, :, :]
            reverse_link = reverse[..., axis, :, :]
            incoming_link = _shift(link, axis, -1)
            incoming_reverse = _shift(reverse_link, axis, -1)
            divergence = (
                divergence
                + _mul(link, reverse_link)
                - _mul(incoming_reverse, incoming_link)
            )
        return divergence

    def action(self, configuration: TwistedSYMConfiguration, /) -> Array:
        forward, reverse = self.field_strengths(configuration)
        curvature = jnp.sum(jnp.real(_trace(_mul(reverse, forward))))
        divergence = self.gauge_divergence(configuration)
        auxiliary = 0.5 * jnp.sum(jnp.real(_trace(_mul(divergence, divergence))))
        return self.prefactor * (curvature + auxiliary)

    def _validate(self, configuration: TwistedSYMConfiguration, /) -> None:
        if not isinstance(configuration, TwistedSYMConfiguration):
            raise TypeError("configuration must be TwistedSYMConfiguration.")
        if (
            configuration.links.plan.plan_id != self.link_plan.plan_id
            or configuration.links.matrix_rank != self.plan.matrix_rank
        ):
            raise ValueError("Twisted-SYM configuration does not match the preparation.")


def prepare_twisted_sym(plan: TwistedSYMPlan, /) -> PreparedTwistedSYMAction:
    """Prepare immutable placement and normalization data before runtime evaluation."""
    if not isinstance(plan, TwistedSYMPlan):
        raise TypeError("plan must be TwistedSYMPlan.")
    link_plan = PFormLatticePlan(
        plan.lattice_shape,
        1,
        maximum_field_elements=plan.maximum_field_elements // 2,
    )
    pairs = tuple(combinations(range(len(plan.lattice_shape)), 2))
    prefactor = jnp.asarray(
        1.0
        / (
            plan.coupling
            * plan.coupling
            * plan.lattice_spacing ** (4 - len(plan.lattice_shape))
        )
    )
    return PreparedTwistedSYMAction(plan, link_plan, pairs, prefactor)


def transform_twisted_configuration(
    configuration: TwistedSYMConfiguration,
    gauge: ArrayLike,
    /,
    *,
    inverse_gauge: ArrayLike | None = None,
) -> TwistedSYMConfiguration:
    if not isinstance(configuration, TwistedSYMConfiguration):
        raise TypeError("configuration must be TwistedSYMConfiguration.")
    inverse_ = invert_gauge_transform(gauge) if inverse_gauge is None else inverse_gauge
    return TwistedSYMConfiguration(
        transform_p_form(configuration.links, gauge, inverse_gauge=inverse_),
        transform_p_form(configuration.reverse_links, gauge, inverse_gauge=inverse_),
    )


class BFSSPlan(StrictModule):
    """Immutable finite Euclidean matrix-quantum-mechanics plan."""

    time_slices: int = eqx.field(static=True)
    matrix_count: int = eqx.field(static=True)
    matrix_rank: int = eqx.field(static=True)
    coupling: float = eqx.field(static=True)
    time_spacing: float = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    maximum_field_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_slices: int,
        matrix_count: int,
        matrix_rank: int,
        /,
        *,
        coupling: float,
        time_spacing: float,
        mass: float = 0.0,
        maximum_field_elements: int = 10_000_000,
    ):
        slices = int(time_slices)
        count = int(matrix_count)
        rank = int(matrix_rank)
        coupling_ = float(coupling)
        spacing = float(time_spacing)
        mass_ = float(mass)
        maximum = int(maximum_field_elements)
        scalars = 2 * slices * (count + 1) * rank * rank
        if slices < 2 or count < 1 or rank < 1:
            raise ValueError(
                "BFSS dimensions must be positive and time_slices at least two."
            )
        if coupling_ <= 0.0 or spacing <= 0.0 or mass_ < 0.0:
            raise ValueError(
                "BFSS coupling/spacing must be positive and mass nonnegative."
            )
        if not all(np.isfinite(value) for value in (coupling_, spacing, mass_)):
            raise ValueError("BFSS scalar parameters must be finite.")
        if maximum < 1 or scalars > maximum:
            raise ValueError(
                f"BFSS configuration requires {scalars} scalar elements; capacity is {maximum}."
            )
        self.time_slices = slices
        self.matrix_count = count
        self.matrix_rank = rank
        self.coupling = coupling_
        self.time_spacing = spacing
        self.mass = mass_
        self.maximum_field_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-complexified-bfss-plan",
                "time_slices": slices,
                "matrix_count": count,
                "matrix_rank": rank,
                "coupling": coupling_,
                "time_spacing": spacing,
                "mass": mass_,
                "maximum_field_elements": maximum,
            }
        )


class BFSSConfiguration(StrictModule):
    matrices: Array
    dual_matrices: Array
    temporal_links: Array
    reverse_temporal_links: Array

    def __init__(
        self,
        matrices: ArrayLike,
        dual_matrices: ArrayLike,
        temporal_links: ArrayLike,
        reverse_temporal_links: ArrayLike,
        /,
    ):
        matrices_ = jnp.asarray(matrices)
        dual_ = jnp.asarray(dual_matrices, dtype=matrices_.dtype)
        forward = jnp.asarray(temporal_links, dtype=matrices_.dtype)
        reverse = jnp.asarray(reverse_temporal_links, dtype=matrices_.dtype)
        if matrices_.ndim != 4 or matrices_.shape[-1] != matrices_.shape[-2]:
            raise ValueError("matrices must have shape (time, count, rank, rank).")
        if dual_.shape != matrices_.shape:
            raise ValueError("dual_matrices must match matrices.")
        expected_links = (matrices_.shape[0], matrices_.shape[-1], matrices_.shape[-1])
        if forward.shape != expected_links or reverse.shape != expected_links:
            raise ValueError("temporal links must have shape (time, rank, rank).")
        if not jnp.issubdtype(matrices_.dtype, jnp.complexfloating):
            matrices_ = matrices_.astype(jnp.complex128)
            dual_ = dual_.astype(jnp.complex128)
            forward = forward.astype(jnp.complex128)
            reverse = reverse.astype(jnp.complex128)
        self.matrices = matrices_
        self.dual_matrices = dual_
        self.temporal_links = forward
        self.reverse_temporal_links = reverse


class PreparedBFSSAction(StrictModule):
    plan: BFSSPlan = eqx.field(static=True)
    prefactor: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(self, plan: BFSSPlan, prefactor: Array, /):
        self.plan = plan
        self.prefactor = prefactor
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-complexified-bfss-action",
                "plan": plan.plan_id,
                "convention": "forward-reverse-periodic-euclidean",
            }
        )
        self.claim = "finite-regulated-reference-only"

    def configuration(
        self,
        matrices: ArrayLike,
        dual_matrices: ArrayLike,
        temporal_links: ArrayLike,
        reverse_temporal_links: ArrayLike,
        /,
    ) -> BFSSConfiguration:
        configuration = BFSSConfiguration(
            matrices, dual_matrices, temporal_links, reverse_temporal_links
        )
        self._validate(configuration)
        return configuration

    def action(self, configuration: BFSSConfiguration, /) -> Array:
        self._validate(configuration)
        x = configuration.matrices
        dual = configuration.dual_matrices
        links = configuration.temporal_links
        reverse_links = configuration.reverse_temporal_links
        next_x = jnp.roll(x, -1, axis=0)
        next_dual = jnp.roll(dual, -1, axis=0)
        derivative = _mul(links[:, None], next_x) - _mul(x, links[:, None])
        dual_derivative = _mul(next_dual, reverse_links[:, None]) - _mul(
            reverse_links[:, None], dual
        )
        kinetic = jnp.sum(jnp.real(_trace(_mul(dual_derivative, derivative))))
        potential = jnp.asarray(0.0, dtype=kinetic.dtype)
        for first, second in combinations(range(self.plan.matrix_count), 2):
            commutator = _mul(x[:, first], x[:, second]) - _mul(x[:, second], x[:, first])
            dual_commutator = _mul(dual[:, second], dual[:, first]) - _mul(
                dual[:, first], dual[:, second]
            )
            potential = potential + jnp.sum(
                jnp.real(_trace(_mul(dual_commutator, commutator)))
            )
        mass_term = jnp.sum(jnp.real(_trace(_mul(dual, x))))
        spacing = self.plan.time_spacing
        reduced = kinetic / (spacing * spacing) + 0.5 * potential
        reduced = reduced + 0.5 * self.plan.mass * self.plan.mass * mass_term
        return self.prefactor * spacing * reduced

    def _validate(self, configuration: BFSSConfiguration, /) -> None:
        if not isinstance(configuration, BFSSConfiguration):
            raise TypeError("configuration must be BFSSConfiguration.")
        expected = (
            self.plan.time_slices,
            self.plan.matrix_count,
            self.plan.matrix_rank,
            self.plan.matrix_rank,
        )
        if configuration.matrices.shape != expected:
            raise ValueError(f"BFSS matrices must have shape {expected}.")


def prepare_bfss(plan: BFSSPlan, /) -> PreparedBFSSAction:
    if not isinstance(plan, BFSSPlan):
        raise TypeError("plan must be BFSSPlan.")
    prefactor = jnp.asarray(1.0 / (plan.coupling * plan.coupling))
    return PreparedBFSSAction(plan, prefactor)


def transform_bfss_configuration(
    configuration: BFSSConfiguration,
    gauge: ArrayLike,
    /,
    *,
    inverse_gauge: ArrayLike | None = None,
) -> BFSSConfiguration:
    if not isinstance(configuration, BFSSConfiguration):
        raise TypeError("configuration must be BFSSConfiguration.")
    inverse_ = invert_gauge_transform(gauge) if inverse_gauge is None else inverse_gauge
    time_slices, _, rank, _ = configuration.matrices.shape
    scalar_plan = PFormLatticePlan(
        (time_slices,), 0, maximum_field_elements=time_slices * rank * rank
    )
    link_plan = PFormLatticePlan(
        (time_slices,), 1, maximum_field_elements=time_slices * rank * rank
    )
    matrices = []
    dual_matrices = []
    for component in range(configuration.matrices.shape[1]):
        matrices.append(
            transform_p_form(
                ComplexifiedPFormField(
                    scalar_plan,
                    configuration.matrices[:, component, None],
                    orientation="forward",
                ),
                gauge,
                inverse_gauge=inverse_,
            ).values[:, 0]
        )
        dual_matrices.append(
            transform_p_form(
                ComplexifiedPFormField(
                    scalar_plan,
                    configuration.dual_matrices[:, component, None],
                    orientation="forward",
                ),
                gauge,
                inverse_gauge=inverse_,
            ).values[:, 0]
        )
    forward = transform_p_form(
        ComplexifiedPFormField(
            link_plan, configuration.temporal_links[:, None], orientation="forward"
        ),
        gauge,
        inverse_gauge=inverse_,
    ).values[:, 0]
    reverse = transform_p_form(
        ComplexifiedPFormField(
            link_plan,
            configuration.reverse_temporal_links[:, None],
            orientation="reverse",
        ),
        gauge,
        inverse_gauge=inverse_,
    ).values[:, 0]
    return BFSSConfiguration(
        jnp.stack(matrices, axis=1),
        jnp.stack(dual_matrices, axis=1),
        forward,
        reverse,
    )


__all__ = [
    "BFSSConfiguration",
    "BFSSPlan",
    "PreparedBFSSAction",
    "PreparedTwistedSYMAction",
    "TwistedSYMConfiguration",
    "TwistedSYMPlan",
    "prepare_bfss",
    "prepare_twisted_sym",
    "transform_bfss_configuration",
    "transform_twisted_configuration",
]

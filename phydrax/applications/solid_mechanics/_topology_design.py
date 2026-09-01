#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import PreparedDensityTransform


def _real_array(name: str, value: ArrayLike, /, *, nonempty: bool = True) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be real-valued and inexact.")
    if nonempty and array.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    return array


def _tree_inner(left: PyTree[Any], right: PyTree[Any], /) -> Array:
    if jax.tree.structure(left) != jax.tree.structure(right):
        raise ValueError("Load and mechanics state must have the same PyTree structure.")
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if not left_leaves:
        raise ValueError("Load and mechanics state PyTrees must contain an array leaf.")
    terms = tuple(
        jnp.real(jnp.vdot(jnp.asarray(load), jnp.asarray(state)))
        for load, state in zip(left_leaves, right_leaves, strict=True)
    )
    return sum(terms[1:], start=terms[0])


class DensityTransform(StrictModule, NonTrainableState):
    """Prepared physical filter and finite-beta projection for topology design.

    The raw design remains the optimization variable. Filtering and projection are
    always differentiable; hard thresholding is deliberately not part of this object.
    """

    prepared: PreparedDensityTransform
    beta: Array
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedDensityTransform,
        /,
        *,
        beta: ArrayLike = 1.0,
    ):
        if not isinstance(prepared, PreparedDensityTransform):
            raise TypeError("prepared must be a PreparedDensityTransform.")
        beta_ = np.asarray(beta)
        if (
            beta_.shape != ()
            or not np.issubdtype(beta_.dtype, np.number)
            or np.issubdtype(beta_.dtype, np.complexfloating)
        ):
            raise TypeError("beta must be one real scalar array.")
        beta_value = float(beta_)
        if not isfinite(beta_value) or beta_value <= 0.0:
            raise ValueError("beta must be finite and strictly positive.")
        self.prepared = prepared
        self.beta = jnp.asarray(beta_value, dtype=float)
        plan = prepared.plan.filter
        self.transform_id = canonical_fingerprint(
            {
                "kind": "topology-density-transform",
                "coordinates": array_tree_fingerprint(plan.coordinates),
                "measures": array_tree_fingerprint(plan.measures),
                "design_mask": array_tree_fingerprint(plan.design_mask),
                "fixed_density": array_tree_fingerprint(plan.fixed_density),
                "radius": plan.radius,
                "eta": float(prepared.plan.projection.eta),
            }
        )

    @property
    def measures(self) -> Array:
        return self.prepared.plan.filter.measures

    @property
    def design_mask(self) -> Array:
        return self.prepared.plan.filter.design_mask

    def filtered(self, raw_density: ArrayLike, /) -> Array:
        """Return the contextual conic-filter output before projection."""

        return self.prepared.filter.apply(raw_density)

    def apply(self, raw_density: ArrayLike, beta: ArrayLike | None = None, /) -> Array:
        """Return physical density at the supplied continuation parameter."""

        selected_beta = self.beta if beta is None else beta
        return self.prepared.apply(raw_density, selected_beta)

    def volume_ratio(
        self,
        raw_density: ArrayLike,
        beta: ArrayLike | None = None,
        /,
    ) -> Array:
        physical = self.apply(raw_density, beta)
        return jnp.sum(physical * self.measures) / jnp.sum(self.measures)


class MaterialInterpolation(StrictModule, NonTrainableState):
    """SIMP interpolation of one scalar or vector of material parameters."""

    minimum: Array
    solid: Array
    penalty: Array
    interpolation_id: str = eqx.field(static=True)

    def __init__(
        self,
        solid: ArrayLike,
        /,
        *,
        minimum: ArrayLike = 1.0e-9,
        penalty: ArrayLike = 3.0,
    ):
        solid_ = np.asarray(solid)
        minimum_ = np.asarray(minimum)
        penalty_ = np.asarray(penalty)
        if solid_.shape != minimum_.shape:
            raise ValueError("solid and minimum material parameters must have one shape.")
        if penalty_.shape != ():
            raise ValueError("penalty must be one scalar array.")
        if not (
            np.issubdtype(solid_.dtype, np.number)
            and not np.issubdtype(solid_.dtype, np.complexfloating)
            and np.issubdtype(minimum_.dtype, np.number)
            and not np.issubdtype(minimum_.dtype, np.complexfloating)
            and np.issubdtype(penalty_.dtype, np.number)
            and not np.issubdtype(penalty_.dtype, np.complexfloating)
        ):
            raise TypeError("Material interpolation parameters must be real-valued.")
        solid_value = np.asarray(solid_, dtype=float)
        minimum_value = np.asarray(minimum_, dtype=float)
        exponent = float(penalty_)
        if (
            np.any(~np.isfinite(solid_value))
            or np.any(~np.isfinite(minimum_value))
            or np.any(minimum_value <= 0.0)
            or np.any(solid_value <= minimum_value)
            or not isfinite(exponent)
            or exponent <= 0.0
        ):
            raise ValueError(
                "Material interpolation requires finite 0 < minimum < solid and "
                "a finite positive penalty."
            )
        self.minimum = jnp.asarray(minimum_value)
        self.solid = jnp.asarray(solid_value)
        self.penalty = jnp.asarray(exponent)
        self.interpolation_id = canonical_fingerprint(
            {
                "kind": "topology-material-interpolation",
                "minimum": array_tree_fingerprint(self.minimum),
                "solid": array_tree_fingerprint(self.solid),
            }
        )

    def __call__(
        self,
        physical_density: ArrayLike,
        penalty: ArrayLike | None = None,
        /,
    ) -> Array:
        density = jnp.asarray(physical_density)
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density) | (density < 0.0) | (density > 1.0)),
            "Physical density must be finite and lie in [0, 1].",
        )
        exponent = self.penalty if penalty is None else jnp.asarray(penalty)
        if exponent.shape != ():
            raise ValueError("penalty must be one scalar array.")
        exponent = eqx.error_if(
            exponent,
            ~jnp.isfinite(exponent) | (exponent <= 0.0),
            "penalty must be finite and strictly positive.",
        )
        if self.solid.shape == ():
            density_power = density**exponent
        else:
            density_power = (
                density.reshape(density.shape + (1,) * self.solid.ndim) ** exponent
            )
        return self.minimum + density_power * (self.solid - self.minimum)


class LoadCase(StrictModule, NonTrainableState):
    """One mechanics load and its scalar response functional."""

    load: PyTree[Array]
    context: Any
    objective: Callable | None = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        load: PyTree[ArrayLike],
        /,
        *,
        objective: Callable | None = None,
        context: Any = None,
        weight: float = 1.0,
        case_id: str,
    ):
        if objective is not None and not callable(objective):
            raise TypeError("objective must be callable or None.")
        leaves = jax.tree.leaves(load)
        if not leaves:
            raise ValueError("load must contain at least one array leaf.")
        load_ = jax.tree.map(lambda value: _real_array("load", value), load)
        weight_ = float(weight)
        identifier = str(case_id)
        if not isfinite(weight_) or weight_ <= 0.0:
            raise ValueError("Load-case weight must be finite and strictly positive.")
        if not identifier:
            raise ValueError("case_id must be non-empty.")
        self.load = load_
        self.context = context
        self.objective = objective
        self.weight = weight_
        self.case_id = identifier

    def value(self, state: PyTree[Any], material: Array, args: Any = None, /) -> Array:
        if self.objective is None:
            value = _tree_inner(self.load, state)
        else:
            value = jnp.asarray(self.objective(state, material, self, args))
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("A load-case objective must return one real scalar array.")
        return value


class Aggregation(StrictModule, NonTrainableState):
    """Differentiable multi-load aggregation with deterministic tie sensitivities."""

    kind: Literal["weighted_sum", "maximum", "p_norm", "ks_max"] = eqx.field(static=True)
    parameter: float = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["weighted_sum", "maximum", "p_norm", "ks_max"] = "weighted_sum",
        /,
        *,
        parameter: float = 8.0,
    ):
        kind_ = str(kind)
        parameter_ = float(parameter)
        if kind_ not in ("weighted_sum", "maximum", "p_norm", "ks_max"):
            raise ValueError("Unknown load aggregation kind.")
        if not isfinite(parameter_) or parameter_ <= 1.0:
            raise ValueError("Aggregation parameter must be finite and greater than one.")
        self.kind = kind_
        self.parameter = parameter_

    def __call__(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        /,
    ) -> Array:
        values_ = _real_array("load values", values)
        if values_.ndim != 1:
            raise ValueError("load values must have shape (load_case_count,).")
        weights_ = jnp.ones_like(values_) if weights is None else jnp.asarray(weights)
        if weights_.shape != values_.shape:
            raise ValueError("Aggregation weights must have the load-values shape.")
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_) | (weights_ <= 0.0)),
            "Aggregation weights must be finite and strictly positive.",
        )
        if self.kind == "weighted_sum":
            return jnp.sum(weights_ * values_)
        if self.kind == "maximum":
            return jnp.max(weights_ * values_)
        if self.kind == "p_norm":
            values_ = eqx.error_if(
                values_,
                jnp.any(values_ < 0.0),
                "p-norm load aggregation requires non-negative values.",
            )
            normalized_weights = weights_ / jnp.sum(weights_)
            return jnp.sum(normalized_weights * values_**self.parameter) ** (
                1.0 / self.parameter
            )
        maximum = jnp.max(values_)
        normalized_weights = weights_ / jnp.sum(weights_)
        return (
            maximum
            + jnp.log(
                jnp.sum(
                    normalized_weights * jnp.exp(self.parameter * (values_ - maximum))
                )
            )
            / self.parameter
        )

    def sensitivities(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        /,
    ) -> Array:
        values_ = jnp.asarray(values)
        return jax.grad(lambda current: self(current, weights))(values_)


class PeriodicHomogenizationCase(StrictModule, NonTrainableState):
    """Macroscopic strain program for one periodic representative cell solve."""

    macroscopic_strain: Array
    case_id: str = eqx.field(static=True)

    def __init__(self, macroscopic_strain: ArrayLike, /, *, case_id: str):
        strain = _real_array("macroscopic_strain", macroscopic_strain)
        identifier = str(case_id)
        if strain.ndim not in (1, 2) or not identifier:
            raise ValueError(
                "macroscopic_strain must be a vector or matrix and case_id non-empty."
            )
        self.macroscopic_strain = strain
        self.case_id = identifier

    def as_load_case(
        self,
        load: PyTree[ArrayLike],
        objective: Callable,
        /,
        *,
        weight: float = 1.0,
    ) -> LoadCase:
        """Bind the macroscopic strain program to one periodic FE load case."""

        if not callable(objective):
            raise TypeError("objective must be callable.")
        return LoadCase(
            load,
            objective=objective,
            context=self,
            weight=weight,
            case_id=self.case_id,
        )


class HillMandelEvidence(StrictModule):
    """Periodic compatibility and microscopic/macroscopic power equivalence."""

    microscopic_power: Array
    macroscopic_power: Array
    power_defect: Array
    power_threshold: Array
    periodic_residual_norm: Array
    periodic_threshold: Array
    finite: Array
    accepted: Array


def hill_mandel_evidence(
    microscopic_stress: ArrayLike,
    microscopic_strain: ArrayLike,
    quadrature_weights: ArrayLike,
    macroscopic_stress: ArrayLike,
    macroscopic_strain: ArrayLike,
    periodic_residual: ArrayLike,
    /,
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
    periodic_tolerance: float = 1.0e-10,
) -> HillMandelEvidence:
    """Audit a periodic-cell root with the Hill–Mandel macrohomogeneity identity."""

    stress = _real_array("microscopic_stress", microscopic_stress)
    strain = _real_array("microscopic_strain", microscopic_strain)
    weights = _real_array("quadrature_weights", quadrature_weights)
    macro_stress = _real_array("macroscopic_stress", macroscopic_stress)
    macro_strain = _real_array("macroscopic_strain", macroscopic_strain)
    periodic = _real_array("periodic_residual", periodic_residual, nonempty=False)
    if stress.shape != strain.shape or stress.ndim < 2:
        raise ValueError("Microscopic stress and strain must share (point, ...) shape.")
    if weights.shape != (stress.shape[0],):
        raise ValueError("quadrature_weights must contain one value per point.")
    if macro_stress.shape != macro_strain.shape or macro_stress.shape != stress.shape[1:]:
        raise ValueError("Macroscopic tensors must match microscopic trailing shape.")
    tolerances = (
        float(relative_tolerance),
        float(absolute_tolerance),
        float(periodic_tolerance),
    )
    if any(not isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("Hill–Mandel tolerances must be finite and non-negative.")
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights) | (weights <= 0.0)),
        "Quadrature weights must be finite and strictly positive.",
    )
    point_power = jnp.sum(stress * strain, axis=tuple(range(1, stress.ndim)))
    microscopic_power = jnp.sum(weights * point_power) / jnp.sum(weights)
    macroscopic_power = jnp.sum(macro_stress * macro_strain)
    defect = jnp.abs(microscopic_power - macroscopic_power)
    power_threshold = tolerances[1] + tolerances[0] * jnp.abs(macroscopic_power)
    periodic_norm = jnp.sqrt(jnp.sum(periodic**2))
    finite = (
        jnp.all(jnp.isfinite(stress))
        & jnp.all(jnp.isfinite(strain))
        & jnp.all(jnp.isfinite(macro_stress))
        & jnp.all(jnp.isfinite(macro_strain))
        & jnp.isfinite(periodic_norm)
    )
    accepted = finite & (defect <= power_threshold) & (periodic_norm <= tolerances[2])
    return HillMandelEvidence(
        microscopic_power,
        macroscopic_power,
        defect,
        jnp.asarray(power_threshold),
        periodic_norm,
        jnp.asarray(tolerances[2], dtype=periodic_norm.dtype),
        finite,
        accepted,
    )


__all__ = [
    "Aggregation",
    "DensityTransform",
    "HillMandelEvidence",
    "LoadCase",
    "MaterialInterpolation",
    "PeriodicHomogenizationCase",
    "hill_mandel_evidence",
]

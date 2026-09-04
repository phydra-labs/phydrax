#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState


_LESFormula = Literal["smagorinsky", "wale", "vreman", "amd"]
_LESFilterFamily = Literal[
    "sharp-fourier-projection",
    "implicit-grid-volume",
    "explicit-filter",
]
_LESTopology = Literal["tensor-product", "unstructured"]
_LESBoundaryClass = Literal["periodic", "wall-bounded", "open", "mixed"]
_LESScaleRule = Literal[
    "cutoff-equivalent",
    "volume-equivalent",
    "kernel-equivalent",
]
_LESCommutationStatus = Literal["commuting", "modeled", "unmodeled"]
_LESRepeatedFilterSemantics = Literal["idempotent", "composed", "unmodeled"]
_LESParameterSourceKind = Literal[
    "user",
    "literature",
    "a-priori",
    "a-posteriori",
]


class ResolvedLESFilter(StrictModule, NonTrainableState):
    """Complete typed semantics of the filter that defines resolved LES fields.

    The identity records the filter family, three-dimensional coordinate frame,
    topology and boundary class, physical scale interpretation, derivative
    commutation treatment, and repeated-filter behavior. Its namespaced digest
    cannot be supplied by, or alias, a dealiasing plan.
    """

    name: str = eqx.field(static=True)
    family: _LESFilterFamily = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    axis_names: tuple[str, str, str] = eqx.field(static=True)
    topology: _LESTopology = eqx.field(static=True)
    boundary_class: _LESBoundaryClass = eqx.field(static=True)
    scale_rule: _LESScaleRule = eqx.field(static=True)
    commutation_status: _LESCommutationStatus = eqx.field(static=True)
    repeated_filter_semantics: _LESRepeatedFilterSemantics = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        family: _LESFilterFamily,
        axis_names: tuple[str, str, str],
        topology: _LESTopology,
        boundary_class: _LESBoundaryClass,
        scale_rule: _LESScaleRule,
        commutation_status: _LESCommutationStatus,
        repeated_filter_semantics: _LESRepeatedFilterSemantics,
    ):
        if not isinstance(name, str):
            raise TypeError("Resolved LES filter name must be a string.")
        normalized = name.strip()
        if not normalized:
            raise ValueError("Resolved LES filter name must be non-empty.")
        if family not in (
            "sharp-fourier-projection",
            "implicit-grid-volume",
            "explicit-filter",
        ):
            raise ValueError("Unsupported resolved LES filter family.")
        if (
            not isinstance(axis_names, tuple)
            or len(axis_names) != 3
            or any(not isinstance(axis, str) or not axis.strip() for axis in axis_names)
        ):
            raise TypeError("Resolved LES axis_names must be three non-empty strings.")
        axes = tuple(axis.strip() for axis in axis_names)
        if len(set(axes)) != 3:
            raise ValueError("Resolved LES axis names must be unique.")
        if topology not in ("tensor-product", "unstructured"):
            raise ValueError("Unsupported resolved LES topology.")
        if boundary_class not in ("periodic", "wall-bounded", "open", "mixed"):
            raise ValueError("Unsupported resolved LES boundary class.")
        if scale_rule not in (
            "cutoff-equivalent",
            "volume-equivalent",
            "kernel-equivalent",
        ):
            raise ValueError("Unsupported resolved LES scale rule.")
        if commutation_status not in ("commuting", "modeled", "unmodeled"):
            raise ValueError("Unsupported LES filter commutation status.")
        if repeated_filter_semantics not in ("idempotent", "composed", "unmodeled"):
            raise ValueError("Unsupported repeated-filter semantics.")
        expected_scale_rule = {
            "sharp-fourier-projection": "cutoff-equivalent",
            "implicit-grid-volume": "volume-equivalent",
            "explicit-filter": "kernel-equivalent",
        }[family]
        if scale_rule != expected_scale_rule:
            raise ValueError("LES filter family and scale rule are inconsistent.")
        if family == "sharp-fourier-projection" and (
            topology != "tensor-product"
            or boundary_class != "periodic"
            or commutation_status != "commuting"
            or repeated_filter_semantics != "idempotent"
        ):
            raise ValueError(
                "Sharp Fourier projection requires periodic tensor-product, "
                "commuting, idempotent semantics."
            )
        self.name = normalized
        self.family = family
        self.dimension = 3
        self.axis_names = axes
        self.topology = topology
        self.boundary_class = boundary_class
        self.scale_rule = scale_rule
        self.commutation_status = commutation_status
        self.repeated_filter_semantics = repeated_filter_semantics
        self.filter_id = canonical_fingerprint(
            {
                "kind": "resolved-les-filter",
                "name": normalized,
                "family": family,
                "dimension": 3,
                "axis_names": list(axes),
                "topology": topology,
                "boundary_class": boundary_class,
                "scale_rule": scale_rule,
                "commutation_status": commutation_status,
                "repeated_filter_semantics": repeated_filter_semantics,
            }
        )


class LESFilterScale(StrictModule):
    """Physical directional LES filter widths with a final ``(x, y, z)`` axis.

    Directional widths are retained instead of being collapsed to one scalar so
    anisotropic models can apply each width to its matching derivative direction.
    A single ``(3,)`` vector remains unexpanded and broadcasts over every leading
    batch or spatial axis. Prepared structured backends should retain their three
    axis-width vectors as factored metadata and construct a broadcast local scale
    only inside evaluation; this local value need not materialize a full stored
    tensor-grid field.
    """

    directional_widths: Array

    def __init__(self, directional_widths: ArrayLike, /):
        widths = jnp.asarray(directional_widths)
        if not jnp.issubdtype(widths.dtype, jnp.inexact):
            widths = widths.astype(jnp.result_type(widths, float))
        if widths.ndim < 1 or widths.shape[-1] != 3:
            raise ValueError("LES directional widths must have trailing dimension 3.")
        if not isinstance(widths, jax.core.Tracer):
            concrete = np.asarray(widths)
            if np.any(~np.isfinite(concrete)) or np.any(concrete <= 0.0):
                raise ValueError("LES directional widths must be finite and positive.")
        self.directional_widths = widths

    @property
    def equivalent_width(self) -> Array:
        """Return the volume-equivalent filter width without discarding factors."""
        widths = self.directional_widths
        return jnp.cbrt(widths[..., 0] * widths[..., 1] * widths[..., 2])


class LESParameterProvenance(StrictModule, NonTrainableState):
    """Bind LES parameters to operating semantics and auditable source evidence."""

    resolved_filter: ResolvedLESFilter
    discretization_id: str = eqx.field(static=True)
    regime: str = eqx.field(static=True)
    source_kind: _LESParameterSourceKind = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        resolved_filter: ResolvedLESFilter,
        discretization_id: str,
        regime: str,
        /,
        *,
        source_kind: _LESParameterSourceKind,
        evidence_ids: tuple[str, ...],
    ):
        if not isinstance(resolved_filter, ResolvedLESFilter):
            raise TypeError("resolved_filter must be a ResolvedLESFilter.")
        if not isinstance(discretization_id, str) or not isinstance(regime, str):
            raise TypeError("LES discretization identity and regime must be strings.")
        discretization = discretization_id.strip()
        regime_ = regime.strip()
        if not discretization or not regime_:
            raise ValueError("LES discretization identity and regime must be non-empty.")
        if source_kind not in ("user", "literature", "a-priori", "a-posteriori"):
            raise ValueError("Unsupported LES parameter source kind.")
        if not isinstance(evidence_ids, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in evidence_ids
        ):
            raise TypeError("LES evidence_ids must be a tuple of non-empty strings.")
        evidence = tuple(sorted(value.strip() for value in evidence_ids))
        if len(set(evidence)) != len(evidence):
            raise ValueError("LES evidence_ids must be unique.")
        if source_kind != "user" and not evidence:
            raise ValueError("Non-user LES parameters require source evidence.")
        self.resolved_filter = resolved_filter
        self.discretization_id = discretization
        self.regime = regime_
        self.source_kind = source_kind
        self.evidence_ids = evidence
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "les-parameter-provenance",
                "filter": resolved_filter.filter_id,
                "discretization": discretization,
                "regime": regime_,
                "source_kind": source_kind,
                "evidence_ids": list(evidence),
            }
        )


class AlgebraicLESInputs(StrictModule):
    """Resolved velocity gradient and physical filter scale for a LES evaluation."""

    velocity_gradient: Array
    filter_scale: LESFilterScale

    def __init__(
        self,
        velocity_gradient: ArrayLike,
        filter_scale: LESFilterScale,
        /,
    ):
        if not isinstance(filter_scale, LESFilterScale):
            raise TypeError("filter_scale must be a LESFilterScale.")
        gradient = jnp.asarray(velocity_gradient)
        if not jnp.issubdtype(gradient.dtype, jnp.inexact):
            gradient = gradient.astype(jnp.result_type(gradient, float))
        if gradient.ndim < 2 or gradient.shape[-2:] != (3, 3):
            raise ValueError("LES velocity gradient must have trailing shape (3, 3).")
        self.velocity_gradient = gradient
        self.filter_scale = filter_scale


class AlgebraicLESResult(StrictModule):
    """Specific eddy viscosity, deviatoric stress, and positive energy transfer."""

    kinematic_viscosity: Array
    specific_deviatoric_stress: Array
    energy_transfer: Array


class AbstractAlgebraicLESModel(StrictModule):
    """Differentiable algebraic LES model with an unfrozen scalar coefficient.

    ``model_id`` identifies only the mathematical formula. Parameter values and
    their operating provenance acquire runtime identity only through
    :class:`PreparedAlgebraicLESModel`.
    """

    coefficient: AbstractAttribute[Array]
    formula: AbstractAttribute[_LESFormula]
    model_id: AbstractAttribute[str]

    def evaluate(self, inputs: AlgebraicLESInputs, /) -> AlgebraicLESResult:
        """Evaluate viscosity, specific deviatoric stress, and energy transfer."""
        return _evaluate_formula(self.formula, self.coefficient, inputs)

    def prepare(self, provenance: LESParameterProvenance, /) -> PreparedAlgebraicLESModel:
        """Freeze the coefficient and bind it to filter/discretization provenance."""
        return PreparedAlgebraicLESModel(self, provenance)


class PreparedAlgebraicLESModel(StrictModule, NonTrainableState):
    """Frozen algebraic LES formula and its sole complete runtime binding identity."""

    coefficient: float = eqx.field(static=True)
    formula: _LESFormula = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    provenance: LESParameterProvenance
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractAlgebraicLESModel,
        provenance: LESParameterProvenance,
        /,
    ):
        if not isinstance(model, AbstractAlgebraicLESModel):
            raise TypeError("model must be an AbstractAlgebraicLESModel.")
        if not isinstance(provenance, LESParameterProvenance):
            raise TypeError("provenance must be LESParameterProvenance.")
        coefficient_array = model.coefficient
        if isinstance(coefficient_array, jax.core.Tracer):
            raise TypeError("Prepared LES coefficients must have concrete values.")
        coefficient = float(np.asarray(coefficient_array))
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("Prepared LES coefficient must be finite and nonnegative.")
        self.coefficient = coefficient
        self.formula = model.formula
        self.model_id = model.model_id
        self.provenance = provenance
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-algebraic-les-model",
                "model": model.model_id,
                "formula": model.formula,
                "coefficient": coefficient,
                "provenance": provenance.provenance_id,
                "filter": provenance.resolved_filter.filter_id,
                "discretization": provenance.discretization_id,
                "regime": provenance.regime,
            }
        )

    def evaluate(self, inputs: AlgebraicLESInputs, /) -> AlgebraicLESResult:
        """Evaluate the frozen formula without introducing a second binding ID."""
        return _evaluate_formula(self.formula, self.coefficient, inputs)


class SmagorinskyLESPlan(AbstractAlgebraicLESModel):
    """Smagorinsky model using the full strain magnitude and deviatoric stress."""

    coefficient: Array
    formula: _LESFormula = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, coefficient: ArrayLike, /):
        self.coefficient = _validated_coefficient(coefficient, "Smagorinsky")
        self.formula = "smagorinsky"
        self.model_id = _formula_id(self.formula)


class WALELESPlan(AbstractAlgebraicLESModel):
    """Wall-adapting local eddy-viscosity model with its exact zero branch."""

    coefficient: Array
    formula: _LESFormula = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, coefficient: ArrayLike, /):
        self.coefficient = _validated_coefficient(coefficient, "WALE")
        self.formula = "wale"
        self.model_id = _formula_id(self.formula)


class VremanLESPlan(AbstractAlgebraicLESModel):
    """Vreman model with directional widths on the gradient derivative axis."""

    coefficient: Array
    formula: _LESFormula = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, coefficient: ArrayLike, /):
        self.coefficient = _validated_coefficient(coefficient, "Vreman")
        self.formula = "vreman"
        self.model_id = _formula_id(self.formula)


class AMDLESPlan(AbstractAlgebraicLESModel):
    """Anisotropic minimum-dissipation model with branchwise zero response."""

    coefficient: Array
    formula: _LESFormula = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, coefficient: ArrayLike, /):
        self.coefficient = _validated_coefficient(coefficient, "AMD")
        self.formula = "amd"
        self.model_id = _formula_id(self.formula)


def _formula_id(formula: _LESFormula, /) -> str:
    return canonical_fingerprint({"kind": "algebraic-les-model", "formula": formula})


def _validated_coefficient(value: ArrayLike, name: str, /) -> Array:
    coefficient = jnp.asarray(value)
    if not jnp.issubdtype(coefficient.dtype, jnp.inexact):
        coefficient = coefficient.astype(jnp.result_type(coefficient, float))
    if coefficient.shape != ():
        raise ValueError(f"{name} LES coefficient must be scalar.")
    if not isinstance(coefficient, jax.core.Tracer):
        concrete = np.asarray(coefficient)
        if not np.isfinite(concrete) or concrete < 0.0:
            raise ValueError(f"{name} LES coefficient must be finite and nonnegative.")
    return coefficient


def _positive_square_root(value: Array, /) -> Array:
    active = value > 0.0
    safe_value = jnp.where(active, value, jnp.ones_like(value))
    return jnp.where(active, jnp.sqrt(safe_value), jnp.zeros_like(value))


def _positive_ratio(numerator: Array, denominator: Array, /) -> Array:
    active = (numerator > 0.0) & (denominator > 0.0)
    safe_numerator = jnp.where(active, numerator, jnp.ones_like(numerator))
    safe_denominator = jnp.where(active, denominator, jnp.ones_like(denominator))
    return jnp.where(
        active,
        safe_numerator / safe_denominator,
        jnp.zeros_like(numerator),
    )


def _strain_tensors(gradient: Array, /) -> tuple[Array, Array]:
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=gradient.dtype)
    deviatoric = strain - trace[..., None, None] * identity / 3.0
    return strain, deviatoric


def _directional_metric(gradient: Array, widths: Array, /) -> Array:
    scaled_gradient = gradient * widths[..., None, :]
    return ein.contract(
        "...ik,...jk->...ij", scaled_gradient, scaled_gradient, backend="jax"
    )


def _smagorinsky_viscosity(
    coefficient: ArrayLike,
    inputs: AlgebraicLESInputs,
    strain: Array,
    /,
) -> Array:
    strain_squared = ein.contract("...ij,...ij->...", strain, strain, backend="jax")
    magnitude = _positive_square_root(2.0 * strain_squared)
    scale = jnp.asarray(coefficient) * inputs.filter_scale.equivalent_width
    return scale * scale * magnitude


def _wale_viscosity(
    coefficient: ArrayLike,
    inputs: AlgebraicLESInputs,
    strain: Array,
    /,
) -> Array:
    gradient = inputs.velocity_gradient
    squared = ein.contract("...ik,...kj->...ij", gradient, gradient, backend="jax")
    symmetric_squared = 0.5 * (squared + jnp.swapaxes(squared, -1, -2))
    trace = jnp.trace(symmetric_squared, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=gradient.dtype)
    deviatoric_squared = symmetric_squared - trace[..., None, None] * identity / 3.0
    strain_squared = ein.contract("...ij,...ij->...", strain, strain, backend="jax")
    squared_invariant = ein.contract(
        "...ij,...ij->...", deviatoric_squared, deviatoric_squared, backend="jax"
    )
    strain_root = _positive_square_root(strain_squared)
    squared_root = _positive_square_root(squared_invariant)
    squared_fourth_root = _positive_square_root(squared_root)
    numerator = squared_invariant * squared_root
    denominator = (
        strain_squared * strain_squared * strain_root
        + squared_invariant * squared_fourth_root
    )
    ratio = _positive_ratio(numerator, denominator)
    scale = jnp.asarray(coefficient) * inputs.filter_scale.equivalent_width
    return scale * scale * ratio


def _vreman_viscosity(coefficient: ArrayLike, inputs: AlgebraicLESInputs, /) -> Array:
    gradient = inputs.velocity_gradient
    beta = _directional_metric(gradient, inputs.filter_scale.directional_widths)
    trace = jnp.trace(beta, axis1=-2, axis2=-1)
    beta_squared = ein.contract("...ij,...ij->...", beta, beta, backend="jax")
    invariant = 0.5 * (trace * trace - beta_squared)
    gradient_squared = ein.contract("...ij,...ij->...", gradient, gradient, backend="jax")
    return jnp.asarray(coefficient) * _positive_square_root(
        _positive_ratio(invariant, gradient_squared)
    )


def _amd_viscosity(
    coefficient: ArrayLike,
    inputs: AlgebraicLESInputs,
    deviatoric_strain: Array,
    /,
) -> Array:
    gradient = inputs.velocity_gradient
    beta = _directional_metric(gradient, inputs.filter_scale.directional_widths)
    production = -ein.contract("...ij,...ij->...", beta, deviatoric_strain, backend="jax")
    gradient_squared = ein.contract("...ij,...ij->...", gradient, gradient, backend="jax")
    return jnp.asarray(coefficient) * _positive_ratio(production, gradient_squared)


def _evaluate_formula(
    formula: _LESFormula,
    coefficient: ArrayLike,
    inputs: AlgebraicLESInputs,
    /,
) -> AlgebraicLESResult:
    if not isinstance(inputs, AlgebraicLESInputs):
        raise TypeError("inputs must be AlgebraicLESInputs.")
    strain, deviatoric_strain = _strain_tensors(inputs.velocity_gradient)
    if formula == "smagorinsky":
        viscosity = _smagorinsky_viscosity(coefficient, inputs, strain)
    elif formula == "wale":
        viscosity = _wale_viscosity(coefficient, inputs, strain)
    elif formula == "vreman":
        viscosity = _vreman_viscosity(coefficient, inputs)
    elif formula == "amd":
        viscosity = _amd_viscosity(coefficient, inputs, deviatoric_strain)
    else:
        raise ValueError(f"Unsupported algebraic LES formula {formula!r}.")
    stress = -2.0 * viscosity[..., None, None] * deviatoric_strain
    transfer = -ein.contract("...ij,...ij->...", stress, strain, backend="jax")
    return AlgebraicLESResult(viscosity, stress, transfer)


__all__ = [
    "AMDLESPlan",
    "AbstractAlgebraicLESModel",
    "AlgebraicLESInputs",
    "AlgebraicLESResult",
    "LESFilterScale",
    "LESParameterProvenance",
    "PreparedAlgebraicLESModel",
    "ResolvedLESFilter",
    "SmagorinskyLESPlan",
    "VremanLESPlan",
    "WALELESPlan",
]

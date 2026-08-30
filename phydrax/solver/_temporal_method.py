#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._diffrax_state_packing import (
    AlgebraStatePackingEvidence,
    ComplexStatePackingEvidence,
)


TemporalEquationForm: TypeAlias = Literal[
    "explicit-ode",
    "additive-ode",
    "implicit-residual",
    "split-residual",
    "sde",
    "geometric-ode",
    "geometric-sde",
    "second-order",
    "partitioned",
]
TemporalMethodClass: TypeAlias = Literal[
    "erk",
    "ssp-rk",
    "dirk",
    "ark",
    "bdf",
    "theta",
    "generalized-alpha",
    "rosenbrock-w",
    "irk",
    "multirate-rk",
    "partitioned",
    "geometric",
    "stochastic-rk",
    "exponential",
    "probabilistic",
    "unknown",
]
NoiseRequirement: TypeAlias = Literal["none", "additive", "commutative", "general"]


class TemporalMethodCapabilities(StrictModule, NonTrainableState):
    """Immutable, inspectable numerical capabilities of one temporal method."""

    equation_forms: tuple[TemporalEquationForm, ...] = eqx.field(static=True)
    method_class: TemporalMethodClass = eqx.field(static=True)
    order: int | None = eqx.field(static=True)
    embedded_order: int | None = eqx.field(static=True)
    stage_order: int | None = eqx.field(static=True)
    dense_order: int | None = eqx.field(static=True)
    strong_orders: tuple[tuple[NoiseRequirement, float], ...] = eqx.field(static=True)
    adaptive: bool = eqx.field(static=True)
    history_depth: int = eqx.field(static=True)
    stage_abscissae: tuple[float, ...] = eqx.field(static=True)
    causal_stage_extent: float = eqx.field(static=True)
    a_stable: bool = eqx.field(static=True)
    l_stable: bool = eqx.field(static=True)
    stiffly_accurate: bool = eqx.field(static=True)
    symplectic: bool = eqx.field(static=True)
    reversible: bool = eqx.field(static=True)
    ssp_coefficient: float | None = eqx.field(static=True)
    noise_requirement: NoiseRequirement = eqx.field(static=True)
    levy_area: str | None = eqx.field(static=True)
    verified: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        equation_forms: tuple[TemporalEquationForm, ...],
        method_class: TemporalMethodClass,
        order: int | None,
        embedded_order: int | None = None,
        stage_order: int | None = None,
        dense_order: int | None = None,
        strong_orders: tuple[tuple[NoiseRequirement, float], ...] = (),
        adaptive: bool,
        history_depth: int = 1,
        stage_abscissae: tuple[float, ...] = (),
        causal_stage_extent: float = 1.0,
        a_stable: bool = False,
        l_stable: bool = False,
        stiffly_accurate: bool = False,
        symplectic: bool = False,
        reversible: bool = False,
        ssp_coefficient: float | None = None,
        noise_requirement: NoiseRequirement = "none",
        levy_area: str | None = None,
        verified: bool = True,
        method_id: str,
    ):
        forms = tuple(equation_forms)
        if not forms or len(set(forms)) != len(forms):
            raise ValueError("Temporal equation forms must be non-empty and unique.")
        for value, owner in (
            (order, "order"),
            (embedded_order, "embedded_order"),
            (stage_order, "stage_order"),
            (dense_order, "dense_order"),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"Temporal {owner} must be positive or None.")
        depth = int(history_depth)
        if depth < 1:
            raise ValueError("Temporal history_depth must be positive.")
        abscissae = tuple(float(value) for value in stage_abscissae)
        if any(not isfinite(value) for value in abscissae):
            raise ValueError("Temporal stage abscissae must be finite.")
        extent = float(causal_stage_extent)
        if not isfinite(extent) or extent <= 0.0:
            raise ValueError("causal_stage_extent must be finite and positive.")
        strong = tuple((kind, float(value)) for kind, value in strong_orders)
        if any(
            kind not in ("none", "additive", "commutative", "general")
            or not isfinite(value)
            or value <= 0.0
            for kind, value in strong
        ):
            raise ValueError("Strong-order records must be finite and positive.")
        ssp = None if ssp_coefficient is None else float(ssp_coefficient)
        if ssp is not None and (not isfinite(ssp) or ssp <= 0.0):
            raise ValueError("ssp_coefficient must be finite and positive or None.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.equation_forms = forms
        self.method_class = method_class
        self.order = None if order is None else int(order)
        self.embedded_order = None if embedded_order is None else int(embedded_order)
        self.stage_order = None if stage_order is None else int(stage_order)
        self.dense_order = None if dense_order is None else int(dense_order)
        self.strong_orders = strong
        self.adaptive = bool(adaptive)
        self.history_depth = depth
        self.stage_abscissae = abscissae
        self.causal_stage_extent = extent
        self.a_stable = bool(a_stable)
        self.l_stable = bool(l_stable)
        self.stiffly_accurate = bool(stiffly_accurate)
        self.symplectic = bool(symplectic)
        self.reversible = bool(reversible)
        self.ssp_coefficient = ssp
        self.noise_requirement = noise_requirement
        self.levy_area = None if levy_area is None else str(levy_area)
        self.verified = bool(verified)
        self.method_id = identifier

    def strong_order(self, structure: NoiseRequirement, /) -> float | None:
        for kind, value in self.strong_orders:
            if kind == structure:
                return value
        return None


class TemporalSolveEvidence(StrictModule, NonTrainableState):
    """Static method/backend configuration retained by a temporal solution."""

    capabilities: TemporalMethodCapabilities
    equation_form: TemporalEquationForm = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)
    adjoint_id: str = eqx.field(static=True)
    event_id: str | None = eqx.field(static=True)
    adaptive: bool = eqx.field(static=True)
    dense: bool = eqx.field(static=True)
    maximum_steps: int | None = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)
    state_packing: ComplexStatePackingEvidence | AlgebraStatePackingEvidence | None

    def __init__(
        self,
        capabilities: TemporalMethodCapabilities,
        /,
        *,
        equation_form: TemporalEquationForm,
        backend_id: str,
        configuration_id: str,
        controller_id: str,
        adjoint_id: str,
        event_id: str | None,
        adaptive: bool,
        dense: bool,
        maximum_steps: int | None,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
        state_packing: ComplexStatePackingEvidence
        | AlgebraStatePackingEvidence
        | None = None,
    ):
        if not isinstance(capabilities, TemporalMethodCapabilities):
            raise TypeError("capabilities must be TemporalMethodCapabilities.")
        values = tuple(
            str(value)
            for value in (
                backend_id,
                configuration_id,
                controller_id,
                adjoint_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Temporal evidence IDs must be non-empty.")
        limit = None if maximum_steps is None else int(maximum_steps)
        if limit is not None and limit < 1:
            raise ValueError("maximum_steps must be positive or None.")
        if precision_evidence is not None and not isinstance(
            precision_evidence,
            PrecisionEvidenceEnvelope,
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        if state_packing is not None and not isinstance(
            state_packing,
            (ComplexStatePackingEvidence, AlgebraStatePackingEvidence),
        ):
            raise TypeError(
                "state_packing must be complex/algebra packing evidence or None."
            )
        self.capabilities = capabilities
        self.equation_form = equation_form
        self.backend_id, self.configuration_id, self.controller_id, self.adjoint_id = (
            values
        )
        self.event_id = None if event_id is None else str(event_id)
        self.adaptive = bool(adaptive)
        self.dense = bool(dense)
        self.maximum_steps = limit
        self.precision_evidence = precision_evidence
        self.state_packing = state_packing


def qualified_type_name(value: Any, /) -> str:
    """Return one stable class-level identifier without inspecting runtime state."""
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def configuration_id(value: Any, /, *, prefix: str) -> str:
    """Fingerprint one static numerical configuration.

    Built-in Equinox/Diffrax modules have deterministic representations. Callers using
    custom objects with process-local representations must provide an explicit outer
    configuration ID instead of treating this digest as portable provenance.
    """
    return f"{prefix}:{canonical_fingerprint({'type': qualified_type_name(value), 'representation': repr(value)})}"


def _capabilities(
    name: str,
    method_class: TemporalMethodClass,
    forms: tuple[TemporalEquationForm, ...],
    order: int | None,
    *,
    embedded: int | None = None,
    dense: int | None = None,
    adaptive: bool,
    a_stable: bool = False,
    l_stable: bool = False,
    stiffly_accurate: bool = False,
    symplectic: bool = False,
    reversible: bool = False,
    noise_requirement: NoiseRequirement = "none",
    strong_orders: tuple[tuple[NoiseRequirement, float], ...] = (),
    levy_area: str | None = None,
) -> TemporalMethodCapabilities:
    return TemporalMethodCapabilities(
        equation_forms=forms,
        method_class=method_class,
        order=order,
        embedded_order=embedded,
        dense_order=dense,
        strong_orders=strong_orders,
        adaptive=adaptive,
        a_stable=a_stable,
        l_stable=l_stable,
        stiffly_accurate=stiffly_accurate,
        symplectic=symplectic,
        reversible=reversible,
        noise_requirement=noise_requirement,
        levy_area=levy_area,
        method_id=f"temporal:diffrax:{name}",
    )


def diffrax_method_capabilities(solver: Any, /) -> TemporalMethodCapabilities:
    """Return verified capabilities for a built-in Diffrax solver."""
    explicit = ("explicit-ode",)
    explicit_sde = ("explicit-ode", "sde")
    additive = ("additive-ode",)
    if isinstance(solver, dfx.Euler):
        return _capabilities(
            "Euler",
            "erk",
            explicit_sde,
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("general", 0.5),),
            noise_requirement="general",
        )
    if isinstance(solver, dfx.Heun):
        return _capabilities(
            "Heun",
            "erk",
            explicit_sde,
            2,
            embedded=1,
            dense=2,
            adaptive=True,
            strong_orders=(("general", 0.5),),
            noise_requirement="general",
        )
    if isinstance(solver, dfx.Midpoint):
        return _capabilities(
            "Midpoint",
            "erk",
            explicit_sde,
            2,
            embedded=1,
            dense=2,
            adaptive=True,
            strong_orders=(("general", 0.5),),
            noise_requirement="general",
        )
    if isinstance(solver, dfx.Ralston):
        return _capabilities(
            "Ralston",
            "erk",
            explicit_sde,
            2,
            embedded=1,
            dense=2,
            adaptive=True,
            strong_orders=(("general", 0.5),),
            noise_requirement="general",
        )
    if isinstance(solver, dfx.Bosh3):
        return _capabilities(
            "Bosh3", "erk", explicit, 3, embedded=2, dense=3, adaptive=True
        )
    if isinstance(solver, dfx.Tsit5):
        return _capabilities(
            "Tsit5", "erk", explicit, 5, embedded=4, dense=5, adaptive=True
        )
    if isinstance(solver, dfx.Dopri5):
        return _capabilities(
            "Dopri5", "erk", explicit, 5, embedded=4, dense=5, adaptive=True
        )
    if isinstance(solver, dfx.Dopri8):
        return _capabilities(
            "Dopri8", "erk", explicit, 8, embedded=7, dense=8, adaptive=True
        )
    if isinstance(solver, dfx.ImplicitEuler):
        return _capabilities(
            "ImplicitEuler",
            "dirk",
            explicit,
            1,
            embedded=2,
            dense=1,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.Kvaerno3):
        return _capabilities(
            "Kvaerno3",
            "dirk",
            explicit,
            3,
            embedded=2,
            dense=3,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.Kvaerno4):
        return _capabilities(
            "Kvaerno4",
            "dirk",
            explicit,
            4,
            embedded=3,
            dense=3,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.Kvaerno5):
        return _capabilities(
            "Kvaerno5",
            "dirk",
            explicit,
            5,
            embedded=4,
            dense=3,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.KenCarp3):
        return _capabilities(
            "KenCarp3",
            "ark",
            additive,
            3,
            embedded=2,
            dense=2,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.KenCarp4):
        return _capabilities(
            "KenCarp4",
            "ark",
            additive,
            4,
            embedded=3,
            dense=3,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.KenCarp5):
        return _capabilities(
            "KenCarp5",
            "ark",
            additive,
            5,
            embedded=4,
            dense=3,
            adaptive=True,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
        )
    if isinstance(solver, dfx.Sil3):
        return _capabilities(
            "Sil3", "ark", additive, 2, embedded=2, dense=2, adaptive=True, a_stable=True
        )
    if isinstance(solver, dfx.SemiImplicitEuler):
        return _capabilities(
            "SemiImplicitEuler",
            "partitioned",
            ("partitioned",),
            1,
            dense=1,
            adaptive=False,
            symplectic=True,
        )
    if isinstance(solver, dfx.LeapfrogMidpoint):
        return _capabilities(
            "LeapfrogMidpoint",
            "partitioned",
            explicit,
            2,
            dense=1,
            adaptive=False,
            reversible=True,
        )
    if isinstance(solver, dfx.ReversibleHeun):
        return _capabilities(
            "ReversibleHeun",
            "erk",
            explicit_sde,
            2,
            embedded=1,
            dense=1,
            adaptive=True,
            reversible=True,
            strong_orders=(("general", 0.5),),
            noise_requirement="general",
        )
    if isinstance(solver, dfx.EulerHeun):
        return _capabilities(
            "EulerHeun",
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("general", 0.5),),
            noise_requirement="general",
        )
    if isinstance(solver, (dfx.ItoMilstein, dfx.StratonovichMilstein)):
        return _capabilities(
            type(solver).__name__,
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("additive", 1.0), ("commutative", 1.0)),
            noise_requirement="commutative",
        )
    if isinstance(solver, dfx.SEA):
        return _capabilities(
            "SEA",
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("additive", 1.0),),
            noise_requirement="additive",
            levy_area="space_time",
        )
    if isinstance(solver, (dfx.SRA1, dfx.ShARK)):
        return _capabilities(
            type(solver).__name__,
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("additive", 1.5),),
            noise_requirement="additive",
            levy_area="space_time",
        )
    if isinstance(solver, dfx.SlowRK):
        return _capabilities(
            "SlowRK",
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("additive", 1.5), ("commutative", 1.5), ("general", 0.5)),
            noise_requirement="general",
            levy_area="space_time",
        )
    if isinstance(solver, dfx.GeneralShARK):
        return _capabilities(
            "GeneralShARK",
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=False,
            strong_orders=(("additive", 1.5), ("commutative", 1.0), ("general", 0.5)),
            noise_requirement="general",
            levy_area="space_time",
        )
    if isinstance(solver, dfx.SPaRK):
        return _capabilities(
            "SPaRK",
            "stochastic-rk",
            ("sde",),
            1,
            dense=1,
            adaptive=True,
            strong_orders=(("additive", 1.5), ("commutative", 1.0), ("general", 0.5)),
            noise_requirement="general",
            levy_area="space_time",
        )
    return TemporalMethodCapabilities(
        equation_forms=("explicit-ode",),
        method_class="unknown",
        order=None,
        adaptive=isinstance(solver, dfx.AbstractAdaptiveSolver),
        method_id=f"temporal:diffrax:unknown:{qualified_type_name(solver)}",
        verified=False,
    )


__all__ = [
    "NoiseRequirement",
    "TemporalEquationForm",
    "TemporalMethodCapabilities",
    "TemporalMethodClass",
    "TemporalSolveEvidence",
    "configuration_id",
    "diffrax_method_capabilities",
    "qualified_type_name",
]

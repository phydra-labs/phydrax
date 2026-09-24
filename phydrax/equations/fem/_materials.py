#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    DOMAIN_REASON_SHIFT,
    guard_derivative_validity,
    reason_bits_where,
)
from ..._differentiation import ComponentAuthority
from ..._fingerprint import canonical_fingerprint
from ..._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    ComponentBinding,
    ModelPorts,
    PortMapping,
)
from ..._nonlinear_precision import NonlinearPrecisionPolicy
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._finite_element_material import _bind_learned_law, ConstitutiveResponse


class LocalImplicitDiagnostics(StrictModule):
    converged: Array
    iterations: Array
    residual_norm: Array
    finite: Array


class FiniteElementAuxiliaryEvaluation(StrictModule):
    """Material/contact/history candidate state returned with one residual."""

    trial_state: object
    successful: Array
    admissible: Array
    retry_requested: Array
    suggested_step: Array
    diagnostics: object

    def __init__(
        self,
        trial_state: object = None,
        /,
        *,
        successful: ArrayLike = True,
        admissible: ArrayLike = True,
        retry_requested: ArrayLike = False,
        suggested_step: ArrayLike = 0.0,
        diagnostics: object = None,
    ):
        successful_ = jnp.asarray(successful, dtype=jnp.bool_)
        admissible_ = jnp.asarray(admissible, dtype=jnp.bool_)
        retry_ = jnp.asarray(retry_requested, dtype=jnp.bool_)
        suggested = jnp.asarray(suggested_step)
        if any(
            value.shape != () for value in (successful_, admissible_, retry_, suggested)
        ):
            raise ValueError("Auxiliary decision values must be scalars.")
        self.trial_state = trial_state
        self.successful = successful_
        self.admissible = admissible_
        self.retry_requested = retry_
        self.suggested_step = suggested
        self.diagnostics = diagnostics

    @property
    def valid(self) -> Array:
        return self.successful & self.admissible


class AbstractLocalImplicitMaterial(AbstractComponentSlot):
    """Bounded local constitutive root slot with implicit-function derivatives.

    The slot confers `MODEL` authority. An implementation declares
    `state_shape`, `max_steps`, `tolerance`, and `model_id` and supplies the
    local residual and the response at a root. `solve` runs `max_steps` Newton
    iterations and differentiates the root by the implicit function theorem
    (`jax.lax.custom_root`), never through the iterations; `evaluate` marks the
    response invalid unless the root converged to `tolerance` and is finite.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.MODEL
    slot_semantic_id: ClassVar[str] = "phydrax.equations.local-implicit-material"

    state_shape: eqx.AbstractVar[tuple[int, ...]]
    max_steps: eqx.AbstractVar[int]
    tolerance: eqx.AbstractVar[float]
    model_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def local_residual(self, state: Array, args: object, /) -> Array:
        """Local residual whose root is the constitutive state."""
        raise NotImplementedError

    @abc.abstractmethod
    def local_response(self, state: Array, args: object, /) -> ConstitutiveResponse:
        """Constitutive response at a local state."""
        raise NotImplementedError

    def _newton_solve(self, function: Callable, initial: Array, args: object) -> Array:
        def body(_, value):
            residual = function(value, args)
            jacobian = jax.jacfwd(lambda candidate: function(candidate, args))(value)
            flat_residual = residual.reshape((-1,))
            flat_jacobian = jacobian.reshape((flat_residual.size, flat_residual.size))
            update = jnp.linalg.solve(flat_jacobian, -flat_residual).reshape(value.shape)
            candidate = value + update
            norm = jnp.linalg.norm(flat_residual)
            return jnp.where(norm <= self.tolerance, value, candidate)

        return jax.lax.fori_loop(0, self.max_steps, body, initial)

    def solve(self, initial_state: ArrayLike, args: object, /) -> Array:
        initial = jnp.asarray(initial_state)
        if initial.shape[-len(self.state_shape) :] != self.state_shape:
            raise ValueError("Initial local state has the wrong trailing shape.")
        function = self.local_residual

        def solve_fn(residual_fn, guess):
            return self._newton_solve(
                lambda state, parameters: residual_fn(state),
                guess,
                None,
            )

        def tangent_solve(linearize, right_hand_side):
            zero = jnp.zeros_like(right_hand_side)
            matrix = jax.jacfwd(linearize)(zero)
            flat_rhs = right_hand_side.reshape((-1,))
            flat_matrix = matrix.reshape((flat_rhs.size, flat_rhs.size))
            return jnp.linalg.solve(flat_matrix, flat_rhs).reshape(right_hand_side.shape)

        return jax.lax.custom_root(
            lambda state: function(state, args),
            initial,
            solve_fn,
            tangent_solve,
        )

    def solve_with_diagnostics(
        self,
        initial_state: ArrayLike,
        args: object,
        /,
    ) -> tuple[Array, LocalImplicitDiagnostics]:
        root = self.solve(initial_state, args)
        residual = jnp.asarray(self.local_residual(root, args))
        norm = jnp.linalg.norm(residual.reshape((-1,)))
        finite = jnp.all(jnp.isfinite(root)) & jnp.all(jnp.isfinite(residual))
        return root, LocalImplicitDiagnostics(
            converged=finite & (norm <= self.tolerance),
            iterations=jnp.asarray(self.max_steps, dtype=jnp.int32),
            residual_norm=norm,
            finite=finite,
        )

    def evaluate(
        self,
        initial_state: ArrayLike,
        args: object,
        /,
    ) -> ConstitutiveResponse:
        root, diagnostics = self.solve_with_diagnostics(initial_state, args)
        result = self.local_response(root, args)
        if not isinstance(result, ConstitutiveResponse):
            raise TypeError("Local material response must return ConstitutiveResponse.")
        return ConstitutiveResponse(
            result.response,
            result.trial_state,
            consistent_tangent=result.consistent_tangent,
            energy=result.energy,
            dissipation=result.dissipation,
            valid=result.valid & diagnostics.converged & diagnostics.finite,
            header=result.header,
            diagnostic=result.diagnostic,
            diagnostics={
                **result.diagnostics,
                "converged": diagnostics.converged,
                "iterations": diagnostics.iterations,
                "residual_norm": diagnostics.residual_norm,
                "finite": diagnostics.finite,
            },
        )


def _root_configuration(
    state_shape: tuple[int, ...], max_steps: int, tolerance: float, model_id: str, /
) -> tuple[tuple[int, ...], int, float, str]:
    shape = tuple(state_shape)
    steps = int(max_steps)
    tolerance_ = float(tolerance)
    identifier = str(model_id)
    if (
        not shape
        or any(size <= 0 for size in shape)
        or steps <= 0
        or tolerance_ <= 0.0
        or not identifier
    ):
        raise ValueError("Local implicit material configuration is invalid.")
    return shape, steps, tolerance_, identifier


@final
class LocalImplicitMaterial(AbstractLocalImplicitMaterial, NonTrainableState):
    """Fixed analytic local root given by pure `residual` and `response` callables.

    `residual(state, args)` and `response(state, args)` receive the material
    coefficients through `args`; the material holds no numeric state of its own
    and is FIXED wherever it is held.
    """

    residual: Callable
    response: Callable
    state_shape: tuple[int, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable,
        response: Callable,
        /,
        *,
        state_shape: tuple[int, ...],
        max_steps: int = 25,
        tolerance: float = 1.0e-10,
        model_id: str,
    ):
        if not callable(residual) or not callable(response):
            raise TypeError("Local material residual and response must be callable.")
        shape, steps, tolerance_, identifier = _root_configuration(
            state_shape, max_steps, tolerance, model_id
        )
        self.residual = residual
        self.response = response
        self.state_shape = shape
        self.max_steps = steps
        self.tolerance = tolerance_
        self.model_id = canonical_fingerprint(
            {
                "kind": "local-implicit-material",
                "declared_id": identifier,
                "state_shape": list(shape),
                "max_steps": steps,
                "tolerance": tolerance_,
            }
        )

    def local_residual(self, state: Array, args: object, /) -> Array:
        return self.residual(state, args)

    def local_response(self, state: Array, args: object, /) -> ConstitutiveResponse:
        return self.response(state, args)


# Domain reason bits of a learned local root's admissibility header.
_ROOT_UNRESOLVED = 1 << DOMAIN_REASON_SHIFT
_RESPONSE_REJECTED = 1 << (DOMAIN_REASON_SHIFT + 1)


@final
class LearnedLocalImplicitMaterial(AbstractLocalImplicitMaterial):
    """Local constitutive root whose residual and response use a learned model child.

    `residual(model, state, args)` and `response(model, state, args)` are pure
    operations that receive the bound model explicitly. The model is bound to
    the local-implicit slot (`binding`, `MODEL` authority) and stays a dynamic
    child with its own array roles, so a `ParameterOwner` model trains through
    the root while a `FrozenModel` stays fixed; the root derivative with respect
    to the model parameters is the implicit-function derivative of the learned
    residual. Optional owner `ports` (with an explicit `port_mapping` for a
    model declaring ports) verify the model's ports, units included.

    Construction admits the model for implicit use: first input and parameter
    derivatives under `MODEL` authority, classical `C^1` value regularity (or a
    branch-margin certificate), deterministic randomness, and a declared
    precision contract whose error floor `tolerance` does not undercut
    (`NonlinearPrecisionPolicy.validate_tolerance` against the model's output
    dtype).

    The response carries an `AdmissibilityHeader` whose margin is
    `tolerance - residual_norm` at the evaluated state, with the `NONFINITE`
    reason and the domain bits `1 << DOMAIN_REASON_SHIFT` (unresolved local
    root) and `1 << (DOMAIN_REASON_SHIFT + 1)` (response rejected by the
    `response` operation). An ineligible response keeps its primal values but is
    invalid and carries NaN derivatives.
    """

    binding: ComponentBinding
    residual: Callable
    response: Callable
    state_shape: tuple[int, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        residual: Callable,
        response: Callable,
        /,
        *,
        state_shape: tuple[int, ...],
        model_id: str,
        max_steps: int = 25,
        tolerance: float = 1.0e-10,
        ports: ModelPorts | None = None,
        port_mapping: PortMapping | None = None,
    ):
        site = "LearnedLocalImplicitMaterial"
        if not callable(residual) or not callable(response):
            raise TypeError("Local material residual and response must be callable.")
        if ports is not None and not isinstance(ports, ModelPorts):
            raise TypeError(f"{site} ports must be ModelPorts or None.")
        shape, steps, tolerance_, identifier = _root_configuration(
            state_shape, max_steps, tolerance, model_id
        )
        binding, contract = _bind_learned_law(
            model, AbstractLocalImplicitMaterial, ports, port_mapping, site
        )
        NonlinearPrecisionPolicy(components=(contract,)).validate_tolerance(
            tolerance_, residual_dtype=contract.model_contract.precision.output_dtype
        )
        self.binding = binding
        self.residual = residual
        self.response = response
        self.state_shape = shape
        self.max_steps = steps
        self.tolerance = tolerance_
        self.model_id = canonical_fingerprint(
            {
                "kind": "learned-local-implicit-material",
                "declared_id": identifier,
                "slot": AbstractLocalImplicitMaterial.slot_semantic_id,
                "ports": None if ports is None else ports.ports_id,
                "state_shape": list(shape),
                "max_steps": steps,
                "tolerance": tolerance_,
            }
        )
        self.evidence_id = canonical_fingerprint(
            {"kind": "learned-local-root-admissibility", "model_id": self.model_id}
        )

    def local_residual(self, state: Array, args: object, /) -> Array:
        return jnp.asarray(self.residual(self.binding.model, state, args))

    def local_response(self, state: Array, args: object, /) -> ConstitutiveResponse:
        result = self.response(self.binding.model, state, args)
        if not isinstance(result, ConstitutiveResponse):
            raise TypeError("Local material response must return ConstitutiveResponse.")
        residual = self.local_residual(state, args)
        norm = jnp.linalg.norm(residual.reshape((-1,)))
        finite = (
            jnp.all(jnp.isfinite(state))
            & jnp.isfinite(norm)
            & jnp.all(jnp.isfinite(result.response))
            & jnp.all(jnp.isfinite(result.trial_state))
        )
        reasons = (
            reason_bits_where(finite, AdmissibilityReason.NONFINITE)
            | reason_bits_where(norm <= self.tolerance, _ROOT_UNRESOLVED)
            | reason_bits_where(jnp.all(result.valid), _RESPONSE_REJECTED)
        )
        header = AdmissibilityHeader(
            self.tolerance - norm, reasons, self.model_id, self.evidence_id
        )
        guarded = guard_derivative_validity(
            (result.response, result.trial_state, result.consistent_tangent),
            header.eligible,
            dependencies=(state, args),
        )
        response, trial_state, tangent = guarded
        return ConstitutiveResponse(
            response,
            trial_state,
            consistent_tangent=tangent,
            energy=result.energy,
            dissipation=result.dissipation,
            valid=result.valid & header.eligible,
            header=header,
            diagnostic=result.diagnostic,
            diagnostics=result.diagnostics,
        )


__all__ = [
    "AbstractLocalImplicitMaterial",
    "FiniteElementAuxiliaryEvaluation",
    "LearnedLocalImplicitMaterial",
    "LocalImplicitDiagnostics",
    "LocalImplicitMaterial",
]

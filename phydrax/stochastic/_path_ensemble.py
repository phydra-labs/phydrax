#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from math import isfinite
from types import BuiltinFunctionType, FunctionType
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
)
from .._strict import StrictModule
from ..dynamics import TimeGrid
from ._wiener import LevyAreaKind, WienerRealization


_CANONICAL_CONFIGURATION_MODULES = frozenset(
    {
        "builtins",
        "diffrax",
        "equinox",
        "jax",
        "jaxlib",
        "lineax",
        "numpy",
        "operator",
        "optimistix",
        "phydrax",
    }
)


def _qualified_type_name(value: Any, /) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _configuration_module_is_canonical(module: str, /) -> bool:
    return module.split(".", 1)[0] in _CANONICAL_CONFIGURATION_MODULES


def _opaque_configuration(
    value: Any,
    /,
    *,
    owner: str,
    identity_name: str,
    caller_identity: str | None,
) -> dict[str, Any]:
    if caller_identity is None:
        raise ValueError(
            f"{identity_name} is required because {owner} contains the "
            f"noncanonical object {_qualified_type_name(value)!r}."
        )
    return {
        "kind": "caller-identified",
        "type": _qualified_type_name(value),
        "identity": caller_identity,
    }


def _configuration_payload(
    value: Any,
    /,
    *,
    owner: str,
    identity_name: str,
    caller_identity: str | None,
) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, Enum):
        enum_type = type(value)
        module = enum_type.__module__
        if not _configuration_module_is_canonical(module):
            return _opaque_configuration(
                value,
                owner=owner,
                identity_name=identity_name,
                caller_identity=caller_identity,
            )
        return {
            "kind": "enum",
            "type": _qualified_type_name(value),
            "name": value.name,
        }
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}
    if isinstance(value, np.generic) or eqx.is_array(value):
        return {
            "kind": "array",
            "content": array_tree_fingerprint(np.asarray(value)),
        }
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is float:
        return {"kind": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "kind": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "value": value.hex()}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _configuration_payload(
                    item,
                    owner=f"{owner}[{index}]",
                    identity_name=identity_name,
                    caller_identity=caller_identity,
                )
                for index, item in enumerate(value)
            ],
        }
    if type(value) is list:
        return {
            "kind": "list",
            "items": [
                _configuration_payload(
                    item,
                    owner=f"{owner}[{index}]",
                    identity_name=identity_name,
                    caller_identity=caller_identity,
                )
                for index, item in enumerate(value)
            ],
        }
    if type(value) is set or type(value) is frozenset:
        items = [
            _configuration_payload(
                item,
                owner=f"{owner}[{index}]",
                identity_name=identity_name,
                caller_identity=caller_identity,
            )
            for index, item in enumerate(value)
        ]
        items.sort(key=canonical_json)
        return {
            "kind": "set" if type(value) is set else "frozenset",
            "items": items,
        }
    if type(value) is dict:
        entries = [
            (
                _configuration_payload(
                    key,
                    owner=f"{owner}.key",
                    identity_name=identity_name,
                    caller_identity=caller_identity,
                ),
                _configuration_payload(
                    item,
                    owner=f"{owner}[{key!r}]",
                    identity_name=identity_name,
                    caller_identity=caller_identity,
                ),
            )
            for key, item in value.items()
        ]
        entries.sort(key=lambda entry: canonical_json(entry[0]))
        return {
            "kind": "mapping",
            "items": [
                {"key": key_payload, "value": item_payload}
                for key_payload, item_payload in entries
            ],
        }
    if isinstance(value, type):
        module = value.__module__
        if _configuration_module_is_canonical(module) and "<" not in value.__qualname__:
            return {
                "kind": "type",
                "value": f"{module}.{value.__qualname__}",
            }
        return _opaque_configuration(
            value,
            owner=owner,
            identity_name=identity_name,
            caller_identity=caller_identity,
        )
    if is_dataclass(value):
        module = type(value).__module__
        if not _configuration_module_is_canonical(module):
            return _opaque_configuration(
                value,
                owner=owner,
                identity_name=identity_name,
                caller_identity=caller_identity,
            )
        return {
            "kind": "module",
            "type": _qualified_type_name(value),
            "fields": [
                {
                    "name": field.name,
                    "value": _configuration_payload(
                        getattr(value, field.name),
                        owner=f"{owner}.{field.name}",
                        identity_name=identity_name,
                        caller_identity=caller_identity,
                    ),
                }
                for field in fields(value)
            ],
        }
    if isinstance(value, (FunctionType, BuiltinFunctionType)):
        module = value.__module__
        qualname = value.__qualname__
        if (
            isinstance(module, str)
            and _configuration_module_is_canonical(module)
            and "<" not in qualname
        ):
            return {"kind": "function", "value": f"{module}.{qualname}"}
        return _opaque_configuration(
            value,
            owner=owner,
            identity_name=identity_name,
            caller_identity=caller_identity,
        )
    value_type = type(value)
    if (
        _configuration_module_is_canonical(value_type.__module__)
        and value_type.__dictoffset__ != 0
    ):
        return {
            "kind": "object",
            "type": _qualified_type_name(value),
            "attributes": _configuration_payload(
                vars(value),
                owner=f"{owner}.__dict__",
                identity_name=identity_name,
                caller_identity=caller_identity,
            ),
        }
    return _opaque_configuration(
        value,
        owner=owner,
        identity_name=identity_name,
        caller_identity=caller_identity,
    )


def _configuration_fingerprint(
    value: Any,
    caller_identity: str | None,
    /,
    *,
    owner: str,
) -> str:
    identity_name = f"{owner}_id"
    if caller_identity is not None and (
        not isinstance(caller_identity, str) or not caller_identity
    ):
        raise ValueError(f"{identity_name} must be a non-empty string or None.")
    if value is None and caller_identity is not None:
        raise ValueError(f"{identity_name} requires a corresponding {owner}.")
    return canonical_fingerprint(
        {
            "kind": "stochastic-path-ensemble-solve-setting-v1",
            "owner": owner,
            "caller_identity": caller_identity,
            "configuration": _configuration_payload(
                value,
                owner=owner,
                identity_name=identity_name,
                caller_identity=caller_identity,
            ),
        }
    )


PathEnsembleStatus: TypeAlias = Literal[0, 1, 2]
PATH_ENSEMBLE_SUCCESS: PathEnsembleStatus = 0
PATH_ENSEMBLE_BACKEND_FAILURE: PathEnsembleStatus = 1
PATH_ENSEMBLE_NONFINITE: PathEnsembleStatus = 2


class StochasticPathEnsemblePlan(StrictModule):
    """Fixed-output, fixed-capacity policy for a keyed stochastic path ensemble.

    Adaptive steps remain internal to the selected Diffrax method. The public state
    shape, output mesh, path count, maximum step count, and Wiener construction are
    immutable for one prepared execution epoch. Canonical library configurations are
    fingerprinted structurally; opaque settings require their corresponding ``*_id``.
    """

    time_grid: TimeGrid
    solver: Any
    stepsize_controller: Any
    adjoint: Any
    event: Any
    dt0: Array
    solver_fingerprint: str = eqx.field(static=True)
    stepsize_controller_fingerprint: str = eqx.field(static=True)
    adjoint_fingerprint: str = eqx.field(static=True)
    event_fingerprint: str = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    wiener_tolerance: float = eqx.field(static=True)
    levy_area: LevyAreaKind = eqx.field(static=True)
    dense: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_grid: TimeGrid,
        /,
        *,
        path_count: int,
        dt0: ArrayLike,
        solver: Any = None,
        solver_id: str | None = None,
        stepsize_controller: Any = None,
        stepsize_controller_id: str | None = None,
        adjoint: Any = None,
        adjoint_id: str | None = None,
        event: Any = None,
        event_id: str | None = None,
        max_steps: int = 4096,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-8,
        wiener_tolerance: float = 1.0e-3,
        levy_area: LevyAreaKind = "brownian",
        dense: bool = False,
        throw: bool = False,
        plan_id: str | None = None,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        count = int(path_count)
        maximum = int(max_steps)
        if count <= 0 or maximum <= 0:
            raise ValueError("path_count and max_steps must be positive.")
        initial_step = jnp.asarray(dt0)
        if initial_step.shape != () or not bool(jnp.isfinite(initial_step)):
            raise ValueError("dt0 must be one finite scalar.")
        relative = float(rtol)
        absolute = float(atol)
        tolerance = float(wiener_tolerance)
        if not all(
            isfinite(value) and value > 0.0 for value in (relative, absolute, tolerance)
        ):
            raise ValueError(
                "rtol, atol, and wiener_tolerance must be finite and positive."
            )
        if levy_area not in ("brownian", "space_time", "space_time_time"):
            raise ValueError("Unknown Levy-area representation.")
        solver_fingerprint = _configuration_fingerprint(solver, solver_id, owner="solver")
        controller_fingerprint = _configuration_fingerprint(
            stepsize_controller,
            stepsize_controller_id,
            owner="stepsize_controller",
        )
        adjoint_fingerprint = _configuration_fingerprint(
            adjoint, adjoint_id, owner="adjoint"
        )
        event_fingerprint = _configuration_fingerprint(event, event_id, owner="event")
        configuration_id = canonical_fingerprint(
            {
                "kind": "stochastic-path-ensemble-configuration-v2",
                "time_grid": {
                    "identity": time_grid.time_id,
                    "content": array_tree_fingerprint(time_grid.times),
                },
                "path_count": count,
                "dt0": array_tree_fingerprint(initial_step),
                "solver": solver_fingerprint,
                "stepsize_controller": controller_fingerprint,
                "adjoint": adjoint_fingerprint,
                "event": event_fingerprint,
                "max_steps": maximum,
                "rtol": relative.hex(),
                "atol": absolute.hex(),
                "wiener_tolerance": tolerance.hex(),
                "levy_area": levy_area,
                "dense": bool(dense),
                "throw": bool(throw),
            }
        )
        resolved_id = configuration_id if plan_id is None else plan_id
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("plan_id must be a non-empty string.")
        self.time_grid = time_grid
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint
        self.event = event
        self.dt0 = initial_step
        self.solver_fingerprint = solver_fingerprint
        self.stepsize_controller_fingerprint = controller_fingerprint
        self.adjoint_fingerprint = adjoint_fingerprint
        self.event_fingerprint = event_fingerprint
        self.path_count = count
        self.max_steps = maximum
        self.rtol = relative
        self.atol = absolute
        self.wiener_tolerance = tolerance
        self.levy_area = levy_area
        self.dense = bool(dense)
        self.throw = bool(throw)
        self.configuration_id = configuration_id
        self.plan_id = resolved_id


class PreparedStochasticPathEnsemble(StrictModule):
    """Validated path-ensemble inputs with a stable realization identity."""

    problem: Any
    plan: StochasticPathEnsemblePlan
    realization: WienerRealization
    initial_states: Array | None
    prepared_id: str = eqx.field(static=True)


class StochasticPathEnsembleResult(StrictModule):
    """Fixed-grid paths with per-path backend and temporal evidence."""

    solution: Any
    states: Array
    times: Array
    path_valid: Array
    status: Array
    accepted_steps: Array
    rejected_steps: Array
    temporal_evidence: Any
    event_mask: Any
    path_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return jnp.all(self.path_valid) & jnp.all(self.status == PATH_ENSEMBLE_SUCCESS)


def prepare_stochastic_path_ensemble(
    problem: Any,
    plan: StochasticPathEnsemblePlan,
    /,
    *,
    realization: WienerRealization | None = None,
    key: Key[Array, ""] | None = None,
    initial_states: ArrayLike | None = None,
) -> PreparedStochasticPathEnsemble:
    """Prepare one bounded stochastic ensemble without executing an integrator."""
    from ..solver._differential import DifferentialProblem

    if not isinstance(problem, DifferentialProblem):
        raise TypeError("problem must be a DifferentialProblem.")
    if not problem.stochastic:
        raise ValueError("StochasticPathEnsemblePlan requires stochastic forcing.")
    if not isinstance(plan, StochasticPathEnsemblePlan):
        raise TypeError("plan must be a StochasticPathEnsemblePlan.")
    grid = np.asarray(plan.time_grid.times, dtype=float)
    if grid[0] < float(problem.t0) or grid[-1] > float(problem.t1):
        raise ValueError("The output TimeGrid must lie inside the problem interval.")
    if realization is None:
        if key is None:
            raise ValueError("key is required when realization is not supplied.")
        selected = WienerRealization.independent(
            key,
            problem.noise_shape,
            support=(float(problem.t0), float(problem.t1)),
            sample_shape=(plan.path_count,),
            tolerance=plan.wiener_tolerance,
            levy_area=plan.levy_area,
            noise_id=problem.noise_id,
            label=f"path-ensemble:{plan.plan_id}",
        )
    else:
        selected = realization
        if key is not None:
            raise ValueError("Supply realization or key, not both.")
    if not isinstance(selected, WienerRealization):
        raise TypeError("realization must be a WienerRealization.")
    if selected.sample_shape != (plan.path_count,):
        raise ValueError("realization sample_shape must equal (path_count,).")
    if (
        selected.noise_shape != problem.noise_shape
        or selected.noise_id != problem.noise_id
    ):
        raise ValueError("realization noise identity must match the problem.")
    initials = None if initial_states is None else jnp.asarray(initial_states)
    if initials is not None:
        expected = (plan.path_count,) + tuple(problem.initial_state.shape)
        if initials.shape != expected or initials.dtype != problem.initial_state.dtype:
            raise ValueError(
                "initial_states must match path_count, state shape, and dtype."
            )
    initial_states_identity = (
        {
            "present": False,
            "content": array_tree_fingerprint(problem.initial_state),
        }
        if initials is None
        else {
            "present": True,
            "content": array_tree_fingerprint(initials),
        }
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-stochastic-path-ensemble-v2",
            "problem": problem.problem_id,
            "plan": plan.plan_id,
            "execution_configuration": plan.configuration_id,
            "realization": selected.realization_id,
            "initial_states": initial_states_identity,
        }
    )
    return PreparedStochasticPathEnsemble(
        problem=problem,
        plan=plan,
        realization=selected,
        initial_states=initials,
        prepared_id=prepared_id,
    )


def solve_stochastic_path_ensemble(
    prepared: PreparedStochasticPathEnsemble, /
) -> StochasticPathEnsembleResult:
    """Execute a prepared ensemble through the canonical Diffrax backend."""
    from ..solver._diffrax_backend import solve_diffrax_ensemble

    if not isinstance(prepared, PreparedStochasticPathEnsemble):
        raise TypeError("prepared must be a PreparedStochasticPathEnsemble.")
    plan = prepared.plan
    solution = solve_diffrax_ensemble(
        prepared.problem,
        save_times=plan.time_grid.times,
        realization=prepared.realization,
        initial_states=prepared.initial_states,
        solver=plan.solver,
        stepsize_controller=plan.stepsize_controller,
        adjoint=plan.adjoint,
        dt0=plan.dt0,
        event=plan.event,
        rtol=plan.rtol,
        atol=plan.atol,
        max_steps=plan.max_steps,
        dense=plan.dense,
        throw=plan.throw,
        solver_configuration_id=plan.configuration_id,
    )
    flat_valid = solution.valid.reshape((plan.path_count, -1))
    path_valid = jnp.all(flat_valid, axis=-1) & solution.backend_successful.reshape(
        (plan.path_count,)
    )
    finite = jnp.all(
        jnp.isfinite(solution.states).reshape((plan.path_count, -1)), axis=-1
    )
    status = jnp.where(
        ~finite,
        PATH_ENSEMBLE_NONFINITE,
        jnp.where(path_valid, PATH_ENSEMBLE_SUCCESS, PATH_ENSEMBLE_BACKEND_FAILURE),
    ).astype(jnp.int32)
    accepted = jnp.asarray(solution.stats["num_accepted_steps"], dtype=jnp.int32).reshape(
        (plan.path_count,)
    )
    rejected = jnp.asarray(solution.stats["num_rejected_steps"], dtype=jnp.int32).reshape(
        (plan.path_count,)
    )
    result_id = canonical_fingerprint(
        {
            "kind": "stochastic-path-ensemble-result-v2",
            "prepared": prepared.prepared_id,
            "execution_configuration": plan.configuration_id,
        }
    )
    return StochasticPathEnsembleResult(
        solution=solution,
        states=solution.states,
        times=solution.times,
        path_valid=path_valid,
        status=status,
        accepted_steps=accepted,
        rejected_steps=rejected,
        temporal_evidence=solution.temporal_evidence,
        event_mask=solution.event_mask,
        path_count=plan.path_count,
        plan_id=plan.plan_id,
        prepared_id=prepared.prepared_id,
        result_id=result_id,
        realization_id=prepared.realization.realization_id,
        coupling_id=prepared.realization.coupling_id,
        approximation_kind="finite-keyed-path-ensemble",
    )


__all__ = [
    "PATH_ENSEMBLE_BACKEND_FAILURE",
    "PATH_ENSEMBLE_NONFINITE",
    "PATH_ENSEMBLE_SUCCESS",
    "PathEnsembleStatus",
    "PreparedStochasticPathEnsemble",
    "StochasticPathEnsemblePlan",
    "StochasticPathEnsembleResult",
    "prepare_stochastic_path_ensemble",
    "solve_stochastic_path_ensemble",
]

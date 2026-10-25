"""Small native fixtures for collective-variable, bias and ABF tests."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.atomistic._dynamics import AtomisticDynamicsPlan
from phydrax.atomistic._term_state import AtomisticTermStateBindings
from phydrax.atomistic.sampling import (
    AbstractBiasForm,
    AtomisticBiasTerm,
    CollectiveVariableKind,
    CollectiveVariablePlan,
    CollectiveVariableProgram,
)
from tests._support.atomistic_dynamics_fixtures import (
    ExternalFieldTerm,
    measure,
    prepare_potential,
    row,
    table,
)


def zero_field() -> ExternalFieldTerm:
    return ExternalFieldTerm("zero", "zero", [0.0])


def cv_program(
    system: Any, kind: CollectiveVariableKind, indices: Any, /, **kwargs: Any
) -> Any:
    return CollectiveVariableProgram(
        (CollectiveVariablePlan(kind, indices, **kwargs).prepare(system),)
    )


def bias_term(
    system: Any, variables: Any, form: AbstractBiasForm, /, **kwargs: Any
) -> AtomisticBiasTerm:
    return AtomisticBiasTerm(variables, form, name="bias", force_group=1, **kwargs)


def biased_dynamics(
    system: Any,
    term: AtomisticBiasTerm,
    integrator: Any,
    /,
    *,
    constraints: Any = None,
    force_ports: tuple[Any, ...] = (),
    physical: tuple[Any, ...] | None = None,
) -> Any:
    terms = (zero_field(), term) if physical is None else (*physical, term)
    potential = prepare_potential(system, terms)
    return AtomisticDynamicsPlan(
        system, potential, integrator, constraints=constraints, force_ports=force_ports
    ).prepare()


def initial_bindings(
    potential: Any, dtype: Any = jnp.float64, /
) -> AtomisticTermStateBindings:
    states = {}
    for term in potential.terms:
        value = term.initial_term_state(dtype)
        if value is not None:
            states[term.name] = value
    return AtomisticTermStateBindings(states)


def nve_table(dynamics: Any) -> Any:
    return table(dynamics, row(measure(dynamics.system), "nve"))


def start(dynamics: Any, thermodynamic: Any, positions: Any, /, *, seed: int = 0) -> Any:
    position = jnp.asarray(positions, dtype=jnp.float64)
    return dynamics.initialize_state(
        position,
        thermodynamic,
        velocity=jnp.zeros_like(position),
        key=jax.random.key(seed),
    )


def kabsch_msd(x: np.ndarray, y: np.ndarray, /) -> float:
    """Independent Kabsch (SVD) proper-rotation mean-square deviation."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.mean(np.sum((xc @ rotation - yc) ** 2, axis=-1)))

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Repository invariants for array-role declarations (D3) and hidden callable state."""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import importlib
import inspect
import pkgutil
import sys
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import (
    ExplicitFreeze,
    NonTrainableState,
    ParameterOwner,
    resolve_array_roles,
)


# Extension points whose implementations may be learned components. A
# NonTrainableState field typed with one of them (or with a trainable model) must
# not exist: the terminal marker would freeze the component silently.
_SLOT_BASES = (
    ("phydrax._model._array", "AbstractArrayModel"),
    ("phydrax.discretization.finite_volume._riemann", "AbstractNumericalFluxPlan"),
    (
        "phydrax.discretization.finite_volume._reconstruction",
        "AbstractFaceReconstructionPlan",
    ),
    ("phydrax.discretization.finite_volume._reconstruction", "AbstractSlopeLimiter"),
    ("phydrax.discretization.finite_volume._closure", "AbstractFaceClosurePlan"),
    ("phydrax.solver._fixed_step", "AbstractAcceptedStepTransform"),
    ("phydrax._numerics._ssp_runge_kutta", "AbstractSSPRKStageTransform"),
    (
        "phydrax.dynamics.identification._neural_transition",
        "AbstractDiscreteModelRolloutTransition",
    ),
    ("phydrax.linalg._preconditioners", "AbstractPreconditioner"),
    ("phydrax.nonlinear._updates", "AbstractNonlinearUpdate"),
    ("phydrax.equations._transport_closures", "AbstractTransportClosure"),
    ("phydrax.equations._material_point", "AbstractMPMConstitutivePlan"),
    ("phydrax.equations._finite_element_material", "AbstractConstitutiveModel"),
    ("phydrax.equations.fem._materials", "AbstractLocalImplicitMaterial"),
    ("phydrax.control._parameterization", "AbstractControlParameterization"),
    ("phydrax.stochastic._state_space", "AbstractTransitionKernel"),
    ("phydrax.stochastic._state_space", "AbstractObservationModel"),
    ("phydrax.stochastic._process", "AbstractPathwiseTransition"),
)

# Containers on the plan-embedded training path (G8) that must stay neutral even
# where no field is typed with a slot directly: prepared solves and solve
# policies stay FIXED by design (D26), so D3 is not enforced transitively.
_REQUIRED_NEUTRAL = (
    ("phydrax.discretization.finite_volume._dynamics", "FiniteVolumeMethodPlan"),
    ("phydrax.discretization.finite_volume._dynamics", "PreparedFiniteVolumeDynamics"),
    (
        "phydrax.discretization.finite_volume._triangle_dynamics",
        "TriangleFiniteVolumeMethodPlan",
    ),
    (
        "phydrax.discretization.finite_volume._triangle_dynamics",
        "PreparedTriangleFiniteVolumeDynamics",
    ),
    (
        "phydrax.discretization.finite_volume._unstructured_dynamics",
        "UnstructuredFiniteVolumeMethodPlan",
    ),
    (
        "phydrax.discretization.finite_volume._unstructured_dynamics",
        "PreparedUnstructuredFiniteVolumeDynamics",
    ),
    (
        "phydrax.discretization.finite_volume._block_amr",
        "PreparedBlockAMRFiniteVolumeDynamics",
    ),
    (
        "phydrax.discretization.finite_difference._flux_differencing",
        "SBPFluxDifferencingMethodPlan",
    ),
    (
        "phydrax.discretization.finite_difference._flux_differencing",
        "PreparedSBPConservationDynamics",
    ),
    ("phydrax.discretization.finite_volume._reconstruction", "MUSCLReconstruction"),
    ("phydrax.discretization.finite_volume._riemann", "EntropyStableFluxPlan"),
    ("phydrax.solver._fixed_step", "CompositeAcceptedStepTransform"),
    ("phydrax.solver._fixed_step", "AbstractFixedStepMethod"),
    ("phydrax.solver._fixed_step", "CallableFixedStepMethod"),
    ("phydrax.solver._fixed_step", "AbstractSSPRKFixedStepMethod"),
    ("phydrax.solver._fixed_step", "SSPRK33FixedStepMethod"),
    ("phydrax.solver._fixed_step", "SSPRK54FixedStepMethod"),
    ("phydrax.solver._fixed_step", "FixedStepProblem"),
    ("phydrax.solver._fixed_step", "FixedStepRolloutPlan"),
    ("phydrax.solver._finite_volume_rollout", "AdaptiveFiniteVolumeRolloutPlan"),
    ("phydrax.solver._finite_volume_rollout", "ScheduledFiniteVolumeRolloutPlan"),
    ("phydrax.equations._transport_closures", "PrandtlTransport"),
    ("phydrax.linalg._preconditioners", "PrecisionCastPreconditioner"),
    ("phydrax.linalg._multigrid", "MultigridLevel"),
    ("phydrax.linalg._subspace_correction", "SubspaceCorrectionTerm"),
    ("phydrax.linalg._block_preconditioning", "BlockFactorizationPreconditioner"),
    (
        "phydrax.atomistic.sampling._model_collective_variable",
        "ModelCollectiveVariableProgram",
    ),
    ("phydrax.stochastic._state_space", "GaussianObservationModel"),
    ("phydrax.closure_data._binding", "PreparedSpectralDriftHook"),
)


def _name(cls: type, /) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _subclasses(root: type, /) -> set[type]:
    seen: set[type] = set()
    stack = [root]
    while stack:
        for child in stack.pop().__subclasses__():
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


@functools.cache
def _repository() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """Import every phydrax module; return its dataclass classes and slot bases."""
    for info in pkgutil.walk_packages(phx.__path__, "phydrax."):
        try:
            importlib.import_module(info.name)
        except ImportError:
            # Optional-dependency modules cannot define built-in classes here.
            continue
    candidates = (
        _subclasses(eqx.Module)
        | _subclasses(NonTrainableState)
        | _subclasses(ParameterOwner)
    )
    classes = tuple(
        sorted(
            (
                cls
                for cls in candidates
                if cls.__module__.startswith("phydrax") and dataclasses.is_dataclass(cls)
            ),
            key=_name,
        )
    )
    slots = tuple(
        getattr(importlib.import_module(module), name) for module, name in _SLOT_BASES
    )
    return classes, slots


@functools.cache
def _class_namespace() -> dict[str, type]:
    namespace: dict[str, type] = {}
    for module in tuple(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("phydrax"):
            continue
        for key, value in vars(module).items():
            if inspect.isclass(value):
                namespace.setdefault(key, value)
    return namespace


@functools.cache
def _hints(cls: type, /) -> dict[str, typing.Any]:
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception:
        hints: dict[str, typing.Any] = {}
        for base in reversed(cls.__mro__):
            module = sys.modules.get(base.__module__)
            namespace = {**_class_namespace(), **(vars(module) if module else {})}
            for key, annotation in vars(base).get("__annotations__", {}).items():
                try:
                    hints[key] = (
                        eval(annotation, namespace)
                        if isinstance(annotation, str)
                        else annotation
                    )
                except Exception:
                    hints[key] = None
        return hints


def _dynamic_fields(cls: type, /) -> list[tuple[dataclasses.Field, typing.Any]]:
    hints = _hints(cls)
    return [
        (field, hints.get(field.name))
        for field in dataclasses.fields(cls)
        if not field.metadata.get("static", False)
    ]


def _admits_component(
    annotation: typing.Any, components: tuple[type, ...], supertypes: frozenset[type], /
) -> bool:
    """Whether a value of `annotation` may be a slot implementation or a model.

    A class admits one when it is a component class or one of its supertypes
    (e.g. `StrictModule`). `Any`, `object`, callables, protocols, type variables
    and unresolved names are not analyzed. `NonTrainableState` types (including
    `ExplicitFreeze`) never count: their instances are terminal.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _admits_component(typing.get_args(annotation)[0], components, supertypes)
    if origin in (type, collections.abc.Callable, typing.Literal, typing.ClassVar):
        return False
    if origin is not None:
        return any(
            _admits_component(argument, components, supertypes)
            for argument in typing.get_args(annotation)
            if argument is not Ellipsis
        )
    if (
        not isinstance(annotation, type)
        or annotation is object
        or getattr(annotation, "_is_protocol", False)
        or issubclass(annotation, NonTrainableState)
    ):
        return False
    return annotation in supertypes or issubclass(annotation, components)


def _admits_inexact_array(annotation: typing.Any, /) -> bool:
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _admits_inexact_array(typing.get_args(annotation)[0])
    if origin in (type, collections.abc.Callable, typing.Literal, typing.ClassVar):
        return False
    if origin is not None:
        return any(
            _admits_inexact_array(argument)
            for argument in typing.get_args(annotation)
            if argument is not Ellipsis
        )
    if not isinstance(annotation, type):
        return False
    if issubclass(annotation, jaxtyping.AbstractArray):
        dtypes = annotation.dtypes
        return not isinstance(dtypes, tuple) or any(
            dtype.startswith(("float", "bfloat", "complex")) for dtype in dtypes
        )
    return issubclass(annotation, (jax.Array, np.ndarray))


@functools.cache
def _components() -> tuple[tuple[type, ...], frozenset[type]]:
    """Component classes (slot bases, trainable models) and all their supertypes."""
    classes, slots = _repository()
    models = tuple(
        cls
        for cls in classes
        if issubclass(cls, ParameterOwner) and not issubclass(cls, NonTrainableState)
    )
    supertypes = frozenset(base for cls in (*slots, *models) for base in cls.__mro__) - {
        object
    }
    return (*slots, ParameterOwner), supertypes


def test_non_trainable_state_never_types_a_field_that_may_hold_a_component() -> None:
    classes, _ = _repository()
    components, supertypes = _components()
    violations = sorted(
        f"{_name(cls)}.{field.name}: {annotation!r}"
        for cls in classes
        if issubclass(cls, NonTrainableState) and not issubclass(cls, ExplicitFreeze)
        for field, annotation in _dynamic_fields(cls)
        if _admits_component(annotation, components, supertypes)
    )
    assert not violations, (
        "NonTrainableState is terminal, so these fields would silently freeze a "
        "trainable component; drop NonTrainableState from the holder and declare "
        "its own arrays with fixed_field, or hold the component in an ExplicitFreeze "
        "holder on purpose:\n" + "\n".join(violations)
    )


def test_plan_embedded_training_path_is_neutral() -> None:
    _repository()
    terminal = sorted(
        f"{module}.{name}"
        for module, name in _REQUIRED_NEUTRAL
        if issubclass(getattr(sys.modules[module], name), NonTrainableState)
    )
    assert not terminal, (
        "These containers carry trainable components on the plan-embedded training "
        "path; they must stay neutral and declare their own arrays with "
        "fixed_field:\n" + "\n".join(terminal)
    )


def test_slot_bases_are_neutral() -> None:
    _, slots = _repository()
    terminal = sorted(
        _name(slot) for slot in slots if issubclass(slot, NonTrainableState)
    )
    assert not terminal, (
        "Slot bases must stay neutral so learned implementations can train; mark "
        "each fixed built-in implementation NonTrainableState instead:\n"
        + "\n".join(terminal)
    )


def _neutral_slot_implementations() -> list[type]:
    classes, slots = _repository()
    return [
        cls
        for cls in classes
        if issubclass(cls, slots)
        and not inspect.isabstract(cls)
        and not cls.__name__.lstrip("_").startswith("Abstract")
        and not issubclass(cls, (NonTrainableState, ParameterOwner))
    ]


def test_neutral_slot_implementations_declare_their_inexact_fields() -> None:
    violations = sorted(
        f"{_name(cls)}.{field.name}: {annotation!r}"
        for cls in _neutral_slot_implementations()
        for field, annotation in _dynamic_fields(cls)
        if "phydrax_role" not in field.metadata and _admits_inexact_array(annotation)
    )
    assert not violations, (
        "Neutral slot implementations have no default role; declare these arrays "
        "with fixed_field or parameter_field, or mark a fixed built-in "
        "NonTrainableState:\n" + "\n".join(violations)
    )


def test_default_constructible_slot_implementations_classify_every_array() -> None:
    violations: list[str] = []
    for cls in _neutral_slot_implementations():
        try:
            instance = cls()
        except Exception:
            continue
        resolution = resolve_array_roles(instance)
        violations.extend(f"{_name(cls)}{path}" for path in resolution.unclassified)
        violations.extend(
            f"{_name(cls)}{path} [{kind}]" for path, kind, _ in resolution.violations
        )
    assert not violations, "\n".join(violations)


def _module_function(x):
    return jnp.tanh(x)


class _Holder(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    function: typing.Any
    static_function: typing.Any = eqx.field(static=True, default=_module_function)

    def __call__(self, x):
        return self.function(self.weight * x)


def _captures(value):
    return lambda x: x * value


@pytest.mark.parametrize(
    "function",
    [
        _module_function,
        lambda x: 2.0 * x,
        _captures(3.0),
        _captures(jnp.arange(3)),
        _Holder(jnp.ones(3), _module_function),
    ],
    ids=["module-function", "lambda", "python-float", "integer-array", "component"],
)
def test_declared_callable_categories_are_admitted(function) -> None:
    phx.require_parameter_roles(_Holder(jnp.ones(3), function), context="probe")


class _FrozenProvider(phx.StrictModule, phx.ExplicitFreeze):
    provider: typing.Any


def test_visible_terminal_providers_are_not_searched() -> None:
    frozen = _FrozenProvider(_captures(jnp.ones(3)))
    phx.require_parameter_roles(_Holder(jnp.ones(3), frozen), context="probe")


@pytest.mark.parametrize(
    ("holder", "route"),
    [
        (
            _Holder(jnp.ones(3), _captures(jnp.ones(3))),
            ".function: closure variable 'value'",
        ),
        (
            _Holder(jnp.ones(3), lambda x, scale=jnp.ones(3): x * scale),
            ".function: default argument 0",
        ),
        (
            _Holder(jnp.ones(3), functools.partial(jnp.multiply, jnp.ones(3))),
            ".function: partial argument 0",
        ),
        (
            _Holder(jnp.ones(3), _captures(_Holder(jnp.ones(3), _module_function))),
            ".function: closure variable 'value' -> .weight",
        ),
        (
            _Holder(jnp.ones(3), _module_function, _captures(jnp.ones(3))),
            ".static_function: static field -> closure variable 'value'",
        ),
    ],
    ids=["closure", "default", "partial", "captured-model", "static-field"],
)
def test_training_preflight_rejects_hidden_inexact_state_with_its_path(
    holder, route
) -> None:
    with pytest.raises(ValueError, match="training entry") as error:
        phx.require_parameter_roles(holder, context="training entry")
    assert route in str(error.value)

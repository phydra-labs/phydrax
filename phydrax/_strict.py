"""
Based off `ihoop.strict`.

Phydrax-specific deviations (kept intentionally):
- Adds `StrictModule` (Equinox integration via a combined metaclass).
- Treats `Abstract*` / `_Abstract*`-named classes as abstract, even without declared abstract elements.
- Allows Equinox internal wrapper classes (`equinox.*`) to subclass concrete strict classes.
- Allows overriding dunder methods (e.g. `__repr__`) from strict bases.
- Resolves `eqx.AbstractVar[T]` / `eqx.AbstractClassVar[T]` annotations that
  `from __future__ import annotations` stringified. Equinox only inspects the raw
  annotation, so without this they silently become concrete dataclass fields.
- Classes that declare `__strict_contract__ = True` (or inherit it) check their
  `phydrax.typing` field contracts, read-only, after Equinox construction.
- Binds methods as Equinox `BoundMethod` pytrees without Equinox's construction
  pass. That pass walks every receiver leaf to warn about JAX-transformed
  fields, so each method access cost the size of the receiver.
"""

from __future__ import annotations

import abc
import dataclasses
import re
import types
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING, TypeVar

import equinox as eqx
from equinox._module._flatten import WRAPPER_FIELD_NAMES
from equinox._module._prebuilt import BoundMethod

from ._typing_plan import validate_constructed


_StrictMetaT = TypeVar("_StrictMetaT", bound="_StrictMeta")
_StrictEqxMetaT = TypeVar("_StrictEqxMetaT", bound="_StrictEqxMeta")


def _is_strict_subclass(cls: type) -> bool:
    return issubclass(cls, Strict) and cls is not Strict


class _StrictMeta(abc.ABCMeta):
    _strict_is_abstract_: bool
    _strict_contract_: bool

    def __new__(
        mcs: type[_StrictMetaT],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> _StrictMetaT:
        """
        Runs when a class inheriting from Strict is defined.
        """
        # just check the initial letters as keyword, that way we can have multiple
        # Strict classes that could resolve metaclass conflicts
        is_defining_strict_itself = (
            name[:6] == "Strict" and namespace.get("__module__") == mcs.__module__
        )

        if not is_defining_strict_itself:
            if not any(issubclass(b, Strict) for b in bases):
                raise TypeError("Classes using _StrictMeta must inherit from Strict.")

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        cls._strict_is_abstract_ = (
            bool(
                cls.__abstractmethods__
                or getattr(cls, "__abstractvars__", ())
                or getattr(cls, "__abstractclassvars__", ())
            )
            or is_defining_strict_itself
        )

        # Skip checks for the base strict class itself
        if is_defining_strict_itself:
            cls._strict_contract_ = False
            return cls

        # Structural contracts are opt-in and inherited; a subclass cannot
        # withdraw an inherited opt-in, so the declaration is only ever `True`.
        if (
            "__strict_contract__" in namespace
            and namespace["__strict_contract__"] is not True
        ):
            raise TypeError(
                f"'{cls.__module__}.{name}' may only declare __strict_contract__ = True."
            )
        cls._strict_contract_ = "__strict_contract__" in namespace or any(
            isinstance(base, _StrictMeta) and base._strict_contract_ for base in bases
        )
        if cls._strict_contract_ and not dataclasses.is_dataclass(cls):
            raise TypeError(
                f"'{cls.__module__}.{name}' declares structural contracts but is not "
                "a dataclass-based StrictModule."
            )

        has_abstract_name = (
            name.startswith("Abstract")
            or name.startswith("_Abstract")
            or namespace.get("__strict_abstract__", False) is True
        )
        # Treat classes with abstract-style names as abstract, even if they have
        # no explicitly-declared abstract elements. This mirrors common patterns
        # in libraries that use naming conventions for abstract bases.
        if not cls._strict_is_abstract_ and has_abstract_name:
            cls._strict_is_abstract_ = True
        if cls._strict_is_abstract_:
            if not has_abstract_name:
                abs_methods = list(cls.__abstractmethods__)
                abs_attrs = list(getattr(cls, "__abstractvars__", ()))
                raise TypeError(
                    f"Abstract class '{cls.__module__}.{name}' must have a name "
                    "starting with 'Abstract' or '_Abstract', or declare "
                    "__strict_abstract__ = True. Abstract elements:"
                    f" methods={abs_methods}, attributes={abs_attrs}"
                )
        else:  # Concrete class
            if has_abstract_name:
                raise TypeError(
                    f"Concrete (final) class '{cls.__module__}.{name}' must not "
                    "have a name starting with 'Abstract' or '_Abstract'."
                )

        for base in bases:
            if not _is_strict_subclass(base):
                continue

            # Cannot inherit from a concrete strict class
            if not getattr(base, "_strict_is_abstract_", True):
                # Allow Equinox's internal initable wrapper classes to subclass
                # concrete modules (e.g. equinox._module's _InitableModule)
                if namespace.get("__module__", "").startswith("equinox."):
                    continue
                raise TypeError(
                    f"Cannot inherit from concrete (final) class '{base.__name__}'. "
                    f"Class '{name}' attempts to inherit from it. "
                    "strict classes are either abstract or final."
                )

        return cls

    # Runtime construction performs the abstract/final refusal and the freeze
    # transition. Static checkers must not see this override: a metaclass
    # `__call__` replaces every concrete generated or custom `__init__` signature,
    # so construction is typed by each concrete class instead.
    if not TYPE_CHECKING:

        def __call__(cls, *args: Any, **kwargs: Any) -> "Strict":
            """Create one concrete strict instance and complete its freeze transition."""
            if getattr(cls, "_strict_is_abstract_", False):
                abs_methods = list(cls.__abstractmethods__)
                abs_attrs = list(getattr(cls, "__abstractvars__", ()))
                raise TypeError(
                    f"Cannot instantiate abstract class {cls.__name__}. "
                    f"Abstract elements: methods={abs_methods}, attributes={abs_attrs}"
                )

            instance = super().__call__(*args, **kwargs)

            if cls._strict_contract_:
                validate_constructed(instance)
            mark_strict_initialized(instance)

            return instance


def mark_strict_initialized(instance: Any, /) -> None:
    """Complete the freeze transition of one strict instance.

    Equinox modules are already frozen by Equinox and must not carry the flag:
    Equinox flattens any non-field `__dict__` entry as wrapper metadata, so every
    unflattened copy would grow `__name__`/`__qualname__` set to its MISSING
    sentinel and break `filter_jit` naming.
    """
    if not isinstance(instance, eqx.Module):
        object.__setattr__(instance, "_strict_initialized", True)


_reconstructible_strict_types: set[type[StrictModule]] = set()
_ReconstructedStrict = TypeVar("_ReconstructedStrict", bound="StrictModule")


def register_strict_reconstruction(value_type: type[StrictModule], /) -> None:
    """Admit one exact concrete StrictModule type for structural reconstruction."""
    if not issubclass(value_type, StrictModule) or value_type._strict_is_abstract_:
        raise TypeError("Strict reconstruction requires a concrete StrictModule type.")
    _reconstructible_strict_types.add(value_type)


def reconstruct_strict(
    value_type: type[_ReconstructedStrict], values: Mapping[str, Any], /
) -> _ReconstructedStrict:
    """Reconstruct registered structure, not mathematical proof authority."""
    if value_type not in _reconstructible_strict_types:
        raise TypeError("Strict reconstruction requires an explicitly registered type.")
    expected = {field.name for field in dataclasses.fields(value_type)}
    if set(values) != expected:
        raise ValueError(
            "Strict reconstruction requires the complete declared field set."
        )
    result = object.__new__(value_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    if value_type._strict_contract_:
        validate_constructed(result)
    mark_strict_initialized(result)
    return result


class Strict(metaclass=_StrictMeta):
    """
    Base class for creating immutable objects with Abstract/Final inheritance.

    Inherit from this class to enforce:
    1. Immutability: Attributes cannot be changed or deleted after __init__ completes.
    2. Abstract/Final: Classes are either abstract (must be subclassed, cannot be
       instantiated) or final (concrete, cannot be subclassed). Abstract classes
       must be named starting with 'Abstract' or '_Abstract'. Concrete classes
       must not start with these prefixes.
    3. Abstract Elements: Use `abc.abstractmethod` for methods and
       `equinox.AbstractVar[Type]` for instance attributes subclasses must define.
       Abstract bases may also provide ordinary overridable default behavior.
    """

    _strict_initialized: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._strict_initialized:
            raise AttributeError(
                f"Cannot set attribute '{name}' on frozen instance "
                f"of {type(self).__name__}. strict objects are immutable "
                "after initialization."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if self._strict_initialized:
            raise AttributeError(
                f"Cannot delete attribute '{name}' on frozen instance "
                f"of {type(self).__name__}. strict objects are immutable "
                "after initialization."
            )
        super().__delattr__(name)


_STRINGIFIED_ABSTRACT = re.compile(
    r"(?:eqx\.|equinox\.)?(AbstractVar|AbstractClassVar)\[(.*)\]", re.DOTALL
)


def _bind_method(function: Any, receiver: "StrictModule", /) -> BoundMethod[Any]:
    """Return Equinox's bound-method pytree of one method without a leaf walk.

    The value is the one `BoundMethod(function, receiver)` builds: the same type,
    fields, wrapper metadata and PyTree structure, holding the receiver as its
    subnode. Equinox's module construction would first flatten every receiver
    leaf to warn about JAX-transformed fields, which the receiver's own
    construction has already checked, and its remaining checks cannot fail for
    these two fields.
    """
    bound = object.__new__(BoundMethod)
    object.__setattr__(bound, "__func__", function)
    object.__setattr__(bound, "__self__", receiver)
    for name in WRAPPER_FIELD_NAMES:
        try:
            value = getattr(function, name)
        except AttributeError:
            continue
        object.__setattr__(bound, name, value)
    return bound


class _StrictEqxMeta(_StrictMeta, type(eqx.Module)):
    def __new__(
        mcs: type[_StrictEqxMetaT],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> _StrictEqxMetaT:
        annotations = namespace.get("__annotations__", {})
        for field_name, annotation in annotations.items():
            if isinstance(annotation, str) and (
                match := _STRINGIFIED_ABSTRACT.fullmatch(annotation.strip())
            ):
                marker = getattr(eqx, match[1])
                annotations[field_name] = marker[match[2]]
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class StrictModule(eqx.Module, Strict, metaclass=_StrictEqxMeta):
    # Equinox freezes assignment; deletion must be refused here because modules
    # never carry `_strict_initialized` (see `mark_strict_initialized`).
    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"Cannot delete attribute '{name}' on frozen instance "
            f"of {type(self).__name__}. strict objects are immutable."
        )

    # Every method access returns a `BoundMethod` pytree, as on any Equinox
    # module, so a bound method still passes through `jit`/`vmap` with its
    # receiver as a PyTree; binding never flattens the receiver (see `_bind_method`).
    if not TYPE_CHECKING:

        def __getattribute__(self, name: str, /) -> Any:
            out = super(eqx.Module, self).__getattribute__(name)
            if (
                isinstance(out, types.MethodType)
                and out.__self__ is self
                and not (name.startswith("__") and name.endswith("__"))
                and name != "_abc_impl"
            ):
                return _bind_method(out.__func__, self)
            return out

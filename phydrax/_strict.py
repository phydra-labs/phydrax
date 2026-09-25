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
"""

import abc
import re
from typing import Any

import equinox as eqx


def _is_strict_subclass(cls: type) -> bool:
    return issubclass(cls, Strict) and cls is not Strict


class _StrictMeta(abc.ABCMeta):
    _strict_is_abstract_: bool

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
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
            return cls

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

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create one concrete strict instance and complete its freeze transition."""
        if getattr(cls, "_strict_is_abstract_", False):
            abs_methods = list(cls.__abstractmethods__)
            abs_attrs = list(getattr(cls, "__abstractvars__", ()))
            raise TypeError(
                f"Cannot instantiate abstract class {cls.__name__}. "
                f"Abstract elements: methods={abs_methods}, attributes={abs_attrs}"
            )

        instance = super().__call__(*args, **kwargs)

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


class _StrictEqxMeta(_StrictMeta, type(eqx.Module)):
    def __new__(mcs, name, bases, namespace, **kwargs):
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

"""
Based off `ihoop.strict`.

Phydrax-specific deviations (kept intentionally):
- Adds `StrictModule` (Equinox integration via a combined metaclass).
- Treats `Abstract*` / `_Abstract*`-named classes as abstract, even without declared abstract elements.
- Allows Equinox internal wrapper classes (`equinox.*`) to subclass concrete strict classes.
- Allows overriding dunder methods (e.g. `__repr__`) from strict bases.
"""

import abc
import inspect
import types
from typing import Annotated, Any, get_args, get_origin, TypeVar

import equinox as eqx


_T = TypeVar("_T")
_ABSTRACT_ATTRIBUTE_MARKER: str = "_AbstractAttributeMarker_Strict"

AbstractAttribute = Annotated[_T, _ABSTRACT_ATTRIBUTE_MARKER]


def _is_strict_subclass(cls: type) -> bool:
    return issubclass(cls, Strict) and cls is not Strict


def _is_abstract_attribute_annotation(annotation: Any) -> bool:
    return (
        get_origin(annotation) is Annotated
        and len(get_args(annotation)) == 2
        and get_args(annotation)[1] == _ABSTRACT_ATTRIBUTE_MARKER
    )


class _StrictMeta(abc.ABCMeta):
    _strict_abstract_attributes_: frozenset[str]
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

        abstract_attributes: set[str] = set()
        for base in bases:
            if hasattr(base, "_strict_abstract_attributes_"):
                abstract_attributes.update(getattr(base, "_strict_abstract_attributes_"))

        # Scan local annotations for new or resolved abstract attributes
        local_annotations = namespace.get("__annotations__", {})
        for attr_name, annotation in local_annotations.items():
            if _is_abstract_attribute_annotation(annotation):
                # Check if it's defined with a value,
                # abstract attrs shouldn't have values
                if attr_name in namespace:
                    raise TypeError(
                        f"Abstract attribute '{name}.{attr_name}' cannot be defined "
                        "with a value."
                    )
                abstract_attributes.add(attr_name)
            else:
                abstract_attributes.discard(attr_name)

        # Class or instance attributes defined directly also resolve abstract attributes
        for attr_name in namespace:
            if (
                not isinstance(namespace[attr_name], types.FunctionType)
                and not isinstance(namespace[attr_name], property)
                and not attr_name.startswith("__")
            ):
                abstract_attributes.discard(attr_name)

        # Determine if the class being defined is abstract
        # Abstract methods are handled by super().__new__
        # creating cls.__abstractmethods__
        current_abstract_attributes = frozenset(abstract_attributes)

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        cls._strict_abstract_attributes_ = current_abstract_attributes
        cls._strict_is_abstract_ = (
            bool(cls.__abstractmethods__ or cls._strict_abstract_attributes_)
            or is_defining_strict_itself
        )

        # Skip checks for the base strict class itself
        if is_defining_strict_itself:
            return cls

        has_abstract_name = name.startswith("Abstract") or name.startswith("_Abstract")
        # Treat classes with abstract-style names as abstract, even if they have
        # no explicitly-declared abstract elements. This mirrors common patterns
        # in libraries that use naming conventions for abstract bases.
        if not cls._strict_is_abstract_ and has_abstract_name:
            cls._strict_is_abstract_ = True
        if cls._strict_is_abstract_:
            if not has_abstract_name:
                abs_methods = list(cls.__abstractmethods__)
                abs_attrs = list(cls._strict_abstract_attributes_)
                raise TypeError(
                    f"Abstract class '{cls.__module__}.{name}' must have a name "
                    "starting with 'Abstract' or '_Abstract'. Abstract elements:"
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

            # Cannot override a concrete method from an strict base
            for meth_name, meth_obj in namespace.items():
                # Allow every class to override its own constructor
                if meth_name == "__init__":
                    continue

                # Only check functions defined directly in this class
                if isinstance(meth_obj, types.FunctionType):
                    # Allow overriding special/dunder methods like __hash__, __repr__, etc.
                    if meth_name.startswith("__") and meth_name.endswith("__"):
                        continue
                    base_meth = getattr(base, meth_name, None)
                    if (
                        base_meth
                        and callable(base_meth)
                        and not getattr(base_meth, "__isabstractmethod__", False)
                    ):
                        base_impl_func = base_meth
                        if hasattr(base_meth, "__func__"):
                            base_impl_func = base_meth.__func__
                        if meth_obj is not base_impl_func:
                            is_originally_abstract = False
                            for super_base in inspect.getmro(base):
                                if super_base is Strict or super_base is object:
                                    break
                                super_base_meth = getattr(super_base, meth_name, None)
                                if super_base_meth and getattr(
                                    super_base_meth, "__isabstractmethod__", False
                                ):
                                    is_originally_abstract = True
                                    break
                                if super_base_meth and super_base is base:
                                    break

                            if not is_originally_abstract:
                                raise TypeError(
                                    f"Cannot override concrete method '{meth_name}' "
                                    f"from base class '{base.__name__}' in class "
                                    f"'{name}'. Concrete methods in strict "
                                    "classes are implicitly final."
                                )
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Runs when an instance of an strict class is created
        """
        if getattr(cls, "_strict_is_abstract_", False):
            abs_methods = list(cls.__abstractmethods__)
            abs_attrs = list(getattr(cls, "_strict_abstract_attributes_", set()))
            raise TypeError(
                f"Cannot instantiate abstract class {cls.__name__}. "
                f"Abstract elements: methods={abs_methods}, attributes={abs_attrs}"
            )

        instance = super().__call__(*args, **kwargs)

        object.__setattr__(instance, "_strict_initialized", True)

        return instance


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
       `AbstractAttribute[Type]` for instance attributes that subclasses must define.
    4. Method Overriding: Concrete methods from base classes cannot be overridden.
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


class _StrictEqxMeta(_StrictMeta, type(eqx.Module)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class StrictModule(eqx.Module, Strict, metaclass=_StrictEqxMeta):
    pass

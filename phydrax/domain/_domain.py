#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import abc
from collections.abc import Callable, Mapping
from math import prod
from typing import Any, TYPE_CHECKING

import jax.numpy as jnp

from .._model import ModelPorts, PortMapping, ValuePort
from .._strict import StrictModule
from ..axes import AxisKey
from ._coordinate import CoordinateSpec


if TYPE_CHECKING:
    from .._model import ModelBinding
    from ._evaluation import FunctionBinding


class Domain(StrictModule):
    """Semantic domain composed from one or more independent joint factors."""

    __strict_abstract__ = True

    @property
    @abc.abstractmethod
    def labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def joint_factors(self) -> tuple["JointFactor", ...]:
        """Atomic factors whose coordinate supports cannot be split implicitly."""
        raise NotImplementedError

    @abc.abstractmethod
    def same_support(self, other: object, /) -> bool:
        """Return whether ``other`` denotes the same labeled measure support."""
        raise NotImplementedError

    @abc.abstractmethod
    def relabel(
        self,
        labels: str | Mapping[str, str],
        /,
    ) -> "Domain":
        """Return the same support with changed public coordinate bindings."""
        raise NotImplementedError

    def coordinate(self, label: str, /) -> CoordinateSpec:
        """Return the static coordinate schema bound to ``label``."""
        factor = self.factor(label)
        return factor.coordinate_specs[factor.labels.index(label)]

    def value_port(self, label: str, /) -> ValuePort:
        """Return the coordinate `ValuePort` of one label.

        Each port is label-scoped: `semantic_id` is the label, a scalar
        coordinate has the single component ID `label`, and an event of shape
        `s` has components `f"{label}[{i}]"` in row-major order. Event axis `j`
        has key `AxisKey(f"domain-label:{label}", str(j))`, so equal coordinate
        schemas under different labels never share a port. Coordinates are
        points, not vector components, so `variance` is `"neutral"`, and
        `representation` is `"domain-coordinates"`. Domains declare no coordinate
        units, space, frame, or normalization, so those fields are undeclared.
        PyTree and graph coordinates have no dense event and raise `ValueError`.
        """
        spec = self.coordinate(label)
        shape = spec.event_shape
        if shape is None:
            raise ValueError(
                f"Domain label {label!r} declares a {spec.kind} coordinate "
                "without a dense event; it has no ValuePort view."
            )
        scope = f"domain-label:{label}"
        return ValuePort(
            label,
            event_shape=shape,
            component_ids=(
                (label,)
                if not shape
                else tuple(f"{label}[{i}]" for i in range(prod(shape)))
            ),
            representation="domain-coordinates",
            axis_keys=tuple(AxisKey(scope, str(axis)) for axis in range(len(shape))),
        )

    def value_ports(self) -> tuple[ValuePort, ...]:
        """Return one coordinate `ValuePort` per label (see `value_port`), in `labels` order."""
        return tuple(self.value_port(label) for label in self.labels)

    def factor(self, label: str, /) -> "JointFactor":
        """Return the complete joint factor owning ``label``."""
        for factor in self.joint_factors:
            if label in factor.labels:
                return factor
        raise KeyError(f"Label {label!r} not in domain {self.labels}.")

    def schema_compatible(self, other: object, /) -> bool:
        """Return whether two domains expose the same evaluation schema."""
        if not isinstance(other, Domain) or self.labels != other.labels:
            return False
        return all(
            self.coordinate(label).compatible(other.coordinate(label))
            for label in self.labels
        )

    def is_subdomain_of(self, other: "Domain", /) -> bool:
        """Return whether every complete factor is preserved by ``other``."""
        if not isinstance(other, Domain):
            return False
        for factor in self.joint_factors:
            matches = tuple(
                candidate
                for candidate in other.joint_factors
                if candidate.labels == factor.labels
            )
            if len(matches) != 1 or not factor.same_support(matches[0]):
                return False
        return True

    def join(self, other: "Domain", /) -> "Domain":
        from ._product_domain import ProductDomain

        if self is other:
            return self
        return ProductDomain(self, other)

    def __matmul__(self, other: "Domain", /) -> "Domain":
        return self.join(other)

    def restrict(self, labels: tuple[str, ...], /) -> "Domain":
        requested = tuple(labels)
        if not requested:
            raise ValueError("A restricted domain must retain at least one label.")
        if len(set(requested)) != len(requested):
            raise ValueError(f"Restriction labels must be unique, got {requested}.")
        unknown = tuple(label for label in requested if label not in self.labels)
        if unknown:
            raise KeyError(f"Labels {unknown} not in domain {self.labels}.")
        selected: list[JointFactor] = []
        requested_set = set(requested)
        for factor in self.joint_factors:
            overlap = requested_set.intersection(factor.labels)
            if overlap and overlap != set(factor.labels):
                raise ValueError(
                    f"Cannot implicitly restrict part of coupled factor {factor.labels}; requested {requested}."
                )
            if overlap:
                selected.append(factor)
        if tuple(label for factor in selected for label in factor.labels) == self.labels:
            return self
        if len(selected) == 1:
            return selected[0]
        from ._product_domain import ProductDomain

        return ProductDomain(*selected)

    def drop(self, labels: str | tuple[str, ...], /) -> "Domain":
        dropped = (labels,) if isinstance(labels, str) else tuple(labels)
        unknown = tuple(label for label in dropped if label not in self.labels)
        if unknown:
            raise KeyError(f"Labels {unknown} not in domain {self.labels}.")
        kept = tuple(label for label in self.labels if label not in set(dropped))
        if not kept:
            raise ValueError("Cannot drop all labels from a domain.")
        return self.restrict(kept)

    def component(
        self,
        spec: Any = None,
        *,
        where: Mapping[str, Callable] | None = None,
        where_all: Any = None,
        weight_all: Any = None,
    ):
        from ._components import DomainComponent
        from ._selection import SelectionSpec

        spec_ = spec if isinstance(spec, SelectionSpec) else SelectionSpec(spec)
        return DomainComponent(
            domain=self,
            spec=spec_,
            where=where,
            where_all=where_all,
            weight_all=weight_all,
        )

    def Function(
        self,
        *deps: str,
        binding: "FunctionBinding | None" = None,
    ):
        """Bind a pointwise callable or explicit batch evaluator to this domain."""
        from ._evaluation import (
            BatchEvaluator,
            FunctionBinding,
            PointwiseEvaluator,
        )
        from ._function import _reject_model_state, DomainFunction

        if binding is not None and not isinstance(binding, FunctionBinding):
            raise TypeError("binding must be a FunctionBinding or None.")
        for dep in deps:
            if dep not in self.labels:
                raise ValueError(
                    f"Unknown dependency label {dep!r}; expected subset of {self.labels}."
                )

        def decorator(function):
            if not callable(function):
                if binding is not None:
                    raise TypeError("Constant domain functions do not accept a binding.")
                return DomainFunction(domain=self, deps=deps, func=function)
            if isinstance(function, BatchEvaluator):
                if binding is not None:
                    raise TypeError(
                        "BatchEvaluator implementations declare their own call contract."
                    )
                evaluator = function
            else:
                evaluator = PointwiseEvaluator(function, binding=binding)
            _reject_model_state(evaluator, context="Domain.Function")
            return DomainFunction(domain=self, deps=deps, func=evaluator)

        return decorator

    def Model(
        self,
        *deps: str,
        binding: "ModelBinding | None" = None,
        port_mapping: PortMapping | None = None,
    ):
        """Bind a model with an explicit domain input contract.

        A model declaring intrinsic ports (a `PortProvider`, such as a fitted ML
        executable) requires `port_mapping`, binding each model input port to
        the `value_port` of one dependency label. Dependencies are packed in
        `deps` order and never repacked, so the labels mapped from the model's
        ordered input ports must be exactly `deps`. The bound field publishes the
        model's output ports unchanged, so `port_mapping` maps inputs only. The
        resulting `PortBindingEvidence` is the field's `port_binding`. Models
        without intrinsic ports take no mapping.
        """
        from .._model import ModelBinding, ModelEvaluator, ModelMetadataProvider
        from .._model._ports import (
            bind_model_ports,
            intrinsic_model_ports,
            require_mapped_order,
        )
        from ._function import (
            _reject_model_state,
            DomainFunction,
            PORT_BINDING_METADATA_KEY,
        )
        from ._model_function import ConcatenatedModelEvaluator

        if binding is not None and not isinstance(binding, ModelBinding):
            raise TypeError("binding must be a ModelBinding or None.")
        if port_mapping is not None:
            if not isinstance(port_mapping, PortMapping):
                raise TypeError("port_mapping must be a PortMapping or None.")
            if port_mapping.outputs:
                raise ValueError(
                    "Domain.Model publishes the model's output ports unchanged; "
                    "port_mapping binds input ports only."
                )

        deps_ = self.labels if not deps else deps
        for dep in deps_:
            if dep not in self.labels:
                raise ValueError(
                    f"Unknown dependency label {dep!r}; expected subset of {self.labels}."
                )

        def decorator(model):
            _reject_model_state(model, context="Domain.Model")
            if isinstance(model, ModelEvaluator):
                declared_binding = model.input_binding()
                if binding is not None and binding != declared_binding:
                    raise ValueError(
                        "Phydrax models declare their ModelBinding; caller overrides must match that declaration."
                    )
                resolved_binding = declared_binding
            else:
                if binding is None:
                    raise TypeError(
                        "Plain callable models require binding=phx.domain.ModelBinding(...)."
                    )
                resolved_binding = binding

            metadata: dict[str, Any] = {}
            if isinstance(model, ModelMetadataProvider):
                declared = model.model_metadata()
                if not isinstance(declared, Mapping):
                    raise TypeError("Model metadata providers must return a mapping.")
                if any(not isinstance(name, str) or not name for name in declared):
                    raise ValueError("Model metadata keys must be nonempty strings.")
                if PORT_BINDING_METADATA_KEY in declared:
                    raise ValueError(
                        f"Model metadata key {PORT_BINDING_METADATA_KEY!r} is reserved "
                        "for the domain port binding."
                    )
                metadata.update(declared)

            site = f"Domain.Model{tuple(deps_)!r}"
            model_ports = intrinsic_model_ports(model)
            if model_ports is not None:
                owner_inputs = tuple(self.value_port(dep) for dep in deps_)
                model_outputs = model_ports.outputs
                mapping = (
                    None
                    if port_mapping is None
                    else PortMapping(
                        inputs=port_mapping.inputs,
                        outputs=[(port.port_id, port.port_id) for port in model_outputs],
                    )
                )
                owner = ModelPorts(inputs=owner_inputs, outputs=model_outputs)
            else:
                mapping, owner = port_mapping, ModelPorts(inputs=(), outputs=())
            evidence = bind_model_ports(model, owner, mapping, site=site)
            if evidence is not None:
                require_mapped_order(
                    evidence,
                    "input",
                    tuple(port.port_id for port in owner.inputs),
                    site=site,
                )
                metadata[PORT_BINDING_METADATA_KEY] = evidence

            return DomainFunction(
                domain=self,
                deps=deps_,
                func=ConcatenatedModelEvaluator(
                    model,
                    domain_labels=self.labels,
                    deps=tuple(deps_),
                    binding=resolved_binding,
                ),
                metadata=metadata,
            )

        return decorator

    def Parameter(
        self,
        init: Any,
        *,
        transform: Callable[[Any], Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        from ._function import (
            _TrainableConstCallable,
            DomainFunction,
            UnaryFieldEvaluator,
        )

        if init is None:
            raise TypeError("Domain.Parameter requires init to be array-like, not None.")

        raw = jnp.asarray(init)
        if not jnp.issubdtype(raw.dtype, jnp.inexact):
            raw = raw.astype("float64")

        if transform is None:
            return DomainFunction(
                domain=self,
                deps=(),
                func=_TrainableConstCallable(raw),
                metadata=metadata,
            )

        if not callable(transform):
            raise TypeError("Domain.Parameter transform must be a callable or None.")

        return DomainFunction(
            domain=self,
            deps=(),
            func=UnaryFieldEvaluator(_TrainableConstCallable(raw), transform),
            metadata=metadata,
        )


class JointFactor(Domain):
    """Atomic support that may own one or several intrinsically coupled coordinates."""

    __strict_abstract__ = True

    @property
    def joint_factors(self) -> tuple["JointFactor", ...]:
        return (self,)

    @property
    @abc.abstractmethod
    def coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def bind_component(
        self,
        selections: Mapping[str, Any],
        /,
    ) -> Any:
        """Validate factor selections and bind their base measure."""
        raise NotImplementedError

    @abc.abstractmethod
    def _same_factor_support(self, other: object, /) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _replace_labels(
        self,
        labels: tuple[str, ...],
        /,
    ) -> "JointFactor":
        raise NotImplementedError

    def same_support(self, other: object, /) -> bool:
        return (
            isinstance(other, JointFactor)
            and self.labels == other.labels
            and self._same_factor_support(other)
        )

    def relabel(
        self,
        labels: str | Mapping[str, str],
        /,
    ) -> "JointFactor":
        if isinstance(labels, str):
            if len(self.labels) != 1:
                raise ValueError(
                    "Relabeling a coupled factor requires a complete label mapping."
                )
            replacement = (labels,)
        else:
            unknown = tuple(label for label in labels if label not in self.labels)
            if unknown:
                raise KeyError(
                    f"Relabel mapping contains unknown labels {unknown}; expected a subset of {self.labels}."
                )
            replacement = tuple(labels.get(label, label) for label in self.labels)
        if len(set(replacement)) != len(replacement):
            raise ValueError(f"Relabeling produced duplicate labels {replacement}.")
        if replacement == self.labels:
            return self
        return self._replace_labels(replacement)

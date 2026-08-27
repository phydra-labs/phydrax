#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
    canonical_mapping,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


ComplexInterchangeSemantics: TypeAlias = Literal[
    "trainable-parameters",
    "frame-coefficients",
    "constrained-frame-coefficients",
    "meromorphic-coefficients",
    "pole-locations",
]

_SEMANTICS = frozenset(
    {
        "trainable-parameters",
        "frame-coefficients",
        "constrained-frame-coefficients",
        "meromorphic-coefficients",
        "pole-locations",
    }
)
_COMPONENT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def _entry_name(value: str, /) -> str:
    name = str(value)
    parts = name.split("/")
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("Complex interchange entry names must be safe relative paths.")
    return name


def _component_dtype(value: str, /) -> str:
    dtype = str(value)
    if dtype not in _COMPONENT_DTYPES:
        raise ValueError(
            "Complex interchange component dtype must be float16, bfloat16, "
            "float32, or float64."
        )
    return dtype


def _complex_component_dtype(value: ArrayLike, /) -> str:
    dtype = jnp.asarray(value).dtype
    if dtype == jnp.dtype(jnp.complex64):
        return "float32"
    if dtype == jnp.dtype(jnp.complex128):
        return "float64"
    raise TypeError("Complex interchange values must use complex64 or complex128.")


def _real_component_dtype(value: ArrayLike, /) -> str:
    dtype = jnp.asarray(value).dtype
    name = str(dtype)
    if name not in _COMPONENT_DTYPES:
        raise TypeError("Complex parameter source leaves must use real floating dtypes.")
    return name


def _complex_value(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.iscomplexobj(array) or array.dtype not in (
        jnp.dtype(jnp.complex64),
        jnp.dtype(jnp.complex128),
    ):
        raise TypeError(f"{name} must be a complex64 or complex128 array.")
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _compose_complex(real: ArrayLike, imaginary: ArrayLike, /) -> Array:
    real_ = jnp.asarray(real)
    imaginary_ = jnp.asarray(imaginary)
    if real_.shape != imaginary_.shape or real_.dtype != imaginary_.dtype:
        raise ValueError("Real and imaginary parameter leaves must match exactly.")
    if not jnp.issubdtype(real_.dtype, jnp.floating):
        raise TypeError("Complex parameter leaves must use real floating components.")
    return real_ + 1j * imaginary_


class ComplexInterchangeEntry(StrictModule, NonTrainableState):
    """One named mathematical complex parameter or coefficient array."""

    value: Array
    name: str = eqx.field(static=True)
    role: str = eqx.field(static=True)
    component_dtype: str = eqx.field(static=True)
    trainable: bool = eqx.field(static=True)
    entry_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        /,
        *,
        role: str,
        component_dtype: str | None = None,
        trainable: bool,
    ):
        name_ = _entry_name(name)
        role_ = str(role)
        if not role_:
            raise ValueError("Complex interchange entry role must be nonempty.")
        value_ = _complex_value(value, name_)
        dtype = _component_dtype(
            _complex_component_dtype(value_)
            if component_dtype is None
            else component_dtype
        )
        self.value = value_
        self.name = name_
        self.role = role_
        self.component_dtype = dtype
        self.trainable = bool(trainable)
        self.entry_id = canonical_fingerprint(
            {
                "kind": "complex-interchange-entry",
                "name": name_,
                "role": role_,
                "component_dtype": dtype,
                "trainable": bool(trainable),
                "value": array_tree_fingerprint(value_),
            }
        )


class ComplexInterchangeState(StrictModule, NonTrainableState):
    """Canonical mathematical complex state independent of internal leaf storage."""

    entries: tuple[ComplexInterchangeEntry, ...]
    semantics: ComplexInterchangeSemantics = eqx.field(static=True)
    provider_kind: str = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    metadata_json: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantics: ComplexInterchangeSemantics,
        provider_kind: str,
        architecture_id: str,
        entries: tuple[ComplexInterchangeEntry, ...],
        /,
        *,
        metadata: Mapping[str, Any] | None = None,
    ):
        if semantics not in _SEMANTICS:
            raise ValueError("Unknown complex interchange semantics.")
        provider = str(provider_kind)
        architecture = str(architecture_id)
        if not provider or not architecture:
            raise ValueError(
                "Complex interchange provider and architecture IDs are required."
            )
        entries_ = tuple(sorted(tuple(entries), key=lambda entry: entry.name))
        if not entries_ or not all(
            isinstance(entry, ComplexInterchangeEntry) for entry in entries_
        ):
            raise TypeError("Complex interchange states require explicit entries.")
        names = tuple(entry.name for entry in entries_)
        if len(set(names)) != len(names):
            raise ValueError("Complex interchange entry names must be unique.")
        metadata_ = canonical_mapping({} if metadata is None else metadata)
        metadata_json = canonical_json(metadata_)
        self.entries = entries_
        self.semantics = semantics
        self.provider_kind = provider
        self.architecture_id = architecture
        self.metadata_json = metadata_json
        self.state_id = canonical_fingerprint(
            {
                "kind": "complex-interchange-state",
                "semantics": semantics,
                "provider_kind": provider,
                "architecture_id": architecture,
                "entries": [entry.entry_id for entry in entries_],
                "metadata": metadata_,
            }
        )

    @property
    def metadata(self) -> dict[str, Any]:
        value = json.loads(self.metadata_json)
        if not isinstance(value, dict):
            raise RuntimeError("Complex interchange metadata is not an object.")
        return value

    def entry(self, name: str, /) -> ComplexInterchangeEntry:
        name_ = _entry_name(name)
        for entry in self.entries:
            if entry.name == name_:
                return entry
        raise KeyError(f"Complex interchange state has no entry {name_!r}.")

    @classmethod
    def from_entries(
        cls,
        semantics: ComplexInterchangeSemantics,
        provider_kind: str,
        architecture_id: str,
        values: Mapping[str, ArrayLike],
        /,
        *,
        roles: Mapping[str, str] | None = None,
        component_dtypes: Mapping[str, str] | None = None,
        trainable: Mapping[str, bool] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ComplexInterchangeState:
        if not values:
            raise ValueError("Complex interchange values cannot be empty.")
        roles_ = (
            {}
            if roles is None
            else {str(key): str(value) for key, value in roles.items()}
        )
        dtypes_ = (
            {}
            if component_dtypes is None
            else {str(key): str(value) for key, value in component_dtypes.items()}
        )
        trainable_ = (
            {}
            if trainable is None
            else {str(key): bool(value) for key, value in trainable.items()}
        )
        names = {str(name) for name in values}
        for mapping, label in (
            (roles_, "roles"),
            (dtypes_, "component_dtypes"),
            (trainable_, "trainable"),
        ):
            unknown = set(mapping) - names
            if unknown:
                raise ValueError(f"Complex interchange {label} contain unknown names.")
        entries = tuple(
            ComplexInterchangeEntry(
                str(name),
                value,
                role=roles_.get(str(name), "parameter"),
                component_dtype=dtypes_.get(str(name)),
                trainable=trainable_.get(str(name), True),
            )
            for name, value in values.items()
        )
        return cls(
            semantics,
            provider_kind,
            architecture_id,
            entries,
            metadata=metadata,
        )


class ComplexImportPolicy(StrictModule, NonTrainableState):
    """Precision and placement policy for importing into existing real leaves."""

    allow_precision_loss: bool = eqx.field(static=True)
    preserve_sharding: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        allow_precision_loss: bool = False,
        preserve_sharding: bool = True,
    ):
        self.allow_precision_loss = bool(allow_precision_loss)
        self.preserve_sharding = bool(preserve_sharding)


_PRECISION_ORDER = {
    "float16": 1,
    "bfloat16": 1,
    "float32": 2,
    "float64": 3,
}


def _cast_component(
    entry: ComplexInterchangeEntry,
    component: Array,
    target: Array,
    policy: ComplexImportPolicy,
    /,
) -> Array:
    target_dtype = _real_component_dtype(target)
    source_dtype = entry.component_dtype
    narrowing = _PRECISION_ORDER[source_dtype] > _PRECISION_ORDER[target_dtype]
    incomparable_low_precision = (
        _PRECISION_ORDER[source_dtype] == _PRECISION_ORDER[target_dtype]
        and source_dtype != target_dtype
    )
    if (narrowing or incomparable_low_precision) and not policy.allow_precision_loss:
        raise ValueError(
            f"Importing {entry.name!r} from {source_dtype} into {target_dtype} "
            "would lose precision."
        )
    result = jnp.asarray(component, dtype=target.dtype)
    if policy.preserve_sharding and isinstance(target, jax.Array):
        result = jax.device_put(result, target.sharding)
    return result


def _state_entries(
    state: ComplexInterchangeState,
    expected: set[str],
    /,
) -> dict[str, ComplexInterchangeEntry]:
    entries = {entry.name: entry for entry in state.entries}
    missing = expected - set(entries)
    unknown = set(entries) - expected
    if missing or unknown:
        raise ValueError(
            "Complex interchange entries do not match the destination; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    return entries


def _size_payload(value: Any, /) -> str | int | list[int]:
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, int):
        return value
    return str(value)


def _complex_linear_architecture(layer: Any, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "complex-linear-interchange-architecture",
            "in_size": _size_payload(layer.in_size),
            "out_size": _size_payload(layer.out_size),
            "use_bias": layer.bias_real is not None,
        }
    )


def _low_rank_architecture(layer: Any, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "low-rank-complex-linear-interchange-architecture",
            "in_size": _size_payload(layer.in_size),
            "out_size": _size_payload(layer.out_size),
            "rank": int(layer.rank),
            "use_bias": layer.bias_real is not None,
        }
    )


def _entry(
    name: str,
    role: str,
    real: Array,
    imaginary: Array,
    /,
    *,
    trainable: bool = True,
) -> ComplexInterchangeEntry:
    return ComplexInterchangeEntry(
        name,
        _compose_complex(real, imaginary),
        role=role,
        component_dtype=_real_component_dtype(real),
        trainable=trainable,
    )


def _export_complex_linear(
    layer: Any, /, *, prefix: str = ""
) -> tuple[ComplexInterchangeEntry, ...]:
    entries = [
        _entry(
            f"{prefix}weight",
            "weight",
            layer.weight_real,
            layer.weight_imag,
        )
    ]
    if layer.bias_real is not None:
        entries.append(
            _entry(
                f"{prefix}bias",
                "bias",
                layer.bias_real,
                layer.bias_imag,
            )
        )
    return tuple(entries)


def _export_low_rank(
    layer: Any, /, *, prefix: str = ""
) -> tuple[ComplexInterchangeEntry, ...]:
    entries = [
        _entry(
            f"{prefix}input_factor",
            "input-factor",
            layer.input_factor_real,
            layer.input_factor_imag,
        ),
        _entry(
            f"{prefix}output_factor",
            "output-factor",
            layer.output_factor_real,
            layer.output_factor_imag,
        ),
    ]
    if layer.bias_real is not None:
        entries.append(
            _entry(
                f"{prefix}bias",
                "bias",
                layer.bias_real,
                layer.bias_imag,
            )
        )
    return tuple(entries)


def _import_pair(
    entry: ComplexInterchangeEntry,
    real: Array,
    imaginary: Array,
    policy: ComplexImportPolicy,
    /,
) -> tuple[Array, Array]:
    if entry.value.shape != real.shape or imaginary.shape != real.shape:
        raise ValueError(f"Complex interchange entry {entry.name!r} has the wrong shape.")
    return (
        _cast_component(entry, jnp.real(entry.value), real, policy),
        _cast_component(entry, jnp.imag(entry.value), imaginary, policy),
    )


def _import_complex_linear(
    layer: Any,
    entries: Mapping[str, ComplexInterchangeEntry],
    policy: ComplexImportPolicy,
    /,
    *,
    prefix: str = "",
):
    weight_real, weight_imag = _import_pair(
        entries[f"{prefix}weight"],
        layer.weight_real,
        layer.weight_imag,
        policy,
    )
    result = eqx.tree_at(
        lambda value: (value.weight_real, value.weight_imag),
        layer,
        (weight_real, weight_imag),
    )
    if layer.bias_real is not None:
        bias_real, bias_imag = _import_pair(
            entries[f"{prefix}bias"],
            layer.bias_real,
            layer.bias_imag,
            policy,
        )
        result = eqx.tree_at(
            lambda value: (value.bias_real, value.bias_imag),
            result,
            (bias_real, bias_imag),
        )
    return result


def _import_low_rank(
    layer: Any,
    entries: Mapping[str, ComplexInterchangeEntry],
    policy: ComplexImportPolicy,
    /,
    *,
    prefix: str = "",
):
    input_real, input_imag = _import_pair(
        entries[f"{prefix}input_factor"],
        layer.input_factor_real,
        layer.input_factor_imag,
        policy,
    )
    output_real, output_imag = _import_pair(
        entries[f"{prefix}output_factor"],
        layer.output_factor_real,
        layer.output_factor_imag,
        policy,
    )
    result = eqx.tree_at(
        lambda value: (
            value.input_factor_real,
            value.input_factor_imag,
            value.output_factor_real,
            value.output_factor_imag,
        ),
        layer,
        (input_real, input_imag, output_real, output_imag),
    )
    if layer.bias_real is not None:
        bias_real, bias_imag = _import_pair(
            entries[f"{prefix}bias"],
            layer.bias_real,
            layer.bias_imag,
            policy,
        )
        result = eqx.tree_at(
            lambda value: (value.bias_real, value.bias_imag),
            result,
            (bias_real, bias_imag),
        )
    return result


def _frame_layout(frame: Any, /) -> tuple[int, int]:
    from ..equations.trefftz._holomorphic_frame import HolomorphicPolynomialFrame
    from ..equations.trefftz._meromorphic import MeromorphicLinearFrame

    if isinstance(frame, HolomorphicPolynomialFrame):
        return frame.complex_output_size, frame.monomial_count
    if isinstance(frame, MeromorphicLinearFrame):
        return frame.complex_output_size, frame.feature_count
    raise TypeError(
        "Complex frame conversion supports HolomorphicPolynomialFrame and "
        "MeromorphicLinearFrame."
    )


def frame_coefficients_to_complex(frame: Any, coordinates: ArrayLike, /) -> Array:
    """Convert canonical real frame coordinates to output-by-feature coefficients."""
    output_size, feature_count = _frame_layout(frame)
    values = jnp.asarray(coordinates)
    expected = 2 * output_size * feature_count
    if values.shape != (expected,) or jnp.iscomplexobj(values):
        raise ValueError(
            "Frame coordinates must be one canonical real coefficient vector."
        )
    blocks = values.reshape((output_size, 2, feature_count))
    return blocks[:, 0] + 1j * blocks[:, 1]


def complex_coefficients_to_frame(frame: Any, coefficients: ArrayLike, /) -> Array:
    """Convert output-by-feature complex coefficients to canonical real coordinates."""
    output_size, feature_count = _frame_layout(frame)
    values = _complex_value(coefficients, "frame coefficients")
    if values.shape != (output_size, feature_count):
        raise ValueError("Complex frame coefficient matrix has invalid shape.")
    return jnp.stack((jnp.real(values), jnp.imag(values)), axis=1).reshape((-1,))


def _constrained_architecture(value: Any, /) -> str:
    certificate = value.frame.linear_frame_certificate()
    return certificate.frame_id


def _recover_free_coordinates(
    value: Any,
    coefficients: Array,
    policy: ComplexImportPolicy,
    /,
) -> Array:
    coefficient_map = value.coefficient_map
    full = complex_coefficients_to_frame(value.frame, coefficients)
    full = full.astype(coefficient_map.particular_coefficients.dtype)
    delta = full - coefficient_map.particular_coefficients
    nullspace = coefficient_map.operator.nullspace_basis
    if coefficient_map.nullity == 0:
        free = jnp.zeros((0,), dtype=full.dtype)
    else:
        free = jnp.swapaxes(nullspace, -1, -2) @ delta
    reconstructed = coefficient_map.particular_coefficients + nullspace @ free
    constraint_residual = (
        coefficient_map.operator.constraint_matrix @ full - coefficient_map.target
    )
    epsilon = jnp.finfo(full.dtype).eps
    scale = jnp.maximum(
        jnp.linalg.norm(full),
        jnp.maximum(jnp.linalg.norm(reconstructed), 1.0),
    )
    tolerance = jnp.maximum(
        coefficient_map.evidence.tolerance,
        512.0 * epsilon * max(full.size, 1) * scale,
    )
    if not bool(jnp.linalg.norm(reconstructed - full) <= tolerance) or not bool(
        jnp.linalg.norm(constraint_residual) <= tolerance
    ):
        raise ValueError(
            "Complex coefficients do not belong to the destination affine set."
        )
    target = value.free_coordinates
    entry = ComplexInterchangeEntry(
        "coefficients",
        coefficients,
        role="frame-coefficients",
        component_dtype=_real_component_dtype(full),
        trainable=False,
    )
    return _cast_component(entry, free, target, policy)


def export_complex_parameters(value: Any, /) -> ComplexInterchangeState:
    """Export supported real-Cartesian providers as mathematical complex state."""
    from ..equations.trefftz._holomorphic import HolomorphicPolynomialPotential
    from ..equations.trefftz._holomorphic_constraints import (
        ConstrainedHolomorphicPotential,
    )
    from ..equations.trefftz._meromorphic import (
        ConstrainedMeromorphicPotential,
        TrainablePoleSet,
    )
    from ..nn.layers._complex_linear import ComplexLinear
    from ..nn.layers._low_rank_complex_linear import LowRankComplexLinear
    from ..nn.models._holomorphic import HolomorphicMLP

    if isinstance(value, ComplexLinear):
        architecture = _complex_linear_architecture(value)
        return ComplexInterchangeState(
            "trainable-parameters",
            "complex-linear",
            architecture,
            _export_complex_linear(value),
            metadata={
                "in_size": _size_payload(value.in_size),
                "out_size": _size_payload(value.out_size),
                "use_bias": value.bias_real is not None,
            },
        )
    if isinstance(value, LowRankComplexLinear):
        architecture = _low_rank_architecture(value)
        return ComplexInterchangeState(
            "trainable-parameters",
            "low-rank-complex-linear",
            architecture,
            _export_low_rank(value),
            metadata={
                "in_size": _size_payload(value.in_size),
                "out_size": _size_payload(value.out_size),
                "rank": int(value.rank),
                "use_bias": value.bias_real is not None,
            },
        )
    if isinstance(value, HolomorphicMLP):
        entries: list[ComplexInterchangeEntry] = []
        layer_kinds = []
        for layer_index, layer in enumerate(value.layers):
            prefix = f"layers/{layer_index}/"
            if isinstance(layer, ComplexLinear):
                entries.extend(_export_complex_linear(layer, prefix=prefix))
                layer_kinds.append("dense")
            elif isinstance(layer, LowRankComplexLinear):
                entries.extend(_export_low_rank(layer, prefix=prefix))
                layer_kinds.append("low-rank")
            else:
                raise TypeError("HolomorphicMLP contains an unsupported affine layer.")
        return ComplexInterchangeState(
            "trainable-parameters",
            "holomorphic-mlp",
            value.architecture_id,
            tuple(entries),
            metadata={
                "normalization_id": value.normalization.normalization_id,
                "layer_kinds": layer_kinds,
                "linear_ranks": list(value.linear_ranks),
                "holomorphic_certificate_id": value.holomorphic_certificate().certificate_id,
            },
        )
    if isinstance(value, HolomorphicPolynomialPotential):
        certificate = value.holomorphic_certificate()
        architecture = certificate.construction_dependencies[0]
        return ComplexInterchangeState(
            "trainable-parameters",
            "holomorphic-polynomial-potential",
            architecture,
            (
                _entry(
                    "coefficients",
                    "polynomial-coefficients",
                    value.coefficient_real,
                    value.coefficient_imag,
                ),
            ),
            metadata={
                "branches": value.branches,
                "maximum_degree": value.maximum_degree,
                "normalization_id": value.normalization.normalization_id,
                "holomorphic_certificate_id": certificate.certificate_id,
            },
        )
    if isinstance(value, ConstrainedHolomorphicPotential):
        coefficients = frame_coefficients_to_complex(
            value.frame, value.coefficient_vector
        )
        certificate = value.holomorphic_certificate()
        return ComplexInterchangeState(
            "constrained-frame-coefficients",
            "constrained-holomorphic-potential",
            _constrained_architecture(value),
            (
                ComplexInterchangeEntry(
                    "coefficients",
                    coefficients,
                    role="frame-coefficients",
                    component_dtype=_real_component_dtype(value.coefficient_vector),
                    trainable=False,
                ),
            ),
            metadata={
                "frame_id": value.frame.linear_frame_certificate().frame_id,
                "prepared_operator_id": value.coefficient_map.operator.prepared_id,
                "affine_map_id": value.coefficient_map.map_id,
                "target": array_tree_fingerprint(value.coefficient_map.target),
                "holomorphic_certificate_id": certificate.certificate_id,
            },
        )
    if isinstance(value, ConstrainedMeromorphicPotential):
        coefficients = frame_coefficients_to_complex(
            value.frame, value.coefficient_vector
        )
        certificate = value.meromorphic_certificate()
        return ComplexInterchangeState(
            "meromorphic-coefficients",
            "constrained-meromorphic-potential",
            value.frame.linear_frame_certificate().frame_id,
            (
                ComplexInterchangeEntry(
                    "coefficients",
                    coefficients,
                    role="regular-and-principal-part-coefficients",
                    component_dtype=_real_component_dtype(value.coefficient_vector),
                    trainable=False,
                ),
            ),
            metadata={
                "frame_id": value.frame.linear_frame_certificate().frame_id,
                "pole_set_id": value.frame.poles.pole_set_id,
                "affine_map_id": value.coefficient_map.map_id,
                "meromorphic_certificate_id": certificate.certificate_id,
            },
        )
    if isinstance(value, TrainablePoleSet):
        architecture = canonical_fingerprint(
            {
                "kind": "trainable-pole-set-interchange-architecture",
                "orders": list(value.orders),
            }
        )
        return ComplexInterchangeState(
            "pole-locations",
            "trainable-pole-set",
            architecture,
            (
                _entry(
                    "locations",
                    "pole-locations",
                    value.location_real,
                    value.location_imag,
                ),
            ),
            metadata={"orders": list(value.orders)},
        )
    raise TypeError(
        "Complex parameter export does not support "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def import_complex_parameters(
    value: Any,
    state: ComplexInterchangeState,
    /,
    *,
    policy: ComplexImportPolicy | None = None,
):
    """Import mathematical complex state into an existing real-Cartesian provider."""
    from ..equations.trefftz._holomorphic import HolomorphicPolynomialPotential
    from ..equations.trefftz._holomorphic_constraints import (
        ConstrainedHolomorphicPotential,
    )
    from ..equations.trefftz._meromorphic import (
        ConstrainedMeromorphicPotential,
        TrainablePoleSet,
    )
    from ..nn.layers._complex_linear import ComplexLinear
    from ..nn.layers._low_rank_complex_linear import LowRankComplexLinear
    from ..nn.models._holomorphic import HolomorphicMLP

    if not isinstance(state, ComplexInterchangeState):
        raise TypeError("state must be ComplexInterchangeState.")
    policy_ = ComplexImportPolicy() if policy is None else policy
    if not isinstance(policy_, ComplexImportPolicy):
        raise TypeError("policy must be ComplexImportPolicy or None.")

    if isinstance(value, ComplexLinear):
        architecture = _complex_linear_architecture(value)
        if (
            state.semantics != "trainable-parameters"
            or state.provider_kind != "complex-linear"
            or state.architecture_id != architecture
        ):
            raise ValueError("ComplexLinear interchange architecture mismatch.")
        expected = {"weight"} | ({"bias"} if value.bias_real is not None else set())
        return _import_complex_linear(value, _state_entries(state, expected), policy_)
    if isinstance(value, LowRankComplexLinear):
        architecture = _low_rank_architecture(value)
        if (
            state.semantics != "trainable-parameters"
            or state.provider_kind != "low-rank-complex-linear"
            or state.architecture_id != architecture
        ):
            raise ValueError("LowRankComplexLinear interchange architecture mismatch.")
        expected = {"input_factor", "output_factor"} | (
            {"bias"} if value.bias_real is not None else set()
        )
        return _import_low_rank(value, _state_entries(state, expected), policy_)
    if isinstance(value, HolomorphicMLP):
        if (
            state.semantics != "trainable-parameters"
            or state.provider_kind != "holomorphic-mlp"
            or state.architecture_id != value.architecture_id
        ):
            raise ValueError("HolomorphicMLP interchange architecture mismatch.")
        expected = set()
        for layer_index, layer in enumerate(value.layers):
            prefix = f"layers/{layer_index}/"
            if isinstance(layer, ComplexLinear):
                expected.add(f"{prefix}weight")
                if layer.bias_real is not None:
                    expected.add(f"{prefix}bias")
            elif isinstance(layer, LowRankComplexLinear):
                expected.update({f"{prefix}input_factor", f"{prefix}output_factor"})
                if layer.bias_real is not None:
                    expected.add(f"{prefix}bias")
            else:
                raise TypeError("HolomorphicMLP contains an unsupported affine layer.")
        entries = _state_entries(state, expected)
        layers = []
        for layer_index, layer in enumerate(value.layers):
            prefix = f"layers/{layer_index}/"
            if isinstance(layer, ComplexLinear):
                layers.append(
                    _import_complex_linear(layer, entries, policy_, prefix=prefix)
                )
            else:
                layers.append(_import_low_rank(layer, entries, policy_, prefix=prefix))
        return eqx.tree_at(lambda model: model.layers, value, tuple(layers))
    if isinstance(value, HolomorphicPolynomialPotential):
        architecture = value.holomorphic_certificate().construction_dependencies[0]
        if (
            state.semantics != "trainable-parameters"
            or state.provider_kind != "holomorphic-polynomial-potential"
            or state.architecture_id != architecture
        ):
            raise ValueError("Holomorphic polynomial interchange architecture mismatch.")
        entry = _state_entries(state, {"coefficients"})["coefficients"]
        real, imaginary = _import_pair(
            entry,
            value.coefficient_real,
            value.coefficient_imag,
            policy_,
        )
        return eqx.tree_at(
            lambda potential: (
                potential.coefficient_real,
                potential.coefficient_imag,
            ),
            value,
            (real, imaginary),
        )
    if isinstance(value, ConstrainedHolomorphicPotential):
        architecture = _constrained_architecture(value)
        if (
            state.semantics != "constrained-frame-coefficients"
            or state.provider_kind != "constrained-holomorphic-potential"
            or state.architecture_id != architecture
        ):
            raise ValueError("Constrained holomorphic interchange architecture mismatch.")
        coefficients = _state_entries(state, {"coefficients"})["coefficients"].value
        free = _recover_free_coordinates(value, coefficients, policy_)
        return eqx.tree_at(lambda potential: potential.free_coordinates, value, free)
    if isinstance(value, ConstrainedMeromorphicPotential):
        architecture = value.frame.linear_frame_certificate().frame_id
        if (
            state.semantics != "meromorphic-coefficients"
            or state.provider_kind != "constrained-meromorphic-potential"
            or state.architecture_id != architecture
        ):
            raise ValueError("Constrained meromorphic interchange architecture mismatch.")
        metadata = state.metadata
        if metadata.get("pole_set_id") != value.frame.poles.pole_set_id:
            raise ValueError("Constrained meromorphic pole-set identity mismatch.")
        coefficients = _state_entries(state, {"coefficients"})["coefficients"].value
        free = _recover_free_coordinates(value, coefficients, policy_)
        return eqx.tree_at(lambda potential: potential.free_coordinates, value, free)
    if isinstance(value, TrainablePoleSet):
        architecture = canonical_fingerprint(
            {
                "kind": "trainable-pole-set-interchange-architecture",
                "orders": list(value.orders),
            }
        )
        if (
            state.semantics != "pole-locations"
            or state.provider_kind != "trainable-pole-set"
            or state.architecture_id != architecture
        ):
            raise ValueError("Trainable pole-set interchange architecture mismatch.")
        entry = _state_entries(state, {"locations"})["locations"]
        real, imaginary = _import_pair(
            entry,
            value.location_real,
            value.location_imag,
            policy_,
        )
        locations = real + 1j * imaginary
        if len(set(complex(item) for item in np.asarray(locations))) != locations.size:
            raise ValueError("Imported pole locations must remain distinct.")
        return eqx.tree_at(
            lambda poles: (poles.location_real, poles.location_imag),
            value,
            (real, imaginary),
        )
    raise TypeError(
        "Complex parameter import does not support "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


__all__ = [
    "ComplexImportPolicy",
    "ComplexInterchangeEntry",
    "ComplexInterchangeSemantics",
    "ComplexInterchangeState",
    "complex_coefficients_to_frame",
    "export_complex_parameters",
    "frame_coefficients_to_complex",
    "import_complex_parameters",
]

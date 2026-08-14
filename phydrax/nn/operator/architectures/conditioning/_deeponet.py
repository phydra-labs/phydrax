#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax._frozendict import frozendict
from phydrax._strict import AbstractAttribute, StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.nn._base import _AbstractBaseModel
from phydrax.nn._keys import EvalKey, split_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.operator.data import FunctionSamples, OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


BranchFusion = Literal["sum", "product", "concat"]


def _query_coordinates(
    query: FunctionSamples,
    coord_dim: int,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    coordinates = query.coordinates_array(case_shape=case_shape)
    if int(coordinates.shape[-1]) != int(coord_dim):
        raise ValueError(
            f"Expected query coordinate dimension {coord_dim}, got "
            f"{coordinates.shape[-1]}."
        )
    return coordinates


def _sample_coordinates(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if not samples.axes and samples.coordinates is None:
        raise ValueError(
            "Variable-sensor branch encoding requires source coordinates or axes."
        )
    return samples.coordinates_array(case_shape=case_shape)


class _AbstractBranchEncoder(StrictModule):
    latent_size: AbstractAttribute[int]

    @abstractmethod
    def __call__(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_ndim: int,
        key: EvalKey = None,
    ) -> Array:
        raise NotImplementedError


class FixedBranchEncoder(_AbstractBranchEncoder):
    """Compatibility encoder that flattens a fixed source discretization."""

    model: _AbstractBaseModel
    latent_size: int

    def __init__(self, model: _AbstractBaseModel, latent_size: int, /):
        self.model = model
        self.latent_size = int(latent_size)
        if _get_size(model.out_size) != self.latent_size:
            raise ValueError(
                "Fixed branch model output size must match latent_size; got "
                f"{model.out_size!r} and {self.latent_size}."
            )

    def __call__(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_ndim: int,
        key: EvalKey = None,
    ) -> Array:
        if samples.values is None:
            raise ValueError("Branch source values cannot be None.")
        values = jnp.asarray(samples.values)
        if int(case_ndim) < 0 or int(case_ndim) > values.ndim:
            raise ValueError("Invalid case_ndim for fixed branch values.")
        case_shape = tuple(int(size) for size in values.shape[: int(case_ndim)])
        flat = values.reshape((-1, prod(values.shape[case_ndim:])))
        encoded = jax.vmap(lambda value: self.model(value, key=key))(flat)
        return jnp.asarray(encoded).reshape(case_shape + (self.latent_size,))


class IntegralBranchEncoder(_AbstractBranchEncoder):
    """Permutation-invariant, quadrature-aware variable-sensor branch encoder."""

    feature_model: _AbstractBaseModel
    mixer: _AbstractBaseModel | None
    latent_size: int
    value_channels: int
    coord_dim: int
    normalize: bool

    def __init__(
        self,
        *,
        feature_model: _AbstractBaseModel,
        latent_size: int,
        value_channels: int | Literal["scalar"] = "scalar",
        coord_dim: int,
        mixer: _AbstractBaseModel | None = None,
        normalize: bool = False,
    ):
        self.feature_model = feature_model
        self.mixer = mixer
        self.latent_size = int(latent_size)
        self.value_channels = _get_size(value_channels)
        self.coord_dim = int(coord_dim)
        self.normalize = bool(normalize)
        if self.latent_size <= 0 or self.coord_dim <= 0:
            raise ValueError("latent_size and coord_dim must be positive.")
        expected_input = self.value_channels + self.coord_dim
        if _get_size(feature_model.in_size) != expected_input:
            raise ValueError(
                f"feature_model.in_size must be {expected_input}; got "
                f"{feature_model.in_size!r}."
            )
        if _get_size(feature_model.out_size) != self.latent_size:
            raise ValueError("feature_model.out_size must match latent_size.")
        if mixer is not None and (
            _get_size(mixer.in_size) != self.latent_size
            or _get_size(mixer.out_size) != self.latent_size
        ):
            raise ValueError("Integral branch mixer must map latent_size to latent_size.")

    def __call__(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_ndim: int,
        key: EvalKey = None,
    ) -> Array:
        if samples.values is None:
            raise ValueError("Branch source values cannot be None.")
        values = jnp.asarray(samples.values)
        sample_shape = samples.sample_shape
        if not sample_shape:
            raise ValueError("IntegralBranchEncoder requires a sampled source geometry.")
        sample_ndim = len(sample_shape)
        if tuple(values.shape[case_ndim : case_ndim + sample_ndim]) != sample_shape:
            raise ValueError(
                "Branch values do not contain the source sample shape after case axes."
            )
        case_shape = tuple(int(size) for size in values.shape[:case_ndim])
        trailing = values.shape[case_ndim + sample_ndim :]
        if not trailing:
            values = values[..., None]
        elif tuple(int(size) for size in trailing) != (self.value_channels,):
            raise ValueError(
                f"Expected {self.value_channels} value channels, got trailing shape "
                f"{trailing}."
            )

        coordinates = _sample_coordinates(samples, case_shape)
        features = jnp.concatenate((values, coordinates), axis=-1)
        flattened = features.reshape((-1, self.value_channels + self.coord_dim))
        point_features = jax.vmap(lambda feature: self.feature_model(feature, key=key))(
            flattened
        )
        point_features = jnp.asarray(point_features).reshape(
            case_shape + sample_shape + (self.latent_size,)
        )

        weights = samples.weights(
            normalized=self.normalize,
            case_shape=case_shape,
        )
        weighted = point_features * weights[..., None]
        sample_axes = tuple(range(len(case_shape), len(case_shape) + len(sample_shape)))
        encoded = jnp.sum(weighted, axis=sample_axes)
        mixer = self.mixer
        if mixer is None:
            return encoded
        flat_cases = encoded.reshape((-1, self.latent_size))
        mixed = jax.vmap(lambda value: mixer(value, key=key))(flat_cases)
        return jnp.asarray(mixed).reshape(case_shape + (self.latent_size,))


class PODBasis(StrictModule, NonTrainableState):
    """Fixed affine reduced output decoder for POD-DeepONet.

    ``offset`` is the fitted spatial mean, not the output-channel bias.  Query
    layout metadata is retained so a basis cannot silently move between
    incompatible tensor-product or point-cloud layouts.
    """

    values: Array
    offset: Array
    latent_size: int
    out_size: int | Literal["scalar"]
    has_offset: bool
    query_axis_names: tuple[str, ...]
    query_axis_bases: tuple[str, ...]
    query_axis_periodic: tuple[bool, ...]
    query_coordinate_dimension: int | None
    query_axis_nodes: tuple[Array, ...]
    query_coordinates: Array | None
    query_weights: Array | None
    geometry_fingerprint: str | None

    def __init__(
        self,
        values: Array,
        /,
        *,
        latent_size: int,
        out_size: int | Literal["scalar"] = "scalar",
        offset: Array | None = None,
        query_layout: FunctionSamples | None = None,
        geometry_fingerprint: str | None = None,
    ):
        basis = jnp.asarray(values)
        self.latent_size = int(latent_size)
        self.out_size = out_size
        out_count = _get_size(out_size)
        if (
            out_size == "scalar"
            and basis.shape[-2:] != (1, self.latent_size)
            and basis.shape[-1:] == (self.latent_size,)
        ):
            basis = basis[..., None, :]
        if basis.shape[-2:] != (out_count, self.latent_size):
            raise ValueError(
                f"POD basis must end in (out_size, latent_size); got {basis.shape}."
            )
        if offset is None:
            offset_ = jnp.zeros(basis.shape[:-1], dtype=basis.dtype)
            has_offset = False
        else:
            offset_ = jnp.asarray(offset, dtype=basis.dtype)
            if out_size == "scalar" and offset_.shape == basis.shape[:-2]:
                offset_ = offset_[..., None]
            if offset_.shape != basis.shape[:-1]:
                raise ValueError(
                    "POD spatial mean must have sample_shape + (out_size,); got "
                    f"{offset_.shape} for basis {basis.shape}."
                )
            has_offset = True
        if query_layout is not None and query_layout.sample_shape != basis.shape[:-2]:
            raise ValueError(
                "POD query layout sample shape must match the basis; got "
                f"{query_layout.sample_shape} and {basis.shape[:-2]}."
            )
        if query_layout is not None and query_layout.geometry_case_shape:
            raise ValueError(
                "POD query layouts must be shared rather than case-dependent."
            )
        self.values = basis
        self.offset = offset_
        self.has_offset = has_offset
        self.query_axis_names = (
            () if query_layout is None else tuple(axis.name for axis in query_layout.axes)
        )
        self.query_axis_bases = (
            ()
            if query_layout is None
            else tuple(axis.basis for axis in query_layout.axes)
        )
        self.query_axis_periodic = (
            ()
            if query_layout is None
            else tuple(axis.periodic for axis in query_layout.axes)
        )
        self.query_coordinate_dimension = (
            None
            if query_layout is None or query_layout.coordinates is None
            else int(query_layout.coordinates.shape[-1])
        )
        self.query_axis_nodes = (
            ()
            if query_layout is None
            else tuple(axis.nodes for axis in query_layout.axes)
        )
        self.query_coordinates = (
            None if query_layout is None else query_layout.coordinates
        )
        self.query_weights = (
            None if query_layout is None else query_layout.weights(case_shape=())
        )
        self.geometry_fingerprint = (
            query_layout.geometry_fingerprint()
            if query_layout is not None and geometry_fingerprint is None
            else geometry_fingerprint
        )

    def validate_query_layout(self, query: FunctionSamples, /) -> None:
        if self.values.shape[:-2] != query.sample_shape:
            raise ValueError(
                "POD basis sample shape must match the query sample shape; got "
                f"{self.values.shape[:-2]} and {query.sample_shape}."
            )
        if self.query_axis_names and (
            tuple(axis.name for axis in query.axes) != self.query_axis_names
            or tuple(axis.basis for axis in query.axes) != self.query_axis_bases
            or tuple(axis.periodic for axis in query.axes) != self.query_axis_periodic
        ):
            raise ValueError("POD basis query axis layout does not match its fit layout.")
        if self.query_coordinate_dimension is not None and (
            query.coordinates is None
            or int(query.coordinates.shape[-1]) != self.query_coordinate_dimension
        ):
            raise ValueError(
                "POD basis point-cloud coordinate dimension does not match its fit layout."
            )
        if (
            self.query_axis_nodes
            or self.query_coordinates is not None
            or self.query_weights is not None
        ) and query.geometry_case_shape:
            raise ValueError(
                "POD basis evaluation requires one shared fixed query layout."
            )

    def _validated_query_value(
        self,
        value: Array,
        query: FunctionSamples,
        /,
    ) -> Array:
        validated = value
        if self.query_axis_nodes:
            for fitted, current in zip(
                self.query_axis_nodes, (axis.nodes for axis in query.axes), strict=True
            ):
                if fitted.shape != current.shape:
                    raise ValueError(
                        "POD query axis node shapes do not match the fit layout."
                    )
                validated = eqx.error_if(
                    validated,
                    jnp.any(fitted != current),
                    "POD query axis nodes do not match the fit layout.",
                )
        if self.query_coordinates is not None:
            assert query.coordinates is not None
            if self.query_coordinates.shape != query.coordinates.shape:
                raise ValueError(
                    "POD point coordinates do not have the fitted query shape."
                )
            validated = eqx.error_if(
                validated,
                jnp.any(self.query_coordinates != query.coordinates),
                "POD point coordinates do not match the fit layout.",
            )
        if self.query_weights is not None:
            current_weights = query.weights(case_shape=())
            if self.query_weights.shape != current_weights.shape:
                raise ValueError("POD query weights do not have the fitted layout shape.")
            validated = eqx.error_if(
                validated,
                jnp.any(self.query_weights != current_weights),
                "POD query quadrature or mask does not match the fit layout.",
            )
        return validated

    def evaluate(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        self.validate_query_layout(query)
        values = self._validated_query_value(self.values, query)
        return jnp.broadcast_to(
            values,
            case_shape + values.shape,
        )

    def evaluate_offset(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        self.validate_query_layout(query)
        offset = self._validated_query_value(self.offset, query)
        return jnp.broadcast_to(offset, case_shape + offset.shape)


def _deeponet_contract(model):
    from phydrax.nn.operator.catalog import operator_architecture_contract

    contract = operator_architecture_contract(model.operator_architecture)
    return replace(
        contract,
        capabilities=replace(
            contract.capabilities,
            requires_fixed_query=isinstance(model.trunk, PODBasis),
        ),
    )


class DeepONet(AbstractOperatorModel):
    """General branch-trunk neural operator with variable sensors and POD decoding.

    A mapping of branch encoders enables MIONet-style multiple functional inputs.
    ``product`` is the canonical MIONet fusion; ``sum`` and learned ``concat`` are
    available without creating separate paper-specific model stacks.
    """

    operator_architecture = "DeepONet"
    _operator_contract_builder = staticmethod(_deeponet_contract)

    branch: _AbstractBaseModel | None
    branches: frozendict[str, _AbstractBranchEncoder]
    branch_mixer: _AbstractBaseModel | None
    fusion: BranchFusion
    trunk: _AbstractBaseModel | PODBasis
    latent_size: int
    coord_dim: int
    out_size: int | Literal["scalar"]
    in_size: int | Literal["scalar"]
    source_key: str | None
    query_chunk_size: int | None
    bias: Array

    def __init__(
        self,
        *,
        branch: _AbstractBaseModel
        | _AbstractBranchEncoder
        | Mapping[str, _AbstractBaseModel | _AbstractBranchEncoder],
        trunk: _AbstractBaseModel | PODBasis,
        coord_dim: int,
        latent_size: int,
        out_size: int | Literal["scalar"] = "scalar",
        in_size: int | Literal["scalar"] = "scalar",
        fusion: BranchFusion = "sum",
        branch_mixer: _AbstractBaseModel | None = None,
        source_key: str | None = None,
        query_chunk_size: int | None = None,
    ):
        self.coord_dim = int(coord_dim)
        self.latent_size = int(latent_size)
        self.out_size = out_size
        self.in_size = in_size
        self.fusion = fusion
        self.branch_mixer = branch_mixer
        self.source_key = source_key
        self.query_chunk_size = (
            None if query_chunk_size is None else int(query_chunk_size)
        )
        self.trunk = trunk
        self.bias = jnp.zeros((_get_size(out_size),), dtype=float)
        if self.coord_dim <= 0 or self.latent_size <= 0:
            raise ValueError("coord_dim and latent_size must be positive.")
        if self.query_chunk_size is not None and self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive.")
        if fusion not in ("sum", "product", "concat"):
            raise ValueError("fusion must be 'sum', 'product', or 'concat'.")

        if isinstance(branch, Mapping):
            branch_items = tuple((str(name), encoder) for name, encoder in branch.items())
            self.branch = None
        else:
            branch_items = ((source_key or "input", branch),)
            self.branch = branch if isinstance(branch, _AbstractBaseModel) else None
        if not branch_items:
            raise ValueError("DeepONet requires at least one branch encoder.")
        encoders: dict[str, _AbstractBranchEncoder] = {}
        for name, encoder in branch_items:
            if isinstance(encoder, _AbstractBranchEncoder):
                if encoder.latent_size != self.latent_size:
                    raise ValueError(
                        f"Branch {name!r} latent size does not match DeepONet."
                    )
                encoders[name] = encoder
            elif isinstance(encoder, _AbstractBaseModel):
                encoders[name] = FixedBranchEncoder(encoder, self.latent_size)
            else:
                raise TypeError(f"Unsupported branch encoder for {name!r}.")
        self.branches = frozendict(encoders)

        if fusion == "concat":
            if branch_mixer is None:
                raise ValueError("concat fusion requires branch_mixer.")
            expected = len(self.branches) * self.latent_size
            if _get_size(branch_mixer.in_size) != expected:
                raise ValueError(f"concat branch_mixer.in_size must be {expected}.")
            if _get_size(branch_mixer.out_size) != self.latent_size:
                raise ValueError("branch_mixer.out_size must match latent_size.")
        elif branch_mixer is not None:
            raise ValueError("branch_mixer is only used by concat fusion.")

        if isinstance(trunk, PODBasis):
            if trunk.latent_size != self.latent_size or trunk.out_size != self.out_size:
                raise ValueError("POD basis sizes must match the DeepONet sizes.")
        else:
            expected_trunk_out = self.latent_size * _get_size(self.out_size)
            if _get_size(trunk.out_size) != expected_trunk_out:
                raise ValueError(
                    "trunk.out_size must be latent_size*out_size; got "
                    f"{trunk.out_size!r} but expected {expected_trunk_out}."
                )
            if _get_size(trunk.in_size) != self.coord_dim:
                raise ValueError(
                    "trunk.in_size must match coord_dim; got "
                    f"{trunk.in_size!r} and {self.coord_dim}."
                )

    def _source_for_branch(
        self,
        batch: OperatorBatch,
        name: str,
        /,
    ) -> FunctionSamples:
        if name in batch.inputs:
            return batch.input(name)
        if len(self.branches) == 1 and len(batch.inputs) == 1:
            return next(iter(batch.inputs.values()))
        raise KeyError(f"DeepONet branch {name!r} has no matching OperatorBatch input.")

    def _encode(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        keys = split_eval_key(key, len(self.branches) + 1)
        encoded = []
        for index, (name, encoder) in enumerate(self.branches.items()):
            source = self._source_for_branch(batch, name)
            encoded.append(
                encoder(
                    source,
                    case_ndim=len(batch.case_axes),
                    key=keys[index],
                )
            )
        if self.fusion == "sum":
            result = encoded[0]
            for value in encoded[1:]:
                result = result + value
            return result / jnp.sqrt(float(len(encoded)))
        if self.fusion == "product":
            result = encoded[0]
            for value in encoded[1:]:
                result = result * value
            return result

        branch_mixer = self.branch_mixer
        assert branch_mixer is not None
        concatenated = jnp.concatenate(encoded, axis=-1)
        shape = concatenated.shape[:-1]
        flat = concatenated.reshape((-1, concatenated.shape[-1]))
        mixed = jax.vmap(lambda value: branch_mixer(value, key=keys[-1]))(flat)
        return jnp.asarray(mixed).reshape(shape + (self.latent_size,))

    def _trunk_basis(
        self,
        query: FunctionSamples,
        case_shape: tuple[int, ...],
        *,
        key: EvalKey,
    ) -> Array:
        trunk = self.trunk
        if isinstance(trunk, PODBasis):
            return trunk.evaluate(query, case_shape=case_shape)
        coordinates = _query_coordinates(query, self.coord_dim, case_shape)
        flat = coordinates.reshape((-1, self.coord_dim))
        if self.query_chunk_size is None:
            evaluated = jax.vmap(lambda point: trunk(point, key=key))(flat)
        else:
            chunks = []
            for start in range(0, int(flat.shape[0]), self.query_chunk_size):
                chunk = flat[start : start + self.query_chunk_size]
                chunks.append(jax.vmap(lambda point: trunk(point, key=key))(chunk))
            evaluated = jnp.concatenate(chunks, axis=0)
        return jnp.asarray(evaluated).reshape(
            case_shape + query.sample_shape + (_get_size(self.out_size), self.latent_size)
        )

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        branch_key, trunk_key = split_eval_key(key, 2)
        coefficients = self._encode(batch, key=branch_key)
        basis = self._trunk_basis(
            batch.require_single_query(),
            batch.case_shape,
            key=trunk_key,
        )
        coefficient_shape = (
            batch.case_shape
            + (1,) * len(batch.require_single_query().sample_shape)
            + (1, self.latent_size)
        )
        output = jnp.sum(
            basis * coefficients.reshape(coefficient_shape),
            axis=-1,
        )
        if isinstance(self.trunk, PODBasis) and self.trunk.has_offset:
            output = output + jnp.broadcast_to(
                self.trunk.offset,
                batch.case_shape + self.trunk.offset.shape,
            )
        output = output + self.bias
        output = (
            output
            * batch.require_single_query().mask_array(case_shape=batch.case_shape)[
                ..., None
            ]
        )
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x, key=key)
        if not isinstance(x, tuple) or len(x) < 2:
            raise ValueError("DeepONet requires a tuple input (branch_input, coords...).")
        if len(self.branches) != 1:
            raise ValueError(
                "Multi-input DeepONet evaluation requires an OperatorBatch keyed by branch."
            )

        coordinates = tuple(jnp.asarray(value) for value in x[1:])
        point_shape: tuple[int, ...] | None = None
        if (
            len(coordinates) == 1
            and coordinates[0].ndim >= 2
            and int(coordinates[0].shape[-1]) == self.coord_dim
        ):
            points = coordinates[0]
            point_shape = tuple(int(size) for size in points.shape[:-1])
            query = FunctionSamples(
                values=None,
                coordinates=points.reshape((-1, self.coord_dim)),
            )
        elif len(coordinates) == self.coord_dim:
            axes = tuple(
                OperatorAxis(f"axis_{index}", nodes)
                for index, nodes in enumerate(coordinates)
            )
            query = FunctionSamples(values=None, axes=axes)
        else:
            raise ValueError(
                "DeepONet coordinates must be coord_dim separate 1D axes or one "
                "point array with trailing coordinate dimension."
            )
        name = next(iter(self.branches))
        batch = OperatorBatch(
            inputs={name: FunctionSamples(values=jnp.asarray(x[0]))},
            queries={"query": query},
        )
        output = self.__call_operator_batch__(batch, key=key)
        if point_shape is not None:
            if self.out_size == "scalar":
                return output.reshape(point_shape)
            return output.reshape(point_shape + (_get_size(self.out_size),))
        return output


__all__ = [
    "BranchFusion",
    "DeepONet",
    "FixedBranchEncoder",
    "IntegralBranchEncoder",
    "PODBasis",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, PyTree

from .._array_tree import ArrayPyTreeSchema
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from .._strict import StrictModule
from ._layout import StateLayout


PlantVectorRole: TypeAlias = Literal["local", "tangent", "local_cotangent", "cotangent"]
PlantModeRole: TypeAlias = Literal[
    "point", "local", "tangent", "local_cotangent", "cotangent"
]
_PLANT_VECTOR_ROLES = ("local", "tangent", "local_cotangent", "cotangent")
_PLANT_MODE_ROLES = ("point", *_PLANT_VECTOR_ROLES)
_DISCRETE_KINDS = frozenset("biu")
_INEXACT_KINDS = frozenset("fc")


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _array(value: Any, name: str, /) -> Array:
    try:
        result = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an array.") from error
    if np.dtype(result.dtype).kind not in frozenset("biufc"):
        raise TypeError(f"{name} must have a numeric or boolean dtype.")
    return result


def _error_if(value: Array, predicate: Array, message: str, /) -> Array:
    if isinstance(predicate, jax_core.Tracer):
        return eqx.error_if(value, predicate, message)
    if bool(predicate):
        raise eqx.EquinoxRuntimeError(message)
    return value


def _identity_objects(
    semantic_provenance: SemanticProvenance,
    numeric_revision: NumericRevision,
    executable_signature: ExecutableSignature,
    /,
) -> None:
    if not isinstance(semantic_provenance, SemanticProvenance):
        raise TypeError("semantic_provenance must be a SemanticProvenance.")
    if not isinstance(numeric_revision, NumericRevision):
        raise TypeError("numeric_revision must be a NumericRevision.")
    if not isinstance(executable_signature, ExecutableSignature):
        raise TypeError("executable_signature must be an ExecutableSignature.")
    if numeric_revision.semantic_id != semantic_provenance.semantic_id:
        raise ValueError(
            "NumericRevision semantic identity does not match SemanticProvenance."
        )


def _common_inexact_dtype(
    schema: ArrayPyTreeSchema,
    indices: tuple[int, ...],
    owner: str,
    /,
) -> np.dtype:
    if not indices:
        raise ValueError(f"{owner} must encode at least one dynamic array leaf.")
    dtypes = tuple(schema.leaves[index].dtype for index in indices)
    non_inexact = tuple(
        schema.leaves[index].path
        for index in indices
        if schema.leaves[index].dtype.kind not in _INEXACT_KINDS
    )
    if non_inexact:
        joined = ", ".join(non_inexact)
        raise TypeError(
            f"{owner} cannot losslessly vectorize dynamic integer or boolean leaves: "
            f"{joined}. Declare immutable plant modes explicitly."
        )
    dtype = dtypes[0]
    if any(candidate != dtype for candidate in dtypes[1:]):
        raise TypeError(
            f"{owner} cannot losslessly vectorize dynamic leaves with unequal dtypes."
        )
    return dtype


def _space_dtype(space: Any, owner: str, /) -> np.dtype:
    structure = space.structure()
    if not isinstance(structure, jax.ShapeDtypeStruct):
        raise TypeError(
            f"{owner} must be one array-valued space for an AbstractStateGeometry."
        )
    return np.dtype(structure.dtype)


def _coordinate_array(
    value: Any,
    size: int,
    dtype: np.dtype,
    owner: str,
    /,
) -> Array:
    coordinates = _array(value, owner)
    if coordinates.shape != (size,):
        raise ValueError(f"{owner} must have shape {(size,)}; got {coordinates.shape}.")
    if np.dtype(coordinates.dtype) != dtype:
        raise TypeError(
            f"{owner} dtype {coordinates.dtype} does not match codec dtype {dtype}."
        )
    return coordinates


def _schema_vector(
    schema: ArrayPyTreeSchema,
    tree: PyTree[Any],
    indices: tuple[int, ...],
    dtype: np.dtype,
    owner: str,
    /,
) -> Array:
    case_shape = schema.validate(tree)
    leaves = schema.flatten(tree)
    vectors = [
        jnp.reshape(
            leaves[index],
            case_shape + (prod(schema.leaves[index].shape),),
        )
        for index in indices
    ]
    result = vectors[0] if len(vectors) == 1 else jnp.concatenate(vectors, axis=-1)
    if np.dtype(result.dtype) != dtype:
        raise TypeError(f"{owner} packing changed the declared dynamic dtype.")
    return result


def _binding_payload(
    semantic_provenance: SemanticProvenance,
    numeric_revision: NumericRevision,
    schema: ArrayPyTreeSchema,
    executable_signature: ExecutableSignature,
    /,
) -> dict[str, str]:
    return {
        "semantic_id": semantic_provenance.semantic_id,
        "numeric_revision_id": numeric_revision.revision_id,
        "schema_id": schema.schema_id,
        "executable_signature_id": executable_signature.signature_id,
    }


def _mode_schema_identity(
    schema_id: str,
    leaves: Sequence[Any],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "plant-dynamic-mode-schema",
            "state_schema": schema_id,
            "leaves": [
                {
                    "path": leaf.path,
                    "shape": list(leaf.shape),
                    "dtype": leaf.dtype.str,
                }
                for leaf in leaves
            ],
        }
    )


def _mode_identity(
    paths: tuple[str, ...],
    mode_schema_id: str,
    role: PlantModeRole,
    binding: dict[str, str],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "plant-fixed-dynamic-mode",
            "paths": list(paths),
            "mode_schema": mode_schema_id,
            "role": role,
            **binding,
        }
    )


class PlantModeSidecar(StrictModule):
    """Exact dynamic discrete leaves and fixed-mode identity evidence."""

    values: tuple[Array, ...]
    identity_values: tuple[Array, ...]
    paths: tuple[str, ...] = eqx.field(static=True)
    mode_schema_id: str = eqx.field(static=True)
    role: PlantModeRole = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    executable_signature_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: Sequence[Any],
        /,
        *,
        paths: Sequence[str],
        mode_schema_id: str,
        role: PlantModeRole,
        semantic_id: str,
        numeric_revision_id: str,
        schema_id: str,
        executable_signature_id: str,
        codec_id: str,
    ):
        paths_ = tuple(str(path) for path in paths)
        if len(set(paths_)) != len(paths_) or any(not path for path in paths_):
            raise ValueError("Dynamic mode sidecar paths must be unique and non-empty.")
        values_ = tuple(
            _array(value, f"Dynamic mode sidecar leaf {path}")
            for path, value in zip(paths_, tuple(values), strict=True)
        )
        if any(np.dtype(value.dtype).kind not in _DISCRETE_KINDS for value in values_):
            raise TypeError(
                "Dynamic mode sidecar values must have boolean or integer dtypes."
            )
        if role not in _PLANT_MODE_ROLES:
            raise ValueError("Unknown dynamic mode sidecar role.")
        binding = {
            "semantic_id": _identifier(semantic_id, "semantic_id"),
            "numeric_revision_id": _identifier(
                numeric_revision_id, "numeric_revision_id"
            ),
            "schema_id": _identifier(schema_id, "schema_id"),
            "executable_signature_id": _identifier(
                executable_signature_id, "executable_signature_id"
            ),
            "codec_id": _identifier(codec_id, "codec_id"),
        }
        self.values = values_
        self.identity_values = tuple(jax.lax.stop_gradient(value) for value in values_)
        self.paths = paths_
        self.mode_schema_id = _identifier(mode_schema_id, "mode_schema_id")
        self.role = role
        self.semantic_id = binding["semantic_id"]
        self.numeric_revision_id = binding["numeric_revision_id"]
        self.schema_id = binding["schema_id"]
        self.executable_signature_id = binding["executable_signature_id"]
        self.codec_id = binding["codec_id"]
        self.mode_id = _mode_identity(paths_, self.mode_schema_id, role, binding)


def _empty_mode_sidecar(
    role: PlantModeRole,
    binding: dict[str, str],
    /,
) -> PlantModeSidecar:
    return PlantModeSidecar(
        (),
        paths=(),
        mode_schema_id=_mode_schema_identity(binding["schema_id"], ()),
        role=role,
        **binding,
    )


class EncodedPlantState(StrictModule):
    """Point coordinates with complete provenance and exact dynamic modes."""

    vector: Array
    mode: PlantModeSidecar
    semantic_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    executable_signature_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector: Any,
        /,
        *,
        semantic_id: str,
        numeric_revision_id: str,
        schema_id: str,
        executable_signature_id: str,
        codec_id: str,
        mode_sidecar: PlantModeSidecar | None = None,
    ):
        binding = {
            "semantic_id": _identifier(semantic_id, "semantic_id"),
            "numeric_revision_id": _identifier(
                numeric_revision_id, "numeric_revision_id"
            ),
            "schema_id": _identifier(schema_id, "schema_id"),
            "executable_signature_id": _identifier(
                executable_signature_id, "executable_signature_id"
            ),
            "codec_id": _identifier(codec_id, "codec_id"),
        }
        if mode_sidecar is not None and not isinstance(mode_sidecar, PlantModeSidecar):
            raise TypeError("mode_sidecar must be a PlantModeSidecar or None.")
        self.vector = _array(vector, "Encoded plant point")
        self.mode = (
            _empty_mode_sidecar("point", binding)
            if mode_sidecar is None
            else mode_sidecar
        )
        self.semantic_id = binding["semantic_id"]
        self.numeric_revision_id = binding["numeric_revision_id"]
        self.schema_id = binding["schema_id"]
        self.executable_signature_id = binding["executable_signature_id"]
        self.codec_id = binding["codec_id"]

    @property
    def mode_sidecar(self) -> PlantModeSidecar:
        return self.mode

    @property
    def mode_paths(self) -> tuple[str, ...]:
        return self.mode.paths

    @property
    def mode_values(self) -> tuple[Array, ...]:
        return self.mode.values

    @property
    def mode_schema_id(self) -> str:
        return self.mode.mode_schema_id

    @property
    def mode_role(self) -> PlantModeRole:
        return self.mode.role

    @property
    def mode_identity(self) -> str:
        return self.mode.mode_id


class EncodedPlantVector(StrictModule):
    """Coordinates with provenance and explicit fixed-mode evidence."""

    vector: Array
    mode: PlantModeSidecar
    role: PlantVectorRole = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    executable_signature_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector: Any,
        role: PlantVectorRole,
        /,
        *,
        semantic_id: str,
        numeric_revision_id: str,
        schema_id: str,
        executable_signature_id: str,
        codec_id: str,
        mode_sidecar: PlantModeSidecar | None = None,
    ):
        if role not in _PLANT_VECTOR_ROLES:
            raise ValueError("Unknown encoded plant vector role.")
        binding = {
            "semantic_id": _identifier(semantic_id, "semantic_id"),
            "numeric_revision_id": _identifier(
                numeric_revision_id, "numeric_revision_id"
            ),
            "schema_id": _identifier(schema_id, "schema_id"),
            "executable_signature_id": _identifier(
                executable_signature_id, "executable_signature_id"
            ),
            "codec_id": _identifier(codec_id, "codec_id"),
        }
        if mode_sidecar is not None and not isinstance(mode_sidecar, PlantModeSidecar):
            raise TypeError("mode_sidecar must be a PlantModeSidecar or None.")
        self.vector = _array(vector, f"Encoded plant {role}")
        self.mode = (
            _empty_mode_sidecar(role, binding) if mode_sidecar is None else mode_sidecar
        )
        self.role = role
        self.semantic_id = binding["semantic_id"]
        self.numeric_revision_id = binding["numeric_revision_id"]
        self.schema_id = binding["schema_id"]
        self.executable_signature_id = binding["executable_signature_id"]
        self.codec_id = binding["codec_id"]

    @property
    def mode_sidecar(self) -> PlantModeSidecar:
        return self.mode

    @property
    def fixed_mode(self) -> PlantModeSidecar:
        return self.mode

    @property
    def mode_paths(self) -> tuple[str, ...]:
        return self.mode.paths

    @property
    def mode_values(self) -> tuple[Array, ...]:
        return self.mode.values

    @property
    def mode_schema_id(self) -> str:
        return self.mode.mode_schema_id

    @property
    def mode_role(self) -> PlantModeRole:
        return self.mode.role

    @property
    def mode_identity(self) -> str:
        return self.mode.mode_id


class EncodedControl(StrictModule):
    """One exact command vector with complete control-codec provenance."""

    vector: Array
    semantic_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    executable_signature_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector: Any,
        /,
        *,
        semantic_id: str,
        numeric_revision_id: str,
        schema_id: str,
        executable_signature_id: str,
        codec_id: str,
    ):
        self.vector = _array(vector, "Encoded control")
        self.semantic_id = _identifier(semantic_id, "semantic_id")
        self.numeric_revision_id = _identifier(numeric_revision_id, "numeric_revision_id")
        self.schema_id = _identifier(schema_id, "schema_id")
        self.executable_signature_id = _identifier(
            executable_signature_id, "executable_signature_id"
        )
        self.codec_id = _identifier(codec_id, "codec_id")


class PlantPowerEvidence(StrictModule):
    """Algebraic duality evidence for a retraction JVP and its exact VJP."""

    tangent: EncodedPlantVector
    local_cotangent: EncodedPlantVector
    physical_power: Array
    local_power: Array
    absolute_residual: Array
    scale: Array
    tolerance: Array
    finite: Array
    valid: Array
    codec_id: str = eqx.field(static=True)


class PlantStateVectorCodec(StrictModule):
    """Lossless complete-state bridge to one four-space state geometry.

    Continuous non-immutable leaves occupy point coordinates. Boolean and
    integer non-immutable leaves occupy an exact sidecar carried by every point
    and differential vector. Declared immutable leaves are checked against and
    restored from the bound template.
    """

    schema: ArrayPyTreeSchema
    layout: StateLayout
    template: Any
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    executable_signature: ExecutableSignature
    immutable_mode_paths: tuple[str, ...] = eqx.field(static=True)
    dynamic_leaf_indices: tuple[int, ...] = eqx.field(static=True)
    vector_leaf_indices: tuple[int, ...] = eqx.field(static=True)
    dynamic_mode_leaf_indices: tuple[int, ...] = eqx.field(static=True)
    immutable_leaf_indices: tuple[int, ...] = eqx.field(static=True)
    dynamic_mode_paths: tuple[str, ...] = eqx.field(static=True)
    dynamic_mode_schema_id: str = eqx.field(static=True)
    point_dtype: np.dtype = eqx.field(static=True)
    local_dtype: np.dtype = eqx.field(static=True)
    tangent_dtype: np.dtype = eqx.field(static=True)
    immutable_mode_fingerprint: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ArrayPyTreeSchema,
        layout: StateLayout,
        template: PyTree[Any],
        immutable_mode_paths: Sequence[str] = (),
        /,
        *,
        semantic_provenance: SemanticProvenance,
        numeric_revision: NumericRevision,
        executable_signature: ExecutableSignature,
    ):
        if not isinstance(schema, ArrayPyTreeSchema):
            raise TypeError("schema must be an ArrayPyTreeSchema.")
        if not isinstance(layout, StateLayout):
            raise TypeError("layout must be a StateLayout.")
        _identity_objects(semantic_provenance, numeric_revision, executable_signature)
        if schema.case_ndim != 0:
            raise ValueError("Plant state codecs require an unbatched state schema.")
        case_shape = schema.validate(template)
        if case_shape:
            raise ValueError("Plant state codec templates must be unbatched.")

        requested_paths = tuple(str(path) for path in immutable_mode_paths)
        if len(set(requested_paths)) != len(requested_paths):
            raise ValueError("immutable_mode_paths must not contain duplicates.")
        unknown = tuple(path for path in requested_paths if path not in schema.leaf_paths)
        if unknown:
            raise ValueError(
                "immutable_mode_paths contains paths absent from the state schema: "
                + ", ".join(unknown)
            )
        requested_set = set(requested_paths)
        paths = tuple(path for path in schema.leaf_paths if path in requested_set)
        immutable = tuple(
            index
            for index, leaf in enumerate(schema.leaves)
            if leaf.path in requested_set
        )
        dynamic = tuple(
            index for index in range(len(schema.leaves)) if index not in immutable
        )
        vector = tuple(
            index
            for index in dynamic
            if schema.leaves[index].dtype.kind in _INEXACT_KINDS
        )
        dynamic_modes = tuple(
            index
            for index in dynamic
            if schema.leaves[index].dtype.kind in _DISCRETE_KINDS
        )
        point_dtype = _common_inexact_dtype(schema, vector, "Plant state codec")
        dynamic_size = sum(prod(schema.leaves[index].shape) for index in vector)
        if dynamic_size != layout.size:
            raise ValueError(
                "Every dynamic inexact state leaf must be represented in point "
                f"storage; schema requires {dynamic_size} values but StateLayout "
                f"has {layout.size}."
            )
        local_dtype = _space_dtype(layout.local_space, "StateLayout local space")
        tangent_dtype = _space_dtype(layout.tangent_space, "StateLayout tangent space")
        if local_dtype != point_dtype or tangent_dtype != point_dtype:
            raise TypeError(
                "Point, local, and tangent spaces must use one lossless coordinate dtype."
            )

        template_leaves = schema.flatten(template)
        immutable_tree = tuple(template_leaves[index] for index in immutable)
        if any(isinstance(leaf, jax_core.Tracer) for leaf in immutable_tree):
            raise TypeError("Immutable mode template leaves must be concrete arrays.")
        immutable_fingerprint = array_tree_fingerprint(immutable_tree)["sha256"]
        mode_leaves = tuple(schema.leaves[index] for index in dynamic_modes)
        mode_paths = tuple(leaf.path for leaf in mode_leaves)
        mode_schema_id = _mode_schema_identity(schema.schema_id, mode_leaves)
        payload = _binding_payload(
            semantic_provenance, numeric_revision, schema, executable_signature
        )
        codec_id = canonical_fingerprint(
            {
                "kind": "plant-state-vector-codec",
                **payload,
                "schema_content_id": schema.content_id,
                "layout_id": layout.layout_id,
                "immutable_mode_paths": list(paths),
                "immutable_mode_fingerprint": immutable_fingerprint,
                "dynamic_mode_paths": list(mode_paths),
                "dynamic_mode_schema_id": mode_schema_id,
            }
        )

        self.schema = schema
        self.layout = layout
        self.template = template
        self.semantic_provenance = semantic_provenance
        self.numeric_revision = numeric_revision
        self.executable_signature = executable_signature
        self.immutable_mode_paths = paths
        self.dynamic_leaf_indices = dynamic
        self.vector_leaf_indices = vector
        self.dynamic_mode_leaf_indices = dynamic_modes
        self.immutable_leaf_indices = immutable
        self.dynamic_mode_paths = mode_paths
        self.dynamic_mode_schema_id = mode_schema_id
        self.point_dtype = point_dtype
        self.local_dtype = local_dtype
        self.tangent_dtype = tangent_dtype
        self.immutable_mode_fingerprint = immutable_fingerprint
        self.codec_id = codec_id

    @property
    def semantic_id(self) -> str:
        return self.semantic_provenance.semantic_id

    @property
    def numeric_revision_id(self) -> str:
        return self.numeric_revision.revision_id

    @property
    def schema_id(self) -> str:
        return self.schema.schema_id

    @property
    def executable_signature_id(self) -> str:
        return self.executable_signature.signature_id

    def _binding(self) -> dict[str, str]:
        return {
            **_binding_payload(
                self.semantic_provenance,
                self.numeric_revision,
                self.schema,
                self.executable_signature,
            ),
            "codec_id": self.codec_id,
        }

    def _check_binding(self, encoded: Any, owner: str, /) -> None:
        expected = self._binding()
        if any(getattr(encoded, name, None) != value for name, value in expected.items()):
            raise ValueError(f"{owner} provenance does not match this codec.")

    def _mode_sidecar(
        self,
        values: Sequence[Any],
        role: PlantModeRole,
        /,
    ) -> PlantModeSidecar:
        return PlantModeSidecar(
            values,
            paths=self.dynamic_mode_paths,
            mode_schema_id=self.dynamic_mode_schema_id,
            role=role,
            **self._binding(),
        )

    def _template_mode(self, role: PlantModeRole, /) -> PlantModeSidecar:
        leaves = self.schema.flatten(self.template)
        return self._mode_sidecar(
            tuple(leaves[index] for index in self.dynamic_mode_leaf_indices),
            role,
        )

    def _check_mode(
        self,
        mode: Any,
        role: PlantModeRole,
        subject: Array,
        owner: str,
        /,
    ) -> Array:
        if not isinstance(mode, PlantModeSidecar):
            raise TypeError(f"{owner} must carry a PlantModeSidecar.")
        binding = self._binding()
        expected_static = {
            "paths": self.dynamic_mode_paths,
            "mode_schema_id": self.dynamic_mode_schema_id,
            "role": role,
            **binding,
        }
        if any(
            getattr(mode, name, None) != value for name, value in expected_static.items()
        ):
            raise ValueError(
                f"{owner} dynamic mode sidecar provenance, schema, paths, or role "
                "does not match this codec operation."
            )
        expected_mode_id = _mode_identity(
            self.dynamic_mode_paths,
            self.dynamic_mode_schema_id,
            role,
            binding,
        )
        if mode.mode_id != expected_mode_id:
            raise ValueError(f"{owner} dynamic mode sidecar identity is stale.")
        if len(mode.values) != len(self.dynamic_mode_leaf_indices) or len(
            mode.identity_values
        ) != len(self.dynamic_mode_leaf_indices):
            raise ValueError(f"{owner} dynamic mode sidecar leaf count is stale.")
        values_equal = jnp.asarray(True)
        for value, identity_value, index in zip(
            mode.values,
            mode.identity_values,
            self.dynamic_mode_leaf_indices,
            strict=True,
        ):
            leaf = self.schema.leaves[index]
            for candidate in (value, identity_value):
                if candidate.shape != leaf.shape:
                    raise ValueError(
                        f"{owner} dynamic mode leaf {leaf.path} shape is stale."
                    )
                if np.dtype(candidate.dtype) != leaf.dtype:
                    raise TypeError(
                        f"{owner} dynamic mode leaf {leaf.path} dtype is stale."
                    )
            values_equal = values_equal & jnp.array_equal(value, identity_value)
        return _error_if(
            subject,
            ~values_equal,
            f"{owner} dynamic mode sidecar values do not match its bound identity.",
        )

    def _check_same_mode(
        self,
        left: PlantModeSidecar,
        right: PlantModeSidecar,
        subject: Array,
        owner: str,
        /,
    ) -> Array:
        modes_equal = jnp.asarray(True)
        for left_value, right_value in zip(left.values, right.values, strict=True):
            modes_equal = modes_equal & jnp.array_equal(left_value, right_value)
        return _error_if(
            subject,
            ~modes_equal,
            f"{owner} cannot cross a dynamic plant mode change.",
        )

    def _point(self, value: Any, owner: str, /) -> Array:
        point = _array(value, owner)
        if point.shape != self.layout.shape:
            raise ValueError(
                f"{owner} must have shape {self.layout.shape}; got {point.shape}."
            )
        if np.dtype(point.dtype) != self.point_dtype:
            raise TypeError(
                f"{owner} dtype {point.dtype} does not match {self.point_dtype}."
            )
        return point

    def _wrap_point(
        self,
        point: Any,
        mode: PlantModeSidecar | None = None,
        /,
    ) -> EncodedPlantState:
        coordinates = self._point(point, "Plant point")
        source = self._template_mode("point") if mode is None else mode
        coordinates = self._check_mode(source, "point", coordinates, "Plant point")
        return EncodedPlantState(
            coordinates,
            mode_sidecar=self._mode_sidecar(source.values, "point"),
            **self._binding(),
        )

    def _wrap_vector(
        self,
        vector: Any,
        role: PlantVectorRole,
        mode: PlantModeSidecar | None = None,
        /,
    ) -> EncodedPlantVector:
        space, dtype = self._role_space(role)
        coordinates = _coordinate_array(
            vector, space.size, dtype, f"Plant {role} coordinates"
        )
        if mode is None:
            source = self._template_mode(role)
        else:
            source = mode
            coordinates = self._check_mode(
                source, source.role, coordinates, f"Plant {role} coordinates"
            )
        return EncodedPlantVector(
            coordinates,
            role,
            mode_sidecar=self._mode_sidecar(source.values, role),
            **self._binding(),
        )

    def _role_space(self, role: PlantVectorRole, /) -> tuple[Any, np.dtype]:
        if role == "local":
            return self.layout.local_space, self.local_dtype
        if role == "tangent":
            return self.layout.tangent_space, self.tangent_dtype
        if role == "local_cotangent":
            return self.layout.local_cotangent_space, self.local_dtype
        if role == "cotangent":
            return self.layout.cotangent_space, self.tangent_dtype
        raise ValueError("Unknown encoded plant vector role.")

    def _decode_vector(
        self, encoded: EncodedPlantVector, role: PlantVectorRole, /
    ) -> PyTree[Array]:
        if not isinstance(encoded, EncodedPlantVector):
            raise TypeError(f"Encoded {role} must be an EncodedPlantVector.")
        self._check_binding(encoded, f"Encoded {role}")
        if encoded.role != role:
            raise ValueError(
                f"Encoded plant vector role {encoded.role!r} cannot be used as {role!r}."
            )
        space, dtype = self._role_space(role)
        coordinates = _coordinate_array(
            encoded.vector, space.size, dtype, f"Encoded {role}"
        )
        coordinates = self._check_mode(encoded.mode, role, coordinates, f"Encoded {role}")
        return space.unflatten(coordinates)

    def encode_point(self, state: PyTree[Any], /) -> EncodedPlantState:
        """Encode continuous coordinates and carry dynamic modes exactly."""
        leaves = self.schema.flatten(state)
        point = _schema_vector(
            self.schema,
            state,
            self.vector_leaf_indices,
            self.point_dtype,
            "Plant point",
        ).reshape(self.layout.shape)
        if self.immutable_leaf_indices:
            template_leaves = self.schema.flatten(self.template)
            modes_equal = jnp.asarray(True)
            for index in self.immutable_leaf_indices:
                modes_equal = modes_equal & jnp.array_equal(
                    leaves[index], template_leaves[index]
                )
            point = _error_if(
                point,
                ~modes_equal,
                "Plant payload changed a declared immutable mode leaf.",
            )
        mode = self._mode_sidecar(
            tuple(leaves[index] for index in self.dynamic_mode_leaf_indices),
            "point",
        )
        return EncodedPlantState(
            self._point(point, "Plant point"),
            mode_sidecar=mode,
            **self._binding(),
        )

    def replace_point_vector(
        self,
        anchor: EncodedPlantState,
        vector: Any,
        /,
    ) -> EncodedPlantState:
        """Replace only continuous coordinates under an anchor's fixed modes."""
        self._encoded_point(anchor, "Plant point replacement anchor")
        return self._wrap_point(vector, anchor.mode)

    def decode_point(self, encoded: EncodedPlantState, /) -> PyTree[Array]:
        """Restore one complete payload after validating coordinates and sidecar."""
        point = self._encoded_point(encoded, "Encoded plant point")
        flat = point.reshape((self.layout.size,))
        template_leaves = self.schema.flatten(self.template)
        mode_by_index = dict(
            zip(
                self.dynamic_mode_leaf_indices,
                encoded.mode.values,
                strict=True,
            )
        )
        vector_indices = set(self.vector_leaf_indices)
        leaves: list[Any] = []
        offset = 0
        for index, leaf in enumerate(self.schema.leaves):
            if index in mode_by_index:
                leaves.append(mode_by_index[index])
                continue
            if index not in vector_indices:
                leaves.append(template_leaves[index])
                continue
            size = prod(leaf.shape)
            leaves.append(flat[offset : offset + size].reshape(leaf.shape))
            offset += size
        return self.schema.unflatten(leaves)

    def _anchor_mode(
        self,
        anchor: EncodedPlantState | None,
        owner: str,
        /,
    ) -> PlantModeSidecar | None:
        if anchor is None:
            return None
        self._encoded_point(anchor, owner)
        return anchor.mode

    def encode_local(
        self,
        local: PyTree[Any],
        /,
        *,
        anchor: EncodedPlantState | None = None,
    ) -> EncodedPlantVector:
        return self._wrap_vector(
            self.layout.local_space.flatten(local),
            "local",
            self._anchor_mode(anchor, "Local-vector mode anchor"),
        )

    def decode_local(self, encoded: EncodedPlantVector, /) -> PyTree[Array]:
        return self._decode_vector(encoded, "local")

    def encode_tangent(
        self,
        tangent: PyTree[Any],
        /,
        *,
        anchor: EncodedPlantState | None = None,
    ) -> EncodedPlantVector:
        return self._wrap_vector(
            self.layout.tangent_space.flatten(tangent),
            "tangent",
            self._anchor_mode(anchor, "Tangent-vector mode anchor"),
        )

    def decode_tangent(self, encoded: EncodedPlantVector, /) -> PyTree[Array]:
        return self._decode_vector(encoded, "tangent")

    def encode_local_cotangent(
        self,
        cotangent: PyTree[Any],
        /,
        *,
        anchor: EncodedPlantState | None = None,
    ) -> EncodedPlantVector:
        return self._wrap_vector(
            self.layout.local_cotangent_space.flatten(cotangent),
            "local_cotangent",
            self._anchor_mode(anchor, "Local-cotangent mode anchor"),
        )

    def decode_local_cotangent(self, encoded: EncodedPlantVector, /) -> PyTree[Array]:
        return self._decode_vector(encoded, "local_cotangent")

    def encode_cotangent(
        self,
        cotangent: PyTree[Any],
        /,
        *,
        anchor: EncodedPlantState | None = None,
    ) -> EncodedPlantVector:
        return self._wrap_vector(
            self.layout.cotangent_space.flatten(cotangent),
            "cotangent",
            self._anchor_mode(anchor, "Cotangent-vector mode anchor"),
        )

    def decode_cotangent(self, encoded: EncodedPlantVector, /) -> PyTree[Array]:
        return self._decode_vector(encoded, "cotangent")

    def retract(
        self,
        state: EncodedPlantState,
        local_tangent: EncodedPlantVector,
        /,
    ) -> EncodedPlantState:
        point = self._encoded_point(state, "Retraction state")
        local = self.decode_local(local_tangent)
        point = self._check_same_mode(state.mode, local_tangent.mode, point, "Retraction")
        return self._wrap_point(self.layout.geometry.retract(point, local), state.mode)

    def inverse_retract(
        self,
        state: EncodedPlantState,
        point: EncodedPlantState,
        /,
    ) -> EncodedPlantVector:
        anchor = self._encoded_point(state, "Inverse-retraction state")
        target = self._encoded_point(point, "Inverse-retraction point")
        anchor = self._check_same_mode(
            state.mode, point.mode, anchor, "Inverse retraction"
        )
        local = self.layout.geometry.inverse_retract(anchor, target)
        return self._wrap_vector(local, "local", state.mode)

    def retraction_jvp(
        self,
        state: EncodedPlantState,
        local_tangent: EncodedPlantVector,
        local_velocity: EncodedPlantVector,
        /,
    ) -> EncodedPlantVector:
        anchor = self._encoded_point(state, "Retraction-JVP state")
        local = self.decode_local(local_tangent)
        velocity = self.decode_local(local_velocity)
        anchor = self._check_same_mode(
            state.mode, local_tangent.mode, anchor, "Retraction JVP"
        )
        anchor = self._check_same_mode(
            state.mode, local_velocity.mode, anchor, "Retraction JVP"
        )
        tangent = self.layout.geometry.retraction_jvp(anchor, local, velocity)
        return self._wrap_vector(tangent, "tangent", state.mode)

    def retraction_inverse_jvp(
        self,
        state: EncodedPlantState,
        point: EncodedPlantState,
        tangent: EncodedPlantVector,
        /,
    ) -> EncodedPlantVector:
        anchor = self._encoded_point(state, "Inverse-retraction-JVP state")
        target = self._encoded_point(point, "Inverse-retraction-JVP point")
        vector = self.decode_tangent(tangent)
        anchor = self._check_same_mode(
            state.mode, point.mode, anchor, "Inverse retraction JVP"
        )
        anchor = self._check_same_mode(
            state.mode, tangent.mode, anchor, "Inverse retraction JVP"
        )
        local = self.layout.geometry.retraction_inverse_jvp(anchor, target, vector)
        return self._wrap_vector(local, "local", state.mode)

    def retraction_vjp(
        self,
        state: EncodedPlantState,
        local_tangent: EncodedPlantVector,
        cotangent: EncodedPlantVector,
        /,
    ) -> EncodedPlantVector:
        anchor = self._encoded_point(state, "Retraction-VJP state")
        local = self.decode_local(local_tangent)
        covector = self.decode_cotangent(cotangent)
        anchor = self._check_same_mode(
            state.mode, local_tangent.mode, anchor, "Retraction VJP"
        )
        anchor = self._check_same_mode(
            state.mode, cotangent.mode, anchor, "Retraction VJP"
        )
        local_covector = self.layout.geometry.retraction_vjp(anchor, local, covector)
        return self._wrap_vector(local_covector, "local_cotangent", state.mode)

    def transport_tangent(
        self,
        state: EncodedPlantState,
        point: EncodedPlantState,
        tangent: EncodedPlantVector,
        /,
    ) -> EncodedPlantVector:
        source = self._encoded_point(state, "Transport source")
        target = self._encoded_point(point, "Transport target")
        vector = self.decode_tangent(tangent)
        source = self._check_same_mode(
            state.mode, point.mode, source, "Tangent transport"
        )
        source = self._check_same_mode(
            state.mode, tangent.mode, source, "Tangent transport"
        )
        transported = self.layout.geometry.transport_tangent(source, target, vector)
        return self._wrap_vector(transported, "tangent", state.mode)

    def transport_cotangent_pullback(
        self,
        state: EncodedPlantState,
        point: EncodedPlantState,
        cotangent: EncodedPlantVector,
        /,
    ) -> EncodedPlantVector:
        source = self._encoded_point(state, "Cotangent-transport source")
        target = self._encoded_point(point, "Cotangent-transport target")
        covector = self.decode_cotangent(cotangent)
        source = self._check_same_mode(
            state.mode, point.mode, source, "Cotangent transport"
        )
        source = self._check_same_mode(
            state.mode, cotangent.mode, source, "Cotangent transport"
        )
        pulled_back = self.layout.geometry.transport_cotangent_pullback(
            source, target, covector
        )
        return self._wrap_vector(pulled_back, "cotangent", state.mode)

    def power_evidence(
        self,
        state: EncodedPlantState,
        local_tangent: EncodedPlantVector,
        local_velocity: EncodedPlantVector,
        cotangent: EncodedPlantVector,
        /,
    ) -> PlantPowerEvidence:
        """Evaluate the same algebraic power through the JVP and exact VJP."""
        tangent = self.retraction_jvp(state, local_tangent, local_velocity)
        local_cotangent = self.retraction_vjp(state, local_tangent, cotangent)
        physical_power = self.layout.cotangent_space.pair(
            self.decode_cotangent(cotangent), self.decode_tangent(tangent)
        )
        local_power = self.layout.local_cotangent_space.pair(
            self.decode_local_cotangent(local_cotangent),
            self.decode_local(local_velocity),
        )
        absolute = jnp.abs(physical_power - local_power)
        real_dtype = np.dtype(jnp.real(absolute).dtype)
        tolerance = jnp.asarray(32.0 * np.finfo(real_dtype).eps, dtype=real_dtype)
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=real_dtype),
            jnp.maximum(jnp.abs(physical_power), jnp.abs(local_power)),
        )
        finite = (
            jnp.all(jnp.isfinite(physical_power))
            & jnp.all(jnp.isfinite(local_power))
            & jnp.all(jnp.isfinite(absolute))
        )
        valid = finite & (absolute <= tolerance * scale)
        return PlantPowerEvidence(
            tangent,
            local_cotangent,
            physical_power,
            local_power,
            absolute,
            scale,
            tolerance,
            finite,
            valid,
            self.codec_id,
        )

    def _encoded_point(self, encoded: EncodedPlantState, owner: str, /) -> Array:
        if not isinstance(encoded, EncodedPlantState):
            raise TypeError(f"{owner} must be an EncodedPlantState.")
        self._check_binding(encoded, owner)
        point = self._point(encoded.vector, owner)
        return self._check_mode(encoded.mode, "point", point, owner)


class ControlVectorCodec(StrictModule):
    """Lossless vector codec for a complete homogeneous command PyTree."""

    schema: ArrayPyTreeSchema
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    executable_signature: ExecutableSignature
    command_dtype: np.dtype = eqx.field(static=True)
    size: int = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ArrayPyTreeSchema,
        /,
        *,
        semantic_provenance: SemanticProvenance,
        numeric_revision: NumericRevision,
        executable_signature: ExecutableSignature,
    ):
        if not isinstance(schema, ArrayPyTreeSchema):
            raise TypeError("schema must be an ArrayPyTreeSchema.")
        _identity_objects(semantic_provenance, numeric_revision, executable_signature)
        indices = tuple(range(len(schema.leaves)))
        command_dtype = _common_inexact_dtype(schema, indices, "Control codec")
        size = sum(prod(leaf.shape) for leaf in schema.leaves)
        payload = _binding_payload(
            semantic_provenance, numeric_revision, schema, executable_signature
        )
        codec_id = canonical_fingerprint(
            {
                "kind": "control-vector-codec",
                **payload,
                "schema_content_id": schema.content_id,
            }
        )
        self.schema = schema
        self.semantic_provenance = semantic_provenance
        self.numeric_revision = numeric_revision
        self.executable_signature = executable_signature
        self.command_dtype = command_dtype
        self.size = size
        self.codec_id = codec_id

    @property
    def semantic_id(self) -> str:
        return self.semantic_provenance.semantic_id

    @property
    def numeric_revision_id(self) -> str:
        return self.numeric_revision.revision_id

    @property
    def schema_id(self) -> str:
        return self.schema.schema_id

    @property
    def executable_signature_id(self) -> str:
        return self.executable_signature.signature_id

    def _binding(self) -> dict[str, str]:
        return {
            **_binding_payload(
                self.semantic_provenance,
                self.numeric_revision,
                self.schema,
                self.executable_signature,
            ),
            "codec_id": self.codec_id,
        }

    def encode_command(self, command: PyTree[Any], /) -> EncodedControl:
        indices = tuple(range(len(self.schema.leaves)))
        vector = _schema_vector(
            self.schema,
            command,
            indices,
            self.command_dtype,
            "Control command",
        )
        return EncodedControl(vector, **self._binding())

    def decode_command(self, encoded: EncodedControl, /) -> PyTree[Array]:
        if not isinstance(encoded, EncodedControl):
            raise TypeError("encoded must be an EncodedControl.")
        expected = self._binding()
        if any(getattr(encoded, name, None) != value for name, value in expected.items()):
            raise ValueError("Encoded control provenance does not match this codec.")
        vector = _array(encoded.vector, "Encoded control")
        expected_rank = self.schema.case_ndim + 1
        if vector.ndim != expected_rank or vector.shape[-1:] != (self.size,):
            raise ValueError(
                "Encoded control shape does not match the schema case rank and size."
            )
        if np.dtype(vector.dtype) != self.command_dtype:
            raise TypeError(
                f"Encoded control dtype {vector.dtype} does not match "
                f"{self.command_dtype}."
            )
        case_shape = vector.shape[: self.schema.case_ndim]
        leaves: list[Array] = []
        offset = 0
        for leaf in self.schema.leaves:
            size = prod(leaf.shape)
            leaves.append(
                vector[..., offset : offset + size].reshape(case_shape + leaf.shape)
            )
            offset += size
        return self.schema.unflatten(leaves)


__all__ = [
    "ControlVectorCodec",
    "EncodedControl",
    "EncodedPlantState",
    "EncodedPlantVector",
    "PlantModeRole",
    "PlantModeSidecar",
    "PlantPowerEvidence",
    "PlantStateVectorCodec",
    "PlantVectorRole",
]

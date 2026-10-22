import gc
from dataclasses import dataclass
from weakref import ref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._host_storage import retained_host_storage_profile


class _Auxiliary:
    def __init__(self, value: object) -> None:
        self.value = value


@jax.tree_util.register_pytree_node_class
class _MetadataNode:
    def __init__(self, children: tuple[object, ...], metadata: object) -> None:
        self.children = children
        self.metadata = metadata

    def tree_flatten(self) -> tuple[tuple[object, ...], object]:
        return self.children, self.metadata

    @classmethod
    def tree_unflatten(
        cls, metadata: object, children: tuple[object, ...]
    ) -> "_MetadataNode":
        return cls(children, metadata)


class _Module(eqx.Module):
    values: np.ndarray
    declaration: object = eqx.field(static=True)


def _profile(value: object) -> tuple[dict[int, object], int, int, int]:
    return retained_host_storage_profile(value, {}, lambda _: False)


def test_public_tree_definition_profiles_all_static_auxiliary_banks() -> None:
    numpy_bank = np.arange(128, dtype=np.float64)
    device_bank = jnp.arange(64, dtype=jnp.float64)
    child = _MetadataNode((), _Auxiliary((numpy_bank[::2], device_bank)))
    tree = jax.tree_util.tree_structure(
        _MetadataNode((child, child, (0, 1, 2)), {"bank": numpy_bank})
    )
    owners, declared, logical, visits = _profile(tree)
    assert owners[id(numpy_bank)] is numpy_bank
    assert owners[id(device_bank)] is device_bank
    assert logical == device_bank.nbytes
    assert declared >= numpy_bank.nbytes + device_bank.nbytes + 256 * tree.num_nodes
    assert visits > tree.num_nodes
    # Shared banks and views do not earn a second payload owner.
    again = retained_host_storage_profile(
        (numpy_bank, device_bank), owners, lambda _: False
    )
    assert again[1] < numpy_bank.nbytes
    assert again[2] == 0


def test_equinox_tree_metadata_traverses_missing_sentinel_namespace() -> None:
    dynamic = np.arange(1024, dtype=np.float64)
    module = _Module(dynamic, _Auxiliary({"static": np.arange(32, dtype=np.int64)}))
    tree = jax.tree_util.tree_structure(module)
    owners, declared, logical, _ = _profile(tree)
    declaration = module.declaration
    assert isinstance(declaration, _Auxiliary)
    assert isinstance(declaration.value, dict)
    static = declaration.value["static"]
    assert owners[id(static)] is static
    assert id(dynamic) not in owners  # Dynamic leaves are not owned by a treedef.
    assert declared >= static.nbytes
    assert logical == 0


def test_tree_auxiliary_callable_is_still_unsupported() -> None:
    tree = jax.tree_util.tree_structure(_MetadataNode((), _Auxiliary(lambda: 0)))
    with pytest.raises(TypeError, match="supported storage representation"):
        _profile(tree)


def test_tree_auxiliary_object_array_remains_unsupported() -> None:
    tree = jax.tree_util.tree_structure(
        _MetadataNode((), _Auxiliary(np.asarray([object()], dtype=object)))
    )
    with pytest.raises(TypeError, match="exclude objects"):
        _profile(tree)


@dataclass
class _WeakCallback:
    bank: np.ndarray
    edge: object = None

    def __call__(self, reference: object) -> None:
        pass


@pytest.mark.parametrize("alive", (False, True))
def test_weak_storage_retains_shell_without_retaining_referent(alive: bool) -> None:
    target = _Auxiliary(np.arange(1024))
    reference = ref(target)
    target_id = id(target)
    if not alive:
        del target
    owners, declared, logical, visits = _profile(reference)
    assert owners == {id(reference): reference}
    assert target_id not in owners
    assert (declared, logical, visits) == (256, 0, 1)
    if alive:
        del target
    gc.collect()
    assert reference() is None


def test_weak_storage_duplicate_shell_and_callback_bank_are_counted_once() -> None:
    target = _Auxiliary(None)
    bank = np.arange(32)
    callback = _WeakCallback(bank)
    reference = ref(target, callback)
    owners, declared, logical, _ = _profile((reference, reference))
    assert owners[id(callback)] is callback
    assert owners[id(bank)] is bank
    assert id(target) not in owners
    single = _profile((reference,))
    assert declared == single[1] + 16
    assert logical == 0


def test_weak_storage_callback_cycles_and_unsupported_callbacks_refuse() -> None:
    target = _Auxiliary(None)
    callback = _WeakCallback(np.arange(2))
    reference = ref(target, callback)
    callback.edge = reference
    with pytest.raises(ValueError, match="acyclic"):
        _profile(reference)
    with pytest.raises(TypeError, match="supported storage representation"):
        _profile(ref(target, lambda _: None))


@pytest.mark.parametrize("shortfall", (0, 1))
def test_weak_storage_exact_memory_boundary(shortfall: int) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    target = _Auxiliary(None)
    reference = ref(target)
    required = _profile(reference)[1]
    with HostBatchTicket(100, bytes_upper=required - shortfall) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100,
            maximum_memory_bytes=required - shortfall,
        )
        if shortfall:
            with pytest.raises(CoordinateEnclosureResourceError):
                budget.retain_owner(reference)
        else:
            budget.retain_owner(reference)


@pytest.mark.parametrize("placement", ("none", "single", "named", "abstract", "partition", "layout"))
def test_shape_metadata_retains_actual_placement_without_numerical_payload(placement: str) -> None:
    from jax.experimental.layout import Format, Layout
    from jax.sharding import (
        AbstractDevice,
        AbstractMesh,
        ManualAxisType,
        Mesh,
        NamedSharding,
        PartitionSpec,
        SingleDeviceSharding,
    )

    spec = PartitionSpec("x")
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("x",))
    abstract = AbstractMesh(
        (2,), ("x",), abstract_device=AbstractDevice("cpu", None, "cpu")
    )
    single = SingleDeviceSharding(jax.devices()[0])
    named = NamedSharding(mesh, spec)
    abstract_named = NamedSharding(abstract, spec, _logical_device_ids=(1, 0))
    layout = Layout((0,), tiling=((8,),), sub_byte_element_size_in_bits=8)
    sharding = {
        "none": None, "single": single, "named": named,
        "abstract": abstract_named, "partition": spec, "layout": Format(layout, single),
    }[placement]
    manual = ManualAxisType(varying=frozenset(("batch",)))
    shape = jax.ShapeDtypeStruct(
        (1 << 30,), np.float64, sharding=sharding, weak_type=True,
        manual_axis_type=manual, is_ref=True,
    )
    owners, declared, logical, _ = _profile(shape)
    assert owners[id(shape)] is shape
    for name in jax.ShapeDtypeStruct.__slots__:
        value = getattr(shape, name)
        assert owners[id(value)] is value
    assert owners[id(manual.varying)] is manual.varying
    assert logical == 0
    assert 0 < declared < 1 << 20  # Abstract extent is not an allocated array.
    if placement == "named":
        assert owners[id(mesh.devices)] is mesh.devices
        assert owners[id(mesh.axis_types)] is mesh.axis_types
    elif placement == "abstract":
        assert owners[id(abstract.abstract_device)] is abstract.abstract_device
        assert owners[id(abstract_named._logical_device_ids)] is abstract_named._logical_device_ids
    elif placement == "layout":
        assert owners[id(layout.tiling)] is layout.tiling
    # Raw PartitionSpec is profiled outside a mesh context, without resolving
    # the shape's .sharding property through an ambient execution environment.


def test_shape_layout_auxiliary_metadata_counts_actual_array_banks() -> None:
    from jax.experimental.layout import Format, Layout
    from jax.sharding import SingleDeviceSharding

    layout = Layout((0,))
    numpy_bank = np.arange(128, dtype=np.float64)
    device_bank = jnp.arange(64, dtype=jnp.float64)
    vars(layout)["actual_auxiliary"] = (numpy_bank[::2], device_bank)
    shape = jax.ShapeDtypeStruct(
        (64,), np.float64,
        sharding=Format(layout, SingleDeviceSharding(jax.devices()[0])),
    )
    tree = jax.tree_util.tree_structure(_MetadataNode((), shape))
    owners, declared, logical, _ = _profile(tree)
    assert owners[id(numpy_bank)] is numpy_bank
    assert owners[id(device_bank)] is device_bank
    assert logical == device_bank.nbytes
    assert declared >= numpy_bank.nbytes + device_bank.nbytes


def test_shape_metadata_does_not_admit_an_opaque_callback() -> None:
    shape = jax.ShapeDtypeStruct((4,), np.float64, _memory_space=lambda: 0)
    with pytest.raises(TypeError, match="supported storage representation"):
        _profile(shape)

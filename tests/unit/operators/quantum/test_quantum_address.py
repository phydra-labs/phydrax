#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.operators.quantum import AbelianGroup, FermionModeOrder
from phydrax.operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumAddressCodec,
    QuantumConfigurationDomain,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)


pytestmark = pytest.mark.strict_jax


def _specification(
    spaces: tuple[LocalSpacePlan, ...],
    coefficient: float = 1.0,
    order: FermionModeOrder | None = None,
) -> QuantumLatticeSpecification:
    local = LocalOperatorPlan(
        spaces[0],
        "identity",
        np.eye(spaces[0].dimension, dtype=np.complex128),
        (0,) * len(spaces[0].charge_labels),
    )
    return QuantumLatticeSpecification(
        spaces,
        (QuantumLatticeTerm((local,), coefficient=coefficient, label="identity-term"),),
        fermion_mode_order=order,
    )


def test_cross_word_field_round_trip_has_exact_words() -> None:
    codec = QuantumAddressCodec((5,) + (2,) * 31)
    coordinate = jnp.asarray((4,) + (0,) * 31, dtype=jnp.int32)
    key = jax.jit(codec.encode)(coordinate)
    np.testing.assert_array_equal(key, np.asarray((2, 0), dtype=np.uint32))
    np.testing.assert_array_equal(jax.jit(codec.decode)(key), coordinate)


def test_hundred_binary_modes_require_four_words_not_rankability() -> None:
    spaces = tuple(LocalSpacePlan.spin(f"s{i}", 1) for i in range(100))
    domain = QuantumConfigurationDomain(
        _specification(spaces), species_ids=("spin-species",) * 100
    )
    coordinate = np.zeros(100, dtype=np.int32)
    coordinate[[0, 31, 32, 99]] = 1
    address = domain.address(coordinate)
    expected = (1 << 99) | (1 << 68) | (1 << 67) | 1
    np.testing.assert_array_equal(
        address.key_words,
        np.asarray(
            tuple((expected >> shift) & (2**32 - 1) for shift in (96, 64, 32, 0)),
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(domain.decode(address.key_words), coordinate)
    assert domain.codec.word_count == 4


def test_zero_key_is_an_active_configuration() -> None:
    space = LocalSpacePlan.spin("s", 1)
    domain = QuantumConfigurationDomain(_specification((space,)), species_ids=("spin",))
    address = domain.address(jnp.asarray((0,), dtype=jnp.int32))
    np.testing.assert_array_equal(address.key_words, np.zeros(1, dtype=np.uint32))
    assert bool(domain.contains_keys(address.key_words))


@pytest.mark.parametrize(
    "key", [(4,), (3,)], ids=["unused-high-bit", "unused-local-code"]
)
def test_noncanonical_or_out_of_range_keys_refuse(key: tuple[int, ...]) -> None:
    space = LocalSpacePlan.boson("b", 3)
    domain = QuantumConfigurationDomain(_specification((space,)), species_ids=("boson",))
    words = jnp.asarray(key, dtype=jnp.uint32)
    assert not bool(domain.contains_keys(words))
    with pytest.raises(Exception, match="outside"):
        domain.from_key(words)


@pytest.mark.parametrize(
    "coordinate",
    [-1, 3, 2**32, 2**64 - 1],
    ids=["negative", "local-range", "uint32-wrap", "uint64-wrap"],
)
def test_coordinates_refuse_before_int32_narrowing(coordinate: int) -> None:
    codec = QuantumAddressCodec((3,))
    dtype = np.int64 if coordinate < 0 else np.uint64
    with pytest.raises(Exception, match="Invalid local-state"):
        codec.encode(np.asarray((coordinate,), dtype=dtype))


def test_domain_identity_is_coefficient_and_resource_independent() -> None:
    space = LocalSpacePlan.spin("s", 1)
    first = QuantumConfigurationDomain(
        _specification((space,), 1.0), species_ids=("spin",), maximum_address_words=1
    )
    second = QuantumConfigurationDomain(
        _specification((space,), 7.0), species_ids=("spin",), maximum_address_words=99
    )
    assert first.domain_id == second.domain_id
    assert first.codec.codec_id == second.codec.codec_id
    second.validate_address(first.address((1,)))


def test_explicit_species_changes_physical_equality() -> None:
    space = LocalSpacePlan.spin("s", 1)
    specification = _specification((space,))
    first = QuantumConfigurationDomain(specification, species_ids=("species-a",))
    second = QuantumConfigurationDomain(specification, species_ids=("species-b",))
    with pytest.raises(ValueError, match="another physical"):
        second.validate_address(first.address((0,)))


def test_mode_order_changes_domain_identity_without_changing_codec() -> None:
    spaces = (LocalSpacePlan.fermion("a", "a"), LocalSpacePlan.fermion("b", "b"))
    first = QuantumConfigurationDomain(
        _specification(spaces, order=FermionModeOrder(("a", "b"))),
        species_ids=("electron",) * 2,
    )
    second = QuantumConfigurationDomain(
        _specification(spaces, order=FermionModeOrder(("b", "a"))),
        species_ids=("electron",) * 2,
    )
    assert first.codec.codec_id == second.codec.codec_id
    assert first.domain_id != second.domain_id


def test_mixed_integral_modular_constraints_are_exact() -> None:
    spaces = tuple(
        LocalSpacePlan(
            site, ("empty", "occupied"), ("number", "parity"), ((0, 0), (1, 1))
        )
        for site in ("a", "b")
    )
    domain = QuantumConfigurationDomain(
        _specification(spaces),
        species_ids=("component",) * 2,
        charge_group=AbelianGroup((None, 2)),
        charge_labels=("number", "parity"),
        total_charge=(1, 3),
    )
    keys = jnp.asarray(((0,), (1,), (2,), (3,)), dtype=jnp.uint32)
    np.testing.assert_array_equal(
        eqx.filter_jit(domain.contains_keys)(keys), (False, True, True, False)
    )
    assert domain.total_charge == (1, 1)


def test_singleton_codec_retains_one_zero_word() -> None:
    codec = QuantumAddressCodec((1, 1))
    np.testing.assert_array_equal(codec.encode((0, 0)), np.zeros(1, dtype=np.uint32))
    assert not bool(codec.contains_keys(jnp.asarray((1,), dtype=jnp.uint32)))


def test_address_word_budget_refuses_before_packing() -> None:
    with pytest.raises(ValueError, match="word budget"):
        QuantumAddressCodec((2,) * 100, maximum_address_words=3)


def test_modular_accumulator_refuses_unrepresentable_intermediates() -> None:
    space = LocalSpacePlan.spin("s", 1)
    with pytest.raises(ValueError, match="accumulator"):
        QuantumConfigurationDomain(
            _specification((space,)),
            species_ids=("spin",),
            charge_group=AbelianGroup((2**63 - 1,)),
            charge_labels=("twice-spin-projection",),
            total_charge=(0,),
        )


def test_local_dimension_refuses_int32_coordinate_overflow_before_allocation() -> None:
    with pytest.raises(ValueError, match="int32"):
        QuantumAddressCodec((2**31,))


def test_statistics_changes_physical_domain_not_packed_layout() -> None:
    first_space = LocalSpacePlan(
        "site", ("a", "b"), (), np.zeros((2, 0), dtype=np.int32), statistics="finite"
    )
    second_space = LocalSpacePlan(
        "site", ("a", "b"), (), np.zeros((2, 0), dtype=np.int32), statistics="spin"
    )
    first = QuantumConfigurationDomain(
        _specification((first_space,)), species_ids=("component",)
    )
    second = QuantumConfigurationDomain(
        _specification((second_space,)), species_ids=("component",)
    )
    assert first.codec.codec_id == second.codec.codec_id
    assert first.domain_id != second.domain_id


def test_domain_decode_refuses_noncanonical_word_bits() -> None:
    space = LocalSpacePlan.spin("s", 1)
    domain = QuantumConfigurationDomain(_specification((space,)), species_ids=("spin",))
    with pytest.raises(Exception, match="outside"):
        domain.decode(jnp.asarray((2,), dtype=jnp.uint32))


def test_packed_key_boundary_never_narrows_uint64_words() -> None:
    space = LocalSpacePlan.spin("s", 1)
    domain = QuantumConfigurationDomain(_specification((space,)), species_ids=("spin",))
    with pytest.raises(TypeError, match="cannot cast"):
        domain.from_key(jnp.asarray((2**32,), dtype=jnp.uint64))


def test_fractional_local_coordinates_refuse_instead_of_truncating() -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        QuantumAddressCodec((3,)).encode(jnp.asarray((1.5,), dtype=jnp.float64))

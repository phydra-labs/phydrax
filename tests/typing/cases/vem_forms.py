"""Canonical VEM scientific values and projected reconstruction ports."""

from typing import assert_type

from phydrax.discretization.vem import (
    conforming_h1_virtual_element,
    conforming_hcurl_virtual_element,
    conforming_hdiv_virtual_element,
    discontinuous_l2_virtual_element,
    PreparedPolyhedralH1VirtualElement3D,
    VirtualElementDiscretization,
    VirtualElementSpec,
)
from phydrax.equations.vem import prepare_virtual_element_field_reconstruction
from phydrax.exterior import FormType, FormValueSpec


def canonical_values() -> None:
    h1 = conforming_h1_virtual_element(1)
    hdiv = conforming_hdiv_virtual_element(1)
    hcurl = conforming_hcurl_virtual_element(1)
    l2 = discontinuous_l2_virtual_element(1)
    assert_type(h1, VirtualElementSpec)
    assert_type(h1.value_spec, FormValueSpec)
    assert_type(hdiv.value_spec, FormValueSpec)
    assert_type(hcurl.form_type, FormType)
    assert_type(l2.value_shape, tuple[int, ...])
    VirtualElementSpec("ConformingHdiv", 1, value_spec=hdiv.value_spec)
    VirtualElementSpec("ConformingHdiv", 1, value_spec=hdiv.value_spec, conformity="Hdiv")  # ty: ignore[unknown-argument]


def declared_reconstruction(space: VirtualElementDiscretization) -> None:
    reconstruction = prepare_virtual_element_field_reconstruction(
        space, channel="l2-projection"
    )
    assert_type(reconstruction.value_port.form, FormValueSpec | None)
    assert_type(space.dof_map.value_spec, FormValueSpec)


def scalar_polyhedral(prepared: PreparedPolyhedralH1VirtualElement3D) -> None:
    assert_type(prepared.value_spec, FormValueSpec)

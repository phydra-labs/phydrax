#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic electronic, lattice, transport, and embedding physics."""

from importlib import import_module

from ._boltzmann import __all__ as _boltzmann_all
from ._derivatives import __all__ as _derivatives_all
from ._electrostatics import __all__ as _electrostatics_all
from ._embedding import __all__ as _embedding_all
from ._embedding_qualification import __all__ as _embedding_qualification_all
from ._finite import __all__ as _finite_all
from ._gamma import __all__ as _gamma_all
from ._kubo import __all__ as _kubo_all
from ._lattice_dynamics import __all__ as _lattice_dynamics_all
from ._lattice_force_constants import __all__ as _lattice_force_constants_all
from ._lattice_frontier import __all__ as _lattice_frontier_all
from ._lattice_provider import __all__ as _lattice_provider_all
from ._lattice_qualification import __all__ as _lattice_qualification_all
from ._lattice_thermodynamics import __all__ as _lattice_thermodynamics_all
from ._lattice_transport import __all__ as _lattice_transport_all
from ._lifecycle import __all__ as _lifecycle_all
from ._magnetism import __all__ as _magnetism_all
from ._many_body import __all__ as _many_body_all
from ._model_scf import __all__ as _model_scf_all
from ._observables import __all__ as _observables_all
from ._orbital_model import __all__ as _orbital_model_all
from ._properties import __all__ as _properties_all
from ._qualification import __all__ as _qualification_all
from ._source import __all__ as _source_all
from ._spectrum import __all__ as _spectrum_all
from ._spin import __all__ as _spin_all
from ._superconductivity import __all__ as _superconductivity_all
from ._topology import __all__ as _topology_all
from ._transport_campaigns import __all__ as _transport_campaigns_all
from ._transport_support import __all__ as _transport_support_all


_FACADE_EXPORT_MODULES = (
    "._boltzmann",
    "._derivatives",
    "._electrostatics",
    "._embedding",
    "._embedding_qualification",
    "._finite",
    "._gamma",
    "._kubo",
    "._lattice_dynamics",
    "._lattice_force_constants",
    "._lattice_frontier",
    "._lattice_provider",
    "._lattice_qualification",
    "._lattice_thermodynamics",
    "._lattice_transport",
    "._lifecycle",
    "._magnetism",
    "._many_body",
    "._model_scf",
    "._observables",
    "._orbital_model",
    "._properties",
    "._qualification",
    "._source",
    "._spectrum",
    "._spin",
    "._superconductivity",
    "._topology",
    "._transport_campaigns",
    "._transport_support",
)


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    *_lifecycle_all,
    *_qualification_all,
    *_embedding_qualification_all,
    *_lattice_qualification_all,
    *_derivatives_all,
    *_boltzmann_all,
    *_electrostatics_all,
    *_embedding_all,
    *_finite_all,
    *_gamma_all,
    *_kubo_all,
    *_lattice_dynamics_all,
    *_lattice_force_constants_all,
    *_lattice_frontier_all,
    *_lattice_provider_all,
    *_lattice_thermodynamics_all,
    *_lattice_transport_all,
    *_magnetism_all,
    *_many_body_all,
    *_model_scf_all,
    *_observables_all,
    *_orbital_model_all,
    *_properties_all,
    *_source_all,
    *_spectrum_all,
    *_spin_all,
    *_superconductivity_all,
    *_topology_all,
    *_transport_campaigns_all,
    *_transport_support_all,
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small physical devices for examples and independent mesh-convergence studies.

All lengths, areas, and densities accepted here are SI. These constructors use
native scoped material/contact/dopant bindings; they do not select a solver.
The silicon parameters are an illustrative isothermal 300 K model, not a
calibrated fabrication process.
"""

from __future__ import annotations

import numpy as np

from ...meshing import MeshAttribute, MeshAttributeRole, MeshPatch, MeshZone, MeshZoneRole
from ._device import DevicePlan, GateContact, MaterialBinding, OhmicContact
from ._materials import DielectricMaterial, SemiconductorMaterial
from ._quantities import PER_CUBIC_METER
from ._support import TransportSupport


def _dopant(support, name, values):
    return MeshAttribute(
        name,
        MeshAttributeRole.MATERIAL,
        support.node_scope(),
        values,
        unit=PER_CUBIC_METER,
    )


def pn_junction(
    nodes: int = 81,
    *,
    length: float = 4.0e-6,
    area: float = 1.0e-12,
    acceptor_density: float = 1.0e21,
    donor_density: float = 1.0e21,
) -> DevicePlan:
    """Abrupt silicon PN junction; terminals are ``anode``, then ``cathode``."""
    if nodes < 5 or min(length, area, acceptor_density, donor_density) <= 0:
        raise ValueError(
            "PN geometry and doping must be positive, with at least five nodes."
        )
    support = TransportSupport.interval(np.linspace(0.0, length, nodes), area=area)
    x = np.asarray(support.positions)[:, 0]
    donors = np.where(x >= length / 2, donor_density, 0.0)
    acceptors = np.where(x < length / 2, acceptor_density, 0.0)
    material = MeshZone("silicon", MeshZoneRole.MATERIAL, support.node_scope())
    return DevicePlan(
        support,
        materials=(MaterialBinding(material, SemiconductorMaterial.silicon()),),
        donor_attributes=(_dopant(support, "donors", donors),),
        acceptor_attributes=(_dopant(support, "acceptors", acceptors),),
        contacts=(
            OhmicContact("anode", support.boundary_patch("anode", side="lower")),
            OhmicContact("cathode", support.boundary_patch("cathode", side="upper")),
        ),
    )


def mos_capacitor(
    semiconductor_nodes: int = 61,
    oxide_nodes: int = 7,
    *,
    semiconductor_depth: float = 2.0e-6,
    oxide_thickness: float = 50.0e-9,
    area: float = 1.0e-12,
    acceptor_density: float = 1.0e21,
    gate_offset: float = 0.0,
) -> DevicePlan:
    """Explicit oxide/p-silicon MOS capacitor; terminals ``gate``, ``bulk``.

    ``gate_offset`` is an imposed work-function voltage offset, not a fitted
    threshold voltage. The default has no additional gate work-function shift.
    """
    if (
        min(semiconductor_nodes, oxide_nodes) < 3
        or min(semiconductor_depth, oxide_thickness, area, acceptor_density) <= 0
    ):
        raise ValueError(
            "MOS dimensions/doping must be positive, with three nodes per layer."
        )
    x = np.concatenate(
        (
            np.linspace(0.0, oxide_thickness, oxide_nodes, endpoint=False),
            np.linspace(
                oxide_thickness,
                oxide_thickness + semiconductor_depth,
                semiconductor_nodes,
            ),
        )
    )
    support = TransportSupport.interval(x, area=area)
    oxide_ids = np.flatnonzero(x < oxide_thickness)
    silicon_ids = np.flatnonzero(x >= oxide_thickness)
    oxide = MeshZone("oxide", MeshZoneRole.MATERIAL, support.node_scope(oxide_ids))
    silicon = MeshZone("silicon", MeshZoneRole.MATERIAL, support.node_scope(silicon_ids))
    acceptors = np.where(x >= oxide_thickness, acceptor_density, 0.0)
    return DevicePlan(
        support,
        materials=(
            MaterialBinding(
                oxide,
                DielectricMaterial(
                    "silicon-dioxide",
                    permittivity=3.9 * 8.8541878128e-12,
                    provenance="Illustrative static SiO2 relative permittivity 3.9",
                ),
            ),
            MaterialBinding(silicon, SemiconductorMaterial.silicon()),
        ),
        acceptor_attributes=(_dopant(support, "acceptors", acceptors),),
        contacts=(
            GateContact(
                "gate",
                support.boundary_patch("gate", side="lower"),
                potential_offset=gate_offset,
            ),
            OhmicContact("bulk", support.boundary_patch("bulk", side="upper")),
        ),
    )


def bipolar_transistor(
    axial_nodes: int = 31,
    transverse_nodes: int = 9,
    *,
    length: float = 3.0e-6,
    height: float = 1.0e-6,
    depth: float = 1.0e-6,
    emitter_density: float = 2.0e21,
    base_density: float = 1.0e21,
    collector_density: float = 5.0e20,
) -> DevicePlan:
    """Two-dimensional lateral NPN with a separate top-surface base terminal.

    Terminals are ``emitter``, ``base``, ``collector``. Emitter occupies the
    first third and base the following sixth of the lateral domain. This is a
    mesh-convergence example, not a process-calibrated transistor geometry.
    """
    if (
        axial_nodes < 13
        or transverse_nodes < 3
        or min(length, height, depth, emitter_density, base_density, collector_density)
        <= 0
    ):
        raise ValueError(
            "BJT geometry/doping must be positive; use at least 13 by 3 nodes."
        )
    support = TransportSupport.tensor_grid(
        (
            np.linspace(0.0, length, axial_nodes),
            np.linspace(0.0, height, transverse_nodes),
        ),
        transverse_measure=depth,
    )
    xy = np.asarray(support.positions)
    x, y = xy[:, 0], xy[:, 1]
    emitter = x < length / 3
    base = (x >= length / 3) & (x <= length / 2)
    donors = np.where(emitter, emitter_density, np.where(base, 0.0, collector_density))
    acceptors = np.where(base, base_density, 0.0)
    base_nodes = np.flatnonzero(base & np.isclose(y, height, rtol=1.0e-12, atol=0.0))
    if base_nodes.size == 0:
        raise ValueError("The mesh must resolve the base terminal.")
    base_patch = MeshPatch("base", support.node_scope(base_nodes))
    silicon = MeshZone("silicon", MeshZoneRole.MATERIAL, support.node_scope())
    return DevicePlan(
        support,
        materials=(MaterialBinding(silicon, SemiconductorMaterial.silicon()),),
        donor_attributes=(_dopant(support, "donors", donors),),
        acceptor_attributes=(_dopant(support, "acceptors", acceptors),),
        contacts=(
            OhmicContact(
                "emitter", support.boundary_patch("emitter", axis=0, side="lower")
            ),
            OhmicContact("base", base_patch),
            OhmicContact(
                "collector", support.boundary_patch("collector", axis=0, side="upper")
            ),
        ),
    )


__all__ = ["bipolar_transistor", "mos_capacitor", "pn_junction"]

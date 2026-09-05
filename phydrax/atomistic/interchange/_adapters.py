#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...discretization import PeriodicCell
from ...units import (
    ANGSTROM,
    conversion_factor,
    DALTON,
    ELEMENTARY_CHARGE,
    KILOCALORIE_PER_MOLE,
    KILOJOULE_PER_MOLE,
    LENGTH,
    SI_REFERENCE_SYSTEM_ID,
    UnitDefinition,
)
from .._classical import (
    HarmonicAnglePotential,
    HarmonicBondPotential,
    LennardJonesPotential,
    PeriodicTorsionPotential,
)
from .._constraints import DistanceConstraintPlan
from .._electrostatics import (
    DirectCoulombPotential,
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
)
from .._force_field import (
    AtomisticForceFieldPlan,
    AtomisticForceFieldProvenance,
    AtomisticNonbondedPolicy,
    ForceFieldTermKind,
    GeneralForceFieldTerm,
    PeriodicTorsionSeriesPotential,
    ReactionFieldPotential,
)
from .._potential_program import AtomisticPotentialProgram
from .._system import AtomisticSystemPlan
from .._topology import MolecularTopologyPlan
from .._units import AtomisticUnitSystem, molar_energy_to_single_system_factor
from ._core import (
    AtomisticInterchangeBundle,
    AtomisticInterchangeReport,
    canonical_source_digest,
    require_mapping_fields,
)


_NANOMETER = UnitDefinition("nm", LENGTH, SI_REFERENCE_SYSTEM_ID, "1e-9")


def _openmm_unit_factors(units: AtomisticUnitSystem, /) -> dict[str, float | str]:
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("OpenMM interchange requires an AtomisticUnitSystem.")
    try:
        length_from_angstrom = float(conversion_factor(ANGSTROM, units.scale.length_unit))
        mass_from_dalton = float(conversion_factor(DALTON, units.mass_unit))
        charge_from_elementary = float(
            conversion_factor(ELEMENTARY_CHARGE, units.charge_unit)
        )
        energy_from_kilojoule = molar_energy_to_single_system_factor(
            KILOJOULE_PER_MOLE,
            units.scale.energy_unit,
            constant_set_id=units.constant_set_id,
        )
        length_to_nanometer = float(
            conversion_factor(units.scale.length_unit, _NANOMETER)
        )
    except ValueError as error:
        raise ValueError(
            "OpenMM interchange requires SI-referenced length, ordinary energy, "
            "mass, and charge units with Avogadro provenance."
        ) from error
    return {
        "length_from_angstrom": length_from_angstrom,
        "energy_from_kilojoule": energy_from_kilojoule,
        "mass_from_dalton": mass_from_dalton,
        "charge_from_elementary": charge_from_elementary,
        "length_to_nanometer": length_to_nanometer,
        "energy_to_kilojoule": 1.0 / energy_from_kilojoule,
        "mass_to_dalton": 1.0 / mass_from_dalton,
        "charge_to_elementary": 1.0 / charge_from_elementary,
        "avogadro_constant_set_id": units.constant_set_id,
    }


def _potential_term_to_mapping(term, /) -> dict[str, Any]:
    common = {"name": term.name, "force_group": term.force_group}
    if isinstance(term, HarmonicBondPotential):
        return {
            **common,
            "kind": "harmonic-bond",
            "stiffness": np.asarray(term.stiffness),
            "equilibrium": np.asarray(term.equilibrium_distance),
        }
    if isinstance(term, HarmonicAnglePotential):
        return {
            **common,
            "kind": "harmonic-angle",
            "stiffness": np.asarray(term.stiffness),
            "equilibrium": np.asarray(term.equilibrium_angle),
        }
    if isinstance(term, PeriodicTorsionPotential):
        return {
            **common,
            "kind": "periodic-torsion",
            "amplitude": np.asarray(term.amplitude),
            "periodicity": np.asarray(term.periodicity),
            "phase": np.asarray(term.phase),
            "improper": term.improper,
        }
    if isinstance(term, LennardJonesPotential):
        return {
            **common,
            "kind": "lennard-jones",
            "epsilon": np.asarray(term.epsilon),
            "sigma": np.asarray(term.sigma),
            "cutoff": term.cutoff,
            "switch_distance": term.switch_distance,
            "combining_rule": term.combining_rule,
            "explicit_epsilon": None
            if term.explicit_epsilon is None
            else np.asarray(term.explicit_epsilon),
            "explicit_sigma": None
            if term.explicit_sigma is None
            else np.asarray(term.explicit_sigma),
        }
    if isinstance(term, DirectCoulombPotential):
        return {**common, "kind": "direct-coulomb"}
    if isinstance(term, EwaldReferencePotential):
        return {
            **common,
            "kind": "ewald-reference",
            "alpha": term.alpha,
            "real_cutoff": term.real_cutoff,
            "reciprocal_extent": term.reciprocal_extent,
            "neutrality": term.neutrality,
            "charge_tolerance": term.charge_tolerance,
        }
    if isinstance(term, ParticleMeshEwaldPotential):
        return {
            **common,
            "kind": "particle-mesh-ewald",
            "alpha": term.alpha,
            "real_cutoff": term.real_cutoff,
            "grid_shape": term.grid_shape,
            "spline_degree": term.spline_degree,
            "neutrality": term.neutrality,
            "charge_tolerance": term.charge_tolerance,
        }
    if isinstance(term, GeneralForceFieldTerm):
        return {
            **common,
            "kind": "general-force-field",
            "term_kind": term.kind.value,
            "arrays": tuple(np.asarray(value) for value in term.arrays),
            "route_indices": np.asarray(term.route_indices),
            "cutoff": term.cutoff,
        }
    raise ValueError(f"Potential term {term.name!r} has no interchange mapping.")


def _potential_term_from_mapping(value: dict[str, Any], /):
    kind = value["kind"]
    common = {
        "name": value["name"],
        "force_group": int(value["force_group"]),
    }
    if kind == "harmonic-bond":
        return HarmonicBondPotential(value["stiffness"], value["equilibrium"], **common)
    if kind == "harmonic-angle":
        return HarmonicAnglePotential(value["stiffness"], value["equilibrium"], **common)
    if kind == "periodic-torsion":
        return PeriodicTorsionPotential(
            value["amplitude"],
            value["periodicity"],
            value["phase"],
            improper=bool(value["improper"]),
            **common,
        )
    if kind == "lennard-jones":
        return LennardJonesPotential(
            value["epsilon"],
            value["sigma"],
            value["cutoff"],
            switch_distance=value["switch_distance"],
            combining_rule=value["combining_rule"],
            explicit_epsilon=value["explicit_epsilon"],
            explicit_sigma=value["explicit_sigma"],
            **common,
        )
    if kind == "direct-coulomb":
        return DirectCoulombPotential(**common)
    if kind == "ewald-reference":
        return EwaldReferencePotential(
            value["alpha"],
            value["real_cutoff"],
            value["reciprocal_extent"],
            neutrality=value["neutrality"],
            charge_tolerance=value["charge_tolerance"],
            **common,
        )
    if kind == "particle-mesh-ewald":
        return ParticleMeshEwaldPotential(
            value["alpha"],
            value["real_cutoff"],
            tuple(value["grid_shape"]),
            spline_degree=value["spline_degree"],
            neutrality=value["neutrality"],
            charge_tolerance=value["charge_tolerance"],
            **common,
        )
    if kind == "general-force-field":
        return GeneralForceFieldTerm(
            ForceFieldTermKind(value["term_kind"]),
            tuple(value["arrays"]),
            route_indices=value["route_indices"],
            cutoff=value["cutoff"],
            **common,
        )
    raise ValueError(f"Unknown atomistic potential mapping kind {kind!r}.")


def force_field_from_mapping(value: dict[str, Any], /) -> AtomisticInterchangeBundle:
    require_mapping_fields(
        value,
        (
            "unit_system",
            "particle_ids",
            "atomic_numbers",
            "masses",
            "atom_type_ids",
            "charges",
            "nonbonded",
        ),
    )
    units = AtomisticUnitSystem.from_dict(value["unit_system"])
    topology_data = value.get("topology", {})
    topology = MolecularTopologyPlan(
        bonds=topology_data.get("bonds"),
        angles=topology_data.get("angles"),
        torsions=topology_data.get("torsions"),
        impropers=topology_data.get("impropers"),
        constraints=topology_data.get("constraints"),
        constraint_distances=topology_data.get("constraint_distances"),
        pair_exceptions=topology_data.get("pair_exceptions"),
        lennard_jones_scales=topology_data.get("lennard_jones_scales"),
        electrostatic_scales=topology_data.get("electrostatic_scales"),
        bond_type_ids=topology_data.get("bond_type_ids"),
        angle_type_ids=topology_data.get("angle_type_ids"),
        torsion_type_ids=topology_data.get("torsion_type_ids"),
        improper_type_ids=topology_data.get("improper_type_ids"),
    )
    system = AtomisticSystemPlan(
        value["particle_ids"],
        value["atomic_numbers"],
        value["masses"],
        units,
        atom_type_ids=value["atom_type_ids"],
        charges=value["charges"],
        active_mask=value.get("active_mask"),
        mobile_mask=value.get("mobile_mask"),
        molecule_ids=value.get("molecule_ids"),
        region_ids=value.get("region_ids"),
        topology=topology,
        cell=value.get("cell"),
        coordinate_map=value.get("coordinate_map"),
        name=value.get("name", "interchanged-system"),
    )
    serialized_terms = value.get("potential_terms")
    legacy_mapping = serialized_terms is None
    terms = (
        []
        if legacy_mapping
        else [_potential_term_from_mapping(term) for term in serialized_terms]
    )
    parameters = value.get("parameters", {})
    if legacy_mapping and "bond_stiffness" in parameters:
        terms.append(
            HarmonicBondPotential(parameters["bond_stiffness"], parameters["bond_length"])
        )
    if legacy_mapping and "angle_stiffness" in parameters:
        terms.append(
            HarmonicAnglePotential(
                parameters["angle_stiffness"], parameters["angle_value"]
            )
        )
    if legacy_mapping and "torsion_amplitude" in parameters:
        if "torsion_mask" in parameters:
            slots = {
                int(particle_id): slot
                for slot, particle_id in enumerate(np.asarray(system.particle_ids))
            }
            routes = np.asarray(
                [
                    [slots[int(particle_id)] for particle_id in route]
                    for route in np.asarray(topology.torsions)
                ],
                dtype=np.int32,
            )
            type_ids = np.asarray(topology.torsion_type_ids)
            terms.append(
                PeriodicTorsionSeriesPotential(
                    np.asarray(parameters["torsion_amplitude"])[type_ids],
                    np.asarray(parameters["torsion_periodicity"])[type_ids],
                    np.asarray(parameters["torsion_phase"])[type_ids],
                    np.asarray(parameters["torsion_mask"])[type_ids],
                    routes,
                )
            )
        else:
            terms.append(
                PeriodicTorsionPotential(
                    parameters["torsion_amplitude"],
                    parameters["torsion_periodicity"],
                    parameters["torsion_phase"],
                )
            )
    nonbonded = value["nonbonded"]
    policy = AtomisticNonbondedPolicy(
        nonbonded["cutoff"],
        switch_distance=nonbonded.get("switch_distance"),
        combining_rule=nonbonded.get("combining_rule", "lorentz-berthelot"),
        electrostatics=nonbonded.get(
            "electrostatics", "pme" if system.cell is not None else "direct"
        ),
        dispersion=nonbonded.get("dispersion", "cutoff"),
        charge_neutrality=nonbonded.get("charge_neutrality", "require-neutral"),
    )
    if legacy_mapping and "epsilon" in nonbonded:
        terms.append(
            LennardJonesPotential(
                nonbonded["epsilon"],
                nonbonded["sigma"],
                policy.cutoff,
                switch_distance=policy.switch_distance,
                combining_rule=policy.combining_rule,
            )
        )
    if legacy_mapping and np.any(np.asarray(value["charges"]) != 0.0):
        if policy.electrostatics == "direct":
            terms.append(DirectCoulombPotential())
        elif policy.electrostatics == "reaction-field":
            terms.append(
                ReactionFieldPotential(
                    nonbonded.get("reaction_field_dielectric", 78.5), policy.cutoff
                )
            )
        elif policy.electrostatics == "ewald":
            terms.append(
                EwaldReferencePotential(
                    nonbonded["ewald_alpha"],
                    policy.cutoff,
                    nonbonded["reciprocal_extent"],
                    neutrality=policy.charge_neutrality,
                )
            )
        else:
            terms.append(
                ParticleMeshEwaldPotential(
                    nonbonded["ewald_alpha"],
                    policy.cutoff,
                    tuple(nonbonded["grid_shape"]),
                    neutrality=policy.charge_neutrality,
                )
            )
    if not terms:
        raise ValueError("Interchange mapping contains no supported potential terms.")
    source_digest = canonical_source_digest(value)
    provenance = AtomisticForceFieldProvenance(
        value.get("source_format", "mapping"),
        (source_digest,),
        value.get("family", "custom"),
        value.get("parameter_set", "explicit"),
        water_model=value.get("water_model"),
        ion_model=value.get("ion_model"),
        typing_source=value.get("typing_source", "explicit"),
        charge_source=value.get("charge_source", "explicit"),
        adapter_id=value.get("adapter_id", "mapping"),
    )
    force_field = AtomisticForceFieldPlan(
        system,
        AtomisticPotentialProgram(terms, coefficients=value.get("coefficients")),
        policy,
        provenance,
        constraint_plan=(
            DistanceConstraintPlan(
                maximum_iterations=value["constraint_policy"]["maximum_iterations"],
                tolerance=value["constraint_policy"]["tolerance"],
            )
            if value.get("constraint_policy") is not None
            else DistanceConstraintPlan()
            if topology.constraints.shape[0]
            else None
        ),
    )
    report = AtomisticInterchangeReport(
        value.get("source_format", "mapping"),
        units,
        tuple(term.name for term in terms),
    )
    return AtomisticInterchangeBundle(force_field, report)


def force_field_to_mapping(bundle: AtomisticInterchangeBundle, /) -> dict[str, Any]:
    if not isinstance(bundle, AtomisticInterchangeBundle):
        raise TypeError("bundle must be AtomisticInterchangeBundle.")
    plan = bundle.force_field
    system = plan.system
    topology = system.topology
    return {
        "unit_system": system.units.to_dict(),
        "source_format": "phydrax",
        "family": plan.provenance.family,
        "parameter_set": plan.provenance.parameter_set,
        "water_model": plan.provenance.water_model,
        "ion_model": plan.provenance.ion_model,
        "typing_source": plan.provenance.typing_source,
        "charge_source": plan.provenance.charge_source,
        "adapter_id": plan.provenance.adapter_id,
        "name": system.name,
        "particle_ids": np.asarray(system.particle_ids),
        "atomic_numbers": np.asarray(system.atomic_numbers),
        "masses": np.asarray(system.masses),
        "atom_type_ids": np.asarray(system.atom_type_ids),
        "charges": np.asarray(system.charges),
        "active_mask": np.asarray(system.active_mask),
        "mobile_mask": np.asarray(system.mobile_mask),
        "molecule_ids": np.asarray(system.molecule_ids),
        "region_ids": np.asarray(system.region_ids),
        "cell": system.cell,
        "coordinate_map": system.coordinate_map,
        "topology": {
            "bonds": np.asarray(topology.bonds),
            "angles": np.asarray(topology.angles),
            "torsions": np.asarray(topology.torsions),
            "impropers": np.asarray(topology.impropers),
            "constraints": np.asarray(topology.constraints),
            "constraint_distances": np.asarray(topology.constraint_distances),
            "pair_exceptions": np.asarray(topology.pair_exceptions),
            "lennard_jones_scales": np.asarray(topology.lennard_jones_scales),
            "electrostatic_scales": np.asarray(topology.electrostatic_scales),
            "bond_type_ids": np.asarray(topology.bond_type_ids),
            "angle_type_ids": np.asarray(topology.angle_type_ids),
            "torsion_type_ids": np.asarray(topology.torsion_type_ids),
            "improper_type_ids": np.asarray(topology.improper_type_ids),
        },
        "nonbonded": {
            "cutoff": plan.nonbonded.cutoff,
            "switch_distance": plan.nonbonded.switch_distance,
            "combining_rule": plan.nonbonded.combining_rule,
            "electrostatics": plan.nonbonded.electrostatics,
            "dispersion": plan.nonbonded.dispersion,
            "charge_neutrality": plan.nonbonded.charge_neutrality,
        },
        "potential_terms": tuple(
            _potential_term_to_mapping(term) for term in plan.potential.terms
        ),
        "constraint_policy": (
            None
            if plan.constraint_plan is None
            else {
                "maximum_iterations": plan.constraint_plan.maximum_iterations,
                "tolerance": plan.constraint_plan.tolerance,
            }
        ),
        "coefficients": np.asarray(plan.potential.coefficients),
        "bundle_id": bundle.source_id,
    }


def _require_optional(module: str):
    root = module.split(".", maxsplit=1)[0]
    if find_spec(root) is None or (module != root and find_spec(module) is None):
        raise ImportError(f"Atomistic interchange requires optional package {module!r}.")
    return import_module(module)


def from_openff_interchange(
    interchange, units: AtomisticUnitSystem, /
) -> AtomisticInterchangeBundle:
    _require_optional("openff.interchange")
    openmm_system = interchange.to_openmm(combine_nonbonded_forces=True)
    topology = interchange.topology
    positions = (
        None
        if interchange.positions is None
        else np.asarray(interchange.positions.m_as("angstrom"))
    )
    cell_vectors = (
        None if interchange.box is None else np.asarray(interchange.box.m_as("angstrom"))
    )
    atomic_numbers = np.asarray(
        [atom.atomic_number for atom in topology.atoms], dtype=np.int32
    )
    return from_openmm_system(
        openmm_system,
        units,
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell_vectors=cell_vectors,
        source_id=canonical_fingerprint(
            {"kind": "openff-interchange", "repr": repr(interchange)}
        ),
    )


def to_openmm_system(bundle: AtomisticInterchangeBundle, /):
    if not isinstance(bundle, AtomisticInterchangeBundle):
        raise TypeError("bundle must be AtomisticInterchangeBundle.")
    openmm = _require_optional("openmm")
    app = _require_optional("openmm.app")
    plan = bundle.force_field
    system_plan = plan.system
    factors = _openmm_unit_factors(system_plan.units)
    active = np.asarray(system_plan.active_mask)
    coordinate_map = system_plan.coordinate_map
    identity_map = (
        not coordinate_map.virtual_rules
        and coordinate_map.sites.capacity == active.size
        and np.array_equal(
            np.asarray(coordinate_map.physical_dof_indices), np.arange(active.size)
        )
    )
    unsupported = []
    warnings = []
    if not np.all(active):
        unsupported.append("inactive padding particles")
    if not identity_map:
        unsupported.append("non-identity interaction-site coordinate map")
    if plan.nonbonded.combining_rule != "lorentz-berthelot":
        unsupported.append(
            f"{plan.nonbonded.combining_rule} Lennard-Jones combining rule"
        )
    terms = tuple(plan.potential.terms)
    coefficients = np.asarray(plan.potential.coefficients)
    bonded = [
        (term, coefficient)
        for term, coefficient in zip(terms, coefficients, strict=True)
        if isinstance(term, HarmonicBondPotential)
    ]
    angles = [
        (term, coefficient)
        for term, coefficient in zip(terms, coefficients, strict=True)
        if isinstance(term, HarmonicAnglePotential)
    ]
    torsions = [
        (term, coefficient)
        for term, coefficient in zip(terms, coefficients, strict=True)
        if isinstance(term, PeriodicTorsionPotential)
        or (
            isinstance(term, GeneralForceFieldTerm)
            and term.kind is ForceFieldTermKind.TORSION_SERIES
        )
    ]
    lennard_jones = [
        (term, coefficient)
        for term, coefficient in zip(terms, coefficients, strict=True)
        if isinstance(term, LennardJonesPotential)
    ]
    if lennard_jones and system_plan.cell is None:
        warnings.append(
            "Finite-cutoff PhydraX Lennard-Jones is exported through OpenMM NoCutoff."
        )
    electrostatic = [
        (term, coefficient)
        for term, coefficient in zip(terms, coefficients, strict=True)
        if isinstance(
            term,
            (DirectCoulombPotential, EwaldReferencePotential, ParticleMeshEwaldPotential),
        )
        or (
            isinstance(term, GeneralForceFieldTerm)
            and term.kind is ForceFieldTermKind.REACTION_FIELD
        )
    ]
    if electrostatic and isinstance(
        electrostatic[0][0], (EwaldReferencePotential, ParticleMeshEwaldPotential)
    ):
        warnings.append(
            "Explicit PhydraX reciprocal parameters are delegated to OpenMM auto-tuning."
        )
    recognized = {
        id(term)
        for collection in (bonded, angles, torsions, lennard_jones, electrostatic)
        for term, _ in collection
    }
    unsupported.extend(term.name for term in terms if id(term) not in recognized)
    if any(len(collection) > 1 for collection in (bonded, angles, lennard_jones)):
        unsupported.append("multiple same-kind OpenMM force terms")
    if len(electrostatic) > 1:
        unsupported.append("multiple electrostatic terms")
    if electrostatic and electrostatic[0][1] != 1.0:
        unsupported.append("scaled electrostatic term")
    if (
        lennard_jones
        and electrostatic
        and lennard_jones[0][0].force_group != electrostatic[0][0].force_group
    ):
        unsupported.append("separate Lennard-Jones and electrostatic force groups")
    report = AtomisticInterchangeReport(
        "phydrax-openmm-export",
        system_plan.units,
        tuple(term.name for term in terms if id(term) in recognized),
        tuple(unsupported),
        tuple(warnings),
        source_energy_unit=KILOJOULE_PER_MOLE,
        avogadro_constant_set_id=system_plan.units.constant_set_id,
    )
    report.require_complete()

    exported = openmm.System()
    for mass in np.asarray(system_plan.masses):
        exported.addParticle(float(mass) * factors["mass_to_dalton"] * openmm.unit.dalton)
    topology = app.Topology()
    chain = topology.addChain("A")
    residues = {}
    atoms = []
    for index, (particle_id, atomic_number, molecule_id) in enumerate(
        zip(
            np.asarray(system_plan.particle_ids),
            np.asarray(system_plan.atomic_numbers),
            np.asarray(system_plan.molecule_ids),
            strict=True,
        )
    ):
        molecule = int(molecule_id)
        if molecule not in residues:
            residues[molecule] = topology.addResidue(
                f"M{molecule}", chain, id=str(molecule)
            )
        element = app.element.Element.getByAtomicNumber(int(atomic_number))
        atoms.append(
            topology.addAtom(
                f"{element.symbol}{index + 1}",
                element,
                residues[molecule],
                id=str(int(particle_id)),
            )
        )
    slot_by_id = {
        int(identifier): index
        for index, identifier in enumerate(np.asarray(system_plan.particle_ids))
    }
    topology_plan = system_plan.topology
    for left_id, right_id in np.asarray(topology_plan.bonds):
        topology.addBond(
            atoms[slot_by_id[int(left_id)]], atoms[slot_by_id[int(right_id)]]
        )

    length_to_nm = factors["length_to_nanometer"]
    energy_to_kj = factors["energy_to_kilojoule"]
    if system_plan.cell is not None:
        vectors = np.asarray(system_plan.cell.vectors) * length_to_nm
        box = tuple(openmm.Vec3(*row) for row in vectors) * openmm.unit.nanometer
        exported.setDefaultPeriodicBoxVectors(*box)
        topology.setPeriodicBoxVectors(box)
    for term, coefficient in bonded:
        force = openmm.HarmonicBondForce()
        for route, type_id in zip(
            np.asarray(topology_plan.bonds),
            np.asarray(topology_plan.bond_type_ids),
            strict=True,
        ):
            force.addBond(
                slot_by_id[int(route[0])],
                slot_by_id[int(route[1])],
                float(term.equilibrium_distance[type_id])
                * length_to_nm
                * openmm.unit.nanometer,
                float(term.stiffness[type_id])
                * coefficient
                * energy_to_kj
                / length_to_nm**2
                * openmm.unit.kilojoule_per_mole
                / openmm.unit.nanometer**2,
            )
        force.setForceGroup(term.force_group)
        exported.addForce(force)
    for term, coefficient in angles:
        force = openmm.HarmonicAngleForce()
        for route, type_id in zip(
            np.asarray(topology_plan.angles),
            np.asarray(topology_plan.angle_type_ids),
            strict=True,
        ):
            force.addAngle(
                *(slot_by_id[int(value)] for value in route),
                float(term.equilibrium_angle[type_id]) * openmm.unit.radian,
                float(term.stiffness[type_id])
                * coefficient
                * energy_to_kj
                * openmm.unit.kilojoule_per_mole
                / openmm.unit.radian**2,
            )
        force.setForceGroup(term.force_group)
        exported.addForce(force)
    for term, coefficient in torsions:
        force = openmm.PeriodicTorsionForce()
        if isinstance(term, GeneralForceFieldTerm):
            amplitude, periodicity, phase, mask = map(np.asarray, term.arrays)
            for route_index, route in enumerate(np.asarray(term.route_indices)):
                row = 0 if amplitude.shape[0] == 1 else route_index
                for column in np.flatnonzero(mask[row]):
                    force.addTorsion(
                        *(int(slot) for slot in route),
                        int(periodicity[row, column]),
                        float(phase[row, column]) * openmm.unit.radian,
                        float(amplitude[row, column] * mask[row, column])
                        * coefficient
                        * energy_to_kj
                        * openmm.unit.kilojoule_per_mole,
                    )
        else:
            routes = topology_plan.impropers if term.improper else topology_plan.torsions
            type_ids = (
                topology_plan.improper_type_ids
                if term.improper
                else topology_plan.torsion_type_ids
            )
            for route, type_id in zip(
                np.asarray(routes), np.asarray(type_ids), strict=True
            ):
                force.addTorsion(
                    *(slot_by_id[int(value)] for value in route),
                    int(term.periodicity[type_id]),
                    float(term.phase[type_id]) * openmm.unit.radian,
                    float(term.amplitude[type_id])
                    * coefficient
                    * energy_to_kj
                    * openmm.unit.kilojoule_per_mole,
                )
        force.setForceGroup(term.force_group)
        exported.addForce(force)
    for route, distance in zip(
        np.asarray(topology_plan.constraints),
        np.asarray(topology_plan.constraint_distances),
        strict=True,
    ):
        exported.addConstraint(
            slot_by_id[int(route[0])],
            slot_by_id[int(route[1])],
            float(distance) * length_to_nm * openmm.unit.nanometer,
        )

    if lennard_jones or electrostatic:
        force = openmm.NonbondedForce()
        lj_term, lj_coefficient = lennard_jones[0] if lennard_jones else (None, 0.0)
        atom_types = np.asarray(system_plan.atom_type_ids)
        charges = (
            np.asarray(system_plan.charges) * factors["charge_to_elementary"]
            if electrostatic
            else np.zeros(active.shape)
        )
        if lj_term is None:
            epsilon = np.zeros(active.shape)
            sigma = np.ones(active.shape)
        else:
            epsilon = (
                np.asarray(lj_term.epsilon)[atom_types] * lj_coefficient * energy_to_kj
            )
            sigma = np.asarray(lj_term.sigma)[atom_types] * length_to_nm
        for charge, sigma_value, epsilon_value in zip(
            charges, sigma, epsilon, strict=True
        ):
            force.addParticle(
                float(charge) * openmm.unit.elementary_charge,
                float(sigma_value) * openmm.unit.nanometer,
                float(epsilon_value) * openmm.unit.kilojoule_per_mole,
            )
        cutoff_nm = plan.nonbonded.cutoff * length_to_nm
        if system_plan.cell is None:
            method = (
                openmm.NonbondedForce.CutoffNonPeriodic
                if plan.nonbonded.electrostatics == "reaction-field"
                else openmm.NonbondedForce.NoCutoff
            )
        else:
            method = {
                "pme": openmm.NonbondedForce.PME,
                "ewald": openmm.NonbondedForce.Ewald,
                "reaction-field": openmm.NonbondedForce.CutoffPeriodic,
                "direct": openmm.NonbondedForce.CutoffPeriodic,
            }[plan.nonbonded.electrostatics]
        force.setNonbondedMethod(method)
        if method != openmm.NonbondedForce.NoCutoff:
            force.setCutoffDistance(cutoff_nm * openmm.unit.nanometer)
        if lennard_jones and plan.nonbonded.switch_distance is not None:
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(
                plan.nonbonded.switch_distance * length_to_nm * openmm.unit.nanometer
            )
        if electrostatic and isinstance(electrostatic[0][0], GeneralForceFieldTerm):
            force.setReactionFieldDielectric(float(electrostatic[0][0].arrays[0]))
        for route, lj_scale, electrostatic_scale in zip(
            np.asarray(topology_plan.pair_exceptions),
            np.asarray(topology_plan.lennard_jones_scales),
            np.asarray(topology_plan.electrostatic_scales),
            strict=True,
        ):
            left, right = (slot_by_id[int(value)] for value in route)
            mixed_sigma = 0.5 * (sigma[left] + sigma[right])
            mixed_epsilon = np.sqrt(epsilon[left] * epsilon[right]) * lj_scale
            force.addException(
                left,
                right,
                charges[left]
                * charges[right]
                * electrostatic_scale
                * openmm.unit.elementary_charge**2,
                mixed_sigma * openmm.unit.nanometer,
                mixed_epsilon * openmm.unit.kilojoule_per_mole,
            )
        force.setForceGroup(
            lennard_jones[0][0].force_group
            if lennard_jones
            else electrostatic[0][0].force_group
        )
        exported.addForce(force)
    return exported, topology, report


def to_openff_interchange(
    bundle: AtomisticInterchangeBundle,
    /,
    *,
    positions: ArrayLike | None = None,
):
    interchange_module = _require_optional("openff.interchange")
    ensure_quantity = _require_optional("openff.units.openmm").ensure_quantity
    openmm = _require_optional("openmm")
    system, topology, _ = to_openmm_system(bundle)
    factors = _openmm_unit_factors(bundle.force_field.system.units)
    converted_positions = (
        None
        if positions is None
        else ensure_quantity(
            np.asarray(positions)
            * factors["length_to_nanometer"]
            * openmm.unit.nanometer,
            "openff",
        )
    )
    converted_box = (
        None
        if bundle.force_field.system.cell is None
        else ensure_quantity(topology.getPeriodicBoxVectors(), "openff")
    )
    return interchange_module.Interchange.from_openmm(
        system=system,
        topology=topology,
        positions=converted_positions,
        box_vectors=converted_box,
    )


def from_openmm_system(
    system,
    units: AtomisticUnitSystem,
    /,
    *,
    atomic_numbers: ArrayLike,
    positions: ArrayLike | None = None,
    cell_vectors: ArrayLike | None = None,
    cutoff: float = 10.0,
    source_id: str = "openmm-system",
) -> AtomisticInterchangeBundle:
    openmm = _require_optional("openmm")
    factors = _openmm_unit_factors(units)
    length_factor = factors["length_from_angstrom"]
    energy_factor = factors["energy_from_kilojoule"]
    count = int(system.getNumParticles())
    numbers = np.asarray(atomic_numbers, dtype=np.int32)
    if numbers.shape != (count,):
        raise ValueError("atomic_numbers must match OpenMM particle count.")
    masses = (
        np.asarray(
            [
                system.getParticleMass(index).value_in_unit(openmm.unit.dalton)
                for index in range(count)
            ]
        )
        * factors["mass_from_dalton"]
    )
    charges = np.zeros((count,))
    sigma = np.ones((count,))
    epsilon = np.zeros((count,))
    bonds, bond_k, bond_r0 = [], [], []
    angles, angle_k, angle_theta = [], [], []
    torsions = {}
    exceptions, lj_scales, electrostatic_scales = [], [], []
    supported, unsupported, warnings = (
        [],
        tuple(
            f"virtual site {index}"
            for index in range(count)
            if system.isVirtualSite(index)
        ),
        [],
    )
    unsupported = list(unsupported)
    electrostatics = "direct"
    dispersion = "cutoff"
    cutoff_value = float(cutoff)
    switch_distance = None
    reaction_field_dielectric = 78.5
    ewald_alpha = None
    grid_shape = None
    reciprocal_extent = None
    periodic_method = False
    for force in system.getForces():
        name = force.__class__.__name__
        if name == "NonbondedForce":
            supported.append(name)
            method = force.getNonbondedMethod()
            periodic_method = method in (
                openmm.NonbondedForce.CutoffPeriodic,
                openmm.NonbondedForce.Ewald,
                openmm.NonbondedForce.PME,
                openmm.NonbondedForce.LJPME,
            )
            if method == openmm.NonbondedForce.NoCutoff:
                electrostatics = "direct"
                warnings.append(
                    "OpenMM NoCutoff Lennard-Jones interactions are bounded by the "
                    "explicit adapter cutoff in PhydraX."
                )
            elif method in (
                openmm.NonbondedForce.CutoffNonPeriodic,
                openmm.NonbondedForce.CutoffPeriodic,
            ):
                electrostatics = "reaction-field"
                reaction_field_dielectric = float(force.getReactionFieldDielectric())
            elif method == openmm.NonbondedForce.Ewald:
                electrostatics = "ewald"
            else:
                electrostatics = "pme"
                if method == openmm.NonbondedForce.LJPME:
                    unsupported.append("NonbondedForce.LJPME dispersion")
            if method != openmm.NonbondedForce.NoCutoff:
                cutoff_value = (
                    force.getCutoffDistance().value_in_unit(openmm.unit.angstrom)
                    * length_factor
                )
            if force.getUseSwitchingFunction():
                switch_distance = (
                    force.getSwitchingDistance().value_in_unit(openmm.unit.angstrom)
                    * length_factor
                )
            if electrostatics in ("ewald", "pme"):
                alpha, nx, ny, nz = force.getPMEParameters()
                alpha_value = alpha.value_in_unit(openmm.unit.angstrom**-1)
                if alpha_value > 0.0:
                    ewald_alpha = alpha_value / length_factor
                else:
                    tolerance = float(force.getEwaldErrorTolerance())
                    ewald_alpha = np.sqrt(-np.log(2.0 * tolerance)) / cutoff_value
                    warnings.append("OpenMM automatic Ewald alpha was reconstructed.")
                if electrostatics == "pme":
                    requested = (int(nx), int(ny), int(nz))
                    grid_shape = tuple(value if value >= 4 else 32 for value in requested)
                    if any(value < 4 for value in requested):
                        warnings.append("OpenMM automatic PME grid was reconstructed.")
                else:
                    reciprocal_extent = 6
                    warnings.append("OpenMM Ewald reciprocal extent was reconstructed.")
            for index in range(count):
                charge, sigma_value, epsilon_value = force.getParticleParameters(index)
                charges[index] = (
                    charge.value_in_unit(openmm.unit.elementary_charge)
                    * factors["charge_from_elementary"]
                )
                sigma[index] = (
                    sigma_value.value_in_unit(openmm.unit.angstrom) * length_factor
                )
                epsilon[index] = (
                    epsilon_value.value_in_unit(openmm.unit.kilojoule_per_mole)
                    * energy_factor
                )
            for index in range(force.getNumExceptions()):
                left, right, charge_product, _, epsilon_value = (
                    force.getExceptionParameters(index)
                )
                exceptions.append((left, right))
                base_charge = charges[left] * charges[right]
                electrostatic_scales.append(
                    0.0
                    if base_charge == 0.0
                    else charge_product.value_in_unit(openmm.unit.elementary_charge**2)
                    * factors["charge_from_elementary"] ** 2
                    / base_charge
                )
                base_epsilon = np.sqrt(epsilon[left] * epsilon[right])
                lj_scales.append(
                    0.0
                    if base_epsilon == 0.0
                    else epsilon_value.value_in_unit(openmm.unit.kilojoule_per_mole)
                    * energy_factor
                    / base_epsilon
                )
        elif name == "HarmonicBondForce":
            supported.append(name)
            for index in range(force.getNumBonds()):
                left, right, length, stiffness = force.getBondParameters(index)
                bonds.append((left, right))
                bond_r0.append(length.value_in_unit(openmm.unit.angstrom) * length_factor)
                bond_k.append(
                    stiffness.value_in_unit(
                        openmm.unit.kilojoule_per_mole / openmm.unit.angstrom**2
                    )
                    * energy_factor
                    / length_factor**2
                )
        elif name == "HarmonicAngleForce":
            supported.append(name)
            for index in range(force.getNumAngles()):
                a, b, c, theta, stiffness = force.getAngleParameters(index)
                angles.append((a, b, c))
                angle_theta.append(theta.value_in_unit(openmm.unit.radian))
                angle_k.append(
                    stiffness.value_in_unit(
                        openmm.unit.kilojoule_per_mole / openmm.unit.radian**2
                    )
                    * energy_factor
                )
        elif name == "PeriodicTorsionForce":
            supported.append(name)
            for index in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, amplitude = force.getTorsionParameters(
                    index
                )
                torsions.setdefault((a, b, c, d), []).append(
                    (
                        amplitude.value_in_unit(openmm.unit.kilojoule_per_mole)
                        * energy_factor,
                        int(periodicity),
                        phase.value_in_unit(openmm.unit.radian),
                    )
                )
        elif name in ("CMMotionRemover", "MonteCarloBarostat"):
            supported.append(name)
        else:
            unsupported.append(name)
    report = AtomisticInterchangeReport(
        "openmm",
        units,
        supported,
        unsupported,
        tuple(warnings),
        source_energy_unit=KILOJOULE_PER_MOLE,
        avogadro_constant_set_id=units.constant_set_id,
    )
    report.require_complete()
    # Fourier components share one topological quartet, not duplicate interactions.
    torsion_shape = (len(torsions), max(map(len, torsions.values()), default=0))
    torsion_k = np.zeros(torsion_shape)
    torsion_n = np.ones(torsion_shape, dtype=np.int32)
    torsion_phase = np.zeros(torsion_shape)
    torsion_mask = np.zeros(torsion_shape)
    for route_index, series in enumerate(torsions.values()):
        for term_index, (amplitude, periodicity, phase) in enumerate(series):
            torsion_k[route_index, term_index] = amplitude
            torsion_n[route_index, term_index] = periodicity
            torsion_phase[route_index, term_index] = phase
            torsion_mask[route_index, term_index] = 1.0
    constraints = []
    constraint_distances = []
    for index in range(system.getNumConstraints()):
        left, right, distance = system.getConstraintParameters(index)
        constraints.append((left, right))
        constraint_distances.append(
            distance.value_in_unit(openmm.unit.angstrom) * length_factor
        )
    resolved_cell = None
    if cell_vectors is not None:
        resolved_cell = np.asarray(cell_vectors, dtype=float) * length_factor
    elif periodic_method:
        resolved_cell = (
            np.asarray(
                [
                    vector.value_in_unit(openmm.unit.angstrom)
                    for vector in system.getDefaultPeriodicBoxVectors()
                ],
                dtype=float,
            )
            * length_factor
        )
    cell = (
        None
        if resolved_cell is None
        else PeriodicCell(resolved_cell, periodic_axes=(True, True, True))
    )
    nonbonded = {
        "cutoff": cutoff_value,
        "switch_distance": switch_distance,
        "epsilon": epsilon,
        "sigma": sigma,
        "electrostatics": electrostatics,
        "dispersion": dispersion,
        "reaction_field_dielectric": reaction_field_dielectric,
    }
    if ewald_alpha is not None:
        nonbonded["ewald_alpha"] = ewald_alpha
    if grid_shape is not None:
        nonbonded["grid_shape"] = grid_shape
    if reciprocal_extent is not None:
        nonbonded["reciprocal_extent"] = reciprocal_extent
    mapping = {
        "unit_system": units.to_dict(),
        "source_format": "openmm",
        "family": "openmm",
        "parameter_set": source_id,
        "adapter_id": "openmm",
        "particle_ids": np.arange(count, dtype=np.int64),
        "atomic_numbers": numbers,
        "masses": masses,
        "atom_type_ids": np.arange(count, dtype=np.int32),
        "charges": charges,
        "positions": None
        if positions is None
        else np.asarray(positions, dtype=float) * length_factor,
        "cell": cell,
        "topology": {
            "bonds": np.asarray(bonds, dtype=np.int64).reshape((-1, 2)),
            "angles": np.asarray(angles, dtype=np.int64).reshape((-1, 3)),
            "torsions": np.asarray(tuple(torsions), dtype=np.int64).reshape((-1, 4)),
            "constraints": np.asarray(constraints, dtype=np.int64).reshape((-1, 2)),
            "constraint_distances": np.asarray(constraint_distances),
            "pair_exceptions": np.asarray(exceptions, dtype=np.int64).reshape((-1, 2)),
            "lennard_jones_scales": np.asarray(lj_scales),
            "electrostatic_scales": np.asarray(electrostatic_scales),
            "bond_type_ids": np.arange(len(bonds), dtype=np.int32),
            "angle_type_ids": np.arange(len(angles), dtype=np.int32),
            "torsion_type_ids": np.arange(len(torsions), dtype=np.int32),
        },
        "parameters": {
            "bond_stiffness": np.asarray(bond_k) if bond_k else None,
            "bond_length": np.asarray(bond_r0) if bond_r0 else None,
            "angle_stiffness": np.asarray(angle_k) if angle_k else None,
            "angle_value": np.asarray(angle_theta) if angle_theta else None,
            "torsion_amplitude": torsion_k if torsions else None,
            "torsion_periodicity": torsion_n if torsions else None,
            "torsion_phase": torsion_phase if torsions else None,
            "torsion_mask": torsion_mask if torsions else None,
        },
        "nonbonded": nonbonded,
    }
    mapping["parameters"] = {
        key: value for key, value in mapping["parameters"].items() if value is not None
    }
    bundle = force_field_from_mapping(mapping)
    return AtomisticInterchangeBundle(bundle.force_field, report)


def from_parmed_structure(
    structure, units: AtomisticUnitSystem, /, *, cutoff: float = 10.0
) -> AtomisticInterchangeBundle:
    parmed = _require_optional("parmed")
    openmm = _require_optional("openmm")
    factors = _openmm_unit_factors(units)
    length_factor = factors["length_from_angstrom"]
    energy_factor = 4.184 * factors["energy_from_kilojoule"]
    atoms = tuple(structure.atoms)
    if not atoms:
        raise ValueError("ParmEd structure contains no atoms.")
    unsupported = []
    supported = ["atoms", "Lennard-Jones", "charges"]
    if structure.has_NBFIX():
        unsupported.append("NBFIX")
    if structure.urey_bradleys:
        unsupported.append("Urey-Bradley terms")
    if structure.rb_torsions:
        unsupported.append("Ryckaert-Bellemans torsions")
    if structure.cmaps:
        unsupported.append("CMAP terms")

    atom_type_keys = tuple(
        (str(atom.type), float(atom.sigma), float(atom.epsilon)) for atom in atoms
    )
    unique_types = tuple(dict.fromkeys(atom_type_keys))
    type_by_key = {key: index for index, key in enumerate(unique_types)}
    atom_type_ids = np.asarray([type_by_key[key] for key in atom_type_keys])
    sigma = np.asarray(
        [1.0 if key[1] <= 0.0 else key[1] * length_factor for key in unique_types]
    )
    epsilon = np.asarray([max(key[2], 0.0) * energy_factor for key in unique_types])
    particle_ids = np.arange(len(atoms), dtype=np.int64)
    molecule_ids = np.asarray([atom.residue.idx for atom in atoms], dtype=np.int32)
    charges = (
        np.asarray([atom.charge for atom in atoms]) * factors["charge_from_elementary"]
    )
    masses = np.asarray([atom.mass for atom in atoms]) * factors["mass_from_dalton"]

    bond_routes, bond_stiffness, bond_length = [], [], []
    for bond in structure.bonds:
        if bond.type is None:
            unsupported.append("unparameterized bond")
            continue
        bond_routes.append((bond.atom1.idx, bond.atom2.idx))
        bond_stiffness.append(2.0 * bond.type.k * energy_factor / length_factor**2)
        bond_length.append(bond.type.req * length_factor)
    if bond_routes:
        supported.append("harmonic bonds")

    angle_routes, angle_stiffness, angle_values = [], [], []
    for angle in structure.angles:
        if angle.type is None:
            unsupported.append("unparameterized angle")
            continue
        angle_routes.append((angle.atom1.idx, angle.atom2.idx, angle.atom3.idx))
        angle_stiffness.append(2.0 * angle.type.k * energy_factor)
        angle_values.append(np.deg2rad(angle.type.theteq))
    if angle_routes:
        supported.append("harmonic angles")

    proper_records: dict[tuple[int, int, int, int], list] = {}
    improper_records: dict[tuple[int, int, int, int], list] = {}
    dihedral_type_list = parmed.topologyobjects.DihedralTypeList
    for dihedral in structure.dihedrals:
        if dihedral.type is None:
            unsupported.append("unparameterized dihedral")
            continue
        route = (
            dihedral.atom1.idx,
            dihedral.atom2.idx,
            dihedral.atom3.idx,
            dihedral.atom4.idx,
        )
        values = (
            tuple(dihedral.type)
            if isinstance(dihedral.type, dihedral_type_list)
            else (dihedral.type,)
        )
        destination = improper_records if dihedral.improper else proper_records
        destination.setdefault(route, []).extend(values)
    if proper_records or improper_records:
        supported.append("periodic torsions")

    exceptions: dict[tuple[int, int], tuple[float, float]] = {}
    for left, right in bond_routes:
        exceptions[tuple(sorted((left, right)))] = (0.0, 0.0)
    for left, _, right in angle_routes:
        exceptions[tuple(sorted((left, right)))] = (0.0, 0.0)
    for route, values in proper_records.items():
        scales = tuple(
            (
                1.0 / value.scnb if value.scnb else 0.0,
                1.0 / value.scee if value.scee else 0.0,
            )
            for value in values
        )
        if any(scale != scales[0] for scale in scales[1:]):
            unsupported.append("inconsistent multi-term 1-4 scales")
        exceptions[tuple(sorted((route[0], route[3])))] = scales[0]
    for adjustment in structure.adjusts:
        left, right = adjustment.atom1.idx, adjustment.atom2.idx
        left_type, right_type = atom_type_ids[left], atom_type_ids[right]
        mixed_sigma = 0.5 * (sigma[left_type] + sigma[right_type])
        observed_sigma = adjustment.type.sigma * length_factor
        if not np.isclose(mixed_sigma, observed_sigma):
            unsupported.append("pair-specific 1-4 sigma")
        base_epsilon = np.sqrt(epsilon[left_type] * epsilon[right_type])
        lj_scale = (
            0.0
            if base_epsilon == 0.0
            else adjustment.type.epsilon * energy_factor / base_epsilon
        )
        exceptions[tuple(sorted((left, right)))] = (
            lj_scale,
            float(adjustment.type.chgscale),
        )

    periodic = structure.box_vectors is not None
    cell = None
    if periodic:
        vectors = np.asarray(
            [
                vector.value_in_unit(openmm.unit.angstrom)
                for vector in structure.box_vectors
            ]
        )
        cell = PeriodicCell(vectors * length_factor)
    topology = MolecularTopologyPlan(
        bonds=np.asarray(bond_routes, dtype=np.int64).reshape((-1, 2)),
        angles=np.asarray(angle_routes, dtype=np.int64).reshape((-1, 3)),
        torsions=np.asarray(tuple(proper_records), dtype=np.int64).reshape((-1, 4)),
        impropers=np.asarray(tuple(improper_records), dtype=np.int64).reshape((-1, 4)),
        pair_exceptions=np.asarray(tuple(exceptions), dtype=np.int64).reshape((-1, 2)),
        lennard_jones_scales=np.asarray([exceptions[pair][0] for pair in exceptions]),
        electrostatic_scales=np.asarray([exceptions[pair][1] for pair in exceptions]),
        bond_type_ids=np.arange(len(bond_routes), dtype=np.int32),
        angle_type_ids=np.arange(len(angle_routes), dtype=np.int32),
    )
    system = AtomisticSystemPlan(
        particle_ids,
        np.asarray([atom.atomic_number for atom in atoms]),
        masses,
        units,
        atom_type_ids=atom_type_ids,
        charges=charges,
        molecule_ids=molecule_ids,
        topology=topology,
        cell=cell,
        name=str(structure.title or "parmed-structure"),
    )
    terms = []
    if bond_routes:
        terms.append(HarmonicBondPotential(bond_stiffness, bond_length))
    if angle_routes:
        terms.append(HarmonicAnglePotential(angle_stiffness, angle_values))
    cutoff_value = float(cutoff) * length_factor
    if np.any(epsilon > 0.0):
        terms.append(LennardJonesPotential(epsilon, sigma, cutoff_value))

    def torsion_term(records, name):
        if not records:
            return None
        maximum = max(len(values) for values in records.values())
        amplitude = np.zeros((len(records), maximum))
        periodicity = np.ones((len(records), maximum), dtype=np.int32)
        phase = np.zeros((len(records), maximum))
        mask = np.zeros((len(records), maximum))
        for route_index, values in enumerate(records.values()):
            for term_index, value in enumerate(values):
                amplitude[route_index, term_index] = value.phi_k * energy_factor
                periodicity[route_index, term_index] = int(value.per)
                phase[route_index, term_index] = np.deg2rad(value.phase)
                mask[route_index, term_index] = 1.0
        return PeriodicTorsionSeriesPotential(
            amplitude,
            periodicity,
            phase,
            mask,
            np.asarray(tuple(records), dtype=np.int32),
            name=name,
        )

    for term in (
        torsion_term(proper_records, "parmed-proper-torsions"),
        torsion_term(improper_records, "parmed-improper-torsions"),
    ):
        if term is not None:
            terms.append(term)
    electrostatics = "pme" if periodic else "direct"
    if np.any(charges != 0.0):
        if periodic:
            alpha = np.sqrt(-np.log(1.0e-4)) / cutoff_value
            terms.append(
                ParticleMeshEwaldPotential(
                    alpha,
                    cutoff_value,
                    (32, 32, 32),
                    neutrality="uniform-background",
                )
            )
        else:
            terms.append(DirectCoulombPotential())
    if not terms:
        raise ValueError("ParmEd structure contains no supported energy terms.")
    policy = AtomisticNonbondedPolicy(
        cutoff_value,
        electrostatics=electrostatics,
        charge_neutrality="uniform-background" if periodic else "require-neutral",
    )
    source_digest = canonical_fingerprint(
        {
            "kind": "parmed-source",
            "title": str(structure.title or "structure"),
            "atoms": len(atoms),
            "residues": len(structure.residues),
        }
    )
    provenance = AtomisticForceFieldProvenance(
        "parmed",
        (source_digest,),
        "parmed",
        str(structure.title or "structure"),
        adapter_id="parmed",
    )
    report = AtomisticInterchangeReport(
        "parmed",
        units,
        tuple(supported),
        tuple(dict.fromkeys(unsupported)),
        source_energy_unit=KILOCALORIE_PER_MOLE,
        avogadro_constant_set_id=units.constant_set_id,
    )
    report.require_complete()
    return AtomisticInterchangeBundle(
        AtomisticForceFieldPlan(
            system, AtomisticPotentialProgram(terms), policy, provenance
        ),
        report,
    )


__all__ = [
    "force_field_from_mapping",
    "force_field_to_mapping",
    "from_openff_interchange",
    "from_openmm_system",
    "from_parmed_structure",
    "to_openmm_system",
    "to_openff_interchange",
]

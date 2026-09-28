#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned Geant4 oracles for showers, Cherenkov photons, and optical transport.

Geant4 (Geant4 Software License, SPDX ``LicenseRef-Geant4``;
https://geant4.web.cern.ch/; Agostinelli et al., Nucl. Instrum. Meth. A 506,
250 (2003); Allison et al., Nucl. Instrum. Meth. A 835, 186 (2016)) is run by a
caller-pinned Python interpreter with the statically linked ``geant4_pybind``
bindings (https://github.com/HaarigerHarald/geant4_pybind) through
:func:`phydrax.run_pinned_command`; nothing is imported into this process and
no Geant4 source is copied. The pinned executable's ``version`` is the
``geant4_pybind`` distribution version, which is verified against the running
interpreter; the bundled Geant4 release and the dataset directories present
are returned as evidence. The electromagnetic runs need ``G4EMLOW``,
``G4ENSDFSTATE``, and ``PhotonEvaporation`` (nuclear level data is read at
particle construction). Validated live: ``geant4_pybind`` 0.1.3 bundling
Geant4 11.4.p01 with ``G4EMLOW8.8``, ``G4ENSDFSTATE3.0``, and
``PhotonEvaporation6.1.2``.

Inputs are JSON decks in SI lengths and eV energies generated from Phydrax
plans; the provider converts to Geant4's internal units. The child environment
carries only ``GEANT4_DATA_DIR`` (the dataset directory of
:class:`Geant4Provider`).

Supported subset, everything else refused before running:

- :func:`run_geant4_shower`: an :class:`~phydrax.solver.EMShowerPlan` on a
  homogeneous voxel geometry (one material; the photon and charged libraries
  name it identically), with the elemental composition of an explicitly named
  Geant4 NIST material at the Phydrax mass density. Every active primary of
  the photon and charged batches is one Geant4 event, in identity order, from
  a position inside the closed geometry box. One Geant4 electromagnetic
  constructor is registered (no hadronic, photonuclear, or electronuclear
  physics). The bremsstrahlung production threshold is the charged
  ``photon_stack`` minimum energy and the e−/e+ production threshold the charged
  cutoff, both realized as Geant4 range cuts found by bisection and at least
  Geant4's 990 eV table edge. Energy deposits are binned in depth along +z at
  the step midpoint.
- :func:`run_geant4_cherenkov`: an SI :class:`~phydrax.optics.transport.OpticalPhotonSourcePlan`
  with Cherenkov emission only and one singly charged parent with one active
  constant-speed step, realized as an electron (charge −1) or positron (+1)
  of the same speed at the step start inside a sphere of the step's radiating
  medium whose radius is the step length. ``RINDEX`` is the plan's refractive
  index on its wavelength nodes; the bulk composition is an explicitly named
  Geant4 NIST material. Only Cherenkov photons of the primary are scored, at
  creation, and then killed; optical transport is not run.
- :func:`run_geant4_optical`: an SI
  :class:`~phydrax.optics.transport.OpticalMonteCarloPlan` whose surfaces form
  a planar stack along ``z``. Every surface id is one plane normal to ``z``
  covering the common rectangle of all vertices exactly once. The two
  outermost planes are detector (acceptance cosine 0) or absorber planes, and
  every inner plane is a dielectric interface of a
  :class:`~phydrax.optics.transport.UnifiedSurfaceModel` with the lossless
  ``"polished"`` finish (Geant4 UNIFIED polished ``dielectric_dielectric``
  border surfaces in both directions). All interfaces reflect and transmit;
  there is no surface-table attenuation; and the detector response is
  :class:`~phydrax.optics.transport.UnitDetectorResponse`. The medium is a
  :class:`~phydrax.optics.transport.SpectralOpticalMedium` with bulk absorption
  and Rayleigh scattering only, each finite at every wavelength node or at
  none. Photons have unit weight and linear polarization and start strictly
  inside a layer of their medium, at wavelengths on the grid. Each photon is
  one Geant4 event with only the absorption, Rayleigh, and boundary processes
  active.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, Literal, TypeAlias

import numpy as np

from ..._external_runtime import (
    ExternalExecutionPolicy,
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import ElectromagneticScaleContract, RelativityScaleContract
from ..._validation import canonical_identifier, positive_finite_float, positive_integer
from ...equations._charged_radiation_interactions import ChargedRadiationParticleKind
from ...interchange._report import AdapterLoss, AdapterReport, AdapterStatus
from ...optics.geometric._nonsequential import (
    NonSequentialBranchMode,
    NonSequentialSurfaceKind,
)
from ...optics.transport._optical_media import _Channel, SpectralOpticalMedium
from ...optics.transport._optical_monte_carlo import (
    OpticalMonteCarloPlan,
    OpticalPhotonState,
    UnitDetectorResponse,
)
from ...optics.transport._optical_sources import (
    ChargedOpticalSteps,
    OpticalPhotonSourcePlan,
)
from ...optics.transport._optical_surfaces import UnifiedSurfaceModel
from ...solver._em_shower import EMShowerPlan, ShowerParticleBatch
from ...typing import parse


Geant4ElectromagneticPhysics: TypeAlias = Literal[
    "standard", "standard-option4", "livermore", "penelope"
]

_DECK = "input.json"
_SUMMARY = "summary.json"
_SHOWER_SCRIPT_NAME = "geant4_shower.py"
_CHERENKOV_SCRIPT_NAME = "geant4_cherenkov.py"
_DEPOSITS = "deposits.npy"
_PHOTONS = "photons.npy"
_PHOTON_COLUMNS = 17
_OPTICAL_SCRIPT_NAME = "geant4_optical.py"
_OUTCOMES = "outcomes.npy"
_OUTCOME_COLUMNS = 14
# Planarity and coverage of the plan's triangles, relative to the scene size.
_PLANE_TOLERANCE = 1.0e-12
# Geant4 exit points lie on a terminal plane to its geometric tolerance.
_EXIT_TOLERANCE = 1.0e-9
_LINEAR_POLARIZATION_TOLERANCE = 1.0e-9
# Lowest energy of Geant4's range-to-energy tables; thresholds below it clamp.
_GEANT4_LOWEST_THRESHOLD_EV = 990.0
_THRESHOLD_RELATIVE_TOLERANCE = 1.0e-6
_DIRECTION_TOLERANCE = 1.0e-12
_WORLD_MARGIN_FRACTION = 0.01
_POLICY = ExternalExecutionPolicy(
    inherit_environment=False, allowed_environment_variables=("GEANT4_DATA_DIR",)
)

_COMMON_SCRIPT = b"""
import json, os, sys
from importlib.metadata import version
import numpy as np
import geant4_pybind as g4

document = json.load(open(sys.argv[1]))
current_event = [0]


def identity():
    return {
        "geant4_pybind": version("geant4_pybind"),
        "geant4": g4.G4Version.strip("$ ").removeprefix("Name: ").strip(),
        "datasets": sorted(os.listdir(os.environ["GEANT4_DATA_DIR"])),
    }


class Events(g4.G4UserEventAction):
    def BeginOfEventAction(self, event):
        current_event[0] = event.GetEventID()


g4.G4Random.setTheSeed(document["seed"])
runs = g4.G4RunManagerFactory.CreateRunManager(g4.G4RunManagerType.Serial)
"""

_SHOWER_SCRIPT = (
    _COMMON_SCRIPT
    + b"""
lower = np.asarray(document["lower_m"], dtype=float)
upper = np.asarray(document["upper_m"], dtype=float)
center = 0.5 * (lower + upper)
half = 0.5 * (upper - lower)
bins = document["depth_bins"]
primaries = document["primaries"]
deposits = np.zeros((len(primaries), bins))


class Detector(g4.G4VUserDetectorConstruction):
    def Construct(self):
        nist = g4.G4NistManager.Instance()
        vacuum = nist.FindOrBuildMaterial("G4_Galactic")
        self.material = nist.BuildMaterialWithNewDensity(
            "phydrax_shower_material",
            document["nist_material"],
            document["density_kg_m3"] * g4.kg / g4.m3,
        )
        world_box = g4.G4Box("world", *((half + document["world_margin_m"]) * g4.m))
        world_logical = g4.G4LogicalVolume(world_box, vacuum, "world")
        world = g4.G4PVPlacement(None, g4.G4ThreeVector(), world_logical, "world", None, False, 0)
        slab_box = g4.G4Box("slab", *(half * g4.m))
        slab_logical = g4.G4LogicalVolume(slab_box, self.material, "slab")
        g4.G4PVPlacement(None, g4.G4ThreeVector(), slab_logical, "slab", world_logical, False, 0)
        return world


detector = Detector()


def realized_threshold(particle, range_m):
    table = g4.G4ProductionCutsTable.GetProductionCutsTable()
    definition = g4.G4ParticleTable.GetParticleTable().FindParticle(particle)
    return table.ConvertRangeToEnergy(definition, detector.material, range_m * g4.m) / g4.eV


class Primary(g4.G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = g4.G4ParticleGun(1)
        self.table = g4.G4ParticleTable.GetParticleTable()

    def GeneratePrimaries(self, event):
        record = primaries[event.GetEventID()]
        self.gun.SetParticleDefinition(self.table.FindParticle(record["particle"]))
        self.gun.SetParticleEnergy(record["kinetic_energy_ev"] * g4.eV)
        position = (np.asarray(record["position_m"]) - center) * g4.m
        self.gun.SetParticlePosition(g4.G4ThreeVector(*position))
        self.gun.SetParticleMomentumDirection(g4.G4ThreeVector(*record["direction"]))
        self.gun.GeneratePrimaryVertex(event)


class Stepping(g4.G4UserSteppingAction):
    def UserSteppingAction(self, step):
        deposit = step.GetTotalEnergyDeposit()
        if deposit <= 0.0:
            return
        pre = step.GetPreStepPoint()
        if pre.GetTouchable().GetVolume().GetName() != "slab":
            return
        midpoint = 0.5 * (pre.GetPosition().z + step.GetPostStepPoint().GetPosition().z)
        fraction = (midpoint / g4.m + half[2]) / (2.0 * half[2])
        index = min(max(int(fraction * bins), 0), bins - 1)
        deposits[current_event[0], index] += deposit / g4.eV


physics = g4.G4VModularPhysicsList()
physics.RegisterPhysics(getattr(g4, document["physics_constructor"])())
runs.SetUserInitialization(detector)
runs.SetUserInitialization(physics)
runs.SetUserAction(Primary())
runs.SetUserAction(Events())
runs.SetUserAction(Stepping())
runs.Initialize()
# Range-to-energy converters exist only once a run has built the cuts table.
runs.BeamOn(0)
range_cuts = {}
for particle, energy in document["production_thresholds_ev"].items():
    low, high = np.log(1.0e-9), np.log(10.0)
    for _ in range(100):
        middle = 0.5 * (low + high)
        if realized_threshold(particle, np.exp(middle)) < energy:
            low = middle
        else:
            high = middle
    range_cuts[particle] = float(np.exp(high))
    physics.SetCutValue(range_cuts[particle] * g4.m, particle)
runs.BeamOn(len(primaries))
np.save("deposits.npy", deposits)
summary = identity()
summary.update(
    {
        "radiation_length_m": detector.material.GetRadlen() / g4.m,
        "density_kg_m3": detector.material.GetDensity() / (g4.kg / g4.m3),
        "range_cuts_m": range_cuts,
        "realized_thresholds_ev": {
            particle: realized_threshold(particle, value)
            for particle, value in range_cuts.items()
        },
    }
)
json.dump(summary, open("summary.json", "w"))
"""
)

_CHERENKOV_SCRIPT = (
    _COMMON_SCRIPT
    + b"""
radius = document["path_length_m"]
events = document["events"]
photons = []
paths = np.zeros(events)


class Detector(g4.G4VUserDetectorConstruction):
    def Construct(self):
        nist = g4.G4NistManager.Instance()
        vacuum = nist.FindOrBuildMaterial("G4_Galactic")
        self.material = nist.FindOrBuildMaterial(document["nist_material"])
        properties = g4.G4MaterialPropertiesTable()
        energies = g4.G4doubleVector([value * g4.eV for value in document["photon_energies_ev"]])
        properties.AddProperty("RINDEX", energies, g4.G4doubleVector(document["refractive_indices"]))
        self.material.SetMaterialPropertiesTable(properties)
        world_box = g4.G4Box("world", *(3 * [2.0 * radius * g4.m]))
        world_logical = g4.G4LogicalVolume(world_box, vacuum, "world")
        world = g4.G4PVPlacement(None, g4.G4ThreeVector(), world_logical, "world", None, False, 0)
        sphere = g4.G4Orb("medium", radius * g4.m)
        medium_logical = g4.G4LogicalVolume(sphere, self.material, "medium")
        g4.G4PVPlacement(None, g4.G4ThreeVector(), medium_logical, "medium", world_logical, False, 0)
        return world


class Primary(g4.G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = g4.G4ParticleGun(1)
        table = g4.G4ParticleTable.GetParticleTable()
        self.gun.SetParticleDefinition(table.FindParticle(document["particle"]))
        self.gun.SetParticleEnergy(document["kinetic_energy_ev"] * g4.eV)
        self.gun.SetParticlePosition(g4.G4ThreeVector())
        self.gun.SetParticleMomentumDirection(g4.G4ThreeVector(*document["direction"]))

    def GeneratePrimaries(self, event):
        self.gun.GeneratePrimaryVertex(event)


class Stepping(g4.G4UserSteppingAction):
    def UserSteppingAction(self, step):
        if step.GetTrack().GetTrackID() != 1:
            return
        pre = step.GetPreStepPoint()
        if pre.GetTouchable().GetVolume().GetName() != "medium":
            return
        event = current_event[0]
        paths[event] += step.GetStepLength() / g4.m
        # G4Cerenkov builds the cone about the step chord at the mean step speed.
        parent = step.GetDeltaPosition().unit()
        beta_pre = pre.GetBeta()
        beta_post = step.GetPostStepPoint().GetBeta()
        for secondary in step.GetSecondaryInCurrentStep():
            if secondary.GetCreatorProcess().GetProcessName() != "Cerenkov":
                continue
            direction = secondary.GetMomentumDirection()
            polarization = secondary.GetPolarization()
            position = secondary.GetPosition()
            photons.append(
                (
                    event,
                    secondary.GetKineticEnergy() / g4.eV,
                    direction.x, direction.y, direction.z,
                    polarization.x, polarization.y, polarization.z,
                    parent.x, parent.y, parent.z,
                    beta_pre, beta_post,
                    position.x / g4.m, position.y / g4.m, position.z / g4.m,
                    secondary.GetGlobalTime() / g4.s,
                )
            )


class Stacking(g4.G4UserStackingAction):
    def ClassifyNewTrack(self, track):
        if track.GetDefinition().GetParticleName() == "opticalphoton":
            return g4.G4ClassificationOfNewTrack.fKill
        return g4.G4ClassificationOfNewTrack.fUrgent


physics = g4.G4VModularPhysicsList()
physics.RegisterPhysics(g4.G4EmStandardPhysics_option4())
physics.RegisterPhysics(g4.G4OpticalPhysics())
runs.SetUserInitialization(Detector())
runs.SetUserInitialization(physics)
runs.SetUserAction(Primary())
runs.SetUserAction(Events())
runs.SetUserAction(Stepping())
runs.SetUserAction(Stacking())
ui = g4.G4UImanager.GetUIpointer()
for process in document["inactive_optical_processes"]:
    ui.ApplyCommand("/process/optical/processActivation %s false" % process)
ui.ApplyCommand("/process/optical/cerenkov/setMaxPhotons %d" % document["maximum_photons_per_step"])
ui.ApplyCommand("/process/optical/cerenkov/setMaxBetaChange %r" % document["maximum_beta_change_percent"])
runs.Initialize()
runs.BeamOn(events)
np.save("photons.npy", np.asarray(photons, dtype=np.float64).reshape(-1, 17))
summary = identity()
summary["path_lengths_m"] = paths.tolist()
json.dump(summary, open("summary.json", "w"))
"""
)

_OPTICAL_SCRIPT = (
    _COMMON_SCRIPT
    + b"""
layers = document["layers"]
photons = document["photons"]
x0, x1, y0, y1 = document["rectangle_m"]
records = np.full((len(photons), 14), np.nan)
scatters = np.zeros(len(photons))


def vector(values):
    return g4.G4ThreeVector(*values)


def properties(record):
    table = g4.G4MaterialPropertiesTable()
    energies = g4.G4doubleVector([value * g4.eV for value in record["photon_energies_ev"]])
    table.AddProperty("RINDEX", energies, g4.G4doubleVector(record["refractive_indices"]))
    for key in ("ABSLENGTH", "RAYLEIGH"):
        if record[key] is not None:
            lengths = g4.G4doubleVector([value * g4.m for value in record[key]])
            table.AddProperty(key, energies, lengths)
    return table


class Detector(g4.G4VUserDetectorConstruction):
    def Construct(self):
        nist = g4.G4NistManager.Instance()
        vacuum = nist.FindOrBuildMaterial("G4_Galactic")
        self.keep = []
        extent = document["world_half_m"]
        world_box = g4.G4Box("world", extent * g4.m, extent * g4.m, extent * g4.m)
        world_logical = g4.G4LogicalVolume(world_box, vacuum, "world")
        world = g4.G4PVPlacement(None, g4.G4ThreeVector(), world_logical, "world", None, False, 0)
        materials = {}
        for key, record in document["media"].items():
            material = nist.BuildMaterialWithNewDensity(
                "phydrax_optical_medium_" + key, "G4_Galactic", vacuum.GetDensity()
            )
            table = properties(record)
            material.SetMaterialPropertiesTable(table)
            materials[key] = material
            self.keep.append(table)
        placements = []
        for index, layer in enumerate(layers):
            low, high = layer["z_m"]
            box = g4.G4Box(
                "layer%d" % index,
                0.5 * (x1 - x0) * g4.m,
                0.5 * (y1 - y0) * g4.m,
                0.5 * (high - low) * g4.m,
            )
            logical = g4.G4LogicalVolume(box, materials[str(layer["medium"])], "layer%d" % index)
            center = vector((0.5 * (x0 + x1) * g4.m, 0.5 * (y0 + y1) * g4.m, 0.5 * (low + high) * g4.m))
            placements.append(
                g4.G4PVPlacement(None, center, logical, "layer%d" % index, world_logical, False, 0)
            )
            self.keep += [box, logical]
        for index in range(len(layers) - 1):
            surface = g4.G4OpticalSurface(
                "interface%d" % index,
                g4.G4OpticalSurfaceModel.unified,
                g4.G4OpticalSurfaceFinish.polished,
                g4.G4SurfaceType.dielectric_dielectric,
            )
            below, above = placements[index], placements[index + 1]
            self.keep += [
                surface,
                g4.G4LogicalBorderSurface("up%d" % index, below, above, surface),
                g4.G4LogicalBorderSurface("down%d" % index, above, below, surface),
            ]
        self.keep += placements
        return world


class Primary(g4.G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = g4.G4ParticleGun(1)
        table = g4.G4ParticleTable.GetParticleTable()
        self.gun.SetParticleDefinition(table.FindParticle("opticalphoton"))

    def GeneratePrimaries(self, event):
        record = photons[event.GetEventID()]
        self.gun.SetParticleEnergy(record["energy_ev"] * g4.eV)
        self.gun.SetParticlePosition(vector([value * g4.m for value in record["position_m"]]))
        self.gun.SetParticleMomentumDirection(vector(record["direction"]))
        self.gun.SetParticlePolarization(vector(record["polarization"]))
        self.gun.SetParticleTime(record["time_s"] * g4.s)
        self.gun.GeneratePrimaryVertex(event)


class Stepping(g4.G4UserSteppingAction):
    def UserSteppingAction(self, step):
        event = current_event[0]
        post = step.GetPostStepPoint()
        process = post.GetProcessDefinedStep().GetProcessName()
        if process == "OpRayleigh":
            scatters[event] += 1
        track = step.GetTrack()
        if track.GetTrackStatus() != g4.G4TrackStatus.fStopAndKill:
            return
        volume = post.GetTouchable().GetVolume()
        name = "" if volume is None else volume.GetName()
        if name == "world":
            code = 0
        elif process == "OpAbsorption":
            code = 1
        elif process == "OpBoundary" and name.startswith("layer"):
            code = 2
        else:
            code = 3
        pre = step.GetPreStepPoint().GetTouchable().GetVolume().GetName()
        position = post.GetPosition()
        direction = post.GetMomentumDirection()
        polarization = post.GetPolarization()
        records[event] = (
            code,
            int(pre.removeprefix("layer")),
            position.x / g4.m, position.y / g4.m, position.z / g4.m,
            direction.x, direction.y, direction.z,
            polarization.x, polarization.y, polarization.z,
            post.GetGlobalTime() / g4.s,
            scatters[event],
            track.GetKineticEnergy() / g4.eV,
        )


physics = g4.G4VModularPhysicsList()
physics.RegisterPhysics(g4.G4OpticalPhysics())
runs.SetUserInitialization(Detector())
runs.SetUserInitialization(physics)
runs.SetUserAction(Primary())
runs.SetUserAction(Events())
runs.SetUserAction(Stepping())
ui = g4.G4UImanager.GetUIpointer()
for process in document["inactive_optical_processes"]:
    ui.ApplyCommand("/process/optical/processActivation %s false" % process)
runs.Initialize()
runs.BeamOn(len(photons))
np.save("outcomes.npy", records)
json.dump(identity(), open("summary.json", "w"))
"""
)


@dataclass(frozen=True, slots=True)
class Geant4Provider:
    """A pinned Python interpreter with ``geant4_pybind`` and its datasets.

    ``executable.version`` is the ``geant4_pybind`` distribution version and
    ``data_directory`` the ``GEANT4_DATA_DIR`` holding the Geant4 datasets of
    that release (external oracle only).
    """

    executable: PinnedExecutable
    data_directory: str

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable Python interpreter.")
        if not isinstance(self.data_directory, str):
            raise TypeError("data_directory must be a path string.")
        directory = Path(self.data_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(
                "data_directory must be an existing Geant4 dataset directory."
            )
        object.__setattr__(self, "data_directory", str(directory))


@dataclass(frozen=True, slots=True)
class Geant4ShowerResult:
    """Geant4 depth-resolved energy deposition of one shower plan.

    ``deposited_energy[event, bin]`` (eV) is binned on ``depth_edges`` (m,
    absolute ``z`` of the plan geometry); events follow the primaries in
    identity order with kinetic energies ``primary_energies`` (eV).
    ``radiation_length`` (m) and ``mass_density`` (kg/m³) are Geant4's values
    for the realized material, ``production_thresholds`` (eV) the realized
    Geant4 thresholds per particle, and ``range_cuts`` (m) the range cuts that
    realize them.
    """

    depth_edges: np.ndarray
    deposited_energy: np.ndarray
    primary_energies: np.ndarray
    radiation_length: float
    mass_density: float
    production_thresholds: tuple[tuple[str, float], ...]
    range_cuts: tuple[tuple[str, float], ...]
    provider_version: str
    geant4_release: str
    datasets: tuple[str, ...]
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport

    @property
    def depth_edges_radiation_lengths(self) -> np.ndarray:
        """Bin edges as depth ``t = (z − z_lower) / X0``."""
        return (self.depth_edges - self.depth_edges[0]) / self.radiation_length

    @property
    def total_deposited_energy(self) -> np.ndarray:
        """Energy deposited in the slab per event (eV)."""
        return np.sum(self.deposited_energy, axis=1)

    def longitudinal_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin centers ``t`` (X0), mean ``dE/dt`` (eV per X0), and its standard error."""
        edges = self.depth_edges_radiation_lengths
        density = self.deposited_energy / np.diff(edges)[None, :]
        count = density.shape[0]
        error = (
            np.std(density, axis=0, ddof=1) / math.sqrt(count)
            if count > 1
            else np.full(density.shape[1], np.inf)
        )
        return 0.5 * (edges[1:] + edges[:-1]), np.mean(density, axis=0), error


@dataclass(frozen=True, slots=True)
class Geant4CherenkovResult:
    """Cherenkov photons Geant4 emitted from the primary, at creation.

    Per photon: ``events``, vacuum ``wavelengths`` (m), unit ``directions`` and
    ``polarizations``, the unit chord ``parent_directions`` and endpoint speeds
    ``parent_start_beta``/``parent_end_beta`` of the emitting primary step
    (Geant4 builds the cone about the chord at the mean speed), emission
    ``positions`` (m, plan frame), and
    ``times`` (s, from the plan step start time). ``path_lengths[event]`` is
    the primary's path length in the medium (m).
    """

    events: np.ndarray
    wavelengths: np.ndarray
    directions: np.ndarray
    polarizations: np.ndarray
    parent_directions: np.ndarray
    parent_start_beta: np.ndarray
    parent_end_beta: np.ndarray
    positions: np.ndarray
    times: np.ndarray
    path_lengths: np.ndarray
    provider_version: str
    geant4_release: str
    datasets: tuple[str, ...]
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport

    @property
    def photon_counts(self) -> np.ndarray:
        """Cherenkov photons of the primary per event."""
        return np.bincount(self.events, minlength=self.path_lengths.shape[0])

    @property
    def emission_cosines(self) -> np.ndarray:
        """Cosine between each photon and the emitting primary direction."""
        return np.sum(self.directions * self.parent_directions, axis=1)


def _physics_constructor(physics: Geant4ElectromagneticPhysics, /) -> str:
    match physics:
        case "standard":
            return "G4EmStandardPhysics"
        case "standard-option4":
            return "G4EmStandardPhysics_option4"
        case "livermore":
            return "G4EmLivermorePhysics"
        case "penelope":
            return "G4EmPenelopePhysics"
        case _:
            assert_never(physics)


def _host(value: object, dtype: type[np.generic], /) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _geant4_particle(batch: ShowerParticleBatch, kind: int, /) -> str:
    match batch.species:
        case "photon":
            return "gamma"
        case "charged":
            match ChargedRadiationParticleKind(kind):
                case ChargedRadiationParticleKind.ELECTRON:
                    return "e-"
                case ChargedRadiationParticleKind.POSITRON:
                    return "e+"
                case unknown:
                    assert_never(unknown)
        case _:
            assert_never(batch.species)


def _shower_primaries(
    plan: EMShowerPlan,
    photons: ShowerParticleBatch | None,
    charged: ShowerParticleBatch | None,
    /,
) -> list[dict[str, object]]:
    records: list[tuple[int, dict[str, object]]] = []
    lower = _host(plan.photon_transport.geometry.lower, np.float64)
    upper = _host(plan.photon_transport.geometry.upper, np.float64)
    for batch, species in ((photons, "photon"), (charged, "charged")):
        if batch is None:
            continue
        if not isinstance(batch, ShowerParticleBatch) or batch.species != species:
            raise TypeError(
                f"{species} primaries must be a {species} ShowerParticleBatch."
            )
        active = _host(batch.active, np.bool_)
        positions = _host(batch.positions, np.float64)[active]
        directions = _host(batch.directions, np.float64)[active]
        energies = _host(batch.energies, np.float64)[active]
        kinds = _host(batch.kinds, np.int32)[active]
        identities = (_host(batch.id_hi, np.uint64)[active] << np.uint64(32)) | _host(
            batch.id_lo, np.uint64
        )[active]
        if np.any(~np.isfinite(energies)) or np.any(energies <= 0.0):
            raise ValueError("Geant4 primaries need finite positive kinetic energies.")
        if np.any(
            np.abs(np.linalg.norm(directions, axis=1) - 1.0) > _DIRECTION_TOLERANCE
        ):
            raise ValueError("Geant4 primaries need unit directions.")
        if np.any(positions < lower) or np.any(positions > upper):
            raise ValueError("Geant4 primaries must start inside the plan geometry.")
        for index in range(energies.shape[0]):
            records.append(
                (
                    int(identities[index]),
                    {
                        "particle": _geant4_particle(batch, int(kinds[index])),
                        "kinetic_energy_ev": float(energies[index]),
                        "position_m": positions[index].tolist(),
                        "direction": directions[index].tolist(),
                    },
                )
            )
    keys = [key for key, _ in records]
    if not records:
        raise ValueError("The Geant4 shower needs at least one active primary.")
    if len(set(keys)) != len(keys):
        raise ValueError("Geant4 shower primaries need distinct identities.")
    return [record for _, record in sorted(records, key=lambda item: item[0])]


def geant4_shower_input(
    plan: EMShowerPlan,
    photons: ShowerParticleBatch | None,
    charged: ShowerParticleBatch | None,
    /,
    *,
    nist_material: str,
    depth_bins: int,
    physics: Geant4ElectromagneticPhysics = "standard-option4",
    random_seed: int = 1,
) -> dict[str, bytes]:
    """Translate a homogeneous-slab shower plan into the Geant4 deck and script."""
    if not isinstance(plan, EMShowerPlan):
        raise TypeError("plan must be an EMShowerPlan.")
    material_name = canonical_identifier(nist_material, "nist_material")
    physics_ = parse(physics, Geant4ElectromagneticPhysics, "physics")
    bins = positive_integer(depth_bins, "depth_bins")
    seed = positive_integer(random_seed, "random_seed")
    photon_transport = plan.photon_transport
    charged_transport = plan.charged_transport
    geometry = photon_transport.geometry
    materials = np.unique(_host(geometry.material_indices, np.int32))
    if materials.shape[0] != 1:
        raise ValueError("The Geant4 shower oracle needs a homogeneous geometry.")
    material = int(materials[0])
    material_id = photon_transport.cross_sections.material_ids[material]
    if charged_transport.materials.material_ids[material] != material_id:
        raise ValueError(
            "Photon and charged libraries must name the shower material identically."
        )
    photon_stack = charged_transport.photon_stack
    electron_stack = photon_transport.electron_stack
    if photon_stack is None or electron_stack is None:
        raise ValueError("Shower transports must both have a secondary stack attached.")
    thresholds = {
        "gamma": photon_stack.minimum_energy,
        "e-": charged_transport.cutoff_energy_ev,
        "e+": charged_transport.cutoff_energy_ev,
    }
    if min(thresholds.values()) < _GEANT4_LOWEST_THRESHOLD_EV:
        raise ValueError(
            "Geant4 production thresholds cannot lie below its 990 eV table edge."
        )
    lower = _host(geometry.lower, np.float64)
    upper = _host(geometry.upper, np.float64)
    document = {
        "plan_id": plan.plan_id,
        "material_id": material_id,
        "nist_material": material_name,
        "density_kg_m3": float(
            _host(photon_transport.cross_sections.mass_density_kg_per_m3, np.float64)[
                material
            ]
        ),
        "lower_m": lower.tolist(),
        "upper_m": upper.tolist(),
        "world_margin_m": _WORLD_MARGIN_FRACTION * float(np.max(upper - lower)),
        "depth_bins": bins,
        "physics": physics_,
        "physics_constructor": _physics_constructor(physics_),
        "production_thresholds_ev": thresholds,
        "photon_cutoff_ev": photon_transport.cutoff_energy,
        "electron_stack_minimum_ev": electron_stack.minimum_energy,
        "maximum_generations": plan.maximum_generations,
        "seed": seed,
        "primaries": _shower_primaries(plan, photons, charged),
    }
    return {
        _SHOWER_SCRIPT_NAME: _SHOWER_SCRIPT,
        _DECK: json.dumps(document, sort_keys=True).encode(),
    }


def _document(deck: Mapping[str, bytes], /) -> dict:
    if not isinstance(deck, Mapping) or _DECK not in deck:
        raise TypeError("deck must be the Geant4 input mapping with input.json.")
    return json.loads(deck[_DECK])


def _summary(summary: bytes, executable: PinnedExecutable, /) -> dict:
    if not isinstance(executable, PinnedExecutable):
        raise TypeError("executable must be a PinnedExecutable.")
    record = json.loads(summary)
    if record["geant4_pybind"] != executable.version:
        raise ValueError(
            f"The interpreter runs geant4_pybind {record['geant4_pybind']}, "
            f"not the pinned {executable.version}."
        )
    return record


def _array(data: bytes, /) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _shower_losses(document: dict, summary: dict, /) -> tuple[AdapterLoss, ...]:
    realized = summary["realized_thresholds_ev"]
    return (
        AdapterLoss(
            "physics.hadronic",
            "export",
            "dropped",
            "Only the Geant4 electromagnetic constructor "
            f"{document['physics_constructor']} is registered: hadronic, "
            "photonuclear, and electronuclear channels are absent, as in the "
            "Phydrax shower model.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "physics.interaction_models",
            "export",
            "transformed",
            "Phydrax tabulated photon coefficients, stopping and scattering powers, "
            "and bremsstrahlung routes are replaced by the Geant4 models of "
            f"{document['physics_constructor']} for the declared composition.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            f"material.{document['material_id']}",
            "export",
            "synthesized",
            f"Elemental composition is Geant4 NIST {document['nist_material']} at "
            f"the Phydrax mass density {document['density_kg_m3']!r} kg/m3.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "cuts.production_thresholds",
            "export",
            "transformed",
            "Phydrax energy thresholds (photon stack "
            f"{document['production_thresholds_ev']['gamma']!r} eV, charged cutoff "
            f"{document['production_thresholds_ev']['e-']!r} eV) are realized as "
            f"Geant4 range cuts {summary['range_cuts_m']!r} m giving "
            f"{realized!r} eV.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "cuts.tracking",
            "export",
            "dropped",
            "Geant4 tracks particles to rest: the Phydrax photon cutoff "
            f"{document['photon_cutoff_ev']!r} eV and photon electron-stack "
            f"threshold {document['electron_stack_minimum_ev']!r} eV have no "
            "Geant4 counterpart.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "plan.maximum_generations",
            "export",
            "dropped",
            "Geant4 transports the complete shower; the Phydrax bound of "
            f"{document['maximum_generations']} generations and its batch "
            "capacities are not imposed.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "deposited_energy",
            "import",
            "transformed",
            "Each Geant4 step deposit is binned at its midpoint depth.",
            changes_interpretation=False,
        ),
    )


def read_geant4_shower(
    deck: Mapping[str, bytes],
    summary: bytes,
    deposits: bytes,
    /,
    *,
    executable: PinnedExecutable,
) -> Geant4ShowerResult:
    """Read the Geant4 shower outputs of ``deck`` into Phydrax units."""
    document = _document(deck)
    record = _summary(summary, executable)
    values = _array(deposits)
    primaries = document["primaries"]
    bins = document["depth_bins"]
    if values.shape != (len(primaries), bins) or not np.all(np.isfinite(values)):
        raise ValueError("Geant4 deposits do not match the deck or are nonfinite.")
    for particle, requested in document["production_thresholds_ev"].items():
        realized = record["realized_thresholds_ev"][particle]
        if abs(realized - requested) > _THRESHOLD_RELATIVE_TOLERANCE * requested:
            raise ValueError(
                f"Geant4 realized a {particle} threshold of {realized} eV, "
                f"not the plan's {requested} eV."
            )
    edges = np.linspace(
        document["lower_m"][2], document["upper_m"][2], bins + 1, dtype=np.float64
    )
    energies = np.asarray(
        [primary["kinetic_energy_ev"] for primary in primaries], dtype=np.float64
    )
    digest = hashlib.sha256(deposits).hexdigest()
    thresholds = tuple(sorted(record["realized_thresholds_ev"].items()))
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "geant4-shower-deposits",
        "Geant4ShowerResult",
        source_id=digest,
        target_id=canonical_fingerprint(
            {
                "kind": "geant4-shower-result",
                "plan_id": document["plan_id"],
                "deposited_energy": array_tree_fingerprint(values),
                "depth_edges": array_tree_fingerprint(edges),
                "radiation_length": record["radiation_length_m"],
                "thresholds": thresholds,
            }
        ),
        coordinate_mapping=(
            "Geant4 frame = plan frame translated to the geometry center",
            "depth = z of the plan geometry along +z",
        ),
        preserved_fields=(
            "primary species, kinetic energies, positions, directions",
            "geometry bounds",
            "mass density",
        ),
        assumptions=(
            "energies in eV, lengths in m",
            "events are primaries in identity order",
        ),
        losses=_shower_losses(document, record),
    )
    return Geant4ShowerResult(
        depth_edges=edges,
        deposited_energy=values,
        primary_energies=energies,
        radiation_length=float(record["radiation_length_m"]),
        mass_density=float(record["density_kg_m3"]),
        production_thresholds=thresholds,
        range_cuts=tuple(sorted(record["range_cuts_m"].items())),
        provider_version=record["geant4_pybind"],
        geant4_release=record["geant4"],
        datasets=tuple(record["datasets"]),
        executable_sha256=executable.sha256,
        license_id=executable.license_id,
        output_sha256=digest,
        report=report,
    )


def _run(
    provider: Geant4Provider,
    inputs: dict[str, bytes],
    script: str,
    artifact: str,
    destination: str | Path,
    timeout: float,
    maximum_output_bytes: int,
    /,
) -> tuple[bytes, bytes]:
    if not isinstance(provider, Geant4Provider):
        raise TypeError("provider must be a Geant4Provider.")
    run = run_pinned_command(
        provider.executable,
        (script, _DECK),
        inputs=inputs,
        outputs=(_SUMMARY,),
        timeout=timeout,
        environment={"GEANT4_DATA_DIR": provider.data_directory},
        execution_policy=_POLICY,
        artifacts=PinnedFileOutputs(
            str(destination),
            (PinnedFileRequest(artifact, maximum_output_bytes),),
            maximum_output_bytes,
        ),
    )
    published = run.file_artifact(artifact)
    data = Path(published.location).read_bytes()
    if hashlib.sha256(data).hexdigest() != published.sha256:
        raise ValueError("The published Geant4 artifact changed after publication.")
    return run.output(_SUMMARY), data


def run_geant4_shower(
    provider: Geant4Provider,
    plan: EMShowerPlan,
    photons: ShowerParticleBatch | None,
    charged: ShowerParticleBatch | None,
    destination: str | Path,
    /,
    *,
    nist_material: str,
    depth_bins: int,
    physics: Geant4ElectromagneticPhysics = "standard-option4",
    random_seed: int = 1,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 28,
) -> Geant4ShowerResult:
    """Run pinned Geant4 on the translated shower and read its depth profile."""
    inputs = geant4_shower_input(
        plan,
        photons,
        charged,
        nist_material=nist_material,
        depth_bins=depth_bins,
        physics=physics,
        random_seed=random_seed,
    )
    summary, deposits = _run(
        provider,
        inputs,
        _SHOWER_SCRIPT_NAME,
        _DEPOSITS,
        destination,
        timeout,
        maximum_output_bytes,
    )
    return read_geant4_shower(inputs, summary, deposits, executable=provider.executable)


def geant4_cherenkov_input(
    plan: OpticalPhotonSourcePlan,
    steps: ChargedOpticalSteps,
    /,
    *,
    nist_material: str,
    events: int,
    random_seed: int = 1,
    maximum_photons_per_step: int = 100,
    maximum_beta_change_percent: float = 10.0,
) -> dict[str, bytes]:
    """Translate a one-step Cherenkov source into the Geant4 deck and script."""
    if not isinstance(plan, OpticalPhotonSourcePlan):
        raise TypeError("plan must be an OpticalPhotonSourcePlan.")
    if not isinstance(steps, ChargedOpticalSteps):
        raise TypeError("steps must be ChargedOpticalSteps.")
    material_name = canonical_identifier(nist_material, "nist_material")
    count = positive_integer(events, "events")
    seed = positive_integer(random_seed, "random_seed")
    photons_per_step = positive_integer(
        maximum_photons_per_step, "maximum_photons_per_step"
    )
    beta_change = float(maximum_beta_change_percent)
    if not 0.0 < beta_change <= 100.0:
        raise ValueError("maximum_beta_change_percent must lie in (0, 100].")
    cherenkov = plan.cherenkov
    if cherenkov is None or plan.scintillation is not None:
        raise ValueError("The Geant4 Cherenkov oracle needs Cherenkov emission only.")
    scale = ElectromagneticScaleContract.si()
    if plan.relativity.scale_id != RelativityScaleContract.si().scale_id:
        raise ValueError("The Geant4 Cherenkov oracle needs the SI relativity scale.")
    if steps.speed_of_light != plan.speed_of_light:
        raise ValueError("Steps and plan must share one exact speed of light.")
    active = _host(steps.active, np.bool_)
    if active.shape[0] != 1 or int(np.sum(active)) != 1:
        raise ValueError("The Geant4 Cherenkov oracle needs one parent with one step.")
    step = int(np.flatnonzero(active[0])[0])
    charge = float(_host(steps.charge_numbers, np.float64)[0])
    if abs(charge) != 1.0:
        raise ValueError("The Geant4 Cherenkov oracle needs a singly charged parent.")
    if float(_host(steps.multiplicities, np.float64)[0]) != 1.0 or not bool(
        _host(steps.complete, np.bool_)[0]
    ):
        raise ValueError("The Geant4 Cherenkov parent must be complete and unweighted.")
    start_beta = float(_host(steps.start_beta, np.float64)[0, step])
    if float(_host(steps.end_beta, np.float64)[0, step]) != start_beta:
        raise ValueError("The Geant4 Cherenkov step must have constant speed.")
    if not 0.0 < start_beta < 1.0:
        raise ValueError("The Geant4 Cherenkov step speed must lie in (0, 1).")
    medium = int(_host(steps.medium_indices, np.int32)[0, step])
    if not 0 <= medium < plan.medium_count or not bool(
        _host(cherenkov.radiating_media, np.bool_)[medium]
    ):
        raise ValueError("The Geant4 Cherenkov step must lie in a radiating medium.")
    start = _host(steps.start_positions, np.float64)[0, step]
    chord = _host(steps.end_positions, np.float64)[0, step] - start
    length = float(np.linalg.norm(chord))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("The Geant4 Cherenkov step must have a positive length.")
    rest_energy_ev = float(
        scale.electron_mass * scale.speed_of_light**2 / scale.elementary_charge
    )
    lorentz = 1.0 / math.sqrt((1.0 - start_beta) * (1.0 + start_beta))
    wavelengths_m = (
        _host(cherenkov.wavelengths, np.float64) * plan.length_per_wavelength_unit
    )
    photon_energies_ev = (
        plan.photon_energy_length / wavelengths_m / float(scale.elementary_charge)
    )
    indices = _host(cherenkov.refractive_indices, np.float64)[medium]
    document = {
        "plan_id": plan.plan_id,
        "medium": medium,
        "nist_material": material_name,
        "photon_energies_ev": photon_energies_ev[::-1].tolist(),
        "refractive_indices": indices[::-1].tolist(),
        "particle": "e-" if charge < 0.0 else "e+",
        "kinetic_energy_ev": rest_energy_ev * (lorentz - 1.0),
        "beta": start_beta,
        "start_m": start.tolist(),
        "start_time_s": float(_host(steps.start_times, np.float64)[0, step]),
        "direction": (chord / length).tolist(),
        "path_length_m": length,
        "events": count,
        "seed": seed,
        "maximum_photons_per_step": photons_per_step,
        "maximum_beta_change_percent": beta_change,
        "inactive_optical_processes": [
            "Scintillation",
            "OpAbsorption",
            "OpRayleigh",
            "OpMieHG",
            "OpBoundary",
            "OpWLS",
            "OpWLS2",
        ],
    }
    return {
        _CHERENKOV_SCRIPT_NAME: _CHERENKOV_SCRIPT,
        _DECK: json.dumps(document, sort_keys=True).encode(),
    }


def _cherenkov_losses(document: dict, /) -> tuple[AdapterLoss, ...]:
    return (
        AdapterLoss(
            "medium.bulk",
            "export",
            "synthesized",
            f"Bulk composition and density are Geant4 NIST {document['nist_material']};"
            " Phydrax optical media carry only optical tables.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            f"medium.{document['medium']}.refractive_index",
            "export",
            "transformed",
            "The refractive index is interpolated linearly in photon energy on the "
            "plan's wavelength nodes; Phydrax interpolates the Frank-Tamm density "
            "linearly in wavelength (second order in the node spacing).",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "steps.speed_law",
            "export",
            "transformed",
            f"The prescribed constant-speed step (beta {document['beta']!r}) becomes "
            f"a Geant4 {document['particle']} transported self-consistently with "
            "energy loss and multiple scattering; yields are compared per path length.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "optical.transport",
            "import",
            "dropped",
            "Photons are scored at creation and killed; absorption, scattering, "
            "and boundary processes are not run.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "optical.secondaries",
            "import",
            "dropped",
            "Cherenkov photons of delta rays and other secondaries are excluded.",
            changes_interpretation=False,
        ),
    )


def read_geant4_cherenkov(
    deck: Mapping[str, bytes],
    summary: bytes,
    photons: bytes,
    /,
    *,
    executable: PinnedExecutable,
) -> Geant4CherenkovResult:
    """Read the Geant4 Cherenkov outputs of ``deck`` into Phydrax units."""
    document = _document(deck)
    record = _summary(summary, executable)
    table = _array(photons)
    events = document["events"]
    paths = np.asarray(record["path_lengths_m"], dtype=np.float64)
    if (
        table.ndim != 2
        or table.shape[1] != _PHOTON_COLUMNS
        or not np.all(np.isfinite(table))
        or paths.shape != (events,)
    ):
        raise ValueError("Geant4 Cherenkov output does not match the deck.")
    event = table[:, 0].astype(np.int64)
    if np.any(event < 0) or np.any(event >= events):
        raise ValueError("Geant4 Cherenkov photons name events outside the deck.")
    digest = hashlib.sha256(photons).hexdigest()
    wavelengths = _photon_energy_length() / table[:, 1]
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "geant4-cherenkov-photons",
        "Geant4CherenkovResult",
        source_id=digest,
        target_id=canonical_fingerprint(
            {
                "kind": "geant4-cherenkov-result",
                "plan_id": document["plan_id"],
                "photons": array_tree_fingerprint(table),
                "path_lengths": array_tree_fingerprint(paths),
            }
        ),
        coordinate_mapping=(
            "Geant4 frame = plan frame translated to the step start",
            "time = plan step start time + Geant4 global time",
            "wavelength = 2 pi hbar c / photon energy",
        ),
        preserved_fields=(
            "step start, direction, and speed",
            "refractive index on the plan wavelength nodes",
        ),
        assumptions=("energies in eV, lengths in m, times in s",),
        losses=_cherenkov_losses(document),
    )
    return Geant4CherenkovResult(
        events=event,
        wavelengths=wavelengths,
        directions=table[:, 2:5],
        polarizations=table[:, 5:8],
        parent_directions=table[:, 8:11],
        parent_start_beta=table[:, 11],
        parent_end_beta=table[:, 12],
        positions=table[:, 13:16] + np.asarray(document["start_m"], dtype=np.float64),
        times=table[:, 16] + document["start_time_s"],
        path_lengths=paths,
        provider_version=record["geant4_pybind"],
        geant4_release=record["geant4"],
        datasets=tuple(record["datasets"]),
        executable_sha256=executable.sha256,
        license_id=executable.license_id,
        output_sha256=digest,
        report=report,
    )


def run_geant4_cherenkov(
    provider: Geant4Provider,
    plan: OpticalPhotonSourcePlan,
    steps: ChargedOpticalSteps,
    destination: str | Path,
    /,
    *,
    nist_material: str,
    events: int,
    random_seed: int = 1,
    maximum_photons_per_step: int = 100,
    maximum_beta_change_percent: float = 10.0,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 28,
) -> Geant4CherenkovResult:
    """Run pinned Geant4 on the translated Cherenkov step and read its photons."""
    inputs = geant4_cherenkov_input(
        plan,
        steps,
        nist_material=nist_material,
        events=events,
        random_seed=random_seed,
        maximum_photons_per_step=maximum_photons_per_step,
        maximum_beta_change_percent=maximum_beta_change_percent,
    )
    summary, photons = _run(
        provider,
        inputs,
        _CHERENKOV_SCRIPT_NAME,
        _PHOTONS,
        destination,
        timeout,
        maximum_output_bytes,
    )
    return read_geant4_cherenkov(inputs, summary, photons, executable=provider.executable)


@dataclass(frozen=True, slots=True)
class Geant4OpticalResult:
    """Terminal outcome of every photon Geant4 transported through the stack.

    Per photon, in launch order, the unit tallies mirror
    :class:`~phydrax.optics.transport.OpticalTransportTallies`:
    ``detector[photon, detector]`` for an exit through a detector plane,
    ``absorption[photon, medium]`` for bulk absorption, or for an exit through
    an absorber plane, in the medium the photon was in, and ``escape[photon]``
    for a lateral exit from the stack. For plane exits, ``exit_surfaces`` is
    the surface id (``-1`` otherwise), and ``exit_positions`` (m),
    ``exit_directions``, the real unit ``exit_polarizations``, and
    ``exit_times`` (s) describe the photon at the plane (NaN otherwise).
    ``rayleigh_scatters`` counts Rayleigh events per photon.
    """

    detector: np.ndarray
    absorption: np.ndarray
    escape: np.ndarray
    exit_surfaces: np.ndarray
    exit_positions: np.ndarray
    exit_directions: np.ndarray
    exit_polarizations: np.ndarray
    exit_times: np.ndarray
    rayleigh_scatters: np.ndarray
    provider_version: str
    geant4_release: str
    datasets: tuple[str, ...]
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


def _plane_record(
    surface: int,
    members: np.ndarray,
    corners: np.ndarray,
    orientation: tuple[np.ndarray, np.ndarray],
    roles: tuple[np.ndarray, np.ndarray, np.ndarray],
    tolerance: float,
    /,
) -> dict[str, float | int]:
    below, above = orientation
    kinds, detectors, acceptances = roles
    heights = corners[members][:, :, 2]
    if (
        np.ptp(heights) > tolerance
        or np.ptp(below[members]) != 0
        or np.ptp(above[members]) != 0
    ):
        raise ValueError(
            "Each surface of the Geant4 optical stack must be one z plane with one "
            "medium order."
        )
    first = int(np.flatnonzero(members)[0])
    return {
        "surface_id": surface,
        "z_m": float(np.mean(heights)),
        "kind": int(kinds[first]),
        "detector": int(detectors[first]),
        "acceptance": float(acceptances[first]),
        "below": int(below[first]),
        "above": int(above[first]),
    }


def _layer_stack(
    plan: OpticalMonteCarloPlan, /
) -> tuple[list[dict[str, float | int]], list[float], float]:
    """Planes normal to z, bottom to top, their common rectangle, and the tolerance."""
    surfaces = plan.surfaces
    if surfaces.volume_attenuation_enabled:
        raise ValueError(
            "The Geant4 optical oracle reads attenuation from the optical medium; "
            "surface-table attenuation is refused."
        )
    if np.any(
        _host(surfaces.branch_modes, np.int32) != int(NonSequentialBranchMode.BOTH)
    ):
        raise ValueError("Geant4 interfaces always reflect and transmit.")
    vertices = _host(surfaces.vertices, np.float64)
    corners = vertices[_host(surfaces.triangles, np.int64)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    tolerance = _PLANE_TOLERANCE * float(np.max(np.abs(vertices)))
    if np.any(np.abs(normals[:, :2]) > _PLANE_TOLERANCE * lengths[:, None]):
        raise ValueError(
            "The Geant4 optical oracle needs surfaces on planes normal to z."
        )
    upward = normals[:, 2] > 0.0
    negative = _host(surfaces.negative_medium_indices, np.int32)
    positive = _host(surfaces.positive_medium_indices, np.int32)
    orientation = (
        np.where(upward, negative, positive),
        np.where(upward, positive, negative),
    )
    roles = (
        _host(surfaces.surface_kinds, np.int32),
        _host(surfaces.detector_indices, np.int32),
        _host(surfaces.detector_acceptance_cosines, np.float64),
    )
    lower = np.min(vertices[:, :2], axis=0)
    upper = np.max(vertices[:, :2], axis=0)
    area = float(np.prod(upper - lower))
    ids = _host(surfaces.surface_ids, np.int32)
    planes = []
    for surface in np.unique(ids):
        members = ids == surface
        flat = corners[members][:, :, :2].reshape(-1, 2)
        if (
            np.any(np.abs(np.min(flat, axis=0) - lower) > tolerance)
            or np.any(np.abs(np.max(flat, axis=0) - upper) > tolerance)
            or abs(0.5 * float(np.sum(lengths[members])) - area) > tolerance * area
        ):
            raise ValueError(
                "Every Geant4 optical plane must cover the stack rectangle exactly once."
            )
        planes.append(
            _plane_record(int(surface), members, corners, orientation, roles, tolerance)
        )
    planes.sort(key=lambda plane: plane["z_m"])
    heights = np.asarray([plane["z_m"] for plane in planes], dtype=np.float64)
    if len(planes) < 2 or np.any(np.diff(heights) <= tolerance):
        raise ValueError("The Geant4 optical stack needs at least two distinct planes.")
    for index, plane in enumerate(planes):
        _check_plane_role(plane, index in (0, len(planes) - 1))
    for lower_plane, upper_plane in zip(planes[:-1], planes[1:], strict=True):
        if lower_plane["above"] != upper_plane["below"]:
            raise ValueError("Adjacent Geant4 optical planes must bound one medium.")
    return (
        planes,
        [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])],
        tolerance,
    )


def _check_plane_role(plane: dict[str, float | int], terminal: bool, /) -> None:
    kind = NonSequentialSurfaceKind(plane["kind"])
    match kind:
        case NonSequentialSurfaceKind.DETECTOR | NonSequentialSurfaceKind.ABSORBER:
            if not terminal:
                raise ValueError(
                    "Detector and absorber planes must be the outermost planes."
                )
            if kind == NonSequentialSurfaceKind.DETECTOR and plane["acceptance"] > 0.0:
                raise ValueError("Geant4 detector planes accept every incidence angle.")
        case NonSequentialSurfaceKind.DIELECTRIC:
            if terminal:
                raise ValueError(
                    "The Geant4 optical stack must end in detector or absorber planes."
                )
        case NonSequentialSurfaceKind.MIRROR:
            raise ValueError("The Geant4 optical oracle refuses mirror surfaces.")
        case _:
            assert_never(kind)


def _optical_lengths(coefficients: np.ndarray, name: str, /) -> list[float] | None:
    """Geant4 lengths (m) ascending in photon energy, or ``None`` when absent."""
    if np.all(coefficients == 0.0):
        return None
    if np.any(coefficients == 0.0):
        raise ValueError(
            f"Geant4 {name} lengths must be finite at every wavelength node or at none."
        )
    return (1.0 / coefficients[::-1]).tolist()


def _optical_media(
    medium: SpectralOpticalMedium,
    media: set[int],
    length_per_wavelength_unit: float,
    /,
) -> tuple[dict[str, dict[str, list[float] | None]], np.ndarray]:
    table = _host(medium.table, np.float64)
    if (
        medium.mie_tables is not None
        or medium.emission is not None
        or np.any(table[..., _Channel.HENYEY_GREENSTEIN] != 0.0)
    ):
        raise ValueError(
            "The Geant4 optical oracle supports bulk absorption and Rayleigh "
            "scattering only (no Henyey-Greenstein, Mie, or wavelength shifting)."
        )
    grid = _host(medium.wavelengths, np.float64) * length_per_wavelength_unit
    energies = _photon_energy_length() / grid
    records = {
        str(index): {
            "photon_energies_ev": energies[::-1].tolist(),
            "refractive_indices": table[index, ::-1, _Channel.REFRACTIVE_INDEX].tolist(),
            "ABSLENGTH": _optical_lengths(
                table[index, :, _Channel.ABSORPTION], "absorption"
            ),
            "RAYLEIGH": _optical_lengths(table[index, :, _Channel.RAYLEIGH], "Rayleigh"),
        }
        for index in sorted(media)
    }
    return records, grid


def _photon_energy_length() -> float:
    """``2π ħ c / e``: photon energy in eV times vacuum wavelength in m."""
    scale = ElectromagneticScaleContract.si()
    return (
        2.0
        * math.pi
        * float(
            scale.reduced_planck_constant * scale.speed_of_light / scale.elementary_charge
        )
    )


def _linear_polarizations(photons: OpticalPhotonState, /) -> np.ndarray:
    """Real unit field directions of linearly polarized Jones vectors."""
    jones = np.asarray(photons.jones_vectors, dtype=np.complex128)
    reference = jones[np.arange(jones.shape[0]), np.argmax(np.abs(jones), axis=1)]
    aligned = jones * (np.conj(reference) / np.abs(reference))[:, None]
    if np.any(np.abs(aligned.imag) > _LINEAR_POLARIZATION_TOLERANCE):
        raise ValueError("Geant4 optical photons carry linear polarization only.")
    axes = _host(photons.transverse_axes, np.float64)
    directions = _host(photons.directions, np.float64)
    field = aligned.real[:, :1] * axes + aligned.real[:, 1:] * np.cross(directions, axes)
    return field / np.linalg.norm(field, axis=1)[:, None]


def _optical_photons(
    photons: OpticalPhotonState,
    planes: list[dict[str, float | int]],
    rectangle: list[float],
    grid: np.ndarray,
    length_per_wavelength_unit: float,
    /,
) -> list[dict[str, object]]:
    if not isinstance(photons, OpticalPhotonState):
        raise TypeError("photons must be an OpticalPhotonState.")
    if photons.weights.ndim != 1:
        raise ValueError("The Geant4 optical oracle needs a rank-one photon batch.")
    if np.any(_host(photons.weights, np.float64) != 1.0):
        raise ValueError("Geant4 optical photons are unweighted; weights must be one.")
    wavelengths = _host(photons.wavelengths, np.float64) * length_per_wavelength_unit
    if np.any(wavelengths < grid[0]) or np.any(wavelengths > grid[-1]):
        raise ValueError("Photon wavelengths must lie on the medium wavelength grid.")
    positions = _host(photons.positions, np.float64)
    heights = np.asarray([plane["z_m"] for plane in planes], dtype=np.float64)
    layer = np.searchsorted(heights, positions[:, 2]) - 1
    inside = (
        (positions[:, 2] > heights[0])
        & (positions[:, 2] < heights[-1])
        & (positions[:, 0] > rectangle[0])
        & (positions[:, 0] < rectangle[1])
        & (positions[:, 1] > rectangle[2])
        & (positions[:, 1] < rectangle[3])
        & ~np.isin(positions[:, 2], heights)
    )
    if not np.all(inside):
        raise ValueError("Geant4 optical photons must start strictly inside the stack.")
    layer_media = np.asarray([plane["above"] for plane in planes[:-1]], dtype=np.int32)
    if np.any(layer_media[layer] != _host(photons.medium_indices, np.int32)):
        raise ValueError("Photon medium indices must match their layer of the stack.")
    polarizations = _linear_polarizations(photons)
    directions = _host(photons.directions, np.float64)
    times = _host(photons.times, np.float64)
    energies = _photon_energy_length() / wavelengths
    return [
        {
            "position_m": positions[index].tolist(),
            "direction": directions[index].tolist(),
            "polarization": polarizations[index].tolist(),
            "energy_ev": float(energies[index]),
            "time_s": float(times[index]),
        }
        for index in range(positions.shape[0])
    ]


def geant4_optical_input(
    plan: OpticalMonteCarloPlan,
    photons: OpticalPhotonState,
    /,
    *,
    length_per_wavelength_unit: float = 1.0,
    random_seed: int = 1,
) -> dict[str, bytes]:
    """Translate a planar-stack optical transport plan and photons into Geant4."""
    if not isinstance(plan, OpticalMonteCarloPlan):
        raise TypeError("plan must be an OpticalMonteCarloPlan.")
    if plan.relativity.scale_id != RelativityScaleContract.si().scale_id:
        raise ValueError("The Geant4 optical oracle needs the SI relativity scale.")
    if not isinstance(plan.detector_response, UnitDetectorResponse):
        raise ValueError("Geant4 detector planes record every photon (unit response).")
    model = plan.surface_model
    if not isinstance(model, UnifiedSurfaceModel):
        raise ValueError("The Geant4 optical oracle needs the polarized UNIFIED model.")
    unit = positive_finite_float(length_per_wavelength_unit, "length_per_wavelength_unit")
    seed = positive_integer(random_seed, "random_seed")
    planes, rectangle, tolerance = _layer_stack(plan)
    reflectivity = _host(model.reflectivity, np.float64)
    for plane in planes[1:-1]:
        surface = int(plane["surface_id"])
        if model.finishes[surface] != "polished" or reflectivity[surface] != 1.0:
            raise ValueError(
                "Geant4 interfaces must be lossless polished UNIFIED dielectric "
                "boundaries."
            )
    heights = [float(plane["z_m"]) for plane in planes]
    layer_media = [int(plane["above"]) for plane in planes[:-1]]
    layers = [
        {"z_m": [low, high], "medium": medium}
        for low, high, medium in zip(heights[:-1], heights[1:], layer_media, strict=True)
    ]
    medium = plan.medium
    if not isinstance(medium, SpectralOpticalMedium):
        raise ValueError("The Geant4 optical oracle needs a SpectralOpticalMedium.")
    media, grid = _optical_media(medium, set(layer_media), unit)
    records = _optical_photons(photons, planes, rectangle, grid, unit)
    extent = float(np.max(np.abs(_host(plan.surfaces.vertices, np.float64))))
    document = {
        "medium_id": medium.medium_id,
        "surface_model_id": model.surface_model_id,
        "planes": planes,
        "layers": layers,
        "rectangle_m": rectangle,
        "world_half_m": 2.0 * extent,
        "exit_tolerance_m": max(tolerance, _EXIT_TOLERANCE * (heights[-1] - heights[0])),
        "media": media,
        "medium_count": medium.medium_count,
        "detector_count": plan.surfaces.detector_count,
        "maximum_interactions": plan.maximum_interactions,
        "seed": seed,
        "photons": records,
        "inactive_optical_processes": [
            "Cerenkov",
            "Scintillation",
            "OpMieHG",
            "OpWLS",
            "OpWLS2",
        ],
    }
    return {
        _OPTICAL_SCRIPT_NAME: _OPTICAL_SCRIPT,
        _DECK: json.dumps(document, sort_keys=True).encode(),
    }


def _optical_losses(document: dict, /) -> tuple[AdapterLoss, ...]:
    return (
        AdapterLoss(
            "media.bulk",
            "export",
            "synthesized",
            "Each optical medium is a G4_Galactic-composition material carrying only "
            "the plan's RINDEX, ABSLENGTH, and RAYLEIGH tables.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "media.interpolation",
            "export",
            "transformed",
            "Geant4 interpolates the optical tables linearly in photon energy; "
            "Phydrax interpolates linearly in wavelength (identical on the nodes).",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "geometry.lateral",
            "export",
            "transformed",
            f"Layer boxes end at the plane rectangle {document['rectangle_m']!r} m; "
            "Geant4 kills photons leaving it laterally (escape), while Phydrax "
            "continues them in the layer medium.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "surfaces.terminal",
            "export",
            "transformed",
            "Terminal detector and absorber planes are the boundary to a world "
            "without RINDEX, which absorbs every arriving photon.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "polarization.phase",
            "import",
            "dropped",
            "Geant4 carries a real linear polarization vector: the elliptical "
            "states of total internal reflection are not representable.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "polarization.in_plane_sign",
            "import",
            "transformed",
            "The validated Geant4 release (11.4.p01) reverses the in-plane (p) "
            "polarization component relative to the s component at every dielectric "
            "crossing, even across matched indices; exit polarizations are returned "
            "as Geant4 reports them, so only |E.s| and |E.p| compare with Phydrax.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "plan.maximum_interactions",
            "export",
            "dropped",
            "Geant4 has no interaction cap; the Phydrax bound of "
            f"{document['maximum_interactions']} interactions is not imposed.",
            changes_interpretation=False,
        ),
    )


def read_geant4_optical(
    deck: Mapping[str, bytes],
    summary: bytes,
    outcomes: bytes,
    /,
    *,
    executable: PinnedExecutable,
) -> Geant4OpticalResult:
    """Read the Geant4 optical outcomes of ``deck`` into per-photon tallies."""
    document = _document(deck)
    record = _summary(summary, executable)
    values = _array(outcomes)
    photons = document["photons"]
    count = len(photons)
    layers = document["layers"]
    if values.shape != (count, _OUTCOME_COLUMNS) or not np.all(np.isfinite(values)):
        raise ValueError("Geant4 optical outcomes do not match the deck.")
    code = values[:, 0].astype(np.int64)
    layer = values[:, 1].astype(np.int64)
    if np.any(code > 2) or np.any(layer < 0) or np.any(layer >= len(layers)):
        raise ValueError("Geant4 ended a photon outside the supported outcomes.")
    energies = np.asarray([photon["energy_ev"] for photon in photons], dtype=np.float64)
    if np.any(np.abs(values[:, 13] - energies) > 1.0e-9 * energies):
        raise ValueError("Geant4 changed a photon energy (unsupported process).")
    media = np.asarray([layer_["medium"] for layer_ in layers], dtype=np.int64)[layer]
    rows = np.arange(count)
    detector = np.zeros((count, document["detector_count"]), dtype=np.float64)
    absorption = np.zeros((count, document["medium_count"]), dtype=np.float64)
    exited = code == 0
    absorbed = ~exited
    absorption[rows[absorbed], media[absorbed]] = 1.0
    exit_surfaces = np.full(count, -1, dtype=np.int64)
    tolerance = document["exit_tolerance_m"]
    for plane in (document["planes"][0], document["planes"][-1]):
        hit = exited & (np.abs(values[:, 4] - plane["z_m"]) <= tolerance)
        exit_surfaces[hit] = plane["surface_id"]
        kind = NonSequentialSurfaceKind(plane["kind"])
        if kind == NonSequentialSurfaceKind.DETECTOR:
            detector[hit, plane["detector"]] = 1.0
        else:
            absorption[rows[hit], media[hit]] = 1.0
    escape = (exited & (exit_surfaces < 0)).astype(np.float64)
    plane_exit = exit_surfaces >= 0
    missing = np.where(plane_exit[:, None], 0.0, np.nan)
    digest = hashlib.sha256(outcomes).hexdigest()
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "geant4-optical-outcomes",
        "Geant4OpticalResult",
        source_id=digest,
        target_id=canonical_fingerprint(
            {
                "kind": "geant4-optical-result",
                "medium_id": document["medium_id"],
                "surface_model_id": document["surface_model_id"],
                "outcomes": array_tree_fingerprint(values),
            }
        ),
        coordinate_mapping=(
            "Geant4 frame = plan frame (lengths in m, times in s)",
            "photon energy = 2 pi hbar c / vacuum wavelength",
        ),
        preserved_fields=(
            "photon positions, directions, linear polarizations, wavelengths, times",
            "plane positions, medium order, refractive indices",
            "absorption and Rayleigh lengths",
        ),
        assumptions=(
            "interfaces are UNIFIED polished dielectric_dielectric border surfaces",
            "tallies are per photon, in launch order",
        ),
        losses=_optical_losses(document),
    )
    return Geant4OpticalResult(
        detector=detector,
        absorption=absorption,
        escape=escape,
        exit_surfaces=exit_surfaces,
        exit_positions=values[:, 2:5] + missing,
        exit_directions=values[:, 5:8] + missing,
        exit_polarizations=values[:, 8:11] + missing,
        exit_times=values[:, 11] + missing[:, 0],
        rayleigh_scatters=values[:, 12].astype(np.int64),
        provider_version=record["geant4_pybind"],
        geant4_release=record["geant4"],
        datasets=tuple(record["datasets"]),
        executable_sha256=executable.sha256,
        license_id=executable.license_id,
        output_sha256=digest,
        report=report,
    )


def run_geant4_optical(
    provider: Geant4Provider,
    plan: OpticalMonteCarloPlan,
    photons: OpticalPhotonState,
    destination: str | Path,
    /,
    *,
    length_per_wavelength_unit: float = 1.0,
    random_seed: int = 1,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 28,
) -> Geant4OpticalResult:
    """Run pinned Geant4 optical transport of ``photons`` through the plan's stack."""
    inputs = geant4_optical_input(
        plan,
        photons,
        length_per_wavelength_unit=length_per_wavelength_unit,
        random_seed=random_seed,
    )
    summary, outcomes = _run(
        provider,
        inputs,
        _OPTICAL_SCRIPT_NAME,
        _OUTCOMES,
        destination,
        timeout,
        maximum_output_bytes,
    )
    return read_geant4_optical(inputs, summary, outcomes, executable=provider.executable)


__all__ = [
    "Geant4CherenkovResult",
    "Geant4ElectromagneticPhysics",
    "Geant4OpticalResult",
    "Geant4Provider",
    "Geant4ShowerResult",
    "geant4_cherenkov_input",
    "geant4_optical_input",
    "geant4_shower_input",
    "read_geant4_cherenkov",
    "read_geant4_optical",
    "read_geant4_shower",
    "run_geant4_cherenkov",
    "run_geant4_optical",
    "run_geant4_shower",
]

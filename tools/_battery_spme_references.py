#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent Eq. 48 reference and explicit, synthetic PyBaMM parameter mapping.

All numbers below are self-authored numerical fixtures, not measured cell data.
No third-party parameter database is loaded. Equations: Marquis et al. (2019),
arXiv:1905.12553v2, section 7, Eq. 48, Table 6, Eq. 49 (CC-BY-4.0).
This module generates observations, never rights attestations or release evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
from scipy.sparse import block_diag, bmat, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply


def _reference_fingerprint(value):
    # Same canonical JSON byte contract, without importing the engine under test
    # into the independently installed reference process.
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


FARADAY = 96485.33212
GAS_CONSTANT = 8.31446261815324
PYBAMM_VERSION = "25.4.2"
PYBAMM_COMMIT = "6c615f7c982af8e552c170d4043dcf7bac7d7345"
PAPER_SOURCE = "https://arxiv.org/html/1905.12553v2#S7"
REFERENCE_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE_BOUNDARIES = (0.0, 0.5, 0.75, 1.0)
REFERENCE_CURRENTS = (0.2, 0.0, -0.05)
REFERENCE_ENGINES = ("self-authored-marquis-eq48", f"pybamm:{PYBAMM_VERSION}")


def reference_runtime_identity(engine):
    """Fingerprint actual installed code/binaries, not the upstream tag assumed."""
    names = (
        ("numpy", "scipy")
        if engine == REFERENCE_ENGINES[0]
        else (
            "pybamm",
            "numpy",
            "scipy",
            "casadi",
            "pybammsolvers",
            "sympy",
            "pandas",
            "xarray",
        )
    )
    packages = {}
    for name in names:
        installed = distribution(name)
        files = []
        for item in sorted(installed.files or (), key=str):
            if item.suffix not in (".py", ".so", ".dylib") and item.name not in (
                "METADATA",
                "LICENSE.txt",
                "LICENSE",
            ):
                continue
            path = installed.locate_file(item)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Installed reference dependency bytes are missing: {item}"
                )
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append((str(item), digest))
        if not files:
            raise ValueError(f"No installed source/binary identity available for {name}.")
        packages[name] = {
            "version": installed.version,
            "source_binary_sha256": _reference_fingerprint(files),
        }
    with Path(__file__).open("rb") as stream:
        adapter_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    content = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "adapter_sha256": adapter_digest,
        "upstream_pybamm_tag_commit": PYBAMM_COMMIT
        if engine == REFERENCE_ENGINES[1]
        else None,
    }
    return {**content, "runtime_id": _reference_fingerprint(content)}


@dataclass(frozen=True, slots=True)
class SyntheticSpmeData:
    """SI numerical benchmark, with no product-specific predictive claim."""

    area: float = 0.1
    lengths: tuple[float, float, float] = (1.0e-4, 5.0e-5, 1.0e-4)
    porosities: tuple[float, float, float] = (0.3, 0.5, 0.3)
    active_fractions: tuple[float, float] = (0.6, 0.6)
    radii: tuple[float, float] = (5.0e-6, 4.0e-6)
    cmax: tuple[float, float] = (3.0e4, 3.0e4)
    solid_diffusivities: tuple[float, float] = (2.0e-14, 1.0e-14)
    exchange_scales: tuple[float, float] = (5.0, 4.0)
    ocp: tuple[float, float] = (0.1, 4.1)
    solid_conductivities: tuple[float, float] = (100.0, 80.0)
    bruggeman: float = 1.5
    electrolyte_diffusivity: float = 2.0e-10
    electrolyte_conductivity: float = 1.1
    transference: float = 0.4
    electrolyte_concentration: float = 1000.0
    temperature: float = 298.15
    initial_stoichiometries: tuple[float, float] = (0.5, 0.5)
    maximum_current: float = 0.5

    def __post_init__(self):
        positive = (
            self.area,
            *self.lengths,
            *self.radii,
            *self.cmax,
            *self.solid_diffusivities,
            *self.exchange_scales,
            *self.solid_conductivities,
            self.bruggeman,
            self.electrolyte_diffusivity,
            self.electrolyte_conductivity,
            self.electrolyte_concentration,
            self.temperature,
            self.maximum_current,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError(
                "Synthetic physical coefficients must be positive and finite."
            )
        fractions = (
            *self.porosities,
            *self.active_fractions,
            *self.initial_stoichiometries,
        )
        if not all(np.isfinite(v) and 0 < v < 1 for v in fractions):
            raise ValueError(
                "Porosity, active fractions and initial occupancy must be interior."
            )
        if not np.isfinite(self.transference) or not 0 <= self.transference <= 1:
            raise ValueError("Transference must lie in [0, 1].")
        if not np.all(np.isfinite(self.ocp)):
            raise ValueError("OCP values must be finite.")

    @property
    def data_id(self):
        return _reference_fingerprint(
            {"kind": "self-authored-spme-equation-fixture", **asdict(self)}
        )

    def mapping_record(self):
        return {
            "data_id": self.data_id,
            "parameters_si": asdict(self),
            "current": "I_pybamm=-I_terminal; i_paper=-I_terminal/area",
            "kinetics": "i0=2*exchange_scale*sqrt(theta*(1-theta)*ce/ce_typ); j0_paper=2*i0",
            "solid_conductivity": "effective sigma: PyBaMM electrode Bruggeman exponent=0",
            "electrolyte": "constant De,kappa,tplus evaluated at ce_typ,T; thermodynamic factor=1",
            "initialization": "uniform theta_n,theta_p,ce_typ; inventories fixed",
            "hold": "piecewise constant; sample at transition uses left limit",
            "output": "SI concentration cell averages, shell inventories, collector voltage difference",
            "paper": PAPER_SOURCE,
            "predictive_parameterization": False,
        }


def native_parameters(data: SyntheticSpmeData = SyntheticSpmeData()):
    """Map synthetic SI data to existing native property and parameter substrates."""
    import jax.numpy as jnp

    from phydrax.applications.battery._properties import (
        ConcentrationTemperaturePropertyLaw,
        ConstantPropertyLaw,
    )
    from phydrax.applications.battery._spm import SpmParameters
    from phydrax.applications.battery._spme_marquis2019 import Marquis2019SpmeParameters

    def constant(value, quantity, unit):
        potential = unit == "V"
        return ConstantPropertyLaw(
            jnp.asarray(value),
            jnp.asarray((0.0, 1.0) if potential else (250.0, 350.0)),
            value_bounds=(-jnp.inf, jnp.inf) if potential else (0.0, jnp.inf),
            quantity=quantity,
            coordinate="stoichiometry" if potential else "temperature",
            value_unit=unit,
            coordinate_unit="1" if potential else "K",
            source_id=data.data_id,
        )

    def electrolyte(value, quantity, unit):
        return ConcentrationTemperaturePropertyLaw(
            jnp.asarray((100.0, 2000.0)),
            jnp.asarray((250.0, 350.0)),
            jnp.full((2, 2), value),
            value_bounds=(0.0, 1.0) if unit == "1" else (0.0, jnp.inf),
            quantity=quantity,
            value_unit=unit,
            source_id=data.data_id,
        )

    spm = SpmParameters(
        electrode_area_m2=data.area,
        negative_electrode_thickness_m=data.lengths[0],
        positive_electrode_thickness_m=data.lengths[2],
        negative_active_material_volume_fraction=data.active_fractions[0],
        positive_active_material_volume_fraction=data.active_fractions[1],
        negative_particle_radius_m=data.radii[0],
        positive_particle_radius_m=data.radii[1],
        negative_maximum_concentration_mol_m3=data.cmax[0],
        positive_maximum_concentration_mol_m3=data.cmax[1],
        temperature_k=data.temperature,
        maximum_absolute_current_a=data.maximum_current,
        negative_stoichiometry_at_empty=0.1,
        negative_stoichiometry_at_full=0.9,
        positive_stoichiometry_at_empty=0.9,
        positive_stoichiometry_at_full=0.1,
        negative_solid_diffusivity=constant(
            data.solid_diffusivities[0], "negative-solid-diffusivity", "m2/s"
        ),
        positive_solid_diffusivity=constant(
            data.solid_diffusivities[1], "positive-solid-diffusivity", "m2/s"
        ),
        negative_exchange_current_density=constant(
            data.exchange_scales[0], "negative-exchange-current", "A/m2"
        ),
        positive_exchange_current_density=constant(
            data.exchange_scales[1], "positive-exchange-current", "A/m2"
        ),
        negative_open_circuit_potential=constant(data.ocp[0], "negative-ocp", "V"),
        positive_open_circuit_potential=constant(data.ocp[1], "positive-ocp", "V"),
    )
    return Marquis2019SpmeParameters(
        spm,
        separator_thickness_m=data.lengths[1],
        negative_electrolyte_porosity=data.porosities[0],
        separator_electrolyte_porosity=data.porosities[1],
        positive_electrolyte_porosity=data.porosities[2],
        bruggeman_coefficient=data.bruggeman,
        typical_electrolyte_concentration_mol_m3=data.electrolyte_concentration,
        electrolyte_diffusivity=electrolyte(
            data.electrolyte_diffusivity, "electrolyte-diffusivity", "m2/s"
        ),
        electrolyte_conductivity=electrolyte(
            data.electrolyte_conductivity, "electrolyte-conductivity", "S/m"
        ),
        transference_number=electrolyte(data.transference, "transference-number", "1"),
        negative_solid_conductivity_s_m=data.solid_conductivities[0],
        positive_solid_conductivity_s_m=data.solid_conductivities[1],
    )


def conservative_projection(
    source_faces, target_faces, source_averages, *, spherical=False
):
    """Exact overlap projection of cell averages; no point interpolation of mass."""
    source, target = np.asarray(source_faces, float), np.asarray(target_faces, float)
    values = np.asarray(source_averages, float)
    if (
        source.ndim != 1
        or target.ndim != 1
        or min(len(source), len(target)) < 2
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(np.diff(source) <= 0)
        or np.any(np.diff(target) <= 0)
        or source[0] != target[0]
        or source[-1] != target[-1]
        or values.shape[-1] != len(source) - 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Projection requires finite cell averages over identical ordered domains."
        )
    if spherical and source[0] < 0:
        raise ValueError("Spherical radii cannot be negative.")
    power = 3 if spherical else 1
    left = np.maximum(target[:-1, None], source[None, :-1])
    right = np.minimum(target[1:, None], source[None, 1:])
    overlaps = np.where(right > left, right**power - left**power, 0.0)
    return values @ (overlaps / np.diff(target**power)[:, None]).T


def _diffusion_matrix(capacities, conductances):
    n = len(capacities)
    incidence = diags(
        (-np.ones(n - 1), np.ones(n - 1)), (0, 1), shape=(n - 1, n), format="csr"
    )
    return -diags(1 / capacities) @ incidence.T @ diags(conductances) @ incidence


def _schedule(times, boundaries, currents, data):
    times, boundaries, currents = (
        np.asarray(times, float),
        np.asarray(boundaries, float),
        np.asarray(currents, float),
    )
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
        or times[0] != 0
        or boundaries.ndim != 1
        or currents.ndim != 1
        or len(boundaries) != len(currents) + 1
        or boundaries[0] != 0
        or boundaries[-1] != times[-1]
        or np.any(np.diff(boundaries) <= 0)
        or not np.all(np.isfinite(boundaries))
        or not np.all(np.isfinite(currents))
        or np.max(np.abs(currents)) > data.maximum_current
    ):
        raise ValueError(
            "Reference requires a finite bounded current/rest schedule starting at zero."
        )
    return times, boundaries, currents


def paper_reference(
    times,
    boundaries,
    currents,
    *,
    data=SyntheticSpmeData(),
    radial_cells=32,
    region_cells=(24, 12, 24),
):
    """Solve the independently assembled Eq. 48 linear transport via sparse expm.

    Small reference-only FV operators; affine forcing is exponentiated exactly in
    time on each hold. Neither native transport nor native voltage code is used.
    """
    times, boundaries, currents = _schedule(times, boundaries, currents, data)
    if (
        type(radial_cells) is not int
        or not 3 <= radial_cells <= 128
        or len(region_cells) != 3
        or any(type(n) is not int or not 2 <= n <= 128 for n in region_cells)
    ):
        raise ValueError(
            "Reference mesh is bounded to 3..128 shells and 2..128 cells/region."
        )
    matrices, capacities, centres, initial, sources = [], [], [], [], []
    for k in range(2):
        radius, length = data.radii[k], data.lengths[2 * k]
        faces = np.linspace(0, radius, radial_cells + 1)
        centre = 0.75 * np.diff(faces**4) / np.diff(faces**3)
        volume = (
            data.active_fractions[k] * data.area * length * np.diff((faces / radius) ** 3)
        )
        multiplicity = (
            data.active_fractions[k] * data.area * length / (4 * np.pi * radius**3 / 3)
        )
        conductance = (
            multiplicity
            * 4
            * np.pi
            * faces[1:-1] ** 2
            * data.solid_diffusivities[k]
            / np.diff(centre)
        )
        matrices.append(_diffusion_matrix(volume, conductance))
        forcing = np.zeros(radial_cells)
        forcing[-1] = (1 if k == 0 else -1) / FARADAY / volume[-1]
        sources.append(forcing)
        capacities.append(volume)
        centres.append(centre)
        initial.append(
            np.full(radial_cells, data.cmax[k] * data.initial_stoichiometries[k])
        )
    widths = np.concatenate(
        [
            np.full(n, length / n)
            for length, n in zip(data.lengths, region_cells, strict=True)
        ]
    )
    porosity = np.repeat(data.porosities, region_cells)
    effective_d = porosity**data.bruggeman * data.electrolyte_diffusivity
    volume = data.area * widths * porosity
    conductance = data.area / (
        widths[:-1] / (2 * effective_d[:-1]) + widths[1:] / (2 * effective_d[1:])
    )
    matrices.append(_diffusion_matrix(volume, conductance))
    forcing = np.concatenate(
        (
            np.full(
                region_cells[0],
                -(1 - data.transference)
                / (FARADAY * data.area * data.lengths[0] * data.porosities[0]),
            ),
            np.zeros(region_cells[1]),
            np.full(
                region_cells[2],
                (1 - data.transference)
                / (FARADAY * data.area * data.lengths[2] * data.porosities[2]),
            ),
        )
    )
    sources.append(forcing)
    capacities.append(volume)
    initial.append(np.full(len(widths), data.electrolyte_concentration))
    matrix = block_diag(matrices, format="csr")
    forcing = np.concatenate(sources)
    state = np.concatenate(initial)
    collected = [state.copy()]
    t = 0.0
    for stop in times[1:]:
        while t < stop:
            index = min(
                np.searchsorted(boundaries, t, side="right") - 1, len(currents) - 1
            )
            end = min(stop, boundaries[index + 1])
            augmented = bmat(
                [
                    [matrix, csr_matrix((forcing * currents[index])[:, None])],
                    [None, csr_matrix((1, 1))],
                ],
                format="csr",
            )
            state = expm_multiply(augmented * (end - t), np.r_[state, 1.0])[:-1]
            t = end
        collected.append(state.copy())
    states = np.asarray(collected)
    observed_current = currents[
        np.clip(np.searchsorted(boundaries, times, side="left") - 1, 0, len(currents) - 1)
    ]
    cn, cp, ce = np.split(states, (radial_cells, 2 * radial_cells), axis=-1)
    surface = []
    for k, concentration in enumerate((cn, cp)):
        specific_area = 3 * data.active_fractions[k] / data.radii[k]
        outward_flux = (
            (-1 if k == 0 else 1)
            * observed_current
            / (data.area * data.lengths[2 * k] * specific_area * FARADAY)
        )
        surface.append(
            concentration[:, -1]
            - outward_flux
            * (data.radii[k] - centres[k][-1])
            / data.solid_diffusivities[k]
        )
    if np.any(ce <= 0) or any(
        np.any((cs <= 0) | (cs >= maximum))
        for cs, maximum in zip(surface, data.cmax, strict=True)
    ):
        raise ValueError("Independent reference left concentration support.")
    cuts = np.cumsum((0, *region_cells))
    electrolyte_regions = [ce[:, cuts[k] : cuts[k + 1]] for k in range(3)]
    voltage = eq48_voltage(data, observed_current, surface, electrolyte_regions)
    return {
        "times_s": times,
        "current_a": observed_current,
        "voltage_v": voltage,
        "negative_surface_concentration_mol_m3": surface[0],
        "positive_surface_concentration_mol_m3": surface[1],
        "negative_amount_mol": cn * capacities[0],
        "positive_amount_mol": cp * capacities[1],
        "electrolyte_amount_mol": ce * capacities[2],
        "electrolyte_concentration_mol_m3": ce,
        "electrolyte_faces_m": np.r_[0, np.cumsum(widths)],
        "mapping": data.mapping_record(),
        "runtime_family": "numpy-scipy-sparse-exponential",
        "engine_id": "self-authored-marquis-eq48",
    }


def eq48_voltage(data, current, surfaces, electrolyte_regions):
    """Paper's electrode-averaged Eq. 48 voltage (not a fitted terminal curve)."""
    reaction = np.zeros_like(current, dtype=float)
    for k in range(2):
        theta = surfaces[k] / data.cmax[k]
        j0 = (
            4
            * data.exchange_scales[k]
            * np.sqrt(theta * (1 - theta))
            * np.mean(
                np.sqrt(electrolyte_regions[2 * k] / data.electrolyte_concentration),
                axis=-1,
            )
        )
        reaction += np.arcsinh(
            current
            / (
                data.area
                * (3 * data.active_fractions[k] / data.radii[k])
                * data.lengths[2 * k]
                * j0
            )
        )
    thermal = 2 * GAS_CONSTANT * data.temperature / FARADAY
    concentration = (
        thermal
        * (1 - data.transference)
        * (
            np.mean(electrolyte_regions[2], axis=-1)
            - np.mean(electrolyte_regions[0], axis=-1)
        )
        / data.electrolyte_concentration
    )
    electrolyte_resistance = (
        sum(
            length / (factor * porosity**data.bruggeman)
            for length, porosity, factor in zip(
                data.lengths, data.porosities, (3, 1, 3), strict=True
            )
        )
        / data.electrolyte_conductivity
    )
    solid_resistance = (
        data.lengths[0] / data.solid_conductivities[0]
        + data.lengths[2] / data.solid_conductivities[1]
    ) / 3
    return (
        data.ocp[1]
        - data.ocp[0]
        + thermal * reaction
        + concentration
        + current / data.area * (electrolyte_resistance + solid_resistance)
    )


def pybamm_parameter_values(data=SyntheticSpmeData()):
    """Construct from an empty mapping, never PyBaMM's bundled parameter sets."""
    os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"
    import pybamm

    if version("pybamm") != PYBAMM_VERSION:
        raise ValueError(
            f"Reference requires immutable PyBaMM {PYBAMM_VERSION} ({PYBAMM_COMMIT})."
        )
    p = {
        "chemistry": "lithium_ion",
        "Electrode height [m]": 1.0,
        "Electrode width [m]": data.area,
        "Separator thickness [m]": data.lengths[1],
        "Separator porosity": data.porosities[1],
        "Separator Bruggeman coefficient (electrolyte)": data.bruggeman,
        "Initial concentration in electrolyte [mol.m-3]": data.electrolyte_concentration,
        "Electrolyte diffusivity [m2.s-1]": data.electrolyte_diffusivity,
        "Electrolyte conductivity [S.m-1]": data.electrolyte_conductivity,
        "Cation transference number": data.transference,
        "Thermodynamic factor": 1.0,
        "Reference temperature [K]": data.temperature,
        "Initial temperature [K]": data.temperature,
        "Ambient temperature [K]": data.temperature,
        "Contact resistance [Ohm]": 0.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Nominal cell capacity [A.h]": data.area
        * data.lengths[0]
        * data.active_fractions[0]
        * data.cmax[0]
        * FARADAY
        / 3600,
        "Current function [A]": "[input]",
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 5.0,
        "Open-circuit voltage at 0% SOC [V]": 2.0,
        "Open-circuit voltage at 100% SOC [V]": 5.0,
    }
    for k, name in enumerate(("Negative", "Positive")):

        def exchange(c_e, c_s_surf, c_s_max, T, scale=data.exchange_scales[k]):
            theta = c_s_surf / c_s_max
            return (
                2
                * scale
                * pybamm.sqrt(theta * (1 - theta) * c_e / data.electrolyte_concentration)
            )

        p.update(
            {
                f"{name} electrode thickness [m]": data.lengths[2 * k],
                f"{name} electrode porosity": data.porosities[2 * k],
                f"{name} electrode active material volume fraction": data.active_fractions[
                    k
                ],
                f"{name} particle radius [m]": data.radii[k],
                f"Maximum concentration in {name.lower()} electrode [mol.m-3]": data.cmax[
                    k
                ],
                f"Initial concentration in {name.lower()} electrode [mol.m-3]": data.initial_stoichiometries[
                    k
                ]
                * data.cmax[k],
                f"{name} particle diffusivity [m2.s-1]": data.solid_diffusivities[k],
                f"{name} electrode conductivity [S.m-1]": data.solid_conductivities[k],
                f"{name} electrode Bruggeman coefficient (electrolyte)": data.bruggeman,
                f"{name} electrode Bruggeman coefficient (electrode)": 0.0,
                f"{name} electrode charge transfer coefficient": 0.5,
                f"{name} electrode exchange-current density [A.m-2]": exchange,
                f"{name} electrode OCP [V]": data.ocp[k],
                f"{name} electrode OCP entropic change [V.K-1]": 0.0,
                f"{name} current collector thickness [m]": 1.0e-5,
                f"{name} current collector conductivity [S.m-1]": 1.0e7,
            }
        )
    return pybamm.ParameterValues(p)


def pybamm_reference(
    times,
    boundaries,
    currents,
    *,
    data=SyntheticSpmeData(),
    radial_cells=32,
    region_cells=(24, 12, 24),
    rtol=1e-10,
    atol=1e-10,
):
    """Execute pinned PyBaMM SPMe; each hold restarts with the retained state.

    PyBaMM's composite voltage uses nonlinear log concentration terms; return it
    separately, and explicitly project its independent concentration solution to
    the linearized Eq.48 voltage for the canonical equation comparison.
    """
    os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"
    import pybamm

    times, boundaries, currents = _schedule(times, boundaries, currents, data)
    parameters = pybamm_parameter_values(data)
    model = pybamm.lithium_ion.SPMe(
        options={"thermal": "isothermal", "electrolyte conductivity": "composite"}
    )
    x = pybamm.standard_spatial_vars
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        var_pts={
            x.x_n: region_cells[0],
            x.x_s: region_cells[1],
            x.x_p: region_cells[2],
            x.r_n: radial_cells,
            x.r_p: radial_cells,
        },
        solver=pybamm.CasadiSolver(mode="safe", rtol=rtol, atol=atol),
    )
    parts = []
    for index, current in enumerate(currents):
        left, right = boundaries[index : index + 2]
        local_times = np.unique(
            np.r_[0.0, times[(times > left) & (times <= right)] - left, right - left]
        )
        solution = simulation.step(
            right - left,
            t_eval=local_times,
            inputs={"Current function [A]": -float(current)},
            save=False,
        )
        selected = times[
            (times >= left if index == 0 else times > left) & (times <= right)
        ]
        surfaces = [
            np.asarray(
                solution[f"X-averaged {name} particle surface concentration [mol.m-3]"](
                    selected
                )
            )
            for name in ("negative", "positive")
        ]
        regions = [
            np.asarray(
                solution[f"{name} electrolyte concentration [mol.m-3]"](selected)
            ).T
            for name in ("Negative", "Separator", "Positive")
        ]
        observed_current = np.full(selected.shape, current)
        parts.append(
            {
                "times_s": selected,
                "current_a": observed_current,
                "voltage_v": eq48_voltage(data, observed_current, surfaces, regions),
                "engine_terminal_voltage_v": np.asarray(
                    solution["Voltage [V]"](selected)
                ),
                "negative_surface_concentration_mol_m3": surfaces[0],
                "positive_surface_concentration_mol_m3": surfaces[1],
                "electrolyte_concentration_mol_m3": np.concatenate(regions, axis=-1),
            }
        )
    result = {
        key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]
    }
    result.update(
        {
            "engine_id": f"pybamm:{PYBAMM_VERSION}",
            "runtime_family": "pybamm-casadi-sundials",
            "mapping": data.mapping_record(),
            "voltage_projection": "Eq48 linearized reconstruction; engine voltage retained separately",
        }
    )
    return result


def reference_payload(engine):
    """Generate finite three-level output without importing PHYDRAX or signing."""
    if engine not in REFERENCE_ENGINES:
        raise ValueError("Reference engine must be an exact registered identity.")
    data = SyntheticSpmeData()
    run = paper_reference if engine == REFERENCE_ENGINES[0] else pybamm_reference
    levels = []
    for radial, cells in zip(
        (32, 64, 128), ((24, 12, 24), (48, 24, 48), (96, 48, 96)), strict=True
    ):
        output = run(
            REFERENCE_TIMES,
            REFERENCE_BOUNDARIES,
            REFERENCE_CURRENTS,
            radial_cells=radial,
            region_cells=cells,
        )
        level = {
            name: np.asarray(output[name]).tolist()
            for name in (
                "times_s",
                "current_a",
                "voltage_v",
                "negative_surface_concentration_mol_m3",
                "positive_surface_concentration_mol_m3",
            )
        }
        if "engine_terminal_voltage_v" in output:
            level["engine_terminal_voltage_v"] = np.asarray(
                output["engine_terminal_voltage_v"]
            ).tolist()
        level["radial_cells"], level["region_cells"] = radial, list(cells)
        levels.append(level)
    return {
        "kind": "spme-reference-observation",
        "engine_id": engine,
        "runtime_family": output["runtime_family"],
        "mapping": data.mapping_record(),
        "levels": levels,
        "runtime": reference_runtime_identity(engine),
    }


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Independent SPMe reference with self-authored SI data; no rights or release claims."
    )
    parser.add_argument("--engine", choices=("paper", "pybamm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    payload = reference_payload(REFERENCE_ENGINES[0 if options.engine == "paper" else 1])
    with options.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
)
from ._closure import CosmologyPhysicalState, PhysicalDependencyProjection
from ._products import (
    CosmologyProductProvenance,
    LinearTransferDescriptor,
    LinearTransferTable,
    MatterPowerDescriptor,
    MatterPowerTable,
    ThermodynamicsHistory,
)
from ._scales import CosmologyScaleContract


class MassiveNeutrinoSpecies(StrictModule, NonTrainableState):
    """One externally evolved non-cold relic species."""

    mass_ev: float = eqx.field(static=True)
    temperature_ratio: float = eqx.field(static=True)
    degeneracy: float = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)
    species_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass_ev: float,
        /,
        *,
        temperature_ratio: float = 0.71611,
        degeneracy: float = 1.0,
        distribution_id: str = "fermi-dirac-zero-chemical-potential",
    ):
        mass = float(mass_ev)
        temperature = float(temperature_ratio)
        weight = float(degeneracy)
        distribution = str(distribution_id).strip()
        if (
            not np.isfinite(mass)
            or mass < 0.0
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or not np.isfinite(weight)
            or weight <= 0.0
            or not distribution
        ):
            raise ValueError("Massive neutrino species parameters are invalid.")
        self.mass_ev = mass
        self.temperature_ratio = temperature
        self.degeneracy = weight
        self.distribution_id = distribution
        self.species_id = canonical_fingerprint(
            {
                "kind": "massive-neutrino-species",
                "mass_ev": mass,
                "temperature_ratio": temperature,
                "degeneracy": weight,
                "distribution_id": distribution,
            }
        )

    def to_mapping(self) -> dict[str, float | str]:
        return {
            "mass_ev": self.mass_ev,
            "temperature_ratio": self.temperature_ratio,
            "degeneracy": self.degeneracy,
            "distribution_id": self.distribution_id,
        }


class CosmologyModelRequest(StrictModule, NonTrainableState):
    """Canonical host-only request for a narrowly declared precision calculation."""

    scale: CosmologyScaleContract
    hubble_constant: float = eqx.field(static=True)
    baryon_density: float = eqx.field(static=True)
    cold_dark_matter_density: float = eqx.field(static=True)
    curvature_density: float = eqx.field(static=True)
    dark_energy_w0: float = eqx.field(static=True)
    dark_energy_wa: float = eqx.field(static=True)
    photon_temperature: float = eqx.field(static=True)
    effective_neutrino_number: float = eqx.field(static=True)
    neutrinos: tuple[MassiveNeutrinoSpecies, ...]
    primordial_amplitude: float = eqx.field(static=True)
    primordial_tilt: float = eqx.field(static=True)
    primordial_pivot: float = eqx.field(static=True)
    reionization_optical_depth: float = eqx.field(static=True)
    transfer_fields: tuple[str, ...] = eqx.field(static=True)
    gauge: str = eqx.field(static=True)
    power_field: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    model_form_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: CosmologyScaleContract,
        /,
        *,
        hubble_constant: float,
        baryon_density: float,
        cold_dark_matter_density: float,
        curvature_density: float = 0.0,
        dark_energy_w0: float = -1.0,
        dark_energy_wa: float = 0.0,
        photon_temperature: float = 2.7255,
        effective_neutrino_number: float = 3.046,
        neutrinos: tuple[MassiveNeutrinoSpecies, ...] = (),
        primordial_amplitude: float = 2.1e-9,
        primordial_tilt: float = 0.965,
        primordial_pivot: float = 0.05,
        reionization_optical_depth: float = 0.054,
        transfer_fields: tuple[str, ...] = (
            "density/cold_baryon",
            "density/total_matter",
        ),
        gauge: str = "synchronous",
        power_field: str = "cold_baryon",
    ):
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be CosmologyScaleContract.")
        scalar_values = tuple(
            float(value)
            for value in (
                hubble_constant,
                baryon_density,
                cold_dark_matter_density,
                curvature_density,
                dark_energy_w0,
                dark_energy_wa,
                photon_temperature,
                effective_neutrino_number,
                primordial_amplitude,
                primordial_tilt,
                primordial_pivot,
                reionization_optical_depth,
            )
        )
        if any(not np.isfinite(value) for value in scalar_values):
            raise ValueError("Linear-theory scalar parameters must be finite.")
        if (
            scalar_values[0] <= 0.0
            or scalar_values[1] < 0.0
            or scalar_values[2] < 0.0
            or scalar_values[6] <= 0.0
            or scalar_values[7] < 0.0
            or scalar_values[8] <= 0.0
            or scalar_values[10] <= 0.0
            or scalar_values[11] < 0.0
        ):
            raise ValueError("Linear-theory positive parameter constraints failed.")
        species = tuple(neutrinos)
        if any(not isinstance(item, MassiveNeutrinoSpecies) for item in species):
            raise TypeError("neutrinos must contain MassiveNeutrinoSpecies values.")
        fields = tuple(str(field).strip() for field in transfer_fields)
        if (
            not fields
            or any(not field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError("Transfer fields must be non-empty and unique.")
        gauge_ = str(gauge).strip()
        if gauge_ not in ("synchronous", "newtonian", "gauge-invariant"):
            raise ValueError("Unknown linear-theory gauge.")
        power_field_ = str(power_field).strip()
        if power_field_ not in ("cold_baryon", "total_matter"):
            raise ValueError("power_field must be cold_baryon or total_matter.")
        (
            self.hubble_constant,
            self.baryon_density,
            self.cold_dark_matter_density,
            self.curvature_density,
            self.dark_energy_w0,
            self.dark_energy_wa,
            self.photon_temperature,
            self.effective_neutrino_number,
            self.primordial_amplitude,
            self.primordial_tilt,
            self.primordial_pivot,
            self.reionization_optical_depth,
        ) = scalar_values
        self.scale = scale
        self.neutrinos = species
        self.transfer_fields = fields
        self.gauge = gauge_
        self.power_field = power_field_
        mapping = self.to_mapping(include_identity=False)
        self.model_form_id = canonical_fingerprint(
            {
                "kind": "external-linear-theory-physical-model",
                "scale": scale.scale_id,
                "neutrino_species_count": len(species),
                "neutrino_distributions": [item.distribution_id for item in species],
            }
        )
        self.request_id = canonical_fingerprint(
            {"kind": "external-linear-theory-request", **mapping}
        )

    @property
    def physical_state(self) -> CosmologyPhysicalState:
        neutrino_values = tuple(
            value
            for species in self.neutrinos
            for value in (
                species.mass_ev,
                species.temperature_ratio,
                species.degeneracy,
            )
        )
        neutrino_names = tuple(
            name
            for index in range(len(self.neutrinos))
            for name in (
                f"massive_neutrino_mass_{index}",
                f"massive_neutrino_temperature_ratio_{index}",
                f"massive_neutrino_degeneracy_{index}",
            )
        )
        values = jnp.asarray(
            (
                self.hubble_constant,
                self.baryon_density,
                self.cold_dark_matter_density,
                self.curvature_density,
                self.dark_energy_w0,
                self.dark_energy_wa,
                self.photon_temperature,
                self.effective_neutrino_number,
                self.primordial_amplitude,
                self.primordial_tilt,
                self.primordial_pivot,
                self.reionization_optical_depth,
                *neutrino_values,
            )
        )
        names = (
            "hubble_constant",
            "baryon_density",
            "cold_dark_matter_density",
            "curvature_density",
            "dark_energy_w0",
            "dark_energy_wa",
            "photon_temperature",
            "effective_neutrino_number",
            "primordial_amplitude",
            "primordial_tilt",
            "primordial_pivot",
            "reionization_optical_depth",
            *neutrino_names,
        )
        return CosmologyPhysicalState(
            values,
            names,
            self.scale.scale_id,
            categorical_ids=tuple(species.distribution_id for species in self.neutrinos),
        )

    @property
    def realization(self):
        return PhysicalDependencyProjection(self.physical_state.names).project(
            self.physical_state
        )

    def to_mapping(self, *, include_identity: bool = True) -> dict[str, object]:
        mapping: dict[str, object] = {
            "hubble_constant": self.hubble_constant,
            "baryon_density": self.baryon_density,
            "cold_dark_matter_density": self.cold_dark_matter_density,
            "curvature_density": self.curvature_density,
            "dark_energy_w0": self.dark_energy_w0,
            "dark_energy_wa": self.dark_energy_wa,
            "photon_temperature": self.photon_temperature,
            "effective_neutrino_number": self.effective_neutrino_number,
            "neutrinos": [species.to_mapping() for species in self.neutrinos],
            "primordial_amplitude": self.primordial_amplitude,
            "primordial_tilt": self.primordial_tilt,
            "primordial_pivot": self.primordial_pivot,
            "reionization_optical_depth": self.reionization_optical_depth,
            "transfer_fields": list(self.transfer_fields),
            "gauge": self.gauge,
            "power_field": self.power_field,
            "scale": self.scale.to_dict(),
        }
        if include_identity:
            mapping["request_id"] = self.request_id
            mapping["model_form_id"] = self.model_form_id
        return mapping


class CosmologyModelResult(StrictModule):
    transfer: LinearTransferTable
    power: MatterPowerTable
    thermodynamics: ThermodynamicsHistory | None
    standard_output: str = eqx.field(static=True)
    standard_error: str = eqx.field(static=True)
    return_code: int = eqx.field(static=True)


class SubprocessCosmologyModelBackend(AbstractExternalBackend, NonTrainableState):
    """Isolated JSON-request/NPZ-result precision-backend protocol."""

    application: str = eqx.field(static=True)
    arguments: tuple[str, ...] = eqx.field(static=True)
    timeout_seconds: float = eqx.field(static=True)
    backend_name: str = eqx.field(static=True)
    backend_version: str = eqx.field(static=True)
    numerical_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        application: str,
        /,
        *,
        arguments: tuple[str, ...] = ("{request}", "{output}"),
        timeout_seconds: float = 600.0,
        backend_name: str = "linear-theory-subprocess",
        backend_version: str = "user-provided",
        numerical_policy_id: str = "user-provided",
    ):
        executable = str(application).strip()
        arguments_ = tuple(str(argument) for argument in arguments)
        timeout = float(timeout_seconds)
        name = str(backend_name).strip()
        version = str(backend_version).strip()
        policy = str(numerical_policy_id).strip()
        if (
            not executable
            or not arguments_
            or not any("{request}" in argument for argument in arguments_)
            or not any("{output}" in argument for argument in arguments_)
            or not np.isfinite(timeout)
            or timeout <= 0.0
            or not name
            or not version
            or not policy
        ):
            raise ValueError("Linear-theory subprocess backend configuration is invalid.")
        self.application = executable
        self.arguments = arguments_
        self.timeout_seconds = timeout
        self.backend_name = name
        self.backend_version = version
        self.numerical_policy_id = policy

    @property
    def name(self) -> str:
        return self.backend_name

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            problem_kinds=("cosmology-linear-transfer", "cosmology-linear-power"),
            execution="host",
            host_only=True,
            supports_matrix_free=False,
            supports_assembled=False,
            coordinate_dtypes=("float64",),
            supports_plan_prepare_solve_refresh=False,
        )

    def availability(self, /) -> BackendAvailability:
        resolved = shutil.which(self.application)
        available = resolved is not None
        return BackendAvailability(
            capabilities=self.capabilities,
            available=available,
            requirement=self.application,
            reason="executable resolved" if available else "executable not found on PATH",
            versions=((self.backend_name, self.backend_version),),
        )

    def run(self, request: CosmologyModelRequest, /) -> CosmologyModelResult:
        if not isinstance(request, CosmologyModelRequest):
            raise TypeError("request must be CosmologyModelRequest.")
        availability = self.availability()
        if not availability.available:
            raise RuntimeError(
                f"Linear-theory backend executable {self.application!r} is unavailable."
            )
        with tempfile.TemporaryDirectory(prefix="phydrax-linear-theory-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            output_path = root / "result.npz"
            request_path.write_text(
                json.dumps(request.to_mapping(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            command = [
                self.application,
                *(
                    argument.format(request=request_path, output=output_path)
                    for argument in self.arguments
                ),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Linear-theory backend failed with code {completed.returncode}: "
                    f"{completed.stderr.strip()}"
                )
            if not output_path.is_file():
                raise RuntimeError(
                    "Linear-theory backend did not produce its NPZ result."
                )
            with np.load(output_path, allow_pickle=False) as output:
                required = {
                    "scale_factors",
                    "wavenumbers",
                    "transfer_values",
                    "power_values",
                    "scale_json",
                }
                missing = required.difference(output.files)
                if missing:
                    raise RuntimeError(
                        f"Linear-theory backend result is missing arrays: {sorted(missing)}"
                    )
                serialized_scale = np.asarray(output["scale_json"])
                if serialized_scale.shape != ():
                    raise RuntimeError(
                        "Linear-theory backend scale metadata must be scalar JSON."
                    )
                result_scale_payload = json.loads(str(serialized_scale.item()))
                if not isinstance(result_scale_payload, dict):
                    raise RuntimeError(
                        "Linear-theory backend scale metadata must decode to a mapping."
                    )
                result_scale = CosmologyScaleContract.from_dict(result_scale_payload)
                if result_scale.scale_id != request.scale.scale_id:
                    raise RuntimeError(
                        "Linear-theory backend result scale does not match its request."
                    )
                scales = jnp.asarray(output["scale_factors"])
                wavenumbers = jnp.asarray(output["wavenumbers"])
                transfer_values = jnp.asarray(output["transfer_values"])
                power_values = jnp.asarray(output["power_values"])
                has_thermodynamics = {
                    "ionization_fraction",
                    "baryon_temperature",
                    "opacity_derivative",
                    "visibility",
                }.issubset(output.files)
                thermodynamics_arrays = (
                    tuple(
                        jnp.asarray(output[name])
                        for name in (
                            "ionization_fraction",
                            "baryon_temperature",
                            "opacity_derivative",
                            "visibility",
                        )
                    )
                    if has_thermodynamics
                    else None
                )
        provenance = CosmologyProductProvenance(
            producer=self.backend_name,
            producer_version=self.backend_version,
            model_form_id=request.model_form_id,
            request_id=request.request_id,
            numerical_policy_id=self.numerical_policy_id,
            physics_policy_id="external-linear-theory",
            scale_id=result_scale.scale_id,
            source_kind="external",
            differentiation="constant",
        )
        transfer = LinearTransferTable(
            scales,
            wavenumbers,
            transfer_values,
            LinearTransferDescriptor(
                request.transfer_fields,
                gauge=request.gauge,
                normalization="relative-to-primordial-curvature",
            ),
            result_scale,
            provenance,
            request.realization,
        )
        power = MatterPowerTable(
            scales,
            wavenumbers,
            power_values,
            MatterPowerDescriptor(
                request.power_field,
                request.power_field,
                gauge=request.gauge,
                stage="linear",
            ),
            result_scale,
            provenance,
            request.realization,
        )
        thermodynamics = (
            ThermodynamicsHistory(
                scales,
                *thermodynamics_arrays,
                result_scale,
                provenance,
                request.realization,
            )
            if thermodynamics_arrays is not None
            else None
        )
        return CosmologyModelResult(
            transfer=transfer,
            power=power,
            thermodynamics=thermodynamics,
            standard_output=completed.stdout,
            standard_error=completed.stderr,
            return_code=completed.returncode,
        )


__all__ = [
    "CosmologyModelRequest",
    "CosmologyModelResult",
    "MassiveNeutrinoSpecies",
    "SubprocessCosmologyModelBackend",
]

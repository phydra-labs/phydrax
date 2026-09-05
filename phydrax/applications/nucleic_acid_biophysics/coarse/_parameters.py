# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rights-checked caller parameter artifacts for published rigid nucleotide equations.

Only independently admitted payloads are used; no calibration/table values ship.
The geometry, sequence strengths, temperature slope, salt scale and end-charge
convention are source data, not defaults inferred from a family label.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from ....atomistic._units import AtomisticUnitSystem
from ....qualification._reference import ReferenceArtifactManifest
from ._published import radial_support


FAMILY_MODELS = {
    "average-dna": {"DNA": "dna1"},
    "groove-salt-dna": {"DNA": "dna2"},
    "sequence-dna": {"DNA": "dna2"},
    "rna": {"RNA": "rna"},
    "dna-rna-hybrid": {"DNA": "dna2", "RNA": "rna", "HYBRID": "dna2"},
}
SITE_NAMES = ("backbone", "base", "stack3", "stack5", "coax")


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numeric values.")
    return array


def _angle(value):
    a, theta0, join = _vector(value, 3, "angular window")
    if a <= 0 or join <= 0 or not 0 <= theta0 <= 2 * math.pi or a * join * join >= 1:
        raise ValueError(
            "Angular windows need positive curvature/matching width and a positive central join."
        )


def _helicity(value):
    a, join = _vector(value, 2, "helicity window")
    if a <= 0 or not -1 < join < 0 or a * join * join >= 1:
        raise ValueError(
            "Helicity matching requires negative join and positive central value."
        )


def _radial(value, kind):
    array = _vector(value, 6, "radial well")
    if array[0] < 0 or array[5] <= 0:
        raise ValueError("Radial amplitude must be nonnegative and width positive.")
    radial_support(array, kind)


def _validate_profile(profile, model, *, screening_required, hybrid):
    required = {
        "backbone",
        "excluded",
        "stacking",
        "hydrogen-bond",
        "cross-stacking",
        "coaxial-stacking",
        "stacking_temperature_coefficient",
    }
    if hybrid:
        required -= {"backbone", "stacking", "stacking_temperature_coefficient"}
    if model == "rna":
        required |= {"p3", "p5"}
    if screening_required:
        required |= {"screening"}
    if set(profile) != required:
        raise ValueError(
            "Profile must supply precisely the complete published interaction decomposition."
        )
    if not hybrid:
        backbone = _vector(profile["backbone"], 3, "FENE [epsilon,r0,extension]")
        if np.any(backbone <= 0):
            raise ValueError(
                "FENE amplitude, equilibrium distance and extension must be positive."
            )
    if set(profile["excluded"]) != {"back-back", "base-base", "back-base"}:
        raise ValueError(
            "Excluded volume requires all three independently parameterized site pairs."
        )
    for values in profile["excluded"].values():
        epsilon, sigma, join = _vector(values, 3, "excluded [epsilon,sigma,join]")
        if epsilon < 0 or not 0 < join < sigma:
            raise ValueError(
                "Repulsive LJ smoothing requires 0 < join < sigma and nonnegative strength."
            )
    angle_names = {
        "hydrogen-bond": {"1", "2", "3", "4", "7", "8"},
        "stacking": {"5", "6", "9", "10"} if model == "rna" else {"4", "5", "6"},
        "cross-stacking": {"1", "2", "3", "7", "8"}
        if model == "rna"
        else {"1", "2", "3", "4", "7", "8"},
        "coaxial-stacking": {"1", "4", "5", "6"},
    }
    if hybrid:
        del angle_names["stacking"]
    for kind, names in angle_names.items():
        term = profile[kind]
        fields = {"radial", "angles"}
        if kind == "stacking" or (kind == "coaxial-stacking" and model != "dna2"):
            fields.add("helicity")
        if kind == "coaxial-stacking" and model == "dna2":
            fields.add("f6")
        if set(term) != fields or set(term["angles"]) != names:
            raise ValueError(
                "Angular product does not match the published chemistry-specific interaction."
            )
        _radial(
            term["radial"],
            "morse" if kind in ("stacking", "hydrogen-bond") else "harmonic",
        )
        for angle in term["angles"].values():
            _angle(angle)
        if "helicity" in fields:
            if len(term["helicity"]) != 2:
                raise ValueError(
                    "Published chiral product requires both helicity windows."
                )
            for values in term["helicity"]:
                _helicity(values)
        if "f6" in fields:
            A, B = _vector(term["f6"], 2, "coaxial f6")
            if A < 0 or not 0 <= B <= math.pi:
                raise ValueError(
                    "DNA2 coaxial quadratic must have nonnegative curvature and physical onset."
                )
    if not hybrid and not math.isfinite(
        float(profile["stacking_temperature_coefficient"])
    ):
        raise ValueError("Stacking temperature coefficient must be finite.")
    if model == "rna":
        for name in ("p3", "p5"):
            vector = _vector(profile[name], 3, name)
            if not np.isclose(np.sum(vector * vector), 1, atol=2e-3):
                raise ValueError(
                    "RNA backbone orientation vectors must be unit vectors in the declared body frame."
                )
    if screening_required:
        screen = profile["screening"]
        if set(screen) != {
            "prefactor",
            "length_per_sqrt_temperature_over_molar",
            "terminal_charge_factor",
        }:
            raise ValueError(
                "Screening needs source-calibrated prefactor, condition scale and terminal charge factor."
            )
        values = np.asarray(list(screen.values()), dtype=float)
        if (
            not np.isfinite(values).all()
            or np.any(values <= 0)
            or screen["terminal_charge_factor"] > 1
        ):
            raise ValueError(
                "Screening constants must be positive; end-charge factor cannot exceed one."
            )


@dataclass(frozen=True, init=False)
class NucleotideParameterArtifact:
    """Immutable source bytes for the independently implemented published equations.

    JSON fields: family, source_model, temperature, salt_concentration, salt_unit,
    geometry, profiles, sequence_strengths. ``geometry`` holds five physical site
    offsets per chemistry; ``profiles`` holds all f1/f2/f3/f4/f5/f6 inputs. Salt
    is mol/litre; temperature is in the explicit AtomisticUnitSystem scale.
    Screening length = declared scale * sqrt(temperature/salt_concentration).
    Cross-acid strengths and parameters are mandatory and never averaged.
    """

    raw_payload: bytes
    manifest: ReferenceArtifactManifest
    units: AtomisticUnitSystem
    family: str
    source_model: str
    temperature: float
    salt_concentration: float

    def __init__(
        self,
        manifest,
        payload: bytes,
        units,
        /,
        *,
        commercial_use=False,
        redistribution=False,
        training_use=False,
        export=False,
    ):
        if not isinstance(manifest, ReferenceArtifactManifest) or not isinstance(
            units, AtomisticUnitSystem
        ):
            raise TypeError(
                "Parameters require ReferenceArtifactManifest and AtomisticUnitSystem."
            )
        if not isinstance(payload, bytes):
            raise TypeError("Parameter payload must be immutable JSON bytes.")
        if (
            len(payload) != manifest.size_bytes
            or hashlib.new(manifest.checksum_algorithm, payload).hexdigest()
            != manifest.checksum
        ):
            raise ValueError("Parameter bytes do not match their source manifest.")
        manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        data = json.loads(payload)
        if set(data) != {
            "family",
            "source_model",
            "temperature",
            "salt_concentration",
            "salt_unit",
            "geometry",
            "profiles",
            "sequence_strengths",
        }:
            raise ValueError(
                "Parameter payload must explicitly specify the full published model and conditions."
            )
        family = data["family"]
        if family not in FAMILY_MODELS:
            raise ValueError("Unsupported nucleotide model family.")
        if not isinstance(data["source_model"], str) or not data["source_model"].strip():
            raise ValueError("A source-model identity is required.")
        temperature, salt = float(data["temperature"]), float(data["salt_concentration"])
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or not math.isfinite(salt)
            or salt < 0
        ):
            raise ValueError(
                "Temperature must be positive and salt concentration nonnegative."
            )
        if data["salt_unit"] != "mole/litre":
            raise ValueError(
                "Published screening/condition convention requires salt in mole/litre."
            )
        models = FAMILY_MODELS[family]
        chemistry = set(models) - {"HYBRID"}
        if (
            set(data["geometry"]) != chemistry
            or set(data["profiles"]) != set(models)
            or set(data["sequence_strengths"]) != set(models)
        ):
            raise ValueError(
                "Geometry and parameter profiles must cover the exact family chemistry."
            )
        for polymer in chemistry:
            geometry = data["geometry"][polymer]
            if set(geometry) != set(SITE_NAMES):
                raise ValueError(
                    "Five explicitly named physical interaction sites are required."
                )
            vectors = {key: _vector(value, 3, key) for key, value in geometry.items()}
            if np.any(vectors["base"][1:] != 0) or vectors["base"][0] <= 0:
                raise ValueError(
                    "The hydrogen-bond site defines the positive body a1 axis."
                )
            if polymer == "DNA":
                if not np.array_equal(
                    vectors["stack3"], vectors["stack5"]
                ) or not np.array_equal(vectors["stack3"], vectors["coax"]):
                    raise ValueError("DNA uses one common stacking/coaxial site.")
                if np.any(vectors["stack3"][1:] != 0) or vectors["backbone"][2] != 0:
                    raise ValueError(
                        "DNA site geometry must match the in-plane published body convention."
                    )
                if models["DNA"] == "dna1" and np.any(vectors["backbone"][1:] != 0):
                    raise ValueError(
                        "Original average DNA sites are collinear; unequal grooves require DNA2."
                    )
                if models["DNA"] == "dna2" and vectors["backbone"][1] == 0:
                    raise ValueError(
                        "DNA2 requires an explicitly displaced unequal-groove backbone site."
                    )
            elif np.array_equal(vectors["stack3"], vectors["stack5"]):
                raise ValueError(
                    "RNA requires distinct directed 3-prime and 5-prime stacking sites."
                )
        for name, model in models.items():
            screened = model == "dna2" or family == "dna-rna-hybrid"
            if screened and salt <= 0:
                raise ValueError(
                    "Screened models require positive molar salt concentration."
                )
            _validate_profile(
                data["profiles"][name],
                model,
                screening_required=screened,
                hybrid=name == "HYBRID",
            )
            if name != "HYBRID":
                coefficient = data["profiles"][name]["stacking_temperature_coefficient"]
                if 1 + coefficient * units.boltzmann_constant * temperature <= 0:
                    raise ValueError(
                        "Effective stacking strength is nonpositive at the declared temperature."
                    )
            strength = data["sequence_strengths"][name]
            required_strengths = (
                {"hydrogen-bond"} if name == "HYBRID" else {"stacking", "hydrogen-bond"}
            )
            if set(strength) != required_strengths:
                raise ValueError(
                    "Sequence strengths must match the permitted bonded/nonbonded chemistry."
                )
            for table in strength.values():
                values = np.asarray(table, dtype=float)
                if (
                    values.shape != (4, 4)
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                ):
                    raise ValueError(
                        "Sequence strengths require finite nonnegative 4x4 matrices."
                    )
            if family in ("average-dna", "groove-salt-dna"):
                stack = np.asarray(strength["stacking"])
                hb = np.asarray(strength["hydrogen-bond"])
                if not np.all(stack == stack[0, 0]) or not np.all(
                    hb[[0, 1, 2, 3], [3, 2, 1, 0]] == hb[0, 3]
                ):
                    raise ValueError(
                        "Average DNA cannot silently carry sequence-dependent well strengths."
                    )
        for name, value in (
            ("raw_payload", payload),
            ("manifest", manifest),
            ("units", units),
            ("family", family),
            ("source_model", data["source_model"]),
            ("temperature", temperature),
            ("salt_concentration", salt),
        ):
            object.__setattr__(self, name, value)

    def data(self):
        """Host-only fresh parse; mutation cannot change the admitted artifact."""
        return json.loads(self.raw_payload)

    @property
    def calibration_gate(self):
        return (
            "Equations are implemented; published numerical equivalence, "
            "duplex/mechanical observables and physical clock calibration require "
            "independently cleared parameters and qualification data."
        )


def nucleotide_reference_sites(construct, parameters):
    """Compile five physical sites plus three exact differential frame markers.

    Frame markers have zero physical mass/charge and are not interaction sites.
    Their forces arise solely from orientation-dependent energy differentiation.
    They enable collinear DNA1 sites without losing spin about the base axis.
    """
    geometry = parameters.data()["geometry"]
    rows = []
    for polymer, sequence in zip(
        construct.polymer_types, construct.sequences, strict=True
    ):
        sites = [geometry[polymer][name] for name in SITE_NAMES]
        rows.extend(
            [sites + [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]] * len(sequence)
        )
    return np.asarray(rows, dtype=float)


__all__ = ["NucleotideParameterArtifact", "nucleotide_reference_sites"]

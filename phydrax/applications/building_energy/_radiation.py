# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Basis-indexed numerical transport; external producers never claim derivatives."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...ein import contract
from ...units import ONE, UnitDefinition
from ._model import _text


class RadiativeBasis(StrictModule):
    labels: tuple[str, ...] = eqx.field(static=True)
    channels: tuple[str, ...] = eqx.field(static=True)
    measure: str = eqx.field(static=True)
    measure_unit: UnitDefinition = eqx.field(static=True)
    weights: Array
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: Sequence[str],
        *,
        basis_id: str,
        measure: str,
        weights: ArrayLike,
        measure_unit: UnitDefinition = ONE,
        channels: Sequence[str] = ("red", "green", "blue"),
    ):
        labels_, channels_ = tuple(labels), tuple(channels)
        if (
            not labels_
            or len(set(labels_)) != len(labels_)
            or not channels_
            or len(set(channels_)) != len(channels_)
        ):
            raise ValueError(
                "Radiative basis requires unique nonempty sample and channel labels."
            )
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(labels_),) or not np.all(np.isfinite(w) & (w > 0)):
            raise ValueError("Radiative measure weights must be finite and positive.")
        self.labels, self.channels = labels_, channels_
        self.measure, self.measure_unit = _text(measure, "measure"), measure_unit
        self.weights = jnp.asarray(w)
        self.basis_id = canonical_fingerprint(
            {
                "basis": _text(basis_id, "basis_id"),
                "labels": labels_,
                "channels": channels_,
                "measure": measure,
                "unit": measure_unit.unit_id,
                "weights": w.tolist(),
            }
        )


class RadiativeOperator(StrictModule):
    """A discrete coefficient map: values already contain any quadrature weights."""

    values: Array
    source: RadiativeBasis
    target: RadiativeBasis
    input_unit: UnitDefinition = eqx.field(static=True)
    output_unit: UnitDefinition = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        source: RadiativeBasis,
        target: RadiativeBasis,
        *,
        input_unit: UnitDefinition,
        output_unit: UnitDefinition,
        provenance: Sequence[str] = (),
    ):
        x = jnp.asarray(values)
        shape = (len(target.labels), len(source.labels), len(source.channels))
        if source.channels != target.channels or x.shape != shape:
            raise ValueError(
                f"Radiative matrix needs aligned channels and shape {shape}."
            )
        self.values = eqx.error_if(
            x, ~jnp.all(jnp.isfinite(x)), "Radiative coefficients must be finite."
        )
        self.source, self.target = source, target
        self.input_unit, self.output_unit = input_unit, output_unit
        self.provenance = tuple(provenance)

    @classmethod
    def from_kernel(
        cls, kernel: ArrayLike, source: RadiativeBasis, target: RadiativeBasis, **kwargs
    ):
        """Discretize an integral kernel using the source basis' explicit measure."""
        return cls(
            jnp.asarray(kernel) * source.weights[None, :, None], source, target, **kwargs
        )

    def apply(self, coefficients: ArrayLike) -> Array:
        x = jnp.asarray(coefficients)
        if x.shape != (len(self.source.labels), len(self.source.channels)):
            raise ValueError(
                "Radiative coefficients do not match the declared source basis."
            )
        return contract("oic,ic->oc", self.values, x)


class RadiativeComposition(StrictModule):
    """Execution-order factors, applied without materializing a dense product."""

    factors: tuple[RadiativeOperator, ...]

    def __init__(self, factors: Sequence[RadiativeOperator]):
        factors_ = tuple(factors)
        if not factors_:
            raise ValueError("Radiative composition needs at least one factor.")
        for left, right in zip(factors_[:-1], factors_[1:], strict=True):
            if (
                left.target.basis_id != right.source.basis_id
                or left.output_unit.unit_id != right.input_unit.unit_id
            ):
                raise ValueError(
                    "Radiative factors require identical intermediate basis, measure, and units."
                )
        self.factors = factors_

    def apply(self, coefficients: ArrayLike) -> Array:
        value = jnp.asarray(coefficients)
        for factor in self.factors:
            value = factor.apply(value)
        return value

    def materialize(self) -> RadiativeOperator:
        matrix = self.factors[0].values
        for factor in self.factors[1:]:
            matrix = contract("oic,itc->otc", factor.values, matrix)
        return RadiativeOperator(
            matrix,
            self.factors[0].source,
            self.factors[-1].target,
            input_unit=self.factors[0].input_unit,
            output_unit=self.factors[-1].output_unit,
            provenance=tuple(p for factor in self.factors for p in factor.provenance),
        )


def import_radiance_matrix(
    data: bytes | str,
    source: RadiativeBasis,
    target: RadiativeBasis,
    *,
    input_unit: UnitDefinition,
    output_unit: UnitDefinition,
    provenance: Sequence[str] = (),
) -> RadiativeOperator:
    """Import Radiance/Frads ASCII matrix output, validating header and all values.

    RGB channels are not silently converted to photometric or thermal power.
    Any channel integration/conversion must be an explicit numerical factor.
    """
    raw = data.encode() if isinstance(data, str) else data
    text = raw.decode("ascii")
    lines = text.splitlines()
    metadata = {}
    body_start = None
    for i, line in enumerate(lines):
        if not line.strip() and metadata:
            body_start = i + 1
            break
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    if body_start is None or not all(
        key in metadata for key in ("NROWS", "NCOLS", "NCOMP", "FORMAT")
    ):
        raise ValueError(
            "Radiance matrix requires dimension/component/format header and blank separator."
        )
    if metadata["FORMAT"] != "ascii":
        raise ValueError(
            "Only explicitly ASCII Radiance matrices are accepted; request -fa from producer."
        )
    shape = (int(metadata["NROWS"]), int(metadata["NCOLS"]), int(metadata["NCOMP"]))
    expected = (len(target.labels), len(source.labels), len(source.channels))
    if shape != expected:
        raise ValueError(
            "Radiance header dimensions do not match declared bases/channels."
        )
    tokens = " ".join(lines[body_start:]).split()
    if len(tokens) != int(np.prod(shape)):
        raise ValueError("Radiance matrix payload is truncated or has trailing values.")
    matrix = np.asarray([float(value) for value in tokens]).reshape(shape)
    return RadiativeOperator(
        matrix,
        source,
        target,
        input_unit=input_unit,
        output_unit=output_unit,
        provenance=(
            *provenance,
            "radiance-ascii-import",
            "sha256:" + hashlib.sha256(raw).hexdigest(),
            "foreign-execution:nondifferentiable",
        ),
    )


def produce_radiance_matrix(
    executable,
    args: Sequence[str],
    source: RadiativeBasis,
    target: RadiativeBasis,
    *,
    inputs: Mapping[str, bytes],
    input_unit: UnitDefinition,
    output_unit: UnitDefinition,
    stdin: bytes = b"",
    output_path: str | None = None,
    timeout: float = 120,
    environment: Mapping[str, str] | None = None,
):
    """Run a pinned bounded Radiance (or Frads CLI) matrix producer and import bytes.

    Producer arguments/scenes are explicit; no shell or implicit binary search.
    The returned external run record retains executable/license/content evidence.
    """
    from ...interchange.energy_runtime import run_radiance_command

    run = run_radiance_command(
        executable,
        tuple(args),
        inputs=inputs,
        stdin=stdin,
        outputs=() if output_path is None else (output_path,),
        timeout=timeout,
        environment=environment,
    )
    run.require_success()
    payload = run.stdout if output_path is None else run.output(output_path)
    operator = import_radiance_matrix(
        payload,
        source,
        target,
        input_unit=input_unit,
        output_unit=output_unit,
        provenance=(f"executable:sha256:{executable.sha256}",),
    )
    return operator, run


def produce_uniform_sky_reference(
    oconv, rtrace, *, environment: Mapping[str, str] | None = None, timeout: float = 120
):
    """Measure an upward irradiance sensor under a uniform unit-radiance hemisphere.

    The analytic result is π per RGB channel. Returned coefficients are actual
    Radiance output, not the analytic answer. Both bounded process records are
    retained. This qualifies one diffuse angular integral, not arbitrary scenes.
    """
    from ...interchange.energy_runtime import run_radiance_command
    from ...units import derived_unit, JOULE, METER, RADIAN, SECOND

    scene = (
        b"void glow sky_glow\n0\n0\n4 1 1 1 0\nsky_glow source sky\n0\n0\n4 0 0 1 180\n"
    )
    octree = run_radiance_command(
        oconv,
        ("-f", "sky.rad"),
        inputs={"sky.rad": scene},
        timeout=timeout,
        environment=environment,
    )
    trace = run_radiance_command(
        rtrace,
        ("-I+", "-h-", "-ab", "1", "-ad", "4096", "-aa", "0", "-ov", "sky.oct"),
        inputs={"sky.oct": octree.stdout},
        stdin=b"0 0 0 0 0 1\n",
        timeout=timeout,
        environment=environment,
    )
    tokens = trace.stdout.decode("ascii").split()
    if len(tokens) != 3:
        raise ValueError(
            "Radiance irradiance qualification requires exactly one RGB sensor result."
        )
    source = RadiativeBasis(
        ("uniform-upper-hemisphere",),
        basis_id="uniform-sky",
        measure="solid-angle",
        weights=(2 * np.pi,),
        measure_unit=derived_unit("sr", ((RADIAN, 2),)),
    )
    target = RadiativeBasis(
        ("upward-irradiance-sensor",),
        basis_id="upward-sensor",
        measure="point-evaluation",
        weights=(1,),
    )
    irradiance = derived_unit("W/m²", ((JOULE, 1), (SECOND, -1), (METER, -2)))
    radiance = derived_unit(
        "W/(m² sr)", ((JOULE, 1), (SECOND, -1), (METER, -2), (RADIAN, -2))
    )
    payload = "NROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n" + " ".join(tokens) + "\n"
    operator = import_radiance_matrix(
        payload,
        source,
        target,
        input_unit=radiance,
        output_unit=irradiance,
        provenance=(
            f"oconv:sha256:{oconv.sha256}",
            f"rtrace:sha256:{rtrace.sha256}",
            "uniform-unit-sky-irradiance-reference",
        ),
    )
    return operator, (octree, trace)


def radiative_heat_gains(
    irradiance: ArrayLike,
    basis: RadiativeBasis,
    *,
    unit: UnitDefinition,
    spectral_weights: ArrayLike,
    receiving_area: ArrayLike,
    absorption_fraction: ArrayLike,
    heat_distribution: ArrayLike,
) -> Array:
    """Convert an explicit irradiance-band response into inward nodal watts.

    Areas are m², absorption fractions are per receiver/channel, and spectral
    weights declare how the input channel convention integrates to broadband
    irradiance. In particular, arbitrary Radiance RGB has no implicit thermal
    conversion. The distribution's receiver columns must each sum to one.
    """
    from ...units import conversion_factor, derived_unit, JOULE, METER, SECOND

    target_unit = derived_unit("W/m²", ((JOULE, 1), (SECOND, -1), (METER, -2)))
    values = jnp.asarray(irradiance) * float(conversion_factor(unit, target_unit))
    count, channels = len(basis.labels), len(basis.channels)
    area, weights = jnp.asarray(receiving_area), jnp.asarray(spectral_weights)
    absorption, distribution = (
        jnp.asarray(absorption_fraction),
        jnp.asarray(heat_distribution),
    )
    if (
        values.shape != (count, channels)
        or area.shape != (count,)
        or weights.shape != (channels,)
        or absorption.shape != values.shape
        or distribution.ndim != 2
        or distribution.shape[1] != count
    ):
        raise ValueError("Radiative heat arrays must match receiver/channel/node axes.")
    invalid = (
        jnp.any(~jnp.isfinite(values) | (values < 0))
        | jnp.any(~jnp.isfinite(area) | (area <= 0))
        | jnp.any(~jnp.isfinite(weights) | (weights < 0))
        | jnp.any(~jnp.isfinite(absorption) | (absorption < 0) | (absorption > 1))
        | jnp.any(~jnp.isfinite(distribution) | (distribution < 0))
        | jnp.any(jnp.abs(jnp.sum(distribution, axis=0) - 1) > 1e-8)
    )
    values = eqx.error_if(
        values,
        invalid,
        "Radiative heat response must be finite, passive, and conservative.",
    )
    receiver_heat = contract("sc,sc,c,s->s", values, absorption, weights, area)
    return contract("ns,s->n", distribution, receiver_heat)

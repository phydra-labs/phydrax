#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native AIMNet2 architecture and scalar-energy potential.

This is the AIMNetcentral 0.2.0 ``AIMNet2`` forward
(``aimnet/models/aimnet2.py``) executed natively:

1. species embedding ``a = afv[Z]`` (``nfeature`` or, with ``d2features``,
   ``nfeature x nshifts`` per atom; padding rows are zero);
2. ``AEVSV`` scalar/vector Gaussian environment and the fused ``ConvSV``
   contractions of atom (``conv_a``) and charge (``conv_q``) features;
3. GELU MLP passes. Pass 0 sees ``[a, conv_a]``, later passes
   ``[a, conv_a, q, conv_q]``. Every pass but the last returns
   ``[q_raw, f_raw, delta_a]``: charges are set (pass 0) or incremented, the
   flexibility is ``f = f_raw^2``, the neutral-charge-equilibration update
   ``q <- q + f (Q - sum q) / (sum f + 1e-6)`` runs per case and channel, and
   ``a <- a + delta_a``. The last pass returns the AIM vector. Pass 0 ends
   linearly; later passes apply GELU to their output as the source does;
4. ordered output stages: energy head, atomic shifts, atomic sum, embedded
   short-range Coulomb subtraction, charge moments, ``DispParam`` and
   ``D3TS``.

Two charge channels (``NSE``) take targets ``Q/2 + (M-1)/2`` and
``Q/2 - (M-1)/2`` from total charge ``Q`` and physical multiplicity ``M``;
total and spin charges are the channel sum and difference. Because the
source adds ``1e-6`` to ``sum f``, the equilibrated charges miss the target by
``1e-6 (Q - sum q_u) / (sum f + 1e-6)``: that residual is published as
``CHARGE_RESIDUAL`` (target minus achieved), never hidden by an exact-charge
claim.

The model energy is exactly the source module output. Calculator-composed
terms (full Coulomb from the predicted charges when the short-range part is
embedded, tabulated DFT-D3) are separate scientific terms recorded by the
source conversion; they are never added here, so nothing is counted twice.
Every term runs on one full directed graph whose radius covers all stage
radii; the AEV, short-range Coulomb and ``D3TS`` stages each mask their own
radius explicitly.

Owner-region (domain) execution decomposes the same passes for
``atomistic._feature_execution``: each pass gathers source atom features (and,
after pass 0, charges) once, runs the receiver environment and MLP on owned
rows, and a declared system stage sums ``q`` and ``f`` per case and channel
over owners before the identical equilibration; the embedded pair stages are
one more exchange of final total charges and TS coefficients. Moments reduce
their owner partials the same way (`AIMNet2Potential.owner_moments`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, assert_never, NamedTuple, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._doc import DOC_KEY0
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import register_artifact_value
from ..._model._fixed_binding import require_identical
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._conditioning import (
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionLevel,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
    SPIN_MULTIPLICITY_PORT,
    TOTAL_CHARGE_PORT,
)
from ...atomistic._dispersion_ts import (
    SOURCE_BOHR_ANGSTROM,
    SOURCE_HARTREE_ELECTRONVOLT,
    TSDispersionPotential,
    TSDispersionReference,
)
from ...atomistic._feature_execution import (
    AtomisticSystemStage,
    owner_system_sum,
    OwnerReductionMode,
)
from ...atomistic._graph import AtomisticGraphExecutionPlan, case_energy_totals
from ...atomistic._hessian import (
    atomistic_edge_distance_locus,
    ATOMISTIC_NONCOINCIDENT_SUPPORT,
)
from ...atomistic._model_inputs import AtomisticGraphInputs
from ...atomistic._potential import (
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
)
from ...atomistic._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticSpeciesKind,
)
from ...ein import contract
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedPayloadSpec,
    StreamedRelationEvidence,
    StreamedRelationResult,
)
from ...typing import checked, PRNGKey
from ...units import (
    ANGSTROM,
    BOHR,
    CHARGE,
    derived_unit,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    ENERGY,
    HARTREE,
    LENGTH,
    UnitDefinition,
)
from ._aimnet2_features import (
    aimnet2_environment,
    AIMNet2ConvSV,
    AIMNet2RadialBasis,
    source_ahg_initialization,
    source_cosine_cutoff,
    SOURCE_CUTOFF_DISTANCE_FLOOR,
    source_default_eta,
    source_default_shifts,
    SOURCE_MINIMUM_SHIFT,
)


AIMNET2_SOURCE_RELEASE = "aimnetcentral==0.2.0"
AIMNET2_SOURCE_COMMIT = "b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55"
# ``Embedding(num_embeddings=64)`` and ``AtomicShift(num_types=64)``.
AIMNET2_SPECIES_CAPACITY = 64
# ``ops.nse(..., epsilon=1.0e-6)`` in ``AIMNet2._update_q`` (elementary charge).
AIMNET2_CHARGE_EPSILON = 1.0e-6
# ``constants.half_Hartree * constants.Bohr``: ordered-pair Coulomb factor (eV A).
AIMNET2_COULOMB_PAIR_FACTOR = 0.5 * SOURCE_HARTREE_ELECTRONVOLT * SOURCE_BOHR_ANGSTROM
# ``ops.exp_cutoff``: ``exp(-1 / (1 - clamp(d / rc, 0, 1 - 1e-6)^2)) / e^-1``.
_EXP_CUTOFF_CEILING = 1.0 - 1.0e-6
_EXP_CUTOFF_NORMALIZATION = 0.36787944117144233
_METHOD_ID = "negative-position-gradient-of-aimnet2-source-model-energy"

# Explicit source units of every dimensioned output port (by semantic id).
_OUTPUT_UNITS: dict[str, UnitDefinition] = {
    "atomistic.aimnet2.atomic-charge": ELEMENTARY_CHARGE,
    "atomistic.aimnet2.atomic-spin-charge": ELEMENTARY_CHARGE,
    "atomistic.aimnet2.charge-residual": ELEMENTARY_CHARGE,
    "atomistic.aimnet2.charge-dipole": derived_unit(
        "e*angstrom", ((ELEMENTARY_CHARGE, 1), (ANGSTROM, 1))
    ),
    "atomistic.aimnet2.charge-quadrupole": derived_unit(
        "e*angstrom^2", ((ELEMENTARY_CHARGE, 1), (ANGSTROM, 2))
    ),
    "atomistic.aimnet2.atomic-ts-c6": derived_unit(
        "hartree*bohr^6", ((HARTREE, 1), (BOHR, 6))
    ),
}
_CHARGE_PORT = ValuePort(
    "atomistic.aimnet2.atomic-charge",
    event_shape=(),
    component_ids=("charge",),
    representation="aimnet2-equilibrated-atomic-charge",
    dimensions=(CHARGE,),
    normalization_id="identity",
)
_SPIN_CHARGE_PORT = ValuePort(
    "atomistic.aimnet2.atomic-spin-charge",
    event_shape=(),
    component_ids=("spin_charge",),
    representation="aimnet2-alpha-minus-beta-channel-charge",
    dimensions=(CHARGE,),
    normalization_id="identity",
)
_AIM_PORT_SEMANTIC = "atomistic.aimnet2.aim-vector"


def _residual_port(channels: int, /) -> ValuePort:
    components = ("charge",) if channels == 1 else ("alpha", "beta")
    return ValuePort(
        "atomistic.aimnet2.charge-residual",
        event_shape=(channels,),
        component_ids=components,
        representation="target-minus-achieved-source-epsilon-residual",
        dimensions=(CHARGE,) * channels,
        normalization_id="identity",
    )


def _moment_port(kind: str, center: bool, /) -> ValuePort:
    origin = "mass-center" if center else "coordinate-origin"
    if kind == "dipole":
        return ValuePort(
            "atomistic.aimnet2.charge-dipole",
            event_shape=(3,),
            component_ids=("x", "y", "z"),
            representation=f"sum-q-r-about-{origin}",
            dimensions=(CHARGE * LENGTH,) * 3,
            frame_id="input-cartesian",
            normalization_id="identity",
        )
    return ValuePort(
        "atomistic.aimnet2.charge-quadrupole",
        event_shape=(6,),
        component_ids=("xx", "yy", "zz", "xy", "yz", "zx"),
        representation=f"sum-q-r-r-diagonal-mean-removed-about-{origin}",
        dimensions=(CHARGE * LENGTH**2,) * 6,
        frame_id="input-cartesian",
        normalization_id="identity",
    )


_C6_PORT = ValuePort(
    "atomistic.aimnet2.atomic-ts-c6",
    event_shape=(),
    component_ids=("c6",),
    representation="aimnet2-dispparam-c6-hartree-bohr6",
    dimensions=(ENERGY * LENGTH**6,),
    normalization_id="identity",
)
# Only alpha ratios enter D3TS, so the absolute polarizability scale is that of
# the admitted DispParam reference table; no physical unit is asserted for it.
_POLARIZABILITY_PORT = ValuePort(
    "atomistic.aimnet2.atomic-ts-polarizability-reference-scale",
    event_shape=(),
    component_ids=("alpha",),
    representation="aimnet2-dispparam-alpha-in-reference-table-scale-ratio-only",
    normalization_id="aimnet2-dispparam-reference-table",
)


class AIMNet2OutputKind(StrEnum):
    """Admitted source output modules (``aimnet.modules``) by meaning."""

    ENERGY_HEAD = "energy-head"  # Output(key_in="aim", key_out="energy", n_out=1)
    DISPERSION_HEAD = "dispersion-head"  # Output(key_in="aim", n_out=2) -> DispParam
    ATOMIC_SHIFT = "atomic-shift"
    ATOMIC_SUM = "atomic-sum"
    SHORT_RANGE_COULOMB = "short-range-coulomb"  # SRCoulomb: energy -= E_sr
    SHORT_RANGE_REPULSION = "short-range-repulsion"  # core.SRRep into energy
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"
    DISPERSION_PARAMETERS = "dispersion-parameters"  # DispParam
    TS_DISPERSION = "ts-dispersion"  # D3TS: energy += E_ts


# Source MLP activations (``torch.nn`` class semantics) and their parameters.
AIMNET2_ACTIVATIONS = frozenset(
    {"gelu", "gelu-tanh", "silu", "relu", "tanh", "sigmoid", "softplus"}
)
_PAIR_KINDS = (
    AIMNet2OutputKind.SHORT_RANGE_COULOMB,
    AIMNet2OutputKind.SHORT_RANGE_REPULSION,
    AIMNet2OutputKind.TS_DISPERSION,
)


def _activation_spec(
    activation: tuple[str, tuple[float, ...]], /
) -> tuple[str, tuple[float, ...]]:
    name, parameters = activation
    values = tuple(float(value) for value in parameters)
    if name not in AIMNET2_ACTIVATIONS:
        raise ValueError(f"Activation {name!r} is not an admitted source activation.")
    expected = 2 if name == "softplus" else 0
    if len(values) != expected or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Activation {name!r} needs {expected} finite parameters.")
    if name == "softplus" and values[0] == 0.0:
        raise ValueError("Softplus beta must be nonzero.")
    return name, values


def apply_activation(value: Array, activation: tuple[str, tuple[float, ...]], /) -> Array:
    """``torch.nn`` activation semantics, including ``Softplus`` thresholding."""
    name, parameters = activation
    match name:
        case "gelu":
            return jax.nn.gelu(value, approximate=False)
        case "gelu-tanh":
            return jax.nn.gelu(value, approximate=True)
        case "silu":
            return jax.nn.silu(value)
        case "relu":
            return jax.nn.relu(value)
        case "tanh":
            return jnp.tanh(value)
        case "sigmoid":
            return jax.nn.sigmoid(value)
        case "softplus":
            beta, threshold = parameters
            scaled = beta * value
            linear = scaled > threshold
            safe = jnp.where(linear, jnp.zeros((), value.dtype), scaled)
            return jnp.where(linear, value, jnp.log1p(jnp.exp(safe)) / beta)
    raise ValueError(f"Activation {name!r} is not admitted.")


def _check_stage_roles(
    kind: AIMNet2OutputKind,
    /,
    *,
    widths: tuple[int, ...],
    last_linear: bool,
    bias: bool,
    activation: tuple[str, tuple[float, ...]],
    reduce_sum: bool,
    frozen: bool,
    species_rows: int,
    center_coordinates: bool,
) -> None:
    """Only the stage kinds whose source module has a setting declare it."""
    head = kind in (AIMNet2OutputKind.ENERGY_HEAD, AIMNet2OutputKind.DISPERSION_HEAD)
    if not head and (widths or not last_linear or not bias or activation != ("gelu", ())):
        raise ValueError(f"{kind.value} stages declare no MLP.")
    if reduce_sum and kind not in (
        AIMNet2OutputKind.ATOMIC_SHIFT,
        AIMNet2OutputKind.SHORT_RANGE_REPULSION,
    ):
        raise ValueError("Only atomic shifts and SRRep declare reduce_sum.")
    if kind is not AIMNet2OutputKind.ATOMIC_SHIFT and (frozen or species_rows != 64):
        raise ValueError("Only atomic shifts declare frozen or species_rows.")
    if (
        isinstance(species_rows, bool)
        or not isinstance(species_rows, int)
        or (species_rows < 2)
    ):
        raise ValueError("species_rows must be an integer of at least two.")
    if center_coordinates and kind not in (
        AIMNet2OutputKind.DIPOLE,
        AIMNet2OutputKind.QUADRUPOLE,
    ):
        raise ValueError("Only charge moments declare center_coordinates.")


def _check_stage_envelopes(
    kind: AIMNet2OutputKind, envelope: str | None, cutoff_function: str | None, /
) -> None:
    """Exactly SRCoulomb declares an envelope and exactly SRRep a cutoff function."""
    if (envelope is not None) != (kind is AIMNet2OutputKind.SHORT_RANGE_COULOMB):
        raise ValueError("Exactly short-range Coulomb stages declare an envelope.")
    if envelope is not None and envelope not in ("exp", "cosine"):
        raise ValueError("Short-range Coulomb envelope must be 'exp' or 'cosine'.")
    if (cutoff_function is not None) != (kind is AIMNet2OutputKind.SHORT_RANGE_REPULSION):
        raise ValueError("Exactly SRRep stages declare a cutoff function.")
    if cutoff_function is not None and cutoff_function not in (
        "none",
        "exp_cutoff",
        "cosine_cutoff",
    ):
        raise ValueError("SRRep cutoff_fn must be none, exp_cutoff or cosine_cutoff.")


def _stage_keys(
    kind: AIMNet2OutputKind, key_in: str | None, key_out: str | None, /
) -> tuple[str, str]:
    """Source ``(key_in, key_out)`` of a stage; only dispersion and moment keys vary."""
    defaults = {
        AIMNet2OutputKind.ENERGY_HEAD: ("aim", "energy"),
        AIMNet2OutputKind.DISPERSION_HEAD: ("aim", "disp_param"),
        AIMNet2OutputKind.ATOMIC_SHIFT: ("energy", "energy"),
        AIMNet2OutputKind.ATOMIC_SUM: ("energy", "energy"),
        AIMNet2OutputKind.SHORT_RANGE_COULOMB: ("charges", "energy"),
        AIMNet2OutputKind.SHORT_RANGE_REPULSION: ("numbers", "energy"),
        AIMNet2OutputKind.DIPOLE: ("charges", "dipole"),
        AIMNet2OutputKind.QUADRUPOLE: ("charges", "quadrupole"),
        AIMNet2OutputKind.DISPERSION_PARAMETERS: ("disp_param", "disp_param"),
        AIMNet2OutputKind.TS_DISPERSION: ("disp_param", "energy"),
    }
    default_in, default_out = defaults[kind]
    source_in = default_in if key_in is None else str(key_in)
    source_out = default_out if key_out is None else str(key_out)
    fixed_in = kind not in (
        AIMNet2OutputKind.DISPERSION_PARAMETERS,
        AIMNet2OutputKind.TS_DISPERSION,
    )
    fixed_out = kind not in (
        AIMNet2OutputKind.DISPERSION_HEAD,
        AIMNet2OutputKind.DISPERSION_PARAMETERS,
        AIMNet2OutputKind.DIPOLE,
        AIMNet2OutputKind.QUADRUPOLE,
    )
    if (fixed_in and source_in != default_in) or (
        fixed_out and source_out != default_out
    ):
        raise ValueError(
            f"{kind.value} stage keys ({source_in!r} -> {source_out!r}) are not the "
            "admitted source data flow."
        )
    return source_in, source_out


def _ts_dispersion_constants(
    kind: AIMNet2OutputKind,
    s6: float,
    s8: float | None,
    a1: float | None,
    a2: float | None,
    /,
) -> tuple[float, float, float, float] | None:
    """Finite ``(s6, s8, a1, a2)`` of a ``D3TS`` stage; no other stage declares them."""
    if kind is AIMNet2OutputKind.TS_DISPERSION:
        if s8 is None or a1 is None or a2 is None:
            raise ValueError("TS dispersion requires s8, a1 and a2.")
        dispersion = (float(s6), float(s8), float(a1), float(a2))
        if not all(math.isfinite(value) for value in dispersion):
            raise ValueError("TS dispersion constants must be finite.")
        return dispersion
    if any(value is not None for value in (s8, a1, a2)) or s6 != 1.0:
        raise ValueError("Only TS dispersion stages declare dispersion constants.")
    return None


def _stage_pair_cutoff(
    kind: AIMNet2OutputKind, pair_cutoff: float | None, /
) -> float | None:
    """The finite positive radius exactly pair stages declare."""
    if (pair_cutoff is not None) != (kind in _PAIR_KINDS):
        raise ValueError("Exactly pair stages declare a pair_cutoff.")
    if pair_cutoff is None:
        return None
    radius = float(pair_cutoff)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("pair_cutoff must be finite and positive.")
    return radius


class AIMNet2OutputStage(StrictModule, NonTrainableState):
    """One source output module, in source execution order.

    Head stages declare their MLP (``hidden``, ``last_linear``, ``bias`` and
    activation); ``key_out`` of a dispersion head names the value its
    ``DispParam`` consumes. ``ATOMIC_SHIFT`` declares its table rows
    (``num_types``), whether the shifts are trainable (``frozen`` mirrors
    ``requires_grad=False``: a fixed-role table, never a detached geometry
    response) and ``reduce_sum`` (summed shifts added to a case energy).
    ``SHORT_RANGE_COULOMB`` uses the ``"exp"`` or ``"cosine"`` envelope.
    ``SHORT_RANGE_REPULSION`` is ``core.SRRep`` with its ``cutoff_function``.
    Pair stages declare ``pair_cutoff``: the radius beyond which they are
    exactly zero (envelopes) or the declared hard route radius (``D3TS`` and an
    un-enveloped ``SRRep``, which the source sums over its neighbor list).
    """

    name: str = eqx.field(static=True)
    kind: AIMNet2OutputKind = eqx.field(static=True)
    hidden: tuple[int, ...] = eqx.field(static=True)
    last_linear: bool = eqx.field(static=True)
    bias: bool = eqx.field(static=True)
    activation: tuple[str, tuple[float, ...]] = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    reduce_sum: bool = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)
    species_rows: int = eqx.field(static=True)
    envelope: str | None = eqx.field(static=True)
    cutoff_function: str | None = eqx.field(static=True)
    center_coordinates: bool = eqx.field(static=True)
    key_in: str = eqx.field(static=True)
    key_out: str = eqx.field(static=True)
    dispersion: tuple[float, float, float, float] | None = eqx.field(static=True)
    pair_cutoff: float | None = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: AIMNet2OutputKind | str,
        /,
        *,
        hidden: Sequence[int] = (),
        last_linear: bool = True,
        bias: bool = True,
        activation: tuple[str, tuple[float, ...]] = ("gelu", ()),
        reduce_sum: bool = False,
        frozen: bool = False,
        species_rows: int = 64,
        envelope: str | None = None,
        cutoff_function: str | None = None,
        center_coordinates: bool = False,
        key_in: str | None = None,
        key_out: str | None = None,
        s6: float = 1.0,
        s8: float | None = None,
        a1: float | None = None,
        a2: float | None = None,
        pair_cutoff: float | None = None,
    ) -> None:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Output stage names must be canonical nonempty text.")
        kind_ = AIMNet2OutputKind(kind)
        widths = tuple(int(width) for width in hidden)
        if any(width <= 0 for width in widths):
            raise ValueError("Output head hidden widths must be positive.")
        activation_ = _activation_spec(activation)
        for flag_name, flag in (
            ("last_linear", last_linear),
            ("bias", bias),
            ("reduce_sum", reduce_sum),
            ("frozen", frozen),
            ("center_coordinates", center_coordinates),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{flag_name} must be a boolean.")
        _check_stage_roles(
            kind_,
            widths=widths,
            last_linear=last_linear,
            bias=bias,
            activation=activation_,
            reduce_sum=reduce_sum,
            frozen=frozen,
            species_rows=species_rows,
            center_coordinates=center_coordinates,
        )
        _check_stage_envelopes(kind_, envelope, cutoff_function)
        source_in, source_out = _stage_keys(kind_, key_in, key_out)
        dispersion = _ts_dispersion_constants(kind_, s6, s8, a1, a2)
        radius = _stage_pair_cutoff(kind_, pair_cutoff)
        output_count = {
            AIMNet2OutputKind.ENERGY_HEAD: 1,
            AIMNet2OutputKind.DISPERSION_HEAD: 2,
        }.get(kind_, 0)
        self.name = name
        self.kind = kind_
        self.hidden = widths
        self.last_linear = last_linear
        self.bias = bias
        self.activation = activation_
        self.output_count = output_count
        self.reduce_sum = reduce_sum
        self.frozen = frozen
        self.species_rows = species_rows
        self.envelope = envelope
        self.cutoff_function = cutoff_function
        self.center_coordinates = center_coordinates
        self.key_in = source_in
        self.key_out = source_out
        self.dispersion = dispersion
        self.pair_cutoff = radius
        self.stage_id = canonical_fingerprint(
            {
                "kind": "aimnet2-output-stage",
                "name": name,
                "stage": kind_.value,
                "hidden": widths,
                "last_linear": last_linear,
                "bias": bias,
                "activation": activation_,
                "reduce_sum": reduce_sum,
                "frozen": frozen,
                "species_rows": species_rows,
                "envelope": envelope,
                "cutoff_function": cutoff_function,
                "center_coordinates": center_coordinates,
                "key_in": source_in,
                "key_out": source_out,
                "dispersion": dispersion,
                "pair_cutoff": radius,
            }
        )

    def rebuilt(self) -> AIMNet2OutputStage:
        """The constructor's value from this stage's own declaration."""
        s6, s8, a1, a2 = (
            (1.0, None, None, None) if self.dispersion is None else self.dispersion
        )
        return AIMNet2OutputStage(
            self.name,
            self.kind,
            hidden=self.hidden,
            last_linear=self.last_linear,
            bias=self.bias,
            activation=self.activation,
            reduce_sum=self.reduce_sum,
            frozen=self.frozen,
            species_rows=self.species_rows,
            envelope=self.envelope,
            cutoff_function=self.cutoff_function,
            center_coordinates=self.center_coordinates,
            key_in=self.key_in,
            key_out=self.key_out,
            s6=s6,
            s8=s8,
            a1=a1,
            a2=a2,
            pair_cutoff=self.pair_cutoff,
        )


def _pass_hidden(hidden: Sequence[Sequence[int]], /) -> tuple[tuple[int, ...], ...]:
    layers = tuple(tuple(int(width) for width in widths) for widths in hidden)
    if not layers:
        raise ValueError("AIMNet2 needs at least one hidden declaration.")
    # Source: MLP(hidden[0]), MLP(h) for h in hidden[1:-1], MLP(hidden[-1]);
    # the source MLP drops nonpositive widths (``hidden = [x for x in hidden if x > 0]``).
    passes = (layers[0], *layers[1:-1], layers[-1])
    return tuple(tuple(width for width in widths if width > 0) for widths in passes)


def _validate_stage_order(stages: Sequence[AIMNet2OutputStage], /) -> None:
    level = "none"
    produced: set[str] = set()
    names: set[str] = set()
    for stage in stages:
        if stage.name in names:
            raise ValueError(f"Duplicate output stage {stage.name!r}.")
        names.add(stage.name)
        match stage.kind:
            case AIMNet2OutputKind.ENERGY_HEAD:
                if level != "none":
                    raise ValueError("The energy head must define the energy first.")
                level = "atom"
            case AIMNet2OutputKind.DISPERSION_HEAD:
                if stage.key_out in produced or stage.key_out in (
                    "aim",
                    "energy",
                    "charges",
                ):
                    raise ValueError(f"Dispersion head key {stage.key_out!r} is reused.")
                produced.add(stage.key_out)
            case AIMNet2OutputKind.ATOMIC_SHIFT | AIMNet2OutputKind.SHORT_RANGE_REPULSION:
                required = "case" if stage.reduce_sum else "atom"
                if level != required:
                    raise ValueError(
                        f"{stage.kind.value} {stage.name!r} needs a {required}-level "
                        "energy."
                    )
            case AIMNet2OutputKind.ATOMIC_SUM:
                if level != "atom":
                    raise ValueError("Atomic sum needs an atom-level energy.")
                level = "case"
            case AIMNet2OutputKind.SHORT_RANGE_COULOMB | AIMNet2OutputKind.TS_DISPERSION:
                if level != "case":
                    raise ValueError(
                        f"{stage.kind.value} adds to a case energy after the atomic sum."
                    )
                if (
                    stage.kind is AIMNet2OutputKind.TS_DISPERSION
                    and stage.key_in not in produced
                ):
                    raise ValueError("D3TS consumes coefficients DispParam has not made.")
            case AIMNet2OutputKind.DISPERSION_PARAMETERS:
                if stage.key_in not in produced:
                    raise ValueError("DispParam consumes a head output that is absent.")
                if stage.key_out != stage.key_in and stage.key_out in produced:
                    raise ValueError("DispParam overwrites another dispersion value.")
                produced.add(stage.key_out)
            case AIMNet2OutputKind.DIPOLE | AIMNet2OutputKind.QUADRUPOLE:
                pass
    if level != "case":
        raise ValueError("An energy-bearing AIMNet2 model must end with a case energy.")
    kinds = [stage.kind for stage in stages]
    for kind in (
        AIMNet2OutputKind.DIPOLE,
        AIMNet2OutputKind.QUADRUPOLE,
        AIMNet2OutputKind.DISPERSION_PARAMETERS,
        AIMNet2OutputKind.TS_DISPERSION,
        AIMNet2OutputKind.SHORT_RANGE_COULOMB,
        AIMNet2OutputKind.SHORT_RANGE_REPULSION,
    ):
        if kinds.count(kind) > 1:
            raise ValueError(f"At most one {kind.value} stage is admitted.")
    if kinds.count(AIMNet2OutputKind.DISPERSION_HEAD) > 1:
        raise ValueError("At most one dispersion head is admitted.")


class AIMNet2Architecture(StrictModule, NonTrainableState):
    """Static source declaration of one AIMNet2 model.

    ``aev_cutoff`` is the scalar AEV radius ``rc_s``. The source's separate
    vector basis is never executed (see ``_aimnet2_features``); a declared one
    is kept by the source reconstruction as inert evidence. ``implemented_species``
    are the atomic numbers the model admits; others refuse per case.
    ``charged_systems=False`` (the rxn family contract) admits only zero total
    charge.
    """

    feature_count: int = eqx.field(static=True)
    d2_features: bool = eqx.field(static=True)
    shift_count: int = eqx.field(static=True)
    combination_count: int = eqx.field(static=True)
    pass_hidden: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    aim_size: int = eqx.field(static=True)
    charge_channels: int = eqx.field(static=True)
    aev_cutoff: float = eqx.field(static=True)
    implemented_species: tuple[int, ...] = eqx.field(static=True)
    charged_systems: bool = eqx.field(static=True)
    outputs: tuple[AIMNet2OutputStage, ...]
    architecture_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        feature_count: int,
        d2_features: bool,
        shift_count: int,
        combination_count: int,
        hidden: Sequence[Sequence[int]],
        aim_size: int,
        charge_channels: int,
        aev_cutoff: float,
        implemented_species: Sequence[int],
        outputs: Sequence[AIMNet2OutputStage],
        charged_systems: bool = True,
    ) -> None:
        counts = {
            "feature_count": feature_count,
            "shift_count": shift_count,
            "combination_count": combination_count,
            "aim_size": aim_size,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(d2_features, bool) or not isinstance(charged_systems, bool):
            raise TypeError("d2_features and charged_systems must be booleans.")
        if charge_channels not in (1, 2):
            raise ValueError("charge_channels must be 1 (closed shell) or 2 (NSE).")
        cutoff = float(aev_cutoff)
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("aev_cutoff must be finite and positive.")
        species = tuple(int(value) for value in implemented_species)
        if (
            not species
            or tuple(sorted(set(species))) != species
            or species[0] <= 0
            or species[-1] >= AIMNET2_SPECIES_CAPACITY
        ):
            raise ValueError(
                "implemented_species must be increasing atomic numbers in [1, 63]."
            )
        stages = tuple(outputs)
        if any(not isinstance(stage, AIMNet2OutputStage) for stage in stages):
            raise TypeError("outputs must contain AIMNet2OutputStage values.")
        _validate_stage_order(stages)
        passes = _pass_hidden(hidden)
        self.feature_count = feature_count
        self.d2_features = d2_features
        self.shift_count = shift_count
        self.combination_count = combination_count
        self.pass_hidden = passes
        self.aim_size = aim_size
        self.charge_channels = charge_channels
        self.aev_cutoff = cutoff
        self.implemented_species = species
        self.charged_systems = charged_systems
        self.outputs = stages
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "aimnet2-architecture",
                "source": AIMNET2_SOURCE_RELEASE,
                "feature_count": feature_count,
                "d2_features": d2_features,
                "shift_count": shift_count,
                "combination_count": combination_count,
                "pass_hidden": passes,
                "aim_size": aim_size,
                "charge_channels": charge_channels,
                "aev_cutoff": cutoff,
                "implemented_species": species,
                "charged_systems": charged_systems,
                "charge_epsilon": AIMNET2_CHARGE_EPSILON,
                "outputs": tuple(stage.stage_id for stage in stages),
            }
        )

    def rebuilt(self) -> AIMNet2Architecture:
        """The constructor's value from this declaration's own fields.

        ``pass_hidden`` is already the per-pass expansion of the source
        ``hidden`` list, which the constructor maps to itself.
        """
        return AIMNet2Architecture(
            feature_count=self.feature_count,
            d2_features=self.d2_features,
            shift_count=self.shift_count,
            combination_count=self.combination_count,
            hidden=self.pass_hidden,
            aim_size=self.aim_size,
            charge_channels=self.charge_channels,
            aev_cutoff=self.aev_cutoff,
            implemented_species=self.implemented_species,
            outputs=tuple(stage.rebuilt() for stage in self.outputs),
            charged_systems=self.charged_systems,
        )

    @property
    def feature_total(self) -> int:
        return self.feature_count * (self.shift_count if self.d2_features else 1)

    @property
    def pass_count(self) -> int:
        return len(self.pass_hidden)

    @property
    def cutoff(self) -> float:
        """Radius of the one directed graph covering every stage radius."""
        radii = [self.aev_cutoff]
        radii.extend(
            stage.pair_cutoff for stage in self.outputs if stage.pair_cutoff is not None
        )
        return max(radii)

    def stage(self, kind: AIMNet2OutputKind, /) -> AIMNet2OutputStage | None:
        for stage in self.outputs:
            if stage.kind is kind:
                return stage
        return None

    def pass_shapes(self) -> tuple[tuple[int, int, bool], ...]:
        """``(inputs, outputs, last_linear)`` of every charge-pass MLP."""
        channels = self.charge_channels
        conv = self.feature_count * (self.shift_count + self.combination_count)
        charge_conv = channels * (self.shift_count + self.combination_count)
        first = self.feature_total + conv
        later = first + channels + charge_conv
        update = self.feature_total + 2 * channels
        count = self.pass_count
        return tuple(
            (
                first if index == 0 else later,
                self.aim_size if index == count - 1 else update,
                index == 0,
            )
            for index in range(count)
        )

    def conditioning_spec(self) -> AtomisticConditioningSpec:
        """Total charge (and NSE multiplicity) the model consumes per case."""
        bound = None if self.charged_systems else 0.0
        fields = [
            AtomisticConditionField(
                "charge",
                TOTAL_CHARGE_PORT,
                unit=ELEMENTARY_CHARGE,
                dtype=jnp.float64,
                required=True,
                minimum=bound,
                maximum=bound,
            )
        ]
        if self.charge_channels == 2:
            fields.append(
                AtomisticConditionField(
                    "multiplicity",
                    SPIN_MULTIPLICITY_PORT,
                    dtype=jnp.float64,
                    required=True,
                    minimum=1.0,
                )
            )
        return AtomisticConditioningSpec(fields)


def _host(value: Array, /) -> np.ndarray:
    """A concrete host copy of one stored leaf (validation boundary only)."""
    return np.asarray(jax.device_get(value))


class AIMNet2MLP(StrictModule):
    """Source ``MLP``: ``Linear`` layers (weights ``(out, in)``) and activations."""

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    last_linear: bool = eqx.field(static=True)
    activation: tuple[str, tuple[float, ...]] = eqx.field(static=True)

    def __init__(
        self,
        weights: Sequence[ArrayLike],
        biases: Sequence[ArrayLike] | None,
        /,
        *,
        last_linear: bool,
        activation: tuple[str, tuple[float, ...]] = ("gelu", ()),
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        matrices = tuple(np.asarray(weight) for weight in weights)
        if not matrices:
            raise ValueError("An MLP needs at least one layer.")
        for index, matrix in enumerate(matrices):
            if matrix.dtype.kind != "f" or matrix.ndim != 2 or 0 in matrix.shape:
                raise ValueError("MLP weights must be nonempty (out, in) float matrices.")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("MLP weights must be finite.")
            if index and matrix.shape[1] != matrices[index - 1].shape[0]:
                raise ValueError("MLP layer widths do not chain.")
        vectors = () if biases is None else tuple(np.asarray(bias) for bias in biases)
        if biases is not None:
            if len(vectors) != len(matrices) or any(
                vector.dtype.kind != "f"
                or vector.shape != (matrix.shape[0],)
                or not np.all(np.isfinite(vector))
                for vector, matrix in zip(vectors, matrices, strict=True)
            ):
                raise ValueError("MLP biases must be finite and match their layers.")
        if not isinstance(last_linear, bool):
            raise TypeError("last_linear must be a boolean.")
        self.weights = tuple(jnp.asarray(matrix, dtype=dtype) for matrix in matrices)
        self.biases = tuple(jnp.asarray(vector, dtype=dtype) for vector in vectors)
        self.last_linear = last_linear
        self.activation = _activation_spec(activation)

    def rebuilt(self) -> AIMNet2MLP:
        """The constructor's value from this MLP's own stored leaves."""
        return AIMNet2MLP(
            [_host(weight) for weight in self.weights],
            [_host(bias) for bias in self.biases] if self.biases else None,
            last_linear=self.last_linear,
            activation=self.activation,
            dtype=self.weights[0].dtype,
        )

    @property
    def in_size(self) -> int:
        return int(self.weights[0].shape[1])

    @property
    def out_size(self) -> int:
        return int(self.weights[-1].shape[0])

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(int(weight.shape[0]) for weight in self.weights[:-1])

    def __call__(self, value: Array, /) -> Array:
        last = len(self.weights) - 1
        for index, weight in enumerate(self.weights):
            value = contract("...i,oi->...o", value, weight.astype(value.dtype))
            if self.biases:
                value = value + self.biases[index].astype(value.dtype)
            if not (self.last_linear and index == last):
                value = apply_activation(value, self.activation)
        return value


def _default_mlp(
    sizes: Sequence[int],
    last_linear: bool,
    key: PRNGKey,
    dtype: DTypeLike,
    /,
    *,
    bias: bool = True,
    activation: tuple[str, tuple[float, ...]] = ("gelu", ()),
) -> AIMNet2MLP:
    """Source initialization: ``xavier_normal_`` weights and zero biases."""
    keys = jr.split(key, len(sizes) - 1)
    weights = [
        np.asarray(
            jr.normal(keys[index], (sizes[index + 1], sizes[index]), dtype=jnp.float64)
            * math.sqrt(2.0 / (sizes[index] + sizes[index + 1]))
        )
        for index in range(len(sizes) - 1)
    ]
    biases = [np.zeros(size, dtype=np.float64) for size in sizes[1:]] if bias else None
    return AIMNet2MLP(
        weights, biases, last_linear=last_linear, activation=activation, dtype=dtype
    )


def _shift_table(shifts: ArrayLike, /) -> np.ndarray:
    values = np.asarray(shifts)
    if values.dtype.kind != "f" or values.ndim != 1 or values.size < 2:
        raise ValueError("Atomic shifts must be one floating table by species.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Atomic shifts must be finite.")
    return values


class AIMNet2AtomicShifts(StrictModule):
    """Source ``AtomicShift`` table (self-atomic energies, eV; trainable)."""

    shifts: Array

    def __init__(self, shifts: ArrayLike, /, *, dtype: DTypeLike = jnp.float64) -> None:
        self.shifts = jnp.asarray(_shift_table(shifts), dtype=dtype)

    def rebuilt(self) -> AIMNet2AtomicShifts:
        return AIMNet2AtomicShifts(_host(self.shifts), dtype=self.shifts.dtype)


class AIMNet2FrozenAtomicShifts(StrictModule, NonTrainableState):
    """Source ``AtomicShift(requires_grad=False)``: the same table as a FIXED role.

    Only the table is excluded from parameter updates; it carries no geometry
    dependence, so coordinate, cell and condition responses are unchanged.
    """

    shifts: Array

    def __init__(self, shifts: ArrayLike, /, *, dtype: DTypeLike = jnp.float64) -> None:
        self.shifts = jnp.asarray(_shift_table(shifts), dtype=dtype)

    def rebuilt(self) -> AIMNet2FrozenAtomicShifts:
        return AIMNet2FrozenAtomicShifts(_host(self.shifts), dtype=self.shifts.dtype)


class AIMNet2ShortRangeRadius(StrictModule, NonTrainableState):
    """The source ``SRCoulomb.rc`` buffer (angstrom), as executed."""

    radius: Array
    value: float = eqx.field(static=True)

    def __init__(self, radius: ArrayLike, /, *, dtype: DTypeLike = jnp.float64) -> None:
        source = np.asarray(radius)
        if source.dtype.kind != "f" or source.shape != () or not np.isfinite(source):
            raise ValueError("The short-range Coulomb radius must be a finite scalar.")
        if float(source) <= 0.0:
            raise ValueError("The short-range Coulomb radius must be positive.")
        self.radius = jnp.asarray(source, dtype=dtype)
        self.value = float(_host(self.radius))

    def rebuilt(self) -> AIMNet2ShortRangeRadius:
        return AIMNet2ShortRangeRadius(_host(self.radius), dtype=self.radius.dtype)


class AIMNet2Repulsion(StrictModule, NonTrainableState):
    """Source ``core.SRRep``: GFN1 repulsion ``exp(-a_ij d^1.5) z_ij / d``.

    ``parameters[Z] = (sqrt(alpha_Z) bohr^-0.75, Zeff_Z sqrt(Hartree bohr / 2))``
    is the source's frozen ``params`` embedding; pairs use the elementwise
    products ``a_ij = p_i p_j``. Every directed route within ``route_cutoff``
    contributes ``f_c(d) exp(-a_ij d^1.5) z_ij / d`` (eV) to its receiver, with
    the source's ``cutoff_function`` (``"none"``, ``"exp_cutoff"`` or
    ``"cosine_cutoff"``) at the ``rc`` buffer.
    """

    parameters: Array
    radius: Array
    cutoff_function: str = eqx.field(static=True)
    route_cutoff: float = eqx.field(static=True)

    def __init__(
        self,
        parameters: ArrayLike,
        radius: ArrayLike,
        /,
        *,
        cutoff_function: str,
        route_cutoff: float,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        table = np.asarray(parameters)
        rc = np.asarray(radius)
        if (
            table.dtype.kind != "f"
            or table.ndim != 2
            or table.shape[1] != 2
            or not np.all(np.isfinite(table))
        ):
            raise ValueError("SRRep parameters must be a finite (S, 2) table.")
        if rc.shape != () or not np.isfinite(rc) or float(rc) <= 0.0:
            raise ValueError("The SRRep rc buffer must be a finite positive scalar.")
        if cutoff_function not in ("none", "exp_cutoff", "cosine_cutoff"):
            raise ValueError("Unknown SRRep cutoff function.")
        self.parameters = jnp.asarray(table, dtype=dtype)
        self.radius = jnp.asarray(rc, dtype=dtype)
        self.cutoff_function = cutoff_function
        self.route_cutoff = float(route_cutoff)

    def rebuilt(self) -> AIMNet2Repulsion:
        return AIMNet2Repulsion(
            _host(self.parameters),
            _host(self.radius),
            cutoff_function=self.cutoff_function,
            route_cutoff=self.route_cutoff,
            dtype=self.parameters.dtype,
        )

    def pair_energy(
        self, species_i: Array, species_j: Array, distance: Array, /
    ) -> Array:
        size = self.parameters.shape[0]
        products = (
            self.parameters[jnp.clip(species_i, 0, size - 1)]
            * self.parameters[jnp.clip(species_j, 0, size - 1)]
        ).astype(distance.dtype)
        energy = jnp.exp(-products[..., 0] * distance**1.5) * products[..., 1] / distance
        match self.cutoff_function:
            case "exp_cutoff":
                energy = energy * _exp_cutoff(distance, self.radius)
            case "cosine_cutoff":
                energy = energy * source_cosine_cutoff(distance, self.radius)
        within = distance < jnp.asarray(self.route_cutoff, dtype=distance.dtype)
        return jnp.where(within, energy, jnp.zeros((), energy.dtype))


class AIMNet2MassTable(StrictModule, NonTrainableState):
    """The source ``Dipole.mass`` buffer used only by centered moments."""

    masses: Array

    def __init__(self, masses: ArrayLike, /, *, dtype: DTypeLike = jnp.float64) -> None:
        values = np.asarray(masses)
        if (
            values.dtype.kind != "f"
            or values.ndim != 1
            or values.size < AIMNET2_SPECIES_CAPACITY
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("The moment mass table must be finite nonnegative masses.")
        self.masses = jnp.asarray(values, dtype=dtype)

    def rebuilt(self) -> AIMNet2MassTable:
        return AIMNet2MassTable(_host(self.masses), dtype=self.masses.dtype)


StageValue: TypeAlias = (
    AIMNet2MLP
    | AIMNet2AtomicShifts
    | AIMNet2FrozenAtomicShifts
    | AIMNet2ShortRangeRadius
    | AIMNet2Repulsion
    | AIMNet2MassTable
    | TSDispersionReference
    | TSDispersionPotential
    | None
)
_Stage = TypeVar("_Stage")


def _stage_value(
    stage: AIMNet2OutputStage, value: StageValue, kind: type[_Stage], /
) -> _Stage:
    """The value an admitted stage owns, narrowed to its declared component."""
    if not isinstance(value, kind):
        raise TypeError(
            f"Output stage {stage.name!r} carries {type(value).__name__}, not "
            f"{kind.__name__}."
        )
    return value


def _shift_table_of(stage: AIMNet2OutputStage, value: StageValue, /) -> Array:
    if not isinstance(value, (AIMNet2AtomicShifts, AIMNet2FrozenAtomicShifts)):
        raise TypeError(f"Output stage {stage.name!r} carries no atomic shift table.")
    return value.shifts


def _check_stage_value(
    stage: AIMNet2OutputStage,
    value: StageValue,
    architecture: AIMNet2Architecture,
    /,
) -> None:
    kind = stage.kind
    match kind:
        case AIMNet2OutputKind.ENERGY_HEAD | AIMNet2OutputKind.DISPERSION_HEAD:
            if (
                not isinstance(value, AIMNet2MLP)
                or value.in_size != architecture.aim_size
                or value.out_size != stage.output_count
                or value.widths != stage.hidden
                or value.last_linear != stage.last_linear
                or bool(value.biases) != stage.bias
                or value.activation != stage.activation
            ):
                raise ValueError(f"Output head {stage.name!r} differs from its stage.")
        case AIMNet2OutputKind.ATOMIC_SHIFT:
            expected = AIMNet2FrozenAtomicShifts if stage.frozen else AIMNet2AtomicShifts
            if (
                not isinstance(value, (AIMNet2AtomicShifts, AIMNet2FrozenAtomicShifts))
                or type(value) is not expected
                or value.shifts.shape[0] != stage.species_rows
            ):
                raise ValueError(
                    f"Atomic shift {stage.name!r} needs its {stage.species_rows} rows "
                    f"as {expected.__name__}."
                )
            if architecture.implemented_species[-1] >= stage.species_rows:
                raise ValueError("Implemented species exceed the atomic shift table.")
        case AIMNet2OutputKind.SHORT_RANGE_COULOMB:
            if not isinstance(value, AIMNet2ShortRangeRadius):
                raise TypeError("Short-range Coulomb needs its source radius.")
            if value.value != stage.pair_cutoff:
                raise ValueError("The short-range Coulomb radius differs from its stage.")
            if value.value > architecture.aev_cutoff:
                raise ValueError(
                    "The short-range Coulomb radius exceeds the AEV radius; the source "
                    "sums it over the model neighbor list only."
                )
        case AIMNet2OutputKind.SHORT_RANGE_REPULSION:
            if (
                not isinstance(value, AIMNet2Repulsion)
                or value.cutoff_function != stage.cutoff_function
                or value.route_cutoff != stage.pair_cutoff
            ):
                raise ValueError(f"SRRep {stage.name!r} differs from its stage.")
            if architecture.implemented_species[-1] >= value.parameters.shape[0]:
                raise ValueError("Implemented species exceed the SRRep parameter table.")
        case AIMNet2OutputKind.DIPOLE | AIMNet2OutputKind.QUADRUPOLE:
            if stage.center_coordinates != isinstance(value, AIMNet2MassTable) or (
                not stage.center_coordinates and value is not None
            ):
                raise ValueError("Centered moments, and only they, need the mass table.")
        case AIMNet2OutputKind.DISPERSION_PARAMETERS:
            if not isinstance(value, TSDispersionReference):
                raise TypeError("DispParam needs a TSDispersionReference.")
            missing = set(architecture.implemented_species) - set(
                value.implemented_species
            )
            if missing:
                raise ValueError(
                    f"Dispersion references do not cover species {sorted(missing)}."
                )
        case AIMNet2OutputKind.TS_DISPERSION:
            if not isinstance(value, TSDispersionPotential):
                raise TypeError("D3TS needs a TSDispersionPotential.")
            if (value.s6, value.s8, value.a1, value.a2) != stage.dispersion or (
                value.pair_cutoff != stage.pair_cutoff
            ):
                raise ValueError("The TS dispersion term differs from its stage.")
            value.require_coverage(architecture.implemented_species)
        case AIMNet2OutputKind.ATOMIC_SUM:
            if value is not None:
                raise ValueError("Atomic sums carry no values.")


def _default_stage_value(
    stage: AIMNet2OutputStage,
    architecture: AIMNet2Architecture,
    key: PRNGKey,
    dtype: DTypeLike,
    reduction: DTypeLike,
    dispersion_reference: TSDispersionReference | None,
    /,
) -> StageValue:
    match stage.kind:
        case AIMNet2OutputKind.ENERGY_HEAD | AIMNet2OutputKind.DISPERSION_HEAD:
            return _default_mlp(
                (architecture.aim_size, *stage.hidden, stage.output_count),
                stage.last_linear,
                key,
                dtype,
                bias=stage.bias,
                activation=stage.activation,
            )
        case AIMNet2OutputKind.ATOMIC_SHIFT:
            # nn.Embedding(num_types, 1, padding_idx=0): N(0, 1), zero padding row.
            values = jr.normal(key, (stage.species_rows,), dtype=jnp.float64)
            values = values.at[0].set(0.0)
            if stage.frozen:
                return AIMNet2FrozenAtomicShifts(values, dtype=reduction)
            return AIMNet2AtomicShifts(values, dtype=reduction)
        case AIMNet2OutputKind.SHORT_RANGE_COULOMB:
            return AIMNet2ShortRangeRadius(np.float64(stage.pair_cutoff), dtype=dtype)
        case AIMNet2OutputKind.SHORT_RANGE_REPULSION:
            raise ValueError(
                "SRRep GFN1 parameters and rc come from the source state; supply "
                "stage_values explicitly."
            )
        case AIMNet2OutputKind.DIPOLE | AIMNet2OutputKind.QUADRUPOLE:
            if stage.center_coordinates:
                raise ValueError("Centered moments need the source mass table.")
            return None
        case AIMNet2OutputKind.DISPERSION_PARAMETERS:
            if dispersion_reference is None:
                raise ValueError("DispParam needs an explicit reference table.")
            return dispersion_reference
        case AIMNet2OutputKind.TS_DISPERSION:
            raise ValueError(
                "D3TS requires a caller-supplied r4r2 table; supply its "
                "TSDispersionPotential in stage_values explicitly."
            )
        case AIMNet2OutputKind.ATOMIC_SUM:
            return None


def _exp_cutoff(distance: Array, radius: Array, /) -> Array:
    """``ops.exp_cutoff`` with torch ``clamp`` derivative semantics."""
    ratio = distance / radius.astype(distance.dtype)
    ceiling = jnp.asarray(_EXP_CUTOFF_CEILING, dtype=distance.dtype)
    inside = (ratio >= 0.0) & (ratio <= ceiling)
    clamped = jnp.where(inside, ratio, jnp.where(ratio < 0.0, 0.0, ceiling))
    return jnp.exp(-1.0 / (1.0 - clamped**2)) / _EXP_CUTOFF_NORMALIZATION


class _PairTerms(StrictModule):
    """Embedded pair stages of one evaluation (static envelope, array leaves)."""

    coulomb: AIMNet2ShortRangeRadius | None
    repulsion: AIMNet2Repulsion | None
    dispersion: TSDispersionPotential | None
    envelope: str | None = eqx.field(static=True)

    def __init__(
        self,
        coulomb: AIMNet2ShortRangeRadius | None,
        repulsion: AIMNet2Repulsion | None,
        dispersion: TSDispersionPotential | None,
        envelope: str | None,
        /,
    ) -> None:
        self.coulomb = coulomb
        self.repulsion = repulsion
        self.dispersion = dispersion
        self.envelope = envelope

    @property
    def radius(self) -> float:
        radii = [0.0]
        if self.coulomb is not None:
            radii.append(self.coulomb.value)
        if self.repulsion is not None:
            radii.append(self.repulsion.route_cutoff)
        if self.dispersion is not None:
            radii.append(self.dispersion.pair_cutoff)
        return max(radii)


def _pair_edge(
    parameters: _PairTerms,
    sender: tuple[Array, Array, Array],
    receiver: tuple[Array, Array, Array],
    edge: Array,
    /,
) -> dict[str, Array]:
    charge_j, coefficients_j, species_j = sender
    charge_i, coefficients_i, species_i = receiver
    distance = edge
    messages = {}
    coulomb = parameters.coulomb
    if coulomb is not None:
        if parameters.envelope == "exp":
            weight = _exp_cutoff(distance, coulomb.radius)
        else:
            weight = source_cosine_cutoff(distance, coulomb.radius)
        messages["coulomb"] = (
            AIMNET2_COULOMB_PAIR_FACTOR * weight * charge_i * charge_j / distance
        )
    if parameters.repulsion is not None:
        messages["repulsion"] = parameters.repulsion.pair_energy(
            species_i, species_j, distance
        )
    dispersion = parameters.dispersion
    if dispersion is not None:
        energy = dispersion.pair_energy(
            (coefficients_i, species_i), (coefficients_j, species_j), distance
        )
        within = distance < jnp.asarray(dispersion.pair_cutoff, dtype=distance.dtype)
        messages["dispersion"] = jnp.where(within, energy, jnp.zeros_like(energy))
    return messages


def _pair_epilogue(
    parameters: Any, receiver: Any, aggregate: dict[str, Array], /
) -> dict[str, Array]:
    del parameters, receiver
    return aggregate


def _case_any(flags: Array, cases: Array, count: int, /) -> Array:
    return jnp.zeros((count,), dtype=jnp.int32).at[cases].add(flags.astype(jnp.int32)) > 0


def _moment_contributions(
    kind: AIMNet2OutputKind, q: Array, positions: Array, /
) -> Array:
    """Per-atom charge-moment terms ``q x`` or ``q [x^2, x * roll(x)]``."""
    if kind is AIMNet2OutputKind.DIPOLE:
        return q * positions
    products = jnp.concatenate(
        (positions**2, positions * jnp.roll(positions, -1, axis=-1)), axis=-1
    )
    return q * products


def _finish_moment(kind: AIMNet2OutputKind, raw: Array, /) -> Array:
    """Dipole as summed, or the source traceless quadrupole of summed terms."""
    if kind is AIMNet2OutputKind.DIPOLE:
        return raw
    diagonal = raw[..., :3] - jnp.mean(raw[..., :3], axis=-1, keepdims=True)
    return jnp.concatenate((diagonal, raw[..., 3:]), axis=-1)


class AIMNet2LayerContext(NamedTuple):
    """Owner layer context: replicated ``(cases, channels)`` charge targets,
    owned-row case indices, embedding-safe atomic numbers and admission."""

    targets: Array
    atom_cases: Array
    numbers: Array
    admitted: Array


class AIMNet2NodeFeatures(NamedTuple):
    """Atom features ``a`` and equilibrated channel charges ``q`` of owned rows."""

    atoms: Array
    charges: Array


class AIMNet2PassOutput(NamedTuple):
    """A non-final pass awaiting its equilibration stage: ``a`` (before its
    delta), unconstrained charges, flexibilities ``f_raw^2`` and delta."""

    atoms: Array
    unconstrained: Array
    flexibility: Array
    delta: Array


class AIMNet2ReadoutFeatures(NamedTuple):
    """Final-pass AIM vectors and the last equilibrated channel charges."""

    aim: Array
    charges: Array


class AIMNet2ChargePartials(NamedTuple):
    """Per-case, per-channel sums of unconstrained charges and flexibilities."""

    charge: Array
    flexibility: Array


class AIMNet2LayerUpdate(NamedTuple):
    """One layered AIMNet2 interaction: states, layer energies, evidence."""

    features: AIMNet2PassOutput | AIMNet2ReadoutFeatures
    energy: Array
    evidence: StreamedRelationEvidence


def _energy_readout_dependencies(
    stages: tuple[AIMNet2OutputStage, ...], /
) -> tuple[int, ...] | None:
    """Trace the executed scalar heads and final TS coefficient assignment."""
    values: dict[str, tuple[int, ...]] = {}
    coefficients: tuple[int, ...] | None = None
    energy: list[int] = []
    has_ts = False
    for index, stage in enumerate(stages):
        match stage.kind:
            case AIMNet2OutputKind.ENERGY_HEAD:
                energy.append(index)
            case AIMNet2OutputKind.DISPERSION_HEAD:
                values[stage.key_out] = (index,)
            case AIMNet2OutputKind.DISPERSION_PARAMETERS:
                coefficients = (*values[stage.key_in], index)
                values[stage.key_out] = coefficients
            case AIMNet2OutputKind.TS_DISPERSION:
                has_ts = True
                coefficients = values[stage.key_in]
            case (
                AIMNet2OutputKind.ATOMIC_SHIFT
                | AIMNet2OutputKind.ATOMIC_SUM
                | AIMNet2OutputKind.SHORT_RANGE_COULOMB
                | AIMNet2OutputKind.SHORT_RANGE_REPULSION
                | AIMNet2OutputKind.DIPOLE
                | AIMNet2OutputKind.QUADRUPOLE
            ):
                pass
            case _ as unreachable:
                assert_never(unreachable)
    if has_ts:
        if (
            coefficients is None
            or stages[coefficients[-1]].kind
            is not AIMNet2OutputKind.DISPERSION_PARAMETERS
        ):
            # A raw TS input needs additional nonzero-polarizability support;
            # a produced key alone is not evidence of positive DispParam data.
            return None
        energy.extend(coefficients)
    return tuple(energy)


def _cutoff_conditions(
    function: str, radius: float, route_radius: float, /
) -> tuple[int, tuple[str, ...]]:
    """Continuity of the actual clamped envelope followed by its route mask."""
    joins: list[float] = []
    match function:
        case "none":
            continuity = -1
            joins.append(route_radius)
        case "exp_cutoff":
            # Underflow of the nonzero real plateau is not a smoothness proof.
            continuity = -1
            ceiling = radius * _EXP_CUTOFF_CEILING
            if ceiling <= route_radius:
                joins.append(ceiling)
            joins.append(route_radius)
        case "cosine_cutoff":
            floor = SOURCE_CUTOFF_DISTANCE_FLOOR
            hard_route = route_radius < max(radius, floor)
            continuity = -1 if hard_route or radius < floor else 0
            if floor <= route_radius:
                joins.append(floor)
            if floor <= radius <= route_radius:
                joins.append(radius)
            if hard_route:
                joins.append(route_radius)
        case _:
            raise ValueError(f"Unknown AIMNet2 cutoff function {function!r}.")
    return continuity, tuple(atomistic_edge_distance_locus(value) for value in joins)


def _activation_condition(
    stage: AIMNet2OutputStage, mlp: AIMNet2MLP, /
) -> tuple[int, str] | None:
    """Only an activation actually executed by the scalar dependency can add a locus."""
    if len(mlp.weights) == 1 and mlp.last_linear:
        return None
    prefix = f"atomistic-nonsmooth-locus:aimnet2-{stage.name}"
    match mlp.activation[0]:
        case "relu":
            return 0, f"{prefix}-relu-zero"
        case "softplus":
            return -1, f"{prefix}-softplus-threshold"
        case "gelu" | "gelu-tanh" | "silu" | "tanh" | "sigmoid":
            return None
        case other:
            raise ValueError(f"Unknown AIMNet2 activation {other!r}.")


class AIMNet2Potential(AbstractAtomisticPotential):
    """Native AIMNet2 source-model energy with typed charge/property outputs.

    Construct with ``key`` for the source family's random initialization, or
    with every numerical component (as the source reconstruction does); both
    run the same validators. Requires atomic numbers, angstrom/eV and
    ``conditioning`` matching ``configuration.conditioning_spec()``. The
    graph may be finite or periodic (explicit image routes, any cell); charge
    moments are unavailable for periodic cases because their value depends on
    the coordinate branch. The energy is the source model output only (see the
    module docstring); its strain derivative is exact for fixed image routes.
    """

    embedding: Array
    basis: AIMNet2RadialBasis
    atom_conv: AIMNet2ConvSV
    charge_conv: AIMNet2ConvSV
    passes: tuple[AIMNet2MLP, ...]
    stage_values: tuple[StageValue, ...]
    configuration: AIMNet2Architecture
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        architecture: AIMNet2Architecture,
        /,
        *,
        precision: AtomisticPrecisionPolicy | None = None,
        embedding: ArrayLike | None = None,
        basis: AIMNet2RadialBasis | None = None,
        atom_conv: AIMNet2ConvSV | None = None,
        charge_conv: AIMNet2ConvSV | None = None,
        passes: Sequence[AIMNet2MLP] | None = None,
        stage_values: Sequence[StageValue] | None = None,
        dispersion_reference: TSDispersionReference | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        if scale.scale_id != AtomisticScaleContract(ANGSTROM, ELECTRONVOLT).scale_id:
            raise ValueError("AIMNet2 source constants are angstrom and electronvolt.")
        if not isinstance(architecture, AIMNet2Architecture):
            raise TypeError("architecture must be an AIMNet2Architecture.")
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
        dtype = jnp.dtype(precision_.compute_dtype)
        reduction = jnp.dtype(precision_.reduction_dtype)
        if (
            embedding is not None
            and basis is not None
            and atom_conv is not None
            and charge_conv is not None
            and passes is not None
            and stage_values is not None
        ):
            if dispersion_reference is not None:
                raise ValueError(
                    "dispersion_reference is for default construction; supply it as "
                    "the DispParam stage value."
                )
            embedding_ = jnp.asarray(embedding, dtype=dtype)
            basis_ = basis
            atom_conv_ = atom_conv
            charge_conv_ = charge_conv
            passes_ = tuple(passes)
            values_ = tuple(stage_values)
        elif any(
            value is not None
            for value in (embedding, basis, atom_conv, charge_conv, passes, stage_values)
        ):
            raise ValueError(
                "Supply every numerical component (embedding, basis, convolutions, "
                "passes and stage values) or none."
            )
        else:
            keys = jr.split(key, 3 + architecture.pass_count + len(architecture.outputs))
            embedding_ = self._default_embedding(architecture, keys[0]).astype(dtype)
            cutoff = architecture.aev_cutoff
            basis_ = AIMNet2RadialBasis(
                source_default_shifts(
                    cutoff, architecture.shift_count, SOURCE_MINIMUM_SHIFT
                ),
                np.float32(
                    source_default_eta(
                        cutoff, architecture.shift_count, SOURCE_MINIMUM_SHIFT
                    )
                ),
                np.float32(cutoff),
                dtype=dtype,
            )
            seeds = np.asarray(jr.randint(keys[1], (2,), 0, 2**31 - 1))
            atom_conv_ = AIMNet2ConvSV(
                source_ahg_initialization(
                    architecture.feature_count,
                    architecture.shift_count,
                    architecture.combination_count,
                    seed=int(seeds[0]),
                ),
                d2_features=architecture.d2_features,
                dtype=dtype,
            )
            charge_conv_ = AIMNet2ConvSV(
                source_ahg_initialization(
                    architecture.charge_channels,
                    architecture.shift_count,
                    architecture.combination_count,
                    seed=int(seeds[1]),
                ),
                d2_features=False,
                dtype=dtype,
            )
            passes_ = tuple(
                _default_mlp(
                    (inputs, *widths, outputs),
                    last_linear,
                    keys[3 + index],
                    dtype,
                )
                for index, (widths, (inputs, outputs, last_linear)) in enumerate(
                    zip(
                        architecture.pass_hidden,
                        architecture.pass_shapes(),
                        strict=True,
                    )
                )
            )
            offset = 3 + architecture.pass_count
            values_ = tuple(
                _default_stage_value(
                    stage,
                    architecture,
                    keys[offset + index],
                    dtype,
                    reduction,
                    dispersion_reference,
                )
                for index, stage in enumerate(architecture.outputs)
            )
        _check_components(
            architecture, embedding_, basis_, atom_conv_, charge_conv_, passes_, values_
        )
        self.embedding = embedding_
        self.basis = basis_
        self.atom_conv = atom_conv_
        self.charge_conv = charge_conv_
        self.passes = passes_
        self.stage_values = values_
        self.configuration = architecture
        self.scale = scale
        self.precision = precision_
        self.architecture_id = _potential_identity(
            architecture, scale, precision_, basis_, values_
        )
        self.method_id = _METHOD_ID

    @property
    def conditioning_spec(self) -> AtomisticConditioningSpec:
        """Typed conditioning this potential consumes, owned by its architecture."""
        return self.configuration.conditioning_spec()

    def validate(self) -> None:
        """Rerun every owning check on this (possibly restored) potential.

        This is a host boundary over concrete arrays. The architecture and every
        stage declaration must equal their constructor rebuild (cached stage and
        architecture identities included); scale and precision must be the exact
        source contracts; every component is rebuilt by its own constructor from
        its stored leaves (finiteness, shapes, coverage, padding dummy rows,
        identities) and must equal what is stored, with leaves in the declared
        compute or reduction dtype; the component/architecture consistency is
        rechecked and the recorded potential identity recomputed. Nothing is
        reinitialized; any mismatch refuses with ``ValueError``.
        """
        architecture = self.configuration
        if not isinstance(architecture, AIMNet2Architecture):
            raise TypeError("configuration must be an AIMNet2Architecture.")
        require_identical(
            architecture, architecture.rebuilt(), "AIMNet2 architecture", ValueError
        )
        expected_scale = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
        if (
            not isinstance(self.scale, AtomisticScaleContract)
            or self.scale.scale_id != expected_scale.scale_id
            or AtomisticScaleContract(
                self.scale.length_unit, self.scale.energy_unit
            ).scale_id
            != self.scale.scale_id
        ):
            raise ValueError("AIMNet2 requires the exact angstrom/electronvolt scale.")
        precision = self.precision
        if (
            not isinstance(precision, AtomisticPrecisionPolicy)
            or AtomisticPrecisionPolicy(
                coordinate_dtype=precision.coordinate_dtype,
                compute_dtype=precision.compute_dtype,
                reduction_dtype=precision.reduction_dtype,
                output_dtype=precision.output_dtype,
            ).policy_id
            != precision.policy_id
        ):
            raise ValueError("The AIMNet2 precision policy does not match its identity.")
        compute = np.dtype(precision.compute_dtype)
        reduction = np.dtype(precision.reduction_dtype)
        embedding = _host(self.embedding)
        if embedding.dtype != compute or not np.all(np.isfinite(embedding)):
            raise ValueError("The AIMNet2 embedding must be finite in the compute dtype.")
        components: tuple[tuple[str, Any, np.dtype[Any]], ...] = (
            ("AEV basis", self.basis, compute),
            ("conv_a", self.atom_conv, compute),
            ("conv_q", self.charge_conv, compute),
            *(
                (f"charge pass {index}", mlp, compute)
                for index, mlp in enumerate(self.passes)
            ),
        )
        for name, component, expected in components:
            _require_component(component, name, expected)
        if len(self.stage_values) != len(architecture.outputs):
            raise ValueError("Stage values must align with the output stages.")
        for stage, value in zip(architecture.outputs, self.stage_values, strict=True):
            if value is None:
                continue
            expected = (
                reduction
                if isinstance(value, (AIMNet2AtomicShifts, AIMNet2FrozenAtomicShifts))
                else compute
            )
            _require_component(value, f"output stage {stage.name!r}", expected)
        _check_components(
            architecture,
            self.embedding,
            self.basis,
            self.atom_conv,
            self.charge_conv,
            self.passes,
            self.stage_values,
        )
        identity = _potential_identity(
            architecture, self.scale, precision, self.basis, self.stage_values
        )
        if identity != self.architecture_id or self.method_id != _METHOD_ID:
            raise ValueError(
                "The AIMNet2 potential does not match its recorded identities."
            )

    @staticmethod
    def _default_embedding(architecture: AIMNet2Architecture, key: PRNGKey, /) -> Array:
        """``orthogonal_`` rows 1: (row 0 zero), expanded over shifts for d2."""
        rows = AIMNET2_SPECIES_CAPACITY - 1
        features = architecture.feature_count
        gaussian = np.asarray(
            jr.normal(key, (max(rows, features), min(rows, features)), dtype=jnp.float64)
        )
        q, r = np.linalg.qr(gaussian)
        q = q * np.sign(np.diag(r))[None, :]
        orthogonal = q if rows >= features else q.T
        table = np.zeros((AIMNET2_SPECIES_CAPACITY, features), dtype=np.float64)
        table[1:] = orthogonal[:rows, :features]
        if architecture.d2_features:
            table = np.repeat(table[:, :, None], architecture.shift_count, axis=2)
            table = table.reshape((AIMNET2_SPECIES_CAPACITY, -1))
        return jnp.asarray(table, dtype=jnp.float64)

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """Branchwise scalar response with fixed species, topology, images and conditions.

        The source cosine floor is C0; hard truncations and thresholded
        Softplus can be discontinuous. Squared ConvSV combinations and epsilon
        charge normalization introduce no vector-zero or flexibility singularity.
        Internal activation/dispersion joins remain explicit unresolved evidence.
        A TS input not proven positive by its executed DispParam dataflow is
        not admitted by this contract.
        """
        stages = self.configuration.outputs
        dependencies = _energy_readout_dependencies(stages)
        if dependencies is None:
            return None
        continuity, aev_joins = _cutoff_conditions(
            "cosine_cutoff", self.basis.radius, self.configuration.aev_cutoff
        )
        conditions = list(aev_joins)
        parameter_conditions: list[str] = []
        coordinate_dtype = np.dtype(self.precision.coordinate_dtype)
        conditions.append(
            atomistic_edge_distance_locus(math.sqrt(np.finfo(coordinate_dtype).tiny))
        )
        if coordinate_dtype != np.dtype(self.precision.compute_dtype):
            conditions.append(
                "atomistic-nonsmooth-locus:aimnet2-compute-distance-join-rounding"
            )
        for index in dependencies:
            stage = stages[index]
            value = self.stage_values[index]
            if stage.kind in (
                AIMNet2OutputKind.ENERGY_HEAD,
                AIMNet2OutputKind.DISPERSION_HEAD,
            ):
                activation = _activation_condition(
                    stage, _stage_value(stage, value, AIMNet2MLP)
                )
                if activation is not None:
                    order, condition = activation
                    continuity = min(continuity, order)
                    parameter_conditions.append(condition)
            elif stage.kind is AIMNet2OutputKind.DISPERSION_PARAMETERS:
                parameter_conditions.append(
                    f"atomistic-nonsmooth-locus:aimnet2-{stage.name}-dispersion-log-bound"
                )
        terms = self._pair_terms()
        for stage, value in zip(stages, self.stage_values, strict=True):
            match stage.kind:
                case AIMNet2OutputKind.SHORT_RANGE_COULOMB:
                    coulomb = _stage_value(stage, value, AIMNet2ShortRangeRadius)
                    function = (
                        "exp_cutoff" if stage.envelope == "exp" else "cosine_cutoff"
                    )
                    order, joins = _cutoff_conditions(
                        function, coulomb.value, terms.radius
                    )
                case AIMNet2OutputKind.SHORT_RANGE_REPULSION:
                    repulsion = _stage_value(stage, value, AIMNet2Repulsion)
                    # Its envelope radius is a fixed numerical buffer, not
                    # trace-safe static metadata. Keep the internal join
                    # unresolved rather than substitute its distinct route radius.
                    order = -1
                    joins = (atomistic_edge_distance_locus(repulsion.route_cutoff),)
                    if repulsion.cutoff_function != "none":
                        joins += (
                            f"atomistic-nonsmooth-locus:aimnet2-{stage.name}-repulsion-envelope",
                        )
                case AIMNet2OutputKind.TS_DISPERSION:
                    dispersion = _stage_value(stage, value, TSDispersionPotential)
                    order, joins = _cutoff_conditions(
                        "none", dispersion.pair_cutoff, dispersion.pair_cutoff
                    )
                    parameter_conditions.append(
                        f"atomistic-nonsmooth-locus:aimnet2-{stage.name}-ts-combination-floor"
                    )
                case _:
                    continue
            continuity = min(continuity, order)
            conditions.extend(joins)
        conditions.extend(parameter_conditions)
        return DerivativeContract(
            (
                SurfaceDerivative(
                    DerivativeSurface.INPUT, GradientLevel.ALMOST_EVERYWHERE
                ),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER,
                    GradientLevel.ALMOST_EVERYWHERE
                    if parameter_conditions
                    else GradientLevel.SMOOTH,
                    conditions=tuple(sorted(set(parameter_conditions))),
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.piecewise_smooth(
                continuity=continuity,
                conditions=tuple(sorted(set(conditions))),
                support=ATOMISTIC_NONCOINCIDENT_SUPPORT,
            ),
        )

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            species_kind=AtomisticSpeciesKind.ATOMIC_NUMBER,
        )

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        architecture = self.configuration
        supported = [
            AtomisticObservable.ENERGY,
            AtomisticObservable.ATOM_ENERGY,
            AtomisticObservable.CHARGES,
            AtomisticObservable.CHARGE_RESIDUAL,
            AtomisticObservable.NODE_EMBEDDINGS,
        ]
        if architecture.charge_channels == 2:
            supported.append(AtomisticObservable.SPIN_CHARGES)
        if architecture.stage(AIMNet2OutputKind.DIPOLE) is not None:
            supported.append(AtomisticObservable.DIPOLE)
        if architecture.stage(AIMNet2OutputKind.QUADRUPOLE) is not None:
            supported.append(AtomisticObservable.QUADRUPOLE)
        if architecture.stage(AIMNet2OutputKind.DISPERSION_PARAMETERS) is not None:
            supported.extend(
                (AtomisticObservable.ATOMIC_C6, AtomisticObservable.ATOMIC_POLARIZABILITY)
            )
        return tuple(supported)

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError(
                "Potential and structure must share one exact scale contract."
            )
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the AIMNet2 precision contract."
            )

    def _targets(self, conditioning: AtomisticConditioning, dtype: Any, /) -> Array:
        charge = jnp.asarray(conditioning.value("charge")).astype(dtype)
        if self.configuration.charge_channels == 1:
            return charge[:, None]
        multiplicity = jnp.asarray(conditioning.value("multiplicity")).astype(dtype)
        half_spin = 0.5 * (multiplicity - 1.0)
        half_charge = 0.5 * charge
        return jnp.stack((half_charge + half_spin, half_charge - half_spin), axis=-1)

    def _require_conditioning(
        self, conditioning: AtomisticConditioning | None, /
    ) -> AtomisticConditioning:
        if conditioning is None:
            raise ValueError("AIMNet2 requires total-charge conditioning.")
        if conditioning.spec.spec_id != self.conditioning_spec.spec_id:
            raise ValueError(
                "Conditioning does not match the AIMNet2 conditioning specification."
            )
        return conditioning

    def _numbers(self, numbers: Array, mask: Array, /) -> tuple[Array, Array]:
        """Embedding-safe atomic numbers and active atoms outside the model."""
        in_table = (numbers >= 0) & (numbers < AIMNET2_SPECIES_CAPACITY)
        safe_numbers = jnp.where(mask & in_table, numbers, 0)
        implemented_table = np.zeros((AIMNET2_SPECIES_CAPACITY,), dtype=bool)
        implemented_table[list(self.configuration.implemented_species)] = True
        unsupported = mask & ~(in_table & jnp.asarray(implemented_table)[safe_numbers])
        return safe_numbers, unsupported

    def _initial_features(self, numbers: Array, weights: Array, /) -> Array:
        dtype = jnp.dtype(self.precision.compute_dtype)
        return self.embedding[numbers].astype(dtype) * weights[:, None]

    def _atom_view(self, features: Array, /) -> Array:
        architecture = self.configuration
        if not architecture.d2_features:
            return features
        return features.reshape(
            (features.shape[0], architecture.feature_count, architecture.shift_count)
        )

    def _environment(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        atom_view: Array,
        charges: Array,
        weights: Array,
        edge: tuple[Array, Array],
        aev_active: Array,
        /,
    ) -> StreamedRelationResult:
        """``conv_a`` (and, after pass 0, ``conv_q``) of pass ``index``."""
        return aimnet2_environment(
            relation,
            self.basis,
            self.atom_conv,
            None if index == 0 else self.charge_conv,
            atom_view,
            charges,
            weights,
            edge,
            aev_active,
        )

    def _pass_output(
        self,
        index: int,
        features: Array,
        charges: Array,
        environment: tuple[Array, ...],
        weights: Array,
        /,
    ) -> Array:
        """MLP output of pass ``index`` from its receiver environment."""
        if index == 0:
            (atom_environment,) = environment
            stacked = jnp.concatenate((features, atom_environment), axis=-1)
        else:
            atom_environment, charge_environment = environment
            stacked = jnp.concatenate(
                (features, atom_environment, charges, charge_environment), axis=-1
            )
        return self.passes[index](stacked) * weights[:, None]

    def _split_pass(
        self, index: int, output: Array, charges: Array, /
    ) -> tuple[Array, Array, Array]:
        """Unconstrained charges, flexibility ``f_raw^2`` and feature delta."""
        channels = self.configuration.charge_channels
        raw_charge = output[:, :channels]
        flexibility = output[:, channels : 2 * channels] ** 2
        delta = output[:, 2 * channels :]
        unconstrained = raw_charge if index == 0 else charges + raw_charge
        return unconstrained, flexibility, delta

    def _pair_terms(self) -> _PairTerms:
        """Embedded pair stages (static radii) in source declaration."""
        coulomb: AIMNet2ShortRangeRadius | None = None
        repulsion: AIMNet2Repulsion | None = None
        dispersion: TSDispersionPotential | None = None
        envelope: str | None = None
        for stage, value in zip(
            self.configuration.outputs, self.stage_values, strict=True
        ):
            match stage.kind:
                case AIMNet2OutputKind.SHORT_RANGE_COULOMB:
                    coulomb = _stage_value(stage, value, AIMNet2ShortRangeRadius)
                    envelope = stage.envelope
                case AIMNet2OutputKind.SHORT_RANGE_REPULSION:
                    repulsion = _stage_value(stage, value, AIMNet2Repulsion)
                case AIMNet2OutputKind.TS_DISPERSION:
                    dispersion = _stage_value(stage, value, TSDispersionPotential)
        return _PairTerms(coulomb, repulsion, dispersion, envelope)

    def _readout_stages(
        self, aim: Array, numbers: Array, mask: Array, weights: Array, /
    ) -> tuple[Array, Array | None]:
        """Energy head plus atomic shifts (in stage order) and TS coefficients."""
        reduction = jnp.dtype(self.precision.reduction_dtype)
        atom_energy = jnp.zeros((aim.shape[0],), dtype=reduction)
        raw_values: dict[str, Array] = {}
        coefficients = None
        for stage, value in zip(
            self.configuration.outputs, self.stage_values, strict=True
        ):
            match stage.kind:
                case AIMNet2OutputKind.ENERGY_HEAD:
                    head = _stage_value(stage, value, AIMNet2MLP)(aim)[:, 0] * weights
                    atom_energy = atom_energy + head.astype(reduction)
                case AIMNet2OutputKind.DISPERSION_HEAD:
                    raw_values[stage.key_out] = (
                        _stage_value(stage, value, AIMNet2MLP)(aim) * weights[:, None]
                    )
                case AIMNet2OutputKind.ATOMIC_SHIFT:
                    shifts = _shift_table_of(stage, value)[
                        jnp.where(numbers < stage.species_rows, numbers, 0)
                    ]
                    atom_energy = atom_energy + shifts.astype(reduction) * mask
                case AIMNet2OutputKind.DISPERSION_PARAMETERS:
                    raw = raw_values[stage.key_in]
                    coefficients = _stage_value(
                        stage, value, TSDispersionReference
                    ).coefficients(raw, numbers)
                    raw_values[stage.key_out] = coefficients
                case AIMNet2OutputKind.TS_DISPERSION:
                    coefficients = raw_values[stage.key_in]
        return atom_energy, coefficients

    def _pair_data(
        self, charges: Array, coefficients: Array | None, numbers: Array, /
    ) -> tuple[Array, Array, Array]:
        """Per-atom ``(total charge, TS coefficients, atomic number)`` pair rows."""
        dtype = charges.dtype
        if coefficients is None:
            coefficients = jnp.zeros((charges.shape[0], 2), dtype=dtype)
        return charges, coefficients.astype(dtype), numbers

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        inputs.require_compatible(self.scale, AtomisticSpeciesKind.ATOMIC_NUMBER)
        if request is not None:
            request.require_supported(self.supported_observables, "AIMNet2Potential")
        conditioning = self._require_conditioning(inputs.conditioning)
        architecture = self.configuration
        graph = inputs.graph
        if graph.cutoff < architecture.cutoff:
            raise ValueError(
                f"The graph radius {graph.cutoff} does not cover the AIMNet2 radius "
                f"{architecture.cutoff}."
            )
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("AIMNet2 requires an explicit masked edge relation.")
        cases, atoms = inputs.case_count, inputs.atom_capacity
        count = cases * atoms
        channels = architecture.charge_channels
        dtype = jnp.dtype(self.precision.compute_dtype)
        reduction = jnp.dtype(self.precision.reduction_dtype)
        mask = inputs.atom_mask
        safe_numbers, unsupported = self._numbers(inputs.species_ids, mask)
        species_ok = ~jnp.any(unsupported.reshape((cases, atoms)), axis=1)
        distance = jnp.asarray(ir.edges["distance"])[:, 0]
        # The graph displacement is receiver minus sender; the source unit
        # vector points from the center (receiver) to its neighbor.
        unit = -jnp.asarray(ir.edges["direction"])
        edge_cases = jnp.asarray(ir.edges["case_index"])
        edge_mask = ir.edge_mask
        coincident = _case_any(edge_mask & ~(distance > 0.0), edge_cases, cases)
        compute_distance = jnp.where(
            edge_mask & (distance > 0.0), distance, jnp.ones((), distance.dtype)
        ).astype(dtype)
        aev_active = edge_mask & (distance < architecture.aev_cutoff)
        weights = mask.astype(dtype)
        relation = graph.topology.streamed
        edge = (compute_distance, unit.astype(dtype))

        targets = self._targets(conditioning, dtype)
        features = self._initial_features(safe_numbers, weights)
        charges = jnp.zeros((count, channels), dtype=dtype)
        aim = None
        last = architecture.pass_count - 1
        for index in range(architecture.pass_count):
            environment = self._environment(
                index,
                relation,
                self._atom_view(features),
                charges,
                weights,
                edge,
                aev_active,
            ).receiver_outputs
            output = self._pass_output(index, features, charges, environment, weights)
            if index == last:
                aim = output
                break
            unconstrained, flexibility, delta = self._split_pass(index, output, charges)
            charges = _equilibrate(targets, unconstrained, flexibility, cases, atoms)
            features = features + delta
        if aim is None:
            raise RuntimeError("AIMNet2 produced no AIM vector.")
        achieved = jnp.sum(charges.reshape((cases, atoms, channels)), axis=1)
        residual = targets - achieved
        total_charge = jnp.sum(charges, axis=-1)

        atom_energy, coefficients = self._readout_stages(aim, safe_numbers, mask, weights)
        moments: dict[AtomisticObservable, AtomisticObservableValue] = {}
        for stage, value in zip(architecture.outputs, self.stage_values, strict=True):
            if stage.kind in (AIMNet2OutputKind.DIPOLE, AIMNet2OutputKind.QUADRUPOLE):
                moments[_moment_observable(stage.kind)] = self._moment(
                    stage, value, inputs, total_charge, safe_numbers
                )
        terms = self._pair_terms()
        if terms.radius > 0.0:
            data = self._pair_data(total_charge, coefficients, safe_numbers)
            pair, _ = self._pair_energies(
                relation,
                terms,
                edge_mask & (distance > 0.0) & (distance < terms.radius),
                compute_distance,
                data,
                data,
                weights,
            )
            atom_energy = atom_energy + pair.astype(reduction)
        atom_energy = atom_energy * mask
        accumulation = relation.plan.accumulation
        energy = case_energy_totals(atom_energy, inputs.atom_cases, cases, accumulation)
        successful = inputs.successful & species_ok & ~coincident
        additional = self._additional(
            request,
            inputs,
            successful,
            charges,
            total_charge,
            residual,
            aim,
            coefficients,
            moments,
        )
        output_dtype = jnp.dtype(self.precision.output_dtype)
        return AtomisticModelOutputs(
            energy.astype(output_dtype),
            atom_energy.reshape((cases, atoms)).astype(output_dtype),
            successful,
            additional=additional,
        )

    def _pair_energies(
        self,
        relation: PreparedStreamedRelation,
        terms: _PairTerms,
        edge_active: Array,
        compute_distance: Array,
        sources: tuple[Array, Array, Array],
        receivers: tuple[Array, Array, Array],
        weights: Array,
        /,
    ) -> tuple[Array, StreamedRelationEvidence]:
        """Receiver-attributed embedded pair energies (``-E_sr + E_rep + E_ts``).

        ``sources`` and ``receivers`` are `_pair_data` rows of the relation's
        sources and receivers (the same rows on one bound graph; gathered owner
        columns and owned rows inside an owner region).
        """
        dtype = receivers[0].dtype
        node = jax.ShapeDtypeStruct((), dtype)
        message = {
            name: node
            for name, present in (
                ("coulomb", terms.coulomb is not None),
                ("repulsion", terms.repulsion is not None),
                ("dispersion", terms.dispersion is not None),
            )
            if present
        }
        evaluation = relation.evaluate(
            StreamedPayloadSpec(message=message, output=message),
            _pair_edge,
            _pair_epilogue,
            terms,
            sources,
            receivers,
            compute_distance,
            edge_active=edge_active,
        )
        result = evaluation.receiver_outputs
        total = jnp.zeros(receivers[0].shape, dtype=dtype)
        if "coulomb" in result:
            total = total - result["coulomb"]
        if "repulsion" in result:
            total = total + result["repulsion"]
        if "dispersion" in result:
            total = total + result["dispersion"]
        return total * weights, evaluation.evidence

    def _moment(
        self,
        stage: AIMNet2OutputStage,
        value: StageValue,
        inputs: AtomisticGraphInputs,
        charges: Array,
        species: Array,
        /,
    ) -> AtomisticObservableValue:
        cases, atoms = inputs.case_count, inputs.atom_capacity
        dtype = charges.dtype
        positions = inputs.positions.astype(dtype).reshape((cases, atoms, 3))
        mask = inputs.atom_mask.reshape((cases, atoms))
        if stage.center_coordinates:
            masses = (
                _stage_value(stage, value, AIMNet2MassTable).masses[species].astype(dtype)
            )
            masses = jnp.where(mask, masses, 0.0).reshape((cases, atoms, 1))
            center = jnp.sum(masses * positions, axis=1) / jnp.sum(masses, axis=1)
            positions = positions - center[:, None, :]
        q = (charges.reshape((cases, atoms)) * mask)[..., None]
        moment = _finish_moment(
            stage.kind, jnp.sum(_moment_contributions(stage.kind, q, positions), axis=1)
        )
        periodic = inputs.graph.topology.lattice_rank > 0 or (
            inputs.graph.topology.minimum_image_cell is not None
        )
        available = jnp.full((cases,), not periodic, dtype=jnp.bool_)
        moment = jnp.where(available[:, None], moment, jnp.zeros((), dtype))
        kind = "dipole" if stage.kind is AIMNet2OutputKind.DIPOLE else "quadrupole"
        return _observable(
            _moment_observable(stage.kind),
            _moment_port(kind, stage.center_coordinates),
            moment,
            available,
            AtomisticConditionLevel.CASE,
            inputs,
        )

    # Owner-region layered decomposition (`atomistic._feature_execution`).
    # Interaction ``k < pass_count`` is charge pass ``k``: the source atom
    # features (and, after pass 0, charges) cross the owner boundary, each
    # receiver forms its environment on the owner relation and runs the pass
    # MLP. System stage ``k`` (after every pass but the last) is the per-case
    # and per-channel charge equilibration, from owner sums of the
    # unconstrained charges and flexibilities, plus the feature delta.
    # Interaction ``pass_count`` (present iff an embedded pair stage exists)
    # exchanges the total charges and TS coefficients of the final pass for
    # the short-range Coulomb, repulsion and TS pair energies.

    @property
    def interaction_count(self) -> int:
        pair = 1 if self._pair_terms().radius > 0.0 else 0
        return self.configuration.pass_count + pair

    @property
    def system_stages(self) -> tuple[AtomisticSystemStage, ...]:
        return tuple(
            AtomisticSystemStage(f"aimnet2-charge-equilibration-{index}", index)
            for index in range(self.configuration.pass_count - 1)
        )

    def species_indices(self, species_ids: Array, atom_mask: Array, /) -> Array:
        """Executor species: atomic numbers of active atoms (zero on padding)."""
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        return jnp.where(mask, jnp.asarray(species_ids, dtype=jnp.int32), 0)

    def layer_context(
        self,
        conditioning: AtomisticConditioning | None,
        atom_cases: Array,
        species: Array,
        atom_mask: Array,
        /,
        *,
        key: PRNGKey | None = None,
    ) -> AIMNet2LayerContext:
        """Replicated channel targets, owned case rows and admitted numbers."""
        del key
        conditioning_ = self._require_conditioning(conditioning)
        dtype = jnp.dtype(self.precision.compute_dtype)
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        numbers, unsupported = self._numbers(jnp.asarray(species), mask)
        return AIMNet2LayerContext(
            self._targets(conditioning_, dtype),
            jnp.asarray(atom_cases, dtype=jnp.int32),
            numbers,
            ~jnp.any(unsupported),
        )

    def admitted(
        self, species: Array, atom_mask: Array, context: AIMNet2LayerContext, /
    ) -> Array:
        """Every owned active atom is an implemented model species."""
        del species, atom_mask
        return context.admitted

    def init_features(
        self, species: Array, atom_mask: Array, context: AIMNet2LayerContext, /
    ) -> AIMNet2NodeFeatures:
        del species
        dtype = jnp.dtype(self.precision.compute_dtype)
        weights = jnp.asarray(atom_mask).astype(dtype)
        return AIMNet2NodeFeatures(
            self._initial_features(context.numbers, weights),
            jnp.zeros(
                (weights.shape[0], self.configuration.charge_channels), dtype=dtype
            ),
        )

    def _final_pair_data(
        self,
        features: AIMNet2ReadoutFeatures,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
    ) -> tuple[Array, Array, Array]:
        dtype = jnp.dtype(self.precision.compute_dtype)
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        _, coefficients = self._readout_stages(
            features.aim, context.numbers, mask, mask.astype(dtype)
        )
        return self._pair_data(
            jnp.sum(features.charges, axis=-1), coefficients, context.numbers
        )

    def prepare_sources(
        self,
        index: int,
        features: AIMNet2NodeFeatures | AIMNet2ReadoutFeatures,
        species: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
    ) -> tuple[Array, ...]:
        """Pass sources ``(atom view[, charges])`` or final pair rows."""
        del species
        if index < self.configuration.pass_count:
            if not isinstance(features, AIMNet2NodeFeatures):
                raise TypeError("A charge pass reads equilibrated node features.")
            view = self._atom_view(features.atoms)
            return (view,) if index == 0 else (view, features.charges)
        if not isinstance(features, AIMNet2ReadoutFeatures):
            raise TypeError("The pair interaction reads the final-pass readout.")
        return self._final_pair_data(features, atom_mask, context)

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: tuple[Array, ...],
        features: AIMNet2NodeFeatures | AIMNet2ReadoutFeatures,
        species: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
        *,
        edge_active: Array | None = None,
        route_identity: tuple[Array, ...] | None = None,
    ) -> AIMNet2LayerUpdate:
        """One charge pass (pending equilibration) or the embedded pair energies.

        ``vectors`` are ``r_receiver - r_source (+ n H)``; distances and unit
        vectors follow the bound graph's finite zero-distance convention.
        """
        del species, route_identity
        architecture = self.configuration
        dtype = jnp.dtype(self.precision.compute_dtype)
        reduction = jnp.dtype(self.precision.reduction_dtype)
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        weights = mask.astype(dtype)
        displacement = jnp.asarray(vectors)
        squared = jnp.sum(displacement * displacement, axis=-1)
        zero = jnp.zeros((), dtype=displacement.dtype)
        tiny = jnp.asarray(jnp.finfo(displacement.dtype).tiny, displacement.dtype)
        distance = jnp.where(squared > zero, jnp.sqrt(jnp.maximum(squared, tiny)), zero)
        safe = jnp.where(distance > zero, distance, jnp.ones((), displacement.dtype))
        unit = -(displacement / safe[:, None])
        active = (
            jnp.ones(distance.shape, dtype=jnp.bool_)
            if edge_active is None
            else jnp.asarray(edge_active, dtype=jnp.bool_)
        )
        compute_distance = jnp.where(
            active & (distance > 0.0), distance, jnp.ones((), distance.dtype)
        ).astype(dtype)
        rows = mask.shape[0]
        if index < architecture.pass_count:
            if not isinstance(features, AIMNet2NodeFeatures):
                raise TypeError("A charge pass updates equilibrated node features.")
            atom_view = sources[0]
            charges = (
                jnp.zeros((atom_view.shape[0], architecture.charge_channels), dtype)
                if index == 0
                else sources[1]
            )
            evaluation = self._environment(
                index,
                relation,
                atom_view,
                charges,
                weights,
                (compute_distance, unit.astype(dtype)),
                active & (distance < architecture.aev_cutoff),
            )
            output = self._pass_output(
                index,
                features.atoms,
                features.charges,
                evaluation.receiver_outputs,
                weights,
            )
            if index == architecture.pass_count - 1:
                updated: AIMNet2PassOutput | AIMNet2ReadoutFeatures = (
                    AIMNet2ReadoutFeatures(output, features.charges)
                )
            else:
                updated = AIMNet2PassOutput(
                    features.atoms, *self._split_pass(index, output, features.charges)
                )
            return AIMNet2LayerUpdate(
                updated, jnp.zeros((rows,), dtype=reduction), evaluation.evidence
            )
        if not isinstance(features, AIMNet2ReadoutFeatures):
            raise TypeError("The pair interaction reads the final-pass readout.")
        terms = self._pair_terms()
        if len(sources) != 3:
            raise ValueError(
                "The pair interaction gathers (charges, coefficients, species) sources."
            )
        pair, evidence = self._pair_energies(
            relation,
            terms,
            active & (distance > 0.0) & (distance < terms.radius),
            compute_distance,
            (sources[0], sources[1], sources[2]),
            self._final_pair_data(features, mask, context),
            weights,
        )
        return AIMNet2LayerUpdate(features, pair.astype(reduction), evidence)

    def system_partials(
        self,
        stage: int,
        features: AIMNet2PassOutput | None,
        species: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
    ) -> AIMNet2ChargePartials:
        """Owned per-case, per-channel sums of ``q`` and ``f`` (source ``nse``)."""
        del stage, species
        if not isinstance(features, AIMNet2PassOutput):
            raise TypeError("Charge equilibration follows a non-final charge pass.")
        cases = context.targets.shape[0]
        rows = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
        zero = jnp.zeros((), dtype=features.unconstrained.dtype)
        return AIMNet2ChargePartials(
            jax.ops.segment_sum(
                jnp.where(rows, features.unconstrained, zero),
                context.atom_cases,
                num_segments=cases,
            ),
            jax.ops.segment_sum(
                jnp.where(rows, features.flexibility, zero),
                context.atom_cases,
                num_segments=cases,
            ),
        )

    def apply_system_context(
        self,
        stage: int,
        features: AIMNet2PassOutput | None,
        system_context: AIMNet2ChargePartials,
        species: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
    ) -> tuple[AIMNet2NodeFeatures, AIMNet2LayerContext]:
        """``q <- q + f (Q - sum q) / (sum f + 1e-6)``; ``a <- a + delta``."""
        del stage, species, atom_mask
        if not isinstance(features, AIMNet2PassOutput):
            raise TypeError("Charge equilibration follows a non-final charge pass.")
        total_flexibility = system_context.flexibility + AIMNET2_CHARGE_EPSILON
        deficit = context.targets - system_context.charge
        cases = context.atom_cases
        charges = (
            features.unconstrained
            + features.flexibility / total_flexibility[cases] * deficit[cases]
        )
        return AIMNet2NodeFeatures(features.atoms + features.delta, charges), context

    def atom_energies(
        self,
        species: Array,
        atom_mask: Array,
        layer_energies: Array,
        features: AIMNet2ReadoutFeatures,
        context: AIMNet2LayerContext,
        /,
    ) -> Array:
        """Energy head, atomic shifts and (when declared) embedded pair energies."""
        del species
        if not isinstance(features, AIMNet2ReadoutFeatures):
            raise TypeError("AIMNet2 atom energies read the final-pass readout.")
        dtype = jnp.dtype(self.precision.compute_dtype)
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        atom_energy, _ = self._readout_stages(
            features.aim, context.numbers, mask, mask.astype(dtype)
        )
        if self.interaction_count > self.configuration.pass_count:
            atom_energy = atom_energy + layer_energies[self.configuration.pass_count]
        return atom_energy * mask

    def global_payload(
        self,
        features: AIMNet2ReadoutFeatures,
        species: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
    ) -> None:
        """The source model energy is local (calculator terms are separate)."""
        del features, species, atom_mask, context
        return None

    def system_energy(self, context: AIMNet2LayerContext, /) -> None:
        del context
        return None

    def owner_fields(
        self,
        features: AIMNet2ReadoutFeatures,
        species: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
    ) -> Mapping[AtomisticObservable, Array]:
        """``CHARGES`` of the owned rows: the channel sum of the final charges.

        Exactly the ``CHARGES`` observable of `evaluate_graph` (the sum over
        charge channels of the last equilibrated charges), zero on padding.
        """
        del species, context
        if not isinstance(features, AIMNet2ReadoutFeatures):
            raise TypeError("AIMNet2 owner fields read the final-pass readout.")
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        total = jnp.sum(features.charges, axis=-1)
        return {
            AtomisticObservable.CHARGES: jnp.where(
                mask, total, jnp.zeros((), dtype=total.dtype)
            )
        }

    def owner_charge_residual(
        self, system_context: AIMNet2ChargePartials, context: AIMNet2LayerContext, /
    ) -> Array:
        """``CHARGE_RESIDUAL`` (target minus achieved) from the LAST stage context.

        The achieved channel charge of equilibrated charges is
        ``sum q_u + sum f / (sum f + 1e-6) (Q - sum q_u)``, an exact function of
        the replicated system context of the final equilibration stage.
        """
        total_flexibility = system_context.flexibility + AIMNET2_CHARGE_EPSILON
        deficit = context.targets - system_context.charge
        achieved = (
            system_context.charge
            + system_context.flexibility / total_flexibility * deficit
        )
        return context.targets - achieved

    def owner_moments(
        self,
        features: AIMNet2ReadoutFeatures,
        positions: Array,
        atom_mask: Array,
        context: AIMNet2LayerContext,
        /,
        *,
        axis_name: str,
        reduction_mode: OwnerReductionMode,
        periodic: bool,
    ) -> dict[AtomisticObservable, tuple[Array, Array]]:
        """Replicated declared charge moments ``(values, available)`` by case.

        Inside a mapped owner region: owned partial sums of the mass center
        (when the stage centers) and then of the moment contributions reduce
        once each through `owner_system_sum`, with the single-device moment
        definitions. Periodic cases have no branch-independent moment and are
        unavailable.
        """
        if not isinstance(features, AIMNet2ReadoutFeatures):
            raise TypeError("AIMNet2 moments read the final-pass readout.")
        cases = context.targets.shape[0]
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        charges = jnp.sum(features.charges, axis=-1)
        dtype = charges.dtype
        rows = context.atom_cases
        q = (charges * mask)[:, None]
        available = jnp.full((cases,), not periodic, dtype=jnp.bool_)
        values: dict[AtomisticObservable, tuple[Array, Array]] = {}
        for stage, value in zip(
            self.configuration.outputs, self.stage_values, strict=True
        ):
            if stage.kind not in (AIMNet2OutputKind.DIPOLE, AIMNet2OutputKind.QUADRUPOLE):
                continue
            coordinates = jnp.asarray(positions).astype(dtype)
            if stage.center_coordinates:
                masses = (
                    _stage_value(stage, value, AIMNet2MassTable)
                    .masses[context.numbers]
                    .astype(dtype)
                )
                masses = jnp.where(mask, masses, 0.0)[:, None]
                weighted, total = owner_system_sum(
                    (
                        jax.ops.segment_sum(
                            masses * coordinates, rows, num_segments=cases
                        ),
                        jax.ops.segment_sum(masses, rows, num_segments=cases),
                    ),
                    axis_name,
                    reduction_mode,
                )
                coordinates = coordinates - (weighted / total)[rows]
            raw = owner_system_sum(
                jax.ops.segment_sum(
                    _moment_contributions(stage.kind, q, coordinates),
                    rows,
                    num_segments=cases,
                ),
                axis_name,
                reduction_mode,
            )
            moment = jnp.where(
                available[:, None],
                _finish_moment(stage.kind, raw),
                jnp.zeros((), dtype),
            )
            values[_moment_observable(stage.kind)] = (moment, available)
        return values

    def _additional(
        self,
        request: AtomisticObservableRequest | None,
        inputs: AtomisticGraphInputs,
        successful: Array,
        channel_charges: Array,
        total_charge: Array,
        residual: Array,
        aim: Array,
        coefficients: Array | None,
        moments: dict[AtomisticObservable, AtomisticObservableValue],
        /,
    ) -> tuple[AtomisticObservableValue, ...]:
        if request is None:
            return ()
        selected = set(request.observables)
        values: list[AtomisticObservableValue] = []
        atom = AtomisticConditionLevel.ATOM
        case = AtomisticConditionLevel.CASE
        if AtomisticObservable.CHARGES in selected:
            values.append(
                _observable(
                    AtomisticObservable.CHARGES,
                    _CHARGE_PORT,
                    total_charge,
                    successful,
                    atom,
                    inputs,
                )
            )
        if AtomisticObservable.SPIN_CHARGES in selected:
            values.append(
                _observable(
                    AtomisticObservable.SPIN_CHARGES,
                    _SPIN_CHARGE_PORT,
                    channel_charges[:, 0] - channel_charges[:, 1],
                    successful,
                    atom,
                    inputs,
                )
            )
        if AtomisticObservable.CHARGE_RESIDUAL in selected:
            values.append(
                _observable(
                    AtomisticObservable.CHARGE_RESIDUAL,
                    _residual_port(self.configuration.charge_channels),
                    residual,
                    successful,
                    case,
                    inputs,
                )
            )
        if AtomisticObservable.NODE_EMBEDDINGS in selected:
            values.append(
                _observable(
                    AtomisticObservable.NODE_EMBEDDINGS,
                    ValuePort(
                        _AIM_PORT_SEMANTIC,
                        event_shape=(self.configuration.aim_size,),
                        component_ids=tuple(
                            f"aim{index}" for index in range(self.configuration.aim_size)
                        ),
                        representation="aimnet2-final-pass-aim-vector",
                        normalization_id="identity",
                    ),
                    aim,
                    successful,
                    atom,
                    inputs,
                )
            )
        for observable in (AtomisticObservable.DIPOLE, AtomisticObservable.QUADRUPOLE):
            if observable in selected:
                moment = moments[observable]
                values.append(
                    _observable(
                        observable,
                        moment.port,
                        moment.values,
                        moment.available & successful,
                        case,
                        inputs,
                    )
                )
        if coefficients is not None:
            for observable, port, column in (
                (AtomisticObservable.ATOMIC_C6, _C6_PORT, 0),
                (AtomisticObservable.ATOMIC_POLARIZABILITY, _POLARIZABILITY_PORT, 1),
            ):
                if observable in selected:
                    values.append(
                        _observable(
                            observable,
                            port,
                            coefficients[:, column] * inputs.atom_mask,
                            successful,
                            atom,
                            inputs,
                        )
                    )
        return tuple(values)

    def __call__(
        self,
        structure: AtomicStructure | AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        conditioning: AtomisticConditioning,
    ) -> Array:
        if isinstance(structure, AtomicStructure):
            batch = AtomisticBatch.from_structure(structure)
            return self.energy(batch, execution, conditioning=conditioning)[0]
        if isinstance(structure, AtomisticBatch):
            return self.energy(structure, execution, conditioning=conditioning)
        raise TypeError("AIMNet2Potential expects AtomicStructure or AtomisticBatch.")


def _moment_observable(kind: AIMNet2OutputKind, /) -> AtomisticObservable:
    if kind is AIMNet2OutputKind.DIPOLE:
        return AtomisticObservable.DIPOLE
    return AtomisticObservable.QUADRUPOLE


def _observable(
    observable: AtomisticObservable,
    port: ValuePort,
    values: Array,
    available: Array,
    level: AtomisticConditionLevel,
    inputs: AtomisticGraphInputs,
    /,
) -> AtomisticObservableValue:
    return AtomisticObservableValue(
        observable,
        port,
        values,
        available,
        level=level,
        case_count=inputs.case_count,
        atom_capacity=inputs.atom_capacity,
        unit=_OUTPUT_UNITS.get(port.semantic_id),
    )


def _equilibrate(
    targets: Array,
    unconstrained: Array,
    flexibility: Array,
    cases: int,
    atoms: int,
    /,
) -> Array:
    """Source ``ops.nse``: ``q + f (Q - sum q) / (sum f + 1e-6)`` per case/channel."""
    channels = targets.shape[-1]
    q = unconstrained.reshape((cases, atoms, channels))
    f = flexibility.reshape((cases, atoms, channels))
    total_flexibility = jnp.sum(f, axis=1, keepdims=True) + AIMNET2_CHARGE_EPSILON
    deficit = targets[:, None, :] - jnp.sum(q, axis=1, keepdims=True)
    return (q + f / total_flexibility * deficit).reshape((cases * atoms, channels))


def _fixed_identity(value: StageValue, /) -> Any:
    if isinstance(value, TSDispersionReference):
        return value.reference_id
    if isinstance(value, TSDispersionPotential):
        return value.potential_id
    if isinstance(value, AIMNet2ShortRangeRadius):
        return value.value
    if isinstance(value, AIMNet2MassTable):
        return array_tree_fingerprint(_host(value.masses))
    if isinstance(value, AIMNet2FrozenAtomicShifts):
        return array_tree_fingerprint(_host(value.shifts))
    if isinstance(value, AIMNet2Repulsion):
        return {
            "repulsion": array_tree_fingerprint(
                (_host(value.parameters), _host(value.radius))
            ),
            "cutoff_function": value.cutoff_function,
            "route_cutoff": value.route_cutoff,
        }
    return None


def _potential_identity(
    architecture: AIMNet2Architecture,
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy,
    basis: AIMNet2RadialBasis,
    stage_values: Sequence[StageValue],
    /,
) -> str:
    """Scientific identity: declaration, contracts, fixed tables and conditioning."""
    return canonical_fingerprint(
        {
            "kind": "aimnet2-potential",
            "architecture": architecture.architecture_id,
            "scale": scale.scale_id,
            "precision": precision.policy_id,
            "basis": basis.basis_id,
            "fixed": tuple(_fixed_identity(value) for value in stage_values),
            "conditioning": architecture.conditioning_spec().spec_id,
        }
    )


def _require_component(
    component: AIMNet2RadialBasis
    | AIMNet2ConvSV
    | AIMNet2MLP
    | AIMNet2AtomicShifts
    | AIMNet2FrozenAtomicShifts
    | AIMNet2ShortRangeRadius
    | AIMNet2Repulsion
    | AIMNet2MassTable
    | TSDispersionReference
    | TSDispersionPotential,
    name: str,
    dtype: np.dtype[Any],
    /,
) -> None:
    """Refuse a component unless its own constructor rebuilds it exactly.

    The rebuild reruns every owning validator on the stored leaves (a restored
    or bypassed value never skips them) and must equal the stored value
    bitwise, identities and structural metadata included. Floating numerical
    leaves must use ``dtype``; masks and identity words retain their owning dtype.
    """
    expected_types = (
        AIMNet2RadialBasis,
        AIMNet2ConvSV,
        AIMNet2MLP,
        AIMNet2AtomicShifts,
        AIMNet2FrozenAtomicShifts,
        AIMNet2ShortRangeRadius,
        AIMNet2Repulsion,
        AIMNet2MassTable,
        TSDispersionReference,
        TSDispersionPotential,
    )
    if not isinstance(component, expected_types):
        raise TypeError(f"{name} is not an AIMNet2 component.")
    if any(
        np.issubdtype(leaf.dtype, np.inexact) and np.dtype(leaf.dtype) != dtype
        for leaf in jax.tree_util.tree_leaves(component)
    ):
        raise ValueError(f"{name} leaves are not in the declared {dtype.name} dtype.")
    require_identical(component, component.rebuilt(), name, ValueError)


def _check_components(
    architecture: AIMNet2Architecture,
    embedding: Array,
    basis: AIMNet2RadialBasis,
    atom_conv: AIMNet2ConvSV,
    charge_conv: AIMNet2ConvSV,
    passes: tuple[AIMNet2MLP, ...],
    stage_values: tuple[StageValue, ...],
    /,
) -> None:
    if embedding.shape != (AIMNET2_SPECIES_CAPACITY, architecture.feature_total):
        raise ValueError("The species embedding does not match the architecture.")
    if not isinstance(basis, AIMNet2RadialBasis):
        raise TypeError("basis must be an AIMNet2RadialBasis.")
    if basis.shift_count != architecture.shift_count:
        raise ValueError("The AEV basis shift count differs from the architecture.")
    if basis.radius != architecture.aev_cutoff:
        raise ValueError("The AEV basis cutoff differs from the architecture.")
    expected = (
        (atom_conv, architecture.feature_count, architecture.d2_features, "conv_a"),
        (charge_conv, architecture.charge_channels, False, "conv_q"),
    )
    for conv, channels, d2, name in expected:
        if (
            not isinstance(conv, AIMNet2ConvSV)
            or conv.channel_count != channels
            or conv.shift_count != architecture.shift_count
            or conv.combination_count != architecture.combination_count
            or conv.d2_features != d2
        ):
            raise ValueError(f"{name} does not match the architecture.")
    if len(passes) != architecture.pass_count:
        raise ValueError("Charge-pass MLP count differs from the architecture.")
    for mlp, widths, (inputs, outputs, last_linear) in zip(
        passes, architecture.pass_hidden, architecture.pass_shapes(), strict=True
    ):
        if (
            not isinstance(mlp, AIMNet2MLP)
            or mlp.in_size != inputs
            or mlp.out_size != outputs
            or mlp.widths != widths
            or mlp.last_linear != last_linear
            or not mlp.biases
            or mlp.activation != ("gelu", ())
        ):
            raise ValueError("A charge-pass MLP does not match the architecture.")
    if len(stage_values) != len(architecture.outputs):
        raise ValueError("Stage values must align with the output stages.")
    for stage, value in zip(architecture.outputs, stage_values, strict=True):
        _check_stage_value(stage, value, architecture)


for _artifact in (
    AIMNet2Potential,
    AIMNet2Architecture,
    AIMNet2OutputStage,
    AIMNet2OutputKind,
    AIMNet2MLP,
    AIMNet2AtomicShifts,
    AIMNet2FrozenAtomicShifts,
    AIMNet2ShortRangeRadius,
    AIMNet2Repulsion,
    AIMNet2MassTable,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_artifact.__name__}", _artifact)
del _artifact


__all__ = [
    "AIMNET2_CHARGE_EPSILON",
    "AIMNET2_COULOMB_PAIR_FACTOR",
    "AIMNET2_SOURCE_COMMIT",
    "AIMNET2_SOURCE_RELEASE",
    "AIMNET2_SPECIES_CAPACITY",
    "AIMNet2Architecture",
    "AIMNet2AtomicShifts",
    "AIMNet2ChargePartials",
    "AIMNet2LayerContext",
    "AIMNet2LayerUpdate",
    "AIMNet2NodeFeatures",
    "AIMNet2PassOutput",
    "AIMNet2ReadoutFeatures",
    "AIMNet2FrozenAtomicShifts",
    "AIMNet2MassTable",
    "AIMNet2Repulsion",
    "AIMNET2_ACTIVATIONS",
    "apply_activation",
    "AIMNet2MLP",
    "AIMNet2OutputKind",
    "AIMNet2OutputStage",
    "AIMNet2Potential",
    "AIMNet2ShortRangeRadius",
]

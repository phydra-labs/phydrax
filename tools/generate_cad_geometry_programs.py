#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Generate the native CAD geometry template programs.

Each straight-line source kind of ``CAD_TEMPLATES`` is lowered once from the
JAX ``evaluate`` its Python class uses, through the native interval lowering
in batched (non-folding) mode, with every coefficient leaf a seed. Three
programs per kind: the value; the tangent (value and first partials), whose
second-order jets carry third derivatives; and the Hessian (value, first and
second partials), whose jets carry fourth derivatives. The output
``native/meshcore/src/cad_geometry_programs.inc`` is checked in; run

    python tools/generate_cad_geometry_programs.py

after a change to a lowered ``evaluate`` or to the template table.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jax_core
from numpy.typing import NDArray

from phydrax.geometry._interval_native import lower_interval_jaxpr
from phydrax.geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    HyperbolaCurve,
    LineCurve,
    ParabolaCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    SpherePatch,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from phydrax.meshing.providers._native_cad_bundle import CAD_TEMPLATES, CadTemplate


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "native"
    / "meshcore"
    / "src"
    / "cad_geometry_programs.inc"
)

type _Entity = AbstractSurfacePatch | AbstractCurve


def _frame(dimension: int, /) -> tuple[NDArray[np.float64], ...]:
    eye = np.eye(dimension, dtype=np.float64)
    return eye[0], eye[1]


def _conic_root() -> ConicParameterRoot:
    circle = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)
    return ConicParameterRoot(circle, Fraction(3, 5), Fraction(4, 5))


def _sample(template: CadTemplate, /) -> _Entity:
    """A valid instance of the template's kind and static configuration."""
    dimension = template.output_dimension
    origin = np.zeros(dimension, dtype=np.float64)
    first, second = _frame(dimension)
    axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    kind = template.kind
    if kind is PlanePatch:
        return PlanePatch(origin, first, second)
    if kind is CylinderPatch:
        return CylinderPatch(origin, first, second, axis, 1.0)
    if kind is ConePatch:
        return ConePatch(origin, first, second, axis, 1.0, 0.25)
    if kind is SpherePatch:
        return SpherePatch(origin, first, second, axis, 1.0)
    if kind is TorusPatch:
        return TorusPatch(origin, first, second, axis, 2.0, 0.5)
    if kind is LineCurve:
        return LineCurve(origin, first)
    if kind is ExactLineCurve:
        return ExactLineCurve(origin, first)
    if kind is CircleCurve:
        return CircleCurve(origin, first, second, 1.0)
    if kind is EllipseCurve:
        return EllipseCurve(origin, first, second, 2.0, 1.0)
    if kind is ParabolaCurve:
        return ParabolaCurve(origin, first, second, 0.5)
    if kind is HyperbolaCurve:
        return HyperbolaCurve(origin, first, second, 2.0, 1.0)
    if kind is ConicRootLinePCurve:
        return ConicRootLinePCurve(_conic_root(), Fraction(1, 2), Fraction(3, 2))
    if kind is TorusRootedCirclePCurve:
        ((_, orientation),) = template.static
        return TorusRootedCirclePCurve(_conic_root(), Fraction(1, 4), orientation)
    if kind is PlaneCylinderSectionPCurve:
        return PlaneCylinderSectionPCurve(
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(0)),
            parameter_range=(0.0, 1.0),
        )
    raise ValueError(f"No generator sample for template {template.name!r}.")


def _leaf(entity: object, path: tuple[str, ...], /) -> object:
    for name in path:
        entity = getattr(entity, name)
    return entity


def _point_function(
    template: CadTemplate, sample: _Entity, derivatives: int, /
) -> Callable[..., jax.Array]:
    """Evaluate one point, and its partial derivatives through ``derivatives``,
    with the coefficient leaves as traced arguments."""

    def value(point: jax.Array, coefficients: Sequence[jax.Array], /) -> jax.Array:
        entity = eqx.tree_at(
            lambda item: tuple(_leaf(item, path) for path in template.fields),
            sample,
            tuple(coefficients),
        )
        argument = point if template.input_dimension == 2 else point[0]
        return jnp.reshape(entity.evaluate(argument), (template.output_dimension,))

    def evaluate(point: jax.Array, *coefficients: jax.Array) -> jax.Array:
        def function(x: jax.Array, /) -> jax.Array:
            return value(x, coefficients)

        parts = [function(point)]
        if derivatives >= 1:
            # Input after input: df/dx0, df/dx1.
            parts.append(jnp.reshape(jax.jacfwd(function)(point).T, (-1,)))
        if derivatives >= 2:
            # Pairs a <= b in order (0, 0), (0, 1), (1, 1).
            hessian = jax.jacfwd(jax.jacfwd(function))(point)
            k = template.input_dimension
            parts.extend(
                hessian[:, first, second]
                for first in range(k)
                for second in range(first, k)
            )
        return jnp.concatenate(parts)

    return evaluate


def _program(template: CadTemplate, derivatives: int, /) -> dict[str, NDArray]:
    sample = _sample(template)
    leaves = tuple(
        np.asarray(_leaf(sample, path), dtype=np.float64) for path in template.fields
    )
    function = _point_function(template, sample, derivatives)
    probe = jax.ShapeDtypeStruct((1, template.input_dimension), np.float64)
    closed = jax.make_jaxpr(jax.vmap(function, in_axes=(0, *(None,) * len(leaves))))(
        probe, *leaves
    )
    jaxpr = closed.jaxpr
    point_variable, *coefficient_variables = jaxpr.invars
    # Coefficients become constants: seeds of the lowered program.
    lowered_jaxpr = jax_core.Jaxpr(
        (*jaxpr.constvars, *coefficient_variables),
        (point_variable,),
        jaxpr.outvars,
        jaxpr.eqns,
        jaxpr.effects,
    )
    constants = (*closed.consts, *leaves)
    program = lower_interval_jaxpr(
        jax_core.ClosedJaxpr(lowered_jaxpr, list(constants)),
        tuple(
            (np.asarray(value, np.float64), np.asarray(value, np.float64))
            for value in constants
        ),
        1,
        template.input_dimension,
        elementwise=True,
    )
    # Point seed template: every seed at its value (the lowering keeps exact
    # constants degenerate); coefficient entries are filled per entity.
    seeds = np.zeros((program.workspace_size,), dtype=np.float64)
    for seed in program.seeds:
        seeds[seed.offset : seed.offset + seed.lower.size] = 0.5 * (
            seed.lower + seed.upper
        )
    output = program.output
    outputs = (
        np.arange(output.offset, output.offset + math.prod(output.shape), dtype=np.int64)
        if output.locations is None
        else np.asarray(output.locations, dtype=np.int64).reshape((-1,))
    )
    coefficient_seeds = program.seeds[len(closed.consts) : len(constants)]
    entries = np.concatenate(
        [
            np.arange(seed.offset, seed.offset + seed.lower.size, dtype=np.int64)
            for seed in coefficient_seeds
        ]
    )
    seeds[entries] = 0.0
    return {
        "instructions": program.instructions.reshape((-1,)),
        "indices": program.indices,
        "parameters": program.parameters,
        "seeds": seeds,
        "coefficients": entries,
        "outputs": outputs,
    }


def _integers(values: NDArray, /) -> str:
    return ", ".join(str(int(value)) for value in values)


def _reals(values: NDArray, /) -> str:
    return ", ".join(float(value).hex() for value in values)


def _array(name: str, kind: str, body: str, /) -> str:
    # A zero-length C++ array is ill-formed; one padding element is never read.
    return f"inline constexpr {kind} {name}[] = {{{body if body else '0'}}};\n"


def render() -> str:
    """The generated include, deterministic for one JAX and Phydrax revision."""
    lines = [
        "// Copyright © 2026 PHYDRA, Inc. All rights reserved.\n",
        "// Generated by tools/generate_cad_geometry_programs.py; do not edit.\n",
        "// Template programs of the straight-line CAD source kinds, lowered from\n",
        "// their JAX evaluate with every coefficient a seed. Included by\n",
        "// cad_geometry.cpp inside namespace phx::mc::cad.\n",
        "\n",
    ]
    rows: list[str] = []
    for identifier, template in enumerate(CAD_TEMPLATES):
        coefficient_count = sum(
            int(np.asarray(_leaf(_sample(template), path)).size)
            for path in template.fields
        )
        programs = []
        for label, derivatives in (("value", 0), ("tangent", 1), ("hessian", 2)):
            program = _program(template, derivatives)
            stem = f"k{identifier}_{label}"
            lines.append(f"// {template.name} {label}\n")
            lines.append(
                _array(
                    f"{stem}_instructions", "int64_t", _integers(program["instructions"])
                )
            )
            lines.append(
                _array(f"{stem}_indices", "int64_t", _integers(program["indices"]))
            )
            lines.append(
                _array(f"{stem}_parameters", "double", _reals(program["parameters"]))
            )
            lines.append(_array(f"{stem}_seeds", "double", _reals(program["seeds"])))
            lines.append(
                _array(
                    f"{stem}_coefficients", "int64_t", _integers(program["coefficients"])
                )
            )
            lines.append(
                _array(f"{stem}_outputs", "int64_t", _integers(program["outputs"]))
            )
            programs.append(
                f"{{{stem}_instructions, {program['instructions'].size // 8}, "
                f"{stem}_indices, {program['indices'].size}, "
                f"{stem}_parameters, {program['parameters'].size}, "
                f"{stem}_seeds, {program['seeds'].size}, "
                f"{stem}_coefficients, {stem}_outputs, {program['outputs'].size}}}"
            )
        rows.append(
            f'    {{"{template.name}", {template.input_dimension}, '
            f"{template.output_dimension}, {coefficient_count}, "
            f"{programs[0]}, {programs[1]}, {programs[2]}}},\n"
        )
    lines.append("\n")
    lines.append(f"inline constexpr int32_t kTemplateCount = {len(CAD_TEMPLATES)};\n")
    lines.append("inline constexpr TemplateDefinition kTemplates[kTemplateCount] = {\n")
    lines.extend(rows)
    lines.append("};\n")
    return "".join(lines)


def main(arguments: Sequence[str], /) -> int:
    if arguments:
        raise SystemExit("usage: python tools/generate_cad_geometry_programs.py")
    OUTPUT.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

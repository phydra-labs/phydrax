#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from phydrax.domain import Domain, DomainFunction

from .._strict import StrictModule


def _domains_compatible(a: Domain, b: Domain, /) -> bool:
    return a.schema_compatible(b)


def _join_target_domains(
    substitutions: Mapping[str, DomainFunction], /
) -> Domain:
    iterator = iter(substitutions.values())
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("pullback requires domain=... when substitutions is empty.") from exc

    target = first.domain
    for replacement in iterator:
        if _domains_compatible(target, replacement.domain):
            continue
        target = target.join(replacement.domain)
    return target


class _PullbackCallable(StrictModule):
    source: DomainFunction
    replacements: tuple[DomainFunction | None, ...]
    replacement_positions: tuple[tuple[int, ...], ...]
    passthrough_positions: tuple[int | None, ...]

    def __init__(
        self,
        source: DomainFunction,
        replacements: tuple[DomainFunction | None, ...],
        replacement_positions: tuple[tuple[int, ...], ...],
        passthrough_positions: tuple[int | None, ...],
    ):
        self.source = source
        self.replacements = replacements
        self.replacement_positions = replacement_positions
        self.passthrough_positions = passthrough_positions

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        source_args: list[Any] = []
        for replacement, positions, passthrough_position in zip(
            self.replacements,
            self.replacement_positions,
            self.passthrough_positions,
            strict=True,
        ):
            if replacement is None:
                if passthrough_position is None:
                    raise RuntimeError("Invalid pullback passthrough plan.")
                source_args.append(args[passthrough_position])
                continue

            replacement_args = tuple(args[position] for position in positions)
            source_args.append(
                replacement.func(*replacement_args, key=key, **kwargs)
            )

        return self.source.func(*source_args, key=key, **kwargs)


def pullback(
    f: DomainFunction,
    substitutions: Mapping[str, DomainFunction],
    /,
    *,
    domain: Domain | None = None,
) -> DomainFunction:
    r"""Compose a domain function with labeled substitution fields.

    Given a function $f(q,p,t)$ and trajectory fields $q(t)$ and $p(t)$,
    ``pullback(f, {"q": q, "p": p})`` constructs the function

    $$
    t \mapsto f(q(t),p(t),t).
    $$

    Dependencies not present in ``substitutions`` pass through unchanged when a
    same-named coordinate exists on the target domain. Every other dependency must
    have an explicit substitution.

    **Arguments:**

    - `f`: Source function to compose.
    - `substitutions`: Mapping from source dependency labels to replacement fields.
    - `domain`: Optional target domain. Required when `substitutions` is empty.

    **Returns:**

    A `DomainFunction` on the target domain.
    """
    if not isinstance(f, DomainFunction):
        raise TypeError(f"pullback expected a DomainFunction, got {type(f).__name__}.")

    substitutions_ = dict(substitutions)
    unknown = tuple(label for label in substitutions_ if label not in f.deps)
    if unknown:
        raise ValueError(
            f"pullback substitutions contain labels not used by the source function: {unknown!r}."
        )
    for label, replacement in substitutions_.items():
        if not isinstance(replacement, DomainFunction):
            raise TypeError(
                "pullback substitutions must be DomainFunction instances; "
                f"substitution {label!r} has type {type(replacement).__name__}."
            )

    if domain is None:
        target = _join_target_domains(substitutions_)
    else:
        target = domain
        for label, replacement in substitutions_.items():
            for replacement_label in replacement.domain.labels:
                if replacement_label not in target.labels:
                    raise ValueError(
                        f"pullback substitution {label!r} uses target label "
                        f"{replacement_label!r}, which is absent from domain {target.labels!r}."
                    )
                if not target.coordinate(replacement_label).compatible(
                    replacement.domain.coordinate(replacement_label)
                ):
                    raise ValueError(
                        f"pullback substitution {label!r} has an incompatible factor "
                        f"for target label {replacement_label!r}."
                    )

    promoted = {
        label: replacement.promote(target)
        for label, replacement in substitutions_.items()
    }

    unresolved = tuple(
        label for label in f.deps if label not in promoted and label not in target.labels
    )
    if unresolved:
        raise ValueError(
            "pullback cannot resolve source dependencies; provide substitutions for "
            f"{unresolved!r}."
        )

    deps = tuple(
        label
        for label in target.labels
        if label in f.deps
        or any(label in replacement.deps for replacement in promoted.values())
    )
    dependency_positions = {label: index for index, label in enumerate(deps)}

    replacements: list[DomainFunction | None] = []
    replacement_positions: list[tuple[int, ...]] = []
    passthrough_positions: list[int | None] = []
    for source_label in f.deps:
        replacement = promoted.get(source_label)
        replacements.append(replacement)
        if replacement is None:
            replacement_positions.append(())
            passthrough_positions.append(dependency_positions[source_label])
        else:
            replacement_positions.append(
                tuple(dependency_positions[label] for label in replacement.deps)
            )
            passthrough_positions.append(None)

    return DomainFunction(
        domain=target,
        deps=deps,
        func=_PullbackCallable(
            source=f,
            replacements=tuple(replacements),
            replacement_positions=tuple(replacement_positions),
            passthrough_positions=tuple(passthrough_positions),
        ),
        metadata=f.metadata,
    )


__all__ = ["pullback"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrustedTorchPickleSource:
    """Explicit permission to execute a pickle from one admitted SHA-256 digest.

    This is the caller's code-execution trust decision, not a model license or
    a fallback from a refused safe tensor format.
    """

    sha256: str
    statement: str

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str):
            raise TypeError("Trusted source sha256 must be a string.")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("Trusted source sha256 must be lowercase hexadecimal.")
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("A trusted pickle source requires a trust statement.")


__all__ = ["TrustedTorchPickleSource"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import sys
from collections.abc import Sequence

from .cli import main as benchmark_main


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the configured CI/local JSON report without embedded result data."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    benchmark_main(["run", *arguments])


if __name__ == "__main__":
    main()

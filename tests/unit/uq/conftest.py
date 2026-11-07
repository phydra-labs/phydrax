#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

# Importing BlackJAX initializes a JAX backend, after which Phydrax refuses to
# install its frontend index patch. Sorted imports place `blackjax` before
# `phydrax` in the test modules, so this directory conftest, which pytest imports
# before collecting them, owns the documented "import Phydrax first" order.
import phydrax  # noqa: F401

# Copyright (C) 2026 Kival
# This file is part of the Arkantos AI Stack Suite.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Basalt Deep Learning Python package (`basalt-dl` on PyPI, `import basalt`)."""

from ._version import __version__

try:
    from ._basalt import (
        Device,
        Dtype,
        Linear,
        Sequential,
        Tensor,
        add,
        matmul,
        mse_loss,
        mul,
        ones,
        randn,
        relu,
        seed_everything,
        zeros,
    )
except ImportError:
    pass

__all__ = [
    "__version__",
    "Device",
    "Dtype",
    "Linear",
    "Sequential",
    "Tensor",
    "add",
    "matmul",
    "mse_loss",
    "mul",
    "ones",
    "randn",
    "relu",
    "seed_everything",
    "zeros",
]

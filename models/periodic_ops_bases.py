"""
Wavelet base variants for periodic ops.

These classes only change the wavelet basis while reusing the strictly
periodic implementation.
"""

from models.periodic_ops_strictly_periodic import PeriodicDWT2D as _BaseDWT
from models.periodic_ops_strictly_periodic import PeriodicIDWT2D as _BaseIDWT


class PeriodicDWT2D_DB2(_BaseDWT):
    def __init__(self, format="cat"):
        super().__init__(wave="db2", format=format)


class PeriodicIDWT2D_DB2(_BaseIDWT):
    def __init__(self):
        super().__init__(wave="db2")


class PeriodicDWT2D_DB4(_BaseDWT):
    def __init__(self, format="cat"):
        super().__init__(wave="db4", format=format)


class PeriodicIDWT2D_DB4(_BaseIDWT):
    def __init__(self):
        super().__init__(wave="db4")


class PeriodicDWT2D_SYM4(_BaseDWT):
    def __init__(self, format="cat"):
        super().__init__(wave="sym4", format=format)


class PeriodicIDWT2D_SYM4(_BaseIDWT):
    def __init__(self):
        super().__init__(wave="sym4")

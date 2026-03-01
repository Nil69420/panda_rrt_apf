"""Backward-compatibility shim.

All classes have been moved to dedicated modules:

* :mod:`panda_rrt.collision`   → :class:`CollisionChecker`
* :mod:`panda_rrt.cspace_apf`  → :class:`CSpaceAPF`
* :mod:`panda_rrt.apf_rrt`     → :class:`APFGuidedRRT`

Import them directly from those modules in new code.
"""

from panda_rrt.collision import CollisionChecker, COLLISION_LINKS, SAFETY_MARGIN  # noqa: F401
from panda_rrt.cspace_apf import CSpaceAPF  # noqa: F401
from panda_rrt.apf_rrt import APFGuidedRRT  # noqa: F401

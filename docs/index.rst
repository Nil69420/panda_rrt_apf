Panda RRT Documentation
=======================

Motion planning for the Franka Emika Panda arm using RRT variants
with C-space Artificial Potential Fields, gradient-descent optimisation,
and B-spline smoothing.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   modules

Modules
-------

Computations
~~~~~~~~~~~~

.. automodule:: panda_rrt.computations
   :members:

.. automodule:: panda_rrt.computations.jit_kernels
   :members:

.. automodule:: panda_rrt.computations.forward_kinematics
   :members:

.. automodule:: panda_rrt.computations.collision
   :members:

.. automodule:: panda_rrt.computations.cspace_apf
   :members:

.. automodule:: panda_rrt.computations.spline_smoother
   :members:

Planners
~~~~~~~~

.. automodule:: panda_rrt.planners
   :members:

.. automodule:: panda_rrt.planners.core
   :members:

.. automodule:: panda_rrt.planners.pure_rrt
   :members:

.. automodule:: panda_rrt.planners.rrt_star
   :members:

.. automodule:: panda_rrt.planners.apf_rrt
   :members:

.. automodule:: panda_rrt.planners.optimizer
   :members:

Benchmark Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: panda_rrt.benchmark_utilities
   :members:

.. automodule:: panda_rrt.benchmark_utilities.runner
   :members:

.. automodule:: panda_rrt.benchmark_utilities.graphs
   :members:

Environment & Scene
~~~~~~~~~~~~~~~~~~~

.. automodule:: panda_rrt.environment
   :members:

.. automodule:: panda_rrt.scene
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

# Panda RRT

Motion planning for the Franka Emika Panda 7-DOF arm using RRT-based planners with C-space Artificial Potential Fields, gradient-descent optimisation, and B-spline smoothing.

## Project Structure

```
panda_rrt/
    computations/           # Numerical modules
        jit_kernels.py      # Numba JIT-accelerated inner loops
        forward_kinematics.py
        collision.py
        cspace_apf.py       # C-space Artificial Potential Field
        spline_smoother.py
    planners/               # Motion planning algorithms
        core.py             # Node, PlannerResult dataclasses
        pure_rrt.py         # Vanilla RRT
        rrt_star.py         # Asymptotically optimal RRT*
        apf_rrt.py          # APF-guided RRT
        optimizer.py        # Gradient-descent path smoothing
    benchmark_utilities/    # Benchmarking and visualisation
        runner.py           # Multi-trial benchmark engine
        graphs.py           # Matplotlib graph generation
        tree_viz.py         # 3-D tree visualisation
        visualisation.py    # PyBullet markers and traces
    config/
        default.yaml        # All tuneable parameters
    environment.py          # PyBullet simulation wrapper
    scene.py                # Obstacle spawning and goal sampling
    main.py                 # Unified CLI entry-point
    benchmark.py            # Benchmark CLI wrapper
    tests/                  # Pytest suite
```

## Installation

```bash
# Clone and install panda-gym fork
cd panda-gym && pip install -e .

# Install dependencies
cd ../panda_rrt
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH=/path/to/panda_rrt:/path/to/stuff:$PYTHONPATH
```

## Quick Start

### Visual Simulation

```bash
# Interactive planner menu
python -m panda_rrt.main

# Direct planner selection
python -m panda_rrt.main --rrt              # Pure RRT
python -m panda_rrt.main --rrt_apf          # APF-Guided RRT
python -m panda_rrt.main --rrt_apf_opt      # APF-RRT + optimisation
python -m panda_rrt.main --rrt_apf_spline   # APF-RRT + B-spline

# Scene selection
python -m panda_rrt.main --rrt_apf --canopy
python -m panda_rrt.main --rrt_apf --passage

# Random start/goal
python -m panda_rrt.main --rrt_apf --random
```

### Headless Demo

```bash
python -m panda_rrt.main --demo
```

### Benchmarks

```bash
# Interactive preset menu
python -m panda_rrt.main --benchmark

# Direct preset selection
python -m panda_rrt.main --benchmark --preset apf --trials 20

# Generate benchmark graphs
python -m panda_rrt.main --graph --preset full --save benchmark.png

# Alternative benchmark CLI
python -m panda_rrt.benchmark --preset rrt --trials 10
python -m panda_rrt.benchmark --graph --save results.png
```

### 3-D Tree Visualisation

```bash
# All planners (2x2 grid)
python -m panda_rrt.main --viz

# Single planner
python -m panda_rrt.main --viz --planner apf_rrt

# Save to file
python -m panda_rrt.main --viz --save tree.png
```

## Planners

| Planner | Description |
|---------|-------------|
| Pure RRT | Vanilla RRT with uniform random sampling |
| RRT* | Asymptotically optimal RRT with near-neighbour rewiring |
| APF-RRT | RRT guided by C-space Artificial Potential Fields |
| APF-RRT+Opt | APF-RRT followed by gradient-descent path optimisation |
| APF-RRT+Spline | APF-RRT followed by cubic B-spline smoothing |

## Configuration

All tuneable parameters are in `config/default.yaml`:

- Collision checking (links, safety margin)
- Planner parameters (goal bias, step size, max iterations)
- APF parameters (attraction/repulsion gains, cutoff radius)
- Optimiser parameters (learning rate, lambda, danger threshold)
- Visualisation parameters (animation speed, marker colours)
- Environment dimensions (table size, obstacle ranges)
- Scene parameters (sampling attempts, minimum distances)

## JIT Acceleration

When Numba is installed, the following hot loops are JIT-compiled:

- Nearest-neighbour search (brute-force L2 scan over the tree)
- Ball-radius near-neighbour search for RRT* (parallel via ``prange``)
- Path length computation
- Smoothness cost evaluation
- Batch smoothness gradient for the optimizer (parallel via ``prange``)
- C-space attractive gradient (piecewise quadratic/conic)
- 3-D APF forces (attractive, repulsive, fused total)
- Steering function
- Joint-limit checking

Kernels marked *parallel* distribute independent iterations across CPU
threads, which helps when trees grow beyond a few hundred nodes or when
the optimizer has many interior waypoints.

If Numba is not available, equivalent NumPy implementations are used automatically.

## Testing

```bash
python -m pytest tests/ -q --tb=short
```

## Documentation

```bash
cd docs && make html
# Open _build/html/index.html
```

## Obstacle Scenes

- **wall**: Original pillars and balls in front of the arm
- **canopy**: Tight cage of objects surrounding the arm (1-6 cm clearance)
- **passage**: Staggered corridor the arm must weave through

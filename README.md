# pdg — Powered Descent Guidance in C++

A production-quality, **dependency-free C++17 library** for convex-optimization-based
rocket landing guidance — the algorithm family behind Mars powered descent and
propulsive booster recovery — packaged around an embedded-suitable SOCP solver,
with a 6-DoF simulator and a Monte Carlo dispersion runner to close the loop.

```
guidance:   lossless convexification (3-DoF)  +  successive convexification (6-DoF,
            free final time, compound state-triggered constraints)
solver:     primal-dual interior-point SOCP (homogeneous self-dual embedding,
            Nesterov-Todd scaling, sparse LDL^T) — fixed memory, bounded iterations
simulation: 6-DoF rigid-body sim, TVC tracking autopilot, deterministic Monte Carlo
```

## Why

Successive convexification for rocket landing is one of the most-cited GNC topics of
the decade, and the math lineage behind operational propulsive landing. Yet nearly
every public implementation is a Python/MATLAB research script tied to a desktop
solver. `pdg` is a from-scratch C++ implementation of the full stack:

| Module | What it implements | Reference |
|---|---|---|
| `pdg::socp` | SOCP interior-point solver: homogeneous self-dual embedding, Nesterov–Todd scaling, Mehrotra predictor–corrector, sparse LDLᵀ with static+dynamic regularization and iterative refinement. Symbolic factorization computed once; `solve()` is allocation-free with a bounded iteration count. | Domahidi et al., *ECOS: An SOCP solver for embedded systems*, ECC 2013 |
| `pdg::lcvx` | 3-DoF minimum-fuel powered descent via **lossless convexification** (the G-FOLD lineage): nonconvex thrust lower bound made convex exactly; free final time by golden-section search reusing one symbolic factorization. | Açıkmeşe & Ploen, JGCD 2007; Blackmore et al., JGCD 2010 |
| `pdg::scvx` | 6-DoF **successive convexification** with free final time (time dilation), first-order-hold exact-STM discretization, virtual control, and a hard trust region with nonlinear-penalty ratio test. | Szmuk & Açıkmeşe, [arXiv:1802.03827](https://arxiv.org/abs/1802.03827); Mao et al. (SCvx) |
| `pdg::scvx` STCs | **(Compound) state-triggered constraints** `g(x,u) < 0 ⇒ c(x,u) ≤ 0` via the `−min(g,0)·c ≤ 0` formulation, with AND/OR trigger composition; built-in speed-triggered angle-of-attack limit. | Szmuk, Reynolds & Açıkmeşe, JGCD 2020; Uzun, Açıkmeşe & Carson, [arXiv:2510.09610](https://arxiv.org/abs/2510.09610) |
| `pdg::sim` | Minimal 6-DoF rigid-body simulator (RK4) with dispersed plant — thrust scale, Isp error, engine misalignment, first-order actuator lag — plus a launcher-style TVC tracking autopilot (attitude + gimbal feedforward, PD loops). | — |
| `pdg::mc` | Deterministic, multithreaded Monte Carlo dispersion runner with per-sample SCvx re-guidance, counter-based RNG (bit-identical results for any thread count), CSV output and touchdown statistics. | — |

No Eigen, no BLAS, no external solver — the whole stack is ~6k lines of C++17 with
only the standard library.

## Quick start

```bash
git clone https://github.com/Krataios14/pdg.git
cd pdg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build            # 5 suites, all algorithm layers
```

### 3-DoF Mars landing (lossless convexification)

```cpp
#include <pdg/lcvx.hpp>

pdg::LcvxParams P;                 // defaults: classic Mars descent case
pdg::LcvxPlanner planner(P);
pdg::LcvxSolution sol;
planner.solveFreeTf(40.0, 100.0, sol);   // golden-section over time of flight
// sol.r / sol.v / sol.thrust / sol.mass per node; sol.fuelUsed; sol.tf
```

Each fixed-`tf` problem is one SOCP (~17 ms at N=55 on a laptop); the free-final-time
search reuses the same symbolic KKT factorization for every evaluation. The
relaxation is *lossless*: `‖u‖ = σ` holds at optimum (asserted in the test suite).

### 6-DoF landing with state-triggered constraints (SCvx)

```cpp
#include <pdg/scvx.hpp>

pdg::ScvxParams P;                 // Falcon-like single-stick, T/W ≈ 1.7
P.stcs.push_back(pdg::makeSpeedTriggeredAoA(/*vTrigger=*/40.0, /*aoaMaxDeg=*/35.0));
pdg::ScvxPlanner planner(P);
pdg::ScvxSolution sol;
planner.solve(sol);                // converges in ~25 iterations, ~0.5 s
// sol.x: 14-state trajectory [m, r, v, q, ω]; sol.u: body thrust; sol.tf free
```

The converged trajectory is dynamically consistent to ~1e-8 (single-shooting defect,
checked nonlinearly every iteration) and satisfies tilt, glide-slope, rate, thrust,
gimbal and state-triggered constraints at the nodes.

### Monte Carlo, closed loop

```cpp
#include <pdg/mc.hpp>

pdg::ScvxParams nominal;
pdg::MonteCarloRunner mc(nominal, pdg::McDispersions{}, pdg::McConfig{});
pdg::McResults res = mc.run();     // re-guides each dispersed sample, flies it
                                   // closed-loop on a dispersed plant
```

Representative output of `ex_monte_carlo 100` (initial-state dispersion + 2% thrust,
1% Isp, 0.2° engine cant, 50 ms actuator lag):

```
guidance converged : 100        lateral error : mean 0.27 m, p95 0.57 m, max 0.81 m
landed             : 100        descent speed : mean 1.63 m/s, max 4.69 m/s
success            : 95%        tilt          : mean 1.6 deg,  max 5.6 deg
```

(~10 s wall time for 100 samples: full SCvx guidance + 6-DoF sim per sample.)

## Design notes for embedded use

* **Fixed memory** — `SocpSolver::setup()` performs all allocation (symbolic
  factorization, workspace); `solve()` allocates nothing. `workspaceBytes()`
  reports the footprint.
* **Bounded, deterministic timing** — fixed sparsity pattern, fixed elimination
  tree, hard iteration caps at every level (IPM iterations, refinement sweeps,
  SCvx iterations, line-search evaluations).
* **No exceptions, no RTTI requirements in the hot path** — status codes
  (`Optimal / PrimalInfeasible / DualInfeasible / MaxIterations / NumericalError`)
  with certified infeasibility detection via the homogeneous embedding.
* **Numerical robustness** — static + dynamic LDLᵀ regularization with iterative
  refinement against the unregularized KKT (the ECOS recipe), RCM ordering with
  dense-column deferral (free-final-time columns), NT scaling in product form.
* The SCvx subproblem keeps an **identical sparsity pattern across iterations**, so
  the symbolic factorization is computed exactly once per planner instance.

## Repository layout

```
include/pdg/    linalg.hpp socp.hpp rocket.hpp lcvx.hpp scvx.hpp sim.hpp mc.hpp
src/            implementations
tests/          unit + integration tests (self-contained harness)
examples/       ex_3dof_mars, ex_6dof_landing, ex_monte_carlo (CSV output)
```

## References

1. B. Açıkmeşe, S. R. Ploen, "Convex Programming Approach to Powered Descent
   Guidance for Mars Landing", *JGCD* 30(5), 2007.
2. L. Blackmore, B. Açıkmeşe, D. P. Scharf, "Minimum-Landing-Error Powered-Descent
   Guidance for Mars Landing Using Convex Optimization", *JGCD* 33(4), 2010.
3. M. Szmuk, B. Açıkmeşe, "Successive Convexification for 6-DoF Mars Rocket Powered
   Landing with Free-Final-Time", AIAA GNC 2018, arXiv:1802.03827.
4. M. Szmuk, T. P. Reynolds, B. Açıkmeşe, "Successive Convexification for Real-Time
   Six-Degree-of-Freedom Powered Descent Guidance with State-Triggered Constraints",
   *JGCD* 43(8), 2020.
5. S. Uzun, B. Açıkmeşe, J. M. Carson III, "Sequential Convex Programming for 6-DoF
   Powered Descent Guidance with Continuous-Time Compound State-Triggered
   Constraints", arXiv:2510.09610, 2025.
6. A. Domahidi, E. Chu, S. Boyd, "ECOS: An SOCP Solver for Embedded Systems",
   *ECC* 2013.
7. Y. Mao, M. Szmuk, B. Açıkmeşe, "Successive Convexification of Non-Convex Optimal
   Control Problems and Its Convergence Properties", *CDC* 2016.
8. L. Vandenberghe, "The CVXOPT Linear and Quadratic Cone Program Solvers", 2010.

## License

MIT — see [LICENSE](LICENSE).

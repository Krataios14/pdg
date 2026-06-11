#include "pdg/lcvx.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace pdg {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

LcvxPlanner::LcvxPlanner(const LcvxParams& params) : prm_(params), N_(params.N) {
    assert(N_ >= 3);
    assert(prm_.Tmax > prm_.Tmin && prm_.Tmin >= 0.0);
    assert(prm_.mWet > prm_.mDry && prm_.mDry > 0.0);
}

// ---------------------------------------------------------------------------
// SOCP assembly for a given time of flight. The triplet EMISSION ORDER is
// deterministic and tf-independent, so the sparsity pattern (and thus the
// symbolic factorization inside the solver) is identical for every tf.
// ---------------------------------------------------------------------------
void LcvxPlanner::assemble(double tf) {
    const int N = N_;
    const double dt = tf / (N - 1);
    const double alpha = 1.0 / (prm_.Isp * prm_.g0);

    prob_ = SocpProblem();
    prob_.n = 11 * N;   // per node: r(3) v(3) z(1) | u(3) sigma(1)
    prob_.c.assign(prob_.n, 0.0);
    // minimize -z_N  (maximize final mass)
    prob_.c[idxZ(N - 1)] = -1.0;

    Triplets& A = prob_.A;
    Vec& b = prob_.b;
    auto addEq = [&](double rhs) { b.push_back(rhs); return static_cast<int>(b.size()) - 1; };

    // ---- dynamics (exact discretization, first-order-hold on u and sigma)
    //  r+ = r + dt v + dt^2/2 g + dt^2 (u/3 + u+/6)
    //  v+ = v + dt g + dt/2 (u + u+)
    //  z+ = z - alpha dt/2 (sigma + sigma+)
    for (int k = 0; k + 1 < N; ++k) {
        for (int ax = 0; ax < 3; ++ax) {
            int row = addEq(0.5 * dt * dt * prm_.g[ax]);
            A.add(row, idxR(k + 1, ax), 1.0);
            A.add(row, idxR(k, ax), -1.0);
            A.add(row, idxV(k, ax), -dt);
            A.add(row, idxU(k, ax), -dt * dt / 3.0);
            A.add(row, idxU(k + 1, ax), -dt * dt / 6.0);
        }
        for (int ax = 0; ax < 3; ++ax) {
            int row = addEq(dt * prm_.g[ax]);
            A.add(row, idxV(k + 1, ax), 1.0);
            A.add(row, idxV(k, ax), -1.0);
            A.add(row, idxU(k, ax), -0.5 * dt);
            A.add(row, idxU(k + 1, ax), -0.5 * dt);
        }
        {
            int row = addEq(0.0);
            A.add(row, idxZ(k + 1), 1.0);
            A.add(row, idxZ(k), -1.0);
            A.add(row, idxS(k), 0.5 * alpha * dt);
            A.add(row, idxS(k + 1), 0.5 * alpha * dt);
        }
    }

    // ---- boundary conditions
    for (int ax = 0; ax < 3; ++ax) {
        int row = addEq(prm_.r0[ax]);
        A.add(row, idxR(0, ax), 1.0);
    }
    for (int ax = 0; ax < 3; ++ax) {
        int row = addEq(prm_.v0[ax]);
        A.add(row, idxV(0, ax), 1.0);
    }
    {
        int row = addEq(std::log(prm_.mWet));
        A.add(row, idxZ(0), 1.0);
    }
    for (int ax = 0; ax < 3; ++ax) {
        int row = addEq(prm_.rf[ax]);
        A.add(row, idxR(N - 1, ax), 1.0);
    }
    for (int ax = 0; ax < 3; ++ax) {
        int row = addEq(prm_.vf[ax]);
        A.add(row, idxV(N - 1, ax), 1.0);
    }

    // ---- cone constraints. LP rows first, then SOC blocks (solver convention).
    Triplets& G = prob_.G;
    Vec& h = prob_.h;
    auto addRow = [&](double rhs) { h.push_back(rhs); return static_cast<int>(h.size()) - 1; };

    const double cosPoint = std::cos(prm_.pointingDeg * kPi / 180.0);
    const bool usePointing = prm_.pointingDeg > 0.0 && prm_.pointingDeg < 90.0;
    const bool useGs = prm_.glideSlopeDeg > 0.0;
    const double cotGs = useGs ? 1.0 / std::tan(prm_.glideSlopeDeg * kPi / 180.0) : 0.0;
    const bool useVmax = prm_.vMax > 0.0;
    const bool useLower = prm_.Tmin > 0.0;

    // reference mass profile: z0 = max-thrust depletion (lower bound on z),
    // zUp = min-thrust depletion (upper bound on z)
    Vec z0v(N), zUpv(N);
    for (int k = 0; k < N; ++k) {
        const double tk = dt * k;
        z0v[k]  = std::log(std::max(prm_.mDry, prm_.mWet - alpha * prm_.Tmax * tk));
        zUpv[k] = std::log(std::max(prm_.mDry, prm_.mWet - alpha * prm_.Tmin * tk));
    }

    // LP block
    for (int k = 0; k < N; ++k) {
        const double e0 = std::exp(-z0v[k]);
        // sigma <= Tmax e^{-z0}(1 - (z - z0)):  sigma + Tmax e0 z <= Tmax e0 (1 + z0)
        {
            int row = addRow(prm_.Tmax * e0 * (1.0 + z0v[k]));
            G.add(row, idxS(k), 1.0);
            G.add(row, idxZ(k), prm_.Tmax * e0);
        }
        // z >= z0 :  -z <= -z0
        {
            int row = addRow(-z0v[k]);
            G.add(row, idxZ(k), -1.0);
        }
        // z <= zUp
        {
            int row = addRow(zUpv[k]);
            G.add(row, idxZ(k), 1.0);
        }
        // thrust pointing: cos(theta) sigma - u_z <= 0
        if (usePointing) {
            int row = addRow(0.0);
            G.add(row, idxS(k), cosPoint);
            G.add(row, idxU(k, 2), -1.0);
        }
    }
    // final mass above dry: -z_N <= -ln(mDry)
    {
        int row = addRow(-std::log(prm_.mDry));
        G.add(row, idxZ(N - 1), -1.0);
    }
    prob_.cone.l = static_cast<int>(h.size());

    // SOC blocks
    for (int k = 0; k < N; ++k) {
        // ||u|| <= sigma : s = (sigma; u) in Q^4
        {
            int row = addRow(0.0);
            G.add(row, idxS(k), -1.0);
            for (int ax = 0; ax < 3; ++ax) {
                row = addRow(0.0);
                G.add(row, idxU(k, ax), -1.0);
            }
            prob_.cone.q.push_back(4);
        }
        // lower thrust bound (convex quadratic):
        //   sigma >= a (1 - dz + dz^2/2),  a = Tmin e^{-z0},  dz = z - z0
        //   <=>  dz^2 <= 2 t  with affine  t = sigma/a - 1 + dz
        //   SOC^3:  || (dz ; t - 1/2) || <= t + 1/2
        if (useLower) {
            const double a = prm_.Tmin * std::exp(-z0v[k]);
            // s = h - G x with:
            // s0 = t + 1/2 = sigma/a + z - z0 - 1/2
            int row = addRow(-z0v[k] - 0.5);
            G.add(row, idxS(k), -1.0 / a);
            G.add(row, idxZ(k), -1.0);
            // s1 = dz = z - z0
            row = addRow(-z0v[k]);
            G.add(row, idxZ(k), -1.0);
            // s2 = t - 1/2 = sigma/a + z - z0 - 3/2
            row = addRow(-z0v[k] - 1.5);
            G.add(row, idxS(k), -1.0 / a);
            G.add(row, idxZ(k), -1.0);
            prob_.cone.q.push_back(3);
        }
        // glide slope: ||r_xy - rf_xy|| <= cot(gs) (r_z - rf_z) : Q^3
        if (useGs) {
            int row = addRow(-cotGs * prm_.rf[2]);
            G.add(row, idxR(k, 2), -cotGs);
            for (int ax = 0; ax < 2; ++ax) {
                row = addRow(-prm_.rf[ax]);
                G.add(row, idxR(k, ax), -1.0);
            }
            prob_.cone.q.push_back(3);
        }
        // speed limit: ||v|| <= vMax : Q^4
        if (useVmax) {
            addRow(prm_.vMax);
            for (int ax = 0; ax < 3; ++ax) {
                int row = addRow(0.0);
                G.add(row, idxV(k, ax), -1.0);
            }
            prob_.cone.q.push_back(4);
        }
    }
}

void LcvxPlanner::extract(double tf, LcvxSolution& out) {
    const int N = N_;
    const Vec& x = solver_.x();
    out.tf = tf;
    out.t.resize(N);
    out.r.resize(N);
    out.v.resize(N);
    out.u.resize(N);
    out.sigma.resize(N);
    out.mass.resize(N);
    out.thrust.resize(N);
    const double dt = tf / (N - 1);
    for (int k = 0; k < N; ++k) {
        out.t[k] = dt * k;
        for (int ax = 0; ax < 3; ++ax) {
            out.r[k][ax] = x[idxR(k, ax)];
            out.v[k][ax] = x[idxV(k, ax)];
            out.u[k][ax] = x[idxU(k, ax)];
        }
        out.sigma[k] = x[idxS(k)];
        out.mass[k] = std::exp(x[idxZ(k)]);
        for (int ax = 0; ax < 3; ++ax) out.thrust[k][ax] = out.mass[k] * out.u[k][ax];
    }
    out.fuelUsed = prm_.mWet - out.mass[N - 1];
    out.socpInfo = solver_.info();
}

bool LcvxPlanner::solveFixedTf(double tf, LcvxSolution& out) {
    assemble(tf);
    if (!solverReady_) {
        solver_.setup(prob_, prm_.socp);
        solverReady_ = true;
    }
    const SocpStatus st = solver_.solve(prob_);
    out.feasible = (st == SocpStatus::Optimal);
    if (out.feasible) extract(tf, out);
    else out.socpInfo = solver_.info();
    return out.feasible;
}

bool LcvxPlanner::solveFreeTf(double tfMin, double tfMax, LcvxSolution& out, int maxEvals) {
    // golden-section search on fuel(tf); infeasible tf treated as +inf
    const double invPhi = 0.6180339887498949;
    int evals = 0;
    LcvxSolution tmp;

    auto fuelAt = [&](double tf) -> double {
        ++evals;
        if (!solveFixedTf(tf, tmp)) return 1e30;
        return tmp.fuelUsed;
    };

    double a = tfMin, c = tfMax;
    double x1 = c - invPhi * (c - a);
    double x2 = a + invPhi * (c - a);
    double f1 = fuelAt(x1);
    LcvxSolution best1 = tmp;
    double f2 = fuelAt(x2);
    LcvxSolution best2 = tmp;

    while (evals < maxEvals && (c - a) > 1e-2 * (tfMax - tfMin)) {
        if (f1 <= f2) {
            c = x2; x2 = x1; f2 = f1; best2 = best1;
            x1 = c - invPhi * (c - a);
            f1 = fuelAt(x1);
            best1 = tmp;
        } else {
            a = x1; x1 = x2; f1 = f2; best1 = best2;
            x2 = a + invPhi * (c - a);
            f2 = fuelAt(x2);
            best2 = tmp;
        }
    }
    out = (f1 <= f2) ? best1 : best2;
    return out.feasible;
}

}  // namespace pdg

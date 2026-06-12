#include "pdg/scvx.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace pdg {

namespace {
constexpr double kPi = 3.14159265358979323846;

// derivative of C(q)*e applied to gradient computations: returns d(C(q)u)/dq as 3x4
void dCuDq(const double* q, const double* u, double out[3][4]) {
    const double q0 = q[0];
    const double* qv = q + 1;
    double qvxu[3] = {qv[1] * u[2] - qv[2] * u[1],
                      qv[2] * u[0] - qv[0] * u[2],
                      qv[0] * u[1] - qv[1] * u[0]};
    const double qvu = qv[0] * u[0] + qv[1] * u[1] + qv[2] * u[2];
    const double ux[3][3] = {{0, -u[2], u[1]}, {u[2], 0, -u[0]}, {-u[1], u[0], 0}};
    for (int i = 0; i < 3; ++i) {
        out[i][0] = 2.0 * (q0 * u[i] + qvxu[i]);
        for (int j = 0; j < 3; ++j) {
            double dij = -u[i] * qv[j] + qv[i] * u[j] - q0 * ux[i][j];
            if (i == j) dij += qvu;
            out[i][1 + j] = 2.0 * dij;
        }
    }
}
}  // namespace

// ---------------------------------------------------------------------------
// built-in STC: speed-triggered angle-of-attack limit
// ---------------------------------------------------------------------------
StateTriggeredConstraint makeSpeedTriggeredAoA(double vTrigger, double aoaMaxDeg) {
    StateTriggeredConstraint stc;
    stc.name = "speed-triggered-aoa";
    stc.triggers.push_back([vTrigger](const double* x, const double*, double* gx, double* gu) {
        const double* v = x + iV;
        const double vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (gx) {
            for (int i = 0; i < kNx; ++i) gx[i] = 0.0;
            if (vn > 1e-9)
                for (int i = 0; i < 3; ++i) gx[iV + i] = -v[i] / vn;
        }
        if (gu)
            for (int j = 0; j < kNu; ++j) gu[j] = 0.0;
        return vTrigger - vn;     // fires (g < 0) when speed exceeds vTrigger
    });
    const double cosA = std::cos(aoaMaxDeg * kPi / 180.0);
    stc.constraint = [cosA](const double* x, const double*, double* gx, double* gu) {
        // c = v . zB + cos(aoaMax) ||v|| <= 0  with zB = C(q) e3
        const double* v = x + iV;
        const double* q = x + iQ;
        const double vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double e3[3] = {0.0, 0.0, 1.0};
        double zB[3];
        quatRotate(q, e3, zB);
        const double c = v[0] * zB[0] + v[1] * zB[1] + v[2] * zB[2] + cosA * vn;
        if (gx) {
            for (int i = 0; i < kNx; ++i) gx[i] = 0.0;
            for (int i = 0; i < 3; ++i) {
                gx[iV + i] = zB[i];
                if (vn > 1e-9) gx[iV + i] += cosA * v[i] / vn;
            }
            double dC[3][4];
            dCuDq(q, e3, dC);
            for (int j = 0; j < 4; ++j)
                gx[iQ + j] = v[0] * dC[0][j] + v[1] * dC[1][j] + v[2] * dC[2][j];
        }
        if (gu)
            for (int j = 0; j < kNu; ++j) gu[j] = 0.0;
        return c;
    };
    return stc;
}

// ---------------------------------------------------------------------------
// planner
// ---------------------------------------------------------------------------
ScvxPlanner::ScvxPlanner(const ScvxParams& params) : prm_(params), K_(params.K) {
    assert(K_ >= 4);
    nStc_ = static_cast<int>(prm_.stcs.size());

    // scaling: chosen so all decision variables are O(1)
    const double rMag = std::max({std::fabs(prm_.r0[0]), std::fabs(prm_.r0[1]),
                                  std::fabs(prm_.r0[2]), 100.0});
    const double vMag = std::max({std::fabs(prm_.v0[0]), std::fabs(prm_.v0[1]),
                                  std::fabs(prm_.v0[2]), 10.0});
    sx_[iM] = prm_.rocket.mWet;
    for (int i = 0; i < 3; ++i) sx_[iR + i] = rMag;
    for (int i = 0; i < 3; ++i) sx_[iV + i] = vMag;
    for (int i = 0; i < 4; ++i) sx_[iQ + i] = 1.0;
    const double wMax = std::max(prm_.omegaMaxDegS * kPi / 180.0, 0.1);
    for (int i = 0; i < 3; ++i) sx_[iW + i] = wMax;
    for (int i = 0; i < 3; ++i) su_[i] = prm_.rocket.Tmax;
    ssig_ = std::max(prm_.tfGuess, 1.0);

    Ad_.assign(K_ - 1, Mat(kNx, kNx));
    Bm_.assign(K_ - 1, Mat(kNx, kNu));
    Bp_.assign(K_ - 1, Mat(kNx, kNu));
    Sd_.assign(K_ - 1, State{});
    zd_.assign(K_ - 1, State{});
    xRef_.assign(K_, State{});
    uRef_.assign(K_, Vec3{});
    xCand_.assign(K_, State{});
    uCand_.assign(K_, Vec3{});
}

void ScvxPlanner::initReference() {
    const RocketParams& R = prm_.rocket;
    const Vec4 qUp = {1.0, 0.0, 0.0, 0.0};
    const double mEnd = R.mWet - 0.4 * (R.mWet - R.mDry);
    for (int k = 0; k < K_; ++k) {
        const double a = static_cast<double>(k) / (K_ - 1);
        State& x = xRef_[k];
        x[iM] = (1.0 - a) * R.mWet + a * mEnd;
        for (int i = 0; i < 3; ++i) {
            x[iR + i] = (1.0 - a) * prm_.r0[i] + a * prm_.rf[i];
            x[iV + i] = (1.0 - a) * prm_.v0[i] + a * prm_.vf[i];
            x[iW + i] = 0.0;
        }
        double q[4];
        quatSlerp(prm_.q0.data(), qUp.data(), a, q);
        for (int i = 0; i < 4; ++i) x[iQ + i] = q[i];

        // hover-ish thrust: u_B = C(q)' (-m g)
        double negG[3] = {-x[iM] * R.g[0], -x[iM] * R.g[1], -x[iM] * R.g[2]};
        // rotate inertial->body with conjugate quaternion
        double qc[4] = {q[0], -q[1], -q[2], -q[3]};
        double uB[3];
        quatRotate(qc, negG, uB);
        double un = std::sqrt(uB[0] * uB[0] + uB[1] * uB[1] + uB[2] * uB[2]);
        const double clamped = std::min(std::max(un, R.Tmin), R.Tmax);
        if (un > 1e-9)
            for (int i = 0; i < 3; ++i) uB[i] *= clamped / un;
        else
            uB[2] = clamped;
        for (int i = 0; i < 3; ++i) uRef_[k][i] = uB[i];
    }
    sigRef_ = prm_.tfGuess;
}

// ---------------------------------------------------------------------------
// discretization: integrate STM + FOH control sensitivities over each interval
// ---------------------------------------------------------------------------
namespace {
struct AugState {
    // packed: x (kNx) | Phi (kNx*kNx) | Pm (kNx*kNu) | Pp (kNx*kNu) | Ps (kNx)
    static constexpr int kSize = kNx + kNx * kNx + 2 * kNx * kNu + kNx;
    double v[kSize];
};
}  // namespace

void ScvxPlanner::discretize() {
    const double dtau = 1.0 / (K_ - 1);
    const int nSub = std::max(2, prm_.rk4Substeps);
    const double hs = dtau / nSub;

    Mat A(kNx, kNx), B(kNx, kNu), Phi(kNx, kNx), PhiLU(kNx, kNx), RHS(kNx, kNu * 2 + 1);
    double f[kNx];

    for (int k = 0; k + 1 < K_; ++k) {
        AugState V;
        // init: x = xRef_[k], Phi = I, P* = 0
        for (int i = 0; i < AugState::kSize; ++i) V.v[i] = 0.0;
        for (int i = 0; i < kNx; ++i) V.v[i] = xRef_[k][i];
        for (int i = 0; i < kNx; ++i) V.v[kNx + i * kNx + i] = 1.0;

        auto deriv = [&](double s, const double* Vv, double* dV) {
            const double* x = Vv;
            const double lm = 1.0 - s / dtau;
            const double lp = s / dtau;
            double u[3];
            for (int j = 0; j < 3; ++j)
                u[j] = lm * uRef_[k][j] + lp * uRef_[k + 1][j];

            rocketDynamics(prm_.rocket, x, u, f);
            rocketJacobians(prm_.rocket, x, u, A, B);

            // xdot = sigma f
            for (int i = 0; i < kNx; ++i) dV[i] = sigRef_ * f[i];
            // Phidot = sigma A Phi
            const double* Pv = Vv + kNx;
            double* dPhi = dV + kNx;
            for (int i = 0; i < kNx; ++i)
                for (int j = 0; j < kNx; ++j) {
                    double s2 = 0.0;
                    for (int l = 0; l < kNx; ++l) s2 += A(i, l) * Pv[l * kNx + j];
                    dPhi[i * kNx + j] = sigRef_ * s2;
                }
            // P*dot = Phi^{-1} [sigma lm B | sigma lp B | f]
            for (int i = 0; i < kNx; ++i)
                for (int j = 0; j < kNx; ++j) PhiLU(i, j) = Pv[i * kNx + j];
            for (int i = 0; i < kNx; ++i) {
                for (int j = 0; j < kNu; ++j) {
                    RHS(i, j) = sigRef_ * lm * B(i, j);
                    RHS(i, kNu + j) = sigRef_ * lp * B(i, j);
                }
                RHS(i, 2 * kNu) = f[i];
            }
            const bool ok = luSolve(PhiLU, RHS);
            (void)ok;
            double* dPm = dV + kNx + kNx * kNx;
            double* dPp = dPm + kNx * kNu;
            double* dPs = dPp + kNx * kNu;
            for (int i = 0; i < kNx; ++i) {
                for (int j = 0; j < kNu; ++j) {
                    dPm[i * kNu + j] = RHS(i, j);
                    dPp[i * kNu + j] = RHS(i, kNu + j);
                }
                dPs[i] = RHS(i, 2 * kNu);
            }
        };

        // RK4
        AugState k1, k2, k3, k4, tmp;
        double s = 0.0;
        for (int step = 0; step < nSub; ++step) {
            deriv(s, V.v, k1.v);
            for (int i = 0; i < AugState::kSize; ++i) tmp.v[i] = V.v[i] + 0.5 * hs * k1.v[i];
            deriv(s + 0.5 * hs, tmp.v, k2.v);
            for (int i = 0; i < AugState::kSize; ++i) tmp.v[i] = V.v[i] + 0.5 * hs * k2.v[i];
            deriv(s + 0.5 * hs, tmp.v, k3.v);
            for (int i = 0; i < AugState::kSize; ++i) tmp.v[i] = V.v[i] + hs * k3.v[i];
            deriv(s + hs, tmp.v, k4.v);
            for (int i = 0; i < AugState::kSize; ++i)
                V.v[i] += hs / 6.0 * (k1.v[i] + 2.0 * k2.v[i] + 2.0 * k3.v[i] + k4.v[i]);
            s += hs;
        }

        // unpack
        Mat& Ak = Ad_[k];
        for (int i = 0; i < kNx; ++i)
            for (int j = 0; j < kNx; ++j) Ak(i, j) = V.v[kNx + i * kNx + j];
        const double* Pm = V.v + kNx + kNx * kNx;
        const double* Pp = Pm + kNx * kNu;
        const double* Ps = Pp + kNx * kNu;
        // B = Phi(end) * P
        for (int i = 0; i < kNx; ++i)
            for (int j = 0; j < kNu; ++j) {
                double sm = 0.0, sp = 0.0;
                for (int l = 0; l < kNx; ++l) {
                    sm += Ak(i, l) * Pm[l * kNu + j];
                    sp += Ak(i, l) * Pp[l * kNu + j];
                }
                Bm_[k](i, j) = sm;
                Bp_[k](i, j) = sp;
            }
        for (int i = 0; i < kNx; ++i) {
            double ss = 0.0;
            for (int l = 0; l < kNx; ++l) ss += Ak(i, l) * Ps[l];
            Sd_[k][i] = ss;
        }
        // defect-free affine term: z = xProp - A x_k - Bm u_k - Bp u_{k+1} - S sigma
        for (int i = 0; i < kNx; ++i) {
            double ax = 0.0;
            for (int l = 0; l < kNx; ++l) ax += Ak(i, l) * xRef_[k][l];
            double bu = 0.0;
            for (int j = 0; j < kNu; ++j)
                bu += Bm_[k](i, j) * uRef_[k][j] + Bp_[k](i, j) * uRef_[k + 1][j];
            zd_[k][i] = V.v[i] - ax - bu - Sd_[k][i] * sigRef_;
        }
    }
}

// ---------------------------------------------------------------------------
// SOCP subproblem assembly (in scaled variables; fixed sparsity pattern)
// ---------------------------------------------------------------------------
void ScvxPlanner::assemble(double trustRadius) {
    const RocketParams& R = prm_.rocket;
    prob_ = SocpProblem();
    prob_.n = numVars();
    prob_.c.assign(prob_.n, 0.0);

    // cost: min -m_N/mWet + wNu ||nu||_1 + wNu sum(stcSlack)
    prob_.c[idxX(K_ - 1, iM)] = -1.0;
    for (int k = 0; k + 1 < K_; ++k)
        for (int i = 0; i < kNx; ++i) prob_.c[idxNuAbs(k, i)] = prm_.wVirtual;
    for (int k = 0; k < K_; ++k)
        for (int s = 0; s < nStc_; ++s) prob_.c[idxStcSlack(k, s)] = prm_.wVirtual;

    Triplets& A = prob_.A;
    Vec& b = prob_.b;
    auto addEq = [&](double rhs) { b.push_back(rhs); return static_cast<int>(b.size()) - 1; };

    // ---- dynamics: x^_{k+1} = A^ x^_k + Bm^ u^_k + Bp^ u^_{k+1} + S^ sig^ + z^ + nu_k
    for (int k = 0; k + 1 < K_; ++k) {
        for (int i = 0; i < kNx; ++i) {
            const double rs = 1.0 / sx_[i];
            int row = addEq(zd_[k][i] * rs);
            A.add(row, idxX(k + 1, i), 1.0);
            for (int j = 0; j < kNx; ++j)
                A.add(row, idxX(k, j), -Ad_[k](i, j) * sx_[j] * rs);
            for (int j = 0; j < kNu; ++j) {
                A.add(row, idxU(k, j), -Bm_[k](i, j) * su_[j] * rs);
                A.add(row, idxU(k + 1, j), -Bp_[k](i, j) * su_[j] * rs);
            }
            A.add(row, idxSig(), -Sd_[k][i] * ssig_ * rs);
            A.add(row, idxNu(k, i), -1.0);
        }
    }

    // ---- boundary conditions (scaled)
    { int r = addEq(1.0); A.add(r, idxX(0, iM), 1.0); }              // m(0) = mWet
    for (int i = 0; i < 3; ++i) {
        int r = addEq(prm_.r0[i] / sx_[iR + i]); A.add(r, idxX(0, iR + i), 1.0);
    }
    for (int i = 0; i < 3; ++i) {
        int r = addEq(prm_.v0[i] / sx_[iV + i]); A.add(r, idxX(0, iV + i), 1.0);
    }
    if (prm_.fixInitialAttitude) {
        for (int i = 0; i < 4; ++i) {
            int r = addEq(prm_.q0[i]); A.add(r, idxX(0, iQ + i), 1.0);
        }
        for (int i = 0; i < 3; ++i) {
            int r = addEq(prm_.w0[i] / sx_[iW + i]); A.add(r, idxX(0, iW + i), 1.0);
        }
    }
    for (int i = 0; i < 3; ++i) {
        int r = addEq(prm_.rf[i] / sx_[iR + i]); A.add(r, idxX(K_ - 1, iR + i), 1.0);
    }
    for (int i = 0; i < 3; ++i) {
        int r = addEq(prm_.vf[i] / sx_[iV + i]); A.add(r, idxX(K_ - 1, iV + i), 1.0);
    }
    if (prm_.fixFinalAttitude) {
        // upright touchdown with the roll degree of freedom left FREE: a single
        // gimbaled engine on the roll axis has no roll authority, so demanding a
        // specific roll angle/rate would make dispersed problems infeasible.
        // q1 = q2 = 0 <=> zero tilt (any roll); wx = wy = 0.
        for (int i = 1; i <= 2; ++i) {
            int r = addEq(0.0); A.add(r, idxX(K_ - 1, iQ + i), 1.0);
        }
        for (int i = 0; i < 2; ++i) {
            int r = addEq(0.0); A.add(r, idxX(K_ - 1, iW + i), 1.0);
        }
    }

    // ---- cone constraints
    Triplets& G = prob_.G;
    Vec& h = prob_.h;
    auto addRow = [&](double rhs) { h.push_back(rhs); return static_cast<int>(h.size()) - 1; };

    // LP rows -------------------------------------------------------------
    for (int k = 0; k < K_; ++k) {
        // mass >= mDry
        { int r = addRow(-R.mDry / R.mWet); G.add(r, idxX(k, iM), -1.0); }
        // min thrust (linearized): nbar' u >= Tmin
        {
            double n[3] = {uRef_[k][0], uRef_[k][1], uRef_[k][2]};
            const double nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (nn > 1e-9) { n[0] /= nn; n[1] /= nn; n[2] /= nn; }
            else           { n[0] = 0.0; n[1] = 0.0; n[2] = 1.0; }
            int r = addRow(-R.Tmin / R.Tmax);
            for (int j = 0; j < 3; ++j) G.add(r, idxU(k, j), -n[j]);
        }
    }
    // sigma bounds
    { int r = addRow(-prm_.tfMin / ssig_); G.add(r, idxSig(), -1.0); }
    { int r = addRow(prm_.tfMax / ssig_); G.add(r, idxSig(), 1.0); }
    // virtual control L1: +-nu <= nuAbs
    for (int k = 0; k + 1 < K_; ++k)
        for (int i = 0; i < kNx; ++i) {
            { int r = addRow(0.0); G.add(r, idxNu(k, i), 1.0); G.add(r, idxNuAbs(k, i), -1.0); }
            { int r = addRow(0.0); G.add(r, idxNu(k, i), -1.0); G.add(r, idxNuAbs(k, i), -1.0); }
        }
    // sigma trust region (hard): |sig^ - sigRef^| <= trustRadius
    {
        const double sref = sigRef_ / ssig_;
        { int r = addRow(sref + trustRadius); G.add(r, idxSig(), 1.0); }
        { int r = addRow(-sref + trustRadius); G.add(r, idxSig(), -1.0); }
    }
    // state-triggered constraints (linearized, with nonneg slack).
    // Exempt at a fully-fixed initial node: the boundary condition is whatever
    // it is, and an STC there would be either redundant or unsatisfiable.
    const int kStc0 = prm_.fixInitialAttitude ? 1 : 0;
    for (int k = 0; k < kStc0; ++k) {
        for (int s = 0; s < nStc_; ++s) {
            int r = addRow(0.0);                  // exempt node: slack >= 0 only
            G.add(r, idxStcSlack(k, s), -1.0);    // (cost drives it to zero)
        }
    }
    for (int k = kStc0; k < K_; ++k) {
        for (int s = 0; s < nStc_; ++s) {
            const StateTriggeredConstraint& stc = prm_.stcs[s];
            const double* x = xRef_[k].data();
            const double* u = uRef_[k].data();

            // effective trigger: AND -> max(g_i), OR -> min(g_i)
            double gEff = (stc.combine == StateTriggeredConstraint::Combine::And)
                              ? -1e300 : 1e300;
            int gIdx = 0;
            std::vector<double> gxBest(kNx, 0.0), guBest(kNu, 0.0);
            for (size_t t = 0; t < stc.triggers.size(); ++t) {
                double gx[kNx] = {0.0}, gu[kNu] = {0.0};
                const double g = stc.triggers[t](x, u, gx, gu);
                const bool take = (stc.combine == StateTriggeredConstraint::Combine::And)
                                      ? (g > gEff) : (g < gEff);
                if (take) {
                    gEff = g;
                    gIdx = static_cast<int>(t);
                    std::copy(gx, gx + kNx, gxBest.begin());
                    std::copy(gu, gu + kNu, guBest.begin());
                }
            }
            (void)gIdx;
            double cx[kNx] = {0.0}, cu[kNu] = {0.0};
            const double cVal = stc.constraint(x, u, cx, cu);

            // h = -min(g,0) c <= 0; grad h = -[g<0] (grad g) c - min(g,0) grad c
            const double gMin = std::min(gEff, 0.0);
            const double act = (gEff < 0.0) ? 1.0 : 0.0;
            double hVal = -gMin * cVal;
            double hx[kNx], hu[kNu];
            for (int i = 0; i < kNx; ++i) hx[i] = -act * gxBest[i] * cVal - gMin * cx[i];
            for (int j = 0; j < kNu; ++j) hu[j] = -act * guBest[j] * cVal - gMin * cu[j];

            // scaled row: sum hx_i sx_i x^_i + sum hu_j su_j u^_j - slack <= rhs
            double rowMax = 1.0;
            for (int i = 0; i < kNx; ++i) rowMax = std::max(rowMax, std::fabs(hx[i] * sx_[i]));
            for (int j = 0; j < kNu; ++j) rowMax = std::max(rowMax, std::fabs(hu[j] * su_[j]));
            const double rs = 1.0 / rowMax;
            double rhs = -hVal;
            for (int i = 0; i < kNx; ++i) rhs += hx[i] * x[i];
            for (int j = 0; j < kNu; ++j) rhs += hu[j] * u[j];
            int r = addRow(rhs * rs);
            for (int i = 0; i < kNx; ++i) G.add(r, idxX(k, i), hx[i] * sx_[i] * rs);
            for (int j = 0; j < kNu; ++j) G.add(r, idxU(k, j), hu[j] * su_[j] * rs);
            G.add(r, idxStcSlack(k, s), -1.0);
            // slack >= 0
            int r2 = addRow(0.0);
            G.add(r2, idxStcSlack(k, s), -1.0);
        }
    }
    prob_.cone.l = static_cast<int>(h.size());

    // SOC blocks ------------------------------------------------------------
    const bool useGs = prm_.glideSlopeDeg > 0.0;
    const double cotGs = useGs ? 1.0 / std::tan(prm_.glideSlopeDeg * kPi / 180.0) : 0.0;
    const double tiltBound = std::sqrt(0.5 * (1.0 - std::cos(prm_.tiltMaxDeg * kPi / 180.0)));
    const double wBound = prm_.omegaMaxDegS * kPi / 180.0 / sx_[iW];
    const double cosGim = std::cos(prm_.gimbalMaxDeg * kPi / 180.0);

    for (int k = 0; k < K_; ++k) {
        // glide slope: || r_xy - rf_xy || <= cot(gs) (r_z - rf_z)   [scaled by sx_r]
        if (useGs) {
            int r = addRow(-cotGs * prm_.rf[2] / sx_[iR + 2]);
            G.add(r, idxX(k, iR + 2), -cotGs);
            for (int i = 0; i < 2; ++i) {
                r = addRow(-prm_.rf[i] / sx_[iR + i]);
                G.add(r, idxX(k, iR + i), -1.0);
            }
            prob_.cone.q.push_back(3);
        }
        // tilt: ||(q1,q2)|| <= sqrt((1-cos tiltMax)/2)
        {
            addRow(tiltBound);
            for (int i = 1; i <= 2; ++i) {
                int r = addRow(0.0);
                G.add(r, idxX(k, iQ + i), -1.0);
            }
            prob_.cone.q.push_back(3);
        }
        // body rate: ||w^|| <= wMax/sw
        {
            addRow(wBound);
            for (int i = 0; i < 3; ++i) {
                int r = addRow(0.0);
                G.add(r, idxX(k, iW + i), -1.0);
            }
            prob_.cone.q.push_back(4);
        }
        // max thrust: ||u^|| <= 1
        {
            addRow(1.0);
            for (int j = 0; j < 3; ++j) {
                int r = addRow(0.0);
                G.add(r, idxU(k, j), -1.0);
            }
            prob_.cone.q.push_back(4);
        }
        // gimbal: ||u^|| <= u^_z / cos(gimbalMax)
        {
            int r = addRow(0.0);
            G.add(r, idxU(k, 2), -1.0 / cosGim);
            for (int j = 0; j < 3; ++j) {
                r = addRow(0.0);
                G.add(r, idxU(k, j), -1.0);
            }
            prob_.cone.q.push_back(4);
        }
        // hard trust region: ||(x^_k - xRef^_k ; u^_k - uRef^_k)|| <= trustRadius
        {
            addRow(trustRadius);
            for (int i = 0; i < kNx; ++i) {
                int r = addRow(-xRef_[k][i] / sx_[i]);
                G.add(r, idxX(k, i), -1.0);
            }
            for (int j = 0; j < kNu; ++j) {
                int r = addRow(-uRef_[k][j] / su_[j]);
                G.add(r, idxU(k, j), -1.0);
            }
            prob_.cone.q.push_back(1 + kNx + kNu);
        }
    }
}

// ---------------------------------------------------------------------------
void ScvxPlanner::extractCandidate(ScvxIterInfo& info) {
    const Vec& xs = solver_.x();
    double dxMax = 0.0, nuMax = 0.0;
    for (int k = 0; k < K_; ++k) {
        for (int i = 0; i < kNx; ++i) {
            xCand_[k][i] = xs[idxX(k, i)] * sx_[i];
            dxMax = std::max(dxMax, std::fabs(xs[idxX(k, i)] - xRef_[k][i] / sx_[i]));
        }
        for (int j = 0; j < kNu; ++j) uCand_[k][j] = xs[idxU(k, j)] * su_[j];
    }
    for (int k = 0; k + 1 < K_; ++k)
        for (int i = 0; i < kNx; ++i)
            nuMax = std::max(nuMax, std::fabs(xs[idxNu(k, i)]));
    sigCand_ = xs[idxSig()] * ssig_;
    dxMax = std::max(dxMax, std::fabs(xs[idxSig()] - sigRef_ / ssig_));

    info.nuMax = nuMax;
    info.dxMax = dxMax;
    info.sigma = sigCand_;
    info.cost = solver_.info().pcost;
}

double ScvxPlanner::nonlinearPenalty(const std::vector<State>& xs, const std::vector<Vec3>& us,
                                     double sig, double* maxDefect) const {
    // J = -m_N/mWet + wVirtual * sum_k sum_i |defect_k,i| / sx_i  (+ STC violations)
    const double dtau = 1.0 / (K_ - 1);
    const int nSub = std::max(2, prm_.rk4Substeps);
    const double hs = dtau / nSub;
    double pen = 0.0, worst = 0.0;
    double f1[kNx], f2[kNx], f3[kNx], f4[kNx], xt[kNx], xp[kNx];
    for (int k = 0; k + 1 < K_; ++k) {
        for (int i = 0; i < kNx; ++i) xp[i] = xs[k][i];
        double s = 0.0;
        auto uAt = [&](double sloc, double* u) {
            const double lp = sloc / dtau, lm = 1.0 - lp;
            for (int j = 0; j < 3; ++j) u[j] = lm * us[k][j] + lp * us[k + 1][j];
        };
        for (int st = 0; st < nSub; ++st) {
            double u[3];
            uAt(s, u);
            rocketDynamics(prm_.rocket, xp, u, f1);
            for (int i = 0; i < kNx; ++i) xt[i] = xp[i] + 0.5 * hs * sig * f1[i];
            uAt(s + 0.5 * hs, u);
            rocketDynamics(prm_.rocket, xt, u, f2);
            for (int i = 0; i < kNx; ++i) xt[i] = xp[i] + 0.5 * hs * sig * f2[i];
            rocketDynamics(prm_.rocket, xt, u, f3);
            for (int i = 0; i < kNx; ++i) xt[i] = xp[i] + hs * sig * f3[i];
            uAt(s + hs, u);
            rocketDynamics(prm_.rocket, xt, u, f4);
            for (int i = 0; i < kNx; ++i)
                xp[i] += hs * sig / 6.0 * (f1[i] + 2.0 * f2[i] + 2.0 * f3[i] + f4[i]);
            s += hs;
        }
        for (int i = 0; i < kNx; ++i) {
            const double d = std::fabs(xp[i] - xs[k + 1][i]) / sx_[i];
            pen += d;
            worst = std::max(worst, d);
        }
    }
    pen *= prm_.wVirtual;

    // STC violations, scaled consistently with the subproblem rows
    const int kStc0 = prm_.fixInitialAttitude ? 1 : 0;
    for (int k = kStc0; k < K_; ++k) {
        for (int sI = 0; sI < nStc_; ++sI) {
            const StateTriggeredConstraint& stc = prm_.stcs[sI];
            const double* x = xs[k].data();
            const double* u = us[k].data();
            double gEff = (stc.combine == StateTriggeredConstraint::Combine::And) ? -1e300 : 1e300;
            double gx[kNx] = {0.0}, gu[kNu] = {0.0};
            for (auto& trig : stc.triggers) {
                const double g = trig(x, u, nullptr, nullptr);
                gEff = (stc.combine == StateTriggeredConstraint::Combine::And)
                           ? std::max(gEff, g) : std::min(gEff, g);
            }
            double cx[kNx] = {0.0}, cu[kNu] = {0.0};
            const double cVal = stc.constraint(x, u, cx, cu);
            const double hVal = -std::min(gEff, 0.0) * cVal;
            if (hVal <= 0.0) continue;
            // same row normalization as assemble()
            const double gMin = std::min(gEff, 0.0);
            const double act = (gEff < 0.0) ? 1.0 : 0.0;
            for (auto& trig : stc.triggers) { trig(x, u, gx, gu); break; }
            double rowMax = 1.0;
            for (int i = 0; i < kNx; ++i)
                rowMax = std::max(rowMax, std::fabs((-act * gx[i] * cVal - gMin * cx[i]) * sx_[i]));
            for (int j = 0; j < kNu; ++j)
                rowMax = std::max(rowMax, std::fabs((-act * gu[j] * cVal - gMin * cu[j]) * su_[j]));
            pen += prm_.wVirtual * hVal / rowMax;
        }
    }

    pen += -xs[K_ - 1][iM] / prm_.rocket.mWet;
    if (maxDefect) *maxDefect = worst;
    return pen;
}

bool ScvxPlanner::solve(ScvxSolution& out) {
    out = ScvxSolution();
    initReference();
    double tr = prm_.trInit;
    double Jref = nonlinearPenalty(xRef_, uRef_, sigRef_);
    bool needDiscretize = true;

    for (int iter = 0; iter < prm_.maxIters; ++iter) {
        if (needDiscretize) {
            discretize();
            needDiscretize = false;
        }
        assemble(tr);
        if (!solverReady_) {
            solver_.setup(prob_, prm_.socp);
            solverReady_ = true;
        }
        const SocpStatus st = solver_.solve(prob_);

        ScvxIterInfo info;
        info.iter = iter;
        info.socpStatus = st;
        info.trustRadius = tr;
        if (st != SocpStatus::Optimal && st != SocpStatus::MaxIterations) {
            // numerical trouble: shrink the trust region and retry
            tr /= prm_.trShrink;
            out.history.push_back(info);
            if (tr < prm_.trMin) break;
            continue;
        }
        extractCandidate(info);
        const double Jlin = info.cost;
        double defect = 0.0;
        const double Jcand = nonlinearPenalty(xCand_, uCand_, sigCand_, &defect);
        info.penalty = Jcand;
        info.maxDefect = defect;

        const double pred = Jref - Jlin;     // >= 0: reference is SOCP-feasible
        const double actual = Jref - Jcand;
        const double rho = (pred > 1e-15) ? actual / pred : 1.0;
        info.rho = rho;

        bool accept = rho >= prm_.rho0;
        info.accepted = accept;
        out.history.push_back(info);

        if (prm_.verbose)
            std::printf("scvx %2d  tr=%8.2e  Jlin=%10.6f  Jnl=%10.6f  rho=%6.2f  tf=%7.3f  "
                        "dx=%8.2e  nu=%8.2e  def=%8.2e  %s\n",
                        iter, tr, Jlin, Jcand, rho, info.sigma, info.dxMax, info.nuMax,
                        defect, accept ? "accept" : "REJECT");

        if (accept) {
            xRef_ = xCand_;
            uRef_ = uCand_;
            sigRef_ = sigCand_;
            Jref = Jcand;
            needDiscretize = true;
            if (rho < prm_.rho1) tr = std::max(tr / prm_.trShrink, prm_.trMin);
            else if (rho >= prm_.rho2) tr = std::min(tr * prm_.trGrow, prm_.trMax);

            const bool feasNow = info.nuMax < prm_.tolNu && defect < 1e-4;
            if ((info.dxMax < prm_.tolDx && feasNow) || (pred < prm_.tolPred && feasNow)) {
                out.converged = true;
                break;
            }
        } else {
            tr /= prm_.trShrink;
            if (tr < prm_.trMin) break;
        }
    }

    if (out.history.empty()) return false;
    double defect = 0.0;
    nonlinearPenalty(xRef_, uRef_, sigRef_, &defect);
    // feasibility of the accepted reference: tiny single-shooting defects
    out.feasible = defect < 1e-3;
    out.tf = sigRef_;
    out.fuelUsed = prm_.rocket.mWet - xRef_[K_ - 1][iM];
    out.t.resize(K_);
    out.x.resize(K_);
    out.u.resize(K_);
    for (int k = 0; k < K_; ++k) {
        out.t[k] = sigRef_ * static_cast<double>(k) / (K_ - 1);
        out.x[k] = xRef_[k];
        quatNormalize(out.x[k].data() + iQ);
        out.u[k] = uRef_[k];
    }
    return out.converged && out.feasible;
}

}  // namespace pdg

#include "pdg/sim.hpp"

#include <algorithm>
#include <cmath>

namespace pdg {

namespace {
constexpr double kPi = 3.14159265358979323846;

Vec3 interpControl(const ScvxSolution& g, double t) {
    const auto& tg = g.t;
    const auto& ug = g.u;
    const int K = static_cast<int>(tg.size());
    if (K == 0) return {0.0, 0.0, 0.0};
    if (t <= tg.front()) return ug.front();
    if (t >= tg.back()) return ug.back();
    int k = 1;
    while (k < K - 1 && tg[k] < t) ++k;
    const double a = (t - tg[k - 1]) / std::max(1e-12, tg[k] - tg[k - 1]);
    Vec3 u;
    for (int j = 0; j < 3; ++j) u[j] = (1.0 - a) * ug[k - 1][j] + a * ug[k][j];
    return u;
}

TouchdownState finalizeTouchdown(const State& x, const State& xPrev, double t, double dt,
                                 double m0, bool landed) {
    TouchdownState td;
    if (!landed) {
        td.landed = false;
        td.time = t;
        td.propellantUsed = m0 - x[iM];
        td.finalState = x;
        return td;
    }
    // interpolate the z=0 crossing for clean touchdown statistics
    const double z0 = xPrev[iR + 2], z1 = x[iR + 2];
    const double a = (z0 > z1) ? z0 / std::max(1e-12, z0 - z1) : 1.0;
    State xtd;
    for (int i = 0; i < kNx; ++i) xtd[i] = xPrev[i] + a * (x[i] - xPrev[i]);
    quatNormalize(xtd.data() + iQ);
    td.landed = true;
    td.time = t - dt + a * dt;
    td.lateralError = std::hypot(xtd[iR + 0], xtd[iR + 1]);
    td.verticalSpeed = -xtd[iV + 2];
    td.lateralSpeed = std::hypot(xtd[iV + 0], xtd[iV + 1]);
    td.tiltDeg =
        std::acos(std::min(1.0, std::max(-1.0, quatTiltCos(xtd.data() + iQ)))) * 180.0 / kPi;
    td.rateDegS = std::sqrt(xtd[iW] * xtd[iW] + xtd[iW + 1] * xtd[iW + 1] +
                            xtd[iW + 2] * xtd[iW + 2]) * 180.0 / kPi;
    td.propellantUsed = m0 - xtd[iM];
    td.finalState = xtd;
    return td;
}
}  // namespace

// ---------------------------------------------------------------------------
// TrajectoryTracker
// ---------------------------------------------------------------------------
TrajectoryTracker::TrajectoryTracker(const RocketParams& nominal, const ScvxSolution& ref,
                                     const TrackingGains& gains)
    : prm_(nominal), ref_(ref), g_(gains) {}

void TrajectoryTracker::refAt(double t, State& xRef, Vec3& uRefB) const {
    const auto& tg = ref_.t;
    const int K = static_cast<int>(tg.size());
    int k = 1;
    double a = 1.0;
    if (K >= 2) {
        if (t <= tg.front()) { k = 1; a = 0.0; }
        else if (t >= tg.back()) { k = K - 1; a = 1.0; }
        else {
            k = 1;
            while (k < K - 1 && tg[k] < t) ++k;
            a = (t - tg[k - 1]) / std::max(1e-12, tg[k] - tg[k - 1]);
        }
    }
    for (int i = 0; i < kNx; ++i)
        xRef[i] = (1.0 - a) * ref_.x[k - 1][i] + a * ref_.x[k][i];
    quatNormalize(xRef.data() + iQ);
    uRefB = interpControl(ref_, t);
}

Vec3 TrajectoryTracker::command(double t, const State& x) const {
    // Structure (standard launcher autopilot):
    //   feedforward = reference gimbal command u_ref(t) and attitude q_ref(t);
    //   translation loop tilts the DESIRED attitude and adjusts thrust magnitude;
    //   attitude loop adds a gimbal increment from the q_ref-relative error.
    // With zero state error this reproduces the open-loop command exactly.
    State xr;
    Vec3 uRefB;
    refAt(t, xr, uRefB);
    const double* qRef = xr.data() + iQ;
    const double m = x[iM];

    // ---- translation correction (inertial), saturated
    Vec3 corr;
    double cn = 0.0;
    for (int i = 0; i < 3; ++i) {
        corr[i] = g_.posP * (xr[iR + i] - x[iR + i]) + g_.posD * (xr[iV + i] - x[iV + i]);
        cn += corr[i] * corr[i];
    }
    cn = std::sqrt(cn);
    if (cn > g_.maxCorrAccel)
        for (int i = 0; i < 3; ++i) corr[i] *= g_.maxCorrAccel / cn;

    Vec3 fRefI;
    if (!ref_.t.empty() && t > ref_.t.back()) {
        // past the end of the reference: hover feedforward (the position/velocity
        // terms keep commanding the terminal descent until touchdown)
        for (int i = 0; i < 3; ++i) fRefI[i] = -m * prm_.g[i];
        double qc0[4] = {qRef[0], -qRef[1], -qRef[2], -qRef[3]};
        quatRotate(qc0, fRefI.data(), uRefB.data());
    } else {
        quatRotate(qRef, uRefB.data(), fRefI.data());
    }
    Vec3 fDesI;
    for (int i = 0; i < 3; ++i) fDesI[i] = fRefI[i] + m * corr[i];

    const double fRefMag = std::max(1e-6, norm2(fRefI.data(), 3));
    double fDesMag = norm2(fDesI.data(), 3);
    fDesMag = std::min(std::max(fDesMag, prm_.Tmin), prm_.Tmax);

    // ---- desired attitude: q_ref composed with the small rotation that takes
    // the reference force direction onto the desired force direction
    double axis[3] = {0.0, 0.0, 0.0};   // rotation vector approx (fRef^ x fDes^)
    {
        const double fd = std::max(1e-6, norm2(fDesI.data(), 3));
        double aHat[3], bHat[3];
        for (int i = 0; i < 3; ++i) {
            aHat[i] = fRefI[i] / fRefMag;
            bHat[i] = fDesI[i] / fd;
        }
        axis[0] = aHat[1] * bHat[2] - aHat[2] * bHat[1];
        axis[1] = aHat[2] * bHat[0] - aHat[0] * bHat[2];
        axis[2] = aHat[0] * bHat[1] - aHat[1] * bHat[0];
    }
    double dq[4] = {1.0, 0.5 * axis[0], 0.5 * axis[1], 0.5 * axis[2]};
    quatNormalize(dq);
    double qDes[4];
    quatMul(dq, qRef, qDes);   // rotate in the inertial frame: q_des = dq (x) q_ref

    // ---- attitude error in the body frame: q_err = q^{-1} (x) q_des
    double qc[4] = {x[iQ + 0], -x[iQ + 1], -x[iQ + 2], -x[iQ + 3]};
    double qe[4];
    quatMul(qc, qDes, qe);
    const double sgn = (qe[0] >= 0.0) ? 1.0 : -1.0;
    const double eX = 2.0 * sgn * qe[1];
    const double eY = 2.0 * sgn * qe[2];

    // ---- PD gimbal increment (+ reference rate feedforward)
    const double ax2 = g_.attP * eX + g_.attD * (xr[iW + 0] - x[iW + 0]);
    const double ay2 = g_.attP * eY + g_.attD * (xr[iW + 1] - x[iW + 1]);
    const double L = std::fabs(prm_.rT[2]) > 1e-9 ? std::fabs(prm_.rT[2]) : 1.0;
    // tau = (L u_y, -L u_x, 0)  =>  u_y += Jx ax / L ; u_x -= Jy ay / L
    double ux = uRefB[0] * (fDesMag / fRefMag) - prm_.J[1] * ay2 / L;
    double uy = uRefB[1] * (fDesMag / fRefMag) + prm_.J[0] * ax2 / L;

    // gimbal envelope: keep the lateral component below ~half the magnitude
    const double maxLat = 0.5 * fDesMag;
    const double latN = std::hypot(ux, uy);
    if (latN > maxLat) {
        ux *= maxLat / latN;
        uy *= maxLat / latN;
    }
    const double uz = std::sqrt(std::max(fDesMag * fDesMag - ux * ux - uy * uy,
                                         0.25 * fDesMag * fDesMag));
    return {ux, uy, uz};
}

// ---------------------------------------------------------------------------
// RocketSim
// ---------------------------------------------------------------------------
RocketSim::RocketSim(const RocketParams& nominal, const PlantDispersions& d)
    : plant_(nominal), disp_(d), misalign_(3, 3) {
    plant_.Isp *= d.ispScale;

    // engine misalignment: rotate commanded thrust by a fixed small cant
    const double a = d.thrustMisalignRad;
    const double az = d.misalignAzimuthRad;
    // axis in body x-y plane, perpendicular to the cant direction
    const double ax[3] = {-std::sin(az), std::cos(az), 0.0};
    const double c = std::cos(a), s = std::sin(a);
    misalign_.setIdentity();
    if (a != 0.0) {
        // Rodrigues formula
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const double kk = ax[i] * ax[j];
                const double K[3][3] = {{0, -ax[2], ax[1]}, {ax[2], 0, -ax[0]}, {-ax[1], ax[0], 0}};
                misalign_(i, j) = c * (i == j ? 1.0 : 0.0) + s * K[i][j] + (1.0 - c) * kk;
            }
    }
}

void RocketSim::step(State& x, const Vec3& cmd0, const Vec3& cmdMid, const Vec3& cmd1,
                     Vec3& uAct, double dt) const {
    // apply misalignment + thrust scale to the commands
    auto actual = [&](const Vec3& c) {
        Vec3 u;
        matVec(misalign_, c.data(), u.data());
        for (int j = 0; j < 3; ++j) u[j] *= disp_.thrustScale;
        // clamp to the physical engine limit
        const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if (un > plant_.Tmax)
            for (int j = 0; j < 3; ++j) u[j] *= plant_.Tmax / un;
        return u;
    };

    Vec3 u0, um, u1;
    if (disp_.actuatorTau > 1e-6) {
        // first-order lag toward the (mid-step) target; lag dominates the
        // in-step command variation, so a single lagged value per stage set
        u0 = uAct;
        const Vec3 tgt = actual(cmdMid);
        const double aMid = 1.0 - std::exp(-0.5 * dt / disp_.actuatorTau);
        const double aEnd = 1.0 - std::exp(-dt / disp_.actuatorTau);
        for (int j = 0; j < 3; ++j) {
            um[j] = uAct[j] + aMid * (tgt[j] - uAct[j]);
            u1[j] = uAct[j] + aEnd * (tgt[j] - uAct[j]);
        }
        uAct = u1;
    } else {
        u0 = actual(cmd0);
        um = actual(cmdMid);
        u1 = actual(cmd1);
        uAct = u1;
    }

    // RK4 with stage-correct control sampling
    double f1[kNx], f2[kNx], f3[kNx], f4[kNx];
    State xt;
    rocketDynamics(plant_, x.data(), u0.data(), f1);
    for (int i = 0; i < kNx; ++i) xt[i] = x[i] + 0.5 * dt * f1[i];
    rocketDynamics(plant_, xt.data(), um.data(), f2);
    for (int i = 0; i < kNx; ++i) xt[i] = x[i] + 0.5 * dt * f2[i];
    rocketDynamics(plant_, xt.data(), um.data(), f3);
    for (int i = 0; i < kNx; ++i) xt[i] = x[i] + dt * f3[i];
    rocketDynamics(plant_, xt.data(), u1.data(), f4);
    for (int i = 0; i < kNx; ++i)
        x[i] += dt / 6.0 * (f1[i] + 2.0 * f2[i] + 2.0 * f3[i] + f4[i]);

    // mass floor (engine flames out at dry mass)
    if (x[iM] < plant_.mDry) x[iM] = plant_.mDry;
    quatNormalize(x.data() + iQ);
}

TouchdownState RocketSim::fly(const State& x0,
                              const std::function<Vec3(double, const State&)>& command,
                              double dt, double tMax) {
    State x = x0;
    const Vec3 u0 = command(0.0, x);
    Vec3 uAct;
    matVec(misalign_, u0.data(), uAct.data());    // start the lag at the first command
    for (int j = 0; j < 3; ++j) uAct[j] *= disp_.thrustScale;

    const double m0 = x[iM];
    double t = 0.0;
    State xPrev = x;
    bool landed = false;
    while (t < tMax) {
        xPrev = x;
        // controller sampled at the step start (zero-order hold across the step,
        // like a discrete autopilot running at 1/dt Hz)
        const Vec3 cmd = command(t, x);
        step(x, cmd, cmd, cmd, uAct, dt);
        t += dt;
        if (x[iR + 2] <= 0.0) {
            landed = true;
            break;
        }
    }
    return finalizeTouchdown(x, xPrev, t, dt, m0, landed);
}

TouchdownState RocketSim::flyTrajectory(const State& x0, const ScvxSolution& guidance,
                                        double dt, double tMax) {
    // open loop: the FOH command profile is followed regardless of the state.
    // Exact-control-time sampling matters here, so bypass the ZOH controller path.
    State x = x0;
    const Vec3 u0 = interpControl(guidance, 0.0);
    Vec3 uAct;
    matVec(misalign_, u0.data(), uAct.data());
    for (int j = 0; j < 3; ++j) uAct[j] *= disp_.thrustScale;

    const double m0 = x[iM];
    double t = 0.0;
    State xPrev = x;
    bool landed = false;
    while (t < tMax) {
        xPrev = x;
        step(x, interpControl(guidance, t), interpControl(guidance, t + 0.5 * dt),
             interpControl(guidance, t + dt), uAct, dt);
        t += dt;
        if (x[iR + 2] <= 0.0) {
            landed = true;
            break;
        }
    }
    return finalizeTouchdown(x, xPrev, t, dt, m0, landed);
}

TouchdownState RocketSim::flyTracked(const State& x0, const TrajectoryTracker& tracker,
                                     double dt, double tMax) {
    return fly(x0, [&](double t, const State& x) { return tracker.command(t, x); }, dt, tMax);
}

}  // namespace pdg

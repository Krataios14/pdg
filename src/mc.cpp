#include "pdg/mc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <thread>

namespace pdg {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

// ---------------------------------------------------------------------------
// NormalSampler
// ---------------------------------------------------------------------------
NormalSampler::NormalSampler(uint64_t seed, uint64_t stream) {
    // mix seed and stream so nearby ids decorrelate
    s_ = seed ^ (stream * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    next();
    next();
}

uint64_t NormalSampler::next() {
    // splitmix64
    s_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double NormalSampler::uniform() {
    return (next() >> 11) * (1.0 / 9007199254740992.0);  // 53-bit mantissa in [0,1)
}

double NormalSampler::normal() {
    if (have_) {
        have_ = false;
        return spare_;
    }
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    const double r = std::sqrt(-2.0 * std::log(u1));
    spare_ = r * std::sin(2.0 * kPi * u2);
    have_ = true;
    return r * std::cos(2.0 * kPi * u2);
}

// ---------------------------------------------------------------------------
// MonteCarloRunner
// ---------------------------------------------------------------------------
MonteCarloRunner::MonteCarloRunner(const ScvxParams& nominal, const McDispersions& disp,
                                   const McConfig& cfg)
    : nominal_(nominal), disp_(disp), cfg_(cfg) {}

McSample MonteCarloRunner::runSample(int id) const {
    NormalSampler rng(cfg_.seed, static_cast<uint64_t>(id));
    McSample out;
    out.id = id;

    // ---- dispersed initial state
    ScvxParams P = nominal_;
    for (int i = 0; i < 3; ++i) {
        P.r0[i] += disp_.r0Sigma[i] * rng.normal();
        P.v0[i] += disp_.v0Sigma[i] * rng.normal();
    }
    // keep dispersed start above the glide-slope floor implied by lateral offset
    P.r0[2] = std::max(P.r0[2], 50.0);
    // initial attitude: tilt about a random horizontal axis
    {
        const double tilt = disp_.attitudeSigmaDeg * kPi / 180.0 * rng.normal();
        const double az = 2.0 * kPi * rng.uniform();
        const double axn[3] = {std::cos(az), std::sin(az), 0.0};
        const double half = 0.5 * tilt;
        double dq[4] = {std::cos(half), std::sin(half) * axn[0], std::sin(half) * axn[1], 0.0};
        double q[4];
        quatMul(dq, nominal_.q0.data(), q);
        quatNormalize(q);
        for (int i = 0; i < 4; ++i) P.q0[i] = q[i];
    }
    for (int i = 0; i < 3; ++i)
        P.w0[i] += disp_.rateSigmaDegS * kPi / 180.0 * rng.normal();
    const double dm = disp_.massSigma * rng.normal();
    P.rocket.mWet = std::max(nominal_.rocket.mDry * 1.05, nominal_.rocket.mWet + dm);
    const RocketParams plantParams = P.rocket;   // full physical engine envelope
    P.rocket.Tmax = nominal_.rocket.Tmax * (1.0 - cfg_.guidanceThrustMargin);

    // ---- dispersed plant (NOT known to guidance)
    PlantDispersions pd;
    pd.thrustScale = 1.0 + disp_.thrustScaleSigma * rng.normal();
    pd.ispScale = 1.0 + disp_.ispScaleSigma * rng.normal();
    pd.thrustMisalignRad = std::fabs(disp_.misalignSigmaDeg * kPi / 180.0 * rng.normal());
    pd.misalignAzimuthRad = 2.0 * kPi * rng.uniform();
    pd.actuatorTau = disp_.actuatorTau;

    // ---- guidance
    const ScvxSolution* traj = &nominalSol_;
    ScvxSolution own;
    if (cfg_.reguide) {
        const auto t0 = std::chrono::steady_clock::now();
        ScvxPlanner planner(P);
        out.guidanceConverged = planner.solve(own);
        out.guidanceTimeMs = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count();
        traj = &own;
        if (!out.guidanceConverged && own.x.empty()) return out;
    } else {
        out.guidanceConverged = nominalReady_;
    }
    out.plannedTf = traj->tf;
    out.plannedFuel = traj->fuelUsed;

    // ---- fly it on the dispersed plant
    State x0{};
    x0[iM] = P.rocket.mWet;
    for (int i = 0; i < 3; ++i) x0[iR + i] = P.r0[i];
    for (int i = 0; i < 3; ++i) x0[iV + i] = P.v0[i];
    for (int i = 0; i < 4; ++i) x0[iQ + i] = P.q0[i];
    for (int i = 0; i < 3; ++i) x0[iW + i] = P.w0[i];

    RocketSim sim(plantParams, pd);
    const double tMax = std::max(60.0, 3.0 * traj->tf);
    if (cfg_.track) {
        TrajectoryTracker tracker(plantParams, *traj, cfg_.gains);
        out.td = sim.flyTracked(x0, tracker, cfg_.simDt, tMax);
    } else {
        out.td = sim.flyTrajectory(x0, *traj, cfg_.simDt, tMax);
    }

    out.success = out.td.landed &&
                  out.td.lateralError <= cfg_.maxLateralError &&
                  out.td.verticalSpeed <= cfg_.maxVerticalSpeed &&
                  out.td.lateralSpeed <= cfg_.maxLateralSpeed &&
                  out.td.tiltDeg <= cfg_.maxTiltDeg;
    return out;
}

McResults MonteCarloRunner::run() {
    McResults res;
    res.samples.resize(cfg_.numSamples);

    if (!cfg_.reguide && !nominalReady_) {
        ScvxPlanner planner(nominal_);
        nominalReady_ = planner.solve(nominalSol_);
    }

    int nThreads = cfg_.threads > 0 ? cfg_.threads
                                    : static_cast<int>(std::thread::hardware_concurrency());
    nThreads = std::max(1, std::min(nThreads, cfg_.numSamples));

    std::mutex printMutex;
    std::vector<std::thread> pool;
    std::vector<int> counter(1, 0);
    std::mutex counterMutex;
    auto worker = [&]() {
        for (;;) {
            int id;
            {
                std::lock_guard<std::mutex> lk(counterMutex);
                if (counter[0] >= cfg_.numSamples) return;
                id = counter[0]++;
            }
            res.samples[id] = runSample(id);
            if (cfg_.verbose) {
                std::lock_guard<std::mutex> lk(printMutex);
                const McSample& s = res.samples[id];
                std::printf("mc %4d: guid=%d landed=%d ok=%d latErr=%7.2f m  vSpd=%5.2f m/s "
                            "tilt=%5.2f deg  fuel=%7.1f kg\n",
                            s.id, (int)s.guidanceConverged, (int)s.td.landed, (int)s.success,
                            s.td.lateralError, s.td.verticalSpeed, s.td.tiltDeg,
                            s.td.propellantUsed);
            }
        }
    };
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    // ---- statistics
    McSummary& s = res.summary;
    s.numSamples = cfg_.numSamples;
    std::vector<double> latErrs;
    double latSum = 0, latSq = 0, vSum = 0, tiltSum = 0, fuelSum = 0;
    for (const McSample& m : res.samples) {
        if (m.guidanceConverged) ++s.numGuidanceConverged;
        if (m.td.landed) {
            ++s.numLanded;
            latErrs.push_back(m.td.lateralError);
            latSum += m.td.lateralError;
            latSq += m.td.lateralError * m.td.lateralError;
            vSum += m.td.verticalSpeed;
            tiltSum += m.td.tiltDeg;
            fuelSum += m.td.propellantUsed;
            s.latErrMax = std::max(s.latErrMax, m.td.lateralError);
            s.vSpeedMax = std::max(s.vSpeedMax, m.td.verticalSpeed);
            s.tiltMaxDeg = std::max(s.tiltMaxDeg, m.td.tiltDeg);
            s.fuelMax = std::max(s.fuelMax, m.td.propellantUsed);
        }
        if (m.success) ++s.numSuccess;
    }
    if (s.numLanded > 0) {
        const double n = s.numLanded;
        s.latErrMean = latSum / n;
        s.latErrStd = std::sqrt(std::max(0.0, latSq / n - s.latErrMean * s.latErrMean));
        s.vSpeedMean = vSum / n;
        s.tiltMeanDeg = tiltSum / n;
        s.fuelMean = fuelSum / n;
        std::sort(latErrs.begin(), latErrs.end());
        s.latErrP95 = latErrs[static_cast<size_t>(0.95 * (latErrs.size() - 1))];
    }
    s.successRate = static_cast<double>(s.numSuccess) / std::max(1, s.numSamples);

    // ---- CSV
    if (!cfg_.csvPath.empty()) {
        std::ofstream f(cfg_.csvPath);
        f << "id,guidance_converged,landed,success,touchdown_time_s,lateral_error_m,"
             "vertical_speed_ms,lateral_speed_ms,tilt_deg,rate_degs,fuel_kg,planned_tf_s,"
             "planned_fuel_kg,guidance_time_ms\n";
        for (const McSample& m : res.samples) {
            f << m.id << ',' << m.guidanceConverged << ',' << m.td.landed << ',' << m.success
              << ',' << m.td.time << ',' << m.td.lateralError << ',' << m.td.verticalSpeed
              << ',' << m.td.lateralSpeed << ',' << m.td.tiltDeg << ',' << m.td.rateDegS << ','
              << m.td.propellantUsed << ',' << m.plannedTf << ',' << m.plannedFuel << ','
              << m.guidanceTimeMs << '\n';
        }
    }
    return res;
}

}  // namespace pdg

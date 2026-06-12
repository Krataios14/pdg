// pdg::mc — Monte Carlo dispersion runner closing the loop between guidance
// (pdg::scvx) and the 6-DoF simulator (pdg::sim).
//
// For each sample, the initial state and plant parameters are dispersed with
// the given 1-sigma values. In `reguide` mode (the realistic configuration),
// SCvx guidance is re-solved from each sample's dispersed initial state — the
// onboard computer knows where it is — and the resulting trajectory is flown
// on the dispersed plant (whose thrust/Isp/misalignment errors guidance does
// NOT know). In open-loop mode the nominal trajectory is flown from the
// dispersed state, which demonstrates dispersion growth without guidance.
//
// Determinism: each sample's random draws come from a counter-based seed
// (seed, sampleId), so results are bit-identical regardless of thread count.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pdg/scvx.hpp"
#include "pdg/sim.hpp"

namespace pdg {

struct McDispersions {
    // initial state (1-sigma, Gaussian)
    Vec3 r0Sigma = {20.0, 20.0, 30.0};      // [m]
    Vec3 v0Sigma = {2.0, 2.0, 3.0};         // [m/s]
    double attitudeSigmaDeg = 2.0;          // tilt about a random horizontal axis
    double rateSigmaDegS = 1.0;             // per-axis body rate
    double massSigma = 100.0;               // wet mass [kg]
    // plant (unknown to guidance)
    double thrustScaleSigma = 0.02;
    double ispScaleSigma = 0.01;
    double misalignSigmaDeg = 0.2;          // engine cant
    double actuatorTau = 0.05;              // deterministic actuator lag [s]
};

struct McConfig {
    int numSamples = 100;
    uint64_t seed = 1234567;
    int threads = 0;            // 0 = hardware concurrency
    bool reguide = true;        // re-solve guidance from each dispersed state
    bool track = true;          // close the TVC tracking loop (false = open loop)
    // Guidance plans with this fraction of max thrust held back, so the tracker
    // retains control authority when the plant underperforms (min-fuel optimal
    // trajectories otherwise end the braking burn thrust-saturated).
    double guidanceThrustMargin = 0.10;
    TrackingGains gains;
    double simDt = 0.01;        // simulator step [s]
    std::string csvPath;        // per-sample CSV output ("" = none)
    // touchdown success criteria (typical landing-gear class limits)
    double maxLateralError = 5.0;   // [m]
    double maxVerticalSpeed = 4.0;  // [m/s]
    double maxLateralSpeed = 2.0;   // [m/s]
    double maxTiltDeg = 7.0;
    bool verbose = false;
};

struct McSample {
    int id = 0;
    bool guidanceConverged = false;
    bool success = false;
    TouchdownState td;
    double plannedTf = 0.0;
    double plannedFuel = 0.0;
    double guidanceTimeMs = 0.0;
};

struct McSummary {
    int numSamples = 0;
    int numGuidanceConverged = 0;
    int numLanded = 0;
    int numSuccess = 0;
    double successRate = 0.0;
    // statistics of landed samples
    double latErrMean = 0.0, latErrStd = 0.0, latErrMax = 0.0, latErrP95 = 0.0;
    double vSpeedMean = 0.0, vSpeedMax = 0.0;
    double tiltMeanDeg = 0.0, tiltMaxDeg = 0.0;
    double fuelMean = 0.0, fuelMax = 0.0;
};

struct McResults {
    std::vector<McSample> samples;
    McSummary summary;
};

class MonteCarloRunner {
public:
    MonteCarloRunner(const ScvxParams& nominal, const McDispersions& disp, const McConfig& cfg);

    // Runs all samples (multithreaded, deterministic) and computes statistics.
    McResults run();

private:
    ScvxParams nominal_;
    McDispersions disp_;
    McConfig cfg_;
    ScvxSolution nominalSol_;   // for open-loop mode
    bool nominalReady_ = false;

    McSample runSample(int id) const;
};

// deterministic counter-based normal sampler (splitmix64 + Box-Muller)
class NormalSampler {
public:
    NormalSampler(uint64_t seed, uint64_t stream);
    double normal();            // ~ N(0,1)
    double uniform();           // ~ U(0,1)

private:
    uint64_t s_;
    bool have_ = false;
    double spare_ = 0.0;
    uint64_t next();
};

}  // namespace pdg

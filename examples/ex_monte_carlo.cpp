// Monte Carlo dispersion analysis: per-sample SCvx re-guidance from dispersed
// initial conditions, flown closed loop (TVC tracking autopilot) on a dispersed
// plant. Writes per-sample results to ex_monte_carlo.csv.
//
// Usage: ex_monte_carlo [numSamples] [seed]
#include <cstdio>
#include <cstdlib>

#include "pdg/mc.hpp"

int main(int argc, char** argv) {
    pdg::ScvxParams nominal;

    pdg::McDispersions disp;            // defaults: see pdg/mc.hpp
    pdg::McConfig cfg;
    cfg.numSamples = (argc > 1) ? std::atoi(argv[1]) : 100;
    cfg.seed = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 1234567;
    cfg.reguide = true;
    cfg.track = true;
    cfg.csvPath = "ex_monte_carlo.csv";
    cfg.verbose = true;

    pdg::MonteCarloRunner mc(nominal, disp, cfg);
    pdg::McResults res = mc.run();

    const pdg::McSummary& s = res.summary;
    std::printf("\nMonte Carlo summary (%d samples)\n", s.numSamples);
    std::printf("  guidance converged : %d\n", s.numGuidanceConverged);
    std::printf("  landed             : %d\n", s.numLanded);
    std::printf("  success            : %d  (%.1f%%)\n", s.numSuccess, 100.0 * s.successRate);
    std::printf("  lateral error      : mean %.2f m, std %.2f m, p95 %.2f m, max %.2f m\n",
                s.latErrMean, s.latErrStd, s.latErrP95, s.latErrMax);
    std::printf("  descent speed      : mean %.2f m/s, max %.2f m/s\n", s.vSpeedMean, s.vSpeedMax);
    std::printf("  tilt at touchdown  : mean %.2f deg, max %.2f deg\n", s.tiltMeanDeg, s.tiltMaxDeg);
    std::printf("  fuel               : mean %.1f kg, max %.1f kg\n", s.fuelMean, s.fuelMax);
    std::printf("per-sample results written to %s\n", cfg.csvPath.c_str());
    return s.successRate > 0.9 ? 0 : 1;
}

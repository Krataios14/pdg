#include "pdg/mc.hpp"
#include "pdg/sim.hpp"
#include "test_framework.hpp"

#include <cmath>

using namespace pdg;

TEST(nominal_plant_flies_guidance_to_target) {
    ScvxParams P;
    ScvxPlanner planner(P);
    ScvxSolution sol;
    CHECK(planner.solve(sol));
    if (sol.x.empty()) return;

    State x0{};
    x0[iM] = P.rocket.mWet;
    for (int i = 0; i < 3; ++i) x0[iR + i] = P.r0[i];
    for (int i = 0; i < 3; ++i) x0[iV + i] = P.v0[i];
    for (int i = 0; i < 4; ++i) x0[iQ + i] = P.q0[i];
    for (int i = 0; i < 3; ++i) x0[iW + i] = P.w0[i];

    RocketSim sim(P.rocket);            // no dispersions
    TouchdownState td = sim.flyTrajectory(x0, sol, 0.005);
    CHECK(td.landed);
    // open-loop replay on the exact plant should land very close to target
    CHECK(td.lateralError < 1.0);
    CHECK(td.verticalSpeed < 2.0);
    CHECK(td.tiltDeg < 2.0);
    CHECK_NEAR(td.time, sol.tf, 0.5);
    CHECK_NEAR(td.propellantUsed, sol.fuelUsed, 30.0);
}

TEST(normal_sampler_deterministic) {
    NormalSampler a(42, 7), b(42, 7), c(42, 8);
    bool same = true, diff = false;
    for (int i = 0; i < 100; ++i) {
        const double va = a.normal(), vb = b.normal(), vc = c.normal();
        same = same && (va == vb);
        diff = diff || (va != vc);
    }
    CHECK(same);
    CHECK(diff);

    // crude sanity on moments
    NormalSampler s(99, 1);
    double sum = 0, sq = 0;
    const int N = 20000;
    for (int i = 0; i < N; ++i) {
        const double v = s.normal();
        sum += v;
        sq += v * v;
    }
    CHECK(std::fabs(sum / N) < 0.05);
    CHECK(std::fabs(sq / N - 1.0) < 0.05);
}

TEST(monte_carlo_small_run) {
    ScvxParams P;
    McDispersions D;
    McConfig C;
    C.numSamples = 6;
    C.threads = 3;
    C.seed = 2024;
    C.reguide = true;
    C.simDt = 0.02;

    MonteCarloRunner mc(P, D, C);
    McResults r1 = mc.run();
    CHECK(static_cast<int>(r1.samples.size()) == 6);
    CHECK(r1.summary.numGuidanceConverged >= 5);   // dispersed cases should mostly converge
    CHECK(r1.summary.numLanded >= 5);
    CHECK(r1.summary.numSuccess >= 4);

    // determinism across thread counts
    McConfig C2 = C;
    C2.threads = 1;
    MonteCarloRunner mc2(P, D, C2);
    McResults r2 = mc2.run();
    for (int i = 0; i < 6; ++i) {
        CHECK(r1.samples[i].td.landed == r2.samples[i].td.landed);
        CHECK_NEAR(r1.samples[i].td.lateralError, r2.samples[i].td.lateralError, 1e-12);
        CHECK_NEAR(r1.samples[i].td.propellantUsed, r2.samples[i].td.propellantUsed, 1e-12);
    }
}

TEST_MAIN()

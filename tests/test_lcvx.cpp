#include "pdg/lcvx.hpp"
#include "test_framework.hpp"

#include <cmath>

using namespace pdg;

namespace {
constexpr double kPi = 3.14159265358979323846;
double norm3(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
}  // namespace

TEST(mars_landing_fixed_tf) {
    LcvxParams P;            // defaults = classic Mars powered-descent case
    LcvxPlanner planner(P);
    LcvxSolution sol;
    CHECK(planner.solveFixedTf(75.0, sol));
    if (!sol.feasible) return;

    // boundary conditions
    CHECK(norm3({sol.r[0][0] - P.r0[0], sol.r[0][1] - P.r0[1], sol.r[0][2] - P.r0[2]}) < 1e-3);
    const int N = P.N;
    CHECK(norm3(sol.r[N - 1]) < 1e-2);
    CHECK(norm3(sol.v[N - 1]) < 1e-3);

    // fuel sanity: positive, less than available
    CHECK(sol.fuelUsed > 0.0);
    CHECK(sol.fuelUsed < P.mWet - P.mDry);

    // LOSSLESS convexification: ||u|| == sigma at every node
    double worst = 0.0;
    for (int k = 0; k < N; ++k) {
        const double nu = norm3(sol.u[k]);
        worst = std::max(worst, std::fabs(nu - sol.sigma[k]) / std::max(1e-9, sol.sigma[k]));
    }
    CHECK(worst < 1e-4);

    // thrust magnitude within (slightly relaxed) physical bounds at every node
    for (int k = 0; k < N; ++k) {
        const double T = norm3(sol.thrust[k]);
        CHECK(T <= P.Tmax * 1.01);
        CHECK(T >= P.Tmin * 0.97);   // small slack: bound is enforced on linearized mass
    }

    // glide slope satisfied
    const double cot = 1.0 / std::tan(P.glideSlopeDeg * kPi / 180.0);
    for (int k = 0; k < N; ++k) {
        const double lat = std::hypot(sol.r[k][0], sol.r[k][1]);
        CHECK(lat <= cot * sol.r[k][2] + 1e-3);
    }

    // mass physically consistent: m_{k+1} >= mDry, decreasing
    for (int k = 0; k + 1 < N; ++k) CHECK(sol.mass[k + 1] <= sol.mass[k] + 1e-9);
    CHECK(sol.mass[N - 1] >= P.mDry - 1e-6);
}

TEST(mars_landing_free_tf) {
    LcvxParams P;
    LcvxPlanner planner(P);
    LcvxSolution sol;
    CHECK(planner.solveFreeTf(40.0, 100.0, sol));
    if (!sol.feasible) return;
    CHECK(sol.tf > 40.0);
    CHECK(sol.tf < 100.0);

    // free-tf fuel should be no worse than two arbitrary fixed times
    LcvxSolution s60, s90;
    LcvxPlanner p2(P);
    const bool f60 = p2.solveFixedTf(60.0, s60);
    LcvxPlanner p3(P);
    const bool f90 = p3.solveFixedTf(90.0, s90);
    if (f60) CHECK(sol.fuelUsed <= s60.fuelUsed + 1.0);
    if (f90) CHECK(sol.fuelUsed <= s90.fuelUsed + 1.0);
}

TEST(infeasible_when_tf_too_short) {
    LcvxParams P;
    LcvxPlanner planner(P);
    LcvxSolution sol;
    // 5 seconds is dynamically impossible from 2.4 km altitude
    CHECK(!planner.solveFixedTf(5.0, sol));
}

TEST(pointing_constraint_respected) {
    LcvxParams P;
    P.pointingDeg = 30.0;
    LcvxPlanner planner(P);
    LcvxSolution sol;
    if (!planner.solveFixedTf(80.0, sol)) {
        CHECK(false);
        return;
    }
    const double cosMax = std::cos(30.0 * kPi / 180.0);
    for (int k = 0; k < P.N; ++k) {
        const double nu = norm3(sol.u[k]);
        if (nu > 1e-6) CHECK(sol.u[k][2] >= cosMax * nu - 1e-5);
    }
}

TEST_MAIN()

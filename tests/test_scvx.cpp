#include "pdg/scvx.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <random>

using namespace pdg;

namespace {
constexpr double kPi = 3.14159265358979323846;

State randomState(std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    State x{};
    x[iM] = 25000.0 + 2000.0 * U(rng);
    for (int i = 0; i < 3; ++i) x[iR + i] = 300.0 * U(rng);
    for (int i = 0; i < 3; ++i) x[iV + i] = 40.0 * U(rng);
    double q[4] = {1.0 + 0.2 * U(rng), 0.2 * U(rng), 0.2 * U(rng), 0.2 * U(rng)};
    quatNormalize(q);
    for (int i = 0; i < 4; ++i) x[iQ + i] = q[i];
    for (int i = 0; i < 3; ++i) x[iW + i] = 0.3 * U(rng);
    return x;
}
}  // namespace

TEST(jacobians_match_finite_differences) {
    RocketParams P;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    for (int trial = 0; trial < 4; ++trial) {
        State x = randomState(rng);
        double u[3] = {30e3 * U(rng), 30e3 * U(rng), 180e3 + 40e3 * U(rng)};

        Mat A(kNx, kNx), B(kNx, kNu);
        rocketJacobians(P, x.data(), u, A, B);

        double f0[kNx], f1[kNx];
        rocketDynamics(P, x.data(), u, f0);

        // df/dx
        for (int j = 0; j < kNx; ++j) {
            const double eps = std::max(1e-6, 1e-7 * std::fabs(x[j]));
            State xp = x;
            xp[j] += eps;
            rocketDynamics(P, xp.data(), u, f1);
            for (int i = 0; i < kNx; ++i) {
                const double fd = (f1[i] - f0[i]) / eps;
                const double scale = std::max({1.0, std::fabs(fd), std::fabs(A(i, j))});
                CHECK_NEAR(A(i, j) / scale, fd / scale, 2e-4);
            }
        }
        // df/du
        for (int j = 0; j < kNu; ++j) {
            const double eps = 1.0;
            double up[3] = {u[0], u[1], u[2]};
            up[j] += eps;
            rocketDynamics(P, x.data(), up, f1);
            for (int i = 0; i < kNx; ++i) {
                const double fd = (f1[i] - f0[i]) / eps;
                const double scale = std::max({1.0, std::fabs(fd), std::fabs(B(i, j))});
                CHECK_NEAR(B(i, j) / scale, fd / scale, 2e-4);
            }
        }
    }
}

TEST(quaternion_utilities) {
    // identity rotation
    double q[4] = {1, 0, 0, 0};
    double v[3] = {1, 2, 3}, out[3];
    quatRotate(q, v, out);
    CHECK_NEAR(out[0], 1.0, 1e-14);
    CHECK_NEAR(out[2], 3.0, 1e-14);

    // 90 deg about z: x -> y
    double q90[4] = {std::cos(kPi / 4), 0, 0, std::sin(kPi / 4)};
    double ex[3] = {1, 0, 0};
    quatRotate(q90, ex, out);
    CHECK_NEAR(out[0], 0.0, 1e-12);
    CHECK_NEAR(out[1], 1.0, 1e-12);

    // quatRotate consistent with quatToDcm
    double qr[4] = {0.9, 0.2, -0.3, 0.1};
    quatNormalize(qr);
    Mat C(3, 3);
    quatToDcm(qr, C);
    double byDcm[3];
    matVec(C, v, byDcm);
    quatRotate(qr, v, out);
    for (int i = 0; i < 3; ++i) CHECK_NEAR(out[i], byDcm[i], 1e-12);

    // tilt of identity = 0 deg
    CHECK_NEAR(quatTiltCos(q), 1.0, 1e-14);
}

TEST(six_dof_landing_converges) {
    ScvxParams P;
    P.verbose = false;
    ScvxPlanner planner(P);
    ScvxSolution sol;
    const bool ok = planner.solve(sol);
    CHECK(ok);
    CHECK(sol.converged);
    CHECK(sol.feasible);
    if (!sol.feasible) return;

    const int K = P.K;
    // boundary conditions
    for (int i = 0; i < 3; ++i) {
        CHECK_NEAR(sol.x[0][iR + i], P.r0[i], 0.5);
        CHECK_NEAR(sol.x[K - 1][iR + i], P.rf[i], 0.5);
        CHECK_NEAR(sol.x[K - 1][iV + i], P.vf[i], 0.1);
    }
    // upright, slow rotation at touchdown
    CHECK(quatTiltCos(sol.x[K - 1].data() + iQ) > 0.9999);

    // fuel positive and within budget
    CHECK(sol.fuelUsed > 0.0);
    CHECK(sol.fuelUsed < P.rocket.mWet - P.rocket.mDry);

    // path constraints at the nodes
    const double cosTilt = std::cos(P.tiltMaxDeg * kPi / 180.0);
    const double wMax = P.omegaMaxDegS * kPi / 180.0;
    const double cosGim = std::cos(P.gimbalMaxDeg * kPi / 180.0);
    for (int k = 0; k < K; ++k) {
        const double* x = sol.x[k].data();
        CHECK(x[iM] >= P.rocket.mDry - 1.0);
        CHECK(quatTiltCos(x + iQ) >= cosTilt - 1e-4);
        const double wn = std::sqrt(x[iW] * x[iW] + x[iW + 1] * x[iW + 1] + x[iW + 2] * x[iW + 2]);
        CHECK(wn <= wMax + 1e-4);
        const double Tn = std::sqrt(sol.u[k][0] * sol.u[k][0] + sol.u[k][1] * sol.u[k][1] +
                                    sol.u[k][2] * sol.u[k][2]);
        CHECK(Tn <= P.rocket.Tmax * 1.001);
        CHECK(Tn >= P.rocket.Tmin * 0.98);     // linearized lower bound
        CHECK(sol.u[k][2] >= cosGim * Tn - 1e-3 * P.rocket.Tmax);
    }
    // time of flight within bounds
    CHECK(sol.tf > P.tfMin);
    CHECK(sol.tf < P.tfMax);
    // converged trajectory is dynamically consistent
    CHECK(sol.history.back().maxDefect < 1e-3);
}

TEST(state_triggered_aoa_constraint) {
    ScvxParams P;
    P.stcs.push_back(makeSpeedTriggeredAoA(40.0, 30.0));
    ScvxPlanner planner(P);
    ScvxSolution sol;
    const bool ok = planner.solve(sol);
    CHECK(ok);
    if (!sol.feasible) return;

    // wherever speed > trigger, AoA must respect the limit (the fixed initial
    // node is exempt: its state is a boundary condition)
    const double cosA = std::cos(30.0 * kPi / 180.0);
    for (int k = 1; k < P.K; ++k) {
        const double* x = sol.x[k].data();
        const double* v = x + iV;
        const double vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (vn > 40.0 + 0.5) {
            const double e3[3] = {0, 0, 1};
            double zB[3];
            quatRotate(x + iQ, e3, zB);
            // c = v.zB + cosA ||v|| <= 0  (allow small linearization slack)
            const double c = v[0] * zB[0] + v[1] * zB[1] + v[2] * zB[2] + cosA * vn;
            CHECK(c <= 0.05 * vn);
        }
    }
}

TEST_MAIN()

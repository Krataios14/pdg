#include "pdg/socp.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <random>

using namespace pdg;

namespace {
// check primal-dual optimality of a returned solution to machine-ish precision
void checkOptimality(const SocpProblem& P, const SocpSolver& S, double tol) {
    const SocpInfo& info = S.info();
    CHECK(info.status == SocpStatus::Optimal);
    CHECK(info.presid < tol);
    CHECK(info.dresid < tol);
    CHECK(info.relGap < 1e-5);
    (void)P;
}
}  // namespace

TEST(lp_box) {
    // min -x1 - 2 x2   s.t.  x1 + x2 <= 1, x >= 0      -> x = (0,1), opt = -2
    SocpProblem P;
    P.n = 2;
    P.c = {-1.0, -2.0};
    P.G.add(0, 0, 1.0); P.G.add(0, 1, 1.0);   // x1 + x2 <= 1
    P.G.add(1, 0, -1.0);                       // -x1 <= 0
    P.G.add(2, 1, -1.0);                       // -x2 <= 0
    P.h = {1.0, 0.0, 0.0};
    P.cone.l = 3;

    SocpSolver S(P);
    CHECK(S.solve(P) == SocpStatus::Optimal);
    checkOptimality(P, S, 1e-6);
    CHECK_NEAR(S.info().pcost, -2.0, 1e-6);
    CHECK_NEAR(S.x()[0], 0.0, 1e-5);
    CHECK_NEAR(S.x()[1], 1.0, 1e-5);
}

TEST(lp_with_equalities) {
    // min c'x  s.t. sum(x) = 1, x >= 0  -> picks smallest c
    SocpProblem P;
    P.n = 4;
    P.c = {3.0, 1.0, 2.0, 5.0};
    for (int j = 0; j < 4; ++j) P.A.add(0, j, 1.0);
    P.b = {1.0};
    for (int j = 0; j < 4; ++j) P.G.add(j, j, -1.0);
    P.h = {0.0, 0.0, 0.0, 0.0};
    P.cone.l = 4;

    SocpSolver S(P);
    CHECK(S.solve(P) == SocpStatus::Optimal);
    CHECK_NEAR(S.info().pcost, 1.0, 1e-6);
    CHECK_NEAR(S.x()[1], 1.0, 1e-5);
}

TEST(min_norm_soc) {
    // min t  s.t. ||x|| <= t,  a'x = 1   ->  t* = 1/||a||
    SocpProblem P;
    const double a[3] = {1.0, 2.0, -2.0};   // ||a|| = 3
    P.n = 4;                                 // (t, x1, x2, x3)
    P.c = {1.0, 0.0, 0.0, 0.0};
    for (int j = 0; j < 3; ++j) P.A.add(0, 1 + j, a[j]);
    P.b = {1.0};
    for (int j = 0; j < 4; ++j) P.G.add(j, j, -1.0);  // s = (t,x) in Q^4
    P.h = {0.0, 0.0, 0.0, 0.0};
    P.cone.q = {4};

    SocpSolver S(P);
    CHECK(S.solve(P) == SocpStatus::Optimal);
    checkOptimality(P, S, 1e-6);
    CHECK_NEAR(S.info().pcost, 1.0 / 3.0, 1e-6);
    // x* = a / ||a||^2
    for (int j = 0; j < 3; ++j) CHECK_NEAR(S.x()[1 + j], a[j] / 9.0, 1e-5);
}

TEST(projection_onto_halfspace_soc) {
    // min t s.t. ||x - x0|| <= t,  g'x <= d.  With x0 violating the halfspace,
    // t* = (g'x0 - d)/||g||.
    const double x0[2] = {2.0, 2.0};
    const double g[2] = {1.0, 0.0};
    const double d = 1.0;            // distance = (2-1)/1 = 1
    SocpProblem P;
    P.n = 3;  // (t, x1, x2)
    P.c = {1.0, 0.0, 0.0};
    // SOC: s = (t, x - x0) in Q^3  ->  rows: -t ; -(x - x0)
    P.G.add(0, 0, -1.0);
    P.G.add(1, 1, -1.0);
    P.G.add(2, 2, -1.0);
    P.h = {0.0, -x0[0], -x0[1]};
    // wait: s = h - Gx; want s1 = x1 - x0 -> G row = -x1, h = -x0  ✓
    // linear: g'x <= d
    P.G.add(3, 1, g[0]);
    P.G.add(3, 2, g[1]);
    P.h.push_back(d);
    P.cone.l = 0;
    P.cone.q = {3};
    // reorder: cone ordering is [LP block, then SOCs] — put the LP row FIRST
    // rebuild with LP row at index 0
    SocpProblem P2;
    P2.n = 3;
    P2.c = P.c;
    P2.G.add(0, 1, g[0]);
    P2.G.add(0, 2, g[1]);
    P2.G.add(1, 0, -1.0);
    P2.G.add(2, 1, -1.0);
    P2.G.add(3, 2, -1.0);
    P2.h = {d, 0.0, -x0[0], -x0[1]};
    P2.cone.l = 1;
    P2.cone.q = {3};

    SocpSolver S(P2);
    CHECK(S.solve(P2) == SocpStatus::Optimal);
    CHECK_NEAR(S.info().pcost, 1.0, 1e-5);
    CHECK_NEAR(S.x()[1], 1.0, 1e-4);  // projection x = (1, 2)
    CHECK_NEAR(S.x()[2], 2.0, 1e-4);
}

TEST(primal_infeasible) {
    // x >= 1 and x <= 0 simultaneously
    SocpProblem P;
    P.n = 1;
    P.c = {1.0};
    P.G.add(0, 0, -1.0);   // -x <= -1  i.e. x >= 1
    P.G.add(1, 0, 1.0);    //  x <= 0
    P.h = {-1.0, 0.0};
    P.cone.l = 2;

    SocpSolver S(P);
    CHECK(S.solve(P) == SocpStatus::PrimalInfeasible);
}

TEST(dual_infeasible_unbounded) {
    // min -x  s.t. x >= 0   (unbounded below)
    SocpProblem P;
    P.n = 1;
    P.c = {-1.0};
    P.G.add(0, 0, -1.0);
    P.h = {0.0};
    P.cone.l = 1;

    SocpSolver S(P);
    CHECK(S.solve(P) == SocpStatus::DualInfeasible);
}

TEST(random_socp_self_consistency) {
    // random feasible SOCPs: verify KKT conditions of the returned solution
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int trial = 0; trial < 5; ++trial) {
        const int n = 8, p = 3;
        SocpProblem P;
        P.n = n;
        P.c.resize(n);
        for (auto& v : P.c) v = U(rng);
        // equalities A x = A x_feas
        Vec xf(n);
        for (auto& v : xf) v = U(rng);
        Vec Ax(p, 0.0);
        for (int r = 0; r < p; ++r)
            for (int j = 0; j < n; ++j) {
                const double a = U(rng);
                if (std::fabs(a) < 0.3) continue;
                P.A.add(r, j, a);
                Ax[r] += a * xf[j];
            }
        P.b = Ax;
        // box: -5 <= x_j <= 5 (keeps things bounded), plus one SOC over first 4 vars
        for (int j = 0; j < n; ++j) { P.G.add(j, j, 1.0); P.h.push_back(5.0); }
        for (int j = 0; j < n; ++j) { P.G.add(n + j, j, -1.0); P.h.push_back(5.0); }
        P.cone.l = 2 * n;
        // SOC: ||(x1,x2,x3)|| <= 4 + x0
        int row = 2 * n;
        P.G.add(row, 0, -1.0); P.h.push_back(4.0);
        P.G.add(row + 1, 1, -1.0); P.h.push_back(0.0);
        P.G.add(row + 2, 2, -1.0); P.h.push_back(0.0);
        P.G.add(row + 3, 3, -1.0); P.h.push_back(0.0);
        P.cone.q = {4};

        SocpSolver S(P);
        const SocpStatus st = S.solve(P);
        CHECK(st == SocpStatus::Optimal);
        if (st == SocpStatus::Optimal) {
            CHECK(S.info().presid < 1e-6);
            CHECK(S.info().dresid < 1e-6);
            CHECK(S.info().relGap < 1e-5);
            // strong duality: pcost == dcost
            CHECK_NEAR(S.info().pcost, S.info().dcost, 1e-4 * (1.0 + std::fabs(S.info().pcost)));
        }
    }
}

TEST(warm_pattern_resolve) {
    // same pattern, changed values: setup once, solve twice
    SocpProblem P;
    P.n = 2;
    P.c = {-1.0, -2.0};
    P.G.add(0, 0, 1.0); P.G.add(0, 1, 1.0);
    P.G.add(1, 0, -1.0);
    P.G.add(2, 1, -1.0);
    P.h = {1.0, 0.0, 0.0};
    P.cone.l = 3;

    SocpSolver S(P);
    CHECK(S.solve(P) == SocpStatus::Optimal);
    CHECK_NEAR(S.info().pcost, -2.0, 1e-6);

    P.h[0] = 2.0;  // relax budget -> opt = -4
    CHECK(S.solve(P) == SocpStatus::Optimal);
    CHECK_NEAR(S.info().pcost, -4.0, 1e-6);
}

TEST_MAIN()

// pdg::lcvx — 3-DoF minimum-fuel powered descent guidance via lossless
// convexification, the algorithm lineage behind G-FOLD / Mars powered descent.
//
// The nonconvex minimum-fuel problem (nonconvex because of the lower thrust
// bound 0 < Tmin <= ||T||) is convexified exactly by introducing the slack
// sigma >= ||u||, u = T/m, and the change of variables z = ln m. The relaxation
// is LOSSLESS: at the optimum ||u|| = sigma, so the solution of the SOCP is a
// global optimum of the original problem.
//
// Free final time is handled by a golden-section search on the (empirically
// unimodal) optimal-fuel-vs-tf profile, each evaluation being one SOCP with an
// identical sparsity pattern (so the symbolic factorization is reused).
//
// References:
//   B. Acikmese, S. Ploen, "Convex Programming Approach to Powered Descent
//     Guidance for Mars Landing", JGCD 30(5), 2007.
//   L. Blackmore, B. Acikmese, D. Scharf, "Minimum-Landing-Error Powered-Descent
//     Guidance for Mars Landing Using Convex Optimization", JGCD 33(4), 2010.
#pragma once

#include <array>
#include <vector>

#include "pdg/socp.hpp"

namespace pdg {

using Vec3 = std::array<double, 3>;

struct LcvxParams {
    // vehicle
    double mWet = 1905.0;            // wet mass [kg]
    double mDry = 1505.0;            // dry mass [kg]
    double Isp  = 225.0;             // specific impulse [s]
    double Tmin = 4972.0;            // min total thrust [N] (> 0 makes the problem nonconvex -> LCvx)
    double Tmax = 13260.0;           // max total thrust [N]
    double g0   = 9.80665;           // standard gravity for Isp [m/s^2]

    // environment (z is "up")
    Vec3 g = {0.0, 0.0, -3.7114};    // local gravity [m/s^2]

    // path constraints (set angle <= 0 or speed <= 0 to disable)
    double glideSlopeDeg = 20.0;     // min elevation angle of r(t)-rf from horizontal
    double pointingDeg   = 45.0;     // max thrust tilt from vertical
    double vMax          = 0.0;      // max speed [m/s]

    // boundary conditions
    Vec3 r0 = {450.0, -330.0, 2400.0};
    Vec3 v0 = {-40.0, 10.0, -10.0};
    Vec3 rf = {0.0, 0.0, 0.0};
    Vec3 vf = {0.0, 0.0, 0.0};

    int N = 55;                      // temporal nodes

    SocpSettings socp;
};

struct LcvxSolution {
    bool feasible = false;
    double tf = 0.0;                 // time of flight [s]
    double fuelUsed = 0.0;           // [kg]
    Vec t;                           // node times
    std::vector<Vec3> r, v, u;       // position, velocity, mass-normalized thrust accel
    Vec sigma;                       // thrust-accel slack (== ||u|| at optimum)
    Vec mass;                        // [kg]
    std::vector<Vec3> thrust;        // physical thrust T = m*u [N]
    SocpInfo socpInfo;
};

class LcvxPlanner {
public:
    explicit LcvxPlanner(const LcvxParams& params);

    // Solve the fixed-final-time SOCP. Returns true if Optimal.
    bool solveFixedTf(double tf, LcvxSolution& out);

    // Golden-section search over tf in [tfMin, tfMax] minimizing fuel.
    // maxEvals bounds the number of SOCP solves.
    bool solveFreeTf(double tfMin, double tfMax, LcvxSolution& out, int maxEvals = 40);

    const LcvxParams& params() const { return prm_; }

private:
    LcvxParams prm_;
    SocpProblem prob_;
    SocpSolver solver_;
    bool solverReady_ = false;
    int N_ = 0;

    // variable indices
    int idxR(int k, int axis) const { return 7 * k + axis; }
    int idxV(int k, int axis) const { return 7 * k + 3 + axis; }
    int idxZ(int k) const           { return 7 * k + 6; }
    int idxU(int k, int axis) const { return 7 * N_ + 4 * k + axis; }
    int idxS(int k) const           { return 7 * N_ + 4 * k + 3; }

    void assemble(double tf);
    void extract(double tf, LcvxSolution& out);
};

}  // namespace pdg

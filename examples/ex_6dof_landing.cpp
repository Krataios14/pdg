// 6-DoF free-final-time landing via successive convexification, with a
// speed-triggered angle-of-attack state-triggered constraint, then flown
// closed loop (TVC tracking autopilot) on the nominal plant.
// Writes ex_6dof_landing.csv.
#include <cmath>
#include <cstdio>
#include <fstream>

#include "pdg/scvx.hpp"
#include "pdg/sim.hpp"

int main() {
    pdg::ScvxParams P;
    P.verbose = true;
    P.stcs.push_back(pdg::makeSpeedTriggeredAoA(/*vTrigger=*/40.0, /*aoaMaxDeg=*/35.0));

    pdg::ScvxPlanner planner(P);
    pdg::ScvxSolution sol;
    if (!planner.solve(sol)) {
        std::printf("SCvx did not converge (iterations: %zu)\n", sol.history.size());
        return 1;
    }

    std::printf("\n6-DoF SCvx landing\n");
    std::printf("  time of flight : %7.3f s\n", sol.tf);
    std::printf("  fuel used      : %7.1f kg\n", sol.fuelUsed);
    std::printf("  SCvx iterations: %zu\n", sol.history.size());

    // closed-loop verification flight
    pdg::State x0{};
    x0[pdg::iM] = P.rocket.mWet;
    for (int i = 0; i < 3; ++i) x0[pdg::iR + i] = P.r0[i];
    for (int i = 0; i < 3; ++i) x0[pdg::iV + i] = P.v0[i];
    for (int i = 0; i < 4; ++i) x0[pdg::iQ + i] = P.q0[i];
    for (int i = 0; i < 3; ++i) x0[pdg::iW + i] = P.w0[i];
    pdg::RocketSim sim(P.rocket);
    pdg::TrajectoryTracker tracker(P.rocket, sol);
    pdg::TouchdownState td = sim.flyTracked(x0, tracker, 0.005);
    std::printf("  closed-loop touchdown: lateral %.2f m, descent %.2f m/s, tilt %.2f deg\n",
                td.lateralError, td.verticalSpeed, td.tiltDeg);

    std::ofstream f("ex_6dof_landing.csv");
    f << "t,m,rx,ry,rz,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz,Tx,Ty,Tz,Tmag,tilt_deg,speed\n";
    for (size_t k = 0; k < sol.t.size(); ++k) {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const double Tm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const double tilt = std::acos(std::min(1.0, std::fabs(pdg::quatTiltCos(x.data() + pdg::iQ)))) *
                            180.0 / 3.14159265358979;
        const double spd = std::sqrt(x[pdg::iV] * x[pdg::iV] +
                                     x[pdg::iV + 1] * x[pdg::iV + 1] +
                                     x[pdg::iV + 2] * x[pdg::iV + 2]);
        f << sol.t[k] << ',' << x[pdg::iM] << ',' << x[pdg::iR] << ',' << x[pdg::iR + 1] << ','
          << x[pdg::iR + 2] << ',' << x[pdg::iV] << ',' << x[pdg::iV + 1] << ',' << x[pdg::iV + 2]
          << ',' << x[pdg::iQ] << ',' << x[pdg::iQ + 1] << ',' << x[pdg::iQ + 2] << ','
          << x[pdg::iQ + 3] << ',' << x[pdg::iW] << ',' << x[pdg::iW + 1] << ',' << x[pdg::iW + 2]
          << ',' << u[0] << ',' << u[1] << ',' << u[2] << ',' << Tm << ',' << tilt << ',' << spd
          << '\n';
    }
    std::printf("trajectory written to ex_6dof_landing.csv\n");
    return 0;
}

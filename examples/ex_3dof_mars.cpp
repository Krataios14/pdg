// Classic Mars powered-descent example (Acikmese & Ploen 2007 / Blackmore 2010
// parameter class) solved with lossless convexification and a free-final-time
// search. Writes the optimal trajectory to ex_3dof_mars.csv.
#include <cmath>
#include <cstdio>
#include <fstream>

#include "pdg/lcvx.hpp"

int main() {
    pdg::LcvxParams P;                  // defaults are the classic Mars case
    pdg::LcvxPlanner planner(P);

    pdg::LcvxSolution sol;
    if (!planner.solveFreeTf(40.0, 100.0, sol)) {
        std::printf("no feasible solution found in the tf range\n");
        return 1;
    }

    std::printf("3-DoF LCvx minimum-fuel landing\n");
    std::printf("  time of flight : %7.2f s\n", sol.tf);
    std::printf("  fuel used      : %7.1f kg (of %.0f kg available)\n", sol.fuelUsed,
                P.mWet - P.mDry);
    std::printf("  final position : (%.3f, %.3f, %.3f) m\n", sol.r.back()[0], sol.r.back()[1],
                sol.r.back()[2]);
    std::printf("  final velocity : (%.3f, %.3f, %.3f) m/s\n", sol.v.back()[0], sol.v.back()[1],
                sol.v.back()[2]);
    std::printf("  SOCP           : %d iterations, %.2f ms (last solve)\n",
                sol.socpInfo.iters, sol.socpInfo.solveTimeMs);

    std::ofstream f("ex_3dof_mars.csv");
    f << "t,rx,ry,rz,vx,vy,vz,Tx,Ty,Tz,Tmag,mass,sigma\n";
    for (size_t k = 0; k < sol.t.size(); ++k) {
        const double Tm = std::sqrt(sol.thrust[k][0] * sol.thrust[k][0] +
                                    sol.thrust[k][1] * sol.thrust[k][1] +
                                    sol.thrust[k][2] * sol.thrust[k][2]);
        f << sol.t[k] << ',' << sol.r[k][0] << ',' << sol.r[k][1] << ',' << sol.r[k][2] << ','
          << sol.v[k][0] << ',' << sol.v[k][1] << ',' << sol.v[k][2] << ','
          << sol.thrust[k][0] << ',' << sol.thrust[k][1] << ',' << sol.thrust[k][2] << ','
          << Tm << ',' << sol.mass[k] << ',' << sol.sigma[k] * sol.mass[k] << '\n';
    }
    std::printf("trajectory written to ex_3dof_mars.csv\n");
    return 0;
}

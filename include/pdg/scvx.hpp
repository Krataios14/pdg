// pdg::scvx — 6-DoF free-final-time powered descent guidance via successive
// convexification, with (compound) state-triggered constraints.
//
// The free final time is handled with a time-dilation variable sigma:
// trajectories live on normalized time tau in [0,1] with dx/dtau = sigma f(x,u),
// so sigma == time of flight. Each SCvx iteration:
//   1. linearizes the dilated dynamics about the reference trajectory and
//      discretizes with a first-order hold on u (exact integration of the
//      state-transition matrix, multiple-shooting style),
//   2. builds a SOCP subproblem with virtual control (L1-penalized, keeps the
//      linearized problem always feasible) and soft quadratic trust regions,
//   3. solves it with pdg::SocpSolver (fixed sparsity pattern -> the symbolic
//      KKT factorization is computed exactly once for the whole SCvx run).
//
// State-triggered constraints implement the formulation
//      g(x,u) < 0   (trigger)   =>   c(x,u) <= 0   (constraint)
// via  h = -min(g, 0) * c <= 0, linearized in each subproblem; compound
// triggers (logical AND / OR) compose g via max / min.
//
// References:
//   M. Szmuk, B. Acikmese, "Successive Convexification for 6-DoF Mars Rocket
//     Powered Landing with Free-Final-Time", arXiv:1802.03827.
//   M. Szmuk, T. Reynolds, B. Acikmese, "Successive Convexification for
//     Real-Time 6-DoF Powered Descent Guidance with State-Triggered
//     Constraints", JGCD 43(8), 2020.
//   S. Uzun, B. Acikmese, J. M. Carson III, "Sequential Convex Programming for
//     6-DoF Powered Descent Guidance with Continuous-Time Compound
//     State-Triggered Constraints", arXiv:2510.09610.
#pragma once

#include <functional>
#include <vector>

#include "pdg/rocket.hpp"
#include "pdg/socp.hpp"

namespace pdg {

// ---------------------------------------------------------------------------
// State-triggered constraints
// ---------------------------------------------------------------------------
// Scalar function with gradients: returns value, fills gradX (size kNx) and
// gradU (size kNu) if non-null.
using StcFunc = std::function<double(const double* x, const double* u,
                                     double* gradX, double* gradU)>;

struct StateTriggeredConstraint {
    enum class Combine { And, Or };

    std::vector<StcFunc> triggers;   // trigger fires where g < 0
    Combine combine = Combine::And;  // AND: all g_i < 0; OR: any g_i < 0
    StcFunc constraint;              // enforced (<= 0) where the trigger fires
    const char* name = "stc";
};

// Built-in example (Szmuk-Reynolds-Acikmese 2020): when speed exceeds vTrigger,
// limit the angle of attack (angle between -v and the body +z axis) to aoaMaxDeg.
StateTriggeredConstraint makeSpeedTriggeredAoA(double vTrigger, double aoaMaxDeg);

// ---------------------------------------------------------------------------
// problem definition
// ---------------------------------------------------------------------------
struct ScvxParams {
    RocketParams rocket;

    // boundary conditions (z-up inertial frame, target at origin)
    Vec3 r0 = {200.0, 100.0, 500.0};
    Vec3 v0 = {-30.0, 5.0, -50.0};
    Vec4 q0 = {1.0, 0.0, 0.0, 0.0};
    Vec3 w0 = {0.0, 0.0, 0.0};
    bool fixInitialAttitude = true;   // if false, q0/w0 are free

    Vec3 rf = {0.0, 0.0, 0.0};
    Vec3 vf = {0.0, 0.0, -1.0};       // small terminal descent rate
    bool fixFinalAttitude = true;     // upright touchdown, zero tilt rates (roll free:
                                      // a single gimbaled engine has no roll authority)

    // free final time
    double tfGuess = 12.0;
    double tfMin = 2.0, tfMax = 60.0;

    // path constraints
    double tiltMaxDeg   = 70.0;       // max tilt from vertical
    double omegaMaxDegS = 60.0;       // max body rate
    double glideSlopeDeg = 20.0;      // min elevation from horizontal (<=0 disables)
    double gimbalMaxDeg = 12.0;       // max thrust angle off body +z

    std::vector<StateTriggeredConstraint> stcs;

    // discretization / SCvx (hard trust region + ratio test, Mao-Szmuk-Acikmese)
    int K = 21;                       // temporal nodes
    int rk4Substeps = 10;             // RK4 steps per interval (discretization)
    int maxIters = 50;                // max accepted+rejected iterations
    double wVirtual = 1e3;            // L1 penalty weight on dynamic infeasibility
    double trInit = 1.0;              // initial trust-region radius (scaled units)
    double trMin = 1e-5, trMax = 8.0; // radius bounds
    double rho0 = 0.0;                // reject step below this ratio
    double rho1 = 0.25, rho2 = 0.8;   // shrink below rho1, grow above rho2
    double trShrink = 2.0, trGrow = 2.5;
    double tolDx = 1e-3;              // convergence: max scaled state change
    double tolNu = 1e-7;              // convergence: max scaled virtual control
    double tolPred = 1e-7;            // convergence: predicted penalty reduction
    SocpSettings socp;
    bool verbose = false;
};

struct ScvxIterInfo {
    int iter = 0;
    double cost = 0.0;        // SOCP objective (linearized penalty cost)
    double penalty = 0.0;     // nonlinear penalty cost of the candidate
    double rho = 0.0;         // actual/predicted reduction ratio
    double trustRadius = 0.0; // radius used for this subproblem
    bool accepted = false;
    double nuMax = 0.0;       // max |virtual control| (scaled)
    double dxMax = 0.0;       // max scaled state deviation from reference
    double sigma = 0.0;       // candidate time of flight
    double maxDefect = 0.0;   // max nonlinear single-shooting defect (scaled)
    SocpStatus socpStatus = SocpStatus::NumericalError;
};

struct ScvxSolution {
    bool converged = false;
    bool feasible = false;            // virtual control ~ 0 on the final iterate
    double tf = 0.0;
    double fuelUsed = 0.0;
    Vec t;                            // node times [s]
    std::vector<State> x;             // states per node
    std::vector<Vec3> u;              // body-frame thrust per node [N]
    std::vector<ScvxIterInfo> history;
};

class ScvxPlanner {
public:
    explicit ScvxPlanner(const ScvxParams& params);
    bool solve(ScvxSolution& out);
    const ScvxParams& params() const { return prm_; }

private:
    ScvxParams prm_;
    int K_ = 0;
    int nStc_ = 0;

    // scaling
    State sx_;
    Vec3 su_;
    double ssig_ = 1.0;

    // reference trajectory
    std::vector<State> xRef_;
    std::vector<Vec3> uRef_;
    double sigRef_ = 0.0;

    // candidate from the latest subproblem
    std::vector<State> xCand_;
    std::vector<Vec3> uCand_;
    double sigCand_ = 0.0;

    // discretization matrices per interval
    std::vector<Mat> Ad_, Bm_, Bp_;
    std::vector<State> Sd_, zd_;

    SocpProblem prob_;
    SocpSolver solver_;
    bool solverReady_ = false;

    // variable layout
    int idxX(int k, int i) const { return kNx * k + i; }
    int idxU(int k, int j) const { return kNx * K_ + kNu * k + j; }
    int idxSig() const           { return kNx * K_ + kNu * K_; }
    int idxNu(int k, int i) const   { return idxSig() + 1 + kNx * k + i; }
    int idxNuAbs(int k, int i) const{ return idxSig() + 1 + kNx * (K_ - 1) + kNx * k + i; }
    int idxStcSlack(int k, int s) const { return idxSig() + 1 + 2 * kNx * (K_ - 1) + nStc_ * k + s; }
    int numVars() const             { return idxSig() + 1 + 2 * kNx * (K_ - 1) + nStc_ * K_; }

    void initReference();
    void discretize();
    void assemble(double trustRadius);
    void extractCandidate(ScvxIterInfo& info);
    // nonlinear penalty J = -m_N/mWet + wVirtual * (scaled L1 defects + STC violations)
    double nonlinearPenalty(const std::vector<State>& xs, const std::vector<Vec3>& us,
                            double sig, double* maxDefect = nullptr) const;
};

}  // namespace pdg

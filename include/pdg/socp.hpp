// pdg::socp — a small, dependency-free second-order cone programming solver
// designed along the lines of embedded conic solvers (ECOS, Clarabel):
//
//   minimize    c'x
//   subject to  A x = b
//               G x + s = h,   s in K = R+^l  x  Q^{q_1} x ... x Q^{q_N}
//
// where Q^q = { (t,u) in R x R^{q-1} : ||u||_2 <= t } is the second-order cone.
//
// Algorithm: primal-dual interior-point method on the homogeneous self-dual
// embedding, with Nesterov-Todd scaling and a Mehrotra predictor-corrector.
// The KKT systems are solved with a sparse LDL^T factorization whose symbolic
// structure is computed ONCE in setup(); solve() performs no allocation, has a
// bounded iteration count, and therefore bounded execution time — the
// properties required for embedded / flight-software style deployment.
//
// References:
//   A. Domahidi, E. Chu, S. Boyd, "ECOS: An SOCP solver for embedded systems",
//     ECC 2013.
//   L. Vandenberghe, "The CVXOPT linear and quadratic cone program solvers", 2010.
#pragma once

#include <string>
#include <vector>

#include "pdg/linalg.hpp"

namespace pdg {

struct ConeSpec {
    int l = 0;                 // dimension of the nonnegative orthant
    std::vector<int> q;        // dimensions of the second-order cones (each >= 1)

    int totalDim() const {
        int m = l;
        for (int qi : q) m += qi;
        return m;
    }
    // barrier degree: l + number of SOCs
    int degree() const { return l + static_cast<int>(q.size()); }
};

struct SocpProblem {
    int n = 0;                 // number of variables
    Vec c;                     // cost, size n
    Triplets A;                // p x n equality matrix (triplets)
    Vec b;                     // size p
    Triplets G;                // m x n cone matrix (triplets)
    Vec h;                     // size m
    ConeSpec cone;

    int p() const { return static_cast<int>(b.size()); }
    int m() const { return static_cast<int>(h.size()); }
};

enum class SocpStatus {
    Optimal,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,      // returned best iterate; may be usable
    NumericalError
};

const char* socpStatusName(SocpStatus s);

struct SocpSettings {
    int    maxIters   = 60;
    double feasTol    = 1e-7;   // primal/dual feasibility tolerance (relative)
    double absGapTol  = 5e-8;   // absolute duality gap tolerance
    double relGapTol  = 5e-8;   // relative duality gap tolerance
    double staticReg  = 1e-8;   // static KKT regularization
    double dynReg     = 1e-8;   // dynamic pivot regularization floor
    int    refineIters = 3;     // iterative refinement sweeps per KKT solve
    double minStep    = 1e-10;  // declare numerical failure below this step size
    double gamma      = 0.98;   // step-size damping (fraction to boundary)
    bool   verbose    = false;
};

struct SocpInfo {
    SocpStatus status = SocpStatus::NumericalError;
    int    iters = 0;
    double pcost = 0.0, dcost = 0.0;
    double presid = 0.0, dresid = 0.0;
    double gap = 0.0, relGap = 0.0;
    double solveTimeMs = 0.0;
};

// The solver separates setup (symbolic analysis + allocation) from solve so that
// problems with a FIXED sparsity pattern but changing numeric data — exactly the
// situation inside an SCvx loop — pay the symbolic cost once.
class SocpSolver {
public:
    SocpSolver() = default;
    explicit SocpSolver(const SocpProblem& prob, const SocpSettings& s = SocpSettings());

    // Symbolic setup. The triplet PATTERN of prob.A/prob.G must not change between
    // setup() and subsequent solve() calls; values may change freely.
    void setup(const SocpProblem& prob, const SocpSettings& s = SocpSettings());

    // Solve with the current numeric data in `prob` (must match setup pattern).
    SocpStatus solve(const SocpProblem& prob);

    const Vec& x() const { return x_; }       // primal solution (length n)
    const Vec& y() const { return y_; }       // equality duals (length p)
    const Vec& z() const { return z_; }       // cone duals (length m)
    const Vec& s() const { return s_; }       // cone slacks (length m)
    const SocpInfo& info() const { return info_; }
    size_t workspaceBytes() const;

private:
    SocpSettings set_;
    int n_ = 0, p_ = 0, m_ = 0, kktDim_ = 0;
    ConeSpec cone_;
    int coneDeg_ = 0;

    // problem data in CSC form (+ triplet scatter maps for numeric refresh)
    SparseCSC Acsc_, Gcsc_;
    std::vector<int> Amap_, Gmap_;
    Vec b_, c_, h_;

    // KKT matrix: [ dI  A'  G' ; A  -dI  0 ; G  0  -W'W-dI ] (upper triangle)
    SparseCSC kkt_;
    std::vector<int> kktAmap_, kktGmap_;   // nnz positions of A', G' blocks
    std::vector<int> kktDiagX_;            // positions of x-block diagonal
    std::vector<int> kktWmap_;             // positions of the -W'W blocks (per cone, dense upper)
    LDLSolver ldl_;

    // iterates
    Vec x_, y_, z_, s_;
    double tau_ = 1.0, kappa_ = 1.0;

    // Nesterov-Todd scaling storage
    Vec wLp_;                              // sqrt(s_i/z_i) for LP cone
    Vec lambda_;                           // scaled point, length m
    std::vector<Vec> socW_;                // w-bar per SOC
    Vec socEta_;                           // eta per SOC

    // work vectors
    Vec r1_, r2_, r3_;  double r4_ = 0.0;
    Vec u1_, u2_;                          // KKT solutions (n+p+m)
    Vec rhs_, ds_, dsAff_, dzAff_, work_, work2_, workm_, workm2_;
    Vec dx_, dy_, dz_, dsv_;
    SocpInfo info_;

    void buildKktPattern();
    void refreshKktValues();               // A,G value refresh (W blocks set separately)
    void setKktScalingBlocks();            // write -W'W into KKT
    bool computeScaling();                 // NT scaling from current (s,z)
    void applyW (const double* v, double* out) const;     // out = W v
    void applyWinv(const double* v, double* out) const;   // out = W^{-1} v
    void jordanMul(const double* u, const double* v, double* out) const; // u o v
    bool jordanDiv(const double* d, double* out) const;   // solve lambda o out = d
    double maxStep(const double* v, const double* dv) const; // sup step in cone
    bool solveKkt(const double* rx, const double* ry, const double* rz, double* u);
    bool inCone(const double* v, double margin) const;
    void bringToCone(double* v) const;
};

}  // namespace pdg

// pdg::linalg — minimal dense/sparse linear algebra for embedded conic optimization.
//
// Design goals:
//  * No external dependencies.
//  * All memory allocated during setup (analyze); factor/solve are allocation-free.
//  * Sparse LDL^T (Davis-style up-looking, fixed elimination tree) with static +
//    dynamic regularization and iterative refinement — the same scheme used by
//    embedded conic solvers such as ECOS.
#pragma once

#include <cstddef>
#include <vector>

namespace pdg {

using Vec = std::vector<double>;

// ---------------------------------------------------------------------------
// Dense helpers (operate on raw pointers; used by solver inner loops)
// ---------------------------------------------------------------------------
double dot(const double* a, const double* b, int n);
double norm2(const double* a, int n);
double normInf(const double* a, int n);
void   axpy(double alpha, const double* x, double* y, int n);  // y += alpha*x
void   scal(double alpha, double* x, int n);
void   copy(const double* src, double* dst, int n);
void   setZero(double* x, int n);

// ---------------------------------------------------------------------------
// Small dense matrix (row-major). Used by the 6-DoF discretizer (14x14 blocks).
// ---------------------------------------------------------------------------
struct Mat {
    int rows = 0, cols = 0;
    Vec a;

    Mat() = default;
    Mat(int r, int c) : rows(r), cols(c), a(static_cast<size_t>(r) * c, 0.0) {}

    double&       operator()(int r, int c)       { return a[static_cast<size_t>(r) * cols + c]; }
    const double& operator()(int r, int c) const { return a[static_cast<size_t>(r) * cols + c]; }

    void resize(int r, int c) { rows = r; cols = c; a.assign(static_cast<size_t>(r) * c, 0.0); }
    void setZero() { std::fill(a.begin(), a.end(), 0.0); }
    void setIdentity();

    static Mat identity(int n);
};

Mat  matMul(const Mat& A, const Mat& B);
void matMulInto(const Mat& A, const Mat& B, Mat& C);          // C = A*B (no alloc if sized)
void matVec(const Mat& A, const double* x, double* y);         // y = A*x
void matAddScaled(Mat& A, const Mat& B, double alpha);         // A += alpha*B
// Solve A*X = B in place via partial-pivot LU (A overwritten). Returns false if singular.
bool luSolve(Mat& A, Mat& B);
bool luSolve(Mat& A, double* b);                               // single rhs

// ---------------------------------------------------------------------------
// Sparse matrices
// ---------------------------------------------------------------------------
struct Triplets {
    std::vector<int> ti, tj;
    Vec tv;
    void clear() { ti.clear(); tj.clear(); tv.clear(); }
    void add(int i, int j, double v) {
        ti.push_back(i); tj.push_back(j); tv.push_back(v);
    }
    size_t size() const { return ti.size(); }
};

// Compressed sparse column. Row indices sorted within each column; duplicates summed.
struct SparseCSC {
    int rows = 0, cols = 0;
    std::vector<int> p;   // col pointers, size cols+1
    std::vector<int> i;   // row indices
    Vec x;                // values

    int nnz() const { return p.empty() ? 0 : p[cols]; }
};

// Build CSC from triplets. If `tripletMap` is non-null it receives, for each input
// triplet, the index into the output value array (so numeric values can later be
// rescattered without re-sorting — duplicate triplets map to the same slot).
SparseCSC buildCSC(int rows, int cols, const Triplets& T, std::vector<int>* tripletMap = nullptr);

// Re-scatter triplet values into an already-built CSC using the map from buildCSC.
void scatterValues(SparseCSC& A, const Triplets& T, const std::vector<int>& tripletMap);

void cscMulAdd (const SparseCSC& A, const double* x, double* y, double alpha = 1.0); // y += a*A*x
void cscTMulAdd(const SparseCSC& A, const double* x, double* y, double alpha = 1.0); // y += a*A'*x
// y += alpha * K * x where K is symmetric and only its UPPER triangle is stored.
void symMulAdd(const SparseCSC& U, const double* x, double* y, double alpha = 1.0);

// Reverse Cuthill–McKee ordering of a symmetric matrix given by its upper triangle.
// Nodes with degree > denseThreshold are deferred to the end of the ordering (this
// contains fill-in from dense rows/columns such as a free-final-time variable that
// couples every dynamics constraint). Returns perm such that newIndex = perm[oldIndex].
std::vector<int> rcmOrder(const SparseCSC& upper, int denseThreshold);

// ---------------------------------------------------------------------------
// Sparse LDL^T for symmetric quasidefinite systems.
//
// The matrix is supplied as the upper triangle in CSC form. `analyze()` computes a
// fill-reducing permutation (RCM with dense-node deferral, or a user permutation),
// the elimination tree and the symbolic factor, and allocates ALL working memory.
// `factor()` and `solve()` never allocate, so worst-case execution time is bounded
// by the (fixed) symbolic structure — the property needed for embedded use.
//
// Regularization: factor() applies dynamic regularization — if a pivot D[k] falls
// below dynReg in magnitude (relative to its expected sign), it is replaced by
// sign*dynReg. solveRefine() performs iterative refinement against the ORIGINAL
// (unregularized) matrix to recover accuracy lost to regularization.
// ---------------------------------------------------------------------------
class LDLSolver {
public:
    // signs[k] = +1 or -1: expected sign of pivot k in the ORIGINAL ordering
    // (quasidefinite structure). userPerm, if given, overrides RCM.
    void analyze(const SparseCSC& upper, const std::vector<int>& signs,
                 const std::vector<int>* userPerm = nullptr, int denseThreshold = -1);

    // Numeric factorization of `upper` (same pattern as analyzed). Static
    // regularization staticReg*sign is ADDED to every diagonal entry; pivots that
    // still cross zero are clamped to sign*dynReg. Returns false on breakdown.
    bool factor(const SparseCSC& upper, double staticReg, double dynReg);

    // Solve K*x = b using the factorization (x and b in ORIGINAL ordering, may alias).
    void solve(const double* b, double* x) const;

    // Solve with `iters` rounds of iterative refinement against `upper`
    // (the unregularized matrix). Returns final residual inf-norm.
    double solveRefine(const SparseCSC& upper, const double* b, double* x, int iters) const;

    int    dim() const { return n_; }
    size_t workspaceBytes() const;
    int    numDynamicRegularizations() const { return numDynReg_; }

private:
    int n_ = 0;
    std::vector<int> perm_, iperm_;          // newIdx = perm_[oldIdx]
    std::vector<int> signsPerm_;
    // permuted upper-triangular copy of the matrix
    SparseCSC P_;
    std::vector<int> scatterMap_;            // input nnz idx -> P_ nnz idx
    // symbolic factor
    std::vector<int> Lp_, parent_, Lnz_, Li_;
    Vec Lx_, D_;
    // work arrays
    mutable std::vector<int> flag_, pattern_, stack_, lnzUsed_;
    mutable Vec Y_;
    mutable Vec workB_, workX_, workR_;
    int numDynReg_ = 0;

    void permuteValues(const SparseCSC& upper);
    void solvePermuted(double* b) const;     // in place, permuted ordering
};

}  // namespace pdg

#include "pdg/linalg.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

namespace pdg {

// ---------------------------------------------------------------------------
// Dense helpers
// ---------------------------------------------------------------------------
double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += a[k] * b[k];
    return s;
}

double norm2(const double* a, int n) { return std::sqrt(dot(a, a, n)); }

double normInf(const double* a, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s = std::max(s, std::fabs(a[k]));
    return s;
}

void axpy(double alpha, const double* x, double* y, int n) {
    for (int k = 0; k < n; ++k) y[k] += alpha * x[k];
}

void scal(double alpha, double* x, int n) {
    for (int k = 0; k < n; ++k) x[k] *= alpha;
}

void copy(const double* src, double* dst, int n) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(double));
}

void setZero(double* x, int n) {
    std::memset(x, 0, static_cast<size_t>(n) * sizeof(double));
}

// ---------------------------------------------------------------------------
// Dense Mat
// ---------------------------------------------------------------------------
void Mat::setIdentity() {
    setZero();
    const int m = std::min(rows, cols);
    for (int k = 0; k < m; ++k) (*this)(k, k) = 1.0;
}

Mat Mat::identity(int n) {
    Mat I(n, n);
    I.setIdentity();
    return I;
}

Mat matMul(const Mat& A, const Mat& B) {
    Mat C(A.rows, B.cols);
    matMulInto(A, B, C);
    return C;
}

void matMulInto(const Mat& A, const Mat& B, Mat& C) {
    assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);
    C.setZero();
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            const double aik = A(i, k);
            if (aik == 0.0) continue;
            const double* brow = &B.a[static_cast<size_t>(k) * B.cols];
            double* crow = &C.a[static_cast<size_t>(i) * C.cols];
            for (int j = 0; j < B.cols; ++j) crow[j] += aik * brow[j];
        }
    }
}

void matVec(const Mat& A, const double* x, double* y) {
    for (int i = 0; i < A.rows; ++i) {
        const double* row = &A.a[static_cast<size_t>(i) * A.cols];
        y[i] = dot(row, x, A.cols);
    }
}

void matAddScaled(Mat& A, const Mat& B, double alpha) {
    assert(A.rows == B.rows && A.cols == B.cols);
    for (size_t k = 0; k < A.a.size(); ++k) A.a[k] += alpha * B.a[k];
}

bool luSolve(Mat& A, Mat& B) {
    assert(A.rows == A.cols && A.rows == B.rows);
    const int n = A.rows;
    for (int k = 0; k < n; ++k) {
        // partial pivot
        int piv = k;
        double best = std::fabs(A(k, k));
        for (int r = k + 1; r < n; ++r) {
            const double v = std::fabs(A(r, k));
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) return false;
        if (piv != k) {
            for (int c = 0; c < n; ++c) std::swap(A(k, c), A(piv, c));
            for (int c = 0; c < B.cols; ++c) std::swap(B(k, c), B(piv, c));
        }
        const double akk = A(k, k);
        for (int r = k + 1; r < n; ++r) {
            const double f = A(r, k) / akk;
            if (f == 0.0) continue;
            A(r, k) = 0.0;
            for (int c = k + 1; c < n; ++c) A(r, c) -= f * A(k, c);
            for (int c = 0; c < B.cols; ++c) B(r, c) -= f * B(k, c);
        }
    }
    // back substitution
    for (int k = n - 1; k >= 0; --k) {
        for (int c = 0; c < B.cols; ++c) {
            double s = B(k, c);
            for (int j = k + 1; j < n; ++j) s -= A(k, j) * B(j, c);
            B(k, c) = s / A(k, k);
        }
    }
    return true;
}

bool luSolve(Mat& A, double* b) {
    Mat B(A.rows, 1);
    for (int r = 0; r < A.rows; ++r) B(r, 0) = b[r];
    if (!luSolve(A, B)) return false;
    for (int r = 0; r < A.rows; ++r) b[r] = B(r, 0);
    return true;
}

// ---------------------------------------------------------------------------
// CSC construction
// ---------------------------------------------------------------------------
SparseCSC buildCSC(int rows, int cols, const Triplets& T, std::vector<int>* tripletMap) {
    const size_t nz = T.size();
    SparseCSC A;
    A.rows = rows;
    A.cols = cols;

    // sort triplets by (col, row) via index sort
    std::vector<int> order(nz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (T.tj[a] != T.tj[b]) return T.tj[a] < T.tj[b];
        return T.ti[a] < T.ti[b];
    });

    A.p.assign(static_cast<size_t>(cols) + 1, 0);
    A.i.reserve(nz);
    A.x.reserve(nz);
    if (tripletMap) tripletMap->assign(nz, -1);

    int lastCol = -1, lastRow = -1;
    for (size_t s = 0; s < nz; ++s) {
        const int t = order[s];
        const int r = T.ti[t], c = T.tj[t];
        assert(r >= 0 && r < rows && c >= 0 && c < cols);
        if (c == lastCol && r == lastRow) {
            A.x.back() += T.tv[t];                // duplicate: sum
        } else {
            A.i.push_back(r);
            A.x.push_back(T.tv[t]);
            lastCol = c;
            lastRow = r;
        }
        if (tripletMap) (*tripletMap)[t] = static_cast<int>(A.x.size()) - 1;
        A.p[static_cast<size_t>(c) + 1] = static_cast<int>(A.x.size());
    }
    // fill column pointers for empty columns
    for (int c = 1; c <= cols; ++c) A.p[c] = std::max(A.p[c], A.p[c - 1]);
    return A;
}

void scatterValues(SparseCSC& A, const Triplets& T, const std::vector<int>& tripletMap) {
    std::fill(A.x.begin(), A.x.end(), 0.0);
    for (size_t t = 0; t < T.size(); ++t) A.x[tripletMap[t]] += T.tv[t];
}

void cscMulAdd(const SparseCSC& A, const double* x, double* y, double alpha) {
    for (int c = 0; c < A.cols; ++c) {
        const double xc = alpha * x[c];
        if (xc == 0.0) continue;
        for (int k = A.p[c]; k < A.p[c + 1]; ++k) y[A.i[k]] += A.x[k] * xc;
    }
}

void cscTMulAdd(const SparseCSC& A, const double* x, double* y, double alpha) {
    for (int c = 0; c < A.cols; ++c) {
        double s = 0.0;
        for (int k = A.p[c]; k < A.p[c + 1]; ++k) s += A.x[k] * x[A.i[k]];
        y[c] += alpha * s;
    }
}

void symMulAdd(const SparseCSC& U, const double* x, double* y, double alpha) {
    for (int c = 0; c < U.cols; ++c) {
        double s = 0.0;
        for (int k = U.p[c]; k < U.p[c + 1]; ++k) {
            const int r = U.i[k];
            s += U.x[k] * x[r];
            if (r != c) y[r] += alpha * U.x[k] * x[c];  // strict-lower mirror
        }
        y[c] += alpha * s;
    }
}

// ---------------------------------------------------------------------------
// RCM ordering with dense-node deferral
// ---------------------------------------------------------------------------
std::vector<int> rcmOrder(const SparseCSC& upper, int denseThreshold) {
    const int n = upper.cols;
    // build full (symmetrized) adjacency, excluding diagonal
    std::vector<int> deg(n, 0);
    for (int c = 0; c < n; ++c) {
        for (int k = upper.p[c]; k < upper.p[c + 1]; ++k) {
            const int r = upper.i[k];
            if (r == c) continue;
            ++deg[r];
            ++deg[c];
        }
    }
    std::vector<int> adjp(static_cast<size_t>(n) + 1, 0);
    for (int v = 0; v < n; ++v) adjp[v + 1] = adjp[v] + deg[v];
    std::vector<int> adj(adjp[n]);
    std::vector<int> fill(n, 0);
    for (int c = 0; c < n; ++c) {
        for (int k = upper.p[c]; k < upper.p[c + 1]; ++k) {
            const int r = upper.i[k];
            if (r == c) continue;
            adj[adjp[r] + fill[r]++] = c;
            adj[adjp[c] + fill[c]++] = r;
        }
    }

    if (denseThreshold < 0) {
        long long sum = 0;
        for (int v = 0; v < n; ++v) sum += deg[v];
        const int avg = n > 0 ? static_cast<int>(sum / std::max(1, n)) : 0;
        denseThreshold = std::max(24, 8 * std::max(1, avg));
    }

    std::vector<char> dense(n, 0);
    for (int v = 0; v < n; ++v) dense[v] = (deg[v] > denseThreshold) ? 1 : 0;

    std::vector<int> order;            // Cuthill–McKee order of non-dense nodes
    order.reserve(n);
    std::vector<char> visited(n, 0);
    std::vector<int> queue(n);

    // BFS over each connected component, starting from min-degree unvisited node
    while (true) {
        int start = -1, bestDeg = n + 1;
        for (int v = 0; v < n; ++v) {
            if (!visited[v] && !dense[v] && deg[v] < bestDeg) { bestDeg = deg[v]; start = v; }
        }
        if (start < 0) break;
        int head = 0, tail = 0;
        queue[tail++] = start;
        visited[start] = 1;
        while (head < tail) {
            const int v = queue[head++];
            order.push_back(v);
            const int b = adjp[v], e = adjp[v + 1];
            const int first = tail;
            for (int k = b; k < e; ++k) {
                const int w = adj[k];
                if (!visited[w] && !dense[w]) {
                    visited[w] = 1;
                    queue[tail++] = w;
                }
            }
            std::sort(queue.begin() + first, queue.begin() + tail,
                      [&](int a, int bb) { return deg[a] < deg[bb]; });
        }
    }
    std::reverse(order.begin(), order.end());  // reverse CM
    for (int v = 0; v < n; ++v)
        if (dense[v]) order.push_back(v);      // dense nodes last

    std::vector<int> perm(n);
    for (int newIdx = 0; newIdx < n; ++newIdx) perm[order[newIdx]] = newIdx;
    return perm;
}

// ---------------------------------------------------------------------------
// LDL^T
// ---------------------------------------------------------------------------
void LDLSolver::analyze(const SparseCSC& upper, const std::vector<int>& signs,
                        const std::vector<int>* userPerm, int denseThreshold) {
    assert(upper.rows == upper.cols);
    n_ = upper.cols;
    assert(static_cast<int>(signs.size()) == n_);

    perm_ = userPerm ? *userPerm : rcmOrder(upper, denseThreshold);
    iperm_.resize(n_);
    for (int v = 0; v < n_; ++v) iperm_[perm_[v]] = v;
    signsPerm_.resize(n_);
    for (int v = 0; v < n_; ++v) signsPerm_[perm_[v]] = signs[v];

    // Build permuted upper triangle P_ = perm(K), with scatter map for fast
    // numeric updates. Ensure every diagonal entry exists (for regularization).
    Triplets T;
    T.ti.reserve(upper.nnz() + n_);
    T.tj.reserve(upper.nnz() + n_);
    T.tv.reserve(upper.nnz() + n_);
    for (int c = 0; c < n_; ++c) {
        for (int k = upper.p[c]; k < upper.p[c + 1]; ++k) {
            const int r = upper.i[k];
            int pr = perm_[r], pc = perm_[c];
            if (pr > pc) std::swap(pr, pc);
            T.add(pr, pc, 0.0);
        }
    }
    for (int v = 0; v < n_; ++v) T.add(v, v, 0.0);  // guaranteed diagonal
    P_ = buildCSC(n_, n_, T, &scatterMap_);
    // scatterMap_ currently maps "triplet idx" -> P_ nnz idx; the first nnz(upper)
    // triplets correspond 1:1 with the entries of `upper` in column order.

    // symbolic factorization (elimination tree + column counts), Davis LDL
    parent_.assign(n_, -1);
    Lnz_.assign(n_, 0);
    flag_.assign(n_, -1);
    for (int k = 0; k < n_; ++k) {
        parent_[k] = -1;
        flag_[k] = k;
        for (int p = P_.p[k]; p < P_.p[k + 1]; ++p) {
            int i = P_.i[p];
            while (i < k && flag_[i] != k) {
                if (parent_[i] == -1) parent_[i] = k;
                ++Lnz_[i];
                flag_[i] = k;
                i = parent_[i];
            }
        }
    }
    Lp_.assign(static_cast<size_t>(n_) + 1, 0);
    for (int k = 0; k < n_; ++k) Lp_[k + 1] = Lp_[k] + Lnz_[k];
    Li_.assign(Lp_[n_], 0);
    Lx_.assign(Lp_[n_], 0.0);
    D_.assign(n_, 0.0);
    Y_.assign(n_, 0.0);
    pattern_.assign(n_, 0);
    stack_.assign(n_, 0);
    lnzUsed_.assign(n_, 0);
    workB_.assign(n_, 0.0);
    workX_.assign(n_, 0.0);
    workR_.assign(n_, 0.0);
}

void LDLSolver::permuteValues(const SparseCSC& upper) {
    std::fill(P_.x.begin(), P_.x.end(), 0.0);
    int t = 0;
    for (int c = 0; c < n_; ++c)
        for (int k = upper.p[c]; k < upper.p[c + 1]; ++k, ++t)
            P_.x[scatterMap_[t]] += upper.x[k];
    // trailing n_ map entries are the guaranteed diagonal (value 0) — nothing to add
}

bool LDLSolver::factor(const SparseCSC& upper, double staticReg, double dynReg) {
    permuteValues(upper);
    numDynReg_ = 0;

    std::fill(lnzUsed_.begin(), lnzUsed_.end(), 0);
    std::fill(flag_.begin(), flag_.end(), -1);
    std::fill(Y_.begin(), Y_.end(), 0.0);

    int* stack = stack_.data();
    int* lnzUsed = lnzUsed_.data();
    for (int k = 0; k < n_; ++k) {
        // pattern of row k of L = nodes reachable in etree from nonzeros of col k
        int top = n_;
        flag_[k] = k;
        for (int p = P_.p[k]; p < P_.p[k + 1]; ++p) {
            const int rowIdx = P_.i[p];
            if (rowIdx > k) continue;
            Y_[rowIdx] += P_.x[p];
            int len = 0;
            for (int i = rowIdx; flag_[i] != k; i = parent_[i]) {
                stack[len++] = i;
                flag_[i] = k;
            }
            while (len > 0) pattern_[--top] = stack[--len];
        }
        double dk = Y_[k] + staticReg * signsPerm_[k];
        Y_[k] = 0.0;
        for (; top < n_; ++top) {
            const int i = pattern_[top];
            const double yi = Y_[i];
            Y_[i] = 0.0;
            const int p2 = Lp_[i] + lnzUsed[i];
            for (int p = Lp_[i]; p < p2; ++p) Y_[Li_[p]] -= Lx_[p] * yi;
            const double lki = yi / D_[i];
            dk -= lki * yi;
            Li_[p2] = k;
            Lx_[p2] = lki;
            ++lnzUsed[i];
        }
        // dynamic regularization: keep pivot away from zero with the expected sign
        const double sgn = static_cast<double>(signsPerm_[k]);
        if (dk * sgn < dynReg) {
            dk = sgn * dynReg;
            ++numDynReg_;
        }
        if (dk == 0.0 || !std::isfinite(dk)) return false;
        D_[k] = dk;
    }
    return true;
}

void LDLSolver::solvePermuted(double* b) const {
    // L y = b
    for (int j = 0; j < n_; ++j) {
        const double bj = b[j];
        if (bj == 0.0) continue;
        const int e = Lp_[j + 1];
        for (int p = Lp_[j]; p < e; ++p) b[Li_[p]] -= Lx_[p] * bj;
    }
    // D z = y
    for (int j = 0; j < n_; ++j) b[j] /= D_[j];
    // L' x = z
    for (int j = n_ - 1; j >= 0; --j) {
        double s = b[j];
        const int e = Lp_[j + 1];
        for (int p = Lp_[j]; p < e; ++p) s -= Lx_[p] * b[Li_[p]];
        b[j] = s;
    }
}

void LDLSolver::solve(const double* b, double* x) const {
    for (int v = 0; v < n_; ++v) workB_[perm_[v]] = b[v];
    solvePermuted(workB_.data());
    for (int v = 0; v < n_; ++v) x[v] = workB_[perm_[v]];
}

double LDLSolver::solveRefine(const SparseCSC& upper, const double* b, double* x,
                              int iters) const {
    solve(b, x);
    double resNorm = 0.0;
    for (int it = 0; it < iters; ++it) {
        // r = b - K x  (K given by `upper`, symmetric)
        copy(b, workR_.data(), n_);
        symMulAdd(upper, x, workR_.data(), -1.0);
        resNorm = normInf(workR_.data(), n_);
        const double bNorm = std::max(1.0, normInf(b, n_));
        if (resNorm <= 1e-14 * bNorm) break;
        solve(workR_.data(), workX_.data());
        axpy(1.0, workX_.data(), x, n_);
    }
    // final residual
    copy(b, workR_.data(), n_);
    symMulAdd(upper, x, workR_.data(), -1.0);
    return normInf(workR_.data(), n_);
}

size_t LDLSolver::workspaceBytes() const {
    size_t bytes = 0;
    bytes += (perm_.size() + iperm_.size() + signsPerm_.size()) * sizeof(int);
    bytes += (P_.p.size() + P_.i.size() + scatterMap_.size()) * sizeof(int);
    bytes += P_.x.size() * sizeof(double);
    bytes += (Lp_.size() + parent_.size() + Lnz_.size() + Li_.size() + flag_.size() +
              pattern_.size()) * sizeof(int);
    bytes += (Lx_.size() + D_.size() + Y_.size() + workB_.size() + workX_.size() +
              workR_.size()) * sizeof(double);
    return bytes;
}

}  // namespace pdg

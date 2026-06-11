#include "pdg/socp.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace pdg {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

const char* socpStatusName(SocpStatus s) {
    switch (s) {
        case SocpStatus::Optimal:          return "Optimal";
        case SocpStatus::PrimalInfeasible: return "PrimalInfeasible";
        case SocpStatus::DualInfeasible:   return "DualInfeasible";
        case SocpStatus::MaxIterations:    return "MaxIterations";
        case SocpStatus::NumericalError:   return "NumericalError";
    }
    return "?";
}

SocpSolver::SocpSolver(const SocpProblem& prob, const SocpSettings& s) { setup(prob, s); }

// ---------------------------------------------------------------------------
// Setup: build CSC data, KKT pattern, symbolic LDL
// ---------------------------------------------------------------------------
void SocpSolver::setup(const SocpProblem& prob, const SocpSettings& s) {
    set_ = s;
    n_ = prob.n;
    p_ = prob.p();
    m_ = prob.m();
    cone_ = prob.cone;
    coneDeg_ = cone_.degree();
    kktDim_ = n_ + p_ + m_;
    assert(cone_.totalDim() == m_ && "cone dimensions must sum to rows of G");
    assert(n_ > 0 && m_ > 0);

    Acsc_ = buildCSC(p_, n_, prob.A, &Amap_);
    Gcsc_ = buildCSC(m_, n_, prob.G, &Gmap_);
    b_ = prob.b;
    c_ = prob.c;
    h_ = prob.h;

    buildKktPattern();

    std::vector<int> signs(kktDim_);
    for (int k = 0; k < kktDim_; ++k) signs[k] = (k < n_) ? +1 : -1;
    ldl_.analyze(kkt_, signs);

    // iterate / work storage
    x_.assign(n_, 0.0);
    y_.assign(p_, 0.0);
    z_.assign(m_, 0.0);
    s_.assign(m_, 0.0);
    wLp_.assign(cone_.l, 1.0);
    lambda_.assign(m_, 0.0);
    socW_.assign(cone_.q.size(), Vec());
    for (size_t k = 0; k < cone_.q.size(); ++k) socW_[k].assign(cone_.q[k], 0.0);
    socEta_.assign(cone_.q.size(), 1.0);

    r1_.assign(n_, 0.0);
    r2_.assign(p_, 0.0);
    r3_.assign(m_, 0.0);
    u1_.assign(kktDim_, 0.0);
    u2_.assign(kktDim_, 0.0);
    rhs_.assign(kktDim_, 0.0);
    ds_.assign(m_, 0.0);
    dsAff_.assign(m_, 0.0);
    dzAff_.assign(m_, 0.0);
    work_.assign(kktDim_, 0.0);
    work2_.assign(kktDim_, 0.0);
    workm_.assign(m_, 0.0);
    workm2_.assign(m_, 0.0);
    dx_.assign(n_, 0.0);
    dy_.assign(p_, 0.0);
    dz_.assign(m_, 0.0);
    dsv_.assign(m_, 0.0);
}

void SocpSolver::buildKktPattern() {
    // KKT (upper triangle):
    //   [ 0   A'   G' ]
    //   [ .   0    0  ]
    //   [ .   .  -W'W ]
    // Triplet order matters: [A' | G' | W blocks]; buildCSC's map then lets us
    // refresh each group independently.
    Triplets T;
    const size_t nA = Acsc_.nnz(), nG = Gcsc_.nnz();
    T.ti.reserve(nA + nG + static_cast<size_t>(m_) * 2);
    T.tj.reserve(T.ti.capacity());
    T.tv.reserve(T.ti.capacity());

    // A' block: A is p x n in CSC; entry (r, c) of A -> KKT (c, n_+r)
    for (int c = 0; c < Acsc_.cols; ++c)
        for (int k = Acsc_.p[c]; k < Acsc_.p[c + 1]; ++k)
            T.add(c, n_ + Acsc_.i[k], 0.0);
    // G' block: entry (r, c) of G -> KKT (c, n_+p_+r)
    for (int c = 0; c < Gcsc_.cols; ++c)
        for (int k = Gcsc_.p[c]; k < Gcsc_.p[c + 1]; ++k)
            T.add(c, n_ + p_ + Gcsc_.i[k], 0.0);
    // W blocks: LP diagonal, then dense upper block per SOC (column-major order)
    const int z0 = n_ + p_;
    for (int i = 0; i < cone_.l; ++i) T.add(z0 + i, z0 + i, -1.0);
    int off = cone_.l;
    for (int qi : cone_.q) {
        for (int col = 0; col < qi; ++col)
            for (int row = 0; row <= col; ++row)
                T.add(z0 + off + row, z0 + off + col, (row == col) ? -1.0 : 0.0);
        off += qi;
    }

    std::vector<int> map;
    kkt_ = buildCSC(kktDim_, kktDim_, T, &map);

    kktAmap_.assign(nA, 0);
    kktGmap_.assign(nG, 0);
    size_t t = 0;
    for (size_t k = 0; k < nA; ++k) kktAmap_[k] = map[t++];
    for (size_t k = 0; k < nG; ++k) kktGmap_[k] = map[t++];
    kktWmap_.assign(map.begin() + t, map.end());
}

void SocpSolver::refreshKktValues() {
    // scatter current A and G values into the KKT (W blocks handled separately)
    for (size_t k = 0; k < kktAmap_.size(); ++k) kkt_.x[kktAmap_[k]] = 0.0;
    for (size_t k = 0; k < kktGmap_.size(); ++k) kkt_.x[kktGmap_[k]] = 0.0;
    size_t t = 0;
    for (int c = 0; c < Acsc_.cols; ++c)
        for (int k = Acsc_.p[c]; k < Acsc_.p[c + 1]; ++k)
            kkt_.x[kktAmap_[t++]] += Acsc_.x[k];
    t = 0;
    for (int c = 0; c < Gcsc_.cols; ++c)
        for (int k = Gcsc_.p[c]; k < Gcsc_.p[c + 1]; ++k)
            kkt_.x[kktGmap_[t++]] += Gcsc_.x[k];
}

// ---------------------------------------------------------------------------
// Cone operations
// ---------------------------------------------------------------------------
bool SocpSolver::inCone(const double* v, double margin) const {
    for (int i = 0; i < cone_.l; ++i)
        if (v[i] < margin) return false;
    int off = cone_.l;
    for (int qi : cone_.q) {
        if (v[off] - norm2(v + off + 1, qi - 1) < margin) return false;
        off += qi;
    }
    return true;
}

void SocpSolver::bringToCone(double* v) const {
    // shift v along the cone identity e so that it is strictly interior
    double margin = kInf;
    for (int i = 0; i < cone_.l; ++i) margin = std::min(margin, v[i]);
    int off = cone_.l;
    for (int qi : cone_.q) {
        margin = std::min(margin, v[off] - norm2(v + off + 1, qi - 1));
        off += qi;
    }
    if (margin <= 1e-8) {
        const double shift = 1.0 - margin;
        for (int i = 0; i < cone_.l; ++i) v[i] += shift;
        off = cone_.l;
        for (int qi : cone_.q) {
            v[off] += shift;
            off += qi;
        }
    }
}

bool SocpSolver::computeScaling() {
    // Nesterov-Todd scaling point W with W z = W^{-1} s = lambda
    for (int i = 0; i < cone_.l; ++i) {
        if (s_[i] <= 0.0 || z_[i] <= 0.0) return false;
        wLp_[i] = std::sqrt(s_[i] / z_[i]);
        lambda_[i] = std::sqrt(s_[i] * z_[i]);
    }
    int off = cone_.l;
    for (size_t k = 0; k < cone_.q.size(); ++k) {
        const int q = cone_.q[k];
        const double* sv = s_.data() + off;
        const double* zv = z_.data() + off;
        const double sJs = sv[0] * sv[0] - dot(sv + 1, sv + 1, q - 1);
        const double zJz = zv[0] * zv[0] - dot(zv + 1, zv + 1, q - 1);
        if (sJs <= 0.0 || zJz <= 0.0) return false;
        const double rho = std::sqrt(sJs), sig = std::sqrt(zJz);
        // normalized points
        Vec& w = socW_[k];
        const double sbar0 = sv[0] / rho, zbar0 = zv[0] / sig;
        double sbzb = sbar0 * zbar0;
        for (int j = 1; j < q; ++j) sbzb -= (sv[j] / rho) * (zv[j] / sig);
        // gamma^2 = (1 + sbar' zbar)/2 with the EUCLIDEAN inner product
        double sdotz = sbar0 * zbar0;
        for (int j = 1; j < q; ++j) sdotz += (sv[j] / rho) * (zv[j] / sig);
        const double gamma = std::sqrt(0.5 * (1.0 + sdotz));
        if (!(gamma > 0.0)) return false;
        // wbar = (sbar + J zbar) / (2 gamma)
        w[0] = (sbar0 + zbar0) / (2.0 * gamma);
        for (int j = 1; j < q; ++j) w[j] = (sv[j] / rho - zv[j] / sig) / (2.0 * gamma);
        socEta_[k] = std::sqrt(rho / sig);
        // lambda = sqrt(rho*sig) * (gamma ; ((gamma+zbar0) sbar1 + (gamma+sbar0) zbar1) / (sbar0+zbar0+2 gamma))
        const double scale = std::sqrt(rho * sig);
        const double den = sbar0 + zbar0 + 2.0 * gamma;
        lambda_[off] = scale * gamma;
        for (int j = 1; j < q; ++j) {
            lambda_[off + j] = scale * ((gamma + zbar0) * (sv[j] / rho) +
                                        (gamma + sbar0) * (zv[j] / sig)) / den;
        }
        (void)sbzb;
        off += q;
    }
    return true;
}

void SocpSolver::setKktScalingBlocks() {
    // write -W'W into the KKT cone block (same entry order as buildKktPattern)
    size_t t = 0;
    for (int i = 0; i < cone_.l; ++i) kkt_.x[kktWmap_[t++]] = -wLp_[i] * wLp_[i];
    for (size_t k = 0; k < cone_.q.size(); ++k) {
        const int q = cone_.q[k];
        const Vec& w = socW_[k];
        const double eta2 = socEta_[k] * socEta_[k];
        // W^2 = eta^2 (2 wbar wbar' - J)   (wbar'J wbar = 1)
        for (int col = 0; col < q; ++col) {
            for (int row = 0; row <= col; ++row) {
                double v = 2.0 * w[row] * w[col];
                if (row == col) v -= (row == 0) ? 1.0 : -1.0;
                kkt_.x[kktWmap_[t++]] = -eta2 * v;
            }
        }
    }
}

void SocpSolver::applyW(const double* v, double* out) const {
    for (int i = 0; i < cone_.l; ++i) out[i] = wLp_[i] * v[i];
    int off = cone_.l;
    for (size_t k = 0; k < cone_.q.size(); ++k) {
        const int q = cone_.q[k];
        const Vec& w = socW_[k];
        // Wbar = [ w0  w1' ; w1  I + w1 w1'/(1+w0) ]
        double w1v1 = 0.0;
        for (int j = 1; j < q; ++j) w1v1 += w[j] * v[off + j];
        const double v0 = v[off];
        out[off] = socEta_[k] * (w[0] * v0 + w1v1);
        const double coef = v0 + w1v1 / (1.0 + w[0]);
        for (int j = 1; j < q; ++j)
            out[off + j] = socEta_[k] * (v[off + j] + coef * w[j]);
        off += q;
    }
}

void SocpSolver::applyWinv(const double* v, double* out) const {
    for (int i = 0; i < cone_.l; ++i) out[i] = v[i] / wLp_[i];
    int off = cone_.l;
    for (size_t k = 0; k < cone_.q.size(); ++k) {
        const int q = cone_.q[k];
        const Vec& w = socW_[k];
        // W^{-1} = (1/eta) J Wbar J
        double w1v1 = 0.0;
        for (int j = 1; j < q; ++j) w1v1 += w[j] * v[off + j];
        const double v0 = v[off];
        out[off] = (w[0] * v0 - w1v1) / socEta_[k];
        const double coef = -v0 + w1v1 / (1.0 + w[0]);
        for (int j = 1; j < q; ++j)
            out[off + j] = (v[off + j] + coef * w[j]) / socEta_[k];
        off += q;
    }
}

void SocpSolver::jordanMul(const double* u, const double* v, double* out) const {
    for (int i = 0; i < cone_.l; ++i) out[i] = u[i] * v[i];
    int off = cone_.l;
    for (int q : cone_.q) {
        const double u0 = u[off], v0 = v[off];
        double uv = 0.0;
        for (int j = 0; j < q; ++j) uv += u[off + j] * v[off + j];
        for (int j = 1; j < q; ++j) out[off + j] = u0 * v[off + j] + v0 * u[off + j];
        out[off] = uv;
        off += q;
    }
}

bool SocpSolver::jordanDiv(const double* d, double* out) const {
    // solve lambda o out = d
    for (int i = 0; i < cone_.l; ++i) {
        if (lambda_[i] == 0.0) return false;
        out[i] = d[i] / lambda_[i];
    }
    int off = cone_.l;
    for (int q : cone_.q) {
        const double l0 = lambda_[off];
        double l1l1 = 0.0, l1d1 = 0.0;
        for (int j = 1; j < q; ++j) {
            l1l1 += lambda_[off + j] * lambda_[off + j];
            l1d1 += lambda_[off + j] * d[off + j];
        }
        const double det = l0 * l0 - l1l1;
        if (det == 0.0 || l0 == 0.0) return false;
        const double x0 = (l0 * d[off] - l1d1) / det;
        for (int j = 1; j < q; ++j) out[off + j] = (d[off + j] - x0 * lambda_[off + j]) / l0;
        out[off] = x0;
        off += q;
    }
    return true;
}

double SocpSolver::maxStep(const double* v, const double* dv) const {
    double alpha = kInf;
    for (int i = 0; i < cone_.l; ++i)
        if (dv[i] < 0.0) alpha = std::min(alpha, -v[i] / dv[i]);
    int off = cone_.l;
    for (int q : cone_.q) {
        // largest a with (v0+a d0)^2 - ||v1+a d1||^2 >= 0:
        // smallest positive root of  c + b a + a2 a^2 = 0
        const double v0 = v[off], d0 = dv[off];
        double d1d1 = 0.0, v1d1 = 0.0, v1v1 = 0.0;
        for (int j = 1; j < q; ++j) {
            d1d1 += dv[off + j] * dv[off + j];
            v1d1 += v[off + j] * dv[off + j];
            v1v1 += v[off + j] * v[off + j];
        }
        const double a2 = d0 * d0 - d1d1;
        const double b = 2.0 * (v0 * d0 - v1d1);
        const double c0 = v0 * v0 - v1v1;  // >= 0 at an interior point
        double root = kInf;
        if (std::fabs(a2) < 1e-14) {
            if (b < 0.0) root = -c0 / b;
        } else {
            const double disc = b * b - 4.0 * a2 * c0;
            if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                const double rA = (-b - sq) / (2.0 * a2);
                const double rB = (-b + sq) / (2.0 * a2);
                const double lo = std::min(rA, rB), hi = std::max(rA, rB);
                if (lo > 1e-14) root = lo;
                else if (hi > 1e-14 && a2 < 0.0) root = hi;
                else if (hi > 1e-14 && c0 <= 0.0) root = hi;
            } else if (a2 < 0.0) {
                root = 0.0;  // should not happen from an interior point
            }
        }
        alpha = std::min(alpha, root);
        off += q;
    }
    return alpha;
}

bool SocpSolver::solveKkt(const double* rx, const double* ry, const double* rz, double* u) {
    copy(rx, rhs_.data(), n_);
    copy(ry, rhs_.data() + n_, p_);
    copy(rz, rhs_.data() + n_ + p_, m_);
    const double res = ldl_.solveRefine(kkt_, rhs_.data(), u, set_.refineIters);
    const double scale = std::max(1.0, normInf(rhs_.data(), kktDim_));
    return std::isfinite(res) && res < 1e-4 * scale;
}

// ---------------------------------------------------------------------------
// Main solve
// ---------------------------------------------------------------------------
SocpStatus SocpSolver::solve(const SocpProblem& prob) {
    const auto t0 = std::chrono::steady_clock::now();
    auto finish = [&](SocpStatus st) {
        info_.status = st;
        info_.solveTimeMs = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
        return st;
    };

    // refresh numeric data
    scatterValues(Acsc_, prob.A, Amap_);
    scatterValues(Gcsc_, prob.G, Gmap_);
    b_ = prob.b;
    c_ = prob.c;
    h_ = prob.h;
    refreshKktValues();
    info_ = SocpInfo();

    const double bNorm = norm2(b_.data(), p_);
    const double cNorm = norm2(c_.data(), n_);
    const double hNorm = norm2(h_.data(), m_);

    // ---- initialization: least-squares primal/dual start (CVXOPT conelp style)
    // identity scaling
    for (int i = 0; i < cone_.l; ++i) wLp_[i] = 1.0;
    for (size_t k = 0; k < cone_.q.size(); ++k) {
        std::fill(socW_[k].begin(), socW_[k].end(), 0.0);
        socW_[k][0] = 1.0;   // wbar = e -> Wbar = I
        socEta_[k] = 1.0;
    }
    setKktScalingBlocks();
    if (!ldl_.factor(kkt_, set_.staticReg, set_.dynReg)) return finish(SocpStatus::NumericalError);

    // primal start: minimize ||s|| s.t. Ax=b, Gx+s=h  ->  rhs (0, b, h), s = -z_part
    setZero(work_.data(), kktDim_);
    copy(b_.data(), work_.data() + n_, p_);
    copy(h_.data(), work_.data() + n_ + p_, m_);
    if (!solveKkt(work_.data(), work_.data() + n_, work_.data() + n_ + p_, u1_.data()))
        return finish(SocpStatus::NumericalError);
    copy(u1_.data(), x_.data(), n_);
    for (int i = 0; i < m_; ++i) s_[i] = -u1_[n_ + p_ + i];
    bringToCone(s_.data());

    // dual start: minimize ||z|| s.t. A'y + G'z = -c  ->  rhs (-c, 0, 0)
    setZero(work_.data(), kktDim_);
    for (int i = 0; i < n_; ++i) work_[i] = -c_[i];
    if (!solveKkt(work_.data(), work_.data() + n_, work_.data() + n_ + p_, u1_.data()))
        return finish(SocpStatus::NumericalError);
    copy(u1_.data() + n_, y_.data(), p_);
    copy(u1_.data() + n_ + p_, z_.data(), m_);
    bringToCone(z_.data());

    tau_ = 1.0;
    kappa_ = 1.0;

    if (set_.verbose)
        std::printf("iter        pcost        dcost      presid      dresid         gap     step\n");

    double alpha = 0.0;
    for (int iter = 0; iter <= set_.maxIters; ++iter) {
        info_.iters = iter;

        // ---- residuals of the homogeneous embedding
        // r1 = A'y + G'z + c tau
        for (int i = 0; i < n_; ++i) r1_[i] = c_[i] * tau_;
        cscTMulAdd(Acsc_, y_.data(), r1_.data());
        cscTMulAdd(Gcsc_, z_.data(), r1_.data());
        // r2 = -A x + b tau
        for (int i = 0; i < p_; ++i) r2_[i] = b_[i] * tau_;
        cscMulAdd(Acsc_, x_.data(), r2_.data(), -1.0);
        // r3 = -G x + h tau - s
        for (int i = 0; i < m_; ++i) r3_[i] = h_[i] * tau_ - s_[i];
        cscMulAdd(Gcsc_, x_.data(), r3_.data(), -1.0);
        // r4 = -c'x - b'y - h'z - kappa
        const double cx = dot(c_.data(), x_.data(), n_);
        const double by = dot(b_.data(), y_.data(), p_);
        const double hz = dot(h_.data(), z_.data(), m_);
        r4_ = -cx - by - hz - kappa_;

        const double mu = (dot(s_.data(), z_.data(), m_) + tau_ * kappa_) / (coneDeg_ + 1);

        // ---- convergence metrics on the de-homogenized iterate
        const double invTau = 1.0 / tau_;
        // ||A x/tau - b|| / (1+||b||) and ||G x/tau + s/tau - h|| / (1+||h||)
        setZero(work_.data(), kktDim_);
        cscMulAdd(Acsc_, x_.data(), work_.data(), invTau);
        axpy(-1.0, b_.data(), work_.data(), p_);
        const double eqRes = norm2(work_.data(), p_) / (1.0 + bNorm);
        for (int i = 0; i < m_; ++i) work_[i] = s_[i] * invTau - h_[i];
        cscMulAdd(Gcsc_, x_.data(), work_.data(), invTau);
        const double coneRes = norm2(work_.data(), m_) / (1.0 + hNorm);
        const double presid = std::max(eqRes, coneRes);
        for (int i = 0; i < n_; ++i) work_[i] = c_[i];
        cscTMulAdd(Acsc_, y_.data(), work_.data(), invTau);
        cscTMulAdd(Gcsc_, z_.data(), work_.data(), invTau);
        const double dresid = norm2(work_.data(), n_) / (1.0 + cNorm);

        const double pcost = cx * invTau;
        const double dcost = -(by + hz) * invTau;
        const double gap = dot(s_.data(), z_.data(), m_) * invTau * invTau;
        const double relGap = gap / std::max({1.0, std::fabs(pcost), std::fabs(dcost)});

        info_.pcost = pcost; info_.dcost = dcost;
        info_.presid = presid; info_.dresid = dresid;
        info_.gap = gap; info_.relGap = relGap;

        if (set_.verbose)
            std::printf("%4d %12.5e %12.5e %11.4e %11.4e %11.4e %8.5f\n",
                        iter, pcost, dcost, presid, dresid, gap, alpha);

        if (presid <= set_.feasTol && dresid <= set_.feasTol &&
            (gap <= set_.absGapTol || relGap <= set_.relGapTol)) {
            scal(invTau, x_.data(), n_);
            scal(invTau, y_.data(), p_);
            scal(invTau, z_.data(), m_);
            scal(invTau, s_.data(), m_);
            return finish(SocpStatus::Optimal);
        }

        // ---- infeasibility certificates
        if (by + hz < -1e-10) {
            // y,z certify primal infeasibility if A'y + G'z ~= 0
            for (int i = 0; i < n_; ++i) work_[i] = 0.0;
            cscTMulAdd(Acsc_, y_.data(), work_.data());
            cscTMulAdd(Gcsc_, z_.data(), work_.data());
            const double certRes = norm2(work_.data(), n_) / std::max(1.0, cNorm);
            if (certRes <= set_.feasTol * (-(by + hz)) / std::max(1.0, hNorm + bNorm)) {
                const double sc = -1.0 / (by + hz);
                scal(sc, y_.data(), p_);
                scal(sc, z_.data(), m_);
                return finish(SocpStatus::PrimalInfeasible);
            }
        }
        if (cx < -1e-10) {
            // x,s certify dual infeasibility (primal unboundedness) if Ax~=0, Gx+s~=0
            setZero(work_.data(), kktDim_);
            cscMulAdd(Acsc_, x_.data(), work_.data());
            const double eq0 = norm2(work_.data(), p_);
            copy(s_.data(), work_.data(), m_);
            cscMulAdd(Gcsc_, x_.data(), work_.data());
            const double cone0 = norm2(work_.data(), m_);
            if (std::max(eq0, cone0) <= set_.feasTol * (-cx)) {
                const double sc = -1.0 / cx;
                scal(sc, x_.data(), n_);
                scal(sc, s_.data(), m_);
                return finish(SocpStatus::DualInfeasible);
            }
        }

        if (iter == set_.maxIters) {
            scal(invTau, x_.data(), n_);
            scal(invTau, y_.data(), p_);
            scal(invTau, z_.data(), m_);
            scal(invTau, s_.data(), m_);
            return finish(SocpStatus::MaxIterations);
        }

        // ---- NT scaling and KKT factorization
        if (!computeScaling()) return finish(SocpStatus::NumericalError);
        setKktScalingBlocks();
        if (!ldl_.factor(kkt_, set_.staticReg, set_.dynReg))
            return finish(SocpStatus::NumericalError);

        // u1 = S^{-1} (c, -b, -h)
        for (int i = 0; i < p_; ++i) work2_[i] = -b_[i];
        for (int i = 0; i < m_; ++i) work2_[p_ + i] = -h_[i];
        if (!solveKkt(c_.data(), work2_.data(), work2_.data() + p_, u1_.data()))
            return finish(SocpStatus::NumericalError);
        const double phi1 = dot(c_.data(), u1_.data(), n_) +
                            dot(b_.data(), u1_.data() + n_, p_) +
                            dot(h_.data(), u1_.data() + n_ + p_, m_);
        const double kot = kappa_ / tau_;
        const double denom = phi1 + kot;
        if (!(std::fabs(denom) > 1e-300)) return finish(SocpStatus::NumericalError);

        // ---- affine (predictor) direction: ds = -lambda o lambda, dkappa = -tau*kappa
        // dtilde3 = -r3 + W(lambda \ ds) = -r3 - s   (since W lambda = s)
        // Solve S u2 = (d1, -d2, -dtilde3) = (-r1, r2, r3 + s)
        for (int i = 0; i < p_; ++i) work2_[i] = r2_[i];
        for (int i = 0; i < m_; ++i) work2_[p_ + i] = r3_[i] + s_[i];
        for (int i = 0; i < n_; ++i) work_[i] = -r1_[i];
        if (!solveKkt(work_.data(), work2_.data(), work2_.data() + p_, u2_.data()))
            return finish(SocpStatus::NumericalError);
        double phi2 = dot(c_.data(), u2_.data(), n_) +
                      dot(b_.data(), u2_.data() + n_, p_) +
                      dot(h_.data(), u2_.data() + n_ + p_, m_);
        // dtilde4 = d4 + dkappa/tau = -r4 - kappa
        double dtau = (-r4_ - kappa_ + phi2) / denom;
        for (int i = 0; i < n_; ++i) dx_[i] = u2_[i] - dtau * u1_[i];
        for (int i = 0; i < p_; ++i) dy_[i] = u2_[n_ + i] - dtau * u1_[n_ + i];
        for (int i = 0; i < m_; ++i) dz_[i] = u2_[n_ + p_ + i] - dtau * u1_[n_ + p_ + i];
        // ds = W(lambda \ d_s) - W^2 dz = -s - W(W dz)
        applyW(dz_.data(), workm_.data());
        applyW(workm_.data(), dsv_.data());
        for (int i = 0; i < m_; ++i) dsv_[i] = -s_[i] - dsv_[i];
        double dkappa = -kappa_ - kot * dtau;

        // affine step length
        double aAff = std::min(maxStep(s_.data(), dsv_.data()),
                               maxStep(z_.data(), dz_.data()));
        if (dtau < 0.0) aAff = std::min(aAff, -tau_ / dtau);
        if (dkappa < 0.0) aAff = std::min(aAff, -kappa_ / dkappa);
        aAff = std::min(aAff, 1.0);

        // save affine cone directions for the corrector
        applyWinv(dsv_.data(), dsAff_.data());   // W^{-1} ds_aff
        applyW(dz_.data(), dzAff_.data());       // W dz_aff
        const double dtauAff = dtau, dkappaAff = dkappa;

        // ---- centering + corrector
        const double sigma = std::min(1.0, std::max(0.0, std::pow(1.0 - aAff, 3)));
        const double oneMinusSigma = 1.0 - sigma;

        // d_s = -lambda o lambda - (W^{-1}ds_aff)o(W dz_aff) + sigma mu e
        jordanMul(lambda_.data(), lambda_.data(), ds_.data());
        jordanMul(dsAff_.data(), dzAff_.data(), workm_.data());
        for (int i = 0; i < m_; ++i) ds_[i] = -ds_[i] - workm_[i];
        for (int i = 0; i < cone_.l; ++i) ds_[i] += sigma * mu;
        {
            int off = cone_.l;
            for (int q : cone_.q) { ds_[off] += sigma * mu; off += q; }
        }
        const double dkapRhs = -tau_ * kappa_ - dtauAff * dkappaAff + sigma * mu;

        // dtilde3 = -(1-sigma) r3 + W(lambda \ d_s)
        if (!jordanDiv(ds_.data(), workm_.data())) return finish(SocpStatus::NumericalError);
        applyW(workm_.data(), workm2_.data());
        for (int i = 0; i < n_; ++i) work_[i] = -oneMinusSigma * r1_[i];
        for (int i = 0; i < p_; ++i) work2_[i] = oneMinusSigma * r2_[i];
        for (int i = 0; i < m_; ++i)
            work2_[p_ + i] = oneMinusSigma * r3_[i] - workm2_[i];
        if (!solveKkt(work_.data(), work2_.data(), work2_.data() + p_, u2_.data()))
            return finish(SocpStatus::NumericalError);
        phi2 = dot(c_.data(), u2_.data(), n_) +
               dot(b_.data(), u2_.data() + n_, p_) +
               dot(h_.data(), u2_.data() + n_ + p_, m_);
        dtau = (-oneMinusSigma * r4_ + dkapRhs / tau_ + phi2) / denom;
        for (int i = 0; i < n_; ++i) dx_[i] = u2_[i] - dtau * u1_[i];
        for (int i = 0; i < p_; ++i) dy_[i] = u2_[n_ + i] - dtau * u1_[n_ + i];
        for (int i = 0; i < m_; ++i) dz_[i] = u2_[n_ + p_ + i] - dtau * u1_[n_ + p_ + i];
        // ds = W(lambda\d_s) - W^2 dz
        applyW(dz_.data(), workm_.data());
        applyW(workm_.data(), dsv_.data());
        for (int i = 0; i < m_; ++i) dsv_[i] = workm2_[i] - dsv_[i];
        dkappa = (dkapRhs - kappa_ * dtau) / tau_;

        // ---- step
        alpha = std::min(maxStep(s_.data(), dsv_.data()),
                         maxStep(z_.data(), dz_.data()));
        if (dtau < 0.0) alpha = std::min(alpha, -tau_ / dtau);
        if (dkappa < 0.0) alpha = std::min(alpha, -kappa_ / dkappa);
        alpha = std::min(1.0, set_.gamma * alpha);
        if (alpha < set_.minStep) {
            scal(invTau, x_.data(), n_);
            scal(invTau, y_.data(), p_);
            scal(invTau, z_.data(), m_);
            scal(invTau, s_.data(), m_);
            return finish(SocpStatus::NumericalError);
        }

        axpy(alpha, dx_.data(), x_.data(), n_);
        axpy(alpha, dy_.data(), y_.data(), p_);
        axpy(alpha, dz_.data(), z_.data(), m_);
        axpy(alpha, dsv_.data(), s_.data(), m_);
        tau_ += alpha * dtau;
        kappa_ += alpha * dkappa;
        if (tau_ <= 0.0 || kappa_ < 0.0 || !std::isfinite(tau_))
            return finish(SocpStatus::NumericalError);
    }
    return finish(SocpStatus::NumericalError);  // unreachable
}

size_t SocpSolver::workspaceBytes() const {
    size_t bytes = ldl_.workspaceBytes();
    bytes += kkt_.x.size() * sizeof(double) + (kkt_.p.size() + kkt_.i.size()) * sizeof(int);
    bytes += (Acsc_.x.size() + Gcsc_.x.size()) * sizeof(double);
    auto vb = [](const Vec& v) { return v.size() * sizeof(double); };
    bytes += vb(x_) + vb(y_) + vb(z_) + vb(s_) + vb(lambda_) + vb(r1_) + vb(r2_) + vb(r3_) +
             vb(u1_) + vb(u2_) + vb(rhs_) + vb(ds_) + vb(dsAff_) + vb(dzAff_) + vb(work_) +
             vb(work2_) + vb(workm_) + vb(workm2_) + vb(dx_) + vb(dy_) + vb(dz_) + vb(dsv_);
    return bytes;
}

}  // namespace pdg

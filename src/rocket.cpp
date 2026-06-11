#include "pdg/rocket.hpp"

#include <cmath>

namespace pdg {

namespace {
inline void cross(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
}  // namespace

void quatNormalize(double* q) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 0.0)
        for (int i = 0; i < 4; ++i) q[i] /= n;
}

void quatToDcm(const double* q, Mat& C) {
    // homogeneous form: C = (q0^2 - qv.qv) I + 2 qv qv' + 2 q0 [qv]x
    const double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    const double s = q0 * q0 - (q1 * q1 + q2 * q2 + q3 * q3);
    C(0, 0) = s + 2.0 * q1 * q1;
    C(0, 1) = 2.0 * (q1 * q2 - q0 * q3);
    C(0, 2) = 2.0 * (q1 * q3 + q0 * q2);
    C(1, 0) = 2.0 * (q1 * q2 + q0 * q3);
    C(1, 1) = s + 2.0 * q2 * q2;
    C(1, 2) = 2.0 * (q2 * q3 - q0 * q1);
    C(2, 0) = 2.0 * (q1 * q3 - q0 * q2);
    C(2, 1) = 2.0 * (q2 * q3 + q0 * q1);
    C(2, 2) = s + 2.0 * q3 * q3;
}

void quatMul(const double* a, const double* b, double* out) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

void quatSlerp(const double* a, const double* b, double t, double* out) {
    double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    double sign = 1.0;
    if (d < 0.0) { d = -d; sign = -1.0; }
    if (d > 0.9995) {
        for (int i = 0; i < 4; ++i) out[i] = (1.0 - t) * a[i] + t * sign * b[i];
        quatNormalize(out);
        return;
    }
    const double th = std::acos(d);
    const double sa = std::sin((1.0 - t) * th) / std::sin(th);
    const double sb = sign * std::sin(t * th) / std::sin(th);
    for (int i = 0; i < 4; ++i) out[i] = sa * a[i] + sb * b[i];
    quatNormalize(out);
}

void quatRotate(const double* q, const double* vB, double* vI) {
    // homogeneous form (exact rotation for unit q, smooth in all of R^4 — the
    // form whose analytic Jacobians rocketJacobians() implements):
    // vI = (q0^2 - qv.qv) vB + 2 qv (qv.vB) + 2 q0 (qv x vB)
    const double q0 = q[0];
    const double* qv = q + 1;
    const double qq = qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2];
    const double qvv = qv[0] * vB[0] + qv[1] * vB[1] + qv[2] * vB[2];
    double t[3];
    cross(qv, vB, t);
    for (int i = 0; i < 3; ++i)
        vI[i] = (q0 * q0 - qq) * vB[i] + 2.0 * qv[i] * qvv + 2.0 * q0 * t[i];
}

double quatTiltCos(const double* q) {
    return 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
}

void rocketDynamics(const RocketParams& p, const double* x, const double* u, double* f) {
    const double m = x[iM];
    const double* v = x + iV;
    const double* q = x + iQ;
    const double* w = x + iW;

    const double uNorm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    f[iM] = -p.alpha() * uNorm;

    f[iR + 0] = v[0];
    f[iR + 1] = v[1];
    f[iR + 2] = v[2];

    double uI[3];
    quatRotate(q, u, uI);
    for (int i = 0; i < 3; ++i) f[iV + i] = uI[i] / m + p.g[i];

    // qdot = 1/2 (-qv.w ; q0 w + qv x w)
    double qvxw[3];
    cross(q + 1, w, qvxw);
    f[iQ + 0] = 0.5 * (-(q[1] * w[0] + q[2] * w[1] + q[3] * w[2]));
    for (int i = 0; i < 3; ++i) f[iQ + 1 + i] = 0.5 * (q[0] * w[i] + qvxw[i]);

    // wdot = J^{-1} ( rT x u - w x (J w) )
    double tq[3], Jw[3], wxJw[3];
    cross(p.rT.data(), u, tq);
    for (int i = 0; i < 3; ++i) Jw[i] = p.J[i] * w[i];
    cross(w, Jw, wxJw);
    for (int i = 0; i < 3; ++i) f[iW + i] = (tq[i] - wxJw[i]) / p.J[i];
}

void rocketJacobians(const RocketParams& p, const double* x, const double* u, Mat& A, Mat& B) {
    A.setZero();
    B.setZero();
    const double m = x[iM];
    const double* q = x + iQ;
    const double* w = x + iW;
    const double q0 = q[0];
    const double* qv = q + 1;

    // dm/du = -alpha u'/||u||
    const double uNorm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (uNorm > 1e-12)
        for (int j = 0; j < 3; ++j) B(iM, j) = -p.alpha() * u[j] / uNorm;

    // dr/dv = I
    for (int i = 0; i < 3; ++i) A(iR + i, iV + i) = 1.0;

    // v block
    double uI[3];
    quatRotate(q, u, uI);
    // dv/dm = -C u / m^2
    for (int i = 0; i < 3; ++i) A(iV + i, iM) = -uI[i] / (m * m);
    // dv/dq: d(Cu)/dq0 = 2 (q0 u + qv x u); d(Cu)/dqv = 2(-u qv' + (qv'u) I + qv u' - q0 [u]x)
    {
        double qvxu[3];
        cross(qv, u, qvxu);
        for (int i = 0; i < 3; ++i)
            A(iV + i, iQ + 0) = 2.0 * (q0 * u[i] + qvxu[i]) / m;
        const double qvu = qv[0] * u[0] + qv[1] * u[1] + qv[2] * u[2];
        // [u]x
        const double ux[3][3] = {{0, -u[2], u[1]}, {u[2], 0, -u[0]}, {-u[1], u[0], 0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double dij = -u[i] * qv[j] + qv[i] * u[j] - q0 * ux[i][j];
                if (i == j) dij += qvu;
                A(iV + i, iQ + 1 + j) = 2.0 * dij / m;
            }
        // dv/du = C/m
        Mat C(3, 3);
        quatToDcm(q, C);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) B(iV + i, j) = C(i, j) / m;
    }

    // q block: dq/dq = 1/2 [0 -w'; w -[w]x],  dq/dw = 1/2 [-qv'; q0 I + [qv]x]
    {
        const double wx[3][3] = {{0, -w[2], w[1]}, {w[2], 0, -w[0]}, {-w[1], w[0], 0}};
        for (int j = 0; j < 3; ++j) A(iQ + 0, iQ + 1 + j) = -0.5 * w[j];
        for (int i = 0; i < 3; ++i) {
            A(iQ + 1 + i, iQ + 0) = 0.5 * w[i];
            for (int j = 0; j < 3; ++j) A(iQ + 1 + i, iQ + 1 + j) = -0.5 * wx[i][j];
        }
        const double qx[3][3] = {{0, -qv[2], qv[1]}, {qv[2], 0, -qv[0]}, {-qv[1], qv[0], 0}};
        for (int j = 0; j < 3; ++j) A(iQ + 0, iW + j) = -0.5 * qv[j];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A(iQ + 1 + i, iW + j) = 0.5 * ((i == j ? q0 : 0.0) + qx[i][j]);
    }

    // w block: dw/dw = J^{-1}(-[w]x J + [Jw]x), dw/du = J^{-1} [rT]x
    {
        double Jw[3];
        for (int i = 0; i < 3; ++i) Jw[i] = p.J[i] * w[i];
        const double wx[3][3] = {{0, -w[2], w[1]}, {w[2], 0, -w[0]}, {-w[1], w[0], 0}};
        const double Jwx[3][3] = {{0, -Jw[2], Jw[1]}, {Jw[2], 0, -Jw[0]}, {-Jw[1], Jw[0], 0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A(iW + i, iW + j) = (-wx[i][j] * p.J[j] + Jwx[i][j]) / p.J[i];
        const double* rT = p.rT.data();
        const double rx[3][3] = {{0, -rT[2], rT[1]}, {rT[2], 0, -rT[0]}, {-rT[1], rT[0], 0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                B(iW + i, j) = rx[i][j] / p.J[i];
    }
}

}  // namespace pdg

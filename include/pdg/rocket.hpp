// pdg::rocket — shared 6-DoF rigid-body rocket model used by the SCvx guidance
// (pdg/scvx.hpp) and the simulator (pdg/sim.hpp).
//
// State (dimension 14), inertial frame z-up, body frame z along the long axis:
//   x = [ m,  r_I (3),  v_I (3),  q_B->I (4, scalar first),  omega_B (3) ]
// Control (dimension 3): u = T_B, thrust vector in the body frame.
//
// Dynamics:
//   mdot = -||u|| / (Isp g0)
//   rdot = v
//   vdot = C(q) u / m + g
//   qdot = 1/2 q  (0, omega)               (quaternion product)
//   wdot = J^{-1} ( rT x u - omega x J omega )
#pragma once

#include <array>

#include "pdg/linalg.hpp"

namespace pdg {

constexpr int kNx = 14;   // state dimension
constexpr int kNu = 3;    // control dimension

using Vec3 = std::array<double, 3>;
using Vec4 = std::array<double, 4>;
using State = std::array<double, kNx>;

// state component offsets
constexpr int iM = 0, iR = 1, iV = 4, iQ = 7, iW = 11;

struct RocketParams {
    // defaults: a Falcon-like single-engine landing stage with wet T/W ~ 1.7
    double mWet = 15000.0;            // [kg]
    double mDry = 10000.0;            // [kg]
    double Isp  = 300.0;              // [s]
    double Tmin = 80e3;               // [N]
    double Tmax = 250e3;              // [N]
    double g0   = 9.80665;
    Vec3 g = {0.0, 0.0, -9.80665};    // inertial gravity
    Vec3 J = {8.0e5, 8.0e5, 1.0e5};   // principal inertia [kg m^2] (body axes)
    Vec3 rT = {0.0, 0.0, -12.0};      // thrust application point in body frame [m]

    double alpha() const { return 1.0 / (Isp * g0); }
};

// ---------------------------------------------------------------------------
// quaternion utilities (scalar-first, q maps body -> inertial)
// ---------------------------------------------------------------------------
void quatNormalize(double* q);
void quatToDcm(const double* q, Mat& C);          // 3x3, x_I = C x_B
void quatMul(const double* a, const double* b, double* out);
void quatSlerp(const double* a, const double* b, double t, double* out);
// rotate body vector into inertial frame without forming C
void quatRotate(const double* q, const double* vB, double* vI);

// tilt angle from vertical: cos(tilt) = 1 - 2(q1^2 + q2^2)
double quatTiltCos(const double* q);

// ---------------------------------------------------------------------------
// dynamics and Jacobians
// ---------------------------------------------------------------------------
// f = dx/dt
void rocketDynamics(const RocketParams& p, const double* x, const double* u, double* f);

// A = df/dx (14x14), B = df/du (14x3); matrices must be pre-sized
void rocketJacobians(const RocketParams& p, const double* x, const double* u, Mat& A, Mat& B);

}  // namespace pdg

// pdg::sim: minimal 6-DoF rigid-body rocket simulator for closing the loop
// around the guidance algorithms.
//
// The simulator integrates the same nonlinear dynamics as pdg::scvx (RK4,
// fixed step) but with a *dispersed* plant: thrust magnitude scale, Isp error,
// dry-mass error, thrust misalignment, and an optional first-order actuator
// lag on the commanded thrust vector. Guidance commands are followed with a
// first-order hold between trajectory nodes.
#pragma once

#include <functional>

#include "pdg/rocket.hpp"
#include "pdg/scvx.hpp"

namespace pdg {

struct PlantDispersions {
    double thrustScale = 1.0;     // actual thrust = scale * commanded
    double ispScale = 1.0;        // actual Isp = scale * nominal
    double thrustMisalignRad = 0.0;  // fixed cant of the engine [rad]
    double misalignAzimuthRad = 0.0; // direction of the cant in the body x-y plane
    double actuatorTau = 0.0;     // first-order lag on T_B command [s] (0 = ideal)
};

struct TouchdownState {
    bool landed = false;          // crossed z=0 with valid interpolation
    double time = 0.0;
    double lateralError = 0.0;    // horizontal distance from target [m]
    double verticalSpeed = 0.0;   // descent rate at touchdown [m/s] (positive down)
    double lateralSpeed = 0.0;    // horizontal speed at touchdown [m/s]
    double tiltDeg = 0.0;         // attitude tilt from vertical [deg]
    double rateDegS = 0.0;        // body rate magnitude [deg/s]
    double propellantUsed = 0.0;  // [kg]
    State finalState{};
};

// PD trajectory-tracking autopilot (guidance + control loop closure):
// translational PD correction on the planned inertial thrust, attitude PD loop
// realized through the thrust vector (TVC). Roll is left uncontrolled, since a
// single gimbaled engine has no roll authority.
struct TrackingGains {
    double posP = 0.35;          // [1/s^2] position correction
    double posD = 1.30;          // [1/s]   velocity correction
    double maxCorrAccel = 4.0;   // [m/s^2] clamp on the translational correction
    double attP = 9.0;           // [1/s^2] attitude loop
    double attD = 8.0;           // [1/s]   rate damping
};

class TrajectoryTracker {
public:
    TrajectoryTracker(const RocketParams& nominal, const ScvxSolution& ref,
                      const TrackingGains& gains = TrackingGains());

    // body-frame thrust command given current time and (true) state
    Vec3 command(double t, const State& x) const;

private:
    RocketParams prm_;
    const ScvxSolution& ref_;
    TrackingGains g_;

    void refAt(double t, State& xRef, Vec3& uRefB) const;
};

class RocketSim {
public:
    RocketSim(const RocketParams& nominal, const PlantDispersions& d = PlantDispersions());

    // Fly a guidance trajectory open loop (FOH on the commanded body thrust),
    // integrating until touchdown (z <= 0) or until tMax. dt is the integrator step.
    TouchdownState flyTrajectory(const State& x0, const ScvxSolution& guidance,
                                 double dt = 0.01, double tMax = 120.0);

    // Fly with the tracking autopilot closed around a guidance trajectory.
    TouchdownState flyTracked(const State& x0, const TrajectoryTracker& tracker,
                              double dt = 0.01, double tMax = 120.0);

    // Generic interface: command callback u_B(t, state). Integrates one run.
    TouchdownState fly(const State& x0,
                       const std::function<Vec3(double, const State&)>& command,
                       double dt = 0.01, double tMax = 120.0);

private:
    RocketParams plant_;          // dispersed parameters used for propagation
    PlantDispersions disp_;
    Mat misalign_;                // 3x3 rotation applied to commanded thrust

    void step(State& x, const Vec3& cmd0, const Vec3& cmdMid, const Vec3& cmd1,
              Vec3& uAct, double dt) const;
};

}  // namespace pdg

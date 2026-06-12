#!/usr/bin/env python3
"""Plot the CSV outputs of the pdg examples (requires matplotlib + pandas).

Usage: python plot_results.py [directory containing the CSVs]
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

base = sys.argv[1] if len(sys.argv) > 1 else "."


def maybe(path):
    p = os.path.join(base, path)
    return pd.read_csv(p) if os.path.exists(p) else None


tdof = maybe("ex_3dof_mars.csv")
if tdof is not None:
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot(tdof.rx, tdof.ry, tdof.rz)
    ax.set_title("3-DoF LCvx trajectory")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax2 = fig.add_subplot(132)
    ax2.plot(tdof.t, tdof.Tmag / 1e3)
    ax2.set_title("thrust magnitude [kN]"); ax2.set_xlabel("t [s]"); ax2.grid(True)
    ax3 = fig.add_subplot(133)
    ax3.plot(tdof.t, tdof.rz)
    ax3.set_title("altitude [m]"); ax3.set_xlabel("t [s]"); ax3.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(base, "ex_3dof_mars.png"), dpi=130)
    print("wrote ex_3dof_mars.png")

sdof = maybe("ex_6dof_landing.csv")
if sdof is not None:
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot(sdof.rx, sdof.ry, sdof.rz)
    ax.set_title("6-DoF SCvx trajectory")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax2 = fig.add_subplot(132)
    ax2.plot(sdof.t, sdof.Tmag / 1e3, label="|T| [kN]")
    ax2.plot(sdof.t, sdof.tilt_deg, label="tilt [deg]")
    ax2.legend(); ax2.set_xlabel("t [s]"); ax2.grid(True)
    ax3 = fig.add_subplot(133)
    ax3.plot(sdof.t, sdof.speed)
    ax3.set_title("speed [m/s]"); ax3.set_xlabel("t [s]"); ax3.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(base, "ex_6dof_landing.png"), dpi=130)
    print("wrote ex_6dof_landing.png")

mc = maybe("ex_monte_carlo.csv")
if mc is not None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ok = mc[mc.success == 1]
    bad = mc[mc.success == 0]
    axes[0].scatter(ok.lateral_error_m, ok.vertical_speed_ms, s=12, label="success")
    if len(bad):
        axes[0].scatter(bad.lateral_error_m, bad.vertical_speed_ms, s=12, c="r", label="fail")
    axes[0].set_xlabel("lateral error [m]"); axes[0].set_ylabel("descent speed [m/s]")
    axes[0].legend(); axes[0].grid(True)
    axes[1].hist(mc.fuel_kg, bins=20)
    axes[1].set_xlabel("fuel [kg]"); axes[1].grid(True)
    axes[2].hist(mc.tilt_deg, bins=20)
    axes[2].set_xlabel("touchdown tilt [deg]"); axes[2].grid(True)
    fig.suptitle("Monte Carlo touchdown dispersion")
    fig.tight_layout()
    fig.savefig(os.path.join(base, "ex_monte_carlo.png"), dpi=130)
    print("wrote ex_monte_carlo.png")

# Parts A and B on one plot: scattered photon energy vs angle, each part with its own fit

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

THETA0 = -2.614  # deg
THETA_READ_ERR = 1 / np.sqrt(12)  # deg
E_TRUE, MEC2_TRUE = 661.657, 510.999  # keV
PARTS = [
    ("Part A, aluminum rod", "scattering_al_results.csv", [], "C0"),
    ("Part B, plastic scintillator", "scattering_plastic_results.csv", [10, -10, 20, -20], "C3"),
]

def line(x, b, m):
    return b + m * x

def fit_part(file, exclude):
    rows = [r for r in csv.DictReader(open(file))
            if int(float(r["reading_deg"])) not in exclude and float(r["reading_deg"]) < 0]
    reading = np.array([float(r["reading_deg"]) for r in rows])
    E_meas = np.array([float(r["E_meas_keV"]) for r in rows]) / 1000  # MeV
    E_err = np.array([float(r["E_meas_err"]) for r in rows]) / 1000  # MeV
    theta = np.abs(reading - THETA0)
    x = 1 - np.cos(np.radians(theta))
    sx = np.sin(np.radians(theta)) * np.radians(THETA_READ_ERR)
    y = 1 / E_meas
    sy = E_err / E_meas**2
    p = np.polyfit(x, y, 1)[::-1]
    for i in range(10):
        sig = np.sqrt(sy**2 + (p[1] * sx) ** 2)
        p, cov = curve_fit(line, x, y, p0=p, sigma=sig, absolute_sigma=True)
    resid = y - line(x, *p)
    chi2 = np.sum((resid / sig) ** 2)
    dof = len(x) - 2
    db, dm = np.sqrt(np.diag(cov))
    return dict(reading=reading, theta=theta, E=E_meas, E_err=E_err, x=x, sx=sx, y=y, sy=sy, sig=sig, p=p,
                resid=resid, chi2_red=chi2 / dof, dof=dof, p_value=stats.chi2.sf(chi2, dof),
                mec2=1000 / p[1], dmec2=1000 * dm / p[1] ** 2, E0=1000 / p[0], dE0=1000 * db / p[0] ** 2)

results = [(name, color, fit_part(file, exclude)) for name, file, exclude, color in PARTS]
for name, color, r in results:
    print(f"{name}: m_e c^2 = {r['mec2']:.2f} +/- {r['dmec2']:.2f} keV, E = {r['E0']:.2f} +/- {r['dE0']:.2f} keV, "
          f"N_sigma(m_e c^2) = {(r['mec2'] - MEC2_TRUE) / r['dmec2']:+.2f}, "
          f"N_sigma(E) = {(r['E0'] - E_TRUE) / r['dE0']:+.2f}, "
          f"chi^2_red = {r['chi2_red']:.2f}, dof = {r['dof']}, p-value = {r['p_value']:.3g}")

os.makedirs("plots/scattering", exist_ok=True)
SIDES = ((False, "s", "negative side"),)
box = "\n".join(f"{name.split(',')[0]}: $\\chi^2_{{\\mathrm{{red}}}}$ = {r['chi2_red']:.2f},  p = {r['p_value']:.2g}"
                for name, color, r in results)

# plot 1
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
xmax = max(r["x"].max() for _, _, r in results) * 1.05
xx = np.linspace(0, xmax, 200)
ax.plot(xx, 1000 / E_TRUE + xx * 1000 / MEC2_TRUE, color="0.6", ls="--", lw=1, label="Compton, $m_ec^2$ = 511 keV")
for name, color, r in results:
    ax.plot(xx, line(xx, *r["p"]), color=color, lw=1,
            label=f"Fit, {name.split(',')[0]}: $m_ec^2$ = {r['mec2']:.1f} $\\pm$ {r['dmec2']:.1f} keV")
    pos = r["reading"] > 0
    for side, mk, lab in SIDES:
        sel = pos if side else ~pos
        ax.errorbar(r["x"][sel], r["y"][sel], xerr=r["sx"][sel], yerr=r["sy"][sel], fmt=mk, color=color,
                    capsize=3, label=name)
        axr.errorbar(r["x"][sel], r["resid"][sel], yerr=r["sig"][sel], fmt=mk, color=color,
                     capsize=3)
ax.text(0.02, 0.97, box, transform=ax.transAxes, va="top", bbox=dict(fc="white", ec="0.8"))
ax.set_ylabel(r"$1/E'$ (1/MeV)")
ax.set_title("Scattered photon energy, parts A and B")
ax.legend(loc="lower right", fontsize=7.5)
ax.grid(alpha=0.3)
axr.axhline(0, color="k", ls="--", lw=1)
axr.set_xlabel(r"$1 - \cos\theta$")
axr.set_ylabel("Residual (1/MeV)")
axr.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/scattering/energy_angle_fit_combined.png", dpi=200)
plt.close(fig)

# plot 2
fig, ax = plt.subplots(figsize=(7.5, 5))
tmax = max(140, max(r["theta"].max() for _, _, r in results) + 8)
tt = np.linspace(0, tmax, 300)
xt = 1 - np.cos(np.radians(tt))
ax.plot(tt, E_TRUE / (1 + E_TRUE / MEC2_TRUE * xt), color="0.6", ls="--", lw=1, label="Compton, $m_ec^2$ = 511 keV")
for name, color, r in results:
    ax.plot(tt, 1000 / line(xt, *r["p"]), color=color, lw=1,
            label=f"Fit, {name.split(',')[0]}: $m_ec^2$ = {r['mec2']:.1f} $\\pm$ {r['dmec2']:.1f} keV")
    pos = r["reading"] > 0
    for side, mk, lab in SIDES:
        sel = pos if side else ~pos
        ax.errorbar(r["theta"][sel], 1000 * r["E"][sel], xerr=THETA_READ_ERR, yerr=1000 * r["E_err"][sel], fmt=mk,
                    color=color, capsize=3, label=name)
ax.text(0.02, 0.04, box, transform=ax.transAxes, va="bottom", bbox=dict(fc="white", ec="0.8"))
ax.set_xlabel(r"Scattering angle $\theta$ (deg)")
ax.set_ylabel(r"$E'$ (keV)")
ax.set_title("Scattered photon energy vs angle, parts A and B")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/scattering/energy_vs_angle_combined.png", dpi=200)
print("\nSaved plots/scattering/energy_angle_fit_combined.png and plots/scattering/energy_vs_angle_combined.png")

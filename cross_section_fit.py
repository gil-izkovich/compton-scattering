# Differential cross section vs angle: Thomson vs Klein-Nishina (polar plot)

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

PART = 1  # 1: aluminum rod, 2: plastic scintillator
if PART == 1:
    FILE, TAG, TARGET, EXCLUDE = "scattering_al_results.csv", "", "aluminum rod", []
else:
    FILE, TAG, TARGET, EXCLUDE = "scattering_plastic_results.csv", "_plastic", "plastic scintillator", [10, -10, 20, -20]
THETA0 = -2.614  # deg
THETA_READ_ERR = 1 / np.sqrt(12)    
E0, MEC2 = 661.657, 510.999  # keV
D_CM, R_CM = 15.0, 4.0
D_ERR_CM = 1 / np.sqrt(12)
R_ERR_CM = 1 / np.sqrt(12) / 2  # half of the diameter error
OMEGA = np.pi * R_CM**2 / D_CM**2  # sr
OMEGA_REL_ERR = np.sqrt((2 * R_ERR_CM / R_CM) ** 2 + (2 * D_ERR_CM / D_CM) ** 2)

# negative side
rows = [r for r in csv.DictReader(open(FILE))
        if int(float(r["reading_deg"])) not in EXCLUDE and float(r["reading_deg"]) < 0]
reading = np.array([float(r["reading_deg"]) for r in rows])
theta = np.abs(reading - THETA0)
area = np.array([float(r["area"]) for r in rows])
area_err = np.array([float(r["area_err"]) for r in rows])
live = np.array([float(r["live_s" if PART == 1 else "real_s"]) for r in rows])
E_meas = np.array([float(r["E_meas_keV"]) for r in rows])

# efficiency at E' (from efficiency_calibration_fit.py)
C_EFF = np.array([-1.04051728, -0.6130298, -0.22412335])
C_EFF_COV = np.array([[0.00312859, -0.00818546, -0.00554359],
                      [-0.00818546, 0.0304425, 0.01951217],
                      [-0.00554359, 0.01951217, 0.01271371]])
eps = []
eps_rel_err = []
for e in E_meas:
    x = np.log(e / 500.0)
    j = np.array([1, x, x**2])
    eps.append(np.exp(j @ C_EFF))
    eps_rel_err.append(np.sqrt(j @ C_EFF_COV @ j))
eps = np.array(eps)
eps_rel_err = np.array(eps_rel_err)

rate = area / live
y = rate / (OMEGA * eps)  # 1/(s sr)
sy = y * np.sqrt((area_err / area) ** 2 + eps_rel_err**2)

# models
R_E2 = (2.8179403262e-13) ** 2 / 1e-24

def thomson(t):
    c = np.cos(np.radians(t))
    return R_E2 * 0.5 * (1 + c**2)

def klein_nishina(t):
    c = np.cos(np.radians(t))
    r = 1 / (1 + E0 / MEC2 * (1 - c))
    return R_E2 * 0.5 * r**2 * (r + 1 / r - (1 - c**2))

def fit(model):
    f = lambda t, N: N * model(t)
    N = np.sum(y * model(theta)) / np.sum(model(theta) ** 2)
    for i in range(10):
        slope = (f(theta + 1e-3, N) - f(theta - 1e-3, N)) / 2e-3  # per degree
        sig = np.sqrt(sy**2 + (slope * THETA_READ_ERR) ** 2)
        (N,), cov = curve_fit(f, theta, y, p0=[N], sigma=sig, absolute_sigma=True)
    resid = y - f(theta, N)
    chi2 = np.sum((resid / sig) ** 2)
    dof = len(y) - 1
    return N, np.sqrt(cov[0, 0]), sig, resid, chi2, dof, stats.chi2.sf(chi2, dof)

print(f"Omega = {OMEGA:.4f} +/- {OMEGA * OMEGA_REL_ERR:.4f} sr (common to all angles)")
results = {}
for name, model in (("Klein-Nishina", klein_nishina), ("Thomson", thomson)):
    results[name] = fit(model)
    N, dN, sig, resid, chi2, dof, pv = results[name]
    print(f"{name:14s} N = {N:.4g} +/- {dN:.2g} (fit) +/- {N * OMEGA_REL_ERR:.2g} (Omega) 1/(s barn)   chi^2_red = {chi2 / dof:.2f}   dof = {dof}   p-value = {pv:.3g}")

print("\nPoints:")
for r_, t_, e_, rt, ep, yy, s_ in zip(reading, theta, E_meas, rate, eps, y, sy):
    print(f"reading {r_:+5.0f}  theta {t_:6.1f}  E' {e_:6.1f} keV  rate {rt:7.3f}/s  eps {ep:.3f}  "
          f"y {yy:7.1f} +/- {s_:.1f} /(s sr)")

# polar plot
theta_signed = reading - THETA0
os.makedirs("plots/scattering", exist_ok=True)
# half circle
fig = plt.figure(figsize=(8, 9))
ax = fig.add_axes([0.04, 0.40, 0.92, 0.62], projection="polar")
ax.set_theta_zero_location("E")
tt = np.linspace(-180, 0, 361)
colors = {"Klein-Nishina": "C0", "Thomson": "C1"}
for name, model in (("Klein-Nishina", klein_nishina), ("Thomson", thomson)):
    N, dN, sig, resid, chi2, dof, pv = results[name]
    ax.plot(np.radians(tt), N * model(np.abs(tt)), color=colors[name], lw=1.5,
            label=f"{name} fit: $\\chi^2_{{\\mathrm{{red}}}}$ = {chi2 / dof:.2f},  p = {pv:.2g}")
ax.errorbar(np.radians(theta_signed), y, yerr=sy, fmt="o", ms=4, color="C2", capsize=2, label="Measured")
ax.set_thetamin(-180)
ax.set_thetamax(0)
ax.set_thetagrids(range(-180, 1, 30), [rf"{abs(a)}$^\circ$" for a in range(-180, 1, 30)])
ax.set_rlabel_position(-90)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_ylim(0, 1.1 * max(results["Klein-Nishina"][0] * R_E2, (y + sy).max()))
fig.suptitle(rf"Rate / ($\Omega\,\varepsilon$) in 1/(s sr) vs scattering angle, {TARGET}", y=0.985)
handles, labels = ax.get_legend_handles_labels()
lax = fig.add_axes([0.1, 0.435, 0.8, 0.06])
lax.axis("off")
lax.legend(handles, labels, fontsize=8, loc="center", ncol=1)
# residuals
for k, name in enumerate(("Klein-Nishina", "Thomson")):
    N, dN, sig, resid, chi2, dof, pv = results[name]
    axr = fig.add_axes([0.13, 0.26 - 0.20 * k, 0.84, 0.16])
    axr.errorbar(theta_signed, resid, yerr=sig, fmt="o", ms=4, color=colors[name], capsize=2)
    axr.axhline(0, color="k", ls="--", lw=1)
    axr.set_xlim(-180, 0)
    axr.set_xticks(range(-180, 1, 30))
    axr.set_ylabel(f"Residual, {name}\n(1/(s sr))", fontsize=9)
    axr.grid(alpha=0.3)
axr.set_xlabel(r"Scattering angle $\theta$ (deg)")
fig.savefig(f"plots/scattering/cross_section_fit{TAG}.png", dpi=200)
print(f"Saved plots/scattering/cross_section_fit{TAG}.png")

# Detector efficiency vs energy, fit in log-log

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

# settings
FILE = "efficiency_calibration.xlsx"
ORDER = 2

st = pd.read_excel(FILE, sheet_name="settings", header=None, index_col=0)[1]
DISTANCE_CM = float(st["distance_cm"])
DISTANCE_ERR_CM = float(st["distance_err_cm"])
RADIUS_CM = float(st["radius_cm"])
RADIUS_ERR_CM = float(st["radius_err_cm"])
ACTIVITY_REL_ERR = float(st["activity_rel_err"])

# read data
df = pd.read_excel(FILE, sheet_name="data")
for c in ["peak_area", "peak_area_err", "live_time_s"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["peak_area", "live_time_s"])
df = df[(df["energy_keV"] - 356.0129).abs() > 0.01].sort_values("energy_keV")
E = df["energy_keV"].to_numpy(float)
I_gamma = df["intensity"].to_numpy(float)
area = df["peak_area"].to_numpy(float)
live = df["live_time_s"].to_numpy(float)
if "peak_area_err" in df:
    area_err = df["peak_area_err"].fillna(np.sqrt(df["peak_area"])).to_numpy(float)
else:
    area_err = np.sqrt(area)
sources = df["source"].astype(str).to_numpy()

days = (pd.to_datetime(df["meas_date"], dayfirst=True) - pd.to_datetime(df["ref_date"], dayfirst=True)).dt.days
A_Bq = df["A0_kBq"].to_numpy(float) * 1e3 * 2 ** (-days.to_numpy(float) / df["half_life_days"].to_numpy(float))

# geometry
def solid_fraction(d, r):
    return r**2 / (4 * d**2)

g = solid_fraction(DISTANCE_CM, RADIUS_CM)
dg_dd = (solid_fraction(DISTANCE_CM + 1e-4, RADIUS_CM) - solid_fraction(DISTANCE_CM - 1e-4, RADIUS_CM)) / 2e-4
dg_dr = (solid_fraction(DISTANCE_CM, RADIUS_CM + 1e-4) - solid_fraction(DISTANCE_CM, RADIUS_CM - 1e-4)) / 2e-4
g_err = np.sqrt((dg_dd * DISTANCE_ERR_CM) ** 2 + (dg_dr * RADIUS_ERR_CM) ** 2)

# efficiency per line
N_expected = A_Bq * I_gamma * live * g
eps = area / N_expected
eps_rel_err = np.sqrt((area_err / area) ** 2 + ACTIVITY_REL_ERR**2 + (g_err / g) ** 2)
eps_err = eps * eps_rel_err

print(f"Omega/4pi = {g:.5f} +/- {g_err:.5f}")
print("\nEfficiency per line:")
for s, e, a, ne, ep, er in zip(sources, E, A_Bq, N_expected, eps, eps_err):
    print(f"{s:<8} {e:8.1f} keV   A = {a / 1e3:8.2f} kBq   N_exp = {ne:10.0f}   eps = {ep:.4f} +/- {er:.4f}")

# fit in log-log
x = np.log(E / 500.0)
y = np.log(eps)
y_err = np.sqrt((area_err / area) ** 2 + ACTIVITY_REL_ERR**2)

def poly(x, *c):
    return sum(ci * x**i for i, ci in enumerate(c))

n_par = ORDER + 1
p, cov = curve_fit(poly, x, y, p0=[np.log(eps.mean())] + [0] * ORDER, sigma=y_err, absolute_sigma=True)
p_err = np.sqrt(np.diag(cov))
resid = y - poly(x, *p)
chi2 = np.sum((resid / y_err) ** 2)
dof = len(E) - n_par
p_value = stats.chi2.sf(chi2, dof) if dof > 0 else float("nan")

print(f"\nFit: ln(eps) = sum c_i * (ln(E/500 keV))^i, order {ORDER}")
for i, (v, e) in enumerate(zip(p, p_err)):
    print(f"  c{i} = {v:.5g} +/- {e:.2g}")
print(f"chi^2_red = {chi2 / dof:.3f}")
print(f"dof = {dof}")
print(f"p-value = {p_value:.4g}")
print(f"plus a common scale error of {100 * g_err / g:.1f}% from the geometry")

def eps_at(E_keV):
    xx = np.log(E_keV / 500.0)
    J = np.array([xx**i for i in range(n_par)])
    val = np.exp(poly(xx, *p))
    rel = np.sqrt(J @ cov @ J + (g_err / g) ** 2)
    return val, val * rel

print("\nEfficiency at energies of the scattered photon:")
for e in [184, 225, 288, 400, 500, 600, 662]:
    v, er = eps_at(e)
    print(f"E = {e:4d} keV   eps = {v:.4f} +/- {er:.4f}")

# plot
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.3]})
Ex = np.geomspace(E.min() * 0.8, E.max() * 1.2, 300)
ax.errorbar(E, eps, yerr=eps_err, fmt="o", capsize=3, label="Calibration lines")
ax.plot(Ex, np.exp(poly(np.log(Ex / 500.0), *p)), "k-", lw=1, label=r"Fit: $\ln\varepsilon = c_0 + c_1 x + c_2 x^2$, $x = \ln(E/500\,\mathrm{keV})$")
# point labels
for i, (s, e, ep) in enumerate(zip(sources, E, eps)):
    right = i < len(E) - 1
    up = i % 2 == 0
    ax.annotate(f"{s} {e:.0f}", (e, ep), textcoords="offset points", xytext=(8 if right else -8, 6 if up else -12),
                ha="left" if right else "right", fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.text(0.98, 0.97, f"$\\chi^2_{{\\mathrm{{red}}}}$ = {chi2 / dof:.2f},  p = {p_value:.3g}",
        transform=ax.transAxes, va="top", ha="right", bbox=dict(fc="white", ec="0.8"))
ax.set_ylabel("Photopeak efficiency")
ax.set_title("Detector efficiency calibration")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3, which="both")
# plain tick labels
ax.set_xlim(E.min() * 0.7, E.max() * 1.45)
yt = [v for v in (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0) if 0.85 * eps.min() <= v <= 1.15 * eps.max()]
ax.set_yticks(yt)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
xt = [v for v in (50, 70, 100, 200, 300, 500, 700, 1000, 1500) if E.min() * 0.7 <= v <= E.max() * 1.45]
axr.set_xticks(xt)
axr.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
axr.xaxis.set_minor_formatter(ticker.NullFormatter())

axr.errorbar(E, resid, yerr=y_err, fmt="o", capsize=3)
axr.axhline(0, color="k", ls="--", lw=1)
axr.set_xlabel("Energy (keV)")
axr.set_ylabel(r"Residual in $\ln\varepsilon$")
axr.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("plots/calibration", exist_ok=True)
fig.savefig("plots/calibration/efficiency_calibration_fit.png", dpi=200)
print("\nSaved plots/calibration/efficiency_calibration_fit.png")
plt.show()

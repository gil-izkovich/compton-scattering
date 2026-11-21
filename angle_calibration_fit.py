# Zero angle: fit the rate vs  angle with a Gaussian + constant, theta0 is the center

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})  
# settings
FILE = "angle_calibration.xlsx"

# read data
df = pd.read_excel(FILE, sheet_name="data")
for c in ["angle_deg", "angle_err_deg", "counts", "live_time_s", "counts_err"]:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.sort_values("angle_deg")
theta = df["angle_deg"].to_numpy(float)
counts = df["counts"].to_numpy(float)
live = df["live_time_s"].to_numpy(float)
theta_err = df["angle_err_deg"].to_numpy(float)
if "counts_err" in df:
    counts_err = df["counts_err"].fillna(np.sqrt(df["counts"])).to_numpy(float)
else:
    counts_err = np.sqrt(counts)

rate = counts / live
rate_err = counts_err / live

def gaussian(t, A, t0, w, B):
    return A * np.exp(-(t - t0) ** 2 / (2 * w**2)) + B

p0 = [rate.max() - rate.min(), theta[np.argmax(rate)], 3, rate.min()]

# fit
sigma = rate_err.copy()
for i in range(5):
    p, cov = curve_fit(gaussian, theta, rate, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=20000)
    slope = (gaussian(theta + 1e-3, *p) - gaussian(theta - 1e-3, *p)) / 2e-3
    sigma = np.sqrt(rate_err**2 + (slope * theta_err) ** 2)
    p0 = p
p_err = np.sqrt(np.diag(cov))
A, theta0, w, B = p
A_err, theta0_err, w_err, B_err = p_err
resid = rate - gaussian(theta, *p)
chi2 = np.sum((resid / sigma) ** 2)
dof = len(theta) - len(p)
p_value = stats.chi2.sf(chi2, dof)

# print results
print("Zero angle calibration: R = A*exp(-(theta-theta0)^2/(2w^2)) + B")
for name, v, e in zip(["A", "theta0", "w", "B"], p, p_err):
    print(f"{name:6s} = {v:.5g} +/- {e:.2g}")
print(f"chi^2_red = {chi2 / dof:.3f}")
print(f"dof = {dof}")
print(f"p-value = {p_value:.4g}")
print(f"\ntheta0 = {theta0:.3f} +/- {theta0_err:.3f} deg")
print(f"True angle = board reading - ({theta0:.3f})")

# plot
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
x = np.linspace(theta.min() - 2, theta.max() + 2, 400)
ax.errorbar(theta, rate, xerr=theta_err, yerr=rate_err, fmt="o", capsize=3, label="Measured rate")
ax.plot(x, gaussian(x, *p), "k-", lw=1, label=r"$R(\theta) = A\,e^{-(\theta-\theta_0)^2/2w^2} + B$")
ax.axvline(theta0, color="r", ls="--", lw=1, label=rf"$\theta_0$ = {theta0:.2f} $\pm$ {theta0_err:.2f}$^\circ$")
ax.text(0.02, 0.04, f"$\\chi^2_{{\\mathrm{{red}}}}$ = {chi2 / dof:.2f},  p = {p_value:.3g}", transform=ax.transAxes, va="bottom", bbox=dict(fc="white", ec="0.8"))
ax.set_ylabel("Count rate (1/s)")
ax.set_title("Zero angle calibration")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

axr.errorbar(theta, resid, yerr=sigma, fmt="o", capsize=3)
axr.axhline(0, color="k", ls="--", lw=1)
axr.set_xlabel("Angle (deg)")
axr.set_ylabel("Residual (1/s)")
axr.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("plots/calibration", exist_ok=True)
fig.savefig("plots/calibration/angle_calibration_fit.png", dpi=200)
print("\nSaved plots/calibration/angle_calibration_fit.png")
plt.show()

# Energy calibration: E = a*ch + b, with chi^2 and p-value

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

# settings
FILE = "energy_calibration.xlsx"

# read data
df = pd.read_excel(FILE, sheet_name="data")
for c in ["energy_keV", "channel", "channel_err"]:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["channel", "channel_err"])
df = df[(df["energy_keV"] - 356.0129).abs() > 0.01].sort_values("channel")
ch = df["channel"].to_numpy(float)
ch_err = df["channel_err"].to_numpy(float)
E = df["energy_keV"].to_numpy(float)
sources = df["source"].fillna("").astype(str).to_numpy() if "source" in df else [""] * len(E)

# model
def model(x, a, b):
    return a * x + b

# fit
p = np.polyfit(ch, E, 1)
for i in range(5):
    sigma = np.abs(p[0] * ch_err)
    p, cov = curve_fit(model, ch, E, p0=p, sigma=sigma, absolute_sigma=True)

p_err = np.sqrt(np.diag(cov))
resid = E - model(ch, *p)
chi2 = np.sum((resid / sigma) ** 2)
dof = len(E) - len(p)
p_value = stats.chi2.sf(chi2, dof)

# print results
print("Energy calibration fit: E = a*ch + b")
for n, v, e in zip(["a", "b"], p, p_err):
    print(f"  {n} = {v:.6g} +/- {e:.2g}")
print(f"chi^2_red = {chi2 / dof:.3f}")
print(f"dof = {dof}")
print(f"p-value = {p_value:.4g}")
print(f"cov(a,b) = {cov[0, 1]:.4g}")
print("\nChannel -> energy:  E = a*ch + b")
print("sigma_E^2 = (a*sigma_ch)^2 + ch^2*var(a) + var(b) + 2*ch*cov(a,b)")
print("\nResiduals:")
for s, e, r, sg in zip(sources, E, resid, sigma):
    print(f"{s:<8} {e:8.2f} keV   residual = {r:7.3f} keV   pull = {r / sg:5.2f}")

# plot
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
x = np.linspace(ch.min() * 0.95, ch.max() * 1.03, 300)
ax.errorbar(ch, E, xerr=ch_err, fmt="o", capsize=3, label="Calibration peaks")
ax.plot(x, model(x, *p), "k-", lw=1, label="Fit")
for i, (s, c_, e) in enumerate(zip(sources, ch, E)):
    side = 1 if i % 2 == 0 else -1
    if c_ > ch.min() + 0.85 * (ch.max() - ch.min()):
        side = -1
    ax.annotate(f"{s} {e:.0f}", (c_, e), textcoords="offset points", xytext=(10 * side, -4 * side),
                ha="left" if side > 0 else "right", fontsize=8)
ax.text(0.02, 0.97, f"$\\chi^2_{{\\mathrm{{red}}}}$ = {chi2 / dof:.2f},  p = {p_value:.3g}",
        transform=ax.transAxes, va="top", bbox=dict(fc="white", ec="0.8"))
ax.set_ylabel("Energy (keV)")
ax.set_title("Energy calibration")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.margins(x=0.06)

axr.errorbar(ch, resid, yerr=sigma, fmt="o", capsize=3)
axr.axhline(0, color="k", ls="--", lw=1)
axr.set_xlabel("Channel")
axr.set_ylabel("Residual (keV)")
axr.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("plots/calibration", exist_ok=True)
fig.savefig("plots/calibration/energy_calibration_fit.png", dpi=200)
print("\nSaved plots/calibration/energy_calibration_fit.png")
plt.show()

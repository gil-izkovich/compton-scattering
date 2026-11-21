# Photopeak fit: Gaussian + linear background in one spectrum

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

FILE = "deg_cal_files/0deg.mca"
LOW = 5150  # first channel
HIGH = 6180  # last channel
BACKGROUND_FILE = ""
PLOT_ONLY = False

def plot_path(kind, suffix=""):
    folder = os.path.join("plots", kind, os.path.basename(os.path.dirname(os.path.abspath(FILE))))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, os.path.splitext(os.path.basename(FILE))[0] + suffix + ".png")

def plot_name():
    folder = os.path.basename(os.path.dirname(os.path.abspath(FILE)))
    stem = os.path.splitext(os.path.basename(FILE))[0]
    if folder == "source_cal_files":
        return re.sub(r"^([A-Za-z]+)(\d+)$", r"\1-\2", stem)
    if folder == "deg_cal_files":
        return re.sub(r"^(minus)?(\d+)deg$", lambda m: ("-" if m.group(1) else "") + m.group(2) + r"$^\circ$", stem)
    return FILE

# read an .mca file
def read_mca(path):
    counts, live, real, in_data = [], None, None, False
    for line in open(path, errors="ignore"):
        line = line.strip()
        if line.startswith("LIVE_TIME"):
            live = float(line.split("-")[-1])
        elif line.startswith("REAL_TIME"):
            real = float(line.split("-")[-1])
        elif line == "<<DATA>>":
            in_data = True
        elif line == "<<END>>":
            in_data = False
        elif in_data and line:
            counts.append(float(line))
    if not counts:
        counts = [float(x) for x in open(path).read().split()]
    return np.array(counts), live, real

counts, live, real = read_mca(FILE)
ch = np.arange(len(counts))
err = np.sqrt(np.maximum(counts, 1))
print(f"File: {FILE}   channels: {len(counts)}   Live time: {live} s   Real time: {real} s")
if live and real:
    print(f"Dead time: {100 * (real - live) / real:.1f} %")

if BACKGROUND_FILE:
    bg, live_bg, _ = read_mca(BACKGROUND_FILE)
    scale = live / live_bg
    err = np.sqrt(np.maximum(counts, 1) + scale**2 * np.maximum(bg, 1))
    counts = counts - scale * bg
    print(f"Background subtracted: {BACKGROUND_FILE}, scaled by {scale:.3f}")

if PLOT_ONLY:
    plt.semilogy(ch, np.maximum(counts, 0.5), lw=0.7)
    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.title(plot_name())
    plt.grid(alpha=0.3)
    out = plot_path("spectra")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()
    raise SystemExit

# fit range
sel = (ch >= LOW) & (ch <= HIGH)
x, y, s = ch[sel], counts[sel], err[sel]

def model(x, a, b, A, mu, sig):
    return a + b * x + A * np.exp(-(x - mu) ** 2 / (2 * sig**2))

b0 = (y[-5:].mean() - y[:5].mean()) / (x[-1] - x[0])
a0 = y[:5].mean() - b0 * x[0]
mu0 = x[np.argmax(y)]
p0 = [a0, b0, y.max() - (a0 + b0 * mu0), mu0, (HIGH - LOW) / 4]

p, cov = curve_fit(model, x, y, p0=p0, sigma=s, absolute_sigma=True, maxfev=20000)
perr = np.sqrt(np.diag(cov))
resid = y - model(x, *p)
chi2 = np.sum((resid / s) ** 2)
dof = len(x) - len(p)
p_value = stats.chi2.sf(chi2, dof)

# results
print(f"\nFit range: channels {LOW} to {HIGH}   chi^2_red = {chi2 / dof:.2f}   dof = {dof}   p = {p_value:.3g}")
A, mu, sig = p[2:5]
dA, dmu, dsig = perr[2:5]
sig = abs(sig)
area = A * sig * np.sqrt(2 * np.pi)
darea = area * np.sqrt(max((dA / A) ** 2 + (dsig / sig) ** 2 + 2 * cov[2, 4] / (A * sig), 0))
print("\nPeak:")
print(f"  center  mu    = {mu:.2f} +/- {dmu:.2f} channels")
print(f"  width   sigma = {sig:.2f} +/- {dsig:.2f} channels   (FWHM = {2.355 * sig:.1f})")
print(f"  area    N     = {area:.0f} +/- {darea:.0f} counts")
if live:
    print(f"  rate          = {area / live:.4g} +/- {darea / live:.2g} counts/s")
print("  Values for the Excel templates:")
print(f"      calibration_template: channel = {mu:.2f}, peak_sigma_ch = {sig:.2f}, peak_area = {area:.0f}, "
      f"channel_err = {dmu:.2f}")
print(f"      efficiency_template : peak_area = {area:.0f}, peak_area_err = {darea:.0f}, live_time_s = {live}")
print(f"      angle_scan_template : counts = {area:.0f}, counts_err = {darea:.0f}, live_time_s = {live}")

# plot
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2]})
xx = np.linspace(LOW, HIGH, 600)
ax.errorbar(x, y, yerr=s, fmt=".", ms=3, alpha=0.6,
            label="Data (background subtracted)" if BACKGROUND_FILE else "Data")
ax.plot(xx, model(xx, *p), "k-", lw=1.2, label="Fit: Gaussian + line")
# Gaussian alone
ax.plot(xx, p[2] * np.exp(-(xx - p[3]) ** 2 / (2 * p[4] ** 2)), "C3--", lw=1.2, label="Gaussian without background")
ax.set_ylabel("Counts")
ax.set_title(f"{plot_name()}: channels {LOW}-{HIGH}")
ax.text(0.02, 0.97, f"$\\chi^2_{{\\mathrm{{red}}}}$ = {chi2 / dof:.2f},  p = {p_value:.3g}", transform=ax.transAxes,
        va="top", bbox=dict(fc="white", ec="0.8"))
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
axr.errorbar(x, resid, yerr=s, fmt=".", ms=3, alpha=0.6)
axr.axhline(0, color="k", lw=1)
axr.set_xlabel("Channel")
axr.set_ylabel("Residual (counts)")
axr.grid(alpha=0.3)
fig.tight_layout()
out = plot_path("peak_fits", f"_ch{LOW}-{HIGH}")
fig.savefig(out, dpi=200)
print(f"\nSaved {out}")
plt.show()
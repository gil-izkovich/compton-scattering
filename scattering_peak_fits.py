# Fit the scattered peak in every spectrum of one part

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
plt.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

PART = 1  # 1: aluminum rod, 2: plastic scintillator
THETA0, THETA0_ERR = -2.614, 0.105  # deg
A_CAL, B_CAL = 0.1189196, -6.0246  # keV/ch, keV
START_A, START_B = 0.1186971, -6.1847
E0, MEC2 = 661.657, 510.999

if PART == 1:
    FOLDER, OUT_CSV = "scattering_al_files", "scattering_al_results.csv"
    FILES = {
        "-30deg.mca": -30,
        "deg45.mca": 45, "-45deg.mca": -45, "deg60.mca": 60, "-60deg.mca": -60,
        "deg75.mca": 75, "-75deg.mca": -75, "-90deg.mca": -90,
        "-105deg.mca": -105, "-120deg.mca": -120, "-135deg.mca": -135,
    }
else:
    # all 18 readings of day 3
    FOLDER, OUT_CSV = "scattering_plastic_files", "scattering_plastic_results.csv"
    FILES = {f"{r}deg.mca": r for r in (10, -10, 20, -20, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
                                        -105, -120, -135, -150)}
REBIN = 8

def read_mca(path):
    counts, live, real, in_data = [], None, None, False
    for line in open(path, errors="ignore"):
        line = line.strip()
        if line.startswith("LIVE_TIME"):
            live = float(line.split("-", 1)[1])
        elif line.startswith("Real Time:"):
            real = float(line.split(":", 1)[1])
        elif line == "<<DATA>>":
            in_data = True
        elif line == "<<END>>":
            in_data = False
        elif in_data and line:
            counts.append(float(line))
    return np.array(counts), live, real

def ch_of(e_kev):
    return (e_kev - START_B) / START_A

def compton(theta_deg):
    return E0 / (1 + E0 / MEC2 * (1 - np.cos(np.radians(theta_deg))))

def gauss(x, A, mu, sig):
    return A * np.exp(-(x - mu) ** 2 / (2 * sig**2))

def one_peak(x, a, b, A, mu, sig):
    return a + b * x + gauss(x, A, mu, sig)

def fit_window(ch, y, s, lo, hi, mu0, sig0):
    sel = (ch >= lo) & (ch <= hi)
    x, yy, ss = ch[sel], y[sel], s[sel]
    b0 = (yy[-5:].mean() - yy[:5].mean()) / (x[-1] - x[0])
    a0 = yy[:5].mean() - b0 * x[0]
    peak0 = max(np.interp(mu0, x, yy) - (a0 + b0 * mu0), 1.0)
    lob = [-np.inf, -np.inf, 0, x[0], 20]
    hib = [np.inf, np.inf, np.inf, x[-1], 600]
    p0 = np.clip([a0, b0, peak0, mu0, sig0], np.array(lob) + 1e-9, np.array(hib) - 1e-9)
    p, cov = curve_fit(one_peak, x, yy, p0=p0, sigma=ss, absolute_sigma=True, bounds=(lob, hib), maxfev=50000)
    return one_peak, p, cov, x, yy, ss

rows = []
os.makedirs(os.path.join("plots", "peak_fits", FOLDER), exist_ok=True)
for name, reading in FILES.items():
    raw, live, real = read_mca(os.path.join(FOLDER, name))
    n = len(raw) // REBIN * REBIN
    counts = raw[:n].reshape(-1, REBIN).sum(1)
    ch = np.arange(0, n, REBIN) + (REBIN - 1) / 2
    err = np.sqrt(np.maximum(counts, 1))
    true = reading - THETA0

    mu, sig = ch_of(compton(true)), 198.51 * np.sqrt(compton(true) / E0) * 1.3
    for it in range(15):
        lo, hi = int(mu - 2 * sig), int(mu + 2 * sig)
        model, p, cov, x, y, s = fit_window(ch, counts, err, lo, hi, mu, sig)
        new_mu, new_sig = p[3], abs(p[4])
        if abs(new_mu - mu) < 1 and abs(new_sig - sig) < 1:
            mu, sig = new_mu, new_sig
            break
        mu, sig = new_mu, new_sig

    perr = np.sqrt(np.diag(cov))
    resid = y - model(x, *p)
    chi2 = np.sum((resid / s) ** 2)
    dof = len(x) - len(p)
    pval = stats.chi2.sf(chi2, dof)
    A, mu, sig = p[2], p[3], abs(p[4])
    dA, dmu, dsig = perr[2], perr[3], perr[4]
    area = A * sig * np.sqrt(2 * np.pi) / REBIN
    darea = area * np.sqrt(max((dA / A) ** 2 + (dsig / sig) ** 2 + 2 * cov[2, 4] / (A * sig), 0))
    e_meas = A_CAL * mu + B_CAL
    de_meas = A_CAL * dmu

    rows.append(dict(file=name, reading_deg=reading, true_angle_deg=round(true, 3), lo=lo, hi=hi,
                     center_ch=mu, center_err=dmu, sigma_ch=sig, sigma_err=dsig,
                     area=area, area_err=darea, live_s=live, real_s=real,
                     E_meas_keV=e_meas, E_meas_err=de_meas, E_compton_keV=compton(true),
                     chi2=chi2, dof=dof, p=pval))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    xx = np.linspace(x[0], x[-1], 600)
    ax.errorbar(x, y, yerr=s, fmt=".", ms=3, alpha=0.6, label="Data")
    ax.plot(xx, model(xx, *p), "k-", lw=1.2, label=f"Fit: Gaussian + line, $E'$ = {e_meas:.1f} keV")
    ax.set_ylabel(f"Counts per {REBIN} channels")
    target = "aluminum rod" if PART == 1 else "plastic, coincidence"
    ax.set_title(rf"{target}, {reading:+d}$^\circ$ on the board (true {true:.1f}$^\circ$): channels {lo}-{hi}")
    ax.text(0.02, 0.97, f"$\\chi^2_{{\\mathrm{{red}}}}$ = {chi2 / dof:.2f},  p = {pval:.3g}", transform=ax.transAxes,
            va="top", bbox=dict(fc="white", ec="0.8"))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    axr.errorbar(x, resid, yerr=s, fmt=".", ms=3, alpha=0.6)
    axr.axhline(0, color="k", lw=1)
    axr.set_xlabel("Channel")
    axr.set_ylabel(f"Residual (counts per {REBIN} channels)")
    axr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join("plots", "peak_fits", FOLDER, os.path.splitext(name)[0] + ".png"), dpi=150)
    plt.close(fig)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'file':20s} {'read':>5s} {'true':>7s} {'window':>11s} {'mu':>8s} {'dmu':>5s} {'sig':>6s} {'area':>7s} {'darea':>6s} "
      f"{'E_meas':>7s} {'E_th':>6s} {'chi2_red':>9s} {'p':>6s}")
for r in rows:
    print(f"{r['file']:20s} {r['reading_deg']:5d} {r['true_angle_deg']:7.1f} {r['lo']:5d}-{r['hi']:<5d} {r['center_ch']:8.1f} "
          f"{r['center_err']:5.1f} {r['sigma_ch']:6.1f} {r['area']:7.0f} {r['area_err']:6.0f} {r['E_meas_keV']:7.1f} "
          f"{r['E_compton_keV']:6.1f} {r['chi2'] / r['dof']:9.2f} {r['p']:6.3f}")
print(f"\nSaved {OUT_CSV} and plots/peak_fits/{FOLDER}/")
